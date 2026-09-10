//! Atomic slot migration: the task a slot range moves under, and the snapshot
//! of that slot range going out.
//!
//! The old way of moving a slot is a tool holding both nodes by the hand:
//! `SETSLOT MIGRATING` here, `SETSLOT IMPORTING` there, then `MIGRATE` for every
//! key in the slot, one round trip each, and `SETSLOT NODE` at the end. It works
//! and it is slow, and while it is running the slot is half on each node, so
//! every client that touches it gets an `ASK` and has to ask twice.
//!
//! The new way is the two nodes doing it themselves. The node taking the slots
//! opens two connections to the node giving them up, logs in as the cluster
//! rather than as a user, and says what it wants. The node giving them up sends
//! a snapshot of the slot range down one connection and every change to it down
//! the other, and when the second has nearly caught up with the first it stops
//! taking writes for a moment, lets the far side finish, and hands the slots
//! over in one step. No client ever sees the slot on two nodes.
//!
//! # What is here
//!
//! The task, which is the record of one of those moves and the thing
//! `CLUSTER MIGRATION STATUS` reports, and both streams. A task is created by
//! `CLUSTER SYNCSLOTS SYNC` arriving from the far side, moves to
//! `wait-rdbchannel`, gets its snapshot written when the second connection
//! arrives with `CLUSTER SYNCSLOTS RDBCHANNEL`, and then every write that lands
//! in the moving slots goes down the first connection behind it until the far
//! side says it has caught up. Then writes stop, what was already running is
//! waited out, and the far side is told the stream has ended. It claims the
//! slots over the bus, this node hears the claim, the task is finished, the
//! writes go again and the keys that have moved are dropped.
//!
//! What is not here is the other side of it: `CLUSTER MIGRATION IMPORT`, which
//! is a node asking to take slots rather than being asked to give them up. Every
//! move against this node is therefore one the far side drives, which is what a
//! real cluster does anyway, since the node taking the slots is the one that
//! starts a move. That is what is left of D-149.
//!
//! # Where the two streams meet
//!
//! Exactly at the freeze the snapshot is read under. The stream is switched on
//! inside it, so there is no instant at which a write is in neither: anything
//! that got in before the freeze is in the snapshot, and anything after it is in
//! the stream. Getting that wrong in either direction is a key the far side
//! never hears about or an `INCR` it runs twice.
//!
//! # The snapshot is not an RDB file
//!
//! It is a stream of ordinary commands: a `FUNCTION RESTORE`, a `SELECT` per
//! database that has anything in it, a `CLUSTER SYNCSLOTS CONF SLOT-INFO` before
//! the first key of each slot, a `RESTORE` per key, and a
//! `CLUSTER SYNCSLOTS SNAPSHOT-EOF` at the end. The far side does not parse a
//! file, it runs what it is sent, which is why a slot range can be sent this way
//! at all: an RDB file is the whole keyspace and there is no way to ask for a
//! sixteen thousandth of one.
//!
//! It is built in memory in one go rather than streamed out as it is read, which
//! is the same trade the full resync in `repl` makes and is there for the same
//! reason: without a fork, the only way to read one moment of the dataset is to
//! stop writes while reading it, and the shorter that is the better.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64};

use yo_common::lock::Lock;
use yo_common::{Code, Error, Result};
use yo_kv::value::{Kind, Str};
use yo_kv::{Ask, rdb};

use crate::proto::{Limits, Proto};
use crate::reply::Out;
use crate::request::{Argv, Step};

use super::super::args::Args;
use super::super::clients::Client;
use super::super::pubsub::Envelope;
use super::super::{Server, keyspec, table};
use super::{ID_LEN, key_slot};

/// How many finished tasks are kept to be asked about afterwards.
///
/// The default of `cluster-slot-migration-max-archived-tasks`. A finished task
/// is a few hundred bytes and the only thing that reads one is an operator
/// asking what happened, so the number only has to be larger than the number of
/// moves anybody looks back over.
const MAX_ARCHIVED: i64 = 32;

/// How far behind the far side may be and still be called caught up.
///
/// The default of `cluster-slot-migration-handoff-max-lag-bytes`. Waiting for
/// nought would mean waiting for a moment that a busy server never has, so the
/// rule is instead that the far side is close enough that it can finish inside
/// the pause rather than before it. A megabyte of commands is a few
/// milliseconds of applying them.
const MAX_LAG: i64 = 1024 * 1024;

/// How long writes may stay paused waiting for the far side to take the slots.
///
/// The default of `cluster-slot-migration-write-pause-timeout`, in
/// milliseconds. Once the pause is on, every client writing to this node is
/// waiting on one node on the other end of one connection, so the pause needs a
/// bound that is short enough to be survivable and long enough that a far side
/// which is merely busy is not given up on. Ten seconds is the reference's
/// answer to that.
const WRITE_PAUSE: i64 = 10 * 1000;

/// How long the far side gets to drain what it has buffered.
///
/// The default of `cluster-slot-migration-sync-buffer-drain-timeout`, in
/// milliseconds, and the reference doubles it when the snapshot itself took
/// longer than that to apply. Hidden, because it is a backstop rather than
/// something to tune.
const DRAIN: i64 = 60 * 1000;

/// The `cluster-slot-migration-*` settings.
///
/// Atomics and not a lock, because two of them are read on the path every
/// propagated write takes and the other two are read by the cron. Nothing here
/// is read together with anything else here, so there is no pair to keep
/// consistent and no reason for them to share one word.
///
/// All four are held even when this node is not a cluster node at all, which is
/// the reference's rule as well: `CONFIG GET` answers them on any server, and a
/// tool reading a setting before deciding what to do wants the number rather
/// than nothing back.
struct Knobs {
    /// `cluster-slot-migration-handoff-max-lag-bytes`.
    lag: AtomicI64,
    /// `cluster-slot-migration-write-pause-timeout`, in milliseconds.
    pause: AtomicI64,
    /// `cluster-slot-migration-sync-buffer-drain-timeout`, in milliseconds.
    drain: AtomicI64,
    /// `cluster-slot-migration-max-archived-tasks`.
    archived: AtomicI64,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            lag: AtomicI64::new(MAX_LAG),
            pause: AtomicI64::new(WRITE_PAUSE),
            drain: AtomicI64::new(DRAIN),
            archived: AtomicI64::new(MAX_ARCHIVED),
        }
    }
}

impl Knobs {
    /// The one word a name stands for.
    fn of(&self, which: Migration) -> &AtomicI64 {
        match which {
            Migration::Lag => &self.lag,
            Migration::Pause => &self.pause,
            Migration::Drain => &self.drain,
            Migration::Archived => &self.archived,
        }
    }
}

/// Which of the four migration settings is being read or written.
///
/// A name rather than a string, so that the config table and the code that
/// wants the number cannot drift apart: adding a row here without giving it a
/// word to answer to does not compile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Migration {
    /// How far behind the far side may be and still be called caught up, in
    /// bytes. Read on every acknowledgement.
    Lag,
    /// How long writes may stay paused, in milliseconds.
    Pause,
    /// How long the far side gets to drain its buffer, in milliseconds.
    Drain,
    /// How many finished tasks are kept.
    Archived,
}

/// Where a task has got to.
///
/// The reference's `asmState`, and the words are its `asmTaskStateToString`,
/// because they go out on the wire in `CLUSTER MIGRATION STATUS` and a tool
/// reads them. Only the states this node can actually be in are here; the rest
/// belong to the side that starts a migration and to the streams that are not
/// written yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum State {
    /// Made and not started, which is what an import task sits at until the
    /// node it is importing from has been dialled.
    None,
    /// Cancelled, by an operator or by something else needing the slot.
    Canceled,
    /// Done with, which the snapshot connection is once the snapshot has gone
    /// out and a whole task is once the slots have changed hands.
    Completed,
    /// Given up on. The reason is in the task's error.
    Failed,
    /// The far side has asked for the slots and the second connection it will
    /// take the snapshot on has not arrived yet.
    WaitRdbChannel,
    /// The second connection has arrived and the snapshot has not been built.
    WaitBgsaveStart,
    /// The snapshot has gone out and the changes since are going out behind it.
    SendStream,
    /// The far side is within a hair of caught up, so the next thing to happen
    /// is writes stopping and the slots changing hands.
    HandoffPrep,
    /// Writes have stopped and what was already running is being waited out.
    Handoff,
    /// Nothing more is coming and the far side has been told so. All that is
    /// left is for it to claim the slots.
    StreamEof,
    /// What the far side calls itself while it is working through the changes
    /// that piled up behind the snapshot. Only ever set as the other end's
    /// state, never as this node's own.
    StreamingBuf,
    /// And what it calls itself once that pile is empty and it is waiting to be
    /// told there is no more coming.
    WaitStreamEof,
}

impl State {
    /// The word this state goes out as.
    fn word(self) -> &'static str {
        match self {
            State::None => "none",
            State::Canceled => "canceled",
            State::Completed => "completed",
            State::Failed => "failed",
            State::WaitRdbChannel => "wait-rdbchannel",
            State::WaitBgsaveStart => "wait-bgsave-start",
            State::SendStream => "send-stream",
            State::HandoffPrep => "handoff-prep",
            State::Handoff => "handoff",
            State::StreamEof => "stream-eof",
            State::StreamingBuf => "streaming-buffer",
            State::WaitStreamEof => "wait-stream-eof",
        }
    }

    /// The state the far side named itself as, if it is one it is allowed to.
    ///
    /// Only the two words a destination sends in an `ACK` are taken. Anything
    /// else is not a state it could be in when it acknowledges, and the
    /// reference answers nothing at all rather than complaining, because there
    /// is nobody on that connection reading for a reply.
    fn dest_word(word: &[u8]) -> Option<State> {
        match word {
            _ if word.eq_ignore_ascii_case(b"streaming-buffer") => Some(State::StreamingBuf),
            _ if word.eq_ignore_ascii_case(b"wait-stream-eof") => Some(State::WaitStreamEof),
            _ => None,
        }
    }

    /// Whether a migration in this state is still sending changes.
    ///
    /// `handoff` is one of them, and that is the whole point of the state: the
    /// writes that were already running when the pause went on still have to be
    /// sent, or the far side takes the slots missing them.
    fn streaming(self) -> bool {
        matches!(
            self,
            State::SendStream | State::HandoffPrep | State::Handoff
        )
    }

    /// Whether writes are stopped while this state lasts.
    fn pausing(self) -> bool {
        matches!(self, State::Handoff | State::StreamEof)
    }
}

/// One slot range on its way from one node to another.
pub(super) struct Task {
    /// The forty characters both nodes call it. Chosen by whoever started it,
    /// which is the node taking the slots.
    id: String,
    /// What is moving, sorted and with the ranges that touch joined up.
    slots: Vec<(u16, u16)>,
    /// The node giving the slots up, as forty characters.
    source: Vec<u8>,
    /// The node taking them. Forty zero bytes until the far side has said who it
    /// is, which is what the reference reports too: its field is a fixed forty
    /// bytes that start out zeroed and it writes all forty of them back out.
    dest: Vec<u8>,
    /// Whether this node is the one taking the slots.
    import: bool,
    /// Where it has got to.
    state: State,
    /// Why it stopped, empty while it has not.
    error: String,
    /// How many times it has been started again after failing.
    retries: i64,
    /// When it was made, as a millisecond since the epoch.
    created: i64,
    /// When it started, or minus one for a task that has not.
    started: i64,
    /// When it ended, or minus one for a task that has not.
    ended: i64,
    /// The connection the far side asked over, which is the one the changes go
    /// down. `None` for a task nobody is holding open.
    ///
    /// The whole row rather than the id, because the changes are handed to the
    /// thread that owns the connection and it takes both the slot number and the
    /// id to be sure of writing to the connection that is still there rather
    /// than to whatever took its place.
    main: Option<Arc<Client>>,
    /// The connection the snapshot goes down.
    rdb: Option<u64>,
    /// Bytes of changes handed to the main channel since the snapshot.
    ///
    /// A count of this stream and not a replication offset. The far side counts
    /// the same bytes as it works through them and says how far it has got, and
    /// the two meeting is what says it is safe to stop taking writes.
    sent: u64,
    /// The last count the far side acknowledged. Never allowed to go backwards.
    acked: u64,
    /// What the far side said it was doing when it last acknowledged.
    dest_state: State,
    /// Where the snapshot connection has got to, which only a failure message
    /// ever reads. `none` until the snapshot has been asked for and `completed`
    /// once it has gone out.
    ///
    /// The reference has a state between the two for a snapshot being written,
    /// and here there is no such moment: the snapshot is built in one go behind
    /// a freeze, so it has either not started or finished.
    rdb_state: State,
    /// When writes stopped for the handoff, or nought if they have not.
    paused: i64,
}

impl Task {
    /// The slot ranges as the reference writes them, which is `1-5 9-9` with a
    /// space between and no trailing one.
    fn slot_words(&self) -> String {
        let mut text = String::new();
        for (at, (from, to)) in self.slots.iter().enumerate() {
            if at > 0 {
                text.push(' ');
            }
            yo_alloc::allow(|| {
                use core::fmt::Write as _;
                let _ = write!(text, "{from}-{to}");
            });
        }
        text
    }

    /// Write this task as the twelve field map `CLUSTER MIGRATION STATUS`
    /// answers with, which is the reference's `replyTaskStatus`.
    fn report(&self, out: &mut Out) {
        out.map(12);
        out.bulk(b"id");
        out.bulk(self.id.as_bytes());
        out.bulk(b"slots");
        out.bulk(self.slot_words().as_bytes());
        out.bulk(b"source");
        out.bulk(&self.source);
        out.bulk(b"dest");
        out.bulk(&self.dest);
        out.bulk(b"operation");
        out.bulk(if self.import { b"import" } else { b"migrate" });
        out.bulk(b"state");
        out.bulk(self.state.word().as_bytes());
        out.bulk(b"last_error");
        out.bulk(self.error.as_bytes());
        out.bulk(b"retries");
        out.int(self.retries);
        out.bulk(b"create_time");
        out.int(self.created);
        out.bulk(b"start_time");
        out.int(self.started);
        out.bulk(b"end_time");
        out.int(self.ended);
        // How long writes were stopped for, which the reference only reports for
        // a migration that got all the way through. One that is still running or
        // that gave up says nought however long it held the server, because the
        // number is there to be read after the fact and a running one has no
        // total yet.
        out.bulk(b"write_pause_ms");
        out.int(if self.import || self.state != State::Completed {
            0
        } else {
            self.ended - self.paused
        });
    }

    /// Say why this task stopped, in the reference's sentence.
    ///
    /// The two states it names are the task's own and the snapshot connection's,
    /// and the first of them is the state the task was in when it went wrong
    /// rather than the one it is about to move to. That is what the reference
    /// reports, because it builds the sentence before it changes the state, and
    /// it is the more useful of the two: knowing a cancelled task is cancelled
    /// says nothing, knowing it was halfway through a snapshot says everything.
    /// So every caller says why first and moves the state second.
    fn blame(&mut self, why: &str) {
        self.error = yo_alloc::allow(|| {
            format!(
                "{why} (state: {}, rdb_channel_state: {})",
                self.state.word(),
                self.rdb_state.word()
            )
        });
    }
}

/// Every migration this node is in or has been in.
///
/// One at a time, which is the reference's rule and not a shortcut: two moves at
/// once would each be pausing writes for the other to catch up. The list is
/// still a list because a finished one is kept to be asked about.
#[derive(Default)]
pub(super) struct Asm {
    inner: Lock<Tasks>,
    /// The `cluster-slot-migration-*` settings, which `CONFIG SET` writes and
    /// everything in here reads.
    knobs: Knobs,
    /// Whether there is a migration sending changes right now.
    ///
    /// Outside the lock because it is read once for every write the server
    /// takes, and on a server that is not moving a slot, which is nearly all of
    /// them nearly all of the time, the answer is no and the lock would be a
    /// contended one on the hot path for nothing.
    streaming: AtomicBool,
    /// The deadline of the write pause a handoff armed, or nought for none.
    ///
    /// Kept because the server has one pause and several things can arm it, so
    /// lifting this one has to be able to say which one it is lifting and leave
    /// anybody else's alone.
    armed: AtomicU64,
}

/// The live task and the ones that have finished, newest first.
#[derive(Default)]
struct Tasks {
    live: Option<Task>,
    done: Vec<Task>,
}

impl Tasks {
    /// Move the live task onto the finished list, which is the reference's
    /// `asmTaskFinalize`.
    fn finish(&mut self, now: i64, keep: usize) {
        let Some(mut task) = self.live.take() else {
            return;
        };
        task.ended = now;
        self.done.insert(0, task);
        self.done.truncate(keep);
    }
}

impl Asm {
    /// `CLUSTER MIGRATION STATUS ALL`, which is every task there is.
    pub(super) fn report_all(&self, out: &mut Out) {
        let tasks = self.inner.lock();
        out.array(usize::from(tasks.live.is_some()) + tasks.done.len());
        for task in tasks.live.iter().chain(tasks.done.iter()) {
            task.report(out);
        }
    }

    /// `CLUSTER MIGRATION STATUS ID <id>`, which is an array of one or of none.
    pub(super) fn report_one(&self, id: &[u8], out: &mut Out) {
        let tasks = self.inner.lock();
        let found = tasks
            .live
            .iter()
            .chain(tasks.done.iter())
            .find(|task| task.id.as_bytes() == id);
        match found {
            Some(task) => {
                out.array(1);
                task.report(out);
            }
            None => out.array(0),
        }
    }

    /// Cancel the live task if it is the one named, and say whether it was.
    ///
    /// `None` means every task, which is what `CANCEL ALL` sends. A finished
    /// task cannot be cancelled and is not counted, so cancelling twice answers
    /// one and then nought.
    pub(super) fn cancel(&self, id: Option<&[u8]>, now: i64) -> i64 {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return 0;
        };
        if id.is_some_and(|want| task.id.as_bytes() != want) {
            return 0;
        }
        task.blame("Cancelled due to user request");
        task.state = State::Canceled;
        self.retire(&mut tasks, now);
        1
    }

    /// Move the live task onto the finished list and shut the stream gate.
    ///
    /// Every way a task ends goes through here, which is what makes the gate
    /// and the task agree: a task that is no longer live cannot be one that is
    /// still being fed.
    fn retire(&self, tasks: &mut Tasks, now: i64) {
        tasks.finish(now, self.knobs.archived.load(Relaxed).max(1) as usize);
        self.streaming.store(false, Relaxed);
    }

    /// Start sending changes, which the snapshot does from inside its freeze.
    fn start_stream(&self) {
        let mut tasks = self.inner.lock();
        if let Some(task) = tasks.live.as_mut()
            && task.state == State::WaitBgsaveStart
        {
            task.state = State::SendStream;
            // The snapshot has been read by the time this runs and there is no
            // moment at which it is half sent, so the connection it goes down is
            // done with as far as anything that reads this is concerned.
            task.rdb_state = State::Completed;
            self.streaming.store(true, Relaxed);
        }
    }

    /// Give up on the live task because a command touched two slots at once.
    ///
    /// The stream is one slot range and a command across two of them cannot be
    /// split, so there is nothing to send that would leave the far side with the
    /// right answer. In cluster mode the routing gate refuses one of these
    /// before it runs, so what is left is a script or a module reaching past
    /// what it declared, and the migration is the thing that gives way.
    fn cross_slot(&self, now: i64) {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return;
        };
        task.blame("Cancelled due to propagating cross slot command");
        task.state = State::Canceled;
        self.retire(&mut tasks, now);
    }

    /// Give up on the live task if one of its connections has gone.
    ///
    /// The reference's `asmCallbackOnFreeClient`. Either connection going is the
    /// end of the migration: the far side cannot be told, and half a slot range
    /// on the far side is exactly what nobody must be left with.
    pub(super) fn forget(&self, conn: u64, now: i64) {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return;
        };
        let main = task.main.as_ref().is_some_and(|row| row.id == conn);
        if !main && task.rdb != Some(conn) {
            return;
        }
        let which = if main { "Main" } else { "RDB" };
        task.blame(&yo_alloc::allow(|| {
            format!("{which} channel - Connection with the peer node was lost")
        }));
        task.state = State::Failed;
        self.retire(&mut tasks, now);
    }

    /// Take the far side's `CLUSTER SYNCSLOTS ACK <state> <offset>`.
    ///
    /// Nothing is written back. The acknowledgement is a number travelling one
    /// way on a connection whose other direction is the change stream, and a
    /// reply on it would be read as a command.
    ///
    /// Once the far side is within `cluster-slot-migration-handoff-max-lag-bytes`
    /// of everything that has been sent, the task moves to `handoff-prep` and
    /// the answer is true, which is the caller's cue to stop writes and hand the
    /// slots over. That is done outside this lock because it freezes the server
    /// and holding a lock the status command wants across a freeze would mean
    /// nobody could even ask what the migration was doing.
    fn ack(&self, conn: u64, state: State, offset: u64) -> bool {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return false;
        };
        if task.import || !task.main.as_ref().is_some_and(|row| row.id == conn) {
            return false;
        }
        task.dest_state = state;
        // Backwards is not an error and not a state to act on. The reference
        // logs it and carries on, because the far side reconnecting and starting
        // its count again is a thing that happens and the older number is simply
        // stale.
        if offset < task.acked {
            return false;
        }
        task.acked = offset;
        let lag = self.knobs.lag.load(Relaxed).max(0) as u64;
        if task.state == State::SendStream && task.acked + lag >= task.sent {
            task.state = State::HandoffPrep;
            return true;
        }
        false
    }

    /// Note that writes have stopped, and say whether they stopped for this.
    ///
    /// False means the task moved on between the acknowledgement and here, which
    /// a cancel arriving at the wrong moment does, and then the pause is not
    /// armed at all rather than armed with nothing left to lift it.
    fn begin_handoff(&self, now: i64, until: u64) -> bool {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return false;
        };
        if task.state != State::HandoffPrep {
            return false;
        }
        task.state = State::Handoff;
        task.paused = now;
        self.armed.store(until, Relaxed);
        true
    }

    /// Tell the far side there is no more coming, and stop sending.
    ///
    /// Run inside the freeze, which is what makes the end of the stream an
    /// instant rather than a guess. The reference watches the socket empty out
    /// instead, because it has one thread and a write that got as far as the
    /// buffer is a write that is already accounted for. Here a write on another
    /// thread can still be running when the pause goes on, so what is waited for
    /// is the write itself and not the bytes it will produce, and the freeze is
    /// the same barrier the snapshot is taken behind.
    ///
    /// The connection is let go of rather than closed. The far side closes it
    /// once it has read the last of the stream, and letting go here is what
    /// keeps that from being read as the connection dropping under a live task.
    fn end_stream(&self) -> Option<Arc<Client>> {
        let mut tasks = self.inner.lock();
        let task = tasks.live.as_mut()?;
        if task.state != State::Handoff {
            return None;
        }
        task.state = State::StreamEof;
        self.streaming.store(false, Relaxed);
        task.rdb = None;
        task.main.take()
    }

    /// Give up on a handoff the far side never finished, which is the reference's
    /// `cluster-slot-migration-write-pause-timeout`.
    ///
    /// The slots stay here and the keys stay here. Everything the far side built
    /// is its to throw away, and it finds out either from the connection ending
    /// or from the slots never moving.
    fn pause_expired(&self, now: i64, timeout: i64) -> bool {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return false;
        };
        if !task.state.pausing() || now - task.paused < timeout {
            return false;
        }
        task.blame(&yo_alloc::allow(|| {
            format!(
                "Write pause timeout during slot handoff: destination did not take ownership within {timeout} ms."
            )
        }));
        task.state = State::Failed;
        self.retire(&mut tasks, now);
        true
    }

    /// Whether a live task is holding the write pause on.
    fn pausing(&self) -> bool {
        let tasks = self.inner.lock();
        tasks.live.as_ref().is_some_and(|task| task.state.pausing())
    }

    /// The slots have changed hands. Say what that did to the task moving them.
    ///
    /// The task has to be moving exactly these slots and no others, which is the
    /// reference's rule and is stricter than it looks: a claim that covers half
    /// of what a migration is moving is not that migration finishing, it is
    /// something else happening to the cluster while a migration was running,
    /// and the migration is given up on rather than reported as done.
    fn config_updated(&self, moved: &[(u16, u16)], now: i64) -> Moved {
        let mut tasks = self.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return Moved::Elsewhere;
        };
        if task.slots == moved {
            if !task.import && task.state == State::StreamEof {
                // The one path a migration finishes down. The error is cleared
                // rather than left, because a task that was retried carries the
                // reason the earlier try stopped and the one that got through
                // did not stop for anything.
                task.error.clear();
                task.state = State::Completed;
                let slots = yo_alloc::allow(|| task.slots.clone());
                self.retire(&mut tasks, now);
                return Moved::Done(slots);
            }
            task.blame("Cancelled due to slots configuration updated");
            task.state = State::Canceled;
            self.retire(&mut tasks, now);
            return Moved::Cancelled;
        }
        if overlapping(&task.slots, moved) {
            task.blame("Cancelled due to slots configuration updated");
            task.state = State::Canceled;
            self.retire(&mut tasks, now);
        }
        Moved::Elsewhere
    }
}

/// What a slot changing hands did to the migration that was moving it.
///
/// It decides two things the caller has to get right: whether the keys behind
/// those slots are dropped, and how the drop is told to anybody following this
/// node.
enum Moved {
    /// This was a migration finishing, and these are the slots it moved.
    Done(Vec<(u16, u16)>),
    /// A migration was moving these and was not ready for them to go, so it was
    /// given up on. Nothing is dropped: the slots were the task's to move and a
    /// task that was cancelled is not a reason to lose keys.
    Cancelled,
    /// No migration was moving them, so they went some other way, which is a
    /// failover or an operator with `CLUSTER SETSLOT`. Whatever this node still
    /// holds for them belongs to somebody else now.
    Elsewhere,
}

/// How the keys a trim drops are told to anybody following this node.
///
/// The three are not a choice, they are three different situations. What varies
/// is whether the deletions are already accounted for by something else on the
/// stream, and whether anybody asked for these keys to go.
#[derive(Clone, Copy)]
enum Trim {
    /// One `TRIMSLOTS` naming the ranges, and a `del` event per key. What a
    /// migration finishing does: an operator asked for this and is watching, and
    /// a replica can work the key list out from the ranges.
    Ranges,
    /// A deletion per key and no events. What a slot that went some other way
    /// does, which is a failover or an operator moving it by hand. Nobody asked
    /// for these keys to go, so a client listening to the keyspace is not told
    /// they did, and there is no command on the stream to carry them.
    Keys,
    /// Neither, because the command that asked for the trim is itself on the
    /// stream. What `TRIMSLOTS` arriving from a master does. The events still
    /// fire, since the far side of a migration is where the operator is looking.
    Already,
}

/// Whether two sorted range lists have a slot in common.
fn overlapping(a: &[(u16, u16)], b: &[(u16, u16)]) -> bool {
    a.iter()
        .any(|(from, to)| b.iter().any(|(start, end)| from <= end && start <= to))
}

/// A list of slots as the ranges the reference would write, sorted and with the
/// ones that touch joined up.
fn joined(slots: &[u16]) -> Vec<(u16, u16)> {
    let mut sorted = yo_alloc::allow(|| slots.to_vec());
    sorted.sort_unstable();
    sorted.dedup();
    let mut ranges: Vec<(u16, u16)> = Vec::new();
    yo_alloc::allow(|| {
        for slot in sorted {
            match ranges.last_mut() {
                Some(last) if u32::from(last.1) + 1 == u32::from(slot) => last.1 = slot,
                _ => ranges.push((slot, slot)),
            }
        }
    });
    ranges
}

/// Whether a slot is in one of the ranges.
fn within(ranges: &[(u16, u16)], slot: u16) -> bool {
    ranges.iter().any(|(from, to)| *from <= slot && slot <= *to)
}

impl Server {
    /// Give up on a migration one of whose connections has just gone.
    ///
    /// Called for every internal connection that ends, which is a handful over
    /// the life of a cluster, and does nothing at all unless one of them was
    /// carrying a migration.
    pub(crate) fn asm_forget(&self, conn: u64) {
        self.cluster.asm.forget(conn, self.now_ms() as i64);
        self.asm_relax();
    }

    /// Start a migration off this node, which is `CLUSTER SYNCSLOTS SYNC`.
    ///
    /// The far side picked the id, so a retry of a migration that failed comes
    /// back with the same one and is the same task started again rather than a
    /// second one. Anything else running is refused, because one at a time is the
    /// rule and telling the far side so is what makes it wait rather than sit
    /// there.
    pub(super) fn asm_begin_migrate(
        &self,
        id: &[u8],
        dest: &[u8],
        slots: Vec<(u16, u16)>,
        row: &Arc<Client>,
    ) -> Result<()> {
        let now = self.now_ms() as i64;
        let source = self.cluster.map.lock().nodes[0].id.clone();
        let dest = if dest.is_empty() {
            vec![0u8; ID_LEN]
        } else {
            dest.to_vec()
        };
        let mut tasks = self.cluster.asm.inner.lock();
        // The same move being tried again keeps its count of tries. Anything
        // else is a new move and the one that failed is given up on, which is
        // what makes a node that has failed a migration take the next one
        // instead of refusing everything forever.
        let mut retries = 0;
        let mut replace = false;
        if let Some(live) = tasks.live.as_ref() {
            if live.state != State::Failed {
                return Err(Error::new(
                    Code::Invalid,
                    "Another ASM task is already in progress",
                ));
            }
            if live.id.as_bytes() == id && !live.import && live.slots == slots && live.dest == dest
            {
                retries = live.retries + 1;
            } else {
                replace = true;
            }
        }
        if replace {
            let live = tasks
                .live
                .as_mut()
                .expect("there is one, or replace is not set");
            live.blame("Cancelled due to new migration requested");
            live.state = State::Canceled;
            self.cluster.asm.retire(&mut tasks, now);
        }
        tasks.live = Some(Task {
            id: yo_alloc::allow(|| String::from_utf8_lossy(id).into_owned()),
            slots,
            source: source.into_bytes(),
            dest,
            import: false,
            state: State::WaitRdbChannel,
            error: String::new(),
            retries,
            created: now,
            started: now,
            ended: -1,
            main: Some(Arc::clone(row)),
            rdb: None,
            sent: 0,
            acked: 0,
            dest_state: State::None,
            paused: 0,
            rdb_state: State::None,
        });
        Ok(())
    }

    /// Take the second connection of a migration, which is
    /// `CLUSTER SYNCSLOTS RDBCHANNEL`.
    ///
    /// Answers the slot ranges to snapshot, having moved the task on to say the
    /// snapshot is being built. The building itself is left to the caller and is
    /// deliberately not done under this lock: it stops every write on the server
    /// for as long as it takes, and holding a second lock across that would mean
    /// nothing could even ask what the migration was doing.
    pub(super) fn asm_take_rdb_channel(&self, id: &[u8], conn: u64) -> Result<Vec<(u16, u16)>> {
        let mut tasks = self.cluster.asm.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return Err(Error::new(
                Code::Invalid,
                "No slot migration task in progress",
            ));
        };
        if task.import || task.state != State::WaitRdbChannel || task.id.as_bytes() != id {
            return Err(Error::new(
                Code::Invalid,
                "Another migration task is already in progress",
            ));
        }
        if task.main.is_none() {
            return Err(Error::new(
                Code::Invalid,
                "Main channel connection is not established",
            ));
        }
        task.rdb = Some(conn);
        task.state = State::WaitBgsaveStart;
        Ok(task.slots.clone())
    }

    /// The snapshot of `slots`, as the stream of commands the far side runs.
    ///
    /// Built at one instant of the dataset, so what comes back is what the slots
    /// held at one point and not a smear across several. The change stream is
    /// switched on inside that same instant, which is the only place it can be
    /// switched on and have the two meet exactly: a write that landed in the gap
    /// would be in the snapshot and in the stream, or in neither.
    pub(super) fn asm_snapshot(&self, slots: &[(u16, u16)]) -> Vec<u8> {
        let (image, _at) = self.at_an_instant(|| {
            let mut out = Out::with_capacity(Proto::Resp2, 4096);
            self.write_snapshot(slots, &mut out);
            self.cluster.asm.start_stream();
            out.into_inner()
        });
        image
    }

    /// Whether anything at all is listening to what this server writes.
    ///
    /// A replica is the usual reason and a migration in flight is the other one,
    /// and the second is why this is not simply [`Server::replicated`]: a node
    /// with no replicas still has to send the changes to the slots it is handing
    /// over, or the far side takes them holding what they looked like a moment
    /// ago.
    pub(crate) fn propagating(&self) -> bool {
        self.replicated() || self.cluster.asm.streaming.load(Relaxed)
    }

    /// Hand one propagated command to a migration, if it belongs to one.
    ///
    /// `wire` is the command already rendered, which is the same bytes the
    /// replication stream carries and the same bytes the far side will be sent,
    /// so nothing is built twice.
    ///
    /// Reading the keys back out of it means decoding it again, which is a pass
    /// over bytes that were just written. That is worth saying out loud: it is
    /// the price of hooking in at the one place everything propagated comes
    /// through, rewrites and expiries and script effects and all, rather than at
    /// each of the hundred places that reach it. It costs nothing at all unless
    /// a migration is running, which is the case this is arranged around.
    pub(crate) fn asm_feed(&self, wire: &[u8]) {
        if !self.cluster.asm.streaming.load(Relaxed) {
            return;
        }
        let mut argv = Argv::new();
        let read = yo_alloc::allow(|| argv.decode(wire, &Limits::DEFAULT));
        if !matches!(read, Ok(Step::Command { .. })) {
            return;
        }
        let args = Args::new(&argv, wire);
        let Some(spec) = table::lookup(args.name()) else {
            return;
        };
        // A command that names no key is not sent at all. That covers `SELECT`,
        // which has nothing to say to a cluster with one database, `MULTI` and
        // `EXEC`, which have nothing to group before the handoff, and `PING`.
        if !keyspec::takes_keys(spec, args, 0) {
            return;
        }
        let mut slot: Option<u16> = None;
        let mut crossed = false;
        keyspec::find(spec, args, 0, &mut |run| {
            for i in 0..run.count {
                let at = run.first + i * run.step;
                if at >= args.len() {
                    continue;
                }
                let this = key_slot(args.get(at));
                match slot {
                    None => slot = Some(this),
                    Some(first) if first != this => crossed = true,
                    Some(_) => {}
                }
            }
        });
        if crossed {
            self.cluster.asm.cross_slot(self.now_ms() as i64);
            return;
        }
        let Some(slot) = slot else {
            return;
        };
        // Handed over while the task is held, which is what puts the stream in
        // one order. Two threads writing two keys of the same slot at the same
        // time both get here, and whichever of them counts its bytes first is
        // the one whose command goes first, rather than the two racing between
        // the counting and the handing over and arriving the wrong way round.
        // It is also what lets the end of the stream be written with nothing
        // able to slip in behind it.
        let mut tasks = self.cluster.asm.inner.lock();
        let Some(task) = tasks.live.as_mut() else {
            return;
        };
        if task.import || !task.state.streaming() {
            return;
        }
        if !task
            .slots
            .iter()
            .any(|(from, to)| (*from..=*to).contains(&slot))
        {
            return;
        }
        task.sent += wire.len() as u64;
        let Some(row) = task.main.clone() else {
            return;
        };
        let shared = yo_alloc::allow(|| Arc::new(wire.to_vec()));
        self.post(
            row.thread.load(Relaxed),
            Envelope::raw(row.conn.load(Relaxed), row.id, shared),
        );
    }

    /// Take an acknowledgement from the node the slots are going to.
    pub(super) fn asm_ack(&self, conn: u64, state: &[u8], offset: u64) {
        let Some(state) = State::dest_word(state) else {
            return;
        };
        if self.cluster.asm.ack(conn, state, offset) {
            self.asm_handoff();
        }
    }

    /// Stop taking writes and close the stream, which is the handoff.
    ///
    /// Three steps in an order that matters. Writes stop first, so that nothing
    /// new can land in a slot that is about to belong to somebody else. Then the
    /// server is frozen, which waits out the writes that were already running
    /// and were let through before the pause; their changes go down the stream
    /// like any others because the task is still in a state that sends them.
    /// Only then is the end of the stream written, and by then there is nothing
    /// that could come after it.
    ///
    /// The pause has a deadline of its own as well as being lifted by hand. If
    /// this node is left holding it because the far side went away in the wrong
    /// half second, it lets go on its own at the same moment the cron gives up
    /// on the task, and a server that stops taking writes forever is not a thing
    /// a bug in here should be able to produce.
    fn asm_handoff(&self) {
        let now = self.now_ms();
        let timeout = self.migration_knob(Migration::Pause).max(0) as u64;
        let until = now.saturating_add(timeout);
        if !self.cluster.asm.begin_handoff(now as i64, until) {
            return;
        }
        self.pause(until, false);
        let done = self.at_an_instant(|| self.cluster.asm.end_stream()).0;
        let Some(row) = done else {
            return;
        };
        let mut out = Out::with_capacity(Proto::Resp2, 64);
        out.array(3);
        out.bulk(b"CLUSTER");
        out.bulk(b"SYNCSLOTS");
        out.bulk(b"STREAM-EOF");
        let shared = yo_alloc::allow(|| Arc::new(out.into_inner()));
        self.post(
            row.thread.load(Relaxed),
            Envelope::raw(row.conn.load(Relaxed), row.id, shared),
        );
    }

    /// The migration's share of the cluster's clock.
    ///
    /// One thing so far: a handoff that has held the server for longer than it
    /// is allowed to is given up on. The far side is not told, because the only
    /// connection to it is the one it is expected to close.
    pub(crate) fn asm_cron(&self) {
        let timeout = self.migration_knob(Migration::Pause).max(0);
        if self
            .cluster
            .asm
            .pause_expired(self.now_ms() as i64, timeout)
        {
            self.asm_relax();
        }
    }

    /// Let writes go again if the handoff that stopped them is over.
    ///
    /// Called wherever a task can end, and once a tick besides, because a task
    /// ends down several paths and the one that must not happen is a server left
    /// paused by a migration that is no longer running.
    pub(crate) fn asm_relax(&self) {
        let armed = self.cluster.asm.armed.load(Relaxed);
        if armed == 0 || self.cluster.asm.pausing() {
            return;
        }
        self.cluster.asm.armed.store(0, Relaxed);
        self.lift(armed, false);
    }

    /// Slots this node was serving have moved to somebody else.
    ///
    /// Called from the bus once a claim carrying a higher epoch has been
    /// believed and the map has been written, which is the moment a migration
    /// has been waiting for: the far side owns the slots and every other node is
    /// being told. What is left is to let the writes go and to drop the keys,
    /// because a node answering for a key in a slot it does not own is two nodes
    /// answering for the same data.
    ///
    /// `demoted` is a node that gave away its last slot and is now following the
    /// node that took them. It keeps its keys. It is about to be sent the whole
    /// dataset by its new master and throwing them away first would only mean
    /// copying them straight back.
    pub(crate) fn asm_slots_moved(&self, lost: &[u16], demoted: bool) {
        if lost.is_empty() {
            return;
        }
        let moved = joined(lost);
        let now = self.now_ms() as i64;
        let outcome = self.cluster.asm.config_updated(&moved, now);
        // First of all, whatever happened. A handoff that got this far is over
        // one way or the other and the writes it stopped have waited long
        // enough.
        self.asm_relax();
        match outcome {
            Moved::Done(slots) => self.trim_slots(&slots, Trim::Ranges),
            Moved::Cancelled => {}
            Moved::Elsewhere if !demoted => self.trim_slots(&moved, Trim::Keys),
            Moved::Elsewhere => {}
        }
    }

    /// Drop the keys of the slots a `TRIMSLOTS` names.
    ///
    /// The command has already been checked and is propagated by the dispatcher
    /// like any other write, so there is nothing to announce here.
    pub(super) fn trim_named_slots(&self, ranges: &[(u16, u16)]) {
        self.trim_slots(ranges, Trim::Already);
    }

    /// Drop the keys of slots this node does not serve any more.
    fn trim_slots(&self, ranges: &[(u16, u16)], how: Trim) {
        // Only the slots that really are somebody else's. A range that came back
        // to this node between the claim and here is not one to empty out, and a
        // slot with nothing in it is not worth naming either.
        let ranges = {
            let map = self.cluster.map.lock();
            let mine: Vec<u16> = yo_alloc::allow(|| {
                ranges
                    .iter()
                    .flat_map(|&(from, to)| from..=to)
                    .filter(|slot| map.owner[usize::from(*slot)] != Some(0))
                    .collect()
            });
            joined(&mine)
        };
        if ranges.is_empty() {
            return;
        }
        let armed = super::super::notify::arm(self, 0);
        for at in 0..self.dbs.len() {
            // Collected before anything is taken, because the walk holds one
            // stripe at a time while it reads and taking a key holds the stripe
            // that key is on, which is the same lock as often as not.
            let mut doomed: Vec<Vec<u8>> = Vec::new();
            self.dbs[at].keys(|key| {
                if within(&ranges, key_slot(key)) {
                    yo_alloc::allow(|| doomed.push(key.to_vec()));
                }
            });
            for key in &doomed {
                if !self.dbs[at].hold(key).del(key) {
                    continue;
                }
                match how {
                    Trim::Keys if self.propagating() => {
                        super::super::repl::announce(self, at, &[b"DEL", key]);
                    }
                    Trim::Keys => {}
                    Trim::Ranges | Trim::Already => super::super::notify::fire(
                        at,
                        super::super::notify::class::GENERIC,
                        "del",
                        key,
                    ),
                }
            }
        }
        if matches!(how, Trim::Ranges) && self.propagating() {
            let mut parts: Vec<Vec<u8>> = Vec::new();
            yo_alloc::allow(|| {
                parts.push(b"TRIMSLOTS".to_vec());
                parts.push(b"RANGES".to_vec());
                parts.push(ranges.len().to_string().into_bytes());
                for (from, to) in &ranges {
                    parts.push(from.to_string().into_bytes());
                    parts.push(to.to_string().into_bytes());
                }
                let wire: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
                super::super::repl::announce(self, 0, &wire);
            });
        }
        super::super::notify::drain(self, armed);
    }

    /// Read one of the `cluster-slot-migration-*` settings.
    pub(crate) fn migration_knob(&self, which: Migration) -> i64 {
        self.cluster.asm.knobs.of(which).load(Relaxed)
    }

    /// Write one of the `cluster-slot-migration-*` settings.
    ///
    /// It takes effect on the next thing that reads it, which for the lag bound
    /// is the next acknowledgement and for the two timeouts is the next tick of
    /// the cron. A migration already running is not restarted and does not need
    /// to be: none of these is remembered anywhere, they are all read fresh
    /// every time they are wanted.
    pub(crate) fn set_migration_knob(&self, which: Migration, value: i64) {
        self.cluster.asm.knobs.of(which).store(value, Relaxed);
    }

    /// The body of [`Server::asm_snapshot`], with nothing writing behind it.
    fn write_snapshot(&self, slots: &[(u16, u16)], out: &mut Out) {
        // Every library on the server, whether or not anything is using one. The
        // far side is about to run keys that may call them and it has no other
        // way of getting them, and `REPLACE` is there because it may already have
        // the same library from an earlier move.
        let functions = yo_alloc::allow(|| {
            let held = self.libraries.lock();
            rdb::functions(held.all().iter().map(|l| &*l.code))
        });
        out.array(4);
        out.bulk(b"FUNCTION");
        out.bulk(b"RESTORE");
        out.bulk(&functions);
        out.bulk(b"REPLACE");

        let wanted = |slot: u16| slots.iter().any(|(from, to)| (*from..=*to).contains(&slot));
        let mut scratch = Out::with_capacity(Proto::Resp2, 4096);
        for (at, db) in self.dbs.iter().enumerate() {
            if db.is_empty() {
                continue;
            }
            out.array(2);
            out.bulk(b"SELECT");
            out.bulk_int(at as i64);

            // Every key of the database once, kept only if its slot is one of the
            // ones moving, and then in slot order so that each slot's keys are
            // together and can be counted before the first of them goes out.
            let mut mine: Vec<(u16, Vec<u8>)> = Vec::new();
            db.keys(|key| {
                let slot = key_slot(key);
                if wanted(slot) {
                    mine.push((slot, key.to_vec()));
                }
            });
            mine.sort_unstable();

            let mut from = 0;
            while from < mine.len() {
                let slot = mine[from].0;
                let mut to = from;
                let mut expires = 0;
                scratch.clear();
                while to < mine.len() && mine[to].0 == slot {
                    let key = &mine[to].1;
                    let mut stripe = db.hold(key);
                    let deadline = match stripe.deadline_of(key) {
                        Ask::At(when) => {
                            expires += 1;
                            Some(when as i64)
                        }
                        _ => None,
                    };
                    // A string goes as the command that would set it rather than
                    // as a dump, which is the reference's rule and is about the
                    // far side rather than about the wire: it can take a `SET`
                    // apart as it arrives, where a dump has to be whole before
                    // any of it can be used. The reference does the same for any
                    // collection above five hundred keys for the same reason, and
                    // this does not, which is D-149.
                    let mut wrote = stripe.kind_of(key) == Some(Kind::String);
                    if wrote {
                        let value = stripe.get(key).ok().flatten();
                        match value {
                            Some(value) => {
                                scratch.array(3);
                                scratch.bulk(b"SET");
                                scratch.bulk(key);
                                match value {
                                    Str::Int(n) => scratch.bulk_int(n),
                                    Str::Bytes(b) => scratch.bulk(b),
                                }
                            }
                            None => wrote = false,
                        }
                        if let (true, Some(when)) = (wrote, deadline) {
                            scratch.array(3);
                            scratch.bulk(b"PEXPIREAT");
                            scratch.bulk(key);
                            scratch.bulk_int(when);
                        }
                    } else if let Some(payload) = stripe.dump(key) {
                        // A key that went between the two walks is not an error
                        // and is simply not in the snapshot, which is what the
                        // far side would have been told by the change stream
                        // anyway.
                        scratch.array(5);
                        scratch.bulk(b"RESTORE");
                        scratch.bulk(key);
                        scratch.bulk_int(deadline.unwrap_or(0));
                        scratch.bulk(&payload);
                        // Absolute rather than a duration, because the far side
                        // runs this at some unknown moment after it was written
                        // and a duration would restart the clock.
                        scratch.bulk(b"ABSTTL");
                    }
                    to += 1;
                }
                // How much is coming, so the far side can make room for it in one
                // go rather than growing its tables all the way up.
                let info = yo_alloc::allow(|| format!("{slot}:{}:{expires}", to - from));
                out.array(5);
                out.bulk(b"CLUSTER");
                out.bulk(b"SYNCSLOTS");
                out.bulk(b"CONF");
                out.bulk(b"SLOT-INFO");
                out.bulk(info.as_bytes());
                out.raw(scratch.as_slice());
                from = to;
            }
        }

        out.array(3);
        out.bulk(b"CLUSTER");
        out.bulk(b"SYNCSLOTS");
        out.bulk(b"SNAPSHOT-EOF");
    }
}

// ---------------------------------------------------------------- the tests

#[cfg(test)]
mod tests {
    use yo_kv::End;
    use yo_kv::strings::{Expire, SetOptions};

    use std::sync::Arc;

    use super::super::super::Server;
    use super::super::super::clients::Client;
    use super::{State, key_slot};
    use crate::proto::Proto;
    use crate::reply::Out;

    /// The default lag bound as an offset, which is what these count in.
    const LAG: u64 = super::MAX_LAG as u64;

    /// The default number of finished tasks kept.
    const KEEP: usize = super::MAX_ARCHIVED as usize;

    /// A server in cluster mode holding every slot, which is what the far side
    /// of a migration talks to.
    fn node() -> Server {
        let mut server = Server::new();
        server.enable_cluster("", 7351);
        server
    }

    /// The reply as text, with the payload bytes readable enough to look for a
    /// command in. A `DUMP` payload is arbitrary bytes, so this is only ever
    /// searched, never compared whole.
    fn text(out: &Out) -> String {
        String::from_utf8_lossy(out.as_slice()).into_owned()
    }

    /// `SET key value`, straight into the keyspace.
    fn set(server: &Server, key: &[u8], value: &[u8], deadline: Option<u64>) {
        let mut opts = SetOptions::PLAIN;
        if let Some(when) = deadline {
            opts.expire = Expire::At(when);
        }
        server.dbs[0]
            .hold(key)
            .set(key, value, opts)
            .expect("the key is new");
    }

    /// The snapshot of every slot, which is what a whole node moving would ask
    /// for and what makes the shape easiest to read.
    /// A connection row standing in for one of the two the far side opens.
    ///
    /// Nothing here writes to it, so what matters is only its id, which is what
    /// a task holds a channel by.
    fn wire(id: u64) -> Arc<Client> {
        Arc::new(Client::new(id))
    }

    fn snapshot(server: &Server) -> String {
        let mut out = Out::with_capacity(Proto::Resp2, 1024);
        server.write_snapshot(&[(0, 16383)], &mut out);
        text(&out)
    }

    /// The stream opens with the libraries and closes with the end marker,
    /// whether or not there is anything in between, because the far side reads
    /// until the marker and would otherwise wait forever on an empty node.
    #[test]
    fn an_empty_node_still_sends_the_functions_and_the_end() {
        let got = snapshot(&node());
        assert!(
            got.starts_with("*4\r\n$8\r\nFUNCTION\r\n$7\r\nRESTORE\r\n"),
            "{got:?}"
        );
        assert!(got.ends_with("*3\r\n$7\r\nCLUSTER\r\n$9\r\nSYNCSLOTS\r\n$12\r\nSNAPSHOT-EOF\r\n"));
        // Nothing in it, so no database is selected and no slot is described.
        assert!(!got.contains("SELECT"), "{got:?}");
        assert!(!got.contains("SLOT-INFO"), "{got:?}");
    }

    /// A string goes as the command that would set it, which is the one place
    /// the reference does not use a dump and the one thing a reader of this
    /// code is most likely to get wrong.
    #[test]
    fn a_string_goes_as_a_set_and_its_deadline_follows_it() {
        let server = node();
        set(&server, b"plain", b"a string", None);
        set(&server, b"dated", b"goes", Some(4_000_000_000_000));
        let got = snapshot(&server);
        assert!(
            got.contains("$3\r\nSET\r\n$5\r\nplain\r\n$8\r\na string\r\n"),
            "{got:?}"
        );
        assert!(
            got.contains("$3\r\nSET\r\n$5\r\ndated\r\n$4\r\ngoes\r\n"),
            "{got:?}"
        );
        // Absolute, and after the value rather than folded into it, because the
        // far side runs these one at a time and a duration would restart the
        // clock at whatever moment it got there.
        assert!(
            got.contains("$9\r\nPEXPIREAT\r\n$5\r\ndated\r\n$13\r\n4000000000000\r\n"),
            "{got:?}"
        );
        assert!(!got.contains("RESTORE\r\n$5\r\ndated"), "{got:?}");
    }

    /// Anything that is not a string goes as a dump, and its deadline rides
    /// along inside the `RESTORE` rather than as a second command.
    #[test]
    fn a_collection_goes_as_a_restore() {
        let server = node();
        server.dbs[0]
            .hold(b"list")
            .push(b"list", End::Right, [b"a".as_slice(), b"b"].into_iter())
            .expect("the key is new");
        let got = snapshot(&server);
        assert!(
            got.contains("$7\r\nRESTORE\r\n$4\r\nlist\r\n$1\r\n0\r\n"),
            "{got:?}"
        );
        assert!(got.contains("$6\r\nABSTTL\r\n"), "{got:?}");
    }

    /// The count in front of each slot is how many keys of that slot are about
    /// to arrive and how many of them have a deadline, so the far side can make
    /// room in one go instead of growing its tables all the way up.
    #[test]
    fn each_slot_is_counted_before_its_keys_arrive() {
        let server = node();
        // Two keys in one slot through the hash tag, so the count has to be two
        // rather than one line per key.
        set(&server, b"{tag}one", b"1", None);
        set(&server, b"{tag}two", b"2", Some(4_000_000_000_000));
        let slot = key_slot(b"{tag}one");
        let got = snapshot(&server);
        let want = format!(
            "$9\r\nSLOT-INFO\r\n${}\r\n{slot}:2:1\r\n",
            format!("{slot}:2:1").len()
        );
        assert!(got.contains(&want), "{want:?} in {got:?}");
        // And the database it is all in, once, in front of the lot.
        assert_eq!(got.matches("$6\r\nSELECT\r\n").count(), 1, "{got:?}");
    }

    /// Only the slots asked for, because the whole point of this over a full
    /// resync is that a sixteen thousandth of the keyspace can be moved.
    #[test]
    fn a_key_outside_the_range_is_left_alone() {
        let server = node();
        set(&server, b"foo", b"in", None);
        set(&server, b"bar", b"out", None);
        let foo = key_slot(b"foo");
        let bar = key_slot(b"bar");
        assert_ne!(foo, bar);
        let mut out = Out::with_capacity(Proto::Resp2, 1024);
        server.write_snapshot(&[(foo, foo)], &mut out);
        let got = text(&out);
        assert!(got.contains("$3\r\nfoo\r\n"), "{got:?}");
        assert!(!got.contains("$3\r\nbar\r\n"), "{got:?}");
    }

    /// The handshake in order, which is what the far side does, and what the
    /// task says at each step.
    #[test]
    fn a_migration_walks_from_the_sync_to_the_snapshot() {
        let server = node();
        let id = [b'b'; 40];
        let dest = [b'c'; 40];
        server
            .asm_begin_migrate(&id, &dest, vec![(0, 100)], &wire(7))
            .expect("nothing else is running");
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_all(&mut out);
        let got = text(&out);
        assert!(got.contains("wait-rdbchannel"), "{got:?}");
        assert!(got.contains("$7\r\nmigrate\r\n"), "{got:?}");
        assert!(got.contains("$5\r\n0-100\r\n"), "{got:?}");

        let slots = server
            .asm_take_rdb_channel(&id, 8)
            .expect("the task is waiting for it");
        assert_eq!(slots, vec![(0, 100)]);
        let image = server.asm_snapshot(&slots);
        assert!(image.ends_with(b"$12\r\nSNAPSHOT-EOF\r\n"));
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&id, &mut out);
        assert!(text(&out).contains("send-stream"), "{:?}", text(&out));
        // And taking the snapshot is what opens the change stream, because the
        // two have to meet at the same instant.
        assert!(server.propagating());
    }

    /// A write to a slot that is moving is counted against the stream, and one
    /// to a slot that is not is left alone.
    #[test]
    fn only_the_slots_that_are_moving_are_streamed() {
        let server = node();
        let foo = key_slot(b"foo");
        let bar = key_slot(b"bar");
        assert_ne!(foo, bar);
        server
            .asm_begin_migrate(&[b'b'; 40], &[b'c'; 40], vec![(foo, foo)], &wire(7))
            .expect("nothing else is running");
        server
            .asm_take_rdb_channel(&[b'b'; 40], 8)
            .expect("the task is waiting for it");
        server.asm_snapshot(&[(foo, foo)]);

        let sent = |server: &Server| server.cluster.asm.inner.lock().live.as_ref().unwrap().sent;
        assert_eq!(sent(&server), 0);
        server.asm_feed(b"*3\r\n$3\r\nSET\r\n$3\r\nbar\r\n$1\r\nx\r\n");
        assert_eq!(sent(&server), 0, "another slot is not this migration");
        server.asm_feed(b"*1\r\n$4\r\nPING\r\n");
        assert_eq!(sent(&server), 0, "a command with no key is not sent");
        let write = b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$1\r\nx\r\n";
        server.asm_feed(write);
        assert_eq!(sent(&server), write.len() as u64);
    }

    /// A command that reaches into two slots at once cannot be sent, because
    /// there is no half of it that leaves the far side right.
    #[test]
    fn a_cross_slot_command_ends_the_migration() {
        let server = node();
        let foo = key_slot(b"foo");
        server
            .asm_begin_migrate(&[b'b'; 40], &[b'c'; 40], vec![(foo, foo)], &wire(7))
            .expect("nothing else is running");
        server
            .asm_take_rdb_channel(&[b'b'; 40], 8)
            .expect("the task is waiting for it");
        server.asm_snapshot(&[(foo, foo)]);
        server.asm_feed(b"*3\r\n$3\r\nDEL\r\n$3\r\nfoo\r\n$3\r\nbar\r\n");
        assert!(!server.propagating());
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&[b'b'; 40], &mut out);
        let got = text(&out);
        assert!(got.contains("canceled"), "{got:?}");
        // The states in the sentence are the ones it was in when it went wrong,
        // not the ones it moved to because it did.
        assert!(
            got.contains(
                "Cancelled due to propagating cross slot command (state: send-stream, rdb_channel_state: completed)"
            ),
            "{got:?}"
        );
    }

    /// The far side saying how far it has got is what moves the task on to the
    /// point where writes would stop.
    #[test]
    fn catching_up_stops_the_writes_and_ends_the_stream() {
        let server = node();
        let id = [b'b'; 40];
        server
            .asm_begin_migrate(&id, &[b'c'; 40], vec![(0, 16383)], &wire(7))
            .expect("nothing else is running");
        server
            .asm_take_rdb_channel(&id, 8)
            .expect("the task is waiting for it");
        server.asm_snapshot(&[(0, 16383)]);
        {
            let mut tasks = server.cluster.asm.inner.lock();
            tasks.live.as_mut().unwrap().sent = LAG * 4;
        }
        // A word that is not one the far side is allowed to send changes
        // nothing, and neither does an acknowledgement on another connection.
        server.asm_ack(7, b"takeover", LAG * 4);
        server.asm_ack(9, b"streaming-buffer", LAG * 4);
        // Still a long way behind.
        server.asm_ack(7, b"streaming-buffer", LAG);
        let state = |server: &Server| server.cluster.asm.inner.lock().live.as_ref().unwrap().state;
        assert_eq!(state(&server), State::SendStream);
        // Going backwards is stale rather than wrong, and is dropped.
        server.asm_ack(7, b"streaming-buffer", 0);
        assert_eq!(
            server.cluster.asm.inner.lock().live.as_ref().unwrap().acked,
            LAG
        );
        // And within a megabyte is close enough, which stops the writes and ends
        // the stream in one go.
        server.asm_ack(7, b"wait-stream-eof", LAG * 3);
        assert_eq!(state(&server), State::StreamEof);
        assert_eq!(server.paused(server.now_ms()), Some(false), "writes only");
        assert!(!server.propagating(), "nothing more is sent");
        // The connection is let go of rather than closed, so the far side
        // hanging up on it afterwards is not read as a migration falling over.
        assert!(
            server
                .cluster
                .asm
                .inner
                .lock()
                .live
                .as_ref()
                .unwrap()
                .main
                .is_none()
        );
        server.asm_forget(7);
        assert_eq!(state(&server), State::StreamEof);
        // And nothing that arrives late is counted, because nothing is listening
        // for it any more.
        server.asm_feed(b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$1\r\nx\r\n");
        assert_eq!(
            server.cluster.asm.inner.lock().live.as_ref().unwrap().sent,
            LAG * 4
        );
    }

    /// A far side that never takes the slots does not get to hold this server
    /// shut forever.
    #[test]
    fn a_handoff_the_far_side_never_finishes_gives_up() {
        let server = node();
        let id = [b'b'; 40];
        server
            .asm_begin_migrate(&id, &[b'c'; 40], vec![(0, 16383)], &wire(7))
            .expect("nothing else is running");
        server
            .asm_take_rdb_channel(&id, 8)
            .expect("the task is waiting for it");
        server.asm_snapshot(&[(0, 16383)]);
        server.asm_ack(7, b"wait-stream-eof", 0);
        assert_eq!(server.paused(server.now_ms()), Some(false));
        // Nothing has taken too long yet.
        server.asm_cron();
        assert_eq!(server.paused(server.now_ms()), Some(false));
        // Now it has. The task fails and the pause goes with it.
        {
            let mut tasks = server.cluster.asm.inner.lock();
            let task = tasks.live.as_mut().unwrap();
            task.paused -= super::WRITE_PAUSE + 1;
        }
        server.asm_cron();
        assert_eq!(server.paused(server.now_ms()), None);
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&id, &mut out);
        let got = text(&out);
        assert!(
            got.contains(
                "Write pause timeout during slot handoff: destination did not take ownership within 10000 ms. (state: stream-eof, rdb_channel_state: completed)"
            ),
            "{got:?}"
        );
        // A migration that failed reports no pause, however long it held the
        // server, because the number is only for one that got through.
        assert!(got.contains("write_pause_ms\r\n:0\r\n"), "{got:?}");
    }

    /// Cancelling a migration that has stopped the writes starts them again,
    /// and does not touch a pause somebody else put on.
    #[test]
    fn cancelling_a_handoff_lets_the_writes_go() {
        let server = node();
        let id = [b'b'; 40];
        server
            .asm_begin_migrate(&id, &[b'c'; 40], vec![(0, 16383)], &wire(7))
            .expect("nothing else is running");
        server
            .asm_take_rdb_channel(&id, 8)
            .expect("the task is waiting for it");
        server.asm_snapshot(&[(0, 16383)]);
        server.asm_ack(7, b"wait-stream-eof", 0);
        assert_eq!(server.paused(server.now_ms()), Some(false));
        assert_eq!(
            server.cluster.asm.cancel(Some(&id), server.now_ms() as i64),
            1
        );
        server.asm_relax();
        assert_eq!(server.paused(server.now_ms()), None);
    }

    /// Play a migration of one slot range as far as `stream-eof`, which is the
    /// point every test below this one starts from.
    fn handed_over(server: &Server, id: &[u8], slots: Vec<(u16, u16)>) {
        server
            .asm_begin_migrate(id, &[b'c'; 40], slots.clone(), &wire(7))
            .expect("nothing else is running");
        server
            .asm_take_rdb_channel(id, 8)
            .expect("the task is waiting for it");
        server.asm_snapshot(&slots);
        server.asm_ack(7, b"wait-stream-eof", 0);
    }

    /// The far side claiming the slots is the end of the migration: the task is
    /// done, the writes go again, and the keys that moved are dropped.
    #[test]
    fn the_slots_changing_hands_finishes_the_migration() {
        let server = node();
        let id = [b'b'; 40];
        let foo = key_slot(b"foo");
        let bar = key_slot(b"bar");
        assert_ne!(foo, bar);
        set(&server, b"foo", b"moved", None);
        set(&server, b"bar", b"stayed", None);
        handed_over(&server, &id, vec![(foo, foo)]);
        assert_eq!(server.paused(server.now_ms()), Some(false));

        server.asm_slots_moved(&[foo], false);
        assert_eq!(server.paused(server.now_ms()), None, "the writes go again");
        assert!(server.cluster.asm.inner.lock().live.is_none());
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&id, &mut out);
        let got = text(&out);
        assert!(got.contains("$9\r\ncompleted\r\n"), "{got:?}");
        // A task that got through says nothing went wrong, even if an earlier
        // try did.
        assert!(got.contains("last_error\r\n$0\r\n\r\n"), "{got:?}");
        // And the keys of the slot that moved are gone, while the rest are not.
        assert!(!server.dbs[0].hold(b"foo").exists(b"foo"));
        assert!(server.dbs[0].hold(b"bar").exists(b"bar"));
    }

    /// A claim that does not name exactly what a migration was moving is not
    /// that migration finishing, so the migration is given up on.
    #[test]
    fn a_claim_for_the_wrong_slots_ends_the_migration() {
        let server = node();
        let id = [b'b'; 40];
        handed_over(&server, &id, vec![(100, 200)]);
        // Half of what it was moving, which is something else happening to the
        // cluster and not this move getting there.
        server.asm_slots_moved(&[150], false);
        assert_eq!(server.paused(server.now_ms()), None);
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&id, &mut out);
        let got = text(&out);
        assert!(
            got.contains(
                "Cancelled due to slots configuration updated (state: stream-eof, rdb_channel_state: completed)"
            ),
            "{got:?}"
        );
    }

    /// A slot going somewhere else while a migration is running is nothing to do
    /// with that migration, and leaves it alone.
    #[test]
    fn a_claim_somewhere_else_leaves_the_migration_alone() {
        let server = node();
        let id = [b'b'; 40];
        handed_over(&server, &id, vec![(100, 200)]);
        server.asm_slots_moved(&[300], false);
        let state = server.cluster.asm.inner.lock().live.as_ref().unwrap().state;
        assert_eq!(state, State::StreamEof);
    }

    /// A slot that moved with no migration behind it still takes its keys with
    /// it, because two nodes answering for the same key is the one thing that
    /// must not happen.
    #[test]
    fn slots_that_moved_on_their_own_drop_their_keys() {
        let server = node();
        let foo = key_slot(b"foo");
        set(&server, b"foo", b"gone", None);
        server.asm_slots_moved(&[foo], false);
        assert!(!server.dbs[0].hold(b"foo").exists(b"foo"));
    }

    /// Except on a node that gave away its last slot. It is following whoever
    /// took them now and is about to be sent the whole dataset, so throwing the
    /// keys away first would only mean copying them straight back.
    #[test]
    fn a_node_that_lost_everything_keeps_its_keys() {
        let server = node();
        let foo = key_slot(b"foo");
        set(&server, b"foo", b"kept", None);
        server.asm_slots_moved(&[foo], true);
        assert!(server.dbs[0].hold(b"foo").exists(b"foo"));
    }

    /// One at a time, because two moves at once would each be pausing writes
    /// for the other to catch up.
    #[test]
    fn a_second_migration_is_refused_while_one_is_running() {
        let server = node();
        server
            .asm_begin_migrate(&[b'b'; 40], &[b'c'; 40], vec![(0, 100)], &wire(7))
            .expect("nothing else is running");
        let err = server
            .asm_begin_migrate(&[b'd'; 40], &[b'c'; 40], vec![(200, 300)], &wire(9))
            .expect_err("one is running");
        assert!(
            err.to_string()
                .contains("Another ASM task is already in progress")
        );
    }

    /// The same move asked for again after it failed is the same task, counted,
    /// rather than a second one, because the far side retries with the id it
    /// picked the first time.
    #[test]
    fn a_retry_of_the_same_move_keeps_its_count() {
        let server = node();
        let id = [b'b'; 40];
        let dest = [b'c'; 40];
        server
            .asm_begin_migrate(&id, &dest, vec![(0, 100)], &wire(7))
            .expect("nothing else is running");
        // The connection the far side asked over goes away, which is the end of
        // that attempt.
        server.asm_forget(7);
        {
            let tasks = server.cluster.asm.inner.lock();
            assert!(tasks.live.is_none());
            assert_eq!(tasks.done[0].state, State::Failed);
            assert!(
                tasks.done[0]
                    .error
                    .contains("Connection with the peer node was lost")
            );
        }
        // A failed task is not live, so the retry is simply the next task, and
        // what matters is that it is taken at all.
        server
            .asm_begin_migrate(&id, &dest, vec![(0, 100)], &wire(11))
            .expect("the failed one does not block it");
    }

    /// Cancelling says how many it cancelled, and a task that has already
    /// finished cannot be cancelled again.
    #[test]
    fn cancelling_counts_once() {
        let server = node();
        let id = [b'b'; 40];
        server
            .asm_begin_migrate(&id, &[b'c'; 40], vec![(0, 100)], &wire(7))
            .expect("nothing else is running");
        assert_eq!(server.cluster.asm.cancel(Some(&[b'z'; 40]), 1), 0);
        assert_eq!(server.cluster.asm.cancel(None, 2), 1);
        assert_eq!(server.cluster.asm.cancel(None, 3), 0);
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&id, &mut out);
        let got = text(&out);
        assert!(got.contains("canceled"), "{got:?}");
        assert!(got.contains("Cancelled due to user request"), "{got:?}");
    }

    /// The snapshot connection cannot arrive before the request it belongs to,
    /// and cannot claim a task that is not the one that is waiting.
    #[test]
    fn the_snapshot_connection_has_to_match_the_task() {
        let server = node();
        let err = server
            .asm_take_rdb_channel(&[b'b'; 40], 8)
            .expect_err("nothing is running");
        assert!(
            err.to_string()
                .contains("No slot migration task in progress")
        );
        server
            .asm_begin_migrate(&[b'b'; 40], &[b'c'; 40], vec![(0, 100)], &wire(7))
            .expect("nothing else is running");
        let err = server
            .asm_take_rdb_channel(&[b'd'; 40], 8)
            .expect_err("that is not the task");
        assert!(
            err.to_string()
                .contains("Another migration task is already in progress")
        );
        // And the right one twice, because the second is a far side that has
        // lost track and must not get a second snapshot.
        server
            .asm_take_rdb_channel(&[b'b'; 40], 8)
            .expect("it matches");
        let err = server
            .asm_take_rdb_channel(&[b'b'; 40], 9)
            .expect_err("it has already been taken");
        assert!(
            err.to_string()
                .contains("Another migration task is already in progress")
        );
    }

    /// Only so many finished tasks are kept, so a cluster that has been moving
    /// slots for a month does not answer `STATUS ALL` with a month of them.
    #[test]
    fn the_finished_list_is_bounded() {
        let server = node();
        for i in 0..(KEEP + 5) {
            let id = format!("{i:040}");
            server
                .asm_begin_migrate(id.as_bytes(), &[b'c'; 40], vec![(0, 100)], &wire(7))
                .expect("the last one was cancelled");
            assert_eq!(server.cluster.asm.cancel(None, 1), 1);
        }
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_all(&mut out);
        let got = text(&out);
        assert!(got.starts_with(&format!("*{KEEP}\r\n")), "{got:?}");
        // Newest first, which is the order the reference keeps them in.
        let newest = format!("{:040}", KEEP + 4);
        assert!(got.contains(&newest), "{got:?}");
    }
}
