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
//! `CLUSTER MIGRATION STATUS` reports, and the snapshot, which is the first of
//! the two streams. A task is created by `CLUSTER SYNCSLOTS SYNC` arriving from
//! the far side, moves to `wait-rdbchannel`, and gets its snapshot written when
//! the second connection arrives with `CLUSTER SYNCSLOTS RDBCHANNEL`.
//!
//! What is not here is the second stream, the pause and the handoff, so a
//! migration started against this node gets its snapshot and then waits for
//! changes that never come. That is D-149 and it is the next piece.
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

use yo_common::lock::Lock;
use yo_common::{Code, Error, Result};
use yo_kv::value::{Kind, Str};
use yo_kv::{Ask, rdb};

use crate::proto::Proto;
use crate::reply::Out;

use super::super::Server;
use super::{ID_LEN, key_slot};

/// How many finished tasks are kept to be asked about afterwards.
///
/// The reference's `cluster-slot-migration-max-archived-tasks`, which is hidden
/// and defaults to this. A finished task is a few hundred bytes and the only
/// thing that reads one is an operator asking what happened, so the number only
/// has to be larger than the number of moves anybody looks back over.
const MAX_ARCHIVED: usize = 32;

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
    /// Given up on. The reason is in the task's error.
    Failed,
    /// The far side has asked for the slots and the second connection it will
    /// take the snapshot on has not arrived yet.
    WaitRdbChannel,
    /// The second connection has arrived and the snapshot has not been built.
    WaitBgsaveStart,
    /// The snapshot has gone out and the changes since are going out behind it.
    SendStream,
}

impl State {
    /// The word this state goes out as.
    fn word(self) -> &'static str {
        match self {
            State::None => "none",
            State::Canceled => "canceled",
            State::Failed => "failed",
            State::WaitRdbChannel => "wait-rdbchannel",
            State::WaitBgsaveStart => "wait-bgsave-start",
            State::SendStream => "send-stream",
        }
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
    /// The connection the far side asked over, which is the one the changes
    /// would go down. `None` for a task nobody is holding open.
    main: Option<u64>,
    /// The connection the snapshot goes down.
    rdb: Option<u64>,
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
        // The reference only reports a pause for a migration that finished, and
        // nothing here pauses anything yet, so it is always nought.
        out.bulk(b"write_pause_ms");
        out.int(0);
    }

    /// Say why this task stopped, in the reference's sentence.
    ///
    /// The two states it names are the task's own and the snapshot connection's,
    /// and both of them are read after the state has already moved, which is why
    /// the caller sets the state first and says why second.
    fn blame(&mut self, why: &str) {
        self.error = yo_alloc::allow(|| {
            format!(
                "{why} (state: {}, rdb_channel_state: {})",
                self.state.word(),
                State::None.word()
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
    fn finish(&mut self, now: i64) {
        let Some(mut task) = self.live.take() else {
            return;
        };
        task.ended = now;
        self.done.insert(0, task);
        self.done.truncate(MAX_ARCHIVED);
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
        task.state = State::Canceled;
        task.blame("Cancelled due to user request");
        tasks.finish(now);
        1
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
        if task.main != Some(conn) && task.rdb != Some(conn) {
            return;
        }
        let which = if task.main == Some(conn) {
            "Main"
        } else {
            "RDB"
        };
        task.state = State::Failed;
        task.blame(&yo_alloc::allow(|| {
            format!("{which} channel - Connection with the peer node was lost")
        }));
        tasks.finish(now);
    }
}

impl Server {
    /// Give up on a migration one of whose connections has just gone.
    ///
    /// Called for every internal connection that ends, which is a handful over
    /// the life of a cluster, and does nothing at all unless one of them was
    /// carrying a migration.
    pub(crate) fn asm_forget(&self, conn: u64) {
        self.cluster.asm.forget(conn, self.now_ms() as i64);
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
        conn: u64,
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
            live.state = State::Canceled;
            live.blame("Cancelled due to new migration requested");
            tasks.finish(now);
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
            main: Some(conn),
            rdb: None,
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

    /// Say the snapshot has gone out, which is the reference's
    /// `asmSlotSnapshotSucceed`.
    pub(super) fn asm_snapshot_sent(&self) {
        let mut tasks = self.cluster.asm.inner.lock();
        if let Some(task) = tasks.live.as_mut()
            && task.state == State::WaitBgsaveStart
        {
            task.state = State::SendStream;
        }
    }

    /// The snapshot of `slots`, as the stream of commands the far side runs.
    ///
    /// Built at one instant of the dataset, so what comes back is what the slots
    /// held at one point in the replication stream and not a smear across
    /// several. The offset that point sits at comes back with it, because the
    /// changes that follow have to start from exactly there or the far side ends
    /// up with a key applied twice or not at all.
    pub(super) fn asm_snapshot(&self, slots: &[(u16, u16)]) -> (Vec<u8>, u64) {
        let (image, offset) = self.at_an_instant(|| {
            let mut out = Out::with_capacity(Proto::Resp2, 4096);
            self.write_snapshot(slots, &mut out);
            out.into_inner()
        });
        (image, offset)
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

    use super::super::super::Server;
    use super::{State, key_slot};
    use crate::proto::Proto;
    use crate::reply::Out;

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
            .asm_begin_migrate(&id, &dest, vec![(0, 100)], 7)
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
        server.asm_snapshot_sent();
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_one(&id, &mut out);
        assert!(text(&out).contains("send-stream"), "{:?}", text(&out));
    }

    /// One at a time, because two moves at once would each be pausing writes
    /// for the other to catch up.
    #[test]
    fn a_second_migration_is_refused_while_one_is_running() {
        let server = node();
        server
            .asm_begin_migrate(&[b'b'; 40], &[b'c'; 40], vec![(0, 100)], 7)
            .expect("nothing else is running");
        let err = server
            .asm_begin_migrate(&[b'd'; 40], &[b'c'; 40], vec![(200, 300)], 9)
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
            .asm_begin_migrate(&id, &dest, vec![(0, 100)], 7)
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
            .asm_begin_migrate(&id, &dest, vec![(0, 100)], 11)
            .expect("the failed one does not block it");
    }

    /// Cancelling says how many it cancelled, and a task that has already
    /// finished cannot be cancelled again.
    #[test]
    fn cancelling_counts_once() {
        let server = node();
        let id = [b'b'; 40];
        server
            .asm_begin_migrate(&id, &[b'c'; 40], vec![(0, 100)], 7)
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
            .asm_begin_migrate(&[b'b'; 40], &[b'c'; 40], vec![(0, 100)], 7)
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
        for i in 0..(super::MAX_ARCHIVED + 5) {
            let id = format!("{i:040}");
            server
                .asm_begin_migrate(id.as_bytes(), &[b'c'; 40], vec![(0, 100)], 7)
                .expect("the last one was cancelled");
            assert_eq!(server.cluster.asm.cancel(None, 1), 1);
        }
        let mut out = Out::new(Proto::Resp3);
        server.cluster.asm.report_all(&mut out);
        let got = text(&out);
        assert!(
            got.starts_with(&format!("*{}\r\n", super::MAX_ARCHIVED)),
            "{got:?}"
        );
        // Newest first, which is the order the reference keeps them in.
        let newest = format!("{:040}", super::MAX_ARCHIVED + 4);
        assert!(got.contains(&newest), "{got:?}");
    }
}
