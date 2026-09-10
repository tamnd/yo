//! Taking slots off another node, which is `CLUSTER MIGRATION IMPORT`.
//!
//! The other half of `asm`. That module is what a node does when somebody asks
//! it to give slots up, and this is what a node does when it is the one asking.
//! Between them they are the whole of atomic slot migration, and the shape of
//! the protocol is easier to see from this end because this end drives it: every
//! message in a move is sent from here and the source only ever answers.
//!
//! # The two connections
//!
//! A move is two sockets to the same node, opened one after the other and both
//! authenticated as internal connections, because the commands they carry are
//! ones no user is allowed to send.
//!
//! The first is the main channel. It asks for the slots with
//! `CLUSTER SYNCSLOTS SYNC` and it is the one the changes made to those slots
//! while the move is running come down. The second is the snapshot channel. It
//! asks with `CLUSTER SYNCSLOTS RDBCHANNEL` and it carries the slots as they
//! were at the instant the source froze to read them.
//!
//! Two rather than one because they overlap. The source starts writing changes
//! to the main channel at the instant it starts building the snapshot, so the
//! two are being written at once and a single connection could only carry them
//! one after the other, which would mean either holding every write on the
//! source until the snapshot was out or losing the ones that happened while it
//! was. What this side does about that is the whole of the state machine: read
//! the snapshot and apply it, let the changes pile up unread behind it, and then
//! work through the pile.
//!
//! Nothing is lost by not reading the main channel during the snapshot, because
//! the source is writing into a socket and the kernel is holding it. What can go
//! wrong is the source's own send buffer growing while it waits, and that is
//! bounded by how long the snapshot takes, which is the same bound a full resync
//! lives with and is why a snapshot here is one slot range rather than a whole
//! keyspace.
//!
//! # One thread
//!
//! A move is a handful of blocking round trips, one very large read and then a
//! stream, which is the same shape being a replica has and the same reason it
//! gets a thread rather than a slot on the reactor. The thread ends when the
//! move does, which is either the slots changing hands or the task failing, and
//! it looks at the task before every step so that an operator cancelling lands
//! within one step rather than whenever the socket next does something.
//!
//! # What runs the commands
//!
//! The ordinary dispatcher, through a session of this module's own, which is the
//! same argument `follow` makes: a second path for applying writes would be a
//! second implementation of every command and the first bug in it would be a
//! slot that quietly disagrees with the node it came from. The session is past
//! the password, because the source is a node rather than a user, and past the
//! redirect that would otherwise send every one of these keys back to the node
//! still holding the slot.
//!
//! Two things it deliberately is not. The thread is not marked as applying from
//! a master, because this node is a master and the keys arriving are about to be
//! its own, deadlines and expiry and all. And the bytes are not put on this
//! node's replication stream as they arrived: what is propagated is what the
//! commands did, which is what a replica of this node needs and is the opposite
//! of what a replica does with its own master's bytes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::proto::{Limits, Proto};
use crate::reply::Out;
use crate::request::{Argv, Step};

use super::super::args::Args;
use super::super::follow::{self, Link};
use super::super::{Flow, Server, Session};
use super::asm::State;

/// How long a read waits before the loop looks around at other things.
///
/// The same tenth of a second the replication link uses and for the same two
/// reasons: it is how often a cancel is noticed and how close to the second an
/// acknowledgement lands.
const POLL: Duration = Duration::from_millis(100);

/// How long a read of the channel that is not the busy one waits.
///
/// Short, because during the snapshot both sockets are read in turn and a full
/// wait on the quiet one would halve the rate the busy one drains at. A read
/// with bytes already waiting answers immediately whatever this says, so it only
/// ever costs anything when there is nothing to read.
const GLANCE: Duration = Duration::from_millis(2);

/// How long the source gets to answer a `SYNC` or to start the snapshot.
///
/// A minute, because what happens on the far side of those messages is a freeze
/// and a walk of every key in the slots, on a node holding a dataset this one
/// has not seen.
const SYNC_TIMEOUT: Duration = Duration::from_secs(60);

/// How often this node says where it has got to.
const ACK_EVERY: Duration = Duration::from_secs(1);

/// How long to wait before asking again after the source said it was not ready.
const RETRY: Duration = Duration::from_millis(200);

/// How many times a `-NOTREADY` is taken for an answer before giving up.
///
/// The source says it when something else has the slots busy, which is a state
/// it gets out of on its own. Waiting forever would leave a task nobody can see
/// the end of, so this is a minute of asking.
const RETRIES: u32 = 300;

/// Everything the thread needs, worked out before it starts.
///
/// Copied rather than looked up as it goes, because the cluster table is behind
/// a lock the rest of the server wants and because a source that is renamed or
/// dropped halfway through a move is a move that should fail rather than one
/// that quietly starts talking to somebody else.
pub(super) struct Job {
    /// The forty characters both nodes call this move.
    pub(super) id: String,
    /// Where the slots are coming from.
    pub(super) host: String,
    /// The port the source answers commands on, which is not its bus port.
    pub(super) port: u16,
    /// What is moving.
    pub(super) slots: Vec<(u16, u16)>,
}

/// Start the thread that drives a move.
///
/// The task is already made and already answerable by `CLUSTER MIGRATION
/// STATUS`, so a machine that will not give out a thread fails it here rather
/// than leaving it at `none` waiting for something that is never coming.
pub(super) fn start(server: &Arc<Server>, job: Job) {
    let name = yo_alloc::allow(|| format!("yo-import-{}", &job.id[..8]));
    let id = job.id.as_bytes().to_vec();
    let shared = Arc::clone(server);
    let spawned = std::thread::Builder::new()
        .name(name)
        .spawn(move || drive(&shared, job));
    if spawned.is_err() {
        server.asm_import_failed(&id, "Failed to start the import thread");
    }
}

/// The whole of a move, from the first dial to the slots changing hands.
fn drive(server: &Arc<Server>, job: Job) {
    let id = job.id.as_bytes().to_vec();
    let mut into = Landing::new(server);
    let ended = run(server, &job, &mut into);
    super::super::forget_session(server, &mut into.session);
    if let Err(why) = ended {
        // Whatever landed before it went wrong. A move that gets part of the way
        // through the snapshot and then loses its connection has written keys of
        // slots this node does not own, and leaving them there would mean the
        // next attempt filled on top of half a dataset. Only the slots that are
        // still somebody else's are touched, so a failure after the handoff has
        // already gone through leaves the new keys alone.
        server.asm_import_trim(&job.slots);
        server.asm_import_failed(&id, &why);
    }
}

/// The body of [`drive`], with every failure coming back as the sentence to
/// blame the task with.
fn run(server: &Arc<Server>, job: &Job, into: &mut Landing) -> Told {
    let id = job.id.as_bytes();
    // Whatever this node is still holding for slots it does not own. A move that
    // is about to fill them would otherwise be filling them on top of leftovers
    // from a move that failed, and the reference clears them here for the same
    // reason.
    server.asm_import_trim(&job.slots);

    step(server, id, State::Connecting, None)?;
    let mut main = dial(server, job, "Main channel")?;
    step(server, id, State::AuthReply, None)?;

    step(server, id, State::SendHandshake, None)?;
    let me = server.cluster_id();
    ask(
        &mut main,
        "Main channel",
        &[b"CLUSTER", b"SYNCSLOTS", b"CONF", b"NODE-ID", me.as_bytes()],
        b"+OK",
        "CLUSTER SYNCSLOTS CONF",
    )?;
    step(server, id, State::HandshakeReply, None)?;

    let held = sync_command(id, &job.slots);
    let sync: Vec<&[u8]> = held.iter().map(Vec::as_slice).collect();
    // `-NOTREADY` is the source saying ask again rather than saying no, so it is
    // the one answer that goes round the loop instead of ending the move.
    let mut tries = 0;
    loop {
        step(server, id, State::SendSyncslots, None)?;
        main.write(&sync).map_err(|e| gone("Main channel", &e))?;
        step(server, id, State::SyncslotsReply, None)?;
        let said = main
            .line(SYNC_TIMEOUT)
            .map_err(|e| gone("Main channel", &e))?;
        if said == b"+RDBCHANNELSYNCSLOTS" {
            break;
        }
        if said.starts_with(b"-NOTREADY") && tries < RETRIES {
            tries += 1;
            std::thread::sleep(RETRY);
            continue;
        }
        return Err(blame(
            "Main channel - Error reply to CLUSTER SYNCSLOTS SYNC from the source",
            &said,
        ));
    }

    step(server, id, State::InitRdbchannel, Some(State::Connecting))?;
    let mut rdb = dial(server, job, "RDB channel")?;
    let at = State::InitRdbchannel;
    step(server, id, at, Some(State::AuthReply))?;
    step(server, id, at, Some(State::RdbchannelRequest))?;
    rdb.write(&[b"CLUSTER", b"SYNCSLOTS", b"RDBCHANNEL", id])
        .map_err(|e| gone("RDB channel", &e))?;
    step(server, id, at, Some(State::RdbchannelReply))?;
    let said = rdb
        .line(SYNC_TIMEOUT)
        .map_err(|e| gone("RDB channel", &e))?;
    if said != b"+SLOTSSNAPSHOT" {
        return Err(blame(
            "RDB channel - Failed to sync with the source node",
            &said,
        ));
    }

    step(
        server,
        id,
        State::AccumulateBuf,
        Some(State::RdbchannelTransfer),
    )?;
    snapshot(server, id, &mut rdb, &mut main, into)?;
    // The second connection has done its whole job and the source has already
    // let go of its end. Dropping it here rather than at the end of the move is
    // what keeps a socket per finished snapshot from piling up on a node that
    // takes slots several times.
    drop(rdb);

    step(server, id, State::ReadyToStream, Some(State::Completed))?;
    changes(server, id, &mut main, into)?;

    // Nothing more is coming and everything that was has landed, so the slots
    // are this node's. The source finds out from the bus, which is the same way
    // every other node does.
    step(server, id, State::Takeover, None)?;
    super::take_slots(server, &job.slots).map_err(|e| yo_alloc::allow(|| e.to_string()))?;
    server.asm_import_done(id);
    Ok(())
}

/// Read the snapshot and apply it, leaving the changes since piling up.
fn snapshot(
    server: &Arc<Server>,
    id: &[u8],
    rdb: &mut Link,
    main: &mut Link,
    into: &mut Landing,
) -> Told {
    let limits = Limits::default();
    let mut argv = Argv::new();
    loop {
        match argv.decode(rdb.held(), &limits) {
            Err(_) => return Err(unreadable("RDB channel")),
            Ok(Step::Command { consumed }) => {
                let what = control(&argv, rdb.held());
                if matches!(what, Control::Write) {
                    into.apply(server, &argv, rdb.held());
                }
                rdb.take(consumed);
                if matches!(what, Control::SnapshotEof) {
                    return Ok(());
                }
                continue;
            }
            Ok(Step::Incomplete) => {}
        }
        alive(server, id)?;
        rdb.fill(POLL).map_err(|e| gone("RDB channel", &e))?;
        // The other socket is read only to get the bytes out of the kernel and
        // into the buffer they will be worked through from. Nothing is decoded
        // and nothing is run: everything on this channel happened after the
        // instant the snapshot was taken and has to land after it.
        main.fill(GLANCE).map_err(|e| gone("Main channel", &e))?;
    }
}

/// Work through the pile and then follow the stream until the source says there
/// is no more coming.
fn changes(server: &Arc<Server>, id: &[u8], main: &mut Link, into: &mut Landing) -> Told {
    let limits = Limits::default();
    let mut argv = Argv::new();
    // Until the pile is empty this node is behind by a known amount and says so.
    // After that it is caught up, and the word changes because the source reads
    // it: `wait-stream-eof` is a destination saying stopping writes now would
    // cost almost nothing.
    let mut caught_up = false;
    let mut acked = Instant::now() - ACK_EVERY;
    let mut sent = u64::MAX;
    step(server, id, State::StreamingBuf, None)?;
    loop {
        let idle = match argv.decode(main.held(), &limits) {
            Err(_) => return Err(unreadable("Main channel")),
            Ok(Step::Command { consumed }) => {
                match control(&argv, main.held()) {
                    // The end of the move. Everything before it has landed,
                    // because it arrived on this connection behind everything
                    // else on it.
                    Control::StreamEof => {
                        main.take(consumed);
                        return Ok(());
                    }
                    // A hint about a slot, which is only worth anything in the
                    // snapshot and is not counted here because the source did
                    // not count it either.
                    Control::SnapshotEof | Control::SlotInfo => {
                        main.take(consumed);
                    }
                    Control::Write => {
                        into.apply(server, &argv, main.held());
                        main.take(consumed);
                        // Counted after it has landed, so the number that goes
                        // out is a count of commands this node has really run.
                        applied(server, id, consumed as u64)?;
                    }
                }
                false
            }
            Ok(Step::Incomplete) => {
                if !caught_up {
                    caught_up = true;
                    step(server, id, State::WaitStreamEof, None)?;
                }
                true
            }
        };
        // Once a second while there is a pile to work through, and again the
        // moment it empties out with a number the source has not been told yet.
        // The second of those is the one that matters: it is what ends the move,
        // and waiting out the rest of a second before sending it would put that
        // wait into every migration.
        if idle || acked.elapsed() >= ACK_EVERY {
            let now = applied(server, id, 0)?;
            if acked.elapsed() >= ACK_EVERY || now != sent {
                acked = Instant::now();
                sent = now;
                let word = if caught_up {
                    State::WaitStreamEof
                } else {
                    State::StreamingBuf
                };
                let at = yo_alloc::allow(|| now.to_string());
                main.write(&[
                    b"CLUSTER",
                    b"SYNCSLOTS",
                    b"ACK",
                    word.word().as_bytes(),
                    at.as_bytes(),
                ])
                .map_err(|e| gone("Main channel - Failed to send ACK", &e))?;
            }
        }
        if idle {
            main.fill(POLL).map_err(|e| gone("Main channel", &e))?;
        }
    }
}

// ------------------------------------------------------------ the small parts

/// What every step of a move answers with: nothing, or the sentence the task is
/// blamed with.
type Told = core::result::Result<(), String>;

/// What a command off either channel is, as far as this side is concerned.
#[derive(Clone, Copy)]
enum Control {
    /// The snapshot has all arrived.
    SnapshotEof,
    /// Nothing more is coming on the main channel.
    StreamEof,
    /// Anything else under `CLUSTER SYNCSLOTS`, which is a hint about a slot or
    /// a word this node has never heard of. Both are skipped, because a newer
    /// source saying something new is not a reason to fail a move.
    SlotInfo,
    /// A write to run.
    Write,
}

/// Read the command at the front of a buffer as one of the four.
fn control(argv: &Argv, buf: &[u8]) -> Control {
    let named = |at: usize, want: &[u8]| {
        argv.arg(buf, at)
            .is_some_and(|w| w.eq_ignore_ascii_case(want))
    };
    if argv.len() < 3 || !named(0, b"cluster") || !named(1, b"syncslots") {
        return Control::Write;
    }
    if named(2, b"snapshot-eof") {
        return Control::SnapshotEof;
    }
    if named(2, b"stream-eof") {
        return Control::StreamEof;
    }
    Control::SlotInfo
}

/// The session everything arriving is run through, and the reply nobody reads.
struct Landing {
    session: Session,
    out: Out,
}

impl Landing {
    fn new(server: &Arc<Server>) -> Landing {
        let mut session = Session::new(server.next_client());
        session.admit(true);
        session.serve_master(true);
        Landing {
            session,
            out: Out::new(Proto::Resp2),
        }
    }

    /// Run one command against this node.
    ///
    /// A command the freeze is holding is run again rather than dropped, for the
    /// same reason the replication link does it: a snapshot being built for
    /// somebody else is a pause and not a reason to lose a key.
    fn apply(&mut self, server: &Server, argv: &Argv, buf: &[u8]) {
        loop {
            self.out.clear();
            let args = Args::new(argv, buf);
            if super::super::execute(server, &mut self.session, args, &mut self.out) != Flow::Hold {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// `CLUSTER SYNCSLOTS SYNC <id> <start> <end> ...`, built once and sent as often
/// as the source says it is not ready.
fn sync_command(id: &[u8], slots: &[(u16, u16)]) -> Vec<Vec<u8>> {
    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(4 + slots.len() * 2);
    yo_alloc::allow(|| {
        parts.push(b"CLUSTER".to_vec());
        parts.push(b"SYNCSLOTS".to_vec());
        parts.push(b"SYNC".to_vec());
        parts.push(id.to_vec());
        for (from, to) in slots {
            parts.push(from.to_string().into_bytes());
            parts.push(to.to_string().into_bytes());
        }
    });
    parts
}

/// Count bytes against the task and answer with the total, or stop the move
/// because the task has gone.
fn applied(server: &Server, id: &[u8], n: u64) -> core::result::Result<u64, String> {
    server.asm_import_applied(id, n).ok_or_else(cancelled)
}

/// Check the task is still the live one without counting anything.
fn alive(server: &Server, id: &[u8]) -> Told {
    applied(server, id, 0).map(|_| ())
}

/// Move the task along, and turn a task that has gone into the sentence saying
/// so.
fn step(server: &Server, id: &[u8], state: State, rdb: Option<State>) -> Told {
    if server.asm_import_at(id, state, rdb) {
        Ok(())
    } else {
        Err(cancelled())
    }
}

/// Open one of the two connections and authenticate it as an internal one.
fn dial(server: &Server, job: &Job, which: &str) -> core::result::Result<Link, String> {
    let mut wire = follow::connect(&job.host, job.port).map_err(|e| {
        yo_alloc::allow(|| format!("{which} - Failed to connect to source node: {e}"))
    })?;
    let secret = server.cluster_secret();
    ask(
        &mut wire,
        which,
        &[b"AUTH", b"internal connection", secret.as_bytes()],
        b"+OK",
        "AUTH",
    )?;
    Ok(wire)
}

/// Send something the answer to which has to be exactly one word.
fn ask(wire: &mut Link, which: &str, parts: &[&[u8]], want: &[u8], what: &str) -> Told {
    let said = wire.command(parts).map_err(|e| gone(which, &e))?;
    if said == want {
        return Ok(());
    }
    Err(blame(
        &yo_alloc::allow(|| format!("{which} - Error reply to {what} from the source")),
        &said,
    ))
}

/// The sentence for a socket that stopped working.
fn gone(which: &str, e: &std::io::Error) -> String {
    yo_alloc::allow(|| format!("{which} - Failed to sync with source node: {e}"))
}

/// The sentence for a source sending something that is not a command at all,
/// which is a source talking a protocol this one does not know.
fn unreadable(which: &str) -> String {
    yo_alloc::allow(|| format!("{which} - the source sent something that is not a command"))
}

/// The sentence for a source that answered something other than what was
/// wanted, with the leading minus taken off so the error does not read as two.
fn blame(what: &str, said: &[u8]) -> String {
    yo_alloc::allow(|| {
        let said = String::from_utf8_lossy(said);
        let said = said.strip_prefix('-').unwrap_or(&said);
        format!("{what}: {said}")
    })
}

/// The sentence for a task somebody else ended while this thread was in the
/// middle of it. Nothing records it, because the task that would have recorded
/// it is already on the finished list with a better reason on it.
fn cancelled() -> String {
    String::from("the task was cancelled")
}
