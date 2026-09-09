//! Being a master: the replication identity, the stream and the link.
//!
//! A replica is a second copy of this server's keyspace that keeps up by being
//! told every change as it happens. Getting there is two halves. First it needs
//! the dataset as it stands, which is a snapshot sent down the socket, and then
//! it needs everything that happens after that snapshot, which is a stream of
//! commands that never ends. The hard part is the seam between the two, because
//! a change that lands in neither is lost forever and a change that lands in
//! both is applied twice, and applying `INCR` twice is not the same as applying
//! it once.
//!
//! # The handshake
//!
//! A replica opens an ordinary connection and sends `PING`, then
//! `REPLCONF listening-port <port>`, then `REPLCONF capa ...`, and then `PSYNC
//! <replid> <offset>`. Everything up to the `PSYNC` is a normal command with a
//! normal reply. The `PSYNC` is where the connection stops being a client: the
//! master answers `+FULLRESYNC <replid> <offset>`, then the snapshot as a bulk
//! string with no trailing newline after it, and from then on writes only the
//! command stream and never replies to anything the replica sends. A replica
//! sends `REPLCONF ACK <offset>` about once a second forever, and a master that
//! answers one of those with `+OK` has put a reply into the middle of a stream
//! the replica is parsing as commands, which a real replica reports as
//! `Protocol error (Master using the inline protocol. Desync?)` and then drops
//! the link. So the silence is not an optimisation, it is the protocol.
//!
//! # The identity and the offset
//!
//! The replication id is forty hex characters naming this server's history, and
//! the offset is how many bytes of stream have gone out under that id. Together
//! they are a position in a history, which is what lets a replica that lost the
//! link for a moment ask to carry on rather than start again: it sends back the
//! id and offset it had, and if the id is still ours and the offset is still in
//! the backlog the master answers `+CONTINUE` and replays the missing bytes.
//!
//! There is a second id for the same reason a chain needs one. When a replica is
//! promoted it keeps the old master's id as `replid2` and takes a new one of its
//! own, so the replicas that were following the old master can be handed over
//! without all of them resyncing from nothing.
//!
//! # The snapshot has to be one instant
//!
//! `SAVE` here walks the databases one stripe at a time and does not stop the
//! server, so the file it writes is not one instant of the keyspace: a write to
//! database nine while database two is being walked is in the file and a write
//! to database two after it has been walked is not. For a file on disk that is a
//! fair trade, because the file is read by itself and nothing is going to be
//! replayed on top of it.
//!
//! It is not a fair trade here. The replica loads the snapshot and then applies
//! the stream from the offset the snapshot was stamped with, so every byte of
//! the keyspace has to be either in the snapshot or after that offset, and never
//! both and never neither. A real server gets that for free by forking, which
//! hands the child an instant of the whole address space and costs the parent
//! nothing but the page faults afterwards. There is no fork here, so a full
//! resync takes every stripe of every database at once, stamps the offset,
//! builds the image and lets go. The server stops for as long as that takes,
//! which is a real cost on a large dataset and is registered as a divergence.
//! The alternative is a replica that is quietly wrong, which is worse than a
//! replica that took a pause to be right.
//!
//! # What goes on the stream
//!
//! Every command that changed something, in the form the replica has to be given
//! rather than the form the client sent. Most commands are the same both ways.
//! The ones that are not are the ones whose result depends on something the
//! replica has not got: the clock, for anything that sets a deadline, and the
//! server's own random state, for `SPOP`. `EXPIRE k 50` becomes `PEXPIREAT k
//! <absolute>`, `SET k v EX 100` becomes `SET k v PXAT <absolute>`, `SPOP s`
//! becomes `SREM s <the member that actually went>`, and `XADD s *` becomes an
//! `XADD` naming the id that was actually made. Without the rewrite the two
//! copies drift apart the moment either of them is asked a question.
//!
//! The rewrite is pushed by the command that knows, through a thread local, the
//! same way this crate already hands keyspace events up from underneath. A
//! command that pushes nothing is sent as it arrived, which is the common case
//! and costs nothing to decide.
//!
//! # What it costs a server with no replica
//!
//! One relaxed load of a count that is zero, per write command. The same shape
//! `MONITOR` and the pub/sub registry use, and for the same reason: nearly every
//! server in the world is running on its own.

use std::sync::Arc;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize};

use core::cell::{Cell, RefCell};

use yo_common::lock::Lock;
use yo_common::{Code, Error, Result};

use crate::reply::Out;

use super::args::{self, Args};
use super::clients::{self, Client};
use super::pubsub::Envelope;
use super::{Server, Session};

/// How many characters name a history, which is Redis's forty hex digits.
pub(super) const ID_LEN: usize = 40;

/// The id a server that has never been anybody's replica reports as its second,
/// which Redis writes as forty zeroes rather than leaving out.
const NO_ID: &[u8; ID_LEN] = b"0000000000000000000000000000000000000000";

/// How much stream is kept for a replica that dropped the link, in bytes.
///
/// Redis's default, and the number `repl-backlog-size` reads back as. A replica
/// that reconnects inside this many bytes carries on where it was and one that
/// falls further behind starts again from a snapshot.
pub(super) const BACKLOG_BYTES: usize = 1024 * 1024;

/// One connection that has stopped being a client and is now a copy of us.
pub(super) struct Replica {
    /// The row every other part of the server knows the connection by, so that
    /// `CLIENT LIST` and `CLIENT KILL` see a replica the way they see anything
    /// else.
    row: Arc<Client>,
    /// The port the replica says it is listening on, which is what `INFO`
    /// reports rather than the port its outgoing connection came from. A replica
    /// that has not said is reported as zero, which is what Redis does too.
    port: AtomicU64,
    /// The last offset it said it had, and when it said so.
    ///
    /// `WAIT` counts the replicas whose acknowledged offset has caught up, and
    /// `INFO` turns the time into a lag in seconds.
    ack: AtomicU64,
    ack_ms: AtomicU64,
    /// Whether the snapshot has gone and the stream has started.
    online: AtomicBool,
}

impl Replica {
    /// The address the replica is reachable at, which is its address and the
    /// port it told us rather than the port it dialled out from.
    fn address(&self) -> (String, u64) {
        let text = self.row.text.lock();
        let peer = text.peer.clone();
        drop(text);
        let host = match peer.iter().rposition(|&b| b == b':') {
            Some(at) => peer[..at].to_vec(),
            None => peer.clone(),
        };
        (
            String::from_utf8_lossy(&host).into_owned(),
            self.port.load(Relaxed),
        )
    }
}

/// Everything about being a master, all of it idle on a server with no replica.
pub(crate) struct Replication {
    /// The forty characters naming the history this server is writing.
    id: Lock<[u8; ID_LEN]>,
    /// The history it was writing before, for a server that was promoted.
    id2: Lock<[u8; ID_LEN]>,
    /// How many bytes of stream have been produced under [`Replication::id`].
    offset: AtomicU64,
    /// The offset the second id runs up to, or minus one when there is no
    /// second id, which is what Redis reports on a server that was never a
    /// replica.
    second: AtomicI64,
    /// What has gone out lately, so a replica that blinked can catch up without
    /// a whole snapshot.
    backlog: Lock<Backlog>,
    /// The connections being fed.
    rows: Lock<Vec<Arc<Replica>>>,
    /// How many there are, so a write command can ask without taking the lock.
    live: AtomicUsize,
    /// The port a connection said it was listening on, before it asked to
    /// become a replica.
    ///
    /// `REPLCONF listening-port` arrives two commands before `PSYNC`, so there
    /// is no replica to hang it on yet and it has to be kept somewhere until
    /// there is. Here rather than on the connection row, because a row is paid
    /// for by every connection on the server and this is paid for only by the
    /// few that are about to become replicas. Oldest first out at a small cap,
    /// so a client that sends the one and never the other cannot grow it.
    ports: Lock<Vec<(u64, u64)>>,
    /// Whether a full resync is building an image right now.
    ///
    /// A write that arrives while this is set is held and run again afterwards,
    /// which is what makes the image one instant. Read on the command path by
    /// every write, so it is a relaxed load of a bool that is nearly always
    /// false, next to the one `CLIENT PAUSE` already costs.
    frozen: AtomicBool,
    /// One resync builds at a time.
    ///
    /// Two threads freezing the server at once would each wait for the other's
    /// writes to drain and neither would be able to, so the second one waits
    /// here instead and then finds a backlog it can probably be caught up from.
    building: Lock<()>,
    /// Which database the stream is on, so `SELECT` goes out only when it has
    /// to. Minus one before anything has been written, which is why the first
    /// command on any database is preceded by a `SELECT` even for database
    /// zero.
    on_db: AtomicI64,
}

impl Default for Replication {
    fn default() -> Replication {
        Replication {
            id: Lock::new(make_id()),
            id2: Lock::new(*NO_ID),
            offset: AtomicU64::new(0),
            second: AtomicI64::new(-1),
            backlog: Lock::new(Backlog::default()),
            rows: Lock::new(Vec::new()),
            live: AtomicUsize::new(0),
            ports: Lock::new(Vec::new()),
            frozen: AtomicBool::new(false),
            building: Lock::new(()),
            on_db: AtomicI64::new(-1),
        }
    }
}

/// The last stretch of stream, kept so a reconnect does not cost a snapshot.
///
/// A ring of a fixed size with the offset of its first byte beside it. `histlen`
/// is how much of the ring is real, which is less than its size only until it
/// has filled once.
#[derive(Default)]
struct Backlog {
    ring: Vec<u8>,
    /// Where the next byte goes.
    at: usize,
    /// How many bytes of the ring are real.
    filled: usize,
    /// The stream offset of the oldest real byte.
    first: u64,
}

impl Backlog {
    /// Take bytes in, dropping whatever falls off the back.
    fn push(&mut self, bytes: &[u8], upto: u64) {
        if self.ring.is_empty() {
            self.ring = vec![0; BACKLOG_BYTES];
            self.first = upto - bytes.len() as u64;
        }
        for &b in bytes {
            self.ring[self.at] = b;
            self.at = (self.at + 1) % BACKLOG_BYTES;
            if self.filled < BACKLOG_BYTES {
                self.filled += 1;
            }
        }
        self.first = upto - self.filled as u64;
    }

    /// Everything from `from` onwards, or `None` if that much history has gone.
    fn since(&self, from: u64, upto: u64) -> Option<Vec<u8>> {
        if self.filled == 0 || from < self.first || from > upto {
            return None;
        }
        let skip = (from - self.first) as usize;
        let want = self.filled - skip;
        let start = (self.at + BACKLOG_BYTES - self.filled + skip) % BACKLOG_BYTES;
        let mut out = Vec::with_capacity(want);
        for i in 0..want {
            out.push(self.ring[(start + i) % BACKLOG_BYTES]);
        }
        Some(out)
    }
}

#[cfg(test)]
impl Server {
    /// Say there is a replica, without one.
    ///
    /// What a test wants to look at is the byte stream and not the socket it
    /// would have gone down, and [`emit`] writes the stream into the backlog
    /// before it looks for anybody to post it to. So a server told this writes
    /// everything a real master would write and posts none of it, which is the
    /// whole of what these tests need and needs no connection at all.
    pub(super) fn pretend_replica(&self) {
        self.repl.live.store(1, Relaxed);
    }

    /// The stream from `from` onwards, as text, and where it has got to.
    ///
    /// Text because every rewrite these tests are about is text, and reading
    /// `*3\r\n$3\r\nSET\r\n` in a failure message beats reading a byte array.
    pub(super) fn stream_since(&self, from: u64) -> (String, u64) {
        let upto = self.repl.offset.load(Acquire);
        let backlog = self.repl.backlog.lock();
        let bytes = backlog.since(from, upto).unwrap_or_default();
        (String::from_utf8_lossy(&bytes).into_owned(), upto)
    }
}

/// Forty hex characters from the operating system's generator.
///
/// From the system and not from the engine's own seeded one, because two servers
/// started from the same image at the same moment must not be able to claim the
/// same history. That is the same reasoning `ACL GENPASS` is built on.
fn make_id() -> [u8; ID_LEN] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut raw = [0u8; ID_LEN / 2];
    yo_common::entropy::fill(&mut raw);
    let mut id = [0u8; ID_LEN];
    for (i, byte) in raw.iter().enumerate() {
        id[i * 2] = HEX[usize::from(byte >> 4)];
        id[i * 2 + 1] = HEX[usize::from(byte & 15)];
    }
    id
}

// ------------------------------------------------------------- the override

thread_local! {
    /// What the running command wants sent instead of itself, if anything.
    ///
    /// A command that leaves this empty is propagated as it arrived. One that
    /// puts something here is propagated as whatever it put, however many
    /// commands that is, and a command that puts an empty list here is
    /// propagated as nothing at all. See the module header for why a thread
    /// local rather than a return value.
    static INSTEAD: RefCell<Option<Vec<Vec<Vec<u8>>>>> = const { RefCell::new(None) };
}

thread_local! {
    /// Whether anything this thread does has somewhere to be copied to.
    ///
    /// A flag rather than a question put to the server, because the call sites
    /// that ask are inside the command bodies, a long way from anything holding
    /// a `Server`, and threading one down to them for a question that is nearly
    /// always no would be a worse trade than a thread local read. The same trade
    /// the keyspace events make, next door.
    ///
    /// It says whether there is a replica and not whether a write is running,
    /// which are two different questions and this is the wider one on purpose. A
    /// read takes keys away, because a lookup that finds one past its deadline
    /// removes it there and then, and `XGROUP CREATE` changes a stream while
    /// carrying no write flag, since the flag is on the subcommand and there is
    /// no subcommand table yet. Both have something to send and neither is a
    /// write by the funnel's reckoning, so the funnel decides whether the
    /// verbatim command may be sent and this decides whether anybody is
    /// listening at all.
    static ARMED: Cell<bool> = const { Cell::new(false) };

    /// The removals nobody asked for, waiting for a lock that is not held.
    static REAPED: RefCell<Vec<Vec<Vec<u8>>>> = const { RefCell::new(Vec::new()) };
}

/// Say whether what runs next is going to be copied, answering what the last
/// answer was.
///
/// Set in front of every write on a server with a replica and put back
/// afterwards, so that a command run by a script or by `EXEC` leaves the flag
/// the way it found it.
pub(super) fn arm(on: bool) -> bool {
    ARMED.replace(on)
}

/// Whether a rewrite would be heard, which is what keeps every call site that
/// would build one off the path of a server with no replica.
#[must_use]
pub(crate) fn armed() -> bool {
    ARMED.get()
}

/// Send this instead of the command that is running.
///
/// Called by a command whose own arguments do not say what happened, so more
/// than once for a command with more than one effect. The first call replaces
/// the command and the rest are added after it.
pub(crate) fn instead(parts: Vec<Vec<u8>>) {
    INSTEAD.with(|cell| {
        let mut held = cell.borrow_mut();
        held.get_or_insert_with(Vec::new).push(parts);
    });
}

/// The same thing said in pieces, which is how nearly every call site has it.
///
/// A rewrite is a command name and a key and usually a number, none of which
/// arrive owned, so this is the shape that keeps the copying in one place rather
/// than a `to_vec` on every argument of every site.
pub(crate) fn rewrite(parts: &[&[u8]]) {
    instead(parts.iter().map(|part| part.to_vec()).collect());
}

/// Send nothing at all for the command that is running.
///
/// For a write command that turned out to change nothing and whose verbatim form
/// would be wrong rather than merely wasteful, which is any of the ones that are
/// rewritten: a `SPOP` on a missing key has no `SREM` to send and must not send
/// the `SPOP`.
pub(crate) fn nothing() {
    INSTEAD.with(|cell| {
        let mut held = cell.borrow_mut();
        held.get_or_insert_with(Vec::new);
    });
}

/// Take whatever the command left, clearing it for the next one.
fn taken() -> Option<Vec<Vec<Vec<u8>>>> {
    INSTEAD.with(|cell| cell.borrow_mut().take())
}

// --------------------------------------------------------------- the server

impl Server {
    /// Whether anybody is being fed, which is the whole cost of this file on a
    /// server that is on its own.
    #[must_use]
    pub(crate) fn replicated(&self) -> bool {
        self.repl.live.load(Relaxed) != 0
    }

    /// The forty characters naming this server's history.
    pub(crate) fn repl_id(&self) -> [u8; ID_LEN] {
        *self.repl.id.lock()
    }

    /// How many bytes of stream have gone out.
    #[must_use]
    pub(crate) fn repl_offset(&self) -> u64 {
        self.repl.offset.load(Acquire)
    }

    /// A handle to every replica, copied out so the bytes can be posted with the
    /// lock let go of.
    ///
    /// The same trade `CLIENT LIST` and the monitor feed make, for the same
    /// reason: a replica that drops while a command is being rendered leaves a
    /// row that is still readable, and the bytes land in a mailbox that is about
    /// to be told the slot has moved on.
    fn replica_rows(&self) -> Vec<Arc<Replica>> {
        let rows = self.repl.rows.lock();
        yo_alloc::allow(|| rows.clone())
    }

    /// Remember the port a connection says it is listening on.
    ///
    /// See [`Replication::ports`] for why it cannot go straight on a replica.
    fn note_replica_port(&self, id: u64, port: u64) {
        const KEEP: usize = 64;
        let mut ports = self.repl.ports.lock();
        yo_alloc::allow(|| {
            if let Some(row) = ports.iter_mut().find(|(who, _)| *who == id) {
                row.1 = port;
                return;
            }
            if ports.len() >= KEEP {
                ports.remove(0);
            }
            ports.push((id, port));
        });
    }

    /// Take the port back out, for a connection that has got as far as `PSYNC`.
    fn take_replica_port(&self, id: u64) -> u64 {
        let mut ports = self.repl.ports.lock();
        match ports.iter().position(|(who, _)| *who == id) {
            Some(at) => ports.remove(at).1,
            None => 0,
        }
    }

    /// Take a connection on as a replica.
    fn take_replica(&self, row: &Arc<Client>) -> Arc<Replica> {
        let mut rows = self.repl.rows.lock();
        let held = yo_alloc::allow(|| {
            let held = Arc::new(Replica {
                row: Arc::clone(row),
                port: AtomicU64::new(self.take_replica_port(row.id)),
                ack: AtomicU64::new(0),
                ack_ms: AtomicU64::new(self.clock.now_ms()),
                online: AtomicBool::new(true),
            });
            rows.push(Arc::clone(&held));
            held
        });
        self.repl.live.store(rows.len(), Release);
        row.set_flag(clients::REPLICA, true);
        self.note_here(row.thread.load(Relaxed), 1);
        held
    }

    /// Let one go, which is the connection closing or being killed.
    pub(crate) fn drop_replica(&self, id: u64) {
        let mut rows = self.repl.rows.lock();
        let Some(at) = rows.iter().position(|r| r.row.id == id) else {
            return;
        };
        let gone = rows.remove(at);
        self.repl.live.store(rows.len(), Release);
        drop(rows);
        gone.row.set_flag(clients::REPLICA, false);
        self.note_here(gone.row.thread.load(Relaxed), -1);
    }

    /// Whether a full resync is holding the keyspace still.
    ///
    /// One relaxed load per write command on a server nobody is syncing from,
    /// which is the same shape and the same cost as the pause check it sits
    /// beside.
    #[must_use]
    pub(crate) fn frozen(&self) -> bool {
        self.repl.frozen.load(Relaxed)
    }

    /// The whole dataset as one image, and the offset it is an image as of.
    ///
    /// Everything between the two has to be nothing at all, which is what the
    /// freeze is for. Writes are held from before the barrier until after the
    /// image is finished, so a change is either inside the image or after the
    /// offset and never both. See the module header for why there is no fork to
    /// get this for free.
    pub(crate) fn snapshot_at_an_instant(&self) -> (Vec<u8>, u64) {
        let building = self.repl.building.lock();
        self.repl.frozen.store(true, Release);
        // A write already running holds the stripe it is writing to, so taking
        // every stripe once and letting it go again is a barrier: once it is
        // through, no write is in flight and the freeze above stops any more
        // starting. One pass is enough, in whatever order, because nothing being
        // waited for can start again behind it.
        for db in &self.dbs {
            for stripe in 0..db.width() {
                drop(db.hold_stripe(stripe));
            }
        }
        let offset = self.repl.offset.load(Acquire);
        let (image, _skipped) = super::persist::build(self);
        self.repl.frozen.store(false, Release);
        drop(building);
        (image, offset)
    }

    /// The replica row for a connection, if it is one.
    fn replica_of(&self, id: u64) -> Option<Arc<Replica>> {
        let rows = self.repl.rows.lock();
        rows.iter().find(|r| r.row.id == id).map(Arc::clone)
    }
}

// ---------------------------------------------------------------- the feed

/// One command as a RESP array, which is the only shape the stream has.
fn render(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + parts.iter().map(|p| p.len() + 16).sum::<usize>());
    out.extend_from_slice(b"*");
    out.extend_from_slice(parts.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    for part in parts {
        out.extend_from_slice(b"$");
        out.extend_from_slice(part.len().to_string().as_bytes());
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(part);
        out.extend_from_slice(b"\r\n");
    }
    out
}

/// Put bytes on the stream: into the backlog, onto the offset, out to everybody.
///
/// The offset moves under the backlog lock so that the backlog and the number
/// never disagree. A replica attaching between the two would otherwise be given
/// an offset the backlog cannot honour.
fn emit(server: &Server, bytes: Vec<u8>) {
    let shared = {
        let mut backlog = server.repl.backlog.lock();
        let upto = server.repl.offset.load(Relaxed) + bytes.len() as u64;
        yo_alloc::allow(|| backlog.push(&bytes, upto));
        server.repl.offset.store(upto, Release);
        yo_alloc::allow(|| Arc::new(bytes))
    };
    for held in server.replica_rows() {
        if !held.online.load(Relaxed) {
            continue;
        }
        server.post(
            held.row.thread.load(Relaxed),
            Envelope::raw(
                held.row.conn.load(Relaxed),
                held.row.id,
                Arc::clone(&shared),
            ),
        );
    }
}

/// Put one command on the stream, with a `SELECT` in front of it if the stream
/// is not on the right database.
fn send(server: &Server, db: usize, parts: &[&[u8]]) {
    let mut bytes = Vec::new();
    if server.repl.on_db.load(Relaxed) != db as i64 {
        let n = db.to_string();
        bytes = render(&[b"SELECT", n.as_bytes()]);
        server.repl.on_db.store(db as i64, Relaxed);
    }
    bytes.extend_from_slice(&render(parts));
    emit(server, bytes);
}

/// Report a command to every replica, in the form it has to be given.
///
/// The caller has already checked [`Server::replicated`] and that the command
/// did not fail, so what is decided here is only the shape.
///
/// `verbatim` is whether the command as it arrived is a fair thing to send when
/// the body said nothing. It is the write flag, and it is false for a read,
/// which sends nothing, and for a container like `XGROUP` whose flags are on its
/// subcommands, which sends what its body pushed and nothing otherwise.
pub(super) fn feed(server: &Server, db: usize, args: Args<'_>, verbatim: bool) {
    let instead = taken();
    yo_alloc::allow(|| match instead {
        None if !verbatim => {}
        None => {
            let parts: Vec<&[u8]> = (0..args.len()).map(|i| args.get(i)).collect();
            send(server, db, &parts);
        }
        Some(each) => {
            for one in &each {
                let parts: Vec<&[u8]> = one.iter().map(Vec::as_slice).collect();
                send(server, db, &parts);
            }
        }
    });
}

/// Report what a command that nobody is running left behind.
///
/// A blocked client is answered by the thread that woke it, a long way outside
/// the funnel and with no arguments to fall back on, so unlike [`feed`] there is
/// no verbatim form here: whatever the answering left is all there is, and an
/// answering that left nothing did nothing.
pub(super) fn served(server: &Server, db: usize) {
    let Some(each) = taken() else {
        return;
    };
    yo_alloc::allow(|| {
        for one in &each {
            let parts: Vec<&[u8]> = one.iter().map(Vec::as_slice).collect();
            send(server, db, &parts);
        }
    });
}

/// Throw away whatever the last command left, for a command that is not
/// propagated at all.
///
/// A read that pushed a rewrite is a bug, but a rewrite left behind by a command
/// that was refused after it pushed one is not, and it must not be handed to the
/// next command that runs on this thread.
pub(super) fn forget() {
    let _ = taken();
}

/// Note a key or a field the storage layer took away on its own.
///
/// A replica never expires and never evicts. It cannot: the two decisions are
/// made from a clock reading and a memory figure that are the master's and not
/// its own, and a replica that made them itself would answer a read differently
/// from the master for as long as the two disagreed. So it holds a key past its
/// deadline until it is told, and being told is this. Redis sends the same thing
/// for the same reason, and it is why a replica's `dbsize` can be ahead of the
/// master's for a moment and never behind it.
///
/// Collected rather than sent, because the point at which a key goes is inside a
/// stripe lock and the send takes a different one.
pub(crate) fn reaped(parts: &[&[u8]]) {
    if !ARMED.get() {
        return;
    }
    REAPED.with_borrow_mut(|list| {
        yo_alloc::allow(|| list.push(parts.iter().map(|part| part.to_vec()).collect()));
    });
}

/// Send what [`reaped`] collected, ahead of whatever the command itself did.
///
/// Ahead, because that is the order it happened in and the order matters: a
/// `SET k v` that found an expired `k` on the way in has to reach a replica as
/// the deletion and then the write, or a replica that has kept a `k` of the
/// wrong type refuses the write it is sent.
pub(super) fn swept(server: &Server, db: usize) {
    if REAPED.with_borrow(Vec::is_empty) {
        return;
    }
    let each = REAPED.with_borrow_mut(core::mem::take);
    yo_alloc::allow(|| {
        for one in &each {
            let parts: Vec<&[u8]> = one.iter().map(Vec::as_slice).collect();
            send(server, db, &parts);
        }
    });
}

// ------------------------------------------------------------- the commands

/// `REPLCONF`, which is how a replica tells a master about itself.
///
/// Everything here is a pair, and a master answers `OK` to a pair it does not
/// know rather than refusing it, because the whole point of the command is that
/// a newer replica can tell an older master things it has never heard of. The
/// one exception is `ACK`, which is not a question and is answered with silence:
/// by the time a replica sends one the connection is a stream, and a reply on it
/// is a protocol error at the other end.
pub(super) fn replconf(
    server: &Server,
    session: &Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    if args.len() < 3 || args.len().is_multiple_of(2) {
        return Err(args::wrong_arity("replconf"));
    }
    let mut at = 1;
    while at < args.len() {
        let name = args.get(at).to_ascii_lowercase();
        match name.as_slice() {
            b"listening-port" => {
                let port = args.int(at + 1)?;
                server.note_replica_port(session.row().id, port.max(0) as u64);
            }
            b"ack" => {
                // Not answered, ever. See the doc comment.
                let ack = args.int(at + 1).unwrap_or(0);
                if let Some(held) = server.replica_of(session.row().id) {
                    held.ack.store(ack.max(0) as u64, Relaxed);
                    held.ack_ms.store(server.clock.now_ms(), Relaxed);
                }
                return Ok(());
            }
            b"getack" => {
                // A master sends this and a replica answers it. Reaching it here
                // means somebody sent it to a master, which Redis refuses.
                return Err(Error::new(
                    Code::Invalid,
                    "REPLCONF GETACK is only supported by a replica",
                ));
            }
            _ => {}
        }
        at += 2;
    }
    out.ok();
    Ok(())
}

/// `PSYNC <replid> <offset>`, which is where a connection stops being a client.
///
/// Answers `+CONTINUE` and the missing bytes when the replica is asking to carry
/// on a history we are still writing and still hold enough of, and a full resync
/// otherwise. `SYNC` is the same thing without the negotiation, from before
/// Redis had one, and is a snapshot with no header in front of it.
pub(super) fn psync(
    server: &Server,
    session: &mut Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let partial = if args.name().eq_ignore_ascii_case(b"sync") {
        if args.len() != 1 {
            return Err(args::wrong_arity("sync"));
        }
        None
    } else {
        if args.len() != 3 {
            return Err(args::wrong_arity("psync"));
        }
        let asked = args.get(1);
        let from = args.int(2)?;
        (asked != b"?" && from >= 0).then(|| (asked.to_vec(), from as u64))
    };
    if session.running() {
        return Err(Error::new(
            Code::Invalid,
            "PSYNC isn't allowed for DENY BLOCKING client",
        ));
    }
    // A partial resync first, since it is the cheap answer and the whole reason
    // the replica bothered to remember where it was.
    if let Some((asked, from)) = partial
        && let Some(bytes) = catch_up(server, &asked, from)
    {
        {
            let id = server.repl_id();
            out.raw(b"+CONTINUE ");
            out.raw(&id);
            out.raw(b"\r\n");
            out.raw(&bytes);
            attach(server, session);
            return Ok(());
        }
    }
    full(server, session, out)
}

/// The bytes a reconnecting replica missed, or `None` if it has to start again.
///
/// A replica asking about the history we are writing is caught up from the
/// backlog. One asking about the history we were writing before we were promoted
/// is caught up too, but only up to the point where the histories part, which is
/// what the second offset records.
fn catch_up(server: &Server, asked: &[u8], from: u64) -> Option<Vec<u8>> {
    let ours = server.repl_id();
    let theirs = *server.repl.id2.lock();
    let second = server.repl.second.load(Relaxed);
    let matches = asked == ours
        || (asked == theirs && second >= 0 && from <= u64::try_from(second).unwrap_or(0));
    if !matches {
        return None;
    }
    let upto = server.repl.offset.load(Acquire);
    let backlog = server.repl.backlog.lock();
    yo_alloc::allow(|| backlog.since(from, upto))
}

/// A full resync: the header, then the whole keyspace as one image.
///
/// The image and the offset are taken with every stripe of every database held,
/// so what the replica loads and what it is then told about are the two halves
/// of one instant and neither overlaps the other. See the module header.
fn full(server: &Server, session: &mut Session, out: &mut Out) -> Result<()> {
    let (image, offset) = server.snapshot_at_an_instant();
    let id = server.repl_id();
    out.raw(b"+FULLRESYNC ");
    out.raw(&id);
    out.raw(b" ");
    out.raw(offset.to_string().as_bytes());
    out.raw(b"\r\n");
    // A bulk string with no newline after it, which is the one place in the
    // protocol where that is so. The replica reads the length and then exactly
    // that many bytes, and anything after them is the stream.
    out.raw(b"$");
    out.raw(image.len().to_string().as_bytes());
    out.raw(b"\r\n");
    out.raw(&image);
    attach(server, session);
    Ok(())
}

/// Turn the connection into a replica, once the answer has been written.
///
/// Everything it was holding as a client goes first: a transaction it had open,
/// the keys it was watching and any subscription. All three are promises of a
/// reply, and a connection that has become a stream has no way left to keep one.
fn attach(server: &Server, session: &mut Session) {
    super::forget_session(server, session);
    server.take_replica(session.row());
}

/// How many replicas have acknowledged everything written so far.
///
/// What `WAIT` answers with. It is the count that is already there rather than
/// one waited for, which is the right answer when it is already enough and is a
/// divergence when it is not. Waiting here means holding a connection until an
/// acknowledgement arrives, which needs the waiter list to be woken by something
/// that is not a key changing, and that is the same wakeup D-115 is about.
pub(super) fn caught_up(server: &Server) -> usize {
    let upto = server.repl_offset();
    server
        .replica_rows()
        .iter()
        .filter(|r| r.online.load(Relaxed) && r.ack.load(Relaxed) >= upto)
        .count()
}

// ------------------------------------------------------------------- report

/// What `ROLE` answers on a master.
///
/// The word, then the offset as an integer, then one row per replica. Only the
/// replicas that are online are listed, because a replica still being sent its
/// snapshot has no offset of its own to report and Redis leaves it out for the
/// same reason. Inside a row every field is a bulk string, including the port
/// and the offset, which is a shape nobody would pick today and is the shape
/// every client library already parses.
pub(super) fn role(server: &Server, out: &mut Out) {
    let rows = server.replica_rows();
    out.array(3);
    out.bulk(b"master");
    out.int(server.repl_offset() as i64);
    let at = out.len();
    let mut n = 0;
    for held in &rows {
        if !held.online.load(Relaxed) {
            continue;
        }
        let (host, port) = held.address();
        out.array(3);
        out.bulk(host.as_bytes());
        out.bulk(port.to_string().as_bytes());
        out.bulk(held.ack.load(Relaxed).to_string().as_bytes());
        n += 1;
    }
    out.close_array(at, n);
}

/// The `Replication` section of `INFO`.
///
/// Redis reports a dozen fields on a master and a few more on a replica. This is
/// the master's set, since nothing here can be a replica yet, plus one `slaveN`
/// row per replica in the order they attached, which is what tooling reads to
/// find out who is following whom.
pub(super) fn info(server: &Server, s: &mut String) {
    use core::fmt::Write as _;
    let rows = server.replica_rows();
    let id = server.repl_id();
    let id2 = *server.repl.id2.lock();
    let offset = server.repl_offset();
    let now = server.clock.now_ms();
    let _ = write!(
        s,
        "# Replication\r\nrole:master\r\nconnected_slaves:{}\r\n",
        rows.len()
    );
    for (i, held) in rows.iter().enumerate() {
        let (host, port) = held.address();
        let state = if held.online.load(Relaxed) {
            "online"
        } else {
            "wait_bgsave"
        };
        let lag = now.saturating_sub(held.ack_ms.load(Relaxed)) / 1000;
        let _ = write!(
            s,
            "slave{i}:ip={host},port={port},state={state},offset={},lag={lag}\r\n",
            held.ack.load(Relaxed),
        );
    }
    let _ = write!(
        s,
        "master_failover_state:no-failover\r\n\
         master_replid:{}\r\nmaster_replid2:{}\r\n\
         master_repl_offset:{offset}\r\nsecond_repl_offset:{}\r\n",
        String::from_utf8_lossy(&id),
        String::from_utf8_lossy(&id2),
        server.repl.second.load(Relaxed),
    );
    // The backlog is made when the first replica attaches and is never given
    // back, which is what `repl_backlog_active` is saying. The histlen is how
    // much of it is real, so a server that has never had a replica reports a
    // backlog that is off and empty rather than one that is on and zero long.
    let (active, first, histlen) = {
        let backlog = server.repl.backlog.lock();
        (
            usize::from(!backlog.ring.is_empty()),
            backlog.first,
            backlog.filled,
        )
    };
    let _ = write!(
        s,
        "repl_backlog_active:{active}\r\nrepl_backlog_size:{BACKLOG_BYTES}\r\n\
         repl_backlog_first_byte_offset:{first}\r\nrepl_backlog_histlen:{histlen}\r\n\r\n",
    );
}
