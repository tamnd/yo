//! Every live connection, in one place any thread can read.
//!
//! `CLIENT INFO` only ever describes the connection asking, so it could be
//! answered out of the session and was. `CLIENT LIST` and `CLIENT KILL` are
//! about the other connections, and on a server with more than one thread the
//! other connections belong to somebody else: their sessions are inside another
//! thread's front, which this thread has no borrow of and must never take one
//! of. So the part of a connection those two commands report is not kept in the
//! session at all. It is kept here, in a row the connection owns and every
//! thread can read.
//!
//! # Why the row is atomics and not a lock
//!
//! A row has exactly one writer, which is the thread the connection is on, and
//! any number of readers, which is whoever ran `CLIENT LIST`. That is the
//! cheapest shape a shared thing can have: a relaxed store is an ordinary store
//! on every machine yo runs on, so the connection pays a store it was paying
//! anyway and nothing else. A lock per connection would put an atomic exchange
//! on the command path for a report almost nobody reads.
//!
//! The strings are the exception, because a string is not a word. The six of
//! them sit behind one small lock per row, and it is taken when a name is set,
//! when a library announces itself, and when a container command records its
//! subcommand, none of which is a hot path. A plain command stores the index of
//! its spec in the table and never goes near the lock.
//!
//! # Why the rows are a vector
//!
//! A connection opening pushes and a connection closing scans for its id and
//! lifts that row out, keeping the ones behind it in the order they opened in,
//! which is the order `CLIENT LIST` reports. That is linear in the number of
//! clients on a disconnect, which sounds worse than it is: the same walk is what
//! `CLIENT LIST` does, Redis keeps its clients in a list and walks it in the
//! same places, and a server with ten thousand connections is doing ten thousand
//! compares on a socket close and nothing on a command.
//!
//! # The pause is here too
//!
//! `CLIENT PAUSE` is not about one connection and does not touch a row, but it
//! is the same shape of problem: one connection arms something that every other
//! connection on every other thread has to see. It is one word on the server,
//! read once per command, and it lives beside the rows because `CLIENT` is what
//! writes it and what clears it.

use std::sync::Arc;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, AtomicU64, AtomicUsize};
use yo_common::lock::Lock;

/// The bit in [`Client::flags`] for a connection subscribed to anything, which
/// the report spells `P`.
pub(super) const SUBSCRIBED: u32 = 1;
/// A transaction is open, which the report spells `x`.
pub(super) const IN_MULTI: u32 = 2;
/// The socket is a Unix socket, which the report spells `U`.
pub(super) const UNIX: u32 = 4;
/// `CLIENT NO-EVICT ON`, which the report spells `e`.
pub(super) const NO_EVICT: u32 = 8;
/// `CLIENT NO-TOUCH ON`, which the report spells `T`.
pub(super) const NO_TOUCH: u32 = 16;
/// Somebody ran `CLIENT KILL` against this connection and the thread that owns
/// it has not noticed yet.
///
/// Not one of the letters. It is in the same word because it is set by another
/// thread and read by the owner on a path that is already loading this word.
pub(super) const KILLED: u32 = 32;
/// The connection sent `MONITOR` and is being fed every command, which the
/// report spells `O`.
///
/// Redis flags a monitor a replica as well and reports only the `O`, because
/// that is the letter for a replica that is a monitor rather than a real one.
/// Here the two are separate bits and a monitor sets only this one, which comes
/// to the same reported letter by a shorter route.
pub(super) const MONITOR: u32 = 128;
/// The connection sent `PSYNC` and is being fed the command stream, which the
/// report spells `S`.
///
/// Set on the connection itself as well as recorded in the replication module,
/// because `CLIENT LIST` and `CLIENT KILL TYPE replica` both ask the connection
/// what it is rather than asking the module who it has.
pub(super) const REPLICA: u32 = 256;
/// The thread that owns the connection has seen the kill and acted on it.
///
/// Set by the owner and read by the owner, so that a connection which cannot be
/// let go of on the turn it was killed, because commands framed out of its
/// buffer are still to run, is not counted a second time by the next turn.
pub(super) const REAPED: u32 = 64;

/// The strings a connection carries, which are the only part of a row that is
/// not a single word.
#[derive(Default)]
pub(super) struct Text {
    /// Where the client is dialling from, as `ip:port`, or the socket path with
    /// `:0` after it for a Unix connection, which is Redis's spelling for both.
    pub(super) peer: Vec<u8>,
    /// The address on this side, in the same two spellings.
    pub(super) local: Vec<u8>,
    /// The name the client gave itself with `CLIENT SETNAME`, empty if none.
    pub(super) name: Vec<u8>,
    /// What `CLIENT SETINFO` was told, empty until it is told.
    pub(super) lib_name: Vec<u8>,
    pub(super) lib_ver: Vec<u8>,
    /// The subcommand of the last command, when it had one.
    pub(super) sub: Vec<u8>,
    /// The ACL user this connection authenticated as.
    ///
    /// Empty means the default user, which is what a connection that never sent
    /// `AUTH` is, so the common case costs no allocation at all.
    pub(super) user: Vec<u8>,
}

/// One connection, as everybody except the connection itself sees it.
///
/// Built when the connection is accepted and dropped when the last reader lets
/// go of it, which is why it is behind an [`Arc`] rather than living in the
/// table: a `CLIENT LIST` copies the handles out under the lock and then formats
/// them with the lock let go of, and a connection that closes in between leaves
/// a row that is still readable rather than a dangling one.
pub struct Client {
    /// The client id, which is what `CLIENT KILL ID` names and what never comes
    /// round again.
    pub(super) id: u64,
    /// Which connection slot on which thread's front, so that a kill can be
    /// carried out by the thread that owns it.
    ///
    /// The slot is reused and the id is not, which is why whoever acts on this
    /// pair checks the id back against the front before it does anything.
    pub(super) conn: AtomicU32,
    pub(super) thread: AtomicUsize,
    /// When the connection was accepted, for `age`.
    pub(super) since_ms: AtomicU64,
    /// The descriptor number, or minus one when there is no socket, which is
    /// every embedded caller.
    pub(super) fd: AtomicI32,
    /// Everything about the connection that is a string.
    pub(super) text: Lock<Text>,
    /// When it last sent a command, for `idle`.
    pub(super) last_ms: AtomicU64,
    /// Bytes read off the socket and bytes handed to it.
    pub(super) net_in: AtomicU64,
    pub(super) net_out: AtomicU64,
    /// Commands run for this connection, and reads that carried at least one.
    ///
    /// The pair behind `avg-pipeline-len-sum` and `avg-pipeline-len-cnt`, which
    /// a client divides one by the other to see how deep the pipelining is.
    pub(super) cmds: AtomicU64,
    pub(super) reads: AtomicU64,
    /// Bytes sitting in the read buffer waiting to be framed, and the room after
    /// them, which are `qbuf` and `qbuf-free`.
    ///
    /// The number as of the last read or flush rather than as of this instant,
    /// which is the only two moments it can change and so the only two worth a
    /// store.
    pub(super) qbuf: AtomicU64,
    pub(super) qbuf_free: AtomicU64,
    /// The reply buffer's room, the largest it has been at a flush, and what was
    /// in it at the last flush, which are `rbs`, `rbp` and `obl`.
    pub(super) rbs: AtomicU64,
    pub(super) rbp: AtomicU64,
    pub(super) obl: AtomicU64,
    /// The bytes of the arguments of the last command, which is `argv-mem`.
    pub(super) argv_mem: AtomicU64,
    /// Where in the command table the last command was, or [`u32::MAX`] before
    /// the connection has sent one.
    ///
    /// An index and not a name because an index is a word, and a word is what a
    /// reader on another thread can take without a lock. The table outlives
    /// every connection, so the index is good for as long as the row is.
    pub(super) spec: AtomicU32,
    /// Whether [`Text::sub`] is worth reading, so that the common case of a
    /// command with no subcommand never takes the lock.
    ///
    /// Released after the subcommand is written and acquired before it is read,
    /// which is what stops a reader pairing a new command with the subcommand of
    /// an older one.
    pub(super) has_sub: AtomicU32,
    /// Which database, which protocol, and the four counts the report gives a
    /// field each.
    pub(super) db: AtomicU32,
    pub(super) resp: AtomicU32,
    pub(super) sub: AtomicU32,
    pub(super) psub: AtomicU32,
    pub(super) ssub: AtomicU32,
    pub(super) watch: AtomicU32,
    /// How many commands are queued behind `MULTI` and how many bytes they hold,
    /// where minus one is Redis's spelling for no transaction at all.
    pub(super) multi: AtomicI64,
    pub(super) multi_mem: AtomicU64,
    /// The letters, as bits. See the constants at the top of this module.
    pub(super) flags: AtomicU32,
}

impl Client {
    /// A row for a connection that has just been accepted.
    pub(super) fn new(id: u64) -> Client {
        Client {
            id,
            conn: AtomicU32::new(u32::MAX),
            thread: AtomicUsize::new(0),
            since_ms: AtomicU64::new(0),
            fd: AtomicI32::new(-1),
            text: Lock::default(),
            last_ms: AtomicU64::new(0),
            net_in: AtomicU64::new(0),
            net_out: AtomicU64::new(0),
            cmds: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            qbuf: AtomicU64::new(0),
            qbuf_free: AtomicU64::new(0),
            rbs: AtomicU64::new(0),
            rbp: AtomicU64::new(0),
            obl: AtomicU64::new(0),
            argv_mem: AtomicU64::new(0),
            spec: AtomicU32::new(u32::MAX),
            has_sub: AtomicU32::new(0),
            db: AtomicU32::new(0),
            resp: AtomicU32::new(2),
            sub: AtomicU32::new(0),
            psub: AtomicU32::new(0),
            ssub: AtomicU32::new(0),
            watch: AtomicU32::new(0),
            multi: AtomicI64::new(-1),
            multi_mem: AtomicU64::new(0),
            flags: AtomicU32::new(0),
        }
    }

    /// Turn one of the flag bits on or off.
    ///
    /// A load, a mask and a store rather than a fetch and modify, which is sound
    /// because the only bit another thread writes is [`KILLED`] and it is only
    /// ever turned on. The worst a lost update can do is leave a kill to the next
    /// command, and every path that acts on a kill is one that runs again.
    pub(super) fn set_flag(&self, bit: u32, on: bool) {
        let was = self.flags.load(Relaxed);
        let now = if on { was | bit } else { was & !bit };
        if now != was {
            self.flags.store(now, Relaxed);
        }
    }

    /// Whether a bit is on.
    pub(super) fn flag(&self, bit: u32) -> bool {
        self.flags.load(Relaxed) & bit != 0
    }

    /// Ask that this connection be closed, and say whether that was news.
    ///
    /// Called by whichever thread ran `CLIENT KILL`, which is very often not the
    /// thread that owns the connection. `Release` so that the owner, which
    /// acquires the same word, is looking at a row it can act on.
    pub(super) fn kill(&self) -> bool {
        let was = self.flags.load(Relaxed);
        if was & KILLED != 0 {
            return false;
        }
        self.flags.store(was | KILLED, Release);
        true
    }

    /// Whether a kill is waiting for the thread that owns this connection.
    pub(super) fn killed(&self) -> bool {
        self.flags.load(Acquire) & KILLED != 0
    }

    /// Note what the last command was.
    ///
    /// The index goes in with a `Release` when there is a subcommand, after the
    /// subcommand itself, so a reader that acquires the pair sees them together.
    /// A command with no subcommand does not go near the lock and does not need
    /// the fence.
    pub(super) fn note_command(&self, at: usize, sub: Option<&[u8]>) {
        match sub {
            Some(sub) => {
                yo_alloc::allow(|| {
                    let mut text = self.text.lock();
                    text.sub.clear();
                    text.sub.extend_from_slice(sub);
                });
                self.spec.store(at as u32, Relaxed);
                self.has_sub.store(1, Release);
            }
            None => {
                self.has_sub.store(0, Relaxed);
                self.spec.store(at as u32, Relaxed);
            }
        }
    }

    /// Replace one of the strings.
    pub(super) fn set_text(&self, pick: fn(&mut Text) -> &mut Vec<u8>, value: &[u8]) {
        yo_alloc::allow(|| {
            let mut text = self.text.lock();
            let into = pick(&mut text);
            into.clear();
            into.extend_from_slice(value);
        });
    }
}

/// Every connection this server has open.
///
/// One list and not one per thread, because the two commands that read it want
/// all of them in the order they were opened, and because a list per thread
/// would still need a lock each and would give `CLIENT LIST` an interleaving to
/// undo.
#[derive(Default)]
pub(super) struct Clients {
    rows: Vec<Arc<Client>>,
}

impl Clients {
    /// Take a new connection on.
    fn add(&mut self, row: &Arc<Client>) {
        yo_alloc::allow(|| self.rows.push(Arc::clone(row)));
    }

    /// Let go of one, by the id that never comes round again.
    fn remove(&mut self, id: u64) {
        if let Some(at) = self.rows.iter().position(|row| row.id == id) {
            // In order rather than by swapping the last row in, because `CLIENT
            // LIST` is read in the order the connections were opened on a real
            // server and people do read it that way. What that costs over the
            // swap is a move of the pointers after the hole, which is less than
            // the scan that found the hole.
            self.rows.remove(at);
        }
    }

    /// How many there are.
    fn len(&self) -> usize {
        self.rows.len()
    }
}

impl super::Server {
    /// Take a connection's row into the table.
    ///
    /// Called once, by whoever accepted it. A session whose row was never
    /// registered is one no other thread can see, which is every embedded
    /// caller and every test that builds a session by hand.
    pub(crate) fn register_client(&self, row: &Arc<Client>) {
        row.thread.store(self.my_slot(), Relaxed);
        self.clients.lock().add(row);
    }

    /// Take it back out, which is the connection ending.
    pub(crate) fn forget_client(&self, id: u64) {
        self.clients.lock().remove(id);
    }

    /// Copy out a handle to every open connection.
    ///
    /// The copy is the point. Formatting a report holds no lock, so a connection
    /// that opens or closes while `CLIENT LIST` is writing does not hold up the
    /// thread it is on, and the row of one that closed is still there to be
    /// read.
    pub(super) fn client_rows(&self) -> Vec<Arc<Client>> {
        let rows = self.clients.lock();
        yo_alloc::allow(|| rows.rows.clone())
    }

    /// How many connections are open, counted from the table.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.clients.lock().len()
    }

    /// Note that `n` more connections have been asked to close.
    pub(super) fn note_kills(&self, n: usize) {
        if n != 0 {
            self.kills.fetch_add(n, Release);
        }
    }

    /// Whether any thread has a kill to carry out.
    ///
    /// One relaxed load, which is what every turn of every loop pays for a
    /// command nearly nobody sends.
    #[must_use]
    pub fn kills(&self) -> usize {
        self.kills.load(Acquire)
    }

    /// Note that one of them has been carried out.
    pub fn kill_done(&self) {
        self.kills.fetch_sub(1, Release);
    }

    /// The rows this thread owns that have been asked to close.
    pub fn my_kills(&self) -> Vec<(u32, u64)> {
        let mine = self.my_slot();
        let rows = self.clients.lock();
        yo_alloc::allow(|| {
            rows.rows
                .iter()
                .filter(|row| row.thread.load(Relaxed) == mine && row.killed() && !row.flag(REAPED))
                .inspect(|row| row.set_flag(REAPED, true))
                .map(|row| (row.conn.load(Relaxed), row.id))
                .collect()
        })
    }

    /// Hold commands until `until_ms`, either all of them or only the writes.
    ///
    /// A pause already running is not replaced, it is widened. The later of the
    /// two deadlines wins and the stricter of the two modes wins, so a client
    /// that asked for everything to stop cannot have that undone by another
    /// client asking for only the writes to stop. That is Redis's rule and it is
    /// the one that makes the command safe to use for a failover, which is what
    /// it is for.
    ///
    /// A pause whose deadline has already gone by counts as no pause, so the
    /// widening only ever looks at one that is still running.
    pub fn pause(&self, until_ms: u64, all: bool) {
        // The deadline shares the word with the mode bit, so it has one bit less
        // than a `u64` to sit in. A pause of a hundred and forty million years
        // is the same as one of two hundred and eighty for everybody who has to
        // live through it.
        let until_ms = until_ms.min(u64::MAX >> 1);
        let want = (until_ms << 1) | u64::from(all);
        let mut have = self.pause.load(Relaxed);
        loop {
            let live = have != 0 && (have >> 1) > self.now_ms();
            let next = if live {
                ((have >> 1).max(until_ms) << 1) | (have & 1) | u64::from(all)
            } else {
                want
            };
            match self
                .pause
                .compare_exchange_weak(have, next, Release, Relaxed)
            {
                Ok(_) => return,
                Err(seen) => have = seen,
            }
        }
    }

    /// Let everybody go, which is `CLIENT UNPAUSE`.
    ///
    /// With one exception: a pause a failover armed is not an operator's to
    /// lift. The whole safety of that command is that no write lands here
    /// between the moment the target replica catches up and the moment it
    /// becomes the master, and a `CLIENT UNPAUSE` that opened that window would
    /// lose whatever went through it. `FAILOVER ABORT` is what lifts that one.
    /// A client's own pause underneath it goes, which is what was asked for.
    pub fn unpause(&self) {
        let keep = if self.failing_over() {
            (u64::MAX >> 1) << 1
        } else {
            0
        };
        self.pause.store(keep, Release);
    }

    /// Undo one particular pause, and only that one.
    ///
    /// The caller says what it armed, and if that is still exactly what is armed
    /// then it goes and if it is not then nothing happens. Widening is the whole
    /// reason: two pauses that overlap are one word, and the second one to be
    /// lifted must not take the first one with it. Anything an operator asked
    /// for is left to [`Self::unpause`], which is what `CLIENT UNPAUSE` is.
    pub fn lift(&self, until_ms: u64, all: bool) {
        let word = (until_ms.min(u64::MAX >> 1) << 1) | u64::from(all);
        let _ = self.pause.compare_exchange(word, 0, Release, Relaxed);
    }

    /// Whether commands are being held right now, and whether that is all of
    /// them.
    ///
    /// `None` is the answer on a server nobody has paused, and it costs one
    /// relaxed load and a test against zero, which is what every command pays.
    /// The deadline is only read on a server where somebody has.
    #[must_use]
    pub fn paused(&self, now_ms: u64) -> Option<bool> {
        let word = self.pause.load(Relaxed);
        if word == 0 || (word >> 1) <= now_ms {
            return None;
        }
        Some(word & 1 == 1)
    }

    /// When the pause runs out, in milliseconds, or zero if none is armed.
    ///
    /// Read by a test rather than by the command path, which asks the question
    /// above instead.
    #[must_use]
    pub fn pause_ends(&self) -> u64 {
        self.pause.load(Relaxed) >> 1
    }
}
