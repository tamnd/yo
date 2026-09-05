//! From a decoded command to a written reply.
//!
//! This is the layer Y23 exists to keep thin. The wire and the embedded API
//! both have to reach the same code, or there are two implementations of `INCR`
//! and one of them is wrong. So `yo-kv` holds one method per command taking
//! ordinary Rust values, and everything here is about the part that is only
//! true on a socket: which keyword goes where, which combinations a real server
//! refuses, and which of the two protocols the answer is spelled in.
//!
//! # What runs a command
//!
//! [`Server`] holds the databases. [`Session`] holds what one connection has
//! chosen: which database, which name it gave itself, what its id is. The
//! protocol version lives in the [`Out`] because that is what needs it, and
//! `HELLO` changes it there.
//!
//! ```
//! use yo_resp::{Argv, Limits, Out, Proto};
//! use yo_resp::dispatch::{Args, Flow, Server, Session, execute};
//!
//! let mut server = Server::new();
//! let mut session = Session::new(1);
//! let mut out = Out::new(Proto::Resp2);
//!
//! let wire = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n";
//! let mut argv = Argv::new();
//! argv.decode(wire, &Limits::default())?;
//! let flow = execute(&mut server, &mut session, Args::new(&argv, wire), &mut out);
//!
//! assert_eq!(flow, Flow::Continue);
//! assert_eq!(out.as_slice(), b"+OK\r\n");
//! # Ok::<(), yo_resp::ProtocolError>(())
//! ```
//!
//! # Errors are values until the last moment
//!
//! A command body returns a [`Result`], and this module turns the error into
//! the line that goes on the wire. That is what keeps the same body usable from
//! the embedded API, where an error is a value with a [`Code`] on it and not a
//! sentence to be parsed.
//!
//! The reply buffer is rolled back to where it was before a failing command
//! wrote anything, so a body that checks its arguments halfway through cannot
//! leave half a reply in front of the error.
//!
//! # Nothing here allocates
//!
//! Arguments are slices of the connection's read buffer, keywords are compared
//! in place, numbers are written straight into the reply, and the pairs of
//! `MSET` reach the store as an iterator rather than a `Vec`. The two places
//! that do allocate, an error message and the text of `INFO`, say so and wrap
//! it, because a shard thread that allocates aborts.

mod args;
mod arrays;
mod backup;
mod bits;
mod blocking;
mod bloom;
mod cms;
mod cpu;
mod cuckoo;
mod geo;
mod graph;
mod hashes;
mod himport;
mod hll;
mod indexing;
mod json;
mod keyspace;
mod lists;
mod migrate;
mod scan;
mod scripting;
mod search;
mod server;
mod sets;
mod streams;
mod strings;
pub mod table;
mod tdigest;
mod topk;
mod ts;
mod vectors;
mod vfilter;
mod zsets;

pub use args::Args;
pub use blocking::{Parked, Waiters};
pub use server::parse_memory;
pub use table::{COMMANDS, Spec, arity_ok, lookup};

use crate::reply::Out;
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use yo_common::lock::{Held, Lock};
use yo_common::{Code, Error};
use yo_kv::cold::Store;
use yo_kv::{Clock, Db, Keyspace};
use yo_search::Registry;

/// How many databases a server has.
///
/// Redis's default is sixteen and its `databases` setting can change it. Ours
/// is sixteen and cannot, which is why `CONFIG GET databases` can answer with a
/// constant. Nothing in the design needs the number to be fixed; nothing yet
/// needs it not to be.
pub const DATABASES: usize = 16;

/// Every database's bit in [`Server::dirty`], which is what a fresh server
/// starts on so that the first maintenance turn asks all of them.
///
/// A `u64` holds sixteen bits with room to spare, and the assertion below is
/// what turns raising [`DATABASES`] past sixty four into a build failure rather
/// than a shift that silently drops the databases past the end.
const ALL_DATABASES: u64 = if DATABASES == 64 {
    u64::MAX
} else {
    (1u64 << DATABASES) - 1
};
const _: () = assert!(DATABASES <= 64);

/// How many keys one command throws away before it leaves the rest to the next.
///
/// A bound and not a loop to the end, because this runs in front of a client
/// that is waiting for its reply, and a server a long way over its limit would
/// otherwise hold that client for as long as it took to walk all the way back
/// under. Sixty four is a batch's worth of commands, so a server that went over
/// by what one batch allocated comes back under in one command, and a server
/// whose limit was just cut in half works through it over the next few thousand
/// rather than in one long stall. Redis bounds the same loop by a time slice
/// instead of a count and hands the rest to a timer; there is no timer here, so
/// the rest goes to the next command that runs.
const EVICT_BUDGET: usize = 64;

/// The `maxstore` a server with no storage limit carries.
///
/// Sixteen exabytes, which is every disk there is and then some, so a server
/// that set a limit this high and a server that set none behave the same way and
/// the only difference is what `CONFIG GET maxstore` says. Zero cannot be the
/// sentinel because zero is a limit with a meaning: nothing may live on the
/// file.
const NO_MAXSTORE: u64 = u64::MAX;

/// What a server says to a command that would allocate when it has no room.
///
/// Redis's `shared.oomerr`, word for word including the full stop, because
/// clients match on the `OOM` prefix and people match on the sentence.
const OOM: &[u8] = b"command not allowed when used memory > 'maxmemory'.";

/// What the connection should do after a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Read the next command.
    Continue,
    /// Write what is buffered and then close, which is what `QUIT` asks for.
    Close,
    /// Nothing was written and nothing is owed yet.
    ///
    /// The client is on the waiter list and its reply comes when a key it named
    /// has something in it or when its deadline passes, whichever happens first.
    /// Until then the connection stops reading commands, because a client that
    /// is waiting for an answer is not a client that has sent another question.
    Block,
}

/// A number one thread adds to and any thread may read.
///
/// The add is a load, an add and a store rather than a fetch and add, which on
/// x86 is three ordinary instructions instead of one locked one. That is sound
/// because every counter here has exactly one writer, which is what the slots
/// below are for: two threads never hold the same counter, so nothing can be
/// lost between the load and the store. A reader can be a command or two behind,
/// and `INFO` on a running server is behind by the time the reply reaches the
/// client anyway.
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// One more.
    fn bump(&self) {
        self.0.store(self.get().wrapping_add(1), Relaxed);
    }

    /// One fewer, stopping at zero.
    ///
    /// The floor is for the gauge, which is the number of open connections: a
    /// close that arrives without its open, which nothing can do now and a
    /// misplaced call could, is a number that stays at zero rather than one
    /// that wraps to eighteen quintillion clients.
    fn drop_one(&self) {
        self.0.store(self.get().saturating_sub(1), Relaxed);
    }

    /// What it says.
    fn get(&self) -> u64 {
        self.0.load(Relaxed)
    }

    /// Back to zero, which is `CONFIG RESETSTAT`.
    fn zero(&self) {
        self.0.store(0, Relaxed);
    }
}

/// The numbers `INFO` reports that this layer cannot see for itself.
///
/// The reactor owns the sockets, so the reactor is what knows how many clients
/// there are. It counts them here and nothing else does anything with them
/// except report them.
#[derive(Debug, Default)]
pub struct Stats {
    /// Connections open right now.
    clients: Counter,
    /// Connections accepted since the server started.
    connections: Counter,
    /// Commands run since the server started, which this layer counts itself.
    commands: Counter,
}

impl Stats {
    /// A connection arrived.
    pub fn opened(&self) {
        self.clients.bump();
        self.connections.bump();
    }

    /// A connection went away.
    pub fn closed(&self) {
        self.clients.drop_one();
    }
}

/// Every thread's [`Stats`] added together, which is what `INFO` answers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Totals {
    /// Connections open right now.
    pub clients: u64,
    /// Connections accepted since the server started.
    pub connections: u64,
    /// Commands run since the server started.
    pub commands: u64,
}

thread_local! {
    /// Which set of counters the running thread writes into.
    ///
    /// Claimed the first time a thread counts anything and kept for as long as
    /// the thread runs. It is a number rather than a pointer, so a thread that
    /// has counted on one server and then counts on another lands in the same
    /// place in both, and a process with two servers in it shares the numbering
    /// between them. That is the tests and it is not `yodb`, which has one.
    static SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// What one thread keeps to itself.
///
/// One of these per thread and not one per server, because a number every
/// thread writes to is a cache line every thread has to own to write to it, and
/// at a few million commands a second that one line is the server. So each
/// thread writes into its own and whoever needs the whole picture, which is
/// `INFO` and the maintenance turn, puts the pieces together when it asks.
///
/// A cache line apart for the same reason, so that two threads writing at once
/// are not two threads passing one line back and forth.
#[derive(Debug)]
#[repr(align(64))]
struct Local {
    /// What the reactor counts.
    stats: Stats,
    /// A counter per command, for `INFO commandstats`.
    cmdstats: CommandStats,
    /// Which databases this thread has run a command against since the
    /// maintenance turn last took the mask.
    ///
    /// One bit per database. The thread ors into it and the turn takes the whole
    /// of it with a swap, which is what keeps a mark that lands during the swap
    /// from being lost: the worst that can happen is a bit the turn has already
    /// taken being set again, and that costs one more look at a database with
    /// nothing to collect.
    dirty: AtomicU64,
    /// The mask this thread's maintenance turn is working from.
    ///
    /// Its own and not a shared one, because a turn reads it in place and then
    /// clears bits of it, and a shared mask cleared that way would lose whatever
    /// another thread marked in between. Every thread turns a loop and every
    /// loop maintains, so what stops the same work being done twice is not the
    /// mask but the stripe lock underneath it: two threads that both look at
    /// database nine take turns, and the second one finds nothing left to move.
    ///
    /// Starts with every database set, so a server that has just been built
    /// looks at all of them once rather than waiting to be told about the ones
    /// something was loaded into before any command ran.
    turn: AtomicU64,
}

impl Default for Local {
    fn default() -> Local {
        Local {
            stats: Stats::default(),
            cmdstats: CommandStats::default(),
            dirty: AtomicU64::new(0),
            turn: AtomicU64::new(ALL_DATABASES),
        }
    }
}

impl Local {
    /// Note that a command has run against these databases.
    fn mark(&self, dbs: u64) {
        self.dirty.store(self.dirty.load(Relaxed) | dbs, Relaxed);
    }

    /// Add `dbs` to what this thread's turn is going to look at.
    fn note(&self, dbs: u64) {
        self.turn.store(self.turn.load(Relaxed) | dbs, Relaxed);
    }

    /// Take `at` off the list of databases this thread's turn will look at.
    fn done(&self, at: usize) {
        self.turn
            .store(self.turn.load(Relaxed) & !(1u64 << at), Relaxed);
    }

    /// Whether this thread's turn still has database `at` to look at.
    fn wanted(&self, at: usize) -> bool {
        self.turn.load(Relaxed) & (1u64 << at) != 0
    }
}

/// Room for one thread, which is what a server starts with.
fn one_thread() -> Box<[Local]> {
    slots(1)
}

/// Room for `threads` of them.
fn slots(threads: usize) -> Box<[Local]> {
    (0..threads.max(1)).map(|_| Local::default()).collect()
}

/// Where the process was started, which is what `dir` defaults to.
///
/// A dot if the working directory cannot be read, which happens when it has
/// been deleted out from under a running process. That is not a reason to
/// refuse to start a server, and it leaves `BACKUP` to fail with the real error
/// from the filesystem if anybody asks for one.
fn working_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// One command's counters, for `INFO commandstats`.
///
/// Three of Redis's five. `usec` and `usec_per_call` are not here because
/// nothing times a command, and timing one means two clock reads around a call
/// that takes tens of nanoseconds to begin with. Redis pays that because Redis
/// has room for it; this does not, and a zero under a name that says microseconds
/// is worse than an absent field, which is the same rule the rest of `INFO`
/// follows.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommandStat {
    /// Times the command ran, whatever it answered.
    pub calls: u64,
    /// Times it was turned away before it ran, which is the wrong number of
    /// arguments or no room under `maxmemory`.
    pub rejected: u64,
    /// Times it ran and answered with an error.
    pub failed: u64,
}

impl CommandStat {
    /// Whether this command has ever been seen.
    ///
    /// A row that has not is left out of the reply, which is what Redis does and
    /// is why the section is a handful of lines on a working server rather than
    /// one line per command in the table.
    const fn seen(&self) -> bool {
        self.calls != 0 || self.rejected != 0 || self.failed != 0
    }
}

/// One command's counters as one thread keeps them.
///
/// The same three numbers as [`CommandStat`], which is what they add up to when
/// `INFO` asks. This is the written form and that is the read one.
#[derive(Debug, Default)]
struct Row {
    /// Times the command ran.
    calls: Counter,
    /// Times it was turned away before it ran.
    rejected: Counter,
    /// Times it ran and answered with an error.
    failed: Counter,
}

/// A counter per command, indexed the way [`table::index_of`] says.
///
/// A flat array and not a map, because the dispatcher is already holding the
/// spec and the spec's position in the table is two addresses subtracted. That
/// makes the counting a load, an add and a store on a row the previous command
/// of the same name has already pulled into cache.
#[derive(Debug)]
struct CommandStats(Box<[Row]>);

impl Default for CommandStats {
    fn default() -> CommandStats {
        CommandStats((0..table::count()).map(|_| Row::default()).collect())
    }
}

impl CommandStats {
    /// The row for one command.
    fn at(&self, spec: &'static Spec) -> &Row {
        &self.0[table::index_of(spec)]
    }
}

/// Where a database gets its store from, asked by database number.
///
/// `None` means that database cannot have one. The caller owns whatever the
/// stores are cut out of, which for `yodb` is one `.yo` file with a log per
/// database, and this crate never learns what any of that is.
pub type StoreSource = dyn FnMut(usize) -> Option<Store> + Send;

/// Every thread that runs commands here shares this server, so it has to be
/// `Send` and `Sync`, and the check is here so that a type added to it that is
/// neither is a compile error where it was added rather than an error in the
/// code that starts the threads.
const _: () = {
    const fn shareable<T: Send + Sync>() {}
    shareable::<Server>();
};

/// Everything a server holds.
///
/// One per process, however many threads are serving out of it. What is inside
/// is either shared outright, which is the counters and the settings, or behind
/// a lock, which is the stripes and the few pieces of state a command can
/// change. What makes this a server rather than a shard is that it is the whole
/// of what a connection can address.
pub struct Server {
    dbs: Vec<Db>,
    /// How many stripes each database is cut into, the same for all of them.
    ///
    /// Kept here as well as in each database so that the flat slot arithmetic
    /// below is a multiply and a divide against a field on the server rather
    /// than a walk asking each database how wide it is.
    width: usize,
    clock: Clock,
    started_ms: u64,
    /// Where the next maintenance turn starts looking, so that a database
    /// under constant write load cannot hold the other fifteen's space.
    ///
    /// Shared, because compaction is asked for from two places: the maintenance
    /// turn, which is one thread, and a command that went over the memory limit
    /// and is trying to get back under it, which is any thread. Two threads that
    /// read the same cursor start on the same database, and what that costs is
    /// one of them finding the other has already moved what was there.
    next_db: AtomicUsize,
    /// One bit per database, set when a command ran against it.
    ///
    /// The maintenance turn after every batch used to ask all sixteen
    /// databases whether they had anything to collect, and asking costs a load
    /// and a store in each one. Fifteen of those are cold lines on a server
    /// where every client is on database zero, which is every server, and the
    /// answer is no every time. This is the cheap half of the question: a
    /// database nobody has touched since it last said no cannot have started
    /// saying yes.
    ///
    /// What the connections are holding, kept by the engine.
    ///
    /// Shared, because every thread has connections and the memory total is one
    /// total. Each thread adds and subtracts its own change rather than storing
    /// a figure it worked out, so two threads whose buffers grew in the same
    /// moment both count.
    conn_bytes: AtomicUsize,
    /// The `maxmemory` limit in bytes, zero when there is not one.
    ///
    /// Zero is the default and it is the whole reason the check in front of
    /// every write is one comparison against a field that is already warm. It
    /// is read by every command on every thread and written by a client that
    /// sends `CONFIG SET`, so it is a number the threads can share rather than
    /// a field one of them owns.
    maxmemory: AtomicU64,
    /// Where a database gets a store from the first time it needs one.
    ///
    /// A closure and not a store, because there are sixteen databases and a
    /// server that fills memory on database zero should not have opened
    /// anything for the other fifteen. Nothing is asked of this until a memory
    /// limit is actually reached, so a server that never fills memory never
    /// opens a file, and a server that has no file never has one of these.
    ///
    /// `None` from the closure means that database cannot have one, which is
    /// how the caller says the file it opened has no more room for logs.
    ///
    /// Behind a lock because it is a closure the caller gave us and there is no
    /// saying it can be run by two threads at once. It is asked once per
    /// database, the first time that database has to move something, so a
    /// server that has reached its memory limit takes this lock sixteen times
    /// in its life.
    store: Lock<Option<Box<StoreSource>>>,
    /// The `maxstore` limit in bytes, `None` when there is not one.
    ///
    /// The storage limit, and the other half of the inversion `14` section 4.1
    /// describes. `maxmemory` is a limit on memory and the right answer to a
    /// memory limit on a system with a file under it is to move data to the
    /// file, not to delete it. Deleting is the right answer to a limit on the
    /// file, and this is that limit.
    ///
    /// Zero is not "no limit" here, which is the one place this reads
    /// differently from `maxmemory` and is the difference that makes a drop in
    /// cache possible. A storage budget of zero bytes means nothing may live on
    /// the file, so migration cannot make room and eviction is the only thing
    /// left, which is Redis exactly. `None` is no limit and is the default,
    /// which with `noeviction` means the database grows until the disk is full
    /// and then writes fail, which is what a database does.
    ///
    /// Shared between the threads the same way `maxmemory` is, and no limit is
    /// [`NO_MAXSTORE`] rather than a second field saying whether the first one
    /// counts. Two fields cannot be read as one, and a limit that was on when
    /// the bytes were read and off by the time the number was is a limit that
    /// answers from a server that never existed.
    maxstore: AtomicU64,
    /// What [`Server::memory_bytes`] said at the last maintenance turn.
    ///
    /// The reading is a walk over every collection in every database and cannot
    /// go on a command path, so the command path reads this instead and is at
    /// most one batch behind. What that costs is overshoot: a server can end a
    /// batch holding one batch's worth of allocation more than its limit before
    /// anything notices. A batch is 64 commands, so that is bounded by what 64
    /// commands can allocate and not by how long the server runs.
    ///
    /// Only kept up to date when there is a limit to judge it against. A server
    /// with no `maxmemory` never reads it and never pays for it.
    ///
    /// Shared, because it is read in front of every write on every thread and
    /// written by whichever thread last took a reading. A reader that catches it
    /// mid write gets one of the two readings and both of them were true a
    /// moment ago, which is all this number ever claims to be.
    used: AtomicUsize,
    /// Which database the next eviction draws from.
    ///
    /// Its own cursor and not [`Server::next_db`], because eviction and
    /// compaction move at different rates and sharing one would make the
    /// database that gets compacted depend on how many keys were evicted.
    ///
    /// Shared for the same reason [`Server::next_db`] is, and with the same
    /// answer: two threads evicting at once may pick the same database, and one
    /// of them finds the other got there first and moves on.
    evict_db: AtomicUsize,
    /// Which database the next active expiry sweep starts at.
    ///
    /// A third cursor for the same reason there is a second one. A sweep runs on
    /// every turn of the loop and compaction runs when there is dead space, so
    /// sharing a cursor would make which database gets swept depend on which one
    /// was last collected.
    expire_db: AtomicUsize,
    /// The millisecond the last active expiry sweep ran on, so the next one on
    /// the same millisecond does not bother.
    ///
    /// One for the server and not one per thread, so the sweeping a server does
    /// is a function of how long it has been running and not of how many threads
    /// it was started with. Two threads that read the same millisecond can both
    /// decide to sweep, which costs one extra sweep of a budget that is already
    /// small and cannot happen twice for the same millisecond more than once per
    /// thread.
    expire_ms: AtomicU64,
    /// Clients parked on a blocking command.
    ///
    /// Behind a lock because a client parks on the thread that ran its command
    /// and is woken by whichever thread later puts something under a key it
    /// named, and those are not the same thread. The lock is only ever taken to
    /// park somebody, to serve somebody or to forget a connection that has gone,
    /// so a command that does not block never touches it.
    waiters: Lock<Waiters>,
    /// How many clients are parked.
    ///
    /// Beside the list rather than read out of it, because every command asks
    /// whether anybody is waiting and nearly every answer is no. Taking a lock
    /// to be told no would be a cache line every thread has to own to ask, which
    /// is the cost the list was put behind a lock to avoid.
    ///
    /// Written under the lock, by whoever changed the list, so the number and
    /// the list agree except while a change is in progress. A reader that asks
    /// during one is told about the moment before it, and the worst that costs
    /// is a walk of the list that serves nobody or one that has not started yet
    /// and happens on the next command instead.
    parked: AtomicUsize,
    /// Sockets `MIGRATE` is holding open to the servers it has talked to.
    ///
    /// Empty on a server nobody has migrated a key out of, which is nearly all
    /// of them, and it costs a vector's three words to be empty.
    ///
    /// Behind a lock because a socket cannot be written by two threads at once
    /// and a cache of them cannot be searched by one while another is taking an
    /// entry out. It is held for the whole of a migration, which is a round trip
    /// to another server, so two threads migrating at the same time take turns.
    /// That is the right way round: the alternative is a socket per thread per
    /// peer, and a `MIGRATE` is not what a server spends its time on.
    peers: Lock<migrate::Peers>,
    /// What each thread that runs commands here keeps to itself.
    ///
    /// A fixed list, because a thread reading its own entry must not have the
    /// list move under it, and how many threads there will be is known before
    /// any of them starts. A server nobody told otherwise has one.
    locals: Box<[Local]>,
    /// How many entries have been handed out.
    claimed: AtomicUsize,
    /// Where `BACKUP` puts its files, and where `CONFIG GET dir` points.
    ///
    /// Absolute, and resolved once when the server is built rather than every
    /// time somebody asks. `BACKUP LIST` answers absolute paths and a client is
    /// entitled to hand one of them to a copy tool, so a relative path that
    /// meant something different after a `chdir` would be a path that stops
    /// working for reasons nobody could see.
    dir: PathBuf,
    /// What backup is running, if one is.
    ///
    /// On the server and not on a session, because a backup outlives the
    /// connection that asked for it and any other connection can seal it.
    ///
    /// Behind a lock because there is one backup at a time and any thread can be
    /// the one that starts, seals or abandons it. It is held while the base file
    /// is written, which is what keeps two `BACKUP START` commands from writing
    /// over each other's files.
    backup: Lock<backup::State>,
    /// Whether a sealed backup is sitting on disk.
    ///
    /// Beside the state rather than read out of it, because every batch of
    /// commands asks whether there is a backup old enough to sweep away and on
    /// nearly every server the answer is that there is no backup at all. A load
    /// answers that. Written under the lock by whoever moved the phase, so a
    /// reader that asks mid-change sees the moment before and sweeps one batch
    /// later, which is a file staying on disk for a few microseconds longer than
    /// it had to.
    sealed: AtomicBool,
    /// The search indexes and the names pointing at them.
    ///
    /// On the server and not on a database, which is the one collection in this
    /// build that is. A real server keeps its indexes in the search module, the
    /// module has one table, and `SELECT 1` followed by `FT._LIST` lists the
    /// indexes made on database zero. `search.rs` has the rest of why.
    ///
    /// A server nobody has made an index on holds two empty vectors here, which
    /// is six words and no allocation.
    ///
    /// Behind a lock because an index is made and dropped by whichever thread
    /// ran the command, and the table it goes in is one table. Only the `FT`
    /// commands take it, so nothing a working server spends its time on comes
    /// through here.
    search: Lock<Registry>,
    /// Set by `SHUTDOWN`, and read by whatever is turning the loop.
    ///
    /// A flag rather than an exit, because the command layer is not what owns
    /// the process. It runs inside a batch that has other commands behind it
    /// and inside a driver that has a socket file to take away and a file to
    /// close, and a server that calls `exit` from a command handler skips all
    /// of that. So the command says stop and the driver stops, on the same turn
    /// and through the same door a signal uses.
    stopping: AtomicBool,
}

impl Server {
    /// A server with [`DATABASES`] empty databases on the system clock.
    #[must_use]
    pub fn new() -> Server {
        let clock = Clock::system();
        Server {
            dbs: (0..DATABASES)
                .map(|_| Db::with_clock(clock.clone(), 1))
                .collect(),
            width: 1,
            started_ms: clock.now_ms(),
            clock,
            next_db: AtomicUsize::new(0),
            conn_bytes: AtomicUsize::new(0),
            maxmemory: AtomicU64::new(0),
            store: Lock::new(None),
            maxstore: AtomicU64::new(NO_MAXSTORE),
            used: AtomicUsize::new(0),
            evict_db: AtomicUsize::new(0),
            expire_db: AtomicUsize::new(0),
            expire_ms: AtomicU64::new(0),
            waiters: Lock::default(),
            parked: AtomicUsize::new(0),
            peers: Lock::default(),
            locals: one_thread(),
            claimed: AtomicUsize::new(0),
            dir: working_dir(),
            backup: Lock::default(),
            sealed: AtomicBool::new(false),
            search: Lock::new(Registry::new()),
            stopping: AtomicBool::new(false),
        }
    }

    /// A server whose databases are cut into `width` stripes each.
    ///
    /// Not reachable from the command line yet. Every command group answers on
    /// a server of any width now and so does everything that walks a whole
    /// database, and the tests run each group at a width of one and a width of
    /// eight and check the two agree.
    ///
    /// What is left before this is what `--threads` sets is the engine. A
    /// database being several objects is what makes more than one thread
    /// possible, and it is not what makes more than one thread happen.
    #[must_use]
    pub fn with_width(width: usize) -> Server {
        let mut server = Server::new();
        // The server's own clock and not a fresh one, because a database
        // reading a different clock from the server it is on is a database
        // whose keys expire against a time nobody set.
        let clock = server.clock.clone();
        server.dbs = (0..DATABASES)
            .map(|_| Db::with_clock(clock.clone(), width))
            .collect();
        server.width = server.dbs[0].width();
        server
    }

    /// A server on a clock the caller moves by hand, for tests.
    #[must_use]
    pub fn with_clock(clock: Clock) -> Server {
        Server {
            dbs: (0..DATABASES)
                .map(|_| Db::with_clock(clock.clone(), 1))
                .collect(),
            width: 1,
            started_ms: clock.now_ms(),
            clock,
            next_db: AtomicUsize::new(0),
            conn_bytes: AtomicUsize::new(0),
            maxmemory: AtomicU64::new(0),
            store: Lock::new(None),
            maxstore: AtomicU64::new(NO_MAXSTORE),
            used: AtomicUsize::new(0),
            evict_db: AtomicUsize::new(0),
            expire_db: AtomicUsize::new(0),
            expire_ms: AtomicU64::new(0),
            waiters: Lock::default(),
            parked: AtomicUsize::new(0),
            peers: Lock::default(),
            locals: one_thread(),
            claimed: AtomicUsize::new(0),
            dir: working_dir(),
            backup: Lock::default(),
            sealed: AtomicBool::new(false),
            search: Lock::new(Registry::new()),
            stopping: AtomicBool::new(false),
        }
    }

    /// One database, by index.
    ///
    /// A caller that knows which key it wants names the one stripe the key is
    /// on rather than working over the whole thing, which is what `at` and its
    /// neighbours on [`Db`] are for. A caller that is about a database rather
    /// than about a key, which is the snapshot walk and a setting, works over
    /// all of them.
    ///
    /// The database is marked as having had something run against it, which is
    /// what this does that [`Server::striped_ref`] does not. Anything that only
    /// reads asks for that one and leaves the mark alone.
    ///
    /// The borrow is shared, and what makes that enough is that a database is
    /// several stripes behind a lock each. A caller that wants to change
    /// something holds the stripe it is changing, so two threads working on two
    /// keys work at once and two working on one key take turns, which is the
    /// whole point of cutting a database up.
    ///
    /// # Panics
    ///
    /// If `i` is not a database. `SELECT` is the only way a client changes the
    /// index and it checks, so an index that is out of range here is a bug in
    /// the caller and not something a client can ask for.
    pub fn striped(&self, i: usize) -> &Db {
        self.mine().mark(1u64 << i);
        &self.dbs[i]
    }

    /// Every keyspace on the server, which is every stripe of every database.
    ///
    /// What the aggregates walk. A total over the whole server is a total over
    /// all of these and the stripe boundaries do not appear in it, which is
    /// what makes the numbers `INFO` reports the same numbers whatever the
    /// server was cut into.
    fn keyspaces(&self) -> impl Iterator<Item = Held<'_, Keyspace>> {
        self.dbs
            .iter()
            .flat_map(|db| (0..db.width()).map(|i| db.hold_stripe(i)))
    }

    /// How many keyspaces there are, counting every stripe of every database.
    ///
    /// The maintenance turns walk these rather than the databases, because a
    /// stripe is the thing that holds an arena and a deadline heap and so it is
    /// the thing that has anything to collect.
    const fn slots(&self) -> usize {
        DATABASES * self.width
    }

    /// Which database slot `i` belongs to.
    const fn slot_db(&self, i: usize) -> usize {
        i / self.width
    }

    /// Keyspace `i` of [`Server::slots`].
    fn slot(&self, i: usize) -> Held<'_, Keyspace> {
        let (db, stripe) = (i / self.width, i % self.width);
        self.dbs[db].hold_stripe(stripe)
    }

    /// Where `BACKUP` writes and what `CONFIG GET dir` answers.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Point the server at a different directory, which `yodb serve --dir` does.
    ///
    /// Only before it is serving. There is no `CONFIG SET dir` here and there
    /// is none on a real server either without turning protected configs on,
    /// for the good reason that moving it out from under a running backup would
    /// leave files nothing can find again.
    pub fn set_dir(&mut self, dir: PathBuf) {
        self.dir = dir;
    }

    /// Drop a sealed backup that has outlived `backup-sealed-ttl`.
    ///
    /// Once per batch, from the same maintenance turn that collects the arena.
    /// It reads two fields and returns on a server that has never taken a
    /// backup, which is nearly all of them.
    pub fn backup_expire(&self) {
        backup::expire(self);
    }

    /// Ask for the server to stop, which is what `SHUTDOWN` does.
    ///
    /// It sets a flag and returns. Nothing here closes a socket, flushes a file
    /// or ends the process, because none of those belong to this layer, and a
    /// batch that is halfway through still has to finish and be written out.
    pub fn stop(&self) {
        self.stopping.store(true, Release);
    }

    /// Whether somebody has asked the server to stop.
    ///
    /// Read once per turn by the loop, next to the flag a signal sets. The two
    /// mean the same thing and are separate only because one arrives from the
    /// operating system and the other from a client.
    #[must_use]
    pub fn stopping(&self) -> bool {
        self.stopping.load(Acquire)
    }

    /// One database, by index, without taking it mutably.
    ///
    /// What the prefetch stage needs. It runs for all 64 commands in a batch
    /// before any of them executes, so it cannot hold the mutable borrow `run`
    /// is about to want, and it does not need one: warming a cache line reads
    /// nothing and changes nothing.
    #[must_use]
    pub fn striped_ref(&self, i: usize) -> &Db {
        &self.dbs[i]
    }

    /// The stripe that answers for a database when a setting is read back.
    ///
    /// A ladder setting and an eviction policy are one number on a real server,
    /// and the fact that every stripe of every database carries a copy of it is
    /// ours rather than the client's problem. A write puts the same value on
    /// every one of them, so any stripe answers for all of them and this is the
    /// first one.
    fn settings(&self) -> Held<'_, Keyspace> {
        self.dbs[0].hold_stripe(0)
    }

    /// Take a new clock reading, which every database is looking at.
    ///
    /// Once per turn of the event loop, which is the only place time moves. A
    /// command asking what the time is gets the answer the whole batch got, so
    /// two keys written by the same batch expire together (`04` section 3).
    ///
    /// Every thread does this on every turn of its own loop and they do not
    /// have to agree about when. The reading is only stored when the
    /// millisecond has changed, so what the threads are sharing is a line that
    /// is written about a thousand times a second and read millions.
    pub fn refresh_clock(&self) {
        self.clock.refresh();
    }

    /// Move every clock here on by `ms`, for tests about expiry.
    ///
    /// The same thing [`Server::set_clock_ms`] does and by the same argument,
    /// except that it moves from wherever the clock is rather than to a stated
    /// moment, which is what a test that wants a key to have expired asks for.
    pub fn advance_clock_ms(&self, ms: u64) {
        let now = self.clock.now_ms() + ms;
        self.set_clock_ms(now);
    }

    /// Move every clock here to `ms` by hand, for tests about expiry.
    ///
    /// A test cannot wait a hundred seconds and a test that waits a hundred
    /// milliseconds is a test that fails on a loaded machine, so time moves on
    /// request. The system clock underneath will overwrite this on the next
    /// [`Server::refresh_clock`], which is why this is only useful in a test
    /// that drives commands directly rather than through the event loop.
    pub fn set_clock_ms(&self, ms: u64) {
        self.clock.set(ms);
    }

    /// Seconds since this server was built.
    #[must_use]
    pub fn uptime_secs(&self) -> u64 {
        self.clock.now_ms().saturating_sub(self.started_ms) / 1000
    }

    /// Bytes held by every database's index and arena, plus the read and reply
    /// buffers of every connection.
    ///
    /// The buffers are in here because they are real and because Redis counts
    /// its own, so leaving them out would make the one number people compare
    /// flattering rather than true. They are not a database, so nothing in the
    /// keyspace can change them and the engine has to say when they move.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.keyspaces().map(|db| db.memory_bytes()).sum::<usize>() + self.conn_bytes()
    }

    /// What the keyspace itself is holding, live records only.
    ///
    /// `used_memory` minus this is what the store costs to run: the index, the
    /// space dead records are sitting in until compaction gets to them, and the
    /// connections' buffers.
    #[must_use]
    pub fn dataset_bytes(&self) -> usize {
        self.keyspaces()
            .map(|db| db.map().arena().live_bytes() as usize)
            .sum()
    }

    /// Bytes the arenas are holding, live and dead together.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.keyspaces()
            .map(|db| db.map().arena().reserved_bytes() as usize)
            .sum()
    }

    /// Bytes the indexes are holding.
    #[must_use]
    pub fn index_bytes(&self) -> usize {
        self.keyspaces()
            .map(|db| db.map().index().memory_bytes())
            .sum()
    }

    /// What arena compaction has cost, across every database.
    ///
    /// The write amplification of value separation, which is invisible from the
    /// outside otherwise: a client that writes a megabyte can leave the store
    /// copying several more, and the only sign of it without these is that the
    /// writes got slower.
    #[must_use]
    pub fn compaction(&self) -> yo_kv::Compaction {
        self.keyspaces().map(|db| db.map().compaction()).fold(
            yo_kv::Compaction::default(),
            |a, b| yo_kv::Compaction {
                walked: a.walked + b.walked,
                moved: a.moved + b.moved,
                bytes: a.bytes + b.bytes,
            },
        )
    }

    /// Arena segments whose pages are real, across every database.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.keyspaces()
            .map(|db| db.map().arena().resident_segments())
            .sum()
    }

    /// What the connections' read and reply buffers are holding.
    #[must_use]
    pub fn conn_bytes(&self) -> usize {
        self.conn_bytes.load(Relaxed)
    }

    /// Note that the connections are holding `delta` bytes more than they were,
    /// or fewer when it is negative.
    ///
    /// A delta and not a total because the alternative is a walk over every
    /// connection, and the walk would have to happen on a turn of the loop
    /// rather than when `INFO` asks, which puts the cost of a report on the
    /// command path of a server nobody is asking.
    pub fn note_conn_bytes(&self, delta: isize) {
        // A read and a write and not a fetch and add, because the number is a
        // sum of signed changes and the saturating part has to happen in the
        // middle. Two threads that change their buffers in the same instant can
        // lose one of the two changes, which is a report that is a few kilobytes
        // out until the next connection on either thread moves it again.
        self.conn_bytes
            .store(self.conn_bytes().saturating_add_signed(delta), Relaxed);
    }

    /// Keys reclaimed by running into them after their deadline.
    #[must_use]
    pub fn expired_keys(&self) -> u64 {
        self.keyspaces().map(|db| db.expired_keys()).sum()
    }

    /// Keys thrown away to make room, which is the other number entirely.
    #[must_use]
    pub fn evicted_keys(&self) -> u64 {
        self.keyspaces().map(|db| db.evicted_keys()).sum()
    }

    /// Every command that has been seen, with its counters.
    ///
    /// Only the ones that have. A server reports a handful of lines rather than
    /// one per command in the table, which is what Redis does and is the
    /// difference between a section a person can read and one they cannot.
    pub fn command_stats(&self) -> impl Iterator<Item = (&'static str, CommandStat)> {
        (0..table::count())
            .map(|at| (table::name_at(at), self.command_stat(at)))
            .filter(|(_, row)| row.seen())
    }

    /// One command's counters, added up over every thread.
    fn command_stat(&self, at: usize) -> CommandStat {
        let mut sum = CommandStat::default();
        for thread in &self.locals {
            let row = &thread.cmdstats.0[at];
            sum.calls += row.calls.get();
            sum.rejected += row.rejected.get();
            sum.failed += row.failed.get();
        }
        sum
    }

    /// The counters the calling thread writes into.
    ///
    /// The first call on a thread claims a set and every call after it is a
    /// thread local read and an index. A server asked to count from more threads
    /// than it was built for wraps round and shares a set, which loses the odd
    /// count between two threads and cannot happen to a server `yodb serve`
    /// built, because that one is told how many threads it will have before it
    /// starts any of them.
    pub fn counted(&self) -> &Stats {
        &self.mine().stats
    }

    /// Everything the calling thread keeps to itself.
    fn mine(&self) -> &Local {
        let mut slot = SLOT.get();
        if slot == usize::MAX {
            slot = self.claimed.fetch_add(1, Relaxed);
            SLOT.set(slot);
        }
        &self.locals[slot % self.locals.len()]
    }

    /// Every thread's numbers added together, which is what `INFO` reports.
    #[must_use]
    pub fn totals(&self) -> Totals {
        let mut sum = Totals::default();
        for thread in &self.locals {
            sum.clients += thread.stats.clients.get();
            sum.connections += thread.stats.connections.get();
            sum.commands += thread.stats.commands.get();
        }
        sum
    }

    /// Put the totals back to zero, which is `CONFIG RESETSTAT`.
    ///
    /// Every thread's set and not only the one asking, since the number the
    /// client is resetting is the sum it was just shown. The open connections
    /// are left alone because that is a gauge and not a total: the connections
    /// are still open.
    pub fn reset_stats(&self) {
        for thread in &self.locals {
            thread.stats.connections.zero();
            thread.stats.commands.zero();
        }
    }

    /// Say how many threads will run commands here, before any of them does.
    ///
    /// What it changes is how many sets of counters there are. Called once at
    /// startup by whoever is about to start the threads, and calling it on a
    /// running server throws away what has been counted so far, which is why it
    /// wants the server to itself.
    pub fn set_threads(&mut self, threads: usize) {
        self.locals = slots(threads);
        self.claimed = AtomicUsize::new(0);
    }

    /// The `maxmemory` limit in bytes, zero when there is not one.
    #[must_use]
    pub fn maxmemory(&self) -> u64 {
        self.maxmemory.load(Relaxed)
    }

    /// Set the limit, and take a reading straight away.
    ///
    /// The reading is here rather than left to the next maintenance turn because
    /// a client that sets the limit and sends a write in the same batch expects
    /// the write to be judged against the limit it just set, and because the
    /// cached number is meaningless until the first time there is a limit to
    /// compare it with.
    ///
    /// Turning the limit on also turns on the running total every slab keeps of
    /// what its collections hold, and turning it off turns that back off, so a
    /// server with no limit is not paying to count something nobody reads. The
    /// first reading after switching it on is the walk that the total starts
    /// from, and it is the only walk.
    pub fn set_maxmemory(&self, bytes: u64) {
        self.maxmemory.store(bytes, Relaxed);
        for db in &self.dbs {
            db.track_memory(bytes != 0);
        }
        self.used.store(self.settled_memory(), Relaxed);
    }

    /// Say where a database should get its store from when it needs one.
    ///
    /// This is what turns the eviction inversion on. Until it is called every
    /// database answers a memory limit by evicting, which is Redis, and after it
    /// is called a database under memory pressure moves values to whatever the
    /// closure hands back instead of throwing keys away.
    ///
    /// Called at most once per database and only under pressure, so a server
    /// that is given a file and never fills memory never touches it.
    pub fn set_store_source(
        &mut self,
        source: impl FnMut(usize) -> Option<Store> + Send + 'static,
    ) {
        *self.store.lock() = Some(Box::new(source));
    }

    /// Whether this server has been given somewhere to put cold values.
    #[must_use]
    pub fn has_store_source(&self) -> bool {
        self.store.lock().is_some()
    }

    /// Open database `at`'s store, if it has not got one and there is one to be
    /// had.
    ///
    /// A store that will not open leaves the database where it was, which is
    /// evicting, because a memory limit that cannot be answered by moving data
    /// still has to be answered.
    fn attach_store(&self, at: usize) {
        if self.slot(at).store_bytes().is_some() {
            return;
        }
        // The closure is run with its lock held and the keyspace is taken after
        // it has answered, so the file is opened once however many threads asked
        // for it and the stripe is not held while a file is being opened.
        let mut source = self.store.lock();
        let Some(source) = source.as_mut() else {
            return;
        };
        if let Some(blocks) = source(at) {
            self.slot(at).attach(blocks);
        }
    }

    /// The `maxstore` limit in bytes, `None` when there is not one.
    #[must_use]
    pub fn maxstore(&self) -> Option<u64> {
        match self.maxstore.load(Relaxed) {
            NO_MAXSTORE => None,
            bytes => Some(bytes),
        }
    }

    /// Set the storage limit, or clear it with `None`.
    ///
    /// Nothing is read here the way [`Server::set_maxmemory`] reads the memory
    /// total, because this limit is compared against a number the store keeps
    /// and answers on demand, not against a walk.
    pub fn set_maxstore(&self, bytes: Option<u64>) {
        self.maxstore.store(bytes.unwrap_or(NO_MAXSTORE), Relaxed);
    }

    /// What every attached store is holding, for `INFO memory`.
    ///
    /// Zero on a server with nothing attached, which is not the same as a server
    /// whose file is empty, and [`Server::regime`] is the field that tells those
    /// two apart.
    #[must_use]
    pub fn store_bytes(&self) -> u64 {
        self.keyspaces().filter_map(|db| db.store_bytes()).sum()
    }

    /// What the file has been asked to do, added up over every database.
    ///
    /// Counters and not levels, so they only ever go up and a run is the
    /// difference between two readings. G9 is a ratio over these: the faults a
    /// run took, divided by the point reads it issued, has to come out at 1.05
    /// or less with a working set ten times memory. There is no way to work that
    /// out from outside the server, so it is reported rather than inferred.
    ///
    /// A fault is a read that went to the store. Whether it also went to the
    /// device depends on the store: a log serves a read out of a resident page
    /// without touching anything. At ten times memory almost every fault is a
    /// real read, which is why the gate is written against this number, but the
    /// two are not the same thing and a run tight against the bar should be
    /// checked against what the operating system says.
    #[must_use]
    pub fn cold_stats(&self) -> yo_kv::tier::Stats {
        let mut total = yo_kv::tier::Stats::default();
        for db in self.keyspaces() {
            let Some(tier) = db.tier() else { continue };
            let s = tier.stats();
            total.demoted += s.demoted;
            total.promoted += s.promoted;
            total.faults += s.faults;
            total.served += s.served;
            total.bytes_out += s.bytes_out;
            total.bytes_in += s.bytes_in;
        }
        total
    }

    /// Which way this server answers a memory limit, in one word for `INFO`.
    ///
    /// `evict` is Redis: a memory limit throws keys away. `migrate` is the
    /// inversion: a memory limit moves values to the file and nothing stored is
    /// lost. A server reports one word rather than leaving an operator to work
    /// it out from a limit, a setting and whether a file happens to be open.
    #[must_use]
    pub fn regime(&self) -> &'static str {
        if (0..self.slots()).any(|at| self.migrates(at)) {
            "migrate"
        } else {
            "evict"
        }
    }

    /// Whether database `at` answers a memory limit by moving values to the
    /// file rather than by throwing keys away.
    ///
    /// Three things have to hold. There has to be somewhere to move them, which
    /// is a store attached to that database or a source that can open one, and
    /// on a server that was never given a file this is false everywhere and
    /// every database behaves exactly as it did.
    /// The storage budget has to be more than nothing, which is what
    /// `maxstore 0` says it is not. And the file has to be under that budget,
    /// because a full file is a storage limit reached and eviction is the right
    /// answer to a storage limit.
    fn migrates(&self, at: usize) -> bool {
        let cap = self.maxstore();
        if cap == Some(0) {
            return false;
        }
        // Out of the stripe first. A match keeps whatever it is looking at
        // alive for the whole of itself, and that would be this stripe held
        // across the arms for no reason.
        let bytes = self.slot(at).store_bytes();
        match bytes {
            Some(held) => cap.is_none_or(|cap| held < cap),
            // Nothing attached, but somewhere to get one from the moment this
            // database needs it, which is what makes the answer yes rather than
            // no. Opening it here would mean `INFO` opened files.
            None => self.store.lock().is_some(),
        }
    }

    /// Take a fresh memory reading, which the maintenance turn does once a batch.
    ///
    /// Nothing at all when there is no limit, which is the default and is every
    /// server that has not asked for one.
    pub fn refresh_memory(&self) {
        if self.maxmemory() != 0 {
            self.used.store(self.settled_memory(), Relaxed);
        }
    }

    /// [`Server::memory_bytes`], asked the cheap way.
    ///
    /// The same number. The difference is that this asks each database only
    /// about the collections that could have moved since the last time, which is
    /// what a batch touched rather than what the server holds, so it can be
    /// asked once a batch and again on every command that is over the limit.
    fn settled_memory(&self) -> usize {
        self.keyspaces()
            .map(|mut db| db.settled_memory_bytes())
            .sum::<usize>()
            + self.conn_bytes()
    }

    /// Make room under the `maxmemory` limit, throwing keys away if that is what
    /// it takes. Answers whether there is anything left it could throw away.
    ///
    /// Redis runs the same thing from `processCommand` before every command and
    /// so does this: a client that writes has to be judged at the moment it
    /// writes, not a batch later, or the limit is a suggestion.
    ///
    /// Three things happen in the loop and all three are needed. Eviction picks
    /// a key and drops it. Compaction gives the pages back, because dropping a
    /// key marks its record dead and returns nothing on its own, so a loop that
    /// only evicted would throw the whole keyspace away and watch the number
    /// stay where it was. The reading is taken again each time round, because
    /// the two of them together are the only thing that moves it.
    ///
    /// # Why running out of budget is not a no
    ///
    /// `false` means there was nothing left to evict, which is `noeviction`, or
    /// a `volatile` policy on a database where nothing has a deadline, or a
    /// keyspace that is already empty. It does not mean the server is still over
    /// its limit, and that difference is Redis's: `performEvictions` answers
    /// `EVICT_FAIL` only when it has run out of things to delete, and
    /// `processCommand` refuses the client on that and on nothing else. Running
    /// out of time part way through a job it is doing well comes back as
    /// `EVICT_RUNNING` and the command goes through, because a server that is
    /// evicting steadily and refusing every write while it does it is worse for
    /// the client than a little overshoot.
    ///
    /// # What the limit is worth
    ///
    /// Space comes back a segment at a time and a segment is two megabytes, so
    /// this holds a server to its limit give or take a segment. A `maxmemory` of
    /// a few hundred megabytes gets what it asked for. A `maxmemory` of four
    /// megabytes is asking for a precision this store does not have.
    pub fn make_room(&self) -> bool {
        let limit = self.maxmemory();
        if limit == 0 || self.used.load(Relaxed) as u64 <= limit {
            return true;
        }
        // The cached reading is a batch old and the batch may have compacted
        // since, so take a fresh one before throwing anything away. It is the
        // settled reading and not the walk, so what this costs is the handful of
        // collections the last batch touched and not the whole database.
        let mut used = self.settled_memory();
        self.used.store(used, Relaxed);
        let mut budget = EVICT_BUDGET;
        while used as u64 > limit {
            let over = used - limit as usize;
            if !self.relieve_step(over) {
                return false;
            }
            self.compact_hard_step();
            used = self.settled_memory();
            self.used.store(used, Relaxed);
            budget -= 1;
            if budget == 0 {
                break;
            }
        }
        true
    }

    /// Give back `over` bytes from whichever database can, by moving values to
    /// the file where there is one and by throwing keys away where there is not.
    ///
    /// The two answers are the eviction inversion and which one a database gets
    /// is [`Server::migrates`]. Answers whether anything was given back at all,
    /// and `false` is what refuses the client's write.
    ///
    /// A store that will not take the bytes counts as nothing given back, so the
    /// write is refused rather than turned into a deletion. A disk that is
    /// misbehaving is a reason to stop accepting writes and it is not a reason
    /// to start losing data that was accepted already.
    ///
    /// Round robin from a cursor rather than always starting at database zero,
    /// so a server using more than one of them does not empty the first before
    /// touching the second. Almost every server is on database zero only, where
    /// this is one call that answers and fifteen that say the map is empty.
    fn relieve_step(&self, over: usize) -> bool {
        let from = self.evict_db.load(Relaxed);
        for turn in 0..self.slots() {
            let i = (from + turn) % self.slots();
            // An empty keyspace has nothing to move and opening a log for one
            // would cost a resident page window to find that out.
            let used = !self.slot(i).is_empty();
            let gave = if used && self.migrates(i) {
                self.attach_store(i);
                // Whether it made room and not whether it moved a key. A round
                // that demoted nothing and handed back a segment is a round
                // that made room, and reading only the count refuses the write
                // that provoked it.
                self.slot(i)
                    .relieve(over)
                    .is_ok_and(yo_kv::tier::Relief::made_room)
            } else {
                self.slot(i).evict_one()
            };
            if gave {
                self.evict_db.store((i + 1) % self.slots(), Relaxed);
                self.mine().mark(1u64 << self.slot_db(i));
                return true;
            }
        }
        false
    }

    /// The sweep the shard loop calls, at most once a millisecond.
    ///
    /// The gate is the whole difference between this and [`Server::expire_step`].
    /// A maintenance slice runs on every turn of the loop and a turn is a
    /// hundred nanoseconds, so an ungated sweep would draw a fresh sample ten
    /// thousand times per millisecond and spend a real share of the shard on
    /// looking for keys that cannot have died since the last look. Nothing in a
    /// database changes fast enough to be worth asking about more often than the
    /// clock can tell the difference, and the clock here is milliseconds.
    ///
    /// A millisecond is also far finer than Redis, whose slow cycle runs at ten
    /// hertz, so this is not the thing that decides how promptly memory comes
    /// back. What it decides is that an idle server sweeps a thousand times a
    /// second rather than a million.
    pub fn expire_slice(&self, budget: usize) -> usize {
        let now = self.clock.now_ms();
        if now == self.expire_ms.load(Relaxed) {
            return 0;
        }
        self.expire_ms.store(now, Relaxed);
        self.expire_step(budget)
    }

    /// Sweep dead keys out of the databases, spending at most `budget` looks.
    ///
    /// Answers what it spent, so the caller can charge its maintenance slice for
    /// it. See [`yo_kv::expiry`] for why the budget is in keys looked at.
    ///
    /// Round robin from its own cursor, and every database gets offered whatever
    /// is left of the budget rather than a sixteenth of it each, so a server on
    /// database zero only, which is nearly every server, spends the whole slice
    /// where the keys are. The fifteen empty ones cost a comparison apiece
    /// because a database with no key carrying a deadline says so without
    /// drawing anything.
    ///
    /// The cursor moves to the database after whichever one did the work, so two
    /// busy databases take turns instead of the lower numbered one starving the
    /// other.
    pub fn expire_step(&self, budget: usize) -> usize {
        let mut spent = 0;
        let from = self.expire_db.load(Relaxed);
        for turn in 0..self.slots() {
            if spent >= budget {
                break;
            }
            let i = (from + turn) % self.slots();
            let c = self.slot(i).expire_cycle(budget - spent);
            spent += c.examined;
            if c.expired > 0 {
                self.expire_db.store((i + 1) % self.slots(), Relaxed);
                self.mine().note(1u64 << self.slot_db(i));
            }
        }
        spent
    }

    /// One slice of compaction for a server that is over its limit.
    ///
    /// Takes the databases in the same order [`Server::compact_step`] does and
    /// stops at the first one that had something to move, and it asks with the
    /// ratios off. See [`Keyspace::compact_hard`] for what that changes.
    fn compact_hard_step(&self) -> Option<usize> {
        let from = self.next_db.load(Relaxed);
        for turn in 0..self.slots() {
            let i = (from + turn) % self.slots();
            if let Some(moved) = self.slot(i).compact_hard() {
                self.next_db.store((i + 1) % self.slots(), Relaxed);
                return Some(moved);
            }
        }
        None
    }

    /// Take what every thread has marked and add it to the turn's own mask.
    ///
    /// The mask the turn works from is its own and not a shared one, because a
    /// mask it read in place and then cleared a bit of would be a mask that lost
    /// whatever another thread marked in between. A swap cannot lose a mark: a
    /// thread that ors while the swap happens either gets its bit in before the
    /// swap or leaves it there afterwards, and the second one costs one look at
    /// a database the turn has already been through.
    fn collect_marks(&self) {
        let mut marked = 0;
        for thread in &self.locals {
            marked |= thread.dirty.swap(0, Relaxed);
        }
        self.mine().note(marked);
    }

    /// Give one database's dead space back, if any database has enough of it to
    /// be worth the move. `None` when no database had a candidate.
    ///
    /// Once per batch, next to the clock. Overwriting a key writes a new record
    /// and counts the old one dead, so without this a server holds everything
    /// it has ever written: 400000 sets over 100000 keys measured at 742 bytes
    /// a key against Redis at 144 for the same load, and the whole difference
    /// was dead records nothing ever came back for.
    ///
    /// At most one segment moves per call and the search starts one database
    /// further along each time, so the cost of asking is a comparison per
    /// database and the cost of acting is bounded by a segment.
    pub fn compact_step(&self) -> Option<usize> {
        self.collect_marks();
        let mine = self.mine();
        let from = self.next_db.load(Relaxed);
        for turn in 0..self.slots() {
            let i = (from + turn) % self.slots();
            // Nothing has run against this database since it last said it had
            // nothing to collect, so it still has nothing to collect and the
            // line it lives on stays where it is.
            let at = self.slot_db(i);
            if !mine.wanted(at) {
                continue;
            }
            if let Some(moved) = self.slot(i).compact_step() {
                self.next_db.store((i + 1) % self.slots(), Relaxed);
                return Some(moved);
            }
            // Only once every stripe of the database has said it has nothing,
            // since the bit is per database and one stripe answering for all of
            // them would stop the others being asked at all.
            if i % self.width == self.width - 1 {
                mine.done(at);
            }
        }
        None
    }
}

impl Default for Server {
    fn default() -> Server {
        Server::new()
    }
}

/// What one connection has chosen.
pub struct Session {
    db: usize,
    id: u64,
    name: Vec<u8>,
    /// The `HIMPORT` fieldsets this connection has prepared.
    ///
    /// Connection state and not keyspace state, which is the reference's design
    /// and not a shortcut: a fieldset is invisible to every other connection and
    /// the keys built from one outlive it.
    sets: himport::Fieldsets,
}

impl Session {
    /// A new connection, on database zero with no name.
    #[must_use]
    pub fn new(id: u64) -> Session {
        Session {
            db: 0,
            id,
            name: Vec::new(),
            sets: himport::Fieldsets::default(),
        }
    }

    /// The connection id, which `HELLO` reports and `CLIENT` will.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Which database this connection is working in.
    #[must_use]
    pub const fn db(&self) -> usize {
        self.db
    }

    /// The name the client gave itself, empty if it gave none.
    #[must_use]
    pub fn name(&self) -> &[u8] {
        &self.name
    }

    /// Put everything back the way it was when the connection was opened.
    ///
    /// The protocol is not here because it is not here: it lives in the reply
    /// buffer, and `RESET` sets it back there.
    pub fn reset(&mut self) {
        self.db = 0;
        self.name.clear();
        // `SELECT` leaves these alone and `RESET` does not, both checked
        // against 8.10.1, which is the one pair of answers you could not guess
        // from what the command is for.
        self.sets.clear();
    }

    /// Record the name from `HELLO ... SETNAME`.
    fn set_name(&mut self, name: &[u8]) {
        yo_alloc::allow(|| {
            self.name.clear();
            self.name.extend_from_slice(name);
        });
    }
}

/// Run one command and write its reply.
///
/// The name is looked up and the arity is checked here, once, so that no body
/// has to. Everything after that is the command's own.
pub fn execute(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Flow {
    // The decoder never produces a command with no name. If one ever arrives,
    // it is not something to answer.
    if args.is_empty() {
        return Flow::Continue;
    }
    resolved(server, session, lookup(args.name()), args, out)
}

/// The same, for a caller that has already found the command.
///
/// The engine frames a command before it runs it, and between those two it also
/// asks which key the command touches so the record can be prefetched. That is
/// two more chances to look the name up, and looking it up three times to run it
/// once is three times the cost of the cheapest thing in the path. So the engine
/// resolves the name where it frames the command, carries the answer on the
/// framed command, and both the other two take it from there.
///
/// `spec` is `None` for a name that is not a command, which is the same thing
/// [`lookup`] says and lands in the same reply.
pub fn resolved(
    server: &Server,
    session: &mut Session,
    spec: Option<&'static Spec>,
    args: Args<'_>,
    out: &mut Out,
) -> Flow {
    if args.is_empty() {
        return Flow::Continue;
    }
    server.mine().stats.commands.bump();

    let Some(spec) = spec else {
        write_error(out, &args::unknown_command(args));
        return Flow::Continue;
    };
    if !arity_ok(spec, args.len()) {
        server.mine().cmdstats.at(spec).rejected.bump();
        write_error(out, &args::wrong_arity(spec.name));
        return Flow::Continue;
    }

    // The limit first, so a server with no `maxmemory`, which is the default and
    // is nearly all of them, pays one comparison against a field that is already
    // warm. Every command and not only the writes, because that is where Redis
    // puts it: making room is the server's job whatever the client asked for,
    // and the flag only decides who gets told no when there is no room to make.
    //
    // The flag is Redis's own `denyoom` and the list of commands carrying it is
    // Redis's list, so a command that only frees is let through with nothing
    // left, which is what lets a client dig itself out with `DEL`.
    if server.maxmemory() != 0 && !server.make_room() && spec.flags.contains(&"denyoom") {
        server.mine().cmdstats.at(spec).rejected.bump();
        out.error_line(b"OOM ", OOM);
        return Flow::Continue;
    }

    // Which databases the maintenance turn after this batch has to ask. Marked
    // for every command and not only for the writes, because a read can make
    // garbage too: a `GET` on a key whose expiry has passed reaps it, and the
    // record it dropped is exactly the kind of thing the collector is for.
    // `COPY`, `SWAPDB` and `FLUSHALL` reach a database nobody selected, so the
    // two groups that hold them mark all of them rather than the session's.
    server.mine().mark(match spec.group {
        "string" | "bitmap" | "hyperloglog" | "geo" | "set" | "hash" | "list" | "zset"
        | "array" | "stream" | "bloom" | "cuckoo" | "cms" | "topk" | "tdigest" | "ts" => {
            1u64 << session.db
        }
        _ => ALL_DATABASES,
    });

    let mark = out.len();
    // Before the group, because the five that block are list commands and would
    // otherwise land in `lists`, which is handed one database and nothing that
    // could park a client. The flag is the right thing to branch on rather than
    // a list of names: it is what `COMMAND INFO` reports about exactly these
    // commands, and the sorted set and stream ones that arrive later carry it
    // too.
    let done = if spec.flags.contains(&"blocking") {
        blocking::execute(server, session, spec, args, out)
    } else {
        match spec.group {
            "string" => {
                let db = session.db;
                strings::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // Its own group and its own file, and the same values underneath:
            // a bitmap is a string, so `STRLEN` on one answers and `SETBIT` on
            // something a `SET` left behind works.
            "bitmap" => {
                let db = session.db;
                bits::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // The same again: a sketch is a string with a documented layout, so
            // `GET` hands one to a client and `SET` takes it back.
            "hyperloglog" => {
                let db = session.db;
                hll::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "set" => {
                let db = session.db;
                sets::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // The one hash command whose state is not in the keyspace. A
            // fieldset belongs to the connection, so this is handed the session
            // as well as the database, the same exception `MIGRATE` gets in the
            // keyspace group for the socket it keeps.
            "hash" if spec.name == "himport" => {
                let db = session.db;
                himport::execute(&server.dbs[db], &mut session.sets, args, out)
                    .map(|()| Flow::Continue)
            }
            // The one group that reaches back into the server after it has
            // written its reply, because a hash is what a search index is
            // made of. What comes back is what the indexes have to be told,
            // which is not the same as whether the command was a write.
            "hash" => {
                let db = session.db;
                let changed = hashes::execute(&server.dbs[db], spec, args, out);
                changed.map(|changed| {
                    indexing::changed(server, db, args.get(1), changed);
                    Flow::Continue
                })
            }
            "list" => {
                let db = session.db;
                lists::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "zset" => {
                let db = session.db;
                zsets::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // A geo key is a sorted set and these are sorted set commands with
            // arithmetic on the way in and on the way out, so a client can ZREM
            // a place out of one and ZCARD it to count them.
            "geo" => {
                let db = session.db;
                geo::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "array" => {
                let db = session.db;
                arrays::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "graph" => {
                let db = session.db;
                graph::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // A document under a key, reached by a path. The group is Redis's
            // module surface and the storage is ours, the same trade the vector
            // set group makes.
            "json" => {
                let db = session.db;
                json::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "vector" => {
                let db = session.db;
                vectors::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "bloom" => {
                let db = session.db;
                bloom::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "cuckoo" => {
                let db = session.db;
                cuckoo::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "cms" => {
                let db = session.db;
                cms::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "topk" => {
                let db = session.db;
                topk::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "tdigest" => {
                let db = session.db;
                tdigest::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "ts" => {
                let db = session.db;
                ts::execute(&server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // The clock is read before the database is borrowed, because every
            // stream command needs the time and it lives on the server. An
            // `XADD` with no ID, an `XCLAIM` working out what is idle and an
            // `XINFO` reporting it all have to agree about what moment this is.
            "stream" => {
                let db = session.db;
                let now = server.now_ms();
                streams::execute(&server.dbs[db], spec, args, now, out).map(|()| Flow::Continue)
            }
            // The one keyspace command that needs more than the databases,
            // because the socket it talks down is held on the server between
            // commands and not opened again for each one.
            "keyspace" if spec.name == "migrate" => {
                migrate::execute(server, session.db, args, out).map(|()| Flow::Continue)
            }
            // Every database and not the one the session is on, because `COPY` takes
            // a `DB n` and writes into a database nobody selected. The other group
            // that reaches back into the server afterwards, and it hands back a list
            // rather than one answer, because `DEL a b c` is three keys and a rename
            // is two.
            "keyspace" => {
                let mut touched = indexing::Touched::new(server);
                let done =
                    keyspace::execute(&server.dbs, session.db, spec, args, out, &mut touched);
                done.map(|()| {
                    indexing::touched(server, &touched);
                    Flow::Continue
                })
            }
            // No database at all, because an index is not a key. The registry
            // is the whole of what these sixteen commands touch, and then
            // `FT.CREATE` hands back the name it made so the keys that
            // already match its prefix can be read into it. The lock goes
            // before the scan runs, since the scan takes it again for every
            // key it reads.
            "search" if spec.name == "FT.SEARCH" => {
                // The two search commands that read documents, and so the two
                // that need the keyspace as well as the registry. They take and
                // let go of the registry themselves, because they cannot hold
                // that and a stripe at the same time.
                search::find(server, session.db, args, out).map(|()| Flow::Continue)
            }
            "search" if spec.name == "FT.AGGREGATE" => {
                search::roll(server, session.db, args, out).map(|()| Flow::Continue)
            }
            "search" => {
                let db = session.db;
                let made = search::execute(&mut server.search.lock(), db, spec, args, out);
                made.map(|made| {
                    if let Some(name) = made {
                        indexing::scan(server, db, name);
                    }
                    Flow::Continue
                })
            }
            "scripting" => scripting::execute(spec, args, out).map(|()| Flow::Continue),
            _ => server::execute(server, session, spec, args, out),
        }
    };
    let flow = match done {
        Ok(flow) => flow,
        Err(e) => {
            out.truncate(mark);
            write_error(out, &e);
            Flow::Continue
        }
    };

    // Counted here and not before the call, which is where Redis counts it, so
    // that `INFO commandstats` leaves out the `INFO` that asked for it in the
    // same way theirs does.
    //
    // Failure is read off the reply rather than off the `Result`, because the
    // two are not the same set. A command that ran out of arguments comes back
    // as an `Err` and a command that was sent the wrong password writes its own
    // error line and comes back `Ok`, and both of those are a call that failed.
    // The first byte at the mark is what a client would branch on, and it is `-`
    // for an error on either protocol and `!` for RESP3's long form.
    let row = server.mine().cmdstats.at(spec);
    row.calls.bump();
    if matches!(out.as_slice().get(mark), Some(b'-' | b'!')) {
        row.failed.bump();
    }
    flow
}

/// The error line for an error value.
///
/// The prefix is what a client branches on, and there are three of them:
/// `WRONGTYPE` for a command sent at the wrong kind of value, `INVALIDOBJ` for a
/// HyperLogLog whose opcodes do not add up, and `ERR` for everything else. The three errors that need a different one,
/// `NOPROTO`, `WRONGPASS` and `OOM`, are written where they are decided rather
/// than routed through here. `OOM` is not a [`Code`] of its own because
/// [`Code::Full`] already covers the string that is too long for
/// `proto-max-bulk-len`, and that one goes out as `ERR` on a real server.
fn write_error(out: &mut Out, e: &Error) {
    let prefix: &[u8] = match e.code() {
        Code::WrongType => b"WRONGTYPE ",
        // Only the HyperLogLog commands answer this one, and the prefix is the
        // sentence a client branches on to tell a sketch it cannot read from a
        // sketch it sent wrong.
        Code::Corrupt => b"INVALIDOBJ ",
        _ => b"ERR ",
    };
    out.error_line(prefix, e.message().as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{Limits, Proto};
    use crate::request::Argv;

    /// Build the wire bytes for a command.
    ///
    /// Tests go through the codec rather than around it, so an argument in a
    /// test is the same borrowed slice a connection produces.
    pub(crate) fn encode(parts: &[&[u8]]) -> Vec<u8> {
        let mut wire = format!("*{}\r\n", parts.len()).into_bytes();
        for p in parts {
            wire.extend_from_slice(format!("${}\r\n", p.len()).as_bytes());
            wire.extend_from_slice(p);
            wire.extend_from_slice(b"\r\n");
        }
        wire
    }

    /// A server, a connection and a buffer, driven the way the reactor will.
    struct Fixture {
        server: Server,
        session: Session,
        argv: Argv,
        out: Out,
    }

    impl Fixture {
        fn new() -> Fixture {
            Fixture::on(Server::new())
        }

        /// The same, on a server whose databases are cut into `width` stripes.
        fn striped(width: usize) -> Fixture {
            Fixture::on(Server::with_width(width))
        }

        fn on(server: Server) -> Fixture {
            Fixture {
                server,
                session: Session::new(7),
                argv: Argv::new(),
                out: Out::new(Proto::Resp2),
            }
        }

        /// Run one command and answer with the bytes it wrote.
        fn run(&mut self, parts: &[&[u8]]) -> String {
            self.flow(parts).1
        }

        /// Run one command and answer with the bytes exactly as written.
        ///
        /// [`Fixture::run`] goes through `from_utf8_lossy`, which is fine for
        /// every reply that is text and destroys a `DUMP` payload, since a
        /// payload is arbitrary bytes and a checksum on the end of them.
        fn raw(&mut self, parts: &[&[u8]]) -> Vec<u8> {
            let wire = encode(parts);
            self.argv.decode(&wire, &Limits::default()).unwrap();
            self.out.clear();
            execute(
                &self.server,
                &mut self.session,
                Args::new(&self.argv, &wire),
                &mut self.out,
            );
            self.out.as_slice().to_vec()
        }

        /// Move every clock in the server on by `ms`.
        fn advance(&mut self, ms: u64) {
            self.server.advance_clock_ms(ms);
        }

        /// The same, with what the connection should do next.
        fn flow(&mut self, parts: &[&[u8]]) -> (Flow, String) {
            let wire = encode(parts);
            self.argv.decode(&wire, &Limits::default()).unwrap();
            self.out.clear();
            let flow = execute(
                &self.server,
                &mut self.session,
                Args::new(&self.argv, &wire),
                &mut self.out,
            );
            (
                flow,
                String::from_utf8_lossy(self.out.as_slice()).into_owned(),
            )
        }
    }

    /// What a client does all day: write the same keys again and again. Every
    /// one of those writes leaves the previous record behind, so a server that
    /// never compacts holds every version of every key it has ever been sent.
    #[test]
    fn rewriting_the_same_keys_does_not_grow_the_server() {
        let mut f = Fixture::new();
        let val = vec![b'v'; 1024];
        let keys: Vec<Vec<u8>> = (0..64).map(|i| format!("key:{i}").into_bytes()).collect();

        for k in &keys {
            f.run(&[b"SET", k, &val]);
        }
        f.server.compact_step();
        let after_first = f.server.memory_bytes();

        // 64 KiB a pass, five hundred passes, and the same 64 keys at the end
        // of it. Thirty two megabytes written to hold sixty four kilobytes,
        // which is the shape of a real workload and is enough churn to fill
        // sixteen segments if nothing ever comes back.
        for _ in 0..500 {
            for k in &keys {
                f.run(&[b"SET", k, &val]);
            }
            f.server.compact_step();
        }

        assert!(
            f.server.memory_bytes() <= after_first * 2,
            "held {} after five hundred passes against {after_first} after one",
            f.server.memory_bytes()
        );
        assert_eq!(f.run(&[b"DBSIZE"]), format!(":{}\r\n", keys.len()));
        assert_eq!(f.run(&[b"STRLEN", b"key:7"]), ":1024\r\n");
    }

    /// The same churn on a database nobody starts on, either side of a quiet
    /// spell long enough for the maintenance turn to stop asking about it.
    ///
    /// The turn after each batch skips a database that has already said it has
    /// nothing to collect and has not been touched since, which is what keeps a
    /// server whose clients are all on database zero from loading and storing
    /// in the other fifteen every batch to be told no. Two things could go
    /// wrong with that. A database might never be marked at all, so this uses
    /// database nine, which nothing marks by accident. And a database whose
    /// mark was cleared might never get it back, so this drains the collector
    /// until it says there is nothing left, checks the mark really is gone, and
    /// then writes another thirty two megabytes through the same sixty four
    /// keys. If either went wrong the server would hold all of it.
    #[test]
    fn a_database_nobody_started_on_is_still_collected() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SELECT", b"9"]), "+OK\r\n");
        let val = vec![b'v'; 1024];
        let keys: Vec<Vec<u8>> = (0..64).map(|i| format!("key:{i}").into_bytes()).collect();

        for k in &keys {
            f.run(&[b"SET", k, &val]);
        }
        while f.server.compact_step().is_some() {}
        assert!(
            !f.server.mine().wanted(9),
            "database nine was drained and should not be asked again until it is written to"
        );
        let after_first = f.server.memory_bytes();

        for _ in 0..500 {
            for k in &keys {
                f.run(&[b"SET", k, &val]);
            }
            f.server.compact_step();
        }

        assert!(
            f.server.memory_bytes() <= after_first * 2,
            "held {} after five hundred passes against {after_first} after one",
            f.server.memory_bytes()
        );
        assert_eq!(f.run(&[b"DBSIZE"]), format!(":{}\r\n", keys.len()));
        assert_eq!(f.run(&[b"STRLEN", b"key:7"]), ":1024\r\n");
        // And nothing landed anywhere else on the way.
        f.run(&[b"SELECT", b"0"]);
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
    }

    #[test]
    fn a_command_goes_from_bytes_to_bytes() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SET", b"k", b"v"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$1\r\nv\r\n");
        assert_eq!(f.run(&[b"GET", b"nosuch"]), "$-1\r\n");
        assert_eq!(f.run(&[b"STRLEN", b"k"]), ":1\r\n");
        // The name is matched whatever case it came in, and so are the options.
        assert_eq!(f.run(&[b"set", b"k", b"v2", b"xx"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$2\r\nv2\r\n");
    }

    #[test]
    fn deleting_counts_keys_removed_and_existing_counts_arguments_matched() {
        let mut f = Fixture::new();
        f.run(&[b"MSET", b"a", b"1", b"b", b"2", b"c", b"3"]);
        // A key named twice exists twice and can only be deleted once, and both
        // of those are Redis's answers rather than tidier ones.
        assert_eq!(f.run(&[b"EXISTS", b"a", b"a", b"nosuch"]), ":2\r\n");
        assert_eq!(f.run(&[b"DEL", b"a", b"a", b"nosuch"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"a"]), ":0\r\n");
        // UNLINK is the same body and reports the same way.
        assert_eq!(f.run(&[b"UNLINK", b"b", b"c"]), ":2\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
    }

    #[test]
    fn type_is_a_simple_string_and_says_none_for_a_key_that_is_not_there() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);
        // A simple string on both protocols, which is unusual: most replies
        // that carry a word are bulk strings.
        assert_eq!(f.run(&[b"TYPE", b"k"]), "+string\r\n");
        assert_eq!(f.run(&[b"TYPE", b"nosuch"]), "+none\r\n");
    }

    #[test]
    fn touch_counts_the_way_exists_counts() {
        let mut f = Fixture::new();
        f.run(&[b"MSET", b"a", b"1", b"b", b"2"]);
        assert_eq!(f.run(&[b"TOUCH", b"a", b"b"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"TOUCH", b"a", b"a"]),
            ":2\r\n",
            "twice counts twice"
        );
        assert_eq!(f.run(&[b"TOUCH", b"a", b"nosuch"]), ":1\r\n");
        assert_eq!(f.run(&[b"TOUCH", b"nosuch"]), ":0\r\n");
    }

    #[test]
    fn a_rename_moves_the_deadline_with_the_value_and_drops_the_one_it_lands_on() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v1", b"EX", b"100"]);
        f.run(&[b"SET", b"b", b"v2", b"EX", b"500"]);

        assert_eq!(f.run(&[b"RENAME", b"a", b"b"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"b"]), "$2\r\nv1\r\n");
        assert_eq!(
            f.run(&[b"TTL", b"b"]),
            ":100\r\n",
            "the source's and not b's"
        );
        assert_eq!(f.run(&[b"EXISTS", b"a"]), ":0\r\n");
    }

    #[test]
    fn a_rename_with_no_source_is_an_error_and_not_a_zero() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"RENAME", b"a", b"b"]), "-ERR no such key\r\n");
        // The source is checked before the destination, so this is the error
        // and not the zero RENAMENX would otherwise answer for a taken name.
        assert_eq!(f.run(&[b"RENAMENX", b"a", b"a"]), "-ERR no such key\r\n");
    }

    #[test]
    fn renamenx_refuses_a_taken_name_including_the_one_it_already_has() {
        let mut f = Fixture::new();
        f.run(&[b"MSET", b"a", b"v1", b"b", b"v2"]);

        assert_eq!(f.run(&[b"RENAMENX", b"a", b"b"]), ":0\r\n");
        assert_eq!(f.run(&[b"GET", b"b"]), "$2\r\nv2\r\n");
        // Renaming onto itself is 0 here and OK for plain RENAME, which is the
        // one call the two disagree about and neither does any work for.
        assert_eq!(f.run(&[b"RENAMENX", b"a", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"RENAME", b"a", b"a"]), "+OK\r\n");
        assert_eq!(f.run(&[b"RENAMENX", b"a", b"c"]), ":1\r\n");
        assert_eq!(f.run(&[b"GET", b"c"]), "$2\r\nv1\r\n");
    }

    #[test]
    fn renaming_a_set_does_not_touch_a_member() {
        let mut f = Fixture::new();
        for i in 0..300 {
            f.run(&[b"SADD", b"s", format!("m{i}").as_bytes()]);
        }
        let before = f.server.memory_bytes();

        assert_eq!(f.run(&[b"RENAME", b"s", b"t"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SCARD", b"t"]), ":300\r\n");
        assert_eq!(f.run(&[b"TYPE", b"t"]), "+set\r\n");
        assert!(
            f.server.memory_bytes().abs_diff(before) < 256,
            "the members were copied: {} against {before}",
            f.server.memory_bytes()
        );
    }

    #[test]
    fn a_copy_is_a_second_value_and_not_a_second_name() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"m1", b"m2"]);

        assert_eq!(f.run(&[b"COPY", b"s", b"t"]), ":1\r\n");
        f.run(&[b"SADD", b"t", b"m3"]);
        assert_eq!(f.run(&[b"SCARD", b"s"]), ":2\r\n", "the original is intact");
        assert_eq!(f.run(&[b"SCARD", b"t"]), ":3\r\n");
    }

    /// Every type a key can hold, copied, because two of them used to panic.
    ///
    /// `COPY` reads the value out of the source through one match on the type
    /// tag, and that match had a catch all at the bottom from back when a set
    /// and a hash were the only bodies. The list and the sorted set landed after
    /// it and nobody came back, so `COPY mylist other` took the shard down. It
    /// is an ordinary command against a type the server supports everywhere
    /// else, so this walks all five rather than the two that were broken: the
    /// point is that the next type cannot land the same way.
    #[test]
    fn every_type_can_be_copied() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v1"]);
        f.run(&[b"SADD", b"set", b"m1"]);
        f.run(&[b"HSET", b"hash", b"f", b"v"]);
        f.run(&[b"RPUSH", b"list", b"a", b"b"]);
        f.run(&[b"ZADD", b"zset", b"1", b"m1"]);

        for name in [
            &b"str"[..],
            &b"set"[..],
            &b"hash"[..],
            &b"list"[..],
            &b"zset"[..],
        ] {
            let dst = [name, b":copy"].concat();
            assert_eq!(
                f.run(&[b"COPY", name, &dst]),
                ":1\r\n",
                "copying {}",
                String::from_utf8_lossy(name)
            );
            assert_eq!(f.run(&[b"TYPE", name]), f.run(&[b"TYPE", &dst]));
        }

        assert_eq!(f.run(&[b"LRANGE", b"list:copy", b"0", b"-1"]), {
            let mut want = String::from("*2\r\n");
            want.push_str("$1\r\na\r\n$1\r\nb\r\n");
            want
        });
        assert_eq!(f.run(&[b"ZSCORE", b"zset:copy", b"m1"]), "$1\r\n1\r\n");

        // And the copy is its own value, not a second name for the source.
        f.run(&[b"RPUSH", b"list:copy", b"c"]);
        assert_eq!(f.run(&[b"LLEN", b"list"]), ":2\r\n");
        assert_eq!(f.run(&[b"LLEN", b"list:copy"]), ":3\r\n");
    }

    #[test]
    fn a_copy_refuses_a_taken_destination_until_it_is_told_it_can_have_it() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v1", b"EX", b"100"]);
        f.run(&[b"SET", b"b", b"v2"]);

        assert_eq!(f.run(&[b"COPY", b"a", b"b"]), ":0\r\n");
        assert_eq!(f.run(&[b"GET", b"b"]), "$2\r\nv2\r\n");
        assert_eq!(f.run(&[b"COPY", b"a", b"b", b"REPLACE"]), ":1\r\n");
        assert_eq!(f.run(&[b"GET", b"b"]), "$2\r\nv1\r\n");
        assert_eq!(f.run(&[b"TTL", b"b"]), ":100\r\n", "the deadline came too");
        assert_eq!(f.run(&[b"COPY", b"nosuch", b"z"]), ":0\r\n");
    }

    #[test]
    fn a_copy_into_another_database_is_a_copy_and_onto_itself_there_is_too() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v1"]);

        // Same key, different database, so this is not the same object and is
        // an ordinary copy. Same key in the same database is the error below.
        assert_eq!(f.run(&[b"COPY", b"a", b"a", b"DB", b"1"]), ":1\r\n");
        f.run(&[b"SELECT", b"1"]);
        assert_eq!(f.run(&[b"GET", b"a"]), "$2\r\nv1\r\n");
        assert_eq!(
            f.run(&[b"COPY", b"a", b"a", b"DB", b"0"]),
            ":0\r\n",
            "taken"
        );
        assert_eq!(
            f.run(&[b"COPY", b"a", b"a", b"DB", b"0", b"REPLACE"]),
            ":1\r\n"
        );
    }

    #[test]
    fn sort_takes_its_options_in_any_order_and_the_last_one_wins() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"l", b"3", b"1", b"2"]);
        assert_eq!(
            f.run(&[b"SORT", b"l"]),
            "*3\r\n$1\r\n1\r\n$1\r\n2\r\n$1\r\n3\r\n"
        );
        // DESC then ASC is ASC, because the only thing ASC does is undo a DESC.
        assert_eq!(
            f.run(&[b"SORT", b"l", b"DESC", b"asc"]),
            "*3\r\n$1\r\n1\r\n$1\r\n2\r\n$1\r\n3\r\n"
        );
        assert_eq!(
            f.run(&[b"sort", b"l", b"LIMIT", b"1", b"1", b"DESC"]),
            "*1\r\n$1\r\n2\r\n"
        );
    }

    #[test]
    fn sort_reads_a_key_per_element_for_by_and_for_get() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"l", b"a", b"b"]);
        f.run(&[b"MSET", b"w_a", b"2", b"w_b", b"1", b"d_b", b"bee"]);
        // `b` weighs less so it comes first, and its `GET` hits where `a`'s
        // misses, which is a nil in the middle of the array and not a short one.
        assert_eq!(
            f.run(&[b"SORT", b"l", b"BY", b"w_*", b"GET", b"#", b"GET", b"d_*"]),
            "*4\r\n$1\r\nb\r\n$3\r\nbee\r\n$1\r\na\r\n$-1\r\n"
        );
    }

    #[test]
    fn sort_store_writes_a_list_and_answers_its_length() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"l", b"3", b"1", b"2"]);
        assert_eq!(f.run(&[b"SORT", b"l", b"STORE", b"out"]), ":3\r\n");
        assert_eq!(f.run(&[b"TYPE", b"out"]), "+list\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"out", b"0", b"-1"]),
            "*3\r\n$1\r\n1\r\n$1\r\n2\r\n$1\r\n3\r\n"
        );
        // An empty result takes the destination with it rather than leaving a
        // list that holds nothing.
        assert_eq!(f.run(&[b"SORT", b"missing", b"STORE", b"out"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"out"]), ":0\r\n");
    }

    #[test]
    fn sort_ro_does_not_know_the_word_store() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"l", b"2", b"1"]);
        assert_eq!(f.run(&[b"SORT_RO", b"l"]), "*2\r\n$1\r\n1\r\n$1\r\n2\r\n");
        assert_eq!(
            f.run(&[b"SORT_RO", b"l", b"STORE", b"d"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"d"]), ":0\r\n");
    }

    #[test]
    fn sort_refuses_what_it_cannot_sort() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SORT", b"nosuchkey"]), "*0\r\n");
        f.run(&[b"SET", b"s", b"x"]);
        assert_eq!(
            f.run(&[b"SORT", b"s"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        f.run(&[b"RPUSH", b"words", b"one", b"two"]);
        assert_eq!(
            f.run(&[b"SORT", b"words"]),
            "-ERR One or more scores can't be converted into double\r\n"
        );
        assert_eq!(
            f.run(&[b"SORT", b"words", b"ALPHA"]),
            "*2\r\n$3\r\none\r\n$3\r\ntwo\r\n"
        );
        assert_eq!(f.run(&[b"SORT", b"words", b"BY"]), "-ERR syntax error\r\n");
    }

    #[test]
    fn move_takes_the_key_out_of_one_database_and_puts_it_in_another() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"RPUSH", b"l", b"a", b"b"]), ":2\r\n");
        assert_eq!(f.run(&[b"MOVE", b"l", b"1"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"l"]), ":0\r\n");
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"l", b"0", b"-1"]),
            "*2\r\n$1\r\na\r\n$1\r\nb\r\n"
        );
        // And back, which proves the body survived the trip rather than being
        // rebuilt from a copy that happened to look the same.
        assert_eq!(f.run(&[b"MOVE", b"l", b"0"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"l"]), ":0\r\n");
    }

    #[test]
    fn move_answers_zero_when_either_end_says_no() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"MOVE", b"nope", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"SET", b"a", b"here"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SET", b"a", b"there"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SELECT", b"0"]), "+OK\r\n");
        // The destination is taken, so nothing moves and the source is still
        // there with what it had.
        assert_eq!(f.run(&[b"MOVE", b"a", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"GET", b"a"]), "$4\r\nhere\r\n");
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"a"]), "$5\r\nthere\r\n");
    }

    #[test]
    fn move_refuses_a_database_that_is_not_one_and_the_one_it_is_on() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"MOVE", b"a", b"0"]),
            "-ERR source and destination objects are the same\r\n"
        );
        assert_eq!(
            f.run(&[b"MOVE", b"a", b"99"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"MOVE", b"a", b"-1"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"MOVE", b"a", b"x"]),
            "-ERR value is not an integer or out of range\r\n"
        );
    }

    #[test]
    fn swapdb_swaps_what_two_connections_would_see() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SET", b"k", b"zero"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SET", b"k", b"one"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SELECT", b"0"]), "+OK\r\n");

        assert_eq!(f.run(&[b"SWAPDB", b"0", b"1"]), "+OK\r\n");
        // Still on database zero, and database zero is a different database.
        assert_eq!(f.run(&[b"GET", b"k"]), "$3\r\none\r\n");
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$4\r\nzero\r\n");
        // A database swapped with itself is fine and changes nothing.
        assert_eq!(f.run(&[b"SWAPDB", b"1", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$4\r\nzero\r\n");
    }

    /// Every database on a server reads the server's clock and not one of its
    /// own. They used to be told the time one at a time and now they share the
    /// reading, so a server that built its databases from a second clock would
    /// answer a deadline worked out against a time nobody had set.
    #[test]
    fn a_wide_server_puts_its_databases_on_its_own_clock() {
        let mut f = Fixture::striped(8);
        f.server.set_clock_ms(1_700_000_000_000);
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"EX", b"100"]), "+OK\r\n");
        assert_eq!(f.run(&[b"EXPIRETIME", b"k"]), ":1700000100\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":100\r\n");
        f.server.set_clock_ms(1_700_000_050_000);
        assert_eq!(f.run(&[b"TTL", b"k"]), ":50\r\n");
    }

    /// The swap is stripe by stripe, so a database cut into more than one
    /// stripe is the case that would catch it exchanging some of the keys and
    /// leaving the rest. Sixteen keys over four stripes is enough that every
    /// stripe has something in it whatever the hashes come out as.
    #[test]
    fn swapdb_swaps_every_stripe_of_a_wide_database() {
        let mut f = Fixture::striped(4);
        for i in 0..16u32 {
            let key = format!("k{i}");
            assert_eq!(f.run(&[b"SET", key.as_bytes(), b"zero"]), "+OK\r\n");
        }
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SET", b"only", b"one"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SELECT", b"0"]), "+OK\r\n");

        assert_eq!(f.run(&[b"SWAPDB", b"0", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":1\r\n");
        assert_eq!(f.run(&[b"GET", b"only"]), "$3\r\none\r\n");
        assert_eq!(f.run(&[b"SELECT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":16\r\n");
        for i in 0..16u32 {
            let key = format!("k{i}");
            assert_eq!(f.run(&[b"GET", key.as_bytes()]), "$4\r\nzero\r\n");
        }
    }

    #[test]
    fn swapdb_says_which_index_it_could_not_read() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"SWAPDB", b"x", b"1"]),
            "-ERR invalid first DB index\r\n"
        );
        assert_eq!(
            f.run(&[b"SWAPDB", b"0", b"y"]),
            "-ERR invalid second DB index\r\n"
        );
        // A number too big to be an index on a server that keeps one in an int
        // is the same complaint, and a plausible one that is not ours is the
        // range complaint instead. The split is Redis's.
        assert_eq!(
            f.run(&[b"SWAPDB", b"99999999999999", b"1"]),
            "-ERR invalid first DB index\r\n"
        );
        assert_eq!(
            f.run(&[b"SWAPDB", b"0", b"99"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"SWAPDB", b"-1", b"0"]),
            "-ERR DB index is out of range\r\n"
        );
    }

    #[test]
    fn wait_answers_zero_replicas_without_waiting() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SET", b"a", b"v"]), "+OK\r\n");
        assert_eq!(f.run(&[b"WAIT", b"0", b"0"]), ":0\r\n");
        // A replica that is never going to arrive, and a timeout that would be
        // a real wait on a server that had one.
        assert_eq!(f.run(&[b"WAIT", b"3", b"1000"]), ":0\r\n");
        // Negative replicas is not an error, because zero is already more than
        // it asked for.
        assert_eq!(f.run(&[b"WAIT", b"-1", b"0"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"WAIT", b"x", b"0"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"WAIT", b"0", b"-1"]),
            "-ERR timeout is negative\r\n"
        );
        assert_eq!(
            f.run(&[b"WAIT", b"0", b"1.5"]),
            "-ERR timeout is not an integer or out of range\r\n"
        );
    }

    #[test]
    fn waitaof_answers_two_zeroes_and_refuses_a_local_wait() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"WAITAOF", b"0", b"0", b"0"]), "*2\r\n:0\r\n:0\r\n");
        assert_eq!(
            f.run(&[b"WAITAOF", b"1", b"0", b"0"]),
            "-ERR WAITAOF cannot be used when numlocal is set but appendonly is disabled.\r\n"
        );
        assert_eq!(
            f.run(&[b"WAITAOF", b"2", b"0", b"0"]),
            "-ERR value is out of range, value must between 0 and 1\r\n"
        );
        assert_eq!(
            f.run(&[b"WAITAOF", b"0", b"-1", b"0"]),
            "-ERR value is out of range, must be positive\r\n"
        );
        // The arguments are all read before the server looks at itself, so a
        // bad timeout beats the append only complaint even with numlocal set.
        assert_eq!(
            f.run(&[b"WAITAOF", b"1", b"0", b"-5"]),
            "-ERR timeout is negative\r\n"
        );
    }

    /// The bytes inside a bulk reply, with the header and the trailing break
    /// taken off. Every `DUMP` test needs this and none of them care how the
    /// length was written.
    fn payload(reply: &[u8]) -> Vec<u8> {
        let head = reply.windows(2).position(|w| w == b"\r\n").unwrap();
        reply[head + 2..reply.len() - 2].to_vec()
    }

    #[test]
    fn a_value_survives_a_dump_and_a_restore() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"hello"]);
        f.run(&[b"RPUSH", b"l", b"a", b"b", b"c"]);
        f.run(&[b"SADD", b"t", b"1", b"2", b"3"]);
        f.run(&[b"SADD", b"u", b"x", b"y"]);
        f.run(&[b"HSET", b"h", b"f", b"1", b"g", b"2"]);
        f.run(&[b"ZADD", b"z", b"1.5", b"a", b"2.5", b"b"]);

        for key in [&b"s"[..], b"l", b"t", b"u", b"h", b"z"] {
            let mut copy = key.to_vec();
            copy.push(b'2');
            let bytes = payload(&f.raw(&[b"DUMP", key]));
            assert_eq!(f.run(&[b"RESTORE", &copy, b"0", &bytes]), "+OK\r\n");
            assert_eq!(f.run(&[b"TYPE", &copy]), f.run(&[b"TYPE", key]));
        }

        assert_eq!(f.run(&[b"GET", b"s2"]), "$5\r\nhello\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"l2", b"0", b"-1"]),
            "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", b"t2"])), ["1", "2", "3"]);
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", b"u2"])), ["x", "y"]);
        assert_eq!(f.run(&[b"HGET", b"h2", b"g"]), "$1\r\n2\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z2", b"b"]), "$3\r\n2.5\r\n");
        // The encoding survives too, since the payload names the plainest legal
        // type and the loader puts the value back on the rung it belongs on.
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"t2"]),
            f.run(&[b"OBJECT", b"ENCODING", b"t"])
        );
    }

    #[test]
    fn a_dumped_hash_keeps_its_field_deadlines() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"keep", b"1", b"go", b"2"]);
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"go"]),
            "*1\r\n:1\r\n"
        );
        let bytes = payload(&f.raw(&[b"DUMP", b"h"]));
        assert_eq!(f.run(&[b"RESTORE", b"h2", b"0", &bytes]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"HTTL", b"h2", b"FIELDS", b"2", b"keep", b"go"]),
            "*2\r\n:-1\r\n:100\r\n"
        );
    }

    #[test]
    fn dump_leaves_the_deadline_behind_and_restore_is_given_a_new_one() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v", b"EX", b"100"]);
        let bytes = payload(&f.raw(&[b"DUMP", b"a"]));
        assert_eq!(f.run(&[b"RESTORE", b"b", b"0", &bytes]), "+OK\r\n");
        assert_eq!(f.run(&[b"TTL", b"b"]), ":-1\r\n");
        assert_eq!(f.run(&[b"RESTORE", b"c", b"5000", &bytes]), "+OK\r\n");
        assert_eq!(f.run(&[b"TTL", b"c"]), ":5\r\n");
        // An absolute deadline that has already gone is not an error. The key is
        // not created and the reply is the same OK a live one gets.
        assert_eq!(
            f.run(&[b"RESTORE", b"d", b"1", &bytes, b"ABSTTL"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"d"]), ":0\r\n");
    }

    #[test]
    fn dump_answers_nothing_for_a_key_that_is_not_there() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"DUMP", b"nope"]), "$-1\r\n");
        f.run(&[b"SET", b"gone", b"v", b"PX", b"10"]);
        f.advance(50);
        assert_eq!(f.run(&[b"DUMP", b"gone"]), "$-1\r\n");
    }

    #[test]
    fn restore_refuses_a_key_that_is_there_unless_it_is_told_to_replace() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"first"]);
        f.run(&[b"SET", b"b", b"second"]);
        let bytes = payload(&f.raw(&[b"DUMP", b"b"]));
        assert_eq!(
            f.run(&[b"RESTORE", b"a", b"0", &bytes]),
            "-BUSYKEY Target key name already exists.\r\n"
        );
        assert_eq!(f.run(&[b"GET", b"a"]), "$5\r\nfirst\r\n");
        assert_eq!(
            f.run(&[b"RESTORE", b"a", b"0", &bytes, b"REPLACE"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"GET", b"a"]), "$6\r\nsecond\r\n");
    }

    /// The busy key comes before the payload, which is not the order the
    /// arguments read in. Whether a key is taken should not depend on whether
    /// the bytes behind it happened to be good.
    #[test]
    fn restore_asks_about_the_key_before_it_looks_at_the_bytes() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v"]);
        assert_eq!(
            f.run(&[b"RESTORE", b"a", b"0", b"rubbish"]),
            "-BUSYKEY Target key name already exists.\r\n"
        );
        // And the options come before even that, so a bad FREQ beats the busy
        // key the same way a bad DB beats a missing source in COPY.
        assert_eq!(
            f.run(&[b"RESTORE", b"a", b"0", b"rubbish", b"FREQ", b"300"]),
            "-ERR Invalid FREQ value, must be >= 0 and <= 255\r\n"
        );
    }

    #[test]
    fn restore_can_tell_a_bad_footer_from_bad_bytes() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"hello"]);
        let good = payload(&f.raw(&[b"DUMP", b"a"]));

        let mut flipped = good.clone();
        flipped[2] ^= 0x40;
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &flipped]),
            "-ERR DUMP payload version or checksum are wrong\r\n"
        );
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", b"short"]),
            "-ERR DUMP payload version or checksum are wrong\r\n"
        );
        // A footer that is right over a body that is not. The type byte says
        // string and there is nothing behind it, so the checksum agrees and the
        // value does not exist.
        let mut truncated = good[..1].to_vec();
        truncated.extend_from_slice(&good[good.len() - 10..good.len() - 8]);
        let crc = yo_common::crc::crc64(0, &truncated);
        truncated.extend_from_slice(&crc.to_le_bytes());
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &truncated]),
            "-ERR Bad data format\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"b"]), ":0\r\n");
    }

    #[test]
    fn restore_checks_the_three_numbers_a_client_can_get_wrong() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v"]);
        let bytes = payload(&f.raw(&[b"DUMP", b"a"]));
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"-1", &bytes]),
            "-ERR Invalid TTL value, must be >= 0\r\n"
        );
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &bytes, b"IDLETIME", b"-1"]),
            "-ERR Invalid IDLETIME value, must be >= 0\r\n"
        );
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &bytes, b"FREQ", b"256"]),
            "-ERR Invalid FREQ value, must be >= 0 and <= 255\r\n"
        );
        // Both are accepted and both are then dropped, which is D-26.
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &bytes, b"IDLETIME", b"90"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"RESTORE", b"c", b"0", &bytes, b"FREQ", b"200", b"REPLACE"]),
            "+OK\r\n"
        );
    }

    /// Neither word is refused for being the wrong one. Each is only accepted
    /// while the other is unset, so the second of the two falls through to the
    /// plain syntax error rather than getting a message of its own.
    #[test]
    fn restore_takes_idletime_or_freq_and_not_both() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"v"]);
        let bytes = payload(&f.raw(&[b"DUMP", b"a"]));
        assert_eq!(
            f.run(&[
                b"RESTORE",
                b"b",
                b"0",
                &bytes,
                b"IDLETIME",
                b"1",
                b"FREQ",
                b"2"
            ]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[
                b"RESTORE",
                b"b",
                b"0",
                &bytes,
                b"FREQ",
                b"2",
                b"IDLETIME",
                b"1"
            ]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &bytes, b"FREQ"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"RESTORE", b"b", b"0", &bytes, b"NOSUCH"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn copy_checks_its_options_before_it_looks_for_anything() {
        let mut f = Fixture::new();
        // No key exists at all, and every one of these is still the option
        // complaint rather than a zero, which is the order a real server uses.
        assert_eq!(
            f.run(&[b"COPY", b"a", b"b", b"DB", b"99"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"COPY", b"a", b"b", b"DB", b"-1"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"COPY", b"a", b"b", b"DB", b"x"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"COPY", b"a", b"b", b"nonsense"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"COPY", b"a", b"a"]),
            "-ERR source and destination objects are the same\r\n"
        );
        // Repeated, reordered and lowercased, and the last DB wins.
        assert_eq!(
            f.run(&[b"COPY", b"a", b"b", b"dB", b"1", b"rEpLaCe", b"db", b"2"]),
            ":0\r\n"
        );
    }

    #[test]
    fn time_is_two_bulk_strings_and_moves() {
        let mut f = Fixture::new();
        let first = f.run(&[b"TIME"]);
        assert!(first.starts_with("*2\r\n$"), "got {first}");
        let parts: Vec<&str> = first.split("\r\n").collect();
        let secs: i64 = parts[2].parse().expect("seconds as decimal text");
        let micros: i64 = parts[4].parse().expect("microseconds as decimal text");
        assert!(secs > 1_700_000_000, "a real wall clock, got {secs}");
        assert!((0..1_000_000).contains(&micros), "got {micros}");
        // The coarse clock the keyspace uses is a cached millisecond that a
        // background tick refreshes, so a TIME built on it would answer the
        // same microsecond twice in a row here.
        assert_ne!(first, f.run(&[b"TIME"]));
    }

    #[test]
    fn a_keyspace_scan_walks_every_key_once() {
        let mut f = Fixture::new();
        for i in 0..500 {
            f.run(&[b"SET", format!("k{i}").as_bytes(), b"v"]);
        }

        let mut seen: Vec<String> = Vec::new();
        let mut cursor = "0".to_owned();
        let mut calls = 0;
        loop {
            let (next, keys) = scan_reply(&f.run(&[b"SCAN", cursor.as_bytes(), b"COUNT", b"32"]));
            seen.extend(keys);
            cursor = next;
            calls += 1;
            assert!(calls < 10_000, "the cursor is not advancing");
            if cursor == "0" {
                break;
            }
        }

        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 500, "every key once and only once");
        // And more than one call to get them, or the COUNT is being ignored and
        // the loop above proved nothing about resuming.
        assert!(calls > 1, "500 keys came back in one batch");
    }

    #[test]
    fn a_scan_narrows_by_pattern_and_by_type() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        f.run(&[b"SADD", b"members", b"a"]);
        f.run(&[b"HSET", b"fields", b"f", b"v"]);

        let all = |f: &mut Fixture, args: &[&[u8]]| {
            let mut out: Vec<String> = Vec::new();
            let mut cursor = "0".to_owned();
            loop {
                let mut line: Vec<&[u8]> = vec![b"SCAN", cursor.as_bytes()];
                line.extend_from_slice(args);
                let (next, keys) = scan_reply(&f.run(&line));
                out.extend(keys);
                cursor = next;
                if cursor == "0" {
                    break;
                }
            }
            out.sort();
            out
        };

        assert_eq!(all(&mut f, &[]), ["fields", "members", "str"]);
        assert_eq!(all(&mut f, &[b"MATCH", b"*e*"]), ["fields", "members"]);
        assert_eq!(all(&mut f, &[b"TYPE", b"set"]), ["members"]);
        // Case insensitive, the same as Redis's own comparison.
        assert_eq!(all(&mut f, &[b"TYPE", b"HASH"]), ["fields"]);
        // A type nothing can hold is not an error, it just matches nothing.
        assert!(all(&mut f, &[b"TYPE", b"list"]).is_empty());
        assert!(all(&mut f, &[b"TYPE", b"banana"]).is_empty());
        // Both filters at once, and they are an and rather than an or.
        assert!(all(&mut f, &[b"MATCH", b"str*", b"TYPE", b"set"]).is_empty());
    }

    #[test]
    fn a_scan_says_what_is_wrong_with_it() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SCAN", b"nope"]), "-ERR invalid cursor\r\n");
        assert_eq!(f.run(&[b"SCAN", b"-1"]), "-ERR invalid cursor\r\n");
        assert_eq!(f.run(&[b"SCAN", b"0", b"MATCH"]), "-ERR syntax error\r\n");
        assert_eq!(
            f.run(&[b"SCAN", b"0", b"COUNT", b"0"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"SCAN", b"0", b"COUNT", b"x"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"SCAN", b"0", b"WAT", b"1"]),
            "-ERR syntax error\r\n"
        );
        // A cursor the client made up is a cursor. It resumes somewhere
        // arbitrary and answers whatever is there, which is what Redis does and
        // is the only behaviour that does not need the server to remember every
        // cursor it has handed out.
        assert!(f.run(&[b"SCAN", b"18446744073709551615"]).starts_with("*2"));
    }

    #[test]
    fn keys_and_randomkey_look_at_the_whole_database() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"KEYS", b"*"]), "*0\r\n");
        assert_eq!(f.run(&[b"RANDOMKEY"]), "$-1\r\n");

        for name in ["one", "two", "three"] {
            f.run(&[b"SET", name.as_bytes(), b"v"]);
        }
        assert_eq!(sorted(&f.run(&[b"KEYS", b"*"])), ["one", "three", "two"]);
        assert_eq!(sorted(&f.run(&[b"KEYS", b"t*"])), ["three", "two"]);
        assert_eq!(f.run(&[b"KEYS", b"nothing"]), "*0\r\n");

        for _ in 0..50 {
            let got = f.run(&[b"RANDOMKEY"]);
            assert!(
                ["$3\r\none\r\n", "$3\r\ntwo\r\n", "$5\r\nthree\r\n"].contains(&got.as_str()),
                "got {got}"
            );
        }
    }

    #[test]
    fn a_walk_does_not_answer_keys_that_have_expired() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"alive", b"v"]);
        f.run(&[b"SET", b"dead", b"v", b"PX", b"1"]);
        f.server.advance_clock_ms(2);
        assert_eq!(
            f.run(&[b"DBSIZE"]),
            ":2\r\n",
            "nothing has collected it yet"
        );

        assert_eq!(f.run(&[b"KEYS", b"*"]), "*1\r\n$5\r\nalive\r\n");
        let (_, keys) = scan_reply(&f.run(&[b"SCAN", b"0", b"COUNT", b"1000"]));
        assert_eq!(keys, ["alive"]);
        for _ in 0..20 {
            assert_eq!(f.run(&[b"RANDOMKEY"]), "$5\r\nalive\r\n");
        }
        // The walk collected it on the way past, which is what makes DBSIZE
        // here answer what Redis answers once its own cycle has been round.
        assert_eq!(f.run(&[b"DBSIZE"]), ":1\r\n");
    }

    #[test]
    fn a_key_deadline_goes_on_and_comes_back_in_all_four_units() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);
        assert_eq!(f.run(&[b"TTL", b"k"]), ":-1\r\n", "there and no deadline");
        assert_eq!(f.run(&[b"TTL", b"nosuch"]), ":-2\r\n", "not there at all");

        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":100\r\n");
        let ms = int(&f.run(&[b"PTTL", b"k"]));
        assert!((99_000..=100_000).contains(&ms), "got {ms}");

        // The absolute pair, derived from the same one number the store kept.
        let at = int(&f.run(&[b"EXPIRETIME", b"k"]));
        let at_ms = int(&f.run(&[b"PEXPIRETIME", b"k"]));
        assert_eq!(at, (at_ms + 500) / 1000);
        assert!(at_ms > 1_700_000_000_000, "an absolute moment, got {at_ms}");

        assert_eq!(f.run(&[b"PERSIST", b"k"]), ":1\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":-1\r\n");
        assert_eq!(
            f.run(&[b"PERSIST", b"k"]),
            ":0\r\n",
            "nothing to take off the second time"
        );
        assert_eq!(f.run(&[b"PERSIST", b"nosuch"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"GET", b"k"]),
            "$1\r\nv\r\n",
            "and the value went through all of that untouched"
        );
    }

    #[test]
    fn every_type_can_be_given_a_deadline_and_it_is_the_same_deadline() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        f.run(&[b"SADD", b"set", b"a", b"b"]);
        f.run(&[b"HSET", b"hash", b"f", b"v"]);

        for key in [b"str".as_slice(), b"set", b"hash"] {
            assert_eq!(f.run(&[b"EXPIRE", key, b"100"]), ":1\r\n");
            assert_eq!(f.run(&[b"TTL", key]), ":100\r\n");
        }
        // The body is not touched by any of that, which is the whole reason the
        // deadline lives in the record and the body lives somewhere else.
        assert_eq!(f.run(&[b"SCARD", b"set"]), ":2\r\n");
        assert_eq!(f.run(&[b"HGET", b"hash", b"f"]), "$1\r\nv\r\n");
        assert_eq!(f.run(&[b"GET", b"str"]), "$1\r\nv\r\n");
    }

    #[test]
    fn a_deadline_that_has_already_gone_deletes_the_key_now() {
        let mut f = Fixture::new();
        for key in [b"a".as_slice(), b"b", b"c", b"d"] {
            f.run(&[b"SET", key, b"v"]);
        }
        // Four ways of naming a moment that has passed, and all four are a
        // delete answering 1 rather than an error. Zero is a moment, minus one
        // is a moment, and the hash field commands refuse the negative one.
        assert_eq!(f.run(&[b"EXPIRE", b"a", b"0"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"b", b"-1"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXPIREAT", b"c", b"1"]), ":1\r\n");
        assert_eq!(f.run(&[b"PEXPIREAT", b"d", b"1"]), ":1\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"EXPIRE", b"a", b"100"]),
            ":0\r\n",
            "and the key really went, so there is nothing to put a deadline on"
        );
    }

    #[test]
    fn the_four_conditions_decide_whether_the_deadline_moves() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);

        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100", b"XX"]), ":0\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":-1\r\n", "and XX left it alone");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100", b"GT"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"EXPIRE", b"k", b"100", b"LT"]),
            ":1\r\n",
            "no deadline reads as infinitely far away, so LT passes where GT fails"
        );

        assert_eq!(f.run(&[b"EXPIRE", b"k", b"50", b"NX"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"50", b"GT"]), ":0\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":100\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"50", b"LT"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"200", b"GT"]), ":1\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":200\r\n");

        // The condition is answered before the past check, so this is a 0 and
        // the key survives. The other order would delete it.
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"0", b"NX"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"0", b"XX"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n", "and XX let it through");
    }

    #[test]
    fn the_conditions_are_a_set_and_not_a_keyword() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);

        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100", b"nx"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"EXPIRE", b"k", b"100", b"nx", b"nx"]),
            ":0\r\n",
            "the same keyword twice means it once, and NX now has a deadline to fail on"
        );

        // XX with LT is the one pair that is not either of them on its own: LT
        // alone would accept a key with no deadline and this does not.
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"200", b"xx", b"gt"]), ":1\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":200\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100", b"gt", b"xx"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100", b"XX", b"LT"]), ":1\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":100\r\n");
        f.run(&[b"PERSIST", b"k"]);
        assert_eq!(
            f.run(&[b"EXPIRE", b"k", b"100", b"XX", b"LT"]),
            ":0\r\n",
            "where LT on its own would have taken it"
        );
        assert_eq!(f.run(&[b"EXPIRE", b"k", b"100", b"LT"]), ":1\r\n");
    }

    #[test]
    fn a_key_is_gone_once_its_moment_passes() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);
        f.run(&[b"EXPIRE", b"k", b"100"]);

        let at = int(&f.run(&[b"PEXPIRETIME", b"k"]));
        f.server.set_clock_ms(at as u64 + 1);
        assert_eq!(f.run(&[b"GET", b"k"]), "$-1\r\n");
        assert_eq!(f.run(&[b"TTL", b"k"]), ":-2\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
    }

    #[test]
    fn the_expiry_commands_refuse_what_a_real_server_refuses() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);
        for (bad, want) in [
            (
                &[b"EXPIRE".as_slice(), b"k", b"soon"][..],
                "-ERR value is not an integer or out of range\r\n",
            ),
            (
                &[b"EXPIRE", b"k", b"100", b"MAYBE"],
                "-ERR Unsupported option MAYBE\r\n",
            ),
            (
                &[b"EXPIRE", b"k", b"100", b"NX", b"XX"],
                "-ERR NX and XX, GT or LT options at the same time are not compatible\r\n",
            ),
            (
                &[b"EXPIRE", b"k", b"100", b"NX", b"GT"],
                "-ERR NX and XX, GT or LT options at the same time are not compatible\r\n",
            ),
            (
                &[b"EXPIRE", b"k", b"100", b"GT", b"LT", b"GT"],
                "-ERR GT and LT options at the same time are not compatible\r\n",
            ),
            // Seconds that overflow when multiplied into milliseconds. Every
            // message names the command it came from.
            (
                &[b"EXPIRE", b"k", b"9223372036854775807"],
                "-ERR invalid expire time in 'expire' command\r\n",
            ),
            (
                &[b"EXPIREAT", b"k", b"9223372036854775807"],
                "-ERR invalid expire time in 'expireat' command\r\n",
            ),
            (
                &[b"PEXPIRE", b"k", b"9223372036854775807"],
                "-ERR invalid expire time in 'pexpire' command\r\n",
            ),
        ] {
            assert_eq!(f.run(bad), want, "for {bad:?}");
        }
        assert_eq!(
            f.run(&[b"TTL", b"k"]),
            ":-1\r\n",
            "and none of those put a deadline on anything"
        );

        // The one of the four that has no arithmetic to overflow. Redis takes
        // it and holds the number as given, and a record here holds forty six
        // bits, so it lands in the year 4199 instead. D-17.
        assert_eq!(
            f.run(&[b"PEXPIREAT", b"k", b"9223372036854775807"]),
            ":1\r\n"
        );
        assert_eq!(f.run(&[b"PEXPIRETIME", b"k"]), ":70368744177663\r\n");
    }

    #[test]
    fn flushing_empties_this_database_or_every_one_of_them() {
        let mut f = Fixture::new();
        f.run(&[b"SELECT", b"0"]);
        f.run(&[b"MSET", b"a", b"1", b"b", b"2"]);
        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"SET", b"c", b"3"]);
        assert_eq!(f.run(&[b"DBSIZE"]), ":1\r\n");
        // ASYNC and SYNC are both taken and neither changes anything, since the
        // keyspace is empty before the OK goes out either way.
        assert_eq!(f.run(&[b"FLUSHDB", b"async"]), "+OK\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        // Only database one was emptied.
        f.run(&[b"SELECT", b"0"]);
        assert_eq!(f.run(&[b"DBSIZE"]), ":2\r\n");
        assert_eq!(f.run(&[b"FLUSHALL", b"SYNC"]), "+OK\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        f.run(&[b"SELECT", b"1"]);
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        // Anything else after the name is a syntax error, and so is a third
        // argument even when the second one is a word we take.
        assert_eq!(f.run(&[b"FLUSHALL", b"nope"]), "-ERR syntax error\r\n");
        assert_eq!(
            f.run(&[b"FLUSHDB", b"sync", b"sync"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn the_script_cache_and_the_library_set_answer_for_being_empty() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SCRIPT", b"FLUSH"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SCRIPT", b"FLUSH", b"async"]), "+OK\r\n");
        assert_eq!(f.run(&[b"FUNCTION", b"FLUSH", b"SYNC"]), "+OK\r\n");
        // Nothing is cached, so nothing is there, one answer per hash asked
        // about.
        assert_eq!(
            f.run(&[b"SCRIPT", b"EXISTS", b"aaaa", b"bbbb"]),
            "*2\r\n:0\r\n:0\r\n"
        );
        assert_eq!(f.run(&[b"FUNCTION", b"LIST"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"FUNCTION", b"LIST", b"LIBRARYNAME", b"x", b"WITHCODE"]),
            "*0\r\n"
        );
        assert_eq!(
            f.run(&[b"FUNCTION", b"DELETE", b"nosuch"]),
            "-ERR Library not found\r\n"
        );

        // Redis's two messages here are its own, one per container, and one of
        // them reads like a typo.
        assert_eq!(
            f.run(&[b"SCRIPT", b"FLUSH", b"nope"]),
            "-ERR SCRIPT FLUSH only support SYNC|ASYNC option\r\n"
        );
        assert_eq!(
            f.run(&[b"FUNCTION", b"FLUSH", b"nope"]),
            "-ERR FUNCTION FLUSH only supports SYNC|ASYNC option\r\n"
        );
        // A second argument after the mode is the generic one instead, because
        // the count is checked before the word is looked at.
        assert_eq!(
            f.run(&[b"FUNCTION", b"FLUSH", b"sync", b"sync"]),
            "-ERR unknown subcommand or wrong number of arguments for 'flush'. Try FUNCTION HELP.\r\n"
        );
        assert_eq!(
            f.run(&[b"FUNCTION", b"LIST", b"bogus"]),
            "-ERR Unknown argument bogus\r\n"
        );
        assert_eq!(
            f.run(&[b"SCRIPT", b"EXISTS"]),
            "-ERR wrong number of arguments for 'script|exists' command\r\n"
        );

        // The ones that need an interpreter are not here, and say so rather
        // than answering OK to a load that loaded nothing.
        assert_eq!(
            f.run(&[b"SCRIPT", b"LOAD", b"return 1"]),
            "-ERR unknown subcommand 'LOAD'. Try SCRIPT HELP.\r\n"
        );
        assert_eq!(
            f.run(&[b"FUNCTION", b"STATS"]),
            "-ERR unknown subcommand 'STATS'. Try FUNCTION HELP.\r\n"
        );
    }

    #[test]
    fn a_counter_is_an_integer_and_not_a_string_of_digits() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"INCR", b"c"]), ":1\r\n");
        assert_eq!(f.run(&[b"INCRBY", b"c", b"41"]), ":42\r\n");
        assert_eq!(f.run(&[b"DECRBY", b"c", b"2"]), ":40\r\n");
        // Read back as a string it is still an integer, written out as digits
        // only because somebody asked for them.
        assert_eq!(f.run(&[b"GET", b"c"]), "$2\r\n40\r\n");
        assert_eq!(f.run(&[b"INCRBYFLOAT", b"c", b"0.5"]), "$4\r\n40.5\r\n");
        // A counter that is not a number is the error the store raises and this
        // layer only spells, which is the whole point of the split.
        f.run(&[b"SET", b"k", b"hello"]);
        assert_eq!(
            f.run(&[b"INCR", b"k"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"INCRBYFLOAT", b"c", b"inf"]),
            "-ERR increment would produce NaN or Infinity\r\n"
        );
    }

    /// Every one of these was read off a running 8.8. They are the answers a
    /// client library's own test suite checks, and the shapes are not
    /// guessable: `DIGEST` is hexadecimal in a bulk string, `MSETEX` is an
    /// integer, `INCREX` is a pair.
    #[test]
    fn the_newer_commands_reply_in_the_shapes_a_real_server_sends() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SET", b"k", b"hello"]), "+OK\r\n");
        // The same digest a real 8.8 answers for the same five bytes, which is
        // what makes `IFDEQ` usable against a mixed deployment.
        assert_eq!(f.run(&[b"DIGEST", b"k"]), "$16\r\n9555e8555c62dcfd\r\n");
        assert_eq!(f.run(&[b"DIGEST", b"nosuch"]), "$-1\r\n");
        assert_eq!(f.run(&[b"MSETEX", b"1", b"a", b"1"]), ":1\r\n");
        assert_eq!(f.run(&[b"MSETEX", b"1", b"a", b"2", b"NX"]), ":0\r\n");
        assert_eq!(f.run(&[b"GET", b"a"]), "$1\r\n1\r\n");
        assert_eq!(f.run(&[b"INCREX", b"n"]), "*2\r\n:1\r\n:1\r\n");
        assert_eq!(
            f.run(&[b"INCREX", b"n", b"BYINT", b"5", b"UBOUND", b"3"]),
            "*2\r\n:1\r\n:0\r\n",
            "a refused increment reports the value it left alone and applied nothing"
        );
        assert_eq!(
            f.run(&[
                b"INCREX",
                b"n",
                b"BYINT",
                b"5",
                b"UBOUND",
                b"3",
                b"SATURATE"
            ]),
            "*2\r\n:3\r\n:2\r\n"
        );
        assert_eq!(f.run(&[b"DELEX", b"a", b"IFEQ", b"2"]), ":0\r\n");
        assert_eq!(f.run(&[b"DELEX", b"a", b"IFEQ", b"1"]), ":1\r\n");
    }

    #[test]
    fn the_same_answers_come_out_in_resp3_spelling() {
        let mut f = Fixture::new();
        assert!(f.run(&[b"HELLO", b"3"]).starts_with("%7\r\n"));
        assert_eq!(f.run(&[b"GET", b"nosuch"]), "_\r\n");
        // A float counter is a double on RESP3 and the digits in a bulk string
        // on RESP2, and `INCRBYFLOAT` is a bulk string on both.
        assert_eq!(
            f.run(&[b"INCREX", b"c", b"BYFLOAT", b"1.5"]),
            "*2\r\n,1.5\r\n,1.5\r\n"
        );
        assert_eq!(f.run(&[b"INCRBYFLOAT", b"f", b"2.5"]), "$3\r\n2.5\r\n");
        // `RESET` puts the protocol back, which is the part that is easy to
        // miss and leaves a pooled connection speaking the wrong one.
        assert_eq!(f.run(&[b"RESET"]), "+RESET\r\n");
        assert_eq!(f.run(&[b"GET", b"nosuch"]), "$-1\r\n");
    }

    #[test]
    fn a_command_nobody_has_heard_of_is_an_error_and_not_a_closed_socket() {
        let mut f = Fixture::new();
        let (flow, reply) = f.flow(&[b"NOPE", b"a", b"b"]);
        assert_eq!(flow, Flow::Continue);
        assert_eq!(
            reply,
            "-ERR unknown command 'NOPE', with args beginning with: 'a' 'b' \r\n"
        );
        // A name with a line ending in it cannot write its own frame into the
        // stream, which is the reason the error writer maps them to spaces.
        let reply = f.run(&[b"NO\r\n+PONG\r\nPE"]);
        assert_eq!(reply.matches("\r\n").count(), 1);
    }

    #[test]
    fn arity_is_checked_before_the_command_is() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"GET"]),
            "-ERR wrong number of arguments for 'get' command\r\n"
        );
        assert_eq!(
            f.run(&[b"MSET", b"k"]),
            "-ERR wrong number of arguments for 'mset' command\r\n"
        );
        // The table says `PING` takes one or more and a real server then
        // refuses three, which is the sort of thing that only shows up against
        // the real thing.
        assert_eq!(
            f.run(&[b"PING", b"a", b"b"]),
            "-ERR wrong number of arguments for 'ping' command\r\n"
        );
        assert_eq!(f.run(&[b"PING"]), "+PONG\r\n");
        assert_eq!(f.run(&[b"PING", b"hi"]), "$2\r\nhi\r\n");
        // `DELEX` takes two or four and nothing between.
        assert_eq!(
            f.run(&[b"DELEX", b"k", b"IFEQ"]),
            "-ERR wrong number of arguments for 'delex' command\r\n"
        );
    }

    /// The option rules, all of them measured against 8.8 rather than read off
    /// the documentation. The surprising one is that `SET` accepts the same
    /// keyword twice and `INCREX` does not.
    #[test]
    fn the_option_combinations_are_the_ones_a_real_server_accepts() {
        let mut f = Fixture::new();
        let syntax = "-ERR syntax error\r\n";
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"NX", b"XX"]), syntax);
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"NX", b"IFEQ", b"a"]), syntax);
        assert_eq!(
            f.run(&[b"SET", b"k", b"v", b"KEEPTTL", b"EX", b"5"]),
            syntax
        );
        assert_eq!(
            f.run(&[b"SET", b"k", b"v", b"EX", b"5", b"PX", b"5"]),
            syntax
        );
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"PERSIST"]), syntax);
        // Twice is fine, and the last one wins.
        assert_eq!(
            f.run(&[b"SET", b"k", b"v", b"EX", b"5", b"EX", b"100"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"XX", b"XX"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"GET", b"GET"]), "$1\r\nv\r\n");
        // `INCREX` refuses what `SET` allows.
        assert_eq!(
            f.run(&[b"INCREX", b"n", b"BYINT", b"1", b"BYINT", b"2"]),
            syntax
        );
        assert_eq!(
            f.run(&[b"INCREX", b"n", b"ENX"]),
            "-ERR ENX flag requires an expiration\r\n"
        );
        assert_eq!(
            f.run(&[b"INCREX", b"n", b"UBOUND", b"abc"]),
            "-ERR UBOUND is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"INCREX", b"n", b"LBOUND", b"10", b"UBOUND", b"5"]),
            "-ERR LBOUND can't be greater than UBOUND\r\n"
        );
        assert_eq!(
            f.run(&[b"LCS", b"a", b"b", b"LEN", b"IDX"]),
            "-ERR If you want both the length and indexes, please just use IDX.\r\n"
        );
    }

    /// Where the expiration rules bite. The one worth the test is `GETEX` on a
    /// key that is not there, which answers null without ever looking at the
    /// expiration it was given.
    #[test]
    fn the_expiry_rules_are_redis_own() {
        let mut f = Fixture::new();
        let bad = "-ERR invalid expire time in 'set' command\r\n";
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"EX", b"0"]), bad);
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"EX", b"-1"]), bad);
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"EXAT", b"0"]), bad);
        assert_eq!(
            f.run(&[b"SET", b"k", b"v", b"EX", b"9999999999999999"]),
            bad
        );
        assert_eq!(
            f.run(&[b"SET", b"k", b"v", b"PX", b"99999999999999999999"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"SETEX", b"k", b"0", b"v"]),
            "-ERR invalid expire time in 'setex' command\r\n"
        );
        assert_eq!(f.run(&[b"GETEX", b"nosuch", b"EX", b"0"]), "$-1\r\n");
        assert_eq!(f.run(&[b"GETEX", b"nosuch", b"EX", b"abc"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"GETEX", b"nosuch", b"KEEPTTL"]),
            "-ERR syntax error\r\n",
            "the option list is still checked before the key is looked up"
        );
        // A deadline in the past is accepted and the key goes with it.
        assert_eq!(f.run(&[b"SET", b"k", b"v"]), "+OK\r\n");
        assert_eq!(f.run(&[b"SET", b"k", b"v", b"EXAT", b"1"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$-1\r\n");
    }

    #[test]
    fn mset_takes_its_pairs_from_the_read_buffer() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"MSET", b"a", b"1", b"b", b"2"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"MGET", b"a", b"b", b"nosuch"]),
            "*3\r\n$1\r\n1\r\n$1\r\n2\r\n$-1\r\n"
        );
        assert_eq!(f.run(&[b"MSETNX", b"b", b"9", b"c", b"3"]), ":0\r\n");
        assert_eq!(f.run(&[b"MSETNX", b"c", b"3", b"d", b"4"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"MSETEX", b"2", b"e", b"5"]),
            "-ERR wrong number of key-value pairs\r\n"
        );
        assert_eq!(
            f.run(&[b"MSETEX", b"0", b"e", b"5"]),
            "-ERR invalid numkeys value\r\n"
        );
        assert_eq!(
            f.run(&[b"MSETEX", b"abc", b"e", b"5"]),
            "-ERR invalid numkeys value\r\n"
        );
    }

    #[test]
    fn lcs_answers_the_length_the_string_and_the_runs() {
        let mut f = Fixture::new();
        f.run(&[b"MSET", b"a", b"ohmytext", b"b", b"mynewtext"]);
        assert_eq!(f.run(&[b"LCS", b"a", b"b"]), "$6\r\nmytext\r\n");
        assert_eq!(f.run(&[b"LCS", b"a", b"b", b"LEN"]), ":6\r\n");
        assert_eq!(
            f.run(&[b"LCS", b"a", b"b", b"IDX", b"MINMATCHLEN", b"4"]),
            "*4\r\n$7\r\nmatches\r\n*1\r\n*2\r\n*2\r\n:4\r\n:7\r\n*2\r\n:5\r\n:8\r\n$3\r\nlen\r\n:6\r\n"
        );
        // Without `IDX` the two options that only mean something with it are
        // accepted and ignored, which is what a real server does.
        assert_eq!(
            f.run(&[b"LCS", b"a", b"b", b"MINMATCHLEN", b"4", b"WITHMATCHLEN"]),
            "$6\r\nmytext\r\n"
        );
    }

    #[test]
    fn select_moves_the_connection_and_the_databases_stay_apart() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"zero"]);
        assert_eq!(f.run(&[b"SELECT", b"4"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$-1\r\n");
        f.run(&[b"SET", b"k", b"four"]);
        assert_eq!(f.run(&[b"SELECT", b"0"]), "+OK\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$4\r\nzero\r\n");
        assert_eq!(
            f.run(&[b"SELECT", b"99"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"SELECT", b"-1"]),
            "-ERR DB index is out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"SELECT", b"abc"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        // `RESET` brings it back to zero.
        f.run(&[b"SELECT", b"4"]);
        f.run(&[b"RESET"]);
        assert_eq!(f.run(&[b"GET", b"k"]), "$4\r\nzero\r\n");
    }

    #[test]
    fn hello_agrees_on_a_protocol_and_refuses_the_ones_that_do_not_exist() {
        let mut f = Fixture::new();
        let reply = f.run(&[b"HELLO"]);
        assert!(reply.starts_with("*14\r\n"), "{reply}");
        assert!(reply.contains("$5\r\nredis\r\n"), "{reply}");
        assert!(reply.contains("$5\r\n8.8.0\r\n"), "{reply}");
        assert!(
            reply.contains(":7\r\n"),
            "the connection id is in there: {reply}"
        );
        assert_eq!(
            f.run(&[b"HELLO", b"4"]),
            "-NOPROTO unsupported protocol version\r\n"
        );
        assert_eq!(
            f.run(&[b"HELLO", b"abc"]),
            "-ERR Protocol version is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"HELLO", b"3", b"SETNAME"]),
            "-ERR Syntax error in HELLO option 'SETNAME'\r\n"
        );
        assert!(
            f.run(&[b"HELLO", b"3", b"SETNAME", b"bob"])
                .starts_with("%7\r\n")
        );
        assert_eq!(f.session.name(), b"bob");
        f.run(&[b"RESET"]);
        assert_eq!(f.session.name(), b"");
    }

    #[test]
    fn command_describes_this_server_in_the_shape_a_driver_reads() {
        let mut f = Fixture::new();
        let count = format!(":{}\r\n", COMMANDS.len());
        assert_eq!(f.run(&[b"COMMAND", b"COUNT"]), count);
        let info = f.run(&[b"COMMAND", b"INFO", b"get"]);
        assert_eq!(
            info,
            "*1\r\n*10\r\n$3\r\nget\r\n:2\r\n*2\r\n+readonly\r\n+fast\r\n:1\r\n:1\r\n:1\r\n\
             *3\r\n+@read\r\n+@string\r\n+@fast\r\n*0\r\n*0\r\n*0\r\n"
        );
        // A null in the list, and the plain one: `$-1` and not `*-1`.
        assert_eq!(f.run(&[b"COMMAND", b"INFO", b"nosuch"]), "*1\r\n$-1\r\n");
        assert_eq!(
            f.run(&[b"COMMAND", b"LIST", b"FILTERBY", b"PATTERN", b"getr*"]),
            "*1\r\n$8\r\ngetrange\r\n"
        );
        assert_eq!(
            f.run(&[b"COMMAND", b"NOPE"]),
            "-ERR unknown subcommand 'NOPE'. Try COMMAND HELP.\r\n"
        );
    }

    /// A cluster aware client asks this question and then routes on the
    /// answer, so `MSETEX`, whose keys are not where the table says, is the one
    /// that matters.
    #[test]
    fn command_getkeys_finds_the_keys_including_the_hidden_ones() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"COMMAND", b"GETKEYS", b"get", b"k"]),
            "*1\r\n$1\r\nk\r\n"
        );
        assert_eq!(
            f.run(&[b"COMMAND", b"GETKEYS", b"mset", b"a", b"1", b"b", b"2"]),
            "*2\r\n$1\r\na\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[
                b"COMMAND", b"GETKEYS", b"msetex", b"2", b"a", b"1", b"b", b"2"
            ]),
            "*2\r\n$1\r\na\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[b"COMMAND", b"GETKEYS", b"ping"]),
            "-ERR The command has no key arguments\r\n"
        );
        assert_eq!(
            f.run(&[b"COMMAND", b"GETKEYS", b"set"]),
            "-ERR Invalid number of arguments specified for command\r\n"
        );
    }

    #[test]
    fn config_answers_what_it_can_and_refuses_what_it_cannot() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"maxmemory"]),
            "*2\r\n$9\r\nmaxmemory\r\n$1\r\n0\r\n"
        );
        // A pattern matches more than one, and a setting two patterns both ask
        // for is still sent once.
        let both = f.run(&[b"CONFIG", b"GET", b"maxmemory*", b"maxmemory"]);
        assert!(both.starts_with("*6\r\n"), "{both}");
        assert_eq!(f.run(&[b"CONFIG", b"GET", b"nosuch"]), "*0\r\n");
        assert_eq!(f.run(&[b"CONFIG", b"SET", b"appendonly", b"no"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"appendonly", b"yes"]),
            "-ERR CONFIG SET failed (possibly related to argument 'appendonly') - can't set immutable config\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"nosuch", b"1"]),
            "-ERR Unknown option or number of arguments for CONFIG SET - 'nosuch'\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"GET"]),
            "-ERR wrong number of arguments for 'config|get' command\r\n"
        );
        // Too few arguments and an odd number of them are different
        // complaints, which is the sort of thing only the real server tells
        // you.
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"appendonly"]),
            "-ERR wrong number of arguments for 'config|set' command\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"appendonly", b"no", b"maxmemory"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(f.run(&[b"CONFIG", b"RESETSTAT"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"CONFIG", b"REWRITE"]),
            "-ERR The server is running without a config file\r\n"
        );
    }

    #[test]
    fn the_eviction_policy_reads_back_what_was_written_to_it() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"maxmemory-policy"]),
            "*2\r\n$16\r\nmaxmemory-policy\r\n$10\r\nnoeviction\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"AllKeys-LFU"]),
            "+OK\r\n",
            "the name is matched without regard to case, like every other one"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"maxmemory-policy"]),
            "*2\r\n$16\r\nmaxmemory-policy\r\n$11\r\nallkeys-lfu\r\n"
        );
        // And INFO agrees with CONFIG, which it did not when it was a literal.
        assert!(
            f.run(&[b"INFO", b"memory"])
                .contains("maxmemory_policy:allkeys-lfu"),
            "INFO and CONFIG disagree about the policy"
        );
        // The refusal names every legal value in the order the real server's
        // enum table lists them, because a client comparing the message compares
        // the whole string.
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"garbage"]),
            "-ERR CONFIG SET failed (possibly related to argument 'maxmemory-policy') - argument(s) must be one of the following: volatile-lru, volatile-lfu, volatile-random, volatile-ttl, volatile-lrm, allkeys-lru, allkeys-lfu, allkeys-random, allkeys-lrm, noeviction\r\n"
        );
        // A bad pair leaves the good one in the same command alone, and the
        // policy is checked by the same pass that checks the numbers.
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"maxmemory-policy"]),
            "*2\r\n$16\r\nmaxmemory-policy\r\n$11\r\nallkeys-lfu\r\n"
        );
        f.run(&[
            b"CONFIG",
            b"SET",
            b"hash-max-listpack-entries",
            b"7",
            b"maxmemory-policy",
            b"nonsense",
        ]);
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-listpack-entries"]),
            "*2\r\n$25\r\nhash-max-listpack-entries\r\n$3\r\n512\r\n"
        );
    }

    #[test]
    fn the_three_eviction_numbers_read_back_too() {
        let mut f = Fixture::new();
        for (name, default, set) in [
            ("maxmemory-samples", "5", "12"),
            ("lfu-log-factor", "10", "3"),
            ("lfu-decay-time", "1", "60"),
        ] {
            let get = || {
                format!(
                    "*2\r\n${}\r\n{name}\r\n${}\r\n{default}\r\n",
                    name.len(),
                    default.len()
                )
            };
            assert_eq!(f.run(&[b"CONFIG", b"GET", name.as_bytes()]), get());
            assert_eq!(
                f.run(&[b"CONFIG", b"SET", name.as_bytes(), set.as_bytes()]),
                "+OK\r\n"
            );
            assert_eq!(
                f.run(&[b"CONFIG", b"GET", name.as_bytes()]),
                format!(
                    "*2\r\n${}\r\n{name}\r\n${}\r\n{set}\r\n",
                    name.len(),
                    set.len()
                )
            );
            // A number that is not a number is refused with the same sentence
            // every other number gets, which names the setting the client typed.
            assert_eq!(
                f.run(&[b"CONFIG", b"SET", name.as_bytes(), b"soon"]),
                format!(
                    "-ERR CONFIG SET failed (possibly related to argument '{name}') - argument couldn't be parsed into an integer\r\n"
                )
            );
        }
    }

    #[test]
    fn the_memory_limit_reads_back_in_bytes_whatever_the_unit_was() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"maxmemory"]),
            "*2\r\n$9\r\nmaxmemory\r\n$1\r\n0\r\n",
            "no limit is the default"
        );
        // The pairing is Redis's and it is a trap: the bare letter is a power of
        // ten and the one with the b is a power of two.
        for (typed, bytes) in [
            (&b"1024"[..], "1024"),
            (b"1k", "1000"),
            (b"1kb", "1024"),
            (b"1M", "1000000"),
            (b"1Mb", "1048576"),
            (b"1gb", "1073741824"),
            (b"100mb", "104857600"),
        ] {
            assert_eq!(f.run(&[b"CONFIG", b"SET", b"maxmemory", typed]), "+OK\r\n");
            assert_eq!(
                f.run(&[b"CONFIG", b"GET", b"maxmemory"]),
                format!("*2\r\n$9\r\nmaxmemory\r\n${}\r\n{bytes}\r\n", bytes.len()),
                "set {}",
                String::from_utf8_lossy(typed)
            );
        }
        assert!(
            f.run(&[b"INFO", b"memory"]).contains("maxmemory:104857600"),
            "the report agrees with the setting"
        );

        // A unit nobody has heard of, and a negative number, which is not a very
        // large one however it is spelled.
        for bad in [&b"1tb"[..], b"-1", b"", b"lots"] {
            assert_eq!(
                f.run(&[b"CONFIG", b"SET", b"maxmemory", bad]),
                "-ERR CONFIG SET failed (possibly related to argument 'maxmemory') - argument must be a memory value\r\n",
                "refused {}",
                String::from_utf8_lossy(bad)
            );
        }
        assert!(
            f.run(&[b"INFO", b"memory"]).contains("maxmemory:104857600"),
            "and the refusal left the old one alone"
        );
    }

    #[test]
    fn a_write_is_refused_when_there_is_no_room_and_nothing_to_evict() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"here", b"already"]);
        // A byte, which is under what an empty server holds, so nothing this
        // command could do would get it under. The default policy is
        // `noeviction`, so nothing is what it does.
        f.run(&[b"CONFIG", b"SET", b"maxmemory", b"1"]);
        assert_eq!(
            f.run(&[b"SET", b"k", b"v"]),
            "-OOM command not allowed when used memory > 'maxmemory'.\r\n"
        );
        assert_eq!(
            f.run(&[b"LPUSH", b"l", b"v"]),
            "-OOM command not allowed when used memory > 'maxmemory'.\r\n"
        );
        // Reading is allowed, and so is the one thing that would help.
        assert_eq!(f.run(&[b"GET", b"here"]), "$7\r\nalready\r\n");
        assert_eq!(f.run(&[b"DEL", b"here"]), ":1\r\n");
        assert!(f.run(&[b"INFO", b"stats"]).contains("evicted_keys:0"));

        // Taking the limit away lets the write through again.
        f.run(&[b"CONFIG", b"SET", b"maxmemory", b"0"]);
        assert_eq!(f.run(&[b"SET", b"k", b"v"]), "+OK\r\n");
    }

    #[test]
    fn an_allkeys_policy_makes_room_instead_of_refusing() {
        let mut f = Fixture::new();
        let val = vec![b'v'; 256];
        for i in 0..24000u32 {
            let k = format!("key:{i:08}");
            f.run(&[b"SET", k.as_bytes(), &val]);
        }
        let full = f.server.memory_bytes();
        assert!(
            full > 3 * 1024 * 1024,
            "the arena is several segments: {full}"
        );

        // Two megabytes under what it is holding, which is one segment's worth,
        // so getting there means giving a whole segment back and not just
        // dropping a few records.
        let limit = full - 2 * 1024 * 1024;
        f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-lru"]);
        f.run(&[
            b"CONFIG",
            b"SET",
            b"maxmemory",
            limit.to_string().as_bytes(),
        ]);

        // Writes keep working the whole way down. The budget means one command
        // does not do it all, so this runs until the server has settled and
        // checks that nothing was refused on the way.
        for i in 0..2000u32 {
            let k = format!("new:{i:08}");
            assert_eq!(
                f.run(&[b"SET", k.as_bytes(), &val]),
                "+OK\r\n",
                "write {i} was refused"
            );
            f.server.refresh_memory();
            if f.server.memory_bytes() <= limit {
                break;
            }
        }
        assert!(
            f.server.memory_bytes() <= limit,
            "it never got under: {} against {limit}",
            f.server.memory_bytes()
        );
        let info = f.run(&[b"INFO", b"stats"]);
        assert!(!info.contains("evicted_keys:0"), "{info}");
        assert!(
            f.run(&[b"DBSIZE"]) != ":0\r\n",
            "and it did not empty the database to get there"
        );
    }

    #[test]
    fn the_running_total_and_the_walk_agree_on_a_mixed_keyspace() {
        // The limit is judged against a number kept as the collections move,
        // rather than found by asking all of them, and the two have to be the
        // same number or the limit is enforced against a fiction. This does the
        // things that move it, which is growing a collection, shrinking one,
        // changing its representation, deleting it and reusing its slot, across
        // all five types, and checks the two against each other as it goes.
        let mut f = Fixture::new();
        f.run(&[b"CONFIG", b"SET", b"maxmemory", b"1gb"]);
        let big = vec![b'v'; 200];

        for i in 0..400u32 {
            let n = i.to_string();
            let n = n.as_bytes();
            f.run(&[b"SADD", b"s", n]);
            f.run(&[b"SADD", b"s2", &big]);
            f.run(&[b"HSET", b"h", n, &big]);
            f.run(&[b"RPUSH", b"l", &big]);
            f.run(&[b"ZADD", b"z", n, n]);
            f.run(&[b"ARSET", b"a", n, &big]);
            if i % 7 == 0 {
                f.run(&[b"SREM", b"s", n]);
                f.run(&[b"HDEL", b"h", n]);
                f.run(&[b"LPOP", b"l"]);
                f.run(&[b"ZREM", b"z", n]);
                f.run(&[b"ARDEL", b"a", n]);
            }
            if i % 53 == 0 {
                // Every type deleted and made again, so a slot goes on the free
                // list and comes back holding something else.
                f.run(&[b"DEL", b"s2"]);
            }
            assert_eq!(
                f.server.settled_memory(),
                f.server.memory_bytes(),
                "after round {i}"
            );
        }

        // The run has to have built something, or the two numbers agreeing is
        // two zeroes agreeing.
        assert_eq!(f.run(&[b"DBSIZE"]), ":6\r\n");
        assert!(
            f.server.memory_bytes() > 512 * 1024,
            "{}",
            f.server.memory_bytes()
        );

        // And it survives the collections going away entirely.
        f.run(&[b"FLUSHALL"]);
        assert_eq!(f.server.settled_memory(), f.server.memory_bytes());
    }

    #[test]
    fn taking_the_limit_away_stops_the_counting_and_putting_it_back_starts_again() {
        // A server with no limit does not keep the running total, so setting a
        // limit on a database that is already full has to start it from a walk.
        // If it did not, the first reading would be zero and the server would
        // think it had all the room in the world.
        let mut f = Fixture::new();
        for i in 0..200u32 {
            let n = i.to_string();
            f.run(&[b"SADD", b"s", n.as_bytes()]);
            f.run(&[b"HSET", b"h", n.as_bytes(), b"value"]);
        }
        f.run(&[b"CONFIG", b"SET", b"maxmemory", b"1gb"]);
        assert_eq!(f.server.settled_memory(), f.server.memory_bytes());

        f.run(&[b"CONFIG", b"SET", b"maxmemory", b"0"]);
        for i in 200..400u32 {
            let n = i.to_string();
            f.run(&[b"SADD", b"s", n.as_bytes()]);
        }
        f.run(&[b"CONFIG", b"SET", b"maxmemory", b"1gb"]);
        assert_eq!(
            f.server.settled_memory(),
            f.server.memory_bytes(),
            "the writes it was not watching are in the number it started from"
        );
    }

    #[test]
    fn evicted_keys_and_expired_keys_are_different_numbers() {
        let mut f = Fixture::new();
        // Nothing has been evicted and nothing can be under the default policy,
        // so this stays at zero while the other one moves.
        f.run(&[b"SET", b"gone", b"v", b"PX", b"1"]);
        f.server.advance_clock_ms(20);
        f.run(&[b"GET", b"gone"]);
        let info = f.run(&[b"INFO", b"stats"]);
        assert!(info.contains("expired_keys:1"), "{info}");
        assert!(info.contains("evicted_keys:0"), "{info}");
    }

    #[test]
    fn the_object_subcommands_follow_the_policy() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"v"]);
        // Under the default the clock is kept and the counter is not, and under
        // an LFU policy it is the other way round. Each subcommand refuses on
        // the side where its reading of the three bytes means nothing.
        assert_eq!(f.run(&[b"OBJECT", b"IDLETIME", b"s"]), ":0\r\n");
        assert!(
            f.run(&[b"OBJECT", b"FREQ", b"s"])
                .starts_with("-ERR An LFU maxmemory policy is not selected"),
        );

        f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-lfu"]);
        assert!(
            f.run(&[b"OBJECT", b"IDLETIME", b"s"])
                .starts_with("-ERR An LFU maxmemory policy is selected"),
        );
        // The key was written under a clock policy, so what comes back is that
        // clock read as a counter. It is a number and not an error, which is the
        // point: switching at runtime does not invalidate anything, it only makes
        // the old field mean something else until the key is used again.
        assert!(
            f.run(&[b"OBJECT", b"FREQ", b"s"]).starts_with(':'),
            "FREQ should answer under an LFU policy"
        );
    }

    #[test]
    fn object_says_which_rung_of_the_ladder_a_key_is_on() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"hello"]);
        f.run(&[b"SET", b"n", b"123"]);
        f.run(&[b"SADD", b"si", b"1", b"2", b"3"]);
        f.run(&[b"SADD", b"ss", b"a", b"b"]);
        f.run(&[b"HSET", b"h", b"f", b"v"]);
        for (key, want) in [
            (b"s".as_slice(), "embstr"),
            (b"n", "int"),
            (b"si", "intset"),
            (b"ss", "listpack"),
            (b"h", "listpack"),
        ] {
            let reply = f.run(&[b"OBJECT", b"ENCODING", key]);
            assert_eq!(reply, format!("${}\r\n{want}\r\n", want.len()));
        }

        // A field deadline widens the blob rather than promoting it, and this
        // is the only place a client can see that happen.
        f.run(&[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"f"]);
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"h"]),
            "$10\r\nlistpackex\r\n"
        );

        assert_eq!(f.run(&[b"OBJECT", b"REFCOUNT", b"s"]), ":1\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"IDLETIME", b"s"]), ":0\r\n");
        assert!(f.run(&[b"OBJECT", b"HELP"]).starts_with("*14\r\n+OBJECT "));
    }

    #[test]
    fn object_answers_nil_for_a_key_that_is_not_there() {
        let mut f = Fixture::new();
        for sub in [b"ENCODING".as_slice(), b"REFCOUNT", b"IDLETIME", b"FREQ"] {
            assert_eq!(
                f.run(&[b"OBJECT", sub, b"nokey"]),
                "$-1\r\n",
                "a nil and not an error, which is what 8.10.1 does"
            );
        }
        // And the key is looked up before FREQ has its complaint, so the
        // complaint only reaches a key that exists.
        f.run(&[b"SET", b"s", b"v"]);
        assert!(
            f.run(&[b"OBJECT", b"FREQ", b"s"])
                .starts_with("-ERR An LFU maxmemory policy is not"),
        );
        assert_eq!(
            f.run(&[b"OBJECT", b"NOPE", b"s"]),
            "-ERR unknown subcommand 'NOPE'. Try OBJECT HELP.\r\n"
        );
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING"]),
            "-ERR wrong number of arguments for 'object|encoding' command\r\n"
        );
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"s", b"extra"]),
            "-ERR wrong number of arguments for 'object|encoding' command\r\n"
        );
        assert_eq!(
            f.run(&[b"OBJECT"]),
            "-ERR wrong number of arguments for 'object' command\r\n"
        );
    }

    #[test]
    fn config_moves_the_ladder_and_object_encoding_agrees() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-listpack-entries"]),
            "*2\r\n$25\r\nhash-max-listpack-entries\r\n$3\r\n512\r\n",
            "512 and not the 128 everyone remembers, which is what 8.10.1 says"
        );
        // The old spelling is the same number under a different name, and a
        // glob that catches both sends both.
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-ziplist-entries"]),
            "*2\r\n$24\r\nhash-max-ziplist-entries\r\n$3\r\n512\r\n"
        );
        assert!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-*"])
                .starts_with("*8\r\n")
        );
        assert!(
            f.run(&[b"CONFIG", b"GET", b"set-max-*"])
                .starts_with("*6\r\n")
        );

        f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2", b"c", b"3"]);
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"h"]), "$8\r\nlistpack\r\n");

        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"hash-max-ziplist-entries", b"2"]),
            "+OK\r\n",
            "written under the old name and read back under the new one"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-listpack-entries"]),
            "*2\r\n$25\r\nhash-max-listpack-entries\r\n$1\r\n2\r\n"
        );
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"h"]),
            "$8\r\nlistpack\r\n",
            "the hash that already exists is left exactly where it was"
        );
        f.run(&[b"HSET", b"h2", b"a", b"1", b"b", b"2", b"c", b"3"]);
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"h2"]),
            "$9\r\nhashtable\r\n",
            "and the next one built goes straight to a table"
        );

        // The set has three of these and all three move.
        f.run(&[b"CONFIG", b"SET", b"set-max-intset-entries", b"2"]);
        f.run(&[b"SADD", b"s", b"1", b"2", b"3"]);
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"s"]), "$8\r\nlistpack\r\n");
        f.run(&[b"CONFIG", b"SET", b"set-max-listpack-value", b"2"]);
        f.run(&[b"SADD", b"s2", b"abcdefgh"]);
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"s2"]),
            "$9\r\nhashtable\r\n"
        );
    }

    #[test]
    fn config_set_takes_all_of_the_ladder_or_none_of_it() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"CONFIG",
                b"SET",
                b"hash-max-listpack-entries",
                b"7",
                b"set-max-listpack-entries",
                b"abc"
            ]),
            "-ERR CONFIG SET failed (possibly related to argument 'set-max-listpack-entries') - argument couldn't be parsed into an integer\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-listpack-entries"]),
            "*2\r\n$25\r\nhash-max-listpack-entries\r\n$3\r\n512\r\n",
            "the pair in front of the bad one did not go in"
        );
        // The name in the complaint is the one that was typed, so the old
        // spelling comes back as the old spelling.
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"hash-max-ziplist-entries", b"abc"]),
            "-ERR CONFIG SET failed (possibly related to argument 'hash-max-ziplist-entries') - argument couldn't be parsed into an integer\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"set-max-intset-entries", b"-1"]),
            "-ERR CONFIG SET failed (possibly related to argument 'set-max-intset-entries') - argument must be between 0 and 9223372036854775807 inclusive\r\n"
        );
        // A number past what an i64 holds is the parse complaint and not the
        // range one, which is upstream reading it before it checks it.
        assert_eq!(
            f.run(&[
                b"CONFIG",
                b"SET",
                b"set-max-intset-entries",
                b"99999999999999999999"
            ]),
            "-ERR CONFIG SET failed (possibly related to argument 'set-max-intset-entries') - argument couldn't be parsed into an integer\r\n"
        );
        assert_eq!(
            f.run(&[
                b"CONFIG",
                b"SET",
                b"set-max-intset-entries",
                b"9223372036854775807"
            ]),
            "+OK\r\n"
        );
    }

    #[test]
    fn a_setting_moved_on_one_database_moved_on_all_of_them() {
        let mut f = Fixture::new();
        f.run(&[b"CONFIG", b"SET", b"hash-max-listpack-entries", b"1"]);
        f.run(&[b"SELECT", b"3"]);
        f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]);
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"h"]),
            "$9\r\nhashtable\r\n",
            "these are one server wide number in Redis, whatever a Keyspace carries"
        );
    }

    #[test]
    fn info_reports_the_numbers_it_can_stand_behind() {
        let mut f = Fixture::new();
        f.run(&[b"MSET", b"a", b"1", b"b", b"2"]);
        let all = f.run(&[b"INFO"]);
        assert!(all.contains("redis_version:8.8.0"), "{all}");
        assert!(
            all.contains(concat!("yo_version:", env!("CARGO_PKG_VERSION"))),
            "{all}"
        );
        assert!(all.contains("db0:keys=2,expires=0,avg_ttl=0"), "{all}");
        assert!(all.contains("role:master"), "{all}");
        // One section is one section.
        let clients = f.run(&[b"INFO", b"clients"]);
        assert!(clients.contains("connected_clients:0"), "{clients}");
        assert!(!clients.contains("redis_version"), "{clients}");
        assert_eq!(f.run(&[b"INFO", b"nosuch"]), "$0\r\n\r\n");
    }

    /// The sections a bare `INFO` gives back, and the ones you have to ask for.
    ///
    /// This is Redis's `unit/info-command` written against the fixture. Every
    /// assertion in it is one of theirs, in their order, and the two fields it
    /// turns on are the two that suite was failing on: `master_repl_offset`,
    /// which is in the default set, and `rejected_calls`, which is not.
    #[test]
    fn commandstats_is_asked_for_and_replication_is_not() {
        let mut f = Fixture::new();
        for arg in ["", "all", "default", "everything"] {
            let info = if arg.is_empty() {
                f.run(&[b"INFO"])
            } else {
                f.run(&[b"INFO", arg.as_bytes()])
            };
            assert!(info.contains("redis_version"), "{arg}: {info}");
            assert!(info.contains("used_cpu_user"), "{arg}: {info}");
            assert!(info.contains("used_memory"), "{arg}: {info}");
            assert!(!info.contains("sentinel_tilt"), "{arg}: {info}");
            let asked = arg == "all" || arg == "everything";
            assert_eq!(
                info.contains("rejected_calls"),
                asked,
                "{arg} should{} carry the command counters: {info}",
                if asked { "" } else { " not" }
            );
        }

        let cpu = f.run(&[b"INFO", b"cpu"]);
        assert!(cpu.contains("used_cpu_user"), "{cpu}");
        assert!(!cpu.contains("used_memory"), "{cpu}");

        // Their case, to make the point that a section name is not case
        // sensitive any more than a command name is.
        let stats = f.run(&[b"INFO", b"commandSTATS"]);
        assert!(!stats.contains("used_memory"), "{stats}");
        assert!(stats.contains("rejected_calls"), "{stats}");

        // Two sections named, and neither of them pulls in a third.
        let pair = f.run(&[b"INFO", b"cpu", b"sentinel"]);
        assert!(pair.contains("used_cpu_user"), "{pair}");
        assert!(!pair.contains("master_repl_offset"), "{pair}");

        let with_all = f.run(&[b"INFO", b"cpu", b"all"]);
        assert!(with_all.contains("used_memory"), "{with_all}");
        assert!(with_all.contains("master_repl_offset"), "{with_all}");
        assert!(with_all.contains("rejected_calls"), "{with_all}");
        // A section named twice is still written once.
        assert_eq!(
            with_all.matches("used_cpu_user_children").count(),
            1,
            "{with_all}"
        );

        let with_default = f.run(&[b"INFO", b"cpu", b"default"]);
        assert!(with_default.contains("used_memory"), "{with_default}");
        assert!(
            with_default.contains("master_repl_offset"),
            "{with_default}"
        );
        assert!(!with_default.contains("rejected_calls"), "{with_default}");
        assert_eq!(
            with_default.matches("used_cpu_user_children").count(),
            1,
            "{with_default}"
        );
    }

    /// The memory section says what this process may use, not what the machine
    /// has.
    ///
    /// The distinction is the whole point of it. A server inside a container
    /// that reports the host's memory is a server whose operator sizes it for
    /// memory it will be killed for touching, so all three numbers are there:
    /// what the machine has, what the cgroup allows, and the quarter of the
    /// tighter one that pools are sized from.
    #[test]
    fn info_memory_reports_the_cap_and_the_quarter_of_it_that_gets_used() {
        let mut f = Fixture::new();
        let info = f.run(&[b"INFO", b"memory"]);
        for field in [
            "total_system_memory:",
            "mem_cgroup_limit:",
            "mem_limit:",
            "mem_budget:",
        ] {
            assert!(info.contains(field), "no {field} in {info}");
        }

        let field = |name: &str| -> u64 {
            info.lines()
                .find_map(|l| l.strip_prefix(name))
                .unwrap_or_else(|| panic!("no {name} in {info}"))
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} is not a number in {info}"))
        };
        let limit = field("mem_limit:");
        assert_eq!(field("mem_budget:"), limit / 4, "{info}");
        // Zero means there is no limit to report, which is a real answer on a
        // machine with no cgroups and no way to ask how big it is.
        if limit != 0 {
            let host = field("total_system_memory:");
            let cgroup = field("mem_cgroup_limit:");
            assert!(
                limit == host || limit == cgroup,
                "the limit came from neither number: {info}"
            );
        }
    }

    /// The three counters, each on the path that raises it.
    ///
    /// `calls` on a command that worked, `failed_calls` on one that ran and
    /// answered with an error, and `rejected_calls` on one that never ran at
    /// all. The last two are the pair that is easy to collapse into one number
    /// and that Redis keeps apart, because a client sending the wrong number of
    /// arguments and a client asking for a list element that is not there are
    /// not the same problem.
    #[test]
    fn a_command_counts_what_it_did_separately_from_what_it_refused() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);
        f.run(&[b"SET", b"k", b"w"]);
        // Ran, and answered with an error, because `k` is not a list.
        f.run(&[b"LPUSH", b"k", b"x"]);
        // Never ran: `LPUSH` takes at least three arguments.
        f.run(&[b"LPUSH", b"k"]);

        let stats = f.run(&[b"INFO", b"commandstats"]);
        assert!(
            stats.contains("cmdstat_set:calls=2,rejected_calls=0,failed_calls=0"),
            "{stats}"
        );
        assert!(
            stats.contains("cmdstat_lpush:calls=1,rejected_calls=1,failed_calls=1"),
            "{stats}"
        );
        assert!(
            !stats.contains("cmdstat_zadd"),
            "a command nobody has sent has no row: {stats}"
        );
    }

    /// A cache that writes with a deadline and never reads back used to hold
    /// every key it had ever written, because lazy expiry needs somebody to walk
    /// past a key before it can reclaim it and nobody ever did.
    #[test]
    fn the_active_sweep_reclaims_keys_no_client_comes_back_for() {
        let mut f = Fixture::new();
        for i in 0..3_000u32 {
            f.run(&[b"SET", format!("d{i}").as_bytes(), b"v", b"PX", b"50"]);
        }
        for i in 0..1_000u32 {
            f.run(&[b"SET", format!("k{i}").as_bytes(), b"v"]);
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":4000\r\n");
        f.advance(100);
        assert_eq!(
            f.run(&[b"DBSIZE"]),
            ":4000\r\n",
            "DBSIZE counts records and nothing has read past the dead ones yet"
        );

        // What the shard loop does, one slice at a time.
        let mut spent = 0;
        for _ in 0..2_000 {
            spent += f.server.expire_step(4096);
            if f.run(&[b"DBSIZE"]) == ":1000\r\n" {
                break;
            }
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":1000\r\n", "spent {spent} looks");
        assert!(f.run(&[b"INFO", b"stats"]).contains("expired_keys:3000"));
        for i in 0..1_000u32 {
            assert_eq!(
                f.run(&[b"GET", format!("k{i}").as_bytes()]),
                "$1\r\nv\r\n",
                "it took a key that had no deadline"
            );
        }
    }

    #[test]
    fn a_sweep_of_a_server_with_no_deadlines_anywhere_costs_nothing() {
        let mut f = Fixture::new();
        for i in 0..2_000u32 {
            f.run(&[b"SET", format!("k{i}").as_bytes(), b"v"]);
        }
        assert_eq!(f.server.expire_step(4096), 0);
        // And one database having them does not make the other fifteen pay.
        f.run(&[b"SELECT", b"3"]);
        f.run(&[b"SET", b"x", b"v", b"PX", b"50"]);
        f.advance(100);
        for _ in 0..64 {
            f.server.expire_step(4096);
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        f.run(&[b"SELECT", b"0"]);
        assert_eq!(f.run(&[b"DBSIZE"]), ":2000\r\n");
        assert_eq!(f.server.expire_step(4096), 0, "and it is quiet again");
    }

    /// The gate, which is what stops a maintenance slice that runs every hundred
    /// nanoseconds from drawing a sample every hundred nanoseconds.
    #[test]
    fn the_sweep_the_loop_calls_runs_at_most_once_a_millisecond() {
        let mut f = Fixture::new();
        for i in 0..500u32 {
            f.run(&[b"SET", format!("d{i}").as_bytes(), b"v", b"PX", b"50"]);
        }
        f.advance(100);
        let at = f.server.striped(0).now_ms();
        f.server.set_clock_ms(at);
        // A small budget, so that one slice cannot finish the job and a second
        // one having nothing to do would mean the gate and not an empty
        // database.
        assert!(f.server.expire_slice(8) > 0, "the first one works");
        for _ in 0..1_000 {
            assert_eq!(
                f.server.expire_slice(8),
                0,
                "the millisecond has not moved and neither should this"
            );
        }
        assert!(
            f.server.striped(0).expires() > 400,
            "there is plenty left to take"
        );
        f.server.set_clock_ms(at + 1);
        assert!(f.server.expire_slice(8) > 0, "and then it goes again");
    }

    /// `expires=` used to be a hardcoded zero, which meant a dashboard watching
    /// how much of a cache is volatile was reading a constant.
    #[test]
    fn info_keyspace_counts_the_keys_that_have_a_deadline() {
        let mut f = Fixture::new();
        f.run(&[b"MSET", b"a", b"1", b"b", b"2", b"c", b"3"]);
        assert!(
            f.run(&[b"INFO", b"keyspace"])
                .contains("db0:keys=3,expires=0"),
            "none of them has one yet"
        );
        f.run(&[b"EXPIRE", b"a", b"1000"]);
        f.run(&[b"EXPIRE", b"b", b"1000"]);
        let two = f.run(&[b"INFO", b"keyspace"]);
        assert!(two.contains("db0:keys=3,expires=2"), "{two}");
        f.run(&[b"PERSIST", b"a"]);
        f.run(&[b"DEL", b"b"]);
        let none = f.run(&[b"INFO", b"keyspace"]);
        assert!(none.contains("db0:keys=2,expires=0"), "{none}");

        // Each database answers for itself, the way Redis reports it.
        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"SET", b"x", b"1", b"EX", b"1000"]);
        let both = f.run(&[b"INFO", b"keyspace"]);
        assert!(both.contains("db0:keys=2,expires=0"), "{both}");
        assert!(both.contains("db1:keys=1,expires=1"), "{both}");
    }

    #[cfg(unix)]
    #[test]
    fn info_cpu_reports_processor_time_that_was_really_measured() {
        let mut f = Fixture::new();
        let cpu = f.run(&[b"INFO", b"cpu"]);
        assert!(cpu.contains("# CPU"), "{cpu}");
        // Redis's unit/info-command asks for this one by name in three tests.
        assert!(cpu.contains("used_cpu_user:"), "{cpu}");
        assert!(cpu.contains("used_cpu_sys:"), "{cpu}");
        assert!(cpu.contains("used_cpu_user_children:0.000000"), "{cpu}");
        assert!(!cpu.contains("redis_version"), "{cpu}");

        // It is a measurement and not a constant, so it goes up when work
        // happens. A tight loop rather than a sleep, because sleeping is the
        // one thing that does not move this number.
        let before = used_cpu_user(&cpu);
        let mut n = 0u64;
        let mut rounds = 0;
        while used_cpu_user(&f.run(&[b"INFO", b"cpu"])) <= before {
            for i in 0..1_000_000u64 {
                n = n.wrapping_add(i.wrapping_mul(i));
            }
            rounds += 1;
            // A bound rather than a spin, so a platform where this number does
            // not move fails here instead of hanging. Even a clock with whole
            // millisecond granularity gets there in the first round or two.
            assert!(rounds < 1_000, "cpu time never moved, n is {n}");
        }
    }

    /// Pull `used_cpu_user` back out of an `INFO cpu` reply.
    #[cfg(unix)]
    fn used_cpu_user(info: &str) -> f64 {
        info.lines()
            .find_map(|l| l.strip_prefix("used_cpu_user:"))
            .expect("no used_cpu_user in the reply")
            .trim()
            .parse()
            .expect("used_cpu_user is not a number")
    }

    /// The safety net under the rule that a body checks its arguments before
    /// it writes anything. `MGET` writes its array header first and then reads
    /// each key, so if a later argument could fail the header would already be
    /// out. Nothing in the string group does that today and this is what would
    /// catch the first one that did.
    #[test]
    fn a_command_that_fails_leaves_nothing_half_written() {
        let mut f = Fixture::new();
        let reply = f.run(&[b"SETRANGE", b"k", b"-1", b"x"]);
        assert_eq!(reply, "-ERR offset is out of range\r\n");
        assert!(!reply.contains(':'), "no integer went out in front of it");
    }

    #[test]
    fn quit_answers_first_and_closes_after() {
        let mut f = Fixture::new();
        let (flow, reply) = f.flow(&[b"QUIT"]);
        assert_eq!(reply, "+OK\r\n");
        assert_eq!(flow, Flow::Close);
    }

    /// A server that has not been asked to stop is not stopping, and one that
    /// has says so without writing anything back.
    ///
    /// The empty reply is the point. Redis answers nothing at all here and the
    /// client sees the socket close, and an `OK` would be a promise from a
    /// process that is about to not exist.
    #[test]
    fn shutdown_writes_nothing_and_sets_the_flag() {
        let mut f = Fixture::new();
        assert!(!f.server.stopping(), "nobody has asked yet");

        let (flow, reply) = f.flow(&[b"SHUTDOWN"]);
        assert_eq!(reply, "");
        assert_eq!(flow, Flow::Close);
        assert!(f.server.stopping());
    }

    /// Every flag combination 8.10.1 takes, and every one it refuses.
    ///
    /// The refusals are the half worth pinning down. `SAVE` and `NOSAVE`
    /// contradict each other, `ABORT` says to do nothing so it cannot be
    /// combined with a word about how to do it, and repeating any one of them
    /// is fine. All of it was read off a running 8.10.1 rather than worked out
    /// from the documentation, which does not say.
    #[test]
    fn shutdown_takes_the_flags_redis_takes() {
        for flags in [
            &[b"NOSAVE".as_slice()][..],
            &[b"SAVE"],
            &[b"NOW"],
            &[b"FORCE"],
            &[b"nosave"],
            &[b"NOW", b"NOW"],
            &[b"SAVE", b"SAVE"],
            &[b"NOSAVE", b"NOW", b"FORCE"],
        ] {
            let mut f = Fixture::new();
            let mut parts = vec![b"SHUTDOWN".as_slice()];
            parts.extend_from_slice(flags);
            let (flow, reply) = f.flow(&parts);
            assert_eq!(reply, "", "SHUTDOWN {flags:?} answered something");
            assert_eq!(flow, Flow::Close, "SHUTDOWN {flags:?} did not close");
            assert!(f.server.stopping(), "SHUTDOWN {flags:?} did not stop");
        }

        for flags in [
            &[b"BOGUS".as_slice()][..],
            &[b"SAVE", b"NOSAVE"],
            &[b"NOSAVE", b"SAVE"],
            &[b"ABORT", b"NOW"],
            &[b"NOSAVE", b"ABORT"],
            &[b"NOW", b"FORCE", b"ABORT"],
        ] {
            let mut f = Fixture::new();
            let mut parts = vec![b"SHUTDOWN".as_slice()];
            parts.extend_from_slice(flags);
            assert_eq!(
                f.run(&parts),
                "-ERR syntax error\r\n",
                "SHUTDOWN {flags:?} was accepted"
            );
            assert!(!f.server.stopping(), "SHUTDOWN {flags:?} stopped anyway");
        }
    }

    /// `ABORT` has nothing to call off, ever.
    ///
    /// A shutdown here is decided and done inside one turn of the loop, so
    /// there is no window in which one is in progress. That makes Redis's
    /// message for a cancel with nothing to cancel the right answer every time
    /// rather than only when nothing happens to be pending. Two `ABORT`s is
    /// still one `ABORT`, which is what 8.10.1 does.
    #[test]
    fn shutdown_abort_never_has_anything_to_abort() {
        let mut f = Fixture::new();
        for parts in [
            &[b"SHUTDOWN".as_slice(), b"ABORT"][..],
            &[b"SHUTDOWN", b"ABORT", b"ABORT"],
        ] {
            assert_eq!(f.run(parts), "-ERR No shutdown in progress.\r\n");
            assert!(!f.server.stopping(), "an abort stopped the server");
        }
    }

    /// A fixture whose server writes into a directory of its own.
    ///
    /// Every test here really writes files, because the whole point of the
    /// command is the files and a backup that is only a state machine would
    /// pass a test suite and fail the first person who tried to restore one.
    /// The directory carries the test's name so that the suite can run its
    /// tests in parallel the way it always does.
    struct Backups {
        f: Fixture,
        dir: PathBuf,
    }

    impl Backups {
        fn new(name: &str) -> Backups {
            let dir = std::env::temp_dir().join(format!("yo-backup-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("could not make a temporary directory");
            let mut f = Fixture::new();
            f.server.set_dir(dir.clone());
            Backups { f, dir }
        }

        fn run(&mut self, parts: &[&[u8]]) -> String {
            self.f.run(parts)
        }

        /// The names in `backupdir`, sorted, so a test can say what is on disk.
        fn files(&self) -> Vec<String> {
            let mut names: Vec<String> = match std::fs::read_dir(self.dir.join("backupdir")) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect(),
                Err(_) => Vec::new(),
            };
            names.sort();
            names
        }

        fn read(&self, name: &str) -> Vec<u8> {
            std::fs::read(self.dir.join("backupdir").join(name)).expect("could not read")
        }
    }

    impl Drop for Backups {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// The four states and the moves between them, in the order a client walks
    /// them, with the files checked at every step.
    #[test]
    fn backup_walks_the_states_the_reference_walks() {
        let mut b = Backups::new("states");
        let status = |b: &mut Backups| b.run(&[b"BACKUP", b"STATUS"]);

        assert!(status(&mut b).contains("idle"));
        assert!(b.files().is_empty(), "an idle server has written a backup");

        assert_eq!(b.run(&[b"BACKUP", b"START"]), "+OK\r\n");
        assert!(status(&mut b).contains("incrementing"));
        assert_eq!(b.files(), ["appendonly.aof.1.base.rdb"]);

        assert_eq!(b.run(&[b"BACKUP", b"SEAL"]), "+OK\r\n");
        assert!(status(&mut b).contains("sealed"));
        assert_eq!(
            b.files(),
            [
                "appendonly.aof.1.base.rdb",
                "appendonly.aof.1.incr.aof",
                "appendonly.aof.manifest",
            ]
        );

        assert_eq!(b.run(&[b"BACKUP", b"CLEANUP"]), "+OK\r\n");
        assert!(status(&mut b).contains("idle"));
        assert!(b.files().is_empty(), "cleanup left something behind");
    }

    /// Every move that is refused, in the reference's words.
    #[test]
    fn backup_refuses_the_moves_the_reference_refuses() {
        let mut b = Backups::new("refusals");

        assert_eq!(
            b.run(&[b"BACKUP", b"SEAL"]),
            "-ERR No backup ready to seal (must be in the incrementing state)\r\n"
        );
        assert_eq!(
            b.run(&[b"BACKUP", b"ABORT"]),
            "-ERR No backup in progress\r\n"
        );
        // Cleanup from idle is not an error, it is a way of saying there was
        // nothing to clean up.
        assert_eq!(b.run(&[b"BACKUP", b"CLEANUP"]), "+OK\r\n");

        b.run(&[b"BACKUP", b"START"]);
        assert_eq!(
            b.run(&[b"BACKUP", b"START"]),
            "-ERR A backup is already in progress, ABORT it first\r\n"
        );
        assert_eq!(
            b.run(&[b"BACKUP", b"CLEANUP"]),
            "-ERR Backup is in progress\r\n"
        );

        b.run(&[b"BACKUP", b"SEAL"]);
        assert_eq!(
            b.run(&[b"BACKUP", b"START"]),
            "-ERR A sealed backup exists, CLEANUP it first\r\n"
        );
        assert_eq!(
            b.run(&[b"BACKUP", b"SEAL"]),
            "-ERR No backup ready to seal (must be in the incrementing state)\r\n"
        );
        assert_eq!(
            b.run(&[b"BACKUP", b"ABORT"]),
            "-ERR No backup in progress\r\n"
        );
    }

    /// An abort takes the base file away and leaves a state saying who did it.
    ///
    /// The next backup takes the next sequence number rather than reusing the
    /// one whose files were just thrown away, so a directory somebody copied a
    /// half finished backup out of cannot end up with two different files under
    /// one name.
    #[test]
    fn backup_abort_removes_the_file_and_says_who_did_it() {
        let mut b = Backups::new("abort");
        b.run(&[b"BACKUP", b"START"]);
        assert_eq!(b.run(&[b"BACKUP", b"ABORT"]), "+OK\r\n");

        let status = b.run(&[b"BACKUP", b"STATUS"]);
        assert!(status.contains("failed"), "{status}");
        assert!(status.contains("aborted by user"), "{status}");
        assert!(b.files().is_empty(), "abort left the base file behind");
        assert_eq!(b.run(&[b"BACKUP", b"LIST"]), "*0\r\n");

        // A start from failed works, and is the second backup.
        assert_eq!(b.run(&[b"BACKUP", b"START"]), "+OK\r\n");
        assert_eq!(b.files(), ["appendonly.aof.2.base.rdb"]);
        let status = b.run(&[b"BACKUP", b"STATUS"]);
        assert!(status.contains("incrementing"), "{status}");
        assert!(!status.contains("aborted"), "the old error was kept");
    }

    /// `LIST` names nothing, then one file, then three, and they are absolute.
    #[test]
    fn backup_list_names_the_files_that_are_pinned_so_far() {
        let mut b = Backups::new("list");
        assert_eq!(b.run(&[b"BACKUP", b"LIST"]), "*0\r\n");

        b.run(&[b"BACKUP", b"START"]);
        let base = b.dir.join("backupdir").join("appendonly.aof.1.base.rdb");
        let base = base.to_string_lossy().into_owned();
        assert_eq!(
            b.run(&[b"BACKUP", b"LIST"]),
            format!("*1\r\n${}\r\n{base}\r\n", base.len())
        );

        b.run(&[b"BACKUP", b"SEAL"]);
        let listed = b.run(&[b"BACKUP", b"LIST"]);
        assert!(listed.starts_with("*3\r\n"), "{listed}");
        // The order is the manifest's order, base then incremental then the
        // manifest itself, which is the order a restore needs them in.
        let names: Vec<&str> = listed
            .lines()
            .filter(|l| l.starts_with('/') || l.contains(":\\"))
            .collect();
        assert_eq!(names.len(), 3, "{listed}");
        assert!(names[0].ends_with("appendonly.aof.1.base.rdb"), "{listed}");
        assert!(names[1].ends_with("appendonly.aof.1.incr.aof"), "{listed}");
        assert!(names[2].ends_with("appendonly.aof.manifest"), "{listed}");
    }

    /// The base file is the dataset as it was at `START` and not at `SEAL`.
    ///
    /// That is D-46 and it is the one thing about this a client can notice, so
    /// it is pinned here rather than left to be discovered by whoever restores
    /// one. The incremental file is empty for the same reason: there is no
    /// append only log underneath this server to copy the writes in between out
    /// of.
    #[test]
    fn a_backup_holds_the_dataset_as_it_was_at_start() {
        let mut b = Backups::new("contents");
        b.run(&[b"SET", b"bk", b"v1"]);
        b.run(&[b"BACKUP", b"START"]);
        b.run(&[b"SET", b"bk", b"v2"]);
        b.run(&[b"BACKUP", b"SEAL"]);

        let base = b.read("appendonly.aof.1.base.rdb");
        assert!(base.starts_with(b"REDIS"), "not an RDB file");
        assert!(base.windows(2).any(|w| w == b"v1"), "the value is missing");
        assert!(
            !base.windows(2).any(|w| w == b"v2"),
            "the base file moved on after START"
        );
        // The aux field a loader acts on, and the one that says this file is
        // the base of an append only file rather than a standalone dump. Its
        // value is the one byte string 1, which the encoder writes as an
        // integer the way a real server writes it.
        let at = base
            .windows(8)
            .position(|w| w == b"aof-base")
            .expect("no aof-base aux field");
        assert_eq!(&base[at + 8..at + 10], b"\xc0\x01", "{:?}", &base[at..]);

        assert!(b.read("appendonly.aof.1.incr.aof").is_empty());
        assert_eq!(
            String::from_utf8(b.read("appendonly.aof.manifest")).expect("the manifest is text"),
            "file appendonly.aof.1.base.rdb seq 1 type b\n\
             file appendonly.aof.1.incr.aof seq 1 type i startoffset 0 endoffset 0\n"
        );
    }

    /// `STATUS` is a map of four pairs on RESP3 and the same pairs flat on
    /// RESP2, which is what every other map shaped reply in this server does.
    #[test]
    fn backup_status_is_a_map_on_resp3_and_a_flat_array_on_resp2() {
        let mut b = Backups::new("status");
        b.f.server.set_clock_ms(1_700_000_000_000);

        assert_eq!(
            b.run(&[b"BACKUP", b"STATUS"]),
            "*8\r\n$5\r\nstate\r\n$4\r\nidle\r\n$5\r\nerror\r\n$0\r\n\r\n\
             $10\r\nstart_time\r\n:0\r\n$8\r\nend_time\r\n:0\r\n"
        );

        b.f.out = Out::new(Proto::Resp3);
        b.run(&[b"BACKUP", b"START"]);
        assert_eq!(
            b.run(&[b"BACKUP", b"STATUS"]),
            "%4\r\n$5\r\nstate\r\n$12\r\nincrementing\r\n$5\r\nerror\r\n$0\r\n\r\n\
             $10\r\nstart_time\r\n:1700000000\r\n$8\r\nend_time\r\n:0\r\n"
        );

        b.run(&[b"BACKUP", b"SEAL"]);
        let sealed = b.run(&[b"BACKUP", b"STATUS"]);
        assert!(sealed.contains("end_time\r\n:1700000000"), "{sealed}");
    }

    /// A sealed backup that nobody cleans up goes away on its own once
    /// `backup-sealed-ttl` seconds have passed since the seal.
    #[test]
    fn a_sealed_backup_is_swept_away_after_the_timeout() {
        let mut b = Backups::new("ttl");
        b.f.server.set_clock_ms(1_000_000);
        assert_eq!(
            b.run(&[b"CONFIG", b"SET", b"backup-sealed-ttl", b"60"]),
            "+OK\r\n"
        );
        b.run(&[b"BACKUP", b"START"]);
        b.run(&[b"BACKUP", b"SEAL"]);

        // A minute short of the deadline, nothing happens.
        b.f.server.set_clock_ms(1_000_000 + 59_000);
        b.f.server.backup_expire();
        assert!(b.run(&[b"BACKUP", b"STATUS"]).contains("sealed"));
        assert_eq!(b.files().len(), 3);

        b.f.server.set_clock_ms(1_000_000 + 60_000);
        b.f.server.backup_expire();
        let status = b.run(&[b"BACKUP", b"STATUS"]);
        assert!(status.contains("idle"), "{status}");
        assert!(b.files().is_empty(), "the timeout left the files behind");

        // Zero is the default and means a sealed backup is kept for ever.
        b.run(&[b"CONFIG", b"SET", b"backup-sealed-ttl", b"0"]);
        b.run(&[b"BACKUP", b"START"]);
        b.run(&[b"BACKUP", b"SEAL"]);
        b.f.server.set_clock_ms(9_000_000_000);
        b.f.server.backup_expire();
        assert!(b.run(&[b"BACKUP", b"STATUS"]).contains("sealed"));
    }

    /// The three settings around the command, read and written the way 8.10.1
    /// reads and writes them.
    #[test]
    fn the_backup_settings_behave_the_way_the_reference_does() {
        let mut b = Backups::new("config");
        let dir = b.dir.to_string_lossy().into_owned();

        assert_eq!(
            b.run(&[b"CONFIG", b"GET", b"dir"]),
            format!("*2\r\n$3\r\ndir\r\n${}\r\n{dir}\r\n", dir.len())
        );
        assert_eq!(
            b.run(&[b"CONFIG", b"GET", b"backupdirname"]),
            "*2\r\n$13\r\nbackupdirname\r\n$9\r\nbackupdir\r\n"
        );
        assert_eq!(
            b.run(&[b"CONFIG", b"GET", b"backup-sealed-ttl"]),
            "*2\r\n$17\r\nbackup-sealed-ttl\r\n$1\r\n0\r\n"
        );

        // `dir` is a protected config, so it is refused even for the value it
        // already holds, and `backupdirname` is immutable.
        assert_eq!(
            b.run(&[b"CONFIG", b"SET", b"dir", dir.as_bytes()]),
            "-ERR CONFIG SET failed (possibly related to argument 'dir') - can't set protected config\r\n"
        );
        assert_eq!(
            b.run(&[b"CONFIG", b"SET", b"backupdirname", b"other"]),
            "-ERR CONFIG SET failed (possibly related to argument 'backupdirname') - can't set immutable config\r\n"
        );
        assert!(
            b.run(&[b"CONFIG", b"SET", b"backup-sealed-ttl", b"abc"])
                .contains("argument couldn't be parsed into an integer")
        );
        assert!(
            b.run(&[b"CONFIG", b"SET", b"backup-sealed-ttl", b"-1"])
                .contains("argument must be between 0 and 9223372036854775807 inclusive")
        );
    }

    /// The help text, which has `HELP` in it twice because the reference's does.
    #[test]
    fn backup_help_is_the_text_the_reference_sends() {
        let mut f = Fixture::new();
        let help = f.run(&[b"BACKUP", b"HELP"]);
        assert!(help.starts_with("*17\r\n"), "{help}");
        assert!(
            help.contains("+BACKUP <subcommand> [<arg> [value] [opt] ...]. Subcommands are:\r\n")
        );
        assert!(help.contains("+    Start a new backup into the configured 'backupdirname'.\r\n"));
        assert!(help.contains("+    Freeze the current backup (BASE + INCR + manifest).\r\n"));
        assert!(help.contains("+    Return this help.\r\n+HELP\r\n+    Print this help.\r\n"));
    }

    /// What a mistyped `BACKUP` gets told.
    ///
    /// The arity error names `backup` where the reference names `backup|start`,
    /// which is D-46: the table reports one arity for the container the way the
    /// reference does, and the per subcommand table that would carry the better
    /// name is not built yet. Every subcommand is exactly two words, so nothing
    /// legal is refused by it.
    #[test]
    fn backup_refuses_what_it_cannot_read() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"BACKUP"]),
            "-ERR wrong number of arguments for 'backup' command\r\n"
        );
        assert_eq!(
            f.run(&[b"BACKUP", b"START", b"x"]),
            "-ERR wrong number of arguments for 'backup' command\r\n"
        );
        assert_eq!(
            f.run(&[b"BACKUP", b"NOPE"]),
            "-ERR unknown subcommand 'NOPE'. Try BACKUP HELP.\r\n"
        );
    }

    #[test]
    fn the_command_counter_counts_every_command_including_the_bad_ones() {
        let mut f = Fixture::new();
        f.run(&[b"PING"]);
        f.run(&[b"NOPE"]);
        f.run(&[b"GET"]);
        assert_eq!(f.server.totals().commands, 3);
    }

    #[test]
    fn what_a_thread_marked_is_taken_by_the_maintenance_turn() {
        let mut server = Server::new();
        server.set_threads(2);
        // A fresh server has every database on the turn's list, so start from
        // nothing to see the one mark arrive.
        server.mine().turn.store(0, Relaxed);
        server.locals[1].mark(1 << 9);
        server.collect_marks();
        assert!(server.mine().wanted(9));
        // And taken once rather than left to be taken again next turn.
        assert_eq!(server.locals[1].dirty.load(Relaxed), 0);
    }

    #[test]
    fn what_two_threads_counted_is_added_up_when_info_asks() {
        let mut server = Server::new();
        server.set_threads(2);
        // Written into the two sets by hand, because what is under test is the
        // adding up and not the claiming, and one test thread can only ever
        // claim one set.
        let ping = lookup(b"PING").expect("PING is a command");
        for (at, calls) in [(0, 2), (1, 3)] {
            let counters = &server.locals[at];
            for _ in 0..calls {
                counters.stats.commands.bump();
                counters.cmdstats.at(ping).calls.bump();
            }
            counters.stats.opened();
        }
        assert_eq!(server.totals().commands, 5);
        assert_eq!(server.totals().clients, 2);
        assert_eq!(server.totals().connections, 2);
        let rows: Vec<_> = server.command_stats().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "ping");
        assert_eq!(rows[0].1.calls, 5);
        // A reset takes the totals and leaves the open connections, which are
        // still open.
        server.reset_stats();
        assert_eq!(server.totals().commands, 0);
        assert_eq!(server.totals().connections, 0);
        assert_eq!(server.totals().clients, 2);
    }

    #[test]
    fn the_parked_count_says_what_the_waiter_list_says() {
        let mut f = Fixture::new();
        assert_eq!(f.server.parked(), 0);
        for client in 1..=3u64 {
            f.session = Session::new(client);
            assert_eq!(f.flow(&[b"BLPOP", b"q", b"0"]).0, Flow::Block);
        }
        assert_eq!(f.server.parked(), 3);
        assert_eq!(f.server.waiters().len(), 3);

        // The three ways the list gets shorter, each of which has to move the
        // number with it, because a number left behind is either a walk of the
        // list that never happens or one that runs off the end of it.
        f.server.drop_waiter(1);
        assert_eq!(f.server.parked(), f.server.waiters().len());
        f.server.forget_waiters(1);
        assert_eq!(f.server.parked(), f.server.waiters().len());
        f.run(&[b"RPUSH", b"q", b"v"]);
        let mut out = Out::new(Proto::Resp2);
        assert!(f.server.serve_waiter(0, 0, &mut out));
        f.server.drop_waiter(0);
        assert_eq!(f.server.parked(), 0);
        assert!(f.server.waiters().is_empty());
    }

    #[test]
    fn a_set_goes_from_bytes_to_bytes() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SADD", b"s", b"a", b"b", b"c"]), ":3\r\n");
        assert_eq!(f.run(&[b"SADD", b"s", b"b", b"d"]), ":1\r\n");
        assert_eq!(f.run(&[b"SCARD", b"s"]), ":4\r\n");
        assert_eq!(f.run(&[b"SISMEMBER", b"s", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"SISMEMBER", b"s", b"z"]), ":0\r\n");
        assert_eq!(f.run(&[b"TYPE", b"s"]), "+set\r\n");
        assert_eq!(
            f.run(&[b"SMISMEMBER", b"s", b"a", b"z", b"d"]),
            "*3\r\n:1\r\n:0\r\n:1\r\n"
        );
        assert_eq!(f.run(&[b"SREM", b"s", b"a", b"z"]), ":1\r\n");
        assert_eq!(f.run(&[b"SCARD", b"s"]), ":3\r\n");
    }

    #[test]
    fn a_set_command_at_a_key_that_is_not_there_answers_empty() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SCARD", b"nope"]), ":0\r\n");
        assert_eq!(f.run(&[b"SISMEMBER", b"nope", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"SREM", b"nope", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"SMEMBERS", b"nope"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"SMISMEMBER", b"nope", b"a", b"b"]),
            "*2\r\n:0\r\n:0\r\n"
        );
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n", "and made nothing");
    }

    #[test]
    fn smembers_answers_a_set_on_resp3_and_an_array_on_resp2() {
        // Not cosmetic. A RESP3 client that gets a `~` hands the caller a set
        // and one that gets a `*` hands it a list, without either of them being
        // told which command was sent.
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"one"]);
        assert_eq!(f.run(&[b"SMEMBERS", b"s"]), "*1\r\n$3\r\none\r\n");

        f.run(&[b"HELLO", b"3"]);
        assert_eq!(f.run(&[b"SMEMBERS", b"s"]), "~1\r\n$3\r\none\r\n");
    }

    #[test]
    fn an_integer_member_comes_back_as_the_digits_it_never_stored() {
        // An intset holds the number, so these digits exist for the first time
        // in the reply buffer.
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"42"]);
        assert_eq!(f.run(&[b"SMEMBERS", b"s"]), "*1\r\n$2\r\n42\r\n");
        assert_eq!(f.run(&[b"SISMEMBER", b"s", b"42"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"SISMEMBER", b"s", b"042"]),
            ":0\r\n",
            "the member is the bytes and not the number they parse to"
        );
    }

    #[test]
    fn the_wrong_command_at_the_wrong_type_says_so_both_ways() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        f.run(&[b"SADD", b"set", b"a"]);

        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        assert_eq!(f.run(&[b"SADD", b"str", b"a"]), wrong);
        assert_eq!(f.run(&[b"SCARD", b"str"]), wrong);
        assert_eq!(f.run(&[b"SMEMBERS", b"str"]), wrong);
        assert_eq!(f.run(&[b"SMISMEMBER", b"str", b"a"]), wrong);
        assert_eq!(f.run(&[b"GET", b"set"]), wrong);
        assert_eq!(f.run(&[b"APPEND", b"set", b"x"]), wrong);
        assert_eq!(f.run(&[b"INCR", b"set"]), wrong);
        assert_eq!(f.run(&[b"STRLEN", b"set"]), wrong);

        // MGET is the one that does not, because Redis gives nil for the odd
        // key out rather than failing the good keys next to it.
        assert_eq!(
            f.run(&[b"MGET", b"str", b"set", b"nope"]),
            "*3\r\n$1\r\nv\r\n$-1\r\n$-1\r\n"
        );
        // And plain SET overwrites any type, which takes the body with it.
        assert_eq!(f.run(&[b"SET", b"set", b"now a string"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TYPE", b"set"]), "+string\r\n");
    }

    #[test]
    fn a_wrongtype_leaves_nothing_half_written() {
        // SMISMEMBER writes an array header and then one reply per member, so
        // it is the first command in the server that could get a header out in
        // front of an error if it checked its key in the wrong order.
        let mut f = Fixture::new();
        f.run(&[b"SET", b"k", b"v"]);
        let reply = f.run(&[b"SMISMEMBER", b"k", b"a", b"b"]);
        assert!(reply.starts_with("-WRONGTYPE"), "got {reply}");
        assert!(!reply.contains('*'), "an array header went out in front");
    }

    #[test]
    fn emptying_a_set_takes_the_key_with_it() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"a", b"b"]);
        assert_eq!(f.run(&[b"DBSIZE"]), ":1\r\n");
        assert_eq!(f.run(&[b"SREM", b"s", b"a", b"b"]), ":2\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"s"]), ":0\r\n");
        assert_eq!(f.run(&[b"TYPE", b"s"]), "+none\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
    }

    /// Pull the cursor and the members out of one `SSCAN` reply.
    ///
    /// Crude on purpose. A test that walked a set through a real client would
    /// be testing the client, and what these tests are about is the shape of
    /// the bytes and the fact that a walk sees every member once.
    fn split_scan(reply: &str) -> (String, Vec<String>) {
        let mut lines = reply.split("\r\n");
        assert_eq!(lines.next(), Some("*2"), "got {reply}");
        lines.next().expect("the cursor header");
        let cursor = lines.next().expect("the cursor").to_owned();
        let header = lines.next().expect("the member header");
        let n: usize = header[1..].parse().expect("a member count");
        let mut members = Vec::with_capacity(n);
        for _ in 0..n {
            lines.next().expect("a member header");
            members.push(lines.next().expect("a member").to_owned());
        }
        (cursor, members)
    }

    #[test]
    fn popping_takes_a_member_off_the_set_and_hands_it_back() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"a", b"b", b"c", b"d"]);

        let one = f.run(&[b"SPOP", b"s"]);
        assert!(
            ["$1\r\na\r\n", "$1\r\nb\r\n", "$1\r\nc\r\n", "$1\r\nd\r\n"].contains(&one.as_str()),
            "got {one}"
        );
        assert_eq!(f.run(&[b"SCARD", b"s"]), ":3\r\n");

        // A count takes that many, and the last one takes the key with it.
        let (_, rest) = ("", f.run(&[b"SPOP", b"s", b"3"]));
        assert!(rest.starts_with("*3\r\n"), "got {rest}");
        assert_eq!(f.run(&[b"EXISTS", b"s"]), ":0\r\n");
        // And a pop at a key that is not there is a nil, not an empty bulk.
        assert_eq!(f.run(&[b"SPOP", b"s"]), "$-1\r\n");
        assert_eq!(f.run(&[b"SPOP", b"s", b"2"]), "*0\r\n");
    }

    #[test]
    fn the_two_draws_disagree_about_the_reply_type_and_they_are_right_to() {
        // The one place in the server where the reply type carries something
        // the command name does not. SPOP's members are distinct so a RESP3
        // client can build a set out of them. SRANDMEMBER with a negative count
        // can hand back the same member three times, and a set would lose two.
        let mut f = Fixture::new();
        f.run(&[b"HELLO", b"3"]);
        f.run(&[b"SADD", b"s", b"a", b"b", b"c"]);

        assert!(f.run(&[b"SPOP", b"s", b"2"]).starts_with("~2\r\n"));
        // And a positive count is an array too, since Redis makes it one.
        assert!(f.run(&[b"SRANDMEMBER", b"s", b"1"]).starts_with("*1\r\n"));

        // A negative count against a set of one is where the difference bites:
        // the same member three times, which is a three element reply and would
        // have been a one element reply if it had gone out as a set.
        f.run(&[b"SADD", b"one", b"z"]);
        assert_eq!(
            f.run(&[b"SRANDMEMBER", b"one", b"-3"]),
            "*3\r\n$1\r\nz\r\n$1\r\nz\r\n$1\r\nz\r\n"
        );
    }

    #[test]
    fn drawing_a_member_removes_nothing_and_says_nil_at_a_missing_key() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"only"]);
        assert_eq!(f.run(&[b"SRANDMEMBER", b"s"]), "$4\r\nonly\r\n");
        assert_eq!(f.run(&[b"SRANDMEMBER", b"s"]), "$4\r\nonly\r\n");
        assert_eq!(f.run(&[b"SCARD", b"s"]), ":1\r\n");

        assert_eq!(f.run(&[b"SRANDMEMBER", b"nope"]), "$-1\r\n");
        // The count form answers an empty array rather than a nil, which is the
        // pair of answers Redis gives and is not the pair it looks like.
        assert_eq!(f.run(&[b"SRANDMEMBER", b"nope", b"3"]), "*0\r\n");
        assert_eq!(f.run(&[b"SRANDMEMBER", b"nope", b"-3"]), "*0\r\n");
        // Asking for more than is there answers all of it once and not padding.
        assert_eq!(f.run(&[b"SRANDMEMBER", b"s", b"9"]), "*1\r\n$4\r\nonly\r\n");
    }

    #[test]
    fn a_pop_count_that_is_not_a_positive_number_says_so() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"a"]);
        let bad = "-ERR value is out of range, must be positive\r\n";
        assert_eq!(f.run(&[b"SPOP", b"s", b"-1"]), bad);
        assert_eq!(f.run(&[b"SPOP", b"s", b"abc"]), bad);
        assert_eq!(f.run(&[b"SCARD", b"s"]), ":1\r\n", "and took nothing");
        // Zero is allowed and is a real answer rather than an error.
        assert_eq!(f.run(&[b"SPOP", b"s", b"0"]), "*0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"s"]), ":1\r\n");
    }

    #[test]
    fn a_scan_walks_a_set_of_any_size_exactly_once() {
        let mut f = Fixture::new();
        let members: Vec<Vec<u8>> = (0..300).map(|i| format!("m{i}").into_bytes()).collect();
        let args: Vec<&[u8]> = [&b"SADD"[..], &b"s"[..]]
            .into_iter()
            .chain(members.iter().map(Vec::as_slice))
            .collect();
        f.run(&args);

        let mut seen = Vec::new();
        let mut cursor = "0".to_owned();
        loop {
            let reply = f.run(&[b"SSCAN", b"s", cursor.as_bytes()]);
            let (next, got) = split_scan(&reply);
            seen.extend(got);
            cursor = next;
            if cursor == "0" {
                break;
            }
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 300, "a walk saw a member twice or missed one");

        // A set small enough to be a listpack answers in one call whatever
        // cursor it was handed, which is what Redis does for that encoding.
        f.run(&[b"SADD", b"small", b"a", b"b", b"c"]);
        let (cursor, got) = split_scan(&f.run(&[b"SSCAN", b"small", b"0", b"COUNT", b"1"]));
        assert_eq!(cursor, "0");
        assert_eq!(got.len(), 3);
        // And a key that is not there is a finished scan of nothing.
        assert_eq!(f.run(&[b"SSCAN", b"nope", b"0"]), "*2\r\n$1\r\n0\r\n*0\r\n");
    }

    #[test]
    fn a_scan_takes_match_and_count_and_refuses_anything_else() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"aa", b"ab", b"ba", b"12", b"13"]);

        let (_, got) = split_scan(&f.run(&[b"SSCAN", b"s", b"0", b"MATCH", b"a*"]));
        let mut got = got;
        got.sort();
        assert_eq!(got, ["aa", "ab"]);

        // An integer member has no digits stored anywhere, so MATCH is the one
        // place a scan pays to write some.
        let (_, got) = split_scan(&f.run(&[b"SSCAN", b"s", b"0", b"MATCH", b"1?"]));
        let mut got = got;
        got.sort();
        assert_eq!(got, ["12", "13"]);

        assert_eq!(f.run(&[b"SSCAN", b"s", b"abc"]), "-ERR invalid cursor\r\n");
        assert_eq!(f.run(&[b"SSCAN", b"s", b"-1"]), "-ERR invalid cursor\r\n");
        assert_eq!(
            f.run(&[b"SSCAN", b"s", b"0", b"NOPE", b"1"]),
            "-ERR syntax error\r\n"
        );
        // A count under one is a syntax error and not a range error, which is
        // the odder of Redis's two answers and the reason it is copied exactly.
        assert_eq!(
            f.run(&[b"SSCAN", b"s", b"0", b"COUNT", b"0"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn moving_a_member_takes_it_off_one_set_and_puts_it_on_another() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"src", b"a", b"b"]);
        f.run(&[b"SADD", b"dst", b"c"]);

        assert_eq!(f.run(&[b"SMOVE", b"src", b"dst", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"SISMEMBER", b"src", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"SISMEMBER", b"dst", b"a"]), ":1\r\n");
        // A member that is not in the source is a zero and moves nothing.
        assert_eq!(f.run(&[b"SMOVE", b"src", b"dst", b"zz"]), ":0\r\n");
        assert_eq!(f.run(&[b"SCARD", b"dst"]), ":2\r\n");

        // A destination that does not exist gets made, and a source that runs
        // out goes away.
        assert_eq!(f.run(&[b"SMOVE", b"src", b"fresh", b"b"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"src"]), ":0\r\n");
        assert_eq!(f.run(&[b"SMEMBERS", b"fresh"]), "*1\r\n$1\r\nb\r\n");
    }

    #[test]
    fn moving_checks_the_types_in_the_order_redis_checks_them() {
        // Not the order it looks like it should be. A source that is not there
        // answers zero without ever looking at the destination, so this is a
        // zero and not a WRONGTYPE even though the destination is a string.
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        f.run(&[b"SADD", b"set", b"a"]);

        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        assert_eq!(f.run(&[b"SMOVE", b"nope", b"str", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"SMOVE", b"str", b"set", b"a"]), wrong);
        assert_eq!(f.run(&[b"SMOVE", b"set", b"str", b"a"]), wrong);
        assert_eq!(f.run(&[b"SPOP", b"str"]), wrong);
        assert_eq!(f.run(&[b"SRANDMEMBER", b"str"]), wrong);
        assert_eq!(f.run(&[b"SSCAN", b"str", b"0"]), wrong);
        assert_eq!(
            f.run(&[b"SISMEMBER", b"set", b"a"]),
            ":1\r\n",
            "and none of that moved anything"
        );
    }

    #[test]
    fn a_scan_leaves_nothing_half_written_when_its_arguments_are_wrong() {
        // SSCAN writes an outer array header before it walks, so it is the
        // command most likely to get bytes out in front of an error.
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"s", b"a"]);
        for bad in [
            &[b"SSCAN".as_slice(), b"s", b"abc"][..],
            &[b"SSCAN".as_slice(), b"s", b"0", b"COUNT", b"nope"][..],
            &[b"SSCAN".as_slice(), b"s", b"0", b"MATCH"][..],
        ] {
            let reply = f.run(bad);
            assert!(reply.starts_with("-ERR"), "got {reply}");
            assert!(!reply.contains('*'), "an array header went out in front");
        }
    }

    #[test]
    fn a_hash_writes_reads_and_deletes_its_fields() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]), ":2\r\n");
        assert_eq!(f.run(&[b"HSET", b"h", b"a", b"9"]), ":0\r\n", "a was there");
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n9\r\n");
        assert_eq!(f.run(&[b"HGET", b"h", b"nope"]), "$-1\r\n");
        assert_eq!(f.run(&[b"HGET", b"nokey", b"a"]), "$-1\r\n");
        assert_eq!(f.run(&[b"HLEN", b"h"]), ":2\r\n");
        assert_eq!(f.run(&[b"HEXISTS", b"h", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"HEXISTS", b"h", b"nope"]), ":0\r\n");
        assert_eq!(f.run(&[b"HSTRLEN", b"h", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"HSTRLEN", b"h", b"nope"]), ":0\r\n");

        // The value the client sent is `9`, so HGET h b must not find the `2`
        // that is a value. A search with a step of one would have.
        assert_eq!(f.run(&[b"HGET", b"h", b"2"]), "$-1\r\n");

        assert_eq!(f.run(&[b"HDEL", b"h", b"a", b"nope"]), ":1\r\n");
        assert_eq!(f.run(&[b"HDEL", b"h", b"b"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"EXISTS", b"h"]),
            ":0\r\n",
            "and losing the last field lost the key"
        );
    }

    #[test]
    fn hgetall_answers_a_map_on_resp3_and_the_same_pairs_flat_on_resp2() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        assert_eq!(f.run(&[b"HGETALL", b"h"]), "*2\r\n$1\r\na\r\n$1\r\n1\r\n");
        assert_eq!(f.run(&[b"HGETALL", b"nokey"]), "*0\r\n");
        assert_eq!(f.run(&[b"HKEYS", b"h"]), "*1\r\n$1\r\na\r\n");
        assert_eq!(f.run(&[b"HVALS", b"h"]), "*1\r\n$1\r\n1\r\n");
        assert_eq!(f.run(&[b"HKEYS", b"nokey"]), "*0\r\n");

        f.run(&[b"HELLO", b"3"]);
        assert_eq!(f.run(&[b"HGETALL", b"h"]), "%1\r\n$1\r\na\r\n$1\r\n1\r\n");
        assert_eq!(
            f.run(&[b"HGETALL", b"nokey"]),
            "%0\r\n",
            "a missing key is the empty hash and never a nil"
        );
        assert_eq!(
            f.run(&[b"HKEYS", b"h"]),
            "*1\r\n$1\r\na\r\n",
            "and the two that answer one side stay arrays"
        );
    }

    #[test]
    fn hmget_answers_once_per_field_and_hmset_answers_ok() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"HMSET", b"h", b"a", b"1", b"c", b"3"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"HMGET", b"h", b"a", b"b", b"c"]),
            "*3\r\n$1\r\n1\r\n$-1\r\n$1\r\n3\r\n",
            "the reply is positional, so b is a nil and not a gap"
        );
        assert_eq!(
            f.run(&[b"HMGET", b"nokey", b"a", b"b"]),
            "*2\r\n$-1\r\n$-1\r\n",
            "and a missing key is all nils rather than an empty array"
        );

        assert_eq!(f.run(&[b"HSETNX", b"h", b"a", b"9"]), ":0\r\n");
        assert_eq!(f.run(&[b"HSETNX", b"h", b"z", b"9"]), ":1\r\n");
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n1\r\n");
    }

    #[test]
    fn a_hash_counts_up_and_says_so_when_it_cannot() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"HINCRBY", b"h", b"n", b"5"]), ":5\r\n");
        assert_eq!(f.run(&[b"HINCRBY", b"h", b"n", b"-7"]), ":-2\r\n");
        assert_eq!(f.run(&[b"HGET", b"h", b"n"]), "$2\r\n-2\r\n");
        assert_eq!(
            f.run(&[b"HINCRBYFLOAT", b"h", b"f", b"10.5"]),
            "$4\r\n10.5\r\n",
            "a bulk string and not a double, on both protocols"
        );

        f.run(&[b"HSET", b"h", b"s", b"words"]);
        let bad = f.run(&[b"HINCRBY", b"h", b"s", b"1"]);
        assert!(
            bad.starts_with("-ERR hash value is not an integer"),
            "{bad}"
        );
        let bad = f.run(&[b"HINCRBY", b"h", b"n", b"nope"]);
        assert!(
            bad.starts_with("-ERR value is not an integer"),
            "a bad argument is not yet a hash value, {bad}"
        );
        assert_eq!(
            f.run(&[b"HGET", b"h", b"s"]),
            "$5\r\nwords\r\n",
            "and neither of them wrote anything"
        );
    }

    #[test]
    fn a_hash_scan_walks_every_pair_once_and_novalues_drops_half_of_it() {
        let mut f = Fixture::new();
        for i in 0..500 {
            let field = format!("field-{i}");
            let value = format!("value-{i}");
            f.run(&[b"HSET", b"h", field.as_bytes(), value.as_bytes()]);
        }

        let mut seen: Vec<String> = Vec::new();
        let mut cursor = "0".to_owned();
        loop {
            let reply = f.run(&[b"HSCAN", b"h", cursor.as_bytes(), b"COUNT", b"32"]);
            let (next, items) = scan_reply(&reply);
            assert_eq!(items.len() % 2, 0, "a pair went out half written");
            for pair in items.chunks(2) {
                assert_eq!(
                    pair[0].strip_prefix("field-"),
                    pair[1].strip_prefix("value-"),
                    "a field came back with someone else's value"
                );
                seen.push(pair[0].clone());
            }
            cursor = next;
            if cursor == "0" {
                break;
            }
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 500, "every field once and only once");

        let (_, items) = scan_reply(&f.run(&[b"HSCAN", b"h", b"0", b"NOVALUES", b"COUNT", b"32"]));
        assert!(
            items.iter().all(|s| s.starts_with("field-")),
            "NOVALUES still sent the values"
        );

        let (_, one) = scan_reply(&f.run(&[
            b"HSCAN",
            b"h",
            b"0",
            b"MATCH",
            b"field-499",
            b"COUNT",
            b"1000",
        ]));
        assert_eq!(one, ["field-499", "value-499"], "MATCH is on the field");
    }

    #[test]
    fn hrandfield_draws_what_it_was_asked_for_and_nests_values_on_resp3() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        assert_eq!(f.run(&[b"HRANDFIELD", b"h"]), "$1\r\na\r\n");
        assert_eq!(f.run(&[b"HRANDFIELD", b"nokey"]), "$-1\r\n");
        assert_eq!(f.run(&[b"HRANDFIELD", b"nokey", b"3"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"HRANDFIELD", b"h", b"3"]),
            "*1\r\n$1\r\na\r\n",
            "a positive count is capped at the size of the hash"
        );
        assert_eq!(
            f.run(&[b"HRANDFIELD", b"h", b"-3"]),
            "*3\r\n$1\r\na\r\n$1\r\na\r\n$1\r\na\r\n",
            "and a negative one repeats itself"
        );
        assert_eq!(
            f.run(&[b"HRANDFIELD", b"h", b"1", b"WITHVALUES"]),
            "*2\r\n$1\r\na\r\n$1\r\n1\r\n",
            "flat on RESP2"
        );

        f.run(&[b"HELLO", b"3"]);
        assert_eq!(
            f.run(&[b"HRANDFIELD", b"h", b"1", b"WITHVALUES"]),
            "*1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n",
            "and nested on RESP3, but still an array and never a map"
        );
    }

    #[test]
    fn every_hash_command_says_wrongtype_and_writes_nothing() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";

        for cmd in [
            &[b"HSET".as_slice(), b"str", b"f", b"v"][..],
            &[b"HMSET".as_slice(), b"str", b"f", b"v"][..],
            &[b"HSETNX".as_slice(), b"str", b"f", b"v"][..],
            &[b"HGET".as_slice(), b"str", b"f"][..],
            &[b"HMGET".as_slice(), b"str", b"f"][..],
            &[b"HDEL".as_slice(), b"str", b"f"][..],
            &[b"HLEN".as_slice(), b"str"][..],
            &[b"HEXISTS".as_slice(), b"str", b"f"][..],
            &[b"HSTRLEN".as_slice(), b"str", b"f"][..],
            &[b"HGETALL".as_slice(), b"str"][..],
            &[b"HKEYS".as_slice(), b"str"][..],
            &[b"HVALS".as_slice(), b"str"][..],
            &[b"HINCRBY".as_slice(), b"str", b"f", b"1"][..],
            &[b"HINCRBYFLOAT".as_slice(), b"str", b"f", b"1"][..],
            &[b"HRANDFIELD".as_slice(), b"str"][..],
            &[b"HRANDFIELD".as_slice(), b"str", b"2"][..],
            &[b"HSCAN".as_slice(), b"str", b"0"][..],
        ] {
            let reply = f.run(cmd);
            assert_eq!(reply, wrong, "{:?}", cmd[0]);
        }
        assert_eq!(
            f.run(&[b"GET", b"str"]),
            "$1\r\nv\r\n",
            "and none of them touched the value"
        );
    }

    #[test]
    fn a_hash_scan_leaves_nothing_half_written_when_its_arguments_are_wrong() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"f", b"v"]);
        for bad in [
            &[b"HSCAN".as_slice(), b"h", b"abc"][..],
            &[b"HSCAN".as_slice(), b"h", b"0", b"COUNT", b"nope"][..],
            &[b"HSCAN".as_slice(), b"h", b"0", b"COUNT", b"0"][..],
            &[b"HSCAN".as_slice(), b"h", b"0", b"MATCH"][..],
        ] {
            let reply = f.run(bad);
            assert!(reply.starts_with("-ERR"), "got {reply}");
            assert!(!reply.contains('*'), "an array header went out in front");
        }
    }

    #[test]
    fn a_field_deadline_goes_on_and_comes_back_in_all_four_units() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]);
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"3", b"a", b"b", b"nope"]),
            "*3\r\n:100\r\n:-1\r\n:-2\r\n",
            "one answer per field, and the two sentinels are TTL's own"
        );

        // The same deadline in the other three units, all of them derived from
        // the one number the store kept.
        let ms = int_reply(&f.run(&[b"HPTTL", b"h", b"FIELDS", b"1", b"a"]));
        assert!((99_000..=100_000).contains(&ms), "got {ms}");
        let at = int_reply(&f.run(&[b"HEXPIRETIME", b"h", b"FIELDS", b"1", b"a"]));
        let at_ms = int_reply(&f.run(&[b"HPEXPIRETIME", b"h", b"FIELDS", b"1", b"a"]));
        assert_eq!(at, at_ms.div_euclid(1000) + i64::from(at_ms % 1000 != 0));
        assert!(at_ms > 1_700_000_000_000, "an absolute moment, got {at_ms}");

        assert_eq!(
            f.run(&[b"HPERSIST", b"h", b"FIELDS", b"3", b"a", b"b", b"nope"]),
            "*3\r\n:1\r\n:-1\r\n:-2\r\n",
            "one for the deadline taken off, and it does not say what it was"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n"
        );
        assert_eq!(
            f.run(&[b"HGET", b"h", b"a"]),
            "$1\r\n1\r\n",
            "and the field is still there with the value it had"
        );
    }

    #[test]
    fn a_deadline_that_has_already_gone_deletes_the_field_now() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]);
        assert_eq!(
            f.run(&[b"HEXPIREAT", b"h", b"1", b"FIELDS", b"1", b"a"]),
            "*1\r\n:2\r\n",
            "two, and not one, because nothing was stored"
        );
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$-1\r\n");
        assert_eq!(f.run(&[b"HLEN", b"h"]), ":1\r\n");

        assert_eq!(
            f.run(&[b"HPEXPIREAT", b"h", b"1", b"FIELDS", b"1", b"b"]),
            "*1\r\n:2\r\n"
        );
        assert_eq!(
            f.run(&[b"EXISTS", b"h"]),
            ":0\r\n",
            "and the last field going took the key with it"
        );

        // Zero is a delete and not an error, where minus one is an error. That
        // is Redis's split and it is easy to get backwards.
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"0", b"FIELDS", b"1", b"a"]),
            "*1\r\n:2\r\n"
        );
    }

    #[test]
    fn a_field_is_gone_once_its_moment_passes() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]);
        assert_eq!(
            f.run(&[b"HPEXPIRE", b"h", b"20", b"FIELDS", b"1", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n1\r\n", "not yet");

        // Time moves once per turn of the event loop and nowhere else, so a
        // test moves it by hand rather than by sleeping. There is nothing to
        // sleep for: the deadline is a number and so is the clock.
        f.server.advance_clock_ms(60);
        assert_eq!(f.run(&[b"HLEN", b"h"]), ":1\r\n");
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"HGETALL", b"h"]),
            "*2\r\n$1\r\nb\r\n$1\r\n2\r\n",
            "and the walks do not hand back a field that has expired"
        );
    }

    #[test]
    fn a_missing_key_answers_the_no_field_sentinel_for_every_field() {
        let mut f = Fixture::new();
        for cmd in [
            &[
                b"HEXPIRE".as_slice(),
                b"nokey",
                b"100",
                b"FIELDS",
                b"2",
                b"a",
                b"b",
            ][..],
            &[b"HTTL".as_slice(), b"nokey", b"FIELDS", b"2", b"a", b"b"][..],
            &[b"HPTTL".as_slice(), b"nokey", b"FIELDS", b"2", b"a", b"b"][..],
            &[
                b"HEXPIRETIME".as_slice(),
                b"nokey",
                b"FIELDS",
                b"2",
                b"a",
                b"b",
            ][..],
            &[
                b"HPERSIST".as_slice(),
                b"nokey",
                b"FIELDS",
                b"2",
                b"a",
                b"b",
            ][..],
        ] {
            assert_eq!(f.run(cmd), "*2\r\n:-2\r\n:-2\r\n", "{:?}", cmd[0]);
        }
    }

    #[test]
    fn writing_a_field_clears_the_deadline_that_was_on_it() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        f.run(&[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"a"]);
        f.run(&[b"HSET", b"h", b"a", b"2"]);
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n",
            "Redis has done this since 7.4, and it is why HGETEX exists"
        );
    }

    #[test]
    fn the_four_conditions_reach_the_store_the_way_they_were_written() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"100", b"XX", b"FIELDS", b"1", b"a"]),
            "*1\r\n:0\r\n",
            "XX on a field with no deadline changes nothing"
        );
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"100", b"NX", b"FIELDS", b"1", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"200", b"NX", b"FIELDS", b"1", b"a"]),
            "*1\r\n:0\r\n",
            "and NX will not move one that is already there"
        );
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"50", b"GT", b"FIELDS", b"1", b"a"]),
            "*1\r\n:0\r\n"
        );
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"500", b"GT", b"FIELDS", b"1", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"HEXPIRE", b"h", b"50", b"LT", b"FIELDS", b"1", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:50\r\n"
        );
    }

    #[test]
    fn the_field_ttl_family_leaves_nothing_half_written_on_a_bad_argument() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        for (bad, want) in [
            (
                &[b"HEXPIRE".as_slice(), b"h", b"-1", b"FIELDS", b"1", b"a"][..],
                "-ERR invalid expire time, must be >= 0",
            ),
            (
                &[
                    b"HEXPIRE".as_slice(),
                    b"h",
                    b"9999999999999999",
                    b"FIELDS",
                    b"1",
                    b"a",
                ][..],
                "-ERR invalid expire time in 'hexpire' command",
            ),
            (
                &[b"HEXPIRE".as_slice(), b"h", b"100", b"FIELD", b"1", b"a"][..],
                "-ERR wrong number of arguments for 'hexpire' command",
            ),
            (
                &[b"HEXPIRE".as_slice(), b"h", b"100", b"FIELDS", b"0", b"a"][..],
                "-ERR Parameter `numFields` should be greater than 0",
            ),
            (
                &[b"HEXPIRE".as_slice(), b"h", b"100", b"FIELDS", b"2", b"a"][..],
                "-ERR wrong number of arguments",
            ),
            (
                &[b"HTTL".as_slice(), b"h", b"FIELDS", b"3", b"a", b"b"][..],
                "-ERR wrong number of arguments",
            ),
        ] {
            let reply = f.run(bad);
            assert!(reply.starts_with(want), "wanted {want}, got {reply}");
            assert!(!reply.contains('*'), "an array header went out in front");
        }
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n",
            "and not one of them put a deadline on anything"
        );
    }

    #[test]
    fn every_field_ttl_command_says_wrongtype_and_writes_nothing() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";

        for cmd in [
            &[b"HEXPIRE".as_slice(), b"str", b"100", b"FIELDS", b"1", b"f"][..],
            &[
                b"HPEXPIRE".as_slice(),
                b"str",
                b"100",
                b"FIELDS",
                b"1",
                b"f",
            ][..],
            &[
                b"HEXPIREAT".as_slice(),
                b"str",
                b"9999999999",
                b"FIELDS",
                b"1",
                b"f",
            ][..],
            &[
                b"HPEXPIREAT".as_slice(),
                b"str",
                b"9999999999999",
                b"FIELDS",
                b"1",
                b"f",
            ][..],
            &[b"HTTL".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
            &[b"HPTTL".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
            &[b"HEXPIRETIME".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
            &[b"HPEXPIRETIME".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
            &[b"HPERSIST".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
        ] {
            assert_eq!(f.run(cmd), wrong, "{:?}", cmd[0]);
        }
        assert_eq!(
            f.run(&[b"GET", b"str"]),
            "$1\r\nv\r\n",
            "and none of them touched the value"
        );
    }

    #[test]
    fn hgetdel_hands_the_value_out_and_then_takes_the_field() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]);
        assert_eq!(
            f.run(&[b"HGETDEL", b"h", b"FIELDS", b"2", b"a", b"nope"]),
            "*2\r\n$1\r\n1\r\n$-1\r\n",
            "positional, so the field that was not there is a nil in its place"
        );
        assert_eq!(f.run(&[b"HLEN", b"h"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"HGETDEL", b"nokey", b"FIELDS", b"1", b"a"]),
            "*1\r\n$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"HGETDEL", b"h", b"FIELDS", b"1", b"b"]),
            "*1\r\n$1\r\n2\r\n"
        );
        assert_eq!(
            f.run(&[b"EXISTS", b"h"]),
            ":0\r\n",
            "and the last field took the key"
        );
    }

    #[test]
    fn hgetex_reads_and_moves_the_deadline_in_one_command() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        assert_eq!(
            f.run(&[b"HGETEX", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n$1\r\n1\r\n"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n",
            "no option means leave it alone, which is the one place this is not GETEX"
        );

        f.run(&[b"HGETEX", b"h", b"EX", b"100", b"FIELDS", b"1", b"a"]);
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:100\r\n"
        );
        f.run(&[b"HGETEX", b"h", b"FIELDS", b"1", b"a"]);
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:100\r\n",
            "and a plain read really does leave it alone"
        );
        assert_eq!(
            f.run(&[b"HGETEX", b"h", b"PERSIST", b"FIELDS", b"1", b"a"]),
            "*1\r\n$1\r\n1\r\n"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n"
        );

        assert_eq!(
            f.run(&[b"HGETEX", b"h", b"EXAT", b"1", b"FIELDS", b"1", b"a"]),
            "*1\r\n$1\r\n1\r\n",
            "the value goes out before the deadline that has already gone is applied"
        );
        assert_eq!(f.run(&[b"EXISTS", b"h"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"HGETEX", b"nokey", b"EX", b"100", b"FIELDS", b"1", b"a"]),
            "*1\r\n$-1\r\n"
        );
    }

    #[test]
    fn hsetex_writes_all_of_it_or_none_of_it() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"HSETEX", b"h", b"FIELDS", b"1", b"a", b"1"]),
            ":1\r\n"
        );
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n1\r\n");
        assert_eq!(
            f.run(&[
                b"HSETEX", b"h", b"FNX", b"FIELDS", b"2", b"a", b"9", b"new", b"9"
            ]),
            ":0\r\n",
            "FNX wants every field named to be missing"
        );
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n1\r\n");
        assert_eq!(
            f.run(&[b"HEXISTS", b"h", b"new"]),
            ":0\r\n",
            "and none of the list was written"
        );
        assert_eq!(
            f.run(&[
                b"HSETEX", b"h", b"FXX", b"FIELDS", b"2", b"a", b"9", b"nope", b"9"
            ]),
            ":0\r\n",
            "and FXX wants every one of them to be there"
        );
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n1\r\n");
        assert_eq!(
            f.run(&[b"HSETEX", b"h", b"FXX", b"FIELDS", b"1", b"a", b"9"]),
            ":1\r\n"
        );
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n9\r\n");

        assert_eq!(
            f.run(&[b"HSETEX", b"gone", b"FXX", b"FIELDS", b"1", b"a", b"1"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"EXISTS", b"gone"]),
            ":0\r\n",
            "a key with no fields cannot meet FXX and is not created trying"
        );
    }

    #[test]
    fn hsetex_clears_the_deadline_unless_it_is_told_to_keep_it() {
        let mut f = Fixture::new();
        f.run(&[b"HSETEX", b"h", b"EX", b"100", b"FIELDS", b"1", b"a", b"1"]);
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:100\r\n"
        );

        f.run(&[b"HSETEX", b"h", b"KEEPTTL", b"FIELDS", b"1", b"a", b"2"]);
        assert_eq!(f.run(&[b"HGET", b"h", b"a"]), "$1\r\n2\r\n");
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:100\r\n",
            "KEEPTTL put back what the write cleared"
        );

        f.run(&[b"HSETEX", b"h", b"FIELDS", b"1", b"a", b"3"]);
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n",
            "and without it a write clears the deadline the way HSET does"
        );

        // Any order, because Redis reads these in a loop and not in a fixed
        // sequence.
        assert_eq!(
            f.run(&[
                b"HSETEX", b"h", b"PX", b"100000", b"FXX", b"FIELDS", b"1", b"a", b"4"
            ]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:100\r\n"
        );

        assert_eq!(
            f.run(&[b"HSETEX", b"h", b"EXAT", b"1", b"FIELDS", b"1", b"a", b"5"]),
            ":1\r\n",
            "written, and not the separate code the HEXPIRE family has for this"
        );
        assert_eq!(
            f.run(&[b"EXISTS", b"h"]),
            ":0\r\n",
            "and storing it and then removing it emptied the hash"
        );
    }

    #[test]
    fn the_last_three_hash_commands_word_their_mistakes_their_own_way() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"h", b"a", b"1"]);
        for (bad, want) in [
            // HGETDEL has three sentences of its own for these three mistakes.
            (
                &[b"HGETDEL".as_slice(), b"h", b"FIELDS", b"0", b"a"][..],
                "-ERR Number of fields must be a positive integer",
            ),
            (
                &[b"HGETDEL".as_slice(), b"h", b"FIELDS", b"2", b"a"][..],
                "-ERR The `numfields` parameter must match the number of arguments",
            ),
            (
                &[b"HGETDEL".as_slice(), b"h", b"FIELD", b"1", b"a"][..],
                "-ERR Mandatory argument FIELDS is missing or not at the right position",
            ),
            // And HGETEX and HSETEX have three different ones between them.
            (
                &[b"HGETEX".as_slice(), b"h", b"FIELDS", b"0", b"a"][..],
                "-ERR invalid number of fields",
            ),
            (
                &[b"HGETEX".as_slice(), b"h", b"FIELDS", b"2", b"a"][..],
                "-ERR wrong number of arguments",
            ),
            (
                &[b"HGETEX".as_slice(), b"h", b"FIELD", b"1", b"a"][..],
                "-ERR unknown argument: FIELD",
            ),
            (
                &[
                    b"HGETEX".as_slice(),
                    b"h",
                    b"KEEPTTL",
                    b"FIELDS",
                    b"1",
                    b"a",
                ][..],
                "-ERR unknown argument: KEEPTTL",
            ),
            (
                &[
                    b"HGETEX".as_slice(),
                    b"h",
                    b"EX",
                    b"100",
                    b"PERSIST",
                    b"FIELDS",
                    b"1",
                    b"a",
                ][..],
                "-ERR Only one of EX, PX, EXAT, PXAT or PERSIST arguments can be specified",
            ),
            (
                &[
                    b"HSETEX".as_slice(),
                    b"h",
                    b"EX",
                    b"1",
                    b"KEEPTTL",
                    b"FIELDS",
                    b"1",
                    b"a",
                    b"1",
                ][..],
                "-ERR Only one of EX, PX, EXAT, PXAT or KEEPTTL arguments can be specified",
            ),
            (
                &[
                    b"HSETEX".as_slice(),
                    b"h",
                    b"FNX",
                    b"FXX",
                    b"FIELDS",
                    b"1",
                    b"a",
                    b"1",
                ][..],
                "-ERR Only one of FXX or FNX arguments can be specified",
            ),
            (
                &[
                    b"HSETEX".as_slice(),
                    b"h",
                    b"FIELDS",
                    b"2",
                    b"a",
                    b"1",
                    b"b",
                ][..],
                "-ERR wrong number of arguments",
            ),
            (
                &[
                    b"HGETEX".as_slice(),
                    b"h",
                    b"EX",
                    b"-1",
                    b"FIELDS",
                    b"1",
                    b"a",
                ][..],
                "-ERR invalid expire time, must be >= 0",
            ),
            (
                &[
                    b"HGETEX".as_slice(),
                    b"h",
                    b"PXAT",
                    b"99999999999999",
                    b"FIELDS",
                    b"1",
                    b"a",
                ][..],
                "-ERR invalid expire time in 'hgetex' command",
            ),
            (
                &[
                    b"HSETEX".as_slice(),
                    b"h",
                    b"EX",
                    b"abc",
                    b"FIELDS",
                    b"1",
                    b"a",
                    b"1",
                ][..],
                "-ERR value is not an integer or out of range",
            ),
        ] {
            let reply = f.run(bad);
            assert!(reply.starts_with(want), "wanted {want}, got {reply}");
            assert!(!reply.contains('*'), "an array header went out in front");
        }
        assert_eq!(
            f.run(&[b"HGET", b"h", b"a"]),
            "$1\r\n1\r\n",
            "and not one of them wrote anything"
        );
        assert_eq!(
            f.run(&[b"HTTL", b"h", b"FIELDS", b"1", b"a"]),
            "*1\r\n:-1\r\n"
        );
    }

    #[test]
    fn the_last_three_hash_commands_say_wrongtype_and_write_nothing() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        for cmd in [
            &[b"HGETDEL".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
            &[b"HGETEX".as_slice(), b"str", b"FIELDS", b"1", b"f"][..],
            &[
                b"HGETEX".as_slice(),
                b"str",
                b"EX",
                b"100",
                b"FIELDS",
                b"1",
                b"f",
            ][..],
            &[b"HSETEX".as_slice(), b"str", b"FIELDS", b"1", b"f", b"v"][..],
        ] {
            assert_eq!(f.run(cmd), wrong, "{:?}", cmd[0]);
        }
        assert_eq!(f.run(&[b"GET", b"str"]), "$1\r\nv\r\n");
    }

    /// The two orders `HIMPORT` juggles, which are not the same order.
    ///
    /// Values arrive in the order the fields were declared in and the hash is
    /// built in sorted order, so the first value is not generally the first
    /// field. And the sort is by length before bytes, which nothing else here
    /// sorts names with: `b` comes before `aa` where a plain byte comparison
    /// would put `aa` first. Both read off 8.10.1.
    #[test]
    fn himport_writes_declared_values_into_sorted_fields() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"b", b"aa", b"a"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k", b"shape", b"1", b"2", b"3"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"HKEYS", b"k"]), bulks(&["a", "b", "aa"]));
        assert_eq!(
            f.run(&[b"HGETALL", b"k"]),
            bulks(&["a", "3", "b", "1", "aa", "2"])
        );
    }

    /// It replaces the key rather than writing over it, so a field the fieldset
    /// does not name is gone afterwards and so is the deadline.
    #[test]
    fn himport_set_replaces_the_whole_key() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"k", b"gone", b"old", b"a", b"old"]);
        f.run(&[b"EXPIRE", b"k", b"100"]);
        f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"a", b"b"]);
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k", b"shape", b"1", b"2"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"HGETALL", b"k"]), bulks(&["a", "1", "b", "2"]));
        assert_eq!(f.run(&[b"TTL", b"k"]), ":-1\r\n");
    }

    /// A fieldset is connection state. `SELECT` leaves them alone and `RESET`
    /// throws them away, and a key built from one outlives it.
    #[test]
    fn himport_fieldsets_belong_to_the_connection_and_not_to_the_keyspace() {
        let mut f = Fixture::new();
        f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"a"]);
        f.run(&[b"SELECT", b"1"]);
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k", b"shape", b"1"]),
            "+OK\r\n"
        );
        f.run(&[b"SELECT", b"0"]);
        assert_eq!(f.run(&[b"RESET"]), "+RESET\r\n");
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k2", b"shape", b"1"]),
            "-ERR no such fieldset\r\n"
        );
    }

    /// Which complaint wins when a line is wrong in more than one place.
    ///
    /// The type of the key beats both of the others, so a `HIMPORT SET` against
    /// a string is a WRONGTYPE even when the fieldset is missing too, which is
    /// the ordering a real server has and not the one the argument order
    /// suggests.
    #[test]
    fn himport_complains_in_the_order_a_real_server_does() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"a", b"b"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"str", b"nope", b"1"]),
            wrong,
            "the type beats a missing fieldset"
        );
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"str", b"shape", b"1"]),
            wrong,
            "and it beats a value count that does not fit"
        );
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k", b"nope", b"1"]),
            "-ERR no such fieldset\r\n"
        );
        // One sentence for too few and for too many alike.
        for values in [&[b"1".as_slice()][..], &[b"1".as_slice(), b"2", b"3"][..]] {
            let mut line: Vec<&[u8]> = vec![b"HIMPORT", b"SET", b"k", b"shape"];
            line.extend_from_slice(values);
            assert_eq!(
                f.run(&line),
                "-ERR value count does not match fieldset field count\r\n",
                "{} values into two fields",
                values.len()
            );
        }
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n");
    }

    /// The arity of each subcommand, and the unknown one.
    #[test]
    fn himport_checks_each_subcommand_count_under_its_own_name() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"HIMPORT"]),
            "-ERR wrong number of arguments for 'himport' command\r\n"
        );
        for (rest, name) in [
            (&["PREPARE"][..], "prepare"),
            (&["PREPARE", "fs"][..], "prepare"),
            (&["SET"][..], "set"),
            (&["SET", "k"][..], "set"),
            (&["SET", "k", "fs"][..], "set"),
            (&["DISCARD"][..], "discard"),
            (&["DISCARD", "a", "b"][..], "discard"),
            (&["DISCARDALL", "x"][..], "discardall"),
        ] {
            let mut line: Vec<&[u8]> = vec![b"HIMPORT"];
            line.extend(rest.iter().map(|a| a.as_bytes()));
            assert_eq!(
                f.run(&line),
                format!("-ERR wrong number of arguments for 'himport|{name}' command\r\n"),
                "HIMPORT {}",
                rest.join(" ")
            );
        }
        assert_eq!(
            f.run(&[b"HIMPORT", b"NOPE", b"x"]),
            "-ERR unknown subcommand 'NOPE'. Try HIMPORT HELP.\r\n"
        );
    }

    /// A `PREPARE` that fails leaves the name pointing where it pointed, which
    /// is the answer of the two that could not be guessed from outside.
    #[test]
    fn a_failed_himport_prepare_leaves_the_old_fieldset_alone() {
        let mut f = Fixture::new();
        f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"a", b"b"]);
        assert_eq!(
            f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"c", b"c"]),
            "-ERR duplicate field name in fieldset\r\n"
        );
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k", b"shape", b"1", b"2"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"HGETALL", b"k"]), bulks(&["a", "1", "b", "2"]));
    }

    /// Preparing the same name twice replaces it, and the two discards count
    /// what they took rather than answering OK.
    #[test]
    fn himport_prepare_replaces_and_the_discards_count() {
        let mut f = Fixture::new();
        f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"a", b"b"]);
        f.run(&[b"HIMPORT", b"PREPARE", b"shape", b"z"]);
        assert_eq!(
            f.run(&[b"HIMPORT", b"SET", b"k", b"shape", b"1"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"HGETALL", b"k"]), bulks(&["z", "1"]));

        assert_eq!(f.run(&[b"HIMPORT", b"DISCARD", b"shape"]), ":1\r\n");
        assert_eq!(f.run(&[b"HIMPORT", b"DISCARD", b"shape"]), ":0\r\n");
        f.run(&[b"HIMPORT", b"PREPARE", b"one", b"a"]);
        f.run(&[b"HIMPORT", b"PREPARE", b"two", b"a"]);
        assert_eq!(f.run(&[b"HIMPORT", b"DISCARDALL"]), ":2\r\n");
        assert_eq!(f.run(&[b"HIMPORT", b"DISCARDALL"]), ":0\r\n");
    }

    /// The one integer of a single element array reply.
    /// The number out of a plain integer reply.
    ///
    /// [`int_reply`] is the same thing wrapped in a one element array, which is
    /// the shape every hash field command answers in.
    fn int(reply: &str) -> i64 {
        let body = reply
            .strip_prefix(':')
            .and_then(|s| s.strip_suffix("\r\n"))
            .unwrap_or_else(|| panic!("wanted an integer, got {reply}"));
        body.parse().expect("an integer")
    }

    fn int_reply(reply: &str) -> i64 {
        let body = reply
            .strip_prefix("*1\r\n:")
            .and_then(|s| s.strip_suffix("\r\n"))
            .unwrap_or_else(|| panic!("wanted one integer, got {reply}"));
        body.parse().expect("an integer")
    }

    /// The cursor and the flat items of a scan reply.
    fn scan_reply(reply: &str) -> (String, Vec<String>) {
        let mut lines = reply.split("\r\n");
        assert_eq!(lines.next(), Some("*2"), "got {reply}");
        lines.next().expect("the cursor header");
        let cursor = lines.next().expect("a cursor").to_owned();
        let header = lines.next().expect("an item count");
        let n: usize = header[1..].parse().expect("a count");
        let mut items = Vec::with_capacity(n);
        for _ in 0..n {
            lines.next().expect("an item header");
            items.push(lines.next().expect("an item").to_owned());
        }
        (cursor, items)
    }

    /// The members of a set reply, sorted, since none of these promise an
    /// order and a test that asserted one would be asserting an accident.
    fn sorted(reply: &str) -> Vec<String> {
        let mut lines = reply.split("\r\n");
        let header = lines.next().expect("a header");
        assert!(
            header.starts_with('*') || header.starts_with('~'),
            "got {reply}"
        );
        let n: usize = header[1..].parse().expect("a member count");
        let mut got = Vec::with_capacity(n);
        for _ in 0..n {
            lines.next().expect("a member header");
            got.push(lines.next().expect("a member").to_owned());
        }
        got.sort();
        got
    }

    #[test]
    fn the_algebra_answers_what_the_sets_share_and_do_not() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"a", b"1", b"2", b"3"]);
        f.run(&[b"SADD", b"b", b"2", b"3", b"4"]);
        f.run(&[b"SADD", b"c", b"3", b"4", b"5"]);

        assert_eq!(sorted(&f.run(&[b"SINTER", b"a", b"b", b"c"])), ["3"]);
        assert_eq!(
            sorted(&f.run(&[b"SUNION", b"a", b"b", b"c"])),
            ["1", "2", "3", "4", "5"]
        );
        assert_eq!(sorted(&f.run(&[b"SDIFF", b"a", b"b"])), ["1"]);
        assert_eq!(sorted(&f.run(&[b"SINTER", b"a"])), ["1", "2", "3"]);

        // A key that is not there is an empty set, which empties an
        // intersection and does nothing at all to a union.
        assert_eq!(f.run(&[b"SINTER", b"a", b"nope"]), "*0\r\n");
        assert_eq!(sorted(&f.run(&[b"SUNION", b"a", b"nope"])), ["1", "2", "3"]);
        assert_eq!(f.run(&[b"SDIFF", b"nope", b"a"]), "*0\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":3\r\n", "and none of it made a key");
    }

    #[test]
    fn the_algebra_answers_a_set_on_resp3_and_an_array_on_resp2() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"a", b"x"]);
        assert_eq!(f.run(&[b"SINTER", b"a"]), "*1\r\n$1\r\nx\r\n");
        assert_eq!(f.run(&[b"SUNION", b"a"]), "*1\r\n$1\r\nx\r\n");
        assert_eq!(f.run(&[b"SDIFF", b"a"]), "*1\r\n$1\r\nx\r\n");

        f.run(&[b"HELLO", b"3"]);
        assert_eq!(f.run(&[b"SINTER", b"a"]), "~1\r\n$1\r\nx\r\n");
        assert_eq!(f.run(&[b"SUNION", b"a"]), "~1\r\n$1\r\nx\r\n");
        assert_eq!(f.run(&[b"SDIFF", b"a"]), "~1\r\n$1\r\nx\r\n");
        assert_eq!(f.run(&[b"SINTER", b"nope"]), "~0\r\n");
    }

    #[test]
    fn a_store_form_writes_a_key_and_answers_how_big_it_is() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"a", b"1", b"2", b"3"]);
        f.run(&[b"SADD", b"b", b"2", b"3", b"4"]);

        assert_eq!(f.run(&[b"SINTERSTORE", b"d", b"a", b"b"]), ":2\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", b"d"])), ["2", "3"]);
        assert_eq!(f.run(&[b"SUNIONSTORE", b"d", b"a", b"b"]), ":4\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", b"d"])), ["1", "2", "3", "4"]);
        assert_eq!(f.run(&[b"SDIFFSTORE", b"d", b"a", b"b"]), ":1\r\n");
        assert_eq!(f.run(&[b"SMEMBERS", b"d"]), "*1\r\n$1\r\n1\r\n");

        // An empty answer deletes the destination rather than leaving an empty
        // set behind, and the destination may be one of the sources.
        assert_eq!(f.run(&[b"SDIFFSTORE", b"d", b"a", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"d"]), ":0\r\n");
        assert_eq!(f.run(&[b"SINTERSTORE", b"a", b"a", b"b"]), ":2\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", b"a"])), ["2", "3"]);

        // And a destination holding something else is overwritten, the same way
        // SET overwrites, rather than refused.
        f.run(&[b"SET", b"str", b"v"]);
        assert_eq!(f.run(&[b"SUNIONSTORE", b"str", b"b"]), ":3\r\n");
        assert_eq!(f.run(&[b"TYPE", b"str"]), "+set\r\n");
    }

    #[test]
    fn sintercard_counts_without_building_and_stops_at_a_limit() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"a", b"1", b"2", b"3", b"4"]);
        f.run(&[b"SADD", b"b", b"2", b"3", b"4", b"5"]);

        assert_eq!(f.run(&[b"SINTERCARD", b"2", b"a", b"b"]), ":3\r\n");
        assert_eq!(
            f.run(&[b"SINTERCARD", b"2", b"a", b"b", b"LIMIT", b"2"]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"SINTERCARD", b"2", b"a", b"b", b"LIMIT", b"0"]),
            ":3\r\n",
            "a limit of zero is no limit"
        );
        assert_eq!(f.run(&[b"SINTERCARD", b"1", b"a"]), ":4\r\n");
        assert_eq!(f.run(&[b"SINTERCARD", b"2", b"a", b"nope"]), ":0\r\n");

        // The counted keys are what make its three error messages its own.
        assert_eq!(
            f.run(&[b"SINTERCARD", b"0", b"a"]),
            "-ERR numkeys should be greater than 0\r\n"
        );
        assert_eq!(
            f.run(&[b"SINTERCARD", b"abc", b"a"]),
            "-ERR numkeys should be greater than 0\r\n"
        );
        assert_eq!(
            f.run(&[b"SINTERCARD", b"3", b"a", b"b"]),
            "-ERR Number of keys can't be greater than number of args\r\n"
        );
        assert_eq!(
            f.run(&[b"SINTERCARD", b"2", b"a", b"b", b"LIMIT", b"-1"]),
            "-ERR LIMIT can't be negative\r\n"
        );
        assert_eq!(
            f.run(&[b"SINTERCARD", b"2", b"a", b"b", b"NOPE", b"1"]),
            "-ERR syntax error\r\n"
        );
        // A key really can be called LIMIT, which is why the count exists.
        f.run(&[b"SADD", b"LIMIT", b"2"]);
        assert_eq!(f.run(&[b"SINTERCARD", b"2", b"a", b"LIMIT"]), ":1\r\n");
    }

    /// The two Redis 8.10 added, which are SINTERCARD's shape over a union and
    /// over a difference. Every number here was read off 8.10.1 first.
    #[test]
    fn sunioncard_and_sdiffcard_count_without_building() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"a", b"1", b"2", b"3", b"4"]);
        f.run(&[b"SADD", b"b", b"3", b"4", b"5", b"6"]);

        assert_eq!(f.run(&[b"SUNIONCARD", b"2", b"a", b"b"]), ":6\r\n");
        assert_eq!(
            f.run(&[b"SUNIONCARD", b"2", b"a", b"b", b"LIMIT", b"2"]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"SUNIONCARD", b"2", b"a", b"b", b"LIMIT", b"0"]),
            ":6\r\n",
            "a limit of zero is no limit"
        );
        assert_eq!(f.run(&[b"SUNIONCARD", b"1", b"a"]), ":4\r\n");
        assert_eq!(
            f.run(&[b"SUNIONCARD", b"2", b"a", b"nope"]),
            ":4\r\n",
            "a missing key adds nothing to a union"
        );

        assert_eq!(f.run(&[b"SDIFFCARD", b"2", b"a", b"b"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"SDIFFCARD", b"2", b"a", b"b", b"LIMIT", b"1"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"SDIFFCARD", b"2", b"b", b"a"]),
            ":2\r\n",
            "a difference is not symmetric"
        );
        assert_eq!(f.run(&[b"SDIFFCARD", b"1", b"a"]), ":4\r\n");
        assert_eq!(f.run(&[b"SDIFFCARD", b"2", b"a", b"nope"]), ":4\r\n");
        assert_eq!(
            f.run(&[b"SDIFFCARD", b"2", b"nope", b"a"]),
            ":0\r\n",
            "nothing taken away from nothing"
        );

        // The same three messages SINTERCARD has, because the line is the same
        // line and is parsed once for all three.
        for name in [b"SUNIONCARD".as_slice(), b"SDIFFCARD".as_slice()] {
            assert_eq!(
                f.run(&[name, b"0", b"a"]),
                "-ERR numkeys should be greater than 0\r\n"
            );
            assert_eq!(
                f.run(&[name, b"abc", b"a"]),
                "-ERR numkeys should be greater than 0\r\n"
            );
            assert_eq!(
                f.run(&[name, b"-1", b"a"]),
                "-ERR numkeys should be greater than 0\r\n"
            );
            assert_eq!(
                f.run(&[name, b"3", b"a", b"b"]),
                "-ERR Number of keys can't be greater than number of args\r\n"
            );
            assert_eq!(
                f.run(&[name, b"2", b"a", b"b", b"LIMIT", b"-1"]),
                "-ERR LIMIT can't be negative\r\n"
            );
            assert_eq!(
                f.run(&[name, b"2", b"a", b"b", b"LIMIT", b"abc"]),
                "-ERR LIMIT can't be negative\r\n",
                "a LIMIT that is not a number gets the negative message too"
            );
            assert_eq!(
                f.run(&[name, b"2", b"a", b"b", b"NOPE", b"1"]),
                "-ERR syntax error\r\n"
            );
            assert_eq!(
                f.run(&[name, b"2", b"a", b"b", b"LIMIT"]),
                "-ERR syntax error\r\n"
            );
            assert_eq!(
                f.run(&[name, b"2", b"a", b"b", b"LIMIT", b"1", b"X"]),
                "-ERR syntax error\r\n"
            );
        }

        // And a key called LIMIT is a key, here as much as on SINTERCARD.
        f.run(&[b"SADD", b"LIMIT", b"2"]);
        assert_eq!(f.run(&[b"SUNIONCARD", b"2", b"a", b"LIMIT"]), ":4\r\n");
        assert_eq!(f.run(&[b"SDIFFCARD", b"2", b"a", b"LIMIT"]), ":3\r\n");
    }

    #[test]
    fn the_algebra_answers_wrongtype_before_it_writes_anything() {
        let mut f = Fixture::new();
        f.run(&[b"SADD", b"a", b"1"]);
        f.run(&[b"SADD", b"d", b"old"]);
        f.run(&[b"SET", b"str", b"v"]);

        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        for bad in [
            &[b"SINTER".as_slice(), b"a", b"str"][..],
            &[b"SUNION".as_slice(), b"str"][..],
            &[b"SDIFF".as_slice(), b"a", b"str"][..],
            &[b"SINTERCARD".as_slice(), b"2", b"a", b"str"][..],
            &[b"SINTERSTORE".as_slice(), b"d", b"a", b"str"][..],
            &[b"SUNIONSTORE".as_slice(), b"d", b"str"][..],
            &[b"SDIFFSTORE".as_slice(), b"d", b"a", b"str"][..],
        ] {
            let reply = f.run(bad);
            assert_eq!(reply, wrong, "for {:?}", bad[0]);
        }
        assert_eq!(
            f.run(&[b"SMEMBERS", b"d"]),
            "*1\r\n$3\r\nold\r\n",
            "and the destination was left alone every time"
        );
    }

    /// The leak a set can spring that nothing on the wire would ever show: the
    /// key goes, the body does not, and `DBSIZE` looks right the whole time.
    #[test]
    fn churning_sets_does_not_grow_the_server() {
        let mut f = Fixture::new();
        let members: Vec<Vec<u8>> = (0..200).map(|i| format!("m{i}").into_bytes()).collect();
        let args: Vec<&[u8]> = std::iter::once(&b"SADD"[..])
            .chain(std::iter::once(&b"s"[..]))
            .chain(members.iter().map(Vec::as_slice))
            .collect();

        f.run(&args);
        f.run(&[b"DEL", b"s"]);
        f.server.compact_step();
        let after_first = f.server.memory_bytes();

        for _ in 0..200 {
            f.run(&args);
            f.run(&[b"DEL", b"s"]);
            f.server.compact_step();
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        assert!(
            f.server.memory_bytes() <= after_first * 2,
            "held {} after two hundred passes against {after_first} after one",
            f.server.memory_bytes()
        );
    }

    // --------------------------------------------------------------- bitmaps

    /// The two single bit commands, and the encoding rule underneath them.
    ///
    /// A write always leaves the value `raw` and a read never re-encodes, which
    /// is why the `int` key here is still `int` after a `GETBIT` and is `raw`
    /// with its first digit changed after a `SETBIT`.
    #[test]
    fn a_bit_is_written_and_read_back_and_a_write_unpacks_an_int() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"SETBIT", b"k", b"7", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"GET", b"k"]), "$1\r\n\u{1}\r\n");
        assert_eq!(f.run(&[b"GETBIT", b"k", b"7"]), ":1\r\n");
        assert_eq!(f.run(&[b"GETBIT", b"k", b"6"]), ":0\r\n");
        assert_eq!(f.run(&[b"GETBIT", b"k", b"100"]), ":0\r\n");
        assert_eq!(f.run(&[b"SETBIT", b"k", b"7", b"0"]), ":1\r\n");

        // Writing a nought past the end still creates the key and still pads.
        assert_eq!(f.run(&[b"SETBIT", b"nk", b"0", b"0"]), ":0\r\n");
        assert_eq!(f.run(&[b"STRLEN", b"nk"]), ":1\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"nk"]), "$3\r\nraw\r\n");

        f.run(&[b"SET", b"num", b"12345"]);
        assert_eq!(f.run(&[b"GETBIT", b"num", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"num"]), "$3\r\nint\r\n");
        assert_eq!(f.run(&[b"SETBIT", b"num", b"1", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"num"]), "$3\r\nraw\r\n");
        assert_eq!(f.run(&[b"GET", b"num"]), "$5\r\nq2345\r\n");
    }

    /// Counting, in bytes and in bits.
    ///
    /// The `0 -5 BIT` row is 25 on a real 8.10.1 and Redis's own documentation
    /// says 22 for it. The server is the thing being copied here.
    #[test]
    fn bits_are_counted_over_a_range_of_bytes_or_of_bits() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"mykey", b"foobar"]);
        assert_eq!(f.run(&[b"BITCOUNT", b"mykey"]), ":26\r\n");
        assert_eq!(f.run(&[b"BITCOUNT", b"mykey", b"0", b"0"]), ":4\r\n");
        assert_eq!(f.run(&[b"BITCOUNT", b"mykey", b"1", b"1"]), ":6\r\n");
        assert_eq!(
            f.run(&[b"BITCOUNT", b"mykey", b"1", b"1", b"BYTE"]),
            ":6\r\n"
        );
        assert_eq!(
            f.run(&[b"BITCOUNT", b"mykey", b"0", b"-5", b"BIT"]),
            ":25\r\n"
        );
        assert_eq!(
            f.run(&[b"BITCOUNT", b"mykey", b"5", b"30", b"BIT"]),
            ":17\r\n"
        );
        assert_eq!(f.run(&[b"BITCOUNT", b"nokey"]), ":0\r\n");

        // A start past the end is left where it is and the end is pulled back,
        // so the range comes out backwards and counts nothing.
        assert_eq!(f.run(&[b"BITCOUNT", b"mykey", b"10", b"20"]), ":0\r\n");

        // A lone start is a syntax error here, where BITPOS allows it.
        assert_eq!(
            f.run(&[b"BITCOUNT", b"mykey", b"0"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"BITCOUNT", b"mykey", b"0", b"1", b"NIB"]),
            "-ERR syntax error\r\n"
        );
    }

    /// Searching, and the one place a miss is not minus one.
    ///
    /// A search for a nought that runs to the end of the string answers the
    /// length in bits, because the string is treated as if it had noughts after
    /// it forever. Give it an explicit end and it answers minus one instead.
    #[test]
    fn a_search_for_a_nought_past_the_end_answers_the_length_in_bits() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"ones", b"\xff\xff\xff"]);
        assert_eq!(f.run(&[b"BITPOS", b"ones", b"0"]), ":24\r\n");
        assert_eq!(f.run(&[b"BITPOS", b"ones", b"0", b"0"]), ":24\r\n");
        assert_eq!(f.run(&[b"BITPOS", b"ones", b"0", b"0", b"-1"]), ":-1\r\n");
        assert_eq!(f.run(&[b"BITPOS", b"ones", b"0", b"0", b"3"]), ":-1\r\n");
        assert_eq!(f.run(&[b"BITPOS", b"ones", b"1"]), ":0\r\n");

        f.run(&[b"SET", b"mid", b"\x00\xff\xf0"]);
        assert_eq!(f.run(&[b"BITPOS", b"mid", b"1", b"0"]), ":8\r\n");
        assert_eq!(f.run(&[b"BITPOS", b"mid", b"1", b"2"]), ":16\r\n");
        assert_eq!(
            f.run(&[b"BITPOS", b"mid", b"1", b"0", b"-1", b"BIT"]),
            ":8\r\n"
        );

        // A missing key is all noughts, so a one is never found and a nought is
        // at position zero.
        assert_eq!(f.run(&[b"BITPOS", b"gone", b"1"]), ":-1\r\n");
        assert_eq!(f.run(&[b"BITPOS", b"gone", b"0"]), ":0\r\n");
    }

    /// The eight operations, with the answers a real server gives for them.
    #[test]
    fn the_eight_combinations_write_what_a_real_server_writes() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"abc"]);
        f.run(&[b"SET", b"b", b"abd"]);
        let cases: &[(&[u8], &str)] = &[
            (b"AND", "ab`"),
            (b"OR", "abg"),
            (b"XOR", "\u{0}\u{0}\u{7}"),
            (b"DIFF", "\u{0}\u{0}\u{3}"),
            (b"DIFF1", "\u{0}\u{0}\u{4}"),
            (b"ANDOR", "ab`"),
            (b"ONE", "\u{0}\u{0}\u{7}"),
        ];
        for (op, want) in cases {
            assert_eq!(f.run(&[b"BITOP", op, b"d", b"a", b"b"]), ":3\r\n", "{op:?}");
            assert_eq!(
                f.run(&[b"GET", b"d"]),
                format!("$3\r\n{want}\r\n"),
                "{op:?}"
            );
        }
        // The one whose answer is not text, so it is compared as bytes.
        assert_eq!(f.run(&[b"BITOP", b"NOT", b"d", b"a"]), ":3\r\n");
        assert_eq!(f.raw(&[b"GET", b"d"]), b"$3\r\n\x9e\x9d\x9c\r\n".to_vec());

        // A missing source is a string of noughts as long as it needs to be, so
        // an AND against one writes three zero bytes rather than nothing.
        assert_eq!(f.run(&[b"BITOP", b"AND", b"d", b"a", b"gone"]), ":3\r\n");
        assert_eq!(f.run(&[b"GET", b"d"]), "$3\r\n\u{0}\u{0}\u{0}\r\n");

        // Every source missing is an empty result, and an empty result takes
        // the destination with it.
        f.run(&[b"SET", b"dest", b"x"]);
        assert_eq!(f.run(&[b"BITOP", b"AND", b"dest", b"g1", b"g2"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"dest"]), ":0\r\n");
    }

    /// What `BITOP` says when it is asked for something it cannot do.
    #[test]
    fn bitop_names_the_operation_in_its_own_complaints() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"a", b"abc"]);
        assert_eq!(
            f.run(&[b"BITOP", b"nope", b"d", b"a"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"BITOP", b"NOT", b"d", b"a", b"a"]),
            "-ERR BITOP NOT must be called with a single source key.\r\n"
        );
        for op in [&b"DIFF"[..], b"DIFF1", b"ANDOR"] {
            assert_eq!(
                f.run(&[b"BITOP", op, b"d", b"a"]),
                format!(
                    "-ERR BITOP {} must be called with at least two source keys.\r\n",
                    String::from_utf8_lossy(op)
                )
            );
        }
        f.run(&[b"LPUSH", b"l", b"x"]);
        assert_eq!(
            f.run(&[b"BITOP", b"AND", b"d", b"a", b"l"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
    }

    /// Packed fields, the three overflow policies and the `#` offset.
    #[test]
    fn bitfield_reads_and_writes_packed_fields() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"BITFIELD", b"bf"]), "*0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"bf"]), ":0\r\n");

        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bf",
                b"INCRBY",
                b"u2",
                b"100",
                b"1",
                b"GET",
                b"u4",
                b"0"
            ]),
            "*2\r\n:1\r\n:0\r\n"
        );
        // The field at bit 100 is two bits wide, so it ends in the thirteenth
        // byte and the value grew to thirteen bytes to hold it.
        assert_eq!(f.run(&[b"STRLEN", b"bf"]), ":13\r\n");

        // A `#` offset counts in fields rather than in bits.
        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bf",
                b"SET",
                b"u8",
                b"#0",
                b"255",
                b"GET",
                b"u8",
                b"#0"
            ]),
            "*2\r\n:0\r\n:255\r\n"
        );

        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bf",
                b"OVERFLOW",
                b"SAT",
                b"INCRBY",
                b"i8",
                b"0",
                b"120",
                b"INCRBY",
                b"i8",
                b"0",
                b"120"
            ]),
            "*2\r\n:119\r\n:127\r\n"
        );
        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bf2",
                b"OVERFLOW",
                b"FAIL",
                b"INCRBY",
                b"u2",
                b"0",
                b"5"
            ]),
            "*1\r\n$-1\r\n"
        );
        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bf3",
                b"OVERFLOW",
                b"WRAP",
                b"INCRBY",
                b"u2",
                b"0",
                b"5"
            ]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"BITFIELD", b"bf3", b"GET", b"i64", b"0"]),
            "*1\r\n:4611686018427387904\r\n"
        );
    }

    /// A bad subcommand anywhere in the line stops all of it.
    ///
    /// Redis checks the whole argument list before it runs any of it, so the
    /// `SET` in front of the bad type here never happens and the key it would
    /// have created is not there afterwards.
    #[test]
    fn a_bad_bitfield_subcommand_leaves_the_key_alone() {
        let mut f = Fixture::new();
        let bad_type = "-ERR Invalid bitfield type. Use something like i16 u8. Note that u64 is not supported but i64 is.\r\n";
        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bad",
                b"SET",
                b"u8",
                b"0",
                b"1",
                b"GET",
                b"u99",
                b"0"
            ]),
            bad_type
        );
        assert_eq!(f.run(&[b"EXISTS", b"bad"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"BITFIELD", b"bad", b"GET", b"u64", b"0"]),
            bad_type
        );
        assert_eq!(
            f.run(&[b"BITFIELD", b"bad", b"GET"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"BITFIELD", b"bad", b"NOPE", b"u8", b"0"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"BITFIELD", b"bad", b"OVERFLOW"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[
                b"BITFIELD",
                b"bad",
                b"OVERFLOW",
                b"NOPE",
                b"GET",
                b"u8",
                b"0"
            ]),
            "-ERR Invalid OVERFLOW type specified\r\n"
        );
        assert_eq!(
            f.run(&[b"BITFIELD", b"bad", b"SET", b"u8", b"0", b"notanum"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        for at in [&b"#-1"[..], b"abc"] {
            assert_eq!(
                f.run(&[b"BITFIELD", b"bad", b"GET", b"u8", at]),
                "-ERR bit offset is not an integer or out of range\r\n"
            );
        }
    }

    /// The read only twin reads, refuses to write, and creates nothing.
    #[test]
    fn bitfield_ro_answers_gets_and_refuses_the_rest() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"n", b"123"]);
        assert_eq!(
            f.run(&[b"BITFIELD_RO", b"n", b"GET", b"u8", b"0"]),
            "*1\r\n:49\r\n"
        );
        // A read does not unpack an int the way a write does.
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"n"]), "$3\r\nint\r\n");

        // An OVERFLOW word is allowed even though nothing here can overflow.
        assert_eq!(
            f.run(&[
                b"BITFIELD_RO",
                b"n",
                b"OVERFLOW",
                b"SAT",
                b"GET",
                b"u8",
                b"0"
            ]),
            "*1\r\n:49\r\n"
        );
        for sub in [&b"SET"[..], b"INCRBY"] {
            assert_eq!(
                f.run(&[b"BITFIELD_RO", b"n", sub, b"u8", b"0", b"1"]),
                "-ERR BITFIELD_RO only supports the GET subcommand\r\n"
            );
        }

        assert_eq!(
            f.run(&[b"BITFIELD_RO", b"gone", b"GET", b"u8", b"100"]),
            "*1\r\n:0\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"gone"]), ":0\r\n");
    }

    /// The offsets a bitmap command will not take.
    #[test]
    fn an_offset_off_the_end_of_the_world_is_refused() {
        let mut f = Fixture::new();
        let bad = "-ERR bit offset is not an integer or out of range\r\n";
        for arg in [&b"abc"[..], b"-1", b"4294967296"] {
            assert_eq!(f.run(&[b"SETBIT", b"k", arg, b"1"]), bad);
            assert_eq!(f.run(&[b"GETBIT", b"k", arg]), bad);
        }
        for arg in [&b"2"[..], b"-1"] {
            assert_eq!(
                f.run(&[b"BITPOS", b"k", arg]),
                "-ERR The bit argument must be 1 or 0.\r\n"
            );
        }
        assert_eq!(
            f.run(&[b"BITPOS", b"k", b"abc"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"BITPOS", b"k", b"0", b"5", b"BIT"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        let bad_bit = "-ERR bit is not an integer or out of range\r\n";
        assert_eq!(f.run(&[b"SETBIT", b"k", b"0", b"2"]), bad_bit);
        assert_eq!(f.run(&[b"SETBIT", b"k", b"0", b"abc"]), bad_bit);
    }

    /// Every one of the seven refuses a key that is not a string.
    #[test]
    fn every_bitmap_command_says_wrongtype() {
        let mut f = Fixture::new();
        f.run(&[b"LPUSH", b"l", b"x"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        let cases: &[&[&[u8]]] = &[
            &[b"SETBIT", b"l", b"0", b"1"],
            &[b"GETBIT", b"l", b"0"],
            &[b"BITCOUNT", b"l"],
            &[b"BITPOS", b"l", b"1"],
            &[b"BITOP", b"AND", b"d", b"l"],
            &[b"BITFIELD", b"l", b"GET", b"u8", b"0"],
            &[b"BITFIELD_RO", b"l", b"GET", b"u8", b"0"],
        ];
        for case in cases {
            assert_eq!(f.run(case), wrong, "{:?}", case[0]);
        }
    }

    // --------------------------------------------------------- hyperloglogs

    #[test]
    fn a_sketch_is_added_to_and_counted() {
        let mut f = Fixture::new();
        // Creating the key counts as a change, even with nothing to add.
        assert_eq!(f.run(&[b"PFADD", b"h"]), ":1\r\n");
        assert_eq!(f.run(&[b"PFADD", b"h"]), ":0\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"h"]), ":0\r\n");
        assert_eq!(f.run(&[b"STRLEN", b"h"]), ":18\r\n");
        // And it is a string, which is not an implementation detail: a client
        // can `GET` a sketch out of one server and `SET` it into another.
        assert_eq!(f.run(&[b"TYPE", b"h"]), "+string\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"h"]), "$3\r\nraw\r\n");

        assert_eq!(f.run(&[b"PFADD", b"h", b"a", b"b", b"c"]), ":1\r\n");
        assert_eq!(f.run(&[b"PFADD", b"h", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"h"]), ":3\r\n");
    }

    #[test]
    fn the_bytes_of_a_sketch_are_the_ones_a_real_server_writes() {
        let mut f = Fixture::new();
        f.run(&[b"PFADD", b"h", b"a", b"b", b"c"]);
        // Not text, so it is compared as bytes.
        let want = b"HYLL\x01\0\0\0\0\0\0\0\0\0\0\x80\x60\xf3\x80\x50\xb1\x84\x4b\xfb\x80\x42\x5a";
        let mut reply = b"$27\r\n".to_vec();
        reply.extend_from_slice(want);
        reply.extend_from_slice(b"\r\n");
        assert_eq!(f.raw(&[b"GET", b"h"]), reply);
    }

    #[test]
    fn counting_several_keys_counts_their_union() {
        let mut f = Fixture::new();
        f.run(&[b"PFADD", b"a", b"x", b"y"]);
        f.run(&[b"PFADD", b"b", b"y", b"z"]);
        assert_eq!(f.run(&[b"PFCOUNT", b"a"]), ":2\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"a", b"b"]), ":3\r\n");
        // A key that is not there is an empty sketch, not an error and not
        // something that gets created by being counted.
        assert_eq!(f.run(&[b"PFCOUNT", b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"a", b"gone"]), ":2\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"gone"]), ":0\r\n");
    }

    #[test]
    fn a_merge_keeps_what_the_destination_had() {
        let mut f = Fixture::new();
        f.run(&[b"PFADD", b"a", b"x", b"y"]);
        f.run(&[b"PFADD", b"b", b"z"]);
        assert_eq!(f.run(&[b"PFMERGE", b"d", b"a", b"b"]), "+OK\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"d"]), ":3\r\n");
        // The destination is one of the sources, so a second merge adds to it.
        f.run(&[b"PFADD", b"c", b"w"]);
        assert_eq!(f.run(&[b"PFMERGE", b"d", b"c"]), "+OK\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"d"]), ":4\r\n");
        // And with no sources it is a no-op that still answers OK and still
        // creates a destination that was not there.
        assert_eq!(f.run(&[b"PFMERGE", b"fresh"]), "+OK\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"fresh"]), ":0\r\n");
    }

    #[test]
    fn the_debug_forms_answer_four_different_shapes() {
        let mut f = Fixture::new();
        f.run(&[b"PFADD", b"h", b"a", b"b", b"c"]);
        assert_eq!(f.run(&[b"PFDEBUG", b"ENCODING", b"h"]), "+sparse\r\n");
        assert_eq!(
            f.run(&[b"PFDEBUG", b"DECODE", b"h"]),
            "$44\r\nZ:8436 v:1,1 Z:4274 v:2,1 Z:3068 v:1,1 Z:603\r\n"
        );
        assert_eq!(f.run(&[b"PFDEBUG", b"TODENSE", b"h"]), ":1\r\n");
        assert_eq!(f.run(&[b"PFDEBUG", b"TODENSE", b"h"]), ":0\r\n");
        assert_eq!(f.run(&[b"PFDEBUG", b"ENCODING", b"h"]), "+dense\r\n");
        assert_eq!(f.run(&[b"STRLEN", b"h"]), ":12304\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", b"h"]), ":3\r\n");
        // A dense sketch has no opcodes left to print.
        assert_eq!(
            f.run(&[b"PFDEBUG", b"DECODE", b"h"]),
            "-ERR HLL encoding is not sparse\r\n"
        );

        // All 16384 registers, of which three are not nought.
        let reply = f.run(&[b"PFDEBUG", b"GETREG", b"h"]);
        assert!(reply.starts_with("*16384\r\n"), "{}", &reply[..16]);
        assert_eq!(reply.matches(":0\r\n").count(), 16381);
        assert_eq!(reply.matches(":1\r\n").count(), 2);
        assert_eq!(reply.matches(":2\r\n").count(), 1);

        assert_eq!(f.run(&[b"PFSELFTEST"]), "+OK\r\n");
    }

    #[test]
    fn a_string_that_is_not_a_sketch_is_refused_with_its_own_sentence() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"plain", b"not a sketch"]);
        let not_hll = "-WRONGTYPE Key is not a valid HyperLogLog string value.\r\n";
        assert_eq!(f.run(&[b"PFADD", b"plain", b"a"]), not_hll);
        assert_eq!(f.run(&[b"PFCOUNT", b"plain"]), not_hll);
        assert_eq!(f.run(&[b"PFMERGE", b"plain"]), not_hll);
        assert_eq!(f.run(&[b"PFDEBUG", b"ENCODING", b"plain"]), not_hll);

        // A key that is not a string at all gets the ordinary sentence, and a
        // destination that would have been written is not created.
        f.run(&[b"RPUSH", b"l", b"x"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        assert_eq!(f.run(&[b"PFADD", b"l", b"a"]), wrong);
        assert_eq!(f.run(&[b"PFCOUNT", b"l"]), wrong);
        assert_eq!(f.run(&[b"PFMERGE", b"dest", b"l"]), wrong);
        assert_eq!(f.run(&[b"EXISTS", b"dest"]), ":0\r\n");
        assert_eq!(f.run(&[b"PFDEBUG", b"GETREG", b"l"]), wrong);
    }

    #[test]
    fn pfdebug_has_its_own_complaints() {
        let mut f = Fixture::new();
        f.run(&[b"PFADD", b"h", b"a"]);
        // The word is quoted exactly as the client spelled it, and this is not
        // the "Try X HELP." sentence every other container command uses.
        assert_eq!(
            f.run(&[b"PFDEBUG", b"NOPE", b"h"]),
            "-ERR Unknown PFDEBUG subcommand 'NOPE'\r\n"
        );
        // Where all three of the real commands take a missing key as empty.
        let gone = "-ERR The specified key does not exist\r\n";
        assert_eq!(f.run(&[b"PFDEBUG", b"GETREG", b"missing"]), gone);
        assert_eq!(f.run(&[b"PFDEBUG", b"DECODE", b"missing"]), gone);
        assert_eq!(f.run(&[b"PFDEBUG", b"ENCODING", b"missing"]), gone);
        assert_eq!(f.run(&[b"PFDEBUG", b"TODENSE", b"missing"]), gone);
        assert_eq!(
            f.run(&[b"PFDEBUG"]),
            "-ERR wrong number of arguments for 'pfdebug' command\r\n"
        );
        assert_eq!(
            f.run(&[b"PFSELFTEST", b"x"]),
            "-ERR wrong number of arguments for 'pfselftest' command\r\n"
        );
    }

    #[test]
    fn a_sketch_whose_opcodes_do_not_add_up_says_so() {
        let mut f = Fixture::new();
        f.run(&[b"PFADD", b"h", b"a", b"b", b"c"]);
        // The sketch with its last byte cut off, which is still a header and a
        // magic and is a run length encoding that stops short of register 16384.
        let reply = f.raw(&[b"GET", b"h"]);
        let short = reply[5..reply.len() - 3].to_vec();
        f.run(&[b"SET", b"h", &short]);
        assert_eq!(
            f.run(&[b"PFCOUNT", b"h"]),
            "-INVALIDOBJ Corrupted HLL object detected\r\n"
        );
    }

    #[test]
    fn a_sketch_survives_a_dump_and_a_restore_in_both_encodings() {
        let mut f = Fixture::new();
        // One that stays sparse and one that has gone dense, since the payload
        // carries the bytes and the two encodings are different lengths.
        f.run(&[b"PFADD", b"small", b"a", b"b", b"c"]);
        for i in 0..10_000u32 {
            let ele = format!("e{i}");
            f.run(&[b"PFADD", b"big", ele.as_bytes()]);
        }
        assert_eq!(f.run(&[b"PFDEBUG", b"ENCODING", b"small"]), "+sparse\r\n");
        assert_eq!(f.run(&[b"PFDEBUG", b"ENCODING", b"big"]), "+dense\r\n");

        for key in [&b"small"[..], b"big"] {
            let mut copy = key.to_vec();
            copy.push(b'2');
            let bytes = payload(&f.raw(&[b"DUMP", key]));
            assert_eq!(f.run(&[b"RESTORE", &copy, b"0", &bytes]), "+OK\r\n");
            // The bytes, the encoding and the estimate all come back, which is
            // the whole of what byte compatibility across a round trip means.
            assert_eq!(f.raw(&[b"GET", &copy]), f.raw(&[b"GET", key]));
            assert_eq!(
                f.run(&[b"PFDEBUG", b"ENCODING", &copy]),
                f.run(&[b"PFDEBUG", b"ENCODING", key])
            );
            assert_eq!(f.run(&[b"PFCOUNT", &copy]), f.run(&[b"PFCOUNT", key]));
        }
        assert_eq!(f.run(&[b"PFCOUNT", b"small2"]), ":3\r\n");
        assert_eq!(f.run(&[b"STRLEN", b"big2"]), ":12304\r\n");
    }

    /// One RESP2 bulk string. The JSON replies are almost all one of these and
    /// the text inside them has quotes in it, so writing the frame out by hand
    /// buries the part of the assertion that matters.
    fn bulk(s: &str) -> String {
        format!("${}\r\n{s}\r\n", s.len())
    }

    /// A RESP2 array of bulk strings, which is what most of the list replies
    /// are and what writing them out by hand in every assertion looks like.
    fn bulks(parts: &[&str]) -> String {
        let mut s = format!("*{}\r\n", parts.len());
        for p in parts {
            s.push_str(&format!("${}\r\n{p}\r\n", p.len()));
        }
        s
    }

    #[test]
    fn a_list_is_pushed_from_both_ends_and_the_left_one_reverses() {
        let mut f = Fixture::new();
        // Each element in turn goes at the head, so the last one sent is at the
        // front when it is over. That reads like a bug in the client and it is
        // what every Redis has always done.
        assert_eq!(f.run(&[b"LPUSH", b"k", b"a", b"b", b"c"]), ":3\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"0", b"-1"]),
            bulks(&["c", "b", "a"])
        );
        assert_eq!(f.run(&[b"RPUSH", b"k", b"d"]), ":4\r\n");
        assert_eq!(f.run(&[b"LLEN", b"k"]), ":4\r\n");
        assert_eq!(f.run(&[b"LPOP", b"k"]), "$1\r\nc\r\n");
        assert_eq!(f.run(&[b"RPOP", b"k"]), "$1\r\nd\r\n");
        assert_eq!(f.run(&[b"LRANGE", b"k", b"0", b"-1"]), bulks(&["b", "a"]));
        assert_eq!(f.run(&[b"TYPE", b"k"]), "+list\r\n");
    }

    #[test]
    fn the_x_pushes_refuse_to_bring_a_list_back_to_life() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"LPUSHX", b"k", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"RPUSHX", b"k", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n");
        f.run(&[b"RPUSH", b"k", b"a"]);
        assert_eq!(f.run(&[b"LPUSHX", b"k", b"z"]), ":2\r\n");
        assert_eq!(f.run(&[b"RPUSHX", b"k", b"y"]), ":3\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"0", b"-1"]),
            bulks(&["z", "a", "y"])
        );
    }

    /// The four ways a pop can come back with nothing, which are three
    /// different replies and a RESP2 client can tell all of them apart.
    #[test]
    fn an_empty_pop_is_a_different_nothing_with_a_count_and_without() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"LPOP", b"nope"]), "$-1\r\n");
        assert_eq!(f.run(&[b"LPOP", b"nope", b"2"]), "*-1\r\n");
        assert_eq!(f.run(&[b"RPOP", b"nope"]), "$-1\r\n");
        assert_eq!(f.run(&[b"RPOP", b"nope", b"2"]), "*-1\r\n");
        f.run(&[b"RPUSH", b"k", b"a", b"b", b"c"]);
        // A count of zero against a list that is there is an empty array and
        // not a null array, which is the fourth answer.
        assert_eq!(f.run(&[b"LPOP", b"k", b"0"]), "*0\r\n");
        assert_eq!(f.run(&[b"LPOP", b"k", b"1"]), bulks(&["a"]));
        // More than there is takes what there is and the key goes with it.
        assert_eq!(f.run(&[b"RPOP", b"k", b"9"]), bulks(&["c", "b"]));
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n");
    }

    #[test]
    fn a_pop_count_has_its_own_sentence_and_a_third_argument_is_an_arity_error() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a"]);
        let range = "-ERR value is out of range, must be positive\r\n";
        assert_eq!(f.run(&[b"LPOP", b"k", b"-1"]), range);
        assert_eq!(f.run(&[b"LPOP", b"k", b"abc"]), range);
        assert_eq!(f.run(&[b"RPOP", b"k", b"-1"]), range);
        // Redis calls this an arity error and not a syntax error, which is a
        // distinction it does not always make.
        assert_eq!(
            f.run(&[b"LPOP", b"k", b"1", b"2"]),
            "-ERR wrong number of arguments for 'lpop' command\r\n"
        );
        assert_eq!(f.run(&[b"LLEN", b"k"]), ":1\r\n");
    }

    #[test]
    fn a_range_takes_negative_ends_and_clamps_the_ones_that_run_off() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a", b"b", b"c"]);
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"0", b"-1"]),
            bulks(&["a", "b", "c"])
        );
        assert_eq!(f.run(&[b"LRANGE", b"k", b"-2", b"-1"]), bulks(&["b", "c"]));
        assert_eq!(f.run(&[b"LRANGE", b"k", b"1", b"1"]), bulks(&["b"]));
        assert_eq!(f.run(&[b"LRANGE", b"k", b"5", b"10"]), "*0\r\n");
        assert_eq!(f.run(&[b"LRANGE", b"k", b"2", b"1"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"-100", b"100"]),
            bulks(&["a", "b", "c"])
        );
        // A key that is not there is an empty range and not a nil, which is the
        // one place a list disagrees with a set.
        assert_eq!(f.run(&[b"LRANGE", b"nope", b"0", b"-1"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"a", b"b"]),
            "-ERR value is not an integer or out of range\r\n"
        );
    }

    #[test]
    fn an_index_reads_and_writes_from_whichever_end_is_nearer() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a", b"b", b"c"]);
        assert_eq!(f.run(&[b"LINDEX", b"k", b"0"]), "$1\r\na\r\n");
        assert_eq!(f.run(&[b"LINDEX", b"k", b"-1"]), "$1\r\nc\r\n");
        assert_eq!(f.run(&[b"LINDEX", b"k", b"99"]), "$-1\r\n");
        assert_eq!(f.run(&[b"LINDEX", b"nope", b"0"]), "$-1\r\n");
        assert_eq!(f.run(&[b"LSET", b"k", b"-1", b"z"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"0", b"-1"]),
            bulks(&["a", "b", "z"])
        );
        // Both ways of missing are errors here rather than a nil, because a
        // list is never empty and there is nothing else the reply could be.
        assert_eq!(
            f.run(&[b"LSET", b"k", b"99", b"z"]),
            "-ERR index out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"LSET", b"nope", b"0", b"z"]),
            "-ERR no such key\r\n"
        );
    }

    #[test]
    fn linsert_says_three_things_with_one_signed_number() {
        let mut f = Fixture::new();
        // Zero for a key that is not there, which is not the same as minus one
        // for a pivot that is not in a list that is.
        assert_eq!(
            f.run(&[b"LINSERT", b"nope", b"BEFORE", b"a", b"x"]),
            ":0\r\n"
        );
        f.run(&[b"RPUSH", b"k", b"a", b"b"]);
        assert_eq!(f.run(&[b"LINSERT", b"k", b"before", b"a", b"X"]), ":3\r\n");
        assert_eq!(f.run(&[b"LINSERT", b"k", b"AFTER", b"b", b"Y"]), ":4\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"0", b"-1"]),
            bulks(&["X", "a", "b", "Y"])
        );
        assert_eq!(
            f.run(&[b"LINSERT", b"k", b"BEFORE", b"zz", b"x"]),
            ":-1\r\n"
        );
        assert_eq!(
            f.run(&[b"LINSERT", b"k", b"SIDEWAYS", b"a", b"x"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn lrem_counts_in_three_directions_and_takes_the_key_when_it_empties() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a", b"b", b"a", b"c", b"a"]);
        assert_eq!(f.run(&[b"LREM", b"k", b"2", b"a"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"k", b"0", b"-1"]),
            bulks(&["b", "c", "a"])
        );
        assert_eq!(f.run(&[b"LREM", b"k", b"-1", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"LRANGE", b"k", b"0", b"-1"]), bulks(&["b", "c"]));
        assert_eq!(f.run(&[b"LREM", b"k", b"0", b"b"]), ":1\r\n");
        assert_eq!(f.run(&[b"LREM", b"k", b"0", b"c"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n");
        assert_eq!(f.run(&[b"LREM", b"nope", b"0", b"a"]), ":0\r\n");
    }

    #[test]
    fn ltrim_keeps_a_window_and_an_empty_one_deletes_the_key() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a", b"b", b"c", b"d"]);
        assert_eq!(f.run(&[b"LTRIM", b"k", b"1", b"-2"]), "+OK\r\n");
        assert_eq!(f.run(&[b"LRANGE", b"k", b"0", b"-1"]), bulks(&["b", "c"]));
        // `LTRIM k 1 0` is the documented way to empty a list, so it has to
        // leave `EXISTS` answering zero rather than leaving an empty one.
        assert_eq!(f.run(&[b"LTRIM", b"k", b"1", b"0"]), "+OK\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"k"]), ":0\r\n");
        assert_eq!(f.run(&[b"LTRIM", b"nope", b"0", b"-1"]), "+OK\r\n");
    }

    #[test]
    fn lpos_walks_from_either_end_and_stops_where_it_is_told() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"p", b"a", b"b", b"c", b"a", b"b", b"c", b"a"]);
        assert_eq!(f.run(&[b"LPOS", b"p", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"LPOS", b"p", b"a", b"RANK", b"-1"]), ":6\r\n");
        assert_eq!(f.run(&[b"LPOS", b"p", b"a", b"RANK", b"2"]), ":3\r\n");
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"COUNT", b"2"]),
            "*2\r\n:0\r\n:3\r\n"
        );
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"RANK", b"-1", b"COUNT", b"0"]),
            "*3\r\n:6\r\n:3\r\n:0\r\n"
        );
        // MAXLEN counts elements looked at and not matches found, so three
        // stops after `a b c` and finds the one match in it.
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"COUNT", b"0", b"MAXLEN", b"3"]),
            "*1\r\n:0\r\n"
        );
        // Nothing found is three different replies depending on how it was
        // asked and whether the key is there at all.
        assert_eq!(f.run(&[b"LPOS", b"p", b"zz"]), "$-1\r\n");
        assert_eq!(f.run(&[b"LPOS", b"p", b"zz", b"COUNT", b"0"]), "*0\r\n");
        assert_eq!(f.run(&[b"LPOS", b"nope", b"a"]), "$-1\r\n");
        assert_eq!(f.run(&[b"LPOS", b"nope", b"a", b"COUNT", b"2"]), "*0\r\n");
    }

    #[test]
    fn lpos_words_its_three_mistakes_the_way_redis_does() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"p", b"a"]);
        // The whole sentence and not a prefix, because the older wording of it
        // is still all over the internet and clients match on the text.
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"RANK", b"0"]),
            "-ERR RANK can't be zero: use 1 to start from the first match, 2 from the second ... or use negative to start from the end of the list\r\n"
        );
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"COUNT", b"-1"]),
            "-ERR COUNT can't be negative\r\n"
        );
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"MAXLEN", b"-1"]),
            "-ERR MAXLEN can't be negative\r\n"
        );
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"RANK"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"LPOS", b"p", b"a", b"FOO", b"1"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn a_move_takes_from_one_end_and_gives_to_another_even_on_one_key() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a", b"b", b"c"]);
        assert_eq!(f.run(&[b"RPOPLPUSH", b"k", b"d"]), "$1\r\nc\r\n");
        assert_eq!(f.run(&[b"LRANGE", b"k", b"0", b"-1"]), bulks(&["a", "b"]));
        assert_eq!(f.run(&[b"LRANGE", b"d", b"0", b"-1"]), bulks(&["c"]));
        assert_eq!(
            f.run(&[b"LMOVE", b"k", b"d", b"LEFT", b"RIGHT"]),
            "$1\r\na\r\n"
        );
        assert_eq!(f.run(&[b"LRANGE", b"d", b"0", b"-1"]), bulks(&["c", "a"]));
        // The same key twice is the documented way to rotate a list and falls
        // out of taking the element before deciding where to put it.
        f.run(&[b"DEL", b"r"]);
        f.run(&[b"RPUSH", b"r", b"1", b"2", b"3"]);
        assert_eq!(f.run(&[b"RPOPLPUSH", b"r", b"r"]), "$1\r\n3\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"r", b"0", b"-1"]),
            bulks(&["3", "1", "2"])
        );
        assert_eq!(
            f.run(&[b"LMOVE", b"nope", b"d", b"LEFT", b"LEFT"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"LMOVE", b"r", b"d", b"LEFT", b"SIDEWAYS"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn a_move_checks_the_destination_before_it_takes_anything() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a", b"b"]);
        f.run(&[b"SET", b"str", b"v"]);
        assert_eq!(
            f.run(&[b"LMOVE", b"k", b"str", b"LEFT", b"LEFT"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        // The element is still where it was, rather than having gone nowhere.
        assert_eq!(f.run(&[b"LRANGE", b"k", b"0", b"-1"]), bulks(&["a", "b"]));
    }

    #[test]
    fn a_block_move_orders_the_block_by_the_ends_and_the_ordering_word() {
        // OBO is what you get from sending LMOVE that many times, BULK keeps
        // the source order. The two only differ when both ends are the same,
        // which is the whole reason the word exists.
        for (from, to, order, want) in [
            ("LEFT", "RIGHT", "OBO", ["a", "b"]),
            ("LEFT", "RIGHT", "BULK", ["a", "b"]),
            ("LEFT", "LEFT", "OBO", ["b", "a"]),
            ("LEFT", "LEFT", "BULK", ["a", "b"]),
            ("RIGHT", "LEFT", "OBO", ["d", "e"]),
            ("RIGHT", "LEFT", "BULK", ["d", "e"]),
            ("RIGHT", "RIGHT", "OBO", ["e", "d"]),
            ("RIGHT", "RIGHT", "BULK", ["d", "e"]),
        ] {
            let mut f = Fixture::new();
            f.run(&[b"RPUSH", b"s", b"a", b"b", b"c", b"d", b"e"]);
            let how = format!("{from} {to} {order}");
            let reply = f.run(&[
                b"LMOVEM",
                b"s",
                b"d",
                from.as_bytes(),
                to.as_bytes(),
                b"COUNT",
                b"2",
                order.as_bytes(),
            ]);
            assert_eq!(reply, bulks(&want), "the reply for {how}");
            assert_eq!(
                f.run(&[b"LRANGE", b"d", b"0", b"-1"]),
                bulks(&want),
                "the destination for {how}"
            );
        }
    }

    #[test]
    fn a_block_move_of_one_needs_no_count_at_all() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"s", b"a", b"b", b"c"]);
        assert_eq!(
            f.run(&[b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT"]),
            bulks(&["a"])
        );
        assert_eq!(f.run(&[b"LRANGE", b"s", b"0", b"-1"]), bulks(&["b", "c"]));
        // Six and seven arguments are neither of the two forms, so the
        // reference calls both of them a syntax error rather than guessing.
        assert_eq!(
            f.run(&[b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT", b"2"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn a_block_move_with_exactly_takes_all_of_them_or_none() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"s", b"a", b"b", b"c"]);
        // A null array and not a null bulk string, which `redis-cli` prints as
        // `(nil)` either way and only the raw wire tells apart. What it would
        // have sent is an array, so its nothing is an array's nothing.
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"EXACTLY", b"99", b"BULK"
            ]),
            "*-1\r\n"
        );
        assert_eq!(
            f.run(&[b"LRANGE", b"s", b"0", b"-1"]),
            bulks(&["a", "b", "c"])
        );
        // COUNT takes what there is, and an emptied source goes away.
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT", b"99", b"BULK"
            ]),
            bulks(&["a", "b", "c"])
        );
        assert_eq!(f.run(&[b"EXISTS", b"s"]), ":0\r\n");
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT", b"1", b"BULK"
            ]),
            "*-1\r\n"
        );
    }

    #[test]
    fn a_block_move_onto_itself_rotates_by_the_count() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"s", b"a", b"b", b"c"]);
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"s", b"LEFT", b"RIGHT", b"COUNT", b"2", b"BULK"
            ]),
            bulks(&["a", "b"])
        );
        assert_eq!(
            f.run(&[b"LRANGE", b"s", b"0", b"-1"]),
            bulks(&["c", "a", "b"])
        );
    }

    #[test]
    fn a_block_move_reads_the_count_before_the_ordering_word() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"s", b"a", b"b"]);
        f.run(&[b"SET", b"str", b"v"]);
        let count = "-ERR count should be greater than 0\r\n";
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT", b"abc", b"NOPE"
            ]),
            count
        );
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT", b"0", b"BULK"
            ]),
            count
        );
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"COUNT", b"1", b"NOPE"
            ]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"d", b"LEFT", b"RIGHT", b"NOPE", b"abc", b"BULK"
            ]),
            "-ERR syntax error\r\n"
        );
        // Every argument is read before the keys are looked at, so a bad count
        // beats a wrong type even when the type is wrong on the source.
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"str", b"d", b"LEFT", b"RIGHT", b"COUNT", b"abc", b"BULK"
            ]),
            count
        );
        assert_eq!(
            f.run(&[
                b"LMOVEM", b"s", b"str", b"LEFT", b"RIGHT", b"COUNT", b"1", b"BULK"
            ]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(f.run(&[b"LRANGE", b"s", b"0", b"-1"]), bulks(&["a", "b"]));
    }

    #[test]
    fn lmpop_answers_from_the_first_key_that_has_anything() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"b", b"1", b"2", b"3"]);
        // The name of the key that answered comes back with the elements,
        // because the client cannot work out which one it was.
        assert_eq!(
            f.run(&[b"LMPOP", b"2", b"a", b"b", b"LEFT", b"COUNT", b"2"]),
            "*2\r\n$1\r\nb\r\n*2\r\n$1\r\n1\r\n$1\r\n2\r\n"
        );
        assert_eq!(
            f.run(&[b"LMPOP", b"2", b"a", b"b", b"RIGHT"]),
            "*2\r\n$1\r\nb\r\n*1\r\n$1\r\n3\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"b"]), ":0\r\n");
        // A null array and not a null, even though what it stands in for is an
        // array holding a key name and then another array.
        assert_eq!(f.run(&[b"LMPOP", b"2", b"a", b"b", b"LEFT"]), "*-1\r\n");
    }

    #[test]
    fn lmpop_has_its_own_words_for_a_count_and_for_a_key_count() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a"]);
        assert_eq!(
            f.run(&[b"LMPOP", b"0", b"k", b"LEFT"]),
            "-ERR numkeys should be greater than 0\r\n"
        );
        assert_eq!(
            f.run(&[b"LMPOP", b"-1", b"k", b"LEFT"]),
            "-ERR numkeys should be greater than 0\r\n"
        );
        assert_eq!(
            f.run(&[b"LMPOP", b"1", b"k", b"LEFT", b"COUNT", b"0"]),
            "-ERR count should be greater than 0\r\n"
        );
        // A key count that eats the direction is a syntax error and not a
        // sentence about key counts, because the direction is simply not there.
        assert_eq!(
            f.run(&[b"LMPOP", b"3", b"k", b"LEFT"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"LMPOP", b"1", b"k", b"LEFT", b"COUNT", b"1", b"x"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"LMPOP", b"1", b"k", b"LEFT", b"FOO", b"1"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"LMPOP", b"1", b"k", b"SIDEWAYS"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(f.run(&[b"LLEN", b"k"]), ":1\r\n");
    }

    #[test]
    fn every_list_command_says_wrongtype_and_writes_nothing() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"str", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        for cmd in [
            &[b"LPUSH".as_slice(), b"str", b"a"][..],
            &[b"RPUSH", b"str", b"a"],
            &[b"LPUSHX", b"str", b"a"],
            &[b"RPUSHX", b"str", b"a"],
            &[b"LPOP", b"str"],
            &[b"LPOP", b"str", b"2"],
            &[b"RPOP", b"str"],
            &[b"LLEN", b"str"],
            &[b"LRANGE", b"str", b"0", b"-1"],
            &[b"LINDEX", b"str", b"0"],
            &[b"LSET", b"str", b"0", b"a"],
            &[b"LINSERT", b"str", b"BEFORE", b"a", b"b"],
            &[b"LREM", b"str", b"0", b"a"],
            &[b"LTRIM", b"str", b"0", b"-1"],
            &[b"LPOS", b"str", b"a"],
            &[b"LPOS", b"str", b"a", b"COUNT", b"0"],
            &[b"RPOPLPUSH", b"str", b"d"],
            &[b"LMOVE", b"str", b"d", b"LEFT", b"LEFT"],
            &[b"LMPOP", b"1", b"str", b"LEFT"],
        ] {
            assert_eq!(f.run(cmd), wrong, "{:?}", String::from_utf8_lossy(cmd[0]));
        }
        assert_eq!(f.run(&[b"GET", b"str"]), "$1\r\nv\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"d"]), ":0\r\n");
    }

    /// A timeout is not an integer and it is not an ordinary float either: the
    /// three sentences it can answer with are its own, and which one a given
    /// argument gets is not what reading the code would suggest.
    #[test]
    fn a_timeout_has_three_ways_of_being_wrong() {
        let mut f = Fixture::new();
        let not_float = "-ERR timeout is not a float or out of range\r\n";
        let range = "-ERR timeout is out of range\r\n";
        for (bad, want) in [
            (&[b"BLPOP".as_slice(), b"k", b"abc"][..], not_float),
            (&[b"BLPOP", b"k", b"nan"], not_float),
            (&[b"BLPOP", b"k", b""], not_float),
            // Whitespace on either side, which `strtold` would take and Redis
            // does not.
            (&[b"BLPOP", b"k", b" 1"], not_float),
            (&[b"BLPOP", b"k", b"1 "], not_float),
            (&[b"BLPOP", b"k", b"-1"], "-ERR timeout is negative\r\n"),
            (&[b"BLPOP", b"k", b"-0.1"], "-ERR timeout is negative\r\n"),
            // These three parse, so they are not the not-a-float error, and all
            // three are further off than an i64 of milliseconds reaches.
            (&[b"BLPOP", b"k", b"1e400"], range),
            (&[b"BLPOP", b"k", b"inf"], range),
            (&[b"BLPOP", b"k", b"9999999999999999"], range),
            (&[b"BRPOP", b"k", b"abc"], not_float),
            (
                &[b"BLMOVE", b"a", b"b", b"LEFT", b"RIGHT", b"abc"],
                not_float,
            ),
            (
                &[b"BRPOPLPUSH", b"a", b"b", b"-1"],
                "-ERR timeout is negative\r\n",
            ),
            (&[b"BLMPOP", b"abc", b"1", b"k", b"LEFT"], not_float),
        ] {
            assert_eq!(f.run(bad), want, "for {bad:?}");
        }
    }

    /// A timeout of exactly zero means no timeout, and there are two ways of
    /// writing exactly zero.
    #[test]
    fn a_zero_timeout_waits_and_the_smallest_positive_one_does_not() {
        let mut f = Fixture::new();
        for timeout in [b"0".as_slice(), b"0.0", b"-0.0"] {
            let (flow, out) = f.flow(&[b"BLPOP", b"k", timeout]);
            assert_eq!(flow, Flow::Block, "for {timeout:?}");
            assert!(out.is_empty(), "for {timeout:?}");
        }
        // Positive, so it is a real deadline, and the deadline is this
        // millisecond. Nothing is written here either: the reply comes from the
        // sweep, which is the engine's and not this layer's.
        let (flow, out) = f.flow(&[b"BLPOP", b"k", b"0.0000001"]);
        assert_eq!(flow, Flow::Block);
        assert!(out.is_empty());
    }

    #[test]
    fn a_blocking_command_that_can_be_answered_answers_like_the_one_it_wraps() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"L", b"a", b"b", b"c", b"d", b"e"]);

        // The one difference from LPOP: the reply names the key that answered,
        // which is what makes BLPOP over several keys usable.
        assert_eq!(
            f.flow(&[b"BLPOP", b"nope", b"L", b"0"]),
            (Flow::Continue, "*2\r\n$1\r\nL\r\n$1\r\na\r\n".to_owned())
        );
        assert_eq!(
            f.run(&[b"BRPOP", b"L", b"0"]),
            "*2\r\n$1\r\nL\r\n$1\r\ne\r\n"
        );
        assert_eq!(
            f.run(&[
                b"BLMPOP", b"0", b"2", b"nope", b"L", b"LEFT", b"COUNT", b"2"
            ]),
            "*2\r\n$1\r\nL\r\n*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"BLMOVE", b"L", b"D", b"LEFT", b"RIGHT", b"0"]),
            "$1\r\nd\r\n"
        );
        assert_eq!(
            f.run(&[b"EXISTS", b"L"]),
            ":0\r\n",
            "and the key went with it"
        );
        assert_eq!(f.run(&[b"LRANGE", b"D", b"0", b"-1"]), "*1\r\n$1\r\nd\r\n");
        // Onto itself, which is how a list is rotated and is a real thing to ask
        // a blocking move for.
        f.run(&[b"RPUSH", b"D", b"x"]);
        assert_eq!(f.run(&[b"BRPOPLPUSH", b"D", b"D", b"0"]), "$1\r\nx\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", b"D", b"0", b"-1"]),
            "*2\r\n$1\r\nx\r\n$1\r\nd\r\n"
        );
    }

    #[test]
    fn blmpop_reads_its_count_and_its_key_count_the_way_lmpop_does() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"k", b"a"]);
        for (bad, want) in [
            (
                &[b"BLMPOP".as_slice(), b"0", b"0", b"k", b"LEFT"][..],
                "-ERR numkeys should be greater than 0\r\n",
            ),
            (
                &[b"BLMPOP", b"0", b"-1", b"k", b"LEFT"],
                "-ERR numkeys should be greater than 0\r\n",
            ),
            // Two keys named and one given, so the word that should have been
            // the direction is a key and there is no direction left.
            (
                &[b"BLMPOP", b"0", b"2", b"k", b"LEFT"],
                "-ERR syntax error\r\n",
            ),
            (
                &[b"BLMPOP", b"0", b"1", b"k", b"SIDEWAYS"],
                "-ERR syntax error\r\n",
            ),
            (
                &[b"BLMPOP", b"0", b"1", b"k", b"LEFT", b"COUNT"],
                "-ERR syntax error\r\n",
            ),
            (
                &[b"BLMPOP", b"0", b"1", b"k", b"LEFT", b"COUNT", b"2", b"x"],
                "-ERR syntax error\r\n",
            ),
            // A count that is not a number at all gets the same sentence a zero
            // or a negative one gets, rather than the usual one about integers.
            (
                &[b"BLMPOP", b"0", b"1", b"k", b"LEFT", b"COUNT", b"0"],
                "-ERR count should be greater than 0\r\n",
            ),
            (
                &[b"BLMPOP", b"0", b"1", b"k", b"LEFT", b"COUNT", b"abc"],
                "-ERR count should be greater than 0\r\n",
            ),
        ] {
            assert_eq!(f.run(bad), want, "for {bad:?}");
        }
        assert_eq!(f.run(&[b"LLEN", b"k"]), ":1\r\n", "and none of them popped");
    }

    #[test]
    fn a_blocking_move_reads_its_directions_before_its_timeout() {
        let mut f = Fixture::new();
        // Both are wrong. Redis checks the directions first, so this is the
        // syntax error and not a complaint about the timeout.
        assert_eq!(
            f.run(&[b"BLMOVE", b"a", b"b", b"UP", b"DOWN", b"abc"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"BLMOVE", b"a", b"b", b"LEFT", b"DOWN", b"0.05"]),
            "-ERR syntax error\r\n"
        );
    }

    /// `BLMOVEM` answers exactly what `LMOVEM` answers when it does not have to
    /// wait, which is the same relationship every other command in this file has
    /// with the one it wraps.
    #[test]
    fn a_blocking_block_move_that_can_be_answered_answers_like_lmovem() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"L", b"a", b"b", b"c", b"d", b"e"]);
        assert_eq!(
            f.flow(&[b"BLMOVEM", b"L", b"D", b"LEFT", b"RIGHT", b"0"]),
            (Flow::Continue, "*1\r\n$1\r\na\r\n".to_owned())
        );
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"L", b"D", b"RIGHT", b"RIGHT", b"0", b"COUNT", b"2", b"OBO"
            ]),
            bulks(&["e", "d"])
        );
        assert_eq!(
            f.run(&[b"LRANGE", b"D", b"0", b"-1"]),
            bulks(&["a", "e", "d"])
        );
        // `EXACTLY` with enough there does not wait either.
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"L", b"D", b"LEFT", b"RIGHT", b"0", b"EXACTLY", b"2", b"BULK"
            ]),
            bulks(&["b", "c"])
        );
        assert_eq!(f.run(&[b"EXISTS", b"L"]), ":0\r\n", "and the key went");
    }

    /// The one thing `BLMOVEM` decides differently from the other five: `COUNT`
    /// is ready as soon as there is anything and `EXACTLY` is not ready until the
    /// whole block has arrived.
    #[test]
    fn a_blocking_block_move_waits_for_the_whole_block_only_under_exactly() {
        let mut f = Fixture::new();
        f.run(&[b"RPUSH", b"L", b"a", b"b"]);
        // Two there and three asked for. `COUNT` takes the two.
        assert_eq!(
            f.flow(&[
                b"BLMOVEM", b"L", b"D", b"LEFT", b"RIGHT", b"0", b"COUNT", b"3", b"BULK"
            ]),
            (Flow::Continue, bulks(&["a", "b"]))
        );

        f.run(&[b"RPUSH", b"L", b"a", b"b"]);
        // The same line with `EXACTLY` parks instead, and takes nothing on the
        // way past.
        assert_eq!(
            f.flow(&[
                b"BLMOVEM", b"L", b"D", b"LEFT", b"RIGHT", b"0", b"EXACTLY", b"3", b"BULK"
            ])
            .0,
            Flow::Block
        );
        assert_eq!(f.run(&[b"LRANGE", b"L", b"0", b"-1"]), bulks(&["a", "b"]));
    }

    #[test]
    fn a_blocking_block_move_reads_its_directions_then_its_timeout_then_its_count() {
        let mut f = Fixture::new();
        let syntax = "-ERR syntax error\r\n";
        // All three are wrong and the directions are read first.
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"a", b"b", b"UP", b"DOWN", b"abc", b"NOPE", b"x", b"y"
            ]),
            syntax
        );
        // Directions fine, timeout and count both wrong, so the timeout wins.
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"a", b"b", b"LEFT", b"RIGHT", b"abc", b"COUNT", b"abc", b"BULK"
            ]),
            "-ERR timeout is not a float or out of range\r\n"
        );
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"a", b"b", b"LEFT", b"RIGHT", b"-1", b"COUNT", b"1", b"BULK"
            ]),
            "-ERR timeout is negative\r\n"
        );
        // And with the timeout fine, the count before the ordering word.
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"a", b"b", b"LEFT", b"RIGHT", b"0", b"COUNT", b"abc", b"NOPE"
            ]),
            "-ERR count should be greater than 0\r\n"
        );
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"a", b"b", b"LEFT", b"RIGHT", b"0", b"COUNT", b"1", b"NOPE"
            ]),
            syntax
        );
        // Seven and eight arguments are neither of the two forms, the same way
        // six and seven are for `LMOVEM`.
        assert_eq!(
            f.run(&[b"BLMOVEM", b"a", b"b", b"LEFT", b"RIGHT", b"0", b"COUNT"]),
            syntax
        );
        assert_eq!(
            f.run(&[
                b"BLMOVEM", b"a", b"b", b"LEFT", b"RIGHT", b"0", b"COUNT", b"2"
            ]),
            syntax
        );
    }

    /// The four ways a blocking command sees a key of another type, and the one
    /// way it does not.
    #[test]
    fn a_blocking_command_errors_on_a_wrong_type_rather_than_waiting_on_it() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"S", b"v"]);
        f.run(&[b"RPUSH", b"D", b"x"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";

        assert_eq!(f.run(&[b"BLPOP", b"S", b"0"]), wrong);
        // Every key is checked even when an earlier one would have blocked, so
        // an empty key in front of a string does not hide it.
        assert_eq!(f.run(&[b"BLPOP", b"E", b"S", b"0"]), wrong);
        assert_eq!(f.run(&[b"BRPOP", b"S", b"0"]), wrong);
        assert_eq!(f.run(&[b"BLMPOP", b"0", b"1", b"S", b"LEFT"]), wrong);
        assert_eq!(f.run(&[b"BRPOPLPUSH", b"S", b"D", b"0"]), wrong);
        // The destination, which is only reached because the source has
        // something in it.
        assert_eq!(f.run(&[b"BRPOPLPUSH", b"D", b"S", b"0"]), wrong);
        assert_eq!(f.run(&[b"LRANGE", b"D", b"0", b"-1"]), "*1\r\n$1\r\nx\r\n");
        assert_eq!(
            f.run(&[b"BLMOVEM", b"S", b"D", b"LEFT", b"RIGHT", b"0"]),
            wrong
        );
        assert_eq!(
            f.run(&[b"BLMOVEM", b"D", b"S", b"LEFT", b"RIGHT", b"0"]),
            wrong
        );

        // And the one that does not: an empty source means the destination is
        // never looked at, so this waits rather than erroring, and on a real
        // server it times out.
        assert_eq!(
            f.flow(&[b"BLMOVE", b"E", b"S", b"LEFT", b"RIGHT", b"0.1"])
                .0,
            Flow::Block
        );
        // `BLMOVEM` has a second way of not being ready, and it hides the
        // destination just as well: the source is a list with two elements in it
        // and `EXACTLY` wants three, so the string never gets looked at.
        assert_eq!(
            f.flow(&[b"BLMOVEM", b"E", b"S", b"LEFT", b"RIGHT", b"0.1"])
                .0,
            Flow::Block
        );
        f.run(&[b"RPUSH", b"E", b"1", b"2"]);
        assert_eq!(
            f.flow(&[
                b"BLMOVEM", b"E", b"S", b"LEFT", b"RIGHT", b"0.1", b"EXACTLY", b"3", b"BULK"
            ])
            .0,
            Flow::Block
        );
    }

    /// The same churn the set and the string get, because a list that leaks a
    /// chunk per push looks exactly like one that does not until it has run for
    /// an afternoon.
    #[test]
    fn churning_lists_does_not_grow_the_server() {
        let mut f = Fixture::new();
        let vals: Vec<Vec<u8>> = (0..200).map(|i| format!("v{i}").into_bytes()).collect();
        let args: Vec<&[u8]> = [&b"RPUSH"[..], &b"k"[..]]
            .into_iter()
            .chain(vals.iter().map(Vec::as_slice))
            .collect();

        f.run(&args);
        f.run(&[b"DEL", b"k"]);
        f.server.compact_step();
        let after_first = f.server.memory_bytes();

        for _ in 0..200 {
            f.run(&args);
            f.run(&[b"LTRIM", b"k", b"1", b"0"]);
            f.server.compact_step();
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        assert!(
            f.server.memory_bytes() <= after_first * 2,
            "held {} after two hundred passes against {after_first} after one",
            f.server.memory_bytes()
        );
    }

    // ------------------------------------------------------------ sorted set

    #[test]
    fn a_sorted_set_takes_scores_and_gives_them_back() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b"]), ":2\r\n");
        assert_eq!(f.run(&[b"ZADD", b"z", b"1", b"a", b"3", b"c"]), ":1\r\n");
        assert_eq!(f.run(&[b"ZCARD", b"z"]), ":3\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"b"]), "$1\r\n2\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"nope"]), "$-1\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"nokey", b"b"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"ZMSCORE", b"z", b"a", b"nope", b"c"]),
            "*3\r\n$1\r\n1\r\n$-1\r\n$1\r\n3\r\n"
        );
        assert_eq!(f.run(&[b"ZREM", b"z", b"a", b"nope"]), ":1\r\n");
        assert_eq!(f.run(&[b"ZCARD", b"z"]), ":2\r\n");
        // The key goes when the last member does.
        assert_eq!(f.run(&[b"ZREM", b"z", b"b", b"c"]), ":2\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"z"]), ":0\r\n");
    }

    #[test]
    fn a_score_is_a_double_on_resp3_and_digits_on_resp2() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1.5", b"a", b"inf", b"b", b"-inf", b"c"]);
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"a"]), "$3\r\n1.5\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"b"]), "$3\r\ninf\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"c"]), "$4\r\n-inf\r\n");

        f.out = Out::new(Proto::Resp3);
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"a"]), ",1.5\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"b"]), ",inf\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"c"]), ",-inf\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"nope"]), "_\r\n");
    }

    #[test]
    fn the_zadd_options_gate_what_gets_written() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"5", b"a"]);
        // NX leaves a member that is there alone, XX will not create one.
        assert_eq!(f.run(&[b"ZADD", b"z", b"NX", b"9", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"a"]), "$1\r\n5\r\n");
        assert_eq!(f.run(&[b"ZADD", b"z", b"XX", b"9", b"new"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"z"]), ":1\r\n");
        // GT and LT only move a score one way.
        assert_eq!(f.run(&[b"ZADD", b"z", b"GT", b"CH", b"3", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"ZADD", b"z", b"GT", b"CH", b"7", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"ZADD", b"z", b"LT", b"CH", b"9", b"a"]), ":0\r\n");
        // CH counts a moved score and plain ZADD does not.
        assert_eq!(f.run(&[b"ZADD", b"z", b"1", b"a", b"1", b"b"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"CH", b"2", b"a", b"2", b"c"]),
            ":2\r\n"
        );
    }

    #[test]
    fn zadd_incr_answers_a_score_or_nothing_at_all() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ZADD", b"z", b"INCR", b"5", b"m"]), "$1\r\n5\r\n");
        assert_eq!(f.run(&[b"ZADD", b"z", b"INCR", b"2", b"m"]), "$1\r\n7\r\n");
        // A gate that refuses is the string nil, because the reply it stands in
        // for is a score.
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"NX", b"INCR", b"2", b"m"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"XX", b"INCR", b"2", b"gone"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"GT", b"INCR", b"-1", b"m"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"GT", b"INCR", b"1", b"m"]),
            "$1\r\n8\r\n"
        );
        assert_eq!(f.run(&[b"ZINCRBY", b"z", b"2", b"m"]), "$2\r\n10\r\n");
        assert_eq!(f.run(&[b"ZINCRBY", b"z", b"1", b"fresh"]), "$1\r\n1\r\n");
    }

    #[test]
    fn the_two_infinities_will_not_be_added_together() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"inf", b"m"]);
        let nan = "-ERR resulting score is not a number (NaN)\r\n";
        assert_eq!(f.run(&[b"ZINCRBY", b"z", b"-inf", b"m"]), nan);
        assert_eq!(f.run(&[b"ZADD", b"z", b"INCR", b"-inf", b"m"]), nan);
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"m"]), "$3\r\ninf\r\n");
        // And a key made for an increment that then fails does not stay behind.
        assert_eq!(f.run(&[b"ZINCRBY", b"gone", b"1", b"m"]), "$1\r\n1\r\n");
    }

    #[test]
    fn zadd_says_its_mistakes_the_way_redis_says_them() {
        let mut f = Fixture::new();
        // The pairs are counted before the options are looked at, so this is a
        // syntax error about having none and not a complaint about NX and XX.
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"NX", b"XX"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"NX", b"XX", b"1", b"a"]),
            "-ERR XX and NX options at the same time are not compatible\r\n"
        );
        let gtlt = "-ERR GT, LT, and/or NX options at the same time are not compatible\r\n";
        assert_eq!(f.run(&[b"ZADD", b"z", b"NX", b"GT", b"1", b"a"]), gtlt);
        assert_eq!(f.run(&[b"ZADD", b"z", b"GT", b"LT", b"1", b"a"]), gtlt);
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"INCR", b"1", b"a", b"2", b"b"]),
            "-ERR INCR option supports a single increment-element pair\r\n"
        );
        // An odd number of arguments after the options.
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"1", b"a", b"2"]),
            "-ERR syntax error\r\n"
        );
        // Every score is read before the first is stored.
        assert_eq!(
            f.run(&[b"ZADD", b"z", b"1", b"a", b"nonsense", b"b"]),
            "-ERR value is not a valid float\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"z"]), ":0\r\n");
    }

    #[test]
    fn a_rank_says_where_a_member_sits_from_either_end() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(f.run(&[b"ZRANK", b"z", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"ZRANK", b"z", b"c"]), ":2\r\n");
        assert_eq!(f.run(&[b"ZREVRANK", b"z", b"c"]), ":0\r\n");
        assert_eq!(f.run(&[b"ZREVRANK", b"z", b"a"]), ":2\r\n");
        // WITHSCORE changes both shapes: the answer and the nothing.
        assert_eq!(
            f.run(&[b"ZRANK", b"z", b"b", b"WITHSCORE"]),
            "*2\r\n:1\r\n$1\r\n2\r\n"
        );
        assert_eq!(f.run(&[b"ZRANK", b"z", b"nope"]), "$-1\r\n");
        assert_eq!(f.run(&[b"ZRANK", b"z", b"nope", b"WITHSCORE"]), "*-1\r\n");
        assert_eq!(f.run(&[b"ZRANK", b"nokey", b"a", b"WITHSCORE"]), "*-1\r\n");
        // A bad option is a syntax error and one argument too many is an arity
        // error, which is Redis's split.
        assert_eq!(
            f.run(&[b"ZRANK", b"z", b"b", b"bogus"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ZREVRANK", b"z", b"b", b"WITHSCORE", b"more"]),
            "-ERR wrong number of arguments for 'zrevrank' command\r\n"
        );
    }

    #[test]
    fn the_two_counts_read_their_two_kinds_of_bound() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(f.run(&[b"ZCOUNT", b"z", b"-inf", b"+inf"]), ":3\r\n");
        assert_eq!(f.run(&[b"ZCOUNT", b"z", b"2", b"3"]), ":2\r\n");
        assert_eq!(f.run(&[b"ZCOUNT", b"z", b"(1", b"3"]), ":2\r\n");
        assert_eq!(f.run(&[b"ZCOUNT", b"z", b"(1", b"(3"]), ":1\r\n");
        assert_eq!(f.run(&[b"ZCOUNT", b"nokey", b"-inf", b"+inf"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"ZCOUNT", b"z", b"bogus", b"3"]),
            "-ERR min or max is not a float\r\n"
        );

        f.run(&[b"ZADD", b"l", b"0", b"a", b"0", b"b", b"0", b"c"]);
        assert_eq!(f.run(&[b"ZLEXCOUNT", b"l", b"-", b"+"]), ":3\r\n");
        assert_eq!(f.run(&[b"ZLEXCOUNT", b"l", b"[a", b"(c"]), ":2\r\n");
        assert_eq!(f.run(&[b"ZLEXCOUNT", b"l", b"(a", b"+"]), ":2\r\n");
        // A bare member is not a bound, because a member can start with any
        // byte and there would be no way to say the bracket if it were optional.
        assert_eq!(
            f.run(&[b"ZLEXCOUNT", b"l", b"a", b"c"]),
            "-ERR min or max not valid string range item\r\n"
        );
    }

    /// The three ways `ZRANGE` can be asked for a window, forwards and back.
    ///
    /// Every byte in here was read off a real 8.10.1 rather than worked out,
    /// because the interesting part of this command is not what it selects, it
    /// is which of the two ends the client is expected to name first.
    #[test]
    fn one_range_command_selects_by_rank_or_score_or_name() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1"]),
            "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"-2", b"-1"]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(f.run(&[b"ZRANGE", b"z", b"5", b"9"]), "*0\r\n");
        assert_eq!(f.run(&[b"ZRANGE", b"nokey", b"0", b"-1"]), "*0\r\n");
        // REV over ranks reverses the walk and leaves the two arguments alone,
        // because a rank counts from the end the walk starts at.
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1", b"REV"]),
            "*3\r\n$1\r\nc\r\n$1\r\nb\r\n$1\r\na\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"(1", b"+inf", b"BYSCORE"]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        // And REV over scores does swap them, since a bound does not count from
        // anywhere. This is the one line of the parse that tells the two apart.
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"+inf", b"(1", b"BYSCORE", b"REV"]),
            "*2\r\n$1\r\nc\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"-", b"+", b"BYLEX"]),
            "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"+", b"-", b"BYLEX", b"REV"]),
            "*3\r\n$1\r\nc\r\n$1\r\nb\r\n$1\r\na\r\n"
        );
    }

    /// The older spellings, which are the same six windows with the mode in the
    /// name and the high end named first on the three that go backwards.
    #[test]
    fn the_older_range_spellings_name_their_high_end_first() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(
            f.run(&[b"ZREVRANGE", b"z", b"0", b"-1"]),
            "*3\r\n$1\r\nc\r\n$1\r\nb\r\n$1\r\na\r\n"
        );
        assert_eq!(
            f.run(&[b"ZREVRANGE", b"z", b"0", b"0", b"WITHSCORES"]),
            "*2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGEBYSCORE", b"z", b"(1", b"3"]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"ZREVRANGEBYSCORE", b"z", b"3", b"(1"]),
            "*2\r\n$1\r\nc\r\n$1\r\nb\r\n"
        );
        // The two arguments the wrong way round is an empty answer and not an
        // error, which is what the swap being in the parse rather than in the
        // window buys.
        assert_eq!(f.run(&[b"ZREVRANGEBYSCORE", b"z", b"(1", b"3"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"ZRANGEBYLEX", b"z", b"[a", b"(c"]),
            "*2\r\n$1\r\na\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[b"ZREVRANGEBYLEX", b"z", b"(c", b"[a"]),
            "*2\r\n$1\r\nb\r\n$1\r\na\r\n"
        );
        // BYSCORE, BYLEX and REV mean nothing to these, so they are not another
        // way of spelling the mode, they are a syntax error.
        for cmd in [
            &[b"ZREVRANGE".as_slice(), b"z", b"0", b"-1", b"BYSCORE"][..],
            &[b"ZRANGEBYSCORE", b"z", b"1", b"3", b"REV"],
            &[b"ZRANGEBYLEX", b"z", b"[a", b"[c", b"BYLEX"],
        ] {
            assert_eq!(f.run(cmd), "-ERR syntax error\r\n", "{:?}", cmd[0]);
        }
    }

    /// `LIMIT` and `WITHSCORES`, which every one of these commands reads and
    /// only some of them accept.
    #[test]
    fn limit_and_withscores_are_read_by_all_of_them_and_refused_afterwards() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(
            f.run(&[
                b"ZRANGE", b"z", b"-inf", b"+inf", b"BYSCORE", b"LIMIT", b"1", b"1"
            ]),
            "*1\r\n$1\r\nb\r\n"
        );
        // A negative offset skips past everything, a negative count is no bound.
        assert_eq!(
            f.run(&[
                b"ZRANGE", b"z", b"-inf", b"+inf", b"BYSCORE", b"LIMIT", b"-1", b"2"
            ]),
            "*0\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ZRANGE", b"z", b"-inf", b"+inf", b"BYSCORE", b"LIMIT", b"0", b"-1"
            ]),
            "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        // The two options in either order, which falls out of the parse loop.
        let both = "*4\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n";
        assert_eq!(
            f.run(&[
                b"ZRANGEBYSCORE",
                b"z",
                b"1",
                b"3",
                b"WITHSCORES",
                b"LIMIT",
                b"0",
                b"2"
            ]),
            both
        );
        assert_eq!(
            f.run(&[
                b"ZRANGEBYSCORE",
                b"z",
                b"1",
                b"3",
                b"LIMIT",
                b"0",
                b"2",
                b"WITHSCORES"
            ]),
            both
        );
        // LIMIT on a range by rank is refused after the whole option list has
        // been read, so this complains about LIMIT and not about WITHSCORES.
        let needs_by = "-ERR syntax error, LIMIT is only supported in combination with either BYSCORE or BYLEX\r\n";
        assert_eq!(
            f.run(&[
                b"ZREVRANGE",
                b"z",
                b"0",
                b"-1",
                b"WITHSCORES",
                b"LIMIT",
                b"0",
                b"1"
            ]),
            needs_by
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1", b"LIMIT", b"0", b"1"]),
            needs_by
        );
        let not_bylex = "-ERR syntax error, WITHSCORES not supported in combination with BYLEX\r\n";
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"-", b"+", b"BYLEX", b"WITHSCORES"]),
            not_bylex
        );
        assert_eq!(
            f.run(&[b"ZRANGEBYLEX", b"z", b"[a", b"[c", b"WITHSCORES"]),
            not_bylex
        );
        // Two modes at once, an option nobody knows, a LIMIT missing its count,
        // and the three number errors, which are three different sentences.
        for cmd in [
            &[
                b"ZRANGE".as_slice(),
                b"z",
                b"0",
                b"-1",
                b"BYSCORE",
                b"BYLEX",
            ][..],
            &[b"ZRANGE", b"z", b"0", b"-1", b"junk"],
            &[b"ZRANGEBYSCORE", b"z", b"1", b"3", b"LIMIT", b"0"],
        ] {
            assert_eq!(f.run(cmd), "-ERR syntax error\r\n", "{cmd:?}");
        }
        assert_eq!(
            f.run(&[b"ZRANGEBYSCORE", b"z", b"bad", b"3"]),
            "-ERR min or max is not a float\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGEBYLEX", b"z", b"a", b"[c"]),
            "-ERR min or max not valid string range item\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGEBYSCORE", b"z", b"1", b"3", b"LIMIT", b"a", b"2"]),
            "-ERR value is not an integer or out of range\r\n"
        );
    }

    /// `WITHSCORES` is the one place in this group where the two protocols
    /// disagree about the shape of the reply and not just the type of a value.
    #[test]
    fn withscores_nests_on_resp3_and_flattens_on_resp2() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1", b"WITHSCORES"]),
            "*6\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        f.out = Out::new(Proto::Resp3);
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1", b"WITHSCORES"]),
            "*3\r\n*2\r\n$1\r\na\r\n,1\r\n*2\r\n$1\r\nb\r\n,2\r\n*2\r\n$1\r\nc\r\n,3\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1"]),
            "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
    }

    /// The store form, which is the same parse with the destination in front.
    #[test]
    fn a_range_store_writes_the_window_into_another_key() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(f.run(&[b"ZRANGESTORE", b"d", b"z", b"0", b"-1"]), ":3\r\n");
        // A window that selects nothing deletes the destination rather than
        // leaving an empty sorted set, because an empty one does not exist.
        assert_eq!(f.run(&[b"ZRANGESTORE", b"d", b"z", b"5", b"9"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"d"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"ZRANGESTORE", b"d", b"z", b"(1", b"+inf", b"BYSCORE"]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"d", b"0", b"-1", b"WITHSCORES"]),
            "*4\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        // The destination is allowed to be the source, because the result is
        // built whole before anything is written over.
        assert_eq!(f.run(&[b"ZRANGESTORE", b"z", b"z", b"1", b"2"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1", b"WITHSCORES"]),
            "*4\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        // It takes every option ZRANGE takes except WITHSCORES, which is a
        // plain syntax error here and not the sentence about BYLEX.
        assert_eq!(
            f.run(&[b"ZRANGESTORE", b"d", b"z", b"0", b"-1", b"WITHSCORES"]),
            "-ERR syntax error\r\n"
        );
    }

    /// The three removals, which are the read side's window with the walk
    /// turned into a removal and no options at all.
    #[test]
    fn the_three_removals_share_their_window_with_the_reads() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(f.run(&[b"ZREMRANGEBYRANK", b"z", b"0", b"0"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"ZRANGE", b"z", b"0", b"-1"]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"ZREMRANGEBYSCORE", b"z", b"(2", b"+inf"]),
            ":1\r\n"
        );
        assert_eq!(f.run(&[b"ZRANGE", b"z", b"0", b"-1"]), "*1\r\n$1\r\nb\r\n");
        // The last member going takes the key with it.
        assert_eq!(f.run(&[b"ZREMRANGEBYLEX", b"z", b"-", b"+"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"z"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"ZREMRANGEBYRANK", b"nokey", b"0", b"-1"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"ZREMRANGEBYRANK", b"z", b"0", b"x"]),
            "-ERR value is not an integer or out of range\r\n"
        );
    }

    /// The algebra, which is one gather and three names for it.
    #[test]
    fn the_three_algebra_commands_combine_scores_and_order_the_answer_once() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        f.run(&[b"ZADD", b"y", b"10", b"b", b"20", b"d"]);
        assert_eq!(
            f.run(&[b"ZUNION", b"2", b"z", b"y"]),
            "*4\r\n$1\r\na\r\n$1\r\nc\r\n$1\r\nb\r\n$1\r\nd\r\n"
        );
        // The scores are added where a member is in both, and the answer comes
        // out in the order those combined scores put it in.
        assert_eq!(
            f.run(&[b"ZUNION", b"2", b"z", b"y", b"WITHSCORES"]),
            "*8\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nc\r\n$1\r\n3\r\n$1\r\nb\r\n$2\r\n12\r\n$1\r\nd\r\n$2\r\n20\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ZUNION",
                b"2",
                b"z",
                b"y",
                b"WEIGHTS",
                b"2",
                b"3",
                b"WITHSCORES"
            ]),
            "*8\r\n$1\r\na\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n6\r\n$1\r\nb\r\n$2\r\n34\r\n$1\r\nd\r\n$2\r\n60\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ZUNION",
                b"2",
                b"z",
                b"y",
                b"AGGREGATE",
                b"MIN",
                b"WITHSCORES"
            ]),
            "*8\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n$1\r\nd\r\n$2\r\n20\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ZUNION",
                b"2",
                b"z",
                b"y",
                b"AGGREGATE",
                b"MAX",
                b"WITHSCORES"
            ]),
            "*8\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nc\r\n$1\r\n3\r\n$1\r\nb\r\n$2\r\n10\r\n$1\r\nd\r\n$2\r\n20\r\n"
        );
        assert_eq!(
            f.run(&[b"ZINTER", b"2", b"z", b"y", b"WITHSCORES"]),
            "*2\r\n$1\r\nb\r\n$2\r\n12\r\n"
        );
        assert_eq!(
            f.run(&[b"ZDIFF", b"2", b"z", b"y", b"WITHSCORES"]),
            "*4\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        assert_eq!(f.run(&[b"ZUNION", b"1", b"nokey"]), "*0\r\n");
        // A plain set is an input, and it behaves as a sorted set in which
        // every member scores one.
        f.run(&[b"SADD", b"p", b"a", b"d"]);
        assert_eq!(
            f.run(&[b"ZUNION", b"2", b"z", b"p", b"WITHSCORES"]),
            "*8\r\n$1\r\nd\r\n$1\r\n1\r\n$1\r\na\r\n$1\r\n2\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        // A difference never combines two scores, so it has nothing for either
        // of the two options to do and refuses both.
        for cmd in [
            &[
                b"ZDIFF".as_slice(),
                b"2",
                b"z",
                b"y",
                b"WEIGHTS",
                b"1",
                b"1",
            ][..],
            &[b"ZDIFF", b"2", b"z", b"y", b"AGGREGATE", b"MIN"],
        ] {
            assert_eq!(f.run(cmd), "-ERR syntax error\r\n", "{cmd:?}");
        }
    }

    /// The count of keys, which is what lets a key be named `WEIGHTS`.
    #[test]
    fn the_algebra_counts_its_keys_and_says_so_when_the_count_is_wrong() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a"]);
        f.run(&[b"ZADD", b"y", b"2", b"b"]);
        // Redis names the command in this one, so each spelling says its own.
        assert_eq!(
            f.run(&[b"ZUNION", b"0", b"z"]),
            "-ERR at least 1 input key is needed for 'zunion' command\r\n"
        );
        assert_eq!(
            f.run(&[b"ZUNION", b"-1", b"z"]),
            "-ERR at least 1 input key is needed for 'zunion' command\r\n"
        );
        assert_eq!(
            f.run(&[b"ZINTERCARD", b"0", b"z"]),
            "-ERR at least 1 input key is needed for 'zintercard' command\r\n"
        );
        // A count bigger than the line is a plain syntax error, which reads
        // oddly and is what Redis says.
        assert_eq!(
            f.run(&[b"ZUNION", b"3", b"z", b"y"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ZUNION", b"x", b"z"]),
            "-ERR value is not an integer or out of range\r\n"
        );
        // A WEIGHTS list that is not one per key is a syntax error, and a
        // weight that is not a number gets a sentence of its own.
        assert_eq!(
            f.run(&[b"ZUNION", b"2", b"z", b"y", b"WEIGHTS", b"1"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ZUNION", b"2", b"z", b"y", b"WEIGHTS", b"a", b"b"]),
            "-ERR weight value is not a float\r\n"
        );
        assert_eq!(
            f.run(&[b"ZUNION", b"2", b"z", b"y", b"AGGREGATE", b"NOPE"]),
            "-ERR syntax error\r\n"
        );
    }

    /// The three store forms, which answer a count and take no WITHSCORES.
    #[test]
    fn the_algebra_stores_answer_a_count_and_delete_an_empty_destination() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        f.run(&[b"ZADD", b"y", b"10", b"b", b"20", b"d"]);
        assert_eq!(f.run(&[b"ZUNIONSTORE", b"d", b"2", b"z", b"y"]), ":4\r\n");
        assert_eq!(
            f.run(&[b"ZRANGE", b"d", b"0", b"-1", b"WITHSCORES"]),
            "*8\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nc\r\n$1\r\n3\r\n$1\r\nb\r\n$2\r\n12\r\n$1\r\nd\r\n$2\r\n20\r\n"
        );
        assert_eq!(f.run(&[b"ZINTERSTORE", b"d", b"2", b"z", b"y"]), ":1\r\n");
        assert_eq!(f.run(&[b"ZDIFFSTORE", b"d", b"2", b"z", b"y"]), ":2\r\n");
        // An empty result deletes the destination rather than leaving an empty
        // sorted set, because an empty one does not exist.
        assert_eq!(
            f.run(&[b"ZINTERSTORE", b"d", b"2", b"z", b"nokey"]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"d"]), ":0\r\n");
        // The destination is allowed to name its own source.
        assert_eq!(f.run(&[b"ZUNIONSTORE", b"z", b"2", b"z", b"y"]), ":4\r\n");
        assert_eq!(f.run(&[b"ZCARD", b"z"]), ":4\r\n");
        for cmd in [
            &[
                b"ZUNIONSTORE".as_slice(),
                b"d",
                b"2",
                b"z",
                b"y",
                b"WITHSCORES",
            ][..],
            &[
                b"ZDIFFSTORE",
                b"d",
                b"2",
                b"z",
                b"y",
                b"WEIGHTS",
                b"1",
                b"1",
            ],
        ] {
            assert_eq!(f.run(cmd), "-ERR syntax error\r\n", "{cmd:?}");
        }
    }

    /// `ZINTERCARD`, which counts without building anything.
    #[test]
    fn intercard_counts_and_stops_at_its_limit() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        f.run(&[b"ZADD", b"y", b"10", b"b", b"20", b"c", b"30", b"d"]);
        assert_eq!(f.run(&[b"ZINTERCARD", b"2", b"z", b"y"]), ":2\r\n");
        // A limit of zero is no limit, which is Redis's reading of it.
        assert_eq!(
            f.run(&[b"ZINTERCARD", b"2", b"z", b"y", b"LIMIT", b"0"]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"ZINTERCARD", b"2", b"z", b"y", b"LIMIT", b"1"]),
            ":1\r\n"
        );
        // A negative limit and a limit that is not a number at all get the same
        // sentence, which looks like a mistake in Redis and is copied as one.
        let bad = "-ERR LIMIT can't be negative\r\n";
        assert_eq!(
            f.run(&[b"ZINTERCARD", b"2", b"z", b"y", b"LIMIT", b"-1"]),
            bad
        );
        assert_eq!(
            f.run(&[b"ZINTERCARD", b"2", b"z", b"y", b"LIMIT", b"x"]),
            bad
        );
        for cmd in [
            &[b"ZINTERCARD".as_slice(), b"3", b"z", b"y"][..],
            &[b"ZINTERCARD", b"2", b"z", b"y", b"LIMIT"],
            &[b"ZINTERCARD", b"2", b"z", b"y", b"junk", b"1"],
        ] {
            assert_eq!(f.run(cmd), "-ERR syntax error\r\n", "{cmd:?}");
        }
    }

    /// `ZRANDMEMBER`, which answers two different shapes out of one name.
    #[test]
    fn a_draw_answers_one_member_or_an_array_of_them() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        // No count is one member or a nil, a count is an array that may be
        // empty, and those are two reply types the client has to tell apart.
        assert_eq!(f.run(&[b"ZRANDMEMBER", b"nokey"]), "$-1\r\n");
        assert_eq!(f.run(&[b"ZRANDMEMBER", b"nokey", b"3"]), "*0\r\n");
        assert_eq!(f.run(&[b"ZRANDMEMBER", b"z", b"0"]), "*0\r\n");
        assert!(f.run(&[b"ZRANDMEMBER", b"z"]).starts_with("$1\r\n"));
        // A positive count draws without replacement, so a count over the size
        // answers the whole set and never a member twice.
        let all = f.run(&[b"ZRANDMEMBER", b"z", b"10"]);
        assert!(all.starts_with("*3\r\n"), "{all}");
        for m in ["a", "b", "c"] {
            assert!(all.contains(m), "{all}");
        }
        // A negative one draws with replacement and answers exactly as many as
        // it was asked for, whatever the size of the set.
        assert!(
            f.run(&[b"ZRANDMEMBER", b"z", b"-5"]).starts_with("*5\r\n"),
            "five draws with replacement"
        );
        assert!(
            f.run(&[b"ZRANDMEMBER", b"z", b"2", b"WITHSCORES"])
                .starts_with("*4\r\n"),
            "two pairs, flat on RESP2"
        );
        f.out = Out::new(Proto::Resp3);
        let got = f.run(&[b"ZRANDMEMBER", b"z", b"2", b"WITHSCORES"]);
        assert!(got.starts_with("*2\r\n*2\r\n"), "{got}");
        assert_eq!(f.run(&[b"ZRANDMEMBER", b"nokey"]), "_\r\n");
        f.out = Out::new(Proto::Resp2);
        assert_eq!(
            f.run(&[b"ZRANDMEMBER", b"z", b"2", b"junk"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANDMEMBER", b"z", b"x"]),
            "-ERR value is not an integer or out of range\r\n"
        );
    }

    /// `ZSCAN`, and the one sorted set reply where a score is not a double.
    #[test]
    fn a_sorted_set_scan_answers_pairs_of_strings_on_both_protocols() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        let all = "*2\r\n$1\r\n0\r\n*6\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n";
        assert_eq!(f.run(&[b"ZSCAN", b"z", b"0"]), all);
        assert_eq!(f.run(&[b"ZSCAN", b"z", b"0", b"COUNT", b"10"]), all);
        assert_eq!(
            f.run(&[b"ZSCAN", b"z", b"0", b"MATCH", b"a*"]),
            "*2\r\n$1\r\n0\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        assert_eq!(
            f.run(&[b"ZSCAN", b"nokey", b"0"]),
            "*2\r\n$1\r\n0\r\n*0\r\n"
        );
        // A score stays a bulk string on RESP3, which is the one place the two
        // protocols agree about a score and everywhere else they do not.
        f.out = Out::new(Proto::Resp3);
        assert_eq!(f.run(&[b"ZSCAN", b"z", b"0"]), all);
        f.out = Out::new(Proto::Resp2);
        assert_eq!(
            f.run(&[b"ZSCAN", b"z", b"0", b"NOVALUES"]),
            "-ERR NOVALUES option can only be used in HSCAN\r\n"
        );
        assert_eq!(f.run(&[b"ZSCAN", b"z", b"-1"]), "-ERR invalid cursor\r\n");
        assert_eq!(
            f.run(&[b"ZSCAN", b"z", b"0", b"COUNT", b"0"]),
            "-ERR syntax error\r\n"
        );
    }

    /// The count is what decides the shape, and its value is not.
    #[test]
    fn a_sorted_set_pop_changes_shape_when_it_is_given_a_count() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        // No count, so one flat pair, and the score is a bulk string on RESP2.
        assert_eq!(f.run(&[b"ZPOPMIN", b"z"]), "*2\r\n$1\r\na\r\n$1\r\n1\r\n");
        assert_eq!(f.run(&[b"ZPOPMAX", b"z"]), "*2\r\n$1\r\nc\r\n$1\r\n3\r\n");
        f.run(&[b"ZADD", b"z", b"1", b"a", b"3", b"c"]);
        // A count, so pairs, and on RESP2 they are flattened into one run.
        assert_eq!(
            f.run(&[b"ZPOPMIN", b"z", b"2"]),
            "*4\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        // An empty array rather than a null, which is where a sorted set pop and
        // a list pop part company, and the same answer a count of zero gives.
        assert_eq!(f.run(&[b"ZPOPMIN", b"nokey"]), "*0\r\n");
        assert_eq!(f.run(&[b"ZPOPMIN", b"nokey", b"2"]), "*0\r\n");
        assert_eq!(f.run(&[b"ZPOPMIN", b"z", b"0"]), "*0\r\n");
        // The last member takes the key with it.
        assert_eq!(
            f.run(&[b"ZPOPMIN", b"z", b"9"]),
            "*2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"z"]), ":0\r\n");

        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b"]);
        f.out = Out::new(Proto::Resp3);
        assert_eq!(f.run(&[b"ZPOPMIN", b"z"]), "*2\r\n$1\r\na\r\n,1\r\n");
        assert_eq!(
            f.run(&[b"ZPOPMIN", b"z", b"1"]),
            "*1\r\n*2\r\n$1\r\nb\r\n,2\r\n"
        );
        f.out = Out::new(Proto::Resp2);
        // Both of these are the range error rather than the usual sentence about
        // integers, which is the odd answer and so the one worth copying.
        let bad = "-ERR value is out of range, must be positive\r\n";
        assert_eq!(f.run(&[b"ZPOPMIN", b"z", b"x"]), bad);
        assert_eq!(f.run(&[b"ZPOPMIN", b"z", b"-1"]), bad);
        assert_eq!(
            f.run(&[b"ZPOPMIN", b"z", b"1", b"2"]),
            "-ERR syntax error\r\n"
        );
    }

    /// `ZMPOP`, which is `LMPOP` with scores and the same parse.
    #[test]
    fn a_multi_key_pop_names_the_key_that_answered_and_nests_its_pairs() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(
            f.run(&[b"ZMPOP", b"2", b"nokey", b"z", b"MIN"]),
            "*2\r\n$1\r\nz\r\n*1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        // Nested on RESP2 as well, because the key name is already in front of
        // the pairs and there is nothing left to flatten into.
        assert_eq!(
            f.run(&[b"ZMPOP", b"1", b"z", b"MAX", b"COUNT", b"2"]),
            "*2\r\n$1\r\nz\r\n*2\r\n*2\r\n$1\r\nc\r\n$1\r\n3\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        // A null array and not a null, the same as LMPOP.
        assert_eq!(f.run(&[b"ZMPOP", b"1", b"nokey", b"MIN"]), "*-1\r\n");
        f.out = Out::new(Proto::Resp3);
        assert_eq!(f.run(&[b"ZMPOP", b"1", b"nokey", b"MIN"]), "_\r\n");
        f.out = Out::new(Proto::Resp2);
        let numkeys = "-ERR numkeys should be greater than 0\r\n";
        for bad in [
            &[b"ZMPOP".as_slice(), b"0", b"z", b"MIN"][..],
            &[b"ZMPOP", b"-1", b"z", b"MIN"],
            &[b"ZMPOP", b"x", b"z", b"MIN"],
        ] {
            assert_eq!(f.run(bad), numkeys, "{:?}", bad[1]);
        }
        let count = "-ERR count should be greater than 0\r\n";
        for bad in [
            &[b"ZMPOP".as_slice(), b"1", b"z", b"MIN", b"COUNT", b"0"][..],
            &[b"ZMPOP", b"1", b"z", b"MIN", b"COUNT", b"-1"],
            &[b"ZMPOP", b"1", b"z", b"MIN", b"COUNT", b"x"],
        ] {
            assert_eq!(f.run(bad), count, "{:?}", bad[5]);
        }
        let syntax = "-ERR syntax error\r\n";
        for bad in [
            // Two keys named and one given, so the word that should have been
            // the direction is a key and there is no direction left.
            &[b"ZMPOP".as_slice(), b"2", b"z", b"MIN"][..],
            &[b"ZMPOP", b"1", b"z", b"SIDEWAYS"],
            &[b"ZMPOP", b"1", b"z", b"MIN", b"junk"],
            &[b"ZMPOP", b"1", b"z", b"MIN", b"COUNT", b"1", b"junk"],
        ] {
            assert_eq!(f.run(bad), syntax, "{bad:?}");
        }
    }

    /// The three that wait, when there is something there and they do not have
    /// to. `BZPOPMIN` is the one reply in the group that is three flat elements.
    #[test]
    fn the_sorted_set_pops_that_wait_answer_like_the_ones_they_wrap() {
        let mut f = Fixture::new();
        f.run(&[b"ZADD", b"z", b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(
            f.flow(&[b"BZPOPMIN", b"nokey", b"z", b"0"]),
            (
                Flow::Continue,
                "*3\r\n$1\r\nz\r\n$1\r\na\r\n$1\r\n1\r\n".to_owned()
            )
        );
        assert_eq!(
            f.run(&[b"BZPOPMAX", b"z", b"0"]),
            "*3\r\n$1\r\nz\r\n$1\r\nc\r\n$1\r\n3\r\n"
        );
        f.run(&[b"ZADD", b"z", b"1", b"a", b"3", b"c"]);
        assert_eq!(
            f.run(&[
                b"BZMPOP", b"0", b"2", b"nokey", b"z", b"MIN", b"COUNT", b"2"
            ]),
            "*2\r\n$1\r\nz\r\n*2\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        f.out = Out::new(Proto::Resp3);
        assert_eq!(
            f.run(&[b"BZPOPMIN", b"z", b"0"]),
            "*3\r\n$1\r\nz\r\n$1\r\nc\r\n,3\r\n"
        );
        f.out = Out::new(Proto::Resp2);
        // Nothing to take, so the client is parked and nothing was written.
        assert_eq!(
            f.flow(&[b"BZPOPMIN", b"z", b"0"]),
            (Flow::Block, String::new())
        );
        assert_eq!(
            f.flow(&[b"BZMPOP", b"0", b"1", b"z", b"MIN"]),
            (Flow::Block, String::new())
        );
        // The timeout is read before the key count, so this complains about the
        // timeout and not about the count.
        assert_eq!(
            f.run(&[b"BZMPOP", b"abc", b"0", b"z", b"MIN"]),
            "-ERR timeout is not a float or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"BZMPOP", b"0", b"0", b"z", b"MIN"]),
            "-ERR numkeys should be greater than 0\r\n"
        );
        assert_eq!(
            f.run(&[b"BZPOPMIN", b"z", b"-1"]),
            "-ERR timeout is negative\r\n"
        );
    }

    /// A parked sorted set client is served by whatever puts a member under one
    /// of its keys, and is not served by something of another type landing
    /// there.
    #[test]
    fn a_parked_sorted_set_client_waits_for_a_member_and_not_for_a_key() {
        let mut f = Fixture::new();
        assert_eq!(f.flow(&[b"BZPOPMIN", b"z", b"0"]).0, Flow::Block);
        assert_eq!(f.server.parked(), 1);
        // A string under the key is not what it asked for, so it stays parked
        // rather than being handed a WRONGTYPE on a command that was accepted.
        f.run(&[b"SET", b"z", b"v"]);
        let mut out = Out::new(Proto::Resp2);
        assert!(!f.server.serve_waiter(0, 0, &mut out));
        assert!(out.as_slice().is_empty());
        f.run(&[b"DEL", b"z"]);
        f.run(&[b"ZADD", b"z", b"5", b"m"]);
        assert!(f.server.serve_waiter(0, 0, &mut out));
        assert_eq!(
            core::str::from_utf8(out.as_slice()).expect("ascii"),
            "*3\r\n$1\r\nz\r\n$1\r\nm\r\n$1\r\n5\r\n"
        );
        // And the member is gone, which is what makes a queue of workers on a
        // sorted set work at all.
        assert_eq!(f.run(&[b"EXISTS", b"z"]), ":0\r\n");
    }

    #[test]
    fn every_sorted_set_command_says_wrongtype_and_writes_nothing() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        for cmd in [
            &[b"ZADD".as_slice(), b"s", b"1", b"a"][..],
            &[b"ZINCRBY", b"s", b"1", b"a"],
            &[b"ZCARD", b"s"],
            &[b"ZSCORE", b"s", b"a"],
            &[b"ZMSCORE", b"s", b"a"],
            &[b"ZREM", b"s", b"a"],
            &[b"ZRANK", b"s", b"a"],
            &[b"ZREVRANK", b"s", b"a"],
            &[b"ZCOUNT", b"s", b"1", b"2"],
            &[b"ZLEXCOUNT", b"s", b"-", b"+"],
            &[b"ZRANGE", b"s", b"0", b"-1"],
            &[b"ZREVRANGE", b"s", b"0", b"-1"],
            &[b"ZRANGEBYSCORE", b"s", b"1", b"2"],
            &[b"ZREVRANGEBYSCORE", b"s", b"2", b"1"],
            &[b"ZRANGEBYLEX", b"s", b"-", b"+"],
            &[b"ZREVRANGEBYLEX", b"s", b"+", b"-"],
            &[b"ZRANGESTORE", b"d", b"s", b"0", b"-1"],
            &[b"ZREMRANGEBYRANK", b"s", b"0", b"-1"],
            &[b"ZREMRANGEBYSCORE", b"s", b"1", b"2"],
            &[b"ZREMRANGEBYLEX", b"s", b"-", b"+"],
            &[b"ZUNION", b"1", b"s"],
            &[b"ZINTER", b"1", b"s"],
            &[b"ZDIFF", b"1", b"s"],
            &[b"ZUNIONSTORE", b"d", b"1", b"s"],
            &[b"ZINTERSTORE", b"d", b"1", b"s"],
            &[b"ZDIFFSTORE", b"d", b"1", b"s"],
            &[b"ZINTERCARD", b"1", b"s"],
            &[b"ZRANDMEMBER", b"s"],
            &[b"ZSCAN", b"s", b"0"],
            &[b"ZPOPMIN", b"s"],
            &[b"ZPOPMAX", b"s", b"2"],
            &[b"ZMPOP", b"1", b"s", b"MIN"],
            &[b"BZPOPMIN", b"s", b"0"],
            &[b"BZPOPMAX", b"s", b"0"],
            &[b"BZMPOP", b"0", b"1", b"s", b"MIN"],
        ] {
            assert_eq!(f.run(cmd), wrong, "{:?}", cmd[0]);
        }
        assert_eq!(f.run(&[b"GET", b"s"]), "$1\r\nv\r\n");
    }

    /// The same churn the set, the string and the list get, because a sorted
    /// set that leaks a tree node per add looks exactly like one that does not
    /// until it has run for an afternoon.
    #[test]
    fn churning_sorted_sets_does_not_grow_the_server() {
        let mut f = Fixture::new();
        let members: Vec<Vec<u8>> = (0..200).map(|i| format!("m{i}").into_bytes()).collect();
        let scores: Vec<Vec<u8>> = (0..200).map(|i| format!("{i}").into_bytes()).collect();
        let mut args: Vec<&[u8]> = vec![b"ZADD", b"z"];
        for i in 0..200 {
            args.push(&scores[i]);
            args.push(&members[i]);
        }

        f.run(&args);
        f.run(&[b"DEL", b"z"]);
        f.server.compact_step();
        let after_first = f.server.memory_bytes();

        for _ in 0..200 {
            f.run(&args);
            f.run(&[b"DEL", b"z"]);
            f.server.compact_step();
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        assert!(
            f.server.memory_bytes() <= after_first * 2,
            "held {} after two hundred passes against {after_first} after one",
            f.server.memory_bytes()
        );
    }

    // ------------------------------------------------------------------- geo

    /// The three places every Redis geo example uses, and one more.
    ///
    /// Every reply this section asserts on came off a running 8.10.1 with these
    /// three loaded, byte for byte, including the number of digits in a
    /// coordinate and the four places on a distance.
    fn sicily(f: &mut Fixture) {
        f.run(&[
            b"GEOADD",
            b"Sicily",
            b"13.361389",
            b"38.115556",
            b"Palermo",
            b"15.087269",
            b"37.502669",
            b"Catania",
        ]);
        f.run(&[
            b"GEOADD",
            b"Sicily",
            b"13.583333",
            b"37.316667",
            b"Agrigento",
        ]);
    }

    #[test]
    fn places_go_in_as_scores_and_come_back_as_positions() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"GEOADD",
                b"Sicily",
                b"13.361389",
                b"38.115556",
                b"Palermo",
                b"15.087269",
                b"37.502669",
                b"Catania"
            ]),
            ":2\r\n"
        );
        // A geo key is a sorted set and says so, which is not an implementation
        // detail either: a client removes a place with ZREM and counts them
        // with ZCARD, and the score is the number a real server stores.
        assert_eq!(f.run(&[b"TYPE", b"Sicily"]), "+zset\r\n");
        assert_eq!(
            f.run(&[b"ZSCORE", b"Sicily", b"Palermo"]),
            "$16\r\n3479099956230698\r\n"
        );
        assert_eq!(
            f.run(&[b"GEOPOS", b"Sicily", b"Palermo", b"NonExisting"]),
            "*2\r\n*2\r\n$18\r\n13.361389338970184\r\n$16\r\n38.1155563954963\r\n*-1\r\n"
        );
        assert_eq!(
            f.run(&[
                b"GEOHASH",
                b"Sicily",
                b"Palermo",
                b"Catania",
                b"NonExisting"
            ]),
            "*3\r\n$11\r\nsqc8b49rny0\r\n$11\r\nsqdtr74hyu0\r\n$-1\r\n"
        );
        // A key that is not there is an empty one, and the two nulls are not
        // the same null: GEOPOS answers the array one and GEOHASH the string
        // one, which a RESP2 client can tell apart.
        assert_eq!(f.run(&[b"GEOPOS", b"nokey", b"a"]), "*1\r\n*-1\r\n");
        assert_eq!(f.run(&[b"GEOHASH", b"nokey", b"a"]), "*1\r\n$-1\r\n");
    }

    #[test]
    fn a_distance_comes_back_with_four_places_in_whatever_unit_was_asked_for() {
        let mut f = Fixture::new();
        sicily(&mut f);
        assert_eq!(
            f.run(&[b"GEODIST", b"Sicily", b"Palermo", b"Catania"]),
            "$11\r\n166274.1516\r\n"
        );
        assert_eq!(
            f.run(&[b"GEODIST", b"Sicily", b"Palermo", b"Catania", b"km"]),
            "$8\r\n166.2742\r\n"
        );
        assert_eq!(
            f.run(&[b"GEODIST", b"Sicily", b"Palermo", b"Catania", b"mi"]),
            "$8\r\n103.3182\r\n"
        );
        // A member that is not there and a key that is not there are the same
        // nil, and the unit is read before the key is looked up, so a bad unit
        // on a missing key is still an error.
        assert_eq!(
            f.run(&[b"GEODIST", b"Sicily", b"Palermo", b"Foo"]),
            "$-1\r\n"
        );
        assert_eq!(f.run(&[b"GEODIST", b"nokey", b"a", b"b"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"GEODIST", b"nokey", b"a", b"b", b"parsecs"]),
            "-ERR unsupported unit provided. please use M, KM, FT, MI\r\n"
        );
        assert_eq!(
            f.run(&[b"GEODIST", b"Sicily", b"a", b"b", b"km", b"extra"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn a_search_finds_what_is_inside_it_nearest_first() {
        let mut f = Fixture::new();
        sicily(&mut f);
        let all = "*3\r\n$7\r\nCatania\r\n$9\r\nAgrigento\r\n$7\r\nPalermo\r\n";
        assert_eq!(
            f.run(&[
                b"GEOSEARCH",
                b"Sicily",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"km",
                b"ASC"
            ]),
            all
        );
        // The older spelling of the same search, which is the same nine boxes
        // and the same order.
        assert_eq!(
            f.run(&[b"GEORADIUS", b"Sicily", b"15", b"37", b"200", b"km", b"ASC"]),
            all
        );
        assert_eq!(
            f.run(&[
                b"GEORADIUS_RO",
                b"Sicily",
                b"15",
                b"37",
                b"200",
                b"km",
                b"ASC"
            ]),
            all
        );
        // A count with no ordering means the nearest ones, so DESC has to be
        // asked for to get the far end.
        assert_eq!(
            f.run(&[
                b"GEORADIUS",
                b"Sicily",
                b"15",
                b"37",
                b"200",
                b"km",
                b"DESC",
                b"COUNT",
                b"1"
            ]),
            "*1\r\n$7\r\nPalermo\r\n"
        );
        assert_eq!(
            f.run(&[
                b"GEORADIUS",
                b"Sicily",
                b"15",
                b"37",
                b"200",
                b"km",
                b"COUNT",
                b"1"
            ]),
            "*1\r\n$7\r\nCatania\r\n"
        );
        // Nothing inside a kilometre of that point, and nothing in a key that
        // is not there, and both are the empty array rather than an error.
        let empty = "*0\r\n";
        assert_eq!(
            f.run(&[
                b"GEOSEARCH",
                b"Sicily",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"1",
                b"km"
            ]),
            empty
        );
        assert_eq!(
            f.run(&[
                b"GEOSEARCH",
                b"nokey",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"1",
                b"km"
            ]),
            empty
        );
        assert_eq!(
            f.run(&[b"GEORADIUSBYMEMBER", b"nokey", b"m", b"1", b"km"]),
            empty
        );
    }

    #[test]
    fn a_search_centred_on_a_member_starts_from_where_that_member_is() {
        let mut f = Fixture::new();
        sicily(&mut f);
        assert_eq!(
            f.run(&[b"GEORADIUSBYMEMBER", b"Sicily", b"Agrigento", b"100", b"km"]),
            "*2\r\n$9\r\nAgrigento\r\n$7\r\nPalermo\r\n"
        );
        // The member itself is nothing away from itself, which is where the
        // fixed point writer's zero shows up on the wire.
        let with_dist = "*2\r\n*2\r\n$9\r\nAgrigento\r\n$6\r\n0.0000\r\n*2\r\n$7\r\nPalermo\r\n$7\r\n90.9778\r\n";
        assert_eq!(
            f.run(&[
                b"GEORADIUSBYMEMBER_RO",
                b"Sicily",
                b"Agrigento",
                b"100",
                b"km",
                b"WITHDIST"
            ]),
            with_dist
        );
        assert_eq!(
            f.run(&[
                b"GEOSEARCH",
                b"Sicily",
                b"FROMMEMBER",
                b"Agrigento",
                b"BYRADIUS",
                b"100",
                b"km",
                b"ASC",
                b"WITHDIST"
            ]),
            with_dist
        );
        assert_eq!(
            f.run(&[b"GEORADIUSBYMEMBER", b"Sicily", b"Nowhere", b"100", b"km"]),
            "-ERR could not decode requested zset member\r\n"
        );
    }

    #[test]
    fn a_box_search_reports_the_distance_the_hash_and_the_coordinates() {
        let mut f = Fixture::new();
        sicily(&mut f);
        // Three options asked for, so each result is a four element array of
        // the member, the distance, the hash and a pair. The order of the three
        // is Redis's and not the order they were written in the command.
        assert_eq!(
            f.run(&[
                b"GEOSEARCH",
                b"Sicily",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYBOX",
                b"400",
                b"400",
                b"km",
                b"ASC",
                b"WITHCOORD",
                b"WITHDIST",
                b"WITHHASH"
            ]),
            "*3\r\n*4\r\n$7\r\nCatania\r\n$7\r\n56.4413\r\n:3479447370796909\r\n*2\r\n\
             $18\r\n15.087267458438873\r\n$17\r\n37.50266842333162\r\n\
             *4\r\n$9\r\nAgrigento\r\n$8\r\n130.4235\r\n:3479030013248308\r\n*2\r\n\
             $18\r\n13.583331406116486\r\n$18\r\n37.316668049938166\r\n\
             *4\r\n$7\r\nPalermo\r\n$8\r\n190.4424\r\n:3479099956230698\r\n*2\r\n\
             $18\r\n13.361389338970184\r\n$16\r\n38.1155563954963\r\n"
        );
    }

    #[test]
    fn a_store_writes_the_hashes_and_a_storedist_writes_the_distances() {
        let mut f = Fixture::new();
        sicily(&mut f);
        let hashes = "*6\r\n$9\r\nAgrigento\r\n$16\r\n3479030013248308\r\n\
                      $7\r\nPalermo\r\n$16\r\n3479099956230698\r\n\
                      $7\r\nCatania\r\n$16\r\n3479447370796909\r\n";
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                b"dst",
                b"Sicily",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"km",
                b"ASC"
            ]),
            ":3\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"dst", b"0", b"-1", b"WITHSCORES"]),
            hashes
        );
        // The same again through the older spelling, which stores the same
        // scores, so a key written by either is a geo key.
        assert_eq!(
            f.run(&[
                b"GEORADIUS",
                b"Sicily",
                b"15",
                b"37",
                b"200",
                b"km",
                b"STORE",
                b"dst3"
            ]),
            ":3\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", b"dst3", b"0", b"-1", b"WITHSCORES"]),
            hashes
        );
        // STOREDIST stores the distance in the search unit instead, and those
        // are full doubles rather than the four places WITHDIST writes. The
        // numbers on the right are what 8.10.1 stored for this search, and they
        // are compared with a tolerance rather than byte for byte because the
        // last bit of a haversine is the platform's sin, cos and asin: this
        // machine and that one disagree in the sixteenth digit, and so do two
        // Redis builds. Everything a client actually reads back is four places
        // and is asserted exactly above.
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                b"dst2",
                b"Sicily",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"km",
                b"ASC",
                b"STOREDIST"
            ]),
            ":3\r\n"
        );
        for (member, want) in [
            ("Catania", 56.441_257_870_158_19),
            ("Agrigento", 130.423_487_067_147_14),
            ("Palermo", 190.442_429_847_757_92),
        ] {
            let reply = f.run(&[b"ZSCORE", b"dst2", member.as_bytes()]);
            let got: f64 = reply
                .trim_start_matches(|c: char| c != '\n')
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{member} scored {reply:?}"));
            assert!(
                (got - want).abs() < 1e-9,
                "{member} scored {got} not {want}"
            );
        }
        // The order they went in is the order the scores put them in, which is
        // the point of storing the distance rather than the hash.
        assert_eq!(
            f.run(&[b"ZRANGE", b"dst2", b"0", b"-1"]),
            "*3\r\n$7\r\nCatania\r\n$9\r\nAgrigento\r\n$7\r\nPalermo\r\n"
        );
        // A search that finds nothing takes the destination with it rather than
        // leaving what was there, and a source key that is not there is a
        // search that finds nothing.
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                b"dst",
                b"nokey",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"km"
            ]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"dst"]), ":0\r\n");
    }

    #[test]
    fn the_gates_on_geoadd_are_the_ones_zadd_has() {
        let mut f = Fixture::new();
        sicily(&mut f);
        // XX on a member that is already where it is changes nothing, and NX on
        // one that is there refuses to move it.
        assert_eq!(
            f.run(&[
                b"GEOADD",
                b"Sicily",
                b"XX",
                b"CH",
                b"13.361389",
                b"38.115556",
                b"Palermo"
            ]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[
                b"GEOADD",
                b"Sicily",
                b"NX",
                b"13.361389",
                b"38.9",
                b"Palermo"
            ]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[
                b"GEOADD",
                b"Sicily",
                b"CH",
                b"13.361389",
                b"38.9",
                b"Palermo"
            ]),
            ":1\r\n"
        );
        // Out of range, and nothing is stored: the whole call is refused rather
        // than the good pairs going in and the bad one stopping it.
        assert_eq!(
            f.run(&[
                b"GEOADD",
                b"new",
                b"13.361389",
                b"38.115556",
                b"here",
                b"181",
                b"38",
                b"there"
            ]),
            "-ERR invalid longitude,latitude pair 181.000000,38.000000\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"new"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"GEOADD", b"new", b"x", b"38", b"here"]),
            "-ERR value is not a valid float\r\n"
        );
        // The count of triples is checked before the two gates are, and a call
        // with no triples at all reaches the same sentence.
        assert_eq!(
            f.run(&[b"GEOADD", b"new", b"13", b"38", b"here", b"and"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"GEOADD", b"new", b"NX", b"XX", b"CH"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"GEOADD", b"new", b"CH", b"CH", b"CH", b"CH"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"GEOADD", b"new", b"NX", b"CH"]),
            "-ERR wrong number of arguments for 'geoadd' command\r\n"
        );
    }

    /// The sentences a search answers, which are its contract as much as the
    /// results are.
    #[test]
    fn every_way_a_search_can_be_written_wrong_has_its_own_sentence() {
        let mut f = Fixture::new();
        sicily(&mut f);
        let cases: &[(&[&[u8]], &str)] = &[
            (
                &[b"GEORADIUS", b"Sicily", b"15", b"37", b"x", b"km"],
                "-ERR need numeric radius\r\n",
            ),
            (
                &[b"GEORADIUS", b"Sicily", b"15", b"37", b"-1", b"km"],
                "-ERR radius cannot be negative\r\n",
            ),
            (
                &[b"GEORADIUS", b"Sicily", b"15", b"37", b"1", b"parsecs"],
                "-ERR unsupported unit provided. please use M, KM, FT, MI\r\n",
            ),
            (
                &[b"GEORADIUS", b"Sicily", b"181", b"37", b"1", b"km"],
                "-ERR invalid longitude,latitude pair 181.000000,37.000000\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYBOX",
                    b"x",
                    b"1",
                    b"km",
                ],
                "-ERR need numeric width\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYBOX",
                    b"1",
                    b"y",
                    b"km",
                ],
                "-ERR need numeric height\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYBOX",
                    b"-1",
                    b"1",
                    b"km",
                ],
                "-ERR height or width cannot be negative\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                    b"ANY",
                ],
                "-ERR the ANY argument requires COUNT argument\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                    b"COUNT",
                    b"0",
                ],
                "-ERR COUNT must be > 0\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                    b"BYBOX",
                    b"1",
                    b"1",
                    b"km",
                ],
                "-ERR syntax error\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMMEMBER",
                    b"Palermo",
                    b"FROMLONLAT",
                    b"1",
                    b"2",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                ],
                "-ERR syntax error\r\n",
            ),
            // The two options a GEOSEARCH cannot leave out, each with its own
            // sentence, and the command quoted the way the client spelled it.
            (
                &[
                    b"geosearch",
                    b"Sicily",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                    b"ASC",
                    b"WITHDIST",
                ],
                "-ERR exactly one of FROMMEMBER or FROMLONLAT can be specified for geosearch\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"ASC",
                    b"WITHDIST",
                ],
                "-ERR exactly one of BYRADIUS and BYBOX can be specified for GEOSEARCH\r\n",
            ),
            // A store cannot also be asked for the distance, and the two
            // families name themselves differently in the same sentence.
            (
                &[
                    b"GEOSEARCHSTORE",
                    b"d",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                    b"WITHCOORD",
                ],
                "-ERR GEOSEARCHSTORE is not compatible with WITHDIST, WITHHASH and WITHCOORD options\r\n",
            ),
            (
                &[
                    b"GEORADIUS",
                    b"Sicily",
                    b"15",
                    b"37",
                    b"1",
                    b"km",
                    b"WITHDIST",
                    b"STORE",
                    b"d",
                ],
                "-ERR STORE option in GEORADIUS is not compatible with WITHDIST, WITHHASH and WITHCOORD options\r\n",
            ),
            // The read only forms have no store at all, so the word is a stray
            // one, and GEOSEARCH's STOREDIST is only a GEOSEARCHSTORE option.
            (
                &[
                    b"GEORADIUS_RO",
                    b"Sicily",
                    b"15",
                    b"37",
                    b"1",
                    b"km",
                    b"STORE",
                    b"d",
                ],
                "-ERR syntax error\r\n",
            ),
            (
                &[
                    b"GEOSEARCH",
                    b"Sicily",
                    b"FROMLONLAT",
                    b"15",
                    b"37",
                    b"BYRADIUS",
                    b"1",
                    b"km",
                    b"STOREDIST",
                ],
                "-ERR syntax error\r\n",
            ),
        ];
        for (parts, want) in cases {
            assert_eq!(&f.run(parts), want, "{:?}", parts[0]);
        }
    }

    /// A wrong type wins over a bad argument, because the key is looked up
    /// first, and every one of the ten says the same thing about it.
    #[test]
    fn every_geo_command_says_wrongtype() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        let cases: &[&[&[u8]]] = &[
            &[b"GEOADD", b"s", b"13", b"38", b"m"],
            &[b"GEOPOS", b"s", b"m"],
            &[b"GEOHASH", b"s", b"m"],
            &[b"GEODIST", b"s", b"a", b"b"],
            &[
                b"GEOSEARCH",
                b"s",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"1",
                b"km",
            ],
            &[
                b"GEOSEARCHSTORE",
                b"d",
                b"s",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"1",
                b"km",
            ],
            &[b"GEORADIUS", b"s", b"15", b"37", b"1", b"km"],
            &[b"GEORADIUS_RO", b"s", b"15", b"37", b"1", b"km"],
            &[b"GEORADIUSBYMEMBER", b"s", b"m", b"1", b"km"],
            &[b"GEORADIUSBYMEMBER_RO", b"s", b"m", b"1", b"km"],
        ];
        for case in cases {
            assert_eq!(f.run(case), wrong, "{:?}", case[0]);
        }
        // And it wins over an argument that will not parse, which is the whole
        // reason the lookup comes first.
        assert_eq!(
            f.run(&[b"GEORADIUS", b"s", b"15", b"37", b"x", b"km"]),
            wrong
        );
    }

    // ----------------------------------------------------------------- array

    #[test]
    fn an_array_writes_at_any_index_and_reads_back_what_it_sent() {
        let mut f = Fixture::new();
        // Three consecutive positions from a high index, and the reply is how
        // many of them were empty before rather than how many were written.
        assert_eq!(
            f.run(&[b"ARSET", b"a", b"1000", b"x", b"y", b"z"]),
            ":3\r\n"
        );
        assert_eq!(f.run(&[b"ARSET", b"a", b"1000", b"X", b"Y"]), ":0\r\n");
        assert_eq!(f.run(&[b"ARGET", b"a", b"1000"]), "$1\r\nX\r\n");
        assert_eq!(f.run(&[b"ARGET", b"a", b"1002"]), "$1\r\nz\r\n");
        // A hole and a key that is not there are the same answer.
        assert_eq!(f.run(&[b"ARGET", b"a", b"999"]), "$-1\r\n");
        assert_eq!(f.run(&[b"ARGET", b"nope", b"0"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"ARMGET", b"a", b"1002", b"999", b"1000"]),
            "*3\r\n$1\r\nz\r\n$-1\r\n$1\r\nX\r\n"
        );
        // Scattered pairs in one command, last write wins within it.
        assert_eq!(f.run(&[b"ARMSET", b"a", b"5", b"p", b"5", b"q"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARGET", b"a", b"5"]), "$1\r\nq\r\n");
    }

    /// The two numbers an array reports are not the same number, and one of
    /// them does not fit a signed integer.
    #[test]
    fn the_length_is_the_high_water_mark_and_the_count_is_the_population() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ARLEN", b"nope"]), ":0\r\n");
        assert_eq!(f.run(&[b"ARCOUNT", b"nope"]), ":0\r\n");
        f.run(&[b"ARMSET", b"a", b"0", b"x", b"9", b"y"]);
        assert_eq!(f.run(&[b"ARLEN", b"a"]), ":10\r\n");
        assert_eq!(f.run(&[b"ARCOUNT", b"a"]), ":2\r\n");
        // Deleting in the middle leaves the high water mark where it was.
        assert_eq!(f.run(&[b"ARDEL", b"a", b"0"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARLEN", b"a"]), ":10\r\n");
        assert_eq!(f.run(&[b"ARCOUNT", b"a"]), ":1\r\n");

        // The top of the space is addressable, and its length is a number with
        // bit sixty three set, so the reply has to be unsigned or it comes back
        // negative.
        f.run(&[b"ARSET", b"top", b"18446744073709551614", b"z"]);
        assert_eq!(f.run(&[b"ARLEN", b"top"]), ":18446744073709551615\r\n");
        assert_eq!(f.run(&[b"ARCOUNT", b"top"]), ":1\r\n");
        // And one past it does not exist, so a write that would reach it fails
        // before any of it lands.
        assert_eq!(
            f.run(&[b"ARSET", b"over", b"18446744073709551614", b"a", b"b"]),
            "-ERR array index overflow\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"over"]), ":0\r\n");
    }

    /// One reply per position and not one per element, which is the whole
    /// reason the range is capped.
    #[test]
    fn a_range_read_answers_for_the_holes_too_and_is_capped_at_a_million() {
        let mut f = Fixture::new();
        f.run(&[b"ARSET", b"a", b"1", b"x"]);
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"a", b"0", b"3"]),
            "*4\r\n$-1\r\n$1\r\nx\r\n$-1\r\n$-1\r\n"
        );
        // The two ends may come in either order, and the answer is reversed
        // rather than empty.
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"a", b"3", b"0"]),
            "*4\r\n$-1\r\n$-1\r\n$1\r\nx\r\n$-1\r\n"
        );
        // A key that is not there reads like an array of nothing but holes.
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"nope", b"0", b"1"]),
            "*2\r\n$-1\r\n$-1\r\n"
        );
        // A range wider than a million positions is refused and not trimmed,
        // because against a missing key it is a request for as many nulls as
        // the range is wide.
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"nope", b"0", b"18446744073709551614"]),
            "-ERR range exceeds maximum of 1000000 items\r\n"
        );
    }

    /// Every index in the argument list is read before the key is touched, so
    /// a bad one at the end leaves nothing half written.
    #[test]
    fn a_bad_index_late_in_the_line_writes_none_of_the_earlier_ones() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"ARMSET", b"a", b"0", b"x", b"-1", b"y"]),
            "-ERR invalid array index\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"a"]), ":0\r\n");
        f.run(&[b"ARSET", b"a", b"0", b"x", b"y", b"z"]);
        assert_eq!(
            f.run(&[b"ARDEL", b"a", b"0", b"01"]),
            "-ERR invalid array index\r\n"
        );
        assert_eq!(f.run(&[b"ARCOUNT", b"a"]), ":3\r\n");
        // An index is unsigned here, so the numbers a list would take are not
        // the last element, they are errors.
        assert_eq!(
            f.run(&[b"ARGET", b"a", b"-1"]),
            "-ERR invalid array index\r\n"
        );
        // And a pair list with an odd tail is an arity error rather than a
        // syntax one.
        assert_eq!(
            f.run(&[b"ARMSET", b"a", b"0", b"x", b"1"]),
            "-ERR wrong number of arguments for 'armset' command\r\n"
        );
        assert_eq!(
            f.run(&[b"ARDELRANGE", b"a", b"0", b"1", b"2"]),
            "-ERR wrong number of arguments for 'ardelrange' command\r\n"
        );
    }

    #[test]
    fn a_range_delete_costs_the_elements_and_takes_the_key_when_it_empties() {
        let mut f = Fixture::new();
        f.run(&[b"ARSET", b"a", b"0", b"0", b"1", b"2", b"3", b"4"]);
        assert_eq!(f.run(&[b"ARDELRANGE", b"a", b"3", b"1"]), ":3\r\n");
        assert_eq!(f.run(&[b"ARCOUNT", b"a"]), ":2\r\n");
        // Two ranges in one command, and the second one covers the whole space
        // without walking it.
        assert_eq!(
            f.run(&[
                b"ARDELRANGE",
                b"a",
                b"100",
                b"200",
                b"0",
                b"18446744073709551614"
            ]),
            ":2\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"a"]), ":0\r\n");
        assert_eq!(f.run(&[b"ARDELRANGE", b"nope", b"0", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"ARDEL", b"nope", b"0"]), ":0\r\n");
    }

    /// A value goes out as the bytes it came in as, whichever of the three ways
    /// the array found to store it.
    #[test]
    fn a_value_comes_back_byte_for_byte_however_it_was_packed() {
        let mut f = Fixture::new();
        let long = vec![b'v'; 200];
        f.run(&[
            b"ARMSET", b"a", b"0", b"42", b"1", b"007", b"2", b"3.5", b"3", b"3.14", b"4",
            b"short", b"5", &long, b"6", b"-0",
        ]);
        // 42 is an integer, 007 is not one because it does not print back the
        // same, 3.5 survives a double and 3.14 does not, and the last two are a
        // word packed string and a blob.
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"a", b"0", b"6"]),
            format!(
                "*7\r\n$2\r\n42\r\n$3\r\n007\r\n$3\r\n3.5\r\n$4\r\n3.14\r\n$5\r\nshort\r\n$200\r\n{}\r\n$2\r\n-0\r\n",
                String::from_utf8_lossy(&long)
            )
        );
    }

    #[test]
    fn an_array_is_a_type_and_an_encoding_a_client_can_see() {
        let mut f = Fixture::new();
        f.run(&[b"ARSET", b"a", b"0", b"x"]);
        assert_eq!(f.run(&[b"TYPE", b"a"]), "+array\r\n");
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"a"]),
            "$12\r\nsliced-array\r\n"
        );
        // And it is a body like any other, so the key commands work on it.
        assert_eq!(f.run(&[b"EXPIRE", b"a", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"PERSIST", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"COPY", b"a", b"b"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARGET", b"b", b"0"]), "$1\r\nx\r\n");
        assert_eq!(f.run(&[b"RENAME", b"a", b"c"]), "+OK\r\n");
        assert_eq!(f.run(&[b"ARCOUNT", b"c"]), ":1\r\n");
    }

    #[test]
    fn every_array_command_refuses_a_key_holding_something_else() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        for cmd in [
            &[b"ARSET".as_ref(), b"s", b"0", b"x"][..],
            &[b"ARMSET".as_ref(), b"s", b"0", b"x"][..],
            &[b"ARGET".as_ref(), b"s", b"0"][..],
            &[b"ARMGET".as_ref(), b"s", b"0"][..],
            &[b"ARGETRANGE".as_ref(), b"s", b"0", b"1"][..],
            &[b"ARLEN".as_ref(), b"s"][..],
            &[b"ARCOUNT".as_ref(), b"s"][..],
            &[b"ARDEL".as_ref(), b"s", b"0"][..],
            &[b"ARDELRANGE".as_ref(), b"s", b"0", b"1"][..],
            &[b"ARINSERT".as_ref(), b"s", b"x"][..],
            &[b"ARRING".as_ref(), b"s", b"4", b"x"][..],
            &[b"ARNEXT".as_ref(), b"s"][..],
            &[b"ARSEEK".as_ref(), b"s", b"1"][..],
            &[b"ARLASTITEMS".as_ref(), b"s", b"1"][..],
            &[b"ARSCAN".as_ref(), b"s", b"0", b"1"][..],
            &[b"ARGREP".as_ref(), b"s", b"0", b"1", b"EXACT", b"v"][..],
            &[b"AROP".as_ref(), b"s", b"0", b"1", b"SUM"][..],
            &[b"ARINFO".as_ref(), b"s"][..],
        ] {
            assert_eq!(f.run(cmd), wrong, "{}", String::from_utf8_lossy(cmd[0]));
        }
    }

    /// Two of the array commands look the key up before they read the index and
    /// the rest read the index first, so the same broken argument gets two
    /// different errors depending on which command it went to.
    #[test]
    fn a_bad_index_reports_the_type_only_where_redis_reports_it() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"v"]);
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        let bad = "-ERR invalid array index\r\n";
        assert_eq!(f.run(&[b"ARGET", b"s", b"-1"]), wrong);
        assert_eq!(f.run(&[b"ARMGET", b"s", b"0", b"-1"]), wrong);
        assert_eq!(f.run(&[b"ARSET", b"s", b"-1", b"x"]), bad);
        assert_eq!(f.run(&[b"ARDEL", b"s", b"-1"]), bad);
        assert_eq!(f.run(&[b"ARSCAN", b"s", b"-1", b"0"]), bad);
        assert_eq!(f.run(&[b"ARGREP", b"s", b"-1", b"0", b"EXACT", b"v"]), bad);
        // And on a key that is an array the index is just an index.
        f.run(&[b"ARSET", b"a", b"0", b"x"]);
        assert_eq!(f.run(&[b"ARGET", b"a", b"-1"]), bad);
        assert_eq!(f.run(&[b"ARGET", b"nope", b"-1"]), bad);
    }

    #[test]
    fn an_append_follows_a_cursor_the_client_can_move() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ARNEXT", b"nope"]), ":0\r\n");
        assert_eq!(f.run(&[b"ARINSERT", b"a", b"x", b"y"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARNEXT", b"a"]), ":2\r\n");
        assert_eq!(f.run(&[b"ARINSERT", b"a", b"z"]), ":2\r\n");
        assert_eq!(f.run(&[b"ARGET", b"a", b"2"]), "$1\r\nz\r\n");

        // A seek says where the next one goes, and a missing key has no cursor
        // to move and is not created by the asking.
        assert_eq!(f.run(&[b"ARSEEK", b"nope", b"5"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"nope"]), ":0\r\n");
        assert_eq!(f.run(&[b"ARSEEK", b"a", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARNEXT", b"a"]), ":100\r\n");
        assert_eq!(f.run(&[b"ARINSERT", b"a", b"far"]), ":100\r\n");
        assert_eq!(f.run(&[b"ARSEEK", b"a", b"0"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARNEXT", b"a"]), ":0\r\n");

        // The top of the space is the one index only ARSEEK will take, and it
        // leaves the cursor with nowhere to go.
        assert_eq!(f.run(&[b"ARSEEK", b"a", b"18446744073709551615"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARNEXT", b"a"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"ARINSERT", b"a", b"x"]),
            "-ERR insert index overflow\r\n"
        );
        assert_eq!(
            f.run(&[b"ARSET", b"a", b"18446744073709551615", b"x"]),
            "-ERR invalid array index\r\n"
        );
    }

    #[test]
    fn a_ring_keeps_the_newest_and_renumbers_them_when_it_is_resized() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ARRING", b"r", b"3", b"a", b"b", b"c"]), ":2\r\n");
        assert_eq!(f.run(&[b"ARRING", b"r", b"3", b"d", b"e"]), ":1\r\n");
        assert_eq!(f.run(&[b"ARLEN", b"r"]), ":3\r\n");
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"r", b"0", b"2"]),
            "*3\r\n$1\r\nd\r\n$1\r\ne\r\n$1\r\nc\r\n"
        );
        // Growing it after it has wrapped puts the survivors back in the order
        // they arrived, which is the whole point of paying for the rebuild.
        assert_eq!(f.run(&[b"ARRING", b"r", b"5", b"f"]), ":3\r\n");
        assert_eq!(
            f.run(&[b"ARGETRANGE", b"r", b"0", b"3"]),
            "*4\r\n$1\r\nc\r\n$1\r\nd\r\n$1\r\ne\r\n$1\r\nf\r\n"
        );
        // The size is read before the key, so a bad one is a bad size wherever
        // it is sent.
        assert_eq!(
            f.run(&[b"ARRING", b"r", b"0", b"x"]),
            "-ERR size must be positive\r\n"
        );
        assert_eq!(
            f.run(&[b"ARRING", b"r", b"big", b"x"]),
            "-ERR invalid size\r\n"
        );
    }

    #[test]
    fn the_last_items_walk_back_from_the_cursor_and_report_the_holes() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ARLASTITEMS", b"nope", b"5"]), "*0\r\n");
        f.run(&[b"ARRING", b"r", b"4", b"a", b"b", b"c", b"d", b"e"]);
        assert_eq!(
            f.run(&[b"ARLASTITEMS", b"r", b"3"]),
            "*3\r\n$1\r\nc\r\n$1\r\nd\r\n$1\r\ne\r\n"
        );
        assert_eq!(
            f.run(&[b"ARLASTITEMS", b"r", b"3", b"rev"]),
            "*3\r\n$1\r\ne\r\n$1\r\nd\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"ARLASTITEMS", b"r", b"99"]),
            "*4\r\n$1\r\nb\r\n$1\r\nc\r\n$1\r\nd\r\n$1\r\ne\r\n",
            "more than there is gets what there is"
        );
        // Nothing asked for is an empty reply, and Redis answers that before it
        // has read the option or looked at the key.
        assert_eq!(f.run(&[b"ARLASTITEMS", b"r", b"0", b"junk"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"ARLASTITEMS", b"r", b"1", b"junk"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ARLASTITEMS", b"r", b"nine"]),
            "-ERR invalid COUNT\r\n"
        );

        // With no cursor the tail of the array is the anchor, and a hole inside
        // the window is reported as one.
        f.run(&[b"ARMSET", b"h", b"0", b"x", b"2", b"z"]);
        assert_eq!(
            f.run(&[b"ARLASTITEMS", b"h", b"5"]),
            "*2\r\n$-1\r\n$1\r\nz\r\n"
        );
    }

    #[test]
    fn a_scan_answers_pairs_for_what_is_there_and_skips_what_is_not() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ARSCAN", b"nope", b"0", b"10"]), "*0\r\n");
        f.run(&[b"ARMSET", b"a", b"0", b"x", b"7", b"y", b"1000000", b"z"]);
        // The whole index space, which ARGETRANGE refuses and this one answers
        // in three visits because holes cost nothing.
        assert_eq!(
            f.run(&[b"ARSCAN", b"a", b"0", b"18446744073709551614"]),
            "*3\r\n*2\r\n:0\r\n$1\r\nx\r\n*2\r\n:7\r\n$1\r\ny\r\n*2\r\n:1000000\r\n$1\r\nz\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ARSCAN",
                b"a",
                b"18446744073709551614",
                b"0",
                b"LIMIT",
                b"1"
            ]),
            "*1\r\n*2\r\n:1000000\r\n$1\r\nz\r\n"
        );
        assert_eq!(f.run(&[b"ARSCAN", b"a", b"1", b"6"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"ARSCAN", b"a", b"0", b"10", b"LIMIT", b"0"]),
            "-ERR LIMIT must be positive\r\n"
        );
        assert_eq!(
            f.run(&[b"ARSCAN", b"a", b"0", b"10", b"NOPE", b"1"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"ARSCAN", b"a", b"0", b"10", b"LIMIT"]),
            "-ERR wrong number of arguments for 'arscan' command\r\n"
        );
    }

    #[test]
    fn a_grep_answers_the_indexes_whose_elements_match() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"ARGREP", b"nope", b"0", b"10", b"EXACT", b"x"]),
            "*0\r\n"
        );
        f.run(&[b"ARSET", b"a", b"0", b"alpha", b"beta", b"gamma", b"ALPHA"]);

        // The two bounds take the ends of the array as well as an index, and a
        // reversed range is walked backwards the way ARSCAN walks one.
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"-", b"+", b"GLOB", b"*a"]),
            "*3\r\n:0\r\n:1\r\n:2\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"+", b"-", b"GLOB", b"*a"]),
            "*3\r\n:2\r\n:1\r\n:0\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"1", b"2", b"GLOB", b"*a"]),
            "*2\r\n:1\r\n:2\r\n"
        );

        // One test each. NOCASE reaches all four of them and it may be written
        // after the pattern it applies to.
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"-", b"+", b"EXACT", b"alpha"]),
            "*1\r\n:0\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"-", b"+", b"EXACT", b"alpha", b"NOCASE"]),
            "*2\r\n:0\r\n:3\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"-", b"+", b"MATCH", b"mm"]),
            "*1\r\n:2\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"-", b"+", b"RE", b"^[bg]"]),
            "*2\r\n:1\r\n:2\r\n"
        );

        // OR is the default and AND has to be asked for, and either way the
        // last of a repeated option wins.
        let both: &[&[u8]] = &[
            b"ARGREP", b"a", b"-", b"+", b"EXACT", b"beta", b"MATCH", b"al",
        ];
        assert_eq!(f.run(both), "*2\r\n:0\r\n:1\r\n");
        assert_eq!(
            f.run(&[
                b"ARGREP", b"a", b"-", b"+", b"EXACT", b"beta", b"MATCH", b"al", b"AND"
            ]),
            "*0\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ARGREP", b"a", b"-", b"+", b"EXACT", b"beta", b"MATCH", b"al", b"AND", b"OR"
            ]),
            "*2\r\n:0\r\n:1\r\n"
        );

        // WITHVALUES turns each hit into a pair, and LIMIT counts the hits and
        // not the positions it had to look at.
        assert_eq!(
            f.run(&[
                b"ARGREP",
                b"a",
                b"-",
                b"+",
                b"MATCH",
                b"a",
                b"WITHVALUES",
                b"LIMIT",
                b"2"
            ]),
            "*2\r\n*2\r\n:0\r\n$5\r\nalpha\r\n*2\r\n:1\r\n$4\r\nbeta\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ARGREP", b"a", b"-", b"+", b"EXACT", b"ALPHA", b"LIMIT", b"1"
            ]),
            "*1\r\n:3\r\n"
        );
    }

    /// Everything ARGREP refuses, in the order it refuses it.
    #[test]
    fn a_grep_reports_a_broken_command_the_way_redis_does() {
        let mut f = Fixture::new();
        f.run(&[b"ARSET", b"a", b"0", b"alpha"]);
        let syntax = "-ERR syntax error\r\n";

        // The bounds are read before the plan, so a bad index beats a bad
        // predicate whichever way round the two are written.
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"-1", b"0", b"NOPE", b"x"]),
            "-ERR invalid array index\r\n"
        );
        assert_eq!(f.run(&[b"ARGREP", b"a", b"0", b"1", b"NOPE", b"x"]), syntax);
        // A keyword with nothing after it, and a command that asks for nothing.
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"NOCASE", b"EXACT"]),
            syntax
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"EXACT", b"x", b"LIMIT"]),
            syntax
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"NOCASE", b"WITHVALUES"]),
            syntax,
            "a command with no predicate in it at all"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"EXACT", b"x", b"LIMIT", b"0"]),
            "-ERR LIMIT must be positive\r\n"
        );
        assert_eq!(
            f.run(&[
                b"ARGREP", b"a", b"0", b"1", b"EXACT", b"x", b"LIMIT", b"nine"
            ]),
            "-ERR value is not an integer or out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"RE", b""]),
            "-ERR regular expression is empty\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"RE", b"(a"]),
            "-ERR invalid regular expression: Missing ')'\r\n"
        );
        assert_eq!(
            f.run(&[b"ARGREP", b"a", b"0", b"1", b"RE", br"(a)\1"]),
            "-ERR regular expression backreferences are not supported\r\n"
        );
        // The arity is minus six, so a predicate keyword with no pattern after
        // it is short by one and never reaches the parser.
        let arity = "-ERR wrong number of arguments for 'argrep' command\r\n";
        assert_eq!(f.run(&[b"ARGREP", b"a", b"0", b"1", b"EXACT"]), arity);
        assert_eq!(f.run(&[b"ARGREP", b"a", b"0", b"1"]), arity);
    }

    #[test]
    fn an_op_reduces_a_range_to_one_number() {
        let mut f = Fixture::new();
        f.run(&[b"ARSET", b"a", b"0", b"1", b"2.5", b"word", b"-4"]);
        assert_eq!(
            f.run(&[b"AROP", b"a", b"0", b"10", b"SUM"]),
            "$4\r\n-0.5\r\n"
        );
        assert_eq!(f.run(&[b"AROP", b"a", b"0", b"10", b"min"]), "$2\r\n-4\r\n");
        assert_eq!(
            f.run(&[b"AROP", b"a", b"0", b"10", b"MAX"]),
            "$3\r\n2.5\r\n"
        );
        assert_eq!(f.run(&[b"AROP", b"a", b"0", b"10", b"USED"]), ":4\r\n");
        assert_eq!(
            f.run(&[b"AROP", b"a", b"0", b"10", b"MATCH", b"word"]),
            ":1\r\n"
        );
        // An aggregate is written with seventeen significant digits, which is
        // Redis's own choice and not what a score comes back as.
        f.run(&[b"ARSET", b"t", b"0", b"0.1", b"0.2"]);
        assert_eq!(
            f.run(&[b"AROP", b"t", b"0", b"10", b"SUM"]),
            "$19\r\n0.30000000000000004\r\n"
        );
        assert_eq!(f.run(&[b"ZADD", b"z", b"0.3", b"m"]), ":1\r\n");
        assert_eq!(f.run(&[b"ZSCORE", b"z", b"m"]), "$3\r\n0.3\r\n");

        // Nothing to work with is a null, and a missing key is a null for the
        // aggregates and a zero for the two that count.
        f.run(&[b"ARSET", b"w", b"0", b"word"]);
        assert_eq!(f.run(&[b"AROP", b"w", b"0", b"10", b"SUM"]), "$-1\r\n");
        assert_eq!(f.run(&[b"AROP", b"nope", b"0", b"10", b"SUM"]), "$-1\r\n");
        assert_eq!(f.run(&[b"AROP", b"nope", b"0", b"10", b"USED"]), ":0\r\n");

        assert_eq!(
            f.run(&[b"AROP", b"a", b"0", b"10", b"NOPE"]),
            "-ERR unknown operation\r\n"
        );
        assert_eq!(
            f.run(&[b"AROP", b"a", b"0", b"10", b"MATCH"]),
            "-ERR MATCH requires a value argument\r\n"
        );
        assert_eq!(
            f.run(&[b"AROP", b"a", b"0", b"10", b"SUM", b"extra"]),
            "-ERR wrong number of arguments for 'arop' command\r\n"
        );
    }

    #[test]
    fn the_info_is_a_map_and_a_missing_key_is_an_error() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"ARINFO", b"nope"]), "-ERR no such key\r\n");
        f.run(&[b"ARINSERT", b"a", b"x", b"y"]);
        let short = f.run(&[b"ARINFO", b"a"]);
        assert!(
            short.starts_with("*14\r\n"),
            "seven pairs on RESP2: {short}"
        );
        assert!(short.contains("$5\r\ncount\r\n:2\r\n"), "{short}");
        assert!(
            short.contains("$17\r\nnext-insert-index\r\n:2\r\n"),
            "{short}"
        );
        assert!(short.contains("$10\r\nslice-size\r\n:4096\r\n"), "{short}");
        let full = f.run(&[b"ARINFO", b"a", b"full"]);
        assert!(full.starts_with("*24\r\n"), "twelve pairs: {full}");
        // Two values one apart are held sparsely, so the dense count is zero and
        // the two dense averages have nothing to average.
        assert!(full.contains("$12\r\ndense-slices\r\n:0\r\n"), "{full}");
        assert!(full.contains("$13\r\nsparse-slices\r\n:1\r\n"), "{full}");
        assert!(
            full.contains("$14\r\navg-dense-size\r\n$1\r\n0\r\n"),
            "{full}"
        );
        assert_eq!(f.run(&[b"ARINFO", b"a", b"nope"]), "-ERR syntax error\r\n");

        // On RESP3 the same reply is a map and the averages are doubles.
        let mut g = Fixture::new();
        g.run(&[b"HELLO", b"3"]);
        g.run(&[b"ARINSERT", b"a", b"x"]);
        let map = g.run(&[b"ARINFO", b"a", b"FULL"]);
        assert!(map.starts_with("%12\r\n"), "{map}");
        assert!(map.contains("$5\r\ncount\r\n:1\r\n"), "{map}");
        assert!(map.contains("$14\r\navg-dense-size\r\n,0\r\n"), "{map}");
    }

    #[test]
    fn a_double_on_the_wire_is_written_the_way_redis_writes_one() {
        let mut f = Fixture::new();
        // Whole numbers up to two to the sixty second come back as integers,
        // and past that the digit generator takes over and uses an exponent.
        for (score, want) in [
            ("3", "3"),
            ("3.5", "3.5"),
            ("0.3", "0.3"),
            ("1e30", "1e+30"),
            ("1e19", "1e+19"),
            ("1e-7", "1e-7"),
            ("0.000001", "0.000001"),
            ("4611686018427387904", "4611686018427387904"),
            ("-0", "-0"),
        ] {
            f.run(&[b"ZADD", b"z", score.as_bytes(), b"m"]);
            assert_eq!(
                f.run(&[b"ZSCORE", b"z", b"m"]),
                format!("${}\r\n{want}\r\n", want.len()),
                "score {score}"
            );
        }

        // The same bytes on RESP3, where the reply is a double rather than a
        // bulk string.
        let mut g = Fixture::new();
        g.run(&[b"HELLO", b"3"]);
        g.run(&[b"ZADD", b"z", b"1e30", b"m"]);
        assert_eq!(g.run(&[b"ZSCORE", b"z", b"m"]), ",1e+30\r\n");
        // The two float increments are not this printer. They go through
        // ld2string in its human mode, which is a fixed point conversion with
        // the trailing zeros taken off, so they never write an exponent, and
        // they reply with a bulk string on both protocols.
        assert_eq!(
            g.run(&[b"INCRBYFLOAT", b"s", b"1e30"]),
            "$31\r\n1000000000000000000000000000000\r\n"
        );
        assert_eq!(g.run(&[b"INCRBYFLOAT", b"t", b"0.1"]), "$3\r\n0.1\r\n");
        assert_eq!(
            g.run(&[b"HINCRBYFLOAT", b"h", b"f", b"1e19"]),
            "$20\r\n10000000000000000000\r\n"
        );
    }

    // ----------------------------------------------------------------- graph

    #[test]
    fn a_node_comes_back_with_the_fields_it_went_in_with() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"G.NADD", b"social", b"ada", b"name", b"Ada", b"born", b"1815"
            ]),
            ":1\r\n"
        );
        // The year comes back as the four bytes that were sent and not as a
        // number, because every property is text and there is nothing on the
        // wire that says which of `1815` and `"1815"` the client meant. The
        // fields are in the document's order, which is sorted by name, because
        // that is what makes a field lookup a binary search.
        assert_eq!(
            f.run(&[b"G.NGET", b"social", b"ada"]),
            "*4\r\n$4\r\nborn\r\n$4\r\n1815\r\n$4\r\nname\r\n$3\r\nAda\r\n"
        );
        // A second write to the same id replaces the document and says so with
        // a zero, so an ingest can count what it created.
        assert_eq!(
            f.run(&[b"G.NADD", b"social", b"ada", b"name", b"Ada Lovelace"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"G.NGET", b"social", b"ada"]),
            "*2\r\n$4\r\nname\r\n$12\r\nAda Lovelace\r\n"
        );
        // A node with no properties is an empty map and not a null, which is
        // how a client tells an isolated node from one that is not there.
        assert_eq!(f.run(&[b"G.NADD", b"social", b"grace"]), ":1\r\n");
        assert_eq!(f.run(&[b"G.NGET", b"social", b"grace"]), "*0\r\n");
        assert_eq!(f.run(&[b"G.NGET", b"social", b"nobody"]), "$-1\r\n");
        assert_eq!(f.run(&[b"G.NGET", b"nokey", b"ada"]), "$-1\r\n");

        // A field with no value creates nothing, because the pairs are checked
        // before the key is touched.
        assert_eq!(
            f.run(&[b"G.NADD", b"fresh", b"n", b"lonely"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"fresh"]), ":0\r\n");

        // On RESP3 the same reply is a map.
        let mut g = Fixture::new();
        g.run(&[b"HELLO", b"3"]);
        g.run(&[b"G.NADD", b"social", b"ada", b"name", b"Ada"]);
        assert_eq!(
            g.run(&[b"G.NGET", b"social", b"ada"]),
            "%1\r\n$4\r\nname\r\n$3\r\nAda\r\n"
        );
    }

    #[test]
    fn an_edge_creates_the_ends_it_needs() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"G.EADD", b"social", b"ada", b"grace", b"FOLLOWS", b"since", b"1843"
            ]),
            ":1\r\n"
        );
        // Neither end was written first and both are there, as empty nodes.
        assert_eq!(f.run(&[b"G.NGET", b"social", b"ada"]), "*0\r\n");
        assert_eq!(f.run(&[b"G.NGET", b"social", b"grace"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"G.OUT", b"social", b"ada", b"FOLLOWS"]),
            "*2\r\n$1\r\n0\r\n*1\r\n$5\r\ngrace\r\n"
        );
        assert_eq!(
            f.run(&[b"G.IN", b"social", b"grace", b"FOLLOWS"]),
            "*2\r\n$1\r\n0\r\n*1\r\n$3\r\nada\r\n"
        );
        // The same pair under the same label again updates the edge rather than
        // making a second one.
        assert_eq!(
            f.run(&[
                b"G.EADD", b"social", b"ada", b"grace", b"FOLLOWS", b"since", b"1844"
            ]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"G.DEG", b"social", b"ada", b"FOLLOWS"]), ":1\r\n");
        // A different label between the same pair is a different edge.
        assert_eq!(
            f.run(&[b"G.EADD", b"social", b"ada", b"grace", b"WORKS_WITH"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"G.DEG", b"social", b"ada", b"WORKS_WITH"]),
            ":1\r\n"
        );

        assert_eq!(
            f.run(&[b"G.EDEL", b"social", b"ada", b"grace", b"FOLLOWS"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"G.EDEL", b"social", b"ada", b"grace", b"FOLLOWS"]),
            ":0\r\n"
        );
        // A label nothing has used, an end that is not there, and a key that is
        // not there are all a zero rather than an error.
        assert_eq!(
            f.run(&[b"G.EDEL", b"social", b"ada", b"grace", b"NEVER"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"G.EDEL", b"social", b"ada", b"nobody", b"FOLLOWS"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"G.EDEL", b"nokey", b"ada", b"grace", b"FOLLOWS"]),
            ":0\r\n"
        );
    }

    /// A run is paged the way `SCAN` is paged, so a client that can walk one
    /// can walk the other.
    #[test]
    fn a_hop_answers_a_cursor_and_a_page() {
        let mut f = Fixture::new();
        for i in 0..25u32 {
            let dst = format!("n{i}");
            f.run(&[b"G.EADD", b"social", b"hub", dst.as_bytes(), b"FOLLOWS"]);
        }
        // Ten without being asked, and the cursor is where to carry on from.
        let first = f.run(&[b"G.OUT", b"social", b"hub", b"FOLLOWS"]);
        assert!(first.starts_with("*2\r\n$2\r\n10\r\n*10\r\n"), "{first}");

        let mut seen = 0;
        let mut cursor = String::from("0");
        loop {
            let page = f.run(&[
                b"G.OUT",
                b"social",
                b"hub",
                b"FOLLOWS",
                b"COUNT",
                b"7",
                b"CURSOR",
                cursor.as_bytes(),
            ]);
            let (head, rest) = page.split_once("\r\n*").expect("a cursor and a page");
            cursor = head
                .rsplit("\r\n")
                .next()
                .expect("the cursor line")
                .to_string();
            seen += rest
                .split_once("\r\n")
                .expect("the page length")
                .0
                .parse::<usize>()
                .expect("a length");
            if cursor == "0" {
                break;
            }
        }
        assert_eq!(seen, 25, "every neighbour once across the pages");

        // A cursor past the end is an empty page and not an error, and so is a
        // key or a label that is not there.
        assert_eq!(
            f.run(&[b"G.OUT", b"social", b"hub", b"FOLLOWS", b"CURSOR", b"900"]),
            "*2\r\n$1\r\n0\r\n*0\r\n"
        );
        assert_eq!(
            f.run(&[b"G.OUT", b"social", b"hub", b"NEVER"]),
            "*2\r\n$1\r\n0\r\n*0\r\n"
        );
        assert_eq!(
            f.run(&[b"G.OUT", b"nokey", b"hub", b"FOLLOWS"]),
            "*2\r\n$1\r\n0\r\n*0\r\n"
        );
        assert_eq!(
            f.run(&[b"G.OUT", b"social", b"hub", b"FOLLOWS", b"COUNT", b"0"]),
            "-ERR COUNT must be a positive integer\r\n"
        );
        assert_eq!(
            f.run(&[b"G.OUT", b"social", b"hub", b"FOLLOWS", b"NOPE", b"1"]),
            "-ERR syntax error\r\n"
        );
    }

    #[test]
    fn a_degree_counts_one_way_or_both() {
        let mut f = Fixture::new();
        f.run(&[b"G.EADD", b"social", b"a", b"b", b"F"]);
        f.run(&[b"G.EADD", b"social", b"a", b"c", b"F"]);
        f.run(&[b"G.EADD", b"social", b"d", b"a", b"F"]);
        assert_eq!(f.run(&[b"G.DEG", b"social", b"a", b"F"]), ":2\r\n");
        assert_eq!(f.run(&[b"G.DEG", b"social", b"a", b"F", b"OUT"]), ":2\r\n");
        assert_eq!(f.run(&[b"G.DEG", b"social", b"a", b"F", b"IN"]), ":1\r\n");
        assert_eq!(f.run(&[b"G.DEG", b"social", b"a", b"F", b"BOTH"]), ":3\r\n");
        assert_eq!(f.run(&[b"G.DEG", b"social", b"nobody", b"F"]), ":0\r\n");
        assert_eq!(f.run(&[b"G.DEG", b"social", b"a", b"NEVER"]), ":0\r\n");
        assert_eq!(f.run(&[b"G.DEG", b"nokey", b"a", b"F"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"G.DEG", b"social", b"a", b"F", b"SIDEWAYS"]),
            "-ERR syntax error\r\n"
        );
    }

    /// A walk answers which nodes it can reach and not by how many routes, so a
    /// node two ways out is in the frontier once.
    #[test]
    fn a_walk_reaches_each_node_once_however_many_ways_there_are() {
        let mut f = Fixture::new();
        for (src, dst) in [
            ("ada", "grace"),
            ("ada", "alan"),
            ("grace", "edsger"),
            ("alan", "edsger"),
            ("edsger", "barbara"),
        ] {
            f.run(&[b"G.EADD", b"social", src.as_bytes(), dst.as_bytes(), b"F"]);
        }
        // Two hops without being asked, the start left out, and edsger once
        // even though both of the first hop's nodes point at it.
        assert_eq!(
            f.run(&[b"G.NEIGH", b"social", b"ada", b"F"]),
            "*3\r\n$5\r\ngrace\r\n$4\r\nalan\r\n$6\r\nedsger\r\n"
        );
        assert_eq!(
            f.run(&[b"G.NEIGH", b"social", b"ada", b"F", b"DEPTH", b"1"]),
            "*2\r\n$5\r\ngrace\r\n$4\r\nalan\r\n"
        );
        let deep = f.run(&[b"G.NEIGH", b"social", b"ada", b"F", b"DEPTH", b"9"]);
        assert!(deep.starts_with("*4\r\n"), "the whole component: {deep}");
        assert!(deep.contains("$7\r\nbarbara\r\n"), "{deep}");
        // COUNT stops the walk rather than trimming what it found.
        assert_eq!(
            f.run(&[b"G.NEIGH", b"social", b"ada", b"F", b"COUNT", b"1"]),
            "*1\r\n$5\r\ngrace\r\n"
        );
        // A node nothing leaves is an empty array and not an error.
        assert_eq!(f.run(&[b"G.NEIGH", b"social", b"barbara", b"F"]), "*0\r\n");
        assert_eq!(f.run(&[b"G.NEIGH", b"social", b"ada", b"NEVER"]), "*0\r\n");
        assert_eq!(f.run(&[b"G.NEIGH", b"nokey", b"ada", b"F"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"G.NEIGH", b"social", b"ada", b"F", b"DEPTH", b"0"]),
            "-ERR DEPTH must be a positive integer\r\n"
        );
        assert_eq!(
            f.run(&[b"G.NEIGH", b"social", b"ada", b"F", b"NOPE", b"1"]),
            "-ERR syntax error\r\n"
        );
    }

    /// The two sided search, which is the whole reason `G.PATH` is a command
    /// and not something a client builds out of `G.OUT`.
    #[test]
    fn a_path_is_the_shortest_one_and_goes_over_any_label() {
        let mut f = Fixture::new();
        // A chain of six, and a shortcut that makes a shorter way round under a
        // second label so the search has to take either kind of hop.
        for i in 0..6u32 {
            let src = format!("n{i}");
            let dst = format!("n{}", i + 1);
            f.run(&[b"G.EADD", b"road", src.as_bytes(), dst.as_bytes(), b"STEP"]);
        }
        assert_eq!(
            f.run(&[b"G.PATH", b"road", b"n0", b"n6"]),
            "*7\r\n$2\r\nn0\r\n$2\r\nn1\r\n$2\r\nn2\r\n$2\r\nn3\r\n$2\r\nn4\r\n$2\r\nn5\r\n$2\r\nn6\r\n"
        );
        f.run(&[b"G.EADD", b"road", b"n0", b"n5", b"JUMP"]);
        assert_eq!(
            f.run(&[b"G.PATH", b"road", b"n0", b"n6"]),
            "*3\r\n$2\r\nn0\r\n$2\r\nn5\r\n$2\r\nn6\r\n"
        );
        // A node to itself is a path of one, and a depth too short to reach is
        // no path at all.
        assert_eq!(
            f.run(&[b"G.PATH", b"road", b"n2", b"n2"]),
            "*1\r\n$2\r\nn2\r\n"
        );
        assert_eq!(
            f.run(&[b"G.PATH", b"road", b"n0", b"n6", b"MAXDEPTH", b"1"]),
            "*0\r\n"
        );
        // Direction counts: the chain only goes one way.
        assert_eq!(f.run(&[b"G.PATH", b"road", b"n6", b"n0"]), "*0\r\n");
        // An unreachable node, a node that is not there, and a key that is not
        // there are the same empty answer.
        f.run(&[b"G.NADD", b"road", b"island"]);
        assert_eq!(f.run(&[b"G.PATH", b"road", b"n0", b"island"]), "*0\r\n");
        assert_eq!(f.run(&[b"G.PATH", b"road", b"n0", b"nobody"]), "*0\r\n");
        assert_eq!(f.run(&[b"G.PATH", b"nokey", b"n0", b"n6"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"G.PATH", b"road", b"n0", b"n6", b"NOPE", b"3"]),
            "-ERR syntax error\r\n"
        );
    }

    /// The point of the escape in the record tag: the keyspace owns a graph key
    /// the way it owns every other key, and none of these commands know a graph
    /// exists.
    #[test]
    fn the_keyspace_sees_a_graph_key_like_any_other() {
        let mut f = Fixture::new();
        f.run(&[b"G.EADD", b"social", b"ada", b"grace", b"F"]);
        assert_eq!(f.run(&[b"TYPE", b"social"]), "+graph\r\n");
        assert_eq!(
            f.run(&[b"OBJECT", b"ENCODING", b"social"]),
            "$9\r\nadjacency\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"social"]), ":1\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":1\r\n");
        assert_eq!(f.run(&[b"KEYS", b"*"]), "*1\r\n$6\r\nsocial\r\n");
        // A graph is counted against the server the way every other body is,
        // which is what `maxmemory` will read when this key is a million nodes.
        // There is no `MEMORY USAGE` command yet, so this asks the server.
        let held = f.server.memory_bytes();
        for i in 0..200u32 {
            let dst = format!("n{i}");
            f.run(&[b"G.EADD", b"big", b"hub", dst.as_bytes(), b"F"]);
        }
        assert!(
            f.server.memory_bytes() > held,
            "two hundred edges cost something: {held} then {}",
            f.server.memory_bytes()
        );
        f.run(&[b"DEL", b"big"]);

        // An expiry, then a rename, then a move to another database, all of
        // which are the keyspace moving a record it cannot look inside.
        assert_eq!(f.run(&[b"EXPIRE", b"social", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"PERSIST", b"social"]), ":1\r\n");
        assert_eq!(f.run(&[b"RENAME", b"social", b"net"]), "+OK\r\n");
        assert_eq!(f.run(&[b"MOVE", b"net", b"1"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"net"]), ":0\r\n");
        f.run(&[b"SELECT", b"1"]);
        assert_eq!(f.run(&[b"G.DEG", b"net", b"ada", b"F"]), ":1\r\n");

        assert_eq!(f.run(&[b"DEL", b"net"]), ":1\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        f.run(&[b"G.NADD", b"g", b"n"]);
        assert_eq!(f.run(&[b"FLUSHDB"]), "+OK\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
    }

    /// Neither `COPY` nor `DUMP` has a byte shape for a graph, so both say so
    /// rather than answering the way they answer for a key that is not there.
    #[test]
    fn a_graph_cannot_be_copied_or_dumped() {
        let mut f = Fixture::new();
        f.run(&[b"G.NADD", b"social", b"ada"]);
        assert_eq!(
            f.run(&[b"COPY", b"social", b"other"]),
            "-ERR COPY is not supported for a graph\r\n"
        );
        assert_eq!(
            f.run(&[b"COPY", b"social", b"other", b"DB", b"1"]),
            "-ERR COPY is not supported for a graph\r\n"
        );
        assert_eq!(
            f.run(&[b"DUMP", b"social"]),
            "-ERR DUMP is not supported for a graph\r\n"
        );
        // A refused copy leaves both keys exactly as they were.
        assert_eq!(f.run(&[b"EXISTS", b"social", b"other"]), ":1\r\n");
    }

    /// A graph key is a key, so the commands for the other types refuse it and
    /// the graph commands refuse theirs.
    #[test]
    fn a_graph_and_a_string_are_the_wrong_type_for_each_other() {
        let wrong = "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        let mut f = Fixture::new();
        f.run(&[b"G.NADD", b"social", b"ada"]);
        assert_eq!(f.run(&[b"GET", b"social"]), wrong);
        assert_eq!(f.run(&[b"LPUSH", b"social", b"x"]), wrong);
        assert_eq!(f.run(&[b"SADD", b"social", b"x"]), wrong);

        f.run(&[b"SET", b"str", b"v"]);
        for cmd in [
            vec![b"G.NADD".as_ref(), b"str", b"n"],
            vec![b"G.NGET".as_ref(), b"str", b"n"],
            vec![b"G.NDEL".as_ref(), b"str", b"n"],
            vec![b"G.EADD".as_ref(), b"str", b"a", b"b", b"F"],
            vec![b"G.EDEL".as_ref(), b"str", b"a", b"b", b"F"],
            vec![b"G.OUT".as_ref(), b"str", b"a", b"F"],
            vec![b"G.IN".as_ref(), b"str", b"a", b"F"],
            vec![b"G.DEG".as_ref(), b"str", b"a", b"F"],
            vec![b"G.NEIGH".as_ref(), b"str", b"a", b"F"],
            vec![b"G.PATH".as_ref(), b"str", b"a", b"b"],
        ] {
            assert_eq!(f.run(&cmd), wrong, "{:?}", cmd[0]);
        }
    }

    /// Every other collection here takes its key with it when its last member
    /// goes, and a graph is no different.
    #[test]
    fn a_graph_goes_when_its_last_node_does() {
        let mut f = Fixture::new();
        f.run(&[
            b"G.EADD", b"social", b"ada", b"grace", b"F", b"since", b"1843",
        ]);
        assert_eq!(f.run(&[b"G.NDEL", b"social", b"ada"]), ":1\r\n");
        // The node and the edges that hung off it are both gone.
        assert_eq!(f.run(&[b"G.NGET", b"social", b"ada"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"G.DEG", b"social", b"grace", b"F", b"IN"]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"G.NDEL", b"social", b"ada"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"social"]), ":1\r\n");

        assert_eq!(f.run(&[b"G.NDEL", b"social", b"grace"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"social"]), ":0\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":0\r\n");
        assert_eq!(f.run(&[b"G.NDEL", b"nokey", b"ada"]), ":0\r\n");

        // The id the removed node had is not handed out again, so a client
        // holding an id from an earlier reply cannot have it mean another node.
        f.run(&[b"G.NADD", b"social", b"first"]);
        f.run(&[b"G.NADD", b"social", b"second"]);
        f.run(&[b"G.NDEL", b"social", b"first"]);
        f.run(&[b"G.EADD", b"social", b"third", b"second", b"F"]);
        assert_eq!(
            f.run(&[b"G.OUT", b"social", b"third", b"F"]),
            "*2\r\n$1\r\n0\r\n*1\r\n$6\r\nsecond\r\n"
        );
    }

    // ------------------------------------------------------------------ json

    /// The two path syntaxes answer different shapes, which is the thing a
    /// client is most likely to be broken by and so the thing to pin first.
    #[test]
    fn a_json_path_answers_a_set_and_a_legacy_path_answers_a_value() {
        let mut f = Fixture::new();
        let doc = br#"{"a":1,"b":{"c":true}}"#;
        assert_eq!(f.run(&[b"JSON.SET", b"doc", b"$", doc]), "+OK\r\n");
        // No path at all is the legacy root and not `$`, so the document comes
        // back as itself rather than wrapped.
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc"]),
            bulk(r#"{"a":1,"b":{"c":true}}"#)
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b"$.a"]), bulk("[1]"));
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".a"]), bulk("1"));
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b"$..c"]), bulk("[true]"));
        // A path that matched nothing is an empty set on one syntax and an
        // error on the other, and the error does not quote the path.
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b"$.nope"]), bulk("[]"));
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b".nope"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"nokey"]), "$-1\r\n");
        // The key is a document to the rest of the keyspace, under the name
        // RedisJSON registers, and every generic command works on it.
        assert_eq!(f.run(&[b"TYPE", b"doc"]), "+ReJSON-RL\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"doc"]), ":1\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"doc"]), bulk("raw"));
        assert_eq!(f.run(&[b"DEL", b"doc"]), ":1\r\n");
        assert_eq!(f.run(&[b"JSON.GET", b"doc"]), "$-1\r\n");
    }

    /// The two error lines RedisJSON sends without a prefix in front of them.
    ///
    /// Every other error this server writes starts `ERR` or `WRONGTYPE`. These
    /// two do not, on a real server, and a differential harness compares the
    /// whole line.
    #[test]
    fn the_two_json_errors_that_carry_no_prefix() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"plain", b"x"]);
        let wrong = "-Existing key has wrong Redis type\r\n";
        assert_eq!(f.run(&[b"JSON.GET", b"plain"]), wrong);
        assert_eq!(f.run(&[b"JSON.SET", b"plain", b"$", b"1"]), wrong);
        assert_eq!(f.run(&[b"JSON.DEL", b"plain"]), wrong);
        assert_eq!(f.run(&[b"JSON.TYPE", b"plain"]), wrong);
        assert_eq!(f.run(&[b"JSON.CLEAR", b"plain"]), wrong);

        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":{"z":1},"b":{"z":2}}"#]);
        // A wildcard that matched something writes to all of it. A wildcard
        // that matched nothing would have to invent a place, and that is the
        // other unprefixed line.
        assert_eq!(f.run(&[b"JSON.SET", b"doc", b"$.*.z", b"9"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc"]),
            bulk(r#"{"a":{"z":9},"b":{"z":9}}"#)
        );
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.*.y", b"9"]),
            "-Err wrong static path\r\n"
        );
    }

    /// What `JSON.SET` does with a path that named nowhere.
    #[test]
    fn json_set_creates_one_field_and_refuses_to_invent_the_rest() {
        let mut f = Fixture::new();
        // A key that is not there can only be written whole.
        assert_eq!(
            f.run(&[b"JSON.SET", b"new", b".a", b"1"]),
            "-ERR new objects must be created at the root\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"new"]), ":0\r\n");
        // The root check comes before NX and XX, which is the order a real
        // server checks them in.
        assert_eq!(
            f.run(&[b"JSON.SET", b"new", b".a", b"1", b"NX"]),
            "-ERR new objects must be created at the root\r\n"
        );
        assert_eq!(f.run(&[b"JSON.SET", b"new", b"$", b"1", b"XX"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.SET", b"new", b"$", b"1", b"NX"]), "+OK\r\n");

        f.run(&[
            b"JSON.SET",
            b"doc",
            b"$",
            br#"{"o":{},"arr":[1,2],"s":"x"}"#,
        ]);
        // One step past a container that is there is a place to write.
        assert_eq!(f.run(&[b"JSON.SET", b"doc", b"$.o.made", b"1"]), "+OK\r\n");
        // One step past something that is not, or past something that is not an
        // object, is not an error and is not a write either.
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.nope.made", b"1"]),
            "$-1\r\n"
        );
        assert_eq!(f.run(&[b"JSON.SET", b"doc", b"$.s.made", b"1"]), "$-1\r\n");
        // An index past the end does not append. JSON.ARRAPPEND appends.
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.arr[5]", b"9"]),
            "-ERR array index out of range\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.arr[2]", b"9"]),
            "-ERR array index out of range\r\n"
        );
        assert_eq!(f.run(&[b"JSON.SET", b"doc", b"$.arr[1]", b"9"]), "+OK\r\n");
        // NX on a path that is there and XX on a path that is not are both a
        // nil and neither changes anything.
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.o.made", b"2", b"NX"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.gone", b"2", b"XX"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc"]),
            bulk(r#"{"o":{"made":1},"s":"x","arr":[1,9]}"#)
        );
        // Text that is not JSON is refused before the key is touched. The
        // line has no `ERR` in front of it, which is this command's and not
        // every command's, and is in D-37.
        assert!(
            f.run(&[b"JSON.SET", b"doc", b"$.s", b"nope"])
                .starts_with("-this is not the start of a value")
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".s"]), bulk("\"x\""));
    }

    /// `JSON.DEL`, `JSON.TYPE`, `JSON.TOGGLE` and `JSON.CLEAR`, each of which
    /// answers a count or a word rather than text.
    #[test]
    fn the_json_commands_that_do_not_answer_text() {
        let mut f = Fixture::new();
        let doc = br#"{"a":1,"t":true,"o":{"x":1},"arr":[1,2],"f":1.5,"s":"x","n":null}"#;
        f.run(&[b"JSON.SET", b"doc", b"$", doc]);

        assert_eq!(f.run(&[b"JSON.TYPE", b"doc"]), bulk("object"));
        assert_eq!(f.run(&[b"JSON.TYPE", b"doc", b".a"]), bulk("integer"));
        assert_eq!(f.run(&[b"JSON.TYPE", b"doc", b".f"]), bulk("number"));
        assert_eq!(
            f.run(&[b"JSON.TYPE", b"doc", b"$.a"]),
            format!("*1\r\n{}", bulk("integer"))
        );
        // The one place a legacy path that matched nothing is a nil rather than
        // an error, which lines up with a key that is not there.
        assert_eq!(f.run(&[b"JSON.TYPE", b"doc", b".nope"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.TYPE", b"nokey"]), "$-1\r\n");

        // A boolean flips and answers the value it now has, as an integer on
        // one syntax and as the word on the other.
        assert_eq!(f.run(&[b"JSON.TOGGLE", b"doc", b"$.t"]), "*1\r\n:0\r\n");
        assert_eq!(f.run(&[b"JSON.TOGGLE", b"doc", b".t"]), bulk("true"));
        // Something that is not a boolean is a hole on one syntax and one
        // sentence covering both cases on the other.
        assert_eq!(f.run(&[b"JSON.TOGGLE", b"doc", b"$.a"]), "*1\r\n$-1\r\n");
        assert_eq!(
            f.run(&[b"JSON.TOGGLE", b"doc", b".a"]),
            "-ERR Path does not exist or not a bool\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.TOGGLE", b"doc", b".nope"]),
            "-ERR Path does not exist or not a bool\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.TOGGLE", b"nokey", b"$.a"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );

        // Clearing empties containers and zeroes numbers and leaves everything
        // else alone, and counts only what it changed.
        assert_eq!(f.run(&[b"JSON.CLEAR", b"doc", b"$.s"]), ":0\r\n");
        assert_eq!(f.run(&[b"JSON.CLEAR", b"doc", b"$.*"]), ":4\r\n");
        assert_eq!(f.run(&[b"JSON.CLEAR", b"doc", b"$.*"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc"]),
            bulk(r#"{"a":0,"f":0,"n":null,"o":{},"s":"x","t":true,"arr":[]}"#)
        );

        // Deleting counts what it removed, and deleting the root is deleting
        // the key.
        assert_eq!(f.run(&[b"JSON.DEL", b"doc", b"$.nope"]), ":0\r\n");
        assert_eq!(f.run(&[b"JSON.DEL", b"doc", b"$.a"]), ":1\r\n");
        // Deleting the last member of the root container deletes the key, the
        // same way popping the last element off a list does. It is a rule about
        // deleting and not about shape: a document written as an empty object
        // by JSON.SET stays, because nothing was removed from it.
        assert_eq!(f.run(&[b"JSON.FORGET", b"doc", b"$.*"]), ":6\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"doc"]), ":0\r\n");
        assert_eq!(f.run(&[b"JSON.GET", b"doc"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.DEL", b"doc"]), ":0\r\n");
        assert_eq!(f.run(&[b"JSON.SET", b"empty", b"$", b"{}"]), "+OK\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"empty"]), ":1\r\n");
        assert_eq!(f.run(&[b"JSON.GET", b"empty"]), bulk("{}"));
        assert_eq!(f.run(&[b"JSON.DEL", b"nokey"]), ":0\r\n");
    }

    /// `JSON.GET` with more than one path, and with a layout.
    ///
    /// The wrapper the reply is built in is laid out too, so what a path
    /// matched starts one level in for a single JSONPath and two for one of
    /// several, and getting that wrong is the kind of thing only a byte for
    /// byte comparison catches.
    #[test]
    fn json_get_lays_out_the_wrapper_it_builds() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":1,"b":[1,{"c":2}]}"#]);

        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$.a", b"$.b"]),
            bulk(r#"{"$.a":[1],"$.b":[[1,{"c":2}]]}"#)
        );
        // Legacy paths are not wrapped, even when there are several of them.
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b".a", b".b"]),
            bulk(r#"{".a":1,".b":[1,{"c":2}]}"#)
        );
        let fmt: &[&[u8]] = &[b"INDENT", b"  ", b"NEWLINE", b"\n", b"SPACE", b" "];
        let mut one = vec![b"JSON.GET".as_slice(), b"doc"];
        one.extend_from_slice(fmt);
        one.push(b"$.b");
        assert_eq!(
            f.run(&one),
            bulk("[\n  [\n    1,\n    {\n      \"c\": 2\n    }\n  ]\n]")
        );
        let mut two = vec![b"JSON.GET".as_slice(), b"doc"];
        two.extend_from_slice(fmt);
        two.push(b"$.a");
        two.push(b"$.nope");
        assert_eq!(
            f.run(&two),
            bulk("{\n  \"$.a\": [\n    1\n  ],\n  \"$.nope\": []\n}")
        );
        // The options are read before the paths and in any order, and a
        // document with nothing to lay out is the same either way.
        let mut root = vec![b"JSON.GET".as_slice(), b"doc", b"SPACE", b" "];
        root.push(b".a");
        assert_eq!(f.run(&root), bulk("1"));
    }

    /// `JSON.MGET`, which is the only command here that reads more than one key
    /// and so the only one whose answer has holes in it.
    #[test]
    fn json_mget_answers_once_per_key_whatever_is_under_them() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"one", b"$", br#"{"a":1}"#]);
        f.run(&[b"JSON.SET", b"two", b"$", br#"{"a":2}"#]);
        f.run(&[b"SET", b"plain", b"x"]);
        assert_eq!(
            f.run(&[b"JSON.MGET", b"one", b"two", b"$.a"]),
            format!("*2\r\n{}{}", bulk("[1]"), bulk("[2]"))
        );
        // A key that is not there and a key holding something else are both a
        // hole rather than an error, the way MGET treats a hash.
        assert_eq!(
            f.run(&[b"JSON.MGET", b"one", b"nokey", b"plain", b".a"]),
            format!("*3\r\n{}$-1\r\n$-1\r\n", bulk("1"))
        );
        // A legacy path that matched nothing is a hole too, because one bad
        // answer should not lose the others.
        assert_eq!(f.run(&[b"JSON.MGET", b"one", b".nope"]), "*1\r\n$-1\r\n");
    }

    /// The four commands that ask how big something is, and the four different
    /// sets of answers they give for the same three failures.
    ///
    /// There is no pattern in this and there is no reading it off the
    /// documentation either. It was read off a running RedisJSON one line at a
    /// time, and it is written down here because the error text is what a client
    /// library branches on.
    #[test]
    fn the_json_commands_that_answer_a_size_disagree_about_every_failure() {
        let mut f = Fixture::new();
        let doc = br#"{"a":[1,2,3],"o":{"x":1,"y":2},"s":"hello","n":7}"#;
        f.run(&[b"JSON.SET", b"doc", b"$", doc]);

        assert_eq!(f.run(&[b"JSON.ARRLEN", b"doc", b".a"]), ":3\r\n");
        assert_eq!(f.run(&[b"JSON.ARRLEN", b"doc", b"$.a"]), "*1\r\n:3\r\n");
        assert_eq!(f.run(&[b"JSON.OBJLEN", b"doc", b".o"]), ":2\r\n");
        assert_eq!(f.run(&[b"JSON.STRLEN", b"doc", b".s"]), ":5\r\n");
        assert_eq!(
            f.run(&[b"JSON.OBJKEYS", b"doc", b".o"]),
            format!("*2\r\n{}{}", bulk("x"), bulk("y"))
        );
        // A JSONPath answers one entry per match and a hole for a match of the
        // wrong kind, which is the one shape all four agree on.
        assert_eq!(
            f.run(&[b"JSON.ARRLEN", b"doc", b"$.*"]),
            "*4\r\n:3\r\n$-1\r\n$-1\r\n$-1\r\n"
        );

        // A legacy path that matched nothing. Two of them are an error and two
        // of them are a nil, and the two errors do not use the same sentence.
        assert_eq!(
            f.run(&[b"JSON.ARRLEN", b"doc", b".nope"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRLEN", b"doc", b".nope"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(f.run(&[b"JSON.OBJLEN", b"doc", b".nope"]), "$-1\r\n");
        // A nil bulk and not an empty array, even though the answer would have
        // been an array, which is what RedisJSON sends here too.
        assert_eq!(f.run(&[b"JSON.OBJKEYS", b"doc", b".nope"]), "$-1\r\n");
        // The JSONPath spelling of the same question is an empty array, since
        // no match is not a failure on that syntax.
        assert_eq!(f.run(&[b"JSON.OBJKEYS", b"doc", b"$.nope"]), "*0\r\n");

        // A legacy path that matched the wrong kind of value. Now two of them
        // are an ERR and two of them are a WRONGTYPE, and it is not the same
        // two.
        assert_eq!(
            f.run(&[b"JSON.ARRLEN", b"doc", b".n"]),
            "-ERR Path does not exist or not an array\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.OBJKEYS", b"doc", b".n"]),
            "-ERR Path does not exist or not an object\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.OBJLEN", b"doc", b".n"]),
            "-WRONGTYPE wrong type of path value - expected object\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRLEN", b"doc", b".n"]),
            "-WRONGTYPE wrong type of path value - expected string\r\n"
        );

        // A key that is not there, where the two syntaxes swap over: the legacy
        // path is the quiet answer and the JSONPath is the error.
        assert_eq!(f.run(&[b"JSON.ARRLEN", b"nokey", b".a"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.OBJLEN", b"nokey", b".a"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.STRLEN", b"nokey", b".a"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.OBJKEYS", b"nokey", b".a"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"JSON.ARRLEN", b"nokey", b"$.a"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
        // Except this one, which answers about the path instead.
        assert_eq!(
            f.run(&[b"JSON.OBJLEN", b"nokey", b"$.a"]),
            "-ERR Path does not exist or not an object\r\n"
        );
    }

    /// `JSON.ARRAPPEND`, `JSON.ARRINSERT`, `JSON.ARRTRIM` and `JSON.ARRPOP`.
    ///
    /// The four of them share one error line for a path that named something
    /// that is not an array, and they disagree about what an index outside the
    /// array means: insert refuses it and the other two clamp.
    #[test]
    fn the_json_array_writes_agree_on_the_errors_and_not_on_the_indexes() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[1,2,3],"n":7}"#]);

        assert_eq!(f.run(&[b"JSON.ARRAPPEND", b"doc", b".a", b"4"]), ":4\r\n");
        assert_eq!(
            f.run(&[b"JSON.ARRAPPEND", b"doc", b"$.a", b"5", b"6"]),
            "*1\r\n:6\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".a"]), bulk("[1,2,3,4,5,6]"));

        // A negative index counts back from the end, and the end itself is a
        // place to insert at, so an insert at the length is an append.
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b".a", b"-1", b"0"]),
            ":7\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b".a"]),
            bulk("[1,2,3,4,5,0,6]")
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b".a", b"7", b"9"]),
            ":8\r\n"
        );
        // One past the end is not, and neither is one before the front.
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b".a", b"9", b"9"]),
            "-ERR index out of bounds\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b".a", b"-9", b"9"]),
            "-ERR index out of bounds\r\n"
        );

        // Trim takes both ends inclusive and clamps both of them, so a start
        // past the end leaves an empty array rather than an error.
        f.run(&[b"JSON.SET", b"doc", b"$.a", b"[1,2,3,4,5]"]);
        assert_eq!(
            f.run(&[b"JSON.ARRTRIM", b"doc", b".a", b"1", b"3"]),
            ":3\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".a"]), bulk("[2,3,4]"));
        assert_eq!(
            f.run(&[b"JSON.ARRTRIM", b"doc", b".a", b"-2", b"99"]),
            ":2\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".a"]), bulk("[3,4]"));
        assert_eq!(
            f.run(&[b"JSON.ARRTRIM", b"doc", b".a", b"9", b"9"]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".a"]), bulk("[]"));

        // Pop clamps as well, its default is the last element, and an empty
        // array pops a nil rather than failing.
        f.run(&[b"JSON.SET", b"doc", b"$.a", b"[1,2,3]"]);
        assert_eq!(f.run(&[b"JSON.ARRPOP", b"doc", b".a"]), bulk("3"));
        assert_eq!(f.run(&[b"JSON.ARRPOP", b"doc", b".a", b"0"]), bulk("1"));
        assert_eq!(f.run(&[b"JSON.ARRPOP", b"doc", b".a", b"99"]), bulk("2"));
        assert_eq!(f.run(&[b"JSON.ARRPOP", b"doc", b".a"]), "$-1\r\n");

        // One sentence covers a path that matched nothing and a path that
        // matched the wrong kind of value, for all four of them.
        for call in [
            &[&b"JSON.ARRAPPEND"[..], b"doc", b"PATH", b"1"][..],
            &[&b"JSON.ARRTRIM"[..], b"doc", b"PATH", b"1", b"1"][..],
            &[&b"JSON.ARRPOP"[..], b"doc", b"PATH", b"1"][..],
            &[&b"JSON.ARRINSERT"[..], b"doc", b"PATH", b"0", b"1"][..],
        ] {
            for path in [&b".n"[..], &b".nope"[..]] {
                let args: Vec<&[u8]> = call
                    .iter()
                    .map(|a| if *a == b"PATH" { path } else { *a })
                    .collect();
                assert_eq!(
                    f.run(&args),
                    "-ERR Path does not exist or not an array\r\n",
                    "{} {}",
                    String::from_utf8_lossy(call[0]),
                    String::from_utf8_lossy(path)
                );
            }
        }

        // A key that is not there is the same sentence for all four, on either
        // syntax, and it is about the key and not about the path.
        assert_eq!(
            f.run(&[b"JSON.ARRAPPEND", b"nokey", b".a", b"1"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRPOP", b"nokey", b"$.a"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );

        // The values are parsed before the key is touched, so text that is not
        // JSON leaves the document alone.
        // Text that is not JSON is refused before the key is touched, and
        // the line has no `ERR` in front of it, which is D-37.
        assert!(
            f.run(&[b"JSON.ARRAPPEND", b"doc", b".a", b"nope"])
                .starts_with("-this is not the start of a value")
        );
        assert_eq!(f.run(&[b"JSON.GET", b"doc", b".a"]), bulk("[]"));
    }

    /// `JSON.ARRINSERT` refuses the whole command when any one of the arrays a
    /// path matched cannot take the index, which is D-36.
    ///
    /// RedisJSON walks the matches, inserts into each one it can, and returns
    /// the error on the first one it cannot, leaving the earlier inserts in the
    /// document. A write here is one list of edits applied together, so either
    /// all of them happen or none of them do.
    #[test]
    fn json_arrinsert_is_all_or_nothing_across_the_matches() {
        let mut f = Fixture::new();
        let doc = br#"{"a":[1,2,3],"n":{"a":[9,8],"in":{"a":[1]}}}"#;
        f.run(&[b"JSON.SET", b"doc", b"$", doc]);
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b"$..a", b"-2", b"0"]),
            "-ERR index out of bounds\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc"]),
            bulk(r#"{"a":[1,2,3],"n":{"a":[9,8],"in":{"a":[1]}}}"#)
        );
        // Every match can take the index, so every match gets it.
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b"$..a", b"0", b"0"]),
            "*3\r\n:4\r\n:3\r\n:2\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc"]),
            bulk(r#"{"a":[0,1,2,3],"n":{"a":[0,9,8],"in":{"a":[0,1]}}}"#)
        );
    }

    /// `JSON.ARRINDEX`, whose stop is exclusive and whose start clamps to the
    /// last element rather than to one past it.
    ///
    /// Both of those read like mistakes and both are what RedisJSON does. The
    /// start is the one that bites: a start of five into an array of four still
    /// looks at the fourth, so a search that should have run out of array comes
    /// back with an answer.
    #[test]
    fn json_arrindex_has_an_exclusive_stop_and_a_start_that_cannot_run_off_the_end() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[1,2,3,1],"n":7}"#]);

        assert_eq!(f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"2"]), ":1\r\n");
        assert_eq!(f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"9"]), ":-1\r\n");
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b"$.a", b"2"]),
            "*1\r\n:1\r\n"
        );

        // Zero as the stop means the end rather than the front, so leaving it
        // off and passing it are the same thing.
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"1", b"1", b"0"]),
            ":3\r\n"
        );
        // The stop is exclusive, so a stop of three does not look at index
        // three.
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"1", b"1", b"3"]),
            ":-1\r\n"
        );

        // The start clamps to the last element in both directions, which is why
        // a start of four, five or minus one all find the 1 at index three.
        for start in [&b"4"[..], &b"5"[..], &b"-1"[..]] {
            assert_eq!(
                f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"1", start]),
                ":3\r\n",
                "{}",
                String::from_utf8_lossy(start)
            );
        }
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"1", b"-100"]),
            ":0\r\n"
        );
        // An empty array is the one case that comes back with nothing, since
        // the stop is zero and the loop never starts.
        f.run(&[b"JSON.SET", b"doc", b"$.a", b"[]"]);
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"1", b"1"]),
            ":-1\r\n"
        );

        // The comparison is structural rather than one of the encoded bytes,
        // because an object in a stored document holds its keys as intern table
        // ids where one parsed off the wire holds them as bytes.
        f.run(&[b"JSON.SET", b"doc", b"$.a", br#"[{"k":1},[1,2],"s"]"#]);
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", br#"{"k":1}"#]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"[1,2]"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".a", b"[2,1]"]),
            ":-1\r\n"
        );

        // Its errors are a third set again: a missing legacy path is the short
        // sentence, the wrong kind of value is a WRONGTYPE, and a key that is
        // not there is about the path on either syntax.
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".nope", b"1"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"doc", b".n", b"1"]),
            "-WRONGTYPE wrong type of path value - expected array\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"nokey", b".a", b"1"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.ARRINDEX", b"nokey", b"$.a", b"1"]),
            "-ERR Path does not exist\r\n"
        );
    }

    /// The number family answers text and keeps an integer an integer until
    /// something in the sum is not one.
    #[test]
    fn the_json_number_family_answers_json_text_and_keeps_its_integers() {
        let mut f = Fixture::new();
        let doc = br#"{"i":7,"f":1.5,"neg":-2,"s":"ab"}"#;
        f.run(&[b"JSON.SET", b"doc", b"$", doc]);

        // A legacy path answers the new value as JSON text in a bulk string,
        // not as a number, which is the shape all three of them use.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".i", b"2"]),
            bulk("9").as_str()
        );
        // A JSONPath answers a bulk string holding a JSON array.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.i", b"2"]),
            bulk("[11]").as_str()
        );
        // Two integers stay an integer and a double anywhere in it makes the
        // answer a double, which the document then holds.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".i", b"2.0"]),
            bulk("13.0").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.TYPE", b"doc", b".i"]),
            bulk("number").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMMULTBY", b"doc", b".f", b"2"]),
            bulk("3.0").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMPOWBY", b"doc", b".neg", b"3"]),
            bulk("-8").as_str()
        );
        // A power of a half is a square root, and the square root of a negative
        // number is the error that says the answer is not a number.
        f.run(&[b"JSON.SET", b"doc", b"$.f", b"1.5"]);
        assert_eq!(
            f.run(&[b"JSON.NUMPOWBY", b"doc", b".f", b"0.5"]),
            bulk("1.224744871391589").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMPOWBY", b"doc", b".neg", b"0.5"]),
            "-ERR result is not a number\r\n"
        );
        // An integer answer that does not fit is refused rather than promoted,
        // and a negative exponent lands in the same error because there is no
        // integer answer to two to the minus one.
        f.run(&[b"JSON.SET", b"doc", b"$.big", b"9223372036854775807"]);
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".big", b"1"]),
            "-ERR numeric overflow\r\n"
        );
        f.run(&[b"JSON.SET", b"doc", b"$.p", b"2"]);
        assert_eq!(
            f.run(&[b"JSON.NUMPOWBY", b"doc", b".p", b"-1"]),
            "-ERR numeric overflow\r\n"
        );
        // A double that leaves the finite numbers is the other error.
        f.run(&[b"JSON.SET", b"doc", b"$.huge", b"1e308"]);
        assert_eq!(
            f.run(&[b"JSON.NUMMULTBY", b"doc", b".huge", b"1e10"]),
            "-ERR result is not a number\r\n"
        );

        // A match that is not a number is a null inside the array on a
        // JSONPath, and a legacy path that found no number at all is the error
        // with the module's own typo in it.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.s", b"1"]),
            bulk("[null]").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.nope", b"1"]),
            bulk("[]").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".s", b"1"]),
            "-ERR Path does not exist or does not contains a number\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".nope", b"1"]),
            "-ERR Path does not exist or does not contains a number\r\n"
        );
        // The operand is JSON and has to be a number. Valid JSON that is not
        // one is a line of its own, and it goes out without a prefix.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".i", b"true"]),
            "-bad input number\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"nokey", b".i", b"1"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"nokey", b"$.i", b"1"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
    }

    /// `JSON.STRAPPEND` puts its path in the middle and makes it optional,
    /// which nothing else in the group does.
    #[test]
    fn json_strappend_reads_its_shape_off_the_argument_count() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"s":"ab","n":1}"#]);

        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b".s", br#""c""#]),
            ":3\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b"$.s", br#""d""#]),
            "*1\r\n:4\r\n"
        );
        // The length is in bytes and not in characters, so one two byte letter
        // takes it up by two.
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b".s", br#""\u00e9""#]),
            ":6\r\n"
        );
        // Three arguments means the value is the last one and the path is the
        // root, so this appends to a document that is a string on its own.
        f.run(&[b"JSON.SET", b"str", b"$", br#""ab""#]);
        assert_eq!(f.run(&[b"JSON.STRAPPEND", b"str", br#""c""#]), ":3\r\n");
        assert_eq!(f.run(&[b"JSON.GET", b"str"]), bulk("\"abc\"").as_str());

        // The value is JSON and has to be a JSON string. A number is a
        // WRONGTYPE about a path value even though it was the value that was
        // wrong, which is the module's wording and not a slip here.
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b".s", b"5"]),
            "-WRONGTYPE wrong type of path value - expected string\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b"$.n", br#""c""#]),
            "*1\r\n$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b".n", br#""c""#]),
            "-ERR Path does not exist or not a string\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b"$.nope", br#""c""#]),
            "*0\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"nokey", br#""c""#]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
    }

    /// A legacy path can match more than one value, and which of them the one
    /// answer comes from is not the same choice twice.
    #[test]
    fn a_legacy_wildcard_write_touches_every_match_and_answers_only_one() {
        let mut f = Fixture::new();
        // Three arrays of one, two and three elements, which tells the first
        // match and the last match apart in a single command.
        let three = br#"{"a":[[7],[7,7],[7,7,7]]}"#;

        f.run(&[b"JSON.SET", b"doc", b"$", three]);
        assert_eq!(
            f.run(&[b"JSON.ARRAPPEND", b"doc", b".a[*]", b"9"]),
            ":4\r\n"
        );
        f.run(&[b"JSON.SET", b"doc", b"$", three]);
        assert_eq!(
            f.run(&[b"JSON.ARRINSERT", b"doc", b".a[*]", b"0", b"9"]),
            ":2\r\n"
        );
        f.run(&[b"JSON.SET", b"doc", b"$", three]);
        assert_eq!(
            f.run(&[b"JSON.ARRTRIM", b"doc", b".a[*]", b"0", b"1"]),
            ":1\r\n"
        );
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[[1,2,3],[4,5,6]]}"#]);
        assert_eq!(
            f.run(&[b"JSON.ARRPOP", b"doc", b".a[*]", b"0"]),
            bulk("1").as_str()
        );
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[1,2,3]}"#]);
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".a[*]", b"10"]),
            bulk("13").as_str()
        );
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":["p","qq","rrr"]}"#]);
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b".a[*]", br#""z""#]),
            ":4\r\n"
        );
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[true,false,true]}"#]);
        assert_eq!(
            f.run(&[b"JSON.TOGGLE", b"doc", b".a[*]"]),
            bulk("false").as_str()
        );
        // Every one of them wrote to all three matches, whichever one it chose
        // to answer about.
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b".a"]),
            bulk("[false,true,false]").as_str()
        );

        // A match of the wrong kind is skipped rather than being the answer, so
        // a path that found a string and then two arrays still answers.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":["x",[1],[1,2]]}"#]);
        assert_eq!(
            f.run(&[b"JSON.ARRAPPEND", b"doc", b".a[*]", b"9"]),
            ":3\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b".a"]),
            bulk(r#"["x",[1,9],[1,2,9]]"#).as_str()
        );
        // Nothing of the right kind anywhere is the error, and that is the only
        // case that is.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":["x","y"]}"#]);
        assert_eq!(
            f.run(&[b"JSON.ARRAPPEND", b"doc", b".a[*]", b"9"]),
            "-ERR Path does not exist or not an array\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.TOGGLE", b"doc", b".a[*]"]),
            "-ERR Path does not exist or not a bool\r\n"
        );
        // The one array that was there and had nothing in it is an answer and
        // not a skip, so the pop answers about it rather than about the array
        // after it.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[[],[2,3]]}"#]);
        assert_eq!(f.run(&[b"JSON.ARRPOP", b"doc", b".a[*]"]), "$-1\r\n");
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b".a"]),
            bulk("[[],[2]]").as_str()
        );
    }

    /// A path that matched a value and something inside that value writes to
    /// both, which is what `$..` and a nested wildcard are for.
    #[test]
    fn a_write_reaches_a_match_that_sits_inside_another_match() {
        let mut f = Fixture::new();
        let nested = br#"{"a":[{"a":[7]},{"a":[7,7]}]}"#;

        f.run(&[b"JSON.SET", b"doc", b"$", nested]);
        assert_eq!(
            f.run(&[b"JSON.ARRAPPEND", b"doc", b"$..a", b"9"]),
            "*3\r\n:3\r\n:2\r\n:3\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":[{"a":[7,9]},{"a":[7,7,9]},9]}]"#).as_str()
        );

        // The same for a trim, where the outer array keeps the two elements the
        // inner writes landed in.
        f.run(&[b"JSON.SET", b"doc", b"$", nested]);
        assert_eq!(
            f.run(&[b"JSON.ARRTRIM", b"doc", b"$..a", b"0", b"0"]),
            "*3\r\n:1\r\n:1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":[{"a":[7]}]}]"#).as_str()
        );

        // And for a number, where the first match is the object the outer array
        // holds and only the two inside it are numbers.
        f.run(&[b"JSON.SET", b"doc", b"$", nested]);
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$..a[0]", b"1"]),
            bulk("[null,8,8]").as_str()
        );
    }

    /// The value a write is given is looked at only once the path has found
    /// something of the right kind to use it on.
    #[test]
    fn a_bad_operand_is_not_the_answer_when_the_path_found_nothing_to_use_it_on() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"n":7,"s":"t"}"#]);

        // A string is not a number, so the path answers first and the `"x"` is
        // never looked at. Same for the value that is not JSON at all.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.s", br#""x""#]),
            bulk("[null]").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.s", b"notjson"]),
            bulk("[null]").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.missing", b"notjson"]),
            bulk("[]").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b".s", br#""x""#]),
            "-ERR Path does not exist or does not contains a number\r\n"
        );
        // A number match anywhere and the value is looked at after all.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.n", br#""x""#]),
            "-bad input number\r\n"
        );

        // JSON.STRAPPEND follows the same order with its own two answers.
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b"$.n", b"1"]),
            "*1\r\n$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b".n", b"1"]),
            "-ERR Path does not exist or not a string\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"doc", b"$.s", b"1"]),
            "-WRONGTYPE wrong type of path value - expected string\r\n"
        );

        // A key that is not there still comes before either of them.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"nope", b"$.a", br#""x""#]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.STRAPPEND", b"nope", b"$.a", b"1"]),
            "-ERR could not perform this operation on a key that doesn't exist\r\n"
        );
    }

    /// RFC 7386 in one test: a null deletes, everything else merges, and a
    /// patch that is not an object replaces what it lands on.
    #[test]
    fn a_merge_patch_adds_replaces_and_deletes_in_one_write() {
        let mut f = Fixture::new();

        // A key that is not there is created at the root, nulls and all,
        // because a deletion with nothing to delete is still what the client
        // sent.
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$", br#"{"x":null,"y":1}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"x":null,"y":1}]"#).as_str()
        );

        // Onto something that is there, a null deletes the member of that name
        // and the rest is merged one level at a time.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":{"b":1,"c":2},"d":3}"#]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$", br#"{"a":{"b":null,"e":4}}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":{"c":2,"e":4},"d":3}]"#).as_str()
        );

        // A patch that is not an object replaces what it is merged onto.
        assert_eq!(f.run(&[b"JSON.MERGE", b"doc", b"$.a", b"[1,2]"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":[1,2],"d":3}]"#).as_str()
        );

        // A patch object onto a value that is not an object starts from an
        // empty object, so this time the null has nothing to delete and is
        // dropped rather than stored.
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.d", br#"{"p":null,"q":9}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":[1,2],"d":{"q":9}}]"#).as_str()
        );

        // A member one level past the end of the document is created and keeps
        // its nulls, two levels past it is a write that did not happen, and a
        // path that would have to invent where it goes is the unprefixed line.
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.new", br#"{"z":null}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$.new"]),
            bulk(r#"[{"z":null}]"#).as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.no.deep", b"1"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.no.*", b"1"]),
            "-Err wrong static path\r\n"
        );

        // A wildcard merges every match.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":{"n":1},"b":{"n":2}}"#]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.*", br#"{"m":0}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":{"m":0,"n":1},"b":{"m":0,"n":2}}]"#).as_str()
        );

        // The three ways to get it wrong.
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$", b"{}", b"more"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"gone", b"$.a", b"1"]),
            "-ERR new objects must be created at the root\r\n"
        );
        f.run(&[b"SET", b"str", b"x"]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"str", b"$", b"1"]),
            "-Existing key has wrong Redis type\r\n"
        );
    }

    /// A descent is the one path that matches a value and something inside that
    /// same value, and the inner merge has to survive the outer one.
    #[test]
    fn a_merge_down_a_descent_keeps_what_the_inner_match_did() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":{"b":1},"c":[2]}"#]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$..*", br#"{"m":1}"#]),
            "+OK\r\n"
        );
        // `a`, `a.b`, `c` and `c[0]` all match. `a.b` is merged first and `a` is
        // merged onto the result, so the `{"m":1}` written into `a.b` is still
        // there. Doing it the other way round would leave `{"a":{"b":1,"m":1}}`.
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":{"b":{"m":1},"m":1},"c":{"m":1}}]"#).as_str()
        );

        // A deletion down the same path, which is the case where the inner
        // merge empties the object the outer one then copies.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":{"b":1},"c":[2]}"#]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$..*", br#"{"a":null}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":{"b":{}},"c":{}}]"#).as_str()
        );
    }

    /// A filter is a selector like any other, so every command that takes a path
    /// takes one, reads and writes alike.
    #[test]
    fn a_filter_path_reads_and_writes_the_members_it_keeps() {
        let mut f = Fixture::new();
        let doc = br#"{"book":[{"t":"a","p":8},{"t":"b","p":13},{"t":"c","p":9}],"cap":10}"#;
        f.run(&[b"JSON.SET", b"doc", b"$", doc]);

        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$.book[?(@.p < 10)].t"]),
            bulk(r#"["a","c"]"#).as_str()
        );
        // `$` inside the expression is the document, so a member can be measured
        // against something that is not inside it.
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$.book[?(@.p < $.cap)].t"]),
            bulk(r#"["a","c"]"#).as_str()
        );
        // The legacy syntax takes one too, and answers the first match.
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"book[?(@.p < 10)].t"]),
            bulk(r#""a""#).as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.TYPE", b"doc", b"$.book[?(@.p > 10)]"]),
            "*1\r\n$6\r\nobject\r\n"
        );

        // A write goes through it as far as a value that is already there. A
        // field that is not there yet has nowhere definite to go, which is the
        // same refusal a wildcard gets.
        assert_eq!(
            f.run(&[b"JSON.NUMINCRBY", b"doc", b"$.book[?(@.p < 10)].p", b"1"]),
            bulk("[9,10]").as_str()
        );
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.book[?(@.p == 13)].t", br#""B""#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.book[?(@.p == 13)].n", b"1"]),
            "-Err wrong static path\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.DEL", b"doc", b"$.book[?(@.p > 9)]"]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"cap":10,"book":[{"p":9,"t":"a"}]}]"#).as_str()
        );

        // A path that does not parse is refused before the document is read, so
        // a key that is not there answers the same way.
        assert!(
            f.run(&[b"JSON.GET", b"doc", b"$.book[?(@.p <)]"])
                .starts_with("-ERR")
        );
        assert!(
            f.run(&[b"JSON.GET", b"nokey", b"$.book[?(@.p <)]"])
                .starts_with("-ERR")
        );
    }

    /// The operators past the comparisons, over the wire rather than in the
    /// parser's own tests, so that a client can reach all of them.
    #[test]
    fn a_filter_takes_the_membership_operators_and_the_methods_too() {
        let mut f = Fixture::new();
        let doc = br#"{"box":[{"t":"a","n":[1,2],"g":"x"},{"t":"b","n":[9],"g":"y"}]}"#;
        f.run(&[b"JSON.SET", b"doc", b"$", doc]);

        for (path, want) in [
            (&b"$.box[?(@.g in [\"x\"])].t"[..], r#"["a"]"#),
            (b"$.box[?(@.g nin [\"x\"])].t", r#"["b"]"#),
            (b"$.box[?(@.n anyof [2,3])].t", r#"["a"]"#),
            (b"$.box[?(@.n subsetof [1,2,3])].t", r#"["a"]"#),
            (b"$.box[?(@.n size 2)].t", r#"["a"]"#),
            (b"$.box[?(@.n empty false)].t", r#"["a","b"]"#),
            (b"$.box[?(@.n.length() == 1)].t", r#"["b"]"#),
            (b"$.box[?(@.n.sum() > 5)].t", r#"["b"]"#),
            (b"$.box[?(@.n[0] + 1 == 2)].t", r#"["a"]"#),
            (b"$.box[?(@~ size 3)].t", r#"["a","b"]"#),
            (b"$.box[?(@.n~)].t", "[]"),
            (b"$.box[?(@.n sizeof 2)].t", r#"["a"]"#),
            (b"$.box[?(-@.n[0] == -9)].t", r#"["b"]"#),
            (b"$.box[?(1 in @.n)].t", r#"["a"]"#),
            (b"$.box[?(\"g\" in @~)].t", r#"["a","b"]"#),
        ] {
            assert_eq!(f.run(&[b"JSON.GET", b"doc", path]), bulk(want).as_str());
        }

        // A write goes through one of these the same way it goes through a
        // comparison.
        assert_eq!(
            f.run(&[b"JSON.SET", b"doc", b"$.box[?(@.n size 1)].g", br#""z""#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$.box[?(@.g == \"z\")].t"]),
            bulk(r#"["b"]"#).as_str()
        );
    }

    /// D-41. RedisJSON refuses this one, and which document it refuses is
    /// decided by how it happens to hold an array of numbers.
    #[test]
    fn a_merge_onto_a_number_inside_an_array_is_a_merge_and_not_an_error() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[1,2]}"#]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.a[0]", br#"{"x":1}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":[{"x":1},2]}]"#).as_str()
        );
        // The same document with one element that is not an integer is the one
        // RedisJSON is happy with, and it goes the same way here.
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[1,"s"]}"#]);
        assert_eq!(
            f.run(&[b"JSON.MERGE", b"doc", b"$.a[0]", br#"{"x":1}"#]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"doc", b"$"]),
            bulk(r#"[{"a":[{"x":1},"s"]}]"#).as_str()
        );
    }

    /// `JSON.MSET` checks what it can before it writes anything and skips the
    /// one thing it cannot, which is a path with nowhere to put its value.
    #[test]
    fn an_mset_writes_every_triple_it_can_and_checks_the_rest_up_front() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$", b"1", b"b", b"$", b"2"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"a", b"$"]), bulk("[1]").as_str());
        assert_eq!(f.run(&[b"JSON.GET", b"b", b"$"]), bulk("[2]").as_str());

        // A repeated key takes the last write.
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$", b"3", b"a", b"$", b"4"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"a", b"$"]), bulk("[4]").as_str());

        // A triple whose path names nowhere is skipped, the others are still
        // written and the reply turns into a nil. Both ways round, because a
        // loop that gave up at the first skip would agree with this on one
        // order and not on the other.
        f.run(&[b"JSON.SET", b"a", b"$", br#"{"n":1}"#]);
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$.no.deep", b"9", b"b", b"$", b"5"]),
            "$-1\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"b", b"$"]), bulk("[5]").as_str());
        assert_eq!(
            f.run(&[b"JSON.MSET", b"b", b"$", b"6", b"a", b"$.no.deep", b"9"]),
            "$-1\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"b", b"$"]), bulk("[6]").as_str());

        // A value that is not JSON, a key holding something else and a path
        // that would have to create a document below its own root are all
        // checked before anything is written, so the good triple next to them
        // does not happen either.
        f.run(&[b"SET", b"str", b"x"]);
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$.n", b"7", b"b", b"$", b"notjson"]),
            "-this is not the start of a value, at byte 0 of the JSON text\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$.n", b"7", b"str", b"$", b"1"]),
            "-Existing key has wrong Redis type\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$.n", b"7", b"gone", b"$.x", b"1"]),
            "-ERR new objects must be created at the root\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"a", b"$.n"]), bulk("[1]").as_str());

        // The two errors a path can be are checked up front as well, so the
        // triple before them is not written either. A wildcard that matched
        // nothing has nowhere to invent, and an index that is not in the array
        // is out of range, and both of them stop the whole command.
        assert_eq!(
            f.run(&[b"JSON.MSET", b"b", b"$", b"8", b"a", b"$.no.*", b"9"]),
            "-Err wrong static path\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.MSET", b"b", b"$", b"8", b"a", b"$[0]", b"9"]),
            "-ERR array index out of range\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", b"b", b"$"]), bulk("[6]").as_str());

        // Every triple is worked out against the keyspace as the command found
        // it, so a second triple on the same key does not see the first one and
        // the last write is the one that stays.
        f.run(&[b"JSON.SET", b"c", b"$", br#"{"n":1}"#]);
        assert_eq!(
            f.run(&[b"JSON.MSET", b"c", b"$", br#"{"n":2}"#, b"c", b"$.n", b"3"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.GET", b"c", b"$"]),
            bulk(r#"[{"n":3}]"#).as_str()
        );

        // An argument count that is not a run of key, path and value is the
        // arity error rather than a syntax one.
        assert_eq!(
            f.run(&[b"JSON.MSET", b"a", b"$", b"1", b"b"]),
            "-ERR wrong number of arguments for 'json.mset' command\r\n"
        );
    }

    /// `JSON.RESP` hands back RESP types, and the marker element is what tells
    /// an empty array and an empty object apart.
    #[test]
    fn json_resp_answers_the_document_as_resp_types() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":1,"b":[2,"c"]}"#]);
        assert_eq!(
            f.run(&[b"JSON.RESP", b"doc"]),
            "*5\r\n+{\r\n$1\r\na\r\n:1\r\n$1\r\nb\r\n*3\r\n+[\r\n:2\r\n$1\r\nc\r\n"
        );
        // A JSONPath wraps the same answer in one more array.
        assert_eq!(
            f.run(&[b"JSON.RESP", b"doc", b"$.b"]),
            "*1\r\n*3\r\n+[\r\n:2\r\n$1\r\nc\r\n"
        );

        f.run(&[
            b"JSON.SET",
            b"doc",
            b"$",
            br#"{"f":2.5,"t":true,"z":null,"e":[],"o":{}}"#,
        ]);
        assert_eq!(f.run(&[b"JSON.RESP", b"doc", b".e"]), "*1\r\n+[\r\n");
        assert_eq!(f.run(&[b"JSON.RESP", b"doc", b".o"]), "*1\r\n+{\r\n");
        // A double goes out as its text, so a client reads the same digits
        // `JSON.GET` would have given it.
        assert_eq!(f.run(&[b"JSON.RESP", b"doc", b".f"]), bulk("2.5").as_str());
        assert_eq!(f.run(&[b"JSON.RESP", b"doc", b".t"]), "+true\r\n");
        assert_eq!(f.run(&[b"JSON.RESP", b"doc", b".z"]), "$-1\r\n");

        // A missing legacy path is an error, a missing JSONPath is an empty
        // array, and a key that is not there is a nil on either.
        assert_eq!(
            f.run(&[b"JSON.RESP", b"doc", b".nope"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(f.run(&[b"JSON.RESP", b"doc", b"$.nope"]), "*0\r\n");
        assert_eq!(f.run(&[b"JSON.RESP", b"gone"]), "$-1\r\n");
        assert_eq!(f.run(&[b"JSON.RESP", b"gone", b"$"]), "$-1\r\n");
    }

    /// `JSON.DEBUG` answers a byte count that is this encoding's, so the test
    /// pins the shapes and that the two syntaxes agree rather than a number
    /// read off another server. That is D-42.
    #[test]
    fn json_debug_answers_a_byte_count_and_its_own_help() {
        let mut f = Fixture::new();
        f.run(&[b"JSON.SET", b"doc", b"$", br#"{"a":[1,2],"s":"hello"}"#]);
        let one = f.run(&[b"JSON.DEBUG", b"MEMORY", b"doc", b".s"]);
        assert!(one.starts_with(':'), "{one}");
        assert_eq!(
            f.run(&[b"JSON.DEBUG", b"memory", b"doc", b"$.s"]),
            format!("*1\r\n{one}")
        );
        let whole = f.run(&[b"JSON.DEBUG", b"MEMORY", b"doc"]);
        assert!(whole.starts_with(':') && whole.len() > one.len(), "{whole}");

        // A key that is not there is a zero on a legacy path and an empty set
        // on a JSONPath, which is the one reader here that does not answer nil
        // for it.
        assert_eq!(f.run(&[b"JSON.DEBUG", b"MEMORY", b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"JSON.DEBUG", b"MEMORY", b"gone", b"$"]), "*0\r\n");
        assert_eq!(
            f.run(&[b"JSON.DEBUG", b"MEMORY", b"doc", b".nope"]),
            "-ERR Path does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.DEBUG", b"MEMORY", b"doc", b"$.nope"]),
            "*0\r\n"
        );

        assert_eq!(
            f.run(&[b"JSON.DEBUG", b"HELP"]),
            "*2\r\n$42\r\nMEMORY <key> [path] - reports memory usage\r\n\
             $34\r\nHELP                - this message\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.DEBUG", b"NOPE"]),
            "-ERR unknown subcommand - try `JSON.DEBUG HELP`\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.DEBUG", b"MEMORY"]),
            "-ERR wrong number of arguments for 'json.debug' command\r\n"
        );
    }

    // ---------------------------------------------------------------- vector

    /// The first `VADD` fixes the dimension and every one after it has to
    /// agree, because there is no create command to say it earlier.
    #[test]
    fn the_first_vadd_decides_how_wide_the_set_is() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]),
            ":1\r\n"
        );
        assert_eq!(f.run(&[b"VDIM", b"v"]), ":2\r\n");
        assert_eq!(f.run(&[b"VCARD", b"v"]), ":1\r\n");
        // A second vector under the same name replaces it and says so with a
        // zero, so an ingest can count what it created.
        assert_eq!(
            f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"east"]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"VCARD", b"v"]), ":1\r\n");
        // Three dimensions into a two dimensional set names both numbers, since
        // a client that gets this wrong needs to know which end is which.
        assert_eq!(
            f.run(&[b"VADD", b"v", b"VALUES", b"3", b"1", b"0", b"0", b"up"]),
            "-ERR Vector dimension mismatch - got 3 but set has 2\r\n"
        );
        // A vector of zeros has no direction, and it is taken anyway and comes
        // back as the origin, because that is what a real server does with it.
        assert_eq!(
            f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"0", b"nowhere"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"VEMB", b"v", b"nowhere"]),
            "*2\r\n$1\r\n0\r\n$1\r\n0\r\n"
        );
        // A set is made with one quantisation and keeps it, and a `VADD` that
        // names another is refused. Naming none names `Q8`, which is why this
        // set is a `Q8` one.
        assert_eq!(
            f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"1", b"other", b"BIN"]),
            "-ERR asked quantization mismatch with existing vector set\r\n"
        );
        // Nothing above created a key, and a set that never took a vector has
        // no dimension to report.
        assert_eq!(f.run(&[b"EXISTS", b"fresh"]), ":0\r\n");
        assert_eq!(f.run(&[b"VDIM", b"fresh"]), "-ERR key does not exist\r\n");
        assert_eq!(f.run(&[b"VCARD", b"fresh"]), ":0\r\n");
    }

    /// What a client sent comes back out, and what a client asked for is a
    /// similarity and not the distance underneath it.
    #[test]
    fn vemb_gives_back_the_vector_and_vsim_gives_back_a_similarity() {
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"3", b"4", b"a"]);
        // The set stored the direction and the length is multiplied back on the
        // way out, so this is `3 4` and not `0.6 0.8`. It is not quite `3 4`
        // either, because nobody named a quantisation and that means `Q8`: the
        // wider coordinate lands on a code exactly and the other one does not.
        // Both numbers are a real server's answers for the same input.
        assert_eq!(
            f.run(&[b"VEMB", b"v", b"a"]),
            "*2\r\n$17\r\n2.992125988006592\r\n$1\r\n4\r\n"
        );
        // NOQUANT is the way to ask for what went in to come back out.
        f.run(&[b"VADD", b"n", b"VALUES", b"2", b"3", b"4", b"a", b"NOQUANT"]);
        assert_eq!(
            f.run(&[b"VEMB", b"n", b"a"]),
            "*2\r\n$1\r\n3\r\n$1\r\n4\r\n"
        );
        // BIN keeps the signs and nothing else, and does not multiply the
        // length back on, since a sign has no length in it to scale.
        f.run(&[b"VADD", b"b", b"VALUES", b"2", b"3", b"-4", b"a", b"BIN"]);
        assert_eq!(
            f.run(&[b"VEMB", b"b", b"a"]),
            "*2\r\n$1\r\n1\r\n$2\r\n-1\r\n"
        );
        assert_eq!(f.run(&[b"VEMB", b"v", b"nobody"]), "*-1\r\n");
        assert_eq!(f.run(&[b"VEMB", b"nokey", b"a"]), "*-1\r\n");

        // On the axes, where the unit vector is exact and so is the dot
        // product, both ends of the scale come out exact: the same direction is
        // 1 and the opposite one is 0, with a right angle at a half.
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"3", b"0", b"a"]);
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"-1", b"0", b"opposite"]);
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"7", b"across"]);
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"VALUES", b"2", b"2", b"0", b"WITHSCORES"]),
            "*6\r\n$1\r\na\r\n$1\r\n1\r\n$6\r\nacross\r\n$3\r\n0.5\r\n\
             $8\r\nopposite\r\n$1\r\n0\r\n"
        );
        // A search from an element leaves that element out, since it is always
        // its own nearest neighbour.
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"ELE", b"a"]),
            "*2\r\n$6\r\nacross\r\n$8\r\nopposite\r\n"
        );
        // An element that is not there is an empty answer and not an error,
        // which is what a missing key gives too.
        assert_eq!(f.run(&[b"VSIM", b"v", b"ELE", b"nobody"]), "*0\r\n");
        assert_eq!(f.run(&[b"VSIM", b"nokey", b"ELE", b"a"]), "*0\r\n");
        // COUNT bounds it and TRUTH reads every vector rather than the codes,
        // which has to agree with the index on a set this small.
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"ELE", b"a", b"COUNT", b"1"]),
            "*1\r\n$6\r\nacross\r\n"
        );
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"ELE", b"a", b"TRUTH"]),
            "*2\r\n$6\r\nacross\r\n$8\r\nopposite\r\n"
        );
        // EF widens how much of the index is read and does not change how many
        // answers come back, so a wide search still returns what COUNT asked
        // for.
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"ELE", b"a", b"COUNT", b"1", b"EF", b"500"]),
            "*1\r\n$6\r\nacross\r\n"
        );

        // On RESP3 a scored search is a map, which is what the vector set
        // module replies and is not what ZRANGE does here.
        let mut g = Fixture::new();
        g.run(&[b"HELLO", b"3"]);
        g.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        assert_eq!(
            g.run(&[b"VSIM", b"v", b"VALUES", b"2", b"1", b"0", b"WITHSCORES"]),
            "%1\r\n$4\r\neast\r\n,1\r\n"
        );
    }

    /// The attribute pair, and the one reply that means two things.
    #[test]
    fn an_attribute_is_bytes_and_an_empty_one_takes_it_off() {
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        assert_eq!(f.run(&[b"VGETATTR", b"v", b"east"]), "$-1\r\n");
        assert_eq!(f.run(&[b"VSETATTR", b"v", b"east", b"{\"k\":1}"]), ":1\r\n");
        assert_eq!(f.run(&[b"VGETATTR", b"v", b"east"]), "$7\r\n{\"k\":1}\r\n");
        // Not parsed as JSON, because nothing reads into it yet and refusing a
        // write for a rule nothing enforces would be the wrong trade.
        assert_eq!(f.run(&[b"VSETATTR", b"v", b"east", b"not json"]), ":1\r\n");
        assert_eq!(f.run(&[b"VGETATTR", b"v", b"east"]), "$8\r\nnot json\r\n");
        // An empty string clears it, which is Redis's spelling of the removal.
        assert_eq!(f.run(&[b"VSETATTR", b"v", b"east", b""]), ":1\r\n");
        assert_eq!(f.run(&[b"VGETATTR", b"v", b"east"]), "$-1\r\n");
        // An element that is not there answers zero rather than being created,
        // since an attribute with no vector under it is not a thing this holds.
        assert_eq!(f.run(&[b"VSETATTR", b"v", b"nobody", b"{}"]), ":0\r\n");
        assert_eq!(f.run(&[b"VSETATTR", b"nokey", b"east", b"{}"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"nokey"]), ":0\r\n");
        // A null for an element with no attribute and a null for one that is
        // not there. VISMEMBER is how a client tells the two apart.
        assert_eq!(f.run(&[b"VGETATTR", b"v", b"nobody"]), "$-1\r\n");
        assert_eq!(f.run(&[b"VISMEMBER", b"v", b"east"]), ":1\r\n");
        assert_eq!(f.run(&[b"VISMEMBER", b"v", b"nobody"]), ":0\r\n");
        assert_eq!(f.run(&[b"VISMEMBER", b"nokey", b"east"]), ":0\r\n");

        // WITHATTRIBS carries it alongside the answers.
        f.run(&[b"VSETATTR", b"v", b"east", b"{\"k\":1}"]);
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"north"]);
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"VALUES", b"2", b"1", b"0", b"WITHATTRIBS"]),
            "*4\r\n$4\r\neast\r\n$7\r\n{\"k\":1}\r\n$5\r\nnorth\r\n$-1\r\n"
        );
    }

    /// The slot a removed element had is reused, and nothing that was beside it
    /// comes back with the next element to get it.
    #[test]
    fn vrem_takes_the_attribute_with_it() {
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        f.run(&[b"VSETATTR", b"v", b"east", b"{\"k\":1}"]);
        assert_eq!(f.run(&[b"VREM", b"v", b"east"]), ":1\r\n");
        assert_eq!(f.run(&[b"VREM", b"v", b"east"]), ":0\r\n");
        assert_eq!(f.run(&[b"VREM", b"nokey", b"east"]), ":0\r\n");
        // The key went with the last element, the way every other collection
        // here works.
        assert_eq!(f.run(&[b"EXISTS", b"v"]), ":0\r\n");

        // The next element is given the slot the removed one had, and it comes
        // with no attribute on it.
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        f.run(&[b"VSETATTR", b"v", b"east", b"{\"k\":1}"]);
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"north"]);
        f.run(&[b"VREM", b"v", b"east"]);
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"1", b"between"]);
        assert_eq!(f.run(&[b"VGETATTR", b"v", b"between"]), "$-1\r\n");
    }

    /// `VINFO` says what the index is before it says anything a client could
    /// mistake for a graph.
    #[test]
    fn vinfo_says_partition_first() {
        let mut f = Fixture::new();
        f.run(&[
            b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east", b"M", b"32",
        ]);
        f.run(&[b"VSETATTR", b"v", b"east", b"{}"]);
        let info = f.run(&[b"VINFO", b"v"]);
        assert!(info.starts_with("*24\r\n$10\r\nindex-type\r\n$9\r\npartition\r\n"));
        // What the client asked for and not what happened to the tuning, which
        // is `10` section 7: M is recorded and changes nothing.
        assert!(info.contains("$6\r\nhnsw-m\r\n:32\r\n"), "{info}");
        assert!(info.contains("$10\r\nvector-dim\r\n:2\r\n"), "{info}");
        assert!(info.contains("$16\r\nattributes-count\r\n:1\r\n"), "{info}");
        // Nobody named a quantisation, so this set is a `Q8` one and every
        // element in it is stored that way.
        assert!(
            info.contains("$10\r\nquant-type\r\n$4\r\nint8\r\n"),
            "{info}"
        );
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"north", b"BIN"]);
        assert!(
            f.run(&[b"VINFO", b"v"])
                .contains("$10\r\nquant-type\r\n$3\r\nbin\r\n")
        );
        assert_eq!(f.run(&[b"VINFO", b"nokey"]), "$-1\r\n");
    }

    /// A set to read ranges of names out of.
    fn named() -> Fixture {
        let mut f = Fixture::new();
        for (i, name) in ["alpha", "beta", "gamma", "delta", "epsilon"]
            .iter()
            .enumerate()
        {
            let x = (i + 1).to_string();
            f.run(&[
                b"VADD",
                b"r",
                b"VALUES",
                b"2",
                x.as_bytes(),
                b"1",
                name.as_bytes(),
            ]);
        }
        f
    }

    /// `VRANGE` reads the names in the order bytes come in and pays no
    /// attention to where the vectors point.
    #[test]
    fn vrange_walks_the_names_and_not_the_vectors() {
        let mut f = named();
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"-", b"+"]),
            "*5\r\n$5\r\nalpha\r\n$4\r\nbeta\r\n$5\r\ndelta\r\n$7\r\nepsilon\r\n$5\r\ngamma\r\n"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"[a", b"[d"]),
            "*2\r\n$5\r\nalpha\r\n$4\r\nbeta\r\n",
            "the high end is a name and not a prefix, so delta is past it"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"(alpha", b"(gamma"]),
            "*3\r\n$4\r\nbeta\r\n$5\r\ndelta\r\n$7\r\nepsilon\r\n"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"[beta", b"[beta"]),
            "*1\r\n$4\r\nbeta\r\n"
        );
        assert_eq!(f.run(&[b"VRANGE", b"r", b"[z", b"+"]), "*0\r\n");
        // Bytes and not letters, so an upper case name sorts before every lower
        // case one rather than beside its own spelling.
        f.run(&[b"VADD", b"r", b"VALUES", b"2", b"1", b"1", b"Beta"]);
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"-", b"[beta"]),
            "*3\r\n$4\r\nBeta\r\n$5\r\nalpha\r\n$4\r\nbeta\r\n"
        );
        assert_eq!(f.run(&[b"VRANGE", b"nokey", b"-", b"+"]), "*0\r\n");
    }

    /// The count cuts the answer after the range is decided, and zero is not
    /// the same as leaving it out.
    #[test]
    fn a_vrange_count_of_zero_asks_for_nothing() {
        let mut f = named();
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"-", b"+", b"2"]),
            "*2\r\n$5\r\nalpha\r\n$4\r\nbeta\r\n"
        );
        assert_eq!(f.run(&[b"VRANGE", b"r", b"-", b"+", b"0"]), "*0\r\n");
        assert!(
            f.run(&[b"VRANGE", b"r", b"-", b"+", b"-1"])
                .starts_with("*5\r\n"),
            "a negative count is no limit at all"
        );
    }

    /// Both ends are read before either is placed, and the count is read before
    /// either end.
    #[test]
    fn vrange_says_which_end_it_could_not_read() {
        let mut f = named();
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"x", b"y"]),
            "-ERR invalid start range format\r\n"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"+", b"x"]),
            "-ERR invalid end range format\r\n",
            "the high end is spelled wrong, which is worth saying before the \
             low end being on the wrong side"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"+", b"-"]),
            "-ERR '-' can only be used as first argument, '+' only as second\r\n"
        );
        // A bracket with nothing after it is not the empty name here, though an
        // element really can be called that.
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"[", b"+"]),
            "-ERR invalid start range format\r\n"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"x", b"+", b"z"]),
            "-ERR invalid COUNT value\r\n"
        );
        assert_eq!(
            f.run(&[b"VRANGE", b"r", b"-", b"+", b"2", b"extra"]),
            "-ERR wrong number of arguments for 'VRANGE' command\r\n"
        );
        f.run(&[b"SET", b"s", b"x"]);
        assert!(
            f.run(&[b"VRANGE", b"s", b"-", b"+"])
                .starts_with("-WRONGTYPE")
        );
    }

    /// The option that asks for something this index does not have says so
    /// rather than doing something else quietly.
    #[test]
    fn reduce_is_refused_and_not_ignored() {
        let mut f = Fixture::new();
        let reduce = f.run(&[
            b"VADD", b"v", b"REDUCE", b"1", b"VALUES", b"2", b"1", b"0", b"east",
        ]);
        assert!(
            reduce.starts_with("-ERR REDUCE is not supported."),
            "{reduce}"
        );
        assert_eq!(f.run(&[b"EXISTS", b"v"]), ":0\r\n");
    }

    /// A filtered search answers with the nearest elements that match, and an
    /// expression that is not one is an error before the key is looked at.
    #[test]
    fn vsim_filter_reads_the_attributes() {
        let mut f = Fixture::new();
        for (name, x, y, attr) in [
            ("a", "1", "0", r#"{"lang":"en","year":1999}"#),
            ("b", "9", "1", r#"{"lang":"fr","year":2005}"#),
            ("c", "8", "2", r#"{"lang":"en","year":1970}"#),
            ("d", "7", "3", r#"{"lang":"en","year":2020}"#),
        ] {
            f.run(&[
                b"VADD",
                b"v",
                b"VALUES",
                b"2",
                x.as_bytes(),
                y.as_bytes(),
                name.as_bytes(),
                b"SETATTR",
                attr.as_bytes(),
            ]);
        }
        // `b` is the nearest to the query and is the one the filter drops, so
        // this is the answer a filter applied afterwards would have got wrong.
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"COUNT",
                b"2",
                b"FILTER",
                b".lang == \"en\"",
            ]),
            "*2\r\n$1\r\na\r\n$1\r\nc\r\n"
        );
        // A number is compared as a number, and the two halves of an `and` both
        // have to hold.
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"FILTER",
                b".lang == 'en' and .year > 1980",
            ]),
            "*2\r\n$1\r\na\r\n$1\r\nd\r\n"
        );
        // A list, and a field an element does not have.
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"FILTER",
                b".lang in ['fr', 'de']",
            ]),
            "*1\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"FILTER",
                b".rating > 3"
            ]),
            "*0\r\n"
        );
        // TRUTH measures every vector, and the filter still decides which ones
        // are measured.
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"TRUTH",
                b"FILTER",
                b".year < 1980",
            ]),
            "*1\r\n$1\r\nc\r\n"
        );
        // VSETATTR moves an element in and out of a filter, which means the tag
        // beside its code was rewritten and not just the string.
        f.run(&[b"VSETATTR", b"v", b"b", r#"{"lang":"en"}"#.as_bytes()]);
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"COUNT",
                b"1",
                b"FILTER",
                b".lang == \"en\"",
            ]),
            "*1\r\n$1\r\nb\r\n"
        );
        // And a VADD that replaces the vector keeps the attribute and the tag,
        // which is the same rewrite from the other end.
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"9", b"2", b"b"]);
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"COUNT",
                b"1",
                b"FILTER",
                b".lang == \"en\"",
            ]),
            "*1\r\n$1\r\nb\r\n"
        );

        // The expression is parsed before the key is read, so a bad one is an
        // error whether or not the key is there.
        let bad = f.run(&[b"VSIM", b"nokey", b"ELE", b"e", b"FILTER", b".k =="]);
        assert_eq!(bad, "-ERR invalid FILTER expression\r\n");
        assert_eq!(
            f.run(&[b"VSIM", b"v", b"ELE", b"a", b"FILTER", b"junk"]),
            "-ERR invalid FILTER expression\r\n"
        );
        // FILTER-EF raises the effort rather than capping it, and zero is
        // Redis's word for no limit, so neither is an error.
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"COUNT",
                b"1",
                b"FILTER-EF",
                b"500",
                b"FILTER",
                b".lang == 'en'",
            ]),
            "*1\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"COUNT",
                b"1",
                b"FILTER-EF",
                b"0"
            ]),
            "*1\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[
                b"VSIM",
                b"v",
                b"VALUES",
                b"2",
                b"9",
                b"1",
                b"FILTER-EF",
                b"lots"
            ]),
            "-ERR EF must be a positive integer\r\n"
        );
    }

    /// A vector set key is a key, so the keyspace owns it the way it owns every
    /// other one and none of those commands know what is inside it.
    #[test]
    fn the_keyspace_sees_a_vector_set_key_like_any_other() {
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        assert_eq!(f.run(&[b"TYPE", b"v"]), "+vectorset\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"v"]), ":1\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"v"]), "$6\r\nrabitq\r\n");
        assert_eq!(f.run(&[b"KEYS", b"*"]), "*1\r\n$1\r\nv\r\n");
        assert_eq!(f.run(&[b"DBSIZE"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXPIRE", b"v", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"TTL", b"v"]), ":100\r\n");
        assert_eq!(f.run(&[b"PERSIST", b"v"]), ":1\r\n");
        assert_eq!(f.run(&[b"DEL", b"v"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", b"v"]), ":0\r\n");

        // And the wrong type is the wrong type in both directions.
        f.run(&[b"SET", b"s", b"1"]);
        assert_eq!(
            f.run(&[b"VADD", b"s", b"VALUES", b"2", b"1", b"0", b"east"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(
            f.run(&[b"VCARD", b"s"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        assert_eq!(
            f.run(&[b"GET", b"v"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        // A graph and a vector set share the escape in the record tag and are
        // still two different types, which is the case the tag alone cannot
        // decide.
        f.run(&[b"G.NADD", b"social", b"ada"]);
        assert_eq!(
            f.run(&[b"VCARD", b"social"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(
            f.run(&[b"G.NGET", b"v", b"ada"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
    }

    /// `VRANDMEMBER` is `SRANDMEMBER` over the element names, in both of its
    /// shapes, off the database's own generator.
    #[test]
    fn vrandmember_has_the_two_shapes_srandmember_has() {
        let mut f = Fixture::new();
        for (i, name) in [&b"a"[..], b"b", b"c"].iter().enumerate() {
            let x = (i + 1).to_string();
            f.run(&[b"VADD", b"v", b"VALUES", b"2", x.as_bytes(), b"1", name]);
        }
        // One element is a bulk string and not an array of one.
        let one = f.run(&[b"VRANDMEMBER", b"v"]);
        assert!(one.starts_with("$1\r\n"), "{one}");
        // A positive count is distinct and stops at the size of the set.
        let mut all = f.run(&[b"VRANDMEMBER", b"v", b"9"]);
        assert!(all.starts_with("*3\r\n"), "{all}");
        for name in ["a", "b", "c"] {
            assert!(all.contains(name), "{all} is missing {name}");
        }
        all = f.run(&[b"VRANDMEMBER", b"v", b"2"]);
        assert!(all.starts_with("*2\r\n"), "{all}");
        // A negative one draws that many and allows repeats.
        let many = f.run(&[b"VRANDMEMBER", b"v", b"-5"]);
        assert!(many.starts_with("*5\r\n"), "{many}");
        // A key that is not there answers the shape that was asked for.
        assert_eq!(f.run(&[b"VRANDMEMBER", b"nokey"]), "$-1\r\n");
        assert_eq!(f.run(&[b"VRANDMEMBER", b"nokey", b"3"]), "*0\r\n");
    }

    /// `VLINKS` answers about the index that is here rather than the graph that
    /// is not, which is D-2.
    #[test]
    fn vlinks_reports_one_layer_of_partition_neighbours() {
        let mut f = Fixture::new();
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"east"]);
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"north"]);
        // One layer deep, because the index is one layer deep, so a client
        // walking layers gets a short list and not a shape it cannot parse.
        assert_eq!(
            f.run(&[b"VLINKS", b"v", b"east"]),
            "*1\r\n*1\r\n$5\r\nnorth\r\n"
        );
        assert_eq!(
            f.run(&[b"VLINKS", b"v", b"east", b"WITHSCORES"]),
            "*1\r\n*2\r\n$5\r\nnorth\r\n$3\r\n0.5\r\n"
        );
        assert_eq!(f.run(&[b"VLINKS", b"v", b"nobody"]), "*-1\r\n");
        assert_eq!(f.run(&[b"VLINKS", b"nokey", b"east"]), "*-1\r\n");
    }

    /// A vector arrives either as digits or as bytes, and the two have to mean
    /// the same thing.
    #[test]
    fn fp32_and_values_are_the_same_vector() {
        let mut f = Fixture::new();
        let mut blob = Vec::new();
        for x in [3.0f32, 4.0] {
            blob.extend_from_slice(&x.to_le_bytes());
        }
        assert_eq!(f.run(&[b"VADD", b"v", b"FP32", &blob, b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"VDIM", b"v"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"VEMB", b"v", b"a"]),
            "*2\r\n$17\r\n2.992125988006592\r\n$1\r\n4\r\n"
        );
        // RAW is the stored bytes and the numbers that turn them back into the
        // client's vector, which for `Q8` is a code a coordinate, the length the
        // vector arrived with and the scale the codes are measured against. The
        // name of the form is a simple string, which is a real server's shape,
        // and all four of these are a real server's answers.
        assert_eq!(
            f.run(&[b"VEMB", b"v", b"a", b"RAW"]),
            "*4\r\n+int8\r\n$2\r\n_\x7f\r\n$1\r\n5\r\n$17\r\n0.800000011920929\r\n"
        );
        // A blob that is not a whole number of floats is not a vector.
        assert_eq!(
            f.run(&[b"VADD", b"w", b"FP32", b"abc", b"a"]),
            "-ERR invalid vector specification\r\n"
        );
        // Neither is a count that promises more than arrived.
        assert_eq!(
            f.run(&[b"VADD", b"w", b"VALUES", b"4", b"1", b"0", b"a"]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"w"]), ":0\r\n");
    }

    // ----------------------------------------------------------------- bloom

    /// The filter a client gets when it does not describe one, and the two
    /// answers an add can give.
    #[test]
    fn bf_add_makes_the_filter_and_says_whether_it_was_new() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"BF.ADD", b"b", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"BF.ADD", b"b", b"hello"]), ":0\r\n");
        assert_eq!(f.run(&[b"BF.EXISTS", b"b", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"BF.EXISTS", b"b", b"never"]), ":0\r\n");
        assert_eq!(f.run(&[b"BF.CARD", b"b"]), ":1\r\n");
        // The defaults are the module's configs and not anything the command
        // said, which is 100 entries at a hundredth and a growth of 2.
        assert_eq!(
            f.run(&[b"BF.INFO", b"b"]),
            "*10\r\n+Capacity\r\n:100\r\n+Size\r\n:240\r\n\
             +Number of filters\r\n:1\r\n+Number of items inserted\r\n:1\r\n\
             +Expansion rate\r\n:2\r\n"
        );
        assert_eq!(f.run(&[b"TYPE", b"b"]), "+MBbloom--\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"b"]), "$3\r\nraw\r\n");
        // A key that is not there has no filter to report on, and answers two
        // different ways about it depending on which command asked.
        assert_eq!(f.run(&[b"BF.CARD", b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"BF.INFO", b"gone"]), "-ERR not found\r\n");
    }

    /// `BF.EXISTS` on a key holding something else answers a miss, and
    /// everything else in the family answers `WRONGTYPE`.
    ///
    /// The two halves of a check and set disagree about what that key is, which
    /// is the module's behaviour and not a decision taken here.
    #[test]
    fn a_wrong_type_is_a_miss_to_the_two_that_only_read_bits() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"text"]);
        assert_eq!(f.run(&[b"BF.EXISTS", b"s", b"x"]), ":0\r\n");
        assert_eq!(f.run(&[b"BF.MEXISTS", b"s", b"x"]), "*1\r\n:0\r\n");
        for cmd in [
            vec![&b"BF.ADD"[..], b"s", b"x"],
            vec![&b"BF.MADD"[..], b"s", b"x"],
            vec![&b"BF.CARD"[..], b"s"],
            vec![&b"BF.INFO"[..], b"s"],
            vec![&b"BF.DEBUG"[..], b"s"],
            vec![&b"BF.SCANDUMP"[..], b"s", b"0"],
        ] {
            let name = String::from_utf8_lossy(cmd[0]).into_owned();
            assert!(f.run(&cmd).starts_with("-WRONGTYPE"), "{name}");
        }
        // The arguments are read before the key is, so a reserve with a bad
        // error rate complains about the rate and never learns about the string.
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"s", b"abc", b"10"]),
            "-ERR bad error rate\r\n"
        );
        assert!(
            f.run(&[b"BF.RESERVE", b"s", b"0.01", b"10"])
                .starts_with("-WRONGTYPE")
        );
    }

    /// A chain grows by its expansion factor and each link is half as wrong as
    /// the one before, which is what makes the whole filter hold its rate.
    #[test]
    fn a_full_filter_grows_a_link_and_a_fixed_one_says_no() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"BF.RESERVE", b"g", b"0.01", b"10"]), "+OK\r\n");
        for i in 0..10u32 {
            assert_eq!(
                f.run(&[b"BF.ADD", b"g", i.to_string().as_bytes()]),
                ":1\r\n"
            );
        }
        assert_eq!(f.run(&[b"BF.INFO", b"g", b"FILTERS"]), "*1\r\n:1\r\n");
        assert_eq!(f.run(&[b"BF.ADD", b"g", b"11"]), ":1\r\n");
        assert_eq!(f.run(&[b"BF.INFO", b"g", b"filters"]), "*1\r\n:2\r\n");
        // Capacity is the sum of every link and not the number that was asked
        // for, so it is 10 and then 10 plus 20.
        assert_eq!(f.run(&[b"BF.INFO", b"g", b"CAPACITY"]), "*1\r\n:30\r\n");
        assert_eq!(
            f.run(&[b"BF.DEBUG", b"g"]),
            "*3\r\n$7\r\nsize:11\r\n\
             $71\r\nbytes:16 bits:128 hashes:8 hashwidth:64 capacity:10 size:10 ratio:0.005\r\n\
             $71\r\nbytes:32 bits:256 hashes:9 hashwidth:64 capacity:20 size:1 ratio:0.0025\r\n"
        );

        // The same filter told not to grow fills instead.
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"n", b"0.01", b"2", b"NONSCALING"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"BF.ADD", b"n", b"a"]), ":1\r\n");
        assert_eq!(f.run(&[b"BF.ADD", b"n", b"b"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"BF.ADD", b"n", b"c"]),
            "-ERR non scaling filter is full\r\n"
        );
        // And an item that is already in it still answers, because membership
        // is checked before fullness.
        assert_eq!(f.run(&[b"BF.ADD", b"n", b"a"]), ":0\r\n");
        // A filter that will not grow has no expansion rate to report, in
        // either of the two spellings that make one.
        assert_eq!(f.run(&[b"BF.INFO", b"n", b"EXPANSION"]), "*1\r\n$-1\r\n");
        f.run(&[b"BF.RESERVE", b"z", b"0.01", b"2", b"EXPANSION", b"0"]);
        assert_eq!(f.run(&[b"BF.INFO", b"z", b"EXPANSION"]), "*1\r\n$-1\r\n");
        // Asking for both at once is refused, which is one of the module's
        // errors that carries no prefix at all.
        assert_eq!(
            f.run(&[
                b"BF.RESERVE",
                b"q",
                b"0.01",
                b"2",
                b"NONSCALING",
                b"EXPANSION",
                b"2"
            ]),
            "-Nonscaling filters cannot expand\r\n"
        );
    }

    /// A multi add stops where the filter did, so the reply can be shorter than
    /// the argument list.
    #[test]
    fn madd_truncates_its_reply_at_the_item_that_did_not_fit() {
        let mut f = Fixture::new();
        f.run(&[b"BF.RESERVE", b"n", b"0.01", b"2", b"NONSCALING"]);
        assert_eq!(
            f.run(&[b"BF.MADD", b"n", b"a", b"b", b"c", b"d"]),
            "*3\r\n:1\r\n:1\r\n-ERR non scaling filter is full\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.MEXISTS", b"n", b"a", b"c"]),
            "*2\r\n:1\r\n:0\r\n"
        );
    }

    /// `BF.INSERT` describes a filter and fills it in one command, with its own
    /// spelling of every complaint.
    #[test]
    fn insert_is_a_reserve_and_a_madd_with_different_errors() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"BF.INSERT",
                b"i",
                b"CAPACITY",
                b"50",
                b"ERROR",
                b"0.001",
                b"ITEMS",
                b"a",
                b"b"
            ]),
            "*2\r\n:1\r\n:1\r\n"
        );
        assert_eq!(f.run(&[b"BF.INFO", b"i", b"CAPACITY"]), "*1\r\n:50\r\n");
        // NOCREATE is the only way to add without making the key.
        assert_eq!(
            f.run(&[b"BF.INSERT", b"gone", b"NOCREATE", b"ITEMS", b"a"]),
            "-ERR not found\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"gone"]), ":0\r\n");
        // The same mistakes as BF.RESERVE, in the sentences this command uses
        // for them, and one sentence where BF.RESERVE has two.
        assert_eq!(
            f.run(&[b"BF.INSERT", b"i", b"CAPACITY", b"abc", b"ITEMS", b"a"]),
            "-Bad capacity\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.INSERT", b"i", b"ERROR", b"2", b"ITEMS", b"a"]),
            "-Bad error rate\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.INSERT", b"i", b"EXPANSION", b"99999", b"ITEMS", b"a"]),
            "-Bad expansion\r\n"
        );
        // An option is matched on its first letter and not on the word, so a
        // token nobody meant as an option is one anyway if it starts with the
        // right letter. NOSUCHTHING is NONSCALING here, and the filter it
        // builds says so.
        assert_eq!(
            f.run(&[b"BF.INSERT", b"ns", b"NOSUCHTHING", b"ITEMS", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(f.run(&[b"BF.INFO", b"ns", b"EXPANSION"]), "*1\r\n$-1\r\n");
        // Only E and N need a second look, one for ERROR against EXPANSION and
        // the other for NOCREATE against NONSCALING, and both stop as soon as
        // they can tell the two apart.
        assert_eq!(
            f.run(&[b"BF.INSERT", b"e1", b"E", b"4", b"ITEMS", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(f.run(&[b"BF.INFO", b"e1", b"EXPANSION"]), "*1\r\n:4\r\n");
        assert_eq!(
            f.run(&[b"BF.INSERT", b"e2", b"ER", b"0.5", b"ITEMS", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.INSERT", b"gone", b"NOC", b"ITEMS", b"a"]),
            "-ERR not found\r\n"
        );
        // A letter that starts nothing is the one case that is refused.
        assert_eq!(
            f.run(&[b"BF.INSERT", b"i", b"ZZZ", b"ITEMS", b"a"]),
            "-Unknown argument received\r\n"
        );
        // Everything after ITEMS is an item, even when it spells an option.
        assert_eq!(
            f.run(&[b"BF.INSERT", b"i", b"ITEMS", b"NOCREATE"]),
            "*1\r\n:1\r\n"
        );
        // And ITEMS with nothing after it is the same as leaving it out.
        assert!(
            f.run(&[b"BF.INSERT", b"i", b"ITEMS"])
                .contains("wrong number of arguments")
        );
    }

    /// A filter dumped a chunk at a time and put back into another key is the
    /// same filter.
    #[test]
    fn a_dump_replays_into_a_filter_that_answers_the_same() {
        let mut f = Fixture::new();
        f.run(&[b"BF.RESERVE", b"src", b"0.01", b"10"]);
        for i in 0..25u32 {
            f.run(&[b"BF.ADD", b"src", i.to_string().as_bytes()]);
        }
        assert_eq!(f.run(&[b"BF.INFO", b"src", b"FILTERS"]), "*1\r\n:2\r\n");

        // Iterator zero asks for the header and every one after it is a running
        // byte offset, and a chunk never spans two links.
        let mut iter = b"0".to_vec();
        let mut chunks = 0;
        loop {
            let raw = f.raw(&[b"BF.SCANDUMP", b"src", &iter]);
            let text = String::from_utf8_lossy(&raw).into_owned();
            let next = text
                .split("\r\n")
                .nth(1)
                .and_then(|n| n.strip_prefix(':'))
                .expect("a two element reply of an iterator and a chunk")
                .to_owned();
            let body = &raw[raw.iter().position(|&b| b == b'$').expect("a bulk chunk")..];
            let data = &body[body
                .windows(2)
                .position(|w| w == b"\r\n")
                .expect("a length line")
                + 2..body.len() - 2];
            if next == "0" {
                assert!(data.is_empty(), "the last chunk is empty");
                break;
            }
            let put = f.run(&[b"BF.LOADCHUNK", b"dst", next.as_bytes(), data]);
            assert_eq!(put, "+OK\r\n", "loading chunk {chunks}");
            iter = next.into_bytes();
            chunks += 1;
        }
        assert_eq!(chunks, 3, "a header and one chunk per link");

        assert_eq!(f.run(&[b"BF.INFO", b"dst"]), f.run(&[b"BF.INFO", b"src"]));
        assert_eq!(f.run(&[b"BF.DEBUG", b"dst"]), f.run(&[b"BF.DEBUG", b"src"]));
        for i in 0..25u32 {
            assert_eq!(
                f.run(&[b"BF.EXISTS", b"dst", i.to_string().as_bytes()]),
                ":1\r\n"
            );
        }

        // A header on top of a filter is refused rather than merged, and so is
        // one that no filter wrote.
        assert_eq!(
            f.run(&[b"BF.LOADCHUNK", b"dst", b"1", b"anything"]),
            "-ERR received bad data\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.LOADCHUNK", b"fresh", b"1", b"anything"]),
            "-ERR received bad data\r\n"
        );
        // An offset past the end of the filter names itself.
        assert_eq!(
            f.run(&[b"BF.LOADCHUNK", b"dst", b"99999", b"x"]),
            "-ERR invalid offset - no link found\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.LOADCHUNK", b"dst", b"nope", b"x"]),
            "-ERR Second argument must be numeric\r\n"
        );
        // The same complaint without the prefix on the way out, which is the
        // module's inconsistency and not a slip here.
        assert_eq!(
            f.run(&[b"BF.SCANDUMP", b"src", b"nope"]),
            "-Second argument must be numeric\r\n"
        );
    }

    /// The argument checks, which have a sentence each and read numbers the way
    /// Redis reads them everywhere else.
    #[test]
    fn reserve_reads_its_numbers_the_way_string2ll_does() {
        let mut f = Fixture::new();
        for (args, want) in [
            (vec![&b"abc"[..], b"10"], "-ERR bad error rate\r\n"),
            (vec![&b"nan"[..], b"10"], "-ERR bad error rate\r\n"),
            (
                vec![&b"0"[..], b"10"],
                "-ERR error rate must be in the range (0.000000, 1.000000)\r\n",
            ),
            (
                vec![&b"1"[..], b"10"],
                "-ERR error rate must be in the range (0.000000, 1.000000)\r\n",
            ),
            (
                vec![&b"inf"[..], b"10"],
                "-ERR error rate must be in the range (0.000000, 1.000000)\r\n",
            ),
            (vec![&b"0.01"[..], b"+10"], "-ERR bad capacity\r\n"),
            (vec![&b"0.01"[..], b"1e2"], "-ERR bad capacity\r\n"),
            (vec![&b"0.01"[..], b"007"], "-ERR bad capacity\r\n"),
            (
                vec![&b"0.01"[..], b"0"],
                "-ERR capacity must be in the range [1, 1073741824]\r\n",
            ),
            (
                vec![&b"0.01"[..], b"1073741825"],
                "-ERR capacity must be in the range [1, 1073741824]\r\n",
            ),
        ] {
            let mut cmd = vec![&b"BF.RESERVE"[..], b"k"];
            cmd.extend(args.iter().copied());
            assert_eq!(f.run(&cmd), want, "{}", String::from_utf8_lossy(args[0]));
        }
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"k", b"0.01", b"10", b"EXPANSION"]),
            "-ERR no expansion\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"k", b"0.01", b"10", b"EXPANSION", b"abc"]),
            "-ERR bad expansion\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"k", b"0.01", b"10", b"EXPANSION", b"32769"]),
            "-ERR expansion must be in the range [0, 32768]\r\n"
        );
        // Trailing rubbish after the capacity is ignored rather than refused.
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"k", b"0.01", b"10", b"junk"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.RESERVE", b"k", b"0.01", b"10"]),
            "-ERR item exists\r\n"
        );
        assert_eq!(
            f.run(&[b"BF.INFO", b"k", b"nosuchfield"]),
            "-Invalid information value\r\n"
        );
        assert!(
            f.run(&[b"BF.INFO", b"k", b"CAPACITY", b"SIZE"])
                .contains("wrong number of arguments")
        );
    }

    /// The RESP3 shapes, which are where this family differs most from RESP2.
    #[test]
    fn the_bloom_family_answers_in_resp3_spelling_too() {
        let mut f = Fixture::new();
        f.out.set_proto(Proto::Resp3);
        assert_eq!(f.run(&[b"BF.ADD", b"b", b"a"]), "#t\r\n");
        assert_eq!(f.run(&[b"BF.ADD", b"b", b"a"]), "#f\r\n");
        assert_eq!(f.run(&[b"BF.MADD", b"b", b"a", b"c"]), "*2\r\n#f\r\n#t\r\n");
        assert_eq!(f.run(&[b"BF.EXISTS", b"b", b"a"]), "#t\r\n");
        assert_eq!(
            f.run(&[b"BF.MEXISTS", b"b", b"a", b"z"]),
            "*2\r\n#t\r\n#f\r\n"
        );
        // The count stays an integer, because it counts rather than answers.
        assert_eq!(f.run(&[b"BF.CARD", b"b"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"BF.INFO", b"b"]),
            "%5\r\n+Capacity\r\n:100\r\n+Size\r\n:240\r\n\
             +Number of filters\r\n:1\r\n+Number of items inserted\r\n:2\r\n\
             +Expansion rate\r\n:2\r\n"
        );
        // One field is a map of one here and a bare array of one on RESP2, so
        // this is the reply where the two protocols carry different facts.
        assert_eq!(
            f.run(&[b"BF.INFO", b"b", b"CAPACITY"]),
            "%1\r\n+Capacity\r\n:100\r\n"
        );
    }

    // ---------------------------------------------------------------- cuckoo

    /// A dump header, which is the four counts and the three widths a filter
    /// writes in front of its fingerprints.
    ///
    /// Written by hand rather than taken from a `CF.SCANDUMP`, because what the
    /// tests below want out of it is the states a filter cannot be put into
    /// from the wire.
    fn cf_header(
        items: u64,
        buckets: u64,
        deletes: u64,
        filters: u64,
        geometry: [u16; 3],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(38);
        for n in [items, buckets, deletes, filters] {
            out.extend_from_slice(&n.to_le_bytes());
        }
        for n in geometry {
            out.extend_from_slice(&n.to_le_bytes());
        }
        out
    }

    /// The filter a client gets when it does not describe one, and the thing a
    /// cuckoo filter does that a Bloom filter cannot, which is count copies and
    /// take them out again.
    #[test]
    fn cf_add_makes_the_filter_and_counts_the_copies() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"CF.ADD", b"d", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.ADD", b"d", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.COUNT", b"d", b"hello"]), ":2\r\n");
        // The NX form is the one that looks first, which is why it is a command
        // of its own rather than an option.
        assert_eq!(f.run(&[b"CF.ADDNX", b"d", b"hello"]), ":0\r\n");
        assert_eq!(f.run(&[b"CF.ADDNX", b"d", b"other"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.EXISTS", b"d", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.EXISTS", b"d", b"no"]), ":0\r\n");
        assert_eq!(
            f.run(&[b"CF.MEXISTS", b"d", b"hello", b"no"]),
            "*2\r\n:1\r\n:0\r\n"
        );
        // The defaults are the module's configs: 1024 entries over buckets of
        // two, twenty kicks and a chain that grows by one.
        assert_eq!(
            f.run(&[b"CF.INFO", b"d"]),
            "*16\r\n+Size\r\n:1080\r\n+Number of buckets\r\n:512\r\n\
             +Number of filters\r\n:1\r\n+Number of items inserted\r\n:3\r\n\
             +Number of items deleted\r\n:0\r\n+Bucket size\r\n:2\r\n\
             +Expansion rate\r\n:1\r\n+Max iterations\r\n:20\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.DEBUG", b"d"]),
            "$79\r\nbktsize:2 buckets:512 items:3 deletes:0 filters:1 \
             max_iterations:20 expansion:1\r\n"
        );
        assert_eq!(f.run(&[b"TYPE", b"d"]), "+MBbloomCF\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"d"]), "$3\r\nraw\r\n");

        // A delete takes one copy, so the same item goes twice and then stops.
        assert_eq!(f.run(&[b"CF.DEL", b"d", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.COUNT", b"d", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.DEL", b"d", b"hello"]), ":1\r\n");
        assert_eq!(f.run(&[b"CF.DEL", b"d", b"hello"]), ":0\r\n");
        assert_eq!(f.run(&[b"CF.COMPACT", b"d"]), "+OK\r\n");

        // A key with no filter under it gets three different sentences and one
        // plain miss, depending on which command asked.
        assert_eq!(f.run(&[b"CF.INFO", b"gone"]), "-ERR not found\r\n");
        assert_eq!(f.run(&[b"CF.DEL", b"gone", b"x"]), "-Not found\r\n");
        assert_eq!(
            f.run(&[b"CF.COMPACT", b"gone"]),
            "-Cuckoo filter was not found\r\n"
        );
        assert_eq!(f.run(&[b"CF.EXISTS", b"gone", b"x"]), ":0\r\n");
        // And `CF.COMPACT` is declared as taking any number of keys and takes
        // exactly one, which is the module's own arity being wrong rather than
        // this table's.
        assert!(
            f.run(&[b"CF.COMPACT", b"a", b"b"])
                .contains("wrong number of arguments")
        );
    }

    /// The four that only read fingerprints treat a key holding something else
    /// as a key with no filter, and everything else answers `WRONGTYPE`.
    #[test]
    fn a_wrong_type_is_a_miss_to_the_four_that_only_read_fingerprints() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"text"]);
        assert_eq!(f.run(&[b"CF.EXISTS", b"s", b"x"]), ":0\r\n");
        assert_eq!(f.run(&[b"CF.MEXISTS", b"s", b"x"]), "*1\r\n:0\r\n");
        assert_eq!(f.run(&[b"CF.COUNT", b"s", b"x"]), ":0\r\n");
        // `CF.DEL` writes and is still in that group, and `CF.COMPACT` writes
        // and is declared read only, so neither of the two halves of the family
        // is the same set as the flags say.
        assert_eq!(f.run(&[b"CF.DEL", b"s", b"x"]), "-Not found\r\n");
        assert_eq!(
            f.run(&[b"CF.COMPACT", b"s"]),
            "-Cuckoo filter was not found\r\n"
        );
        for cmd in [
            vec![&b"CF.ADD"[..], b"s", b"x"],
            vec![&b"CF.ADDNX"[..], b"s", b"x"],
            vec![&b"CF.INSERT"[..], b"s", b"ITEMS", b"x"],
            vec![&b"CF.INSERTNX"[..], b"s", b"ITEMS", b"x"],
            vec![&b"CF.INFO"[..], b"s"],
            vec![&b"CF.DEBUG"[..], b"s"],
            vec![&b"CF.SCANDUMP"[..], b"s", b"0"],
            vec![&b"CF.LOADCHUNK"[..], b"s", b"2", b"x"],
            vec![&b"CF.RESERVE"[..], b"s", b"64"],
        ] {
            let name = String::from_utf8_lossy(cmd[0]).into_owned();
            assert!(f.run(&cmd).starts_with("-WRONGTYPE"), "{name}");
        }
    }

    /// `CF.RESERVE` reads its options by name in an order of its own, and the
    /// first pair with a given name is the only one it looks at.
    #[test]
    fn reserve_complains_about_its_options_in_the_order_it_looks_for_them() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"CF.RESERVE",
                b"r",
                b"64",
                b"BUCKETSIZE",
                b"1",
                b"MAXITERATIONS",
                b"7",
                b"EXPANSION",
                b"4"
            ]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.DEBUG", b"r"]),
            "$77\r\nbktsize:1 buckets:64 items:0 deletes:0 filters:1 \
             max_iterations:7 expansion:4\r\n"
        );
        assert_eq!(f.run(&[b"CF.RESERVE", b"r", b"64"]), "-ERR item exists\r\n");

        assert_eq!(f.run(&[b"CF.RESERVE", b"q", b"abc"]), "-Bad capacity\r\n");
        assert_eq!(
            f.run(&[b"CF.RESERVE", b"q", b"1"]),
            "-Capacity must be in the range [2 * BUCKETSIZE, 1073741824]\r\n"
        );
        // The range is the bucket size's and not a constant, so a capacity that
        // was fine at two slots a bucket is not at four.
        assert_eq!(
            f.run(&[b"CF.RESERVE", b"q", b"7", b"BUCKETSIZE", b"4"]),
            "-Capacity must be in the range [2 * BUCKETSIZE, 1073741824]\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.RESERVE", b"q", b"8", b"BUCKETSIZE", b"4"]),
            "+OK\r\n"
        );

        // The capacity is checked last, so a command that is wrong twice
        // answers about the option. Which option it answers about is the order
        // the module looks for them in and not the order they were written, so
        // a bad kick budget wins over a bad bucket size wherever the two sit.
        assert_eq!(
            f.run(&[b"CF.RESERVE", b"q2", b"64", b"BUCKETSIZE", b"0"]),
            "-BUCKETSIZE: value must be in the range [1, 255]\r\n"
        );
        assert_eq!(
            f.run(&[
                b"CF.RESERVE",
                b"q2",
                b"64",
                b"EXPANSION",
                b"xx",
                b"BUCKETSIZE",
                b"0"
            ]),
            "-BUCKETSIZE: value must be in the range [1, 255]\r\n"
        );
        assert_eq!(
            f.run(&[
                b"CF.RESERVE",
                b"q2",
                b"64",
                b"MAXITERATIONS",
                b"0",
                b"BUCKETSIZE",
                b"0"
            ]),
            "-MAXITERATIONS: value must be in the range [1, 65535]\r\n"
        );
        // A second pair with a name that has already been read is not looked at
        // at all, so this one is a filter with buckets of one rather than an
        // error about a bucket size of zero.
        assert_eq!(
            f.run(&[
                b"CF.RESERVE",
                b"q3",
                b"64",
                b"BUCKETSIZE",
                b"1",
                b"BUCKETSIZE",
                b"0"
            ]),
            "+OK\r\n"
        );
        // A pair nobody knows is dropped, which is the opposite of what
        // `CF.INSERT` does with the same mistake.
        assert_eq!(
            f.run(&[b"CF.RESERVE", b"q4", b"64", b"NOSUCH", b"9"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.DEBUG", b"q4"]),
            "$78\r\nbktsize:2 buckets:32 items:0 deletes:0 filters:1 \
             max_iterations:20 expansion:1\r\n"
        );
        // And an option with nothing after it leaves an odd number of them,
        // which is an arity error rather than a complaint about the option.
        assert!(
            f.run(&[b"CF.RESERVE", b"q5", b"64", b"BUCKETSIZE"])
                .contains("wrong number of arguments")
        );
    }

    /// `CF.INSERT` is a reserve and a multi add, with a grammar that agrees
    /// with `CF.RESERVE` about nothing.
    #[test]
    fn insert_checks_every_occurrence_and_matches_on_the_first_letter() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"CAPACITY", b"64", b"ITEMS", b"a", b"b"]),
            "*2\r\n:1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.DEBUG", b"i"]),
            "$78\r\nbktsize:2 buckets:32 items:2 deletes:0 filters:1 \
             max_iterations:20 expansion:1\r\n"
        );
        // The NX form has three answers rather than two, which is why it stays
        // integers on both protocols.
        assert_eq!(
            f.run(&[b"CF.INSERTNX", b"i", b"ITEMS", b"a", b"c"]),
            "*2\r\n:0\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.INSERT", b"gone", b"NOCREATE", b"ITEMS", b"a"]),
            "-ERR not found\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"gone"]), ":0\r\n");

        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"CAPACITY", b"abc", b"ITEMS", b"a"]),
            "-Bad capacity\r\n"
        );
        // The bucket size cannot be given here, so the range names the config
        // that holds it instead of the option `CF.RESERVE` names.
        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"CAPACITY", b"2", b"ITEMS", b"a"]),
            "-Capacity must be in the range [cf-bucket-size * 2, 1073741824]\r\n"
        );
        // Every occurrence is checked, which is where this differs from
        // `CF.RESERVE`: the second `CAPACITY` is an error even though the first
        // one is the one that would have been used.
        assert_eq!(
            f.run(&[
                b"CF.INSERT",
                b"i",
                b"CAPACITY",
                b"8",
                b"CAPACITY",
                b"2",
                b"ITEMS",
                b"a"
            ]),
            "-Capacity must be in the range [cf-bucket-size * 2, 1073741824]\r\n"
        );
        // An option is one letter and not a word, so `NOSUCH` is `NOCREATE` and
        // `ITEMSXYZ` is `ITEMS`, and only a letter that starts nothing is
        // refused.
        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"NOSUCH", b"ITEMS", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"ITEMSXYZ", b"a"]),
            "*1\r\n:1\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"ZZZ", b"ITEMS", b"a"]),
            "-Unknown argument received\r\n"
        );
        // Everything after ITEMS is an item, even when it spells an option.
        assert_eq!(
            f.run(&[b"CF.INSERT", b"i", b"ITEMS", b"NOCREATE"]),
            "*1\r\n:1\r\n"
        );
        // And the two ways of sending no items at all are the same complaint.
        assert!(
            f.run(&[b"CF.INSERT", b"i", b"ITEMS"])
                .contains("wrong number of arguments")
        );
        assert!(
            f.run(&[b"CF.INSERT", b"i", b"CAPACITY"])
                .contains("wrong number of arguments")
        );
    }

    /// The two walls a filter can hit, which say different things and are not
    /// the same wall.
    #[test]
    fn a_full_filter_and_one_that_ran_out_of_filters_answer_differently() {
        let mut f = Fixture::new();
        f.run(&[
            b"CF.RESERVE",
            b"s",
            b"4",
            b"BUCKETSIZE",
            b"1",
            b"EXPANSION",
            b"0",
        ]);
        for i in 0..4u32 {
            assert_eq!(
                f.run(&[b"CF.ADD", b"s", i.to_string().as_bytes()]),
                ":1\r\n"
            );
        }
        assert_eq!(f.run(&[b"CF.ADD", b"s", b"4"]), "-Filter is full\r\n");
        assert_eq!(f.run(&[b"CF.ADDNX", b"s", b"zz"]), "-Filter is full\r\n");
        // The add commands say it in a sentence and the insert commands say it
        // in the array, one value per item, and the array is never short.
        assert_eq!(
            f.run(&[b"CF.INSERT", b"s", b"ITEMS", b"p", b"q"]),
            "*2\r\n:-1\r\n:-1\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.INSERTNX", b"s", b"ITEMS", b"0", b"q"]),
            "*2\r\n:0\r\n:-1\r\n"
        );

        // A chain that is allowed to grow stops for a different reason, and the
        // count it stops at is the filter limit rather than the room: this one
        // gives up with three slots free. Loading a chain that already has
        // every filter it is allowed shows why, since it refuses an item
        // straight into an empty one.
        let full = cf_header(0, 4, 0, 32, [1, 20, 1]);
        assert_eq!(f.run(&[b"CF.LOADCHUNK", b"g", b"1", &full]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"CF.ADD", b"g", b"q"]),
            "-Maximum expansions reached\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.INFO", b"g"]),
            "*16\r\n+Size\r\n:680\r\n+Number of buckets\r\n:4\r\n\
             +Number of filters\r\n:32\r\n+Number of items inserted\r\n:0\r\n\
             +Number of items deleted\r\n:0\r\n+Bucket size\r\n:1\r\n\
             +Expansion rate\r\n:1\r\n+Max iterations\r\n:20\r\n"
        );
    }

    /// A filter dumped a chunk at a time and put back under another key is the
    /// same filter, and the headers that describe one nobody could build are
    /// refused on the way in.
    #[test]
    fn a_cuckoo_dump_replays_into_a_filter_that_answers_the_same() {
        let mut f = Fixture::new();
        f.run(&[
            b"CF.RESERVE",
            b"src",
            b"8",
            b"BUCKETSIZE",
            b"2",
            b"EXPANSION",
            b"2",
        ]);
        for i in 0..40u32 {
            f.run(&[b"CF.ADD", b"src", i.to_string().as_bytes()]);
        }
        // Position zero asks for the header and every one after it is a byte
        // offset across every filter laid end to end, and the walk ends on a
        // zero and a nil rather than an empty chunk.
        let mut pos = b"0".to_vec();
        let mut chunks = 0;
        loop {
            let raw = f.raw(&[b"CF.SCANDUMP", b"src", &pos]);
            let head = String::from_utf8_lossy(&raw[..raw.len().min(24)]).into_owned();
            let next = head
                .split("\r\n")
                .nth(1)
                .and_then(|n| n.strip_prefix(':'))
                .expect("a two element reply of a position and a chunk")
                .to_owned();
            if next == "0" {
                assert!(raw.ends_with(b"$-1\r\n"), "the walk ends on a nil");
                break;
            }
            let body = &raw[raw.iter().position(|&b| b == b'$').expect("a bulk chunk")..];
            let at = body
                .windows(2)
                .position(|w| w == b"\r\n")
                .expect("a length line")
                + 2;
            let data = &body[at..body.len() - 2];
            assert_eq!(
                f.run(&[b"CF.LOADCHUNK", b"dst", next.as_bytes(), data]),
                "+OK\r\n",
                "loading chunk {chunks}"
            );
            pos = next.into_bytes();
            chunks += 1;
        }
        assert!(chunks >= 2, "a header and at least one chunk");

        assert_eq!(f.run(&[b"CF.INFO", b"dst"]), f.run(&[b"CF.INFO", b"src"]));
        assert_eq!(f.run(&[b"CF.DEBUG", b"dst"]), f.run(&[b"CF.DEBUG", b"src"]));
        for i in 0..40u32 {
            assert_eq!(
                f.run(&[b"CF.EXISTS", b"dst", i.to_string().as_bytes()]),
                ":1\r\n"
            );
        }

        // A filter with nothing in it hands out no header at all, so a client
        // that dumps one has nothing to load back.
        f.run(&[b"CF.RESERVE", b"empty", b"4", b"BUCKETSIZE", b"1"]);
        assert_eq!(
            f.run(&[b"CF.SCANDUMP", b"empty", b"0"]),
            "*2\r\n:0\r\n$-1\r\n"
        );

        // The positions this end will not take, which are not the same set at
        // both ends: a dump refuses a negative one and a load takes it as an
        // offset and fails to find anything there.
        assert_eq!(
            f.run(&[b"CF.SCANDUMP", b"src", b"nope"]),
            "-Invalid position\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.SCANDUMP", b"src", b"-1"]),
            "-Invalid position\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.LOADCHUNK", b"dst", b"0", b"x"]),
            "-Invalid position\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.LOADCHUNK", b"dst", b"99999", b"x"]),
            "-Couldn't load chunk!\r\n"
        );
        // A header on top of a filter is refused rather than merged.
        let good = cf_header(0, 8, 0, 1, [2, 20, 1]);
        assert_eq!(
            f.run(&[b"CF.LOADCHUNK", b"dst", b"1", &good]),
            "-ERR item exists\r\n"
        );
        // A chunk that is not the size of a header where a header should have
        // been is one sentence, and one that is the size of a header and
        // describes a filter nobody could build is another.
        assert_eq!(
            f.run(&[b"CF.LOADCHUNK", b"n1", b"1", b"short"]),
            "-Invalid header\r\n"
        );
        for (why, bad) in [
            ("no filters at all", cf_header(0, 8, 0, 0, [2, 20, 1])),
            ("no buckets", cf_header(0, 0, 0, 1, [2, 20, 1])),
            (
                "a bucket count that is not a power of two",
                cf_header(0, 3, 0, 1, [2, 20, 1]),
            ),
            ("an empty bucket", cf_header(0, 8, 0, 1, [0, 20, 1])),
            ("no kicks", cf_header(0, 8, 0, 1, [2, 0, 1])),
            (
                "a growth nobody could reach",
                cf_header(0, 8, 0, 1, [2, 20, 32769]),
            ),
            (
                "a chain that cannot grow and did",
                cf_header(0, 8, 0, 2, [2, 20, 0]),
            ),
            // The count is written in eight bytes and read into two, so a
            // number that is a multiple of the second arrives as none.
            (
                "a filter count that wraps",
                cf_header(0, 8, 0, 65_536, [2, 20, 1]),
            ),
        ] {
            assert_eq!(
                f.run(&[b"CF.LOADCHUNK", b"bad", b"1", &bad]),
                "-Couldn't create filter!\r\n",
                "{why}"
            );
        }
    }

    /// The RESP3 shapes, which are where this family differs most from RESP2
    /// and where one of its answers stops being readable.
    #[test]
    fn the_cuckoo_family_answers_in_resp3_spelling_too() {
        let mut f = Fixture::new();
        f.out.set_proto(Proto::Resp3);
        assert_eq!(f.run(&[b"CF.ADD", b"c", b"a"]), "#t\r\n");
        assert_eq!(f.run(&[b"CF.ADD", b"c", b"a"]), "#t\r\n");
        assert_eq!(f.run(&[b"CF.ADDNX", b"c", b"a"]), "#f\r\n");
        assert_eq!(f.run(&[b"CF.EXISTS", b"c", b"a"]), "#t\r\n");
        assert_eq!(
            f.run(&[b"CF.MEXISTS", b"c", b"a", b"z"]),
            "*2\r\n#t\r\n#f\r\n"
        );
        assert_eq!(f.run(&[b"CF.DEL", b"c", b"a"]), "#t\r\n");
        assert_eq!(f.run(&[b"CF.DEL", b"c", b"z"]), "#f\r\n");
        // The count stays an integer, because it counts rather than answers.
        assert_eq!(f.run(&[b"CF.COUNT", b"c", b"a"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"CF.INFO", b"c"]),
            "%8\r\n+Size\r\n:1080\r\n+Number of buckets\r\n:512\r\n\
             +Number of filters\r\n:1\r\n+Number of items inserted\r\n:1\r\n\
             +Number of items deleted\r\n:1\r\n+Bucket size\r\n:2\r\n\
             +Expansion rate\r\n:1\r\n+Max iterations\r\n:20\r\n"
        );

        // `CF.INSERT` writes a boolean per item here and an integer per item on
        // RESP2, and minus one has nowhere to go in a boolean, so a RESP3
        // client cannot tell an item that did not fit from one that is already
        // there. `CF.INSERTNX` keeps its integers for exactly that reason.
        f.run(&[
            b"CF.RESERVE",
            b"s",
            b"4",
            b"BUCKETSIZE",
            b"1",
            b"EXPANSION",
            b"0",
        ]);
        assert_eq!(
            f.run(&[
                b"CF.INSERT",
                b"s",
                b"ITEMS",
                b"a",
                b"b",
                b"c",
                b"d",
                b"e",
                b"f"
            ]),
            "*6\r\n#t\r\n#t\r\n#t\r\n#f\r\n#f\r\n#f\r\n"
        );
        assert_eq!(
            f.run(&[b"CF.INSERTNX", b"s", b"ITEMS", b"a", b"zz"]),
            "*2\r\n:0\r\n:-1\r\n"
        );
        assert_eq!(f.run(&[b"CF.ADD", b"s", b"zzz"]), "-Filter is full\r\n");
        // The end of a dump is a nil and not an empty chunk, which is one
        // underscore here and a negative length on RESP2.
        assert_eq!(f.run(&[b"CF.SCANDUMP", b"c", b"9999"]), "*2\r\n:0\r\n_\r\n");
    }

    // ------------------------------------------------------------------- cms

    /// A sketch is made from either end, and both constructors look at the key
    /// before they look at their arguments.
    #[test]
    fn a_sketch_is_made_from_a_size_or_from_an_error_rate() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"CMS.INITBYDIM", b"d", b"100", b"5"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"CMS.INFO", b"d"]),
            "*6\r\n+width\r\n:100\r\n+depth\r\n:5\r\n+count\r\n:0\r\n"
        );
        assert_eq!(f.run(&[b"TYPE", b"d"]), "+CMSk-TYPE\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"d"]), "$3\r\nraw\r\n");
        // Two over the error rounded up, and the log of the probability over the
        // log of a half rounded up, which for these two is 200 by 6.
        assert_eq!(
            f.run(&[b"CMS.INITBYPROB", b"p", b"0.01", b"0.03"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.INFO", b"p"]),
            "*6\r\n+width\r\n:200\r\n+depth\r\n:6\r\n+count\r\n:0\r\n"
        );
        // The key is checked first, so a width of zero at a key that is already
        // there is about the key and not about the width.
        assert_eq!(
            f.run(&[b"CMS.INITBYDIM", b"d", b"0", b"2"]),
            "-CMS: key already exists\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.INITBYDIM", b"new", b"0", b"2"]),
            "-CMS: invalid width\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.INITBYDIM", b"new", b"2", b"0"]),
            "-CMS: invalid depth\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.INITBYPROB", b"new", b"0", b"0.5"]),
            "-CMS: invalid overestimation value\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.INITBYPROB", b"new", b"0.1", b"1"]),
            "-CMS: invalid prob value\r\n"
        );
        // A probability whose float conversion is zero has no depth, and a width
        // past a signed sixty four bit integer has no width, and both are the
        // same sentence.
        assert_eq!(
            f.run(&[b"CMS.INITBYPROB", b"new", b"0.5", b"1e-46"]),
            "-CMS: invalid init arguments\r\n"
        );
        // And a sketch bigger than a gibibyte of counters is refused here where
        // the reference reserves address space nobody has touched, which is
        // D-47.
        assert_eq!(
            f.run(&[b"CMS.INITBYDIM", b"new", b"268435457", b"1"]),
            "-CMS: Insufficient memory to create the key\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"new"]), ":0\r\n");
    }

    /// Every pair is parsed before any of them lands, the counters saturate,
    /// and the count is a signed total of what was asked for.
    #[test]
    fn increments_are_parsed_whole_and_the_counters_saturate() {
        let mut f = Fixture::new();
        f.run(&[b"CMS.INITBYDIM", b"c", b"100", b"4"]);
        assert_eq!(
            f.run(&[b"CMS.INCRBY", b"c", b"a", b"3", b"b", b"4"]),
            "*2\r\n:3\r\n:4\r\n"
        );
        // An item that is incremented twice in one call sees its own first
        // increment in the reply to the second.
        assert_eq!(
            f.run(&[b"CMS.INCRBY", b"c", b"a", b"1", b"a", b"1"]),
            "*2\r\n:4\r\n:5\r\n"
        );
        // A bad number anywhere means nothing at all is applied.
        assert_eq!(
            f.run(&[b"CMS.INCRBY", b"c", b"a", b"9", b"b", b"x"]),
            "-CMS: Cannot parse number\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.INCRBY", b"c", b"a", b"9", b"b", b"-1"]),
            "-CMS: Number cannot be negative\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.QUERY", b"c", b"a", b"b"]),
            "*2\r\n:5\r\n:4\r\n"
        );
        // The counters stop at four billion and the item that stopped says so in
        // its own slot while the one beside it answers a number.
        f.run(&[b"CMS.INCRBY", b"c", b"a", b"4294967295"]);
        assert_eq!(
            f.run(&[b"CMS.INCRBY", b"c", b"a", b"1", b"b", b"1"]),
            "*2\r\n-CMS: INCRBY overflow\r\n:5\r\n"
        );
        assert_eq!(f.run(&[b"CMS.QUERY", b"c", b"a"]), "*1\r\n:4294967295\r\n");
        // The count is what was asked for rather than what landed, and it is
        // signed, so a big enough total comes back negative.
        f.run(&[b"CMS.INITBYDIM", b"w", b"4", b"1"]);
        f.run(&[b"CMS.INCRBY", b"w", b"x", b"9223372036854775807"]);
        f.run(&[b"CMS.INCRBY", b"w", b"x", b"1"]);
        assert_eq!(
            f.run(&[b"CMS.INFO", b"w"]),
            "*6\r\n+width\r\n:4\r\n+depth\r\n:1\r\n+count\r\n:-9223372036854775808\r\n"
        );
        // An odd number of arguments after the key is an arity error and not a
        // syntax one.
        assert!(
            f.run(&[b"CMS.INCRBY", b"c", b"a", b"1", b"b"])
                .contains("wrong number of arguments")
        );
        assert_eq!(
            f.run(&[b"CMS.INCRBY", b"nope", b"a", b"1"]),
            "-CMS: key does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.QUERY", b"nope", b"a"]),
            "-CMS: key does not exist\r\n"
        );
    }

    /// A merge overwrites its destination, and it is worked out in full before
    /// any of it is written.
    #[test]
    fn a_merge_lands_whole_or_not_at_all() {
        let mut f = Fixture::new();
        for name in [&b"m1"[..], b"m2", b"dst"] {
            f.run(&[b"CMS.INITBYDIM", name, b"64", b"3"]);
        }
        f.run(&[b"CMS.INCRBY", b"m1", b"a", b"5"]);
        f.run(&[b"CMS.INCRBY", b"m2", b"a", b"7"]);
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"2", b"m1", b"m2"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"CMS.QUERY", b"dst", b"a"]), "*1\r\n:12\r\n");
        // Overwritten and not added to, so the same merge twice is the same
        // answer twice.
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"2", b"m1", b"m2"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"CMS.QUERY", b"dst", b"a"]), "*1\r\n:12\r\n");
        assert_eq!(
            f.run(&[
                b"CMS.MERGE",
                b"dst",
                b"2",
                b"m1",
                b"m2",
                b"WEIGHTS",
                b"2",
                b"3"
            ]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"CMS.QUERY", b"dst", b"a"]), "*1\r\n:31\r\n");
        // A cell times a weight is checked wide rather than wrapped, so this is
        // a refusal and the destination is left exactly as it was.
        assert_eq!(
            f.run(&[
                b"CMS.MERGE",
                b"dst",
                b"1",
                b"m1",
                b"WEIGHTS",
                b"4611686018427387904"
            ]),
            "-CMS: MERGE overflow\r\n"
        );
        assert_eq!(f.run(&[b"CMS.QUERY", b"dst", b"a"]), "*1\r\n:31\r\n");
        // The destination comes first, then the count, then the layout, then the
        // weights, then the sources one at a time.
        f.run(&[b"CMS.INITBYDIM", b"wide", b"128", b"3"]);
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"gone", b"1", b"m1"]),
            "-CMS: key does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"0", b"m1"]),
            "-CMS: Number of keys must be positive\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"3", b"m1"]),
            "-CMS: wrong number of keys\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"1", b"m1", b"WEIGHTS", b"1", b"2"]),
            "-CMS: wrong number of keys/weights\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"1", b"wide"]),
            "-CMS: width/depth is not equal\r\n"
        );
        assert_eq!(
            f.run(&[b"CMS.MERGE", b"dst", b"1", b"gone"]),
            "-CMS: key does not exist\r\n"
        );
    }

    /// A key holding anything else is `WRONGTYPE` to all six, and a key holding
    /// a sketch is refused by the two commands that would have to serialise it.
    #[test]
    fn a_sketch_is_a_module_key_to_the_rest_of_the_keyspace() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"text"]);
        for cmd in [
            vec![&b"CMS.INITBYDIM"[..], b"s", b"8", b"2"],
            vec![&b"CMS.INCRBY"[..], b"s", b"a", b"1"],
            vec![&b"CMS.QUERY"[..], b"s", b"a"],
            vec![&b"CMS.INFO"[..], b"s"],
            vec![&b"CMS.MERGE"[..], b"s", b"1", b"s"],
        ] {
            let name = String::from_utf8_lossy(cmd[0]).into_owned();
            let reply = f.run(&cmd);
            // The two constructors see the key before anything else and say so
            // in the module's own words, and the rest are `WRONGTYPE`.
            assert!(
                reply.starts_with("-WRONGTYPE") || reply == "-CMS: key already exists\r\n",
                "{name}: {reply}"
            );
        }
        f.run(&[b"CMS.INITBYDIM", b"c", b"64", b"2"]);
        // Redis refuses to copy a module key that has no copy callback, and
        // these are its words rather than ours. `DUMP` is the other half of
        // D-48: the reference has a payload for one of these and we do not.
        assert_eq!(
            f.run(&[b"COPY", b"c", b"c2"]),
            "-ERR not supported for this module key\r\n"
        );
        assert_eq!(
            f.run(&[b"DUMP", b"c"]),
            "-ERR DUMP is not supported for this module key\r\n"
        );
        // A graph is nobody's module and keeps its own sentence.
        f.run(&[b"G.NADD", b"g", b"a"]);
        assert_eq!(
            f.run(&[b"COPY", b"g", b"g2"]),
            "-ERR COPY is not supported for a graph\r\n"
        );
        assert_eq!(
            f.run(&[b"DUMP", b"g"]),
            "-ERR DUMP is not supported for a graph\r\n"
        );
        // Everything that does not need a byte shape works on a sketch key the
        // way it works on any other.
        assert_eq!(f.run(&[b"EXPIRE", b"c", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"PERSIST", b"c"]), ":1\r\n");
        assert_eq!(f.run(&[b"RENAME", b"c", b"c3"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TYPE", b"c3"]), "+CMSk-TYPE\r\n");
        assert_eq!(f.run(&[b"DEL", b"c3"]), ":1\r\n");
    }

    // ------------------------------------------------------------------ topk

    /// `TOPK.RESERVE` takes three arguments or six, and looks at the key before
    /// it looks at any of them.
    #[test]
    fn a_reserve_takes_three_arguments_or_six() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"TOPK.RESERVE", b"t", b"5"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"TOPK.INFO", b"t"]),
            "*8\r\n+k\r\n:5\r\n+width\r\n:8\r\n+depth\r\n:7\r\n+decay\r\n$3\r\n0.9\r\n"
        );
        // Four arguments and five are an arity error rather than a defaulted
        // depth or decay.
        for cmd in [
            vec![&b"TOPK.RESERVE"[..], b"u", b"5", b"8"],
            vec![&b"TOPK.RESERVE"[..], b"u", b"5", b"8", b"7"],
        ] {
            assert!(f.run(&cmd).contains("wrong number of arguments"));
        }
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"u", b"5", b"8", b"7", b"0.5"]),
            "+OK\r\n"
        );
        // The key is checked first, so a reserve with nothing else right at a
        // key that is taken still says the key is taken.
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"u", b"0", b"0", b"0", b"9"]),
            "-TopK: key already exists\r\n"
        );
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"v", b"0"]),
            "-TopK: invalid k\r\n"
        );
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"v", b"1", b"0", b"7", b"0.9"]),
            "-TopK: invalid width\r\n"
        );
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"v", b"1", b"8", b"x", b"0.9"]),
            "-TopK: invalid depth\r\n"
        );
        // Zero is out and one is in, which is the module's `> 0` and `<= 1`.
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"v", b"1", b"8", b"7", b"0"]),
            "-TopK: invalid decay value. must be '<= 1' & '> 0'\r\n"
        );
        assert_eq!(
            f.run(&[b"TOPK.RESERVE", b"v", b"1", b"8", b"7", b"1"]),
            "+OK\r\n"
        );
        // Past the cap, with the one sentence in the family that has a prefix.
        assert_eq!(
            f.run(&[
                b"TOPK.RESERVE",
                b"w",
                b"1",
                b"4294967295",
                b"4294967295",
                b"0.9"
            ]),
            "-ERR Insufficient memory to create topk data structure\r\n"
        );
    }

    /// What the sketch keeps, and the three ways of asking about it.
    #[test]
    fn the_kept_set_is_what_query_and_list_answer_from() {
        let mut f = Fixture::new();
        f.run(&[b"TOPK.RESERVE", b"t", b"2", b"1000", b"5", b"0.9"]);
        // A null an item while there is room, then the name of whatever was
        // pushed out.
        assert_eq!(
            f.run(&[b"TOPK.ADD", b"t", b"a", b"b"]),
            "*2\r\n$-1\r\n$-1\r\n"
        );
        assert_eq!(f.run(&[b"TOPK.INCRBY", b"t", b"a", b"10"]), "*1\r\n$-1\r\n");
        // Two slots are full and `c` arrives with a count of one, which is not
        // under the smallest kept count, so it takes that slot straight away.
        assert_eq!(f.run(&[b"TOPK.ADD", b"t", b"c"]), "*1\r\n$1\r\nb\r\n");
        assert_eq!(f.run(&[b"TOPK.INCRBY", b"t", b"c", b"5"]), "*1\r\n$-1\r\n");
        assert_eq!(
            f.run(&[b"TOPK.QUERY", b"t", b"a", b"b", b"c"]),
            "*3\r\n:1\r\n:0\r\n:1\r\n"
        );
        // The table still counts what the kept set let go of.
        assert_eq!(
            f.run(&[b"TOPK.COUNT", b"t", b"a", b"b", b"c"]),
            "*3\r\n:11\r\n:1\r\n:6\r\n"
        );
        assert_eq!(f.run(&[b"TOPK.LIST", b"t"]), "*2\r\n$1\r\na\r\n$1\r\nc\r\n");
        assert_eq!(
            f.run(&[b"TOPK.LIST", b"t", b"WITHCOUNT"]),
            "*4\r\n$1\r\na\r\n:11\r\n$1\r\nc\r\n:6\r\n"
        );
        // Any prefix of the keyword turns the counts on, the empty string
        // included, and only a longer word or a different one is refused.
        assert_eq!(
            f.run(&[b"TOPK.LIST", b"t", b"w"]),
            f.run(&[b"TOPK.LIST", b"t", b"WITHCOUNT"])
        );
        assert_eq!(
            f.run(&[b"TOPK.LIST", b"t", b""]),
            f.run(&[b"TOPK.LIST", b"t", b"WITHCOUNT"])
        );
        assert_eq!(
            f.run(&[b"TOPK.LIST", b"t", b"WITHCOUNTS"]),
            "-WITHCOUNT keyword expected\r\n"
        );
        // And the keyword is looked at before the key, so a missing key with a
        // bad keyword complains about the keyword.
        assert_eq!(
            f.run(&[b"TOPK.LIST", b"missing", b"nope"]),
            "-WITHCOUNT keyword expected\r\n"
        );
        assert_eq!(
            f.run(&[b"TOPK.LIST", b"missing"]),
            "-TopK: key does not exist\r\n"
        );
        // An item counted zero times is kept and not listed.
        f.run(&[b"TOPK.RESERVE", b"z", b"3"]);
        assert_eq!(
            f.run(&[b"TOPK.INCRBY", b"z", b"nothing", b"0"]),
            "*1\r\n$-1\r\n"
        );
        assert_eq!(f.run(&[b"TOPK.QUERY", b"z", b"nothing"]), "*1\r\n:1\r\n");
        assert_eq!(f.run(&[b"TOPK.LIST", b"z"]), "*0\r\n");
    }

    /// `TOPK.INCRBY` applies as it goes, so a bad increment leaves everything
    /// before it counted, and the reply counts what it wrote.
    #[test]
    fn an_increment_is_applied_as_it_goes_and_stops_at_a_bad_one() {
        let mut f = Fixture::new();
        f.run(&[b"TOPK.RESERVE", b"t", b"5", b"1000", b"5", b"0.9"]);
        // Three pairs, the middle one bad: two elements come back, one of them
        // the error, and the array header says two rather than three. That last
        // part is D-51 and it is why a client here stays in step.
        assert_eq!(
            f.run(&[b"TOPK.INCRBY", b"t", b"a", b"3", b"b", b"-1", b"c", b"4"]),
            format!(
                "*2\r\n$-1\r\n-{}\r\n",
                "TopK: increment must be an integer greater or equal to 0                            and smaller or equal to 100,000"
            )
        );
        assert_eq!(
            f.run(&[b"TOPK.COUNT", b"t", b"a", b"b", b"c"]),
            "*3\r\n:3\r\n:0\r\n:0\r\n"
        );
        // A hundred thousand is in and one more is out.
        assert_eq!(
            f.run(&[b"TOPK.INCRBY", b"t", b"a", b"100000"]),
            "*1\r\n$-1\r\n"
        );
        assert!(
            f.run(&[b"TOPK.INCRBY", b"t", b"a", b"100001"])
                .contains("smaller or equal to 100,000")
        );
        // Pairs have to be pairs.
        assert!(
            f.run(&[b"TOPK.INCRBY", b"t", b"a", b"1", b"b"])
                .contains("wrong number of arguments")
        );
        assert_eq!(f.run(&[b"TOPK.COUNT", b"t", b"a"]), "*1\r\n:100003\r\n");
    }

    /// The RESP3 shapes, which are the two the protocols disagree about.
    #[test]
    fn a_query_is_a_bool_and_info_is_a_map_on_resp3() {
        let mut f = Fixture::new();
        f.run(&[b"HELLO", b"3"]);
        f.run(&[b"TOPK.RESERVE", b"t", b"2", b"8", b"7", b"0.5"]);
        f.run(&[b"TOPK.ADD", b"t", b"a"]);
        assert_eq!(
            f.run(&[b"TOPK.QUERY", b"t", b"a", b"b"]),
            "*2\r\n#t\r\n#f\r\n"
        );
        // The count stays an integer on both protocols.
        assert_eq!(f.run(&[b"TOPK.COUNT", b"t", b"a"]), "*1\r\n:1\r\n");
        assert_eq!(
            f.run(&[b"TOPK.INFO", b"t"]),
            "%4\r\n+k\r\n:2\r\n+width\r\n:8\r\n+depth\r\n:7\r\n+decay\r\n,0.5\r\n"
        );
        assert_eq!(f.run(&[b"TOPK.ADD", b"t", b"a"]), "*1\r\n_\r\n");
    }

    /// A top k key answers the module sentences the other sketch families
    /// answer, and its own word for its type.
    #[test]
    fn a_top_k_sketch_is_a_module_key_to_the_rest_of_the_keyspace() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"text"]);
        for cmd in [
            vec![&b"TOPK.RESERVE"[..], b"s", b"5"],
            vec![&b"TOPK.ADD"[..], b"s", b"a"],
            vec![&b"TOPK.INCRBY"[..], b"s", b"a", b"1"],
            vec![&b"TOPK.QUERY"[..], b"s", b"a"],
            vec![&b"TOPK.COUNT"[..], b"s", b"a"],
            vec![&b"TOPK.LIST"[..], b"s"],
            vec![&b"TOPK.INFO"[..], b"s"],
        ] {
            let name = String::from_utf8_lossy(cmd[0]).into_owned();
            let reply = f.run(&cmd);
            assert!(
                reply.starts_with("-WRONGTYPE") || reply == "-TopK: key already exists\r\n",
                "{name}: {reply}"
            );
        }
        f.run(&[b"TOPK.RESERVE", b"t", b"5"]);
        assert_eq!(
            f.run(&[b"COPY", b"t", b"t2"]),
            "-ERR not supported for this module key\r\n"
        );
        assert_eq!(
            f.run(&[b"DUMP", b"t"]),
            "-ERR DUMP is not supported for this module key\r\n"
        );
        assert_eq!(f.run(&[b"EXPIRE", b"t", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"PERSIST", b"t"]), ":1\r\n");
        assert_eq!(f.run(&[b"RENAME", b"t", b"t3"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TYPE", b"t3"]), "+TopK-TYPE\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"t3"]), "$3\r\nraw\r\n");
        assert_eq!(f.run(&[b"DEL", b"t3"]), ":1\r\n");
        // Every one of the six that is not the constructor says the same thing
        // about a key that is not there.
        assert_eq!(
            f.run(&[b"TOPK.INFO", b"t3"]),
            "-TopK: key does not exist\r\n"
        );
    }

    // --------------------------------------------------------------- tdigest

    /// `TDIGEST.CREATE` takes two arguments or four, and the keyword search is a
    /// search rather than a lookup.
    #[test]
    fn a_create_takes_two_arguments_or_four_and_reads_the_last_one() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"TDIGEST.CREATE", b"t"]), "+OK\r\n");
        // A hundred is the default and the capacity is six times it plus ten.
        assert_eq!(
            f.run(&[b"TDIGEST.INFO", b"t"]),
            "*18\r\n+Compression\r\n:100\r\n+Capacity\r\n:610\r\n+Merged nodes\r\n:0\r\n\
             +Unmerged nodes\r\n:0\r\n+Merged weight\r\n:0\r\n+Unmerged weight\r\n:0\r\n\
             +Observations\r\n:0\r\n+Total compressions\r\n:0\r\n+Memory usage\r\n:9840\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"t"]),
            "-ERR T-Digest: key already exists\r\n"
        );
        // Three arguments is an arity error and not a missing keyword.
        assert!(
            f.run(&[b"TDIGEST.CREATE", b"u", b"COMPRESSION"])
                .contains("wrong number of arguments")
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"u", b"COMPRESSION", b"1000"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"v", b"compression", b"1"]),
            "+OK\r\n"
        );
        // The word is looked for across both trailing arguments and the number
        // is then read out of the last one whatever was found, so this looks for
        // a number inside the word `COMPRESSION` and does not find one.
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"w", b"100", b"COMPRESSION"]),
            "-ERR T-Digest: error parsing compression parameter\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"w", b"NOPE", b"100"]),
            "-ERR T-Digest: wrong keyword\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"w", b"COMPRESSION", b"1.5"]),
            "-ERR T-Digest: error parsing compression parameter\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"w", b"COMPRESSION", b"0"]),
            "-ERR T-Digest: compression parameter needs to be a positive integer\r\n"
        );
        // The reference's own ceiling, which is where the capacity stops fitting
        // in an int, and one past it.
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"w", b"COMPRESSION", b"357913942"]),
            "-ERR T-Digest: allocation failed\r\n"
        );
        // And ours, which is a gibibyte of centroids and is D-52.
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"w", b"COMPRESSION", b"100000000"]),
            "-ERR T-Digest: allocation failed\r\n"
        );
        // The key is checked before the arguments, so a bad compression at a key
        // that is already a digest still says the key is taken.
        assert_eq!(
            f.run(&[b"TDIGEST.CREATE", b"t", b"COMPRESSION", b"0"]),
            "-ERR T-Digest: key already exists\r\n"
        );
    }

    /// The four samples every note about this family is written against, and the
    /// answers a real 8.10.1 gives for them.
    #[test]
    fn the_quantile_family_answers_what_the_module_answers() {
        let mut f = Fixture::new();
        f.run(&[b"TDIGEST.CREATE", b"s"]);
        assert_eq!(
            f.run(&[b"TDIGEST.ADD", b"s", b"1", b"2", b"3", b"4"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"TDIGEST.MIN", b"s"]), "$1\r\n1\r\n");
        assert_eq!(f.run(&[b"TDIGEST.MAX", b"s"]), "$1\r\n4\r\n");
        // The cdf of a sample is the weight below it plus half its own.
        assert_eq!(
            f.run(&[b"TDIGEST.CDF", b"s", b"1", b"2", b"3", b"4"]),
            "*4\r\n$5\r\n0.125\r\n$5\r\n0.375\r\n$5\r\n0.625\r\n$5\r\n0.875\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"s", b"0", b"0.5", b"1"]),
            "*3\r\n$1\r\n1\r\n$1\r\n3\r\n$1\r\n4\r\n"
        );
        // Out of order, the walk restarts, and 0.5 answers 3 either way while
        // the two after it are read from the front again.
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"s", b"0.5", b"0.1", b"0.9"]),
            "*3\r\n$1\r\n3\r\n$1\r\n1\r\n$1\r\n4\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.RANK", b"s", b"0", b"1", b"3", b"4", b"5"]),
            "*5\r\n:-1\r\n:0\r\n:2\r\n:3\r\n:4\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.REVRANK", b"s", b"0", b"1", b"3", b"4", b"5"]),
            "*5\r\n:4\r\n:3\r\n:1\r\n:0\r\n:-1\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.BYRANK", b"s", b"0", b"1", b"3", b"4"]),
            "*4\r\n$1\r\n1\r\n$1\r\n2\r\n$1\r\n4\r\n$3\r\ninf\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.BYREVRANK", b"s", b"0", b"1", b"3", b"4"]),
            "*4\r\n$1\r\n4\r\n$1\r\n3\r\n$1\r\n1\r\n$4\r\n-inf\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"s", b"0", b"1"]),
            "$3\r\n2.5\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"s", b"0.25", b"0.75"]),
            "$3\r\n2.5\r\n"
        );
        // The ranges, which are separate sentences from the parse failures.
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"s", b"1.1"]),
            "-ERR T-Digest: quantile should be in [0,1]\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"s", b"zzz"]),
            "-ERR T-Digest: error parsing quantile\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.CDF", b"s", b"zzz"]),
            "-ERR T-Digest: error parsing cdf\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.RANK", b"s", b"zzz"]),
            "-ERR T-Digest: error parsing value\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.BYRANK", b"s", b"-1"]),
            "-ERR T-Digest: rank needs to be non negative\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.BYRANK", b"s", b"1.5"]),
            "-ERR T-Digest: error parsing rank\r\n"
        );
        // Both cuts have their own parse sentence and share the range one, and
        // equal cuts are refused rather than answering nothing.
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"s", b"zzz", b"0.9"]),
            "-ERR T-Digest: error parsing low_cut_percentile\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"s", b"0.1", b"zzz"]),
            "-ERR T-Digest: error parsing high_cut_percentile\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"s", b"0.1", b"1.1"]),
            "-ERR T-Digest: low_cut_percentile and high_cut_percentile should be in [0,1]\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"s", b"0.5", b"0.5"]),
            "-ERR T-Digest: low_cut_percentile should be lower than high_cut_percentile\r\n"
        );
    }

    /// An empty digest answers every question, and answers most of them with
    /// something that is not a number.
    #[test]
    fn an_empty_digest_has_an_answer_for_everything() {
        let mut f = Fixture::new();
        f.run(&[b"TDIGEST.CREATE", b"e"]);
        assert_eq!(f.run(&[b"TDIGEST.MIN", b"e"]), "$3\r\nnan\r\n");
        assert_eq!(f.run(&[b"TDIGEST.MAX", b"e"]), "$3\r\nnan\r\n");
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"e", b"0", b"1"]),
            "*2\r\n$3\r\nnan\r\n$3\r\nnan\r\n"
        );
        assert_eq!(f.run(&[b"TDIGEST.CDF", b"e", b"0"]), "*1\r\n$3\r\nnan\r\n");
        assert_eq!(
            f.run(&[b"TDIGEST.TRIMMED_MEAN", b"e", b"0.1", b"0.9"]),
            "$3\r\nnan\r\n"
        );
        // Minus two, which is a number no rank on a digest with samples in it
        // can ever be.
        assert_eq!(
            f.run(&[b"TDIGEST.RANK", b"e", b"0", b"1"]),
            "*2\r\n:-2\r\n:-2\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.REVRANK", b"e", b"0", b"1"]),
            "*2\r\n:-2\r\n:-2\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.BYRANK", b"e", b"0", b"5"]),
            "*2\r\n$3\r\nnan\r\n$3\r\nnan\r\n"
        );
        // A reset puts a digest with samples back into exactly this state.
        f.run(&[b"TDIGEST.ADD", b"e", b"1", b"2", b"3"]);
        assert_eq!(f.run(&[b"TDIGEST.RESET", b"e"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TDIGEST.MIN", b"e"]), "$3\r\nnan\r\n");
        // Down to the compression count, so a reset digest and a fresh one of
        // the same compression report the same nine numbers.
        f.run(&[b"TDIGEST.CREATE", b"e2"]);
        assert_eq!(
            f.run(&[b"TDIGEST.INFO", b"e"]),
            f.run(&[b"TDIGEST.INFO", b"e2"])
        );
    }

    /// The double parser is Redis's and not this engine's, and the two disagree
    /// at both ends of the range.
    #[test]
    fn a_sample_is_read_the_way_redis_reads_a_double() {
        let mut f = Fixture::new();
        f.run(&[b"TDIGEST.CREATE", b"a"]);
        // Overflow and underflow are parse failures rather than an infinity and
        // a zero, which is where this parts company with the rest of the engine.
        for bad in [
            &b"nan"[..],
            b"1e400",
            b"-1e400",
            b"1e309",
            b"1e-400",
            b"",
            b" 1",
            b"1 ",
            b"1e",
            b"--1",
        ] {
            assert_eq!(
                f.run(&[b"TDIGEST.ADD", b"a", bad]),
                "-ERR T-Digest: error parsing val parameter\r\n",
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
        // An infinity spelled out parses and is then refused for being one, with
        // a different sentence.
        for word in [&b"inf"[..], b"-inf", b"+INF", b"Infinity"] {
            assert_eq!(
                f.run(&[b"TDIGEST.ADD", b"a", word]),
                "-ERR T-Digest: val parameter needs to be a finite number\r\n",
                "{}",
                String::from_utf8_lossy(word)
            );
        }
        // These all parse: hex, a bare point either side, and the smallest
        // subnormal the reference will take.
        for good in [&b"0x10"[..], b".5", b"1.", b"1e-320", b"-0", b"0"] {
            assert_eq!(
                f.run(&[b"TDIGEST.ADD", b"a", good]),
                "+OK\r\n",
                "{}",
                String::from_utf8_lossy(good)
            );
        }
        // Nothing landed from the failures, so six samples is what there is.
        assert!(
            f.run(&[b"TDIGEST.INFO", b"a"])
                .contains("Observations\r\n:6\r\n")
        );
        // Every value is parsed before any is added, so this whole command is a
        // no op.
        assert_eq!(
            f.run(&[b"TDIGEST.ADD", b"a", b"1", b"zzz"]),
            "-ERR T-Digest: error parsing val parameter\r\n"
        );
        assert!(
            f.run(&[b"TDIGEST.INFO", b"a"])
                .contains("Observations\r\n:6\r\n")
        );
    }

    /// What a merge does to its destination, to its inputs and to the buffer
    /// split `TDIGEST.INFO` reports.
    #[test]
    fn a_merge_sweeps_the_destination_between_its_inputs() {
        let mut f = Fixture::new();
        f.run(&[b"TDIGEST.CREATE", b"m1", b"COMPRESSION", b"100"]);
        f.run(&[b"TDIGEST.ADD", b"m1", b"1", b"2", b"3"]);
        f.run(&[b"TDIGEST.CREATE", b"m2", b"COMPRESSION", b"200"]);
        f.run(&[b"TDIGEST.ADD", b"m2", b"4", b"5", b"6"]);
        assert_eq!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"2", b"m1", b"m2"]),
            "+OK\r\n"
        );
        // The destination did not exist, so the compression is the largest of
        // the inputs. The three from the first input were swept in before the
        // three from the second arrived, which is the one visible effect of the
        // reference folding one input at a time.
        let info = f.run(&[b"TDIGEST.INFO", b"d"]);
        assert!(info.contains("Compression\r\n:200\r\n"), "{info}");
        assert!(info.contains("Merged nodes\r\n:3\r\n"), "{info}");
        assert!(info.contains("Unmerged nodes\r\n:3\r\n"), "{info}");
        assert!(info.contains("Total compressions\r\n:1\r\n"), "{info}");
        assert_eq!(f.run(&[b"TDIGEST.MIN", b"d"]), "$1\r\n1\r\n");
        assert_eq!(f.run(&[b"TDIGEST.MAX", b"d"]), "$1\r\n6\r\n");
        // Reading a source sweeps it too, so a merge writes to keys it only
        // reads from.
        assert!(
            f.run(&[b"TDIGEST.INFO", b"m1"])
                .contains("Merged nodes\r\n:3\r\n")
        );
        // Without OVERRIDE the destination joins its own inputs, so this takes
        // it to nine observations and keeps its own compression.
        f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"m1"]);
        let info = f.run(&[b"TDIGEST.INFO", b"d"]);
        assert!(info.contains("Observations\r\n:9\r\n"), "{info}");
        assert!(info.contains("Compression\r\n:200\r\n"), "{info}");
        // With OVERRIDE the old destination is dropped and the compression goes
        // back to the largest of the inputs.
        f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"m1", b"OVERRIDE"]);
        let info = f.run(&[b"TDIGEST.INFO", b"d"]);
        assert!(info.contains("Observations\r\n:3\r\n"), "{info}");
        assert!(info.contains("Compression\r\n:100\r\n"), "{info}");
        // And COMPRESSION beats both.
        f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"m1", b"COMPRESSION", b"500"]);
        assert!(
            f.run(&[b"TDIGEST.INFO", b"d"])
                .contains("Compression\r\n:500\r\n")
        );
        // Naming the destination as a source folds it in twice.
        f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"d"]);
        assert!(
            f.run(&[b"TDIGEST.INFO", b"d"])
                .contains("Observations\r\n:12\r\n")
        );
        // The arguments, in the order the reference checks them.
        assert_eq!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"zzz", b"m1"]),
            "-ERR T-Digest: error parsing numkeys\r\n"
        );
        assert_eq!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"0", b"m1"]),
            "-ERR T-Digest: numkeys needs to be a positive integer\r\n"
        );
        assert!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"3", b"m1", b"m2"])
                .contains("wrong number of arguments")
        );
        assert!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"m1", b"COMPRESSION"])
                .contains("wrong number of arguments")
        );
        assert_eq!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"m1", b"NOPE"]),
            "-ERR T-Digest: wrong keyword\r\n"
        );
        // A source that is not there stops the whole thing, and the destination
        // is left as it was.
        assert_eq!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"2", b"m1", b"gone"]),
            "-ERR T-Digest: key does not exist\r\n"
        );
        assert!(
            f.run(&[b"TDIGEST.INFO", b"d"])
                .contains("Observations\r\n:12\r\n")
        );
        // A destination that is not there and is also named as a source is the
        // same sentence rather than an empty merge.
        assert_eq!(
            f.run(&[b"TDIGEST.MERGE", b"gone", b"1", b"gone"]),
            "-ERR T-Digest: key does not exist\r\n"
        );
    }

    /// The RESP3 shapes, which are the two the protocols disagree about.
    #[test]
    fn a_digest_answers_doubles_and_a_map_on_resp3() {
        let mut f = Fixture::new();
        f.run(&[b"HELLO", b"3"]);
        f.run(&[b"TDIGEST.CREATE", b"s"]);
        f.run(&[b"TDIGEST.ADD", b"s", b"1", b"2", b"3", b"4"]);
        assert_eq!(f.run(&[b"TDIGEST.MIN", b"s"]), ",1\r\n");
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"s", b"0", b"1"]),
            "*2\r\n,1\r\n,4\r\n"
        );
        assert_eq!(f.run(&[b"TDIGEST.CDF", b"s", b"1"]), "*1\r\n,0.125\r\n");
        // The two infinities and the NaN go out as the bare words.
        assert_eq!(f.run(&[b"TDIGEST.BYRANK", b"s", b"4"]), "*1\r\n,inf\r\n");
        assert_eq!(
            f.run(&[b"TDIGEST.BYREVRANK", b"s", b"4"]),
            "*1\r\n,-inf\r\n"
        );
        f.run(&[b"TDIGEST.CREATE", b"e"]);
        assert_eq!(f.run(&[b"TDIGEST.MIN", b"e"]), ",nan\r\n");
        // The ranks stay integers on both protocols.
        assert_eq!(f.run(&[b"TDIGEST.RANK", b"s", b"1"]), "*1\r\n:0\r\n");
        // Every question above swept the buffer in, so the four samples are all
        // merged by now and the compression count says it happened once.
        assert_eq!(
            f.run(&[b"TDIGEST.INFO", b"s"]),
            "%9\r\n+Compression\r\n:100\r\n+Capacity\r\n:610\r\n+Merged nodes\r\n:4\r\n\
             +Unmerged nodes\r\n:0\r\n+Merged weight\r\n:4\r\n+Unmerged weight\r\n:0\r\n\
             +Observations\r\n:4\r\n+Total compressions\r\n:1\r\n+Memory usage\r\n:9840\r\n"
        );
    }

    /// A t digest key answers the module sentences the other sketch families
    /// answer, and its own word for its type.
    #[test]
    fn a_t_digest_is_a_module_key_to_the_rest_of_the_keyspace() {
        let mut f = Fixture::new();
        f.run(&[b"SET", b"s", b"text"]);
        for cmd in [
            vec![&b"TDIGEST.CREATE"[..], b"s"],
            vec![&b"TDIGEST.RESET"[..], b"s"],
            vec![&b"TDIGEST.ADD"[..], b"s", b"1"],
            vec![&b"TDIGEST.MIN"[..], b"s"],
            vec![&b"TDIGEST.MAX"[..], b"s"],
            vec![&b"TDIGEST.QUANTILE"[..], b"s", b"0.5"],
            vec![&b"TDIGEST.CDF"[..], b"s", b"1"],
            vec![&b"TDIGEST.TRIMMED_MEAN"[..], b"s", b"0.1", b"0.9"],
            vec![&b"TDIGEST.RANK"[..], b"s", b"1"],
            vec![&b"TDIGEST.REVRANK"[..], b"s", b"1"],
            vec![&b"TDIGEST.BYRANK"[..], b"s", b"0"],
            vec![&b"TDIGEST.BYREVRANK"[..], b"s", b"0"],
            vec![&b"TDIGEST.INFO"[..], b"s"],
        ] {
            let name = String::from_utf8_lossy(cmd[0]).into_owned();
            let reply = f.run(&cmd);
            assert!(reply.starts_with("-WRONGTYPE"), "{name}: {reply}");
        }
        // The merge checks its destination the same way, and its sources too.
        f.run(&[b"TDIGEST.CREATE", b"t"]);
        assert!(
            f.run(&[b"TDIGEST.MERGE", b"s", b"1", b"t"])
                .starts_with("-WRONGTYPE")
        );
        assert!(
            f.run(&[b"TDIGEST.MERGE", b"d", b"1", b"s"])
                .starts_with("-WRONGTYPE")
        );
        assert_eq!(
            f.run(&[b"COPY", b"t", b"t2"]),
            "-ERR not supported for this module key\r\n"
        );
        assert_eq!(
            f.run(&[b"DUMP", b"t"]),
            "-ERR DUMP is not supported for this module key\r\n"
        );
        assert_eq!(f.run(&[b"EXPIRE", b"t", b"100"]), ":1\r\n");
        assert_eq!(f.run(&[b"PERSIST", b"t"]), ":1\r\n");
        assert_eq!(f.run(&[b"RENAME", b"t", b"t3"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TYPE", b"t3"]), "+TDIS-TYPE\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"t3"]), "$3\r\nraw\r\n");
        assert_eq!(f.run(&[b"DEL", b"t3"]), ":1\r\n");
        // An empty digest is still a key, so the twelve that are not the
        // constructor all say the same thing once it is gone.
        assert_eq!(
            f.run(&[b"TDIGEST.INFO", b"t3"]),
            "-ERR T-Digest: key does not exist\r\n"
        );
        // The key is looked at before the arguments, so a bad argument at a key
        // that is not there still says the key is not there.
        assert_eq!(
            f.run(&[b"TDIGEST.QUANTILE", b"t3", b"zzz"]),
            "-ERR T-Digest: key does not exist\r\n"
        );
    }

    // -------------------------------------------------------------------- ts

    /// A `TS.INFO` reply with the memory usage taken out of it.
    ///
    /// That number is what a series costs here rather than what one costs in the
    /// module, which is D-53, and it moves whenever the layout of a chunk does.
    /// Everything either side of it is the wire contract and is worth pinning
    /// down exactly, so the tests below check the whole reply with the one
    /// number lifted out.
    fn without_memory(reply: &str) -> String {
        let head = "+memoryUsage\r\n:";
        let at = reply.find(head).expect("every TS.INFO reports memory");
        let rest = &reply[at + head.len()..];
        let end = rest.find("\r\n").expect("and it is a whole number");
        format!("{}{}", &reply[..at + head.len()], &rest[end..])
    }

    /// A series is made empty and still says it has a chunk, and the options are
    /// read before the key is looked at.
    #[test]
    fn a_series_is_made_empty_and_reports_on_itself() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"TS.CREATE", b"t"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TYPE", b"t"]), "+TSDB-TYPE\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"t"]), "$3\r\nraw\r\n");
        // Fourteen fields, so twenty eight elements. An empty series reports one
        // chunk and zero at both ends, and neither the chunk type nor the
        // duplicate policy is ever a nil.
        assert_eq!(
            without_memory(&f.run(&[b"TS.INFO", b"t"])),
            "*28\r\n\
             +totalSamples\r\n:0\r\n\
             +memoryUsage\r\n:\r\n\
             +firstTimestamp\r\n:0\r\n\
             +lastTimestamp\r\n:0\r\n\
             +retentionTime\r\n:0\r\n\
             +chunkCount\r\n:1\r\n\
             +chunkSize\r\n:4096\r\n\
             +chunkType\r\n+compressed\r\n\
             +duplicatePolicy\r\n+block\r\n\
             +labels\r\n*0\r\n\
             +sourceKey\r\n$-1\r\n\
             +rules\r\n*0\r\n\
             +ignoreMaxTimeDiff\r\n:0\r\n\
             +ignoreMaxValDiff\r\n$1\r\n0\r\n"
        );
        // A key that is already there is about the key whatever it holds, and
        // the existence is what is checked rather than the type.
        assert_eq!(
            f.run(&[b"TS.CREATE", b"t"]),
            "-ERR TSDB: key already exists\r\n"
        );
        assert_eq!(f.run(&[b"SET", b"str", b"x"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"TS.CREATE", b"str"]),
            "-ERR TSDB: key already exists\r\n"
        );
        // But the arguments are read first, so a bad one at a key that is there
        // answers about the argument.
        assert_eq!(
            f.run(&[b"TS.CREATE", b"t", b"RETENTION", b"abc"]),
            "-ERR TSDB: Couldn't parse RETENTION\r\n"
        );
        // The seven that will not make a series say WRONGTYPE about a key
        // holding something else, where the two that would say a sentence.
        // The word is inside the sentence and not in front of it, because the
        // module writes its own error text and Redis puts ERR on the front of
        // anything a module writes.
        let wrong = "-ERR WRONGTYPE Operation against a key holding the wrong kind of value\r\n";
        assert_eq!(f.run(&[b"TS.INFO", b"str"]), wrong);
        assert_eq!(f.run(&[b"TS.GET", b"str"]), wrong);
        assert_eq!(f.run(&[b"TS.ALTER", b"str"]), wrong);
        assert_eq!(f.run(&[b"TS.DEL", b"str", b"0", b"1"]), wrong);
        assert_eq!(f.run(&[b"TS.INCRBY", b"str", b"1"]), wrong);
        assert_eq!(
            f.run(&[b"TS.ADD", b"str", b"1", b"1"]),
            "-ERR TSDB: the key is not a TSDB key\r\n"
        );
        // And the ones that will not make one say so about a key that is gone.
        assert_eq!(
            f.run(&[b"TS.INFO", b"nope"]),
            "-ERR TSDB: the key does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.GET", b"nope"]),
            "-ERR TSDB: the key does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.ALTER", b"nope"]),
            "-ERR TSDB: the key does not exist\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.DEL", b"nope", b"1", b"2"]),
            "-ERR TSDB: the key does not exist\r\n"
        );
    }

    /// Every option word, including the ones that are wrong, and the scan that
    /// finds them.
    #[test]
    fn the_options_are_a_keyword_scan_and_not_a_grammar() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"TS.CREATE",
                b"t",
                b"RETENTION",
                b"5000",
                b"ENCODING",
                b"UNCOMPRESSED",
                b"CHUNK_SIZE",
                b"128",
                b"DUPLICATE_POLICY",
                b"LAST",
                b"IGNORE",
                b"10",
                b"0.5",
                b"LABELS",
                b"room",
                b"kitchen"
            ]),
            "+OK\r\n"
        );
        let info = f.run(&[b"TS.INFO", b"t"]);
        assert!(info.contains("+retentionTime\r\n:5000\r\n"), "{info}");
        assert!(info.contains("+chunkSize\r\n:128\r\n"), "{info}");
        assert!(info.contains("+chunkType\r\n+uncompressed\r\n"), "{info}");
        assert!(info.contains("+duplicatePolicy\r\n+last\r\n"), "{info}");
        assert!(info.contains("+ignoreMaxTimeDiff\r\n:10\r\n"), "{info}");
        // A plain double here, where a sample value out of TS.GET is the
        // shortest digits that read back as the same number.
        assert!(
            info.contains("+ignoreMaxValDiff\r\n$3\r\n0.5\r\n"),
            "{info}"
        );
        assert!(
            info.contains("+labels\r\n*1\r\n*2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n"),
            "{info}"
        );

        // A word that is not an option is read past rather than refused.
        assert_eq!(f.run(&[b"TS.CREATE", b"junk", b"FOO"]), "+OK\r\n");
        // LABELS eats everything after it in pairs, and the later scans still
        // look inside what it ate, so this sets a retention and stores a label
        // called RETENTION at the same time.
        assert_eq!(
            f.run(&[
                b"TS.CREATE",
                b"g",
                b"LABELS",
                b"a",
                b"b",
                b"RETENTION",
                b"5"
            ]),
            "+OK\r\n"
        );
        let greedy = f.run(&[b"TS.INFO", b"g"]);
        assert!(greedy.contains("+retentionTime\r\n:5\r\n"), "{greedy}");
        assert!(
            greedy.contains("*2\r\n$1\r\na\r\n$1\r\nb\r\n*2\r\n$9\r\nRETENTION\r\n$1\r\n5\r\n"),
            "{greedy}"
        );

        // Every way an option can be wrong, in the order the module reads them.
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"LABELS", b"a", b"b(c"]),
            "-ERR TSDB: Couldn't parse LABELS\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"LABELS", b"", b"b"]),
            "-ERR TSDB: Couldn't parse LABELS\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"RETENTION"]),
            "-ERR TSDB: Couldn't parse RETENTION\r\n"
        );
        // A retention below zero is one of the two the module writes with no
        // ERR in front of it, where one that is not a number gets one.
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"RETENTION", b"-1"]),
            "-TSDB: Couldn't parse RETENTION\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"CHUNK_SIZE", b"abc"]),
            "-ERR TSDB: Couldn't parse CHUNK_SIZE\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"CHUNK_SIZE", b"100"]),
            "-ERR TSDB: CHUNK_SIZE value must be a multiple of 8 in the range [48 .. 1048576]\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"ENCODING", b"nope"]),
            "-ERR TSDB: unknown ENCODING parameter\r\n"
        );
        // And an ENCODING with nothing behind it is an arity error where every
        // other keyword in the same spot is a sentence.
        assert!(
            f.run(&[b"TS.CREATE", b"e", b"ENCODING"])
                .contains("wrong number of arguments for 'ts.create' command")
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"DUPLICATE_POLICY"]),
            "-ERR TSDB: Couldn't parse DUPLICATE_POLICY\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"DUPLICATE_POLICY", b"nope"]),
            "-ERR TSDB: Unknown DUPLICATE_POLICY\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"IGNORE", b"10"]),
            "-ERR TSDB: Couldn't parse IGNORE\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.CREATE", b"e", b"IGNORE", b"-1", b"1"]),
            "-ERR TSDB: IGNORE arguments cannot be negative\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"e"]), ":0\r\n");

        // An alter changes what was named and leaves the rest alone, and reads
        // an encoding only far enough to refuse a bad one.
        assert_eq!(f.run(&[b"TS.ALTER", b"t", b"RETENTION", b"9"]), "+OK\r\n");
        let after = f.run(&[b"TS.INFO", b"t"]);
        assert!(after.contains("+retentionTime\r\n:9\r\n"), "{after}");
        assert!(after.contains("+chunkSize\r\n:128\r\n"), "{after}");
        assert!(after.contains("+duplicatePolicy\r\n+last\r\n"), "{after}");
        assert_eq!(
            f.run(&[b"TS.ALTER", b"t", b"ENCODING", b"nope"]),
            "-ERR TSDB: unknown ENCODING parameter\r\n"
        );
        // An encoding it does take is still not applied.
        assert_eq!(
            f.run(&[b"TS.ALTER", b"t", b"ENCODING", b"COMPRESSED"]),
            "+OK\r\n"
        );
        assert!(
            f.run(&[b"TS.INFO", b"t"])
                .contains("+chunkType\r\n+uncompressed\r\n")
        );
    }

    /// Samples go in, come back out and are refused for the reasons the module
    /// refuses them.
    #[test]
    fn samples_land_where_they_are_put_and_the_newest_comes_back() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"TS.ADD", b"t", b"100", b"1.5"]), ":100\r\n");
        // The series was made on the way in.
        assert_eq!(f.run(&[b"TYPE", b"t"]), "+TSDB-TYPE\r\n");
        assert_eq!(f.run(&[b"TS.ADD", b"t", b"200", b"2"]), ":200\r\n");
        // A sample value goes out as a simple string of the shortest digits
        // that read back as the same number.
        assert_eq!(f.run(&[b"TS.GET", b"t"]), "*2\r\n:200\r\n+2\r\n");
        assert_eq!(f.run(&[b"TS.ADD", b"t", b"300", b"1e300"]), ":300\r\n");
        assert_eq!(f.run(&[b"TS.GET", b"t"]), "*2\r\n:300\r\n+1E300\r\n");
        // An empty series has no newest sample and answers an empty array
        // rather than a nil.
        assert_eq!(f.run(&[b"TS.CREATE", b"empty"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TS.GET", b"empty"]), "*0\r\n");

        // The value is read before the key, so a bad one against a key holding
        // a string is about the value.
        assert_eq!(f.run(&[b"SET", b"str", b"x"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"TS.ADD", b"str", b"1", b".5"]),
            "-ERR TSDB: invalid value\r\n"
        );
        // The grammar is tighter than the one a number argument usually gets:
        // no leading plus, no bare fraction, no infinity and nothing that does
        // not fit.
        for bad in [
            &b".5"[..],
            b"1.",
            b"+1",
            b" 1",
            b"0x10",
            b"inf",
            b"1e400",
            b"--1",
            b"1e",
        ] {
            assert_eq!(
                f.run(&[b"TS.ADD", b"v", b"1", bad]),
                "-ERR TSDB: invalid value\r\n",
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
        // And a reading that is not a number is one of three words.
        assert_eq!(f.run(&[b"TS.ADD", b"v", b"1", b"NaN"]), ":1\r\n");

        // A timestamp that is not a number, and one that is and is below zero,
        // are two different sentences.
        assert_eq!(
            f.run(&[b"TS.ADD", b"t", b"abc", b"1"]),
            "-ERR TSDB: invalid timestamp\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.ADD", b"t", b"-1", b"1"]),
            "-ERR TSDB: invalid timestamp, must be a nonnegative integer\r\n"
        );

        // A repeated timestamp is blocked by default, and ON_DUPLICATE on the
        // command beats what the series was told.
        assert_eq!(
            f.run(&[b"TS.ADD", b"t", b"300", b"7"]),
            "-ERR TSDB: Error at upsert, update is not supported when DUPLICATE_POLICY is set to BLOCK mode, or either current or new value is NaN and DUPLICATE_POLICY is MAX/MIN/SUM\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.ADD", b"t", b"300", b"7", b"ON_DUPLICATE", b"LAST"]),
            ":300\r\n"
        );
        assert_eq!(f.run(&[b"TS.GET", b"t"]), "*2\r\n:300\r\n+7\r\n");
        // ON_DUPLICATE is only read when the key was already there, which is
        // why a policy word that is not a policy passes on a fresh key.
        assert_eq!(
            f.run(&[b"TS.ADD", b"fresh", b"1", b"1", b"ON_DUPLICATE", b"nope"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.ADD", b"fresh", b"2", b"1", b"ON_DUPLICATE", b"nope"]),
            "-ERR TSDB: Unknown DUPLICATE_POLICY\r\n"
        );

        // Retention is exact and it is checked before anything else happens, so
        // a sample landing behind the window is refused rather than trimmed.
        assert_eq!(f.run(&[b"TS.CREATE", b"r", b"RETENTION", b"50"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TS.ADD", b"r", b"1000", b"1"]), ":1000\r\n");
        assert_eq!(f.run(&[b"TS.ADD", b"r", b"960", b"1"]), ":960\r\n");
        assert_eq!(
            f.run(&[b"TS.ADD", b"r", b"940", b"1"]),
            "-ERR TSDB: Timestamp is older than retention\r\n"
        );
        // And the window trims as it moves.
        assert_eq!(f.run(&[b"TS.ADD", b"r", b"1100", b"1"]), ":1100\r\n");
        assert!(
            f.run(&[b"TS.INFO", b"r"])
                .contains("+totalSamples\r\n:1\r\n")
        );

        // An ignore window drops a sample close enough to the newest one to be
        // uninteresting, and answers the newest timestamp so a client can tell.
        assert_eq!(
            f.run(&[
                b"TS.CREATE",
                b"i",
                b"DUPLICATE_POLICY",
                b"LAST",
                b"IGNORE",
                b"10",
                b"0.5"
            ]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"TS.ADD", b"i", b"1000", b"1"]), ":1000\r\n");
        assert_eq!(f.run(&[b"TS.ADD", b"i", b"1005", b"1.2"]), ":1000\r\n");
        assert_eq!(f.run(&[b"TS.ADD", b"i", b"1005", b"9"]), ":1005\r\n");
    }

    /// Every triple in a `TS.MADD` is answered on its own, and none of them
    /// makes a series.
    #[test]
    fn a_madd_answers_each_triple_and_creates_nothing() {
        let mut f = Fixture::new();
        assert_eq!(f.run(&[b"TS.CREATE", b"a"]), "+OK\r\n");
        assert_eq!(f.run(&[b"TS.CREATE", b"b"]), "+OK\r\n");
        assert_eq!(
            f.run(&[
                b"TS.MADD", b"a", b"100", b"1", b"b", b"100", b"2", b"a", b"200", b"3"
            ]),
            "*3\r\n:100\r\n:100\r\n:200\r\n"
        );
        // A key that is not a series is an error in its own slot and the ones
        // after it still land.
        assert_eq!(f.run(&[b"SET", b"str", b"x"]), "+OK\r\n");
        assert_eq!(
            f.run(&[
                b"TS.MADD", b"gone", b"1", b"1", b"str", b"1", b"1", b"a", b"300", b"4"
            ]),
            "*3\r\n\
             -ERR TSDB: the key is not a TSDB key\r\n\
             -ERR TSDB: the key is not a TSDB key\r\n\
             :300\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"gone"]), ":0\r\n");
        // A bad value and a bad timestamp are answered in their slots too.
        assert_eq!(
            f.run(&[b"TS.MADD", b"a", b"400", b"zzz", b"a", b"abc", b"1"]),
            "*2\r\n-ERR TSDB: invalid value\r\n-ERR TSDB: invalid timestamp\r\n"
        );
        // And a list that is not made of triples is an arity error.
        assert!(
            f.run(&[b"TS.MADD", b"a", b"1", b"1", b"a"])
                .contains("wrong number of arguments for 'ts.madd' command")
        );
    }

    /// The two increments, which only ever write forwards.
    #[test]
    fn an_increment_walks_the_newest_value_up_and_down() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"t", b"5", b"TIMESTAMP", b"100"]),
            ":100\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"t", b"5", b"TIMESTAMP", b"100"]),
            ":100\r\n"
        );
        // Two on one timestamp add up rather than collide, because the sample
        // goes in under the last policy whatever the series says.
        assert_eq!(f.run(&[b"TS.GET", b"t"]), "*2\r\n:100\r\n+10\r\n");
        assert_eq!(
            f.run(&[b"TS.DECRBY", b"t", b"3", b"TIMESTAMP", b"200"]),
            ":200\r\n"
        );
        assert_eq!(f.run(&[b"TS.GET", b"t"]), "*2\r\n:200\r\n+7\r\n");
        // A timestamp behind the newest sample is the other of the two errors
        // the module writes with no ERR in front of it.
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"t", b"1", b"TIMESTAMP", b"150"]),
            "-TSDB: timestamp must be equal to or higher than the maximum existing timestamp\r\n"
        );
        // The increment goes through the ordinary number reader, so it takes
        // what a sample value will not and refuses a NaN that a sample value
        // takes.
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"p", b"+5", b"TIMESTAMP", b"1"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"q", b".5", b"TIMESTAMP", b"1"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"t", b"nan"]),
            "-ERR TSDB: invalid increase/decrease value\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"t", b"zzz"]),
            "-ERR TSDB: invalid increase/decrease value\r\n"
        );
        // A key holding something else is WRONGTYPE and is answered before the
        // number is looked at.
        assert_eq!(f.run(&[b"SET", b"str", b"x"]), "+OK\r\n");
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"str", b"zzz"]),
            "-ERR WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        // A TIMESTAMP keyword with nothing behind it is about the timestamp.
        // The reference reads one past the end of its own arguments here and
        // answers whatever was in that memory, so there is nothing to copy and
        // this answers the same thing every time.
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"t", b"1", b"TIMESTAMP"]),
            "-ERR TSDB: invalid timestamp\r\n"
        );
        // And one behind a LABELS is a label name rather than the keyword, so
        // this lands at the clock rather than at 5.
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"lab", b"1", b"LABELS", b"TIMESTAMP", b"5"]),
            format!(":{}\r\n", f.server.now_ms())
        );
        // Adding to a series whose newest value is not a number has no answer.
        assert_eq!(f.run(&[b"TS.ADD", b"n", b"1", b"nan"]), ":1\r\n");
        assert_eq!(
            f.run(&[b"TS.INCRBY", b"n", b"1", b"TIMESTAMP", b"2"]),
            "-ERR TSDB: cannot increment/decrement NaN value\r\n"
        );
    }

    /// Deleting a span, both ends included.
    #[test]
    fn deleting_takes_out_a_span_and_answers_how_many_went() {
        let mut f = Fixture::new();
        for at in [b"100".as_slice(), b"200", b"300", b"400"] {
            f.run(&[b"TS.ADD", b"t", at, b"1"]);
        }
        assert_eq!(f.run(&[b"TS.DEL", b"t", b"200", b"300"]), ":2\r\n");
        assert!(
            f.run(&[b"TS.INFO", b"t"])
                .contains("+totalSamples\r\n:2\r\n")
        );
        // Ends the wrong way round take nothing out rather than being an error.
        assert_eq!(f.run(&[b"TS.DEL", b"t", b"400", b"100"]), ":0\r\n");
        // The two open ends.
        assert_eq!(f.run(&[b"TS.DEL", b"t", b"-", b"+"]), ":2\r\n");
        // A series everything has been deleted from keeps its chunk and reports
        // zero at both ends again.
        let empty = f.run(&[b"TS.INFO", b"t"]);
        assert!(empty.contains("+totalSamples\r\n:0\r\n"), "{empty}");
        assert!(empty.contains("+chunkCount\r\n:1\r\n"), "{empty}");
        assert!(empty.contains("+firstTimestamp\r\n:0\r\n"), "{empty}");
        assert!(empty.contains("+lastTimestamp\r\n:0\r\n"), "{empty}");
        assert_eq!(f.run(&[b"TS.DEL", b"t", b"0", b"1000"]), ":0\r\n");
        // The two ends have their own sentences.
        assert_eq!(
            f.run(&[b"TS.DEL", b"t", b"abc", b"5"]),
            "-ERR TSDB: wrong fromTimestamp\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.DEL", b"t", b"5", b"abc"]),
            "-ERR TSDB: wrong toTimestamp\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.DEL", b"t", b"-5", b"5"]),
            "-ERR TSDB: wrong fromTimestamp\r\n"
        );
    }

    /// What RESP3 changes, which is the two places a number is written and the
    /// shape of `TS.INFO`.
    #[test]
    fn resp3_writes_a_sample_as_a_double_and_the_info_as_a_map() {
        let mut f = Fixture::new();
        f.out = Out::new(Proto::Resp3);
        assert_eq!(
            f.run(&[b"TS.CREATE", b"t", b"LABELS", b"room", b"kitchen"]),
            "+OK\r\n"
        );
        assert_eq!(f.run(&[b"TS.ADD", b"t", b"100", b"1e300"]), ":100\r\n");
        // A double rather than the simple string RESP2 gets.
        assert_eq!(f.run(&[b"TS.GET", b"t"]), "*2\r\n:100\r\n,1e+300\r\n");
        assert_eq!(
            without_memory(&f.run(&[b"TS.INFO", b"t"])),
            "%14\r\n\
             +totalSamples\r\n:1\r\n\
             +memoryUsage\r\n:\r\n\
             +firstTimestamp\r\n:100\r\n\
             +lastTimestamp\r\n:100\r\n\
             +retentionTime\r\n:0\r\n\
             +chunkCount\r\n:1\r\n\
             +chunkSize\r\n:4096\r\n\
             +chunkType\r\n+compressed\r\n\
             +duplicatePolicy\r\n+block\r\n\
             +labels\r\n%1\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
             +sourceKey\r\n_\r\n\
             +rules\r\n%0\r\n\
             +ignoreMaxTimeDiff\r\n:0\r\n\
             +ignoreMaxValDiff\r\n,0\r\n"
        );
    }

    /// Reading a span back, both ways round, with the two ends and the three
    /// things that trim what comes out.
    #[test]
    fn a_range_walks_a_span_and_a_revrange_walks_it_backwards() {
        let mut f = Fixture::new();
        for (at, v) in [
            (b"100".as_slice(), b"1".as_slice()),
            (b"200", b"2"),
            (b"300", b"3"),
            (b"400", b"4"),
        ] {
            f.run(&[b"TS.ADD", b"t", at, v]);
        }
        assert_eq!(
            f.run(&[b"TS.RANGE", b"t", b"-", b"+"]),
            "*4\r\n*2\r\n:100\r\n+1\r\n*2\r\n:200\r\n+2\r\n\
             *2\r\n:300\r\n+3\r\n*2\r\n:400\r\n+4\r\n"
        );
        // Both ends are included.
        assert_eq!(
            f.run(&[b"TS.RANGE", b"t", b"150", b"350"]),
            "*2\r\n*2\r\n:200\r\n+2\r\n*2\r\n:300\r\n+3\r\n"
        );
        // Backwards, and the count takes from the front of what comes out, so
        // backwards it takes the newest.
        assert_eq!(
            f.run(&[b"TS.REVRANGE", b"t", b"-", b"+", b"COUNT", b"2"]),
            "*2\r\n*2\r\n:400\r\n+4\r\n*2\r\n:300\r\n+3\r\n"
        );
        // Ends the wrong way round are empty rather than an error.
        assert_eq!(f.run(&[b"TS.RANGE", b"t", b"400", b"100"]), "*0\r\n");
        // The two filters.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"FILTER_BY_VALUE",
                b"2",
                b"3"
            ]),
            "*2\r\n*2\r\n:200\r\n+2\r\n*2\r\n:300\r\n+3\r\n"
        );
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"FILTER_BY_TS",
                b"100",
                b"400"
            ]),
            "*2\r\n*2\r\n:100\r\n+1\r\n*2\r\n:400\r\n+4\r\n"
        );
        // A word that is not an option is ignored wherever it sits.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"ZZZ",
                b"FILTER_BY_TS",
                b"400"
            ]),
            "*1\r\n*2\r\n:400\r\n+4\r\n"
        );
        // `LATEST` means nothing until there is a compaction rule to follow.
        assert_eq!(
            f.run(&[b"TS.RANGE", b"t", b"-", b"+", b"LATEST", b"COUNT", b"1"]),
            "*1\r\n*2\r\n:100\r\n+1\r\n"
        );
    }

    /// The bucketing, which is one column a reduction and a flat row.
    #[test]
    fn aggregation_puts_one_column_a_reduction_in_a_flat_row() {
        let mut f = Fixture::new();
        for (at, v) in [
            (b"100".as_slice(), b"1".as_slice()),
            (b"200", b"2"),
            (b"300", b"3"),
            (b"400", b"4"),
        ] {
            f.run(&[b"TS.ADD", b"t", at, v]);
        }
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"200"
            ]),
            "*3\r\n*2\r\n:0\r\n+1\r\n*2\r\n:200\r\n+2.5\r\n*2\r\n:400\r\n+4\r\n"
        );
        // Three reductions is a row of four and not a row of two with a nested
        // three in it.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"AGGREGATION",
                b"min,max,count",
                b"200"
            ]),
            "*3\r\n\
             *4\r\n:0\r\n+1\r\n+1\r\n+1\r\n\
             *4\r\n:200\r\n+2\r\n+3\r\n+2\r\n\
             *4\r\n:400\r\n+4\r\n+4\r\n+1\r\n"
        );
        // The timestamp a bucket is reported under.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"200",
                b"BUCKETTIMESTAMP",
                b"+"
            ]),
            "*3\r\n*2\r\n:200\r\n+1\r\n*2\r\n:400\r\n+2.5\r\n*2\r\n:600\r\n+4\r\n"
        );
        // An alignment moves where the bucket edges land.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"100",
                b"400",
                b"ALIGN",
                b"100",
                b"AGGREGATION",
                b"sum",
                b"200"
            ]),
            "*2\r\n*2\r\n:100\r\n+3\r\n*2\r\n:300\r\n+7\r\n"
        );
        // A `COUNT` sitting where the reduction name belongs is that name, and
        // the scan for a real one starts again two words later.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"AGGREGATION",
                b"count",
                b"200"
            ]),
            "*3\r\n*2\r\n:0\r\n+1\r\n*2\r\n:200\r\n+2\r\n*2\r\n:400\r\n+1\r\n"
        );
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"t",
                b"-",
                b"+",
                b"AGGREGATION",
                b"count",
                b"200",
                b"COUNT",
                b"1"
            ]),
            "*1\r\n*2\r\n:0\r\n+1\r\n"
        );
    }

    /// `EMPTY` fills the gaps between readings and nothing else, and `last`
    /// carries two different things depending on which kind of empty it is.
    #[test]
    fn empty_fills_a_gap_and_last_carries_the_reading_before_it() {
        let mut f = Fixture::new();
        for (at, v) in [
            (b"0".as_slice(), b"1".as_slice()),
            (b"100", b"2"),
            (b"500", b"nan"),
            (b"600", b"3"),
        ] {
            f.run(&[b"TS.ADD", b"g", at, v]);
        }
        // Without `EMPTY` the buckets with nothing in them are not there at all,
        // and neither is the one holding only a reading that is not a number.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"g",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"100"
            ]),
            "*3\r\n*2\r\n:0\r\n+1\r\n*2\r\n:100\r\n+2\r\n*2\r\n:600\r\n+3\r\n"
        );
        // The sum of nothing is zero rather than not a number.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"g",
                b"-",
                b"+",
                b"AGGREGATION",
                b"sum",
                b"100",
                b"EMPTY"
            ]),
            "*7\r\n*2\r\n:0\r\n+1\r\n*2\r\n:100\r\n+2\r\n*2\r\n:200\r\n+0\r\n\
             *2\r\n:300\r\n+0\r\n*2\r\n:400\r\n+0\r\n*2\r\n:500\r\n+0\r\n\
             *2\r\n:600\r\n+3\r\n"
        );
        // Buckets 200 through 400 have no readings at all and carry the reading
        // before the gap either way round. Bucket 500 has a reading that is not
        // a number, so it carries whatever the bucket before it in the reading
        // direction answered, which is 2 forwards and 3 backwards.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"g",
                b"-",
                b"+",
                b"AGGREGATION",
                b"last",
                b"100",
                b"EMPTY"
            ]),
            "*7\r\n*2\r\n:0\r\n+1\r\n*2\r\n:100\r\n+2\r\n*2\r\n:200\r\n+2\r\n\
             *2\r\n:300\r\n+2\r\n*2\r\n:400\r\n+2\r\n*2\r\n:500\r\n+2\r\n\
             *2\r\n:600\r\n+3\r\n"
        );
        assert_eq!(
            f.run(&[
                b"TS.REVRANGE",
                b"g",
                b"-",
                b"+",
                b"AGGREGATION",
                b"last",
                b"100",
                b"EMPTY"
            ]),
            "*7\r\n*2\r\n:600\r\n+3\r\n*2\r\n:500\r\n+3\r\n*2\r\n:400\r\n+2\r\n\
             *2\r\n:300\r\n+2\r\n*2\r\n:200\r\n+2\r\n*2\r\n:100\r\n+2\r\n\
             *2\r\n:0\r\n+1\r\n"
        );
        // And a window that opens on that bucket has nothing in range before it
        // to carry, so it answers not a number.
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"g",
                b"500",
                b"600",
                b"AGGREGATION",
                b"last",
                b"100",
                b"EMPTY"
            ]),
            "*2\r\n*2\r\n:500\r\n+NaN\r\n*2\r\n:600\r\n+3\r\n"
        );
    }

    /// The sentences a read answers when its options do not add up, which are
    /// the module's own word for word.
    #[test]
    fn a_range_says_what_the_module_says_when_the_options_do_not_add_up() {
        let mut f = Fixture::new();
        f.run(&[b"TS.ADD", b"t", b"100", b"1"]);
        f.run(&[b"SET", b"str", b"x"]);
        let cases: &[(&[&[u8]], &str)] = &[
            (
                &[b"TS.RANGE", b"t"],
                "-ERR wrong number of arguments for 'ts.range' command\r\n",
            ),
            // The key is resolved before a single option is read.
            (
                &[b"TS.RANGE", b"gone", b"-", b"+", b"COUNT", b"x"],
                "-ERR TSDB: the key does not exist\r\n",
            ),
            (
                &[b"TS.RANGE", b"str", b"-", b"+"],
                "-ERR WRONGTYPE Operation against a key holding the wrong kind of value\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"abc", b"+"],
                "-ERR TSDB: wrong fromTimestamp\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"abc"],
                "-ERR TSDB: wrong toTimestamp\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"COUNT"],
                "-ERR TSDB: COUNT argument is missing\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"COUNT", b"x"],
                "-ERR TSDB: Couldn't parse COUNT\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"COUNT", b"0"],
                "-ERR TSDB: Invalid COUNT value\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"AGGREGATION", b"avg"],
                "-ERR TSDB: Couldn't parse AGGREGATION\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"AGGREGATION", b"avg", b"x"],
                "-ERR TSDB: Couldn't parse AGGREGATION\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"nope",
                    b"100",
                ],
                "-ERR TSDB: Unknown aggregation type\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg,,min",
                    b"100",
                ],
                "-ERR TSDB: Empty aggregation type in list\r\n",
            ),
            // The list of names is read before the width is looked at.
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"AGGREGATION", b"nope", b"0"],
                "-ERR TSDB: Unknown aggregation type\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"AGGREGATION", b"avg", b"0"],
                "-ERR TSDB: bucketDuration must be greater than zero\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                    b"100",
                    b"X",
                    b"EMPTY",
                ],
                "-ERR TSDB: EMPTY flag should be the 3rd or 5th flag after AGGREGATION flag\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                    b"100",
                    b"BUCKETTIMESTAMP",
                    b"z",
                ],
                "-ERR TSDB: unknown BUCKETTIMESTAMP parameter\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                    b"100",
                    b"X",
                    b"Y",
                    b"BUCKETTIMESTAMP",
                    b"-",
                ],
                "-ERR TSDB: BUCKETTIMESTAMP flag should be the 3rd or 4th flag after \
                 AGGREGATION flag\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"ALIGN",
                    b"z",
                    b"AGGREGATION",
                    b"avg",
                    b"100",
                ],
                "-ERR TSDB: unknown ALIGN parameter\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"ALIGN", b"5"],
                "-ERR TSDB: ALIGN parameter can only be used with AGGREGATION\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"ALIGN",
                    b"-",
                    b"AGGREGATION",
                    b"avg",
                    b"100",
                ],
                "-ERR TSDB: start alignment can only be used with explicit start timestamp\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"FILTER_BY_VALUE", b"1"],
                "-ERR TSDB: FILTER_BY_VALUE one or more arguments are missing\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"FILTER_BY_VALUE",
                    b"x",
                    b"2",
                ],
                "-ERR TSDB: Couldn't parse MIN\r\n",
            ),
            (
                &[
                    b"TS.RANGE",
                    b"t",
                    b"-",
                    b"+",
                    b"FILTER_BY_VALUE",
                    b"1",
                    b"y",
                ],
                "-ERR TSDB: Couldn't parse MAX\r\n",
            ),
            (
                &[b"TS.RANGE", b"t", b"-", b"+", b"FILTER_BY_TS"],
                "-ERR TSDB: FILTER_BY_TS one or more arguments are missing\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{:?}", argv.last());
        }
        // The one sentence here that is yo's own rather than the module's, which
        // is D-54. A read that would build more rows than yo will build is
        // refused instead of attempted.
        f.run(&[b"TS.ADD", b"wide", b"0", b"1"]);
        f.run(&[b"TS.ADD", b"wide", b"1000000000000", b"2"]);
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"wide",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"1",
                b"EMPTY"
            ]),
            "-ERR TSDB: the requested range holds too many empty buckets\r\n"
        );
    }

    /// What RESP3 changes on a read, which is only how a number is written.
    #[test]
    fn resp3_writes_a_read_value_as_a_double() {
        let mut f = Fixture::new();
        f.out = Out::new(Proto::Resp3);
        for (at, v) in [
            (b"0".as_slice(), b"1".as_slice()),
            (b"100", b"2"),
            (b"500", b"nan"),
            (b"600", b"3"),
        ] {
            f.run(&[b"TS.ADD", b"g", at, v]);
        }
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"g",
                b"0",
                b"100",
                b"AGGREGATION",
                b"avg,min",
                b"200"
            ]),
            "*1\r\n*3\r\n:0\r\n,1.5\r\n,1\r\n"
        );
        assert_eq!(
            f.run(&[
                b"TS.RANGE",
                b"g",
                b"500",
                b"600",
                b"AGGREGATION",
                b"last",
                b"100",
                b"EMPTY"
            ]),
            "*2\r\n*2\r\n:500\r\n,nan\r\n*2\r\n:600\r\n,3\r\n"
        );
    }

    /// Two series with an overlap and a gap each, plus a third holding nothing,
    /// which is what the joined reads are measured against.
    fn joined() -> Fixture {
        let mut f = Fixture::new();
        f.run(&[b"TS.CREATE", b"z"]);
        for (at, v) in [
            (b"10".as_slice(), b"1".as_slice()),
            (b"20", b"2"),
            (b"40", b"4"),
            (b"50", b"5"),
        ] {
            f.run(&[b"TS.ADD", b"x", at, v]);
        }
        for (at, v) in [
            (b"20".as_slice(), b"20".as_slice()),
            (b"30", b"30"),
            (b"50", b"50"),
            (b"60", b"60"),
        ] {
            f.run(&[b"TS.ADD", b"y", at, v]);
        }
        f
    }

    /// The joined read lines its keys up on the timestamp and writes a row as
    /// the timestamp and then a nested array of the columns, which is the one
    /// shape in the family that is not the flat pair.
    #[test]
    fn an_nrange_joins_its_keys_on_the_timestamp() {
        let mut f = joined();
        // One key still nests, so the shape does not depend on the count.
        assert_eq!(
            f.run(&[b"TS.NRANGE", b"1", b"x", b"-", b"+"]),
            "*4\r\n*2\r\n:10\r\n*1\r\n+1\r\n*2\r\n:20\r\n*1\r\n+2\r\n\
             *2\r\n:40\r\n*1\r\n+4\r\n*2\r\n:50\r\n*1\r\n+5\r\n"
        );
        // A key with no reading where another key has one writes NaN there.
        assert_eq!(
            f.run(&[b"TS.NRANGE", b"2", b"x", b"y", b"-", b"+"]),
            "*6\r\n*2\r\n:10\r\n*2\r\n+1\r\n+NaN\r\n\
             *2\r\n:20\r\n*2\r\n+2\r\n+20\r\n\
             *2\r\n:30\r\n*2\r\n+NaN\r\n+30\r\n\
             *2\r\n:40\r\n*2\r\n+4\r\n+NaN\r\n\
             *2\r\n:50\r\n*2\r\n+5\r\n+50\r\n\
             *2\r\n:60\r\n*2\r\n+NaN\r\n+60\r\n"
        );
        // A series holding nothing is a column of NaN and never a row of its
        // own, and the same key twice answers twice.
        assert_eq!(
            f.run(&[b"TS.NRANGE", b"2", b"x", b"z", b"20", b"40"]),
            "*2\r\n*2\r\n:20\r\n*2\r\n+2\r\n+NaN\r\n*2\r\n:40\r\n*2\r\n+4\r\n+NaN\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.NRANGE", b"2", b"x", b"x", b"40", b"50"]),
            "*2\r\n*2\r\n:40\r\n*2\r\n+4\r\n+4\r\n*2\r\n:50\r\n*2\r\n+5\r\n+5\r\n"
        );
        // COUNT is applied to the joined rows and not to each key, so backwards
        // it gives the newest joined row rather than the newest of each.
        assert_eq!(
            f.run(&[
                b"TS.NREVRANGE",
                b"2",
                b"x",
                b"y",
                b"-",
                b"+",
                b"COUNT",
                b"1"
            ]),
            "*1\r\n*2\r\n:60\r\n*2\r\n+NaN\r\n+60\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.NRANGE", b"2", b"x", b"y", b"-", b"+", b"COUNT", b"1"]),
            "*1\r\n*2\r\n:10\r\n*2\r\n+1\r\n+NaN\r\n"
        );
        // The two sample filters are settled a key at a time, before the join.
        assert_eq!(
            f.run(&[
                b"TS.NRANGE",
                b"2",
                b"x",
                b"y",
                b"-",
                b"+",
                b"FILTER_BY_VALUE",
                b"2",
                b"30"
            ]),
            "*4\r\n*2\r\n:20\r\n*2\r\n+2\r\n+20\r\n\
             *2\r\n:30\r\n*2\r\n+NaN\r\n+30\r\n\
             *2\r\n:40\r\n*2\r\n+4\r\n+NaN\r\n\
             *2\r\n:50\r\n*2\r\n+5\r\n+NaN\r\n"
        );
    }

    /// The aggregation on a joined read names one reduction a key and then the
    /// one bucket width, and each name may be a comma list, so a row can be
    /// wider than the key count.
    #[test]
    fn an_nrange_aggregation_names_one_reduction_a_key() {
        let mut f = joined();
        assert_eq!(
            f.run(&[
                b"TS.NRANGE",
                b"2",
                b"x",
                b"y",
                b"-",
                b"+",
                b"AGGREGATION",
                b"sum",
                b"sum",
                b"20"
            ]),
            "*4\r\n*2\r\n:0\r\n*2\r\n+1\r\n+NaN\r\n\
             *2\r\n:20\r\n*2\r\n+2\r\n+50\r\n\
             *2\r\n:40\r\n*2\r\n+9\r\n+50\r\n\
             *2\r\n:60\r\n*2\r\n+NaN\r\n+60\r\n"
        );
        // A comma list on the first key widens the row to three columns.
        assert_eq!(
            f.run(&[
                b"TS.NRANGE",
                b"2",
                b"x",
                b"y",
                b"-",
                b"+",
                b"AGGREGATION",
                b"sum,count",
                b"avg",
                b"20"
            ]),
            "*4\r\n*2\r\n:0\r\n*3\r\n+1\r\n+1\r\n+NaN\r\n\
             *2\r\n:20\r\n*3\r\n+2\r\n+1\r\n+25\r\n\
             *2\r\n:40\r\n*3\r\n+9\r\n+2\r\n+50\r\n\
             *2\r\n:60\r\n*3\r\n+NaN\r\n+NaN\r\n+60\r\n"
        );
        // Everything behind the width moves along with it, so BUCKETTIMESTAMP
        // sits one or two past the width whatever the key count is.
        assert_eq!(
            f.run(&[
                b"TS.NRANGE",
                b"2",
                b"x",
                b"y",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"sum",
                b"100",
                b"EMPTY",
                b"BUCKETTIMESTAMP",
                b"end"
            ]),
            "*1\r\n*2\r\n:100\r\n*2\r\n+3\r\n+160\r\n"
        );
        // A COUNT landing in one of the name slots is a reduction name and not
        // the keyword, and the read then has no count at all.
        assert_eq!(
            f.run(&[
                b"TS.NRANGE",
                b"2",
                b"x",
                b"y",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"COUNT",
                b"100"
            ]),
            "*1\r\n*2\r\n:0\r\n*2\r\n+3\r\n+4\r\n"
        );
    }

    /// The sentences a joined read answers when it does not add up, which are
    /// the module's own and come out in the module's own order.
    #[test]
    fn an_nrange_says_what_the_module_says_when_it_does_not_add_up() {
        let mut f = joined();
        f.run(&[b"SET", b"str", b"hi"]);
        let bad_keys = "-ERR TSDB: numkeys must be a positive integer\r\n";
        let numkeys = "-ERR TSDB: the number of AGGREGATION arguments \
                       must be equal to numkeys\r\n";
        let cases: &[(&[&[u8]], &str)] = &[
            (&[b"TS.NRANGE", b"0", b"x", b"-", b"+"], bad_keys),
            (&[b"TS.NRANGE", b"-1", b"x", b"-", b"+"], bad_keys),
            (&[b"TS.NRANGE", b"abc", b"x", b"-", b"+"], bad_keys),
            // Not enough words behind the count for the keys and both ends of
            // the span, which is an arity error however many keys were named.
            (
                &[b"TS.NRANGE", b"2", b"x", b"-", b"+"],
                "-ERR wrong number of arguments for 'ts.nrange' command\r\n",
            ),
            (
                &[b"TS.NRANGE", b"99", b"x", b"-", b"+"],
                "-ERR wrong number of arguments for 'ts.nrange' command\r\n",
            ),
            // The reduction names are read before the two ends of the span,
            // which no other option is.
            (
                &[
                    b"TS.NRANGE",
                    b"2",
                    b"x",
                    b"y",
                    b"abc",
                    b"+",
                    b"AGGREGATION",
                    b"nope",
                    b"sum",
                    b"100",
                ],
                "-ERR TSDB: Unknown aggregation type\r\n",
            ),
            (
                &[b"TS.NRANGE", b"2", b"x", b"y", b"abc", b"+"],
                "-ERR TSDB: wrong fromTimestamp\r\n",
            ),
            (
                &[b"TS.NRANGE", b"2", b"x", b"y", b"-", b"abc"],
                "-ERR TSDB: wrong toTimestamp\r\n",
            ),
            // A name slot that is missing or holds a number is the count
            // sentence, and a width slot that is itself a reduction name is
            // that sentence as well.
            (
                &[
                    b"TS.NRANGE",
                    b"2",
                    b"x",
                    b"y",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                ],
                numkeys,
            ),
            (
                &[
                    b"TS.NRANGE",
                    b"2",
                    b"x",
                    b"y",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"100",
                    b"sum",
                    b"100",
                ],
                numkeys,
            ),
            (
                &[
                    b"TS.NRANGE",
                    b"2",
                    b"x",
                    b"y",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                    b"sum",
                    b"sum",
                    b"100",
                ],
                numkeys,
            ),
            (
                &[
                    b"TS.NRANGE",
                    b"2",
                    b"x",
                    b"y",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                    b"sum",
                    b"abc",
                ],
                "-ERR TSDB: Couldn't parse AGGREGATION\r\n",
            ),
            (
                &[
                    b"TS.NRANGE",
                    b"2",
                    b"x",
                    b"y",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"avg",
                    b"sum",
                    b"0",
                ],
                "-ERR TSDB: bucketDuration must be greater than zero\r\n",
            ),
            // With one key none of that applies and the plain parser runs, so a
            // lone width is a missing width rather than a count mismatch.
            (
                &[b"TS.NRANGE", b"1", b"x", b"-", b"+", b"AGGREGATION", b"100"],
                "-ERR TSDB: Couldn't parse AGGREGATION\r\n",
            ),
            (
                &[
                    b"TS.NRANGE",
                    b"1",
                    b"x",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"100",
                    b"200",
                ],
                "-ERR TSDB: Unknown aggregation type\r\n",
            ),
            // The keys come last and in the order they were named.
            (
                &[b"TS.NRANGE", b"2", b"x", b"nope", b"-", b"+"],
                "-ERR TSDB: the key does not exist\r\n",
            ),
            (
                &[b"TS.NRANGE", b"2", b"str", b"nope", b"-", b"+"],
                "-ERR WRONGTYPE Operation against a key \
                 holding the wrong kind of value\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{argv:?}");
        }
    }

    /// `TS.READ`, which is a key, one timestamp and everything from there on.
    #[test]
    fn a_read_walks_from_a_timestamp_to_the_end_of_the_series() {
        let mut f = joined();
        assert_eq!(
            f.run(&[b"TS.READ", b"x", b"-"]),
            "*4\r\n*2\r\n:10\r\n+1\r\n*2\r\n:20\r\n+2\r\n\
             *2\r\n:40\r\n+4\r\n*2\r\n:50\r\n+5\r\n"
        );
        // A plus is the last sample on its own, and a timestamp between two
        // samples starts at the one behind it.
        assert_eq!(
            f.run(&[b"TS.READ", b"x", b"+"]),
            "*1\r\n*2\r\n:50\r\n+5\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.READ", b"x", b"25"]),
            "*2\r\n*2\r\n:40\r\n+4\r\n*2\r\n:50\r\n+5\r\n"
        );
        // Past the end, a series holding nothing and a key that is not there
        // are all the empty array rather than an error.
        assert_eq!(f.run(&[b"TS.READ", b"x", b"99"]), "*0\r\n");
        assert_eq!(f.run(&[b"TS.READ", b"z", b"-"]), "*0\r\n");
        assert_eq!(f.run(&[b"TS.READ", b"z", b"+"]), "*0\r\n");
        assert_eq!(f.run(&[b"TS.READ", b"nope", b"-"]), "*0\r\n");
        // The timestamp refusal goes out with nothing in front of it, and a key
        // holding something else answers the bare WRONGTYPE rather than the
        // module's prefixed one, both unlike the rest of the family.
        assert_eq!(
            f.run(&[b"TS.READ", b"x", b"abc"]),
            "-TSDB: invalid timestamp\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.READ", b"x", b"-1"]),
            "-TSDB: invalid timestamp\r\n"
        );
        f.run(&[b"SET", b"str", b"hi"]);
        assert_eq!(
            f.run(&[b"TS.READ", b"str", b"-"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        // Anything other than exactly three words is an arity error, so there
        // is nowhere to put an option even though the table says minus three.
        assert_eq!(
            f.run(&[b"TS.READ", b"x"]),
            "-ERR wrong number of arguments for 'ts.read' command\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.READ", b"x", b"-", b"COUNT", b"1"]),
            "-ERR wrong number of arguments for 'ts.read' command\r\n"
        );
    }

    /// The keys of a joined read sit behind a count, so `COMMAND GETKEYS` has
    /// to read the count to find them.
    #[test]
    fn getkeys_reads_the_count_of_a_joined_read() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[
                b"COMMAND",
                b"GETKEYS",
                b"TS.NRANGE",
                b"2",
                b"a",
                b"b",
                b"-",
                b"+"
            ]),
            "*2\r\n$1\r\na\r\n$1\r\nb\r\n"
        );
        assert_eq!(
            f.run(&[
                b"COMMAND",
                b"GETKEYS",
                b"TS.NREVRANGE",
                b"1",
                b"a",
                b"-",
                b"+"
            ]),
            "*1\r\n$1\r\na\r\n"
        );
        // A count of zero, or one too large for the words that follow it, is
        // the server's own refusal and not the module's.
        for n in [b"0".as_slice(), b"9", b"abc"] {
            assert_eq!(
                f.run(&[b"COMMAND", b"GETKEYS", b"TS.NRANGE", n, b"a", b"-", b"+"]),
                "-ERR Invalid arguments specified for command\r\n"
            );
        }
    }

    /// The five series every test of the label surface works against.
    fn labelled() -> Fixture {
        let mut f = Fixture::new();
        f.run(&[
            b"TS.CREATE",
            b"a",
            b"LABELS",
            b"room",
            b"kitchen",
            b"x",
            b"1",
        ]);
        f.run(&[
            b"TS.CREATE",
            b"b",
            b"LABELS",
            b"room",
            b"bedroom",
            b"x",
            b"2",
        ]);
        f.run(&[b"TS.CREATE", b"c", b"LABELS", b"room", b"kitchen"]);
        f.run(&[b"TS.CREATE", b"d"]);
        f.run(&[b"TS.CREATE", b"e", b"LABELS", b"r", b"bb", b"r", b"b"]);
        f.run(&[b"TS.ADD", b"a", b"100", b"1.5"]);
        f.run(&[b"TS.ADD", b"b", b"200", b"2"]);
        f
    }

    /// The filter grammar, which is four steps and a `strtok` rather than a
    /// grammar, and which every command that searches on labels shares.
    #[test]
    fn a_filter_is_taken_apart_the_way_the_module_takes_one_apart() {
        let mut f = labelled();
        let cases: &[(&[&[u8]], &str)] = &[
            // The plain forms, and the order the answer comes back in, which is
            // by key name and not by anything the series remembers.
            (
                &[b"TS.QUERYINDEX", b"room=kitchen"],
                "*2\r\n$1\r\na\r\n$1\r\nc\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room=kitchen", b"x=1"],
                "*1\r\n$1\r\na\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room=(kitchen,bedroom)"],
                "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n",
            ),
            // An empty list still counts as something that says which series to
            // take, it just never takes any.
            (&[b"TS.QUERYINDEX", b"room=()"], "*0\r\n"),
            // Absent and present, neither of which stands on its own.
            (
                &[b"TS.QUERYINDEX", b"x=", b"room=kitchen"],
                "*1\r\n$1\r\nc\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room=kitchen", b"x!="],
                "*1\r\n$1\r\na\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room!=kitchen", b"x!="],
                "-ERR TSDB: please provide at least one matcher\r\n",
            ),
            // A run of separators is one separator and everything past the
            // second field is dropped, so all three of these ask one question.
            (
                &[b"TS.QUERYINDEX", b"room==kitchen"],
                "*2\r\n$1\r\na\r\n$1\r\nc\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room=kitchen=zz"],
                "*2\r\n$1\r\na\r\n$1\r\nc\r\n",
            ),
            (&[b"TS.QUERYINDEX", b"room!!=kitchen", b"x=1"], "*0\r\n"),
            // A bracket is only a list when it sits straight behind the
            // separator, and then the label in front of it has to be there.
            (&[b"TS.QUERYINDEX", b"()=1"], "*0\r\n"),
            (
                &[b"TS.QUERYINDEX", b"=(1)"],
                "-ERR TSDB: failed parsing labels\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room=(kitchen,)"],
                "-ERR TSDB: failed parsing labels\r\n",
            ),
            (
                &[b"TS.QUERYINDEX", b"room=(kitchen"],
                "-ERR TSDB: failed parsing labels\r\n",
            ),
            (&[b"TS.QUERYINDEX", b"room=x()"], "*0\r\n"),
            (
                &[b"TS.QUERYINDEX", b"nonsense"],
                "-ERR TSDB: failed parsing labels\r\n",
            ),
            // Nothing here says which series to take.
            (
                &[b"TS.QUERYINDEX", b"room!=kitchen"],
                "-ERR TSDB: please provide at least one matcher\r\n",
            ),
            // Names and values are both compared byte for byte.
            (&[b"TS.QUERYINDEX", b"ROOM=kitchen"], "*0\r\n"),
            (&[b"TS.QUERYINDEX", b"room=KITCHEN"], "*0\r\n"),
            (
                &[b"TS.QUERYINDEX"],
                "-ERR wrong number of arguments for 'ts.queryindex' command\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{:?}", argv.last());
        }
    }

    /// `TS.QUERYLABELS`, whose filter is the one that is allowed to be missing.
    #[test]
    fn querylabels_says_which_names_are_worn_and_what_they_are_set_to() {
        let mut f = labelled();
        let cases: &[(&[&[u8]], &str)] = &[
            (
                &[b"TS.QUERYLABELS", b"LABELS"],
                "*3\r\n$1\r\nr\r\n$4\r\nroom\r\n$1\r\nx\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"LABELS", b"FILTER", b"room=kitchen"],
                "*2\r\n$4\r\nroom\r\n$1\r\nx\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"VALUES", b"room"],
                "*2\r\n$7\r\nbedroom\r\n$7\r\nkitchen\r\n",
            ),
            // The series wearing `r` twice contributes the smaller of the two
            // here, which is not the one it was written down as first.
            (&[b"TS.QUERYLABELS", b"VALUES", b"r"], "*1\r\n$1\r\nb\r\n"),
            (&[b"TS.QUERYLABELS", b"VALUES", b"nolabel"], "*0\r\n"),
            (
                &[b"TS.QUERYLABELS", b"VALUES"],
                "-ERR wrong number of arguments for 'ts.querylabels' command\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"ZZZ"],
                "-ERR TSDB: unknown subtype, must be one of LABELS|VALUES\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"LABELS", b"ZZZ"],
                "-ERR TSDB: unknown argument, expected FILTER\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"LABELS", b"FILTER"],
                "-ERR TSDB: FILTER given with no filter expressions\r\n",
            ),
            // With no filter at all every series is taken, which is why the
            // first case here answers about `r` as well. A filter that is there
            // still has to say which series to take.
            (
                &[b"TS.QUERYLABELS", b"LABELS", b"FILTER", b"room!=kitchen"],
                "-ERR TSDB: please provide at least one matcher\r\n",
            ),
            (
                &[
                    b"TS.QUERYLABELS",
                    b"LABELS",
                    b"FILTER",
                    b"room=kitchen",
                    b"x=",
                ],
                "*1\r\n$4\r\nroom\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{:?}", argv.last());
        }
    }

    /// `TS.MGET`, the newest sample of every series a filter takes, and the two
    /// ways of asking for the labels back alongside it.
    #[test]
    fn mget_writes_the_newest_sample_and_the_labels_that_were_asked_for() {
        let mut f = labelled();
        let cases: &[(&[&[u8]], &str)] = &[
            // A series with no samples writes an empty array where the sample
            // goes rather than dropping out of the reply.
            (
                &[b"TS.MGET", b"FILTER", b"room=kitchen"],
                "*2\r\n*3\r\n$1\r\na\r\n*0\r\n*2\r\n:100\r\n+1.5\r\n\
                 *3\r\n$1\r\nc\r\n*0\r\n*0\r\n",
            ),
            (
                &[b"TS.MGET", b"WITHLABELS", b"FILTER", b"room=kitchen"],
                "*2\r\n*3\r\n$1\r\na\r\n*2\r\n*2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
                 *2\r\n$1\r\nx\r\n$1\r\n1\r\n*2\r\n:100\r\n+1.5\r\n\
                 *3\r\n$1\r\nc\r\n*1\r\n*2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n*0\r\n",
            ),
            // A selected label the series does not wear is a nil, not a gap.
            (
                &[
                    b"TS.MGET",
                    b"SELECTED_LABELS",
                    b"x",
                    b"FILTER",
                    b"room=kitchen",
                ],
                "*2\r\n*3\r\n$1\r\na\r\n*1\r\n*2\r\n$1\r\nx\r\n$1\r\n1\r\n\
                 *2\r\n:100\r\n+1.5\r\n\
                 *3\r\n$1\r\nc\r\n*1\r\n*2\r\n$1\r\nx\r\n$-1\r\n*0\r\n",
            ),
            // The other half of the duplicated name rule. This one takes the
            // first written down where `TS.QUERYLABELS` takes the smallest.
            (
                &[b"TS.MGET", b"SELECTED_LABELS", b"r", b"FILTER", b"r=b"],
                "*1\r\n*3\r\n$1\r\ne\r\n*1\r\n*2\r\n$1\r\nr\r\n$2\r\nbb\r\n*0\r\n",
            ),
            (
                &[b"TS.MGET", b"WITHLABELS", b"FILTER", b"r=b"],
                "*1\r\n*3\r\n$1\r\ne\r\n*2\r\n*2\r\n$1\r\nr\r\n$2\r\nbb\r\n\
                 *2\r\n$1\r\nr\r\n$1\r\nb\r\n*0\r\n",
            ),
            // A word that is not an option is ignored, but a missing `FILTER`
            // is an arity error whatever else was written.
            (
                &[b"TS.MGET", b"ZZZ", b"FILTER", b"room=bedroom"],
                "*1\r\n*3\r\n$1\r\nb\r\n*0\r\n*2\r\n:200\r\n+2\r\n",
            ),
            (
                &[b"TS.MGET", b"a", b"b", b"c"],
                "-ERR wrong number of arguments for 'ts.mget' command\r\n",
            ),
            (
                &[b"TS.MGET", b"FILTER"],
                "-ERR wrong number of arguments for 'ts.mget' command\r\n",
            ),
            // Both keyword checks happen before the filter is read, and the two
            // sentences spell the second keyword without its `ED`.
            (
                &[
                    b"TS.MGET",
                    b"WITHLABELS",
                    b"SELECTED_LABELS",
                    b"x",
                    b"FILTER",
                    b"bad",
                ],
                "-ERR TSDB: cannot accept WITHLABELS and SELECT_LABELS together\r\n",
            ),
            (
                &[b"TS.MGET", b"SELECTED_LABELS", b"FILTER", b"bad"],
                "-ERR TSDB: SELECT_LABELS should have at least 1 parameter\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{:?}", argv.last());
        }
    }

    /// What RESP3 changes across the label surface, which is a set where there
    /// was an array and a map where there was a pair of them.
    #[test]
    fn resp3_writes_the_label_surface_as_sets_and_maps() {
        let mut f = labelled();
        f.out = Out::new(Proto::Resp3);
        let cases: &[(&[&[u8]], &str)] = &[
            (
                &[b"TS.QUERYINDEX", b"room=kitchen"],
                "~2\r\n$1\r\na\r\n$1\r\nc\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"LABELS"],
                "~3\r\n$1\r\nr\r\n$4\r\nroom\r\n$1\r\nx\r\n",
            ),
            (
                &[b"TS.QUERYLABELS", b"VALUES", b"room"],
                "~2\r\n$7\r\nbedroom\r\n$7\r\nkitchen\r\n",
            ),
            // The key stops being the first of three and becomes the map key,
            // and the labels stop being pairs and become a map of their own.
            (
                &[b"TS.MGET", b"FILTER", b"room=kitchen"],
                "%2\r\n$1\r\na\r\n*2\r\n%0\r\n*2\r\n:100\r\n,1.5\r\n\
                 $1\r\nc\r\n*2\r\n%0\r\n*0\r\n",
            ),
            (
                &[b"TS.MGET", b"WITHLABELS", b"FILTER", b"room=kitchen"],
                "%2\r\n$1\r\na\r\n*2\r\n%2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
                 $1\r\nx\r\n$1\r\n1\r\n*2\r\n:100\r\n,1.5\r\n\
                 $1\r\nc\r\n*2\r\n%1\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n*0\r\n",
            ),
            (
                &[
                    b"TS.MGET",
                    b"SELECTED_LABELS",
                    b"x",
                    b"FILTER",
                    b"room=kitchen",
                ],
                "%2\r\n$1\r\na\r\n*2\r\n%1\r\n$1\r\nx\r\n$1\r\n1\r\n\
                 *2\r\n:100\r\n,1.5\r\n\
                 $1\r\nc\r\n*2\r\n%1\r\n$1\r\nx\r\n_\r\n*0\r\n",
            ),
            // A map with a name in it twice, which is what a series wearing one
            // label name twice turns into.
            (
                &[b"TS.MGET", b"WITHLABELS", b"FILTER", b"r=b"],
                "%1\r\n$1\r\ne\r\n*2\r\n%2\r\n$1\r\nr\r\n$2\r\nbb\r\n\
                 $1\r\nr\r\n$1\r\nb\r\n*0\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{:?}", argv.last());
        }
    }

    /// The same five series with enough samples in them for a group to have
    /// something to fold.
    fn spanned() -> Fixture {
        let mut f = labelled();
        f.run(&[b"TS.ADD", b"a", b"200", b"2.5"]);
        f.run(&[b"TS.ADD", b"c", b"100", b"10"]);
        f.run(&[b"TS.ADD", b"c", b"300", b"30"]);
        f
    }

    /// A span read out of every series a filter takes, with and without a group
    /// over the top of it.
    #[test]
    fn mrange_reads_every_series_and_folds_the_groups_it_is_asked_for() {
        let mut f = spanned();
        let cases: &[(&[&[u8]], &str)] = &[
            (
                &[b"TS.MRANGE", b"-", b"+", b"FILTER", b"room=kitchen"],
                "*2\r\n*3\r\n$1\r\na\r\n*0\r\n*2\r\n*2\r\n:100\r\n+1.5\r\n*2\r\n:200\r\n+2.5\r\n\
                 *3\r\n$1\r\nc\r\n*0\r\n*2\r\n*2\r\n:100\r\n+10\r\n*2\r\n:300\r\n+30\r\n",
            ),
            // Newest first is applied to each series before anything else sees
            // the rows.
            (
                &[
                    b"TS.MREVRANGE",
                    b"-",
                    b"+",
                    b"WITHLABELS",
                    b"FILTER",
                    b"room=kitchen",
                ],
                "*2\r\n*3\r\n$1\r\na\r\n*2\r\n*2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
                 *2\r\n$1\r\nx\r\n$1\r\n1\r\n*2\r\n*2\r\n:200\r\n+2.5\r\n*2\r\n:100\r\n+1.5\r\n\
                 *3\r\n$1\r\nc\r\n*1\r\n*2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
                 *2\r\n*2\r\n:300\r\n+30\r\n*2\r\n:100\r\n+10\r\n",
            ),
            // A label a series does not wear comes back against a nil rather
            // than being left out.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"SELECTED_LABELS",
                    b"x",
                    b"FILTER",
                    b"room=kitchen",
                ],
                "*2\r\n*3\r\n$1\r\na\r\n*1\r\n*2\r\n$1\r\nx\r\n$1\r\n1\r\n\
                 *2\r\n*2\r\n:100\r\n+1.5\r\n*2\r\n:200\r\n+2.5\r\n\
                 *3\r\n$1\r\nc\r\n*1\r\n*2\r\n$1\r\nx\r\n$-1\r\n\
                 *2\r\n*2\r\n:100\r\n+10\r\n*2\r\n:300\r\n+30\r\n",
            ),
            // The fold: 100 is in both series and adds up, the other two are in
            // one each and are still rows.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"sum",
                ],
                "*1\r\n*3\r\n$12\r\nroom=kitchen\r\n*0\r\n*3\r\n*2\r\n:100\r\n+11.5\r\n\
                 *2\r\n:200\r\n+2.5\r\n*2\r\n:300\r\n+30\r\n",
            ),
            // RESP2 has nowhere to put the reducer and the member keys, so a
            // group wearing labels writes them as two more labels.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"WITHLABELS",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"max",
                ],
                "*1\r\n*3\r\n$12\r\nroom=kitchen\r\n*3\r\n*2\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
                 *2\r\n$11\r\n__reducer__\r\n$3\r\nmax\r\n\
                 *2\r\n$10\r\n__source__\r\n$3\r\na,c\r\n\
                 *3\r\n*2\r\n:100\r\n+10\r\n*2\r\n:200\r\n+2.5\r\n*2\r\n:300\r\n+30\r\n",
            ),
            // A count is applied to each member and then again to the fold.
            (
                &[
                    b"TS.MREVRANGE",
                    b"-",
                    b"+",
                    b"COUNT",
                    b"1",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"count",
                ],
                "*1\r\n*3\r\n$12\r\nroom=kitchen\r\n*0\r\n*1\r\n*2\r\n:300\r\n+1\r\n",
            ),
            // Nothing wears the label, so nothing is in any group.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"nope",
                    b"REDUCE",
                    b"sum",
                ],
                "*0\r\n",
            ),
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"sum,avg",
                    b"100",
                    b"FILTER",
                    b"room=bedroom",
                ],
                "*1\r\n*3\r\n$1\r\nb\r\n*0\r\n*1\r\n*3\r\n:200\r\n+2\r\n+2\r\n",
            ),
            // The errors, in the order they are looked for.
            (
                &[b"TS.MRANGE", b"-", b"+", b"room=kitchen"],
                "-ERR TSDB: missing FILTER argument\r\n",
            ),
            (
                &[b"TS.MRANGE", b"-", b"+", b"FILTER"],
                "-ERR TSDB: missing labels for filter argument\r\n",
            ),
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"sum",
                    b"FILTER",
                    b"room=kitchen",
                ],
                "-ERR TSDB: GROUPBY should always come after filter\r\n",
            ),
            // The group is four words from the end here, so the length is what
            // is wrong with it.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"sum",
                    b"x",
                ],
                "-ERR wrong number of arguments for 'ts.mrange' command\r\n",
            ),
            // And here it is not, so its words are filters and answer first.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"nope",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"sum",
                    b"x",
                ],
                "-ERR TSDB: failed parsing labels\r\n",
            ),
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"twa",
                ],
                "-ERR TSDB: Invalid reducer type\r\n",
            ),
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"sum,avg",
                    b"100",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"sum",
                ],
                "-ERR TSDB: GROUPBY is not allowed when multiple aggregators are specified\r\n",
            ),
            // The label list ends at a keyword, so this is a `COUNT` with a
            // `FILTER` where its number should be.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"SELECTED_LABELS",
                    b"COUNT",
                    b"FILTER",
                    b"room=kitchen",
                ],
                "-ERR TSDB: Couldn't parse COUNT\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{argv:?}");
        }
    }

    /// The multi key reads on RESP3, where the key becomes a map key and the
    /// reducer and the member keys become fields of their own.
    #[test]
    fn resp3_writes_a_multi_key_read_as_a_map_of_four() {
        let mut f = spanned();
        f.out = Out::new(Proto::Resp3);
        let cases: &[(&[&[u8]], &str)] = &[
            (
                &[b"TS.MRANGE", b"-", b"+", b"FILTER", b"room=bedroom"],
                "%1\r\n$1\r\nb\r\n*3\r\n%0\r\n%1\r\n$11\r\naggregators\r\n*0\r\n\
                 *1\r\n*2\r\n:200\r\n,2\r\n",
            ),
            // The reductions a read asked for, which RESP2 has no room for at
            // all and which is empty on a read that asked for none.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"AGGREGATION",
                    b"sum,avg",
                    b"100",
                    b"FILTER",
                    b"room=bedroom",
                ],
                "%1\r\n$1\r\nb\r\n*3\r\n%0\r\n%1\r\n$11\r\naggregators\r\n*2\r\n$3\r\nsum\r\n\
                 $3\r\navg\r\n*1\r\n*3\r\n:200\r\n,2\r\n,2\r\n",
            ),
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"sum",
                ],
                "%1\r\n$12\r\nroom=kitchen\r\n*4\r\n%0\r\n%1\r\n$8\r\nreducers\r\n*1\r\n\
                 $3\r\nsum\r\n%1\r\n$7\r\nsources\r\n*2\r\n$1\r\na\r\n$1\r\nc\r\n\
                 *3\r\n*2\r\n:100\r\n,11.5\r\n*2\r\n:200\r\n,2.5\r\n*2\r\n:300\r\n,30\r\n",
            ),
            // The labels hold only the pair the group was made on, because the
            // reducer and the sources have somewhere else to go.
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"WITHLABELS",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"room",
                    b"REDUCE",
                    b"max",
                ],
                "%1\r\n$12\r\nroom=kitchen\r\n*4\r\n%1\r\n$4\r\nroom\r\n$7\r\nkitchen\r\n\
                 %1\r\n$8\r\nreducers\r\n*1\r\n$3\r\nmax\r\n\
                 %1\r\n$7\r\nsources\r\n*2\r\n$1\r\na\r\n$1\r\nc\r\n\
                 *3\r\n*2\r\n:100\r\n,10\r\n*2\r\n:200\r\n,2.5\r\n*2\r\n:300\r\n,30\r\n",
            ),
            (
                &[
                    b"TS.MRANGE",
                    b"-",
                    b"+",
                    b"FILTER",
                    b"room=kitchen",
                    b"GROUPBY",
                    b"nope",
                    b"REDUCE",
                    b"sum",
                ],
                "%0\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{argv:?}");
        }
    }

    /// `TS.CREATERULE`, whose refusals come in an order of their own.
    #[test]
    fn createrule_checks_the_two_keys_last_and_the_two_links_after_that() {
        let mut f = Fixture::new();
        f.run(&[b"TS.CREATE", b"src"]);
        f.run(&[b"TS.CREATE", b"dst"]);
        f.run(&[b"SET", b"plain", b"v"]);
        let cases: &[(&[&[u8]], &str)] = &[
            // The width is read before the reduction, the reduction before the
            // width being above zero, and all three before either key is looked
            // at, so a command that is wrong twice complains about the first.
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"dst",
                    b"AGGREGATION",
                    b"nope",
                    b"x",
                ],
                "-ERR TSDB: Couldn't parse AGGREGATION\r\n",
            ),
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"dst",
                    b"AGGREGATION",
                    b"nope",
                    b"10",
                ],
                "-ERR TSDB: Unknown aggregation type\r\n",
            ),
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"dst",
                    b"AGGREGATION",
                    b"avg",
                    b"0",
                ],
                "-ERR TSDB: bucketDuration must be greater than zero\r\n",
            ),
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"dst",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                    b"x",
                ],
                "-ERR TSDB: Couldn't parse alignTimestamp\r\n",
            ),
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"src",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                ],
                "-ERR TSDB: the source key and destination key should be different\r\n",
            ),
            // A key holding something else answers the same as a key that is not
            // there at all, because the source is looked up first and neither of
            // them is a series.
            (
                &[
                    b"TS.CREATERULE",
                    b"nope",
                    b"plain",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                ],
                "-ERR TSDB: the key does not exist\r\n",
            ),
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"nope",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                ],
                "-ERR TSDB: the key does not exist\r\n",
            ),
            // A keyword other than AGGREGATION is an arity error rather than a
            // syntax one, because the arity is all that is checked.
            (
                &[b"TS.CREATERULE", b"src", b"dst", b"NOPE", b"avg", b"10"],
                "-ERR wrong number of arguments for 'ts.createrule' command\r\n",
            ),
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"dst",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                ],
                "+OK\r\n",
            ),
            // The link is now in place, so the same rule again is refused from
            // the destination's end.
            (
                &[
                    b"TS.CREATERULE",
                    b"src",
                    b"dst",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                ],
                "-ERR TSDB: the destination key already has a src rule\r\n",
            ),
            // A source that is already someone's destination, and a destination
            // that is already someone's source, are two different sentences.
            (
                &[
                    b"TS.CREATERULE",
                    b"dst",
                    b"src",
                    b"AGGREGATION",
                    b"avg",
                    b"10",
                ],
                "-ERR TSDB: the source key already has a source rule\r\n",
            ),
            (&[b"TS.DELETERULE", b"src", b"dst"], "+OK\r\n"),
            (
                &[b"TS.DELETERULE", b"src", b"dst"],
                "-ERR TSDB: compaction rule does not exist\r\n",
            ),
            // The source is looked up and the destination is not, so a missing
            // destination is a missing rule and a missing source is a missing
            // key, which is the other way round from `TS.CREATERULE`.
            (
                &[b"TS.DELETERULE", b"src", b"nope"],
                "-ERR TSDB: compaction rule does not exist\r\n",
            ),
            (
                &[b"TS.DELETERULE", b"nope", b"dst"],
                "-ERR TSDB: the key does not exist\r\n",
            ),
        ];
        for (argv, want) in cases {
            let got = f.run(argv);
            assert_eq!(&got, want, "{argv:?}");
        }
    }

    /// What a rule writes, which is every bucket but the one it is filling.
    #[test]
    fn a_rule_writes_a_bucket_when_a_later_reading_closes_it() {
        let mut f = Fixture::new();
        f.run(&[b"TS.CREATE", b"src"]);
        f.run(&[b"TS.CREATE", b"dst"]);
        // The readings written before the rule was made are not folded, so the
        // destination is still empty after the first two.
        f.run(&[b"TS.ADD", b"src", b"10", b"1"]);
        f.run(&[
            b"TS.CREATERULE",
            b"src",
            b"dst",
            b"AGGREGATION",
            b"sum",
            b"100",
        ]);
        f.run(&[b"TS.ADD", b"src", b"20", b"2"]);
        assert_eq!(f.run(&[b"TS.RANGE", b"dst", b"-", b"+"]), "*0\r\n");
        // The bucket the rule is filling holds only what it was given, so it is
        // 2 rather than 3, and it is written when a reading lands past it.
        assert_eq!(f.run(&[b"TS.GET", b"dst", b"LATEST"]), "*2\r\n:0\r\n+2\r\n");
        f.run(&[b"TS.ADD", b"src", b"110", b"4"]);
        assert_eq!(
            f.run(&[b"TS.RANGE", b"dst", b"-", b"+"]),
            "*1\r\n*2\r\n:0\r\n+2\r\n"
        );
        // A reading into a bucket that has already been written works that
        // bucket out again over everything the source now holds.
        f.run(&[b"TS.ADD", b"src", b"30", b"8"]);
        assert_eq!(
            f.run(&[b"TS.RANGE", b"dst", b"-", b"+"]),
            "*1\r\n*2\r\n:0\r\n+11\r\n"
        );
        // Deleting from the source works the buckets it touched out again and
        // reopens the newest one, so `LATEST` starts from the whole bucket.
        assert_eq!(f.run(&[b"TS.DEL", b"src", b"0", b"25"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"TS.RANGE", b"dst", b"-", b"+"]),
            "*1\r\n*2\r\n:0\r\n+8\r\n"
        );
        assert_eq!(
            f.run(&[b"TS.GET", b"dst", b"LATEST"]),
            "*2\r\n:100\r\n+4\r\n"
        );
        // The link shows on both ends, and dropping either key takes it down.
        assert!(f.run(&[b"TS.INFO", b"dst"]).contains("sourceKey"));
        f.run(&[b"DEL", b"dst"]);
        assert_eq!(
            f.run(&[b"TS.DELETERULE", b"src", b"dst"]),
            "-ERR TSDB: compaction rule does not exist\r\n"
        );
    }

    /// The three shapes an `XADD` id can take, and the one rule behind all of
    /// them.
    #[test]
    fn xadd_ids_only_ever_go_up() {
        let mut f = Fixture::new();
        // A bare millisecond is that millisecond and sequence zero.
        assert_eq!(f.run(&[b"XADD", b"s", b"5", b"a", b"1"]), "$3\r\n5-0\r\n");
        // And `5-*` is the next free sequence inside it.
        assert_eq!(f.run(&[b"XADD", b"s", b"5-*", b"a", b"2"]), "$3\r\n5-1\r\n");
        assert_eq!(f.run(&[b"XADD", b"s", b"5-*", b"a", b"3"]), "$3\r\n5-2\r\n");
        assert_eq!(f.run(&[b"XADD", b"s", b"6-9", b"a", b"4"]), "$3\r\n6-9\r\n");
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":4\r\n");

        assert!(
            f.run(&[b"XADD", b"s", b"6-9", b"a", b"5"])
                .contains("equal or smaller")
        );
        assert!(
            f.run(&[b"XADD", b"s", b"0-0", b"a", b"5"])
                .contains("must be greater than 0-0")
        );
        assert!(
            f.run(&[b"XADD", b"s", b"nonsense", b"a", b"5"])
                .contains("Invalid stream ID")
        );
        // The pairs have to be pairs, and Redis calls an odd one an arity error
        // rather than a syntax error even though the table has already passed.
        assert!(
            f.run(&[b"XADD", b"s", b"*", b"a"])
                .contains("wrong number of arguments")
        );

        // `NOMKSTREAM` on a key that is not there is a null and not a zero, so a
        // producer can tell nobody is consuming this yet from the write landed.
        assert_eq!(
            f.run(&[b"XADD", b"gone", b"NOMKSTREAM", b"*", b"a", b"1"]),
            "$-1\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"TYPE", b"s"]), "+stream\r\n");
        assert_eq!(f.run(&[b"OBJECT", b"ENCODING", b"s"]), "$6\r\nstream\r\n");
    }

    /// The trim options, which are three keywords that disagree about how many
    /// arguments they take.
    #[test]
    fn trimming_reads_its_options_the_way_redis_does() {
        let mut f = Fixture::new();
        for i in 1..=10u32 {
            f.run(&[b"XADD", b"s", format!("{i}-1").as_bytes(), b"a", b"1"]);
        }
        assert_eq!(f.run(&[b"XTRIM", b"s", b"MAXLEN", b"4"]), ":6\r\n");
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":4\r\n");
        assert_eq!(f.run(&[b"XTRIM", b"s", b"MINID", b"9"]), ":2\r\n");
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":2\r\n");

        // One argument after the keyword and the `~` is read as the threshold,
        // which is what a real server does and is the reason this is a number
        // complaint and not a syntax one.
        assert!(
            f.run(&[b"XTRIM", b"s", b"MAXLEN", b"~"])
                .contains("not an integer")
        );
        assert!(
            f.run(&[b"XTRIM", b"s", b"MAXLEN", b"-1"])
                .contains("MAXLEN argument must be >= 0")
        );
        // The strategy check runs before the approximation check, so a LIMIT
        // with neither is told about the missing strategy.
        assert!(
            f.run(&[b"XTRIM", b"s", b"LIMIT", b"5"])
                .contains("without specifying a trimming strategy")
        );
        assert!(
            f.run(&[b"XTRIM", b"s", b"MAXLEN", b"5", b"LIMIT", b"5"])
                .contains("without the special ~ option")
        );
        assert!(
            f.run(&[b"XTRIM", b"s", b"MAXLEN", b"5", b"MINID", b"5"])
                .contains("at the same time are not compatible")
        );
        // NOMKSTREAM is XADD's and XTRIM does not take it.
        assert!(
            f.run(&[b"XTRIM", b"s", b"NOMKSTREAM", b"MAXLEN", b"5"])
                .contains("syntax error")
        );
        assert_eq!(f.run(&[b"XTRIM", b"missing", b"MAXLEN", b"5"]), ":0\r\n");
    }

    /// `XRANGE`, whose two kinds of nothing are the thing worth pinning.
    #[test]
    fn xrange_looks_the_key_up_before_it_reads_the_count() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"5-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"6-1", b"b", b"2"]);

        assert_eq!(
            f.run(&[b"XRANGE", b"s", b"-", b"+"]),
            "*2\r\n*2\r\n$3\r\n5-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n\
             *2\r\n$3\r\n6-1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        assert_eq!(
            f.run(&[b"XREVRANGE", b"s", b"+", b"-", b"COUNT", b"1"]),
            "*1\r\n*2\r\n$3\r\n6-1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        // The exclusive bound is stepped after the missing sequence is filled
        // in, so `(6` is `6-` and the largest sequence there is, minus one, and
        // `6-1` is still in the range.
        assert_eq!(
            f.run(&[b"XRANGE", b"s", b"-", b"(6"]),
            "*2\r\n*2\r\n$3\r\n5-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n\
             *2\r\n$3\r\n6-1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        assert_eq!(
            f.run(&[b"XRANGE", b"s", b"(5-1", b"+"]),
            "*1\r\n*2\r\n$3\r\n6-1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        assert!(
            f.run(&[b"XRANGE", b"s", b"(-", b"+"])
                .contains("Invalid stream ID")
        );

        // The two kinds of nothing. A key that is not there is an empty array
        // and a key that is there with a count of zero is a null array, because
        // the lookup happens first.
        assert_eq!(
            f.run(&[b"XRANGE", b"missing", b"-", b"+", b"COUNT", b"0"]),
            "*0\r\n"
        );
        assert_eq!(
            f.run(&[b"XRANGE", b"s", b"-", b"+", b"COUNT", b"0"]),
            "*-1\r\n"
        );
        f.run(&[b"SET", b"str", b"v"]);
        assert!(
            f.run(&[b"XRANGE", b"str", b"-", b"+", b"COUNT", b"0"])
                .starts_with("-WRONGTYPE")
        );
        // The count is read in a loop, so the last one wins.
        assert_eq!(
            f.run(&[b"XRANGE", b"s", b"-", b"+", b"COUNT", b"2", b"COUNT", b"1"]),
            "*1\r\n*2\r\n$3\r\n5-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
    }

    /// `XDEL` and `XACK` check every id before they touch any of them.
    #[test]
    fn a_bad_id_late_in_the_list_stops_the_whole_command() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        assert!(
            f.run(&[b"XDEL", b"s", b"1-1", b"nonsense"])
                .contains("Invalid stream ID")
        );
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":2\r\n");
        assert_eq!(f.run(&[b"XDEL", b"s", b"1-1", b"9-9"]), ":1\r\n");
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":1\r\n");
        assert_eq!(f.run(&[b"XDEL", b"missing", b"1-1"]), ":0\r\n");
        assert_eq!(f.run(&[b"XACK", b"missing", b"g", b"1-1"]), ":0\r\n");
    }

    /// `XGROUP`, and the two different complaints it makes about arguments.
    #[test]
    fn xgroup_has_an_arity_per_subcommand() {
        let mut f = Fixture::new();
        assert!(
            f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"$"])
                .contains("requires the key")
        );
        assert_eq!(
            f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"$", b"MKSTREAM"]),
            "+OK\r\n"
        );
        // A second CREATE is BUSYGROUP and not an ordinary error, because a
        // client racing another one to make a group branches on the prefix.
        assert!(
            f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"$"])
                .starts_with("-BUSYGROUP")
        );
        assert_eq!(
            f.run(&[b"XGROUP", b"CREATECONSUMER", b"s", b"g", b"c"]),
            ":1\r\n"
        );
        assert_eq!(
            f.run(&[b"XGROUP", b"CREATECONSUMER", b"s", b"g", b"c"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[b"XGROUP", b"DELCONSUMER", b"s", b"g", b"c"]),
            ":0\r\n"
        );

        // Below the subcommand's own arity is an arity error naming the pair.
        let short = f.run(&[b"XGROUP", b"DESTROY", b"s"]);
        assert!(
            short.contains("wrong number of arguments for 'xgroup|destroy' command"),
            "{short}"
        );
        // At or above it in a shape the handler will not take is the other one.
        let odd = f.run(&[b"XGROUP", b"SETID", b"s", b"g", b"0", b"ENTRIESREAD"]);
        assert!(
            odd.contains("unknown subcommand or wrong number of arguments for 'SETID'"),
            "{odd}"
        );
        assert!(
            f.run(&[b"XGROUP", b"NOSUCH", b"s"])
                .contains("Try XGROUP HELP")
        );

        assert_eq!(f.run(&[b"XGROUP", b"SETID", b"s", b"g", b"0"]), "+OK\r\n");
        assert!(
            f.run(&[b"XGROUP", b"SETID", b"s", b"nogroup", b"0"])
                .starts_with("-NOGROUP")
        );
        assert_eq!(f.run(&[b"XGROUP", b"DESTROY", b"s", b"g"]), ":1\r\n");
        assert_eq!(f.run(&[b"XGROUP", b"DESTROY", b"s", b"g"]), ":0\r\n");
        assert!(
            f.run(&[b"XGROUP", b"DESTROY", b"missing", b"g"])
                .contains("requires the key")
        );
    }

    /// A group read, an acknowledgement, and what is left in between.
    #[test]
    fn xreadgroup_hands_out_and_xack_takes_back() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);

        let first = f.run(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c1",
            b"COUNT",
            b"1",
            b"STREAMS",
            b"s",
            b">",
        ]);
        assert_eq!(
            first,
            "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        // A history read names its stream even with nothing to show, which is
        // the difference between it and a `>` read that found nothing.
        assert_eq!(
            f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c2", b"STREAMS", b"s", b"0"]),
            "*1\r\n*2\r\n$1\r\ns\r\n*0\r\n"
        );
        assert_eq!(
            f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b"0"]),
            "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );

        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g"]),
            "*4\r\n:1\r\n$3\r\n1-1\r\n$3\r\n1-1\r\n*1\r\n*2\r\n$2\r\nc1\r\n$1\r\n1\r\n"
        );
        assert_eq!(f.run(&[b"XACK", b"s", b"g", b"1-1"]), ":1\r\n");
        assert_eq!(f.run(&[b"XACK", b"s", b"g", b"1-1"]), ":0\r\n");
        // Empty is four nulls and not a zero with three empty things.
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g"]),
            "*4\r\n:0\r\n$-1\r\n$-1\r\n*-1\r\n"
        );

        // A history read of an entry that has since been deleted is the id with
        // a null beside it, so the consumer can still acknowledge it.
        f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b">"]);
        f.run(&[b"XDEL", b"s", b"2-1"]);
        assert_eq!(
            f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b"0"]),
            "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n2-1\r\n$-1\r\n"
        );

        // The group lookup runs before the id parse, so a `+` at a stream with
        // no such group is told about the group and not about the id.
        assert!(
            f.run(&[
                b"XREADGROUP",
                b"GROUP",
                b"nope",
                b"c",
                b"STREAMS",
                b"s",
                b"+"
            ])
            .starts_with("-NOGROUP")
        );
        assert!(
            f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c", b"STREAMS", b"s", b"$"])
                .contains("meaningless in the context of XREADGROUP")
        );
        assert!(
            f.run(&[b"XREAD", b"GROUP", b"g", b"c", b"STREAMS", b"s", b"0"])
                .contains("only supported by XREADGROUP")
        );
        assert!(
            f.run(&[
                b"XREADGROUP",
                b"GROUP",
                b"g",
                b"c",
                b"STREAMS",
                b"s",
                b"a",
                b"b"
            ])
            .contains("Unbalanced 'xreadgroup' list of streams")
        );
    }

    /// `XREAD` without `BLOCK`, which answers now and takes nothing for an
    /// answer.
    #[test]
    fn xread_with_no_block_writes_the_null_itself() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        assert_eq!(
            f.run(&[b"XREAD", b"STREAMS", b"s", b"0"]),
            "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        // Nothing new is a null array and not an empty one, and a stream with
        // nothing new is left out rather than sent with an empty list.
        assert_eq!(f.run(&[b"XREAD", b"STREAMS", b"s", b"1-1"]), "*-1\r\n");
        assert_eq!(f.run(&[b"XREAD", b"STREAMS", b"missing", b"0"]), "*-1\r\n");
        f.run(&[b"XADD", b"other", b"1-1", b"b", b"2"]);
        assert_eq!(
            f.run(&[b"XREAD", b"STREAMS", b"s", b"other", b"1-1", b"0"]),
            "*1\r\n*2\r\n$5\r\nother\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        // `$` is the last id, so nothing that is already there comes back.
        assert_eq!(f.run(&[b"XREAD", b"STREAMS", b"s", b"$"]), "*-1\r\n");
        // And `+` is the last entry, whatever COUNT says.
        assert_eq!(
            f.run(&[b"XREAD", b"COUNT", b"5", b"STREAMS", b"s", b"+"]),
            "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        // A count of zero means unlimited here, which is the opposite of what it
        // means to XRANGE.
        assert_eq!(
            f.run(&[b"XREAD", b"COUNT", b"0", b"STREAMS", b"s", b"0"]),
            "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        // Milliseconds as a whole number, where BLPOP takes seconds as a float.
        assert!(
            f.run(&[b"XREAD", b"BLOCK", b"0.5", b"STREAMS", b"s", b"$"])
                .contains("not an integer")
        );
        assert!(
            f.run(&[b"XREAD", b"BLOCK", b"-1", b"STREAMS", b"s", b"$"])
                .contains("timeout is negative")
        );
        assert!(
            f.run(&[b"XREAD", b"STREAMS", b"s", b"other", b"0"])
                .contains("Unbalanced 'xread' list of streams")
        );
    }

    /// A blocked reader, and the two ways it stops being blocked.
    #[test]
    fn a_blocked_xread_wakes_on_the_next_entry() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        let (flow, reply) = f.flow(&[b"XREAD", b"BLOCK", b"0", b"STREAMS", b"s", b"$"]);
        assert_eq!(flow, Flow::Block);
        assert!(reply.is_empty());

        // Everybody parked on the stream gets the entry, because a read takes
        // nothing away. That is the difference between this and BLPOP.
        let (flow, _) = f.flow(&[b"XREAD", b"BLOCK", b"0", b"STREAMS", b"s", b"$"]);
        assert_eq!(flow, Flow::Block);
        assert_eq!(f.server.parked(), 2);

        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        let want = "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n2-1\r\n*2\r\n$1\r\na\r\n$1\r\n2\r\n";
        for at in 0..2 {
            let mut out = Out::new(Proto::Resp2);
            assert!(f.server.serve_waiter(at, 0, &mut out));
            assert_eq!(core::str::from_utf8(out.as_slice()).expect("ascii"), want);
        }

        // And a deadline that runs out is a null array, the same as a plain
        // XREAD that found nothing.
        f.server.forget_waiters(7);
        let (flow, _) = f.flow(&[b"XREAD", b"BLOCK", b"50", b"STREAMS", b"s", b"$"]);
        assert_eq!(flow, Flow::Block);
        let mut out = Out::new(Proto::Resp2);
        assert!(!f.server.serve_waiter(0, 0, &mut out));
        assert!(out.as_slice().is_empty());
        assert!(f.server.serve_waiter(0, u64::MAX, &mut out));
        assert_eq!(
            core::str::from_utf8(out.as_slice()).expect("ascii"),
            "*-1\r\n"
        );
    }

    /// A blocked group reader whose group is destroyed under it.
    #[test]
    fn losing_a_group_while_blocked_is_the_ordinary_sentence() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"$"]);
        let (flow, _) = f.flow(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c",
            b"BLOCK",
            b"0",
            b"STREAMS",
            b"s",
            b">",
        ]);
        assert_eq!(flow, Flow::Block);

        f.run(&[b"XGROUP", b"DESTROY", b"s", b"g"]);
        let mut out = Out::new(Proto::Resp2);
        assert!(f.server.serve_waiter(0, 0, &mut out));
        // The ordinary sentence and not a special one about having been parked,
        // which is what a running 8.10 sends.
        assert_eq!(
            core::str::from_utf8(out.as_slice()).expect("ascii"),
            "-NOGROUP No such key 's' or consumer group 'g' in XREADGROUP with GROUP option\r\n"
        );
    }

    /// `XCLAIM`, whose argument shape is the odd one in the group.
    #[test]
    fn xclaim_reads_ids_until_one_will_not_parse() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b">"]);

        // Everything after the first argument that is not an id is an option, so
        // a `-` is an unrecognised option and not a bad id.
        assert!(
            f.run(&[b"XCLAIM", b"s", b"g", b"c2", b"0", b"-"])
                .contains("Unrecognized XCLAIM option '-'")
        );
        assert_eq!(
            f.run(&[b"XCLAIM", b"s", b"g", b"c2", b"0", b"1-1", b"JUSTID"]),
            "*1\r\n$3\r\n1-1\r\n"
        );
        // An id that is pending but whose entry has gone is an empty answer, and
        // it leaves the pending list on the way past.
        f.run(&[b"XDEL", b"s", b"2-1"]);
        assert_eq!(
            f.run(&[b"XCLAIM", b"s", b"g", b"c2", b"0", b"2-1"]),
            "*0\r\n"
        );
        assert!(
            f.run(&[b"XPENDING", b"s", b"g"])
                .starts_with("*4\r\n:1\r\n")
        );
        assert!(
            f.run(&[b"XCLAIM", b"s", b"nope", b"c", b"0", b"1-1"])
                .starts_with("-NOGROUP")
        );
        assert!(
            f.run(&[b"XCLAIM", b"s", b"g", b"c", b"nan", b"1-1"])
                .contains("Invalid min-idle-time argument for XCLAIM")
        );
    }

    /// `XAUTOCLAIM`, and the third value nobody expects.
    #[test]
    fn xautoclaim_reports_what_it_dropped() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b">"]);
        f.run(&[b"XDEL", b"s", b"1-1"]);

        // The cursor, what was claimed, and what was dropped for no longer being
        // in the stream. The third one is what makes a sweep converge.
        assert_eq!(
            f.run(&[b"XAUTOCLAIM", b"s", b"g", b"c2", b"0", b"-", b"JUSTID"]),
            "*3\r\n$3\r\n0-0\r\n*1\r\n$3\r\n2-1\r\n*1\r\n$3\r\n1-1\r\n"
        );
        assert!(
            f.run(&[b"XAUTOCLAIM", b"s", b"g", b"c2", b"0", b"-", b"COUNT", b"0"])
                .contains("COUNT must be > 0")
        );
        assert!(
            f.run(&[b"XAUTOCLAIM", b"s", b"nope", b"c", b"0", b"-"])
                .starts_with("-NOGROUP")
        );
    }

    /// `XDELEX`, which is `XDEL` with a say in what the groups keep.
    #[test]
    fn xdelex_answers_one_integer_an_id() {
        let mut f = Fixture::new();
        for i in 1..=4 {
            f.run(&[b"XADD", b"s", format!("{i}-1").as_bytes(), b"a", b"1"]);
        }
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c",
            b"COUNT",
            b"2",
            b"STREAMS",
            b"s",
            b">",
        ]);

        // One means gone and minus one means it was not there to start with.
        assert_eq!(
            f.run(&[b"XDELEX", b"s", b"IDS", b"2", b"1-1", b"9-9"]),
            "*2\r\n:1\r\n:-1\r\n"
        );
        // `KEEPREF` leaves the pending entry behind, so the group still counts
        // the one it was handed even though the entry has gone.
        assert!(
            f.run(&[b"XPENDING", b"s", b"g"])
                .starts_with("*4\r\n:2\r\n")
        );
        // `DELREF` takes it out of every pending list on the way past.
        assert_eq!(
            f.run(&[b"XDELEX", b"s", b"DELREF", b"IDS", b"1", b"2-1"]),
            "*1\r\n:1\r\n"
        );
        // `1-1` is still in the list, because the delete before it said KEEPREF.
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g"]),
            "*4\r\n:1\r\n$3\r\n1-1\r\n$3\r\n1-1\r\n*1\r\n*2\r\n$1\r\nc\r\n$1\r\n1\r\n"
        );

        // Two means somebody still wants it, and the question is wider than the
        // name: the group's bookmark is at `2-1`, so `4-1` is above it and is
        // refused even though no consumer has ever been handed it.
        assert_eq!(
            f.run(&[b"XDELEX", b"s", b"ACKED", b"IDS", b"2", b"3-1", b"4-1"]),
            "*2\r\n:2\r\n:2\r\n"
        );

        // A key that is not there answers minus ones without reading the IDs.
        assert_eq!(
            f.run(&[b"XDELEX", b"nope", b"IDS", b"2", b"bad", b"worse"]),
            "*2\r\n:-1\r\n:-1\r\n"
        );
        // A key that is there validates every ID before deleting any of them.
        assert!(
            f.run(&[b"XDELEX", b"s", b"IDS", b"2", b"3-1", b"bad"])
                .starts_with("-ERR Invalid stream ID")
        );
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":2\r\n");

        assert!(
            f.run(&[b"XDELEX", b"s", b"IDS", b"0", b"1-1"])
                .contains("Number of IDs must be a positive integer")
        );
        assert!(
            f.run(&[b"XDELEX", b"s", b"IDS", b"2", b"1-1"])
                .contains("The `numids` parameter must match the number of arguments")
        );
        // The condition is one word, so a second one is a syntax error, and so
        // is one ID more than the count promised.
        assert!(
            f.run(&[b"XDELEX", b"s", b"KEEPREF", b"DELREF", b"IDS", b"1", b"1-1"])
                .starts_with("-ERR syntax error")
        );
        assert!(
            f.run(&[b"XDELEX", b"s", b"IDS", b"1", b"1-1", b"2-1"])
                .starts_with("-ERR syntax error")
        );
        // The key is looked up first, so the wrong type beats the syntax.
        f.run(&[b"SET", b"str", b"v"]);
        assert!(
            f.run(&[b"XDELEX", b"str", b"BOGUS", b"IDS", b"0", b"1-1"])
                .starts_with("-WRONGTYPE")
        );
    }

    /// `XACKDEL`, whose reply is about the pending list and not about the log.
    #[test]
    fn xackdel_reports_what_the_group_was_holding() {
        let mut f = Fixture::new();
        for i in 1..=3 {
            f.run(&[b"XADD", b"s", format!("{i}-1").as_bytes(), b"a", b"1"]);
        }
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c",
            b"COUNT",
            b"1",
            b"STREAMS",
            b"s",
            b">",
        ]);

        // Minus one is not about the stream: `2-1` is sitting there unread and
        // still answers minus one, because the group was not holding it. It also
        // stays, since only an ID that was acknowledged is deleted.
        assert_eq!(
            f.run(&[b"XACKDEL", b"s", b"g", b"IDS", b"2", b"1-1", b"2-1"]),
            "*2\r\n:1\r\n:-1\r\n"
        );
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":2\r\n");

        // A missing group is minus one an ID and not a NOGROUP.
        assert_eq!(
            f.run(&[b"XACKDEL", b"s", b"nope", b"IDS", b"1", b"2-1"]),
            "*1\r\n:-1\r\n"
        );
        assert_eq!(
            f.run(&[b"XACKDEL", b"nope", b"g", b"IDS", b"1", b"2-1"]),
            "*1\r\n:-1\r\n"
        );

        // The acknowledgement happens whatever the condition says, so an ACKED
        // that answers two has still emptied the pending list.
        f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c", b"STREAMS", b"s", b">"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g2", b"0"]);
        assert_eq!(
            f.run(&[b"XACKDEL", b"s", b"g", b"ACKED", b"IDS", b"1", b"2-1"]),
            "*1\r\n:2\r\n"
        );
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g"]),
            "*4\r\n:1\r\n$3\r\n3-1\r\n$3\r\n3-1\r\n*1\r\n*2\r\n$1\r\nc\r\n$1\r\n1\r\n"
        );
        assert_eq!(f.run(&[b"XLEN", b"s"]), ":2\r\n");
    }

    /// `XNACK`, which hands an entry back to nobody.
    #[test]
    fn xnack_releases_an_entry_for_the_next_claim() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b">"]);
        // Twice, so the delivery count is two and the words have something to
        // do with it.
        f.run(&[b"XCLAIM", b"s", b"g", b"c1", b"0", b"1-1", b"2-1"]);

        assert_eq!(
            f.run(&[b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"1", b"1-1"]),
            ":1\r\n"
        );
        // No owner, no idle time, and the count left where it was. A released
        // entry reads as idle for longer than any min-idle-time, which is what
        // puts it at the front of the next claim.
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"]),
            "*2\r\n*4\r\n$3\r\n1-1\r\n$0\r\n\r\n:-1\r\n:2\r\n*4\r\n$3\r\n2-1\r\n$2\r\nc1\r\n:0\r\n:2\r\n"
        );
        // The consumer no longer holds it, so a filtered XPENDING skips it.
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10", b"c1"]),
            "*1\r\n*4\r\n$3\r\n2-1\r\n$2\r\nc1\r\n:0\r\n:2\r\n"
        );
        // The bookmark did not move, so a `>` read will not hand it out again.
        assert_eq!(
            f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c2", b"STREAMS", b"s", b">"]),
            "*-1\r\n"
        );
        // A claim at any min-idle-time takes it.
        assert_eq!(
            f.run(&[
                b"XAUTOCLAIM",
                b"s",
                b"g",
                b"c2",
                b"99999999",
                b"-",
                b"JUSTID"
            ]),
            "*3\r\n$3\r\n0-0\r\n*1\r\n$3\r\n1-1\r\n*0\r\n"
        );

        // `SILENT` takes one off the count rather than putting it back to zero,
        // which only shows on an entry that has been handed out more than once.
        // It was delivered and then claimed, so it is on two and goes to one.
        f.run(&[b"XNACK", b"s", b"g", b"SILENT", b"IDS", b"1", b"1-1"]);
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"])
                .contains(":-1\r\n:1\r\n")
        );
        // And it stops at zero rather than wrapping.
        f.run(&[b"XNACK", b"s", b"g", b"SILENT", b"IDS", b"1", b"1-1"]);
        f.run(&[b"XNACK", b"s", b"g", b"SILENT", b"IDS", b"1", b"1-1"]);
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"])
                .contains(":-1\r\n:0\r\n")
        );
        // `FATAL` puts it at the ceiling, and `RETRYCOUNT` wins over the word.
        f.run(&[b"XNACK", b"s", b"g", b"FATAL", b"IDS", b"1", b"1-1"]);
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"])
                .contains(":9223372036854775807\r\n")
        );
        f.run(&[
            b"XNACK",
            b"s",
            b"g",
            b"FATAL",
            b"IDS",
            b"1",
            b"1-1",
            b"RETRYCOUNT",
            b"3",
        ]);
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"])
                .contains(":-1\r\n:3\r\n")
        );

        // Releasing something the group is not holding is zero, and `FORCE`
        // makes the pending entry rather than answering zero. A forced entry
        // starts at zero, since there was no earlier count to keep.
        f.run(&[b"XACK", b"s", b"g", b"2-1"]);
        assert_eq!(
            f.run(&[b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"1", b"2-1"]),
            ":0\r\n"
        );
        assert_eq!(
            f.run(&[
                b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"1", b"2-1", b"FORCE"
            ]),
            ":1\r\n"
        );
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"])
                .contains(":-1\r\n:0\r\n")
        );
        // `FORCE` on an ID the stream does not have is still zero.
        assert_eq!(
            f.run(&[
                b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"1", b"9-9", b"FORCE"
            ]),
            ":0\r\n"
        );

        // The group is looked up before the mode word, and it raises rather
        // than answering per ID the way the two delete commands do.
        assert_eq!(
            f.run(&[b"XNACK", b"s", b"nope", b"BOGUS", b"IDS", b"1", b"1-1"]),
            "-NOGROUP No such key 's' or consumer group 'nope'\r\n"
        );
        assert!(
            f.run(&[b"XNACK", b"s", b"g", b"BOGUS", b"IDS", b"1", b"1-1"])
                .starts_with("-ERR")
        );
        // Its own sentences, which are not the ones XDELEX uses.
        assert!(
            f.run(&[b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"0", b"1-1"])
                .contains("numids must be a positive integer")
        );
        assert!(
            f.run(&[b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"2", b"1-1"])
                .contains("number of IDs doesn't match numids")
        );
        // Everything past the counted IDs is an option, so one too many is an
        // option nobody recognises and not a count that does not add up.
        assert!(
            f.run(&[b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"1", b"1-1", b"2-1"])
                .contains("Unrecognized XNACK option '2-1'")
        );
    }

    /// `XINFO`, which is where the shape of the storage shows through.
    #[test]
    fn xinfo_reports_the_stream_the_groups_and_the_consumers() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c1",
            b"COUNT",
            b"1",
            b"STREAMS",
            b"s",
            b">",
        ]);

        let info = f.run(&[b"XINFO", b"STREAM", b"s"]);
        // Ten pairs, since the six idempotency fields have nothing behind them
        // here and a zero would claim they had. That is D-27.
        assert!(info.starts_with("*20\r\n"), "{info}");
        assert!(info.contains("$6\r\nlength\r\n:2\r\n"), "{info}");
        assert!(
            info.contains("$17\r\nlast-generated-id\r\n$3\r\n2-1\r\n"),
            "{info}"
        );
        assert!(info.contains("$13\r\nentries-added\r\n:2\r\n"), "{info}");
        assert!(info.contains("$6\r\ngroups\r\n:1\r\n"), "{info}");

        let groups = f.run(&[b"XINFO", b"GROUPS", b"s"]);
        assert!(groups.starts_with("*1\r\n*12\r\n"), "{groups}");
        assert!(groups.contains("$9\r\nconsumers\r\n:1\r\n"), "{groups}");
        assert!(groups.contains("$7\r\npending\r\n:1\r\n"), "{groups}");
        assert!(groups.contains("$3\r\nlag\r\n:1\r\n"), "{groups}");

        // A consumer that has never been given anything reports minus one for
        // inactive rather than the moment it turned up, which is what tells a
        // worker that is stuck from one that has nothing to do.
        f.run(&[b"XGROUP", b"CREATECONSUMER", b"s", b"g", b"c2"]);
        let consumers = f.run(&[b"XINFO", b"CONSUMERS", b"s", b"g"]);
        assert!(consumers.starts_with("*2\r\n"), "{consumers}");
        assert!(
            consumers.contains("$8\r\ninactive\r\n:-1\r\n"),
            "{consumers}"
        );
        // And in name order, which the storage does not hold them in.
        let c1 = consumers.find("c1").unwrap();
        let c2 = consumers.find("c2").unwrap();
        assert!(c1 < c2, "{consumers}");

        let full = f.run(&[b"XINFO", b"STREAM", b"s", b"FULL"]);
        assert!(full.starts_with("*18\r\n"), "{full}");
        assert!(full.contains("$12\r\nnacked-count\r\n:0\r\n"), "{full}");
        assert!(full.contains("$11\r\nactive-time\r\n"), "{full}");

        assert!(
            f.run(&[b"XINFO", b"STREAM", b"missing"])
                .contains("no such key")
        );
        assert!(
            f.run(&[b"XINFO", b"GROUPS", b"missing"])
                .contains("no such key")
        );
        assert!(
            f.run(&[b"XINFO", b"CONSUMERS", b"s", b"nope"])
                .starts_with("-NOGROUP")
        );
        assert!(
            f.run(&[b"XINFO", b"NOSUCH", b"s"])
                .contains("Try XINFO HELP")
        );
        assert!(f.run(&[b"XINFO", b"HELP"]).contains("XINFO <subcommand>"));
        assert!(f.run(&[b"XGROUP", b"HELP"]).contains("XGROUP <subcommand>"));
    }

    /// `XPENDING`'s long form, which reads its arguments by counting them.
    #[test]
    fn xpending_takes_the_consumer_only_when_the_count_comes_out_right() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        f.run(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]);
        f.run(&[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b">"]);

        let list = f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10"]);
        assert_eq!(list, "*1\r\n*4\r\n$3\r\n1-1\r\n$2\r\nc1\r\n:0\r\n:1\r\n");
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10", b"c1"]),
            "*1\r\n*4\r\n$3\r\n1-1\r\n$2\r\nc1\r\n:0\r\n:1\r\n"
        );
        // A consumer nobody has heard of holds nothing rather than erroring.
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"10", b"nope"]),
            "*0\r\n"
        );
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g", b"IDLE", b"0", b"-", b"+", b"10"]),
            list
        );
        // IDLE is only read at position three.
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"IDLE", b"0"])
                .contains("syntax error")
        );
        assert!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+"])
                .contains("syntax error")
        );
        assert_eq!(
            f.run(&[b"XPENDING", b"s", b"g", b"-", b"+", b"-1"]),
            "*0\r\n"
        );
        assert!(
            f.run(&[b"XPENDING", b"missing", b"g"])
                .starts_with("-NOGROUP")
        );
    }

    /// `XSETID`, which is three counters and two refusals.
    #[test]
    fn xsetid_will_not_go_below_what_is_there() {
        let mut f = Fixture::new();
        f.run(&[b"XADD", b"s", b"5-5", b"a", b"1"]);
        assert_eq!(f.run(&[b"XSETID", b"s", b"9-9"]), "+OK\r\n");
        assert_eq!(
            f.run(&[
                b"XSETID",
                b"s",
                b"10-1",
                b"ENTRIESADDED",
                b"7",
                b"MAXDELETEDID",
                b"9-1"
            ]),
            "+OK\r\n"
        );
        let info = f.run(&[b"XINFO", b"STREAM", b"s"]);
        assert!(info.contains("$13\r\nentries-added\r\n:7\r\n"), "{info}");
        assert!(
            info.contains("$20\r\nmax-deleted-entry-id\r\n$3\r\n9-1\r\n"),
            "{info}"
        );

        assert!(
            f.run(&[b"XSETID", b"s", b"1-1"])
                .contains("smaller than the target stream top item")
        );
        assert!(
            f.run(&[b"XSETID", b"s", b"10-1", b"ENTRIESADDED", b"-1"])
                .contains("entries_added must be positive")
        );
        assert!(
            f.run(&[b"XSETID", b"missing", b"1-1"])
                .contains("no such key")
        );
    }

    /// RESP3, where the two reads answer a map and the entries stay an array.
    #[test]
    fn xread_answers_a_map_on_resp3_and_the_fields_stay_flat() {
        let mut f = Fixture::new();
        f.run(&[b"HELLO", b"3"]);
        f.run(&[b"XADD", b"s", b"1-1", b"a", b"1"]);
        // A map header and then the key and the entries side by side, with no
        // two element array wrapping the pair.
        assert_eq!(
            f.run(&[b"XREAD", b"STREAMS", b"s", b"0"]),
            "%1\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        // The fields are still one flat array and not a map, which is Redis's
        // shape and is what every consumer written before RESP3 expects.
        assert_eq!(
            f.run(&[b"XRANGE", b"s", b"-", b"+"]),
            "*1\r\n*2\r\n$3\r\n1-1\r\n*2\r\n$1\r\na\r\n$1\r\n1\r\n"
        );
        assert_eq!(f.run(&[b"XREAD", b"STREAMS", b"s", b"1-1"]), "_\r\n");
    }

    /// A store to migrate values into, so a test can watch the inversion.
    ///
    /// A vector rather than a file for the same reason the tier's own tests use
    /// one: the file work has not attached a real store yet, and what this is
    /// checking is the policy above the store rather than the store.
    struct Mem {
        blobs: Vec<Vec<u8>>,
    }

    impl yo_kv::cold::Blocks for Mem {
        fn put(&mut self, bytes: &[u8]) -> yo_common::Result<yo_common::Addr> {
            self.blobs.push(bytes.to_vec());
            Ok(yo_common::Addr::new(
                yo_common::Space::Log,
                (self.blobs.len() - 1) as u64,
            ))
        }

        fn get(&self, at: yo_common::Addr) -> yo_common::Result<&[u8]> {
            self.blobs
                .get(at.offset() as usize)
                .map(Vec::as_slice)
                .ok_or_else(|| {
                    yo_common::Error::new(yo_common::Code::Corrupt, "no chunk at that address")
                })
        }

        fn bytes(&self) -> u64 {
            self.blobs.iter().map(|b| b.len() as u64).sum()
        }
    }

    /// A server holding several segments of strings, with somewhere to put them.
    ///
    /// Answers the fixture and what it was holding when it stopped filling.
    fn filled(attach: bool) -> (Fixture, usize) {
        let mut f = Fixture::new();
        if attach {
            f.server
                .striped(0)
                .hold_stripe(0)
                .attach(Box::new(Mem { blobs: Vec::new() }));
        }
        let val = vec![b'v'; 256];
        for i in 0..24000u32 {
            let k = format!("key:{i:08}");
            f.run(&[b"SET", k.as_bytes(), &val]);
        }
        let full = f.server.memory_bytes();
        assert!(full > 3 * 1024 * 1024, "the arena is several segments");
        (f, full)
    }

    /// Write until the server is under `limit` or the writes run out.
    ///
    /// The same shape the eviction test uses. A memory limit is enforced in
    /// front of a command, so nothing happens until something is written, and
    /// the budget means one command does not do the whole job.
    fn press(f: &mut Fixture, limit: usize) {
        let val = vec![b'v'; 256];
        for i in 0..3000u32 {
            let k = format!("new:{i:08}");
            assert_eq!(
                f.run(&[b"SET", k.as_bytes(), &val]),
                "+OK\r\n",
                "write {i} was refused"
            );
            f.server.refresh_memory();
            if f.server.memory_bytes() <= limit {
                return;
            }
        }
        panic!(
            "it never got under: {} against {limit}",
            f.server.memory_bytes()
        );
    }

    #[test]
    fn the_storage_limit_reads_back_and_minus_one_is_no_limit() {
        let mut f = Fixture::new();
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"maxstore"]),
            "*2\r\n$8\r\nmaxstore\r\n$2\r\n-1\r\n",
            "no limit is the default"
        );
        // The same memory value parser `maxmemory` uses, and the same trap in
        // it, plus the one spelling that means no limit at all.
        for (typed, bytes) in [
            (&b"0"[..], "0"),
            (b"1024", "1024"),
            (b"1k", "1000"),
            (b"1gb", "1073741824"),
            (b"-1", "-1"),
        ] {
            assert_eq!(f.run(&[b"CONFIG", b"SET", b"maxstore", typed]), "+OK\r\n");
            assert_eq!(
                f.run(&[b"CONFIG", b"GET", b"maxstore"]),
                format!("*2\r\n$8\r\nmaxstore\r\n${}\r\n{bytes}\r\n", bytes.len()),
                "set {}",
                String::from_utf8_lossy(typed)
            );
        }
        for bad in [&b"1tb"[..], b"-2", b"", b"lots"] {
            assert_eq!(
                f.run(&[b"CONFIG", b"SET", b"maxstore", bad]),
                "-ERR CONFIG SET failed (possibly related to argument 'maxstore') - argument must be a memory value or -1\r\n",
                "refused {}",
                String::from_utf8_lossy(bad)
            );
        }
        // Nothing is attached, so the answer to a memory limit is still Redis's.
        let info = f.run(&[b"INFO", b"memory"]);
        assert!(info.contains("maxstore:-1"), "{info}");
        assert!(info.contains("yo_memory_regime:evict"), "{info}");
        assert!(info.contains("yo_store_bytes:0"), "{info}");
    }

    #[test]
    fn a_memory_limit_moves_values_to_the_file_instead_of_dropping_keys() {
        // The inversion. The same pressure that makes a Redis server throw keys
        // away makes this one move values to the file, and afterwards every key
        // is still there and still answers with what was stored in it.
        let (mut f, full) = filled(true);
        let keys = f.run(&[b"DBSIZE"]);
        assert!(
            f.run(&[b"INFO", b"memory"])
                .contains("yo_memory_regime:migrate"),
            "a database with somewhere to put values migrates"
        );

        let limit = full - 2 * 1024 * 1024;
        f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-lru"]);
        f.run(&[
            b"CONFIG",
            b"SET",
            b"maxmemory",
            limit.to_string().as_bytes(),
        ]);
        press(&mut f, limit);

        assert!(
            f.run(&[b"INFO", b"stats"]).contains("evicted_keys:0"),
            "nothing was thrown away"
        );
        let after: usize = f.run(&[b"DBSIZE"])[1..]
            .trim_end()
            .parse()
            .expect("a count");
        let before: usize = keys[1..].trim_end().parse().expect("a count");
        assert!(after > before, "the keys that came in are all still here");
        assert!(
            f.server.store_bytes() > 0,
            "and what came out of memory went to the file"
        );
        // And the values read back, which is the part that makes it a migration
        // rather than a loss.
        let val = format!("$256\r\n{}\r\n", "v".repeat(256));
        assert_eq!(f.run(&[b"GET", b"key:00000000"]), val);
        assert_eq!(f.run(&[b"GET", b"key:00023999"]), val);
    }

    #[test]
    fn a_storage_limit_of_zero_restores_redis_behaviour_exactly() {
        // The documented setting for a drop in cache. A file that may hold
        // nothing cannot be migrated to, so eviction is all that is left, and
        // the server behaves exactly as it did before any of this existed.
        let (mut f, full) = filled(true);
        f.run(&[b"CONFIG", b"SET", b"maxstore", b"0"]);
        assert!(
            f.run(&[b"INFO", b"memory"])
                .contains("yo_memory_regime:evict"),
            "nothing may go to the file"
        );

        let limit = full - 2 * 1024 * 1024;
        f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-lru"]);
        f.run(&[
            b"CONFIG",
            b"SET",
            b"maxmemory",
            limit.to_string().as_bytes(),
        ]);
        press(&mut f, limit);

        assert!(
            !f.run(&[b"INFO", b"stats"]).contains("evicted_keys:0"),
            "keys were thrown away, which is what was asked for"
        );
        assert_eq!(f.server.store_bytes(), 0, "and the file was never written");
    }

    #[test]
    fn a_full_file_goes_back_to_evicting() {
        // A storage limit reached is a storage limit, and eviction is the right
        // answer to one. The budget here is a few kilobytes, so the first round
        // of migration fills it and everything after that is evicted.
        let (mut f, full) = filled(true);
        f.run(&[b"CONFIG", b"SET", b"maxstore", b"64kb"]);
        let limit = full - 2 * 1024 * 1024;
        f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-lru"]);
        f.run(&[
            b"CONFIG",
            b"SET",
            b"maxmemory",
            limit.to_string().as_bytes(),
        ]);
        press(&mut f, limit);

        assert!(f.server.store_bytes() >= 64 * 1024, "the file filled up");
        assert!(
            !f.run(&[b"INFO", b"stats"]).contains("evicted_keys:0"),
            "and then it started evicting"
        );
        assert!(
            f.run(&[b"INFO", b"memory"])
                .contains("yo_memory_regime:evict"),
            "and it says so"
        );
    }
    // ------------------------------------------------------------- stripes

    /// Every string command, run twice: once on a database that is one keyspace
    /// and once on a database that is eight, with the same commands in the same
    /// order and the replies compared byte for byte.
    ///
    /// This is the whole claim the striping rests on. A key belongs to one
    /// stripe and to no other, so the answer to a command cannot depend on how
    /// many stripes there are, and the way to check that is to ask the same
    /// question of two servers that differ in nothing else.
    ///
    /// The keys are chosen to land on different stripes rather than to look
    /// tidy. `MSET a 1 b 2 c 3` over eight stripes is only a test of anything if
    /// those three keys are not all on the same one, and at eight stripes three
    /// keys land together about one time in fifty.
    #[test]
    fn the_string_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            // The single key commands, which are the ones that get handed one
            // stripe at the dispatch site.
            &[b"SET", b"k1", b"v1"],
            &[b"SET", b"k2", b"v2"],
            &[b"GET", b"k1"],
            &[b"GET", b"nothing"],
            &[b"GETSET", b"k1", b"v1b"],
            &[b"SETNX", b"k1", b"no"],
            &[b"SETNX", b"k3", b"yes"],
            &[b"APPEND", b"k3", b"!"],
            &[b"STRLEN", b"k3"],
            &[b"SETRANGE", b"k3", b"1", b"XY"],
            &[b"GETRANGE", b"k3", b"0", b"-1"],
            &[b"INCR", b"n1"],
            &[b"INCRBY", b"n1", b"41"],
            &[b"DECRBY", b"n1", b"2"],
            &[b"INCRBYFLOAT", b"f1", b"1.5"],
            &[b"SETEX", b"e1", b"100", b"v"],
            &[b"PSETEX", b"e2", b"100000", b"v"],
            &[b"GETEX", b"e1", b"PERSIST"],
            &[b"GETDEL", b"k2"],
            &[b"GET", b"k2"],
            &[b"DIGEST", b"k1"],
            &[b"DELEX", b"k3"],
            // The five that name more than one key, which are the ones that
            // cannot be handed one stripe at all.
            &[b"MSET", b"a", b"1", b"b", b"2", b"c", b"3"],
            &[b"MGET", b"a", b"b", b"c", b"missing"],
            &[b"MSETNX", b"d", b"4", b"e", b"5"],
            &[b"MSETNX", b"e", b"6", b"f", b"7"],
            &[b"MGET", b"d", b"e", b"f"],
            &[b"MSETEX", b"2", b"g", b"7", b"h", b"8", b"NX"],
            &[b"MSETEX", b"2", b"g", b"9", b"h", b"9", b"NX"],
            &[b"MSETEX", b"2", b"g", b"9", b"h", b"9", b"XX"],
            &[b"MGET", b"g", b"h"],
            &[b"SET", b"s1", b"ohmytext"],
            &[b"SET", b"s2", b"mynewtext"],
            &[b"LCS", b"s1", b"s2"],
            &[b"LCS", b"s1", b"s2", b"LEN"],
            &[b"LCS", b"s1", b"s2", b"IDX", b"MINMATCHLEN", b"4"],
            &[b"LCS", b"s1", b"s2", b"IDX", b"WITHMATCHLEN"],
            &[b"LCS", b"s1", b"gone"],
            // And the errors, which have to be the same errors.
            &[b"MSET", b"odd"],
            &[b"LCS", b"s1", b"s2", b"LEN", b"IDX"],
            &[b"MGET"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// The keys of an `MSET` really do end up on different stripes.
    ///
    /// Without this the test above could pass on a server whose stripe number
    /// happened to be a constant, which is a striped database in name only.
    #[test]
    fn a_striped_database_spreads_the_keys_it_is_given() {
        let mut f = Fixture::striped(8);
        for i in 0..256 {
            let key = format!("key:{i}");
            f.run(&[b"SET", key.as_bytes(), b"v"]);
        }
        assert_eq!(f.run(&[b"DBSIZE"]), ":256\r\n");
    }

    /// A wrong type stops an `MGET` no more than it does on one stripe: the key
    /// that is not a string comes back nil and the rest of the reply is intact.
    #[test]
    fn a_wrong_type_in_the_middle_of_an_mget_is_still_one_nil() {
        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for f in [&mut one, &mut many] {
            f.run(&[b"SET", b"str", b"v"]);
            // Planted rather than pushed. `RPUSH` belongs to the list group,
            // which has not been taught about stripes yet and would refuse the
            // wide server. What is under test is what `MGET` does when it walks
            // onto a key that is not a string, and that does not care how the
            // key got there.
            f.server
                .striped(0)
                .hold(b"list")
                .push(b"list", yo_kv::End::Right, core::iter::once(&b"v"[..]))
                .expect("a new list");
        }
        assert_eq!(
            one.run(&[b"MGET", b"str", b"list", b"gone"]),
            many.run(&[b"MGET", b"str", b"list", b"gone"])
        );
    }

    /// The same claim for the keyspace group, and the same way of checking it.
    ///
    /// `SORT` is not in the script because it is the one command in that file
    /// that has not been taught about stripes, and `SCAN`, `KEYS` and
    /// `RANDOMKEY` are not in it either, because those three do not promise an
    /// order and comparing two replies byte for byte would be asserting one.
    /// They get tests of their own below.
    #[test]
    fn the_keyspace_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"SET", b"k1", b"v1"],
            &[b"SET", b"k2", b"v2"],
            &[b"EXISTS", b"k1", b"k2", b"k1", b"gone"],
            &[b"TYPE", b"k1"],
            &[b"TYPE", b"gone"],
            &[b"TOUCH", b"k1", b"k2", b"k1", b"gone"],
            &[b"EXPIRE", b"k1", b"100"],
            &[b"TTL", b"k1"],
            &[b"EXPIRE", b"k1", b"200", b"NX"],
            &[b"PERSIST", b"k1"],
            &[b"TTL", b"k1"],
            &[b"PEXPIREAT", b"k2", b"1900000000000"],
            &[b"EXPIRETIME", b"k2"],
            &[b"PEXPIRETIME", b"k2"],
            &[b"PERSIST", b"k2"],
            &[b"OBJECT", b"ENCODING", b"k1"],
            &[b"OBJECT", b"REFCOUNT", b"k1"],
            &[b"OBJECT", b"IDLETIME", b"k1"],
            &[b"OBJECT", b"FREQ", b"k1"],
            &[b"OBJECT", b"ENCODING", b"gone"],
            &[b"OBJECT", b"HELP"],
            &[b"RENAME", b"k1", b"k9"],
            &[b"GET", b"k9"],
            &[b"RENAME", b"gone", b"x"],
            &[b"RENAMENX", b"k9", b"k2"],
            &[b"RENAMENX", b"k9", b"k8"],
            &[b"GET", b"k8"],
            &[b"COPY", b"k8", b"c1"],
            &[b"COPY", b"k8", b"c1"],
            &[b"COPY", b"k8", b"c1", b"REPLACE"],
            &[b"COPY", b"k8", b"k8"],
            &[b"COPY", b"gone", b"c2"],
            &[b"COPY", b"k8", b"k8", b"DB", b"1"],
            &[b"COPY", b"k8", b"c9", b"DB", b"9"],
            &[b"MOVE", b"c1", b"1"],
            &[b"MOVE", b"c1", b"1"],
            &[b"MOVE", b"k8", b"0"],
            &[b"DEL", b"k2", b"gone"],
            &[b"UNLINK", b"k8", b"k8"],
            &[b"DBSIZE"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }

        // `RESTORE` needs bytes a client would have got from a `DUMP`, so the
        // payload is taken from the store rather than parsed back out of a
        // reply that is not text. Both servers dump the same key and the bytes
        // are the same bytes, which is the first half of what is being checked
        // here.
        for f in [&mut one, &mut many] {
            f.run(&[b"SET", b"d1", b"payload"]);
            let payload = f
                .server
                .striped(0)
                .hold(b"d1")
                .dump(b"d1")
                .expect("a key that is there");
            assert!(
                f.run(&[b"DUMP", b"d1"])
                    .starts_with(&format!("${}", payload.len())),
                "a payload of the length the store gave"
            );
            assert_eq!(f.run(&[b"DUMP", b"gone"]), "$-1\r\n");
            assert_eq!(f.run(&[b"RESTORE", b"d2", b"0", &payload]), "+OK\r\n");
            assert_eq!(f.run(&[b"GET", b"d2"]), "$7\r\npayload\r\n");
            assert_eq!(
                f.run(&[b"RESTORE", b"d2", b"0", &payload]),
                "-BUSYKEY Target key name already exists.\r\n"
            );
            assert_eq!(
                f.run(&[b"RESTORE", b"d3", b"0", b"rubbish"]),
                "-ERR DUMP payload version or checksum are wrong\r\n"
            );
        }
    }

    /// A `SCAN` of a database of eight stripes comes back with all of it.
    ///
    /// The cursor is the thing under test. It has to carry the stripe as well
    /// as the place in it, so a client that stops at one stripe and comes back
    /// carries on in that stripe and not at the top of the database, and the
    /// walk has to end once rather than eight times.
    #[test]
    fn a_scan_of_a_striped_database_walks_all_of_it() {
        let mut f = Fixture::striped(8);
        for i in 0..500 {
            let key = format!("key:{i}");
            f.run(&[b"SET", key.as_bytes(), b"v"]);
        }

        let mut seen = Vec::new();
        let mut cursor = "0".to_owned();
        let mut calls = 0;
        loop {
            let reply = f.run(&[b"SCAN", cursor.as_bytes(), b"COUNT", b"10"]);
            let (next, keys) = scan_reply(&reply);
            seen.extend(keys);
            cursor = next;
            calls += 1;
            assert!(calls < 5_000, "a scan that will not finish");
            if cursor == "0" {
                break;
            }
        }
        seen.sort();
        assert_eq!(seen.len(), 500, "a quiet scan answered a key twice");
        assert_eq!(seen, sorted(&f.run(&[b"KEYS", b"*"])));

        // And the options still work when the walk is over several stripes,
        // since a `MATCH` is applied to keys a stripe handed up and a `TYPE` is
        // applied by each stripe on the way.
        let reply = f.run(&[b"SCAN", b"0", b"COUNT", b"1000", b"MATCH", b"key:4?"]);
        let (_, keys) = scan_reply(&reply);
        assert_eq!(keys.len(), 10, "key:40 through key:49");
        let reply = f.run(&[b"SCAN", b"0", b"COUNT", b"1000", b"TYPE", b"list"]);
        let (_, keys) = scan_reply(&reply);
        assert!(keys.is_empty(), "nothing here is a list");
    }

    /// `RANDOMKEY` on a striped database answers a key from any of the stripes.
    ///
    /// The draw picks the stripe first, so the thing that can go wrong is that
    /// it always picks the same one, and two hundred draws over eight stripes
    /// would make that obvious.
    #[test]
    fn a_random_key_can_come_from_any_stripe() {
        let mut f = Fixture::striped(8);
        assert_eq!(f.run(&[b"RANDOMKEY"]), "$-1\r\n");
        for i in 0..200 {
            let key = format!("key:{i}");
            f.run(&[b"SET", key.as_bytes(), b"v"]);
        }
        let mut homes = std::collections::HashSet::new();
        for _ in 0..200 {
            let got = f.run(&[b"RANDOMKEY"]);
            let key = got.split("\r\n").nth(1).expect("a key").to_owned();
            assert_eq!(f.run(&[b"EXISTS", key.as_bytes()]), ":1\r\n");
            homes.insert(f.server.striped(0).stripe_of(key.as_bytes()));
        }
        assert_eq!(homes.len(), 8, "some stripe was never drawn from");
    }

    /// Two keys that are not on the same stripe, which is what `RENAME` and
    /// `COPY` have to cope with and what a test has to arrange rather than
    /// hope for.
    fn apart(f: &mut Fixture, src: &str) -> String {
        let home = f.server.striped(0).stripe_of(src.as_bytes());
        for i in 0..1_000 {
            let dst = format!("dst:{i}");
            if f.server.striped(0).stripe_of(dst.as_bytes()) != home {
                return dst;
            }
        }
        panic!("eight stripes and a thousand keys all landed in one place");
    }

    /// A rename whose two keys are on two stripes moves the value, the deadline
    /// and, for a collection, the body itself.
    #[test]
    fn a_rename_across_stripes_takes_everything_with_it() {
        let mut f = Fixture::striped(8);
        let dst = apart(&mut f, "src");
        let (src, dst) = (b"src".as_slice(), dst.as_bytes());

        f.run(&[b"SET", src, b"v"]);
        f.run(&[b"EXPIRE", src, b"100"]);
        assert_eq!(f.run(&[b"RENAME", src, dst]), "+OK\r\n");
        assert_eq!(f.run(&[b"EXISTS", src, dst]), ":1\r\n");
        assert_eq!(f.run(&[b"GET", dst]), "$1\r\nv\r\n");
        assert_eq!(f.run(&[b"TTL", dst]), ":100\r\n", "the deadline came too");

        // A list, because a string lives in its record and a collection lives
        // in a slab, and the second of those is the one that can be left
        // behind. Planted through the store, since the list group has not been
        // taught about stripes yet.
        f.server
            .striped(0)
            .hold(src)
            .push(src, yo_kv::End::Right, [&b"a"[..], &b"b"[..]].into_iter())
            .expect("a new list");
        assert_eq!(f.run(&[b"RENAME", src, dst]), "+OK\r\n");
        assert_eq!(f.run(&[b"TYPE", dst]), "+list\r\n");
        assert_eq!(
            f.server.striped(0).hold(dst).llen(dst).expect("a list"),
            2,
            "the members are on the stripe the key moved to"
        );

        // And `RENAMENX` still refuses a destination that is taken, which is
        // the one answer the cross stripe path has to work out for itself.
        f.run(&[b"SET", src, b"v"]);
        assert_eq!(f.run(&[b"RENAMENX", src, dst]), ":0\r\n");
        assert_eq!(f.run(&[b"TYPE", dst]), "+list\r\n", "and left it alone");
        assert_eq!(f.run(&[b"GET", src]), "$1\r\nv\r\n", "and left the source");
    }

    /// And a copy across two stripes leaves both keys behind it.
    #[test]
    fn a_copy_across_stripes_leaves_the_source_where_it_was() {
        let mut f = Fixture::striped(8);
        let dst = apart(&mut f, "src");
        let (src, dst) = (b"src".as_slice(), dst.as_bytes());

        f.run(&[b"SET", src, b"v"]);
        assert_eq!(f.run(&[b"COPY", src, dst]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", src, dst]), ":2\r\n");
        assert_eq!(
            f.run(&[b"COPY", src, dst]),
            ":0\r\n",
            "the destination is taken"
        );
        f.run(&[b"SET", src, b"w"]);
        assert_eq!(f.run(&[b"COPY", src, dst, b"REPLACE"]), ":1\r\n");
        assert_eq!(f.run(&[b"GET", dst]), "$1\r\nw\r\n");

        // A collection is cloned rather than moved, so both keys have a body of
        // their own afterwards and writing to one does not show up in the
        // other.
        f.run(&[b"DEL", src, dst]);
        f.server
            .striped(0)
            .hold(src)
            .push(src, yo_kv::End::Right, [&b"a"[..], &b"b"[..]].into_iter())
            .expect("a new list");
        assert_eq!(f.run(&[b"COPY", src, dst]), ":1\r\n");
        f.server
            .striped(0)
            .hold(src)
            .push(src, yo_kv::End::Right, core::iter::once(&b"c"[..]))
            .expect("a list that is there");
        assert_eq!(f.server.striped(0).hold(src).llen(src).expect("a list"), 3);
        assert_eq!(f.server.striped(0).hold(dst).llen(dst).expect("a list"), 2);
    }

    /// Every bitmap command, on one stripe and on eight, replies compared byte
    /// for byte.
    ///
    /// `BITOP` is the one that names more than one key and it is where the work
    /// went. The rest are single key commands that now find their own stripe,
    /// and they are here because the cheapest way to be sure the routing is
    /// right is to ask.
    #[test]
    fn the_bitmap_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"SET", b"k1", b"foobar"],
            &[b"SETBIT", b"b1", b"7", b"1"],
            &[b"SETBIT", b"b1", b"7", b"0"],
            &[b"GETBIT", b"k1", b"6"],
            &[b"GETBIT", b"k1", b"100"],
            &[b"BITCOUNT", b"k1"],
            &[b"BITCOUNT", b"k1", b"0", b"0"],
            &[b"BITCOUNT", b"k1", b"5", b"30", b"BIT"],
            &[b"BITPOS", b"k1", b"1"],
            &[b"BITPOS", b"k1", b"0", b"2"],
            &[b"BITPOS", b"k1", b"1", b"2", b"-1", b"BIT"],
            &[
                b"BITFIELD",
                b"bf",
                b"SET",
                b"u8",
                b"0",
                b"255",
                b"GET",
                b"u8",
                b"0",
            ],
            &[
                b"BITFIELD",
                b"bf",
                b"OVERFLOW",
                b"SAT",
                b"INCRBY",
                b"u8",
                b"0",
                b"10",
            ],
            &[b"BITFIELD_RO", b"bf", b"GET", b"u8", b"0"],
            // The multi key one, over sources that are not on one stripe unless
            // eight stripes have folded into one.
            &[b"SET", b"s1", b"abc"],
            &[b"SET", b"s2", b"abd"],
            &[b"SET", b"s3", b"a"],
            &[b"BITOP", b"AND", b"d1", b"s1", b"s2"],
            &[b"GET", b"d1"],
            &[b"BITOP", b"OR", b"d2", b"s1", b"s2", b"s3"],
            &[b"GET", b"d2"],
            &[b"BITOP", b"XOR", b"d3", b"s1", b"s2"],
            &[b"STRLEN", b"d3"],
            &[b"BITOP", b"NOT", b"d4", b"s1"],
            &[b"STRLEN", b"d4"],
            &[b"BITOP", b"DIFF", b"d5", b"s1", b"s2"],
            &[b"BITOP", b"DIFF1", b"d6", b"s1", b"s2"],
            &[b"BITOP", b"ANDOR", b"d7", b"s1", b"s2"],
            &[b"BITOP", b"ONE", b"d8", b"s1", b"s2"],
            // A source that is not there reads as empty, and a result with
            // nothing in it deletes the destination rather than writing one.
            &[b"BITOP", b"AND", b"d1", b"gone", b"also-gone"],
            &[b"EXISTS", b"d1"],
            &[b"BITOP", b"OR", b"d9", b"s1", b"gone"],
            &[b"GET", b"d9"],
            // And the errors, which have to be the same errors. The key that
            // is not a string is planted below rather than pushed here, since
            // the list group has not been taught about stripes yet.
            &[b"BITOP", b"AND", b"d1", b"s1", b"list"],
            &[b"BITOP", b"AND", b"list", b"s1", b"s2"],
            &[b"BITOP", b"NOT", b"d1", b"s1", b"s2"],
            &[b"BITOP", b"DIFF", b"d1", b"s1"],
            &[b"BITOP", b"NOPE", b"d1", b"s1"],
            &[b"BITCOUNT", b"list"],
            &[b"BITFIELD_RO", b"bf", b"SET", b"u8", b"0", b"1"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for f in [&mut one, &mut many] {
            plant_list(f, b"list");
        }
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// A list under `key`, put there through the store.
    ///
    /// What a test does when it wants a key of the wrong type on a striped
    /// server, because the command that would make one is in a group that has
    /// not been taught about stripes yet.
    fn plant_list(f: &mut Fixture, key: &[u8]) {
        f.server
            .striped(0)
            .hold(key)
            .push(key, yo_kv::End::Right, core::iter::once(&b"x"[..]))
            .expect("a new list");
    }

    /// A `BITOP` whose keys are on two stripes reads both of them.
    ///
    /// The test above spreads its keys by hashing and would still pass if one
    /// stripe were doing all the work, since the answers would be the same. This
    /// one puts the destination and the two sources where they are known not to
    /// share a stripe.
    #[test]
    fn a_bitop_across_stripes_reads_every_source() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "src");
        let (src, far) = (b"src".as_slice(), other.as_bytes());
        assert_ne!(
            f.server.striped(0).stripe_of(src),
            f.server.striped(0).stripe_of(far),
            "the two keys are the point of the test"
        );

        f.run(&[b"SET", src, b"abc"]);
        f.run(&[b"SET", far, b"abd"]);
        assert_eq!(f.run(&[b"BITOP", b"AND", far, src, far]), ":3\r\n");
        assert_eq!(
            f.run(&[b"GET", far]),
            "$3\r\nab`\r\n",
            "a destination that is also a source"
        );
        f.run(&[b"SET", far, b"abd"]);
        assert_eq!(f.run(&[b"BITOP", b"XOR", src, src, far]), ":3\r\n");
        assert_eq!(
            f.run(&[b"GET", src]),
            "$3\r\n\0\0\x07\r\n",
            "and the other way round"
        );

        // A result of nothing deletes a destination on whatever stripe it is
        // on, and a source of the wrong type is refused before anything is
        // written.
        f.run(&[b"SET", src, b"abc"]);
        f.run(&[b"DEL", far]);
        assert_eq!(f.run(&[b"BITOP", b"AND", src, far, b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", src]), ":0\r\n");
        f.run(&[b"SET", src, b"abc"]);
        f.run(&[b"DEL", far]);
        plant_list(&mut f, far);
        assert_eq!(
            f.run(&[b"BITOP", b"OR", b"out", src, far]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", b"out"]), ":0\r\n");
    }

    /// Every HyperLogLog command, on one stripe and on eight.
    #[test]
    fn the_hyperloglog_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"PFADD", b"h1", b"a", b"b", b"c"],
            &[b"PFADD", b"h1", b"a"],
            &[b"PFADD", b"h2"],
            &[b"PFADD", b"h2", b"c", b"d", b"e"],
            &[b"PFCOUNT", b"h1"],
            &[b"PFCOUNT", b"h2"],
            &[b"PFCOUNT", b"missing"],
            // The two that name more than one key.
            &[b"PFCOUNT", b"h1", b"h2"],
            &[b"PFCOUNT", b"h1", b"missing"],
            &[b"PFMERGE", b"m", b"h1", b"h2"],
            &[b"PFCOUNT", b"m"],
            &[b"STRLEN", b"m"],
            &[b"PFMERGE", b"m"],
            &[b"PFCOUNT", b"m"],
            &[b"PFMERGE", b"m2", b"missing"],
            &[b"PFCOUNT", b"m2"],
            // The debugging ones, which are single key and change what they
            // look at.
            &[b"PFDEBUG", b"ENCODING", b"h1"],
            &[b"PFDEBUG", b"DECODE", b"h1"],
            &[b"PFDEBUG", b"TODENSE", b"h1"],
            &[b"PFDEBUG", b"ENCODING", b"h1"],
            &[b"PFDEBUG", b"TODENSE", b"h1"],
            &[b"PFCOUNT", b"h1", b"h2"],
            &[b"PFSELFTEST"],
            // And the errors.
            &[b"SET", b"plain", b"not a sketch at all"],
            &[b"PFADD", b"plain", b"a"],
            &[b"PFCOUNT", b"plain"],
            &[b"PFCOUNT", b"h1", b"plain"],
            &[b"PFMERGE", b"plain", b"h1"],
            &[b"PFMERGE", b"m", b"plain"],
            &[b"PFDEBUG", b"ENCODING", b"gone"],
            &[b"PFDEBUG", b"NOPE", b"h1"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// Every set command, on one stripe and on eight.
    ///
    /// The commands that answer members answer them in whatever order the set
    /// or the table they were built in holds them, so those replies are
    /// compared as sets. Everything else is compared byte for byte. Two servers
    /// agreeing on the order would be a fact about the tables and not about the
    /// answer, and asserting it would make this test fail for a reason nobody
    /// cares about.
    #[test]
    fn the_set_group_answers_the_same_however_many_stripes_there_are() {
        const UNORDERED: [&str; 4] = ["SMEMBERS", "SINTER", "SUNION", "SDIFF"];
        let script: &[&[&[u8]]] = &[
            &[b"SADD", b"s1", b"a", b"b", b"c"],
            &[b"SADD", b"s1", b"a"],
            &[b"SADD", b"s2", b"b", b"c", b"d"],
            &[b"SADD", b"ints", b"1", b"2", b"3"],
            &[b"SCARD", b"s1"],
            &[b"SISMEMBER", b"s1", b"a"],
            &[b"SISMEMBER", b"s1", b"z"],
            &[b"SMISMEMBER", b"s1", b"a", b"z", b"c"],
            &[b"SMEMBERS", b"s1"],
            &[b"SREM", b"s1", b"c"],
            &[b"SADD", b"s1", b"c"],
            &[b"SSCAN", b"s1", b"0"],
            &[b"SSCAN", b"s1", b"0", b"COUNT", b"100", b"MATCH", b"a*"],
            // The two draws, on a set of one member, which is the only shape
            // whose answer two servers have to agree on.
            &[b"SADD", b"one", b"m"],
            &[b"SRANDMEMBER", b"one"],
            &[b"SRANDMEMBER", b"one", b"-3"],
            &[b"SRANDMEMBER", b"gone"],
            &[b"SPOP", b"one"],
            &[b"SPOP", b"one"],
            &[b"SPOP", b"gone", b"2"],
            // The one that names two keys.
            &[b"SMOVE", b"s1", b"s2", b"a"],
            &[b"SMOVE", b"s1", b"s2", b"zzz"],
            &[b"SMOVE", b"gone", b"s2", b"a"],
            &[b"SMEMBERS", b"s1"],
            &[b"SMEMBERS", b"s2"],
            // The algebra.
            &[b"SINTER", b"s1", b"s2"],
            &[b"SUNION", b"s1", b"s2"],
            &[b"SDIFF", b"s2", b"s1"],
            &[b"SINTER", b"s1", b"gone"],
            &[b"SUNION", b"s1", b"gone"],
            &[b"SDIFF", b"gone", b"s1"],
            &[b"SINTER", b"ints", b"s1"],
            &[b"SINTERCARD", b"2", b"s1", b"s2"],
            &[b"SINTERCARD", b"2", b"s1", b"s2", b"LIMIT", b"1"],
            &[b"SUNIONCARD", b"2", b"s1", b"s2"],
            &[b"SDIFFCARD", b"2", b"s2", b"s1"],
            &[b"SINTERSTORE", b"d1", b"s1", b"s2"],
            &[b"SMEMBERS", b"d1"],
            &[b"SUNIONSTORE", b"d2", b"s1", b"s2"],
            &[b"SCARD", b"d2"],
            &[b"SDIFFSTORE", b"d3", b"s2", b"s1"],
            &[b"SCARD", b"d3"],
            // An empty result deletes the destination rather than storing a
            // set with nothing in it.
            &[b"SINTERSTORE", b"d4", b"s1", b"gone"],
            &[b"EXISTS", b"d4"],
            // And a destination that is also a source.
            &[b"SUNIONSTORE", b"s2", b"s1", b"s2"],
            &[b"SCARD", b"s2"],
            // The errors, which have to be the same errors.
            &[b"SET", b"str", b"v"],
            &[b"SADD", b"str", b"a"],
            &[b"SINTER", b"s1", b"str"],
            &[b"SINTERSTORE", b"d5", b"s1", b"str"],
            &[b"EXISTS", b"d5"],
            &[b"SMOVE", b"str", b"s2", b"a"],
            &[b"SMOVE", b"s1", b"str", b"b"],
            &[b"SMOVE", b"gone", b"str", b"b"],
            &[b"SINTERCARD", b"0", b"s1"],
            &[b"SINTERCARD", b"3", b"s1", b"s2"],
            &[b"SINTERCARD", b"2", b"s1", b"s2", b"LIMIT", b"-1"],
            &[b"SPOP", b"s1", b"-1"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            let name = String::from_utf8_lossy(parts[0]).to_uppercase();
            if UNORDERED.contains(&name.as_str()) && a.starts_with(['*', '~']) {
                assert_eq!(sorted(&a), sorted(&b), "{name}");
            } else {
                assert_eq!(a, b, "{name}");
            }
        }
    }

    /// The algebra over sets that are known to be on different stripes.
    #[test]
    fn a_set_operation_across_stripes_reads_every_set() {
        let mut f = Fixture::striped(8);
        let second = apart(&mut f, "s1");
        let third = apart(&mut f, &second);
        let (s1, s2, s3) = (b"s1".as_slice(), second.as_bytes(), third.as_bytes());

        f.run(&[b"SADD", s1, b"a", b"b", b"c"]);
        f.run(&[b"SADD", s2, b"b", b"c", b"d"]);
        assert_eq!(sorted(&f.run(&[b"SINTER", s1, s2])), ["b", "c"]);
        assert_eq!(
            sorted(&f.run(&[b"SUNION", s1, s2])),
            ["a", "b", "c", "d"],
            "a union of two stripes is both of them"
        );
        assert_eq!(sorted(&f.run(&[b"SDIFF", s1, s2])), ["a"]);
        assert_eq!(f.run(&[b"SINTERCARD", b"2", s1, s2]), ":2\r\n");
        assert_eq!(f.run(&[b"SUNIONCARD", b"2", s1, s2]), ":4\r\n");
        assert_eq!(f.run(&[b"SDIFFCARD", b"2", s1, s2]), ":1\r\n");

        // A destination on a third stripe, and then one that is also a source.
        assert_eq!(f.run(&[b"SINTERSTORE", s3, s1, s2]), ":2\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", s3])), ["b", "c"]);
        assert_eq!(f.run(&[b"SUNIONSTORE", s2, s1, s2]), ":4\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", s2])), ["a", "b", "c", "d"]);
        assert_eq!(f.run(&[b"SDIFFSTORE", s3, s2, s1]), ":1\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", s3])), ["d"]);

        // An empty result deletes a destination wherever it is, and a key of
        // the wrong type stops the command before the destination is touched.
        assert_eq!(f.run(&[b"SINTERSTORE", s3, s1, b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", s3]), ":0\r\n");
        f.run(&[b"SET", s3, b"v"]);
        assert_eq!(
            f.run(&[b"SINTER", s1, s3]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(f.run(&[b"GET", s3]), "$1\r\nv\r\n", "and left it alone");
    }

    /// An `SMOVE` whose two keys are on two stripes.
    #[test]
    fn a_move_across_stripes_takes_the_member_with_it() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "src");
        let (src, dst) = (b"src".as_slice(), other.as_bytes());

        f.run(&[b"SADD", src, b"a", b"b"]);
        f.run(&[b"SADD", dst, b"c"]);
        assert_eq!(f.run(&[b"SMOVE", src, dst, b"a"]), ":1\r\n");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", src])), ["b"]);
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", dst])), ["a", "c"]);
        assert_eq!(f.run(&[b"SMOVE", src, dst, b"a"]), ":0\r\n", "it has gone");

        // A destination that is not there is created on its own stripe, and a
        // source that loses its last member is deleted from its own.
        f.run(&[b"DEL", dst]);
        assert_eq!(f.run(&[b"SMOVE", src, dst, b"b"]), ":1\r\n");
        assert_eq!(f.run(&[b"EXISTS", src]), ":0\r\n", "the source is empty");
        assert_eq!(sorted(&f.run(&[b"SMEMBERS", dst])), ["b"]);

        // And a source that is not there answers zero without ever asking what
        // the destination holds, which is Redis's order and not the obvious
        // one.
        f.run(&[b"SET", dst, b"v"]);
        assert_eq!(f.run(&[b"SMOVE", src, dst, b"b"]), ":0\r\n");
        f.run(&[b"SADD", src, b"b"]);
        assert_eq!(
            f.run(&[b"SMOVE", src, dst, b"b"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
    }

    /// A count and a merge over sketches that are known to be on two stripes.
    #[test]
    fn a_pfcount_and_a_pfmerge_reach_across_stripes() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "src");
        let (src, far) = (b"src".as_slice(), other.as_bytes());

        for i in 0..150 {
            let ele = format!("e:{i}");
            f.run(&[b"PFADD", src, ele.as_bytes()]);
        }
        for i in 150..200 {
            let ele = format!("e:{i}");
            f.run(&[b"PFADD", far, ele.as_bytes()]);
        }
        // The three numbers a real server gives for these elements, which are
        // the numbers the single stripe tests in the keyspace crate check too.
        assert_eq!(f.run(&[b"PFCOUNT", src]), ":151\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", far]), ":49\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", src, far]), ":199\r\n");

        // A merge whose destination is on a third stripe, and then one that
        // writes into a source.
        let dest = apart(&mut f, &other);
        assert_eq!(f.run(&[b"PFMERGE", dest.as_bytes(), src, far]), "+OK\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", dest.as_bytes()]), ":199\r\n");
        assert_eq!(f.run(&[b"PFMERGE", far, src]), "+OK\r\n");
        assert_eq!(f.run(&[b"PFCOUNT", far]), ":199\r\n", "and kept its own");
        assert_eq!(f.run(&[b"PFCOUNT", src]), ":151\r\n", "and left the source");
    }

    /// Every sorted set command, on one stripe and on eight.
    ///
    /// Every reply here is compared byte for byte, unlike the set group, because
    /// a sorted set answers in rank order and members sharing a score come out
    /// in the order of their bytes. There is nothing left for the table the
    /// answer was built in to decide.
    #[test]
    fn the_sorted_set_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"ZADD", b"z1", b"1", b"a", b"2", b"b", b"3", b"c"],
            &[b"ZADD", b"z1", b"NX", b"9", b"a"],
            &[b"ZADD", b"z1", b"XX", b"CH", b"5", b"a"],
            &[b"ZADD", b"z1", b"GT", b"CH", b"1", b"a"],
            &[b"ZADD", b"z1", b"INCR", b"2", b"a"],
            &[b"ZINCRBY", b"z1", b"1.5", b"b"],
            &[b"ZADD", b"z2", b"1", b"b", b"2", b"c", b"3", b"d"],
            &[b"ZADD", b"lex", b"0", b"a", b"0", b"b", b"0", b"c"],
            &[b"ZADD", b"one", b"1", b"m"],
            &[b"ZCARD", b"z1"],
            &[b"ZCARD", b"gone"],
            &[b"ZSCORE", b"z1", b"a"],
            &[b"ZSCORE", b"z1", b"zz"],
            &[b"ZMSCORE", b"z1", b"a", b"zz", b"c"],
            &[b"ZRANK", b"z1", b"c"],
            &[b"ZRANK", b"z1", b"c", b"WITHSCORE"],
            &[b"ZREVRANK", b"z1", b"c"],
            &[b"ZRANK", b"z1", b"gone"],
            &[b"ZCOUNT", b"z1", b"-inf", b"+inf"],
            &[b"ZCOUNT", b"z1", b"(1", b"3"],
            &[b"ZLEXCOUNT", b"lex", b"-", b"+"],
            // The range commands, which are one parse and one walk.
            &[b"ZRANGE", b"z1", b"0", b"-1"],
            &[b"ZRANGE", b"z1", b"0", b"-1", b"WITHSCORES"],
            &[b"ZRANGE", b"z1", b"1", b"9", b"BYSCORE"],
            &[b"ZRANGE", b"z1", b"9", b"1", b"BYSCORE", b"REV"],
            &[b"ZRANGE", b"lex", b"[a", b"(c", b"BYLEX"],
            &[b"ZREVRANGE", b"z1", b"0", b"-1"],
            &[
                b"ZRANGEBYSCORE",
                b"z1",
                b"-inf",
                b"+inf",
                b"LIMIT",
                b"1",
                b"1",
            ],
            &[b"ZREVRANGEBYLEX", b"lex", b"+", b"-"],
            &[b"ZSCAN", b"z1", b"0"],
            &[b"ZSCAN", b"z1", b"0", b"MATCH", b"a*", b"COUNT", b"100"],
            // The draw, on a sorted set of one member, which is the only shape
            // whose answer two servers have to agree on.
            &[b"ZRANDMEMBER", b"one"],
            &[b"ZRANDMEMBER", b"one", b"-3", b"WITHSCORES"],
            &[b"ZRANDMEMBER", b"gone"],
            // The one that copies a window into another key.
            &[b"ZRANGESTORE", b"d0", b"z1", b"0", b"1"],
            &[b"ZRANGE", b"d0", b"0", b"-1", b"WITHSCORES"],
            &[b"ZRANGESTORE", b"d0", b"z1", b"5", b"1"],
            &[b"EXISTS", b"d0"],
            // The algebra, in both its shapes.
            &[b"ZUNION", b"2", b"z1", b"z2"],
            &[b"ZUNION", b"2", b"z1", b"z2", b"WITHSCORES"],
            &[
                b"ZUNION",
                b"2",
                b"z1",
                b"z2",
                b"WEIGHTS",
                b"2",
                b"3",
                b"AGGREGATE",
                b"MAX",
                b"WITHSCORES",
            ],
            &[b"ZINTER", b"2", b"z1", b"z2", b"WITHSCORES"],
            &[b"ZDIFF", b"2", b"z1", b"z2", b"WITHSCORES"],
            &[b"ZDIFF", b"2", b"gone", b"z1"],
            &[b"ZINTERCARD", b"2", b"z1", b"z2"],
            &[b"ZINTERCARD", b"2", b"z1", b"z2", b"LIMIT", b"1"],
            &[b"ZUNIONSTORE", b"d1", b"2", b"z1", b"z2"],
            &[b"ZRANGE", b"d1", b"0", b"-1", b"WITHSCORES"],
            &[
                b"ZINTERSTORE",
                b"d2",
                b"2",
                b"z1",
                b"z2",
                b"AGGREGATE",
                b"MIN",
            ],
            &[b"ZRANGE", b"d2", b"0", b"-1", b"WITHSCORES"],
            &[b"ZDIFFSTORE", b"d3", b"2", b"z1", b"z2"],
            &[b"ZCARD", b"d3"],
            // An empty result deletes the destination rather than storing a
            // sorted set with nothing in it.
            &[b"ZINTERSTORE", b"d4", b"2", b"z1", b"gone"],
            &[b"EXISTS", b"d4"],
            // A plain set is a sorted set where every score is one, so it is a
            // legal input to all of these.
            &[b"SADD", b"plain", b"a", b"x"],
            &[b"ZUNIONSTORE", b"d5", b"2", b"z1", b"plain"],
            &[b"ZRANGE", b"d5", b"0", b"-1", b"WITHSCORES"],
            // And a destination that is also a source.
            &[b"ZUNIONSTORE", b"z2", b"2", b"z1", b"z2"],
            &[b"ZRANGE", b"z2", b"0", b"-1", b"WITHSCORES"],
            // The three removals and the two pops.
            &[b"ZREM", b"d5", b"x", b"nothere"],
            &[b"ZREMRANGEBYRANK", b"d5", b"0", b"0"],
            &[b"ZREMRANGEBYSCORE", b"d1", b"-inf", b"1"],
            &[b"ZREMRANGEBYLEX", b"lex", b"[a", b"[a"],
            &[b"ZPOPMIN", b"z1"],
            &[b"ZPOPMAX", b"z1", b"2"],
            &[b"ZPOPMIN", b"gone"],
            &[b"ZMPOP", b"2", b"gone", b"z2", b"MIN"],
            &[b"ZMPOP", b"2", b"gone", b"nothere", b"MAX", b"COUNT", b"2"],
            // The errors, which have to be the same errors.
            &[b"SET", b"str", b"v"],
            &[b"ZADD", b"str", b"1", b"a"],
            &[b"ZSCORE", b"str", b"a"],
            &[b"ZADD", b"z1", b"nan", b"a"],
            &[b"ZUNION", b"2", b"z1", b"str"],
            &[b"ZUNIONSTORE", b"d6", b"2", b"z1", b"str"],
            &[b"EXISTS", b"d6"],
            &[b"ZINTERCARD", b"0", b"z1"],
            &[b"ZINTERCARD", b"2", b"z1", b"z2", b"LIMIT", b"-1"],
            &[b"ZRANGESTORE", b"d7", b"str", b"0", b"-1"],
            &[b"ZMPOP", b"1", b"str", b"MIN"],
            &[b"ZPOPMIN", b"z1", b"-1"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// The algebra over sorted sets that are known to be on different stripes.
    #[test]
    fn a_sorted_set_operation_across_stripes_reads_every_input() {
        let mut f = Fixture::striped(8);
        let second = apart(&mut f, "z1");
        let third = apart(&mut f, &second);
        let (z1, z2, z3) = (b"z1".as_slice(), second.as_bytes(), third.as_bytes());

        f.run(&[b"ZADD", z1, b"1", b"a", b"2", b"b"]);
        f.run(&[b"ZADD", z2, b"3", b"b", b"4", b"c"]);
        // a is 1, c is 4, b is 2 and 3 added together, which is the order they
        // come out in and the answer that says both stripes were read.
        assert_eq!(
            f.run(&[b"ZUNION", b"2", z1, z2]),
            "*3\r\n$1\r\na\r\n$1\r\nc\r\n$1\r\nb\r\n"
        );
        assert_eq!(f.run(&[b"ZINTER", b"2", z1, z2]), "*1\r\n$1\r\nb\r\n");
        assert_eq!(f.run(&[b"ZDIFF", b"2", z1, z2]), "*1\r\n$1\r\na\r\n");
        assert_eq!(f.run(&[b"ZINTERCARD", b"2", z1, z2]), ":1\r\n");
        assert_eq!(
            f.run(&[b"ZINTERCARD", b"2", z1, z2, b"LIMIT", b"1"]),
            ":1\r\n"
        );

        // A destination on a third stripe, and the weights and the aggregate
        // reaching every input.
        assert_eq!(f.run(&[b"ZUNIONSTORE", z3, b"2", z1, z2]), ":3\r\n");
        assert_eq!(f.run(&[b"ZSCORE", z3, b"b"]), "$1\r\n5\r\n");
        assert_eq!(
            f.run(&[
                b"ZUNIONSTORE",
                z3,
                b"2",
                z1,
                z2,
                b"WEIGHTS",
                b"2",
                b"3",
                b"AGGREGATE",
                b"MAX"
            ]),
            ":3\r\n"
        );
        assert_eq!(f.run(&[b"ZSCORE", z3, b"b"]), "$1\r\n9\r\n");
        assert_eq!(f.run(&[b"ZINTERSTORE", z3, b"2", z1, z2]), ":1\r\n");
        assert_eq!(f.run(&[b"ZCARD", z3]), ":1\r\n");
        assert_eq!(f.run(&[b"ZDIFFSTORE", z3, b"2", z2, z1]), ":1\r\n");
        assert_eq!(f.run(&[b"ZSCORE", z3, b"c"]), "$1\r\n4\r\n");

        // A pop over keys on several stripes takes from the first one that has
        // anything, which is what makes the order of the keys matter.
        let popped = format!(
            "*2\r\n${}\r\n{second}\r\n*1\r\n*2\r\n$1\r\nb\r\n$1\r\n3\r\n",
            second.len()
        );
        assert_eq!(f.run(&[b"ZMPOP", b"3", b"gone", z2, z1, b"MIN"]), popped);
        f.run(&[b"ZADD", z2, b"3", b"b"]);

        // An empty result deletes a destination wherever it is, and an input of
        // the wrong type stops the command before the destination is touched.
        assert_eq!(f.run(&[b"ZINTERSTORE", z3, b"2", z1, b"gone"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", z3]), ":0\r\n");
        f.run(&[b"SET", z3, b"v"]);
        assert_eq!(
            f.run(&[b"ZUNION", b"2", z1, z3]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(f.run(&[b"GET", z3]), "$1\r\nv\r\n", "and left it alone");

        // And a destination that is also a source works across stripes for the
        // reason it works on one: the whole result is built before anything is
        // written.
        assert_eq!(f.run(&[b"ZUNIONSTORE", z2, b"2", z1, z2]), ":3\r\n");
        assert_eq!(f.run(&[b"ZSCORE", z2, b"b"]), "$1\r\n5\r\n");
        assert_eq!(f.run(&[b"ZCARD", z2]), ":3\r\n");
    }

    /// A `ZRANGESTORE` whose two keys are on two stripes.
    #[test]
    fn a_range_store_across_stripes_copies_the_window() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "src");
        let third = apart(&mut f, &other);
        let (src, dst, plain) = (b"src".as_slice(), other.as_bytes(), third.as_bytes());

        f.run(&[b"ZADD", src, b"1", b"a", b"2", b"b", b"3", b"c"]);
        assert_eq!(f.run(&[b"ZRANGESTORE", dst, src, b"0", b"1"]), ":2\r\n");
        assert_eq!(
            f.run(&[b"ZRANGE", dst, b"0", b"-1", b"WITHSCORES"]),
            "*4\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n"
        );
        assert_eq!(f.run(&[b"ZCARD", src]), ":3\r\n", "the source kept its own");

        // A window walked backwards takes the other end of the sorted set and
        // still stores what it took in score order.
        assert_eq!(
            f.run(&[
                b"ZRANGESTORE",
                dst,
                src,
                b"+inf",
                b"-inf",
                b"BYSCORE",
                b"REV",
                b"LIMIT",
                b"0",
                b"2"
            ]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", dst, b"0", b"-1"]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );

        // An empty window deletes the destination on its own stripe, and a
        // source of the wrong type is refused before the destination is touched.
        assert_eq!(f.run(&[b"ZRANGESTORE", dst, src, b"5", b"1"]), ":0\r\n");
        assert_eq!(f.run(&[b"EXISTS", dst]), ":0\r\n");
        f.run(&[b"ZRANGESTORE", dst, src, b"0", b"-1"]);
        f.run(&[b"SET", plain, b"v"]);
        assert_eq!(
            f.run(&[b"ZRANGESTORE", dst, plain, b"0", b"-1"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(
            f.run(&[b"ZCARD", dst]),
            ":3\r\n",
            "and left the destination"
        );
    }

    /// Every list command, on one stripe and on eight.
    ///
    /// The blocking six are in here too, both when they can be answered on the
    /// spot and when they cannot, since a command that parks its client writes
    /// nothing at all and two servers have to agree about that as much as they
    /// agree about a reply.
    #[test]
    fn the_list_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"RPUSH", b"l1", b"a", b"b", b"c"],
            &[b"LPUSH", b"l1", b"z"],
            &[b"RPUSHX", b"l1", b"d"],
            &[b"LPUSHX", b"gone", b"x"],
            &[b"RPUSHX", b"gone", b"x"],
            &[b"LLEN", b"l1"],
            &[b"LLEN", b"gone"],
            &[b"LRANGE", b"l1", b"0", b"-1"],
            &[b"LRANGE", b"l1", b"1", b"2"],
            &[b"LRANGE", b"l1", b"5", b"9"],
            &[b"LINDEX", b"l1", b"0"],
            &[b"LINDEX", b"l1", b"-1"],
            &[b"LINDEX", b"l1", b"99"],
            &[b"LSET", b"l1", b"0", b"y"],
            &[b"LINSERT", b"l1", b"BEFORE", b"b", b"aa"],
            &[b"LINSERT", b"l1", b"AFTER", b"nothere", b"x"],
            &[b"LPOS", b"l1", b"b"],
            &[b"LPOS", b"l1", b"b", b"COUNT", b"0"],
            &[b"LPOS", b"l1", b"nothere"],
            &[b"LPOS", b"l1", b"b", b"RANK", b"-1", b"MAXLEN", b"2"],
            &[b"LREM", b"l1", b"1", b"aa"],
            &[b"LTRIM", b"l1", b"0", b"3"],
            &[b"LRANGE", b"l1", b"0", b"-1"],
            &[b"LPOP", b"l1"],
            &[b"RPOP", b"l1"],
            &[b"LPOP", b"l1", b"2"],
            &[b"LPOP", b"gone"],
            &[b"LPOP", b"gone", b"2"],
            &[b"EXISTS", b"l1"],
            // The ones that name two keys, and the one that takes a block of
            // elements rather than the one on the end.
            &[b"RPUSH", b"src", b"a", b"b", b"c", b"d"],
            &[b"LMOVE", b"src", b"dst", b"LEFT", b"RIGHT"],
            &[b"RPOPLPUSH", b"src", b"dst"],
            &[b"LRANGE", b"dst", b"0", b"-1"],
            &[b"LMOVE", b"gone", b"dst", b"LEFT", b"RIGHT"],
            &[b"LMOVEM", b"src", b"dst", b"LEFT", b"RIGHT"],
            &[
                b"LMOVEM", b"src", b"dst", b"LEFT", b"RIGHT", b"COUNT", b"2", b"BULK",
            ],
            &[
                b"LMOVEM", b"dst", b"dst", b"LEFT", b"RIGHT", b"COUNT", b"2", b"OBO",
            ],
            &[b"LRANGE", b"dst", b"0", b"-1"],
            &[
                b"LMOVEM", b"src", b"dst", b"LEFT", b"RIGHT", b"EXACTLY", b"9", b"BULK",
            ],
            &[b"LMPOP", b"2", b"gone", b"dst", b"LEFT"],
            &[b"LMPOP", b"2", b"gone", b"dst", b"RIGHT", b"COUNT", b"2"],
            &[b"LMPOP", b"1", b"gone", b"LEFT"],
            // The blocking ones, first with something there to answer them and
            // then with nothing, which parks the client and writes nothing.
            &[b"RPUSH", b"q", b"a", b"b", b"c"],
            &[b"BLPOP", b"gone", b"q", b"0"],
            &[b"BRPOP", b"q", b"0"],
            &[b"BLMPOP", b"0", b"2", b"gone", b"q", b"LEFT"],
            &[b"RPUSH", b"q", b"x", b"y", b"z"],
            &[b"BLMOVE", b"q", b"dst", b"LEFT", b"RIGHT", b"0"],
            &[b"BRPOPLPUSH", b"q", b"dst", b"0"],
            &[b"BLMOVEM", b"q", b"dst", b"LEFT", b"RIGHT", b"0"],
            &[b"BLPOP", b"q", b"0"],
            &[b"BLMOVE", b"q", b"dst", b"LEFT", b"RIGHT", b"0"],
            // The errors, which have to be the same errors.
            &[b"SET", b"plain", b"v"],
            &[b"LPUSH", b"plain", b"a"],
            &[b"LLEN", b"plain"],
            &[b"LMOVE", b"dst", b"plain", b"LEFT", b"RIGHT"],
            &[b"LRANGE", b"dst", b"0", b"-1"],
            &[b"LMOVEM", b"dst", b"plain", b"LEFT", b"RIGHT"],
            &[b"LSET", b"gone", b"0", b"v"],
            &[b"LSET", b"dst", b"99", b"v"],
            &[b"LPOP", b"dst", b"-1"],
            &[b"LMPOP", b"0", b"dst", b"LEFT"],
            &[b"LPOS", b"dst", b"a", b"RANK", b"0"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// An `LMOVE` and an `LMOVEM` whose two keys are on two stripes.
    #[test]
    fn a_list_move_across_stripes_takes_the_elements_with_it() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "src");
        let third = apart(&mut f, &other);
        let (src, dst, plain) = (b"src".as_slice(), other.as_bytes(), third.as_bytes());

        f.run(&[b"RPUSH", src, b"a", b"b", b"c", b"d"]);
        assert_eq!(
            f.run(&[b"LMOVE", src, dst, b"LEFT", b"RIGHT"]),
            "$1\r\na\r\n"
        );
        assert_eq!(f.run(&[b"RPOPLPUSH", src, dst]), "$1\r\nd\r\n");
        assert_eq!(
            f.run(&[b"LRANGE", dst, b"0", b"-1"]),
            "*2\r\n$1\r\nd\r\n$1\r\na\r\n",
            "one went on each end of the destination"
        );
        assert_eq!(
            f.run(&[b"LRANGE", src, b"0", b"-1"]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );

        // A block of them, which under BULK arrives in the order it left.
        assert_eq!(
            f.run(&[
                b"LMOVEM", src, dst, b"LEFT", b"RIGHT", b"COUNT", b"2", b"BULK"
            ]),
            "*2\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"LRANGE", dst, b"0", b"-1"]),
            "*4\r\n$1\r\nd\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"
        );
        assert_eq!(
            f.run(&[b"EXISTS", src]),
            ":0\r\n",
            "and the source is gone with its last element"
        );

        // An `EXACTLY` the source cannot fill moves nothing, and a source that
        // is not there at all is the two kinds of nothing the two commands have.
        f.run(&[b"RPUSH", src, b"e", b"f"]);
        assert_eq!(
            f.run(&[
                b"LMOVEM", src, dst, b"LEFT", b"RIGHT", b"EXACTLY", b"3", b"BULK"
            ]),
            "*-1\r\n"
        );
        assert_eq!(f.run(&[b"LLEN", src]), ":2\r\n", "and took none of them");
        assert_eq!(
            f.run(&[b"LMOVE", b"gone", dst, b"LEFT", b"RIGHT"]),
            "$-1\r\n"
        );
        assert_eq!(
            f.run(&[b"LMOVEM", b"gone", dst, b"LEFT", b"RIGHT"]),
            "*-1\r\n"
        );

        // A destination of the wrong type is refused before anything is taken,
        // which is the order that matters most here, since an element already
        // out of the source would have nowhere to go back to.
        f.run(&[b"SET", plain, b"v"]);
        assert_eq!(
            f.run(&[b"LMOVE", src, plain, b"LEFT", b"RIGHT"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(
            f.run(&[b"LLEN", src]),
            ":2\r\n",
            "and left the source alone"
        );
        assert_eq!(
            f.run(&[b"LMOVEM", src, plain, b"LEFT", b"RIGHT"]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(f.run(&[b"LLEN", src]), ":2\r\n");
    }

    /// A parked client served by a push that landed on another stripe.
    ///
    /// A waiter remembers the database and not the stripe, which is the point:
    /// serving it runs the same attempt the command ran, and the attempt finds
    /// the stripe each of its keys is on for itself.
    #[test]
    fn a_parked_client_is_served_from_the_stripe_its_key_is_on() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "q");
        let (q, far) = (b"q".as_slice(), other.as_bytes());

        assert_eq!(f.flow(&[b"BLPOP", q, far, b"0"]).0, Flow::Block);
        assert_eq!(f.server.parked(), 1);
        f.run(&[b"RPUSH", far, b"v"]);
        let mut out = Out::new(Proto::Resp2);
        assert!(f.server.serve_waiter(0, 0, &mut out));
        let want = format!("*2\r\n${}\r\n{other}\r\n$1\r\nv\r\n", other.len());
        assert_eq!(core::str::from_utf8(out.as_slice()).expect("ascii"), want);
        assert_eq!(
            f.run(&[b"EXISTS", far]),
            ":0\r\n",
            "and it took the element with it"
        );

        // And a move across two stripes is served the same way, by the push
        // that fills its source.
        f.server.forget_waiters(7);
        assert_eq!(
            f.flow(&[b"BLMOVE", q, far, b"LEFT", b"RIGHT", b"0"]).0,
            Flow::Block
        );
        f.run(&[b"RPUSH", q, b"w"]);
        let mut out = Out::new(Proto::Resp2);
        assert!(f.server.serve_waiter(0, 0, &mut out));
        assert_eq!(
            core::str::from_utf8(out.as_slice()).expect("ascii"),
            "$1\r\nw\r\n"
        );
        assert_eq!(f.run(&[b"LRANGE", far, b"0", b"-1"]), "*1\r\n$1\r\nw\r\n");
    }

    /// Every stream command, on one stripe and on eight.
    ///
    /// Every ID is written out rather than left to the clock, so the two servers
    /// are being compared on what they store and not on how long the test took
    /// to get from one of them to the other.
    #[test]
    fn the_stream_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"XADD", b"s", b"1-1", b"a", b"1"],
            &[b"XADD", b"s", b"2-1", b"b", b"2", b"c", b"3"],
            &[b"XADD", b"s", b"3-1", b"d", b"4"],
            &[b"XADD", b"s", b"1-1", b"e", b"5"],
            &[b"XADD", b"nomk", b"NOMKSTREAM", b"1-1", b"a", b"1"],
            &[b"XLEN", b"s"],
            &[b"XLEN", b"gone"],
            &[b"XRANGE", b"s", b"-", b"+"],
            &[b"XRANGE", b"s", b"2", b"+", b"COUNT", b"1"],
            &[b"XRANGE", b"gone", b"-", b"+", b"COUNT", b"0"],
            &[b"XRANGE", b"s", b"-", b"+", b"COUNT", b"0"],
            &[b"XREVRANGE", b"s", b"+", b"-"],
            &[b"XREAD", b"COUNT", b"2", b"STREAMS", b"s", b"0"],
            &[b"XREAD", b"STREAMS", b"s", b"gone", b"0", b"0"],
            &[b"XREAD", b"STREAMS", b"s", b"$"],
            // The groups, which is where most of the state is.
            &[b"XGROUP", b"CREATE", b"s", b"g", b"0"],
            &[b"XGROUP", b"CREATE", b"s", b"g", b"0"],
            &[b"XGROUP", b"CREATE", b"gone", b"g", b"0"],
            &[b"XGROUP", b"CREATE", b"made", b"g", b"$", b"MKSTREAM"],
            &[b"XGROUP", b"CREATECONSUMER", b"s", b"g", b"idle"],
            &[b"XREADGROUP", b"GROUP", b"g", b"c1", b"STREAMS", b"s", b">"],
            &[
                b"XREADGROUP",
                b"GROUP",
                b"g",
                b"c1",
                b"COUNT",
                b"1",
                b"STREAMS",
                b"s",
                b"0",
            ],
            &[
                b"XREADGROUP",
                b"GROUP",
                b"nope",
                b"c1",
                b"STREAMS",
                b"s",
                b">",
            ],
            &[b"XPENDING", b"s", b"g"],
            &[b"XPENDING", b"s", b"g", b"-", b"+", b"10"],
            &[b"XPENDING", b"s", b"g", b"-", b"+", b"10", b"c1"],
            &[b"XPENDING", b"s", b"nope"],
            &[b"XCLAIM", b"s", b"g", b"c2", b"0", b"1-1"],
            &[b"XCLAIM", b"s", b"g", b"c2", b"0", b"2-1", b"JUSTID"],
            &[b"XAUTOCLAIM", b"s", b"g", b"c3", b"0", b"0"],
            &[b"XACK", b"s", b"g", b"1-1"],
            &[b"XACK", b"s", b"g", b"1-1"],
            &[b"XNACK", b"s", b"g", b"FAIL", b"IDS", b"1", b"2-1"],
            &[b"XPENDING", b"s", b"g"],
            &[b"XINFO", b"STREAM", b"s"],
            &[b"XINFO", b"GROUPS", b"s"],
            &[b"XINFO", b"CONSUMERS", b"s", b"g"],
            &[b"XINFO", b"STREAM", b"gone"],
            // Deleting, trimming and moving the ID on.
            &[b"XDEL", b"s", b"3-1"],
            &[b"XDELEX", b"s", b"DELREF", b"IDS", b"1", b"2-1"],
            &[b"XACKDEL", b"s", b"g", b"KEEPREF", b"IDS", b"1", b"1-1"],
            &[b"XADD", b"s", b"9-1", b"z", b"9"],
            &[b"XTRIM", b"s", b"MAXLEN", b"1"],
            &[b"XTRIM", b"s", b"MINID", b"9"],
            &[b"XSETID", b"s", b"99-1"],
            &[b"XSETID", b"s", b"1-1"],
            &[b"XLEN", b"s"],
            &[b"XGROUP", b"SETID", b"s", b"g", b"0"],
            &[b"XGROUP", b"DELCONSUMER", b"s", b"g", b"c1"],
            &[b"XGROUP", b"DESTROY", b"s", b"g"],
            &[b"XGROUP", b"DESTROY", b"s", b"g"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"XADD", b"plain", b"1-1", b"a", b"1"],
            &[b"XLEN", b"plain"],
            &[b"XREAD", b"STREAMS", b"plain", b"0"],
            &[b"XRANGE", b"s", b"bogus", b"+"],
            &[b"XADD", b"s", b"1-1", b"a"],
            &[b"XREAD", b"STREAMS", b"s", b"gone", b"0"],
            &[b"XREADGROUP", b"GROUP", b"g", b"c", b"STREAMS", b"s", b"$"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// An `XREAD` and an `XREADGROUP` naming two keys on two stripes.
    ///
    /// Nothing is shared between the two streams, so the only thing this can go
    /// wrong at is looking both of them up, which is exactly what a read that
    /// held one database and walked it would get wrong.
    #[test]
    fn a_stream_read_across_stripes_reads_every_key() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "s1");
        let (s1, s2) = (b"s1".as_slice(), other.as_bytes());

        f.run(&[b"XADD", s1, b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", s2, b"2-1", b"b", b"2"]);
        let got = f.run(&[b"XREAD", b"STREAMS", s1, s2, b"0", b"0"]);
        assert!(got.starts_with("*2\r\n"), "both streams answered: {got}");
        assert!(got.contains("1-1"), "the first one is in there: {got}");
        assert!(got.contains("2-1"), "and so is the second: {got}");

        // A group read looks its group up on every key before it reads any of
        // them, so a group that is missing on the far key stops the near one.
        f.run(&[b"XGROUP", b"CREATE", s1, b"g", b"0"]);
        let got = f.run(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c",
            b"STREAMS",
            s1,
            s2,
            b">",
            b">",
        ]);
        assert!(got.starts_with("-NOGROUP"), "{got}");
        assert_eq!(
            f.run(&[b"XPENDING", s1, b"g"]),
            "*4\r\n:0\r\n$-1\r\n$-1\r\n*-1\r\n",
            "and read nothing from the key that did have the group"
        );

        f.run(&[b"XGROUP", b"CREATE", s2, b"g", b"0"]);
        let got = f.run(&[
            b"XREADGROUP",
            b"GROUP",
            b"g",
            b"c",
            b"STREAMS",
            s1,
            s2,
            b">",
            b">",
        ]);
        assert!(got.starts_with("*2\r\n"), "now both are read: {got}");
    }

    /// A client parked on an `XREAD` woken by an entry on another stripe.
    #[test]
    fn a_parked_stream_reader_is_served_from_the_stripe_its_key_is_on() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "s1");
        let (s1, far) = (b"s1".as_slice(), other.as_bytes());
        f.run(&[b"XADD", s1, b"1-1", b"a", b"1"]);
        f.run(&[b"XADD", far, b"1-1", b"a", b"1"]);

        assert_eq!(
            f.flow(&[b"XREAD", b"BLOCK", b"0", b"STREAMS", s1, far, b"$", b"$"])
                .0,
            Flow::Block
        );
        f.run(&[b"XADD", far, b"2-1", b"b", b"2"]);
        let mut out = Out::new(Proto::Resp2);
        assert!(f.server.serve_waiter(0, 0, &mut out));
        let want = format!(
            "*1\r\n*2\r\n${}\r\n{other}\r\n*1\r\n*2\r\n$3\r\n2-1\r\n*2\r\n$1\r\nb\r\n$1\r\n2\r\n",
            other.len()
        );
        assert_eq!(core::str::from_utf8(out.as_slice()).expect("ascii"), want);
    }

    /// Every JSON command, on one stripe and on eight.
    #[test]
    fn the_json_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[
                b"JSON.SET",
                b"d",
                b"$",
                br#"{"a":1,"b":[1,2,3],"s":"hi","t":true}"#,
            ],
            &[b"JSON.SET", b"d", b"$.a", b"2"],
            &[b"JSON.SET", b"d", b"$.new", b"9", b"NX"],
            &[b"JSON.SET", b"d", b"$.new", b"8", b"NX"],
            &[b"JSON.SET", b"d", b"$.nope", b"7", b"XX"],
            &[b"JSON.GET", b"d"],
            &[b"JSON.GET", b"d", b"$.b"],
            &[b"JSON.GET", b"gone", b"$"],
            &[b"JSON.TYPE", b"d", b"$.b"],
            &[b"JSON.TYPE", b"d", b"$.s"],
            &[b"JSON.TOGGLE", b"d", b"$.t"],
            &[b"JSON.ARRLEN", b"d", b"$.b"],
            &[b"JSON.OBJLEN", b"d", b"$"],
            &[b"JSON.OBJKEYS", b"d", b"$"],
            &[b"JSON.STRLEN", b"d", b"$.s"],
            &[b"JSON.STRAPPEND", b"d", b"$.s", br#""there""#],
            &[b"JSON.ARRAPPEND", b"d", b"$.b", b"4"],
            &[b"JSON.ARRINSERT", b"d", b"$.b", b"0", b"0"],
            &[b"JSON.ARRINDEX", b"d", b"$.b", b"3"],
            &[b"JSON.ARRTRIM", b"d", b"$.b", b"1", b"3"],
            &[b"JSON.ARRPOP", b"d", b"$.b"],
            &[b"JSON.NUMINCRBY", b"d", b"$.a", b"5"],
            &[b"JSON.NUMMULTBY", b"d", b"$.a", b"2"],
            &[b"JSON.NUMPOWBY", b"d", b"$.a", b"2"],
            &[b"JSON.MERGE", b"d", b"$", br#"{"a":null,"m":1}"#],
            &[b"JSON.RESP", b"d", b"$.b"],
            &[b"JSON.DEBUG", b"MEMORY", b"d"],
            &[b"JSON.CLEAR", b"d", b"$.b"],
            &[b"JSON.DEL", b"d", b"$.m"],
            &[b"JSON.FORGET", b"d", b"$.nothere"],
            // The two that name more than one key.
            &[
                b"JSON.MSET",
                b"m1",
                b"$",
                b"1",
                b"m2",
                b"$",
                b"2",
                b"m3",
                b"$",
                b"3",
            ],
            &[b"JSON.MGET", b"m1", b"m2", b"m3", b"gone", b"$"],
            &[b"JSON.MSET", b"m1", b"$", b"9", b"m2", b"$.deep", b"9"],
            &[b"JSON.GET", b"m1", b"$"],
            &[b"JSON.MSET", b"m1", b"$", b"nonsense", b"m2", b"$", b"5"],
            &[b"JSON.GET", b"m2", b"$"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"JSON.GET", b"plain", b"$"],
            &[b"JSON.SET", b"plain", b"$", b"1"],
            &[b"JSON.MGET", b"m1", b"plain", b"$"],
            &[b"JSON.SET", b"d", b"$.b", b"["],
            &[b"JSON.DEL", b"plain"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// A `JSON.MSET` and a `JSON.MGET` whose keys are on several stripes.
    ///
    /// `JSON.MSET` works every triple out against the keyspace as it was before
    /// the command and writes nothing until all of them are known to work, so
    /// the thing to check is that a triple that cannot be written stops the
    /// ones on other stripes as well as the ones on its own.
    #[test]
    fn a_json_multi_write_across_stripes_reaches_every_key() {
        let mut f = Fixture::striped(8);
        let second = apart(&mut f, "m1");
        let third = apart(&mut f, &second);
        let (m1, m2, m3) = (b"m1".as_slice(), second.as_bytes(), third.as_bytes());

        assert_eq!(
            f.run(&[b"JSON.MSET", m1, b"$", b"1", m2, b"$", b"2", m3, b"$", b"3"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"JSON.MGET", m1, m2, m3, b"gone", b"$"]),
            "*4\r\n$3\r\n[1]\r\n$3\r\n[2]\r\n$3\r\n[3]\r\n$-1\r\n"
        );

        // A value that is not JSON is refused before anything is written, and
        // the key on the far stripe keeps what it had.
        assert_eq!(
            f.run(&[b"JSON.MSET", m1, b"$", b"9", m2, b"$", b"nonsense"]),
            "-this is not the start of a value, at byte 0 of the JSON text\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", m1, b"$"]), "$3\r\n[1]\r\n");

        // A path that names nowhere is not an error. That triple is skipped,
        // the ones on the other stripes are still written, and the reply is a
        // nil rather than OK.
        assert_eq!(
            f.run(&[
                b"JSON.MSET",
                m1,
                b"$",
                b"9",
                m2,
                b"$.deep",
                b"9",
                m3,
                b"$",
                b"7"
            ]),
            "$-1\r\n"
        );
        assert_eq!(f.run(&[b"JSON.GET", m1, b"$"]), "$3\r\n[9]\r\n");
        assert_eq!(f.run(&[b"JSON.GET", m2, b"$"]), "$3\r\n[2]\r\n");
        assert_eq!(f.run(&[b"JSON.GET", m3, b"$"]), "$3\r\n[7]\r\n");
    }

    /// Every geospatial command, on one stripe and on eight.
    #[test]
    fn the_geo_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[
                b"GEOADD",
                b"g",
                b"13.361389",
                b"38.115556",
                b"palermo",
                b"15.087269",
                b"37.502669",
                b"catania",
            ],
            &[
                b"GEOADD",
                b"g",
                b"NX",
                b"13.361389",
                b"38.115556",
                b"palermo",
            ],
            &[b"GEOADD", b"g", b"XX", b"CH", b"13.4", b"38.1", b"palermo"],
            &[b"GEOPOS", b"g", b"palermo", b"nothere"],
            &[b"GEOHASH", b"g", b"palermo", b"catania"],
            &[b"GEODIST", b"g", b"palermo", b"catania"],
            &[b"GEODIST", b"g", b"palermo", b"catania", b"KM"],
            &[b"GEODIST", b"g", b"palermo", b"nothere"],
            &[
                b"GEOSEARCH",
                b"g",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"KM",
                b"ASC",
                b"WITHCOORD",
                b"WITHDIST",
                b"WITHHASH",
            ],
            &[
                b"GEOSEARCH",
                b"g",
                b"FROMMEMBER",
                b"palermo",
                b"BYBOX",
                b"400",
                b"400",
                b"KM",
                b"DESC",
            ],
            &[
                b"GEORADIUS",
                b"g",
                b"15",
                b"37",
                b"200",
                b"KM",
                b"COUNT",
                b"1",
            ],
            &[b"GEORADIUSBYMEMBER", b"g", b"palermo", b"200", b"KM"],
            &[b"GEORADIUSBYMEMBER_RO", b"g", b"nothere", b"200", b"KM"],
            &[
                b"GEOSEARCHSTORE",
                b"dst",
                b"g",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"KM",
            ],
            &[b"ZRANGE", b"dst", b"0", b"-1"],
            &[
                b"GEOSEARCHSTORE",
                b"dst",
                b"g",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"1",
                b"M",
                b"STOREDIST",
            ],
            &[b"EXISTS", b"dst"],
            &[
                b"GEORADIUS",
                b"g",
                b"15",
                b"37",
                b"200",
                b"KM",
                b"STORE",
                b"dst",
            ],
            &[b"ZCARD", b"dst"],
            // And the errors.
            &[b"GEOADD", b"g", b"181", b"38", b"nowhere"],
            &[b"SET", b"plain", b"v"],
            &[b"GEOPOS", b"plain", b"a"],
            &[b"GEOSEARCH", b"g", b"FROMLONLAT", b"15", b"37"],
            &[
                b"GEOSEARCHSTORE",
                b"dst",
                b"g",
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"KM",
                b"WITHCOORD",
            ],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// A `GEOSEARCHSTORE` whose two keys are on two stripes.
    #[test]
    fn a_geo_search_store_across_stripes_writes_what_it_found() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "g");
        let third = apart(&mut f, &other);
        let (g, dst, plain) = (b"g".as_slice(), other.as_bytes(), third.as_bytes());

        f.run(&[
            b"GEOADD",
            g,
            b"13.361389",
            b"38.115556",
            b"palermo",
            b"15.087269",
            b"37.502669",
            b"catania",
        ]);
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                dst,
                g,
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"KM",
                b"ASC",
            ]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"ZRANGE", dst, b"0", b"-1"]),
            "*2\r\n$7\r\npalermo\r\n$7\r\ncatania\r\n",
            "the geohash is the score, so the order is not the search order"
        );
        assert_eq!(f.run(&[b"ZCARD", g]), ":2\r\n", "the source is untouched");

        // `STOREDIST` stores the distance in the unit the search was asked in,
        // which is the destination stripe's sorted set and not the source's.
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                dst,
                g,
                b"FROMMEMBER",
                b"palermo",
                b"BYRADIUS",
                b"200",
                b"KM",
                b"STOREDIST",
            ]),
            ":2\r\n"
        );
        assert_eq!(
            f.run(&[b"ZSCORE", dst, b"palermo"]),
            "$1\r\n0\r\n",
            "the centre is nought away from itself"
        );

        // A search that found nothing deletes the destination on its own
        // stripe, and a source of the wrong type is refused with the
        // destination left alone.
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                dst,
                g,
                b"FROMLONLAT",
                b"0",
                b"0",
                b"BYRADIUS",
                b"1",
                b"M",
            ]),
            ":0\r\n"
        );
        assert_eq!(f.run(&[b"EXISTS", dst]), ":0\r\n");
        f.run(&[
            b"GEOSEARCHSTORE",
            dst,
            g,
            b"FROMLONLAT",
            b"15",
            b"37",
            b"BYRADIUS",
            b"200",
            b"KM",
        ]);
        f.run(&[b"SET", plain, b"v"]);
        assert_eq!(
            f.run(&[
                b"GEOSEARCHSTORE",
                dst,
                plain,
                b"FROMLONLAT",
                b"15",
                b"37",
                b"BYRADIUS",
                b"200",
                b"KM",
            ]),
            "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n"
        );
        assert_eq!(
            f.run(&[b"ZCARD", dst]),
            ":2\r\n",
            "and left the destination"
        );
    }

    /// Every time series command, on one stripe and on eight.
    ///
    /// Every timestamp is written out rather than left to the clock, so the two
    /// servers are compared on the samples they hold and not on how long the
    /// test took to get from one of them to the other.
    #[test]
    fn the_time_series_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[
                b"TS.CREATE",
                b"ts:a",
                b"LABELS",
                b"sensor",
                b"a",
                b"room",
                b"1",
            ],
            &[b"TS.CREATE", b"ts:a"],
            &[b"TS.ALTER", b"ts:a", b"RETENTION", b"0"],
            &[b"TS.ADD", b"ts:a", b"1000", b"1.5"],
            &[
                b"TS.ADD", b"ts:b", b"1000", b"2", b"LABELS", b"sensor", b"b", b"room", b"1",
            ],
            &[
                b"TS.MADD", b"ts:a", b"2000", b"2.5", b"ts:b", b"2000", b"3", b"gone", b"1", b"1",
            ],
            &[b"TS.INCRBY", b"ts:a", b"1", b"TIMESTAMP", b"3000"],
            &[b"TS.DECRBY", b"ts:a", b"0.5", b"TIMESTAMP", b"4000"],
            &[b"TS.GET", b"ts:a"],
            &[b"TS.GET", b"gone"],
            &[b"TS.RANGE", b"ts:a", b"-", b"+"],
            &[b"TS.RANGE", b"ts:a", b"1000", b"3000", b"COUNT", b"2"],
            &[
                b"TS.RANGE",
                b"ts:a",
                b"-",
                b"+",
                b"AGGREGATION",
                b"avg",
                b"2000",
            ],
            &[b"TS.REVRANGE", b"ts:a", b"-", b"+"],
            &[b"TS.NRANGE", b"2", b"ts:a", b"ts:b", b"-", b"+"],
            &[b"TS.NREVRANGE", b"2", b"ts:a", b"ts:b", b"-", b"+"],
            &[b"TS.NRANGE", b"2", b"ts:a", b"gone", b"-", b"+"],
            &[b"TS.READ", b"ts:a", b"0"],
            &[b"TS.READ", b"ts:a", b"+"],
            // The filters, which are the ones that have to walk every stripe.
            &[b"TS.QUERYINDEX", b"sensor=a"],
            &[b"TS.QUERYINDEX", b"room=1"],
            &[b"TS.QUERYINDEX", b"room=9"],
            &[b"TS.QUERYLABELS", b"LABELS", b"FILTER", b"room=1"],
            &[
                b"TS.QUERYLABELS",
                b"VALUES",
                b"sensor",
                b"FILTER",
                b"room=1",
            ],
            &[b"TS.MGET", b"WITHLABELS", b"FILTER", b"room=1"],
            &[
                b"TS.MGET",
                b"SELECTED_LABELS",
                b"sensor",
                b"FILTER",
                b"sensor=a",
            ],
            &[b"TS.MRANGE", b"-", b"+", b"FILTER", b"room=1"],
            &[
                b"TS.MREVRANGE",
                b"-",
                b"+",
                b"WITHLABELS",
                b"FILTER",
                b"sensor=a",
            ],
            &[
                b"TS.MRANGE",
                b"-",
                b"+",
                b"FILTER",
                b"room=1",
                b"GROUPBY",
                b"room",
                b"REDUCE",
                b"max",
            ],
            &[b"TS.INFO", b"ts:a"],
            // And a rule, which is the one thing here that names two keys.
            &[
                b"TS.CREATERULE",
                b"ts:a",
                b"ts:down",
                b"AGGREGATION",
                b"avg",
                b"1000",
            ],
            &[b"TS.CREATE", b"ts:down"],
            &[
                b"TS.CREATERULE",
                b"ts:a",
                b"ts:down",
                b"AGGREGATION",
                b"avg",
                b"1000",
            ],
            &[b"TS.ADD", b"ts:a", b"5000", b"4"],
            &[b"TS.ADD", b"ts:a", b"6000", b"5"],
            &[b"TS.RANGE", b"ts:down", b"-", b"+"],
            &[b"TS.GET", b"ts:down", b"LATEST"],
            &[b"TS.INFO", b"ts:down"],
            &[b"TS.DEL", b"ts:a", b"5000", b"6000"],
            &[b"TS.RANGE", b"ts:down", b"-", b"+"],
            &[b"TS.DELETERULE", b"ts:a", b"ts:down"],
            &[b"TS.DELETERULE", b"ts:a", b"ts:down"],
            &[b"TS.DEL", b"ts:a", b"0", b"1000"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"TS.ADD", b"plain", b"1", b"1"],
            &[b"TS.GET", b"plain"],
            &[b"TS.READ", b"plain", b"0"],
            &[b"TS.ALTER", b"gone", b"RETENTION", b"0"],
            &[b"TS.RANGE", b"gone", b"-", b"+"],
            &[b"TS.INFO", b"gone"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// A compaction rule whose two ends are on two stripes.
    ///
    /// This is the one thing in the family that walks from a key to another key,
    /// and it walks it in both directions: a sample on the source closes a
    /// bucket on the destination, a `LATEST` read on the destination folds the
    /// bucket the source is still filling, and a delete on the source rewrites
    /// what the destination already held. The same script is run against a
    /// server one stripe wide, where the two keys share a store, and against one
    /// eight stripes wide, where they do not.
    #[test]
    fn a_compaction_rule_across_stripes_reaches_both_ends() {
        let mut many = Fixture::striped(8);
        let other = apart(&mut many, "src");
        let (src, dst) = (b"src".as_slice(), other.as_bytes());
        let mut one = Fixture::new();
        let mut both = |parts: &[&[u8]]| {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
            a
        };

        both(&[b"TS.CREATE", src]);
        both(&[b"TS.CREATE", dst]);
        assert_eq!(
            both(&[b"TS.CREATERULE", src, dst, b"AGGREGATION", b"avg", b"1000"]),
            "+OK\r\n"
        );
        both(&[b"TS.ADD", src, b"1000", b"1"]);
        both(&[b"TS.ADD", src, b"1500", b"3"]);
        // The bucket the source is filling is not written down yet, and asking
        // for it works it out off the source.
        assert_eq!(both(&[b"TS.RANGE", dst, b"-", b"+"]), "*0\r\n");
        let open = both(&[b"TS.GET", dst, b"LATEST"]);
        assert!(open.contains(":1000"), "the open bucket is folded: {open}");

        // A sample past the bucket closes it, which is the write that has to
        // land on the other stripe.
        both(&[b"TS.ADD", src, b"2000", b"5"]);
        let got = both(&[b"TS.RANGE", dst, b"-", b"+"]);
        assert!(got.starts_with("*1\r\n"), "the bucket was written: {got}");
        assert!(got.contains(":1000"), "{got}");

        // And a delete on the source takes it away again.
        both(&[b"TS.DEL", src, b"1000", b"1999"]);
        assert_eq!(both(&[b"TS.RANGE", dst, b"-", b"+"]), "*0\r\n");

        // Both ends still know about each other, and the link comes apart from
        // the source.
        assert!(
            both(&[b"TS.INFO", dst]).contains("src"),
            "the source is named"
        );
        assert_eq!(both(&[b"TS.DELETERULE", src, dst]), "+OK\r\n");
        assert_eq!(
            both(&[b"TS.DELETERULE", src, dst]),
            "-ERR TSDB: compaction rule does not exist\r\n"
        );
    }

    /// A label filter takes the series it names wherever they landed.
    #[test]
    fn a_label_query_across_stripes_finds_every_series() {
        let names: [&[u8]; 6] = [b"q:1", b"q:2", b"q:3", b"q:4", b"q:5", b"q:6"];
        let mut many = Fixture::striped(8);
        let mut homes: Vec<usize> = names
            .iter()
            .map(|name| many.server.striped(0).stripe_of(name))
            .collect();
        homes.sort_unstable();
        homes.dedup();
        assert!(homes.len() > 1, "the six keys are not all on one stripe");

        let mut one = Fixture::new();
        let mut both = |parts: &[&[u8]]| {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
            a
        };
        for name in &names {
            both(&[b"TS.CREATE", name, b"LABELS", b"room", b"1"]);
            both(&[b"TS.ADD", name, b"1000", b"1"]);
        }

        let got = both(&[b"TS.QUERYINDEX", b"room=1"]);
        assert!(got.starts_with("*6\r\n"), "every series answered: {got}");
        assert!(both(&[b"TS.MGET", b"FILTER", b"room=1"]).starts_with("*6\r\n"));
        assert!(both(&[b"TS.MRANGE", b"-", b"+", b"FILTER", b"room=1"]).starts_with("*6\r\n"));
        assert_eq!(
            both(&[b"TS.QUERYLABELS", b"LABELS", b"FILTER", b"room=1"]),
            "*1\r\n$4\r\nroom\r\n"
        );
    }

    /// Every hash command, and the field import beside it, on one stripe and on
    /// eight.
    ///
    /// `HRANDFIELD` with a count draws from the stripe's own generator and two
    /// stripes do not draw the same numbers, so the only draw here is off a hash
    /// holding one field, where every generator gives the same answer.
    #[test]
    fn the_hash_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"HSET", b"h", b"a", b"1", b"b", b"2"],
            &[b"HMSET", b"h", b"c", b"3"],
            &[b"HSETNX", b"h", b"a", b"9"],
            &[b"HSETNX", b"h", b"d", b"4"],
            &[b"HGET", b"h", b"a"],
            &[b"HGET", b"h", b"nope"],
            &[b"HMGET", b"h", b"a", b"nope"],
            &[b"HLEN", b"h"],
            &[b"HEXISTS", b"h", b"a"],
            &[b"HSTRLEN", b"h", b"a"],
            &[b"HGETALL", b"h"],
            &[b"HKEYS", b"h"],
            &[b"HVALS", b"h"],
            &[b"HINCRBY", b"h", b"a", b"5"],
            &[b"HINCRBYFLOAT", b"h", b"a", b"1.5"],
            &[b"HSCAN", b"h", b"0"],
            &[b"HSCAN", b"h", b"0", b"MATCH", b"a", b"COUNT", b"10"],
            &[b"HSCAN", b"h", b"0", b"NOVALUES"],
            &[b"HDEL", b"h", b"d"],
            &[b"HSET", b"one", b"f", b"v"],
            &[b"HRANDFIELD", b"one"],
            &[b"HRANDFIELD", b"one", b"1", b"WITHVALUES"],
            // The field deadlines.
            &[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"a"],
            &[b"HTTL", b"h", b"FIELDS", b"1", b"a"],
            &[b"HPTTL", b"h", b"FIELDS", b"1", b"a"],
            &[b"HEXPIRETIME", b"h", b"FIELDS", b"1", b"a"],
            &[b"HPEXPIRETIME", b"h", b"FIELDS", b"1", b"a"],
            &[b"HPERSIST", b"h", b"FIELDS", b"1", b"a"],
            &[b"HPEXPIREAT", b"h", b"1", b"FIELDS", b"1", b"b"],
            &[b"HGET", b"h", b"b"],
            // The three that came later and word everything their own way.
            &[b"HSETEX", b"h", b"EX", b"100", b"FIELDS", b"1", b"e", b"5"],
            &[b"HGETEX", b"h", b"PERSIST", b"FIELDS", b"1", b"e"],
            &[b"HGETDEL", b"h", b"FIELDS", b"1", b"e"],
            &[b"HGET", b"h", b"e"],
            // And the import, whose key is the third word.
            &[b"HIMPORT", b"PREPARE", b"fs", b"x", b"y"],
            &[b"HIMPORT", b"SET", b"imp", b"fs", b"1", b"2"],
            &[b"HGETALL", b"imp"],
            &[b"HIMPORT", b"SET", b"imp", b"nofs", b"1", b"2"],
            &[b"HIMPORT", b"DISCARD", b"fs"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"HSET", b"plain", b"a", b"1"],
            &[b"HGETALL", b"plain"],
            &[b"HGET", b"gone", b"a"],
            &[b"HINCRBY", b"h", b"a", b"nan"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        // The field deadlines are absolute milliseconds worked out from the
        // clock, so both servers are put on the same one rather than left to
        // read the wall a moment apart.
        one.server.set_clock_ms(1_700_000_000_000);
        many.server.set_clock_ms(1_700_000_000_000);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// Every array command, on one stripe and on eight.
    #[test]
    fn the_array_group_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"ARSET", b"a", b"0", b"x", b"y", b"z"],
            &[b"ARMSET", b"a", b"5", b"p", b"7", b"q"],
            &[b"ARGET", b"a", b"1"],
            &[b"ARGET", b"a", b"99"],
            &[b"ARMGET", b"a", b"0", b"5", b"99"],
            &[b"ARGETRANGE", b"a", b"0", b"7"],
            &[b"ARLEN", b"a"],
            &[b"ARCOUNT", b"a"],
            &[b"ARINSERT", b"a", b"m", b"n"],
            &[b"ARSCAN", b"a", b"0", b"20"],
            &[b"ARSCAN", b"a", b"0", b"20", b"LIMIT", b"2"],
            &[b"ARGREP", b"a", b"0", b"20", b"EXACT", b"x"],
            &[b"ARGREP", b"a", b"0", b"20", b"GLOB", b"*", b"WITHVALUES"],
            &[b"ARLASTITEMS", b"a", b"2"],
            &[b"ARLASTITEMS", b"a", b"2", b"REV"],
            &[b"ARNEXT", b"a"],
            &[b"ARSEEK", b"a", b"3"],
            &[b"AROP", b"a", b"0", b"20", b"USED"],
            &[b"AROP", b"a", b"0", b"20", b"MATCH", b"x"],
            &[b"ARINFO", b"a"],
            &[b"ARINFO", b"a", b"FULL"],
            &[b"ARDEL", b"a", b"0"],
            &[b"ARDELRANGE", b"a", b"1", b"2"],
            &[b"ARCOUNT", b"a"],
            &[b"ARRING", b"r", b"3", b"1", b"2", b"3", b"4"],
            &[b"ARGETRANGE", b"r", b"0", b"9"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"ARGET", b"plain", b"0"],
            &[b"ARSET", b"plain", b"0", b"v"],
            &[b"ARGET", b"gone", b"0"],
            &[b"ARSET", b"a", b"bad", b"v"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// Every graph and vector set command, on one stripe and on eight.
    ///
    /// `VRANDMEMBER` is not in here for the reason `HRANDFIELD` with a count is
    /// not: it draws from the stripe's generator, and the stripes do not share
    /// one.
    #[test]
    fn the_graph_and_vector_groups_answer_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"G.NADD", b"g", b"n1", b"name", b"one"],
            &[b"G.NADD", b"g", b"n2", b"name", b"two"],
            &[b"G.NADD", b"g", b"n3"],
            &[b"G.NGET", b"g", b"n1"],
            &[b"G.NGET", b"g", b"gone"],
            &[b"G.EADD", b"g", b"n1", b"n2", b"knows"],
            &[b"G.EADD", b"g", b"n2", b"n3", b"knows"],
            &[b"G.OUT", b"g", b"n1", b"knows"],
            &[b"G.IN", b"g", b"n2", b"knows"],
            &[b"G.DEG", b"g", b"n1", b"knows"],
            &[b"G.DEG", b"g", b"n2", b"knows", b"BOTH"],
            &[b"G.NEIGH", b"g", b"n1", b"knows", b"DEPTH", b"2"],
            &[b"G.PATH", b"g", b"n1", b"n3"],
            &[b"G.EDEL", b"g", b"n1", b"n2", b"knows"],
            &[b"G.NDEL", b"g", b"n3"],
            &[b"G.NGET", b"g", b"n3"],
            // The vector set, which is one index under one key.
            &[b"VADD", b"v", b"VALUES", b"2", b"1", b"0", b"e1"],
            &[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"e2"],
            &[b"VCARD", b"v"],
            &[b"VDIM", b"v"],
            &[b"VEMB", b"v", b"e1"],
            &[b"VSIM", b"v", b"VALUES", b"2", b"1", b"0"],
            &[b"VSIM", b"v", b"ELE", b"e1"],
            &[b"VISMEMBER", b"v", b"e1"],
            &[b"VISMEMBER", b"v", b"gone"],
            &[b"VSETATTR", b"v", b"e1", b"{\"k\":1}"],
            &[b"VGETATTR", b"v", b"e1"],
            &[b"VRANGE", b"v", b"-", b"+"],
            &[b"VLINKS", b"v", b"e1"],
            &[b"VINFO", b"v"],
            &[b"VREM", b"v", b"e2"],
            &[b"VCARD", b"v"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"G.NGET", b"plain", b"n1"],
            &[b"VCARD", b"plain"],
            &[b"G.NADD", b"gone2", b"n"],
            &[b"VEMB", b"gone3", b"e"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// Every bloom filter, cuckoo filter, count min sketch, top k and t digest
    /// command, on one stripe and on eight.
    #[test]
    fn the_probabilistic_groups_answer_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            // The bloom filter.
            &[b"BF.RESERVE", b"bf", b"0.01", b"100"],
            &[b"BF.ADD", b"bf", b"a"],
            &[b"BF.ADD", b"bf", b"a"],
            &[b"BF.MADD", b"bf", b"b", b"c"],
            &[b"BF.EXISTS", b"bf", b"a"],
            &[b"BF.MEXISTS", b"bf", b"a", b"zz"],
            &[b"BF.CARD", b"bf"],
            &[b"BF.INFO", b"bf"],
            &[b"BF.INFO", b"bf", b"CAPACITY"],
            &[b"BF.DEBUG", b"bf"],
            &[b"BF.INSERT", b"made", b"CAPACITY", b"50", b"ITEMS", b"x"],
            &[b"BF.EXISTS", b"made", b"x"],
            &[b"BF.SCANDUMP", b"bf", b"0"],
            // The cuckoo filter.
            &[b"CF.RESERVE", b"cf", b"100"],
            &[b"CF.ADD", b"cf", b"a"],
            &[b"CF.ADDNX", b"cf", b"a"],
            &[b"CF.COUNT", b"cf", b"a"],
            &[b"CF.EXISTS", b"cf", b"a"],
            &[b"CF.MEXISTS", b"cf", b"a", b"zz"],
            &[b"CF.INSERT", b"cf", b"ITEMS", b"b", b"c"],
            &[b"CF.DEL", b"cf", b"a"],
            &[b"CF.COMPACT", b"cf"],
            &[b"CF.INFO", b"cf"],
            &[b"CF.DEBUG", b"cf"],
            &[b"CF.SCANDUMP", b"cf", b"0"],
            // The count min sketch.
            &[b"CMS.INITBYDIM", b"cms", b"100", b"5"],
            &[b"CMS.INITBYPROB", b"cms2", b"0.01", b"0.01"],
            &[b"CMS.INCRBY", b"cms", b"a", b"5", b"b", b"3"],
            &[b"CMS.QUERY", b"cms", b"a", b"b", b"gone"],
            &[b"CMS.INFO", b"cms"],
            // The top k sketch.
            &[b"TOPK.RESERVE", b"tk", b"3"],
            &[b"TOPK.ADD", b"tk", b"a", b"b", b"a"],
            &[b"TOPK.INCRBY", b"tk", b"c", b"4"],
            &[b"TOPK.QUERY", b"tk", b"a", b"zz"],
            &[b"TOPK.COUNT", b"tk", b"a", b"c"],
            &[b"TOPK.LIST", b"tk"],
            &[b"TOPK.LIST", b"tk", b"WITHCOUNT"],
            &[b"TOPK.INFO", b"tk"],
            // The t digest.
            &[b"TDIGEST.CREATE", b"td"],
            &[b"TDIGEST.ADD", b"td", b"1", b"2", b"3", b"4", b"5"],
            &[b"TDIGEST.MIN", b"td"],
            &[b"TDIGEST.MAX", b"td"],
            &[b"TDIGEST.QUANTILE", b"td", b"0.5"],
            &[b"TDIGEST.CDF", b"td", b"3"],
            &[b"TDIGEST.RANK", b"td", b"3"],
            &[b"TDIGEST.REVRANK", b"td", b"3"],
            &[b"TDIGEST.BYRANK", b"td", b"0"],
            &[b"TDIGEST.BYREVRANK", b"td", b"0"],
            &[b"TDIGEST.TRIMMED_MEAN", b"td", b"0.1", b"0.9"],
            &[b"TDIGEST.INFO", b"td"],
            &[b"TDIGEST.RESET", b"td"],
            &[b"TDIGEST.MIN", b"td"],
            // And the errors.
            &[b"SET", b"plain", b"v"],
            &[b"BF.ADD", b"plain", b"a"],
            &[b"CF.ADD", b"plain", b"a"],
            &[b"CMS.QUERY", b"plain", b"a"],
            &[b"TOPK.ADD", b"plain", b"a"],
            &[b"TDIGEST.ADD", b"plain", b"1"],
            &[b"CMS.INFO", b"gone"],
            &[b"TOPK.INFO", b"gone"],
            &[b"TDIGEST.INFO", b"gone"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// The two sketch merges, with their sources on stripes of their own.
    ///
    /// These are the only two commands in the ten groups that name more than one
    /// key, and both read a run of sources and write a destination, so both go
    /// wrong in the same way if a merge holds one store and looks every source up
    /// in it.
    #[test]
    fn a_sketch_merge_across_stripes_reads_every_source() {
        let mut many = Fixture::striped(8);
        let other = apart(&mut many, "s1");
        let (s1, s2) = (b"s1".as_slice(), other.as_bytes());
        let mut one = Fixture::new();
        let mut both = |parts: &[&[u8]]| {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
            a
        };

        // The count min sketch. The destination has to be the sources' shape,
        // and it is named first, so all three keys are read before anything is
        // written.
        for key in [b"cd".as_slice(), s1, s2] {
            both(&[b"CMS.INITBYDIM", key, b"100", b"5"]);
        }
        both(&[b"CMS.INCRBY", s1, b"x", b"5"]);
        both(&[b"CMS.INCRBY", s2, b"x", b"3"]);
        assert_eq!(
            both(&[b"CMS.MERGE", b"cd", b"2", s1, s2]),
            "+OK\r\n",
            "the merge took both sources"
        );
        assert_eq!(both(&[b"CMS.QUERY", b"cd", b"x"]), "*1\r\n:8\r\n");
        // And with weights, which are read against the sources in order.
        both(&[b"CMS.MERGE", b"cd", b"2", s1, s2, b"WEIGHTS", b"2", b"1"]);
        assert_eq!(both(&[b"CMS.QUERY", b"cd", b"x"]), "*1\r\n:13\r\n");
        // A source that is not a sketch is answered before anything is written.
        both(&[b"SET", b"plain", b"v"]);
        assert!(both(&[b"CMS.MERGE", b"cd", b"2", s1, b"plain"]).starts_with('-'));
        assert_eq!(both(&[b"CMS.QUERY", b"cd", b"x"]), "*1\r\n:13\r\n");

        // The t digest, which builds its destination and then puts it in place.
        // The two source keys are used again here, so what they held goes first.
        both(&[b"FLUSHALL"]);
        both(&[b"TDIGEST.CREATE", b"td"]);
        both(&[b"TDIGEST.CREATE", s1]);
        both(&[b"TDIGEST.CREATE", s2]);
        both(&[b"TDIGEST.ADD", s1, b"1", b"2"]);
        both(&[b"TDIGEST.ADD", s2, b"9", b"10"]);
        assert_eq!(both(&[b"TDIGEST.MERGE", b"td", b"2", s1, s2]), "+OK\r\n");
        assert_eq!(both(&[b"TDIGEST.MIN", b"td"]), "$1\r\n1\r\n");
        assert_eq!(both(&[b"TDIGEST.MAX", b"td"]), "$2\r\n10\r\n");
    }

    /// Every shape of `SORT`, on one stripe and on eight.
    ///
    /// The key it sorts, the keys a `BY` names, the keys a `GET` names and the
    /// destination are four different names and nothing lines them up, so on
    /// eight stripes this script is reading and writing all over the database
    /// while on one it is doing what it always did.
    #[test]
    fn the_sort_command_answers_the_same_however_many_stripes_there_are() {
        let script: &[&[&[u8]]] = &[
            &[b"RPUSH", b"l", b"3", b"1", b"2", b"10"],
            &[b"SORT", b"l"],
            &[b"SORT", b"l", b"DESC"],
            &[b"SORT", b"l", b"ALPHA"],
            &[b"SORT", b"l", b"LIMIT", b"1", b"2"],
            &[b"SORT_RO", b"l"],
            // A weight per element, so the order comes off keys the command
            // never named.
            &[
                b"MSET", b"w_1", b"4", b"w_2", b"3", b"w_3", b"2", b"w_10", b"1",
            ],
            &[b"SORT", b"l", b"BY", b"w_*"],
            &[b"SORT", b"l", b"BY", b"w_*", b"DESC"],
            &[b"DEL", b"w_2"],
            &[b"SORT", b"l", b"BY", b"w_*"],
            // And the answer off another set of keys again, with `#` mixed in
            // so the rows are not all lookups.
            &[b"MSET", b"d_1", b"one", b"d_3", b"three"],
            &[b"SORT", b"l", b"BY", b"w_*", b"GET", b"#", b"GET", b"d_*"],
            // A pattern that reaches into a hash, which is another key again.
            &[b"HSET", b"h_1", b"f", b"9"],
            &[b"HSET", b"h_2", b"f", b"8"],
            &[b"HSET", b"h_3", b"f", b"7"],
            &[b"HSET", b"h_10", b"f", b"6"],
            &[b"SORT", b"l", b"BY", b"h_*->f"],
            &[b"SORT", b"l", b"BY", b"nosort", b"GET", b"h_*->f"],
            // The destination, which is a fourth place to land.
            &[b"SORT", b"l", b"BY", b"w_*", b"STORE", b"out"],
            &[b"LRANGE", b"out", b"0", b"-1"],
            &[b"SORT", b"l", b"STORE", b"l"],
            &[b"LRANGE", b"l", b"0", b"-1"],
            // An empty result takes the destination away rather than leaving a
            // list of nothing behind.
            &[b"SORT", b"missing", b"STORE", b"out"],
            &[b"EXISTS", b"out"],
            // A set and a sorted set sort the same way a list does, and a set
            // written to a destination is sorted even when nothing asked.
            &[b"SADD", b"s", b"c", b"a", b"b"],
            &[b"SORT", b"s", b"ALPHA"],
            &[b"SORT", b"s", b"BY", b"nosort", b"STORE", b"out"],
            &[b"LRANGE", b"out", b"0", b"-1"],
            &[b"ZADD", b"z", b"3", b"c", b"1", b"a", b"2", b"b"],
            &[b"SORT", b"z", b"BY", b"nosort"],
            &[b"SORT", b"z", b"ALPHA", b"DESC"],
            // And the two ways it refuses: a key of the wrong type, and an
            // element that is not a number under a numeric sort.
            &[b"SET", b"str", b"v"],
            &[b"SORT", b"str"],
            &[b"RPUSH", b"words", b"one", b"two"],
            &[b"SORT", b"words"],
            &[b"SORT_RO", b"l", b"STORE", b"out"],
        ];

        let mut one = Fixture::new();
        let mut many = Fixture::striped(8);
        for parts in script {
            let a = one.run(parts);
            let b = many.run(parts);
            assert_eq!(a, b, "{}", String::from_utf8_lossy(parts[0]));
        }
    }

    /// One `SORT` whose four kinds of key are on stripes of their own.
    ///
    /// The script above spreads keys around by writing enough of them, and this
    /// one checks the spread rather than trusting it: the list, the weight key
    /// for one of its elements and the destination are asserted to be in three
    /// places before the command runs.
    #[test]
    fn a_sort_across_stripes_reads_every_pattern_key() {
        let mut f = Fixture::striped(8);
        let out = apart(&mut f, "l");
        let (list, dest) = (b"l".as_slice(), out.as_bytes());

        f.run(&[b"RPUSH", list, b"a", b"b", b"c", b"d"]);
        f.run(&[
            b"MSET", b"w_a", b"4", b"w_b", b"3", b"w_c", b"2", b"w_d", b"1",
        ]);
        f.run(&[
            b"MSET", b"d_a", b"A", b"d_b", b"B", b"d_c", b"C", b"d_d", b"D",
        ]);

        // The weights are four keys and they are not all in one place, which is
        // the thing that would go unnoticed if the command held a stripe.
        let db = f.server.striped(0);
        let weights: Vec<usize> = [b"w_a", b"w_b", b"w_c", b"w_d"]
            .iter()
            .map(|k| db.stripe_of(k.as_slice()))
            .collect();
        assert!(
            weights.iter().any(|s| *s != weights[0]),
            "the four weight keys all landed on one stripe, so this proves nothing"
        );

        assert_eq!(
            f.run(&[b"SORT", list, b"BY", b"w_*", b"GET", b"d_*"]),
            "*4\r\n$1\r\nD\r\n$1\r\nC\r\n$1\r\nB\r\n$1\r\nA\r\n",
            "the order came off the weights and the answer off the data keys"
        );
        assert_eq!(
            f.run(&[b"SORT", list, b"BY", b"w_*", b"STORE", dest]),
            ":4\r\n"
        );
        assert_eq!(
            f.run(&[b"LRANGE", dest, b"0", b"-1"]),
            "*4\r\n$1\r\nd\r\n$1\r\nc\r\n$1\r\nb\r\n$1\r\na\r\n",
            "the destination is on a stripe of its own and got the whole answer"
        );
    }

    /// A `CONFIG SET` reaches every stripe, so where a key landed does not
    /// decide what shape it is stored in.
    ///
    /// This is the setting that would go wrong quietly. A stripe that kept the
    /// old ladder would hold the same hash in a different encoding from the
    /// stripe next to it, and the only thing that would ever say so is
    /// `OBJECT ENCODING`, which is why the check is on that.
    #[test]
    fn a_setting_reaches_every_stripe_and_reads_back_from_any_of_them() {
        let mut f = Fixture::striped(8);
        let other = apart(&mut f, "h");
        let (first, second) = (b"h".as_slice(), other.as_bytes());

        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"hash-max-listpack-entries", b"2"]),
            "+OK\r\n"
        );
        assert_eq!(
            f.run(&[b"CONFIG", b"GET", b"hash-max-listpack-entries"]),
            "*2\r\n$25\r\nhash-max-listpack-entries\r\n$1\r\n2\r\n",
            "the read comes off one stripe and has to answer for all of them"
        );
        for key in [first, second] {
            f.run(&[b"HSET", key, b"a", b"1", b"b", b"2"]);
            assert_eq!(
                f.run(&[b"OBJECT", b"ENCODING", key]),
                "$8\r\nlistpack\r\n",
                "two fields is still under the ladder"
            );
            f.run(&[b"HSET", key, b"c", b"3"]);
            assert_eq!(
                f.run(&[b"OBJECT", b"ENCODING", key]),
                "$9\r\nhashtable\r\n",
                "three fields is over it, on whichever stripe the key is on"
            );
        }

        // And the policy, which every stripe has to agree about for the same
        // reason: an eviction draws from one stripe at a time.
        assert_eq!(
            f.run(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-lru"]),
            "+OK\r\n"
        );
        let db = f.server.striped(0);
        assert!(
            (0..db.width()).all(|i| db.hold_stripe(i).policy().name() == "allkeys-lru"),
            "a stripe kept the old policy"
        );
    }

    /// What an index holds, as the two numbers `FT.INFO` reports about it.
    ///
    /// Read off the registry rather than parsed back out of an `FT.INFO` reply,
    /// because the reply is thirty odd fields and these two are the ones the
    /// keyspace hook moves.
    fn held(f: &Fixture, name: &[u8]) -> (usize, u32) {
        let search = f.server.search.lock();
        let index = search.named(name).expect("the index is there");
        (index.held.docs.len(), index.held.docs.last())
    }

    /// A hash written under an index's prefix reaches it, and one written
    /// outside the prefix does not.
    #[test]
    fn a_hash_that_is_written_reaches_the_index_that_follows_it() {
        let mut f = Fixture::new();
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);
        f.run(&[b"HSET", b"p:1", b"t", b"running dogs"]);
        assert_eq!(held(&f, b"ix"), (1, 1));
        f.run(&[b"HSET", b"other:1", b"t", b"running dogs"]);
        assert_eq!(held(&f, b"ix"), (1, 1));

        // Every field of the key and not the one the command named, since a
        // document is read from nothing every time.
        f.run(&[b"HSET", b"p:1", b"u", b"beta"]);
        f.run(&[b"HDEL", b"p:1", b"u"]);
        assert_eq!(held(&f, b"ix"), (1, 3));
        let search = f.server.search.lock();
        let index = search.named(b"ix").expect("there");
        assert_eq!(index.held.docs.id(b"p:1"), Some(3));
    }

    /// A fresh index reads the keys that were already there, and walks past a
    /// key of the wrong type without counting a failure.
    #[test]
    fn a_fresh_index_reads_the_keys_that_were_already_there() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        f.run(&[b"SET", b"p:str", b"not a hash"]);
        f.run(&[b"HSET", b"q:1", b"t", b"beta"]);
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);

        assert_eq!(held(&f, b"ix"), (1, 1));
        let search = f.server.search.lock();
        let index = search.named(b"ix").expect("there");
        assert_eq!(index.trouble.whole().failures(), 0);
    }

    /// `SKIPINITIALSCAN` leaves what was there alone, and a later write to one
    /// of those keys still lands.
    #[test]
    fn an_index_that_skipped_the_scan_fills_up_on_the_next_write() {
        let mut f = Fixture::new();
        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SKIPINITIALSCAN",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);
        assert_eq!(held(&f, b"ix"), (0, 0));
        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (1, 1));
    }

    /// A command that changed nothing leaves the document where it was, which
    /// is not the same as a command that was not a write.
    ///
    /// All five of these were measured against 8.10.1. Writing the same value
    /// again moves the number and a deadline set for later does not, which is
    /// the pair that makes the rule "the fields are not what they were" rather
    /// than "this was a write".
    #[test]
    fn only_a_real_change_gives_the_document_a_new_number() {
        let mut f = Fixture::new();
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);
        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (1, 1));

        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (1, 2), "the same value still rewrites");

        for quiet in [
            vec![b"HSETNX".as_slice(), b"p:1", b"t", b"other"],
            vec![b"HDEL".as_slice(), b"p:1", b"nosuch"],
            vec![b"HGET".as_slice(), b"p:1", b"t"],
            vec![b"HGETALL".as_slice(), b"p:1"],
            vec![b"HEXPIRE".as_slice(), b"p:1", b"100", b"FIELDS", b"1", b"t"],
            vec![b"HPERSIST".as_slice(), b"p:1", b"FIELDS", b"1", b"t"],
            vec![
                b"HGETEX".as_slice(),
                b"p:1",
                b"EX",
                b"100",
                b"FIELDS",
                b"1",
                b"t",
            ],
            vec![b"HGETDEL".as_slice(), b"p:1", b"FIELDS", b"1", b"nosuch"],
        ] {
            f.run(&quiet);
            assert_eq!(held(&f, b"ix"), (1, 2), "{:?} moved the document", quiet[0]);
        }

        // And the ones that do change something.
        f.run(&[b"HSET", b"p:2", b"n", b"1"]);
        f.run(&[b"HINCRBY", b"p:2", b"n", b"1"]);
        assert_eq!(held(&f, b"ix"), (2, 4));
        // A deadline that has already passed takes the field away, and taking
        // the last field away takes the key and the document with it. The
        // number still moves on the way past, because the field going and the
        // key going are two separate pieces of news and the first of them
        // writes the document one last time.
        f.run(&[b"HEXPIRE", b"p:2", b"0", b"FIELDS", b"1", b"n"]);
        assert_eq!(held(&f, b"ix"), (1, 5));
    }

    /// The two ways of emptying a hash, which do not leave the same thing
    /// behind. `HDEL` of the last field spends no number and is counted as a
    /// refusal, and a deadline that has already passed spends one on a document
    /// nobody sees and is counted as nothing. Measured against 8.10.1 and not
    /// something anyone would guess.
    #[test]
    fn a_key_emptied_by_a_deadline_spends_a_number_and_one_emptied_by_hdel_does_not() {
        /// The index's own failure count.
        fn refused(f: &Fixture, name: &[u8]) -> u64 {
            let search = f.server.search.lock();
            let index = search.named(name).expect("the index is there");
            index.trouble.whole().failures()
        }

        let mut f = Fixture::new();
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);
        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (1, 1));
        f.run(&[b"HDEL", b"p:1", b"t"]);
        assert_eq!(
            held(&f, b"ix"),
            (0, 1),
            "HDEL of the last field spends none"
        );
        assert_eq!(refused(&f, b"ix"), 1, "and is counted as a refusal");

        f.run(&[b"HSET", b"p:2", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (1, 2));
        f.run(&[b"HEXPIRE", b"p:2", b"0", b"FIELDS", b"1", b"t"]);
        assert_eq!(held(&f, b"ix"), (0, 3), "a deadline spends one");
        assert_eq!(refused(&f, b"ix"), 1, "and is counted as nothing");

        f.run(&[b"HSET", b"p:3", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (1, 4));
        f.run(&[b"HGETDEL", b"p:3", b"FIELDS", b"1", b"t"]);
        assert_eq!(held(&f, b"ix"), (0, 5), "and so does HGETDEL");

        // Two fields and one command is one rewrite and not two, whichever way
        // the fields go.
        f.run(&[b"HSET", b"p:4", b"t", b"alpha", b"u", b"beta"]);
        assert_eq!(held(&f, b"ix"), (1, 6));
        f.run(&[b"HEXPIRE", b"p:4", b"0", b"FIELDS", b"2", b"t", b"u"]);
        assert_eq!(held(&f, b"ix"), (0, 7));
        assert_eq!(refused(&f, b"ix"), 1);
    }

    /// `HSETEX` with a deadline that has already passed is two pieces of news
    /// from one command, so the number moves twice and the value never reaches
    /// the index.
    #[test]
    fn a_field_written_already_past_its_deadline_moves_the_number_twice() {
        let mut f = Fixture::new();
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
            b"u",
            b"TEXT",
        ]);
        f.run(&[b"HSET", b"p:1", b"u", b"keepme"]);
        assert_eq!(held(&f, b"ix"), (1, 1));
        f.run(&[
            b"HSETEX", b"p:1", b"EXAT", b"1", b"FIELDS", b"1", b"t", b"zqx",
        ]);
        assert_eq!(
            held(&f, b"ix"),
            (1, 3),
            "the key lived and the field did not"
        );

        // And the same when the key does not survive it.
        f.run(&[b"HSET", b"p:2", b"t", b"alpha"]);
        assert_eq!(held(&f, b"ix"), (2, 4));
        f.run(&[
            b"HSETEX", b"p:2", b"EXAT", b"1", b"FIELDS", b"1", b"t", b"zqx",
        ]);
        assert_eq!(held(&f, b"ix"), (1, 6));
    }

    /// The number one key is indexed under, or `None` when it holds no
    /// document.
    fn number(f: &Fixture, name: &[u8], key: &[u8]) -> Option<u32> {
        let search = f.server.search.lock();
        let index = search.named(name).expect("the index is there");
        index.held.docs.id(key)
    }

    /// An index over `p:` with one document under `p:1`, which is where four of
    /// the tests below start.
    fn indexed() -> Fixture {
        let mut f = Fixture::new();
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);
        f.run(&[b"HSET", b"p:1", b"t", b"alpha"]);
        f
    }

    /// Every way a keyspace command takes a key away leaves no document behind,
    /// and none of them spends a number or is counted as a refusal.
    #[test]
    fn a_key_a_keyspace_command_takes_away_loses_its_document() {
        for take in [
            vec![b"DEL".as_slice(), b"p:1"],
            vec![b"UNLINK".as_slice(), b"p:1"],
            vec![b"PEXPIREAT".as_slice(), b"p:1", b"1"],
            vec![b"EXPIRE".as_slice(), b"p:1", b"-1"],
        ] {
            let mut f = indexed();
            assert_eq!(held(&f, b"ix"), (1, 1));
            f.run(&take);
            assert_eq!(held(&f, b"ix"), (0, 1), "{:?} left something", take[0]);
            let search = f.server.search.lock();
            let index = search.named(b"ix").expect("the index is there");
            assert_eq!(index.trouble.whole().failures(), 0, "{:?}", take[0]);
        }

        // A deadline that has not passed yet is not one of them.
        let mut f = indexed();
        f.run(&[b"EXPIRE", b"p:1", b"1000"]);
        assert_eq!(held(&f, b"ix"), (1, 1));
        f.run(&[b"PERSIST", b"p:1"]);
        assert_eq!(held(&f, b"ix"), (1, 1));
    }

    /// A rename inside the prefix keeps the number the document had, which is
    /// the one write on a followed key that does not spend one. Out of the
    /// prefix is an erase and into it is a fresh reading, both measured.
    #[test]
    fn a_rename_inside_the_prefix_keeps_the_number_the_document_had() {
        let mut f = indexed();
        f.run(&[b"RENAME", b"p:1", b"p:2"]);
        assert_eq!(held(&f, b"ix"), (1, 1), "nothing was read again");
        assert_eq!(number(&f, b"ix", b"p:2"), Some(1));
        assert_eq!(number(&f, b"ix", b"p:1"), None);

        f.run(&[b"RENAME", b"p:2", b"q:1"]);
        assert_eq!(held(&f, b"ix"), (0, 1), "out of the prefix is an erase");

        f.run(&[b"RENAME", b"q:1", b"p:3"]);
        assert_eq!(held(&f, b"ix"), (1, 2), "and into it is a reading");
        assert_eq!(number(&f, b"ix", b"p:3"), Some(2));

        // `RENAMENX` goes the same way, and the one that answers zero changes
        // nothing.
        f.run(&[b"HSET", b"p:4", b"t", b"beta"]);
        assert_eq!(f.run(&[b"RENAMENX", b"p:3", b"p:4"]), ":0\r\n");
        assert_eq!(held(&f, b"ix"), (2, 3));
        f.run(&[b"RENAMENX", b"p:3", b"p:5"]);
        assert_eq!(number(&f, b"ix", b"p:5"), Some(2));
    }

    /// A rename over a key that already had a document leaves one document and
    /// not two. A real server leaves both, and D-64 is that difference.
    #[test]
    fn a_rename_over_a_document_leaves_one_of_them() {
        let mut f = indexed();
        f.run(&[b"HSET", b"p:2", b"t", b"beta"]);
        assert_eq!(held(&f, b"ix"), (2, 2));
        f.run(&[b"RENAME", b"p:1", b"p:2"]);
        assert_eq!(held(&f, b"ix"), (1, 2));
        assert_eq!(number(&f, b"ix", b"p:2"), Some(1));
    }

    /// A key that arrives under the prefix by being copied or restored is read
    /// as a new document, and one that is written over by something that is not
    /// a hash is erased without a word.
    #[test]
    fn a_key_that_arrives_under_the_prefix_is_read_and_one_overwritten_is_erased() {
        let mut f = indexed();
        f.run(&[b"HSET", b"q:1", b"t", b"beta"]);
        f.run(&[b"COPY", b"q:1", b"p:2"]);
        assert_eq!(held(&f, b"ix"), (2, 2));
        assert_eq!(number(&f, b"ix", b"p:2"), Some(2));

        // Out of the prefix, where the source keeps the document it had.
        f.run(&[b"COPY", b"p:1", b"q:2"]);
        assert_eq!(held(&f, b"ix"), (2, 2));

        // Over a key that has one, which is a new reading and not a rename.
        f.run(&[b"COPY", b"q:1", b"p:1", b"REPLACE"]);
        assert_eq!(held(&f, b"ix"), (2, 3));
        assert_eq!(number(&f, b"ix", b"p:1"), Some(3));

        // And a string landing on top of a document takes it away, spending no
        // number and counting no failure.
        f.run(&[b"SET", b"s:1", b"plain"]);
        f.run(&[b"COPY", b"s:1", b"p:1", b"REPLACE"]);
        assert_eq!(held(&f, b"ix"), (1, 3));
        let dump = f.run(&[b"DUMP", b"q:1"]);
        assert!(dump.starts_with('$'), "{dump}");
    }

    /// The keyspace group reads a key back on database zero whatever database
    /// the command ran on, which is measured and is not what the hash commands
    /// do. A `COPY` into another database indexes nothing and takes away
    /// whatever the destination had, and a `RESTORE` anywhere else is invisible.
    #[test]
    fn the_keyspace_group_reads_database_zero_whatever_database_it_ran_on() {
        let mut f = indexed();
        f.run(&[b"HSET", b"p:2", b"t", b"beta"]);
        assert_eq!(held(&f, b"ix"), (2, 2));
        // Into database one, so the indexes look for `p:2` on database zero,
        // find the one that is still there and read it again.
        f.run(&[b"COPY", b"p:1", b"p:2", b"DB", b"1", b"REPLACE"]);
        assert_eq!(held(&f, b"ix"), (2, 3));
        // And with nothing under that name on database zero, the copy leaves
        // the index one document lighter than it found it.
        f.run(&[b"DEL", b"p:2"]);
        assert_eq!(held(&f, b"ix"), (1, 3));
        f.run(&[b"COPY", b"p:1", b"p:2", b"DB", b"1", b"REPLACE"]);
        assert_eq!(held(&f, b"ix"), (1, 3), "the copy landed out of sight");

        // A restore on another database is the same story.
        let dump = f.run(&[b"DUMP", b"p:1"]);
        assert!(dump.starts_with('$'), "{dump}");
        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"HSET", b"q:1", b"t", b"gamma"]);
        f.run(&[b"RENAME", b"q:1", b"p:3"]);
        assert_eq!(held(&f, b"ix"), (1, 3), "and so is a rename");
    }

    /// `MOVE` is not a change at all, because an index follows a key by name
    /// and a write on any database still reaches it.
    #[test]
    fn a_move_leaves_the_document_where_it_is() {
        let mut f = indexed();
        f.run(&[b"MOVE", b"p:1", b"1"]);
        assert_eq!(held(&f, b"ix"), (1, 1), "the key moved and nothing else");
        assert_eq!(number(&f, b"ix", b"p:1"), Some(1));

        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"HSET", b"p:1", b"t", b"beta"]);
        assert_eq!(held(&f, b"ix"), (1, 2), "and a write there still lands");
        f.run(&[b"DEL", b"p:1"]);
        assert_eq!(held(&f, b"ix"), (0, 2));
    }

    /// A flush takes every index with it, whichever database it flushed.
    #[test]
    fn a_flush_drops_the_indexes() {
        for flush in [b"FLUSHALL".as_slice(), b"FLUSHDB"] {
            let mut f = indexed();
            f.run(&[flush]);
            assert!(f.server.search.lock().is_empty(), "{flush:?} kept an index");
            assert_eq!(f.run(&[b"FT._LIST"]), "*0\r\n");
        }

        // Even on a database no index ever read, which is what a real server
        // does and is not what anyone would guess.
        let mut f = indexed();
        f.run(&[b"SELECT", b"9"]);
        f.run(&[b"FLUSHDB"]);
        assert!(f.server.search.lock().is_empty());
    }

    /// A key that will not read is counted against the index and against the
    /// field, and `FT.INFO` says so.
    #[test]
    fn a_hash_that_will_not_read_is_counted_where_ft_info_reports_it() {
        let mut f = Fixture::new();
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"n",
            b"NUMERIC",
        ]);
        f.run(&[b"HSET", b"p:1", b"n", b"notanumber"]);
        assert_eq!(held(&f, b"ix"), (0, 0));

        let reply = f.run(&[b"FT.INFO", b"ix"]);
        assert!(
            reply.contains("SEARCH_NUMERIC_VALUE_INVALID Invalid numeric value: 'notanumber'"),
            "{reply}"
        );
        assert!(reply.contains("hash_indexing_failures"), "{reply}");
    }

    /// An index can only be made on database zero, and the check comes after
    /// the `IFNX` shortcut and before everything else.
    #[test]
    fn an_index_can_only_be_made_on_database_zero() {
        let mut f = Fixture::new();
        f.run(&[b"FT.CREATE", b"ix", b"SCHEMA", b"t", b"TEXT"]);
        f.run(&[b"SELECT", b"1"]);
        let refused = "-Cannot create index on db != 0\r\n";
        assert_eq!(
            f.run(&[b"FT.CREATE", b"jx", b"SCHEMA", b"t", b"TEXT"]),
            refused
        );
        // The name is taken, and it still answers about the database.
        assert_eq!(
            f.run(&[b"FT.CREATE", b"ix", b"SCHEMA", b"t", b"TEXT"]),
            refused
        );
        // And so does one whose arguments are nonsense.
        assert_eq!(
            f.run(&[b"FT.CREATE", b"zz", b"BOGUS", b"SCHEMA", b"t", b"TEXT"]),
            refused
        );
        // `IFNX` over a name that is taken is the one that gets through.
        assert_eq!(
            f.run(&[b"FT._CREATEIFNX", b"ix", b"SCHEMA", b"t", b"TEXT"]),
            "+OK\r\n"
        );
        assert_eq!(f.server.search.lock().len(), 1);
    }

    /// The scan reads the database the create was run on, and after that the
    /// index follows its keys in every database.
    ///
    /// The asymmetry is a real server's, measured, and it is the sort of thing
    /// nobody would arrive at by choosing.
    #[test]
    fn the_scan_is_one_database_and_the_following_is_all_of_them() {
        let mut f = Fixture::new();
        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"HSET", b"p:9", b"t", b"on one"]);
        f.run(&[b"SELECT", b"0"]);
        f.run(&[b"HSET", b"p:0", b"t", b"on zero"]);
        f.run(&[
            b"FT.CREATE",
            b"ix",
            b"PREFIX",
            b"1",
            b"p:",
            b"SCHEMA",
            b"t",
            b"TEXT",
        ]);
        assert_eq!(held(&f, b"ix"), (1, 1), "the scan read database zero only");

        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"HSET", b"p:8", b"t", b"later"]);
        assert_eq!(
            held(&f, b"ix"),
            (2, 2),
            "and then it follows every database"
        );
    }

    /// Four documents over the two kinds of field a query can ask about, which
    /// is the corpus the searches below read.
    fn corpus(f: &mut Fixture) {
        f.run(&[
            b"FT.CREATE",
            b"sx",
            b"PREFIX",
            b"1",
            b"d:",
            b"SCHEMA",
            b"t",
            b"TEXT",
            b"g",
            b"TAG",
            b"n",
            b"NUMERIC",
        ]);
        for (key, text, tag, number) in [
            (b"d:1".as_slice(), "alpha beta", "aa,bb", "1"),
            (b"d:2", "alpha gamma", "bb", "2"),
            (b"d:3", "delta", "cc", "3"),
            (b"d:4", "alpha beta gamma", "aa,cc", "4"),
        ] {
            f.run(&[
                b"HSET",
                key,
                b"t",
                text.as_bytes(),
                b"g",
                tag.as_bytes(),
                b"n",
                number.as_bytes(),
            ]);
        }
    }

    /// A search answers a total and then a row for every key in the window,
    /// with the fields of that key after it.
    #[test]
    fn a_search_answers_a_total_and_then_the_rows() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta"]),
            "*3\r\n:1\r\n$3\r\nd:3\r\n*6\r\n$1\r\nt\r\n$5\r\ndelta\r\n$1\r\ng\r\n$2\r\ncc\r\n$1\r\nn\r\n$1\r\n3\r\n"
        );
        // The fields are what the key holds and not what the schema names, so
        // a field nobody indexed comes back too.
        f.run(&[b"HSET", b"d:3", b"extra", b"more"]);
        assert!(f.run(&[b"FT.SEARCH", b"sx", b"delta"]).contains("extra"));
        // `NOCONTENT` leaves the keys on their own, and `LIMIT 0 0` leaves
        // the total on its own.
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"NOCONTENT"]),
            "*2\r\n:1\r\n$3\r\nd:3\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"LIMIT", b"0", b"0"]),
            "*1\r\n:3\r\n"
        );
    }

    /// The window is ten rows when nobody said, and the cap is on how wide it
    /// is rather than on where it starts.
    #[test]
    fn the_window_is_ten_rows_and_a_million_wide_at_most() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"alpha",
                b"NOCONTENT",
                b"LIMIT",
                b"1",
                b"1"
            ]),
            "*2\r\n:3\r\n$3\r\nd:2\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"LIMIT", b"0"]),
            "-SEARCH_PARSE_ARGS LIMIT requires two arguments\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"LIMIT", b"0", b"-1"]),
            "-SEARCH_PARSE_ARGS LIMIT needs two numeric arguments\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"LIMIT", b"0", b"1000001"]),
            "-SEARCH_LIMIT_OVER LIMIT exceeds maximum of 1000000\r\n"
        );
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"alpha",
                b"NOCONTENT",
                b"LIMIT",
                b"999999",
                b"1000000"
            ]),
            "*1\r\n:3\r\n"
        );
    }

    /// `RETURN 0` reads on the wire like `NOCONTENT` and is not the same
    /// thing, because a later `RETURN` puts the fields back and a later
    /// `RETURN` after a `NOCONTENT` does not.
    #[test]
    fn a_return_of_nothing_is_not_the_same_as_nocontent() {
        let mut f = Fixture::new();
        corpus(&mut f);
        let bare = "*2\r\n:1\r\n$3\r\nd:3\r\n";
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"RETURN", b"0"]),
            bare
        );
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"delta",
                b"NOCONTENT",
                b"RETURN",
                b"1",
                b"t"
            ]),
            bare
        );
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"delta",
                b"RETURN",
                b"0",
                b"RETURN",
                b"1",
                b"t"
            ]),
            "*3\r\n:1\r\n$3\r\nd:3\r\n*2\r\n$1\r\nt\r\n$5\r\ndelta\r\n"
        );
    }

    /// The count after `RETURN` counts words and not fields, so the `AS` and
    /// the name after it are two of them.
    #[test]
    fn the_count_after_return_counts_words() {
        let mut f = Fixture::new();
        corpus(&mut f);
        // Two words is one renamed field, and the name is the one it comes
        // back under.
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"delta",
                b"RETURN",
                b"3",
                b"t",
                b"AS",
                b"x"
            ]),
            "*3\r\n:1\r\n$3\r\nd:3\r\n*2\r\n$1\r\nx\r\n$5\r\ndelta\r\n"
        );
        // A count that stops on the `AS` has nothing to rename to, and one
        // that reaches past the last word is short an argument.
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"RETURN", b"2", b"t", b"AS"]),
            "-SEARCH_PARSE_ARGS RETURN path AS name - must be accompanied with NAME\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"RETURN", b"3", b"t", b"AS"]),
            "-SEARCH_PARSE_ARGS Bad arguments for RETURN: Expected an argument, but none provided\r\n"
        );
        // A count that stops before the `AS` asks for a field called `AS`,
        // which no key holds, and a field the key does not hold is left out
        // rather than sent empty.
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"RETURN", b"1", b"AS"]),
            "*3\r\n:1\r\n$3\r\nd:3\r\n*0\r\n"
        );
    }

    /// A `FILTER` is a numeric range written outside the query, and it is only
    /// the wrong way round on a field the schema holds as a number.
    #[test]
    fn a_filter_is_a_range_written_outside_the_query() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"alpha",
                b"NOCONTENT",
                b"FILTER",
                b"n",
                b"2",
                b"4"
            ]),
            "*3\r\n:2\r\n$3\r\nd:2\r\n$3\r\nd:4\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"FILTER", b"n", b"2"]),
            "-SEARCH_PARSE_ARGS FILTER requires 3 arguments\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"FILTER", b"n", b"x", b"1"]),
            "-SEARCH_PARSE_ARGS Bad lower range: x\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"FILTER", b"n", b"2", b"1"]),
            "-SEARCH_SYNTAX Invalid numeric range (min > max): @n:[2.000000 1.000000]\r\n"
        );
        // The same range on a field that is not a number at all, and on a
        // field that is not there, answers nothing rather than refusing.
        for field in [b"g".as_slice(), b"nope"] {
            assert_eq!(
                f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"FILTER", field, b"2", b"1"]),
                "*1\r\n:0\r\n"
            );
        }
    }

    /// The index is resolved before the arguments after it are read, so a name
    /// that is not there answers about the name whatever else is wrong.
    #[test]
    fn the_index_is_found_before_the_arguments_are_read() {
        let mut f = Fixture::new();
        corpus(&mut f);
        let missing = "-SEARCH_INDEX_NOT_FOUND Index not found: nope\r\n";
        assert_eq!(f.run(&[b"FT.SEARCH", b"nope", b"alpha", b"BOGUS"]), missing);
        assert_eq!(
            f.run(&[b"FT.EXPLAIN", b"nope", b"alpha", b"BOGUS"]),
            missing
        );
        // And the arguments are read before the query is, so a query that
        // will not parse still answers about the argument.
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"@@@", b"BOGUS"]),
            "-SEARCH_ARG_UNRECOGNIZED Unknown argument `BOGUS` at position 1 for <main>\r\n"
        );
    }

    /// `INKEYS` filters the answer before the total is taken, which is not
    /// where a client would guess it happens.
    #[test]
    fn inkeys_comes_off_the_total() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[
                b"FT.SEARCH",
                b"sx",
                b"alpha",
                b"NOCONTENT",
                b"INKEYS",
                b"1",
                b"d:1"
            ]),
            "*2\r\n:1\r\n$3\r\nd:1\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"NOCONTENT", b"INKEYS", b"0"]),
            "*1\r\n:0\r\n"
        );
    }

    /// The fields come from the database the session is on, and a row whose
    /// key will not load there is dropped from the reply and taken off the
    /// total.
    ///
    /// Measured against a real server, which follows a key on every database
    /// and then loads it from one.
    #[test]
    fn the_fields_are_read_from_the_session_database() {
        let mut f = Fixture::new();
        corpus(&mut f);
        f.run(&[b"SELECT", b"1"]);
        f.run(&[b"HSET", b"d:9", b"t", b"delta", b"n", b"9"]);
        // Both documents are in the index, and only one of them is in this
        // database.
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"NOCONTENT"]),
            "*3\r\n:2\r\n$3\r\nd:3\r\n$3\r\nd:9\r\n"
        );
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"RETURN", b"1", b"n"]),
            "*3\r\n:1\r\n$3\r\nd:9\r\n*2\r\n$1\r\nn\r\n$1\r\n9\r\n"
        );
    }

    /// The deeper protocol answers a map of five rather than an array, with
    /// every row a map of its own.
    #[test]
    fn the_third_protocol_answers_a_map_of_five() {
        let mut f = Fixture::new();
        corpus(&mut f);
        f.out = Out::new(Proto::Resp3);
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"delta", b"RETURN", b"1", b"n"]),
            concat!(
                "%5\r\n+attributes\r\n*0\r\n+format\r\n+STRING\r\n+results\r\n*1\r\n",
                "%3\r\n+id\r\n$3\r\nd:3\r\n+extra_attributes\r\n%1\r\n$1\r\nn\r\n$1\r\n3\r\n",
                "+values\r\n*0\r\n+total_results\r\n:1\r\n+warning\r\n*0\r\n"
            )
        );
    }

    /// A window of nothing is a client asking for the count on its own, and a
    /// window of nothing that starts somewhere else is a contradiction all
    /// three commands refuse in the same words.
    #[test]
    fn a_window_of_nothing_has_to_start_at_the_top() {
        let mut f = Fixture::new();
        corpus(&mut f);
        let refused = "-SEARCH_LIMIT_OVER The `offset` of the LIMIT must be 0 when `num` is 0\r\n";
        assert_eq!(
            f.run(&[b"FT.SEARCH", b"sx", b"alpha", b"LIMIT", b"1", b"0"]),
            refused
        );
        assert_eq!(
            f.run(&[b"FT.EXPLAIN", b"sx", b"alpha", b"LIMIT", b"1", b"0"]),
            refused
        );
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LIMIT", b"1", b"0"]),
            refused
        );
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LIMIT", b"0", b"0"]),
            "*1\r\n:3\r\n"
        );
    }

    /// An aggregation answers a count and then a list of properties for every
    /// row, which is empty until something asks for a field.
    #[test]
    fn an_aggregation_answers_a_count_and_then_the_properties() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha"]),
            "*4\r\n:1\r\n*0\r\n*0\r\n*0\r\n"
        );
        // Every row, and not the ten a search would have cut it down to.
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LOAD", b"1", b"@t"]),
            concat!(
                "*4\r\n:3\r\n*2\r\n$1\r\nt\r\n$10\r\nalpha beta\r\n",
                "*2\r\n$1\r\nt\r\n$11\r\nalpha gamma\r\n",
                "*2\r\n$1\r\nt\r\n$16\r\nalpha beta gamma\r\n"
            )
        );
        // Ascending document number, because nothing sorts the answer. The
        // second and fourth documents are the ones the window lands on and the
        // best scoring one is not among them.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"1",
                b"@n",
                b"LIMIT",
                b"1",
                b"2"
            ]),
            "*3\r\n:3\r\n*2\r\n$1\r\nn\r\n$1\r\n2\r\n*2\r\n$1\r\nn\r\n$1\r\n4\r\n"
        );
        // A query nothing answers is a count of nothing and no rows at all.
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"nope", b"LOAD", b"1", b"@t"]),
            "*1\r\n:0\r\n"
        );
    }

    /// `LOAD` counts words rather than fields, names the property after the
    /// path unless an `AS` renames it, and reads everything the key holds when
    /// it is given a star.
    #[test]
    fn a_load_counts_words_and_can_rename_what_it_reads() {
        let mut f = Fixture::new();
        corpus(&mut f);
        // Three words, which are the path, the `AS` and the name.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"3",
                b"@t",
                b"AS",
                b"text"
            ]),
            concat!(
                "*4\r\n:3\r\n*2\r\n$4\r\ntext\r\n$10\r\nalpha beta\r\n",
                "*2\r\n$4\r\ntext\r\n$11\r\nalpha gamma\r\n",
                "*2\r\n$4\r\ntext\r\n$16\r\nalpha beta gamma\r\n"
            )
        );
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"*",
                b"LIMIT",
                b"0",
                b"1"
            ]),
            concat!(
                "*2\r\n:3\r\n*6\r\n$1\r\nt\r\n$10\r\nalpha beta\r\n",
                "$1\r\ng\r\n$5\r\naa,bb\r\n$1\r\nn\r\n$1\r\n1\r\n"
            )
        );
        // A field the key does not hold is left out rather than sent empty.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"2",
                b"@n",
                b"@nope",
                b"LIMIT",
                b"0",
                b"2"
            ]),
            "*3\r\n:3\r\n*2\r\n$1\r\nn\r\n$1\r\n1\r\n*2\r\n$1\r\nn\r\n$1\r\n2\r\n"
        );
    }

    /// The `LOAD` grammar, which has four ways to go wrong and one of them is
    /// only reported once the rest of the argument list has read cleanly.
    #[test]
    fn a_load_refuses_a_count_it_cannot_use() {
        let mut f = Fixture::new();
        corpus(&mut f);
        let head = "-SEARCH_PARSE_ARGS Bad arguments for LOAD: ";
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LOAD", b"x"]),
            format!("{head}Expected number of fields or `*`\r\n")
        );
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LOAD", b"-1", b"@t"]),
            format!("{head}Value is outside acceptable bounds\r\n")
        );
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LOAD", b"5", b"@t"]),
            format!("{head}Expected an argument, but none provided\r\n")
        );
        assert_eq!(
            f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", b"LOAD"]),
            format!("{head}Expected an argument, but none provided\r\n")
        );
        // A count that runs out on the `AS` is held back, because the word
        // after it is read as an argument of its own and may be worth an error
        // of its own. Nothing follows here, so the held back line is the one.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"2",
                b"@t",
                b"AS"
            ]),
            "-SEARCH_PARSE_ARGS LOAD path AS name - must be accompanied with NAME\r\n"
        );
        // And here the word after it is one an aggregation stops taking once a
        // step has been read, so that is what the client hears about.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"2",
                b"@t",
                b"AS",
                b"VERBATIM"
            ]),
            "-SEARCH_ARG_UNRECOGNIZED Unknown argument `VERBATIM` at position 5 for <main>\r\n"
        );
        // A `LOAD 0` is a step that names nothing. It shuts the same door
        // without becoming a loader, so the count stays the one a query with no
        // `LOAD` gets.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"0",
                b"LIMIT",
                b"0",
                b"1"
            ]),
            "*2\r\n:1\r\n*0\r\n"
        );
    }

    /// Reading a step of the pipeline stops the words about the search itself
    /// being taken, and `LIMIT` and `TIMEOUT` are not steps.
    #[test]
    fn a_pipeline_step_closes_the_door_on_the_search_words() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LOAD",
                b"1",
                b"@t",
                b"VERBATIM"
            ]),
            "-SEARCH_ARG_UNRECOGNIZED Unknown argument `VERBATIM` at position 4 for <main>\r\n"
        );
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"LIMIT",
                b"0",
                b"1",
                b"VERBATIM"
            ]),
            "*2\r\n:1\r\n*0\r\n"
        );
        // Three words a search takes that this command names in its refusal
        // rather than calling them unknown.
        for word in [b"RETURN".as_slice(), b"SUMMARIZE", b"HIGHLIGHT"] {
            let name = core::str::from_utf8(word).expect("the three words are text");
            assert_eq!(
                f.run(&[b"FT.AGGREGATE", b"sx", b"alpha", word]),
                format!("-SEARCH_PARSE_ARGS {name} is not supported on FT.AGGREGATE\r\n")
            );
        }
    }

    /// `ADDSCORES` writes the score as a property to twelve significant digits
    /// where `WITHSCORES` writes it beside the row in full.
    #[test]
    fn addscores_writes_a_shorter_score_than_withscores() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"ADDSCORES",
                b"LOAD",
                b"1",
                b"@n",
                b"LIMIT",
                b"0",
                b"2"
            ]),
            concat!(
                "*3\r\n:3\r\n",
                "*4\r\n$7\r\n__score\r\n$14\r\n0.356674943939\r\n$1\r\nn\r\n$1\r\n1\r\n",
                "*4\r\n$7\r\n__score\r\n$14\r\n0.356674943939\r\n$1\r\nn\r\n$1\r\n2\r\n"
            )
        );
        // `NOCONTENT` takes the properties away and leaves whatever was asked
        // for beside them, and a sort key is always null because nothing sorts
        // by one yet.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"NOCONTENT",
                b"WITHSCORES",
                b"LIMIT",
                b"0",
                b"2"
            ]),
            "*3\r\n:1\r\n$18\r\n0.3566749439387324\r\n$18\r\n0.3566749439387324\r\n"
        );
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"WITHSORTKEYS",
                b"LOAD",
                b"1",
                b"@n",
                b"LIMIT",
                b"0",
                b"1"
            ]),
            "*3\r\n:3\r\n$-1\r\n*2\r\n$1\r\nn\r\n$1\r\n1\r\n"
        );
    }

    /// The one scorer that has to see the whole answer first turns the count
    /// into the real total and hands the rows back backwards.
    #[test]
    fn a_normalising_scorer_answers_the_rows_backwards() {
        let mut f = Fixture::new();
        corpus(&mut f);
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"SCORER",
                b"BM25STD.NORM",
                b"ADDSCORES",
                b"LOAD",
                b"1",
                b"@n",
                b"LIMIT",
                b"1",
                b"2"
            ]),
            concat!(
                "*3\r\n:3\r\n",
                "*4\r\n$7\r\n__score\r\n$1\r\n1\r\n$1\r\nn\r\n$1\r\n2\r\n",
                "*4\r\n$7\r\n__score\r\n$1\r\n1\r\n$1\r\nn\r\n$1\r\n1\r\n"
            )
        );
        // Without `ADDSCORES` nothing on the row needs the score, so the rows
        // come back the way every other query answers them.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"SCORER",
                b"BM25STD.NORM",
                b"LOAD",
                b"1",
                b"@n",
                b"LIMIT",
                b"1",
                b"2"
            ]),
            "*3\r\n:3\r\n*2\r\n$1\r\nn\r\n$1\r\n2\r\n*2\r\n$1\r\nn\r\n$1\r\n4\r\n"
        );
    }

    /// The deeper protocol answers the same map of five a search answers, with
    /// the `id` gone because an aggregation is about the properties.
    #[test]
    fn an_aggregation_answers_a_map_of_five_as_well() {
        let mut f = Fixture::new();
        corpus(&mut f);
        f.out = Out::new(Proto::Resp3);
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"ADDSCORES",
                b"WITHSCORES",
                b"WITHSORTKEYS",
                b"LOAD",
                b"1",
                b"@n",
                b"LIMIT",
                b"0",
                b"1"
            ]),
            concat!(
                "%5\r\n+attributes\r\n*0\r\n+format\r\n+STRING\r\n+results\r\n*1\r\n",
                "%4\r\n+score\r\n,0.3566749439387324\r\n+sortkey\r\n_\r\n",
                "+extra_attributes\r\n%2\r\n$7\r\n__score\r\n$14\r\n0.356674943939\r\n",
                "$1\r\nn\r\n$1\r\n1\r\n+values\r\n*0\r\n",
                "+total_results\r\n:3\r\n+warning\r\n*0\r\n"
            )
        );
        // The count is worked out from the rows the reply reached under this
        // protocol, where under RESP2 it is worked out from the first of them.
        assert_eq!(
            f.run(&[
                b"FT.AGGREGATE",
                b"sx",
                b"alpha",
                b"NOCONTENT",
                b"LIMIT",
                b"0",
                b"1"
            ]),
            concat!(
                "%5\r\n+attributes\r\n*0\r\n+format\r\n+STRING\r\n+results\r\n*1\r\n",
                "%1\r\n+values\r\n*0\r\n+total_results\r\n:1\r\n+warning\r\n*0\r\n"
            )
        );
    }
}
