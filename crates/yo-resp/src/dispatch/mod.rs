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
mod bits;
mod blocking;
mod cpu;
mod geo;
mod graph;
mod hashes;
mod hll;
mod json;
mod keyspace;
mod lists;
mod migrate;
mod scan;
mod scripting;
mod server;
mod sets;
mod streams;
mod strings;
pub mod table;
mod vectors;
mod vfilter;
mod zsets;

pub use args::Args;
pub use blocking::{Parked, Waiters};
pub use server::parse_memory;
pub use table::{COMMANDS, Spec, arity_ok, lookup};

use crate::reply::Out;
use yo_common::{Code, Error};
use yo_kv::cold::Blocks;
use yo_kv::{Clock, Keyspace};

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

/// The numbers `INFO` reports that this layer cannot see for itself.
///
/// The reactor owns the sockets, so the reactor is what knows how many clients
/// there are. It writes these directly and nothing here does anything with them
/// except report them.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Connections open right now.
    pub clients: u64,
    /// Connections accepted since the server started.
    pub connections: u64,
    /// Commands run since the server started, which this layer counts itself.
    pub commands: u64,
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

/// A counter per command, indexed the way [`table::index_of`] says.
///
/// A flat array and not a map, because the dispatcher is already holding the
/// spec and the spec's position in the table is two addresses subtracted. That
/// makes the counting a load, an add and a store on a row the previous command
/// of the same name has already pulled into cache.
struct CommandStats(Box<[CommandStat]>);

impl Default for CommandStats {
    fn default() -> CommandStats {
        CommandStats(vec![CommandStat::default(); table::count()].into_boxed_slice())
    }
}

impl CommandStats {
    /// The row for one command.
    fn at(&mut self, spec: &'static Spec) -> &mut CommandStat {
        &mut self.0[table::index_of(spec)]
    }
}

/// Where a database gets its store from, asked by database number.
///
/// `None` means that database cannot have one. The caller owns whatever the
/// stores are cut out of, which for `yodb` is one `.yo` file with a log per
/// database, and this crate never learns what any of that is.
pub type StoreSource = dyn FnMut(usize) -> Option<Box<dyn Blocks>>;

/// Everything a server holds.
///
/// One of these per shard thread, not one per process: the databases inside are
/// not `Sync` and are reached by sending their thread a command. What makes
/// this a server rather than a shard is that it is the whole of what a
/// connection can address.
pub struct Server {
    dbs: Vec<Keyspace>,
    clock: Clock,
    started_ms: u64,
    /// Where the next maintenance turn starts looking, so that a database
    /// under constant write load cannot hold the other fifteen's space.
    next_db: usize,
    /// One bit per database, set when a command ran against it.
    ///
    /// The maintenance turn after every batch used to ask all sixteen
    /// databases whether they had anything to collect, and asking costs a load
    /// and a store in each one. Fifteen of those are cold lines on a server
    /// where every client is on database zero, which is every server, and the
    /// answer is no every time. This is the cheap half of the question: a
    /// database nobody has touched since it last said no cannot have started
    /// saying yes.
    dirty: u64,
    /// What the connections are holding, kept by the engine.
    conn_bytes: usize,
    /// The `maxmemory` limit in bytes, zero when there is not one.
    ///
    /// Zero is the default and it is the whole reason the check in front of
    /// every write is one comparison against a field that is already warm.
    maxmemory: u64,
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
    store: Option<Box<StoreSource>>,
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
    maxstore: Option<u64>,
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
    used: usize,
    /// Which database the next eviction draws from.
    ///
    /// Its own cursor and not [`Server::next_db`], because eviction and
    /// compaction move at different rates and sharing one would make the
    /// database that gets compacted depend on how many keys were evicted.
    evict_db: usize,
    /// Which database the next active expiry sweep starts at.
    ///
    /// A third cursor for the same reason there is a second one. A sweep runs on
    /// every turn of the loop and compaction runs when there is dead space, so
    /// sharing a cursor would make which database gets swept depend on which one
    /// was last collected.
    expire_db: usize,
    /// The millisecond the last active expiry sweep ran on, so the next one on
    /// the same millisecond does not bother.
    expire_ms: u64,
    /// Clients parked on a blocking command.
    waiters: Waiters,
    /// Sockets `MIGRATE` is holding open to the servers it has talked to.
    ///
    /// Empty on a server nobody has migrated a key out of, which is nearly all
    /// of them, and it costs a vector's three words to be empty.
    peers: migrate::Peers,
    /// The numbers the reactor keeps for `INFO`.
    pub stats: Stats,
    /// A counter per command, for `INFO commandstats`.
    cmdstats: CommandStats,
    /// Set by `SHUTDOWN`, and read by whatever is turning the loop.
    ///
    /// A flag rather than an exit, because the command layer is not what owns
    /// the process. It runs inside a batch that has other commands behind it
    /// and inside a driver that has a socket file to take away and a file to
    /// close, and a server that calls `exit` from a command handler skips all
    /// of that. So the command says stop and the driver stops, on the same turn
    /// and through the same door a signal uses.
    stopping: bool,
}

impl Server {
    /// A server with [`DATABASES`] empty databases on the system clock.
    #[must_use]
    pub fn new() -> Server {
        let clock = Clock::system();
        Server {
            dbs: (0..DATABASES)
                .map(|_| Keyspace::with_clock(clock))
                .collect(),
            clock,
            started_ms: clock.now_ms(),
            next_db: 0,
            dirty: ALL_DATABASES,
            conn_bytes: 0,
            maxmemory: 0,
            store: None,
            maxstore: None,
            used: 0,
            evict_db: 0,
            expire_db: 0,
            expire_ms: 0,
            waiters: Waiters::default(),
            peers: migrate::Peers::default(),
            stats: Stats::default(),
            cmdstats: CommandStats::default(),
            stopping: false,
        }
    }

    /// A server on a clock the caller moves by hand, for tests.
    #[must_use]
    pub fn with_clock(clock: Clock) -> Server {
        Server {
            dbs: (0..DATABASES)
                .map(|_| Keyspace::with_clock(clock))
                .collect(),
            clock,
            started_ms: clock.now_ms(),
            next_db: 0,
            dirty: ALL_DATABASES,
            conn_bytes: 0,
            maxmemory: 0,
            store: None,
            maxstore: None,
            used: 0,
            evict_db: 0,
            expire_db: 0,
            expire_ms: 0,
            waiters: Waiters::default(),
            peers: migrate::Peers::default(),
            stats: Stats::default(),
            cmdstats: CommandStats::default(),
            stopping: false,
        }
    }

    /// One database, by index.
    ///
    /// # Panics
    ///
    /// If `i` is not a database. `SELECT` is the only way a client changes the
    /// index and it checks, so an index that is out of range here is a bug in
    /// the caller and not something a client can ask for.
    pub fn db(&mut self, i: usize) -> &mut Keyspace {
        // The borrow is mutable, so assume it is used. Anything that only reads
        // has [`Server::db_ref`] and does not come through here.
        self.dirty |= 1u64 << i;
        &mut self.dbs[i]
    }

    /// Ask for the server to stop, which is what `SHUTDOWN` does.
    ///
    /// It sets a flag and returns. Nothing here closes a socket, flushes a file
    /// or ends the process, because none of those belong to this layer, and a
    /// batch that is halfway through still has to finish and be written out.
    pub fn stop(&mut self) {
        self.stopping = true;
    }

    /// Whether somebody has asked the server to stop.
    ///
    /// Read once per turn by the loop, next to the flag a signal sets. The two
    /// mean the same thing and are separate only because one arrives from the
    /// operating system and the other from a client.
    #[must_use]
    pub fn stopping(&self) -> bool {
        self.stopping
    }

    /// One database, by index, without taking it mutably.
    ///
    /// What the prefetch stage needs. It runs for all 64 commands in a batch
    /// before any of them executes, so it cannot hold the mutable borrow `run`
    /// is about to want, and it does not need one: warming a cache line reads
    /// nothing and changes nothing.
    ///
    /// # Panics
    ///
    /// As [`Server::db`].
    #[must_use]
    pub fn db_ref(&self, i: usize) -> &Keyspace {
        &self.dbs[i]
    }

    /// Take a new clock reading and give it to every database.
    ///
    /// Once per turn of the event loop, which is the only place time moves. A
    /// command asking what the time is gets the answer the whole batch got, so
    /// two keys written by the same batch expire together (`04` section 3).
    pub fn refresh_clock(&mut self) {
        self.clock.refresh();
        let now = self.clock.now_ms();
        for db in &mut self.dbs {
            db.clock_mut().set(now);
        }
    }

    /// Move every clock here to `ms` by hand, for tests about expiry.
    ///
    /// A test cannot wait a hundred seconds and a test that waits a hundred
    /// milliseconds is a test that fails on a loaded machine, so time moves on
    /// request. The system clock underneath will overwrite this on the next
    /// [`Server::refresh_clock`], which is why this is only useful in a test
    /// that drives commands directly rather than through the event loop.
    pub fn set_clock_ms(&mut self, ms: u64) {
        self.clock.set(ms);
        for db in &mut self.dbs {
            db.clock_mut().set(ms);
        }
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
        self.dbs.iter().map(Keyspace::memory_bytes).sum::<usize>() + self.conn_bytes
    }

    /// What the keyspace itself is holding, live records only.
    ///
    /// `used_memory` minus this is what the store costs to run: the index, the
    /// space dead records are sitting in until compaction gets to them, and the
    /// connections' buffers.
    #[must_use]
    pub fn dataset_bytes(&self) -> usize {
        self.dbs
            .iter()
            .map(|db| db.map().arena().live_bytes() as usize)
            .sum()
    }

    /// Bytes the arenas are holding, live and dead together.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.dbs
            .iter()
            .map(|db| db.map().arena().reserved_bytes() as usize)
            .sum()
    }

    /// Bytes the indexes are holding.
    #[must_use]
    pub fn index_bytes(&self) -> usize {
        self.dbs
            .iter()
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
        self.dbs.iter().map(|db| db.map().compaction()).fold(
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
        self.dbs
            .iter()
            .map(|db| db.map().arena().resident_segments())
            .sum()
    }

    /// What the connections' read and reply buffers are holding.
    #[must_use]
    pub const fn conn_bytes(&self) -> usize {
        self.conn_bytes
    }

    /// Note that the connections are holding `delta` bytes more than they were,
    /// or fewer when it is negative.
    ///
    /// A delta and not a total because the alternative is a walk over every
    /// connection, and the walk would have to happen on a turn of the loop
    /// rather than when `INFO` asks, which puts the cost of a report on the
    /// command path of a server nobody is asking.
    pub fn note_conn_bytes(&mut self, delta: isize) {
        self.conn_bytes = self.conn_bytes.saturating_add_signed(delta);
    }

    /// Keys reclaimed by running into them after their deadline.
    #[must_use]
    pub fn expired_keys(&self) -> u64 {
        self.dbs.iter().map(Keyspace::expired_keys).sum()
    }

    /// Keys thrown away to make room, which is the other number entirely.
    #[must_use]
    pub fn evicted_keys(&self) -> u64 {
        self.dbs.iter().map(Keyspace::evicted_keys).sum()
    }

    /// Every command that has been seen, with its counters.
    ///
    /// Only the ones that have. A server reports a handful of lines rather than
    /// one per command in the table, which is what Redis does and is the
    /// difference between a section a person can read and one they cannot.
    pub fn command_stats(&self) -> impl Iterator<Item = (&'static str, CommandStat)> {
        self.cmdstats
            .0
            .iter()
            .enumerate()
            .filter(|(_, row)| row.seen())
            .map(|(at, row)| (table::name_at(at), *row))
    }

    /// The `maxmemory` limit in bytes, zero when there is not one.
    #[must_use]
    pub const fn maxmemory(&self) -> u64 {
        self.maxmemory
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
    pub fn set_maxmemory(&mut self, bytes: u64) {
        self.maxmemory = bytes;
        for db in &mut self.dbs {
            db.track_memory(bytes != 0);
        }
        self.used = self.settled_memory();
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
        source: impl FnMut(usize) -> Option<Box<dyn Blocks>> + 'static,
    ) {
        self.store = Some(Box::new(source));
    }

    /// Whether this server has been given somewhere to put cold values.
    #[must_use]
    pub const fn has_store_source(&self) -> bool {
        self.store.is_some()
    }

    /// Open database `at`'s store, if it has not got one and there is one to be
    /// had.
    ///
    /// A store that will not open leaves the database where it was, which is
    /// evicting, because a memory limit that cannot be answered by moving data
    /// still has to be answered.
    fn attach_store(&mut self, at: usize) {
        if self.dbs[at].store_bytes().is_some() {
            return;
        }
        let Some(source) = self.store.as_mut() else {
            return;
        };
        if let Some(blocks) = source(at) {
            self.dbs[at].attach(blocks);
        }
    }

    /// The `maxstore` limit in bytes, `None` when there is not one.
    #[must_use]
    pub const fn maxstore(&self) -> Option<u64> {
        self.maxstore
    }

    /// Set the storage limit, or clear it with `None`.
    ///
    /// Nothing is read here the way [`Server::set_maxmemory`] reads the memory
    /// total, because this limit is compared against a number the store keeps
    /// and answers on demand, not against a walk.
    pub const fn set_maxstore(&mut self, bytes: Option<u64>) {
        self.maxstore = bytes;
    }

    /// What every attached store is holding, for `INFO memory`.
    ///
    /// Zero on a server with nothing attached, which is not the same as a server
    /// whose file is empty, and [`Server::regime`] is the field that tells those
    /// two apart.
    #[must_use]
    pub fn store_bytes(&self) -> u64 {
        self.dbs.iter().filter_map(Keyspace::store_bytes).sum()
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
        for db in &self.dbs {
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
        if (0..self.dbs.len()).any(|at| self.migrates(at)) {
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
        if self.maxstore == Some(0) {
            return false;
        }
        match self.dbs[at].store_bytes() {
            Some(held) => self.maxstore.is_none_or(|cap| held < cap),
            // Nothing attached, but somewhere to get one from the moment this
            // database needs it, which is what makes the answer yes rather than
            // no. Opening it here would mean `INFO` opened files.
            None => self.store.is_some(),
        }
    }

    /// Take a fresh memory reading, which the maintenance turn does once a batch.
    ///
    /// Nothing at all when there is no limit, which is the default and is every
    /// server that has not asked for one.
    pub fn refresh_memory(&mut self) {
        if self.maxmemory != 0 {
            self.used = self.settled_memory();
        }
    }

    /// [`Server::memory_bytes`], asked the cheap way.
    ///
    /// The same number. The difference is that this asks each database only
    /// about the collections that could have moved since the last time, which is
    /// what a batch touched rather than what the server holds, so it can be
    /// asked once a batch and again on every command that is over the limit.
    fn settled_memory(&mut self) -> usize {
        self.dbs
            .iter_mut()
            .map(Keyspace::settled_memory_bytes)
            .sum::<usize>()
            + self.conn_bytes
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
    pub fn make_room(&mut self) -> bool {
        if self.maxmemory == 0 || self.used as u64 <= self.maxmemory {
            return true;
        }
        // The cached reading is a batch old and the batch may have compacted
        // since, so take a fresh one before throwing anything away. It is the
        // settled reading and not the walk, so what this costs is the handful of
        // collections the last batch touched and not the whole database.
        self.used = self.settled_memory();
        let mut budget = EVICT_BUDGET;
        while self.used as u64 > self.maxmemory {
            let over = self.used - self.maxmemory as usize;
            if !self.relieve_step(over) {
                return false;
            }
            self.compact_hard_step();
            self.used = self.settled_memory();
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
    fn relieve_step(&mut self, over: usize) -> bool {
        for turn in 0..self.dbs.len() {
            let i = (self.evict_db + turn) % self.dbs.len();
            // An empty database has nothing to move and opening a log for one
            // would cost a resident page window to find that out.
            let gave = if !self.dbs[i].is_empty() && self.migrates(i) {
                self.attach_store(i);
                // Whether it made room and not whether it moved a key. A round
                // that demoted nothing and handed back a segment is a round
                // that made room, and reading only the count refuses the write
                // that provoked it.
                self.dbs[i]
                    .relieve(over)
                    .is_ok_and(yo_kv::tier::Relief::made_room)
            } else {
                self.dbs[i].evict_one()
            };
            if gave {
                self.evict_db = (i + 1) % self.dbs.len();
                self.dirty |= 1u64 << i;
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
    pub fn expire_slice(&mut self, budget: usize) -> usize {
        let now = self.clock.now_ms();
        if now == self.expire_ms {
            return 0;
        }
        self.expire_ms = now;
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
    pub fn expire_step(&mut self, budget: usize) -> usize {
        let mut spent = 0;
        for turn in 0..self.dbs.len() {
            if spent >= budget {
                break;
            }
            let i = (self.expire_db + turn) % self.dbs.len();
            let c = self.dbs[i].expire_cycle(budget - spent);
            spent += c.examined;
            if c.expired > 0 {
                self.expire_db = (i + 1) % self.dbs.len();
                self.dirty |= 1u64 << i;
            }
        }
        spent
    }

    /// One slice of compaction for a server that is over its limit.
    ///
    /// Takes the databases in the same order [`Server::compact_step`] does and
    /// stops at the first one that had something to move, and it asks with the
    /// ratios off. See [`Keyspace::compact_hard`] for what that changes.
    fn compact_hard_step(&mut self) -> Option<usize> {
        for turn in 0..self.dbs.len() {
            let i = (self.next_db + turn) % self.dbs.len();
            if let Some(moved) = self.dbs[i].compact_hard() {
                self.next_db = (i + 1) % self.dbs.len();
                return Some(moved);
            }
        }
        None
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
    pub fn compact_step(&mut self) -> Option<usize> {
        for turn in 0..self.dbs.len() {
            let i = (self.next_db + turn) % self.dbs.len();
            // Nothing has run against this database since it last said it had
            // nothing to collect, so it still has nothing to collect and the
            // line it lives on stays where it is.
            if self.dirty & (1 << i) == 0 {
                continue;
            }
            if let Some(moved) = self.dbs[i].compact_step() {
                self.next_db = (i + 1) % self.dbs.len();
                return Some(moved);
            }
            self.dirty &= !(1u64 << i);
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
}

impl Session {
    /// A new connection, on database zero with no name.
    #[must_use]
    pub fn new(id: u64) -> Session {
        Session {
            db: 0,
            id,
            name: Vec::new(),
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
pub fn execute(server: &mut Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Flow {
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
    server: &mut Server,
    session: &mut Session,
    spec: Option<&'static Spec>,
    args: Args<'_>,
    out: &mut Out,
) -> Flow {
    if args.is_empty() {
        return Flow::Continue;
    }
    server.stats.commands += 1;

    let Some(spec) = spec else {
        write_error(out, &args::unknown_command(args));
        return Flow::Continue;
    };
    if !arity_ok(spec, args.len()) {
        server.cmdstats.at(spec).rejected += 1;
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
    if server.maxmemory != 0 && !server.make_room() && spec.flags.contains(&"denyoom") {
        server.cmdstats.at(spec).rejected += 1;
        out.error_line(b"OOM ", OOM);
        return Flow::Continue;
    }

    // Which databases the maintenance turn after this batch has to ask. Marked
    // for every command and not only for the writes, because a read can make
    // garbage too: a `GET` on a key whose expiry has passed reaps it, and the
    // record it dropped is exactly the kind of thing the collector is for.
    // `COPY`, `SWAPDB` and `FLUSHALL` reach a database nobody selected, so the
    // two groups that hold them mark all of them rather than the session's.
    server.dirty |= match spec.group {
        "string" | "bitmap" | "hyperloglog" | "geo" | "set" | "hash" | "list" | "zset"
        | "array" | "stream" => 1u64 << session.db,
        _ => ALL_DATABASES,
    };

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
                strings::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // Its own group and its own file, and the same values underneath:
            // a bitmap is a string, so `STRLEN` on one answers and `SETBIT` on
            // something a `SET` left behind works.
            "bitmap" => {
                let db = session.db;
                bits::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // The same again: a sketch is a string with a documented layout, so
            // `GET` hands one to a client and `SET` takes it back.
            "hyperloglog" => {
                let db = session.db;
                hll::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "set" => {
                let db = session.db;
                sets::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "hash" => {
                let db = session.db;
                hashes::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "list" => {
                let db = session.db;
                lists::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "zset" => {
                let db = session.db;
                zsets::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // A geo key is a sorted set and these are sorted set commands with
            // arithmetic on the way in and on the way out, so a client can ZREM
            // a place out of one and ZCARD it to count them.
            "geo" => {
                let db = session.db;
                geo::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "array" => {
                let db = session.db;
                arrays::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "graph" => {
                let db = session.db;
                graph::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // A document under a key, reached by a path. The group is Redis's
            // module surface and the storage is ours, the same trade the vector
            // set group makes.
            "json" => {
                let db = session.db;
                json::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            "vector" => {
                let db = session.db;
                vectors::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
            }
            // The clock is read before the database is borrowed, because every
            // stream command needs the time and it lives on the server. An
            // `XADD` with no ID, an `XCLAIM` working out what is idle and an
            // `XINFO` reporting it all have to agree about what moment this is.
            "stream" => {
                let db = session.db;
                let now = server.now_ms();
                streams::execute(&mut server.dbs[db], spec, args, now, out).map(|()| Flow::Continue)
            }
            // The one keyspace command that needs more than the databases,
            // because the socket it talks down is held on the server between
            // commands and not opened again for each one.
            "keyspace" if spec.name == "migrate" => {
                migrate::execute(server, session.db, args, out).map(|()| Flow::Continue)
            }
            // Every database and not the one the session is on, because `COPY` takes
            // a `DB n` and writes into a database nobody selected.
            "keyspace" => keyspace::execute(&mut server.dbs, session.db, spec, args, out)
                .map(|()| Flow::Continue),
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
    let row = server.cmdstats.at(spec);
    row.calls += 1;
    if matches!(out.as_slice().get(mark), Some(b'-' | b'!')) {
        row.failed += 1;
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
            Fixture {
                server: Server::new(),
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
                &mut self.server,
                &mut self.session,
                Args::new(&self.argv, &wire),
                &mut self.out,
            );
            self.out.as_slice().to_vec()
        }

        /// Move every clock in the server on by `ms`.
        fn advance(&mut self, ms: u64) {
            for db in 0..DATABASES {
                self.server.db(db).clock_mut().advance(ms);
            }
        }

        /// The same, with what the connection should do next.
        fn flow(&mut self, parts: &[&[u8]]) -> (Flow, String) {
            let wire = encode(parts);
            self.argv.decode(&wire, &Limits::default()).unwrap();
            self.out.clear();
            let flow = execute(
                &mut self.server,
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
        assert_eq!(
            f.server.dirty & (1 << 9),
            0,
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
        f.server.db(0).clock_mut().advance(2);
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
        f.server.db(0).clock_mut().advance(20);
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
        let at = f.server.db(0).clock().now_ms();
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
            f.server.db(0).expires() > 400,
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

    #[test]
    fn the_command_counter_counts_every_command_including_the_bad_ones() {
        let mut f = Fixture::new();
        f.run(&[b"PING"]);
        f.run(&[b"NOPE"]);
        f.run(&[b"GET"]);
        assert_eq!(f.server.stats.commands, 3);
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
        f.server.db(0).clock_mut().advance(60);
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

        // And the one that does not: an empty source means the destination is
        // never looked at, so this waits rather than erroring, and on a real
        // server it times out.
        assert_eq!(
            f.flow(&[b"BLMOVE", b"E", b"S", b"LEFT", b"RIGHT", b"0.1"])
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
        assert_eq!(f.server.waiters().len(), 1);
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
        // A vector of zeros has no direction, so a cosine set has nowhere to
        // put it.
        assert_eq!(
            f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"0", b"nowhere"]),
            "-ERR a cosine collection compares directions and a vector of length zero has none\r\n"
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
        // The set stored the unit vector and the length is multiplied back on
        // the way out, so `3 4` and not `0.6 0.8`.
        assert_eq!(
            f.run(&[b"VEMB", b"v", b"a"]),
            "*2\r\n$1\r\n3\r\n$1\r\n4\r\n"
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
        // The quantisation is recorded and not applied, which is D-32, so it
        // reports back what was sent.
        assert!(
            info.contains("$10\r\nquant-type\r\n$3\r\nf32\r\n"),
            "{info}"
        );
        f.run(&[b"VADD", b"v", b"VALUES", b"2", b"0", b"1", b"north", b"BIN"]);
        assert!(
            f.run(&[b"VINFO", b"v"])
                .contains("$10\r\nquant-type\r\n$3\r\nbin\r\n")
        );
        assert_eq!(f.run(&[b"VINFO", b"nokey"]), "$-1\r\n");
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
            "*2\r\n$1\r\n3\r\n$1\r\n4\r\n"
        );
        // RAW is the stored form and the number that turns it back into the
        // client's, which is the unit vector and the length it arrived with.
        let raw = f.run(&[b"VEMB", b"v", b"a", b"RAW"]);
        assert!(raw.starts_with("*3\r\n$3\r\nf32\r\n$8\r\n"), "{raw}");
        assert!(raw.ends_with("$1\r\n5\r\n"), "{raw}");
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
        assert_eq!(f.server.waiters().len(), 2);

        f.run(&[b"XADD", b"s", b"2-1", b"a", b"2"]);
        let want = "*1\r\n*2\r\n$1\r\ns\r\n*1\r\n*2\r\n$3\r\n2-1\r\n*2\r\n$1\r\na\r\n$1\r\n2\r\n";
        for at in 0..2 {
            let mut out = Out::new(Proto::Resp2);
            assert!(f.server.serve_waiter(at, 0, &mut out));
            assert_eq!(core::str::from_utf8(out.as_slice()).expect("ascii"), want);
        }

        // And a deadline that runs out is a null array, the same as a plain
        // XREAD that found nothing.
        f.server.waiters_mut().forget(7);
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
            f.server.db(0).attach(Box::new(Mem { blobs: Vec::new() }));
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
}
