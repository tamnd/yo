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
mod cpu;
mod hashes;
mod keyspace;
mod scripting;
mod server;
mod sets;
mod strings;
pub mod table;

pub use args::Args;
pub use table::{COMMANDS, Spec, arity_ok, lookup};

use crate::reply::Out;
use yo_common::{Code, Error};
use yo_kv::{Clock, Keyspace};

/// How many databases a server has.
///
/// Redis's default is sixteen and its `databases` setting can change it. Ours
/// is sixteen and cannot, which is why `CONFIG GET databases` can answer with a
/// constant. Nothing in the design needs the number to be fixed; nothing yet
/// needs it not to be.
pub const DATABASES: usize = 16;

/// What the connection should do after a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Read the next command.
    Continue,
    /// Write what is buffered and then close, which is what `QUIT` asks for.
    Close,
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
    /// What the connections are holding, kept by the engine.
    conn_bytes: usize,
    /// The numbers the reactor keeps for `INFO`.
    pub stats: Stats,
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
            conn_bytes: 0,
            stats: Stats::default(),
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
            conn_bytes: 0,
            stats: Stats::default(),
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
        &mut self.dbs[i]
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
            if let Some(moved) = self.dbs[i].compact_step() {
                self.next_db = (i + 1) % self.dbs.len();
                return Some(moved);
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
    server.stats.commands += 1;

    let Some(spec) = lookup(args.name()) else {
        write_error(out, &args::unknown_command(args));
        return Flow::Continue;
    };
    if !arity_ok(spec, args.len()) {
        write_error(out, &args::wrong_arity(spec.name));
        return Flow::Continue;
    }

    let mark = out.len();
    let done = match spec.group {
        "string" => {
            let db = session.db;
            strings::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
        }
        "set" => {
            let db = session.db;
            sets::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
        }
        "hash" => {
            let db = session.db;
            hashes::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
        }
        // Every database and not the one the session is on, because `COPY` takes
        // a `DB n` and writes into a database nobody selected.
        "keyspace" => {
            keyspace::execute(&mut server.dbs, session.db, spec, args, out).map(|()| Flow::Continue)
        }
        "scripting" => scripting::execute(spec, args, out).map(|()| Flow::Continue),
        _ => server::execute(server, session, spec, args, out),
    };
    match done {
        Ok(flow) => flow,
        Err(e) => {
            out.truncate(mark);
            write_error(out, &e);
            Flow::Continue
        }
    }
}

/// The error line for an error value.
///
/// The prefix is what a client branches on, and there are only two of them in
/// this milestone: `WRONGTYPE` for a command sent at the wrong kind of value,
/// and `ERR` for everything else. The two errors that need a third, `NOPROTO`
/// and `WRONGPASS`, are written by `HELLO` itself.
fn write_error(out: &mut Out, e: &Error) {
    let prefix: &[u8] = match e.code() {
        Code::WrongType => b"WRONGTYPE ",
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
        assert_eq!(f.run(&[b"DBSIZE"]), ":2\r\n", "nothing has collected it yet");

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
        assert!(both.starts_with("*4\r\n"), "{both}");
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
}
