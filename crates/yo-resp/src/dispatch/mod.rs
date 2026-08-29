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
mod keyspace;
mod scripting;
mod server;
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
        "keyspace" => {
            let db = session.db;
            keyspace::execute(&mut server.dbs[db], spec, args, out).map(|()| Flow::Continue)
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
}
