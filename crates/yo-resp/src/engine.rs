//! Connections, framing and buffers: the seam between the loop and the
//! commands.
//!
//! `yo-reactor` knows how to run a batch and nothing about what a command is.
//! `dispatch` knows how to run a command and nothing about where the bytes came
//! from. This module is the piece in between, and it is the piece a server is
//! missing until it exists: the read buffer a command's arguments point into,
//! the framing that says where one command ends and the next begins, the reply
//! buffer that holds an answer until the batch is done, and the state a
//! connection keeps between the two.
//!
//! # Two halves
//!
//! [`Wire`] is a pair rather than a thing. The connection half is the front,
//! and it is in a module of its own that cannot name a [`Server`]: the buffers,
//! the decoder pool, the framing, the sessions and the queue of framed work.
//! The other half is the server, which is the databases and the numbers `INFO`
//! reports. The line matters because it is the line the threads run along: a
//! front belongs to the thread that accepted its connections and is reached by
//! nothing else, and the server is the handle every thread holds a copy of.
//! Everything that needs both is a method on `Wire` and there are three of them,
//! which are running a command, answering a client that blocked and forgetting a
//! client that has gone.
//!
//! # What a piece of work is
//!
//! [`Cmd`] is three numbers: which connection, which decoder holds the
//! arguments, and where in that connection's buffer they point. It is `Copy`
//! and twenty four bytes, so it crosses an intake lane without touching the
//! heap, and it carries no borrow, which is what lets the reactor hold sixty
//! four of them while the engine owns the bytes they name.
//!
//! The decoders are pooled. Framing takes one out of the pool per command,
//! `run` puts it back, and a connection with a half read command keeps hold of
//! one so that a bulk arriving in ten reads is decoded once rather than ten
//! times. In the steady state the pool is as large as the deepest batch and
//! nothing here allocates at all.
//!
//! # One write per connection
//!
//! Replies accumulate in the connection's [`Out`](crate::reply::Out) and go out
//! in [`Wire::flush`], which is one call to the sink per connection touched by
//! the batch and never one per reply. That is the syscall shape `04` section 2
//! asks for, and it is the one aki got wrong: its `HGETALL` profile spent 69.7
//! percent of its time in write syscalls.
//!
//! # What is not here
//!
//! Sockets. [`Sink`] is where the bytes go and the io_uring reactor implements
//! it later, which keeps this module testable without a network and keeps the
//! ring out of the crate that parses the protocol.
//!
//! The hash the first walk computes warms the bucket and is then thrown away,
//! because `yo-kv`'s commands take keys rather than hashes. The prefetch is the
//! part that is worth a cache miss; hashing a short key twice is a few
//! nanoseconds, and removing the second one means a hashed form of every
//! command method, which is a change to make with a benchmark rather than on
//! the way past.
//!
//! ```
//! use yo_resp::engine::{Recorder, Wire, pump};
//! use yo_reactor::Reactor;
//!
//! let mut r = Reactor::inline(Wire::new(Recorder::new()));
//! let conn = r.engine_mut().accept();
//!
//! r.engine_mut().feed(conn, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n*2\r\n$3\r\nGET\r\n$1\r\nk\r\n");
//! let mut batch = Vec::new();
//! assert_eq!(pump(&mut r, &mut batch), 2);
//!
//! assert_eq!(r.engine().sink().sent(conn), b"+OK\r\n$1\r\nv\r\n");
//! ```

use std::sync::Arc;

use yo_reactor::{BATCH_MAX, Engine, Reactor};

use crate::dispatch::table;
use crate::dispatch::{self, Flow, Server};
use crate::front::{Front, Wrote};
use crate::proto::Limits;
use yo_kv::Keyspace;

pub use crate::front::Cmd;

/// Which connection. An index, reused after a connection closes.
pub type ConnId = u32;

/// Where replies go.
///
/// One call per connection per batch, with however many replies are waiting.
/// The network reactor implements this over io_uring, a test implements it over
/// a `Vec`, and neither this module nor `dispatch` has to know which.
pub trait Sink {
    /// Take up to all of `bytes` for `conn`, and say how many were taken.
    ///
    /// Fewer than were offered means the socket is full: what is left stays in
    /// the connection's reply buffer and is offered again on the next flush.
    fn write(&mut self, conn: ConnId, bytes: &[u8]) -> usize;

    /// The connection is finished with and its id is about to be reused.
    fn closed(&mut self, conn: ConnId) {
        let _ = conn;
    }
}

/// A sink that keeps everything, for tests and for a driver with no socket.
#[derive(Debug, Default)]
pub struct Recorder {
    sent: Vec<Vec<u8>>,
    closed: Vec<ConnId>,
}

impl Recorder {
    /// An empty one.
    #[must_use]
    pub fn new() -> Recorder {
        Recorder::default()
    }

    /// Everything written to a connection so far.
    #[must_use]
    pub fn sent(&self, conn: ConnId) -> &[u8] {
        self.sent.get(conn as usize).map_or(&[], Vec::as_slice)
    }

    /// Whether a connection was closed.
    #[must_use]
    pub fn was_closed(&self, conn: ConnId) -> bool {
        self.closed.contains(&conn)
    }

    /// Forget what was written, keeping the room it was written into.
    pub fn clear(&mut self) {
        for c in &mut self.sent {
            c.clear();
        }
        self.closed.clear();
    }
}

impl Sink for Recorder {
    fn write(&mut self, conn: ConnId, bytes: &[u8]) -> usize {
        // A test sink, so the growth here is not on anybody's data path.
        yo_alloc::allow(|| {
            if self.sent.len() <= conn as usize {
                self.sent.resize_with(conn as usize + 1, Vec::new);
            }
            self.sent[conn as usize].extend_from_slice(bytes);
        });
        bytes.len()
    }

    fn closed(&mut self, conn: ConnId) {
        yo_alloc::allow(|| self.closed.push(conn));
    }
}

/// The engine: connections on one side, the command layer on the other.
///
/// One per thread, and it is two halves rather than one thing. The front is the
/// connections and everything they own, which never leaves the thread that
/// accepted them. [`Server`] is the databases, and every thread has a handle on
/// the same one. This type is where the two meet, and every method on it that is
/// not a one line delegation is a method that genuinely needs both: running a
/// command, answering a client that blocked, and forgetting a client that has
/// gone.
pub struct Wire<S> {
    front: Front<S>,
    server: Arc<Server>,
}

impl<S: Sink> Wire<S> {
    /// An engine with an empty server.
    #[must_use]
    pub fn new(sink: S) -> Wire<S> {
        Wire::with_server(Server::new(), sink)
    }

    /// An engine over a server the caller built, which is how a test gives it a
    /// clock it can move by hand.
    #[must_use]
    pub fn with_server(server: Server, sink: S) -> Wire<S> {
        Wire::over(Arc::new(server), sink)
    }

    /// An engine over a server that already exists, which is how the second
    /// thread and every thread after it gets one.
    ///
    /// Each thread builds its own front and they never see each other's. What
    /// they share is behind the handle, and the reason the handle is counted
    /// rather than borrowed is that the threads outlive whichever call started
    /// them by design: a scope that borrows would tie the server's lifetime to
    /// a frame that is meant to return.
    #[must_use]
    pub fn over(server: Arc<Server>, sink: S) -> Wire<S> {
        Wire {
            front: Front::new(sink),
            server,
        }
    }

    /// The databases and the numbers `INFO` reports.
    #[must_use]
    pub fn server(&self) -> &Server {
        &self.server
    }

    /// Another handle on the same server, for building the next thread's
    /// engine.
    #[must_use]
    pub fn shared(&self) -> Arc<Server> {
        Arc::clone(&self.server)
    }

    /// The server, for the few settings that have to be made before it is
    /// serving.
    ///
    /// That is the directory and the thread count, both of which are read
    /// everywhere and written once at startup, so they are settings and not
    /// state. This works while this engine holds the only handle, which is the
    /// case from the moment the server is built until the threads are started,
    /// and it is the caller's job to do its setting up in that window.
    ///
    /// # Panics
    ///
    /// If a second handle already exists, because there is no honest answer to
    /// give: changing the directory under a thread that is already serving out
    /// of it is the bug this would otherwise hide.
    pub fn server_mut(&mut self) -> &mut Server {
        Arc::get_mut(&mut self.server)
            .expect("the server is set up before the threads that share it are started")
    }

    /// Where the replies went.
    #[must_use]
    pub const fn sink(&self) -> &S {
        self.front.sink()
    }

    /// The same, mutably.
    pub const fn sink_mut(&mut self) -> &mut S {
        self.front.sink_mut()
    }

    /// Change the protocol limits, which is `proto-max-bulk-len` and friends.
    pub fn set_limits(&mut self, limits: Limits) {
        self.front.set_limits(limits);
    }

    /// Open a connection and give back its id.
    pub fn accept(&mut self) -> ConnId {
        self.server.counted().opened();
        let at = self.front.open();
        self.note_buffers();
        at
    }

    /// Tell the server what the connection buffers are holding now.
    ///
    /// The front cannot reach the server, so it keeps the change and this is
    /// where it is handed over: at the end of whichever call moved a buffer.
    fn note_buffers(&mut self) {
        let delta = self.front.buffer_delta();
        if delta != 0 {
            self.server.note_conn_bytes(delta);
        }
    }

    /// The peer went away.
    ///
    /// Whatever is buffered for it is dropped rather than written, and the slot
    /// comes back as soon as the commands already framed out of its buffer have
    /// run, because those commands' arguments still point into it.
    pub fn hangup(&mut self, conn: ConnId) {
        if !self.front.live(conn) {
            return;
        }
        self.front.mark_gone(conn);
        // A parked client holds its own commands, and those commands are what
        // `pending` counts, so leaving it parked here would leave the slot owed
        // to a connection that is never going to be answered. They go back to
        // the queue and run as the no-ops a gone connection's commands are.
        if self.front.blocked(conn) {
            self.front.unpark(conn);
        }
        if self.front.pending(conn) == 0 {
            self.release(conn);
        }
        self.note_buffers();
    }

    /// Answer everybody who can be answered, and let go of everybody whose
    /// deadline has passed.
    ///
    /// The walk is over the waiter list rather than over the connections, so it
    /// costs what blocking costs and not what the server costs. Every caller
    /// checks that somebody is parked before calling, which is the load and the
    /// branch a server with nobody blocked pays.
    fn serve_waiters(&mut self) {
        let now = self.server.now_ms();
        let mut at = 0;
        while at < self.server.parked() {
            let p = self.server.waiters().at(at);
            // The slot is reused and the client id is not. `release` forgets
            // waiters, so this should never fire; it is here because being
            // wrong about it writes a reply into somebody else's socket rather
            // than dropping one.
            if !self.front.answers(p.conn, p.client) {
                self.server.drop_waiter(at);
                continue;
            }
            // The front cannot reach the databases and the server cannot reach
            // the connections, so the two halves are taken apart here and the
            // one buffer this waiter needs is handed over.
            let served = {
                let Wire { server, front } = self;
                server.serve_waiter(at, now, front.out(p.conn))
            };
            if served {
                self.server.drop_waiter(at);
                self.front.unpark(p.conn);
                self.front.soil(p.conn);
            } else {
                at += 1;
            }
        }
    }

    /// How many connections are open.
    #[must_use]
    pub fn clients(&self) -> usize {
        self.front.clients()
    }

    /// Commands framed and waiting for the reactor.
    #[must_use]
    pub fn ready(&self) -> usize {
        self.front.ready()
    }

    /// Connections with a reply that has not gone out yet.
    ///
    /// Non zero means a socket was full and what is left is being held for a
    /// later flush, which a driver waiting on readability needs to know: there
    /// is work here that no incoming byte will ever wake it up for.
    #[must_use]
    pub fn owed(&self) -> usize {
        self.front.owed()
    }

    /// Whether a client has asked the server to stop.
    ///
    /// The driver reads this once a turn, next to the flag a signal sets, and
    /// leaves its loop when either is set. Asked after the batch rather than
    /// during it, so the `SHUTDOWN` and everything that shared its batch is
    /// finished and written out before anything closes.
    #[must_use]
    pub fn stopping(&self) -> bool {
        self.server.stopping()
    }

    /// Decoders in the pool, which is the high water mark of one batch.
    #[must_use]
    pub fn decoders(&self) -> usize {
        self.front.decoders()
    }

    /// What every connection's read and reply buffers are holding.
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.front.buffer_bytes()
    }

    /// Take bytes off a connection and frame whatever commands they complete.
    ///
    /// Anything left over stays in the connection's buffer, half a command
    /// included, so the caller hands over whatever the socket gave it without
    /// looking at it.
    pub fn feed(&mut self, conn: ConnId, bytes: &[u8]) {
        self.front.feed(conn, bytes);
        self.note_buffers();
    }

    /// Hand the slot and its buffers back, and let the server go of the client.
    fn release(&mut self, conn: ConnId) {
        let Some(client) = self.front.close(conn) else {
            return;
        };
        self.forget(client);
    }

    /// The server side of a connection ending.
    ///
    /// It happens in the same call the slot was freed in, and before anything
    /// else can run, because the slot is handed out again by the next accept
    /// and a waiter still holding this client id would then be a waiter
    /// pointing at somebody else's connection.
    fn forget(&mut self, client: u64) {
        self.server.forget_waiters(client);
        self.server.counted().closed();
    }

    /// Move up to `max` framed commands into `into`.
    ///
    /// The reactor wants a batch it owns, and the front keeps the buffers, so
    /// what crosses between them is this: numbers, no borrows.
    pub fn take_ready(&mut self, into: &mut Vec<Cmd>, max: usize) -> usize {
        self.front.take_ready(into, max)
    }

    /// Take a clock reading for the whole batch.
    ///
    /// `04` section 5: once per turn, never per command, so every command in a
    /// batch compares against the same millisecond and two keys written
    /// together expire together.
    pub fn tick(&mut self) {
        self.server.refresh_clock();
    }

    /// Do one batch's worth of housekeeping.
    ///
    /// Today that is one segment of arena compaction at most, which is what
    /// stops a server that rewrites the same keys from holding every version of
    /// them. It is separate from [`Wire::tick`] because the clock has to move
    /// before a batch runs and this does not: it can wait until the replies are
    /// out, and the driver decides when that is.
    ///
    /// Per batch and not per turn of the loop. A turn can carry one command or
    /// a thousand, so a per turn call means the rate at which garbage is
    /// collected has nothing to do with the rate at which it is made, and on a
    /// saturated server the second one wins. That was measured: with this on
    /// the loop's turn the server settled at seven segments for six segments'
    /// worth of keys, which is where an unloaded process running the same
    /// writes settled at six.
    pub fn maintain(&mut self) -> Option<usize> {
        // Before the compaction and not after it, because the reading the next
        // batch judges its limit against should be the one taken after the last
        // batch's writes rather than the one taken after this call's collecting.
        // Both are true, and the first is the one that is a batch old at worst.
        // Nothing at all on a server with no `maxmemory`, which is the default.
        self.server.refresh_memory();
        // Two fields and a return on a server that has never taken a backup,
        // which is nearly all of them. It is here rather than on a timer for the
        // same reason the compaction is: one loop turns everything.
        self.server.backup_expire();
        self.server.compact_step()
    }
}

impl<S: Sink> Engine for Wire<S> {
    type Work = Cmd;

    fn key_hash(&self, cmd: &Cmd) -> Option<u64> {
        // Before the argument list is built, because most of the commands that
        // get this far and answer `None` answer it on the spec alone, and
        // building an `Args` to then throw it away is the sort of thing that
        // does not show up in a profile and does show up in a total.
        let spec = table::at(cmd.spec)?;
        if spec.first_key <= 0 {
            return None;
        }
        let args = self.front.args(cmd);
        // The first key only. A command with more than one, which is `MSET` and
        // `MGET`, warms the first and takes the miss on the rest; warming all of
        // them means a hash list per command and that is the batch's own job
        // once multi key commands are worth measuring.
        let key = args.opt(spec.first_key as usize)?;
        Some(Keyspace::hash_of(key))
    }

    fn prefetch(&self, cmd: &Cmd, hash: u64) {
        let db = self.front.db(cmd.conn());
        // The hash picks the stripe as well as the record, so this warms the
        // line the command is going to read and not a line on some other
        // stripe. It is the same hash the command itself will route on, which
        // is why the stripe is worked out from a hash rather than from a key.
        self.server.striped_ref(db).prefetch_hashed(hash);
    }

    fn run(&mut self, cmd: Cmd, _hash: Option<u64>) -> yo_reactor::Flow {
        let conn = cmd.conn();
        // Framed with the batch that blocked, so it is a command the client sent
        // before it knew it would be waiting. It keeps its decoder and it keeps
        // its place in `pending`, which is what stops the buffer it points into
        // being compacted while it waits.
        if self.front.blocked(conn) {
            self.front.park(conn, cmd);
            return yo_reactor::Flow::Next;
        }

        // The one place both halves are held at once. The front hands over the
        // arguments, the session and the reply buffer, the server hands over
        // the databases, and the command layer sees the two as one call.
        let flow = if self.front.start(&cmd) {
            let Wire { front, server } = self;
            let (args, session, out) = front.parts(&cmd);
            let spec = table::at(cmd.spec);
            dispatch::resolved(server, session, spec, args, out)
        } else {
            // Nobody to answer, or nobody who should be. The decoder still has
            // to come back and the slot still has to be released, which is why
            // this is not an early return.
            Flow::Continue
        };

        self.front.done(&cmd);
        if self.front.gone(conn) {
            if self.front.pending(conn) == 0 {
                self.release(conn);
            }
        } else {
            match flow {
                Flow::Close => {
                    self.front.quit(conn);
                    self.front.soil(conn);
                }
                // Nothing was written, so there is nothing to flush and no
                // reason to put this connection on the dirty list. The waiter
                // carries the slot from here on, and it needs to know which one:
                // the command layer only ever saw the client id.
                Flow::Block => {
                    self.front.block(conn);
                    let client = self.front.client(conn);
                    self.server.bind_waiter(client, conn);
                }
                Flow::Continue => self.front.soil(conn),
            }
        }

        // After each command and not once per batch. A client blocked on two
        // keys and woken by `RPUSH b` then `RPUSH a` in one pipeline has to
        // answer with `b`, because that is the push that was in front of it, and
        // it can only do that if it was served in between the two.
        if self.server.parked() != 0 {
            self.serve_waiters();
        }
        yo_reactor::Flow::Next
    }

    fn flush(&mut self) {
        // The deadline sweep, and it is here because this is the one thing the
        // driver calls on a turn that ran nothing at all. A client whose timeout
        // passes while the server is idle is answered within the loop's idle
        // wait, which is 20ms and is finer than the 10hz Redis checks its own
        // blocked clients at.
        if self.server.parked() != 0 {
            self.server.refresh_clock();
            self.serve_waiters();
        }

        // Taken and put back so the loop below can reach the rest of the
        // engine. The capacity comes back with it, so this is not an
        // allocation.
        let mut dirty = self.front.take_dirty();
        let mut at = 0;
        while at < dirty.len() {
            let conn = dirty[at];
            match self.front.write_out(conn) {
                // The socket was full. The connection stays on the list with
                // what is left of its reply, and the next flush offers it
                // again, which is the whole of the backpressure story here.
                Wrote::Owed => at += 1,
                Wrote::Done => {
                    dirty.swap_remove(at);
                }
                Wrote::Ended(client) => {
                    self.forget(client);
                    dirty.swap_remove(at);
                }
            }
        }
        self.front.give_dirty(dirty);
        self.note_buffers();
    }

    fn maintain(&mut self, budget: &mut yo_reactor::Budget) {
        // The clock is the first thing the maintenance slice does, because
        // everything else in it compares against a time.
        if !budget.spend(1) {
            return;
        }
        self.tick();
        // Then the dead keys, which is what stops a cache that writes with a
        // deadline and never reads back from holding every key it has ever
        // written. One unit a key looked at, so the slice bounds the sweep the
        // same way it bounds everything else in here, and a server where nothing
        // has a deadline spends nothing at all.
        let looks = budget.left() as usize;
        let spent = self.server.expire_slice(looks);
        budget.spend(u32::try_from(spent).unwrap_or(u32::MAX));
    }
}

/// Run everything that is framed, in batches, and write the replies.
///
/// The inline driver: it is what a caller who is already on the shard thread
/// uses in place of the loop, and it goes through the same two walks the loop
/// goes through (`15` section 7). `batch` is the caller's, so a driver in a hot
/// loop hands the same `Vec` back every time and never allocates.
pub fn pump<S: Sink>(reactor: &mut Reactor<Wire<S>>, batch: &mut Vec<Cmd>) -> usize {
    let mut ran = 0;
    reactor.engine_mut().tick();
    loop {
        batch.clear();
        if reactor.engine_mut().take_ready(batch, BATCH_MAX) == 0 {
            break;
        }
        // The command path, and therefore the thing Y7 is about. The guard is
        // what arms `yo-alloc`, and it covers dispatch and nothing else: framing
        // before it and writing the replies after it are both allowed to reach
        // for the heap, and only running the commands is not.
        //
        // It goes here rather than around the whole loop because `take_ready`
        // and `flush` are on the other side of that line, and because a batch is
        // the unit a caller can reason about. Under the default mode this is one
        // relaxed load.
        let armed = yo_alloc::guard();
        ran += reactor.execute_all(batch.drain(..));
        drop(armed);
        reactor.engine_mut().flush();
        // After the replies are out, so the batch that made the garbage is not
        // the batch that waits for it to be collected.
        reactor.engine_mut().maintain();
    }
    // Once more, for a connection with something to say and nothing to run: a
    // protocol error, or a socket that was full the last time round.
    reactor.engine_mut().flush();
    // And once for a turn that ran nothing at all, which is where a server that
    // has gone quiet catches up on what the last busy turn left behind.
    reactor.engine_mut().maintain();
    ran
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire bytes for a command, built the way a client would.
    fn wire(args: &[&[u8]]) -> Vec<u8> {
        let mut b = format!("*{}\r\n", args.len()).into_bytes();
        for a in args {
            b.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
            b.extend_from_slice(a);
            b.extend_from_slice(b"\r\n");
        }
        b
    }

    fn engine() -> (Reactor<Wire<Recorder>>, ConnId, Vec<Cmd>) {
        let mut r = Reactor::inline(Wire::new(Recorder::new()));
        let conn = r.engine_mut().accept();
        (r, conn, Vec::new())
    }

    /// Where the fixed clock a blocking test moves by hand starts.
    const START_MS: u64 = 1_000_000;

    /// The same, on a clock the test moves rather than the system's.
    ///
    /// A test about a timeout cannot wait for one: waiting a hundred
    /// milliseconds is a test that fails on a loaded machine and waiting a
    /// hundred seconds is not a test.
    fn timed() -> (Reactor<Wire<Recorder>>, ConnId, Vec<Cmd>) {
        let server = crate::dispatch::Server::with_clock(yo_kv::Clock::fixed(START_MS));
        let mut r = Reactor::inline(Wire::with_server(server, Recorder::new()));
        let conn = r.engine_mut().accept();
        (r, conn, Vec::new())
    }

    #[test]
    fn a_pipelined_batch_comes_back_in_order_and_in_one_write() {
        let (mut r, conn, mut batch) = engine();
        let mut stream = wire(&[b"SET", b"k", b"v"]);
        stream.extend(wire(&[b"GET", b"k"]));
        stream.extend(wire(&[b"INCR", b"n"]));

        r.engine_mut().feed(conn, &stream);
        assert_eq!(r.engine().ready(), 3);
        assert_eq!(pump(&mut r, &mut batch), 3);

        assert_eq!(r.engine().sink().sent(conn), b"+OK\r\n$1\r\nv\r\n:1\r\n");
        assert_eq!(r.engine().ready(), 0);
    }

    /// The framing has to survive a command arriving in pieces, because that is
    /// what a socket does.
    #[test]
    fn a_command_split_across_reads_resumes_rather_than_restarts() {
        let (mut r, conn, mut batch) = engine();
        let bytes = wire(&[b"SET", b"key", b"value"]);

        for at in 1..bytes.len() {
            r.engine_mut().feed(conn, &bytes[at - 1..at]);
            assert_eq!(r.engine().ready(), 0, "not a command yet at {at}");
        }
        r.engine_mut().feed(conn, &bytes[bytes.len() - 1..]);
        assert_eq!(r.engine().ready(), 1);
        assert_eq!(pump(&mut r, &mut batch), 1);
        assert_eq!(r.engine().sink().sent(conn), b"+OK\r\n");

        // And the value that arrived in single bytes is the value that was
        // stored, which is the part a naive resume gets wrong.
        r.engine_mut().feed(conn, &wire(&[b"GET", b"key"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(conn), b"+OK\r\n$5\r\nvalue\r\n");
    }

    #[test]
    fn two_connections_are_two_sessions_over_one_server() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();

        r.engine_mut().feed(a, &wire(&[b"SELECT", b"3"]));
        r.engine_mut().feed(a, &wire(&[b"SET", b"k", b"a"]));
        r.engine_mut().feed(b, &wire(&[b"SET", b"k", b"b"]));
        r.engine_mut().feed(a, &wire(&[b"GET", b"k"]));
        r.engine_mut().feed(b, &wire(&[b"GET", b"k"]));
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().sink().sent(a), b"+OK\r\n+OK\r\n$1\r\na\r\n");
        assert_eq!(r.engine().sink().sent(b), b"+OK\r\n$1\r\nb\r\n");
        assert_eq!(r.engine().clients(), 2);
    }

    /// The point of the whole exercise: two engines, two threads, one server.
    #[test]
    fn two_threads_write_into_one_server() {
        const EACH: usize = 200;

        let first = Wire::new(Recorder::new());
        let second = Wire::over(first.shared(), Recorder::new());
        let server = first.shared();

        std::thread::scope(|s| {
            for (at, engine) in [first, second].into_iter().enumerate() {
                s.spawn(move || {
                    let mut r = Reactor::inline(engine);
                    let mut batch = Vec::new();
                    let conn = r.engine_mut().accept();
                    for i in 0..EACH {
                        let key = format!("t{at}:{i}");
                        r.engine_mut()
                            .feed(conn, &wire(&[b"SET", key.as_bytes(), b"v"]));
                        pump(&mut r, &mut batch);
                    }
                });
            }
        });

        // Every key both threads wrote is in the one database, which is the
        // whole claim: the fronts were separate and the keyspace was not.
        assert_eq!(server.striped_ref(0).len(), 2 * EACH);
        // And both threads counted into the same total, each from its own set
        // of counters, which is what the sum over the threads is for.
        assert_eq!(server.totals().connections, 2);
    }

    #[test]
    fn quit_is_answered_and_then_the_connection_goes() {
        let (mut r, conn, mut batch) = engine();
        r.engine_mut().feed(conn, &wire(&[b"PING"]));
        r.engine_mut().feed(conn, &wire(&[b"QUIT"]));
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().sink().sent(conn), b"+PONG\r\n+OK\r\n");
        assert!(r.engine().sink().was_closed(conn));
        assert_eq!(r.engine().clients(), 0);

        // The slot comes back, buffers and all.
        let again = r.engine_mut().accept();
        assert_eq!(again, conn);
        assert_eq!(r.engine().clients(), 1);
    }

    /// Redis's own unit/quit, which caught this: we answered the `QUIT` and
    /// then ran the `SET` behind it.
    #[test]
    fn what_a_client_pipelined_behind_quit_is_never_run() {
        let (mut r, conn, mut batch) = engine();
        let mut stream = wire(&[b"QUIT"]);
        stream.extend(wire(&[b"SET", b"foo", b"bar"]));
        r.engine_mut().feed(conn, &stream);
        // Both were framed, because framing happens before anything runs.
        assert_eq!(r.engine().ready(), 2);
        pump(&mut r, &mut batch);

        // One reply and not two, and the connection is gone.
        assert_eq!(r.engine().sink().sent(conn), b"+OK\r\n");
        assert!(r.engine().sink().was_closed(conn));

        // And the write never happened, which is the part a client can see
        // after it reconnects. The recorder is cleared first because the next
        // connection lands back in the slot this one just left, and what was
        // written to the slot before is still sitting in it.
        r.engine_mut().sink_mut().clear();
        let next = r.engine_mut().accept();
        r.engine_mut().feed(next, &wire(&[b"GET", b"foo"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(next), b"$-1\r\n");
    }

    /// A connection that never said `HELLO` is answered in RESP2, whatever the
    /// last client in that slot was speaking.
    ///
    /// The protocol is kept in the reply buffer and the reply buffer outlives
    /// the connection, so this is the one piece of connection state that a
    /// recycled slot used to carry over. A client got a RESP3 null back from
    /// the first `GET` that missed and could not parse it, which is as bad as a
    /// compatibility bug gets: nothing the client did caused it and nothing it
    /// could send would have avoided it.
    #[test]
    fn a_slot_that_last_spoke_resp3_answers_the_next_client_in_resp2() {
        let (mut r, conn, mut batch) = engine();
        r.engine_mut().feed(conn, &wire(&[b"HELLO", b"3"]));
        r.engine_mut().feed(conn, &wire(&[b"GET", b"nothing"]));
        pump(&mut r, &mut batch);
        assert!(r.engine().sink().sent(conn).ends_with(b"_\r\n"));
        r.engine_mut().feed(conn, &wire(&[b"QUIT"]));
        pump(&mut r, &mut batch);

        r.engine_mut().sink_mut().clear();
        let next = r.engine_mut().accept();
        assert_eq!(next, conn, "the same slot, which is what this is about");
        r.engine_mut().feed(next, &wire(&[b"GET", b"nothing"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(next), b"$-1\r\n");
    }

    /// The other way a connection ends, which does not throw anything away.
    #[test]
    fn commands_that_arrived_before_a_protocol_error_are_still_answered() {
        let (mut r, conn, mut batch) = engine();
        let mut stream = wire(&[b"SET", b"k", b"v"]);
        stream.extend(wire(&[b"GET", b"k"]));
        stream.extend_from_slice(b"*1\r\n+notabulk\r\n");
        r.engine_mut().feed(conn, &stream);
        pump(&mut r, &mut batch);

        // Both good commands were complete and correct before the stream went
        // wrong, so both are answered and the error comes after them.
        let sent = r.engine().sink().sent(conn);
        assert!(
            sent.starts_with(b"+OK\r\n$1\r\nv\r\n-ERR Protocol error: "),
            "{sent:?}"
        );
        assert!(r.engine().sink().was_closed(conn));
    }

    #[test]
    fn a_protocol_error_is_written_and_closes_the_connection() {
        let (mut r, conn, mut batch) = engine();
        // A multibulk that says its first argument is a bulk and then does not.
        r.engine_mut().feed(conn, b"*1\r\n+notabulk\r\n");
        pump(&mut r, &mut batch);

        let sent = r.engine().sink().sent(conn);
        assert!(sent.starts_with(b"-ERR Protocol error: "), "{sent:?}");
        assert!(r.engine().sink().was_closed(conn));
        assert_eq!(r.engine().clients(), 0);
    }

    /// Redis's own `unit/protocol` walks a list of malformed frames, each on a
    /// fresh connection, which means every one of them after the first runs on
    /// a decoder that came back to the pool part way through a command.
    #[test]
    fn a_decoder_that_came_back_mid_command_starts_the_next_one_clean() {
        let (mut r, conn, mut batch) = engine();
        // Stops inside the third argument, on a length that is not a length.
        r.engine_mut()
            .feed(conn, b"*3\r\n$3\r\nSET\r\n$1\r\nx\r\n$blabla\r\n");
        pump(&mut r, &mut batch);
        let sent = r.engine().sink().sent(conn);
        assert!(
            sent.starts_with(b"-ERR Protocol error: invalid bulk length"),
            "{sent:?}"
        );

        // The slot that decoder was in is now the slot the next connection
        // gets, and it has to be at the start of a command and not half way
        // through the one that went wrong.
        r.engine_mut().sink_mut().clear();
        let next = r.engine_mut().accept();
        r.engine_mut().feed(next, &wire(&[b"GET", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(next), b"$-1\r\n");

        r.engine_mut().sink_mut().clear();
        let third = r.engine_mut().accept();
        r.engine_mut().feed(third, b"*1\r\n+notabulk\r\n");
        pump(&mut r, &mut batch);
        let sent = r.engine().sink().sent(third);
        assert!(sent.starts_with(b"-ERR Protocol error: "), "{sent:?}");
    }

    /// A client that hangs up mid batch is the case that gets a server killed:
    /// the commands already framed still point into its buffer.
    #[test]
    fn a_hangup_with_commands_in_flight_waits_for_them() {
        let (mut r, conn, mut batch) = engine();
        r.engine_mut().feed(conn, &wire(&[b"SET", b"k", b"v"]));
        r.engine_mut().feed(conn, &wire(&[b"GET", b"k"]));

        batch.clear();
        r.engine_mut().take_ready(&mut batch, BATCH_MAX);
        r.engine_mut().hangup(conn);
        assert_eq!(r.engine().clients(), 1, "still holding the buffer");

        r.execute_all(batch.drain(..));
        r.engine_mut().flush();
        assert_eq!(r.engine().clients(), 0);
        assert!(r.engine().sink().sent(conn).is_empty(), "nobody to answer");

        // And the slot is usable again, with the decoders both back in the
        // pool rather than lost with the connection.
        let decoders = r.engine().decoders();
        let again = r.engine_mut().accept();
        assert_eq!(again, conn);
        r.engine_mut().feed(again, &wire(&[b"PING"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(again), b"+PONG\r\n");
        assert_eq!(r.engine().decoders(), decoders);
    }

    /// The claim that the steady state does not allocate, checked the only way
    /// a library test can check it: nothing grows.
    #[test]
    fn the_buffers_and_the_decoder_pool_stop_growing() {
        let (mut r, conn, mut batch) = engine();
        let mut stream = Vec::new();
        for i in 0..32 {
            stream.extend(wire(&[b"SET", format!("k{i}").as_bytes(), b"v"]));
        }

        r.engine_mut().feed(conn, &stream);
        pump(&mut r, &mut batch);
        let decoders = r.engine().decoders();
        let batch_cap = batch.capacity();

        for _ in 0..10 {
            r.engine_mut().feed(conn, &stream);
            pump(&mut r, &mut batch);
        }
        assert_eq!(r.engine().decoders(), decoders, "the pool is reused");
        assert_eq!(batch.capacity(), batch_cap, "the batch buffer is reused");
        assert!(
            decoders <= BATCH_MAX + 1,
            "{decoders} decoders for 32 commands"
        );
    }

    /// The read buffer holds what has not been dealt with yet and nothing else.
    ///
    /// A client that pipelines sixteen commands, waits for the sixteen replies
    /// and goes again is what `redis-benchmark -P 16` does and what half of the
    /// clients in the world do. Every one of those rounds leaves the buffer
    /// exactly caught up, and a buffer that never drops what it has already
    /// dealt with grows to everything the connection has ever sent: 16 MiB
    /// apiece on server3 for four connections sending 100000 sets each.
    #[test]
    fn a_pipelining_client_does_not_grow_the_read_buffer() {
        let (mut r, conn, mut batch) = engine();
        let mut round = Vec::new();
        for i in 0..16 {
            round.extend(wire(&[b"SET", format!("k{i}").as_bytes(), b"v"]));
        }

        r.engine_mut().feed(conn, &round);
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();
        let after_one = r.engine().buffer_bytes();

        // A thousand rounds is sixteen thousand commands and about a megabyte
        // of wire bytes, which is a hundred times what the buffer starts with.
        for _ in 0..1000 {
            r.engine_mut().feed(conn, &round);
            pump(&mut r, &mut batch);
            r.engine_mut().sink_mut().clear();
        }

        assert_eq!(
            r.engine().buffer_bytes(),
            after_one,
            "the buffers grew over a thousand rounds of the same sixteen commands"
        );
        assert!(
            r.engine().server().memory_bytes() >= after_one,
            "the buffers are counted in what the server reports"
        );
    }

    /// Half a command in the buffer is the case compaction has to be careful
    /// about, because the decoder holding it kept offsets into those bytes.
    #[test]
    fn a_command_split_across_reads_survives_compaction() {
        let (mut r, conn, mut batch) = engine();
        let cmd = wire(&[b"SET", b"key", b"value"]);
        let (head, tail) = cmd.split_at(cmd.len() - 4);

        // A complete command, so that there is something in front to drop, then
        // most of a second one.
        r.engine_mut().feed(conn, &wire(&[b"PING"]));
        r.engine_mut().feed(conn, head);
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(conn), b"+PONG\r\n");

        // The rest of it arrives after the buffer has been compacted under it.
        r.engine_mut().feed(conn, tail);
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(conn), b"+PONG\r\n+OK\r\n");

        r.engine_mut().feed(conn, &wire(&[b"GET", b"key"]));
        pump(&mut r, &mut batch);
        assert!(r.engine().sink().sent(conn).ends_with(b"$5\r\nvalue\r\n"));
    }

    /// The two walks are the reactor's, not this module's, so the test is that
    /// the engine can be driven by them at all: same commands, same replies.
    #[test]
    fn the_batch_goes_through_the_reactors_two_walks() {
        let (mut r, conn, mut batch) = engine();
        for i in 0..100 {
            r.engine_mut()
                .feed(conn, &wire(&[b"INCR", format!("k{}", i % 7).as_bytes()]));
        }
        let ran = pump(&mut r, &mut batch);

        assert_eq!(ran, 100);
        assert_eq!(r.commands(), 100);
        // Two batches, because a hundred commands do not fit in sixty four.
        assert_eq!(r.turns(), 2);
        // The hundredth command is the fifteenth `INCR` of `k1`.
        assert!(r.engine().sink().sent(conn).ends_with(b":15\r\n"));
    }

    /// A sink that takes four bytes at a time, which is what a full socket
    /// looks like from in here.
    #[derive(Default)]
    struct Trickle {
        sent: Vec<u8>,
        writes: usize,
    }

    impl Sink for Trickle {
        fn write(&mut self, _conn: ConnId, bytes: &[u8]) -> usize {
            self.writes += 1;
            let n = bytes.len().min(4);
            self.sent.extend_from_slice(&bytes[..n]);
            n
        }
    }

    /// A blocking command that does not block costs nothing: no waiter, no
    /// allocation, the same three lines the non blocking one runs.
    #[test]
    fn a_blpop_on_a_list_with_something_in_it_never_waits() {
        let (mut r, conn, mut batch) = engine();
        r.engine_mut().feed(conn, &wire(&[b"RPUSH", b"q", b"a"]));
        r.engine_mut().feed(conn, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);

        assert_eq!(
            r.engine().sink().sent(conn),
            b":1\r\n*2\r\n$1\r\nq\r\n$1\r\na\r\n"
        );
        assert_eq!(r.engine().server().parked(), 0);
    }

    /// The whole point: a client with nothing to pop is answered later, by
    /// somebody else's command.
    #[test]
    fn a_parked_client_is_answered_by_another_connections_push() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();

        r.engine_mut().feed(a, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);
        assert!(r.engine().sink().sent(a).is_empty(), "nothing to say yet");
        assert_eq!(r.engine().server().parked(), 1);

        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"one"]));
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().sink().sent(a), b"*2\r\n$1\r\nq\r\n$3\r\none\r\n");
        // The push still reports the length it made, even though the element was
        // gone again before the reply was written.
        assert_eq!(r.engine().sink().sent(b), b":1\r\n");
        assert_eq!(r.engine().server().parked(), 0);
    }

    /// A push to a key nobody named, and a key of another type on a key
    /// somebody did: neither is a wake up, and the client stays parked.
    #[test]
    fn only_a_list_arriving_under_a_named_key_wakes_a_waiter() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();
        r.engine_mut().feed(a, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);

        r.engine_mut()
            .feed(b, &wire(&[b"RPUSH", b"elsewhere", b"x"]));
        r.engine_mut().feed(b, &wire(&[b"SADD", b"q", b"x"]));
        pump(&mut r, &mut batch);

        assert!(r.engine().sink().sent(a).is_empty());
        assert_eq!(r.engine().server().parked(), 1, "still waiting");
        // And the set is intact, so the waiter did not take anything out of it
        // on its way past.
        assert_eq!(r.engine().sink().sent(b), b":1\r\n:1\r\n");
    }

    /// Two workers on one queue, which is what `BLPOP` is for. They are served
    /// in the order they arrived and not in whatever order the list is walked.
    #[test]
    fn two_parked_clients_are_served_in_the_order_they_arrived() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();
        let c = r.engine_mut().accept();

        r.engine_mut().feed(a, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);
        r.engine_mut().feed(b, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().server().parked(), 2);

        r.engine_mut()
            .feed(c, &wire(&[b"RPUSH", b"q", b"first", b"second"]));
        pump(&mut r, &mut batch);

        assert_eq!(
            r.engine().sink().sent(a),
            b"*2\r\n$1\r\nq\r\n$5\r\nfirst\r\n"
        );
        assert_eq!(
            r.engine().sink().sent(b),
            b"*2\r\n$1\r\nq\r\n$6\r\nsecond\r\n"
        );
        assert_eq!(r.engine().server().parked(), 0);
    }

    /// A client waiting for an answer is not a client that has sent another
    /// question, so what it pipelined behind its `BLPOP` waits for the `BLPOP`.
    #[test]
    fn what_a_client_pipelined_behind_a_block_waits_for_the_block() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();

        // Framed together, so the `PING` is already on its way to the reactor
        // when the `BLPOP` in front of it parks.
        let mut stream = wire(&[b"BLPOP", b"q", b"0"]);
        stream.extend(wire(&[b"PING"]));
        r.engine_mut().feed(a, &stream);
        pump(&mut r, &mut batch);
        assert!(
            r.engine().sink().sent(a).is_empty(),
            "the PING went out in front of the answer it was sent behind"
        );

        // And one that arrives while it is parked is not even framed.
        r.engine_mut().feed(a, &wire(&[b"ECHO", b"after"]));
        pump(&mut r, &mut batch);
        assert!(r.engine().sink().sent(a).is_empty());

        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"x"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(a),
            b"*2\r\n$1\r\nq\r\n$1\r\nx\r\n+PONG\r\n$5\r\nafter\r\n"
        );
    }

    /// Redis serves parked clients after every command rather than once per
    /// turn of the loop, and a pipeline is where the difference shows: the
    /// waiter has to be served between the two pushes, so it answers with the
    /// key the first push filled and not with the one it named first.
    #[test]
    fn a_waiter_is_served_between_two_pipelined_pushes() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();
        r.engine_mut()
            .feed(a, &wire(&[b"BLPOP", b"p1", b"p2", b"0"]));
        pump(&mut r, &mut batch);

        let mut stream = wire(&[b"RPUSH", b"p2", b"second"]);
        stream.extend(wire(&[b"RPUSH", b"p1", b"first"]));
        r.engine_mut().feed(b, &stream);
        pump(&mut r, &mut batch);

        assert_eq!(
            r.engine().sink().sent(a),
            b"*2\r\n$2\r\np2\r\n$6\r\nsecond\r\n"
        );
        // Which leaves the key it named first holding what was pushed to it.
        r.engine_mut()
            .feed(b, &wire(&[b"LRANGE", b"p1", b"0", b"-1"]));
        pump(&mut r, &mut batch);
        assert!(
            r.engine()
                .sink()
                .sent(b)
                .ends_with(b"*1\r\n$5\r\nfirst\r\n")
        );
    }

    /// A `BLMOVE` that serves itself is a push, so it wakes the client waiting
    /// on the key it pushed to, in the same moment and without a turn of the
    /// loop in between.
    #[test]
    fn a_waiter_woken_by_another_waiter() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();
        let c = r.engine_mut().accept();

        r.engine_mut()
            .feed(a, &wire(&[b"BLMOVE", b"x", b"y", b"LEFT", b"RIGHT", b"0"]));
        pump(&mut r, &mut batch);
        r.engine_mut().feed(b, &wire(&[b"BLPOP", b"y", b"0"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().server().parked(), 2);

        r.engine_mut().feed(c, &wire(&[b"RPUSH", b"x", b"chain"]));
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().sink().sent(a), b"$5\r\nchain\r\n");
        assert_eq!(
            r.engine().sink().sent(b),
            b"*2\r\n$1\r\ny\r\n$5\r\nchain\r\n"
        );
        assert_eq!(r.engine().server().parked(), 0);
    }

    /// A waiter on one database is not woken by a push on another, even though
    /// the key has the same name.
    #[test]
    fn a_waiter_is_only_woken_on_the_database_it_blocked_on() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();
        r.engine_mut().feed(a, &wire(&[b"SELECT", b"3"]));
        r.engine_mut().feed(a, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(a), b"+OK\r\n");

        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"wrongdb"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(a), b"+OK\r\n", "still waiting");

        r.engine_mut().feed(b, &wire(&[b"SELECT", b"3"]));
        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"rightdb"]));
        pump(&mut r, &mut batch);
        assert!(r.engine().sink().sent(a).ends_with(b"$7\r\nrightdb\r\n"));
    }

    /// The deadline sweep, which runs on a turn that has nothing else to do.
    #[test]
    fn a_client_that_waited_long_enough_gets_a_null_array() {
        let (mut r, conn, mut batch) = timed();
        r.engine_mut().feed(conn, &wire(&[b"BLPOP", b"q", b"30"]));
        pump(&mut r, &mut batch);
        assert!(r.engine().sink().sent(conn).is_empty());

        r.engine_mut().server_mut().set_clock_ms(START_MS + 29_999);
        pump(&mut r, &mut batch);
        assert!(
            r.engine().sink().sent(conn).is_empty(),
            "a millisecond short"
        );

        r.engine_mut().server_mut().set_clock_ms(START_MS + 30_000);
        pump(&mut r, &mut batch);
        // A null array and not a null string, which a RESP2 client can see.
        assert_eq!(r.engine().sink().sent(conn), b"*-1\r\n");
        assert_eq!(r.engine().server().parked(), 0);
    }

    /// The four that answer with something other than a two element array all
    /// answer a timeout the same way, which is not what the reply shape would
    /// suggest and is what Redis does.
    #[test]
    fn every_blocking_command_times_out_with_the_same_null_array() {
        for cmd in [
            &[b"BLPOP".as_slice(), b"q", b"0.001"][..],
            &[b"BRPOP", b"q", b"0.001"],
            &[b"BLMOVE", b"q", b"d", b"LEFT", b"RIGHT", b"0.001"],
            &[b"BRPOPLPUSH", b"q", b"d", b"0.001"],
            &[b"BLMPOP", b"0.001", b"1", b"q", b"LEFT"],
        ] {
            let (mut r, conn, mut batch) = timed();
            r.engine_mut().feed(conn, &wire(cmd));
            pump(&mut r, &mut batch);
            r.engine_mut().server_mut().set_clock_ms(START_MS + 1);
            pump(&mut r, &mut batch);
            assert_eq!(r.engine().sink().sent(conn), b"*-1\r\n", "for {cmd:?}");
        }
    }

    /// A client that gave up does not go on holding a claim on the queue: the
    /// element that arrives after it stays where it was put.
    #[test]
    fn a_waiter_that_timed_out_does_not_eat_a_later_push() {
        let (mut r, a, mut batch) = timed();
        let b = r.engine_mut().accept();
        r.engine_mut().feed(a, &wire(&[b"BLPOP", b"q", b"1"]));
        pump(&mut r, &mut batch);
        r.engine_mut().server_mut().set_clock_ms(START_MS + 1000);
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(a), b"*-1\r\n");

        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"late"]));
        r.engine_mut()
            .feed(b, &wire(&[b"LRANGE", b"q", b"0", b"-1"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(a), b"*-1\r\n", "nothing more");
        assert!(r.engine().sink().sent(b).ends_with(b"*1\r\n$4\r\nlate\r\n"));
    }

    /// A `BLPOP key 0` has no deadline, so nothing but the connection closing
    /// will ever take it off the list. That makes the close path the one that
    /// has to be right, or a waiter outlives its client and the slot it names
    /// gets handed to somebody else.
    #[test]
    fn a_client_that_goes_away_while_it_waits_takes_its_waiter_with_it() {
        let (mut r, a, mut batch) = engine();
        let b = r.engine_mut().accept();
        r.engine_mut().feed(a, &wire(&[b"BLPOP", b"q", b"0"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().server().parked(), 1);

        r.engine_mut().hangup(a);
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().server().parked(), 0);
        assert_eq!(r.engine().clients(), 1);

        // The slot is handed straight back out, which is what the waiter would
        // have been pointing at.
        let again = r.engine_mut().accept();
        assert_eq!(again, a);
        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"x"]));
        r.engine_mut()
            .feed(again, &wire(&[b"LRANGE", b"q", b"0", b"-1"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(again), b"*1\r\n$1\r\nx\r\n");
    }

    /// The same, with commands the client had already sent sitting behind the
    /// block. Those are what `pending` counts, so a close that forgets them is a
    /// connection slot that never comes back.
    #[test]
    fn a_hangup_while_parked_gives_back_the_slot_and_the_decoders() {
        let (mut r, a, mut batch) = engine();
        let mut stream = wire(&[b"BLPOP", b"q", b"0"]);
        stream.extend(wire(&[b"PING"]));
        stream.extend(wire(&[b"PING"]));
        r.engine_mut().feed(a, &stream);
        pump(&mut r, &mut batch);

        let decoders = r.engine().decoders();
        r.engine_mut().hangup(a);
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().clients(), 0);
        assert!(r.engine().sink().was_closed(a));
        assert_eq!(r.engine().decoders(), decoders, "the pool came back whole");
        let again = r.engine_mut().accept();
        assert_eq!(again, a);
        r.engine_mut().feed(again, &wire(&[b"PING"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(again), b"+PONG\r\n");
    }

    #[test]
    fn a_reply_the_socket_would_not_take_is_offered_again() {
        let mut r = Reactor::inline(Wire::new(Trickle::default()));
        let conn = r.engine_mut().accept();
        let mut batch = Vec::new();

        r.engine_mut().feed(conn, &wire(&[b"PING"]));
        pump(&mut r, &mut batch);
        // Two flushes in a pump, so four bytes and then three.
        assert_eq!(r.engine().sink().sent, b"+PONG\r\n");
        assert_eq!(r.engine().sink().writes, 2);
    }
}
