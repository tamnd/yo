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
//! Replies accumulate in the connection's [`Out`] and go out in [`Wire::flush`],
//! which is one call to the sink per connection touched by the batch and never
//! one per reply. That is the syscall shape `04` section 2 asks for, and it is
//! the one aki got wrong: its `HGETALL` profile spent 69.7 percent of its time
//! in write syscalls.
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

use std::collections::VecDeque;

use yo_reactor::{BATCH_MAX, Engine, Reactor};

use crate::dispatch::{Args, Flow, Server, Session, execute, lookup};
use crate::error::ProtocolError;
use crate::proto::{Limits, Proto};
use crate::reply::Out;
use crate::request::{Argv, Step};
use yo_kv::Strings;

/// Which connection. An index, reused after a connection closes.
pub type ConnId = u32;

/// The read buffer a connection starts with.
///
/// Redis's query buffer starts at sixteen kilobytes for the same reason: it is
/// larger than every command a client actually sends, so the buffer grows once
/// at accept time and then never again.
const READ_BUF: usize = 16 * 1024;

/// The reply buffer a connection starts with.
const OUT_BUF: usize = 16 * 1024;

/// How many arguments a decoder has room for before it grows.
const ARGV_HINT: usize = 8;

/// One framed command, waiting to run.
///
/// Names the bytes rather than holding them, so the reactor can queue a batch
/// of these while the engine keeps ownership of every buffer they point into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cmd {
    conn: ConnId,
    slot: u32,
    base: usize,
}

impl Cmd {
    /// The connection this command arrived on.
    #[must_use]
    pub const fn conn(&self) -> ConnId {
        self.conn
    }
}

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

/// One connection's state.
struct Conn {
    live: bool,
    session: Session,
    out: Out,
    /// What has arrived and not yet been framed away.
    buf: Vec<u8>,
    /// How much of `buf` the framing has consumed.
    head: usize,
    /// The decoder holding a command that has not all arrived.
    partial: Option<u32>,
    /// Commands framed out of this buffer and not yet run.
    pending: u32,
    /// This connection is on its way out, once what is buffered has gone.
    closing: bool,
    /// A protocol error waiting for the commands in front of it to answer.
    ///
    /// The framing finds the error before any of the batch it was framed with
    /// has run, and writing the error there would put it in front of replies
    /// the client is still owed. Redis answers in order, so this waits until
    /// nothing is pending and goes out last.
    deferred: Option<ProtocolError>,
    /// Everything still queued for this connection is thrown away unanswered.
    ///
    /// `QUIT` sets this and a protocol error does not, which is the difference
    /// between the two ways a connection ends. A client that pipelines `QUIT`
    /// and then `SET` has said goodbye and then said something after it, and
    /// Redis answers the goodbye and drops the rest. A client that sends two
    /// good commands and then a malformed one gets both good ones answered,
    /// because they were complete and correct before the stream went wrong.
    skip: bool,
    /// The peer is gone, so there is nothing to answer and nothing to write.
    gone: bool,
    /// Already on the dirty list.
    dirty: bool,
}

impl Conn {
    fn new(id: u64) -> Conn {
        // Accept time, which is the one moment a connection is allowed to cost
        // an allocation. Everything after this reuses these two buffers.
        yo_alloc::allow(|| Conn {
            live: true,
            session: Session::new(id),
            out: Out::with_capacity(Proto::Resp2, OUT_BUF),
            buf: Vec::with_capacity(READ_BUF),
            head: 0,
            partial: None,
            pending: 0,
            closing: false,
            deferred: None,
            skip: false,
            gone: false,
            dirty: false,
        })
    }

    /// Back to how it was at accept time, buffers kept.
    fn reset(&mut self, id: u64) {
        self.live = true;
        self.session = Session::new(id);
        self.out.clear();
        self.buf.clear();
        self.head = 0;
        self.partial = None;
        self.pending = 0;
        self.closing = false;
        self.deferred = None;
        self.skip = false;
        self.gone = false;
        self.dirty = false;
    }

    /// Drop what the framing has already read, when nothing points into it.
    ///
    /// A command's arguments are offsets into this buffer and a half read
    /// command's resume state is another, so this only runs when there is
    /// neither: after a batch, which is where a pipelining connection spends
    /// most of its life.
    fn compact(&mut self) {
        if self.pending > 0 || self.partial.is_some() || self.head == 0 {
            return;
        }
        if self.head == self.buf.len() {
            self.buf.clear();
        } else {
            self.buf.drain(..self.head);
        }
        self.head = 0;
    }
}

/// The engine: connections on one side, the command layer on the other.
///
/// One per shard thread. Everything in it belongs to that thread, including the
/// databases, which is what makes the whole path lock free rather than merely
/// uncontended.
pub struct Wire<S> {
    server: Server,
    sink: S,
    conns: Vec<Conn>,
    /// Connection slots that closed and can be handed out again.
    free: Vec<ConnId>,
    /// The decoder pool.
    argvs: Vec<Argv>,
    spare: Vec<u32>,
    /// Framed and not yet handed to the reactor.
    ready: VecDeque<Cmd>,
    /// Connections this batch wrote to.
    dirty: Vec<ConnId>,
    /// Where a protocol error line is built before it is copied into a reply.
    scratch: Vec<u8>,
    limits: Limits,
    next_id: u64,
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
        Wire {
            server,
            sink,
            conns: Vec::new(),
            free: Vec::new(),
            argvs: Vec::new(),
            spare: Vec::new(),
            ready: VecDeque::with_capacity(BATCH_MAX),
            dirty: Vec::with_capacity(16),
            scratch: Vec::with_capacity(128),
            limits: Limits::default(),
            next_id: 1,
        }
    }

    /// The databases and the numbers `INFO` reports.
    #[must_use]
    pub const fn server(&self) -> &Server {
        &self.server
    }

    /// The same, for a caller that owns both ends.
    pub const fn server_mut(&mut self) -> &mut Server {
        &mut self.server
    }

    /// Where the replies went.
    #[must_use]
    pub const fn sink(&self) -> &S {
        &self.sink
    }

    /// The same, mutably.
    pub const fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Change the protocol limits, which is `proto-max-bulk-len` and friends.
    pub fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Open a connection and give back its id.
    ///
    /// Reuses a closed connection's slot and its two buffers when there is one,
    /// so a server with a churning client population allocates for the high
    /// water mark and not for the total.
    pub fn accept(&mut self) -> ConnId {
        let id = self.next_id;
        self.next_id += 1;
        self.server.stats.clients += 1;
        self.server.stats.connections += 1;

        match self.free.pop() {
            Some(at) => {
                self.conns[at as usize].reset(id);
                at
            }
            None => {
                let conn = Conn::new(id);
                yo_alloc::allow(|| self.conns.push(conn));
                (self.conns.len() - 1) as ConnId
            }
        }
    }

    /// The peer went away.
    ///
    /// Whatever is buffered for it is dropped rather than written, and the slot
    /// comes back as soon as the commands already framed out of its buffer have
    /// run, because those commands' arguments still point into it.
    pub fn hangup(&mut self, conn: ConnId) {
        let c = &mut self.conns[conn as usize];
        if !c.live {
            return;
        }
        c.gone = true;
        c.closing = true;
        if c.pending == 0 {
            self.release(conn);
        }
    }

    /// How many connections are open.
    #[must_use]
    pub fn clients(&self) -> usize {
        self.conns.iter().filter(|c| c.live).count()
    }

    /// Commands framed and waiting for the reactor.
    #[must_use]
    pub fn ready(&self) -> usize {
        self.ready.len()
    }

    /// Connections with a reply that has not gone out yet.
    ///
    /// Non zero means a socket was full and what is left is being held for a
    /// later flush, which a driver waiting on readability needs to know: there
    /// is work here that no incoming byte will ever wake it up for.
    #[must_use]
    pub fn owed(&self) -> usize {
        self.dirty.len()
    }

    /// Decoders in the pool, which is the high water mark of one batch.
    #[must_use]
    pub fn decoders(&self) -> usize {
        self.argvs.len()
    }

    /// Take bytes off a connection and frame whatever commands they complete.
    ///
    /// Anything left over stays in the connection's buffer, half a command
    /// included, so the caller hands over whatever the socket gave it without
    /// looking at it.
    pub fn feed(&mut self, conn: ConnId, bytes: &[u8]) {
        {
            let c = &mut self.conns[conn as usize];
            if !c.live || c.closing {
                return;
            }
            // The buffer is sized for a command at accept time, so this only
            // grows for a client sending a bulk larger than that, which is a
            // real allocation for a real reason.
            yo_alloc::allow(|| c.buf.extend_from_slice(bytes));
        }
        self.frame(conn);
    }

    /// Move as many complete commands as possible out of the read buffer.
    fn frame(&mut self, conn: ConnId) {
        loop {
            let base = self.conns[conn as usize].head;
            let slot = match self.conns[conn as usize].partial.take() {
                Some(slot) => slot,
                None => self.take_decoder(),
            };

            let step = {
                let c = &self.conns[conn as usize];
                self.argvs[slot as usize].decode(&c.buf[base..], &self.limits)
            };

            match step {
                Ok(Step::Command { consumed }) => {
                    self.conns[conn as usize].head += consumed;
                    if self.argvs[slot as usize].is_empty() {
                        // `*0` and a blank inline line: consumed, not answered.
                        self.spare.push(slot);
                    } else {
                        if self.ready.len() == self.ready.capacity() {
                            yo_alloc::allow(|| self.ready.reserve(BATCH_MAX));
                        }
                        self.ready.push_back(Cmd { conn, slot, base });
                        self.conns[conn as usize].pending += 1;
                    }
                }
                Ok(Step::Incomplete) => {
                    // Hold the decoder so the rest of this command resumes
                    // where it stopped instead of being read again from the
                    // front every time more of it arrives.
                    self.conns[conn as usize].partial = Some(slot);
                    break;
                }
                Err(e) => {
                    self.spare.push(slot);
                    let c = &mut self.conns[conn as usize];
                    // Held rather than written, so it lands behind the replies
                    // to the commands that were framed in front of it out of
                    // the same read.
                    c.deferred = Some(e);
                    // Redis closes after a protocol error and so do we: the two
                    // ends no longer agree on where the next command starts.
                    c.closing = true;
                    self.soil(conn);
                    break;
                }
            }
        }
        self.conns[conn as usize].compact();
    }

    /// A decoder from the pool, or a new one the first time round.
    ///
    /// The one from the pool is reset before it goes out, because a decoder can
    /// come back to the pool part way through a command: a protocol error stops
    /// framing where it is, and a connection that hangs up with half a command
    /// in its buffer hands its decoder back too. Either one leaves a resume
    /// point behind, and a resume point is an offset into a buffer that is
    /// about to stop being the same buffer. A decoder taken here is always
    /// starting a command, never continuing one, since a continuation comes off
    /// the connection's own `partial` and never off the pool.
    fn take_decoder(&mut self) -> u32 {
        match self.spare.pop() {
            Some(slot) => {
                self.argvs[slot as usize].reset();
                slot
            }
            None => yo_alloc::allow(|| {
                self.argvs.push(Argv::with_capacity(ARGV_HINT));
                (self.argvs.len() - 1) as u32
            }),
        }
    }

    /// Note that this connection has something to write.
    fn soil(&mut self, conn: ConnId) {
        let c = &mut self.conns[conn as usize];
        if !c.dirty {
            c.dirty = true;
            if self.dirty.len() == self.dirty.capacity() {
                yo_alloc::allow(|| self.dirty.reserve(16));
            }
            self.dirty.push(conn);
        }
    }

    /// Hand the slot and its buffers back.
    fn release(&mut self, conn: ConnId) {
        {
            let c = &mut self.conns[conn as usize];
            if !c.live {
                return;
            }
            if let Some(slot) = c.partial.take() {
                self.spare.push(slot);
            }
            c.live = false;
            c.dirty = false;
            c.out.clear();
            c.buf.clear();
            c.head = 0;
        }
        self.server.stats.clients = self.server.stats.clients.saturating_sub(1);
        self.sink.closed(conn);
        yo_alloc::allow(|| self.free.push(conn));
    }

    /// Move up to `max` framed commands into `into`.
    ///
    /// The reactor wants a batch it owns, and the engine keeps the buffers, so
    /// what crosses between them is this: numbers, no borrows.
    pub fn take_ready(&mut self, into: &mut Vec<Cmd>, max: usize) -> usize {
        let n = max.min(self.ready.len());
        into.extend(self.ready.drain(..n));
        n
    }

    /// Offer one connection's replies to the sink, and say whether it still
    /// owes bytes afterwards.
    fn write_out(&mut self, conn: ConnId) -> bool {
        {
            let c = &self.conns[conn as usize];
            if !c.live {
                return false;
            }
        }
        // A protocol error goes out once everything in front of it has.
        if self.conns[conn as usize].pending == 0
            && let Some(e) = self.conns[conn as usize].deferred.take()
        {
            self.scratch.clear();
            e.write_reply(&mut self.scratch);
            self.conns[conn as usize].out.raw(&self.scratch);
        }

        let taken = {
            let c = &self.conns[conn as usize];
            if c.out.is_empty() {
                0
            } else {
                // One write for the whole batch's replies, never one per reply.
                self.sink.write(conn, c.out.as_slice())
            }
        };

        let c = &mut self.conns[conn as usize];
        if taken >= c.out.len() {
            c.out.clear();
        } else {
            c.out.consume(taken);
        }

        if !c.out.is_empty() {
            return true;
        }
        c.dirty = false;
        if c.closing && c.pending == 0 {
            self.release(conn);
        } else {
            c.compact();
        }
        false
    }

    /// Take a clock reading for the whole batch.
    ///
    /// `04` section 5: once per turn, never per command, so every command in a
    /// batch compares against the same millisecond and two keys written
    /// together expire together.
    pub fn tick(&mut self) {
        self.server.refresh_clock();
    }
}

impl<S: Sink> Engine for Wire<S> {
    type Work = Cmd;

    fn key_hash(&self, cmd: &Cmd) -> Option<u64> {
        let c = &self.conns[cmd.conn as usize];
        let args = Args::new(&self.argvs[cmd.slot as usize], &c.buf[cmd.base..]);
        let spec = lookup(args.name())?;
        if spec.first_key <= 0 {
            return None;
        }
        // The first key only. A command with more than one, which is `MSET` and
        // `MGET`, warms the first and takes the miss on the rest; warming all of
        // them means a hash list per command and that is the batch's own job
        // once multi key commands are worth measuring.
        let key = args.opt(spec.first_key as usize)?;
        Some(Strings::hash_of(key))
    }

    fn prefetch(&self, cmd: &Cmd, hash: u64) {
        let db = self.conns[cmd.conn as usize].session.db();
        self.server.db_ref(db).prefetch(hash);
    }

    fn run(&mut self, cmd: Cmd, _hash: Option<u64>) -> yo_reactor::Flow {
        let flow = {
            let c = &mut self.conns[cmd.conn as usize];
            c.pending -= 1;
            if c.gone || c.skip {
                // Nobody to answer, or nobody who should be. The decoder still
                // has to come back and the slot still has to be released, which
                // is why this is not an early return.
                Flow::Continue
            } else {
                let args = Args::new(&self.argvs[cmd.slot as usize], &c.buf[cmd.base..]);
                execute(&mut self.server, &mut c.session, args, &mut c.out)
            }
        };

        self.spare.push(cmd.slot);
        let c = &self.conns[cmd.conn as usize];
        if c.gone {
            if c.pending == 0 {
                self.release(cmd.conn);
            }
        } else {
            if flow == Flow::Close {
                let c = &mut self.conns[cmd.conn as usize];
                c.closing = true;
                // Anything the client pipelined behind the `QUIT` was sent
                // before it knew the answer, and running it would be acting on
                // a connection that has already been said goodbye to.
                c.skip = true;
            }
            self.soil(cmd.conn);
        }
        yo_reactor::Flow::Next
    }

    fn flush(&mut self) {
        // Taken and put back so the loop below can reach the rest of the
        // engine. The capacity comes back with it, so this is not an
        // allocation.
        let mut dirty = core::mem::take(&mut self.dirty);
        let mut at = 0;
        while at < dirty.len() {
            let conn = dirty[at];
            let owed = self.write_out(conn);
            if owed {
                // The socket was full. The connection stays on the list with
                // what is left of its reply, and the next flush offers it
                // again, which is the whole of the backpressure story here.
                at += 1;
            } else {
                dirty.swap_remove(at);
            }
        }
        self.dirty = dirty;
    }

    fn maintain(&mut self, budget: &mut yo_reactor::Budget) {
        // The clock is the first thing the maintenance slice does, because
        // everything else in it compares against a time.
        if budget.spend(1) {
            self.tick();
        }
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
        ran += reactor.execute_all(batch.drain(..));
        reactor.engine_mut().flush();
    }
    // Once more, for a connection with something to say and nothing to run: a
    // protocol error, or a socket that was full the last time round.
    reactor.engine_mut().flush();
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
