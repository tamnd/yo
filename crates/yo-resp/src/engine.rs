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
use yo_kv::Keyspace;

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
    /// This client is parked on a blocking command.
    ///
    /// While it is set, framing stops: whatever the client pipelined behind its
    /// `BLPOP` stays in the read buffer unread, which is what a client waiting
    /// for an answer means and is what Redis does with the same bytes.
    blocked: bool,
    /// Commands framed before it blocked and not run yet.
    ///
    /// A batch is framed before any of it runs, so a `BLPOP` can be the first of
    /// sixty four commands and the other sixty three are already on their way to
    /// the reactor when it parks. They come back here and go to the front of the
    /// queue when the client wakes up, in the order they arrived.
    ///
    /// They are still counted in `pending`, which is what stops the read buffer
    /// being compacted under the offsets they hold.
    parked: Vec<Cmd>,
    /// What the two buffers were holding the last time anybody counted.
    ///
    /// The connection's share of `INFO memory`, kept here so that reporting it
    /// is a subtraction against this rather than a walk over every connection.
    held: usize,
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
            blocked: false,
            parked: Vec::new(),
            held: 0,
        })
    }

    /// What the two buffers cost the process, which is the room they are
    /// holding and not the bytes in use: both keep their capacity between
    /// batches on purpose.
    fn size(&self) -> usize {
        self.buf.capacity() + self.out.capacity()
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
        self.blocked = false;
        // The room it took stays, the way the two buffers' does.
        self.parked.clear();
    }

    /// Drop what the framing has already read, when nothing points into it.
    ///
    /// A framed command's arguments are offsets from the front of this buffer,
    /// so this waits for the batch to run. After a batch is where a pipelining
    /// connection spends most of its life, so that is not much of a wait.
    ///
    /// A half read command is not in the way. Its decoder was handed
    /// `buf[head..]` and every offset it kept is from the front of that slice,
    /// and `head` does not move until the command is complete, so the bytes it
    /// is waiting on are exactly the bytes this keeps. They arrive at the front
    /// instead of at `head` and the decoder cannot tell the difference.
    ///
    /// Waiting for it anyway is what made a read buffer grow to everything the
    /// connection had ever sent. The framing loop only ever stops on an
    /// incomplete command, and a buffer that ends on a command boundary gives
    /// one of those on the next turn round: an empty slice, nothing decoded,
    /// `Step::Incomplete`. So a connection that is exactly up to date always had
    /// a decoder parked on it, this always returned early, and `head` walked
    /// forward with the bytes behind it kept forever. Measured on server3, four
    /// connections sending 100000 sets each held 16 MiB of read buffer apiece,
    /// and fifty connections sending 8000 each held 1 MiB apiece: in both cases
    /// every byte the connection had ever sent.
    fn compact(&mut self) {
        if self.pending > 0 || self.head == 0 {
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

        let at = match self.free.pop() {
            Some(at) => {
                // A reused slot keeps its buffers, so what it holds is already
                // counted and this only puts the id back in service.
                self.conns[at as usize].reset(id);
                at
            }
            None => {
                let conn = Conn::new(id);
                yo_alloc::allow(|| self.conns.push(conn));
                (self.conns.len() - 1) as ConnId
            }
        };
        self.note_size(at);
        at
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
        // A parked client holds its own commands, and those commands are what
        // `pending` counts, so leaving it parked here would leave the slot owed
        // to a connection that is never going to be answered. They go back to
        // the queue and run as the no-ops a gone connection's commands are.
        if c.blocked {
            self.unpark(conn);
        }
        if self.conns[conn as usize].pending == 0 {
            self.release(conn);
        }
    }

    /// The client is not waiting any more: give it back its commands.
    ///
    /// The ones it had already sent go to the front of the queue in the order
    /// they arrived, ahead of anything any other connection has waiting, because
    /// they were framed before any of that was. Then framing starts again on
    /// whatever arrived while it was parked.
    fn unpark(&mut self, conn: ConnId) {
        let mut parked = {
            let c = &mut self.conns[conn as usize];
            c.blocked = false;
            core::mem::take(&mut c.parked)
        };
        // Back to front, since each one goes on the front.
        while let Some(cmd) = parked.pop() {
            if self.ready.len() == self.ready.capacity() {
                yo_alloc::allow(|| self.ready.reserve(BATCH_MAX));
            }
            self.ready.push_front(cmd);
        }
        // Empty now, and back where it lives so its room is not paid for twice.
        self.conns[conn as usize].parked = parked;
        if !self.conns[conn as usize].closing {
            self.frame(conn);
        }
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
        while at < self.server.waiters().len() {
            let p = self.server.waiters().at(at);
            {
                let c = &self.conns[p.conn as usize];
                // The slot is reused and the client id is not. `release`
                // forgets waiters, so this should never fire; it is here
                // because being wrong about it writes a reply into somebody
                // else's socket rather than dropping one.
                if !c.live || c.session.id() != p.client {
                    self.server.waiters_mut().drop_at(at);
                    continue;
                }
            }
            // The engine cannot reach the databases and the server cannot reach
            // the connections, so the two halves are taken apart here and the
            // one buffer this waiter needs is handed over.
            let served = {
                let Wire { server, conns, .. } = self;
                server.serve_waiter(at, now, &mut conns[p.conn as usize].out)
            };
            if served {
                self.server.waiters_mut().drop_at(at);
                self.unpark(p.conn);
                self.soil(p.conn);
            } else {
                at += 1;
            }
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

    /// What every connection's read and reply buffers are holding.
    ///
    /// The walk is fine here because this is a test and a report, and the
    /// number the running server uses is the one kept by `note_size`.
    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.conns.iter().map(Conn::size).sum()
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
        self.note_size(conn);
    }

    /// Tell the server what this connection's buffers are holding now, if it
    /// has changed since the last time anybody asked.
    ///
    /// Once per read and once per flush, which is where a buffer can grow, and
    /// two loads and a compare when nothing has moved. The alternative is a
    /// walk over every connection on a turn of the loop, which puts the cost of
    /// a report nobody has asked for on the command path.
    fn note_size(&mut self, conn: ConnId) {
        let c = &mut self.conns[conn as usize];
        let now = c.size();
        if now == c.held {
            return;
        }
        let delta = now as isize - c.held as isize;
        c.held = now;
        self.server.note_conn_bytes(delta);
    }

    /// Move as many complete commands as possible out of the read buffer.
    ///
    /// Nothing at all while the client is parked. The bytes stay where they are
    /// and `head` does not move, so a client that pipelines `BLPOP` and then
    /// `PING` gets the `PING` answered when the `BLPOP` is, and in that order.
    fn frame(&mut self, conn: ConnId) {
        if self.conns[conn as usize].blocked {
            return;
        }
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
                // Every slot handed out here comes back to `spare` exactly
                // once, so `spare` never holds more than `argvs` has slots.
                // Sizing it here means the pushes that give a slot back never
                // touch the allocator, and those are on the command path while
                // this is not: a decoder is made once per depth of pipelining
                // the connection has ever reached. `spare` is empty right now,
                // which is why we are down here at all.
                self.spare.reserve(self.argvs.len());
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
            c.blocked = false;
            c.out.clear();
            c.buf.clear();
            c.head = 0;
        }
        // Before the slot goes back, because the slot is handed out again and a
        // waiter on a client that has gone would then be a waiter pointing at
        // somebody else's connection. The id is what makes it findable and the
        // id is about to stop being this connection's.
        let client = self.conns[conn as usize].session.id();
        self.server.waiters_mut().forget(client);
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
        self.note_size(conn);
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
        self.server.compact_step()
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
        Some(Keyspace::hash_of(key))
    }

    fn prefetch(&self, cmd: &Cmd, hash: u64) {
        let db = self.conns[cmd.conn as usize].session.db();
        self.server.db_ref(db).prefetch(hash);
    }

    fn run(&mut self, cmd: Cmd, _hash: Option<u64>) -> yo_reactor::Flow {
        // Framed with the batch that blocked, so it is a command the client sent
        // before it knew it would be waiting. It keeps its decoder and it keeps
        // its place in `pending`, which is what stops the buffer it points into
        // being compacted while it waits.
        if self.conns[cmd.conn as usize].blocked {
            yo_alloc::allow(|| self.conns[cmd.conn as usize].parked.push(cmd));
            return yo_reactor::Flow::Next;
        }

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
            match flow {
                Flow::Close => {
                    let c = &mut self.conns[cmd.conn as usize];
                    c.closing = true;
                    // Anything the client pipelined behind the `QUIT` was sent
                    // before it knew the answer, and running it would be acting
                    // on a connection that has already been said goodbye to.
                    c.skip = true;
                    self.soil(cmd.conn);
                }
                // Nothing was written, so there is nothing to flush and no
                // reason to put this connection on the dirty list. The waiter
                // carries the slot from here on, and it needs to know which one:
                // the command layer only ever saw the client id.
                Flow::Block => {
                    self.conns[cmd.conn as usize].blocked = true;
                    let client = self.conns[cmd.conn as usize].session.id();
                    self.server.waiters_mut().bind(client, cmd.conn);
                }
                Flow::Continue => self.soil(cmd.conn),
            }
        }

        // After each command and not once per batch. A client blocked on two
        // keys and woken by `RPUSH b` then `RPUSH a` in one pipeline has to
        // answer with `b`, because that is the push that was in front of it, and
        // it can only do that if it was served in between the two.
        if !self.server.waiters().is_empty() {
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
        if !self.server.waiters().is_empty() {
            self.server.refresh_clock();
            self.serve_waiters();
        }

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
        assert_eq!(r.engine().server().waiters().len(), 0);
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
        assert_eq!(r.engine().server().waiters().len(), 1);

        r.engine_mut().feed(b, &wire(&[b"RPUSH", b"q", b"one"]));
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().sink().sent(a), b"*2\r\n$1\r\nq\r\n$3\r\none\r\n");
        // The push still reports the length it made, even though the element was
        // gone again before the reply was written.
        assert_eq!(r.engine().sink().sent(b), b":1\r\n");
        assert_eq!(r.engine().server().waiters().len(), 0);
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
        assert_eq!(r.engine().server().waiters().len(), 1, "still waiting");
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
        assert_eq!(r.engine().server().waiters().len(), 2);

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
        assert_eq!(r.engine().server().waiters().len(), 0);
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
        assert_eq!(r.engine().server().waiters().len(), 2);

        r.engine_mut().feed(c, &wire(&[b"RPUSH", b"x", b"chain"]));
        pump(&mut r, &mut batch);

        assert_eq!(r.engine().sink().sent(a), b"$5\r\nchain\r\n");
        assert_eq!(
            r.engine().sink().sent(b),
            b"*2\r\n$1\r\ny\r\n$5\r\nchain\r\n"
        );
        assert_eq!(r.engine().server().waiters().len(), 0);
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
        assert_eq!(r.engine().server().waiters().len(), 0);
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
        assert_eq!(r.engine().server().waiters().len(), 1);

        r.engine_mut().hangup(a);
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().server().waiters().len(), 0);
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
