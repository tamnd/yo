//! What one thread owns: the connections, their buffers and the framing.
//!
//! A server on several threads is two halves that have to be told apart before
//! either of them can move. One half is per connection and belongs to whichever
//! thread accepted it: the read buffer, the decoder holding a half read command,
//! the session, the reply buffer and the queue of commands framed and not yet
//! run. The other half is the keyspace, which every thread reaches and which is
//! behind the stripes. This module is the first half, and the line is drawn by
//! the compiler rather than by a comment: nothing in this file can name a
//! [`Server`], because it does not import one.
//!
//! [`Wire`] is where the two meet. Everything that needs both, which is running
//! a command, answering a blocked client and forgetting a client that has gone,
//! is a method there and calls into here for the connection half. Everything
//! that needs only the connections is a method here, which is why framing can be
//! tested against a [`Front`] with no database anywhere in the test.
//!
//! [`Server`]: crate::dispatch::Server
//! [`Wire`]: crate::engine::Wire

use std::collections::VecDeque;

use yo_reactor::BATCH_MAX;

use crate::dispatch::table::lookup_index;
use crate::dispatch::{Args, Session};
use crate::engine::{ConnId, Sink};
use crate::error::ProtocolError;
use crate::proto::{Limits, Proto};
use crate::reply::Out;
use crate::request::{Argv, Step};

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
/// of these while the front keeps ownership of every buffer they point into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cmd {
    pub(crate) conn: ConnId,
    pub(crate) slot: u32,
    pub(crate) base: usize,
    /// Which command this is, as a position in the command table.
    ///
    /// Resolved once, here, because the name is otherwise looked up twice more
    /// on the way to running it: once to work out which key to prefetch and once
    /// to dispatch. A position rather than a reference because this struct is
    /// queued by the thousand and two bytes is what it costs.
    ///
    /// Past the end of the table for a name that is no command, which needs no
    /// flag of its own and no `Option`, because that is what the lookup already
    /// answers and what the dispatcher already has a reply for.
    pub(crate) spec: u16,
}

impl Cmd {
    /// The connection this command arrived on.
    #[must_use]
    pub const fn conn(&self) -> ConnId {
        self.conn
    }
}

/// What one connection's replies did on their way to the socket.
pub(crate) enum Wrote {
    /// The socket took less than was offered, so what is left is held for the
    /// next flush and the connection stays on the dirty list.
    Owed,
    /// Everything went out and the connection is still open.
    Done,
    /// Everything went out and the connection ended with it. The client id is
    /// the one thing the server has to hear about, because a waiter is found by
    /// it and the slot is about to belong to somebody else.
    Ended(u64),
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
        // The protocol lives in the reply buffer and the reply buffer is kept,
        // so it has to be put back by hand. Without this a client that opened a
        // connection into a slot the last client had spoken RESP3 on would be
        // answered in RESP3 without ever sending `HELLO`, which is a nil it
        // cannot parse on the first `GET` that misses.
        self.out.set_proto(Proto::Resp2);
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

/// The connection side of the server, and all of it belongs to one thread.
///
/// Connections, their buffers, the decoder pool, the framing and the queue of
/// work it produces. There is one of these per I/O thread and they share
/// nothing, which is why none of it is behind a lock and none of it is atomic.
pub(crate) struct Front<S> {
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
    /// How much the buffers have grown or shrunk since anybody last asked.
    ///
    /// `INFO memory` reports what every connection is holding and that total
    /// lives on the server, which this side cannot reach. So the change is kept
    /// here and taken by [`Wire`] at the end of whatever call made it, which is
    /// as timely as reporting it on the spot and does not put the server on the
    /// other end of a framing call.
    ///
    /// [`Wire`]: crate::engine::Wire
    moved: isize,
}

impl<S: Sink> Front<S> {
    /// A front with no connections and nothing pooled.
    pub(crate) fn new(sink: S) -> Front<S> {
        Front {
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
            moved: 0,
        }
    }

    /// Where the replies went.
    pub(crate) const fn sink(&self) -> &S {
        &self.sink
    }

    /// The same, mutably.
    pub(crate) const fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Change the protocol limits, which is `proto-max-bulk-len` and friends.
    pub(crate) fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Open a connection and give back its slot.
    ///
    /// Reuses a closed connection's slot and its two buffers when there is one,
    /// so a server with a churning client population allocates for the high
    /// water mark and not for the total.
    pub(crate) fn open(&mut self) -> ConnId {
        let id = self.next_id;
        self.next_id += 1;

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

    /// Take bytes off a connection and frame whatever commands they complete.
    ///
    /// Anything left over stays in the connection's buffer, half a command
    /// included, so the caller hands over whatever the socket gave it without
    /// looking at it.
    pub(crate) fn feed(&mut self, conn: ConnId, bytes: &[u8]) {
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

    /// Note what this connection's buffers are holding now, if it has changed
    /// since the last time anybody asked.
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
        self.moved += delta;
    }

    /// How much the buffers have moved since this was last called.
    pub(crate) fn buffer_delta(&mut self) -> isize {
        core::mem::take(&mut self.moved)
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
                        // Here and not later, because the name is in front of
                        // the argument list that was just decoded and this is
                        // the last place that holds both it and nothing else to
                        // do. Everything downstream takes the number.
                        let spec = {
                            let c = &self.conns[conn as usize];
                            let args = Args::new(&self.argvs[slot as usize], &c.buf[base..]);
                            lookup_index(args.name())
                        };
                        self.ready.push_back(Cmd {
                            conn,
                            slot,
                            base,
                            spec,
                        });
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
    pub(crate) fn soil(&mut self, conn: ConnId) {
        let c = &mut self.conns[conn as usize];
        if !c.dirty {
            c.dirty = true;
            if self.dirty.len() == self.dirty.capacity() {
                yo_alloc::allow(|| self.dirty.reserve(16));
            }
            self.dirty.push(conn);
        }
    }

    /// Hand the slot and its buffers back, and say which client has gone.
    ///
    /// `None` for a slot that was already closed. The id is what the server
    /// finds a waiter by, and the caller forgets it before anything else runs,
    /// because this slot is on the free list from here and the next accept
    /// hands it to somebody else.
    pub(crate) fn close(&mut self, conn: ConnId) -> Option<u64> {
        {
            let c = &mut self.conns[conn as usize];
            if !c.live {
                return None;
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
        let client = self.conns[conn as usize].session.id();
        self.sink.closed(conn);
        yo_alloc::allow(|| self.free.push(conn));
        Some(client)
    }

    /// Move up to `max` framed commands into `into`.
    ///
    /// The reactor wants a batch it owns, and the front keeps the buffers, so
    /// what crosses between them is this: numbers, no borrows.
    pub(crate) fn take_ready(&mut self, into: &mut Vec<Cmd>, max: usize) -> usize {
        let n = max.min(self.ready.len());
        into.extend(self.ready.drain(..n));
        n
    }

    /// Offer one connection's replies to the sink.
    pub(crate) fn write_out(&mut self, conn: ConnId) -> Wrote {
        {
            let c = &self.conns[conn as usize];
            if !c.live {
                return Wrote::Done;
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
            return Wrote::Owed;
        }
        c.dirty = false;
        let ending = c.closing && c.pending == 0;
        if ending {
            if let Some(client) = self.close(conn) {
                return Wrote::Ended(client);
            }
        } else {
            c.compact();
            self.note_size(conn);
        }
        Wrote::Done
    }

    /// The dirty list, taken so the caller can walk it and reach the rest of
    /// the front at the same time. The capacity comes back with it, so this is
    /// not an allocation.
    pub(crate) fn take_dirty(&mut self) -> Vec<ConnId> {
        core::mem::take(&mut self.dirty)
    }

    /// The dirty list, given back with whatever is still owed on it.
    pub(crate) fn give_dirty(&mut self, dirty: Vec<ConnId>) {
        self.dirty = dirty;
    }

    /// How many connections are open.
    pub(crate) fn clients(&self) -> usize {
        self.conns.iter().filter(|c| c.live).count()
    }

    /// Commands framed and waiting for the reactor.
    pub(crate) fn ready(&self) -> usize {
        self.ready.len()
    }

    /// Connections with a reply that has not gone out yet.
    pub(crate) fn owed(&self) -> usize {
        self.dirty.len()
    }

    /// Decoders in the pool, which is the high water mark of one batch.
    pub(crate) fn decoders(&self) -> usize {
        self.argvs.len()
    }

    /// What every connection's read and reply buffers are holding.
    ///
    /// The walk is fine here because this is a test and a report, and the
    /// number the running server uses is the one kept by `note_size`.
    pub(crate) fn buffer_bytes(&self) -> usize {
        self.conns.iter().map(Conn::size).sum()
    }

    /// Whether the slot is open.
    pub(crate) fn live(&self, conn: ConnId) -> bool {
        self.conns[conn as usize].live
    }

    /// Whether the peer has gone.
    pub(crate) fn gone(&self, conn: ConnId) -> bool {
        self.conns[conn as usize].gone
    }

    /// Commands framed out of this connection's buffer and not yet run.
    pub(crate) fn pending(&self, conn: ConnId) -> u32 {
        self.conns[conn as usize].pending
    }

    /// Whether this client is parked on a blocking command.
    pub(crate) fn blocked(&self, conn: ConnId) -> bool {
        self.conns[conn as usize].blocked
    }

    /// The client id, which is what the server knows a connection by.
    pub(crate) fn client(&self, conn: ConnId) -> u64 {
        self.conns[conn as usize].session.id()
    }

    /// The database this connection has selected.
    pub(crate) fn db(&self, conn: ConnId) -> usize {
        self.conns[conn as usize].session.db()
    }

    /// Whether this slot is still the client the server thinks it is.
    ///
    /// A slot is reused and a client id is not, so a waiter that named a client
    /// is only about this connection while both agree.
    pub(crate) fn answers(&self, conn: ConnId, client: u64) -> bool {
        let c = &self.conns[conn as usize];
        c.live && c.session.id() == client
    }

    /// Where a reply for this connection goes.
    pub(crate) fn out(&mut self, conn: ConnId) -> &mut Out {
        &mut self.conns[conn as usize].out
    }

    /// The peer went away.
    pub(crate) fn mark_gone(&mut self, conn: ConnId) {
        let c = &mut self.conns[conn as usize];
        c.gone = true;
        c.closing = true;
    }

    /// The client said goodbye.
    ///
    /// Anything it pipelined behind the `QUIT` was sent before it knew the
    /// answer, and running it would be acting on a connection that has already
    /// been said goodbye to.
    pub(crate) fn quit(&mut self, conn: ConnId) {
        let c = &mut self.conns[conn as usize];
        c.closing = true;
        c.skip = true;
    }

    /// The client is waiting on a blocking command.
    pub(crate) fn block(&mut self, conn: ConnId) {
        self.conns[conn as usize].blocked = true;
    }

    /// Hold a command that was framed with the batch that blocked.
    pub(crate) fn park(&mut self, conn: ConnId, cmd: Cmd) {
        yo_alloc::allow(|| self.conns[conn as usize].parked.push(cmd));
    }

    /// The client is not waiting any more: give it back its commands.
    ///
    /// The ones it had already sent go to the front of the queue in the order
    /// they arrived, ahead of anything any other connection has waiting, because
    /// they were framed before any of that was. Then framing starts again on
    /// whatever arrived while it was parked.
    pub(crate) fn unpark(&mut self, conn: ConnId) {
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

    /// A command is off the queue: take it out of the count, and say whether it
    /// should run at all.
    ///
    /// It should not when the peer has gone or has said goodbye, and the answer
    /// is `false` rather than an early return because the decoder still has to
    /// come back and the slot still has to be released.
    pub(crate) fn start(&mut self, cmd: &Cmd) -> bool {
        let c = &mut self.conns[cmd.conn as usize];
        c.pending -= 1;
        !(c.gone || c.skip)
    }

    /// The three things running a command needs from this side: the arguments,
    /// the session they run against, and where the reply goes.
    pub(crate) fn parts(&mut self, cmd: &Cmd) -> (Args<'_>, &mut Session, &mut Out) {
        let c = &mut self.conns[cmd.conn as usize];
        let args = Args::new(&self.argvs[cmd.slot as usize], &c.buf[cmd.base..]);
        (args, &mut c.session, &mut c.out)
    }

    /// The arguments alone, for a caller that is only reading them.
    pub(crate) fn args(&self, cmd: &Cmd) -> Args<'_> {
        let c = &self.conns[cmd.conn as usize];
        Args::new(&self.argvs[cmd.slot as usize], &c.buf[cmd.base..])
    }

    /// The command is finished with its decoder.
    pub(crate) fn done(&mut self, cmd: &Cmd) {
        self.spare.push(cmd.slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Recorder;

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

    /// A front and one connection on it. No server anywhere, which is the
    /// point: framing is this side's work alone.
    fn front() -> (Front<Recorder>, ConnId) {
        let mut f = Front::new(Recorder::new());
        let conn = f.open();
        (f, conn)
    }

    #[test]
    fn a_pipelined_read_frames_every_command_in_it() {
        let (mut f, conn) = front();
        let mut bytes = wire(&[b"SET", b"k", b"v"]);
        bytes.extend_from_slice(&wire(&[b"GET", b"k"]));
        f.feed(conn, &bytes);

        let mut batch = Vec::new();
        assert_eq!(f.take_ready(&mut batch, 64), 2);
        assert_eq!(f.args(&batch[0]).name(), b"SET");
        assert_eq!(f.args(&batch[1]).name(), b"GET");
        assert_eq!(f.pending(conn), 2);
    }

    #[test]
    fn a_command_split_across_reads_is_framed_once_it_is_whole() {
        let (mut f, conn) = front();
        let bytes = wire(&[b"SET", b"k", b"v"]);
        let (head, tail) = bytes.split_at(9);

        f.feed(conn, head);
        let mut batch = Vec::new();
        assert_eq!(f.take_ready(&mut batch, 64), 0);

        f.feed(conn, tail);
        assert_eq!(f.take_ready(&mut batch, 64), 1);
        assert_eq!(f.args(&batch[0]).name(), b"SET");
    }

    #[test]
    fn a_protocol_error_stops_the_framing_and_closes_the_connection() {
        let (mut f, conn) = front();
        f.feed(conn, b"*x\r\n");
        assert_eq!(f.take_ready(&mut Vec::new(), 64), 0);
        assert_eq!(f.owed(), 1);

        // Nothing is owed to the client afterwards and the slot has gone back,
        // which is what a closed connection means on this side.
        assert!(matches!(f.write_out(conn), Wrote::Ended(_)));
        assert!(!f.live(conn));
        assert!(f.sink().sent(conn).starts_with(b"-ERR"));
    }

    #[test]
    fn a_closed_slot_is_handed_out_again_with_its_buffers() {
        let (mut f, conn) = front();
        f.feed(conn, &wire(&[b"PING"]));
        let held = f.buffer_bytes();
        assert_eq!(f.close(conn), Some(1));

        let next = f.open();
        assert_eq!(next, conn, "the slot comes back");
        assert_eq!(f.client(next), 2, "the client id does not");
        assert_eq!(f.buffer_bytes(), held, "and neither buffer was given up");
    }

    #[test]
    fn the_buffers_are_reported_as_they_move_and_only_once() {
        let (mut f, conn) = front();
        assert!(f.buffer_delta() > 0, "accept made two buffers");
        assert_eq!(f.buffer_delta(), 0, "and nobody is told about them twice");

        f.feed(conn, &wire(&[b"PING"]));
        assert_eq!(f.buffer_delta(), 0, "a command that fits moves nothing");
    }
}
