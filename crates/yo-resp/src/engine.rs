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
use crate::dispatch::{self, Flow, Parked, Server};
use crate::front::{Front, Wrote};
use crate::proto::Limits;
use yo_kv::Keyspace;

pub use crate::front::Cmd;

/// Which connection. An index, reused after a connection closes.
pub type ConnId = u32;

/// Keys a housekeeping call is allowed to look at while hunting dead ones.
///
/// The same number the loop's maintenance slice gets, because it buys the same
/// thing: the sweep walks twenty keys at a time, so this is a couple of hundred
/// draws in the worst case and one comparison in the common one, where no key
/// in the database carries a deadline at all.
const SWEEP_LOOKS: usize = yo_reactor::MAINTENANCE_UNITS as usize;

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
    /// This thread's parked clients, copied out of the shared list.
    ///
    /// Here rather than in `serve_waiters` so that a server with blocked
    /// clients on it does not allocate once a batch. It is empty between
    /// batches and it is only ever this thread's, like everything else on this
    /// side of the engine.
    parked: Vec<Parked>,
    /// Messages published for this thread's connections, copied out of the
    /// mailbox.
    ///
    /// Here for the reason `parked` is here: a server with subscribers on it
    /// should not allocate a vector once a batch to drain into. It is empty
    /// between batches.
    post: Vec<dispatch::Envelope>,
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
            parked: Vec::new(),
            post: Vec::new(),
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
        let at = self.front.open(self.server.next_client());
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

    /// Answer everybody this thread can answer, and let go of everybody whose
    /// deadline has passed.
    ///
    /// The walk is over the waiter list rather than over the connections, so it
    /// costs what blocking costs and not what the server costs. Every caller
    /// checks that somebody is parked before calling, which is the load and the
    /// branch a server with nobody blocked pays.
    ///
    /// Only this thread's waiters, because a reply goes into a buffer this
    /// thread owns and another thread's waiter is another thread's to answer.
    /// The list is copied out under the lock and then let go of, so the work of
    /// answering does not hold up a thread trying to park a client.
    fn serve_waiters(&mut self) {
        let now = self.server.now_ms();
        let mine = self.server.my_slot();
        self.server.waiters().mine(mine, &mut self.parked);
        for at in 0..self.parked.len() {
            let p = self.parked[at];
            // The slot is reused and the client id is not. `release` forgets
            // waiters, so this should never fire; it is here because being
            // wrong about it writes a reply into somebody else's socket rather
            // than dropping one.
            if !self.front.answers(p.conn, p.client) {
                self.server.forget_waiters(p.client);
                continue;
            }
            // The front cannot reach the databases and the server cannot reach
            // the connections, so the two halves are taken apart here and the
            // one buffer this waiter needs is handed over.
            let served = {
                let Wire { server, front, .. } = self;
                server.serve_waiter(p.client, now, front.out(p.conn))
            };
            if served {
                self.server.forget_waiters(p.client);
                self.front.unpark(p.conn);
                self.front.soil(p.conn);
            }
        }
        self.parked.clear();
    }

    /// Write out everything published for this thread's connections.
    ///
    /// The mailbox is emptied under its lock and then let go of, so a thread
    /// rendering a thousand messages is not holding up the publishers filling
    /// its box. The client id on each envelope is checked against the slot
    /// because a slot is reused and an id is not, which is the same guard the
    /// waiter list uses and for the same reason: being wrong here writes into
    /// somebody else's socket rather than dropping a message.
    fn deliver(&mut self) {
        // Taken and put back so the loop can reach the front, the way the dirty
        // list is. The capacity comes back with it.
        let mut post = core::mem::take(&mut self.post);
        self.server.take_mail(&mut post);
        for env in post.drain(..) {
            let conn = env.conn();
            if !self.front.answers(conn, env.client()) {
                continue;
            }
            env.write(self.front.out(conn));
            self.front.soil(conn);
        }
        self.post = post;
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

    /// Clients of this thread's that are blocked on a key.
    ///
    /// The other thing a driver waiting on readability needs to know, and for
    /// the same reason `owed` is: there is work here that no incoming byte will
    /// wake it for. A blocked client is answered by a write another thread made
    /// or by its own deadline passing, and neither of those is a byte arriving
    /// on this thread's poller, so a driver that reads this keeps its wait short
    /// while anybody is waiting on it.
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.server.parked_here()
    }

    /// Mail waiting for this thread, plus subscribers of its own that mail
    /// could arrive for.
    ///
    /// The third thing a driver waiting on readability needs to know, and for
    /// the reason the other two are: a published message is a write another
    /// thread made and no byte arriving here will wake this thread for it. So a
    /// thread that has a subscriber keeps its wait short, and one that has none
    /// is not affected.
    #[must_use]
    pub fn posted(&self) -> usize {
        self.server.posted()
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
        // Before the slot goes back, because the watches this connection took
        // are rows on the server and the session that names them is about to be
        // reused by whoever gets the slot next.
        if let Some(session) = self.front.session_mut(conn) {
            dispatch::forget_session(&self.server, session);
        }
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
    /// That is the dead keys and then one segment of arena compaction at most,
    /// which between them are what stop a server that rewrites the same keys,
    /// or writes them under a deadline and never reads them back, from holding
    /// every version of everything it has ever been sent. It is separate from
    /// [`Wire::tick`] because the clock has to move before a batch runs and this
    /// does not: it can wait until the replies are out, and the driver decides
    /// when that is.
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
        // The keys whose deadline has passed with nobody there to read them
        // back. A slice's worth at most and gated to once a millisecond inside,
        // so a driver that calls this after every batch does not turn a busy
        // server into a server that spends its time sampling.
        self.server.expire_slice(SWEEP_LOOKS);
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
            let Wire { front, server, .. } = self;
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
        if self.server.parked_here() != 0 {
            self.serve_waiters();
        }
        yo_reactor::Flow::Next
    }

    fn flush(&mut self) {
        // The deadline sweep, and it is here because this is the one thing the
        // driver calls on a turn that ran nothing at all. A client whose timeout
        // passes while the server is idle is answered within the loop's idle
        // wait, which the loop shortens to a millisecond on a thread that has
        // somebody waiting. That is finer than the 10hz Redis checks its own
        // blocked clients at.
        //
        // This thread's count and not the server's, because the sweep can only
        // answer this thread's waiters, so on any other thread it is a lock
        // taken to find nothing.
        if self.server.parked_here() != 0 {
            self.server.refresh_clock();
            self.serve_waiters();
        }

        // Then the published messages, before the write out below and after
        // everything this batch answered, which is the order a client that
        // publishes to itself sees on a real server: the count first and the
        // message second, checked on the wire against 8.10.1.
        if self.server.mail_here() != 0 {
            self.deliver();
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
    // Once for a turn that ran nothing at all, which is where a server that has
    // gone quiet catches up on what the last busy turn left behind.
    reactor.engine_mut().maintain();
    // Then once more for a connection with something to say and nothing to run:
    // a protocol error, or a socket that was full the last time round, or a
    // subscriber the housekeeping above owes the news that a key it was told to
    // watch reached its deadline. That last one is why the flush is after the
    // call rather than before it: an idle server turns every twenty
    // milliseconds, and news that waits for the next turn is news that arrives
    // twenty milliseconds after the thing it is about.
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
    ///
    /// The server is told it will have two threads before either starts, the
    /// way `yodb serve` tells it. Without that it has one set of counters and
    /// both threads land on it, which is the wrap round `Server::mine_at`
    /// documents and which loses counts: a bump is a load and a store rather
    /// than a fetch and add, because the fast path is one thread writing its
    /// own set and paying for a locked instruction on every command to make a
    /// shared set exact would be paying it on the path that is never shared.
    /// Miri found this by running the two threads far enough apart to lose one,
    /// which a real machine does rarely enough to have passed here for months.
    #[test]
    fn two_threads_write_into_one_server() {
        const EACH: usize = 200;

        let mut server = Server::new();
        server.set_threads(2);
        let first = Wire::with_server(server, Recorder::new());
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

    /// A blocked client is answered into a buffer one thread owns, so it is
    /// that thread's to answer and nobody else's to throw away.
    #[test]
    fn a_waiter_belongs_to_the_thread_that_parked_it() {
        let mut server = Server::new();
        server.set_threads(2);
        let first = Wire::with_server(server, Recorder::new());
        let second = Wire::over(first.shared(), Recorder::new());
        let server = first.shared();

        let parked = std::sync::Barrier::new(2);
        let swept = std::sync::Barrier::new(2);

        std::thread::scope(|s| {
            let (parked, swept) = (&parked, &swept);
            s.spawn(move || {
                let mut r = Reactor::inline(first);
                let mut batch = Vec::new();
                let conn = r.engine_mut().accept();
                r.engine_mut().feed(conn, &wire(&[b"BLPOP", b"a", b"0"]));
                pump(&mut r, &mut batch);
                parked.wait();

                // Turns with nothing on them, each of which walks a list whose
                // one other entry belongs to the thread next door.
                for _ in 0..50 {
                    pump(&mut r, &mut batch);
                }
                swept.wait();
                assert!(r.engine().sink().sent(conn).is_empty(), "nothing to say");
            });
            s.spawn(move || {
                let mut r = Reactor::inline(second);
                let mut batch = Vec::new();
                let conn = r.engine_mut().accept();
                r.engine_mut().feed(conn, &wire(&[b"BLPOP", b"b", b"0"]));
                pump(&mut r, &mut batch);
                parked.wait();
                swept.wait();

                // The push comes in on a second connection, because the first
                // one is not reading anything while it waits.
                let pusher = r.engine_mut().accept();
                r.engine_mut().feed(pusher, &wire(&[b"RPUSH", b"b", b"v"]));
                pump(&mut r, &mut batch);
                assert_eq!(
                    r.engine().sink().sent(conn),
                    b"*2\r\n$1\r\nb\r\n$1\r\nv\r\n",
                    "served by the thread that parked it"
                );
            });
        });

        assert_eq!(server.parked(), 1, "and the other one is still waiting");
    }

    /// The count a thread branches on before it reaches for the shared list is
    /// its own, because the list is one lock and a thread can only answer what
    /// it parked itself. Branching on the server wide count instead would put
    /// every thread through that lock after every command as soon as one client
    /// blocked anywhere.
    #[test]
    fn a_thread_counts_the_clients_it_blocked_and_nobody_else_s() {
        let mut server = Server::new();
        server.set_threads(2);
        let first = Wire::with_server(server, Recorder::new());
        let second = Wire::over(first.shared(), Recorder::new());
        let server = first.shared();

        let parked = std::sync::Barrier::new(2);
        let looked = std::sync::Barrier::new(2);

        std::thread::scope(|s| {
            let (parked, looked) = (&parked, &looked);
            s.spawn(move || {
                let mut r = Reactor::inline(first);
                let mut batch = Vec::new();
                let conn = r.engine_mut().accept();
                r.engine_mut().feed(conn, &wire(&[b"BLPOP", b"a", b"0"]));
                pump(&mut r, &mut batch);
                assert_eq!(r.engine().waiting(), 1, "the one this thread blocked");
                parked.wait();
                looked.wait();

                // A second client of this thread's that never blocked, opened
                // and closed. It is not on the list, so the count stays where
                // it was rather than following the disconnect down.
                let other = r.engine_mut().accept();
                r.engine_mut().feed(other, &wire(&[b"PING"]));
                pump(&mut r, &mut batch);
                r.engine_mut().hangup(other);
                pump(&mut r, &mut batch);
                assert_eq!(r.engine().waiting(), 1, "still just the blocked one");
            });
            s.spawn(move || {
                let mut r = Reactor::inline(second);
                let mut batch = Vec::new();
                parked.wait();

                // A thread with nothing of its own blocked, on a server that
                // has one client blocked on it.
                pump(&mut r, &mut batch);
                assert_eq!(r.engine().waiting(), 0, "none of them are this one's");
                assert_eq!(r.engine().server().parked(), 1, "one on the server");
                looked.wait();
            });
        });

        assert_eq!(server.parked(), 1);
    }

    /// Two fronts hand out connection slots from zero, so the number that tells
    /// two clients apart cannot come from a front.
    #[test]
    fn client_ids_are_the_server_s_to_hand_out() {
        let first = Wire::new(Recorder::new());
        let second = Wire::over(first.shared(), Recorder::new());
        let mut a = Reactor::inline(first);
        let mut b = Reactor::inline(second);

        let (one, two) = (a.engine_mut().accept(), b.engine_mut().accept());
        assert_eq!(one, two, "the same slot on each front");

        // HELLO answers with the connection id, which is the number CLIENT
        // KILL and CLIENT UNPAUSE take, so two fronts agreeing on it is two
        // clients that cannot be told apart. Protocol three so that the proto
        // field in the same reply is not one of the ids being looked for.
        let mut batch = Vec::new();
        a.engine_mut().feed(one, &wire(&[b"HELLO", b"3"]));
        b.engine_mut().feed(two, &wire(&[b"HELLO", b"3"]));
        pump(&mut a, &mut batch);
        pump(&mut b, &mut batch);

        let first = String::from_utf8_lossy(a.engine().sink().sent(one)).into_owned();
        let second = String::from_utf8_lossy(b.engine().sink().sent(two)).into_owned();
        assert!(first.contains(":1\r\n"), "{first}");
        assert!(second.contains(":2\r\n"), "{second}");
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
        // Fifty is a twentieth of that and it is what runs under Miri, where
        // sixteen thousand commands through the whole engine was a quarter of
        // an hour. The check below is that the size is the one it was after the
        // first round, exactly, so a buffer that keeps anything at all is
        // caught on the second round and every one after it, whichever count
        // this is.
        let rounds = if cfg!(miri) { 50 } else { 1000 };
        for _ in 0..rounds {
            r.engine_mut().feed(conn, &round);
            pump(&mut r, &mut batch);
            r.engine_mut().sink_mut().clear();
        }

        assert_eq!(
            r.engine().buffer_bytes(),
            after_one,
            "the buffers grew over {rounds} rounds of the same sixteen commands"
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

    /// The whole point of a mailbox: a publish on one connection turns into
    /// bytes on another, in the same flush.
    #[test]
    fn a_published_message_lands_on_the_subscriber() {
        let (mut r, sub, mut batch) = engine();
        let pubr = r.engine_mut().accept();

        r.engine_mut().feed(sub, &wire(&[b"SUBSCRIBE", b"news"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*3\r\n$9\r\nsubscribe\r\n$4\r\nnews\r\n:1\r\n"
        );
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(pubr, &wire(&[b"PUBLISH", b"news", b"hi"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(pubr), b":1\r\n");
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*3\r\n$7\r\nmessage\r\n$4\r\nnews\r\n$2\r\nhi\r\n"
        );
    }

    /// A pattern subscriber is told which of its patterns matched as well as
    /// which channel the message went to, so the reply is one field longer.
    #[test]
    fn a_pattern_subscriber_is_told_the_pattern_and_the_channel() {
        let (mut r, sub, mut batch) = engine();
        let pubr = r.engine_mut().accept();

        r.engine_mut().feed(sub, &wire(&[b"PSUBSCRIBE", b"ne*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(pubr, &wire(&[b"PUBLISH", b"news", b"hi"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(pubr), b":1\r\n");
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*4\r\n$8\r\npmessage\r\n$3\r\nne*\r\n$4\r\nnews\r\n$2\r\nhi\r\n"
        );
    }

    /// A RESP2 client that has subscribed to anything can only leave, ping or
    /// subscribe to something else until it unsubscribes, because on RESP2 a
    /// message and a reply are the same shape and a client reading one cannot
    /// tell them apart.
    #[test]
    fn resp2_takes_almost_nothing_from_a_subscriber() {
        let (mut r, conn, mut batch) = engine();

        r.engine_mut().feed(conn, &wire(&[b"SUBSCRIBE", b"a"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(conn, &wire(&[b"GET", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(conn),
            b"-ERR Can't execute 'get': only (P|S)SUBSCRIBE / (P|S)UNSUBSCRIBE / PING / QUIT / RESET are allowed in this context\r\n"
        );
        r.engine_mut().sink_mut().clear();

        // Ping is allowed, and answers in the shape the mode uses.
        r.engine_mut().feed(conn, &wire(&[b"PING"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(conn),
            b"*2\r\n$4\r\npong\r\n$0\r\n\r\n"
        );
        r.engine_mut().sink_mut().clear();

        // And unsubscribing puts the connection back to ordinary work.
        r.engine_mut().feed(conn, &wire(&[b"UNSUBSCRIBE", b"a"]));
        r.engine_mut().feed(conn, &wire(&[b"GET", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(conn),
            b"*3\r\n$11\r\nunsubscribe\r\n$1\r\na\r\n:0\r\n$-1\r\n"
        );
    }

    /// Shard channels are their own namespace. A name subscribed as a shard
    /// channel does not hear a plain publish to the same name, and a pattern
    /// never matches a shard publish.
    #[test]
    fn a_shard_channel_and_a_pattern_do_not_hear_each_other() {
        let (mut r, sub, mut batch) = engine();
        let pubr = r.engine_mut().accept();

        r.engine_mut().feed(sub, &wire(&[b"SSUBSCRIBE", b"sx"]));
        r.engine_mut().feed(sub, &wire(&[b"PSUBSCRIBE", b"s*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(pubr, &wire(&[b"SPUBLISH", b"sx", b"one"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(pubr), b":1\r\n");
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*3\r\n$8\r\nsmessage\r\n$2\r\nsx\r\n$3\r\none\r\n"
        );
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(pubr, &wire(&[b"PUBLISH", b"sx", b"two"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(pubr), b":1\r\n");
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*4\r\n$8\r\npmessage\r\n$2\r\ns*\r\n$2\r\nsx\r\n$3\r\ntwo\r\n"
        );
    }

    /// A subscriber that hangs up stops being one, which matters because the
    /// registry holds a connection id and that id gets handed to the next
    /// client through the door.
    #[test]
    fn a_subscriber_that_goes_away_leaves_the_registry() {
        let (mut r, sub, mut batch) = engine();
        let pubr = r.engine_mut().accept();

        r.engine_mut().feed(sub, &wire(&[b"SUBSCRIBE", b"news"]));
        pump(&mut r, &mut batch);
        r.engine_mut().hangup(sub);
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(pubr, &wire(&[b"PUBLISH", b"news", b"hi"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(pubr), b":0\r\n");

        // And the slot is clean for whoever gets it next.
        let next = r.engine_mut().accept();
        assert_eq!(next, sub);
        r.engine_mut()
            .feed(pubr, &wire(&[b"PUBLISH", b"news", b"hi"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(next), b"");
    }

    /// On RESP3 a message is a push, not a reply, so it can be read off a
    /// connection that is doing something else, and that connection is free to
    /// run ordinary commands while it is subscribed.
    ///
    /// It also pins the order a publish to yourself comes out in. Nothing in
    /// the code special cases it: the count is the reply to the command and the
    /// message is delivered on the way out with everybody else's, so the count
    /// is first.
    #[test]
    fn resp3_delivers_a_message_as_a_push() {
        let (mut r, conn, mut batch) = engine();

        r.engine_mut().feed(conn, &wire(&[b"HELLO", b"3"]));
        r.engine_mut().feed(conn, &wire(&[b"SUBSCRIBE", b"a"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(conn, &wire(&[b"GET", b"k"]));
        r.engine_mut().feed(conn, &wire(&[b"PUBLISH", b"a", b"w"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(conn),
            b"_\r\n:1\r\n>3\r\n$7\r\nmessage\r\n$1\r\na\r\n$1\r\nw\r\n"
        );
    }

    /// A write publishes twice, once on the channel named after the key and
    /// once on the channel named after the event, in that order.
    #[test]
    fn a_write_reaches_a_keyspace_subscriber() {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();

        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"KEA"]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__key*@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(writer), b"+OK\r\n");
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*4\r\n$8\r\npmessage\r\n$12\r\n__key*@0__:*\r\n\
              $16\r\n__keyspace@0__:k\r\n$3\r\nset\r\n\
              *4\r\n$8\r\npmessage\r\n$12\r\n__key*@0__:*\r\n\
              $18\r\n__keyevent@0__:set\r\n$1\r\nk\r\n"
        );
    }

    /// The setting is off by default, so a subscriber on the notification
    /// channels of a server nobody has turned them on for hears nothing.
    #[test]
    fn a_write_says_nothing_until_the_setting_turns_it_on() {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();

        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__key*@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        pump(&mut r, &mut batch);
        assert_eq!(r.engine().sink().sent(sub), b"");
    }

    /// `g` without `$` is the generic class and not the string one, so a
    /// delete goes out and the write that made the key does not.
    #[test]
    fn only_the_classes_that_were_asked_for_are_published() {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();

        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"Eg"]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__key*@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"DEL", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*4\r\n$8\r\npmessage\r\n$12\r\n__key*@0__:*\r\n\
              $18\r\n__keyevent@0__:del\r\n$1\r\nk\r\n"
        );
    }

    /// A command that took a deadline with it says two things, and they come
    /// out in the order the server did them rather than all at the end.
    #[test]
    fn a_write_with_a_deadline_on_it_says_two_things() {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();

        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"EA"]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__keyevent@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SETEX", b"k", b"100", b"v"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*4\r\n$8\r\npmessage\r\n$16\r\n__keyevent@0__:*\r\n\
              $18\r\n__keyevent@0__:set\r\n$1\r\nk\r\n\
              *4\r\n$8\r\npmessage\r\n$16\r\n__keyevent@0__:*\r\n\
              $21\r\n__keyevent@0__:expire\r\n$1\r\nk\r\n"
        );
    }

    /// A subscriber on every event, and the writer that will make them.
    ///
    /// The three collection tests below all start the same way and all care
    /// about the order of what came out rather than about the bytes, so the
    /// setup is here once and the checking is done by [`fired`].
    fn watching() -> (Reactor<Wire<Recorder>>, ConnId, ConnId, Vec<Cmd>) {
        watching_flags(b"EA")
    }

    /// The same, for a test that needs a class `A` does not turn on.
    fn watching_flags(flags: &[u8]) -> (Reactor<Wire<Recorder>>, ConnId, ConnId, Vec<Cmd>) {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", flags]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__keyevent@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();
        (r, sub, writer, batch)
    }

    /// The event and key of every notification the subscriber has been sent.
    ///
    /// Written against the wire bytes because that is what the subscriber
    /// actually got, and a command that fires four events in a fixed order
    /// makes for a byte literal nobody can read.
    fn fired(r: &Reactor<Wire<Recorder>>, sub: ConnId) -> Vec<(String, String)> {
        fired_on(r, sub, 0)
    }

    /// The same, for a test watching a database other than the one the writer is
    /// on, which is the two commands that put a key somewhere else.
    fn fired_on(r: &Reactor<Wire<Recorder>>, sub: ConnId, db: usize) -> Vec<(String, String)> {
        let head = format!("__keyevent@{db}__:");
        let sent = String::from_utf8_lossy(r.engine().sink().sent(sub)).into_owned();
        let mut out = Vec::new();
        let mut parts = sent.split("\r\n");
        while let Some(p) = parts.next() {
            let Some(event) = p.strip_prefix(head.as_str()) else {
                continue;
            };
            // The pattern itself comes past on every frame ahead of the channel
            // and is not one of these.
            if event == "*" {
                continue;
            }
            parts.next();
            let key = parts.next().unwrap_or_default();
            out.push((event.to_owned(), key.to_owned()));
        }
        out
    }

    /// A pop that took the last of a list says what it did and then that the
    /// key is gone, because a list with nothing in it is not a key.
    #[test]
    fn taking_the_last_of_a_collection_says_the_key_went_with_it() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"k", b"a"]));
        r.engine_mut().feed(writer, &wire(&[b"LPOP", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("rpush", "k"), ("lpop", "k"), ("del", "k")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A move whose destination already holds the member says only the half
    /// that happened, since there was nothing to add on the far side.
    #[test]
    fn a_move_onto_a_member_already_there_says_only_the_removal() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut().feed(writer, &wire(&[b"SADD", b"a", b"m"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"SADD", b"b", b"m", b"n"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SMOVE", b"a", b"b", b"m"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("srem", "a"), ("del", "a")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// Writing a member the score it is already sitting at is not a write, and
    /// the reply says as much about it as the silence does.
    #[test]
    fn a_score_that_did_not_move_says_nothing() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut()
            .feed(writer, &wire(&[b"ZADD", b"z", b"4", b"m"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"ZADD", b"z", b"4", b"m"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"ZINCRBY", b"z", b"0", b"m"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // And one that does move says so, so the silence above is the score
        // and not the subscriber having gone away.
        r.engine_mut()
            .feed(writer, &wire(&[b"ZINCRBY", b"z", b"1", b"m"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), [("zincr".to_owned(), "z".to_owned())]);
    }

    /// A write that trimmed says two things, and a trim that found nothing over
    /// the threshold says only the one.
    #[test]
    fn a_stream_write_says_what_the_trim_behind_it_took() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut()
            .feed(writer, &wire(&[b"XADD", b"s", b"1-1", b"f", b"v"]));
        r.engine_mut().feed(
            writer,
            &wire(&[b"XADD", b"s", b"MAXLEN", b"9", b"2-1", b"f", b"v"]),
        );
        r.engine_mut().feed(
            writer,
            &wire(&[b"XADD", b"s", b"MAXLEN", b"1", b"3-1", b"f", b"v"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("xadd", "s"), ("xadd", "s"), ("xadd", "s"), ("xtrim", "s")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// Acknowledging an entry that has already been deleted from under the
    /// group takes nothing out of the log, so it says nothing, even though the
    /// reply calls it deleted.
    #[test]
    fn acknowledging_an_entry_that_is_already_gone_says_nothing() {
        let (mut r, sub, writer, mut batch) = watching();

        for cmd in [
            wire(&[b"XADD", b"s", b"1-1", b"f", b"v"]),
            wire(&[b"XGROUP", b"CREATE", b"s", b"g", b"0"]),
            wire(&[b"XREADGROUP", b"GROUP", b"g", b"c", b"STREAMS", b"s", b">"]),
            wire(&[b"XDEL", b"s", b"1-1"]),
        ] {
            r.engine_mut().feed(writer, &cmd);
        }
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(
            writer,
            &wire(&[b"XACKDEL", b"s", b"g", b"IDS", b"1", b"1-1"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);
    }

    /// Taking the last field out of a hash says the key went with it, the same
    /// as taking the last of a list or a set does.
    #[test]
    fn emptying_a_hash_says_the_key_went_with_the_last_field() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut()
            .feed(writer, &wire(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]));
        r.engine_mut().feed(writer, &wire(&[b"HDEL", b"h", b"a"]));
        // The second names a field that has already gone and one that has not,
        // so it still removed something and the hash is empty behind it.
        r.engine_mut()
            .feed(writer, &wire(&[b"HDEL", b"h", b"b", b"a"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("hset", "h"), ("hdel", "h"), ("hdel", "h"), ("del", "h")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A deadline that has already passed takes the field with it, so what
    /// comes out is the removal and not the deadline.
    #[test]
    fn a_field_deadline_already_past_reads_as_a_removal() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut()
            .feed(writer, &wire(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(
            writer,
            &wire(&[b"HEXPIRE", b"h", b"0", b"FIELDS", b"1", b"a"]),
        );
        r.engine_mut().feed(
            writer,
            &wire(&[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"b"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("hdel", "h"), ("hexpire", "h")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// Writing fields under a deadline that has already gone says all three
    /// things in order: the write, the removal it brought on, and the key.
    #[test]
    fn a_write_under_a_deadline_already_gone_says_the_write_first() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut().feed(
            writer,
            &wire(&[b"HSETEX", b"h", b"EXAT", b"1", b"FIELDS", b"1", b"a", b"1"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("hset", "h"), ("hdel", "h"), ("del", "h")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// Clearing a deadline is only news for a field that had one to clear, and
    /// the reply cannot be read for that: it is the value either way.
    #[test]
    fn clearing_a_deadline_that_was_never_set_says_nothing() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut()
            .feed(writer, &wire(&[b"HSET", b"h", b"a", b"1", b"b", b"2"]));
        r.engine_mut().feed(
            writer,
            &wire(&[b"HEXPIRE", b"h", b"100", b"FIELDS", b"1", b"a"]),
        );
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"HPERSIST", b"h", b"FIELDS", b"1", b"b"]));
        r.engine_mut().feed(
            writer,
            &wire(&[b"HGETEX", b"h", b"PERSIST", b"FIELDS", b"1", b"b"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // And the field that did have one says so, so the silence above is the
        // deadline and not the subscriber having gone away.
        r.engine_mut().feed(
            writer,
            &wire(&[b"HGETEX", b"h", b"PERSIST", b"FIELDS", b"2", b"a", b"b"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), [("hpersist".to_owned(), "h".to_owned())]);
    }

    /// A name that was free is news on its own, and a name that was taken is
    /// not, whatever the write did to what was under it.
    #[test]
    fn a_key_that_was_not_there_before_says_so() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"En");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"w"]));
        r.engine_mut().feed(writer, &wire(&[b"APPEND", b"k", b"x"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"a"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"b"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("new", "k"), ("new", "l")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// And it arrives in front of the write that made it, because at the moment
    /// it is said the write has not finished happening yet.
    #[test]
    fn the_news_of_a_new_key_comes_before_the_write_that_made_it() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"EAn");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"a", b"1"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"b", b"2"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        // A rename is a key arriving under a name that was already taken, and
        // it is still a key arriving: what was there is gone and what is there
        // now was somewhere else a moment ago.
        r.engine_mut().feed(writer, &wire(&[b"RENAME", b"a", b"b"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("new", "b"), ("rename_from", "a"), ("rename_to", "b")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A store form is the other case: the name stays where it stands and only
    /// what is under it changes, so there is no key arriving to say anything
    /// about unless the destination was not there at all.
    #[test]
    fn writing_over_a_destination_is_not_a_key_arriving() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"EAn");

        r.engine_mut()
            .feed(writer, &wire(&[b"RPUSH", b"l", b"c", b"a", b"b"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"d", b"x"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SORT", b"l", b"ALPHA", b"STORE", b"d"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), [("sortstore".to_owned(), "d".to_owned())]);
        r.engine_mut().sink_mut().clear();

        // And the same store onto a name nobody is using says both, which is
        // what makes the silence above the destination and not the flag.
        r.engine_mut().feed(writer, &wire(&[b"DEL", b"d"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"SORT", b"l", b"ALPHA", b"STORE", b"d"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("del", "d"), ("new", "d"), ("sortstore", "d")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A write that throws away the whole of what was under a name says so, and
    /// says the kind changed when it did.
    #[test]
    fn replacing_a_value_says_what_went_and_whether_the_kind_changed() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Eoc");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"w"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"a"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"l", b"v"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("overwritten", "k"),
                ("overwritten", "l"),
                ("type_changed", "l")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// And a write that reaches into the value that is already there says
    /// nothing, however much of it moves.
    #[test]
    fn a_write_that_changes_part_of_a_value_has_not_replaced_it() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Eoc");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"1"]));
        r.engine_mut().feed(writer, &wire(&[b"APPEND", b"k", b"2"]));
        r.engine_mut().feed(writer, &wire(&[b"INCR", b"k"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"SETRANGE", b"k", b"0", b"9"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"a"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"b"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);
    }

    /// The one place the pair comes last rather than first, because there the
    /// destination is a key arriving and not a value changing, so nothing
    /// notices it going until the command says what it did.
    #[test]
    fn a_rename_says_what_it_replaced_after_saying_what_it_did() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"EAnoc");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"a", b"1"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"b", b"x"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(writer, &wire(&[b"RENAME", b"a", b"b"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("new", "b"),
                ("rename_from", "a"),
                ("rename_to", "b"),
                ("overwritten", "b"),
                ("type_changed", "b")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A store form is the other way round, since there the name stays where it
    /// stands and the old value goes before the command has done anything.
    #[test]
    fn a_store_form_says_what_it_replaced_before_saying_what_it_did() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"EAnoc");

        r.engine_mut().feed(writer, &wire(&[b"SADD", b"s", b"m"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"d", b"q"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SINTERSTORE", b"d", b"s"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("overwritten", "d"),
                ("type_changed", "d"),
                ("sinterstore", "d")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A bit write says so when it did something and stays quiet when it did
    /// not, which is not the same as whether it was a write.
    #[test]
    fn a_bit_write_that_left_the_value_alone_says_nothing() {
        let (mut r, sub, writer, mut batch) = watching();

        // `a` is 0x61, so the second bit from the top is already one and the
        // first of these three changes nothing. The second clears it and the
        // third finds it clear.
        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"abc"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SETBIT", b"k", b"1", b"1"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"SETBIT", b"k", b"1", b"0"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"SETBIT", b"k", b"1", b"0"]));
        // And one that writes a zero into a value too short to hold it, which
        // changed no bit that was there and still counts, because the bytes it
        // wrote the zero into were not there before.
        r.engine_mut()
            .feed(writer, &wire(&[b"SETBIT", b"k", b"100", b"0"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("setbit", "k"), ("setbit", "k")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// And `BITFIELD` follows the same rule one subcommand at a time, so a call
    /// that wrote every field back the way it found it says nothing.
    #[test]
    fn a_bitfield_that_wrote_the_same_values_back_says_nothing() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"abc"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        // `a` again, written back over itself.
        r.engine_mut().feed(
            writer,
            &wire(&[b"BITFIELD", b"k", b"SET", b"u8", b"0", b"97"]),
        );
        r.engine_mut()
            .feed(writer, &wire(&[b"BITFIELD", b"k", b"GET", b"u8", b"0"]));
        r.engine_mut().feed(
            writer,
            &wire(&[b"BITFIELD", b"k", b"INCRBY", b"u8", b"0", b"0"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // One that does change a field, and one that only makes the value
        // longer without changing a bit that was in it.
        r.engine_mut().feed(
            writer,
            &wire(&[b"BITFIELD", b"k", b"SET", b"u8", b"0", b"98"]),
        );
        r.engine_mut().feed(
            writer,
            &wire(&[b"BITFIELD", b"k", b"SET", b"u8", b"800", b"0"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("setbit", "k"), ("setbit", "k")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A sketch says `pfadd` when a register moved, and a merge says it under
    /// the same name whatever it merged.
    #[test]
    fn the_sketch_commands_say_what_a_mass_add_says() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut().feed(writer, &wire(&[b"PFADD", b"h", b"a"]));
        // The same element again, which moves nothing.
        r.engine_mut().feed(writer, &wire(&[b"PFADD", b"h", b"a"]));
        // And no elements at all on a sketch that is already there.
        r.engine_mut().feed(writer, &wire(&[b"PFADD", b"h"]));
        // A merge with no sources, which touches nothing and says it anyway.
        r.engine_mut().feed(writer, &wire(&[b"PFMERGE", b"d"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"PFMERGE", b"d", b"h"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("pfadd", "h"), ("pfadd", "d"), ("pfadd", "d")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A geo key is a sorted set, so writing to one says what the `ZADD`
    /// underneath says, and storing a search says a name of its own.
    #[test]
    fn the_geo_commands_say_what_the_sorted_set_under_them_did() {
        let (mut r, sub, writer, mut batch) = watching();

        let point: &[&[u8]] = &[b"GEOADD", b"g", b"13.361389", b"38.115556", b"P"];
        r.engine_mut().feed(writer, &wire(point));
        // The same member at the same place, which is neither an add nor a move.
        r.engine_mut().feed(writer, &wire(point));
        // The same member somewhere else, which is a move and is a write.
        r.engine_mut().feed(
            writer,
            &wire(&[b"GEOADD", b"g", b"14.0", b"38.115556", b"P"]),
        );
        r.engine_mut().feed(
            writer,
            &wire(&[
                b"GEORADIUS",
                b"g",
                b"14.0",
                b"38.0",
                b"200",
                b"km",
                b"STORE",
                b"d",
            ]),
        );
        r.engine_mut().feed(
            writer,
            &wire(&[
                b"GEOSEARCHSTORE",
                b"e",
                b"g",
                b"FROMLONLAT",
                b"14.0",
                b"38.0",
                b"BYRADIUS",
                b"200",
                b"km",
            ]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("zadd", "g"),
                ("zadd", "g"),
                ("georadiusstore", "d"),
                ("geosearchstore", "e")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// And a store whose search found nothing deletes the destination and says
    /// so, which is the rule every store form follows.
    #[test]
    fn a_geo_store_that_found_nothing_takes_the_destination_with_it() {
        let (mut r, sub, writer, mut batch) = watching();

        r.engine_mut().feed(
            writer,
            &wire(&[b"GEOADD", b"g", b"13.361389", b"38.115556", b"P"]),
        );
        r.engine_mut().feed(writer, &wire(&[b"SET", b"d", b"x"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(
            writer,
            &wire(&[
                b"GEORADIUS",
                b"g",
                b"1.0",
                b"1.0",
                b"1",
                b"km",
                b"STORE",
                b"d",
            ]),
        );
        // And again, now that the destination is not there, which deletes
        // nothing and says nothing.
        r.engine_mut().feed(
            writer,
            &wire(&[
                b"GEORADIUS",
                b"g",
                b"1.0",
                b"1.0",
                b"1",
                b"km",
                b"STORE",
                b"d",
            ]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("del", "d")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A key that arrives on a database other than the one that asked for it
    /// says it is new there, and says nothing on the database the command ran
    /// on.
    #[test]
    fn a_key_that_lands_on_another_database_is_new_over_there() {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"EAnoc"]),
        );
        r.engine_mut().feed(
            sub,
            &wire(&[b"PSUBSCRIBE", b"__keyevent@0__:*", b"__keyevent@1__:*"]),
        );
        r.engine_mut().feed(writer, &wire(&[b"SET", b"a", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"COPY", b"a", b"b", b"DB", b"1"]));
        r.engine_mut().feed(writer, &wire(&[b"MOVE", b"k", b"1"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired_on(&r, sub, 1),
            [
                ("new", "b"),
                ("copy_to", "b"),
                ("new", "k"),
                ("move_to", "k")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
        // And nothing of the sort on the database the two commands ran on,
        // which hears only the half of the move that happened there.
        assert_eq!(
            fired_on(&r, sub, 0),
            [("move_from", "k")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A read that found nothing says so, once for each key it went looking
    /// for and in the order it was given them.
    ///
    /// The class is not in `A`, the same way it is not in Redis's, so these ask
    /// for it by letter.
    #[test]
    fn a_read_that_found_nothing_says_which_key_it_was() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"a", b"v"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"MGET", b"a", b"nk", b"nk"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"EXISTS", b"nj", b"a"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "nk"), ("keymiss", "nk"), ("keymiss", "nj")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A write says nothing about a name that was free, and the shapes of the
    /// same command that read say it.
    #[test]
    fn only_the_shape_of_a_write_that_reads_says_it() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        // A plain `SET` never looks, and neither does a `BITFIELD` with a
        // write anywhere in the line.
        r.engine_mut().feed(writer, &wire(&[b"SET", b"a", b"v"]));
        r.engine_mut().feed(
            writer,
            &wire(&[b"BITFIELD", b"nk", b"SET", b"u8", b"0", b"1"]),
        );
        r.engine_mut().feed(writer, &wire(&[b"LPOP", b"nk"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // The two that do, which are the same two commands.
        r.engine_mut()
            .feed(writer, &wire(&[b"SET", b"nj", b"v", b"GET"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"BITFIELD", b"nl", b"GET", b"u8", b"0"]));
        r.engine_mut().feed(writer, &wire(&[b"GETDEL", b"nm"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "nj"), ("keymiss", "nl"), ("keymiss", "nm")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A store form says it for the keys it read and not for the one it is
    /// about to write, however empty that name is.
    #[test]
    fn a_store_form_says_it_only_for_its_sources() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut()
            .feed(writer, &wire(&[b"SINTERSTORE", b"dst", b"nk", b"nj"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"ZUNIONSTORE", b"dst", b"1", b"nz"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"ZRANGESTORE", b"dst", b"nz", b"0", b"-1"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("keymiss", "nk"),
                ("keymiss", "nj"),
                ("keymiss", "nz"),
                ("keymiss", "nz")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// Both stream reads find their keys behind `STREAMS`, and `XREAD` looks
    /// each of them up twice, once to resolve the identifier it was handed and
    /// once to serve from it.
    #[test]
    fn the_stream_reads_say_it_for_the_keys_behind_the_keyword() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut().feed(
            writer,
            &wire(&[b"XREAD", b"STREAMS", b"nk", b"nj", b"0", b"0"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("keymiss", "nk"),
                ("keymiss", "nj"),
                ("keymiss", "nk"),
                ("keymiss", "nj")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A command that failed while it was still reading its own arguments never
    /// looked a key up, so it says nothing, and one that failed on what it
    /// found keeps what it had already said.
    #[test]
    fn an_argument_that_did_not_parse_takes_the_miss_back() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut()
            .feed(writer, &wire(&[b"GETRANGE", b"nk", b"x", b"-1"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"LPOS", b"nk", b"a", b"RANK", b"0"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // A `WRONGTYPE` is an answer about what was under a key, which means
        // the lookups happened and the misses in front of the one that failed
        // stand.
        r.engine_mut().feed(writer, &wire(&[b"SET", b"s", b"v"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SINTER", b"nk", b"s"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), [("keymiss".to_owned(), "nk".to_owned())]);
    }

    /// A read over several keys says nothing about the ones behind a key that
    /// holds the wrong thing, because the command stops there and never looks
    /// at them.
    #[test]
    fn a_read_stops_missing_where_it_stops_looking() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"s", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"x"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"SINTER", b"s", b"nk"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // And a read that does not stop keeps going. `MGET` answers a nil for
        // the list and goes on to look at the key behind it.
        r.engine_mut().feed(writer, &wire(&[b"MGET", b"l", b"nk"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), [("keymiss".to_owned(), "nk".to_owned())]);
    }

    /// The keys a `SORT` builds out of its elements say it too, and they are
    /// the one set of keys nothing could have asked about in front of the
    /// command, since they do not exist until it is running.
    #[test]
    fn the_keys_a_sort_pattern_names_say_it_as_they_are_read() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut()
            .feed(writer, &wire(&[b"RPUSH", b"l", b"1", b"2"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"w_1", b"5"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(
            writer,
            &wire(&[b"SORT", b"l", b"BY", b"w_*", b"GET", b"p_*"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "w_2"), ("keymiss", "p_2"), ("keymiss", "p_1")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A key that was there and had run out says both things in the order they
    /// happened: the deadline first, because the probe that noticed the key was
    /// gone is what reaped it.
    #[test]
    fn a_deadline_that_passed_is_news_before_the_miss_it_causes() {
        let (mut r, sub, mut batch) = timed();
        let writer = r.engine_mut().accept();
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"EgAm"]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__keyevent@0__:*"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"SET", b"k", b"v", b"PX", b"10"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine().server().advance_clock_ms(50);
        r.engine_mut().feed(writer, &wire(&[b"GET", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("expired", "k"), ("keymiss", "k")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A module read says it the same way a core read does, because the module
    /// API opens its key through the same lookup.
    #[test]
    fn a_module_read_says_it_the_way_a_core_read_does() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut().feed(writer, &wire(&[b"JSON.GET", b"nk"]));
        r.engine_mut().feed(writer, &wire(&[b"TS.GET", b"nj"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"BF.EXISTS", b"nl", b"x"]));
        r.engine_mut().feed(writer, &wire(&[b"TDIGEST.MIN", b"nm"]));
        r.engine_mut().feed(writer, &wire(&[b"TOPK.LIST", b"nn"]));
        r.engine_mut().feed(writer, &wire(&[b"CMS.INFO", b"no"]));
        r.engine_mut().feed(writer, &wire(&[b"VCARD", b"np"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [
                ("keymiss", "nk"),
                ("keymiss", "nj"),
                ("keymiss", "nl"),
                ("keymiss", "nm"),
                ("keymiss", "nn"),
                ("keymiss", "no"),
                ("keymiss", "np")
            ]
            .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );

        // And the one of them that reads a list of keys says it for each,
        // without stopping at the first empty name.
        r.engine_mut().sink_mut().clear();
        r.engine_mut()
            .feed(writer, &wire(&[b"JSON.MGET", b"nk", b"nj", b"$"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "nk"), ("keymiss", "nj")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// The module commands that say nothing, which are the writes, the two
    /// reads that were measured quiet, and the whole of the search group bar
    /// the pair that reads a key rather than an index.
    #[test]
    fn the_quiet_module_commands_stay_quiet() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut()
            .feed(writer, &wire(&[b"JSON.SET", b"nk", b"$", b"1"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"BF.ADD", b"nj", b"x"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"TS.ADD", b"nl", b"1000", b"1"]));
        r.engine_mut().feed(writer, &wire(&[b"TS.INFO", b"nm"]));
        r.engine_mut().feed(writer, &wire(&[b"CF.COMPACT", b"nn"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"JSON.DEBUG", b"HELP"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"FT.GET", b"ni", b"no"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"FT.SEARCH", b"ni", b"*"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // The three that do have a key of their own to be missing.
        r.engine_mut()
            .feed(writer, &wire(&[b"JSON.DEBUG", b"MEMORY", b"np"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"FT.SUGGET", b"nq", b"x"]));
        r.engine_mut().feed(writer, &wire(&[b"FT.SUGLEN", b"nr"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "np"), ("keymiss", "nq"), ("keymiss", "nr")]
                .map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// A module read keeps a miss that its own arguments went on to spoil,
    /// where a core read in the same shape takes it back.
    ///
    /// The two are the same question asked in a different order. A core command
    /// reads everything it was sent and then looks, a module command opens its
    /// key and then reads the rest.
    #[test]
    fn a_module_read_keeps_the_miss_a_later_argument_spoiled() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut()
            .feed(writer, &wire(&[b"JSON.GET", b"nk", b"$..["]));
        r.engine_mut().feed(
            writer,
            &wire(&[b"VSIM", b"nj", b"ELE", b"e", b"COUNT", b"x"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "nk"), ("keymiss", "nj")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );
    }

    /// The two merges read a destination and then their sources, and stop at
    /// the first source that is not there because that is where they fail.
    #[test]
    fn the_module_merges_stop_at_the_first_empty_source() {
        let (mut r, sub, writer, mut batch) = watching_flags(b"Em");

        r.engine_mut().feed(
            writer,
            &wire(&[b"TDIGEST.MERGE", b"nk", b"2", b"nj", b"nl"]),
        );
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("keymiss", "nk"), ("keymiss", "nj")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );

        // The sketch writes its destination rather than reading it, so an empty
        // name there is not a miss, and it is the end of the command, so the
        // sources behind it are never opened.
        r.engine_mut().sink_mut().clear();
        r.engine_mut()
            .feed(writer, &wire(&[b"CMS.MERGE", b"nk", b"1", b"nj"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), []);

        // With a destination that is there, the sources are read and the first
        // empty one is the last thing looked at.
        r.engine_mut()
            .feed(writer, &wire(&[b"CMS.INITBYDIM", b"cm", b"100", b"5"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut()
            .feed(writer, &wire(&[b"CMS.MERGE", b"cm", b"2", b"nj", b"nl"]));
        pump(&mut r, &mut batch);
        assert_eq!(fired(&r, sub), [("keymiss".to_owned(), "nj".to_owned())]);
    }

    /// A subscriber on the four subkey channels and the writer that will feed
    /// them.
    ///
    /// The flags name no class channel, so what the subscriber gets is only
    /// what those four published and nothing is in the answer twice.
    fn watching_fields(flags: &[u8]) -> (Reactor<Wire<Recorder>>, ConnId, ConnId, Vec<Cmd>) {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", flags]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__subkey*@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();
        (r, sub, writer, batch)
    }

    /// The channel and payload of every subkey notification the subscriber got.
    ///
    /// Read off the wire rather than checked as one byte literal, because a
    /// command that publishes on all four channels at once makes for a literal
    /// nobody can hold in their head.
    fn carried(r: &Reactor<Wire<Recorder>>, sub: ConnId) -> Vec<(String, String)> {
        let sent = String::from_utf8_lossy(r.engine().sink().sent(sub)).into_owned();
        let mut out = Vec::new();
        let mut parts = sent.split("\r\n");
        while let Some(p) = parts.next() {
            if p != "pmessage" {
                continue;
            }
            // Each of the three that follow is a length and then the bytes, and
            // the first of them is the pattern, which is the same every time.
            let mut next = || {
                parts.next();
                parts.next().unwrap_or_default().to_owned()
            };
            next();
            let channel = next();
            out.push((channel, next()));
        }
        out
    }

    /// The four channels each spell the same event a different way, and the
    /// field list they carry is length prefixed so that a field holding a comma
    /// reads back as one field and not two.
    #[test]
    fn the_subkey_channels_carry_the_fields_an_event_touched() {
        let (mut r, sub, writer, mut batch) = watching_fields(b"ASTIV");

        r.engine_mut()
            .feed(writer, &wire(&[b"HSET", b"h", b"a,b", b"1", b"c", b"2"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            carried(&r, sub),
            [
                ("__subkeyspace@0__:h", "hset|3:a,b,1:c"),
                ("__subkeyevent@0__:hset", "1:h|3:a,b,1:c"),
                ("__subkeyspaceitem@0__:h\na,b", "hset"),
                ("__subkeyspaceitem@0__:h\nc", "hset"),
                ("__subkeyspaceevent@0__:hset|h", "3:a,b,1:c"),
            ]
            .map(|(c, p)| (c.to_owned(), p.to_owned()))
        );
    }

    /// A key holding a newline cannot be told apart from the field spelled
    /// after it on the per field channel, so that one channel is left out for
    /// it rather than sent something nobody can read back.
    #[test]
    fn a_key_holding_a_newline_skips_the_per_field_channel() {
        let (mut r, sub, writer, mut batch) = watching_fields(b"ASTIV");

        r.engine_mut()
            .feed(writer, &wire(&[b"HSET", b"h\nx", b"f", b"1"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            carried(&r, sub),
            [
                ("__subkeyspace@0__:h\nx", "hset|1:f"),
                ("__subkeyevent@0__:hset", "3:h\nx|1:f"),
                ("__subkeyspaceevent@0__:hset|h\nx", "1:f"),
            ]
            .map(|(c, p)| (c.to_owned(), p.to_owned()))
        );
    }

    /// An event with no fields behind it goes out on the two ordinary channels
    /// and on none of these four, however they are set, which is every event
    /// outside the hash class and the `del` behind an emptied hash with it.
    #[test]
    fn an_event_with_no_fields_stays_off_the_subkey_channels() {
        let (mut r, sub, writer, mut batch) = watching_fields(b"AS");

        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"RPUSH", b"l", b"a"]));
        r.engine_mut()
            .feed(writer, &wire(&[b"HSET", b"h", b"f", b"1"]));
        r.engine_mut().feed(writer, &wire(&[b"HDEL", b"h", b"f"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            carried(&r, sub),
            [
                ("__subkeyspace@0__:h", "hset|1:f"),
                ("__subkeyspace@0__:h", "hdel|1:f"),
            ]
            .map(|(c, p)| (c.to_owned(), p.to_owned()))
        );
    }

    /// A subscriber, a writer and a clock the test moves by hand.
    ///
    /// The same arrangement [`watching`] sets up, on the fixed clock
    /// [`timed`] builds, because every deadline in a test has to arrive on
    /// request rather than in its own time.
    fn watching_clock() -> (Reactor<Wire<Recorder>>, ConnId, ConnId, Vec<Cmd>) {
        let (mut r, sub, mut batch) = timed();
        let writer = r.engine_mut().accept();
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"EA"]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__keyevent@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();
        (r, sub, writer, batch)
    }

    /// A key that reached its deadline says so when a reader trips over it, and
    /// the reader's own command says nothing, because as far as it is concerned
    /// the key was never there.
    #[test]
    fn a_deadline_that_passed_is_news_when_a_reader_finds_it() {
        let (mut r, sub, writer, mut batch) = watching_clock();

        r.engine_mut()
            .feed(writer, &wire(&[b"SET", b"k", b"v", b"PX", b"10"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine().server().advance_clock_ms(50);
        r.engine_mut().feed(writer, &wire(&[b"GET", b"k"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("expired", "k")].map(|(e, k)| (e.to_owned(), k.to_owned())),
            "and not a del alongside it, which is a different piece of news"
        );
    }

    /// And a key nobody ever reads back says it too, because the housekeeping
    /// the driver runs between batches goes looking for them.
    ///
    /// This is the whole reason a cache that writes under a deadline and never
    /// reads does not grow forever, and it is worth a test of its own: the sweep
    /// lives behind a driver call rather than behind a command, so nothing in
    /// the command tests would notice if it stopped running.
    #[test]
    fn a_deadline_that_passed_is_news_with_nobody_reading() {
        let (mut r, sub, writer, mut batch) = watching_clock();

        r.engine_mut()
            .feed(writer, &wire(&[b"SET", b"k", b"v", b"PX", b"10"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine().server().advance_clock_ms(50);
        // Nothing to run, so this turn is housekeeping and nothing else.
        pump(&mut r, &mut batch);
        assert_eq!(
            fired(&r, sub),
            [("expired", "k")].map(|(e, k)| (e.to_owned(), k.to_owned()))
        );

        r.engine_mut().sink_mut().clear();
        pump(&mut r, &mut batch);
        assert!(fired(&r, sub).is_empty(), "and it only goes once");
    }

    /// A key an eviction took says that instead, since a client that lost a key
    /// to a memory limit and a client whose key ran out of time are owed two
    /// different explanations.
    #[test]
    fn a_key_a_limit_took_says_it_was_evicted() {
        let (mut r, sub, writer, mut batch) = watching();

        let val = vec![b'v'; 256];
        for i in 0..2000u32 {
            let k = format!("key:{i:08}");
            r.engine_mut()
                .feed(writer, &wire(&[b"SET", k.as_bytes(), &val]));
        }
        pump(&mut r, &mut batch);
        r.engine().server().refresh_memory();
        let full = r.engine().server().memory_bytes();
        r.engine_mut().sink_mut().clear();

        // Under what it is already holding, so the next write has to take
        // something out before it can put anything in.
        let limit = (full / 2).to_string();
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"maxmemory-policy", b"allkeys-random"]),
        );
        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"maxmemory", limit.as_bytes()]),
        );
        r.engine_mut()
            .feed(writer, &wire(&[b"SET", b"newcomer", &val]));
        pump(&mut r, &mut batch);

        let events = fired(&r, sub);
        assert!(
            events.iter().any(|(e, _)| e == "evicted"),
            "the write made room and never said so: {events:?}"
        );
        assert!(
            events
                .iter()
                .all(|(e, k)| e != "evicted" || k != "newcomer"),
            "the key the write was for is the one key it cannot have taken"
        );
    }

    /// Inside a transaction each command's notifications go out before the
    /// next command runs, so `EXEC` does not bunch them all up at the end.
    #[test]
    fn a_transaction_publishes_between_its_commands_and_not_after_them() {
        let (mut r, sub, mut batch) = engine();
        let writer = r.engine_mut().accept();

        r.engine_mut().feed(
            writer,
            &wire(&[b"CONFIG", b"SET", b"notify-keyspace-events", b"EA"]),
        );
        r.engine_mut()
            .feed(sub, &wire(&[b"PSUBSCRIBE", b"__keyevent@0__:*"]));
        pump(&mut r, &mut batch);
        r.engine_mut().sink_mut().clear();

        r.engine_mut().feed(writer, &wire(&[b"MULTI"]));
        r.engine_mut().feed(writer, &wire(&[b"SET", b"k", b"v"]));
        r.engine_mut().feed(writer, &wire(&[b"DEL", b"k"]));
        r.engine_mut().feed(writer, &wire(&[b"EXEC"]));
        pump(&mut r, &mut batch);
        assert_eq!(
            r.engine().sink().sent(sub),
            b"*4\r\n$8\r\npmessage\r\n$16\r\n__keyevent@0__:*\r\n\
              $18\r\n__keyevent@0__:set\r\n$1\r\nk\r\n\
              *4\r\n$8\r\npmessage\r\n$16\r\n__keyevent@0__:*\r\n\
              $18\r\n__keyevent@0__:del\r\n$1\r\nk\r\n"
        );
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
