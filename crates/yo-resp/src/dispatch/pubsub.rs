//! `SUBSCRIBE`, `PUBLISH`, `PUBSUB` and the six other pub/sub commands.
//!
//! Pub/sub is the one part of the command surface that is not a question and an
//! answer. Everything else in this server reads or writes a value and replies to
//! the connection that asked; a publish writes into the reply buffers of
//! connections that are not asking for anything and may not even be on this
//! thread. That is the whole design problem here and the rest of this file is
//! how it is solved.
//!
//! # The three namespaces
//!
//! Channels, patterns and shard channels, and they do not mix. `PUBLISH c`
//! reaches everybody subscribed to the channel `c` and everybody whose pattern
//! matches `c`, and reaches no shard subscriber. `SPUBLISH c` reaches the shard
//! subscribers of `c` and nobody else, not even a pattern that matches it. Both
//! were checked against 8.10.1 rather than read off the documentation, because
//! the shard channels look like ordinary channels with a different command in
//! front of them and they are not.
//!
//! The count in a subscribe or unsubscribe reply follows the same split.
//! `SUBSCRIBE` and `PSUBSCRIBE` report channels plus patterns as one number, and
//! `SSUBSCRIBE` reports shard channels on their own.
//!
//! # Where a subscription lives
//!
//! In two places, which is what makes delivery possible. The connection keeps
//! the names it subscribed to, because that is what an unsubscribe with no
//! arguments has to walk and what a connection going away has to give back. The
//! server keeps a table from name to the list of connections listening on it,
//! because a publish arrives on a connection that knows nothing about any of
//! them.
//!
//! A connection that never subscribes pays one null pointer for the first of
//! those, and a server nobody has subscribed to pays one relaxed load for the
//! second. That is the same shape `WATCH` uses and for the same reason: the
//! common case is a server where this whole file never runs.
//!
//! # Getting the message across a thread
//!
//! The publisher cannot write into the subscriber's reply buffer. The buffers
//! belong to the front, one front per thread, and a thread owns its own: two
//! threads appending to one buffer is a data race and locking every reply buffer
//! so that publishes can reach them would put a lock on the path of every reply
//! the server ever sends, to pay for a feature most servers never use.
//!
//! So a publish does not deliver. It renders nothing, takes no reply buffer and
//! writes an [`Envelope`] into a mailbox belonging to the thread the subscriber
//! is on. That thread drains its own mailbox at the end of its next batch, when
//! it already holds its own connections, and renders the message there. The
//! payload is behind an `Arc`, so one `PUBLISH` to a thousand subscribers copies
//! the bytes once.
//!
//! The mailboxes are their own array and not a field on `Local`. `Local` is a
//! cache line per thread on purpose, so that the counters every command bumps are
//! never a line two threads fight over, and a publisher reaching into thread N's
//! `Local` would undo exactly that. A mailbox is a line a publisher is supposed
//! to touch, so it gets its own.
//!
//! # What a subscriber costs a sleeping thread
//!
//! The poller has no wakeup of its own, so a thread asleep on its socket does
//! not find out that mail arrived. What it does instead is stay on the short
//! wait while it has any subscriber of its own, which each mailbox counts beside
//! its queue. That bounds delivery by the short wait rather than the idle one,
//! and a thread with no subscribers is not affected at all.
//!
//! # Why a client publishing to itself needs no special case
//!
//! Because the order falls out. Redis answers the `PUBLISH` first and pushes the
//! message second, checked byte for byte on the wire against 8.10.1, and a
//! mailbox drained after the batch produces that order without being asked to.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

use super::table::Spec;
use super::{Args, Flow, Server, Session, args};
use crate::reply::Out;
use yo_common::lock::Lock;
use yo_common::{Error, Result, glob};

/// Which of the three namespaces something is in.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// An ordinary channel, reached by `SUBSCRIBE` and `PUBLISH`.
    Channel,
    /// A glob, reached by `PSUBSCRIBE` and matched against a published channel.
    Pattern,
    /// A shard channel, reached by `SSUBSCRIBE` and `SPUBLISH` and by nothing
    /// else.
    Shard,
}

impl Kind {
    /// The first word of a subscribe reply in this namespace.
    const fn joined(self) -> &'static [u8] {
        match self {
            Kind::Channel => b"subscribe",
            Kind::Pattern => b"psubscribe",
            Kind::Shard => b"ssubscribe",
        }
    }

    /// The first word of an unsubscribe reply.
    const fn left(self) -> &'static [u8] {
        match self {
            Kind::Channel => b"unsubscribe",
            Kind::Pattern => b"punsubscribe",
            Kind::Shard => b"sunsubscribe",
        }
    }

    /// The first word of a delivered message.
    const fn word(self) -> &'static [u8] {
        match self {
            Kind::Channel => b"message",
            Kind::Pattern => b"pmessage",
            Kind::Shard => b"smessage",
        }
    }
}

/// Every namespace, for the walks that have to cover all three.
const KINDS: [Kind; 3] = [Kind::Channel, Kind::Pattern, Kind::Shard];

/// What one connection has subscribed to.
///
/// Behind a `Box` on the session, so a connection that never subscribes holds a
/// null pointer and not three empty vectors.
#[derive(Default)]
pub(crate) struct Subs {
    channels: Vec<Vec<u8>>,
    patterns: Vec<Vec<u8>>,
    shard: Vec<Vec<u8>>,
}

impl Subs {
    /// The names in one namespace.
    fn list(&self, kind: Kind) -> &Vec<Vec<u8>> {
        match kind {
            Kind::Channel => &self.channels,
            Kind::Pattern => &self.patterns,
            Kind::Shard => &self.shard,
        }
    }

    /// The same, to be added to.
    fn list_mut(&mut self, kind: Kind) -> &mut Vec<Vec<u8>> {
        match kind {
            Kind::Channel => &mut self.channels,
            Kind::Pattern => &mut self.patterns,
            Kind::Shard => &mut self.shard,
        }
    }

    /// How many subscriptions of every kind this connection holds.
    ///
    /// What decides whether the connection is in subscribe mode, and so what the
    /// RESP2 gate reads. Not the same as the number the replies carry.
    fn total(&self) -> usize {
        self.channels.len() + self.patterns.len() + self.shard.len()
    }

    /// The number a subscribe or unsubscribe reply in this namespace carries.
    ///
    /// Channels and patterns are counted together and shard channels on their
    /// own, which is not a guess: after `SUBSCRIBE a b` and `PSUBSCRIBE p*`,
    /// 8.10.1 answers 3 to another `SUBSCRIBE` and 1 to an `SSUBSCRIBE`.
    fn reported(&self, kind: Kind) -> usize {
        match kind {
            Kind::Shard => self.shard.len(),
            _ => self.channels.len() + self.patterns.len(),
        }
    }
}

impl Session {
    /// Whether this connection is in subscribe mode.
    ///
    /// One null check on a connection that has never subscribed, which is what
    /// the funnel asks before every command on RESP2.
    pub(crate) fn subscribed(&self) -> bool {
        self.subs.as_ref().is_some_and(|s| s.total() != 0)
    }

    /// How many subscriptions this connection holds, of any kind.
    fn sub_total(&self) -> usize {
        self.subs.as_ref().map_or(0, |s| s.total())
    }

    /// The number the next reply in this namespace carries.
    fn sub_count(&self, kind: Kind) -> usize {
        self.subs.as_ref().map_or(0, |s| s.reported(kind))
    }

    /// How many names it is listening to in each namespace, which is what
    /// `CLIENT INFO` reports as `sub`, `psub` and `ssub`.
    ///
    /// Not [`Session::sub_count`], which answers the number a subscribe reply
    /// carries and counts channels and patterns together.
    pub(super) fn sub_counts(&self) -> (usize, usize, usize) {
        self.subs.as_ref().map_or((0, 0, 0), |s| {
            (s.channels.len(), s.patterns.len(), s.shard.len())
        })
    }

    /// This connection's subscriptions, made if it has none yet.
    fn subs_mut(&mut self) -> &mut Subs {
        if self.subs.is_none() {
            self.subs = Some(yo_alloc::allow(Box::<Subs>::default));
        }
        // Just made if it was not there.
        self.subs.as_mut().unwrap()
    }
}

/// One connection listening on one name.
///
/// The client id as well as the slot, because a slot is reused and a client id
/// is not, so a row that outlived its connection is caught at delivery rather
/// than writing into whoever got the slot next.
#[derive(Clone, Copy)]
struct Row {
    client: u64,
    conn: u32,
    /// Which thread owns the connection, and so which mailbox its messages go
    /// in.
    thread: usize,
}

/// Every subscription on the server.
#[derive(Default)]
pub(crate) struct Registry {
    channels: HashMap<Vec<u8>, Vec<Row>>,
    patterns: HashMap<Vec<u8>, Vec<Row>>,
    shard: HashMap<Vec<u8>, Vec<Row>>,
    /// How many rows there are in all three tables.
    ///
    /// Counted as it changes rather than walked, because it is what the server
    /// wide "is anybody subscribed" load reads and that load is in front of
    /// every `PUBLISH`.
    rows: usize,
    /// How many connections hold at least one subscription, which is
    /// `pubsub_clients` in `INFO`.
    clients: usize,
}

impl Registry {
    /// One namespace's table.
    fn table(&self, kind: Kind) -> &HashMap<Vec<u8>, Vec<Row>> {
        match kind {
            Kind::Channel => &self.channels,
            Kind::Pattern => &self.patterns,
            Kind::Shard => &self.shard,
        }
    }

    /// The same, to be changed.
    fn table_mut(&mut self, kind: Kind) -> &mut HashMap<Vec<u8>, Vec<Row>> {
        match kind {
            Kind::Channel => &mut self.channels,
            Kind::Pattern => &mut self.patterns,
            Kind::Shard => &mut self.shard,
        }
    }

    /// Note that a connection is listening on a name.
    ///
    /// The caller has already checked that the connection was not listening on
    /// it, because the connection's own list is the cheaper of the two to ask.
    fn add(&mut self, kind: Kind, name: &[u8], row: Row) {
        yo_alloc::allow(|| {
            self.table_mut(kind)
                .entry(name.to_vec())
                .or_default()
                .push(row)
        });
        self.rows += 1;
    }

    /// Note that it is not, dropping the name when the last one goes.
    ///
    /// A name with no listeners has to go rather than stay empty, because
    /// `PUBSUB CHANNELS` is defined as the names somebody is listening on and it
    /// reads these keys.
    fn remove(&mut self, kind: Kind, name: &[u8], client: u64) {
        let table = self.table_mut(kind);
        let Some(rows) = table.get_mut(name) else {
            return;
        };
        let mut gone = false;
        if let Some(at) = rows.iter().position(|r| r.client == client) {
            rows.swap_remove(at);
            gone = true;
        }
        if rows.is_empty() {
            table.remove(name);
        }
        if gone {
            self.rows -= 1;
        }
    }
}

/// What one message is, once, however many connections it goes to.
struct Body {
    channel: Vec<u8>,
    payload: Vec<u8>,
}

/// What is being carried, which is a message for a subscriber or a line for a
/// monitor.
///
/// Two things go into a mailbox because two things have the same problem: they
/// are written by whichever thread ran the command and they have to land on a
/// connection some other thread owns. Giving `MONITOR` a second mailbox would
/// have meant a second lock, a second length to poll and a second drain in the
/// engine, all to move bytes that arrive by the same route for the same reason.
enum Cargo {
    /// A `PUBLISH`, a `SPUBLISH` or a keyspace notification.
    Message {
        kind: Kind,
        /// The pattern that matched, for a pattern delivery and nothing else.
        ///
        /// Not shared, because each pattern delivery has its own and there is
        /// only ever one connection per pattern per publish.
        pattern: Vec<u8>,
        body: Arc<Body>,
    },
    /// One `MONITOR` line, rendered once and shared by every monitor.
    Line(Arc<Vec<u8>>),
}

/// One message or one line on its way to one connection.
///
/// The body is shared, so a `PUBLISH` to a thousand subscribers copies the
/// channel and the payload once and hands out a thousand refcount bumps.
pub(crate) struct Envelope {
    conn: u32,
    client: u64,
    cargo: Cargo,
}

impl Envelope {
    /// A rendered `MONITOR` line for one watcher.
    pub(crate) fn line(conn: u32, client: u64, line: Arc<Vec<u8>>) -> Envelope {
        Envelope {
            conn,
            client,
            cargo: Cargo::Line(line),
        }
    }

    /// Which connection slot this is for.
    pub(crate) const fn conn(&self) -> u32 {
        self.conn
    }

    /// Which client the slot has to still be, for this to be delivered.
    pub(crate) const fn client(&self) -> u64 {
        self.client
    }

    /// Render it into a connection's reply buffer.
    ///
    /// A push and not an array, which on RESP3 is the `>` type and on RESP2 is
    /// an ordinary array because RESP2 has no way to say this. That is the whole
    /// reason a RESP2 client in subscribe mode is not allowed to send much: its
    /// library has no way to tell a message from the reply to whatever it sent.
    ///
    /// A monitor line is a simple string on both protocols and not a push, which
    /// is Redis's choice and is why `redis-cli monitor` works against a RESP2
    /// server: the connection has stopped being a client, so there is no reply
    /// it could be confused with.
    pub(crate) fn write(&self, out: &mut Out) {
        match &self.cargo {
            Cargo::Line(line) => out.simple(line),
            Cargo::Message {
                kind: Kind::Pattern,
                pattern,
                body,
            } => {
                out.push(4);
                out.bulk(Kind::Pattern.word());
                out.bulk(pattern);
                out.bulk(&body.channel);
                out.bulk(&body.payload);
            }
            Cargo::Message { kind, body, .. } => {
                out.push(3);
                out.bulk(kind.word());
                out.bulk(&body.channel);
                out.bulk(&body.payload);
            }
        }
    }
}

/// One thread's incoming messages.
///
/// A cache line of its own, because this is the one place a thread writes into
/// another thread's memory on purpose and it should not be sharing a line with
/// anything a thread writes to for itself.
#[derive(Default)]
#[repr(align(64))]
pub(crate) struct Mailbox {
    /// What has been posted and not yet delivered.
    queue: Lock<Vec<Envelope>>,
    /// How much is in the queue, so a thread can ask without taking the lock.
    len: AtomicUsize,
    /// How many of this thread's connections are subscribed to anything.
    ///
    /// Read by the driver, which keeps a thread with subscribers on the short
    /// poller wait. Without it a thread asleep on an idle socket would not look
    /// at its mailbox until something else woke it, and the something else may
    /// never come.
    here: AtomicUsize,
}

/// Room for `threads` mailboxes, one each.
pub(crate) fn boxes(threads: usize) -> Box<[Mailbox]> {
    (0..threads.max(1)).map(|_| Mailbox::default()).collect()
}

/// What `INFO` reports about pub/sub.
pub(crate) struct Counts {
    pub(crate) clients: usize,
    pub(crate) channels: usize,
    pub(crate) patterns: usize,
    pub(crate) shard: usize,
}

impl Server {
    /// Whether anybody anywhere is subscribed to anything.
    ///
    /// The one thing a `PUBLISH` on an idle server costs, and it is a relaxed
    /// load of a word that is zero on nearly every server. Relaxed is enough for
    /// the same reason it is enough for watches: a subscribe that has not been
    /// published yet has not been answered either, so no client can be waiting
    /// on a message it was never told it would get.
    pub(crate) fn anyone_subscribed(&self) -> bool {
        self.subs.load(Relaxed) != 0
    }

    /// Note how many subscriptions there are, after the table changed.
    fn note_subs(&self, reg: &Registry) {
        self.subs.store(reg.rows, Relaxed);
    }

    /// Note that this thread has one more, or one fewer, connection that mail
    /// could arrive for, which is a subscriber or a monitor.
    pub(super) fn note_here(&self, thread: usize, by: isize) {
        let Some(mail) = self.mail.get(thread) else {
            return;
        };
        let was = mail.here.load(Relaxed);
        let now = if by < 0 {
            was.saturating_sub(1)
        } else {
            was.saturating_add(1)
        };
        mail.here.store(now, Relaxed);
    }

    /// Leave a message for another thread to deliver.
    pub(super) fn post(&self, thread: usize, env: Envelope) {
        let Some(mail) = self.mail.get(thread) else {
            return;
        };
        let mut queue = mail.queue.lock();
        yo_alloc::allow(|| queue.push(env));
        mail.len.store(queue.len(), Relaxed);
    }

    /// How much mail is waiting for this thread.
    pub(crate) fn mail_here(&self) -> usize {
        self.mail[self.my_slot()].len.load(Relaxed)
    }

    /// Whether this thread has any reason to keep looking at its mailbox.
    ///
    /// Mail waiting, or a subscriber of its own that mail could arrive for. The
    /// driver reads this the way it reads the blocked client count, and for the
    /// same reason: it is work no incoming byte will wake the thread up for.
    pub(crate) fn posted(&self) -> usize {
        let mail = &self.mail[self.my_slot()];
        mail.len.load(Relaxed) + mail.here.load(Relaxed)
    }

    /// Take everything waiting for this thread.
    pub(crate) fn take_mail(&self, into: &mut Vec<Envelope>) {
        let at = self.my_slot();
        let mut queue = self.mail[at].queue.lock();
        yo_alloc::allow(|| into.append(&mut queue));
        self.mail[at].len.store(0, Relaxed);
    }

    /// What `INFO` reports about pub/sub.
    pub(crate) fn pubsub_counts(&self) -> Counts {
        let reg = self.pubsub.lock();
        Counts {
            clients: reg.clients,
            channels: reg.channels.len(),
            patterns: reg.patterns.len(),
            shard: reg.shard.len(),
        }
    }
}

/// Run one of the nine.
pub(crate) fn execute(
    server: &Server,
    session: &mut Session,
    spec: &'static Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<Flow> {
    match spec.name {
        "subscribe" => join(server, session, args, out, Kind::Channel),
        "psubscribe" => join(server, session, args, out, Kind::Pattern),
        "ssubscribe" => join(server, session, args, out, Kind::Shard),
        "unsubscribe" => leave(server, session, args, out, Kind::Channel),
        "punsubscribe" => leave(server, session, args, out, Kind::Pattern),
        "sunsubscribe" => leave(server, session, args, out, Kind::Shard),
        "publish" => publish(server, args, out, Kind::Channel),
        "spublish" => publish(server, args, out, Kind::Shard),
        _ => introspect(server, args, out)?,
    }
    Ok(Flow::Continue)
}

/// `SUBSCRIBE`, `PSUBSCRIBE` and `SSUBSCRIBE`.
///
/// One reply per name, whether or not it was already subscribed to. Subscribing
/// twice is not an error and not a second subscription: it answers again with
/// the count unchanged, which is what 8.10.1 does.
fn join(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out, kind: Kind) {
    let client = session.id();
    let conn = session.conn;
    let thread = server.my_slot();
    let was = session.sub_total();
    let mut reg = server.pubsub.lock();
    for i in 1..args.len() {
        let name = args.get(i);
        let subs = session.subs_mut();
        if !subs.list(kind).iter().any(|held| held == name) {
            yo_alloc::allow(|| subs.list_mut(kind).push(name.to_vec()));
            reg.add(
                kind,
                name,
                Row {
                    client,
                    conn,
                    thread,
                },
            );
        }
        out.push(3);
        out.bulk(kind.joined());
        out.bulk(name);
        out.uint(subs.reported(kind) as u64);
    }
    if was == 0 && session.sub_total() != 0 {
        reg.clients += 1;
        server.note_here(thread, 1);
    }
    server.note_subs(&reg);
}

/// `UNSUBSCRIBE`, `PUNSUBSCRIBE` and `SUNSUBSCRIBE`.
///
/// With names, one reply each, whether or not the connection was subscribed to
/// them. With no names, one reply per name it actually holds, so the counts walk
/// down, and one reply carrying a null name when it holds none at all. All three
/// shapes were checked against 8.10.1, including the last one, which is the only
/// place in the protocol a subscribe reply names nothing.
fn leave(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out, kind: Kind) {
    let client = session.id();
    let thread = server.my_slot();
    let was = session.sub_total();
    let mut reg = server.pubsub.lock();
    if args.len() > 1 {
        for i in 1..args.len() {
            let name = args.get(i);
            drop_one(session, &mut reg, kind, client, name);
            out.push(3);
            out.bulk(kind.left());
            out.bulk(name);
            out.uint(session.sub_count(kind) as u64);
        }
    } else {
        // Copied out because the loop below changes the list it came from, and
        // there are only ever as many of these as this one connection asked
        // for.
        let held = session
            .subs
            .as_ref()
            .map_or_else(Vec::new, |s| yo_alloc::allow(|| s.list(kind).clone()));
        if held.is_empty() {
            out.push(3);
            out.bulk(kind.left());
            out.nil();
            out.uint(session.sub_count(kind) as u64);
        }
        for name in held {
            drop_one(session, &mut reg, kind, client, &name);
            out.push(3);
            out.bulk(kind.left());
            out.bulk(&name);
            out.uint(session.sub_count(kind) as u64);
        }
    }
    if was != 0 && session.sub_total() == 0 {
        reg.clients -= 1;
        server.note_here(thread, -1);
    }
    server.note_subs(&reg);
}

/// Take one name off a connection, on both sides.
fn drop_one(session: &mut Session, reg: &mut Registry, kind: Kind, client: u64, name: &[u8]) {
    let Some(subs) = session.subs.as_mut() else {
        return;
    };
    let Some(at) = subs.list(kind).iter().position(|held| held == name) else {
        return;
    };
    subs.list_mut(kind).swap_remove(at);
    reg.remove(kind, name, client);
}

/// `PUBLISH` and `SPUBLISH`.
///
/// The reply is how many connections it went to, counted here and not after the
/// messages land, which is what Redis reports too: it is the number of
/// subscribers at the moment of the publish and not the number that were still
/// there when the bytes went out.
fn publish(server: &Server, args: Args<'_>, out: &mut Out, kind: Kind) {
    if !server.anyone_subscribed() {
        out.uint(0);
        return;
    }
    out.uint(deliver(server, kind, args.get(1), args.get(2)));
}

/// Put a message in front of everybody listening for it, and say how many that
/// was.
///
/// Split out from [`publish`] because a keyspace notification is a publish that
/// no client asked for and that has no reply to write, so it needs everything
/// this does and none of what is around it. The caller checks
/// [`Server::anyone_subscribed`] first, since that is the check that makes a
/// publish to an empty server free and both callers want it.
pub(crate) fn deliver(server: &Server, kind: Kind, channel: &[u8], payload: &[u8]) -> u64 {
    let reg = server.pubsub.lock();
    let mut body: Option<Arc<Body>> = None;
    let mut sent = 0u64;
    if let Some(rows) = reg.table(kind).get(channel) {
        for row in rows {
            let body = shared(&mut body, channel, payload);
            server.post(
                row.thread,
                Envelope {
                    conn: row.conn,
                    client: row.client,
                    cargo: Cargo::Message {
                        kind,
                        pattern: Vec::new(),
                        body: Arc::clone(body),
                    },
                },
            );
            sent += 1;
        }
    }
    // Patterns are the channel namespace only. `SPUBLISH sx` does not reach a
    // `PSUBSCRIBE s*`, checked against 8.10.1, which is the one thing about
    // shard channels you would get wrong by assuming they are channels.
    if kind == Kind::Channel {
        for (pattern, rows) in &reg.patterns {
            if !glob::matches(pattern, channel) {
                continue;
            }
            for row in rows {
                let body = shared(&mut body, channel, payload);
                let env = yo_alloc::allow(|| Envelope {
                    conn: row.conn,
                    client: row.client,
                    cargo: Cargo::Message {
                        kind: Kind::Pattern,
                        pattern: pattern.clone(),
                        body: Arc::clone(body),
                    },
                });
                server.post(row.thread, env);
                sent += 1;
            }
        }
    }
    sent
}

/// The message body, built the first time it turns out somebody is listening.
///
/// A publish to nobody is the common case on a server that has any subscribers
/// at all, since most channels have none, and it should not copy the payload to
/// find that out.
fn shared<'a>(body: &'a mut Option<Arc<Body>>, channel: &[u8], payload: &[u8]) -> &'a Arc<Body> {
    body.get_or_insert_with(|| {
        yo_alloc::allow(|| {
            Arc::new(Body {
                channel: channel.to_vec(),
                payload: payload.to_vec(),
            })
        })
    })
}

/// `PUBSUB`, which is six subcommands and the only one of the nine that is not
/// about this connection.
fn introspect(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if sub.eq_ignore_ascii_case(b"CHANNELS") {
        names(server, args, out, Kind::Channel)
    } else if sub.eq_ignore_ascii_case(b"SHARDCHANNELS") {
        names(server, args, out, Kind::Shard)
    } else if sub.eq_ignore_ascii_case(b"NUMSUB") {
        counts(server, args, out, Kind::Channel);
        Ok(())
    } else if sub.eq_ignore_ascii_case(b"SHARDNUMSUB") {
        counts(server, args, out, Kind::Shard);
        Ok(())
    } else if sub.eq_ignore_ascii_case(b"NUMPAT") {
        // The arity of a subcommand, which this server has no table for yet, so
        // the two that take no arguments check for themselves. D-114.
        if args.len() != 2 {
            return Err(args::wrong_arity_sub("pubsub", "numpat"));
        }
        out.uint(server.pubsub.lock().patterns.len() as u64);
        Ok(())
    } else if sub.eq_ignore_ascii_case(b"HELP") {
        if args.len() != 2 {
            return Err(args::wrong_arity_sub("pubsub", "help"));
        }
        super::server::help(out, PUBSUB_HELP);
        Ok(())
    } else {
        Err(args::unknown_subcommand(sub, "PUBSUB"))
    }
}

/// `PUBSUB CHANNELS` and `PUBSUB SHARDCHANNELS`.
///
/// The names somebody is listening on, which is why a name with no listeners is
/// dropped from the table rather than left empty.
fn names(server: &Server, args: Args<'_>, out: &mut Out, kind: Kind) -> Result<()> {
    // Redis checks the argument count inside the subcommand here rather than
    // from a table, so a second pattern is the generic syntax error and not an
    // arity error. Checked against 8.10.1, which is the only way anybody would
    // know the two subcommands answer differently shaped errors to the same
    // mistake.
    if args.len() > 3 {
        return Err(unknown_or_arity(args.get(1), "PUBSUB"));
    }
    let pattern = args.opt(2);
    let reg = server.pubsub.lock();
    let table = reg.table(kind);
    let hit = |name: &Vec<u8>| pattern.is_none_or(|p| glob::matches(p, name));
    // Counted and then written, rather than collected, because the length has to
    // go out in front and a list of names nobody asked to keep is an allocation
    // on a command that already holds the registry lock.
    out.array(table.keys().filter(|name| hit(name)).count());
    for name in table.keys().filter(|name| hit(name)) {
        out.bulk(name);
    }
    Ok(())
}

/// `PUBSUB NUMSUB` and `PUBSUB SHARDNUMSUB`.
///
/// A flat array of name and count pairs on both protocols, which is worth saying
/// because it looks like a map and RESP3 has one.
fn counts(server: &Server, args: Args<'_>, out: &mut Out, kind: Kind) {
    let reg = server.pubsub.lock();
    out.array((args.len() - 2) * 2);
    for i in 2..args.len() {
        let name = args.get(i);
        out.bulk(name);
        out.uint(reg.table(kind).get(name).map_or(0, Vec::len) as u64);
    }
}

/// The error a container answers to a subcommand it knows with arguments it does
/// not.
fn unknown_or_arity(sub: &[u8], container: &str) -> Error {
    yo_alloc::allow(|| {
        let sub = String::from_utf8_lossy(sub);
        Error::new(
            yo_common::Code::Invalid,
            format!(
                "unknown subcommand or wrong number of arguments for '{sub}'. Try {container} HELP."
            ),
        )
    })
}

/// Everything a connection ending or resetting has to give back.
///
/// Its subscriptions are rows on the server, so a connection that dropped its
/// own list without saying so would leave the server delivering into a slot
/// somebody else now has, and would keep every `PUBLISH` paying for subscribers
/// that are not there.
pub(crate) fn release(server: &Server, session: &mut Session) {
    let Some(subs) = session.subs.take() else {
        return;
    };
    if subs.total() == 0 {
        return;
    }
    let client = session.id();
    let thread = server.my_slot();
    let mut reg = server.pubsub.lock();
    for kind in KINDS {
        for name in subs.list(kind) {
            reg.remove(kind, name, client);
        }
    }
    reg.clients -= 1;
    server.note_here(thread, -1);
    server.note_subs(&reg);
}

/// The commands a RESP2 connection in subscribe mode may still send.
///
/// Redis's list, from the check in `processCommand`, and it is short for a
/// protocol reason and not a policy one: RESP2 sends a message as an ordinary
/// array, so a client that had a reply outstanding could not tell the two apart.
/// RESP3 has a push type and so has no restriction at all.
fn allowed(name: &str) -> bool {
    matches!(
        name,
        "subscribe"
            | "unsubscribe"
            | "psubscribe"
            | "punsubscribe"
            | "ssubscribe"
            | "sunsubscribe"
            | "ping"
            | "quit"
            | "reset"
    )
}

/// The refusal a RESP2 connection in subscribe mode gets, if this command is one
/// it gets.
///
/// `None` for every command on RESP3, for every connection that has not
/// subscribed to anything, and for anything running inside `EXEC` or inside a
/// script. The last two are not an oversight: a real server makes this check in
/// `processCommand`, and neither a queued command nor a script's call goes
/// through it. `MULTI`, `SUBSCRIBE z`, `GET x`, `EXEC` answers the transaction
/// on 8.10.1 even though the `GET` would have been refused had it been sent on
/// its own.
pub(crate) fn refused(session: &Session, spec: &Spec, out: &Out) -> Option<Error> {
    if out.proto().is_resp3() || session.running() || session.scripted() {
        return None;
    }
    if !session.subscribed() || allowed(spec.name) {
        return None;
    }
    Some(yo_alloc::allow(|| {
        let name = spec.name;
        Error::new(
            yo_common::Code::Invalid,
            format!(
                "Can't execute '{name}': only (P|S)SUBSCRIBE / (P|S)UNSUBSCRIBE / \
                 PING / QUIT / RESET are allowed in this context"
            ),
        )
    }))
}

/// `PING` answers differently to a RESP2 connection in subscribe mode.
///
/// A two element array with `pong` in front rather than a simple string, because
/// on RESP2 everything that arrives while subscribed has to look like a message
/// so that a client library can read them all the same way. RESP3 answers the
/// ordinary `+PONG`.
pub(crate) fn ping(session: &Session, args: Args<'_>, out: &mut Out) -> bool {
    if out.proto().is_resp3() || session.running() || session.scripted() || !session.subscribed() {
        return false;
    }
    out.array(2);
    out.bulk(b"pong");
    out.bulk(args.opt(1).unwrap_or(b""));
    true
}

/// What `PUBSUB HELP` says.
const PUBSUB_HELP: &[&str] = &[
    "PUBSUB <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "CHANNELS [<pattern>]",
    "    Return the currently active channels matching a <pattern> (default: '*').",
    "NUMPAT",
    "    Return number of subscriptions to patterns.",
    "NUMSUB [<channel> ...]",
    "    Return the number of subscribers for the specified channels, excluding",
    "    pattern subscriptions(default: no channels).",
    "SHARDCHANNELS [<pattern>]",
    "    Return the currently active shard level channels matching a <pattern> (default: '*').",
    "SHARDNUMSUB [<shardchannel> ...]",
    "    Return the number of subscribers for the specified shard level channel(s)",
    "HELP",
    "    Print this help.",
];
