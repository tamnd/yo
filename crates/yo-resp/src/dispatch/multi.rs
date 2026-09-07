//! `MULTI`, `EXEC`, `DISCARD`, `WATCH` and `UNWATCH`.
//!
//! A transaction here is what it is in Redis and not what it is in a database
//! with a log: the commands between `MULTI` and `EXEC` are not run when they
//! are sent, they are held on the connection and run in one go when `EXEC`
//! arrives. Nothing is rolled back, because nothing was done yet, and a command
//! that fails inside `EXEC` leaves its error in the middle of the array and the
//! rest of the queue still runs. That is worth saying plainly because the word
//! transaction promises more than this delivers, and a client that expects
//! atomicity and gets it, but expects rollback and does not, will write code
//! that is wrong in a way nothing here can catch.
//!
//! # Where the queue lives
//!
//! On the [`Session`], as the wire bytes each command arrived as. Holding the
//! bytes rather than the decoded arguments is what makes the queue own its
//! contents: the arguments a command was framed with are slices of the
//! connection's read buffer, and that buffer is reused by the next command long
//! before `EXEC` is sent. Re-decoding at `EXEC` costs one pass over bytes that
//! are already in cache, which is nothing next to running the command.
//!
//! It also means the queued command goes back through the same funnel a fresh
//! command does, so it is counted in `INFO commandstats`, it is checked against
//! `maxmemory`, and it touches watched keys, all of which a real server does and
//! none of which we would have got for free from a queue of pre-resolved calls.
//!
//! # Where the watches live
//!
//! Beside the key, on the [`Server`], and not beside the client. This is the
//! half of `WATCH` that a single threaded server lets you get wrong. If a
//! connection remembered the value it read and compared it at `EXEC`, the
//! comparison would be right on one thread and useless on several, because
//! another thread can write and undo a key between the two reads and leave the
//! value looking untouched. So what a watch records is a stamp on a row keyed by
//! database and key, every write bumps the stamp on the rows for the keys it
//! named, and `EXEC` fails when a stamp moved. Writing the same value back still
//! bumps it, which is Redis's rule as well.
//!
//! The row carries whether the key existed at the last look, because Redis has
//! one exception to "a write on a watched key dirties it": deleting a key that
//! was not there does nothing at all, so it does not dirty. The rule that
//! reproduces it is to bump when the key is live now or was live before, which
//! is the same thing as saying a write that could not have changed anything
//! changes nothing.
//!
//! Expiry never reaches this code, and neither does eviction, so a stamp on its
//! own would miss both. `EXEC` also compares whether each watched key is still
//! there, which covers them without either of them having to know that watches
//! exist.
//!
//! # What it costs when nobody is watching
//!
//! One relaxed load. [`Server::watching`] counts the rows, it is zero on every
//! server where no client has ever sent `WATCH`, and the whole of the rest of
//! this file is behind that check.

use std::collections::HashMap;

use super::table::Spec;
use super::{Args, Flow, Server, Session, resolved, write_error};
use crate::proto::Limits;
use crate::reply::Out;
use yo_common::{Code, Error, Result};

/// The six commands `MULTI` does not queue.
///
/// Redis's list, from the check in `processCommand`. Four of them are the
/// transaction commands themselves, and the other two are the ones that end a
/// connection or put it back to the start, which would be unreachable if they
/// went in the queue. Everything else is queued, including the ones that look
/// like they could not be: `SELECT` inside a transaction is queued and applied
/// when `EXEC` runs it.
fn exempt(name: &str) -> bool {
    matches!(
        name,
        "exec" | "discard" | "multi" | "watch" | "unwatch" | "quit" | "reset"
    )
}

/// The commands held since `MULTI`, and whether one of them was refused.
#[derive(Default)]
pub(crate) struct Queue {
    /// Each queued command, as the wire bytes it arrived as.
    cmds: Vec<Vec<u8>>,
    /// Whether the funnel turned one of them away.
    ///
    /// A queued command that is not a command at all, or that has the wrong
    /// number of arguments, is refused when it is sent rather than when the
    /// transaction runs, and the transaction is dead from that moment. It still
    /// answers `QUEUED` to everything after it, because Redis does, and then
    /// `EXEC` refuses to run any of it.
    dirty: bool,
}

impl Queue {
    /// How many commands are waiting.
    fn len(&self) -> usize {
        self.cmds.len()
    }
}

/// One key this connection is watching, and what it looked like at the time.
pub(crate) struct Watched {
    db: usize,
    key: Vec<u8>,
    /// The row's stamp when the watch was taken.
    stamp: u64,
    /// Whether the key was there when the watch was taken.
    live: bool,
}

/// The stamp on one watched key.
struct Row {
    stamp: u64,
    /// How many connections are watching this key, so the row can go away when
    /// the last of them stops.
    watchers: u32,
    /// Whether the key was there at the last look.
    live: bool,
}

/// Every watched key on the server.
///
/// Keyed by a hash of the database and the key rather than by the pair itself,
/// so that a lookup on the write path builds nothing. The bucket holds the
/// database and key it was made from and they are compared, so a hash collision
/// costs a comparison rather than a wrong answer.
#[derive(Default)]
pub(crate) struct Watches {
    rows: HashMap<u64, Vec<(usize, Vec<u8>, Row)>>,
}

impl Watches {
    /// How many keys are watched, near enough for the count on the server.
    ///
    /// Buckets rather than rows, because the only thing anybody asks this is
    /// whether it is zero and the two agree about that exactly.
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// The row for a key, made if it is not there yet.
    fn entry(&mut self, db: usize, key: &[u8], live: bool) -> &mut Row {
        let bucket = self.rows.entry(mix(db, key)).or_default();
        let at = bucket.iter().position(|(d, k, _)| *d == db && k == key);
        let at = match at {
            Some(at) => at,
            None => {
                bucket.push((
                    db,
                    key.to_vec(),
                    Row {
                        stamp: 0,
                        watchers: 0,
                        live,
                    },
                ));
                bucket.len() - 1
            }
        };
        &mut bucket[at].2
    }

    /// The row for a key, if anybody is watching it.
    fn find(&mut self, db: usize, key: &[u8]) -> Option<&mut Row> {
        let bucket = self.rows.get_mut(&mix(db, key))?;
        bucket
            .iter_mut()
            .find(|(d, k, _)| *d == db && k == key)
            .map(|(_, _, row)| row)
    }

    /// Let go of one watcher's claim on a key, dropping the row with the last.
    fn drop_one(&mut self, db: usize, key: &[u8]) {
        let at = mix(db, key);
        let Some(bucket) = self.rows.get_mut(&at) else {
            return;
        };
        let Some(i) = bucket.iter().position(|(d, k, _)| *d == db && k == key) else {
            return;
        };
        bucket[i].2.watchers -= 1;
        if bucket[i].2.watchers == 0 {
            bucket.swap_remove(i);
        }
        if bucket.is_empty() {
            self.rows.remove(&at);
        }
    }
}

/// Where a database and a key hash to together.
fn mix(db: usize, key: &[u8]) -> u64 {
    yo_common::wyhash::hash_key(key) ^ (db as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Run one of the five.
pub(crate) fn execute(
    server: &Server,
    session: &mut Session,
    spec: &'static Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<Flow> {
    match spec.name {
        "multi" => {
            // The one error in this file a real server raises from the command
            // rather than from the funnel, and the difference shows: this does
            // not kill the transaction it was sent inside, so the commands
            // before and after it still run.
            if session.multi.is_some() {
                return Err(Error::new(Code::Invalid, "MULTI calls can not be nested"));
            }
            session.multi = Some(Queue::default());
            out.ok();
            Ok(Flow::Continue)
        }
        "discard" => {
            if session.multi.is_none() {
                return Err(Error::new(Code::Invalid, "DISCARD without MULTI"));
            }
            discard(server, session);
            out.ok();
            Ok(Flow::Continue)
        }
        "watch" => {
            if session.multi.is_some() {
                return Err(Error::new(
                    Code::Invalid,
                    "WATCH inside MULTI is not allowed",
                ));
            }
            watch(server, session, args);
            out.ok();
            Ok(Flow::Continue)
        }
        "unwatch" => {
            unwatch(server, session);
            out.ok();
            Ok(Flow::Continue)
        }
        _ => Ok(exec(server, session, out)),
    }
}

/// `WATCH key [key ...]`.
fn watch(server: &Server, session: &mut Session, args: Args<'_>) {
    let db = session.db();
    let mut watches = server.watches.lock();
    for i in 1..args.len() {
        let key = args.get(i);
        // Watching the same key twice is one watch, which matters because the
        // row counts watchers and a connection that let go twice would take
        // somebody else's row with it.
        if session.watching.iter().any(|w| w.db == db && w.key == key) {
            continue;
        }
        let live = server.dbs[db].hold(key).kind_of(key).is_some();
        let row = watches.entry(db, key, live);
        row.watchers += 1;
        // The row's own idea of whether the key is there is refreshed here, and
        // has to be, because the row may have outlived a stretch where nothing
        // wrote to the key and everything that did happen to it happened
        // through expiry.
        row.live = live;
        let stamp = row.stamp;
        yo_alloc::allow(|| {
            session.watching.push(Watched {
                db,
                key: key.to_vec(),
                stamp,
                live,
            });
        });
    }
    server.recount(&watches);
}

/// `UNWATCH`, and the same thing every other end of a watch does.
fn unwatch(server: &Server, session: &mut Session) {
    if session.watching.is_empty() {
        return;
    }
    let mut watches = server.watches.lock();
    for w in session.watching.drain(..) {
        watches.drop_one(w.db, &w.key);
    }
    server.recount(&watches);
}

/// Whether every watched key is still what it was, letting all of them go.
///
/// The two questions are the stamp and whether the key is there, and the second
/// one is not covered by the first: a key that expired or was evicted went away
/// without any command naming it, so nothing bumped its stamp.
fn checked(server: &Server, session: &mut Session) -> bool {
    if session.watching.is_empty() {
        return true;
    }
    let mut ok = true;
    let mut watches = server.watches.lock();
    for w in session.watching.drain(..) {
        if let Some(row) = watches.find(w.db, &w.key)
            && row.stamp != w.stamp
        {
            ok = false;
        }
        if ok {
            let live = server.dbs[w.db].hold(&w.key).kind_of(&w.key).is_some();
            if live != w.live {
                ok = false;
            }
        }
        watches.drop_one(w.db, &w.key);
    }
    server.recount(&watches);
    ok
}

/// Throw the transaction away, watches and all.
fn discard(server: &Server, session: &mut Session) {
    session.multi = None;
    unwatch(server, session);
}

/// `EXEC`.
fn exec(server: &Server, session: &mut Session, out: &mut Out) -> Flow {
    let Some(queue) = session.multi.take() else {
        // Not an `Err`, because the caller has already taken the transaction
        // apart and there is nothing left to unwind.
        write_error(out, &Error::new(Code::Invalid, "EXEC without MULTI"));
        return Flow::Continue;
    };
    // Both of these have to happen whatever the answer is, so they are done
    // before anything is written: a failed `EXEC` lets go of the watches the
    // same way a successful one does.
    let clean = checked(server, session);
    if queue.dirty {
        // Redis checks this first, so a transaction that is both dirty from a
        // refused command and dirty from a watch says the first of the two.
        out.error_line(
            b"EXECABORT ",
            b"Transaction discarded because of previous errors.",
        );
        return Flow::Continue;
    }
    if !clean {
        out.nil_array();
        return Flow::Continue;
    }
    out.array(queue.len());
    let limits = Limits::default();
    // Taken out of the session rather than made here, so that the room it has
    // for spans outlives the transaction and only the first `EXEC` on a
    // connection pays for it. It is taken rather than borrowed because the
    // arguments are read out of it while `resolved` is holding the session.
    let mut argv = std::mem::take(&mut session.replay);
    let mut flow = Flow::Continue;
    // Nothing else may run against the databases between the first of these and
    // the last, which on a server with one shard thread is true because there is
    // nobody else to run. It is not true with `--threads` above one, and that is
    // D-112.
    // A real server runs a queued command through `call` and not through
    // `processCommand`, so the refusals `processCommand` makes are not made
    // again here. Only one of them can tell the difference today, which is the
    // RESP2 subscribe mode gate, and this is how it finds out.
    let was = session.running;
    session.running = true;
    for wire in &queue.cmds {
        if argv.decode(wire, &limits).is_err() {
            // Unreachable: these bytes were built here from a command that had
            // already been decoded once. An element still has to go in the
            // array, because the length is already written.
            write_error(out, &Error::new(Code::Invalid, "Protocol error"));
            continue;
        }
        let args = Args::new(&argv, wire);
        let spec = super::lookup(args.name());
        if resolved(server, session, spec, args, out) == Flow::Close {
            flow = Flow::Close;
        }
    }
    session.running = was;
    session.replay = argv;
    flow
}

/// Hold a command instead of running it, because a transaction is open.
///
/// The reply is `QUEUED` even when the transaction is already dead, which looks
/// wrong and is what a real server does. The client finds out at `EXEC`, once,
/// rather than at whichever command happened to be the one that broke it.
pub(crate) fn queue(session: &mut Session, args: Args<'_>, out: &mut Out) -> Flow {
    if let Some(q) = session.multi.as_mut()
        && !q.dirty
    {
        yo_alloc::allow(|| {
            let mut wire = format!("*{}\r\n", args.len()).into_bytes();
            for i in 0..args.len() {
                let a = args.get(i);
                wire.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
                wire.extend_from_slice(a);
                wire.extend_from_slice(b"\r\n");
            }
            q.cmds.push(wire);
        });
    }
    out.simple(b"QUEUED");
    Flow::Continue
}

/// Everything a connection ending has to let go of.
pub(crate) fn release(server: &Server, session: &mut Session) {
    session.multi = None;
    unwatch(server, session);
}

impl Session {
    /// Whether a transaction is open on this connection.
    pub(crate) const fn in_multi(&self) -> bool {
        self.multi.is_some()
    }

    /// Whether this command has to be held rather than run.
    pub(crate) fn queues(&self, name: &str) -> bool {
        self.multi.is_some() && !exempt(name)
    }

    /// Mark the open transaction, if there is one, as one that cannot run.
    ///
    /// This is Redis's `flagTransaction`, and what is worth knowing about it is
    /// which errors call it. The ones the funnel raises before a command body
    /// is reached do: an unknown name, the wrong number of arguments, a command
    /// that is not allowed in a transaction, no room under `maxmemory`. The ones
    /// a command body raises do not, which is why `MULTI` inside `MULTI` is an
    /// error that leaves the transaction alive.
    pub(crate) fn dirty_multi(&mut self) {
        if let Some(q) = self.multi.as_mut() {
            q.dirty = true;
        }
    }
}

/// Refuse a command and take the whole transaction down with it.
///
/// `EXEC` is the one command the funnel cannot simply turn away, because a
/// client that sent it is waiting for the transaction to be over one way or
/// another. Redis answers the refusal with the transaction's error rather than
/// the command's, with the reason spliced in, which is what makes `EXEC x` read
/// as an abort and not as a typo.
pub(crate) fn abort(server: &Server, session: &mut Session, why: &str, out: &mut Out) {
    discard(server, session);
    yo_alloc::allow(|| {
        let line = format!("Transaction discarded because of: {why}");
        out.error_line(b"EXECABORT ", line.as_bytes());
    });
}

/// Which keys a command that has just run may have changed.
enum Touch<'a> {
    /// The keys it named, at these positions in its own argument list.
    Named(Args<'a>, super::table::KeySpan),
    /// Every watched key in these databases, for the commands that reach keys
    /// they did not name.
    Sweep(usize, usize),
}

/// Bump the stamp on every watched key a write may have changed.
///
/// Called after the command ran rather than before, so that the key's liveness
/// is read at the point Redis would have signalled the change. Only for the
/// commands carrying Redis's own `write` flag, and only when somebody is
/// watching something, which on nearly every server is never.
pub(crate) fn touched(server: &Server, session: &Session, spec: &Spec, args: Args<'_>) {
    // A script's writes come through here one at a time as the script makes
    // them, so the call that ran the script has nothing left to say. Treating
    // `EVAL` as a write of its own would dirty every watched key in the database
    // on any script, including one that only read.
    if spec.group == "scripting" {
        return;
    }
    let db = session.db();
    let touch = match spec.name {
        // The three that empty or exchange whole databases, and the two that
        // reach a database the connection did not select. None of them can be
        // described by a key span, so all of them ask every watched row in range
        // whether it changed.
        "flushall" | "swapdb" | "copy" | "move" => Touch::Sweep(0, super::DATABASES),
        "flushdb" => Touch::Sweep(db, db + 1),
        // A write whose keys are somewhere the table cannot describe, which is
        // what Redis's `movablekeys` means and is why `SORT ... STORE dst` has
        // it: the destination is behind a keyword and no triple can find it.
        // Asking every watched row in the database is more work than reading the
        // keys would have been and cannot miss one.
        _ if spec.flags.contains(&"movablekeys") => Touch::Sweep(db, db + 1),
        _ => match super::table::key_span(spec, args, 0) {
            Ok(span) => Touch::Named(args, span),
            // A write with a count that did not parse cannot have got far enough
            // to change anything, and a write with no keys at all has nothing to
            // mark.
            Err(_) => return,
        },
    };
    let mut watches = server.watches.lock();
    match touch {
        Touch::Named(args, span) => {
            for i in 0..span.count {
                let key = args.get(span.first + i * span.step);
                let Some(live) = probe(server, &mut watches, db, key) else {
                    continue;
                };
                bump(&mut watches, db, key, live);
            }
        }
        Touch::Sweep(from, to) => {
            // The rows first, because reading them and writing them cannot
            // happen in the same borrow, and there are only ever as many of
            // these as somebody has asked to watch.
            let mut hit: Vec<(usize, Vec<u8>)> = Vec::new();
            yo_alloc::allow(|| {
                for bucket in watches.rows.values() {
                    for (rdb, key, _) in bucket {
                        if *rdb >= from && *rdb < to {
                            hit.push((*rdb, key.clone()));
                        }
                    }
                }
            });
            for (rdb, key) in hit {
                let live = server.dbs[rdb].hold(&key).kind_of(&key).is_some();
                bump(&mut watches, rdb, &key, live);
            }
        }
    }
}

/// Whether a key is there, for a key somebody is watching.
///
/// `None` for a key nobody is watching, which saves the stripe lock. That is
/// the whole reason this is not two lines at the call site: on a server where
/// one client watches one key, every other write goes through here and none of
/// them should touch the keyspace again.
fn probe(server: &Server, watches: &mut Watches, db: usize, key: &[u8]) -> Option<bool> {
    watches.find(db, key)?;
    Some(server.dbs[db].hold(key).kind_of(key).is_some())
}

/// Move a watched key's stamp on, if the write could have changed anything.
///
/// A key that was not there before and is not there now was not changed by
/// anything, which is how `DEL` on a key that never existed leaves a watch
/// alone. Everything else counts, including a `SET` that wrote the value that
/// was already there.
fn bump(watches: &mut Watches, db: usize, key: &[u8], live: bool) {
    let Some(row) = watches.find(db, key) else {
        return;
    };
    if row.live || live {
        row.stamp = row.stamp.wrapping_add(1);
    }
    row.live = live;
}

/// What the funnel does with a command it is about to refuse.
///
/// Split out because the three refusals in `resolved` all do the same two
/// things and `EXEC` is the exception in all three of them.
pub(crate) fn refuse(
    server: &Server,
    session: &mut Session,
    spec: Option<&'static Spec>,
    e: &Error,
    out: &mut Out,
) {
    if spec.is_some_and(|s| s.name == "exec") {
        abort(server, session, e.message(), out);
        return;
    }
    session.dirty_multi();
    write_error(out, e);
}

/// `SHUTDOWN` and nothing else, today: the commands a transaction may not hold.
pub(crate) fn refused_in_multi(spec: &Spec) -> Option<Error> {
    spec.flags
        .contains(&"no_multi")
        .then(|| Error::new(Code::Invalid, "Command not allowed inside a transaction"))
}
