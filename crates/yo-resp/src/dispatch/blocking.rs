//! The six list commands that wait, and the machinery that lets a client wait.
//!
//! `BLPOP` is `LPOP` with one difference: when there is nothing to pop, the
//! client waits instead of being told no. Everything here is about that wait,
//! and nothing here knows anything about lists that [`super::lists`] does not
//! already know.
//!
//! # The command is kept, not the client
//!
//! A parked client is a [`Waiter`]: the keys it named, what it wanted to do with
//! them, and when to give up. It is not a suspended stack and it is not a task.
//! Answering it later is running the same attempt again against a database that
//! has changed since, which is why [`Want::attempt`] is the whole of both paths.
//! The command handler calls it once to see whether the client has to wait at
//! all, and the retry calls it again each time something might have arrived.
//!
//! That is also why the six commands cost nothing when they do not block. A
//! `BLPOP` on a list with something in it runs the same three lines `LPOP` runs
//! and never touches the waiter list.
//!
//! # A slot is reused and a client id is not
//!
//! A waiter remembers both. The slot is where that connection's reply buffer
//! is, and the client id is what says the connection sitting on that slot is
//! still the one that blocked. The engine takes a waiter off the list when its
//! connection closes, so the check should never fail, and it is there because
//! the cost of being wrong about it is a reply written into somebody else's
//! socket.
//!
//! # What wakes a waiter
//!
//! Any command at all, which is more than is needed and is not the cost it
//! sounds like: the engine looks at whether anybody is parked before it looks at
//! anything else, so a server with no blocked clients pays one load and one
//! branch per command and nothing more. Narrowing it to writes would save
//! nothing measurable and would need a rule about which commands can put a list
//! under a key, which `RENAME`, `COPY` and `RESTORE` all make longer than it
//! looks.
//!
//! What is left is that the waiter list is walked rather than indexed by key, so
//! a server with a thousand parked workers walks a thousand entries per command.
//! The fix when that matters is an index from key to waiter, not a different
//! rule about when to look.

use yo_common::{Code, Error, Result, num};
use yo_kv::{Db, End, Entry, Member, Movem, ZEnd};

use super::args::{self, Args, NOT_AN_INT};
use super::lists::{BAD_MPOP_COUNT, BAD_NUMKEYS, end_of, movem_options};
use super::streams;
use super::table::Spec;
use super::zsets;
use super::{Flow, Server, Session};
use crate::reply::Out;

/// What Redis says about a timeout it cannot read as a number.
const NOT_A_FLOAT: &str = "timeout is not a float or out of range";
/// What it says about one it can read and will not take.
const NEGATIVE: &str = "timeout is negative";
/// And about one so far away that milliseconds do not fit in an `i64`.
const OUT_OF_RANGE: &str = "timeout is out of range";
/// `WAIT` and `WAITAOF` take their timeout in whole milliseconds rather than in
/// seconds, so a timeout they cannot read is a different complaint again.
const TIMEOUT_NOT_AN_INT: &str = "timeout is not an integer or out of range";
/// What `WAITAOF` says about a `numlocal` that is neither of the two it takes.
const NOT_ZERO_OR_ONE: &str = "value is out of range, value must between 0 and 1";
/// And about a negative `numreplicas`.
const NOT_POSITIVE: &str = "value is out of range, must be positive";
/// And what it says when asked to wait for a file the server does not keep. The
/// full stop at the end is Redis's and is the one message in the group that has
/// one, which is why it is worth writing down rather than tidying up.
const NO_AOF: &str = "WAITAOF cannot be used when numlocal is set but appendonly is disabled.";

/// Run one blocking command.
///
/// `Flow::Block` means nothing was written and the client is on the waiter
/// list. The engine is what knows which socket that client is on, so it is the
/// engine that finishes the registration and the engine that stops reading
/// commands from a connection that is now waiting for one.
///
/// # Errors
///
/// A timeout that is not a timeout, a direction that is not a direction, and a
/// key holding something that is not a list.
pub(super) fn execute(
    server: &mut Server,
    session: &Session,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<Flow> {
    // The two that wait on replication rather than on a key. They are here
    // because they carry the blocking flag and that flag is what routes a
    // command to this file, and they leave immediately because there is nothing
    // for them to wait for yet. See [`replication`] for what they answer.
    if spec.name == "wait" || spec.name == "waitaof" {
        return replication(spec.name, args, out).map(|()| Flow::Continue);
    }
    let now = server.now_ms();
    // The two stream reads, which are here for the same reason the list six
    // are and leave through a different door. `BLOCK` is optional on both, so
    // where `BLPOP` always has a timeout to read, `XREAD` may have been told to
    // answer now and take nothing for an answer. That is the difference between
    // parking and writing the null, and it cannot be said with a deadline of
    // `None`, which already means wait for as long as it takes.
    if spec.name == "xread" || spec.name == "xreadgroup" {
        let db = session.db();
        let want = streams::parse_read(spec.name, args, server.striped(db), now)?;
        let block = Block::xread(want.keys, want.reads);
        if block.now(server.striped(db), now, out)? {
            return Ok(Flow::Continue);
        }
        let Some(deadline) = want.wait else {
            // No `BLOCK` at all, so nothing arriving is the answer and not a
            // reason to wait for it. A null array on both protocols, which is
            // also what a `BLOCK` that runs out sends.
            out.nil_array();
            return Ok(Flow::Continue);
        };
        server.park(session.id(), db, deadline, block);
        return Ok(Flow::Block);
    }
    let last = args.len() - 1;
    let (deadline, block) = match spec.name {
        // The keys are everything between the name and the timeout, so `BLPOP a
        // b c 0` waits on three keys and answers with whichever one arrives
        // first rather than with the first one named.
        "blpop" | "brpop" => {
            let end = if spec.name == "blpop" {
                End::Left
            } else {
                End::Right
            };
            let deadline = timeout(args.get(last), now)?;
            (deadline, Block::pop((1..last).map(|i| args.get(i)), end))
        }
        // The directions before the timeout, which is the order Redis checks
        // them in, so `BLMOVE a b UP DOWN nonsense` is a syntax error and not a
        // complaint about the timeout.
        "blmove" => {
            let (from, to) = (end_of(args.get(3))?, end_of(args.get(4))?);
            let deadline = timeout(args.get(5), now)?;
            (deadline, Block::moved(args.get(1), args.get(2), from, to))
        }
        // The same order again with one more thing to read: ends, then timeout,
        // then the options behind it. `BLMOVEM s d UP DOWN abc` complains about
        // the directions and `BLMOVEM s d LEFT RIGHT abc COUNT abc BULK` about
        // the timeout, both measured against 8.10.1 rather than assumed, because
        // a line wrong in two places has exactly one right answer.
        "blmovem" => {
            let (from, to) = (end_of(args.get(3))?, end_of(args.get(4))?);
            let deadline = timeout(args.get(5), now)?;
            let mv = movem_options(args, 6, from, to)?;
            (deadline, Block::movem(args.get(1), args.get(2), mv))
        }
        "brpoplpush" => {
            let deadline = timeout(args.get(3), now)?;
            (
                deadline,
                Block::moved(args.get(1), args.get(2), End::Right, End::Left),
            )
        }
        "blmpop" => mpop(args, now)?,
        // The sorted set three, which are the same three shapes again with a
        // different collection under them. `BZPOPMIN` reads its keys up to the
        // timeout the way `BLPOP` does, and `BZMPOP` counts them the way
        // `BLMPOP` does.
        "bzpopmin" | "bzpopmax" => {
            let end = zsets::end_of_name(spec.name);
            let deadline = timeout(args.get(last), now)?;
            (deadline, Block::zpop((1..last).map(|i| args.get(i)), end))
        }
        "bzmpop" => {
            let deadline = timeout(args.get(1), now)?;
            let (end, from, to, count) = zsets::parse_mpop(args, 2)?;
            (
                deadline,
                Block::zmpop((from..to).map(|i| args.get(i)), end, count),
            )
        }
        // The table and this match are checked against each other by
        // `cargo xtask check`, so a name reaching here is a table row without a
        // handler and there is nothing sensible to answer.
        _ => return Err(args::syntax()),
    };

    let db = session.db();
    if block.now(server.striped(db), now, out)? {
        return Ok(Flow::Continue);
    }
    server.park(session.id(), db, deadline, block);
    Ok(Flow::Block)
}

/// `BLMPOP timeout numkeys key [key ...] LEFT|RIGHT [COUNT count]`.
///
/// The same parse as `LMPOP` shifted along by one, including the check that the
/// key count leaves room for the direction behind it. `BLMPOP 0 2 k LEFT` names
/// two keys and only gives one, so the word that should have been the direction
/// is a key and there is no direction left, which Redis calls a syntax error
/// rather than anything about counts.
fn mpop(args: Args<'_>, now: u64) -> Result<(Option<u64>, Block)> {
    let deadline = timeout(args.get(1), now)?;
    let numkeys = match args.int(2) {
        Ok(n) if n > 0 => usize::try_from(n).unwrap_or(usize::MAX),
        _ => return Err(Error::new(Code::Invalid, BAD_NUMKEYS)),
    };
    if numkeys >= args.len() - 3 {
        return Err(args::syntax());
    }
    let at = 3 + numkeys;
    let end = end_of(args.get(at))?;
    let mut want = 1usize;
    if at + 1 < args.len() {
        if args.len() != at + 3 || !args::is(args.get(at + 1), b"count") {
            return Err(args::syntax());
        }
        want = match args.int(at + 2) {
            Ok(n) if n > 0 => usize::try_from(n).unwrap_or(usize::MAX),
            _ => return Err(Error::new(Code::Invalid, BAD_MPOP_COUNT)),
        };
    }
    Ok((
        deadline,
        Block::mpop((3..at).map(|i| args.get(i)), end, want),
    ))
}

/// `WAIT numreplicas timeout` and `WAITAOF numlocal numreplicas timeout`.
///
/// Both of them ask the same question, which is whether this connection's writes
/// have got somewhere durable, and both of them answer zero here. There are no
/// replicas because there is no replication, and there is no append only file
/// because `appendonly` is fixed at `no`, so nothing can ever move either count
/// off zero and there is nothing to wait for. Redis in the same state gives the
/// same numbers, it just takes the timeout to do it, and that is registered as
/// D-25.
///
/// What is not a formality is the argument checking, because that is what a
/// client sees when it gets something wrong, and the three numbers are read by
/// three different Redis helpers with three different complaints. `numlocal` is
/// a range and says so. `numreplicas` is a positive number for `WAITAOF` and any
/// number at all for `WAIT`, where a negative one is accepted and satisfied on
/// the spot because zero replicas is already more than it asked for. The timeout
/// is milliseconds here and not the seconds the list commands take, so it does
/// not go through [`timeout`] above, and a negative one is refused with its own
/// message rather than the range one.
fn replication(name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
    let aof = name == "waitaof";
    // `WAITAOF` has one number in front of the two `WAIT` has, and everything
    // after it is in the same place, so the offset is the whole difference.
    let at = usize::from(aof);
    let mut wants_local = false;
    if aof {
        let local = whole(args.get(1))?;
        if !(0..=1).contains(&local) {
            return Err(Error::new(Code::Invalid, NOT_ZERO_OR_ONE));
        }
        wants_local = local == 1;
    }
    let replicas = whole(args.get(at + 1))?;
    if aof && replicas < 0 {
        return Err(Error::new(Code::Invalid, NOT_POSITIVE));
    }
    let ms = whole(args.get(at + 2)).map_err(|_| Error::new(Code::Invalid, TIMEOUT_NOT_AN_INT))?;
    if ms < 0 {
        return Err(Error::new(Code::Invalid, NEGATIVE));
    }
    // The one complaint here that is about the server rather than about the
    // arguments, and the reason it comes last is that Redis reads all three
    // arguments before it looks at itself. `appendonly` is `no` here and cannot
    // be set, so asking to wait for a local copy is asking for something that
    // cannot happen rather than something that has not happened yet.
    if wants_local {
        return Err(Error::new(Code::Invalid, NO_AOF));
    }
    if aof {
        // Two integers and not a map, whichever protocol is in use. The local
        // count is first and it is zero for the same reason the other one is:
        // this server has no append only file to be behind.
        out.array(2);
        out.int(0);
        out.int(0);
    } else {
        out.int(0);
    }
    Ok(())
}

/// A whole number argument, with the message Redis gives when it is not one.
fn whole(arg: &[u8]) -> Result<i64> {
    num::parse_i64(arg).ok_or_else(|| Error::new(Code::Invalid, NOT_AN_INT))
}

/// The moment to give up at, or `None` for a wait with no end to it.
///
/// Seconds as a float on the wire and a millisecond deadline here. Redis reads
/// it as a long double, refuses a negative one, multiplies by a thousand and
/// refuses what will not fit in an `i64`, and treats a timeout of exactly zero
/// as no timeout at all. All four of those are visible from a client:
///
/// - `-0.0` is not negative, so it is accepted, and it is zero, so it waits
///   forever. `-0.1` is refused.
/// - `1e400` and `inf` parse, so they are not the not-a-float error, and both
///   are further away than an `i64` of milliseconds reaches, so they are the out
///   of range one.
/// - `0.0000001` is a real timeout however small, so it expires on the next turn
///   of the loop rather than waiting for anything.
fn timeout(arg: &[u8], now: u64) -> Result<Option<u64>> {
    let Some(secs) = num::parse_f64(arg) else {
        return Err(Error::new(Code::Invalid, NOT_A_FLOAT));
    };
    if secs < 0.0 {
        return Err(Error::new(Code::Invalid, NEGATIVE));
    }
    let ms = secs * 1000.0;
    // `>` rather than a negated `<=`, and the two are not the same: an infinite
    // timeout is greater than the bound and lands here, while a NaN would be
    // neither, which is why the parse refuses one before this line is reached.
    if ms > i64::MAX as f64 {
        return Err(Error::new(Code::Invalid, OUT_OF_RANGE));
    }
    if ms <= 0.0 {
        return Ok(None);
    }
    Ok(Some(now.saturating_add(ms as u64)))
}

/// What a parked client is still trying to do.
enum Want {
    /// `BLPOP` and `BRPOP`: one element off the first key that has one, with the
    /// reply saying which key that turned out to be.
    Pop { end: End },
    /// `BLMOVE` and `BRPOPLPUSH`: one element, onto an end of another list.
    Move { dst: Vec<u8>, from: End, to: End },
    /// `BLMOVEM`: a block of them, onto an end of another list.
    ///
    /// The only want in this file where how many elements are there decides
    /// whether the client is ready, rather than just whether any are. `COUNT`
    /// takes what has arrived and so wakes on the first push, and `EXACTLY`
    /// waits until the source actually holds the whole block.
    MoveM { dst: Vec<u8>, mv: Movem },
    /// `BLMPOP`: up to `count` elements off the first key that has any.
    Mpop { end: End, count: usize },
    /// `BZPOPMIN` and `BZPOPMAX`: one member and its score off the first sorted
    /// set that has one, with the reply saying which key that turned out to be.
    ZPop { end: ZEnd },
    /// `BZMPOP`: up to `count` members off the first sorted set that has any.
    ZMpop { end: ZEnd, count: usize },
    /// `XREAD BLOCK` and `XREADGROUP BLOCK`: whatever has arrived on any of the
    /// streams since the ID this asked from.
    ///
    /// Unlike the other six this takes nothing away, so several clients parked
    /// on one stream all get the same entry rather than one of them getting it.
    /// That is the whole point of a stream over a list, and it costs nothing
    /// here because the attempt is a read.
    XRead(streams::Reads),
}

impl Want {
    /// Try to do it now.
    ///
    /// `Ok(true)` means a reply was written and the client is finished with.
    /// `Ok(false)` means there was nothing to take and nothing was written.
    ///
    /// `strict` is the difference between the two callers. The command handler
    /// passes `true`, so `BLPOP string 0` is a `WRONGTYPE` on the spot the way
    /// `LPOP string` is. The retry passes `false`, so a key somebody has since
    /// made into a set is skipped rather than turned into an error on a command
    /// that was accepted seconds ago. That is what a running Redis does: a
    /// `SADD` to a key a client is blocked on leaves it blocked, and it times
    /// out in its own time.
    ///
    /// # Errors
    ///
    /// Whatever the keyspace says, which under `strict` includes a key of
    /// another type.
    fn attempt(
        &self,
        keys: &[Vec<u8>],
        db: &mut Db,
        now: u64,
        out: &mut Out,
        strict: bool,
    ) -> Result<bool> {
        match self {
            // The one arm that needs to know what time it is, because a group
            // read records when each entry was handed out. The other six take
            // an element off a collection and the clock does not come into it.
            Want::XRead(r) => streams::read(db, keys, r, now, strict, out),
            Want::Pop { end } => {
                for key in keys {
                    if !ready(db, key, strict)? {
                        continue;
                    }
                    out.array(2);
                    out.bulk(key);
                    db.at(key).pop_into(key, *end, 1, |e| element(out, e))?;
                    return Ok(true);
                }
                Ok(false)
            }
            Want::Mpop { end, count } => {
                for key in keys {
                    if !ready(db, key, strict)? {
                        continue;
                    }
                    out.array(2);
                    out.bulk(key);
                    let mark = out.len();
                    let n = db
                        .at(key)
                        .pop_into(key, *end, *count, |e| element(out, e))?;
                    out.close_array(mark, n);
                    return Ok(true);
                }
                Ok(false)
            }
            // Three elements and not two, because `BZPOPMIN` puts the key, the
            // member and the score side by side rather than pairing the last
            // two. That is Redis's shape and it is not the shape `ZPOPMIN` has.
            Want::ZPop { end } => {
                for key in keys {
                    if !zready(db, key, strict)? {
                        continue;
                    }
                    out.array(3);
                    out.bulk(key);
                    db.at(key).zpop(key, *end, 1, |m, sc| {
                        member(out, m);
                        out.double(sc);
                    })?;
                    return Ok(true);
                }
                Ok(false)
            }
            Want::ZMpop { end, count } => {
                for key in keys {
                    if !zready(db, key, strict)? {
                        continue;
                    }
                    out.array(2);
                    out.bulk(key);
                    let mark = out.len();
                    let n = db.at(key).zpop(key, *end, *count, |m, sc| {
                        out.array(2);
                        member(out, m);
                        out.double(sc);
                    })?;
                    out.close_array(mark, n);
                    return Ok(true);
                }
                Ok(false)
            }
            // The source's length first, so that an empty source never reaches
            // the destination's type check. `BLMOVE empty string LEFT RIGHT 0.1`
            // times out on a running Redis rather than answering `WRONGTYPE`,
            // because the destination is only looked at once there is something
            // to put in it, and this order gives that answer.
            Want::Move { dst, from, to } => {
                let src = &keys[0];
                if !ready(db, src, strict)? {
                    return Ok(false);
                }
                match db.lmove(src, dst, *from, *to, |v| out.bulk(v)) {
                    Ok(true) => Ok(true),
                    // The source had something in it a line ago and this is the
                    // only thread that could have taken it.
                    Ok(false) => Ok(false),
                    Err(e) if strict => Err(e),
                    // The destination is not a list any more. Nothing was taken,
                    // because `lmove` checks the destination before it pops, so
                    // the client goes back to waiting with the queue as it was.
                    Err(_) => Ok(false),
                }
            }
            // The same shape as `Move` with a different question about the
            // source. `ready` asks whether there is anything and that is not
            // enough here, because an `EXACTLY` client is not ready until the
            // whole block has arrived, and asking it any earlier would take
            // nothing and answer nothing while looking like it had tried.
            Want::MoveM { dst, mv } => {
                let src = &keys[0];
                let have = match db.at(src).llen(src) {
                    Ok(n) => n,
                    Err(e) if strict => return Err(e),
                    Err(_) => return Ok(false),
                };
                // Not ready is not the same as nothing to do, so the
                // destination is never looked at from here. `BLMOVEM empty
                // string LEFT RIGHT 0.1` times out on a running 8.10.1 rather
                // than answering `WRONGTYPE`, and so does an `EXACTLY` whose
                // source is short, both of which were measured.
                if have == 0 || (mv.exactly && have < mv.count) {
                    return Ok(false);
                }
                let mark = out.len();
                let mut n = 0;
                match db.lmovem(src, dst, *mv, |v| {
                    out.bulk(v);
                    n += 1;
                }) {
                    Ok(_) => {}
                    Err(e) if strict => return Err(e),
                    // As `Move`: the destination stopped being a list while
                    // this client waited, and nothing was taken.
                    Err(_) => {
                        out.truncate(mark);
                        return Ok(false);
                    }
                }
                out.close_array(mark, n);
                Ok(true)
            }
        }
    }
}

/// Whether this key is a list with something in it.
///
/// A key of the wrong type is an error to the command handler and not one to the
/// retry, which is the whole of what `strict` decides.
fn ready(db: &mut Db, key: &[u8], strict: bool) -> Result<bool> {
    match db.at(key).llen(key) {
        Ok(n) => Ok(n > 0),
        Err(e) if strict => Err(e),
        Err(_) => Ok(false),
    }
}

/// The same for a sorted set, which has its own emptiness to ask about.
fn zready(db: &mut Db, key: &[u8], strict: bool) -> Result<bool> {
    match db.at(key).zcard(key) {
        Ok(n) => Ok(n > 0),
        Err(e) if strict => Err(e),
        Err(_) => Ok(false),
    }
}

/// One element as the client sees it, the same as [`super::lists`] writes it.
#[inline]
fn element(out: &mut Out, e: Entry<'_>) {
    match e {
        Entry::Int(n) => out.bulk_int(n),
        Entry::Str(s) => out.bulk(s),
    }
}

/// One member as the client sees it, the same as [`super::zsets`] writes it.
#[inline]
fn member(out: &mut Out, m: Member<'_>) {
    match m {
        Member::Int(n) => out.bulk_int(n),
        Member::Str(s) => out.bulk(s),
    }
}

/// A parsed blocking command, ready to be tried or to be parked.
pub struct Block {
    /// The keys, already copied out of the connection's read buffer.
    ///
    /// This is the allocation blocking costs and it is once per block rather
    /// than once per attempt. The arguments are slices of a buffer that is
    /// reused as soon as the batch is over, and a waiter outlives the batch.
    keys: Vec<Vec<u8>>,
    want: Want,
}

impl Block {
    /// `BLPOP` and `BRPOP`.
    fn pop<'a>(keys: impl Iterator<Item = &'a [u8]>, end: End) -> Block {
        Block {
            keys: owned(keys),
            want: Want::Pop { end },
        }
    }

    /// `BLMPOP`.
    fn mpop<'a>(keys: impl Iterator<Item = &'a [u8]>, end: End, count: usize) -> Block {
        Block {
            keys: owned(keys),
            want: Want::Mpop { end, count },
        }
    }

    /// `BZPOPMIN` and `BZPOPMAX`.
    fn zpop<'a>(keys: impl Iterator<Item = &'a [u8]>, end: ZEnd) -> Block {
        Block {
            keys: owned(keys),
            want: Want::ZPop { end },
        }
    }

    /// `BZMPOP`.
    fn zmpop<'a>(keys: impl Iterator<Item = &'a [u8]>, end: ZEnd, count: usize) -> Block {
        Block {
            keys: owned(keys),
            want: Want::ZMpop { end, count },
        }
    }

    /// `BLMOVE` and `BRPOPLPUSH`.
    fn moved(src: &[u8], dst: &[u8], from: End, to: End) -> Block {
        yo_alloc::allow(|| Block {
            keys: vec![src.to_vec()],
            want: Want::Move {
                dst: dst.to_vec(),
                from,
                to,
            },
        })
    }

    /// `BLMOVEM`.
    fn movem(src: &[u8], dst: &[u8], mv: Movem) -> Block {
        yo_alloc::allow(|| Block {
            keys: vec![src.to_vec()],
            want: Want::MoveM {
                dst: dst.to_vec(),
                mv,
            },
        })
    }

    /// Do it now if it can be done now.
    ///
    /// # Errors
    ///
    /// A key of another type, which is an error rather than a wait.
    fn now(&self, db: &mut Db, now: u64, out: &mut Out) -> Result<bool> {
        self.want.attempt(&self.keys, db, now, out, true)
    }

    /// `XREAD BLOCK` and `XREADGROUP BLOCK`, whose keys and IDs were read
    /// together by [`streams::parse_read`] because neither makes sense alone.
    fn xread(keys: Vec<Vec<u8>>, reads: streams::Reads) -> Block {
        Block {
            keys,
            want: Want::XRead(reads),
        }
    }
}

/// The keys a blocking command named, copied so they outlive the read buffer.
fn owned<'a>(keys: impl Iterator<Item = &'a [u8]>) -> Vec<Vec<u8>> {
    yo_alloc::allow(|| keys.map(<[u8]>::to_vec).collect())
}

/// One parked client.
struct Waiter {
    /// The client id, which is never reused.
    client: u64,
    /// The slot its reply buffer is on, which is.
    conn: u32,
    /// The database it was on when it blocked. A push into another database is
    /// not this client's push, even when the key has the same name.
    db: usize,
    /// The millisecond to give up at, or `None` for `BLPOP key 0`, which waits
    /// for as long as the connection is open.
    deadline: Option<u64>,
    keys: Vec<Vec<u8>>,
    want: Want,
}

/// Every parked client, oldest first.
///
/// The order is the order they blocked in and it is the order they are served
/// in, which is what makes a queue with several workers on it fair: two clients
/// blocked on the same key take the two elements a `RPUSH q first second` adds
/// in the order they arrived. A `Vec` is the right structure for that while the
/// list is short, and it is short, because a waiter is a client doing nothing.
#[derive(Default)]
pub struct Waiters {
    list: Vec<Waiter>,
}

/// Where the reply to a parked client has to go.
#[derive(Debug, Clone, Copy)]
pub struct Parked {
    /// The slot holding its reply buffer.
    pub conn: u32,
    /// The client that was on that slot when it blocked.
    pub client: u64,
}

impl Waiters {
    /// Whether anybody is waiting.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// How many clients are parked, which is what `INFO clients` calls
    /// `blocked_clients`.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Where the waiter at `at` has to be answered.
    ///
    /// # Panics
    ///
    /// If `at` is past the end, which only a caller that ignored [`Waiters::len`]
    /// can manage.
    #[must_use]
    pub fn at(&self, at: usize) -> Parked {
        let w = &self.list[at];
        Parked {
            conn: w.conn,
            client: w.client,
        }
    }

    /// The database the waiter at `at` blocked on.
    ///
    /// # Panics
    ///
    /// As [`Waiters::at`].
    #[must_use]
    pub fn db_of(&self, at: usize) -> usize {
        self.list[at].db
    }

    /// Take a waiter off the list.
    ///
    /// # Panics
    ///
    /// As [`Waiters::at`].
    pub fn drop_at(&mut self, at: usize) {
        self.list.remove(at);
    }

    /// Take off every waiter belonging to a client that has gone.
    ///
    /// Called when a connection closes rather than left for the deadline sweep
    /// to find, because a `BLPOP key 0` on a connection nobody will ever write
    /// to again has no deadline to be found by.
    pub fn forget(&mut self, client: u64) {
        self.list.retain(|w| w.client != client);
    }

    /// Say which slot the waiter this client just registered is answered on.
    ///
    /// The command layer knows which client blocked and the engine knows which
    /// slot that client is on, so the slot is filled in afterwards by the half
    /// that has it. A client can only be parked once, since it is not reading
    /// commands while it waits, so the search finds the one that was just added.
    pub fn bind(&mut self, client: u64, conn: u32) {
        if let Some(w) = self.list.iter_mut().rev().find(|w| w.client == client) {
            w.conn = conn;
        }
    }

    /// Park a client that could not be answered.
    ///
    /// The slot is filled in by [`Waiters::bind`] once the engine has it, so
    /// this leaves it at zero rather than pretending to know.
    fn park(&mut self, client: u64, db: usize, deadline: Option<u64>, block: Block) {
        yo_alloc::allow(|| {
            self.list.push(Waiter {
                client,
                conn: 0,
                db,
                deadline,
                keys: block.keys,
                want: block.want,
            });
        });
    }

    /// Try to answer the waiter at `at`, and say whether it is finished with.
    ///
    /// `true` means a reply is in `out` and the caller should take the waiter
    /// off the list, which covers both a client that got what it asked for and
    /// one that ran out of time.
    ///
    /// The attempt comes before the deadline, so a push that landed in the same
    /// millisecond the client gave up in serves it rather than racing it.
    ///
    /// # Panics
    ///
    /// As [`Waiters::at`].
    fn try_serve(&self, at: usize, dbs: &mut [Db], now: u64, out: &mut Out) -> bool {
        let w = &self.list[at];
        let mark = out.len();
        match w.want.attempt(&w.keys, &mut dbs[w.db], now, out, false) {
            Ok(true) => return true,
            Ok(false) => {}
            // `strict` is off, so nothing in there returns an error today.
            // Putting the buffer back is what makes it safe to be wrong about
            // that later.
            Err(_) => out.truncate(mark),
        }
        if w.deadline.is_some_and(|d| now >= d) {
            // A null array for all six, `BLMOVE` and `BRPOPLPUSH` included,
            // even though what they send when they succeed is a single element.
            // That is Redis's and it is not what reading the reply schema would
            // suggest: a RESP2 client sees `*-1` and not `$-1`.
            out.nil_array();
            return true;
        }
        false
    }
}

impl Server {
    /// The clock reading this batch is working against.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.clock.now_ms()
    }

    /// Who is parked, for the engine and for `INFO`.
    #[must_use]
    pub const fn waiters(&self) -> &Waiters {
        &self.waiters
    }

    /// The same, for the engine, which is what binds and forgets them.
    pub const fn waiters_mut(&mut self) -> &mut Waiters {
        &mut self.waiters
    }

    /// Park a client on a command that could not be answered yet.
    pub(super) fn park(&mut self, client: u64, db: usize, deadline: Option<u64>, block: Block) {
        self.waiters.park(client, db, deadline, block);
    }

    /// Try to answer the waiter at `at`, writing into the buffer the engine
    /// found for it, and say whether it is finished with.
    ///
    /// The engine cannot reach the databases and this cannot reach the
    /// connections, so the two meet here: the caller hands in one connection's
    /// reply buffer and gets back whether to unpark the client behind it.
    ///
    /// # Panics
    ///
    /// If `at` is not a waiter.
    pub fn serve_waiter(&mut self, at: usize, now: u64, out: &mut Out) -> bool {
        // Serving a waiter pops an element, which makes garbage, and it happens
        // outside `execute` so nothing else has marked the database for the
        // maintenance turn.
        self.dirty |= 1u64 << self.waiters.db_of(at);
        self.waiters.try_serve(at, &mut self.dbs, now, out)
    }
}
