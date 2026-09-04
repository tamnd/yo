//! The stream commands, on the wire.
//!
//! The same shape as [`super::lists`]: the name has been looked up and the
//! arity has been checked, so this turns arguments into a call on [`Db`]
//! and the answer into a reply. What makes this group longer than the others is
//! not the storage, which is already there, it is that fifteen commands share
//! four argument shapes and disagree about every one of them at the edges.
//!
//! # Three prefixes and not two
//!
//! Every other group answers with `ERR` or `WRONGTYPE`, which is what
//! [`super::write_error`] knows how to write. Streams add `NOGROUP` and
//! `BUSYGROUP`, and a client library branches on those the way it branches on
//! `WRONGTYPE`, so they are written where they are decided rather than routed
//! through an error code. That is the same thing `NOPROTO`, `WRONGPASS` and
//! `OOM` already do.
//!
//! There are three different `NOGROUP` sentences and they are not
//! interchangeable. [`yo_kv::streams`] holds all three with a note on each
//! saying which commands use it.
//!
//! # Two kinds of nothing, again
//!
//! `XRANGE missing - + COUNT 0` is an empty array and `XRANGE there - + COUNT 0`
//! is a null array, because Redis looks the key up before it notices the count.
//! `XREAD` with nothing new is a null array and `XREADGROUP` reading its own
//! history is the key with an empty list beside it. `XADD NOMKSTREAM` on a
//! missing key is a null. All four were read off a running 8.10.1, because the
//! reasoning gives the wrong answer for at least the first two.
//!
//! # Nothing is collected on the way out
//!
//! An entry goes into the out buffer as it is walked and the array header is
//! written afterwards through [`Out::close_array`], so `XRANGE key - +` over a
//! million entries is never a `Vec` of a million anything on this thread. The
//! two exceptions are `XCLAIM` and `XAUTOCLAIM`, which get a list of IDs back
//! from the keyspace because the claim has to finish before the reply can start.
//! Those allocate, and say so at the call.

use yo_common::num::{DIGITS_MAX, u64_digits};
use yo_common::{Code, Error, Result, num};
use yo_kv::stream::{Consumer, Fate, Fields, Filter, Group, Id, Refs, Retry, Stream};
use yo_kv::streams::{self as kv, Add, Claim, Read, Start, Trim};
use yo_kv::{Db, Entry};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What a trim says about a `MAXLEN` below zero. The full stop is Redis's.
const MAXLEN_NEGATIVE: &str = "The MAXLEN argument must be >= 0.";
/// And about a `LIMIT` below zero.
const LIMIT_NEGATIVE: &str = "The LIMIT argument must be >= 0.";
/// And about a `LIMIT` on a trim that was not asked to be approximate.
const LIMIT_NO_APPROX: &str = "syntax error, LIMIT cannot be used without the special ~ option";
/// And about a `LIMIT` with no trim at all behind it.
const LIMIT_NO_STRATEGY: &str =
    "syntax error, LIMIT cannot be used without specifying a trimming strategy";
/// And about a command that asked for both trims at once.
const BOTH_STRATEGIES: &str =
    "syntax error, MAXLEN and MINID options at the same time are not compatible";
/// What `XSETID` says about an `ENTRIESADDED` below zero.
const ADDED_NOT_POSITIVE: &str = "entries_added must be positive";
/// What `XGROUP` says about an `ENTRIESREAD` that is neither a count nor `-1`.
const READ_NOT_POSITIVE: &str = "value for ENTRIESREAD must be positive or -1";
/// What `XAUTOCLAIM` says about a `COUNT` of zero or less.
const AUTOCLAIM_COUNT: &str = "COUNT must be > 0";
/// The four `XCLAIM` numbers that carry their own complaint, and the one
/// `XAUTOCLAIM` shares the shape of.
const BAD_MIN_IDLE: &str = "Invalid min-idle-time argument for XCLAIM";
const BAD_MIN_IDLE_AUTO: &str = "Invalid min-idle-time argument for XAUTOCLAIM";
const BAD_IDLE: &str = "Invalid IDLE option argument for XCLAIM";
const BAD_TIME: &str = "Invalid TIME option argument for XCLAIM";
const BAD_RETRY: &str = "Invalid RETRYCOUNT option argument for XCLAIM";
/// What a range says about a bound whose exclusive form has nowhere to go.
///
/// `XRANGE key (18446744073709551615-18446744073709551615 +` asks for the ID
/// after the last one there is, and there is not one.
const BAD_INTERVAL_START: &str = "invalid start ID for the interval";
const BAD_INTERVAL_END: &str = "invalid end ID for the interval";
/// What `XREAD` says about a `BLOCK` it cannot read. Milliseconds as a whole
/// number here, where `BLPOP` takes seconds as a float, so the two commands
/// refuse the same argument with different sentences and `XREAD BLOCK 1.5` is an
/// error on a real server.
const TIMEOUT_NOT_AN_INT: &str = "timeout is not an integer or out of range";
/// And about a negative one.
const TIMEOUT_NEGATIVE: &str = "timeout is negative";
/// What `XREADGROUP` says about the two IDs that mean nothing to it.
///
/// Long, and worth having verbatim, because it is the sentence a client sees
/// when it sends `XREAD`'s habits to `XREADGROUP`.
const DOLLAR_MEANINGLESS: &str = "The $ ID is meaningless in the context of XREADGROUP: you want to read the history of this consumer by specifying a proper ID, or use the > ID to get new messages. The $ ID would just return an empty result set.";
const PLUS_MEANINGLESS: &str = "The + ID is meaningless in the context of XREADGROUP: you want to read the history of this consumer by specifying a proper ID, or use the + ID to get new messages. The + ID would just return an empty result set.";
/// What `XREADGROUP` says when the `GROUP` it needs is not there.
const MISSING_GROUP: &str = "Missing GROUP option for XREADGROUP";
/// What `XDELEX` and `XACKDEL` say about a `numids` that is not a count.
const IDS_NOT_POSITIVE: &str = "Number of IDs must be a positive integer";
/// And about one that does not match how many IDs came after it.
const IDS_MISMATCH: &str = "The `numids` parameter must match the number of arguments";
/// `XNACK` asks both of those questions in its own words, which are not those.
const NACK_IDS_NOT_POSITIVE: &str = "numids must be a positive integer";
const NACK_IDS_MISMATCH: &str = "number of IDs doesn't match numids";
/// And it reads the word in front of `IDS` as a position rather than an option,
/// so a wrong one gets this rather than a syntax error.
const NACK_MODE: &str = "mode must be SILENT, FAIL, or FATAL";
/// What it says about a `RETRYCOUNT` below zero.
const NACK_RETRY_NEGATIVE: &str = "Invalid RETRYCOUNT value, must be >= 0";
/// And what `XREAD` says about the two options that are not its.
const GROUP_IS_NOT_XREAD: &str =
    "The GROUP option is only supported by XREADGROUP. You called XREAD instead.";
const NOACK_IS_NOT_XREAD: &str =
    "The NOACK option is only supported by XREADGROUP. You called XREAD instead.";

/// How many field value pairs an `XADD` writes without allocating.
///
/// The keyspace takes a slice of pairs, so the pairs have to be somewhere, and a
/// `Vec` would be an allocation on the dispatch thread for every `XADD` that
/// runs. Thirty two is past what any real producer sends: a stream entry is a
/// record and a record with more than thirty two fields is a document. Beyond it
/// the allocation is taken deliberately rather than the command refused.
const INLINE_FIELDS: usize = 32;

/// Run one stream command.
///
/// `now` comes in rather than being read here because the clock lives on the
/// server and this is handed one database, and because every command in the
/// batch has to agree about what time it is: an `XADD` and the `XCLAIM` behind
/// it reading two different milliseconds would make an entry that was idle
/// before it was delivered.
///
/// # Errors
///
/// Whatever the keyspace says, plus the argument complaints above.
pub(super) fn execute(
    db: &mut Db,
    spec: &Spec,
    args: Args<'_>,
    now: u64,
    out: &mut Out,
) -> Result<()> {
    match spec.name {
        "xadd" => xadd(db, args, now, out)?,
        "xlen" => {
            let key = args.get(1);
            out.uint(db.at(key).stream(key)?.map_or(0, Stream::len));
        }
        "xdel" => {
            let key = args.get(1);
            out.uint(db.at(key).xdel(key, ids(args, 2)?)?);
        }
        "xdelex" => delex(db, args, out)?,
        "xackdel" => ackdel(db, args, out)?,
        "xnack" => nack(db, args, out)?,
        "xtrim" => xtrim(db, args, out)?,
        "xrange" => range(db, args, false, out)?,
        "xrevrange" => range(db, args, true, out)?,
        "xack" => {
            let (key, group) = (args.get(1), args.get(2));
            out.uint(db.at(key).xack(key, group, ids(args, 3)?)?);
        }
        "xsetid" => setid(db, args, out)?,
        "xgroup" => group(db, args, now, out)?,
        "xinfo" => info(db, args, now, out)?,
        "xpending" => pending(db, args, now, out)?,
        "xclaim" => claim(db, args, now, out)?,
        "xautoclaim" => autoclaim(db, args, now, out)?,
        other => unreachable!("the table sent {other} to the stream group"),
    }
    Ok(())
}

// ---------------------------------------------------------------- writing

/// `XADD key [NOMKSTREAM] [trim] id field value [field value ...]`.
fn xadd(db: &mut Db, args: Args<'_>, now: u64, out: &mut Out) -> Result<()> {
    let stripe = db.at(args.get(1));
    let node = stripe.stream_limits().max_node_entries;
    let opts = trimming(args, 2, true, node)?;
    let at = opts.at;
    // The ID and at least one pair, in pairs. Redis calls this an arity error
    // and not a syntax error even though the arity in the table already passed,
    // because the table cannot count what comes after a variable run of options.
    if at + 2 >= args.len() || !(args.len() - at - 1).is_multiple_of(2) {
        return Err(args::wrong_arity("xadd"));
    }
    let id = add_id(args.get(at))?;
    let key = args.get(1);
    let pairs = (args.len() - at - 1) / 2;
    let field = |i: usize| (args.get(at + 1 + i * 2), args.get(at + 2 + i * 2));

    let written = if pairs <= INLINE_FIELDS {
        let mut buf: [(&[u8], &[u8]); INLINE_FIELDS] = [(b"", b""); INLINE_FIELDS];
        for (i, slot) in buf[..pairs].iter_mut().enumerate() {
            *slot = field(i);
        }
        stripe.xadd(key, id, &buf[..pairs], opts.trim, opts.mkstream, now)?
    } else {
        // A producer sending more than thirty two fields an entry gets one
        // allocation for the pair list and nothing else.
        let fields: Vec<(&[u8], &[u8])> = yo_alloc::allow(|| (0..pairs).map(field).collect());
        stripe.xadd(key, id, &fields, opts.trim, opts.mkstream, now)?
    };
    match written {
        Some(id) => id_out(out, id),
        // `NOMKSTREAM` on a key that is not there, which is a null and not a
        // zero, so a producer can tell "nobody is consuming this yet" from
        // "the write happened".
        None => out.nil(),
    }
    Ok(())
}

/// `XTRIM key strategy`.
fn xtrim(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let stripe = db.at(args.get(1));
    let node = stripe.stream_limits().max_node_entries;
    let opts = trimming(args, 2, false, node)?;
    // A trailing argument the option loop did not recognise, and a command with
    // no strategy at all, are both a syntax error. `XTRIM key` cannot reach here
    // because the arity stops it, but `XTRIM key LIMIT 5` can and does.
    if opts.at != args.len() || matches!(opts.trim, Trim::None) {
        return Err(args::syntax());
    }
    out.uint(stripe.xtrim(args.get(1), opts.trim)?);
    Ok(())
}

/// What the trim options came to, and where the arguments after them start.
struct Trimmed {
    trim: Trim,
    /// `XADD`'s `NOMKSTREAM`, which `XTRIM` does not take.
    mkstream: bool,
    /// The first argument that was not one of these options, which is the ID for
    /// `XADD` and the end of the command for `XTRIM`.
    at: usize,
}

/// The trimming options `XADD` and `XTRIM` share.
///
/// Redis reads these with a `moreargs` count rather than with a fixed shape, and
/// that is visible from a client: `XTRIM key MAXLEN ~` has one argument after
/// the keyword, so the `~` is read as the threshold and the complaint is that it
/// is not a number. Two arguments after it and the `~` is the approximate flag.
/// This mirrors that rather than tidying it up, because a client library that
/// builds `MAXLEN ~ n` and gets the count wrong should see what a real server
/// tells it.
///
/// The implicit limit is the other thing that has to be copied rather than
/// invented. An approximate trim with no `LIMIT` gets one anyway, at a hundred
/// times `stream-node-max-entries`, so `XTRIM key MAXLEN ~ 0` on a stream of
/// thirty thousand removes ten thousand and leaves twenty. `LIMIT 0` means no
/// limit, and an exact trim never has one.
fn trimming(args: Args<'_>, at: usize, xadd: bool, node: usize) -> Result<Trimmed> {
    let mut maxlen: Option<u64> = None;
    let mut minid: Option<Id> = None;
    let mut approx = false;
    let mut limit: Option<u64> = None;
    let mut limited = false;
    let mut mkstream = true;
    let mut i = at;

    while i < args.len() {
        let opt = args.get(i);
        let more = args.len() - i - 1;
        if xadd && args::is(opt, b"nomkstream") {
            mkstream = false;
            i += 1;
        } else if (args::is(opt, b"maxlen") || args::is(opt, b"minid")) && more > 0 {
            if maxlen.is_some() || minid.is_some() {
                return Err(Error::new(Code::Invalid, BOTH_STRATEGIES));
            }
            let len = args::is(opt, b"maxlen");
            let mut n = i + 1;
            let next = args.get(n);
            if more >= 2 && (next == b"~" || next == b"=") {
                approx = next == b"~";
                n += 1;
            }
            if len {
                maxlen = Some(non_negative(args.int(n)?, MAXLEN_NEGATIVE)?);
            } else {
                minid = Some(strict_id(args.get(n))?);
            }
            i = n + 1;
        } else if args::is(opt, b"limit") && more > 0 {
            limit = Some(non_negative(args.int(i + 1)?, LIMIT_NEGATIVE)?);
            limited = true;
            i += 2;
        } else {
            break;
        }
    }

    if limited {
        // The strategy complaint first, which is Redis's order and not the
        // obvious one: `XTRIM key LIMIT 5` has no `~` either, and what it is
        // told about is the missing trim.
        if maxlen.is_none() && minid.is_none() {
            return Err(Error::new(Code::Invalid, LIMIT_NO_STRATEGY));
        }
        if !approx {
            return Err(Error::new(Code::Invalid, LIMIT_NO_APPROX));
        }
    }
    let limit = match (approx, limit) {
        // An exact trim takes no limit at all, and stopping one early would mean
        // it had not done what it was asked.
        (false, _) => None,
        (true, Some(0)) => None,
        (true, Some(n)) => Some(n),
        (true, None) => (node > 0).then(|| 100 * node as u64),
    };
    let trim = match (maxlen, minid) {
        (Some(len), _) => Trim::MaxLen {
            len,
            exact: !approx,
            limit,
        },
        (_, Some(id)) => Trim::MinId {
            id,
            exact: !approx,
            limit,
        },
        _ => Trim::None,
    };
    Ok(Trimmed {
        trim,
        mkstream,
        at: i,
    })
}

/// `XSETID key id [ENTRIESADDED n] [MAXDELETEDID id]`.
fn setid(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let last = strict_id(args.get(2))?;
    let mut added = None;
    let mut deleted = None;
    let mut i = 3;
    while i < args.len() {
        if i + 1 >= args.len() {
            return Err(args::syntax());
        }
        let opt = args.get(i);
        if args::is(opt, b"entriesadded") {
            added = Some(non_negative(args.int(i + 1)?, ADDED_NOT_POSITIVE)?);
        } else if args::is(opt, b"maxdeletedid") {
            deleted = Some(strict_id(args.get(i + 1))?);
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }
    let key = args.get(1);
    db.at(key).xsetid(key, last, added, deleted)?;
    out.ok();
    Ok(())
}

// ---------------------------------------------------------------- reading

/// `XRANGE key start end [COUNT n]` and `XREVRANGE key end start [COUNT n]`.
///
/// The keyspace takes the low bound first either way, so the two argument orders
/// are swapped here once rather than in every walk behind it.
fn range(db: &mut Db, args: Args<'_>, rev: bool, out: &mut Out) -> Result<()> {
    let (lo, hi) = if rev { (3, 2) } else { (2, 3) };
    let start = bound(args.get(lo), true)?;
    let end = bound(args.get(hi), false)?;
    // Redis reads this as a loop rather than as one optional pair, so
    // `XRANGE k - + COUNT 2 COUNT 1` is legal and the last one wins.
    let mut count: Option<usize> = None;
    let mut i = 4;
    while i < args.len() {
        if !args::is(args.get(i), b"count") || i + 1 >= args.len() {
            return Err(args::syntax());
        }
        count = Some(args.int(i + 1)?.max(0).unsigned_abs() as usize);
        i += 2;
    }

    // The key before the count, which is the order Redis looks at them in and is
    // why a missing key answers an empty array while a key that is there with a
    // count of zero answers a null array.
    let key = args.get(1);
    let stripe = db.at(key);
    if stripe.stream(key)?.is_none() {
        out.array(0);
        return Ok(());
    }
    if count == Some(0) {
        out.nil_array();
        return Ok(());
    }
    let mark = out.len();
    let n = stripe.xrange_into(key, start, end, count, rev, |id, fields| {
        entry(out, id, fields);
        true
    })?;
    out.close_array(mark, n);
    Ok(())
}

/// `XPENDING key group` and `XPENDING key group [IDLE ms] start end count [consumer]`.
///
/// Two commands under one name. The short one is a summary of the whole pending
/// list and the long one is the list itself, and they have nothing in common
/// past the first two arguments.
fn pending(db: &mut Db, args: Args<'_>, now: u64, out: &mut Out) -> Result<()> {
    let (key, name) = (args.get(1), args.get(2));
    if args.len() == 3 {
        return summary(db, key, name, out);
    }
    // Redis takes six to nine arguments here and reads the consumer only when
    // the count comes out exactly right, so a tenth argument is an error and an
    // extra one in the middle is quietly ignored. This is that rule and not a
    // tidier one, because a client sending the extra argument gets an answer
    // from a real server and should get the same one here.
    if args.len() > 9 {
        return Err(args::syntax());
    }
    let idle = args::is(args.get(3), b"idle");
    let at = if idle { 5 } else { 3 };
    if args.len() < at + 3 {
        return Err(args::syntax());
    }
    let want = Filter {
        min_idle: if idle {
            args.int(4)?.max(0).unsigned_abs()
        } else {
            0
        },
        start: bound(args.get(at), true)?,
        end: bound(args.get(at + 1), false)?,
        count: Some(args.int(at + 2)?.max(0).unsigned_abs() as usize),
        owner: None,
    };
    // The consumer filter is a slot rather than a name, because a NACK records
    // which slot holds it and comparing slots is what the walk does per entry.
    let mut want = want;
    if args.len() == at + 4 {
        let who = args.get(at + 3);
        let Some(s) = db.at(key).stream(key)? else {
            nogroup(out, &kv::no_key_or_group(key, name));
            return Ok(());
        };
        let Some(g) = s.group(name) else {
            nogroup(out, &kv::no_key_or_group(key, name));
            return Ok(());
        };
        match g.slot(who) {
            Some(slot) => want.owner = Some(slot),
            // A consumer nobody has heard of holds nothing, which is an empty
            // list rather than an error.
            None => {
                out.array(0);
                return Ok(());
            }
        }
    }

    let mark = out.len();
    let seen = db
        .at(key)
        .xpending_into(key, name, want, now, |id, nack, c| {
            out.array(4);
            id_out(out, id);
            // An entry `XNACK` released has no owner and no idle time. Redis writes
            // an empty name and a minus one for it rather than leaving it out, so a
            // sweeper can see there is work here that nobody has picked up.
            match c {
                Some(c) => {
                    out.bulk(c.name());
                    out.uint(nack.idle(now));
                }
                None => {
                    out.bulk(b"");
                    out.int(-1);
                }
            }
            out.uint(nack.count());
            true
        })?;
    match seen {
        Some(n) => out.close_array(mark, n),
        None => nogroup(out, &kv::no_key_or_group(key, name)),
    }
    Ok(())
}

/// `XPENDING key group`, which is four values about the whole list.
///
/// Empty is four nulls and not a zero with three empty things, and the per
/// consumer counts are bulk strings where every other count in the group is an
/// integer. Both of those are Redis's and neither is what writing this from the
/// documentation would produce.
fn summary(db: &mut Db, key: &[u8], name: &[u8], out: &mut Out) -> Result<()> {
    let Some(s) = db.at(key).stream(key)? else {
        nogroup(out, &kv::no_key_or_group(key, name));
        return Ok(());
    };
    let Some(g) = s.group(name) else {
        nogroup(out, &kv::no_key_or_group(key, name));
        return Ok(());
    };
    out.array(4);
    out.uint(g.pending_len() as u64);
    match g.pending_bounds() {
        Some((low, high)) => {
            id_out(out, low);
            id_out(out, high);
            let mark = out.len();
            let mut n = 0;
            let mut prev: Option<&[u8]> = None;
            while let Some(who) = next_name(g.pending_counts().map(|(name, _)| name), prev) {
                let held = g
                    .pending_counts()
                    .find(|(name, _)| *name == who)
                    .map_or(0, |(_, n)| n);
                out.array(2);
                out.bulk(who);
                // A bulk string, which is what Redis sends here and only here.
                out.bulk_u64(held as u64);
                prev = Some(who);
                n += 1;
            }
            out.close_array(mark, n);
        }
        None => {
            out.nil();
            out.nil();
            out.nil_array();
        }
    }
    Ok(())
}

// ------------------------------------------------- the pending list, in 8.x

/// `XDELEX key [KEEPREF|DELREF|ACKED] IDS numids id [id ...]`.
///
/// `XDEL` with a say in what happens to the consumer groups that were handed the
/// entry, and one integer back per ID instead of a count, because a caller that
/// asked for `ACKED` needs to know which of its IDs stayed.
fn delex(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    // The key is looked up before any of the syntax is read, which is visible
    // from a client: `XDELEX somestring BOGUS IDS 1 1-1` is a wrong type and not
    // a syntax error, and so is the same command with a `numids` of zero.
    let here = db.at(key).stream(key)?.is_some();
    let (refs, at) = refs_and_ids(args, 2)?;
    let n = args.len() - at;
    // A key that is not there answers before the IDs are read, so
    // `XDELEX missing IDS 2 bad bad` is two minus ones and not a complaint about
    // the IDs. Redis only parses them once it has something to look them up in.
    if !here {
        missing(out, n);
        return Ok(());
    }
    let ids = ids_in(args, at, args.len())?;
    out.array(n);
    db.at(key)
        .xdelex(key, refs, ids, |fate| out.int(fate.code()))
}

/// `XACKDEL key group [KEEPREF|DELREF|ACKED] IDS numids id [id ...]`.
///
/// The same reply, about a different question. Here minus one means the group
/// was not holding the ID rather than that the stream does not have it, so an
/// entry that is sitting in the stream unread answers minus one and stays.
fn ackdel(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, name) = (args.get(1), args.get(2));
    let here = db.at(key).stream(key)?.is_some();
    let (refs, at) = refs_and_ids(args, 3)?;
    let n = args.len() - at;
    if !here {
        missing(out, n);
        return Ok(());
    }
    let ids = ids_in(args, at, args.len())?;
    out.array(n);
    db.at(key)
        .xackdel(key, name, refs, ids, |fate| out.int(fate.code()))
}

/// `XNACK key group <SILENT|FAIL|FATAL> IDS numids id [id ...] [RETRYCOUNT n] [FORCE]`.
///
/// The other half of `XACK`: a consumer saying it could not do the work, so the
/// entry goes back to the group rather than out of it. It comes back with no
/// owner and reading as idle for longer than any `min-idle-time` a claim can
/// name, which is what puts it at the front of the next `XAUTOCLAIM`.
///
/// The bookmark does not move, so a `>` read will not hand it out again. That is
/// the whole design: a released entry is offered to a claim and not to the
/// group's next reader, which is the only way it cannot be delivered twice over.
fn nack(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, name) = (args.get(1), args.get(2));
    // Before the mode word, which is visible: `XNACK k nosuchgroup BOGUS ...` is
    // told about the group and not about the mode.
    if !db
        .at(key)
        .stream(key)?
        .is_some_and(|s| s.group(name).is_some())
    {
        nogroup(out, &kv::no_key_or_group(key, name));
        return Ok(());
    }
    // The three words differ in one thing only, the delivery count, and `SILENT`
    // takes one off it rather than putting it back to zero. That is only visible
    // on an entry that has been handed out more than once, so it is easy to
    // measure the wrong rule off a message that has only been delivered once.
    let mode = match args.get(3) {
        w if args::is(w, b"silent") => Retry::Down,
        w if args::is(w, b"fail") => Retry::Keep,
        w if args::is(w, b"fatal") => Retry::Max,
        _ => return Err(bad(NACK_MODE)),
    };
    if !args::is(args.get(4), b"ids") {
        return Err(args::syntax());
    }
    let want = count(args, 5, NACK_IDS_NOT_POSITIVE)?;
    let (at, left) = (6, args.len().saturating_sub(6));
    if left < want {
        return Err(bad(NACK_IDS_MISMATCH));
    }
    let end = at + want;
    let ids = ids_in(args, at, end)?;

    // Everything past the IDs is an option, which is why one ID too many is
    // reported as an option nobody recognises rather than as a count that does
    // not add up. `XDELEX` refuses the same shape as a syntax error.
    let mut retry = None;
    let mut force = false;
    let mut i = end;
    while i < args.len() {
        let opt = args.get(i);
        let more = args.len() - i - 1;
        if args::is(opt, b"force") {
            force = true;
            i += 1;
        } else if args::is(opt, b"retrycount") && more > 0 {
            retry = Some(Retry::At(non_negative(
                args.int(i + 1)?,
                NACK_RETRY_NEGATIVE,
            )?));
            i += 2;
        } else {
            return Err(unrecognised("XNACK", opt));
        }
    }
    // `RETRYCOUNT` wins over the word, so `FATAL ... RETRYCOUNT 3` leaves the
    // count at three and not at the ceiling `FATAL` on its own would set.
    let done = db
        .at(key)
        .xnack(key, name, retry.unwrap_or(mode), force, ids)?;
    out.uint(done.expect("the group was there a moment ago"));
    Ok(())
}

/// The `[KEEPREF|DELREF|ACKED] IDS numids id [id ...]` tail two of the three
/// share, read from `at`. Answers what the word said and where the IDs start.
///
/// The condition is one word and not a set of flags, so `KEEPREF DELREF` is a
/// syntax error rather than the second one winning. `XNACK`'s `FORCE` is a flag
/// and repeating that is fine, which is a difference between two commands that
/// went in at the same time and is worth copying rather than smoothing over.
fn refs_and_ids(args: Args<'_>, at: usize) -> Result<(Refs, usize)> {
    let word = args.get(at);
    let (refs, at) = if args::is(word, b"keepref") {
        (Refs::Keep, at + 1)
    } else if args::is(word, b"delref") {
        (Refs::Drop, at + 1)
    } else if args::is(word, b"acked") {
        (Refs::Acked, at + 1)
    } else {
        (Refs::Keep, at)
    };
    if !args::is(args.get(at), b"ids") {
        return Err(args::syntax());
    }
    let want = count(args, at + 1, IDS_NOT_POSITIVE)?;
    let start = at + 2;
    let left = args.len().saturating_sub(start);
    if left < want {
        return Err(bad(IDS_MISMATCH));
    }
    if left > want {
        return Err(args::syntax());
    }
    Ok((refs, start))
}

/// A `numids`, which has to be a number above zero and carries its own sentence.
fn count(args: Args<'_>, at: usize, msg: &'static str) -> Result<usize> {
    match num::parse_i64(args.get(at)) {
        Some(n) if n > 0 => Ok(usize::try_from(n).unwrap_or(usize::MAX)),
        _ => Err(bad(msg)),
    }
}

/// An array of `n` minus ones, which is what the two delete commands answer for
/// a key that is not there.
fn missing(out: &mut Out, n: usize) {
    out.array(n);
    for _ in 0..n {
        out.int(Fate::Missing.code());
    }
}

/// `ERR Unrecognized <COMMAND> option '<opt>'`, which two of these commands use
/// for anything they did not expect after the arguments they count.
fn unrecognised(name: &str, opt: &[u8]) -> Error {
    yo_alloc::allow(|| {
        Error::fmt(
            Code::Invalid,
            format_args!(
                "Unrecognized {name} option '{}'",
                String::from_utf8_lossy(opt)
            ),
        )
    })
}

// ---------------------------------------------------------------- claiming

/// `XCLAIM key group consumer min-idle-time id [id ...] [options]`.
///
/// The IDs are read until one does not parse and everything after that is an
/// option, which is how Redis tells the two apart. It means `XCLAIM k g c 0 -`
/// complains about an unrecognised option rather than about an ID, and that is
/// the sentence a real server sends.
fn claim(db: &mut Db, args: Args<'_>, now: u64, out: &mut Out) -> Result<()> {
    let (key, name, who) = (args.get(1), args.get(2), args.get(3));
    let min_idle = millis(args.int(4).map_err(|_| bad(BAD_MIN_IDLE))?);
    let mut at = 5;
    while at < args.len() && Id::parse(args.get(at), 0).is_some() {
        at += 1;
    }
    let count = at - 5;

    let mut how = Claim {
        group: name,
        consumer: who,
        min_idle,
        time: now,
        ..Claim::default()
    };
    let mut justid = false;
    let mut last: Option<Id> = None;
    let mut i = at;
    while i < args.len() {
        let opt = args.get(i);
        let more = args.len() - i - 1;
        if args::is(opt, b"force") {
            how.force = true;
            i += 1;
        } else if args::is(opt, b"justid") {
            justid = true;
            i += 1;
        } else if args::is(opt, b"idle") && more > 0 {
            let ms = millis(args.int(i + 1).map_err(|_| bad(BAD_IDLE))?);
            how.time = now.saturating_sub(ms);
            i += 2;
        } else if args::is(opt, b"time") && more > 0 {
            how.time = millis(args.int(i + 1).map_err(|_| bad(BAD_TIME))?);
            i += 2;
        } else if args::is(opt, b"retrycount") && more > 0 {
            how.retry = Some(millis(args.int(i + 1).map_err(|_| bad(BAD_RETRY))?));
            i += 2;
        } else if args::is(opt, b"lastid") && more > 0 {
            // Undocumented and real. Redis's replication sends it so that a
            // replica's group bookmark ends up where the primary's is.
            last = Some(strict_id(args.get(i + 1))?);
            i += 2;
        } else {
            return Err(unrecognised("XCLAIM", opt));
        }
    }
    // `JUSTID` is not only a reply shape. Redis reads it as "I am not really
    // taking delivery of this", so the delivery count stays where it was.
    how.bump = !justid;

    // The one place in this file that collects, because the claim has to finish
    // before the reply can start: an entry that turns out to have been deleted
    // leaves the pending list on the way past, and the reply only carries what
    // survived that.
    let (took, mut gone) = yo_alloc::allow(|| {
        let mut gone = Vec::new();
        let ids: Vec<Id> = (5..5 + count)
            .filter_map(|j| Id::parse(args.get(j), 0))
            .collect();
        (db.at(key).xclaim(key, &ids, how, now, &mut gone), gone)
    });
    gone.clear();
    let Some(took) = took? else {
        nogroup(out, &kv::no_key_or_group(key, name));
        return Ok(());
    };
    if let Some(id) = last {
        move_bookmark(db, key, name, id)?;
    }
    entries(db, key, &took, justid, out)
}

/// `XAUTOCLAIM key group consumer min-idle-time start [COUNT n] [JUSTID]`.
///
/// Three values back rather than one: where to carry on from, what was claimed,
/// and what was dropped for no longer being in the stream. The third one is why
/// a sweep with this converges instead of handing the same dead entry round
/// forever.
fn autoclaim(db: &mut Db, args: Args<'_>, now: u64, out: &mut Out) -> Result<()> {
    let (key, name, who) = (args.get(1), args.get(2), args.get(3));
    let min_idle = millis(args.int(4).map_err(|_| bad(BAD_MIN_IDLE_AUTO))?);
    let start = bound(args.get(5), true)?;
    let mut count = 100;
    let mut justid = false;
    let mut i = 6;
    while i < args.len() {
        let opt = args.get(i);
        if args::is(opt, b"justid") {
            justid = true;
            i += 1;
        } else if args::is(opt, b"count") && i + 1 < args.len() {
            let n = args.int(i + 1)?;
            if n <= 0 {
                return Err(Error::new(Code::Invalid, AUTOCLAIM_COUNT));
            }
            count = usize::try_from(n).unwrap_or(usize::MAX);
            i += 2;
        } else {
            return Err(args::syntax());
        }
    }

    let how = Claim {
        group: name,
        consumer: who,
        min_idle,
        time: now,
        bump: !justid,
        ..Claim::default()
    };
    let (claimed, gone) = yo_alloc::allow(|| {
        let mut gone = Vec::new();
        (
            db.at(key)
                .xautoclaim(key, start, how, count, now, &mut gone),
            gone,
        )
    });
    let Some((cursor, took)) = claimed? else {
        nogroup(out, &kv::no_key_or_group(key, name));
        return Ok(());
    };

    out.array(3);
    // `0-0` at the end of the sweep, which is what a caller loops until.
    id_out(out, cursor.unwrap_or(Id::MIN));
    entries(db, key, &took, justid, out)?;
    out.array(gone.len());
    for &id in &gone {
        id_out(out, id);
    }
    Ok(())
}

/// The claimed IDs as the reply carries them, which is either the entries or the
/// bare IDs.
///
/// Every ID here was in the stream a moment ago, so the lookup finds it. Doing
/// it per ID rather than as one walk is a probe each into a structure that is
/// already warm, and it keeps the claim and the reply from having to hold the
/// stream at the same time.
fn entries(db: &mut Db, key: &[u8], took: &[Id], justid: bool, out: &mut Out) -> Result<()> {
    if justid {
        out.array(took.len());
        for &id in took {
            id_out(out, id);
        }
        return Ok(());
    }
    let mark = out.len();
    let mut n = 0;
    for &id in took {
        let found = db
            .at(key)
            .xrange_into(key, id, id, Some(1), false, |id, fields| {
                entry(out, id, fields);
                true
            })?;
        n += found;
    }
    out.close_array(mark, n);
    Ok(())
}

/// `XCLAIM LASTID`, which moves the group's bookmark forward and never back.
fn move_bookmark(db: &mut Db, key: &[u8], name: &[u8], id: Id) -> Result<()> {
    let Some(s) = db.at(key).stream_mut(key)? else {
        return Ok(());
    };
    if let Some(g) = s.group_mut(name)
        && id > g.last_id()
    {
        let read = g.entries_read();
        g.set_id(id, read);
    }
    Ok(())
}

// ------------------------------------------------------------------ groups

/// `XGROUP CREATE|SETID|DESTROY|CREATECONSUMER|DELCONSUMER|HELP`.
///
/// Two different complaints about the arguments and they are not
/// interchangeable. Below the subcommand's own arity is an arity error naming
/// the pair, `xgroup|create`, because Redis holds subcommands in its command
/// table with an arity each. Enough arguments in a shape the handler will not
/// take is the longer sentence pointing at `XGROUP HELP`.
fn group(db: &mut Db, args: Args<'_>, now: u64, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    let n = args.len();
    if args::is(sub, b"create") {
        arity(n, -5, "create")?;
        return create(db, args, out);
    }
    if args::is(sub, b"setid") {
        arity(n, -5, "setid")?;
        // Five arguments or seven, so the `ENTRIESREAD` pair is all or nothing.
        if n != 5 && n != 7 {
            return Err(unknown_or_arity(sub, "XGROUP"));
        }
        let read = entries_read(args, 5, n)?;
        let at = start_at(args.get(4))?;
        let key = args.get(2);
        return match db.at(key).xgroup_setid(key, args.get(3), at, read)? {
            Some(()) => {
                out.ok();
                Ok(())
            }
            None => {
                nogroup(out, &kv::no_group(args.get(3), args.get(2)));
                Ok(())
            }
        };
    }
    if args::is(sub, b"destroy") {
        arity(n, 4, "destroy")?;
        let key = args.get(2);
        out.int(i64::from(db.at(key).xgroup_destroy(key, args.get(3))?));
        return Ok(());
    }
    if args::is(sub, b"createconsumer") {
        arity(n, 5, "createconsumer")?;
        let key = args.get(2);
        let made = db
            .at(key)
            .xgroup_create_consumer(key, args.get(3), args.get(4), now)?;
        return match made {
            Some(made) => {
                out.int(i64::from(made));
                Ok(())
            }
            None => {
                nogroup(out, &kv::no_group(args.get(3), args.get(2)));
                Ok(())
            }
        };
    }
    if args::is(sub, b"delconsumer") {
        arity(n, 5, "delconsumer")?;
        let key = args.get(2);
        return match db
            .at(key)
            .xgroup_del_consumer(key, args.get(3), args.get(4))?
        {
            Some(held) => {
                out.uint(held);
                Ok(())
            }
            None => {
                nogroup(out, &kv::no_group(args.get(3), args.get(2)));
                Ok(())
            }
        };
    }
    if args::is(sub, b"help") {
        arity(n, 2, "help")?;
        super::server::help(out, GROUP_HELP);
        return Ok(());
    }
    Err(args::unknown_subcommand(sub, "XGROUP"))
}

/// `XGROUP CREATE key group id|$ [MKSTREAM] [ENTRIESREAD n]`.
///
/// The two options come in either order, so this is a loop and not a pair of
/// positions.
fn create(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut mkstream = false;
    let mut read = None;
    let mut i = 5;
    while i < args.len() {
        let opt = args.get(i);
        if args::is(opt, b"mkstream") {
            mkstream = true;
            i += 1;
        } else if args::is(opt, b"entriesread") && i + 1 < args.len() {
            read = entries_read(args, i + 1, i + 2)?;
            i += 2;
        } else {
            return Err(unknown_or_arity(args.get(1), "XGROUP"));
        }
    }
    let at = start_at(args.get(4))?;
    let key = args.get(2);
    if db
        .at(key)
        .xgroup_create(key, args.get(3), at, mkstream, read)?
    {
        out.ok();
    } else {
        // Its own prefix, because a client that races another one to create the
        // same group treats this as "somebody else got there" and not as an
        // error to report.
        out.error_line(b"BUSYGROUP ", kv::GROUP_EXISTS.as_bytes());
    }
    Ok(())
}

/// `ENTRIESREAD n`, where `-1` means the counter is not known.
///
/// The two are different and the difference is visible: a group whose counter is
/// unknown reports a null for it and works its lag out from where its bookmark
/// sits, and a group whose counter is zero is claiming to have read nothing.
fn entries_read(args: Args<'_>, at: usize, end: usize) -> Result<Option<u64>> {
    if at >= end {
        return Ok(None);
    }
    if !args::is(args.get(at - 1), b"entriesread") {
        return Err(unknown_or_arity(args.get(1), "XGROUP"));
    }
    let n = args.int(at)?;
    if n < -1 {
        return Err(Error::new(Code::Invalid, READ_NOT_POSITIVE));
    }
    Ok((n >= 0).then(|| n.unsigned_abs()))
}

/// `$` or an ID, which is where a group's bookmark is being put.
fn start_at(arg: &[u8]) -> Result<Start> {
    if arg == b"$" {
        return Ok(Start::Last);
    }
    Ok(Start::At(strict_id(arg)?))
}

// -------------------------------------------------------------------- info

/// `XINFO STREAM|GROUPS|CONSUMERS|HELP`.
fn info(db: &mut Db, args: Args<'_>, now: u64, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    let n = args.len();
    if args::is(sub, b"stream") {
        arity(n, -3, "stream")?;
        let mut count = 10;
        let full = n > 3;
        if full {
            if !args::is(args.get(3), b"full") || (n != 4 && n != 6) {
                return Err(unknown_or_arity(sub, "XINFO"));
            }
            if n == 6 {
                if !args::is(args.get(4), b"count") {
                    return Err(unknown_or_arity(sub, "XINFO"));
                }
                let want = args.int(5)?;
                // Zero or less means every one of them, which is the opposite of
                // what a count of zero means to `XRANGE`.
                count = if want <= 0 {
                    usize::MAX
                } else {
                    usize::try_from(want).unwrap_or(usize::MAX)
                };
            }
        }
        let key = args.get(2);
        let Some(s) = db.at(key).stream(key)? else {
            return Err(Error::new(Code::NotFound, kv::NO_SUCH_KEY));
        };
        if full {
            full_info(s, count, out);
        } else {
            stream_info(s, out);
        }
        return Ok(());
    }
    if args::is(sub, b"groups") {
        arity(n, 3, "groups")?;
        let key = args.get(2);
        let Some(s) = db.at(key).stream(key)? else {
            return Err(Error::new(Code::NotFound, kv::NO_SUCH_KEY));
        };
        groups_info(s, out);
        return Ok(());
    }
    if args::is(sub, b"consumers") {
        arity(n, 4, "consumers")?;
        let (key, name) = (args.get(2), args.get(3));
        let Some(s) = db.at(key).stream(key)? else {
            return Err(Error::new(Code::NotFound, kv::NO_SUCH_KEY));
        };
        let Some(g) = s.group(name) else {
            nogroup(out, &kv::no_group(name, key));
            return Ok(());
        };
        consumers_info(g, now, out);
        return Ok(());
    }
    if args::is(sub, b"help") {
        arity(n, 2, "help")?;
        super::server::help(out, INFO_HELP);
        return Ok(());
    }
    Err(args::unknown_subcommand(sub, "XINFO"))
}

/// `XINFO STREAM key`, which is ten pairs about the stream and two entries.
///
/// Redis 8.10 sends sixteen. The six that are not here are its idempotency
/// tracking, `idmp-duration`, `idmp-maxsize`, `pids-tracked`, `iids-tracked`,
/// `iids-added` and `iids-duplicates`, and there is none of that here, so they
/// are left out rather than answered with zeros that would claim a feature. That
/// is D-26.
///
/// `radix-tree-keys` and `radix-tree-nodes` are the node count and are what the
/// storage here has rather than what a rax would report, which is D-27.
fn stream_info(s: &Stream, out: &mut Out) {
    out.map(10);
    header(s, out);
    out.bulk(b"groups");
    out.uint(s.groups().count() as u64);
    out.bulk(b"first-entry");
    one(s, s.first_id(), out);
    out.bulk(b"last-entry");
    one(s, s.top_id(), out);
}

/// `XINFO STREAM key FULL [COUNT n]`, which is the same seven fields with the
/// entries and the whole group tree behind them.
fn full_info(s: &Stream, count: usize, out: &mut Out) {
    out.map(9);
    header(s, out);
    out.bulk(b"entries");
    let mark = out.len();
    let n = s.range(Id::MIN, Id::MAX, Some(count), |id, fields| {
        entry(out, id, fields);
        true
    });
    out.close_array(mark, n);

    out.bulk(b"groups");
    let mark = out.len();
    let mut wrote = 0;
    let mut prev: Option<&[u8]> = None;
    while let Some(name) = next_name(s.groups().map(|(name, _)| name), prev) {
        let g = s.group(name).expect("a name the walk just found");
        out.map(8);
        out.bulk(b"name");
        out.bulk(name);
        out.bulk(b"last-delivered-id");
        id_out(out, g.last_id());
        out.bulk(b"entries-read");
        maybe_uint(out, g.entries_read());
        out.bulk(b"lag");
        maybe_uint(out, s.lag(g));
        out.bulk(b"pel-count");
        out.uint(g.pending_len() as u64);
        // How many of those nobody is holding, which is what `XNACK` makes.
        out.bulk(b"nacked-count");
        out.uint(g.nacked_len() as u64);
        out.bulk(b"pending");
        let at = out.len();
        let want = Filter {
            count: Some(count),
            ..Filter::default()
        };
        // The delivery time and not the idle time, which is the difference
        // between this list and the one `XPENDING` writes.
        let seen = g.pending_range(want, 0, |id, nack, c| {
            out.array(4);
            id_out(out, id);
            // The delivery time of a released entry is the zero it was put back
            // at, and its consumer is the empty name, the same as `XPENDING`.
            out.bulk(c.map_or(&b""[..], Consumer::name));
            out.uint(nack.time());
            out.uint(nack.count());
            true
        });
        out.close_array(at, seen);
        out.bulk(b"consumers");
        let at = out.len();
        let mut consumers = 0;
        let mut before: Option<&[u8]> = None;
        while let Some(who) = next_name(g.consumers().map(Consumer::name), before) {
            let c = g
                .consumer(g.slot(who).expect("a name the walk just found"))
                .expect("the slot that name is on");
            out.map(5);
            out.bulk(b"name");
            out.bulk(who);
            out.bulk(b"seen-time");
            out.uint(c.seen());
            out.bulk(b"active-time");
            active(out, c);
            out.bulk(b"pel-count");
            out.uint(c.len() as u64);
            out.bulk(b"pending");
            let here = out.len();
            let mut held = 0;
            for id in c.pending().take(count) {
                let Some(nack) = g.nack(id) else { continue };
                out.array(3);
                id_out(out, id);
                out.uint(nack.time());
                out.uint(nack.count());
                held += 1;
            }
            out.close_array(here, held);
            before = Some(who);
            consumers += 1;
        }
        out.close_array(at, consumers);
        prev = Some(name);
        wrote += 1;
    }
    out.close_array(mark, wrote);
}

/// The seven fields both `XINFO STREAM` forms start with, in Redis's order.
fn header(s: &Stream, out: &mut Out) {
    out.bulk(b"length");
    out.uint(s.len());
    out.bulk(b"radix-tree-keys");
    out.uint(s.nodes() as u64);
    out.bulk(b"radix-tree-nodes");
    // An empty stream reports one node and no keys on a real server, because the
    // rax always has its root.
    out.uint(s.nodes().max(1) as u64);
    out.bulk(b"last-generated-id");
    id_out(out, s.last_id());
    out.bulk(b"max-deleted-entry-id");
    id_out(out, s.max_deleted_id());
    out.bulk(b"entries-added");
    out.uint(s.added());
    out.bulk(b"recorded-first-entry-id");
    id_out(out, s.first_id().unwrap_or(Id::MIN));
}

/// `XINFO GROUPS key`.
fn groups_info(s: &Stream, out: &mut Out) {
    let mark = out.len();
    let mut n = 0;
    let mut prev: Option<&[u8]> = None;
    while let Some(name) = next_name(s.groups().map(|(name, _)| name), prev) {
        let g = s.group(name).expect("a name the walk just found");
        out.map(6);
        out.bulk(b"name");
        out.bulk(name);
        out.bulk(b"consumers");
        out.uint(g.consumers().count() as u64);
        out.bulk(b"pending");
        out.uint(g.pending_len() as u64);
        out.bulk(b"last-delivered-id");
        id_out(out, g.last_id());
        out.bulk(b"entries-read");
        maybe_uint(out, g.entries_read());
        out.bulk(b"lag");
        maybe_uint(out, s.lag(g));
        prev = Some(name);
        n += 1;
    }
    out.close_array(mark, n);
}

/// `XINFO CONSUMERS key group`.
///
/// `idle` is how long since the consumer said anything and `inactive` is how
/// long since it was given anything, and a consumer that has never been given
/// anything reports minus one for the second rather than the moment it turned
/// up. That is the difference between a worker that is stuck and one that has
/// nothing to do, which is the whole reason Redis keeps two clocks per consumer.
fn consumers_info(g: &Group, now: u64, out: &mut Out) {
    let mark = out.len();
    let mut n = 0;
    let mut prev: Option<&[u8]> = None;
    while let Some(who) = next_name(g.consumers().map(Consumer::name), prev) {
        let c = g
            .consumer(g.slot(who).expect("a name the walk just found"))
            .expect("the slot that name is on");
        out.map(4);
        out.bulk(b"name");
        out.bulk(who);
        out.bulk(b"pending");
        out.uint(c.len() as u64);
        out.bulk(b"idle");
        out.uint(now.saturating_sub(c.seen()));
        out.bulk(b"inactive");
        match c.active() {
            Some(at) => out.uint(now.saturating_sub(at)),
            None => out.int(-1),
        }
        prev = Some(who);
        n += 1;
    }
    out.close_array(mark, n);
}

/// The next name in byte order after `prev`, or the first of all.
///
/// Redis holds groups and consumers in a rax, so it reports both in name order
/// for free. They are vectors here, in the order they were made, and a consumer
/// slot can never move because a NACK holds the index. So the order is found
/// rather than stored: one pass per name that answers the smallest name above
/// the last one written. That is quadratic in the number of names and it is the
/// right trade, because `XINFO` is a diagnostic command, the lists are short,
/// and the alternative is either an allocation per call on the dispatch thread
/// or a second index to keep in step for the benefit of a command nobody sends
/// in a loop.
fn next_name<'a, I>(names: I, prev: Option<&[u8]>) -> Option<&'a [u8]>
where
    I: Iterator<Item = &'a [u8]>,
{
    names.filter(|name| prev.is_none_or(|p| *name > p)).min()
}

/// One entry, or a null when the stream has none.
fn one(s: &Stream, id: Option<Id>, out: &mut Out) {
    let mut wrote = false;
    if let Some(id) = id {
        s.range(id, id, Some(1), |id, fields| {
            entry(out, id, fields);
            wrote = true;
            true
        });
    }
    if !wrote {
        out.nil();
    }
}

/// A counter that may not be known, which `XINFO` reports as a null.
fn maybe_uint(out: &mut Out, n: Option<u64>) {
    match n {
        Some(n) => out.uint(n),
        None => out.nil(),
    }
}

/// A consumer's active time, which is minus one when it has never had anything.
fn active(out: &mut Out, c: &Consumer) {
    match c.active() {
        Some(at) => out.uint(at),
        None => out.int(-1),
    }
}

// ------------------------------------------------------------------ reading

/// Where one stream in an `XREAD` or `XREADGROUP` list is being read from.
pub(super) enum At {
    /// `XREAD`: everything after this ID, which is what `$` and `+` resolve to.
    After(Id),
    /// `XREADGROUP >`: whatever the group has never handed out.
    New,
    /// `XREADGROUP <id>`: this consumer's own pending entries above the ID.
    Mine(Id),
    /// `XREADGROUP $` and `XREADGROUP +`, which are refused.
    ///
    /// Kept as a value rather than raised where it is read, because Redis looks
    /// the group up first and this is visible: `XREADGROUP GROUP g c STREAMS k +`
    /// at a stream with no group `g` is told about the group, and only a client
    /// that had the group right is told the `+` was meaningless.
    Meaningless(&'static str),
}

/// A parsed `XREAD` or `XREADGROUP`, without the keys.
///
/// The keys live on the waiter beside this, because the waiter list is walked by
/// key and a blocking read is the same shape as a blocking pop from there.
pub(super) struct Reads {
    /// One per key, in the order the keys were named.
    at: Vec<At>,
    count: Option<usize>,
    /// `GROUP g c`, which is what makes it an `XREADGROUP`.
    group: Option<(Vec<u8>, Vec<u8>)>,
    noack: bool,
}

/// A read that has been parsed and not yet tried.
pub(super) struct Parsed {
    /// Whether `BLOCK` was there at all, and the deadline it named.
    ///
    /// Two layers because there are three answers and not two: no `BLOCK` and
    /// answer now, `BLOCK 0` and wait for as long as it takes, and `BLOCK n`
    /// and wait until then.
    pub wait: Option<Option<u64>>,
    /// The stream names, copied out of the read buffer because a waiter outlives
    /// the batch that made it.
    pub keys: Vec<Vec<u8>>,
    /// Everything else the command said.
    pub reads: Reads,
}

/// Read `XREAD` or `XREADGROUP`, and say whether it was asked to wait.
///
/// The outer `Option` is whether `BLOCK` was there at all and the inner one is
/// whether it named a deadline, so a command with no `BLOCK` answers straight
/// away and a `BLOCK 0` waits for as long as the connection is open.
///
/// `$` and `+` are resolved here, once, against the stream as it is now. That is
/// not an optimisation, it is the meaning: `XREAD BLOCK 0 STREAMS k $` waits for
/// what arrives after the moment the command was sent, so resolving it again on
/// each retry would make it wait for what arrives after the last time anything
/// happened, and it would never answer.
///
/// # Errors
///
/// The option and ID complaints, and a key holding something that is not a
/// stream, which `$` finds because it has to look.
pub(super) fn parse_read(name: &str, args: Args<'_>, db: &mut Db, now: u64) -> Result<Parsed> {
    let grouped = name == "xreadgroup";
    let mut count = None;
    let mut wait = None;
    let mut group = None;
    let mut noack = false;
    let mut streams = None;
    let mut i = 1;
    while i < args.len() {
        let opt = args.get(i);
        let more = args.len() - i - 1;
        if args::is(opt, b"count") && more > 0 {
            // Zero or less is unlimited here, which is the opposite of what a
            // count of zero means to `XRANGE`.
            let n = args.int(i + 1)?;
            count = (n > 0).then(|| usize::try_from(n).unwrap_or(usize::MAX));
            i += 2;
        } else if args::is(opt, b"block") && more > 0 {
            let ms = num::parse_i64(args.get(i + 1)).ok_or_else(|| bad(TIMEOUT_NOT_AN_INT))?;
            if ms < 0 {
                return Err(bad(TIMEOUT_NEGATIVE));
            }
            // Milliseconds and a deadline, so nothing downstream has to know
            // which unit the client used.
            wait = Some((ms > 0).then(|| now.saturating_add(ms.unsigned_abs())));
            i += 2;
        } else if args::is(opt, b"group") && more > 1 {
            if !grouped {
                return Err(bad(GROUP_IS_NOT_XREAD));
            }
            group = Some((args.get(i + 1), args.get(i + 2)));
            i += 3;
        } else if args::is(opt, b"noack") {
            if !grouped {
                return Err(bad(NOACK_IS_NOT_XREAD));
            }
            noack = true;
            i += 1;
        } else if args::is(opt, b"streams") {
            streams = Some(i + 1);
            break;
        } else {
            return Err(args::syntax());
        }
    }
    let Some(at) = streams else {
        return Err(args::syntax());
    };
    if grouped && group.is_none() {
        return Err(bad(MISSING_GROUP));
    }
    // Keys and IDs in two runs of the same length, which is the one argument
    // shape in Redis that cannot be checked by counting from the front.
    let left = args.len() - at;
    if left == 0 || !left.is_multiple_of(2) {
        return Err(unbalanced(name));
    }
    let n = left / 2;

    let mut where_from = yo_alloc::allow(|| Vec::with_capacity(n));
    for j in 0..n {
        let key = args.get(at + j);
        let arg = args.get(at + n + j);
        where_from.push(if grouped {
            match arg {
                b">" => At::New,
                b"$" => At::Meaningless(DOLLAR_MEANINGLESS),
                b"+" => At::Meaningless(PLUS_MEANINGLESS),
                _ => At::Mine(strict_id(arg)?),
            }
        } else {
            At::After(match arg {
                b"$" => db.at(key).stream(key)?.map_or(Id::MIN, Stream::last_id),
                // The last entry rather than the next one, so `+` answers with
                // what is already there and then behaves like `$`. On a stream
                // with nothing in it there is no last entry to step back from,
                // and Redis falls back to the last ID it handed out.
                b"+" => {
                    let s = db.at(key).stream(key)?;
                    let last = s.as_ref().map_or(Id::MIN, |s| s.last_id());
                    s.and_then(Stream::top_id)
                        .and_then(Id::prev)
                        .unwrap_or(last)
                }
                _ => strict_id(arg)?,
            })
        });
    }
    let keys = yo_alloc::allow(|| (0..n).map(|j| args.get(at + j).to_vec()).collect());
    let group = yo_alloc::allow(|| group.map(|(g, c)| (g.to_vec(), c.to_vec())));
    Ok(Parsed {
        wait,
        keys,
        reads: Reads {
            at: where_from,
            count,
            group,
            noack,
        },
    })
}

/// Try to answer an `XREAD` or `XREADGROUP` now.
///
/// `Ok(true)` means a reply was written and the client is finished with, which
/// covers the `NOGROUP` case as well as the ordinary one. `Ok(false)` means
/// nothing was there and nothing was written, so the caller either waits or
/// answers with a null.
///
/// `strict` is what the two callers disagree about, the same as it is for the
/// list pops. The command handler passes `true`, so a key holding a string is a
/// `WRONGTYPE` on the spot. The retry passes `false`, so a key somebody has since
/// turned into a string is skipped and the client goes on waiting, which is what
/// a real server does: a blocked `XREAD` whose key is deleted and replaced with
/// a string times out rather than erroring.
///
/// The group is looked up for every key before any of them is read, because that
/// is the order Redis checks them in and it is visible: a two key `XREADGROUP`
/// where the second key has no such group answers `NOGROUP` and reads nothing
/// from the first.
///
/// # Errors
///
/// A key holding something that is not a stream, under `strict`.
pub(super) fn read(
    db: &mut Db,
    keys: &[Vec<u8>],
    r: &Reads,
    now: u64,
    strict: bool,
    out: &mut Out,
) -> Result<bool> {
    if let Some((name, _)) = &r.group {
        for key in keys {
            let missing = match db.at(key).stream(key) {
                Ok(Some(s)) => s.group(name).is_none(),
                Ok(None) => true,
                Err(e) if strict => return Err(e),
                Err(_) => true,
            };
            if missing {
                // The same sentence whether the group was never there or was
                // destroyed while this client waited on it, which was read off a
                // running server rather than assumed: a blocked `XREADGROUP`
                // whose group is dropped gets exactly this.
                nogroup(out, &kv::no_group_for_read(key, name));
                return Ok(true);
            }
        }
    }

    // After the group check and before anything is read, which is where Redis
    // notices it and is why this is not raised at the parse.
    for at in &r.at {
        if let At::Meaningless(msg) = at {
            return Err(bad(msg));
        }
    }

    let mark = out.len();
    let mut wrote = 0;
    for (key, at) in keys.iter().zip(&r.at) {
        let here = out.len();
        // RESP3 sends a map of stream name to entries and RESP2 sends an array
        // of two element arrays, so the pair has a wrapper on one protocol and
        // not on the other.
        if !out.proto().is_resp3() {
            out.array(2);
        }
        out.bulk(key);
        let body = out.len();
        let got = match one_stream(db, key, at, r, now, out) {
            Ok(n) => n,
            Err(e) if strict => return Err(e),
            Err(_) => {
                out.truncate(here);
                continue;
            }
        };
        match got {
            // A history read always names its stream, even with nothing to show,
            // because the empty list is the answer and not the absence of one.
            Some(n) => {
                out.close_array(body, n);
                wrote += 1;
            }
            None => out.truncate(here),
        }
    }
    if wrote == 0 {
        out.truncate(mark);
        return Ok(false);
    }
    out.close_map(mark, wrote);
    Ok(true)
}

/// One stream's entries, or `None` when it has nothing to report.
///
/// The array header is left to the caller, which is what lets a stream that had
/// nothing be dropped from the reply after the walk rather than counted before
/// it.
fn one_stream(
    db: &mut Db,
    key: &[u8],
    at: &At,
    r: &Reads,
    now: u64,
    out: &mut Out,
) -> Result<Option<usize>> {
    match at {
        At::After(after) => {
            let n = db.at(key).xread_into(key, *after, r.count, |id, fields| {
                entry(out, id, fields);
                true
            })?;
            Ok((n > 0).then_some(n))
        }
        At::Meaningless(_) => unreachable!("read refuses these before it walks"),
        At::New | At::Mine(_) => {
            let (group, consumer) = r.group.as_ref().expect("a group read names its group");
            let history = matches!(at, At::Mine(_));
            let want = Read {
                group,
                consumer,
                from: match at {
                    At::Mine(after) => kv::From::Pending(*after),
                    _ => kv::From::New,
                },
                count: r.count,
                noack: r.noack,
            };
            let n = db.at(key).xreadgroup_into(key, want, now, |id, fields| {
                match fields {
                    Some(fields) => entry(out, id, fields),
                    // A history read can name an entry that has since been
                    // deleted, and Redis writes the ID with a null beside it
                    // rather than leaving it out, so the consumer can see what
                    // it is still holding and acknowledge it.
                    None => {
                        out.array(2);
                        id_out(out, id);
                        out.nil();
                    }
                }
                true
            })?;
            Ok(match n {
                Some(n) if history || n > 0 => Some(n),
                _ => None,
            })
        }
    }
}

/// `ERR Unbalanced 'x' list of streams: ...`, which names the command and lists
/// the IDs that command takes.
fn unbalanced(name: &str) -> Error {
    if name == "xreadgroup" {
        return Error::new(
            Code::Invalid,
            "Unbalanced 'xreadgroup' list of streams: for each stream key an ID or '>' must be specified.",
        );
    }
    Error::new(
        Code::Invalid,
        "Unbalanced 'xread' list of streams: for each stream key an ID, '+', or '$' must be specified.",
    )
}

// ----------------------------------------------------------------- helpers

/// An ID exactly as `XADD` takes it, which is the one command with three forms.
fn add_id(arg: &[u8]) -> Result<Add> {
    if arg == b"*" {
        return Ok(Add::Auto);
    }
    // `5-*` is the next free sequence inside millisecond five, which is how a
    // producer that has its own millisecond writes several entries into it
    // without tracking the sequence itself.
    if let Some(ms) = arg.strip_suffix(b"-*") {
        return match Id::parse(ms, 0) {
            Some(id) => Ok(Add::Seq(id.ms)),
            None => Err(bad(kv::BAD_ID)),
        };
    }
    Ok(Add::At(strict_id(arg)?))
}

/// An ID with no special forms, which is what most of the group takes.
fn strict_id(arg: &[u8]) -> Result<Id> {
    Id::parse(arg, 0).ok_or_else(|| bad(kv::BAD_ID))
}

/// A range bound, which takes `-`, `+` and a `(` for an exclusive one.
///
/// The exclusive step happens after the missing sequence is filled in, and that
/// is visible: `XRANGE key - (6` still returns `6-0`, because `(6` is `6-` and
/// the largest sequence there is, stepped back by one. Getting that the other
/// way round would drop the whole of millisecond six.
fn bound(arg: &[u8], low: bool) -> Result<Id> {
    let (arg, open) = match arg.strip_prefix(b"(") {
        Some(rest) => (rest, true),
        None => (arg, false),
    };
    // `(-` and `(+` are refused rather than treated as the end they name,
    // because the ID after the last one and the ID before the first one are the
    // two that do not exist.
    if !open {
        if arg == b"-" {
            return Ok(Id::MIN);
        }
        if arg == b"+" {
            return Ok(Id::MAX);
        }
    }
    let id = Id::parse(arg, if low { 0 } else { u64::MAX }).ok_or_else(|| bad(kv::BAD_ID))?;
    if !open {
        return Ok(id);
    }
    let stepped = if low { id.next() } else { id.prev() };
    stepped.ok_or_else(|| {
        bad(if low {
            BAD_INTERVAL_START
        } else {
            BAD_INTERVAL_END
        })
    })
}

/// Every ID from `at` onwards, checked before any of them is used.
///
/// Redis reads all of them before it deletes or acknowledges anything, so
/// `XDEL key 1-1 nonsense` deletes nothing rather than deleting the first and
/// then complaining. Checking and then walking again is two passes over a short
/// argument list and no allocation, which is the trade this whole layer makes.
fn ids<'a>(args: Args<'a>, at: usize) -> Result<impl Iterator<Item = Id> + 'a> {
    ids_in(args, at, args.len())
}

/// The same over a run that stops short of the end, which is what the three 8.x
/// commands need: they count their IDs and take options after them.
fn ids_in<'a>(args: Args<'a>, at: usize, end: usize) -> Result<impl Iterator<Item = Id> + 'a> {
    for i in at..end {
        strict_id(args.get(i))?;
    }
    // Every one of these parsed a moment ago, so the second pass cannot fail and
    // the `filter_map` drops nothing.
    Ok((at..end).filter_map(move |i| Id::parse(args.get(i), 0)))
}

/// One entry as the client sees it, which is its ID and its fields laid flat.
///
/// The fields are one array and not a map on either protocol, which is Redis's
/// shape and not what RESP3 would suggest. A client reading a stream entry gets
/// `[name, value, name, value]` from a real 8.10 whether or not it asked for
/// RESP3, so sending a map here would break every consumer that upgraded.
fn entry(out: &mut Out, id: Id, fields: Fields<'_>) {
    out.array(2);
    id_out(out, id);
    out.array(fields.len() * 2);
    for (name, value) in fields {
        element(out, name);
        element(out, value);
    }
}

/// One field name or value as the client sees it.
///
/// A field stored as an integer has no digits anywhere until this line, because
/// a listpack holds the number and not its text. That is the same argument a
/// list element and a set member get.
#[inline]
fn element(out: &mut Out, e: Entry<'_>) {
    match e {
        Entry::Int(n) => out.bulk_int(n),
        Entry::Str(s) => out.bulk(s),
    }
}

/// One ID as the `ms-seq` bulk string every stream reply carries it as.
///
/// Formatted into a stack buffer rather than through [`Id::to_vec`], because
/// every reply in this file writes at least one of these and an `XRANGE` over a
/// million entries writes a million.
fn id_out(out: &mut Out, id: Id) {
    let mut buf = [0u8; DIGITS_MAX * 2 + 1];
    let mut digits = [0u8; DIGITS_MAX];
    let ms = u64_digits(&mut digits, id.ms);
    let mut n = ms.len();
    buf[..n].copy_from_slice(ms);
    buf[n] = b'-';
    n += 1;
    let mut digits = [0u8; DIGITS_MAX];
    let seq = u64_digits(&mut digits, id.seq);
    buf[n..n + seq.len()].copy_from_slice(seq);
    out.bulk(&buf[..n + seq.len()]);
}

/// A `NOGROUP` line, which is the third error prefix this group needs and is not
/// one [`super::write_error`] knows how to write.
fn nogroup(out: &mut Out, msg: &str) {
    out.error_line(b"NOGROUP ", msg.as_bytes());
}

/// An error with a message this file owns.
fn bad(msg: &'static str) -> Error {
    Error::new(Code::Invalid, msg)
}

/// A number that may not be negative, with the message the option carries.
fn non_negative(n: i64, msg: &'static str) -> Result<u64> {
    if n < 0 {
        return Err(Error::new(Code::Invalid, msg));
    }
    Ok(n.unsigned_abs())
}

/// A millisecond count from an argument that Redis reads as signed and then
/// clamps, so a negative one means none rather than an error.
fn millis(n: i64) -> u64 {
    if n < 0 { 0 } else { n.unsigned_abs() }
}

/// The arity of one subcommand, which Redis holds in its command table with the
/// container's name in front of it.
fn arity(n: usize, want: i32, sub: &'static str) -> Result<()> {
    let n = n as i32;
    let ok = if want < 0 { n >= -want } else { n == want };
    if ok {
        return Ok(());
    }
    Err(args::wrong_arity_sub("xgroup", sub))
}

/// `ERR unknown subcommand or wrong number of arguments for 'x'. Try Y HELP.`
///
/// The other half of the pair above. A subcommand that had enough arguments and
/// then did not fit the shape gets this rather than an arity error, because
/// Redis's table can only count them and the handler is what knows the shape.
fn unknown_or_arity(sub: &[u8], container: &str) -> Error {
    yo_alloc::allow(|| {
        Error::fmt(
            Code::Unsupported,
            format_args!(
                "unknown subcommand or wrong number of arguments for '{}'. Try {container} HELP.",
                String::from_utf8_lossy(sub)
            ),
        )
    })
}

/// What `XGROUP HELP` says, which is Redis's text and not ours.
const GROUP_HELP: &[&str] = &[
    "XGROUP <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "CREATE <key> <groupname> <id|$> [option]",
    "    Create a new consumer group. Options are:",
    "    * MKSTREAM",
    "      Create the empty stream if it does not exist.",
    "    * ENTRIESREAD entries_read",
    "      Set the group's entries_read counter (internal use).",
    "CREATECONSUMER <key> <groupname> <consumer>",
    "    Create a new consumer in the specified group.",
    "DELCONSUMER <key> <groupname> <consumer>",
    "    Remove the specified consumer.",
    "DESTROY <key> <groupname>",
    "    Remove the specified group.",
    "SETID <key> <groupname> <id|$> [ENTRIESREAD entries_read]",
    "    Set the current group ID and entries_read counter.",
    "HELP",
    "    Print this help.",
];

/// And what `XINFO HELP` says, missing bracket included, because it is missing
/// on a real server too.
const INFO_HELP: &[&str] = &[
    "XINFO <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "CONSUMERS <key> <groupname>",
    "    Show consumers of <groupname>.",
    "GROUPS <key>",
    "    Show the stream consumer groups.",
    "STREAM <key> [FULL [COUNT <count>]",
    "    Show information about the stream.",
    "HELP",
    "    Print this help.",
];
