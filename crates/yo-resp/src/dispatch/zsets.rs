//! The sorted set commands, on the wire.
//!
//! The same shape as [`super::sets`]: the name has been looked up and the arity
//! has been checked, so this turns arguments into a call on [`Keyspace`] and the
//! answer into a reply. No decision about sorted sets is made here, because the
//! wire and the embedded API have to reach the same code or there are two
//! implementations of `ZADD` and one of them is wrong (Y23).
//!
//! # A score is a double and the reply type says so
//!
//! On RESP3 every score goes out as `,`, the double type, and on RESP2 as a bulk
//! string of the same digits. That is Redis's split and [`Out::double`] is the
//! one place that knows about it, so nothing here has to ask which protocol the
//! connection is on.
//!
//! The infinities are words in both, `inf` and `-inf`, which is what makes
//! `ZADD key inf m` followed by `ZSCORE key m` round trip. A NaN never reaches a
//! reply, because every command that takes a score refuses one on the way in and
//! the two ways to make one out of legal scores are refused by the store.
//!
//! # Where the arguments are parsed
//!
//! Three kinds of number arrive on this path and they have three different
//! errors, which is not tidiness, it is the contract. A score that will not
//! parse is `value is not a valid float`, a score range bound is `min or max is
//! not a float`, and a lexical bound is `min or max not valid string range
//! item`. A client that matches on those is doing something ugly against every
//! Redis in the world, so they are copied exactly.

use yo_common::{Error, Result};
use yo_kv::{Keyspace, Query, ZAdd, ZBound};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What Redis says when a score range bound will not parse.
const BAD_RANGE: &str = "min or max is not a float";
/// And when a lexical one will not.
const BAD_LEX: &str = "min or max not valid string range item";
/// `ZADD NX` with `XX`.
const NX_AND_XX: &str = "XX and NX options at the same time are not compatible";
/// `ZADD NX` with either of the two that compare against a score that is
/// already there, and `GT` with `LT`. Redis says all three in one sentence.
const NX_AND_GT_LT: &str = "GT, LT, and/or NX options at the same time are not compatible";
/// `ZADD INCR` with more than one pair.
const ONE_PAIR: &str = "INCR option supports a single increment-element pair";

/// Run one sorted set command.
///
/// # Errors
///
/// A key holding something that is not a sorted set, an option where one was
/// not expected, and the three number errors above.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "zadd" => zadd(db, args, out)?,
        "zincrby" => {
            let by = score(args.get(2))?;
            // Never nil, because `ZINCRBY` has no gate to refuse it, so the
            // `None` this cannot produce would be a bug rather than an answer.
            let now = db
                .zincrby(args.get(1), args.get(3), by, ZAdd::default())?
                .expect("ZINCRBY has no gate that can refuse a member");
            out.double(now);
        }
        "zcard" => out.int(count(db.zcard(args.get(1))?)),
        "zscore" => match db.zscore(args.get(1), args.get(2))? {
            Some(s) => out.double(s),
            None => out.nil(),
        },
        "zmscore" => {
            out.array(args.len() - 2);
            // One walk over the arguments and one key lookup, because the memo
            // in the keyspace answers the second and later calls without going
            // back to the map. A `Vec` of scores in between would be an
            // allocation per call on a thread that must not allocate.
            for i in 2..args.len() {
                match db.zscore(args.get(1), args.get(i))? {
                    Some(s) => out.double(s),
                    None => out.nil(),
                }
            }
        }
        "zrem" => out.int(count(db.zrem(args.get(1), members(args))?)),
        "zrank" | "zrevrank" => rank(db, spec.name == "zrevrank", args, out)?,
        "zcount" => {
            let q = Query::score(bound(args.get(2))?, bound(args.get(3))?);
            out.int(count(db.zcount(args.get(1), &q)?));
        }
        "zlexcount" => {
            let q = Query::lex(lex(args.get(2))?, lex(args.get(3))?);
            out.int(count(db.zcount(args.get(1), &q)?));
        }
        // The table and this match are checked against each other by
        // `cargo xtask check`, so a name reaching here is a table row without a
        // handler and there is nothing sensible to answer.
        other => unreachable!("{other} is not a sorted set command"),
    }
    Ok(())
}

/// `ZADD key [NX|XX] [GT|LT] [CH] [INCR] score member [score member ...]`.
///
/// The options come off the front until something that is not one of them
/// appears, and what is left has to be an even number of arguments. That means
/// `ZADD key 1 a 2` is a syntax error and not a complaint about `2` not being a
/// member, which is Redis's answer and falls out of counting rather than being
/// a rule of its own.
///
/// The scores are parsed before anything is written, all of them, so
/// `ZADD key 1 a nonsense b` adds nothing at all. Redis does the same and it is
/// the only behaviour that makes the command safe to retry.
fn zadd(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (mut nx, mut xx, mut gt, mut lt, mut incr) = (false, false, false, false, false);
    let mut opts = ZAdd::default();
    let mut at = 2;
    while at < args.len() {
        let arg = args.get(at);
        if args::is(arg, b"nx") {
            nx = true;
        } else if args::is(arg, b"xx") {
            xx = true;
        } else if args::is(arg, b"gt") {
            gt = true;
        } else if args::is(arg, b"lt") {
            lt = true;
        } else if args::is(arg, b"ch") {
            opts.changed = true;
        } else if args::is(arg, b"incr") {
            incr = true;
        } else {
            break;
        }
        at += 1;
    }
    // Counting comes before the incompatible options, which is the one bit of
    // this order that is not obvious: `ZADD key NX XX` is a syntax error about
    // having no pairs and not a complaint about NX and XX, because Redis counts
    // what is left before it looks at what it collected.
    let left = args.len() - at;
    if left == 0 || !left.is_multiple_of(2) {
        return Err(args::syntax());
    }
    if nx && xx {
        return Err(Error::new(yo_common::Code::Invalid, NX_AND_XX));
    }
    // One sentence for all three, which is Redis's and not a shortening of it.
    if (nx && (gt || lt)) || (gt && lt) {
        return Err(Error::new(yo_common::Code::Invalid, NX_AND_GT_LT));
    }
    if incr && left != 2 {
        return Err(Error::new(yo_common::Code::Invalid, ONE_PAIR));
    }
    opts.gate = if nx {
        yo_kv::Gate::IfMissing
    } else if xx {
        yo_kv::Gate::IfPresent
    } else {
        yo_kv::Gate::Always
    };
    opts.only = if gt {
        yo_kv::Move::Up
    } else if lt {
        yo_kv::Move::Down
    } else {
        yo_kv::Move::Any
    };
    // Every score, before the first one is stored.
    for i in (at..args.len()).step_by(2) {
        score(args.get(i))?;
    }
    if incr {
        return match db.zincrby(args.get(1), args.get(at + 1), score(args.get(at))?, opts)? {
            Some(now) => {
                out.double(now);
                Ok(())
            }
            // A gate refused it. Redis answers the string nil here and not the
            // array one, because the reply this is standing in for is a score.
            None => {
                out.nil();
                Ok(())
            }
        };
    }
    let pairs = (at..args.len())
        .step_by(2)
        .map(|i| (score(args.get(i)).unwrap_or(0.0), args.get(i + 1)));
    out.int(count(db.zadd(args.get(1), pairs, opts)?));
    Ok(())
}

/// `ZRANK key member [WITHSCORE]` and `ZREVRANK`.
///
/// `WITHSCORE` changes the shape of the answer both ways: the rank becomes a two
/// element array of the rank and the score, and a member that is not there
/// becomes the array nil rather than the string one. A RESP2 client tells those
/// two apart, so sending the wrong one is a client that hangs on a reply it
/// cannot parse.
fn rank(db: &mut Keyspace, rev: bool, args: Args<'_>, out: &mut Out) -> Result<()> {
    // An argument too many is the arity error and not the syntax one, which is
    // Redis's split and looks backwards until you see that the option is
    // optional: `ZRANK k m junk` is a bad option, `ZRANK k m WITHSCORE junk` is
    // one argument too many.
    let withscore = match args.len() {
        3 => false,
        4 if args::is(args.get(3), b"withscore") => true,
        4 => return Err(args::syntax()),
        _ => return Err(args::wrong_arity(if rev { "zrevrank" } else { "zrank" })),
    };
    match db.zrank(args.get(1), args.get(2), rev)? {
        Some((at, s)) => {
            if withscore {
                out.array(2);
            }
            out.int(count(at));
            if withscore {
                out.double(s);
            }
        }
        None if withscore => out.nil_array(),
        None => out.nil(),
    }
    Ok(())
}

/// A score, which is the plainest of the three numbers here.
fn score(arg: &[u8]) -> Result<f64> {
    yo_common::num::parse_f64(arg)
        .ok_or_else(|| Error::new(yo_common::Code::Invalid, args::NOT_A_FLOAT))
}

/// One end of a score range: `1`, `(1` for exclusive, and the two infinities.
///
/// The bracket is the whole of the syntax, so `(` on its own is a bound with no
/// number behind it and is refused by the parse rather than by a length check.
fn bound(arg: &[u8]) -> Result<ZBound> {
    let (open, digits) = match arg.split_first() {
        Some((b'(', rest)) => (true, rest),
        _ => (false, arg),
    };
    let Some(at) = yo_common::num::parse_f64(digits) else {
        return Err(Error::new(yo_common::Code::Invalid, BAD_RANGE));
    };
    Ok(if open {
        ZBound::open(at)
    } else {
        ZBound::closed(at)
    })
}

/// One end of a lexical range: `-`, `+`, `[member` or `(member`.
///
/// A bare member is not a bound. `ZRANGEBYLEX key a b` is an error and not a
/// range from `a` to `b`, because a member could start with any byte and there
/// would be no way to say `[` if the brackets were optional.
fn lex(arg: &[u8]) -> Result<yo_kv::Lex<'_>> {
    match arg.split_first() {
        Some((b'-', b"")) => Ok(yo_kv::Lex::Min),
        Some((b'+', b"")) => Ok(yo_kv::Lex::Max),
        Some((b'[', rest)) => Ok(yo_kv::Lex::Incl(rest)),
        Some((b'(', rest)) => Ok(yo_kv::Lex::Excl(rest)),
        _ => Err(Error::new(yo_common::Code::Invalid, BAD_LEX)),
    }
}

/// The members of a command that takes a list of them, as an iterator.
///
/// An iterator and not a `Vec`, so `ZREM key a b c` allocates nothing at all,
/// which is why [`Keyspace::zrem`] takes one.
fn members(args: Args<'_>) -> impl Iterator<Item = &[u8]> + Clone {
    (2..args.len()).map(move |i| args.get(i))
}

/// A count as the wire wants it.
///
/// Every count in this file comes from a collection that cannot hold more than
/// twenty four million elements, so the conversion cannot fail and saturating is
/// a shape rather than a decision.
#[inline]
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}
