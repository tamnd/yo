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
//!
//! # Nine range commands and one range
//!
//! `ZRANGE` has three by-modes and `REV` doubles them, `ZREVRANGE`,
//! `ZREVRANGEBYSCORE` and `ZREVRANGEBYLEX` are older spellings of three of the
//! six, `ZRANGESTORE` is a seventh spelling, and the three `ZREMRANGE` forms are
//! three more. Writing them apart is nine chances to get an exclusive bound or a
//! negative index wrong in exactly one of them.
//!
//! So [`parse_range`] turns any of them into one [`Query`], and what the command
//! does with the window that comes back is all that separates it from the
//! others: walk it, remove it, or walk it into another key. The older spellings
//! are the same parse with the by-mode fixed and the two bound arguments the
//! other way round, because `ZREVRANGEBYSCORE key max min` names its high end
//! first and `ZRANGE key min max BYSCORE REV` does not.
//!
//! `WITHSCORES` nests each pair on RESP3 and flattens it on RESP2, which is the
//! one place in this group where the two protocols disagree about the shape of
//! a reply rather than the type of one value in it.

use yo_common::{Error, Result};
use yo_kv::{Keyspace, Member, Query, ZAdd, ZBound};

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
/// `LIMIT` on a range that is by rank, where it means nothing.
const LIMIT_NEEDS_BY: &str =
    "syntax error, LIMIT is only supported in combination with either BYSCORE or BYLEX";
/// `WITHSCORES` on a lexical range, where Redis refuses it even though it could
/// answer, because every score in a lexical range is meant to be the same one.
const SCORES_NOT_BYLEX: &str = "syntax error, WITHSCORES not supported in combination with BYLEX";

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
        // The six that answer members, which are one parse and one walk.
        "zrange" | "zrevrange" | "zrangebyscore" | "zrevrangebyscore" | "zrangebylex"
        | "zrevrangebylex" => {
            let form = Form::of(spec.name);
            let (q, withscores) = parse_range(form, args, 1)?;
            let w = db.zwindow(args.get(1), &q)?;
            // The header before the members, because a window knows its own
            // length before anything is walked, which is the whole reason
            // `zwindow` and `zwalk` are two calls.
            let nested = withscores && out.proto().is_resp3();
            out.array(if withscores && !nested {
                w.count * 2
            } else {
                w.count
            });
            db.zwalk(args.get(1), w, |m, sc| {
                if nested {
                    out.array(2);
                }
                write_member(out, m);
                if withscores {
                    out.double(sc);
                }
            })?;
        }
        // The same parse, with the destination in front and no WITHSCORES.
        "zrangestore" => {
            let (q, _) = parse_range(Form::Store, args, 2)?;
            out.int(count(db.zrangestore(args.get(1), args.get(2), &q)?));
        }
        // And the same parse again with the walk turned into a removal. These
        // three have their by-mode in the name and take no options at all, so
        // the arity check has already done the whole of the syntax.
        "zremrangebyrank" => {
            let q = Query::rank(args.int(2)?, args.int(3)?);
            out.int(count(db.zremrange(args.get(1), &q)?));
        }
        "zremrangebyscore" => {
            let q = Query::score(bound(args.get(2))?, bound(args.get(3))?);
            out.int(count(db.zremrange(args.get(1), &q)?));
        }
        "zremrangebylex" => {
            let q = Query::lex(lex(args.get(2))?, lex(args.get(3))?);
            out.int(count(db.zremrange(args.get(1), &q)?));
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

/// Which of the seven spellings a range command is.
///
/// The by-mode and the direction are what the older names carry instead of
/// options, and `Swapped` is the third thing they carry: `ZREVRANGEBYSCORE key
/// max min` names its high end first, where `ZRANGE key min max BYSCORE REV`
/// names its low end first and reverses only the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// `ZRANGE`, which takes all of the options.
    Range,
    /// `ZRANGESTORE`, which takes all of them except `WITHSCORES`, since the
    /// destination is a sorted set and keeps the scores whatever is asked for.
    Store,
    /// `ZREVRANGE`, which is by rank and backwards.
    RevRange,
    /// `ZRANGEBYSCORE` and `ZREVRANGEBYSCORE`.
    ByScore { rev: bool },
    /// `ZRANGEBYLEX` and `ZREVRANGEBYLEX`.
    ByLex { rev: bool },
}

impl Form {
    /// Which one this name is.
    fn of(name: &str) -> Form {
        match name {
            "zrangestore" => Form::Store,
            "zrevrange" => Form::RevRange,
            "zrangebyscore" => Form::ByScore { rev: false },
            "zrevrangebyscore" => Form::ByScore { rev: true },
            "zrangebylex" => Form::ByLex { rev: false },
            "zrevrangebylex" => Form::ByLex { rev: true },
            _ => Form::Range,
        }
    }

    /// Whether `BYSCORE`, `BYLEX` and `REV` mean anything to this spelling.
    ///
    /// They do not to the older ones, which carry their mode in the name, so
    /// `ZREVRANGE key 0 -1 BYSCORE` is a syntax error rather than a way of
    /// saying `ZREVRANGEBYSCORE`.
    fn takes_mode(self) -> bool {
        matches!(self, Form::Range | Form::Store)
    }
}

/// Turn a range command's arguments into one [`Query`], and say whether the
/// client asked for the scores.
///
/// `key` is where the key is, which is one for every one of these except
/// `ZRANGESTORE`, whose source is the second argument. The two bound arguments
/// follow it and the options follow those.
fn parse_range<'a>(form: Form, args: Args<'a>, key: usize) -> Result<(Query<'a>, bool)> {
    let (mut lo, mut hi) = (key + 1, key + 2);
    let (mut byscore, mut bylex, mut rev) = (false, false, false);
    match form {
        Form::Range | Form::Store => {}
        Form::RevRange => rev = true,
        Form::ByScore { rev: r } => {
            byscore = true;
            rev = r;
        }
        Form::ByLex { rev: r } => {
            bylex = true;
            rev = r;
        }
    }

    let mut withscores = false;
    let mut limit: Option<(i64, i64)> = None;
    let mut at = key + 3;
    while at < args.len() {
        let arg = args.get(at);
        // `WITHSCORES` and `LIMIT` are read by every spelling and refused
        // afterwards if they do not go with the mode, which is why
        // `ZREVRANGE key 0 -1 WITHSCORES LIMIT 0 1` complains about LIMIT and
        // not about the word after it.
        if form != Form::Store && args::is(arg, b"withscores") {
            withscores = true;
        } else if args::is(arg, b"limit") && at + 2 < args.len() {
            limit = Some((args.int(at + 1)?, args.int(at + 2)?));
            at += 2;
        } else if form.takes_mode() && args::is(arg, b"byscore") {
            byscore = true;
        } else if form.takes_mode() && args::is(arg, b"bylex") {
            bylex = true;
        } else if form.takes_mode() && args::is(arg, b"rev") {
            rev = true;
        } else {
            return Err(args::syntax());
        }
        at += 1;
    }
    if byscore && bylex {
        return Err(args::syntax());
    }
    // The high end is named first whenever a reverse walk is over scores or
    // names, and it is not when the walk is over ranks, because a rank counts
    // from the end the walk starts at and a bound does not. That holds for the
    // older spellings and for `ZRANGE ... REV` alike, so the swap happens here,
    // once, after the options have said which mode this is.
    if rev && (byscore || bylex) {
        core::mem::swap(&mut lo, &mut hi);
    }
    if withscores && bylex {
        return Err(Error::new(yo_common::Code::Invalid, SCORES_NOT_BYLEX));
    }
    if limit.is_some() && !byscore && !bylex {
        return Err(Error::new(yo_common::Code::Invalid, LIMIT_NEEDS_BY));
    }

    let mut q = if byscore {
        Query::score(bound(args.get(lo))?, bound(args.get(hi))?)
    } else if bylex {
        Query::lex(lex(args.get(lo))?, lex(args.get(hi))?)
    } else {
        Query::rank(args.int(lo)?, args.int(hi)?)
    }
    .rev(rev);
    if let Some((offset, take)) = limit {
        // A negative offset skips more members than there could ever be, which
        // is the empty answer Redis gives. A negative count is no bound at all.
        q = q.limit(
            usize::try_from(offset).unwrap_or(usize::MAX),
            usize::try_from(take).ok(),
        );
    }
    Ok((q, withscores))
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

/// One member as the client sees it.
///
/// A member is stored as an integer when it looks like one, the same as a set
/// member and a list element, and goes back out as the digits it arrived as.
#[inline]
fn write_member(out: &mut Out, m: Member<'_>) {
    match m {
        Member::Int(n) => out.bulk_int(n),
        Member::Str(s) => out.bulk(s),
    }
}
