//! The set commands, on the wire.
//!
//! The same shape as [`super::strings`]: the name has been looked up and the
//! arity has been checked, so this turns arguments into a call on
//! [`Keyspace`] and the answer into a reply. No decisions about sets are made
//! here, and none about representations, because the wire and the embedded API
//! have to reach the same code or there are two implementations of `SADD` and
//! one of them is wrong (Y23).
//!
//! # No allocation on the way in
//!
//! Every command that takes a list of members hands the store an iterator over
//! [`Args`] rather than collecting them into a `Vec` first. `SADD key a b c` on
//! this thread allocates nothing at all, which is the point of Y1 and the
//! reason [`Keyspace::sadd`] takes an iterator in the first place.
//!
//! # The set reply type
//!
//! `SMEMBERS` and the algebra commands answer RESP3's set type, `~`, and not an
//! array. Redis does this and clients act on it: a RESP3 client that gets a `~`
//! builds a set rather than a list, so a Python client hands back `set` instead
//! of `list` without being told what command it sent. On RESP2 it is an array,
//! because RESP2 has no set type, and [`Out::set`] is the one place that knows
//! the difference.

use yo_common::num::{DIGITS_MAX, i64_digits};
use yo_common::{Code, Error, Result, glob_matches, parse_i64};
use yo_kv::{Cursor, Keyspace, Member};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What Redis says when `SPOP` is given a count it will not take.
///
/// The same sentence for a negative number and for something that is not a
/// number at all, which looks like a mistake and is what `getRangeLongFromObject`
/// does when a command hands it a message to use.
const BAD_POP_COUNT: &str = "value is out of range, must be positive";
/// What Redis says when a scan cursor is not a number.
const BAD_CURSOR: &str = "invalid cursor";
/// What `SSCAN` walks when the client does not say.
const SCAN_COUNT: usize = 10;

/// Run one set command.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "sadd" => out.int(count(db.sadd(args.get(1), members(args))?)),
        "srem" => out.int(count(db.srem(args.get(1), members(args))?)),
        "scard" => out.int(count(db.scard(args.get(1))?)),
        "sismember" => out.int(i64::from(db.sismember(args.get(1), args.get(2))?)),
        // The two that want the body more than once go through `with_set`, not
        // through `Keyspace::smismember` and `Keyspace::smembers`. Those two
        // answer a `Vec` and an iterator, which is the right shape for an
        // embedded caller who wants the answer in one piece, and the wrong shape
        // here: the reply is written a member at a time straight into the
        // connection's out buffer, so a `Vec` in between would be an allocation
        // per call on a thread that must not allocate.
        // The header goes out inside the callback and not in front of the call,
        // because `with_set` is where WRONGTYPE is decided and a body checks its
        // arguments before it writes anything.
        "smismember" => db.with_set(args.get(1), |set| {
            out.array(args.len() - 2);
            for m in members(args) {
                out.int(i64::from(set.is_some_and(|s| s.contains(m))));
            }
        })?,
        "smembers" => db.with_set(args.get(1), |set| match set {
            Some(s) => {
                out.set(s.len());
                for m in s.iter() {
                    write_member(out, m);
                }
            }
            // A key that is not there is the empty set and not a nil, which is
            // Redis's answer and is what makes iterating the reply safe to write
            // without a check in front of it.
            None => out.set(0),
        })?,
        // The two draws, and the pair of them is the one place where a reply
        // type carries information the command name does not. `SPOP key 3`
        // answers a set, because the three members are distinct and a RESP3
        // client can safely build one out of them. `SRANDMEMBER key -3` answers
        // an array, because it can hand back the same member three times and a
        // set would silently lose two of them. Redis draws the line in the same
        // place and this is why.
        "spop" => match args.len() {
            2 => match db.spop(args.get(1))? {
                Some(m) => out.bulk(&m),
                None => out.nil(),
            },
            3 => {
                let got = db.spop_n(args.get(1), pop_count(args.get(2))?)?;
                out.set(got.len());
                for m in &got {
                    out.bulk(m);
                }
            }
            _ => return Err(args::syntax()),
        },
        "srandmember" => match args.len() {
            2 => db.srandmember(args.get(1), |m| match m {
                Some(m) => write_member(out, m),
                None => out.nil(),
            })?,
            3 => {
                let count = args.int(2)?;
                let start = out.len();
                let mut n = 0;
                db.srandmember_n(args.get(1), count, |m| {
                    write_member(out, m);
                    n += 1;
                })?;
                out.close_array(start, n);
            }
            _ => return Err(args::syntax()),
        },
        "smove" => out.int(i64::from(db.smove(
            args.get(1),
            args.get(2),
            args.get(3),
        )?)),
        "sscan" => scan(db, args, out)?,
        other => unreachable!("{other} is not a set command"),
    }
    Ok(())
}

/// `SSCAN key cursor [MATCH pattern] [COUNT count]`.
///
/// The reply is a cursor and then the members, and the members are written
/// before their own header because `MATCH` decides how many there are as it
/// goes. See [`Out::close_array`], which is the whole of that.
///
/// The cursor goes out as a bulk string of unsigned digits rather than through
/// the integer path, because ours packs a partition count into the top bits and
/// a large enough collection would make it wider than an `i64`.
fn scan(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let cursor = parse_cursor(args.get(2))?;
    let mut pattern = None;
    let mut count = SCAN_COUNT;
    let mut i = 3;
    while i < args.len() {
        let rest = args.len() - i;
        if args::is(args.get(i), b"match") && rest >= 2 {
            pattern = Some(args.get(i + 1));
        } else if args::is(args.get(i), b"count") && rest >= 2 {
            // Redis reads a count under one as a syntax error and not as a
            // range error, which is worth copying exactly because it is the
            // odder of the two answers.
            count = match args.int(i + 1)? {
                n if n >= 1 => usize::try_from(n).unwrap_or(usize::MAX),
                _ => return Err(args::syntax()),
            };
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }

    // The reply is built into a scratch position and the cursor is written
    // first, so the header for the members cannot go out until the walk is
    // over. Nothing is written before the arguments have all been checked,
    // which is what lets the dispatcher roll a failed command back cleanly.
    out.array(2);
    let at = out.len();
    let mut n = 0;
    let next = db.sscan(args.get(1), cursor, count, |m| {
        if matches(pattern, m) {
            write_member(out, m);
            n += 1;
        }
    })?;
    out.close_array(at, n);
    // And the cursor goes in front of the members the same way their header
    // went in front of them, because the walk is what produced it.
    let members = out.len() - at;
    out.bulk_u64(next.raw());
    let cursor = out.len() - at - members;
    out.hoist(at, cursor);
    Ok(())
}

/// A `SPOP` count, which cannot be negative and cannot be something else.
fn pop_count(arg: &[u8]) -> Result<usize> {
    match parse_i64(arg) {
        Some(n) if n >= 0 => Ok(usize::try_from(n).unwrap_or(usize::MAX)),
        _ => Err(Error::new(Code::Invalid, BAD_POP_COUNT)),
    }
}

/// A cursor as the client sent it back.
///
/// Unsigned, because ours uses the top bits, and Redis parses a cursor with
/// `strtoull` too. Anything else is `invalid cursor`, which is its own sentence
/// and not the usual one about integers.
fn parse_cursor(arg: &[u8]) -> Result<Cursor> {
    if arg.is_empty() || !arg.iter().all(u8::is_ascii_digit) {
        return Err(Error::new(Code::Invalid, BAD_CURSOR));
    }
    let mut raw: u64 = 0;
    for b in arg {
        raw = raw
            .checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(b - b'0')))
            .ok_or_else(|| Error::new(Code::Invalid, BAD_CURSOR))?;
    }
    Ok(Cursor::from_raw(raw))
}

/// Whether a member survives a `MATCH` pattern.
///
/// A member stored as an integer has no digits anywhere, so this is the one
/// place a scan pays to write some, into twenty one bytes of stack rather than
/// into a `Vec` that would be an allocation per member.
#[inline]
fn matches(pattern: Option<&[u8]>, m: Member<'_>) -> bool {
    let Some(pattern) = pattern else {
        return true;
    };
    match m {
        Member::Str(s) => glob_matches(pattern, s),
        Member::Int(n) => {
            let mut buf = [0u8; DIGITS_MAX];
            glob_matches(pattern, i64_digits(&mut buf, n))
        }
    }
}

/// Every argument after the key, which for these commands is every member.
#[inline]
fn members(args: Args<'_>) -> impl Iterator<Item = &[u8]> + Clone {
    (2..args.len()).map(move |i| args.get(i))
}

/// One member as the client sees it.
///
/// An integer member has no digits anywhere until this line, because an intset
/// holds the number and not its text. Formatting it here rather than on the way
/// in is Y18: a set of a thousand integers is two kilobytes in memory and only
/// becomes reply text for the members somebody actually asked for.
#[inline]
fn write_member(out: &mut Out, m: Member<'_>) {
    match m {
        Member::Int(n) => out.bulk_int(n),
        Member::Str(s) => out.bulk(s),
    }
}

/// A count as the integer the reply carries.
///
/// Saturating rather than wrapping, for the reason [`super::strings`] gives:
/// nothing counted here can reach `i64::MAX`, and a count that came back wrong
/// is better reported as an implausible number than as a negative one.
#[inline]
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}
