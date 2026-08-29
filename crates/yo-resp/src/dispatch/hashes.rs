//! The hash commands, on the wire.
//!
//! The same shape as [`super::sets`]: the name has been looked up and the arity
//! has been checked, so this turns arguments into a call on [`Keyspace`] and the
//! answer into a reply. No decisions about hashes are made here and none about
//! representations, because the wire and the embedded API have to reach the same
//! code or there are two implementations of `HSET` and one of them is wrong
//! (Y23).
//!
//! # No allocation on the way in
//!
//! `HSET key f1 v1 f2 v2` hands the store an iterator over pairs of [`Args`]
//! rather than collecting them, and `HMGET` and `HDEL` do the same with their
//! field lists. Nothing on this path allocates whatever the argument count is.
//!
//! # The map reply
//!
//! `HGETALL` answers RESP3's map type, `%`, and not a flat array, and clients
//! act on it: a RESP3 client that gets a `%` hands back a dictionary without
//! being told what command it sent, where on RESP2 it has to know that the
//! reply is alternating and pair it up itself. [`Out::map`] is the one place
//! that knows the difference, and it writes twice the count as an array header
//! on RESP2 so the flat form comes out right.
//!
//! `HRANDFIELD key count WITHVALUES` is the other one, and it is not a map even
//! on RESP3, because a negative count can hand back the same field twice and a
//! map would silently lose one of them. Redis draws the line in the same place.

use yo_common::{Code, Error, Result, glob_matches, parse_i64};
use yo_kv::hash::Text;
use yo_kv::{Cursor, Keyspace};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What Redis says when a scan cursor is not a number.
const BAD_CURSOR: &str = "invalid cursor";
/// What Redis says when `HINCRBY` is given something that is not an integer.
const NOT_AN_INT: &str = "value is not an integer or out of range";
/// And `HINCRBYFLOAT`.
const NOT_A_FLOAT: &str = "value is not a valid float";
/// What `HSCAN` walks when the client does not say.
const SCAN_COUNT: usize = 10;

/// Run one hash command.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        // HSET and HMSET are the same write and differ only in the reply, which
        // is why HMSET has been deprecated since 4.0 and still has to work.
        "hset" | "hmset" => {
            if args.len() < 4 || !args.len().is_multiple_of(2) {
                return Err(args::wrong_arity(spec.name));
            }
            let added = db.hset(args.get(1), pairs(args))?;
            if spec.name == "hset" {
                out.int(count(added));
            } else {
                out.ok();
            }
        }
        "hsetnx" => out.int(i64::from(db.hsetnx(
            args.get(1),
            args.get(2),
            args.get(3),
        )?)),
        "hget" => db.hget(args.get(1), args.get(2), |t| match t {
            Some(t) => write_text(out, t),
            None => out.nil(),
        })?,
        "hdel" => out.int(count(db.hdel(args.get(1), fields(args, 2))?)),
        "hlen" => out.int(count(db.hlen(args.get(1))?)),
        "hexists" => out.int(i64::from(db.hexists(args.get(1), args.get(2))?)),
        "hstrlen" => out.int(count(db.hstrlen(args.get(1), args.get(2))?)),
        "hmget" => {
            out.array(args.len() - 2);
            db.hmget(args.get(1), fields(args, 2), |t| match t {
                Some(t) => write_text(out, t),
                None => out.nil(),
            })?;
        }
        // The three walks. They go through `with_hash` rather than through
        // `Keyspace::hgetall`, because every one of them needs the count for its
        // header before it needs the pairs, and asking `HLEN` for it would be a
        // second key lookup on the commands most likely to be in a loop.
        //
        // The header goes out inside the callback and not in front of the call,
        // because `with_hash` is where WRONGTYPE is decided and nothing should
        // be written before that is known.
        "hgetall" => db.with_hash(args.get(1), |hash| match hash {
            Some(h) => {
                out.map(h.len());
                for (field, value) in h.iter() {
                    write_text(out, field);
                    write_text(out, value);
                }
            }
            // A key that is not there is the empty hash and not a nil, which is
            // Redis's answer and what makes iterating the reply safe with no
            // check in front of it.
            None => out.map(0),
        })?,
        "hkeys" | "hvals" => {
            let want_keys = spec.name == "hkeys";
            db.with_hash(args.get(1), |hash| match hash {
                Some(h) => {
                    out.array(h.len());
                    for (field, value) in h.iter() {
                        write_text(out, if want_keys { field } else { value });
                    }
                }
                None => out.array(0),
            })?;
        }
        "hincrby" => out.int(db.hincrby(args.get(1), args.get(2), incr_int(args.get(3))?)?),
        // HINCRBYFLOAT answers a bulk string and not a double, on RESP3 as well
        // as RESP2. Redis never changed it, because the exact digits are the
        // point: a client that got a double back would have to trust its own
        // formatting to round trip, and this way the server's formatting is
        // what the client sees and what the next read returns.
        "hincrbyfloat" => {
            let by = incr_float(args.get(3))?;
            out.bulk_double(db.hincrbyfloat(args.get(1), args.get(2), by)?);
        }
        "hrandfield" => randfield(db, args, out)?,
        "hscan" => scan(db, args, out)?,
        other => unreachable!("{other} is not a hash command"),
    }
    Ok(())
}

/// `HRANDFIELD key [count [WITHVALUES]]`.
///
/// Three shapes. Without a count it is one field or a nil. With one it is an
/// array of fields, and with `WITHVALUES` on RESP3 it is an array of two element
/// arrays while on RESP2 it is one flat list. That split is Redis's and it is
/// the only place in the hash group where the two protocols disagree about the
/// shape rather than about the type.
fn randfield(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    match args.len() {
        2 => db.hrandfield(args.get(1), |pair| match pair {
            Some((field, _)) => write_text(out, field),
            None => out.nil(),
        })?,
        3 | 4 => {
            let with_values = match args.len() {
                4 if args::is(args.get(3), b"withvalues") => true,
                4 => return Err(args::syntax()),
                _ => false,
            };
            let n = args.int(2)?;
            // WITHVALUES nests each pair on RESP3 and flattens it on RESP2,
            // which is the one place in this group where the two protocols
            // disagree about the shape rather than about the type. It is not a
            // map on either, because a negative count can hand the same field
            // back twice and a map would silently lose one of them.
            let nested = with_values && out.proto().is_resp3();
            // The reply is written before its own header because a positive
            // count is capped at the size of the hash and the walk is what
            // finds out how many that is.
            let start = out.len();
            let mut written = 0;
            db.hrandfield_n(args.get(1), n, |field, value| {
                if nested {
                    out.array(2);
                }
                write_text(out, field);
                written += 1;
                if with_values {
                    write_text(out, value);
                    if !nested {
                        written += 1;
                    }
                }
            })?;
            out.close_array(start, written);
        }
        _ => return Err(args::syntax()),
    }
    Ok(())
}

/// `HSCAN key cursor [MATCH pattern] [COUNT count] [NOVALUES]`.
///
/// The reply is a cursor and then the pairs, and the pairs are written before
/// their own header because `MATCH` decides how many there are as it goes.
///
/// `NOVALUES` is Redis 7.4 and it is not a shortcut for `HKEYS`: it still walks
/// in windows, so it is the only way to page through the fields of a large hash
/// without also pulling every value across the wire.
///
/// `MATCH` matches the field and never the value, which is what a client would
/// assume and worth stating because the same walk has both in hand.
fn scan(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let cursor = parse_cursor(args.get(2))?;
    let mut pattern = None;
    let mut count = SCAN_COUNT;
    let mut novalues = false;
    let mut i = 3;
    while i < args.len() {
        let rest = args.len() - i;
        if args::is(args.get(i), b"match") && rest >= 2 {
            pattern = Some(args.get(i + 1));
        } else if args::is(args.get(i), b"count") && rest >= 2 {
            // A count under one is a syntax error and not a range error, which
            // is the odder of the two answers and so the one worth copying
            // exactly.
            count = match args.int(i + 1)? {
                n if n >= 1 => usize::try_from(n).unwrap_or(usize::MAX),
                _ => return Err(args::syntax()),
            };
        } else if args::is(args.get(i), b"novalues") {
            novalues = true;
            i += 1;
            continue;
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }

    // Nothing goes out until every argument has been checked, which is what
    // lets the dispatcher roll a failed command back cleanly.
    out.array(2);
    let at = out.len();
    let mut n = 0;
    let next = db.hscan(args.get(1), cursor, count, |field, value| {
        if !matches(pattern, field) {
            return;
        }
        write_text(out, field);
        n += 1;
        if !novalues {
            write_text(out, value);
            n += 1;
        }
    })?;
    out.close_array(at, n);
    // And the cursor goes in front of the pairs the same way their header went
    // in front of them, because the walk is what produced it.
    let body = out.len() - at;
    out.bulk_u64(next.raw());
    let cursor = out.len() - at - body;
    out.hoist(at, cursor);
    Ok(())
}

/// The field and value pairs of an `HSET`, without collecting them.
///
/// Argument one is the key, so the first field is two and the first value is
/// three. The walk steps on the value index and reads the field behind it,
/// which is what keeps the two from drifting apart when the arity check moves.
#[inline]
fn pairs(args: Args<'_>) -> impl Iterator<Item = (&[u8], &[u8])> + Clone {
    (3..args.len())
        .step_by(2)
        .map(move |i| (args.get(i - 1), args.get(i)))
}

/// The field list of an `HDEL` or an `HMGET`, without collecting it.
#[inline]
fn fields(args: Args<'_>, from: usize) -> impl Iterator<Item = &[u8]> {
    (from..args.len()).map(move |i| args.get(i))
}

/// A field or a value, written where it lies.
///
/// An integer held as one is formatted straight into the reply buffer and never
/// stored as digits anywhere, which is Y18 and is what keeps a hash of counters
/// at the size it is.
#[inline]
fn write_text(out: &mut Out, t: Text<'_>) {
    match t {
        Text::Int(n) => out.bulk_int(n),
        Text::Str(s) => out.bulk(s),
    }
}

/// An `HINCRBY` amount.
///
/// The message is the string command's and not the hash's. Redis uses the value
/// sentence for a bad argument and the hash sentence only for a field that
/// already holds the wrong thing, which reads backwards until you notice that
/// the argument is not a hash value yet.
fn incr_int(arg: &[u8]) -> Result<i64> {
    parse_i64(arg).ok_or_else(|| Error::new(Code::Invalid, NOT_AN_INT))
}

/// An `HINCRBYFLOAT` amount, on the same rule.
fn incr_float(arg: &[u8]) -> Result<f64> {
    yo_common::num::parse_f64(arg).ok_or_else(|| Error::new(Code::Invalid, NOT_A_FLOAT))
}

/// A cursor as the client sent it back.
fn parse_cursor(arg: &[u8]) -> Result<Cursor> {
    match std::str::from_utf8(arg).ok().and_then(|s| s.parse().ok()) {
        Some(raw) => Ok(Cursor::from_raw(raw)),
        None => Err(Error::new(Code::Invalid, BAD_CURSOR)),
    }
}

/// Whether a field passes the `MATCH` pattern, if there is one.
#[inline]
fn matches(pattern: Option<&[u8]>, field: Text<'_>) -> bool {
    let Some(p) = pattern else {
        return true;
    };
    match field {
        Text::Str(s) => glob_matches(p, s),
        // A field held as an integer has to be written out to be matched, which
        // is the one place a number becomes digits before the reply. It is on
        // the MATCH path only, and only for the numeric fields, so the common
        // HSCAN pays nothing for it.
        Text::Int(n) => {
            let mut buf = [0u8; yo_common::num::DIGITS_MAX];
            glob_matches(p, yo_common::num::i64_digits(&mut buf, n))
        }
    }
}

/// A count as the integer the reply carries.
#[inline]
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}
