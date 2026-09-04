//! The bitmap commands, from the wire.
//!
//! Seven commands over the same string values [`strings`](super::strings)
//! writes, and the same two rules as that file: nothing is written to the reply
//! until the arguments are known to be good, and nothing allocates.
//!
//! The second rule is what shapes `BITOP` and `BITFIELD` here. `BITOP` hands its
//! source keys over as an iterator rather than as a list, and `BITFIELD` walks
//! its arguments twice, once to check every subcommand and once to run them, so
//! that a `BITFIELD` with two hundred subcommands never has two hundred of
//! anything in memory. The second walk is not a cost worth avoiding and it is
//! not a workaround either: Redis checks the whole argument list before it runs
//! any of it, so a bad field type in the last subcommand leaves the key
//! untouched and uncreated, and the two walks are how that is kept true.
//!
//! Every error sentence in this file was read off a running 8.10.1. Several of
//! them are close enough to each other to swap by accident: an offset that is
//! not a number is "bit offset is not an integer or out of range" while the
//! value in `SETBIT` is "bit is not an integer or out of range", and `BITPOS`
//! has a third one with a full stop on the end of it.

use super::args::{self, Args, is};
use super::table::Spec;
use crate::reply::Out;
use yo_common::num::parse_i64;
use yo_common::{Code, Error, Result};
use yo_kv::bitmaps::{Sub, SubOp, Unit};
use yo_kv::bits::{Field, Op, Overflow};
use yo_kv::{Db, bitmaps};

/// What Redis says when a bit offset is not a number or is off the end.
const BAD_OFFSET: &str = "bit offset is not an integer or out of range";
/// What Redis says when `SETBIT`'s value is not a nought or a one.
const BAD_BIT: &str = "bit is not an integer or out of range";
/// What Redis says when `BITPOS` is asked about a bit that is not there.
///
/// It has a full stop on the end where the other two do not.
const BAD_SEARCH_BIT: &str = "The bit argument must be 1 or 0.";
/// What Redis says about a `BITFIELD` type it does not recognise.
const BAD_TYPE: &str =
    "Invalid bitfield type. Use something like i16 u8. Note that u64 is not supported but i64 is.";
/// What Redis says about an `OVERFLOW` word it does not recognise.
const BAD_OVERFLOW: &str = "Invalid OVERFLOW type specified";
/// What Redis says when `BITFIELD_RO` is asked to write.
const RO_GET_ONLY: &str = "BITFIELD_RO only supports the GET subcommand";

/// Run one bitmap command.
pub(super) fn execute(db: &mut Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "setbit" => {
            let offset = offset(args.get(2))?;
            let bit = match parse_i64(args.get(3)) {
                Some(0) => false,
                Some(1) => true,
                _ => return Err(Error::new(Code::Invalid, BAD_BIT)),
            };
            let key = args.get(1);
            out.int(i64::from(db.at(key).setbit(key, offset, bit)?));
        }
        "getbit" => {
            let offset = offset(args.get(2))?;
            let key = args.get(1);
            out.int(i64::from(db.at(key).getbit(key, offset)?));
        }
        "bitcount" => {
            let range = range(args, 2)?;
            let key = args.get(1);
            let set = db.at(key).bitcount(key, range)?;
            out.int(i64::try_from(set).unwrap_or(i64::MAX));
        }
        "bitpos" => bitpos(db, args, out)?,
        "bitop" => bitop(db, args, out)?,
        "bitfield" => bitfield(db, args, out, false)?,
        "bitfield_ro" => bitfield(db, args, out, true)?,
        _ => return Err(args::syntax()),
    }
    Ok(())
}

/// `BITPOS key bit [start [end [BYTE | BIT]]]`.
fn bitpos(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    // The bit comes before the indexes, so a client that sends a word here gets
    // the "not an integer" sentence and one that sends a 2 gets the other one.
    let bit = match args.int(2)? {
        0 => false,
        1 => true,
        _ => return Err(Error::new(Code::Invalid, BAD_SEARCH_BIT)),
    };
    let (start, end, unit) = match args.len() {
        3 => (None, None, Unit::Byte),
        // A lone start is allowed here, where `BITCOUNT` refuses it, and it is
        // what turns the search into one that runs to the end of the string.
        4 => (Some(args.int(3)?), None, Unit::Byte),
        _ => {
            let (start, end, unit) = range(args, 3)?.ok_or_else(args::syntax)?;
            (Some(start), Some(end), unit)
        }
    };
    let key = args.get(1);
    out.int(db.at(key).bitpos(key, bit, start, end, unit)?);
    Ok(())
}

/// `BITOP op dest src [src ...]`.
fn bitop(db: &mut Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let op = Op::parse(args.get(1)).ok_or_else(args::syntax)?;
    let sources = args.len() - 3;
    if op == Op::Not && sources != 1 {
        return Err(bad(op, "must be called with a single source key."));
    }
    if sources < 2 && matches!(op, Op::Diff | Op::Diff1 | Op::AndOr) {
        return Err(bad(op, "must be called with at least two source keys."));
    }
    let srcs = (3..args.len()).map(|i| args.get(i));
    out.int(count(db.bitop(op, args.get(2), srcs)?));
    Ok(())
}

/// `BITOP AND must be called with ...`, which names the operation in the middle.
fn bad(op: Op, tail: &str) -> Error {
    Error::fmt(Code::Invalid, format_args!("BITOP {} {tail}", op.name()))
}

/// `BITFIELD key [subcommand ...]` and its read only twin.
///
/// Two walks over the same arguments. The first one parses every subcommand and
/// throws the result away, which is what makes a bad word in the last one leave
/// the key alone, and it works out how far the value has to grow on the way
/// past. The second one parses them again and runs them, writing each reply as
/// it goes.
fn bitfield(db: &mut Db, args: Args<'_>, out: &mut Out, readonly: bool) -> Result<()> {
    let mut grow: Option<usize> = None;
    let mut at = 2;
    let mut n = 0;
    let mut on = Overflow::Wrap;
    while at < args.len() {
        let (sub, next) = parse(args, at, &mut on, readonly)?;
        if let Some(sub) = sub {
            n += 1;
            if sub.op != SubOp::Get {
                let need = bitmaps::reach(&sub);
                grow = Some(grow.map_or(need, |had: usize| had.max(need)));
            }
        }
        at = next;
    }

    out.array(n);
    let key = args.get(1);
    db.at(key).bitfield_with(key, grow, |bytes| {
        let mut at = 2;
        let mut on = Overflow::Wrap;
        while at < args.len() {
            // The arguments have been through `parse` once already, so anything
            // it could refuse has been refused and the second pass cannot fail.
            let (sub, next) = match parse(args, at, &mut on, readonly) {
                Ok(step) => step,
                Err(_) => break,
            };
            if let Some(sub) = sub {
                match bitmaps::apply(bytes, sub) {
                    Some(n) => out.int(n),
                    None => out.nil(),
                }
            }
            at = next;
        }
    })?;
    Ok(())
}

/// One `BITFIELD` subcommand, and where the next one starts.
///
/// `None` for `OVERFLOW`, which is not a subcommand and writes no reply: it
/// changes what the subcommands after it do and that is all. The word is carried
/// in `on` rather than answered so that the caller does not have to know which
/// of the two kinds it just read.
fn parse(
    args: Args<'_>,
    at: usize,
    on: &mut Overflow,
    readonly: bool,
) -> Result<(Option<Sub>, usize)> {
    let word = args.get(at);
    if is(word, b"overflow") {
        let arg = args.opt(at + 1).ok_or_else(args::syntax)?;
        *on = Overflow::parse(arg).ok_or_else(|| Error::new(Code::Invalid, BAD_OVERFLOW))?;
        return Ok((None, at + 2));
    }
    let get = is(word, b"get");
    let set = is(word, b"set");
    let incr = is(word, b"incrby");
    if !get && !set && !incr {
        return Err(args::syntax());
    }
    // The read only form refuses a write before it looks at the field, which is
    // why this is not folded into the arity check below.
    if readonly && !get {
        return Err(Error::new(Code::Unsupported, RO_GET_ONLY));
    }
    let words = if get { 3 } else { 4 };
    if at + words > args.len() {
        return Err(args::syntax());
    }
    let field =
        Field::parse(args.get(at + 1)).ok_or_else(|| Error::new(Code::Invalid, BAD_TYPE))?;
    let bit = at_bit(args.get(at + 2), field)?;
    let op = if get {
        SubOp::Get
    } else {
        let n = args.int(at + 3)?;
        if set { SubOp::Set(n) } else { SubOp::Incr(n) }
    };
    Ok((
        Some(Sub {
            op,
            field,
            at: bit,
            on: *on,
        }),
        at + words,
    ))
}

/// A `BITFIELD` offset, which is a bit index or a `#` and a field index.
///
/// The `#` form multiplies by the width of the field, so `#2` on a `u8` is bit
/// sixteen, and it is there so that a caller treating the string as an array
/// does not have to do the multiplication in its own language.
fn at_bit(arg: &[u8], field: Field) -> Result<u64> {
    let bad = || Error::new(Code::Invalid, BAD_OFFSET);
    let (digits, scale) = match arg.split_first() {
        Some((b'#', rest)) => (rest, u64::from(field.bits())),
        _ => (arg, 1),
    };
    let n = parse_i64(digits).ok_or_else(bad)?;
    let n = u64::try_from(n).map_err(|_| bad())?;
    let bit = n.checked_mul(scale).ok_or_else(bad)?;
    // The last bit of the field has to be inside the world too, or a field at
    // the very top would be accepted here and refused by the keyspace with the
    // other sentence.
    if field.last_bit(bit) > bitmaps::BIT_OFFSET_MAX {
        return Err(bad());
    }
    Ok(bit)
}

/// A `SETBIT` or `GETBIT` offset.
fn offset(arg: &[u8]) -> Result<u64> {
    match parse_i64(arg) {
        Some(n) if n >= 0 && n as u64 <= bitmaps::BIT_OFFSET_MAX => Ok(n as u64),
        _ => Err(Error::new(Code::Invalid, BAD_OFFSET)),
    }
}

/// The `start end [BYTE | BIT]` tail `BITCOUNT` and `BITPOS` share.
///
/// `None` when there is no tail at all. A lone start is a syntax error, and so
/// is a fourth word that is not one of the two units, which is why `BITPOS k 0 5
/// BIT` is refused: the `BIT` is read as the end index and the command runs out
/// of places to put it.
fn range(args: Args<'_>, at: usize) -> Result<Option<(i64, i64, Unit)>> {
    if args.len() <= at {
        return Ok(None);
    }
    if args.len() < at + 2 || args.len() > at + 3 {
        return Err(args::syntax());
    }
    let start = args.int(at)?;
    let end = args.int(at + 1)?;
    let unit = match args.opt(at + 2) {
        None => Unit::Byte,
        Some(w) if is(w, b"byte") => Unit::Byte,
        Some(w) if is(w, b"bit") => Unit::Bit,
        Some(_) => return Err(args::syntax()),
    };
    Ok(Some((start, end, unit)))
}

/// A length as the integer the reply wants.
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}
