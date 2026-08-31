//! The array commands, on the wire.
//!
//! The same shape as [`super::lists`]: the name has been looked up and the
//! arity has been checked, so this turns arguments into a call on [`Keyspace`]
//! and the answer into a reply. No decisions about arrays are made here.
//!
//! # Every index is read before any of them is used
//!
//! `ARMGET`, `ARMSET`, `ARDEL` and `ARDELRANGE` all take a list of indices, and
//! all four of them read the whole list before they touch the key. That is
//! Redis's behaviour and it matters: `ARDEL k 1 2 nope` deletes nothing rather
//! than deleting two things and then failing, and `ARMGET k 1 nope` writes no
//! part of an array header. Doing it any other way leaves a client that got an
//! error unable to say what happened.
//!
//! # The two numbers
//!
//! `ARLEN` is the highest populated index plus one and `ARCOUNT` is how many
//! indices hold something. They are both unsigned, and `ARLEN` genuinely can be
//! a number that does not fit an `i64`, which is why the replies here go
//! through [`Out::uint`].

use yo_common::num::parse_i64;
use yo_common::{Code, Error, Result};
use yo_kv::arrays::{parse_index, parse_seek_index};
use yo_kv::{ArrayElement, Keyspace};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// Run one array command.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "arset" => {
            let index = parse_index(args.get(2))?;
            let values = (3..args.len()).map(|i| args.get(i));
            out.uint(db.arset(args.get(1), index, values)?);
        }
        "armset" => {
            // Pairs, so an odd tail is an arity error and not a syntax one.
            if !args.len().is_multiple_of(2) {
                return Err(args::wrong_arity(spec.name));
            }
            for i in (2..args.len()).step_by(2) {
                parse_index(args.get(i))?;
            }
            let pairs = (2..args.len())
                .step_by(2)
                .map(|i| (parse_index(args.get(i)).unwrap_or(0), args.get(i + 1)));
            out.uint(db.armset(args.get(1), pairs)?);
        }
        "arget" => {
            let index = index_after_type(db, args.get(1), args.get(2))?;
            match db.arget(args.get(1), index)? {
                Some(e) => element(out, e),
                None => out.nil(),
            }
        }
        "armget" => {
            for i in 2..args.len() {
                index_after_type(db, args.get(1), args.get(i))?;
            }
            out.array(args.len() - 2);
            let indices = (2..args.len()).map(|i| parse_index(args.get(i)).unwrap_or(0));
            db.arget_into(args.get(1), indices, |el| reply(out, el))?;
        }
        "argetrange" => {
            let start = parse_index(args.get(2))?;
            let end = parse_index(args.get(3))?;
            // The header carries the count, and the count is known from the two
            // ends before a single position is read, so this needs none of the
            // mark and close dance the collection walks use.
            let mark = out.len();
            let mut n = 0;
            let len = db.argetrange(args.get(1), start, end, |el| {
                reply(out, el);
                n += 1;
            })?;
            debug_assert_eq!(len, n);
            out.close_array(mark, usize::try_from(n).unwrap_or(usize::MAX));
        }
        "arlen" => out.uint(db.arlen(args.get(1))?),
        "arcount" => out.uint(db.arcount(args.get(1))?),
        "ardel" => {
            for i in 2..args.len() {
                parse_index(args.get(i))?;
            }
            let indices = (2..args.len()).map(|i| parse_index(args.get(i)).unwrap_or(0));
            out.uint(db.ardel(args.get(1), indices)?);
        }
        "ardelrange" => {
            if !args.len().is_multiple_of(2) {
                return Err(args::wrong_arity(spec.name));
            }
            for i in 2..args.len() {
                parse_index(args.get(i))?;
            }
            let ranges = (2..args.len()).step_by(2).map(|i| {
                (
                    parse_index(args.get(i)).unwrap_or(0),
                    parse_index(args.get(i + 1)).unwrap_or(0),
                )
            });
            out.uint(db.ardelrange(args.get(1), ranges)?);
        }
        "arinsert" => {
            let values = (2..args.len()).map(|i| args.get(i));
            out.uint(db.arinsert(args.get(1), values)?);
        }
        "arring" => {
            // Redis reads the size before it looks at the key, so a bad size
            // against a string is a bad size and not a wrong type.
            let size =
                parse_i64(args.get(2)).ok_or_else(|| Error::new(Code::Invalid, "invalid size"))?;
            if size <= 0 {
                return Err(Error::new(Code::Invalid, "size must be positive"));
            }
            let values = (3..args.len()).map(|i| args.get(i));
            out.uint(db.arring(args.get(1), size as u64, values)?);
        }
        "arnext" => match db.arnext(args.get(1))? {
            Some(index) => out.uint(index),
            // The cursor is at the top of the space and there is no next index
            // to name, which is the one thing this command cannot answer with a
            // number.
            None => out.nil(),
        },
        "arseek" => {
            let index = parse_seek_index(args.get(2))?;
            out.uint(u64::from(db.arseek(args.get(1), index)?));
        }
        "arlastitems" => {
            let count = args
                .int(2)
                .map_err(|_| Error::new(Code::Invalid, "invalid COUNT"))?;
            // Nothing asked for is an empty reply, and Redis answers it before
            // it has read the option or looked at the key, so this does too.
            if count <= 0 {
                out.array(0);
                return Ok(());
            }
            let newest_first = match args.len() {
                3 => false,
                4 if args::is(args.get(3), b"REV") => true,
                4 => return Err(args::syntax()),
                _ => return Err(args::wrong_arity(spec.name)),
            };
            let mark = out.len();
            let n = db.arlastitems(args.get(1), count as u64, newest_first, |el| reply(out, el))?;
            out.close_array(mark, usize::try_from(n).unwrap_or(usize::MAX));
        }
        "arscan" => {
            let start = parse_index(args.get(2))?;
            let end = parse_index(args.get(3))?;
            let limit = match args.len() {
                4 => u64::MAX,
                6 if args::is(args.get(4), b"LIMIT") => {
                    let n = args.int(5)?;
                    if n <= 0 {
                        return Err(Error::new(Code::Invalid, "LIMIT must be positive"));
                    }
                    n as u64
                }
                6 => return Err(args::syntax()),
                _ => return Err(args::wrong_arity(spec.name)),
            };
            let mark = out.len();
            let n = db.arscan(args.get(1), start, end, limit, |index, el| {
                // A pair per element rather than a flat list, so a client can
                // read the reply without knowing whether it asked for a limit.
                out.array(2);
                out.uint(index);
                element(out, el);
            })?;
            out.close_array(mark, usize::try_from(n).unwrap_or(usize::MAX));
        }
        other => unreachable!("the table sent {other} to the array group"),
    }
    Ok(())
}

/// An index, but with the type of the key reported first when it is a bad one.
///
/// `ARGET` and `ARMGET` are the only two array commands that look the key up
/// before they read the index, so `ARGET stringkey -1` is a wrong type where
/// `ARSET stringkey -1 v` is a bad index. The difference is visible to a client
/// and there is no reasoning behind it beyond the order the two commands happen
/// to be written in, so this reproduces it without paying for it: the type is
/// only looked at once the index has already failed, and the ordinary path is
/// still one lookup.
fn index_after_type(db: &mut Keyspace, key: &[u8], bytes: &[u8]) -> Result<u64> {
    match parse_index(bytes) {
        Ok(index) => Ok(index),
        Err(e) => {
            db.arlen(key)?;
            Err(e)
        }
    }
}

/// One element, or a null for a hole.
fn reply(out: &mut Out, el: Option<ArrayElement<'_>>) {
    match el {
        Some(e) => element(out, e),
        None => out.nil(),
    }
}

/// One element as the bulk string a client sees.
///
/// A value stored as a number is formatted here, into a stack buffer, and
/// copied once into the reply. Formatting it when it was stored would have cost
/// the same work on the write path and the bytes to hold it afterwards.
fn element(out: &mut Out, e: ArrayElement<'_>) {
    match e {
        ArrayElement::Str(s) => out.bulk(s),
        ArrayElement::Short(ref s) => out.bulk(s.as_bytes()),
        ArrayElement::Int(n) => out.bulk_int(n),
        ArrayElement::Float(_) => {
            let mut buf = [0u8; yo_kv::array::ELEMENT_MAX];
            out.bulk(e.text(&mut buf));
        }
    }
}
