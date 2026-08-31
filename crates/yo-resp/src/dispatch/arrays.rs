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

use yo_common::Result;
use yo_kv::arrays::parse_index;
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
            let index = parse_index(args.get(2))?;
            match db.arget(args.get(1), index)? {
                Some(e) => element(out, e),
                None => out.nil(),
            }
        }
        "armget" => {
            for i in 2..args.len() {
                parse_index(args.get(i))?;
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
        other => unreachable!("the table sent {other} to the array group"),
    }
    Ok(())
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
