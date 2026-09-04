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

use yo_common::num::{DOUBLE_MAX, parse_i64, write_g17};
use yo_common::{Code, Error, Result};
use yo_kv::arrays::{Aggregate, Grep, Op, Test, parse_grep_bound, parse_index, parse_seek_index};
use yo_kv::{ArrayElement, Db, Keyspace};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// Run one array command.
///
/// Every command in the group names one key and names it first, so the stripe
/// is found once here and everything below goes on taking a keyspace.
pub(super) fn execute(db: &mut Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    let db = db.at(args.get(1));
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
        "argrep" => {
            // Both bounds are read before the plan is, so `ARGREP k x 0 EXACT`
            // is a bad index and not a syntax error, whichever way round the
            // two mistakes are written.
            let start = parse_grep_bound(args.get(2))?;
            let end = parse_grep_bound(args.get(3))?;
            let mut grep = Grep::new();
            let (mut all, mut nocase, mut withvalues) = (false, false, false);
            let mut limit = u64::MAX;
            // One pass, so predicates and options mix freely and the last of a
            // repeated option wins. That is Redis's parser and it is the reason
            // `ARGREP k 0 1 NOCASE RE a` and `ARGREP k 0 1 RE a NOCASE` are the
            // same command.
            let mut i = 4;
            while i < args.len() {
                let token = args.get(i);
                let test = match token {
                    t if args::is(t, b"EXACT") => Some(Test::Exact),
                    t if args::is(t, b"MATCH") => Some(Test::Match),
                    t if args::is(t, b"GLOB") => Some(Test::Glob),
                    t if args::is(t, b"RE") => Some(Test::Re),
                    _ => None,
                };
                if let Some(test) = test {
                    if i + 1 >= args.len() {
                        return Err(args::syntax());
                    }
                    grep.push(test, args.get(i + 1))?;
                    i += 2;
                    continue;
                }
                match token {
                    t if args::is(t, b"LIMIT") => {
                        if i + 1 >= args.len() {
                            return Err(args::syntax());
                        }
                        let n = args.int(i + 1)?;
                        if n <= 0 {
                            return Err(Error::new(Code::Invalid, "LIMIT must be positive"));
                        }
                        limit = n as u64;
                        i += 2;
                    }
                    t if args::is(t, b"AND") => {
                        all = true;
                        i += 1;
                    }
                    t if args::is(t, b"OR") => {
                        all = false;
                        i += 1;
                    }
                    t if args::is(t, b"WITHVALUES") => {
                        withvalues = true;
                        i += 1;
                    }
                    t if args::is(t, b"NOCASE") => {
                        nocase = true;
                        i += 1;
                    }
                    _ => return Err(args::syntax()),
                }
            }
            // Asking for nothing is a syntax error rather than an empty reply,
            // because a client that meant to send a predicate and lost it in a
            // shell should hear about it.
            if grep.is_empty() {
                return Err(args::syntax());
            }
            grep.compile(all, nocase)?;
            let mark = out.len();
            let n = db.argrep(args.get(1), start, end, limit, &mut grep, |index, el| {
                // A bare index unless the values were asked for, and a pair of
                // index and value when they were. Either way one entry per hit,
                // which is what the header at the end counts.
                if withvalues {
                    out.array(2);
                }
                out.uint(index);
                if withvalues {
                    element(out, el);
                }
            })?;
            out.close_array(mark, usize::try_from(n).unwrap_or(usize::MAX));
        }
        "arop" => {
            let start = parse_index(args.get(2))?;
            let end = parse_index(args.get(3))?;
            let op = match args.get(4) {
                w if args::is(w, b"SUM") => Op::Sum,
                w if args::is(w, b"MIN") => Op::Min,
                w if args::is(w, b"MAX") => Op::Max,
                w if args::is(w, b"AND") => Op::And,
                w if args::is(w, b"OR") => Op::Or,
                w if args::is(w, b"XOR") => Op::Xor,
                w if args::is(w, b"MATCH") => Op::Match,
                w if args::is(w, b"USED") => Op::Used,
                _ => return Err(Error::new(Code::Invalid, "unknown operation")),
            };
            // MATCH is the only one that takes a value, and it says so in its
            // own words rather than as an arity error.
            if op == Op::Match {
                if args.len() != 6 {
                    return Err(Error::new(Code::Invalid, "MATCH requires a value argument"));
                }
            } else if args.len() != 5 {
                return Err(args::wrong_arity(spec.name));
            }
            match db.arop(args.get(1), start, end, op, args.get(5))? {
                Aggregate::Int(n) => out.int(n),
                Aggregate::Num(d) => {
                    // Redis prints an aggregate with seventeen significant
                    // digits and not with the shortest round trip printer every
                    // other reply uses, so this is the one place those two
                    // disagree.
                    let mut buf = [0u8; DOUBLE_MAX];
                    out.bulk(write_g17(&mut buf, d));
                }
                Aggregate::None => out.nil(),
            }
        }
        "arinfo" => {
            let full = match args.len() {
                2 => false,
                3 if args::is(args.get(2), b"FULL") => true,
                _ => return Err(args::syntax()),
            };
            let info = db.arinfo(args.get(1), full)?;
            out.map(if full { 12 } else { 7 });
            out.bulk(b"count");
            out.uint(info.count);
            out.bulk(b"len");
            out.uint(info.len);
            out.bulk(b"next-insert-index");
            out.uint(info.next_insert);
            out.bulk(b"slices");
            out.uint(info.slices);
            out.bulk(b"directory-size");
            out.uint(info.directory_size);
            out.bulk(b"super-dir-entries");
            // Redis grows a second level above the directory once one level is
            // wasteful. We do not have one, and the number is here so that a
            // client reading the map finds the field it expects (D-20).
            out.uint(0);
            out.bulk(b"slice-size");
            out.uint(info.slice_size);
            if full {
                out.bulk(b"dense-slices");
                out.uint(info.dense_slices);
                out.bulk(b"sparse-slices");
                out.uint(info.sparse_slices);
                out.bulk(b"avg-dense-size");
                out.double(info.avg_dense_size);
                out.bulk(b"avg-dense-fill");
                out.double(info.avg_dense_fill);
                out.bulk(b"avg-sparse-size");
                out.double(info.avg_sparse_size);
            }
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
