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

use yo_common::Result;
use yo_kv::{Keyspace, Member};

use super::args::Args;
use super::table::Spec;
use crate::reply::Out;

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
        other => unreachable!("{other} is not a set command"),
    }
    Ok(())
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
