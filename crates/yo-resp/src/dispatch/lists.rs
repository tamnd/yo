//! The list commands, on the wire.
//!
//! The same shape as [`super::sets`]: the name has been looked up and the arity
//! has been checked, so this turns arguments into a call on [`Db`] and the
//! answer into a reply. No decisions about lists are made here and none about
//! representations, because the wire and the embedded API have to reach the same
//! code or there are two implementations of `LPUSH` and one of them is wrong
//! (Y23).
//!
//! # Two kinds of nothing
//!
//! `LPOP key` on a missing key is a null and `LPOP key 2` is a null array, and a
//! RESP2 client can tell those apart on the wire: one is `$-1` and the other is
//! `*-1`. `LPOS key x` is a null and `LPOS key x COUNT 2` is an empty array,
//! which is not the same thing again. Every one of those was read off a running
//! 8.8 rather than reasoned about, because the reasoning gives the wrong answer
//! at least twice: an empty array would be the sensible reply to `LPOP key 2` on
//! a missing key and it is not what Redis sends.
//!
//! # Nothing is collected on the way out
//!
//! `LRANGE`, `LPOS COUNT 0` and the count forms of `LPOP` all write straight
//! into the connection's out buffer as they walk, and the array header goes on
//! afterwards through [`Out::close_array`]. A list of ten thousand elements is
//! never a `Vec` of ten thousand anything on this thread (Y18).

use yo_common::{Code, Error, Result, parse_i64};
use yo_kv::{Db, End, Entry, Movem, Order};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What Redis says about a `LPOP` count it will not take.
///
/// The same sentence for a negative number and for something that is not a
/// number at all, which is `getRangeLongFromObject` being handed a message to
/// use instead of the usual one.
const BAD_POP_COUNT: &str = "value is out of range, must be positive";
/// `LMPOP`'s two, which are its own sentences and not the usual ones.
pub(super) const BAD_NUMKEYS: &str = "numkeys should be greater than 0";
/// `LMPOP`'s count, and `LMOVEM`'s, which turned out to be the same sentence.
/// Both say it for zero, for a negative and for something that is not a number
/// at all, the way `SPOP`'s message does.
pub(super) const BAD_MPOP_COUNT: &str = "count should be greater than 0";
/// What `LPOS` says about the two options that may not be negative.
const BAD_COUNT: &str = "COUNT can't be negative";
const BAD_MAXLEN: &str = "MAXLEN can't be negative";

/// Run one list command.
pub(super) fn execute(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "lpush" => {
            let key = args.get(1);
            out.int(count(db.hold(key).push(key, End::Left, rest(args))?));
        }
        "rpush" => {
            let key = args.get(1);
            out.int(count(db.hold(key).push(key, End::Right, rest(args))?));
        }
        "lpushx" => {
            let key = args.get(1);
            out.int(count(db.hold(key).pushx(key, End::Left, rest(args))?));
        }
        "rpushx" => {
            let key = args.get(1);
            out.int(count(db.hold(key).pushx(key, End::Right, rest(args))?));
        }
        "lpop" => pop(db, spec, args, End::Left, out)?,
        "rpop" => pop(db, spec, args, End::Right, out)?,
        "llen" => {
            let key = args.get(1);
            out.int(count(db.hold(key).llen(key)?));
        }
        "lrange" => {
            let (start, stop) = (args.int(2)?, args.int(3)?);
            let key = args.get(1);
            let mark = out.len();
            let mut n = 0;
            for e in db.hold(key).lrange(key, start, stop)? {
                element(out, e);
                n += 1;
            }
            out.close_array(mark, n);
        }
        "lindex" => {
            let key = args.get(1);
            match db.hold(key).lindex(key, args.int(2)?)? {
                Some(e) => element(out, e),
                None => out.nil(),
            }
        }
        "lset" => {
            let key = args.get(1);
            db.hold(key).lset(key, args.int(2)?, args.get(3))?;
            out.ok();
        }
        "linsert" => {
            let before = if args::is(args.get(2), b"before") {
                true
            } else if args::is(args.get(2), b"after") {
                false
            } else {
                return Err(args::syntax());
            };
            let key = args.get(1);
            out.int(
                db.hold(key)
                    .linsert(key, before, args.get(3), args.get(4))?,
            );
        }
        "lrem" => {
            let key = args.get(1);
            out.int(count(db.hold(key).lrem(key, args.int(2)?, args.get(3))?));
        }
        "ltrim" => {
            let key = args.get(1);
            db.hold(key).ltrim(key, args.int(2)?, args.int(3)?)?;
            out.ok();
        }
        "lpos" => lpos(db, args, out)?,
        // `RPOPLPUSH` is `LMOVE` with its two ends fixed, and Redis says as much
        // in its own source. It stays a separate name because it is the one
        // every client library still sends.
        "rpoplpush" => moved(db, args.get(1), args.get(2), End::Right, End::Left, out)?,
        "lmove" => {
            let from = end_of(args.get(3))?;
            let to = end_of(args.get(4))?;
            moved(db, args.get(1), args.get(2), from, to, out)?;
        }
        "lmovem" => movem(db, args, out)?,
        "lmpop" => mpop(db, args, out)?,
        other => unreachable!("the table sent {other} to the list group"),
    }
    Ok(())
}

/// `LPOP key [count]` and `RPOP key [count]`.
///
/// The two forms are different replies and not one reply with a length, which
/// is why the count is looked at before anything is popped: without it the
/// answer is one element or a null, and with it an array or a null array, even
/// when the count is one.
fn pop(db: &Db, spec: &Spec, args: Args<'_>, end: End, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    if args.len() == 2 {
        let mut got = false;
        db.hold(key).pop_into(key, end, 1, |e| {
            element(out, e);
            got = true;
        })?;
        if !got {
            out.nil();
        }
        return Ok(());
    }
    // The arity in the table is a minimum, because a count is optional, so a
    // third argument has to be caught here. Redis calls this an arity error and
    // not a syntax error, which is a distinction it does not always make.
    if args.len() != 3 {
        return Err(args::wrong_arity(spec.name));
    }
    // Redis reads this with its own message rather than the usual one, so
    // `LPOP k abc` and `LPOP k -1` are the same error and neither is `value is
    // not an integer`.
    let want = match args.int(2) {
        Ok(n) if n >= 0 => usize::try_from(n).unwrap_or(usize::MAX),
        _ => return Err(Error::new(Code::Invalid, BAD_POP_COUNT)),
    };
    // A key that is not there is a null array and not an empty one, and a key
    // that is there with a count of zero is an empty array. Telling those apart
    // needs the length before the pop, because a pop of zero cannot report the
    // difference.
    // The stripe once, since the length and the pop are the same key.
    let mut stripe = db.hold(key);
    if stripe.llen(key)? == 0 {
        out.nil_array();
        return Ok(());
    }
    let mark = out.len();
    let n = stripe.pop_into(key, end, want, |e| element(out, e))?;
    out.close_array(mark, n);
    Ok(())
}

/// `LPOS key element [RANK rank] [COUNT count] [MAXLEN len]`.
///
/// Without `COUNT` this answers one position or a null. With it, an array,
/// which is empty rather than null when nothing matched and when the key is not
/// there. Those are three different replies for what a client might reasonably
/// call the same outcome, and all three are Redis's.
fn lpos(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut rank = 1i64;
    let mut wanted: Option<usize> = None;
    let mut maxlen = 0usize;
    let mut i = 3;
    while i < args.len() {
        // An option with nothing after it is a syntax error and not an option
        // with a default, so the pair is checked before either half is read.
        if i + 1 >= args.len() {
            return Err(args::syntax());
        }
        let opt = args.get(i);
        if args::is(opt, b"rank") {
            rank = args.int(i + 1)?;
        } else if args::is(opt, b"count") {
            wanted = Some(non_negative(args.int(i + 1)?, BAD_COUNT)?);
        } else if args::is(opt, b"maxlen") {
            maxlen = non_negative(args.int(i + 1)?, BAD_MAXLEN)?;
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }

    let (key, want) = (args.get(1), args.get(2));
    match wanted {
        Some(n) => {
            let mark = out.len();
            let found = db
                .hold(key)
                .lpos_into(key, want, rank, n, maxlen, |at| out.int(count(at)))?;
            out.close_array(mark, found);
        }
        None => {
            let mut found = None;
            db.hold(key)
                .lpos_into(key, want, rank, 1, maxlen, |at| found = Some(at))?;
            match found {
                Some(at) => out.int(count(at)),
                None => out.nil(),
            }
        }
    }
    Ok(())
}

/// `LMPOP numkeys key [key ...] LEFT|RIGHT [COUNT count]`, and the reply that
/// says which key answered.
///
/// The keys are tried in the order they were sent and the first one holding
/// anything is the one that is popped, which is the whole point of the command:
/// one round trip instead of a `LLEN` per key followed by an `LPOP`.
fn mpop(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let numkeys = match args.int(1) {
        Ok(n) if n > 0 => usize::try_from(n).unwrap_or(usize::MAX),
        _ => return Err(Error::new(Code::Invalid, BAD_NUMKEYS)),
    };
    // A count that runs off the end is a syntax error and not a message about
    // key counts, because the direction that should have followed the keys is
    // simply not there. `LMPOP 3 k LEFT` is the case, and Redis says `syntax
    // error` to it.
    if numkeys >= args.len() - 2 {
        return Err(args::syntax());
    }
    let at = 2 + numkeys;
    let end = end_of(args.get(at))?;
    let mut want = 1usize;
    if at + 1 < args.len() {
        if args.len() != at + 3 || !args::is(args.get(at + 1), b"count") {
            return Err(args::syntax());
        }
        want = match args.int(at + 2) {
            Ok(n) if n > 0 => usize::try_from(n).unwrap_or(usize::MAX),
            _ => return Err(Error::new(Code::Invalid, BAD_MPOP_COUNT)),
        };
    }

    // Every key at once, so the one that answers is the first that had anything
    // in it at one moment rather than the first that had anything in it when
    // the walk reached it.
    let mut held = db.hold_keys((2..at).map(|i| args.get(i)));
    for i in 2..at {
        let key = args.get(i);
        // `LLEN` and not a pop that reports nothing, because the reply carries
        // the name of the key that answered and has to be written before the
        // elements are. A key of another type is an error here, the same as it
        // would be if it were the only key named.
        // Each key on its own stripe, found once and used for both calls.
        let stripe = held.stripe_mut(db.stripe_of(key));
        if stripe.llen(key)? == 0 {
            continue;
        }
        out.array(2);
        out.bulk(key);
        let mark = out.len();
        let n = stripe.pop_into(key, end, want, |e| element(out, e))?;
        out.close_array(mark, n);
        return Ok(());
    }
    // A null array and not a null, even though the reply that would have been
    // there is a two element array holding a key name. Redis writes it with
    // `addReplyNullArray` and a RESP2 client sees `*-1`.
    out.nil_array();
    Ok(())
}

/// `LMOVE` and `RPOPLPUSH`, which are the same command.
fn moved(db: &Db, src: &[u8], dst: &[u8], from: End, to: End, out: &mut Out) -> Result<()> {
    // The element is written from inside the move rather than answered by it,
    // because what it is borrowed from is a stripe that is still held while the
    // reply is being written and is let go of straight after.
    if !db.lmove(src, dst, from, to, |v| out.bulk(v))? {
        out.nil();
    }
    Ok(())
}

/// `LMOVEM src dst LEFT|RIGHT LEFT|RIGHT [COUNT|EXACTLY n OBO|BULK]`.
///
/// The trailing block is all or nothing: with it, both the count and the
/// ordering word have to be there, and `LMOVEM s d LEFT RIGHT COUNT 2` on its
/// own is a syntax error. Without it the command moves one element, which is
/// `LMOVE` with an array around the answer.
///
/// A count that is zero, negative or not a number at all gets the same sentence,
/// which is 8.10.1's behaviour and reads the way `SPOP`'s does: whatever the
/// client typed, what it needs told is which argument was wrong.
///
/// Nothing moved is a nil and not an empty array, so a caller can tell a source
/// that was not there from one that was. It is a null *array* and not a null
/// bulk string, which matters and is not visible through `redis-cli`, because
/// that prints `(nil)` for both. `LMOVE` answers `$-1` because what it would
/// have sent is one element, and `LMOVEM` answers `*-1` because what it would
/// have sent is an array, which is the same rule that makes `LPOP k` a `$-1` and
/// `LPOP k 2` a `*-1`. Read off the wire rather than off a client.
fn movem(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let from = end_of(args.get(3))?;
    let to = end_of(args.get(4))?;
    let block = movem_options(args, 5, from, to)?;
    // Written before the header for the same reason the set algebra's are: how
    // many moved is what the move produced, and an `EXACTLY` that came up short
    // produces none of them.
    let start = out.len();
    let mut n = 0;
    db.lmovem(args.get(1), args.get(2), block, |v| {
        out.bulk(v);
        n += 1;
    })?;
    if n == 0 {
        out.nil_array();
    } else {
        out.close_array(start, n);
    }
    Ok(())
}

/// The `COUNT|EXACTLY n OBO|BULK` block, wherever it sits.
///
/// `at` is where the first of the three would be, which is 5 for `LMOVEM` and 6
/// for `BLMOVEM`, since the only difference between the two lines is the timeout
/// the second one carries in front of the options. Nothing at `at` is the short
/// form and means one element in source order, all three is the long form, and
/// anything in between is a syntax error rather than a shorter spelling.
pub(super) fn movem_options(args: Args<'_>, at: usize, from: End, to: End) -> Result<Movem> {
    if args.len() == at {
        return Ok(Movem {
            from,
            to,
            count: 1,
            exactly: false,
            order: Order::Bulk,
        });
    }
    if args.len() != at + 3 {
        return Err(args::syntax());
    }
    let exactly = if args::is(args.get(at), b"exactly") {
        true
    } else if args::is(args.get(at), b"count") {
        false
    } else {
        return Err(args::syntax());
    };
    // The count is read before the ordering word, which is only visible on a
    // line that is wrong in both places: 8.10.1 answers about the count there.
    // The order the two are looked at in is the sort of thing a client's own
    // test suite pins, so it was measured rather than picked.
    let count = match parse_i64(args.get(at + 1)) {
        Some(n) if n > 0 => usize::try_from(n).unwrap_or(usize::MAX),
        _ => return Err(Error::new(Code::Invalid, BAD_MPOP_COUNT)),
    };
    let order = if args::is(args.get(at + 2), b"obo") {
        Order::OneByOne
    } else if args::is(args.get(at + 2), b"bulk") {
        Order::Bulk
    } else {
        return Err(args::syntax());
    };
    Ok(Movem {
        from,
        to,
        count,
        exactly,
        order,
    })
}

/// `LEFT` or `RIGHT`, and a syntax error for anything else.
pub(super) fn end_of(arg: &[u8]) -> Result<End> {
    if args::is(arg, b"left") {
        Ok(End::Left)
    } else if args::is(arg, b"right") {
        Ok(End::Right)
    } else {
        Err(args::syntax())
    }
}

/// A number that may not be negative, with the message the option carries.
fn non_negative(n: i64, msg: &'static str) -> Result<usize> {
    if n < 0 {
        return Err(Error::new(Code::Invalid, msg));
    }
    Ok(usize::try_from(n).unwrap_or(usize::MAX))
}

/// Every argument after the key, which for the push commands is every element.
#[inline]
fn rest(args: Args<'_>) -> impl Iterator<Item = &[u8]> + Clone {
    (2..args.len()).map(move |i| args.get(i))
}

/// One element as the client sees it.
///
/// An element stored as an integer has no digits anywhere until this line,
/// because a listpack holds the number and not its text. Formatting it here
/// rather than on the way in is Y18, the same argument a set member gets.
#[inline]
fn element(out: &mut Out, e: Entry<'_>) {
    match e {
        Entry::Int(n) => out.bulk_int(n),
        Entry::Str(s) => out.bulk(s),
    }
}

/// A length or an index as the integer the reply carries.
///
/// The saturation cannot happen: a list is bounded by memory long before it is
/// bounded by `i64`. It is here because the alternative is a cast that clippy
/// will not have and a panic that would be worse than a wrong number.
#[inline]
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}
