//! `TOPK.*`, the heavy keeper RedisBloom put on the wire.
//!
//! The sketch is [`yo_sketch::topk::TopK`] and this is the wire in front of it,
//! the same split as `super::cms`. Seven commands: one to make the sketch, two
//! to feed it, three to ask it questions and one to report its shape.
//!
//! # Errors
//!
//! Nine sentences. Eight are bare, with no `ERR` in front, because the module
//! writes them with its own prefix and that prefix is `TopK:`. The ninth is the
//! one about memory, which the module wrote with `ERR` inside the string, so it
//! goes out with a prefix the other eight do not have. The one about the
//! increment carries twenty eight spaces in the middle of it, which is a line
//! continuation in the module's C that ended up in the sentence, and it is
//! copied because it is what a client's error handler is matching on.
//!
//! # The order the checks happen in
//!
//! `TOPK.RESERVE` looks at the key before its arguments, so a bad width against
//! a key that is taken answers that the key is taken. It also takes three
//! arguments or six and nothing in between, and four or five is a wrong arity
//! rather than a missing decay. `TOPK.LIST` is the other way round: the keyword
//! is checked before the key, so `TOPK.LIST missing BAD` complains about the
//! keyword. `TOPK.INCRBY` reads and applies one pair at a time, so a bad
//! increment in the middle leaves everything before it applied, which is the
//! opposite of `CMS.INCRBY` and is worth knowing before writing a retry.
//!
//! # Where the two protocols disagree
//!
//! `TOPK.QUERY`, which is a bool on RESP3 and an integer on RESP2, and
//! `TOPK.INFO`, which is a map of four on RESP3 and a flat array of eight on
//! RESP2 and whose decay goes out through the double reply either way.
//! `TOPK.COUNT` is an integer on both.
//!
//! # What a client can see that is different
//!
//! Three things, all of them written down. The decay draw is deterministic here
//! and random on the reference, which is D-49 and which means the counts differ
//! between the two servers on any sketch narrow enough to collide. A counter
//! stops at the ceiling rather than wrapping, D-50. And a reply cut short by a
//! bad increment is a shorter array here rather than an array header that
//! promises more elements than follow it, D-51.

use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};
use yo_kv::{Db, Foreign, Keyspace};
use yo_sketch::topk::TopK;

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// The width a `TOPK.RESERVE` with no shape asks for.
const DEFAULT_WIDTH: u32 = 8;
/// The depth it asks for.
const DEFAULT_DEPTH: u32 = 7;
/// And the decay.
const DEFAULT_DECAY: f64 = 0.9;
/// The largest increment `TOPK.INCRBY` takes, which the module wrote as a
/// hundred thousand and put in the error message with a comma in it.
const MAX_INCREMENT: i64 = 100_000;

/// What `TOPK.RESERVE` says about a key that is already there, whatever it
/// holds. The existence is what is checked and not the type, so a key holding a
/// string gets this and not `WRONGTYPE`.
const EXISTS: &[u8] = b"TopK: key already exists";
/// What the other six say about a key that is not there.
const MISSING: &[u8] = b"TopK: key does not exist";
/// A `k` that is not a whole number between one and four billion.
const BAD_K: &[u8] = b"TopK: invalid k";
/// A width that is not.
const BAD_WIDTH: &[u8] = b"TopK: invalid width";
/// A depth that is not.
const BAD_DEPTH: &[u8] = b"TopK: invalid depth";
/// A decay outside zero to one. The quotes and the order of the two bounds are
/// the module's.
const BAD_DECAY: &[u8] = b"TopK: invalid decay value. must be '<= 1' & '> 0'";
/// A sketch too large to build. This is the one error in the family that carries
/// a prefix, because the module wrote the prefix into the sentence itself.
const NO_MEMORY: &[u8] = b"ERR Insufficient memory to create topk data structure";
/// A word after the key in `TOPK.LIST` that is not a prefix of `WITHCOUNT`.
const KEYWORD: &[u8] = b"WITHCOUNT keyword expected";
/// An increment that is not a whole number from zero to a hundred thousand. The
/// run of spaces is a line continuation in the module's C that became part of
/// the string, and clients match on the whole sentence.
const BAD_INCREMENT: &[u8] = b"TopK: increment must be an integer greater or equal to 0                            and smaller or equal to 100,000";

/// What a key holding anything else gets.
const WRONG_KIND: &str = "Operation against a key holding the wrong kind of value";

/// A sketch under a key.
#[derive(Debug)]
pub(super) struct TopKBody {
    /// The heavy keeper. Everything `TOPK.INFO` reports comes off it.
    t: TopK,
}

impl Foreign for TopKBody {
    fn type_name(&self) -> &'static str {
        // The module's own name for the type, which is the one word in the
        // family that is not spelled `TopK` the way the errors spell it.
        "TopK-TYPE"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.t.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // A sketch that has counted nothing is still a key, the same as an
        // empty filter on any of the other families.
        false
    }
}

pub(super) fn execute(db: &mut Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    // Every command here names one sketch and names it first, so the stripe is
    // found once and everything below goes on taking a keyspace.
    let db = db.at(args.get(1));
    match spec.name {
        "topk.reserve" => reserve(db, args, out),
        "topk.add" => add(db, args, out),
        "topk.incrby" => incrby(db, args, out),
        "topk.query" => query(db, args, out),
        "topk.count" => count(db, args, out),
        "topk.list" => list(db, args, out),
        "topk.info" => info(db, args, out),
        other => unreachable!("{other} is not a top k command"),
    }
}

/// `TOPK.RESERVE key k [width depth decay]`.
///
/// The three optional arguments come as a set or not at all, so the command is
/// three words long or six and anything else is a wrong arity. The defaults are
/// a width of eight and a depth of seven, which is fifty six buckets and is
/// small enough that anything past a handful of distinct items collides.
fn reserve(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 3 && args.len() != 6 {
        return Err(args::wrong_arity("topk.reserve"));
    }
    let key = args.get(1);
    if db.kind_of(key).is_some() {
        out.error(EXISTS);
        return Ok(());
    }
    let Some(k) = size(args.get(2)) else {
        out.error(BAD_K);
        return Ok(());
    };
    let (mut width, mut depth, mut decay) = (DEFAULT_WIDTH, DEFAULT_DEPTH, DEFAULT_DECAY);
    if args.len() == 6 {
        let Some(w) = size(args.get(3)) else {
            out.error(BAD_WIDTH);
            return Ok(());
        };
        let Some(d) = size(args.get(4)) else {
            out.error(BAD_DEPTH);
            return Ok(());
        };
        // A NaN fails both comparisons and so is refused without being named,
        // which is what the module's `decay <= 0 || decay > 1` does too.
        let Some(rate) = parse_f64(args.get(5)).filter(|&n| n > 0.0 && n <= 1.0) else {
            out.error(BAD_DECAY);
            return Ok(());
        };
        (width, depth, decay) = (w, d, rate);
    }
    let Some(t) = TopK::new(k, width, depth, decay) else {
        out.error(NO_MEMORY);
        return Ok(());
    };
    db.put_foreign(key, Box::new(TopKBody { t }));
    out.ok();
    Ok(())
}

/// `TOPK.ADD key item [item ...]`, which is one occurrence of each.
///
/// The reply is one element per item: a null when the kept set did not change,
/// and the item that lost its place when it did. An item can expel itself out of
/// nothing, since a slot nothing has reached yet answers a null too.
fn add(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = write(db, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    out.array(args.len() - 2);
    for i in 2..args.len() {
        expelled(body.t.add(args.get(i), 1), out);
    }
    Ok(())
}

/// `TOPK.INCRBY key item increment [item increment ...]`.
///
/// Pairs are read and applied one at a time, so the first bad increment ends the
/// command with everything before it already counted. That is the module's
/// behaviour and it is the one thing here a client has to write code around.
fn incrby(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if !args.len().is_multiple_of(2) {
        return Err(args::wrong_arity("topk.incrby"));
    }
    let Some(body) = write(db, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    // The length is written after the elements because a bad increment stops
    // the walk, and an array header that promises more than it delivers leaves
    // the connection out of step. D-51.
    let start = out.len();
    let mut written = 0;
    for i in (2..args.len()).step_by(2) {
        match parse_i64(args.get(i + 1)).filter(|&n| (0..=MAX_INCREMENT).contains(&n)) {
            Some(by) => expelled(body.t.add(args.get(i), by as u32), out),
            None => {
                out.error(BAD_INCREMENT);
                written += 1;
                break;
            }
        }
        written += 1;
    }
    out.close_array(start, written);
    Ok(())
}

/// Either the item that lost its place or a null, which is what both of the two
/// feeding commands answer per item.
fn expelled(item: Option<Box<[u8]>>, out: &mut Out) {
    match item {
        Some(item) => out.bulk(&item),
        None => out.nil(),
    }
}

/// `TOPK.QUERY key item [item ...]`, which asks whether each item is being kept
/// and not how often it has been seen.
fn query(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    out.array(args.len() - 2);
    for i in 2..args.len() {
        out.bool(body.t.query(args.get(i)));
    }
    Ok(())
}

/// `TOPK.COUNT key item [item ...]`, which is the sketch's estimate for each.
///
/// An item the sketch is not keeping still has a count, and it is the one the
/// buckets happen to hold, which for a rare item that shared a bucket with a
/// common one reads as zero. The module's own documentation calls this reply
/// approximate and warns that it can be well under the truth.
fn count(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    out.array(args.len() - 2);
    for i in 2..args.len() {
        out.uint(u64::from(body.t.count_of(args.get(i))));
    }
    Ok(())
}

/// `TOPK.LIST key [WITHCOUNT]`, heaviest first.
///
/// The keyword is compared against as many characters as the client sent, so
/// every prefix of `WITHCOUNT` turns the counts on, including the empty string,
/// and only a longer word or a different one is refused. That is a `strncasecmp`
/// with the wrong length in the module and it is observable, so it is copied.
fn list(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 2 && args.len() != 3 {
        return Err(args::wrong_arity("topk.list"));
    }
    let mut counts = false;
    if let Some(word) = args.opt(2) {
        if !is_prefix_of(word, b"withcount") {
            out.error(KEYWORD);
            return Ok(());
        }
        counts = true;
    }
    let Some(body) = read(db, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    let kept = body.t.list();
    out.array(kept.len() * if counts { 2 } else { 1 });
    for (item, n) in kept {
        out.bulk(item);
        if counts {
            out.uint(u64::from(n));
        }
    }
    Ok(())
}

/// `TOPK.INFO key`, which is the four numbers the sketch was made with.
fn info(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    out.map(4);
    out.simple(b"k");
    out.uint(u64::from(body.t.k()));
    out.simple(b"width");
    out.uint(u64::from(body.t.width()));
    out.simple(b"depth");
    out.uint(u64::from(body.t.depth()));
    out.simple(b"decay");
    out.double(body.t.decay());
    Ok(())
}

/// An argument that has to be a whole number a `u32` can hold and not zero,
/// which is what all three of the sizes are.
fn size(arg: &[u8]) -> Option<u32> {
    parse_i64(arg)
        .filter(|&n| n >= 1 && n <= i64::from(u32::MAX))
        .map(|n| n as u32)
}

/// Whether `word` is a prefix of `full`, ignoring case, which is the comparison
/// `TOPK.LIST` does and is not the comparison anything else in this engine does.
fn is_prefix_of(word: &[u8], full: &[u8]) -> bool {
    word.len() <= full.len() && word.eq_ignore_ascii_case(&full[..word.len()])
}

/// The sketch under `key` for writing, or `None` if the key is not there.
fn write<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d mut TopKBody>> {
    match db.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<TopKBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// The same, for reading.
fn read<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d TopKBody>> {
    match db.foreign(key)? {
        Some(body) => match body.downcast_ref::<TopKBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}
