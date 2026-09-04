//! `CMS.*`, the count min sketch family RedisBloom put on the wire.
//!
//! The sketch is in `yo-sketch` and this is the wire in front of it, the same
//! split as `super::bloom` and `super::cuckoo`. Six commands, no dump and no
//! delete, and the shapes are small enough that what took the work here was the
//! order the checks happen in rather than the answers.
//!
//! # Errors
//!
//! Eighteen sentences and every one of them is bare, with no `ERR` in front,
//! because the module writes them with its own prefix and that prefix is `CMS:`.
//! They are copied word for word including the two that start with a capital in
//! the middle of a sentence and the one that says `keys/weights`.
//!
//! # The order the checks happen in
//!
//! Both `CMS.INITBYDIM` and `CMS.INITBYPROB` look at the key before they look at
//! their arguments, so a bad width against a key that is taken answers that the
//! key is taken. `CMS.MERGE` resolves its destination first, then reads the key
//! count, then the layout, then the weights, and only then walks the sources,
//! and each source is checked for existence, type and dimensions before the next
//! one is looked at. All of that is observable, so all of it is copied.
//!
//! # Where the two protocols disagree
//!
//! Only `CMS.INFO`, which is a flat array of six on RESP2 and a map of three on
//! RESP3. `CMS.QUERY` and `CMS.INCRBY` answer an array of integers on both, and
//! `CMS.INCRBY` writes an error into that array for an item whose counters have
//! reached the ceiling rather than answering a number.
//!
//! # What a client can see that is different
//!
//! A sketch larger than a gibibyte of counters is refused here and made on the
//! reference, which is D-47. Everything else matches a real Redis 8.10.1 with
//! RedisBloom in it, hash included, so the same increments produce the same
//! counts on either server.

use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};
use yo_kv::{Db, Foreign, Keyspace};
use yo_sketch::cms::{Cms, dims_from};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What the two constructors say about a key that is already there, whatever it
/// holds. The existence is what is checked and not the type, so a key holding a
/// string gets this and not `WRONGTYPE`.
const EXISTS: &[u8] = b"CMS: key already exists";
/// What the other four say about a key that is not.
const MISSING: &[u8] = b"CMS: key does not exist";
/// A width that is not a positive integer.
const BAD_WIDTH: &[u8] = b"CMS: invalid width";
/// A depth that is not.
const BAD_DEPTH: &[u8] = b"CMS: invalid depth";
/// An error tolerance outside zero to one.
const BAD_ERROR: &[u8] = b"CMS: invalid overestimation value";
/// A probability outside zero to one.
const BAD_PROB: &[u8] = b"CMS: invalid prob value";
/// A pair of tolerances that are both in range and ask for dimensions that are
/// not a size, which is a width past `i64::MAX` or a depth of nothing.
const BAD_INIT: &[u8] = b"CMS: invalid init arguments";
/// A sketch too large to build. The reference says this when `calloc` fails and
/// this says it at the cap in D-47.
const NO_MEMORY: &[u8] = b"CMS: Insufficient memory to create the key";
/// An increment that is not an integer.
const BAD_NUMBER: &[u8] = b"CMS: Cannot parse number";
/// One that is an integer and is below zero.
const NEGATIVE: &[u8] = b"CMS: Number cannot be negative";
/// An item whose counters have all reached the ceiling, which is written into
/// the reply array in place of that item's count.
const INCR_OVERFLOW: &[u8] = b"CMS: INCRBY overflow";
/// A key count that is not an integer.
const BAD_NUMKEYS: &[u8] = b"CMS: invalid numkeys";
/// One that is an integer and is not positive.
const NOT_POSITIVE: &[u8] = b"CMS: Number of keys must be positive";
/// A key count that does not match what came after it.
const WRONG_KEYS: &[u8] = b"CMS: wrong number of keys";
/// A `WEIGHTS` block that is not the same length as the key list.
const WRONG_WEIGHTS: &[u8] = b"CMS: wrong number of keys/weights";
/// A weight that is not an integer.
const BAD_WEIGHT: &[u8] = b"CMS: invalid weight value";
/// A source whose shape is not the destination's.
const NOT_EQUAL: &[u8] = b"CMS: width/depth is not equal";
/// A merge whose weighted sums do not fit in a counter.
const MERGE_OVERFLOW: &[u8] = b"CMS: MERGE overflow";

/// What a key holding anything else gets.
const WRONG_KIND: &str = "Operation against a key holding the wrong kind of value";

/// A sketch under a key.
#[derive(Debug)]
pub(super) struct CmsBody {
    /// The table. Everything `CMS.INFO` reports comes off it.
    c: Cms,
}

impl Foreign for CmsBody {
    fn type_name(&self) -> &'static str {
        // The module's own name for the type, hyphen and capitals included,
        // which is what a client sees from `TYPE` on a real server.
        "CMSk-TYPE"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.c.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // A sketch that has counted nothing is still a key, the same as an
        // empty filter on either of the other two families.
        false
    }
}

/// Run one count min sketch command.
///
/// Every command here reaches its key through the two helpers at the bottom of
/// the file, and both of them find the stripe the key they were given is on.
/// That is what `CMS.MERGE` needs, since it reads a run of sources and writes a
/// destination and the sources can be anywhere.
pub(super) fn execute(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "cms.initbydim" => initbydim(db, args, out),
        "cms.initbyprob" => initbyprob(db, args, out),
        "cms.incrby" => incrby(db, args, out),
        "cms.query" => query(db, args, out),
        "cms.merge" => merge(db, args, out),
        "cms.info" => info(db, args, out),
        other => unreachable!("{other} is not a count min sketch command"),
    }
}

/// `CMS.INITBYDIM key width depth`, which is the sketch stated rather than
/// derived.
fn initbydim(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    if db.hold(key).kind_of(key).is_some() {
        out.error(EXISTS);
        return Ok(());
    }
    let Some(width) = positive(args.get(2)) else {
        out.error(BAD_WIDTH);
        return Ok(());
    };
    let Some(depth) = positive(args.get(3)) else {
        out.error(BAD_DEPTH);
        return Ok(());
    };
    build(db, key, width, depth, out);
    Ok(())
}

/// `CMS.INITBYPROB key error probability`, which is the sketch asked for in the
/// terms the client actually has: how far off the count is allowed to be and how
/// often it is allowed to be that far off.
fn initbyprob(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    if db.hold(key).kind_of(key).is_some() {
        out.error(EXISTS);
        return Ok(());
    }
    let Some(error) = fraction(args.get(2)) else {
        out.error(BAD_ERROR);
        return Ok(());
    };
    let Some(prob) = fraction(args.get(3)) else {
        out.error(BAD_PROB);
        return Ok(());
    };
    let Some((width, depth)) = dims_from(error, prob) else {
        out.error(BAD_INIT);
        return Ok(());
    };
    build(db, key, width, depth, out);
    Ok(())
}

/// Make the sketch both constructors ended up asking for, or say why not.
fn build(db: &Db, key: &[u8], width: u64, depth: u64, out: &mut Out) {
    let Some(c) = Cms::new(width, depth) else {
        out.error(NO_MEMORY);
        return;
    };
    db.hold(key).put_foreign(key, Box::new(CmsBody { c }));
    out.ok();
}

/// `CMS.INCRBY key item increment [item increment ...]`.
///
/// Every pair is parsed before any of them is applied, so a command with a bad
/// number in the middle of it changes nothing at all. Once they are all good the
/// pairs go in left to right, which means a repeated item sees its own earlier
/// increment: `a 5 a 5` answers five and then ten.
fn incrby(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    if !args.len().is_multiple_of(2) {
        return Err(args::wrong_arity("cms.incrby"));
    }
    let mut stripe = db.hold(args.get(1));
    let body = match write(&mut stripe, args.get(1))? {
        Some(body) => body,
        None => {
            out.error(MISSING);
            return Ok(());
        }
    };
    for i in (3..args.len()).step_by(2) {
        match parse_i64(args.get(i)) {
            None => {
                out.error(BAD_NUMBER);
                return Ok(());
            }
            Some(n) if n < 0 => {
                out.error(NEGATIVE);
                return Ok(());
            }
            Some(_) => {}
        }
    }
    out.array((args.len() - 2) / 2);
    for i in (2..args.len()).step_by(2) {
        let by = parse_i64(args.get(i + 1)).expect("every pair was parsed above");
        let count = body.c.incr(args.get(i), by);
        // A counter that has reached the ceiling is an error where the count
        // would have been, and an item can reach it without this command being
        // the one that pushed it there.
        if count == u32::MAX {
            out.error(INCR_OVERFLOW);
        } else {
            out.uint(u64::from(count));
        }
    }
    Ok(())
}

/// `CMS.QUERY key item [item ...]`, which is a count per item and never an
/// error inside the array.
fn query(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut stripe = db.hold(args.get(1));
    let Some(body) = read(&mut stripe, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    out.array(args.len() - 2);
    for i in 2..args.len() {
        out.uint(u64::from(body.c.count_of(args.get(i))));
    }
    Ok(())
}

/// `CMS.MERGE dest numkeys source [source ...] [WEIGHTS weight [weight ...]]`.
///
/// The destination is overwritten and not added to, so the way to add is to name
/// the destination among its own sources, and `CMS.MERGE d 2 d s` is what a
/// client that means `d += s` has to send. Every source has to be exactly the
/// destination's shape, since a merge is counter by counter and two sketches of
/// different widths do not have the same counters.
fn merge(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let dest = args.get(1);
    // Read before the rest of the command is parsed, because a `CMS.MERGE`
    // naming a destination that is not there answers that whatever else is
    // wrong with it. The shape is read again below, once the stripes are held.
    {
        let mut stripe = db.hold(dest);
        if read(&mut stripe, dest)?.is_none() {
            out.error(MISSING);
            return Ok(());
        }
    }
    let Some(count) = parse_i64(args.get(2)) else {
        out.error(BAD_NUMKEYS);
        return Ok(());
    };
    if count <= 0 {
        out.error(NOT_POSITIVE);
        return Ok(());
    }
    // The key count decides where everything else is, so a count that does not
    // land on either the end of the command or the `WEIGHTS` keyword is the
    // only thing wrong with it however the rest reads.
    let Ok(count) = usize::try_from(count) else {
        out.error(WRONG_KEYS);
        return Ok(());
    };
    let Some(after) = 3usize.checked_add(count).filter(|&at| at <= args.len()) else {
        out.error(WRONG_KEYS);
        return Ok(());
    };
    let weights = match args.opt(after) {
        None => None,
        Some(word) if args::is(word, b"weights") => Some(after + 1),
        Some(_) => {
            out.error(WRONG_KEYS);
            return Ok(());
        }
    };
    if let Some(first) = weights {
        if args.len() - first != count {
            out.error(WRONG_WEIGHTS);
            return Ok(());
        }
        // The weights are read before the sources are, so a weight that is not
        // a number is answered before a source that is not there.
        for i in first..args.len() {
            if parse_i64(args.get(i)).is_none() {
                out.error(BAD_WEIGHT);
                return Ok(());
            }
        }
    }
    let weight_at = |n: usize| match weights {
        Some(first) => parse_i64(args.get(first + n)).expect("every weight was parsed above"),
        None => 1,
    };

    // Every stripe the command names, held across both passes and the write, so
    // that no source can be added to between being checked and being summed and
    // the destination cannot be written by anyone else in between either.
    let onto = db.stripe_of(dest);
    let mut held = db.hold_keys(std::iter::once(dest).chain((0..count).map(|i| args.get(3 + i))));
    let Some(body) = read(held.stripe_mut(onto), dest)? else {
        out.error(MISSING);
        return Ok(());
    };
    let (width, depth) = (body.c.width(), body.c.depth());

    // First pass: every source has to be there, be a sketch and be the right
    // shape, and they are checked in the order they were written, so the first
    // source that is wrong is the one reported.
    for i in 0..count {
        let key = args.get(3 + i);
        match read(held.stripe_mut(db.stripe_of(key)), key)? {
            None => {
                out.error(MISSING);
                return Ok(());
            }
            Some(src) if src.c.width() != width || src.c.depth() != depth => {
                out.error(NOT_EQUAL);
                return Ok(());
            }
            Some(_) => {}
        }
    }

    // Second pass: the sums, into an accumulator the destination never sees
    // unless all of them fit. That is what makes an overflow leave the
    // destination alone, and it is also what lets the destination be one of its
    // own sources without reading half merged counters.
    let mut acc = read(held.stripe_mut(onto), dest)?
        .expect("the destination is still there")
        .c
        .merge_start();
    for i in 0..count {
        let key = args.get(3 + i);
        let src =
            read(held.stripe_mut(db.stripe_of(key)), key)?.expect("checked in the first pass");
        if !src.c.merge_add(&mut acc, weight_at(i)) {
            out.error(MERGE_OVERFLOW);
            return Ok(());
        }
    }
    let body = write(held.stripe_mut(onto), dest)?.expect("the destination is still there");
    body.c.merge_finish(acc);
    out.ok();
    Ok(())
}

/// `CMS.INFO key`, which is the shape and the running total and takes no field.
fn info(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut stripe = db.hold(args.get(1));
    let Some(body) = read(&mut stripe, args.get(1))? else {
        out.error(MISSING);
        return Ok(());
    };
    out.map(3);
    out.simple(b"width");
    out.uint(body.c.width());
    out.simple(b"depth");
    out.uint(body.c.depth());
    out.simple(b"count");
    // The total is signed and has wrapped if enough has been added, so this is
    // the one number in the reply that can come back negative.
    out.int(body.c.count());
    Ok(())
}

/// An argument that has to be a positive count of something.
fn positive(arg: &[u8]) -> Option<u64> {
    parse_i64(arg).filter(|&n| n > 0).map(|n| n as u64)
}

/// One that has to be strictly between zero and one, which rules out a NaN
/// without saying so.
fn fraction(arg: &[u8]) -> Option<f64> {
    parse_f64(arg).filter(|&n| n > 0.0 && n < 1.0)
}

/// The sketch under `key` for writing, or `None` if the key is not there.
fn write<'k>(stripe: &'k mut Keyspace, key: &[u8]) -> Result<Option<&'k mut CmsBody>> {
    match stripe.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<CmsBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// The same, for reading.
fn read<'k>(stripe: &'k mut Keyspace, key: &[u8]) -> Result<Option<&'k CmsBody>> {
    match stripe.foreign(key)? {
        Some(body) => match body.downcast_ref::<CmsBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}
