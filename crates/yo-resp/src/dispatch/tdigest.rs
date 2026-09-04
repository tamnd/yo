//! `TDIGEST.*`, the quantile sketch RedisBloom put on the wire.
//!
//! The digest is [`yo_sketch::tdigest::TDigest`] and this is the wire in front
//! of it, the same split as `super::topk`. Fourteen commands: one to make a
//! digest, one to empty it, two to feed it, eight to ask it questions and one to
//! report its shape.
//!
//! # Errors
//!
//! Twenty two sentences, all of them carrying `ERR`, which is unlike the top k
//! and the count min families where the module wrote its own prefix instead. The
//! word in front of every one of them is `T-Digest:` with the hyphen, which is
//! not how the type is spelled anywhere else.
//!
//! # The order the checks happen in
//!
//! Every command except `TDIGEST.CREATE` and `TDIGEST.MERGE` checks the key
//! first and its arguments second, so a bad quantile against a key that is not
//! there says the key is not there. `TDIGEST.CREATE` checks the key first as
//! well but for the opposite reason: it is the one command the key is supposed
//! to be missing for. Arguments are parsed all the way through before any of
//! them is applied, so `TDIGEST.ADD key 1 nope` adds nothing at all, which is
//! the opposite of `TOPK.INCRBY`.
//!
//! # How a double is read
//!
//! Not the way the rest of this engine reads one. The module goes through
//! Redis's `string2d`, which refuses a NaN, refuses anything that overflowed to
//! an infinity while parsing, and refuses anything that underflowed to a zero,
//! while accepting the word `inf` and every case of it. So `1e400` is a parse
//! error and `inf` is a value, `1e-400` is a parse error and `1e-320` is a
//! value. Both of those pairs are one keystroke apart and a client that gets a
//! number back from one server and an error from the other has to find out why,
//! so the rule is copied here rather than left to the engine's own parser.
//!
//! # What the keyword search does
//!
//! `TDIGEST.CREATE key COMPRESSION n` looks for the word anywhere in the two
//! arguments after the key and then reads the compression from the second of
//! them whatever it found, so `TDIGEST.CREATE key 100 COMPRESSION` looks for a
//! number in the word `COMPRESSION` and complains that it is not one.
//! `TDIGEST.MERGE` does the same over everything after its input keys.
//!
//! # What a client can see that is different
//!
//! One thing, D-52: a digest whose node arrays would be over a gibibyte is
//! refused where the reference asks the allocator for them.

use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};
use yo_kv::{Db, Foreign, Keyspace};
use yo_sketch::tdigest::TDigest;

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// The compression a `TDIGEST.CREATE` with no compression asks for.
const DEFAULT_COMPRESSION: i64 = 100;

/// What `TDIGEST.CREATE` says about a key that is already a digest.
const EXISTS: &str = "T-Digest: key already exists";
/// What everything else says about a key that is not there.
const MISSING: &str = "T-Digest: key does not exist";
/// A compression that is not a whole number.
const BAD_COMPRESSION: &str = "T-Digest: error parsing compression parameter";
/// A compression that is a whole number and is not positive.
const COMPRESSION_RANGE: &str = "T-Digest: compression parameter needs to be a positive integer";
/// A word where `COMPRESSION` or, for a merge, `OVERRIDE` was expected.
const KEYWORD: &str = "T-Digest: wrong keyword";
/// A digest too large to build.
const NO_MEMORY: &str = "T-Digest: allocation failed";
/// The same, said differently by the merge.
const NO_MEMORY_DEST: &str = "T-Digest: allocation of destination digest failed";
/// A sample that is not a number.
const BAD_VAL: &str = "T-Digest: error parsing val parameter";
/// A sample that is a number and is an infinity.
const NOT_FINITE: &str = "T-Digest: val parameter needs to be a finite number";
/// A weight that no longer fits.
const OVERFLOW: &str = "T-Digest: overflow detected";
/// A numkeys that is not a whole number.
const BAD_NUMKEYS: &str = "T-Digest: error parsing numkeys";
/// A numkeys that is a whole number and is not positive.
const NUMKEYS_RANGE: &str = "T-Digest: numkeys needs to be a positive integer";
/// A quantile that is not a number.
const BAD_QUANTILE: &str = "T-Digest: error parsing quantile";
/// A quantile that is a number and is outside the unit interval.
const QUANTILE_RANGE: &str = "T-Digest: quantile should be in [0,1]";
/// A cutoff that is not a number.
const BAD_CDF: &str = "T-Digest: error parsing cdf";
/// A value that is not a number, which is the rank commands' wording for it.
const BAD_VALUE: &str = "T-Digest: error parsing value";
/// A rank that is not a whole number.
const BAD_RANK: &str = "T-Digest: error parsing rank";
/// A rank that is a whole number and is negative.
const RANK_NEGATIVE: &str = "T-Digest: rank needs to be non negative";
/// A low cut that is not a number.
const BAD_LOW: &str = "T-Digest: error parsing low_cut_percentile";
/// A high cut that is not a number.
const BAD_HIGH: &str = "T-Digest: error parsing high_cut_percentile";
/// A cut outside the unit interval.
const CUT_RANGE: &str = "T-Digest: low_cut_percentile and high_cut_percentile should be in [0,1]";
/// Two cuts in the wrong order, which includes the two of them being equal.
const CUT_ORDER: &str = "T-Digest: low_cut_percentile should be lower than high_cut_percentile";

/// What a key holding anything else gets.
const WRONG_KIND: &str = "Operation against a key holding the wrong kind of value";

/// A digest under a key.
#[derive(Debug)]
pub(super) struct TDigestBody {
    /// The digest. Everything `TDIGEST.INFO` reports comes off it.
    t: TDigest,
}

impl Foreign for TDigestBody {
    fn type_name(&self) -> &'static str {
        // The module's name for the type, which spells digest as `DIS`.
        "TDIS-TYPE"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.t.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // A digest holding no samples is still a key, the same as an empty
        // filter on any of the other families.
        false
    }
}

/// Run one t digest command.
///
/// Every command here reaches its key through the two helpers at the bottom of
/// the file, and both of them find the stripe the key they were given is on.
/// That is what `TDIGEST.MERGE` needs, since it reads a run of sources and
/// writes a destination and the sources can be anywhere.
pub(super) fn execute(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    // The merge is the one that reads from keys it was not given first, so it
    // goes before a stripe is chosen. Everything else names one digest and
    // names it first, so that stripe is found once and held for the whole
    // command.
    if spec.name == "tdigest.merge" {
        return merge(db, args, out);
    }
    let mut held = db.hold(args.get(1));
    let db = &mut *held;
    match spec.name {
        "tdigest.create" => create(db, args, out),
        "tdigest.reset" => reset(db, args, out),
        "tdigest.add" => add(db, args, out),
        "tdigest.min" => ends(db, args, out, false),
        "tdigest.max" => ends(db, args, out, true),
        "tdigest.quantile" => quantile(db, args, out),
        "tdigest.cdf" => cdf(db, args, out),
        "tdigest.trimmed_mean" => trimmed_mean(db, args, out),
        "tdigest.rank" => rank(db, args, out, false),
        "tdigest.revrank" => rank(db, args, out, true),
        "tdigest.byrank" => by_rank(db, args, out, false),
        "tdigest.byrevrank" => by_rank(db, args, out, true),
        "tdigest.info" => info(db, args, out),
        other => unreachable!("{other} is not a t digest command"),
    }
}

/// `TDIGEST.CREATE key [COMPRESSION compression]`.
///
/// Two words or four and nothing in between. The compression decides how many
/// centroids the digest is allowed, which is six times it plus ten, and a
/// hundred of them is accurate to about a percent in the middle of the
/// distribution and far better than that at the ends.
fn create(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 2 && args.len() != 4 {
        return Err(args::wrong_arity("tdigest.create"));
    }
    let key = args.get(1);
    if read(db, key)?.is_some() {
        return Err(Error::new(Code::Invalid, EXISTS));
    }
    let mut compression = DEFAULT_COMPRESSION;
    if args.len() == 4 {
        if !args::is(args.get(2), b"compression") && !args::is(args.get(3), b"compression") {
            return Err(Error::new(Code::Invalid, KEYWORD));
        }
        // The number is read from the last argument whether or not that is the
        // one after the keyword, which is the module's search and not a lookup.
        compression = size(args.get(3))?;
    }
    let Some(t) = TDigest::new(compression) else {
        return Err(Error::new(Code::Invalid, NO_MEMORY));
    };
    db.put_foreign(key, Box::new(TDigestBody { t }));
    out.ok();
    Ok(())
}

/// `TDIGEST.RESET key`, which empties the digest and keeps its shape.
fn reset(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = digest(db, args.get(1))?;
    body.t.reset();
    out.ok();
    Ok(())
}

/// `TDIGEST.ADD key value [value ...]`, one sample of weight one each.
///
/// Every value is parsed before any of them is added, so a command that is going
/// to fail changes nothing.
fn add(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let mut values = Vec::with_capacity(args.len() - 2);
    for i in 2..args.len() {
        let Some(v) = double(args.get(i)) else {
            return Err(Error::new(Code::Invalid, BAD_VAL));
        };
        if !v.is_finite() {
            return Err(Error::new(Code::Invalid, NOT_FINITE));
        }
        values.push(v);
    }
    for v in values {
        if body.t.add(v, 1).is_err() {
            return Err(Error::new(Code::Invalid, OVERFLOW));
        }
    }
    out.ok();
    Ok(())
}

/// `TDIGEST.MERGE destination numkeys source [source ...] [COMPRESSION c] [OVERRIDE]`.
///
/// The destination is built from scratch and then put in place, so a merge that
/// fails halfway leaves the old destination alone. Without `OVERRIDE` the
/// destination joins its own inputs, and a destination that is also named as a
/// source is therefore merged in twice, which is the module's behaviour and is
/// visible in the weights afterwards.
///
/// The compression of the result is the destination's if it exists, the largest
/// of the inputs' if it does not, and whatever `COMPRESSION` says if it is
/// there.
fn merge(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let dest = args.get(1);
    let dest_compression = read(&mut db.hold(dest), dest)?.map(|body| body.t.compression());
    let Some(numkeys) = parse_i64(args.get(2)) else {
        return Err(Error::new(Code::Invalid, BAD_NUMKEYS));
    };
    if numkeys <= 0 {
        return Err(Error::new(Code::Invalid, NUMKEYS_RANGE));
    }
    let sources = usize::try_from(numkeys).unwrap_or(usize::MAX);
    if sources > args.len() - 3 {
        return Err(args::wrong_arity("tdigest.merge"));
    }
    let rest = sources + 3;
    let mut compression = dest_compression;
    let mut override_dest = false;
    if rest < args.len() {
        let at = (rest..args.len()).find(|&i| args::is(args.get(i), b"compression"));
        if let Some(at) = at {
            if at + 1 >= args.len() {
                return Err(args::wrong_arity("tdigest.merge"));
            }
            compression = Some(size(args.get(at + 1))?);
        }
        let has_override = (rest..args.len()).any(|i| args::is(args.get(i), b"override"));
        if has_override {
            override_dest = true;
            if at.is_none() {
                // An override with no compression asked for goes back to the
                // largest of the inputs even when the destination exists.
                compression = None;
            }
        }
        if at.is_none() && !has_override {
            return Err(Error::new(Code::Invalid, KEYWORD));
        }
    }
    // Every input is checked before any of them is touched, so a merge naming a
    // key that is not there leaves the rest of them as they were.
    let mut largest = 0;
    for i in 3..rest {
        let name = args.get(i);
        let found = if name == dest {
            dest_compression
        } else {
            read(&mut db.hold(name), name)?.map(|body| body.t.compression())
        };
        let Some(c) = found else {
            return Err(Error::new(Code::Invalid, MISSING));
        };
        largest = largest.max(c);
    }
    let Some(mut into) = TDigest::new(compression.unwrap_or(largest)) else {
        return Err(Error::new(Code::Invalid, NO_MEMORY_DEST));
    };
    if !override_dest && dest_compression.is_some() {
        let from = compressed(&mut db.hold(dest), dest)?;
        fold(&mut into, &from)?;
    }
    for i in 3..rest {
        let key = args.get(i);
        let from = compressed(&mut db.hold(key), key)?;
        fold(&mut into, &from)?;
    }
    db.hold(dest)
        .put_foreign(dest, Box::new(TDigestBody { t: into }));
    out.ok();
    Ok(())
}

/// The centroids of the digest under `key`, swept first.
///
/// The sweep is a write to a key the merge only reads from, which is what the
/// reference does as well: merging out of a digest leaves that digest with its
/// buffer folded in and its compression count one higher.
fn compressed(db: &mut Keyspace, key: &[u8]) -> Result<Vec<(f64, i64)>> {
    let body = digest(db, key)?;
    if body.t.compress().is_err() {
        return Err(Error::new(Code::Invalid, OVERFLOW));
    }
    Ok(body.t.centroids())
}

/// Add every centroid of one input to the digest being built.
///
/// The sweep at the top is the one thing here that shows up in a reply. Each
/// input is folded in on its own and the digest is swept before each of them
/// rather than once at the end, so a merge of two digests of three samples each
/// leaves three merged nodes and three unmerged ones and a compression count of
/// one, and `TDIGEST.INFO` says so. Sweeping once at the end would say six
/// unmerged and none merged instead.
fn fold(into: &mut TDigest, from: &[(f64, i64)]) -> Result<()> {
    if into.compress().is_err() {
        return Err(Error::new(Code::Invalid, OVERFLOW));
    }
    for &(mean, weight) in from {
        if into.add(mean, weight).is_err() {
            return Err(Error::new(Code::Invalid, OVERFLOW));
        }
    }
    Ok(())
}

/// `TDIGEST.MIN key` and `TDIGEST.MAX key`, which are NaN on an empty digest.
fn ends(db: &mut Keyspace, args: Args<'_>, out: &mut Out, top: bool) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let value = if body.t.size() > 0 {
        if top { body.t.max() } else { body.t.min() }
    } else {
        f64::NAN
    };
    out.double(value);
    Ok(())
}

/// `TDIGEST.QUANTILE key quantile [quantile ...]`.
///
/// The walk over the centroids is carried from one quantile to the next while
/// they do not decrease, so the list is split into runs first and each run costs
/// one pass. A client that sends its quantiles in order pays for one pass in
/// total, and one that does not gets the same answers at more cost, which is why
/// the split is here rather than a rule about the arguments.
fn quantile(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let mut wanted = Vec::with_capacity(args.len() - 2);
    for i in 2..args.len() {
        let Some(q) = double(args.get(i)) else {
            return Err(Error::new(Code::Invalid, BAD_QUANTILE));
        };
        if q < 0.0 || q > 1.0 {
            return Err(Error::new(Code::Invalid, QUANTILE_RANGE));
        }
        wanted.push(q);
    }
    let mut values = vec![0.0; wanted.len()];
    let mut at = 0;
    while at < wanted.len() {
        let mut end = at;
        while end + 1 < wanted.len() && wanted[end] <= wanted[end + 1] {
            end += 1;
        }
        body.t.quantiles(&wanted[at..=end], &mut values[at..=end]);
        at = end + 1;
    }
    out.array(values.len());
    for v in values {
        out.double(v);
    }
    Ok(())
}

/// `TDIGEST.CDF key value [value ...]`, the fraction at or below each value.
fn cdf(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let mut wanted = Vec::with_capacity(args.len() - 2);
    for i in 2..args.len() {
        let Some(v) = double(args.get(i)) else {
            return Err(Error::new(Code::Invalid, BAD_CDF));
        };
        wanted.push(v);
    }
    out.array(wanted.len());
    for v in wanted {
        let answer = body.t.cdf(v);
        out.double(answer);
    }
    Ok(())
}

/// `TDIGEST.TRIMMED_MEAN key low high`, the mean of what is left after both
/// tails are cut.
fn trimmed_mean(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let Some(low) = double(args.get(2)) else {
        return Err(Error::new(Code::Invalid, BAD_LOW));
    };
    let Some(high) = double(args.get(3)) else {
        return Err(Error::new(Code::Invalid, BAD_HIGH));
    };
    if low < 0.0 || low > 1.0 || high < 0.0 || high > 1.0 {
        return Err(Error::new(Code::Invalid, CUT_RANGE));
    }
    if low >= high {
        return Err(Error::new(Code::Invalid, CUT_ORDER));
    }
    let value = body.t.trimmed_mean(low, high);
    out.double(value);
    Ok(())
}

/// `TDIGEST.RANK key value [value ...]` and its reverse.
///
/// The rank of a value is how many samples are below it plus half of those that
/// are equal to it, which is the cdf read back as a count. Under the smallest
/// sample the answer is minus one and over the largest it is the number of
/// samples, and the two swap over for the reverse. An empty digest answers minus
/// two to everything.
fn rank(db: &mut Keyspace, args: Args<'_>, out: &mut Out, reverse: bool) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let mut wanted = Vec::with_capacity(args.len() - 2);
    for i in 2..args.len() {
        let Some(v) = double(args.get(i)) else {
            return Err(Error::new(Code::Invalid, BAD_VALUE));
        };
        wanted.push(v);
    }
    #[allow(clippy::cast_precision_loss)]
    let size = body.t.size() as f64;
    let min = body.t.min();
    let max = body.t.max();
    out.array(wanted.len());
    for v in wanted {
        let answer = if size == 0.0 {
            -2.0
        } else if v < min {
            if reverse { size } else { -1.0 }
        } else if v > max {
            if reverse { -1.0 } else { size }
        } else {
            let at = body.t.cdf(v) * size;
            // The two directions round differently, which is the module's doing
            // and not a symmetry it forgot: forward rounds a half down and
            // reverse rounds a half away from zero.
            let at = if reverse { at.round() } else { half_down(at) };
            if reverse { (size - at).round() } else { at }
        };
        #[allow(clippy::cast_possible_truncation)]
        out.int(answer as i64);
    }
    Ok(())
}

/// `TDIGEST.BYRANK key rank [rank ...]` and its reverse, which are the other way
/// round: a rank in and a value out.
///
/// A rank at or past the number of samples answers an infinity, positive going
/// forwards and negative going backwards, and every rank on an empty digest
/// answers NaN.
fn by_rank(db: &mut Keyspace, args: Args<'_>, out: &mut Out, reverse: bool) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let mut wanted = Vec::with_capacity(args.len() - 2);
    for i in 2..args.len() {
        let Some(rank) = parse_i64(args.get(i)) else {
            return Err(Error::new(Code::Invalid, BAD_RANK));
        };
        if rank < 0 {
            return Err(Error::new(Code::Invalid, RANK_NEGATIVE));
        }
        wanted.push(rank);
    }
    #[allow(clippy::cast_precision_loss)]
    let size = body.t.size() as f64;
    out.array(wanted.len());
    for rank in wanted {
        #[allow(clippy::cast_precision_loss)]
        let rank = rank as f64;
        let answer = if size == 0.0 {
            f64::NAN
        } else if rank == 0.0 {
            if reverse { body.t.max() } else { body.t.min() }
        } else if rank >= size {
            if reverse {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else {
            let at = if reverse { size - rank - 1.0 } else { rank };
            body.t.quantile(at / size)
        };
        out.double(answer);
    }
    Ok(())
}

/// `TDIGEST.INFO key`, which is the nine numbers the digest keeps about itself.
fn info(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = digest(db, args.get(1))?;
    let t = &body.t;
    out.map(9);
    out.simple(b"Compression");
    out.int(t.compression());
    out.simple(b"Capacity");
    out.uint(t.capacity() as u64);
    out.simple(b"Merged nodes");
    out.uint(t.merged_nodes() as u64);
    out.simple(b"Unmerged nodes");
    out.uint(t.unmerged_nodes() as u64);
    out.simple(b"Merged weight");
    out.int(t.merged_weight());
    out.simple(b"Unmerged weight");
    out.int(t.unmerged_weight());
    out.simple(b"Observations");
    out.int(t.size());
    out.simple(b"Total compressions");
    out.int(t.compressions());
    out.simple(b"Memory usage");
    out.uint(t.reported_bytes());
    Ok(())
}

/// Round to a whole number, sending a half towards zero.
///
/// C's `round` sends a half away from zero and there is no library function for
/// the other direction, so the module wrote one out of `modf`, and the forward
/// rank is the only caller it has.
fn half_down(f: f64) -> f64 {
    let whole = f.trunc();
    let frac = f - whole;
    if frac.abs() <= 0.5 {
        return whole;
    }
    if whole >= 0.0 {
        whole + 1.0
    } else {
        whole - 1.0
    }
}

/// A double the way the module reads one, which the module doc explains.
fn double(arg: &[u8]) -> Option<f64> {
    let v = parse_f64(arg)?;
    if v.is_finite() {
        // A zero the digits did not ask for is an underflow and is refused.
        if v == 0.0 && !is_zero(arg) {
            return None;
        }
        return Some(v);
    }
    // An infinity is a value if that is what the client wrote and an overflow
    // otherwise.
    if is_infinity(arg) { Some(v) } else { None }
}

/// Does this spell an infinity, in any case and with a sign or without one?
fn is_infinity(arg: &[u8]) -> bool {
    let body = match arg.first() {
        Some(b'+' | b'-') => &arg[1..],
        _ => arg,
    };
    body.eq_ignore_ascii_case(b"inf") || body.eq_ignore_ascii_case(b"infinity")
}

/// Does this spell a zero, meaning every digit in it is one?
fn is_zero(arg: &[u8]) -> bool {
    let mut digits = arg.iter().take_while(|&&c| c != b'e' && c != b'E');
    digits.all(|&c| !c.is_ascii_digit() || c == b'0')
}

/// An argument that has to be a whole number, which for a compression also has
/// to be positive.
fn size(arg: &[u8]) -> Result<i64> {
    let Some(n) = parse_i64(arg) else {
        return Err(Error::new(Code::Invalid, BAD_COMPRESSION));
    };
    if n <= 0 {
        return Err(Error::new(Code::Invalid, COMPRESSION_RANGE));
    }
    Ok(n)
}

/// The digest under `key`, or the error the family answers when it is not there.
///
/// Every command takes it mutably, the reading ones included, because a question
/// asked of a digest sweeps its buffer in before it answers.
fn digest<'k>(db: &'k mut Keyspace, key: &[u8]) -> Result<&'k mut TDigestBody> {
    match write(db, key)? {
        Some(body) => Ok(body),
        None => Err(Error::new(Code::Invalid, MISSING)),
    }
}

/// The digest under `key` for writing, or `None` if the key is not there.
fn write<'k>(db: &'k mut Keyspace, key: &[u8]) -> Result<Option<&'k mut TDigestBody>> {
    match db.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<TDigestBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// The same, for the two places that only want to know what is there.
fn read<'k>(db: &'k mut Keyspace, key: &[u8]) -> Result<Option<&'k TDigestBody>> {
    match db.foreign(key)? {
        Some(body) => match body.downcast_ref::<TDigestBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}
