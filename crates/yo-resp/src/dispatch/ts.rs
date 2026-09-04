//! `TS.*`, the time series family RedisTimeSeries put on the wire.
//!
//! The series itself is in `yo-series` and this is the wire in front of it, the
//! same split as `super::cms` and the other module families. Nine commands
//! here: the key type, the two that shape a series, the four that write samples
//! into it, the one that takes the newest sample back out and the one that
//! reports on it. Reading a span, the label index and compaction rules come
//! next.
//!
//! # Errors
//!
//! Nearly all of them are one sentence starting `TSDB:` behind Redis's own
//! `ERR`, and they are copied word for word. Two are written straight to the
//! client with no prefix at all, a negative `RETENTION` and a `TS.INCRBY` whose
//! timestamp goes backwards, which is not a style anyone chose but is what a
//! client sees and so is what this writes.
//!
//! # The options are a scan and not a grammar
//!
//! Every option word is looked for across the whole command rather than read in
//! order, so a key called `RETENTION` is the `RETENTION` keyword, a word that is
//! not an option at all is ignored, and `LABELS` swallows everything after it in
//! pairs while the other keywords still find themselves inside what it
//! swallowed. All of that is visible from a client, so all of it is copied. A
//! client that writes its options in the documented order never sees any of it.
//!
//! # The order the checks happen in
//!
//! `TS.CREATE` and `TS.ALTER` read their options before they look at the key, so
//! a bad retention against a key that is already there answers about the
//! retention. `TS.ADD` is the other way round for everything except the value
//! and the timestamp, which it reads first. `TS.MADD` never creates a key and
//! never reads an option, so a missing key inside it is the key is not a TSDB
//! key rather than anything about creating one.
//!
//! # Where the two protocols disagree
//!
//! A sample value is a simple string of the shortest digits that read back as
//! the same number on RESP2 and a RESP3 double on RESP3, which are two
//! renderings of one number: `1E-1` against `0.1`. `TS.INFO` is a flat array of
//! twenty eight on RESP2 and a map of fourteen on RESP3, and its labels follow
//! the same split one level down. The value half of the ignore window inside
//! `TS.INFO` is a plain double rather than the shortest digits, so half a degree
//! reads `0.5` there and `5E-1` out of `TS.GET`. That is two reply helpers
//! inside one module rather than a decision, and both are copied.
//!
//! # One thing the reference does not decide
//!
//! `TS.INCRBY key n TIMESTAMP` with nothing behind the keyword reads one past
//! the end of its own argument list on a real server, so what it answers depends
//! on what was in that memory: a fresh key gets `invalid timestamp`, a key
//! holding samples has been seen to get the backwards error and to get the
//! increment itself used as the timestamp. There is no behaviour there to copy,
//! so this answers `invalid timestamp` every time, which is what the reference
//! answers in the one case where it is reading memory it owns.
//!
//! # What a client can see that is different
//!
//! `TS.INFO` reports a memory usage of its own, which is D-53. It has to: the
//! number there is the module's own allocator arithmetic over a chunk layout
//! that is not this one, and an empty series here does not hold the four
//! kibibytes an empty series holds there. Everything else matches a real Redis
//! 8.10.1 with RedisTimeSeries in it, sample for sample and error for error.

use yo_common::num::{DOUBLE_MAX, parse_f64, parse_i64, write_dragonbox};
use yo_common::{Code, Error, Result};
use yo_kv::{Foreign, Keyspace};
use yo_series::{Encoding, Policy, Refused, Sample, Series};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What `TS.CREATE` says about a key that is already there, whatever it holds.
/// The existence is what is checked and not the type, so a key holding a string
/// gets this rather than `WRONGTYPE`.
const EXISTS: &[u8] = b"TSDB: key already exists";
/// What the commands that will not create one say about a key that is not
/// there.
const MISSING: &[u8] = b"TSDB: the key does not exist";
/// What the two that would have created one say about a key holding something
/// else. Every other command in the family answers `WRONGTYPE` for the same
/// key.
const NOT_A_SERIES: &[u8] = b"TSDB: the key is not a TSDB key";
/// A label with an empty name, an empty value, or a value holding one of the
/// three characters a filter expression is written with.
const BAD_LABELS: &[u8] = b"TSDB: Couldn't parse LABELS";
/// A retention that is missing or is not a whole number.
const BAD_RETENTION: &[u8] = b"TSDB: Couldn't parse RETENTION";
/// A chunk size that is missing or is not a whole number.
const BAD_CHUNK: &[u8] = b"TSDB: Couldn't parse CHUNK_SIZE";
/// One that is a number and is not a size a chunk may be.
const CHUNK_RANGE: &[u8] =
    b"TSDB: CHUNK_SIZE value must be a multiple of 8 in the range [48 .. 1048576]";
/// An encoding that is neither of the two.
const BAD_ENCODING: &[u8] = b"TSDB: unknown ENCODING parameter";
/// A duplicate policy keyword with nothing behind it.
const BAD_POLICY: &[u8] = b"TSDB: Couldn't parse DUPLICATE_POLICY";
/// One with a word behind it that is not a policy.
const UNKNOWN_POLICY: &[u8] = b"TSDB: Unknown DUPLICATE_POLICY";
/// An ignore window missing a half, or with a half that is not a number.
const BAD_IGNORE: &[u8] = b"TSDB: Couldn't parse IGNORE";
/// One whose halves are numbers and are below zero.
const NEGATIVE_IGNORE: &[u8] = b"TSDB: IGNORE arguments cannot be negative";
/// A timestamp that is not a whole number.
const BAD_TIMESTAMP: &[u8] = b"TSDB: invalid timestamp";
/// One that is a whole number and is below zero.
const NEGATIVE_TIMESTAMP: &[u8] = b"TSDB: invalid timestamp, must be a nonnegative integer";
/// A value that is not one the module's own reader accepts.
const BAD_VALUE: &[u8] = b"TSDB: invalid value";
/// An increment that is not a number.
const BAD_INCREMENT: &[u8] = b"TSDB: invalid increase/decrease value";
/// An increment onto a series whose newest value is not a number, which has no
/// answer to give.
const NAN_INCREMENT: &[u8] = b"TSDB: cannot increment/decrement NaN value";
/// A sample so far behind the newest one that retention has already gone past
/// where it would have landed.
const TOO_OLD: &[u8] = b"TSDB: Timestamp is older than retention";
/// A sample on a timestamp that is taken, under a policy that will not have it
/// replaced. One sentence for two rather different cases, which is the
/// module's.
const UPSERT: &[u8] = b"TSDB: Error at upsert, update is not supported when DUPLICATE_POLICY is set to BLOCK mode, or either current or new value is NaN and DUPLICATE_POLICY is MAX/MIN/SUM";
/// A `TS.DEL` whose first timestamp is not one.
const BAD_FROM: &[u8] = b"TSDB: wrong fromTimestamp";
/// One whose second is not.
const BAD_TO: &[u8] = b"TSDB: wrong toTimestamp";
/// A retention below zero, which the module writes with no prefix at all where
/// a retention that is not a number gets one.
const BARE_RETENTION: &[u8] = b"TSDB: Couldn't parse RETENTION";
/// A `TS.INCRBY` or `TS.DECRBY` whose timestamp is behind the newest sample,
/// which is the other one the module writes bare.
const BARE_BACKWARDS: &[u8] =
    b"TSDB: timestamp must be equal to or higher than the maximum existing timestamp";

/// What a key holding anything else gets from the seven commands that will not
/// create a series.
///
/// The word is inside the sentence rather than in front of it, so a client sees
/// `ERR WRONGTYPE ...` and not the bare `WRONGTYPE ...` every other command in
/// the server answers. That is the module writing its own error text and Redis
/// putting its own prefix on anything a module writes, and it is what a real
/// server sends, so it is what this sends.
const WRONG_KIND: &str = "WRONGTYPE Operation against a key holding the wrong kind of value";

/// The smallest a chunk may be.
const CHUNK_MIN: i64 = 48;
/// The largest.
const CHUNK_MAX: i64 = 1_048_576;

/// A series under a key.
#[derive(Debug)]
pub(super) struct TsBody {
    /// The samples and the settings. Everything `TS.INFO` reports comes off it.
    s: Series,
}

impl Foreign for TsBody {
    fn type_name(&self) -> &'static str {
        // The module's own name for the type, hyphen and capitals included,
        // which is what a client sees from `TYPE` on a real server.
        "TSDB-TYPE"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.s.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // A series with no samples in it is still a key. A client is expected to
        // create one and then write to it, and it would be a surprise to find it
        // gone in between.
        false
    }
}

pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "ts.create" => create(db, &args, out),
        "ts.alter" => alter(db, &args, out),
        "ts.add" => add(db, &args, out),
        "ts.madd" => madd(db, &args, out),
        "ts.incrby" => incr(db, &args, out, true),
        "ts.decrby" => incr(db, &args, out, false),
        "ts.del" => del(db, &args, out),
        "ts.get" => get(db, &args, out),
        "ts.info" => info(db, &args, out),
        other => unreachable!("{other} is not a time series command"),
    }
}

/// `TS.CREATE key [RETENTION n] [ENCODING e] [CHUNK_SIZE n] [DUPLICATE_POLICY p]
/// [IGNORE t v] [LABELS name value ...]`.
fn create(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    let opts = match options(args) {
        Ok(opts) => opts,
        Err(bad) => return said(bad, "ts.create", out),
    };
    let key = args.get(1);
    if db.kind_of(key).is_some() {
        return say(out, EXISTS);
    }
    let mut s = Series::new();
    apply(&mut s, opts);
    db.put_foreign(key, Box::new(TsBody { s }));
    out.ok();
    Ok(())
}

/// `TS.ALTER key [RETENTION n] [CHUNK_SIZE n] [DUPLICATE_POLICY p] [IGNORE t v]
/// [LABELS name value ...]`.
///
/// An encoding is read and then thrown away, because a series that already holds
/// samples cannot be told to store them a different way and the module does not
/// try. Everything that was not named is left as it was.
fn alter(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    let mut opts = match options(args) {
        Ok(opts) => opts,
        Err(bad) => return said(bad, "ts.alter", out),
    };
    opts.encoding = None;
    let Some(body) = write(db, args.get(1))? else {
        return say(out, MISSING);
    };
    apply(&mut body.s, opts);
    out.ok();
    Ok(())
}

/// `TS.ADD key timestamp value [option ...]`, which makes the series if it is
/// not there and reads the same options `TS.CREATE` does when it does.
///
/// On a series that is already there the only option that means anything is
/// `ON_DUPLICATE`, and the rest are read past. That is why a `DUPLICATE_POLICY`
/// on a `TS.ADD` against a key that exists does nothing at all.
fn add(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let Some(value) = number(args.get(3)) else {
        return say(out, BAD_VALUE);
    };
    let at = match moment(db, args.get(2)) {
        Ok(at) => at,
        Err(msg) => return say(out, msg),
    };
    let over = if db.kind_of(key).is_none() {
        let opts = match options(args) {
            Ok(opts) => opts,
            Err(bad) => return said(bad, "ts.add", out),
        };
        let mut s = Series::new();
        apply(&mut s, opts);
        db.put_foreign(key, Box::new(TsBody { s }));
        None
    } else {
        if write(db, key).is_err() {
            return say(out, NOT_A_SERIES);
        }
        match find(args, b"ON_DUPLICATE") {
            None => None,
            Some(at) => match policy_at(args, at) {
                Ok(policy) => Some(policy),
                Err(bad) => return said(bad, "ts.add", out),
            },
        }
    };
    let body = write(db, key)?.expect("the series is there by now");
    store(body, at, value, over, out);
    Ok(())
}

/// `TS.MADD key timestamp value [key timestamp value ...]`.
///
/// Every triple is answered in its own slot and a bad one does not stop the ones
/// after it, so this is the only command in the family whose reply can hold both
/// timestamps and errors. It creates nothing: a key that is not already a series
/// is an error in its slot.
fn madd(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 4 || !(args.len() - 1).is_multiple_of(3) {
        return Err(args::wrong_arity("ts.madd"));
    }
    out.array((args.len() - 1) / 3);
    for i in (1..args.len()).step_by(3) {
        let Some(value) = number(args.get(i + 2)) else {
            out.error_line(b"ERR ", BAD_VALUE);
            continue;
        };
        let at = match moment(db, args.get(i + 1)) {
            Ok(at) => at,
            Err(msg) => {
                out.error_line(b"ERR ", msg);
                continue;
            }
        };
        match write(db, args.get(i)) {
            Ok(Some(body)) => store(body, at, value, None, out),
            Ok(None) | Err(_) => out.error_line(b"ERR ", NOT_A_SERIES),
        }
    }
    Ok(())
}

/// `TS.INCRBY key n [TIMESTAMP t] [option ...]` and `TS.DECRBY`, which are one
/// command with the sign flipped.
///
/// This is an add whose value is the newest value plus or minus the number
/// given, so it only ever writes at or past the newest sample and a timestamp
/// behind that is an error rather than a backfill. The sample goes in under the
/// last policy whatever the series says, which is what makes two of these on one
/// timestamp add up rather than collide.
fn incr(db: &mut Keyspace, args: &Args<'_>, out: &mut Out, up: bool) -> Result<()> {
    let key = args.get(1);
    let name = if up { "ts.incrby" } else { "ts.decrby" };
    let exists = db.kind_of(key).is_some();
    if exists {
        // A key holding something else is `WRONGTYPE` here rather than the
        // sentence `TS.ADD` gives, and it is answered before the number is even
        // looked at.
        write(db, key)?;
    }
    let Some(by) = parse_f64(args.get(2)).filter(|n| !n.is_nan()) else {
        return say(out, BAD_INCREMENT);
    };

    // `LABELS` swallows the rest of the command, so a `TIMESTAMP` behind one is
    // a label name and not the keyword. Both are looked for past the key and the
    // number so that a key called `TIMESTAMP` stays a key.
    let labels_at = find_from(args, 3, b"LABELS");
    let stamp_at =
        find_from(args, 3, b"TIMESTAMP").filter(|&at| labels_at.is_none_or(|labels| at < labels));
    let at = match stamp_at {
        None => now(db),
        Some(at) => match args.opt(at + 1) {
            None => return say(out, BAD_TIMESTAMP),
            Some(b"*") => now(db),
            Some(word) => match parse_i64(word) {
                Some(at) => at,
                None => return say(out, BAD_TIMESTAMP),
            },
        },
    };

    if !exists {
        let opts = match options(args) {
            Ok(opts) => opts,
            Err(bad) => return said(bad, name, out),
        };
        let mut s = Series::new();
        apply(&mut s, opts);
        db.put_foreign(key, Box::new(TsBody { s }));
    }
    let body = write(db, key)?.expect("the series is there by now");
    let last = body.s.last_sample();
    if last.is_some_and(|s| at < s.at) {
        out.error(BARE_BACKWARDS);
        return Ok(());
    }
    let base = last.map_or(0.0, |s| s.value);
    if base.is_nan() {
        return say(out, NAN_INCREMENT);
    }
    let value = if up { base + by } else { base - by };
    store(body, at, value, Some(Policy::Last), out);
    Ok(())
}

/// `TS.DEL key from to`, both ends included, which answers how many samples
/// went.
fn del(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    let Some(from) = span(args.get(2), b"-", 0) else {
        return say(out, BAD_FROM);
    };
    let Some(to) = span(args.get(3), b"+", i64::MAX) else {
        return say(out, BAD_TO);
    };
    let Some(body) = write(db, args.get(1))? else {
        return say(out, MISSING);
    };
    out.uint(body.s.delete(from, to) as u64);
    Ok(())
}

/// `TS.GET key`, which is the newest sample, or an empty array when there is not
/// one.
fn get(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return say(out, MISSING);
    };
    match body.s.last_sample() {
        None => out.array(0),
        Some(sample) => {
            out.array(2);
            out.int(sample.at);
            value(out, sample.value);
        }
    }
    Ok(())
}

/// `TS.INFO key`, which is fourteen fields about the series and takes no field
/// name.
fn info(db: &mut Keyspace, args: &Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return say(out, MISSING);
    };
    let s = &body.s;
    let (ignore_time, ignore_value) = s.ignore();
    out.map(14);
    out.simple(b"totalSamples");
    out.uint(s.len() as u64);
    out.simple(b"memoryUsage");
    out.uint(s.memory_bytes() as u64);
    // A series with nothing in it reports zero for both ends rather than saying
    // it has no ends, and so does one everything has been deleted from.
    out.simple(b"firstTimestamp");
    out.int(s.first().unwrap_or(0));
    out.simple(b"lastTimestamp");
    out.int(s.last().unwrap_or(0));
    out.simple(b"retentionTime");
    out.int(s.retention());
    out.simple(b"chunkCount");
    out.uint(s.chunk_count() as u64);
    out.simple(b"chunkSize");
    out.uint(s.chunk_bytes() as u64);
    out.simple(b"chunkType");
    out.simple(s.encoding().name().as_bytes());
    // Never a nil: a series that has not been told what to do about a repeated
    // timestamp reports the default rather than nothing.
    out.simple(b"duplicatePolicy");
    out.simple(s.policy().unwrap_or(Policy::Block).name().as_bytes());
    out.simple(b"labels");
    if out.proto().is_resp3() {
        out.map(s.labels().len());
        for (name, value) in s.labels() {
            out.bulk(name);
            out.bulk(value);
        }
    } else {
        out.array(s.labels().len());
        for (name, value) in s.labels() {
            out.array(2);
            out.bulk(name);
            out.bulk(value);
        }
    }
    // Both of these wait on compaction rules, which are the next piece of the
    // family to land.
    out.simple(b"sourceKey");
    out.nil();
    out.simple(b"rules");
    out.map(0);
    out.simple(b"ignoreMaxTimeDiff");
    out.int(ignore_time);
    out.simple(b"ignoreMaxValDiff");
    // A plain double and not the shortest digits a sample value gets, so half a
    // degree is 0.5 here and 5E-1 out of `TS.GET`. That is two different reply
    // helpers inside one module rather than a decision, and both are copied.
    out.double(ignore_value);
    Ok(())
}

/// Puts a sample in and writes the reply, which is where all four of the
/// commands that write samples end up.
///
/// The timestamp that comes back is not always the one that went in. A sample
/// close enough to the newest one to be uninteresting is dropped and the newest
/// timestamp is answered instead, which is how a client tells the two apart.
fn store(body: &mut TsBody, at: i64, value: f64, over: Option<Policy>, out: &mut Out) {
    match body.s.add(Sample::new(at, value), over) {
        Ok(when) => out.int(when),
        Err(Refused::Old) => out.error_line(b"ERR ", TOO_OLD),
        Err(Refused::Duplicate) => out.error_line(b"ERR ", UPSERT),
    }
}

/// Writes one of the sentences above with the `ERR` the module's own helper puts
/// in front of it.
fn say(out: &mut Out, msg: &[u8]) -> Result<()> {
    out.error_line(b"ERR ", msg);
    Ok(())
}

/// What the option words on a command said, with `None` for the ones that were
/// not there at all. That last part is what `TS.ALTER` needs: it changes what
/// was named and nothing else.
#[derive(Debug, Default)]
struct Options {
    /// How far back to keep samples.
    retention: Option<i64>,
    /// How much room to give a chunk.
    chunk_bytes: Option<usize>,
    /// How to store them.
    encoding: Option<Encoding>,
    /// What to do about a repeated timestamp.
    policy: Option<Policy>,
    /// The name and value pairs the series can be found by.
    labels: Option<Vec<(Vec<u8>, Vec<u8>)>>,
    /// How close to the newest sample a reading has to be to be dropped.
    ignore: Option<(i64, f64)>,
}

/// What went wrong reading the option words.
#[derive(Debug)]
enum Bad {
    /// A sentence the module writes through its own helper, which puts `ERR` in
    /// front of it.
    Said(&'static [u8]),
    /// One it writes straight to the client with nothing in front of it.
    Bare(&'static [u8]),
    /// The arity reply, which is what `ENCODING` with nothing behind it gets
    /// where every other keyword in the same spot gets a sentence.
    Arity,
}

/// Writes whatever `bad` says, in the shape the module says it in.
fn said(bad: Bad, name: &'static str, out: &mut Out) -> Result<()> {
    match bad {
        Bad::Said(msg) => say(out, msg),
        Bad::Bare(msg) => {
            out.error(msg);
            Ok(())
        }
        Bad::Arity => Err(args::wrong_arity(name)),
    }
}

/// Reads every option word out of a command, in the order the module reads them,
/// which is the order their errors come out in.
fn options(args: &Args<'_>) -> core::result::Result<Options, Bad> {
    let mut opts = Options::default();

    if let Some(at) = find(args, b"LABELS") {
        let first = at + 1;
        let mut labels = Vec::new();
        // Everything to the end, in pairs, with a trailing odd word dropped.
        for i in 0..args.len().saturating_sub(first) / 2 {
            let name = args.get(first + i * 2);
            let value = args.get(first + i * 2 + 1);
            // A label value has to be usable inside a filter expression, and the
            // three characters a filter is written with would make it
            // unreadable, so they are refused where the label is set instead.
            let ok = !name.is_empty()
                && !value.is_empty()
                && !value.iter().any(|b| matches!(b, b'(' | b')' | b','));
            if !ok {
                return Err(Bad::Said(BAD_LABELS));
            }
            labels.push((name.to_vec(), value.to_vec()));
        }
        opts.labels = Some(labels);
    }

    if let Some(at) = find(args, b"RETENTION") {
        let Some(n) = args.opt(at + 1).and_then(parse_i64) else {
            return Err(Bad::Said(BAD_RETENTION));
        };
        if n < 0 {
            return Err(Bad::Bare(BARE_RETENTION));
        }
        opts.retention = Some(n);
    }

    if let Some(at) = find(args, b"CHUNK_SIZE") {
        let Some(n) = args.opt(at + 1).and_then(parse_i64) else {
            return Err(Bad::Said(BAD_CHUNK));
        };
        // The range check runs first, so by the time the size is looked at as a
        // count of bytes it is known to be positive.
        if !(CHUNK_MIN..=CHUNK_MAX).contains(&n) || !(n as usize).is_multiple_of(8) {
            return Err(Bad::Said(CHUNK_RANGE));
        }
        opts.chunk_bytes = Some(n as usize);
    }

    if let Some(at) = find(args, b"ENCODING") {
        let Some(word) = args.opt(at + 1) else {
            return Err(Bad::Arity);
        };
        opts.encoding = Some(if args::is(word, b"uncompressed") {
            Encoding::Uncompressed
        } else if args::is(word, b"compressed") {
            Encoding::Compressed
        } else {
            return Err(Bad::Said(BAD_ENCODING));
        });
    }

    if let Some(at) = find(args, b"DUPLICATE_POLICY") {
        opts.policy = Some(policy_at(args, at)?);
    }

    if let Some(at) = find(args, b"IGNORE") {
        let time = args.opt(at + 1).and_then(parse_i64);
        let value = args.opt(at + 2).and_then(parse_f64);
        let (Some(time), Some(value)) = (time, value) else {
            return Err(Bad::Said(BAD_IGNORE));
        };
        if time < 0 || value < 0.0 {
            return Err(Bad::Said(NEGATIVE_IGNORE));
        }
        opts.ignore = Some((time, value));
    }

    Ok(opts)
}

/// The policy named just after `at`.
fn policy_at(args: &Args<'_>, at: usize) -> core::result::Result<Policy, Bad> {
    let Some(word) = args.opt(at + 1) else {
        return Err(Bad::Said(BAD_POLICY));
    };
    Policy::parse(word).ok_or(Bad::Said(UNKNOWN_POLICY))
}

/// Puts whatever was named onto a series and leaves the rest alone.
fn apply(s: &mut Series, opts: Options) {
    if let Some(n) = opts.retention {
        s.set_retention(n);
    }
    if let Some(n) = opts.chunk_bytes {
        s.set_chunk_bytes(n);
    }
    if let Some(encoding) = opts.encoding {
        s.set_encoding(encoding);
    }
    if let Some(policy) = opts.policy {
        s.set_policy(policy);
    }
    if let Some(labels) = opts.labels {
        s.set_labels(labels);
    }
    if let Some((time, value)) = opts.ignore {
        s.set_ignore(time, value);
    }
}

/// Where `word` first appears in the command, past the command name.
///
/// The module looks for each keyword across the whole argument list rather than
/// walking it once, which is why the key itself can be read as a keyword. See
/// the note at the top of the file.
fn find(args: &Args<'_>, word: &[u8]) -> Option<usize> {
    find_from(args, 1, word)
}

/// The same, starting at `from`, which is what the two increments need so that a
/// key called `TIMESTAMP` stays a key.
fn find_from(args: &Args<'_>, from: usize, word: &[u8]) -> Option<usize> {
    (from..args.len()).find(|&i| args::is(args.get(i), word))
}

/// One end of a `TS.DEL` span, which is a timestamp that is not below zero or
/// the one character standing for as far as that end goes.
fn span(arg: &[u8], open: &[u8], edge: i64) -> Option<i64> {
    if arg == open {
        return Some(edge);
    }
    parse_i64(arg).filter(|&n| n >= 0)
}

/// A timestamp argument, which is a whole number of milliseconds or a star for
/// whenever the server thinks it is now.
fn moment(db: &Keyspace, arg: &[u8]) -> core::result::Result<i64, &'static [u8]> {
    if arg == b"*" {
        return Ok(now(db));
    }
    let Some(at) = parse_i64(arg) else {
        return Err(BAD_TIMESTAMP);
    };
    if at < 0 {
        return Err(NEGATIVE_TIMESTAMP);
    }
    Ok(at)
}

/// The server's clock, as a timestamp.
fn now(db: &Keyspace) -> i64 {
    db.clock().now_ms() as i64
}

/// A sample value, in the grammar the module's own reader accepts.
///
/// That is an optional minus, at least one digit, an optional fraction with at
/// least one digit and an optional exponent with at least one digit, or one of
/// the three spellings of a NaN. No leading plus, no bare fraction, no space at
/// either end, and no infinity: a number too large to hold is refused rather
/// than stored as one. The increment on `TS.INCRBY` goes through the ordinary
/// number reader instead and takes all of those, which is not a distinction
/// anyone designed.
fn number(arg: &[u8]) -> Option<f64> {
    if nan_word(arg) {
        return Some(f64::NAN);
    }
    let mut at = usize::from(arg.first() == Some(&b'-'));
    if !digits(arg, &mut at) {
        return None;
    }
    if arg.get(at) == Some(&b'.') {
        at += 1;
        if !digits(arg, &mut at) {
            return None;
        }
    }
    if matches!(arg.get(at), Some(b'e' | b'E')) {
        at += 1;
        if matches!(arg.get(at), Some(b'+' | b'-')) {
            at += 1;
        }
        if !digits(arg, &mut at) {
            return None;
        }
    }
    if at != arg.len() {
        return None;
    }
    parse_f64(arg).filter(|value| value.is_finite())
}

/// Steps `at` over a run of digits, and answers whether there was one.
fn digits(arg: &[u8], at: &mut usize) -> bool {
    let start = *at;
    while arg.get(*at).is_some_and(u8::is_ascii_digit) {
        *at += 1;
    }
    *at > start
}

/// Whether `arg` is one of the three ways the module spells a reading that is
/// not a number.
fn nan_word(arg: &[u8]) -> bool {
    match arg.len() {
        3 => arg.eq_ignore_ascii_case(b"nan"),
        4 => arg.eq_ignore_ascii_case(b"-nan") || arg.eq_ignore_ascii_case(b"+nan"),
        _ => false,
    }
}

/// A sample value on the way out: the shortest digits that read back as the same
/// number, as a simple string on RESP2 and a double on RESP3.
fn value(out: &mut Out, d: f64) {
    if out.proto().is_resp3() {
        out.double(d);
        return;
    }
    let mut buf = [0u8; DOUBLE_MAX];
    out.simple(write_dragonbox(&mut buf, d));
}

/// The wrong kind of key, worded the way the module words it.
///
/// The keyspace answers its own `WRONGTYPE` for a key holding a native type and
/// the downcast answers for a key holding another module's value, and both of
/// those reach a client as the same sentence, so both come through here.
fn wrong_kind() -> Error {
    Error::new(Code::Invalid, WRONG_KIND)
}

/// The series under `key` for writing, or `None` if the key is not there.
fn write<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d mut TsBody>> {
    match db.foreign_mut(key) {
        Ok(Some(body)) => match body.downcast_mut::<TsBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(wrong_kind()),
        },
        Ok(None) => Ok(None),
        Err(e) if e.code() == Code::WrongType => Err(wrong_kind()),
        Err(e) => Err(e),
    }
}

/// The same, for reading.
fn read<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d TsBody>> {
    match db.foreign(key) {
        Ok(Some(body)) => match body.downcast_ref::<TsBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(wrong_kind()),
        },
        Ok(None) => Ok(None),
        Err(e) if e.code() == Code::WrongType => Err(wrong_kind()),
        Err(e) => Err(e),
    }
}
