//! `BF.*`, the Bloom filter family RedisBloom put on the wire.
//!
//! The filter itself is in `yo-sketch` and this is the wire in front of it: the
//! argument grammar, the reply shapes and the errors. The structure is a copy of
//! RedisBloom's on purpose, and `yo_sketch::bloom` says why at length. What is
//! decided here is the rest of the compatibility, which turned out to be the
//! larger half.
//!
//! # Errors
//!
//! A module writes its own error lines and Redis does not touch them, so the
//! prefix is whatever the module author typed. RedisBloom is inconsistent about
//! it in a way that is visible to any client that branches on the first word:
//! `BF.RESERVE` answers `ERR bad error rate` with a prefix and `Nonscaling
//! filters cannot expand` without one, `BF.INSERT` answers `Bad capacity` where
//! `BF.RESERVE` answers `ERR bad capacity` for the same mistake, and
//! `BF.SCANDUMP` and `BF.LOADCHUNK` disagree with each other about whether
//! `Second argument must be numeric` carries `ERR`. All of that is copied
//! rather than tidied, because a client that already has a branch on one of
//! these sentences is the reason to implement the family at all.
//!
//! That is why the errors here are two kinds. The ones a core Redis would have
//! produced, which is `WRONGTYPE` and the arity, come back as an [`Error`] and
//! the dispatcher writes them. The module's own sentences are written into the
//! reply here, prefix and all, and the function answers `Ok`. The two also
//! differ in where they can go: `BF.MADD` writes an error in the middle of an
//! array and carries on, which no error return could express.
//!
//! # What a client can see that is different
//!
//! Nothing so far, which is what the harness in `bfcmp.py` was written to
//! check. The geometry, the growth rule, the error tightening, the hash, the
//! bit order, the dumped header layout and the iterator values all match a real
//! Redis 8.10.1 with RedisBloom in it, so a `BF.SCANDUMP` from one loads into
//! the other in either direction.

use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};
use yo_kv::{Foreign, Keyspace};
use yo_sketch::bloom::{Added, Bloom, Load, MAX_CAPACITY, MAX_EXPANSION, MIN_CAPACITY};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// The error rate a filter is built at when the client did not say, which is
/// `bf-error-rate` and is not settable here.
const DEFAULT_ERROR: f64 = 0.01;
/// The capacity of the first link when the client did not say, which is
/// `bf-initial-size`.
const DEFAULT_CAPACITY: u64 = 100;
/// How much bigger each link is than the last when the client did not say,
/// which is `bf-expansion-factor`.
const DEFAULT_EXPANSION: u32 = 2;

/// What a command that needs a filter says about a key that has none.
const NOT_FOUND: &str = "not found";
/// What `BF.RESERVE` says about a key that already has one.
const ITEM_EXISTS: &str = "item exists";
/// An error rate that is not a number.
const BAD_ERROR_RATE: &str = "bad error rate";
/// One that is, and is not between nothing and everything. The zeros are the
/// module's own `%f` of its two bounds and are copied with them.
const ERROR_RANGE: &str = "error rate must be in the range (0.000000, 1.000000)";
/// A capacity that is not a number.
const BAD_CAPACITY: &str = "bad capacity";
/// One that is and is out of range.
const CAPACITY_RANGE: &str = "capacity must be in the range [1, 1073741824]";
/// `EXPANSION` with nothing after it.
const NO_EXPANSION: &str = "no expansion";
/// An expansion that is not a number.
const BAD_EXPANSION: &str = "bad expansion";
/// One that is and is out of range.
const EXPANSION_RANGE: &str = "expansion must be in the range [0, 32768]";
/// What a chain that will not grow says when its one link is full.
const FULL: &str = "non scaling filter is full";
/// A `BF.LOADCHUNK` header that no filter wrote.
const BAD_DATA: &str = "received bad data";
/// A chunk whose offset is past the end of the filter.
const NO_LINK: &str = "invalid offset - no link found";
/// One that starts inside a link and runs off the end of it. The capital in the
/// middle is the module's.
const TOO_BIG: &str = "invalid chunk - Too big for current filter";
/// `BF.LOADCHUNK` with an iterator that is not a number, which is the one place
/// this sentence carries a prefix.
const LOAD_NOT_NUMERIC: &str = "Second argument must be numeric";

/// The same sentence from `BF.SCANDUMP`, where it does not.
const SCAN_NOT_NUMERIC: &[u8] = b"Second argument must be numeric";
/// Both spellings of a chain that cannot grow, in one command.
const CANNOT_EXPAND: &[u8] = b"Nonscaling filters cannot expand";
/// A `BF.INFO` field nobody has.
const BAD_INFO: &[u8] = b"Invalid information value";
/// `BF.INSERT`'s spelling of a bad capacity, which is not `BF.RESERVE`'s.
const INSERT_CAPACITY: &[u8] = b"Bad capacity";
/// And of a bad error rate.
const INSERT_ERROR: &[u8] = b"Bad error rate";
/// And of a bad expansion, which covers the out of range case too rather than
/// having a second sentence for it the way `BF.RESERVE` does.
const INSERT_EXPANSION: &[u8] = b"Bad expansion";
/// A token `BF.INSERT` does not know, which `BF.RESERVE` would have ignored.
const UNKNOWN_ARG: &[u8] = b"Unknown argument received";

/// A filter under a key.
#[derive(Debug)]
pub(super) struct BloomBody {
    /// The chain. There is nothing else to keep: everything `BF.INFO` reports
    /// is derived from it rather than recorded beside it, which is why a filter
    /// that arrived through `BF.LOADCHUNK` answers the same as one that was
    /// built here.
    b: Bloom,
}

impl Foreign for BloomBody {
    fn type_name(&self) -> &'static str {
        // The module's type name, dashes and all. A client that has a filter
        // and asks `TYPE` gets this from a real server, and libraries branch on
        // it, so it is not somewhere to be tidier than the reference.
        "MBbloom--"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.b.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // A filter with nothing in it is still a filter. `BF.RESERVE` makes one
        // and the client expects to find it there, so this never says the key
        // can go: the only ways out are `DEL` and an expiry.
        false
    }
}

pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "bf.reserve" => reserve(db, args, out),
        "bf.add" => add(db, args, out),
        "bf.madd" => madd(db, args, out),
        "bf.insert" => insert(db, args, out),
        "bf.exists" => exists(db, args, out),
        "bf.mexists" => mexists(db, args, out),
        "bf.scandump" => scandump(db, args, out),
        "bf.loadchunk" => loadchunk(db, args, out),
        "bf.info" => info(db, args, out),
        "bf.card" => card(db, args, out),
        "bf.debug" => debug(db, args, out),
        other => unreachable!("{other} is not a bloom filter command"),
    }
}

/// `BF.RESERVE key error capacity [EXPANSION n] [NONSCALING]`.
///
/// Everything is checked before the key is looked at, which is the reference's
/// order and is visible: `BF.RESERVE` against a key holding a string with a bad
/// error rate answers about the error rate and not about the string.
fn reserve(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let capacity = capacity(args.get(3))?;
    let error = rate(args.get(2))?;
    let mut growth = DEFAULT_EXPANSION;
    let mut fixed = false;
    let mut asked_to_grow = false;
    let mut i = 4;
    while i < args.len() {
        let arg = args.get(i);
        if args::is(arg, b"nonscaling") {
            fixed = true;
            i += 1;
        } else if args::is(arg, b"expansion") {
            let Some(n) = args.opt(i + 1) else {
                return Err(bf(NO_EXPANSION));
            };
            growth = expansion(n)?;
            asked_to_grow = true;
            i += 2;
        } else {
            // Anything else is dropped on the floor. `BF.RESERVE k 0.01 10
            // junk` answers OK on the reference, and refusing it here would
            // reject commands a real server accepts.
            i += 1;
        }
    }
    if fixed && asked_to_grow {
        out.error(CANNOT_EXPAND);
        return Ok(());
    }
    let key = args.get(1);
    if write(db, key)?.is_some() {
        return Err(bf(ITEM_EXISTS));
    }
    put(db, key, Bloom::new(capacity, error, growth, fixed));
    out.ok();
    Ok(())
}

/// `BF.ADD key item`, which makes the filter if the key is free.
fn add(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = open(db, args.get(1))?;
    match body.b.add(args.get(2)) {
        Added::Yes => out.bool(true),
        Added::Already => out.bool(false),
        Added::Full => out.error_line(b"ERR ", FULL.as_bytes()),
    }
    Ok(())
}

/// `BF.MADD key item [item ...]`.
///
/// The reply stops where the filter did. A chain that fills part way through
/// answers with one element per item it managed and the error as the last one,
/// so the array is shorter than the argument list, which is the reference's
/// shape and not an accident of writing it this way.
fn madd(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = open(db, args.get(1))?;
    each(&mut body.b, args, 2, out);
    Ok(())
}

/// `BF.INSERT key [CAPACITY n] [ERROR e] [EXPANSION n] [NOCREATE] [NONSCALING]
/// ITEMS item [item ...]`.
///
/// The same work as `BF.RESERVE` and `BF.MADD` in one command, with its own
/// error sentences for the same mistakes.
fn insert(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut capacity = DEFAULT_CAPACITY;
    let mut error = DEFAULT_ERROR;
    let mut growth = DEFAULT_EXPANSION;
    let mut fixed = false;
    let mut create = true;
    let mut items = None;
    let mut i = 2;
    while i < args.len() {
        match option(args.get(i)) {
            Some(Opt::Items) => {
                items = Some(i + 1);
                break;
            }
            Some(Opt::NoCreate) => {
                create = false;
                i += 1;
            }
            Some(Opt::NonScaling) => {
                fixed = true;
                i += 1;
            }
            Some(Opt::Capacity) => {
                let Some(n) = args.opt(i + 1).and_then(number) else {
                    out.error(INSERT_CAPACITY);
                    return Ok(());
                };
                capacity = n as u64;
                i += 2;
            }
            Some(Opt::Error) => {
                let Some(e) = args.opt(i + 1).and_then(fraction) else {
                    out.error(INSERT_ERROR);
                    return Ok(());
                };
                error = e;
                i += 2;
            }
            Some(Opt::Expansion) => {
                let Some(n) = args.opt(i + 1).and_then(factor) else {
                    out.error(INSERT_EXPANSION);
                    return Ok(());
                };
                growth = n;
                i += 2;
            }
            None => {
                out.error(UNKNOWN_ARG);
                return Ok(());
            }
        }
    }
    // `ITEMS` with nothing after it is the same complaint as no `ITEMS` at all,
    // because both leave the command with nothing to do and the reference
    // answers the arity for both.
    let Some(first) = items.filter(|&at| at < args.len()) else {
        return Err(args::wrong_arity("bf.insert"));
    };
    let key = args.get(1);
    let body = match write(db, key)? {
        Some(body) => body,
        None if create => {
            put(db, key, Bloom::new(capacity, error, growth, fixed));
            write(db, key)?.expect("the filter was just created")
        }
        None => return Err(bf(NOT_FOUND)),
    };
    each(&mut body.b, args, first, out);
    Ok(())
}

/// One of the six words `BF.INSERT` takes before its items.
enum Opt {
    Capacity,
    Error,
    Expansion,
    NoCreate,
    NonScaling,
    Items,
}

/// Which of them an argument is, which is decided on the first letter and not
/// on the word.
///
/// `BF.INSERT k NOSUCH ITEMS x` builds a filter that will not grow on a real
/// server, because the module reads an `N` and reaches for `NONSCALING` without
/// looking at the rest, and `ITEMSXYZ` is `ITEMS`. Only `E` and `N` need a
/// second look, one to tell `ERROR` from `EXPANSION` and the other to tell
/// `NOCREATE` from `NONSCALING`, and both of those are exactly as far as the
/// module looks: `E` on its own is `EXPANSION` and `NOC` is `NOCREATE`.
///
/// This is a bug in the module and it is copied anyway. A client that has been
/// sending a misspelled option for years is getting a filter that does not grow
/// and does not know it, and answering an error here would break the one thing
/// it does rely on. `BF.RESERVE` has no such rule, and reads its two keywords
/// whole and ignores everything else, which is the other half of the same
/// inconsistency.
fn option(arg: &[u8]) -> Option<Opt> {
    let rest = arg.get(1).copied().unwrap_or(0).to_ascii_uppercase();
    match arg.first().copied().unwrap_or(0).to_ascii_uppercase() {
        b'C' => Some(Opt::Capacity),
        b'E' if rest == b'R' => Some(Opt::Error),
        b'E' => Some(Opt::Expansion),
        b'I' => Some(Opt::Items),
        b'N' if arg.len() >= 3 && arg[..3].eq_ignore_ascii_case(b"noc") => Some(Opt::NoCreate),
        b'N' => Some(Opt::NonScaling),
        _ => None,
    }
}

/// The shared body of `BF.MADD` and `BF.INSERT`: one answer per item until one
/// of them does not fit.
fn each(b: &mut Bloom, args: Args<'_>, from: usize, out: &mut Out) {
    let start = out.len();
    let mut n = 0;
    for i in from..args.len() {
        n += 1;
        match b.add(args.get(i)) {
            Added::Yes => out.bool(true),
            Added::Already => out.bool(false),
            Added::Full => {
                out.error_line(b"ERR ", FULL.as_bytes());
                break;
            }
        }
    }
    out.close_array(start, n);
}

/// `BF.EXISTS key item`.
///
/// A key holding something else answers zero rather than `WRONGTYPE`, which is
/// the reference's behaviour and is the one place in the family where it is.
/// `BF.ADD` on the same key does answer `WRONGTYPE`, so the two halves of a
/// check and set disagree about what that key is.
fn exists(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let found = peek(db, args.get(1)).is_some_and(|b| b.b.contains(args.get(2)));
    out.bool(found);
    Ok(())
}

/// `BF.MEXISTS key item [item ...]`, with the same tolerance.
fn mexists(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = peek(db, args.get(1));
    out.array(args.len() - 2);
    for i in 2..args.len() {
        out.bool(body.is_some_and(|b| b.b.contains(args.get(i))));
    }
    Ok(())
}

/// `BF.SCANDUMP key iterator`.
///
/// Iterator zero asks for the header and comes back with one. After that the
/// iterator is one past the last byte handed over, counted across every link's
/// bit array laid end to end, and zero means there is nothing left. A chunk
/// never spans two links, so a chain of `n` links takes `n + 2` calls.
fn scandump(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return Err(bf(NOT_FOUND));
    };
    let Some(iter) = parse_i64(args.get(2)) else {
        out.error(SCAN_NOT_NUMERIC);
        return Ok(());
    };
    out.array(2);
    if iter == 0 {
        let header = body.b.header();
        out.int(1);
        out.bulk(&header);
    } else {
        let (next, data) = body.b.chunk(iter);
        out.int(next);
        out.bulk(data);
    }
    Ok(())
}

/// `BF.LOADCHUNK key iterator data`.
///
/// The key decides which of the two things this command is. A key that is not
/// there is being created, and only the header can do that, so iterator one is
/// the header and anything else has nothing to attach to. A key that is there
/// is being filled in, and every iterator including one is an offset into the
/// filter that is already sitting under it.
///
/// So a header aimed at an existing key is not a special case here and does not
/// need to be: its iterator of one is smaller than the header itself, which is
/// already the one arithmetic that cannot be an offset. Loading a header on top
/// of a filter and loading nonsense on top of one are the same refusal for the
/// same reason, which is the reference's behaviour and falls out rather than
/// being written down.
fn loadchunk(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(iter) = parse_i64(args.get(2)) else {
        return Err(bf(LOAD_NOT_NUMERIC));
    };
    let key = args.get(1);
    let data = args.get(3);
    let Some(body) = write(db, key)? else {
        if iter != 1 {
            return Err(bf(NOT_FOUND));
        }
        let Some(b) = Bloom::from_header(data) else {
            return Err(bf(BAD_DATA));
        };
        put(db, key, b);
        out.ok();
        return Ok(());
    };
    match body.b.load(iter, data) {
        Ok(()) => out.ok(),
        Err(Load::BadData) => return Err(bf(BAD_DATA)),
        Err(Load::NoLink) => return Err(bf(NO_LINK)),
        Err(Load::TooBig) => return Err(bf(TOO_BIG)),
    }
    Ok(())
}

/// `BF.INFO key [CAPACITY | SIZE | FILTERS | ITEMS | EXPANSION]`.
///
/// The whole thing is a map of five. One field is a map of one on RESP3 and a
/// bare one element array on RESP2, which is the reference's shape and is the
/// one reply in the family where the two protocols do not carry the same
/// information: a RESP2 client that asked for one field gets the number without
/// being told which field it is, which it already knows.
fn info(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() > 3 {
        return Err(args::wrong_arity("bf.info"));
    }
    let Some(body) = read(db, args.get(1))? else {
        return Err(bf(NOT_FOUND));
    };
    let b = &body.b;
    let Some(field) = args.opt(2) else {
        out.map(5);
        out.simple(b"Capacity");
        out.uint(b.capacity());
        out.simple(b"Size");
        out.uint(b.reported_size());
        out.simple(b"Number of filters");
        out.uint(b.filters() as u64);
        out.simple(b"Number of items inserted");
        out.uint(b.len());
        out.simple(b"Expansion rate");
        match b.expansion() {
            Some(n) => out.uint(u64::from(n)),
            None => out.nil(),
        }
        return Ok(());
    };
    let (name, value): (&[u8], Option<u64>) = if args::is(field, b"capacity") {
        (b"Capacity", Some(b.capacity()))
    } else if args::is(field, b"size") {
        (b"Size", Some(b.reported_size()))
    } else if args::is(field, b"filters") {
        (b"Number of filters", Some(b.filters() as u64))
    } else if args::is(field, b"items") {
        (b"Number of items inserted", Some(b.len()))
    } else if args::is(field, b"expansion") {
        (b"Expansion rate", b.expansion().map(u64::from))
    } else {
        out.error(BAD_INFO);
        return Ok(());
    };
    if out.proto().is_resp3() {
        out.map(1);
        out.simple(name);
    } else {
        out.array(1);
    }
    match value {
        Some(n) => out.uint(n),
        None => out.nil(),
    }
    Ok(())
}

/// `BF.CARD key`, which is how many items went in and not how many bits are set.
///
/// Zero for a key that is not there, where `BF.INFO` answers an error for the
/// same key. Both are the reference's.
fn card(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let n = read(db, args.get(1))?.map_or(0, |b| b.b.len());
    out.uint(n);
    Ok(())
}

/// `BF.DEBUG key`, which is the chain's size and then a line per link.
///
/// The lines are the module's own text and the numbers in them are what a
/// client would otherwise have to take a `BF.SCANDUMP` apart to see, so this is
/// worth having exactly rather than approximately.
fn debug(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return Err(bf(NOT_FOUND));
    };
    // A whole reply's worth of formatting, off any path that matters, so it is
    // allowed to reach for the heap the way `XINFO` and the script errors do.
    let lines = yo_alloc::allow(|| {
        let mut lines = Vec::with_capacity(body.b.filters() + 1);
        lines.push(format!("size:{}", body.b.len()));
        for l in body.b.links() {
            lines.push(format!(
                "bytes:{} bits:{} hashes:{} hashwidth:64 capacity:{} size:{} ratio:{}",
                l.bytes,
                l.bits,
                l.hashes,
                l.capacity,
                l.size,
                significant(l.error),
            ));
        }
        lines
    });
    out.array(lines.len());
    for line in &lines {
        out.bulk(line.as_bytes());
    }
    Ok(())
}

/// A double the way C's `%g` writes one, which is what the module printed the
/// error rate with.
///
/// Six significant digits, an exponent when the number is smaller than `1e-4`
/// or not smaller than `1e6`, trailing zeros dropped, and the exponent itself
/// at least two digits with a sign. Rust's own `{}` writes the shortest text
/// that round trips instead, which agrees on `0.005` and disagrees on
/// `0.00000000005`, where `%g` says `5e-11`.
fn significant(d: f64) -> String {
    let sci = format!("{d:.5e}");
    let (mantissa, exponent) = sci.split_once('e').expect("a scientific form has an e");
    let exponent: i32 = exponent.parse().expect("and a whole number after it");
    if !(-4..6).contains(&exponent) {
        let m = trim(mantissa);
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", exponent.abs())
    } else {
        let places = (5 - exponent).max(0) as usize;
        trim(&format!("{d:.places$}")).to_string()
    }
}

/// Trailing zeros after a decimal point, and then the point itself, taken off.
fn trim(s: &str) -> &str {
    match s.contains('.') {
        true => s.trim_end_matches('0').trim_end_matches('.'),
        false => s,
    }
}

/// An error rate for `BF.RESERVE`, which has a sentence for each way it can be
/// wrong.
fn rate(arg: &[u8]) -> Result<f64> {
    match parse_f64(arg) {
        None => Err(bf(BAD_ERROR_RATE)),
        Some(e) if e <= 0.0 || e >= 1.0 => Err(bf(ERROR_RANGE)),
        Some(e) => Ok(e),
    }
}

/// A capacity for `BF.RESERVE`, likewise.
fn capacity(arg: &[u8]) -> Result<u64> {
    match parse_i64(arg) {
        None => Err(bf(BAD_CAPACITY)),
        Some(n) if !(MIN_CAPACITY..=MAX_CAPACITY).contains(&n) => Err(bf(CAPACITY_RANGE)),
        Some(n) => Ok(n as u64),
    }
}

/// An expansion factor for `BF.RESERVE`. Zero is allowed and means the chain
/// will not grow, which is `NONSCALING` said another way.
fn expansion(arg: &[u8]) -> Result<u32> {
    match parse_i64(arg) {
        None => Err(bf(BAD_EXPANSION)),
        Some(n) if !(0..=MAX_EXPANSION).contains(&n) => Err(bf(EXPANSION_RANGE)),
        Some(n) => Ok(n as u32),
    }
}

/// The same three for `BF.INSERT`, which has one sentence each and so has no
/// use for the difference between a number it cannot read and one it can.
fn number(arg: &[u8]) -> Option<i64> {
    parse_i64(arg).filter(|n| (MIN_CAPACITY..=MAX_CAPACITY).contains(n))
}

/// An error rate for `BF.INSERT`.
fn fraction(arg: &[u8]) -> Option<f64> {
    parse_f64(arg).filter(|e| *e > 0.0 && *e < 1.0)
}

/// An expansion factor for `BF.INSERT`.
fn factor(arg: &[u8]) -> Option<u32> {
    parse_i64(arg)
        .filter(|n| (0..=MAX_EXPANSION).contains(n))
        .map(|n| n as u32)
}

/// One of the module's `ERR` prefixed sentences.
fn bf(msg: &'static str) -> Error {
    Error::new(Code::Invalid, msg)
}

/// Put a new filter under `key`.
fn put(db: &mut Keyspace, key: &[u8], b: Bloom) {
    db.put_foreign(key, Box::new(BloomBody { b }));
}

/// The filter under `key`, making an empty one at the configured defaults if
/// the key is free, which is what `BF.ADD` and `BF.MADD` do.
fn open<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<&'d mut BloomBody> {
    if write(db, key)?.is_none() {
        put(
            db,
            key,
            Bloom::new(DEFAULT_CAPACITY, DEFAULT_ERROR, DEFAULT_EXPANSION, false),
        );
    }
    // The borrow above ended with the `if`, so this is a second lookup rather
    // than the same one held across the insert. One hash on the create path is
    // not worth an unsafe reborrow.
    Ok(write(db, key)?.expect("the filter is there either way"))
}

/// The filter under `key` for writing, or `None` if the key is not there.
///
/// An error for a key holding anything else, foreign bodies included, which is
/// the one case the keyspace cannot decide on its own because only this file
/// knows which foreign body it wanted.
fn write<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d mut BloomBody>> {
    match db.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<BloomBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// The same, for reading.
fn read<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d BloomBody>> {
    match db.foreign(key)? {
        Some(body) => match body.downcast_ref::<BloomBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// And for the two that would rather answer no than complain.
fn peek<'d>(db: &'d mut Keyspace, key: &[u8]) -> Option<&'d BloomBody> {
    db.foreign(key)
        .ok()
        .flatten()
        .and_then(<dyn Foreign>::downcast_ref::<BloomBody>)
}

/// What a key holding anything else gets.
const WRONG_KIND: &str = "Operation against a key holding the wrong kind of value";
