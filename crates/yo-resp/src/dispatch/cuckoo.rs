//! `CF.*`, the cuckoo filter family RedisBloom put on the wire.
//!
//! The filter is in `yo-sketch` and this is the wire in front of it, the same
//! split as `super::bloom`. What is decided here is the argument grammar, the
//! reply shapes and the errors, and as with the Bloom family that turned out to
//! be the larger half of the compatibility.
//!
//! # Errors
//!
//! Two kinds again, and the split is not tidier here than it was there. `CF.DEL`
//! answers `Not found` about a key that is not there and `CF.INFO` answers `ERR
//! not found` about the same key, and `CF.COMPACT` has a third sentence of its
//! own for it. Those are the module's lines and they are copied word for word,
//! prefix and all, because a client that branches on one of them is the reason
//! to implement the family.
//!
//! # Where the two protocols disagree
//!
//! `CF.INSERT` answers one value per item, and an item that did not fit is
//! minus one on RESP2 and false on RESP3, so a RESP3 client cannot tell a
//! refusal from an item that was already there. `CF.INSERTNX` answers integers
//! on both, because it has three outcomes rather than two and minus one has to
//! survive. Both are the reference's shapes.
//!
//! # What a client can see that is different
//!
//! Nothing found so far. The geometry, the hash, the fingerprint, the kick and
//! its rollback, the compaction order, the dumped header and the positions a
//! dump walks through all match a real Redis 8.10.1 with RedisBloom in it, so a
//! `CF.SCANDUMP` from one loads into the other in either direction.

use yo_common::num::parse_i64;
use yo_common::{Code, Error, Result};
use yo_kv::{Db, Foreign, Keyspace};
use yo_sketch::cuckoo::{
    Cuckoo, HEADER, Insert, MAX_BUCKET_SIZE, MAX_CAPACITY, MAX_EXPANSION, MAX_ITERATIONS,
};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// The capacity a filter gets when the client did not say, which is
/// `cf-initial-size`.
const DEFAULT_CAPACITY: u64 = 1024;
/// Slots per bucket when the client did not say, which is `cf-bucket-size`. One
/// is a plain cuckoo table and fills at about half, four is where the occupancy
/// stops improving, and two is the module's answer.
const DEFAULT_BUCKET_SIZE: u16 = 2;
/// How many kicks an insert does before it gives up, which is
/// `cf-max-iterations`.
const DEFAULT_ITERATIONS: u16 = 20;
/// How much wider each filter is than the one before it, which is
/// `cf-expansion-factor`.
const DEFAULT_EXPANSION: u16 = 1;

/// What the commands that want a filter say about a key that has none.
const NOT_FOUND: &str = "not found";
/// What `CF.RESERVE` says about a key that already has one.
const ITEM_EXISTS: &str = "item exists";

/// `CF.DEL`'s own spelling of a key it could not use, which has no prefix and a
/// capital where the other one has none.
const DEL_NOT_FOUND: &[u8] = b"Not found";
/// And `CF.COMPACT`'s, which is a third sentence for the same key.
const NO_FILTER: &[u8] = b"Cuckoo filter was not found";
/// A capacity that is not a number.
const BAD_CAPACITY: &[u8] = b"Bad capacity";
/// One that is a number and is out of range. The bound is written in terms of
/// the option because the option decides it.
const RESERVE_RANGE: &[u8] = b"Capacity must be in the range [2 * BUCKETSIZE, 1073741824]";
/// The same range from `CF.INSERT`, which cannot be told a bucket size and so
/// names the config that holds it instead.
const INSERT_RANGE: &[u8] = b"Capacity must be in the range [cf-bucket-size * 2, 1073741824]";
/// A bucket size that is not a number.
const BAD_BUCKET: &[u8] = b"Couldn't parse BUCKETSIZE";
/// One that is and is out of range.
const BUCKET_RANGE: &[u8] = b"BUCKETSIZE: value must be in the range [1, 255]";
/// A kick budget that is not a number.
const BAD_ITERATIONS: &[u8] = b"Couldn't parse MAXITERATIONS";
/// One that is and is out of range.
const ITERATION_RANGE: &[u8] = b"MAXITERATIONS: value must be in the range [1, 65535]";
/// A growth factor that is not a number.
const BAD_EXPANSION: &[u8] = b"Couldn't parse EXPANSION";
/// One that is and is out of range.
const EXPANSION_RANGE: &[u8] = b"EXPANSION: value must be in the range [0, 32768]";
/// A filter that has no room for an item and was told not to grow.
const FULL: &[u8] = b"Filter is full";
/// One that has no room and has grown as far as it is allowed.
const MAX_EXPANSIONS: &[u8] = b"Maximum expansions reached";
/// A token `CF.INSERT` does not know.
const UNKNOWN_ARG: &[u8] = b"Unknown argument received";
/// A dump position that is not a number, or is one this end of the command will
/// not take.
const BAD_POSITION: &[u8] = b"Invalid position";
/// A chunk that arrived where a header should have been and is not the size of
/// one.
const BAD_HEADER: &[u8] = b"Invalid header";
/// One that is the size of a header and describes a filter that could not be
/// built.
const NO_CREATE: &[u8] = b"Couldn't create filter!";
/// A chunk that does not fit where it says it goes. One sentence covers both an
/// offset past every filter and a chunk that runs off the end of one, which is
/// the module's own economy and not this file's.
const BAD_CHUNK: &[u8] = b"Couldn't load chunk!";

/// What a key holding anything else gets.
const WRONG_KIND: &str = "Operation against a key holding the wrong kind of value";

/// A filter under a key.
#[derive(Debug)]
pub(super) struct CuckooBody {
    /// The chain. Everything `CF.INFO` and `CF.DEBUG` report is derived from it
    /// rather than recorded beside it, so a filter that arrived through
    /// `CF.LOADCHUNK` answers exactly the way one built here does.
    c: Cuckoo,
}

impl Foreign for CuckooBody {
    fn type_name(&self) -> &'static str {
        // The module's own name for the type, which is what a client that has a
        // filter sees from `TYPE` on a real server.
        "MBbloomCF"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.c.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // A filter with nothing in it is still a filter, the same as the Bloom
        // side: `CF.RESERVE` makes one and the client expects to find it.
        false
    }
}

pub(super) fn execute(db: &mut Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    // Every command here names one filter and names it first, so the stripe is
    // found once and everything below goes on taking a keyspace.
    let db = db.at(args.get(1));
    match spec.name {
        "cf.reserve" => reserve(db, args, out),
        "cf.add" => add(db, args, out, false),
        "cf.addnx" => add(db, args, out, true),
        "cf.insert" => insert(db, args, out, false),
        "cf.insertnx" => insert(db, args, out, true),
        "cf.exists" => exists(db, args, out),
        "cf.mexists" => mexists(db, args, out),
        "cf.count" => count(db, args, out),
        "cf.del" => del(db, args, out),
        "cf.scandump" => scandump(db, args, out),
        "cf.loadchunk" => loadchunk(db, args, out),
        "cf.info" => info(db, args, out),
        "cf.debug" => debug(db, args, out),
        "cf.compact" => compact(db, args, out),
        other => unreachable!("{other} is not a cuckoo filter command"),
    }
}

/// `CF.RESERVE key capacity [BUCKETSIZE n] [MAXITERATIONS n] [EXPANSION n]`.
///
/// Everything after the capacity is read as name and value pairs, so an odd
/// number of them is an arity error rather than a complaint about the last one.
/// A pair whose name is none of the three is dropped on the floor, which is what
/// the reference does and is the opposite of what `CF.INSERT` does with the same
/// mistake.
fn reserve(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len().is_multiple_of(2) {
        return Err(args::wrong_arity("cf.reserve"));
    }
    let Some(capacity) = parse_i64(args.get(2)) else {
        out.error(BAD_CAPACITY);
        return Ok(());
    };
    let (bucket_size, max_iterations, expansion) = match geometry(args) {
        Ok(g) => g,
        Err(msg) => {
            out.error(msg);
            return Ok(());
        }
    };
    // The capacity is checked last and against the bucket size, so a command
    // with a bad capacity and a bad bucket size answers about the bucket size.
    if capacity < 2 * i64::from(bucket_size) || capacity > MAX_CAPACITY {
        out.error(RESERVE_RANGE);
        return Ok(());
    }
    let key = args.get(1);
    if write(db, key)?.is_some() {
        return Err(cf(ITEM_EXISTS));
    }
    put(
        db,
        key,
        Cuckoo::new(capacity as u64, bucket_size, max_iterations, expansion),
    );
    out.ok();
    Ok(())
}

/// The three numeric options `CF.RESERVE` takes, in the order it complains
/// about them.
///
/// That order is not the order they are written in: a command with a bad kick
/// budget and a bad bucket size answers about the budget wherever the two sit,
/// because the module looks for each keyword in turn rather than walking the
/// arguments. The first pair with a given name wins and a second pair with the
/// same name is not even looked at, which is the same thing seen from the other
/// side.
fn geometry(args: Args<'_>) -> core::result::Result<(u16, u16, u16), &'static [u8]> {
    let iterations = setting(
        args,
        b"maxiterations",
        (1, MAX_ITERATIONS),
        BAD_ITERATIONS,
        ITERATION_RANGE,
    )?;
    let bucket_size = setting(
        args,
        b"bucketsize",
        (1, MAX_BUCKET_SIZE),
        BAD_BUCKET,
        BUCKET_RANGE,
    )?;
    let expansion = setting(
        args,
        b"expansion",
        (0, MAX_EXPANSION),
        BAD_EXPANSION,
        EXPANSION_RANGE,
    )?;
    Ok((
        bucket_size.unwrap_or(i64::from(DEFAULT_BUCKET_SIZE)) as u16,
        iterations.unwrap_or(i64::from(DEFAULT_ITERATIONS)) as u16,
        expansion.unwrap_or(i64::from(DEFAULT_EXPANSION)) as u16,
    ))
}

/// One option's value, with its own sentence for each way it can be wrong.
fn setting(
    args: Args<'_>,
    name: &[u8],
    bounds: (i64, i64),
    bad: &'static [u8],
    range: &'static [u8],
) -> core::result::Result<Option<i64>, &'static [u8]> {
    let Some(value) = pair(args, name) else {
        return Ok(None);
    };
    match parse_i64(value) {
        None => Err(bad),
        Some(n) if n < bounds.0 || n > bounds.1 => Err(range),
        Some(n) => Ok(Some(n)),
    }
}

/// The value of the first pair called `name`, if the client sent one.
fn pair<'a>(args: Args<'a>, name: &[u8]) -> Option<&'a [u8]> {
    (3..args.len())
        .step_by(2)
        .find(|&i| args::is(args.get(i), name))
        .map(|i| args.get(i + 1))
}

/// `CF.ADD key item` and `CF.ADDNX key item`, either of which makes the filter
/// if the key is free.
///
/// The difference is what a filter that already has the item does. `CF.ADD`
/// puts a second copy in and `CF.COUNT` then says two, which is the thing a
/// cuckoo filter can do and a Bloom filter cannot. `CF.ADDNX` answers false and
/// leaves it alone, which costs a lookup and is why it is a separate command.
fn add(db: &mut Keyspace, args: Args<'_>, out: &mut Out, unique: bool) -> Result<()> {
    let body = open(db, args.get(1))?;
    let item = args.get(2);
    let done = match unique {
        true => body.c.insert_unique(item),
        false => body.c.insert(item),
    };
    match done {
        Insert::Yes => out.bool(true),
        Insert::Exists => out.bool(false),
        Insert::Full => out.error(FULL),
        Insert::MaxFilters => out.error(MAX_EXPANSIONS),
    }
    Ok(())
}

/// `CF.INSERT key [CAPACITY n] [NOCREATE] ITEMS item [item ...]`, and the same
/// for `CF.INSERTNX`.
///
/// One answer per item and the array is never short, which is where this differs
/// from `BF.MADD`: a filter that fills up part way through answers minus one for
/// every item after that rather than stopping and writing an error into the
/// middle of the array.
fn insert(db: &mut Keyspace, args: Args<'_>, out: &mut Out, unique: bool) -> Result<()> {
    let name = match unique {
        true => "cf.insertnx",
        false => "cf.insert",
    };
    let mut capacity = DEFAULT_CAPACITY;
    let mut create = true;
    let mut items = None;
    let mut i = 2;
    while i < args.len() {
        match word(args.get(i)) {
            Some(Opt::Items) => {
                items = Some(i + 1);
                break;
            }
            Some(Opt::NoCreate) => {
                create = false;
                i += 1;
            }
            Some(Opt::Capacity) => {
                let Some(value) = args.opt(i + 1) else {
                    return Err(args::wrong_arity(name));
                };
                let Some(n) = parse_i64(value) else {
                    out.error(BAD_CAPACITY);
                    return Ok(());
                };
                // Every occurrence is checked, unlike `CF.RESERVE`, so a second
                // `CAPACITY` that is out of range is an error even though the
                // first one would have been used.
                if !(2 * i64::from(DEFAULT_BUCKET_SIZE)..=MAX_CAPACITY).contains(&n) {
                    out.error(INSERT_RANGE);
                    return Ok(());
                }
                capacity = n as u64;
                i += 2;
            }
            None => {
                out.error(UNKNOWN_ARG);
                return Ok(());
            }
        }
    }
    // `ITEMS` with nothing after it and no `ITEMS` at all are the same
    // complaint, because both leave the command with nothing to do.
    let Some(first) = items.filter(|&at| at < args.len()) else {
        return Err(args::wrong_arity(name));
    };
    let key = args.get(1);
    let body = match write(db, key)? {
        Some(body) => body,
        None if create => {
            put(
                db,
                key,
                Cuckoo::new(
                    capacity,
                    DEFAULT_BUCKET_SIZE,
                    DEFAULT_ITERATIONS,
                    DEFAULT_EXPANSION,
                ),
            );
            write(db, key)?.expect("the filter was just created")
        }
        None => return Err(cf(NOT_FOUND)),
    };
    out.array(args.len() - first);
    for i in first..args.len() {
        let item = args.get(i);
        let done = match unique {
            true => body.c.insert_unique(item),
            false => body.c.insert(item),
        };
        match (unique, done) {
            (true, Insert::Yes) => out.int(1),
            (true, Insert::Exists) => out.int(0),
            (true, _) => out.int(-1),
            (false, Insert::Yes) => out.bool(true),
            // A refusal is minus one on RESP2 and false on RESP3, which is one
            // value written two ways rather than two answers.
            (false, _) if out.proto().is_resp3() => out.bool(false),
            (false, _) => out.int(-1),
        }
    }
    Ok(())
}

/// One of the three words `CF.INSERT` takes before its items.
enum Opt {
    Capacity,
    NoCreate,
    Items,
}

/// Which of them an argument is, which is decided on the first letter and not on
/// the word.
///
/// `CF.INSERT k NOSUCH ITEMS x` means `NOCREATE` on a real server and
/// `ITEMSXYZ` means `ITEMS`, the same shortcut `BF.INSERT` takes and for the
/// same reason: the module switches on one character. `CF.RESERVE` reads its
/// keywords whole and ignores what it does not know, so the two halves of the
/// family disagree about what a typo is.
fn word(arg: &[u8]) -> Option<Opt> {
    match arg.first().copied().unwrap_or(0).to_ascii_uppercase() {
        b'C' => Some(Opt::Capacity),
        b'I' => Some(Opt::Items),
        b'N' => Some(Opt::NoCreate),
        _ => None,
    }
}

/// `CF.EXISTS key item`.
///
/// A key holding something else answers a miss rather than `WRONGTYPE`, which
/// the three read commands and `CF.DEL` all do and the write commands do not.
fn exists(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let found = peek(db, args.get(1)).is_some_and(|b| b.c.contains(args.get(2)));
    out.bool(found);
    Ok(())
}

/// `CF.MEXISTS key item [item ...]`, with the same tolerance.
fn mexists(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let body = peek(db, args.get(1));
    out.array(args.len() - 2);
    for i in 2..args.len() {
        out.bool(body.is_some_and(|b| b.c.contains(args.get(i))));
    }
    Ok(())
}

/// `CF.COUNT key item`, which is how many copies of the item the filter thinks
/// it has and is an integer on both protocols.
fn count(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let n = peek(db, args.get(1)).map_or(0, |b| b.c.count(args.get(2)));
    out.uint(n);
    Ok(())
}

/// `CF.DEL key item`, which takes one copy out and compacts the chain if enough
/// has been deleted to be worth it.
fn del(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = peek_mut(db, args.get(1)) else {
        out.error(DEL_NOT_FOUND);
        return Ok(());
    };
    let gone = body.c.remove(args.get(2));
    out.bool(gone);
    Ok(())
}

/// `CF.SCANDUMP key position`.
///
/// Position zero asks for the header and comes back with one and the position to
/// ask with next. After that the position is one past the last byte handed over,
/// counted across every filter's fingerprints laid end to end, and zero with a
/// nil means there is nothing left. A filter with nothing in it answers that
/// straight away rather than handing out a header, so a client that dumps an
/// empty filter has nothing to load back.
fn scandump(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return Err(cf(NOT_FOUND));
    };
    // A negative position is refused here and taken as an offset by
    // `CF.LOADCHUNK`, which is the reference's asymmetry and not a typo.
    let Some(pos) = parse_i64(args.get(2)).filter(|&n| n >= 0) else {
        out.error(BAD_POSITION);
        return Ok(());
    };
    out.array(2);
    let header;
    let (next, data) = if body.c.is_empty() {
        (0, &[][..])
    } else if pos == 0 {
        header = body.c.header();
        (1, &header[..])
    } else {
        body.c.chunk(pos)
    };
    out.int(next);
    if next == 0 {
        out.nil();
    } else {
        out.bulk(data);
    }
    Ok(())
}

/// `CF.LOADCHUNK key position data`.
///
/// The key decides which of the two things this command is, the same as
/// `BF.LOADCHUNK`. A key that is not there is being created and only the header
/// can do that, so position one is the header and anything else has nothing to
/// attach to. A key that is there is being filled in, and position one is the
/// header again and so is refused outright rather than reaching for an offset.
fn loadchunk(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(pos) = parse_i64(args.get(2)).filter(|&n| n != 0) else {
        out.error(BAD_POSITION);
        return Ok(());
    };
    let key = args.get(1);
    let data = args.get(3);
    let Some(body) = write(db, key)? else {
        if pos != 1 {
            return Err(cf(NOT_FOUND));
        }
        if data.len() != HEADER {
            out.error(BAD_HEADER);
            return Ok(());
        }
        let Some(c) = Cuckoo::from_header(data) else {
            out.error(NO_CREATE);
            return Ok(());
        };
        put(db, key, c);
        out.ok();
        return Ok(());
    };
    if pos == 1 {
        return Err(cf(ITEM_EXISTS));
    }
    if body.c.load(pos, data) {
        out.ok();
    } else {
        out.error(BAD_CHUNK);
    }
    Ok(())
}

/// `CF.INFO key`, which is the whole shape of the chain and takes no field.
///
/// The bucket count it reports is the first filter's however far the chain has
/// grown, so a chain of three filters at an expansion of two says the same
/// number as the day it was reserved. That is the reference's answer and a
/// client working out how much room is left from it will be wrong, which is
/// why `CF.DEBUG` exists.
fn info(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return Err(cf(NOT_FOUND));
    };
    let c = &body.c;
    out.map(8);
    out.simple(b"Size");
    out.uint(c.reported_size());
    out.simple(b"Number of buckets");
    out.uint(c.buckets());
    out.simple(b"Number of filters");
    out.uint(c.filters() as u64);
    out.simple(b"Number of items inserted");
    out.uint(c.len());
    out.simple(b"Number of items deleted");
    out.uint(c.deleted());
    out.simple(b"Bucket size");
    out.uint(u64::from(c.bucket_size()));
    out.simple(b"Expansion rate");
    out.uint(u64::from(c.expansion()));
    out.simple(b"Max iterations");
    out.uint(u64::from(c.max_iterations()));
    Ok(())
}

/// `CF.DEBUG key`, which is one line of the numbers `CF.INFO` reports and the
/// ones it does not.
fn debug(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        return Err(cf(NOT_FOUND));
    };
    let c = &body.c;
    // One reply's worth of formatting off any path that matters, so it is
    // allowed to reach for the heap the way `BF.DEBUG` and `XINFO` do.
    let line = yo_alloc::allow(|| {
        format!(
            "bktsize:{} buckets:{} items:{} deletes:{} filters:{} max_iterations:{} expansion:{}",
            c.bucket_size(),
            c.buckets(),
            c.len(),
            c.deleted(),
            c.filters(),
            c.max_iterations(),
            c.expansion(),
        )
    });
    out.bulk(line.as_bytes());
    Ok(())
}

/// `CF.COMPACT key`, which pulls the newer filters down into the older ones and
/// drops the ones that empty.
///
/// A delete does this on its own once more than a tenth of the filter has been
/// deleted, and stops at the first filter that would not empty. This carries on
/// down the whole chain, so an item that could not move a moment ago moves now.
/// The module has the command down as a read and it is left that way here, flags
/// and category and all, even though it writes.
fn compact(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 2 {
        return Err(args::wrong_arity("cf.compact"));
    }
    let Some(body) = peek_mut(db, args.get(1)) else {
        out.error(NO_FILTER);
        return Ok(());
    };
    body.c.compact(true);
    out.ok();
    Ok(())
}

/// One of the module's `ERR` prefixed sentences.
fn cf(msg: &'static str) -> Error {
    Error::new(Code::Invalid, msg)
}

/// Put a new filter under `key`.
fn put(db: &mut Keyspace, key: &[u8], c: Cuckoo) {
    db.put_foreign(key, Box::new(CuckooBody { c }));
}

/// The filter under `key`, making an empty one at the defaults if the key is
/// free, which is what `CF.ADD` and `CF.ADDNX` do.
fn open<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<&'d mut CuckooBody> {
    if write(db, key)?.is_none() {
        put(
            db,
            key,
            Cuckoo::new(
                DEFAULT_CAPACITY,
                DEFAULT_BUCKET_SIZE,
                DEFAULT_ITERATIONS,
                DEFAULT_EXPANSION,
            ),
        );
    }
    // The borrow above ended with the `if`, so this is a second lookup rather
    // than one held across the insert.
    Ok(write(db, key)?.expect("the filter is there either way"))
}

/// The filter under `key` for writing, or `None` if the key is not there.
fn write<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d mut CuckooBody>> {
    match db.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<CuckooBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// The same, for reading.
fn read<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d CuckooBody>> {
    match db.foreign(key)? {
        Some(body) => match body.downcast_ref::<CuckooBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// And for the ones that would rather answer nothing than complain.
fn peek<'d>(db: &'d mut Keyspace, key: &[u8]) -> Option<&'d CuckooBody> {
    db.foreign(key)
        .ok()
        .flatten()
        .and_then(<dyn Foreign>::downcast_ref::<CuckooBody>)
}

/// The same for the two that write, which is `CF.DEL` and `CF.COMPACT`. Both
/// treat a key holding a string as a key with no filter under it.
fn peek_mut<'d>(db: &'d mut Keyspace, key: &[u8]) -> Option<&'d mut CuckooBody> {
    db.foreign_mut(key)
        .ok()
        .flatten()
        .and_then(<dyn Foreign>::downcast_mut::<CuckooBody>)
}
