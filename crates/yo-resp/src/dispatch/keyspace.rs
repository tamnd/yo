//! The keyspace commands, from the wire.
//!
//! These are the ones that do not care what a value is. `DEL` deletes a string
//! today and will delete a list without a line changing here, because the
//! question it asks is about the key and not about what is under it.
//!
//! Four of them are here rather than in M3 with the rest of the group, and the
//! reason is that Redis's own test suite cannot run a single file without them.
//! In external mode the suite says `FLUSHALL` before every `start_server` block
//! and gives up on the whole file if that fails, and most test bodies then say
//! `DEL` to get to a known state. A compatibility suite that cannot start is
//! not a compatibility suite, so these landed early.
//!
//! `TYPE` reads the tag in the record's meta byte, so it answers whatever the
//! key actually holds and does not grow a case each time a type lands. Today
//! that is `string`, `set` or `none`, and the day the hash lands it is `hash`
//! too without a line here changing.

use super::args::{self, Args};
use super::scan;
use super::table::Spec;
use crate::reply::Out;
use yo_common::{Code, Error, Held, Result, glob_matches};
use yo_kv::rdb::Bad;
use yo_kv::sort::Sort;
use yo_kv::{Applied, Ask, Cond, Db, Holds, Keyspace, Kind, MAX_AT, Moved};

/// Milliseconds in a second, which is the whole of what the p in `PTTL` means.
const SECOND: i64 = 1000;

/// What `COPY` says for a `DB` that names a database this server does not have.
///
/// A negative number lands here too rather than in the integer parser, because
/// `-1` is a perfectly good integer and is not a database.
const DB_OUT_OF_RANGE: &str = "DB index is out of range";

/// And what it says when the source and the destination are the same thing.
///
/// The same key in the same database, so `COPY a a DB 1` never reaches this.
const SAME_OBJECT: &str = "source and destination objects are the same";

/// What `RESTORE` says about a key that is already there without `REPLACE`.
///
/// The full stop is Redis's and so is the prefix. `BUSYKEY` is a code a client
/// branches on, the same way it branches on `WRONGTYPE`, which is why this is
/// written where it is decided rather than turned into an [`Error`].
const BUSY_KEY: &[u8] = b"BUSYKEY Target key name already exists.";

/// What `RESTORE` says about a negative ttl.
const BAD_TTL: &str = "Invalid TTL value, must be >= 0";

/// And about a negative `IDLETIME`.
const BAD_IDLETIME: &str = "Invalid IDLETIME value, must be >= 0";

/// And about a `FREQ` outside the byte an LFU counter fits in.
const BAD_FREQ: &str = "Invalid FREQ value, must be >= 0 and <= 255";

/// What `RESTORE` says when the footer is wrong, whichever half of it.
///
/// One message for two conditions, which is Redis's choice and is the right one:
/// a client that gets this has bytes it should not send again, and knowing
/// whether the version or the checksum was the problem does not change that.
const BAD_FOOTER: &str = "DUMP payload version or checksum are wrong";

/// And when the footer was fine and the rest was not.
const BAD_PAYLOAD: &str = "Bad data format";

/// What Redis says when `NX` is given alongside any of the other three.
const NX_WITH_OTHERS: &str = "NX and XX, GT or LT options at the same time are not compatible";

/// And when `GT` and `LT` are given together, which is the one pair that
/// contradicts itself without `NX` being involved.
const GT_WITH_LT: &str = "GT and LT options at the same time are not compatible";

/// What `OBJECT FREQ` says when the policy is not an LFU one.
///
/// The two readings share one field, so under any other policy those bits are a
/// clock and reporting them as a frequency would be reporting a number that
/// means nothing. The second half, about switching at runtime taking time to
/// adjust, is upstream's wording and it is a description of exactly that.
const NOT_LFU: &str = "An LFU maxmemory policy is not selected, access frequency not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust.";

/// What `OBJECT IDLETIME` says when the policy is an LFU one, which is the same
/// sentence the other way round.
const IS_LFU: &str = "An LFU maxmemory policy is selected, idle time not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust.";

/// The text `OBJECT HELP` prints, one line an entry.
const OBJECT_HELP: &[&str] = &[
    "OBJECT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "ENCODING <key>",
    "    Return the kind of internal representation used in order to store the value",
    "    associated with a <key>.",
    "FREQ <key>",
    "    Return the access frequency index of the <key>. The returned integer is",
    "    proportional to the logarithm of the real access frequency.",
    "IDLETIME <key>",
    "    Return the idle time of the <key>, that is the approximated number of",
    "    seconds elapsed since the last access to the value.",
    "REFCOUNT <key>",
    "    Return the number of references of the value associated with the key.",
    "HELP",
    "    Print this help.",
];

/// Run one keyspace command.
///
/// Every database and not one, because `COPY key dst DB n` writes into a
/// database the connection is not on. Everything else here takes `at` and looks
/// no further, and the borrow of the slice ends at the top of each arm.
///
/// Almost every arm below names one key and reaches one stripe, which is the
/// shape that makes a striped database worth having. The three that do not are
/// the two that name two keys, which take the two stripes those keys are on,
/// and the three walks, which are about a database rather than about a key and
/// so go through [`Db`] and reach all of them.
pub(super) fn execute(
    dbs: &[Db],
    at: usize,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    // The two that reach a database nobody selected, and the reason this
    // function is handed every database rather than one.
    match spec.name {
        "copy" => return copy(dbs, at, args, out),
        "move" => return move_key(dbs, at, args, out),
        _ => {}
    }
    let db = &dbs[at];
    match spec.name {
        // `UNLINK` is `DEL` with the freeing moved to a background thread on a
        // real server. Ours frees on the spot, which is what `UNLINK` promises
        // a client: the key is gone from the keyspace when the reply arrives.
        // The promise is about visibility and not about which thread did the
        // work, so this is the same body rather than a divergence.
        "del" | "unlink" => {
            let mut gone = 0i64;
            for i in 1..args.len() {
                let key = args.get(i);
                if db.hold(key).del(key) {
                    gone += 1;
                }
            }
            out.int(gone);
        }
        // A key named twice counts twice, which looks like a bug and is what
        // Redis does. `EXISTS k k` on one key answers two.
        "exists" => {
            let mut found = 0i64;
            for i in 1..args.len() {
                let key = args.get(i);
                if db.hold(key).exists(key) {
                    found += 1;
                }
            }
            out.int(found);
        }
        "type" => {
            // Through `type_name` and not `kind_of`, because a foreign body
            // knows its own word and the tag it rides on does not. A client
            // asking about a graph is told `graph`.
            let key = args.get(1);
            let name = match db.hold(key).type_name(key) {
                Some(name) => name.as_bytes(),
                None => &b"none"[..],
            };
            // A simple string on both protocols, which is unusual enough to be
            // worth saying out loud: most replies that carry a word are bulk
            // strings and this one is not.
            out.simple(name);
        }
        // A key named twice counts twice here as well, and on a real server the
        // difference between this and `EXISTS` is that this moves the key up the
        // eviction order. There is no eviction yet, so the two are the same walk
        // and the day it lands the bump goes in the store and not here.
        // One key at a time and not one call, because two of the keys named can
        // be on two stripes. The count is the same either way: a key named
        // twice counts twice here as well.
        "touch" => {
            let mut hit = 0i64;
            for i in 1..args.len() {
                let key = args.get(i);
                hit += db.hold(key).touch(core::iter::once(key)) as i64;
            }
            out.int(hit);
        }
        "rename" | "renamenx" => rename(db, spec.name, args, out)?,
        "expire" | "pexpire" | "expireat" | "pexpireat" => expire(db, spec.name, args, out)?,
        "persist" => {
            let key = args.get(1);
            out.int(i64::from(db.hold(key).persist(key)));
        }
        "ttl" | "pttl" | "expiretime" | "pexpiretime" => ask(db, spec.name, args, out),
        "object" => object(db, args, out)?,
        "scan" => scan(db, args, out)?,
        "keys" => keys(db, args.get(1), out),
        // The one command here that is handed the whole database rather than a
        // stripe, because it is the only one that reads keys nobody named.
        // `BY w_*` and `GET w_*` are resolved inside the store, one lookup per
        // element, and every one of those lookups can land on a different
        // stripe, so there is nothing to route once at this end.
        "sort" | "sort_ro" => sort(db, spec.name, args, out)?,
        // One arm and not a guarded pair, because a guard and the arm behind
        // it would each want this stripe and the second would be waiting on the
        // first.
        "dump" => {
            let key = args.get(1);
            let mut stripe = db.hold(key);
            if is_foreign(&mut stripe, key) {
                return Err(no_dump(&mut stripe, key));
            }
            match stripe.dump(key) {
                Some(payload) => out.bulk(&payload),
                None => out.nil(),
            }
        }
        "restore" => restore(db, args, out)?,
        // Every stripe and not one, and the stripe is drawn first so that the
        // key is still drawn from the database rather than from whichever
        // stripe happened to be asked. See [`Db::random_key`].
        "randomkey" => {
            if !db.random_key(|key| out.bulk(key)) {
                out.nil();
            }
        }
        other => unreachable!("keyspace command with no body: {other}"),
    }
    Ok(())
}

/// `SCAN cursor [MATCH pattern] [COUNT count] [TYPE type]`.
///
/// The cursor names a place in the database and the database is several
/// stripes, so it names the stripe as well. That is [`Db::scan`] and nothing
/// here has to know about it.
///
/// The cursor names a place in the keyspace and not a place in memory, so it
/// stays right across a resize and a client can hold one for as long as it
/// likes. What it does not do is stay right across a `SELECT`, because two
/// databases are two keyspaces and a position in one means nothing in the
/// other. Redis has the same rule and neither server enforces it, since the
/// only thing a cursor from elsewhere can do is answer the wrong keys.
///
/// `COUNT` is a floor and not a ceiling, and a batch can come back empty with a
/// cursor that is not zero. That is the one thing a client has to get right and
/// it is the same thing Redis asks of it.
///
/// An unknown `TYPE` is not an error. Redis compares the word against the type
/// of each key it walks past, so a type nothing can hold matches nothing and
/// the scan runs to the end answering nothing, which is what happens here.
fn scan(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let cursor = scan::parse_cursor(args.get(1))?;
    let mut pattern = None;
    let mut count = scan::COUNT;
    let mut ty = None;
    let mut impossible = false;
    let mut i = 2;
    while i < args.len() {
        let rest = args.len() - i;
        if args::is(args.get(i), b"match") && rest >= 2 {
            pattern = Some(args.get(i + 1));
        } else if args::is(args.get(i), b"count") && rest >= 2 {
            // A count under one is a syntax error and not a range error, which
            // is the odder of the two answers and is Redis's.
            count = match args.int(i + 1)? {
                n if n >= 1 => usize::try_from(n).unwrap_or(usize::MAX),
                _ => return Err(args::syntax()),
            };
        } else if args::is(args.get(i), b"type") && rest >= 2 {
            match kind_named(args.get(i + 1)) {
                Some(k) => ty = Some(k),
                None => impossible = true,
            }
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }

    // The keys are written first because the walk is what produces the cursor,
    // and the cursor is moved in front of them afterwards. Nothing goes out
    // before the arguments have all been checked, which is what lets a failed
    // command roll back cleanly. See [`scan::reply`], which is all of that and
    // is shared with the three collection scans.
    scan::reply(out, |out| {
        let mut n = 0;
        let next = db.scan(cursor, count, ty, |key| {
            if impossible || pattern.is_some_and(|p| !glob_matches(p, key)) {
                return;
            }
            out.bulk(key);
            n += 1;
        });
        Ok((next, n))
    })
}

/// `KEYS pattern`.
///
/// One walk of every key in the database with the shard doing nothing else,
/// which is the command's whole reputation and is deserved. It is here because
/// tooling and Redis's own test suite both want it, and `SCAN` is the answer
/// for anything that runs against a database somebody is using.
fn keys(db: &Db, pattern: &[u8], out: &mut Out) {
    let at = out.len();
    let mut n = 0;
    db.keys(|key| {
        if glob_matches(pattern, key) {
            out.bulk(key);
            n += 1;
        }
    });
    out.close_array(at, n);
}

/// The type a `TYPE` option names, or `None` for a word that is not a type.
///
/// Case insensitive, because Redis compares with `strcasecmp` and `TYPE Set`
/// works there.
fn kind_named(arg: &[u8]) -> Option<Kind> {
    [
        Kind::String,
        Kind::Hash,
        Kind::Set,
        Kind::Zset,
        Kind::List,
        Kind::Stream,
    ]
    .into_iter()
    .find(|kind| args::is(arg, kind.name().as_bytes()))
}

/// What Redis says when `COPY` is handed a key a module owns.
///
/// Its own sentence, word for word. A module type with no copy callback is
/// refused there and every module type we answer for is one of those, so the
/// bloom filter, the cuckoo filter and the count min sketch all say this. The
/// exception on a real server is `ReJSON-RL`, which does have the callback and
/// copies, and ours does not yet.
const MODULE_KEY: &str = "not supported for this module key";

/// Whether `key` holds a body the keyspace cannot carry on its own.
///
/// `COPY` and `DUMP` are the two commands that have to ask. Every foreign body
/// is one: none of them has a byte shape here yet, so both would have to make
/// one up.
fn is_foreign(db: &mut Keyspace, key: &[u8]) -> bool {
    db.kind_of(key) == Some(Kind::Foreign)
}

/// What `COPY` says about a source there is no way to duplicate.
///
/// A graph is an engine that lives above the keyspace and there is no generic
/// way to ask one for a copy of itself, which is a decision rather than an
/// oversight: a deep copy of ten million edges is not something a client should
/// get from a command that looks like `SET`. Nothing on a real server answers
/// for a graph, so that one gets our own words. The rest are module types a
/// real server does have an answer for, and it is the refusal above.
fn no_copy(db: &mut Keyspace, key: &[u8]) -> Error {
    match db.type_name(key) {
        Some("graph") => Error::new(Code::Unsupported, "COPY is not supported for a graph"),
        _ => Error::new(Code::Unsupported, MODULE_KEY),
    }
}

/// What `DUMP` says about the same key.
///
/// A real server dumps every one of these, because a module type that can be
/// copied can be serialised and the sketches carry their own reader and writer.
/// Ours has no byte shape for a foreign body yet, so this is a refusal where
/// Redis has a payload, which is D-48 and not a sentence anyone can match.
fn no_dump(db: &mut Keyspace, key: &[u8]) -> Error {
    match db.type_name(key) {
        Some("graph") => Error::new(Code::Unsupported, "DUMP is not supported for a graph"),
        _ => Error::new(
            Code::Unsupported,
            "DUMP is not supported for this module key",
        ),
    }
}

/// `RENAME src dst` and `RENAMENX src dst`.
///
/// Both answer an error for a source that is not there, which is unusual: every
/// other command in this file treats a missing key as an ordinary answer. It is
/// the same sentence for both and it is checked before anything else, so
/// `RENAMENX nope nope` is `no such key` and not zero.
///
/// Two keys, so two stripes, and only sometimes the same one. On the same
/// stripe this is the store's own rename, which moves thirteen bytes and leaves
/// the body where it is. Across two it is a take and an import, which moves the
/// body from one slab to the other and still does not copy it.
fn rename(db: &Db, name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
    let nx = name == "renamenx";
    let (src, dst) = (args.get(1), args.get(2));
    let (from, to) = (db.stripe_of(src), db.stripe_of(dst));
    let done = if from == to {
        db.hold_stripe(from).rename(src, dst, nx)
    } else {
        // Both at once and in stripe order, so the key is never in neither
        // stripe and never in both, whoever else is reading either of them.
        let mut held = db.hold_many([from, to].into_iter());
        rename_across(&mut held, from, to, src, dst, nx)
    };
    let done = done.found()?;
    if nx {
        out.int(i64::from(done == Moved::Ok));
    } else {
        out.ok();
    }
    Ok(())
}

/// A rename whose two keys are not on the same stripe.
///
/// The same three answers the store's own rename gives, decided in the same
/// order: a source that is not there, then a destination that is with `NX`,
/// then the move itself. The two keys cannot be the same key here, because the
/// same key is the same stripe, so the case that answers `Ok` for doing nothing
/// does not arise.
///
/// The destination is asked about before the source is taken, which is not the
/// cost worry it is in `COPY`, since a take does not clone. It is that a take
/// cannot be put back: the body is out of the slab by then and the key is gone.
fn rename_across(
    held: &mut Holds<'_>,
    from: usize,
    to: usize,
    src: &[u8],
    dst: &[u8],
    nx: bool,
) -> Moved {
    if !held.stripe_mut(from).exists(src) {
        return Moved::Missing;
    }
    if nx && held.stripe_mut(to).exists(dst) {
        return Moved::Taken;
    }
    let rec = held
        .stripe_mut(from)
        .take(src)
        .expect("the source was live a line ago");
    held.stripe_mut(to).import(dst, rec);
    Moved::Ok
}

/// `COPY src dst [DB n] [REPLACE]`.
///
/// The order the four ways this can fail are checked in is the order a real
/// server checks them, and it is not the order you would write from scratch.
/// The options are parsed first, so a bad `DB` is refused before anyone asks
/// whether the source exists. Then source and destination being the same thing
/// is an error. Only then does a missing source become the ordinary zero.
///
/// Being the same thing means the same key in the same database, so
/// `COPY a a DB 1` is a real copy and `COPY a a DB 0` from database zero is
/// the error. That is why the check is down here and not next to the argument
/// parsing: it needs to know which database was asked for.
fn copy(dbs: &[Db], at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (src, dst) = (args.get(1), args.get(2));
    let mut into = at;
    let mut replace = false;
    let mut i = 3;
    while i < args.len() {
        let arg = args.get(i);
        if args::is(arg, b"replace") {
            replace = true;
            i += 1;
            continue;
        }
        if !args::is(arg, b"db") || i + 1 >= args.len() {
            return Err(args::syntax());
        }
        // A last one wins rather than a refusal, because `COPY a b DB 1 DB 2`
        // lands in database two on a real server and does not complain.
        let n = args.int(i + 1)?;
        into = usize::try_from(n)
            .ok()
            .filter(|n| *n < dbs.len())
            .ok_or_else(|| Error::new(Code::Invalid, DB_OUT_OF_RANGE))?;
        i += 2;
    }
    if into == at && src == dst {
        return Err(Error::new(Code::Invalid, SAME_OBJECT));
    }
    let (from, to) = (spot(dbs, at, src), spot(dbs, into, dst));
    let done = match hold_both(dbs, from, to) {
        Both::One(mut ks) => match ks.copy(src, dst, replace) {
            // The one `Moved` a caller cannot answer with a number, because
            // zero would mean the destination was taken and this is a source
            // there is no way to duplicate. See `Moved::Unsupported`.
            Moved::Unsupported => return Err(no_copy(&mut ks, src)),
            done => done,
        },
        // Two keyspaces, whether that is two databases or two stripes of one,
        // so the value comes out of the first standing on its own before the
        // second is touched. Both are held for the whole of it, so neither end
        // can move while the record is in the air.
        //
        // The destination is asked about first, which is the opposite order from
        // the single keyspace path above and answers the same thing. Export
        // clones the body, so asking first is the difference between a refused
        // copy of a million member set costing nothing and costing the set.
        Both::Two(mut from, mut to) => {
            if !replace && to.exists(dst) {
                out.int(0);
                return Ok(());
            }
            if is_foreign(&mut from, src) {
                return Err(no_copy(&mut from, src));
            }
            let Some(rec) = from.export(src) else {
                out.int(0);
                return Ok(());
            };
            to.import(dst, rec);
            Moved::Ok
        }
    };
    out.int(i64::from(done == Moved::Ok));
    Ok(())
}

/// Where a key lives: which database, and which stripe of it.
///
/// `COPY` and `MOVE` are the two commands that can name two of these at once,
/// and they are the reason it is a pair rather than a stripe. Two keys in one
/// database on one stripe are one keyspace and everything else is two, and the
/// only way to say that in one line is to compare both halves.
type Spot = (usize, usize);

/// Which database and stripe `key` is on.
fn spot(dbs: &[Db], db: usize, key: &[u8]) -> Spot {
    (db, dbs[db].stripe_of(key))
}

/// The one or two keyspaces two spots name.
enum Both<'d> {
    /// One, because the two spots are the same spot.
    One(Held<'d, Keyspace>),
    /// Two, the source first and the destination second whichever order the
    /// two of them were taken in.
    Two(Held<'d, Keyspace>, Held<'d, Keyspace>),
}

/// Both spots, held.
///
/// The lower spot is taken first whichever of the two it is, so that two
/// clients copying in opposite directions cannot each end up holding what the
/// other is waiting for. The same spot twice is held once, since holding one
/// stripe twice is a wait for yourself.
fn hold_both(dbs: &[Db], from: Spot, to: Spot) -> Both<'_> {
    if from == to {
        return Both::One(dbs[from.0].hold_stripe(from.1));
    }
    if from < to {
        let source = dbs[from.0].hold_stripe(from.1);
        let dest = dbs[to.0].hold_stripe(to.1);
        Both::Two(source, dest)
    } else {
        let dest = dbs[to.0].hold_stripe(to.1);
        let source = dbs[from.0].hold_stripe(from.1);
        Both::Two(source, dest)
    }
}

/// `RESTORE key ttl payload [REPLACE] [ABSTTL] [IDLETIME n] [FREQ n]`.
///
/// The order the four checks happen in is the whole of what a client can see
/// here, and it is not the order they read in. Every option is parsed first,
/// then the busy key, then the ttl, then the footer, then the bytes. So a
/// `RESTORE` at a key that exists is refused before anybody looks at the
/// payload, which matters: whether a key is busy should not depend on whether
/// the bytes behind it happened to be good.
///
/// `IDLETIME` and `FREQ` cannot both be given, and the way Redis says so is
/// worth copying exactly. Neither word is rejected for being the wrong one, they
/// are only accepted while the other has not been seen, so the second of the two
/// falls through to the syntax error rather than getting a message of its own.
///
/// A ttl is milliseconds from now unless `ABSTTL`, in which case it is a unix
/// time, and a zero means no deadline at all in both readings.
fn restore(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let mut db = db.hold(key);
    let mut replace = false;
    let mut absolute = false;
    let mut idle = -1i64;
    let mut freq = -1i64;
    let mut i = 4;
    while i < args.len() {
        let arg = args.get(i);
        let more = args.len() - i - 1;
        if args::is(arg, b"replace") {
            replace = true;
        } else if args::is(arg, b"absttl") {
            absolute = true;
        } else if args::is(arg, b"idletime") && more >= 1 && freq == -1 {
            idle = args.int(i + 1)?;
            if idle < 0 {
                return Err(Error::new(Code::Invalid, BAD_IDLETIME));
            }
            i += 1;
        } else if args::is(arg, b"freq") && more >= 1 && idle == -1 {
            freq = args.int(i + 1)?;
            if !(0..=255).contains(&freq) {
                return Err(Error::new(Code::Invalid, BAD_FREQ));
            }
            i += 1;
        } else {
            return Err(args::syntax());
        }
        i += 1;
    }
    if !replace && db.exists(key) {
        out.error(BUSY_KEY);
        return Ok(());
    }
    let ttl = args.int(2)?;
    if ttl < 0 {
        return Err(Error::new(Code::Invalid, BAD_TTL));
    }
    let now = db.clock().now_ms();
    let ttl = ttl as u64;
    let expire_at = match ttl {
        0 => None,
        // Clamped rather than refused, because Redis does not check this at all
        // and a client that asks for a deadline past the end of the range should
        // get the end of the range and not an error nobody else gives.
        _ if absolute => Some(ttl.min(MAX_AT)),
        _ => Some(now.saturating_add(ttl).min(MAX_AT)),
    };
    // Both numbers are checked and then dropped, which is D-26. They set the
    // eviction metadata on a real server, and the store has readers for that
    // metadata and no writer, so a restored key starts with the idle time and
    // the counter any new key gets. The checking is not wasted work even so,
    // because the four messages above are the whole of what a client can see
    // about these two options today.
    let _ = (idle, freq);
    match db.restore(key, args.get(3), expire_at, replace) {
        Ok(_) => out.ok(),
        Err(Bad::Footer) => return Err(Error::new(Code::Invalid, BAD_FOOTER)),
        Err(Bad::Format) => return Err(Error::new(Code::Invalid, BAD_PAYLOAD)),
    }
    Ok(())
}

/// `MOVE key db`.
///
/// `COPY key key DB n` with the source deleted, except that it does not go
/// through `COPY`, because a move does not need the clone. The store splits a
/// value coming out of a database into two calls for exactly this reason: an
/// export leaves the key where it is and clones the body, and a take pulls the
/// body out of the slab and deletes the key. Moving a set of a million members
/// through the export would build a second set of a million members and then
/// throw the first one away a line later.
///
/// Both failure modes answer zero rather than complaining: a source that is not
/// there, and a destination that is. What is an error is a database index that
/// is not one, and moving a key into the database it is already in, which Redis
/// calls the same object error and which is checked before the keys are looked
/// at at all.
///
/// The destination is asked about first, which is the opposite of the order
/// Redis checks in and answers the same thing both ways round. Here it is not
/// about cost, it is that a take that has to be put back is not something this
/// can do: the body is out of the slab by then and the key is gone.
fn move_key(dbs: &[Db], at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let n = args.int(2)?;
    let into = usize::try_from(n)
        .ok()
        .filter(|n| *n < dbs.len())
        .ok_or_else(|| Error::new(Code::Invalid, DB_OUT_OF_RANGE))?;
    if into == at {
        return Err(Error::new(Code::Invalid, SAME_OBJECT));
    }
    // One key and two databases, so the stripe is the same number twice on a
    // server whose databases are all the same width, which is every server.
    // Worked out from each database even so, because a width that is read from
    // one and assumed of the other is the kind of assumption that holds until
    // somebody adds a knob.
    let (from, to) = (spot(dbs, at, key), spot(dbs, into, key));
    // Two databases, so never one keyspace, which is why this one has no
    // single keyspace path the way `COPY` does.
    let Both::Two(mut from, mut to) = hold_both(dbs, from, to) else {
        unreachable!("a move into the database the key is already in was refused above")
    };
    if to.exists(key) {
        out.int(0);
        return Ok(());
    }
    let Some(rec) = from.take(key) else {
        out.int(0);
        return Ok(());
    };
    to.import(key, rec);
    out.int(1);
    Ok(())
}

/// `SORT key [BY pattern] [LIMIT offset count] [GET pattern ...] [ASC|DESC]
/// [ALPHA] [STORE destination]`, and `SORT_RO` without the last one.
///
/// The options can come in any order and any of them can be given more than
/// once, in which case the last one wins. That is not a design, it is what
/// falls out of Redis's parser being a loop over the arguments with an
/// assignment in each arm, and clients rely on it.
///
/// `ASC` is not stored anywhere because it is the default and its only job is
/// to undo a `DESC` that came before it.
///
/// `GET` is the one option that accumulates rather than replaces, so
/// `GET # GET w_*` asks for two things per element and not for the second one
/// twice. The patterns are collected into a small vector, which is the one
/// allocation this parser makes and only when a `GET` was given at all.
fn sort(db: &Db, name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
    let read_only = name == "sort_ro";
    let key = args.get(1);
    let mut opts = Sort::default();
    let mut get: Vec<&[u8]> = Vec::new();
    let mut store: Option<&[u8]> = None;
    let mut i = 2;
    while i < args.len() {
        let arg = args.get(i);
        let rest = args.len() - i;
        if args::is(arg, b"asc") {
            opts.desc = false;
        } else if args::is(arg, b"desc") {
            opts.desc = true;
        } else if args::is(arg, b"alpha") {
            opts.alpha = true;
        } else if args::is(arg, b"by") && rest >= 2 {
            opts.by = Some(args.get(i + 1));
            i += 2;
            continue;
        } else if args::is(arg, b"get") && rest >= 2 {
            get.push(args.get(i + 1));
            i += 2;
            continue;
        } else if args::is(arg, b"limit") && rest >= 3 {
            opts.limit = Some((args.int(i + 1)?, args.int(i + 2)?));
            i += 3;
            continue;
        } else if args::is(arg, b"store") && rest >= 2 && !read_only {
            store = Some(args.get(i + 1));
            i += 2;
            continue;
        } else {
            // Which is where `SORT_RO k STORE d` lands, because `STORE` is not a
            // word `SORT_RO` knows. Redis answers the same syntax error rather
            // than one that names the option, since to its parser there is no
            // option there to name.
            return Err(args::syntax());
        }
        i += 1;
    }
    opts.get = &get;

    match store {
        Some(dst) => out.int(i64::try_from(db.sort_store(key, dst, &opts)?).unwrap_or(i64::MAX)),
        None => {
            let rows = db.sort(key, &opts)?;
            out.array(rows.len());
            for row in rows {
                match row {
                    Some(v) => out.bulk(&v),
                    None => out.nil(),
                }
            }
        }
    }
    Ok(())
}

/// `EXPIRE`, `PEXPIRE`, `EXPIREAT` and `PEXPIREAT`.
///
/// `key ttl [NX|XX|GT|LT]`, and the four differ only in what the number means:
/// seconds or milliseconds, and counted from now or from the epoch. All four
/// become one absolute millisecond here, which is the same shape the hash field
/// versions take and for the same reason, so there is one place the condition
/// rules are applied and four commands that cannot drift apart.
///
/// The reply is 1 or 0 and it cannot tell you which of the two things happened
/// when it says 1: the deadline went on, or the deadline had already passed and
/// the key went away. The store answers that distinction and the wire throws it
/// away, because that is what Redis puts on the wire.
fn expire(db: &Db, name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut db = db.hold(args.get(1));
    let relative = matches!(name, "expire" | "pexpire");
    let scale = if matches!(name, "pexpire" | "pexpireat") {
        1
    } else {
        SECOND
    };
    let at = moment(args.int(2)?, scale, relative, name, db.clock().now_ms())?;
    let cond = condition(args)?;
    out.int(match db.expire(args.get(1), at, cond) {
        // Nothing there, or the condition said no. Redis does not distinguish.
        Applied::Missing | Applied::NotMet => 0,
        Applied::Ok | Applied::Deleted => 1,
    });
    Ok(())
}

/// The absolute millisecond a client's number means, or why it is not one.
///
/// Unlike the hash field commands, a negative number is fine here and means a
/// moment that has already gone, which deletes the key. `EXPIRE key -1` is a
/// delete on a real server and has been for years, so the only failure is the
/// arithmetic overflowing, which is exactly the check Redis makes.
///
/// The clamp at the top is D-17. Redis holds a key's deadline in a full signed
/// long and takes `PEXPIREAT key 9223372036854775807`. A record here holds it in
/// forty six bits, so anything past that becomes the year 4199, which is not the
/// number the client named and is the same behaviour: the key does not expire.
/// The command succeeds either way and only `PTTL` can tell the difference.
fn moment(by: i64, scale: i64, relative: bool, name: &str, now: u64) -> Result<u64> {
    let ms = by
        .checked_mul(scale)
        .and_then(|ms| {
            if relative {
                ms.checked_add(now as i64)
            } else {
                Some(ms)
            }
        })
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("invalid expire time in '{name}' command"),
            )
        })?;
    // A moment before the epoch is a moment that has gone, so it clamps to zero
    // rather than failing. Zero is in the past for every clock this will ever
    // run against, which is what makes that clamp safe to make silently.
    Ok(ms.clamp(0, MAX_AT as i64) as u64)
}

/// `NX`, `XX`, `GT` and `LT`, of which the four writers take any number.
///
/// It reads as one keyword and it is a set, which is the shape Redis gave it and
/// is worth spelling out. `EXPIRE key 100 XX GT` is legal and means both, the
/// same keyword twice is legal and means once, and two of them that contradict
/// each other are a named error rather than a generic syntax one.
///
/// Only two pairs are legal, `XX GT` and `XX LT`, and `XX GT` says the same
/// thing as `GT` on its own because `GT` already refuses a key with no deadline.
/// So the five states below are the whole of it.
fn condition(args: Args<'_>) -> Result<Cond> {
    let (mut nx, mut xx, mut gt, mut lt) = (false, false, false, false);
    for i in 3..args.len() {
        let arg = args.get(i);
        match arg {
            a if args::is(a, b"nx") => nx = true,
            a if args::is(a, b"xx") => xx = true,
            a if args::is(a, b"gt") => gt = true,
            a if args::is(a, b"lt") => lt = true,
            _ => {
                return Err(Error::new(
                    Code::Invalid,
                    format!("Unsupported option {}", String::from_utf8_lossy(arg)),
                ));
            }
        }
    }
    if nx && (xx || gt || lt) {
        return Err(Error::new(Code::Invalid, NX_WITH_OTHERS));
    }
    if gt && lt {
        return Err(Error::new(Code::Invalid, GT_WITH_LT));
    }
    Ok(match (nx, xx, gt, lt) {
        (true, ..) => Cond::NotSet,
        (_, _, true, _) => Cond::Greater,
        (_, true, _, true) => Cond::LessAndSet,
        (_, _, _, true) => Cond::Less,
        (_, true, ..) => Cond::AlreadySet,
        _ => Cond::Always,
    })
}

/// `TTL`, `PTTL`, `EXPIRETIME` and `PEXPIRETIME`.
///
/// One question in four units. The first pair answer what is left and the second
/// pair answer when it falls due, and both pairs use the same -2 for a key that
/// is not there and -1 for a key with no deadline, so only a real answer is ever
/// converted.
///
/// Both seconds forms round to nearest, so four hundred milliseconds left
/// answers 0 and six hundred answers 1. The hash field versions of these
/// commands round up instead, which is not something anyone would guess and is
/// what 8.10.1 does: `HPEXPIRE h 400 FIELDS 1 f` then `HTTL` answers 1 where
/// `PEXPIRE k 400` then `TTL` answers 0. Two commands, two roundings, checked
/// against a real server rather than reasoned about.
fn ask(db: &Db, name: &str, args: Args<'_>, out: &mut Out) {
    let key = args.get(1);
    let mut db = db.hold(key);
    let now = db.clock().now_ms();
    let asked = db.deadline_of(key);
    let millis = name == "pttl" || name == "pexpiretime";
    let ms = if name == "ttl" || name == "pttl" {
        asked.remaining_ms(now)
    } else {
        match asked {
            Ask::Missing => -2,
            Ask::NoDeadline => -1,
            Ask::At(at) => at as i64,
        }
    };
    out.int(if millis || ms < 0 {
        ms
    } else {
        (ms + SECOND / 2) / SECOND
    });
}

/// `OBJECT ENCODING | REFCOUNT | IDLETIME | FREQ key`, and `OBJECT HELP`.
///
/// `ENCODING` is the one worth having and the other three are here because
/// clients call them without thinking. It is the only window onto the size
/// ladder from outside, so it is the command that says whether the ladder is
/// really Redis's ladder: a set of three numbers has to say `intset` and a hash
/// of two hundred fields has to say `listpack`, at the same counts a real server
/// says them at.
///
/// A missing key answers nil rather than an error, on all four. That reads like
/// a bug and it is what 8.10.1 does, checked rather than assumed, and it used to
/// be `no such key` years ago which is where the confusion comes from.
fn object(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if args::is(sub, b"help") {
        if args.len() != 2 {
            return Err(args::unknown_subcommand(sub, "OBJECT"));
        }
        out.array(OBJECT_HELP.len());
        for line in OBJECT_HELP {
            out.simple(line.as_bytes());
        }
        return Ok(());
    }

    let named = ["encoding", "refcount", "idletime", "freq"]
        .into_iter()
        .find(|n| args::is(sub, n.as_bytes()));
    let Some(named) = named else {
        return Err(args::unknown_subcommand(sub, "OBJECT"));
    };
    // The arity in the table is a minimum, so the count of a subcommand that
    // takes exactly one key is checked here, and the name of the command it
    // complains about is the container and the subcommand joined by a pipe.
    if args.len() != 3 {
        return Err(args::wrong_arity_sub("object", named));
    }
    // The lookup happens before FREQ refuses, so `OBJECT FREQ missing` is a nil
    // and not the policy complaint. Same order as Redis, which reaches for the
    // key first and asks about the policy after.
    // The key is at argument two here, because argument one is the subcommand,
    // and `HELP` above is the one that names no key at all and so has no stripe
    // to be on.
    let key = args.get(2);
    let mut db = db.hold(key);
    if !db.exists(key) {
        out.nil();
        return Ok(());
    }
    match named {
        "encoding" => {
            let name = db
                .encoding_name(key)
                .expect("the key is there, so it has an encoding");
            out.bulk(name.as_bytes());
        }
        // One reference, always. Redis shares the small integers and answers a
        // huge number for those, or it did: 8.10.1 answers 1 for `SET k 123`
        // like it does for everything else, so 1 is the whole answer here.
        "refcount" => out.int(1),
        // Seconds since the key was last used. Asking is not using, so the
        // number does not reset itself by being read, and it counts from the
        // moment the key was written for a key nothing has read since.
        "idletime" => {
            if !db.policy().is_clock() {
                return Err(Error::new(Code::Unsupported, IS_LFU));
            }
            let idle = db.idle_secs(key).expect("the key is there");
            out.int(i64::try_from(idle).unwrap_or(i64::MAX));
        }
        "freq" => {
            if !db.policy().is_lfu() {
                return Err(Error::new(Code::Unsupported, NOT_LFU));
            }
            let freq = db.freq(key).expect("the key is there");
            out.int(i64::from(freq));
        }
        other => unreachable!("no body for object {other}"),
    }
    Ok(())
}
