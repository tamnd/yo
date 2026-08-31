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
use yo_common::{Code, Error, Result, glob_matches};
use yo_kv::{Applied, Ask, Cond, Keyspace, Kind, MAX_AT, Moved};

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

/// What Redis says when `NX` is given alongside any of the other three.
const NX_WITH_OTHERS: &str = "NX and XX, GT or LT options at the same time are not compatible";

/// And when `GT` and `LT` are given together, which is the one pair that
/// contradicts itself without `NX` being involved.
const GT_WITH_LT: &str = "GT and LT options at the same time are not compatible";

/// What `OBJECT FREQ` says on a server that is not counting accesses.
///
/// Which is every server here, because there is no eviction yet, so this is the
/// only thing it can say. Redis says it too whenever the policy is not an LFU
/// one, and the second half about switching at runtime is upstream's wording
/// and not ours.
const NOT_LFU: &str = "An LFU maxmemory policy is not selected, access frequency not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust.";

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
pub(super) fn execute(
    dbs: &mut [Keyspace],
    at: usize,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    if spec.name == "copy" {
        return copy(dbs, at, args, out);
    }
    let db = &mut dbs[at];
    match spec.name {
        // `UNLINK` is `DEL` with the freeing moved to a background thread on a
        // real server. Ours frees on the spot, which is what `UNLINK` promises
        // a client: the key is gone from the keyspace when the reply arrives.
        // The promise is about visibility and not about which thread did the
        // work, so this is the same body rather than a divergence.
        "del" | "unlink" => {
            let mut gone = 0i64;
            for i in 1..args.len() {
                if db.del(args.get(i)) {
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
                if db.exists(args.get(i)) {
                    found += 1;
                }
            }
            out.int(found);
        }
        "type" => {
            let name = match db.kind_of(args.get(1)) {
                Some(k) => k.name().as_bytes(),
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
        "touch" => out.int(db.touch((1..args.len()).map(|i| args.get(i))) as i64),
        "rename" | "renamenx" => rename(db, spec.name, args, out)?,
        "expire" | "pexpire" | "expireat" | "pexpireat" => expire(db, spec.name, args, out)?,
        "persist" => out.int(i64::from(db.persist(args.get(1)))),
        "ttl" | "pttl" | "expiretime" | "pexpiretime" => ask(db, spec.name, args, out),
        "object" => object(db, args, out)?,
        "scan" => scan(db, args, out)?,
        "keys" => keys(db, args.get(1), out),
        "randomkey" => match db.random_key() {
            Some(key) => out.bulk(key),
            None => out.nil(),
        },
        other => unreachable!("keyspace command with no body: {other}"),
    }
    Ok(())
}

/// `SCAN cursor [MATCH pattern] [COUNT count] [TYPE type]`.
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
fn scan(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
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
fn keys(db: &mut Keyspace, pattern: &[u8], out: &mut Out) {
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

/// A cursor as the client sent it back.
///
/// Unsigned, because ours uses the top bits and Redis parses a cursor with
/// `strtoull` too.
/// `RENAME src dst` and `RENAMENX src dst`.
///
/// Both answer an error for a source that is not there, which is unusual: every
/// other command in this file treats a missing key as an ordinary answer. It is
/// the same sentence for both and it is checked before anything else, so
/// `RENAMENX nope nope` is `no such key` and not zero.
fn rename(db: &mut Keyspace, name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
    let nx = name == "renamenx";
    let done = db.rename(args.get(1), args.get(2), nx).found()?;
    if nx {
        out.int(i64::from(done == Moved::Ok));
    } else {
        out.ok();
    }
    Ok(())
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
fn copy(dbs: &mut [Keyspace], at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
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
    let done = if into == at {
        dbs[at].copy(src, dst, replace)
    } else {
        // Two databases, so the value comes out of one standing on its own
        // before the other is touched. The borrow of the first ends with the
        // export, which is the reason the pair exists as two calls.
        //
        // The destination is asked about first, which is the opposite order from
        // the single database path above and answers the same thing. Export
        // clones the body, so asking first is the difference between a refused
        // copy of a million member set costing nothing and costing the set.
        if !replace && dbs[into].exists(dst) {
            out.int(0);
            return Ok(());
        }
        let Some(rec) = dbs[at].export(src) else {
            out.int(0);
            return Ok(());
        };
        dbs[into].import(dst, rec);
        Moved::Ok
    };
    out.int(i64::from(done == Moved::Ok));
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
fn expire(db: &mut Keyspace, name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
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
fn ask(db: &mut Keyspace, name: &str, args: Args<'_>, out: &mut Out) {
    let now = db.clock().now_ms();
    let asked = db.deadline_of(args.get(1));
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
fn object(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
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
    let key = args.get(2);
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
        // Nothing is tracking access time, and zero is what a key just touched
        // would say anyway, which is every key by the time this reaches it.
        "idletime" => out.int(0),
        "freq" => return Err(Error::new(Code::Unsupported, NOT_LFU)),
        other => unreachable!("no body for object {other}"),
    }
    Ok(())
}
