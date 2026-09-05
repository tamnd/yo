//! The hash commands, on the wire.
//!
//! The same shape as [`super::sets`]: the name has been looked up and the arity
//! has been checked, so this turns arguments into a call on [`Keyspace`] and the
//! answer into a reply. No decisions about hashes are made here and none about
//! representations, because the wire and the embedded API have to reach the same
//! code or there are two implementations of `HSET` and one of them is wrong
//! (Y23).
//!
//! # No allocation on the way in
//!
//! `HSET key f1 v1 f2 v2` hands the store an iterator over pairs of [`Args`]
//! rather than collecting them, and `HMGET` and `HDEL` do the same with their
//! field lists. Nothing on this path allocates whatever the argument count is.
//!
//! # The map reply
//!
//! `HGETALL` answers RESP3's map type, `%`, and not a flat array, and clients
//! act on it: a RESP3 client that gets a `%` hands back a dictionary without
//! being told what command it sent, where on RESP2 it has to know that the
//! reply is alternating and pair it up itself. [`Out::map`] is the one place
//! that knows the difference, and it writes twice the count as an array header
//! on RESP2 so the flat form comes out right.
//!
//! `HRANDFIELD key count WITHVALUES` is the other one, and it is not a map even
//! on RESP3, because a negative count can hand back the same field twice and a
//! map would silently lose one of them. Redis draws the line in the same place.

use yo_common::{Code, Error, Result, glob_matches, parse_i64};
use yo_kv::hash::Text;
use yo_kv::{Applied, Ask, Cond, Db, Exists, Expire, Keyspace, MAX_AT};

use super::args::{self, Args};
use super::indexing::Change;
use super::scan;
use super::table::Spec;
use crate::reply::Out;

/// What Redis says when a scan cursor is not a number.
/// What Redis says when `HINCRBY` is given something that is not an integer.
const NOT_AN_INT: &str = "value is not an integer or out of range";
/// And `HINCRBYFLOAT`.
const NOT_A_FLOAT: &str = "value is not a valid float";
/// What `HSCAN` walks when the client does not say.
/// What the `HEXPIRE` family says about a negative time.
///
/// Not the same message an out of range one gets, which is easy to miss: a
/// negative argument is refused outright rather than being read as a moment that
/// has already passed, so `HEXPIRE key -1` is an error where `HEXPIRE key 0` is
/// a delete.
const BAD_EXPIRE: &str = "invalid expire time, must be >= 0";
/// And what it says when the count of fields does not match the fields.
///
/// Redis leaves the command name off this one where it puts it on every other
/// arity message, which reads like an oversight and is what a client sees.
const FIELD_COUNT: &str = "wrong number of arguments";
/// And when `numfields` is not a count at all.
const BAD_NUMFIELDS: &str = "Parameter `numFields` should be greater than 0";
/// A thousand milliseconds, for the commands that count in seconds.
const SECOND: i64 = 1000;
/// What `HGETDEL` says about a field count that is not one.
///
/// The last three hash commands were added years after the `HEXPIRE` family and
/// they do not agree with it, or with each other, about how to phrase any of
/// this. There are three wordings for the same three mistakes and they are all
/// upstream, so they are all here.
const DEL_BAD_COUNT: &str = "Number of fields must be a positive integer";
/// And when its count does not match the fields.
const DEL_MISMATCH: &str = "The `numfields` parameter must match the number of arguments";
/// And when `FIELDS` is not where it should be.
const DEL_NO_FIELDS: &str = "Mandatory argument FIELDS is missing or not at the right position";
/// What `HGETEX` and `HSETEX` say about a field count that is not one.
const EX_BAD_COUNT: &str = "invalid number of fields";
/// And when their count does not match what follows.
const EX_MISMATCH: &str = "wrong number of arguments";
/// What `HGETEX` says when it is given two ways to set the same deadline.
const GETEX_ONE_OF: &str = "Only one of EX, PX, EXAT, PXAT or PERSIST arguments can be specified";
/// And `HSETEX`, which has `KEEPTTL` where `HGETEX` has `PERSIST`.
const SETEX_ONE_OF: &str = "Only one of EX, PX, EXAT, PXAT or KEEPTTL arguments can be specified";
/// And when it is given both of the field conditions.
const SETEX_ONE_COND: &str = "Only one of FXX or FNX arguments can be specified";

/// Run one hash command.
///
/// Every command in the group names one key and names it first, so the stripe
/// is found once here and everything below goes on taking a keyspace. A hash
/// lives on one stripe whatever is done to it, since nothing here reads a
/// second key.
///
/// What comes back is what a search index needs in order to know that its copy
/// of the document is stale, which is not the same as whether the command was a
/// write: `HSETNX` on a field that is there and `HDEL` of a field that is not
/// both change nothing, and a real server leaves the document alone for both.
/// The field deadline commands are the surprise. Setting or clearing a deadline
/// is not a change by this measure, because the values are the same afterwards,
/// and a real server does not reindex for it. `HEXPIRE key 0` is, because a
/// deadline that has already passed takes the field away.
pub(super) fn execute(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<Change> {
    let mut held = db.hold(args.get(1));
    let db = &mut *held;
    let changed = match spec.name {
        // HSET and HMSET are the same write and differ only in the reply, which
        // is why HMSET has been deprecated since 4.0 and still has to work.
        "hset" | "hmset" => {
            if args.len() < 4 || !args.len().is_multiple_of(2) {
                return Err(args::wrong_arity(spec.name));
            }
            let added = db.hset(args.get(1), pairs(args))?;
            if spec.name == "hset" {
                out.int(count(added));
            } else {
                out.ok();
            }
            // Writing a field the value it already had still counts. A real
            // server throws the document away and reads the key again either
            // way, so the document comes back with a new number, and matching
            // that is the whole point of this answer.
            Change::Fields
        }
        "hsetnx" => {
            let wrote = db.hsetnx(args.get(1), args.get(2), args.get(3))?;
            out.int(i64::from(wrote));
            Change::when(wrote)
        }
        "hget" => {
            db.hget(args.get(1), args.get(2), |t| match t {
                Some(t) => write_text(out, t),
                None => out.nil(),
            })?;
            Change::Nothing
        }
        "hdel" => {
            let gone = db.hdel(args.get(1), fields(args, 2))?;
            out.int(count(gone));
            // `Taken` and not `Fields`, which is the whole of the difference
            // between the two: `HDEL` of the last fields is a key the indexes
            // go to read and do not find, and they count that.
            Change::taken(gone > 0)
        }
        "hlen" => {
            out.int(count(db.hlen(args.get(1))?));
            Change::Nothing
        }
        "hexists" => {
            out.int(i64::from(db.hexists(args.get(1), args.get(2))?));
            Change::Nothing
        }
        "hstrlen" => {
            out.int(count(db.hstrlen(args.get(1), args.get(2))?));
            Change::Nothing
        }
        "hmget" => {
            out.array(args.len() - 2);
            db.hmget(args.get(1), fields(args, 2), |t| match t {
                Some(t) => write_text(out, t),
                None => out.nil(),
            })?;
            Change::Nothing
        }
        // The three walks. They go through `with_hash` rather than through
        // `Keyspace::hgetall`, because every one of them needs the count for its
        // header before it needs the pairs, and asking `HLEN` for it would be a
        // second key lookup on the commands most likely to be in a loop.
        //
        // The header goes out inside the callback and not in front of the call,
        // because `with_hash` is where WRONGTYPE is decided and nothing should
        // be written before that is known.
        "hgetall" => {
            db.with_hash(args.get(1), |hash| match hash {
                Some(h) => {
                    out.map(h.len());
                    for (field, value) in h.iter() {
                        write_text(out, field);
                        write_text(out, value);
                    }
                }
                // A key that is not there is the empty hash and not a nil,
                // which is Redis's answer and what makes iterating the reply
                // safe with no check in front of it.
                None => out.map(0),
            })?;
            Change::Nothing
        }
        "hkeys" | "hvals" => {
            let want_keys = spec.name == "hkeys";
            db.with_hash(args.get(1), |hash| match hash {
                Some(h) => {
                    out.array(h.len());
                    for (field, value) in h.iter() {
                        write_text(out, if want_keys { field } else { value });
                    }
                }
                None => out.array(0),
            })?;
            Change::Nothing
        }
        "hincrby" => {
            out.int(db.hincrby(args.get(1), args.get(2), incr_int(args.get(3))?)?);
            Change::Fields
        }
        // HINCRBYFLOAT answers a bulk string and not a double, on RESP3 as well
        // as RESP2. Redis never changed it, because the exact digits are the
        // point: a client that got a double back would have to trust its own
        // formatting to round trip, and this way the server's formatting is
        // what the client sees and what the next read returns.
        "hincrbyfloat" => {
            let by = incr_float(args.get(3))?;
            out.human_double(db.hincrbyfloat(args.get(1), args.get(2), by)?);
            Change::Fields
        }
        "hrandfield" => {
            randfield(db, args, out)?;
            Change::Nothing
        }
        "hscan" => {
            scan(db, args, out)?;
            Change::Nothing
        }
        // The field TTL family. All four setters turn into one absolute
        // millisecond and all five readers turn into one question, which is why
        // there are two helpers here and not nine.
        "hexpire" | "hpexpire" | "hexpireat" | "hpexpireat" => expire(db, spec.name, args, out)?,
        "httl" | "hpttl" | "hexpiretime" | "hpexpiretime" | "hpersist" => {
            ask(db, spec.name, args, out)?;
            Change::Nothing
        }
        "hgetdel" => getdel(db, args, out)?,
        "hgetex" => {
            getex(db, args, out)?;
            Change::Nothing
        }
        "hsetex" => setex(db, args, out)?,
        other => unreachable!("{other} is not a hash command"),
    };
    Ok(changed)
}

/// `HRANDFIELD key [count [WITHVALUES]]`.
///
/// Three shapes. Without a count it is one field or a nil. With one it is an
/// array of fields, and with `WITHVALUES` on RESP3 it is an array of two element
/// arrays while on RESP2 it is one flat list. That split is Redis's and it is
/// the only place in the hash group where the two protocols disagree about the
/// shape rather than about the type.
fn randfield(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    match args.len() {
        2 => db.hrandfield(args.get(1), |pair| match pair {
            Some((field, _)) => write_text(out, field),
            None => out.nil(),
        })?,
        3 | 4 => {
            let with_values = match args.len() {
                4 if args::is(args.get(3), b"withvalues") => true,
                4 => return Err(args::syntax()),
                _ => false,
            };
            let n = args.int(2)?;
            // WITHVALUES nests each pair on RESP3 and flattens it on RESP2,
            // which is the one place in this group where the two protocols
            // disagree about the shape rather than about the type. It is not a
            // map on either, because a negative count can hand the same field
            // back twice and a map would silently lose one of them.
            let nested = with_values && out.proto().is_resp3();
            // The reply is written before its own header because a positive
            // count is capped at the size of the hash and the walk is what
            // finds out how many that is.
            let start = out.len();
            let mut written = 0;
            db.hrandfield_n(args.get(1), n, |field, value| {
                if nested {
                    out.array(2);
                }
                write_text(out, field);
                written += 1;
                if with_values {
                    write_text(out, value);
                    if !nested {
                        written += 1;
                    }
                }
            })?;
            out.close_array(start, written);
        }
        _ => return Err(args::syntax()),
    }
    Ok(())
}

/// `HSCAN key cursor [MATCH pattern] [COUNT count] [NOVALUES]`.
///
/// The reply is a cursor and then the pairs, and the pairs are written before
/// their own header because `MATCH` decides how many there are as it goes.
///
/// `NOVALUES` is Redis 7.4 and it is not a shortcut for `HKEYS`: it still walks
/// in windows, so it is the only way to page through the fields of a large hash
/// without also pulling every value across the wire.
///
/// `MATCH` matches the field and never the value, which is what a client would
/// assume and worth stating because the same walk has both in hand.
fn scan(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let cursor = scan::parse_cursor(args.get(2))?;
    let mut pattern = None;
    let mut count = scan::COUNT;
    let mut novalues = false;
    let mut i = 3;
    while i < args.len() {
        let rest = args.len() - i;
        if args::is(args.get(i), b"match") && rest >= 2 {
            pattern = Some(args.get(i + 1));
        } else if args::is(args.get(i), b"count") && rest >= 2 {
            // A count under one is a syntax error and not a range error, which
            // is the odder of the two answers and so the one worth copying
            // exactly.
            count = match args.int(i + 1)? {
                n if n >= 1 => usize::try_from(n).unwrap_or(usize::MAX),
                _ => return Err(args::syntax()),
            };
        } else if args::is(args.get(i), b"novalues") {
            novalues = true;
            i += 1;
            continue;
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }

    // Nothing goes out until every argument has been checked, which is what
    // lets the dispatcher roll a failed command back cleanly.
    scan::reply(out, |out| {
        let mut n = 0;
        let next = db.hscan(args.get(1), cursor, count, |field, value| {
            if !matches(pattern, field) {
                return;
            }
            write_text(out, field);
            n += 1;
            if !novalues {
                write_text(out, value);
                n += 1;
            }
        })?;
        Ok((next, n))
    })
}

/// `HEXPIRE`, `HPEXPIRE`, `HEXPIREAT` and `HPEXPIREAT`.
///
/// `key ttl [NX|XX|GT|LT] FIELDS numfields field [field ...]`, and the four
/// differ only in what the number means: seconds or milliseconds, and from now
/// or from the epoch. All four become one absolute millisecond here, so the
/// store has one method and the condition rules are applied in one place.
///
/// Redis refuses a negative number outright rather than reading it as a moment
/// that has already gone, so `HEXPIRE key -1` is an error while `HEXPIRE key 0`
/// deletes the field. Both messages are its, and they are different messages.
/// What comes back is whether a field went, which is the only outcome of the
/// four that a search index cares about. Setting a deadline for later leaves the
/// values alone, and a real server does not reread the key for it.
fn expire(db: &mut Keyspace, name: &str, args: Args<'_>, out: &mut Out) -> Result<Change> {
    // The p is milliseconds and the at is absolute, which is the whole of the
    // difference between the four.
    let relative = matches!(name, "hexpire" | "hpexpire");
    let scale = if matches!(name, "hpexpire" | "hpexpireat") {
        1
    } else {
        SECOND
    };
    let at = moment(args.int(2)?, scale, relative, name, db.clock().now_ms())?;
    let (cond, from) = condition(args, 3)?;
    let fields = field_list(args, from, name)?;

    out.array(fields.len());
    let mut gone = false;
    db.hexpire(args.get(1), at, cond, fields.iter(args), |applied| {
        gone |= applied == Applied::Deleted;
        out.int(applied as i64);
    })?;
    Ok(Change::when(gone))
}

/// The absolute millisecond a client's number means, or why it is not one.
///
/// `scale` turns the unit into milliseconds and `relative` says whether it
/// counts from now or from the epoch, so all eight commands that take a deadline
/// come through here.
///
/// The two failures are different messages and it is easy to get them backwards.
/// A negative number is refused outright rather than read as a moment that has
/// gone, so `HEXPIRE key -1` is an error where `HEXPIRE key 0` is a delete. A
/// number past the ceiling names the command instead. Both are answered before
/// any field is touched, because Redis refuses the whole command rather than
/// expiring the fields it got to first.
fn moment(by: i64, scale: i64, relative: bool, name: &str, now: u64) -> Result<u64> {
    if by < 0 {
        return Err(Error::new(Code::Invalid, BAD_EXPIRE));
    }
    by.checked_mul(scale)
        .and_then(|ms| {
            if relative {
                ms.checked_add(now as i64)
            } else {
                Some(ms)
            }
        })
        .and_then(|ms| u64::try_from(ms).ok())
        .filter(|&ms| ms <= MAX_AT)
        .ok_or_else(|| out_of_range(name))
}

/// `HTTL`, `HPTTL`, `HEXPIRETIME`, `HPEXPIRETIME` and `HPERSIST`.
///
/// `key FIELDS numfields field [field ...]` for all five. The first four are the
/// same question answered in four units, and `HPERSIST` asks the same question
/// and then takes the deadline off, which is why it is here and not with the
/// writers: its reply is built out of the same three cases.
fn ask(db: &mut Keyspace, name: &str, args: Args<'_>, out: &mut Out) -> Result<()> {
    let fields = field_list(args, 2, name)?;
    out.array(fields.len());
    let now = db.clock().now_ms();
    if name == "hpersist" {
        return db.hpersist(args.get(1), fields.iter(args), |asked| {
            out.int(match asked {
                Ask::Missing => -2,
                Ask::NoDeadline => -1,
                // Redis replies 1 for a deadline taken off and does not say
                // what it was, so the moment is dropped here.
                Ask::At(_) => 1,
            });
        });
    }
    // What is left against when it falls due, and seconds against
    // milliseconds. The two sentinels, -2 and -1, are the same in every unit,
    // so only a real answer is converted.
    let left = name == "httl" || name == "hpttl";
    let millis = name == "hpttl" || name == "hpexpiretime";
    db.httl(args.get(1), fields.iter(args), |asked| {
        let ms = if left {
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
            // Rounded up, so a field with half a second left answers 1 and not
            // 0, which is what Redis does and is the answer a client can act
            // on. i64::div_ceil is still unstable, and this arm has already
            // established that the number is not negative.
            (ms + SECOND - 1) / SECOND
        });
    })
}

/// `HGETDEL key FIELDS numfields field [field ...]`.
///
/// The value goes out and the field goes away, which a client could not do
/// without a race before this existed. The reply is positional the way `HMGET`'s
/// is, so a field that was not there is a nil in its own place.
fn getdel(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<Change> {
    if !args::is(args.get(2), b"fields") {
        return Err(Error::new(Code::Invalid, DEL_NO_FIELDS));
    }
    let fields = ex_field_list(args, 2, 1, DEL_BAD_COUNT, DEL_MISMATCH)?;
    out.array(fields.len());
    // A value handed back is a field taken away, so the reply is also the
    // answer to whether the key changed.
    let mut took = false;
    db.hgetdel(args.get(1), fields.iter(args), |t| match t {
        Some(t) => {
            took = true;
            write_text(out, t);
        }
        None => out.nil(),
    })?;
    Ok(Change::when(took))
}

/// `HGETEX key [EX s | PX ms | EXAT ts | PXAT ts | PERSIST] FIELDS n f [f ...]`.
///
/// A plain `HGETEX` with no option leaves the deadline where it is, which is the
/// one place this disagrees with `GETEX`, and it is why [`Expire::Keep`] is the
/// default here rather than [`Expire::Clear`].
fn getex(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let opts = options(db.clock().now_ms(), args, "hgetex")?;
    let fields = ex_field_list(args, opts.fields_at, 1, EX_BAD_COUNT, EX_MISMATCH)?;
    out.array(fields.len());
    db.hgetex(args.get(1), opts.expire, fields.iter(args), |t| match t {
        Some(t) => write_text(out, t),
        None => out.nil(),
    })
}

/// `HSETEX key [FNX|FXX] [EX .. | KEEPTTL] FIELDS n field value [field value]`.
///
/// One integer back, and it is all of it or none of it. The count after `FIELDS`
/// is a count of pairs and not of arguments, which is the only place in the hash
/// group where that word means something other than one argument each.
///
/// A deadline that has already passed writes the fields and takes them away
/// again, and a search index hears about both, so one command moves `max_doc_id`
/// twice and the value it was handed is never indexed at all.
fn setex(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<Change> {
    let now = db.clock().now_ms();
    let opts = options(now, args, "hsetex")?;
    let fields = ex_field_list(args, opts.fields_at, 2, EX_BAD_COUNT, EX_MISMATCH)?;
    let wrote = db.hsetex(args.get(1), opts.exists, opts.expire, fields.pairs(args))?;
    out.int(i64::from(wrote));
    let past = matches!(opts.expire, Expire::At(at) if at <= now);
    Ok(if wrote && past {
        Change::Twice
    } else {
        Change::when(wrote)
    })
}

/// What `HGETEX` and `HSETEX` were asked for, and where their fields start.
#[derive(Debug, Clone, Copy)]
struct Options {
    /// `FNX` or `FXX`, which only `HSETEX` has.
    exists: Exists,
    /// The deadline clause, whichever of the six spellings it arrived in.
    expire: Expire,
    /// The index of the `FIELDS` keyword.
    fields_at: usize,
}

/// Read the option clause of `HGETEX` or `HSETEX`, up to `FIELDS`.
///
/// One loop rather than a fixed order, because Redis takes these in any order:
/// `HSETEX key EX 100 FNX FIELDS 1 f v` is accepted and so is the other way
/// round. An option it does not know is reported by name, which is these two
/// commands' wording and not the `HEXPIRE` family's.
///
/// The two differ in one token each, `PERSIST` against `KEEPTTL`, and they say
/// so in their own words when given two deadlines. Neither accepts the other's
/// token: `HGETEX key KEEPTTL` is an unknown argument and not a syntax error.
fn options(now: u64, args: Args<'_>, name: &str) -> Result<Options> {
    let setting = name == "hsetex";
    let mut opts = Options {
        exists: Exists::Always,
        // A write clears the deadline unless told otherwise, and a read leaves
        // it alone unless told otherwise. Two commands, two defaults.
        expire: if setting { Expire::Clear } else { Expire::Keep },
        fields_at: 0,
    };
    let mut had_expire = false;
    let mut had_exists = false;
    let one_of = if setting { SETEX_ONE_OF } else { GETEX_ONE_OF };

    let mut i = 2;
    while i < args.len() {
        let arg = args.get(i);
        if args::is(arg, b"fields") {
            opts.fields_at = i;
            return Ok(opts);
        }
        // The unit and where it counts from, which is all four of these.
        let clause = match arg {
            a if args::is(a, b"ex") => Some((SECOND, true)),
            a if args::is(a, b"px") => Some((1, true)),
            a if args::is(a, b"exat") => Some((SECOND, false)),
            a if args::is(a, b"pxat") => Some((1, false)),
            _ => None,
        };
        if let Some((scale, relative)) = clause {
            if had_expire {
                return Err(Error::new(Code::Invalid, one_of));
            }
            had_expire = true;
            opts.expire = Expire::At(moment(args.int(i + 1)?, scale, relative, name, now)?);
            i += 2;
            continue;
        }
        if (setting && args::is(arg, b"keepttl")) || (!setting && args::is(arg, b"persist")) {
            if had_expire {
                return Err(Error::new(Code::Invalid, one_of));
            }
            had_expire = true;
            // KEEPTTL on a write and PERSIST on a read are opposites of each
            // other and both are the other command's default, which reads
            // oddly until you notice that a write clears and a read does not.
            opts.expire = if setting { Expire::Keep } else { Expire::Clear };
            i += 1;
            continue;
        }
        if setting && (args::is(arg, b"fnx") || args::is(arg, b"fxx")) {
            if had_exists {
                return Err(Error::new(Code::Invalid, SETEX_ONE_COND));
            }
            had_exists = true;
            opts.exists = if args::is(arg, b"fnx") {
                Exists::IfMissing
            } else {
                Exists::IfPresent
            };
            i += 1;
            continue;
        }
        return Err(unknown(arg));
    }
    // Every argument read and no FIELDS in them, which the arity check cannot
    // catch because the option clause has no fixed width.
    Err(args::wrong_arity(name))
}

/// `ERR unknown argument: x`, spelled the way the client sent it.
fn unknown(arg: &[u8]) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!("unknown argument: {}", String::from_utf8_lossy(arg)),
    )
}

/// Read `FIELDS numfields ...` for the last three hash commands.
///
/// `step` is how many arguments a field takes, which is one everywhere except
/// `HSETEX`, where the count is a count of pairs.
///
/// The messages are handed in because these three do not agree with the
/// `HEXPIRE` family or with each other about how to word either mistake. That is
/// upstream and it is what a client sees, so it is copied rather than tidied.
fn ex_field_list(
    args: Args<'_>,
    at: usize,
    step: usize,
    bad_count: &str,
    mismatch: &str,
) -> Result<Fields> {
    let Some(len) = parse_i64(args.get(at + 1))
        .filter(|&n| n > 0)
        .and_then(|n| usize::try_from(n).ok())
    else {
        return Err(Error::new(Code::Invalid, bad_count));
    };
    let from = at + 2;
    // Checked, because a count near the top of a usize would wrap into a width
    // that happens to match and take the walk past the end of the arguments.
    if len.checked_mul(step).and_then(|w| from.checked_add(w)) != Some(args.len()) {
        return Err(Error::new(Code::Invalid, mismatch));
    }
    Ok(Fields { from, len })
}

/// Where the fields of a field TTL command are, once `FIELDS numfields` is read.
///
/// The count has to match what follows exactly. Redis checks it and refuses the
/// command, which matters more than it looks: a client that miscounted has sent
/// a command that would otherwise silently expire the wrong number of fields.
#[derive(Debug, Clone, Copy)]
struct Fields {
    from: usize,
    len: usize,
}

impl Fields {
    #[inline]
    const fn len(self) -> usize {
        self.len
    }

    #[inline]
    fn iter(self, args: Args<'_>) -> impl Iterator<Item = &[u8]> {
        (self.from..self.from + self.len).map(move |i| args.get(i))
    }

    /// The same run read as pairs, which is what `HSETEX` counts.
    ///
    /// Clone, because the store walks the pairs once to check `FNX` or `FXX`
    /// over the whole list before it writes any of them.
    #[inline]
    fn pairs(self, args: Args<'_>) -> impl Iterator<Item = (&[u8], &[u8])> + Clone {
        (0..self.len).map(move |k| {
            let at = self.from + k * 2;
            (args.get(at), args.get(at + 1))
        })
    }
}

/// Read `FIELDS numfields field [field ...]` starting at `at`.
fn field_list(args: Args<'_>, at: usize, name: &str) -> Result<Fields> {
    if !args::is(args.get(at), b"fields") {
        return Err(args::wrong_arity(name));
    }
    let n = args.int(at + 1)?;
    if n < 1 {
        return Err(Error::new(Code::Invalid, BAD_NUMFIELDS));
    }
    let len = usize::try_from(n).unwrap_or(usize::MAX);
    let from = at + 2;
    if args.len() != from + len {
        return Err(Error::new(Code::Invalid, FIELD_COUNT));
    }
    Ok(Fields { from, len })
}

/// The optional `NX`, `XX`, `GT` or `LT`, and where the rest starts.
fn condition(args: Args<'_>, at: usize) -> Result<(Cond, usize)> {
    let cond = match args.get(at) {
        a if args::is(a, b"nx") => Cond::NotSet,
        a if args::is(a, b"xx") => Cond::AlreadySet,
        a if args::is(a, b"gt") => Cond::Greater,
        a if args::is(a, b"lt") => Cond::Less,
        _ => return Ok((Cond::Always, at)),
    };
    Ok((cond, at + 1))
}

/// `ERR invalid expire time in 'x' command`, which is the out of range one.
fn out_of_range(name: &str) -> Error {
    Error::new(
        Code::Invalid,
        format!("invalid expire time in '{name}' command"),
    )
}

/// The field and value pairs of an `HSET`, without collecting them.
///
/// Argument one is the key, so the first field is two and the first value is
/// three. The walk steps on the value index and reads the field behind it,
/// which is what keeps the two from drifting apart when the arity check moves.
#[inline]
fn pairs(args: Args<'_>) -> impl Iterator<Item = (&[u8], &[u8])> + Clone {
    (3..args.len())
        .step_by(2)
        .map(move |i| (args.get(i - 1), args.get(i)))
}

/// The field list of an `HDEL` or an `HMGET`, without collecting it.
#[inline]
fn fields(args: Args<'_>, from: usize) -> impl Iterator<Item = &[u8]> {
    (from..args.len()).map(move |i| args.get(i))
}

/// A field or a value, written where it lies.
///
/// An integer held as one is formatted straight into the reply buffer and never
/// stored as digits anywhere, which is Y18 and is what keeps a hash of counters
/// at the size it is.
#[inline]
fn write_text(out: &mut Out, t: Text<'_>) {
    match t {
        Text::Int(n) => out.bulk_int(n),
        Text::Str(s) => out.bulk(s),
    }
}

/// An `HINCRBY` amount.
///
/// The message is the string command's and not the hash's. Redis uses the value
/// sentence for a bad argument and the hash sentence only for a field that
/// already holds the wrong thing, which reads backwards until you notice that
/// the argument is not a hash value yet.
fn incr_int(arg: &[u8]) -> Result<i64> {
    parse_i64(arg).ok_or_else(|| Error::new(Code::Invalid, NOT_AN_INT))
}

/// An `HINCRBYFLOAT` amount, on the same rule.
fn incr_float(arg: &[u8]) -> Result<f64> {
    yo_common::num::parse_f64(arg).ok_or_else(|| Error::new(Code::Invalid, NOT_A_FLOAT))
}

/// Whether a field passes the `MATCH` pattern, if there is one.
#[inline]
fn matches(pattern: Option<&[u8]>, field: Text<'_>) -> bool {
    let Some(p) = pattern else {
        return true;
    };
    match field {
        Text::Str(s) => glob_matches(p, s),
        // A field held as an integer has to be written out to be matched, which
        // is the one place a number becomes digits before the reply. It is on
        // the MATCH path only, and only for the numeric fields, so the common
        // HSCAN pays nothing for it.
        Text::Int(n) => {
            let mut buf = [0u8; yo_common::num::DIGITS_MAX];
            glob_matches(p, yo_common::num::i64_digits(&mut buf, n))
        }
    }
}

/// A count as the integer the reply carries.
#[inline]
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}
