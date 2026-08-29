//! The string commands, from the wire.
//!
//! Every body here parses its arguments, calls one method on [`Keyspace`], and
//! writes one reply. There is no logic about what a command means in this file,
//! because that logic is in `yo-kv` where the embedded API reaches it too (Y23).
//! What is here is the part that only the wire has: which keyword goes where,
//! which combinations Redis refuses, and which of the two protocols the answer
//! is spelled in.
//!
//! Two rules hold everywhere in this file.
//!
//! Nothing is written to the reply until the arguments are known to be good.
//! The dispatcher rolls the buffer back when a body returns an error, so a body
//! that broke this rule would be caught rather than corrupting the stream, but
//! the rule is what makes the rollback a safety net rather than the mechanism.
//!
//! Nothing allocates. The arguments are slices of the read buffer, the keywords
//! are compared in place, and the numbers are written straight into the reply.
//! A shard thread that allocates aborts, and the four commands M2 is measured
//! on all go through here.
//!
//! The option rules are Redis's, checked against a running 8.8 rather than read
//! off the documentation, because the documentation does not say that `SET k v
//! EX 5 EX 5` is accepted and `SET k v EX 5 PX 5` is not.

use super::args::{self, Args, is, syntax};
use super::table::Spec;
use crate::reply::Out;
use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result, xxh3};
use yo_kv::{Compare, Exists, Expire, IncrEx, IncrExpire, Keyspace, Num, SetOptions, Str};

/// What Redis says when a digest is not sixteen hexadecimal characters.
const BAD_DIGEST: &str = "must be exactly 16 hexadecimal characters";
/// What Redis says when `SETRANGE` is given a negative offset.
const BAD_OFFSET: &str = "offset is out of range";
/// What Redis says when `MSETEX` is given a count it cannot use.
const BAD_NUMKEYS: &str = "invalid numkeys value";
/// What Redis says when `MSETEX`'s count does not match what follows it.
const BAD_PAIRS: &str = "wrong number of key-value pairs";
/// What Redis says when `LCS` is asked for the length and the indexes at once.
const LEN_AND_IDX: &str = "If you want both the length and indexes, please just use IDX.";

/// Run one string command.
///
/// The name has already been looked up and the arity has already been checked,
/// so this matches on the table's own spelling of the name rather than on what
/// the client sent. The match is over twenty six short strings, which compiles
/// to a switch on the length and then a compare; the table grows to about two
/// hundred and fifty commands by M8 and this becomes a jump through an index
/// stored in the [`Spec`], which does not change any of the bodies.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "get" => match db.get(args.get(1))? {
            Some(v) => write_str(out, v),
            None => out.nil(),
        },
        "set" => set(db, args, out)?,
        "getset" => match db.getset(args.get(1), args.get(2))? {
            Some(v) => out.bulk(&v),
            None => out.nil(),
        },
        "getdel" => match db.getdel(args.get(1))? {
            Some(v) => out.bulk(&v),
            None => out.nil(),
        },
        "getex" => getex(db, args, out)?,
        "setnx" => out.int(i64::from(db.setnx(args.get(1), args.get(2))?)),
        "setex" => {
            db.setex(args.get(1), args.int(2)?, args.get(3))?;
            out.ok();
        }
        "psetex" => {
            db.psetex(args.get(1), args.int(2)?, args.get(3))?;
            out.ok();
        }
        "mset" => {
            let n = pair_count(args, "mset")?;
            db.mset(pairs(args, 1, n))?;
            out.ok();
        }
        "msetnx" => {
            let n = pair_count(args, "msetnx")?;
            out.int(i64::from(db.msetnx(pairs(args, 1, n))?));
        }
        "mget" => {
            // One key at a time rather than `Keyspace::mget`, which collects the
            // answers into a `Vec` for a caller that wants them all at once.
            // The wire wants them one at a time and in order, and a `Vec` here
            // would be an allocation per call on a thread that must not.
            out.array(args.len() - 1);
            for i in 1..args.len() {
                match db.mget_one(args.get(i)) {
                    Some(v) => write_str(out, v),
                    None => out.nil(),
                }
            }
        }
        "append" => out.int(count(db.append(args.get(1), args.get(2))?)),
        "strlen" => out.int(count(db.strlen(args.get(1))?)),
        "setrange" => {
            let offset =
                usize::try_from(args.int(2)?).map_err(|_| Error::new(Code::Invalid, BAD_OFFSET))?;
            out.int(count(db.setrange(args.get(1), offset, args.get(3))?));
        }
        // `SUBSTR` is `GETRANGE` under the name it had before 2.0, and Redis
        // still ships both as separate entries in its table.
        "getrange" | "substr" => {
            let (start, end) = (args.int(2)?, args.int(3)?);
            out.bulk(&db.getrange(args.get(1), start, end)?);
        }
        "incr" => out.int(db.incr(args.get(1))?),
        "decr" => out.int(db.decr(args.get(1))?),
        "incrby" => out.int(db.incrby(args.get(1), args.int(2)?)?),
        "decrby" => out.int(db.decrby(args.get(1), args.int(2)?)?),
        // A bulk string on both protocols, not a RESP3 double. Redis has never
        // changed this one and a client that parses the digits would break.
        "incrbyfloat" => out.bulk_double(db.incrbyfloat(args.get(1), args.float(2)?)?),
        "lcs" => lcs(db, args, out)?,
        "msetex" => msetex(db, args, out)?,
        "delex" => delex(db, args, out)?,
        "digest" => match db.digest(args.get(1))? {
            Some(h) => out.bulk(&xxh3::hex(h)),
            None => out.nil(),
        },
        "increx" => increx(db, args, out)?,
        // Unreachable: the dispatcher only sends this function commands whose
        // group is `string`, and every one of those is above. An error rather
        // than a panic, because a table that has grown a row nobody wrote a
        // body for should be a command that does not work and not a server
        // that stops answering.
        _ => return Err(args::unknown_command(args)),
    }
    Ok(())
}

/// A stored value, as the bulk string the client is expecting.
///
/// An int encoded value never has its digits written down anywhere until
/// somebody asks for them, and this is where they get written.
fn write_str(out: &mut Out, v: Str<'_>) {
    match v {
        Str::Int(n) => out.bulk_int(n),
        Str::Bytes(b) => out.bulk(b),
    }
}

/// A length or a count as the integer the reply carries.
///
/// Saturating rather than wrapping. Nothing this counts can reach `i64::MAX`,
/// since a value is capped well below it, and a length that came back wrong is
/// better reported as an implausible number than as a negative one.
fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// How many key and value pairs `MSET` and `MSETNX` were given.
///
/// An even argument count means a value without its key, which Redis reports as
/// a wrong number of arguments even though the arity in its table is satisfied.
fn pair_count(args: Args<'_>, name: &str) -> Result<usize> {
    if args.len() % 2 != 1 {
        return Err(args::wrong_arity(name));
    }
    Ok((args.len() - 1) / 2)
}

/// `count` key and value pairs starting at argument `from`.
///
/// Borrowed straight out of the read buffer and never collected, which is what
/// [`Keyspace::mset`] takes an iterator for.
fn pairs<'a>(
    args: Args<'a>,
    from: usize,
    count: usize,
) -> impl Iterator<Item = (&'a [u8], &'a [u8])> + Clone {
    (0..count).map(move |i| (args.get(from + 2 * i), args.get(from + 2 * i + 1)))
}

// ------------------------------------------------------------------ expiry

/// Which of the four expiration keywords was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    /// `EX`, a number of seconds from now.
    Sec,
    /// `PX`, a number of milliseconds from now.
    Millis,
    /// `EXAT`, an absolute unix second.
    SecAt,
    /// `PXAT`, an absolute unix millisecond.
    MillisAt,
}

impl Unit {
    /// The keyword, or `None` if this argument is not one.
    fn parse(arg: &[u8]) -> Option<Unit> {
        if is(arg, b"EX") {
            Some(Unit::Sec)
        } else if is(arg, b"PX") {
            Some(Unit::Millis)
        } else if is(arg, b"EXAT") {
            Some(Unit::SecAt)
        } else if is(arg, b"PXAT") {
            Some(Unit::MillisAt)
        } else {
            None
        }
    }

    /// Whether the number is counted in seconds.
    fn is_seconds(self) -> bool {
        matches!(self, Unit::Sec | Unit::SecAt)
    }

    /// Whether the number is measured from now rather than from the epoch.
    fn is_relative(self) -> bool {
        matches!(self, Unit::Sec | Unit::Millis)
    }
}

/// The absolute unix millisecond an expiration option means.
///
/// Redis's range rule is one check over the number before it is scaled: zero or
/// less is refused, and a number of seconds that would not fit in a signed
/// millisecond is refused. A deadline in the past is not refused, it is
/// accepted and the key is gone immediately, which is how `SET k v EXAT 1`
/// behaves and is worth keeping since it is how a client deletes on a schedule.
fn deadline(unit: Unit, n: i64, now: u64, name: &str) -> Result<u64> {
    if n <= 0 || (unit.is_seconds() && n > i64::MAX / 1000) {
        return Err(args::invalid_expire(name));
    }
    let ms = if unit.is_seconds() { n * 1000 } else { n };
    // Both halves are positive and neither is anywhere near the top of the
    // range, so the sum cannot overflow: `now` is about 2^41 and `ms` is at
    // most 2^63 divided by a thousand.
    let ms = ms as u64;
    Ok(if unit.is_relative() { now + ms } else { ms })
}

// --------------------------------------------------------------------- SET

/// The option bits `SET` tracks, which are Redis's `OBJ_SET_*` flags.
///
/// A keyword given twice is accepted everywhere in `SET` and the last one wins,
/// which is why these are only ever tested against the keywords they conflict
/// with and never against themselves.
mod bits {
    /// `NX`.
    pub const NX: u16 = 1 << 0;
    /// `XX`.
    pub const XX: u16 = 1 << 1;
    /// `EX`.
    pub const EX: u16 = 1 << 2;
    /// `PX`.
    pub const PX: u16 = 1 << 3;
    /// `EXAT`.
    pub const EXAT: u16 = 1 << 4;
    /// `PXAT`.
    pub const PXAT: u16 = 1 << 5;
    /// `KEEPTTL`.
    pub const KEEPTTL: u16 = 1 << 6;
    /// `PERSIST`, which `GETEX` and `INCREX` take and `SET` does not.
    pub const PERSIST: u16 = 1 << 7;
    /// Any of the four `IF` conditions.
    pub const IF: u16 = 1 << 8;
    /// Every way of naming a deadline.
    pub const ANY_EXPIRE: u16 = EX | PX | EXAT | PXAT;
}

/// The bit an expiration keyword sets.
fn unit_bit(unit: Unit) -> u16 {
    match unit {
        Unit::Sec => bits::EX,
        Unit::Millis => bits::PX,
        Unit::SecAt => bits::EXAT,
        Unit::MillisAt => bits::PXAT,
    }
}

/// `SET key value [NX|XX] [GET] [EX s|PX ms|EXAT ts|PXAT ts|KEEPTTL]
/// [IFEQ v|IFNE v|IFDEQ d|IFDNE d]`.
fn set(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, val) = (args.get(1), args.get(2));
    let mut opts = SetOptions::PLAIN;
    let mut seen = 0u16;
    // The expiration and the condition are remembered as positions and read at
    // the end, because a keyword may be given twice and the last one wins, and
    // because Redis checks the whole option list for syntax before it looks at
    // any of the values.
    let mut expire: Option<(Unit, usize)> = None;
    let mut cond: Option<(&[u8], usize)> = None;
    let mut i = 3;
    while i < args.len() {
        let o = args.get(i);
        if is(o, b"NX") && seen & (bits::XX | bits::IF) == 0 {
            seen |= bits::NX;
            opts = opts.if_missing();
            i += 1;
        } else if is(o, b"XX") && seen & (bits::NX | bits::IF) == 0 {
            seen |= bits::XX;
            opts = opts.if_present();
            i += 1;
        } else if is(o, b"GET") {
            opts = opts.returning();
            i += 1;
        } else if is(o, b"KEEPTTL") && seen & bits::ANY_EXPIRE == 0 {
            seen |= bits::KEEPTTL;
            opts = opts.expiring(Expire::Keep);
            i += 1;
        } else if let Some(u) = Unit::parse(o)
            && seen & (bits::KEEPTTL | (bits::ANY_EXPIRE & !unit_bit(u))) == 0
            && i + 1 < args.len()
        {
            seen |= unit_bit(u);
            expire = Some((u, i + 1));
            i += 2;
        } else if is_condition(o)
            && seen & (bits::NX | bits::XX) == 0
            && cond.is_none_or(|(k, _)| is(o, k))
            && i + 1 < args.len()
        {
            seen |= bits::IF;
            cond = Some((o, i + 1));
            i += 2;
        } else {
            return Err(syntax());
        }
    }

    if let Some((u, at)) = expire {
        let ms = deadline(u, args.int(at)?, db.clock().now_ms(), "set")?;
        opts = opts.expiring(Expire::At(ms));
    }
    if let Some((keyword, at)) = cond {
        opts.compare = Some(condition(keyword, args.get(at))?);
    }

    let done = db.set(key, val, opts)?;
    if opts.get {
        match done.previous {
            Some(v) => out.bulk(&v),
            None => out.nil(),
        }
    } else if done.stored {
        out.ok();
    } else {
        // `SET k v NX` on a key that is there is a null and not an error, on
        // both protocols. It is the RESP2 null bulk string, which is the one
        // `Out::nil` writes.
        out.nil();
    }
    Ok(())
}

/// Whether this argument is one of the four compare and swap keywords.
fn is_condition(arg: &[u8]) -> bool {
    is(arg, b"IFEQ") || is(arg, b"IFNE") || is(arg, b"IFDEQ") || is(arg, b"IFDNE")
}

/// The condition a keyword and its argument mean.
///
/// The digest forms take the sixteen hexadecimal characters `DIGEST` hands out
/// and nothing else, so a client that sends the digest as a number or with a
/// `0x` in front of it gets told exactly what is wrong with it.
fn condition<'a>(keyword: &[u8], arg: &'a [u8]) -> Result<Compare<'a>> {
    if is(keyword, b"IFEQ") {
        Ok(Compare::Equal(arg))
    } else if is(keyword, b"IFNE") {
        Ok(Compare::NotEqual(arg))
    } else {
        let d = xxh3::from_hex(arg).ok_or_else(|| Error::new(Code::Invalid, BAD_DIGEST))?;
        if is(keyword, b"IFDEQ") {
            Ok(Compare::DigestEqual(d))
        } else {
            Ok(Compare::DigestNotEqual(d))
        }
    }
}

// ------------------------------------------------------------------- GETEX

/// `GETEX key [EX s|PX ms|EXAT ts|PXAT ts|PERSIST]`.
fn getex(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let mut seen = 0u16;
    let mut expire: Option<(Unit, usize)> = None;
    let mut i = 2;
    while i < args.len() {
        let o = args.get(i);
        if is(o, b"PERSIST") && seen & bits::ANY_EXPIRE == 0 {
            seen |= bits::PERSIST;
            i += 1;
        } else if let Some(u) = Unit::parse(o)
            && seen & (bits::PERSIST | (bits::ANY_EXPIRE & !unit_bit(u))) == 0
            && i + 1 < args.len()
        {
            seen |= unit_bit(u);
            expire = Some((u, i + 1));
            i += 2;
        } else {
            return Err(syntax());
        }
    }

    // Redis looks the key up before it looks at the expiration value, so
    // `GETEX nosuch EX abc` and `GETEX nosuch EX 0` are both a null rather than
    // an error. The syntax of the option list is still checked first, which is
    // why that loop is above this and not below it.
    if !db.exists(key) {
        out.nil();
        return Ok(());
    }

    let wanted = match expire {
        Some((u, at)) => Expire::At(deadline(u, args.int(at)?, db.clock().now_ms(), "getex")?),
        None if seen & bits::PERSIST != 0 => Expire::Clear,
        None => Expire::Keep,
    };
    match db.getex(key, wanted)? {
        Some(v) => write_str(out, v),
        None => out.nil(),
    }
    Ok(())
}

// ------------------------------------------------------------------ MSETEX

/// `MSETEX numkeys key value [key value ...] [NX|XX]
/// [EX s|PX ms|EXAT ts|PXAT ts|KEEPTTL]`.
fn msetex(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let n = parse_i64(args.get(1))
        .filter(|&n| n > 0)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::new(Code::Invalid, BAD_NUMKEYS))?;
    let end = n
        .checked_mul(2)
        .and_then(|pairs| pairs.checked_add(2))
        .filter(|&end| end <= args.len())
        .ok_or_else(|| Error::new(Code::Invalid, BAD_PAIRS))?;

    let mut seen = 0u16;
    let mut exists = Exists::Always;
    let mut expire = Expire::Clear;
    let mut at: Option<(Unit, usize)> = None;
    let mut i = end;
    while i < args.len() {
        let o = args.get(i);
        if is(o, b"NX") && seen & bits::XX == 0 {
            seen |= bits::NX;
            exists = Exists::IfMissing;
            i += 1;
        } else if is(o, b"XX") && seen & bits::NX == 0 {
            seen |= bits::XX;
            exists = Exists::IfPresent;
            i += 1;
        } else if is(o, b"KEEPTTL") && seen & bits::ANY_EXPIRE == 0 {
            seen |= bits::KEEPTTL;
            expire = Expire::Keep;
            i += 1;
        } else if let Some(u) = Unit::parse(o)
            && seen & (bits::KEEPTTL | (bits::ANY_EXPIRE & !unit_bit(u))) == 0
            && i + 1 < args.len()
        {
            seen |= unit_bit(u);
            at = Some((u, i + 1));
            i += 2;
        } else {
            return Err(syntax());
        }
    }
    if let Some((u, pos)) = at {
        expire = Expire::At(deadline(u, args.int(pos)?, db.clock().now_ms(), "msetex")?);
    }

    out.int(i64::from(db.msetex(pairs(args, 2, n), exists, expire)?));
    Ok(())
}

// ------------------------------------------------------------------- DELEX

/// `DELEX key [IFEQ v|IFNE v|IFDEQ d|IFDNE d]`.
///
/// The arity in the table is a minimum of two, and a real server then refuses
/// anything that is not two or four arguments as a wrong number of them rather
/// than as a syntax error.
fn delex(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 2 && args.len() != 4 {
        return Err(args::wrong_arity("delex"));
    }
    let compare = if args.len() == 4 {
        if !is_condition(args.get(2)) {
            return Err(syntax());
        }
        Some(condition(args.get(2), args.get(3))?)
    } else {
        None
    };
    out.int(i64::from(db.delex(args.get(1), compare)));
    Ok(())
}

// ------------------------------------------------------------------ INCREX

/// `INCREX key [BYINT n|BYFLOAT f] [SATURATE] [LBOUND l] [UBOUND u]
/// [EX s|PX ms|EXAT ts|PXAT ts|PERSIST] [ENX]`.
///
/// Unlike `SET`, `INCREX` refuses a keyword it has already seen. It is a newer
/// command and it was written with a stricter parser, and a client that sends
/// `BYINT 1 BYINT 2` has a bug either way.
fn increx(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    /// `BYINT` or `BYFLOAT`.
    const BY: u16 = 1 << 9;
    /// `SATURATE`.
    const SATURATE: u16 = 1 << 10;
    /// `LBOUND`.
    const LBOUND: u16 = 1 << 11;
    /// `UBOUND`.
    const UBOUND: u16 = 1 << 12;
    /// `ENX`.
    const ENX: u16 = 1 << 13;

    let key = args.get(1);
    let mut seen = 0u16;
    let mut opts = IncrEx::PLAIN;
    // The bounds are read after the loop, because whether they are integers or
    // floats depends on a `BYFLOAT` that may not have been reached yet.
    let mut int_kind = true;
    let mut by: Option<usize> = None;
    let mut lower: Option<usize> = None;
    let mut upper: Option<usize> = None;
    let mut at: Option<(Unit, usize)> = None;
    let mut i = 2;
    while i < args.len() {
        let o = args.get(i);
        if (is(o, b"BYINT") || is(o, b"BYFLOAT")) && seen & BY == 0 && i + 1 < args.len() {
            seen |= BY;
            int_kind = is(o, b"BYINT");
            by = Some(i + 1);
            i += 2;
        } else if is(o, b"SATURATE") && seen & SATURATE == 0 {
            seen |= SATURATE;
            opts = opts.saturating();
            i += 1;
        } else if is(o, b"LBOUND") && seen & LBOUND == 0 && i + 1 < args.len() {
            seen |= LBOUND;
            lower = Some(i + 1);
            i += 2;
        } else if is(o, b"UBOUND") && seen & UBOUND == 0 && i + 1 < args.len() {
            seen |= UBOUND;
            upper = Some(i + 1);
            i += 2;
        } else if is(o, b"PERSIST") && seen & (bits::ANY_EXPIRE | bits::PERSIST) == 0 {
            seen |= bits::PERSIST;
            i += 1;
        } else if is(o, b"ENX") && seen & ENX == 0 {
            seen |= ENX;
            i += 1;
        } else if let Some(u) = Unit::parse(o)
            && seen & (bits::PERSIST | bits::ANY_EXPIRE) == 0
            && i + 1 < args.len()
        {
            seen |= unit_bit(u);
            at = Some((u, i + 1));
            i += 2;
        } else {
            return Err(syntax());
        }
    }
    if seen & ENX != 0 && at.is_none() {
        return Err(Error::new(Code::Invalid, "ENX flag requires an expiration"));
    }

    if let Some(pos) = by {
        opts = opts.by(number(args.get(pos), int_kind, "Increment")?);
    }
    let lower = lower
        .map(|pos| number(args.get(pos), int_kind, "LBOUND"))
        .transpose()?;
    let upper = upper
        .map(|pos| number(args.get(pos), int_kind, "UBOUND"))
        .transpose()?;
    opts = opts.between(lower, upper);
    if let Some((u, pos)) = at {
        let ms = deadline(u, args.int(pos)?, db.clock().now_ms(), "increx")?;
        opts = opts.expiring(if seen & ENX != 0 {
            IncrExpire::AtIfNone(ms)
        } else {
            IncrExpire::At(ms)
        });
    } else if seen & bits::PERSIST != 0 {
        opts = opts.expiring(IncrExpire::Persist);
    }

    let done = db.increx(key, opts)?;
    out.array(2);
    write_num(out, done.value);
    write_num(out, done.applied);
    Ok(())
}

/// An `INCREX` number, in the kind the increment is counted in.
///
/// The message names the option it came from, which is `Increment` for the
/// amount itself, because `ERR value is not an integer` in a command with four
/// numbers in it does not tell the client which one to look at.
fn number(arg: &[u8], int_kind: bool, what: &str) -> Result<Num> {
    if int_kind {
        parse_i64(arg).map(Num::Int).ok_or_else(|| {
            Error::fmt(
                Code::Invalid,
                format_args!("{what} is not an integer or out of range"),
            )
        })
    } else {
        parse_f64(arg)
            .map(Num::Float)
            .ok_or_else(|| Error::fmt(Code::Invalid, format_args!("{what} is not a valid float")))
    }
}

/// An `INCREX` number, as the client sees it.
///
/// The integer form is an integer on both protocols. The float form is a RESP3
/// double and a bulk string of the same digits on RESP2, which is what
/// [`Out::double`] already does.
fn write_num(out: &mut Out, n: Num) {
    match n {
        Num::Int(v) => out.int(v),
        Num::Float(v) => out.double(v),
    }
}

// --------------------------------------------------------------------- LCS

/// `LCS key1 key2 [LEN] [IDX] [MINMATCHLEN n] [WITHMATCHLEN]`.
///
/// `MINMATCHLEN` and `WITHMATCHLEN` without `IDX` are accepted and ignored,
/// which is what a real server does.
fn lcs(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (a, b) = (args.get(1), args.get(2));
    let (mut want_len, mut want_idx, mut with_len) = (false, false, false);
    let mut minmatchlen = 0u32;
    let mut i = 3;
    while i < args.len() {
        let o = args.get(i);
        if is(o, b"LEN") {
            want_len = true;
            i += 1;
        } else if is(o, b"IDX") {
            want_idx = true;
            i += 1;
        } else if is(o, b"WITHMATCHLEN") {
            with_len = true;
            i += 1;
        } else if is(o, b"MINMATCHLEN") && i + 1 < args.len() {
            // A negative minimum is not an error, it is no minimum at all.
            minmatchlen = u32::try_from(args.int(i + 1)?.max(0)).unwrap_or(u32::MAX);
            i += 2;
        } else {
            return Err(syntax());
        }
    }
    if want_len && want_idx {
        return Err(Error::new(Code::Invalid, LEN_AND_IDX));
    }

    if want_idx {
        let idx = db.lcs_idx(a, b, minmatchlen)?;
        // A map on RESP3 and the same pairs flattened into an array on RESP2,
        // which is what `Out::map` writes and what a real server sends.
        out.map(2);
        out.bulk(b"matches");
        out.array(idx.matches.len());
        for m in &idx.matches {
            out.array(if with_len { 3 } else { 2 });
            out.array(2);
            out.int(i64::from(m.a.0));
            out.int(i64::from(m.a.1));
            out.array(2);
            out.int(i64::from(m.b.0));
            out.int(i64::from(m.b.1));
            if with_len {
                out.int(i64::from(m.len));
            }
        }
        out.bulk(b"len");
        out.int(count(idx.len));
    } else if want_len {
        out.int(count(db.lcs_len(a, b)?));
    } else {
        out.bulk(&db.lcs(a, b)?);
    }
    Ok(())
}
