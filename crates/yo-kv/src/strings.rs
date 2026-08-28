//! The string type and its commands.
//!
//! One method per Redis command, taking and returning ordinary Rust values.
//! There is no command enum here and no dispatch: this is the layer the wire
//! calls into and the layer the embedded API calls into, and Y23 says those two
//! have to be the same code rather than two implementations of the same idea.
//! Anything that is about parsing arguments or writing a reply lives above.
//!
//! Errors carry Redis's own message text, because it ends up on the wire
//! verbatim, and a [`Code`] alongside it, because the embedded caller should be
//! matching on a value rather than on a string (P5).

use crate::clock::Clock;
use crate::value::{self, Encoding, Str};
use std::borrow::Cow;
use yo_common::num::parse_i64;
use yo_common::{Code, Error, Result};
use yo_index::RawMap;

/// What Redis says when a value should have been a number and was not.
const NOT_AN_INT: &str = "value is not an integer or out of range";
/// What Redis says when a value should have been a float and was not.
const NOT_A_FLOAT: &str = "value is not a valid float";
/// What Redis says when the result of a counter would leave the range.
const WOULD_OVERFLOW: &str = "increment or decrement would overflow";
/// What Redis says when a write would make a string too long.
const TOO_LONG: &str = "string exceeds maximum allowed size (proto-max-bulk-len)";
/// What we say when a key is longer than this band holds.
const KEY_TOO_LONG: &str = "key exceeds maximum allowed size";
/// What Redis says when an offset is negative or past the end of the world.
const BAD_OFFSET: &str = "offset is out of range";

/// The longest key this band stores.
///
/// Redis's limit is 512 MiB for a key as well as for a value. A key that long is
/// not a key, it is a value in the wrong place, and holding the ceiling down
/// here is what lets [`STRING_MAX`] be a constant rather than a function of the
/// key in hand.
pub const KEY_MAX: usize = 64 * 1024;

/// The largest string this band stores.
///
/// Redis's limit is 512 MiB. Ours is a segment, because a string lives in the
/// arena and the arena hands out at most one segment's worth in one piece. The
/// band above this is the log region (`06` section 2) and lands with tiering in
/// M5, at which point this constant goes up to Redis's. It is a divergence and
/// it is listed as one rather than left for somebody to discover.
pub const STRING_MAX: usize = RawMap::max_record() - RawMap::header_len() - KEY_MAX - 16;

/// Whether a `SET` should go ahead given what is already there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Exists {
    /// Store whatever is there. Plain `SET`.
    #[default]
    Always,
    /// Only if the key is absent. `SET NX`, and `SETNX`.
    IfMissing,
    /// Only if the key is present. `SET XX`.
    IfPresent,
}

/// What a write should do with the key's deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Expire {
    /// Leave the key with no deadline. Plain `SET`, and `GETEX PERSIST`.
    #[default]
    Clear,
    /// Leave whatever deadline was there. `SET KEEPTTL`, and plain `GETEX`.
    Keep,
    /// Expire at this absolute unix millisecond. `EX`, `PX`, `EXAT`, `PXAT`.
    At(u64),
}

/// Everything `SET` can be asked to do beyond storing the value.
#[derive(Debug, Clone, Copy, Default)]
pub struct SetOptions<'a> {
    /// `NX` or `XX`.
    pub exists: Exists,
    /// `EX`, `PX`, `EXAT`, `PXAT` or `KEEPTTL`.
    pub expire: Expire,
    /// `IFEQ`: store only if the current value is exactly these bytes.
    ///
    /// Redis 8.4's compare and set. A missing key never compares equal, so
    /// `IFEQ` on a key that is not there does not store.
    pub compare: Option<&'a [u8]>,
    /// `GET`: hand back what was there, whether or not the write happened.
    pub get: bool,
}

impl<'a> SetOptions<'a> {
    /// No options at all, which is plain `SET`.
    pub const PLAIN: SetOptions<'static> = SetOptions {
        exists: Exists::Always,
        expire: Expire::Clear,
        compare: None,
        get: false,
    };

    /// This, but only if the key is missing.
    #[must_use]
    pub const fn if_missing(mut self) -> SetOptions<'a> {
        self.exists = Exists::IfMissing;
        self
    }

    /// This, but only if the key is present.
    #[must_use]
    pub const fn if_present(mut self) -> SetOptions<'a> {
        self.exists = Exists::IfPresent;
        self
    }

    /// This, with a deadline.
    #[must_use]
    pub const fn expiring(mut self, e: Expire) -> SetOptions<'a> {
        self.expire = e;
        self
    }

    /// This, but only if the current value is exactly `bytes`.
    #[must_use]
    pub const fn if_equal(mut self, bytes: &'a [u8]) -> SetOptions<'a> {
        self.compare = Some(bytes);
        self
    }

    /// This, returning the previous value.
    #[must_use]
    pub const fn returning(mut self) -> SetOptions<'a> {
        self.get = true;
        self
    }
}

/// What a `SET` did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SetOutcome {
    /// Whether the value was written. `NX`, `XX` and `IFEQ` can all say no.
    pub stored: bool,
    /// The previous value, when `GET` was asked for and there was one.
    ///
    /// Owned, because the record it lived in has been freed by the time this is
    /// handed back. `SET ... GET` is the one string command that cannot avoid a
    /// copy, and it is not on the gate list.
    pub previous: Option<Vec<u8>>,
}

/// A shard's strings.
///
/// Not `Sync`, like everything else that hangs off a shard. One of these
/// belongs to one thread and is reached by sending that thread a command.
pub struct Strings {
    map: RawMap,
    clock: Clock,
    /// Keys that were found dead on the way to answering something else.
    expired: u64,
}

impl Strings {
    /// An empty store on the system clock.
    pub fn new() -> Strings {
        Strings::with_clock(Clock::system())
    }

    /// An empty store on a clock of the caller's choosing.
    pub fn with_clock(clock: Clock) -> Strings {
        Strings {
            map: RawMap::new(),
            clock,
            expired: 0,
        }
    }

    /// The clock expiry compares against.
    #[inline]
    pub const fn clock(&self) -> &Clock {
        &self.clock
    }

    /// The clock, to refresh once per turn of the loop.
    #[inline]
    pub const fn clock_mut(&mut self) -> &mut Clock {
        &mut self.clock
    }

    /// The map underneath, for statistics and for compaction.
    #[inline]
    pub const fn map(&self) -> &RawMap {
        &self.map
    }

    /// How many keys are stored, including any that are dead and not yet
    /// noticed. This is Redis's `DBSIZE`, which counts the same way.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether anything is stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Keys reclaimed by running into them after their deadline.
    ///
    /// Redis calls this `expired_keys` in `INFO stats` and counts both lazy and
    /// active expiry into it. Only lazy expiry exists so far, so only lazy
    /// expiry is counted, and the active cycle in the maintenance slice will add
    /// to the same number when it lands (`14` section 1).
    #[inline]
    pub const fn expired_keys(&self) -> u64 {
        self.expired
    }

    /// Bytes held by the index and the arena.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.map.memory_bytes()
    }

    /// Ask the cache for the bucket this key will land in.
    ///
    /// The first of the loop's two walks (`04` section 3) calls this.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        self.map.prefetch(hash);
    }

    /// The hash this store files `key` under.
    #[inline]
    #[must_use]
    pub fn hash_of(key: &[u8]) -> u64 {
        RawMap::hash_of(key)
    }

    // ---------------------------------------------------------------- reading

    /// `GET key`.
    pub fn get(&mut self, key: &[u8]) -> Option<Str<'_>> {
        self.reap(key);
        self.peek(key)
    }

    /// `MGET key [key ...]`.
    ///
    /// Every dead key is reaped first and the whole answer is then read from a
    /// store nobody is going to mutate, which is what lets all of the returned
    /// values borrow from it at once instead of being copied out one at a time.
    pub fn mget<'a>(&'a mut self, keys: &[&[u8]]) -> Vec<Option<Str<'a>>> {
        for k in keys {
            self.reap(k);
        }
        let me: &Strings = self;
        keys.iter().map(|k| me.peek(k)).collect()
    }

    /// `STRLEN key`, which is zero for a key that is not there.
    pub fn strlen(&mut self, key: &[u8]) -> usize {
        self.get(key).map_or(0, |v| v.len())
    }

    /// `EXISTS key`, for one key.
    pub fn exists(&mut self, key: &[u8]) -> bool {
        self.reap(key);
        self.map.contains(key)
    }

    /// `OBJECT ENCODING key`.
    pub fn encoding(&mut self, key: &[u8]) -> Option<Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        Some(value::Meta::from_byte(rec[0]).encoding())
    }

    /// The key's deadline as an absolute unix millisecond, if it has one.
    pub fn expire_at(&mut self, key: &[u8]) -> Option<u64> {
        self.reap(key);
        value::expire_at(self.map.get(key)?)
    }

    /// `GETRANGE key start end`, and `SUBSTR`, which is the same command.
    ///
    /// Both ends are inclusive and both may be negative, counting back from the
    /// end. Everything out of range clamps, and a start past the end gives the
    /// empty string rather than an error, which is Redis's behaviour and not an
    /// oversight in it.
    ///
    /// Borrowed for a string, owned for an integer, because an integer's digits
    /// do not exist anywhere until somebody asks for them.
    pub fn getrange(&mut self, key: &[u8], start: i64, end: i64) -> Cow<'_, [u8]> {
        self.reap(key);
        let Some(v) = self.peek(key) else {
            return Cow::Borrowed(&[]);
        };
        match v {
            Str::Bytes(b) => match range_of(b.len(), start, end) {
                Some((s, e)) => Cow::Borrowed(&b[s..e]),
                None => Cow::Borrowed(&[]),
            },
            Str::Int(n) => {
                let text = Str::Int(n).to_vec();
                match range_of(text.len(), start, end) {
                    Some((s, e)) => Cow::Owned(text[s..e].to_vec()),
                    None => Cow::Owned(Vec::new()),
                }
            }
        }
    }

    // ---------------------------------------------------------------- writing

    /// `SET key value [NX|XX] [GET] [IFEQ old] [EX s|PX ms|EXAT s|PXAT ms|KEEPTTL]`.
    ///
    /// The order the conditions are tested in is Redis's: the key is looked at
    /// once, `NX`, `XX` and `IFEQ` all decide against that one look, and `GET`
    /// reports what was there whether or not the write went ahead.
    pub fn set(&mut self, key: &[u8], val: &[u8], opts: SetOptions<'_>) -> Result<SetOutcome> {
        check_len(key, val.len())?;
        self.reap(key);

        let present = self.map.get(key);
        let mut out = SetOutcome::default();
        if opts.get
            && let Some(rec) = present
        {
            out.previous = Some(value::read(rec).to_vec());
        }
        let allowed = match opts.exists {
            Exists::Always => true,
            Exists::IfMissing => present.is_none(),
            Exists::IfPresent => present.is_some(),
        };
        let matches = match opts.compare {
            // A key that is not there is not equal to anything, including the
            // empty string.
            Some(want) => present.is_some_and(|rec| value::read(rec).eq_bytes(want)),
            None => true,
        };
        if !allowed || !matches {
            return Ok(out);
        }

        let deadline = match opts.expire {
            Expire::Clear => None,
            Expire::At(ms) => Some(ms),
            Expire::Keep => present.and_then(value::expire_at),
        };
        self.store(key, val, deadline);
        out.stored = true;
        Ok(out)
    }

    /// `SET key value`, with nothing else asked for.
    pub fn set_plain(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        check_len(key, val.len())?;
        self.store(key, val, None);
        Ok(())
    }

    /// `SETNX key value`, which answers whether it stored.
    pub fn setnx(&mut self, key: &[u8], val: &[u8]) -> Result<bool> {
        Ok(self.set(key, val, SetOptions::PLAIN.if_missing())?.stored)
    }

    /// `SETEX key seconds value`.
    ///
    /// A zero or negative time to live is an error and not a delete, which is
    /// what Redis does: `SETEX k 0 v` returns `ERR invalid expire time`.
    pub fn setex(&mut self, key: &[u8], seconds: i64, val: &[u8]) -> Result<()> {
        let ms = seconds
            .checked_mul(1000)
            .ok_or_else(|| invalid_expire("SETEX"))?;
        self.psetex(key, ms, val)
    }

    /// `PSETEX key milliseconds value`.
    pub fn psetex(&mut self, key: &[u8], millis: i64, val: &[u8]) -> Result<()> {
        if millis <= 0 {
            return Err(invalid_expire("PSETEX"));
        }
        let at = self.deadline_in(millis, "psetex")?;
        self.set(key, val, SetOptions::PLAIN.expiring(Expire::At(at)))?;
        Ok(())
    }

    /// `GETSET key value`, which is `SET key value GET` without the options.
    pub fn getset(&mut self, key: &[u8], val: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.set(key, val, SetOptions::PLAIN.returning())?.previous)
    }

    /// `GETDEL key`.
    pub fn getdel(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.reap(key);
        let had = self.map.get(key).map(|rec| value::read(rec).to_vec());
        if had.is_some() {
            self.map.del(key);
        }
        had
    }

    /// `GETEX key [EX s|PX ms|EXAT s|PXAT ms|PERSIST]`.
    ///
    /// [`Expire::Keep`] is plain `GETEX`, which reads without touching the
    /// deadline, and [`Expire::Clear`] is `GETEX PERSIST`.
    pub fn getex(&mut self, key: &[u8], expire: Expire) -> Option<Str<'_>> {
        self.reap(key);
        if expire != Expire::Keep {
            let current = self.map.get(key).and_then(value::expire_at);
            let wanted = match expire {
                Expire::At(ms) => Some(ms),
                _ => None,
            };
            if current != wanted && self.map.get(key).is_some() {
                // The value does not change, only the header in front of it, so
                // this reads the value out and writes the whole record back. A
                // deadline that is added or removed changes the record's length,
                // so there is nothing to overwrite in place.
                let rec = self.map.get(key).expect("checked just above");
                let bytes = value::read(rec).to_vec();
                self.store(key, &bytes, wanted);
            }
        }
        self.peek(key)
    }

    /// `DEL key`, for one key. Answers whether it was there.
    pub fn del(&mut self, key: &[u8]) -> bool {
        self.reap(key);
        self.map.del(key)
    }

    /// `MSET key value [key value ...]`.
    ///
    /// Always succeeds, always overwrites, and always clears any deadline the
    /// keys had, which is `SET` without options applied to each pair in turn.
    pub fn mset(&mut self, pairs: &[(&[u8], &[u8])]) -> Result<()> {
        for &(k, v) in pairs {
            check_len(k, v.len())?;
        }
        for &(k, v) in pairs {
            self.store(k, v, None);
        }
        Ok(())
    }

    /// `MSETNX key value [key value ...]`, which stores all of them or none.
    ///
    /// The whole set of keys is checked before anything is written, so a
    /// duplicate key inside one call does not defeat itself.
    pub fn msetnx(&mut self, pairs: &[(&[u8], &[u8])]) -> Result<bool> {
        for &(k, v) in pairs {
            check_len(k, v.len())?;
        }
        for &(k, _) in pairs {
            self.reap(k);
            if self.map.contains(k) {
                return Ok(false);
            }
        }
        for &(k, v) in pairs {
            self.store(k, v, None);
        }
        Ok(true)
    }

    /// `APPEND key value`, answering the new length.
    ///
    /// Appending to a key that is not there creates it, which makes `APPEND` on
    /// an empty key the same as `SET`. Any deadline the key had is kept, which
    /// is Redis's behaviour: `APPEND` is not a fresh `SET`.
    pub fn append(&mut self, key: &[u8], tail: &[u8]) -> Result<usize> {
        self.reap(key);
        let Some(rec) = self.map.get(key) else {
            check_len(key, tail.len())?;
            self.store(key, tail, None);
            return Ok(tail.len());
        };
        let deadline = value::expire_at(rec);
        let mut joined = value::read(rec).to_vec();
        check_len(key, joined.len() + tail.len())?;
        joined.extend_from_slice(tail);
        let len = joined.len();
        self.store_raw(key, &joined, deadline);
        Ok(len)
    }

    /// `SETRANGE key offset value`, answering the new length.
    ///
    /// A write past the end pads with zero bytes, and a write of nothing to a
    /// key that is not there creates nothing and answers zero. Both of those are
    /// Redis's, and both are the kind of edge a client library's test suite
    /// checks.
    pub fn setrange(&mut self, key: &[u8], offset: usize, val: &[u8]) -> Result<usize> {
        self.reap(key);
        if val.is_empty() {
            return Ok(self.peek(key).map_or(0, |v| v.len()));
        }
        let end = offset
            .checked_add(val.len())
            .ok_or_else(|| Error::new(Code::Invalid, BAD_OFFSET))?;
        check_len(key, end)?;

        let (mut bytes, deadline) = match self.map.get(key) {
            Some(rec) => (value::read(rec).to_vec(), value::expire_at(rec)),
            None => (Vec::new(), None),
        };
        if bytes.len() < end {
            bytes.resize(end, 0);
        }
        bytes[offset..end].copy_from_slice(val);
        let len = bytes.len();
        self.store_raw(key, &bytes, deadline);
        Ok(len)
    }

    // --------------------------------------------------------------- counters

    /// `INCR key`.
    #[inline]
    pub fn incr(&mut self, key: &[u8]) -> Result<i64> {
        self.incrby(key, 1)
    }

    /// `DECR key`.
    #[inline]
    pub fn decr(&mut self, key: &[u8]) -> Result<i64> {
        self.decrby(key, 1)
    }

    /// `DECRBY key decrement`.
    ///
    /// Negating first would overflow on `i64::MIN`, which is why the decrement
    /// is carried through as a subtraction rather than turned into an addition.
    pub fn decrby(&mut self, key: &[u8], by: i64) -> Result<i64> {
        self.count(key, by, true)
    }

    /// `INCRBY key increment`, and with an increment of one, `INCR`.
    ///
    /// This is the command the milestone's gate is about, so the path it takes
    /// is worth stating. A key that is already int encoded is one probe, an add
    /// and an eight byte store back into the record the probe landed on. No
    /// arena allocation, no free, no second record, and no rehash. Every other
    /// case falls through to a rewrite, which is what `INCR` on a string that
    /// happens to look like a number costs.
    #[inline]
    pub fn incrby(&mut self, key: &[u8], by: i64) -> Result<i64> {
        self.count(key, by, false)
    }

    fn count(&mut self, key: &[u8], by: i64, subtract: bool) -> Result<i64> {
        check_len(key, 0)?;
        let hash = RawMap::hash_of(key);
        let now = self.clock.now_ms();

        // One probe, and the mutable borrow ends inside this block whichever way
        // it goes, so the slow paths below are free to reallocate.
        let mut current: Option<i64> = None;
        let mut deadline: Option<u64> = None;
        let mut dead = false;
        if let Some(rec) = self.map.value_mut_hashed(hash, key) {
            if value::is_expired(rec, now) {
                dead = true;
            } else {
                deadline = value::expire_at(rec);
                match value::read_int_in_place(rec) {
                    Some((n, at)) => {
                        let next = step(n, by, subtract)?;
                        value::write_int_in_place(rec, at, next);
                        return Ok(next);
                    }
                    None => {
                        current = Some(
                            value::read(rec)
                                .as_int()
                                .ok_or_else(|| Error::new(Code::Invalid, NOT_AN_INT))?,
                        );
                    }
                }
            }
        }

        if dead {
            self.map.del(key);
            self.expired += 1;
            deadline = None;
        }
        let next = step(current.unwrap_or(0), by, subtract)?;
        self.store_int(key, next, deadline);
        Ok(next)
    }

    /// `INCRBYFLOAT key increment`.
    ///
    /// The result is stored as a string, never as an integer, because Redis
    /// stores it with its own formatting and `OBJECT ENCODING` reports `embstr`
    /// afterwards even when the number came out whole.
    pub fn incrbyfloat(&mut self, key: &[u8], by: f64) -> Result<f64> {
        check_len(key, 0)?;
        if !by.is_finite() {
            return Err(Error::new(Code::Invalid, NOT_A_FLOAT));
        }
        self.reap(key);
        let (current, deadline) = match self.map.get(key) {
            Some(rec) => {
                let text = value::read(rec).to_vec();
                let n = parse_f64(&text).ok_or_else(|| Error::new(Code::Invalid, NOT_A_FLOAT))?;
                (n, value::expire_at(rec))
            }
            None => (0.0, None),
        };
        let next = current + by;
        if !next.is_finite() {
            return Err(Error::new(
                Code::Invalid,
                "increment would produce NaN or Infinity",
            ));
        }
        let mut text = Vec::with_capacity(32);
        yo_common::num::push_double(&mut text, next);
        self.store_raw(key, &text, deadline);
        Ok(next)
    }

    // ---------------------------------------------------------------- private

    /// The value under `key` without reaping first.
    ///
    /// Every public read reaps before calling this, so a caller that skips the
    /// reap would be reading a value the clock says is gone.
    #[inline]
    fn peek(&self, key: &[u8]) -> Option<Str<'_>> {
        Some(value::read(self.map.get(key)?))
    }

    /// Drop `key` if its deadline has passed.
    ///
    /// This is lazy expiry and it is half of the story. The other half is the
    /// active cycle in the maintenance slice, which is what stops a key nobody
    /// ever reads again from holding its memory forever (`14` section 1).
    #[inline]
    fn reap(&mut self, key: &[u8]) {
        let now = self.clock.now_ms();
        let dead = self.map.get(key).is_some_and(|r| value::is_expired(r, now));
        if dead {
            self.map.del(key);
            self.expired += 1;
        }
    }

    /// Store `val` under `key`, choosing the encoding from the bytes.
    fn store(&mut self, key: &[u8], val: &[u8], deadline: Option<u64>) {
        let enc = Encoding::of(val);
        let len = value::record_len(enc, val.len(), deadline.is_some());
        self.map.set_with(key, len, |out| {
            value::write_record(out, enc, val, deadline);
        });
    }

    /// Store `val` under `key` as text, never as an integer.
    ///
    /// `APPEND`, `SETRANGE` and `INCRBYFLOAT` all leave a `raw` string behind in
    /// Redis even when the result reads as a number, and `OBJECT ENCODING` is
    /// tested on exactly that.
    fn store_raw(&mut self, key: &[u8], val: &[u8], deadline: Option<u64>) {
        let len = value::record_len(Encoding::Raw, val.len(), deadline.is_some());
        self.map.set_with(key, len, |out| {
            value::write_record(out, Encoding::Raw, val, deadline);
        });
    }

    /// Store an integer the caller already has, without formatting it first.
    fn store_int(&mut self, key: &[u8], n: i64, deadline: Option<u64>) {
        let len = value::record_len(Encoding::Int, 0, deadline.is_some());
        self.map.set_with(key, len, |out| {
            value::write_int_record(out, n, deadline);
        });
    }

    /// `millis` from now, as an absolute unix millisecond.
    fn deadline_in(&self, millis: i64, what: &str) -> Result<u64> {
        u64::try_from(millis)
            .ok()
            .and_then(|ms| self.clock.now_ms().checked_add(ms))
            .ok_or_else(|| invalid_expire(what))
    }
}

impl Default for Strings {
    fn default() -> Strings {
        Strings::new()
    }
}

impl Str<'_> {
    /// Whether this value's string form is exactly `want`.
    ///
    /// Used by `IFEQ`, which compares against what the client would have read.
    /// An int encoded 42 is equal to `"42"` and not to `"042"`, and doing this
    /// without materialising the digits is why the integer arm exists.
    #[inline]
    fn eq_bytes(&self, want: &[u8]) -> bool {
        match self {
            Str::Bytes(b) => *b == want,
            Str::Int(n) => match parse_i64(want) {
                Some(w) => *n == w,
                None => false,
            },
        }
    }
}

/// Add or subtract, refusing to wrap.
#[inline]
fn step(n: i64, by: i64, subtract: bool) -> Result<i64> {
    let r = if subtract {
        n.checked_sub(by)
    } else {
        n.checked_add(by)
    };
    r.ok_or_else(|| Error::new(Code::Invalid, WOULD_OVERFLOW))
}

/// Refuse a key or a value this band cannot hold.
#[inline]
fn check_len(key: &[u8], len: usize) -> Result<()> {
    if key.len() > KEY_MAX {
        return Err(Error::new(Code::Full, KEY_TOO_LONG));
    }
    if len > STRING_MAX {
        return Err(Error::new(Code::Full, TOO_LONG));
    }
    Ok(())
}

fn invalid_expire(what: &str) -> Error {
    Error::new(
        Code::Invalid,
        format!("invalid expire time in '{what}' command"),
    )
}

/// Turn Redis's inclusive, possibly negative range into a half open one.
///
/// Returns `None` when the range selects nothing, which the caller answers with
/// the empty string.
fn range_of(len: usize, start: i64, end: i64) -> Option<(usize, usize)> {
    if len == 0 {
        return None;
    }
    let n = len as i64;
    let clamp = |i: i64| -> i64 { if i < 0 { (n + i).max(0) } else { i.min(n) } };
    let s = clamp(start);
    // The end is inclusive, so one past it is where the slice stops.
    let e = if end < 0 {
        (n + end + 1).max(0)
    } else {
        (end + 1).min(n)
    };
    if s >= e {
        None
    } else {
        Some((s as usize, e as usize))
    }
}

/// Parse a float the way Redis's `getLongDoubleFromObject` does.
///
/// Whitespace, an empty string, `nan` and anything with trailing rubbish are all
/// refused. The infinities are accepted, because `INCRBYFLOAT k inf` is a thing
/// Redis accepts on the way in and refuses on the way out.
fn parse_f64(bytes: &[u8]) -> Option<f64> {
    if bytes.is_empty() || bytes[0].is_ascii_whitespace() {
        return None;
    }
    let text = core::str::from_utf8(bytes).ok()?;
    if text.trim() != text {
        return None;
    }
    let v: f64 = text.parse().ok()?;
    if v.is_nan() { None } else { Some(v) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::EMBSTR_MAX;

    /// A store on a fixed clock, so expiry is a function of what the test does
    /// and not of how long the test takes to run.
    fn store() -> Strings {
        Strings::with_clock(Clock::fixed(1_000))
    }

    fn got(s: &mut Strings, key: &[u8]) -> Option<Vec<u8>> {
        s.get(key).map(|v| v.to_vec())
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut s = store();
        assert_eq!(got(&mut s, b"k"), None);
        s.set_plain(b"k", b"hello").unwrap();
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"hello"[..]));
        assert_eq!(s.strlen(b"k"), 5);
        assert_eq!(s.len(), 1);
        s.set_plain(b"k", b"bye").unwrap();
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"bye"[..]));
        assert_eq!(s.len(), 1, "overwriting made a second key");
    }

    #[test]
    fn a_value_comes_back_exactly_as_it_went_in() {
        let mut s = store();
        for text in [&b""[..], b"0", b"007", b"-0", b"+1", b"9223372036854775808"] {
            s.set_plain(b"k", text).unwrap();
            assert_eq!(got(&mut s, b"k").as_deref(), Some(text), "{text:?}");
        }
    }

    #[test]
    fn object_encoding_matches_redis() {
        let mut s = store();
        s.set_plain(b"n", b"42").unwrap();
        assert_eq!(s.encoding(b"n"), Some(Encoding::Int));
        s.set_plain(b"z", b"007").unwrap();
        assert_eq!(s.encoding(b"z"), Some(Encoding::Embstr));
        s.set_plain(b"e", &[b'x'; EMBSTR_MAX]).unwrap();
        assert_eq!(s.encoding(b"e"), Some(Encoding::Embstr));
        s.set_plain(b"r", &[b'x'; EMBSTR_MAX + 1]).unwrap();
        assert_eq!(s.encoding(b"r"), Some(Encoding::Raw));
        assert_eq!(s.encoding(b"missing"), None);
        // What APPEND leaves behind is raw even though it reads as a number.
        s.set_plain(b"a", b"1").unwrap();
        s.append(b"a", b"2").unwrap();
        assert_eq!(s.encoding(b"a"), Some(Encoding::Raw));
    }

    #[test]
    fn nx_and_xx_decide_against_what_is_there() {
        let mut s = store();
        assert!(
            !s.set(b"k", b"v", SetOptions::PLAIN.if_present())
                .unwrap()
                .stored
        );
        assert_eq!(got(&mut s, b"k"), None);
        assert!(
            s.set(b"k", b"v", SetOptions::PLAIN.if_missing())
                .unwrap()
                .stored
        );
        assert!(
            !s.set(b"k", b"w", SetOptions::PLAIN.if_missing())
                .unwrap()
                .stored
        );
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"v"[..]));
        assert!(
            s.set(b"k", b"w", SetOptions::PLAIN.if_present())
                .unwrap()
                .stored
        );
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"w"[..]));
        assert!(s.setnx(b"fresh", b"1").unwrap());
        assert!(!s.setnx(b"fresh", b"2").unwrap());
    }

    #[test]
    fn ifeq_compares_against_the_string_the_client_would_have_read() {
        let mut s = store();
        // A key that is not there is not equal to anything.
        assert!(
            !s.set(b"k", b"v", SetOptions::PLAIN.if_equal(b""))
                .unwrap()
                .stored
        );
        s.set_plain(b"k", b"42").unwrap();
        assert!(
            !s.set(b"k", b"v", SetOptions::PLAIN.if_equal(b"43"))
                .unwrap()
                .stored
        );
        // Int encoded, so the comparison is against the digits and not the bytes
        // in the record, and "042" is not "42".
        assert!(
            !s.set(b"k", b"v", SetOptions::PLAIN.if_equal(b"042"))
                .unwrap()
                .stored
        );
        assert!(
            s.set(b"k", b"v", SetOptions::PLAIN.if_equal(b"42"))
                .unwrap()
                .stored
        );
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn get_reports_the_old_value_whether_or_not_the_write_happened() {
        let mut s = store();
        assert_eq!(
            s.set(b"k", b"a", SetOptions::PLAIN.returning())
                .unwrap()
                .previous,
            None
        );
        let out = s.set(b"k", b"b", SetOptions::PLAIN.returning()).unwrap();
        assert!(out.stored);
        assert_eq!(out.previous.as_deref(), Some(&b"a"[..]));
        // Refused by NX, and still reports what is there.
        let out = s
            .set(b"k", b"c", SetOptions::PLAIN.if_missing().returning())
            .unwrap();
        assert!(!out.stored);
        assert_eq!(out.previous.as_deref(), Some(&b"b"[..]));
        assert_eq!(s.getset(b"k", b"d").unwrap().as_deref(), Some(&b"b"[..]));
    }

    #[test]
    fn a_key_is_gone_the_millisecond_its_deadline_arrives() {
        let mut s = store();
        s.set(b"k", b"v", SetOptions::PLAIN.expiring(Expire::At(1_500)))
            .unwrap();
        assert_eq!(s.expire_at(b"k"), Some(1_500));
        s.clock_mut().set(1_499);
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"v"[..]));
        s.clock_mut().set(1_500);
        assert_eq!(got(&mut s, b"k"), None);
        assert_eq!(s.len(), 0, "the dead key was not reclaimed");
        assert_eq!(s.expired_keys(), 1);
    }

    #[test]
    fn keepttl_keeps_the_deadline_and_a_plain_set_clears_it() {
        let mut s = store();
        s.set(b"k", b"v", SetOptions::PLAIN.expiring(Expire::At(9_000)))
            .unwrap();
        s.set(b"k", b"w", SetOptions::PLAIN.expiring(Expire::Keep))
            .unwrap();
        assert_eq!(s.expire_at(b"k"), Some(9_000));
        s.set_plain(b"k", b"x").unwrap();
        assert_eq!(s.expire_at(b"k"), None);
    }

    #[test]
    fn setex_refuses_a_time_to_live_that_is_not_one() {
        let mut s = store();
        assert!(s.setex(b"k", 0, b"v").is_err());
        assert!(s.setex(b"k", -1, b"v").is_err());
        assert_eq!(got(&mut s, b"k"), None);
        s.setex(b"k", 10, b"v").unwrap();
        assert_eq!(s.expire_at(b"k"), Some(11_000));
        s.psetex(b"p", 250, b"v").unwrap();
        assert_eq!(s.expire_at(b"p"), Some(1_250));
    }

    #[test]
    fn getex_reads_and_retimes_in_one_go() {
        let mut s = store();
        s.set(b"k", b"v", SetOptions::PLAIN.expiring(Expire::At(5_000)))
            .unwrap();
        // Plain GETEX leaves the deadline alone.
        assert_eq!(
            s.getex(b"k", Expire::Keep).map(|v| v.to_vec()).as_deref(),
            Some(&b"v"[..])
        );
        assert_eq!(s.expire_at(b"k"), Some(5_000));
        // PERSIST clears it.
        assert!(s.getex(b"k", Expire::Clear).is_some());
        assert_eq!(s.expire_at(b"k"), None);
        // And a new deadline replaces it.
        assert!(s.getex(b"k", Expire::At(7_000)).is_some());
        assert_eq!(s.expire_at(b"k"), Some(7_000));
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"v"[..]));
        assert!(s.getex(b"missing", Expire::At(7_000)).is_none());
    }

    #[test]
    fn getdel_hands_the_value_over_and_keeps_nothing() {
        let mut s = store();
        s.set_plain(b"k", b"v").unwrap();
        assert_eq!(s.getdel(b"k").as_deref(), Some(&b"v"[..]));
        assert_eq!(s.getdel(b"k"), None);
        assert_eq!(s.len(), 0);
        s.set_plain(b"k", b"v").unwrap();
        assert!(s.del(b"k"));
        assert!(!s.del(b"k"));
    }

    #[test]
    fn mset_writes_every_pair_and_msetnx_writes_none_of_them() {
        let mut s = store();
        s.mset(&[(&b"a"[..], &b"1"[..]), (&b"b"[..], &b"2"[..])])
            .unwrap();
        let vals = s.mget(&[&b"a"[..], &b"b"[..], &b"missing"[..]]);
        let vals: Vec<_> = vals.iter().map(|v| v.map(|v| v.to_vec())).collect();
        assert_eq!(vals[0].as_deref(), Some(&b"1"[..]));
        assert_eq!(vals[1].as_deref(), Some(&b"2"[..]));
        assert_eq!(vals[2], None);

        assert!(
            !s.msetnx(&[(&b"b"[..], &b"9"[..]), (&b"c"[..], &b"3"[..])])
                .unwrap()
        );
        assert_eq!(got(&mut s, b"c"), None, "msetnx wrote part of the set");
        assert_eq!(got(&mut s, b"b").as_deref(), Some(&b"2"[..]));
        assert!(
            s.msetnx(&[(&b"c"[..], &b"3"[..]), (&b"d"[..], &b"4"[..])])
                .unwrap()
        );
        assert_eq!(got(&mut s, b"d").as_deref(), Some(&b"4"[..]));
    }

    #[test]
    fn mget_reaps_before_it_reads() {
        let mut s = store();
        s.set(b"a", b"1", SetOptions::PLAIN.expiring(Expire::At(1_100)))
            .unwrap();
        s.set_plain(b"b", b"2").unwrap();
        s.clock_mut().set(1_100);
        let vals = s.mget(&[&b"a"[..], &b"b"[..]]);
        assert!(vals[0].is_none(), "a dead key came back from mget");
        assert!(vals[1].is_some());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn append_creates_extends_and_keeps_the_deadline() {
        let mut s = store();
        assert_eq!(s.append(b"k", b"one").unwrap(), 3);
        assert_eq!(s.append(b"k", b" two").unwrap(), 7);
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"one two"[..]));
        s.set(b"t", b"a", SetOptions::PLAIN.expiring(Expire::At(4_000)))
            .unwrap();
        s.append(b"t", b"b").unwrap();
        assert_eq!(s.expire_at(b"t"), Some(4_000));
        assert_eq!(got(&mut s, b"t").as_deref(), Some(&b"ab"[..]));
    }

    #[test]
    fn setrange_pads_with_zero_bytes() {
        let mut s = store();
        assert_eq!(s.setrange(b"k", 0, b"").unwrap(), 0);
        assert_eq!(got(&mut s, b"k"), None, "an empty write created a key");
        assert_eq!(s.setrange(b"k", 3, b"xy").unwrap(), 5);
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"\0\0\0xy"[..]));
        s.set_plain(b"h", b"Hello World").unwrap();
        assert_eq!(s.setrange(b"h", 6, b"Redis").unwrap(), 11);
        assert_eq!(got(&mut s, b"h").as_deref(), Some(&b"Hello Redis"[..]));
    }

    #[test]
    fn getrange_counts_from_both_ends_and_clamps() {
        let mut s = store();
        s.set_plain(b"k", b"This is a string").unwrap();
        assert_eq!(&*s.getrange(b"k", 0, 3), b"This");
        assert_eq!(&*s.getrange(b"k", -3, -1), b"ing");
        assert_eq!(&*s.getrange(b"k", 0, -1), b"This is a string");
        assert_eq!(&*s.getrange(b"k", 10, 100), b"string");
        // A start past the end, and a range that runs backwards, are both empty.
        assert_eq!(&*s.getrange(b"k", 100, 200), b"");
        assert_eq!(&*s.getrange(b"k", 5, 2), b"");
        assert_eq!(&*s.getrange(b"missing", 0, -1), b"");
        // An int encoded value ranges over its digits.
        s.set_plain(b"n", b"12345").unwrap();
        assert_eq!(&*s.getrange(b"n", 1, 3), b"234");
        assert_eq!(&*s.getrange(b"n", 9, 9), b"");
    }

    #[test]
    fn incr_counts_and_refuses_what_is_not_a_number() {
        let mut s = store();
        assert_eq!(s.incr(b"k").unwrap(), 1);
        assert_eq!(s.incr(b"k").unwrap(), 2);
        assert_eq!(s.incrby(b"k", 40).unwrap(), 42);
        assert_eq!(s.decr(b"k").unwrap(), 41);
        assert_eq!(s.decrby(b"k", 41).unwrap(), 0);
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"0"[..]));
        assert_eq!(s.encoding(b"k"), Some(Encoding::Int));

        s.set_plain(b"t", b"hello").unwrap();
        let e = s.incr(b"t").unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), NOT_AN_INT);
        // The refused increment left the value alone.
        assert_eq!(got(&mut s, b"t").as_deref(), Some(&b"hello"[..]));
    }

    #[test]
    fn incr_works_on_a_number_that_is_stored_as_text() {
        let mut s = store();
        // Appending onto an existing key leaves a raw string, which INCR still
        // counts. Appending onto a key that is not there does not, because
        // Redis runs the new value through tryObjectEncoding on create.
        s.append(b"k", b"1").unwrap();
        assert_eq!(s.encoding(b"k"), Some(Encoding::Int));
        s.append(b"k", b"0").unwrap();
        assert_eq!(s.encoding(b"k"), Some(Encoding::Raw));
        assert_eq!(s.incr(b"k").unwrap(), 11);
        assert_eq!(
            s.encoding(b"k"),
            Some(Encoding::Int),
            "INCR did not re-encode"
        );
        // A leading zero is not a number to string2ll, so it is not one here.
        s.set_plain(b"z", b"007").unwrap();
        assert!(s.incr(b"z").is_err());
    }

    #[test]
    fn a_counter_refuses_to_wrap() {
        let mut s = store();
        s.set_plain(b"k", b"9223372036854775807").unwrap();
        let e = s.incr(b"k").unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), WOULD_OVERFLOW);
        assert_eq!(
            got(&mut s, b"k").as_deref(),
            Some(&b"9223372036854775807"[..])
        );
        s.set_plain(b"m", b"-9223372036854775808").unwrap();
        assert!(s.decr(b"m").is_err());
        // Subtracting i64::MIN is the case negating first would get wrong.
        s.set_plain(b"d", b"0").unwrap();
        assert!(s.decrby(b"d", i64::MIN).is_err());
    }

    #[test]
    fn incr_keeps_the_deadline_and_reaps_a_dead_key_first() {
        let mut s = store();
        s.set(b"k", b"5", SetOptions::PLAIN.expiring(Expire::At(2_000)))
            .unwrap();
        assert_eq!(s.incr(b"k").unwrap(), 6);
        assert_eq!(s.expire_at(b"k"), Some(2_000), "the deadline was dropped");
        // Past the deadline, the counter starts again from zero and the key has
        // no deadline any more.
        s.clock_mut().set(2_000);
        assert_eq!(s.incr(b"k").unwrap(), 1);
        assert_eq!(s.expire_at(b"k"), None);
        assert_eq!(s.expired_keys(), 1);
    }

    /// The gate is about this path, so it gets its own test: incrementing an int
    /// encoded value must not touch the arena at all.
    #[test]
    fn incr_on_an_int_does_not_allocate() {
        let mut s = store();
        s.set_plain(b"k", b"1").unwrap();
        let before = s.map().arena().live_bytes();
        for want in 2..1_000 {
            assert_eq!(s.incr(b"k").unwrap(), want);
        }
        assert_eq!(
            s.map().arena().live_bytes(),
            before,
            "INCR moved the record"
        );
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"999"[..]));
    }

    #[test]
    fn incrbyfloat_formats_the_way_redis_does() {
        let mut s = store();
        assert_eq!(s.incrbyfloat(b"k", 10.5).unwrap(), 10.5);
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"10.5"[..]));
        assert_eq!(s.incrbyfloat(b"k", 0.1).unwrap(), 10.6);
        // A whole result is still stored as a string, never as an integer.
        s.set_plain(b"n", b"5").unwrap();
        assert_eq!(s.incrbyfloat(b"n", 1.0).unwrap(), 6.0);
        assert_eq!(got(&mut s, b"n").as_deref(), Some(&b"6"[..]));
        assert_eq!(s.encoding(b"n"), Some(Encoding::Raw));

        s.set_plain(b"t", b"hello").unwrap();
        let e = s.incrbyfloat(b"t", 1.0).unwrap_err();
        assert_eq!(e.message(), NOT_A_FLOAT);
        assert!(s.incrbyfloat(b"k", f64::NAN).is_err());
        assert!(s.incrbyfloat(b"k", f64::INFINITY).is_err());
    }

    #[test]
    fn a_value_that_is_too_long_is_an_error_and_not_a_panic() {
        let mut s = store();
        let huge = vec![b'x'; STRING_MAX + 1];
        let e = s.set_plain(b"k", &huge).unwrap_err();
        assert_eq!(e.code(), Code::Full);
        assert_eq!(e.message(), TOO_LONG);
        assert!(s.append(b"k", &huge).is_err());
        assert!(s.setrange(b"k", STRING_MAX, b"x").is_err());
        let long_key = vec![b'k'; KEY_MAX + 1];
        assert_eq!(s.set_plain(&long_key, b"v").unwrap_err().code(), Code::Full);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn exists_and_strlen_agree_with_get() {
        let mut s = store();
        assert!(!s.exists(b"k"));
        assert_eq!(s.strlen(b"k"), 0);
        s.set(
            b"k",
            b"12345",
            SetOptions::PLAIN.expiring(Expire::At(2_000)),
        )
        .unwrap();
        assert!(s.exists(b"k"));
        assert_eq!(s.strlen(b"k"), 5);
        s.clock_mut().set(2_000);
        assert!(!s.exists(b"k"));
        assert_eq!(s.strlen(b"k"), 0);
    }

    #[test]
    fn the_store_reports_what_it_is_holding() {
        let mut s = Strings::new();
        assert!(s.is_empty());
        assert!(s.memory_bytes() > 0, "an empty index still has buckets");
        s.set_plain(b"k", b"v").unwrap();
        assert!(!s.is_empty());
        assert_eq!(s.len(), 1);
        // The clock is the system one, so it is somewhere after 2020.
        assert!(s.clock().now_ms() > 1_577_836_800_000);
        s.prefetch(Strings::hash_of(b"k"));
        assert_eq!(got(&mut s, b"k").as_deref(), Some(&b"v"[..]));
    }
}
