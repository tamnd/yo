//! The bitmap commands, which are string commands wearing a different hat.
//!
//! A bitmap in Redis is a string, and that is not an implementation detail a
//! caller can ignore: `SET k "A"` then `GETBIT k 1` answers 1, because `A` is
//! `0x41` and the second bit from the top of that byte is set. So there is no
//! bitmap type here either, and everything in this file works on the same
//! string records [`strings`](crate::strings) writes. The kernels are in
//! [`bits`]; this is where a key turns into bytes, where a write
//! is allowed to grow a value and where Redis's edges live.
//!
//! Three of those edges are worth stating up front, because all three have been
//! measured on a real server rather than reasoned about.
//!
//! A write always leaves the value `raw`. `SET n 12345` reports `int` and a
//! `SETBIT n 0 0` that changes nothing at all still reports `raw` afterwards,
//! because Redis unshares the object before it looks at a bit. A read does not:
//! `GETBIT n 3` on the same key leaves it `int`. That is why the in place fast
//! path below only takes a record that is already raw.
//!
//! A write creates the key and pads it with zero bytes, even when the bit being
//! written is zero and the byte is past the end. `SETBIT nokey 0 0` on an empty
//! database leaves a one byte string behind.
//!
//! A `BITFIELD` is checked all the way through before any of it runs, so a bad
//! field type in the last subcommand leaves the key untouched and, if it was not
//! there, uncreated. That ordering is the wire layer's to keep, and it is why
//! [`Keyspace::bitfield`] takes a list of already parsed subcommands rather than
//! words to parse.

use crate::bits::{self, Field, Op, Overflow};
use crate::db::Db;
use crate::keyspace::Keyspace;
use crate::strings::{STRING_MAX, check_len};
use crate::value::{self, Kind, Str};
use yo_common::num::{self, DIGITS_MAX};
use yo_common::{Code, Error, Result};
use yo_index::RawMap;

/// What Redis says about an offset that is not a number or is off the end.
const BAD_BIT_OFFSET: &str = "bit offset is not an integer or out of range";
/// What Redis says when a write would make a string too long.
const TOO_LONG: &str = "string exceeds maximum allowed size (proto-max-bulk-len)";

/// The highest bit `SETBIT` and `GETBIT` take.
///
/// It is 4 Gi bits, which is 512 MiB, which is Redis's string ceiling. Ours is a
/// segment and smaller than that, so a write between the two limits is refused
/// by the length check with the "string exceeds maximum allowed size" sentence
/// rather than by this one. Both are Redis's own sentences and the boundary
/// between them is where we diverge.
pub const BIT_OFFSET_MAX: u64 = 4 * 1024 * 1024 * 1024 - 1;

/// Whether a range's two ends count bytes or bits.
///
/// `BITCOUNT` and `BITPOS` both take an optional `BYTE` or `BIT` word after
/// their two indexes, and both default to `BYTE`. The word is only allowed once
/// both indexes are there: `BITPOS k 0 5 BIT` is not a bit ranged search from
/// bit five, it is an error, because `BIT` is read as the end index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Unit {
    /// Indexes count bytes. The default.
    #[default]
    Byte,
    /// Indexes count bits.
    Bit,
}

/// One `BITFIELD` subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sub {
    /// Which of the three it is, and what it carries.
    pub op: SubOp,
    /// The width and signedness of the field.
    pub field: Field,
    /// Where the field starts, in bits.
    ///
    /// The `#n` form a client can send is `n` times the width, and multiplying
    /// it out is the wire layer's job.
    pub at: u64,
    /// What to do if the value will not fit. Ignored by `GET`.
    pub on: Overflow,
}

/// The three things a `BITFIELD` subcommand does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubOp {
    /// `GET`, which never writes and never creates the key.
    Get,
    /// `SET`, answering the value that was there before.
    Set(i64),
    /// `INCRBY`, answering the value afterwards.
    Incr(i64),
}

impl SubOp {
    /// Whether this one writes, which is what decides how far the value grows.
    const fn writes(self) -> bool {
        !matches!(self, SubOp::Get)
    }
}

impl Keyspace {
    /// `GETBIT key offset`.
    ///
    /// A missing key, and any offset past the end of a key that is there, read
    /// as zero. Nothing is created and nothing is re-encoded.
    pub fn getbit(&mut self, key: &[u8], offset: u64) -> Result<bool> {
        if offset > BIT_OFFSET_MAX {
            return Err(Error::new(Code::Invalid, BAD_BIT_OFFSET));
        }
        self.reap(key);
        self.string_only(key)?;
        // A bitmap is a string, so it can have been demoted like any other, and
        // the bit being asked about is somewhere in it. Warmed rather than
        // thawed: reading a bit out of a cold bitmap is a read like any other
        // and the doorkeeper decides whether it earns its way back.
        self.warm(key)?;
        let mut digits = [0u8; DIGITS_MAX];
        let bytes = self.bitmap(key, &mut digits);
        let byte = (offset / 8) as usize;
        Ok(bytes.get(byte).is_some_and(|b| b & mask(offset) != 0))
    }

    /// `SETBIT key offset value`, answering the bit that was there before.
    ///
    /// The value grows to hold the offset, padded with zero bytes, and keeps
    /// whatever deadline it had. A key that was not there is created, even when
    /// the bit being written is zero.
    pub fn setbit(&mut self, key: &[u8], offset: u64, bit: bool) -> Result<bool> {
        if offset > BIT_OFFSET_MAX {
            return Err(Error::new(Code::Invalid, BAD_BIT_OFFSET));
        }
        let byte = (offset / 8) as usize;
        check_len(key, byte + 1)?;
        self.thaw(key)?;
        let now = self.clock.now_ms();
        let hash = RawMap::hash_of(key);

        // The fast path: the key is there, it is raw already, and the byte is
        // inside it, so the write is one probe and one byte. This is the shape a
        // bitmap is used in, a fixed size map of ids that was sized once and is
        // written to for the rest of its life, and it is the only path that does
        // not touch the arena. The kind check sits inside the probe for the
        // reason `INCR`'s does: the byte holding it is already loaded here.
        let mut dead = false;
        if let Some(rec) = self.map.value_mut_hashed(hash, key) {
            if value::kind(rec) != Kind::String {
                return Err(crate::keyspace::wrong_type());
            }
            if value::is_expired(rec, now) {
                dead = true;
            } else if let Some(b) = value::raw_in_place(rec).and_then(|it| it.get_mut(byte)) {
                let had = *b & mask(offset) != 0;
                if bit {
                    *b |= mask(offset);
                } else {
                    *b &= !mask(offset);
                }
                return Ok(had);
            }
        }
        if dead {
            self.drop_key(key);
            self.expired += 1;
        }

        // The slow path, which is every first write to a key and every write
        // that makes it longer. Through the one scratch buffer, the way `APPEND`
        // and `SETRANGE` go, since the old bytes are needed in hand while
        // `store_raw` wants the database.
        let mut bytes = std::mem::take(&mut self.scratch);
        bytes.clear();
        let deadline = match self.map.get(key) {
            Some(rec) => {
                value::read(rec).write_to(&mut bytes);
                value::expire_at(rec)
            }
            None => None,
        };
        if bytes.len() <= byte {
            bytes.resize(byte + 1, 0);
        }
        let had = bytes[byte] & mask(offset) != 0;
        if bit {
            bytes[byte] |= mask(offset);
        } else {
            bytes[byte] &= !mask(offset);
        }
        self.store_raw(key, &bytes, deadline);
        self.scratch = bytes;
        Ok(had)
    }

    /// `BITCOUNT key [start end [BYTE | BIT]]`.
    ///
    /// A missing key, an empty string and a range that ends before it starts all
    /// answer zero. The two indexes may be negative, counting from the end, and
    /// both are clamped rather than refused.
    pub fn bitcount(&mut self, key: &[u8], range: Option<(i64, i64, Unit)>) -> Result<u64> {
        self.reap(key);
        self.string_only(key)?;
        self.warm(key)?;
        let mut digits = [0u8; DIGITS_MAX];
        let bytes = self.bitmap(key, &mut digits);
        let Some((start, end, unit)) = range else {
            return Ok(bits::count(bytes));
        };
        match window(bytes.len(), start, end, unit) {
            Some((from, to)) => Ok(bits::count_range(bytes, from, to)),
            None => Ok(0),
        }
    }

    /// `BITPOS key bit [start [end [BYTE | BIT]]]`.
    ///
    /// Answers minus one when there is no such bit, with the one exception Redis
    /// carved out: looking for a zero with no end index given, over a range that
    /// is all ones, answers the first bit past the end of the string. The idea is
    /// that a string is followed by an infinity of zeros unless the caller said
    /// where to stop. Giving an explicit end turns that back into minus one, and
    /// so does asking about a range that is empty once it has been clamped.
    pub fn bitpos(
        &mut self,
        key: &[u8],
        bit: bool,
        start: Option<i64>,
        end: Option<i64>,
        unit: Unit,
    ) -> Result<i64> {
        self.reap(key);
        self.string_only(key)?;
        self.warm(key)?;
        let here = self.map.get(key).is_some();
        let mut digits = [0u8; DIGITS_MAX];
        let bytes = self.bitmap(key, &mut digits);
        if bytes.is_empty() {
            // A missing key is all zeros, so a zero is at bit nought and a one is
            // nowhere. An empty string that is really there answers minus one
            // either way, since there is no bit nought to point at.
            return Ok(if !bit && !here { 0 } else { -1 });
        }
        let all = bytes.len() as u64 * 8;
        let (from, to) = match (start, end) {
            (None, _) => (0, all),
            (Some(s), None) => match window(bytes.len(), s, -1, unit) {
                Some(r) => r,
                None => return Ok(-1),
            },
            (Some(s), Some(e)) => match window(bytes.len(), s, e, unit) {
                Some(r) => r,
                None => return Ok(-1),
            },
        };
        match bits::find(bytes, bit, from, to) {
            Some(at) => Ok(at as i64),
            None if !bit && end.is_none() => Ok(all as i64),
            None => Ok(-1),
        }
    }

    /// `BITOP op dest src [src ...]`, answering the length of the result.
    ///
    /// A result with no bytes in it deletes the destination, and any other
    /// result creates it whatever it holds, so a `BITOP AND` over sources that
    /// share nothing leaves a destination full of zero bytes rather than no
    /// destination at all. Sources that are shorter than the longest read as
    /// zeros past their end, and a source that is not there reads as empty.
    ///
    /// # Panics
    ///
    /// If `srcs` is empty, or holds more than one key for [`Op::Not`]. Both are
    /// refused with a message on the wire before this is called.
    pub fn bitop<'k, I>(&mut self, op: Op, dest: &[u8], srcs: I) -> Result<usize>
    where
        I: Iterator<Item = &'k [u8]> + Clone,
    {
        for src in srcs.clone() {
            self.reap(src);
            self.string_only(src)?;
            // Every source at once, so every one of them has to be in memory
            // rather than in the one buffer a fault serves out of. `BITOP` over
            // demoted sources brings them back, which is also what a client
            // running it in a loop wants.
            self.thaw(src)?;
        }
        // The sources have to be copied out before the destination can be
        // written, since they are borrowed from the map and the write wants the
        // database back. They go end to end into the scratch buffer with their
        // boundaries in `rows`, and the result goes on the end of the same
        // buffer, so a `BITOP` over any number of sources is one buffer and no
        // allocation past whatever growing that buffer costs.
        let mut flat = std::mem::take(&mut self.scratch);
        let mut ends = std::mem::take(&mut self.rows);
        flat.clear();
        ends.clear();
        let mut digits = [0u8; DIGITS_MAX];
        for src in srcs.clone() {
            let bytes = self.bitmap(src, &mut digits);
            flat.extend_from_slice(bytes);
            ends.push(flat.len());
        }
        // As long as the longest source, `NOT` included: complementing a source
        // cannot make it longer, and there is only ever the one of them.
        let len = bits::width(parts(&flat, &ends));
        if len > STRING_MAX {
            self.scratch = flat;
            self.rows = ends;
            return Err(Error::new(Code::Invalid, TOO_LONG));
        }

        let split = flat.len();
        flat.resize(split + len, 0);
        // The sources and the destination are in the same buffer, so they have
        // to be split apart before one can be read while the other is written.
        let (read, write) = flat.split_at_mut(split);
        bits::combine(op, parts(read, &ends), write);

        let outcome = if len == 0 {
            self.del(dest);
            Ok(0)
        } else {
            self.reap(dest);
            match self.string_only(dest) {
                Ok(()) => {
                    self.store_raw(dest, &flat[split..], None);
                    Ok(len)
                }
                Err(e) => Err(e),
            }
        };
        self.scratch = flat;
        self.rows = ends;
        outcome
    }

    /// `BITFIELD key [subcommand ...]`, answering one reply per subcommand.
    ///
    /// A `None` in the answers is the nil an `OVERFLOW FAIL` subcommand gives
    /// when its value would not fit; that one does not write and the ones around
    /// it still do. The subcommands are expected to have been checked already,
    /// which is what makes it safe for this to be the point of no return.
    ///
    /// The value grows once, before anything runs, to hold the last bit any
    /// writing subcommand touches. That happens even if every one of those
    /// writes then fails its overflow check, which is Redis's behaviour and
    /// falls out of it growing the string before it looks at the values.
    pub fn bitfield(&mut self, key: &[u8], ops: &[Sub]) -> Result<Vec<Option<i64>>> {
        let grow = ops.iter().filter(|s| s.op.writes()).map(reach).max();
        self.bitfield_with(key, grow, |bytes| {
            ops.iter().map(|&sub| apply(bytes, sub)).collect()
        })
    }

    /// `BITFIELD`, with the subcommands run against the value in place.
    ///
    /// This is the form the wire uses. It hands over the bytes and lets the
    /// caller walk its own arguments a second time, calling [`apply`] on each,
    /// which is what lets a `BITFIELD` with two hundred subcommands write two
    /// hundred replies without a list of them existing anywhere.
    ///
    /// `grow` is how many bytes the value has to reach, which is the last byte
    /// any writing subcommand touches, and `None` for a call that only reads.
    /// The growing happens once and before anything runs, even if every one of
    /// those writes then fails its overflow check, because that is what Redis
    /// does: it makes the string long enough while it is looking up the key and
    /// only then starts on the values. A call that only reads stores nothing,
    /// which is what keeps `BITFIELD k GET u8 0` from turning an `embstr` into a
    /// `raw`.
    pub fn bitfield_with<T>(
        &mut self,
        key: &[u8],
        grow: Option<usize>,
        run: impl FnOnce(&mut [u8]) -> T,
    ) -> Result<T> {
        self.reap(key);
        self.string_only(key)?;
        // Every path here materialises the value and most of them write it
        // back, so this thaws rather than asking the doorkeeper about a value
        // that is going to be resident when the command ends anyway.
        self.thaw(key)?;
        let need = grow.unwrap_or(0);
        check_len(key, need)?;

        // Every path materialises the value, including the read only one, so
        // that an int encoded key reads as the digits it prints as.
        let mut bytes = std::mem::take(&mut self.scratch);
        bytes.clear();
        let deadline = match self.map.get(key) {
            Some(rec) => {
                value::read(rec).write_to(&mut bytes);
                value::expire_at(rec)
            }
            None => None,
        };
        if bytes.len() < need {
            bytes.resize(need, 0);
        }
        let out = run(&mut bytes);
        if grow.is_some() {
            self.store_raw(key, &bytes, deadline);
        }
        self.scratch = bytes;
        Ok(out)
    }

    /// The bytes of a string key, as the bit commands want to see them.
    ///
    /// A missing key is empty, which is what every one of these commands treats
    /// it as. An int encoded key is the digits it would print as, because that
    /// is the string it is: `SET n 65` then `GETBIT n 1` is asking about the
    /// character `6`. The digits are written into the caller's buffer so that the
    /// ordinary case, a raw string, is still a borrow and not a copy.
    fn bitmap<'a>(&'a self, key: &[u8], digits: &'a mut [u8; DIGITS_MAX]) -> &'a [u8] {
        match self.peek(key) {
            None => &[],
            Some(Str::Bytes(b)) => b,
            Some(Str::Int(n)) => num::i64_digits(digits, n),
        }
    }
}

impl Db {
    /// `BITOP op dest src [src ...]` over a database of any width.
    ///
    /// Every key on one stripe is that one stripe's `BITOP`, which is every
    /// `BITOP` on a database of one stripe and every `BITOP` whose keys were
    /// hash tagged into the same place. That path is the old one, byte for byte.
    ///
    /// The rest is the same work with the reads spread out. The sources are
    /// copied out of the stripes they are on, one at a time, into a buffer this
    /// database owns rather than one a stripe owns, since no stripe can be held
    /// while the next one is being read. They are combined there and the result
    /// is written to whichever stripe the destination is on.
    ///
    /// # Panics
    ///
    /// As [`Keyspace::bitop`].
    pub fn bitop<'k, I>(&mut self, op: Op, dest: &'k [u8], srcs: I) -> Result<usize>
    where
        I: Iterator<Item = &'k [u8]> + Clone,
    {
        if let Some(home) = self.one_stripe(std::iter::once(dest).chain(srcs.clone())) {
            return self.stripe_mut(home).bitop(op, dest, srcs);
        }
        for src in srcs.clone() {
            let stripe = self.at(src);
            stripe.reap(src);
            stripe.string_only(src)?;
            stripe.thaw(src)?;
        }

        let (mut flat, mut ends) = self.take_scratch();
        flat.clear();
        ends.clear();
        let mut digits = [0u8; DIGITS_MAX];
        for src in srcs.clone() {
            let bytes = self.at_ref(src).bitmap(src, &mut digits);
            flat.extend_from_slice(bytes);
            ends.push(flat.len());
        }
        let len = bits::width(parts(&flat, &ends));
        if len > STRING_MAX {
            self.put_scratch(flat, ends);
            return Err(Error::new(Code::Invalid, TOO_LONG));
        }

        let split = flat.len();
        flat.resize(split + len, 0);
        let (read, write) = flat.split_at_mut(split);
        bits::combine(op, parts(read, &ends), write);

        let outcome = if len == 0 {
            self.at(dest).del(dest);
            Ok(0)
        } else {
            let stripe = self.at(dest);
            stripe.reap(dest);
            match stripe.string_only(dest) {
                Ok(()) => {
                    stripe.store_raw(dest, &flat[split..], None);
                    Ok(len)
                }
                Err(e) => Err(e),
            }
        };
        self.put_scratch(flat, ends);
        outcome
    }
}

/// The sources of a `BITOP`, out of the buffer they were copied into.
///
/// The boundaries are the end of each source, so the first one starts at nought
/// and each of the others starts where the one before it ended. Written as a
/// zip over two views of the same list rather than as a running offset, because
/// the iterator has to be cloneable and a clone of a running offset would carry
/// whatever the original had reached.
fn parts<'a>(flat: &'a [u8], ends: &'a [usize]) -> impl Iterator<Item = &'a [u8]> + Clone {
    std::iter::once(0)
        .chain(ends.iter().copied())
        .zip(ends.iter().copied())
        .map(|(from, to)| &flat[from..to])
}

/// Run one subcommand against a value, answering what the client is owed.
///
/// `None` is the nil an `OVERFLOW FAIL` subcommand gives when its value would
/// not fit; that one writes nothing and the ones around it still do. A `SET`
/// answers what was there before and an `INCRBY` answers what is there now,
/// which is not symmetry anybody would have chosen but is what Redis does.
///
/// The bytes have to be long enough already, which is [`reach`]'s job.
#[must_use]
pub fn apply(bytes: &mut [u8], sub: Sub) -> Option<i64> {
    let had = bits::get(bytes, sub.at, sub.field);
    match sub.op {
        SubOp::Get => Some(had),
        SubOp::Set(val) => bits::setting(sub.field, val, sub.on).map(|next| {
            bits::set(bytes, sub.at, sub.field, next);
            had
        }),
        SubOp::Incr(by) => bits::adding(sub.field, had, by, sub.on).inspect(|&next| {
            bits::set(bytes, sub.at, sub.field, next);
        }),
    }
}

/// How many bytes a value needs before `sub` can be written into it.
#[must_use]
pub const fn reach(sub: &Sub) -> usize {
    (sub.field.last_bit(sub.at) / 8 + 1) as usize
}

/// The bit `offset` names inside its byte.
///
/// Bit zero is the top bit, which is the convention all of these commands use.
#[inline]
const fn mask(offset: u64) -> u8 {
    0x80 >> (offset % 8)
}

/// A start and end index turned into a half open range of bits.
///
/// `None` for a range that holds nothing, which is what an empty value, an
/// out of range start or a backwards range all come to. Negative indexes count
/// from the end and both ends are clamped, so `BITCOUNT k -100 100` over a three
/// byte string is the whole string rather than an error.
fn window(len: usize, start: i64, end: i64, unit: Unit) -> Option<(u64, u64)> {
    let items = match unit {
        Unit::Byte => len as i64,
        Unit::Bit => (len as i64).checked_mul(8)?,
    };
    if items == 0 {
        return None;
    }
    // The two ends are not clamped the same way, and the difference is what
    // makes `BITCOUNT k 10 20` over a three byte string answer zero rather than
    // counting its last byte. A negative index counts back from the end and
    // stops at the front, the end index is pulled back to the last item, and a
    // start past the last item is left where it is so that the range comes out
    // backwards and is thrown away below.
    let back = |i: i64| if i < 0 { (items + i).max(0) } else { i };
    let (from, to) = (back(start), back(end).min(items - 1));
    if from > to {
        return None;
    }
    let scale = match unit {
        Unit::Byte => 8,
        Unit::Bit => 1,
    };
    Some(((from * scale) as u64, ((to + 1) * scale) as u64))
}

/// The largest value a bit range can name, for a caller checking its own limit.
///
/// Nothing here uses it; it is the ceiling [`STRING_MAX`] imposes expressed in
/// bits, which is what a client asking "how big can this bitmap be" wants.
#[must_use]
pub const fn max_bits() -> u64 {
    STRING_MAX as u64 * 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyspace::Keyspace;

    fn db() -> Keyspace {
        Keyspace::new()
    }

    /// The source list `bitop` takes, out of the keys a test wants to name.
    fn keys<'k>(names: &'k [&'k [u8]]) -> impl Iterator<Item = &'k [u8]> + Clone {
        names.iter().copied()
    }

    #[test]
    fn a_bit_is_set_and_read_back() {
        let mut db = db();
        assert!(!db.setbit(b"k", 7, true).expect("a bit"));
        assert!(db.getbit(b"k", 7).expect("a bit"));
        assert!(!db.getbit(b"k", 6).expect("a bit"));
        assert_eq!(db.strlen(b"k").expect("a length"), 1);
        assert_eq!(
            db.get(b"k").expect("a value").expect("bytes").to_vec(),
            b"\x01"
        );
        // The answer is what was there, not what is there now.
        assert!(db.setbit(b"k", 7, false).expect("a bit"));
        assert!(!db.setbit(b"k", 7, false).expect("a bit"));
    }

    #[test]
    fn a_write_creates_and_pads_even_when_the_bit_is_zero() {
        let mut db = db();
        assert!(!db.setbit(b"k", 0, false).expect("a bit"));
        assert!(db.exists(b"k"));
        assert_eq!(db.strlen(b"k").expect("a length"), 1);
        db.setbit(b"k", 40, true).expect("a bit");
        assert_eq!(db.strlen(b"k").expect("a length"), 6);
    }

    #[test]
    fn a_write_leaves_the_value_raw_and_a_read_does_not() {
        let mut db = db();
        db.set_plain(b"n", b"12345").expect("a set");
        assert_eq!(db.encoding(b"n"), Some(value::Encoding::Int));
        // Reading a bit out of an int is reading a bit out of its digits.
        assert!(db.getbit(b"n", 3).expect("a bit"));
        assert_eq!(db.encoding(b"n"), Some(value::Encoding::Int));
        // Writing one, even a write that changes nothing, does not leave an int.
        assert!(!db.setbit(b"n", 0, false).expect("a bit"));
        assert_eq!(db.encoding(b"n"), Some(value::Encoding::Raw));
        assert_eq!(
            db.get(b"n").expect("a value").expect("bytes").to_vec(),
            b"12345"
        );
    }

    #[test]
    fn a_write_keeps_the_deadline() {
        let mut db = db();
        db.setex(b"k", 100, b"abc").expect("a set");
        db.setbit(b"k", 40, true).expect("a bit");
        assert_eq!(db.strlen(b"k").expect("a length"), 6);
        assert!(db.expire_at(b"k").is_some());
        // And so does the fast path, which does not go near the deadline.
        db.setbit(b"k", 1, true).expect("a bit");
        assert!(db.expire_at(b"k").is_some());
    }

    #[test]
    fn counting_takes_the_ranges_a_real_server_takes() {
        let mut db = db();
        db.set_plain(b"k", b"foobar").expect("a set");
        let count = |db: &mut Keyspace, r| db.bitcount(b"k", r).expect("a count");
        assert_eq!(count(&mut db, None), 26);
        assert_eq!(count(&mut db, Some((0, 0, Unit::Byte))), 4);
        assert_eq!(count(&mut db, Some((1, 1, Unit::Byte))), 6);
        assert_eq!(count(&mut db, Some((0, -5, Unit::Byte))), 10);
        assert_eq!(count(&mut db, Some((5, 30, Unit::Bit))), 17);
        // Redis's own documentation says 22 for this one. A real 8.10.1 says 25,
        // and 25 is what counting the first 44 bits of `foobar` by hand gives,
        // so the documentation is wrong and this is not a divergence.
        assert_eq!(count(&mut db, Some((0, -5, Unit::Bit))), 25);
        // Clamped at both ends, empty when it is backwards.
        assert_eq!(count(&mut db, Some((-100, 100, Unit::Byte))), 26);
        assert_eq!(count(&mut db, Some((2, 1, Unit::Byte))), 0);
        assert_eq!(count(&mut db, Some((5, 3, Unit::Bit))), 0);
        // A start past the end is nothing, not the whole string.
        assert_eq!(count(&mut db, Some((10, 20, Unit::Byte))), 0);
        assert_eq!(db.bitcount(b"gone", None).expect("a count"), 0);
    }

    #[test]
    fn searching_takes_the_ranges_a_real_server_takes() {
        let mut db = db();
        db.set_plain(b"ones", b"\xff\xff\xff").expect("a set");
        db.set_plain(b"mix", b"\x00\xff\x00").expect("a set");
        let pos = |db: &mut Keyspace, k: &[u8], bit, s, e| {
            db.bitpos(k, bit, s, e, Unit::Byte).expect("a position")
        };
        assert_eq!(pos(&mut db, b"mix", true, None, None), 8);
        assert_eq!(pos(&mut db, b"mix", false, None, None), 0);
        assert_eq!(pos(&mut db, b"mix", true, Some(2), None), -1);
        assert_eq!(pos(&mut db, b"mix", true, Some(-1), Some(-1)), -1);
        assert_eq!(pos(&mut db, b"mix", false, Some(-100), None), 0);
        // The one exception: no end given, all ones, so the answer is the first
        // bit past the end of the string.
        assert_eq!(pos(&mut db, b"ones", false, None, None), 24);
        assert_eq!(pos(&mut db, b"ones", false, Some(-1), None), 24);
        // An explicit end takes that away again.
        assert_eq!(pos(&mut db, b"ones", false, Some(0), Some(-1)), -1);
        assert_eq!(pos(&mut db, b"ones", false, Some(0), Some(100)), -1);
        // And so does a range that is empty once it has been clamped.
        assert_eq!(pos(&mut db, b"ones", false, Some(10), None), -1);
        assert_eq!(pos(&mut db, b"ones", false, Some(3), None), -1);
        assert_eq!(pos(&mut db, b"ones", true, Some(10), None), -1);
        assert_eq!(pos(&mut db, b"ones", false, Some(2), Some(1)), -1);
        assert_eq!(
            db.bitpos(b"ones", false, Some(5), Some(20), Unit::Bit)
                .expect("a position"),
            -1
        );
    }

    #[test]
    fn searching_an_absent_or_empty_key() {
        let mut db = db();
        let pos = |db: &mut Keyspace, k: &[u8], bit| {
            db.bitpos(k, bit, None, None, Unit::Byte)
                .expect("a position")
        };
        // A key that is not there is all zeros, so a zero is at the front.
        assert_eq!(pos(&mut db, b"gone", false), 0);
        assert_eq!(pos(&mut db, b"gone", true), -1);
        // A key that is there and empty has no bits at all.
        db.set_plain(b"empty", b"").expect("a set");
        assert_eq!(pos(&mut db, b"empty", false), -1);
        assert_eq!(pos(&mut db, b"empty", true), -1);
        assert_eq!(
            db.bitcount(b"empty", Some((0, -1, Unit::Byte)))
                .expect("a count"),
            0
        );
    }

    #[test]
    fn combining_writes_a_destination_and_deletes_an_empty_one() {
        let mut db = db();
        db.set_plain(b"a", b"\xf0\x0f\xff").expect("a set");
        db.set_plain(b"b", b"\xff\x00").expect("a set");
        let n = db
            .bitop(Op::And, b"d", keys(&[b"a", b"b"]))
            .expect("a length");
        assert_eq!(n, 3);
        assert_eq!(
            db.get(b"d").expect("a value").expect("bytes").to_vec(),
            b"\xf0\x00\x00"
        );
        // A destination full of nothing is still a destination.
        db.set_plain(b"z", b"\x00\x00").expect("a set");
        let n = db
            .bitop(Op::And, b"d", keys(&[b"a", b"z"]))
            .expect("a length");
        assert_eq!(n, 3);
        assert!(db.exists(b"d"));
        // Sources that are all missing take the destination with them.
        let n = db
            .bitop(Op::Or, b"d", keys(&[b"no1", b"no2"]))
            .expect("a length");
        assert_eq!(n, 0);
        assert!(!db.exists(b"d"));
    }

    #[test]
    fn combining_reads_an_int_key_as_its_digits() {
        let mut db = db();
        db.set_plain(b"n", b"12345").expect("a set");
        db.bitop(Op::Or, b"d", keys(&[b"n"])).expect("a length");
        assert_eq!(
            db.get(b"d").expect("a value").expect("bytes").to_vec(),
            b"12345"
        );
    }

    #[test]
    fn a_field_is_read_written_and_incremented() {
        let mut db = db();
        let u8f = Field::new(false, 8).expect("a width");
        let sub = |op, at| Sub {
            op,
            field: u8f,
            at,
            on: Overflow::Wrap,
        };
        let out = db
            .bitfield(b"k", &[sub(SubOp::Set(255), 0), sub(SubOp::Get, 0)])
            .expect("replies");
        assert_eq!(out, vec![Some(0), Some(255)]);
        assert_eq!(db.strlen(b"k").expect("a length"), 1);

        let out = db
            .bitfield(b"k", &[sub(SubOp::Incr(10), 0)])
            .expect("replies");
        assert_eq!(out, vec![Some(9)], "wrapped round");

        // A failing write answers nothing and leaves the field alone, and the
        // subcommands around it still run.
        let fail = Sub {
            on: Overflow::Fail,
            ..sub(SubOp::Incr(250), 0)
        };
        let out = db
            .bitfield(b"k", &[fail, sub(SubOp::Get, 0)])
            .expect("replies");
        assert_eq!(out, vec![None, Some(9)]);
    }

    #[test]
    fn a_read_only_bitfield_creates_nothing_and_re_encodes_nothing() {
        let mut db = db();
        let f = Field::new(true, 16).expect("a width");
        let get = Sub {
            op: SubOp::Get,
            field: f,
            at: 0,
            on: Overflow::Wrap,
        };
        assert_eq!(
            db.bitfield(b"gone", &[get]).expect("replies"),
            vec![Some(0)]
        );
        assert!(!db.exists(b"gone"));

        db.set_plain(b"s", b"hello").expect("a set");
        assert_eq!(db.encoding(b"s"), Some(value::Encoding::Embstr));
        db.bitfield(b"s", &[get]).expect("replies");
        assert_eq!(
            db.encoding(b"s"),
            Some(value::Encoding::Embstr),
            "still short"
        );
    }

    #[test]
    fn a_write_grows_the_value_even_when_every_write_fails() {
        let mut db = db();
        let f = Field::new(false, 8).expect("a width");
        let sub = Sub {
            op: SubOp::Set(300),
            field: f,
            at: 64,
            on: Overflow::Fail,
        };
        assert_eq!(db.bitfield(b"k", &[sub]).expect("replies"), vec![None]);
        assert_eq!(db.strlen(b"k").expect("a length"), 9);
    }

    #[test]
    fn a_bit_command_on_the_wrong_type_says_so() {
        let mut db = db();
        let member: &[u8] = b"x";
        db.sadd(b"s", std::iter::once(member)).expect("a member");
        assert!(db.getbit(b"s", 0).is_err());
        assert!(db.setbit(b"s", 0, true).is_err());
        assert!(db.bitcount(b"s", None).is_err());
        assert!(db.bitpos(b"s", true, None, None, Unit::Byte).is_err());
        assert!(db.bitop(Op::Or, b"d", keys(&[b"s"])).is_err());
        let f = Field::new(false, 8).expect("a width");
        let sub = Sub {
            op: SubOp::Get,
            field: f,
            at: 0,
            on: Overflow::Wrap,
        };
        assert!(db.bitfield(b"s", &[sub]).is_err());
    }

    #[test]
    fn an_offset_past_the_end_of_the_world_is_refused() {
        let mut db = db();
        assert!(db.setbit(b"k", BIT_OFFSET_MAX + 1, true).is_err());
        assert!(db.getbit(b"k", BIT_OFFSET_MAX + 1).is_err());
        // And one inside Redis's limit but outside ours is refused too, with the
        // other sentence. This is the divergence [`STRING_MAX`] is about.
        assert!(db.setbit(b"k", BIT_OFFSET_MAX, true).is_err());
        assert!(max_bits() < BIT_OFFSET_MAX);
    }
}
