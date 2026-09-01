//! The RDB payload that `DUMP` hands out and `RESTORE` takes back.
//!
//! A payload is one value with no key and no deadline, wrapped in ten bytes that
//! say it is intact:
//!
//! ```text
//! +------+------------------+---------+-----------+
//! | type | the object       | version | crc64     |
//! | 1 B  | as many as it is | 2 B LE  | 8 B LE    |
//! +------+------------------+---------+-----------+
//!                                     ^ over everything to its left
//! ```
//!
//! This is not a file format even though it is spelled like one. There is no
//! header, no database selector and no end of file opcode, because the whole
//! point is that it fits in a bulk string. The version is here so that a server
//! reading a payload can refuse one from a newer server rather than misread it,
//! and the checksum is here because `RESTORE` takes bytes from a client and a
//! client is allowed to be wrong.
//!
//! # Two version numbers and not one
//!
//! The version we stamp on a payload and the version we will read are different
//! numbers, because they are answers to opposite questions. What we write is a
//! promise about how old a server can be and still understand us, so it is as
//! low as it can be. What we read is a statement about how new a server can be
//! before a type byte might not mean what it used to, so it is as high as has
//! actually been checked. [`VERSION`] and [`READS_UP_TO`] say which is which.
//!
//! # It has to be Redis's bytes, not ours
//!
//! Nothing here is an internal format we get to choose. `MIGRATE` sends this to
//! another server and `RESTORE` accepts it from any client, so a payload we
//! produce has to load into a real Redis and a payload a real Redis produces has
//! to load here. That is the only reason CRC64 exists in `yo-common`, and it is
//! why the type bytes below are copied from `rdb.h` rather than numbered from
//! zero in the order this file happens to handle them.
//!
//! # Writing the simple shape and reading every shape
//!
//! The two directions are deliberately not symmetric. Reading accepts every
//! encoding a modern Redis emits, because we do not get to pick what arrives.
//! Writing picks the plainest legal type for each kind, a count followed by the
//! elements, because every one of those loads into Redis 8.2 and one shape per
//! kind is one shape to get right.
//!
//! # Copying the blob when there is one
//!
//! That is the shape for values that are stored as a structure. A value that is
//! already sitting in one packed blob does not go through it, because
//! [`crate::listpack`] and [`crate::intset`] are byte compatible with Redis's
//! own on purpose, so the payload for one of those is the blob with a length in
//! front of it. A small set, a small hash and a small sorted set are one memcpy
//! each instead of a walk that decodes every element and encodes it again, and
//! they are the overwhelming majority of what `DUMP` and `MIGRATE` are pointed
//! at.
//!
//! The rule for which type byte a value gets is the same word `OBJECT ENCODING`
//! answers with and not the body underneath it, so a set that calls itself a
//! hashtable is walked even in the corner where its members happen to still be
//! in one intset run. There is one rule and one place to read it.
//!
//! A hash that has been widened for field deadlines is not copied. That band
//! carries a third element per field and keeps it after the last deadline has
//! been taken off, so the blob it holds is not the blob `HASH_LISTPACK` means
//! and the walk is what makes it one.
//!
//! What that is worth, from `benches/rdb.rs` at a hundred elements, walked
//! against copied:
//!
//! ```text
//!   set of text      7.57 us    3.41 us    2.2x
//!   set of integers  1.32 us    0.57 us    2.3x
//!   hash            24.94 us    3.95 us    6.3x
//!   sorted set      21.04 us    3.82 us    5.5x
//! ```
//!
//! The same rows at a thousand elements, which is past every packed band and is
//! therefore the walk in both runs, moved by under one percent, so nothing here
//! was paid for by the values that do not benefit. What is left on the copied
//! rows is a checksum over the payload and one allocation to put it in, and both
//! of those are paid whichever way the payload was built, which is why the hash
//! and the sorted set gain more than the two sets do: their walk was the more
//! expensive one, not their copy the cheaper.
//!
//! The load side pays about five percent for this on a hash and a sorted set,
//! because a listpack entry has to be decoded where a count prefixed element is
//! read straight off a length. Copying the blob is not free on the way back in
//! and the trade is still worth making, since a payload is written once and this
//! is a five percent loss against a five hundred percent gain.
//!
//! The load side also stopped asking a listpack for element `i`. There is no
//! offset table in a listpack, so `get(i)` walks from the front and a loop that
//! asks for every element in turn costs the square of the count. A hundred field
//! hash loaded in 81 us and loads in 60. That was there before any of this and
//! the only thing that ever reached it was a payload from a real server, which
//! is the case that matters most.
//!
//! # Taking the blob back
//!
//! A payload for a sorted set on the packed band goes the other way too. The
//! blob that arrives is the layout that band uses, so it moves in whole rather
//! than being added a member at a time, and the difference is not small: adding
//! costs a scan to see whether the member is already there and a second scan to
//! find where it belongs, both over everything added so far, so it is the square
//! of the count twice over with a memmove on each one. A hundred member sorted
//! set restored in 534 us and restores in 4.6 us.
//!
//! The blob is checked before it is taken. This band answers a rank query by
//! position and by nothing else, so a payload that says it is a sorted set while
//! not being sorted would answer `ZRANGE` with the wrong members and never say
//! why, and a payload with the same member twice would report a length nothing
//! else agrees with. One pass rules out both, since strictly increasing means no
//! two members compare equal on the score and then equal on the bytes. A blob
//! that fails the check, or that is past this server's limits, is handed back
//! and walked, which is what the reader did with every payload before this.
//!
//! A sorted set past the band is sized from the count now, the way a set and a
//! hash already were. It used to start packed whatever the count said, fill to
//! the band limit at a scan a member, and throw the listpack away. A thousand
//! member sorted set restored in 1.23 ms and restores in 88 us.
//!
//! What is left is the hash, which has the same shape of problem for the same
//! reason and is not fixed here. Its blob is not sorted, so ruling out a
//! duplicate field is not free the way it is for a sorted set, and that wants
//! its own change rather than being smuggled into this one.
//!
//! # Compression
//!
//! Redis compresses strings over twenty bytes with LZF when `rdbcompression` is
//! on, which it is by default. Nothing here compresses on the way out, because
//! an uncompressed string is legal and every reader accepts it. Decompression on
//! the way in is not optional, because payloads arriving from a real Redis are
//! full of LZF strings.

use std::borrow::Cow;

use yo_common::crc::crc64;
use yo_common::num::{self, DIGITS_MAX};

use crate::hash::{self, Hash};
use crate::intset::Intset;
use crate::keys::{Body, Record};
use crate::list::{self, List};
use crate::listpack::{Entry, Listpack};
use crate::set::{self, Set};
use crate::zset::{self, Zset};

/// The RDB version this server writes into the footer.
///
/// Redis refuses a payload whose version is above its own, so this being right
/// is the difference between a payload another server will look at and one it
/// throws away without reading. Lower is friendlier, and twelve is as low as
/// this can go: it is the version that introduced the hash with field deadlines,
/// which is a shape this server writes.
pub const VERSION: u16 = 12;

/// The highest version in a footer this server will still read.
///
/// A different number from [`VERSION`], and the two mean opposite things. What
/// we write is a promise about how old a server can be and still understand us.
/// What we read is a statement about how new a server can be before we stop
/// trusting that a type byte still means what it used to.
///
/// Fifteen because that is what a Redis 8.10.1 stamps on a payload, read off one
/// over a socket rather than out of a header file. Refusing it is not a small
/// bug: it means `RESTORE` turns down every payload a current server produces,
/// with a message about the checksum that sends the reader to entirely the wrong
/// place. That is what this constant existing separately is here to stop.
///
/// It goes up when a newer server has been checked and not before. The guard is
/// worth keeping rather than removing, because the day Redis reuses a type byte
/// for a different layout, refusing to read it is the only safe answer and a
/// wrong value is worse than no value.
pub const READS_UP_TO: u16 = 15;

/// The footer: two bytes of version and eight of checksum.
const FOOTER: usize = 10;

// The object type byte. These are `rdb.h`, and the gaps are types this server
// cannot hold, so they are not named.
const T_STRING: u8 = 0;
const T_LIST: u8 = 1;
const T_SET: u8 = 2;
const T_ZSET: u8 = 3;
const T_HASH: u8 = 4;
const T_ZSET_2: u8 = 5;
const T_SET_INTSET: u8 = 11;
const T_HASH_LISTPACK: u8 = 16;
const T_ZSET_LISTPACK: u8 = 17;
const T_LIST_QUICKLIST_2: u8 = 18;
const T_SET_LISTPACK: u8 = 20;
const T_HASH_METADATA: u8 = 24;
const T_HASH_LISTPACK_EX: u8 = 25;

// The length encoding, `00` and `01` in the top two bits for six and fourteen
// bit lengths, then two whole byte forms, and `11` for the special encodings.
const LEN_6BIT: u8 = 0;
const LEN_14BIT: u8 = 1;
const LEN_32BIT: u8 = 0x80;
const LEN_64BIT: u8 = 0x81;
const LEN_ENCODED: u8 = 3;

// What a `11` length means: three integer widths and a compressed blob.
const ENC_INT8: u64 = 0;
const ENC_INT16: u64 = 1;
const ENC_INT32: u64 = 2;
const ENC_LZF: u64 = 3;

/// A quicklist node holding a listpack rather than one long value.
const NODE_PACKED: u64 = 2;
/// A quicklist node that is one value too big for a listpack.
const NODE_PLAIN: u64 = 1;

/// Why a payload was not accepted.
///
/// Two variants because `RESTORE` has two complaints and a client can tell them
/// apart. A bad footer means the bytes were damaged or came from a newer server,
/// and everything else means they were intact and still did not make sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bad {
    /// The version is from the future or the checksum does not match.
    Footer,
    /// The bytes are self consistent and are not a value this server can hold.
    Format,
}

/// Where a load is allowed to put what it builds.
///
/// The four representation thresholds, borrowed rather than copied, because a
/// restore has to land in the same band the same data would have landed in had
/// it been written a command at a time. A hash restored into a listpack on a
/// server configured for tables would answer the wrong thing to `OBJECT
/// ENCODING` and would be a different size for the rest of its life.
#[derive(Debug, Clone, Copy)]
pub struct Limits<'a> {
    /// `set-max-intset-entries` and the two listpack thresholds.
    pub set: &'a set::Limits,
    /// `hash-max-listpack-entries` and `hash-max-listpack-value`.
    pub hash: &'a hash::Limits,
    /// `list-max-listpack-size`, as bytes or as a count.
    pub list: &'a list::Limits,
    /// `zset-max-listpack-entries` and `zset-max-listpack-value`.
    pub zset: &'a zset::Limits,
}

// ---------------------------------------------------------------------------
// The wrapper: version and checksum.
// ---------------------------------------------------------------------------

/// Put the version and the checksum on the end of a serialised object.
fn seal(mut body: Vec<u8>) -> Vec<u8> {
    body.extend_from_slice(&VERSION.to_le_bytes());
    let crc = crc64(0, &body);
    body.extend_from_slice(&crc.to_le_bytes());
    body
}

/// Check the footer and hand back everything in front of it.
///
/// The version check comes before the checksum, which is the order Redis uses
/// and the order that gives the better answer: a payload from a newer server is
/// usually intact, and telling somebody their bytes are corrupt when they are
/// merely from next year sends them looking in the wrong place.
fn unseal(payload: &[u8]) -> Result<&[u8], Bad> {
    if payload.len() < FOOTER {
        return Err(Bad::Footer);
    }
    let split = payload.len() - FOOTER;
    let (body, foot) = payload.split_at(split);
    let version = u16::from_le_bytes([foot[0], foot[1]]);
    if version > READS_UP_TO {
        return Err(Bad::Footer);
    }
    let stored = u64::from_le_bytes(foot[2..].try_into().expect("ten byte footer, eight left"));
    if stored != crc64(0, &payload[..payload.len() - 8]) {
        return Err(Bad::Footer);
    }
    Ok(body)
}

// ---------------------------------------------------------------------------
// Writing.
// ---------------------------------------------------------------------------

/// Serialise a record's value, footer and all.
///
/// The deadline does not go in. `DUMP` deliberately drops it and `RESTORE` takes
/// a fresh one as an argument, because a payload that travels for a while would
/// otherwise arrive already expired or, worse, silently alive for longer than
/// anybody meant.
///
/// `None` for a value with no RDB shape at all, which today is only the sparse
/// array. No command on the wire can create one yet, so no client can reach this,
/// and it is a `None` rather than a panic so that the day the document commands
/// land the answer is a missing key and not a dead server.
pub(crate) fn dump(rec: &Record) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    match rec.body() {
        Body::String(bytes) => {
            out.push(T_STRING);
            put_str(&mut out, bytes);
        }
        Body::List(list) => {
            out.push(T_LIST);
            put_len(&mut out, list.len() as u64);
            for element in list.iter() {
                put_entry(&mut out, element);
            }
        }
        Body::Set(set) => match (set.encoding(), set.packed_bytes()) {
            (set::Encoding::Intset, Some(blob)) => {
                out.push(T_SET_INTSET);
                put_str(&mut out, blob);
            }
            (set::Encoding::Listpack, Some(blob)) => {
                out.push(T_SET_LISTPACK);
                put_str(&mut out, blob);
            }
            _ => {
                out.push(T_SET);
                put_len(&mut out, set.len() as u64);
                for member in set.iter() {
                    put_entry(&mut out, member);
                }
            }
        },
        Body::Zset(zset) => match zset.packed_bytes() {
            Some(blob) => {
                out.push(T_ZSET_LISTPACK);
                put_str(&mut out, blob);
            }
            None => {
                out.push(T_ZSET_2);
                put_len(&mut out, zset.len() as u64);
                zset.walk(0, zset.len(), false, |member, score| {
                    put_entry(&mut out, member);
                    out.extend_from_slice(&score.to_le_bytes());
                });
            }
        },
        Body::Hash(hash) => put_hash(&mut out, hash),
        Body::Array(_) => return None,
    }
    Some(seal(out))
}

/// A hash, in the plain shape or the one that carries field deadlines.
///
/// Two types because the deadline costs a length prefixed number on every single
/// field, and the overwhelming majority of hashes have no deadline anywhere.
/// Redis makes the same split for the same reason, and the trick it uses is
/// worth copying: the earliest deadline in the hash goes in the header, and each
/// field stores the difference from it plus one, so a field with no deadline is
/// a zero and everything else is a small number rather than a full timestamp.
fn put_hash(out: &mut Vec<u8>, hash: &Hash) {
    let Some(soonest) = hash.soonest_deadline() else {
        if let Some(blob) = hash.packed_bytes() {
            out.push(T_HASH_LISTPACK);
            put_str(out, blob);
            return;
        }
        out.push(T_HASH);
        put_len(out, hash.len() as u64);
        for (field, value) in hash.iter() {
            put_entry(out, field);
            put_entry(out, value);
        }
        return;
    };
    out.push(T_HASH_METADATA);
    out.extend_from_slice(&soonest.to_le_bytes());
    put_len(out, hash.len() as u64);
    for i in 0..hash.len() {
        let (field, value) = hash.at(i).expect("index is under the length");
        // Saturating rather than subtracting, because `soonest_deadline` is
        // documented as a lower bound and a bound that is early by a millisecond
        // would underflow into a deadline a few hundred million years out.
        let ttl = match hash.deadline_at(i) {
            Some(at) => at.saturating_sub(soonest) + 1,
            None => 0,
        };
        put_len(out, ttl);
        put_entry(out, field);
        put_entry(out, value);
    }
}

/// A length, in the smallest of the four forms that holds it.
fn put_len(out: &mut Vec<u8>, n: u64) {
    if n < 1 << 6 {
        out.push((LEN_6BIT << 6) | n as u8);
    } else if n < 1 << 14 {
        out.push((LEN_14BIT << 6) | (n >> 8) as u8);
        out.push(n as u8);
    } else if n <= u64::from(u32::MAX) {
        out.push(LEN_32BIT);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(LEN_64BIT);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

/// A string, integer encoded when that is both possible and shorter.
fn put_str(out: &mut Vec<u8>, s: &[u8]) {
    // Redis only tries the integer encoding on strings short enough to be one,
    // which saves parsing every long value that starts with a digit.
    if s.len() <= 11
        && let Some(n) = num::parse_i64(s)
        && let mut buf = [0u8; DIGITS_MAX]
        && num::i64_digits(&mut buf, n) == s
        && put_int(out, n)
    {
        return;
    }
    put_len(out, s.len() as u64);
    out.extend_from_slice(s);
}

/// An element straight out of a collection.
///
/// A listpack already knows whether it is holding an integer, so an integer
/// element goes out in the integer encoding without ever being formatted into
/// digits and parsed back. That is the same saving the reply path makes and it
/// is why elements come back as an [`Entry`] rather than as bytes.
fn put_entry(out: &mut Vec<u8>, entry: Entry<'_>) {
    match entry {
        Entry::Int(n) => {
            if !put_int(out, n) {
                let mut buf = [0u8; DIGITS_MAX];
                let digits = num::i64_digits(&mut buf, n);
                put_len(out, digits.len() as u64);
                out.extend_from_slice(digits);
            }
        }
        Entry::Str(s) => put_str(out, s),
    }
}

/// An integer in one of the three widths, or `false` if it does not fit any.
///
/// There is no 64 bit form. A number past `i32` goes out as digits, which is
/// what Redis does, and it is not the oversight it looks like: the encoding is
/// there to make short strings shorter and a nineteen digit number in eight
/// bytes saves eleven bytes on a value that is already rare.
fn put_int(out: &mut Vec<u8>, n: i64) -> bool {
    if let Ok(v) = i8::try_from(n) {
        out.push((LEN_ENCODED << 6) | ENC_INT8 as u8);
        out.push(v as u8);
    } else if let Ok(v) = i16::try_from(n) {
        out.push((LEN_ENCODED << 6) | ENC_INT16 as u8);
        out.extend_from_slice(&v.to_le_bytes());
    } else if let Ok(v) = i32::try_from(n) {
        out.push((LEN_ENCODED << 6) | ENC_INT32 as u8);
        out.extend_from_slice(&v.to_le_bytes());
    } else {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Reading.
// ---------------------------------------------------------------------------

/// A position in a payload, and the only thing allowed to advance it.
///
/// Every read goes through here so that a truncated payload is one error at one
/// place rather than a bounds check per field that somebody eventually forgets.
struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    const fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, at: 0 }
    }

    fn byte(&mut self) -> Result<u8, Bad> {
        let b = *self.buf.get(self.at).ok_or(Bad::Format)?;
        self.at += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Bad> {
        let end = self.at.checked_add(n).ok_or(Bad::Format)?;
        let s = self.buf.get(self.at..end).ok_or(Bad::Format)?;
        self.at = end;
        Ok(s)
    }

    /// A length, refusing the `11` forms that are not lengths at all.
    fn len(&mut self) -> Result<usize, Bad> {
        match self.len_or_encoding()? {
            (n, false) => usize::try_from(n).map_err(|_| Bad::Format),
            (_, true) => Err(Bad::Format),
        }
    }

    /// A length, and whether it was one of the special encodings instead.
    fn len_or_encoding(&mut self) -> Result<(u64, bool), Bad> {
        let first = self.byte()?;
        match first >> 6 {
            LEN_6BIT => Ok((u64::from(first & 0x3f), false)),
            LEN_14BIT => {
                let second = self.byte()?;
                Ok(((u64::from(first & 0x3f) << 8) | u64::from(second), false))
            }
            LEN_ENCODED => Ok((u64::from(first & 0x3f), true)),
            // The remaining two bit pattern is `10`, where the whole first byte
            // says which width follows rather than carrying any of the length.
            _ => match first {
                LEN_32BIT => {
                    let b = self.take(4)?;
                    Ok((
                        u64::from(u32::from_be_bytes(b.try_into().expect("four bytes"))),
                        false,
                    ))
                }
                LEN_64BIT => {
                    let b = self.take(8)?;
                    Ok((
                        u64::from_be_bytes(b.try_into().expect("eight bytes")),
                        false,
                    ))
                }
                _ => Err(Bad::Format),
            },
        }
    }

    /// A string, whichever of the five ways it was written.
    ///
    /// Borrowed when the bytes are already there and owned when they had to be
    /// built, which is the integer encodings and LZF. Most strings in a payload
    /// are plain, so most of them cost nothing here.
    fn str(&mut self) -> Result<Cow<'a, [u8]>, Bad> {
        let (n, encoded) = self.len_or_encoding()?;
        if !encoded {
            let n = usize::try_from(n).map_err(|_| Bad::Format)?;
            return Ok(Cow::Borrowed(self.take(n)?));
        }
        let value = match n {
            ENC_INT8 => i64::from(self.byte()? as i8),
            ENC_INT16 => {
                let b = self.take(2)?;
                i64::from(i16::from_le_bytes(b.try_into().expect("two bytes")))
            }
            ENC_INT32 => {
                let b = self.take(4)?;
                i64::from(i32::from_le_bytes(b.try_into().expect("four bytes")))
            }
            ENC_LZF => {
                let packed = self.len()?;
                let plain = self.len()?;
                let bytes = self.take(packed)?;
                return unpack(bytes, plain).map(Cow::Owned).ok_or(Bad::Format);
            }
            _ => return Err(Bad::Format),
        };
        let mut buf = [0u8; DIGITS_MAX];
        Ok(Cow::Owned(num::i64_digits(&mut buf, value).to_vec()))
    }

    /// A score in the binary form, which is `ZSET_2` and everything since.
    fn double(&mut self) -> Result<f64, Bad> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes(b.try_into().expect("eight bytes")))
    }

    /// A score in the old text form, which only `ZSET` uses.
    ///
    /// A length byte and then that many digits, with three of the lengths
    /// reserved to mean the three values that have no digits.
    fn double_text(&mut self) -> Result<f64, Bad> {
        match self.byte()? {
            255 => Ok(f64::NEG_INFINITY),
            254 => Ok(f64::INFINITY),
            253 => Ok(f64::NAN),
            n => {
                let digits = self.take(n as usize)?;
                num::parse_f64(digits).ok_or(Bad::Format)
            }
        }
    }

    /// Whether every byte has been read, which a well formed payload has.
    const fn done(&self) -> bool {
        self.at == self.buf.len()
    }
}

/// LZF, the one compression Redis puts in an RDB payload.
///
/// A control byte either introduces a run of literals or points backwards into
/// what has already been written. The back reference is allowed to overlap what
/// it is producing, which is how a long run of one byte compresses, so the copy
/// has to go one byte at a time rather than through a slice copy.
///
/// `plain` is the length the payload claims the result will be, and it is used
/// as the bound rather than trusted, so a payload claiming four bytes and
/// describing four gigabytes stops at four.
fn unpack(packed: &[u8], plain: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(plain.min(1 << 20));
    let mut i = 0;
    while i < packed.len() {
        let ctrl = usize::from(packed[i]);
        i += 1;
        if ctrl < 32 {
            let run = ctrl + 1;
            let end = i.checked_add(run)?;
            if end > packed.len() || out.len() + run > plain {
                return None;
            }
            out.extend_from_slice(&packed[i..end]);
            i = end;
        } else {
            let mut run = ctrl >> 5;
            if run == 7 {
                run += usize::from(*packed.get(i)?);
                i += 1;
            }
            let back = ((ctrl & 0x1f) << 8) + usize::from(*packed.get(i)?) + 1;
            i += 1;
            let run = run + 2;
            if back > out.len() || out.len() + run > plain {
                return None;
            }
            let from = out.len() - back;
            for at in from..from + run {
                out.push(out[at]);
            }
        }
    }
    (out.len() == plain).then_some(out)
}

/// Turn a payload back into a value.
///
/// `now` is here for one reason: a hash can carry deadlines and a field whose
/// deadline has already gone is not put back. Restoring it would leave a field
/// that the very next read would delete, and a count that is wrong until
/// somebody looks.
pub(crate) fn load(payload: &[u8], limits: Limits<'_>, now: u64) -> Result<Body, Bad> {
    let body = unseal(payload)?;
    let mut r = Reader::new(body);
    let kind = r.byte()?;
    let value = match kind {
        T_STRING => Body::String(r.str()?.into_owned()),
        T_LIST => read_list(&mut r, limits.list)?,
        T_LIST_QUICKLIST_2 => read_quicklist(&mut r, limits.list)?,
        T_SET => read_set(&mut r, limits.set)?,
        T_SET_INTSET => read_intset(&mut r, limits.set)?,
        T_SET_LISTPACK => read_set_listpack(&mut r, limits.set)?,
        T_ZSET | T_ZSET_2 => read_zset(&mut r, limits.zset, kind == T_ZSET_2)?,
        T_ZSET_LISTPACK => read_zset_listpack(&mut r, limits.zset)?,
        T_HASH => read_hash(&mut r, limits.hash)?,
        T_HASH_METADATA => read_hash_metadata(&mut r, limits.hash, now)?,
        T_HASH_LISTPACK => read_hash_listpack(&mut r, limits.hash, false, now)?,
        T_HASH_LISTPACK_EX => read_hash_listpack(&mut r, limits.hash, true, now)?,
        _ => return Err(Bad::Format),
    };
    // Trailing bytes mean the payload was not what it said it was, even though
    // everything read so far parsed. Redis is stricter than it looks here and so
    // is this, because a payload with something extra on the end is either a
    // different version's idea of the same type or somebody probing.
    if !r.done() {
        return Err(Bad::Format);
    }
    Ok(value)
}

/// An empty collection is not a value, it is a deleted key.
///
/// Redis calls this `emptykey` and refuses the payload rather than creating a
/// key that every command would treat as missing. A zero length collection
/// cannot be produced by any command, so a payload holding one was either
/// hand written or corrupted in a way the checksum happened to survive.
fn non_empty(n: usize) -> Result<usize, Bad> {
    if n == 0 { Err(Bad::Format) } else { Ok(n) }
}

fn read_list(r: &mut Reader<'_>, limits: &list::Limits) -> Result<Body, Bad> {
    let n = non_empty(r.len()?)?;
    let mut list = List::new();
    for _ in 0..n {
        list.push_back(&r.str()?, limits);
    }
    Ok(Body::List(list))
}

/// A quicklist, which is a count of nodes and then a blob each.
///
/// A packed node is a whole listpack and a plain node is a single value that was
/// too long to pack, and both of them are written as one string, so the only
/// difference is whether the string is parsed or pushed.
fn read_quicklist(r: &mut Reader<'_>, limits: &list::Limits) -> Result<Body, Bad> {
    let nodes = non_empty(r.len()?)?;
    let mut list = List::new();
    for _ in 0..nodes {
        let container = r.len_or_encoding()?.0;
        let blob = r.str()?;
        match container {
            NODE_PLAIN => list.push_back(&blob, limits),
            NODE_PACKED => {
                let lp = Listpack::from_bytes(&blob).map_err(|_| Bad::Format)?;
                let mut buf = [0u8; DIGITS_MAX];
                for entry in lp.iter() {
                    list.push_back(text(entry, &mut buf), limits);
                }
            }
            _ => return Err(Bad::Format),
        }
    }
    if list.is_empty() {
        return Err(Bad::Format);
    }
    Ok(Body::List(list))
}

fn read_set(r: &mut Reader<'_>, limits: &set::Limits) -> Result<Body, Bad> {
    let n = non_empty(r.len()?)?;
    let first = r.str()?;
    // The hint and the first member together are what decide the band, so the
    // first member is read before the set is built rather than after.
    let mut set = Set::with_hint(&first, n, limits);
    set.add(&first, limits);
    for _ in 1..n {
        set.add(&r.str()?, limits);
    }
    Ok(Body::Set(set))
}

fn read_intset(r: &mut Reader<'_>, limits: &set::Limits) -> Result<Body, Bad> {
    let blob = r.str()?;
    let ints = Intset::from_bytes(&blob).map_err(|_| Bad::Format)?;
    non_empty(ints.len())?;
    let mut buf = [0u8; DIGITS_MAX];
    let mut set = Set::with_hint(num::i64_digits(&mut buf, ints.at(0)), ints.len(), limits);
    for v in ints.iter() {
        set.add(num::i64_digits(&mut buf, v), limits);
    }
    Ok(Body::Set(set))
}

fn read_set_listpack(r: &mut Reader<'_>, limits: &set::Limits) -> Result<Body, Bad> {
    let blob = r.str()?;
    let lp = Listpack::from_bytes(&blob).map_err(|_| Bad::Format)?;
    non_empty(lp.len())?;
    let mut buf = [0u8; DIGITS_MAX];
    let first = text(lp.get(0).ok_or(Bad::Format)?, &mut buf).to_vec();
    let mut set = Set::with_hint(&first, lp.len(), limits);
    for entry in lp.iter() {
        set.add(text(entry, &mut buf), limits);
    }
    Ok(Body::Set(set))
}

fn read_zset(r: &mut Reader<'_>, limits: &zset::Limits, binary: bool) -> Result<Body, Bad> {
    let n = non_empty(r.len()?)?;
    // Sized from the count, the way `read_set` and `read_hash` are. A sorted set
    // that is going to end up on the table should start there, rather than fill
    // the packed band to its limit at a scan a member and then throw it away.
    let mut zset = Zset::with_hint(n, limits);
    for _ in 0..n {
        let member = r.str()?;
        let score = if binary {
            r.double()?
        } else {
            r.double_text()?
        };
        zset.add(&member, score, limits);
    }
    Ok(Body::Zset(zset))
}

fn read_zset_listpack(r: &mut Reader<'_>, limits: &zset::Limits) -> Result<Body, Bad> {
    let blob = r.str()?;
    let lp = Listpack::from_bytes(&blob).map_err(|_| Bad::Format)?;
    if lp.is_empty() || lp.len() % 2 != 0 {
        return Err(Bad::Format);
    }
    // The payload is already the layout the packed band uses, so the fast answer
    // is to take it whole rather than to add a member at a time. `from_packed`
    // hands the blob back when it will not have it, and then the walk below
    // rebuilds it, which is what happens to a payload that is out of order or
    // past this server's limits.
    let lp = match Zset::from_packed(lp, limits) {
        Ok(zset) => return Ok(Body::Zset(zset)),
        Err(lp) => lp,
    };
    let mut zset = Zset::with_hint(lp.len() / 2, limits);
    let mut member = [0u8; DIGITS_MAX];
    let mut score = [0u8; DIGITS_MAX];
    // Walked and not indexed. A listpack has no offset table, so asking it for
    // element `i` costs a walk from the front and asking it for every element in
    // turn costs the square of the count.
    let mut walk = lp.iter();
    while let Some(entry) = walk.next() {
        let name = text(entry, &mut member).to_vec();
        let at = text(walk.next().ok_or(Bad::Format)?, &mut score);
        let at = num::parse_f64(at).ok_or(Bad::Format)?;
        zset.add(&name, at, limits);
    }
    Ok(Body::Zset(zset))
}

fn read_hash(r: &mut Reader<'_>, limits: &hash::Limits) -> Result<Body, Bad> {
    let n = non_empty(r.len()?)?;
    let mut hash = Hash::with_hint(n, limits);
    for _ in 0..n {
        let field = r.str()?;
        let value = r.str()?;
        hash.set(&field, &value, limits);
    }
    Ok(Body::Hash(hash))
}

/// A hash with field deadlines, which is a header, a count and then triples.
///
/// The header is the earliest deadline in the hash and each field holds its own
/// distance from it, plus one so that a zero can mean no deadline at all.
fn read_hash_metadata(r: &mut Reader<'_>, limits: &hash::Limits, now: u64) -> Result<Body, Bad> {
    let soonest = u64::from_le_bytes(r.take(8)?.try_into().expect("eight bytes"));
    let n = non_empty(r.len()?)?;
    let mut hash = Hash::with_hint(n, limits);
    for _ in 0..n {
        let ttl = r.len_or_encoding()?.0;
        let field = r.str()?;
        let value = r.str()?;
        put_field(
            &mut hash,
            &field,
            &value,
            deadline(soonest, ttl),
            limits,
            now,
        );
    }
    if hash.is_empty() {
        return Err(Bad::Format);
    }
    Ok(Body::Hash(hash))
}

/// A hash packed into one listpack, with or without the deadline column.
fn read_hash_listpack(
    r: &mut Reader<'_>,
    limits: &hash::Limits,
    with_ttl: bool,
    now: u64,
) -> Result<Body, Bad> {
    let soonest = if with_ttl {
        u64::from_le_bytes(r.take(8)?.try_into().expect("eight bytes"))
    } else {
        0
    };
    let blob = r.str()?;
    let lp = Listpack::from_bytes(&blob).map_err(|_| Bad::Format)?;
    let step = if with_ttl { 3 } else { 2 };
    if lp.is_empty() || lp.len() % step != 0 {
        return Err(Bad::Format);
    }
    let mut hash = Hash::with_hint(lp.len() / step, limits);
    let mut field_buf = [0u8; DIGITS_MAX];
    let mut value_buf = [0u8; DIGITS_MAX];
    // Walked and not indexed, for the reason `read_zset_listpack` gives: element
    // `i` of a listpack costs a walk from the front.
    let mut walk = lp.iter();
    while let Some(entry) = walk.next() {
        let field = text(entry, &mut field_buf).to_vec();
        let value = text(walk.next().ok_or(Bad::Format)?, &mut value_buf).to_vec();
        // The packed form holds the deadline as an absolute time, not as a
        // distance from the header, which is the one place the two hash layouts
        // disagree about the same number.
        let at = if with_ttl {
            match walk.next().ok_or(Bad::Format)? {
                Entry::Int(0) => None,
                Entry::Int(n) => Some(u64::try_from(n).map_err(|_| Bad::Format)?),
                Entry::Str(_) => return Err(Bad::Format),
            }
        } else {
            None
        };
        put_field(&mut hash, &field, &value, at, limits, now);
    }
    let _ = soonest;
    if hash.is_empty() {
        return Err(Bad::Format);
    }
    Ok(Body::Hash(hash))
}

/// A field's absolute deadline from the header and its stored distance.
const fn deadline(soonest: u64, ttl: u64) -> Option<u64> {
    if ttl == 0 {
        None
    } else {
        Some(soonest + ttl - 1)
    }
}

/// Put one field in, unless its deadline has already gone.
fn put_field(
    hash: &mut Hash,
    field: &[u8],
    value: &[u8],
    at: Option<u64>,
    limits: &hash::Limits,
    now: u64,
) {
    if let Some(at) = at
        && at <= now
    {
        return;
    }
    hash.set(field, value, limits);
    if let Some(at) = at {
        hash.expire(field, at, crate::ttl::Cond::Always, now);
    }
}

/// A listpack entry as bytes, formatting an integer into the caller's buffer.
fn text<'a>(entry: Entry<'a>, buf: &'a mut [u8; DIGITS_MAX]) -> &'a [u8] {
    match entry {
        Entry::Int(n) => num::i64_digits(buf, n),
        Entry::Str(s) => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttl::{Ask, Cond};

    fn limits() -> (set::Limits, hash::Limits, list::Limits, zset::Limits) {
        (
            set::Limits::DEFAULT,
            hash::Limits::DEFAULT,
            list::Limits::default(),
            zset::Limits::DEFAULT,
        )
    }

    fn round_trip(body: Body) -> Body {
        let (s, h, l, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &l,
            zset: &z,
        };
        let rec = Record::new(body, None);
        let payload = dump(&rec).expect("this type has an RDB shape");
        load(&payload, all, 0).expect("what we wrote we can read")
    }

    fn set_of(members: &[&[u8]]) -> Set {
        let l = set::Limits::DEFAULT;
        let mut set = Set::with_hint(members[0], members.len(), &l);
        for m in members {
            set.add(m, &l);
        }
        set
    }

    #[test]
    fn a_payload_is_ten_bytes_longer_than_the_object() {
        let rec = Record::new(Body::String(b"hello".to_vec()), None);
        let payload = dump(&rec).expect("a string has an RDB shape");
        // One type byte, one length byte, five of text, and the footer.
        assert_eq!(payload.len(), 1 + 1 + 5 + FOOTER);
        assert_eq!(payload[0], T_STRING);
    }

    #[test]
    fn a_string_that_is_a_number_goes_out_as_one() {
        let rec = Record::new(Body::String(b"1234".to_vec()), None);
        let payload = dump(&rec).expect("a string has an RDB shape");
        // The type byte, the encoding byte, two bytes of integer, and the footer.
        assert_eq!(payload.len(), 1 + 1 + 2 + FOOTER);
        let Body::String(back) = round_trip(Body::String(b"1234".to_vec())) else {
            panic!("a string came back as something else");
        };
        assert_eq!(back, b"1234");
    }

    /// A number with a leading zero is not the same string as the number, so it
    /// has to stay text or the round trip changes the value.
    #[test]
    fn a_string_that_only_looks_like_a_number_stays_text() {
        for s in [&b"007"[..], b"+7", b"-0", b" 7", b"9223372036854775808"] {
            let Body::String(back) = round_trip(Body::String(s.to_vec())) else {
                panic!("a string came back as something else");
            };
            assert_eq!(back, s, "{} did not survive", String::from_utf8_lossy(s));
        }
    }

    #[test]
    fn every_integer_width_survives() {
        for n in [0i64, 1, -1, 127, -128, 128, -129, 32767, -32768, 32768] {
            let mut buf = [0u8; DIGITS_MAX];
            let s = num::i64_digits(&mut buf, n).to_vec();
            let Body::String(back) = round_trip(Body::String(s.clone())) else {
                panic!("a string came back as something else");
            };
            assert_eq!(back, s, "{n} did not survive");
        }
        // Past `i32` there is no encoding, so it goes as digits and still has to
        // come back the same.
        let big = b"2147483648".to_vec();
        let Body::String(back) = round_trip(Body::String(big.clone())) else {
            panic!("a string came back as something else");
        };
        assert_eq!(back, big);
    }

    #[test]
    fn a_set_comes_back_with_the_same_members() {
        let set = set_of(&[b"alpha", b"beta", b"gamma"]);
        let Body::Set(back) = round_trip(Body::Set(set)) else {
            panic!("a set came back as something else");
        };
        assert_eq!(back.len(), 3);
        for m in [&b"alpha"[..], b"beta", b"gamma"] {
            assert!(
                back.contains(m),
                "{} went missing",
                String::from_utf8_lossy(m)
            );
        }
    }

    /// An all integer set is held as an intset here and the round trip has to
    /// land it back in the same band, not in a listpack that happens to hold the
    /// same members.
    #[test]
    fn an_integer_set_comes_back_as_an_integer_set() {
        let set = set_of(&[b"1", b"2", b"3"]);
        let was = set.encoding();
        let Body::Set(back) = round_trip(Body::Set(set)) else {
            panic!("a set came back as something else");
        };
        assert_eq!(back.encoding(), was);
        assert_eq!(back.len(), 3);
        assert!(back.contains(b"2"));
    }

    #[test]
    fn a_list_keeps_its_order() {
        let l = list::Limits::default();
        let mut list = List::new();
        for v in [&b"one"[..], b"two", b"three"] {
            list.push_back(v, &l);
        }
        let Body::List(back) = round_trip(Body::List(list)) else {
            panic!("a list came back as something else");
        };
        let mut seen = Vec::new();
        for e in back.iter() {
            let mut buf = Vec::new();
            e.write_to(&mut buf);
            seen.push(buf);
        }
        assert_eq!(
            seen,
            vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
        );
    }

    #[test]
    fn a_sorted_set_keeps_its_scores() {
        let l = zset::Limits::DEFAULT;
        let mut zset = Zset::new();
        zset.add(b"a", 1.5, &l);
        zset.add(b"b", -2.0, &l);
        zset.add(b"c", f64::INFINITY, &l);
        let Body::Zset(back) = round_trip(Body::Zset(zset)) else {
            panic!("a sorted set came back as something else");
        };
        assert_eq!(back.score(b"a"), Some(1.5));
        assert_eq!(back.score(b"b"), Some(-2.0));
        assert_eq!(back.score(b"c"), Some(f64::INFINITY));
    }

    #[test]
    fn a_hash_with_no_deadlines_uses_the_plain_type() {
        let l = hash::Limits::DEFAULT;
        let mut hash = Hash::new();
        // Past the packed band, so this is the table and there is no blob to
        // copy. The small case is the listpack one and it is tested below.
        for i in 0..1000 {
            hash.set(format!("f{i}").as_bytes(), b"1", &l);
        }
        assert_eq!(hash.encoding(), hash::Encoding::Hashtable);
        let rec = Record::new(Body::Hash(hash.clone()), None);
        assert_eq!(dump(&rec).expect("a hash has an RDB shape")[0], T_HASH);
        let Body::Hash(back) = round_trip(Body::Hash(hash)) else {
            panic!("a hash came back as something else");
        };
        assert_eq!(back.len(), 1000);
        assert_eq!(back.get(b"f7").map(|v| v.byte_len()), Some(1));
    }

    /// A value that is one packed blob goes out as the blob.
    ///
    /// The type byte is the thing being pinned here. Every one of these round
    /// trips already, through the walk, and the point of the check is that it is
    /// no longer going through the walk.
    #[test]
    fn a_packed_value_goes_out_as_its_blob() {
        let (sl, hl, _, zl) = limits();
        let mut hash = Hash::new();
        hash.set(b"one", b"1", &hl);
        hash.set(b"two", b"2", &hl);
        assert_eq!(hash.encoding(), hash::Encoding::Listpack);

        let mut set = Set::new();
        set.add(b"alpha", &sl);
        set.add(b"beta", &sl);
        assert_eq!(set.encoding(), set::Encoding::Listpack);

        let mut ints = Set::new();
        ints.add(b"1", &sl);
        ints.add(b"9", &sl);
        assert_eq!(ints.encoding(), set::Encoding::Intset);

        let mut zset = Zset::new();
        zset.add(b"a", 1.5, &zl);
        assert_eq!(zset.encoding(), zset::Encoding::Listpack);

        for (want, body) in [
            (T_HASH_LISTPACK, Body::Hash(hash)),
            (T_SET_LISTPACK, Body::Set(set)),
            (T_SET_INTSET, Body::Set(ints)),
            (T_ZSET_LISTPACK, Body::Zset(zset)),
        ] {
            let rec = Record::new(body.clone(), None);
            let payload = dump(&rec).expect("a packed value has an RDB shape");
            assert_eq!(payload[0], want, "wrong type byte for {body:?}");
            // And the walk is gone, not merely bypassed: what comes back has to
            // be the same value or the copy was of the wrong bytes.
            assert_eq!(
                format!("{:?}", round_trip(body.clone())),
                format!("{body:?}")
            );
        }
    }

    /// A hash that has been widened for deadlines is walked even once they have
    /// all gone, because the blob it holds still has the third element per field
    /// and `HASH_LISTPACK` has no room for it.
    #[test]
    fn a_widened_hash_is_not_copied() {
        let l = hash::Limits::DEFAULT;
        let mut hash = Hash::new();
        hash.set(b"one", b"1", &l);
        hash.expire(b"one", 5_000, Cond::Always, 0);
        hash.persist(b"one");
        // The bound leans early and only a reap that walks puts it right, so
        // this is what it takes to get a hash that is on the wider band and has
        // nothing left to say about deadlines.
        hash.reap(6_000);
        assert_eq!(hash.encoding(), hash::Encoding::ListpackEx);
        assert_eq!(hash.soonest_deadline(), None);
        let rec = Record::new(Body::Hash(hash.clone()), None);
        assert_eq!(dump(&rec).expect("a hash has an RDB shape")[0], T_HASH);
        let Body::Hash(back) = round_trip(Body::Hash(hash)) else {
            panic!("a hash came back as something else");
        };
        assert_eq!(back.len(), 1);
        assert_eq!(back.deadline(b"one"), Ask::NoDeadline);
    }

    #[test]
    fn a_hash_carries_its_field_deadlines_across() {
        let l = hash::Limits::DEFAULT;
        let mut hash = Hash::new();
        hash.set(b"keep", b"1", &l);
        hash.set(b"timed", b"2", &l);
        hash.expire(b"timed", 5_000, Cond::Always, 1_000);
        let rec = Record::new(Body::Hash(hash.clone()), None);
        assert_eq!(
            dump(&rec).expect("a hash has an RDB shape")[0],
            T_HASH_METADATA
        );
        let (s, h, li, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &li,
            zset: &z,
        };
        let payload = dump(&rec).expect("a hash has an RDB shape");
        let Body::Hash(back) = load(&payload, all, 1_000).expect("it reads back") else {
            panic!("a hash came back as something else");
        };
        assert_eq!(back.len(), 2);
        assert_eq!(back.deadline(b"timed"), crate::ttl::Ask::At(5_000));
        assert_eq!(back.deadline(b"keep"), crate::ttl::Ask::NoDeadline);
    }

    /// A field whose deadline went while the payload was in flight is not put
    /// back, because the next read would delete it anyway and a count that is
    /// wrong until somebody looks is worse than a field that never arrived.
    #[test]
    fn a_field_that_expired_in_transit_does_not_come_back() {
        let l = hash::Limits::DEFAULT;
        let mut hash = Hash::new();
        hash.set(b"keep", b"1", &l);
        hash.set(b"gone", b"2", &l);
        hash.expire(b"gone", 5_000, Cond::Always, 1_000);
        let rec = Record::new(Body::Hash(hash), None);
        let payload = dump(&rec).expect("a hash has an RDB shape");
        let (s, h, li, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &li,
            zset: &z,
        };
        let Body::Hash(back) = load(&payload, all, 9_000).expect("it reads back") else {
            panic!("a hash came back as something else");
        };
        assert_eq!(back.len(), 1);
        assert!(back.contains(b"keep"));
        assert!(!back.contains(b"gone"));
    }

    #[test]
    fn a_flipped_byte_is_caught() {
        let rec = Record::new(Body::String(b"hello there".to_vec()), None);
        let good = dump(&rec).expect("a string has an RDB shape");
        for i in 0..good.len() {
            let mut bad = good.clone();
            bad[i] ^= 1;
            let (s, h, l, z) = limits();
            let all = Limits {
                set: &s,
                hash: &h,
                list: &l,
                zset: &z,
            };
            assert!(
                load(&bad, all, 0).is_err(),
                "byte {i} could be changed without anything noticing"
            );
        }
    }

    #[test]
    fn a_payload_from_a_newer_server_is_refused() {
        let rec = Record::new(Body::String(b"hello".to_vec()), None);
        let mut payload = dump(&rec).expect("a string has an RDB shape");
        let n = payload.len();
        payload[n - 10] = 99;
        // The checksum has to be put right, or this would pass for the wrong
        // reason and the version check would never be reached.
        let crc = crc64(0, &payload[..n - 8]);
        payload[n - 8..].copy_from_slice(&crc.to_le_bytes());
        let (s, h, l, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &l,
            zset: &z,
        };
        assert_eq!(load(&payload, all, 0).unwrap_err(), Bad::Footer);
    }

    /// Every version up to the one a current Redis stamps is read, and the one
    /// after it is not.
    ///
    /// This is the check that was missing when `RESTORE` was turning down every
    /// payload a real 8.10.1 produced. The old code compared against the version
    /// it writes, which is deliberately old so that old servers accept us, so
    /// making one number do both jobs meant refusing everything modern.
    #[test]
    fn a_payload_is_read_up_to_the_version_that_has_been_checked() {
        let rec = Record::new(Body::String(b"hello".to_vec()), None);
        let (s, h, l, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &l,
            zset: &z,
        };
        for stamp in [VERSION, READS_UP_TO, READS_UP_TO + 1] {
            let mut payload = dump(&rec).expect("a string has an RDB shape");
            let n = payload.len();
            payload[n - 10..n - 8].copy_from_slice(&stamp.to_le_bytes());
            let crc = crc64(0, &payload[..n - 8]);
            payload[n - 8..].copy_from_slice(&crc.to_le_bytes());
            let got = load(&payload, all, 0);
            if stamp > READS_UP_TO {
                assert_eq!(got.unwrap_err(), Bad::Footer, "{stamp} should be refused");
            } else {
                assert!(got.is_ok(), "{stamp} should be read");
            }
        }
        const {
            assert!(
                VERSION <= READS_UP_TO,
                "a server that cannot read what it writes is no use to anybody"
            )
        };
    }

    #[test]
    fn a_payload_shorter_than_its_footer_is_refused() {
        for n in 0..FOOTER {
            assert_eq!(unseal(&vec![0u8; n]), Err(Bad::Footer));
        }
    }

    /// Nothing here should be able to panic on bytes a client made up, so the
    /// whole space of short payloads gets tried with a correct footer on it.
    #[test]
    fn arbitrary_bytes_are_an_error_and_not_a_panic() {
        let (s, h, l, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &l,
            zset: &z,
        };
        for kind in 0u8..=26 {
            for len in 0usize..6 {
                for fill in [0u8, 1, 0x40, 0x80, 0x81, 0xc0, 0xc3, 0xff] {
                    let mut body = vec![kind];
                    body.extend(std::iter::repeat_n(fill, len));
                    let payload = seal(body);
                    let _ = load(&payload, all, 0);
                }
            }
        }
    }

    #[test]
    fn lzf_unpacks_a_literal_run() {
        // One control byte saying four literals, then the four.
        assert_eq!(
            unpack(&[3, b'a', b'b', b'c', b'd'], 4).as_deref(),
            Some(&b"abcd"[..])
        );
    }

    /// The case the byte at a time copy exists for: a back reference that reads
    /// bytes it is in the middle of writing.
    #[test]
    fn lzf_unpacks_an_overlapping_reference() {
        // One literal `a`, then a reference one byte back for five bytes. The
        // low five bits of the control byte and the byte after it are the
        // distance, and they are both zero because a distance is stored one
        // less than it is.
        let packed = [0u8, b'a', 3 << 5, 0];
        assert_eq!(unpack(&packed, 6).as_deref(), Some(&b"aaaaaa"[..]));
    }

    #[test]
    fn lzf_refuses_a_reference_to_nothing() {
        assert_eq!(unpack(&[(3 << 5), 0], 5), None);
        assert_eq!(unpack(&[3, b'a'], 4), None);
    }

    #[test]
    fn an_empty_collection_is_not_a_value() {
        let mut body = vec![T_SET];
        put_len(&mut body, 0);
        let payload = seal(body);
        let (s, h, l, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &l,
            zset: &z,
        };
        assert_eq!(load(&payload, all, 0).unwrap_err(), Bad::Format);
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let mut body = vec![T_STRING];
        put_str(&mut body, b"hello");
        body.push(0);
        let payload = seal(body);
        let (s, h, l, z) = limits();
        let all = Limits {
            set: &s,
            hash: &h,
            list: &l,
            zset: &z,
        };
        assert_eq!(load(&payload, all, 0).unwrap_err(), Bad::Format);
    }

    #[test]
    fn every_length_form_round_trips() {
        for n in [0u64, 63, 64, 16383, 16384, u64::from(u32::MAX), 1 << 40] {
            let mut out = Vec::new();
            put_len(&mut out, n);
            let mut r = Reader::new(&out);
            assert_eq!(r.len_or_encoding(), Ok((n, false)), "{n} did not survive");
            assert!(r.done(), "{n} left bytes behind");
        }
    }
}
