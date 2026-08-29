//! The inline band: one packed blob, walked linearly.
//!
//! A collection under a hundred and twenty eight elements is stored as one blob
//! with no index, walked from the front. This is Redis's listpack, in Redis's
//! bytes, and it is the bottom rung of the size ladder in `05` section 4.
//!
//! # Why, and it is not the reason the spec gives
//!
//! `05` section 4.1 argues for this band on speed, citing L6: a dense positional
//! structure probes in about 70 ns where a listpack walk costs 1 to 2 ns, a fifty
//! times gap. Half of that reproduces and half of it does not. A walk here does
//! cost 1.9 ns an element, which is L6's number. A probe in our element table
//! costs 13 ns and not 70, so the gap it was being compared against is not there,
//! and at eight members the table wins on every operation: 13.7 ns against 25.2
//! to find a member that is present, 5.6 against 43.9 to find one that is not,
//! 0.5 ns an element against 1.9 to walk the whole thing, and 240 ns against 280
//! to build it. At a hundred and twenty eight the table is twenty times faster to
//! probe. There is no crossover. `benches/listpack.rs` is where those come from.
//!
//! What the band is actually for is memory, and there the gap is real and the
//! other way round. With an eleven byte member the blob costs 13.1 bytes an
//! element and the table costs 31.0, because the table pays twelve bytes of row
//! and about eight of slot on top of the name while the blob pays an encoding
//! byte and a back length. G8 asks for a set member to cost under three bytes
//! plus its payload. The blob comes in at 2.1 and the table at 20. A server
//! holding a million small hashes is holding them here or it is not holding them.
//!
//! The threshold is not ours to move anyway. `OBJECT ENCODING` has to say
//! `listpack` for exactly the collections Redis says it for, so the promotion
//! points are `hash-max-listpack-entries` and its neighbours whatever we would
//! have picked. Worth knowing which argument is load bearing, though, because the
//! speed one would have sent us looking for a faster walk and the real one sends
//! us to the arena.
//!
//! It is byte compatible with a Redis listpack, not merely similar in spirit.
//! `05` section 4.1 asks for that so an RDB export is a copy rather than a
//! transcode, and it means the encodings, the header, the terminator and the
//! back length are all Redis's. Every boundary here was read off `listpack.c`
//! from the 8.10.1 tarball, which is the same version `yo-compat` pins, and the
//! ones that are easy to get a byte wrong are pinned in the tests with Redis's
//! own numbers written out.
//!
//! ```text
//! +---------+--------+---------+-----+---------+------+
//! | u32 len | u16 n  | entry 0 | ... | entry k | 0xFF |
//! +---------+--------+---------+-----+---------+------+
//!   total bytes, header included         terminator
//! ```
//!
//! An entry is an encoding byte, then its payload, then a back length, and the
//! back length is what makes the walk work in both directions. A forward walk
//! reads the encoding and steps over the payload. A backward walk reads the back
//! length from its last byte leftward and steps over the whole entry, which is
//! how `SPOP` reaches the end of a blob without walking it from the front, and
//! how the downward scan cursor in [`crate::scan`] works in this band.
//!
//! # What is here and what is not
//!
//! Everything a collection needs to hold its elements: append, read by position,
//! find, replace, insert and delete, all of them working on the blob in place.
//! Every one of them is linear in the number of elements, on purpose, because
//! the band is bounded and an index would cost more than it saved.
//!
//! Not the promotion policy. When a collection stops being small is a decision
//! for the collection, since Redis makes it configurable per type and the
//! thresholds have to keep matching `hash-max-listpack-entries` and its
//! neighbours. This module holds elements and says how many bytes they cost.

use yo_common::{parse_i64, push_i64};

/// Header is four bytes of total length and two of element count.
const HDR: usize = 6;

/// The terminator, which is also the encoding byte that means end.
const END: u8 = 0xFF;

/// What the element count field holds when the real count does not fit.
///
/// A collection in this band holds a hundred and twenty eight elements, so this
/// never comes from us. It comes from a listpack somebody else wrote, and the
/// answer to it is to walk and count.
const COUNT_UNKNOWN: u16 = 65535;

/// The largest count that fits in the field.
const COUNT_MAX: usize = 65534;

/// A packed blob of elements.
///
/// Owns its bytes today. When the arena lands under it the bytes move there and
/// this becomes a view, which is why nothing here hands out a `Vec` or takes
/// one back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listpack {
    bytes: Vec<u8>,
}

/// One element, as it is stored.
///
/// Redis stores a member that looks like an integer as an integer, so `SADD s 1`
/// and `SADD s 01` are two different members that are stored two different ways.
/// Handing back which one it was is what lets a caller answer `OBJECT ENCODING`
/// and write an RDB without re-deciding it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry<'a> {
    /// Stored as an integer, in one of the six integer encodings.
    Int(i64),
    /// Stored as bytes.
    Str(&'a [u8]),
}

impl Entry<'_> {
    /// The element as a client would see it, appended to `out`.
    ///
    /// An integer entry is formatted here, which is the same round trip Redis
    /// does on the way out, because the client asked for a member and members
    /// are strings on the wire.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        match self {
            Entry::Int(n) => push_i64(out, *n),
            Entry::Str(s) => out.extend_from_slice(s),
        }
    }

    /// The element as bytes, allocating only for an integer.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }

    /// Whether this element is the given bytes.
    ///
    /// An integer entry is compared as an integer, so the needle is parsed once
    /// by the caller rather than the entry being formatted once per candidate.
    #[must_use]
    #[inline]
    fn is(&self, needle: &[u8], as_int: Option<i64>) -> bool {
        match (self, as_int) {
            (Entry::Int(n), Some(v)) => *n == v,
            (Entry::Int(_), None) => false,
            (Entry::Str(s), _) => *s == needle,
        }
    }
}

impl Default for Listpack {
    fn default() -> Listpack {
        Listpack::new()
    }
}

impl Listpack {
    /// An empty blob, which is a header and a terminator and nothing else.
    #[must_use]
    pub fn new() -> Listpack {
        let mut bytes = Vec::with_capacity(HDR + 1 + 64);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, END]);
        let mut lp = Listpack { bytes };
        lp.set_total(HDR + 1);
        lp
    }

    /// Take bytes somebody else wrote, after checking them.
    ///
    /// An RDB, a `.yo` file and a `RESTORE` all arrive this way, so the walk is
    /// not optional. A blob that does not check out is refused whole rather than
    /// read up to the bad entry, because half a collection is worse than none.
    pub fn from_bytes(bytes: &[u8]) -> Result<Listpack, Malformed> {
        if bytes.len() < HDR + 1 {
            return Err(Malformed::Short);
        }
        let total = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if total != bytes.len() {
            return Err(Malformed::Length);
        }
        if bytes[total - 1] != END {
            return Err(Malformed::Terminator);
        }
        // Walk it. Every entry has to decode, its back length has to agree with
        // how long the entry actually was, and the last one has to land exactly
        // on the terminator rather than past it.
        let mut at = HDR;
        let mut seen = 0usize;
        while at < total - 1 {
            let (_, len) = decode(&bytes[at..total - 1]).ok_or(Malformed::Entry)?;
            let back = backlen_len(len);
            if at + len + back > total - 1 {
                return Err(Malformed::Entry);
            }
            if read_backlen(&bytes[..at + len + back]) != Some(len) {
                return Err(Malformed::BackLength);
            }
            at += len + back;
            seen += 1;
        }
        let count = u16::from_le_bytes([bytes[4], bytes[5]]);
        if count != COUNT_UNKNOWN && count as usize != seen {
            return Err(Malformed::Count);
        }
        Ok(Listpack {
            bytes: bytes.to_vec(),
        })
    }

    /// The bytes, ready to be written to a file or an RDB unchanged.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// How many elements.
    ///
    /// The header answers this, which is why it is here and why it is kept
    /// right on every edit. A blob from elsewhere with an unknown count is
    /// walked instead, once, rather than being rejected.
    #[must_use]
    pub fn len(&self) -> usize {
        let count = u16::from_le_bytes([self.bytes[4], self.bytes[5]]);
        if count == COUNT_UNKNOWN {
            self.iter().count()
        } else {
            count as usize
        }
    }

    /// Whether there is nothing in it, which for Redis means it does not exist.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.len() == HDR + 1
    }

    /// What the blob costs, which is what it costs on disk too.
    #[inline]
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Every element, front to back.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            bytes: &self.bytes,
            at: HDR,
        }
    }

    /// The element at a position, counting from the front.
    ///
    /// Linear, because the blob is linear. That is the whole design: at a
    /// hundred and twenty eight elements the walk is cheaper than the index that
    /// would have avoided it.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Entry<'_>> {
        self.iter().nth(index)
    }

    /// The element at a position, counting from the back.
    ///
    /// Reads the back length of the last entry and steps left, which is what the
    /// trailing length field is for and why a `RPOP` on a small list does not
    /// walk the whole blob.
    #[must_use]
    pub fn get_back(&self, from_end: usize) -> Option<Entry<'_>> {
        let mut end = self.bytes.len() - 1;
        for _ in 0..=from_end {
            // Stepping onto the header means the walk asked for more elements
            // than are here. Without this the header's own bytes decode as an
            // entry and the answer is nonsense rather than nothing.
            if end <= HDR {
                return None;
            }
            let len = read_backlen(&self.bytes[..end])?;
            end = end.checked_sub(len + backlen_len(len))?;
        }
        if end < HDR {
            return None;
        }
        decode(&self.bytes[end..self.bytes.len() - 1]).map(|(e, _)| e)
    }

    /// Where an element is, or nothing.
    ///
    /// `step` is what makes this work for a hash. A hash in this band is field,
    /// value, field, value, so a field lookup is a find with a step of two, which
    /// is the same trick Redis's `lpFind` plays and the reason a hash does not
    /// need a second structure down here.
    #[must_use]
    pub fn find(&self, needle: &[u8], step: usize) -> Option<usize> {
        let as_int = parse_i64(needle);
        let step = step.max(1);
        // Counted down rather than `at % step`, because `step` is a runtime value
        // and the remainder compiles to a real division on every element. That is
        // twenty cycles to answer a question about a two element cycle, and it
        // cost more than the comparison it was guarding.
        let mut until = 0usize;
        for (at, e) in self.iter().enumerate() {
            if until == 0 {
                if e.is(needle, as_int) {
                    return Some(at);
                }
                until = step;
            }
            until -= 1;
        }
        None
    }

    /// Add an element at the end.
    pub fn push(&mut self, value: &[u8]) {
        let at = self.bytes.len() - 1;
        self.splice(at, 0, Some(value), 1);
    }

    /// Put an element in front of the one at `index`.
    ///
    /// An index at or past the end appends, which is what a sorted insert wants
    /// when the new element sorts last and saves the caller a branch.
    pub fn insert(&mut self, index: usize, value: &[u8]) {
        let at = self.offset_of(index).unwrap_or(self.bytes.len() - 1);
        self.splice(at, 0, Some(value), 1);
    }

    /// Overwrite the element at `index`, keeping its position.
    ///
    /// `HSET` on a field that is already there, and `ZADD` on a member whose
    /// score has changed but whose place has not.
    pub fn replace(&mut self, index: usize, value: &[u8]) -> bool {
        let Some(at) = self.offset_of(index) else {
            return false;
        };
        let old = self.entry_bytes(at);
        self.splice(at, old, Some(value), 0);
        true
    }

    /// Take out `count` elements starting at `index`.
    ///
    /// `HDEL` takes two, a field and its value, and it has to take them as one
    /// edit or the blob is briefly a hash with an odd number of entries.
    pub fn delete(&mut self, index: usize, count: usize) -> bool {
        let Some(at) = self.offset_of(index) else {
            return false;
        };
        let mut end = at;
        let mut gone = 0usize;
        while gone < count && end < self.bytes.len() - 1 {
            end += self.entry_bytes(end);
            gone += 1;
        }
        if gone == 0 {
            return false;
        }
        self.splice(at, end - at, None, -(gone as i32));
        true
    }

    /// Byte offset of the element at `index`, or nothing if it is past the end.
    fn offset_of(&self, index: usize) -> Option<usize> {
        let mut at = HDR;
        for _ in 0..index {
            if at >= self.bytes.len() - 1 {
                return None;
            }
            at += self.entry_bytes(at);
        }
        if at >= self.bytes.len() - 1 {
            None
        } else {
            Some(at)
        }
    }

    /// How many bytes the entry at `at` occupies, back length included.
    fn entry_bytes(&self, at: usize) -> usize {
        let (_, len) = decode(&self.bytes[at..self.bytes.len() - 1]).expect("our own blob decodes");
        len + backlen_len(len)
    }

    /// The one edit primitive: drop some bytes, put some back, fix the header.
    ///
    /// Everything that changes the blob goes through here, so there is one place
    /// that can leave the length or the count wrong and it is eleven lines long.
    fn splice(&mut self, at: usize, remove: usize, insert: Option<&[u8]>, delta: i32) {
        let mut buf = [0u8; 16];
        let encoded: Vec<u8> = match insert {
            Some(v) => {
                let (head, payload) = encode(v, &mut buf);
                let mut e = Vec::with_capacity(head.len() + v.len() + 5);
                e.extend_from_slice(head);
                if payload {
                    e.extend_from_slice(v);
                }
                let len = e.len();
                write_backlen(&mut e, len);
                e
            }
            None => Vec::new(),
        };
        self.bytes.splice(at..at + remove, encoded);
        let total = self.bytes.len();
        self.set_total(total);
        let count = i64::from(u16::from_le_bytes([self.bytes[4], self.bytes[5]]));
        let count = usize::try_from(count + i64::from(delta)).unwrap_or(0);
        let count = u16::try_from(count.min(COUNT_MAX)).expect("clamped to the field");
        self.bytes[4..6].copy_from_slice(&count.to_le_bytes());
    }

    /// Write the total length into the header.
    fn set_total(&mut self, total: usize) {
        let total = u32::try_from(total).expect("the inline band is far under 4 GiB");
        self.bytes[0..4].copy_from_slice(&total.to_le_bytes());
    }
}

/// A forward walk.
#[derive(Debug)]
pub struct Iter<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = Entry<'a>;

    #[inline]
    fn next(&mut self) -> Option<Entry<'a>> {
        if self.at >= self.bytes.len() - 1 {
            return None;
        }
        let (entry, len) = decode(&self.bytes[self.at..self.bytes.len() - 1])?;
        self.at += len + backlen_len(len);
        Some(entry)
    }
}

/// Why a blob from somewhere else was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Malformed {
    /// Shorter than an empty listpack.
    Short,
    /// The header's total length is not the length of what arrived.
    Length,
    /// It does not end in a terminator.
    Terminator,
    /// An entry's encoding is not one of the fourteen.
    Entry,
    /// An entry's back length disagrees with how long the entry is.
    BackLength,
    /// The header's element count is not how many elements are in it.
    Count,
}

/// The encoding bytes for an element, written into `buf`.
///
/// Redis's fourteen encodings, and the choice between them is the same one
/// `lpEncodeGetType` makes: an element that parses as an integer is stored as
/// one, in the narrowest form that holds it, and everything else is stored as
/// bytes with a length that is six, twelve or thirty two bits wide.
///
/// The flag says whether the element's own bytes follow the encoding. An integer
/// is entirely inside its encoding, and that is where the memory target in G8
/// comes from: a set of small integers costs two bytes an element, encoding and
/// back length, with nothing else stored at all.
fn encode<'b>(v: &[u8], buf: &'b mut [u8; 16]) -> (&'b [u8], bool) {
    if let Some(n) = parse_i64(v) {
        let head: &[u8] = match n {
            0..=127 => {
                buf[0] = n as u8;
                &buf[..1]
            }
            -4096..=4095 => {
                let u = (n as u16) & 0x1FFF;
                buf[0] = 0xC0 | (u >> 8) as u8;
                buf[1] = (u & 0xFF) as u8;
                &buf[..2]
            }
            -32768..=32767 => {
                buf[0] = 0xF1;
                buf[1..3].copy_from_slice(&(n as i16).to_le_bytes());
                &buf[..3]
            }
            -8_388_608..=8_388_607 => {
                buf[0] = 0xF2;
                buf[1..4].copy_from_slice(&(n as i32).to_le_bytes()[..3]);
                &buf[..4]
            }
            -2_147_483_648..=2_147_483_647 => {
                buf[0] = 0xF3;
                buf[1..5].copy_from_slice(&(n as i32).to_le_bytes());
                &buf[..5]
            }
            _ => {
                buf[0] = 0xF4;
                buf[1..9].copy_from_slice(&n.to_le_bytes());
                &buf[..9]
            }
        };
        return (head, false);
    }
    let head: &[u8] = match v.len() {
        0..=63 => {
            buf[0] = 0x80 | v.len() as u8;
            &buf[..1]
        }
        64..=4095 => {
            buf[0] = 0xE0 | (v.len() >> 8) as u8;
            buf[1] = (v.len() & 0xFF) as u8;
            &buf[..2]
        }
        _ => {
            buf[0] = 0xF0;
            buf[1..5].copy_from_slice(&(v.len() as u32).to_le_bytes());
            &buf[..5]
        }
    };
    (head, true)
}

/// Read one entry, and say how many bytes it took before its back length.
///
/// Inlined on purpose. It hands back a fat enum and a length, and left out of
/// line that pair goes through memory once per element, which is most of what a
/// walk costs.
#[inline]
fn decode(b: &[u8]) -> Option<(Entry<'_>, usize)> {
    let first = *b.first()?;
    // A string encoding's payload starts after its length, so both arms below
    // hand back the same pair and the caller does not care which it was.
    let (at, len) = match first {
        0x00..=0x7F => return Some((Entry::Int(i64::from(first)), 1)),
        0x80..=0xBF => (1, (first & 0x3F) as usize),
        0xC0..=0xDF => {
            let raw = (u16::from(first & 0x1F) << 8) | u16::from(*b.get(1)?);
            // Thirteen bits, signed, so the top bit of the thirteen is the sign.
            let n = if raw & 0x1000 != 0 {
                i64::from(raw) - 8192
            } else {
                i64::from(raw)
            };
            return Some((Entry::Int(n), 2));
        }
        0xE0..=0xEF => {
            let lo = *b.get(1)?;
            (2, (usize::from(first & 0x0F) << 8) | usize::from(lo))
        }
        0xF0 => {
            let n = u32::from_le_bytes([*b.get(1)?, *b.get(2)?, *b.get(3)?, *b.get(4)?]);
            (5, n as usize)
        }
        0xF1 => {
            let n = i16::from_le_bytes([*b.get(1)?, *b.get(2)?]);
            return Some((Entry::Int(i64::from(n)), 3));
        }
        0xF2 => {
            // Twenty four bits, sign extended by putting them in the top of a
            // thirty two bit word and shifting back down.
            let n = i32::from_le_bytes([0, *b.get(1)?, *b.get(2)?, *b.get(3)?]) >> 8;
            return Some((Entry::Int(i64::from(n)), 4));
        }
        0xF3 => {
            let n = i32::from_le_bytes([*b.get(1)?, *b.get(2)?, *b.get(3)?, *b.get(4)?]);
            return Some((Entry::Int(i64::from(n)), 5));
        }
        0xF4 => {
            let mut w = [0u8; 8];
            w.copy_from_slice(b.get(1..9)?);
            return Some((Entry::Int(i64::from_le_bytes(w)), 9));
        }
        // 0xF5 to 0xFE are unused by Redis, and 0xFF is the terminator, which
        // the caller has already stopped before.
        _ => return None,
    };
    let s = b.get(at..at + len)?;
    Some((Entry::Str(s), at + len))
}

/// How many bytes the back length of an entry of `len` bytes takes.
///
/// Seven bits a byte, so the boundaries are `2^7 - 1`, `2^14 - 1` and so on, and
/// they are Redis's `lpEncodeBacklenBytes` boundaries exactly. They are checked
/// against the real ones in the tests rather than taken on trust, because a
/// listpack whose back lengths are a byte out is one Redis walks off the end of.
#[inline]
const fn backlen_len(len: usize) -> usize {
    if len <= 127 {
        1
    } else if len <= 16383 {
        2
    } else if len <= 2_097_151 {
        3
    } else if len <= 268_435_455 {
        4
    } else {
        5
    }
}

/// Append the back length for an entry of `len` bytes.
///
/// The first byte holds the high seven bits and every later byte has its top bit
/// set, which is what lets it be read from the right hand end leftward.
fn write_backlen(out: &mut Vec<u8>, len: usize) {
    match backlen_len(len) {
        1 => out.push(len as u8),
        2 => {
            out.push((len >> 7) as u8);
            out.push((len & 127) as u8 | 128);
        }
        3 => {
            out.push((len >> 14) as u8);
            out.push(((len >> 7) & 127) as u8 | 128);
            out.push((len & 127) as u8 | 128);
        }
        4 => {
            out.push((len >> 21) as u8);
            out.push(((len >> 14) & 127) as u8 | 128);
            out.push(((len >> 7) & 127) as u8 | 128);
            out.push((len & 127) as u8 | 128);
        }
        _ => {
            out.push((len >> 28) as u8);
            out.push(((len >> 21) & 127) as u8 | 128);
            out.push(((len >> 14) & 127) as u8 | 128);
            out.push(((len >> 7) & 127) as u8 | 128);
            out.push((len & 127) as u8 | 128);
        }
    }
}

/// Read a back length that ends at the last byte of `upto`.
///
/// Walks left while the top bit is set, seven bits at a time, which is the
/// mirror image of how it was written.
fn read_backlen(upto: &[u8]) -> Option<usize> {
    let mut val = 0usize;
    let mut shift = 0u32;
    let mut at = upto.len().checked_sub(1)?;
    loop {
        let b = *upto.get(at)?;
        val |= usize::from(b & 127) << shift;
        if b & 128 == 0 {
            return Some(val);
        }
        shift += 7;
        if shift > 28 {
            return None;
        }
        at = at.checked_sub(1)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(members: &[&[u8]]) -> Listpack {
        let mut lp = Listpack::new();
        for m in members {
            lp.push(m);
        }
        lp
    }

    fn all(lp: &Listpack) -> Vec<Vec<u8>> {
        lp.iter().map(|e| e.to_vec()).collect()
    }

    #[test]
    fn an_empty_blob_is_a_header_and_a_terminator() {
        let lp = Listpack::new();
        assert!(lp.is_empty());
        assert_eq!(lp.len(), 0);
        assert_eq!(lp.byte_len(), 7);
        assert_eq!(lp.as_bytes(), &[7, 0, 0, 0, 0, 0, 0xFF]);
        assert_eq!(lp.get(0), None);
        assert_eq!(lp.iter().count(), 0);
    }

    #[test]
    fn what_goes_in_comes_out_in_order() {
        let lp = of(&[b"one", b"two", b"three"]);
        assert_eq!(lp.len(), 3);
        assert_eq!(
            all(&lp),
            vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
        );
        assert_eq!(lp.get(1), Some(Entry::Str(b"two")));
        assert_eq!(lp.get(3), None);
    }

    /// Redis stores a member that parses as an integer as an integer, and the
    /// narrowest one that holds it. Every boundary is here because every one of
    /// them is a different encoding byte, and a byte wrong is an RDB Redis will
    /// not read.
    #[test]
    fn an_integer_takes_the_narrowest_encoding_that_holds_it() {
        for (text, first, len) in [
            (&b"0"[..], 0x00u8, 1usize),
            (b"127", 0x7F, 1),
            (b"128", 0xC0, 2),
            (b"4095", 0xCF, 2),
            (b"-4096", 0xD0, 2),
            (b"-1", 0xDF, 2),
            (b"4096", 0xF1, 3),
            (b"-4097", 0xF1, 3),
            (b"32767", 0xF1, 3),
            (b"32768", 0xF2, 4),
            (b"8388607", 0xF2, 4),
            (b"8388608", 0xF3, 5),
            (b"2147483647", 0xF3, 5),
            (b"2147483648", 0xF4, 9),
            (b"-9223372036854775808", 0xF4, 9),
        ] {
            let lp = of(&[text]);
            let at = HDR;
            assert_eq!(
                lp.as_bytes()[at],
                first,
                "{} took the wrong encoding",
                String::from_utf8_lossy(text)
            );
            assert_eq!(lp.byte_len(), HDR + len + 1 + 1, "{first:#x}");
            assert_eq!(
                lp.get(0),
                Some(Entry::Int(parse_i64(text).expect("a number"))),
                "{first:#x}"
            );
            assert_eq!(all(&lp), vec![text.to_vec()], "and it formats back");
        }
    }

    /// What `string2ll` refuses is a string, and it has to stay one, because
    /// `SADD s 01` and `SADD s 1` are two different members to Redis.
    #[test]
    fn something_that_only_looks_like_a_number_stays_a_string() {
        for text in [&b"01"[..], b"+1", b"1 ", b" 1", b"1.0", b"-0", b""] {
            let lp = of(&[text]);
            assert_eq!(
                lp.get(0),
                Some(Entry::Str(text)),
                "{}",
                String::from_utf8_lossy(text)
            );
        }
    }

    #[test]
    fn a_string_takes_the_narrowest_length_field() {
        for (len, first, head) in [(1usize, 0x81u8, 1usize), (63, 0xBF, 1), (64, 0xE0, 2)] {
            let s = vec![b'x'; len];
            let lp = of(&[&s]);
            assert_eq!(lp.as_bytes()[HDR], first, "length {len}");
            assert_eq!(
                lp.byte_len(),
                HDR + head + len + backlen_len(head + len) + 1
            );
            assert_eq!(lp.get(0), Some(Entry::Str(&s[..])));
        }
    }

    /// Over 4095 bytes the length field is the thirty two bit one, and the entry
    /// is long enough that its own back length needs two bytes, which is the
    /// other boundary in the same test.
    #[test]
    fn a_long_string_takes_the_wide_length_and_a_wide_back_length() {
        let s = vec![b'y'; 5000];
        let lp = of(&[&s, b"after"]);
        assert_eq!(lp.as_bytes()[HDR], 0xF0);
        assert_eq!(backlen_len(5005), 2);
        assert_eq!(lp.get(0), Some(Entry::Str(&s[..])));
        assert_eq!(lp.get(1), Some(Entry::Str(b"after")));
        assert_eq!(lp.get_back(0), Some(Entry::Str(b"after")));
        assert_eq!(lp.get_back(1), Some(Entry::Str(&s[..])));
    }

    #[test]
    fn the_back_length_reads_the_same_as_it_was_written() {
        for len in [1usize, 127, 128, 16382, 16383, 16384, 2_097_150, 2_097_151] {
            let mut out = Vec::new();
            write_backlen(&mut out, len);
            assert_eq!(out.len(), backlen_len(len), "length {len}");
            assert_eq!(read_backlen(&out), Some(len), "length {len}");
        }
    }

    /// Reading itself back is not enough, because a wrong boundary is wrong
    /// consistently. These are Redis's `lpEncodeBacklenBytes` boundaries written
    /// out from `listpack.c` at 8.10.1, which is the version `yo-compat` pins,
    /// and they are what a listpack from an RDB will have been written with.
    #[test]
    fn the_back_length_boundaries_are_the_ones_redis_uses() {
        for (len, want) in [
            (0usize, 1usize),
            (127, 1),
            (128, 2),
            (16383, 2),
            (16384, 3),
            (2_097_151, 3),
            (2_097_152, 4),
            (268_435_455, 4),
            (268_435_456, 5),
        ] {
            assert_eq!(backlen_len(len), want, "an entry of {len} bytes");
        }
    }

    #[test]
    fn a_walk_backward_reaches_every_element() {
        let lp = of(&[b"a", b"bb", b"1", b"999999", b"dddd"]);
        let back: Vec<Vec<u8>> = (0..5)
            .map(|i| lp.get_back(i).expect("in range").to_vec())
            .collect();
        let mut forward = all(&lp);
        forward.reverse();
        assert_eq!(back, forward);
        assert_eq!(lp.get_back(5), None);
    }

    #[test]
    fn find_locates_an_element_however_it_is_stored() {
        let lp = of(&[b"alpha", b"42", b"01", b"beta"]);
        assert_eq!(lp.find(b"alpha", 1), Some(0));
        assert_eq!(lp.find(b"42", 1), Some(1), "stored as an integer");
        assert_eq!(lp.find(b"01", 1), Some(2), "stored as a string");
        assert_eq!(lp.find(b"beta", 1), Some(3));
        assert_eq!(lp.find(b"gamma", 1), None);
        assert_eq!(lp.find(b"1", 1), None, "01 is not 1");
    }

    /// A hash in this band is field, value, field, value, and a field lookup has
    /// to skip the values or a value that happens to equal a field name comes
    /// back as one.
    #[test]
    fn find_with_a_step_only_looks_at_the_fields() {
        let lp = of(&[b"name", b"age", b"age", b"41"]);
        assert_eq!(lp.find(b"name", 2), Some(0));
        assert_eq!(lp.find(b"age", 2), Some(2), "the value at 1 is not a field");
        assert_eq!(lp.find(b"41", 2), None);
        assert_eq!(lp.get(3), Some(Entry::Int(41)));
    }

    #[test]
    fn inserting_puts_an_element_in_front_of_another() {
        let mut lp = of(&[b"a", b"c"]);
        lp.insert(1, b"b");
        assert_eq!(all(&lp), vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        lp.insert(0, b"start");
        lp.insert(99, b"end");
        assert_eq!(lp.len(), 5);
        assert_eq!(all(&lp)[0], b"start".to_vec());
        assert_eq!(all(&lp)[4], b"end".to_vec());
    }

    #[test]
    fn replacing_keeps_the_position_and_can_change_the_size() {
        let mut lp = of(&[b"a", b"b", b"c"]);
        assert!(lp.replace(1, b"a much longer value than before"));
        assert_eq!(lp.len(), 3);
        assert_eq!(all(&lp)[1], b"a much longer value than before".to_vec());
        assert!(lp.replace(1, b"7"));
        assert_eq!(lp.get(1), Some(Entry::Int(7)), "and can shrink to an int");
        assert_eq!(all(&lp), vec![b"a".to_vec(), b"7".to_vec(), b"c".to_vec()]);
        assert!(!lp.replace(9, b"nothing there"));
    }

    #[test]
    fn deleting_takes_out_a_run_in_one_edit() {
        let mut lp = of(&[b"f1", b"v1", b"f2", b"v2", b"f3", b"v3"]);
        assert!(lp.delete(2, 2), "a field and its value together");
        assert_eq!(lp.len(), 4);
        assert_eq!(
            all(&lp),
            vec![
                b"f1".to_vec(),
                b"v1".to_vec(),
                b"f3".to_vec(),
                b"v3".to_vec()
            ]
        );
        assert!(!lp.delete(9, 1));
        assert!(
            lp.delete(0, 99),
            "asking for more than is there takes the rest"
        );
        assert!(lp.is_empty());
        assert_eq!(lp.len(), 0);
    }

    /// Every edit has to leave the header right, because the header is what
    /// `from_bytes` checks and what a reader trusts.
    #[test]
    fn the_header_survives_every_edit() {
        let mut lp = Listpack::new();
        for i in 0..64u32 {
            lp.push(format!("member-{i}").as_bytes());
        }
        for i in 0..20 {
            lp.delete(i, 1);
            lp.replace(i, b"replaced");
            lp.insert(i, b"1234567");
        }
        let total = u32::from_le_bytes([lp.bytes[0], lp.bytes[1], lp.bytes[2], lp.bytes[3]]);
        assert_eq!(total as usize, lp.byte_len());
        assert_eq!(lp.len(), lp.iter().count());
        assert_eq!(Listpack::from_bytes(lp.as_bytes()), Ok(lp.clone()));
    }

    #[test]
    fn a_blob_round_trips_through_its_bytes() {
        let lp = of(&[b"a", b"12345", b"", &[b'z'; 200]]);
        let back = Listpack::from_bytes(lp.as_bytes()).expect("our own bytes check out");
        assert_eq!(back, lp);
        assert_eq!(all(&back), all(&lp));
    }

    /// Bytes from an RDB or a `RESTORE` are somebody else's, so every way they
    /// can be wrong is a refusal and not a panic.
    #[test]
    fn a_blob_that_does_not_check_out_is_refused() {
        assert_eq!(Listpack::from_bytes(&[]), Err(Malformed::Short));
        assert_eq!(Listpack::from_bytes(&[0; 4]), Err(Malformed::Short));

        let good = of(&[b"alpha", b"beta"]);

        let mut wrong_len = good.as_bytes().to_vec();
        wrong_len[0] = 99;
        assert_eq!(Listpack::from_bytes(&wrong_len), Err(Malformed::Length));

        let mut no_end = good.as_bytes().to_vec();
        let last = no_end.len() - 1;
        no_end[last] = 0x00;
        assert_eq!(Listpack::from_bytes(&no_end), Err(Malformed::Terminator));

        let mut wrong_count = good.as_bytes().to_vec();
        wrong_count[4] = 7;
        assert_eq!(Listpack::from_bytes(&wrong_count), Err(Malformed::Count));

        let mut bad_entry = good.as_bytes().to_vec();
        bad_entry[HDR] = 0xF7;
        assert_eq!(Listpack::from_bytes(&bad_entry), Err(Malformed::Entry));

        let mut bad_back = good.as_bytes().to_vec();
        bad_back[HDR + 6] = 3;
        assert_eq!(Listpack::from_bytes(&bad_back), Err(Malformed::BackLength));
    }

    /// A blob written by somebody who did not keep the count, which Redis does
    /// above 65534 elements, is walked rather than rejected.
    #[test]
    fn an_unknown_count_is_walked_and_not_refused() {
        let mut lp = of(&[b"a", b"b", b"c"]);
        lp.bytes[4..6].copy_from_slice(&COUNT_UNKNOWN.to_le_bytes());
        let back = Listpack::from_bytes(lp.as_bytes()).expect("unknown is allowed");
        assert_eq!(back.len(), 3);
    }

    fn hex(lp: &Listpack) -> String {
        lp.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The claim this module makes is that the bytes are Redis's bytes, and the
    /// only way to check that is against Redis's bytes.
    ///
    /// These came out of `lpAppend` in `listpack.c` from the 8.10.1 tarball,
    /// compiled and run, not out of reading the source and working out what it
    /// would do. The vectors are the boundaries: every integer encoding and both
    /// ends of each, the strings that look like integers and must not become
    /// them, the empty string, and a hash shaped pack where a value is an
    /// integer and the fields around it are not.
    #[test]
    fn the_bytes_are_the_ones_redis_writes() {
        assert_eq!(hex(&Listpack::new()), "070000000000ff");

        assert_eq!(
            hex(&of(&[b"one", b"two", b"three"])),
            "180000000300836f6e65048374776f0485746872656506ff"
        );

        let ints: Vec<&[u8]> = vec![
            b"0",
            b"127",
            b"128",
            b"4095",
            b"-4096",
            b"-1",
            b"4096",
            b"-4097",
            b"32767",
            b"32768",
            b"8388607",
            b"8388608",
            b"2147483647",
            b"2147483648",
            b"-9223372036854775808",
        ];
        assert_eq!(
            hex(&of(&ints)),
            "4d0000000f0000017f01c08002cfff02d00002dfff02f1001003f1ffef03f1ff7f\
             03f200800004f2ffff7f04f30000800005f3ffffff7f05f4000000800000000009\
             f4000000000000008009ff"
        );

        let not_ints: Vec<&[u8]> = vec![b"01", b"+1", b"1 ", b" 1", b"1.0", b"-0", b""];
        assert_eq!(
            hex(&of(&not_ints)),
            "22000000070082303103822b3103823120038220310383312e3004822d30038001ff"
        );

        assert_eq!(
            hex(&of(&[b"name", b"age", b"age", b"41"])),
            "190000000400846e616d6505836167650483616765042901ff"
        );
    }

    /// The same check for the length boundaries, where the whole middle of the
    /// blob is one repeated byte and only the ends carry any information.
    #[test]
    fn a_long_element_is_framed_the_way_redis_frames_it() {
        for (len, total, head, tail) in [
            (
                63usize,
                72usize,
                "480000000100bf7878787878",
                "78787878787840ff",
            ),
            (64, 74, "4a0000000100e04078787878", "78787878787842ff"),
            (4095, 4106, "0a1000000100efff78787878", "78787878782081ff"),
            (4096, 4110, "0e1000000100f00010000078", "78787878782085ff"),
        ] {
            let lp = of(&[&vec![b'x'; len]]);
            let h = hex(&lp);
            assert_eq!(lp.byte_len(), total, "a {len} byte element");
            assert_eq!(&h[..head.len()], head, "a {len} byte element");
            assert_eq!(&h[h.len() - tail.len()..], tail, "a {len} byte element");
        }

        let lp = of(&[&vec![b'y'; 5000], b"after"]);
        let h = hex(&lp);
        assert_eq!(lp.byte_len(), 5021);
        assert_eq!(&h[..24], "9d1300000200f08813000079");
        assert_eq!(&h[h.len() - 16..], "85616674657206ff");
    }

    /// The band it exists for. A hundred and twenty eight elements, every one of
    /// them findable, and the whole thing inside a few cache lines.
    #[test]
    fn a_full_inline_band_still_reads_correctly() {
        let members: Vec<Vec<u8>> = (0..128u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut lp = Listpack::new();
        for m in &members {
            lp.push(m);
        }
        assert_eq!(lp.len(), 128);
        for (i, m) in members.iter().enumerate() {
            assert_eq!(lp.find(m, 1), Some(i));
        }
        assert!(lp.byte_len() < 1024, "{} bytes", lp.byte_len());
    }
}
