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
//! cost 1 to 2 ns an element, which is L6's number. A probe in our element table
//! costs 8 ns and not 70, so the gap it was being compared against is not there,
//! and at eight members the blob is ahead on both of them: 8.2 ns against 7.0 to
//! find a member that is present, 4.2 against 8.5 to find one that is not, 0.4 ns
//! an element against 1.3 to walk the whole thing, and 268 ns against 261 to
//! build it. At a hundred and twenty eight the table is six times faster to probe
//! and the gap only widens from there. `benches/listpack.rs` is where those come
//! from.
//!
//! The find numbers used to be much worse for the blob and have been re-measured
//! twice, once when the scan stopped decoding every element it walked past and
//! again when it stopped waiting for the header byte to work out where the next
//! element starts. Both are written up on `scan_for` below. Neither changes the
//! conclusion, because the conclusion never rested on them.
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
//!
//! # The element codec is shared
//!
//! [`crate::chunk`] holds the same entries in a run with a cursor at each end
//! rather than in a blob with a header, so it needs the encoding and not the
//! container. `entry_len`, `write_entry`, `decode` and `read_backlen`
//! are `pub(crate)` for that, and they are the only things a second holder of
//! these bytes needs. Two copies of the fourteen encodings is how a list and a
//! set end up disagreeing about what `SADD s 1` stored.

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

    /// How many bytes a client would see, without formatting anything.
    ///
    /// This is `HSTRLEN` and `STRLEN`, which both have to answer for a value
    /// stored as an integer, and neither of them should have to write the digits
    /// out to count them.
    #[must_use]
    #[inline]
    pub fn byte_len(&self) -> usize {
        match self {
            Entry::Int(n) => yo_common::num::i64_len(*n),
            Entry::Str(s) => s.len(),
        }
    }

    /// The element as bytes, allocating only for an integer.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
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

    /// The entry region on its own, without the header or the terminator.
    ///
    /// [`crate::chunk`] holds entries in exactly this encoding, so promoting a
    /// list out of the packed band is one copy of this slice rather than a walk
    /// that re-encodes every element.
    #[inline]
    #[must_use]
    pub(crate) fn entries(&self) -> &[u8] {
        &self.bytes[HDR..self.bytes.len() - 1]
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
    /// would have avoided it. From whichever end is nearer, though, because a
    /// list in this band holds eight kilobytes and that is four hundred odd
    /// entries rather than a hundred and twenty eight, and `LINDEX key -1` on
    /// one of those should not read all of it.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Entry<'_>> {
        let at = self.offset_of(index)?;
        decode(&self.bytes[at..self.bytes.len() - 1]).map(|(e, _)| e)
    }

    /// A forward walk that starts at `index` rather than at the front.
    ///
    /// `LRANGE key 300 320` on a packed list would otherwise decode three
    /// hundred entries and throw them away, which is what a `skip` on the walk
    /// does.
    pub fn iter_from(&self, index: usize) -> Iter<'_> {
        Iter {
            bytes: &self.bytes,
            at: self
                .offset_of(index)
                .unwrap_or(self.bytes.len().saturating_sub(1)),
        }
    }

    /// Every element, back to front.
    ///
    /// The trailing length on each entry is what makes this cost the same per
    /// element as the forward walk. `LPOS` with a negative rank counts matches
    /// from the tail and stops when it has enough, so walking forward and
    /// keeping the answers would be the wrong shape as well as the wrong cost.
    pub fn iter_back(&self) -> RevIter<'_> {
        let entries = self.entries();
        RevIter {
            bytes: entries,
            at: entries.len(),
        }
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
        self.find_parsed(needle, parse_i64(needle), step)
    }

    /// The same walk with the needle already parsed.
    ///
    /// Set algebra asks one member of one set about every other set, so the
    /// parse would otherwise happen once per question about the same bytes. It
    /// is also the only form that can answer about a member which was never
    /// text: an intset holds the number and the digits do not exist anywhere
    /// until somebody writes them.
    #[must_use]
    pub fn find_parsed(&self, needle: &[u8], as_int: Option<i64>, step: usize) -> Option<usize> {
        scan_for(self.entries(), needle, as_int, step)
    }

    /// Every place an element is, front to back, handed over as they are found.
    ///
    /// `limit` is how many elements may be looked at with 0 meaning all of them,
    /// `hit` says whether to carry on, and what comes back is how many elements
    /// were looked at. The walk itself is `scan_each` below.
    pub fn find_each(
        &self,
        needle: &[u8],
        as_int: Option<i64>,
        limit: usize,
        hit: &mut dyn FnMut(usize) -> bool,
    ) -> usize {
        scan_each(self.entries(), needle, as_int, limit, hit)
    }

    /// The same from the back, with indexes counted from the last element.
    pub fn find_each_back(
        &self,
        needle: &[u8],
        as_int: Option<i64>,
        limit: usize,
        hit: &mut dyn FnMut(usize) -> bool,
    ) -> usize {
        scan_each_back(self.entries(), needle, as_int, limit, hit)
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
    ///
    /// Forward from the header or backward from the terminator, whichever is
    /// the shorter walk. Going backward reads the length each entry carries
    /// behind it, which is the same field [`Listpack::get_back`] reads and the
    /// reason that field is there. A list in the packed band is eight kilobytes
    /// and four hundred odd entries, not a hundred and twenty eight like the
    /// other packed bands, so the half that this saves is worth having.
    ///
    /// [`Listpack::len`] is a header read at every size this crate builds: the
    /// count field only stops being the count past sixty five thousand entries
    /// and no band here comes close.
    fn offset_of(&self, index: usize) -> Option<usize> {
        let n = self.len();
        if index >= n {
            return None;
        }
        if index * 2 <= n {
            let mut at = HDR;
            for _ in 0..index {
                at += self.entry_bytes(at);
            }
            return Some(at);
        }
        let mut end = self.bytes.len() - 1;
        for _ in index..n {
            let len = read_backlen(&self.bytes[..end])?;
            end = end.checked_sub(len + backlen_len(len))?;
        }
        Some(end)
    }

    /// How many bytes the entry at `at` occupies, back length included.
    fn entry_bytes(&self, at: usize) -> usize {
        let (_, len) = decode(&self.bytes[at..self.bytes.len() - 1]).expect("our own blob decodes");
        len + backlen_len(len)
    }

    /// The one edit primitive: drop some bytes, put some back, fix the header.
    ///
    /// Everything that changes the blob goes through here, so there is one place
    /// that can leave the length or the count wrong.
    ///
    /// It writes the new entry into the blob rather than building it in a `Vec`
    /// and handing that to `Vec::splice`. The `Vec` was a malloc and a free on
    /// every `RPUSH`, `LPUSH`, `HSET`, `SADD` and `ZADD` that landed in the
    /// packed band, which is most of them, for a buffer of a few dozen bytes
    /// that never outlived the call. Making the hole first and then writing the
    /// head, the payload and the back length straight into it costs one
    /// `copy_within` of the tail, which `Vec::splice` was doing as well as the
    /// allocation.
    fn splice(&mut self, at: usize, remove: usize, insert: Option<&[u8]>, delta: i32) {
        let mut buf = [0u8; 16];
        // Both halves of the new entry, measured before anything moves. An
        // integer entry is entirely in its head and has no payload, which is
        // what `encode` says with its second answer.
        let (head, body) = match insert {
            Some(v) => {
                let (head, payload) = encode(v, &mut buf);
                (head, if payload { v } else { &[][..] })
            }
            None => (&[][..], &[][..]),
        };
        let entry = head.len() + body.len();
        // A pure removal puts nothing back, so it has no back length either.
        let add = if entry == 0 {
            0
        } else {
            entry + backlen_len(entry)
        };

        // Size the hole before writing into it. Growing moves the tail rightward
        // and shrinking moves it leftward, and `copy_within` is a `memmove` in
        // both directions, so an overlap reads what it should either way.
        let old = self.bytes.len();
        match add.cmp(&remove) {
            std::cmp::Ordering::Greater => {
                self.bytes.resize(old + (add - remove), 0);
                self.bytes.copy_within(at + remove..old, at + add);
            }
            std::cmp::Ordering::Less => {
                self.bytes.copy_within(at + remove..old, at + add);
                self.bytes.truncate(old - (remove - add));
            }
            std::cmp::Ordering::Equal => {}
        }
        if add > 0 {
            let hole = &mut self.bytes[at..at + add];
            hole[..head.len()].copy_from_slice(head);
            hole[head.len()..entry].copy_from_slice(body);
            write_backlen_into(&mut hole[entry..], entry);
        }

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

/// A backward walk.
#[derive(Debug)]
pub struct RevIter<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Iterator for RevIter<'a> {
    type Item = Entry<'a>;

    #[inline]
    fn next(&mut self) -> Option<Entry<'a>> {
        if self.at == 0 {
            return None;
        }
        let len = read_backlen(&self.bytes[..self.at])?;
        let start = self.at.checked_sub(len + backlen_len(len))?;
        let (entry, _) = decode(&self.bytes[start..self.at])?;
        self.at = start;
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

/// How many bytes an entry for `v` takes, its back length included.
///
/// The chunk has to know before it writes, because a chunk that runs out of
/// room halfway through an entry has no way to put itself back.
#[inline]
pub(crate) fn entry_len(v: &[u8]) -> usize {
    let mut buf = [0u8; 16];
    let (head, payload) = encode(v, &mut buf);
    let len = head.len() + if payload { v.len() } else { 0 };
    len + backlen_len(len)
}

/// Write one entry into `dst`, and say how many bytes it took.
///
/// `dst` must be at least [`entry_len`] long, which every caller knows because
/// it asked first.
#[inline]
pub(crate) fn write_entry(dst: &mut [u8], v: &[u8]) -> usize {
    let mut buf = [0u8; 16];
    let (head, payload) = encode(v, &mut buf);
    dst[..head.len()].copy_from_slice(head);
    let mut at = head.len();
    if payload {
        dst[at..at + v.len()].copy_from_slice(v);
        at += v.len();
    }
    at + write_backlen_into(&mut dst[at..], at)
}

/// Read one entry, and say how many bytes it took before its back length.
///
/// Inlined on purpose. It hands back a fat enum and a length, and left out of
/// line that pair goes through memory once per element, which is most of what a
/// walk costs.
#[inline]
pub(crate) fn decode(b: &[u8]) -> Option<(Entry<'_>, usize)> {
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

/// Eight bytes of `s` starting at `at`, as a number.
///
/// The caller has already checked that they are there, so the `try_into` cannot
/// fail and the compiler knows it, which is what keeps this to one unaligned
/// load on every target this runs on.
#[inline(always)]
fn word(s: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(s[at..at + 8].try_into().expect("eight bytes"))
}

/// The needle of a scan, with everything that does not change worked out once.
///
/// This exists because of what the generated code looked like without it. The
/// comparison started as a length check, a first byte check and then `a == b`,
/// which is a call to `memcmp`, and on the workload that actually matters none
/// of the first two filter anything: list elements are overwhelmingly a fixed
/// shape with a varying tail, so a million of them called `element:00000000`
/// through `element:00999999` are all the same length and all start with the
/// same letter. Every element paid for the call.
///
/// Comparing whole words instead fixes that, and eight bytes from each end
/// covers any length up to sixteen exactly, because at that length the two
/// windows overlap and between them cover the whole value. Longer than sixteen
/// and the two words are a filter in front of the call rather than a
/// replacement for it, which is fine, since a value agreeing on both ends and
/// differing in the middle is rare enough to be worth a `memcmp` when it turns
/// up.
///
/// Working the two words out here rather than in the loop is not a
/// micro-optimisation, it is two loads an element. Left inline the compiler
/// reloads them from the needle every time round, because the calls further down
/// the body could have written to it as far as it knows, and it has no way to
/// prove otherwise.
struct Needle<'a> {
    bytes: &'a [u8],
    /// The length, which is the first thing every element is rejected on.
    len: usize,
    /// The first and last eight bytes, both zero and never looked at when the
    /// value is shorter than eight bytes.
    head: u64,
    tail: u64,
    /// What the value is as a number, if it is one. An element stored under an
    /// integer encoding can only match this.
    num: Option<i64>,
}

impl<'a> Needle<'a> {
    fn new(bytes: &'a [u8], num: Option<i64>) -> Needle<'a> {
        let len = bytes.len();
        let wide = len >= 8;
        Needle {
            bytes,
            len,
            head: if wide { word(bytes, 0) } else { 0 },
            tail: if wide { word(bytes, len - 8) } else { 0 },
            num,
        }
    }

    /// Whether a string payload of the same length is this value.
    ///
    /// `always` rather than `inline`, and it is worth saying why, because this
    /// is the difference between three and a half nanoseconds an element and one
    /// and a half. Left to its own judgement the compiler kept this out of line,
    /// so the scan below made a call per element, spilled around it, and paid
    /// more to set the arguments up than the comparison itself costs.
    ///
    /// Every length here is taken from `p` and not from the needle, which reads
    /// like a pointless difference and is not. The caller has already checked
    /// that they are equal, but nothing in the types says so, so a bound written
    /// in terms of the needle is a bound the compiler has to check against `p`
    /// all over again, and it emitted a second comparison and a panic landing pad
    /// on the hot path to do it. Written this way the check that lets the first
    /// word be read is the same check that says the value is long enough to have
    /// two words at all.
    #[inline(always)]
    fn is(&self, p: &[u8]) -> bool {
        debug_assert_eq!(p.len(), self.len, "the caller checks the length first");
        let n = p.len();
        if n < 8 {
            return p == self.bytes;
        }
        word(p, 0) == self.head
            && word(p, n - 8) == self.tail
            && (n <= 16 || p[8..n - 8] == self.bytes[8..n - 8])
    }
}

/// Walk a run of entries looking for `needle`, and say which one it was.
///
/// `b` is entries and nothing else, which is what a chunk holds and what a
/// listpack holds between its header and its terminator, so both callers get the
/// same walk. `step` is the hash trick: a field lookup over field, value, field,
/// value is a find with a step of two.
///
/// # Why this is not the obvious walk
///
/// The obvious walk is `self.iter().position(|e| e.is(needle, as_int))`, which
/// is what this was, and it costs about 6.7 nanoseconds an element. That is
/// twenty seven cycles to answer "are these bytes those bytes", and almost none
/// of it is the comparison. Every step ran the whole of [`decode`], which is a
/// fourteen way match with a bounds checked read per header byte, built an
/// [`Entry`] out of what it found, handed that back through the iterator, and
/// only then compared. On a million element `LINSERT` that is three and a third
/// milliseconds of work to reach an insert that takes two hundred and eighty
/// five nanoseconds.
///
/// So the walk below never builds an `Entry` and never reads a payload it is not
/// about to compare. The header alone says how long the entry is, the length
/// rejects most elements with one comparison, and what gets past that is
/// compared as two words rather than as a call. See [`Needle`] for that half of
/// it.
///
/// The integer arm is the rare one and it is deliberately left to `decode`. It
/// is four sign extensions of different widths, getting one of them subtly wrong
/// is exactly the kind of bug that hides for a year, and having a second copy of
/// them here to save a call on a path that is cold in every list workload is a
/// bad trade.
///
/// # What was left on the table by the first version of that
///
/// The walk above got to about 2.3 nanoseconds an element and stopped, and it
/// stopped there because of one instruction. Stepping to the next element is
/// `at += len + 2`, `len` comes out of the encoding byte that was just loaded,
/// and the loaded value is therefore in the way of working out where the next
/// load goes. Every element paid a load latency plus the arithmetic stacked on
/// it before the element after it could start, which is about eight cycles, and
/// nothing else in the loop mattered because everything else could run while that
/// chain was resolving. It is not a throughput problem and no amount of removing
/// instructions from the body would have touched it.
///
/// The way out is that on the path that matters the length is already known. An
/// element is only compared when its length equals the needle's, and the needle's
/// length has been in a register since before the loop started, so on that path
/// the step can be written in terms of the needle instead of in terms of the
/// byte that was just read. The next element's address is then worked out while
/// this one is still being compared, and the loop halves to about four cycles an
/// element. That is why the length test is a branch of its own below rather than
/// the first half of the comparison, which is the shape it had and reads more
/// naturally.
///
/// It only pays when the lengths do match, and on a list they either all match
/// or none of them do, which is the same property the two word comparison in
/// [`Needle`] leans on. A million element `LINSERT` went from 1.17 ms to 456 us
/// on it and the bare search from 2.37 ms to 904 us, both on an M4.
#[inline]
pub(crate) fn scan_for(b: &[u8], needle: &[u8], as_int: Option<i64>, step: usize) -> Option<usize> {
    let mut got = None;
    let needle = Needle::new(needle, as_int);
    // Four walks out of one body, and both const parameters earn their keep the
    // same way. A list finds by stepping over every element and a hash steps
    // over every other one, and carrying a counter for a step that is always one
    // costs four instructions and a branch on the hottest loop in `LINSERT`.
    // `LIMITED` is the same argument for `LPOS`'s `MAXLEN`, which every other
    // caller passes as no limit at all. As constants both fold away entirely.
    if step <= 1 {
        walk::<true, false, _>(b, &needle, 1, 0, &mut |at| {
            got = Some(at);
            false
        });
    } else {
        walk::<false, false, _>(b, &needle, step, 0, &mut |at| {
            got = Some(at);
            false
        });
    }
    got
}

/// Walk a run of entries handing back every `needle` in it, front to back.
///
/// This is [`scan_for`] without the stop on the first one, which is `LPOS` and
/// `LREM` rather than `LINSERT`. `hit` is given each index as it is found and
/// says whether to keep going, so a `COUNT` stops the walk where it is reached
/// rather than after reading the rest of the list. `limit` is how many elements
/// may be looked at, with 0 meaning no limit, which is `MAXLEN`.
///
/// What comes back is how many elements were looked at, which is what a caller
/// spanning several runs of entries needs in order to carry one `MAXLEN` budget
/// across all of them.
#[inline]
pub(crate) fn scan_each(
    b: &[u8],
    needle: &[u8],
    as_int: Option<i64>,
    limit: usize,
    mut hit: &mut dyn FnMut(usize) -> bool,
) -> usize {
    let needle = Needle::new(needle, as_int);
    if limit == 0 {
        walk::<true, false, _>(b, &needle, 1, 0, &mut hit)
    } else {
        walk::<true, true, _>(b, &needle, 1, limit, &mut hit)
    }
}

/// The same walk from the other end, with indexes counted from the back.
///
/// The index handed to `hit` is 0 for the last element, 1 for the one before
/// it and so on, because this does not know how many elements are in front of
/// the run it was given and the caller does. `LPOS` with a negative rank is the
/// only thing that wants this, and it wants it because a rank of -1 has to find
/// the last match without reading past it.
#[inline]
pub(crate) fn scan_each_back(
    b: &[u8],
    needle: &[u8],
    as_int: Option<i64>,
    limit: usize,
    mut hit: &mut dyn FnMut(usize) -> bool,
) -> usize {
    let needle = Needle::new(needle, as_int);
    if limit == 0 {
        walk_back::<false, _>(b, &needle, 0, &mut hit)
    } else {
        walk_back::<true, _>(b, &needle, limit, &mut hit)
    }
}

/// How long the entry at `at` is, split into its header and its payload, and
/// whether the payload is text.
///
/// Every encoding except the short string, which the walk below handles itself.
/// An integer encoding is entirely header, so its payload length is zero and it
/// can never match a string of any length.
///
/// Out of line on purpose, and it is not because this is rare in general, it is
/// because of what having it inline did to the loop it was in. Thirteen arms
/// need registers, and the register allocator paid for them by spilling the
/// needle's two comparison words to the stack and reloading them on every single
/// element, including the overwhelming majority that never reach this function
/// at all. A call on the encodings a list does not use is a good trade for two
/// loads on the ones it does.
#[inline(never)]
fn head_at(b: &[u8], at: usize) -> Option<(usize, usize, bool)> {
    let tag = *b.get(at)?;
    Some(match tag {
        0x00..=0x7F => (1, 0, false),
        0x80..=0xBF => (1, (tag & 0x3F) as usize, true),
        0xC0..=0xDF => (2, 0, false),
        0xE0..=0xEF => (
            2,
            (usize::from(tag & 0x0F) << 8) | usize::from(*b.get(at + 1)?),
            true,
        ),
        0xF0 => (
            5,
            u32::from_le_bytes([
                *b.get(at + 1)?,
                *b.get(at + 2)?,
                *b.get(at + 3)?,
                *b.get(at + 4)?,
            ]) as usize,
            true,
        ),
        0xF1 => (3, 0, false),
        0xF2 => (4, 0, false),
        0xF3 => (5, 0, false),
        0xF4 => (9, 0, false),
        // 0xF5 to 0xFE are unused by Redis and 0xFF is the terminator, which is
        // not in this run. Either way there is nothing after it that can be read
        // as an entry.
        _ => return None,
    })
}

/// The scan itself, with `EVERY` saying whether `step` is worth carrying and
/// `LIMITED` whether `limit` is.
///
/// `hit` is a generic and it was a `&mut dyn` first, on the reasoning that it is
/// only reached on a match and so is called a handful of times against a million
/// iterations of the body around it. That reasoning is right about a long list
/// and wrong about a short one. A blob holds at most a hundred and twenty eight
/// elements and the thing it does all day is find one that is there, so the call
/// is not one in a million, it is one in four, and it cost 12 percent on the
/// eight member row in `benches/listpack.rs`. As a generic the sink for a find
/// inlines back to a store and a break, which is what the loop had before it took
/// a sink at all. It is two copies of this function and not more, because every
/// caller that wants every match already goes through a `&mut dyn` of its own.
///
/// What comes back is how many elements were looked at, which for the single
/// answer case is not interesting and folds away with everything else.
#[inline]
fn walk<const EVERY: bool, const LIMITED: bool, F: FnMut(usize) -> bool>(
    b: &[u8],
    needle: &Needle<'_>,
    step: usize,
    limit: usize,
    hit: &mut F,
) -> usize {
    let want = needle.len;
    let mut at = 0usize;
    let mut idx = 0usize;
    // Counted down rather than `idx % step`, because `step` is a runtime value
    // and the remainder compiles to a real division on every element. That is
    // twenty cycles to answer a question about a two element cycle, and it cost
    // more than the comparison it was guarding.
    let mut until = 0usize;
    while at < b.len() {
        if LIMITED && idx == limit {
            break;
        }
        let tag = b[at];
        // The one encoding a list is actually made of, given a path with nothing
        // in it. A string of sixty three bytes or less is a one byte header, and
        // its total is at most sixty four so its back length is one byte too,
        // which means stepping to the next element is an add of a number that
        // came straight out of the tag. No second read, no table, and one branch
        // instead of the five the general match below needs.
        //
        // The general walk is not slow because of how many instructions it runs.
        // It is slow because of how many of its branches are taken: a core
        // retires about one taken branch a cycle, and a five way tag match that
        // ends in a four way back length match spends more time being fetched
        // than being executed. Straightening the common case out is worth more
        // than anything done to the comparison inside it.
        if EVERY && tag & 0xC0 == 0x80 {
            let len = (tag & 0x3F) as usize;
            if len != want {
                at += len + 2;
                idx += 1;
                continue;
            }
            let Some(p) = b.get(at + 1..at + 1 + want) else {
                break;
            };
            let matched = needle.is(p);
            // `want` rather than `len`, which are the same number here and are
            // not the same instruction: one of them is in the way of the next
            // load and the other has been in a register all along. It halves the
            // loop. `scan_for` above has the long version.
            at += want + 2;
            idx += 1;
            if matched && !hit(idx - 1) {
                break;
            }
            continue;
        }
        let Some((hdr, len, text)) = head_at(b, at) else {
            break;
        };
        let total = hdr + len;
        let mut matched = false;
        if EVERY || until == 0 {
            matched = if text {
                let Some(p) = b.get(at + hdr..at + total) else {
                    break;
                };
                len == want && needle.is(p)
            } else {
                // `at` is inside `b`, which is the loop condition, so this
                // slice is always there.
                needle
                    .num
                    .is_some_and(|v| matches!(decode(&b[at..]), Some((Entry::Int(n), _)) if n == v))
            };
            if !EVERY {
                until = step;
            }
        }
        if !EVERY {
            until -= 1;
        }
        at += total + backlen_len(total);
        idx += 1;
        if matched && !hit(idx - 1) {
            break;
        }
    }
    idx
}

/// The same walk from the back, which only a list ever asks for.
///
/// No `EVERY`, because the caller that steps is a hash and a hash has no back to
/// walk from. The step from one entry to the one in front of it is the back
/// length that ends where this entry starts, and reading it is a byte and a
/// branch in the case that covers everything up to a hundred and twenty seven
/// bytes, which is every element the fast path in [`walk`] handles and then
/// some. Longer than that and it falls back to [`read_backlen`], which is the
/// leftward walk.
#[inline]
fn walk_back<const LIMITED: bool, F: FnMut(usize) -> bool>(
    b: &[u8],
    needle: &Needle<'_>,
    limit: usize,
    hit: &mut F,
) -> usize {
    let want = needle.len;
    let mut at = b.len();
    let mut idx = 0usize;
    while at > 0 {
        if LIMITED && idx == limit {
            break;
        }
        // A back length of one byte has its top bit clear and holds the whole
        // value, which is what `write_backlen_into` writes and what makes this
        // one load rather than a loop.
        let last = b[at - 1];
        let (total, blen) = if last < 128 {
            (usize::from(last), 1)
        } else {
            let Some(t) = read_backlen(&b[..at]) else {
                break;
            };
            (t, backlen_len(t))
        };
        let Some(start) = at.checked_sub(total + blen) else {
            break;
        };
        let tag = b[start];
        let matched = if tag & 0xC0 == 0x80 {
            let len = usize::from(tag & 0x3F);
            match b.get(start + 1..start + 1 + len) {
                Some(p) => len == want && needle.is(p),
                None => break,
            }
        } else {
            match head_at(b, start) {
                Some((hdr, len, true)) => match b.get(start + hdr..start + hdr + len) {
                    Some(p) => len == want && needle.is(p),
                    None => break,
                },
                Some((_, _, false)) => needle.num.is_some_and(
                    |v| matches!(decode(&b[start..]), Some((Entry::Int(n), _)) if n == v),
                ),
                None => break,
            }
        };
        at = start;
        idx += 1;
        if matched && !hit(idx - 1) {
            break;
        }
    }
    idx
}

/// How many bytes the back length of an entry of `len` bytes takes.
///
/// Seven bits a byte, so the boundaries are `2^7 - 1`, `2^14 - 1` and so on, and
/// they are Redis's `lpEncodeBacklenBytes` boundaries exactly. They are checked
/// against the real ones in the tests rather than taken on trust, because a
/// listpack whose back lengths are a byte out is one Redis walks off the end of.
#[inline]
pub(crate) const fn backlen_len(len: usize) -> usize {
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

/// Write the back length for an entry of `len` bytes, and say how long it was.
///
/// The first byte holds the high seven bits and every later byte has its top bit
/// set, which is what lets it be read from the right hand end leftward.
#[inline]
fn write_backlen_into(dst: &mut [u8], len: usize) -> usize {
    let n = backlen_len(len);
    // High seven bits first, then seven at a time downward, and every byte
    // after the first carries the continuation bit that stops the leftward
    // walk from running past the front of the entry.
    for (i, b) in dst[..n].iter_mut().enumerate() {
        let shift = 7 * (n - 1 - i);
        *b = ((len >> shift) & 127) as u8 | if i == 0 { 0 } else { 128 };
    }
    n
}

/// Read a back length that ends at the last byte of `upto`.
///
/// Walks left while the top bit is set, seven bits at a time, which is the
/// mirror image of how it was written.
pub(crate) fn read_backlen(upto: &[u8]) -> Option<usize> {
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

    /// Every edit used to build the new entry in a `Vec` and throw it away, so
    /// a blob with room in it still paid a malloc and a free per write. These
    /// three shapes are the whole of `splice`: an edit that grows the blob, one
    /// that shrinks it, and one that leaves it the same size. None of them may
    /// touch the allocator once the blob's own buffer is big enough.
    #[test]
    fn editing_a_blob_that_has_room_does_not_allocate() {
        let mut lp = Listpack::new();
        for i in 0..200 {
            lp.push(format!("member:{i:04}").as_bytes());
        }
        // Down to a hundred and back up, so the buffer is at its high water
        // mark and nothing below measures growth.
        lp.delete(100, 100);
        for i in 0..100 {
            lp.push(format!("member:{i:04}").as_bytes());
        }

        // Built up here rather than inside the count, because `format!` is an
        // allocation of the test's own and would drown out what is measured.
        let names: Vec<(Vec<u8>, Vec<u8>)> = (0..100)
            .map(|i| {
                (
                    format!("other:{i:05}").into_bytes(),
                    format!("member:{i:04}").into_bytes(),
                )
            })
            .collect();

        let (_, allocs) = crate::tally::counted(|| {
            for (i, (other, original)) in names.iter().enumerate() {
                // Same length, so the blob does not change size at all.
                lp.replace(i, other);
                // Shorter, then back to the original length.
                lp.replace(i, b"x");
                lp.replace(i, original);
            }
        });
        assert_eq!(allocs, 0, "editing allocated {allocs} times");
        assert_eq!(lp.len(), 200);
        assert_eq!(lp.get(0), Some(Entry::Str(b"member:0000")));
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
            let mut buf = [0u8; 5];
            let n = write_backlen_into(&mut buf, len);
            let out = &buf[..n];
            assert_eq!(out.len(), backlen_len(len), "length {len}");
            assert_eq!(read_backlen(out), Some(len), "length {len}");
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

    /// One of every encoding, at every length that changes which path through
    /// the scan an element takes.
    ///
    /// The lengths are the ones that matter to the comparison rather than
    /// arbitrary: nothing, under a word, exactly a word, between one and two
    /// words where the two window compare covers the whole value, exactly two,
    /// and past two where it stops being a whole answer and becomes a filter in
    /// front of `memcmp`. Sixty three and sixty four are the last length with a
    /// one byte header and the first with two, which is the boundary the short
    /// path is drawn on, and a hundred and twenty seven and a hundred and twenty
    /// eight are where the back length grows a second byte, which is the
    /// boundary the backward walk is drawn on.
    fn every_encoding() -> Vec<Vec<u8>> {
        let mut members: Vec<Vec<u8>> = Vec::new();
        // One of each integer encoding, including the boundaries where Redis
        // steps up to a wider one and both signs of each.
        for n in [
            0i64,
            127,
            -1,
            4095,
            -4096,
            32767,
            -32768,
            8_388_607,
            -8_388_608,
            2_147_483_647,
            -2_147_483_648,
            i64::MAX,
            i64::MIN,
        ] {
            members.push(n.to_string().into_bytes());
        }
        for len in [
            0usize, 1, 7, 8, 9, 15, 16, 17, 31, 63, 64, 100, 125, 126, 127, 128, 200,
        ] {
            let mut v = vec![b'a'; len];
            // A varying tail, so that two different lengths are not two
            // prefixes of each other and the length check is doing work rather
            // than being the only thing that separates them.
            if len > 0 {
                v[len - 1] = b'0' + (len % 10) as u8;
            }
            members.push(v);
        }
        members
    }

    /// The scan has a short path for the one encoding a list is made of and a
    /// general one for the other thirteen, and the danger with two paths is that
    /// they disagree about some element neither author had in mind. So this
    /// builds a blob holding every encoding, at every length that changes which
    /// path an element takes, and asks for each of them in turn.
    ///
    /// The lengths are the ones that matter to the comparison rather than
    /// arbitrary: nothing, under a word, exactly a word, between one and two
    /// words where the two window compare covers the whole value, exactly two,
    /// and past two where it stops being a whole answer and becomes a filter in
    /// front of `memcmp`. Sixty three and sixty four are the last length with a
    /// one byte header and the first with two, which is the boundary the short
    /// path is drawn on.
    #[test]
    fn both_paths_through_the_scan_agree_about_every_encoding() {
        let members = every_encoding();
        let lp = of(&members.iter().map(Vec::as_slice).collect::<Vec<_>>());
        assert_eq!(lp.len(), members.len());
        for (at, m) in members.iter().enumerate() {
            assert_eq!(lp.find(m, 1), Some(at), "member {at} went missing");
        }
        // And a handful that are not there, each one a near miss of something
        // that is: a different length, the same length with a different first
        // byte, and the same length with a different last byte.
        for miss in [
            b"aaaaaaaaaaaa".as_slice(),
            b"baaaaaa7".as_slice(),
            b"aaaaaaa9".as_slice(),
            b"128".as_slice(),
            b"-2".as_slice(),
        ] {
            assert_eq!(lp.find(miss, 1), None, "{miss:?} is not in here");
        }
    }

    /// The same blob under a step of two, which is the other instantiation of
    /// the walk and the one a hash uses. Nothing at an odd position may be
    /// found, whatever it is encoded as.
    #[test]
    fn a_stepped_scan_agrees_with_itself_about_every_encoding() {
        let members: Vec<Vec<u8>> = (0..40i32)
            .map(|i| {
                if i % 3 == 0 {
                    (i64::from(i) * 1000 - 20_000).to_string().into_bytes()
                } else {
                    format!("field:{i:0width$}", width = (i % 20) as usize).into_bytes()
                }
            })
            .collect();
        let lp = of(&members.iter().map(Vec::as_slice).collect::<Vec<_>>());
        for (at, m) in members.iter().enumerate() {
            let want = if at % 2 == 0 {
                // Every member here is distinct, so an even one is found where
                // it is and an odd one is not found at all.
                Some(at)
            } else {
                None
            };
            assert_eq!(lp.find(m, 2), want, "member {at} under a step of two");
        }
    }

    /// The backward walk finds its way from one entry to the one in front of it
    /// through the back length rather than through a header, so it is a third
    /// path over the same bytes and it has to agree with the other two about all
    /// of them. Every member is asked for from the back and has to come back at
    /// the position the forward walk gives it.
    #[test]
    fn the_backward_scan_agrees_with_the_forward_one_about_every_encoding() {
        let members = every_encoding();
        let lp = of(&members.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let n = members.len();
        for (at, m) in members.iter().enumerate() {
            let mut got = Vec::new();
            lp.find_each_back(m, parse_i64(m), 0, &mut |back| {
                got.push(n - back - 1);
                true
            });
            assert_eq!(got, vec![at], "member {at} from the back");
        }
        for miss in [
            b"aaaaaaaaaaaa".as_slice(),
            b"baaaaaa7".as_slice(),
            b"aaaaaaa9".as_slice(),
            b"128".as_slice(),
            b"-2".as_slice(),
        ] {
            let mut got = 0usize;
            lp.find_each_back(miss, parse_i64(miss), 0, &mut |_| {
                got += 1;
                true
            });
            assert_eq!(got, 0, "{miss:?} is not in here");
        }
    }

    /// Both walks over a blob where the same value is in it several times, which
    /// is what `LPOS` and `LREM` are actually for and what the single answer
    /// scan never exercises. The stop and the budget are checked here too, since
    /// they are the two things the walk carries that a find does not.
    #[test]
    fn a_walk_over_every_match_gives_them_all_in_order_from_either_end() {
        let members: Vec<Vec<u8>> = (0..30)
            .map(|i| {
                if i % 4 == 0 {
                    b"x".to_vec()
                } else {
                    format!("element:{i:08}").into_bytes()
                }
            })
            .collect();
        let lp = of(&members.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let want: Vec<usize> = (0..30).filter(|i| i % 4 == 0).collect();

        let mut got = Vec::new();
        let looked = lp.find_each(b"x", None, 0, &mut |at| {
            got.push(at);
            true
        });
        assert_eq!(got, want);
        assert_eq!(looked, 30, "no budget means the whole thing is read");

        let mut got = Vec::new();
        lp.find_each_back(b"x", None, 0, &mut |back| {
            got.push(29 - back);
            true
        });
        got.reverse();
        assert_eq!(got, want, "the same matches, found the other way round");

        // A stop after two, which must not read the rest.
        let mut got = Vec::new();
        let looked = lp.find_each(b"x", None, 0, &mut |at| {
            got.push(at);
            got.len() < 2
        });
        assert_eq!(got, vec![0, 4]);
        assert_eq!(looked, 5, "the walk stopped where the second match was");

        // And a budget, which is `MAXLEN`: ten elements looked at reaches the
        // matches at 0, 4 and 8 and nothing after them.
        let mut got = Vec::new();
        let looked = lp.find_each(b"x", None, 10, &mut |at| {
            got.push(at);
            true
        });
        assert_eq!(got, vec![0, 4, 8]);
        assert_eq!(looked, 10);

        let mut got = Vec::new();
        let looked = lp.find_each_back(b"x", None, 10, &mut |back| {
            got.push(29 - back);
            true
        });
        assert_eq!(got, vec![28, 24, 20], "ten from the back is 20 up");
        assert_eq!(looked, 10);
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
