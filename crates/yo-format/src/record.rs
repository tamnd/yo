//! The record, which is the unit of everything: a commit, a replay step, and a
//! thing compaction either copies or drops.
//!
//! `06` section 2.1 for the layout, `06` section 3 for the ordering rule that
//! makes the layout safe.
//!
//! Two details that the table in the specification does not make obvious, and
//! that everything downstream depends on.
//!
//! **`len` is exact and the stride is aligned.** Records are eight byte aligned,
//! but the alignment is padding between records rather than padding inside one.
//! If `len` were rounded up, a value's length would be `len` minus the header
//! minus the key minus somewhere between zero and seven, and nobody could say
//! which. So `len` is the exact byte count and the next record starts at
//! [`RecordRef::stride`] bytes later. A reader that walks by `len` instead of by
//! `stride` desynchronises on the first odd sized value, which is why there is a
//! test for exactly that.
//!
//! **The trailer, and why it is not optional.** `07` section 4 says a per
//! record checksum lives in the record's own trailer for records marked
//! durable, and the field table in `06` does not list one. This module
//! reconciles them: [`record_flags::CHECKSUMMED`] is the mark, and when it is
//! set the last four bytes of the record are a CRC32C over everything before
//! them, `len` included.
//!
//! It started out as a real choice, on the reasoning that in `none` mode nobody
//! reads the record back off a disk and four bytes on a sixty byte record is
//! real money. That reasoning is wrong, and the way it is wrong is worth
//! keeping written down, because it is the sort of thing that reads as
//! reasonable right up until a fuzzer finds it.
//!
//! The flag lives in the record. So a single bit flip in the flags byte turns
//! the check off, and with it off there is nothing left to notice that the bit
//! flipped. Worse than undetected: clearing the bit also moves the trailer
//! boundary, so those four checksum bytes become four bytes of value, and a
//! reader hands back a value four bytes longer than the one that was stored and
//! is confident about it. A self describing checksum cannot describe its own
//! absence.
//!
//! So [`RecordRef::parse`] refuses a record with the bit clear rather than
//! believing it, and [`RecordHeader::fill`] always sets it. The flag stays in
//! the layout so a later version can define what an unchecksummed record means
//! with a second bit that is itself covered by something.

use crate::{align_up, get_u8, get_u16, get_u32, get_u64, put_u8, put_u16, put_u32, put_u64};
use yo_common::{Code, Error, Result, crc32c};

/// The header without a TTL: `len`, `kind`, `flags`, `klen`, `prev`.
pub const HEADER_LEN: usize = 16;

/// The header with a TTL, which adds eight bytes at offset 16.
pub const HEADER_LEN_TTL: usize = 24;

/// The trailer, when [`record_flags::CHECKSUMMED`] is set.
pub const TRAILER_LEN: usize = 4;

/// The largest key, because `klen` is a `u16`.
///
/// Redis allows 512 MB keys and we do not, deliberately. A key is a thing you
/// look up by, it lives in the index bucket's neighbourhood, and 64 KiB is
/// already three orders of magnitude past any key anyone has a reason to use.
/// A caller that wants a 512 MB key wants a value.
pub const MAX_KEY_LEN: usize = u16::MAX as usize;

/// The `flags` byte at offset 5.
pub mod record_flags {
    /// The value is not here; the index entry points at the cold tier.
    pub const TIERED: u8 = 1 << 0;
    /// The value is compressed.
    pub const COMPRESSED: u8 = 1 << 1;
    /// An eight byte `ttl_ms` sits at offset 16, before the key.
    pub const HAS_TTL: u8 = 1 << 2;
    /// The collection this record belongs to has a shape tag in the catalogue.
    pub const SHAPE_TAGGED: u8 = 1 << 3;
    /// The last four bytes are a CRC32C over everything before them.
    ///
    /// Always set on anything this version writes, and a record with it clear
    /// is rejected rather than read. See the note at the top of this module:
    /// a checksum whose own presence is announced by an unprotected bit is not
    /// a checksum.
    pub const CHECKSUMMED: u8 = 1 << 4;
}

/// What a record holds. `06` section 2.1.
///
/// This is deliberately not the type a reader gets back. A version one reader
/// must skip a `kind` it has never heard of rather than refuse the file (`07`
/// section 9), so [`RecordRef`] keeps the raw byte and offers this as a
/// question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordKind {
    /// A plain string value.
    String = 0,
    /// One chunk of a collection, or of a value too large for a page.
    CollectionChunk = 1,
    /// A document, whose value is the YOJB value byte for byte. See
    /// [`crate::document`].
    Document = 2,
    /// A vector. See [`crate::vector`].
    Vector = 3,
    /// A graph node.
    GraphNode = 4,
    /// A graph adjacency run.
    GraphAdj = 5,
    /// A checkpoint marker.
    Checkpoint = 6,
    /// A deletion.
    Tombstone = 7,
    /// An index change that replay applies directly instead of reinserting.
    IndexDelta = 8,
}

impl RecordKind {
    /// Every kind this version knows, in order.
    pub const ALL: [RecordKind; 9] = [
        RecordKind::String,
        RecordKind::CollectionChunk,
        RecordKind::Document,
        RecordKind::Vector,
        RecordKind::GraphNode,
        RecordKind::GraphAdj,
        RecordKind::Checkpoint,
        RecordKind::Tombstone,
        RecordKind::IndexDelta,
    ];

    /// The kind for a byte, or `None` if this version has not heard of it.
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<RecordKind> {
        match b {
            0 => Some(RecordKind::String),
            1 => Some(RecordKind::CollectionChunk),
            2 => Some(RecordKind::Document),
            3 => Some(RecordKind::Vector),
            4 => Some(RecordKind::GraphNode),
            5 => Some(RecordKind::GraphAdj),
            6 => Some(RecordKind::Checkpoint),
            7 => Some(RecordKind::Tombstone),
            8 => Some(RecordKind::IndexDelta),
            _ => None,
        }
    }

    /// The byte.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Whether a record of this kind carries a key.
    ///
    /// Chunks do not: they are addressed by the header record that lists them,
    /// and repeating the key in every chunk of a large collection would be the
    /// dominant cost of storing it.
    #[must_use]
    pub const fn carries_a_key(self) -> bool {
        !matches!(self, RecordKind::CollectionChunk)
    }
}

/// The bytes needed for a record with this shape.
///
/// # Errors
///
/// [`Code::Invalid`] if the key is longer than [`MAX_KEY_LEN`], or if the total
/// would not fit in the `u32` that has to hold it.
pub fn total_len(flags: u8, klen: usize, vlen: usize) -> Result<usize> {
    if klen > MAX_KEY_LEN {
        return Err(
            Error::new(Code::Invalid, "the key is longer than 65535 bytes")
                .with_detail(format!("klen={klen}")),
        );
    }
    let n = header_len(flags) + klen + vlen + trailer_len(flags);
    if n > u32::MAX as usize {
        return Err(Error::new(
            Code::Invalid,
            "the record does not fit in a u32",
        ));
    }
    Ok(n)
}

/// The header length implied by `flags`.
#[inline]
#[must_use]
pub const fn header_len(flags: u8) -> usize {
    if flags & record_flags::HAS_TTL != 0 {
        HEADER_LEN_TTL
    } else {
        HEADER_LEN
    }
}

/// The trailer length implied by `flags`.
///
/// Always [`TRAILER_LEN`] for anything this version will accept. The flag is
/// still read, because a record with it clear has to be rejected rather than
/// reinterpreted, and [`RecordRef::parse`] is where that happens. See the note
/// on [`record_flags::CHECKSUMMED`] for why the flag cannot be allowed to mean
/// what it says.
#[inline]
#[must_use]
pub const fn trailer_len(flags: u8) -> usize {
    if flags & record_flags::CHECKSUMMED != 0 {
        TRAILER_LEN
    } else {
        0
    }
}

/// A record about to be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader {
    /// What the value is. A raw byte rather than a [`RecordKind`] so that a
    /// future version can be written by a build that predates it, which is not
    /// as strange as it sounds: compaction copies records it does not interpret.
    pub kind: u8,
    /// See [`record_flags`].
    pub flags: u8,
    /// The previous address in this key's chain, or 0.
    pub prev: u64,
    /// Unix milliseconds, meaningful only when [`record_flags::HAS_TTL`] is set.
    pub ttl_ms: u64,
}

impl RecordHeader {
    /// A header for a live value with no TTL, checksummed.
    #[must_use]
    pub const fn new(kind: RecordKind) -> RecordHeader {
        RecordHeader {
            kind: kind.as_u8(),
            flags: record_flags::CHECKSUMMED,
            prev: 0,
            ttl_ms: 0,
        }
    }

    /// Sets the TTL and the flag that says it is there.
    #[must_use]
    pub const fn with_ttl(mut self, unix_ms: u64) -> RecordHeader {
        self.flags |= record_flags::HAS_TTL;
        self.ttl_ms = unix_ms;
        self
    }

    /// Chains this record to the previous version of the same key.
    #[must_use]
    pub const fn after(mut self, prev: u64) -> RecordHeader {
        self.prev = prev;
        self
    }

    /// Writes everything except `len`, and returns the exact record length.
    ///
    /// The first four bytes are left untouched, which is the point. `06`
    /// section 3 requires `len` to be stored last with a release store, so that
    /// a reader either sees a whole record or sees a zero and stops. Writing it
    /// here would put a length in front of a body that is not there yet, and a
    /// crash in between produces a record that claims bytes it does not have.
    ///
    /// The caller finishes with [`seal_len`], which is the release store's
    /// payload. Ordering is the caller's job because this crate does not know
    /// what memory the buffer is in.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for an oversized key, [`Code::Full`] if `buf` is too
    /// small for the record.
    pub fn fill(&self, buf: &mut [u8], key: &[u8], value: &[u8]) -> Result<usize> {
        // Set rather than checked. A caller that built a header by hand and
        // forgot the flag would otherwise write a record that this crate's own
        // parser refuses, and failing at read time for a mistake made at write
        // time is the worst place to put the error.
        let flags = self.flags | record_flags::CHECKSUMMED;
        let n = total_len(flags, key.len(), value.len())?;
        if buf.len() < n {
            return Err(
                Error::new(Code::Full, "the record does not fit in the buffer")
                    .with_detail(format!("need={n} have={}", buf.len())),
            );
        }
        let h = header_len(flags);
        put_u8(buf, 4, self.kind);
        put_u8(buf, 5, flags);
        put_u16(buf, 6, key.len() as u16);
        put_u64(buf, 8, self.prev);
        if flags & record_flags::HAS_TTL != 0 {
            put_u64(buf, 16, self.ttl_ms);
        }
        buf[h..h + key.len()].copy_from_slice(key);
        let v = h + key.len();
        buf[v..v + value.len()].copy_from_slice(value);

        // The `len` field is part of what the trailer covers, and it is not in
        // the buffer yet, so it is fed in from the value that is about to go
        // there. Doing it any other way means either writing `len` early, which
        // breaks the ordering rule, or leaving the length out of the checksum,
        // which leaves the one field a torn write is most likely to damage
        // unprotected.
        let c = crc32c(0, &(n as u32).to_le_bytes());
        let c = crc32c(c, &buf[4..n - TRAILER_LEN]);
        put_u32(buf, n - TRAILER_LEN, c);
        Ok(n)
    }
}

/// Stores `len`, which is what publishes a record.
///
/// Separate from [`RecordHeader::fill`] so that the release store in the log is
/// visibly the last thing that happens. The caller does the fence.
///
/// # Panics
///
/// If `len` is zero, because zero is the end of log sentinel and a record of
/// length zero would end the log at itself.
pub fn seal_len(buf: &mut [u8], len: usize) {
    assert!(len != 0, "zero is the end of log sentinel, not a length");
    put_u32(buf, 0, len as u32);
}

/// A record read back out of a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordRef<'a> {
    /// The exact length, header and trailer included.
    pub len: u32,
    /// The raw kind byte. Use [`RecordRef::kind`] to ask what it means.
    pub kind: u8,
    /// See [`record_flags`].
    pub flags: u8,
    /// The previous address in this key's chain, or 0.
    pub prev: u64,
    /// Unix milliseconds, or `None` if the record has no TTL.
    pub ttl_ms: Option<u64>,
    /// The key, borrowed from the page.
    pub key: &'a [u8],
    /// The value, borrowed from the page.
    pub value: &'a [u8],
}

impl<'a> RecordRef<'a> {
    /// Parses the record at the front of `bytes`.
    ///
    /// `Ok(None)` means `len` was zero, which is the end of the log and not an
    /// error. That is the normal way replay finishes (`06` section 4) and it is
    /// also what a half written tail looks like, which is exactly the point:
    /// the two are indistinguishable and both mean stop here.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the length is impossible, if the record runs past
    /// the end of `bytes`, or if a checksummed record fails its checksum.
    pub fn parse(bytes: &'a [u8]) -> Result<Option<RecordRef<'a>>> {
        if bytes.len() < 4 {
            return Ok(None);
        }
        let len = get_u32(bytes, 0) as usize;
        if len == 0 {
            return Ok(None);
        }
        let flags = get_u8(bytes, 5);
        if flags & record_flags::CHECKSUMMED == 0 {
            // Not "this record has no checksum". There is no such record, so
            // this is a flipped bit in the flags byte, and it has to be caught
            // here because clearing this particular bit is the one corruption
            // the checksum cannot catch: it turns the checksum off.
            return Err(
                Error::new(Code::Corrupt, "a record with its checksum flag clear")
                    .with_detail(format!("flags={flags:#04x}")),
            );
        }
        let h = header_len(flags);
        let t = trailer_len(flags);
        let klen = get_u16(bytes, 6) as usize;

        if len < h + klen + t {
            return Err(
                Error::new(Code::Corrupt, "the record is shorter than its own header")
                    .with_detail(format!("len={len} header={h} klen={klen} trailer={t}")),
            );
        }
        if len > bytes.len() {
            // A tail that was cut mid record. `06` section 4 calls for
            // truncating to the last good record, so this is corruption the
            // caller is expected to handle rather than a reason to give up.
            return Err(
                Error::new(Code::Corrupt, "the record runs past the end of the page")
                    .with_detail(format!("len={len} available={}", bytes.len())),
            );
        }

        if flags & record_flags::CHECKSUMMED != 0 {
            let want = get_u32(bytes, len - TRAILER_LEN);
            let got = crc32c(0, &bytes[..len - TRAILER_LEN]);
            if want != got {
                return Err(Error::new(Code::Corrupt, "record checksum mismatch")
                    .with_detail(format!("stored={want:#010x} computed={got:#010x}")));
            }
        }

        let ttl_ms = if flags & record_flags::HAS_TTL != 0 {
            Some(get_u64(bytes, 16))
        } else {
            None
        };

        Ok(Some(RecordRef {
            len: len as u32,
            kind: get_u8(bytes, 4),
            flags,
            prev: get_u64(bytes, 8),
            ttl_ms,
            key: &bytes[h..h + klen],
            value: &bytes[h + klen..len - t],
        }))
    }

    /// How far the next record is, which is [`RecordRef::len`] rounded up to
    /// eight.
    ///
    /// Walk by this and not by `len`.
    #[must_use]
    pub fn stride(&self) -> usize {
        align_up(self.len as usize)
    }

    /// The kind, if this version knows it.
    ///
    /// `None` is not an error. It means a newer writer put something here and
    /// `len` says how far to jump to get past it.
    #[must_use]
    pub fn kind(&self) -> Option<RecordKind> {
        RecordKind::from_u8(self.kind)
    }

    /// Whether this record deletes its key.
    #[must_use]
    pub fn is_tombstone(&self) -> bool {
        self.kind == RecordKind::Tombstone.as_u8()
    }

    /// Whether the value is elsewhere and this record only points at it.
    #[must_use]
    pub fn is_tiered(&self) -> bool {
        self.flags & record_flags::TIERED != 0
    }
}

/// Walks the records in a page payload.
///
/// Stops at the first `len == 0`, and yields an error for the first record that
/// does not parse, after which it stops too. Both are how replay ends, so
/// neither is exceptional.
pub struct RecordIter<'a> {
    bytes: &'a [u8],
    at: usize,
    done: bool,
}

impl<'a> RecordIter<'a> {
    /// Walks `bytes`, which is a page payload, not a whole page.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> RecordIter<'a> {
        RecordIter {
            bytes,
            at: 0,
            done: false,
        }
    }

    /// The offset the walk stopped at, which is what a truncation truncates to.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.at
    }
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Result<RecordRef<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.at >= self.bytes.len() {
            return None;
        }
        match RecordRef::parse(&self.bytes[self.at..]) {
            Ok(Some(r)) => {
                self.at += r.stride();
                Some(Ok(r))
            }
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RECORD_ALIGN;

    fn write(h: RecordHeader, key: &[u8], value: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 4096];
        let n = h.fill(&mut buf, key, value).unwrap();
        seal_len(&mut buf, n);
        buf.truncate(align_up(n));
        buf
    }

    #[test]
    fn a_record_round_trips() {
        let h = RecordHeader::new(RecordKind::String).after(4096);
        let buf = write(h, b"greeting", b"hello");
        let r = RecordRef::parse(&buf).unwrap().unwrap();
        assert_eq!(r.kind(), Some(RecordKind::String));
        assert_eq!(r.key, b"greeting");
        assert_eq!(r.value, b"hello");
        assert_eq!(r.prev, 4096);
        assert_eq!(r.ttl_ms, None);
        assert!(!r.is_tombstone());
    }

    #[test]
    fn every_field_lands_where_the_specification_says() {
        let h = RecordHeader::new(RecordKind::Document)
            .with_ttl(1_700_000_000_000)
            .after(0x1122_3344_5566_7788);
        let buf = write(h, b"k", b"v");
        assert_eq!(get_u32(&buf, 0) as usize, HEADER_LEN_TTL + 1 + 1 + 4);
        assert_eq!(get_u8(&buf, 4), 2, "document is kind 2");
        assert_eq!(
            get_u8(&buf, 5),
            record_flags::CHECKSUMMED | record_flags::HAS_TTL
        );
        assert_eq!(get_u16(&buf, 6), 1);
        assert_eq!(get_u64(&buf, 8), 0x1122_3344_5566_7788);
        assert_eq!(get_u64(&buf, 16), 1_700_000_000_000);
        assert_eq!(buf[24], b'k');
        assert_eq!(buf[25], b'v');
    }

    #[test]
    fn a_ttl_costs_eight_bytes_and_moves_the_key() {
        let plain = write(RecordHeader::new(RecordKind::String), b"key", b"value");
        let ttl = write(
            RecordHeader::new(RecordKind::String).with_ttl(1),
            b"key",
            b"value",
        );
        assert_eq!(get_u32(&ttl, 0) - get_u32(&plain, 0), 8);
        let r = RecordRef::parse(&ttl).unwrap().unwrap();
        assert_eq!(r.ttl_ms, Some(1));
        assert_eq!(r.key, b"key");
        assert_eq!(r.value, b"value");
    }

    #[test]
    fn a_zero_length_is_the_end_of_the_log_and_not_an_error() {
        assert!(RecordRef::parse(&[0u8; 64]).unwrap().is_none());
        assert!(RecordRef::parse(&[]).unwrap().is_none());
        assert!(RecordRef::parse(&[1, 2, 3]).unwrap().is_none());
    }

    #[test]
    fn len_is_exact_so_a_value_of_any_length_survives() {
        // The failure this guards against is padding counted as value. Any odd
        // length would show it, so every length from zero to thirty two is
        // checked.
        for n in 0..=32usize {
            let value: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let buf = write(RecordHeader::new(RecordKind::String), b"k", &value);
            let r = RecordRef::parse(&buf).unwrap().unwrap();
            assert_eq!(r.value, &value[..], "value of {n} bytes came back wrong");
            assert_eq!(r.value.len(), n);
        }
    }

    #[test]
    fn the_stride_is_aligned_even_when_the_length_is_not() {
        let buf = write(RecordHeader::new(RecordKind::String), b"k", b"abc");
        let r = RecordRef::parse(&buf).unwrap().unwrap();
        assert_eq!(r.len as usize, HEADER_LEN + 1 + 3 + TRAILER_LEN);
        assert_eq!(r.len % RECORD_ALIGN as u32, 0);

        let buf = write(RecordHeader::new(RecordKind::String), b"k", b"ab");
        let r = RecordRef::parse(&buf).unwrap().unwrap();
        assert_eq!(r.len as usize, 23);
        assert_eq!(r.stride(), 24, "the padding is between records, not inside");
    }

    #[test]
    fn walking_a_page_by_stride_stays_in_step() {
        // Odd sized values on purpose: walking by `len` desynchronises on the
        // first one and this test is the reason `stride` exists.
        let mut page = vec![0u8; 4096];
        let mut at = 0usize;
        let mut written = Vec::new();
        for i in 0..40usize {
            let key = format!("key{i}");
            let value = vec![b'v'; i];
            let h = RecordHeader::new(RecordKind::String).after(at as u64);
            let n = h.fill(&mut page[at..], key.as_bytes(), &value).unwrap();
            seal_len(&mut page[at..], n);
            written.push((key, value));
            at += align_up(n);
        }

        let got: Vec<_> = RecordIter::new(&page).map(|r| r.unwrap()).collect();
        assert_eq!(got.len(), 40);
        for (r, (key, value)) in got.iter().zip(&written) {
            assert_eq!(r.key, key.as_bytes());
            assert_eq!(r.value, &value[..]);
        }
    }

    #[test]
    fn the_iterator_reports_where_it_stopped() {
        let mut page = vec![0u8; 512];
        let h = RecordHeader::new(RecordKind::String);
        let n = h.fill(&mut page, b"a", b"bb").unwrap();
        seal_len(&mut page, n);
        let mut it = RecordIter::new(&page);
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert_eq!(it.offset(), align_up(n), "the tail is here");
    }

    #[test]
    fn a_flipped_bit_anywhere_in_a_checksummed_record_is_caught() {
        let good = write(
            RecordHeader::new(RecordKind::String).with_ttl(99),
            b"the key",
            b"the value, which is long enough to be worth checking",
        );
        let len = get_u32(&good, 0) as usize;
        for i in 0..len {
            let mut bad = good.clone();
            bad[i] ^= 0x20;
            let r = RecordRef::parse(&bad);
            // A hit in `len` may shorten the record into the end of log
            // sentinel or past the buffer. Any of those is fine. What is not
            // fine is a record that parses and claims the damaged bytes are the
            // value.
            match r {
                Err(_) => {}
                Ok(None) => {}
                Ok(Some(rec)) => panic!("byte {i} was not caught, got {rec:?}"),
            }
        }
    }

    /// A caller that builds a header by hand and forgets the flag gets the
    /// checksum anyway. The alternative is a record that this crate's own parser
    /// refuses, which turns a mistake made at write time into a failure at read
    /// time, and that is the worst place to put it.
    #[test]
    fn a_header_written_without_the_checksum_flag_gets_one_anyway() {
        let h = RecordHeader {
            kind: RecordKind::String.as_u8(),
            flags: 0,
            prev: 0,
            ttl_ms: 0,
        };
        let buf = write(h, b"k", b"v");
        assert_eq!(get_u32(&buf, 0) as usize, HEADER_LEN + 2 + TRAILER_LEN);
        assert_ne!(get_u8(&buf, 5) & record_flags::CHECKSUMMED, 0);
        let r = RecordRef::parse(&buf).unwrap().unwrap();
        assert_eq!(r.value, b"v");
    }

    /// Clearing the flag is the one corruption a checksum cannot catch, because
    /// what it corrupts is the checksum itself. So the flag being clear is
    /// treated as damage rather than as a record that chose not to have one.
    #[test]
    fn a_record_with_its_checksum_flag_cleared_is_corruption() {
        let mut buf = write(RecordHeader::new(RecordKind::String), b"key", b"value");
        buf[5] &= !record_flags::CHECKSUMMED;
        let err = RecordRef::parse(&buf).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
    }

    #[test]
    fn a_length_that_is_shorter_than_the_header_is_corruption() {
        let mut buf = write(RecordHeader::new(RecordKind::String), b"key", b"value");
        put_u32(&mut buf, 0, 12);
        let err = RecordRef::parse(&buf).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("len=12"));
    }

    #[test]
    fn a_record_cut_off_by_a_torn_write_is_corruption() {
        let buf = write(RecordHeader::new(RecordKind::String), b"key", b"value");
        let err = RecordRef::parse(&buf[..8]).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("available=8"));
    }

    #[test]
    fn an_unknown_kind_is_skipped_rather_than_refused() {
        // `07` section 9: a version one reader jumps a kind it does not know by
        // `len` and carries on. If this ever starts failing, the format has
        // stopped being forward compatible.
        let h = RecordHeader {
            kind: 200,
            flags: record_flags::CHECKSUMMED,
            prev: 0,
            ttl_ms: 0,
        };
        let mut page = vec![0u8; 512];
        let n = h.fill(&mut page, b"future", b"stuff").unwrap();
        seal_len(&mut page, n);
        let after = align_up(n);
        let m = RecordHeader::new(RecordKind::String)
            .fill(&mut page[after..], b"k", b"v")
            .unwrap();
        seal_len(&mut page[after..], m);

        let got: Vec<_> = RecordIter::new(&page).map(|r| r.unwrap()).collect();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].kind(), None, "not a kind this version knows");
        assert_eq!(got[0].key, b"future");
        assert_eq!(got[1].kind(), Some(RecordKind::String));
    }

    #[test]
    fn a_key_larger_than_a_u16_is_refused_rather_than_truncated() {
        let key = vec![b'k'; MAX_KEY_LEN + 1];
        let mut buf = vec![0u8; MAX_KEY_LEN + 64];
        let err = RecordHeader::new(RecordKind::String)
            .fill(&mut buf, &key, b"v")
            .unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
    }

    #[test]
    fn a_buffer_with_no_room_says_how_much_it_needed() {
        let mut buf = [0u8; 8];
        let err = RecordHeader::new(RecordKind::String)
            .fill(&mut buf, b"key", b"value")
            .unwrap_err();
        assert_eq!(err.code(), Code::Full);
        assert!(err.detail().unwrap().contains("have=8"));
    }

    #[test]
    fn kinds_round_trip_and_chunks_have_no_key() {
        for k in RecordKind::ALL {
            assert_eq!(RecordKind::from_u8(k.as_u8()), Some(k));
        }
        assert_eq!(RecordKind::from_u8(9), None);
        assert!(!RecordKind::CollectionChunk.carries_a_key());
        assert!(RecordKind::String.carries_a_key());
        assert!(RecordKind::Tombstone.carries_a_key());
    }

    #[test]
    fn a_tombstone_is_a_record_with_no_value() {
        let buf = write(RecordHeader::new(RecordKind::Tombstone), b"gone", b"");
        let r = RecordRef::parse(&buf).unwrap().unwrap();
        assert!(r.is_tombstone());
        assert_eq!(r.value, b"");
        assert_eq!(r.key, b"gone");
    }
}
