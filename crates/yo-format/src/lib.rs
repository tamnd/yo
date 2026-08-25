//! The byte layouts of a `.yo` file, and nothing else.
//!
//! This crate is the written form of `07-yo-file-format.md`. It knows how to
//! turn each structure into bytes and how to read it back, and it knows nothing
//! about files, memory, shards or the engine. That separation is deliberate:
//! the engine and `yo check` both encode with this crate, and the independent
//! minimal reader deliberately does not, so a change here that is not also a
//! change to the specification shows up as the two disagreeing.
//!
//! Rules that hold everywhere below.
//!
//! Little endian, unconditionally. A big endian machine byte swaps on the way
//! in and on the way out, and pays for it, because a format whose byte order
//! depends on the writer is not a format.
//!
//! Every checksum is CRC32C over the bytes named, with the checksum field
//! itself read as zero. That way the check is the same computation whether you
//! are about to write the bytes or have just read them.
//!
//! No allocation. Everything encodes into a caller supplied slice and decodes
//! from a borrowed one. The engine calls these from inside the shard loop and
//! the shard loop does not allocate.

#![deny(missing_docs)]

pub mod catalog;
pub mod page;
pub mod record;
pub mod superblock;

pub use catalog::{Band, CatalogEntry, Model, ValueType};
pub use page::{PAGE_HEADER_LEN, PageHeader};
pub use record::{RecordHeader, RecordKind, RecordRef, record_flags};
pub use superblock::{CheckpointEntry, Superblock, superblock_flags};

/// The sixteen bytes at offset zero of every `.yo` file.
///
/// Sixteen and not eight so that the human readable part survives a hex dump,
/// and trailing NULs rather than spaces so that a C string comparison of the
/// first twelve bytes does the right thing.
pub const MAGIC: [u8; 16] = *b"tamndyo fmt001\0\0";

/// The format this build writes.
pub const FORMAT_VERSION: u32 = 1;

/// The lowest reader version that can read what this build writes.
///
/// Section 9 of the format document is the whole policy: a change a version one
/// reader can skip past does not move this, and a change it would misread does.
/// It is not the same number as [`FORMAT_VERSION`] and conflating them is how a
/// reader ends up refusing a file it could have read.
pub const MIN_READER_VERSION: u32 = 1;

// A build that writes files it cannot read is a build nobody should get. The
// two numbers are equal today and the assertion exists for the day they are
// not, because the mistake it catches is a one character edit.
const _: () = assert!(MIN_READER_VERSION <= FORMAT_VERSION);

/// A superblock slot, and therefore the offset of the second one.
pub const SUPERBLOCK_LEN: usize = 16 * 1024;

/// Where the data starts, which is after both superblock slots.
pub const DATA_START: u64 = 2 * SUPERBLOCK_LEN as u64;

/// The default segment size, and the size a file gets if nobody chooses.
pub const DEFAULT_PAGE_SIZE: u32 = 16384;

/// The smallest legal segment size.
///
/// Four kibibytes is the torn write unit the format assumes and nothing
/// smaller would be a unit at all.
pub const MIN_PAGE_SIZE: u32 = 4096;

/// The largest legal segment size.
pub const MAX_PAGE_SIZE: u32 = 65536;

/// A log page is 32 MiB laid across contiguous segments.
///
/// F2's constant. Three resident pages is about 96 MiB of working memory, and
/// the contiguity is what makes writing one page one submission rather than a
/// scatter list.
pub const LOG_PAGE_LEN: u64 = 32 * 1024 * 1024;

/// Records are eight byte aligned, so every length rounds up to this.
pub const RECORD_ALIGN: usize = 8;

/// Is `n` a legal segment size?
///
/// Powers of two between [`MIN_PAGE_SIZE`] and [`MAX_PAGE_SIZE`]. Anything else
/// is refused at creation rather than at the first write, because a file with a
/// nonsensical segment size is not a file anyone can recover.
#[must_use]
pub const fn is_legal_page_size(n: u32) -> bool {
    n.is_power_of_two() && n >= MIN_PAGE_SIZE && n <= MAX_PAGE_SIZE
}

/// `n` rounded up to the next multiple of [`RECORD_ALIGN`].
#[inline]
#[must_use]
pub const fn align_up(n: usize) -> usize {
    n.next_multiple_of(RECORD_ALIGN)
}

// ---------------------------------------------------------------------------
// The little endian primitives every layout below is built from.
//
// These exist rather than `from_le_bytes` at each call site because each call
// site would need its own slice indexing and its own panic, and there are about
// two hundred of them.
// ---------------------------------------------------------------------------

/// Reads a `u8` at `off`, or 0 if the slice is too short.
///
/// Short reads return zero rather than panicking because every caller here has
/// already checked the length of the whole structure, and a bounds check per
/// field is a branch per field on the recovery path.
#[inline]
#[must_use]
pub fn get_u8(b: &[u8], off: usize) -> u8 {
    b.get(off).copied().unwrap_or(0)
}

/// Reads a little endian `u16` at `off`, or 0 if the slice is too short.
#[inline]
#[must_use]
pub fn get_u16(b: &[u8], off: usize) -> u16 {
    match b.get(off..off + 2) {
        Some(s) => u16::from_le_bytes([s[0], s[1]]),
        None => 0,
    }
}

/// Reads a little endian `u32` at `off`, or 0 if the slice is too short.
#[inline]
#[must_use]
pub fn get_u32(b: &[u8], off: usize) -> u32 {
    match b.get(off..off + 4) {
        Some(s) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => 0,
    }
}

/// Reads a little endian `u64` at `off`, or 0 if the slice is too short.
#[inline]
#[must_use]
pub fn get_u64(b: &[u8], off: usize) -> u64 {
    match b.get(off..off + 8) {
        Some(s) => u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
        None => 0,
    }
}

/// Writes `v` at `off`. Does nothing if the slice is too short.
#[inline]
pub fn put_u8(b: &mut [u8], off: usize, v: u8) {
    if let Some(slot) = b.get_mut(off) {
        *slot = v;
    }
}

/// Writes `v` little endian at `off`. Does nothing if the slice is too short.
#[inline]
pub fn put_u16(b: &mut [u8], off: usize, v: u16) {
    if let Some(s) = b.get_mut(off..off + 2) {
        s.copy_from_slice(&v.to_le_bytes());
    }
}

/// Writes `v` little endian at `off`. Does nothing if the slice is too short.
#[inline]
pub fn put_u32(b: &mut [u8], off: usize, v: u32) {
    if let Some(s) = b.get_mut(off..off + 4) {
        s.copy_from_slice(&v.to_le_bytes());
    }
}

/// Writes `v` little endian at `off`. Does nothing if the slice is too short.
#[inline]
pub fn put_u64(b: &mut [u8], off: usize, v: u64) {
    if let Some(s) = b.get_mut(off..off + 8) {
        s.copy_from_slice(&v.to_le_bytes());
    }
}

/// CRC32C over `bytes`, with the four bytes at `skip` treated as zero.
///
/// Every checksum in the format is defined this way, so it is one function
/// rather than a convention each structure re-implements. The field is skipped
/// rather than excluded so that the covered range stays contiguous and stays
/// easy to state in the specification.
#[must_use]
pub fn checksum_skipping(bytes: &[u8], skip: usize) -> u32 {
    if skip + 4 > bytes.len() {
        return yo_common::crc32c(0, bytes);
    }
    let c = yo_common::crc32c(0, &bytes[..skip]);
    let c = yo_common::crc32c(c, &[0, 0, 0, 0]);
    yo_common::crc32c(c, &bytes[skip + 4..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_magic_is_what_the_specification_says() {
        assert_eq!(MAGIC.len(), 16);
        assert_eq!(&MAGIC[..14], b"tamndyo fmt001");
        assert_eq!(&MAGIC[14..], b"\0\0");
    }

    #[test]
    fn page_sizes_are_powers_of_two_in_range() {
        assert!(is_legal_page_size(4096));
        assert!(is_legal_page_size(16384));
        assert!(is_legal_page_size(65536));
        assert!(!is_legal_page_size(2048), "below the torn write unit");
        assert!(!is_legal_page_size(131_072), "above the maximum");
        assert!(!is_legal_page_size(12288), "not a power of two");
        assert!(!is_legal_page_size(0));
        assert!(is_legal_page_size(DEFAULT_PAGE_SIZE));
    }

    #[test]
    fn a_log_page_is_a_whole_number_of_segments_at_every_legal_size() {
        let mut n = MIN_PAGE_SIZE;
        while n <= MAX_PAGE_SIZE {
            assert_eq!(
                LOG_PAGE_LEN % u64::from(n),
                0,
                "a 32 MiB log page must divide into {n} byte segments"
            );
            n *= 2;
        }
        assert_eq!(LOG_PAGE_LEN / u64::from(DEFAULT_PAGE_SIZE), 2048);
    }

    #[test]
    fn alignment_rounds_up_and_leaves_aligned_values_alone() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 8);
        assert_eq!(align_up(8), 8);
        assert_eq!(align_up(9), 16);
        assert_eq!(align_up(RECORD_ALIGN * 3), RECORD_ALIGN * 3);
    }

    #[test]
    fn data_starts_after_both_superblock_slots() {
        assert_eq!(DATA_START, 32768);
        assert_eq!(SUPERBLOCK_LEN, 16384);
    }

    #[test]
    fn short_reads_give_zero_rather_than_panicking() {
        let b = [1u8, 2, 3];
        assert_eq!(get_u8(&b, 0), 1);
        assert_eq!(get_u8(&b, 9), 0);
        assert_eq!(get_u16(&b, 0), 0x0201);
        assert_eq!(
            get_u16(&b, 2),
            0,
            "would need two bytes and only one is left"
        );
        assert_eq!(get_u32(&b, 0), 0);
        assert_eq!(get_u64(&b, 0), 0);
    }

    #[test]
    fn short_writes_do_nothing_rather_than_panicking() {
        let mut b = [0u8; 3];
        put_u32(&mut b, 0, 0xdead_beef);
        assert_eq!(b, [0, 0, 0], "no room, so nothing was written");
        put_u16(&mut b, 0, 0x1234);
        assert_eq!(b, [0x34, 0x12, 0]);
    }

    #[test]
    fn round_trips_are_little_endian_on_every_machine() {
        let mut b = [0u8; 8];
        put_u64(&mut b, 0, 0x0102_0304_0506_0708);
        assert_eq!(b, [8, 7, 6, 5, 4, 3, 2, 1], "little endian, byte for byte");
        assert_eq!(get_u64(&b, 0), 0x0102_0304_0506_0708);
    }

    #[test]
    fn the_checksum_reads_its_own_field_as_zero() {
        let mut b = vec![0u8; 32];
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = i as u8;
        }
        let want = checksum_skipping(&b, 28);
        // Whatever ends up in the field, the answer is the same, which is what
        // makes "compute then store" and "read then verify" one computation.
        put_u32(&mut b, 28, want);
        assert_eq!(checksum_skipping(&b, 28), want);
        put_u32(&mut b, 28, 0xffff_ffff);
        assert_eq!(checksum_skipping(&b, 28), want);
        // And a change anywhere else does move it.
        b[3] ^= 1;
        assert_ne!(checksum_skipping(&b, 28), want);
    }
}
