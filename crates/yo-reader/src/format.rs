//! The `.yo` layout, written out again from `07` and `06`.
//!
//! Every constant and every offset in this module is a second transcription of
//! the specification. None of it is imported from `yo-format`, and that is the
//! only reason this crate is worth having: two transcriptions that agree are
//! evidence, and one transcription used twice is not.
//!
//! Which means the useful failure mode here is a disagreement, not a panic. If
//! this module and `yo-format` ever differ, one of them has misread the
//! specification and the interesting question is which. So the decoders are
//! written the dull way, offset by offset, with the field name next to the
//! number, and nothing is factored into a helper that would let a mistake apply
//! itself in two places at once.

use crate::crc::{crc32c, crc32c_skipping};
use crate::error::{Error, Result};

/// The first sixteen bytes of the file.
pub const MAGIC: [u8; 16] = *b"tamndyo fmt001\0\0";

/// The format this reader understands.
pub const FORMAT_VERSION: u32 = 1;

/// One superblock slot.
pub const SUPERBLOCK_LEN: usize = 16 * 1024;

/// Where the checksum sits inside a slot.
pub const SUPERBLOCK_CRC_OFFSET: usize = 16380;

/// Two slots, then data.
pub const DATA_START: u64 = 2 * SUPERBLOCK_LEN as u64;

/// A log segment, which is also a region of the file.
pub const LOG_PAGE_LEN: u64 = 32 * 1024 * 1024;

/// The header at the front of every log segment.
pub const PAGE_HEADER_LEN: usize = 32;

/// `YOLG`.
pub const PAGE_MAGIC: u32 = 0x594f_4c47;

/// One checkpoint entry, one per shard.
pub const CHECKPOINT_ENTRY_LEN: usize = 64;

/// Redis hash slots, which is how many entries the shard table expands to.
pub const SLOT_COUNT: usize = 16384;

/// Records are eight byte aligned, with the padding between them.
pub const RECORD_ALIGN: usize = 8;

/// A record header with no TTL.
pub const RECORD_HEADER_LEN: usize = 16;

/// A record header with one.
pub const RECORD_HEADER_LEN_TTL: usize = 24;

/// The record trailer, which is always present.
pub const RECORD_TRAILER_LEN: usize = 4;

/// Bits in the superblock `flags` field.
pub mod superblock_flags {
    /// The database was closed on purpose.
    pub const CLEAN_SHUTDOWN: u32 = 1 << 0;
    /// Values are encrypted.
    pub const ENCRYPTED: u32 = 1 << 1;
    /// There is an archival root.
    pub const HAS_ARCHIVAL: u32 = 1 << 2;
    /// Some values live only on the cold tier.
    pub const TIERING_ENGAGED: u32 = 1 << 3;
}

/// Bits in a record's `flags` byte.
pub mod record_flags {
    /// The value is on the cold tier, not here.
    pub const TIERED: u8 = 1 << 0;
    /// The value is compressed.
    pub const COMPRESSED: u8 = 1 << 1;
    /// An eight byte expiry sits at offset 16.
    pub const HAS_TTL: u8 = 1 << 2;
    /// The collection has a shape tag in the catalogue.
    pub const SHAPE_TAGGED: u8 = 1 << 3;
    /// The last four bytes are a CRC32C.
    pub const CHECKSUMMED: u8 = 1 << 4;
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(v)
}

/// A page size the format allows: a power of two from 4 KiB to 64 KiB.
#[must_use]
pub const fn is_legal_page_size(n: u32) -> bool {
    n.is_power_of_two() && n >= 4096 && n <= 65536
}

/// The header half of a superblock slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    /// What wrote it.
    pub format_version: u32,
    /// The lowest reader it will accept.
    pub min_reader_version: u32,
    /// Segment size.
    pub page_size: u32,
    /// How many shards.
    pub shard_count: u32,
    /// Higher wins between the two slots.
    pub seq: u64,
    /// How big the file was at this checkpoint.
    pub file_size: u64,
    /// Identifies the database across copies.
    pub file_uuid: [u8; 16],
    /// When the file was made.
    pub created_unix_ms: u64,
    /// When this checkpoint was taken.
    pub checkpoint_unix_ms: u64,
    /// Logical databases.
    pub db_count: u32,
    /// See [`superblock_flags`].
    pub flags: u32,
    /// Replication id, high bits.
    pub replid_hi: u64,
    /// Replication id, middle bits.
    pub replid_lo: u64,
    /// Replication id, low bits.
    pub replid_ext: u32,
    /// Where the run length encoded shard table starts inside the slot.
    pub shard_table_off: u32,
    /// How long that encoding is.
    pub shard_table_len: u16,
    /// The catalogue, or 0.
    pub catalog_addr: u64,
    /// The free segment list.
    pub free_list_addr: u64,
    /// The archival root, or 0.
    pub archival_root: u64,
}

impl Superblock {
    /// Decodes a 16 KiB slot.
    ///
    /// The order of the checks is deliberate and matches the engine's: length,
    /// magic, checksum, then version. Checking the version before the checksum
    /// reports a corrupt slot as a file from the future, which sends whoever is
    /// holding it looking for a newer build that does not exist.
    ///
    /// # Errors
    ///
    /// If the slot is the wrong size, is not a `.yo` file, fails its checksum,
    /// needs a newer reader, or carries a page size or shard count that cannot
    /// be right.
    pub fn decode(slot: &[u8]) -> Result<Superblock> {
        if slot.len() != SUPERBLOCK_LEN {
            return Err(Error::new("a superblock slot is 16384 bytes"));
        }
        if slot[..16] != MAGIC {
            return Err(Error::new(
                "not a .yo file: the first sixteen bytes are not the magic",
            ));
        }
        let want = u32_at(slot, SUPERBLOCK_CRC_OFFSET);
        let got = crc32c_skipping(slot, SUPERBLOCK_CRC_OFFSET);
        if want != got {
            return Err(Error::new(format!(
                "superblock checksum mismatch: stored {want:#010x}, computed {got:#010x}"
            )));
        }
        let min_reader_version = u32_at(slot, 20);
        if min_reader_version > FORMAT_VERSION {
            return Err(Error::new(format!(
                "this file wants a reader of version {min_reader_version} and this is {FORMAT_VERSION}"
            )));
        }
        let page_size = u32_at(slot, 24);
        if !is_legal_page_size(page_size) {
            return Err(Error::new(format!(
                "the segment size {page_size} is not a legal value"
            )));
        }
        let shard_count = u32_at(slot, 28);
        if shard_count == 0 {
            return Err(Error::new("a file with no shards"));
        }

        let mut file_uuid = [0u8; 16];
        file_uuid.copy_from_slice(&slot[48..64]);

        Ok(Superblock {
            format_version: u32_at(slot, 16),
            min_reader_version,
            page_size,
            shard_count,
            seq: u64_at(slot, 32),
            file_size: u64_at(slot, 40),
            file_uuid,
            created_unix_ms: u64_at(slot, 64),
            checkpoint_unix_ms: u64_at(slot, 72),
            db_count: u32_at(slot, 80),
            flags: u32_at(slot, 84),
            replid_hi: u64_at(slot, 88),
            replid_lo: u64_at(slot, 96),
            replid_ext: u32_at(slot, 104),
            shard_table_off: u32_at(slot, 108),
            catalog_addr: u64_at(slot, 112),
            free_list_addr: u64_at(slot, 120),
            archival_root: u64_at(slot, 128),
            // The reserved run in the table in `07` section 2 is 24 bytes, and
            // the paragraph under it puts `shard_table_len` at 156. They cannot
            // both be right. The paragraph is the one the engine follows, so it
            // is the one followed here, and this comment is the note to check
            // if the two crates ever disagree about a file.
            shard_table_len: u16_at(slot, 156),
        })
    }

    /// Where the checkpoint entries start inside the slot.
    #[must_use]
    pub const fn checkpoint_off(&self) -> usize {
        self.shard_table_off as usize + self.shard_table_len as usize
    }

    /// Whether every checkpoint entry fits below the checksum.
    #[must_use]
    pub fn checkpoints_fit(&self) -> bool {
        let end = self
            .checkpoint_off()
            .saturating_add(self.shard_count as usize * CHECKPOINT_ENTRY_LEN);
        self.shard_table_off as usize >= 160 && end <= SUPERBLOCK_CRC_OFFSET
    }

    /// Whether the file was closed on purpose.
    #[must_use]
    pub const fn clean_shutdown(&self) -> bool {
        self.flags & superblock_flags::CLEAN_SHUTDOWN != 0
    }
}

/// One shard's position in the log at the last checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointEntry {
    /// The oldest address still in the file.
    pub log_begin: u64,
    /// Stable up to here.
    pub log_head: u64,
    /// Read only up to here.
    pub log_read_only: u64,
    /// Where replay starts.
    pub log_tail: u64,
    /// Where the index image is.
    pub index_image_addr: u64,
    /// How long it is.
    pub index_image_len: u64,
    /// Keys at the checkpoint.
    pub key_count: u64,
    /// The shard's epoch.
    pub epoch: u32,
}

impl CheckpointEntry {
    /// Decodes 64 bytes.
    ///
    /// # Errors
    ///
    /// If the length is wrong, the checksum fails, or the four addresses are
    /// not in order. The last one matters most: addresses that go backwards
    /// describe a log region with a negative length, and replaying one of those
    /// builds a database that looks fine out of whatever bytes were lying
    /// around.
    pub fn decode(buf: &[u8]) -> Result<CheckpointEntry> {
        if buf.len() != CHECKPOINT_ENTRY_LEN {
            return Err(Error::new("a checkpoint entry is 64 bytes"));
        }
        let want = u32_at(buf, 60);
        let got = crc32c_skipping(buf, 60);
        if want != got {
            return Err(Error::new(format!(
                "checkpoint entry checksum mismatch: stored {want:#010x}, computed {got:#010x}"
            )));
        }
        let e = CheckpointEntry {
            log_begin: u64_at(buf, 0),
            log_head: u64_at(buf, 8),
            log_read_only: u64_at(buf, 16),
            log_tail: u64_at(buf, 24),
            index_image_addr: u64_at(buf, 32),
            index_image_len: u64_at(buf, 40),
            key_count: u64_at(buf, 48),
            epoch: u32_at(buf, 56),
        };
        if !(e.log_begin <= e.log_head
            && e.log_head <= e.log_read_only
            && e.log_read_only <= e.log_tail)
        {
            return Err(Error::new(format!(
                "the four log addresses are not in order: begin {} head {} read_only {} tail {}",
                e.log_begin, e.log_head, e.log_read_only, e.log_tail
            )));
        }
        Ok(e)
    }
}

/// Expands the run length encoded slot to shard mapping.
///
/// Pairs of `u16`: a run length, then the shard the run belongs to.
///
/// # Errors
///
/// If the byte count is not a whole number of pairs, a run is zero long, a
/// shard id is out of range, or the runs do not add up to exactly
/// [`SLOT_COUNT`]. Every one of those is what a flipped bit looks like, and
/// every one of them would otherwise send a key to a shard that is not there.
pub fn decode_shard_table(bytes: &[u8], shard_count: u32) -> Result<Vec<u16>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::new("the shard table is not a whole number of runs"));
    }
    let mut out = Vec::with_capacity(SLOT_COUNT);
    for pair in bytes.as_chunks::<4>().0 {
        let run = u16_at(pair, 0) as usize;
        let shard = u16_at(pair, 2);
        if run == 0 {
            return Err(Error::new("a zero length run in the shard table"));
        }
        if u32::from(shard) >= shard_count {
            return Err(Error::new(format!(
                "a slot points at shard {shard} and there are {shard_count}"
            )));
        }
        if out.len() + run > SLOT_COUNT {
            return Err(Error::new("the runs cover more than 16384 slots"));
        }
        out.resize(out.len() + run, shard);
    }
    if out.len() != SLOT_COUNT {
        return Err(Error::new(format!(
            "the runs cover {} slots and there are {SLOT_COUNT}",
            out.len()
        )));
    }
    Ok(out)
}

/// The header at the front of a log segment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageHeader {
    /// Which shard owns the segment.
    pub shard: u32,
    /// The log address of the first payload byte.
    pub page_addr: u64,
    /// Payload bytes written, header excluded.
    pub used: u32,
    /// Of those, how many are no longer pointed at.
    pub dead_bytes: u32,
    /// The shard epoch when the page was opened.
    pub epoch: u32,
}

impl PageHeader {
    /// Decodes the first 32 bytes of a segment.
    ///
    /// The checksum covers the header only. The payload is not checksummed as a
    /// whole because that would turn every append into a read of the entire
    /// page, so a torn tail is found by walking records instead.
    ///
    /// # Errors
    ///
    /// If the buffer is short, the magic is wrong, or the header checksum
    /// fails.
    pub fn decode(page: &[u8]) -> Result<PageHeader> {
        if page.len() < PAGE_HEADER_LEN {
            return Err(Error::new("a page header is 32 bytes"));
        }
        let magic = u32_at(page, 0);
        if magic != PAGE_MAGIC {
            return Err(Error::new(format!(
                "not a log page: magic {magic:#010x}, wanted {PAGE_MAGIC:#010x}"
            )));
        }
        let want = u32_at(page, 28);
        let got = crc32c_skipping(&page[..PAGE_HEADER_LEN], 28);
        if want != got {
            return Err(Error::new(format!(
                "page header checksum mismatch: stored {want:#010x}, computed {got:#010x}"
            )));
        }
        Ok(PageHeader {
            shard: u32_at(page, 4),
            page_addr: u64_at(page, 8),
            used: u32_at(page, 16),
            dead_bytes: u32_at(page, 20),
            epoch: u32_at(page, 24),
        })
    }
}

/// A record, with its bytes copied out.
///
/// The engine hands back borrows into its own page buffer. This copies, because
/// the reader has no page buffer to lend and because being obviously right is
/// the entire job here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// Exact byte count, trailer included. The next record is [`Record::stride`]
    /// further on, not this much.
    pub len: u32,
    /// The raw kind byte, not an enum. A reader that refuses a kind it has not
    /// heard of refuses files written by anything newer than itself, and `07`
    /// section 9 says skip instead.
    pub kind: u8,
    /// See [`record_flags`].
    pub flags: u8,
    /// The previous address in this key's chain, or 0.
    pub prev: u64,
    /// Expiry, if the flag says there is one.
    pub ttl_ms: Option<u64>,
    /// The key.
    pub key: Vec<u8>,
    /// The value.
    pub value: Vec<u8>,
}

impl Record {
    /// How far the next record starts, which is `len` rounded up to eight.
    ///
    /// Walking by `len` desynchronises on the first record whose length is not
    /// already a multiple of eight, and then every record after it is garbage
    /// that happens to parse or does not.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.len.next_multiple_of(RECORD_ALIGN as u32) as usize
    }

    /// Whether this record deletes its key. Kind 7.
    #[must_use]
    pub const fn is_tombstone(&self) -> bool {
        self.kind == 7
    }
}

/// Parses the record at the front of `bytes`.
///
/// `Ok(None)` means the length field was zero, which is the end of the log
/// rather than a problem. A half written tail looks exactly the same, which is
/// the point: both mean stop here.
///
/// # Errors
///
/// If the checksum flag is clear, the length is impossible, the record runs
/// past what was given, or the trailer does not match.
pub fn parse_record(bytes: &[u8]) -> Result<Option<Record>> {
    if bytes.len() < 4 {
        return Ok(None);
    }
    let len = u32_at(bytes, 0) as usize;
    if len == 0 {
        return Ok(None);
    }
    let flags = bytes[5];
    if flags & record_flags::CHECKSUMMED == 0 {
        // This is not a record without a checksum, because there is no such
        // record. It is a flipped bit in the flags byte, and it has to be
        // caught by looking at the bit, because clearing this particular bit is
        // the one corruption a checksum cannot catch: it turns the checksum
        // off, and it moves the trailer boundary so those four bytes become
        // four bytes of value that the caller is then handed with confidence.
        return Err(Error::new(format!(
            "a record with its checksum flag clear: flags {flags:#04x}"
        )));
    }
    let h = if flags & record_flags::HAS_TTL != 0 {
        RECORD_HEADER_LEN_TTL
    } else {
        RECORD_HEADER_LEN
    };
    let t = RECORD_TRAILER_LEN;
    if bytes.len() < 8 {
        return Err(Error::new("a record shorter than its own length field"));
    }
    let klen = u16_at(bytes, 6) as usize;
    if len < h + klen + t {
        return Err(Error::new(format!(
            "the record is shorter than its own header: len {len}, header {h}, klen {klen}, trailer {t}"
        )));
    }
    if len > bytes.len() {
        return Err(Error::new(format!(
            "the record runs past the end of the page: len {len}, available {}",
            bytes.len()
        )));
    }
    let want = u32_at(bytes, len - t);
    let got = crc32c(0, &bytes[..len - t]);
    if want != got {
        return Err(Error::new(format!(
            "record checksum mismatch: stored {want:#010x}, computed {got:#010x}"
        )));
    }
    let ttl_ms = if flags & record_flags::HAS_TTL != 0 {
        Some(u64_at(bytes, 16))
    } else {
        None
    };
    Ok(Some(Record {
        len: len as u32,
        kind: bytes[4],
        flags,
        prev: u64_at(bytes, 8),
        ttl_ms,
        key: bytes[h..h + klen].to_vec(),
        value: bytes[h + klen..len - t].to_vec(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_length_is_the_end_and_not_an_error() {
        assert_eq!(parse_record(&[0, 0, 0, 0]).unwrap(), None);
        assert_eq!(parse_record(&[]).unwrap(), None);
        assert_eq!(parse_record(&[1, 2]).unwrap(), None);
    }

    #[test]
    fn the_checksum_flag_cannot_be_talked_out_of_itself() {
        let mut r = [0u8; 32];
        r[0] = 32;
        r[5] = 0;
        let e = parse_record(&r).unwrap_err();
        assert!(e.to_string().contains("checksum flag clear"), "{e}");
    }

    #[test]
    fn a_length_that_cannot_hold_the_header_is_refused() {
        let mut r = [0u8; 32];
        r[0] = 8;
        r[5] = record_flags::CHECKSUMMED;
        let e = parse_record(&r).unwrap_err();
        assert!(e.to_string().contains("shorter than its own header"), "{e}");
    }

    #[test]
    fn a_short_page_header_is_an_error_and_not_a_panic() {
        assert!(PageHeader::decode(&[0u8; 31]).is_err());
        assert!(PageHeader::decode(&[]).is_err());
    }

    #[test]
    fn a_short_superblock_is_an_error_and_not_a_panic() {
        assert!(Superblock::decode(&[0u8; 100]).is_err());
        assert!(Superblock::decode(&[]).is_err());
    }

    #[test]
    fn the_shard_table_has_to_cover_every_slot() {
        // One run of everything, one shard.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(SLOT_COUNT as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(decode_shard_table(&bytes, 1).unwrap().len(), SLOT_COUNT);

        // One run short.
        let mut short = Vec::new();
        short.extend_from_slice(&1000u16.to_le_bytes());
        short.extend_from_slice(&0u16.to_le_bytes());
        assert!(decode_shard_table(&short, 1).is_err());

        // A shard that is not there.
        let mut bad = Vec::new();
        bad.extend_from_slice(&(SLOT_COUNT as u16).to_le_bytes());
        bad.extend_from_slice(&9u16.to_le_bytes());
        assert!(decode_shard_table(&bad, 4).is_err());

        // A zero length run, which would otherwise loop forever in a decoder
        // written the obvious way.
        let mut zero = vec![0u8; 4];
        zero[2] = 0;
        assert!(decode_shard_table(&zero, 1).is_err());
    }

    #[test]
    fn a_ragged_shard_table_is_refused() {
        assert!(decode_shard_table(&[0u8; 5], 1).is_err());
    }

    #[test]
    fn checkpoint_addresses_have_to_go_forwards() {
        let mut buf = [0u8; CHECKPOINT_ENTRY_LEN];
        // tail below read_only.
        buf[24..32].copy_from_slice(&1u64.to_le_bytes());
        buf[16..24].copy_from_slice(&5u64.to_le_bytes());
        let crc = crc32c_skipping(&buf, 60);
        buf[60..64].copy_from_slice(&crc.to_le_bytes());
        let e = CheckpointEntry::decode(&buf).unwrap_err();
        assert!(e.to_string().contains("not in order"), "{e}");
    }

    #[test]
    fn stride_is_len_rounded_up() {
        let r = Record {
            len: 21,
            kind: 0,
            flags: record_flags::CHECKSUMMED,
            prev: 0,
            ttl_ms: None,
            key: Vec::new(),
            value: Vec::new(),
        };
        assert_eq!(r.stride(), 24);
    }
}
