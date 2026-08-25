//! The log page header.
//!
//! `07` section 4. Thirty two bytes at the front of every segment that holds
//! log records, and its checksum covers the header only.
//!
//! That last part is a decision rather than an oversight. A whole page checksum
//! would have to be recomputed on every append, which turns an append into a
//! read of the whole page, and the append is the commit (`06` section 3). Torn
//! tails are found by walking records instead: the `len == 0` sentinel says
//! where the writing stopped, and the per record trailer says whether what came
//! before it survived.

use crate::{checksum_skipping, get_u32, get_u64, put_u32, put_u64};
use yo_common::{Code, Error, Result};

/// The header, in bytes. Records start here.
pub const PAGE_HEADER_LEN: usize = 32;

/// `YOLG`, little endian, so it reads as those four characters in a hex dump.
pub const PAGE_MAGIC: u32 = 0x594F_4C47;

/// Where the checksum lives.
const CRC_OFFSET: usize = 28;

/// The header at the front of a log page.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageHeader {
    /// Which shard owns this page. Two shards never touch each other's pages,
    /// so this is a check rather than a lookup, and it is the check that catches
    /// a segment allocated to two shards at once.
    pub shard: u32,
    /// The log address of byte zero of the payload, meaning of byte 32 of the
    /// page. Log addresses are logically infinite, so this is what maps a
    /// physical segment back to a position in the log.
    pub page_addr: u64,
    /// Bytes of the payload that have been written, header excluded.
    pub used: u32,
    /// Of those, how many belong to records the index no longer points at.
    /// Compaction reads this and nothing else to decide what to do (`06`
    /// section 5).
    pub dead_bytes: u32,
    /// The epoch the page was last written in.
    pub epoch: u32,
}

impl PageHeader {
    /// Writes the header and its checksum into the first 32 bytes of `page`.
    ///
    /// # Panics
    ///
    /// If `page` is shorter than [`PAGE_HEADER_LEN`].
    pub fn encode(&self, page: &mut [u8]) {
        assert!(page.len() >= PAGE_HEADER_LEN, "a page holds its own header");
        put_u32(page, 0, PAGE_MAGIC);
        put_u32(page, 4, self.shard);
        put_u64(page, 8, self.page_addr);
        put_u32(page, 16, self.used);
        put_u32(page, 20, self.dead_bytes);
        put_u32(page, 24, self.epoch);
        let crc = checksum_skipping(&page[..CRC_OFFSET + 4], CRC_OFFSET);
        put_u32(page, CRC_OFFSET, crc);
    }

    /// Reads the header back, checking the magic and the checksum.
    pub fn decode(page: &[u8]) -> Result<PageHeader> {
        if page.len() < PAGE_HEADER_LEN {
            return Err(Error::new(Code::Invalid, "shorter than a page header"));
        }
        let magic = get_u32(page, 0);
        if magic != PAGE_MAGIC {
            return Err(Error::new(Code::Corrupt, "this segment is not a log page")
                .with_detail(format!("magic={magic:#010x} want={PAGE_MAGIC:#010x}")));
        }
        let want = get_u32(page, CRC_OFFSET);
        let got = checksum_skipping(&page[..CRC_OFFSET + 4], CRC_OFFSET);
        if want != got {
            return Err(
                Error::new(Code::Corrupt, "log page header checksum mismatch")
                    .with_detail(format!("stored={want:#010x} computed={got:#010x}")),
            );
        }
        let h = PageHeader {
            shard: get_u32(page, 4),
            page_addr: get_u64(page, 8),
            used: get_u32(page, 16),
            dead_bytes: get_u32(page, 20),
            epoch: get_u32(page, 24),
        };
        if h.dead_bytes > h.used {
            return Err(
                Error::new(Code::Corrupt, "more dead bytes than written bytes")
                    .with_detail(format!("dead_bytes={} used={}", h.dead_bytes, h.used)),
            );
        }
        Ok(h)
    }

    /// The fraction of this page that is dead, between 0 and 1.
    ///
    /// An empty page is 0 rather than a division by zero, because an empty page
    /// is not worth compacting and that is the only question this answers.
    #[must_use]
    pub fn dead_fraction(&self) -> f64 {
        if self.used == 0 {
            return 0.0;
        }
        f64::from(self.dead_bytes) / f64::from(self.used)
    }

    /// Whether this page is worth compacting, at `06` section 5's threshold.
    #[must_use]
    pub fn wants_compaction(&self) -> bool {
        self.dead_fraction() > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_PAGE_SIZE;

    fn a_page() -> Vec<u8> {
        let mut page = vec![0u8; DEFAULT_PAGE_SIZE as usize];
        PageHeader {
            shard: 3,
            page_addr: 32 * 1024 * 1024,
            used: 4096,
            dead_bytes: 1024,
            epoch: 9,
        }
        .encode(&mut page);
        page
    }

    #[test]
    fn the_magic_reads_as_yolg_in_a_hex_dump() {
        let page = a_page();
        assert_eq!(&page[..4], b"GLOY", "little endian, so reversed on disk");
        assert_eq!(PAGE_MAGIC.to_be_bytes(), *b"YOLG");
    }

    #[test]
    fn a_page_header_round_trips() {
        let page = a_page();
        let h = PageHeader::decode(&page).unwrap();
        assert_eq!(h.shard, 3);
        assert_eq!(h.page_addr, 32 * 1024 * 1024);
        assert_eq!(h.used, 4096);
        assert_eq!(h.dead_bytes, 1024);
        assert_eq!(h.epoch, 9);
    }

    #[test]
    fn every_field_lands_where_the_specification_says() {
        let page = a_page();
        assert_eq!(get_u32(&page, 0), PAGE_MAGIC);
        assert_eq!(get_u32(&page, 4), 3);
        assert_eq!(get_u64(&page, 8), 32 * 1024 * 1024);
        assert_eq!(get_u32(&page, 16), 4096);
        assert_eq!(get_u32(&page, 20), 1024);
        assert_eq!(get_u32(&page, 24), 9);
    }

    #[test]
    fn the_checksum_covers_the_header_and_stops_there() {
        let mut page = a_page();
        // Writing a record does not invalidate the header, which is the whole
        // reason the checksum is scoped this way.
        page[PAGE_HEADER_LEN] = 0xff;
        page[9000] = 0xff;
        assert!(PageHeader::decode(&page).is_ok());

        for i in 0..PAGE_HEADER_LEN {
            let mut bad = page.clone();
            bad[i] ^= 0x10;
            assert!(PageHeader::decode(&bad).is_err(), "byte {i} was not caught");
        }
    }

    #[test]
    fn a_segment_that_is_not_a_log_page_says_so() {
        let mut page = a_page();
        put_u32(&mut page, 0, 0x1234_5678);
        let err = PageHeader::decode(&page).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("0x12345678"));
    }

    #[test]
    fn more_dead_than_written_is_refused() {
        let mut page = vec![0u8; 64];
        PageHeader {
            used: 100,
            dead_bytes: 200,
            ..PageHeader::default()
        }
        .encode(&mut page);
        let err = PageHeader::decode(&page).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
    }

    #[test]
    fn the_compaction_trigger_is_a_dead_fraction_over_a_half() {
        let mut h = PageHeader {
            used: 1000,
            ..PageHeader::default()
        };
        h.dead_bytes = 500;
        assert!(!h.wants_compaction(), "exactly a half does not trigger");
        h.dead_bytes = 501;
        assert!(h.wants_compaction());

        let empty = PageHeader::default();
        assert_eq!(empty.dead_fraction(), 0.0);
        assert!(!empty.wants_compaction());
    }

    #[test]
    fn a_short_buffer_is_an_error_and_not_a_panic() {
        assert_eq!(
            PageHeader::decode(&[0u8; 8]).unwrap_err().code(),
            Code::Invalid
        );
    }
}
