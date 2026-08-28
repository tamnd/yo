//! The superblock, its two slots, the run length encoded shard table and the
//! per shard checkpoint entries.
//!
//! `07` sections 2 and 3. The superblock is the only thing in the file that is
//! overwritten in place, which is why there are two of them and why the choice
//! between them is a sequence number guarded by a checksum. Everything else in
//! a `.yo` file is append only.
//!
//! One thing worth stating because it is easy to get wrong: the slot checksum
//! covers all 16380 bytes before it, which means the shard table and the
//! checkpoint entries are inside it. So the order is always write the header,
//! write the table, write the entries, then [`seal`], and only then hand the
//! 16 KiB to the kernel.

use crate::{
    DATA_START, FORMAT_VERSION, MAGIC, MIN_READER_VERSION, SUPERBLOCK_LEN, checksum_skipping,
    get_u16, get_u32, get_u64, is_legal_page_size, put_u16, put_u32, put_u64,
};
use yo_common::{Code, Error, Result, SLOT_COUNT};

/// Where the checksum lives, and therefore how much of the slot it covers.
pub const CRC_OFFSET: usize = 16380;

/// The default offset of the shard table, which is the end of the header.
pub const DEFAULT_SHARD_TABLE_OFF: u32 = 160;

/// A per shard checkpoint entry, in bytes.
pub const CHECKPOINT_ENTRY_LEN: usize = 64;

/// The `flags` word at offset 84.
pub mod superblock_flags {
    /// The database was closed cleanly, so there is nothing to replay.
    ///
    /// This bit plus the absence of both sidecars is what makes a plain `cp` of
    /// the file a valid backup (`07` section 6).
    pub const CLEAN_SHUTDOWN: u32 = 1 << 0;
    /// The segments are encrypted.
    pub const ENCRYPTED: u32 = 1 << 1;
    /// `archival_root` points at something.
    pub const HAS_ARCHIVAL: u32 = 1 << 2;
    /// The larger than memory path is engaged, so some values live only on disk.
    pub const TIERING_ENGAGED: u32 = 1 << 3;
}

/// The header half of a superblock slot, decoded.
///
/// The shard table and the checkpoint entries are not in here on purpose. They
/// are variable length, they live further into the same 16 KiB, and a caller
/// that only wants to know the page size should not pay to decode 16384 slot
/// assignments to find out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superblock {
    /// The format this slot was written by.
    pub format_version: u32,
    /// The lowest reader version that may read it.
    pub min_reader_version: u32,
    /// Segment size, fixed at creation.
    pub page_size: u32,
    /// Shard count, fixed at creation.
    pub shard_count: u32,
    /// Monotonic. The higher of the two valid slots wins.
    pub seq: u64,
    /// Segments times `page_size` plus [`DATA_START`].
    pub file_size: u64,
    /// Identifies this database across copies of the file.
    pub file_uuid: [u8; 16],
    /// When the file was created.
    pub created_unix_ms: u64,
    /// When this checkpoint was taken.
    pub checkpoint_unix_ms: u64,
    /// Logical databases, default 16.
    pub db_count: u32,
    /// See [`superblock_flags`].
    pub flags: u32,
    /// The replication id, high 64 bits of the 40 hex characters.
    pub replid_hi: u64,
    /// The replication id, next 64 bits.
    pub replid_lo: u64,
    /// The replication id, last 32 bits.
    pub replid_ext: u32,
    /// Offset within the slot of the run length encoded shard table.
    pub shard_table_off: u32,
    /// Length in bytes of that encoding.
    pub shard_table_len: u16,
    /// Address of the collection catalogue, or 0 if there are no collections.
    pub catalog_addr: u64,
    /// Address of the free segment list.
    pub free_list_addr: u64,
    /// Address of the archival root, or 0.
    pub archival_root: u64,
}

impl Default for Superblock {
    fn default() -> Self {
        Superblock {
            format_version: FORMAT_VERSION,
            min_reader_version: MIN_READER_VERSION,
            page_size: crate::DEFAULT_PAGE_SIZE,
            shard_count: 1,
            seq: 0,
            file_size: DATA_START,
            file_uuid: [0; 16],
            created_unix_ms: 0,
            checkpoint_unix_ms: 0,
            db_count: 16,
            flags: 0,
            replid_hi: 0,
            replid_lo: 0,
            replid_ext: 0,
            shard_table_off: DEFAULT_SHARD_TABLE_OFF,
            shard_table_len: 0,
            catalog_addr: 0,
            free_list_addr: 0,
            archival_root: 0,
        }
    }
}

impl Superblock {
    /// Writes the header fields into the first 160 bytes of `slot`.
    ///
    /// Leaves the checksum alone, because the bytes it covers have not all been
    /// written yet. Call [`seal`] once the table and the entries are in.
    ///
    /// # Panics
    ///
    /// If `slot` is not exactly [`SUPERBLOCK_LEN`] bytes. That is a programmer
    /// error rather than a corrupt file, so it is not an `Err`.
    pub fn encode(&self, slot: &mut [u8]) {
        assert_eq!(slot.len(), SUPERBLOCK_LEN, "a superblock slot is 16 KiB");
        slot[..16].copy_from_slice(&MAGIC);
        put_u32(slot, 16, self.format_version);
        put_u32(slot, 20, self.min_reader_version);
        put_u32(slot, 24, self.page_size);
        put_u32(slot, 28, self.shard_count);
        put_u64(slot, 32, self.seq);
        put_u64(slot, 40, self.file_size);
        slot[48..64].copy_from_slice(&self.file_uuid);
        put_u64(slot, 64, self.created_unix_ms);
        put_u64(slot, 72, self.checkpoint_unix_ms);
        put_u32(slot, 80, self.db_count);
        put_u32(slot, 84, self.flags);
        put_u64(slot, 88, self.replid_hi);
        put_u64(slot, 96, self.replid_lo);
        put_u32(slot, 104, self.replid_ext);
        put_u32(slot, 108, self.shard_table_off);
        put_u64(slot, 112, self.catalog_addr);
        put_u64(slot, 120, self.free_list_addr);
        put_u64(slot, 128, self.archival_root);
        // 136 to 156 is reserved and must be zero. The table in `07` section 2
        // says the reserved run is 24 bytes, and the paragraph under it says
        // `shard_table_len` is the two bytes at 156. The paragraph wins, so the
        // run is 20 bytes and 158 to 160 is padding.
        slot[136..156].fill(0);
        put_u16(slot, 156, self.shard_table_len);
        put_u16(slot, 158, 0);
    }

    /// Reads the header fields, checking everything that makes the file ours.
    ///
    /// In order: the length, the magic, the checksum, then the version. The
    /// checksum comes before the version check so that a corrupt slot is
    /// reported as corrupt rather than as a version from the future, which is
    /// what happens if a flipped bit lands in the version field.
    pub fn decode(slot: &[u8]) -> Result<Superblock> {
        if slot.len() != SUPERBLOCK_LEN {
            return Err(Error::new(
                Code::Invalid,
                "a superblock slot is 16384 bytes",
            ));
        }
        if slot[..16] != MAGIC {
            return Err(Error::new(
                Code::Invalid,
                "not a .yo file: the first sixteen bytes are not the magic",
            ));
        }
        let want = get_u32(slot, CRC_OFFSET);
        let got = checksum_skipping(slot, CRC_OFFSET);
        if want != got {
            return Err(Error::new(Code::Corrupt, "superblock checksum mismatch")
                .with_detail(format!("stored={want:#010x} computed={got:#010x}")));
        }

        let min_reader_version = get_u32(slot, 20);
        if min_reader_version > FORMAT_VERSION {
            return Err(
                Error::new(Code::VersionTooNew, "this file needs a newer yo to read it")
                    .with_detail(format!(
                        "file_min_reader={min_reader_version} this_reader={FORMAT_VERSION}"
                    )),
            );
        }

        let page_size = get_u32(slot, 24);
        if !is_legal_page_size(page_size) {
            return Err(
                Error::new(Code::Corrupt, "the segment size is not a legal value")
                    .with_detail(format!("page_size={page_size}")),
            );
        }
        let shard_count = get_u32(slot, 28);
        if shard_count == 0 {
            return Err(Error::new(Code::Corrupt, "a file with no shards"));
        }

        let mut file_uuid = [0u8; 16];
        file_uuid.copy_from_slice(&slot[48..64]);

        Ok(Superblock {
            format_version: get_u32(slot, 16),
            min_reader_version,
            page_size,
            shard_count,
            seq: get_u64(slot, 32),
            file_size: get_u64(slot, 40),
            file_uuid,
            created_unix_ms: get_u64(slot, 64),
            checkpoint_unix_ms: get_u64(slot, 72),
            db_count: get_u32(slot, 80),
            flags: get_u32(slot, 84),
            replid_hi: get_u64(slot, 88),
            replid_lo: get_u64(slot, 96),
            replid_ext: get_u32(slot, 104),
            shard_table_off: get_u32(slot, 108),
            shard_table_len: get_u16(slot, 156),
            catalog_addr: get_u64(slot, 112),
            free_list_addr: get_u64(slot, 120),
            archival_root: get_u64(slot, 128),
        })
    }

    /// Where the checkpoint entries begin, which is right after the table.
    #[must_use]
    pub const fn checkpoint_off(&self) -> usize {
        self.shard_table_off as usize + self.shard_table_len as usize
    }

    /// Whether every checkpoint entry fits inside the slot.
    ///
    /// A file whose shard count and table length do not leave room for the
    /// entries is corrupt in a way that would otherwise show up as a checkpoint
    /// full of zeroes, which reads as a valid empty database.
    #[must_use]
    pub fn checkpoints_fit(&self) -> bool {
        let end = self
            .checkpoint_off()
            .saturating_add(self.shard_count as usize * CHECKPOINT_ENTRY_LEN);
        self.shard_table_off as usize >= 160 && end <= CRC_OFFSET
    }
}

/// Computes the slot checksum, stores it, and returns it.
///
/// # Panics
///
/// If `slot` is not exactly [`SUPERBLOCK_LEN`] bytes.
pub fn seal(slot: &mut [u8]) -> u32 {
    assert_eq!(slot.len(), SUPERBLOCK_LEN, "a superblock slot is 16 KiB");
    let crc = checksum_skipping(slot, CRC_OFFSET);
    put_u32(slot, CRC_OFFSET, crc);
    crc
}

/// Picks the live slot out of the two.
///
/// Returns the slot index, 0 or 1, and its decoded header. Higher `seq` wins,
/// and a slot that fails to decode does not get a vote. Both failing is the one
/// case that is not recoverable here, and the two errors are returned so that
/// `yo check` can print them both instead of guessing which one the user cares
/// about.
///
/// # Errors
///
/// Returns the slot A error if neither slot decodes. The slot B error is in the
/// detail field.
pub fn pick(a: &[u8], b: &[u8]) -> Result<(usize, Superblock)> {
    match (Superblock::decode(a), Superblock::decode(b)) {
        (Ok(sa), Ok(sb)) => {
            // Equal sequence numbers should not happen, and if they do the file
            // was written by something that is not us. Taking A is arbitrary but
            // it is at least deterministic, which matters more here than being
            // right, because both slots claim to be the same checkpoint.
            if sb.seq > sa.seq {
                Ok((1, sb))
            } else {
                Ok((0, sa))
            }
        }
        (Ok(sa), Err(_)) => Ok((0, sa)),
        (Err(_), Ok(sb)) => Ok((1, sb)),
        (Err(ea), Err(eb)) => Err(ea.with_detail(format!("slot B: {eb}"))),
    }
}

/// One shard's position in its log, as of the last checkpoint.
///
/// Sixty four bytes, one cache line, which is not an accident: recovery reads
/// every one of these and nothing else before it starts replaying.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointEntry {
    /// The oldest address still in the file.
    pub log_begin: u64,
    /// The boundary between the stable region and the read only region.
    pub log_head: u64,
    /// The boundary between the read only region and the mutable region.
    pub log_read_only: u64,
    /// Where the next append goes, and where replay starts.
    pub log_tail: u64,
    /// Where the checkpointed index image lives.
    pub index_image_addr: u64,
    /// How long it is.
    pub index_image_len: u64,
    /// Keys in this shard at the checkpoint, for `INFO` and for a sanity check
    /// after replay.
    pub key_count: u64,
    /// The shard's epoch at the checkpoint.
    pub epoch: u32,
}

impl CheckpointEntry {
    /// Writes the entry and its checksum into 64 bytes.
    ///
    /// # Panics
    ///
    /// If `buf` is not exactly [`CHECKPOINT_ENTRY_LEN`] bytes.
    pub fn encode(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), CHECKPOINT_ENTRY_LEN, "an entry is 64 bytes");
        put_u64(buf, 0, self.log_begin);
        put_u64(buf, 8, self.log_head);
        put_u64(buf, 16, self.log_read_only);
        put_u64(buf, 24, self.log_tail);
        put_u64(buf, 32, self.index_image_addr);
        put_u64(buf, 40, self.index_image_len);
        put_u64(buf, 48, self.key_count);
        put_u32(buf, 56, self.epoch);
        let crc = checksum_skipping(buf, 60);
        put_u32(buf, 60, crc);
    }

    /// Reads one entry back, checking its checksum and its ordering.
    ///
    /// The ordering check is the useful one. Four addresses that are not
    /// monotonic describe a log with a negative sized region, and replaying
    /// from that produces a plausible looking database out of arbitrary bytes.
    pub fn decode(buf: &[u8]) -> Result<CheckpointEntry> {
        if buf.len() != CHECKPOINT_ENTRY_LEN {
            return Err(Error::new(Code::Invalid, "a checkpoint entry is 64 bytes"));
        }
        let want = get_u32(buf, 60);
        let got = checksum_skipping(buf, 60);
        if want != got {
            return Err(Error::new(
                Code::Corrupt,
                "checkpoint entry checksum mismatch",
            ));
        }
        let e = CheckpointEntry {
            log_begin: get_u64(buf, 0),
            log_head: get_u64(buf, 8),
            log_read_only: get_u64(buf, 16),
            log_tail: get_u64(buf, 24),
            index_image_addr: get_u64(buf, 32),
            index_image_len: get_u64(buf, 40),
            key_count: get_u64(buf, 48),
            epoch: get_u32(buf, 56),
        };
        if !e.addresses_are_ordered() {
            return Err(
                Error::new(Code::Corrupt, "the four log addresses are not in order").with_detail(
                    format!(
                        "begin={} head={} read_only={} tail={}",
                        e.log_begin, e.log_head, e.log_read_only, e.log_tail
                    ),
                ),
            );
        }
        Ok(e)
    }

    /// `begin <= head <= read_only <= tail`, which `06` section 2 requires.
    #[must_use]
    pub const fn addresses_are_ordered(&self) -> bool {
        self.log_begin <= self.log_head
            && self.log_head <= self.log_read_only
            && self.log_read_only <= self.log_tail
    }
}

/// Run length encodes a slot to shard mapping into `out`.
///
/// Returns the number of bytes written. The encoding is a sequence of four byte
/// pairs, each a `u16` run length followed by a `u16` shard id. The
/// uncompressed form is 32 KiB of `u16`, which does not fit in a 16 KiB slot at
/// all, and the compressed form is four bytes per contiguous range. Slot to
/// shard assignment is contiguous by construction, so a sixty four shard
/// database spends 256 bytes here.
///
/// # Errors
///
/// [`Code::Full`] if the encoding does not fit in `out`, and [`Code::Invalid`]
/// if `slot_shard` is not exactly [`SLOT_COUNT`] entries.
pub fn encode_shard_table(slot_shard: &[u16], out: &mut [u8]) -> Result<usize> {
    if slot_shard.len() != SLOT_COUNT as usize {
        return Err(Error::new(
            Code::Invalid,
            "the shard table has one entry per Redis slot",
        )
        .with_detail(format!("got={} want={SLOT_COUNT}", slot_shard.len())));
    }
    let mut n = 0usize;
    let mut i = 0usize;
    while i < slot_shard.len() {
        let shard = slot_shard[i];
        let mut run = 1usize;
        while i + run < slot_shard.len() && slot_shard[i + run] == shard {
            run += 1;
        }
        // A run cannot exceed 16384 and a u16 holds 65535, so one pair always
        // covers a whole run and there is no splitting to think about.
        if n + 4 > out.len() {
            return Err(Error::new(
                Code::Full,
                "the shard table does not fit in the superblock",
            ));
        }
        put_u16(out, n, run as u16);
        put_u16(out, n + 2, shard);
        n += 4;
        i += run;
    }
    Ok(n)
}

/// Expands a run length encoded shard table into `out`.
///
/// # Errors
///
/// [`Code::Corrupt`] if the runs do not add up to exactly [`SLOT_COUNT`], if a
/// run is zero long, or if a shard id is at or above `shard_count`. All three
/// are things a flipped bit produces and all three would otherwise route a key
/// to a shard that does not exist.
pub fn decode_shard_table(bytes: &[u8], shard_count: u32, out: &mut [u16]) -> Result<()> {
    if out.len() != SLOT_COUNT as usize {
        return Err(Error::new(
            Code::Invalid,
            "the output needs one entry per Redis slot",
        ));
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::new(
            Code::Corrupt,
            "the shard table is not a whole number of runs",
        ));
    }
    let mut filled = 0usize;
    for pair in bytes.as_chunks::<4>().0 {
        let run = u16::from_le_bytes([pair[0], pair[1]]) as usize;
        let shard = u16::from_le_bytes([pair[2], pair[3]]);
        if run == 0 {
            return Err(Error::new(Code::Corrupt, "a zero length run"));
        }
        if u32::from(shard) >= shard_count {
            return Err(
                Error::new(Code::Corrupt, "a slot points at a shard that is not there")
                    .with_detail(format!("shard={shard} shard_count={shard_count}")),
            );
        }
        if filled + run > out.len() {
            return Err(Error::new(
                Code::Corrupt,
                "the runs cover more than 16384 slots",
            ));
        }
        out[filled..filled + run].fill(shard);
        filled += run;
    }
    if filled != out.len() {
        return Err(
            Error::new(Code::Corrupt, "the runs do not cover every slot")
                .with_detail(format!("covered={filled} want={SLOT_COUNT}")),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_PAGE_SIZE;

    // How far apart the bit flips in `a_flipped_bit_anywhere_in_the_slot_is_caught`
    // are. A prime, so the stride does not fall into step with the four byte
    // fields and keep landing on the same byte of each.
    #[cfg(miri)]
    const STEP: usize = 2039;
    #[cfg(not(miri))]
    const STEP: usize = 1;

    // The offsets a stride has no business missing: the ends of the magic, the
    // version, the page size, the checksum's own four bytes, and the last byte
    // of the slot.
    #[cfg(miri)]
    const BY_HAND: [usize; 9] = [
        0,
        15,
        16,
        20,
        24,
        CRC_OFFSET,
        CRC_OFFSET + 3,
        16382,
        SUPERBLOCK_LEN - 1,
    ];
    #[cfg(not(miri))]
    const BY_HAND: [usize; 0] = [];

    fn a_slot() -> ([u8; SUPERBLOCK_LEN], Superblock) {
        let sb = Superblock {
            page_size: DEFAULT_PAGE_SIZE,
            shard_count: 8,
            seq: 42,
            file_size: DATA_START + 2048 * u64::from(DEFAULT_PAGE_SIZE),
            file_uuid: [0xab; 16],
            created_unix_ms: 1_700_000_000_000,
            checkpoint_unix_ms: 1_700_000_001_000,
            flags: superblock_flags::CLEAN_SHUTDOWN,
            replid_hi: 0x0123_4567_89ab_cdef,
            replid_lo: 0xfedc_ba98_7654_3210,
            replid_ext: 0xdead_beef,
            shard_table_len: 4,
            catalog_addr: 32768,
            free_list_addr: 65536,
            ..Superblock::default()
        };
        let mut buf = [0u8; SUPERBLOCK_LEN];
        sb.encode(&mut buf);
        seal(&mut buf);
        (buf, sb)
    }

    #[test]
    fn a_superblock_round_trips_field_for_field() {
        let (buf, sb) = a_slot();
        assert_eq!(Superblock::decode(&buf).unwrap(), sb);
    }

    #[test]
    fn every_field_lands_at_the_offset_the_specification_names() {
        let (buf, sb) = a_slot();
        assert_eq!(&buf[..16], &MAGIC);
        assert_eq!(get_u32(&buf, 16), sb.format_version);
        assert_eq!(get_u32(&buf, 20), sb.min_reader_version);
        assert_eq!(get_u32(&buf, 24), sb.page_size);
        assert_eq!(get_u32(&buf, 28), sb.shard_count);
        assert_eq!(get_u64(&buf, 32), sb.seq);
        assert_eq!(get_u64(&buf, 40), sb.file_size);
        assert_eq!(&buf[48..64], &sb.file_uuid);
        assert_eq!(get_u64(&buf, 64), sb.created_unix_ms);
        assert_eq!(get_u64(&buf, 72), sb.checkpoint_unix_ms);
        assert_eq!(get_u32(&buf, 80), sb.db_count);
        assert_eq!(get_u32(&buf, 84), sb.flags);
        assert_eq!(get_u64(&buf, 88), sb.replid_hi);
        assert_eq!(get_u64(&buf, 96), sb.replid_lo);
        assert_eq!(get_u32(&buf, 104), sb.replid_ext);
        assert_eq!(get_u32(&buf, 108), sb.shard_table_off);
        assert_eq!(get_u64(&buf, 112), sb.catalog_addr);
        assert_eq!(get_u64(&buf, 120), sb.free_list_addr);
        assert_eq!(get_u64(&buf, 128), sb.archival_root);
        assert!(buf[136..156].iter().all(|&b| b == 0), "reserved is zero");
        assert_eq!(get_u16(&buf, 156), sb.shard_table_len);
        assert_eq!(
            get_u32(&buf, CRC_OFFSET),
            checksum_skipping(&buf, CRC_OFFSET)
        );
    }

    #[test]
    fn a_flipped_bit_anywhere_in_the_slot_is_caught() {
        let (good, _) = a_slot();
        // Every byte, not a sample. Sixteen thousand checksums is a few
        // milliseconds and it is the difference between believing the checksum
        // covers the whole slot and knowing it.
        //
        // Under Miri each of those iterations copies and then checksums 16 KiB
        // interpreted rather than executed, and sixteen thousand of them do not
        // finish inside a CI job's six hour ceiling. So Miri walks a stride and
        // visits the offsets that mean something by hand. What Miri is here for
        // is whether this code has undefined behaviour, and a stride runs every
        // line of it. The claim that the checksum covers all 16 KiB is still
        // checked on every ordinary run, which is where it belongs.
        for i in (0..SUPERBLOCK_LEN).step_by(STEP).chain(BY_HAND) {
            let mut bad = good;
            bad[i] ^= 0x40;
            let err = Superblock::decode(&bad).unwrap_err();
            // A hit inside the magic is reported as "not a .yo file" and a hit
            // anywhere else as corruption. Either way it does not decode.
            assert!(
                matches!(err.code(), Code::Corrupt | Code::Invalid),
                "byte {i} was not caught: {err}"
            );
        }
    }

    #[test]
    fn a_file_from_the_future_is_refused_by_name() {
        let (mut buf, _) = a_slot();
        put_u32(&mut buf, 20, FORMAT_VERSION + 1);
        seal(&mut buf);
        let err = Superblock::decode(&buf).unwrap_err();
        assert_eq!(err.code(), Code::VersionTooNew);
        assert!(err.detail().unwrap().contains("file_min_reader=2"));
    }

    #[test]
    fn corruption_is_reported_before_the_version_is_believed() {
        // A bit flip that lands in min_reader_version must not come back as
        // VersionTooNew, because that sends the user looking for a release that
        // does not exist.
        let (mut buf, _) = a_slot();
        put_u32(&mut buf, 20, 0x4000_0000);
        let err = Superblock::decode(&buf).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
    }

    #[test]
    fn a_nonsense_page_size_is_refused_even_with_a_good_checksum() {
        let (mut buf, _) = a_slot();
        put_u32(&mut buf, 24, 12288);
        seal(&mut buf);
        let err = Superblock::decode(&buf).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("12288"));
    }

    #[test]
    fn the_higher_sequence_number_wins_and_a_bad_slot_does_not_vote() {
        let (mut a, _) = a_slot();
        let mut b = a;
        put_u64(&mut b, 32, 43);
        seal(&mut b);
        assert_eq!(pick(&a, &b).unwrap().0, 1, "B is newer");
        assert_eq!(pick(&b, &a).unwrap().0, 0, "B is still newer");

        // Tear slot B. A wins even though its sequence number is lower, which is
        // the whole point of writing them alternately.
        b[9000] ^= 0xff;
        assert_eq!(pick(&a, &b).unwrap().0, 0);

        // Tear both and the file is gone, with both reasons in one error.
        a[9000] ^= 0xff;
        let err = pick(&a, &b).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("slot B"));
    }

    #[test]
    fn equal_sequence_numbers_pick_a_deterministically() {
        let (a, _) = a_slot();
        let b = a;
        assert_eq!(pick(&a, &b).unwrap().0, 0);
    }

    #[test]
    fn a_checkpoint_entry_round_trips_and_fills_one_cache_line() {
        assert_eq!(CHECKPOINT_ENTRY_LEN, yo_common::CACHE_LINE);
        let e = CheckpointEntry {
            log_begin: 0,
            log_head: 1 << 20,
            log_read_only: 2 << 20,
            log_tail: 3 << 20,
            index_image_addr: 1 << 30,
            index_image_len: 4096,
            key_count: 1_000_000,
            epoch: 17,
        };
        let mut buf = [0u8; CHECKPOINT_ENTRY_LEN];
        e.encode(&mut buf);
        assert_eq!(CheckpointEntry::decode(&buf).unwrap(), e);
    }

    #[test]
    fn out_of_order_log_addresses_are_corruption_not_an_empty_log() {
        let e = CheckpointEntry {
            log_begin: 100,
            log_head: 50,
            log_read_only: 200,
            log_tail: 300,
            ..CheckpointEntry::default()
        };
        let mut buf = [0u8; CHECKPOINT_ENTRY_LEN];
        e.encode(&mut buf);
        let err = CheckpointEntry::decode(&buf).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("head=50"));
    }

    #[test]
    fn a_flipped_bit_in_a_checkpoint_entry_is_caught() {
        let e = CheckpointEntry {
            log_tail: 1 << 40,
            ..CheckpointEntry::default()
        };
        let mut good = [0u8; CHECKPOINT_ENTRY_LEN];
        e.encode(&mut good);
        for i in 0..CHECKPOINT_ENTRY_LEN {
            let mut bad = good;
            bad[i] ^= 1;
            assert!(
                CheckpointEntry::decode(&bad).is_err(),
                "byte {i} was not caught"
            );
        }
    }

    #[test]
    fn a_contiguous_shard_table_costs_four_bytes_per_shard() {
        let shards = 64u16;
        let per = SLOT_COUNT as usize / shards as usize;
        let table: Vec<u16> = (0..SLOT_COUNT as usize).map(|s| (s / per) as u16).collect();

        let mut out = [0u8; 1024];
        let n = encode_shard_table(&table, &mut out).unwrap();
        assert_eq!(n, 4 * shards as usize);

        let mut back = vec![0u16; SLOT_COUNT as usize];
        decode_shard_table(&out[..n], u32::from(shards), &mut back).unwrap();
        assert_eq!(back, table);
    }

    #[test]
    fn one_shard_is_one_run() {
        let table = vec![0u16; SLOT_COUNT as usize];
        let mut out = [0u8; 64];
        let n = encode_shard_table(&table, &mut out).unwrap();
        assert_eq!(n, 4);
        assert_eq!(get_u16(&out, 0), SLOT_COUNT, "one run of every slot");
    }

    #[test]
    fn the_worst_case_table_is_refused_rather_than_truncated() {
        // Alternating shards is the pathological input: every slot is its own
        // run, so the encoding is 64 KiB and cannot fit. It should say so.
        let table: Vec<u16> = (0..SLOT_COUNT).map(|s| s % 2).collect();
        let mut out = [0u8; 1024];
        let err = encode_shard_table(&table, &mut out).unwrap_err();
        assert_eq!(err.code(), Code::Full);
    }

    #[test]
    fn a_table_that_does_not_cover_every_slot_is_corruption() {
        let bytes = [0x00u8, 0x40, 0x00, 0x00]; // one run of 16384 slots
        let mut out = vec![0u16; SLOT_COUNT as usize];
        decode_shard_table(&bytes, 1, &mut out).unwrap();

        let short = [0x10u8, 0x00, 0x00, 0x00]; // one run of 16 slots
        let err = decode_shard_table(&short, 1, &mut out).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("covered=16"));
    }

    #[test]
    fn a_slot_pointing_at_a_shard_that_is_not_there_is_corruption() {
        let bytes = [0x00u8, 0x40, 0x09, 0x00]; // every slot to shard 9
        let mut out = vec![0u16; SLOT_COUNT as usize];
        let err = decode_shard_table(&bytes, 8, &mut out).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("shard=9"));
    }

    #[test]
    fn a_zero_length_run_is_corruption_and_not_an_infinite_loop() {
        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let mut out = vec![0u16; SLOT_COUNT as usize];
        assert_eq!(
            decode_shard_table(&bytes, 1, &mut out).unwrap_err().code(),
            Code::Corrupt
        );
    }

    #[test]
    fn the_checkpoint_entries_have_to_fit_where_the_header_says_they_do() {
        let sb = Superblock {
            shard_count: 8,
            shard_table_len: 32,
            ..Superblock::default()
        };
        assert_eq!(sb.checkpoint_off(), 192);
        assert!(sb.checkpoints_fit());

        // 253 shards of 64 bytes each, starting at 192, ends at 16384, which is
        // past the checksum.
        let too_many = Superblock {
            shard_count: 253,
            ..sb.clone()
        };
        assert!(!too_many.checkpoints_fit());

        let overlapping = Superblock {
            shard_table_off: 8,
            ..sb
        };
        assert!(!overlapping.checkpoints_fit(), "the table is in the header");
    }
}
