//! The `.yo` file: two superblock slots, the root flip, and where regions live.
//!
//! `07` sections 2, 3 and 5. The layout is
//!
//! ```text
//! 0      superblock slot A, 16 KiB
//! 16384  superblock slot B, 16 KiB
//! 32768  region 0, 32 MiB
//! ...    region 1, 32 MiB
//! ```
//!
//! A region is one log page, which is [`LOG_PAGE_LEN`] bytes, which is a whole
//! number of segments. Nothing outside the two slots is ever overwritten with
//! different bytes, so the only thing in the file that can be caught half
//! written is a slot, and that is what the second slot is for.
//!
//! **Every region says who it belongs to.** A region begins with a page header
//! carrying `shard` and `page_addr`, so opening a file means reading 32 bytes
//! per region and nothing else. A ten gigabyte file has around 320 of them,
//! which is 320 small reads, not the 640 thousand it would be if the unit here
//! were the segment. That is the whole reason the region exists as a concept:
//! the file is self describing and open time does not grow with the segment
//! size.
//!
//! **The root flip is the durability boundary.** A checkpoint syncs the data,
//! then writes the slot that is not live, then syncs again. Until that second
//! sync returns, the old slot is still the live one and the file still
//! describes the older, complete state. There is no window in which both slots
//! describe something that was never true.

use crate::io as fio;
use crate::log_file::LogFile;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use yo_common::{Code, Error, Result, SLOT_COUNT};
use yo_format::superblock::{
    CHECKPOINT_ENTRY_LEN, CRC_OFFSET, DEFAULT_SHARD_TABLE_OFF, decode_shard_table,
    encode_shard_table, pick, seal,
};
use yo_format::{
    CheckpointEntry, DATA_START, LOG_PAGE_LEN, PAGE_HEADER_LEN, PageHeader, SUPERBLOCK_LEN,
    Superblock, is_legal_page_size, superblock_flags,
};

/// One region, which is one log page's worth of contiguous segments.
pub const REGION_LEN: u64 = LOG_PAGE_LEN;

/// Where every written region is, keyed by the shard and the log address of the
/// page in it.
type RegionMap = HashMap<(u32, u64), u64>;

/// The byte offset of region `index`.
#[must_use]
pub const fn region_offset(index: u64) -> u64 {
    DATA_START + index * REGION_LEN
}

/// What a new file is created with.
///
/// All of it is frozen at creation except the fields that a checkpoint writes,
/// which is why it is a separate type from [`Superblock`]. A caller cannot
/// accidentally change the segment size of an existing database by passing the
/// wrong struct.
#[derive(Debug, Clone)]
pub struct CreateOptions {
    /// Segment size. See [`yo_format::is_legal_page_size`].
    pub page_size: u32,
    /// How many shards, which is normally one per core.
    pub shard_count: u32,
    /// Logical databases, the `SELECT` range.
    pub db_count: u32,
    /// Identifies this database across copies of the file. Zero is allowed and
    /// means nobody bothered, which is fine until two copies meet.
    pub file_uuid: [u8; 16],
    /// Creation time. Passed in rather than read from a clock here, so that a
    /// test can produce the same bytes twice.
    pub created_unix_ms: u64,
}

impl Default for CreateOptions {
    fn default() -> CreateOptions {
        CreateOptions {
            page_size: yo_format::DEFAULT_PAGE_SIZE,
            shard_count: 1,
            db_count: 16,
            file_uuid: [0; 16],
            created_unix_ms: 0,
        }
    }
}

/// Everything a checkpoint records that is not already in the superblock.
///
/// The four log addresses per shard are in `entries`, and they are the reason a
/// checkpoint exists at all: they are where replay starts.
#[derive(Debug, Clone, Copy)]
pub struct Checkpoint<'a> {
    /// One entry per shard, in shard order. Must be exactly `shard_count` long.
    pub entries: &'a [CheckpointEntry],
    /// Slot to shard assignment, one entry per Redis slot, or `None` to leave
    /// the table empty and put every slot on shard 0.
    pub slot_shard: Option<&'a [u16]>,
    /// Address of the collection catalogue, or 0.
    pub catalog_addr: u64,
    /// Address of the free segment list, or 0.
    pub free_list_addr: u64,
    /// Whether this checkpoint is the one written on the way out. Setting it on
    /// a checkpoint that is not the last one before close is how a database
    /// comes back missing everything written after it, so it defaults to false
    /// everywhere and only the shutdown path sets it.
    pub clean_shutdown: bool,
    /// When this checkpoint was taken.
    pub unix_ms: u64,
}

impl<'a> Checkpoint<'a> {
    /// A checkpoint that records `entries` and changes nothing else.
    #[must_use]
    pub fn new(entries: &'a [CheckpointEntry]) -> Checkpoint<'a> {
        Checkpoint {
            entries,
            slot_shard: None,
            catalog_addr: 0,
            free_list_addr: 0,
            clean_shutdown: false,
            unix_ms: 0,
        }
    }
}

/// Hands out regions, and is the one thing every shard's log shares.
///
/// Bump allocation, because there is no free list yet. A region that stops
/// being needed is not reused until compaction and the free segment list land,
/// which is `07` section 5 and is not in M1. Until then a file only grows, and
/// that is a known gap rather than a design.
pub(crate) struct Alloc {
    file: Arc<File>,
    next: u64,
}

impl Alloc {
    /// Grows the file by a region and returns its offset.
    pub(crate) fn take(&mut self) -> Result<u64> {
        let index = self.next;
        let off = region_offset(index);
        fio::grow_to(&self.file, off + REGION_LEN)
            .map_err(|e| io_err("could not grow the file by a region", &e))?;
        self.next = index + 1;
        Ok(off)
    }

    /// How many regions have been handed out.
    pub(crate) const fn used(&self) -> u64 {
        self.next
    }
}

/// An open `.yo` file.
///
/// Owns the descriptor and the superblock, and hands each shard a [`LogFile`]
/// that shares both. It does not own any data structures, does not know what a
/// key is and never parses a record. Everything above it talks in log
/// addresses.
pub struct Yo {
    path: PathBuf,
    file: Arc<File>,
    sb: Superblock,
    live: usize,
    slot_shard: Vec<u16>,
    entries: Vec<CheckpointEntry>,
    /// Where every region this file has written is.
    regions: RegionMap,
    alloc: Arc<Mutex<Alloc>>,
}

impl std::fmt::Debug for Yo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Yo")
            .field("path", &self.path)
            .field("seq", &self.sb.seq)
            .field("live_slot", &self.live)
            .field("shard_count", &self.sb.shard_count)
            .field("regions", &self.regions.len())
            .finish()
    }
}

impl Yo {
    /// Creates a file that does not exist yet.
    ///
    /// Both slots are written, B first with sequence 0 and A second with
    /// sequence 1, so that a crash anywhere in here leaves either no file, a
    /// file with one good slot, or a file with two. It never leaves a file with
    /// two bad ones.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for a shape that cannot work, [`Code::Io`] for
    /// anything the filesystem says, including the file already being there.
    pub fn create(path: impl AsRef<Path>, opts: &CreateOptions) -> Result<Yo> {
        let path = path.as_ref().to_path_buf();
        if !is_legal_page_size(opts.page_size) {
            return Err(
                Error::new(Code::Invalid, "that is not a legal segment size")
                    .with_detail(format!("page_size={}", opts.page_size)),
            );
        }
        if opts.shard_count == 0 {
            return Err(Error::new(
                Code::Invalid,
                "a database needs at least one shard",
            ));
        }

        let file = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| io_err("could not create the file", &e))?;
        let file = Arc::new(file);
        fio::grow_to(&file, DATA_START).map_err(|e| io_err("could not size the file", &e))?;

        let sb = Superblock {
            page_size: opts.page_size,
            shard_count: opts.shard_count,
            seq: 0,
            file_size: DATA_START,
            file_uuid: opts.file_uuid,
            created_unix_ms: opts.created_unix_ms,
            checkpoint_unix_ms: opts.created_unix_ms,
            db_count: opts.db_count,
            ..Superblock::default()
        };
        let entries = vec![CheckpointEntry::default(); opts.shard_count as usize];

        // B first, so that the slot which is about to become live is written
        // over a slot that is already good.
        let slot_b = encode_slot(&sb, &[], &entries)?;
        write_slot(&file, 1, &slot_b)?;
        let mut sb_a = sb;
        sb_a.seq = 1;
        let slot_a = encode_slot(&sb_a, &[], &entries)?;
        write_slot(&file, 0, &slot_a)?;
        fio::sync_all(&file).map_err(|e| io_err("could not sync the new file", &e))?;

        Ok(Yo {
            path,
            alloc: Arc::new(Mutex::new(Alloc {
                file: Arc::clone(&file),
                next: 0,
            })),
            file,
            sb: sb_a,
            live: 0,
            slot_shard: vec![0; SLOT_COUNT as usize],
            entries,
            regions: HashMap::new(),
        })
    }

    /// Opens a file that already exists.
    ///
    /// Reads both slots, picks the live one, expands the shard table, decodes
    /// the checkpoint entries, then walks the region headers. That last step is
    /// the only part that grows with the size of the file, and it is 32 bytes
    /// per 32 MiB.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if neither slot decodes, if the shard table does not
    /// cover every slot, if a checkpoint entry fails its checksum or its
    /// ordering, or if a region header is damaged. [`Code::VersionTooNew`] if
    /// the file wants a reader we are not.
    pub fn open(path: impl AsRef<Path>) -> Result<Yo> {
        let path = path.as_ref().to_path_buf();
        let file = File::options()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| io_err("could not open the file", &e))?;
        let file = Arc::new(file);

        let mut a = vec![0u8; SUPERBLOCK_LEN];
        let mut b = vec![0u8; SUPERBLOCK_LEN];
        let na = fio::read_at(&file, 0, &mut a).map_err(|e| io_err("could not read slot A", &e))?;
        let nb = fio::read_at(&file, SUPERBLOCK_LEN as u64, &mut b)
            .map_err(|e| io_err("could not read slot B", &e))?;
        if na < SUPERBLOCK_LEN || nb < SUPERBLOCK_LEN {
            return Err(Error::new(
                Code::Corrupt,
                "the file is too short to hold two superblocks",
            )
            .with_detail(format!("slot_a={na} slot_b={nb} want={SUPERBLOCK_LEN}")));
        }
        let (live, sb) = pick(&a, &b)?;
        if !sb.checkpoints_fit() {
            return Err(Error::new(
                Code::Corrupt,
                "the checkpoint entries do not fit in the superblock slot",
            )
            .with_detail(format!(
                "shard_table_off={} shard_table_len={} shard_count={}",
                sb.shard_table_off, sb.shard_table_len, sb.shard_count
            )));
        }
        let slot: &[u8] = if live == 0 { &a } else { &b };

        let mut slot_shard = vec![0u16; SLOT_COUNT as usize];
        if sb.shard_table_len > 0 {
            let off = sb.shard_table_off as usize;
            let end = off + sb.shard_table_len as usize;
            decode_shard_table(&slot[off..end], sb.shard_count, &mut slot_shard)?;
        }

        let mut entries = Vec::with_capacity(sb.shard_count as usize);
        let mut off = sb.checkpoint_off();
        for _ in 0..sb.shard_count {
            entries.push(CheckpointEntry::decode(
                &slot[off..off + CHECKPOINT_ENTRY_LEN],
            )?);
            off += CHECKPOINT_ENTRY_LEN;
        }

        let (regions, used) = scan_regions(&file)?;

        Ok(Yo {
            path,
            alloc: Arc::new(Mutex::new(Alloc {
                file: Arc::clone(&file),
                next: used,
            })),
            file,
            sb,
            live,
            slot_shard,
            entries,
            regions,
        })
    }

    /// The live superblock.
    #[must_use]
    pub const fn superblock(&self) -> &Superblock {
        &self.sb
    }

    /// Which slot the live superblock is in, 0 or 1.
    #[must_use]
    pub const fn live_slot(&self) -> usize {
        self.live
    }

    /// Where the file is.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The slot to shard table, one entry per Redis slot.
    #[must_use]
    pub fn slot_shard(&self) -> &[u16] {
        &self.slot_shard
    }

    /// The checkpoint entry for `shard`, which is where its replay starts.
    #[must_use]
    pub fn checkpoint_entry(&self, shard: u32) -> Option<CheckpointEntry> {
        self.entries.get(shard as usize).copied()
    }

    /// Whether the last checkpoint was written on the way out.
    ///
    /// False means replay. It is false for a file that is currently open, since
    /// [`Yo::checkpoint`] clears the bit on every checkpoint that does not set
    /// it.
    #[must_use]
    pub const fn was_clean(&self) -> bool {
        self.sb.flags & superblock_flags::CLEAN_SHUTDOWN != 0
    }

    /// How many regions have been handed out.
    #[must_use]
    pub fn region_count(&self) -> u64 {
        self.alloc
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .used()
    }

    /// The log for one shard.
    ///
    /// Takes the regions that belong to `shard` out of the file's map and gives
    /// them to the log, so calling this twice for the same shard hands the
    /// second caller a log with no history. That is on purpose: two logs over
    /// one shard would each believe they own the tail.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if `shard` is not a shard of this database.
    pub fn log(&mut self, shard: u32) -> Result<LogFile> {
        if shard >= self.sb.shard_count {
            return Err(Error::new(Code::Invalid, "that shard is not in this file")
                .with_detail(format!("shard={shard} shard_count={}", self.sb.shard_count)));
        }
        let mut mine = HashMap::new();
        self.regions.retain(|(s, page_addr), off| {
            if *s == shard {
                mine.insert(*page_addr, *off);
                false
            } else {
                true
            }
        });
        Ok(LogFile::new(
            shard,
            Arc::clone(&self.file),
            Arc::clone(&self.alloc),
            mine,
        ))
    }

    /// Writes a checkpoint and flips the root to it.
    ///
    /// Data first, then the slot, then the slot's own sync. The order is the
    /// whole guarantee: a slot that points at a log address is only ever
    /// written after the bytes at that address are on the device, so a
    /// checkpoint can be behind the truth but never ahead of it.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if `entries` is not one per shard, [`Code::Full`] if
    /// the table and the entries do not fit in 16 KiB, [`Code::Io`] for
    /// anything the filesystem says.
    pub fn checkpoint(&mut self, cp: &Checkpoint<'_>) -> Result<()> {
        if cp.entries.len() != self.sb.shard_count as usize {
            return Err(
                Error::new(Code::Invalid, "a checkpoint needs one entry per shard").with_detail(
                    format!("got={} want={}", cp.entries.len(), self.sb.shard_count),
                ),
            );
        }
        for e in cp.entries {
            if !e.addresses_are_ordered() {
                return Err(Error::new(
                    Code::Invalid,
                    "a checkpoint entry's log addresses are not in order",
                ));
            }
        }

        // Everything the new slot is about to point at, before the pointer.
        fio::sync_data(&self.file)
            .map_err(|e| io_err("could not sync the data before the checkpoint", &e))?;

        let len = self
            .file
            .metadata()
            .map_err(|e| io_err("could not measure the file", &e))?
            .len();

        let mut sb = self.sb.clone();
        sb.seq += 1;
        sb.file_size = len;
        sb.checkpoint_unix_ms = cp.unix_ms;
        sb.catalog_addr = cp.catalog_addr;
        sb.free_list_addr = cp.free_list_addr;
        if cp.clean_shutdown {
            sb.flags |= superblock_flags::CLEAN_SHUTDOWN;
        } else {
            sb.flags &= !superblock_flags::CLEAN_SHUTDOWN;
        }

        let table = cp.slot_shard.unwrap_or(&[]);
        let bytes = encode_slot(&sb, table, cp.entries)?;
        let target = 1 - self.live;
        write_slot(&self.file, target, &bytes)?;
        fio::sync_all(&self.file).map_err(|e| io_err("could not sync the checkpoint", &e))?;

        // Only now. Everything above this line could have been thrown away and
        // the file would still describe the previous checkpoint.
        sb.shard_table_off = DEFAULT_SHARD_TABLE_OFF;
        sb.shard_table_len = table_len(table)?;
        self.sb = sb;
        self.live = target;
        self.entries = cp.entries.to_vec();
        if !table.is_empty() {
            self.slot_shard = table.to_vec();
        }
        Ok(())
    }

    /// Makes everything written through this file durable, without a
    /// checkpoint.
    ///
    /// # Errors
    ///
    /// Whatever the filesystem says.
    pub fn sync(&self) -> Result<()> {
        fio::sync_data(&self.file).map_err(|e| io_err("could not sync the file", &e))
    }
}

/// Encodes a whole 16 KiB slot: header, table, entries, then the checksum.
///
/// The order matters and it is the reason this is one function rather than
/// three calls at the call site. The slot checksum covers all 16380 bytes in
/// front of it, so sealing before the entries are in produces a slot that fails
/// its own checksum.
fn encode_slot(
    sb: &Superblock,
    slot_shard: &[u16],
    entries: &[CheckpointEntry],
) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; SUPERBLOCK_LEN];
    let off = DEFAULT_SHARD_TABLE_OFF as usize;
    let len = if slot_shard.is_empty() {
        0
    } else {
        encode_shard_table(slot_shard, &mut buf[off..CRC_OFFSET])?
    };

    let mut sb = sb.clone();
    sb.shard_table_off = DEFAULT_SHARD_TABLE_OFF;
    sb.shard_table_len = u16::try_from(len)
        .map_err(|_| Error::new(Code::Full, "the shard table does not fit in a u16"))?;
    if !sb.checkpoints_fit() {
        return Err(
            Error::new(Code::Full, "the table and the entries do not fit in 16 KiB").with_detail(
                format!(
                    "shard_table_len={} shard_count={}",
                    sb.shard_table_len, sb.shard_count
                ),
            ),
        );
    }
    // The header is bytes 0 to 160 and the table starts at 160, so this does not
    // write over what was just encoded.
    sb.encode(&mut buf);

    let mut at = sb.checkpoint_off();
    for e in entries {
        e.encode(&mut buf[at..at + CHECKPOINT_ENTRY_LEN]);
        at += CHECKPOINT_ENTRY_LEN;
    }
    seal(&mut buf);
    Ok(buf)
}

fn table_len(slot_shard: &[u16]) -> Result<u16> {
    if slot_shard.is_empty() {
        return Ok(0);
    }
    let mut scratch = vec![0u8; CRC_OFFSET - DEFAULT_SHARD_TABLE_OFF as usize];
    let n = encode_shard_table(slot_shard, &mut scratch)?;
    u16::try_from(n).map_err(|_| Error::new(Code::Full, "the shard table does not fit in a u16"))
}

fn write_slot(file: &File, slot: usize, bytes: &[u8]) -> Result<()> {
    let off = (slot * SUPERBLOCK_LEN) as u64;
    fio::write_at(file, off, bytes).map_err(|e| io_err("could not write a superblock slot", &e))
}

/// Walks the region headers and rebuilds the map from log pages to offsets.
///
/// A region whose first 32 bytes are all zero has never been written, which is
/// what a file grown by an allocation that then crashed looks like. Anything
/// else that is not a valid header is damage, and damage stops the open. The
/// alternative is to skip it and hand the region out again, which would write
/// over whatever was there and call the result a clean database.
fn scan_regions(file: &File) -> Result<(RegionMap, u64)> {
    let len = file
        .metadata()
        .map_err(|e| io_err("could not measure the file", &e))?
        .len();
    let count = len.saturating_sub(DATA_START) / REGION_LEN;

    let mut regions = HashMap::new();
    let mut used = 0u64;
    let mut head = [0u8; PAGE_HEADER_LEN];
    for i in 0..count {
        let off = region_offset(i);
        let n = fio::read_at(file, off, &mut head)
            .map_err(|e| io_err("could not read a region header", &e))?;
        if n < PAGE_HEADER_LEN {
            break;
        }
        if head.iter().all(|b| *b == 0) {
            continue;
        }
        let h = PageHeader::decode(&head)
            .map_err(|e| e.with_detail(format!("region={i} offset={off}")))?;
        if let Some(other) = regions.insert((h.shard, h.page_addr), off) {
            return Err(
                Error::new(Code::Corrupt, "two regions claim the same log page").with_detail(
                    format!(
                        "shard={} page_addr={} here={off} there={other}",
                        h.shard, h.page_addr
                    ),
                ),
            );
        }
        used = i + 1;
    }
    Ok((regions, used))
}

/// Wraps a filesystem error without losing what the operating system said.
pub(crate) fn io_err(what: &str, e: &std::io::Error) -> Error {
    Error::new(Code::Io, what.to_string()).with_detail(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let mut p = std::env::temp_dir();
            p.push(format!("yo-file-{name}-{}.yo", std::process::id()));
            let _ = std::fs::remove_file(&p);
            Tmp(p)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn entries(n: usize) -> Vec<CheckpointEntry> {
        vec![CheckpointEntry::default(); n]
    }

    #[test]
    fn a_new_file_is_two_slots_and_nothing_else() {
        let t = Tmp::new("new");
        let db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        assert_eq!(db.live_slot(), 0);
        assert_eq!(db.superblock().seq, 1);
        assert_eq!(db.region_count(), 0);
        assert_eq!(std::fs::metadata(&t.0).unwrap().len(), DATA_START);
    }

    #[test]
    fn creating_over_an_existing_file_is_refused() {
        let t = Tmp::new("exists");
        Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let err = Yo::create(&t.0, &CreateOptions::default()).unwrap_err();
        assert_eq!(err.code(), Code::Io);
    }

    #[test]
    fn what_was_created_is_what_comes_back() {
        let t = Tmp::new("reopen");
        let opts = CreateOptions {
            page_size: 8192,
            shard_count: 4,
            db_count: 3,
            file_uuid: [7; 16],
            created_unix_ms: 1_700_000_000_000,
        };
        drop(Yo::create(&t.0, &opts).unwrap());

        let db = Yo::open(&t.0).unwrap();
        let sb = db.superblock();
        assert_eq!(sb.page_size, 8192);
        assert_eq!(sb.shard_count, 4);
        assert_eq!(sb.db_count, 3);
        assert_eq!(sb.file_uuid, [7; 16]);
        assert_eq!(sb.created_unix_ms, 1_700_000_000_000);
        assert_eq!(db.live_slot(), 0);
        assert_eq!(db.checkpoint_entry(3), Some(CheckpointEntry::default()));
        assert_eq!(db.checkpoint_entry(4), None);
    }

    #[test]
    fn a_checkpoint_flips_to_the_other_slot_and_back_again() {
        let t = Tmp::new("flip");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        assert_eq!(db.live_slot(), 0);

        let e = entries(1);
        db.checkpoint(&Checkpoint::new(&e)).unwrap();
        assert_eq!(db.live_slot(), 1);
        assert_eq!(db.superblock().seq, 2);

        db.checkpoint(&Checkpoint::new(&e)).unwrap();
        assert_eq!(db.live_slot(), 0);
        assert_eq!(db.superblock().seq, 3);

        drop(db);
        let db = Yo::open(&t.0).unwrap();
        assert_eq!(db.live_slot(), 0);
        assert_eq!(db.superblock().seq, 3);
    }

    #[test]
    fn the_four_log_addresses_survive_a_reopen() {
        let t = Tmp::new("addresses");
        let mut db = Yo::create(
            &t.0,
            &CreateOptions {
                shard_count: 2,
                ..CreateOptions::default()
            },
        )
        .unwrap();
        let e = vec![
            CheckpointEntry {
                log_begin: 0,
                log_head: 4096,
                log_read_only: 8192,
                log_tail: 9000,
                key_count: 12,
                epoch: 5,
                ..CheckpointEntry::default()
            },
            CheckpointEntry {
                log_tail: 77,
                ..CheckpointEntry::default()
            },
        ];
        db.checkpoint(&Checkpoint {
            catalog_addr: 4242,
            unix_ms: 999,
            ..Checkpoint::new(&e)
        })
        .unwrap();
        drop(db);

        let db = Yo::open(&t.0).unwrap();
        assert_eq!(db.checkpoint_entry(0).unwrap(), e[0]);
        assert_eq!(db.checkpoint_entry(1).unwrap(), e[1]);
        assert_eq!(db.superblock().catalog_addr, 4242);
        assert_eq!(db.superblock().checkpoint_unix_ms, 999);
    }

    #[test]
    fn a_shard_table_round_trips_through_a_checkpoint() {
        let t = Tmp::new("table");
        let mut db = Yo::create(
            &t.0,
            &CreateOptions {
                shard_count: 4,
                ..CreateOptions::default()
            },
        )
        .unwrap();
        let mut table = vec![0u16; SLOT_COUNT as usize];
        for (i, s) in table.iter_mut().enumerate() {
            *s = (i / 4096) as u16;
        }
        let e = entries(4);
        db.checkpoint(&Checkpoint {
            slot_shard: Some(&table),
            ..Checkpoint::new(&e)
        })
        .unwrap();
        drop(db);

        let db = Yo::open(&t.0).unwrap();
        assert_eq!(db.slot_shard(), table.as_slice());
        // Four runs, four bytes each.
        assert_eq!(db.superblock().shard_table_len, 16);
    }

    #[test]
    fn the_clean_bit_says_whether_replay_is_needed() {
        let t = Tmp::new("clean");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        assert!(!db.was_clean(), "a file that was just created is open");
        let e = entries(1);
        db.checkpoint(&Checkpoint {
            clean_shutdown: true,
            ..Checkpoint::new(&e)
        })
        .unwrap();
        drop(db);
        assert!(Yo::open(&t.0).unwrap().was_clean());

        let mut db = Yo::open(&t.0).unwrap();
        db.checkpoint(&Checkpoint::new(&e)).unwrap();
        drop(db);
        assert!(
            !Yo::open(&t.0).unwrap().was_clean(),
            "an ordinary checkpoint clears it again"
        );
    }

    #[test]
    fn a_torn_slot_loses_to_the_intact_one() {
        let t = Tmp::new("torn");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let e = entries(1);
        // Slot B is live and holds sequence 2.
        db.checkpoint(&Checkpoint::new(&e)).unwrap();
        assert_eq!(db.live_slot(), 1);
        drop(db);

        let f = File::options().write(true).open(&t.0).unwrap();
        fio::write_at(&f, SUPERBLOCK_LEN as u64 + 200, b"garbage").unwrap();
        drop(f);

        let db = Yo::open(&t.0).unwrap();
        assert_eq!(db.live_slot(), 0, "B failed its checksum, so A wins");
        assert_eq!(db.superblock().seq, 1);
    }

    #[test]
    fn both_slots_gone_is_an_error_that_names_both() {
        let t = Tmp::new("bothgone");
        Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let f = File::options().write(true).open(&t.0).unwrap();
        fio::write_at(&f, 300, b"x").unwrap();
        fio::write_at(&f, SUPERBLOCK_LEN as u64 + 300, b"x").unwrap();
        drop(f);

        let err = Yo::open(&t.0).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("slot B"));
    }

    #[test]
    fn a_file_that_is_not_ours_is_refused_before_anything_else() {
        let t = Tmp::new("notours");
        std::fs::write(&t.0, vec![0x41u8; SUPERBLOCK_LEN * 2]).unwrap();
        let err = Yo::open(&t.0).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        assert!(err.message().contains("not a .yo file"));
    }

    #[test]
    fn a_file_too_short_for_two_slots_says_so() {
        let t = Tmp::new("short");
        std::fs::write(&t.0, vec![0u8; 100]).unwrap();
        let err = Yo::open(&t.0).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.message().contains("too short"));
    }

    #[test]
    fn a_checkpoint_with_the_wrong_number_of_entries_is_refused() {
        let t = Tmp::new("wrongcount");
        let mut db = Yo::create(
            &t.0,
            &CreateOptions {
                shard_count: 2,
                ..CreateOptions::default()
            },
        )
        .unwrap();
        let e = entries(1);
        let err = db.checkpoint(&Checkpoint::new(&e)).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
    }

    #[test]
    fn a_checkpoint_with_unordered_addresses_never_reaches_the_file() {
        let t = Tmp::new("unordered");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let e = vec![CheckpointEntry {
            log_begin: 100,
            log_head: 50,
            log_read_only: 200,
            log_tail: 300,
            ..CheckpointEntry::default()
        }];
        let err = db.checkpoint(&Checkpoint::new(&e)).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(db.superblock().seq, 1, "the sequence did not move");
    }

    #[test]
    fn asking_for_a_shard_that_is_not_there_is_an_error() {
        let t = Tmp::new("noshard");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        assert!(db.log(0).is_ok());
        assert_eq!(db.log(1).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_damaged_region_header_stops_the_open() {
        let t = Tmp::new("badregion");
        {
            let db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
            let mut alloc = db.alloc.lock().unwrap();
            let off = alloc.take().unwrap();
            drop(alloc);
            // Not zeroes and not a page header, which is what a half written
            // region looks like after the wrong kind of crash.
            fio::write_at(&db.file, off, b"this was never a log page").unwrap();
            fio::sync_all(&db.file).unwrap();
        }
        let err = Yo::open(&t.0).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("region=0"));
    }

    #[test]
    fn an_allocated_but_unwritten_region_is_free_again_on_reopen() {
        let t = Tmp::new("unwritten");
        {
            let db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
            db.alloc.lock().unwrap().take().unwrap();
            fio::sync_all(&db.file).unwrap();
        }
        let db = Yo::open(&t.0).unwrap();
        assert_eq!(
            db.region_count(),
            0,
            "nothing was written into it, so it is not anybody's"
        );
    }
}
