//! Opening a `.yo` file and walking it.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::format::{
    CHECKPOINT_ENTRY_LEN, CheckpointEntry, DATA_START, LOG_PAGE_LEN, PAGE_HEADER_LEN, PageHeader,
    Record, SUPERBLOCK_LEN, Superblock, decode_shard_table, parse_record,
};
use crate::io as rio;

/// How much of a segment to read at a time when walking records.
///
/// A segment is 32 MiB and most of them are not full. Reading `used` bytes
/// rounded up to this is the difference between a check that runs in seconds
/// and one that reads the whole file.
const READ_BLOCK: usize = 64 * 1024;

/// What became of one superblock slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotStatus {
    /// It decoded, and this is its sequence number.
    Good {
        /// Higher wins.
        seq: u64,
    },
    /// It did not, and this is why.
    Bad(Error),
}

impl SlotStatus {
    /// Whether the slot decoded.
    #[must_use]
    pub const fn is_good(&self) -> bool {
        matches!(self, SlotStatus::Good { .. })
    }
}

/// A written region of the file, as found by reading its header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// Which region this is, counting from the start of the data area.
    pub index: u64,
    /// Its byte offset in the file.
    pub offset: u64,
    /// What the header said, when it said anything. All zeroes and meaningless
    /// when [`Region::damage`] is set.
    pub header: PageHeader,
    /// Why the header did not decode, if it did not.
    ///
    /// A damaged region is kept in the list rather than dropped, because the
    /// person running a check is usually running it because of exactly this and
    /// needs to know how much of the file is behind it.
    pub damage: Option<Error>,
}

impl Region {
    /// Whether the header decoded.
    #[must_use]
    pub const fn is_good(&self) -> bool {
        self.damage.is_none()
    }
}

/// A `.yo` file, open for reading.
///
/// Holds the file, the live superblock and where every written region is. It
/// does not hold any record bytes, so a reader over a ten gigabyte file costs
/// about as much as one over an empty file.
#[derive(Debug)]
pub struct Reader {
    path: PathBuf,
    file: File,
    sb: Superblock,
    live: usize,
    slots: [SlotStatus; 2],
    regions: Vec<Region>,
}

impl Reader {
    /// Opens a file and reads enough of it to describe it.
    ///
    /// That is two superblock slots and one 32 byte header per region, so open
    /// time goes with the size of the file and not with the number of keys in
    /// it. Records are read later, a region at a time, by whoever asks.
    ///
    /// Unlike the engine, a region whose header will not decode is recorded and
    /// stepped over rather than being a reason to give up. The engine stops
    /// because handing out a damaged region and writing to it is how a silent
    /// corruption happens. This crate is never going to write anything, and the
    /// person running it is usually running it precisely because something is
    /// damaged, so stopping at the first bad region would hide the rest of the
    /// file from them.
    ///
    /// # Errors
    ///
    /// If the file will not open, if neither superblock slot decodes, or if the
    /// live slot describes a shard table or a set of checkpoint entries that
    /// does not fit inside it.
    pub fn open(path: &Path) -> Result<Reader> {
        let file = File::open(path).map_err(|e| Error::new(format!("{}: {e}", path.display())))?;

        let mut a = vec![0u8; SUPERBLOCK_LEN];
        let mut b = vec![0u8; SUPERBLOCK_LEN];
        rio::read_exact_at(&file, 0, &mut a).map_err(|e| Error::from(e).at(0))?;
        rio::read_exact_at(&file, SUPERBLOCK_LEN as u64, &mut b)
            .map_err(|e| Error::from(e).at(SUPERBLOCK_LEN as u64))?;

        let da = Superblock::decode(&a);
        let db = Superblock::decode(&b);
        let slots = [
            match &da {
                Ok(s) => SlotStatus::Good { seq: s.seq },
                Err(e) => SlotStatus::Bad(e.clone().at(0)),
            },
            match &db {
                Ok(s) => SlotStatus::Good { seq: s.seq },
                Err(e) => SlotStatus::Bad(e.clone().at(SUPERBLOCK_LEN as u64)),
            },
        ];

        // Higher sequence number wins, and a slot that did not decode does not
        // get a vote. Both failing is the one case nobody can recover from, and
        // both reasons go in the message because guessing which one the reader
        // cares about is how you print the less useful of the two.
        let (live, sb) = match (da, db) {
            (Ok(sa), Ok(sb)) => {
                if sb.seq > sa.seq {
                    (1, sb)
                } else {
                    (0, sa)
                }
            }
            (Ok(sa), Err(_)) => (0, sa),
            (Err(_), Ok(sb)) => (1, sb),
            (Err(ea), Err(eb)) => {
                return Err(Error::new(format!(
                    "neither superblock decodes. slot A: {ea}. slot B: {eb}"
                )));
            }
        };

        if !sb.checkpoints_fit() {
            return Err(Error::new(format!(
                "the checkpoint entries do not fit in the slot: table at {} for {} bytes, {} shards",
                sb.shard_table_off, sb.shard_table_len, sb.shard_count
            ))
            .at(live as u64 * SUPERBLOCK_LEN as u64));
        }

        let regions = scan_regions(&file)?;
        Ok(Reader {
            path: path.to_path_buf(),
            file,
            sb,
            live,
            slots,
            regions,
        })
    }

    /// The file this was opened on.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The live superblock.
    #[must_use]
    pub const fn superblock(&self) -> &Superblock {
        &self.sb
    }

    /// Which slot won, 0 or 1.
    #[must_use]
    pub const fn live_slot(&self) -> usize {
        self.live
    }

    /// What became of each slot, including the one that lost.
    ///
    /// A file where one slot is damaged still opens, and that is the design
    /// working: the whole reason there are two is so a crash in the middle of
    /// writing one leaves the other intact. But it is also the thing anyone
    /// running a check wants told, so it is here rather than swallowed.
    #[must_use]
    pub const fn slots(&self) -> &[SlotStatus; 2] {
        &self.slots
    }

    /// Every written region, in file order.
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The slot to shard mapping, expanded.
    ///
    /// An empty table in the file means every slot belongs to shard 0, which is
    /// what a single shard database writes rather than 16384 entries saying the
    /// same thing.
    ///
    /// # Errors
    ///
    /// If the table cannot be read or does not decode.
    pub fn shard_table(&self) -> Result<Vec<u16>> {
        if self.sb.shard_table_len == 0 {
            return Ok(vec![0u16; crate::format::SLOT_COUNT]);
        }
        let off = self.slot_offset() + u64::from(self.sb.shard_table_off);
        let mut buf = vec![0u8; self.sb.shard_table_len as usize];
        rio::read_exact_at(&self.file, off, &mut buf).map_err(|e| Error::from(e).at(off))?;
        decode_shard_table(&buf, self.sb.shard_count).map_err(|e| e.at(off))
    }

    /// One checkpoint entry per shard, in shard order.
    ///
    /// # Errors
    ///
    /// If an entry cannot be read, fails its checksum, or has addresses that go
    /// backwards.
    pub fn checkpoints(&self) -> Result<Vec<CheckpointEntry>> {
        let base = self.slot_offset() + self.sb.checkpoint_off() as u64;
        let mut out = Vec::with_capacity(self.sb.shard_count as usize);
        let mut buf = [0u8; CHECKPOINT_ENTRY_LEN];
        for shard in 0..u64::from(self.sb.shard_count) {
            let off = base + shard * CHECKPOINT_ENTRY_LEN as u64;
            rio::read_exact_at(&self.file, off, &mut buf).map_err(|e| Error::from(e).at(off))?;
            out.push(
                CheckpointEntry::decode(&buf)
                    .map_err(|e| Error::new(format!("shard {shard}: {e}")).at(off))?,
            );
        }
        Ok(out)
    }

    /// Every record in one region, in the order they were appended.
    ///
    /// Stops at the first zero length field, which is the end of what was
    /// written, and stops at `used` whichever comes first. A record that will
    /// not parse ends the walk with an error carrying the byte offset it was
    /// at, because everything after a record of unknown length is unreachable:
    /// the only way to find the next record is to trust this one's length.
    ///
    /// # Errors
    ///
    /// If the region's header did not decode, if the region cannot be read, or
    /// if a record in it does not parse.
    pub fn records(&self, region: &Region) -> Result<Vec<Record>> {
        if let Some(d) = &region.damage {
            return Err(Error::new(format!(
                "region {} has no usable header, so there is no way to know where its records end: {d}",
                region.index
            ))
            .at(region.offset));
        }
        let used = region.header.used as usize;
        if used == 0 {
            return Ok(Vec::new());
        }
        // The sentinel that ends the walk lives just past the last record, so
        // read four bytes more than `used` claims, and never more than the
        // segment holds.
        let want = (PAGE_HEADER_LEN + used + 4)
            .next_multiple_of(READ_BLOCK)
            .min(LOG_PAGE_LEN as usize);
        let mut buf = vec![0u8; want];
        let got = rio::read_at(&self.file, region.offset, &mut buf)
            .map_err(|e| Error::from(e).at(region.offset))?;
        buf.truncate(got);

        let end = (PAGE_HEADER_LEN + used).min(buf.len());
        let mut out = Vec::new();
        let mut at = PAGE_HEADER_LEN;
        while at < end {
            let here = region.offset + at as u64;
            match parse_record(&buf[at..end]) {
                Ok(Some(r)) => {
                    at += r.stride();
                    out.push(r);
                }
                Ok(None) => break,
                Err(e) => return Err(e.at(here)),
            }
        }
        Ok(out)
    }

    /// Where the live slot starts in the file.
    const fn slot_offset(&self) -> u64 {
        self.live as u64 * SUPERBLOCK_LEN as u64
    }
}

/// Walks the data area reading one header per region.
///
/// A region whose first 32 bytes are all zero was allocated and never written,
/// and it is simply not there as far as anyone is concerned. Anything else that
/// does not decode is damage, and it goes in the list with a header of zeroes
/// so that whoever is looking at the file can see there is a hole and where.
fn scan_regions(file: &File) -> Result<Vec<Region>> {
    let len = file
        .metadata()
        .map_err(|e| Error::new(format!("cannot stat the file: {e}")))?
        .len();
    if len <= DATA_START {
        return Ok(Vec::new());
    }
    let count = (len - DATA_START) / LOG_PAGE_LEN;
    let mut out = Vec::new();
    let mut head = [0u8; PAGE_HEADER_LEN];
    for index in 0..count {
        let offset = DATA_START + index * LOG_PAGE_LEN;
        let n = rio::read_at(file, offset, &mut head).map_err(|e| Error::from(e).at(offset))?;
        if n < PAGE_HEADER_LEN || head.iter().all(|&b| b == 0) {
            continue;
        }
        let (header, damage) = match PageHeader::decode(&head) {
            Ok(h) => (h, None),
            Err(e) => (PageHeader::default(), Some(e.at(offset))),
        };
        out.push(Region {
            index,
            offset,
            header,
            damage,
        });
    }
    Ok(out)
}
