//! One shard's log, backed by regions of a `.yo` file.
//!
//! This is the other half of the seam that `yo-record` opens. The log fills
//! pages and hands them to a `PageSink`; this is the sink that puts them in a
//! file, and the `PageSource` that reads them back.
//!
//! **A page becomes a region the first time it is written.** The log addresses
//! its pages logically, starting at zero and going up forever, and nothing in
//! the log knows or cares where a page lands. The mapping is here, it is one
//! entry per 32 MiB, and it is rebuilt at open time from the headers rather
//! than stored anywhere. There is no allocation table to keep consistent with
//! the data, because there is no allocation table.
//!
//! **The read cache only grows.** [`LogFile::page_bytes`] hands out a borrow
//! that lives as long as the borrow of `self`, which means it cannot free
//! anything while a caller might still be holding a page. Freeing takes
//! [`LogFile::clear_cache`], which takes `&mut self`, and `&mut self` is the
//! proof that nobody is. Compaction and recovery each read a bounded number of
//! pages and then clear, so this stays small in practice, and
//! [`LogFile::cache_bytes`] is there for a caller that wants to check rather
//! than assume.

use crate::file::{Alloc, REGION_LEN, io_err};
use crate::io as fio;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex, PoisonError};
use yo_common::Result;
use yo_format::{PAGE_HEADER_LEN, PageHeader};
use yo_record::{PageSink, PageSource, PageWrite};

/// Reads are rounded up to this, because a device does not do better than a
/// block and a page that is nearly empty should not cost 32 MiB to look at.
const READ_BLOCK: usize = 4096;

/// Page images read back from the file.
///
/// Each entry owns its allocation outright. The pointers come from
/// [`Box::into_raw`], so growing the `Vec` moves the pointers and not the bytes
/// they point at, and a borrow handed out earlier stays valid.
struct Cache {
    entries: Vec<(u64, *mut [u8])>,
    bytes: usize,
}

impl Cache {
    const fn new() -> Cache {
        Cache {
            entries: Vec::new(),
            bytes: 0,
        }
    }

    fn find(&self, page_addr: u64) -> Option<*mut [u8]> {
        self.entries
            .iter()
            .find(|(a, _)| *a == page_addr)
            .map(|(_, p)| *p)
    }

    fn insert(&mut self, page_addr: u64, buf: Box<[u8]>) -> *mut [u8] {
        self.bytes += buf.len();
        let p = Box::into_raw(buf);
        self.entries.push((page_addr, p));
        p
    }

    /// Frees one page. Only reachable through `&mut LogFile`.
    fn forget(&mut self, page_addr: u64) {
        let Some(i) = self.entries.iter().position(|(a, _)| *a == page_addr) else {
            return;
        };
        let (_, p) = self.entries.swap_remove(i);
        // SAFETY: `p` came from `Box::into_raw` in `insert` and was removed from
        // `entries` just now, so this is the only pointer to it left and nothing
        // has handed out a borrow that outlives the `&mut self` we got here
        // through.
        let b = unsafe { Box::from_raw(p) };
        self.bytes -= b.len();
    }

    fn clear(&mut self) {
        for (_, p) in self.entries.drain(..) {
            // SAFETY: same as `forget`. Each pointer came from `Box::into_raw`,
            // appears once, and is being taken out of the cache as it is freed.
            drop(unsafe { Box::from_raw(p) });
        }
        self.bytes = 0;
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        self.clear();
    }
}

/// One shard's log pages, in a file.
///
/// Not `Sync`, and that is not an accident. A log belongs to one shard, a shard
/// belongs to one core, and the read cache below relies on there being exactly
/// one thread in here at a time.
pub struct LogFile {
    shard: u32,
    file: Arc<File>,
    alloc: Arc<Mutex<Alloc>>,
    /// `page_addr` to file offset, for the pages this shard has written.
    regions: HashMap<u64, u64>,
    cache: UnsafeCell<Cache>,
    written_upto: u64,
    durable_upto: u64,
    writes: u64,
    syncs: u64,
}

impl std::fmt::Debug for LogFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogFile")
            .field("shard", &self.shard)
            .field("regions", &self.regions.len())
            .field("written_upto", &self.written_upto)
            .field("durable_upto", &self.durable_upto)
            .finish()
    }
}

impl LogFile {
    pub(crate) fn new(
        shard: u32,
        file: Arc<File>,
        alloc: Arc<Mutex<Alloc>>,
        regions: HashMap<u64, u64>,
    ) -> LogFile {
        LogFile {
            shard,
            file,
            alloc,
            regions,
            cache: UnsafeCell::new(Cache::new()),
            written_upto: 0,
            durable_upto: 0,
            writes: 0,
            syncs: 0,
        }
    }

    /// Which shard this log belongs to.
    #[must_use]
    pub const fn shard(&self) -> u32 {
        self.shard
    }

    /// How many log pages this shard has regions for.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// The log addresses this shard has pages for, in order.
    ///
    /// Recovery walks this to find where its records are.
    #[must_use]
    pub fn page_addrs(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.regions.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// How many byte ranges have been handed to the file.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }

    /// How many syncs have been asked for. Group commit is the claim that this
    /// stays far below the commit count.
    #[must_use]
    pub const fn syncs(&self) -> u64 {
        self.syncs
    }

    /// How much memory the read cache is holding.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        // SAFETY: `LogFile` is not `Sync`, so there is one thread in here, and
        // this reads a field without creating a reference that outlives the
        // call.
        unsafe { (*self.cache.get()).bytes }
    }

    /// Frees every cached page.
    ///
    /// Takes `&mut self` because that is the only way to know that no borrow
    /// handed out by [`LogFile::page_bytes`] is still alive.
    pub fn clear_cache(&mut self) {
        self.cache.get_mut().clear();
    }

    /// The offset of the region holding `page_addr`, allocating one if this is
    /// the first write to that page.
    fn region_for(&mut self, page_addr: u64) -> Result<u64> {
        if let Some(off) = self.regions.get(&page_addr) {
            return Ok(*off);
        }
        let off = self
            .alloc
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()?;
        self.regions.insert(page_addr, off);
        Ok(off)
    }

    /// Reads a page back, no more of it than there is anything in.
    ///
    /// The header says how much of the payload has been written, so a page with
    /// four kilobytes of records in it costs one block to read rather than the
    /// whole 32 MiB. A header that does not decode means the caller is looking
    /// at damage, and then the whole region comes back so that `yo check` has
    /// something to look at.
    fn read_page(&self, off: u64) -> Option<Box<[u8]>> {
        let mut head = [0u8; PAGE_HEADER_LEN];
        if fio::read_at(&self.file, off, &mut head).ok()? < PAGE_HEADER_LEN {
            return None;
        }
        let want = match PageHeader::decode(&head) {
            // The four bytes past the last record are the end of page sentinel,
            // and a reader that stops before them cannot tell that the page
            // ends.
            Ok(h) => (PAGE_HEADER_LEN + h.used as usize + 4)
                .next_multiple_of(READ_BLOCK)
                .min(REGION_LEN as usize),
            Err(_) => REGION_LEN as usize,
        };
        let mut buf = vec![0u8; want];
        let n = fio::read_at(&self.file, off, &mut buf).ok()?;
        if n < PAGE_HEADER_LEN {
            return None;
        }
        buf.truncate(n);
        Some(buf.into_boxed_slice())
    }
}

impl PageSink for LogFile {
    fn write(&mut self, w: PageWrite<'_>) -> Result<()> {
        let off = self.region_for(w.page_addr)?;
        fio::write_at(&self.file, off + w.offset as u64, w.bytes)
            .map_err(|e| io_err("could not write a log page", &e))?;
        self.writes += 1;
        self.written_upto = self.written_upto.max(w.covers_upto);
        // Whatever was cached for this page is now behind the file. Dropping it
        // is sound here and only here, because `write` holds `&mut self`.
        self.cache.get_mut().forget(w.page_addr);
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        fio::sync_data(&self.file).map_err(|e| io_err("could not sync the log", &e))?;
        self.syncs += 1;
        // Exactly what was handed over before the sync, and nothing handed over
        // after it.
        self.durable_upto = self.written_upto;
        Ok(())
    }

    fn durable_upto(&self) -> u64 {
        self.durable_upto
    }
}

impl PageSource for LogFile {
    fn page_bytes(&self, page_addr: u64) -> Option<&[u8]> {
        // SAFETY: `LogFile` is not `Sync`, so no other thread is in here. The
        // `&mut Cache` below lives only until the end of this call and no borrow
        // handed out to a caller is derived from it: the pointer that gets
        // returned is a copy of one that came out of `Box::into_raw`, whose
        // allocation is not the `Vec`'s and is not touched by the `Vec` growing.
        let cache = unsafe { &mut *self.cache.get() };
        let p = match cache.find(page_addr) {
            Some(p) => p,
            None => {
                let off = *self.regions.get(&page_addr)?;
                // An I/O error becomes `None` here, because the trait has no
                // room for anything else. The caller turns that into its own
                // error: recovery says the tail page is not in the store, which
                // is the truth from where it is standing.
                cache.insert(page_addr, self.read_page(off)?)
            }
        };
        // SAFETY: `p` owns a live allocation from `Box::into_raw`, held by the
        // cache. Nothing frees it without `&mut self`, which the returned borrow
        // rules out for as long as it is alive.
        Some(unsafe { &*p })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{Checkpoint, CreateOptions, Yo};
    use std::path::PathBuf;
    use yo_format::{LOG_PAGE_LEN, RecordHeader, RecordKind};
    use yo_record::{Durability, Log, LogConfig};

    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let mut p = std::env::temp_dir();
            p.push(format!("yo-logfile-{name}-{}.yo", std::process::id()));
            let _ = std::fs::remove_file(&p);
            Tmp(p)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn cfg(shard: u32) -> LogConfig {
        LogConfig {
            shard,
            page_len: LOG_PAGE_LEN as usize,
            durability: Durability::Group,
            ..LogConfig::default()
        }
    }

    /// Writes one block into the page at `page_addr`, with `mark` repeated
    /// through the payload so a reader can tell the pages apart.
    fn put_page(sink: &mut LogFile, page_addr: u64, mark: u8) {
        let mut page = vec![0u8; READ_BLOCK];
        PageHeader {
            shard: sink.shard(),
            page_addr,
            used: 64,
            dead_bytes: 0,
            epoch: 1,
        }
        .encode(&mut page);
        page[PAGE_HEADER_LEN..PAGE_HEADER_LEN + 64].fill(mark);
        sink.write(PageWrite {
            page_addr,
            offset: 0,
            bytes: &page,
            covers_upto: page_addr + 64,
        })
        .unwrap();
    }

    #[test]
    fn a_page_becomes_a_region_the_first_time_it_is_written() {
        let t = Tmp::new("firstwrite");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        assert_eq!(sink.region_count(), 0);

        let mut page = vec![0u8; READ_BLOCK];
        PageHeader {
            shard: 0,
            page_addr: 0,
            used: 8,
            dead_bytes: 0,
            epoch: 1,
        }
        .encode(&mut page);
        page[PAGE_HEADER_LEN..PAGE_HEADER_LEN + 8].copy_from_slice(b"hello yo");
        sink.write(PageWrite {
            page_addr: 0,
            offset: 0,
            bytes: &page,
            covers_upto: 8,
        })
        .unwrap();

        assert_eq!(sink.region_count(), 1);
        assert_eq!(db.region_count(), 1);
        assert_eq!(
            std::fs::metadata(&t.0).unwrap().len(),
            yo_format::DATA_START + REGION_LEN
        );
    }

    #[test]
    fn a_written_page_reads_back_and_costs_one_block() {
        let t = Tmp::new("readback");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();

        let mut page = vec![0u8; READ_BLOCK];
        PageHeader {
            shard: 0,
            page_addr: 0,
            used: 8,
            dead_bytes: 0,
            epoch: 3,
        }
        .encode(&mut page);
        page[PAGE_HEADER_LEN..PAGE_HEADER_LEN + 8].copy_from_slice(b"hello yo");
        sink.write(PageWrite {
            page_addr: 0,
            offset: 0,
            bytes: &page,
            covers_upto: 8,
        })
        .unwrap();

        let got = sink.page_bytes(0).unwrap();
        let h = PageHeader::decode(got).unwrap();
        assert_eq!(h.epoch, 3);
        assert_eq!(h.used, 8);
        assert_eq!(&got[PAGE_HEADER_LEN..PAGE_HEADER_LEN + 8], b"hello yo");
        assert_eq!(
            got.len(),
            READ_BLOCK,
            "a nearly empty page does not cost 32 MiB to look at"
        );
        assert_eq!(sink.cache_bytes(), READ_BLOCK);

        sink.clear_cache();
        assert_eq!(sink.cache_bytes(), 0);
    }

    /// The cache hands out borrows tied to `&self`, so several of them are alive
    /// at once and every one of them has to stay good while later reads push new
    /// entries in behind them. That is the whole reason each entry owns its own
    /// allocation instead of living in the `Vec`, and this is the test that says
    /// so. Miri under both borrow models is what actually checks it.
    #[test]
    fn several_page_borrows_stay_good_while_more_pages_are_read() {
        let t = Tmp::new("aliasing");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        for i in 0..8u64 {
            put_page(&mut sink, i * LOG_PAGE_LEN, 0xa0 + i as u8);
        }
        assert_eq!(sink.region_count(), 8);

        let mut held: Vec<&[u8]> = Vec::new();
        for i in 0..8u64 {
            let page = sink.page_bytes(i * LOG_PAGE_LEN).unwrap();
            held.push(page);
            // Everything read so far is still readable and still says what it
            // said, after the read that just happened moved the cache's `Vec`.
            for (j, earlier) in held.iter().enumerate() {
                assert_eq!(earlier[PAGE_HEADER_LEN], 0xa0 + j as u8);
                assert_eq!(
                    PageHeader::decode(earlier).unwrap().page_addr,
                    j as u64 * LOG_PAGE_LEN
                );
            }
        }
        assert_eq!(sink.cache_bytes(), 8 * READ_BLOCK);
        drop(held);
        sink.clear_cache();
        assert_eq!(sink.cache_bytes(), 0);
    }

    #[test]
    fn writing_a_page_again_throws_away_what_was_cached_for_it() {
        let t = Tmp::new("stale");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        put_page(&mut sink, 0, 0x11);
        assert_eq!(sink.page_bytes(0).unwrap()[PAGE_HEADER_LEN], 0x11);
        assert_eq!(sink.cache_bytes(), READ_BLOCK);

        put_page(&mut sink, 0, 0x22);
        assert_eq!(
            sink.cache_bytes(),
            0,
            "the cached image went with the write"
        );
        assert_eq!(sink.page_bytes(0).unwrap()[PAGE_HEADER_LEN], 0x22);
        assert_eq!(sink.region_count(), 1, "the same page, not a new region");
    }

    #[test]
    fn a_page_nobody_has_written_is_not_there() {
        let t = Tmp::new("missing");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let sink = db.log(0).unwrap();
        assert!(sink.page_bytes(0).is_none());
        assert!(sink.page_bytes(LOG_PAGE_LEN).is_none());
    }

    #[test]
    fn a_sync_makes_durable_what_was_written_before_it() {
        let t = Tmp::new("durable");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        let bytes = vec![0u8; READ_BLOCK];

        assert_eq!(sink.durable_upto(), 0);
        sink.write(PageWrite {
            page_addr: 0,
            offset: 0,
            bytes: &bytes,
            covers_upto: 100,
        })
        .unwrap();
        assert_eq!(sink.durable_upto(), 0, "written is not durable");
        sink.sync().unwrap();
        assert_eq!(sink.durable_upto(), 100);
        assert_eq!(sink.syncs(), 1);
        assert_eq!(sink.writes(), 1);
    }

    #[test]
    fn two_shards_get_their_own_regions() {
        let t = Tmp::new("twoshards");
        let mut db = Yo::create(
            &t.0,
            &CreateOptions {
                shard_count: 2,
                ..CreateOptions::default()
            },
        )
        .unwrap();
        let bytes = vec![1u8; READ_BLOCK];
        let mut a = db.log(0).unwrap();
        let mut b = db.log(1).unwrap();
        // Both write log address 0, which is a different place for each of them.
        a.write(PageWrite {
            page_addr: 0,
            offset: 0,
            bytes: &bytes,
            covers_upto: 1,
        })
        .unwrap();
        b.write(PageWrite {
            page_addr: 0,
            offset: 0,
            bytes: &bytes,
            covers_upto: 1,
        })
        .unwrap();
        assert_eq!(db.region_count(), 2, "two regions, not one shared one");
    }

    /// The whole point of the crate in one test: write records through a real
    /// file, checkpoint, drop everything, open it again and read them back.
    #[test]
    fn a_log_over_a_file_survives_being_closed_and_reopened() {
        let t = Tmp::new("logreopen");
        let keys: Vec<String> = (0..50u32).map(|i| format!("key:{i}")).collect();
        let values: Vec<String> = (0..50u32).map(|i| format!("value number {i}")).collect();

        let (tail, epoch, addrs) = {
            let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
            let mut log = Log::open(cfg(0), db.log(0).unwrap(), 0).unwrap();
            let h = RecordHeader::new(RecordKind::String);
            let mut addrs = Vec::new();
            for (k, v) in keys.iter().zip(&values) {
                addrs.push(log.append(&h, k.as_bytes(), v.as_bytes()).unwrap().addr);
            }
            log.advance_epoch();
            log.commit_pending().unwrap();
            let tail = log.tail();
            let epoch = log.epoch();
            let entries = [log.checkpoint_entry(0, 0, keys.len() as u64)];
            drop(log);

            db.checkpoint(&Checkpoint {
                clean_shutdown: true,
                ..Checkpoint::new(&entries)
            })
            .unwrap();
            (tail, epoch, addrs)
        };

        let mut db = Yo::open(&t.0).unwrap();
        assert!(db.was_clean());
        let entry = db.checkpoint_entry(0).unwrap();
        assert_eq!(entry.log_tail, tail);
        assert_eq!(entry.epoch, epoch);
        assert_eq!(entry.key_count, 50);

        let sink = db.log(0).unwrap();
        assert_eq!(sink.region_count(), 1);
        let mut log = Log::recover(cfg(0), sink, entry.log_tail).unwrap();
        assert_eq!(log.epoch(), epoch);
        for ((a, k), v) in addrs.iter().zip(&keys).zip(&values) {
            let r = log.read(*a).unwrap();
            assert_eq!(r.key, k.as_bytes());
            assert_eq!(r.value, v.as_bytes());
        }

        // And it keeps going from where it stopped, without writing over what
        // is already in the tail page.
        let h = RecordHeader::new(RecordKind::String);
        let more = log.append(&h, b"late", b"after the reopen").unwrap().addr;
        assert_eq!(more, tail);
        log.commit_pending().unwrap();
        assert_eq!(log.read(more).unwrap().value, b"after the reopen");
        assert_eq!(log.read(addrs[0]).unwrap().value, b"value number 0");
    }
}
