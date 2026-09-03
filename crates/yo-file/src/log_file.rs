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
//!
//! **A point read does not go through it.** [`LogFile::read_into`] copies the
//! bytes the caller asked for and caches nothing, because the caller there is a
//! client naming a key whose value was moved out of memory, and over a store
//! larger than memory the page it lands in is a different one every time. Read
//! those through the cache and a run of random reads pulls 32 MiB per key and
//! ends up holding the whole file, which is the opposite of what a file is for.

use crate::file::{Alloc, REGION_LEN, io_err};
use crate::io as fio;
use crate::ring::RingWriter;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex, PoisonError};
use yo_common::{Code, Error, Result};
use yo_format::{PAGE_HEADER_LEN, PageHeader};
use yo_record::{PageSink, PageSource, PageWrite};
use yo_uring::RingConfig;

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
    /// Set by [`LogFile::use_ring`], and then every write goes through it
    /// instead of through `pwrite`.
    ///
    /// In a cell for the same reason the cache is: [`PageSource::page_bytes`]
    /// takes `&self` and has to wait for an outstanding write to that page
    /// before it reads the page back, or it reads bytes the kernel has not put
    /// there yet.
    ring: Option<UnsafeCell<RingWriter>>,
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
            .field("ring", &self.ring.is_some())
            .field("written_upto", &self.written_upto)
            .field("durable_upto", &self.durable_upto())
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
            ring: None,
            written_upto: 0,
            durable_upto: 0,
            writes: 0,
            syncs: 0,
        }
    }

    /// Moves this log's writes onto the submission ring.
    ///
    /// This is what `06` section 3 needs to reach two hundred thousand durable
    /// commits a second. Without it the shard stops for every `pwrite` and every
    /// `fdatasync`; with it the shard hands the bytes over and keeps going, and
    /// a commit that is waiting on durability is parked on an address rather
    /// than on a syscall.
    ///
    /// Call it before the first write. There is no reason to call it later and
    /// the counters would not line up if you did.
    ///
    /// On Linux this is io_uring. On macOS and Windows it is the same state
    /// machine over synchronous storage (`04` section 7), which is correct and
    /// tested and slower, and [`LogFile::is_uring`] is how a benchmark row says
    /// which of the two produced it.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if this log is already in ring mode or has already
    /// written something, or whatever the ring says about the configuration and
    /// the kernel.
    pub fn use_ring(&mut self, config: &RingConfig) -> Result<()> {
        if self.ring.is_some() {
            return Err(Error::new(Code::Invalid, "this log is already on a ring"));
        }
        if self.writes > 0 {
            return Err(Error::new(
                Code::Invalid,
                "a log switches to the ring before its first write, not after",
            ));
        }
        let w = RingWriter::new(Arc::clone(&self.file), config)?;
        self.ring = Some(UnsafeCell::new(w));
        Ok(())
    }

    /// Whether this log is on a ring at all.
    #[must_use]
    pub const fn is_ringed(&self) -> bool {
        self.ring.is_some()
    }

    /// Whether the ring under this log is a real io_uring, which off Linux it is
    /// not.
    ///
    /// Every benchmark row carries this, because a number from the portable
    /// backend and a number from io_uring are not the same measurement and a
    /// table that does not say which is which is how you publish four wrong
    /// numbers.
    #[must_use]
    pub fn is_uring(&self) -> bool {
        self.read_ring(RingWriter::is_uring).unwrap_or(false)
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

    /// Asks the ring something, if there is one.
    fn read_ring<T>(&self, f: impl FnOnce(&RingWriter) -> T) -> Option<T> {
        let cell = self.ring.as_ref()?;
        // SAFETY: `LogFile` is not `Sync`, so there is one thread in here, and
        // the only thing that hands out a `&mut` to the writer through a `&self`
        // is `with_ring`, which does not nest with this.
        Some(f(unsafe { &*cell.get() }))
    }

    /// Runs `f` against the ring, if there is one.
    ///
    /// The one place `&mut RingWriter` comes out of a `&self`. It has exactly
    /// one caller, [`PageSource::page_bytes`], and the aliasing argument lives
    /// here rather than being repeated at every accessor.
    fn with_ring<T>(&self, f: impl FnOnce(&mut RingWriter) -> T) -> Option<T> {
        let cell = self.ring.as_ref()?;
        // SAFETY: `LogFile` is not `Sync`, so there is one thread in here. The
        // `&mut` lives only for the call and nothing it returns borrows from the
        // writer, so no second reference to it can be alive at the same time.
        Some(f(unsafe { &mut *cell.get() }))
    }

    /// How many byte ranges have been handed to the file.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }

    /// How many syncs actually reached the device.
    ///
    /// Group commit is the claim that this stays far below the commit count. In
    /// ring mode it is the ring's count, because that is where the decision to
    /// issue one or skip one is made.
    #[must_use]
    pub fn syncs(&self) -> u64 {
        self.read_ring(RingWriter::syncs).unwrap_or(self.syncs)
    }

    /// The log address below which everything has been handed over, durable or
    /// not.
    #[must_use]
    pub const fn written_upto(&self) -> u64 {
        self.written_upto
    }

    /// How many times a write had to wait for the ring.
    ///
    /// Zero in group mode, because the sync boundary has already drained
    /// everything by the time the next page is staged. Anything else means the
    /// ring is too shallow for the load or the device is behind.
    #[must_use]
    pub fn stalls(&self) -> u64 {
        self.read_ring(RingWriter::stalls).unwrap_or(0)
    }

    /// Submissions the ring has not seen come back.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.read_ring(RingWriter::in_flight).unwrap_or(0)
    }

    /// Waits for everything handed over to become durable.
    ///
    /// A no op without a ring, where a write is already durable by the time
    /// [`PageSink::sync`] returns. With one this is the shutdown and checkpoint
    /// path, and it is what a test calls before it looks at the file.
    ///
    /// # Errors
    ///
    /// The first failure found on the way.
    pub fn drain(&mut self) -> Result<()> {
        match self.ring.as_mut() {
            Some(c) => c.get_mut().drain(),
            None => Ok(()),
        }
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
    /// at damage, and then the whole region comes back so that `yodb check` has
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
        // Whatever was cached for this page is now behind the file. Dropping it
        // is sound here and only here, because `write` holds `&mut self`. It
        // happens before the write in ring mode too, since the write is out of
        // this thread's hands the moment it is submitted.
        self.cache.get_mut().forget(w.page_addr);
        self.writes += 1;
        self.written_upto = self.written_upto.max(w.covers_upto);
        if let Some(c) = self.ring.as_mut() {
            return c
                .get_mut()
                .write(w.page_addr, off + w.offset as u64, w.bytes, w.covers_upto);
        }
        fio::write_at(&self.file, off + w.offset as u64, w.bytes)
            .map_err(|e| io_err("could not write a log page", &e))?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        if let Some(c) = self.ring.as_mut() {
            // Records the address and returns. The ring issues the `fsync` from
            // `poll`, once the writes it has to cover have landed, because
            // io_uring runs a queued sync in parallel with the writes queued
            // before it and a sync that overtakes them makes a durability claim
            // about bytes that are not there.
            return c.get_mut().sync();
        }
        // Nothing has reached the file since the last sync, so there is nothing
        // for this one to make durable. Skipping it is not an optimisation for
        // its own sake: group commit runs off a timer, most ticks find an idle
        // shard, and a real sync costs milliseconds on the devices that matter.
        // Paying that per idle tick would put a floor under the shard's latency
        // that has nothing to do with the work it is doing.
        //
        // Safe because this file has one owner. `written_upto` moves only in
        // `write`, which takes `&mut self`, so equality here means no byte was
        // handed over that the last `sync_data` did not already cover.
        if self.durable_upto == self.written_upto {
            return Ok(());
        }
        fio::sync_data(&self.file).map_err(|e| io_err("could not sync the log", &e))?;
        self.syncs += 1;
        // Exactly what was handed over before the sync, and nothing handed over
        // after it.
        self.durable_upto = self.written_upto;
        Ok(())
    }

    fn durable_upto(&self) -> u64 {
        self.read_ring(RingWriter::durable_upto)
            .unwrap_or(self.durable_upto)
    }

    fn poll(&mut self) -> Result<()> {
        match self.ring.as_mut() {
            Some(c) => c.get_mut().poll(),
            // Without a ring there is nothing outstanding to hear about, since
            // `write` and `sync` are finished by the time they return.
            None => Ok(()),
        }
    }
}

impl PageSource for LogFile {
    fn page_bytes(&self, page_addr: u64) -> Option<&[u8]> {
        // A write to this page may still be in the ring, and a `pread` would
        // then read what is under it rather than what was written. Waiting is
        // the only honest answer, and it costs nothing on the path this is
        // actually on: compaction and recovery read cold pages, which by
        // definition have no write outstanding. A failure stays with the ring
        // and comes out of the next `poll`, because there is no room for one
        // here.
        self.with_ring(RingWriter::quiesce);
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

    fn read_into(&self, page_addr: u64, offset: usize, into: &mut [u8]) -> Option<usize> {
        // A page already in the cache is free to read out of, and a page in the
        // cache is one compaction or recovery is working through right now, so
        // going to the device for it would be a second copy of bytes that are
        // already here.
        //
        // SAFETY: as in `page_bytes`. `LogFile` is not `Sync`, the `&mut Cache`
        // dies at the end of the statement, and the borrow below is derived from
        // a `Box::into_raw` allocation the `Vec` does not own.
        let cached = unsafe { &mut *self.cache.get() }
            .find(page_addr)
            .map(|p| unsafe { &*p });
        if let Some(bytes) = cached {
            if offset >= bytes.len() {
                return Some(0);
            }
            let n = into.len().min(bytes.len() - offset);
            into[..n].copy_from_slice(&bytes[offset..offset + n]);
            return Some(n);
        }
        // Same reason as `page_bytes`: a write still in the ring has not reached
        // the file, and reading under it gives back whatever was there before.
        self.with_ring(RingWriter::quiesce);
        let off = *self.regions.get(&page_addr)?;
        // Not cached. That is the point of this method, and it is why a store
        // larger than memory can be read at all.
        fio::read_at(&self.file, off + offset as u64, into).ok()
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
    fn syncing_an_idle_log_does_not_touch_the_device() {
        let t = Tmp::new("idlesync");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();

        // Nothing written yet, so there is nothing to make durable.
        sink.sync().unwrap();
        sink.sync().unwrap();
        assert_eq!(sink.syncs(), 0, "an empty log has nothing to sync");

        put_page(&mut sink, 0, 0x11);
        sink.sync().unwrap();
        assert_eq!(sink.syncs(), 1);

        // The timer keeps firing on a shard that has gone quiet. Only the first
        // of these had anything behind it.
        for _ in 0..10 {
            sink.sync().unwrap();
        }
        assert_eq!(sink.syncs(), 1, "an idle tick is free");
        assert_eq!(sink.durable_upto(), sink.written_upto);

        // Work again, and the next sync is real again.
        put_page(&mut sink, LOG_PAGE_LEN, 0x22);
        sink.sync().unwrap();
        assert_eq!(sink.syncs(), 2);
    }

    /// Ring mode is a different way to get the bytes there, not a different
    /// result. Same writes, same file, same reads back.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn a_ringed_log_writes_the_same_bytes_as_a_plain_one() {
        let t = Tmp::new("ringsame");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        sink.use_ring(&RingConfig::plain().with_entries(64))
            .unwrap();
        assert!(sink.is_ringed());

        for i in 0..8u64 {
            put_page(&mut sink, i * LOG_PAGE_LEN, 0xc0 + i as u8);
        }
        sink.sync().unwrap();
        sink.drain().unwrap();

        assert_eq!(sink.region_count(), 8);
        assert_eq!(sink.writes(), 8);
        assert_eq!(sink.syncs(), 1, "eight pages, one sync");
        assert_eq!(sink.durable_upto(), sink.written_upto());
        for i in 0..8u64 {
            let got = sink.page_bytes(i * LOG_PAGE_LEN).unwrap();
            assert_eq!(got[PAGE_HEADER_LEN], 0xc0 + i as u8);
            assert_eq!(PageHeader::decode(got).unwrap().used, 64);
        }
    }

    /// The one that would be a silent corruption if `page_bytes` did not wait.
    /// A read of a page with a write still in the ring has to see the write.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn reading_a_page_waits_for_the_write_still_in_the_ring() {
        let t = Tmp::new("ringread");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        sink.use_ring(&RingConfig::plain().with_entries(64))
            .unwrap();

        put_page(&mut sink, 0, 0x11);
        put_page(&mut sink, LOG_PAGE_LEN, 0x22);
        // No drain and no sync. Whatever is outstanding is outstanding.
        assert_eq!(sink.page_bytes(0).unwrap()[PAGE_HEADER_LEN], 0x11);
        assert_eq!(
            sink.page_bytes(LOG_PAGE_LEN).unwrap()[PAGE_HEADER_LEN],
            0x22
        );
        assert_eq!(sink.in_flight(), 0, "the read left something outstanding");
    }

    /// Turns the loop until the sink says `upto` is durable, and gives up
    /// rather than hanging if it never gets there.
    ///
    /// How many turns that takes is a property of the backend and not of the
    /// caller. The portable one does the write and the fsync inside the call
    /// that submits them, so the first poll after a sync already has the
    /// answer. Real io_uring needs at least two: one to pick the write
    /// completion up, which is what lets the fsync go out at all, and another
    /// to pick the fsync completion up. A test that polls once is a test that
    /// only passes off Linux, which is how this one first went out.
    fn poll_until_durable(sink: &mut LogFile, upto: u64) {
        spin_until(|| {
            sink.poll().unwrap();
            sink.durable_upto() >= upto
        });
    }

    /// How long a test waits for a completion before it calls it a failure.
    ///
    /// Thirty seconds, and the number is not about how long an fsync takes. It
    /// is about the difference between a machine that is not going to answer and
    /// one that is busy, and thirty seconds is far past anything the second kind
    /// needs while still being a test that ends.
    const SPIN_LIMIT: std::time::Duration = std::time::Duration::from_secs(30);

    /// Turn `f` until it says yes, and fail rather than hang if it never does.
    ///
    /// Bounded by the clock and not by a count of turns, which is what these
    /// tests used to do and is what made two of them flaky on a CI runner. A
    /// hundred thousand non blocking polls is a few milliseconds of spinning on
    /// a quiet machine, and an fsync against network attached storage on a
    /// loaded shared runner does not come back inside that. The count was a
    /// timeout in disguise, and a timeout measured in the wrong unit.
    ///
    /// The yield is what makes the wait cheap. Spinning at full speed on a
    /// single core runner takes the core away from whatever has to make the
    /// progress being waited for, which turns a slow answer into no answer.
    fn spin_until(mut f: impl FnMut() -> bool) {
        let start = std::time::Instant::now();
        while start.elapsed() < SPIN_LIMIT {
            if f() {
                return;
            }
            std::thread::yield_now();
        }
        panic!("waited {SPIN_LIMIT:?} and it never happened");
    }

    /// A sync in ring mode is a request, not an answer. Nothing is durable
    /// until the poll that picks the fsync up.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn a_ringed_sync_does_not_claim_durability_by_itself() {
        let t = Tmp::new("ringdurable");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        sink.use_ring(&RingConfig::plain()).unwrap();

        put_page(&mut sink, 0, 0x33);
        assert_eq!(sink.written_upto(), 64);
        sink.sync().unwrap();
        assert_eq!(
            sink.durable_upto(),
            0,
            "a sync that has not landed is not one"
        );
        poll_until_durable(&mut sink, 64);

        // And an idle shard still does not touch the device.
        for _ in 0..10 {
            sink.sync().unwrap();
            sink.poll().unwrap();
        }
        assert_eq!(sink.syncs(), 1, "ten idle ticks and one real sync");
    }

    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn a_log_switches_to_the_ring_before_its_first_write_and_not_after() {
        let t = Tmp::new("ringlate");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let mut sink = db.log(0).unwrap();
        sink.use_ring(&RingConfig::plain()).unwrap();
        assert!(sink.use_ring(&RingConfig::plain()).is_err(), "twice");

        let mut plain = db.log(0).unwrap();
        put_page(&mut plain, 0, 0x44);
        assert!(
            plain.use_ring(&RingConfig::plain()).is_err(),
            "after a write"
        );
    }

    /// The whole stack over the ring: records in, checkpoint, reopen, records
    /// out. Group commit still groups, and the parked callers still get their
    /// answer, they just get it from `poll` instead of from `sync`.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn a_ringed_log_survives_being_closed_and_reopened() {
        let t = Tmp::new("ringreopen");
        let keys: Vec<String> = (0..50u32).map(|i| format!("key:{i}")).collect();
        let values: Vec<String> = (0..50u32).map(|i| format!("value number {i}")).collect();

        let (tail, epoch, addrs) = {
            let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
            let mut sink = db.log(0).unwrap();
            sink.use_ring(&RingConfig::plain()).unwrap();
            let mut log = Log::open(cfg(0), sink, 0).unwrap();
            let h = RecordHeader::new(RecordKind::String);
            let mut addrs = Vec::new();
            let mut parked = Vec::new();
            for (k, v) in keys.iter().zip(&values) {
                let a = log.append(&h, k.as_bytes(), v.as_bytes()).unwrap();
                addrs.push(a.addr);
                if let yo_record::CommitAction::WaitFor(at) = a.action {
                    parked.push(at);
                }
            }
            assert_eq!(parked.len(), 50, "group mode answers nobody early");

            log.advance_epoch();
            log.commit_pending().unwrap();
            let last = parked.iter().copied().max().unwrap();
            spin_until(|| {
                log.poll().unwrap();
                log.durable_upto() >= last
            });
            assert!(
                parked.iter().all(|&at| at <= log.durable_upto()),
                "somebody is still parked after the commit landed"
            );

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

        let sink = db.log(0).unwrap();
        let log = Log::recover(cfg(0), sink, entry.log_tail).unwrap();
        for ((a, k), v) in addrs.iter().zip(&keys).zip(&values) {
            let r = log.read(*a).unwrap();
            assert_eq!(r.key, k.as_bytes());
            assert_eq!(r.value, v.as_bytes());
        }
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

    /// The thing a store larger than memory is for. Most of these records are
    /// in pages that left the resident window long ago, `read` says so, and
    /// every one of them still comes back byte for byte off the file.
    #[test]
    fn a_record_whose_page_left_the_window_still_reads_off_the_file() {
        let t = Tmp::new("faultin");
        let mut db = Yo::create(&t.0, &CreateOptions::default()).unwrap();
        let small = LogConfig {
            page_len: 8192,
            resident_pages: 2,
            ..cfg(0)
        };
        let mut log = Log::open(small, db.log(0).unwrap(), 0).unwrap();

        let h = RecordHeader::new(RecordKind::CollectionChunk);
        let mut wrote = Vec::new();
        for i in 0..400u32 {
            let v = format!("value number {i}, ").repeat(6);
            let addr = log.append(&h, b"", v.as_bytes()).unwrap().addr;
            wrote.push((addr, v));
        }
        log.commit_pending().unwrap();

        let mut gone = 0;
        let mut out = Vec::new();
        for (addr, want) in &wrote {
            if log.read(*addr).is_err() {
                gone += 1;
            }
            out.clear();
            log.read_value_into(*addr, &mut out).unwrap();
            assert_eq!(
                out,
                want.as_bytes(),
                "the record at {addr} did not come back the same"
            );
        }
        assert!(
            gone > 300,
            "only {gone} of the 400 were out of the window, so this mostly read memory"
        );
        assert_eq!(
            log.sink().cache_bytes(),
            0,
            "point reads filled the page cache, which is what makes a store larger than memory unreadable"
        );
    }
}
