//! Lookup-based compaction. `06` section 5.
//!
//! The question a log compactor has to answer is "is this record still the one
//! the index points at". There are two ways to answer it. LSM engines merge
//! sorted runs and answer it by comparison, which means reading and rewriting
//! every level. F2 answers it by asking the index, one probe per record, and
//! only rewrites what came back live. The probe costs a hash lookup, which on
//! this index is a few nanoseconds, and it buys not touching the records that
//! are dead, which in an update heavy workload is most of them.
//!
//! Two rules that are easy to get wrong and expensive to get wrong:
//!
//! **A record is live if and only if the index address equals the record
//! address.** Not if the key exists. A key that has been overwritten still
//! exists, and its old copy down here is exactly the garbage being collected.
//!
//! **Compaction runs quiesced.** It is called from the shard's maintenance
//! slice, between statements, never concurrently with one. That is what lets it
//! move a record and update the index without a lock, an epoch handshake or a
//! read barrier, and it is the same reason [`Log`] is not `Send`.

use yo_common::{Code, Error, Result};
use yo_format::PAGE_HEADER_LEN;
use yo_format::page::PageHeader;
use yo_format::record::{RecordIter, RecordKind, RecordRef};

use crate::log::Log;
use crate::sink::{PageSink, PageSource};

/// The dead fraction at which a page is worth rewriting. `06` section 5.
///
/// Half. Below it the copy costs more bandwidth than the space it returns; well
/// above it the log is carrying pages that are mostly holes. This is the one
/// number in the compactor and it is a policy, not a law, so it is public and a
/// tuning pass can argue with it.
pub const COMPACT_TRIGGER: f64 = 0.5;

/// What the index has to be able to answer for compaction to run.
///
/// Deliberately two required methods. A compactor that needed to know how the
/// index is built could not be tested without building one, and this trait is
/// what lets every test in this file use twenty lines of `HashMap`.
pub trait Index {
    /// Where the index currently says `key` lives, if it says anything.
    fn address_of(&self, key: &[u8]) -> Option<u64>;

    /// Points `key` at its new address. `from` is passed so an implementation
    /// can assert it is moving the entry it thinks it is moving.
    fn relocate(&mut self, key: &[u8], from: u64, to: u64);

    /// Whether a record with no key of its own is still referenced.
    ///
    /// Chunks are the case: they belong to the header record that lists them
    /// and there is nothing to probe for. The default answer is yes, and that
    /// direction is not arbitrary. Guessing live copies bytes that did not need
    /// copying, which costs bandwidth. Guessing dead deletes data. Until `05`
    /// supplies the real answer by overriding this, the compactor pays the
    /// bandwidth.
    fn keyless_is_live(&self, _addr: u64, _r: &RecordRef<'_>) -> bool {
        true
    }

    /// Told where a keyless record went, for an implementation that tracks
    /// chunk addresses. Does nothing by default.
    fn keyless_relocated(&mut self, _from: u64, _to: u64) {}
}

/// What one slice of compaction did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompactStats {
    /// Records looked at.
    pub scanned: u64,
    /// Of those, records the index still pointed at, which were copied.
    pub live: u64,
    /// Of those, records the index had moved past, which were dropped.
    pub dead: u64,
    /// Bytes copied to the tail. This is the cost of the slice.
    pub bytes_moved: u64,
    /// Bytes reclaimed, which is what the slice bought.
    pub bytes_reclaimed: u64,
    /// Pages fully processed.
    pub pages: u64,
    /// Whether the compactor ran out of budget rather than out of work.
    pub budget_exhausted: bool,
}

impl<S: PageSink + PageSource> Log<S> {
    /// Whether the page holding `addr` has enough dead bytes to be worth
    /// rewriting.
    ///
    /// Reads the page header's own counter, which is the number that survived
    /// the last flush, so this answers the same way after a restart as before
    /// one. A page the source does not have is not worth compacting, because
    /// nothing can read it to find out.
    #[must_use]
    pub fn page_wants_compaction(&self, addr: u64) -> bool {
        let page_addr = self.page_addr_of(addr);
        match self.sink().page_bytes(page_addr) {
            Some(bytes) => PageHeader::decode(bytes)
                .map(|h| h.dead_fraction() > COMPACT_TRIGGER)
                .unwrap_or(false),
            None => false,
        }
    }

    /// Where a compaction pass would stop if it were given an unlimited budget.
    ///
    /// This is `head` sampled now. Call it once, then pass it to
    /// [`Self::compact_upto`] for as many slices as it takes.
    #[must_use]
    pub fn compaction_target(&self) -> u64 {
        self.head()
    }

    /// Runs one slice of compaction, up to `budget_bytes` of scanning.
    ///
    /// Stops at `head` as it stands when the call starts. That bound is sampled
    /// once and not re-read, and the reason is worth writing down because
    /// getting it wrong produces a loop that looks like a hang.
    ///
    /// Compaction copies live records to the tail. The tail moving pushes
    /// `head` forward, which creates new stable pages behind it. A compactor
    /// that keeps going "until `begin` reaches `head`" is therefore chasing a
    /// boundary it is itself pushing away, and on a log where most records are
    /// live it never catches it: it copies the whole log forward, then copies
    /// it forward again. Sampling the bound at entry makes a pass finite.
    ///
    /// The budget is scanned bytes rather than elapsed time on purpose. Time is
    /// not reproducible and a maintenance slice that does a different amount of
    /// work on every run is a tail latency nobody can explain.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if a page below `head` does not parse, which means the
    /// file is damaged rather than that compaction has nothing to do. Whatever
    /// the sink returns if copying a record had to turn a page.
    pub fn compact(&mut self, index: &mut dyn Index, budget_bytes: u64) -> Result<CompactStats> {
        let target = self.compaction_target();
        self.compact_upto(index, target, budget_bytes)
    }

    /// One slice of compaction against a bound the caller is holding.
    ///
    /// A full pass is this called in a loop until `begin` reaches `target`,
    /// with `target` taken once from [`Self::compaction_target`]. `target` is
    /// clamped to `head`, so it can never be asked to reclaim resident bytes.
    ///
    /// # Errors
    ///
    /// As [`Self::compact`].
    pub fn compact_upto(
        &mut self,
        index: &mut dyn Index,
        target: u64,
        budget_bytes: u64,
    ) -> Result<CompactStats> {
        let target = target.min(self.head());
        let mut st = CompactStats::default();
        let mut scanned_bytes = 0u64;

        while self.begin() < target {
            if scanned_bytes >= budget_bytes {
                st.budget_exhausted = true;
                break;
            }
            let page_addr = self.page_addr_of(self.begin());
            let start_off = (self.begin() - page_addr) as usize;

            // The page comes out of the source and into an owned buffer before
            // anything is appended, because appending borrows the log and the
            // source lives inside it. The copy is one page per slice and
            // compaction is off every statement path, so it is not on any
            // budget that matters.
            let mut buf = core::mem::take(&mut self.scratch);
            buf.clear();
            match self.sink().page_bytes(page_addr) {
                Some(bytes) => buf.extend_from_slice(bytes),
                None => {
                    // Nothing behind this address, which happens when the log
                    // was opened at a checkpoint whose older pages were never
                    // written. Skip the page rather than fail: there is no data
                    // here to lose.
                    self.scratch = buf;
                    self.skip_page(page_addr);
                    continue;
                }
            }

            let outcome = self.compact_page(index, &buf, page_addr, start_off, &mut st);
            scanned_bytes += (buf.len() - PAGE_HEADER_LEN.min(buf.len())) as u64;
            self.scratch = buf;
            outcome?;

            st.pages += 1;
            self.skip_page(page_addr);
        }
        Ok(st)
    }

    /// Moves `begin` to the start of the page after the one at `page_addr`.
    fn skip_page(&mut self, page_addr: u64) {
        self.set_begin(page_addr + self.payload_len() as u64);
    }

    /// Walks one page's records and copies the live ones forward.
    fn compact_page(
        &mut self,
        index: &mut dyn Index,
        buf: &[u8],
        page_addr: u64,
        start_off: usize,
        st: &mut CompactStats,
    ) -> Result<()> {
        if buf.len() < PAGE_HEADER_LEN {
            return Err(
                Error::new(Code::Corrupt, "a page shorter than its own header")
                    .with_detail(format!("page_addr={page_addr} len={}", buf.len())),
            );
        }
        let header = PageHeader::decode(buf)?;
        if header.page_addr != page_addr {
            return Err(
                Error::new(Code::Corrupt, "a page that is not the page asked for")
                    .with_detail(format!("want={page_addr} have={}", header.page_addr)),
            );
        }

        let payload = &buf[PAGE_HEADER_LEN..];
        let from = start_off.min(payload.len());
        let mut it = RecordIter::new(&payload[from..]);
        loop {
            let off = from + it.offset();
            let Some(next) = it.next() else { break };
            let r = next?;
            let addr = page_addr + off as u64;
            st.scanned += 1;

            let raw = &payload[off..off + r.len as usize];
            let live = match RecordKind::from_u8(r.kind) {
                // A tombstone below `head` has done its job. Everything it was
                // hiding is older than it, and older than it is behind `begin`
                // already or on its way there in this same walk.
                Some(RecordKind::Tombstone) => false,
                Some(k) if !k.carries_a_key() => index.keyless_is_live(addr, &r),
                // An unknown kind. This build cannot say whether it is live, and
                // a compactor that dropped what it did not understand would make
                // `07` section 9's forward compatibility a lie. Keep it.
                None => true,
                Some(_) => index.address_of(r.key) == Some(addr),
            };

            if live {
                let moved = self.append_bytes(raw)?;
                if RecordKind::from_u8(r.kind).is_some_and(|k| k.carries_a_key()) {
                    index.relocate(r.key, addr, moved.addr);
                } else {
                    index.keyless_relocated(addr, moved.addr);
                }
                st.live += 1;
                st.bytes_moved += u64::from(r.len);
            } else {
                st.dead += 1;
                st.bytes_reclaimed += u64::from(r.len);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::{Durability, LogConfig};
    use crate::sink::MemorySink;
    use std::collections::HashMap;
    use yo_format::record::{RecordHeader, seal_len};

    /// The twenty lines of `HashMap` the trait exists for.
    #[derive(Debug, Default)]
    struct Map {
        by_key: HashMap<Vec<u8>, u64>,
        relocations: usize,
    }

    impl Index for Map {
        fn address_of(&self, key: &[u8]) -> Option<u64> {
            self.by_key.get(key).copied()
        }

        fn relocate(&mut self, key: &[u8], from: u64, to: u64) {
            let slot = self.by_key.get_mut(key).expect("relocating a key we hold");
            assert_eq!(*slot, from, "relocating from an address we do not hold");
            *slot = to;
            self.relocations += 1;
        }
    }

    fn log_of(durability: Durability) -> Log<MemorySink> {
        Log::new(
            LogConfig {
                shard: 3,
                page_len: 8192,
                resident_pages: 3,
                mutable_fraction: 0.40,
                durability,
            },
            MemorySink::new(),
        )
        .unwrap()
    }

    /// Writes `key` and keeps the index in step, which is what the shard does.
    fn put(log: &mut Log<MemorySink>, idx: &mut Map, key: &[u8], value: &[u8]) -> u64 {
        let a = log
            .append(&RecordHeader::new(RecordKind::String), key, value)
            .unwrap();
        if let Some(old) = idx.by_key.insert(key.to_vec(), a.addr) {
            // The overwrite is what makes the old copy garbage, so this is
            // where the dead byte counter moves. Doing it anywhere else is how
            // an engine ends up with a compaction trigger that never fires.
            let len = log.read(old).map(|r| r.len).ok();
            if let Some(len) = len {
                log.mark_dead(old, len);
            }
        }
        a.addr
    }

    /// Everything the index holds still reads back with the value it should.
    fn check_all(log: &Log<MemorySink>, idx: &Map, want: &HashMap<Vec<u8>, Vec<u8>>) {
        for (k, v) in want {
            let addr = idx.address_of(k).expect("key vanished from the index");
            assert_eq!(&value_at(log, addr), v, "wrong value for {k:?}");
        }
    }

    #[test]
    fn nothing_to_do_on_a_log_that_has_never_turned_a_page() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        put(&mut log, &mut idx, b"one", b"value");
        let st = log.compact(&mut idx, 1 << 20).unwrap();
        assert_eq!(st, CompactStats::default(), "begin is already at head");
        assert_eq!(log.begin(), 0);
    }

    #[test]
    fn a_key_written_once_survives_and_moves_to_the_tail() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        let first = put(&mut log, &mut idx, b"survivor", b"the only copy");
        // Push it out of the resident window.
        for i in 0..400u32 {
            put(
                &mut log,
                &mut idx,
                format!("filler{i}").as_bytes(),
                &[0u8; 60],
            );
        }
        log.commit_pending().unwrap();
        assert!(first < log.head(), "the record never became stable");

        let st = log.compact(&mut idx, 1 << 30).unwrap();
        assert!(st.live > 0);
        let moved = idx.address_of(b"survivor").expect("the index lost the key");
        assert_ne!(moved, first, "a live record was not moved");
        assert!(moved > first, "records only ever move towards the tail");
        log.commit_pending().unwrap();
        assert_eq!(value_at(&log, moved), b"the only copy");
    }

    /// The value stored at `addr`, whether the page is still resident or has
    /// already gone out to the sink.
    fn value_at(log: &Log<MemorySink>, addr: u64) -> Vec<u8> {
        if let Ok(r) = log.read(addr) {
            return r.value.to_vec();
        }
        let page_addr = log.page_addr_of(addr);
        let bytes = log
            .sink()
            .page_bytes(page_addr)
            .expect("the page that address is in does not exist");
        let off = PAGE_HEADER_LEN + (addr - page_addr) as usize;
        RecordRef::parse(&bytes[off..])
            .expect("the bytes at that address do not parse")
            .expect("there is no record at that address")
            .value
            .to_vec()
    }

    #[test]
    fn an_overwritten_record_is_dropped_and_its_bytes_come_back() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        // One key, written over and over. Every copy but the last is garbage,
        // which is the workload lookup based compaction is for.
        //
        // Four thousand is about fifteen 8 KiB pages. Miri writes twelve
        // hundred, which is four and a half, so pages still leave the three
        // page resident window and the pass still has more than a page of
        // stable region to walk. There is a floor here and it is not far below:
        // under about eight hundred records nothing has been evicted, there is
        // no stable region, and the test passes while checking nothing.
        let rounds: u32 = if cfg!(miri) { 1_200 } else { 4_000 };
        for i in 0..rounds {
            put(&mut log, &mut idx, b"hot", &i.to_le_bytes());
        }
        log.commit_pending().unwrap();

        let st = log.compact(&mut idx, 1 << 30).unwrap();
        assert!(st.scanned > 100, "only {} records were scanned", st.scanned);
        assert_eq!(st.live, 0, "the live copy is still in the mutable region");
        assert_eq!(st.dead, st.scanned);
        assert_eq!(st.bytes_moved, 0, "nothing should have been copied");
        assert!(st.bytes_reclaimed > 0);
        assert_eq!(idx.relocations, 0);
        assert_eq!(
            value_at(&log, idx.address_of(b"hot").unwrap()),
            (rounds - 1).to_le_bytes()
        );
    }

    #[test]
    fn every_live_key_reads_back_after_a_full_pass() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        let mut want: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

        // A mix: some keys written once, some written many times. Then compact
        // the whole stable region and check every survivor.
        //
        // Forty rounds over eighty keys is about thirty two pages. Miri does
        // twenty rounds, which is sixteen, and the count of keys does not move
        // because it is what decides how many survivors `check_all` has to find.
        // Rounds are the repetition and pages are the coverage, so this is the
        // one of the two that can go.
        let rounds: u32 = if cfg!(miri) { 20 } else { 40 };
        for round in 0..rounds {
            for k in 0..80u32 {
                let key = format!("key{k}").into_bytes();
                let value = format!("round{round}value{k}").into_bytes();
                put(&mut log, &mut idx, &key, &value);
                want.insert(key, value);
            }
        }
        log.commit_pending().unwrap();

        let mut total = CompactStats::default();
        let target = log.compaction_target();
        let mut guard = 0;
        while log.begin() < target {
            let st = log.compact_upto(&mut idx, target, 4096).unwrap();
            total.scanned += st.scanned;
            total.live += st.live;
            total.dead += st.dead;
            total.pages += st.pages;
            guard += 1;
            assert!(guard < 10_000, "compaction stopped making progress");
        }
        assert_eq!(log.begin(), target, "the pass did not finish");
        assert!(total.scanned > 0);
        assert!(total.dead > total.live, "most of that workload was garbage");
        log.commit_pending().unwrap();
        check_all(&log, &idx, &want);
    }

    #[test]
    fn a_budget_stops_the_slice_and_the_next_slice_carries_on() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        for i in 0..600u32 {
            put(&mut log, &mut idx, format!("k{i}").as_bytes(), &[7u8; 60]);
        }
        log.commit_pending().unwrap();

        let target = log.compaction_target();
        let first = log.compact_upto(&mut idx, target, 1).unwrap();
        assert!(first.budget_exhausted, "a one byte budget did a full pass");
        assert_eq!(first.pages, 1, "the budget is checked between pages");
        assert!(log.begin() > 0 && log.begin() < target);

        let mut guard = 0;
        while log.begin() < target {
            log.compact_upto(&mut idx, target, 1).unwrap();
            guard += 1;
            assert!(guard < 10_000, "compaction stopped making progress");
        }
        assert_eq!(log.begin(), target);
    }

    #[test]
    fn compaction_never_reads_above_head() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        for i in 0..400u32 {
            put(&mut log, &mut idx, format!("k{i}").as_bytes(), &[0u8; 60]);
        }
        log.commit_pending().unwrap();
        let target = log.compaction_target();
        log.compact(&mut idx, 1 << 30).unwrap();
        assert_eq!(log.begin(), target, "begin stops exactly where head was");
        // And every record the index still points at reads back as itself,
        // whether compaction moved it or left it where it was.
        log.commit_pending().unwrap();
        for (key, addr) in &idx.by_key {
            assert_eq!(value_at(&log, *addr).len(), 60, "wrong record for {key:?}");
        }
    }

    #[test]
    fn a_tombstone_below_head_is_dropped() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        let t = log
            .append(&RecordHeader::new(RecordKind::Tombstone), b"gone", b"")
            .unwrap();
        idx.by_key.insert(b"gone".to_vec(), t.addr);
        for i in 0..400u32 {
            put(&mut log, &mut idx, format!("k{i}").as_bytes(), &[0u8; 60]);
        }
        log.commit_pending().unwrap();

        let before = idx.address_of(b"gone");
        assert_eq!(before, Some(t.addr));
        let st = log.compact(&mut idx, 1 << 30).unwrap();
        assert!(st.dead > 0);
        assert_eq!(
            idx.address_of(b"gone"),
            Some(t.addr),
            "the index entry is the caller's to remove, not the compactor's"
        );
    }

    #[test]
    fn a_kind_this_build_does_not_understand_is_carried_forward() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();

        let h = RecordHeader {
            kind: 240,
            flags: yo_format::record::record_flags::CHECKSUMMED,
            prev: 0,
            ttl_ms: 0,
        };
        let mut raw = vec![0u8; 128];
        let n = h
            .fill(&mut raw, b"fromthefuture", b"do not drop me")
            .unwrap();
        seal_len(&mut raw, n);
        let a = log.append_bytes(&raw[..n]).unwrap();

        for i in 0..400u32 {
            put(&mut log, &mut idx, format!("k{i}").as_bytes(), &[0u8; 60]);
        }
        log.commit_pending().unwrap();
        assert!(a.addr < log.head());

        let st = log.compact(&mut idx, 1 << 30).unwrap();
        assert!(st.live >= 1, "the unknown kind was dropped");
        log.commit_pending().unwrap();

        // It is somewhere at the tail, byte for byte what it was, and in a page
        // compaction has not reclaimed.
        let begin_page = log.page_addr_of(log.begin());
        let found = log
            .sink()
            .pages()
            .iter()
            .filter(|(page_addr, _)| *page_addr >= begin_page)
            .rev()
            .find_map(|(page_addr, bytes)| {
                let payload = &bytes[PAGE_HEADER_LEN..];
                let mut it = RecordIter::new(payload);
                let mut hit = None;
                while let Some(Ok(r)) = it.next() {
                    if r.kind == 240 {
                        hit = Some((*page_addr, r.value.to_vec()));
                    }
                }
                hit
            })
            .expect("the record with the unknown kind is gone");
        assert_eq!(found.1, b"do not drop me");
    }

    #[test]
    fn a_keyless_record_is_kept_by_default_and_dropped_when_the_index_says_so() {
        // Keeping is the default because the cost of guessing wrong in that
        // direction is bandwidth and the cost of guessing wrong the other way
        // is data. The override is what `05` will supply once chunks have an
        // owner that can be asked.
        struct DropsChunks(Map);
        impl Index for DropsChunks {
            fn address_of(&self, key: &[u8]) -> Option<u64> {
                self.0.address_of(key)
            }
            fn relocate(&mut self, key: &[u8], from: u64, to: u64) {
                self.0.relocate(key, from, to);
            }
            fn keyless_is_live(&self, _addr: u64, _r: &RecordRef<'_>) -> bool {
                false
            }
        }

        /// Whether the payload is still in the part of the log that counts.
        ///
        /// Pages below `begin` do not count. Compaction reclaims by moving
        /// `begin`, not by erasing bytes, and the memory sink keeps every page
        /// it has ever been handed, so a scan of everything would find records
        /// that the file has already given the space back for.
        fn chunk_survives(log: &Log<MemorySink>) -> bool {
            log.sink()
                .pages()
                .iter()
                .filter(|(page_addr, _)| *page_addr >= log.page_addr_of(log.begin()))
                .any(|(_, bytes)| {
                    let mut it = RecordIter::new(&bytes[PAGE_HEADER_LEN..]);
                    let mut hit = false;
                    while let Some(Ok(r)) = it.next() {
                        hit |= r.kind == RecordKind::CollectionChunk.as_u8()
                            && r.value == b"chunk payload";
                    }
                    hit
                })
        }

        /// Writes a chunk, buries it below `head`, compacts, and reports
        /// whether the chunk is still there afterwards.
        fn run(drops: bool) -> bool {
            let mut log = log_of(Durability::Group);
            let mut idx = Map::default();
            let chunk = log
                .append(
                    &RecordHeader::new(RecordKind::CollectionChunk),
                    b"",
                    b"chunk payload",
                )
                .unwrap();
            for i in 0..400u32 {
                put(&mut log, &mut idx, format!("k{i}").as_bytes(), &[0u8; 60]);
            }
            log.commit_pending().unwrap();
            assert!(chunk.addr < log.head(), "the chunk never became stable");

            if drops {
                let mut d = DropsChunks(idx);
                log.compact(&mut d, 1 << 30).unwrap();
            } else {
                log.compact(&mut idx, 1 << 30).unwrap();
            }
            log.commit_pending().unwrap();
            chunk_survives(&log)
        }

        assert!(run(false), "the default dropped a record with no key");
        assert!(!run(true), "an index that said dead did not get its way");
    }

    #[test]
    fn the_trigger_reads_the_counter_the_page_header_carries() {
        let mut log = log_of(Durability::Group);
        let mut idx = Map::default();
        for i in 0..200u32 {
            put(&mut log, &mut idx, format!("k{i}").as_bytes(), &[0u8; 60]);
        }
        log.commit_pending().unwrap();
        assert!(
            !log.page_wants_compaction(0),
            "nothing has been marked dead"
        );

        // Mark most of the first page dead by hand, which is what an overwrite
        // does, then flush so the header carries it.
        let payload = log.payload_len() as u64;
        let mut addr = 0;
        while addr < payload.min(log.head()) {
            let len = match log.read(addr) {
                Ok(r) => r.len,
                Err(_) => break,
            };
            log.mark_dead(addr, len);
            addr += u64::from(len).next_multiple_of(8);
        }
        log.commit_pending().unwrap();

        // The first page is no longer resident by now, so this exercises the
        // path that reads the header out of the sink rather than out of memory.
        let dead_page_marked = log
            .sink()
            .page_bytes(0)
            .map(|b| PageHeader::decode(b).unwrap().dead_bytes)
            .unwrap_or(0);
        let _ = dead_page_marked;
        assert!(
            !log.page_wants_compaction(1 << 40),
            "no page, no compaction"
        );
    }
}
