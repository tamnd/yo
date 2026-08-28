//! The hybrid log itself.
//!
//! `06` section 2. One per shard, owned by that shard, and not `Send`, which is
//! not a limitation but the design: if a log could be sent, two threads could
//! append to it, and then installing a record would need a compare and swap.
//! FASTER needs that CAS because several threads race to install. This log has
//! one writer, so installing a record is a store, and that difference is worth
//! more than everything else in this file put together.
//!
//! ```text
//!   <-- older                                                     newer -->
//!   +--------------+---------------+------------------+-----------------+
//!   |   stable     |   read-only   |     mutable      |   unallocated   |
//!   +--------------+---------------+------------------+-----------------+
//!   ^              ^               ^                  ^
//!   begin      head_address    read_only_address     tail_address
//! ```
//!
//! Addresses are payload addresses. Page headers are physical and do not
//! consume log address space, so page `k` covers exactly
//! `[k * payload_len, (k+1) * payload_len)` and the arithmetic is a divide
//! rather than a divide with a correction. This is also what makes the page
//! header's `page_addr` field mean what `07` section 4 says it means, which is
//! the log address of byte zero of the payload.
//!
//! **The tail sentinel.** After every append the four bytes where the next
//! record's `len` will go are set to zero, and a fresh page starts with those
//! four bytes zeroed. That is what makes `len == 0` mean end of log without
//! anybody having to zero a 32 MiB page on reuse, which would cost about three
//! milliseconds of pure memory bandwidth every time the log turns a page.
//!
//! The sentinel is only half of it. Writes go out a block at a time, so a flush
//! part way through a page also sends the rest of the block, and that part of
//! the buffer still holds the previous tenant of the ring slot. A store must
//! never be given those bytes: a record's checksum covers its own bytes and says
//! nothing about the address they belong at, so an old record landing at the
//! same offset in a new page parses, and a reader walks into it. So the tail of
//! the block past the sentinel is zeroed on the way out. Under a block per
//! flush, against a page turn that would be eight thousand times that.

use core::marker::PhantomData;
use core::sync::atomic::{Ordering, fence};

use yo_common::{Code, Error, Result};
use yo_format::page::PAGE_HEADER_LEN;
use yo_format::record::{RecordHeader, RecordRef, seal_len, total_len};
use yo_format::superblock::CheckpointEntry;
use yo_format::{LOG_PAGE_LEN, PageHeader, align_up};

use crate::sink::{PageSink, PageSource, PageWrite};

/// How many 32 MiB pages stay in memory. F2's three, which is about 96 MiB.
pub const DEFAULT_RESIDENT_PAGES: usize = 3;

/// How much of the resident window updates in place. `06` section 2's 40%.
pub const DEFAULT_MUTABLE_FRACTION: f64 = 0.40;

/// The unit a write is aligned to, which is the torn write unit `07` section 1
/// assumes and nothing stronger.
pub const FLUSH_BLOCK: usize = 4096;

/// What a commit is worth, and what it costs. `06` section 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    /// Append to the page in memory and reply. Nothing reaches the store until
    /// a page fills.
    None,
    /// Append, and write at page boundaries without asking for a sync. The
    /// operating system decides when it lands, so a process crash is survived
    /// and a machine crash is not.
    ///
    /// With an asynchronous sink the page boundary hands the bytes over and
    /// submits them, and nothing waits for the completion. That is the same
    /// promise reached the same way: the write is the kernel's problem from
    /// there, and this mode never claimed to know when the kernel gets to it.
    Os,
    /// Append, and hold the reply until the containing page has been synced.
    /// One sync serves every commit in the page, which is the entire difference
    /// between 267 commits a second and the number this milestone is gated on.
    #[default]
    Group,
    /// Append and sync, per commit. Correct, slow, and explicitly not a gate
    /// row.
    Sync,
}

impl Durability {
    /// The name this mode is published under, which every number that quotes a
    /// commit rate has to state.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Durability::None => "none",
            Durability::Os => "os",
            Durability::Group => "group",
            Durability::Sync => "sync",
        }
    }

    /// The mode named by `s`, if it is one.
    #[must_use]
    pub fn parse(s: &str) -> Option<Durability> {
        match s {
            "none" => Some(Durability::None),
            "os" => Some(Durability::Os),
            "group" => Some(Durability::Group),
            "sync" => Some(Durability::Sync),
            _ => None,
        }
    }
}

/// Which of the four regions an address falls in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Below `begin`. Compaction has been through and the bytes are gone. Not
    /// in the diagram in `06` because the diagram starts at `begin`, but a
    /// stale index entry can still name an address down here and the answer has
    /// to be something other than a panic.
    Reclaimed,
    /// In the file, not in memory. Reading it faults a page.
    Stable,
    /// In memory, but updates go to the tail rather than in place.
    ReadOnly,
    /// In memory and updatable in place.
    Mutable,
    /// At or past the tail. Nothing has been written here.
    Unallocated,
}

/// What the caller has to do before it replies to whoever asked for the write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitAction {
    /// Reply now.
    Reply,
    /// Park until [`Log::durable_upto`] is at or past this address.
    WaitFor(u64),
}

/// The result of an append.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Append {
    /// Where the record went. This is what the index stores.
    pub addr: u64,
    /// The record's exact length, trailer included.
    pub len: u32,
    /// Whether the caller may reply yet.
    pub action: CommitAction,
}

/// How a log is shaped.
#[derive(Debug, Clone, Copy)]
pub struct LogConfig {
    /// Which shard this log belongs to. Stamped into every page header, so that
    /// a segment that somehow ended up owned by two shards is caught on read
    /// rather than believed.
    pub shard: u32,
    /// The physical page size, header included. [`LOG_PAGE_LEN`] in production.
    /// Tests use something small, because the behaviour under test is what
    /// happens at a page boundary and a test should be able to reach one.
    pub page_len: usize,
    /// How many pages stay in memory.
    pub resident_pages: usize,
    /// What fraction of the resident window updates in place.
    pub mutable_fraction: f64,
    /// What a commit is worth.
    pub durability: Durability,
}

impl Default for LogConfig {
    fn default() -> LogConfig {
        LogConfig {
            shard: 0,
            page_len: LOG_PAGE_LEN as usize,
            resident_pages: DEFAULT_RESIDENT_PAGES,
            mutable_fraction: DEFAULT_MUTABLE_FRACTION,
            durability: Durability::Group,
        }
    }
}

impl LogConfig {
    /// Rejects a shape that cannot work, at construction, with a reason.
    fn validate(&self) -> Result<()> {
        if self.page_len <= PAGE_HEADER_LEN + FLUSH_BLOCK {
            return Err(
                Error::new(Code::Invalid, "a page has to hold more than its header")
                    .with_detail(format!("page_len={}", self.page_len)),
            );
        }
        if !self.page_len.is_multiple_of(FLUSH_BLOCK) {
            return Err(
                Error::new(Code::Invalid, "a page is a whole number of 4 KiB blocks")
                    .with_detail(format!("page_len={}", self.page_len)),
            );
        }
        if self.resident_pages < 2 {
            // One resident page means the page being appended to is also the
            // page being evicted, and the read-only region has nowhere to live.
            return Err(Error::new(
                Code::Invalid,
                "at least two pages have to be resident",
            ));
        }
        if !(0.0..=1.0).contains(&self.mutable_fraction) {
            return Err(Error::new(
                Code::Invalid,
                "the mutable fraction is between 0 and 1",
            ));
        }
        Ok(())
    }
}

/// One resident page.
struct Page {
    buf: Box<[u8]>,
    /// The log address of payload byte zero, or `None` for an empty slot.
    page_addr: Option<u64>,
    /// Payload bytes claimed by records.
    used: u32,
    /// Of those, how many belong to records the index has moved away from.
    dead: u32,
    /// The physical offset up to which this page has been handed to the sink.
    flushed: usize,
    /// Whether anything has been appended since the last flush.
    dirty: bool,
}

/// The per shard hybrid log.
///
/// Not `Send` and not `Sync`, deliberately. See the module documentation. That
/// is a load bearing property rather than an oversight, so it is checked:
///
/// ```compile_fail
/// use yo_record::{Durability, Log, LogConfig};
/// use yo_record::sink::MemorySink;
///
/// fn needs_send<T: Send>(_: T) {}
///
/// let cfg = LogConfig { page_len: 8192, durability: Durability::None, ..LogConfig::default() };
/// let log = Log::new(cfg, MemorySink::new()).unwrap();
/// needs_send(log);
/// ```
pub struct Log<S: PageSink> {
    cfg: LogConfig,
    payload_len: usize,
    sink: S,
    pages: Vec<Page>,
    begin: u64,
    head: u64,
    read_only: u64,
    tail: u64,
    epoch: u32,
    appends: u64,
    /// Where a record is staged when compaction copies it, so that reading the
    /// old page and writing the new one are not two live borrows of the same
    /// vector. Compaction is quiesced and budgeted, so a memcpy through here is
    /// off every statement path.
    pub(crate) scratch: Vec<u8>,
    _not_send: PhantomData<*const ()>,
}

impl<S: PageSink> core::fmt::Debug for Log<S> {
    /// The four addresses and the epoch. Not the page buffers, which are 96 MiB
    /// of mostly zeroes and would turn one assertion failure into a scroll.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Log")
            .field("shard", &self.cfg.shard)
            .field("begin", &self.begin)
            .field("head", &self.head)
            .field("read_only", &self.read_only)
            .field("tail", &self.tail)
            .field("epoch", &self.epoch)
            .field("appends", &self.appends)
            .field("durability", &self.cfg.durability)
            .finish_non_exhaustive()
    }
}

impl<S: PageSink> Log<S> {
    /// A new, empty log.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the configuration cannot work, with the field that
    /// is wrong in the detail.
    pub fn new(cfg: LogConfig, sink: S) -> Result<Log<S>> {
        Log::open(cfg, sink, 0)
    }

    /// A log that continues from `at`, with an empty tail page.
    ///
    /// Right when `at` is the start of a page, which is the case for a fresh
    /// file and for a checkpoint taken at a page boundary. Wrong when `at` is
    /// in the middle of a page that already holds records, because the tail
    /// page starts out zeroed here and the first flush would write those zeroes
    /// over the records in front of it. Use [`Log::recover`] for that, which is
    /// the case recovery actually hits.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the configuration cannot work.
    pub fn open(cfg: LogConfig, sink: S, at: u64) -> Result<Log<S>> {
        cfg.validate()?;
        let payload_len = cfg.page_len - PAGE_HEADER_LEN;
        let pages = (0..cfg.resident_pages)
            .map(|_| Page {
                buf: vec![0u8; cfg.page_len].into_boxed_slice(),
                page_addr: None,
                used: 0,
                dead: 0,
                flushed: 0,
                dirty: false,
            })
            .collect();
        let mut log = Log {
            cfg,
            payload_len,
            sink,
            pages,
            begin: at,
            head: at,
            read_only: at,
            tail: at,
            epoch: 0,
            appends: 0,
            scratch: Vec::new(),
            _not_send: PhantomData,
        };
        log.open_tail_page();
        Ok(log)
    }

    // -- accessors ----------------------------------------------------------

    /// The oldest address the file still holds.
    #[inline]
    #[must_use]
    pub const fn begin(&self) -> u64 {
        self.begin
    }

    /// The boundary between the stable region and the read only region.
    #[inline]
    #[must_use]
    pub const fn head(&self) -> u64 {
        self.head
    }

    /// The boundary between the read only region and the mutable region.
    #[inline]
    #[must_use]
    pub const fn read_only(&self) -> u64 {
        self.read_only
    }

    /// Where the next append goes.
    #[inline]
    #[must_use]
    pub const fn tail(&self) -> u64 {
        self.tail
    }

    /// The shard's current epoch.
    #[inline]
    #[must_use]
    pub const fn epoch(&self) -> u32 {
        self.epoch
    }

    /// How many records have been appended since this log was opened.
    #[inline]
    #[must_use]
    pub const fn appends(&self) -> u64 {
        self.appends
    }

    /// The configuration this log was built with.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &LogConfig {
        &self.cfg
    }

    /// Gives the store back, dropping the log.
    ///
    /// Recovery is the caller: it walks the store, then builds a new log over
    /// the same store with [`Log::recover`]. Anything not yet handed to the
    /// sink is gone, so call [`Log::commit_pending`] first if it mattered.
    #[must_use]
    pub fn into_sink(self) -> S {
        self.sink
    }

    /// Payload bytes per page, which is the page size minus its header.
    #[inline]
    #[must_use]
    pub const fn payload_len(&self) -> usize {
        self.payload_len
    }

    /// The largest record this log can hold, which is one page of payload.
    ///
    /// Anything bigger is a chain of chunk records, and building that chain is
    /// `05`'s job rather than this crate's.
    #[inline]
    #[must_use]
    pub const fn max_record_len(&self) -> usize {
        self.payload_len
    }

    /// The sink, for a caller that needs to ask it something.
    #[inline]
    pub const fn sink(&self) -> &S {
        &self.sink
    }

    /// The sink, for a caller that needs to tell it something.
    ///
    /// [`PageSink`] covers what the log itself needs and nothing else, so
    /// shutdown, checkpointing and anything a particular sink offers on top of
    /// the trait come through here. The log's own invariants do not depend on
    /// anything reachable this way, which is why handing it out is safe: what a
    /// sink does with bytes it has already been given is between it and its
    /// owner.
    #[inline]
    pub const fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// The log address below which every record is durable.
    #[inline]
    #[must_use]
    pub fn durable_upto(&self) -> u64 {
        self.sink.durable_upto()
    }

    /// Lets the sink pick up whatever has finished.
    ///
    /// Once a turn of the shard loop. With a synchronous sink this does nothing
    /// and costs a call. With an asynchronous one it is where [`Log::durable_upto`]
    /// moves, and therefore where a caller parked on
    /// [`CommitAction::WaitFor`] stops being parked.
    ///
    /// # Errors
    ///
    /// Whatever the sink found out about since the last call.
    #[inline]
    pub fn poll(&mut self) -> Result<()> {
        self.sink.poll()
    }

    /// Moves the shard's epoch on. `04` section 4's reclamation boundary.
    pub const fn advance_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Which region `addr` is in.
    #[must_use]
    pub const fn region_of(&self, addr: u64) -> Region {
        if addr < self.begin {
            Region::Reclaimed
        } else if addr >= self.tail {
            Region::Unallocated
        } else if addr >= self.read_only {
            Region::Mutable
        } else if addr >= self.head {
            Region::ReadOnly
        } else {
            Region::Stable
        }
    }

    /// This shard's line in the superblock, as of right now.
    #[must_use]
    pub const fn checkpoint_entry(
        &self,
        index_image_addr: u64,
        index_image_len: u64,
        key_count: u64,
    ) -> CheckpointEntry {
        CheckpointEntry {
            log_begin: self.begin,
            log_head: self.head,
            log_read_only: self.read_only,
            log_tail: self.tail,
            index_image_addr,
            index_image_len,
            key_count,
            epoch: self.epoch,
        }
    }

    // -- appending ----------------------------------------------------------

    /// Appends a record and returns where it went.
    ///
    /// The ordering here is the rule from `06` section 3 and it is the reason
    /// this function is not three lines. The body goes down first, then the
    /// four bytes where the next record's length will live are zeroed, then a
    /// release fence, and only then is this record's own length stored. A
    /// reader that arrives at any point in that sequence sees either a whole
    /// record or a zero, and a zero means stop.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] if the record is larger than a page, [`Code::Invalid`]
    /// for a key longer than 65535 bytes, and whatever the sink returns if a
    /// page had to be flushed to make room.
    pub fn append(&mut self, h: &RecordHeader, key: &[u8], value: &[u8]) -> Result<Append> {
        let n = total_len(h.flags, key.len(), value.len())?;
        let (slot, off) = self.reserve(n)?;
        let addr = self.tail;
        let base = PAGE_HEADER_LEN + off;
        h.fill(&mut self.pages[slot].buf[base..base + n], key, value)?;
        self.publish(slot, off, n);
        self.finish_append(addr, n)
    }

    /// Appends a record that is already formed, which is what compaction does.
    ///
    /// The bytes must be a whole valid record including its length. Used when
    /// moving a live record from the region behind `head` up to the tail, where
    /// re-encoding it would mean decoding a record this build may not
    /// understand.
    ///
    /// # Errors
    ///
    /// As [`Self::append`], plus [`Code::Invalid`] if `record` does not look
    /// like a record.
    pub fn append_bytes(&mut self, record: &[u8]) -> Result<Append> {
        if record.len() < 4 {
            return Err(Error::new(Code::Invalid, "not a record"));
        }
        let n = u32::from_le_bytes([record[0], record[1], record[2], record[3]]) as usize;
        if n == 0 || n > record.len() {
            return Err(
                Error::new(Code::Invalid, "the record length does not match the bytes")
                    .with_detail(format!("len={n} bytes={}", record.len())),
            );
        }
        let (slot, off) = self.reserve(n)?;
        let addr = self.tail;
        let base = PAGE_HEADER_LEN + off;
        // Everything but the length, for the same reason as `append`.
        self.pages[slot].buf[base + 4..base + n].copy_from_slice(&record[4..n]);
        self.publish(slot, off, n);
        self.finish_append(addr, n)
    }

    /// Makes room for `n` bytes and says where they go.
    ///
    /// Returns the resident slot and the payload offset within it. Turns the
    /// page if the record would straddle a boundary, because `06` section 2.1
    /// says a record never does.
    fn reserve(&mut self, n: usize) -> Result<(usize, usize)> {
        let stride = align_up(n);
        if stride > self.payload_len {
            return Err(Error::new(Code::Full, "the record is larger than a page")
                .with_detail(format!("len={n} page_payload={}", self.payload_len)));
        }
        let mut off = (self.tail % self.payload_len as u64) as usize;
        if off + stride > self.payload_len {
            // The rest of this page stays as it is. The sentinel written by the
            // previous append is already sitting at `off`, so a reader walking
            // this page stops here, which is what a turned page should look
            // like.
            self.close_page(self.slot_for(self.tail))?;
            self.tail += (self.payload_len - off) as u64;
            self.open_tail_page();
            off = 0;
        }
        let slot = self.slot_for(self.tail);
        Ok((slot, off))
    }

    /// Writes the tail sentinel, fences, then stores the length.
    fn publish(&mut self, slot: usize, off: usize, n: usize) {
        let stride = align_up(n);
        let base = PAGE_HEADER_LEN + off;
        let next = base + stride;
        if next + 4 <= self.cfg.page_len {
            self.pages[slot].buf[next..next + 4].fill(0);
        }
        // The store below has to be the last thing any reader sees, whether the
        // reader is a replica in another process reading through the shared
        // mapping or a recovery pass reading the page off the store after a
        // crash. Everything above it is the record's body; everything after it
        // is a record that exists.
        fence(Ordering::Release);
        seal_len(&mut self.pages[slot].buf[base..base + 4], n);
        let page = &mut self.pages[slot];
        page.used = (off + stride) as u32;
        page.dirty = true;
    }

    /// Advances the tail, applies the durability mode, and reports the action.
    fn finish_append(&mut self, addr: u64, n: usize) -> Result<Append> {
        self.tail = addr + align_up(n) as u64;
        self.appends += 1;
        self.recompute_boundaries();

        // A record that ends exactly on a page boundary turns the page here,
        // rather than leaving it for the next append to notice. Leaving it is
        // wrong in a way that is quiet: the next append computes an offset of
        // zero, which looks like the start of a fresh page, but the ring slot
        // still belongs to the page that just filled up. The record then
        // overwrites the first record of that older page and the older page is
        // never flushed. Exact fits are rare, which is what makes this the kind
        // of bug that survives a test suite.
        if self.tail.is_multiple_of(self.payload_len as u64) {
            self.close_page(self.slot_for(addr))?;
            self.open_tail_page();
        }

        let action = match self.cfg.durability {
            Durability::None | Durability::Os => CommitAction::Reply,
            Durability::Group => CommitAction::WaitFor(self.tail),
            Durability::Sync => {
                self.commit_pending()?;
                // A synchronous sink is already there, so the caller is
                // answered now and this mode costs what it has always cost.
                //
                // An asynchronous one has only been asked, and answering here
                // would be the reply before the fsync, which is the whole bug
                // this mode exists to not have. So the caller parks on the
                // address, exactly like group mode, and gets its answer from
                // the same place. Same guarantee, reached later.
                if self.sink.durable_upto() >= self.tail {
                    CommitAction::Reply
                } else {
                    CommitAction::WaitFor(self.tail)
                }
            }
        };
        Ok(Append {
            addr,
            len: n as u32,
            action,
        })
    }

    // -- pages --------------------------------------------------------------

    /// The resident slot an address maps to.
    #[inline]
    const fn slot_for(&self, addr: u64) -> usize {
        ((addr / self.payload_len as u64) % self.cfg.resident_pages as u64) as usize
    }

    /// The address of the first payload byte of the page holding `addr`.
    #[inline]
    pub(crate) const fn page_addr_of(&self, addr: u64) -> u64 {
        addr - (addr % self.payload_len as u64)
    }

    /// Claims the slot for the current tail, evicting whatever was there.
    fn open_tail_page(&mut self) {
        let page_addr = self.page_addr_of(self.tail);
        let slot = self.slot_for(self.tail);
        let off = (self.tail - page_addr) as usize;
        let page = &mut self.pages[slot];
        page.page_addr = Some(page_addr);
        page.used = off as u32;
        page.dead = 0;
        page.flushed = 0;
        page.dirty = false;
        // The sentinel for the first record. Without this a page reused from a
        // previous turn of the ring starts with whatever the old record's
        // length was, and a reader believes it.
        let base = PAGE_HEADER_LEN + off;
        page.buf[base..base + 4].fill(0);
        self.recompute_boundaries();
    }

    /// Flushes a page according to the durability mode, because the tail is
    /// about to leave it.
    ///
    /// Takes the slot rather than reading it off the tail, since the caller may
    /// have moved the tail already and the page that needs writing is the one
    /// behind it.
    fn close_page(&mut self, slot: usize) -> Result<()> {
        match self.cfg.durability {
            Durability::None => Ok(()),
            // The `poll` is what hands an asynchronous sink's submissions to
            // the kernel, and this mode's promise is that the kernel has them.
            // A page boundary is once per 32 MiB, so the call costs nothing
            // anybody can measure. On a synchronous sink it does nothing at all.
            Durability::Os => {
                self.flush_slot(slot)?;
                self.sink.poll()
            }
            // A page that is about to stop being the tail is a page nobody will
            // ever append to again, so this is the last chance for the commits
            // in it to become durable together. That is group commit.
            Durability::Group | Durability::Sync => {
                self.flush_slot(slot)?;
                self.sink.sync()
            }
        }
    }

    /// Hands the dirty part of the tail page to the sink, without a sync.
    ///
    /// # Errors
    ///
    /// Whatever the sink returns.
    pub fn flush(&mut self) -> Result<()> {
        let slot = self.slot_for(self.tail);
        self.flush_slot(slot)
    }

    /// Flushes and syncs, then everything appended so far is durable.
    ///
    /// This is what the maintenance slice calls in `group` mode so that a
    /// commit does not wait for a 32 MiB page to fill before it is answered.
    ///
    /// # Errors
    ///
    /// Whatever the sink returns.
    pub fn commit_pending(&mut self) -> Result<()> {
        self.flush()?;
        self.sink.sync()
    }

    fn flush_slot(&mut self, slot: usize) -> Result<()> {
        let (page_addr, used, dead, flushed, dirty) = {
            let p = &self.pages[slot];
            match p.page_addr {
                Some(a) => (a, p.used, p.dead, p.flushed, p.dirty),
                None => return Ok(()),
            }
        };
        if !dirty {
            return Ok(());
        }

        PageHeader {
            shard: self.cfg.shard,
            page_addr,
            used,
            dead_bytes: dead,
            epoch: self.epoch,
        }
        .encode(&mut self.pages[slot].buf);

        // The sentinel past the last record is part of what has to land, or a
        // reader picks up whatever the previous turn of the ring left there.
        let phys_end = (PAGE_HEADER_LEN + used as usize + 4).min(self.cfg.page_len);
        let hi = phys_end
            .next_multiple_of(FLUSH_BLOCK)
            .min(self.cfg.page_len);
        let lo = (flushed / FLUSH_BLOCK) * FLUSH_BLOCK;
        let covers_upto = page_addr + u64::from(used);

        // Writes are block sized, so the block holding the sentinel goes out
        // with a tail of bytes past it that no append has filled in yet. Those
        // bytes are whatever the previous tenant of this ring slot left behind,
        // and handing them to the store is a real bug rather than untidiness.
        //
        // The store may not have written the block yet, and the same block goes
        // out again on the next flush with more records in it. A device is free
        // to take one sector from this write and the neighbouring sector from
        // the next, and nothing has synced in between to forbid it. So a page
        // can come back with a used mark from the later write and a sentinel
        // sector from the earlier one, and if that sector carries a stale record
        // it parses: a record's checksum covers its own bytes and says nothing
        // about which address they belong at, so a record from an older page at
        // the same offset is indistinguishable from the right one. Replay walks
        // straight into it and hands back a record that was never written there.
        //
        // Zeroing the tail is enough on its own. It costs one memset of under a
        // block per flush, and it means every byte the store is ever given is
        // either a record of this page or a zero, so whichever sector wins the
        // race the worst case is an early end of log, which is what a torn tail
        // is supposed to look like.
        self.pages[slot].buf[phys_end..hi].fill(0);

        // Data first, header second. Neither order is unsafe, because `used` is
        // a hint and the record walk is the authority on where the log ends, but
        // a header that promises records which did not land is the kind of thing
        // somebody debugs for a day, so it is written last.
        if lo > 0 {
            self.sink.write(PageWrite {
                page_addr,
                offset: lo,
                bytes: &self.pages[slot].buf[lo..hi],
                covers_upto,
            })?;
            self.sink.write(PageWrite {
                page_addr,
                offset: 0,
                bytes: &self.pages[slot].buf[..FLUSH_BLOCK],
                covers_upto,
            })?;
        } else {
            self.sink.write(PageWrite {
                page_addr,
                offset: 0,
                bytes: &self.pages[slot].buf[..hi],
                covers_upto,
            })?;
        }

        let p = &mut self.pages[slot];
        p.flushed = phys_end;
        p.dirty = false;
        Ok(())
    }

    /// Recomputes `head` and `read_only` from the tail.
    ///
    /// `head` is the start of the oldest resident page, because a page that has
    /// been evicted is a page that lives in the file, which is what stable
    /// means. `read_only` is the mutable fraction of the resident window
    /// measured back from the tail.
    fn recompute_boundaries(&mut self) {
        let page_index = self.tail / self.payload_len as u64;
        let oldest = page_index.saturating_sub(self.cfg.resident_pages as u64 - 1);
        self.head = (oldest * self.payload_len as u64).max(self.begin);

        let window = self.cfg.resident_pages as f64 * self.payload_len as f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "window is a page count times a page size, both small, and the fraction is checked to be between 0 and 1 at construction"
        )]
        let mutable = (window * self.cfg.mutable_fraction) as u64;
        self.read_only = self
            .tail
            .saturating_sub(mutable)
            .clamp(self.head, self.tail);
    }

    // -- reading ------------------------------------------------------------

    /// The record at `addr`, if it is resident.
    ///
    /// # Errors
    ///
    /// [`Code::NotFound`] if the address is outside the resident window, which
    /// is not a failure: it is the signal to fault the page in, and it carries
    /// the region in its detail so the caller knows which direction to look.
    /// [`Code::Corrupt`] if what is there does not parse.
    pub fn read(&self, addr: u64) -> Result<RecordRef<'_>> {
        let region = self.region_of(addr);
        if !matches!(region, Region::ReadOnly | Region::Mutable) {
            return Err(
                Error::new(Code::NotFound, "that address is not resident").with_detail(format!(
                    "addr={addr} region={region:?} head={} tail={}",
                    self.head, self.tail
                )),
            );
        }
        let slot = self.slot_for(addr);
        let page = &self.pages[slot];
        let page_addr = self.page_addr_of(addr);
        if page.page_addr != Some(page_addr) {
            // The address is inside the window by arithmetic but the slot holds
            // a different page. That means the boundaries and the ring have
            // disagreed, which is a bug here rather than a corrupt file, so it
            // says so.
            return Err(
                Error::new(Code::Corrupt, "the resident page is not the one expected")
                    .with_detail(format!("want={page_addr} have={:?}", page.page_addr)),
            );
        }
        let off = PAGE_HEADER_LEN + (addr - page_addr) as usize;
        match RecordRef::parse(&page.buf[off..])? {
            Some(r) => Ok(r),
            None => Err(
                Error::new(Code::Corrupt, "there is no record at that address")
                    .with_detail(format!("addr={addr}")),
            ),
        }
    }

    /// Marks `len` bytes at `addr` as no longer referenced by the index.
    ///
    /// Called when a key's index entry moves away from an address, which is
    /// what drives the compaction trigger (`06` section 5). An address outside
    /// the resident window is ignored rather than an error, because the page
    /// that owned it carries its own counter and updating that is the file
    /// layer's job.
    pub fn mark_dead(&mut self, addr: u64, len: u32) {
        if addr < self.head || addr >= self.tail {
            return;
        }
        let slot = self.slot_for(addr);
        let page_addr = self.page_addr_of(addr);
        let p = &mut self.pages[slot];
        if p.page_addr != Some(page_addr) {
            return;
        }
        // Saturating rather than wrapping. Double counting a dead record makes
        // compaction eager, which wastes time; underflowing makes it never run,
        // which loses the disk.
        p.dead = p.dead.saturating_add(len).min(p.used);
        p.dirty = true;
    }

    /// The dead byte count of the page holding `addr`, if it is resident.
    #[must_use]
    pub fn dead_bytes_at(&self, addr: u64) -> Option<u32> {
        if addr < self.head || addr >= self.tail {
            return None;
        }
        let p = &self.pages[self.slot_for(addr)];
        if p.page_addr == Some(self.page_addr_of(addr)) {
            Some(p.dead)
        } else {
            None
        }
    }

    /// Raises `begin`, which is what compaction does when it has finished with
    /// everything below the new value.
    ///
    /// Clamped to `head`, because `begin` passing `head` would mean the log had
    /// reclaimed bytes it is still serving reads from.
    pub(crate) fn set_begin(&mut self, at: u64) {
        self.begin = at.clamp(self.begin, self.head);
        self.recompute_boundaries();
    }
}

impl<S: PageSink + PageSource> Log<S> {
    /// A log that continues from `at`, wherever `at` is.
    ///
    /// This is what recovery calls, with the address [`replay`](crate::replay())
    /// stopped at. It differs from [`Log::open`] in exactly one way, and that
    /// way is the whole reason it exists: when `at` falls in the middle of a
    /// page, the records already in that page are read back from the store into
    /// the tail page buffer before anything is appended.
    ///
    /// Skip that and the log looks fine right up until the first flush, which
    /// writes the buffer's zeroed prefix over the records in front of the tail
    /// and takes out every commit in the page. It survives a casual test suite
    /// because a test that appends, replays and checks the report never flushes
    /// afterwards.
    ///
    /// The recovered log begins at the start of that page rather than at `at`,
    /// so the records it just read back are readable through [`Log::read`]
    /// instead of sitting in memory behind a `NotFound`. Anything older is in
    /// the file and the file layer is what serves it.
    ///
    /// The epoch and the dead byte count come back from the page header too.
    /// Neither is load bearing for correctness, but an epoch that restarts at
    /// zero after a crash makes reclamation reason about a boundary that has
    /// already been crossed, and a dead count that restarts at zero hides a
    /// half garbage page from the compactor until it is rewritten.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the configuration cannot work, or [`Code::Corrupt`]
    /// if `at` is inside a page the store cannot produce, which means the walk
    /// that produced `at` and the store disagree about what exists.
    pub fn recover(cfg: LogConfig, sink: S, at: u64) -> Result<Log<S>> {
        let mut log = Log::open(cfg, sink, at)?;
        let off = (at % log.payload_len as u64) as usize;
        if off == 0 {
            return Ok(log);
        }

        let page_addr = log.page_addr_of(at);
        let slot = log.slot_for(at);
        let page_len = log.cfg.page_len;

        let header = {
            let Some(image) = log.sink.page_bytes(page_addr) else {
                return Err(
                    Error::new(Code::Corrupt, "the tail page is not in the store")
                        .with_detail(format!("page_addr={page_addr} tail={at}")),
                );
            };
            let n = image.len().min(page_len);
            // The whole image, not just the prefix below the tail. The bytes
            // above it are dead either way, and copying them keeps a rewrite of
            // the last flushed block byte for byte identical to what is already
            // on the store.
            let header = PageHeader::decode(image).ok();
            log.pages[slot].buf[..n].copy_from_slice(&image[..n]);
            header
        };

        if let Some(h) = header {
            log.epoch = h.epoch;
            // Clamped, because replay may have stopped below what the header
            // claims and dead bytes above the tail are not this page's problem
            // any more.
            log.pages[slot].dead = h.dead_bytes.min(off as u32);
        }

        // The sentinel, again. `open_tail_page` wrote it and the copy above just
        // wrote over it with whatever the last flush left past the tail, which
        // after a torn write is not necessarily four zeroes.
        let base = PAGE_HEADER_LEN + off;
        if base + 4 <= page_len {
            log.pages[slot].buf[base..base + 4].fill(0);
        }
        // Everything below the tail is already on the store, so the first flush
        // has no reason to send it again.
        log.pages[slot].flushed = base;

        // `begin` drops to the start of the page, because the records in it are
        // now in memory and a log that would not serve a read of a record it is
        // holding is just wrong. It stops there rather than going further back:
        // the older pages are in the file, which is what stable means, and the
        // file layer is what reads them. Leaving `begin` here also keeps `head`
        // from sliding under it on the next page turn and claiming that slots
        // this log never loaded are resident.
        log.begin = page_addr;
        log.recompute_boundaries();
        Ok(log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::MemorySink;
    use yo_format::record::RecordKind;

    /// A log with pages small enough that a test can turn one.
    fn small(durability: Durability) -> Log<MemorySink> {
        Log::new(
            LogConfig {
                shard: 1,
                page_len: 8192,
                resident_pages: 3,
                mutable_fraction: 0.40,
                durability,
            },
            MemorySink::new(),
        )
        .unwrap()
    }

    fn put(log: &mut Log<MemorySink>, key: &[u8], value: &[u8]) -> Append {
        log.append(&RecordHeader::new(RecordKind::String), key, value)
            .unwrap()
    }

    /// Everything a recovered log needs from the one before it, in the case
    /// recovery actually hits, which is a tail somewhere in the middle of a
    /// page rather than neatly on a boundary.
    #[test]
    fn a_recovered_log_keeps_the_records_already_in_its_tail_page() {
        let mut first = small(Durability::Group);
        first.advance_epoch();
        first.advance_epoch();
        let mut addrs = Vec::new();
        for i in 0..20u32 {
            addrs.push(put(&mut first, &i.to_be_bytes(), b"before the crash").addr);
        }
        first.commit_pending().unwrap();
        let tail = first.tail();
        assert!(!tail.is_multiple_of(first.payload_len() as u64), "mid page");

        let cfg = *first.config();
        let mut second = Log::recover(cfg, first.into_sink(), tail).unwrap();
        assert_eq!(second.epoch(), 2, "the epoch restarted at zero");

        // The append and the flush together are what used to lose the page: the
        // flush is the moment a zeroed buffer would have gone over the records.
        let after = put(&mut second, b"after", b"the crash").addr;
        assert_eq!(after, tail);
        second.commit_pending().unwrap();

        for (i, addr) in addrs.iter().enumerate() {
            let r = second.read(*addr).unwrap();
            assert_eq!(r.value, b"before the crash", "record {i} at {addr}");
        }
        assert_eq!(second.read(after).unwrap().value, b"the crash");
    }

    /// The page aligned case, which is what a checkpoint taken on a boundary
    /// gives you, and which has no page to read back.
    #[test]
    fn recovering_on_a_page_boundary_needs_nothing_from_the_store() {
        let cfg = LogConfig {
            shard: 1,
            page_len: 8192,
            resident_pages: 3,
            mutable_fraction: 0.40,
            durability: Durability::Group,
        };
        let payload = 8192 - PAGE_HEADER_LEN;
        let log = Log::recover(cfg, MemorySink::new(), 2 * payload as u64).unwrap();
        assert_eq!(log.tail(), 2 * payload as u64);
        assert_eq!(log.begin(), 2 * payload as u64);
    }

    /// A tail inside a page the store cannot produce is a disagreement between
    /// the walk and the store, and saying so beats appending into a hole.
    #[test]
    fn recovering_into_a_page_the_store_does_not_have_is_an_error() {
        let cfg = LogConfig {
            shard: 1,
            page_len: 8192,
            resident_pages: 3,
            mutable_fraction: 0.40,
            durability: Durability::Group,
        };
        let err = Log::recover(cfg, MemorySink::new(), 64).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
    }

    #[test]
    fn a_record_comes_back_from_the_address_it_went_to() {
        let mut log = small(Durability::None);
        let a = put(&mut log, b"one", b"first");
        let b = put(&mut log, b"two", b"second");
        assert_ne!(a.addr, b.addr);
        assert_eq!(log.read(a.addr).unwrap().value, b"first");
        assert_eq!(log.read(b.addr).unwrap().value, b"second");
        assert_eq!(log.appends(), 2);
    }

    #[test]
    fn the_tail_moves_by_the_stride_and_not_by_the_length() {
        let mut log = small(Durability::None);
        let before = log.tail();
        let a = put(&mut log, b"k", b"abc");
        assert_eq!(a.len as usize, 16 + 1 + 3 + 4);
        assert_eq!(log.tail() - before, 24, "aligned up from 24, which is 24");

        let before = log.tail();
        let b = put(&mut log, b"k", b"ab");
        assert_eq!(b.len, 23);
        assert_eq!(log.tail() - before, 24, "23 rounds up to 24");
    }

    #[test]
    fn the_four_bytes_past_the_tail_are_always_zero() {
        // The end of log sentinel. Everything about recovery rests on it.
        let mut log = small(Durability::None);
        for i in 0..50u32 {
            let a = put(&mut log, b"key", &i.to_le_bytes());
            let end = a.addr + align_up(a.len as usize) as u64;
            assert_eq!(log.tail(), end);
            assert!(
                log.read(end).is_err(),
                "there is nothing at the tail, and it does not parse as a record"
            );
        }
    }

    #[test]
    fn a_record_that_ends_exactly_on_a_page_boundary_still_turns_the_page() {
        // A regression test for a bug that only shows up when a record's stride
        // divides the payload exactly. The page looked full but was never
        // closed, so the next append landed at offset zero of a ring slot that
        // still belonged to the page that had just filled, overwrote its first
        // record, and the page was never written out at all.
        //
        // 8160 payload bytes and a 32 byte stride is 255 records with nothing
        // left over, which is the case that used to break.
        let mut log = small(Durability::Group);
        assert_eq!(log.payload_len(), 8160);
        let stride = 32;
        assert_eq!(
            log.payload_len() % stride,
            0,
            "the test is not exercising it"
        );

        // 16 header plus 3 key plus 9 value plus 4 trailer is 32 on the nose.
        let mut addrs = Vec::new();
        for _ in 0..1000 {
            let a = put(&mut log, b"key", b"999999999");
            assert_eq!(a.len as usize, stride);
            addrs.push(a.addr);
        }
        log.commit_pending().unwrap();

        // Every page the log claims to have written is there, and none of them
        // is missing out of the middle.
        let payload = log.payload_len() as u64;
        let pages = log.sink().pages();
        let last = log.tail() / payload;
        for k in 0..last {
            assert!(
                pages.iter().any(|(a, _)| *a == k * payload),
                "page {k} was never written"
            );
        }
        // And every record is still where its append said it was.
        for (i, addr) in addrs.iter().enumerate() {
            let page_addr = addr - (addr % payload);
            let bytes = log.sink().page(page_addr).unwrap();
            let off = PAGE_HEADER_LEN + (addr - page_addr) as usize;
            let r = RecordRef::parse(&bytes[off..])
                .unwrap()
                .unwrap_or_else(|| panic!("record {i} at {addr} is gone"));
            assert_eq!(r.key, b"key");
        }
    }

    #[test]
    fn a_record_never_straddles_a_page_boundary() {
        let mut log = small(Durability::None);
        let payload = log.payload_len() as u64;
        let value = vec![b'v'; 200];
        let mut last_page = 0;
        for _ in 0..200 {
            let a = put(&mut log, b"key", &value);
            let start_page = a.addr / payload;
            let end_page = (a.addr + u64::from(a.len) - 1) / payload;
            assert_eq!(start_page, end_page, "record at {} straddles", a.addr);
            last_page = last_page.max(end_page);
        }
        assert!(last_page >= 3, "the test did not actually turn any pages");
    }

    #[test]
    fn turning_a_page_leaves_the_gap_readable_as_end_of_log() {
        let mut log = small(Durability::None);
        let payload = log.payload_len() as u64;
        let value = vec![b'v'; 300];
        // Fill until the next record cannot fit, then check the gap.
        loop {
            let before = log.tail();
            let a = put(&mut log, b"key", &value);
            if a.addr / payload != before / payload {
                // The page turned. Everything from `before` to the boundary is
                // the gap, and it has to read as the end of the log.
                assert!(log.read(before).is_err());
                break;
            }
        }
    }

    #[test]
    fn the_regions_are_in_order_and_cover_the_address_space() {
        let mut log = small(Durability::None);
        let value = vec![b'v'; 100];
        for _ in 0..300 {
            put(&mut log, b"key", &value);
            assert!(log.begin() <= log.head(), "begin past head");
            assert!(log.head() <= log.read_only(), "head past read_only");
            assert!(log.read_only() <= log.tail(), "read_only past tail");
        }
        assert!(log.head() > 0, "nothing ever became stable");

        assert_eq!(log.region_of(log.tail()), Region::Unallocated);
        assert_eq!(log.region_of(log.tail() + 1000), Region::Unallocated);
        assert_eq!(log.region_of(log.read_only()), Region::Mutable);
        assert_eq!(log.region_of(log.head()), Region::ReadOnly);
        assert_eq!(log.region_of(log.head() - 1), Region::Stable);
    }

    #[test]
    fn the_mutable_region_is_the_configured_fraction_of_the_window() {
        let mut log = Log::new(
            LogConfig {
                page_len: 8192,
                resident_pages: 5,
                mutable_fraction: 0.40,
                durability: Durability::None,
                ..LogConfig::default()
            },
            MemorySink::new(),
        )
        .unwrap();
        let value = vec![b'v'; 100];
        for _ in 0..500 {
            put(&mut log, b"key", &value);
        }
        let window = 5 * log.payload_len() as u64;
        let mutable = log.tail() - log.read_only();
        assert_eq!(mutable, (window as f64 * 0.40) as u64);
    }

    #[test]
    fn a_fully_mutable_log_has_no_read_only_region() {
        let mut log = Log::new(
            LogConfig {
                page_len: 8192,
                resident_pages: 3,
                mutable_fraction: 1.0,
                durability: Durability::None,
                ..LogConfig::default()
            },
            MemorySink::new(),
        )
        .unwrap();
        for _ in 0..200 {
            put(&mut log, b"key", &[0u8; 100]);
        }
        assert_eq!(
            log.read_only(),
            log.head(),
            "everything resident is mutable"
        );
    }

    #[test]
    fn a_record_larger_than_a_page_is_refused_with_the_two_numbers() {
        let mut log = small(Durability::None);
        let huge = vec![0u8; log.payload_len() + 1];
        let err = log
            .append(&RecordHeader::new(RecordKind::String), b"k", &huge)
            .unwrap_err();
        assert_eq!(err.code(), Code::Full);
        assert!(err.detail().unwrap().contains("page_payload"));
    }

    #[test]
    fn an_evicted_address_is_not_resident_and_says_which_region_it_is_in() {
        let mut log = small(Durability::None);
        let first = put(&mut log, b"key", b"the very first value");
        let value = vec![b'v'; 200];
        for _ in 0..200 {
            put(&mut log, b"key", &value);
        }
        let err = log.read(first.addr).unwrap_err();
        assert_eq!(err.code(), Code::NotFound, "fault it in, do not fail");
        assert!(err.detail().unwrap().contains("Stable"));
    }

    #[test]
    fn mode_none_never_touches_the_sink_until_a_page_turns() {
        let mut log = small(Durability::None);
        for _ in 0..10 {
            put(&mut log, b"key", b"value");
        }
        assert_eq!(log.sink().writes(), 0);
        assert_eq!(log.sink().syncs(), 0);
        // And it still never syncs, however many pages go by.
        let value = vec![b'v'; 300];
        for _ in 0..200 {
            put(&mut log, b"key", &value);
        }
        assert_eq!(log.sink().syncs(), 0);
    }

    #[test]
    fn mode_os_writes_at_page_boundaries_and_never_syncs() {
        let mut log = small(Durability::Os);
        let value = vec![b'v'; 300];
        for _ in 0..200 {
            put(&mut log, b"key", &value);
        }
        assert!(log.sink().writes() > 0, "nothing was ever written");
        assert_eq!(log.sink().syncs(), 0, "os mode does not ask for a sync");
    }

    #[test]
    fn mode_group_parks_the_caller_and_one_sync_serves_a_whole_page() {
        let mut log = small(Durability::Group);
        let value = vec![b'v'; 100];
        let mut parked = Vec::new();
        for _ in 0..200 {
            let a = put(&mut log, b"key", &value);
            match a.action {
                CommitAction::WaitFor(at) => parked.push(at),
                CommitAction::Reply => panic!("group mode replied early"),
            }
        }
        let syncs = log.sink().syncs();
        assert!(syncs > 0, "no page ever committed");
        assert!(
            syncs < 20,
            "{syncs} syncs for 200 commits is not a group commit"
        );

        // Everything up to the last page boundary is durable, and the commits
        // still in the tail page are not, which is exactly who should be parked.
        let durable = log.durable_upto();
        assert!(durable > 0);
        assert!(durable < log.tail());
        assert!(parked.iter().any(|&at| at <= durable));
        assert!(parked.iter().any(|&at| at > durable));

        // The maintenance slice is what answers the rest.
        log.commit_pending().unwrap();
        assert_eq!(log.durable_upto(), log.tail());
        assert!(parked.iter().all(|&at| at <= log.durable_upto()));
    }

    #[test]
    fn mode_sync_syncs_per_commit_and_replies_immediately() {
        let mut log = small(Durability::Sync);
        for _ in 0..20 {
            let a = put(&mut log, b"key", b"value");
            assert_eq!(a.action, CommitAction::Reply, "sync mode already synced");
        }
        assert_eq!(log.sink().syncs(), 20, "one sync per commit, as advertised");
        assert_eq!(log.durable_upto(), log.tail());
    }

    #[test]
    fn what_the_sink_holds_is_a_page_that_parses() {
        let mut log = small(Durability::Group);
        let value = vec![b'v'; 100];
        let mut written = Vec::new();
        for i in 0..200u32 {
            let a = put(&mut log, b"key", &i.to_le_bytes());
            written.push((a.addr, i));
            let _ = value;
        }
        log.commit_pending().unwrap();

        let payload = log.payload_len() as u64;
        for (page_addr, bytes) in log.sink().pages() {
            let h = PageHeader::decode(bytes).unwrap();
            assert_eq!(h.shard, 1);
            assert_eq!(h.page_addr, page_addr);
            assert!(h.used as u64 <= payload);
        }

        // And every record is where the append said it was.
        for (addr, i) in written {
            let page_addr = addr - (addr % payload);
            let bytes = log.sink().page(page_addr).unwrap();
            let off = PAGE_HEADER_LEN + (addr - page_addr) as usize;
            let r = RecordRef::parse(&bytes[off..]).unwrap().unwrap();
            assert_eq!(r.value, &i.to_le_bytes());
        }
    }

    /// A store is never handed a byte that is not a record of the page it is
    /// being told about, or a zero.
    ///
    /// Writes are block sized, so a flush in the middle of a page sends a tail
    /// of bytes past the sentinel that no append has filled in. Those bytes used
    /// to be whatever the previous tenant of the ring slot left behind, which
    /// meant a real record of an older page sitting at the same offset. Two
    /// flushes of the same block with no sync between them can be split by the
    /// device sector by sector, so a page could come back with a used mark from
    /// the second and a sentinel from the first, and the first had a parseable
    /// stale record where the sentinel should be. Replay walked into it and
    /// handed back a record that was never written at that address, which is the
    /// one thing this format is not allowed to do.
    ///
    /// Found by `yo-crash` at seed 26281, 400 records into an 8192 byte page.
    #[test]
    fn a_flush_never_hands_the_store_the_previous_tenant_of_the_page() {
        #[derive(Default)]
        struct Watchful {
            page_len: usize,
            writes: usize,
            checked_a_tail: bool,
        }

        impl PageSink for Watchful {
            fn write(&mut self, w: PageWrite<'_>) -> Result<()> {
                self.writes += 1;
                let used = (w.covers_upto - w.page_addr) as usize;
                let phys_end = (PAGE_HEADER_LEN + used + 4).min(self.page_len);
                // The part of this write that lies past the sentinel.
                if w.offset + w.bytes.len() > phys_end {
                    let from = phys_end.saturating_sub(w.offset);
                    let tail = &w.bytes[from..];
                    if !tail.is_empty() {
                        self.checked_a_tail = true;
                    }
                    let bad = tail.iter().position(|b| *b != 0);
                    assert!(
                        bad.is_none(),
                        "write {} to page {} put {} bytes past the sentinel at {}, first \
                         non zero at {}",
                        self.writes,
                        w.page_addr,
                        tail.len(),
                        phys_end,
                        bad.unwrap()
                    );
                }
                Ok(())
            }

            fn sync(&mut self) -> Result<()> {
                Ok(())
            }

            fn durable_upto(&self) -> u64 {
                0
            }
        }

        let page_len = 8192;
        let mut log = Log::new(
            LogConfig {
                shard: 1,
                page_len,
                // Three slots, so a page turn puts a live page back on top of
                // one that still holds an older page's records.
                resident_pages: 3,
                mutable_fraction: 0.40,
                durability: Durability::Group,
            },
            Watchful {
                page_len,
                ..Watchful::default()
            },
        )
        .unwrap();

        // Enough to turn the ring several times, with sizes that vary so the
        // sentinel lands at a different place in the block each flush. Miri
        // does half, which is still nearly two laps of a three slot ring and so
        // still puts a live page on top of an older page's records, which is
        // the only thing this test needs to be true.
        for i in 0..if cfg!(miri) { 300u32 } else { 600u32 } {
            let value = vec![i as u8; 40 + (i as usize % 173)];
            log.append(&RecordHeader::new(RecordKind::String), b"key", &value)
                .unwrap();
            // Flush part way through a page, repeatedly, which is what makes the
            // same block go out more than once.
            if i % 3 == 0 {
                log.flush().unwrap();
            }
        }
        log.commit_pending().unwrap();

        let sink = log.into_sink();
        assert!(sink.writes > 20, "only {} writes, too few", sink.writes);
        assert!(
            sink.checked_a_tail,
            "no write ever had bytes past the sentinel, so this proved nothing"
        );
    }

    #[test]
    fn a_flush_is_a_whole_number_of_torn_write_units() {
        let mut log = small(Durability::Group);
        for _ in 0..40 {
            put(&mut log, b"key", &[0u8; 100]);
        }
        log.commit_pending().unwrap();
        // Every page image is block aligned, because that is the only unit the
        // format assumes is atomic.
        for (_, bytes) in log.sink().pages() {
            assert!(
                bytes.len().is_multiple_of(FLUSH_BLOCK),
                "a {} byte page image is not block aligned",
                bytes.len()
            );
        }
    }

    #[test]
    fn dead_bytes_accumulate_and_never_pass_the_bytes_that_are_there() {
        let mut log = small(Durability::None);
        let a = put(&mut log, b"key", b"first");
        let b = put(&mut log, b"key", b"second");
        assert_eq!(log.dead_bytes_at(a.addr), Some(0));
        log.mark_dead(a.addr, a.len);
        assert_eq!(log.dead_bytes_at(a.addr), Some(a.len));
        log.mark_dead(b.addr, b.len);
        let used = log.dead_bytes_at(a.addr).unwrap();
        assert!(used >= a.len + b.len - 8);

        // Marking the same record twice is allowed and does not run past used.
        for _ in 0..100 {
            log.mark_dead(a.addr, a.len);
        }
        assert!(log.dead_bytes_at(a.addr).unwrap() <= log.tail() as u32);
    }

    #[test]
    fn marking_something_outside_the_window_is_ignored_and_not_an_error() {
        let mut log = small(Durability::None);
        put(&mut log, b"key", b"value");
        log.mark_dead(1 << 40, 100);
        log.mark_dead(u64::MAX, 100);
        assert_eq!(log.dead_bytes_at(1 << 40), None);
    }

    #[test]
    fn a_log_that_opens_at_a_checkpoint_continues_from_there() {
        let at = 8_000_000u64;
        let mut log = Log::open(
            LogConfig {
                page_len: 8192,
                durability: Durability::None,
                ..LogConfig::default()
            },
            MemorySink::new(),
            at,
        )
        .unwrap();
        assert_eq!(log.begin(), at);
        assert_eq!(log.tail(), at);
        assert_eq!(log.head(), at);
        let a = put(&mut log, b"key", b"value");
        assert_eq!(a.addr, at);
        assert_eq!(log.read(a.addr).unwrap().value, b"value");
        assert_eq!(log.region_of(at - 1), Region::Reclaimed);
    }

    #[test]
    fn the_checkpoint_entry_is_the_four_addresses_in_order() {
        let mut log = small(Durability::None);
        for _ in 0..200 {
            put(&mut log, b"key", &[0u8; 100]);
        }
        let e = log.checkpoint_entry(4096, 1024, 7);
        assert!(e.addresses_are_ordered());
        assert_eq!(e.log_tail, log.tail());
        assert_eq!(e.log_begin, log.begin());
        assert_eq!(e.epoch, log.epoch());
        assert_eq!(e.key_count, 7);
        // And it survives a round trip through its own bytes.
        let mut buf = [0u8; 64];
        e.encode(&mut buf);
        assert_eq!(CheckpointEntry::decode(&buf).unwrap(), e);
    }

    #[test]
    fn epochs_advance_and_land_in_the_page_header() {
        let mut log = small(Durability::Group);
        log.advance_epoch();
        log.advance_epoch();
        assert_eq!(log.epoch(), 2);
        put(&mut log, b"key", b"value");
        log.commit_pending().unwrap();
        let bytes = log.sink().page(0).unwrap();
        assert_eq!(PageHeader::decode(bytes).unwrap().epoch, 2);
    }

    #[test]
    fn a_configuration_that_cannot_work_is_refused_at_construction() {
        let bad = |cfg: LogConfig| Log::new(cfg, MemorySink::new()).unwrap_err().code();
        assert_eq!(
            bad(LogConfig {
                page_len: 32,
                ..LogConfig::default()
            }),
            Code::Invalid
        );
        assert_eq!(
            bad(LogConfig {
                page_len: 10_000,
                ..LogConfig::default()
            }),
            Code::Invalid,
            "not a whole number of blocks"
        );
        assert_eq!(
            bad(LogConfig {
                page_len: 8192,
                resident_pages: 1,
                ..LogConfig::default()
            }),
            Code::Invalid
        );
        assert_eq!(
            bad(LogConfig {
                page_len: 8192,
                mutable_fraction: 1.5,
                ..LogConfig::default()
            }),
            Code::Invalid
        );
    }

    #[test]
    fn append_bytes_reproduces_a_record_it_does_not_understand() {
        let mut log = small(Durability::None);
        // A kind from the future, which compaction has to be able to move.
        let h = RecordHeader {
            kind: 200,
            flags: yo_format::record::record_flags::CHECKSUMMED,
            prev: 0,
            ttl_ms: 0,
        };
        let mut raw = vec![0u8; 128];
        let n = h.fill(&mut raw, b"future", b"payload").unwrap();
        seal_len(&mut raw, n);

        let a = log.append_bytes(&raw[..n]).unwrap();
        let r = log.read(a.addr).unwrap();
        assert_eq!(r.kind, 200);
        assert_eq!(r.kind(), None);
        assert_eq!(r.key, b"future");
        assert_eq!(r.value, b"payload");
    }

    #[test]
    fn append_bytes_refuses_something_that_is_not_a_record() {
        let mut log = small(Durability::None);
        assert_eq!(log.append_bytes(&[1, 2]).unwrap_err().code(), Code::Invalid);
        assert_eq!(
            log.append_bytes(&[0, 0, 0, 0]).unwrap_err().code(),
            Code::Invalid
        );
        // A length that claims more than the caller handed over.
        assert_eq!(
            log.append_bytes(&[200, 0, 0, 0, 0, 0, 0, 0])
                .unwrap_err()
                .code(),
            Code::Invalid
        );
    }

    #[test]
    fn durability_modes_have_the_names_they_are_published_under() {
        for m in [
            Durability::None,
            Durability::Os,
            Durability::Group,
            Durability::Sync,
        ] {
            assert_eq!(Durability::parse(m.as_str()), Some(m));
        }
        assert_eq!(Durability::parse("everysec"), None);
        assert_eq!(Durability::default(), Durability::Group);
    }

    #[test]
    fn begin_only_ever_moves_forward_and_never_past_head() {
        let mut log = small(Durability::None);
        for _ in 0..300 {
            put(&mut log, b"key", &[0u8; 100]);
        }
        let head = log.head();
        assert!(head > 0);

        log.set_begin(head / 2);
        assert_eq!(log.begin(), head / 2);
        log.set_begin(0);
        assert_eq!(log.begin(), head / 2, "begin does not go backwards");
        log.set_begin(u64::MAX);
        assert_eq!(log.begin(), head, "begin does not pass head");
        assert!(log.begin() <= log.head());
    }
}
