//! The arena: 2 MiB segments, a bump pointer, and dead byte accounting.
//!
//! This is the whole value allocator for a shard (`05` section 3). There is no
//! free list, no size class and no header per allocation. Allocation is an add
//! and a compare. Deallocation does not exist, it just increments a counter on
//! the owning segment, and space comes back through compaction rather than
//! through free.
//!
//! L11 is the reason. aki's `*STORE` family ran at 0.30 to 0.55x because its
//! arena was grow only, so `SINTERSTORE` into an existing key leaked the old
//! result every time. The fix is reclaim, not malloc: a general allocator would
//! also fix the leak and would cost 20 to 60 ns per allocation with a size class
//! lookup and a possible lock, against the roughly 1 ns this costs.
//!
//! # Ownership
//!
//! An `Arena` belongs to exactly one shard and is never shared. It is `!Sync`
//! and `!Send` by construction, so handing one to another thread does not
//! compile. That is the same rule as everything else the shard owns (`04`
//! section 1) and it is what makes an unsynchronised bump pointer correct.

#![deny(missing_docs)]

use core::marker::PhantomData;
use core::ptr::NonNull;
use std::alloc::{Layout, alloc, dealloc};
use yo_common::{Addr, Space};

/// Bytes in one segment. Two megabytes, which is the huge page size on x86-64
/// and on aarch64 with a 4 KiB base page.
pub const SEGMENT_SIZE: usize = 2 * 1024 * 1024;

/// `SEGMENT_SIZE` as a shift, so that address to segment is a shift and a mask
/// rather than a divide.
pub const SEGMENT_SHIFT: u32 = 21;

/// Bytes of header at the front of every segment.
pub const HEADER_SIZE: usize = 32;

/// Allocation alignment. Sixteen covers every scalar the engine stores and
/// keeps SIMD loads over collection bodies aligned.
pub const ALIGN: usize = 16;

/// The largest single allocation a segment can hold.
///
/// Anything larger belongs in the log region (`06`), not the arena, and
/// [`Arena::alloc`] returns `None` rather than growing a segment to fit.
pub const MAX_ALLOC: usize = SEGMENT_SIZE - HEADER_SIZE;

const _: () = {
    assert!(SEGMENT_SIZE == 1 << SEGMENT_SHIFT);
    assert!(HEADER_SIZE.is_multiple_of(ALIGN));
};

/// The in memory and on disk header at offset 0 of every segment.
///
/// Laid out exactly as `05` section 3.1 specifies. `bump` here is a checkpoint
/// rather than the live cursor: see [`Arena::alloc`] for why, and
/// [`Arena::sync_headers`] for how it is brought up to date.
#[repr(C)]
#[derive(Debug)]
struct Header {
    bump: u64,
    dead_bytes: u64,
    epoch_retired: u32,
    flags: u32,
    next: u64,
}

const _: () = assert!(size_of::<Header>() == HEADER_SIZE);

struct Segment {
    base: NonNull<u8>,
}

impl Segment {
    /// Allocate one segment from the system.
    ///
    /// Aligned to its own size so that the kernel can back it with a huge page,
    /// which also makes it free to register with io_uring later (Y16).
    fn new() -> Segment {
        let layout = Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE)
            .expect("segment layout is a compile time constant");

        // Segment growth is off the command path by construction: it happens
        // once per 2 MiB of live data. It is still a heap allocation, so it is
        // marked rather than hidden.
        let raw = yo_alloc::allow(|| {
            // SAFETY: the layout has a non zero size and a power of two align.
            unsafe { alloc(layout) }
        });
        let base = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        // SAFETY: `base` is a fresh, uniquely owned, correctly aligned
        // allocation of `SEGMENT_SIZE` bytes, so writing a `Header` at offset 0
        // is in bounds and cannot alias anything.
        unsafe {
            base.cast::<Header>().write(Header {
                bump: HEADER_SIZE as u64,
                dead_bytes: 0,
                epoch_retired: 0,
                flags: 0,
                next: 0,
            });
        }

        // Miri has no kernel to ask, and a foreign call it cannot run would
        // stop the interpreter rather than fail softly.
        #[cfg(all(target_os = "linux", not(miri)))]
        {
            // Ask for a huge page. If the kernel says no, nothing breaks, so
            // the return value is deliberately ignored.
            // SAFETY: `base` points at `SEGMENT_SIZE` bytes we own, and madvise
            // with MADV_HUGEPAGE does not change the mapping's validity.
            unsafe {
                libc::madvise(
                    base.as_ptr().cast::<libc::c_void>(),
                    SEGMENT_SIZE,
                    libc::MADV_HUGEPAGE,
                );
            }
        }

        Segment { base }
    }

    #[inline]
    fn header(&self) -> &Header {
        // SAFETY: every segment has a `Header` written at offset 0 by `new`,
        // and the arena owns the segment exclusively.
        unsafe { self.base.cast::<Header>().as_ref() }
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        // SAFETY: as `header`, and `&mut self` proves exclusive access.
        unsafe { self.base.cast::<Header>().as_mut() }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(SEGMENT_SIZE, SEGMENT_SIZE)
            .expect("segment layout is a compile time constant");
        // SAFETY: `base` came from `alloc` with exactly this layout and has not
        // been freed, because only `Drop` frees it and it runs once.
        unsafe { dealloc(self.base.as_ptr(), layout) }
    }
}

/// A shard's arena.
///
/// Not `Send` and not `Sync`. Both are removed by the `PhantomData` below
/// rather than by a negative impl, so that this compiles on stable.
pub struct Arena {
    segs: Vec<Segment>,
    /// Index of the segment being bumped.
    cur: usize,
    /// `cur << SEGMENT_SHIFT`, kept so that forming an address is an add.
    cur_base_offset: u64,
    /// Next free byte in the current segment.
    cur_ptr: *mut u8,
    /// One past the last usable byte in the current segment.
    cur_end: *mut u8,
    /// Segment start, so that `cur_ptr` can be turned into an offset.
    cur_start: *mut u8,
    /// Total live bytes handed out, for `INFO memory`.
    allocated: u64,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Arena {
    /// A new arena with one segment ready.
    pub fn new() -> Arena {
        let mut a = Arena {
            segs: Vec::new(),
            cur: 0,
            cur_base_offset: 0,
            cur_ptr: core::ptr::null_mut(),
            cur_end: core::ptr::null_mut(),
            cur_start: core::ptr::null_mut(),
            allocated: 0,
            _not_send_sync: PhantomData,
        };
        a.push_segment();
        a
    }

    fn push_segment(&mut self) {
        let seg = Segment::new();
        let base = seg.base.as_ptr();
        yo_alloc::allow(|| self.segs.push(seg));
        self.cur = self.segs.len() - 1;
        self.cur_base_offset = (self.cur as u64) << SEGMENT_SHIFT;
        self.cur_start = base;
        // SAFETY: both offsets are within the `SEGMENT_SIZE` allocation.
        unsafe {
            self.cur_ptr = base.add(HEADER_SIZE);
            self.cur_end = base.add(SEGMENT_SIZE);
        }
    }

    /// Allocate `len` bytes and return their address and a writable view.
    ///
    /// Returns `None` only when `len` exceeds [`MAX_ALLOC`], which means the
    /// caller should be using the log region instead. Running out of segments
    /// is not a `None`, it allocates another one.
    ///
    /// The live cursor is `cur_ptr`, not the segment header's `bump`. The
    /// header sits on a different cache line from the bytes being handed out,
    /// so updating it on every allocation would touch two lines instead of one
    /// and roughly double the cost of the fast path. It is brought up to date
    /// on segment switch and by [`Arena::sync_headers`], which is what a
    /// checkpoint calls before it writes.
    #[inline]
    pub fn alloc(&mut self, len: usize) -> Option<(Addr, &mut [u8])> {
        let size = (len + (ALIGN - 1)) & !(ALIGN - 1);
        if size > MAX_ALLOC {
            return None;
        }

        let p = self.cur_ptr;
        // Compare addresses rather than pointers. The obvious form is
        // `p.add(size) > self.cur_end`, and it is undefined behaviour: forming
        // the pointer is the thing that is illegal once it runs past the
        // segment, so the comparison never gets to say no. Miri catches it,
        // release mode does not, and the difference between the two versions is
        // a subtract instead of a lea. `cur_ptr` never passes `cur_end`, so the
        // subtraction below cannot wrap.
        if size > self.cur_end.addr() - p.addr() {
            return self.alloc_slow(len, size);
        }
        // SAFETY: `size` fits in what is left of the current segment by the
        // check above, so this lands inside the allocation or one past its end.
        let np = unsafe { p.add(size) };
        self.cur_ptr = np;
        self.allocated += size as u64;

        // SAFETY: `p` is the start of a `size` byte run inside the current
        // segment that has not been handed out before, because the bump pointer
        // only moves forward and no two allocations overlap.
        let bytes = unsafe { core::slice::from_raw_parts_mut(p, len) };
        // SAFETY: `p` is inside the current segment, so the difference is a
        // valid in bounds offset.
        let within = unsafe { p.offset_from(self.cur_start) } as u64;
        // SAFETY: the offset is below 4 PiB by construction, since it is a
        // segment index shifted left by 21 plus an offset below 2 MiB.
        let addr = unsafe { Addr::new_unchecked(Space::Arena, self.cur_base_offset + within) };
        Some((addr, bytes))
    }

    #[cold]
    #[inline(never)]
    fn alloc_slow(&mut self, len: usize, size: usize) -> Option<(Addr, &mut [u8])> {
        debug_assert!(size <= MAX_ALLOC);
        debug_assert!(len <= size);
        // Retire the segment we are leaving by writing its final bump back.
        self.checkpoint_current();
        self.push_segment();

        let p = self.cur_ptr;
        // SAFETY: a fresh segment has `MAX_ALLOC` bytes free and `size` fits.
        let np = unsafe { p.add(size) };
        self.cur_ptr = np;
        self.allocated += size as u64;

        // SAFETY: as in `alloc`, on a segment that was just created. The slice
        // is `len` rather than `size` so that both paths hand back the same
        // thing and a caller cannot tell which one it took.
        let bytes = unsafe { core::slice::from_raw_parts_mut(p, len) };
        // SAFETY: as in `alloc`.
        let within = unsafe { p.offset_from(self.cur_start) } as u64;
        // SAFETY: as in `alloc`.
        let addr = unsafe { Addr::new_unchecked(Space::Arena, self.cur_base_offset + within) };
        Some((addr, bytes))
    }

    /// Copy `data` into the arena.
    ///
    /// The common case for a key or a small value: one bump and one memcpy,
    /// which is what `05` section 3.3 promises for `SINTERSTORE` and friends.
    #[inline]
    pub fn put(&mut self, data: &[u8]) -> Option<Addr> {
        let (addr, out) = self.alloc(data.len())?;
        out.copy_from_slice(data);
        Some(addr)
    }

    /// Read `len` bytes at `addr`.
    ///
    /// # Panics
    ///
    /// If `addr` is not an arena address, or if the run would leave the
    /// segment. Both mean the index and the arena disagree, which is corruption
    /// rather than a condition to recover from.
    #[inline]
    pub fn get(&self, addr: Addr, len: usize) -> &[u8] {
        let ptr = self.resolve(addr, len);
        // SAFETY: `resolve` checked the space, the segment index and that the
        // whole run is inside that segment.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }

    /// Mutable form of [`Arena::get`].
    #[inline]
    pub fn get_mut(&mut self, addr: Addr, len: usize) -> &mut [u8] {
        let ptr = self.resolve(addr, len);
        // SAFETY: as `get`, and `&mut self` proves no other view is live.
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    #[inline]
    fn resolve(&self, addr: Addr, len: usize) -> *mut u8 {
        assert_eq!(
            addr.space(),
            Some(Space::Arena),
            "address does not point into the arena"
        );
        let off = addr.offset();
        let seg = (off >> SEGMENT_SHIFT) as usize;
        let within = (off & (SEGMENT_SIZE as u64 - 1)) as usize;
        let s = self
            .segs
            .get(seg)
            .unwrap_or_else(|| panic!("arena address names segment {seg}, which does not exist"));
        assert!(
            within >= HEADER_SIZE && within + len <= SEGMENT_SIZE,
            "arena run at {within}+{len} leaves segment {seg}"
        );
        // SAFETY: `within + len <= SEGMENT_SIZE` was just checked.
        unsafe { s.base.as_ptr().add(within) }
    }

    /// Record that `len` bytes at `addr` are no longer live.
    ///
    /// Does not reuse the space. It raises the owning segment's dead byte
    /// counter, which is what makes the segment a compaction candidate once it
    /// passes [`Arena::COMPACT_RATIO`].
    #[inline]
    pub fn free(&mut self, addr: Addr, len: usize) {
        assert_eq!(
            addr.space(),
            Some(Space::Arena),
            "address does not point into the arena"
        );
        let size = (len + (ALIGN - 1)) & !(ALIGN - 1);
        let seg = (addr.offset() >> SEGMENT_SHIFT) as usize;
        let s = &mut self.segs[seg];
        let h = s.header_mut();
        h.dead_bytes = (h.dead_bytes + size as u64).min(SEGMENT_SIZE as u64);
        self.allocated = self.allocated.saturating_sub(size as u64);
    }

    /// The dead byte fraction at which a segment is queued for compaction.
    pub const COMPACT_RATIO: f64 = 0.5;

    /// Segments whose dead byte fraction has passed [`Arena::COMPACT_RATIO`].
    ///
    /// The current segment is never a candidate, because it is still being
    /// filled and its dead fraction is not yet meaningful.
    pub fn compaction_candidates(&self) -> Vec<usize> {
        let threshold = (SEGMENT_SIZE as f64 * Self::COMPACT_RATIO) as u64;
        (0..self.segs.len())
            .filter(|&i| i != self.cur)
            .filter(|&i| self.segs[i].header().dead_bytes >= threshold)
            .collect()
    }

    /// Dead bytes in one segment.
    pub fn dead_bytes(&self, seg: usize) -> u64 {
        self.segs[seg].header().dead_bytes
    }

    /// How many segments exist.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segs.len()
    }

    /// Bytes handed out and not yet freed.
    #[inline]
    pub fn live_bytes(&self) -> u64 {
        self.allocated
    }

    /// Bytes of address space held, including the space dead bytes occupy.
    #[inline]
    pub fn reserved_bytes(&self) -> u64 {
        (self.segs.len() * SEGMENT_SIZE) as u64
    }

    /// Bytes still free in the segment currently being bumped.
    #[inline]
    pub fn remaining_in_current(&self) -> usize {
        // SAFETY: both pointers are inside the same allocation, with `cur_ptr`
        // never past `cur_end`.
        (unsafe { self.cur_end.offset_from(self.cur_ptr) }) as usize
    }

    /// Write the live bump cursor into the current segment's header.
    ///
    /// Called on segment switch, and by anything that is about to read headers
    /// as though they were authoritative: a checkpoint, a compaction pass, or a
    /// test.
    #[inline]
    pub fn sync_headers(&mut self) {
        self.checkpoint_current();
    }

    fn checkpoint_current(&mut self) {
        if self.segs.is_empty() {
            return;
        }
        // SAFETY: `cur_ptr` and `cur_start` are in the same allocation.
        let bump = unsafe { self.cur_ptr.offset_from(self.cur_start) } as u64;
        let cur = self.cur;
        self.segs[cur].header_mut().bump = bump;
    }

    /// The recorded bump of a segment, as its header holds it.
    ///
    /// Call [`Arena::sync_headers`] first if the current segment matters.
    pub fn recorded_bump(&self, seg: usize) -> u64 {
        self.segs[seg].header().bump
    }

    /// Mark a segment as retired at `epoch`.
    ///
    /// It is not freed here. `04` section 4 reclaims it once the shard's epoch
    /// has advanced by two, which is when no reference from a previous loop
    /// iteration can still exist.
    pub fn retire(&mut self, seg: usize, epoch: u32) {
        assert_ne!(seg, self.cur, "cannot retire the segment being bumped");
        self.segs[seg].header_mut().epoch_retired = epoch;
    }

    /// The epoch a segment was retired at, or 0 if it is live.
    pub fn epoch_retired(&self, seg: usize) -> u32 {
        self.segs[seg].header().epoch_retired
    }
}

impl Default for Arena {
    fn default() -> Arena {
        Arena::new()
    }
}

impl core::fmt::Debug for Arena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Arena")
            .field("segments", &self.segs.len())
            .field("current", &self.cur)
            .field("live_bytes", &self.allocated)
            .field("reserved_bytes", &self.reserved_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_and_addresses() {
        let mut a = Arena::new();
        let (a1, _) = a.alloc(1).unwrap();
        let (a2, _) = a.alloc(1).unwrap();
        assert_eq!(a1.space(), Some(Space::Arena));
        // One byte rounds up to ALIGN, so consecutive allocations are ALIGN apart.
        assert_eq!(a2.offset() - a1.offset(), ALIGN as u64);
        assert_eq!(a1.offset() % ALIGN as u64, 0);
        assert_eq!(a1.offset(), HEADER_SIZE as u64);
    }

    #[test]
    fn round_trips_bytes() {
        let mut a = Arena::new();
        let addr = a.put(b"hello world").unwrap();
        assert_eq!(a.get(addr, 11), b"hello world");
    }

    #[test]
    fn many_values_all_survive() {
        let mut a = Arena::new();
        let mut addrs = Vec::new();
        for i in 0..10_000u32 {
            let v = format!("value-{i}");
            addrs.push((a.put(v.as_bytes()).unwrap(), v));
        }
        for (addr, want) in &addrs {
            assert_eq!(a.get(*addr, want.len()), want.as_bytes());
        }
    }

    #[test]
    fn grows_past_one_segment() {
        let mut a = Arena::new();
        let chunk = vec![7u8; 64 * 1024];
        // 2 MiB of 64 KiB chunks is 32, so 100 crosses segments several times.
        let addrs: Vec<Addr> = (0..100).map(|_| a.put(&chunk).unwrap()).collect();
        assert!(a.segment_count() > 1, "never grew");
        for addr in addrs {
            // Compared as slices rather than with a byte at a time loop. Both
            // check all six megabytes, but slice equality on bytes bottoms out
            // in memcmp, which Miri runs as a shim instead of interpreting, and
            // that is the difference between this test taking four minutes
            // under Miri and taking seconds. Nothing is skipped to get it.
            assert_eq!(a.get(addr, chunk.len()), chunk.as_slice());
        }
    }

    #[test]
    fn values_stay_put_across_growth() {
        // The property that makes an address stable: growing the arena must not
        // move anything already handed out.
        let mut a = Arena::new();
        let first = a.put(b"first").unwrap();
        let chunk = vec![0u8; 512 * 1024];
        for _ in 0..20 {
            a.put(&chunk).unwrap();
        }
        assert!(a.segment_count() > 1);
        assert_eq!(a.get(first, 5), b"first");
    }

    #[test]
    fn oversized_allocation_is_refused_not_grown() {
        let mut a = Arena::new();
        assert!(a.alloc(MAX_ALLOC).is_some());
        let mut b = Arena::new();
        assert!(b.alloc(MAX_ALLOC + 1).is_none());
        assert_eq!(b.segment_count(), 1, "a refusal must not allocate");
    }

    #[test]
    fn largest_allocation_fits_exactly() {
        let mut a = Arena::new();
        let (addr, bytes) = a.alloc(MAX_ALLOC).unwrap();
        assert_eq!(bytes.len(), MAX_ALLOC);
        assert_eq!(a.remaining_in_current(), 0);
        assert_eq!(addr.offset(), HEADER_SIZE as u64);
    }

    #[test]
    fn free_accounts_dead_bytes() {
        let mut a = Arena::new();
        let addr = a.put(&[1u8; 1024]).unwrap();
        assert_eq!(a.dead_bytes(0), 0);
        assert_eq!(a.live_bytes(), 1024);
        a.free(addr, 1024);
        assert_eq!(a.dead_bytes(0), 1024);
        assert_eq!(a.live_bytes(), 0);
    }

    #[test]
    fn a_half_dead_segment_becomes_a_candidate() {
        let mut a = Arena::new();
        let chunk = vec![0u8; 128 * 1024];
        let mut addrs = Vec::new();
        // Fill segment 0 and start segment 1 so that segment 0 is not current.
        while a.segment_count() < 2 {
            addrs.push(a.put(&chunk).unwrap());
        }
        assert!(a.compaction_candidates().is_empty());
        // Kill half of segment 0.
        for addr in addrs
            .iter()
            .filter(|x| x.offset() < SEGMENT_SIZE as u64)
            .take(8)
        {
            a.free(*addr, chunk.len());
        }
        assert_eq!(a.compaction_candidates(), vec![0]);
    }

    #[test]
    fn headers_record_the_bump_after_sync() {
        let mut a = Arena::new();
        a.put(&[0u8; 100]).unwrap();
        a.sync_headers();
        assert_eq!(a.recorded_bump(0), HEADER_SIZE as u64 + 112);
    }

    #[test]
    fn a_filled_segment_records_its_own_bump() {
        let mut a = Arena::new();
        let chunk = vec![0u8; 256 * 1024];
        while a.segment_count() < 2 {
            a.put(&chunk).unwrap();
        }
        // Segment 0 was checkpointed when it was left behind.
        assert!(a.recorded_bump(0) > HEADER_SIZE as u64);
    }

    #[test]
    fn retirement_records_an_epoch() {
        let mut a = Arena::new();
        let chunk = vec![0u8; 256 * 1024];
        while a.segment_count() < 2 {
            a.put(&chunk).unwrap();
        }
        assert_eq!(a.epoch_retired(0), 0);
        a.retire(0, 42);
        assert_eq!(a.epoch_retired(0), 42);
    }

    #[test]
    #[should_panic(expected = "does not point into the arena")]
    fn a_log_address_is_refused() {
        let a = Arena::new();
        a.get(Addr::new(Space::Log, 64), 8);
    }

    #[test]
    #[should_panic(expected = "which does not exist")]
    fn an_address_past_the_end_is_refused() {
        let a = Arena::new();
        a.get(Addr::new(Space::Arena, 99u64 << SEGMENT_SHIFT), 8);
    }

    #[test]
    #[should_panic(expected = "leaves segment")]
    fn a_run_that_leaves_its_segment_is_refused() {
        let a = Arena::new();
        a.get(Addr::new(Space::Arena, SEGMENT_SIZE as u64 - 8), 64);
    }

    #[test]
    fn zero_length_allocation_is_legal() {
        // A zero length value is a real Redis value, so this must not be a
        // special case at any call site.
        let mut a = Arena::new();
        let addr = a.put(b"").unwrap();
        assert_eq!(a.get(addr, 0), b"");
    }
}
