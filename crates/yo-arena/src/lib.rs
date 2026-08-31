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
    /// Whether this segment is on the free list.
    ///
    /// Out here rather than in the header because the header is in the segment,
    /// and a segment that has given its pages back answers a read of its header
    /// by faulting one of them straight back in. Anything that looks at every
    /// segment has to be able to skip the free ones without touching them.
    /// On the free list, and therefore with its pages handed back: the mapping
    /// stays, so every address in it is as valid as it was, but nothing may be
    /// read out of it before it is written. The kernel refills on the first
    /// touch and what it refills with is not what was there.
    free: bool,
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

        Segment { base, free: false }
    }

    /// Hand the pages back to the kernel, keeping the mapping.
    ///
    /// For a segment that has been compacted and is waiting on the free list.
    /// Nothing is stored in it, but the pages it wrote on are resident until
    /// somebody says otherwise, and resident is what a process is measured on:
    /// two megabytes of a store's own garbage, held against it in `INFO memory`
    /// and in `ps`, for bytes that are not keeping anything.
    ///
    /// The address range stays valid, which is why this is `MADV_DONTNEED` and
    /// not a `dealloc`. Freeing the allocation would take the addresses with
    /// it, and an address here is a segment index that other addresses are
    /// built from, so the segment has to keep its place in the list even when
    /// it is keeping nothing else.
    ///
    /// Linux only. `MADV_DONTNEED` there means the next read of a page gives
    /// zeroes, which is a promise the caller does not need but the kernel does
    /// keep. On the BSDs the same name means something looser and on Windows
    /// there is no `madvise` at all, so those platforms hold the pages. That
    /// costs address space that was already reserved and nothing else.
    fn decay(&mut self) {
        self.free = true;
        // Miri has no kernel to ask, as in `new`.
        #[cfg(all(target_os = "linux", not(miri)))]
        {
            // SAFETY: `base` points at `SEGMENT_SIZE` bytes this segment owns.
            // `MADV_DONTNEED` on a private mapping drops the pages and leaves
            // the mapping, so every pointer into it stays valid, and nothing
            // reads out of a free segment before `revive` has rewritten it.
            unsafe {
                libc::madvise(
                    self.base.as_ptr().cast::<libc::c_void>(),
                    SEGMENT_SIZE,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    /// Put a header back on a segment coming off the free list.
    ///
    /// Unconditional, because a segment that has given its pages back reads as
    /// zeroes and the header has to be there before anything is bumped into
    /// it.
    fn revive(&mut self) {
        self.free = false;
        // SAFETY: as `new`. The mapping was never given up, only its pages.
        unsafe {
            self.base.cast::<Header>().write(Header {
                bump: HEADER_SIZE as u64,
                dead_bytes: 0,
                epoch_retired: 0,
                flags: 0,
                next: 0,
            });
        }
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
    /// Segments with nothing live in them, ready to be bumped again.
    ///
    /// A segment goes on here only through [`Arena::reclaim`], which is only
    /// called after compaction has moved every live record out of it. Reusing
    /// the segment index is safe because every address that ever pointed into
    /// it is dead: the index holds the new address of everything that moved,
    /// and nothing else can name arena bytes.
    free_segs: Vec<usize>,
    /// Dead bytes across every segment, so that deciding whether compaction is
    /// worth doing is one comparison rather than a walk over the headers.
    dead_total: u64,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Arena {
    /// A new arena holding nothing at all.
    ///
    /// The first segment waits for the first allocation. A server has sixteen
    /// databases and on almost every server fifteen of them stay empty for the
    /// life of the process, so a segment each is thirty megabytes of address
    /// space held for nobody and thirty megabytes added to what `INFO memory`
    /// reports before a single key exists.
    ///
    /// The empty arena works because the bump cursor starts as three null
    /// pointers and `cur_end - cur_ptr` is then zero, so the first `alloc`
    /// cannot fit and takes the slow path, which is where segments come from.
    /// No branch is added to the fast path for this.
    pub fn new() -> Arena {
        Arena {
            segs: Vec::new(),
            cur: 0,
            cur_base_offset: 0,
            cur_ptr: core::ptr::null_mut(),
            cur_end: core::ptr::null_mut(),
            cur_start: core::ptr::null_mut(),
            allocated: 0,
            free_segs: Vec::new(),
            dead_total: 0,
            _not_send_sync: PhantomData,
        }
    }

    /// Start bumping a new segment: one off the free list if compaction has
    /// emptied one, and a fresh allocation otherwise.
    fn push_segment(&mut self) {
        let next = match self.free_segs.pop() {
            Some(seg) => {
                // Its pages went back to the kernel when it was freed, so the
                // header has to be written again before anything is bumped
                // into it.
                self.segs[seg].revive();
                seg
            }
            None => {
                let seg = Segment::new();
                yo_alloc::allow(|| self.segs.push(seg));
                self.segs.len() - 1
            }
        };
        let base = self.segs[next].base.as_ptr();
        self.cur = next;
        self.cur_base_offset = (self.cur as u64) << SEGMENT_SHIFT;
        self.cur_start = base;
        // SAFETY: both offsets are within the `SEGMENT_SIZE` allocation.
        unsafe {
            self.cur_ptr = base.add(HEADER_SIZE);
            self.cur_end = base.add(SEGMENT_SIZE);
        }
    }

    /// Bytes a request of `len` actually takes: rounded up to [`ALIGN`], and
    /// never zero.
    ///
    /// The floor is what lets an arena start with no segment at all. An empty
    /// arena has a cursor with nothing between it and its end, so any request
    /// bigger than nothing misses and takes the slow path, which is where a
    /// segment comes from. Without the floor a zero length request would fit in
    /// a segment that does not exist and hand back a slice built on a null
    /// pointer. It is also the more honest answer: two zero length allocations
    /// are two different things and deserve two different addresses.
    #[inline(always)]
    const fn slot(len: usize) -> usize {
        let rounded = (len + (ALIGN - 1)) & !(ALIGN - 1);
        if rounded == 0 { ALIGN } else { rounded }
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
        let size = Self::slot(len);
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

    /// Copy a `len` byte run to a fresh allocation and return where it landed.
    ///
    /// What compaction moves a record with. The obvious way to write it is to
    /// read the bytes into a `Vec` and then write the `Vec` back, and that is a
    /// heap allocation per record on a thread that is not allowed one. This
    /// goes arena to arena with one `memcpy` and no allocator call.
    ///
    /// # Panics
    ///
    /// As [`Arena::get`], and if `len` is over [`MAX_ALLOC`].
    pub fn copy_within(&mut self, src: Addr, len: usize) -> Addr {
        let from = self.resolve(src, len);
        let (addr, out) = self
            .alloc(len)
            .expect("a run already in the arena fits in the arena");
        // SAFETY: `from` is a `len` byte run inside a segment, `out` is a fresh
        // `len` byte allocation, and a fresh allocation cannot overlap a run
        // that is already live because the bump pointer only moves forward.
        unsafe { core::ptr::copy_nonoverlapping(from, out.as_mut_ptr(), len) };
        addr
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
        let size = Self::slot(len);
        let seg = (addr.offset() >> SEGMENT_SHIFT) as usize;
        let s = &mut self.segs[seg];
        let h = s.header_mut();
        let was = h.dead_bytes;
        h.dead_bytes = (h.dead_bytes + size as u64).min(SEGMENT_SIZE as u64);
        // The running total takes what the segment actually took, not what was
        // asked for, so that the clamp above cannot drift the two apart.
        self.dead_total += h.dead_bytes - was;
        self.allocated = self.allocated.saturating_sub(size as u64);
    }

    /// Put an emptied segment back on the free list.
    ///
    /// The caller has to have moved every live record out of it first, which is
    /// what [`compaction_candidates`](Arena::compaction_candidates) exists to
    /// pick a segment for. There is no way to check that here: the arena has
    /// never known which of its bytes are live, only how many, which is the
    /// whole reason allocation costs an add.
    ///
    /// The segment keeps its index and its memory. Only its header is reset, so
    /// the next [`Arena::alloc`] that runs out of room bumps through this one
    /// again instead of asking the system for two more megabytes. Without this
    /// a server that overwrites the same keys grows without limit: aki's L11,
    /// where the `*STORE` family ran at 0.30 to 0.55x because the arena was
    /// grow only.
    ///
    /// # Panics
    ///
    /// If `seg` is the segment being bumped, which still has a live cursor
    /// pointing into it.
    pub fn reclaim(&mut self, seg: usize) {
        assert_ne!(seg, self.cur, "cannot reclaim the segment being bumped");
        debug_assert!(
            !self.free_segs.contains(&seg),
            "segment {seg} is already on the free list"
        );
        let h = self.segs[seg].header_mut();
        self.dead_total -= h.dead_bytes;
        h.bump = HEADER_SIZE as u64;
        h.dead_bytes = 0;
        h.epoch_retired = 0;
        h.flags = 0;
        // Every free segment gives its pages back, with none kept warm. Keeping
        // one warm was tried, on the reasoning that a store which is compacting
        // is usually about to want a segment and refaulting two megabytes to
        // hand it straight back is work for nothing. It cost 4 MB of resident
        // set on the standard load, which is more than the whole margin the
        // memory gate was missing by, and the work it saves is one madvise and
        // one huge page fault per two megabytes written.
        self.segs[seg].decay();
        yo_alloc::allow(|| self.free_segs.push(seg));
    }

    /// Dead bytes across every segment.
    #[inline]
    pub fn dead_bytes_total(&self) -> u64 {
        self.dead_total
    }

    /// Segments that are empty and waiting to be bumped again.
    #[inline]
    pub fn free_segments(&self) -> usize {
        self.free_segs.len()
    }

    /// Whether one segment is on the free list.
    ///
    /// The flag lives beside the mapping and not in it, because on Linux a free
    /// segment has given its pages back and everything in it, header included,
    /// reads as zeroes until something bumps through it again. Asking the
    /// header whether a segment is free would therefore answer one thing on
    /// Linux and another on macOS, which is exactly the sort of question this
    /// exists to stop a caller from asking.
    #[inline]
    pub fn is_free(&self, seg: usize) -> bool {
        self.segs[seg].free
    }

    /// The dead byte fraction at which a segment is worth compacting.
    ///
    /// A quarter and not a half. The number is a trade and both directions are
    /// real: at a half a store holds twice what it is keeping, which loses the
    /// memory column of M2's gate outright, and at a tenth every byte of
    /// garbage costs nine bytes of copying to get back. A quarter holds about a
    /// third more than it is keeping and copies three bytes per byte, and only
    /// on the bytes that were overwritten in the first place. A workload that
    /// writes each key once never compacts at all.
    pub const COMPACT_RATIO: f64 = 0.25;

    /// The fraction of everything held that has to be dead before compaction is
    /// worth starting at all.
    ///
    /// The per segment ratio says which segment, this says whether. Without it
    /// a store with one half dead segment out of forty compacts that segment,
    /// gains 2 MiB it did not need and pays for it on a command path.
    ///
    /// An eighth, which is where it stops mattering. In the steady state a
    /// store holds about `live / (1 - this)`, so an eighth is a seventh more
    /// than it is keeping. Going lower buys nothing, because below this the
    /// binding constraint is [`Arena::COMPACT_RATIO`]: no segment qualifies
    /// until one of them is a quarter dead, and until one does there is nothing
    /// to compact however keen this is to start. Measured over 100000 keys of
    /// 64 bytes rewritten between four and thirteen times, a quarter holds 7
    /// segments, an eighth holds 6, and a sixteenth and a thirty second hold 7
    /// again. Run time was the same to the tenth of a second at every one.
    pub const GARBAGE_RATIO: f64 = 0.125;

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

    /// How many segments have passed [`Arena::COMPACT_RATIO`].
    ///
    /// The same test as [`Arena::compaction_candidates`] without the vector.
    /// The compactor asks this to size its next slice of work and the answer it
    /// wants is a depth, not a list, and it asks on a path that is not allowed
    /// to allocate.
    ///
    /// Free segments are skipped before their header is touched rather than
    /// after. A free segment has given its pages back, so reading its header to
    /// find out it has nothing dead in it would fault two megabytes back in to
    /// answer a question about a segment that is empty.
    pub fn candidate_count(&self) -> usize {
        let threshold = (SEGMENT_SIZE as f64 * Self::COMPACT_RATIO) as u64;
        (0..self.segs.len())
            .filter(|&i| i != self.cur && !self.segs[i].free)
            .filter(|&i| self.segs[i].header().dead_bytes >= threshold)
            .count()
    }

    /// The candidate with the most dead bytes, or `None` if there is nothing
    /// worth moving.
    ///
    /// The emptiest segment first, because it is the one whose live records
    /// cost the least to move and whose two megabytes come back either way.
    ///
    /// The first test is the one that runs on a healthy store, and it is a
    /// single comparison against a counter that `free` keeps. Walking the
    /// headers only happens once the store is actually holding garbage.
    pub fn worst_candidate(&self) -> Option<usize> {
        let enough = (self.reserved_bytes() as f64 * Self::GARBAGE_RATIO) as u64;
        if self.dead_total < enough {
            return None;
        }
        let threshold = (SEGMENT_SIZE as f64 * Self::COMPACT_RATIO) as u64;
        (0..self.segs.len())
            // A free segment has nothing to move, and reading its header to
            // find that out would fault a page back in for one that has given
            // its pages up.
            .filter(|&i| i != self.cur && !self.segs[i].free)
            .map(|i| (self.segs[i].header().dead_bytes, i))
            .filter(|&(dead, _)| dead >= threshold)
            .max()
            .map(|(_, i)| i)
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

    /// The segment being bumped, which is the one nothing may reclaim.
    #[inline]
    pub fn current_segment(&self) -> usize {
        self.cur
    }

    /// Bytes handed out and not yet freed.
    #[inline]
    pub fn live_bytes(&self) -> u64 {
        self.allocated
    }

    /// Bytes actually held, including the space dead bytes occupy.
    ///
    /// A decayed segment is not in here. Its address range is still reserved
    /// and always will be, because addresses are built out of segment indices,
    /// but the pages behind it went back to the kernel and this is the number
    /// that ends up in `INFO memory`, where address space nobody is paying for
    /// would be a lie in the direction that flatters us.
    #[inline]
    pub fn reserved_bytes(&self) -> u64 {
        (self.resident_segments() * SEGMENT_SIZE) as u64
    }

    /// Segments whose pages are real, which is every segment that is not on the
    /// free list.
    #[inline]
    pub fn resident_segments(&self) -> usize {
        self.segs.len() - self.free_segs.len()
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
        assert_eq!(b.segment_count(), 0, "a refusal must not allocate");
        assert!(b.alloc(1).is_some());
        assert_eq!(b.segment_count(), 1, "the first allocation takes a segment");
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
    fn a_reclaimed_segment_is_bumped_through_again() {
        let mut a = Arena::new();
        let chunk = vec![0u8; 256 * 1024];
        while a.segment_count() < 2 {
            a.put(&chunk).unwrap();
        }
        let held = a.segment_count();
        a.reclaim(0);
        assert_eq!(a.free_segments(), 1);
        assert_eq!(a.dead_bytes_total(), 0, "reclaim clears the segment's dead");

        // Fill the current segment. The next one has to be the reclaimed one,
        // so nothing is asked of the system.
        while a.free_segments() > 0 {
            a.put(&chunk).unwrap();
        }
        assert_eq!(
            a.segment_count(),
            held,
            "grew while holding an empty segment"
        );
        assert_eq!(a.current_segment(), 0);

        // And it is being written from the top, not from where it was left.
        let (addr, _) = a.alloc(16).unwrap();
        assert!(addr.offset() < SEGMENT_SIZE as u64);
    }

    /// A free segment gives its pages back, and comes back from that intact
    /// when something wants it again.
    ///
    /// The pages are the point. An emptied segment that keeps them is two
    /// megabytes of a store's own garbage counted against it by `ps` and by
    /// `INFO memory` for holding nothing.
    #[test]
    fn a_free_segment_gives_its_pages_back() {
        let mut a = Arena::new();
        let chunk = vec![7u8; 256 * 1024];
        while a.segment_count() < 4 {
            a.put(&chunk).unwrap();
        }
        let held = a.reserved_bytes();

        a.reclaim(0);
        a.reclaim(1);
        a.reclaim(2);
        assert_eq!(a.free_segments(), 3);
        assert_eq!(
            a.reserved_bytes(),
            held - 3 * SEGMENT_SIZE as u64,
            "all three should have gone back to the kernel"
        );

        // Take all three back, and read out of the last one to prove the pages
        // came with them.
        while a.free_segments() > 0 {
            a.put(&chunk).unwrap();
        }
        assert_eq!(a.reserved_bytes(), held, "all four are in use again");
        let addr = a.put(&chunk).unwrap();
        assert_eq!(a.get(addr, chunk.len()), &chunk[..]);
    }

    #[test]
    fn nothing_moves_until_the_garbage_is_worth_it() {
        let mut a = Arena::new();
        let chunk = vec![0u8; 128 * 1024];
        let mut addrs = Vec::new();
        // Ten segments, with one of them half dead. That is 1 MiB of garbage
        // against 20 MiB held, which is not worth a command path pause even
        // though the segment itself is past the per segment ratio.
        while a.segment_count() < 10 {
            addrs.push(a.put(&chunk).unwrap());
        }
        for addr in addrs
            .iter()
            .filter(|x| x.offset() < SEGMENT_SIZE as u64)
            .take(8)
        {
            a.free(*addr, chunk.len());
        }
        assert_eq!(a.compaction_candidates(), vec![0], "segment 0 is half dead");
        assert_eq!(
            a.worst_candidate(),
            None,
            "one segment in ten is not enough"
        );

        // Kill most of the rest and it becomes worth it.
        for addr in addrs.iter().skip(16).take(80) {
            a.free(*addr, chunk.len());
        }
        assert!(a.worst_candidate().is_some());
    }

    #[test]
    #[should_panic(expected = "cannot reclaim the segment being bumped")]
    fn the_segment_being_written_cannot_be_reclaimed() {
        let mut a = Arena::new();
        a.put(b"x").unwrap();
        a.reclaim(a.current_segment());
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
        let mut a = Arena::new();
        // Something has to be in it, or the complaint is that the segment does
        // not exist rather than that the run leaves it.
        a.put(b"x").unwrap();
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
