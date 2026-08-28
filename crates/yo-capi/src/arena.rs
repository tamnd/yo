//! Caller scoped result storage.
//!
//! `dx/02` section 3.1 calls this the single most important performance
//! decision in the ABI, and the reason is arithmetic rather than taste. A
//! binding in a managed language gets a 3x budget against in process Rust, and
//! a `malloc` and `free` per result spends all of it before the host language
//! has done anything. One arena per call scope turns a loop over ten thousand
//! results into one allocation and ten thousand pointer bumps.
//!
//! This is not the engine's record arena. That one hands out addresses, keeps
//! segment headers and gets compacted. This one hands out bytes, is rewound
//! wholesale, and never has to be understood by anyone but the generated `core`
//! layer of a binding.

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// The first chunk. Big enough that a batch of 64 rows never needs a second one,
/// small enough that a binding creating an arena per request is not paying for
/// memory it will not touch.
const FIRST_CHUNK: usize = 64 * 1024;

/// Every allocation is aligned to this, which is enough for any scalar a
/// binding will read out of the arena without a memcpy.
const ALIGN: usize = 16;

/// The byte a chunk is filled with on reset in a debug build.
///
/// Not zero, because zero is what a fresh page already holds and a use after
/// reset would keep working by accident. `0xdb` reads as an obvious sentinel in
/// a hex dump and makes a stale string non NUL terminated.
#[cfg(debug_assertions)]
const POISON: u8 = 0xdb;

struct Chunk {
    ptr: NonNull<u8>,
    cap: usize,
}

impl Chunk {
    fn new(cap: usize) -> Chunk {
        let layout =
            Layout::from_size_align(cap, ALIGN).expect("chunk layout is valid by construction");
        // SAFETY: `cap` is never zero, so this is a legal allocation request.
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Chunk { ptr, cap }
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(self.cap, ALIGN).expect("the layout that allocated it");
        // SAFETY: `ptr` came from `alloc` with this exact layout and is freed once.
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// A bump allocator the caller resets and frees.
pub struct ResultArena {
    chunks: Vec<Chunk>,
    /// Which chunk is being filled.
    current: usize,
    /// How far into the current chunk.
    bump: usize,
    /// Handed out since the last reset, across all chunks.
    used: u64,
    /// The thread that made it. Checked in debug builds, because an arena that
    /// crosses threads is a binding bug and should fail in the binding's own
    /// test suite rather than in someone's production process.
    #[cfg(debug_assertions)]
    owner: std::thread::ThreadId,
}

impl ResultArena {
    /// A new arena with its first chunk already mapped.
    pub fn new() -> ResultArena {
        ResultArena {
            chunks: vec![Chunk::new(FIRST_CHUNK)],
            current: 0,
            bump: 0,
            used: 0,
            #[cfg(debug_assertions)]
            owner: std::thread::current().id(),
        }
    }

    /// Fails loudly in debug if the arena has wandered onto another thread.
    #[inline]
    pub fn check_thread(&self) {
        #[cfg(debug_assertions)]
        assert_eq!(
            self.owner,
            std::thread::current().id(),
            "an arena was used from a thread other than the one that created it"
        );
    }

    /// Bytes handed out since the last reset.
    #[inline]
    pub fn used(&self) -> u64 {
        self.used
    }

    /// Copies `src` into the arena and returns a pointer to the copy.
    ///
    /// The copy stays valid until the next reset or free and not one
    /// instruction longer.
    pub fn put(&mut self, src: &[u8]) -> NonNull<u8> {
        let dst = self.alloc(src.len());
        // SAFETY: `alloc` returned a run of at least `src.len()` bytes that no
        // one else holds, and `src` cannot overlap it because it was live before
        // the run existed.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), src.len()) };
        dst
    }

    /// Copies `src` and appends a NUL, for the message fields in `yo_error`.
    pub fn put_cstr(&mut self, src: &str) -> NonNull<u8> {
        let dst = self.alloc(src.len() + 1);
        // SAFETY: the run is one byte longer than `src`, and the two cannot
        // overlap for the same reason as in `put`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), src.len());
            dst.as_ptr().add(src.len()).write(0);
        }
        dst
    }

    /// Reserves `len` bytes. Zero length still gets a real, aligned, unique
    /// pointer, because handing back null for an empty value would make every
    /// caller special case it.
    pub fn alloc(&mut self, len: usize) -> NonNull<u8> {
        let need = len.max(1).next_multiple_of(ALIGN);
        let chunk = &self.chunks[self.current];
        if need > chunk.cap - self.bump {
            self.grow(need);
        }
        let chunk = &self.chunks[self.current];
        // SAFETY: `bump` is at most `cap` and `need` fits in what is left, both
        // guaranteed above, so this lands inside the chunk.
        let p = unsafe { chunk.ptr.as_ptr().add(self.bump) };
        self.bump += need;
        self.used += len as u64;
        // SAFETY: the chunk pointer is non null and `bump` never leaves it.
        unsafe { NonNull::new_unchecked(p) }
    }

    #[cold]
    fn grow(&mut self, need: usize) {
        // Reuse a chunk we already have before making another one. A binding
        // that resets per batch settles on a working set after a few batches and
        // stops allocating entirely, which is the whole point of reset keeping
        // capacity.
        for i in self.current + 1..self.chunks.len() {
            if self.chunks[i].cap >= need {
                self.current = i;
                self.bump = 0;
                return;
            }
        }
        let last = self.chunks[self.chunks.len() - 1].cap;
        let cap = (last * 2).max(need.next_power_of_two());
        self.chunks.push(Chunk::new(cap));
        self.current = self.chunks.len() - 1;
        self.bump = 0;
    }

    /// Rewinds to the start, keeping every chunk.
    pub fn reset(&mut self) {
        #[cfg(debug_assertions)]
        for chunk in &self.chunks {
            // SAFETY: the arena owns every chunk outright and nothing may hold a
            // live view across a reset, which is the documented contract.
            unsafe { std::ptr::write_bytes(chunk.ptr.as_ptr(), POISON, chunk.cap) };
        }
        self.current = 0;
        self.bump = 0;
        self.used = 0;
    }
}

impl Default for ResultArena {
    fn default() -> ResultArena {
        ResultArena::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_are_aligned_and_distinct() {
        let mut a = ResultArena::new();
        let mut seen = Vec::new();
        for len in [1usize, 7, 16, 17, 100] {
            let p = a.alloc(len);
            assert_eq!(
                p.as_ptr() as usize % ALIGN,
                0,
                "len {len} came back misaligned"
            );
            assert!(
                !seen.contains(&p.as_ptr()),
                "len {len} reused a live pointer"
            );
            seen.push(p.as_ptr());
        }
    }

    #[test]
    fn an_empty_allocation_is_still_a_real_pointer() {
        let mut a = ResultArena::new();
        let p = a.alloc(0);
        let q = a.alloc(0);
        assert_ne!(p.as_ptr(), q.as_ptr());
    }

    #[test]
    fn contents_survive_later_allocations() {
        let mut a = ResultArena::new();
        let mut live = Vec::new();
        for i in 0..2_000u32 {
            let bytes = i.to_le_bytes();
            live.push((a.put(&bytes), bytes));
        }
        for (p, want) in live {
            // SAFETY: every one of these runs is still live, since nothing has
            // been reset or freed.
            let got = unsafe { std::slice::from_raw_parts(p.as_ptr(), 4) };
            assert_eq!(got, want);
        }
    }

    #[test]
    fn reset_keeps_capacity_and_stops_allocating() {
        let mut a = ResultArena::new();
        for _ in 0..64 {
            a.alloc(4096);
        }
        let after_first = a.chunks.len();
        assert!(after_first > 1, "the test did not actually grow the arena");
        for _ in 0..8 {
            a.reset();
            for _ in 0..64 {
                a.alloc(4096);
            }
        }
        assert_eq!(a.chunks.len(), after_first, "reset is not reusing chunks");
    }

    #[test]
    fn used_counts_bytes_asked_for_not_bytes_rounded_up() {
        let mut a = ResultArena::new();
        a.alloc(1);
        a.alloc(30);
        assert_eq!(a.used(), 31);
        a.reset();
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn a_run_larger_than_a_chunk_still_works() {
        let mut a = ResultArena::new();
        let big = vec![7u8; FIRST_CHUNK * 3];
        let p = a.put(&big);
        // SAFETY: the run is live and is exactly this long.
        let got = unsafe { std::slice::from_raw_parts(p.as_ptr(), big.len()) };
        assert_eq!(got, &big[..]);
    }

    #[test]
    fn a_c_string_is_terminated() {
        let mut a = ResultArena::new();
        let p = a.put_cstr("hello");
        // SAFETY: `put_cstr` wrote six bytes here and they are still live.
        let got = unsafe { std::slice::from_raw_parts(p.as_ptr(), 6) };
        assert_eq!(got, b"hello\0");
    }
}
