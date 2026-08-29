//! A global allocator for this crate's own tests that counts instead of aborting.
//!
//! `yo-alloc` is the production answer to Y7 and it aborts the process on an
//! allocation a shard thread was not supposed to make. That is the right
//! behaviour for a server and the wrong one for a test, where an abort takes
//! every other test in the binary down with it and prints a byte count instead
//! of a name. So the test build of this crate counts.
//!
//! Counting also says more than forbidding does. A claim like "`SPOP` allocates
//! nothing on the way out" is only worth writing down next to the number the old
//! path produced, and a counter can assert both halves: zero here, one per
//! member there. A guard that aborts can only assert the first half.
//!
//! The counter is per thread, because `cargo test` runs tests on several at once
//! and a shared number would be whatever the other tests happened to be doing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// How many times this thread has asked the system allocator for memory.
    ///
    /// Const initialised and holding a type with no destructor, so touching it
    /// neither allocates nor registers anything, which it has to be since the
    /// allocator itself reads it.
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

#[global_allocator]
static TALLY: Tally = Tally;

/// The system allocator with a counter in front of it.
struct Tally;

// SAFETY: every method forwards to `System`, which upholds the `GlobalAlloc`
// contract. The counter is a thread local integer and touches no pointer.
unsafe impl GlobalAlloc for Tally {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A `Vec` that grows is a `realloc` and is exactly the cost being
        // counted, so it counts as an allocation here.
        bump();
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn bump() {
    // `try_with` rather than `with`, because a thread that is being torn down
    // can still allocate after its thread locals have gone, and a panic from
    // inside the allocator is not a recoverable thing.
    let _ = COUNT.try_with(|c| c.set(c.get() + 1));
}

/// Run `f` and answer with its value and how many times it allocated.
pub fn counted<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = COUNT.with(Cell::get);
    let out = f();
    let after = COUNT.with(Cell::get);
    (out, after - before)
}
