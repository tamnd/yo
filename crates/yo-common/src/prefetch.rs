//! Telling the cache what the next walk is going to want.
//!
//! `04` section 3 walks a drained batch twice. The first walk works out which
//! index bucket each command will land in and asks for that line; the second
//! walk executes, and by then the line is on its way or already there. The
//! whole batch is the prefetch distance, which is 64 rather than Valkey's or
//! Redis 8.4's 16, because Y1 means there is no lock held across the window and
//! no other thread that can invalidate a bucket between the ask and the use.
//!
//! There is no stable portable intrinsic for this, so there are three
//! implementations here and they are all one instruction. x86_64 gets
//! `prefetcht0`, aarch64 gets `prfm pldl1keep`, and anything else gets nothing
//! at all, because a hint that has to be emulated is not a hint. Miri also gets
//! nothing, since it does not run inline assembly and there is no correctness
//! in here for it to check.

/// Ask for the cache line at `p`, for reading, into every level of cache.
///
/// A hint and only a hint. It cannot fault, it cannot fail, and it does not
/// change what any later load returns. The only thing it can do wrong is be
/// pointed at a line nobody wants, which costs bandwidth and nothing else.
///
/// The pointer is not dereferenced, so it does not have to be aligned and it
/// does not have to be readable, but it should point at something real or the
/// prefetch is just noise.
#[inline(always)]
pub fn prefetch_read(p: *const u8) {
    // Miri does not do inline assembly and has nothing to say about a hint, so
    // it gets the version where the hint is not there.
    #[cfg(miri)]
    let _ = p;

    #[cfg(all(not(miri), target_arch = "x86_64"))]
    // SAFETY: `_mm_prefetch` is a hint. It does not read through the pointer,
    // it cannot fault whatever the pointer holds, and it is available on every
    // x86_64 because SSE is part of the baseline.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p.cast::<i8>(), core::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(all(not(miri), target_arch = "aarch64"))]
    // SAFETY: `prfm` is a hint. The architecture defines it as having no effect
    // other than on performance, it does not fault on an address the load would
    // have faulted on, and the operand is only read.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) p,
            options(nostack, readonly, preserves_flags),
        );
    }

    #[cfg(all(not(miri), not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
    let _ = p;
}

/// The same hint for a reference, which is the shape the call sites have.
#[inline(always)]
pub fn prefetch<T>(r: &T) {
    prefetch_read(core::ptr::from_ref(r).cast::<u8>());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// There is nothing to assert about a hint except that it happened and the
    /// program carried on, which is exactly what this checks. It is here so
    /// that a build for an architecture without one of the two instructions
    /// still compiles and runs this path.
    #[test]
    fn a_hint_changes_nothing_it_can_be_asked_about() {
        let v: Vec<u64> = (0..1024).collect();
        for i in (0..v.len()).step_by(8) {
            prefetch(&v[i]);
        }
        assert_eq!(v[1023], 1023);
        prefetch_read(v.as_ptr().cast());
        assert_eq!(v.iter().sum::<u64>(), (0..1024u64).sum());
    }

    /// A pointer past the end is still only a hint. This is the case that would
    /// be a segfault if the implementation ever became a load.
    #[test]
    fn a_line_nobody_owns_is_still_only_a_hint() {
        let v = [1u8, 2, 3, 4];
        // SAFETY: `add` on the one past the end pointer is in bounds for the
        // pointer arithmetic rules, and nothing dereferences it here.
        let past = unsafe { v.as_ptr().add(v.len()) };
        prefetch_read(past);
        assert_eq!(v[0], 1);
    }
}
