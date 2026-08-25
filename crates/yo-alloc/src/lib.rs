//! A global allocator that turns an accidental heap allocation on a shard path
//! into a crash.
//!
//! Y7 says there is no global allocator call on a command path. That is easy to
//! write down and impossible to keep by review once more than one person is in
//! the codebase, because the allocating constructs in Rust are the comfortable
//! ones: `format!`, `to_vec`, `collect`, `Box::new`, a `Vec` that grows, a
//! `String` built to make an error message. Each costs tens of nanoseconds
//! against a 150 ns budget, and none of them looks wrong in a diff.
//!
//! So the rule is enforced instead of reviewed. A shard thread marks itself
//! [`enter_no_alloc`] before the command loop, and from that point any
//! allocation aborts the process with a message naming the size. Setup, arena
//! growth and anything else that legitimately needs the heap wraps itself in
//! [`allow`], which is a visible, greppable, deliberate act.
//!
//! # Cost when it is off
//!
//! The check is one thread local load and a branch, on a path that already
//! calls into the system allocator. It is not measurable next to `malloc`.
//! Non shard threads never set the flag and pay the same single branch.
//!
//! # Using it
//!
//! ```no_run
//! # use yo_alloc::YoAlloc;
//! #[global_allocator]
//! static ALLOC: YoAlloc = YoAlloc::new();
//! ```
//!
//! The engine installs this in its binaries and in its test and bench targets.
//! A library consumer of `yodb` does not get it, because choosing a global
//! allocator is the application's call and never a library's.

#![deny(missing_docs)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Zero means allocation is allowed. Anything above zero forbids it.
    ///
    /// A counter rather than a flag so that [`allow`] nests correctly, which
    /// matters because arena growth can be reached from more than one depth.
    static FORBID: Cell<u32> = const { Cell::new(0) };
}

/// Mark this thread as a shard thread: from here on, allocating aborts.
///
/// Called once by each shard as it enters its loop. There is no matching exit
/// in normal operation because a shard thread never stops being one.
#[inline]
pub fn enter_no_alloc() {
    FORBID.with(|f| f.set(f.get().saturating_add(1)));
}

/// Undo one [`enter_no_alloc`].
///
/// Exists for tests and for the embedded single thread mode (`15` section 7),
/// where the caller's thread is temporarily the shard and then goes back to
/// being the caller's thread.
#[inline]
pub fn exit_no_alloc() {
    FORBID.with(|f| f.set(f.get().saturating_sub(1)));
}

/// Whether allocation is currently forbidden on this thread.
#[inline]
pub fn is_forbidden() -> bool {
    FORBID.with(|f| f.get()) > 0
}

/// Run `f` with allocation permitted, then restore the previous state.
///
/// Every call to this is a claim that the work inside is off the command path.
/// Wrapping a command path in it to silence an abort is the one way to misuse
/// this module, so the calls are meant to be few and easy to find.
#[inline]
pub fn allow<T>(f: impl FnOnce() -> T) -> T {
    let saved = FORBID.with(|c| c.replace(0));
    let guard = Restore(saved);
    let out = f();
    drop(guard);
    out
}

struct Restore(u32);

impl Drop for Restore {
    #[inline]
    fn drop(&mut self) {
        FORBID.with(|c| c.set(self.0));
    }
}

/// The allocator. Delegates to the system allocator and checks the flag first.
#[derive(Debug, Default, Clone, Copy)]
pub struct YoAlloc;

impl YoAlloc {
    /// A new allocator.
    pub const fn new() -> YoAlloc {
        YoAlloc
    }
}

#[cold]
#[inline(never)]
fn violation(layout: Layout, what: &str) -> ! {
    // No formatting machinery here on purpose. `format!` allocates, and this is
    // the one place in the process where allocating is known to be unavailable.
    // Two `write_str` calls and an integer written by hand cost nothing and
    // cannot recurse.
    use std::io::Write as _;
    let mut buf = [0u8; 32];
    let n = write_usize(&mut buf, layout.size());
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(b"yo: allocation on a shard thread: ");
    let _ = err.write_all(what.as_bytes());
    let _ = err.write_all(b" of ");
    let _ = err.write_all(&buf[..n]);
    let _ = err.write_all(
        b" bytes.\nThis is Y7: no global allocator call on a command path.\n\
          Move the allocation to setup, or wrap it in yo_alloc::allow if it is\n\
          genuinely off the command path.\n",
    );
    let _ = err.flush();
    std::process::abort()
}

fn write_usize(buf: &mut [u8; 32], mut v: usize) -> usize {
    if v == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut tmp = [0u8; 32];
    let mut n = 0;
    while v > 0 {
        tmp[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    for i in 0..n {
        buf[i] = tmp[n - 1 - i];
    }
    n
}

// SAFETY: every method forwards to `System`, which upholds the `GlobalAlloc`
// contract. The added check only ever diverges before calling through, so no
// pointer is created, invalidated or leaked by it.
unsafe impl GlobalAlloc for YoAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_forbidden() {
            violation(layout, "alloc");
        }
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc(layout) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_forbidden() {
            violation(layout, "alloc_zeroed");
        }
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if is_forbidden() {
            violation(layout, "realloc");
        }
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Deallocation is deliberately not checked. A value allocated during
        // setup and dropped on the shard thread is normal and harmless, and
        // aborting on it would make the rule unusable. What costs time is the
        // allocation, and that is what is caught.
        //
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_permitted() {
        assert!(!is_forbidden());
    }

    #[test]
    fn enter_and_exit_are_balanced() {
        assert!(!is_forbidden());
        enter_no_alloc();
        assert!(is_forbidden());
        enter_no_alloc();
        assert!(is_forbidden());
        exit_no_alloc();
        assert!(is_forbidden(), "one exit must not undo two enters");
        exit_no_alloc();
        assert!(!is_forbidden());
    }

    #[test]
    fn allow_permits_and_restores() {
        enter_no_alloc();
        assert!(is_forbidden());
        let v = allow(|| {
            assert!(!is_forbidden());
            vec![1u8, 2, 3]
        });
        assert_eq!(v.len(), 3);
        assert!(is_forbidden(), "allow must restore the previous state");
        exit_no_alloc();
    }

    #[test]
    fn allow_nests() {
        enter_no_alloc();
        allow(|| {
            allow(|| assert!(!is_forbidden()));
            assert!(!is_forbidden());
        });
        assert!(is_forbidden());
        exit_no_alloc();
    }

    #[test]
    fn allow_restores_when_the_body_panics() {
        enter_no_alloc();
        let r = std::panic::catch_unwind(|| {
            allow(|| panic!("boom"));
        });
        assert!(r.is_err());
        assert!(
            is_forbidden(),
            "a panic inside allow must not leave the thread permitted"
        );
        exit_no_alloc();
    }

    /// The flag is per thread. A shard marking itself must not affect the
    /// accept loop or a test harness thread.
    #[test]
    fn the_flag_does_not_cross_threads() {
        enter_no_alloc();
        let other = std::thread::spawn(is_forbidden).join().unwrap();
        assert!(!other, "another thread saw this thread's flag");
        exit_no_alloc();
    }

    #[test]
    fn integers_render_without_allocating() {
        let mut buf = [0u8; 32];
        for (v, want) in [
            (0usize, "0"),
            (7, "7"),
            (1024, "1024"),
            (2097152, "2097152"),
        ] {
            let n = write_usize(&mut buf, v);
            assert_eq!(std::str::from_utf8(&buf[..n]).unwrap(), want);
        }
    }
}
