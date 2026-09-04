//! The same code, built twice, so a model checker can read one of them.
//!
//! Normally these are the standard library's types. Under `--cfg loom` they are
//! loom's, which behave the same but record every load, store and interleaving
//! so that every order two threads could have run in gets tried. The lock is
//! the one piece of this crate where being nearly right is being wrong, so it
//! is built on this rather than on the standard types directly.
//!
//! The only real difference is `UnsafeCell`. Loom's version hands out its
//! pointer through a closure so it can see the access; the standard one just
//! returns it. The shim below gives both the closure shape, which costs nothing
//! in a build that is not being checked. `yo-shard` has the same shim for the
//! same reason, and neither crate depends on the other's.

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, Ordering};
#[cfg(loom)]
pub(crate) use loom::thread::yield_now;

#[cfg(not(loom))]
pub(crate) use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(loom))]
pub(crate) use std::thread::yield_now;

/// A cell whose contents may be mutated through a shared reference.
#[cfg(not(loom))]
pub(crate) struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    #[inline(always)]
    pub(crate) const fn new(value: T) -> UnsafeCell<T> {
        UnsafeCell(core::cell::UnsafeCell::new(value))
    }

    /// Run `f` with a pointer to the contents.
    #[inline(always)]
    pub(crate) fn with<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }

    /// Reach the contents through an exclusive reference, which needs nothing.
    #[inline(always)]
    pub(crate) fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    /// Take the value back out.
    #[inline(always)]
    pub(crate) fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

/// A cell whose contents may be mutated through a shared reference.
#[cfg(loom)]
pub(crate) struct UnsafeCell<T>(loom::cell::UnsafeCell<T>);

#[cfg(loom)]
impl<T> UnsafeCell<T> {
    pub(crate) fn new(value: T) -> UnsafeCell<T> {
        UnsafeCell(loom::cell::UnsafeCell::new(value))
    }

    pub(crate) fn with<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        self.0.with(|p| f(p.cast_mut()))
    }

    pub(crate) fn get_mut(&mut self) -> &mut T {
        // SAFETY: loom's cell has no exclusive accessor, and an exclusive
        // reference to the cell is already proof that nothing else is looking.
        self.0.with_mut(|p| unsafe { &mut *p })
    }

    pub(crate) fn into_inner(self) -> T {
        self.0.into_inner()
    }
}
