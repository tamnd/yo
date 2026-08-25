//! One shim so the lock free code can be built twice.
//!
//! Normally these are the standard library's types. Under `--cfg loom` they are
//! loom's, which look the same but record every load, store and interleaving so
//! that a model checker can try all of them. `05` section 1.4 asks for the lane
//! to be proved rather than argued about, and this is what makes that possible
//! without keeping two copies of the code.
//!
//! The only real difference is `UnsafeCell`. Loom's version hands out its
//! pointer through a closure so it can see the access; the standard one just
//! returns it. The shim below gives both the closure shape, which costs nothing
//! in a release build.

#[cfg(loom)]
pub(crate) use loom::sync::Arc;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(loom))]
pub(crate) use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
pub(crate) use std::sync::Arc;

/// A cell whose contents may be mutated through a shared reference.
#[cfg(not(loom))]
#[derive(Debug)]
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
}

/// A cell whose contents may be mutated through a shared reference.
#[cfg(loom)]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(loom::cell::UnsafeCell<T>);

#[cfg(loom)]
impl<T> UnsafeCell<T> {
    pub(crate) fn new(value: T) -> UnsafeCell<T> {
        UnsafeCell(loom::cell::UnsafeCell::new(value))
    }

    pub(crate) fn with<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        self.0.with(|p| f(p.cast_mut()))
    }
}
