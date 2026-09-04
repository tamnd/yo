//! One owner at a time, for the state that more than one thread can reach.
//!
//! The engine is built so that most state has a single owner and needs no lock
//! at all. The stripes are the exception. A stripe is a piece of the keyspace,
//! and once a server runs commands on more than one thread the same stripe can
//! be wanted by two of them at once. This is the thing that decides which one
//! gets it.
//!
//! It is a spin lock, and that is a deliberate choice rather than a shortcut. A
//! stripe is held for one command, which is tens or hundreds of nanoseconds, so
//! a waiter that parks in the kernel would spend more time going to sleep and
//! waking up than it would have spent waiting. The wait here is a short spin
//! and then a yield, which is the shape that fits a hold time this short. It is
//! the wrong shape for anything held across a syscall, so nothing held across a
//! syscall should use it.
//!
//! ```
//! use yo_common::lock::Lock;
//!
//! let counter = Lock::new(0u64);
//! *counter.lock() += 1;
//! assert_eq!(*counter.lock(), 1);
//! ```
//!
//! # Taking two of them
//!
//! Two locks taken at once are a deadlock waiting for the order to disagree,
//! and the answer is the one the command layer already uses: when a command
//! names keys in several stripes, the stripes are taken in stripe order, so two
//! commands that want the same pair want it the same way round. Nothing here
//! enforces that, because a lock cannot see the other locks.
//!
//! What it can see is the other half of the same mistake, which is one thread
//! taking the same lock twice. That is a hang in a release build and there is
//! nothing to see when it happens, so a debug build remembers who holds a lock
//! and panics rather than spinning forever. Tests and the fuzzers run in debug
//! builds, so the mistake is a failure with a message instead of a test that
//! never finishes.

use core::hint;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::sync::{AtomicBool, Ordering, UnsafeCell, yield_now};

/// How many times a waiter spins before it starts yielding the core instead.
///
/// Short, because the point of the spin is to cover a hold that ends in tens of
/// nanoseconds. If it did not end that fast the waiter is better off letting
/// the holder have the core back, and that is what the yield is for.
#[cfg(not(loom))]
const SPINS: u32 = 40;

/// Under the model checker there is no such thing as waiting a little, and
/// every spin is another interleaving to try, so a waiter goes straight to the
/// yield and the model stays small enough to finish.
#[cfg(loom)]
const SPINS: u32 = 0;

/// A value that one thread at a time can reach.
///
/// [`lock`](Lock::lock) waits for it and hands back a [`Held`], which derefs
/// to the value and releases the lock when it is dropped. A caller holding the
/// lock by exclusive reference skips all of that through
/// [`get_mut`](Lock::get_mut), which is how single threaded code and setup code
/// reach the value for free.
pub struct Lock<T> {
    held: AtomicBool,
    /// Who holds it, in debug builds, for the re-entrancy check. Zero is free.
    #[cfg(debug_assertions)]
    owner: core::sync::atomic::AtomicU64,
    value: UnsafeCell<T>,
}

// SAFETY: the lock is what makes the value safe to share. Only one thread can
// hold the lock at a time, and a `Held` is the only way to reach the value
// through a shared reference, so the value is never touched by two threads at
// once. It has to move between threads for that to be worth anything, which is
// why the bound is `Send` and not `Sync`.
unsafe impl<T: Send> Sync for Lock<T> {}
// SAFETY: sending the lock sends the value, which `Send` already allows.
unsafe impl<T: Send> Send for Lock<T> {}

impl<T> Lock<T> {
    /// Put `value` behind a lock that nobody holds yet.
    #[cfg(not(loom))]
    pub const fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            #[cfg(debug_assertions)]
            owner: core::sync::atomic::AtomicU64::new(0),
            value: UnsafeCell::new(value),
        }
    }

    /// The same, for the model checker, whose atomics cannot be built in a
    /// constant because each one registers itself with the running model.
    #[cfg(loom)]
    pub fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            #[cfg(debug_assertions)]
            owner: core::sync::atomic::AtomicU64::new(0),
            value: UnsafeCell::new(value),
        }
    }

    /// Wait for the lock and take it.
    ///
    /// # Panics
    ///
    /// In a debug build, if the calling thread already holds this lock. In a
    /// release build that case spins forever instead, which is the usual
    /// bargain for a check that costs something on the hot path.
    #[inline]
    pub fn lock(&self) -> Held<'_, T> {
        if !self.take() {
            self.wait();
        }
        self.claim();
        Held {
            lock: self,
            stays: PhantomData,
        }
    }

    /// Take the lock if it is free, and give up rather than wait if it is not.
    ///
    /// Returns `None` if another thread holds it, and also if the calling
    /// thread does, since a lock cannot be taken twice by anyone.
    #[inline]
    pub fn try_lock(&self) -> Option<Held<'_, T>> {
        if !self.take() {
            return None;
        }
        self.claim();
        Some(Held {
            lock: self,
            stays: PhantomData,
        })
    }

    /// Reach the value without locking, because the caller owns the lock.
    ///
    /// An exclusive reference to the lock is already proof that no other thread
    /// can be holding it, so this costs nothing at all. Setup, teardown and
    /// everything a single threaded server does can go through here.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// Take the value back out and drop the lock around it.
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }

    /// Whether somebody holds it, which is only ever a hint.
    ///
    /// True can be stale by the time the caller reads it and so can false. It
    /// is here for statistics and for tests, not for deciding anything.
    #[inline]
    pub fn is_held(&self) -> bool {
        self.held.load(Ordering::Relaxed)
    }

    /// One attempt at the flag, with no waiting and no bookkeeping.
    #[inline]
    fn take(&self) -> bool {
        self.held
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// The slow path: spin while it is held, then yield, until it is ours.
    ///
    /// The load in the middle is what keeps this from being a queue of writes
    /// to the same cache line. A waiter that keeps trying to swap the flag
    /// takes the line away from the holder over and over and makes the hold
    /// longer, so a waiter reads until the flag looks free and only then tries.
    #[cold]
    fn wait(&self) {
        self.mine_already();
        let mut spins = 0;
        loop {
            while self.held.load(Ordering::Relaxed) {
                if spins < SPINS {
                    spins += 1;
                    hint::spin_loop();
                } else {
                    yield_now();
                }
            }
            if self.take() {
                return;
            }
        }
    }

    /// Record that this thread holds it, in the builds that keep track.
    #[inline]
    fn claim(&self) {
        #[cfg(debug_assertions)]
        self.owner.store(me(), Ordering::Relaxed);
    }

    /// Forget who holds it, before the flag says anybody can.
    #[inline]
    fn disclaim(&self) {
        #[cfg(debug_assertions)]
        self.owner.store(0, Ordering::Relaxed);
    }

    /// Panic instead of spinning forever on a lock this thread already holds.
    #[inline]
    fn mine_already(&self) {
        #[cfg(debug_assertions)]
        assert!(
            self.owner.load(Ordering::Relaxed) != me(),
            "this thread already holds this lock, and waiting for itself will \
             never end"
        );
    }
}

impl<T: Default> Default for Lock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// The lock, held.
///
/// Derefs to the value, and gives it back when it is dropped. It cannot be sent
/// to another thread, because the thread that took a lock is the thread that
/// has to release it.
pub struct Held<'a, T> {
    lock: &'a Lock<T>,
    /// A guard is tied to the thread that took the lock, the same way a
    /// `MutexGuard` is, and a raw pointer in a field is how a type says it does
    /// not go to another thread.
    stays: PhantomData<*const ()>,
}

// SAFETY: `&Held<T>` gives out `&T` and nothing else, so sharing the guard is
// sharing the value. The raw pointer above took `Sync` away along with `Send`,
// and this puts back the half that was true.
unsafe impl<T: Sync> Sync for Held<'_, T> {}

impl<T> Deref for Held<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: we hold the lock, so no other thread has a reference to the
        // value, and the guard borrows the lock so it cannot go away first.
        self.lock.value.with(|p| unsafe { &*p })
    }
}

impl<T> DerefMut for Held<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as above, and the exclusive borrow of the guard is what makes
        // this the only reference to the value that exists.
        self.lock.value.with(|p| unsafe { &mut *p })
    }
}

impl<T> Drop for Held<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.disclaim();
        self.lock.held.store(false, Ordering::Release);
    }
}

/// A number that means this thread and no other, for the debug check.
///
/// A counter rather than the thread id, because the standard one cannot be had
/// as a number on stable and getting it touches an `Arc`. Numbers are never
/// reused, so a thread that has exited cannot be mistaken for a live one, and
/// zero is kept back to mean nobody. During thread teardown the slot may be
/// gone, in which case there is no answer and the check quietly does not fire.
#[cfg(debug_assertions)]
fn me() -> u64 {
    use core::cell::Cell;
    use core::sync::atomic::AtomicU64;

    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static ME: Cell<u64> = const { Cell::new(0) };
    }

    ME.try_with(|slot| {
        let mut id = slot.get();
        if id == 0 {
            id = NEXT.fetch_add(1, Ordering::Relaxed);
            slot.set(id);
        }
        id
    })
    .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many threads the contended tests run. Four is enough for a waiter
    /// to find the lock held, and small enough to be four on a laptop too.
    const HANDS: u64 = 4;

    // Scaled down under Miri, which interprets every instruction and runs the
    // threads itself. The contention still happens, only the repetition goes.
    #[cfg(miri)]
    const ROUNDS: u64 = 20;
    #[cfg(not(miri))]
    const ROUNDS: u64 = 250;

    #[cfg(miri)]
    const HOLDS: u64 = 3;
    #[cfg(not(miri))]
    const HOLDS: u64 = 20;

    #[cfg(miri)]
    const INSIDE: u64 = 50;
    #[cfg(not(miri))]
    const INSIDE: u64 = 2_000;

    #[test]
    fn what_goes_in_is_what_comes_out_the_next_time_it_is_taken() {
        let lock = Lock::new(Vec::new());
        lock.lock().push(1u8);
        lock.lock().push(2);
        assert_eq!(*lock.lock(), vec![1, 2]);
        assert_eq!(lock.into_inner(), vec![1, 2]);
    }

    #[test]
    fn a_held_lock_cannot_be_taken_and_a_dropped_one_can() {
        let lock = Lock::new(0u32);
        let held = lock.lock();
        assert!(lock.is_held());
        assert!(lock.try_lock().is_none());
        drop(held);
        assert!(!lock.is_held());
        assert!(lock.try_lock().is_some());
    }

    #[test]
    fn an_owner_with_an_exclusive_reference_pays_nothing() {
        let mut lock = Lock::new(0u32);
        *lock.get_mut() = 7;
        assert_eq!(*lock.lock(), 7);
    }

    /// The one thing a lock is for. Without it the count comes out short,
    /// because a read and a write from two threads lose one of the writes.
    #[test]
    fn every_increment_from_every_thread_lands() {
        let lock = Lock::new(0u64);
        std::thread::scope(|s| {
            for _ in 0..HANDS {
                s.spawn(|| {
                    for _ in 0..ROUNDS {
                        *lock.lock() += 1;
                    }
                });
            }
        });
        assert_eq!(lock.into_inner(), HANDS * ROUNDS);
    }

    /// Contended on purpose: the work inside the lock is long enough that the
    /// waiters get past the spin and into the yield, which is the path the
    /// short test above never reaches.
    #[test]
    fn a_waiter_that_runs_out_of_spins_still_gets_the_lock() {
        let lock = Lock::new(0u64);
        std::thread::scope(|s| {
            for _ in 0..HANDS {
                s.spawn(|| {
                    for _ in 0..HOLDS {
                        let mut held = lock.lock();
                        for _ in 0..INSIDE {
                            *held += 1;
                            hint::spin_loop();
                        }
                    }
                });
            }
        });
        assert_eq!(lock.into_inner(), HANDS * HOLDS * INSIDE);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "already holds this lock")]
    fn taking_it_twice_on_one_thread_says_so_instead_of_hanging() {
        let lock = Lock::new(0u32);
        let _first = lock.lock();
        let _second = lock.lock();
    }

    #[cfg(debug_assertions)]
    #[test]
    fn trying_it_twice_on_one_thread_just_fails() {
        let lock = Lock::new(0u32);
        let _first = lock.lock();
        assert!(lock.try_lock().is_none());
    }
}

/// The lock, model checked.
///
/// The tests above run one interleaving each, whichever one the machine
/// happened to pick that time. These run all of them: loom takes the two
/// threads apart at every atomic operation and tries every order they could
/// have gone in, which is the only way to be sure that an ordering is right
/// rather than merely never seen to be wrong on the machines it was run on.
///
/// Built and run separately, since the whole crate has to be compiled against
/// loom's atomics for it to see anything: `RUSTFLAGS="--cfg loom" cargo test -p
/// yo-common --release --lib lock::loom`.
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;

    /// What the lock is guarding, in a form loom watches.
    ///
    /// A plain counter would not do. Loom only sees the reads and writes it is
    /// told about, and the value inside a lock is reached through a pointer
    /// that leaves the closure, so an ordinary field would be invisible to it
    /// and every one of these tests would pass whatever the orderings said.
    /// This is loom's own cell, so every access through it is recorded and an
    /// access that raced with another is a failed model rather than a guess.
    type Guarded = loom::cell::UnsafeCell<usize>;

    fn bump(lock: &Lock<Guarded>) {
        let held = lock.lock();
        // SAFETY: the lock is held, so this is the only pointer to the value.
        held.with_mut(|p| unsafe { *p += 1 });
    }

    fn read(lock: &Lock<Guarded>) -> usize {
        let held = lock.lock();
        // SAFETY: as above, and this one only reads.
        held.with(|p| unsafe { *p })
    }

    /// One counter, two threads, one increment each. An increment is a read
    /// and a write, so any interleaving where both threads are inside at once
    /// either loses one of them or is a race loom reports outright.
    #[test]
    fn two_threads_cannot_both_be_inside() {
        loom::model(|| {
            let lock = loom::sync::Arc::new(Lock::new(Guarded::new(0)));
            let other = lock.clone();
            let hand = loom::thread::spawn(move || bump(&other));
            bump(&lock);
            hand.join().unwrap();
            assert_eq!(read(&lock), 2, "an increment was lost");
        });
    }

    /// What one thread wrote under the lock is what the next thread reads
    /// under it. This is what the release on the way out and the acquire on
    /// the way in are for, and without either of them loom finds an order
    /// where the second thread is looking at the value while the first is
    /// still writing it.
    #[test]
    fn what_the_last_holder_wrote_is_what_the_next_one_sees() {
        loom::model(|| {
            let lock = loom::sync::Arc::new(Lock::new(Guarded::new(0)));
            let other = lock.clone();
            let hand = loom::thread::spawn(move || bump(&other));
            let seen = read(&lock);
            assert!(seen == 0 || seen == 1, "read a value nobody wrote");
            hand.join().unwrap();
            assert_eq!(read(&lock), 1);
        });
    }

    /// A take that gives up rather than waits leaves the lock as it found it,
    /// so the thread that does hold it is not disturbed and the next take
    /// still works.
    #[test]
    fn a_take_that_fails_changes_nothing() {
        loom::model(|| {
            let lock = loom::sync::Arc::new(Lock::new(Guarded::new(0)));
            let other = lock.clone();
            let hand = loom::thread::spawn(move || {
                if let Some(held) = other.try_lock() {
                    // SAFETY: the take succeeded, so the lock is held here.
                    held.with_mut(|p| unsafe { *p += 1 });
                }
            });
            bump(&lock);
            hand.join().unwrap();
            let count = read(&lock);
            assert!(count == 1 || count == 2, "count is {count}");
        });
    }
}
