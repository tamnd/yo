//! A bounded single producer single consumer ring.
//!
//! This is the only way anything crosses a shard boundary. `05` section 1.4:
//! shards do not share data structures, they hand each other owned values, and
//! the handoff is one relaxed store plus one release store on a queue that has
//! exactly one writer and exactly one reader. No compare and swap anywhere on
//! this path.
//!
//! Head and tail live on separate cache lines, and each side caches its view of
//! the other so that the common case reads no shared line at all. A producer
//! that has room only touches the tail line it already owns.

use crate::sync::{Arc, AtomicUsize, Ordering, UnsafeCell};
use core::cell::Cell;
use core::mem::MaybeUninit;

/// Anything that must not share a cache line with anything else.
#[repr(align(64))]
struct Pad<T>(T);

struct Ring<T> {
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
    /// Next index the consumer will read.
    head: Pad<AtomicUsize>,
    /// Next index the producer will write.
    tail: Pad<AtomicUsize>,
}

// SAFETY: the ring itself never touches its contents. `Sender` and `Receiver`
// are the only things that do, there is at most one of each, and each is `Send`
// but not `Sync`, so the two ends are the two threads and there are no others.
unsafe impl<T: Send> Send for Ring<T> {}
// SAFETY: as above. The `Arc` needs `Sync` to be shared between the two ends,
// and sharing it is safe because neither end reads the other's slots.
unsafe impl<T: Send> Sync for Ring<T> {}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Relaxed);
        for i in head..tail {
            // SAFETY: every index in `head..tail` was written by the producer
            // and not yet read by the consumer, so it holds an initialised `T`.
            // We are in `drop` with `&mut self`, so both ends are gone.
            self.buf[i & self.mask].with(|p| unsafe { (*p).assume_init_drop() });
        }
    }
}

/// The producing end. One per lane, owned by one thread.
pub struct Sender<T> {
    ring: Arc<Ring<T>>,
    /// The producer's stale view of `head`. Refreshed only when the ring looks
    /// full, which is the whole point of keeping it.
    cached_head: Cell<usize>,
}

/// The consuming end. One per lane, owned by the shard thread.
pub struct Receiver<T> {
    ring: Arc<Ring<T>>,
    cached_tail: Cell<usize>,
}

// SAFETY: a `Sender` may move to another thread, and moving it is the only way
// to use it, but it is `!Sync` (the `Cell`) so it cannot be used from two
// threads at once. That is exactly the single producer rule.
unsafe impl<T: Send> Send for Sender<T> {}
// SAFETY: as above, for the single consumer.
unsafe impl<T: Send> Send for Receiver<T> {}

/// Build a lane with room for `capacity` items, rounded up to a power of two.
///
/// Bounded on purpose. An unbounded lane turns a slow shard into an out of
/// memory kill instead of into backpressure, and backpressure is information the
/// caller can act on.
pub fn lane<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "a lane needs room for at least one item");
    let cap = capacity.next_power_of_two();
    let mut buf = Vec::with_capacity(cap);
    for _ in 0..cap {
        buf.push(UnsafeCell::new(MaybeUninit::uninit()));
    }
    let ring = Arc::new(Ring {
        buf: buf.into_boxed_slice(),
        mask: cap - 1,
        head: Pad(AtomicUsize::new(0)),
        tail: Pad(AtomicUsize::new(0)),
    });
    (
        Sender {
            ring: Arc::clone(&ring),
            cached_head: Cell::new(0),
        },
        Receiver {
            ring,
            cached_tail: Cell::new(0),
        },
    )
}

impl<T> Sender<T> {
    /// How many items the lane can hold.
    pub fn capacity(&self) -> usize {
        self.ring.mask + 1
    }

    /// Push one item, handing the value back if the lane is full.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        let ring = &*self.ring;
        // Relaxed: this thread is the only writer of `tail`.
        let tail = ring.tail.0.load(Ordering::Relaxed);
        if tail.wrapping_sub(self.cached_head.get()) > ring.mask {
            // Only now is it worth touching the consumer's line.
            self.cached_head.set(ring.head.0.load(Ordering::Acquire));
            if tail.wrapping_sub(self.cached_head.get()) > ring.mask {
                return Err(value);
            }
        }
        // SAFETY: the slot at `tail` is beyond the consumer's `head` by the
        // check above, so the consumer will not read it until we publish, and
        // it holds no live value because the consumer took ownership of
        // whatever was there before `head` passed it.
        ring.buf[tail & ring.mask].with(|p| unsafe { (*p).write(value) });
        // Release: the write above must be visible before the index that
        // exposes it.
        ring.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Whether the lane currently has nothing in it.
    ///
    /// Only a hint from this side, which is all the park protocol needs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let ring = &*self.ring;
        ring.head.0.load(Ordering::Acquire) == ring.tail.0.load(Ordering::Acquire)
    }
}

impl<T> Receiver<T> {
    /// Take one item, or `None` if the lane is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let ring = &*self.ring;
        // Relaxed: this thread is the only writer of `head`.
        let head = ring.head.0.load(Ordering::Relaxed);
        if head == self.cached_tail.get() {
            self.cached_tail.set(ring.tail.0.load(Ordering::Acquire));
            if head == self.cached_tail.get() {
                return None;
            }
        }
        // SAFETY: `head` is strictly below the producer's published `tail`, so
        // this slot holds a value the producer wrote and released, and no other
        // reader exists.
        let value = ring.buf[head & ring.mask].with(|p| unsafe { (*p).assume_init_read() });
        // Release: the read above must complete before the producer is told the
        // slot is reusable.
        ring.head.0.store(head.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Whether the lane currently has nothing in it.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let ring = &*self.ring;
        ring.head.0.load(Ordering::Relaxed) == ring.tail.0.load(Ordering::Acquire)
    }

    /// How many items are waiting.
    #[inline]
    pub fn len(&self) -> usize {
        let ring = &*self.ring;
        ring.tail
            .0
            .load(Ordering::Acquire)
            .wrapping_sub(ring.head.0.load(Ordering::Relaxed))
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    // Scaled down under Miri, which interprets every instruction. The wrap, the
    // full case and the empty case all still happen; only the repetition goes.
    #[cfg(miri)]
    const WRAPS: usize = 300;
    #[cfg(not(miri))]
    const WRAPS: usize = 10_000;

    #[cfg(miri)]
    const HANDOFF: usize = 2_000;
    #[cfg(not(miri))]
    const HANDOFF: usize = 200_000;

    #[test]
    fn capacity_rounds_up_to_a_power_of_two() {
        let (tx, _rx) = lane::<u8>(5);
        assert_eq!(tx.capacity(), 8);
    }

    #[test]
    fn fills_then_refuses() {
        let (tx, rx) = lane::<u32>(4);
        for i in 0..4 {
            assert!(tx.push(i).is_ok());
        }
        assert_eq!(tx.push(99), Err(99));
        assert_eq!(rx.pop(), Some(0));
        assert!(tx.push(99).is_ok());
        assert_eq!(rx.pop(), Some(1));
        assert_eq!(rx.pop(), Some(2));
        assert_eq!(rx.pop(), Some(3));
        assert_eq!(rx.pop(), Some(99));
        assert_eq!(rx.pop(), None);
    }

    #[test]
    fn survives_wrapping_many_times() {
        let (tx, rx) = lane::<usize>(2);
        for i in 0..WRAPS {
            assert!(tx.push(i).is_ok());
            assert_eq!(rx.pop(), Some(i));
        }
        assert!(rx.is_empty());
    }

    #[test]
    fn hands_values_across_a_thread() {
        const N: usize = HANDOFF;
        let (tx, rx) = lane::<usize>(64);
        let producer = std::thread::spawn(move || {
            let mut i = 0;
            while i < N {
                if tx.push(i).is_ok() {
                    i += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
        let mut want = 0;
        while want < N {
            match rx.pop() {
                Some(v) => {
                    assert_eq!(v, want, "lane delivered out of order");
                    want += 1;
                }
                None => std::hint::spin_loop(),
            }
        }
        producer.join().unwrap();
    }

    #[test]
    fn drops_what_is_left_behind() {
        #[derive(Debug)]
        struct Counted(std::sync::Arc<AtomicU32>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let n = std::sync::Arc::new(AtomicU32::new(0));
        {
            let (tx, rx) = lane::<Counted>(8);
            for _ in 0..5 {
                tx.push(Counted(std::sync::Arc::clone(&n))).unwrap();
            }
            drop(rx.pop());
            assert_eq!(n.load(Ordering::Relaxed), 1);
        }
        assert_eq!(n.load(Ordering::Relaxed), 5, "four leaked on drop");
    }
}

/// The lane, checked by loom rather than argued about.
///
/// Run with `RUSTFLAGS="--cfg loom" cargo test -p yo-shard --release spsc::loom`.
/// Loom enumerates the interleavings a weak memory model permits, so a passing
/// run means the orderings are right, not that the test got lucky. The counts
/// are small on purpose: the state space grows fast and two or three items are
/// enough to exercise the wrap, the full case, and the empty case.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;

    #[test]
    fn handoff_is_ordered_and_lossless() {
        loom::model(|| {
            let (tx, rx) = lane::<usize>(2);
            let producer = loom::thread::spawn(move || {
                for i in 0..3usize {
                    while tx.push(i).is_err() {
                        loom::thread::yield_now();
                    }
                }
            });
            let mut want = 0usize;
            while want < 3 {
                if let Some(v) = rx.pop() {
                    assert_eq!(v, want, "lane delivered out of order");
                    want += 1;
                } else {
                    loom::thread::yield_now();
                }
            }
            producer.join().unwrap();
            assert!(rx.pop().is_none(), "lane delivered something extra");
        });
    }

    #[test]
    fn a_full_lane_never_overwrites() {
        loom::model(|| {
            let (tx, rx) = lane::<usize>(2);
            assert!(tx.push(10).is_ok());
            assert!(tx.push(11).is_ok());
            let producer = loom::thread::spawn(move || {
                // This can only land after the consumer has taken something.
                while tx.push(12).is_err() {
                    loom::thread::yield_now();
                }
            });
            assert_eq!(rx.pop(), Some(10));
            assert_eq!(rx.pop(), Some(11));
            let mut last = None;
            while last.is_none() {
                last = rx.pop();
                if last.is_none() {
                    loom::thread::yield_now();
                }
            }
            assert_eq!(last, Some(12));
            producer.join().unwrap();
        });
    }
}
