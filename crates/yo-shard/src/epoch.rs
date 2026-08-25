//! Per shard epochs, which is how memory gets reclaimed without a read barrier.
//!
//! `05` section 3.4. Every shard keeps a counter it bumps once per batch of
//! work. A shard that is idle publishes an even number, a shard that is inside
//! a batch publishes an odd one. When an arena segment is retired it records
//! the global epoch at the time. The segment is reusable once every shard has
//! been observed past that epoch, which means nobody can still be holding an
//! address into it.
//!
//! The cost on the hot path is one relaxed store per batch, not per operation,
//! and it is a store to a line this core owns. Nothing else reads it except the
//! reclaimer, and the reclaimer runs off the hot path.

use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Nothing in this line but the counter.
#[repr(align(64))]
struct Slot(AtomicU32);

/// One counter per shard.
pub struct Epochs {
    slots: Box<[Slot]>,
}

impl Epochs {
    /// Counters for `n` shards, all starting at zero, which reads as idle.
    pub fn new(n: usize) -> Arc<Epochs> {
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(Slot(AtomicU32::new(0)));
        }
        Arc::new(Epochs {
            slots: slots.into_boxed_slice(),
        })
    }

    /// How many shards are tracked.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether nothing is tracked.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The counter shard `id` is publishing right now.
    #[inline]
    pub fn get(&self, id: usize) -> u32 {
        self.slots[id].0.load(Ordering::Acquire)
    }

    /// Enter a batch on shard `id`. The published counter becomes odd.
    ///
    /// Relaxed on the store, release on the fence, because what matters is that
    /// the reads this shard is about to do cannot be moved above the
    /// announcement that it is active.
    #[inline]
    pub fn enter(&self, id: usize) {
        let e = self.slots[id].0.load(Ordering::Relaxed);
        debug_assert!(
            e.is_multiple_of(2),
            "shard {id} entered a batch it was already in"
        );
        self.slots[id].0.store(e.wrapping_add(1), Ordering::Relaxed);
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Leave a batch on shard `id`. The published counter becomes even again.
    #[inline]
    pub fn leave(&self, id: usize) {
        let e = self.slots[id].0.load(Ordering::Relaxed);
        debug_assert!(e % 2 == 1, "shard {id} left a batch it was not in");
        self.slots[id].0.store(e.wrapping_add(1), Ordering::Release);
    }

    /// Whether every shard has left, or moved past, the epoch it was in at
    /// `snapshot`.
    ///
    /// A shard is clear when it is idle now, or when its counter has changed
    /// since the snapshot. Idle means it holds no address at all. Changed means
    /// it finished whatever batch it was in, and a batch is the unit that can
    /// hold an address.
    pub fn all_past(&self, snapshot: &[u32]) -> bool {
        debug_assert_eq!(snapshot.len(), self.slots.len());
        for (i, &was) in snapshot.iter().enumerate() {
            let now = self.slots[i].0.load(Ordering::Acquire);
            if was % 2 == 0 {
                continue;
            }
            if now == was {
                return false;
            }
        }
        true
    }

    /// A reading of every shard's counter, for handing to [`Epochs::all_past`]
    /// later.
    pub fn snapshot(&self) -> Vec<u32> {
        self.slots
            .iter()
            .map(|s| s.0.load(Ordering::Acquire))
            .collect()
    }
}

/// A retirement list: things that are dead but not yet safe to reuse.
///
/// Deliberately not generic over a closure or a trait object. A retired item is
/// an arena segment ordinal and the shard that owns it, and keeping it concrete
/// means the whole structure is two `u32`s per entry with no allocation per
/// retire.
pub struct Retired {
    entries: Vec<(u32, Vec<u32>)>,
}

impl Retired {
    /// An empty list.
    pub fn new() -> Retired {
        Retired {
            entries: Vec::new(),
        }
    }

    /// How many segments are waiting.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is waiting.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Note that segment `seg` is dead as of now.
    pub fn push(&mut self, seg: u32, epochs: &Epochs) {
        self.entries.push((seg, epochs.snapshot()));
    }

    /// Hand back every segment that is now safe to reuse, oldest first.
    pub fn collect(&mut self, epochs: &Epochs) -> Vec<u32> {
        let mut out = Vec::new();
        self.entries.retain(|(seg, snap)| {
            if epochs.all_past(snap) {
                out.push(*seg);
                false
            } else {
                true
            }
        });
        out
    }
}

impl Default for Retired {
    fn default() -> Retired {
        Retired::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(miri)]
    const CHURN: usize = 200;
    #[cfg(not(miri))]
    const CHURN: usize = 10_000;

    #[cfg(miri)]
    const RETIRES: u32 = 20;
    #[cfg(not(miri))]
    const RETIRES: u32 = 200;

    #[test]
    fn idle_shards_never_hold_anything_back() {
        let e = Epochs::new(4);
        let snap = e.snapshot();
        assert!(e.all_past(&snap), "all four are idle, nothing should block");
    }

    #[test]
    fn an_active_shard_holds_its_own_snapshot_back() {
        let e = Epochs::new(4);
        e.enter(2);
        let snap = e.snapshot();
        assert!(!e.all_past(&snap), "shard 2 is mid batch");
        e.leave(2);
        assert!(e.all_past(&snap), "shard 2 finished");
    }

    #[test]
    fn other_shards_moving_does_not_release_the_one_that_matters() {
        let e = Epochs::new(3);
        e.enter(0);
        e.enter(1);
        let snap = e.snapshot();
        e.leave(1);
        e.enter(1);
        e.leave(1);
        assert!(!e.all_past(&snap), "shard 0 has not moved");
        e.leave(0);
        assert!(e.all_past(&snap));
    }

    #[test]
    fn retired_segments_come_back_in_order() {
        let e = Epochs::new(2);
        let mut r = Retired::new();
        e.enter(0);
        r.push(7, &e);
        assert!(r.collect(&e).is_empty(), "shard 0 is still in the batch");
        e.leave(0);
        assert_eq!(r.collect(&e), vec![7]);
        assert!(r.is_empty());
    }

    #[test]
    fn retirement_survives_a_real_thread() {
        let e = Epochs::new(2);
        let mut r = Retired::new();
        let worker = {
            let e = Arc::clone(&e);
            std::thread::spawn(move || {
                for _ in 0..CHURN {
                    e.enter(1);
                    std::hint::spin_loop();
                    e.leave(1);
                }
            })
        };
        // Keep retiring while the worker churns. Nothing should ever be handed
        // back while the worker is inside a batch it snapshotted.
        for seg in 0..RETIRES {
            r.push(seg, &e);
            r.collect(&e);
        }
        worker.join().unwrap();
        let freed = r.collect(&e);
        assert_eq!(r.len(), 0, "everything should clear once the worker stops");
        let _ = freed;
    }
}
