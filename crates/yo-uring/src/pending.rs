//! Where a submission's state waits for its completion.
//!
//! `04` section 7: a command that has to fault a page in does not block the
//! loop and does not spawn a task. It parks what it was doing in a slot keyed
//! by the submission's user data, and the completion drain picks it up on the
//! next turn of the loop. That is the difference between the 183 thousand
//! transactions per second rung of the io_uring ladder and the 16.5 thousand
//! one, and this is the table it parks in.
//!
//! A slab rather than a map. The key is an index this table hands out, so
//! there is no hashing on either side, and a slot that comes free goes on a
//! free list rather than leaving a hole to scan past. The generation counter is
//! the part that matters for correctness: a slot is reused as soon as it is
//! freed, so a completion for a submission that was cancelled or timed out
//! would otherwise land on whoever holds the slot now.
//!
//! Nothing here is shared across threads. The table belongs to one shard, like
//! everything else on the data path.

use yo_common::{Code, Error, Result};

use crate::token::{Kind, MAX_SLOT, Token};

/// One entry, occupied or not.
struct Slot<T> {
    /// Bumped every time the slot is filled. An occupied slot's generation is
    /// what the token carries, so a token whose generation does not match is a
    /// completion for a previous occupant.
    generation: u32,
    /// The parked state, or the next free slot when this one is empty.
    state: Option<T>,
    /// Valid only while `state` is `None`. `u32::MAX` ends the list.
    next_free: u32,
}

/// The parked state of every submission this shard has in flight.
pub struct Pending<T> {
    slots: Vec<Slot<T>>,
    free: u32,
    live: u32,
    cap: u32,
}

const END: u32 = u32::MAX;

impl<T> Pending<T> {
    /// An empty table that will not grow past `capacity` slots.
    ///
    /// The cap exists because the ring has one too. A table that can grow past
    /// the ring's depth is a table that will, and the extra entries are all
    /// submissions the kernel never accepted.
    ///
    /// # Panics
    ///
    /// If `capacity` is zero or above [`MAX_SLOT`], neither of which a caller
    /// can reach by accident: the ring's own construction refuses both first.
    #[must_use]
    pub fn with_capacity(capacity: u32) -> Pending<T> {
        assert!(capacity > 0, "a pending table with no room in it");
        assert!(
            capacity <= MAX_SLOT + 1,
            "a pending table larger than the tag can address"
        );
        Pending {
            slots: Vec::new(),
            free: END,
            live: 0,
            cap: capacity,
        }
    }

    /// How many submissions are parked.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.live
    }

    /// Whether anything is parked.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// How many more will fit.
    #[must_use]
    pub const fn room(&self) -> u32 {
        self.cap - self.live
    }

    /// Parks `state` and returns the tag to submit with.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] when the table is at capacity. That is backpressure and
    /// not a failure: the caller stops submitting, drains, and comes back. It
    /// is the same answer the ring gives when its submission queue is full, so
    /// a caller only has one case to handle.
    pub fn park(&mut self, kind: Kind, state: T) -> Result<Token> {
        if self.live >= self.cap {
            return Err(Error::new(Code::Full, "no room to park another submission")
                .with_detail(format!("live={} capacity={}", self.live, self.cap)));
        }
        let slot = if self.free == END {
            let slot = u32::try_from(self.slots.len()).expect("bounded by the capacity check");
            self.slots.push(Slot {
                generation: 0,
                state: None,
                next_free: END,
            });
            slot
        } else {
            let slot = self.free;
            self.free = self.slots[slot as usize].next_free;
            slot
        };
        let s = &mut self.slots[slot as usize];
        // Wrapping is fine and is why this is a generation and not a counter of
        // anything. Four billion reuses of one slot between a submission and
        // its completion is not a thing that happens, and if it did the answer
        // would be the same as no generation at all rather than worse.
        s.generation = s.generation.wrapping_add(1);
        s.state = Some(state);
        self.live += 1;
        Ok(Token::new(kind, slot, s.generation))
    }

    /// Takes back the state a completion belongs to.
    ///
    /// `None` means the completion is stale: the slot was freed and refilled
    /// since it was submitted, or the tag came back naming a slot that was
    /// never handed out. Both are dropped rather than applied, which is the
    /// whole reason the generation is in the tag.
    pub fn take(&mut self, token: Token) -> Option<T> {
        let slot = token.slot() as usize;
        let s = self.slots.get_mut(slot)?;
        if s.generation != token.generation() || s.state.is_none() {
            return None;
        }
        let state = s.state.take();
        s.next_free = self.free;
        self.free = slot as u32;
        self.live -= 1;
        state
    }

    /// Whether a completion with this tag would find its state.
    #[must_use]
    pub fn holds(&self, token: Token) -> bool {
        self.slots
            .get(token.slot() as usize)
            .is_some_and(|s| s.generation == token.generation() && s.state.is_some())
    }

    /// Empties the table, handing back everything still parked.
    ///
    /// What a shard does on the way down: every submission that will never
    /// complete has state attached, and dropping it silently is how a caller
    /// ends up waiting forever on a reply that was never going to come.
    pub fn drain(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.live as usize);
        for s in &mut self.slots {
            if let Some(state) = s.state.take() {
                out.push(state);
            }
            s.next_free = END;
        }
        self.free = END;
        self.live = 0;
        // Every slot is free now, so the free list is rebuilt from the front
        // rather than left as whatever order the drain happened to produce.
        for (i, s) in self.slots.iter_mut().enumerate().rev() {
            s.next_free = self.free;
            self.free = i as u32;
        }
        out
    }
}

impl<T> core::fmt::Debug for Pending<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pending")
            .field("live", &self.live)
            .field("capacity", &self.cap)
            .field("slots", &self.slots.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_goes_in_comes_back_out() {
        let mut p: Pending<u32> = Pending::with_capacity(8);
        let a = p.park(Kind::Write, 10).unwrap();
        let b = p.park(Kind::Read, 20).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(a.kind(), Kind::Write);
        assert_eq!(b.kind(), Kind::Read);
        assert_eq!(p.take(b), Some(20));
        assert_eq!(p.take(a), Some(10));
        assert!(p.is_empty());
    }

    #[test]
    fn a_freed_slot_is_used_again_rather_than_left_as_a_hole() {
        let mut p: Pending<u32> = Pending::with_capacity(4);
        let a = p.park(Kind::Write, 1).unwrap();
        p.take(a).unwrap();
        let b = p.park(Kind::Write, 2).unwrap();
        assert_eq!(
            a.slot(),
            b.slot(),
            "the free list did not give the slot back"
        );
        assert_ne!(
            a.generation(),
            b.generation(),
            "the generation did not move"
        );
    }

    /// The bug the generation exists to stop. A cancelled submission's
    /// completion arrives after the slot has been reused, and without the
    /// generation it would hand the new occupant's state to the old caller.
    #[test]
    fn a_completion_for_a_slot_that_moved_on_is_dropped() {
        let mut p: Pending<&str> = Pending::with_capacity(4);
        let stale = p.park(Kind::Recv, "the connection that went away").unwrap();
        p.take(stale).unwrap();
        let live = p.park(Kind::Recv, "somebody else entirely").unwrap();
        assert_eq!(stale.slot(), live.slot());

        assert_eq!(p.take(stale), None, "a stale tag found somebody's state");
        assert!(p.holds(live), "and it took the live one down with it");
        assert_eq!(p.take(live), Some("somebody else entirely"));
    }

    #[test]
    fn a_tag_naming_a_slot_that_was_never_handed_out_finds_nothing() {
        let mut p: Pending<u32> = Pending::with_capacity(4);
        assert_eq!(p.take(Token::new(Kind::Write, 3, 1)), None);
        assert_eq!(p.take(Token::new(Kind::Write, MAX_SLOT, 0)), None);
        let a = p.park(Kind::Write, 7).unwrap();
        // Right slot, wrong generation.
        let wrong = Token::new(Kind::Write, a.slot(), a.generation().wrapping_add(1));
        assert_eq!(p.take(wrong), None);
        assert_eq!(p.take(a), Some(7));
    }

    #[test]
    fn a_full_table_says_full_rather_than_growing() {
        let mut p: Pending<u32> = Pending::with_capacity(2);
        let a = p.park(Kind::Write, 1).unwrap();
        p.park(Kind::Write, 2).unwrap();
        assert_eq!(p.room(), 0);
        let e = p.park(Kind::Write, 3).unwrap_err();
        assert_eq!(e.code(), Code::Full);
        // And it works again the moment something completes, which is what
        // makes this backpressure rather than a wall.
        p.take(a).unwrap();
        assert!(p.park(Kind::Write, 4).is_ok());
    }

    #[test]
    fn shutting_down_hands_back_everything_still_parked() {
        let mut p: Pending<u32> = Pending::with_capacity(8);
        let a = p.park(Kind::Write, 1).unwrap();
        p.park(Kind::Read, 2).unwrap();
        p.park(Kind::Fsync, 3).unwrap();
        p.take(a).unwrap();

        let mut left = p.drain();
        left.sort_unstable();
        assert_eq!(left, vec![2, 3]);
        assert!(p.is_empty());
        assert_eq!(p.room(), 8);
        // And the table still works after a drain, because a shard that stops
        // and restarts is a test, not a special case.
        assert!(p.park(Kind::Write, 9).is_ok());
    }

    /// The slab arithmetic under a long mixed run, which is where an off by one
    /// in the free list would show up as a leaked slot rather than as a crash.
    #[test]
    fn the_free_list_does_not_leak_over_a_long_run() {
        let mut p: Pending<u64> = Pending::with_capacity(64);
        let mut held: Vec<(Token, u64)> = Vec::new();
        let mut next = 0u64;
        for step in 0..2000u64 {
            // A saw tooth: fill for a while, drain for a while, so the free
            // list is walked both ways rather than only ever pushed.
            let filling = (step / 37) % 2 == 0;
            if filling && p.room() > 0 {
                let t = p.park(Kind::Write, next).unwrap();
                held.push((t, next));
                next += 1;
            } else if let Some((t, want)) = held.pop() {
                assert_eq!(p.take(t), Some(want), "at step {step}");
            }
            assert_eq!(p.len() as usize, held.len(), "at step {step}");
        }
        for (t, want) in held {
            assert_eq!(p.take(t), Some(want));
        }
        assert!(p.is_empty());
    }
}
