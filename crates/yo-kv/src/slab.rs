//! Somewhere to keep a value that is not bytes, addressed by a small number.
//!
//! A string lives in the record: the map hands back the bytes and the bytes are
//! the value, which is what makes `GET` one lookup and one cache miss. A set
//! cannot do that. It owns three allocations that grow and shrink as members
//! come and go, and rewriting the record on every `SADD` to keep them inline
//! would be a copy of the whole set per member added.
//!
//! So the record holds four bytes saying where, and this is what those four
//! bytes point into:
//!
//! ```text
//!   record                     slab
//! +------+-------+         +---+---+---+---+---+
//! | meta | u32 3 | ------> | 0 | 1 | 2 | 3 | 4 |
//! +------+-------+         +---+---+---+---+---+
//!    kind says              free    ^   set
//!    which slab                     |
//!                          the number in the record
//! ```
//!
//! That is a second cache miss on the way to a set, and there is no arrangement
//! that avoids it, because the set is bigger than a record and lives longer than
//! any one command. What it does avoid is a second *hash* and a second *lookup*:
//! the number comes out of the record the key lookup already fetched, so the
//! second miss is a dependent load and not another trip through the index.
//!
//! # One slab per type, not one slab of an enum
//!
//! There will be a slab of sets, then one of hashes, then lists and sorted sets.
//! The alternative is one slab holding an enum over all of them, which would be
//! one field instead of four, and it would make every slot as big as the largest
//! type and put a discriminant check in front of every access. The type tag in
//! the meta byte already says which type a key holds, so that check would be the
//! second time the same question got asked. Four fields it is.
//!
//! # Reuse, and what happens if a number outlives its value
//!
//! Freed slots go on a free list threaded through the vacancies, so an insert
//! after a delete costs the same as the first insert and the vector does not
//! grow forever under a churning workload.
//!
//! That means a number can be handed out twice, and a stale number would read
//! somebody else's value rather than nothing. There is no generation counter to
//! catch it, and the reason is that the only way to hold a stale number is for a
//! record to outlive the [`Slab::remove`] that freed it, which is one function
//! putting the two in the wrong order. A generation would turn that bug into a
//! `None` at the cost of eight bytes a record for every key of every type, which
//! is paying forever for a mistake that is caught the first time it is made.
//! What the slab does promise is that [`Slab::remove`] on a slot that is already
//! free answers `None` and leaves the free list alone, so a double free is inert
//! rather than a loop in the list.

use std::mem;

/// The number that means no slot, which is the end of the free list.
const NONE: u32 = u32::MAX;

/// The most slots there can be.
///
/// One short of the whole u32 range, because the top value is spoken for as the
/// end of the free list.
pub const MAX_SLOTS: usize = NONE as usize;

/// A slot is a value or a step along the free list.
#[derive(Debug)]
enum Slot<T> {
    Filled(T),
    /// The next free slot, or [`NONE`].
    Free(u32),
}

/// Values addressed by a small stable number.
#[derive(Debug)]
pub struct Slab<T> {
    slots: Vec<Slot<T>>,
    /// The first free slot, or [`NONE`] when every slot is filled.
    free: u32,
    /// How many slots are filled, which is not `slots.len()`.
    len: usize,
}

impl<T> Slab<T> {
    /// An empty slab, which has not allocated anything.
    #[must_use]
    pub fn new() -> Slab<T> {
        Slab {
            slots: Vec::new(),
            free: NONE,
            len: 0,
        }
    }

    /// An empty slab with room for `n` values before it grows.
    #[must_use]
    pub fn with_capacity(n: usize) -> Slab<T> {
        Slab {
            slots: Vec::with_capacity(n),
            free: NONE,
            len: 0,
        }
    }

    /// How many values are in it.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Put `value` in and answer where it went.
    ///
    /// A free slot if there is one, and the end otherwise. Nothing moves, so
    /// every number handed out before stays good.
    ///
    /// # Panics
    ///
    /// If there are already [`MAX_SLOTS`] slots. That is four billion values of
    /// one type in one database, which is not a number a caller can reach by
    /// accident, and the alternative is a `Result` on the hot path of every
    /// `SADD` against a key that does not exist yet.
    pub fn insert(&mut self, value: T) -> u32 {
        if self.free != NONE {
            let at = self.free as usize;
            let Slot::Free(next) = self.slots[at] else {
                unreachable!("the free list only ever points at free slots");
            };
            self.free = next;
            self.slots[at] = Slot::Filled(value);
            self.len += 1;
            return at as u32;
        }
        assert!(self.slots.len() < MAX_SLOTS, "slab is full");
        let at = self.slots.len() as u32;
        self.slots.push(Slot::Filled(value));
        self.len += 1;
        at
    }

    /// The value at `at`, or `None` if that slot is free or does not exist.
    #[inline]
    pub fn get(&self, at: u32) -> Option<&T> {
        match self.slots.get(at as usize) {
            Some(Slot::Filled(v)) => Some(v),
            _ => None,
        }
    }

    /// The value at `at`, to be changed in place.
    #[inline]
    pub fn get_mut(&mut self, at: u32) -> Option<&mut T> {
        match self.slots.get_mut(at as usize) {
            Some(Slot::Filled(v)) => Some(v),
            _ => None,
        }
    }

    /// Take the value at `at` out and free the slot.
    ///
    /// Answers `None` if the slot was already free, without touching the free
    /// list, so freeing twice is inert rather than a loop.
    pub fn remove(&mut self, at: u32) -> Option<T> {
        match self.slots.get_mut(at as usize) {
            Some(slot @ Slot::Filled(_)) => {
                let taken = mem::replace(slot, Slot::Free(self.free));
                self.free = at;
                self.len -= 1;
                match taken {
                    Slot::Filled(v) => Some(v),
                    Slot::Free(_) => unreachable!("just matched on filled"),
                }
            }
            _ => None,
        }
    }

    /// Every value in it, in no order a caller should lean on.
    ///
    /// This is for counting bytes and for saving, both of which want all of
    /// them and neither of which cares which came first.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|s| match s {
            Slot::Filled(v) => Some(v),
            Slot::Free(_) => None,
        })
    }

    /// Drop everything and hand the memory back.
    ///
    /// This is `FLUSHALL`, where keeping a vector of four million free slots
    /// around for a database the client just emptied would be the wrong answer.
    pub fn clear(&mut self) {
        self.slots = Vec::new();
        self.free = NONE;
        self.len = 0;
    }

    /// What the slots themselves cost, not counting what the values point at.
    ///
    /// The caller adds the values, because only the caller knows how to ask
    /// them. A slab of sets sums `Set::memory_bytes` over [`Slab::iter`].
    pub fn memory_bytes(&self) -> usize {
        self.slots.capacity() * mem::size_of::<Slot<T>>()
    }
}

impl<T> Default for Slab<T> {
    fn default() -> Slab<T> {
        Slab::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_slab_holds_nothing_and_has_allocated_nothing() {
        let s: Slab<String> = Slab::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.get(0), None);
        assert_eq!(s.memory_bytes(), 0);
    }

    #[test]
    fn what_goes_in_comes_back_out_at_the_number_it_was_given() {
        let mut s = Slab::new();
        let a = s.insert("a".to_string());
        let b = s.insert("b".to_string());
        let c = s.insert("c".to_string());
        assert_eq!((a, b, c), (0, 1, 2), "the first three go at the end");
        assert_eq!(s.get(a).map(String::as_str), Some("a"));
        assert_eq!(s.get(b).map(String::as_str), Some("b"));
        assert_eq!(s.get(c).map(String::as_str), Some("c"));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn a_value_can_be_changed_where_it_lies() {
        let mut s = Slab::new();
        let a = s.insert(vec![1u8]);
        s.get_mut(a).expect("filled").push(2);
        assert_eq!(s.get(a), Some(&vec![1, 2]));
        assert_eq!(s.get_mut(9), None, "past the end");
    }

    #[test]
    fn removing_hands_the_value_back_and_the_others_keep_their_numbers() {
        let mut s = Slab::new();
        let a = s.insert("a".to_string());
        let b = s.insert("b".to_string());
        let c = s.insert("c".to_string());

        assert_eq!(s.remove(b), Some("b".to_string()));
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(b), None);
        assert_eq!(
            s.get(a).map(String::as_str),
            Some("a"),
            "a did not move when b left"
        );
        assert_eq!(s.get(c).map(String::as_str), Some("c"));
    }

    #[test]
    fn a_freed_slot_is_the_next_one_used() {
        let mut s = Slab::new();
        s.insert(0);
        let b = s.insert(1);
        s.insert(2);

        s.remove(b);
        let next = s.insert(9);
        assert_eq!(next, b, "the hole was filled rather than the vector grown");
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(b), Some(&9));
    }

    #[test]
    fn the_free_list_gives_the_holes_back_in_reverse() {
        // Not a promise to callers, but it is the shape a list threaded head
        // first has, and a test that walks it is how a broken link shows up.
        let mut s = Slab::new();
        let n: Vec<u32> = (0..5).map(|i| s.insert(i)).collect();
        for i in [1, 3, 4] {
            s.remove(n[i]);
        }
        assert_eq!(s.len(), 2);

        assert_eq!(s.insert(50), 4);
        assert_eq!(s.insert(51), 3);
        assert_eq!(s.insert(52), 1);
        assert_eq!(s.len(), 5);

        // And once the holes run out it grows again.
        assert_eq!(s.insert(53), 5);
        assert_eq!(s.get(0), Some(&0), "the untouched ones are untouched");
        assert_eq!(s.get(2), Some(&2));
    }

    #[test]
    fn freeing_twice_is_inert_rather_than_a_loop_in_the_list() {
        // The failure this guards against is not the second remove. It is the
        // insert after it: a free list with a slot on it twice hands the same
        // number to two live values, and the second one silently overwrites the
        // first. So the test is that the numbers after a double free are still
        // all different.
        let mut s = Slab::new();
        let a = s.insert("a".to_string());
        let b = s.insert("b".to_string());

        assert_eq!(s.remove(a), Some("a".to_string()));
        assert_eq!(s.remove(a), None, "already free");
        assert_eq!(s.remove(a), None, "still already free");
        assert_eq!(s.len(), 1);

        let x = s.insert("x".to_string());
        let y = s.insert("y".to_string());
        let z = s.insert("z".to_string());
        assert_eq!(x, a, "the one real hole came back");
        assert_ne!(y, x);
        assert_ne!(z, x);
        assert_ne!(z, y);
        assert_eq!(s.len(), 4);
        assert_eq!(
            s.get(b).map(String::as_str),
            Some("b"),
            "b was never touched"
        );
    }

    #[test]
    fn removing_something_that_was_never_there_answers_nothing() {
        let mut s: Slab<u8> = Slab::new();
        assert_eq!(s.remove(0), None);
        assert_eq!(s.remove(7), None);
        assert_eq!(s.len(), 0);
        assert_eq!(s.insert(1), 0, "and it did not corrupt the free list");
    }

    #[test]
    fn iterating_sees_the_values_and_not_the_holes() {
        let mut s = Slab::new();
        let n: Vec<u32> = (0..6).map(|i| s.insert(i * 10)).collect();
        s.remove(n[0]);
        s.remove(n[3]);
        s.remove(n[5]);

        let mut got: Vec<i32> = s.iter().copied().collect();
        got.sort_unstable();
        assert_eq!(got, [10, 20, 40]);
        assert_eq!(got.len(), s.len());
    }

    #[test]
    fn clearing_hands_the_memory_back_and_starts_the_numbers_again() {
        let mut s = Slab::with_capacity(64);
        for i in 0..64 {
            s.insert(i);
        }
        assert!(s.memory_bytes() >= 64 * mem::size_of::<Slot<i32>>());

        s.clear();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.memory_bytes(), 0, "the vector went, not just the values");
        assert_eq!(s.get(0), None);
        assert_eq!(s.insert(1), 0, "numbering starts over");
    }

    #[test]
    fn a_churning_workload_does_not_grow_the_vector() {
        // The reason the free list exists. Ten thousand keys created and deleted
        // one after another is a shape a real server sees, and without reuse it
        // would leave ten thousand dead slots behind.
        let mut s = Slab::with_capacity(4);
        let before = s.memory_bytes();
        for i in 0..10_000 {
            let at = s.insert(i);
            assert_eq!(at, 0, "the same slot every time");
            assert_eq!(s.remove(at), Some(i));
        }
        assert_eq!(s.len(), 0);
        assert_eq!(s.memory_bytes(), before);
    }
}
