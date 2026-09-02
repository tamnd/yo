//! A set of the records a caller has marked, so a sample can draw from those
//! and only those.
//!
//! There is one thing this is for. A key with a deadline is rare in most
//! databases and common in some, and both the active expire cycle and the
//! `volatile-*` eviction policies need to find one without walking past the
//! keys that have no deadline. Redis solves it with a second dictionary,
//! `db->expires`, and this is the same idea with less in it: the main index
//! already knows where every record is, so the second structure only has to
//! hold the addresses and does not need the keys again.
//!
//! It is a set of addresses and not a set of keys because an address is eight
//! bytes whatever the key is, and because the map that owns this is the only
//! thing that ever moves a record, so it is also the only thing that has to fix
//! this up. That is three places: an overwrite that reallocates, a delete, and
//! compaction moving a record between segments. All three are inside
//! [`RawMap`](crate::RawMap), which is why this is not public.
//!
//! # Why linear probing and no tombstones
//!
//! A tagged set churns. A cache that writes ten million keys with a deadline
//! and lets them expire inserts and removes ten million times, and a set that
//! left a tombstone behind for each one would degrade into a scan and then need
//! a rebuild to get out of it. Backward shift deletion, which is Knuth's
//! algorithm R, keeps the invariant that a probe stops at the first empty slot
//! without leaving anything behind, so a set that has been emptied and refilled
//! a thousand times probes exactly as well as a fresh one.
//!
//! The cost is that a delete moves the entries behind it, and at three quarters
//! load that is a handful of slots in the same cache line. That is the right
//! trade here because the alternative is paying on every read forever to save
//! on the deletes.

use yo_common::Addr;

/// The smallest table, in slots.
///
/// Sixty four addresses is half a kilobyte, which is what a database with a
/// handful of deadlines costs. Small enough that a database with none is not
/// worth thinking about, and large enough that the first few hundred `EXPIRE`
/// calls do not grow it repeatedly.
const MIN_SLOTS: usize = 64;

/// The load the table grows at, as a numerator over [`LOAD_DEN`].
const LOAD_NUM: usize = 3;

/// The denominator of [`LOAD_NUM`].
const LOAD_DEN: usize = 4;

/// The multiplier that spreads addresses over slots.
///
/// Addresses are not random. They are segment base plus an offset that steps by
/// the size of the record in front, so consecutive allocations land a few tens
/// of bytes apart and the low bits carry almost nothing: the arena aligns every
/// record, so the bottom bits are always zero. A multiply and a shift from the
/// top spreads the bits that do vary across the whole slot range, which the raw
/// address masked would not.
const MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// The addresses a caller has marked.
///
/// [`Addr::NONE`] is an empty slot, which costs nothing to arrange: it is zero,
/// and no record is ever at address zero because the arena's own header is
/// there.
#[derive(Debug)]
pub(crate) struct Tagged {
    slots: Vec<Addr>,
    len: usize,
}

impl Tagged {
    /// An empty set that has not allocated anything.
    ///
    /// Nothing until the first tag, because most databases never tag anything
    /// and a table nobody uses is still a table somebody paid for.
    pub(crate) fn new() -> Tagged {
        Tagged {
            slots: Vec::new(),
            len: 0,
        }
    }

    /// How many addresses are tagged.
    #[inline]
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Bytes this holds, for the memory report.
    #[inline]
    pub(crate) fn memory_bytes(&self) -> usize {
        self.slots.capacity() * size_of::<Addr>()
    }

    /// Where `addr` wants to sit.
    #[inline]
    fn home(&self, addr: Addr) -> usize {
        let mask = self.slots.len() - 1;
        ((addr.to_bits().wrapping_mul(MIX) >> 32) as usize) & mask
    }

    /// The slot `addr` is in, if it is in one.
    #[inline]
    fn find(&self, addr: Addr) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut at = self.home(addr);
        loop {
            let s = self.slots[at];
            if s == addr {
                return Some(at);
            }
            if s == Addr::NONE {
                return None;
            }
            at = (at + 1) & mask;
        }
    }

    /// Whether `addr` is tagged.
    #[inline]
    pub(crate) fn contains(&self, addr: Addr) -> bool {
        self.find(addr).is_some()
    }

    /// Tag `addr`, and say whether that was a change.
    ///
    /// Tagging something already tagged is not an error and not a second entry.
    /// The caller is a write path that knows what the new record wants to be and
    /// not necessarily what the old one was, so idempotence here is what keeps
    /// the caller from having to ask first.
    pub(crate) fn insert(&mut self, addr: Addr) -> bool {
        debug_assert_ne!(addr, Addr::NONE, "the empty slot is not an address");
        if (self.len + 1) * LOAD_DEN > self.slots.len() * LOAD_NUM {
            self.resize(self.slots.len().max(MIN_SLOTS / 2) * 2);
        }
        let mask = self.slots.len() - 1;
        let mut at = self.home(addr);
        loop {
            let s = self.slots[at];
            if s == addr {
                return false;
            }
            if s == Addr::NONE {
                self.slots[at] = addr;
                self.len += 1;
                return true;
            }
            at = (at + 1) & mask;
        }
    }

    /// Untag `addr`, and say whether it was tagged.
    pub(crate) fn remove(&mut self, addr: Addr) -> bool {
        let Some(at) = self.find(addr) else {
            return false;
        };
        self.len -= 1;
        self.shift_back(at);
        // Only well below the growth load, so a set sitting near a power of two
        // boundary does not grow and shrink on alternate writes. A quarter
        // against three quarters is a factor of three between the two, which is
        // a whole doubling of slack.
        if self.slots.len() > MIN_SLOTS && self.len * LOAD_DEN < self.slots.len() {
            self.resize(self.slots.len() / 2);
        }
        true
    }

    /// Close the hole at `at` by pulling later entries back into it.
    ///
    /// Knuth's algorithm R. An entry can move back into the hole when its home
    /// is not inside the stretch between the hole and where it is now, because
    /// moving it back past its own home is what would make a probe miss it.
    fn shift_back(&mut self, at: usize) {
        let mask = self.slots.len() - 1;
        let mut hole = at;
        loop {
            self.slots[hole] = Addr::NONE;
            let mut j = hole;
            loop {
                j = (j + 1) & mask;
                let s = self.slots[j];
                if s == Addr::NONE {
                    return;
                }
                if !between(hole, self.home(s), j) {
                    break;
                }
            }
            self.slots[hole] = self.slots[j];
            hole = j;
        }
    }

    /// Grow or shrink to `slots`, which must be a power of two.
    ///
    /// `yo_alloc::for_the_data` and not a fix. A database that has taken its ten
    /// millionth key with a deadline has to have doubled this table along the
    /// way, and there is no arrangement of it that holds ten million addresses
    /// in the room it had for sixty four. The shrink is the same claim from the
    /// other side, and the reason it is not a per command cost is the hysteresis
    /// above it: growth is at three quarters and shrink is at one quarter, so a
    /// set sitting on a boundary does not resize on alternate writes.
    fn resize(&mut self, slots: usize) {
        debug_assert!(slots.is_power_of_two());
        debug_assert!(self.len * LOAD_DEN <= slots * LOAD_NUM);
        let fresh = yo_alloc::for_the_data(|| vec![Addr::NONE; slots]);
        let old = core::mem::replace(&mut self.slots, fresh);
        let mask = slots - 1;
        for a in old {
            if a == Addr::NONE {
                continue;
            }
            let mut at = self.home(a);
            while self.slots[at] != Addr::NONE {
                at = (at + 1) & mask;
            }
            self.slots[at] = a;
        }
    }

    /// Walk tagged addresses starting from wherever `r` lands, until `out` says
    /// stop or every one has been offered.
    ///
    /// A run of consecutive slots and not a fresh draw per entry, because the
    /// slot an address sits in is its hash and not its key, so a run here is
    /// already unrelated to anything a client can control. One random start and
    /// then a walk costs one multiply for the whole sample, where a draw per
    /// entry would cost a random access per entry into a table that is mostly
    /// cache resident anyway.
    pub(crate) fn sample(&self, r: u64, mut out: impl FnMut(Addr) -> bool) {
        if self.len == 0 {
            return;
        }
        let mask = self.slots.len() - 1;
        let mut at = (r as usize) & mask;
        for _ in 0..self.slots.len() {
            let s = self.slots[at];
            if s != Addr::NONE && !out(s) {
                return;
            }
            at = (at + 1) & mask;
        }
    }
}

/// Whether `x` lies in the stretch of slots after `lo` and up to `hi`, going
/// forwards and wrapping.
#[inline]
const fn between(lo: usize, x: usize, hi: usize) -> bool {
    if lo <= hi {
        lo < x && x <= hi
    } else {
        x > lo || x <= hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Space;

    fn a(n: u64) -> Addr {
        Addr::new(Space::Arena, n * 64)
    }

    #[test]
    fn an_empty_set_holds_nothing_and_costs_nothing() {
        let t = Tagged::new();
        assert_eq!(t.len(), 0);
        assert_eq!(t.memory_bytes(), 0);
        assert!(!t.contains(a(1)));
        let mut seen = 0;
        t.sample(0, |_| {
            seen += 1;
            true
        });
        assert_eq!(seen, 0);
    }

    #[test]
    fn what_goes_in_comes_back_out() {
        let mut t = Tagged::new();
        for i in 1..1_000 {
            assert!(t.insert(a(i)), "{i} was not there");
        }
        assert_eq!(t.len(), 999);
        for i in 1..1_000 {
            assert!(t.contains(a(i)), "{i} went missing");
        }
        assert!(!t.contains(a(1_000)));
    }

    #[test]
    fn tagging_twice_is_tagging_once() {
        let mut t = Tagged::new();
        assert!(t.insert(a(7)));
        assert!(!t.insert(a(7)));
        assert_eq!(t.len(), 1);
        assert!(t.remove(a(7)));
        assert!(!t.remove(a(7)));
        assert_eq!(t.len(), 0);
    }

    /// The point of backward shift deletion: everything still there is still
    /// findable after any pattern of removals, with nothing left behind.
    #[test]
    fn removing_the_middle_leaves_the_rest_findable() {
        let mut t = Tagged::new();
        for i in 1..2_000 {
            t.insert(a(i));
        }
        for i in (1..2_000).step_by(3) {
            assert!(t.remove(a(i)), "{i} should have been there");
        }
        for i in 1..2_000 {
            let want = i % 3 != 1;
            assert_eq!(t.contains(a(i)), want, "{i}");
        }
        assert_eq!(t.len(), 1_999 - (1..2_000).step_by(3).count());
    }

    /// A cache that fills and empties over and over does not degrade, which is
    /// the thing tombstones would have taken away.
    #[test]
    fn churn_does_not_leave_anything_behind() {
        let mut t = Tagged::new();
        for round in 0..50u64 {
            for i in 1..200 {
                t.insert(a(round * 1_000 + i));
            }
            for i in 1..200 {
                assert!(t.remove(a(round * 1_000 + i)));
            }
            assert_eq!(t.len(), 0, "round {round}");
        }
        // Back to the floor, rather than holding whatever the peak was.
        assert!(t.memory_bytes() <= MIN_SLOTS * size_of::<Addr>());
    }

    #[test]
    fn a_sample_offers_every_tagged_address_and_no_others() {
        let mut t = Tagged::new();
        for i in 1..500 {
            t.insert(a(i));
        }
        let mut seen = std::collections::HashSet::new();
        t.sample(12_345, |x| {
            assert!(seen.insert(x), "{x:?} came round twice");
            true
        });
        assert_eq!(seen.len(), 499);
        for i in 1..500 {
            assert!(seen.contains(&a(i)));
        }
    }

    #[test]
    fn a_sample_stops_when_it_is_told_to() {
        let mut t = Tagged::new();
        for i in 1..500 {
            t.insert(a(i));
        }
        let mut seen = 0;
        t.sample(99, |_| {
            seen += 1;
            seen < 20
        });
        assert_eq!(seen, 20);
    }

    /// Two samples from different places do not start in the same place, which
    /// is what stops the expire cycle sweeping the same twenty keys forever.
    #[test]
    fn a_sample_starts_where_it_is_told_to() {
        let mut t = Tagged::new();
        for i in 1..500 {
            t.insert(a(i));
        }
        let first = |r| {
            let mut got = None;
            t.sample(r, |x| {
                got = Some(x);
                false
            });
            got.expect("something is tagged")
        };
        let mut starts = std::collections::HashSet::new();
        for r in 0..64u64 {
            starts.insert(first(r * 7));
        }
        assert!(starts.len() > 8, "{} distinct starts", starts.len());
    }

    #[test]
    fn the_stretch_test_wraps() {
        assert!(between(2, 3, 5));
        assert!(between(2, 5, 5));
        assert!(!between(2, 2, 5));
        assert!(!between(2, 6, 5));
        // Wrapped: the stretch after 6 and up to 2 in a table of eight.
        assert!(between(6, 7, 2));
        assert!(between(6, 0, 2));
        assert!(between(6, 2, 2));
        assert!(!between(6, 6, 2));
        assert!(!between(6, 3, 2));
    }
}
