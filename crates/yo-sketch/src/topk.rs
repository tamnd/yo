//! The heavy keeper, which answers what the commonest things are.
//!
//! A count min sketch can tell you how often it has seen an item you name, and
//! it cannot tell you what it has been seeing, because it never keeps an item.
//! This one keeps `k` of them. Underneath is a table of `depth` rows and `width`
//! buckets, and a bucket holds a fingerprint and a count rather than a count on
//! its own, so a bucket knows whether the item arriving is the one it is
//! counting. If it is, the count goes up. If it is not, the count comes down,
//! but only sometimes: the chance of it coming down is `decay` to the power of
//! the count, so a bucket holding a big number is nearly impossible to shift and
//! a bucket holding a small one gives way at once. That is the whole idea of
//! heavy keeper, and it is why a rare item cannot evict a common one no matter
//! how many rare items arrive.
//!
//! Beside the table is a heap of the `k` items with the biggest counts, keyed on
//! the smallest, so an arriving item is compared against the smallest of the
//! kept ones and takes its place when it is bigger.
//!
//! ```
//! use yo_sketch::topk::TopK;
//!
//! let mut t = TopK::new(2, 100, 5, 0.9).expect("a sketch that small always fits");
//! t.add(b"apple", 10);
//! t.add(b"pear", 3);
//! t.add(b"fig", 1);
//! assert!(t.query(b"apple"));
//! assert!(!t.query(b"fig"), "only two are kept and this is the smallest");
//! assert_eq!(t.count_of(b"apple"), 10);
//! ```
//!
//! # Where this differs from the module
//!
//! The structure, the hash and the heap are RedisBloom's, down to the order the
//! heap comes out in, because `TOPK.LIST` is a reply a client reads and the
//! order of it is a fact about the heap rather than about the counts.
//!
//! The draw is not. The C calls `rand()` for every unit of every increment that
//! lands on a bucket held by something else, which means the same commands
//! against the same server twice give different counts, and it means a replica
//! replaying the same commands is not the same sketch. Twenty adds of one item
//! and two hundred of another into a single bucket gave 133, 115 and 146 on
//! three runs of one server here. This draws from a generator that belongs to
//! the sketch and is advanced by the sketch, so the same commands in the same
//! order always give the same answer, on this server and on anything replaying
//! its writes. The distribution is the same one. D-49 has the argument.
//!
//! The other difference is arithmetic rather than probability. A bucket counter
//! is a `u32` and the C lets it wrap, so a bucket that has been incremented
//! about forty three thousand times at the maximum increment reads as empty and
//! is handed to the next item that asks. This one stops at the ceiling. D-50.
//!
//! # The loop that is not a loop
//!
//! The C walks the decay one unit at a time, so `TOPK.INCRBY item 100000` on a
//! contended bucket is a hundred thousand calls to `rand()` per row. Nearly all
//! of them do nothing, because the chance of a decay is `decay` to the power of
//! a count that is usually large. The number of trials before the first one that
//! lands is a geometric variable, so this draws that number in one go and skips
//! straight to it. It is the same distribution and the work is proportional to
//! how many times the counter actually moves rather than to the size of the
//! increment.

use crate::hash::murmur2_32;

/// The seed the fingerprint is hashed with, which is the module's `GA`.
const FINGERPRINT_SEED: u32 = 1919;

/// The most bytes one sketch is allowed, a gibibyte across the table and the
/// heap.
///
/// The reference has no limit. `k`, the width and the depth are each free up to
/// four billion and the allocation is a `calloc`, which on Linux hands back
/// address space nobody has touched, so a client can ask for a sketch of
/// sixteen exabytes and get `OK`. Refusing at a real number is D-47, the same
/// row the count min sketch is on.
pub const MAX_BYTES: u64 = 1 << 30;

/// One bucket of the table: what it is counting, and how far it has counted.
///
/// A count of zero means the bucket is free whatever the fingerprint says, which
/// is why the counter saturating rather than wrapping matters.
#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    fp: u32,
    count: u32,
}

/// One of the `k` slots, which unlike a bucket does keep the item.
#[derive(Debug, Default)]
struct Slot {
    fp: u32,
    count: u32,
    /// `None` for a slot nothing has reached yet.
    item: Option<Box<[u8]>>,
}

/// A heavy keeper: a table of counting buckets and a heap of the biggest.
#[derive(Debug)]
pub struct TopK {
    /// How many items are kept.
    k: u32,
    /// Buckets per row.
    width: u32,
    /// Rows, which is how many buckets an item touches.
    depth: u32,
    /// How reluctant a held bucket is to give way, between zero and one.
    decay: f64,
    /// The table, row by row, so row `i` starts at `i * width`.
    cells: Box<[Cell]>,
    /// The `k` slots, as a min heap on the count.
    heap: Box<[Slot]>,
    /// The generator the decay draws from, which belongs to this sketch so that
    /// the same writes in the same order always give the same sketch.
    rng: u64,
}

impl TopK {
    /// An empty sketch, or `None` if it would be larger than [`MAX_BYTES`].
    ///
    /// All three sizes have to be at least one and `decay` has to be above zero
    /// and no more than one. The caller has already checked that, because each
    /// of them has its own sentence on the wire.
    #[must_use]
    pub fn new(k: u32, width: u32, depth: u32, decay: f64) -> Option<TopK> {
        let cells = u64::from(width).checked_mul(u64::from(depth))?;
        let bytes = cells
            .checked_mul(size_of::<Cell>() as u64)?
            .checked_add(u64::from(k).checked_mul(size_of::<Slot>() as u64)?)?;
        if k == 0 || cells == 0 || bytes > MAX_BYTES {
            return None;
        }
        let heap = (0..k).map(|_| Slot::default()).collect::<Vec<_>>();
        Some(TopK {
            k,
            width,
            depth,
            decay,
            cells: vec![Cell::default(); cells as usize].into_boxed_slice(),
            heap: heap.into_boxed_slice(),
            // Any fixed value does, since what matters is that it is the same
            // one on a replica. This is the golden ratio constant the mixer
            // below steps by, mixed with the shape so that two sketches of
            // different sizes do not draw the same numbers in the same order.
            rng: 0x9e37_79b9_7f4a_7c15
                ^ (u64::from(k) << 32)
                ^ (u64::from(width) << 16)
                ^ u64::from(depth),
        })
    }

    /// How many items are kept.
    #[must_use]
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Buckets per row.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Rows.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// How reluctant a held bucket is to give way.
    #[must_use]
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// What the sketch costs, the kept items included.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let items: usize = self
            .heap
            .iter()
            .map(|slot| slot.item.as_ref().map_or(0, |item| item.len()))
            .sum();
        size_of::<TopK>()
            + self.cells.len() * size_of::<Cell>()
            + self.k as usize * size_of::<Slot>()
            + items
    }

    /// The next number in `(0, 1)`, from the sketch's own generator.
    ///
    /// SplitMix64, which is three multiplies and some shifts and passes the
    /// usual test batteries. It is here rather than behind a dependency because
    /// the whole of it is the four lines below and a sketch that reproduces has
    /// to know exactly which generator it is reproducing.
    fn unit(&mut self) -> f64 {
        self.rng = self.rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // The top 53 bits are the ones a double can hold exactly, and the half
        // keeps the answer off both ends.
        ((z >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// How many trials it takes before one of them lands, when each has
    /// probability `p`.
    ///
    /// [`u32::MAX`] stands for "not within any number of trials anyone is going
    /// to make", which is what a probability of zero means and is what a count
    /// past about seven thousand gives at the default decay.
    fn trials_until(&mut self, p: f64) -> u32 {
        if p >= 1.0 {
            return 1;
        }
        // Zero, and also anything small enough that one subtracted from it is
        // still one, which at the default decay is any count past about three
        // hundred and fifty. A bucket that big is never going to give way and
        // the arithmetic below would divide by a logarithm of one.
        let miss = (1.0 - p).ln();
        if p <= 0.0 || miss >= 0.0 || !miss.is_finite() {
            return u32::MAX;
        }
        // The number of trials up to and including the first one that lands is
        // geometric, and `ln(u) / ln(1 - p)` is the standard way to draw one.
        let n = (self.unit().ln() / miss).floor() + 1.0;
        if n >= f64::from(u32::MAX) {
            u32::MAX
        } else {
            n as u32
        }
    }

    /// Where `item` lands in row `row`.
    fn index(&self, item: &[u8], row: u32) -> usize {
        let h = u64::from(murmur2_32(item, row));
        (u64::from(row) * u64::from(self.width) + h % u64::from(self.width)) as usize
    }

    /// Add `increment` occurrences of `item`, answering the item it pushed out
    /// of the kept set if it pushed one out.
    ///
    /// The answer is `None` both when nothing was displaced and when what was
    /// displaced was an empty slot, which the wire layer cannot tell apart
    /// either: both are a null in the reply.
    pub fn add(&mut self, item: &[u8], increment: u32) -> Option<Box<[u8]>> {
        let fp = murmur2_32(item, FINGERPRINT_SEED);
        let heap_min = self.heap[0].count;
        let mut max_count = 0u32;

        for row in 0..self.depth {
            let at = self.index(item, row);
            if self.cells[at].count == 0 {
                // A free bucket takes the item whatever it was counting before.
                self.cells[at] = Cell {
                    fp,
                    count: increment,
                };
                max_count = max_count.max(increment);
            } else if self.cells[at].fp == fp {
                // Ours, so it just goes up. The C wraps here and this stops, for
                // the reason in the module comment: a wrapped counter reads as a
                // free bucket. D-50.
                self.cells[at].count = self.cells[at].count.saturating_add(increment);
                max_count = max_count.max(self.cells[at].count);
            } else if let Some(count) = self.contest(at, fp, increment) {
                max_count = max_count.max(count);
            }
        }

        if max_count < heap_min {
            return None;
        }
        match self.slot_of(item, fp) {
            // Not the larger of the two, because the count in the table can have
            // been decayed since this item was put in the heap and the table is
            // the one that is right.
            Some(at) => {
                self.heap[at].count = max_count;
                self.heapify_down(at);
                None
            }
            None => {
                let expelled = self.heap[0].item.take();
                self.heap[0] = Slot {
                    fp,
                    count: max_count,
                    item: Some(item.into()),
                };
                self.heapify_down(0);
                expelled
            }
        }
    }

    /// Spend `increment` arrivals trying to take a bucket that something else
    /// holds, answering the count if the bucket changed hands.
    ///
    /// Each arrival gets one chance to knock the count down by one, and that
    /// chance is `decay` to the power of the count, so the bucket gets harder to
    /// move the more it holds. If the count reaches zero the bucket is ours and
    /// whatever is left of the increment is the new count.
    fn contest(&mut self, at: usize, fp: u32, increment: u32) -> Option<u32> {
        let mut left = increment;
        while left > 0 {
            let count = self.cells[at].count;
            let p = self.decay.powf(f64::from(count));
            let trials = self.trials_until(p);
            if trials == u32::MAX || trials > left {
                // Nothing landed, and there are no arrivals left to spend.
                return None;
            }
            left -= trials - 1;
            self.cells[at].count -= 1;
            if self.cells[at].count == 0 {
                self.cells[at] = Cell { fp, count: left };
                return Some(left);
            }
            left -= 1;
        }
        None
    }

    /// Whether the sketch is keeping `item`, which is what `TOPK.QUERY` asks.
    #[must_use]
    pub fn query(&self, item: &[u8]) -> bool {
        self.slot_of(item, murmur2_32(item, FINGERPRINT_SEED))
            .is_some()
    }

    /// How often the sketch thinks it has seen `item`.
    ///
    /// The largest of the buckets that are counting this fingerprint, which is
    /// the other way round from a count min sketch and is right here: a bucket
    /// this item shares with a heavier one has been decayed rather than added
    /// to, so the small readings are the wrong ones. The one exception is an
    /// item the heap is keeping, where a bucket reading below the smallest kept
    /// count cannot be about this item and is left out.
    #[must_use]
    pub fn count_of(&self, item: &[u8]) -> u32 {
        let fp = murmur2_32(item, FINGERPRINT_SEED);
        let heap_min = self.heap[0].count;
        let kept = self.slot_of(item, fp).is_some();
        let mut most = 0;
        for row in 0..self.depth {
            let cell = self.cells[self.index(item, row)];
            if cell.fp == fp && (!kept || cell.count >= heap_min) {
                most = most.max(cell.count);
            }
        }
        most
    }

    /// The kept items, heaviest first, as pairs of item and count.
    ///
    /// A slot nothing has reached is left out, and so is one holding an item
    /// whose count is still zero, which `TOPK.INCRBY item 0` can produce.
    #[must_use]
    pub fn list(&self) -> Vec<(&[u8], u32)> {
        let mut order: Vec<&Slot> = self.heap.iter().collect();
        // Stable, so slots with equal counts stay in the order the heap holds
        // them. That order is what the reference answers with and it is a fact
        // about how the heap was built rather than about the counts.
        order.sort_by_key(|slot| std::cmp::Reverse(slot.count));
        order
            .iter()
            .take_while(|slot| slot.count != 0)
            .filter_map(|slot| slot.item.as_deref().map(|item| (item, slot.count)))
            .collect()
    }

    /// Which slot holds `item`, if one does.
    ///
    /// Walked from the back, because the reference walks it from the back and
    /// answers the last match. Two slots cannot hold the same item, so the
    /// direction only shows in what the code costs.
    fn slot_of(&self, item: &[u8], fp: u32) -> Option<usize> {
        (0..self.heap.len())
            .rev()
            .find(|&at| self.heap[at].fp == fp && self.heap[at].item.as_deref() == Some(item))
    }

    /// Push the slot at `start` down until the heap is a heap again.
    ///
    /// This is the reference's sift, quirk for quirk, because the order the
    /// slots end up in is the order `TOPK.LIST` answers in when counts are
    /// equal. The quirk is that a slot sinks past a child holding the same count
    /// rather than stopping at it, so inserting five items with a count of one
    /// each does not leave them in the order they arrived.
    fn heapify_down(&mut self, start: usize) {
        let len = self.heap.len();
        let mut start = start;
        if len < 2 || (len - 2) / 2 < start {
            return;
        }
        let mut child = 2 * start + 1;
        if child + 1 < len && self.heap[child].count > self.heap[child + 1].count {
            child += 1;
        }
        if self.heap[child].count > self.heap[start].count {
            return;
        }
        let top = std::mem::take(&mut self.heap[start]);
        loop {
            self.heap[start] = std::mem::take(&mut self.heap[child]);
            start = child;
            if (len - 2) / 2 < child {
                break;
            }
            child = 2 * child + 1;
            if child + 1 < len && self.heap[child].count > self.heap[child + 1].count {
                child += 1;
            }
            if self.heap[child].count >= top.count {
                break;
            }
        }
        self.heap[start] = top;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_heavy_item_stays_and_the_light_ones_do_not() {
        let mut t = TopK::new(3, 1000, 5, 0.9).unwrap();
        t.add(b"heavy", 1000);
        for i in 0..500 {
            t.add(format!("light{i}").as_bytes(), 1);
        }
        assert!(
            t.query(b"heavy"),
            "five hundred rare items did not shift it"
        );
        assert_eq!(t.count_of(b"heavy"), 1000);
        let listed = t.list();
        assert_eq!(listed[0].0, b"heavy");
        assert_eq!(listed[0].1, 1000);
        assert_eq!(listed.len(), 3, "the other two slots are rare items");
    }

    #[test]
    fn the_kept_set_is_the_biggest_k_and_the_list_is_sorted() {
        let mut t = TopK::new(3, 1000, 5, 0.9).unwrap();
        for (item, n) in [(&b"a"[..], 10), (b"b", 7), (b"c", 5), (b"d", 3)] {
            t.add(item, n);
        }
        assert_eq!(
            t.list(),
            vec![(&b"a"[..], 10), (&b"b"[..], 7), (&b"c"[..], 5)]
        );
        assert!(!t.query(b"d"));
        // The table still counts what the heap has let go.
        assert_eq!(t.count_of(b"d"), 3);
        // And an item that grows takes a kept one's place.
        assert_eq!(t.add(b"d", 4).as_deref(), Some(&b"c"[..]));
        assert_eq!(
            t.list(),
            vec![(&b"a"[..], 10), (&b"b"[..], 7), (&b"d"[..], 7)]
        );
    }

    /// The order five items with a count of one each come out in, which is the
    /// reference's, and is the whole reason the sift above is copied rather than
    /// written the way anyone would write it.
    #[test]
    fn equal_counts_come_out_in_the_order_the_heap_holds_them() {
        let mut t = TopK::new(5, 1000, 7, 0.9).unwrap();
        for item in [&b"v"[..], b"w", b"x", b"y", b"z"] {
            t.add(item, 1);
        }
        let listed: Vec<&[u8]> = t.list().iter().map(|(item, _)| *item).collect();
        assert_eq!(listed, vec![&b"x"[..], b"z", b"y", b"v", b"w"]);
    }

    #[test]
    fn an_item_with_a_count_of_zero_is_kept_and_not_listed() {
        let mut t = TopK::new(3, 100, 5, 0.9).unwrap();
        assert!(t.add(b"nothing", 0).is_none());
        assert!(t.query(b"nothing"), "it is in the heap");
        assert_eq!(t.count_of(b"nothing"), 0);
        assert!(t.list().is_empty(), "and a count of zero is not listed");
    }

    #[test]
    fn a_contested_bucket_gives_way_slowly() {
        // One bucket in one row, so everything collides with everything.
        let mut t = TopK::new(1, 1, 1, 0.9).unwrap();
        t.add(b"held", 20);
        for _ in 0..200 {
            t.add(b"other", 1);
        }
        // Two hundred arrivals against a count of twenty took the bucket, which
        // is what the reference does too, and the count that is left is a long
        // way under two hundred because most of the arrivals were spent knocking
        // the old count down.
        let taken = t.count_of(b"other");
        assert!(taken > 0 && taken < 200, "took the bucket at {taken}");
        assert_eq!(t.count_of(b"held"), 0);
    }

    #[test]
    fn the_same_writes_give_the_same_sketch_every_time() {
        let run = || {
            let mut t = TopK::new(2, 4, 2, 0.9).unwrap();
            for i in 0..500 {
                t.add(format!("item{}", i % 7).as_bytes(), 3);
            }
            t.list()
                .iter()
                .map(|(item, n)| (item.to_vec(), *n))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
        assert_eq!(run(), run());
    }

    #[test]
    fn a_counter_stops_at_the_ceiling_rather_than_wrapping() {
        let mut t = TopK::new(1, 1, 1, 0.9).unwrap();
        t.add(b"x", u32::MAX - 1);
        t.add(b"x", 100);
        assert_eq!(t.count_of(b"x"), u32::MAX);
    }

    #[test]
    fn a_decay_of_one_gives_way_at_every_arrival() {
        let mut t = TopK::new(1, 1, 1, 1.0).unwrap();
        t.add(b"held", 10);
        for _ in 0..10 {
            t.add(b"other", 1);
        }
        // Ten arrivals against a count of ten, each of which lands for certain,
        // so the tenth one takes the empty bucket.
        assert_eq!(t.count_of(b"other"), 1);
        assert_eq!(t.count_of(b"held"), 0);
    }

    #[test]
    fn a_sketch_larger_than_the_cap_is_not_made() {
        assert!(TopK::new(1, 1, 1, 0.9).is_some());
        assert!(TopK::new(1, u32::MAX, u32::MAX, 0.9).is_none());
        assert!(TopK::new(u32::MAX, 8, 7, 0.9).is_none());
        assert!(TopK::new(0, 8, 7, 0.9).is_none());
        assert!(TopK::new(1, 0, 7, 0.9).is_none());
    }
}
