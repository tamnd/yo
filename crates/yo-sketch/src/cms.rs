//! The count min sketch, which answers how many times it has seen something.
//!
//! A table of `depth` rows and `width` counters. An item is hashed once per row
//! with the row number as the seed, and the increment is added to the counter
//! that hash picks out in that row. The answer to "how many" is the smallest of
//! the `depth` counters the item points at, and it is never too low, because
//! every one of them was incremented every time the item arrived. It can be too
//! high, because some other item may share a counter, and taking the minimum is
//! what makes that unlikely: an item is only overcounted when it collides with
//! something in every row at once.
//!
//! That is the whole structure. There is no deletion, because a decrement would
//! be wrong for every item sharing the counter, and there is no way to list what
//! is in it, because nothing is: the items themselves are never stored, which is
//! the point.
//!
//! ```
//! use yo_sketch::cms::Cms;
//!
//! let mut c = Cms::new(200, 5).expect("a sketch that small always fits");
//! assert_eq!(c.incr(b"apple", 3), 3);
//! assert_eq!(c.incr(b"apple", 4), 7);
//! assert_eq!(c.count_of(b"apple"), 7);
//! assert_eq!(c.count_of(b"pear"), 0);
//! ```
//!
//! # Why these dimensions and this hash
//!
//! Both are RedisBloom's. `CMS.INITBYPROB` turns a pair of tolerances into a
//! width and a depth with two lines of floating point, one of which runs in
//! `float` and the other in `double`, and the answers those two lines give are
//! observable through `CMS.INFO`, so they are copied rather than rewritten. The
//! hash is `MurmurHash2` in its 32 bit shape with the row number as the seed,
//! which is not the hash the Bloom filter in the same module uses and is not
//! derived from it either.
//!
//! Unlike the Bloom and cuckoo filters there is no dump command here, so none of
//! this is a file format. It is copied anyway, because `CMS.QUERY` answers a
//! number and a client that gets a different number off two servers holding the
//! same increments has no way to tell which one is lying.
//!
//! # Saturation
//!
//! A counter is a `u32` and stops at `u32::MAX` rather than wrapping, and an
//! item whose smallest counter has reached that ceiling is reported as an error
//! by the wire layer rather than as a count. The running total in `count` is a
//! signed 64 bit number that really does wrap, which is the module's behaviour
//! and is visible through `CMS.INFO` after enough has been added.

use crate::hash::murmur2_32;

/// How many counters one sketch is allowed, which is a gibibyte of them.
///
/// The reference has no limit of its own. It asks `calloc` for the whole table
/// and on Linux that succeeds for anything up to the address space, so a client
/// can ask for a hundred and forty terabytes of counters and get `OK`, then
/// bring the server down by touching them. Refusing at a real number is D-47.
pub const MAX_CELLS: u64 = 1 << 28;

/// A count min sketch.
#[derive(Debug)]
pub struct Cms {
    /// Counters per row.
    width: u64,
    /// Rows, which is how many counters an item touches.
    depth: u64,
    /// The table, row by row, so row `i` starts at `i * width`.
    cells: Box<[u32]>,
    /// Everything ever added, whether it fitted in the counters or not.
    count: i64,
}

impl Cms {
    /// An empty sketch, or `None` if it would be larger than [`MAX_CELLS`].
    ///
    /// Both dimensions have to be at least one. The caller has already decided
    /// what to say about a zero, because the two have different sentences on the
    /// wire, so this treats it as a size like any other and refuses it here
    /// only if the product is silly.
    #[must_use]
    pub fn new(width: u64, depth: u64) -> Option<Cms> {
        let cells = width.checked_mul(depth)?;
        if cells == 0 || cells > MAX_CELLS {
            return None;
        }
        Some(Cms {
            width,
            depth,
            cells: vec![0u32; cells as usize].into_boxed_slice(),
            count: 0,
        })
    }

    /// Counters per row.
    #[must_use]
    pub fn width(&self) -> u64 {
        self.width
    }

    /// Rows.
    #[must_use]
    pub fn depth(&self) -> u64 {
        self.depth
    }

    /// Everything ever added to this sketch, wrapping included.
    #[must_use]
    pub fn count(&self) -> i64 {
        self.count
    }

    /// What the sketch costs.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        size_of::<Cms>() + self.cells.len() * size_of::<u32>()
    }

    /// Where `item` lands in row `row`.
    fn index(&self, item: &[u8], row: u64) -> usize {
        // The row number is the seed, and it fits in a `u32` because the cell
        // cap keeps the depth well under four billion.
        let h = u64::from(murmur2_32(item, row as u32));
        (row * self.width + h % self.width) as usize
    }

    /// Add `by` to every counter `item` points at and answer the smallest of
    /// them afterwards.
    ///
    /// A counter that would go past `u32::MAX` stops there, so the answer can be
    /// `u32::MAX` without `by` having been that large. `by` is expected to be
    /// zero or more; a negative one would take counters down and is refused on
    /// the wire rather than here.
    pub fn incr(&mut self, item: &[u8], by: i64) -> u32 {
        let mut min = u32::MAX;
        for row in 0..self.depth {
            let at = self.index(item, row);
            let cell = &mut self.cells[at];
            let room = i64::from(u32::MAX - *cell);
            *cell = if room < by {
                u32::MAX
            } else {
                *cell + by as u32
            };
            min = min.min(*cell);
        }
        // The total is what was asked for rather than what fitted, and it wraps
        // rather than saturating, so a sketch that has been pushed past the
        // ceiling reports a negative count.
        self.count = self.count.wrapping_add(by);
        min
    }

    /// How many times the sketch thinks it has seen `item`, which is never too
    /// low and is sometimes too high.
    #[must_use]
    pub fn count_of(&self, item: &[u8]) -> u32 {
        (0..self.depth)
            .map(|row| self.cells[self.index(item, row)])
            .min()
            .unwrap_or(0)
    }

    /// An empty accumulator shaped like this sketch, which is where a merge is
    /// built before it replaces anything.
    #[must_use]
    pub fn merge_start(&self) -> Merge {
        Merge {
            cells: vec![0u32; self.cells.len()],
            count: 0,
        }
    }

    /// Add this sketch to `into`, `weight` times, answering whether it fitted.
    ///
    /// A source counter times its weight has to land between zero and
    /// `u32::MAX`, and so does the running sum, or the whole merge is off. That
    /// makes a negative weight legal against a counter that is zero and illegal
    /// against one that is not, which reads like an accident and is what the
    /// reference does.
    ///
    /// `into` is left half written when this answers `false`. Nothing has been
    /// written to any real sketch at that point, so the caller drops it.
    pub fn merge_add(&self, into: &mut Merge, weight: i64) -> bool {
        if into.cells.len() != self.cells.len() {
            return false;
        }
        for (dst, &src) in into.cells.iter_mut().zip(self.cells.iter()) {
            // In `i128` because the C multiplies a counter by a weight that can
            // be `i64::MAX` and reads the answer as if it had not overflowed.
            let value = i128::from(src) * i128::from(weight);
            let sum = i128::from(*dst) + value;
            if value < 0 || sum > i128::from(u32::MAX) {
                return false;
            }
            *dst = sum as u32;
        }
        into.count = into.count.wrapping_add(weight.wrapping_mul(self.count));
        true
    }

    /// Replace this sketch's counters with a merge that fitted.
    ///
    /// The destination is overwritten and not added to, so a sketch that was not
    /// named among the sources loses everything it had. That is what makes
    /// naming the destination among its own sources the way to add rather than
    /// replace, and it is why this takes the accumulator by value.
    ///
    /// # Panics
    ///
    /// If the accumulator did not come from [`Cms::merge_start`] on a sketch of
    /// these dimensions.
    pub fn merge_finish(&mut self, from: Merge) {
        assert_eq!(from.cells.len(), self.cells.len(), "merged the wrong shape");
        self.cells = from.cells.into_boxed_slice();
        self.count = from.count;
    }
}

/// A merge in progress, which is a table of counters and nothing else.
#[derive(Debug)]
pub struct Merge {
    /// The weighted sums so far.
    cells: Vec<u32>,
    /// The weighted totals so far.
    count: i64,
}

/// The width and depth a pair of tolerances asks for, or `None` if either comes
/// out too large to be a size at all.
///
/// Two lines of the reference's arithmetic, and they do not agree with each
/// other. The width is `ceil(2 / error)` in `double`, so an error of a
/// billionth gives exactly two billion. The depth is `ceil(log10(prob) /
/// log10(0.5))` in `float`, because the C calls `log10f` and the argument is
/// converted on the way in, so a probability below the smallest float is read as
/// zero and refused however far a `double` could have carried it. Both of those
/// are visible through `CMS.INFO` and both are copied.
///
/// `error` and `prob` are expected to be strictly between zero and one. The
/// caller checks that, because it has a different sentence for each of them.
#[must_use]
pub fn dims_from(error: f64, prob: f64) -> Option<(u64, u64)> {
    let width = (2.0 / error).ceil();
    // `as f32` is the conversion the C does when it passes a double to
    // `log10f`, and it is the whole reason a probability of 1e-46 is refused.
    let depth = (f64::from(prob as f32).log10() / 0.5f64.log10()).ceil();
    let fits = |n: f64| n.is_finite() && n >= 1.0 && n <= i64::MAX as f64;
    if !fits(width) || !fits(depth) {
        return None;
    }
    Some((width as u64, depth as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_is_never_too_low() {
        let mut c = Cms::new(4, 2).unwrap();
        for i in 0..50 {
            let item = format!("item{i}");
            c.incr(item.as_bytes(), 1);
        }
        // Four counters and fifty items, so every answer is an overcount and
        // none of them is under one.
        for i in 0..50 {
            let item = format!("item{i}");
            assert!(c.count_of(item.as_bytes()) >= 1);
        }
    }

    #[test]
    fn a_wide_sketch_gets_the_small_counts_exactly_right() {
        let mut c = Cms::new(2000, 5).unwrap();
        for i in 0..100 {
            let item = format!("k{i}");
            c.incr(item.as_bytes(), i);
        }
        for i in 0..100 {
            let item = format!("k{i}");
            assert_eq!(c.count_of(item.as_bytes()), i as u32, "at {i}");
        }
        assert_eq!(c.count(), (0..100).sum::<i64>());
    }

    #[test]
    fn a_counter_stops_at_the_ceiling_and_the_total_carries_on() {
        let mut c = Cms::new(8, 1).unwrap();
        assert_eq!(c.incr(b"x", i64::from(u32::MAX) - 1), u32::MAX - 1);
        assert_eq!(c.incr(b"x", 5), u32::MAX);
        assert_eq!(c.count(), i64::from(u32::MAX) + 4);
        // And the total wraps where the counter did not.
        c.incr(b"x", i64::MAX);
        assert!(c.count() < 0, "the running total is signed and wraps");
    }

    #[test]
    fn a_merge_is_a_replacement_and_not_an_addition() {
        let mut a = Cms::new(64, 3).unwrap();
        let mut b = Cms::new(64, 3).unwrap();
        a.incr(b"a", 7);
        b.incr(b"b", 5);
        let mut m = a.merge_start();
        assert!(b.merge_add(&mut m, 2));
        a.merge_finish(m);
        assert_eq!(a.count_of(b"a"), 0, "what the destination had is gone");
        assert_eq!(a.count_of(b"b"), 10);
        assert_eq!(a.count(), 10);
    }

    #[test]
    fn a_merge_that_would_overflow_a_counter_is_refused_whole() {
        let mut a = Cms::new(16, 1).unwrap();
        let b = {
            let mut b = Cms::new(16, 1).unwrap();
            b.incr(b"x", 4);
            b
        };
        let mut m = a.merge_start();
        assert!(
            !b.merge_add(&mut m, i64::MAX),
            "four times that is not a u32"
        );
        // Nothing was written, because a refusal never reaches the sketch.
        assert_eq!(a.count_of(b"x"), 0);
        let mut m = a.merge_start();
        assert!(b.merge_add(&mut m, i64::from(u32::MAX) / 4));
        a.merge_finish(m);
        assert_eq!(a.count_of(b"x"), (u32::MAX / 4) * 4);
    }

    #[test]
    fn a_negative_weight_is_only_a_problem_when_there_is_something_to_weigh() {
        let empty = Cms::new(16, 1).unwrap();
        let mut held = Cms::new(16, 1).unwrap();
        held.incr(b"x", 1);
        let mut m = empty.merge_start();
        assert!(
            empty.merge_add(&mut m, -1),
            "nothing times minus one is fine"
        );
        assert!(!held.merge_add(&mut m, -1));
    }

    #[test]
    fn a_sketch_larger_than_the_cap_is_not_made() {
        assert!(Cms::new(MAX_CELLS, 1).is_some());
        assert!(Cms::new(MAX_CELLS + 1, 1).is_none());
        assert!(Cms::new(MAX_CELLS / 2, 3).is_none());
        assert!(Cms::new(u64::MAX, u64::MAX).is_none(), "the product wraps");
        assert!(Cms::new(0, 4).is_none());
    }

    /// The numbers on the right came off a live Redis 8.10.1 through
    /// `CMS.INITBYPROB` and `CMS.INFO`.
    #[test]
    fn the_dimensions_are_the_ones_the_module_computes() {
        assert_eq!(dims_from(0.001, 0.01), Some((2000, 7)));
        assert_eq!(dims_from(0.01, 0.5), Some((200, 1)));
        assert_eq!(dims_from(0.0001, 0.0001), Some((20000, 14)));
        assert_eq!(dims_from(0.5, 0.5), Some((4, 1)));
        assert_eq!(dims_from(0.9, 0.9), Some((3, 1)));
        assert_eq!(dims_from(0.06, 0.6), Some((34, 1)));
        assert_eq!(dims_from(1e-9, 1e-9), Some((2_000_000_000, 30)));
        // The depth runs in `float`, so the smallest denormal double is read as
        // zero and this is refused rather than answering 1074.
        assert_eq!(dims_from(0.5, 1e-46), None);
        // But a probability that survives the conversion is answered from the
        // denormal it lands on, which is why this is 149 and not the 150 the
        // same line in `double` would give.
        assert_eq!(dims_from(0.5, 1e-45), Some((4, 149)));
        // The width is the one that runs in `double`, and it is refused when it
        // is past what a signed 64 bit number could hold.
        assert_eq!(
            dims_from(2.2e-19, 0.5),
            Some((9_090_909_090_909_091_840, 1))
        );
        assert_eq!(dims_from(1.9e-19, 0.5), None);
        // A probability that rounds to one asks for no rows at all, and the
        // reference refuses that here rather than making an empty sketch.
        assert_eq!(dims_from(0.5, 0.999_999_99), None);
        assert_eq!(dims_from(0.5, 0.999_999_9), Some((4, 1)));
    }
}
