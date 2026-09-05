//! The numeric index: which documents hold a number, and which of those fall
//! between two others.
//!
//! ```
//! use yo_search::nums::{Ends, Nums};
//!
//! let mut n = Nums::new();
//! n.add(1, 10.0);
//! n.add(2, 20.5);
//! n.add(3, -3.0);
//! n.settle();
//!
//! assert_eq!(n.range(Ends::shut(0.0, 20.5)), [1, 2]);
//! assert_eq!(n.range(Ends::shut(0.0, 20.5).top_open()), [1]);
//! assert_eq!(n.range(Ends::all()), [1, 2, 3]);
//! ```
//!
//! # Why it is two lists and not one
//!
//! Answering a range means finding where it starts, and finding where anything
//! starts means the values are in order. Keeping them in order as they arrive
//! means moving half the index on every document, which is the wrong cost for a
//! load of ten million hashes. So what arrives goes on the end of a short list
//! in the order it came, and [`Nums::settle`] folds that list into the long
//! ordered one in one pass. A read looks in both, binary searching the ordered
//! list and walking the short one, so a query is correct whether or not anything
//! has settled and is fast whenever the writer has had a moment to settle it.
//!
//! # What is not a number
//!
//! A value that is not a number never reaches here, and a NaN is refused at the
//! door rather than being stored, because a NaN compares false against
//! everything and one of them in the ordered list would cut it in half. A range
//! with a NaN end matches nothing for the same reason, which is what asking for
//! it means.

use crate::posts::Id;
use crate::query::Range;

/// One document's number.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    value: f64,
    id: Id,
}

/// The ends of a range and whether each end is part of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ends {
    /// The bottom, which may be negative infinity.
    pub min: f64,
    /// The top, which may be infinity.
    pub max: f64,
    /// Whether the bottom is excluded.
    pub min_open: bool,
    /// Whether the top is excluded.
    pub max_open: bool,
}

impl Ends {
    /// A range with both ends included.
    #[must_use]
    pub const fn shut(min: f64, max: f64) -> Ends {
        Ends {
            min,
            max,
            min_open: false,
            max_open: false,
        }
    }

    /// Every number there is.
    #[must_use]
    pub const fn all() -> Ends {
        Ends::shut(f64::NEG_INFINITY, f64::INFINITY)
    }

    /// The same range with the bottom excluded.
    #[must_use]
    pub const fn bottom_open(self) -> Ends {
        Ends {
            min_open: true,
            ..self
        }
    }

    /// The same range with the top excluded.
    #[must_use]
    pub const fn top_open(self) -> Ends {
        Ends {
            max_open: true,
            ..self
        }
    }

    /// Whether a number is in the range.
    #[must_use]
    pub fn holds(&self, value: f64) -> bool {
        self.above(value) && self.below(value)
    }

    /// Whether a number is at or past the bottom.
    fn above(&self, value: f64) -> bool {
        if self.min_open {
            value > self.min
        } else {
            value >= self.min
        }
    }

    /// Whether a number is at or before the top.
    fn below(&self, value: f64) -> bool {
        if self.max_open {
            value < self.max
        } else {
            value <= self.max
        }
    }
}

impl From<&Range> for Ends {
    fn from(range: &Range) -> Ends {
        Ends {
            min: range.min,
            max: range.max,
            min_open: range.min_open,
            max_open: range.max_open,
        }
    }
}

/// Every number one field holds, over every document that has it.
#[derive(Debug, Clone, Default)]
pub struct Nums {
    kept: Vec<Point>,
    fresh: Vec<Point>,
    last: Id,
}

impl Nums {
    /// An index with nothing in it.
    #[must_use]
    pub fn new() -> Nums {
        Nums::default()
    }

    /// Records that a document holds a number.
    ///
    /// A document may hold more than one, which is what a document database
    /// means by an array of numbers under one path, and each of them is its own
    /// entry. A NaN is not a number and is dropped.
    pub fn add(&mut self, id: Id, value: f64) {
        if value.is_nan() {
            return;
        }
        self.fresh.push(Point { value, id });
        self.last = self.last.max(id);
    }

    /// Folds everything that has arrived into the ordered list.
    ///
    /// Worth doing after a batch of writes and not after each one. Nothing goes
    /// wrong if it is never called, reads only get slower.
    pub fn settle(&mut self) {
        if self.fresh.is_empty() {
            return;
        }
        self.fresh
            .sort_unstable_by(|a, b| a.value.total_cmp(&b.value).then(a.id.cmp(&b.id)));
        if self.kept.is_empty() {
            std::mem::swap(&mut self.kept, &mut self.fresh);
            return;
        }
        let mut both = Vec::with_capacity(self.kept.len() + self.fresh.len());
        let (mut left, mut right) = (0, 0);
        while left < self.kept.len() && right < self.fresh.len() {
            let a = self.kept[left];
            let b = self.fresh[right];
            if a.value.total_cmp(&b.value).then(a.id.cmp(&b.id)).is_le() {
                both.push(a);
                left += 1;
            } else {
                both.push(b);
                right += 1;
            }
        }
        both.extend_from_slice(&self.kept[left..]);
        both.extend_from_slice(&self.fresh[right..]);
        self.kept = both;
        self.fresh.clear();
    }

    /// The documents whose number is in the range, in order and each once.
    ///
    /// A document that holds the number twice, or holds two numbers that are
    /// both in the range, is one answer and not two.
    #[must_use]
    pub fn range(&self, ends: Ends) -> Vec<Id> {
        let mut out = Vec::new();
        if ends.min.is_nan() || ends.max.is_nan() {
            return out;
        }
        let from = self.kept.partition_point(|p| !ends.above(p.value));
        for point in &self.kept[from..] {
            if !ends.below(point.value) {
                break;
            }
            out.push(point.id);
        }
        for point in &self.fresh {
            if ends.holds(point.value) {
                out.push(point.id);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// How many numbers are held, counting a document twice if it holds two.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kept.len() + self.fresh.len()
    }

    /// Whether no document holds a number here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The largest document number seen, or zero when there is none.
    #[must_use]
    pub const fn last(&self) -> Id {
        self.last
    }

    /// The smallest and largest number held, or `None` when there is none.
    #[must_use]
    pub fn bounds(&self) -> Option<(f64, f64)> {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for point in self.kept.iter().chain(&self.fresh) {
            low = low.min(point.value);
            high = high.max(point.value);
        }
        (!self.is_empty()).then_some((low, high))
    }

    /// How many different numbers are held.
    ///
    /// Settled first or not, which is why it does the work rather than reading
    /// the ordered list off. It is a debug reply and not a query, so the cost is
    /// paid where it shows.
    #[must_use]
    pub fn spread(&self) -> usize {
        let mut all: Vec<f64> = self
            .kept
            .iter()
            .chain(&self.fresh)
            .map(|p| p.value)
            .collect();
        all.sort_unstable_by(f64::total_cmp);
        all.dedup_by(|a, b| a.total_cmp(b).is_eq());
        all.len()
    }

    /// How many bytes the entries take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.len() * size_of::<Point>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(values: &[f64]) -> Nums {
        let mut n = Nums::new();
        for (at, value) in values.iter().enumerate() {
            n.add(u32::try_from(at).expect("small test") + 1, *value);
        }
        n.settle();
        n
    }

    /// The ends are asked for one at a time and each one may be in or out.
    #[test]
    fn either_end_may_be_left_out() {
        let n = built(&[10.0, 20.0, 30.0]);
        assert_eq!(n.range(Ends::shut(10.0, 30.0)), [1, 2, 3]);
        assert_eq!(n.range(Ends::shut(10.0, 30.0).bottom_open()), [2, 3]);
        assert_eq!(n.range(Ends::shut(10.0, 30.0).top_open()), [1, 2]);
        assert_eq!(
            n.range(Ends::shut(10.0, 30.0).bottom_open().top_open()),
            [2]
        );
        assert_eq!(n.range(Ends::shut(20.0, 20.0)), [2]);
        assert_eq!(n.range(Ends::shut(20.0, 20.0).top_open()), Vec::<Id>::new());
    }

    /// Both ends may be infinite, and an infinity that is stored is a number
    /// like any other rather than a way of saying there is nothing there.
    #[test]
    fn an_end_may_be_infinite() {
        let n = built(&[-1.0, 0.0, 1.0, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(n.range(Ends::all()), [1, 2, 3, 4, 5]);
        assert_eq!(
            n.range(Ends {
                min: 0.0,
                max: f64::INFINITY,
                min_open: false,
                max_open: false
            }),
            [2, 3, 4]
        );
        assert_eq!(
            n.range(Ends {
                min: 0.0,
                max: f64::INFINITY,
                min_open: false,
                max_open: true
            }),
            [2, 3]
        );
        assert_eq!(n.range(Ends::all().bottom_open()), [1, 2, 3, 4]);
    }

    /// A range that starts above where it ends holds nothing, and so does one
    /// with a NaN on either end, because that is what asking for it means.
    #[test]
    fn a_range_that_holds_nothing_answers_with_nothing() {
        let n = built(&[1.0, 2.0, 3.0]);
        assert_eq!(n.range(Ends::shut(3.0, 1.0)), Vec::<Id>::new());
        assert_eq!(n.range(Ends::shut(f64::NAN, 1.0)), Vec::<Id>::new());
        assert_eq!(n.range(Ends::shut(1.0, f64::NAN)), Vec::<Id>::new());
        assert_eq!(n.range(Ends::shut(10.0, 20.0)), Vec::<Id>::new());
        assert_eq!(Nums::new().range(Ends::all()), Vec::<Id>::new());
    }

    /// A NaN is not a number and does not go in, because one of them in the
    /// ordered list compares false against everything and cuts it in half.
    #[test]
    fn a_nan_is_not_stored() {
        let mut n = Nums::new();
        n.add(1, f64::NAN);
        n.add(2, 1.0);
        n.settle();
        assert_eq!(n.len(), 1);
        assert_eq!(n.range(Ends::all()), [2]);
        assert_eq!(n.spread(), 1);
    }

    /// A read is right whether or not anything has settled, because it looks in
    /// the ordered list and in what has arrived since.
    #[test]
    fn a_read_before_settling_is_the_same_read() {
        let mut n = Nums::new();
        for (id, value) in [(1, 5.0), (2, 1.0), (3, 9.0)] {
            n.add(id, value);
        }
        assert_eq!(n.range(Ends::shut(1.0, 5.0)), [1, 2]);
        n.settle();
        assert_eq!(n.range(Ends::shut(1.0, 5.0)), [1, 2]);
        n.add(4, 2.0);
        assert_eq!(n.range(Ends::shut(1.0, 5.0)), [1, 2, 4]);
        n.settle();
        assert_eq!(n.range(Ends::shut(1.0, 5.0)), [1, 2, 4]);
        assert_eq!(n.range(Ends::all()), [1, 2, 3, 4]);
    }

    /// Settling twice over is the same as settling once, and settling nothing
    /// is not a special case.
    #[test]
    fn settling_again_changes_nothing() {
        let mut n = built(&[3.0, 1.0, 2.0]);
        let once = n.range(Ends::all());
        n.settle();
        n.settle();
        assert_eq!(n.range(Ends::all()), once);
        assert_eq!(n.len(), 3);
    }

    /// A document may hold the same number twice or two numbers in the same
    /// range, and it is one answer either way.
    #[test]
    fn a_document_answers_once_however_many_numbers_it_has() {
        let mut n = Nums::new();
        n.add(1, 1.0);
        n.add(1, 2.0);
        n.add(1, 2.0);
        n.add(2, 3.0);
        n.settle();
        assert_eq!(n.len(), 4);
        assert_eq!(n.range(Ends::all()), [1, 2]);
        assert_eq!(n.range(Ends::shut(1.0, 2.0)), [1]);
        assert_eq!(n.spread(), 3);
    }

    /// Zero has two spellings and they are the same number, which matters
    /// because the ordering used here is the one that says otherwise.
    #[test]
    fn there_is_only_one_zero() {
        let n = built(&[0.0, -0.0, 1.0]);
        assert_eq!(n.range(Ends::shut(0.0, 0.0)), [1, 2]);
        assert_eq!(n.range(Ends::shut(-0.0, -0.0)), [1, 2]);
        assert_eq!(
            n.range(Ends::shut(0.0, f64::INFINITY).bottom_open()),
            [3],
            "neither spelling of zero is above zero"
        );
    }

    /// What the index says about itself, which is what the debug reply reports.
    #[test]
    fn an_index_says_what_it_holds() {
        let empty = Nums::new();
        assert!(empty.is_empty());
        assert_eq!(empty.bounds(), None);
        assert_eq!(empty.last(), 0);
        assert_eq!(empty.spread(), 0);
        assert_eq!(empty.bytes(), 0);
        let n = built(&[10.0, 20.5, -3.0, 10.0, 1.0]);
        assert_eq!(n.len(), 5);
        assert_eq!(n.bounds(), Some((-3.0, 20.5)));
        assert_eq!(n.last(), 5);
        assert_eq!(n.spread(), 4);
        assert!(n.bytes() > 0);
    }

    /// A range in the middle of a large index is found by searching and not by
    /// walking, so the answer is the same however far in it is.
    #[test]
    fn a_range_is_found_wherever_it_sits() {
        let mut n = Nums::new();
        for id in 1..=10_000u32 {
            n.add(id, f64::from(id));
        }
        n.settle();
        assert_eq!(n.range(Ends::shut(5000.0, 5002.0)), [5000, 5001, 5002]);
        assert_eq!(n.range(Ends::shut(1.0, 1.0)), [1]);
        assert_eq!(n.range(Ends::shut(10_000.0, 20_000.0)), [10_000]);
        assert_eq!(n.range(Ends::all()).len(), 10_000);
    }

    /// The range a query asked for is the range the index is given, without the
    /// caller having to take it apart.
    #[test]
    fn a_query_range_is_a_range() {
        let n = built(&[1.0, 2.0, 3.0]);
        let asked = Range {
            field: b"n".as_slice().into(),
            min: 1.0,
            max: 3.0,
            min_open: true,
            max_open: false,
        };
        assert_eq!(n.range(Ends::from(&asked)), [2, 3]);
    }
}
