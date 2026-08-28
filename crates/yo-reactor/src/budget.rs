//! What the maintenance slice is allowed to spend.
//!
//! `04` section 6 asks for a fixed instruction budget rather than a wall clock
//! budget, and the difference matters. A wall clock budget reads a timer,
//! which is a syscall or at best a serialising instruction, and it gives a
//! different amount of work on a busy machine than on an idle one, so the tail
//! latency it produces is not reproducible. A unit budget is a subtraction, it
//! is the same on every machine, and a run that overshoots does so by one item
//! rather than by however long that item took.
//!
//! A unit is whatever the caller says it is. The engine picks a cost per item
//! it does, keeps the scale consistent between the things it does, and the
//! budget only has to be monotone: more units means more work.

/// Units a maintenance slice gets by default, per turn of the loop.
///
/// Sized so that the slice is a small fraction of a full batch of commands
/// rather than a competitor to it. Expiry sampling is the usual spender and it
/// walks twenty keys at a time, so this is tens of samples in the worst case
/// and nothing at all in the common one, where there is no work waiting.
pub const MAINTENANCE_UNITS: u32 = 4096;

/// A slice's remaining allowance.
///
/// The contract is one call: [`Budget::spend`] returns false when the caller
/// should stop. It is not an error and it is not something to report. A
/// maintenance pass that runs out of budget has done part of its work and will
/// be back on the next turn, which is a hundred nanoseconds away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    left: u32,
    spent: u32,
}

impl Budget {
    /// A budget of `units`.
    #[must_use]
    pub const fn new(units: u32) -> Budget {
        Budget {
            left: units,
            spent: 0,
        }
    }

    /// A budget of [`MAINTENANCE_UNITS`].
    #[must_use]
    pub const fn standard() -> Budget {
        Budget::new(MAINTENANCE_UNITS)
    }

    /// A budget of nothing, which is what a turn hands to an engine that has
    /// asked not to be given a slice.
    #[must_use]
    pub const fn none() -> Budget {
        Budget::new(0)
    }

    /// Charge `units` and say whether there is anything left to do after it.
    ///
    /// The charge always goes through, even when it takes the budget past the
    /// end. Refusing it would mean the caller has to ask before every item and
    /// then do the item anyway, which is two branches for the same answer.
    #[inline]
    pub const fn spend(&mut self, units: u32) -> bool {
        self.spent = self.spent.saturating_add(units);
        self.left = self.left.saturating_sub(units);
        self.left > 0
    }

    /// What is left.
    #[must_use]
    #[inline]
    pub const fn left(&self) -> u32 {
        self.left
    }

    /// What has gone, which is what a counter reports rather than what the
    /// slice decides on.
    #[must_use]
    #[inline]
    pub const fn spent(&self) -> u32 {
        self.spent
    }

    /// Whether there is any allowance at all.
    #[must_use]
    #[inline]
    pub const fn is_spent(&self) -> bool {
        self.left == 0
    }
}

impl Default for Budget {
    fn default() -> Budget {
        Budget::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_runs_out_and_says_so() {
        let mut b = Budget::new(10);
        assert!(b.spend(4));
        assert!(b.spend(5));
        assert!(!b.spend(1), "the tenth unit is the last one");
        assert!(b.is_spent());
        assert_eq!(b.spent(), 10);
    }

    #[test]
    fn overspending_is_allowed_and_recorded() {
        let mut b = Budget::new(10);
        assert!(
            !b.spend(1000),
            "one item can cost more than the whole slice"
        );
        assert_eq!(b.left(), 0);
        assert_eq!(b.spent(), 1000, "what it cost, not what it was allowed");
    }

    #[test]
    fn an_empty_budget_stops_before_the_first_item() {
        let mut b = Budget::none();
        assert!(b.is_spent());
        assert!(!b.spend(1));
    }

    #[test]
    fn spending_cannot_wrap() {
        let mut b = Budget::new(u32::MAX);
        assert!(b.spend(u32::MAX - 1));
        assert!(!b.spend(u32::MAX));
        assert_eq!(b.left(), 0);
        assert_eq!(b.spent(), u32::MAX);
    }
}
