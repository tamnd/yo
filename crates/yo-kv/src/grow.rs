//! How the arrays underneath a collection get bigger.
//!
//! `Vec` doubles, and doubling is the right policy while an array is small and
//! the wrong one once it is large. The cost of being wrong is not subtle. A
//! sorted set of six hundred thousand members holds a row array with room for a
//! million and change, because a million and change is the next power of two,
//! and the four hundred thousand rows nobody asked for are ten megabytes of
//! nothing. The name blob under it does the same thing on top of that. Measured
//! on a six hundred thousand member sorted set with sixteen byte members, the
//! slack was thirty of the fifty six bytes an element cost, which is more than
//! everything else in the structure put together.
//!
//! So this doubles under a threshold and grows by a quarter over it. Under the
//! threshold the slack is a handful of kilobytes however wrong the policy is,
//! and the copies are what matter. Over it the copies are amortised either way
//! and the slack is megabytes, so the slack is what matters.
//!
//! # What the quarter costs
//!
//! An element is copied about five times over the life of an array that grows by
//! a quarter, against about twice for one that doubles. Filling a twenty five
//! megabyte row array moves a hundred and twenty megabytes instead of fifty,
//! which is a few milliseconds per million inserts on any machine worth running
//! on, and it buys back six megabytes on average and twelve at the worst point.
//! Y14 says a row that wins on throughput and loses on memory is a fail, so this
//! is the direction that trade goes.
//!
//! # Why not ask for exactly what is needed
//!
//! Because then every insert is a reallocation and a copy, which is quadratic.
//! A growth factor over one is what makes an append amortised constant, and the
//! only question is which one. A quarter is the smallest factor where the
//! constant is still small enough to not show up in a benchmark.

use core::mem::size_of;

/// Where doubling stops paying for itself, in bytes of allocation.
///
/// Under this an array's worst case slack is sixty four kilobytes, which is not
/// worth a single extra memcpy to avoid. A server holding a million small
/// collections never reaches it, so the small case keeps `Vec`'s policy exactly.
const DOUBLE_UNDER: usize = 64 * 1024;

/// The next capacity for an array that has `cap` and needs at least `want`.
///
/// Public and separate from [`reserve`] so that it can be tested without an
/// allocation, and so that a structure holding its bytes some other way than in
/// a `Vec` can use the same policy.
#[must_use]
pub fn next_capacity(cap: usize, want: usize, elem: usize) -> usize {
    if want <= cap {
        return cap;
    }
    let grown = if cap.saturating_mul(elem) < DOUBLE_UNDER {
        cap.saturating_mul(2)
    } else {
        // A quarter more, and never fewer than one more, so that an array whose
        // capacity is under four still moves.
        cap.saturating_add((cap / 4).max(1))
    };
    grown.max(want)
}

/// Make room for `extra` more elements, growing by the policy above.
///
/// [`Vec::reserve_exact`] rather than [`Vec::reserve`], because the point is to
/// take the size this decided and not the size `Vec` would have decided.
pub fn reserve<T>(v: &mut Vec<T>, extra: usize) {
    let want = v.len() + extra;
    if want <= v.capacity() {
        return;
    }
    let next = next_capacity(v.capacity(), want, size_of::<T>());
    v.reserve_exact(next - v.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_arrays_double_and_large_ones_grow_by_a_quarter() {
        // A row is twenty four bytes, so the threshold is somewhere near two
        // thousand seven hundred rows.
        assert_eq!(next_capacity(0, 1, 24), 1);
        assert_eq!(next_capacity(4, 5, 24), 8);
        assert_eq!(next_capacity(1024, 1025, 24), 2048);
        // Past sixty four kilobytes, a quarter.
        assert_eq!(next_capacity(4096, 4097, 24), 5120);
        assert_eq!(next_capacity(1_000_000, 1_000_001, 24), 1_250_000);
        // A caller asking for more than the policy would give gets what it
        // asked for, which is what `with_capacity` on a known size wants.
        assert_eq!(next_capacity(4096, 100_000, 24), 100_000);
        // Nothing to do.
        assert_eq!(next_capacity(16, 16, 24), 16);
        assert_eq!(next_capacity(16, 0, 24), 16);
    }

    #[test]
    fn a_reserve_takes_the_size_the_policy_chose() {
        let mut v: Vec<u64> = Vec::new();
        for _ in 0..200_000 {
            reserve(&mut v, 1);
            v.push(0);
        }
        // Eight byte elements, so the threshold is eight thousand of them, and
        // everything after that grew by a quarter. The worst the slack can be
        // is a quarter and it is nowhere near a double.
        assert!(
            v.capacity() < v.len() + v.len() / 4 + 8,
            "held room for {} to store {}",
            v.capacity(),
            v.len()
        );
    }
}
