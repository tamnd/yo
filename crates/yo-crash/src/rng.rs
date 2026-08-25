//! A random number generator you can write down.
//!
//! The whole harness is worthless if a failure cannot be reproduced. A trial
//! that fails prints its seed, and running that seed again has to produce the
//! same trial, byte for byte, on any machine and any target. That rules out
//! anything seeded from the clock, anything that consults the operating system,
//! and anything whose output depends on the width of a pointer.
//!
//! So this is `splitmix64`, which is nine lines and has no state beyond a
//! `u64`. It is not cryptographic and does not need to be. The property under
//! test is a durability invariant, and an adversary who can predict where the
//! next fault lands is welcome to.

/// A seeded stream of numbers, reproducible everywhere.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A stream from a seed.
    #[must_use]
    pub const fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// The seed a stream would need to be at this point again.
    ///
    /// Every trial takes one of these and prints it on failure, so a hundred
    /// thousand trial run that fails on trial 74,113 hands back a number that
    /// reproduces trial 74,113 on its own in a millisecond.
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// The next number.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `n`, or 0 when `n` is 0.
    ///
    /// Biased, by about one part in 2^64 divided by `n`. Everything this picks
    /// is smaller than a few thousand, so the bias is not measurable and a
    /// rejection loop would only add a way for the harness to hang.
    pub const fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// A number in `lo..=hi`.
    pub const fn between(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        lo + self.below(hi - lo + 1)
    }

    /// True with probability `num` in `den`.
    pub const fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den as usize) < num as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_is_the_same_stream() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_are_different_streams() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let same = (0..1000).filter(|_| a.next_u64() == b.next_u64()).count();
        assert_eq!(same, 0);
    }

    #[test]
    fn the_stream_does_not_settle_on_one_value() {
        // A generator with a bad constant can reach a fixed point and sit there,
        // and every trial after that is the same trial. That failure is silent
        // and it makes the whole run worthless, so it is worth one assertion.
        let mut r = Rng::new(0);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            seen.insert(r.next_u64());
        }
        assert_eq!(seen.len(), 10_000, "the stream repeats itself");
    }

    #[test]
    fn below_stays_below() {
        let mut r = Rng::new(99);
        for n in 1..50usize {
            for _ in 0..200 {
                assert!(r.below(n) < n);
            }
        }
        assert_eq!(r.below(0), 0, "no division by zero");
    }

    #[test]
    fn below_reaches_both_ends() {
        let mut r = Rng::new(7);
        let mut lo = false;
        let mut hi = false;
        for _ in 0..1000 {
            match r.below(8) {
                0 => lo = true,
                7 => hi = true,
                _ => {}
            }
        }
        assert!(
            lo && hi,
            "a generator that never picks an end is not uniform"
        );
    }

    #[test]
    fn between_covers_its_range_inclusive() {
        let mut r = Rng::new(4);
        let mut seen = [false; 5];
        for _ in 0..1000 {
            let v = r.between(2, 6);
            assert!((2..=6).contains(&v));
            seen[v - 2] = true;
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(r.between(5, 5), 5);
        assert_eq!(r.between(9, 3), 9, "a backwards range is its own low end");
    }

    #[test]
    fn chance_is_roughly_the_odds_it_says() {
        let mut r = Rng::new(31);
        let hits = (0..10_000).filter(|_| r.chance(1, 4)).count();
        assert!((2200..2800).contains(&hits), "got {hits} in 10000");
    }
}
