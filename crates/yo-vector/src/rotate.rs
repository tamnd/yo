//! The random rotation every vector goes through before it is quantised
//! (`10` section 3).
//!
//! RaBitQ quantises a coordinate to its sign, and a sign carries information
//! only when the coordinates are all about the same size. Real embeddings are
//! not like that: a handful of dimensions hold most of the energy, and the sign
//! of the rest is close to a coin toss. A random rotation fixes it, because
//! rotating a vector by a random orthogonal matrix spreads its length evenly
//! over the coordinates while leaving every distance and every angle exactly
//! where it was. That is the whole reason the estimator's error bound holds.
//!
//! ```
//! use yo_vector::Rotation;
//!
//! let r = Rotation::new(8, 42);
//! let mut v = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! r.apply(&mut v);
//!
//! // The length did not move, and neither did anything else about the vector.
//! let len: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
//! assert!((len - 1.0).abs() < 1e-5);
//! // The one coordinate that held everything now holds a share of it.
//! assert!(v.iter().all(|x| x.abs() < 0.95));
//! ```
//!
//! # Why this shape and not a matrix
//!
//! The obvious rotation is a dense `D` by `D` orthogonal matrix from a QR
//! decomposition, and it costs `D^2` multiplies per vector. At 768 dimensions
//! that is 590 thousand multiplies to insert one vector, and the ingest target
//! is fifty thousand vectors a second on one core, which asks for 29 GFLOP/s of
//! nothing but rotation. It does not fit, so the rotation has to be structured.
//!
//! The usual structured answer is a random sign flip followed by a Hadamard
//! transform, which is `D log D` and lovely, and needs `D` to be a power of two.
//! Padding 768 up to 1024 would make a one bit code 128 bytes instead of 96,
//! which is a third more index for every vector in the collection, so that is
//! not free either.
//!
//! What is here is the same idea without the power of two. A round flips the
//! sign of a random half of the coordinates, pairs them all up at random, and
//! replaces each pair with its sum and its difference over the square root of
//! two. Every step of that is orthogonal by construction rather than
//! approximately orthogonal after rounding, and a quarter turn on a pair splits
//! whatever it was holding evenly between the two, which is the part that
//! actually spreads a spike out. Pairing at random rather than by a fixed stride
//! is what lets any coordinate reach any other, so `log2(D)` rounds and a couple
//! more is enough.
//!
//! The angle is not random, and that is deliberate. A round with a random angle
//! keeps roughly nine tenths of what the larger side was holding, so a spike
//! only decays like `0.9^rounds` and it takes something like sixty rounds at 256
//! dimensions to flatten. A quarter turn halves it every time and takes eight.
//! The randomness the estimator needs comes from the pairing and the signs, and
//! there is plenty of it.
//!
//! # It is not written down anywhere
//!
//! A rotation is `dim` and a seed, and both live in the collection's catalogue
//! entry. Rebuilding it is deterministic on every machine and every target,
//! because the generator underneath is, so the file never holds the tables and
//! two processes that open the same collection compute the same rotation.

use yo_common::Rng;

/// A quarter turn on a pair is a sum and a difference, both over this.
const INV_ROOT2: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// One sweep: a sign for every coordinate and a pairing of all of them.
///
/// The pairs are stored rather than the permutation they came from, because
/// rotating in place off a pair list is one pass over the vector and no scratch
/// buffer, where a permutation would want somewhere to write the shuffled copy.
#[derive(Debug)]
struct Round {
    /// The sign mask for every coordinate, either nothing or the top bit, ready
    /// to be exclusive ored straight into the float.
    ///
    /// This used to be one bit per coordinate, unpacked with a shift and a mask
    /// inside the loop, which is four integer operations per float and stops
    /// the loop being one wide exclusive or over a run of them. A word each is
    /// `dim` times `rounds` times four bytes for the whole collection, a few
    /// kilobytes, and no vector anywhere pays for it twice.
    signs: Vec<u32>,
    /// `(i, j)` for each pair, which together cover every index once, or every
    /// index but one when the dimension is odd.
    pairs: Vec<(u32, u32)>,
}

/// A random orthogonal transform, rebuilt from its seed rather than stored.
#[derive(Debug)]
pub struct Rotation {
    dim: usize,
    seed: u64,
    rounds: Vec<Round>,
}

impl Rotation {
    /// The rotation a collection of `dim` dimensional vectors uses, from the
    /// seed in its catalogue entry.
    ///
    /// # Panics
    ///
    /// If `dim` is zero. A collection of vectors with no coordinates is a
    /// mistake made somewhere further up rather than a case to handle.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Rotation {
        assert!(dim > 0, "a vector has at least one dimension");
        let mut rng = Rng::new(seed);
        let sweeps = sweeps(dim);
        let mut rounds = Vec::with_capacity(sweeps);
        // The identity, shuffled fresh for every round, which is what makes the
        // pairing different each time and so lets a coordinate reach across the
        // whole vector rather than staying in the half it started in.
        let mut order: Vec<u32> = (0..dim as u32).collect();
        for _ in 0..sweeps {
            shuffle(&mut order, &mut rng);
            let half = dim / 2;
            let pairs: Vec<(u32, u32)> = (0..half)
                .map(|k| (order[2 * k], order[2 * k + 1]))
                .collect();
            // Drawn in this order because that is the order they were drawn in
            // when they were two separate tables, and the rotation a seed gives
            // has to stay the one it always gave.
            let flip = bits(dim, &mut rng);
            let turn = bits(half, &mut rng);
            let mut signs: Vec<u32> = (0..dim)
                .map(|i| u32::from((flip[i / 64] >> (i % 64)) & 1 == 1) << 31)
                .collect();
            // Turning a pair the other way round is the same thing as flipping
            // the sign of its second coordinate first. With b negated the sum
            // becomes the difference and the difference becomes the sum, and
            // both of those are exact in floating point, so folding the turn
            // into the sign here gives bit for bit what the branch inside the
            // pair loop used to give and the loop no longer has a coin toss to
            // mispredict in it.
            for (k, &(_, j)) in pairs.iter().enumerate() {
                if (turn[k / 64] >> (k % 64)) & 1 == 1 {
                    signs[j as usize] ^= 1 << 31;
                }
            }
            rounds.push(Round { signs, pairs });
        }
        Rotation { dim, seed, rounds }
    }

    /// How many coordinates a vector this rotates has.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The seed this was built from, which is what a catalogue stores.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Rotate a vector where it lies.
    ///
    /// # Panics
    ///
    /// If `v` is not [`Rotation::dim`] long.
    pub fn apply(&self, v: &mut [f32]) {
        assert_eq!(
            v.len(),
            self.dim,
            "this rotation is for {} dimensions and was handed {}",
            self.dim,
            v.len()
        );
        for round in &self.rounds {
            // A sign is the top bit of the float, so flipping one is an xor and
            // there is no branch to mispredict on a coin toss. Zipping two
            // slices of the same length rather than indexing one by a counter
            // is what lets the compiler drop the bounds check and do a whole
            // register of them at once.
            for (c, &sign) in v.iter_mut().zip(&round.signs) {
                *c = f32::from_bits(c.to_bits() ^ sign);
            }
            for &(i, j) in &round.pairs {
                let (i, j) = (i as usize, j as usize);
                let (a, b) = (v[i], v[j]);
                v[i] = (a + b) * INV_ROOT2;
                v[j] = (a - b) * INV_ROOT2;
            }
        }
    }
}

/// How many sweeps it takes for a spike to be spread over the whole vector.
///
/// A round splits whatever a coordinate is holding between two, so the set one
/// can have reached after `k` of them is `2^k` wide. `log2(dim)` rounded up is
/// where that covers the vector, and the two on top are because the pairings are
/// drawn independently and so overlap near the end rather than tiling neatly.
fn sweeps(dim: usize) -> usize {
    ((usize::BITS - (dim - 1).leading_zeros()) as usize + 2).max(4)
}

/// Fisher and Yates, so that every pairing is equally likely.
fn shuffle(order: &mut [u32], rng: &mut Rng) {
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
}

/// `n` coin tosses, packed.
fn bits(n: usize, rng: &mut Rng) -> Vec<u64> {
    (0..n.div_ceil(64)).map(|_| rng.next_u64()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// A number in `[0, 1)` from the generator, for the tests that want one.
    fn unit(rng: &mut Rng) -> f32 {
        (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// A handful of vectors that are not all alike.
    fn sample(dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| unit(&mut rng) * 2.0 - 1.0).collect())
            .collect()
    }

    #[test]
    fn a_rotation_keeps_every_length_and_every_angle() {
        let r = Rotation::new(64, 7);
        let vs = sample(64, 8, 11);
        for v in &vs {
            let mut spun = v.clone();
            r.apply(&mut spun);
            let before = dot(v, v).sqrt();
            let after = dot(&spun, &spun).sqrt();
            assert!((before - after).abs() < 1e-4, "{before} became {after}");
        }
        // And the angle between any two of them, which is what a distance is
        // made of and so is the property that actually has to survive.
        for a in 0..vs.len() {
            for b in 0..a {
                let mut x = vs[a].clone();
                let mut y = vs[b].clone();
                let before = dot(&x, &y);
                r.apply(&mut x);
                r.apply(&mut y);
                let after = dot(&x, &y);
                assert!((before - after).abs() < 1e-3, "{before} became {after}");
            }
        }
    }

    /// The rotation is not written into the file, it is rebuilt from a seed, so
    /// a build that computes a different one from the same seed cannot read a
    /// collection an older build wrote. Nothing else in the crate would notice:
    /// the codes would still be self consistent and recall would still look
    /// fine, and only the vectors already on disk would be wrong. So the bits
    /// are pinned here.
    ///
    /// If this fails and the change was deliberate, the file format version is
    /// the thing that has to move, not this number.
    #[test]
    fn a_seed_gives_the_rotation_it_has_always_given() {
        for (dim, want) in [
            (8usize, 0xef2f_9c0c_1aad_5cdfu64),
            (33, 0x97e7_79cf_5820_4a3f),
            (128, 0xde6f_cf6b_69cf_490c),
        ] {
            let mut v: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / 8.0).collect();
            Rotation::new(dim, 0xB0A7).apply(&mut v);
            let mut got: u64 = 0xcbf2_9ce4_8422_2325;
            for c in &v {
                for byte in c.to_bits().to_le_bytes() {
                    got ^= u64::from(byte);
                    got = got.wrapping_mul(0x0100_0000_01b3);
                }
            }
            assert_eq!(got, want, "the rotation at {dim} dimensions has moved");
        }
    }

    #[test]
    fn the_same_seed_is_the_same_rotation() {
        let v = sample(32, 1, 3).pop().expect("one vector");
        let mut a = v.clone();
        let mut b = v;
        Rotation::new(32, 99).apply(&mut a);
        Rotation::new(32, 99).apply(&mut b);
        assert_eq!(a, b);

        let mut c = a.clone();
        Rotation::new(32, 100).apply(&mut c);
        assert_ne!(a, c, "two seeds should not be one rotation");
    }

    /// The point of the whole thing: a vector whose length sits in one
    /// coordinate comes out with it spread over all of them, which is what
    /// makes the sign of a coordinate worth a bit.
    #[test]
    fn a_spike_comes_out_flat() {
        for dim in [64usize, 256, 768] {
            let r = Rotation::new(dim, 5);
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            r.apply(&mut v);

            let even = 1.0 / (dim as f32).sqrt();
            let biggest = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(biggest < even * 5.0, "{dim}: {biggest} against {even}");
            // And nothing is left sitting at zero either, which is the failure
            // a fixed pairing would have: half the vector never touched.
            let alive = v.iter().filter(|x| x.abs() > even / 4.0).count();
            assert!(alive > dim * 3 / 4, "{dim}: only {alive} coordinates moved");
        }
    }

    #[test]
    fn an_odd_dimension_still_rotates() {
        let r = Rotation::new(7, 1);
        let mut v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let before = dot(&v, &v).sqrt();
        r.apply(&mut v);
        let after = dot(&v, &v).sqrt();
        assert!((before - after).abs() < 1e-3);
    }

    #[test]
    fn one_dimension_is_only_a_sign() {
        // There is nothing to pair it with, so all a round can do is flip it,
        // and the length still comes out where it went in.
        let r = Rotation::new(1, 1);
        let mut v = [3.0f32];
        r.apply(&mut v);
        assert_eq!(v[0].abs(), 3.0);
    }
}
