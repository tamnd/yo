//! The squared distance between two full precision vectors.
//!
//! Every other distance in this crate is a popcount over codes, and this is the
//! one place a float distance is still measured: ranking centroids to place a
//! vector, LIRE's sweep deciding whether a member has drifted, and the rerank
//! that settles the order of the candidates a scan produced. Half the time an
//! ingest spends is inside this function, so it lives on its own and is written
//! for the compiler rather than for the reader.
//!
//! # Why it is written this way
//!
//! The obvious loop indexes both slices by a counter bounded by one of the two
//! lengths. The compiler cannot prove the other index is in range, so it emits a
//! bounds check per element, and a bounds check in the middle of a loop body is
//! a branch the vectoriser will not cross.
//!
//! [`slice::as_chunks`] hands back a slice of fixed size arrays instead. Indexing
//! an eight element array by a constant between zero and seven needs no check at
//! all, so the body becomes eight independent multiply adds over two aligned
//! runs of floats and comes out as vector instructions on every target this
//! builds for, without a single intrinsic or a `target_feature` gate.
//!
//! It is the same lesson `rabitq::fixed_dot` records: a length the compiler
//! knows turns a loop into straight line code, and a length it has to read at
//! run time does not.

/// How many differences are accumulated side by side.
///
/// Eight floats is a 256 bit vector, which is one AVX2 register and two NEON
/// ones, and the eight partial sums are independent so the multiply adds do not
/// queue up behind each other the way one running total would.
const LANES: usize = 8;

/// The squared euclidean distance between `a` and `b`.
///
/// Squared, because nothing here wants the root: an ordering by distance is the
/// same ordering as by distance squared, and the root would be a division and a
/// square root per comparison to move every number to a different scale and no
/// further apart.
///
/// Vectors of different lengths are measured over the shorter of the two, which
/// is what the old form did by accident and no caller relies on, since every
/// caller here holds two vectors of the collection's dimension.
#[must_use]
pub(crate) fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (xs, x_tail) = a[..n].as_chunks::<LANES>();
    let (ys, _) = b[..n].as_chunks::<LANES>();

    let mut totals = [0.0f32; LANES];
    for (x, y) in xs.iter().zip(ys) {
        for k in 0..LANES {
            let d = x[k] - y[k];
            totals[k] += d * d;
        }
    }

    let mut sum = 0.0f32;
    for total in totals {
        sum += total;
    }
    for (x, y) in x_tail.iter().zip(&b[n - x_tail.len()..]) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// A float in nought to one, the same way `coarse` makes its test data.
    fn unit(rng: &mut Rng) -> f32 {
        (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// The straightforward one, which is what the answer has to match.
    fn plain(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Every length either side of the lane width, since the tail is the part
    /// the chunked form has to get right and the whole numbers of lanes are the
    /// part it is fast for.
    #[test]
    fn it_agrees_with_the_obvious_loop_at_every_length() {
        let mut rng = Rng::new(0x51D1);
        let a: Vec<f32> = (0..200).map(|_| unit(&mut rng) * 4.0 - 2.0).collect();
        let b: Vec<f32> = (0..200).map(|_| unit(&mut rng) * 4.0 - 2.0).collect();
        for n in 0..=200 {
            let want = plain(&a[..n], &b[..n]);
            let got = sqdist(&a[..n], &b[..n]);
            assert!(
                (got - want).abs() <= want.abs() * 1e-5 + 1e-6,
                "at {n} dimensions, {got} against {want}"
            );
        }
    }

    #[test]
    fn a_vector_is_no_distance_from_itself() {
        let mut rng = Rng::new(7);
        let a: Vec<f32> = (0..128).map(|_| unit(&mut rng)).collect();
        assert_eq!(sqdist(&a, &a), 0.0);
    }

    /// Nothing here relies on it, but the old form measured over the shorter of
    /// the two and a caller that hands in a short slice should not get a panic
    /// out of a change that was only supposed to make this faster.
    #[test]
    fn different_lengths_are_measured_over_the_shorter_one() {
        let a = [1.0f32; 20];
        let b = [0.0f32; 9];
        assert_eq!(sqdist(&a, &b), 9.0);
        assert_eq!(sqdist(&b, &a), 9.0);
    }
}
