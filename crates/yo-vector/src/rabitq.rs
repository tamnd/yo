//! RaBitQ, the quantiser the searchable form of a vector is written in
//! (`10` section 3).
//!
//! A 768 dimensional embedding is 3072 bytes of `f32`, and ten million of them
//! is 30 GB. RaBitQ writes the same vector as one bit per dimension, which is 96
//! bytes and a 32x reduction, and that is the difference between an index that
//! sits in memory and one that does not.
//!
//! Binary quantisation on its own is old and it is not good enough: keeping only
//! the sign of each coordinate throws away how far the point is from the
//! boundary, and the distances that come back are biased in a way that no amount
//! of rerank hides. RaBitQ's contribution is the estimator. It stores one extra
//! number per vector, the cosine between the vector and the corner of the cube
//! it was rounded to, and dividing by that turns a biased guess into an unbiased
//! one with an error bound that shrinks as `1/sqrt(D)`. The ordering that comes
//! out of the codes alone is then good enough that rerank only has to look at a
//! small multiple of `k` real vectors.
//!
//! ```
//! use yo_vector::{Bits, Quantizer};
//!
//! let q = Quantizer::new(64, Bits::One, 7);
//! let centroid = vec![0.0f32; 64];
//! let mut code = vec![0u8; q.code_bytes()];
//!
//! let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
//! let coded = q.encode(&v, &centroid, &mut code);
//!
//! // The query is prepared once and then measured against many codes.
//! let query = q.query(&v, &centroid);
//! let guess = query.distance(&code, &coded);
//! // A vector against itself, so the answer should be near zero.
//! assert!(guess < 0.2, "{guess}");
//! ```
//!
//! # What is stored
//!
//! Per vector: the code, and two `f32`. The first is the length of the residual,
//! which is the vector minus its partition's centroid, and it is what turns an
//! angle in the unit sphere back into a distance. The second is the correction,
//! which is the estimator. `10` section 3's table says one `f32` and it is two,
//! because both are needed and neither can be recovered from the other. Eight
//! bytes on top of 96 is still a 30x reduction rather than 32x.
//!
//! # One bit and four
//!
//! [`Bits::One`] rounds each coordinate to a sign and [`Bits::Four`] rounds it
//! to one of sixteen levels between the smallest and the largest coordinate the
//! vector has. Four bits is four times the index and roughly a quarter of the
//! error, and which one a collection wants is a question about the embedding
//! family rather than about the engine, which is why both are here and the
//! choice is per collection.
//!
//! The two share one code path. Reconstructing a code is `lo + level * delta`
//! either way, with `lo` and `delta` fixed by the dimension for one bit and
//! measured per vector for four, so the estimator is written once and the only
//! thing that changes is how many bits a level takes.

use crate::rotate::Rotation;

/// How many bits a coordinate is rounded to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bits {
    /// The sign, which is 96 bytes for a 768 dimensional vector.
    One,
    /// Sixteen levels, which is 384 bytes for the same vector and about a
    /// quarter of the error.
    Four,
}

impl Bits {
    /// How many bits one coordinate takes.
    #[must_use]
    pub fn count(self) -> usize {
        match self {
            Bits::One => 1,
            Bits::Four => 4,
        }
    }
}

/// What a code needs alongside it to be measured against a query.
///
/// `lo` and `delta` are only worth storing for [`Bits::Four`]. At one bit they
/// are the same two numbers for every vector in the collection, so a store that
/// keeps them per vector is keeping the dimension eight bytes at a time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coded {
    /// The length of the vector minus its centroid.
    pub norm: f32,
    /// The cosine between the vector and what its code reconstructs to, which
    /// is the estimator's correction and the whole of RaBitQ over plain binary
    /// quantisation.
    pub correction: f32,
    /// The value level zero reconstructs to, over the length of the
    /// reconstruction.
    pub lo: f32,
    /// What one level is worth, over the length of the reconstruction.
    pub delta: f32,
}

/// The quantiser for one collection: its rotation and its width.
pub struct Quantizer {
    rot: Rotation,
    bits: Bits,
}

impl Quantizer {
    /// The quantiser for `dim` dimensional vectors at `bits`, with `seed`
    /// choosing the rotation.
    ///
    /// # Panics
    ///
    /// If `dim` is zero.
    #[must_use]
    pub fn new(dim: usize, bits: Bits, seed: u64) -> Quantizer {
        Quantizer {
            rot: Rotation::new(dim, seed),
            bits,
        }
    }

    /// How many coordinates a vector has.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rot.dim()
    }

    /// How wide a coordinate is written.
    #[must_use]
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// The seed the rotation was built from, which is what a catalogue stores.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.rot.seed()
    }

    /// How many bytes one code takes.
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.dim().div_ceil(8) * self.bits.count()
    }

    /// Write `v`'s code against the centroid of the partition it is going into.
    ///
    /// # Panics
    ///
    /// If `v` or `centroid` is not [`Quantizer::dim`] long, or `code` is not
    /// [`Quantizer::code_bytes`] long.
    pub fn encode(&self, v: &[f32], centroid: &[f32], code: &mut [u8]) -> Coded {
        let mut x = self.residual(v, centroid);
        let norm = length(&x);
        code.fill(0);
        if norm == 0.0 {
            // The vector is the centroid. There is no direction to write down,
            // and a correction of one keeps the estimator from dividing by
            // zero if anyone measures against it anyway.
            return Coded {
                norm: 0.0,
                correction: 1.0,
                lo: 0.0,
                delta: 0.0,
            };
        }
        for c in &mut x {
            *c /= norm;
        }
        self.rot.apply(&mut x);
        let mut coded = match self.bits {
            Bits::One => sign_code(&x, code),
            Bits::Four => level_code(&x, code),
        };
        coded.norm = norm;
        coded
    }

    /// Prepare a query against the centroid of a partition being scanned.
    ///
    /// This is the per partition half of a search and it happens once, where
    /// the estimate against a code happens once per vector in the partition.
    ///
    /// # Panics
    ///
    /// If `q` or `centroid` is not [`Quantizer::dim`] long.
    #[must_use]
    pub fn query(&self, q: &[f32], centroid: &[f32]) -> Query {
        let mut x = self.residual(q, centroid);
        let norm = length(&x);
        if norm > 0.0 {
            for c in &mut x {
                *c /= norm;
            }
        }
        self.rot.apply(&mut x);
        let sum = x.iter().sum();
        Query {
            bits: self.bits,
            rotated: x,
            sum,
            norm,
        }
    }

    /// `v - centroid`.
    fn residual(&self, v: &[f32], centroid: &[f32]) -> Vec<f32> {
        assert_eq!(
            v.len(),
            self.dim(),
            "this collection holds {} dimensional vectors and was handed {}",
            self.dim(),
            v.len()
        );
        assert_eq!(
            centroid.len(),
            self.dim(),
            "the centroid is {} dimensional and the collection is {}",
            centroid.len(),
            self.dim()
        );
        v.iter().zip(centroid).map(|(a, b)| a - b).collect()
    }
}

/// A query, rotated once and then measured against every code in a partition.
pub struct Query {
    bits: Bits,
    /// The query's residual, unit length and rotated.
    rotated: Vec<f32>,
    /// The sum of the rotated coordinates, which the estimator needs and which
    /// does not depend on the code it is being compared against.
    sum: f32,
    /// The length of the query's residual.
    norm: f32,
}

impl Query {
    /// The estimated squared distance between the query and a coded vector.
    ///
    /// Squared rather than the distance itself because the square root is
    /// monotone, so it changes no ordering and nothing above this needs it.
    ///
    /// # Panics
    ///
    /// If `code` is not as long as the codes this query's quantiser writes.
    #[must_use]
    pub fn distance(&self, code: &[u8], coded: &Coded) -> f32 {
        let cos = self.cosine(code, coded);
        // The law of cosines on the triangle the centroid makes with the two
        // points, which is why the residual lengths had to be kept.
        (self.norm * self.norm + coded.norm * coded.norm - 2.0 * self.norm * coded.norm * cos)
            .max(0.0)
    }

    /// The estimated cosine between the query's residual and the coded one.
    ///
    /// This is RaBitQ's estimator: the inner product against what the code
    /// reconstructs to, divided by the cosine between that reconstruction and
    /// the vector it came from. The division is the part that makes it unbiased.
    #[must_use]
    pub fn cosine(&self, code: &[u8], coded: &Coded) -> f32 {
        if coded.norm == 0.0 || self.norm == 0.0 {
            return 0.0;
        }
        let levels = match self.bits {
            Bits::One => sign_dot(code, &self.rotated),
            Bits::Four => level_dot(code, &self.rotated),
        };
        (coded.lo * self.sum + coded.delta * levels) / coded.correction
    }
}

/// The sign of each coordinate, packed most significant bit first.
///
/// The reconstruction is `(2 * bit - 1) / sqrt(D)`, which is a unit vector
/// pointing at one corner of the cube, so `lo` and `delta` fall out of the
/// dimension and the correction is the sum of the absolute coordinates over
/// `sqrt(D)`.
fn sign_code(x: &[f32], code: &mut [u8]) -> Coded {
    let mut abs = 0.0f32;
    for (i, &c) in x.iter().enumerate() {
        abs += c.abs();
        if c >= 0.0 {
            code[i / 8] |= 0x80 >> (i % 8);
        }
    }
    let root = (x.len() as f32).sqrt();
    Coded {
        norm: 0.0,
        correction: abs / root,
        lo: -1.0 / root,
        delta: 2.0 / root,
    }
}

/// Sixteen levels between the smallest and the largest coordinate, packed two
/// to a byte with the earlier coordinate in the high nibble.
fn level_code(x: &[f32], code: &mut [u8]) -> Coded {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &c in x {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    // A vector whose coordinates are all the same value has no range to divide
    // up. It cannot happen after a rotation of a non zero residual, and a
    // division by zero here would be a silent NaN rather than a loud one.
    let delta = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
    let mut recon = 0.0f32;
    let mut dot = 0.0f32;
    for (i, &c) in x.iter().enumerate() {
        let level = (((c - lo) / delta).round() as i32).clamp(0, 15) as u8;
        if i % 2 == 0 {
            code[i / 2] |= level << 4;
        } else {
            code[i / 2] |= level;
        }
        let back = lo + f32::from(level) * delta;
        recon += back * back;
        dot += back * c;
    }
    let len = recon.sqrt();
    Coded {
        norm: 0.0,
        correction: dot / len,
        lo: lo / len,
        delta: delta / len,
    }
}

/// The sum of the query's coordinates where the code has a bit set.
fn sign_dot(code: &[u8], q: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &qi) in q.iter().enumerate() {
        if code[i / 8] & (0x80 >> (i % 8)) != 0 {
            sum += qi;
        }
    }
    sum
}

/// The query's coordinates weighted by the level each one was rounded to.
fn level_dot(code: &[u8], q: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &qi) in q.iter().enumerate() {
        let byte = code[i / 2];
        let level = if i % 2 == 0 { byte >> 4 } else { byte & 0x0f };
        sum += f32::from(level) * qi;
    }
    sum
}

fn length(v: &[f32]) -> f32 {
    v.iter().map(|c| c * c).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// Vectors that look a little like embeddings: not uniform, with a few
    /// coordinates carrying more than their share, which is the case a rotation
    /// is there to handle.
    fn corpus(dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|i| {
                        let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
                        let heavy = if i < dim / 16 { 6.0 } else { 1.0 };
                        (u * 2.0 - 1.0) * heavy
                    })
                    .collect();
                let len = length(&v);
                for c in &mut v {
                    *c /= len;
                }
                v
            })
            .collect()
    }

    fn mean(vs: &[Vec<f32>]) -> Vec<f32> {
        let dim = vs[0].len();
        let mut c = vec![0.0f32; dim];
        for v in vs {
            for (a, b) in c.iter_mut().zip(v) {
                *a += b;
            }
        }
        for a in &mut c {
            *a /= vs.len() as f32;
        }
        c
    }

    fn exact(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Encode a corpus and return the codes with what goes beside them.
    fn encode_all(q: &Quantizer, vs: &[Vec<f32>], c: &[f32]) -> (Vec<u8>, Vec<Coded>) {
        let width = q.code_bytes();
        let mut codes = vec![0u8; width * vs.len()];
        let mut meta = Vec::with_capacity(vs.len());
        for (i, v) in vs.iter().enumerate() {
            meta.push(q.encode(v, c, &mut codes[i * width..(i + 1) * width]));
        }
        (codes, meta)
    }

    /// How often the true ten nearest are inside the `keep` best the codes
    /// picked, which is the number that decides whether rerank has anything to
    /// work with.
    fn recall(bits: Bits, dim: usize, keep: usize) -> f32 {
        let vs = corpus(dim, 800, 1);
        let qs = corpus(dim, 40, 2);
        let c = mean(&vs);
        let q = Quantizer::new(dim, bits, 99);
        let (codes, meta) = encode_all(&q, &vs, &c);
        let width = q.code_bytes();

        let mut hits = 0usize;
        for query in &qs {
            let mut truth: Vec<(usize, f32)> = vs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, exact(query, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: Vec<usize> = truth[..10].iter().map(|(i, _)| *i).collect();

            let prepared = q.query(query, &c);
            let mut guess: Vec<(usize, f32)> = (0..vs.len())
                .map(|i| {
                    let code = &codes[i * width..(i + 1) * width];
                    (i, prepared.distance(code, &meta[i]))
                })
                .collect();
            guess.sort_by(|a, b| a.1.total_cmp(&b.1));
            let got: Vec<usize> = guess[..keep].iter().map(|(i, _)| *i).collect();
            hits += want.iter().filter(|i| got.contains(i)).count();
        }
        hits as f32 / (qs.len() * 10) as f32
    }

    /// Not a test, a table. Run it with
    /// `cargo test -p yo-vector --release -- --ignored --nocapture recall_table`
    /// when the estimator changes, because the two recall tests below only
    /// assert a floor and this is where the floor came from.
    #[test]
    #[ignore = "prints a table rather than asserting anything"]
    fn recall_table() {
        for (bits, name) in [(Bits::One, "1 bit"), (Bits::Four, "4 bit")] {
            for dim in [128usize, 256, 768] {
                for keep in [10usize, 20, 40, 100] {
                    println!(
                        "{name} dim {dim} keep {keep}: {:.3}",
                        recall(bits, dim, keep)
                    );
                }
            }
        }
    }

    #[test]
    fn a_code_is_the_width_it_says_it_is() {
        assert_eq!(Quantizer::new(768, Bits::One, 1).code_bytes(), 96);
        assert_eq!(Quantizer::new(768, Bits::Four, 1).code_bytes(), 384);
        // A dimension that is not a multiple of eight rounds up, because a bit
        // cannot be half in a byte.
        assert_eq!(Quantizer::new(100, Bits::One, 1).code_bytes(), 13);
    }

    #[test]
    fn one_bit_finds_the_true_neighbours_inside_a_short_rerank() {
        // Forty candidates for ten answers, which is the 4k rerank the search
        // path defaults to.
        let r = recall(Bits::One, 256, 40);
        assert!(r >= 0.95, "recall at 10 was {r}");
    }

    #[test]
    fn four_bits_is_better_than_one() {
        // Twenty candidates for ten answers, which one bit does not manage and
        // four does, so this measures the difference rather than asserting it.
        let one = recall(Bits::One, 128, 20);
        let four = recall(Bits::Four, 128, 20);
        assert!(four > one, "one bit got {one} and four bits got {four}");
        assert!(four >= 0.95, "four bit recall at 10 was {four}");
    }

    #[test]
    fn the_estimate_is_close_to_the_truth_rather_than_merely_ordered() {
        let dim = 256;
        let vs = corpus(dim, 200, 5);
        let qs = corpus(dim, 20, 6);
        let c = mean(&vs);
        let q = Quantizer::new(dim, Bits::One, 3);
        let (codes, meta) = encode_all(&q, &vs, &c);
        let width = q.code_bytes();

        let mut worst = 0.0f32;
        let mut bias = 0.0f32;
        let mut n = 0usize;
        for query in &qs {
            let prepared = q.query(query, &c);
            for (i, v) in vs.iter().enumerate() {
                let truth = exact(query, v);
                let guess = prepared.distance(&codes[i * width..(i + 1) * width], &meta[i]);
                let err = (guess - truth) / truth;
                worst = worst.max(err.abs());
                bias += err;
                n += 1;
            }
        }
        let bias = bias / n as f32;
        // Unbiased is the claim, so the average error should sit near zero
        // rather than merely being small in absolute value.
        assert!(
            bias.abs() < 0.02,
            "the estimate is off by {bias} on average"
        );
        assert!(worst < 0.5, "the worst estimate was off by {worst}");
    }

    #[test]
    fn a_vector_sitting_on_its_centroid_is_not_a_division_by_zero() {
        let q = Quantizer::new(16, Bits::One, 1);
        let c = vec![0.5f32; 16];
        let mut code = vec![0u8; q.code_bytes()];
        let coded = q.encode(&c, &c, &mut code);
        assert_eq!(coded.norm, 0.0);
        assert!(code.iter().all(|b| *b == 0));

        let query = q.query(&[1.0f32; 16], &c);
        let d = query.distance(&code, &coded);
        assert!(d.is_finite(), "{d}");
        // The centroid is where it says it is, so the distance is the query's
        // own residual and nothing else.
        let want: f32 = (0..16).map(|_| 0.25f32).sum();
        assert!((d - want).abs() < 1e-3, "{d} against {want}");
    }

    #[test]
    fn a_query_sitting_on_the_centroid_is_not_a_division_by_zero() {
        let q = Quantizer::new(16, Bits::One, 1);
        let c = vec![0.5f32; 16];
        let mut code = vec![0u8; q.code_bytes()];
        let v: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let coded = q.encode(&v, &c, &mut code);
        let d = q.query(&c, &c).distance(&code, &coded);
        assert!(d.is_finite(), "{d}");
    }

    #[test]
    fn a_code_is_written_over_whatever_was_in_the_buffer() {
        let q = Quantizer::new(32, Bits::One, 1);
        let c = vec![0.0f32; 32];
        let v: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let mut fresh = vec![0u8; q.code_bytes()];
        let mut dirty = vec![0xffu8; q.code_bytes()];
        let a = q.encode(&v, &c, &mut fresh);
        let b = q.encode(&v, &c, &mut dirty);
        assert_eq!(fresh, dirty);
        assert_eq!(a, b);
    }

    #[test]
    fn the_same_seed_is_the_same_code() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).cos()).collect();
        let c = vec![0.0f32; 64];
        let mut a = vec![0u8; 8];
        let mut b = vec![0u8; 8];
        Quantizer::new(64, Bits::One, 12).encode(&v, &c, &mut a);
        Quantizer::new(64, Bits::One, 12).encode(&v, &c, &mut b);
        assert_eq!(a, b);
        let mut d = vec![0u8; 8];
        Quantizer::new(64, Bits::One, 13).encode(&v, &c, &mut d);
        assert_ne!(a, d, "two seeds should not be one code");
    }
}
