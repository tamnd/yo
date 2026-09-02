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
//!
//! # A code is bit planes, and that is what makes the scan fast
//!
//! The obvious layout writes a coordinate's level in the bits next to it, and
//! then measuring a code against a query is a multiply per dimension. That is
//! 768 multiplies per candidate and it does not fit inside a millisecond search.
//!
//! So a code is stored transposed. Plane `b` holds bit `b` of every
//! coordinate's level, one bit per coordinate, `dim` bits rounded up to whole
//! 64 bit words, and the planes run least significant first. A one bit code is
//! one plane and a four bit code is four, and the byte count is the same either
//! way.
//!
//! The query is quantised to [`Bits::query_bits`] and transposed the same way.
//! Then the sum of the code's levels times the query's levels is
//!
//! ```text
//! sum over a, b of 2^(a+b) * popcount(code plane a AND query plane b)
//! ```
//!
//! which is four ANDs and four popcounts per word for a one bit code, against
//! 64 float multiplies for the same 64 coordinates. The sums are exact
//! integers, so the arithmetic is also better behaved than the float version it
//! replaces, and nothing is left to round until the end.
//!
//! # The query is quantised finer than the code
//!
//! Quantising the query costs accuracy, and how much was measured rather than
//! assumed. At one bit the query at four bits is off by about a third of what
//! the code itself is off by, which is lost in the quadrature and does not
//! matter. At four bits the code is ten times more accurate and the same four
//! bit query is off by four times as much as the code, which throws away the
//! entire reason anyone would pay for four bit codes.
//!
//! So the query width follows the code width: four bits against a one bit code
//! and eight against a four bit one. That puts the query's error back at about
//! a third of the code's in both cases, and [`Query::cosine`] against
//! [`Query::cosine_exact`] is the test that holds it there.

use crate::rotate::Rotation;

/// How many 64 bit words one plane of a `dim` dimensional code takes.
fn words_of(dim: usize) -> usize {
    dim.div_ceil(64)
}

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
    /// How many bits one coordinate takes, which is also how many planes a code
    /// is written in.
    #[must_use]
    pub fn count(self) -> usize {
        match self {
            Bits::One => 1,
            Bits::Four => 4,
        }
    }

    /// How many bits a query is quantised to before it is scanned against codes
    /// this wide.
    ///
    /// Four bits either side is what RaBitQ specifies, and it is right for a one
    /// bit code and wrong for a four bit one, because a four bit code is ten
    /// times more accurate and a four bit query is not. Eight is what puts the
    /// query's error back under the code's, and the cost is four more ANDs and
    /// four more popcounts per word on the path that was already paying for four
    /// times the bytes.
    #[must_use]
    pub fn query_bits(self) -> usize {
        match self {
            Bits::One => 4,
            Bits::Four => 8,
        }
    }

    /// The largest level a coordinate this wide can round to.
    fn top(self) -> u64 {
        (1 << self.count()) - 1
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
    ///
    /// A plane is whole 64 bit words because the scan reads words, so a
    /// dimension that is not a multiple of 64 pays for the rest of its last
    /// word. At the dimensions embedding families actually use there is nothing
    /// to pay.
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        words_of(self.dim()) * 8 * self.bits.count()
    }

    /// A vector in the frame everything else here works in.
    ///
    /// The rotation is linear, so `rotate(v - c)` is `rotate(v) - rotate(c)`,
    /// and an index that keeps its centroids already rotated never has to
    /// rotate one again. That is what [`Quantizer::encode_rotated`] and
    /// [`Quantizer::query_rotated`] are for, and the rotation is the expensive
    /// half of both of the two calls above them.
    ///
    /// # Panics
    ///
    /// If `v` is not [`Quantizer::dim`] long.
    #[must_use]
    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        let mut x = v.to_vec();
        self.rot.apply(&mut x);
        x
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
        if norm > 0.0 {
            for c in &mut x {
                *c /= norm;
            }
        }
        self.rot.apply(&mut x);
        self.write(&x, norm, code)
    }

    /// The same as [`Quantizer::encode`] with both sides already rotated.
    ///
    /// # Panics
    ///
    /// If `x` or `centroid` is not [`Quantizer::dim`] long, or `code` is not
    /// [`Quantizer::code_bytes`] long.
    pub fn encode_rotated(&self, x: &[f32], centroid: &[f32], code: &mut [u8]) -> Coded {
        let mut r = self.residual(x, centroid);
        let norm = length(&r);
        if norm > 0.0 {
            for c in &mut r {
                *c /= norm;
            }
        }
        self.write(&r, norm, code)
    }

    /// The code of a rotated unit residual whose original length was `norm`.
    fn write(&self, x: &[f32], norm: f32, code: &mut [u8]) -> Coded {
        assert_eq!(
            code.len(),
            self.code_bytes(),
            "a code here is {} bytes and the buffer is {}",
            self.code_bytes(),
            code.len()
        );
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
        let mut coded = match self.bits {
            Bits::One => sign_code(x, code),
            Bits::Four => level_code(x, code),
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
        self.prepare(x, norm)
    }

    /// The same as [`Quantizer::query`] with both sides already rotated.
    ///
    /// A search rotates its query once and then meets every partition it probes
    /// through this, so the rotation is paid for once rather than once per
    /// partition.
    ///
    /// # Panics
    ///
    /// If `q` or `centroid` is not [`Quantizer::dim`] long.
    #[must_use]
    pub fn query_rotated(&self, q: &[f32], centroid: &[f32]) -> Query {
        let mut x = self.residual(q, centroid);
        let norm = length(&x);
        if norm > 0.0 {
            for c in &mut x {
                *c /= norm;
            }
        }
        self.prepare(x, norm)
    }

    /// Quantise and transpose a rotated unit residual into the form the scan
    /// meets a code with.
    fn prepare(&self, x: Vec<f32>, norm: f32) -> Query {
        // The sum is taken from the unquantised coordinates because it is one
        // number computed once, so there is nothing to gain by approximating
        // it and it is half of what the estimator adds up.
        let sum = x.iter().sum();
        let words = words_of(self.dim());
        let wide = self.bits.query_bits();
        let top = (1u64 << wide) - 1;
        let (lo, hi) = span(&x);
        let delta = step(lo, hi, top);
        let mut planes = vec![0u64; wide * words];
        for (i, &c) in x.iter().enumerate() {
            let level = level_of(c, lo, delta, top);
            for (b, plane) in planes.chunks_exact_mut(words).enumerate() {
                plane[i / 64] |= ((level >> b) & 1) << (i % 64);
            }
        }
        Query {
            bits: self.bits,
            words,
            rotated: x,
            planes,
            lo,
            delta,
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

/// A query, rotated and quantised once and then measured against every code in
/// a partition.
pub struct Query {
    /// The width of the codes this is measured against, so the scan knows how
    /// many planes each one has.
    bits: Bits,
    /// The words one plane takes, for both this query and those codes.
    words: usize,
    /// The query's residual, unit length and rotated, which is only what
    /// [`Query::cosine_exact`] reads.
    rotated: Vec<f32>,
    /// The same coordinates at [`Bits::query_bits`], transposed into planes.
    planes: Vec<u64>,
    /// What level zero of the query reconstructs to.
    lo: f32,
    /// What one level of the query is worth.
    delta: f32,
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
    /// the vector it came from. The division is the part that makes it
    /// unbiased. The inner product is the popcount scan.
    ///
    /// # Panics
    ///
    /// If `code` is not as long as the codes this query's quantiser writes.
    #[must_use]
    pub fn cosine(&self, code: &[u8], coded: &Coded) -> f32 {
        if coded.norm == 0.0 || self.norm == 0.0 {
            return 0.0;
        }
        let (total, cross) = packed_dot(code, self.bits.count(), self.words, &self.planes);
        // Undo the query's quantisation: every level was `lo + delta * level`,
        // so the sum over the code's levels needs the `lo` part weighted by how
        // much level the code is carrying and the `delta` part by the cross
        // term the popcounts just measured.
        let levels = self.lo * total as f32 + self.delta * cross as f32;
        (coded.lo * self.sum + coded.delta * levels) / coded.correction
    }

    /// The same estimate with the query left at full precision.
    ///
    /// This is the reference the popcount scan is checked against, and it is
    /// public because a divergence between the two is worth being able to
    /// measure from outside. It reads one coordinate at a time and it is not
    /// what a search should call.
    ///
    /// # Panics
    ///
    /// If `code` is not as long as the codes this query's quantiser writes.
    #[must_use]
    pub fn cosine_exact(&self, code: &[u8], coded: &Coded) -> f32 {
        if coded.norm == 0.0 || self.norm == 0.0 {
            return 0.0;
        }
        let levels = exact_dot(code, self.bits.count(), self.words, &self.rotated);
        (coded.lo * self.sum + coded.delta * levels) / coded.correction
    }
}

/// The smallest and the largest coordinate.
fn span(x: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &c in x {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

/// What one level is worth when `lo` to `hi` is cut into `top` of them.
///
/// A vector whose coordinates are all the same value has no range to divide up.
/// It cannot happen after a rotation of a non zero residual, and a division by
/// zero here would be a silent NaN rather than a loud one.
fn step(lo: f32, hi: f32, top: u64) -> f32 {
    if hi > lo { (hi - lo) / top as f32 } else { 1.0 }
}

/// Which level a coordinate rounds to.
fn level_of(c: f32, lo: f32, delta: f32, top: u64) -> u64 {
    (((c - lo) / delta).round() as i64).clamp(0, top as i64) as u64
}

/// Write one word of one plane.
fn put(code: &mut [u8], words: usize, plane: usize, w: usize, v: u64) {
    let at = (plane * words + w) * 8;
    code[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// The sign of each coordinate, written into the one plane a one bit code has.
///
/// The reconstruction is `(2 * bit - 1) / sqrt(D)`, which is a unit vector
/// pointing at one corner of the cube, so `lo` and `delta` fall out of the
/// dimension and the correction is the sum of the absolute coordinates over
/// `sqrt(D)`.
fn sign_code(x: &[f32], code: &mut [u8]) -> Coded {
    let words = words_of(x.len());
    let mut abs = 0.0f32;
    for (w, chunk) in x.chunks(64).enumerate() {
        let mut bits = 0u64;
        for (k, &c) in chunk.iter().enumerate() {
            abs += c.abs();
            if c >= 0.0 {
                bits |= 1 << k;
            }
        }
        put(code, words, 0, w, bits);
    }
    let root = (x.len() as f32).sqrt();
    Coded {
        norm: 0.0,
        correction: abs / root,
        lo: -1.0 / root,
        delta: 2.0 / root,
    }
}

/// Sixteen levels between the smallest and the largest coordinate, written
/// across the four planes a four bit code has.
fn level_code(x: &[f32], code: &mut [u8]) -> Coded {
    let words = words_of(x.len());
    let top = Bits::Four.top();
    let (lo, hi) = span(x);
    let delta = step(lo, hi, top);
    let mut recon = 0.0f32;
    let mut dot = 0.0f32;
    for (w, chunk) in x.chunks(64).enumerate() {
        let mut planes = [0u64; 4];
        for (k, &c) in chunk.iter().enumerate() {
            let level = level_of(c, lo, delta, top);
            for (b, plane) in planes.iter_mut().enumerate() {
                *plane |= ((level >> b) & 1) << k;
            }
            let back = lo + level as f32 * delta;
            recon += back * back;
            dot += back * c;
        }
        for (b, &plane) in planes.iter().enumerate() {
            put(code, words, b, w, plane);
        }
    }
    let len = recon.sqrt();
    Coded {
        norm: 0.0,
        correction: dot / len,
        lo: lo / len,
        delta: delta / len,
    }
}

/// The scan: the sum of the code's levels, and the sum of the code's levels
/// times the query's, both exact.
///
/// This is the loop the whole search spends its time in. Every plane of the
/// code is met by every plane of the query, and a meeting is an AND and a
/// popcount over 64 coordinates at a time.
// Clippy wants `as_chunks::<8>()` here and it is 21 percent slower: 17.99
// microseconds against 14.89 for `scan/one/768`, and 118.58 against 96.96 for
// `scan/four/768`, both the minimum per iteration out of the same pair of runs.
// Walking two byte slices side by side is what vectorises, and walking a slice
// of eight byte arrays against a slice of words is what does not.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn packed_dot(code: &[u8], planes: usize, words: usize, query: &[u64]) -> (u32, u32) {
    assert_eq!(
        code.len(),
        planes * words * 8,
        "a code here is {} bytes and this one is {}",
        planes * words * 8,
        code.len()
    );
    let mut total = 0u32;
    let mut cross = 0u32;
    for (a, plane) in code.chunks_exact(words * 8).enumerate() {
        let mut ones = 0u32;
        for chunk in plane.chunks_exact(8) {
            ones += word(chunk).count_ones();
        }
        total += ones << a;
        for (b, qp) in query.chunks_exact(words).enumerate() {
            let mut acc = 0u32;
            for (chunk, &qw) in plane.chunks_exact(8).zip(qp) {
                acc += (word(chunk) & qw).count_ones();
            }
            cross += acc << (a + b);
        }
    }
    (total, cross)
}

/// The same dot product against a query that was never quantised, one
/// coordinate at a time. The reference, not the scan.
fn exact_dot(code: &[u8], planes: usize, words: usize, query: &[f32]) -> f32 {
    assert_eq!(
        code.len(),
        planes * words * 8,
        "a code here is {} bytes and this one is {}",
        planes * words * 8,
        code.len()
    );
    let mut sum = 0.0f32;
    for (i, &qi) in query.iter().enumerate() {
        let mut level = 0u32;
        for b in 0..planes {
            let at = (b * words + i / 64) * 8;
            level |= (((word(&code[at..at + 8]) >> (i % 64)) & 1) as u32) << b;
        }
        sum += level as f32 * qi;
    }
    sum
}

/// Eight bytes as a word, the way a code stores one.
fn word(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight bytes"))
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

    fn residual(v: &[f32], c: &[f32]) -> Vec<f32> {
        v.iter().zip(c).map(|(a, b)| a - b).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
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
        // A plane is whole words, so a dimension that is not a multiple of 64
        // pays for the rest of its last one. A hundred coordinates is two
        // words either way.
        assert_eq!(Quantizer::new(100, Bits::One, 1).code_bytes(), 16);
        assert_eq!(Quantizer::new(128, Bits::One, 1).code_bytes(), 16);
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

    /// What quantising the query costs, measured against what the code itself
    /// costs, because that is the only comparison that means anything.
    ///
    /// The query's error and the code's are independent, so they add in
    /// quadrature, and a query error a third of the code's makes the whole
    /// estimate five percent worse. That is the bar. An absolute threshold here
    /// would pass at four bit codes while the query was throwing away the whole
    /// reason to pay for them, which is exactly what the first cut did.
    #[test]
    fn the_query_is_quantised_finer_than_the_code_it_is_measured_against() {
        for (bits, dim) in [
            (Bits::One, 128),
            (Bits::One, 256),
            (Bits::One, 768),
            (Bits::Four, 256),
            (Bits::Four, 768),
        ] {
            let vs = corpus(dim, 200, 5);
            let qs = corpus(dim, 20, 6);
            let c = mean(&vs);
            let q = Quantizer::new(dim, bits, 3);
            let (codes, meta) = encode_all(&q, &vs, &c);
            let width = q.code_bytes();

            // How far the scan is from the same estimate on an unquantised
            // query, and how far that estimate is from the truth.
            let mut from_query = 0.0f32;
            let mut from_code = 0.0f32;
            for query in &qs {
                let prepared = q.query(query, &c);
                let qr = residual(query, &c);
                let qn = length(&qr);
                for (i, v) in vs.iter().enumerate() {
                    let code = &codes[i * width..(i + 1) * width];
                    let fast = prepared.cosine(code, &meta[i]);
                    let slow = prepared.cosine_exact(code, &meta[i]);
                    let vr = residual(v, &c);
                    let truth = dot(&qr, &vr) / (qn * length(&vr));
                    from_query += (fast - slow).abs();
                    from_code += (slow - truth).abs();
                }
            }
            let ratio = from_query / from_code;
            assert!(
                ratio < 0.5,
                "{dim} at {bits:?}: the query costs {ratio} of what the code costs"
            );
        }
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
        // And the coordinates that are not there are not set either, because
        // the scan popcounts whole words and would count them.
        assert!(fresh[4..].iter().all(|b| *b == 0), "{fresh:?}");
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

    /// The planes are the layout the file will hold, so a code has to read back
    /// as the levels that went into it.
    #[test]
    fn a_code_reads_back_as_the_levels_it_was_written_from() {
        let dim = 200;
        let q = Quantizer::new(dim, Bits::Four, 4);
        let c = vec![0.0f32; dim];
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).sin()).collect();
        let mut code = vec![0u8; q.code_bytes()];
        let coded = q.encode(&v, &c, &mut code);

        // Pull every level back out of the planes and rebuild the vector the
        // code stands for. It should point the same way the original does.
        let words = words_of(dim);
        let mut back = vec![0.0f32; dim];
        for (i, b) in back.iter_mut().enumerate() {
            let mut level = 0u32;
            for p in 0..4 {
                let at = (p * words + i / 64) * 8;
                level |= (((word(&code[at..at + 8]) >> (i % 64)) & 1) as u32) << p;
            }
            *b = coded.lo + level as f32 * coded.delta;
        }
        let mut spun: Vec<f32> = v.iter().map(|c| c / length(&v)).collect();
        crate::Rotation::new(dim, q.seed()).apply(&mut spun);
        let cos = back
            .iter()
            .zip(&spun)
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .abs()
            / length(&back);
        assert!(cos > 0.95, "the code points somewhere else: {cos}");
    }
}
