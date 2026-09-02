//! MUVERA: a set of token vectors as one vector, so late interaction retrieval
//! runs on the index that is already here (`10` section 6).
//!
//! Late interaction is where retrieval quality went. A ColBERT style model does
//! not give a document one embedding, it gives every token one, and it scores a
//! query against a document with Chamfer similarity: for each query token, the
//! best match anywhere in the document, averaged over the query. That is a much
//! better score than one vector against one vector, and it is much more
//! expensive, because there is no single vector to put in an index and the
//! score is a loop over two sets rather than a dot product.
//!
//! The usual answer is a second index over every token of every document, so a
//! collection of a million documents at a hundred tokens each becomes a hundred
//! million vector index, plus a gather and a scoring pass on top. That is a
//! whole second system with its own memory, its own tuning and its own failure
//! modes.
//!
//! MUVERA (NeurIPS 2024, arXiv:2405.19504) does away with it. It maps a set of
//! token vectors to one fixed length vector, a Fixed Dimensional Encoding, such
//! that the dot product of a query's encoding with a document's approximates the
//! Chamfer similarity between the two sets, with a proven bound on the error.
//! So multi vector retrieval costs an encode at write time, the index that is
//! already here, and a different rerank function. No second index.
//!
//! # How it works
//!
//! The trick is to cut the vector space into buckets, with random hyperplanes,
//! and then compare a query's tokens only against the document tokens that
//! landed in the same bucket.
//!
//! A document's encoding holds, for each bucket, the average of the document
//! tokens that fell in it. A query's encoding holds, for each bucket, the sum of
//! the query tokens that fell in it. Take the dot product of the two and the
//! bucket contributes, for each query token in it, that token against the
//! average of the document tokens near it. If the hyperplanes did their job then
//! the document token that maximises the dot product is in the same bucket, and
//! the bucket average is close enough to it, so the sum over buckets is close to
//! the sum of maxima that Chamfer wanted.
//!
//! Two details make the difference between that being an argument and it being
//! true.
//!
//! An empty bucket in a document is filled with the document tokens whose own
//! bucket is nearest in Hamming distance. Without it, a query token in a bucket
//! the document did not reach contributes nothing, when the truth is that it
//! still has a best match somewhere in the document, and short documents lose
//! badly. With it, a one token document has that token in every bucket, which is
//! exactly right, because that token is the best match for every query token.
//!
//! The whole construction is repeated with independent hyperplanes and the
//! results are laid end to end. One repetition is a coin toss on whether a query
//! token and its true best match landed together. Several repetitions average
//! that away, and this is the knob that actually buys accuracy.
//!
//! # What the repetitions buy
//!
//! Three hundred documents of twenty four tokens each, forty queries, where a
//! query is six of one document's own tokens with noise on them and the answer
//! is the document it came from. How often the encoding alone, with no rerank,
//! puts that document first:
//!
//! ```text
//! repetitions      1      2      4      8     16     32
//! ranked first  .825   .900   .950   .950  1.000  1.000
//! ```
//!
//! That is the shape to expect. Nothing else moves it nearly as much: buckets
//! and block width change how long the encoding is far more than how good it
//! is, which is why the default puts eight repetitions on sixteen buckets
//! rather than the other way round.
//!
//! # What is different here
//!
//! The paper ends with an optional random projection of the whole encoding down
//! to a smaller dimension, to make it cheap to store and search. There is no
//! point doing that here. [`crate::Quantizer`] already turns a vector into one
//! bit a dimension with an error bound that a random projection does not have,
//! so projecting first would throw away accuracy to save space that RaBitQ was
//! going to save anyway, and better. The per bucket projection that shrinks each
//! block from the token dimension down to [`Shape::dproj`] is still here,
//! because that one is what keeps the encoding from being buckets times token
//! dimension long.
//!
//! The encodings come out unit length. The index ranks by squared distance, and
//! for vectors of equal length that ordering is exactly the dot product ordering
//! the approximation is stated in, so normalising is what makes the two agree.
//! What it costs is the length of a document's raw encoding, which mostly says
//! how many tokens crowded into each bucket rather than anything about whether
//! the document is a good answer, and [`chamfer`] on the candidates puts back
//! any ordering that lost.
//!
//! ```
//! use yo_vector::muvera::{Encoder, Shape, chamfer};
//!
//! let dim = 16;
//! let enc = Encoder::new(dim, Shape::default(), 7);
//!
//! // Two tokens for the document, one for the query, laid out end to end.
//! let doc: Vec<f32> = (0..2 * dim).map(|i| if i % dim == i / dim { 1.0 } else { 0.0 }).collect();
//! let query: Vec<f32> = (0..dim).map(|i| f32::from(u8::from(i == 1))).collect();
//!
//! // The query token is the second document token, so Chamfer is 1.
//! assert!((chamfer(&query, &doc, dim) - 1.0).abs() < 1e-6);
//!
//! // And both sides encode to one vector of the same fixed length.
//! assert_eq!(enc.document(&doc).len(), enc.fde_dim());
//! assert_eq!(enc.query(&query).len(), enc.fde_dim());
//! ```

use yo_common::Rng;

/// How big an encoding is and how much accuracy it buys.
///
/// The three numbers trade the same way in every experiment in the paper: more
/// buckets and more repetitions track Chamfer more closely and cost a longer
/// encoding, and the encoding's length is the product of all three.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// How many random hyperplanes cut the space, so there are `2^ksim`
    /// buckets.
    ///
    /// This is the one to think about against the number of tokens a document
    /// has. Buckets well past the token count means most of them are empty and
    /// filled from a neighbour, which is not wrong but is not buying anything
    /// either.
    pub ksim: usize,
    /// How many numbers each bucket's block is squeezed down to.
    ///
    /// Without this a block would be the token dimension long and the encoding
    /// would be buckets times that, which at 128 dimensional tokens and sixteen
    /// buckets is two thousand numbers for one repetition.
    pub dproj: usize,
    /// How many times the whole thing is done again with fresh hyperplanes.
    ///
    /// Whether a query token and its true best match land in the same bucket is
    /// a coin toss that this averages out, and it is the knob that buys
    /// accuracy rather than just length.
    pub reps: usize,
}

impl Default for Shape {
    /// Sixteen buckets, sixteen numbers a block, eight repetitions, which is
    /// two thousand numbers whatever the token dimension is.
    ///
    /// That is the middle of the range the paper measures, and it is a
    /// reasonable place to start for the hundred or so tokens a passage has. At
    /// one bit a dimension it is a 256 byte code, against the 100 token by 128
    /// dimension set it stands in for, which would be 12 kilobytes of floats.
    fn default() -> Shape {
        Shape {
            ksim: 4,
            dproj: 16,
            reps: 8,
        }
    }
}

/// Turns a set of token vectors into one vector.
///
/// Built from a token dimension, a [`Shape`] and a seed, and nothing else, so
/// two processes that were told the same three things build the same encoder
/// and the hyperplanes never have to be written down. Same rule as
/// [`crate::Rotation`], for the same reason.
pub struct Encoder {
    dim: usize,
    shape: Shape,
    /// `reps * ksim` hyperplane normals, `dim` long each, in that order.
    planes: Vec<f32>,
    /// `reps` projection matrices of `dproj` rows by `dim`, already scaled.
    proj: Vec<f32>,
}

impl Encoder {
    /// An encoder for `dim` dimensional tokens.
    ///
    /// # Panics
    ///
    /// If `dim` is zero, if `ksim` is not between 1 and 16, or if `dproj` or
    /// `reps` is zero.
    #[must_use]
    pub fn new(dim: usize, shape: Shape, seed: u64) -> Encoder {
        assert!(dim > 0, "a token has to have a dimension");
        assert!(
            (1..=16).contains(&shape.ksim),
            "ksim is {}, and a bucket index is built by shifting, so it has to \
             stay somewhere a machine can count to",
            shape.ksim
        );
        assert!(shape.dproj > 0, "a block has to have a width");
        assert!(shape.reps > 0, "there has to be at least one repetition");
        let mut rng = Rng::new(seed);
        let planes = (0..shape.reps * shape.ksim * dim)
            .map(|_| gauss(&mut rng))
            .collect();
        // Signs over the square root of the width, which is the sketch that
        // preserves a dot product in expectation. It is baked into the matrix
        // so the encode is a plain multiply.
        let scale = 1.0 / (shape.dproj as f32).sqrt();
        let proj = (0..shape.reps * shape.dproj * dim)
            .map(|_| {
                if rng.next_u64() & 1 == 0 {
                    scale
                } else {
                    -scale
                }
            })
            .collect();
        Encoder {
            dim,
            shape,
            planes,
            proj,
        }
    }

    /// The token dimension this was built for.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The shape this was built with.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// How long an encoding is, which is what to build the index at.
    #[must_use]
    pub fn fde_dim(&self) -> usize {
        self.shape.reps * self.buckets() * self.shape.dproj
    }

    /// Encode a document's tokens, laid out end to end.
    ///
    /// A bucket holds the average of the tokens that fell in it, and a bucket
    /// no token reached is filled from the tokens whose own bucket is nearest,
    /// so that every bucket has something to say. That filling is what lets a
    /// short document compete: a query token in a bucket the document never
    /// reached still has a best match in the document, and without the fill it
    /// would contribute nothing at all.
    ///
    /// # Panics
    ///
    /// If `tokens` is empty or is not a whole number of [`Encoder::dim`]
    /// vectors.
    #[must_use]
    pub fn document(&self, tokens: &[f32]) -> Vec<f32> {
        self.encode(tokens, true)
    }

    /// Encode a query's tokens, laid out end to end.
    ///
    /// A bucket holds the sum of the tokens that fell in it, not the average,
    /// because every query token is supposed to contribute its own best match
    /// to the score rather than share one. A bucket no query token reached is
    /// left at zero, because there is nothing there to ask for.
    ///
    /// # Panics
    ///
    /// If `tokens` is empty or is not a whole number of [`Encoder::dim`]
    /// vectors.
    #[must_use]
    pub fn query(&self, tokens: &[f32]) -> Vec<f32> {
        self.encode(tokens, false)
    }

    fn buckets(&self) -> usize {
        1usize << self.shape.ksim
    }

    fn encode(&self, tokens: &[f32], document: bool) -> Vec<f32> {
        let dim = self.dim;
        assert!(!tokens.is_empty(), "there is nothing to encode");
        assert_eq!(
            tokens.len() % dim,
            0,
            "the tokens are {} numbers, which is not a whole number of {dim} \
             dimensional vectors",
            tokens.len()
        );
        let n = tokens.len() / dim;
        let buckets = self.buckets();
        let mut out = vec![0.0f32; self.fde_dim()];
        let mut codes = vec![0u32; n];
        let mut totals = vec![0.0f32; buckets * dim];
        let mut counts = vec![0u32; buckets];
        let mut fill = vec![0.0f32; dim];
        for r in 0..self.shape.reps {
            totals.fill(0.0);
            counts.fill(0);
            for (t, code) in codes.iter_mut().enumerate() {
                let x = &tokens[t * dim..(t + 1) * dim];
                let k = self.bucket(r, x);
                *code = k as u32;
                counts[k] += 1;
                for (into, c) in totals[k * dim..(k + 1) * dim].iter_mut().zip(x) {
                    *into += c;
                }
            }
            for k in 0..buckets {
                let at = (r * buckets + k) * self.shape.dproj;
                if counts[k] > 0 {
                    // The document wants the average of what landed here, the
                    // query wants the sum, and both are linear so it makes no
                    // difference whether the scaling happens before the
                    // projection or after.
                    let scale = if document {
                        1.0 / f64::from(counts[k]) as f32
                    } else {
                        1.0
                    };
                    self.project(r, &totals[k * dim..(k + 1) * dim], scale, at, &mut out);
                } else if document {
                    let hits = nearest_by_hamming(tokens, dim, &codes, k as u32, &mut fill);
                    self.project(r, &fill, 1.0 / hits as f32, at, &mut out);
                }
            }
        }
        // Both sides come out unit length, and that is what makes the index's
        // squared distance order the same as the dot product order the
        // approximation is written in. The constant the paper carries, one over
        // the number of query tokens and one over the number of repetitions,
        // goes with it: it is one scale over the whole vector, so it moves the
        // estimate's value and not the order of anything, and the order is all
        // the index reads.
        unit(&mut out);
        out
    }

    /// Which bucket a token falls in: one bit per hyperplane, which side of it.
    fn bucket(&self, rep: usize, x: &[f32]) -> usize {
        let planes = &self.planes[rep * self.shape.ksim * self.dim..];
        let mut code = 0usize;
        for b in 0..self.shape.ksim {
            let plane = &planes[b * self.dim..(b + 1) * self.dim];
            code |= usize::from(dot(plane, x) > 0.0) << b;
        }
        code
    }

    /// Squeeze one bucket's `dim` numbers down to `dproj` of them, into `out`
    /// at `at`.
    fn project(&self, rep: usize, x: &[f32], scale: f32, at: usize, out: &mut [f32]) {
        let m = &self.proj[rep * self.shape.dproj * self.dim..];
        for j in 0..self.shape.dproj {
            out[at + j] = dot(&m[j * self.dim..(j + 1) * self.dim], x) * scale;
        }
    }
}

/// The sum of the tokens whose bucket is nearest `want` in Hamming distance,
/// into `fill`, and how many there were.
///
/// Ties are summed rather than broken, because there is no ordering of the
/// tokens that means anything and a rule that picks one of them would be an
/// arbitrary one dressed up as a decision.
fn nearest_by_hamming(
    tokens: &[f32],
    dim: usize,
    codes: &[u32],
    want: u32,
    fill: &mut [f32],
) -> u32 {
    let mut best = u32::MAX;
    let mut hits = 0u32;
    for (t, code) in codes.iter().enumerate() {
        let apart = (code ^ want).count_ones();
        if apart > best {
            continue;
        }
        if apart < best {
            best = apart;
            hits = 0;
            fill.fill(0.0);
        }
        hits += 1;
        for (into, c) in fill.iter_mut().zip(&tokens[t * dim..(t + 1) * dim]) {
            *into += c;
        }
    }
    hits
}

/// The Chamfer similarity of a query's tokens against a document's: for each
/// query token, the best it does against any document token, averaged over the
/// query.
///
/// This is the score the encoding approximates, and it is what to rerank the
/// candidates with once the index has narrowed the collection down, the same
/// way [`crate::Partitions::search`] reranks estimates against the full
/// precision vectors. It is quadratic in the token counts, which is why it runs
/// on the handful the index handed back and not on the collection.
///
/// # Panics
///
/// If `dim` is zero, or if either side is not a whole number of `dim`
/// dimensional vectors.
#[must_use]
pub fn chamfer(query: &[f32], doc: &[f32], dim: usize) -> f32 {
    assert!(dim > 0, "a token has to have a dimension");
    assert_eq!(query.len() % dim, 0, "the query is not whole tokens");
    assert_eq!(doc.len() % dim, 0, "the document is not whole tokens");
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for q in query.chunks_exact(dim) {
        let mut best = f32::NEG_INFINITY;
        for p in doc.chunks_exact(dim) {
            let d = dot(q, p);
            if d > best {
                best = d;
            }
        }
        total += best;
    }
    total / (query.len() / dim) as f32
}

/// A dot product, with eight running totals rather than one.
///
/// The same reason as [`crate::partition`]'s squared distance, and it is worth
/// repeating because it is not obvious: adding floats is not associative, so a
/// compiler is not allowed to turn one accumulator into a vector of them, and
/// the one line version is a chain of dependent adds four cycles apart. This is
/// the whole of an encode. Projecting one bucket is a row of the matrix against
/// it, there are buckets times repetitions of those, and at the default shape
/// and 128 dimensional tokens that is a quarter of a million multiply adds for
/// one document, all of them here.
///
/// The totals are summed in a fixed order at the end, so the answer is
/// deterministic, and it is a different answer from the one line version by the
/// last bit or so in the way any two orderings of a float sum are.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut totals = [0.0f32; 8];
    let mut i = 0;
    while i + 8 <= a.len() {
        for (k, total) in totals.iter_mut().enumerate() {
            *total += a[i + k] * b[i + k];
        }
        i += 8;
    }
    let mut sum = 0.0f32;
    for total in totals {
        sum += total;
    }
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// A standard normal, by Box Muller. Only ever called while an encoder is being
/// built, so throwing half of each pair away costs nothing worth saving.
fn gauss(rng: &mut Rng) -> f32 {
    let u1 = (uniform(rng)).max(f32::MIN_POSITIVE);
    let u2 = uniform(rng);
    (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
}

fn uniform(rng: &mut Rng) -> f32 {
    (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32
}

fn unit(v: &mut [f32]) {
    let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
    if len > 0.0 {
        for c in v {
            *c /= len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Partitions, Tuning, Vectors};

    /// A collection of token sets with the shape a late interaction model
    /// produces: a document is about a few things out of a much larger pool,
    /// and its tokens point near the directions of those things rather than
    /// anywhere at all.
    fn corpus(dim: usize, docs: usize, tokens: usize, topics: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        let concepts: Vec<Vec<f32>> = (0..topics).map(|_| draw(dim, &mut rng)).collect();
        (0..docs)
            .map(|_| {
                let about: Vec<usize> = (0..4).map(|_| rng.below(topics)).collect();
                let mut out = Vec::with_capacity(tokens * dim);
                for t in 0..tokens {
                    let base = &concepts[about[t % about.len()]];
                    out.extend_from_slice(&near(base, 0.5, &mut rng));
                }
                out
            })
            .collect()
    }

    /// Queries, each one taken from a particular document.
    ///
    /// This is the part that took a diagnostic to get right, and it is worth
    /// writing down. The obvious test corpus is unrelated documents and
    /// unrelated queries, and it measures nothing at all: over three hundred
    /// documents of that kind the best Chamfer score is 0.331 and the thirtieth
    /// best is 0.261, so which ten are the top ten is decided by noise in the
    /// third decimal place and no approximation of any quality can recover
    /// them. A recall number against that ground truth looks like a verdict on
    /// the encoder and is a verdict on the random number generator.
    ///
    /// A real query set is not like that. A query came from somewhere, and the
    /// passage it came from is the answer, which is exactly how MS MARCO is
    /// built and what recall at k means there. So a query here is a handful of
    /// one document's own tokens with noise on them, and the test is whether
    /// the document it came from comes back.
    fn queries(
        docs: &[Vec<f32>],
        dim: usize,
        n: usize,
        len: usize,
        seed: u64,
    ) -> Vec<(usize, Vec<f32>)> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let from = rng.below(docs.len());
                let doc = &docs[from];
                let have = doc.len() / dim;
                let mut q = Vec::with_capacity(len * dim);
                for _ in 0..len {
                    let t = rng.below(have);
                    let token = &doc[t * dim..(t + 1) * dim];
                    q.extend_from_slice(&near(token, 0.35, &mut rng));
                }
                (from, q)
            })
            .collect()
    }

    /// A unit vector near `base`, off by `how much`.
    fn near(base: &[f32], off: f32, rng: &mut Rng) -> Vec<f32> {
        let noise = draw(base.len(), rng);
        let mut v: Vec<f32> = base.iter().zip(&noise).map(|(c, o)| c + o * off).collect();
        unit(&mut v);
        v
    }

    fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| gauss(rng)).collect();
        unit(&mut v);
        v
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// How often the document a query came from is in the encoding's own top
    /// `keep` out of the whole collection.
    fn found(dim: usize, shape: Shape, keep: usize, seed: u64) -> f32 {
        let enc = Encoder::new(dim, shape, seed);
        let docs = corpus(dim, 300, 24, 64, seed);
        let qs = queries(&docs, dim, 40, 6, seed ^ 0x5eed);
        let fdes: Vec<Vec<f32>> = docs.iter().map(|d| enc.document(d)).collect();

        let mut hits = 0usize;
        for (from, q) in &qs {
            let f = enc.query(q);
            let mut by: Vec<(usize, f32)> = fdes
                .iter()
                .enumerate()
                .map(|(i, d)| (i, dot(&f, d)))
                .collect();
            by.select_nth_unstable_by(keep, |a, b| b.1.total_cmp(&a.1));
            hits += usize::from(by[..keep].iter().any(|(i, _)| i == from));
        }
        hits as f32 / qs.len() as f32
    }

    #[test]
    fn an_encoding_is_the_length_it_says_it_is() {
        let shape = Shape {
            ksim: 3,
            dproj: 8,
            reps: 4,
        };
        let enc = Encoder::new(32, shape, 1);
        assert_eq!(enc.fde_dim(), 4 * 8 * 8);
        assert_eq!(enc.dim(), 32);

        // Whatever the token count is, and that is the whole point of it.
        for tokens in [1usize, 2, 40] {
            let set = corpus(32, 1, tokens, 4, 9).remove(0);
            assert_eq!(enc.document(&set).len(), enc.fde_dim());
            assert_eq!(enc.query(&set).len(), enc.fde_dim());
        }
    }

    #[test]
    fn the_same_seed_is_the_same_encoder() {
        let set = corpus(24, 1, 9, 4, 3).remove(0);
        let a = Encoder::new(24, Shape::default(), 77).document(&set);
        let b = Encoder::new(24, Shape::default(), 77).document(&set);
        let other = Encoder::new(24, Shape::default(), 78).document(&set);
        assert_eq!(a, b);
        assert_ne!(a, other);
    }

    #[test]
    fn chamfer_is_the_best_match_for_each_query_token() {
        let dim = 4;
        // Two document tokens: one along x, one along y.
        let doc = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // Two query tokens: one along y, one halfway between x and z.
        let half = core::f32::consts::FRAC_1_SQRT_2;
        let query = [0.0, 1.0, 0.0, 0.0, half, 0.0, half, 0.0];
        // The first matches y exactly, the second does 1/sqrt(2) against x.
        assert!((chamfer(&query, &doc, dim) - (1.0 + half) / 2.0).abs() < 1e-6);

        // It is not symmetric, and it is not supposed to be: a query of one
        // token that the document has exactly is a perfect score, and the
        // document scored against that query is not, because its other token
        // has nothing to match.
        let one = [1.0f32, 0.0, 0.0, 0.0];
        assert!((chamfer(&one, &doc, dim) - 1.0).abs() < 1e-6);
        assert!((chamfer(&doc, &one, dim) - 0.5).abs() < 1e-6);
        assert_eq!(chamfer(&[], &doc, dim), 0.0);
    }

    /// The fill for empty buckets, checked by its cleanest consequence: a
    /// document with one token has that token in every bucket, because it is
    /// the best match for anything a query can ask.
    #[test]
    fn a_one_token_document_fills_every_bucket() {
        let dim = 16;
        let shape = Shape {
            ksim: 3,
            dproj: 8,
            reps: 2,
        };
        let enc = Encoder::new(dim, shape, 11);
        let one = corpus(dim, 1, 1, 4, 5).remove(0);
        let fde = enc.document(&one);

        // Every block of every repetition is the same projected token, so no
        // block is empty and they all agree.
        for r in 0..shape.reps {
            let first = &fde[r * 8 * shape.dproj..][..shape.dproj];
            assert!(first.iter().any(|c| c.abs() > 1e-6), "rep {r} is empty");
            for k in 1..8 {
                let block = &fde[(r * 8 + k) * shape.dproj..][..shape.dproj];
                for (a, b) in first.iter().zip(block) {
                    assert!((a - b).abs() < 1e-6, "rep {r} bucket {k} differs");
                }
            }
        }
    }

    #[test]
    fn the_encoding_finds_the_document_a_query_came_from() {
        let got = found(48, Shape::default(), 10, 4242);
        assert!(
            got >= 0.9,
            "the encoding's top ten held it {got} of the time"
        );
    }

    /// The paper's claim about which knob matters, and the one that tells you
    /// what to turn when recall is short. Whether a query token and its best
    /// match land in the same bucket is a coin toss, and repetitions are what
    /// average it out.
    #[test]
    fn more_repetitions_find_it_more_often() {
        let one = found(
            48,
            Shape {
                reps: 1,
                ..Shape::default()
            },
            1,
            4242,
        );
        let many = found(
            48,
            Shape {
                reps: 16,
                ..Shape::default()
            },
            1,
            4242,
        );
        assert!(
            many > one + 0.1,
            "sixteen repetitions found it {many} of the time against one repetition's {one}"
        );
    }

    /// The record log, holding the encodings, so the index can rerank.
    struct Fdes(Vec<Vec<f32>>);

    impl Vectors for Fdes {
        fn get(&self, id: u64, into: &mut [f32]) -> bool {
            match self.0.get(id as usize) {
                Some(v) => {
                    into.copy_from_slice(v);
                    true
                }
                None => false,
            }
        }
    }

    /// End to end, which is the thing that has to work: encodings in the
    /// ordinary partition index, searched with an encoded query, and the
    /// candidates reranked with exact Chamfer.
    ///
    /// No second index, no postings over every token of every document, and the
    /// exact score runs on the forty documents the index handed back rather
    /// than on all three hundred.
    #[test]
    fn retrieval_then_a_chamfer_rerank_finds_the_right_document() {
        let dim = 48;
        let enc = Encoder::new(dim, Shape::default(), 909);
        let docs = corpus(dim, 300, 24, 64, 909);
        let qs = queries(&docs, dim, 40, 6, 0xbeef);
        let fdes = Fdes(docs.iter().map(|d| enc.document(d)).collect());

        let mut ix = Partitions::new(
            enc.fde_dim(),
            Bits::One,
            7,
            Tuning {
                posting: 48,
                ..Tuning::default()
            },
        );
        for (id, f) in fdes.0.iter().enumerate() {
            ix.insert(id as u64, f);
        }
        ix.maintain(&fdes, 1 << 20);

        let mut first = 0usize;
        for (from, q) in &qs {
            let best = ix
                .search(&enc.query(q), 40, &fdes)
                .into_iter()
                .map(|h| (h.id as usize, chamfer(q, &docs[h.id as usize], dim)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            first += usize::from(best.map(|(id, _)| id) == Some(*from));
        }
        let got = first as f32 / qs.len() as f32;
        assert!(
            got >= 0.9,
            "the right document came first {got} of the time"
        );
    }

    /// What the whole thing is standing in for, priced.
    ///
    /// A Chamfer scan over the collection is every query token against every
    /// token of every document. The encoding turns that into one dot product a
    /// document, and then one Chamfer against the few the index kept.
    #[test]
    fn the_rerank_is_the_only_chamfer_anyone_pays_for() {
        let dim = 32;
        let enc = Encoder::new(dim, Shape::default(), 5);
        let docs = corpus(dim, 200, 24, 32, 5);
        let (from, q) = queries(&docs, dim, 1, 6, 17).remove(0);

        let scanned: usize = docs.iter().map(|d| d.len() / dim).sum::<usize>() * (q.len() / dim);
        let fdes: Vec<Vec<f32>> = docs.iter().map(|d| enc.document(d)).collect();
        let f = enc.query(&q);
        let mut by: Vec<(usize, f32)> = fdes
            .iter()
            .enumerate()
            .map(|(i, d)| (i, dot(&f, d)))
            .collect();
        by.sort_by(|a, b| b.1.total_cmp(&a.1));
        let kept: Vec<usize> = by[..20].iter().map(|(i, _)| *i).collect();
        let reranked: usize =
            kept.iter().map(|i| docs[*i].len() / dim).sum::<usize>() * (q.len() / dim);

        assert!(
            kept.contains(&from),
            "the document it came from was dropped"
        );
        assert!(
            reranked * 8 < scanned,
            "reranking {reranked} token pairs against a full scan's {scanned} is not a saving"
        );
    }
}
