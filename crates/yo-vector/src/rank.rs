//! Coding the centroids so a search does not have to read them, which was tried
//! and does not pay.
//!
//! Nothing here is on the search path. This file is the measurement and the
//! reasoning behind a decision not to do something, kept because the idea is an
//! obvious one that will occur to somebody again, and it takes a day to build
//! and ten minutes to disprove once the right thing is measured.
//!
//! # The idea
//!
//! A search ranks every centroid, because [`coarse`](crate::coarse) says why
//! picking the probe head out of a tree over them costs more recall than it
//! saves time. What is left over from that decision is a cost that does not
//! depend on the probe count at all: 2963 centroids at 1024 dimensions is twelve
//! megabytes read to answer a query that then reads two.
//!
//! So do to the centroids what is already done to the members. Quantise each one
//! against the mean of all of them, scan the codes with the same estimator, then
//! take a shortlist several times wider than the head being asked for and
//! measure only those exactly. Every centroid still gets looked at on every
//! query, which is the part a tree cannot offer, so recall would be untouched
//! and the reading would be a fraction of what it was.
//!
//! The recall half of that is true. It was built, and on a million MS-MARCO
//! passages it returned the same probe head and the same recall at 10 to four
//! decimal places at every probe and rerank setting in `example recall`. The
//! time half is false.
//!
//! # Why it does not pay
//!
//! `what_ranking_the_centroids_costs` measures the two ways of ranking the same
//! centroids against each other, back to back on one machine, at a head of 128.
//! On a quiet 13900K:
//!
//! ```text
//! centroids  dimensions   read in full   coded and reranked
//!      2963        1024        521.7 us            1696.1 us
//!      2963         128         70.8 us             615.3 us
//!     10000         768       1923.6 us            5464.4 us
//! ```
//!
//! Coding them is three to nine times slower than reading them. The premise was
//! that twelve megabytes is a bandwidth problem, and it is not one: reading them
//! in full is a flat, vectorised, perfectly prefetched pass that measures out at
//! better than twenty gigabytes a second, so it is already going as fast as the
//! memory will go and there is nothing between it and the hardware to remove.
//!
//! What replaced it was not cheaper. The scan has to be four bit, and that is
//! forced rather than chosen: `how_much_slack_the_shortlist_needs` shows one bit
//! does not rank centroids at all.
//!
//! ```text
//! head of   one bit   four bit
//!      16      1563         72
//!     128      2265        257
//!     512      2716        895
//! ```
//!
//! Those are how far down the estimate's order the true head reaches, out of
//! three thousand, which is how wide the shortlist would have to be. At one bit
//! the shortlist is most of the collection and the exact pass comes back in
//! full. The reason is the frame. A member is coded against the centroid it
//! belongs to, which is a local reference picked to be near it. A centroid is
//! coded against the mean of every centroid, which is near nothing. Centroids
//! are clustered, so a one bit residual from a global mean mostly says which
//! cluster, and the whole true head is inside one cluster.
//!
//! So it has to be four bit, and a four bit estimator is arithmetic rather than
//! reading: four planes of the code against several of the query, per centroid.
//! That trades a pass that was running at memory speed for a pass that is
//! compute bound, and it loses. Then the exact rerank of the shortlist is added
//! on top, and unlike the pass it replaced it reads scattered rows rather than a
//! run, so it does not prefetch either.
//!
//! # What would have to change for it to be worth another look
//!
//! A four bit scan that is closer to four times the one bit cost than to what it
//! is now, which is a question about `rabitq.rs` and not about this. Or
//! centroids coded against a local reference rather than a global mean, so that
//! one bit is enough, which is a hierarchy and therefore is the thing
//! [`coarse`](crate::coarse) already measured and rejected on the search path.
//!
//! Neither is worth doing for this. Ranking the centroids is about half a
//! millisecond of a six and a half millisecond MS-MARCO query, and the other six
//! are in the posting scan.

#![cfg(test)]

use crate::dist::sqdist;
use crate::rabitq::{Bits, Coded, Quantizer};
use yo_common::Rng;

/// Centroids the way a real collection makes them: a handful of broad groups
/// with the points inside a group close together. Uniform noise has no such
/// structure, and structure is what makes ranking centroids hard.
fn clumped(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut unit = || (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
    let groups = 24;
    let hubs: Vec<f32> = (0..groups * dim).map(|_| unit()).collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let h = (i % groups) * dim;
        for d in 0..dim {
            out.push(hubs[h + d] + (unit() - 0.5) * 0.2);
        }
    }
    out
}

/// The true `want` nearest centroids to `u`, nearest first.
fn truth(u: &[f32], centroids: &[f32], dim: usize, n: usize, want: usize) -> Vec<usize> {
    let mut by: Vec<(usize, f32)> = (0..n)
        .map(|p| (p, sqdist(u, &centroids[p * dim..(p + 1) * dim])))
        .collect();
    by.sort_by(|a, b| a.1.total_cmp(&b.1));
    by.truncate(want);
    by.into_iter().map(|(p, _)| p).collect()
}

/// The centroids coded against their own mean, which is the thing being weighed.
struct Codes {
    quant: Quantizer,
    mean: Vec<f32>,
    codes: Vec<u8>,
    meta: Vec<Coded>,
}

impl Codes {
    fn build(centroids: &[f32], dim: usize, n: usize, bits: Bits) -> Codes {
        let quant = Quantizer::new(dim, bits, 0x51de_0001);
        let mut mean = vec![0.0f32; dim];
        for p in 0..n {
            for (m, c) in mean.iter_mut().zip(&centroids[p * dim..(p + 1) * dim]) {
                *m += *c;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }
        let width = quant.code_bytes();
        let mut codes = vec![0u8; n * width];
        let meta: Vec<Coded> = (0..n)
            .map(|p| {
                quant.encode_rotated(
                    &centroids[p * dim..(p + 1) * dim],
                    &mean,
                    &mut codes[p * width..(p + 1) * width],
                )
            })
            .collect();
        Codes {
            quant,
            mean,
            codes,
            meta,
        }
    }

    /// The estimate for every centroid, which is the pass that was supposed to
    /// replace reading them.
    fn estimate(&self, u: &[f32], scores: &mut Vec<f32>) {
        scores.clear();
        scores.resize(self.meta.len(), 0.0);
        self.quant
            .query_rotated(u, &self.mean)
            .scan(&self.codes, &self.meta, scores);
    }

    /// The head as the whole scheme would have produced it: a shortlist by
    /// estimate, then the truth over the shortlist.
    fn head(&self, u: &[f32], centroids: &[f32], want: usize, scores: &mut Vec<f32>) -> Vec<usize> {
        let dim = self.quant.dim();
        let n = self.meta.len();
        self.estimate(u, scores);
        let order = |a: &(usize, f32), b: &(usize, f32)| a.1.total_cmp(&b.1);
        let wide = (want * 4).clamp(128, 1024).min(n);
        let mut by: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        by.select_nth_unstable_by(wide - 1, order);
        by.truncate(wide);
        for entry in &mut by {
            entry.1 = sqdist(u, &centroids[entry.0 * dim..(entry.0 + 1) * dim]);
        }
        by.select_nth_unstable_by(want - 1, order);
        by.truncate(want);
        by.sort_unstable_by(order);
        by.into_iter().map(|(p, _)| p).collect()
    }
}

/// The head, read in full, which is what the search actually does.
fn full_head(u: &[f32], centroids: &[f32], dim: usize, n: usize, want: usize) -> Vec<usize> {
    let order = |a: &(usize, f32), b: &(usize, f32)| a.1.total_cmp(&b.1);
    let mut by: Vec<(usize, f32)> = (0..n)
        .map(|p| (p, sqdist(u, &centroids[p * dim..(p + 1) * dim])))
        .collect();
    by.select_nth_unstable_by(want - 1, order);
    by.truncate(want);
    by.sort_unstable_by(order);
    by.into_iter().map(|(p, _)| p).collect()
}

/// The half of the idea that worked, kept so that the doc's claim about it is
/// checkable: at four bits the shortlist really does bring back the exact head.
#[test]
fn coding_the_centroids_would_have_given_the_right_answer() {
    for dim in [32usize, 128] {
        let n = 2000;
        let centroids = clumped(n, dim, 2);
        let coded = Codes::build(&centroids, dim, n, Bits::Four);
        let queries = clumped(100, dim, 3);
        let mut scores = Vec::new();
        for i in 0..100 {
            let u = &queries[i * dim..(i + 1) * dim];
            assert_eq!(
                coded.head(u, &centroids, 16, &mut scores),
                truth(u, &centroids, dim, n, 16),
                "at dimension {dim}, query {i}"
            );
        }
    }
}

/// The half that did not: what the two ways of ranking the same centroids cost,
/// measured back to back on one machine. This is the table in the module doc.
///
/// Worth running under `--release`, because reading them in full is a vectorised
/// loop and a debug build is measuring the wrong thing entirely.
#[test]
#[ignore = "prints a table rather than asserting anything"]
fn what_ranking_the_centroids_costs() {
    println!("centroids  dimensions   read in full   coded and reranked");
    for &(n, dim) in &[(2963usize, 1024usize), (2963, 128), (10_000, 768)] {
        let centroids = clumped(n, dim, 2);
        let coded = Codes::build(&centroids, dim, n, Bits::Four);
        let queries = clumped(200, dim, 3);
        let want = 128;
        let mut scores = Vec::new();
        let mut sink = 0usize;

        // A warm pass each, because the first one pays for the pages.
        for i in 0..200 {
            let u = &queries[i * dim..(i + 1) * dim];
            sink += full_head(u, &centroids, dim, n, want).len();
            sink += coded.head(u, &centroids, want, &mut scores).len();
        }
        let at = std::time::Instant::now();
        for i in 0..200 {
            sink += full_head(&queries[i * dim..(i + 1) * dim], &centroids, dim, n, want).len();
        }
        let full = at.elapsed().as_nanos() as f64 / 200.0 / 1000.0;
        let at = std::time::Instant::now();
        for i in 0..200 {
            sink += coded
                .head(
                    &queries[i * dim..(i + 1) * dim],
                    &centroids,
                    want,
                    &mut scores,
                )
                .len();
        }
        let bits = at.elapsed().as_nanos() as f64 / 200.0 / 1000.0;
        println!("{n:9} {dim:11} {full:11.1} us {bits:16.1} us  ({sink})");
    }
}

/// Why the codes would have to be four bit, which is the measurement that
/// decides whether the idea can be made cheap enough to be worth it. For a head
/// of `want`, how far down the estimate's order the true head reaches, which is
/// how wide the shortlist would have to be.
#[test]
#[ignore = "prints a table rather than asserting anything"]
fn how_much_slack_the_shortlist_needs() {
    let n = 3000;
    println!("dimensions  head of   one bit   four bit");
    for dim in [128usize, 1024] {
        let centroids = clumped(n, dim, 2);
        let queries = clumped(100, dim, 3);
        for want in [16usize, 128, 512] {
            let mut reach = Vec::new();
            for bits in [Bits::One, Bits::Four] {
                let coded = Codes::build(&centroids, dim, n, bits);
                let mut worst = 0usize;
                let mut scores = Vec::new();
                for i in 0..100 {
                    let u = &queries[i * dim..(i + 1) * dim];
                    let head = truth(u, &centroids, dim, n, want);
                    coded.estimate(u, &mut scores);
                    let mut order: Vec<usize> = (0..n).collect();
                    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
                    let mut place = vec![0usize; n];
                    for (rank, &p) in order.iter().enumerate() {
                        place[p] = rank;
                    }
                    for p in &head {
                        worst = worst.max(place[*p] + 1);
                    }
                }
                reach.push(worst);
            }
            println!("{dim:10} {want:8} {:9} {:10}", reach[0], reach[1]);
        }
    }
}
