//! Ranking the centroids without reading them.
//!
//! # The intercept
//!
//! Every search ranks every centroid, because [`coarse`](crate::coarse) says
//! why picking the probe head out of an anchor tree costs more recall than it
//! saves time. That decision stands and this is not a second attempt at it. What
//! is left over from it is a fixed cost that does not depend on the probe count
//! at all: an MS-MARCO probe sweep is a straight line of about 48 microseconds a
//! partition with a 700 microsecond intercept, and the intercept is 2963
//! centroids at 1024 dimensions, which is twelve megabytes read to answer a
//! query that then reads two.
//!
//! Twelve megabytes is not a distance problem, it is a bandwidth problem. The
//! arithmetic is a few million multiply adds, which is microseconds; the reading
//! is most of the L3 cache walked from a standing start, on every query,
//! whatever the query is. So the way out is not to look at fewer centroids, it
//! is to make a centroid smaller.
//!
//! # So the centroids get the same treatment their members do
//!
//! Every centroid is quantised against the mean of all the centroids, exactly
//! the way a member is quantised against the centroid of its partition, with the
//! same [`Quantizer`] and the same estimator.
//!
//! Then the estimate is not trusted. The scan picks a shortlist several times
//! wider than the head being asked for and only those centroids are read and
//! measured exactly, so the answer is the same list of partitions in the same
//! order unless the estimator was wrong by more than the width of the shortlist.
//! That is the difference between this and the anchor layer, and it is the whole
//! reason this one is allowed on the search path: an anchor tree never looks at
//! a centroid under an anchor it did not walk, so a centroid it misses cannot be
//! recovered at any price. Here every centroid is looked at on every query. Only
//! the looking got cheaper.
//!
//! # Why the codes here are four bit when the members are one bit
//!
//! Because a centroid is a harder thing to estimate than a member, and the
//! measurement is not close. `how_much_slack_the_shortlist_needs` asks, for a
//! head of `want`, how far down the estimate's order the true head reaches,
//! which is the width the shortlist has to have. On three thousand clustered
//! centroids at 1024 dimensions:
//!
//! ```text
//! want   one bit   four bit
//!   16      1563         72
//!  128      2265        257
//!  512      2716        895
//! ```
//!
//! One bit does not rank centroids at all. The reason is the frame: a member is
//! coded against the centroid it belongs to, which is a local reference chosen
//! to be near it, while a centroid is coded against the mean of every centroid,
//! which is near nothing. Centroids are clustered, so a one bit residual from a
//! global mean mostly says which cluster, and the whole true head is inside one
//! cluster, which is exactly the distinction it cannot make. Four bits gets the
//! frame back: the true head reaches about twice the ask, so a shortlist of
//! [`WIDEN`] times the ask carries it with room to spare.
//!
//! Four bits is still small. A 1024 dimensional centroid is 512 bytes rather
//! than 4096, so the pass over 2963 of them reads 1.5 megabytes instead of
//! twelve and stays in cache.
//!
//! # Where it is not used
//!
//! Below [`FLOOR`] partitions there is nothing to save. A few hundred centroids
//! is a few microseconds to rank exactly and this would add its own pass on top
//! of that, so a small collection keeps the plain scan and so does every test
//! written before this existed.
//!
//! It is not used to place a vector either. Placement goes through the anchor
//! layer, which is cheaper still because it does not touch every centroid, and
//! the accuracy it loses there costs one vector sitting next to where it
//! belonged rather than a query missing a partition.

use crate::dist::sqdist;
use crate::rabitq::{Bits, Coded, Quantizer};

/// How many partitions there have to be before the codes are worth keeping.
///
/// The same number the anchor layer uses, for a related reason: under a few
/// hundred centroids the pass being replaced is already short enough that
/// replacing it is noise, and a collection that small is where an approximate
/// step has the least ground to be approximate over.
pub(crate) const FLOOR: usize = 256;

/// How many centroids are measured exactly for each one the caller asked for.
///
/// The estimator has to be wrong by more than this many places for the answer to
/// change. The module doc has the measurement: at four bits the true head
/// reaches about twice the ask, so two would be the edge and this is the margin
/// over it. It is a multiple rather than a constant because the caller's ask
/// grows, and a filtered search that has widened to a thousand partitions wants
/// its thousandth centroid to be about right too.
const WIDEN: usize = 4;

/// The fewest centroids measured exactly, whatever the ask multiplies out to.
///
/// A search for the single nearest partition that reranked four of them would be
/// leaning the whole answer on the estimator's top four. The measured worst case
/// for a head of sixteen is rank 72, which [`WIDEN`] alone would not cover.
const LEAST: usize = 128;

/// The most centroids measured exactly, whatever the ask multiplies out to.
///
/// Only a filtered search that is starving reaches past this, and it reaches
/// there to widen rather than to rank: it is going to scan those partitions
/// looking for members that pass, and whether its six hundredth partition is
/// really its six hundredth changes nothing about which members pass. So past
/// this the estimate's own order is handed back, and the exactly measured part
/// stays a bounded read no matter how far a search widens.
const CEILING: usize = 1024;

/// The centroids, coded against their own mean.
#[derive(Debug, Default)]
pub(crate) struct Ranker {
    /// The four bit quantiser, which shares the member quantiser's rotation
    /// because a rotation is decided by the dimension and the seed alone. That
    /// sharing is what lets an already rotated centroid and an already rotated
    /// query meet here without either being rotated again.
    quant: Option<Quantizer>,
    /// What every centroid is coded against, which is the mean of all of them.
    mean: Vec<f32>,
    /// The codes, end to end, in partition order.
    codes: Vec<u8>,
    /// What each code needs beside it, in the same order.
    meta: Vec<Coded>,
    /// How many centroids there were when `mean` was last taken.
    built: usize,
}

impl Ranker {
    /// Whether there are codes to scan.
    pub(crate) fn ready(&self) -> bool {
        !self.meta.is_empty()
    }

    /// Whether the codes should be built or rebuilt for `n` partitions.
    ///
    /// The mean moves as partitions are added and dropped, and a code measured
    /// against a mean that has moved is a worse estimate rather than a wrong
    /// one, so this is a quality knob and not a correctness one. A quarter is
    /// the same threshold the anchor layer rebuilds on and it means both of them
    /// happen on the same insert, which is one stall rather than two.
    pub(crate) fn stale(&self, n: usize) -> bool {
        if n < FLOOR {
            return self.ready();
        }
        !self.ready() || n * 4 > self.built * 5 || n * 5 < self.built * 4
    }

    /// Take the mean again and re-encode every centroid against it.
    ///
    /// `quant` is the collection's own quantiser, which is read for its
    /// dimension and its seed and is not the one the codes here are written
    /// with.
    pub(crate) fn rebuild(&mut self, quant: &Quantizer, centroids: &[f32], n: usize) {
        let dim = quant.dim();
        self.codes.clear();
        self.meta.clear();
        self.mean.clear();
        self.quant = None;
        self.built = n;
        if n < FLOOR {
            return;
        }
        let coder = Quantizer::new(dim, Bits::Four, quant.seed());
        self.mean.resize(dim, 0.0);
        for p in 0..n {
            for (m, c) in self.mean.iter_mut().zip(&centroids[p * dim..(p + 1) * dim]) {
                *m += *c;
            }
        }
        for m in &mut self.mean {
            *m /= n as f32;
        }
        let width = coder.code_bytes();
        self.codes.resize(n * width, 0);
        self.meta.reserve(n);
        for p in 0..n {
            let coded = coder.encode_rotated(
                &centroids[p * dim..(p + 1) * dim],
                &self.mean,
                &mut self.codes[p * width..(p + 1) * width],
            );
            self.meta.push(coded);
        }
        self.quant = Some(coder);
    }

    /// Code a centroid that has just been added at the end.
    pub(crate) fn added(&mut self, x: &[f32]) {
        let Some(coder) = self.quant.as_ref() else {
            return;
        };
        let width = coder.code_bytes();
        let at = self.codes.len();
        self.codes.resize(at + width, 0);
        let coded = coder.encode_rotated(x, &self.mean, &mut self.codes[at..at + width]);
        self.meta.push(coded);
    }

    /// Code a centroid that has moved, which is what a split does to the
    /// partition it splits.
    pub(crate) fn moved(&mut self, p: usize, x: &[f32]) {
        let Some(coder) = self.quant.as_ref() else {
            return;
        };
        let width = coder.code_bytes();
        self.meta[p] =
            coder.encode_rotated(x, &self.mean, &mut self.codes[p * width..(p + 1) * width]);
    }

    /// Forget the centroid at `p`, where the last one has moved into its place.
    ///
    /// The same swap the partition table itself does, for the same reason: a
    /// partition index is a position, so dropping one renames another and a code
    /// left under the old name would be handed back as somebody else.
    pub(crate) fn dropped(&mut self, p: usize) {
        let Some(coder) = self.quant.as_ref() else {
            return;
        };
        let width = coder.code_bytes();
        let last = self.meta.len() - 1;
        self.meta.swap_remove(p);
        if p != last {
            let (head, tail) = self.codes.split_at_mut(last * width);
            head[p * width..(p + 1) * width].copy_from_slice(&tail[..width]);
        }
        self.codes.truncate(last * width);
    }

    /// The `want` nearest partitions to the rotated query `u`, nearest first.
    ///
    /// `scores` is the caller's buffer so that a search does not allocate one
    /// per query, and it comes back holding the estimates, which nobody outside
    /// here has any use for.
    pub(crate) fn head(
        &self,
        u: &[f32],
        centroids: &[f32],
        want: usize,
        scores: &mut Vec<f32>,
    ) -> Vec<usize> {
        let Some(coder) = self.quant.as_ref() else {
            return Vec::new();
        };
        let dim = coder.dim();
        let n = self.meta.len();
        let want = want.min(n);
        scores.clear();
        scores.resize(n, 0.0);
        coder
            .query_rotated(u, &self.mean)
            .scan(&self.codes, &self.meta, scores);

        // A shortlist by estimate, then the truth over the shortlist. Partial
        // rather than a sort, because the order inside the shortlist is about to
        // be thrown away and replaced with the real one.
        let by_estimate = |a: &(usize, f32), b: &(usize, f32)| a.1.total_cmp(&b.1);
        let wide = (want * WIDEN).clamp(LEAST, CEILING).min(n);
        let mut by: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        by.select_nth_unstable_by(wide - 1, by_estimate);
        // Past the shortlist the estimate is the answer, so that part needs
        // putting in order but not measuring.
        if want > wide {
            by[wide..].select_nth_unstable_by(want - wide - 1, by_estimate);
            by[wide..want].sort_unstable_by(by_estimate);
        }
        by.truncate(want.max(wide));
        for entry in &mut by[..wide] {
            entry.1 = sqdist(u, &centroids[entry.0 * dim..(entry.0 + 1) * dim]);
        }
        by[..wide].sort_unstable_by(by_estimate);
        by.truncate(want);
        by.into_iter().map(|(p, _)| p).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    fn spread(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n * dim)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32)
            .collect()
    }

    /// Centroids the way a real collection makes them: a handful of broad groups
    /// with the points inside a group close together. Uniform noise has no such
    /// structure, and structure is what makes ranking centroids hard, so the
    /// tests that care about accuracy use this one.
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

    fn quant(dim: usize) -> Quantizer {
        Quantizer::new(dim, Bits::One, 0x51de_0001)
    }

    fn truth(u: &[f32], centroids: &[f32], dim: usize, n: usize, want: usize) -> Vec<usize> {
        let mut by: Vec<(usize, f32)> = (0..n)
            .map(|p| (p, sqdist(u, &centroids[p * dim..(p + 1) * dim])))
            .collect();
        by.sort_by(|a, b| a.1.total_cmp(&b.1));
        by.truncate(want);
        by.into_iter().map(|(p, _)| p).collect()
    }

    #[test]
    fn a_small_collection_gets_no_codes_at_all() {
        let dim = 8;
        let q = quant(dim);
        let mut r = Ranker::default();
        r.rebuild(&q, &spread(FLOOR - 1, dim, 1), FLOOR - 1);
        assert!(!r.ready(), "under the floor there is nothing to code");
        assert!(!r.stale(FLOOR - 1), "and nothing to rebuild");
    }

    /// The claim the whole module rests on: the head that comes back is the head
    /// an exact scan would have given, not merely a good one.
    #[test]
    fn the_head_is_the_head_an_exact_scan_would_have_given() {
        for dim in [32usize, 128] {
            let n = 2000;
            let q = quant(dim);
            let centroids = clumped(n, dim, 2);
            let mut r = Ranker::default();
            r.rebuild(&q, &centroids, n);
            assert!(r.ready());

            let queries = clumped(200, dim, 3);
            let mut scores = Vec::new();
            for i in 0..200 {
                let u = &queries[i * dim..(i + 1) * dim];
                let want = truth(u, &centroids, dim, n, 16);
                let got = r.head(u, &centroids, 16, &mut scores);
                assert_eq!(got, want, "at dimension {dim}, query {i}");
            }
        }
    }

    /// The same claim on data with no structure at all, which is a different
    /// kind of hard: every distance sits on top of every other one.
    #[test]
    fn uniform_noise_is_ranked_exactly_too() {
        let dim = 64;
        let n = 1500;
        let q = quant(dim);
        let centroids = spread(n, dim, 12);
        let mut r = Ranker::default();
        r.rebuild(&q, &centroids, n);

        let queries = spread(200, dim, 13);
        let mut scores = Vec::new();
        for i in 0..200 {
            let u = &queries[i * dim..(i + 1) * dim];
            let want = truth(u, &centroids, dim, n, 16);
            assert_eq!(r.head(u, &centroids, 16, &mut scores), want, "query {i}");
        }
    }

    /// One partition asked for is the case where a mistake is not recoverable
    /// further down, so it gets its own check.
    #[test]
    fn the_nearest_partition_is_the_nearest_partition() {
        let dim = 24;
        let n = 900;
        let q = quant(dim);
        let centroids = clumped(n, dim, 4);
        let mut r = Ranker::default();
        r.rebuild(&q, &centroids, n);

        let queries = clumped(300, dim, 5);
        let mut scores = Vec::new();
        let mut found = 0;
        for i in 0..300 {
            let u = &queries[i * dim..(i + 1) * dim];
            if r.head(u, &centroids, 1, &mut scores) == truth(u, &centroids, dim, n, 1) {
                found += 1;
            }
        }
        assert_eq!(
            found,
            300,
            "the nearest centroid was missed {} times",
            300 - found
        );
    }

    /// Past [`CEILING`] the estimate is handed back rather than the truth, so
    /// what is checked there is that the head is still the head and the tail is
    /// still nearly right, not that the tail is exact.
    #[test]
    fn an_ask_past_the_ceiling_keeps_its_head_exact() {
        let dim = 16;
        let n = 3000;
        let q = quant(dim);
        let centroids = clumped(n, dim, 14);
        let mut r = Ranker::default();
        r.rebuild(&q, &centroids, n);

        let queries = clumped(20, dim, 15);
        let mut scores = Vec::new();
        for i in 0..20 {
            let u = &queries[i * dim..(i + 1) * dim];
            let got = r.head(u, &centroids, 2000, &mut scores);
            assert_eq!(got.len(), 2000);
            assert_eq!(
                got[..256],
                truth(u, &centroids, dim, n, 256)[..],
                "the exactly measured part is not exact"
            );
            // What the estimated tail is allowed to do is lose partitions at the
            // far edge of the ask, where a partition and the one that replaced
            // it are the same distance away for any purpose. What it is not
            // allowed to do is lose one from the middle, so the check is that
            // everything well inside the ask survived.
            let inside = truth(u, &centroids, dim, n, 1500);
            let lost = inside.iter().filter(|p| !got.contains(p)).count();
            assert_eq!(lost, 0, "the tail lost {lost} partitions from the middle");
        }
    }

    /// Adding, moving and dropping have to leave the codes lined up with the
    /// centroids they describe, and the way to find out they are not is to ask
    /// for an answer that only comes out right if they are.
    #[test]
    fn the_codes_follow_the_centroids_through_every_edit() {
        let dim = 16;
        let n = 600;
        let q = quant(dim);
        let mut centroids = clumped(n, dim, 6);
        let mut r = Ranker::default();
        r.rebuild(&q, &centroids, n);
        let mut n = n;

        let extra = clumped(40, dim, 7);
        for i in 0..40 {
            let x = &extra[i * dim..(i + 1) * dim];
            centroids.extend_from_slice(x);
            r.added(x);
            n += 1;
        }

        // A centroid moved on top of another one, which is the shape a split
        // leaves behind.
        let moved: Vec<f32> = centroids[500 * dim..501 * dim].to_vec();
        centroids[11 * dim..12 * dim].copy_from_slice(&moved);
        r.moved(11, &moved);

        for p in [3usize, 0, 400] {
            let last = n - 1;
            centroids.copy_within(last * dim..(last + 1) * dim, p * dim);
            centroids.truncate(last * dim);
            r.dropped(p);
            n -= 1;
        }
        assert_eq!(r.meta.len(), n);
        assert_eq!(r.codes.len(), n * r.quant.as_ref().unwrap().code_bytes());

        let queries = clumped(200, dim, 8);
        let mut scores = Vec::new();
        let mut found = 0;
        for i in 0..200 {
            let u = &queries[i * dim..(i + 1) * dim];
            if r.head(u, &centroids, 1, &mut scores) == truth(u, &centroids, dim, n, 1) {
                found += 1;
            }
        }
        assert_eq!(
            found, 200,
            "a code and its centroid came apart somewhere in the edits"
        );
    }

    #[test]
    fn asking_for_more_partitions_than_there_are_gives_all_of_them() {
        let dim = 8;
        let n = 300;
        let q = quant(dim);
        let centroids = clumped(n, dim, 9);
        let mut r = Ranker::default();
        r.rebuild(&q, &centroids, n);
        let mut scores = Vec::new();
        let got = r.head(&centroids[..dim], &centroids, 5000, &mut scores);
        assert_eq!(got.len(), n);
        assert_eq!(got[0], 0, "a centroid is nearest to itself");
    }

    /// For a head of `want`, how far down the estimate's order the true head
    /// reaches. That distance is the slack [`WIDEN`] has to carry, and the table
    /// this prints is the one quoted in the module doc.
    #[test]
    #[ignore = "prints a table rather than asserting anything"]
    fn how_much_slack_the_shortlist_needs() {
        let n = 3000;
        for dim in [128usize, 1024] {
            for bits in [Bits::One, Bits::Four] {
                let centroids = clumped(n, dim, 2);
                let coder = Quantizer::new(dim, bits, 0x51de_0001);
                let mut mean = vec![0.0f32; dim];
                for p in 0..n {
                    for (m, c) in mean.iter_mut().zip(&centroids[p * dim..(p + 1) * dim]) {
                        *m += *c;
                    }
                }
                for m in &mut mean {
                    *m /= n as f32;
                }
                let width = coder.code_bytes();
                let mut codes = vec![0u8; n * width];
                let meta: Vec<Coded> = (0..n)
                    .map(|p| {
                        coder.encode_rotated(
                            &centroids[p * dim..(p + 1) * dim],
                            &mean,
                            &mut codes[p * width..(p + 1) * width],
                        )
                    })
                    .collect();

                let queries = clumped(100, dim, 3);
                for want in [16usize, 128, 512] {
                    let mut worst = 0usize;
                    for i in 0..100 {
                        let u = &queries[i * dim..(i + 1) * dim];
                        let head = truth(u, &centroids, dim, n, want);
                        let mut scores = vec![0.0; n];
                        coder
                            .query_rotated(u, &mean)
                            .scan(&codes, &meta, &mut scores);
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
                    println!("dim {dim:5} {bits:?} want {want:4} reaches rank {worst:5} of {n}");
                }
            }
        }
    }
}
