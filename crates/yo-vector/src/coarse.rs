//! A layer over the centroids, so that finding the nearest partition is not a
//! walk over every partition.
//!
//! # Why this is here
//!
//! [`Partitions`](crate::Partitions) keeps one centroid per partition and finds
//! the nearest one by measuring against all of them. That is the right thing at
//! a hundred partitions and it is the wrong thing at four thousand, and the
//! partition count is `n / posting`, so it grows with the collection. Placing a
//! vector does one of those lookups, so building the index is quadratic in the
//! number of vectors. Measured on 128 dimensional vectors by
//! `examples/ingest.rs`, with the placement already 86 percent of the time by
//! eight hundred thousand: 34 thousand a second at two hundred thousand vectors,
//! 22 thousand at four hundred thousand and 13 thousand at eight hundred
//! thousand. Doubling the collection halves the rate, which is the shape of a
//! quadratic and is exactly what an index that never rebuilds is supposed to not
//! do.
//!
//! So there is a second level. The centroids are themselves clustered under a
//! smaller set of anchors, and a lookup measures against the anchors first and
//! then only against the centroids under the nearest few. With `A` anchors over
//! `P` centroids a lookup costs `A + k * P / A` distances instead of `P`, which
//! is smallest at `A = sqrt(P)` and is then about `2 * sqrt(k * P)`. At four
//! thousand partitions that is a few hundred distances rather than four
//! thousand.
//!
//! This is SPANN's shape and not an invention: SPANN puts an SPTAG index over
//! its posting centroids for the same reason. The difference is that one level
//! of anchors is enough here, because `P` is already about `sqrt(n)` and the
//! square root of that is small enough that scanning it is free.
//!
//! # What it costs in answers
//!
//! The layer is approximate, in the same way and for the same reason the codes
//! under it are. An exact version is possible, by keeping a radius per anchor
//! and pruning on the triangle inequality, and it does not work: at a hundred
//! and twenty eight dimensions the distances concentrate enough that almost
//! nothing prunes, which is the ordinary curse of dimensionality and is why
//! every published index at this scale is approximate.
//!
//! So it is approximate and the compensation is over collection. A lookup walks
//! anchors until it has a few hundred candidates and the caller then ranks those
//! exactly, so the answer is only wrong when the true nearest centroid sits
//! under an anchor further away than every anchor holding a few hundred near
//! misses.
//!
//! # It is only used to place a vector, and that is the whole design
//!
//! The first version of this was used everywhere a nearest centroid was wanted,
//! and it made the index ten times slower rather than faster. That is worth
//! writing down, because the reason is not obvious and it decides the shape of
//! everything here.
//!
//! With [`KEEP`] at 64 the layer looked at 75 centroids out of 619 and got the
//! nearest one wrong 11.4 percent of the time. Inside LIRE's sweep that is
//! ruinous. The sweep decides whether a member should move, and if the answer is
//! sometimes a partition that is not actually the nearest, members leak out of
//! partitions that then fall under the merge threshold, and the members they
//! leaked into push other partitions over the split threshold. It is a feedback
//! loop with nothing damping it: 199 splits became 9715 and 0 merges became
//! 9097.
//!
//! Placing a new vector has no such loop. The vector has no partition yet, so a
//! near miss costs one vector sitting in a partition next to the right one, and
//! nothing about that makes the next placement worse. That is a recall cost and
//! it is bounded and it is measurable, which the other one was not. So the sweep
//! keeps its exact comparison, which after `sweep` was made local is a
//! comparison against three centroids and cheap, and the layer is used for the
//! one thing that is left: [`Partitions::insert`](crate::Partitions::insert).
//!
//! # When it is not there
//!
//! Below [`FLOOR`] partitions there is no layer and the caller scans. A few
//! hundred centroids is a few microseconds and the layer would only add its own
//! scan on top, so the collections where this could go wrong are exactly the
//! ones where it is not used. That also means every existing test, all of which
//! are well under the floor, measures the same code path it always did.

use crate::dist::sqdist;

/// How many partitions there have to be before a layer is worth having.
///
/// A lookup through the layer costs `sqrt(P)` anchors plus [`KEEP`] centroids,
/// so it only starts saving once `P` is comfortably past `KEEP`, and just above
/// the floor it is close to a wash: at 256 partitions the shortlist is every
/// centroid there is and the anchor scan is 16 distances on top. That is
/// deliberate. The floor is not where the saving begins, it is where the layer
/// stops being pure overhead, and a small collection is a few microseconds to
/// scan either way. What the floor really buys is that the collections where an
/// approximate placement could hurt recall the most, the ones with few enough
/// partitions that each one covers a lot of ground, do not use it at all.
pub(crate) const FLOOR: usize = 256;

/// The fewest candidates a lookup collects before it stops walking anchors.
///
/// A lookup for one centroid that stopped at the first anchor would be trusting
/// the anchor layer completely, which is the arrangement that turns a coarse
/// index into a recall problem. Collecting a few hundred and ranking them
/// exactly costs a few microseconds and means the layer only has to get the
/// neighbourhood right rather than the answer.
///
/// It is 256 rather than the 64 it started at because 64 was measured and it was
/// not good enough: at 595 partitions a shortlist of 75 got the nearest centroid
/// wrong 11.4 percent of the time, where 268 got it wrong 0.013 percent of the
/// time. That is the curse of dimensionality doing what it always does, and it
/// means the saving here is the 2.2x that comes of ranking half the centroids
/// rather than the 10x that a flat layer looks like it should give on paper. The
/// number that makes it worth having anyway is that the cost stops growing:
/// `KEEP` is a constant, so a lookup is `sqrt(P) + KEEP` distances at any size,
/// where the scan it replaces is `P`.
const KEEP: usize = 256;

/// The most anchors a lookup will walk.
///
/// An anchor holds about `sqrt(P)` centroids, so filling a shortlist of [`KEEP`]
/// takes `KEEP / sqrt(P)` of them, which is five at four thousand partitions and
/// falls as the collection grows. Sixteen is well past what that needs and the
/// cap is here so that the nearest anchors can be picked into a fixed array
/// instead of sorted into one that has to be allocated. Where the cap does bite,
/// which is a collection barely over [`FLOOR`], sixteen anchors is most of the
/// layer anyway.
const WALK: usize = 16;

/// How many times a rebuild moves the anchors to the middle of what they hold.
///
/// The anchors start as a stride sample of the centroids, which is a random
/// sample because the order centroids sit in is the order partitions split in
/// and that has nothing to do with where they are. A random sample is where
/// k-means starts and not where it finishes, and finishing it matters here more
/// than it looks like it should, because an anchor in the wrong place does not
/// just cost its own accuracy, it makes its neighbours hold the centroids it
/// should have had.
///
/// Measured on the shortlist test, which is a thousand centroids of uniform
/// noise in thirty two dimensions and is the hardest case there is for this: no
/// rounds finds the true nearest centroid 362 times in 500, one finds it 378,
/// two 394, three 400 and six 407, and after that it stops moving. Three is
/// where the curve flattens.
///
/// It costs `A * P` distances a round, so a rebuild at four thousand partitions
/// is about a million of them, which is tens of milliseconds, and rebuilds
/// happen when the partition count has moved a quarter, which is every quarter
/// million inserts. Amortised that is nothing. It is a stall rather than a cost,
/// and at ten million vectors it would be a stall worth removing, which is what
/// seeding a rebuild from the anchors it already has would do.
const ROUNDS: usize = 3;

/// The centroids, clustered.
#[derive(Debug, Default)]
pub(crate) struct Coarse {
    /// The anchor points, `dim` floats each end to end.
    points: Vec<f32>,
    /// The partitions under each anchor.
    under: Vec<Vec<u32>>,
    /// Which anchor each partition is under, indexed by partition.
    owner: Vec<u32>,
    /// How many partitions there were when the anchors were last chosen.
    built: usize,
}

impl Coarse {
    /// Whether there is a layer to use.
    pub(crate) fn ready(&self) -> bool {
        !self.under.is_empty()
    }

    /// How many anchors there are, which is what a test looks at to know the
    /// layer is doing something.
    #[cfg(test)]
    pub(crate) fn anchors(&self) -> usize {
        self.under.len()
    }

    /// Whether the layer should be built or rebuilt for a collection that now
    /// has `n` partitions.
    ///
    /// Choosing anchors is `O(P * sqrt(P))`, so it cannot happen per insert. It
    /// happens when the partition count has moved by a quarter, which over a
    /// collection growing to `P` is a logarithmic number of rebuilds and works
    /// out at a small constant per insert.
    pub(crate) fn stale(&self, n: usize) -> bool {
        if n < FLOOR {
            return self.ready();
        }
        !self.ready() || n * 4 > self.built * 5 || n * 5 < self.built * 4
    }

    /// Choose anchors again for `n` centroids.
    ///
    /// The anchors start as a stride sample of the centroids, which is a random
    /// sample for the reason given on [`ROUNDS`], and then [`ROUNDS`] rounds of
    /// Lloyd move them to the middle of what they hold. That is plain k-means
    /// with a cheap seed rather than k-means++, because the seed matters much
    /// less than the rounds do when the rounds are this cheap and there are only
    /// `sqrt(n)` centres to place.
    pub(crate) fn rebuild(&mut self, centroids: &[f32], dim: usize, n: usize) {
        self.points.clear();
        self.under.clear();
        self.owner.clear();
        self.built = n;
        if n < FLOOR {
            return;
        }
        let a = (n as f64).sqrt().ceil() as usize;
        let stride = n / a;
        self.points.reserve(a * dim);
        for i in 0..a {
            let at = (i * stride).min(n - 1) * dim;
            self.points.extend_from_slice(&centroids[at..at + dim]);
        }
        self.under = vec![Vec::new(); a];
        self.owner = vec![0; n];
        self.assign(centroids, dim, n);
        for _ in 0..ROUNDS {
            self.recentre(centroids, dim);
            self.assign(centroids, dim, n);
        }
    }

    /// File every centroid under the anchor it is nearest, from scratch.
    fn assign(&mut self, centroids: &[f32], dim: usize, n: usize) {
        for list in &mut self.under {
            list.clear();
        }
        for p in 0..n {
            let x = &centroids[p * dim..(p + 1) * dim];
            let owner = self.nearest_anchor(x, dim);
            self.owner[p] = owner as u32;
            self.under[owner].push(p as u32);
        }
    }

    /// Move every anchor to the mean of what is under it.
    ///
    /// An anchor with nothing under it stays where it is rather than being moved
    /// somewhere useful. Reseeding it would be the textbook thing and it is not
    /// worth the code: an empty anchor costs one distance per lookup and never
    /// contributes a candidate, and after a round of this there are very few of
    /// them.
    fn recentre(&mut self, centroids: &[f32], dim: usize) {
        for (a, list) in self.under.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let at = &mut self.points[a * dim..(a + 1) * dim];
            at.fill(0.0);
            for &p in list {
                let x = &centroids[p as usize * dim..(p as usize + 1) * dim];
                for (s, v) in at.iter_mut().zip(x) {
                    *s += *v;
                }
            }
            let by = list.len() as f32;
            for s in at {
                *s /= by;
            }
        }
    }

    /// Take note of a partition that has just been added at index `p`.
    pub(crate) fn added(&mut self, p: usize, x: &[f32], dim: usize) {
        if !self.ready() {
            return;
        }
        debug_assert_eq!(p, self.owner.len(), "a partition is added at the end");
        let owner = self.nearest_anchor(x, dim);
        self.owner.push(owner as u32);
        self.under[owner].push(p as u32);
    }

    /// Take note of a partition that has been dropped at index `p`, where the
    /// partition that was last has moved into its place.
    ///
    /// The move is why this cannot just forget `p`. Partition indices are
    /// positions in a vector that gets a swap remove, so dropping one renames
    /// another, and an anchor list holding the old name would hand back a
    /// partition that is now somebody else.
    pub(crate) fn dropped(&mut self, p: usize) {
        if !self.ready() {
            return;
        }
        let last = self.owner.len() - 1;
        let owner = self.owner[p] as usize;
        self.under[owner].retain(|&q| q as usize != p);
        if p != last {
            let moved = self.owner[last] as usize;
            for q in &mut self.under[moved] {
                if *q as usize == last {
                    *q = p as u32;
                }
            }
            self.owner[p] = moved as u32;
        }
        self.owner.pop();
    }

    /// Take note of a centroid that has moved, which happens when a split
    /// rewrites one in place.
    pub(crate) fn moved(&mut self, p: usize, x: &[f32], dim: usize) {
        if !self.ready() {
            return;
        }
        let was = self.owner[p] as usize;
        let now = self.nearest_anchor(x, dim);
        if now == was {
            return;
        }
        self.under[was].retain(|&q| q as usize != p);
        self.under[now].push(p as u32);
        self.owner[p] = now as u32;
    }

    /// The partitions worth measuring `x` against, written into `out` unranked.
    ///
    /// Unranked because the caller is going to measure them properly anyway, and
    /// the whole job of this is to hand over a short list rather than an answer.
    pub(crate) fn shortlist(&self, x: &[f32], dim: usize, out: &mut Vec<u32>) {
        out.clear();
        // The nearest few anchors, by insertion into a fixed array rather than
        // by sorting all of them into a `Vec`. This runs on the insert path and
        // the insert path does not allocate, which the alloc gate will say out
        // loud once vectors reach the wire.
        let mut near = [(0u32, f32::INFINITY); WALK];
        let mut held = 0usize;
        for i in 0..self.under.len() {
            let d = sqdist(x, &self.points[i * dim..(i + 1) * dim]);
            if held == WALK && d >= near[WALK - 1].1 {
                continue;
            }
            let mut at = held.min(WALK - 1);
            while at > 0 && near[at - 1].1 > d {
                near[at] = near[at - 1];
                at -= 1;
            }
            near[at] = (i as u32, d);
            held = (held + 1).min(WALK);
        }
        for &(i, _) in &near[..held] {
            let list = &self.under[i as usize];
            out.extend_from_slice(list);
            // At least two anchors whatever the counts say, because the boundary
            // between the first two is exactly where a single anchor is most
            // likely to have the wrong side of the answer.
            if out.len() >= KEEP && out.len() > list.len() {
                break;
            }
        }
    }

    /// The anchor `x` is nearest to.
    fn nearest_anchor(&self, x: &[f32], dim: usize) -> usize {
        let mut best = 0;
        let mut at = f32::INFINITY;
        for i in 0..self.under.len() {
            let d = sqdist(x, &self.points[i * dim..(i + 1) * dim]);
            if d < at {
                at = d;
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// `n` centroids spread over a `dim` dimensional cube.
    fn spread(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n * dim)
            .map(|_| (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32)
            .collect()
    }

    fn built(n: usize, dim: usize, seed: u64) -> (Coarse, Vec<f32>) {
        let centroids = spread(n, dim, seed);
        let mut c = Coarse::default();
        c.rebuild(&centroids, dim, n);
        (c, centroids)
    }

    /// The whole point of the floor is that a small collection is untouched, so
    /// this is the test that says the change is inert where it is not wanted.
    #[test]
    fn a_small_collection_gets_no_layer_at_all() {
        let (c, _) = built(FLOOR - 1, 8, 1);
        assert!(!c.ready(), "under the floor there is nothing to look at");
        assert!(!c.stale(FLOOR - 1), "and nothing to rebuild");
    }

    /// Every centroid is under exactly one anchor and every anchor's list holds
    /// only centroids that exist. This is the invariant every other method has
    /// to keep, so it gets checked after each of them below.
    fn intact(c: &Coarse, n: usize) {
        assert_eq!(c.owner.len(), n, "one owner per partition");
        let mut seen = vec![0usize; n];
        for (a, list) in c.under.iter().enumerate() {
            for &p in list {
                assert!((p as usize) < n, "anchor {a} holds partition {p} of {n}");
                assert_eq!(c.owner[p as usize] as usize, a, "partition {p} disagrees");
                seen[p as usize] += 1;
            }
        }
        for (p, times) in seen.iter().enumerate() {
            assert_eq!(*times, 1, "partition {p} is under {times} anchors");
        }
    }

    #[test]
    fn every_centroid_ends_up_under_exactly_one_anchor() {
        let n = 1000;
        let (c, _) = built(n, 16, 2);
        assert!(c.ready());
        // sqrt(1000) rounded up, which is what keeps the two halves of the
        // lookup near enough the same size.
        assert_eq!(c.anchors(), 32);
        intact(&c, n);
    }

    #[test]
    fn adding_and_dropping_partitions_keeps_the_lists_straight() {
        let dim = 16;
        let n = 400;
        let (mut c, mut centroids) = built(n, dim, 3);
        let mut n = n;

        let extra = spread(50, dim, 4);
        for i in 0..50 {
            let x = &extra[i * dim..(i + 1) * dim];
            centroids.extend_from_slice(x);
            c.added(n, x, dim);
            n += 1;
        }
        intact(&c, n);

        // Drop from the middle, from the end, and from the middle again, which
        // is the order that exercises the rename and the case where there is
        // nothing to rename.
        for p in [7usize, 0, 100] {
            let last = n - 1;
            centroids.copy_within(last * dim..(last + 1) * dim, p * dim);
            centroids.truncate(last * dim);
            c.dropped(p);
            n -= 1;
            intact(&c, n);
        }
        c.dropped(n - 1);
        n -= 1;
        intact(&c, n);
    }

    #[test]
    fn a_centroid_that_moves_moves_between_anchors() {
        let dim = 16;
        let n = 500;
        let (mut c, mut centroids) = built(n, dim, 5);
        // Put centroid 3 exactly on top of anchor 9, which is somewhere else.
        let target: Vec<f32> = c.points[9 * dim..10 * dim].to_vec();
        centroids[3 * dim..4 * dim].copy_from_slice(&target);
        c.moved(3, &target, dim);
        assert_eq!(c.owner[3], 9, "it belongs to the anchor it is sitting on");
        intact(&c, n);
    }

    /// The layer is allowed to be approximate, and it is not allowed to be
    /// approximate about the neighbourhood. Over a thousand centroids, the true
    /// nearest one has to be in the shortlist almost every time, because the
    /// caller ranks the shortlist exactly and anything not in it cannot win.
    #[test]
    fn the_nearest_centroid_is_almost_always_on_the_shortlist() {
        let dim = 32;
        let n = 1000;
        let (c, centroids) = built(n, dim, 6);
        let queries = spread(500, dim, 7);
        let mut out = Vec::new();
        let mut found = 0;
        for q in 0..500 {
            let x = &queries[q * dim..(q + 1) * dim];
            let truth = (0..n)
                .min_by(|&i, &j| {
                    sqdist(x, &centroids[i * dim..(i + 1) * dim])
                        .total_cmp(&sqdist(x, &centroids[j * dim..(j + 1) * dim]))
                })
                .unwrap();
            c.shortlist(x, dim, &mut out);
            assert!(
                out.len() >= KEEP,
                "a shortlist of {} is too short",
                out.len()
            );
            assert!(
                out.len() < n,
                "a shortlist of everything is not a shortlist"
            );
            if out.contains(&(truth as u32)) {
                found += 1;
            }
        }
        // Four hundred, and the number is what it is rather than what I would
        // like it to be. Uniform noise has no clusters in it, so there is no
        // structure for the anchors to find and this is the worst this can be:
        // on the clustered corpus `examples/ingest.rs` builds, which is the
        // shape every real embedding collection has, the same layer gets the
        // nearest centroid wrong 0.013 percent of the time. The bar is set just
        // under what was measured so that a change which quietly makes it worse
        // fails here rather than in somebody's recall.
        assert!(
            found >= 390,
            "the nearest centroid was on the shortlist {found} times in 500"
        );
    }

    /// The claim that makes the layer worth having is that a lookup stops
    /// costing more as the collection grows, so the shortlist has to stay near
    /// [`KEEP`] rather than at some fraction of the partition count. Ten times
    /// as many centroids, and the same amount of work.
    #[test]
    fn a_bigger_collection_does_not_mean_a_bigger_shortlist() {
        let dim = 16;
        let x = spread(1, dim, 9);
        let mut out = Vec::new();
        let mut sizes = Vec::new();
        for n in [900usize, 9000] {
            let (c, _) = built(n, dim, 8);
            c.shortlist(&x, dim, &mut out);
            assert!(
                out.len() >= KEEP,
                "a shortlist of {} is too short",
                out.len()
            );
            sizes.push(out.len());
        }
        assert!(
            sizes[1] < sizes[0] * 2,
            "{} candidates at 900 centroids and {} at 9000",
            sizes[0],
            sizes[1]
        );
    }

    #[test]
    fn the_layer_is_rebuilt_when_the_collection_has_moved_a_quarter() {
        let (c, _) = built(1000, 8, 10);
        assert!(!c.stale(1000));
        assert!(!c.stale(1100), "a tenth is not worth a rebuild");
        assert!(c.stale(1400), "nearly a half is");
        assert!(c.stale(700), "and so is shrinking by a third");
    }
}
