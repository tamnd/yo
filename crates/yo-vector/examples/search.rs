//! Where a search spends its time, split into the three things it does.
//!
//! SIFT1M on gamingpc says recall clears the M6 gate at 0.9598 and latency does
//! not: p50 is 1.1 ms and p99 is about 1.5 ms against a bar of 1 ms. A total
//! does not say what to fix, and the last two times the ingest path was slow the
//! breakdown found something specific that the total had hidden, so this is the
//! same measurement for the read path.
//!
//! ```text
//! cargo run --release -p yo-vector --example search
//! cargo run --release -p yo-vector --example search -- 1000000
//! ```
//!
//! A search does three things. It ranks every centroid to decide which
//! partitions to look in, it scans the postings of the partitions it picked, and
//! it rereads the surviving candidates in full precision to settle the order.
//! Only the middle one depends on `probe`, and that is what makes the split
//! measurable from outside without instrumenting the library: run the same
//! queries at several probe counts and the cost is a straight line, whose slope
//! is what one partition costs to scan and whose intercept is what ranking the
//! centroids costs. Rerank is the difference between a full search and the same
//! search stopped before the rerank, which the public API already exposes as
//! [`Partitions::candidates`].
//!
//! The vectors are synthetic, as in `examples/ingest.rs`, and for the same
//! reason: latency is a function of the partition count, the dimension and the
//! posting length, and clustered synthetic data at a hundred and twenty eight
//! dimensions gives the same shape as SIFT. Recall needs a real dataset and
//! recall is not what this looks at. `examples/recall.rs` is that one.
//!
//! # Where it stands
//!
//! Half a million vectors at 128 dimensions on an M4 Max, 1331 partitions of
//! about 376 members each, one bit codes and rerank 16:
//!
//! ```text
//!   probe     scanned  candidates      search      rerank
//!       1         376         68us        100us         32us
//!       8        3005        103us        134us         31us
//!      32       12021        223us        260us         36us
//!      64       24042        385us        421us         36us
//!     128       48084        703us        799us         95us
//!
//! ranking 1331 centroids: 63 us
//! scanning one partition of 376: 5.0 us
//! ```
//!
//! Rerank is flat, which is what it should be, because the number of candidates
//! reranked is `k` times `rerank` and has nothing to do with `probe`. So the
//! rerank is not the problem and reading the full vectors back is not the
//! problem, which was worth ruling out before anything else.
//!
//! The scan is. A one off timer inside `candidates_where` split the probe 64
//! row into 67 microseconds ranking centroids, 39 preparing one query per
//! partition, 240 scanning postings and 37 picking the best of what the scan
//! found. That timer is not in the library, because a timer on the hot path
//! costs more than some of the things it measures, and the numbers are written
//! down here instead.
//!
//! Two thirds of a search is the estimator, at 9.9 nanoseconds a member. The
//! bench in `benches/rabitq.rs` says the same thing from the other direction: a
//! thousand one bit codes at 128 dimensions is 9.1 microseconds. So the way to
//! the millisecond is to make the scan meet more than one code at a time, and
//! the preparation and the selection are each a tenth and worth having after
//! that rather than before it.

use std::time::{Duration, Instant};
use yo_common::Rng;
use yo_vector::{Bits, Partitions, Tuning, Vectors};

/// How many queries each cell of the table is measured over.
///
/// Two hundred is enough for a stable median at these times and few enough that
/// the whole table is a few seconds. The number being reported is a median
/// rather than a mean because one query that lands during a page fault is worth
/// nothing to anybody trying to work out where the milliseconds go.
const QUERIES: usize = 200;

/// The answers asked for, which sets how many candidates get reranked.
const K: usize = 10;

/// The vectors, in full precision, which is what a rerank reads.
///
/// A `Vec` rather than a log, so what this measures is the distance work and
/// not the storage. The real rerank goes through the log and will cost more, and
/// knowing what the floor is under it is the point of measuring it this way.
struct Store {
    dim: usize,
    data: Vec<f32>,
}

impl Vectors for Store {
    fn get(&self, id: u64, into: &mut [f32]) -> bool {
        let at = id as usize * self.dim;
        let Some(v) = self.data.get(at..at + self.dim) else {
            return false;
        };
        into.copy_from_slice(v);
        true
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(500_000);
    let dim = 128;
    let store = corpus(dim, n, 200, 0x9e37);

    let mut ix = Partitions::new(dim, Bits::One, 0x51f7, Tuning::default());
    let mut buf = vec![0f32; dim];
    let built = Instant::now();
    for id in 0..n as u64 {
        store.get(id, &mut buf);
        ix.insert(id, &buf);
        if ix.needs_maintenance() {
            ix.maintain(&store, 4);
        }
    }
    let built = built.elapsed();

    // Queries near the corpus rather than uniform, because a query that is
    // nowhere near anything makes every partition equally bad and the scan then
    // measures a case nobody has.
    let mut rng = Rng::new(0xbeef);
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| {
            let at = (rng.next_u64() as usize % n) * dim;
            let mut q = store.data[at..at + dim].to_vec();
            let noise = draw(dim, &mut rng);
            for (x, e) in q.iter_mut().zip(&noise) {
                *x += e * 0.08;
            }
            unit(&mut q);
            q
        })
        .collect();

    let per = n as f64 / ix.partitions() as f64;
    println!(
        "{n} vectors, {dim} dimensions, {} partitions, {per:.0} members a partition, built in {built:?}",
        ix.partitions()
    );
    println!();
    println!(
        "{:>7}{:>12}{:>12}{:>12}{:>12}",
        "probe", "scanned", "candidates", "search", "rerank"
    );

    let probes = [1usize, 2, 4, 8, 16, 32, 64, 128];
    let mut points = Vec::new();
    for probe in probes {
        ix.retune(Tuning {
            probe,
            rerank: 16,
            ..Tuning::default()
        });
        let cand = median(&queries, |q| {
            let t = Instant::now();
            let out = ix.candidates(q, K * 16);
            assert!(!out.is_empty());
            t.elapsed()
        });
        let full = median(&queries, |q| {
            let t = Instant::now();
            let out = ix.search(q, K, &store);
            assert_eq!(out.len(), K);
            t.elapsed()
        });
        // Not a subtraction of two medians pretending to be a median of
        // differences. It is close enough here because the two runs see the same
        // queries in the same order, and it is the only way to get at the rerank
        // without a timer inside the library.
        let rerank = full.saturating_sub(cand);
        println!(
            "{probe:>7}{:>12.0}{:>11}us{:>11}us{:>11}us",
            probe as f64 * per,
            cand.as_micros(),
            full.as_micros(),
            rerank.as_micros(),
        );
        points.push((probe as f64, cand.as_secs_f64() * 1e6));
    }

    // The straight line through the candidate timings. Ranking the centroids
    // happens once whatever `probe` is, so it is the intercept, and scanning a
    // partition happens `probe` times, so it is the slope.
    let (rank, each) = fit(&points);
    println!();
    println!("ranking {} centroids: {rank:.0} us", ix.partitions());
    println!("scanning one partition of {per:.0}: {each:.1} us");
    println!(
        "which puts a probe 64 search at {:.0} us before rerank",
        rank + each * 64.0
    );
}

/// The median of `f` over the queries.
///
/// A median rather than a mean because one query that lands during a page fault
/// is worth nothing to anybody trying to work out where the milliseconds go.
fn median(queries: &[Vec<f32>], mut f: impl FnMut(&[f32]) -> Duration) -> Duration {
    // A short pass thrown away, so the table is not measuring the first touch
    // of pages that every later row gets for free.
    for q in queries.iter().take(16) {
        f(q);
    }
    let mut took: Vec<Duration> = queries.iter().map(|q| f(q)).collect();
    took.sort_unstable();
    took[took.len() / 2]
}

/// Least squares through `(x, y)`, returning the intercept and the slope.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let d = n * sxx - sx * sx;
    if d == 0.0 {
        return (sy / n, 0.0);
    }
    let slope = (n * sxy - sx * sy) / d;
    ((sy - slope * sx) / n, slope)
}

/// `n` vectors on the unit sphere, gathered around `clusters` centres.
fn corpus(dim: usize, n: usize, clusters: usize, seed: u64) -> Store {
    let mut rng = Rng::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(dim, &mut rng)).collect();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let centre = &centres[i % clusters];
        let mut v = draw(dim, &mut rng);
        for (x, c) in v.iter_mut().zip(centre) {
            *x = *x * 0.35 + c;
        }
        unit(&mut v);
        data.extend_from_slice(&v);
    }
    Store { dim, data }
}

/// One gaussian vector, by Box-Muller off the uniform generator.
fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    while v.len() < dim {
        let u1 = ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        v.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if v.len() < dim {
            v.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    v
}

fn unit(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v {
            *x /= n;
        }
    }
}
