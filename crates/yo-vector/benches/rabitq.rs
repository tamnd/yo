//! What quantising a vector costs, and what scanning a partition of codes
//! costs.
//!
//! The two numbers M6's gate rests on are here. Ingest is `≥50k` vectors per
//! second per core (G13), and encode is the whole of what a vector costs on the
//! way in once the append is discounted, so `encode/768` has to come in under 20
//! microseconds and should come in far under it. Search is `≤1 ms` p99 at recall
//! 0.95 (G12), and the scan is where nearly all of that goes, so `scan/768` is
//! the row that says whether the budget is there.
//!
//! The rows to watch:
//!
//!   - `encode` against the dimension. It is a rotation plus a pass, both linear
//!     in the dimension times `log2` of it, so it should climb a little faster
//!     than the dimension does and nothing worse.
//!   - `scan/one` against `scan/exact_one` at the same dimension. That is the
//!     popcount scan against the same estimator with the query left in floats,
//!     which is a multiply per dimension per code, and it is the whole reason a
//!     code is stored as bit planes.
//!   - `scan/one` against `scan/four`. Four bits is four times the bytes and it
//!     also meets an eight bit query rather than a four bit one, so it is eight
//!     times the popcounts and the gap should be nearer eight than four.
//!   - `query` against `scan`. The query is prepared once per partition and the
//!     scan runs once per vector in it, so the preparation should be lost in the
//!     noise at any partition size worth having. It is not, yet.
//!
//! # Where it stands
//!
//! On an M4 Max, minimum per iteration, one run:
//!
//! ```text
//! dim   scan/one   scan/exact_one   scan/four   scan/exact_four
//! 128      9.1 us         110.5 us     45.2 us          325.4 us
//! 256     10.3 us         233.8 us     65.1 us          748.4 us
//! 768     20.6 us         721.5 us    169.9 us         2738.1 us
//! ```
//!
//! So a thousand codes at 768 dimensions is 20 microseconds against a whole
//! search budget of a millisecond, and the popcount scan is 35 times the float
//! one at one bit and 16 times at four. Encode at the same dimension is about 8
//! microseconds, which is 120 thousand vectors a second on one core against a
//! target of fifty thousand, so ingest has room too.
//!
//! The `exact` rows get worse faster than the real ones do, and that is the
//! layout talking rather than the estimator. Reading a coordinate's level out of
//! bit planes means touching one plane per bit, so the float path pays four
//! loads a coordinate at four bits where it used to pay half of one. Bit planes
//! are only worth having if the scan is popcounts, and these rows are what says
//! so.
//!
//! The row that is now worth watching is `query` against `scan`. Preparing a
//! query is 9.8 microseconds at 768 dimensions and scanning a partition of a
//! thousand codes is 20.6, so the preparation is no longer lost in the noise. It
//! happens once per partition probed, because the residual is taken against that
//! partition's centroid, and a search probes tens of them. The fix is that the
//! rotation is linear, so rotating the centroids once when they are built turns
//! the per partition work into a subtraction, and it lands with the partition
//! store.
//!
//! # Reading these on a machine someone else is using
//!
//! Same rule as everywhere else here: criterion's mean picks up whatever else
//! the box is doing, so the comparable number is the minimum per iteration
//! across samples, out of `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_vector::{Bits, Coded, Quantizer};

/// `n` vectors of `dim` dimensions, unit length, which is what an embedding
/// family hands over.
fn corpus(dim: usize, n: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(0x5eed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim)
                .map(|_| (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32 - 0.5)
                .collect();
            let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
            for c in &mut v {
                *c /= len;
            }
            v
        })
        .collect()
}

/// A partition: the codes laid out end to end the way they are scanned, and
/// what goes beside each of them.
fn partition(q: &Quantizer, vs: &[Vec<f32>], centroid: &[f32]) -> (Vec<u8>, Vec<Coded>) {
    let width = q.code_bytes();
    let mut codes = vec![0u8; width * vs.len()];
    let meta = vs
        .iter()
        .enumerate()
        .map(|(i, v)| q.encode(v, centroid, &mut codes[i * width..(i + 1) * width]))
        .collect();
    (codes, meta)
}

fn bench_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("rabitq/encode");
    for dim in [128usize, 256, 768, 1536] {
        let vs = corpus(dim, 64);
        let centroid = vec![0.0f32; dim];
        for (bits, name) in [(Bits::One, "one"), (Bits::Four, "four")] {
            let q = Quantizer::new(dim, bits, 7);
            let mut code = vec![0u8; q.code_bytes()];
            g.bench_with_input(BenchmarkId::new(name, dim), &dim, |b, _| {
                let mut i = 0usize;
                b.iter(|| {
                    i = (i + 1) % vs.len();
                    black_box(q.encode(black_box(&vs[i]), &centroid, &mut code))
                });
            });
        }
    }
    g.finish();
}

fn bench_scan(c: &mut Criterion) {
    let mut g = c.benchmark_group("rabitq/scan");
    // A partition holds about the square root of the collection, so a thousand
    // codes is the shape of a partition in a million vector collection.
    let n = 1024;
    for dim in [128usize, 256, 768] {
        let vs = corpus(dim, n);
        let centroid = vec![0.0f32; dim];
        let query = corpus(dim, 1).pop().expect("one vector");
        for (bits, name) in [(Bits::One, "one"), (Bits::Four, "four")] {
            let q = Quantizer::new(dim, bits, 7);
            let (codes, meta) = partition(&q, &vs, &centroid);
            let width = q.code_bytes();
            let prepared = q.query(&query, &centroid);
            g.bench_with_input(BenchmarkId::new(name, dim), &dim, |b, _| {
                b.iter(|| {
                    let mut best = f32::INFINITY;
                    for i in 0..n {
                        let d = prepared.distance(&codes[i * width..(i + 1) * width], &meta[i]);
                        best = best.min(d);
                    }
                    black_box(best)
                });
            });
            // The same scan against a query that was never quantised, which is
            // a float multiply per dimension per code and is what the popcount
            // scan replaced. It is here so the claim about the speedup is a
            // ratio inside one run rather than a number remembered from an
            // older one.
            g.bench_with_input(
                BenchmarkId::new(format!("exact/{name}"), dim),
                &dim,
                |b, _| {
                    b.iter(|| {
                        let mut best = f32::INFINITY;
                        for i in 0..n {
                            let c =
                                prepared.cosine_exact(&codes[i * width..(i + 1) * width], &meta[i]);
                            best = best.min(c);
                        }
                        black_box(best)
                    });
                },
            );
        }
    }
    g.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut g = c.benchmark_group("rabitq/query");
    for dim in [128usize, 256, 768] {
        let query = corpus(dim, 1).pop().expect("one vector");
        let centroid = vec![0.0f32; dim];
        let q = Quantizer::new(dim, Bits::One, 7);
        g.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| black_box(q.query(black_box(&query), &centroid)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_encode, bench_scan, bench_query);
criterion_main!(benches);
