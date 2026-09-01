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
//!   - `scan/one` against `scan/four` at the same dimension. Four bits is four
//!     times the bytes, so the gap is what reading four times the memory costs
//!     and it should be well under four times, because a one bit scan is not
//!     memory bound at this size.
//!   - `query` against `scan`. The query is prepared once per partition and the
//!     scan runs once per vector in it, so the preparation should be lost in the
//!     noise at any partition size worth having.
//!
//! # Where it stands
//!
//! On an M4 Max, encode at 768 dimensions is 7.6 microseconds, which is 132
//! thousand vectors a second on one core against a target of fifty thousand, so
//! ingest has room. The scan is the problem: 923 microseconds for a thousand one
//! bit codes at 768 dimensions, which is 900 nanoseconds a vector when the whole
//! search has a millisecond. That is the reference estimator doing one float
//! multiply per dimension per code, and it is why RaBitQ's paper quantises the
//! query and scans with popcounts instead. That is the next change, and these
//! rows are what it will be measured against.
//!
//! The four bit scan being faster than the one bit scan is the same story from
//! the other end. Both do a float multiply per dimension, and the one bit path
//! adds a shift and a test per dimension to pull the bit out, so it pays more to
//! read less. Neither number is what the shape of the data can do.
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
