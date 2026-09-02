//! What a search and an insert cost once the codes are under an index.
//!
//! `benches/rabitq.rs` says what one scan costs. This says what a whole query
//! costs, which is the number M6's gate is actually written in: recall at 10 of
//! 0.95 or better with p99 at or under a millisecond (G12), and fifty thousand
//! vectors a second per core on the way in (G13).
//!
//! The rows to watch:
//!
//!   - `search` against the dimension and the collection size. It is the
//!     centroid ranking plus `probe` scans plus the rerank, and it is the row
//!     the millisecond is measured against.
//!   - `search` against `candidates` at the same shape. The gap is what rerank
//!     costs, which is a squared distance per candidate against the full
//!     precision vector, and it should stay small enough that nobody is ever
//!     tempted to turn it off.
//!   - `insert`, which is a centroid ranking plus an encode plus an append.
//!     Twenty microseconds is the whole budget at fifty thousand a second.
//!
//! # Where it stands
//!
//! On a 13th Gen Intel Core i9-13900K with nothing else running, criterion's
//! middle estimate:
//!
//! ```text
//! shape                 search    candidates    insert
//! 128 dims, 100k      139.8 us      112.4 us   22.9 us
//! 768 dims, 20k       309.5 us      284.6 us   50.5 us
//! ```
//!
//! So G12's millisecond has room: a query at 768 dimensions is under a third of
//! it, and rerank is 25 microseconds of that and buys the difference between an
//! estimate and an answer.
//!
//! G13's fifty thousand vectors a second per core is not there yet. An insert at
//! 128 dimensions is 22.9 microseconds, which is 44 thousand a second, and at
//! 768 it is 50.5, which is 20 thousand. Encode is 8 microseconds of that, so
//! most of it is the centroid ranking, which is a float distance against every
//! centroid and is linear in the number of partitions where everything else here
//! is not. At 20k vectors and a posting of 256 that is 78 centroids, and at ten
//! million it would be three thousand.
//!
//! That is one problem with one answer, and it is the same answer as everywhere
//! else in this crate: quantise the centroids and scan them with popcounts too.
//! Until then this is the row that says how far off the gate is.
//!
//! # Reading these on a machine someone else is using
//!
//! Same rule as everywhere else here: criterion's mean picks up whatever else
//! the box is doing, so the comparable number is the minimum per iteration
//! across samples, out of `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};
use yo_common::Rng;
use yo_vector::{Bits, Partitions, Tuning, Vectors};

/// The record log, for a bench: every vector by id, where the id is where it
/// sits.
struct Store(Vec<f32>, usize);

impl Vectors for Store {
    fn get(&self, id: u64, into: &mut [f32]) -> bool {
        let at = id as usize * self.1;
        match self.0.get(at..at + self.1) {
            Some(v) => {
                into.copy_from_slice(v);
                true
            }
            None => false,
        }
    }
}

impl Store {
    fn at(&self, i: usize) -> &[f32] {
        &self.0[i * self.1..(i + 1) * self.1]
    }

    fn len(&self) -> usize {
        self.0.len() / self.1
    }
}

/// Vectors with a few coordinates carrying more than their share and a cluster
/// structure over the top, which is the shape a real embedding family has and
/// the shape a partitioned index is built for.
fn corpus(dim: usize, n: usize, clusters: usize, seed: u64) -> Store {
    let mut rng = Rng::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(dim, &mut rng)).collect();
    let mut all = Vec::with_capacity(n * dim);
    for i in 0..n {
        let off = draw(dim, &mut rng);
        let mut v: Vec<f32> = centres[i % clusters]
            .iter()
            .zip(&off)
            .map(|(c, o)| c + o * 0.7)
            .collect();
        unit(&mut v);
        all.extend_from_slice(&v);
    }
    Store(all, dim)
}

fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|i| {
            let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
            let heavy = if i < dim / 16 { 6.0 } else { 1.0 };
            (u * 2.0 - 1.0) * heavy
        })
        .collect();
    unit(&mut v);
    v
}

fn unit(v: &mut [f32]) {
    let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
    for c in v {
        *c /= len;
    }
}

fn build(store: &Store) -> Partitions {
    let mut ix = Partitions::new(store.1, Bits::One, 7, Tuning::default());
    for id in 0..store.len() as u64 {
        ix.insert(id, store.at(id as usize));
        if id % 256 == 0 {
            ix.maintain(store, 4096);
        }
    }
    ix.maintain(store, 1 << 24);
    ix
}

/// The shapes: SIFT sized at 128 dimensions and an embedding sized one at 768,
/// which are the two collections G12 names.
const SHAPES: [(usize, usize); 2] = [(128, 100_000), (768, 20_000)];

/// How many ids the insert row writes before it takes them out again.
const BATCH: u64 = 1024;

fn bench_partition(c: &mut Criterion) {
    for (dim, n) in SHAPES {
        // One index per shape, shared by all three rows, because building a
        // hundred thousand vector index three times over is most of the run.
        let store = corpus(dim, n, 64, 0x5eed);
        let ix = build(&store);
        let queries = corpus(dim, 64, 64, 0xc0ffee);
        let id = BenchmarkId::new(format!("{dim}"), n);

        let mut g = c.benchmark_group("partition/search");
        g.sample_size(40);
        g.bench_with_input(id.clone(), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % queries.len();
                black_box(ix.search(black_box(queries.at(i)), 10, &store))
            });
        });
        g.finish();

        let mut g = c.benchmark_group("partition/candidates");
        g.sample_size(40);
        g.bench_with_input(id.clone(), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % queries.len();
                black_box(ix.candidates(black_box(queries.at(i)), 40))
            });
        });
        g.finish();

        // The write path, which is G13. The index is built once and then held
        // at the size it was built at: every thousand inserts, the thousand ids
        // just written are taken out again untimed. So this is the cost of an
        // insert into a collection this size, rather than the cost of the
        // collection growing under the measurement.
        let mut g = c.benchmark_group("partition/insert");
        let mut into = build(&store);
        g.bench_with_input(id, &n, |b, _| {
            b.iter_custom(|iters| {
                let mut spent = Duration::ZERO;
                for k in 0..iters {
                    let v = queries.at(k as usize % queries.len());
                    let id = n as u64 + k % BATCH;
                    let at = Instant::now();
                    into.insert(id, black_box(v));
                    spent += at.elapsed();
                    if k % BATCH == BATCH - 1 {
                        for back in 0..BATCH {
                            into.remove(n as u64 + back);
                        }
                    }
                }
                spent
            });
        });
        g.finish();
    }
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
