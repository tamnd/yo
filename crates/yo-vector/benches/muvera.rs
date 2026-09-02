//! What late interaction retrieval costs once it is one vector.
//!
//! The claim MUVERA makes is that multi vector retrieval costs an encode at
//! write time, the index that is already there, and a rerank. This prices the
//! two ends of that.
//!
//! The rows to watch:
//!
//!   - `encode/document` against the token count. This is on the write path,
//!     next to the twenty microseconds an insert already costs, so it wants to
//!     be small against that rather than small in the abstract.
//!   - `encode/query`, which is on the read path and is inside the millisecond
//!     G12 is written in, along with the search it precedes.
//!   - `chamfer`, one query against one document, times however many candidates
//!     get reranked. The whole design is that this runs on the few dozen the
//!     index kept and never on the collection, and the row says what the
//!     difference is worth.
//!
//! # Reading these on a machine someone else is using
//!
//! Same rule as everywhere else here: criterion's mean picks up whatever else
//! the box is doing, so the comparable number is the minimum per iteration
//! across samples, out of `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_vector::muvera::{Encoder, Shape, chamfer};

/// ColBERT sized tokens, and the passage lengths a late interaction model
/// actually produces.
const DIM: usize = 128;
const TOKENS: [usize; 3] = [16, 64, 128];

fn tokens(dim: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| gauss(rng)).collect();
        let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
        for c in &mut v {
            *c /= len;
        }
        out.extend_from_slice(&v);
    }
    out
}

fn gauss(rng: &mut Rng) -> f32 {
    let u1 = ((rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32).max(f32::MIN_POSITIVE);
    let u2 = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
    (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
}

fn bench_muvera(c: &mut Criterion) {
    let mut rng = Rng::new(0x5eed);
    let enc = Encoder::new(DIM, Shape::default(), 7);
    let query = tokens(DIM, 32, &mut rng);

    let mut g = c.benchmark_group("muvera/encode");
    for n in TOKENS {
        let doc = tokens(DIM, n, &mut rng);
        g.bench_with_input(BenchmarkId::new("document", n), &n, |b, _| {
            b.iter(|| black_box(enc.document(black_box(&doc))));
        });
    }
    g.bench_function("query", |b| {
        b.iter(|| black_box(enc.query(black_box(&query))));
    });
    g.finish();

    // One candidate's worth of rerank. Multiply by however many the index
    // handed back, and compare against doing it to the whole collection.
    let mut g = c.benchmark_group("muvera/chamfer");
    for n in TOKENS {
        let doc = tokens(DIM, n, &mut rng);
        g.bench_with_input(BenchmarkId::new("32 by", n), &n, |b, _| {
            b.iter(|| black_box(chamfer(black_box(&query), black_box(&doc), DIM)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_muvera);
criterion_main!(benches);
