//! What an order key costs to make, and how that cost moves as one spot is
//! hammered.
//!
//! Two numbers matter here and they are not the same number. A push has to be
//! free, because it is the hot path and a list is mostly a queue, so `end` is
//! measured on its own and the answer should be arithmetic and nothing else. An
//! insert is allowed to cost more, because `LINSERT` is rare, but what it must
//! not do is get slower without bound as the same spot is used again and again.
//! The descent is linear in the length of the key it is descending through and
//! the key grows by a byte every eight inserts, so the cost at one spot grows
//! like the number of inserts over eight, and `between_deep` is here to show
//! that curve rather than assume it.
//!
//! # Reading these on a machine someone else is using
//!
//! The same rule as `intset.rs`: take the minimum per iteration across samples
//! out of `target/criterion/<group>/<id>/new/sample.json`, not criterion's mean,
//! because contention only ever adds time.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::orderkey::{ORDER_KEY_MAX, between, end};

/// How many inserts have already landed at the spot being measured. Eight
/// inserts to the byte, so these are keys of roughly 8, 20, 133 and 1258 bytes.
const HAMMERED: [usize; 4] = [0, 100, 1_000, 10_000];

fn depths() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &HAMMERED[..1]
    } else {
        &HAMMERED
    }
}

/// The upper neighbour after `n` inserts between the same two keys, which is the
/// state the next insert has to descend through.
fn hammered(n: usize) -> Vec<u8> {
    let lo = end(0);
    let mut hi = end(1).to_vec();
    let mut out = [0u8; ORDER_KEY_MAX];
    for _ in 0..n {
        let len = between(&lo, &hi, &mut out).expect("a variable key does not wedge");
        hi.clear();
        hi.extend_from_slice(&out[..len]);
    }
    hi
}

fn bench_push(c: &mut Criterion) {
    let mut g = c.benchmark_group("orderkey");
    g.throughput(Throughput::Elements(1));
    // A different sequence every time, so nothing folds into a constant.
    g.bench_function("end", |b| {
        let mut seq = 0i64;
        b.iter(|| {
            seq += 1;
            black_box(end(black_box(seq)))
        });
    });
    g.finish();
}

fn bench_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("orderkey_between");
    g.throughput(Throughput::Elements(1));

    for &n in depths() {
        let lo = end(0);
        let hi = hammered(n);
        let mut out = [0u8; ORDER_KEY_MAX];
        g.bench_with_input(BenchmarkId::new("between_deep", n), &n, |b, _| {
            b.iter(|| black_box(between(black_box(&lo), black_box(&hi), &mut out)));
        });
    }

    // The ordinary case, and the one a real list mostly sees: two neighbours
    // that came off the two ends and have never been subdivided.
    let (lo, hi) = (end(0), end(1));
    let mut out = [0u8; ORDER_KEY_MAX];
    g.bench_function("between_fresh", |b| {
        b.iter(|| black_box(between(black_box(&lo), black_box(&hi), &mut out)));
    });
    g.finish();
}

criterion_group!(benches, bench_push, bench_insert);
criterion_main!(benches);
