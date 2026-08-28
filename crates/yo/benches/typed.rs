//! What the typed layer costs.
//!
//! G6 wants a point read under 150 ns in process, and the honest question about
//! a typed handle is what it adds on top of the map underneath it. So every
//! group here has a `raw/` row measuring [`RawMap`] directly and a `typed/` row
//! measuring the same operation through `Map<K, V>`. The difference between the
//! two is the whole cost of the shape, the handle and the encoding, and it is
//! meant to be nothing.
//!
//! `get` and `with` are separate rows on purpose. `get` allocates the value,
//! `with` does not, and Y29 is the claim that the second one is available
//! whenever a caller wants it rather than being the only way to read.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo::{MEMORY, Map};
use yo_index::RawMap;

fn key(i: usize) -> String {
    format!("key:{i:012}")
}

/// The same cache hierarchy sizes the index benchmark uses: L2, L3, and DRAM.
const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &SIZES[..1]
    } else {
        &SIZES
    }
}

fn filled(n: usize) -> (RawMap, Map<String, u64>) {
    let mut raw = RawMap::new();
    let db = yo::open(MEMORY).expect("in memory always opens");
    let typed = db.map::<String, u64>("hits").expect("a fresh name");
    for i in 0..n {
        let k = key(i);
        raw.set(k.as_bytes(), &(i as u64).to_le_bytes());
        typed
            .set(k.as_str(), &(i as u64))
            .expect("room for a record");
    }
    (raw, typed)
}

fn bench_read(c: &mut Criterion) {
    let mut g = c.benchmark_group("read");
    for &n in sizes() {
        let (raw, typed) = filled(n);
        let keys: Vec<String> = (0..64).map(|i| key(i * (n / 64).max(1))).collect();
        g.throughput(Throughput::Elements(1));

        let mut at = 0usize;
        g.bench_with_input(BenchmarkId::new("raw", n), &n, |b, _| {
            b.iter(|| {
                at = (at + 1) & 63;
                black_box(raw.get(keys[at].as_bytes()))
            });
        });

        g.bench_with_input(BenchmarkId::new("typed/with", n), &n, |b, _| {
            b.iter(|| {
                at = (at + 1) & 63;
                black_box(typed.with(keys[at].as_str(), |v| v))
            });
        });

        g.bench_with_input(BenchmarkId::new("typed/get", n), &n, |b, _| {
            b.iter(|| {
                at = (at + 1) & 63;
                black_box(typed.get(keys[at].as_str()))
            });
        });
    }
    g.finish();
}

fn bench_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("write");
    let n = sizes()[0];
    let keys: Vec<String> = (0..n).map(key).collect();
    g.throughput(Throughput::Elements(1));

    g.bench_function("raw", |b| {
        let mut raw = RawMap::new();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % n;
            raw.set(keys[i].as_bytes(), &(i as u64).to_le_bytes());
        });
    });

    g.bench_function("typed", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let typed = db.map::<String, u64>("hits").expect("a fresh name");
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % n;
            typed.set(keys[i].as_str(), &(i as u64)).expect("room");
        });
    });

    g.finish();
}

criterion_group!(benches, bench_read, bench_write);
criterion_main!(benches);
