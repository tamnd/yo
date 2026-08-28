//! The index budget.
//!
//! M0's exit gate is a probe at or under 4 ns, and aki's raw map numbers as the
//! comparison row: Get 46.5 ns at one core, Set 49.7 ns. Two separate things
//! are measured here on purpose.
//!
//! `bucket/*` is the prefilter alone, in cache, with no memory system in the
//! way. That is the number the 4 ns gate is about and it should be flat in the
//! number of entries.
//!
//! `map/*` is a real lookup including the record fetch, so it carries two cache
//! misses at large sizes and will not be flat. That is the number that compares
//! to aki, and comparing it to the gate is a category error.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_index::{Bucket, RawMap};

fn key(i: usize) -> Vec<u8> {
    format!("key:{i:012}").into_bytes()
}

/// Sizes chosen against the cache hierarchy rather than round numbers. At 1e3
/// the whole map is in L2, at 1e5 it is around L3, and at 1e7 every probe is a
/// trip to DRAM.
const SIZES: [usize; 4] = [1_000, 100_000, 1_000_000, 10_000_000];

/// The sizes to actually run.
///
/// CI runs every benchmark once to check that it still builds and still runs,
/// and gets nothing out of filling a ten million key map to look something up
/// in it a single time. `YO_BENCH_SMOKE` is that run. Anything measuring
/// anything leaves it unset.
fn sizes() -> &'static [usize] {
    if smoke() { &SIZES[..1] } else { &SIZES }
}

/// The sizes the fill benchmark builds a whole map for, which is the most
/// expensive thing in this file and the first thing a smoke run should drop.
fn fill_sizes() -> &'static [usize] {
    const FILL: [usize; 2] = [100_000, 1_000_000];
    if smoke() { &FILL[..0] } else { &FILL }
}

fn smoke() -> bool {
    std::env::var_os("YO_BENCH_SMOKE").is_some()
}

fn bench_bucket(c: &mut Criterion) {
    let mut g = c.benchmark_group("bucket");

    // A full bucket, so the SWAR compare has the most work it will ever have.
    let mut b = Bucket::EMPTY;
    for i in 0..yo_index::SLOTS {
        b.set(
            i,
            (i as u8) + 1,
            yo_common::Addr::new(yo_common::Space::Arena, (i as u64) * 64),
        );
    }

    g.bench_function("match_hit", |bench| {
        bench.iter(|| {
            let mut n = 0usize;
            for slot in black_box(&b).match_tag(black_box(4u8)) {
                n += slot;
            }
            n
        })
    });

    g.bench_function("match_miss", |bench| {
        bench.iter(|| {
            let mut n = 0usize;
            for slot in black_box(&b).match_tag(black_box(200u8)) {
                n += slot;
            }
            n
        })
    });

    g.finish();
}

fn bench_map(c: &mut Criterion) {
    let mut g = c.benchmark_group("map");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let mut m = RawMap::new();
        for i in 0..n {
            m.set(&key(i), b"0123456789abcdef0123456789abcdef");
        }
        let probes: Vec<Vec<u8>> = (0..1024).map(|i| key(i * 7919 % n)).collect();

        g.bench_with_input(BenchmarkId::new("get_hit", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(m.get(black_box(&probes[i])))
            })
        });

        let misses: Vec<Vec<u8>> = (0..1024).map(|i| key(n + i)).collect();
        g.bench_with_input(BenchmarkId::new("get_miss", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(m.get(black_box(&misses[i])))
            })
        });
    }

    // Set is measured on a fresh map per batch, because a set into a map that
    // is already at steady state is a different operation from a set that grows
    // the index, and mixing them gives a number that describes neither.
    for &n in fill_sizes() {
        g.bench_with_input(BenchmarkId::new("set_fill", n), &n, |bench, &n| {
            let keys: Vec<Vec<u8>> = (0..n).map(key).collect();
            bench.iter_batched_ref(
                RawMap::new,
                |m| {
                    for k in &keys {
                        m.set(black_box(k), b"0123456789abcdef0123456789abcdef");
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    // Overwrite in place at steady state, which is the SET a Redis client
    // actually issues most of the time.
    for &n in sizes() {
        let mut m = RawMap::new();
        for i in 0..n {
            m.set(&key(i), b"0123456789abcdef0123456789abcdef");
        }
        let probes: Vec<Vec<u8>> = (0..1024).map(|i| key(i * 7919 % n)).collect();
        g.bench_with_input(BenchmarkId::new("set_over", n), &n, |bench, _| {
            let mut i = 0usize;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(m.set(black_box(&probes[i]), b"0123456789abcdef0123456789abcdef"))
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_bucket, bench_map);
criterion_main!(benches);
