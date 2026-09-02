//! What a hop costs, which is the whole of G14.
//!
//! The gate is a one hop under 500 nanoseconds and a two hop over a ten million
//! edge graph under 50 microseconds. Both numbers are about dependent loads
//! rather than about arithmetic, so the rows to watch are:
//!
//!   - `graph/one hop`, against the degree. A probe for the run header and then
//!     a sequential read, so the slope of this against the degree is the read
//!     and the intercept is the probe.
//!   - `graph/two hops`, serial against prefetched, over a ten million edge
//!     graph. The frontier is known before any of it is read, so the second
//!     version issues every probe before it uses any of them. The difference
//!     between the two rows is the whole reason `Adjacency::prefetch` exists,
//!     and if there is no difference then the hint is not earning its place.
//!   - `graph/link` and `graph/unlink`, the write path. An unlink scans the run
//!     at each end, so it is priced against a degree rather than as one number.
//!
//! # Reading these on a machine someone else is using
//!
//! Criterion's mean picks up whatever else the box is doing, so the comparable
//! number is the minimum per iteration across samples, out of
//! `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_graph::{Adjacency, Dir};

const FOLLOWS: u32 = 1;

/// Ten million edges at an average degree of twenty, which is the graph G14 is
/// written against.
const NODES: u64 = 500_000;
const DEGREE: u64 = 20;

fn build(nodes: u64, degree: u64, seed: u64) -> Adjacency {
    let mut g = Adjacency::out_only();
    let mut rng = Rng::new(seed);
    for src in 0..nodes {
        for _ in 0..degree {
            g.link(src, rng.next_u64() % nodes, FOLLOWS, 0);
        }
    }
    g
}

/// A two hop with nothing told to the cache in advance: every probe waits for
/// the neighbour that named it.
fn two_hops(g: &Adjacency, from: u64, out: &mut Vec<u64>) {
    out.clear();
    for hop in g.neighbours(from, FOLLOWS, Dir::Out) {
        out.extend_from_slice(g.neighbours(*hop, FOLLOWS, Dir::Out));
    }
}

/// The same walk with the frontier's headers asked for first. The frontier is
/// known as soon as the first hop is read, so there is no reason for the second
/// hop's probes to be serial.
fn two_hops_prefetched(g: &Adjacency, from: u64, out: &mut Vec<u64>) {
    out.clear();
    let first = g.neighbours(from, FOLLOWS, Dir::Out);
    for hop in first {
        g.prefetch(*hop, FOLLOWS, Dir::Out);
    }
    for hop in first {
        out.extend_from_slice(g.neighbours(*hop, FOLLOWS, Dir::Out));
    }
}

fn bench_hops(c: &mut Criterion) {
    let g = build(NODES, DEGREE, 0x11ee);
    let mut rng = Rng::new(9);

    let mut group = c.benchmark_group("graph/one hop");
    for degree in [4u64, 20, 200] {
        let h = build(50_000, degree, 0x5eed + degree);
        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, _| {
            b.iter(|| {
                let node = rng.next_u64() % 50_000;
                // Summed rather than counted. The length is in the run header
                // and reading it never touches the run, so a version that
                // stopped at `len()` would time the probe alone and be flat
                // across the degree, which is not what this row is for.
                black_box(
                    black_box(&h)
                        .neighbours(node, FOLLOWS, Dir::Out)
                        .iter()
                        .sum::<u64>(),
                )
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("graph/two hops");
    let mut out = Vec::with_capacity(1024);
    group.bench_function("serial", |b| {
        b.iter(|| {
            two_hops(black_box(&g), rng.next_u64() % NODES, &mut out);
            black_box(out.len())
        });
    });
    group.bench_function("prefetched", |b| {
        b.iter(|| {
            two_hops_prefetched(black_box(&g), rng.next_u64() % NODES, &mut out);
            black_box(out.len())
        });
    });
    group.finish();
}

fn bench_writes(c: &mut Criterion) {
    let mut rng = Rng::new(0xfeed);

    let mut group = c.benchmark_group("graph/link");
    for degree in [4u64, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, degree| {
            // A fresh plane each batch, because linking into one that keeps
            // growing measures the arena's growth as much as the link.
            b.iter_batched_ref(
                Adjacency::out_only,
                |g| {
                    for src in 0..2000u64 {
                        for _ in 0..*degree {
                            g.link(src, rng.next_u64() % 2000, FOLLOWS, 0);
                        }
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();

    let mut group = c.benchmark_group("graph/unlink");
    for degree in [4u64, 20, 200] {
        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, degree| {
            b.iter_batched_ref(
                || {
                    let mut g = Adjacency::new();
                    for src in 0..1000u64 {
                        for i in 0..*degree {
                            g.link(src, i, FOLLOWS, 0);
                        }
                    }
                    g
                },
                |g| {
                    for src in 0..1000u64 {
                        for i in 0..*degree {
                            black_box(g.unlink(src, i, FOLLOWS));
                        }
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hops, bench_writes);
criterion_main!(benches);
