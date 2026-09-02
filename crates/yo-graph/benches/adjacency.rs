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
//! # Where it stands
//!
//! On a 13th Gen Intel Core i9-13900K with nothing else running, criterion's
//! middle estimate:
//!
//! ```text
//! degree                     4        20       200
//! one hop              23.7 ns   36.2 ns  219.5 ns
//! link                 60.0 ns   32.1 ns         -
//! unlink              148.3 ns  161.2 ns  171.7 ns
//! ```
//!
//! ```text
//! two hops over ten million edges     serial  2.06 us
//!                                 prefetched  1.60 us
//! ```
//!
//! The one hop row is a straight line: fit it and the intercept is 19.7
//! nanoseconds and the slope is 1.0 nanoseconds an edge. That is the probe and
//! the sequential read, and it is the cost model in `11` section 4 measured
//! rather than asserted. G14 asks for a one hop under 500 nanoseconds, so the
//! degree it holds out to is about 480, which is past the degree of all but a
//! handful of nodes in any real graph.
//!
//! The two hop is 2.06 microseconds against a 50 microsecond gate, over a graph
//! of ten million edges at an average degree of twenty, which is exactly the
//! shape G14 names. Asking for the frontier's headers before reading any of
//! them takes it to 1.60, so the hint is worth 22 per cent and it earns the
//! call. It is worth being clear about why the gate is not close: the walk
//! reads 400 neighbours through 20 runs, the runs are contiguous, and the only
//! serial thing left is the 20 probes, which is what the prefetch attacks.
//!
//! Link is faster at degree 20 than at degree 4 and that is not a mistake. Both
//! rows link into 2000 nodes, so the degree 4 row is 2000 new run headers
//! against 8000 links and the degree 20 row is the same 2000 against 40000. The
//! per link cost falls because the fixed cost per node is being spread.
//!
//! Unlink is flatter against the degree than the scan in it suggests, because
//! at degree 200 the hundred comparisons that scan costs are a sequential read
//! of a run already in L1, which is about 25 nanoseconds of the 172.
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
