//! What the cold form costs to read, which is the other half of the trade the
//! size buys.
//!
//! [`Csr`](yo_graph::Csr) is about an order of magnitude smaller than the hot
//! plane and the question this answers is what that costs a traversal. A cold
//! hop is one probe into the offset table and then a decode, and the decode is
//! a shift and a mask an edge with no branch in it, so the rows to watch are:
//!
//!   - `graph/cold one hop`, against the degree, which is the decode.
//!   - `graph/hop`, cold against hot over the same graph. This is the row the
//!     promotion decision in `11` section 2 turns on: if cold reading is two or
//!     three times a hot read then a graph being walked hard wants promoting,
//!     and if it is thirty then the cold form is an archive rather than a tier.
//!   - `graph/cold degree`, which does not decode anything, so it should be two
//!     loads whatever the degree is.
//!   - `graph/cold build`, the encoder, which runs on a sweep rather than on
//!     the write path and only has to be quick enough not to be noticed.
//!
//! # Where it stands
//!
//! On a 13th Gen Intel Core i9-13900K with nothing else running, criterion's
//! middle estimate:
//!
//! ```text
//! degree                     4        20       200
//! cold one hop         36.6 ns   65.7 ns  414.6 ns
//! cold degree          13.6 ns         -   17.0 ns
//! ```
//!
//! The one hop row is a straight line again: fit it and the intercept is 28.6
//! nanoseconds and the slope is 1.93 nanoseconds an edge. Against the hot
//! plane's 19.7 and 1.0, the probe costs 9 nanoseconds more because it is two
//! dependent loads rather than one, and the decode is a little under twice a
//! sequential read of a `u64` array. Two shifts, a mask and an add an edge, for
//! a tenth of the memory, is the whole trade and it is a good one.
//!
//! The degree row is flat, 13.6 at degree 4 and 17.0 at degree 200, because it
//! reads the offset and the degree and stops. It is faster than the hop's own
//! intercept for the same reason: it never touches the gaps.
//!
//! ```text
//! over ten million edges                 hot       cold
//! one hop                            79.3 ns   103.3 ns
//! two hops, serial                    2.06 us   1.90 us
//! two hops, prefetched                1.60 us   1.71 us
//! ```
//!
//! This is the comparison the promotion decision in `11` section 2 turns on and
//! it does not say what I expected it to. A single cold hop is 1.30 times a hot
//! one, so the cold form is a tier rather than an archive and there is no size
//! of graph where reading it is out of the question.
//!
//! The two hop rows are more interesting than that. Serial, the cold form is
//! quicker than the hot one, 1.90 microseconds against 2.06, because the graph
//! is a tenth of the size and the walk is waiting on memory rather than on
//! arithmetic. Prefetched, it is slower, 1.71 against 1.60, because once the
//! misses are hidden there is nothing left for the decode to hide behind. So
//! the answer to when a graph wants promoting is not about the walk being hot,
//! it is about whether the walk is issuing its probes ahead of itself, and a
//! walk that is not should arguably stay cold. That is worth writing into O-G2
//! rather than guessing at.
//!
//! Both are far under the 50 microsecond gate in G14 either way.
//!
//! The one hop rows here are twice the ones in the degree table above because
//! this graph is ten million edges rather than one million and does not fit in
//! cache. That difference, 36.6 against 79.3 for the hot plane at the same
//! degree, is the cache miss and nothing else.
//!
//! ```text
//! order_by_degree over ten million edges   35.7 ms
//! build over ten million edges            355.6 ms
//! ```
//!
//! Both run on a sweep rather than on the write path. The encoder is 28 million
//! edges a second, most of which is the sort.
//!
//! # Reading these on a machine someone else is using
//!
//! Criterion's mean picks up whatever else the box is doing, so the comparable
//! number is the minimum per iteration across samples, out of
//! `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_graph::{Adjacency, Csr, Dir, csr};

const FOLLOWS: u32 = 1;

/// Ten million edges at an average degree of twenty, the graph G14 is written
/// against and the same one the hot bench uses.
const NODES: u32 = 500_000;
const DEGREE: u32 = 20;

fn edges(nodes: u32, degree: u32, seed: u64) -> Vec<(u32, u32)> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(nodes as usize * degree as usize);
    for src in 0..nodes {
        for _ in 0..degree {
            out.push((src, (rng.next_u64() % u64::from(nodes)) as u32));
        }
    }
    out
}

fn bench_read(c: &mut Criterion) {
    let mut rng = Rng::new(0xc01d);

    let mut group = c.benchmark_group("graph/cold one hop");
    for degree in [4u32, 20, 200] {
        let cold = Csr::build(
            50_000,
            &mut edges(50_000, degree, 0x5eed + u64::from(degree)),
        );
        let mut out = Vec::with_capacity(1024);
        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, _| {
            b.iter(|| {
                let node = (rng.next_u64() % 50_000) as u32;
                black_box(&cold).neighbours_into(node, &mut out);
                black_box(out.iter().sum::<u32>())
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("graph/cold degree");
    for degree in [4u32, 200] {
        let cold = Csr::build(
            50_000,
            &mut edges(50_000, degree, 0x5eed + u64::from(degree)),
        );
        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, _| {
            b.iter(|| black_box(&cold).degree((rng.next_u64() % 50_000) as u32));
        });
    }
    group.finish();

    // The same graph in both forms, so the two rows are comparable to each
    // other rather than only to themselves.
    let list = edges(NODES, DEGREE, 0x11ee);
    let cold = Csr::build(NODES, &mut list.clone());
    let mut hot = Adjacency::out_only();
    for (s, d) in &list {
        hot.link(u64::from(*s), u64::from(*d), FOLLOWS, 0);
    }
    hot.compact();

    let mut group = c.benchmark_group("graph/hop");
    let mut out = Vec::with_capacity(1024);
    group.bench_function("hot", |b| {
        b.iter(|| {
            let node = rng.next_u64() % u64::from(NODES);
            black_box(
                black_box(&hot)
                    .neighbours(node, FOLLOWS, Dir::Out)
                    .iter()
                    .sum::<u64>(),
            )
        });
    });
    group.bench_function("cold", |b| {
        b.iter(|| {
            let node = (rng.next_u64() % u64::from(NODES)) as u32;
            black_box(&cold).neighbours_into(node, &mut out);
            black_box(out.iter().sum::<u32>())
        });
    });
    group.finish();

    // A two hop over the same ten million edges, which is the number G14 names.
    let mut group = c.benchmark_group("graph/cold two hops");
    let mut hop = Vec::with_capacity(64);
    group.bench_function("serial", |b| {
        b.iter(|| {
            let node = (rng.next_u64() % u64::from(NODES)) as u32;
            cold.neighbours_into(node, &mut hop);
            let mut n = 0usize;
            for next in &hop {
                cold.neighbours_into(*next, &mut out);
                n += out.len();
            }
            black_box(n)
        });
    });
    group.bench_function("prefetched", |b| {
        b.iter(|| {
            let node = (rng.next_u64() % u64::from(NODES)) as u32;
            cold.neighbours_into(node, &mut hop);
            for next in &hop {
                cold.prefetch(*next);
            }
            let mut n = 0usize;
            for next in &hop {
                cold.neighbours_into(*next, &mut out);
                n += out.len();
            }
            black_box(n)
        });
    });
    group.finish();

    // What the ordering pass is worth, in the only unit that matters for it.
    let mut group = c.benchmark_group("graph/cold order");
    group.bench_function("degree", |b| {
        b.iter(|| {
            let to = csr::order_by_degree(NODES, black_box(&list));
            black_box(to[0])
        });
    });
    group.finish();

    let mut group = c.benchmark_group("graph/cold build");
    group.bench_function("ten million edges", |b| {
        b.iter_batched_ref(
            || list.clone(),
            |l| black_box(Csr::build(NODES, l).edges()),
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_read);
criterion_main!(benches);
