//! What a whole graph computation costs, and what the published algorithm buys
//! over the obvious one.
//!
//! The rows:
//!
//!   - `algo/snapshot` is flattening the adjacency plane into the dense arrays
//!     the algorithms run over. It is paid once per computation, or once per
//!     batch of them, and it is the number that says whether a snapshot is a
//!     reasonable thing to take.
//!   - `algo/bfs` compares the direction optimizing search against a plain top
//!     down one over the same graph from the same source. Same answer, and the
//!     gap is the whole reason the switch is in there.
//!   - `algo/wcc` compares Afforest against plain union find over every edge.
//!     Same answer again, and the gap is what the two neighbour sample buys.
//!
//! # The graph
//!
//! R-MAT at the Graph500 probabilities, scale 18 and degree 16, which is 262
//! thousand nodes and 4.2 million edges. That is the standard synthetic social
//! graph and it is the shape both of these algorithms are written for: a heavy
//! tail, a giant component, and a search whose middle levels hold nearly all of
//! the edges. A uniform random graph would flatter neither algorithm and would
//! not be what anybody runs them on.
//!
//! # Where it stands
//!
//! One core of an Apple M4, over the graph above, which comes out as 262144
//! nodes and 4194304 edges once it is built. Minimum per iteration.
//!
//! | Row | Time | Rate |
//! |---|---|---|
//! | `snapshot` | 29.59 ms | 7.1 ns an edge |
//! | `bfs/direction optimizing` | 1.89 ms | 2.21 billion edges a second |
//! | `bfs/top down` | 6.30 ms | 0.67 billion edges a second |
//! | `wcc/afforest` | 3.37 ms | 1.25 billion edges a second |
//! | `wcc/union find` | 11.28 ms | 0.37 billion edges a second |
//!
//! So the direction switch is worth 3.33x and the two neighbour sample is worth
//! 3.35x, both on a single core with no threads anywhere in either of them. The
//! rates count every edge in the graph rather than only the ones the algorithm
//! looked at, which is the honest way round: an algorithm that skips edges is
//! supposed to get the credit for skipping them.
//!
//! Taking the snapshot costs about as much as five breadth first searches, and
//! it is worth saying plainly that a caller who wants one search over a graph
//! that is about to change again should not be taking one. The snapshot is for
//! the case where several algorithms run over the same graph, or the same one
//! runs from many sources, which is what a graph computation usually is.
//!
//! # Reading these on a machine someone else is using
//!
//! Criterion's mean picks up whatever else the box is doing, so the comparable
//! number is the minimum per iteration across samples, out of
//! `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_graph::algo::{UNREACHED, bfs, wcc};
use yo_graph::{Graph, NO_PROPS, Snapshot};

const LABEL: u32 = 1;
const SCALE: u32 = 18;
const DEGREE: u32 = 16;

/// R-MAT with the Graph500 probabilities.
fn rmat(scale: u32, degree: u32, seed: u64) -> Vec<(u64, u64)> {
    let nodes = 1u64 << scale;
    let mut rng = Rng::new(seed);
    let mut edges = Vec::with_capacity((nodes as usize) * (degree as usize));
    for _ in 0..nodes * u64::from(degree) {
        let (mut r, mut c) = (0u64, 0u64);
        for level in 0..scale {
            let bit = 1u64 << (scale - 1 - level);
            let p = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            if p < 0.57 {
            } else if p < 0.76 {
                c |= bit;
            } else if p < 0.95 {
                r |= bit;
            } else {
                r |= bit;
                c |= bit;
            }
        }
        edges.push((r, c));
    }
    edges
}

fn build(scale: u32, degree: u32) -> Graph {
    let mut g = Graph::new();
    for id in 0..1u64 << scale {
        g.add_node(id).expect("a node");
    }
    for (src, dst) in rmat(scale, degree, 0x9e37) {
        g.link(src, dst, LABEL, NO_PROPS).expect("an edge");
    }
    g
}

/// Plain top down breadth first search, as the thing to beat.
fn top_down(g: &Snapshot, src: u32) -> Vec<u32> {
    let mut depth = vec![UNREACHED; g.nodes() as usize];
    depth[src as usize] = 0;
    let mut frontier = vec![src];
    let mut next = Vec::new();
    let mut d = 1;
    while !frontier.is_empty() {
        next.clear();
        for node in &frontier {
            for to in g.out(*node) {
                if depth[*to as usize] == UNREACHED {
                    depth[*to as usize] = d;
                    next.push(*to);
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
        d += 1;
    }
    depth
}

/// Union find over every edge, as the thing to beat.
fn union_find(g: &Snapshot) -> Vec<u32> {
    let mut of: Vec<u32> = (0..g.nodes()).collect();
    fn root(of: &mut [u32], mut at: u32) -> u32 {
        while of[at as usize] != at {
            at = of[at as usize];
        }
        at
    }
    for node in 0..g.nodes() {
        for to in g.out(node) {
            let (a, b) = (root(&mut of, node), root(&mut of, *to));
            let (high, low) = if a > b { (a, b) } else { (b, a) };
            of[high as usize] = low;
        }
    }
    for node in 0..g.nodes() {
        let r = root(&mut of, node);
        of[node as usize] = r;
    }
    of
}

fn bench_algo(c: &mut Criterion) {
    let g = build(SCALE, DEGREE);
    let s = Snapshot::of(&g);
    // The source is the highest degree node, because a search from a leaf ends
    // in two levels and measures nothing.
    let src = (0..s.nodes())
        .max_by_key(|n| s.out_degree(*n))
        .expect("a node");
    assert_eq!(bfs(&s, src), top_down(&s, src), "the same search");
    assert_eq!(wcc(&s).labels(), union_find(&s), "the same components");

    let mut group = c.benchmark_group("algo");
    group.sample_size(20);
    group.bench_function("snapshot", |b| {
        b.iter(|| black_box(Snapshot::of(black_box(&g))));
    });
    group.bench_function("bfs/direction optimizing", |b| {
        b.iter(|| black_box(bfs(black_box(&s), src)));
    });
    group.bench_function("bfs/top down", |b| {
        b.iter(|| black_box(top_down(black_box(&s), src)));
    });
    group.bench_function("wcc/afforest", |b| {
        b.iter(|| black_box(wcc(black_box(&s))));
    });
    group.bench_function("wcc/union find", |b| {
        b.iter(|| black_box(union_find(black_box(&s))));
    });
    group.finish();
}

criterion_group!(benches, bench_algo);
criterion_main!(benches);
