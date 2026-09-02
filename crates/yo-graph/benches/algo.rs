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
//!   - `algo/pagerank` compares ten rounds pulled against ten rounds pushed.
//!     Identical arithmetic in a different order, so the gap is entirely about
//!     which array gets written to at a random offset.
//!   - `algo/triangle count` compares the degree ordering against numbering the
//!     nodes in the order they already had. Same count, and the gap is the
//!     Ortmann and Brandes point about the ordering being the algorithm.
//!   - `algo/sssp` compares delta stepping against Dijkstra with a binary heap
//!     over the same weights from the same source. Same distances, and the gap
//!     is what settling a band at a time buys over settling one node at a time.
//!   - `algo/scc` is Tarjan on its own, because there is no slower obvious form
//!     worth timing against it. The row is there to say what strong components
//!     cost next to the weak ones two rows above.
//!   - `algo/community` puts Leiden, Louvain and label propagation next to each
//!     other on the same graph. These are three answers to one question rather
//!     than one answer written three ways, so the run also prints what each of
//!     them found and what it is worth by modularity, and the time is only half
//!     the comparison.
//!   - `algo/betweenness` is Brandes over sixty four sampled sources. There is no
//!     row for the exact form because the exact form is a search per node, which
//!     on this graph is four hours, and the sampling is the reason anybody can
//!     ask this question about a graph this size at all.
//!
//! # The graph
//!
//! R-MAT at the Graph500 probabilities, scale 18 and degree 16, which is 262144
//! nodes and 4194304 edges. That is the standard synthetic social graph and it
//! is the shape these algorithms are written for: a heavy tail, a giant
//! component, and a search whose middle levels hold nearly all of the edges. A
//! uniform random graph would flatter none of them and would not be what anybody
//! runs them on.
//!
//! Counting triangles is superlinear in the edge count, so it gets its own
//! smaller graph, scale 15 at the same degree, which is 32768 nodes, 524288
//! edges and 6665423 triangles. On the big one a single count is seven seconds,
//! which is a real number and not one worth paying twenty times on every run.
//! The community rows run on the same smaller graph for the same reason: Leiden
//! is several passes over several levels, and half a second a run is already the
//! slowest thing here.
//!
//! The shortest path rows want weights, and they get the GAP suite's: uniform
//! integers from 1 to 255, drawn from a fixed seed so both rows see the same
//! graph. Weights that all match would make delta stepping into breadth first
//! search and would prove nothing about either row.
//!
//! # Where it stands
//!
//! One core each, minimum per iteration, on an Apple M4 and on an i9-13900K.
//!
//! Over the big graph, 262144 nodes and 4194304 edges:
//!
//! | Row | M4 | 13900K | Rate on the M4 |
//! |---|---|---|---|
//! | `snapshot` | 29.58 ms | 33.55 ms | 7.1 ns an edge |
//! | `bfs/direction optimizing` | 1.89 ms | 2.14 ms | 2.21 billion edges a second |
//! | `bfs/top down` | 6.30 ms | 6.45 ms | 0.67 billion edges a second |
//! | `wcc/afforest` | 3.37 ms | 3.79 ms | 1.25 billion edges a second |
//! | `wcc/union find` | 11.28 ms | 22.16 ms | 0.37 billion edges a second |
//! | `pagerank/pull 10 rounds` | 35.42 ms | 33.65 ms | 1.18 billion edges a second a round |
//! | `pagerank/push 10 rounds` | 37.11 ms | 46.13 ms | 1.13 billion edges a second a round |
//! | `sssp/delta stepping` | 24.07 ms | 21.73 ms | 0.17 billion edges a second |
//! | `sssp/dijkstra` | 44.65 ms | 44.13 ms | 0.09 billion edges a second |
//! | `scc/tarjan` | 26.98 ms | 15.54 ms | 0.16 billion edges a second |
//! | `betweenness/64 pivots` | 805.1 ms | 903.4 ms | 12.6 ms a source |
//!
//! Over the small graph, 32768 nodes and 524288 edges:
//!
//! | Row | M4 | 13900K | What it found |
//! |---|---|---|---|
//! | `triangle count/degree ordered` | 106.9 ms | 114.9 ms | 62.4 million triangles a second |
//! | `triangle count/unordered` | 277.6 ms | 288.8 ms | 24.0 million triangles a second |
//! | `community/leiden` | 526.3 ms | 489.9 ms | 8516 communities, modularity 0.0922 |
//! | `community/louvain` | 133.5 ms | 112.9 ms | 8516 communities, modularity 0.0894 |
//! | `community/label propagation` | 13.84 ms | 12.96 ms | 8518 communities, modularity 0.0001 |
//!
//! The direction switch is worth 3.33x, the two neighbour sample is worth 3.35x
//! and the degree ordering is worth 2.60x on the M4, and 3.01x, 5.85x and 2.51x
//! on the 13900K, all on a single core with no threads anywhere in any of them.
//! The rates count every edge in the graph rather than only the ones the
//! algorithm looked at, which is the honest way round: an algorithm that skips
//! edges is supposed to get the credit for skipping them.
//!
//! Delta stepping is worth 1.85x over the heap on the M4 and 2.03x on the
//! 13900K, both at a band width of four, which is a quarter of what the paper's
//! guidance gives for these weights. That is not the paper being wrong, it is
//! the paper being about a different machine: a wide band is a way of finding
//! independent work for several cores, and on one core the redundant relaxations
//! it creates are just cost. Strong components come out at about the price of
//! seven breadth first searches, which is the honest way to think about Tarjan:
//! it is a single pass, and the pass is expensive because it is not a pass the
//! prefetcher can help with.
//!
//! # What the community rows say and do not say
//!
//! Leiden is four times the cost of Louvain here and buys 3 percent of
//! modularity for it, which is the trade the 2019 paper describes rather than an
//! argument for either one. The number worth looking at is label propagation's
//! 0.0001. It runs in a fortieth of Louvain's time and finds nothing at all, and
//! that is a fact about the graph rather than about the algorithm: R-MAT has no
//! communities planted in it, so the giant component has no seam for a label to
//! stop at and one label runs through the whole of it. On a graph with real
//! groups in it label propagation gets most of the way there for that price.
//! Modularity of 0.09 from any of the three is a low number and it should be
//! read as one.
//!
//! Pulling beats pushing by 5 percent on the M4 and by 37 percent on the
//! 13900K, which is worth a sentence because it is the same code. At 262
//! thousand nodes the score array is a megabyte, so on a machine where a random
//! write into it is about as cheap as a random read there is almost nothing in
//! it, and on a machine where the store queue is what runs out there is a lot.
//! Pull is the form to write either way, since it is never the slower one and it
//! is the one that stays correct when somebody adds threads.
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
use std::collections::BinaryHeap;
use std::hint::black_box;
use yo_common::Rng;
use yo_graph::algo::pagerank::DAMPING;
use yo_graph::algo::{
    UNREACHABLE, UNREACHED, betweenness_with, bfs, label_propagation, leiden, louvain, modularity,
    pagerank_with, scc, sssp, triangle_count, wcc,
};
use yo_graph::{Graph, NO_PROPS, Snapshot};

const LABEL: u32 = 1;
const SCALE: u32 = 18;
const DEGREE: u32 = 16;
const SMALL: u32 = 15;
const ROUNDS: u32 = 10;

/// The heaviest edge, which is what the GAP suite uses.
const HEAVIEST: u32 = 255;

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
    build_with(scale, degree, false)
}

/// The graph, optionally with the node ids shuffled first.
///
/// R-MAT hands out its highest degree nodes at its lowest ids, because that is
/// what the recursion does, so a graph left in that numbering arrives at any
/// algorithm already sorted by degree. That is not what a real graph looks like
/// and it quietly hands the win to whichever variant was going to benefit from
/// a degree order. Shuffling costs nothing, changes no answer, since none of
/// these is a question about node ids, and makes the comparison the one that
/// matters.
fn build_with(scale: u32, degree: u32, shuffle: bool) -> Graph {
    let n = 1u64 << scale;
    let mut name: Vec<u64> = (0..n).collect();
    if shuffle {
        let mut rng = Rng::new(0x5eed);
        for at in (1..n as usize).rev() {
            name.swap(at, (rng.next_u64() % (at as u64 + 1)) as usize);
        }
    }

    let mut g = Graph::new();
    for id in &name {
        g.add_node(*id).expect("a node");
    }
    for (src, dst) in rmat(scale, degree, 0x9e37) {
        g.link(name[src as usize], name[dst as usize], LABEL, NO_PROPS)
            .expect("an edge");
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

/// The same rounds of PageRank written the other way round, walking outgoing
/// edges and adding into the neighbour, as the thing pulling has to beat.
fn push(g: &Snapshot, rounds: u32) -> Vec<f32> {
    let n = g.nodes() as usize;
    let d = DAMPING;
    let mut score = vec![1.0 / n as f32; n];
    let mut next = vec![0f32; n];
    for _ in 0..rounds {
        next.fill(0.0);
        let mut stuck = 0f64;
        for (node, score) in score.iter().enumerate() {
            let out = g.out_degree(node as u32);
            if out == 0 {
                stuck += f64::from(*score);
                continue;
            }
            let share = score / out as f32;
            for to in g.out(node as u32) {
                next[*to as usize] += share;
            }
        }
        let base = ((1.0 - f64::from(d)) + f64::from(d) * stuck) / n as f64;
        for node in 0..n {
            score[node] = (base + f64::from(d) * f64::from(next[node])) as f32;
        }
    }
    score
}

/// The same triangle count with the nodes left in the order they came in,
/// which is the ordering the degree ordering has to beat.
fn triangles_unordered(g: &Snapshot) -> u64 {
    let n = g.nodes() as usize;
    let mut up: Vec<Vec<u32>> = vec![Vec::new(); n];
    for node in 0..n as u32 {
        for other in g.out(node).iter().chain(g.into_(node)) {
            if *other > node {
                up[node as usize].push(*other);
            } else if *other < node {
                up[*other as usize].push(node);
            }
        }
    }
    for mine in &mut up {
        mine.sort_unstable();
        mine.dedup();
    }
    let mut found = 0u64;
    for node in 0..n {
        for other in &up[node] {
            let (a, b) = (&up[node], &up[*other as usize]);
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => {
                        found += 1;
                        i += 1;
                        j += 1;
                    }
                }
            }
        }
    }
    found
}

/// Dijkstra with a binary heap, as the thing delta stepping has to beat.
fn dijkstra(g: &Snapshot, weights: &[u32], src: u32) -> Vec<u64> {
    let mut far = vec![UNREACHABLE; g.nodes() as usize];
    let mut todo = BinaryHeap::new();
    far[src as usize] = 0;
    todo.push((std::cmp::Reverse(0u64), src));
    while let Some((std::cmp::Reverse(at), node)) = todo.pop() {
        if at > far[node as usize] {
            continue;
        }
        let from = g.out_at(node);
        for (i, to) in g.out(node).iter().enumerate() {
            let now = at + u64::from(weights[from + i]);
            if now < far[*to as usize] {
                far[*to as usize] = now;
                todo.push((std::cmp::Reverse(now), *to));
            }
        }
    }
    far
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
    let pulled = pagerank_with(&s, DAMPING, 0.0, ROUNDS);
    let pushed = push(&s, ROUNDS);
    let apart = pulled
        .scores()
        .iter()
        .zip(&pushed)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(apart < 1e-5, "the same rounds, {apart} apart");
    // GAP's weights: uniform from 1 to 255, off a fixed seed so both shortest
    // path rows walk the same graph.
    let mut rng = Rng::new(0x5a17);
    let weights: Vec<u32> = (0..s.edges())
        .map(|_| 1 + (rng.next_u64() % u64::from(HEAVIEST)) as u32)
        .collect();
    assert_eq!(
        sssp(&s, &weights, src),
        dijkstra(&s, &weights, src),
        "the same distances"
    );
    // Counting triangles is superlinear in the edge count, so it gets its own
    // smaller graph. On the big one a single count is seven seconds, which is a
    // real number and not one worth paying twenty times over on every run.
    let small = Snapshot::of(&build_with(SMALL, DEGREE, true));
    assert_eq!(
        triangle_count(&small),
        triangles_unordered(&small),
        "the same triangles"
    );
    eprintln!(
        "the small graph has {} nodes, {} edges and {} triangles",
        small.nodes(),
        small.edges(),
        triangle_count(&small)
    );
    for (name, c) in [
        ("leiden", leiden(&small)),
        ("louvain", louvain(&small)),
        ("label propagation", label_propagation(&small)),
    ] {
        eprintln!(
            "{name} found {} communities at modularity {:.4}",
            c.count(),
            modularity(&small, c.labels())
        );
    }

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
    // A fixed count of rounds rather than a run to convergence, so the number
    // is a round and not a property of how quickly this particular graph
    // settles.
    group.bench_function("pagerank/pull 10 rounds", |b| {
        b.iter(|| black_box(pagerank_with(black_box(&s), DAMPING, 0.0, ROUNDS)));
    });
    group.bench_function("pagerank/push 10 rounds", |b| {
        b.iter(|| black_box(push(black_box(&s), ROUNDS)));
    });
    group.bench_function("betweenness/64 pivots", |b| {
        b.iter(|| black_box(betweenness_with(black_box(&s), 64)));
    });
    group.bench_function("community/leiden", |b| {
        b.iter(|| black_box(leiden(black_box(&small))));
    });
    group.bench_function("community/louvain", |b| {
        b.iter(|| black_box(louvain(black_box(&small))));
    });
    group.bench_function("community/label propagation", |b| {
        b.iter(|| black_box(label_propagation(black_box(&small))));
    });
    group.bench_function("sssp/delta stepping", |b| {
        b.iter(|| black_box(sssp(black_box(&s), black_box(&weights), src)));
    });
    group.bench_function("sssp/dijkstra", |b| {
        b.iter(|| black_box(dijkstra(black_box(&s), black_box(&weights), src)));
    });
    group.bench_function("scc/tarjan", |b| {
        b.iter(|| black_box(scc(black_box(&s))));
    });
    group.bench_function("triangle count/degree ordered", |b| {
        b.iter(|| black_box(triangle_count(black_box(&small))));
    });
    group.bench_function("triangle count/unordered", |b| {
        b.iter(|| black_box(triangles_unordered(black_box(&small))));
    });
    group.finish();
}

criterion_group!(benches, bench_algo);
criterion_main!(benches);
