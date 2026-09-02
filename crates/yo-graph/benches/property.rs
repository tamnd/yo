//! What a hop costs once there are properties behind it.
//!
//! `adjacency` prices the plane on its own, which is what G14 is written
//! against. It is not what a caller pays. A recommendation walk reads a field
//! off every edge it crosses and a field off every node it lands on, and those
//! reads are lookups in a document collection rather than a sequential run, so
//! they are a different kind of cost and they need their own rows.
//!
//! The rows:
//!
//!   - `graph/hop` compares the bare run against the same run with the edge
//!     document read and against the same run with the neighbour document read
//!     as well. The gap between the first two is what an edge property costs
//!     per edge, and the gap between the second and the third is what a node
//!     property costs, which is a probe per neighbour and so the expensive one.
//!   - `graph/two hops` is the same three, over a graph big enough that none of
//!     it is in cache.
//!   - `graph/link` is the write path with a document behind every edge, which
//!     is the number an ingest is limited by.
//!   - `graph/find` is a query that starts at an index rather than at an id,
//!     which is how a real query starts.
//!
//! # Why the node read is the one to watch
//!
//! Following a run is sequential and following a neighbour into its properties
//! is not. A node property read is a hash lookup and then a document, so a hop
//! that reads a field off each neighbour turns one dependent load into `deg` of
//! them. That is the cost `11` section 4 prices `out::<Follows>().nodes()` at,
//! and it is the reason the typed surface has to keep the two apart rather than
//! hand back hydrated nodes by default.
//!
//! # Where it stands
//!
//! On an i9-13900K, cargo 1.98.0, nothing else running:
//!
//!   - `hop/plain` is 22.1 ns at degree 4 and 33.0 ns at degree 20, which is the
//!     plane on its own and agrees with what `adjacency` says.
//!   - `hop/edge props` is 189.9 ns at degree 4 and 945.5 ns at degree 20. Take
//!     the plain run off and divide by the degree and an edge document costs
//!     42 ns at degree 4 and 45.6 ns at degree 20, so it is flat per edge, which
//!     is what a probe into a collection that fits in cache should look like.
//!   - `hop/both props` is 397.5 ns at degree 4 and 1.78 us at degree 20. The
//!     node document adds 52 ns and 41.7 ns per neighbour on top of the edge
//!     one, so the two reads cost about the same each and a hop that reads both
//!     is roughly 90 ns a neighbour against 1.6 ns a neighbour for the run.
//!   - `two hops/plain` over 200k nodes at degree 20 is 1.54 us and
//!     `two hops/prefetched` is 972 ns, a 37 percent cut for issuing the header
//!     loads before reading any of them.
//!   - `two hops/both props` is 92.0 us. That is 400 neighbours each read twice,
//!     so about 113 ns a read against the 42 to 46 ns the warm case pays, and
//!     the gap is the part of the collection that is not in cache.
//!   - `link/with props` is 10.2 ms for 40k links, so 256 ns a link against the
//!     32 to 60 ns `adjacency` measures for the plane alone. The document write
//!     is about 200 ns of that and it is what an ingest is limited by.
//!   - `find/count` is 55.3 ns, which is the index answering without touching a
//!     document.
//!   - `find/hop from each` is 163.8 us for a bucket of about 3125 nodes at
//!     degree 4, so 52 ns a node to find it and follow its run.
//!
//! The G14 gate is two hops over 10M edges under 50 us and the plane clears it
//! by a factor of thirty. A walk that reads a field off everything it touches
//! does not, at 92 us, and it is worth being plain about which of those two a
//! caller is paying for. The 113 ns a read is a dependent load into a hash
//! table, so the way down is fewer of them rather than a faster one: read the
//! edge documents for a whole run before reading any of the node documents, so
//! the probes overlap the way the prefetched row overlaps the header loads.
//! That is the same trick twice and it is why the typed surface hands back ids
//! rather than hydrating nodes on the way past.
//!
//! # Reading these on a machine someone else is using
//!
//! Criterion's mean picks up whatever else the box is doing, so the comparable
//! number is the minimum per iteration across samples, out of
//! `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_doc::{Builder, IndexKind, Key};
use yo_graph::{Dir, Graph};

const FOLLOWS: u32 = 1;

/// Half a million nodes at an average degree of twenty, which is the graph G14
/// names, scaled down to what fits beside half a million documents.
const NODES: u64 = 200_000;
const DEGREE: u64 = 20;

/// A node's properties: a name and a bucket, which is the shape a real node
/// carries and enough of it that reading one field means skipping another.
fn node_doc(id: u64) -> Vec<u8> {
    let mut b = Builder::new();
    b.begin_object().expect("open");
    b.key(b"bucket").expect("key");
    b.int((id % 16) as i64).expect("int");
    b.key(b"name").expect("key");
    b.text(&format!("n{id}")).expect("text");
    b.end_object().expect("close");
    b.finish().expect("finished").to_vec()
}

/// An edge's properties: one integer, which is what an edge usually carries.
fn edge_doc(weight: i64) -> Vec<u8> {
    let mut b = Builder::new();
    b.begin_object().expect("open");
    b.key(b"weight").expect("key");
    b.int(weight).expect("int");
    b.end_object().expect("close");
    b.finish().expect("finished").to_vec()
}

fn build(nodes: u64, degree: u64, seed: u64) -> Graph {
    let mut g = Graph::new();
    for id in 0..nodes {
        g.put_node(id, &node_doc(id)).expect("a node");
    }
    let mut rng = Rng::new(seed);
    for src in 0..nodes {
        for _ in 0..degree {
            let dst = rng.next_u64() % nodes;
            g.link(src, dst, FOLLOWS, &edge_doc((dst % 5) as i64))
                .expect("linked");
        }
    }
    g
}

/// The run alone, summed so that the read happens rather than only the probe.
fn plain(g: &Graph, from: u64) -> u64 {
    g.neighbours(from, FOLLOWS, Dir::Out).iter().sum()
}

/// The run with each edge's weight read.
fn with_edges(g: &Graph, from: u64) -> i64 {
    let mut total = 0;
    for slot in g.edge_slots(from, FOLLOWS, Dir::Out) {
        total += g
            .edge(*slot)
            .and_then(|d| d.get(b"weight"))
            .and_then(|v| v.as_int())
            .unwrap_or(0);
    }
    total
}

/// The run with each edge's weight and each neighbour's bucket read, which is
/// what a filtered recommendation walk actually does.
fn with_both(g: &Graph, from: u64) -> i64 {
    let mut total = 0;
    for (node, slot) in g.hop(from, FOLLOWS, Dir::Out) {
        let w = g
            .edge(slot)
            .and_then(|d| d.get(b"weight"))
            .and_then(|v| v.as_int())
            .unwrap_or(0);
        let b = g
            .node(node)
            .and_then(|d| d.get(b"bucket"))
            .and_then(|v| v.as_int())
            .unwrap_or(0);
        total += w * b;
    }
    total
}

fn bench_hop(c: &mut Criterion) {
    let mut rng = Rng::new(0x91ee);

    let mut group = c.benchmark_group("graph/hop");
    for degree in [4u64, 20] {
        let g = build(20_000, degree, 0x5eed + degree);
        group.bench_with_input(BenchmarkId::new("plain", degree), &degree, |b, _| {
            b.iter(|| black_box(plain(black_box(&g), rng.next_u64() % 20_000)));
        });
        group.bench_with_input(BenchmarkId::new("edge props", degree), &degree, |b, _| {
            b.iter(|| black_box(with_edges(black_box(&g), rng.next_u64() % 20_000)));
        });
        group.bench_with_input(BenchmarkId::new("both props", degree), &degree, |b, _| {
            b.iter(|| black_box(with_both(black_box(&g), rng.next_u64() % 20_000)));
        });
    }
    group.finish();
}

fn bench_two_hops(c: &mut Criterion) {
    let g = build(NODES, DEGREE, 0x11ee);
    let mut rng = Rng::new(7);

    let mut group = c.benchmark_group("graph/two hops");
    group.bench_function("plain", |b| {
        b.iter(|| {
            let from = rng.next_u64() % NODES;
            let mut total = 0u64;
            for hop in g.neighbours(from, FOLLOWS, Dir::Out) {
                total = total.wrapping_add(plain(&g, *hop));
            }
            black_box(total)
        });
    });
    group.bench_function("prefetched", |b| {
        b.iter(|| {
            let from = rng.next_u64() % NODES;
            let first = g.neighbours(from, FOLLOWS, Dir::Out);
            for hop in first {
                g.prefetch(*hop, FOLLOWS, Dir::Out);
            }
            let mut total = 0u64;
            for hop in first {
                total = total.wrapping_add(plain(&g, *hop));
            }
            black_box(total)
        });
    });
    group.bench_function("both props", |b| {
        b.iter(|| {
            let from = rng.next_u64() % NODES;
            let mut total = 0i64;
            for hop in g.neighbours(from, FOLLOWS, Dir::Out).to_vec() {
                total = total.wrapping_add(with_both(&g, hop));
            }
            black_box(total)
        });
    });
    group.finish();
}

fn bench_link(c: &mut Criterion) {
    let mut rng = Rng::new(0xfeed);
    let props = edge_doc(3);

    let mut group = c.benchmark_group("graph/link");
    group.bench_function("with props", |b| {
        // A fresh graph each batch, because linking into one that keeps growing
        // measures the collection's growth as much as the link.
        b.iter_batched_ref(
            Graph::new,
            |g| {
                for src in 0..2000u64 {
                    for _ in 0..DEGREE {
                        g.link(src, rng.next_u64() % 2000, FOLLOWS, &props)
                            .expect("linked");
                    }
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_find(c: &mut Criterion) {
    // The other half of a query: it starts at a property, not at an id.
    let mut g = build(50_000, 4, 0x1234);
    g.index_nodes("$.bucket", IndexKind::Equality)
        .expect("indexed");
    let mut rng = Rng::new(3);

    let mut group = c.benchmark_group("graph/find");
    group.bench_function("count", |b| {
        b.iter(|| {
            let bucket = (rng.next_u64() % 16) as i64;
            black_box(g.count_nodes("$.bucket", &Key::int(bucket)).unwrap_or(0))
        });
    });
    group.bench_function("hop from each", |b| {
        b.iter(|| {
            let bucket = (rng.next_u64() % 16) as i64;
            let mut ids = Vec::with_capacity(4096);
            g.find_nodes("$.bucket", &Key::int(bucket), |id, _| ids.push(id))
                .expect("found");
            let mut total = 0u64;
            for id in &ids {
                total = total.wrapping_add(plain(&g, *id));
            }
            black_box(total)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_hop, bench_two_hops, bench_link, bench_find);
criterion_main!(benches);
