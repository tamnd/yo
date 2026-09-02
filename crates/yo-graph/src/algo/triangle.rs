//! Counting triangles, in the ordered form.
//!
//! Schank and Wagner, "Finding, Counting and Listing All Triangles in Large
//! Graphs", WEA 2005, which is the `forward` algorithm, plus the degree
//! ordering from Ortmann and Brandes, "Triangle Listing Algorithms: Back from
//! the Diversion", ALENEX 2014, whose point is that most of the published
//! variants are the same algorithm under different orderings and the ordering is
//! what decides how fast they are.
//!
//! # What is being counted
//!
//! Three nodes with an edge between each pair, counted once however many ways
//! there are to walk it. Direction is ignored, a self loop is not an edge for
//! this purpose, and two parallel edges between the same pair are one edge. That
//! is the only definition that makes the answer a property of the graph rather
//! than of how it happened to be written down, and it is what every published
//! number is counting, so a number from here can be checked against one from
//! somewhere else.
//!
//! # The ordering is the whole algorithm
//!
//! The naive count walks every pair of neighbours of every node and asks whether
//! they are joined, which counts each triangle six times and spends its whole
//! life on the highest degree node in the graph.
//!
//! The ordered form gives every node a rank, and only ever looks from a node to
//! the neighbours that outrank it. A triangle then has exactly one lowest ranked
//! corner and is found exactly once, from there.
//!
//! Which way round the rank goes is the whole thing. Rank by degree with the
//! lowest first, so that every node looks up at the neighbours with more edges
//! than it has. The hub is then at the top of the order with almost nothing
//! above it, so its list is nearly empty and the work lands on the nodes that
//! have three edges each. Rank it the other way and the hub carries a list of
//! every node it touches and is intersected against all of them, which is
//! measurably worse than not ordering at all.
//!
//! On a graph where every node has about the same degree the ordering does not
//! matter and the cost is the same either way.
//!
//! # Intersecting two sorted lists
//!
//! A merge when the two are about the same length, and a binary search of the
//! long one for each member of the short one when one is more than 32 times the
//! other. A merge of a 3 element list against a 400 thousand element list reads
//! all 400 thousand, and looking up three of them costs about sixty loads, so
//! the switch is worth having and the exact ratio it happens at is not.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! for (a, b) in [(1u64, 2u64), (2, 3), (3, 1)] {
//!     g.link(a, b, 1, NO_PROPS)?;
//! }
//!
//! assert_eq!(algo::triangle_count(&Snapshot::of(&g)), 1);
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;

/// When one list is this much longer than the other, search it instead of
/// walking it.
const SKEW: usize = 32;

/// How many triangles the graph has, reading it as undirected and simple.
#[must_use]
pub fn triangle_count(g: &Snapshot) -> u64 {
    let n = g.nodes() as usize;
    if n < 3 {
        return 0;
    }

    let (at, up) = upward(g);
    let mut found = 0u64;
    for node in 0..n {
        let mine = &up[at[node] as usize..at[node + 1] as usize];
        for other in mine {
            let theirs = &up[at[*other as usize] as usize..at[*other as usize + 1] as usize];
            // Both lists hold ranks above this node, so anything in both is a
            // node joined to both ends of this edge, which is a triangle.
            found += common(mine, theirs);
        }
    }
    found
}

/// The graph as one list per node of the neighbours that outrank it, sorted,
/// with the lists themselves indexed by rank.
///
/// Everything past this point is in rank space rather than dense ids, which
/// costs a translation here and saves one on every comparison afterwards.
fn upward(g: &Snapshot) -> (Vec<u64>, Vec<u32>) {
    let n = g.nodes() as usize;

    // Lowest degree first, and the lower node first when two are the same, so
    // that the order is the graph's and not the order a hash table happened to
    // hand its nodes back in. Lowest first is what puts the hubs at the top,
    // where almost nothing outranks them and their lists come out empty.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|node| (g.out_degree(*node) + g.in_degree(*node), *node));
    let mut rank = vec![0u32; n];
    for (at, node) in order.iter().enumerate() {
        rank[*node as usize] = at as u32;
    }

    // One list per rank, holding the ranks above it. Built by hand rather than
    // as a counting sort, because the duplicates that a parallel edge and the
    // two directions of the same edge produce have to go before the offsets are
    // worked out, and a count that has to be corrected afterwards is a count
    // that is done twice.
    let mut at = vec![0u64; n + 1];
    let mut up: Vec<u32> = Vec::new();
    let mut mine: Vec<u32> = Vec::new();
    for (r, node) in order.iter().enumerate() {
        mine.clear();
        for side in [g.out(*node), g.into_(*node)] {
            for other in side {
                let other = rank[*other as usize];
                if other > r as u32 {
                    mine.push(other);
                }
            }
        }
        mine.sort_unstable();
        mine.dedup();
        up.extend_from_slice(&mine);
        at[r + 1] = up.len() as u64;
    }
    (at, up)
}

/// How many entries two ascending lists have in common.
fn common(a: &[u32], b: &[u32]) -> u64 {
    if a.len() > b.len() * SKEW {
        return search(b, a);
    }
    if b.len() > a.len() * SKEW {
        return search(a, b);
    }
    let (mut i, mut j, mut found) = (0, 0, 0);
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
    found
}

/// The same answer, for the case where `short` is very much the shorter.
///
/// Each step searches only the part of `long` that is left, because both lists
/// ascend and a member of `short` cannot be behind the one before it.
fn search(short: &[u32], long: &[u32]) -> u64 {
    let (mut from, mut found) = (0, 0);
    for want in short {
        match long[from..].binary_search(want) {
            Ok(at) => {
                found += 1;
                from += at + 1;
            }
            Err(at) => from += at,
        }
        if from >= long.len() {
            break;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use std::collections::BTreeSet;
    use yo_common::Rng;

    /// Every triple, asked about directly. Cubic and obviously right.
    fn reference(g: &Snapshot) -> u64 {
        let n = g.nodes() as usize;
        let mut near: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
        for node in 0..n as u32 {
            for other in g.out(node).iter().chain(g.into_(node)) {
                if *other != node {
                    near[node as usize].insert(*other);
                    near[*other as usize].insert(node);
                }
            }
        }
        let mut found = 0;
        for a in 0..n as u32 {
            for b in a + 1..n as u32 {
                if !near[a as usize].contains(&b) {
                    continue;
                }
                for c in b + 1..n as u32 {
                    if near[a as usize].contains(&c) && near[b as usize].contains(&c) {
                        found += 1;
                    }
                }
            }
        }
        found
    }

    fn linked(edges: &[(u64, u64)]) -> Graph {
        let mut g = Graph::new();
        for (from, to) in edges {
            g.link(*from, *to, 1, NO_PROPS).expect("an edge");
        }
        g
    }

    #[test]
    fn three_nodes_joined_up_are_one_triangle() {
        let s = Snapshot::of(&linked(&[(1, 2), (2, 3), (3, 1)]));
        assert_eq!(triangle_count(&s), 1);
    }

    #[test]
    fn a_chain_has_none() {
        let s = Snapshot::of(&linked(&[(1, 2), (2, 3), (3, 4), (4, 5)]));
        assert_eq!(triangle_count(&s), 0);
    }

    /// Five nodes all joined to each other have five choose three triangles.
    #[test]
    fn a_complete_graph_has_all_of_them() {
        let mut edges = Vec::new();
        for a in 0..5u64 {
            for b in a + 1..5 {
                edges.push((a, b));
            }
        }
        assert_eq!(triangle_count(&Snapshot::of(&linked(&edges))), 10);
    }

    #[test]
    fn which_way_the_edges_point_makes_no_difference() {
        let one = Snapshot::of(&linked(&[(1, 2), (2, 3), (3, 1)]));
        let other = Snapshot::of(&linked(&[(1, 2), (1, 3), (2, 3)]));
        assert_eq!(triangle_count(&one), triangle_count(&other));
    }

    #[test]
    fn a_self_loop_and_a_second_edge_are_not_a_triangle() {
        let s = Snapshot::of(&linked(&[(1, 1), (1, 2), (2, 1), (2, 2)]));
        assert_eq!(triangle_count(&s), 0);

        // And they do not turn one triangle into several either.
        let s = Snapshot::of(&linked(&[(1, 2), (2, 1), (2, 3), (3, 1), (3, 3)]));
        assert_eq!(triangle_count(&s), 1);
    }

    #[test]
    fn too_few_nodes_to_have_one() {
        assert_eq!(triangle_count(&Snapshot::default()), 0);
        assert_eq!(triangle_count(&Snapshot::of(&linked(&[(1, 2)]))), 0);
    }

    /// A hub joined to everything, which is the shape the degree ordering is
    /// there for, and the shape that takes the skewed intersection path.
    #[test]
    fn a_hub_over_a_ring() {
        let size = 2000u64;
        let mut edges: Vec<(u64, u64)> = (0..size).map(|i| (i, (i + 1) % size)).collect();
        edges.extend((0..size).map(|i| (size + 1, i)));
        let s = Snapshot::of(&linked(&edges));
        // Every edge of the ring closes with the hub, and the ring has as many
        // edges as it has nodes.
        assert_eq!(triangle_count(&s), u64::from(size as u32));
    }

    #[test]
    fn it_agrees_with_the_slow_one() {
        let mut rng = Rng::new(0x7a13);
        for case in 0..60 {
            let nodes = 3 + rng.next_u64() % 40;
            let edges: Vec<(u64, u64)> = (0..nodes * 4)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            assert_eq!(triangle_count(&s), reference(&s), "case {case}");
        }
    }
}
