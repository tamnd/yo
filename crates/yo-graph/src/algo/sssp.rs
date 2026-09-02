//! Single source shortest paths, by delta stepping.
//!
//! Meyer and Sanders, "Delta-stepping: a parallelizable shortest path
//! algorithm", Journal of Algorithms 2003, which is the algorithm the GAP suite
//! measures for this problem.
//!
//! # Why not Dijkstra
//!
//! Dijkstra settles one node at a time, always the closest one left, which is
//! why it needs a heap and why every step is a dependent chain: pop, look, push,
//! pop. It is optimal in the number of nodes it settles and it is slow, because
//! a heap of ten million nodes is a random walk through memory and there is
//! nothing else for the machine to do while it waits.
//!
//! Delta stepping settles a band at a time. Nodes are put in buckets of width
//! `delta` by their tentative distance, and the whole of the lowest non-empty
//! bucket is worked on at once. Inside a band the order does not matter, so the
//! reads are independent and the machine can have a lot of them in flight, and
//! the cost of that is doing some work twice when a node is improved after it
//! has already been looked at.
//!
//! # Light and heavy
//!
//! An edge lighter than `delta` can move a node within the current band or into
//! the next one, so relaxing it can put work back into the bucket that is being
//! emptied. An edge heavier than `delta` always lands further away than the
//! current band, so it can never do that.
//!
//! That is the split the algorithm turns on. The current bucket is emptied over
//! and over using only the light edges, until nothing more lands in it, and then
//! the heavy edges of everything that came out of it are relaxed once. Without
//! the split a heavy edge would be relaxed every time round the inner loop for
//! no possible gain.
//!
//! # Picking delta
//!
//! Too small and it is Dijkstra with extra bookkeeping, one node a bucket. Too
//! large and it is Bellman-Ford, one bucket for the whole graph and every edge
//! relaxed over and over. The paper's guidance works out, for integer weights,
//! as the largest weight over the average degree, which is what [`sssp`] uses
//! when it is not told otherwise.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! for (a, b) in [(1u64, 2u64), (2, 3), (1, 3)] {
//!     g.link(a, b, 1, NO_PROPS)?;
//! }
//!
//! // Every edge weighs one, since no edge has a `cost` field on it.
//! let (s, w) = Snapshot::weighted(&g, &[1], b"cost", 1);
//! let far = algo::sssp(&s, &w, s.dense(1).unwrap());
//! assert_eq!(far[s.dense(3).unwrap() as usize], 1);
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;

/// How far away a node the search never reached is.
pub const UNREACHABLE: u64 = u64::MAX;

/// How far every node is from `src`, or [`UNREACHABLE`].
///
/// `weights` is what [`Snapshot::weighted`] handed back, in the same order as
/// the outgoing runs. A shorter one is treated as a graph whose remaining edges
/// weigh nothing, because a read should not panic on a caller's arithmetic, and
/// that is a mistake the caller will see in the answer immediately.
#[must_use]
pub fn sssp(g: &Snapshot, weights: &[u32], src: u32) -> Vec<u64> {
    sssp_with(g, weights, src, pick(g, weights))
}

/// The same, with the band width spelled out.
///
/// A `delta` of zero is read as one, since a band has to have some width. Very
/// large weights against a very small `delta` means a lot of empty buckets to
/// step over, which costs time and no memory, since the buckets are a ring
/// however far apart the distances are.
#[must_use]
pub fn sssp_with(g: &Snapshot, weights: &[u32], src: u32, delta: u32) -> Vec<u64> {
    let n = g.nodes();
    let mut far = vec![UNREACHABLE; n as usize];
    if src >= n {
        return far;
    }
    let delta = u64::from(delta.max(1));

    // A ring of buckets, wide enough that an edge relaxed out of the bucket
    // being emptied cannot reach round to a bucket that has already been done.
    // The heaviest edge moves a node at most that many bands forward.
    let heaviest = u64::from(weights.iter().copied().max().unwrap_or(0));
    let ring = (heaviest / delta + 2) as usize;
    let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); ring];

    far[src as usize] = 0;
    bucket[0].push(src);
    let mut waiting = 1usize;

    let mut band = 0u64;
    let mut done: Vec<u32> = Vec::new();
    while waiting > 0 {
        let at = (band % ring as u64) as usize;
        done.clear();

        // The light edges, over and over, because relaxing one can put a node
        // back into the bucket that is being emptied.
        while !bucket[at].is_empty() {
            let here = std::mem::take(&mut bucket[at]);
            waiting -= here.len();
            for node in here {
                // Somebody found a shorter way to this node after it went into
                // the bucket, so it belongs to an earlier band and has already
                // been dealt with there.
                if far[node as usize] / delta != band {
                    continue;
                }
                done.push(node);
                // Nothing relaxed below can make this node itself any closer,
                // since no edge weighs less than nothing, so the distance it is
                // working from is fixed for the whole of this loop.
                let at = far[node as usize];
                for (to, weight) in near(g, weights, node) {
                    if u64::from(weight) <= delta {
                        let now = at + u64::from(weight);
                        relax(&mut far, &mut bucket, &mut waiting, delta, ring, to, now);
                    }
                }
            }
        }

        // Then the heavy ones, once, for everything that came out of the band.
        for node in &done {
            let at = far[*node as usize];
            for (to, weight) in near(g, weights, *node) {
                if u64::from(weight) > delta {
                    let now = at + u64::from(weight);
                    relax(&mut far, &mut bucket, &mut waiting, delta, ring, to, now);
                }
            }
        }
        band += 1;
    }
    far
}

/// One node's edges, as a neighbour and what it costs to take it.
fn near<'a>(
    g: &'a Snapshot,
    weights: &'a [u32],
    node: u32,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    let from = g.out_at(node);
    g.out(node)
        .iter()
        .enumerate()
        .map(move |(i, to)| (*to, weights.get(from + i).copied().unwrap_or(0)))
}

/// Take a shorter way to a node, if this is one.
fn relax(
    far: &mut [u64],
    bucket: &mut [Vec<u32>],
    waiting: &mut usize,
    delta: u64,
    ring: usize,
    node: u32,
    now: u64,
) {
    if now >= far[node as usize] {
        return;
    }
    far[node as usize] = now;
    bucket[(now / delta % ring as u64) as usize].push(node);
    *waiting += 1;
}

/// The band width to use when the caller has no opinion.
///
/// The heaviest edge over the average degree, which is the integer reading of
/// the paper's guidance. The point of it is that a band should hold about one
/// step's worth of the graph: with an average degree of sixteen and weights up
/// to a hundred, a band a hundred wide would take in most of the frontier at
/// once and a band one wide would take in almost none of it.
fn pick(g: &Snapshot, weights: &[u32]) -> u32 {
    let heaviest = weights.iter().copied().max().unwrap_or(1);
    let degree = (g.edges() / u64::from(g.nodes().max(1))).max(1);
    (u64::from(heaviest) / degree).max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use std::collections::BinaryHeap;
    use yo_common::Rng;
    use yo_doc::Builder;

    /// Dijkstra with a heap, as the thing delta stepping has to agree with.
    fn reference(g: &Snapshot, weights: &[u32], src: u32) -> Vec<u64> {
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

    fn costing(cost: i64) -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("an object");
        b.key(b"cost").expect("a key");
        b.int(cost).expect("a number");
        b.end_object().expect("an end");
        b.finish().expect("a document").to_vec()
    }

    /// A graph where every edge carries its own cost.
    fn weighted(edges: &[(u64, u64, i64)]) -> (Snapshot, Vec<u32>) {
        let mut g = Graph::new();
        for (from, to, cost) in edges {
            g.link(*from, *to, 1, &costing(*cost)).expect("an edge");
        }
        Snapshot::weighted(&g, &[1], b"cost", 1)
    }

    #[test]
    fn a_chain_adds_up() {
        let (s, w) = weighted(&[(1, 2, 5), (2, 3, 7), (3, 4, 2)]);
        let far = sssp(&s, &w, s.dense(1).expect("1"));
        assert_eq!(far[s.dense(4).expect("4") as usize], 14);
    }

    #[test]
    fn the_cheap_way_round_wins() {
        // Straight there costs 100, the long way costs 3.
        let (s, w) = weighted(&[(1, 4, 100), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
        let far = sssp(&s, &w, s.dense(1).expect("1"));
        assert_eq!(far[s.dense(4).expect("4") as usize], 3);
    }

    #[test]
    fn what_cannot_be_reached_stays_unreachable() {
        let (s, w) = weighted(&[(1, 2, 1), (3, 4, 1)]);
        let far = sssp(&s, &w, s.dense(1).expect("1"));
        assert_eq!(far[s.dense(3).expect("3") as usize], UNREACHABLE);
        assert_eq!(far[s.dense(1).expect("1") as usize], 0);
    }

    #[test]
    fn direction_is_respected() {
        let (s, w) = weighted(&[(1, 2, 1)]);
        let far = sssp(&s, &w, s.dense(2).expect("2"));
        assert_eq!(far[s.dense(1).expect("1") as usize], UNREACHABLE);
    }

    #[test]
    fn a_source_that_is_not_a_node() {
        let (s, w) = weighted(&[(1, 2, 1)]);
        assert!(sssp(&s, &w, 99).iter().all(|far| *far == UNREACHABLE));
    }

    #[test]
    fn an_edge_that_weighs_nothing_is_free() {
        let (s, w) = weighted(&[(1, 2, 0), (2, 3, 0)]);
        let far = sssp(&s, &w, s.dense(1).expect("1"));
        assert_eq!(far[s.dense(3).expect("3") as usize], 0);
    }

    /// A weight far bigger than the band, which is the heavy edge path.
    #[test]
    fn a_very_heavy_edge() {
        let (s, w) = weighted(&[(1, 2, 1), (2, 3, 1_000_000), (1, 3, 999_999)]);
        let far = sssp(&s, &w, s.dense(1).expect("1"));
        assert_eq!(far[s.dense(3).expect("3") as usize], 999_999);
    }

    #[test]
    fn all_the_same_weight_is_the_hop_count() {
        let edges: Vec<(u64, u64, i64)> = (0..50u64).map(|i| (i, i + 1, 1)).collect();
        let (s, w) = weighted(&edges);
        let far = sssp(&s, &w, s.dense(0).expect("0"));
        for id in 0..=50u64 {
            assert_eq!(far[s.dense(id).expect("a node") as usize], id);
        }
    }

    /// Whatever band width it is given, the answer is the same answer.
    #[test]
    fn the_band_width_does_not_change_the_answer() {
        let mut rng = Rng::new(0x5551);
        let edges: Vec<(u64, u64, i64)> = (0..600)
            .map(|_| {
                (
                    rng.next_u64() % 100,
                    rng.next_u64() % 100,
                    (rng.next_u64() % 50) as i64,
                )
            })
            .collect();
        let (s, w) = weighted(&edges);
        let src = s.dense(0).expect("0");
        let want = reference(&s, &w, src);
        for delta in [0u32, 1, 3, 16, 1000] {
            assert_eq!(sssp_with(&s, &w, src, delta), want, "delta {delta}");
        }
    }

    #[test]
    fn it_agrees_with_dijkstra() {
        let mut rng = Rng::new(0x5550);
        for case in 0..40 {
            let nodes = 2 + rng.next_u64() % 80;
            let edges: Vec<(u64, u64, i64)> = (0..nodes * 3)
                .map(|_| {
                    (
                        rng.next_u64() % nodes,
                        rng.next_u64() % nodes,
                        (rng.next_u64() % 200) as i64,
                    )
                })
                .collect();
            let (s, w) = weighted(&edges);
            let src = rng.next_u64() as u32 % s.nodes();
            assert_eq!(sssp(&s, &w, src), reference(&s, &w, src), "case {case}");
        }
    }

    /// The weights themselves, since a wrong one is a wrong answer everywhere.
    #[test]
    fn a_weight_comes_off_the_edge_it_belongs_to() {
        let mut g = Graph::new();
        g.link(1, 2, 1, &costing(7)).expect("an edge");
        g.link(1, 3, 1, &costing(9)).expect("an edge");
        // No cost on this one at all, so it takes the default.
        g.link(2, 3, 1, NO_PROPS).expect("an edge");
        let (s, w) = Snapshot::weighted(&g, &[1], b"cost", 4);

        let one = s.dense(1).expect("1");
        let seen: Vec<(u64, u32)> = s
            .out(one)
            .iter()
            .enumerate()
            .map(|(i, to)| (s.id(*to), w[s.out_at(one) + i]))
            .collect();
        assert!(seen.contains(&(2, 7)), "{seen:?}");
        assert!(seen.contains(&(3, 9)), "{seen:?}");

        let two = s.dense(2).expect("2");
        assert_eq!(w[s.out_at(two)], 4);
    }

    /// A weight that is not a number, or is one nobody can travel along.
    #[test]
    fn a_weight_that_is_not_a_weight() {
        let mut b = Builder::new();
        b.begin_object().expect("an object");
        b.key(b"cost").expect("a key");
        b.text("free").expect("some text");
        b.end_object().expect("an end");
        let text = b.finish().expect("a document").to_vec();

        let mut g = Graph::new();
        g.link(1, 2, 1, &text).expect("an edge");
        g.link(2, 3, 1, &costing(-5)).expect("an edge");
        let (_, w) = Snapshot::weighted(&g, &[1], b"cost", 6);
        assert_eq!(w, vec![6, 6]);
    }
}
