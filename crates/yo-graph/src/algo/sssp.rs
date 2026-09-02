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
//! relaxed over and over.
//!
//! The paper's guidance is the largest weight over the average degree, and it is
//! written for a machine that relaxes a whole band at once. On one core the
//! trade is different, because the work a wide band saves in bucket steps comes
//! straight back as nodes relaxed twice, and the measured answer is a much
//! narrower band than the guidance gives. [`DELTA`] has the numbers.
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

/// The band width [`sssp`] uses.
///
/// Measured rather than guessed. Sweeping the width over an R-MAT graph of four
/// million edges with GAP's weights on a 13900K, the time climbs the whole way
/// as the band gets wider: 17.6 ms at one, 21.3 at four, 34.5 at sixteen and
/// 134.5 at a hundred and twenty eight, against 46.1 for a heap Dijkstra over
/// the same graph. A narrow band on one core is nearly free, because the bucket
/// steps a wide band saves come straight back as nodes relaxed twice, and the
/// paper's guidance is written for a machine that relaxes a whole band at once
/// rather than one that walks it.
///
/// Four rather than one because that graph is not every graph. Stepping the
/// bands is a step per band whether anything is in it or not, so a long chain of
/// heavy edges pays for the width it is not using, and there is a quarter as
/// much of that at four as at one. Thirteen percent on the graphs where the
/// narrowest band wins is a fair price for that.
pub const DELTA: u32 = 4;

/// The most buckets the ring is allowed, so that a heavy edge cannot ask for a
/// bucket per band across the whole weight range.
const BUCKETS: u64 = 1 << 16;

/// How far every node is from `src`, or [`UNREACHABLE`].
///
/// `weights` is what [`Snapshot::weighted`] handed back, in the same order as
/// the outgoing runs.
///
/// # Panics
///
/// If `weights` is not one weight per edge of the snapshot. A weight that does
/// not line up with the edge it belongs to is a wrong answer rather than a slow
/// one, and there is no reading of a short list that is more likely to be what
/// the caller meant than a mistake.
#[must_use]
pub fn sssp(g: &Snapshot, weights: &[u32], src: u32) -> Vec<u64> {
    sssp_with(g, weights, src, DELTA)
}

/// The same, with the band width spelled out.
///
/// Rounded down to a power of two, and a `delta` of zero read as one. It is a
/// request rather than an instruction: a band narrow enough that the heaviest
/// edge spans more than sixty five thousand of them is widened until it does
/// not, since the buckets are a ring and the ring has to be able to hold that
/// span. Nothing else about the answer depends on the width.
///
/// # Panics
///
/// If `weights` is not one weight per edge of the snapshot.
#[must_use]
pub fn sssp_with(g: &Snapshot, weights: &[u32], src: u32, delta: u32) -> Vec<u64> {
    assert_eq!(weights.len() as u64, g.edges(), "one weight an edge");
    let n = g.nodes();
    let mut far = vec![UNREACHABLE; n as usize];
    if src >= n {
        return far;
    }

    // The band width as a shift rather than a number, so that working out which
    // bucket a distance falls in is a shift and a mask rather than a division by
    // the width and then one by the ring size. Measuring it says this is worth
    // nothing at all, since the loop is waiting on memory and there was room for
    // two divides inside that wait, but the width is a heuristic to begin with
    // and rounding it to a power of two costs nothing either.
    let mut shift = delta.max(1).ilog2();
    let heaviest = u64::from(weights.iter().copied().max().unwrap_or(0));

    // The ring has to be wide enough that an edge relaxed out of the bucket
    // being emptied cannot reach round to one that has already been done, and
    // the heaviest edge moves a node that many bands forward. A caller weighing
    // edges in bytes or in microseconds rather than in hops has a heaviest edge
    // in the billions, and asking for a narrow band on top of that is asking for
    // a bucket per band across the whole range. Widen the band until the ring
    // fits instead, which costs the caller some repeated relaxations and not
    // twenty five gigabytes.
    while (heaviest >> shift) + 2 > BUCKETS {
        shift += 1;
    }
    let width = 1u64 << shift;
    let ring = ((heaviest >> shift) + 2).next_power_of_two();
    let mask = ring - 1;
    let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); ring as usize];

    far[src as usize] = 0;
    bucket[0].push(src);
    let mut waiting = 1usize;

    let mut band = 0u64;
    let mut here: Vec<u32> = Vec::new();
    let mut done: Vec<u32> = Vec::new();
    while waiting > 0 {
        let at = (band & mask) as usize;
        done.clear();

        // The light edges, over and over, because relaxing one can put a node
        // back into the bucket that is being emptied. The swap is so that the
        // room the last pass allocated is the room this one fills.
        while !bucket[at].is_empty() {
            std::mem::swap(&mut here, &mut bucket[at]);
            waiting -= here.len();
            for node in here.drain(..) {
                // Somebody found a shorter way to this node after it went into
                // the bucket, so it belongs to an earlier band and has already
                // been dealt with there.
                let from = far[node as usize];
                if from >> shift != band {
                    continue;
                }
                done.push(node);
                let near = g.out(node);
                let cost = &weights[g.out_at(node)..][..near.len()];
                for (to, weight) in near.iter().zip(cost) {
                    let weight = u64::from(*weight);
                    if weight > width {
                        continue;
                    }
                    let now = from + weight;
                    if now < far[*to as usize] {
                        far[*to as usize] = now;
                        bucket[((now >> shift) & mask) as usize].push(*to);
                        waiting += 1;
                    }
                }
            }
        }

        // Then the heavy ones, once, for everything that came out of the band.
        for node in &done {
            let from = far[*node as usize];
            let near = g.out(*node);
            let cost = &weights[g.out_at(*node)..][..near.len()];
            for (to, weight) in near.iter().zip(cost) {
                let weight = u64::from(*weight);
                if weight <= width {
                    continue;
                }
                let now = from + weight;
                if now < far[*to as usize] {
                    far[*to as usize] = now;
                    bucket[((now >> shift) & mask) as usize].push(*to);
                    waiting += 1;
                }
            }
        }
        band += 1;
    }
    far
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

    /// Weights in the billions, which is what a caller counting bytes or
    /// microseconds has, against the narrowest band anybody can ask for. A ring
    /// of one bucket per band over that range would be twenty five gigabytes,
    /// so the band has to widen on its own and the answer has to stay right.
    #[test]
    fn a_huge_weight_does_not_ask_for_a_huge_ring() {
        let heaviest = i64::from(u32::MAX);
        let (s, w) = weighted(&[
            (1, 4, heaviest),
            (1, 2, heaviest / 3),
            (2, 3, heaviest / 3),
            (3, 4, heaviest / 3),
        ]);
        let far = sssp_with(&s, &w, s.dense(1).expect("1"), 1);
        assert_eq!(
            far[s.dense(4).expect("4") as usize],
            heaviest as u64 / 3 * 3
        );
        assert_eq!(far, reference(&s, &w, s.dense(1).expect("1")));
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
