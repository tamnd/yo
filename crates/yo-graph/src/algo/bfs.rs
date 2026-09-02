//! Breadth first search, direction optimizing.
//!
//! Beamer, Asanović and Patterson, "Direction-Optimizing Breadth-First Search",
//! SC12. The same algorithm the GAP benchmark suite measures, which is what
//! makes a number from here comparable with a published one.
//!
//! # The idea, which is one observation
//!
//! A textbook breadth first search is top down: take the frontier, look at
//! every edge that leaves it, and keep the neighbours nobody has reached yet.
//! On a graph with a heavy tail that is enormously wasteful in the middle of the
//! search, because by then most of the neighbours of the frontier have already
//! been reached and every one of those edges is checked to find that out.
//!
//! The other way round is bottom up: take the nodes nobody has reached yet, and
//! for each one look at its incoming edges until one of them comes from the
//! frontier. That stops at the first hit rather than reading the whole run, so
//! when the frontier is large it does a fraction of the work. It is terrible
//! when the frontier is small, because it reads every unreached node to find
//! that almost none of them are next to it.
//!
//! So a search does both. It starts top down, switches to bottom up when the
//! frontier's edges outnumber what is left by more than `ALPHA`, and switches
//! back when the frontier has shrunk below the node count over `BETA`. On
//! Kronecker and social graphs the middle two or three levels are where nearly
//! all the edges are, and those are the levels that go bottom up.
//!
//! # What the numbers are
//!
//! `ALPHA` at 15 and `BETA` at 18 are the paper's, tuned on Kronecker graphs and
//! carried by the GAP suite since. They are constants here rather than
//! parameters because a caller has no way to pick better ones and a search that
//! is 5 percent off the best switch point is still a different complexity class
//! from one that never switches.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! for i in 0..4u64 {
//!     g.link(i, i + 1, 1, NO_PROPS)?;
//! }
//!
//! let s = Snapshot::of(&g);
//! let depth = algo::bfs(&s, s.dense(0).unwrap());
//! assert_eq!(depth[s.dense(3).unwrap() as usize], 3);
//! assert_eq!(depth[s.dense(0).unwrap() as usize], 0);
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;
use crate::algo::Bits;

/// The depth of a node the search never got to.
pub const UNREACHED: u32 = u32::MAX;

/// Go bottom up when the frontier's edges are more than this fraction of the
/// edges that have not been looked at.
const ALPHA: u64 = 15;

/// Go back to top down when the frontier is smaller than the graph over this.
const BETA: u32 = 18;

/// How many hops each node is from `src`, or [`UNREACHED`].
///
/// Following edges the way they point. A search that should treat the graph as
/// undirected wants [`bfs_both`].
#[must_use]
pub fn bfs(g: &Snapshot, src: u32) -> Vec<u32> {
    search(g, src, false)
}

/// The same, following edges either way.
///
/// Reachability in an undirected reading of the graph, which is what a
/// friend-of-a-friend question means and what a directed search does not
/// answer.
#[must_use]
pub fn bfs_both(g: &Snapshot, src: u32) -> Vec<u32> {
    search(g, src, true)
}

fn search(g: &Snapshot, src: u32, both: bool) -> Vec<u32> {
    let n = g.nodes();
    let mut depth = vec![UNREACHED; n as usize];
    if src >= n {
        return depth;
    }
    depth[src as usize] = 0;

    let mut frontier = vec![src];
    let mut next = Vec::new();
    let mut curr_bits = Bits::new(n);
    let mut next_bits = Bits::new(n);

    // The two counters the switch is decided on. `scout` is how many edges
    // leave the frontier, which is what a top down step is about to read, and
    // `left` is how many edges have not been looked at yet, which is what a
    // bottom up step is bounded by. Both are estimates that the paper keeps
    // exactly, and keeping them costs a degree lookup per node reached.
    let mut scout = u64::from(degree(g, src, both));
    let mut left = g.edges() * if both { 2 } else { 1 };

    let mut d = 1;
    while !frontier.is_empty() {
        if scout > left / ALPHA {
            // Into the bitmap, and stay bottom up while the frontier is either
            // still growing or still big. The paper's condition, and the reason
            // it is not simply "while it is big" is that the level where the
            // frontier peaks is the one worth doing bottom up most of all.
            curr_bits.clear();
            for node in &frontier {
                curr_bits.set(*node);
            }
            let mut awake = frontier.len() as u32;
            let mut done = false;
            loop {
                let woke = bottom_up(g, &depth, &curr_bits, &mut next_bits, both);
                if woke == 0 {
                    // Nothing unreached is next to the frontier, so there is
                    // nothing left to reach and the whole search is over.
                    done = true;
                    break;
                }
                next_bits.for_each(|node| depth[node as usize] = d);
                std::mem::swap(&mut curr_bits, &mut next_bits);
                next_bits.clear();
                d += 1;
                let grew = woke >= awake;
                awake = woke;
                if !grew && woke <= n / BETA {
                    break;
                }
            }
            if done {
                break;
            }
            frontier.clear();
            curr_bits.for_each(|node| frontier.push(node));
            // Nothing counted the edges the bottom up steps read, so the
            // estimate is rebuilt from the frontier that came out of them.
            scout = frontier
                .iter()
                .map(|n| u64::from(degree(g, *n, both)))
                .sum();
            left = left.saturating_sub(scout);
            continue;
        }

        left = left.saturating_sub(scout);
        scout = 0;
        next.clear();
        for node in &frontier {
            g.prefetch(*node);
        }
        for node in &frontier {
            for to in g.out(*node) {
                if depth[*to as usize] == UNREACHED {
                    depth[*to as usize] = d;
                    scout += u64::from(degree(g, *to, both));
                    next.push(*to);
                }
            }
            if both {
                for to in g.into_(*node) {
                    if depth[*to as usize] == UNREACHED {
                        depth[*to as usize] = d;
                        scout += u64::from(degree(g, *to, both));
                        next.push(*to);
                    }
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
        d += 1;
    }
    depth
}

/// One bottom up step: every unreached node looks for a parent in `curr`.
///
/// The early exit is the whole point. A node with a thousand incoming edges
/// whose first one comes from the frontier reads one edge, and in the level
/// where most of the graph is being reached that is most nodes.
fn bottom_up(g: &Snapshot, depth: &[u32], curr: &Bits, next: &mut Bits, both: bool) -> u32 {
    let mut woke = 0;
    for node in 0..g.nodes() {
        if depth[node as usize] != UNREACHED {
            continue;
        }
        let found = g.into_(node).iter().any(|from| curr.get(*from))
            || (both && g.out(node).iter().any(|from| curr.get(*from)));
        if found {
            next.set(node);
            woke += 1;
        }
    }
    woke
}

/// How many edges a step from this node would read.
#[inline]
fn degree(g: &Snapshot, node: u32, both: bool) -> u32 {
    if both {
        g.out_degree(node) + g.in_degree(node)
    } else {
        g.out_degree(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use yo_common::Rng;

    /// Plain top down, as the thing the real one has to agree with.
    fn reference(g: &Snapshot, src: u32) -> Vec<u32> {
        let mut depth = vec![UNREACHED; g.nodes() as usize];
        depth[src as usize] = 0;
        let mut queue = std::collections::VecDeque::from([src]);
        while let Some(node) = queue.pop_front() {
            for to in g.out(node) {
                if depth[*to as usize] == UNREACHED {
                    depth[*to as usize] = depth[node as usize] + 1;
                    queue.push_back(*to);
                }
            }
        }
        depth
    }

    fn linked(edges: &[(u64, u64)]) -> Graph {
        let mut g = Graph::new();
        for (src, dst) in edges {
            g.link(*src, *dst, 1, NO_PROPS).unwrap();
        }
        g
    }

    #[test]
    fn a_chain_is_as_deep_as_it_is_long() {
        let mut g = Graph::new();
        for i in 0..999u64 {
            g.link(i, i + 1, 1, NO_PROPS).unwrap();
        }
        let s = Snapshot::of(&g);
        let depth = bfs(&s, 0);
        for i in 0..1000u32 {
            assert_eq!(depth[i as usize], i, "at {i}");
        }
    }

    #[test]
    fn what_the_edges_do_not_point_at_is_unreached() {
        let s = Snapshot::of(&linked(&[(0, 1), (1, 2), (3, 2)]));
        let depth = bfs(&s, 0);
        assert_eq!(depth, vec![0, 1, 2, UNREACHED]);
        // The same graph read either way reaches all of it.
        assert_eq!(bfs_both(&s, 0), vec![0, 1, 2, 3]);
        // And from a node with nothing leaving it, only itself.
        assert_eq!(bfs(&s, 3), vec![UNREACHED, UNREACHED, 1, 0]);
    }

    #[test]
    fn a_source_that_is_not_a_node_reaches_nothing() {
        let s = Snapshot::of(&linked(&[(0, 1)]));
        assert_eq!(bfs(&s, 7), vec![UNREACHED, UNREACHED]);
        let empty = Snapshot::of(&Graph::new());
        assert!(bfs(&empty, 0).is_empty());
    }

    #[test]
    fn a_self_loop_and_a_cycle_do_not_change_a_depth() {
        let s = Snapshot::of(&linked(&[(0, 0), (0, 1), (1, 2), (2, 0)]));
        assert_eq!(bfs(&s, 0), vec![0, 1, 2]);
    }

    /// The whole point of the algorithm is that it switches, and the only thing
    /// that matters about the switch is that the answer does not change. This
    /// graph is a hub with a hundred thousand edges hanging off it, which is
    /// shaped to make the second level cross the ALPHA threshold.
    #[test]
    fn the_two_directions_agree_on_a_graph_that_switches() {
        let mut rng = Rng::new(0x5eed);
        let mut g = Graph::new();
        let n = 20_000u64;
        for i in 1..n {
            g.link(0, i, 1, NO_PROPS).unwrap();
        }
        for _ in 0..100_000 {
            let src = rng.next_u64() % n;
            let dst = rng.next_u64() % n;
            g.link(src, dst, 1, NO_PROPS).unwrap();
        }
        let s = Snapshot::of(&g);
        assert_eq!(bfs(&s, 0), reference(&s, 0), "from the hub");
        let far = s.dense(n - 1).unwrap();
        assert_eq!(bfs(&s, far), reference(&s, far), "from a leaf");
    }

    /// A grid is the case a bottom up step is worst at and the search still has
    /// to be right on it, because the frontier never gets big enough to switch.
    #[test]
    fn a_grid_is_the_distance_you_would_walk() {
        let side = 40u64;
        let mut g = Graph::new();
        let at = |r: u64, c: u64| r * side + c;
        for r in 0..side {
            for c in 0..side {
                if c + 1 < side {
                    g.link(at(r, c), at(r, c + 1), 1, NO_PROPS).unwrap();
                }
                if r + 1 < side {
                    g.link(at(r, c), at(r + 1, c), 1, NO_PROPS).unwrap();
                }
            }
        }
        let s = Snapshot::of(&g);
        let depth = bfs(&s, 0);
        for r in 0..side {
            for c in 0..side {
                let node = s.dense(at(r, c)).unwrap();
                assert_eq!(depth[node as usize] as u64, r + c, "at {r},{c}");
            }
        }
    }

    /// Random graphs, checked edge for edge against the plain search, because
    /// the switch has enough state in it that one shape is not enough.
    #[test]
    fn it_agrees_with_a_plain_search_on_a_hundred_random_graphs() {
        let mut rng = Rng::new(0xa11ce);
        for trial in 0..100 {
            let n = 1 + rng.below(60) as u64;
            let m = rng.below(200);
            let mut g = Graph::new();
            for i in 0..n {
                g.add_node(i).unwrap();
            }
            for _ in 0..m {
                let src = rng.next_u64() % n;
                let dst = rng.next_u64() % n;
                g.link(src, dst, 1, NO_PROPS).unwrap();
            }
            let s = Snapshot::of(&g);
            for src in 0..s.nodes() {
                assert_eq!(bfs(&s, src), reference(&s, src), "trial {trial} from {src}");
            }
        }
    }
}
