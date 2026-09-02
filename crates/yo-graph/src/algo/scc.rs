//! Strongly connected components, in one pass, without the stack.
//!
//! Tarjan, "Depth-First Search and Linear Graph Algorithms", SIAM J. Comput.
//! 1972, which is still the fastest thing there is on one core: every node and
//! every edge is looked at once and the answer comes out of the same walk.
//!
//! # Strongly, not weakly
//!
//! [`super::wcc()`] joins two nodes if an edge runs between them either way.
//! This joins them only if each can be reached from the other, following edges
//! the way they point. A chain of a thousand nodes is one weak component and a
//! thousand strong ones. Which of the two a caller wants is usually obvious from
//! the question: "is this reachable from that" is strong, "are these two things
//! part of the same pile" is weak.
//!
//! # Why it is written with an explicit stack
//!
//! Tarjan's algorithm is a depth first search, and the textbook form is
//! recursive. The recursion goes as deep as the longest path in the graph, so a
//! chain of ten million nodes is ten million frames, which overflows the stack
//! and takes the process with it. That is not a rare shape either: a log, a
//! version history and a linked list are all chains.
//!
//! So the frames live in a `Vec` on the heap. Each one is a node and how far
//! through its neighbours the walk had got, which is eight bytes rather than the
//! hundred or so a real frame costs, and the depth is then bounded by memory
//! rather than by a limit nobody set on purpose.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! // A three node cycle, and a fourth node hanging off it.
//! for (a, b) in [(1u64, 2u64), (2, 3), (3, 1), (3, 4)] {
//!     g.link(a, b, 1, NO_PROPS)?;
//! }
//!
//! let s = Snapshot::of(&g);
//! let c = algo::scc(&s);
//! assert_eq!(c.count(), 2);
//! assert!(c.same(s.dense(1).unwrap(), s.dense(3).unwrap()));
//! assert!(!c.same(s.dense(1).unwrap(), s.dense(4).unwrap()));
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;
use crate::algo::{Bits, Components};

/// A node the walk has not reached, and a component not decided yet.
const NONE: u32 = u32::MAX;

/// The strongly connected components of `g`.
///
/// Every node is in exactly one, and a node with no cycle through it is in one
/// of its own. Like [`super::wcc()`], a component is named by the lowest dense
/// id in it, so the labels do not depend on which node the walk happened to
/// start from and two runs over the same snapshot agree.
#[must_use]
pub fn scc(g: &Snapshot) -> Components {
    let n = g.nodes() as usize;
    let mut of = vec![NONE; n];
    if n == 0 {
        return Components { of, count: 0 };
    }

    // When the walk first saw each node, and the earliest node it can get back
    // to. A node whose two are equal is the root of a component.
    let mut seen = vec![NONE; n];
    let mut low = vec![NONE; n];
    let mut when = 0u32;

    // The nodes that have been seen and not yet given a component, and a bitmap
    // of the same thing, because the test is per edge and searching a stack per
    // edge is what makes a naive version quadratic.
    let mut open: Vec<u32> = Vec::new();
    let mut is_open = Bits::new(g.nodes());

    // The walk itself: a node, and how far down its neighbours it had got.
    let mut work: Vec<(u32, u32)> = Vec::new();
    let mut count = 0;

    for root in 0..n as u32 {
        if seen[root as usize] != NONE {
            continue;
        }
        work.push((root, 0));
        while let Some((node, from)) = work.pop() {
            if from == 0 {
                seen[node as usize] = when;
                low[node as usize] = when;
                when += 1;
                open.push(node);
                is_open.set(node);
            }

            // Down to the first neighbour nobody has been to, taking the ones
            // already open into account on the way past.
            let near = g.out(node);
            let mut down = false;
            for (at, to) in near.iter().enumerate().skip(from as usize) {
                if seen[*to as usize] == NONE {
                    work.push((node, at as u32 + 1));
                    work.push((*to, 0));
                    down = true;
                    break;
                }
                if is_open.get(*to) {
                    low[node as usize] = low[node as usize].min(seen[*to as usize]);
                }
            }
            if down {
                continue;
            }

            // Nothing left below this node. If it never found a way back past
            // itself then everything opened since is one component.
            if low[node as usize] == seen[node as usize] {
                let at = open.iter().rposition(|open| *open == node).unwrap_or(0);
                let members = &open[at..];
                let label = members.iter().copied().min().unwrap_or(node);
                for member in members {
                    of[*member as usize] = label;
                    is_open.unset(*member);
                }
                open.truncate(at);
                count += 1;
            }

            // Hand what this node can reach back up to whoever called it.
            if let Some((up, _)) = work.last() {
                low[*up as usize] = low[*up as usize].min(low[node as usize]);
            }
        }
    }

    Components { of, count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use yo_common::Rng;

    /// Reachable both ways, asked one node at a time. Quadratic and obviously
    /// right, which is the job.
    fn reference(g: &Snapshot) -> Vec<u32> {
        let n = g.nodes() as usize;
        let reach = |src: u32, back: bool| {
            let mut got = vec![false; n];
            let mut todo = vec![src];
            got[src as usize] = true;
            while let Some(node) = todo.pop() {
                let near = if back { g.into_(node) } else { g.out(node) };
                for to in near {
                    if !got[*to as usize] {
                        got[*to as usize] = true;
                        todo.push(*to);
                    }
                }
            }
            got
        };
        let mut of = vec![0u32; n];
        for node in 0..n as u32 {
            let (down, up) = (reach(node, false), reach(node, true));
            of[node as usize] = (0..n as u32)
                .find(|other| down[*other as usize] && up[*other as usize])
                .unwrap_or(node);
        }
        of
    }

    fn linked(edges: &[(u64, u64)]) -> Graph {
        let mut g = Graph::new();
        for (from, to) in edges {
            g.link(*from, *to, 1, NO_PROPS).expect("an edge");
        }
        g
    }

    #[test]
    fn a_cycle_is_one_component() {
        let s = Snapshot::of(&linked(&[(1, 2), (2, 3), (3, 1)]));
        let c = scc(&s);
        assert_eq!(c.count(), 1);
        assert_eq!(c.labels(), [0, 0, 0]);
    }

    #[test]
    fn a_chain_is_all_singletons() {
        let s = Snapshot::of(&linked(&[(1, 2), (2, 3), (3, 4)]));
        let c = scc(&s);
        assert_eq!(c.count(), 4);
        assert_eq!(c.labels(), [0, 1, 2, 3]);
    }

    /// The same edges read the other way round are one weak component and four
    /// strong ones, which is the difference between the two questions.
    #[test]
    fn it_is_not_the_weak_answer() {
        let s = Snapshot::of(&linked(&[(1, 2), (2, 3), (3, 4)]));
        assert_eq!(super::super::wcc(&s).count(), 1);
        assert_eq!(scc(&s).count(), 4);
    }

    #[test]
    fn two_cycles_joined_one_way_stay_apart() {
        let s = Snapshot::of(&linked(&[(1, 2), (2, 1), (3, 4), (4, 3), (2, 3)]));
        let c = scc(&s);
        assert_eq!(c.count(), 2);
        assert!(c.same(s.dense(1).expect("1"), s.dense(2).expect("2")));
        assert!(!c.same(s.dense(2).expect("2"), s.dense(3).expect("3")));

        // And joining them back the other way makes them one.
        let s = Snapshot::of(&linked(&[(1, 2), (2, 1), (3, 4), (4, 3), (2, 3), (3, 2)]));
        assert_eq!(scc(&s).count(), 1);
    }

    #[test]
    fn a_self_loop_is_its_own_component() {
        let s = Snapshot::of(&linked(&[(1, 1), (1, 2)]));
        let c = scc(&s);
        assert_eq!(c.count(), 2);
        assert!(!c.same(s.dense(1).expect("1"), s.dense(2).expect("2")));
    }

    #[test]
    fn nothing_at_all() {
        let c = scc(&Snapshot::default());
        assert_eq!(c.count(), 0);
        assert!(c.is_empty());
        assert_eq!(c.largest(), None);
    }

    #[test]
    fn an_isolated_node_counts() {
        let mut g = Graph::new();
        g.add_node(9).expect("a node");
        g.link(1, 2, 1, NO_PROPS).expect("an edge");
        g.link(2, 1, 1, NO_PROPS).expect("an edge");
        let c = scc(&Snapshot::of(&g));
        assert_eq!(c.count(), 2);
        assert_eq!(c.largest(), Some((0, 2)));
    }

    /// The reason the frames are on the heap. A recursive Tarjan dies here.
    #[test]
    fn a_very_deep_chain_does_not_blow_the_stack() {
        let deep = 150_000u64;
        let edges: Vec<(u64, u64)> = (0..deep).map(|i| (i, i + 1)).collect();
        let c = scc(&Snapshot::of(&linked(&edges)));
        assert_eq!(u64::from(c.count()), deep + 1);
    }

    /// And a very long cycle, which is one component the same depth down.
    #[test]
    fn a_very_deep_cycle_is_one_component() {
        let deep = 150_000u64;
        let mut edges: Vec<(u64, u64)> = (0..deep).map(|i| (i, i + 1)).collect();
        edges.push((deep, 0));
        let c = scc(&Snapshot::of(&linked(&edges)));
        assert_eq!(c.count(), 1);
        assert_eq!(c.largest(), Some((0, deep as u32 + 1)));
    }

    #[test]
    fn it_agrees_with_the_slow_one() {
        let mut rng = Rng::new(0x5cc);
        for case in 0..80 {
            let nodes = 2 + rng.next_u64() % 40;
            let edges: Vec<(u64, u64)> = (0..nodes * 2)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            assert_eq!(scc(&s).labels(), reference(&s), "case {case}");
        }
    }
}
