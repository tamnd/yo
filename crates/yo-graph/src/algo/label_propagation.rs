//! Communities, by everyone taking the label most of their neighbours have.
//!
//! Raghavan, Albert and Kumara, "Near linear time algorithm to detect community
//! structures in large-scale networks", Physical Review E 2007.
//!
//! # What it is for
//!
//! There is no objective function here. Every node starts with its own label,
//! and then over and over each node takes whichever label the most of its
//! neighbours are holding. Densely joined groups agree with themselves quickly,
//! a label spreads to the edge of the group it started in and stops, and what
//! comes out is a set of communities nobody asked for by number or by size.
//!
//! That is the whole appeal. It is a pass over the edges per round and a handful
//! of rounds, which is as close to linear as community detection gets, and it
//! needs no parameter at all: not a resolution, not a community count, not a
//! threshold. [`super::louvain()`] optimises modularity and will give a better
//! partition by that measure, and it costs a lot more to get there. This is the
//! one to reach for on a graph too big to think about, or as the starting point
//! for something else.
//!
//! # Where the randomness is, and why the answer is still fixed
//!
//! Two things in the paper are random: the order the nodes are visited in each
//! round, and which label wins when several are tied. Both matter, and dropping
//! either one is what makes naive implementations collapse a whole graph into
//! one community. Both come from [`yo_common::Rng`] on a fixed seed here, so a
//! caller who runs this twice over the same snapshot gets the same communities
//! twice.
//!
//! # When it stops
//!
//! Not when nothing changes. With ties broken at random a pair of nodes can
//! swap labels forever without the partition meaning anything different, so the
//! paper stops on a weaker condition: every node holds a label that at least as
//! many of its neighbours hold as any other. That is a state the algorithm can
//! actually reach, and it is what "everyone agrees with their neighbourhood"
//! means. A node already holding a winning label keeps it rather than moving to
//! another winner, which is what makes that state reachable at all.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! // Two triangles joined by one edge.
//! for (a, b) in [(1u64, 2u64), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)] {
//!     g.link(a, b, 1, NO_PROPS)?;
//! }
//!
//! let s = Snapshot::of(&g);
//! let c = algo::label_propagation(&s);
//! assert_eq!(c.count(), 2);
//! assert!(c.same(s.dense(1).unwrap(), s.dense(2).unwrap()));
//! assert!(!c.same(s.dense(1).unwrap(), s.dense(5).unwrap()));
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;
use crate::algo::{Components, tidy};
use yo_common::Rng;

/// How many rounds before giving up on everybody agreeing.
///
/// The paper reports 95 percent of nodes settled within five rounds on graphs
/// of every size it tried, so a graph still moving after this many is one where
/// the extra rounds are two nodes trading a label rather than progress.
pub const ROUNDS: u32 = 100;

/// The seed, so that two runs over one snapshot agree.
const SEED: u64 = 0x1abe_15ee;

/// The communities of `g`, by label propagation.
///
/// Edges are read both ways, since a community is not a question about which
/// way an edge points, and a self loop is ignored because a node holding its own
/// label up as evidence for its own label says nothing.
#[must_use]
pub fn label_propagation(g: &Snapshot) -> Components {
    label_propagation_with(g, ROUNDS)
}

/// The same, stopping after at most `rounds` whether or not everybody agrees.
///
/// A caller who wants a rough grouping in bounded time can ask for two rounds
/// and get one. The result is a partition either way, just not one that has
/// settled.
#[must_use]
pub fn label_propagation_with(g: &Snapshot, rounds: u32) -> Components {
    let n = g.nodes() as usize;
    let mut label: Vec<u32> = (0..g.nodes()).collect();
    if n == 0 {
        return tidy(label);
    }

    // How many neighbours hold each label, and which labels that is, so the
    // counts can be cleared in the time the node took rather than in the time
    // the whole graph would take.
    let mut count = vec![0u32; n];
    let mut seen: Vec<u32> = Vec::new();

    let mut rng = Rng::new(SEED);
    let mut order: Vec<u32> = (0..g.nodes()).collect();
    for _ in 0..rounds {
        shuffle(&mut order, &mut rng);
        // Each node is updated in place and the next one sees it, which is what
        // the paper calls the asynchronous form. Updating everybody off the
        // previous round instead makes two joined nodes swap labels forever.
        let mut agreed = true;
        for node in &order {
            seen.clear();
            for to in g.out(*node).iter().chain(g.into_(*node)) {
                if to == node {
                    continue;
                }
                let at = label[*to as usize] as usize;
                if count[at] == 0 {
                    seen.push(label[*to as usize]);
                }
                count[at] += 1;
            }

            // The most held label, picking uniformly among the ties by keeping
            // each new one with probability one over how many have been seen.
            let (mut best, mut ties, mut pick) = (0u32, 0u32, label[*node as usize]);
            for at in &seen {
                let held = count[*at as usize];
                if held > best {
                    (best, ties, pick) = (held, 1, *at);
                } else if held == best {
                    ties += 1;
                    if rng.next_u64().is_multiple_of(u64::from(ties)) {
                        pick = *at;
                    }
                }
            }

            // Already holding a winning label is agreement, and moving off it
            // to another winner would be churn that never settles.
            let content = seen.is_empty() || count[label[*node as usize] as usize] == best;
            for at in &seen {
                count[*at as usize] = 0;
            }
            if !content {
                label[*node as usize] = pick;
                agreed = false;
            }
        }
        if agreed {
            break;
        }
    }
    tidy(label)
}

/// Fisher and Yates, since the visiting order is part of the algorithm.
fn shuffle(order: &mut [u32], rng: &mut Rng) {
    for at in (1..order.len()).rev() {
        order.swap(at, (rng.next_u64() % (at as u64 + 1)) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::wcc;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use yo_common::Rng;

    fn linked(edges: &[(u64, u64)]) -> Graph {
        let mut g = Graph::new();
        for (from, to) in edges {
            g.link(*from, *to, 1, NO_PROPS).expect("an edge");
        }
        g
    }

    /// A clique of `size` nodes, numbered from `first`.
    fn clique(first: u64, size: u64) -> Vec<(u64, u64)> {
        let mut edges = Vec::new();
        for a in first..first + size {
            for b in a + 1..first + size {
                edges.push((a, b));
            }
        }
        edges
    }

    #[test]
    fn two_cliques_on_a_thread() {
        let mut edges = clique(0, 8);
        edges.extend(clique(100, 8));
        edges.push((7, 100));
        let s = Snapshot::of(&linked(&edges));
        let c = label_propagation(&s);
        assert_eq!(c.count(), 2);
        for pair in [(0u64, 5u64), (100, 105)] {
            let (a, b) = (s.dense(pair.0).expect("a"), s.dense(pair.1).expect("b"));
            assert!(c.same(a, b), "{pair:?} should be together");
        }
        assert!(!c.same(s.dense(0).expect("0"), s.dense(100).expect("100")));
    }

    /// Four cliques in a ring, which is the shape that catches an
    /// implementation that lets one label run away with the whole graph.
    #[test]
    fn four_cliques_in_a_ring() {
        let mut edges = Vec::new();
        for group in 0..4u64 {
            edges.extend(clique(group * 100, 10));
        }
        for group in 0..4u64 {
            edges.push((group * 100 + 9, (group + 1) % 4 * 100));
        }
        let s = Snapshot::of(&linked(&edges));
        let c = label_propagation(&s);
        assert_eq!(c.count(), 4);
        for group in 0..4u64 {
            let a = s.dense(group * 100 + 1).expect("a");
            let b = s.dense(group * 100 + 5).expect("b");
            assert!(c.same(a, b), "group {group}");
        }
    }

    /// A community can never span two components, whatever else it does.
    #[test]
    fn it_never_crosses_a_component() {
        let mut rng = Rng::new(0x1ab0);
        for case in 0..30 {
            let nodes = 2 + rng.next_u64() % 60;
            let edges: Vec<(u64, u64)> = (0..nodes)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            let (weak, c) = (wcc(&s), label_propagation(&s));
            for node in 0..s.nodes() {
                for other in 0..s.nodes() {
                    if c.same(node, other) {
                        assert!(weak.same(node, other), "case {case}, {node} and {other}");
                    }
                }
            }
            assert!(c.count() >= weak.count(), "case {case}");
        }
    }

    #[test]
    fn a_graph_with_no_edges_is_all_singletons() {
        let mut g = Graph::new();
        for id in 0..5u64 {
            g.add_node(id).expect("a node");
        }
        let c = label_propagation(&Snapshot::of(&g));
        assert_eq!(c.count(), 5);
        assert_eq!(c.labels(), [0, 1, 2, 3, 4]);
    }

    #[test]
    fn nothing_at_all() {
        let c = label_propagation(&Snapshot::default());
        assert_eq!(c.count(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn a_self_loop_does_not_vote() {
        let s = Snapshot::of(&linked(&[(1, 1), (1, 2), (2, 3), (3, 1)]));
        let c = label_propagation(&s);
        assert_eq!(c.count(), 1);
    }

    /// A clique cannot come out as anything but one community.
    #[test]
    fn one_clique_is_one_community() {
        let s = Snapshot::of(&linked(&clique(0, 12)));
        assert_eq!(label_propagation(&s).count(), 1);
    }

    #[test]
    fn no_rounds_at_all_leaves_everybody_alone() {
        let s = Snapshot::of(&linked(&clique(0, 6)));
        let c = label_propagation_with(&s, 0);
        assert_eq!(c.count(), 6);
    }

    /// The whole point of the fixed seed.
    #[test]
    fn two_runs_agree() {
        let mut edges = clique(0, 20);
        edges.extend(clique(100, 20));
        edges.push((3, 104));
        let s = Snapshot::of(&linked(&edges));
        assert_eq!(
            label_propagation(&s).labels(),
            label_propagation(&s).labels()
        );
    }

    /// Every label is the lowest numbered node holding it.
    #[test]
    fn the_labels_are_tidy() {
        let mut edges = clique(0, 5);
        edges.extend(clique(50, 5));
        let s = Snapshot::of(&linked(&edges));
        let c = label_propagation(&s);
        for node in 0..s.nodes() {
            assert_eq!(c.of(c.of(node)), c.of(node), "node {node}");
            assert!(c.of(node) <= node);
        }
    }

    /// Direction is not a community question: reversing every edge cannot
    /// change the answer.
    #[test]
    fn direction_does_not_matter() {
        let mut edges = clique(0, 7);
        edges.extend(clique(100, 7));
        edges.push((6, 100));
        let forward = Snapshot::of(&linked(&edges));
        let flipped: Vec<(u64, u64)> = edges.iter().map(|(a, b)| (*b, *a)).collect();
        let back = Snapshot::of(&linked(&flipped));

        // The snapshots number their nodes the same way, since both renumber in
        // sorted id order, so the labels are comparable directly.
        assert_eq!(
            label_propagation(&forward).labels(),
            label_propagation(&back).labels()
        );
    }
}
