//! Weakly connected components, by Afforest.
//!
//! Sutton, Ben-Nun and Barak, "Optimizing Parallel Graph Connectivity
//! Computation via Subgraph Sampling", IPDPS 2018. The GAP suite's connected
//! components kernel is this algorithm.
//!
//! # Why not union find over every edge
//!
//! Because almost every edge is redundant. A real graph has one component with
//! most of the graph in it, and once a node is in that component every further
//! edge from it only says so again. Union find over all of them is O(m) unions
//! that mostly do nothing, and each one is a pointer chase into a table the size
//! of the graph.
//!
//! Afforest turns that around in three parts. First it unions each node with its
//! first two neighbours only, which is a fixed two passes and enough to have
//! built the giant component with high probability, because a component with a
//! million nodes in it does not survive being sampled at two edges a node.
//! Second it works out which component that is, by sampling a thousand nodes and
//! taking the most common answer, which costs nothing next to a pass over the
//! graph. Third it does the real union find pass, but only over nodes that are
//! not already in the big component, which on a social graph is a few percent of
//! them.
//!
//! The result is the same components a full union find gives. The sampling
//! decides how much work is skipped, not what the answer is: a wrong guess at
//! the biggest component costs a slower run and nothing else, which is why the
//! sample can be small and fixed.
//!
//! # Weakly, and what that means here
//!
//! Edges are read both ways. Two nodes are in the same component if there is a
//! path between them ignoring which way the edges point, which is what "weakly"
//! means and what a component is for. Strongly connected components, where the
//! path has to exist in both directions, are a different algorithm and are not
//! this one.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! g.link(1, 2, 1, NO_PROPS)?;
//! g.link(3, 4, 1, NO_PROPS)?;
//!
//! let s = Snapshot::of(&g);
//! let c = algo::wcc(&s);
//! assert_eq!(c.count(), 2);
//! assert!(c.same(s.dense(1).unwrap(), s.dense(2).unwrap()));
//! assert!(!c.same(s.dense(1).unwrap(), s.dense(3).unwrap()));
//! # Ok::<(), yo_common::Error>(())
//! ```

use yo_common::Rng;

use crate::Snapshot;

/// How many neighbours of each node the sampling pass links.
///
/// Two, from the paper. One is not enough on a graph with a lot of degree one
/// nodes and three costs a pass to find almost nothing new.
const ROUNDS: usize = 2;

/// How many nodes are looked at to guess which component is the big one.
///
/// A thousand, from the paper. The guess only has to be right often enough to
/// be worth making, and at a thousand samples a component holding more than a
/// tenth of the graph is found essentially always.
const SAMPLE: usize = 1024;

/// The seed the sample is drawn with.
///
/// Fixed, so that two runs over the same snapshot do the same work and give the
/// same representative for every component. See the module docs on [`super`].
const SEED: u64 = 0x00c0_ffee;

/// Which component each node is in.
#[derive(Debug, Clone)]
pub struct Components {
    /// The representative of each node's component, which is the smallest dense
    /// id in it.
    of: Vec<u32>,
    count: u32,
}

impl Components {
    /// The component `node` is in, named by the lowest numbered node in it.
    ///
    /// # Panics
    ///
    /// If `node` is not a node of the snapshot this was computed from.
    #[must_use]
    pub fn of(&self, node: u32) -> u32 {
        self.of[node as usize]
    }

    /// Whether two nodes are in the same component.
    #[must_use]
    pub fn same(&self, a: u32, b: u32) -> bool {
        self.of(a) == self.of(b)
    }

    /// How many components there are, counting an isolated node as its own.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// How many nodes were labelled.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.of.len() as u32
    }

    /// Whether there were no nodes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.of.is_empty()
    }

    /// The component with the most nodes in it, and how many that is.
    ///
    /// `None` for a graph with no nodes. The lowest numbered of them when two
    /// are the same size, so the answer does not depend on iteration order.
    #[must_use]
    pub fn largest(&self) -> Option<(u32, u32)> {
        let mut size = vec![0u32; self.of.len()];
        for c in &self.of {
            size[*c as usize] += 1;
        }
        size.iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .max_by_key(|(at, n)| (**n, std::cmp::Reverse(*at)))
            .map(|(at, n)| (at as u32, *n))
    }

    /// The label of every node, in dense id order.
    #[must_use]
    pub fn labels(&self) -> &[u32] {
        &self.of
    }
}

/// The weakly connected components of `g`.
#[must_use]
pub fn wcc(g: &Snapshot) -> Components {
    let n = g.nodes() as usize;
    let mut of: Vec<u32> = (0..n as u32).collect();
    if n == 0 {
        return Components { of, count: 0 };
    }

    // One: link each node to its first `ROUNDS` neighbours. Outgoing only,
    // because an edge read from both ends is the same union twice and this pass
    // is about touching as few edges as possible.
    for r in 0..ROUNDS {
        for node in 0..n as u32 {
            if let Some(to) = g.out(node).get(r) {
                link(&mut of, node, *to);
            }
        }
        compress(&mut of);
    }

    // Two: which component is the big one, from a fixed sample.
    let big = frequent(&of);

    // Three: everything not already in it, both ways, skipping the neighbours
    // the sampling pass already linked.
    for node in 0..n as u32 {
        if of[node as usize] == big {
            continue;
        }
        for to in g.out(node).iter().skip(ROUNDS) {
            link(&mut of, node, *to);
        }
        for from in g.into_(node) {
            link(&mut of, node, *from);
        }
    }
    compress(&mut of);

    // The representative a component ends up with is the lowest id in it,
    // because `link` always hooks the higher of two onto the lower, so the
    // labels are stable and comparable between runs.
    let count = of
        .iter()
        .enumerate()
        .filter(|(at, c)| **c == *at as u32)
        .count() as u32;
    Components { of, count }
}

/// Hook two nodes' trees together, the higher onto the lower.
///
/// The paper's `link`, which is union find with union by index rather than by
/// rank. Union by index is what makes the label deterministic, and the paths
/// this leaves behind are flattened by [`compress`] rather than at every find.
fn link(of: &mut [u32], a: u32, b: u32) {
    let (mut p1, mut p2) = (of[a as usize], of[b as usize]);
    while p1 != p2 {
        let (high, low) = if p1 > p2 { (p1, p2) } else { (p2, p1) };
        let up = of[high as usize];
        if up == low {
            break;
        }
        if up == high {
            of[high as usize] = low;
            break;
        }
        p1 = of[up as usize];
        p2 = of[low as usize];
    }
}

/// Point every node straight at the root of its tree.
///
/// A pass rather than a find per node, because the pass is sequential over an
/// array and a find is a chase down a pointer chain. Two levels at a time, so a
/// long chain shortens by half each time round rather than one link.
fn compress(of: &mut [u32]) {
    for node in 0..of.len() {
        while of[node] != of[of[node] as usize] {
            of[node] = of[of[node] as usize];
        }
    }
}

/// The most common label in a fixed sample of the whole array.
fn frequent(of: &[u32]) -> u32 {
    let mut rng = Rng::new(SEED);
    let mut seen: Vec<(u32, u32)> = Vec::with_capacity(SAMPLE);
    for _ in 0..SAMPLE.min(of.len()) {
        let label = of[rng.below(of.len())];
        match seen.iter_mut().find(|(l, _)| *l == label) {
            Some((_, n)) => *n += 1,
            // A sample of a thousand over a graph with a giant component has a
            // handful of distinct labels in it, so a linear scan beats a hash
            // table and cannot be worse than the sample size squared.
            None => seen.push((label, 1)),
        }
    }
    seen.iter()
        .max_by_key(|(_, n)| *n)
        .map_or(0, |(label, _)| *label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use yo_common::Rng;

    /// Plain union find over every edge, as the thing Afforest has to agree
    /// with. Slow and obviously right, which is the job.
    fn reference(g: &Snapshot) -> Vec<u32> {
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

    fn linked(edges: &[(u64, u64)]) -> Graph {
        let mut g = Graph::new();
        for (src, dst) in edges {
            g.link(*src, *dst, 1, NO_PROPS).unwrap();
        }
        g
    }

    #[test]
    fn two_pieces_are_two_components() {
        let s = Snapshot::of(&linked(&[(0, 1), (1, 2), (5, 6)]));
        let c = wcc(&s);
        assert_eq!(c.count(), 2);
        assert_eq!(c.len(), 5);
        assert!(c.same(0, 2), "one piece");
        assert!(!c.same(0, 3), "the other");
        assert_eq!(c.of(0), 0, "named by the lowest node in it");
        assert_eq!(c.largest(), Some((0, 3)));
    }

    /// The direction of an edge does not decide a weak component, which is the
    /// whole difference between this and a strongly connected one.
    #[test]
    fn an_edge_joins_both_of_its_ends_whichever_way_it_points() {
        let s = Snapshot::of(&linked(&[(0, 1), (2, 1)]));
        let c = wcc(&s);
        assert_eq!(c.count(), 1);
        assert!(c.same(0, 2));
    }

    #[test]
    fn a_node_with_no_edges_is_its_own_component() {
        let mut g = Graph::new();
        g.link(0, 1, 1, NO_PROPS).unwrap();
        g.add_node(9).unwrap();
        let s = Snapshot::of(&g);
        let c = wcc(&s);
        assert_eq!(c.count(), 2);
        assert_eq!(c.largest(), Some((0, 2)));
        assert!(!c.same(0, s.dense(9).unwrap()));
    }

    #[test]
    fn nothing_at_all_has_no_components() {
        let c = wcc(&Snapshot::of(&Graph::new()));
        assert!(c.is_empty());
        assert_eq!(c.count(), 0);
        assert_eq!(c.largest(), None);
    }

    #[test]
    fn a_self_loop_and_a_parallel_edge_change_nothing() {
        let s = Snapshot::of(&linked(&[(0, 0), (0, 1), (0, 1), (1, 1)]));
        let c = wcc(&s);
        assert_eq!(c.count(), 1);
        assert_eq!(c.labels(), [0, 0]);
    }

    /// A hundred random graphs, checked against plain union find. Small ones,
    /// because the point is to hit the shapes the sampling can get wrong: a
    /// graph with no giant component, a graph that is all one component, and a
    /// graph of nothing but degree one nodes.
    #[test]
    fn it_agrees_with_union_find_on_a_hundred_random_graphs() {
        let mut rng = Rng::new(0xbeef);
        for trial in 0..100 {
            let n = 1 + rng.below(80) as u64;
            let m = rng.below(120);
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
            let want = reference(&s);
            let got = wcc(&s);
            assert_eq!(got.labels(), want, "trial {trial}");
        }
    }

    /// The shape the algorithm is for: one component with almost everything in
    /// it and a long tail of small ones.
    #[test]
    fn a_graph_with_a_giant_component_comes_out_right() {
        let mut rng = Rng::new(0x9a1);
        let n = 20_000u64;
        let mut g = Graph::new();
        for i in 0..n {
            g.add_node(i).unwrap();
        }
        // A ring through the first ninety percent, so that part is certainly
        // one component, and then random edges over the whole graph.
        let big = n * 9 / 10;
        for i in 0..big {
            g.link(i, (i + 1) % big, 1, NO_PROPS).unwrap();
        }
        for _ in 0..5_000 {
            let src = big + rng.next_u64() % (n - big);
            let dst = big + rng.next_u64() % (n - big);
            g.link(src, dst, 1, NO_PROPS).unwrap();
        }
        let s = Snapshot::of(&g);
        let want = reference(&s);
        let got = wcc(&s);
        assert_eq!(got.labels(), want);
        let (label, size) = got.largest().unwrap();
        assert_eq!(label, 0);
        assert!(size >= big as u32, "the ring is one component: {size}");
    }
}
