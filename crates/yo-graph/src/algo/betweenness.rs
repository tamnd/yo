//! How often each node sits in the middle of somebody else's shortest path.
//!
//! Brandes, "A faster algorithm for betweenness centrality", Journal of
//! Mathematical Sociology 2001, sampled the way Brandes and Pich describe in
//! "Centrality estimation in large networks", Int. J. Bifurcation and Chaos
//! 2007.
//!
//! # What it measures, and why it is not PageRank
//!
//! [`super::pagerank()`] says a node is important if important nodes point at
//! it. Betweenness says a node is important if traffic has to go through it. The
//! two disagree in exactly the interesting place: the one badly connected node
//! joining two otherwise separate halves of a network has almost no PageRank and
//! the highest betweenness in the graph. That is the node whose failure splits
//! the network, the account brokering between two communities, the router
//! everything crosses.
//!
//! # How Brandes made it affordable
//!
//! Written out of the definition it is a sum over every pair of nodes, which
//! means counting shortest paths between all of them and is cubic. Brandes'
//! observation is that the whole sum can be accumulated one source at a time in
//! the time of a single search: run a breadth first search from `s` counting how
//! many shortest paths reach each node, then walk the search back out from the
//! furthest node inwards accumulating what each node owes its predecessors. That
//! turns the problem into one search per source, and nothing else.
//!
//! It is still one search per source, which on a graph with ten million nodes is
//! ten million searches. Hence the sampling.
//!
//! # Why sampling is honest here
//!
//! Each source contributes its own independent share of the total, so running
//! the accumulation from a random sample of sources and scaling by how much of
//! the graph was sampled is an unbiased estimate of the real thing. Brandes and
//! Pich also make the point that the sources have to be picked uniformly at
//! random: sampling the highest degree nodes, which sounds smarter, is biased
//! and can be much worse than sampling at random.
//!
//! The sources are drawn from [`yo_common::Rng`] on a fixed seed, so the
//! estimate is an estimate but it is the same estimate every time.
//!
//! # Which way the edges point
//!
//! A shortest path follows edges the way they point, the same as [`super::bfs()`]
//! and [`super::sssp()`]. A caller who wants the undirected reading should say so
//! in the graph by linking both ways.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! // Two triangles that can only reach each other through node 3.
//! for (a, b) in [(1u64, 2u64), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4)] {
//!     g.link(a, b, 1, NO_PROPS)?;
//! }
//!
//! let s = Snapshot::of(&g);
//! let c = algo::betweenness(&s);
//! // Node 3 is on the path between both halves and nothing else is.
//! assert_eq!(c.top(1)[0].0, s.dense(3).unwrap());
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;
use crate::algo::bfs::UNREACHED;
use yo_common::Rng;

/// How many sources [`betweenness`] runs from.
///
/// Brandes and Pich report that a few hundred sources put the ranking of the top
/// nodes within a few percent of the exact answer on graphs of every size they
/// tried, and that what the sample size has to grow with is the accuracy wanted
/// rather than the size of the graph.
pub const PIVOTS: u32 = 256;

const SEED: u64 = 0xb173_eee0;

/// How central each node is, and how it was worked out.
#[derive(Debug, Clone)]
pub struct Between {
    of: Vec<f64>,
    pivots: u32,
    exact: bool,
}

impl Between {
    /// One node's score.
    ///
    /// # Panics
    ///
    /// If `node` is not a node of the snapshot this was computed from.
    #[must_use]
    pub fn of(&self, node: u32) -> f64 {
        self.of[node as usize]
    }

    /// Every node's score, in dense id order.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.of
    }

    /// How many sources it ran from.
    #[must_use]
    pub fn pivots(&self) -> u32 {
        self.pivots
    }

    /// Whether every node was used as a source, which makes this the real
    /// answer rather than an estimate of it.
    #[must_use]
    pub fn exact(&self) -> bool {
        self.exact
    }

    /// The `n` most central nodes, highest first.
    ///
    /// The lower numbered node first when two scores match, so the answer does
    /// not depend on the sort.
    #[must_use]
    pub fn top(&self, n: usize) -> Vec<(u32, f64)> {
        let mut all: Vec<(u32, f64)> = self
            .of
            .iter()
            .enumerate()
            .map(|(node, score)| (node as u32, *score))
            .collect();
        all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(n);
        all
    }
}

/// An estimate of every node's betweenness, from [`PIVOTS`] random sources.
#[must_use]
pub fn betweenness(g: &Snapshot) -> Between {
    betweenness_with(g, PIVOTS)
}

/// The same, from a sample of the size asked for.
///
/// More sources is a better estimate and a proportionally longer wait, and
/// asking for at least as many as there are nodes is the same as asking for
/// [`betweenness_exact`].
#[must_use]
pub fn betweenness_with(g: &Snapshot, pivots: u32) -> Between {
    let n = g.nodes();
    if pivots >= n {
        return betweenness_exact(g);
    }
    let mut from: Vec<u32> = (0..n).collect();
    // Only the front of the shuffle is needed, so only the front is done.
    let mut rng = Rng::new(SEED);
    for at in 0..pivots as usize {
        let take = at + (rng.next_u64() % (n as u64 - at as u64)) as usize;
        from.swap(at, take);
    }
    from.truncate(pivots as usize);

    let mut c = accumulate(g, &from);
    // Each source stands for the ones that were not picked.
    let scale = f64::from(n) / f64::from(pivots);
    for score in &mut c.of {
        *score *= scale;
    }
    c
}

/// Every node's betweenness, from every source, which is the real answer.
///
/// One breadth first search per node, so a graph of any size is a long wait.
/// Here to check the estimate against and for graphs small enough that exact is
/// affordable.
#[must_use]
pub fn betweenness_exact(g: &Snapshot) -> Between {
    let all: Vec<u32> = (0..g.nodes()).collect();
    let mut c = accumulate(g, &all);
    c.exact = true;
    c
}

/// Brandes' accumulation, run from each source in turn.
fn accumulate(g: &Snapshot, from: &[u32]) -> Between {
    let n = g.nodes() as usize;
    let mut of = vec![0f64; n];
    if n == 0 {
        return Between {
            of,
            pivots: 0,
            exact: false,
        };
    }

    // How far each node is, how many shortest paths reach it, and what it owes.
    // All three are cleared after each source through the visit order rather
    // than by wiping the whole array, so a source that reaches a hundred nodes
    // costs a hundred rather than the size of the graph.
    let mut depth = vec![UNREACHED; n];
    let mut paths = vec![0f64; n];
    let mut owed = vec![0f64; n];
    let mut order: Vec<u32> = Vec::new();

    for src in from {
        order.clear();
        depth[*src as usize] = 0;
        paths[*src as usize] = 1.0;

        // Out from the source, counting shortest paths as it goes. A node one
        // level further on gains every path that reached whoever found it.
        let mut head = 0usize;
        order.push(*src);
        while head < order.len() {
            let node = order[head];
            head += 1;
            let next = depth[node as usize] + 1;
            for to in g.out(node) {
                if depth[*to as usize] == UNREACHED {
                    depth[*to as usize] = next;
                    order.push(*to);
                }
                if depth[*to as usize] == next {
                    paths[*to as usize] += paths[node as usize];
                }
            }
        }

        // Then back in, furthest first, which is the order the accumulation
        // needs: a node cannot know what it owes until everything beyond it
        // does. The predecessors are read off the incoming side rather than
        // stored on the way out, which is what keeps this linear in memory.
        for node in order.iter().rev() {
            if depth[*node as usize] > 0 {
                let share = (1.0 + owed[*node as usize]) / paths[*node as usize];
                let back = depth[*node as usize] - 1;
                for to in g.into_(*node) {
                    if depth[*to as usize] == back {
                        owed[*to as usize] += paths[*to as usize] * share;
                    }
                }
            }
            if node != src {
                of[*node as usize] += owed[*node as usize];
            }
        }

        for node in &order {
            depth[*node as usize] = UNREACHED;
            paths[*node as usize] = 0.0;
            owed[*node as usize] = 0.0;
        }
    }

    Between {
        of,
        pivots: from.len() as u32,
        exact: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Both ways round, which is how an undirected graph is said here.
    fn undirected(edges: &[(u64, u64)]) -> Graph {
        let mut both: Vec<(u64, u64)> = Vec::new();
        for (a, b) in edges {
            both.push((*a, *b));
            both.push((*b, *a));
        }
        linked(&both)
    }

    /// Straight off the definition: for every pair, how many of the shortest
    /// paths between them go through each node in the middle.
    fn reference(g: &Snapshot) -> Vec<f64> {
        let n = g.nodes() as usize;
        // Shortest path counts and distances, from and to every node.
        let count = |src: u32, back: bool| {
            let mut far = vec![u32::MAX; n];
            let mut paths = vec![0f64; n];
            far[src as usize] = 0;
            paths[src as usize] = 1.0;
            let mut order = vec![src];
            let mut head = 0;
            while head < order.len() {
                let node = order[head];
                head += 1;
                let next = far[node as usize] + 1;
                let near = if back { g.into_(node) } else { g.out(node) };
                for to in near {
                    if far[*to as usize] == u32::MAX {
                        far[*to as usize] = next;
                        order.push(*to);
                    }
                    if far[*to as usize] == next {
                        paths[*to as usize] += paths[node as usize];
                    }
                }
            }
            (far, paths)
        };

        let out: Vec<(Vec<u32>, Vec<f64>)> = (0..n as u32).map(|s| count(s, false)).collect();
        let into: Vec<(Vec<u32>, Vec<f64>)> = (0..n as u32).map(|s| count(s, true)).collect();

        let mut of = vec![0f64; n];
        for (s, (far_s, count_s)) in out.iter().enumerate() {
            for (t, (far_t, count_t)) in into.iter().enumerate() {
                if s == t || far_s[t] == u32::MAX {
                    continue;
                }
                let (far, all) = (far_s[t], count_s[t]);
                for (v, of) in of.iter_mut().enumerate() {
                    if v == s || v == t {
                        continue;
                    }
                    let (there, back) = (far_s[v], far_t[v]);
                    if there == u32::MAX || back == u32::MAX || there + back != far {
                        continue;
                    }
                    *of += count_s[v] * count_t[v] / all;
                }
            }
        }
        of
    }

    #[test]
    fn the_middle_of_a_chain() {
        // 1 to 2 to 3, so only node 2 is ever in the middle, and it is in the
        // middle of the one pair that has to cross it.
        let s = Snapshot::of(&undirected(&[(1, 2), (2, 3)]));
        let c = betweenness_exact(&s);
        assert!((c.of(s.dense(2).expect("2")) - 2.0).abs() < 1e-9);
        assert_eq!(c.of(s.dense(1).expect("1")), 0.0);
        assert_eq!(c.of(s.dense(3).expect("3")), 0.0);
        assert!(c.exact());
    }

    #[test]
    fn the_bridge_between_two_halves() {
        let mut edges = Vec::new();
        for a in 0..5u64 {
            for b in a + 1..5 {
                edges.push((a, b));
                edges.push((a + 10, b + 10));
            }
        }
        edges.push((4, 10));
        let s = Snapshot::of(&undirected(&edges));
        let c = betweenness_exact(&s);
        let top = c.top(2);
        let ends = [s.dense(4).expect("4"), s.dense(10).expect("10")];
        assert!(ends.contains(&top[0].0), "{top:?}");
        assert!(ends.contains(&top[1].0), "{top:?}");
    }

    #[test]
    fn a_clique_spreads_it_evenly() {
        let mut edges = Vec::new();
        for a in 0..6u64 {
            for b in a + 1..6 {
                edges.push((a, b));
            }
        }
        let s = Snapshot::of(&undirected(&edges));
        let c = betweenness_exact(&s);
        // Everybody is next to everybody, so nobody is ever in the middle.
        assert!(c.scores().iter().all(|score| score.abs() < 1e-9));
    }

    #[test]
    fn it_agrees_with_the_definition() {
        let mut rng = Rng::new(0xb17e);
        for case in 0..40 {
            let nodes = 2 + rng.next_u64() % 25;
            let edges: Vec<(u64, u64)> = (0..nodes * 2)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            let (mine, theirs) = (betweenness_exact(&s), reference(&s));
            for node in 0..s.nodes() {
                let apart = (mine.of(node) - theirs[node as usize]).abs();
                assert!(apart < 1e-9, "case {case}, node {node}, {apart} out");
            }
        }
    }

    /// The same, on graphs read both ways, because an undirected graph has
    /// twice as many shortest paths to get wrong.
    #[test]
    fn it_agrees_with_the_definition_both_ways() {
        let mut rng = Rng::new(0xb17f);
        for case in 0..30 {
            let nodes = 3 + rng.next_u64() % 20;
            let edges: Vec<(u64, u64)> = (0..nodes)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&undirected(&edges));
            let (mine, theirs) = (betweenness_exact(&s), reference(&s));
            for node in 0..s.nodes() {
                assert!(
                    (mine.of(node) - theirs[node as usize]).abs() < 1e-9,
                    "case {case}, node {node}"
                );
            }
        }
    }

    /// The estimate has to put the same node at the top as the real answer on a
    /// graph where one node obviously belongs there.
    #[test]
    fn the_estimate_finds_the_bridge() {
        let mut edges = Vec::new();
        for group in 0..2u64 {
            for a in 0..30u64 {
                for b in a + 1..30 {
                    edges.push((group * 100 + a, group * 100 + b));
                }
            }
        }
        edges.push((29, 100));
        let s = Snapshot::of(&undirected(&edges));
        let sampled = betweenness_with(&s, 20);
        let exact = betweenness_exact(&s);
        assert!(!sampled.exact());
        assert_eq!(sampled.pivots(), 20);

        let ends = [s.dense(29).expect("29"), s.dense(100).expect("100")];
        assert!(ends.contains(&sampled.top(1)[0].0));
        assert!(ends.contains(&exact.top(1)[0].0));
    }

    /// And the estimate has to be near the real number, not just in the right
    /// order. Sampling half the sources gets an individual node wrong by a
    /// fifth of the largest score now and then, which is what an estimate is,
    /// so what is checked is the error across the whole graph: on average a
    /// small fraction of the largest score, and never wildly out.
    #[test]
    fn the_estimate_is_close() {
        let mut rng = Rng::new(0xb180);
        let nodes = 200u64;
        let edges: Vec<(u64, u64)> = (0..nodes * 4)
            .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
            .collect();
        let s = Snapshot::of(&undirected(&edges));
        let exact = betweenness_exact(&s);
        let sampled = betweenness_with(&s, 100);
        let most = exact.top(1)[0].1;

        let apart: Vec<f64> = (0..s.nodes())
            .map(|node| (sampled.of(node) - exact.of(node)).abs())
            .collect();
        let mean = apart.iter().sum::<f64>() / f64::from(s.nodes());
        let worst = apart.iter().copied().fold(0f64, f64::max);
        assert!(mean < most / 20.0, "{mean} on average out of {most}");
        assert!(worst < most / 3.0, "{worst} at worst out of {most}");
    }

    #[test]
    fn asking_for_everybody_is_the_exact_answer() {
        let s = Snapshot::of(&undirected(&[(1, 2), (2, 3), (3, 4)]));
        let all = betweenness_with(&s, 99);
        assert!(all.exact());
        assert_eq!(all.scores(), betweenness_exact(&s).scores());
    }

    #[test]
    fn nothing_at_all() {
        let c = betweenness(&Snapshot::default());
        assert!(c.scores().is_empty());
        assert!(c.top(3).is_empty());
        assert_eq!(c.pivots(), 0);
    }

    #[test]
    fn a_graph_with_no_edges() {
        let mut g = Graph::new();
        for id in 0..4u64 {
            g.add_node(id).expect("a node");
        }
        let c = betweenness(&Snapshot::of(&g));
        assert!(c.scores().iter().all(|score| *score == 0.0));
    }

    /// Direction is the whole answer here, unlike in the community algorithms.
    #[test]
    fn one_way_edges_are_read_one_way() {
        // A path that only runs one way, so 2 is in the middle of exactly one
        // ordered pair rather than two.
        let s = Snapshot::of(&linked(&[(1, 2), (2, 3)]));
        let c = betweenness_exact(&s);
        assert!((c.of(s.dense(2).expect("2")) - 1.0).abs() < 1e-9);
    }

    /// Two shortest paths through different nodes split the credit.
    #[test]
    fn a_tie_is_shared() {
        // 1 reaches 4 through either 2 or 3, in two hops either way.
        let s = Snapshot::of(&linked(&[(1, 2), (1, 3), (2, 4), (3, 4)]));
        let c = betweenness_exact(&s);
        assert!((c.of(s.dense(2).expect("2")) - 0.5).abs() < 1e-9);
        assert!((c.of(s.dense(3).expect("3")) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn two_runs_agree() {
        let mut rng = Rng::new(0xb181);
        let edges: Vec<(u64, u64)> = (0..200)
            .map(|_| (rng.next_u64() % 60, rng.next_u64() % 60))
            .collect();
        let s = Snapshot::of(&undirected(&edges));
        assert_eq!(
            betweenness_with(&s, 10).scores(),
            betweenness_with(&s, 10).scores()
        );
    }
}
