//! PageRank, pulled rather than pushed.
//!
//! Page, Brin, Motwani and Winograd, "The PageRank Citation Ranking: Bringing
//! Order to the Web", 1999, in the pull form the GAP benchmark suite measures.
//!
//! # Pull, not push
//!
//! The same round can be written two ways. Push walks the outgoing edges and
//! adds each node's share into its neighbours, which writes to a random place in
//! the score array for every edge in the graph. Pull walks the incoming edges
//! and reads each neighbour's share, which reads from a random place and writes
//! to one that is sequential.
//!
//! Both do the same arithmetic and the pull form is faster on one core for the
//! ordinary reason: a random read can be in flight next to nine other random
//! reads, and a random write cannot be reordered against a read of the same
//! array. It is also the form that stays correct if somebody adds threads later,
//! because two threads pulling into different nodes never write the same word,
//! whereas two threads pushing into the same neighbour do.
//!
//! # Dangling nodes get their mass back
//!
//! A node with no outgoing edges has a score and nowhere to send it. The GAP
//! implementation drops it, so the vector it computes sums to less than one and
//! the shortfall grows with how many dead ends a graph has. That is fine for
//! ranking, since every score is short by roughly the same factor, and it is
//! wrong for anything that reads a score as a probability.
//!
//! So the dangling mass is collected each round and spread over every node,
//! which is the random surfer restarting when they hit a page with no links, and
//! which is what the original paper describes. The vector sums to one, and the
//! cost is one extra pass over an array that was being read anyway.
//!
//! # Where the precision goes
//!
//! Scores are `f32` and every sum is accumulated in `f64` before it is stored.
//! The array of shares is read once per edge from a random offset, so its size
//! is what the round costs on a big graph, and eight bytes a node would double
//! that traffic to buy precision that ranking cannot use. The accumulator is the
//! part that actually needs the bits, because a node with a million incoming
//! edges is adding a million small numbers, and that is where a naive `f32`
//! running total loses digits.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! // Everybody points at 99, so 99 wins.
//! for i in 0..5u64 {
//!     g.link(i, 99, 1, NO_PROPS)?;
//! }
//!
//! let r = algo::pagerank(&Snapshot::of(&g));
//! assert!(r.converged());
//! assert_eq!(r.top(1)[0].0, Snapshot::of(&g).dense(99).unwrap());
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;

/// How often the random surfer follows a link rather than starting again.
///
/// The paper's 0.85, and everybody else's, so a score from here is comparable
/// with a score from anywhere else. It is worth knowing that the number is also
/// what decides how long a round takes to converge: the error falls by a factor
/// of the damping each round, so 0.85 needs about forty rounds to reach 1e-3 and
/// 0.99 needs about seven hundred.
pub const DAMPING: f32 = 0.85;

/// How still the vector has to be, summed over every node, to call it done.
pub const EPSILON: f64 = 1e-6;

/// How many rounds to run before giving up on converging.
///
/// At the default damping the vector is still long before this, so hitting the
/// cap means either a damping close to one or a graph that is pathological, and
/// either way the caller wants to be told rather than made to wait.
pub const ROUNDS: u32 = 100;

/// What a run worked out, and how sure it is.
#[derive(Debug, Clone, Default)]
pub struct Rank {
    of: Vec<f32>,
    rounds: u32,
    delta: f64,
    settled: bool,
}

impl Rank {
    /// One node's score.
    ///
    /// # Panics
    ///
    /// If `node` is not a node of the snapshot this was computed over.
    #[must_use]
    pub fn of(&self, node: u32) -> f32 {
        self.of[node as usize]
    }

    /// Every score, indexed by dense id.
    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.of
    }

    /// How many rounds it took.
    #[must_use]
    pub fn rounds(&self) -> u32 {
        self.rounds
    }

    /// How much the last round moved the vector, summed over every node.
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Whether it settled rather than running out of rounds.
    ///
    /// Against the epsilon this run was asked for, not the default one, so a
    /// caller who asked for something stricter is told the truth about it.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.settled
    }

    /// The `k` highest scoring nodes, best first.
    ///
    /// Ties go to the lower dense id, so two runs over the same graph list them
    /// in the same order. Asking for more than there are gives all of them.
    #[must_use]
    pub fn top(&self, k: usize) -> Vec<(u32, f32)> {
        let mut all: Vec<(u32, f32)> = self
            .of
            .iter()
            .enumerate()
            .map(|(at, score)| (at as u32, *score))
            .collect();
        let k = k.min(all.len());
        if k < all.len() {
            // Only the first k have to be in order, and a full sort of a
            // hundred million nodes to answer top ten is most of the run.
            all.select_nth_unstable_by(k, |a, b| better(*a, *b));
            all.truncate(k);
        }
        all.sort_unstable_by(|a, b| better(*a, *b));
        all
    }
}

/// Higher score first, and the lower node when they are the same.
fn better(a: (u32, f32), b: (u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// PageRank at the usual damping, to the usual precision.
#[must_use]
pub fn pagerank(g: &Snapshot) -> Rank {
    pagerank_with(g, DAMPING, EPSILON, ROUNDS)
}

/// PageRank with the three numbers spelled out.
///
/// `epsilon` is on the sum of the per node change, not on the largest one, so it
/// gets harder to reach as the graph gets bigger. That is deliberate: it is the
/// L1 distance between one round and the next, which is the thing that says the
/// distribution has stopped moving.
#[must_use]
pub fn pagerank_with(g: &Snapshot, damping: f32, epsilon: f64, rounds: u32) -> Rank {
    let n = g.nodes() as usize;
    if n == 0 {
        // Nothing to move, so nothing left to settle.
        return Rank {
            settled: true,
            ..Rank::default()
        };
    }

    let start = 1.0 / n as f32;
    let mut score = vec![start; n];
    let mut share = vec![0f32; n];
    let mut delta = f64::INFINITY;
    let mut round = 0;

    while round < rounds && delta >= epsilon {
        // What each node hands to each of its neighbours this round, worked out
        // once so the inner loop is a read rather than a read and a divide.
        let mut stuck = 0f64;
        for node in 0..n {
            let out = g.out_degree(node as u32);
            if out == 0 {
                stuck += f64::from(score[node]);
                share[node] = 0.0;
            } else {
                share[node] = score[node] / out as f32;
            }
        }
        // Everyone gets the same floor: the surfer who jumped, plus the surfer
        // who landed on a dead end and had to jump.
        let base = ((1.0 - f64::from(damping)) + f64::from(damping) * stuck) / n as f64;

        delta = 0.0;
        for (node, score) in score.iter_mut().enumerate() {
            let mut sum = 0f64;
            for from in g.into_(node as u32) {
                sum += f64::from(share[*from as usize]);
            }
            let next = base + f64::from(damping) * sum;
            delta += (next - f64::from(*score)).abs();
            *score = next as f32;
        }
        round += 1;
    }

    Rank {
        of: score,
        rounds: round,
        delta,
        settled: delta < epsilon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;
    use crate::{Graph, Snapshot};
    use yo_common::Rng;

    /// The same iteration written the obvious way, in double the precision, as
    /// the thing the real one has to agree with.
    fn reference(g: &Snapshot, rounds: u32) -> Vec<f64> {
        let n = g.nodes() as usize;
        let d = f64::from(DAMPING);
        let mut score = vec![1.0 / n as f64; n];
        for _ in 0..rounds {
            let mut next = vec![0f64; n];
            let mut stuck = 0f64;
            for (node, score) in score.iter().enumerate() {
                let out = g.out_degree(node as u32);
                if out == 0 {
                    stuck += score;
                } else {
                    let share = score / f64::from(out);
                    for to in g.out(node as u32) {
                        next[*to as usize] += share;
                    }
                }
            }
            let base = ((1.0 - d) + d * stuck) / n as f64;
            for got in &mut next {
                *got = base + d * *got;
            }
            score = next;
        }
        score
    }

    fn linked(edges: &[(u64, u64)]) -> Graph {
        let mut g = Graph::new();
        for (from, to) in edges {
            g.link(*from, *to, 1, NO_PROPS).expect("an edge");
        }
        g
    }

    #[test]
    fn a_ring_gives_everybody_the_same_score() {
        let edges: Vec<(u64, u64)> = (0..10u64).map(|i| (i, (i + 1) % 10)).collect();
        let s = Snapshot::of(&linked(&edges));
        let r = pagerank(&s);
        assert!(r.converged(), "a ring settles");
        for node in 0..s.nodes() {
            assert!((r.of(node) - 0.1).abs() < 1e-5, "{}", r.of(node));
        }
    }

    #[test]
    fn the_node_everybody_points_at_wins() {
        let edges: Vec<(u64, u64)> = (0..20u64).map(|i| (i, 100)).collect();
        let s = Snapshot::of(&linked(&edges));
        let r = pagerank(&s);
        let hub = s.dense(100).expect("the hub");
        let top = r.top(3);
        assert_eq!(top[0].0, hub);
        // Everybody else is a leaf with nothing pointing at them, so they are
        // all on the floor and the hub is far above it.
        assert!(top[0].1 > 10.0 * top[1].1, "{top:?}");
    }

    #[test]
    fn the_scores_add_up_to_one() {
        // A graph with dead ends in it, which is the case where dropping the
        // dangling mass would show.
        let s = Snapshot::of(&linked(&[(0, 1), (1, 2), (3, 1), (4, 5)]));
        let r = pagerank(&s);
        let total: f64 = r.scores().iter().map(|s| f64::from(*s)).sum();
        assert!((total - 1.0).abs() < 1e-4, "{total}");
    }

    #[test]
    fn no_damping_is_the_uniform_vector() {
        let s = Snapshot::of(&linked(&[(0, 1), (1, 2), (2, 0), (3, 0)]));
        let r = pagerank_with(&s, 0.0, EPSILON, ROUNDS);
        for node in 0..s.nodes() {
            assert!((r.of(node) - 0.25).abs() < 1e-6, "{}", r.of(node));
        }
    }

    #[test]
    fn an_empty_graph_has_no_scores() {
        let r = pagerank(&Snapshot::default());
        assert!(r.scores().is_empty());
        assert_eq!(r.rounds(), 0);
        assert!(r.top(5).is_empty());
    }

    #[test]
    fn one_node_holds_everything() {
        let mut g = Graph::new();
        g.add_node(7).expect("a node");
        let r = pagerank(&Snapshot::of(&g));
        assert!((r.of(0) - 1.0).abs() < 1e-6, "{}", r.of(0));
    }

    #[test]
    fn a_self_loop_keeps_what_it_is_given() {
        let s = Snapshot::of(&linked(&[(0, 0), (1, 0), (2, 0)]));
        let r = pagerank(&s);
        let sink = s.dense(0).expect("the sink");
        assert!(r.of(sink) > 0.7, "{}", r.of(sink));
    }

    #[test]
    fn two_runs_agree_to_the_bit() {
        let mut rng = Rng::new(0x51ee);
        let edges: Vec<(u64, u64)> = (0..2000)
            .map(|_| (rng.next_u64() % 300, rng.next_u64() % 300))
            .collect();
        let s = Snapshot::of(&linked(&edges));
        assert_eq!(pagerank(&s).scores(), pagerank(&s).scores());
    }

    #[test]
    fn it_says_when_it_ran_out_of_rounds() {
        let s = Snapshot::of(&linked(&[(0, 1), (1, 2), (2, 0)]));
        let r = pagerank_with(&s, DAMPING, 1e-30, 5);
        assert_eq!(r.rounds(), 5);
        assert!(!r.converged());
        assert!(r.delta() > 0.0);
    }

    /// Against the obvious implementation, over graphs nobody chose.
    #[test]
    fn it_agrees_with_the_slow_one() {
        let mut rng = Rng::new(0xbead);
        for case in 0..40 {
            let nodes = 2 + rng.next_u64() % 60;
            let edges: Vec<(u64, u64)> = (0..nodes * 3)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            let mine = pagerank(&s);
            let theirs = reference(&s, mine.rounds());
            for node in 0..s.nodes() {
                let (a, b) = (f64::from(mine.of(node)), theirs[node as usize]);
                assert!((a - b).abs() < 1e-5, "case {case} node {node}: {a} {b}");
            }
        }
    }
}
