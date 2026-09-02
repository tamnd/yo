//! Communities, by moving nodes until modularity stops going up.
//!
//! Two algorithms and the measure they both optimise.
//!
//! [`louvain()`] is Blondel, Guillaume, Lambiotte and Lefebvre, "Fast unfolding
//! of communities in large networks", J. Stat. Mech. 2008, with the queue driven
//! local move from Traag, "Faster unfolding of communities", Phys. Rev. E 2015.
//!
//! [`leiden()`] is Traag, Waltman and van Eck, "From Louvain to Leiden:
//! guaranteeing well-connected communities", Scientific Reports 2019.
//!
//! # What modularity is
//!
//! A community is supposed to be a group with more edges inside it than you
//! would expect by chance. Modularity is that sentence as a number: for each
//! community, the share of all edge ends that are inside it, minus the share you
//! would get if the same nodes kept their degrees and rewired at random. It runs
//! from about -0.5 to 1, and a real social graph split sensibly comes out around
//! 0.4 to 0.7.
//!
//! The resolution turns the dial on what "expected" means. Above one, chance
//! looks more likely and communities come out smaller; below one, larger. It is
//! the honest way to deal with modularity's resolution limit, which is that at
//! resolution one no method can see a community much smaller than the square
//! root of the edge count.
//!
//! # Why Leiden and not just Louvain
//!
//! Louvain has a defect that took eleven years to write down: a community it
//! returns can be internally disconnected. It happens when a node that was
//! acting as the only bridge inside its community moves out, and the community
//! is then aggregated into a single node before anybody notices it fell into two
//! pieces. Once aggregated the pieces can never be separated again. On real
//! graphs the 2019 paper found this in a few percent of communities, and it is
//! not a rounding error: a "community" in two halves with nothing joining them
//! is not a community by any reading.
//!
//! Leiden fixes it by putting a step between the moving and the aggregating. The
//! partition found by moving is refined: inside each community, nodes start
//! alone again and merge only into subsets that are well connected to the rest
//! of the community, and merges are chosen randomly among the good ones rather
//! than greedily. The graph is then aggregated on the refined subsets rather
//! than on the communities, so a community that fell into two pieces arrives at
//! the next level as two nodes and can still be pulled apart.
//!
//! Both are here because the difference is worth being able to measure, and
//! because Louvain is still the thing everybody else reports.
//!
//! # Why the answer is fixed
//!
//! The visiting order is random in both, and Leiden's merges are random by
//! design, which is where the guarantee comes from. All of it is drawn from
//! [`yo_common::Rng`] on a fixed seed, so two runs over one snapshot agree.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot, algo};
//!
//! let mut g = Graph::new();
//! // Two four cliques joined by a single edge.
//! for (a, b) in [(1u64, 2u64), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] {
//!     g.link(a, b, 1, NO_PROPS)?;
//!     g.link(a + 10, b + 10, 1, NO_PROPS)?;
//! }
//! g.link(4, 11, 1, NO_PROPS)?;
//!
//! let s = Snapshot::of(&g);
//! let c = algo::leiden(&s);
//! assert_eq!(c.count(), 2);
//! assert!(algo::modularity(&s, c.labels()) > 0.4);
//! # Ok::<(), yo_common::Error>(())
//! ```

use crate::Snapshot;
use crate::algo::{Components, tidy};
use yo_common::Rng;

/// The resolution that makes modularity mean what the 2004 paper says.
pub const RESOLUTION: f64 = 1.0;

/// How hard Leiden's refinement leans towards the best merge it can see.
///
/// The randomness is the point: a refinement that always took the best merge
/// would be Louvain's greedy step again and would lose the guarantee. Small
/// enough that a clearly better merge nearly always wins, large enough that a
/// close second is a real possibility, which is the value the paper uses.
const THETA: f64 = 0.01;

/// Enough levels that a real graph has run out of them long before.
///
/// Each level shrinks the graph to its community count, so a graph that has not
/// stopped in this many is one where a level is merging almost nothing.
const LEVELS: u32 = 64;

/// How many times the whole thing is run again from the original nodes.
///
/// Nearly always two: one to find a partition and one to confirm that nothing
/// wants to move out of it. The cap is there because a run whose randomness
/// keeps finding a different partition of the same quality would otherwise not
/// stop, and past a handful of them the difference is not worth the time.
const PASSES: u32 = 8;

const SEED: u64 = 0x1ead_e401;

/// The communities of `g`, by the Leiden method.
///
/// The one to use. It costs about a third more than [`louvain()`] and it cannot
/// hand back a community that is internally disconnected.
#[must_use]
pub fn leiden(g: &Snapshot) -> Components {
    leiden_with(g, RESOLUTION)
}

/// The same, at a resolution other than one.
#[must_use]
pub fn leiden_with(g: &Snapshot, resolution: f64) -> Components {
    unfold(g, resolution, true)
}

/// The communities of `g`, by the Louvain method.
///
/// Here to be measured against, and because it is what everybody else reports.
/// A community it returns may be internally disconnected, which is the whole
/// reason [`leiden()`] exists.
#[must_use]
pub fn louvain(g: &Snapshot) -> Components {
    louvain_with(g, RESOLUTION)
}

/// The same, at a resolution other than one.
#[must_use]
pub fn louvain_with(g: &Snapshot, resolution: f64) -> Components {
    unfold(g, resolution, false)
}

/// How good a partition of `g` is, at resolution one.
///
/// `labels` is one label per node in dense id order, and the labels themselves
/// mean nothing beyond which nodes share one, so anything that groups the nodes
/// will do: a [`Components`] from any of these algorithms, or a caller's own
/// grouping being checked against them.
///
/// # Panics
///
/// If `labels` is not one label per node.
#[must_use]
pub fn modularity(g: &Snapshot, labels: &[u32]) -> f64 {
    modularity_with(g, labels, RESOLUTION)
}

/// The same, at a resolution other than one.
///
/// # Panics
///
/// If `labels` is not one label per node.
#[must_use]
pub fn modularity_with(g: &Snapshot, labels: &[u32], resolution: f64) -> f64 {
    let w = Weighted::of(g);
    assert_eq!(labels.len(), w.nodes(), "one label a node");
    let (labels, groups) = renumber(labels);
    w.quality(&labels, groups, resolution)
}

/// The undirected weighted view the two methods run over.
///
/// Modularity is not a question about which way an edge points, so the two
/// directions of the snapshot are folded into one neighbour list per node with a
/// weight, and a pair of nodes joined twice becomes one edge of weight two. That
/// fold is also what makes aggregation work: the graph of communities is another
/// one of these, so a level is the same code as the level below it.
#[derive(Clone)]
struct Weighted {
    at: Vec<u64>,
    to: Vec<u32>,
    w: Vec<f64>,
    /// The weight of each node's own loop, counted once.
    self_w: Vec<f64>,
    /// Every edge end at a node, with a self loop counting twice.
    strength: Vec<f64>,
    /// Every edge end in the graph, which is what modularity divides by.
    total: f64,
}

impl Weighted {
    /// Both directions of a snapshot, folded together.
    fn of(g: &Snapshot) -> Weighted {
        let n = g.nodes() as usize;
        let mut at = vec![0u64; n + 1];
        for node in 0..n {
            let both = g.out_degree(node as u32) + g.in_degree(node as u32);
            at[node + 1] = at[node] + u64::from(both);
        }

        // Every incident edge lands in the list once: an edge stored from u to v
        // is in u's outgoing side and v's incoming side. A self loop is in both
        // sides of the same node, so it is taken off the outgoing side and left
        // out of the incoming one.
        let mut to = vec![0u32; at[n] as usize];
        let mut self_w = vec![0f64; n];
        let mut fill = at.clone();
        for node in 0..n as u32 {
            for other in g.out(node) {
                if *other == node {
                    self_w[node as usize] += 1.0;
                    continue;
                }
                to[fill[node as usize] as usize] = *other;
                fill[node as usize] += 1;
            }
            for other in g.into_(node) {
                if *other == node {
                    continue;
                }
                to[fill[node as usize] as usize] = *other;
                fill[node as usize] += 1;
            }
        }

        // Then the duplicates are combined, which is what turns a pair of nodes
        // joined three ways into one edge of weight three.
        let mut edges: Vec<(u32, f64)> = Vec::new();
        let mut out = Vec::with_capacity(to.len());
        let mut w = Vec::with_capacity(to.len());
        let mut next = vec![0u64; n + 1];
        for node in 0..n {
            let mine = &mut to[at[node] as usize..fill[node] as usize];
            mine.sort_unstable();
            edges.clear();
            for other in mine.iter() {
                match edges.last_mut() {
                    Some((last, weight)) if last == other => *weight += 1.0,
                    _ => edges.push((*other, 1.0)),
                }
            }
            for (other, weight) in &edges {
                out.push(*other);
                w.push(*weight);
            }
            next[node + 1] = out.len() as u64;
        }

        Weighted::new(next, out, w, self_w)
    }

    /// The pieces, with the strengths worked out from them.
    fn new(at: Vec<u64>, to: Vec<u32>, w: Vec<f64>, self_w: Vec<f64>) -> Weighted {
        let n = self_w.len();
        let mut strength = vec![0f64; n];
        for node in 0..n {
            let mine = at[node] as usize..at[node + 1] as usize;
            strength[node] = w[mine].iter().sum::<f64>() + 2.0 * self_w[node];
        }
        let total = strength.iter().sum();
        Weighted {
            at,
            to,
            w,
            self_w,
            strength,
            total,
        }
    }

    fn nodes(&self) -> usize {
        self.self_w.len()
    }

    /// One node's neighbours and what each edge weighs.
    fn near(&self, node: u32) -> (&[u32], &[f64]) {
        let mine = self.at[node as usize] as usize..self.at[node as usize + 1] as usize;
        (&self.to[mine.clone()], &self.w[mine])
    }

    /// The modularity of a partition, with the labels already numbered from
    /// zero.
    fn quality(&self, of: &[u32], groups: usize, resolution: f64) -> f64 {
        if self.total == 0.0 {
            return 0.0;
        }
        let mut inside = vec![0f64; groups];
        let mut tot = vec![0f64; groups];
        for node in 0..self.nodes() {
            let mine = of[node] as usize;
            tot[mine] += self.strength[node];
            inside[mine] += 2.0 * self.self_w[node];
            let (near, w) = self.near(node as u32);
            for (other, weight) in near.iter().zip(w) {
                if of[*other as usize] as usize == mine {
                    inside[mine] += weight;
                }
            }
        }
        (0..groups)
            .map(|c| inside[c] / self.total - resolution * (tot[c] / self.total).powi(2))
            .sum()
    }
}

/// The two methods, which differ only in what the graph is aggregated on.
///
/// The whole thing is run again from the original nodes, starting from the
/// partition the run before it found, until a run changes nothing. That is what
/// buys node optimality: a run that begins by asking every node on its own
/// whether it would rather be somewhere else, and ends with the same partition
/// it started from, has answered no for every node. One pass cannot say that,
/// because after the first level a node only ever moves as part of the group it
/// was aggregated into.
fn unfold(g: &Snapshot, resolution: f64, refined: bool) -> Components {
    let base = Weighted::of(g);
    let n = base.nodes();
    let mut answer: Vec<u32> = (0..n as u32).collect();
    if n == 0 {
        return tidy(answer);
    }

    let mut rng = Rng::new(SEED);
    for _ in 0..PASSES {
        let next = pass(&base, &answer, resolution, refined, &mut rng);
        if next == answer {
            break;
        }
        answer = next;
    }
    tidy(answer)
}

/// One run of the level loop, from the original nodes up.
fn pass(base: &Weighted, start: &[u32], resolution: f64, refined: bool, rng: &mut Rng) -> Vec<u32> {
    let n = base.nodes();
    let mut answer = vec![0u32; n];
    let mut w = base.clone();
    // Where each of the original nodes has ended up in the current graph.
    let mut at: Vec<u32> = (0..n as u32).collect();
    let (mut comm, _) = renumber(start);

    for _ in 0..LEVELS {
        local_move(&w, &mut comm, resolution, rng);
        let (tidied, groups) = renumber(&comm);
        for (node, at) in at.iter().enumerate() {
            answer[node] = tidied[*at as usize];
        }
        // Every node in a community of its own means there was nothing to
        // aggregate and the next level would do exactly this one again.
        if groups == w.nodes() {
            break;
        }

        let split = if refined {
            refine(&w, &tidied, groups, resolution, rng)
        } else {
            tidied.clone()
        };
        let (next, next_comm, moved) = aggregate(&w, &split, &tidied);
        for at in &mut at {
            *at = moved[*at as usize];
        }
        w = next;
        comm = next_comm;
    }
    // Numbered by the first node holding each label, so that two runs that
    // found the same partition hand back the same vector and the caller above
    // can tell they agreed.
    renumber(&answer).0
}

/// Move each node to whichever community it does modularity the most good in,
/// over and over until nobody wants to move.
///
/// The queue is Traag's 2015 point and it is most of the running time. A round
/// over every node is nearly all wasted once the partition is nearly settled,
/// because a node can only want to move if one of its neighbours moved. So the
/// nodes to look at are held in a queue, and moving a node puts its neighbours
/// back on it.
fn local_move(g: &Weighted, comm: &mut [u32], resolution: f64, rng: &mut Rng) {
    let n = g.nodes();
    if n == 0 || g.total == 0.0 {
        return;
    }
    let mut tot = vec![0f64; n];
    let mut size = vec![0u32; n];
    for node in 0..n {
        tot[comm[node] as usize] += g.strength[node];
        size[comm[node] as usize] += 1;
    }
    // Communities nobody is in, so that a node can leave for one when every
    // community it can see would be worse than being alone.
    let mut free: Vec<u32> = (0..n as u32).filter(|c| size[*c as usize] == 0).collect();

    let mut queue: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut queue, rng);
    let mut queued = vec![true; n];
    let mut head = 0usize;

    // The weight from the node being moved to each community it can see.
    let mut link = vec![0f64; n];
    let mut seen: Vec<u32> = Vec::new();

    while head < queue.len() {
        let node = queue[head];
        head += 1;
        queued[node as usize] = false;
        let was = comm[node as usize];
        let strength = g.strength[node as usize];

        tot[was as usize] -= strength;
        size[was as usize] -= 1;
        if size[was as usize] == 0 {
            free.push(was);
        }

        seen.clear();
        let (near, w) = g.near(node);
        for (other, weight) in near.iter().zip(w) {
            let at = comm[*other as usize] as usize;
            if link[at] == 0.0 {
                seen.push(comm[*other as usize]);
            }
            link[at] += weight;
        }

        // Staying put is the baseline, and an empty community is the floor: it
        // is worth nothing, which beats a neighbourhood that is worth less.
        let value = |c: u32, link: &[f64]| {
            link[c as usize] - resolution * strength * tot[c as usize] / g.total
        };
        let mut best = was;
        let mut most = value(was, &link);
        if most < 0.0 && size[was as usize] > 0 {
            while let Some(empty) = free.pop() {
                if size[empty as usize] == 0 {
                    (best, most) = (empty, 0.0);
                    free.push(empty);
                    break;
                }
            }
        }
        for c in &seen {
            let worth = value(*c, &link);
            // Ties go to the lowest numbered community, so the answer does not
            // depend on the order the neighbours came in.
            if worth > most || (worth == most && *c < best) {
                (best, most) = (*c, worth);
            }
        }
        for c in &seen {
            link[*c as usize] = 0.0;
        }

        comm[node as usize] = best;
        tot[best as usize] += strength;
        size[best as usize] += 1;
        if best == was {
            continue;
        }
        // Only a neighbour outside where this node landed can have been made
        // to want to move by it moving.
        for other in near {
            if comm[*other as usize] != best && !queued[*other as usize] {
                queued[*other as usize] = true;
                queue.push(*other);
            }
        }
    }
}

/// Split each community back into the pieces that are well connected to it.
///
/// This is the step Louvain does not have. Inside a community every node starts
/// alone again and merges only into a subset that is well connected to the rest
/// of the community, and only if the merge is worth something. A node acting as
/// the sole bridge inside its community, which is the case Louvain gets wrong,
/// is not well connected to it and so stays on its own and arrives at the next
/// level as its own node.
///
/// The merge is chosen at random among the ones worth making, weighted towards
/// the better ones. Taking the best one every time would be the greedy step
/// again, and the guarantee comes from not doing that.
fn refine(g: &Weighted, comm: &[u32], groups: usize, resolution: f64, rng: &mut Rng) -> Vec<u32> {
    let n = g.nodes();
    let mut refined: Vec<u32> = (0..n as u32).collect();
    if g.total == 0.0 {
        return refined;
    }

    // The nodes of each community, together.
    let mut at = vec![0u32; groups + 1];
    for c in comm {
        at[*c as usize + 1] += 1;
    }
    for c in 0..groups {
        at[c + 1] += at[c];
    }
    let mut member = vec![0u32; n];
    let mut fill = at.clone();
    for (node, c) in comm.iter().enumerate() {
        member[fill[*c as usize] as usize] = node as u32;
        fill[*c as usize] += 1;
    }

    // Per refined subset: its total strength, and how much of it points at the
    // rest of the community it lives in.
    let mut tot = g.strength.clone();
    let mut out = vec![0f64; n];
    let mut link = vec![0f64; n];
    let mut seen: Vec<u32> = Vec::new();
    let mut pick: Vec<(u32, f64)> = Vec::new();

    let mut order: Vec<u32> = Vec::new();
    for c in 0..groups {
        let mine = &member[at[c] as usize..at[c + 1] as usize];
        if mine.len() < 3 {
            continue;
        }
        let whole: f64 = mine.iter().map(|node| g.strength[*node as usize]).sum();

        // How much each node points at the rest of its own community, which is
        // both the well connected test for it and the starting value for the
        // subset it is alone in.
        for node in mine {
            let (near, w) = g.near(*node);
            out[*node as usize] = near
                .iter()
                .zip(w)
                .filter(|(other, _)| comm[**other as usize] as usize == c)
                .map(|(_, weight)| *weight)
                .sum();
        }

        order.clear();
        order.extend_from_slice(mine);
        shuffle(&mut order, rng);
        for node in &order {
            let node = *node;
            // Only a node still on its own can start a merge, and only one that
            // is well connected to the rest of its community should.
            if refined[node as usize] != node || tot[node as usize] != g.strength[node as usize] {
                continue;
            }
            let strength = g.strength[node as usize];
            if out[node as usize] < resolution * strength * (whole - strength) / g.total {
                continue;
            }

            seen.clear();
            let (near, w) = g.near(node);
            for (other, weight) in near.iter().zip(w) {
                if comm[*other as usize] as usize != c {
                    continue;
                }
                let into = refined[*other as usize] as usize;
                if into == node as usize {
                    continue;
                }
                if link[into] == 0.0 {
                    seen.push(refined[*other as usize]);
                }
                link[into] += weight;
            }

            pick.clear();
            let mut top = f64::NEG_INFINITY;
            for subset in &seen {
                let there = tot[*subset as usize];
                if out[*subset as usize] < resolution * there * (whole - there) / g.total {
                    continue;
                }
                let worth = link[*subset as usize] - resolution * strength * there / g.total;
                if worth >= 0.0 {
                    top = top.max(worth);
                    pick.push((*subset, worth));
                }
            }

            // Weighted by exp of the gain over theta, with the best one taken
            // off first so the exponential cannot run away.
            if !pick.is_empty() {
                let mut sum = 0.0;
                for (_, worth) in &mut pick {
                    *worth = ((*worth - top) / THETA).exp();
                    sum += *worth;
                }
                let mut want = uniform(rng) * sum;
                let mut into = pick[pick.len() - 1].0;
                for (subset, weight) in &pick {
                    want -= weight;
                    if want <= 0.0 {
                        into = *subset;
                        break;
                    }
                }
                refined[node as usize] = into;
                tot[into as usize] += strength;
                // What the subset points at the rest of the community changes by
                // what the node brought, less twice what the two already had
                // between them, which is now inside.
                out[into as usize] += out[node as usize] - 2.0 * link[into as usize];
                tot[node as usize] = 0.0;
            }

            for subset in &seen {
                link[*subset as usize] = 0.0;
            }
        }
    }
    refined
}

/// Shrink the graph so that each group of `split` is one node.
///
/// Returns the smaller graph, the community each of its nodes starts in, and
/// where each of the old nodes went. For Louvain `split` and `comm` are the same
/// thing, and every new node starts alone. For Leiden `split` is finer, and a
/// new node starts in the community the piece it came from belonged to, which is
/// what carries the partition down to the next level.
fn aggregate(g: &Weighted, split: &[u32], comm: &[u32]) -> (Weighted, Vec<u32>, Vec<u32>) {
    let (moved, n) = renumber(split);

    let mut self_w = vec![0f64; n];
    let mut edges: Vec<(u32, u32, f64)> = Vec::new();
    for node in 0..g.nodes() {
        let mine = moved[node];
        self_w[mine as usize] += g.self_w[node];
        let (near, w) = g.near(node as u32);
        for (other, weight) in near.iter().zip(w) {
            let theirs = moved[*other as usize];
            if theirs == mine {
                // Both ends of this edge are in here now, and it is going to be
                // seen once from each end.
                self_w[mine as usize] += weight / 2.0;
            } else {
                edges.push((mine, theirs, *weight));
            }
        }
    }

    edges.sort_unstable_by_key(|(from, to, _)| (*from, *to));
    let mut at = vec![0u64; n + 1];
    let mut to = Vec::new();
    let mut w = Vec::new();
    for (from, other, weight) in &edges {
        match to.last() {
            Some(last) if *last == *other && at[*from as usize + 1] == to.len() as u64 => {
                *w.last_mut().expect("a weight") += weight;
            }
            _ => {
                to.push(*other);
                w.push(*weight);
                at[*from as usize + 1] = to.len() as u64;
            }
        }
        at[*from as usize + 1] = to.len() as u64;
    }
    for node in 0..n {
        at[node + 1] = at[node + 1].max(at[node]);
    }

    // Which community each of the new nodes starts in.
    let mut starts = vec![0u32; n];
    for node in 0..g.nodes() {
        starts[moved[node] as usize] = comm[node];
    }
    let (starts, _) = renumber(&starts);
    (Weighted::new(at, to, w, self_w), starts, moved)
}

/// The same grouping with the labels numbered from zero, and how many there are.
fn renumber(of: &[u32]) -> (Vec<u32>, usize) {
    let mut seen = vec![u32::MAX; of.len()];
    let mut next = 0u32;
    let mut out = vec![0u32; of.len()];
    for (node, at) in of.iter().enumerate() {
        let seen = &mut seen[*at as usize];
        if *seen == u32::MAX {
            *seen = next;
            next += 1;
        }
        out[node] = *seen;
    }
    (out, next as usize)
}

/// Fisher and Yates, since the order nodes are looked at in is part of both.
fn shuffle(order: &mut [u32], rng: &mut Rng) {
    for at in (1..order.len()).rev() {
        order.swap(at, (rng.next_u64() % (at as u64 + 1)) as usize);
    }
}

/// A number in `[0, 1)`, off the top 53 bits, which is all a double holds.
fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{label_propagation, wcc};
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

    fn clique(first: u64, size: u64) -> Vec<(u64, u64)> {
        let mut edges = Vec::new();
        for a in first..first + size {
            for b in a + 1..first + size {
                edges.push((a, b));
            }
        }
        edges
    }

    /// Cliques in a ring, joined one edge apiece, which is the graph everybody
    /// tests community detection on because the answer is not in doubt.
    fn ring(groups: u64, size: u64) -> Vec<(u64, u64)> {
        let mut edges = Vec::new();
        for group in 0..groups {
            edges.extend(clique(group * 1000, size));
        }
        for group in 0..groups {
            edges.push((group * 1000, (group + 1) % groups * 1000 + 1));
        }
        edges
    }

    /// Modularity worked out the slow way, straight off the definition, one
    /// pair of nodes at a time.
    fn slow(g: &Snapshot, of: &[u32], resolution: f64) -> f64 {
        let n = g.nodes() as usize;
        let mut a = vec![vec![0f64; n]; n];
        for node in 0..n as u32 {
            for other in g.out(node) {
                a[node as usize][*other as usize] += 1.0;
                a[*other as usize][node as usize] += 1.0;
            }
        }
        let degree: Vec<f64> = (0..n).map(|node| a[node].iter().sum()).collect();
        let total: f64 = degree.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        let mut q = 0.0;
        for i in 0..n {
            for j in 0..n {
                if of[i] == of[j] {
                    q += a[i][j] - resolution * degree[i] * degree[j] / total;
                }
            }
        }
        q / total
    }

    #[test]
    fn the_measure_agrees_with_the_definition() {
        let mut rng = Rng::new(0x9d1);
        for case in 0..40 {
            let nodes = 2 + rng.next_u64() % 30;
            let edges: Vec<(u64, u64)> = (0..nodes * 2)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            let of: Vec<u32> = (0..s.nodes())
                .map(|_| (rng.next_u64() % 3) as u32)
                .collect();
            for resolution in [0.5, 1.0, 2.0] {
                let (mine, theirs) = (
                    modularity_with(&s, &of, resolution),
                    slow(&s, &of, resolution),
                );
                assert!(
                    (mine - theirs).abs() < 1e-9,
                    "case {case} at {resolution}, {mine} against {theirs}"
                );
            }
        }
    }

    #[test]
    fn the_measure_knows_a_good_split_from_a_bad_one() {
        let s = Snapshot::of(&linked(&ring(4, 8)));
        let good: Vec<u32> = (0..s.nodes()).map(|node| node / 8).collect();
        let one = vec![0u32; s.nodes() as usize];
        let each: Vec<u32> = (0..s.nodes()).collect();
        assert!(modularity(&s, &good) > 0.6, "{}", modularity(&s, &good));
        assert!((modularity(&s, &one)).abs() < 1e-9);
        assert!(modularity(&s, &each) < 0.0);
    }

    #[test]
    fn both_find_the_ring_of_cliques() {
        let s = Snapshot::of(&linked(&ring(6, 10)));
        for c in [leiden(&s), louvain(&s)] {
            assert_eq!(c.count(), 6);
            for group in 0..6u64 {
                let a = s.dense(group * 1000 + 2).expect("a");
                let b = s.dense(group * 1000 + 7).expect("b");
                assert!(c.same(a, b), "group {group}");
            }
        }
    }

    #[test]
    fn both_beat_label_propagation_for_modularity() {
        let s = Snapshot::of(&linked(&ring(8, 6)));
        let quick = modularity(&s, label_propagation(&s).labels());
        for c in [leiden(&s), louvain(&s)] {
            assert!(modularity(&s, c.labels()) >= quick - 1e-9);
        }
    }

    /// The Leiden guarantee, which is the reason it is here. Every community it
    /// hands back is connected inside itself.
    #[test]
    fn leiden_communities_are_never_disconnected() {
        let mut rng = Rng::new(0x1ead);
        for case in 0..30 {
            let nodes = 10 + rng.next_u64() % 90;
            let edges: Vec<(u64, u64)> = (0..nodes * 3)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            let c = leiden(&s);
            assert!(connected(&s, c.labels()), "case {case}");
        }
    }

    /// Whether every group of `of` is joined up inside itself, walked with the
    /// edges that stay inside the group.
    fn connected(g: &Snapshot, of: &[u32]) -> bool {
        let n = g.nodes() as usize;
        let mut seen = vec![false; n];
        let mut groups = std::collections::HashSet::new();
        for node in 0..n {
            if seen[node] || !groups.insert(of[node]) {
                if !seen[node] {
                    return false;
                }
                continue;
            }
            let mut todo = vec![node as u32];
            seen[node] = true;
            while let Some(at) = todo.pop() {
                for other in g.out(at).iter().chain(g.into_(at)) {
                    if of[*other as usize] == of[node] && !seen[*other as usize] {
                        seen[*other as usize] = true;
                        todo.push(*other);
                    }
                }
            }
        }
        true
    }

    #[test]
    fn a_community_never_crosses_a_component() {
        let mut rng = Rng::new(0x1ea0);
        for case in 0..30 {
            let nodes = 2 + rng.next_u64() % 50;
            let edges: Vec<(u64, u64)> = (0..nodes)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            let weak = wcc(&s);
            for c in [leiden(&s), louvain(&s)] {
                for node in 0..s.nodes() {
                    for other in 0..s.nodes() {
                        if c.same(node, other) {
                            assert!(weak.same(node, other), "case {case}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn one_clique_is_one_community() {
        let s = Snapshot::of(&linked(&clique(0, 15)));
        assert_eq!(leiden(&s).count(), 1);
        assert_eq!(louvain(&s).count(), 1);
    }

    #[test]
    fn a_higher_resolution_cuts_finer() {
        let s = Snapshot::of(&linked(&ring(4, 12)));
        let coarse = leiden_with(&s, 0.25).count();
        let plain = leiden_with(&s, 1.0).count();
        let fine = leiden_with(&s, 6.0).count();
        assert!(coarse <= plain, "{coarse} against {plain}");
        assert!(fine > plain, "{fine} against {plain}");
    }

    #[test]
    fn nothing_at_all() {
        for c in [leiden(&Snapshot::default()), louvain(&Snapshot::default())] {
            assert_eq!(c.count(), 0);
            assert!(c.is_empty());
        }
        assert_eq!(modularity(&Snapshot::default(), &[]), 0.0);
    }

    #[test]
    fn a_graph_with_no_edges_is_all_singletons() {
        let mut g = Graph::new();
        for id in 0..6u64 {
            g.add_node(id).expect("a node");
        }
        let s = Snapshot::of(&g);
        assert_eq!(leiden(&s).count(), 6);
        assert_eq!(louvain(&s).count(), 6);
    }

    #[test]
    fn a_self_loop_does_not_break_the_measure() {
        // A triangle with a loop on one corner is worth nothing however it is
        // split, so what is being checked is that the loop is counted the same
        // way by both sides and does not turn into a negative.
        let s = Snapshot::of(&linked(&[(1, 1), (1, 2), (2, 3), (3, 1)]));
        for c in [leiden(&s), louvain(&s)] {
            assert!(modularity(&s, c.labels()).abs() < 1e-9);
        }
    }

    #[test]
    fn two_runs_agree() {
        let s = Snapshot::of(&linked(&ring(5, 9)));
        assert_eq!(leiden(&s).labels(), leiden(&s).labels());
        assert_eq!(louvain(&s).labels(), louvain(&s).labels());
    }

    #[test]
    fn direction_does_not_matter() {
        let edges = ring(4, 8);
        let forward = Snapshot::of(&linked(&edges));
        let flipped: Vec<(u64, u64)> = edges.iter().map(|(a, b)| (*b, *a)).collect();
        let back = Snapshot::of(&linked(&flipped));
        assert_eq!(leiden(&forward).labels(), leiden(&back).labels());
    }

    /// Nothing either of them returns can be improved by moving one node.
    #[test]
    fn no_single_node_move_helps() {
        let mut rng = Rng::new(0x1ea2);
        for case in 0..15 {
            let nodes = 20 + rng.next_u64() % 40;
            let edges: Vec<(u64, u64)> = (0..nodes * 4)
                .map(|_| (rng.next_u64() % nodes, rng.next_u64() % nodes))
                .collect();
            let s = Snapshot::of(&linked(&edges));
            for c in [leiden(&s), louvain(&s)] {
                let mut of = c.labels().to_vec();
                let now = modularity(&s, &of);
                for node in 0..s.nodes() {
                    let was = of[node as usize];
                    for other in c.labels() {
                        of[node as usize] = *other;
                        let then = modularity(&s, &of);
                        assert!(then <= now + 1e-9, "case {case}, node {node}");
                    }
                    of[node as usize] = was;
                }
            }
        }
    }

    /// The labels are the lowest numbered node in each community.
    #[test]
    fn the_labels_are_tidy() {
        let s = Snapshot::of(&linked(&ring(4, 7)));
        let c = leiden(&s);
        for node in 0..s.nodes() {
            assert_eq!(c.of(c.of(node)), c.of(node));
            assert!(c.of(node) <= node);
        }
    }
}
