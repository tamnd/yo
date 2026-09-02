//! Numbering a graph by recursive bisection, which is where the bits are.
//!
//! [`csr`](crate::csr) says it plainly: on soc-LiveJournal1 the encoder is 1.18
//! bits over the entropy of the gaps it is given, so nothing left in the code is
//! worth having. What is left is in the numbering. A gap between two neighbours
//! is the distance between two node ids, and node ids are ours to choose, so the
//! question is which numbering makes the neighbours of a node land next to each
//! other. [`order_by_degree`](crate::csr::order_by_degree) answers it with one
//! sort and gets 0.73 bits on that graph. This answers it properly.
//!
//! # What this is
//!
//! Recursive graph bisection, from Dhulipala, Kabiljo, Karrer, Ottaviano and
//! Pupyrev, "Compressing Graphs and Indexes with Recursive Graph Bisection",
//! KDD 2016. The graph is read as a bipartite one: every node is a document to
//! be numbered, and the lists it appears in are its terms. Split the documents
//! in half, then repeatedly swap documents between the halves whenever the swap
//! lowers a cost function that stands in for the size of the compressed output,
//! then recurse into each half. The order the documents end up in is the
//! numbering.
//!
//! The cost of one term split across two halves of sizes `n1` and `n2`, holding
//! `d1` and `d2` of that term's occurrences, is
//!
//! ```text
//! d1 * log2(n1 / (d1 + 1)) + d2 * log2(n2 / (d2 + 1))
//! ```
//!
//! which is the paper's, and is what a list of `d` ids drawn out of a range of
//! `n` costs under a log gap code. Minimising the sum of that over every term is
//! minimising an estimate of the whole encoded size, and the estimate is good
//! enough that the real number moves with it.
//!
//! # What it is worth
//!
//! On the two public graphs, bits an edge through the cold form, against the ids
//! as they arrive and against [`order_by_degree`](crate::csr::order_by_degree):
//!
//! ```text
//!                    as they came   degree ordered   bisected
//! soc-LiveJournal1          17.99            19.00      15.00
//! web-Google                23.19            20.21      15.04
//! ```
//!
//! Twelve minutes on eight cores of server3 for the LiveJournal one, which is
//! sixty nine million edges, and eight seconds for web-Google.
//!
//! On R-MAT it wins nothing, 9.72 against degree ordering's 9.38, and that is
//! the useful control rather than a disappointment. R-MAT's structure is its
//! hubs: every list contains some of the same few thousand nodes, and giving
//! those the small ids is already close to the best numbering there is. A graph
//! with real communities has a different structure and this finds it. The test
//! below builds one out of sixty four groups of sixty four nodes, all of the
//! same degree so that degree ordering has nothing to sort on, and shuffles the
//! ids so nothing but the edges says where the groups are: 12.34 bits an edge as
//! they came, 12.33 degree ordered, 9.21 bisected.
//!
//! # Why this and not layered label propagation
//!
//! LLP is what WebGraph uses and it is what the note in `csr.rs` said was
//! coming. It is not here, and the reason is that the paper above is newer than
//! it, beats it on exactly the kind of graph we are stuck on, and is simpler to
//! be sure of. LLP runs label propagation at a sweep of resolution parameters
//! and keeps the clustering that codes best, so it has a parameter list, a
//! random restart and a quality criterion. Bisection has a leaf size and an
//! iteration cap, its objective is written down above, and every swap it makes
//! lowers that objective by an amount it can print. Facebook reported it beating
//! LLP on social graphs and being several times quicker; the later work on it
//! (Mackenzie and others, 2019 through 2021) is about faster convergence rather
//! than about a better objective, and Zuckerli, which is the strongest published
//! result on these graphs, keeps this ordering and improves the code that runs
//! after it.
//!
//! # What it is not
//!
//! It is not fast. It is `O(m log n)` with an iteration count on the front of
//! it, and on a graph with seventy million edges it is minutes rather than the
//! second [`order_by_degree`](crate::csr::order_by_degree) takes. That is the
//! trade the cold form is for: numbering happens once when a graph settles and
//! the encoded bytes are then read forever.
//!
//! It is not a clustering. The output is an order and nothing else. Two nodes
//! ending up adjacent means the encoder charges less for the pair, not that
//! anything here believes they are related.
//!
//! It is not adaptive. A graph that has changed since it was numbered stays
//! numbered the way it was, and the new edges go into the hot form. Renumbering
//! is a rebuild.

use yo_common::Rng;

/// The knobs, all three of them.
///
/// The defaults are the paper's, and both of the first two buy less than they
/// look like they should: doubling the iteration cap is worth hundredths of a
/// bit an edge because the swap loop stops early on almost every split, and
/// halving the leaf is worth about as much because sixteen ids in a group of
/// five hundred and twelve are already adjacent.
#[derive(Debug, Clone, Copy)]
pub struct Tuning {
    /// How many swap rounds one split gets before it moves on.
    ///
    /// A round that swaps nothing ends the split early, which is what usually
    /// happens well before this, so the cap is a bound on the worst case rather
    /// than a target.
    pub iterations: u32,
    /// The smallest partition worth splitting.
    pub leaf: u32,
    /// How many threads the recursion may spread over.
    ///
    /// One by default, because a numbering that depends on how many cores the
    /// machine had would be a numbering nobody can reproduce. This one does not:
    /// the split of a partition is decided before either half is recursed into,
    /// so the halves are independent and the answer is the same at any thread
    /// count. The cost of a thread is one scratch set, which is eight bytes a
    /// node.
    pub threads: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            iterations: 20,
            leaf: 16,
            threads: 1,
        }
    }
}

/// A partition below this many documents is recursed into on the same thread
/// whatever the budget says, because a thread costs a scratch set and a join and
/// a small partition is not worth either.
const SPLIT_OFF: usize = 1 << 16;

/// Number the nodes by recursive bisection, with the defaults.
///
/// Returns the new id of every old id, so `out[old]` is `new`, which is the
/// same shape [`order_by_degree`](crate::csr::order_by_degree) hands back and
/// takes the same [`renumber`](crate::csr::renumber) to apply.
#[must_use]
pub fn order(nodes: u32, edges: &[(u32, u32)]) -> Vec<u32> {
    order_with(nodes, edges, &Tuning::default())
}

/// The same with the knobs exposed.
///
/// # Panics
///
/// If a worker thread panics, which it does not, and the join is what would
/// otherwise swallow it.
#[must_use]
pub fn order_with(nodes: u32, edges: &[(u32, u32)], tuning: &Tuning) -> Vec<u32> {
    let mut to = vec![0u32; nodes as usize];
    if nodes == 0 {
        return to;
    }
    let lists = Lists::build(nodes, edges);
    // The starting order is the one the caller handed over, which is what the
    // paper does. Starting from a shuffle instead was measured and is a tenth
    // of a bit worse on R-MAT: the first split has to do all the work either
    // way, and starting from an order that already means something gives it a
    // better half to improve rather than a random one.
    let mut docs: Vec<u32> = (0..nodes).collect();
    let mut scratch = Scratch::new(nodes, lists.widest());
    descend(
        &lists,
        &mut docs,
        tuning,
        &mut scratch,
        tuning.threads.max(1),
    );
    for (new, old) in docs.iter().enumerate() {
        to[*old as usize] = new as u32;
    }
    to
}

/// The lists every document appears in.
///
/// A document is a node and its terms are the adjacency lists that hold it,
/// which for the out lists the cold form encodes means its in neighbours: two
/// nodes share a term when something points at both of them, and those are
/// exactly the pairs whose ids want to be close. Numbering for both directions
/// at once would take the union of the in and out neighbours here and nothing
/// else, and would be a worse answer for either direction on its own.
struct Lists {
    /// Where each document's terms start, with a final entry for the end.
    start: Vec<u64>,
    /// The terms, document by document.
    items: Vec<u32>,
}

impl Lists {
    fn build(nodes: u32, edges: &[(u32, u32)]) -> Lists {
        let n = nodes as usize;
        let mut start = vec![0u64; n + 1];
        for (_, d) in edges {
            start[*d as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut items = vec![0u32; edges.len()];
        let mut at = start.clone();
        for (s, d) in edges {
            items[at[*d as usize] as usize] = *s;
            at[*d as usize] += 1;
        }
        Lists { start, items }
    }

    #[inline]
    fn of(&self, doc: u32) -> &[u32] {
        let from = self.start[doc as usize] as usize;
        let to = self.start[doc as usize + 1] as usize;
        &self.items[from..to]
    }

    /// The largest number of documents any one term holds, which is how far the
    /// logarithm table has to go.
    fn widest(&self) -> u32 {
        let mut deg = vec![0u32; self.start.len() - 1];
        for t in &self.items {
            deg[*t as usize] += 1;
        }
        deg.into_iter().max().unwrap_or(0)
    }
}

/// The arrays one thread reuses down its whole recursion.
struct Scratch {
    /// How much of each term is in the left half, over the partition being
    /// split, and zero everywhere else.
    left_deg: Vec<u32>,
    /// The same for the right half.
    right_deg: Vec<u32>,
    /// The terms that are not zero, so the two above can be cleared without
    /// walking the whole graph at every one of the two million splits.
    touched: Vec<u32>,
    /// The left half as gain and document, sorted by gain.
    left: Vec<(f32, u32)>,
    /// The right half, the same way.
    right: Vec<(f32, u32)>,
    /// `log2(k)` for every `k` a degree can reach, plus one.
    ///
    /// The inner loop asks for four of these per edge it looks at, and the
    /// alternative is four calls to `log2` per edge, which is the difference
    /// between minutes and an afternoon on a real graph.
    log: Vec<f32>,
}

impl Scratch {
    fn new(nodes: u32, widest: u32) -> Scratch {
        let mut log = vec![0.0f32; widest as usize + 3];
        for (k, v) in log.iter_mut().enumerate() {
            *v = (k as f32).max(1.0).log2();
        }
        Scratch {
            left_deg: vec![0u32; nodes as usize],
            right_deg: vec![0u32; nodes as usize],
            touched: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            log,
        }
    }
}

/// Split one partition, then each of its halves.
fn descend(lists: &Lists, docs: &mut [u32], tuning: &Tuning, sc: &mut Scratch, budget: usize) {
    if docs.len() <= tuning.leaf.max(2) as usize {
        return;
    }
    let mid = docs.len() / 2;
    split(lists, docs, tuning, sc, mid);

    let (left, right) = docs.split_at_mut(mid);
    // The two halves share nothing that is still being read, so this is the
    // whole of the parallelism and it needs no locks. It also changes no
    // answer: the split above is finished before either side is looked at.
    if budget > 1 && right.len() >= SPLIT_OFF {
        let half = budget / 2;
        let widest = sc.log.len() as u32;
        std::thread::scope(|s| {
            let worker = s.spawn(|| {
                let mut own = Scratch::new(lists.start.len() as u32 - 1, widest);
                descend(lists, right, tuning, &mut own, budget - half);
            });
            descend(lists, left, tuning, sc, half);
            worker.join().expect("the recursion does not panic");
        });
    } else {
        descend(lists, left, tuning, sc, budget);
        descend(lists, right, tuning, sc, budget);
    }
}

/// Move documents across the middle for as long as it pays.
fn split(lists: &Lists, docs: &mut [u32], tuning: &Tuning, sc: &mut Scratch, mid: usize) {
    let Scratch {
        left_deg,
        right_deg,
        touched,
        left,
        right,
        log,
    } = sc;

    touched.clear();
    for (i, doc) in docs.iter().enumerate() {
        let left_side = i < mid;
        for term in lists.of(*doc) {
            let t = *term as usize;
            // A term nobody in this partition has touched yet is one that has
            // to be put back to zero on the way out, and this is the only place
            // that knows which those are.
            if left_deg[t] == 0 && right_deg[t] == 0 {
                touched.push(*term);
            }
            if left_side {
                left_deg[t] += 1;
            } else {
                right_deg[t] += 1;
            }
        }
    }

    let logn1 = (mid as f32).log2();
    let logn2 = ((docs.len() - mid) as f32).log2();
    for _ in 0..tuning.iterations {
        left.clear();
        right.clear();
        for doc in &docs[..mid] {
            left.push((
                gain(lists.of(*doc), left_deg, right_deg, logn1, logn2, log),
                *doc,
            ));
        }
        for doc in &docs[mid..] {
            right.push((
                gain(lists.of(*doc), right_deg, left_deg, logn2, logn1, log),
                *doc,
            ));
        }
        // Best first down both lists, with the document id breaking ties so a
        // graph always numbers the same way whatever the sort did with equal
        // keys.
        let by_gain = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        left.sort_unstable_by(by_gain);
        right.sort_unstable_by(by_gain);

        // The pair at the top of both lists is the one with the most to gain by
        // trading places, and once a pair is not worth trading neither is any
        // pair below it, because both lists only go down from here.
        let mut swaps = 0usize;
        for i in 0..mid.min(docs.len() - mid) {
            if left[i].0 + right[i].0 <= 0.0 {
                break;
            }
            let (a, b) = (left[i].1, right[i].1);
            for term in lists.of(a) {
                let t = *term as usize;
                left_deg[t] -= 1;
                right_deg[t] += 1;
            }
            for term in lists.of(b) {
                let t = *term as usize;
                right_deg[t] -= 1;
                left_deg[t] += 1;
            }
            left[i].1 = b;
            right[i].1 = a;
            swaps += 1;
        }

        for (k, e) in left.iter().enumerate() {
            docs[k] = e.1;
        }
        for (k, e) in right.iter().enumerate() {
            docs[mid + k] = e.1;
        }
        if swaps == 0 {
            break;
        }
    }

    for term in touched.iter() {
        left_deg[*term as usize] = 0;
        right_deg[*term as usize] = 0;
    }
}

/// What the objective drops by if this document changes sides.
///
/// `here` and `logn_here` are the half it is on now. A term that has this
/// document and nothing else on this side leaves an empty half behind, which the
/// cost function prices at zero, and that is the term that pays for most of what
/// this pass achieves.
#[inline]
fn gain(
    terms: &[u32],
    here: &[u32],
    there: &[u32],
    logn_here: f32,
    logn_there: f32,
    log: &[f32],
) -> f32 {
    let mut total = 0.0f32;
    for term in terms {
        let t = *term as usize;
        let (h, o) = (here[t], there[t]);
        let before = charge(h, logn_here, log) + charge(o, logn_there, log);
        let after = charge(h - 1, logn_here, log) + charge(o + 1, logn_there, log);
        total += before - after;
    }
    total
}

/// What one half of one term costs: `d * log2(n / (d + 1))`.
#[inline]
fn charge(d: u32, logn: f32, log: &[f32]) -> f32 {
    d as f32 * (logn - log[d as usize + 1])
}

/// A numbering that is worth nothing, for the control the tests need.
///
/// Bisection is only interesting if the graph has structure, and the way to
/// show that is to run it against a graph that has none and see it do nothing.
/// A shuffle is the other end of the same argument: a numbering this bad makes
/// the encoder pay what a structureless graph pays.
#[must_use]
pub fn shuffled(nodes: u32, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut order: Vec<u32> = (0..nodes).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut to = vec![0u32; nodes as usize];
    for (new, old) in order.iter().enumerate() {
        to[*old as usize] = new as u32;
    }
    to
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Csr;
    use crate::csr;

    /// R-MAT with the Graph500 probabilities, the same generator `csr.rs` uses,
    /// at a size a debug build can chew through.
    fn rmat(scale: u32, degree: u32, seed: u64) -> Vec<(u32, u32)> {
        let nodes = 1u32 << scale;
        let mut rng = Rng::new(seed);
        let mut edges = Vec::with_capacity((nodes as usize) * (degree as usize));
        for _ in 0..(nodes as u64) * u64::from(degree) {
            let (mut r, mut c) = (0u32, 0u32);
            for level in 0..scale {
                let bit = 1u32 << (scale - 1 - level);
                let p = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                if p < 0.57 {
                } else if p < 0.76 {
                    c |= bit;
                } else if p < 0.95 {
                    r |= bit;
                } else {
                    r |= bit;
                    c |= bit;
                }
            }
            edges.push((r, c));
        }
        edges
    }

    fn uniform(nodes: u32, degree: u32, seed: u64) -> Vec<(u32, u32)> {
        let mut rng = Rng::new(seed);
        let mut edges = Vec::with_capacity((nodes as usize) * (degree as usize));
        for src in 0..nodes {
            for _ in 0..degree {
                edges.push((src, (rng.next_u64() % u64::from(nodes)) as u32));
            }
        }
        edges
    }

    fn bits(nodes: u32, edges: &[(u32, u32)], to: &[u32]) -> f64 {
        let mut copy = edges.to_vec();
        csr::renumber(&mut copy, to);
        Csr::build(nodes, &mut copy).bits_per_edge()
    }

    /// A graph made of communities, with the ids shuffled so that nothing but
    /// the edges says where they are. Every node has the same degree, so degree
    /// ordering has nothing to sort by and this is a clean read on whether the
    /// pass finds the structure.
    fn communities(groups: u32, size: u32, inside: u32, across: u32, seed: u64) -> Vec<(u32, u32)> {
        let nodes = groups * size;
        let mut rng = Rng::new(seed);
        let mut edges = Vec::new();
        let names = shuffled(nodes, seed ^ 0x5eed);
        for src in 0..nodes {
            let home = (src / size) * size;
            for _ in 0..inside {
                let d = home + (rng.next_u64() % u64::from(size)) as u32;
                edges.push((names[src as usize], names[d as usize]));
            }
            for _ in 0..across {
                let d = (rng.next_u64() % u64::from(nodes)) as u32;
                edges.push((names[src as usize], names[d as usize]));
            }
        }
        edges
    }

    /// The whole point. On a graph whose communities are real, the numbering has
    /// to find them, and finding them has to be worth bits.
    #[test]
    fn bisection_beats_degree_ordering_on_a_graph_with_communities() {
        let (groups, size) = (64u32, 64u32);
        let nodes = groups * size;
        let edges = communities(groups, size, 12, 2, 7);
        let plain = bits(nodes, &edges, &identity(nodes));
        let degree = bits(nodes, &edges, &csr::order_by_degree(nodes, &edges));
        let bisected = bits(nodes, &edges, &order(nodes, &edges));
        assert!(
            bisected < degree - 2.0,
            "bisection {bisected:.2}, degree {degree:.2}, as they came {plain:.2}"
        );
    }

    /// R-MAT is the synthetic graph the rest of this crate measures on, and it
    /// is not a community graph: its hubs are shared by everything, so degree
    /// ordering is close to the best numbering there is for it and bisection
    /// does not beat it. Recording that here rather than leaving it to be
    /// rediscovered, because it is the reason the real graphs are the ones the
    /// module documentation quotes.
    #[test]
    fn r_mat_is_not_a_community_graph_and_degree_ordering_is_enough_for_it() {
        let scale = 12;
        let nodes = 1u32 << scale;
        let edges = rmat(scale, 8, 7);
        let plain = bits(nodes, &edges, &identity(nodes));
        let degree = bits(nodes, &edges, &csr::order_by_degree(nodes, &edges));
        let bisected = bits(nodes, &edges, &order(nodes, &edges));
        assert!(
            bisected < plain,
            "bisection {bisected:.2} did not even beat the ids as they came, {plain:.2}"
        );
        assert!(
            bisected > degree,
            "bisection {bisected:.2} now beats degree ordering {degree:.2} on R-MAT, which is a better result than this test was written for"
        );
    }

    /// The control. A graph with no structure has nothing for an ordering to
    /// find, and a pass that claimed a win here would be finding an artefact of
    /// the encoder rather than a property of the graph.
    #[test]
    fn there_is_nothing_to_win_on_a_graph_with_no_structure() {
        let nodes = 1u32 << 11;
        let edges = uniform(nodes, 8, 11);
        let plain = bits(nodes, &edges, &identity(nodes));
        let bisected = bits(nodes, &edges, &order(nodes, &edges));
        assert!(
            (bisected - plain).abs() < 0.5,
            "uniform moved from {plain:.2} to {bisected:.2}"
        );
    }

    /// A numbering has to be a numbering. Two nodes given the same id would
    /// silently drop edges at the renumber and the encoder would happily encode
    /// what was left.
    #[test]
    fn the_answer_is_a_permutation() {
        let nodes = 1u32 << 10;
        let edges = rmat(10, 8, 3);
        let to = order(nodes, &edges);
        let mut seen = vec![false; nodes as usize];
        for new in &to {
            assert!(!seen[*new as usize], "{new} twice");
            seen[*new as usize] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    /// The same graph numbers the same way twice, and the same way on any
    /// number of threads. A numbering that moved with the core count would make
    /// every published bits an edge number unreproducible.
    #[test]
    fn the_numbering_does_not_depend_on_the_machine() {
        let nodes = 1u32 << 11;
        let edges = rmat(11, 8, 5);
        let once = order(nodes, &edges);
        assert_eq!(once, order(nodes, &edges));
        let threaded = order_with(
            nodes,
            &edges,
            &Tuning {
                threads: 4,
                ..Tuning::default()
            },
        );
        assert_eq!(once, threaded);
    }

    /// The cost function has to say that a term entirely on one side is free,
    /// because that is the whole force pulling communities together.
    #[test]
    fn a_term_that_stays_together_is_charged_nothing() {
        let log: Vec<f32> = (0..16).map(|k| (k as f32).max(1.0).log2()).collect();
        // Four occurrences in a half of four is log2(4/5) a piece, which is
        // negative, so a term with nowhere else to be is better than free.
        assert!(charge(4, 2.0, &log) < 0.0);
        // Split evenly across two halves of four it costs something.
        assert!(charge(2, 2.0, &log) + charge(2, 2.0, &log) > charge(4, 2.0, &log));
    }

    /// Shuffling is the other control: it should make the encoder pay, and it
    /// should be undone by the pass rather than fought by it.
    #[test]
    fn a_shuffle_costs_and_bisection_takes_most_of_it_back() {
        let scale = 12;
        let nodes = 1u32 << scale;
        let mut edges = rmat(scale, 8, 13);
        let plain = bits(nodes, &edges, &identity(nodes));
        csr::renumber(&mut edges, &shuffled(nodes, 99));
        let shuffled_bits = bits(nodes, &edges, &identity(nodes));
        let bisected = bits(nodes, &edges, &order(nodes, &edges));
        assert!(shuffled_bits > plain, "{shuffled_bits:.2} vs {plain:.2}");
        assert!(
            bisected < shuffled_bits - 1.0,
            "shuffled {shuffled_bits:.2}, bisected {bisected:.2}"
        );
    }

    fn identity(nodes: u32) -> Vec<u32> {
        (0..nodes).collect()
    }
}
