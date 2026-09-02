//! The graph flattened into dense arrays, which is the shape every algorithm
//! wants (`11` section 8).
//!
//! [`crate::Adjacency`] is built for a graph that is being changed: a run is
//! found by hashing the node and the label, it is grown and shrunk in place, and
//! a node is whatever `u64` the caller decided to call it. That is the right
//! structure for a traversal, which touches a handful of runs and wants each one
//! to be a probe and a sequential read, and it is the wrong structure for
//! PageRank, which touches every run twenty times and wants node zero's
//! neighbours to be the first thing in the array.
//!
//! So an algorithm runs over one of these instead. It is the same edges,
//! renumbered from zero, in one flat CSR per direction, taken at a point in
//! time. Nothing in here can change the graph and nothing here notices when the
//! graph changes, which is the trade: a snapshot goes stale, and in exchange
//! every algorithm below it is arrays and index arithmetic with no hashing on
//! any inner loop.
//!
//! ```
//! use yo_graph::{Graph, NO_PROPS, Snapshot};
//!
//! const FOLLOWS: u32 = 1;
//!
//! let mut g = Graph::new();
//! g.link(10, 20, FOLLOWS, NO_PROPS)?;
//! g.link(20, 30, FOLLOWS, NO_PROPS)?;
//!
//! let s = Snapshot::of(&g);
//! // Ids are renumbered from zero, in the order the graph's own ids sort.
//! assert_eq!(s.nodes(), 3);
//! assert_eq!(s.dense(20), Some(1));
//! assert_eq!(s.out(1), [2]);
//! assert_eq!(s.into_(1), [0]);
//! assert_eq!(s.id(2), 30);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # Why the numbering is sorted and not arrival order
//!
//! Because it has to be reproducible. Two snapshots of the same graph have to
//! give the same dense id to the same node, or a caller cannot hold a PageRank
//! vector from one run and a component table from another and compare them by
//! index. The property store hands its ids back in whatever order its table
//! happens to hold them, so the ids are sorted, and sorting is also the cheapest
//! way to build the reverse map for the case that matters.
//!
//! # Two ways to turn a graph's id into a dense one
//!
//! A graph whose ids came out of a counter, which is every graph the wire path
//! builds, has ids that are already `0..n`. That gets a direct table: one
//! `Vec<u32>` indexed by the id, one load per lookup, no comparisons. A graph
//! whose ids are hashes or timestamps gets a binary search over the sorted ids,
//! which is about twenty dependent loads instead of one.
//!
//! The rule is the table when the highest id is less than twice the node count,
//! so the table is never more than twice the size of the ids it replaces, and
//! the choice is made once when the snapshot is built rather than per lookup.
//! It only matters while the snapshot is being built, because after that every
//! algorithm works in dense ids and never asks.
//!
//! # The reverse index is transposed and not read
//!
//! The plane can index incoming edges and this does not use that. The incoming
//! CSR here is the transpose of the outgoing one, which is a counting sort over
//! the edges that were projected, because that is the only way the two can be
//! guaranteed to be the same set of edges. It also means an algorithm that needs
//! predecessors, which is most of the interesting ones, works on a graph built
//! with [`crate::Graph::out_only`].

use crate::{Dir, Graph};

/// A graph as dense arrays: ids renumbered from zero, one CSR per direction.
#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    /// The graph's own id for each dense id, ascending.
    ids: Vec<u64>,
    /// Where node `i`'s outgoing neighbours start, with a final total.
    out_at: Vec<u64>,
    /// Every outgoing neighbour, grouped by source.
    out_to: Vec<u32>,
    /// The same, transposed.
    in_at: Vec<u64>,
    in_to: Vec<u32>,
}

impl Snapshot {
    /// Every node and every edge of `g`, under every label.
    #[must_use]
    pub fn of(g: &Graph) -> Snapshot {
        let labels = g.labels().to_vec();
        Snapshot::labelled(g, &labels)
    }

    /// Every node of `g`, and the edges under the labels named.
    ///
    /// Every node, including the ones that have no edge under any of these
    /// labels, because an algorithm's answer is a vector indexed by node and a
    /// node that was left out of the numbering would shift every answer after
    /// it. An isolated node is an empty run and costs two offsets.
    #[must_use]
    pub fn labelled(g: &Graph, labels: &[u32]) -> Snapshot {
        let mut ids: Vec<u64> = g.node_props().iter().map(|(id, _)| id).collect();
        ids.sort_unstable();
        let n = ids.len();
        let map = Map::of(&ids);

        // One pass to count and one to fill, which is the standard CSR build.
        // The counts are accumulated one to the right so that the prefix sum
        // leaves the starts in place and the fill can use the same array as its
        // cursor.
        let mut out_at = vec![0u64; n + 1];
        for label in labels {
            g.adjacency().for_each_run(*label, Dir::Out, |node, ns, _| {
                let Some(at) = map.dense(node) else { return };
                out_at[at as usize + 1] += ns.len() as u64;
            });
        }
        for i in 0..n {
            out_at[i + 1] += out_at[i];
        }
        let mut out_to = vec![0u32; out_at[n] as usize];
        let mut cursor = out_at.clone();
        for label in labels {
            g.adjacency().for_each_run(*label, Dir::Out, |node, ns, _| {
                let Some(at) = map.dense(node) else { return };
                for to in ns {
                    // A neighbour the map does not know cannot happen: the
                    // plane's ends are nodes and every node is in the map. It
                    // is skipped rather than asserted because a snapshot is a
                    // read and a read should not be able to panic.
                    let Some(to) = map.dense(*to) else { continue };
                    out_to[cursor[at as usize] as usize] = to;
                    cursor[at as usize] += 1;
                }
            });
        }

        let (in_at, in_to) = transpose(n, &out_at, &out_to);
        Snapshot {
            ids,
            out_at,
            out_to,
            in_at,
            in_to,
        }
    }

    /// How many nodes there are, which is the length of every answer.
    #[must_use]
    pub fn nodes(&self) -> u32 {
        self.ids.len() as u32
    }

    /// How many edges were projected, counting a parallel edge as its own.
    #[must_use]
    pub fn edges(&self) -> u64 {
        self.out_to.len() as u64
    }

    /// Whether there is nothing here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The graph's own id for a dense one.
    ///
    /// # Panics
    ///
    /// If `node` is not a node of this snapshot, which is a bug in the caller:
    /// every dense id an algorithm can be holding came out of `0..nodes()`.
    #[must_use]
    pub fn id(&self, node: u32) -> u64 {
        self.ids[node as usize]
    }

    /// The dense id for one of the graph's, or `None` if it has no such node.
    #[must_use]
    pub fn dense(&self, id: u64) -> Option<u32> {
        self.ids.binary_search(&id).ok().map(|at| at as u32)
    }

    /// Node `node`'s outgoing neighbours.
    #[must_use]
    pub fn out(&self, node: u32) -> &[u32] {
        run(&self.out_at, &self.out_to, node)
    }

    /// Node `node`'s incoming neighbours.
    ///
    /// The trailing underscore is because `in` is a keyword, and the name is
    /// still `in` because that is the word for what it is.
    #[must_use]
    pub fn into_(&self, node: u32) -> &[u32] {
        run(&self.in_at, &self.in_to, node)
    }

    /// Neighbours in whichever direction, for an algorithm that takes one.
    #[must_use]
    pub fn neighbours(&self, node: u32, dir: Dir) -> &[u32] {
        match dir {
            Dir::Out => self.out(node),
            Dir::In => self.into_(node),
        }
    }

    /// How many edges leave `node`.
    #[must_use]
    pub fn out_degree(&self, node: u32) -> u32 {
        self.out(node).len() as u32
    }

    /// How many edges arrive at `node`.
    #[must_use]
    pub fn in_degree(&self, node: u32) -> u32 {
        self.into_(node).len() as u32
    }

    /// Ask the cache for a node's outgoing run, before the loop that reads it.
    ///
    /// The same call [`crate::Adjacency::prefetch`] is for and much cheaper to
    /// serve, because a dense run is one load of the offset and then a
    /// contiguous read rather than a hash and a probe.
    pub fn prefetch(&self, node: u32) {
        let at = self.out_at[node as usize] as usize;
        if at < self.out_to.len() {
            yo_common::prefetch(&self.out_to[at]);
        }
    }

    /// Resident bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.ids.capacity() * size_of::<u64>()
            + (self.out_at.capacity() + self.in_at.capacity()) * size_of::<u64>()
            + (self.out_to.capacity() + self.in_to.capacity()) * size_of::<u32>()
    }
}

/// One node's slice out of a CSR.
#[inline]
fn run<'a>(at: &[u64], to: &'a [u32], node: u32) -> &'a [u32] {
    let i = node as usize;
    let (from, upto) = (at[i] as usize, at[i + 1] as usize);
    &to[from..upto]
}

/// The reverse CSR, as a counting sort over the forward one.
fn transpose(n: usize, out_at: &[u64], out_to: &[u32]) -> (Vec<u64>, Vec<u32>) {
    let mut in_at = vec![0u64; n + 1];
    for to in out_to {
        in_at[*to as usize + 1] += 1;
    }
    for i in 0..n {
        in_at[i + 1] += in_at[i];
    }
    let mut in_to = vec![0u32; out_to.len()];
    let mut cursor = in_at.clone();
    for from in 0..n {
        let (a, b) = (out_at[from] as usize, out_at[from + 1] as usize);
        for to in &out_to[a..b] {
            in_to[cursor[*to as usize] as usize] = from as u32;
            cursor[*to as usize] += 1;
        }
    }
    (in_at, in_to)
}

/// A graph's ids to dense ones, either way round.
#[derive(Debug)]
enum Map<'a> {
    /// The ids are close enough to `0..n` to index an array with.
    Table(Vec<u32>),
    /// They are not, so they are searched for.
    Search(&'a [u64]),
}

/// A dense id that no node has, for a hole in the table.
const NONE: u32 = u32::MAX;

impl<'a> Map<'a> {
    /// Whichever of the two suits these ids, which are sorted.
    fn of(ids: &'a [u64]) -> Map<'a> {
        let top = ids.last().copied().unwrap_or(0);
        // Twice the node count, so the table is never more than twice the size
        // of the sorted ids it is standing in for, and a graph numbered from a
        // counter is always under it.
        if !ids.is_empty() && top < 2 * ids.len() as u64 {
            let mut table = vec![NONE; top as usize + 1];
            for (at, id) in ids.iter().enumerate() {
                table[*id as usize] = at as u32;
            }
            return Map::Table(table);
        }
        Map::Search(ids)
    }

    #[inline]
    fn dense(&self, id: u64) -> Option<u32> {
        match self {
            Map::Table(table) => match table.get(id as usize) {
                Some(&NONE) | None => None,
                Some(at) => Some(*at),
            },
            Map::Search(ids) => ids.binary_search(&id).ok().map(|at| at as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NO_PROPS;

    /// A graph with a hop in it, built the way the wire path builds one.
    fn chain(n: u64) -> Graph {
        let mut g = Graph::new();
        for i in 0..n - 1 {
            g.link(i, i + 1, 1, NO_PROPS).unwrap();
        }
        g
    }

    #[test]
    fn the_dense_ids_are_the_graphs_ids_in_order() {
        let mut g = Graph::new();
        g.link(900, 100, 1, NO_PROPS).unwrap();
        g.link(100, 500, 1, NO_PROPS).unwrap();
        let s = Snapshot::of(&g);
        assert_eq!(s.nodes(), 3);
        assert_eq!(s.edges(), 2);
        assert_eq!((s.id(0), s.id(1), s.id(2)), (100, 500, 900));
        assert_eq!(s.dense(500), Some(1));
        assert_eq!(s.dense(501), None);
        assert_eq!(s.out(0), [1]);
        assert_eq!(s.out(2), [0]);
        assert_eq!(s.into_(0), [2]);
        assert!(s.out(1).is_empty());
    }

    /// An isolated node is in the numbering, or every answer after it would be
    /// about a different node than the caller thinks.
    #[test]
    fn a_node_with_no_edges_is_still_a_node() {
        let mut g = Graph::new();
        g.link(0, 2, 1, NO_PROPS).unwrap();
        g.add_node(1).unwrap();
        let s = Snapshot::of(&g);
        assert_eq!(s.nodes(), 3);
        assert_eq!(s.id(1), 1);
        assert!(s.out(1).is_empty());
        assert!(s.into_(1).is_empty());
        assert_eq!(s.out(0), [2]);
    }

    #[test]
    fn only_the_labels_asked_for_come_along() {
        let mut g = Graph::new();
        g.link(0, 1, 7, NO_PROPS).unwrap();
        g.link(0, 2, 9, NO_PROPS).unwrap();
        let all = Snapshot::of(&g);
        assert_eq!(all.out(0), [1, 2]);
        let one = Snapshot::labelled(&g, &[9]);
        assert_eq!(one.nodes(), 3, "every node, whatever the labels");
        assert_eq!(one.out(0), [2]);
        assert_eq!(one.edges(), 1);
        let none = Snapshot::labelled(&g, &[]);
        assert_eq!(none.nodes(), 3);
        assert_eq!(none.edges(), 0);
    }

    /// The transpose is the forward index read the other way, edge for edge,
    /// including a parallel edge and a self loop.
    #[test]
    fn the_reverse_index_is_the_forward_one_transposed() {
        let mut g = Graph::new();
        g.link(0, 1, 1, NO_PROPS).unwrap();
        g.link(0, 1, 1, NO_PROPS).unwrap();
        g.link(1, 1, 1, NO_PROPS).unwrap();
        let s = Snapshot::of(&g);
        assert_eq!(s.out(0), [1, 1]);
        assert_eq!(s.into_(1), [0, 0, 1]);
        assert_eq!(s.out(1), [1]);
        assert_eq!(s.edges(), 3);
        assert_eq!(s.in_degree(1), 3);
        assert_eq!(s.out_degree(0), 2);

        let mut forward = 0;
        let mut back = 0;
        for i in 0..s.nodes() {
            forward += s.out(i).len();
            back += s.into_(i).len();
        }
        assert_eq!(forward, back);
    }

    /// A graph built with only outgoing edges indexed still gets predecessors,
    /// because the reverse index here is built and not read.
    #[test]
    fn an_out_only_graph_still_has_predecessors() {
        let mut g = Graph::out_only();
        g.link(0, 1, 1, NO_PROPS).unwrap();
        g.link(2, 1, 1, NO_PROPS).unwrap();
        assert!(g.neighbours(1, 1, Dir::In).is_empty(), "not in the plane");
        let s = Snapshot::of(&g);
        assert_eq!(s.into_(1), [0, 2]);
    }

    /// The two maps are the same map, which is what makes the fast one safe to
    /// choose.
    #[test]
    fn a_dense_numbering_and_a_scattered_one_agree() {
        let dense: Vec<u64> = (0..64).collect();
        let scattered: Vec<u64> = (0..64).map(|i| i * 1000 + 7).collect();
        let table = Map::of(&dense);
        let search = Map::of(&scattered);
        assert!(matches!(table, Map::Table(_)), "a counter gets the table");
        assert!(matches!(search, Map::Search(_)), "hashes get the search");
        for i in 0..64u64 {
            assert_eq!(table.dense(i), Some(i as u32));
            assert_eq!(search.dense(i * 1000 + 7), Some(i as u32));
        }
        assert_eq!(table.dense(64), None);
        assert_eq!(table.dense(u64::MAX), None);
        assert_eq!(search.dense(8), None);

        // A hole in the middle is a hole and not the node next to it.
        let holed = [0u64, 1, 3];
        let map = Map::of(&holed);
        assert!(matches!(map, Map::Table(_)));
        assert_eq!(map.dense(2), None);
        assert_eq!(map.dense(3), Some(2));
    }

    #[test]
    fn an_empty_graph_snapshots_to_nothing() {
        let s = Snapshot::of(&Graph::new());
        assert!(s.is_empty());
        assert_eq!(s.nodes(), 0);
        assert_eq!(s.edges(), 0);
        assert_eq!(s.dense(0), None);
    }

    #[test]
    fn a_long_chain_reads_back_end_to_end() {
        let s = Snapshot::of(&chain(10_000));
        assert_eq!(s.nodes(), 10_000);
        assert_eq!(s.edges(), 9_999);
        for i in 0..s.nodes() - 1 {
            assert_eq!(s.out(i), [i + 1], "at {i}");
        }
        assert!(s.out(9_999).is_empty());
        s.prefetch(0);
        assert!(s.memory_bytes() > 0);
    }
}
