//! A property graph: the adjacency plane with a document behind every node and
//! every edge (`11` section 3).
//!
//! [`crate::Adjacency`] is the structure and [`crate::Props`] is what hangs off
//! it. This is the two of them together, plus the one thing neither of them can
//! own on its own: which edge slot an edge got.
//!
//! ```
//! use yo_doc::Builder;
//! use yo_graph::{Dir, Graph};
//!
//! const FOLLOWS: u32 = 1;
//!
//! fn doc(f: impl FnOnce(&mut Builder) -> yo_common::Result<()>) -> Vec<u8> {
//!     let mut b = Builder::new();
//!     f(&mut b).unwrap();
//!     b.finish().unwrap().to_vec()
//! }
//!
//! let mut g = Graph::new();
//! g.put_node(1, &doc(|b| { b.begin_object()?; b.key(b"name")?; b.text("ada")?; b.end_object() }))?;
//! g.put_node(2, &doc(|b| { b.begin_object()?; b.key(b"name")?; b.text("grace")?; b.end_object() }))?;
//! let e = g.link(1, 2, FOLLOWS, &doc(|b| { b.begin_object()?; b.key(b"since")?; b.int(2026)?; b.end_object() }))?;
//!
//! assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), [2]);
//! assert_eq!(g.node(2).and_then(|n| n.get(b"name").and_then(|v| v.as_text())), Some("grace"));
//! assert_eq!(g.edge(e).and_then(|n| n.get(b"since").and_then(|v| v.as_int())), Some(2026));
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # A node is its properties
//!
//! The adjacency plane has no node table. A node is a run of neighbours, so a
//! node with no edges is not in it at all, and asking whether a node exists is
//! not a question it can answer.
//!
//! So the node property store is the node table. [`Graph::put_node`] with an
//! empty object is how an isolated node exists, [`Graph::has_node`] is a lookup
//! in it, and [`Graph::nodes`] is its count. That is one structure doing two
//! jobs rather than two structures that can disagree about which nodes there
//! are, and the empty object it costs is four bytes.
//!
//! # Edge slots
//!
//! An edge's properties are keyed by a slot, and the slot is what the adjacency
//! plane carries beside each neighbour. [`Graph::link`] hands one out and
//! [`Graph::unlink`] gives it back, through a free list, because an edge store
//! that only ever counts up turns a graph that churns into a store that grows
//! forever.
//!
//! A slot is reused only after the properties under it are gone, which is what
//! stops a new edge from inheriting an old edge's fields. That ordering is the
//! whole of why the free list is here rather than in [`crate::Adjacency`]: the
//! plane does not know there is a property store, and a free list that hands
//! out a slot whose document is still there would be worse than no free list.
//!
//! # Parallel edges
//!
//! Linking the same pair twice under the same label leaves two edges, because
//! that is what [`crate::Adjacency::link`] does and what a property graph means
//! by a multigraph. Each gets its own slot and so its own properties, which is
//! the point: two `RATED` edges between the same person and the same film with
//! different scores and different dates is the case, not the corner case.

use yo_common::{Code, Error, Result};
use yo_doc::{Doc, IndexKind, Key};

use crate::{Adjacency, Dir, Props};

/// An empty object, which is what an isolated node's properties are.
///
/// A container header with a count of zero, in the flags a `Builder` writes an
/// empty object with. It is four bytes written out rather than a `Builder` run
/// because it is a constant and building it would allocate once per node that
/// has no properties, and a test below holds it against what a `Builder`
/// actually produces so it cannot drift.
const EMPTY_OBJECT: [u8; 4] = EMPTY_OBJECT_HEAD.to_le_bytes();

/// Tag 7 is a container, bit 4 is sorted and bit 5 carries offsets, which is
/// what every object this version writes has, and the count is in the top three
/// bytes. `yo_format::doc_flags` names the same bits.
const EMPTY_OBJECT_HEAD: u32 = 7 | (1 << 4) | (1 << 5);

/// What to pass as the properties of a node or an edge that has none.
///
/// An edge still needs a document, because the property store is keyed by the
/// edge slot and a slot with nothing under it would be a slot that later reads
/// have to guard against. This is the four bytes that document is, so a caller
/// with nothing to say does not have to run a `Builder` to say it.
pub const NO_PROPS: &[u8] = &EMPTY_OBJECT;

/// A property graph.
#[derive(Debug, Default)]
pub struct Graph {
    adj: Adjacency,
    nodes: Props,
    edges: Props,
    /// The next slot never handed out.
    next: u32,
    /// Slots handed back, whose properties are already gone.
    free: Vec<u32>,
    /// Every label that has an edge, in order. Small, because a schema has
    /// tens of edge types and not thousands, and a linear structure beats a
    /// hash at that size.
    labels: Vec<u32>,
}

impl Graph {
    /// An empty graph that indexes both directions.
    #[must_use]
    pub fn new() -> Graph {
        Graph {
            adj: Adjacency::new(),
            ..Graph::default()
        }
    }

    /// An empty graph that indexes outgoing edges only.
    ///
    /// Half the adjacency memory, and [`Graph::neighbours`] in [`Dir::In`]
    /// answers nothing. [`Graph::remove_node`] cannot find the edges that point
    /// at a node either, so removing a node from one of these leaves the edges
    /// into it in place, and it says so by refusing.
    #[must_use]
    pub fn out_only() -> Graph {
        Graph {
            adj: Adjacency::out_only(),
            ..Graph::default()
        }
    }

    /// Stores `props` under node `id`, replacing whatever was there, and
    /// answers whether the node is new.
    ///
    /// # Errors
    ///
    /// Whatever [`Props::put`] answers: the document is malformed, or a value
    /// an index covers cannot be an index key.
    pub fn put_node(&mut self, id: u64, props: &[u8]) -> Result<bool> {
        self.nodes.put(id, props)
    }

    /// Makes sure node `id` exists, with no properties if it did not.
    ///
    /// # Errors
    ///
    /// Only if a declared index refuses an empty object, which cannot happen,
    /// and is returned rather than swallowed for the same reason every other
    /// write here returns.
    pub fn add_node(&mut self, id: u64) -> Result<bool> {
        if self.nodes.contains(id) {
            return Ok(false);
        }
        self.nodes.put(id, &EMPTY_OBJECT)
    }

    /// Node `id`'s properties.
    #[must_use]
    pub fn node(&self, id: u64) -> Option<Doc<'_>> {
        self.nodes.get(id)
    }

    /// Whether the graph has node `id`.
    #[must_use]
    pub fn has_node(&self, id: u64) -> bool {
        self.nodes.contains(id)
    }

    /// How many nodes the graph has.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// How many edges the graph has, counting a parallel edge as its own.
    #[must_use]
    pub fn edges(&self) -> usize {
        self.adj.edges()
    }

    /// Every label with an edge under it, in order.
    #[must_use]
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// Adds an edge from `src` to `dst` under `label`, with `props` behind it,
    /// and answers the slot the properties are under.
    ///
    /// Both ends are added as nodes if they were not there, because an edge
    /// whose endpoints are not nodes is a dangling reference that every later
    /// read has to guard against.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] when every edge slot is taken, which is four billion live
    /// edges. Otherwise whatever [`Props::put`] answers about `props`.
    pub fn link(&mut self, src: u64, dst: u64, label: u32, props: &[u8]) -> Result<u32> {
        self.add_node(src)?;
        self.add_node(dst)?;
        let slot = self.take_slot()?;
        // The properties go in first. If the document is refused, the slot goes
        // back and the adjacency plane never heard about the edge, so a failed
        // link leaves nothing behind.
        if let Err(e) = self.edges.put(u64::from(slot), props) {
            self.free.push(slot);
            return Err(e);
        }
        self.adj.link(src, dst, label, slot);
        if let Err(at) = self.labels.binary_search(&label) {
            self.labels.insert(at, label);
        }
        Ok(slot)
    }

    /// Removes one edge from `src` to `dst` under `label`, and answers the slot
    /// it was under.
    ///
    /// With parallel edges it removes one of them and which one is whatever the
    /// run's order left, the same as [`crate::Adjacency::unlink`].
    pub fn unlink(&mut self, src: u64, dst: u64, label: u32) -> Option<u32> {
        let slot = self.adj.unlink(src, dst, label)?;
        self.release(slot);
        Some(slot)
    }

    /// Edge `slot`'s properties.
    #[must_use]
    pub fn edge(&self, slot: u32) -> Option<Doc<'_>> {
        self.edges.get(u64::from(slot))
    }

    /// Replaces edge `slot`'s properties.
    ///
    /// # Errors
    ///
    /// [`Code::NotFound`] if no edge is under that slot, so that a caller that
    /// held a slot across a removal is told rather than quietly creating
    /// properties for an edge that is gone. Otherwise whatever [`Props::put`]
    /// answers.
    pub fn put_edge(&mut self, slot: u32, props: &[u8]) -> Result<()> {
        if !self.edges.contains(u64::from(slot)) {
            return Err(Error::new(Code::NotFound, "no edge is under that slot")
                .with_detail(format!("slot={slot}")));
        }
        self.edges.put(u64::from(slot), props)?;
        Ok(())
    }

    /// Takes node `id` out, along with every edge at either end of it.
    ///
    /// Answers whether the node was there.
    ///
    /// # Errors
    ///
    /// [`Code::Unsupported`] on a graph built by [`Graph::out_only`] when the
    /// node has any outgoing edge, because the edges pointing at it cannot be
    /// found and removing it would leave them pointing at nothing. A node with
    /// no edges at all is removed either way.
    pub fn remove_node(&mut self, id: u64) -> Result<bool> {
        if !self.nodes.contains(id) {
            return Ok(false);
        }
        if !self.adj.indexes_incoming() {
            let any = self
                .labels
                .iter()
                .any(|&l| !self.adj.neighbours(id, l, Dir::Out).is_empty());
            if any {
                return Err(Error::new(
                    Code::Unsupported,
                    "this graph does not index incoming edges, so a node with edges cannot be removed",
                )
                .with_detail(format!("node={id}")));
            }
        }
        // The labels are copied because unlinking borrows the graph, and there
        // are tens of them.
        let labels = self.labels.clone();
        for label in labels {
            // Out first, then in. Each end is snapshotted before anything is
            // removed, because unlinking moves the last entry of a run into the
            // hole it made and a walk over a run being edited would skip
            // whatever moved.
            let out: Vec<u64> = self.adj.neighbours(id, label, Dir::Out).to_vec();
            for dst in out {
                if let Some(slot) = self.adj.unlink(id, dst, label) {
                    self.release(slot);
                }
            }
            let into: Vec<u64> = self.adj.neighbours(id, label, Dir::In).to_vec();
            for src in into {
                if let Some(slot) = self.adj.unlink(src, id, label) {
                    self.release(slot);
                }
            }
        }
        Ok(self.nodes.remove(id))
    }

    /// The neighbours of `node` under `label` in `dir`, in one contiguous run.
    #[must_use]
    pub fn neighbours(&self, node: u64, label: u32, dir: Dir) -> &[u64] {
        self.adj.neighbours(node, label, dir)
    }

    /// The edge slots beside those neighbours, in the same order.
    #[must_use]
    pub fn edge_slots(&self, node: u64, label: u32, dir: Dir) -> &[u32] {
        self.adj.edge_slots(node, label, dir)
    }

    /// The neighbours of `node` under `label` in `dir`, each with the slot of
    /// the edge that got there.
    pub fn hop(&self, node: u64, label: u32, dir: Dir) -> impl Iterator<Item = (u64, u32)> {
        self.adj
            .neighbours(node, label, dir)
            .iter()
            .copied()
            .zip(self.adj.edge_slots(node, label, dir).iter().copied())
    }

    /// How many edges `node` has under `label` in `dir`.
    #[must_use]
    pub fn degree(&self, node: u64, label: u32, dir: Dir) -> usize {
        self.adj.degree(node, label, dir)
    }

    /// Warms the run `node` is about to be walked through.
    pub fn prefetch(&self, node: u64, label: u32, dir: Dir) {
        self.adj.prefetch(node, label, dir);
    }

    /// Declares an index over a path into node properties, and backfills it.
    ///
    /// # Errors
    ///
    /// The same as [`Props::create_index`].
    pub fn index_nodes(&mut self, path: &str, kind: IndexKind) -> Result<()> {
        self.nodes.create_index(path, kind)
    }

    /// Declares an index over a path into edge properties, and backfills it.
    ///
    /// # Errors
    ///
    /// The same as [`Props::create_index`].
    pub fn index_edges(&mut self, path: &str, kind: IndexKind) -> Result<()> {
        self.edges.create_index(path, kind)
    }

    /// Calls `f` for every node whose properties have `key` under `path`.
    ///
    /// # Errors
    ///
    /// The same as [`Props::find`]: the path has to be indexed.
    pub fn find_nodes(&self, path: &str, key: &Key, f: impl FnMut(u64, Doc<'_>)) -> Result<usize> {
        self.nodes.find(path, key, f)
    }

    /// How many nodes have `key` under `path`, without reading any of them.
    ///
    /// # Errors
    ///
    /// The same as [`Graph::find_nodes`].
    pub fn count_nodes(&self, path: &str, key: &Key) -> Result<usize> {
        self.nodes.count(path, key)
    }

    /// How many edges have `key` under `path`, without reading any of them.
    ///
    /// # Errors
    ///
    /// The same as [`Graph::find_edges`].
    pub fn count_edges(&self, path: &str, key: &Key) -> Result<usize> {
        self.edges.count(path, key)
    }

    /// Calls `f` for every edge slot whose properties have `key` under `path`.
    ///
    /// # Errors
    ///
    /// The same as [`Props::find`].
    pub fn find_edges(
        &self,
        path: &str,
        key: &Key,
        mut f: impl FnMut(u32, Doc<'_>),
    ) -> Result<usize> {
        self.edges.find(path, key, |slot, doc| {
            // A slot is a u32 that was widened on the way in, so this cannot
            // truncate, and a store handed something else is not this store.
            if let Ok(slot) = u32::try_from(slot) {
                f(slot, doc);
            }
        })
    }

    /// The node properties, for the document operations that are the document
    /// model's rather than the graph's.
    #[must_use]
    pub fn node_props(&self) -> &Props {
        &self.nodes
    }

    /// The edge properties, likewise.
    #[must_use]
    pub fn edge_props(&self) -> &Props {
        &self.edges
    }

    /// The adjacency plane underneath.
    #[must_use]
    pub fn adjacency(&self) -> &Adjacency {
        &self.adj
    }

    /// What the whole graph weighs.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.adj.bytes()
            + self.nodes.memory_bytes()
            + self.edges.memory_bytes()
            + self.free.capacity() * size_of::<u32>()
            + self.labels.capacity() * size_of::<u32>()
    }

    /// A slot to put an edge's properties under.
    fn take_slot(&mut self) -> Result<u32> {
        if let Some(slot) = self.free.pop() {
            return Ok(slot);
        }
        if self.next == u32::MAX {
            return Err(Error::new(Code::Full, "this graph has no edge slots left"));
        }
        let slot = self.next;
        self.next += 1;
        Ok(slot)
    }

    /// Gives a slot back, after its properties are gone.
    fn release(&mut self, slot: u32) {
        self.edges.remove(u64::from(slot));
        self.free.push(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_doc::{Builder, Value};

    const FOLLOWS: u32 = 1;
    const BLOCKS: u32 = 2;

    fn doc(f: impl FnOnce(&mut Builder) -> Result<()>) -> Vec<u8> {
        let mut b = Builder::new();
        f(&mut b).expect("built");
        b.finish().expect("finished").to_vec()
    }

    fn named(name: &str) -> Vec<u8> {
        doc(|b| {
            b.begin_object()?;
            b.key(b"name")?;
            b.text(name)?;
            b.end_object()
        })
    }

    fn since(year: i64) -> Vec<u8> {
        doc(|b| {
            b.begin_object()?;
            b.key(b"since")?;
            b.int(year)?;
            b.end_object()
        })
    }

    #[test]
    fn the_empty_object_constant_is_an_empty_object() {
        // It is written as four bytes rather than built, so this is the check
        // that those four bytes are the ones a builder would have produced.
        let built = doc(|b| {
            b.begin_object()?;
            b.end_object()
        });
        assert_eq!(&EMPTY_OBJECT[..], &built[..]);
        let v = Value::new(&EMPTY_OBJECT).expect("readable");
        assert!(v.validate());
        assert!(v.is_empty());
    }

    #[test]
    fn a_node_exists_once_it_has_properties() {
        let mut g = Graph::new();
        assert!(!g.has_node(1));
        assert!(g.add_node(1).unwrap());
        assert!(!g.add_node(1).unwrap(), "adding twice is not two nodes");
        assert!(g.has_node(1));
        assert_eq!(g.nodes(), 1);
        assert_eq!(g.edges(), 0);
    }

    #[test]
    fn linking_creates_the_endpoints() {
        let mut g = Graph::new();
        let e = g.link(1, 2, FOLLOWS, &since(2026)).unwrap();
        assert!(g.has_node(1) && g.has_node(2));
        assert_eq!(g.nodes(), 2);
        assert_eq!(g.edges(), 1);
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), [2]);
        assert_eq!(g.neighbours(2, FOLLOWS, Dir::In), [1]);
        assert_eq!(
            g.edge(e)
                .and_then(|d| d.get(b"since"))
                .and_then(|v| v.as_int()),
            Some(2026)
        );
        assert_eq!(g.labels(), [FOLLOWS]);
    }

    #[test]
    fn a_hop_gives_the_neighbour_and_the_edge_together() {
        let mut g = Graph::new();
        g.link(1, 2, FOLLOWS, &since(2024)).unwrap();
        g.link(1, 3, FOLLOWS, &since(2025)).unwrap();
        let mut seen: Vec<(u64, i64)> = g
            .hop(1, FOLLOWS, Dir::Out)
            .map(|(n, slot)| {
                let year = g
                    .edge(slot)
                    .and_then(|d| d.get(b"since"))
                    .and_then(|v| v.as_int())
                    .expect("an edge has its year");
                (n, year)
            })
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![(2, 2024), (3, 2025)]);
    }

    #[test]
    fn parallel_edges_each_keep_their_own_properties() {
        let mut g = Graph::new();
        let a = g.link(1, 2, FOLLOWS, &since(2024)).unwrap();
        let b = g.link(1, 2, FOLLOWS, &since(2026)).unwrap();
        assert_ne!(a, b);
        assert_eq!(g.edges(), 2);
        assert_eq!(g.degree(1, FOLLOWS, Dir::Out), 2);
        assert_eq!(
            g.edge(a)
                .and_then(|d| d.get(b"since"))
                .and_then(|v| v.as_int()),
            Some(2024)
        );
        assert_eq!(
            g.edge(b)
                .and_then(|d| d.get(b"since"))
                .and_then(|v| v.as_int()),
            Some(2026)
        );
    }

    #[test]
    fn a_freed_slot_is_reused_without_its_old_properties() {
        let mut g = Graph::new();
        let a = g.link(1, 2, FOLLOWS, &since(2024)).unwrap();
        assert_eq!(g.unlink(1, 2, FOLLOWS), Some(a));
        assert!(g.edge(a).is_none(), "an unlinked edge keeps nothing");

        // The same slot comes back, and the point of the test is that what is
        // under it is the new edge's document and not the old one's.
        let b = g.link(3, 4, FOLLOWS, &since(2026)).unwrap();
        assert_eq!(a, b, "a freed slot is handed out again");
        assert_eq!(
            g.edge(b)
                .and_then(|d| d.get(b"since"))
                .and_then(|v| v.as_int()),
            Some(2026)
        );
    }

    #[test]
    fn removing_a_node_takes_its_edges_at_both_ends() {
        let mut g = Graph::new();
        g.link(1, 2, FOLLOWS, &since(2024)).unwrap();
        g.link(3, 2, FOLLOWS, &since(2025)).unwrap();
        g.link(2, 4, BLOCKS, &since(2026)).unwrap();
        g.link(1, 3, FOLLOWS, &since(2023)).unwrap();
        assert_eq!(g.edges(), 4);

        assert!(g.remove_node(2).unwrap());
        assert!(!g.has_node(2));
        // The three edges that touched node 2 are gone and the one that did not
        // is still there.
        assert_eq!(g.edges(), 1);
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), [3]);
        assert!(g.neighbours(3, FOLLOWS, Dir::Out).is_empty());
        assert!(g.neighbours(4, BLOCKS, Dir::In).is_empty());
        // Their properties went with them, and the surviving edge kept its
        // own, which is the half of this that a store that cleared everything
        // would also pass.
        assert_eq!(g.edge_props().len(), 1);
        let left = g.edge_slots(1, FOLLOWS, Dir::Out)[0];
        assert_eq!(
            g.edge(left)
                .and_then(|d| d.get(b"since"))
                .and_then(|v| v.as_int()),
            Some(2023)
        );
    }

    #[test]
    fn removing_a_node_with_parallel_edges_takes_all_of_them() {
        // The case a snapshot of the run guards against: unlinking moves the
        // last entry into the hole, so a walk over a live run would skip one.
        let mut g = Graph::new();
        for year in 2020..2030 {
            g.link(1, 2, FOLLOWS, &since(year)).unwrap();
        }
        assert_eq!(g.edges(), 10);
        assert!(g.remove_node(2).unwrap());
        assert_eq!(g.edges(), 0);
        assert!(g.neighbours(1, FOLLOWS, Dir::Out).is_empty());
        assert!(g.edge_props().is_empty());
    }

    #[test]
    fn removing_a_node_that_is_not_there_says_so() {
        let mut g = Graph::new();
        g.add_node(1).unwrap();
        assert!(!g.remove_node(2).unwrap());
        assert!(g.remove_node(1).unwrap());
        assert_eq!(g.nodes(), 0);
    }

    #[test]
    fn an_out_only_graph_refuses_to_remove_a_linked_node() {
        let mut g = Graph::out_only();
        g.link(1, 2, FOLLOWS, &since(2026)).unwrap();
        // Node 1 has an outgoing edge and no way to find what points at it.
        assert!(g.remove_node(1).is_err());
        // Node 2 has no outgoing edge, so there is nothing to leave dangling
        // at this end, and it goes.
        assert!(g.remove_node(2).unwrap());
    }

    #[test]
    fn an_edge_slot_that_is_gone_refuses_a_write() {
        let mut g = Graph::new();
        let e = g.link(1, 2, FOLLOWS, &since(2024)).unwrap();
        assert!(g.put_edge(e, &since(2025)).is_ok());
        g.unlink(1, 2, FOLLOWS);
        assert!(
            g.put_edge(e, &since(2026)).is_err(),
            "a slot held across a removal is not an edge"
        );
    }

    #[test]
    fn nodes_are_found_by_an_indexed_property() {
        let mut g = Graph::new();
        g.index_nodes("$.name", IndexKind::Equality).unwrap();
        g.put_node(1, &named("ada")).unwrap();
        g.put_node(2, &named("grace")).unwrap();
        g.put_node(3, &named("ada")).unwrap();

        let mut found = Vec::new();
        let n = g
            .find_nodes("$.name", &Key::text("ada"), |id, _| found.push(id))
            .unwrap();
        assert_eq!(n, 2);
        found.sort_unstable();
        assert_eq!(found, vec![1, 3]);
    }

    #[test]
    fn edges_are_found_by_an_indexed_property() {
        let mut g = Graph::new();
        g.index_edges("$.since", IndexKind::Equality).unwrap();
        let a = g.link(1, 2, FOLLOWS, &since(2026)).unwrap();
        g.link(1, 3, FOLLOWS, &since(2024)).unwrap();
        let c = g.link(2, 3, BLOCKS, &since(2026)).unwrap();

        let mut found = Vec::new();
        let n = g
            .find_edges("$.since", &Key::int(2026), |slot, _| found.push(slot))
            .unwrap();
        assert_eq!(n, 2);
        found.sort_unstable();
        let mut want = vec![a, c];
        want.sort_unstable();
        assert_eq!(found, want);
    }

    #[test]
    fn labels_are_the_ones_with_edges_in_order() {
        let mut g = Graph::new();
        g.link(1, 2, BLOCKS, &since(2026)).unwrap();
        g.link(1, 3, FOLLOWS, &since(2026)).unwrap();
        g.link(1, 4, FOLLOWS, &since(2026)).unwrap();
        assert_eq!(g.labels(), [FOLLOWS, BLOCKS]);
    }

    #[test]
    fn a_graph_that_churns_does_not_grow_forever() {
        // The failure the free list is for. Ten thousand edges added and
        // removed one at a time, with only one ever live, so a slot allocator
        // that only counted up would leave an edge store ten thousand
        // documents deep holding one edge.
        let mut g = Graph::new();
        for year in 0..10_000i64 {
            g.link(1, 2, FOLLOWS, &since(year)).unwrap();
            assert!(g.unlink(1, 2, FOLLOWS).is_some());
        }
        g.link(1, 2, FOLLOWS, &since(2026)).unwrap();
        assert_eq!(g.edges(), 1);
        assert_eq!(g.edge_props().len(), 1);
    }

    #[test]
    fn the_names_of_edge_properties_are_stored_once() {
        let mut g = Graph::new();
        for dst in 2..1002u64 {
            g.link(1, dst, FOLLOWS, &since(2026)).unwrap();
        }
        assert_eq!(g.edges(), 1000);
        // One field name for a thousand edges, which is the whole reason edge
        // properties are documents rather than their own store.
        assert_eq!(g.edge_props().keys().len(), 1);
    }
}
