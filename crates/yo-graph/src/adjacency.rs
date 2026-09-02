//! The adjacency plane in its hot form: a run of neighbours per (node, label,
//! direction), appended to and deleted from in place.
//!
//! A graph without a query language is an adjacency structure with good
//! ergonomics, so this is what everything else in the graph model stands on.
//! `11` section 2 gives adjacency two forms and this is the mutable one. The
//! other is zu's node group CSR, which reaches 8 bits an edge because it never
//! changes; this one is 12 bytes an edge because every operation on it has to
//! be O(1).
//!
//! # The shape
//!
//! A run is the neighbours of one node under one label in one direction, and it
//! is contiguous. That is the whole performance argument: a one hop is a probe
//! for the run header and then a sequential read, and the read is over `u64`
//! node ids with nothing else interleaved, so eight neighbours arrive per cache
//! line.
//!
//! The edge slots live in a second array indexed the same way. Keeping them
//! apart rather than storing `(neighbour, edge)` pairs costs nothing and saves
//! a third of the memory traffic on the common walk, because a traversal that
//! only wants to know where it can go next never reads an edge slot at all.
//! Interleaving them would also make every neighbour an unaligned load out of a
//! 12 byte stride.
//!
//! # Growing and shrinking
//!
//! Runs are cut from two shared arenas rather than allocated one by one,
//! because a graph is mostly nodes with a handful of edges and a `Vec` header
//! per node would cost more than the edges do. A run's capacity comes off a
//! fixed ladder: doubling while it is small, then a quarter more each step, so
//! the slack a hub carries is bounded by 25 per cent instead of by 100. Growing
//! copies the run into the next size up and gives the old block to a free list,
//! so the space is reused rather than lost.
//!
//! Deleting is swap with last and a decrement, which is how `08` section 4
//! deletes from a dense member vector and it is the same reason: an O(1) delete
//! is worth giving up the order of a run that has no order to give up. A run
//! that falls to half of its capacity is moved down to the smallest size that
//! fits, which leaves 2x of hysteresis so a run sitting on a boundary does not
//! copy itself every time it gains and loses one edge.
//!
//! # What it costs
//!
//! Twelve bytes an edge is the payload and it is not the whole bill. There is
//! one 32 byte run header per (node, label, direction) that has ever been
//! linked, and there is the capacity slack. On a graph shaped like LiveJournal,
//! most nodes with a few edges and a thin tail of hubs, the measured numbers
//! are 18.1 bytes an edge as it is built and 15.2 after a sweep, which is 12.0
//! of payload and 3.2 of run headers against an average degree of 13. The test
//! at the bottom of this file is where those come from.
//!
//! The cold form is where 8 bits an edge comes from, and the ladder in `03` is
//! what lets both be true at once. Against what this replaces it is already the
//! cheap end: a pointer chased adjacency list is 16 bytes for the pair before
//! the per node allocation header, and Neo4j's relationship store is 34.
//!
//! # What this does not do
//!
//! It does not look for a duplicate before it links. That check is linear in
//! the degree, and the degree is exactly the thing that can be a hub with
//! twenty thousand edges on it, so paying it on every insert would trade the
//! write path away to enforce something the layer above can enforce with one
//! probe. An edge table keyed by (source, destination, label) is where upsert
//! semantics belong, and it is what `G.EADD` will sit on.
//!
//! It also does not delete a node, because finding every run a node has means
//! knowing every label it was ever linked under, and that is the node table's
//! job rather than this one's.

use yo_common::prefetch;

/// Which end of an edge a run is stored under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir {
    /// Edges leaving the node.
    Out,
    /// Edges arriving at the node.
    In,
}

/// How many sizes the capacity ladder has.
///
/// Eighty eight of them reach a run of four billion edges, which is the most a
/// `u32` offset can address anyway, and the rest are there so the arithmetic at
/// the top never has to think about the end.
const CLASSES: usize = 96;

/// The capacity ladder, in edges.
const LADDER: [u32; CLASSES] = ladder();

const LIVE: u8 = 1;
const INCOMING: u8 = 2;

/// Doubling to 16 and then a quarter more, rounded up to four so the steps just
/// past the change of policy are still steps.
///
/// Sixteen is where the doubling stops and it is measured rather than picked. A
/// run of 30 edges in a run sized 32 wastes nothing worth counting, but a run of
/// 33 in a run sized 64 wastes half of itself, and the degrees between about 5
/// and 50 are where most of a real graph's edges live. Doubling all the way to
/// 64 costs 1.47 bytes of capacity per byte of edge across that band; stopping
/// at 16 costs 1.16, and the whole graph goes from 1.23 to 1.13. What it buys
/// back is two more copies per edge over the life of a run, which is 24 bytes of
/// memcpy against 12 bytes of edge, and that is not a trade anybody notices.
const fn ladder() -> [u32; CLASSES] {
    let mut out = [0u32; CLASSES];
    let mut cap: u64 = 1;
    let mut i = 0;
    while i < CLASSES {
        out[i] = if cap > u32::MAX as u64 {
            u32::MAX
        } else {
            cap as u32
        };
        cap = if cap < 16 {
            cap * 2
        } else {
            (cap + cap / 4 + 3) & !3
        };
        i += 1;
    }
    out
}

/// The header of one run, and the only thing the table stores.
///
/// Thirty two bytes, so two of them fit a cache line and neither straddles it,
/// which is what makes a probe that misses cost one load and nothing else.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    node: u64,
    at: u32,
    len: u32,
    cap: u32,
    label: u32,
    flags: u8,
}

/// The neighbours of every node, under every label, in both directions.
///
/// ```
/// use yo_graph::{Adjacency, Dir};
///
/// const FOLLOWS: u32 = 1;
///
/// let mut g = Adjacency::new();
/// g.link(1, 2, FOLLOWS, 100);
/// g.link(1, 3, FOLLOWS, 101);
///
/// assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), &[2, 3]);
/// assert_eq!(g.neighbours(2, FOLLOWS, Dir::In), &[1]);
/// assert_eq!(g.degree(1, FOLLOWS, Dir::Out), 2);
/// ```
#[derive(Debug)]
pub struct Adjacency {
    slots: Vec<Slot>,
    live: usize,
    filled: usize,
    edges: usize,
    entries: usize,
    neighbour: Vec<u64>,
    edge: Vec<u32>,
    free: Vec<Vec<u32>>,
    both: bool,
}

impl Default for Adjacency {
    fn default() -> Adjacency {
        Adjacency::new()
    }
}

impl Adjacency {
    /// An empty plane that indexes both directions, so `In` answers as well as
    /// `Out` does.
    #[must_use]
    pub fn new() -> Adjacency {
        Adjacency::build(true)
    }

    /// An empty plane that indexes outgoing edges only.
    ///
    /// This halves the memory and halves the work an insert does, and it is the
    /// right choice whenever nothing asks the graph who points at a node.
    /// `neighbours` under [`Dir::In`] then answers nothing at all, which is why
    /// it is a decision at construction rather than a flag on a call: a walk
    /// that silently found no incoming edges because of how the plane was built
    /// would look exactly like a node that has none.
    #[must_use]
    pub fn out_only() -> Adjacency {
        Adjacency::build(false)
    }

    fn build(both: bool) -> Adjacency {
        Adjacency {
            slots: Vec::new(),
            live: 0,
            filled: 0,
            edges: 0,
            entries: 0,
            neighbour: Vec::new(),
            edge: Vec::new(),
            free: Vec::new(),
            both,
        }
    }

    /// Whether incoming edges are indexed.
    #[must_use]
    pub fn indexes_incoming(&self) -> bool {
        self.both
    }

    /// How many edges have been linked and not unlinked.
    #[must_use]
    pub fn edges(&self) -> usize {
        self.edges
    }

    /// Whether any edge is linked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges == 0
    }

    /// How many runs hold at least one edge.
    #[must_use]
    pub fn runs(&self) -> usize {
        self.filled
    }

    /// Add an edge from `src` to `dst` under `label`, carrying `edge` as the
    /// slot of the edge record.
    ///
    /// This appends. It does not look for an edge that is already there, for
    /// the reason in the module docs, so linking the same pair twice leaves two
    /// entries and unlinking it once leaves one.
    pub fn link(&mut self, src: u64, dst: u64, label: u32, edge: u32) {
        let s = self.run_for(src, label, 0);
        self.push(s, dst, edge);
        if self.both {
            let d = self.run_for(dst, label, INCOMING);
            self.push(d, src, edge);
        }
        self.edges += 1;
    }

    /// Remove one edge from `src` to `dst` under `label`, and answer with the
    /// edge slot it was carrying.
    ///
    /// Costs a scan of the run at each end, because finding which position an
    /// edge sits at is the one thing a plane keyed by node rather than by edge
    /// cannot do in a step. A caller that already knows the position wants
    /// [`Adjacency::unlink_at`].
    pub fn unlink(&mut self, src: u64, dst: u64, label: u32) -> Option<u32> {
        let s = self.find(src, label, 0)?;
        let i = self.position(s, dst)?;
        let edge = self.take(s, i).1;
        if self.both
            && let Some(d) = self.find(dst, label, INCOMING)
            && let Some(j) = self.position(d, src)
        {
            self.take(d, j);
        }
        self.edges -= 1;
        Some(edge)
    }

    /// Remove the edge at position `i` of one run, and answer with the
    /// neighbour and the edge slot that were there.
    ///
    /// This is the O(1) primitive and it touches one end only, so the other end
    /// still holds its half of the edge. It is for a caller that tracks
    /// positions itself and will do both. Whatever used to be last has moved
    /// into `i`.
    pub fn unlink_at(&mut self, node: u64, label: u32, dir: Dir, i: usize) -> Option<(u64, u32)> {
        let s = self.find(node, label, incoming(dir))?;
        if i >= self.slots[s].len as usize {
            return None;
        }
        Some(self.take(s, i))
    }

    /// The neighbours of `node` under `label` in `dir`, in one contiguous run.
    ///
    /// One probe and then a sequential read. The order is whatever inserting
    /// and deleting left behind, because a delete moves the last entry into the
    /// hole it made.
    #[must_use]
    pub fn neighbours(&self, node: u64, label: u32, dir: Dir) -> &[u64] {
        match self.find(node, label, incoming(dir)) {
            Some(s) => {
                let (at, len) = (self.slots[s].at as usize, self.slots[s].len as usize);
                &self.neighbour[at..at + len]
            }
            None => &[],
        }
    }

    /// The edge slots of `node` under `label` in `dir`, lined up one for one
    /// with [`Adjacency::neighbours`].
    #[must_use]
    pub fn edge_slots(&self, node: u64, label: u32, dir: Dir) -> &[u32] {
        match self.find(node, label, incoming(dir)) {
            Some(s) => {
                let (at, len) = (self.slots[s].at as usize, self.slots[s].len as usize);
                &self.edge[at..at + len]
            }
            None => &[],
        }
    }

    /// How many edges `node` has under `label` in `dir`.
    #[must_use]
    pub fn degree(&self, node: u64, label: u32, dir: Dir) -> usize {
        match self.find(node, label, incoming(dir)) {
            Some(s) => self.slots[s].len as usize,
            None => 0,
        }
    }

    /// Ask the cache for the slot a run's header will be found in.
    ///
    /// A multi hop walk knows its whole next frontier before it reads any of
    /// it, so it can issue these across the frontier and then come back and
    /// read. That is the same two walk shape `04` section 3 drains a command
    /// batch with, and it is what the two hop budget in G14 is actually
    /// spending: the probes are dependent loads, and the only way to make them
    /// cheap is to stop them being serial.
    pub fn prefetch(&self, node: u64, label: u32, dir: Dir) {
        if self.slots.is_empty() {
            return;
        }
        let i = bucket(hash(node, label, incoming(dir)), self.slots.len());
        prefetch(&self.slots[i]);
    }

    /// Resident bytes, counting the table, both arenas, and everything the free
    /// lists are still holding.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.slots.capacity() * size_of::<Slot>()
            + self.neighbour.capacity() * size_of::<u64>()
            + self.edge.capacity() * size_of::<u32>()
            + self.free.capacity() * size_of::<Vec<u32>>()
            + self
                .free
                .iter()
                .map(|f| f.capacity() * size_of::<u32>())
                .sum::<usize>()
    }

    /// Rebuild the table and the arenas with nothing spare in them.
    ///
    /// A run that emptied leaves its header behind, a run that shrank leaves
    /// slack, and a free list holds blocks nothing has asked for again. None of
    /// that is worth chasing on the write path, so this is the sweep that
    /// reclaims it, and it is the natural thing to run before a settled part of
    /// the graph is promoted into the cold form. Every run comes out sized to
    /// exactly what it holds and laid out one after another, which is also the
    /// order the promotion wants to read them in.
    pub fn compact(&mut self) {
        let mut keep: Vec<Slot> = self
            .slots
            .iter()
            .copied()
            .filter(|s| s.flags & LIVE != 0 && s.len > 0)
            .collect();
        let mut neighbour = Vec::with_capacity(self.entries);
        let mut edge = Vec::with_capacity(self.entries);
        for slot in &mut keep {
            let (at, len) = (slot.at as usize, slot.len as usize);
            let to = neighbour.len() as u32;
            neighbour.extend_from_slice(&self.neighbour[at..at + len]);
            edge.extend_from_slice(&self.edge[at..at + len]);
            slot.at = to;
            slot.cap = slot.len;
        }
        self.neighbour = neighbour;
        self.edge = edge;
        self.free = Vec::new();
        self.live = keep.len();
        self.filled = keep.len();
        // Sized off the live count rather than rounded up to a power of two,
        // which is the whole reason the table is indexed by a multiply instead
        // of a mask. A graph with two hundred thousand runs would otherwise get
        // half a million slots and carry three bytes an edge it never uses.
        self.slots = vec![Slot::default(); (keep.len() * 4 / 3).max(16)];
        for slot in keep {
            let i = self.vacancy(slot.node, slot.label, slot.flags & INCOMING);
            self.slots[i] = slot;
        }
    }

    fn find(&self, node: u64, label: u32, incoming: u8) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let n = self.slots.len();
        let mut i = bucket(hash(node, label, incoming), n);
        loop {
            let s = &self.slots[i];
            if s.flags & LIVE == 0 {
                return None;
            }
            if s.node == node && s.label == label && s.flags & INCOMING == incoming {
                return Some(i);
            }
            i += 1;
            if i == n {
                i = 0;
            }
        }
    }

    fn position(&self, s: usize, node: u64) -> Option<usize> {
        let (at, len) = (self.slots[s].at as usize, self.slots[s].len as usize);
        self.neighbour[at..at + len].iter().position(|n| *n == node)
    }

    /// The slot for a run, made if it was not there.
    fn run_for(&mut self, node: u64, label: u32, incoming: u8) -> usize {
        if (self.live + 1) * 4 > self.slots.len() * 3 {
            self.regrow();
        }
        let n = self.slots.len();
        let mut i = bucket(hash(node, label, incoming), n);
        loop {
            let s = &self.slots[i];
            if s.flags & LIVE == 0 {
                self.slots[i] = Slot {
                    node,
                    at: 0,
                    len: 0,
                    cap: 0,
                    label,
                    flags: LIVE | incoming,
                };
                self.live += 1;
                return i;
            }
            if s.node == node && s.label == label && s.flags & INCOMING == incoming {
                return i;
            }
            i += 1;
            if i == n {
                i = 0;
            }
        }
    }

    /// Where a key that is known not to be there belongs.
    fn vacancy(&self, node: u64, label: u32, incoming: u8) -> usize {
        let n = self.slots.len();
        let mut i = bucket(hash(node, label, incoming), n);
        while self.slots[i].flags & LIVE != 0 {
            i += 1;
            if i == n {
                i = 0;
            }
        }
        i
    }

    fn regrow(&mut self) {
        // A quarter more rather than double, for the same reason the table is
        // not a power of two. Growth by a factor g leaves the load factor
        // wandering between 0.75 and 0.75 over g, so doubling means half the
        // table is empty for most of its life. A quarter holds it between 0.6
        // and 0.75, and the price is four rehashes per run over the whole
        // build rather than two.
        let want = (self.slots.len() + self.slots.len() / 4).max(16);
        let old = core::mem::replace(&mut self.slots, vec![Slot::default(); want]);
        for slot in old {
            if slot.flags & LIVE != 0 {
                let i = self.vacancy(slot.node, slot.label, slot.flags & INCOMING);
                self.slots[i] = slot;
            }
        }
    }

    fn push(&mut self, s: usize, node: u64, edge: u32) {
        let Slot {
            mut at,
            len,
            mut cap,
            ..
        } = self.slots[s];
        if len == cap {
            let want = LADDER[ceil_class(cap + 1)];
            let to = self.alloc(want);
            if len > 0 {
                self.copy_run(at, to, len as usize);
                self.release(at, cap);
            }
            at = to;
            cap = want;
        }
        let i = at as usize + len as usize;
        self.neighbour[i] = node;
        self.edge[i] = edge;
        self.slots[s].at = at;
        self.slots[s].cap = cap;
        self.slots[s].len = len + 1;
        self.entries += 1;
        if len == 0 {
            self.filled += 1;
        }
    }

    /// Swap with last, decrement, and give back capacity the run has outgrown.
    fn take(&mut self, s: usize, i: usize) -> (u64, u32) {
        let Slot { at, len, cap, .. } = self.slots[s];
        let (at, last) = (at as usize, at as usize + len as usize - 1);
        let gone = (self.neighbour[at + i], self.edge[at + i]);
        self.neighbour[at + i] = self.neighbour[last];
        self.edge[at + i] = self.edge[last];
        self.slots[s].len = len - 1;
        self.entries -= 1;
        if len == 1 {
            self.filled -= 1;
        }
        self.shrink(s, cap);
        gone
    }

    fn shrink(&mut self, s: usize, cap: u32) {
        let len = self.slots[s].len;
        if len == 0 {
            self.release(self.slots[s].at, cap);
            self.slots[s].at = 0;
            self.slots[s].cap = 0;
            return;
        }
        // Half of the capacity is the trigger and the smallest size that fits
        // is the destination, so a run has to lose half of itself before it is
        // moved and it will not be moved again until it has doubled.
        if len * 2 > cap {
            return;
        }
        let want = LADDER[ceil_class(len)];
        if want >= cap {
            return;
        }
        let at = self.slots[s].at;
        let to = self.alloc(want);
        self.copy_run(at, to, len as usize);
        self.release(at, cap);
        self.slots[s].at = to;
        self.slots[s].cap = want;
    }

    fn copy_run(&mut self, from: u32, to: u32, len: usize) {
        let (from, to) = (from as usize, to as usize);
        self.neighbour.copy_within(from..from + len, to);
        self.edge.copy_within(from..from + len, to);
    }

    fn alloc(&mut self, cap: u32) -> u32 {
        let class = ceil_class(cap);
        if let Some(list) = self.free.get_mut(class)
            && let Some(at) = list.pop()
        {
            return at;
        }
        let cap = cap as usize;
        let at = self.neighbour.len();
        assert!(at + cap <= u32::MAX as usize, "the adjacency arena is full");
        // An eighth more when it has to grow, not double. These are the
        // biggest things here by a long way, so a doubling that lands one edge
        // past the last one leaves half the plane allocated and never touched.
        // Nothing is rehashed on the way, it is one copy, so the growth can be
        // much finer here than the table's.
        if at + cap > self.neighbour.capacity() {
            let cur = self.neighbour.capacity();
            let want = (cur + cur / 8).max(at + cap).max(64);
            self.neighbour.reserve_exact(want - at);
            self.edge.reserve_exact(want - at);
        }
        self.neighbour.resize(at + cap, 0);
        self.edge.resize(at + cap, 0);
        at as u32
    }

    /// A block goes back under the largest size that certainly fits inside it,
    /// so a block that came out of a compaction at some exact length is still
    /// reusable and is never handed out as more room than it has.
    fn release(&mut self, at: u32, cap: u32) {
        let class = floor_class(cap);
        while self.free.len() <= class {
            self.free.push(Vec::new());
        }
        self.free[class].push(at);
    }
}

/// The smallest ladder size that holds `n` edges.
///
/// A linear walk, because the answer is in the first few entries for nearly
/// every node in a real graph and a binary search over 96 entries would be
/// slower for the case that matters.
fn ceil_class(n: u32) -> usize {
    LADDER.iter().position(|c| *c >= n).unwrap_or(CLASSES - 1)
}

/// The largest ladder size that fits inside `n` edges.
fn floor_class(n: u32) -> usize {
    let at = ceil_class(n);
    if LADDER[at] > n {
        at.saturating_sub(1)
    } else {
        at
    }
}

fn incoming(dir: Dir) -> u8 {
    match dir {
        Dir::Out => 0,
        Dir::In => INCOMING,
    }
}

/// Lemire's reduction: the top bits of the hash scaled onto `n`, so the table
/// can be any size at all rather than a power of two. That matters more here
/// than the one multiply costs, because rounding a run count up to a power of
/// two is up to twice the table for nothing.
#[inline]
fn bucket(h: u64, n: usize) -> usize {
    ((u128::from(h) * n as u128) >> 64) as usize
}

/// Node ids are usually dense small integers and there are only ever a handful
/// of distinct labels, so the label and the direction are spread across the
/// whole word before the finaliser rather than after it. Without that, every
/// label of one node lands in a run of neighbouring slots and a probe for one
/// walks over the others.
#[inline]
fn hash(node: u64, label: u32, incoming: u8) -> u64 {
    let tag = (u64::from(label) << 1) | u64::from(incoming >> 1);
    let mut x = node ^ tag.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 29;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    const FOLLOWS: u32 = 1;
    const WORKS_AT: u32 = 2;

    fn sorted(v: &[u64]) -> Vec<u64> {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn a_slot_is_one_half_of_a_cache_line() {
        assert_eq!(size_of::<Slot>(), 32);
    }

    #[test]
    fn the_ladder_only_ever_goes_up() {
        for w in LADDER.windows(2) {
            assert!(w[1] > w[0] || w[0] == u32::MAX, "{w:?} does not go up");
        }
        assert_eq!(LADDER[4], 16, "doubling should run out at 16");
        assert_eq!(
            LADDER[5], 20,
            "and a quarter more should be the step after it"
        );
        assert!(
            LADDER.contains(&u32::MAX),
            "the ladder should reach the end of a u32"
        );
        // The two ends of a size, which is what the free list keys on.
        assert_eq!(LADDER[ceil_class(17)], 20);
        assert_eq!(LADDER[floor_class(19)], 16);
        assert_eq!(LADDER[floor_class(20)], 20);
        assert_eq!(LADDER[ceil_class(1)], 1);
    }

    #[test]
    fn a_run_is_the_neighbours_that_were_linked_to_it() {
        let mut g = Adjacency::new();
        for (i, dst) in [7u64, 9, 11].iter().enumerate() {
            g.link(1, *dst, FOLLOWS, i as u32);
        }
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), &[7, 9, 11]);
        assert_eq!(g.edge_slots(1, FOLLOWS, Dir::Out), &[0, 1, 2]);
        assert_eq!(g.degree(1, FOLLOWS, Dir::Out), 3);
        assert_eq!(g.edges(), 3);
        assert_eq!(g.runs(), 4);
    }

    #[test]
    fn a_node_nobody_linked_has_no_neighbours_rather_than_no_answer() {
        let g = Adjacency::new();
        assert!(g.neighbours(1, FOLLOWS, Dir::Out).is_empty());
        assert!(g.edge_slots(1, FOLLOWS, Dir::Out).is_empty());
        assert_eq!(g.degree(1, FOLLOWS, Dir::Out), 0);
        assert!(g.is_empty());
        g.prefetch(1, FOLLOWS, Dir::Out);
    }

    #[test]
    fn an_edge_is_readable_from_both_ends() {
        let mut g = Adjacency::new();
        g.link(1, 2, FOLLOWS, 10);
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), &[2]);
        assert_eq!(g.neighbours(2, FOLLOWS, Dir::In), &[1]);
        assert_eq!(g.edge_slots(2, FOLLOWS, Dir::In), &[10]);
        assert!(g.neighbours(2, FOLLOWS, Dir::Out).is_empty());
    }

    #[test]
    fn one_label_is_not_another() {
        let mut g = Adjacency::new();
        g.link(1, 2, FOLLOWS, 10);
        g.link(1, 3, WORKS_AT, 11);
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), &[2]);
        assert_eq!(g.neighbours(1, WORKS_AT, Dir::Out), &[3]);
    }

    #[test]
    fn a_self_loop_is_at_both_of_its_ends_and_they_are_different_runs() {
        let mut g = Adjacency::new();
        g.link(1, 1, FOLLOWS, 5);
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), &[1]);
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::In), &[1]);
        assert_eq!(g.unlink(1, 1, FOLLOWS), Some(5));
        assert!(g.neighbours(1, FOLLOWS, Dir::Out).is_empty());
        assert!(g.neighbours(1, FOLLOWS, Dir::In).is_empty());
    }

    #[test]
    fn out_only_stores_nothing_incoming() {
        let mut both = Adjacency::new();
        let mut out = Adjacency::out_only();
        for i in 0..1000u64 {
            both.link(i, i + 1, FOLLOWS, i as u32);
            out.link(i, i + 1, FOLLOWS, i as u32);
        }
        assert_eq!(out.neighbours(500, FOLLOWS, Dir::Out), &[501]);
        assert!(out.neighbours(500, FOLLOWS, Dir::In).is_empty());
        assert_eq!(both.neighbours(500, FOLLOWS, Dir::In), &[499]);
        assert!(!out.indexes_incoming());
        assert!(
            out.bytes() * 3 < both.bytes() * 2,
            "{} against {}",
            out.bytes(),
            both.bytes()
        );
    }

    #[test]
    fn unlinking_takes_the_edge_off_both_ends() {
        let mut g = Adjacency::new();
        g.link(1, 2, FOLLOWS, 10);
        g.link(1, 3, FOLLOWS, 11);
        assert_eq!(g.unlink(1, 2, FOLLOWS), Some(10));
        assert_eq!(g.neighbours(1, FOLLOWS, Dir::Out), &[3]);
        assert!(g.neighbours(2, FOLLOWS, Dir::In).is_empty());
        assert_eq!(g.edges(), 1);
        assert_eq!(g.unlink(1, 2, FOLLOWS), None);
        assert_eq!(g.unlink(9, 9, FOLLOWS), None);
    }

    #[test]
    fn a_delete_moves_the_last_edge_into_the_hole() {
        let mut g = Adjacency::new();
        for dst in 1..=5u64 {
            g.link(0, dst, FOLLOWS, dst as u32);
        }
        g.unlink(0, 2, FOLLOWS);
        // The order is gone but nothing else is, and the edge slot went with
        // the neighbour it belonged to.
        let n = g.neighbours(0, FOLLOWS, Dir::Out);
        let e = g.edge_slots(0, FOLLOWS, Dir::Out);
        assert_eq!(sorted(n), vec![1, 3, 4, 5]);
        for (i, node) in n.iter().enumerate() {
            assert_eq!(u64::from(e[i]), *node, "the pairing survived the swap");
        }
    }

    #[test]
    fn unlink_at_moves_the_last_entry_into_the_position_it_took() {
        let mut g = Adjacency::new();
        for dst in 1..=4u64 {
            g.link(0, dst, FOLLOWS, dst as u32);
        }
        assert_eq!(g.unlink_at(0, FOLLOWS, Dir::Out, 0), Some((1, 1)));
        assert_eq!(g.neighbours(0, FOLLOWS, Dir::Out), &[4, 2, 3]);
        assert_eq!(g.unlink_at(0, FOLLOWS, Dir::Out, 9), None);
        assert_eq!(g.unlink_at(7, FOLLOWS, Dir::Out, 0), None);
    }

    #[test]
    fn a_run_survives_growing_through_every_size_it_passes() {
        let mut g = Adjacency::out_only();
        let n = 5000u64;
        for dst in 0..n {
            g.link(0, dst, FOLLOWS, dst as u32);
        }
        assert_eq!(g.degree(0, FOLLOWS, Dir::Out), n as usize);
        assert_eq!(
            g.neighbours(0, FOLLOWS, Dir::Out),
            (0..n).collect::<Vec<_>>()
        );
        assert_eq!(
            g.edge_slots(0, FOLLOWS, Dir::Out),
            (0..n as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_hub_that_empties_gives_its_block_back() {
        let mut g = Adjacency::out_only();
        for dst in 0..4000u64 {
            g.link(0, dst, FOLLOWS, 0);
        }
        let full = g.bytes();
        for dst in 0..4000u64 {
            assert!(g.unlink(0, dst, FOLLOWS).is_some());
        }
        assert_eq!(g.degree(0, FOLLOWS, Dir::Out), 0);
        assert_eq!(g.runs(), 0);
        // Filling a second node to the same size reuses what the first gave
        // back rather than asking the arena for more.
        for dst in 0..4000u64 {
            g.link(1, dst, FOLLOWS, 0);
        }
        assert!(g.bytes() <= full + full / 4, "{} against {full}", g.bytes());
    }

    #[test]
    fn a_run_that_grows_and_shrinks_does_not_copy_itself_on_a_boundary() {
        // Sixteen is the last doubling, so this sits astride it. What is
        // checked is that the capacity settles rather than that anything is
        // fast: an implementation that shrank on the exact fit would move the
        // run on every one of these, and the arena would grow without end.
        let mut g = Adjacency::out_only();
        for dst in 0..16u64 {
            g.link(0, dst, FOLLOWS, 0);
        }
        // The first cycle does grow the run once, from the 16 it fits exactly
        // into to the 20 above it. Everything after that is the claim.
        g.link(0, 999, FOLLOWS, 0);
        g.unlink(0, 999, FOLLOWS);
        let settled = g.bytes();
        for _ in 0..100 {
            g.link(0, 999, FOLLOWS, 0);
            g.unlink(0, 999, FOLLOWS);
        }
        assert_eq!(g.bytes(), settled);
        assert_eq!(g.degree(0, FOLLOWS, Dir::Out), 16);
    }

    #[test]
    fn a_run_that_loses_most_of_itself_gives_the_room_back() {
        let mut g = Adjacency::out_only();
        for dst in 0..4000u64 {
            g.link(0, dst, FOLLOWS, 0);
        }
        for dst in 0..3990u64 {
            g.unlink(0, dst, FOLLOWS);
        }
        assert_eq!(g.degree(0, FOLLOWS, Dir::Out), 10);
        // The run itself came down on the way, without waiting for a sweep.
        assert!(g.slots.iter().any(|s| s.len == 10 && s.cap <= 16));
        g.compact();
        assert_eq!(g.degree(0, FOLLOWS, Dir::Out), 10);
        assert!(g.bytes() < 4000, "{} bytes for ten edges", g.bytes());
    }

    #[test]
    fn compact_drops_the_runs_that_emptied() {
        let mut g = Adjacency::new();
        for i in 0..2000u64 {
            g.link(i, i + 1, FOLLOWS, i as u32);
        }
        for i in 0..1990u64 {
            g.unlink(i, i + 1, FOLLOWS);
        }
        assert_eq!(g.runs(), 20);
        let before = g.bytes();
        g.compact();
        assert_eq!(g.runs(), 20);
        assert_eq!(g.edges(), 10);
        assert_eq!(g.neighbours(1995, FOLLOWS, Dir::Out), &[1996]);
        assert_eq!(g.neighbours(1996, FOLLOWS, Dir::In), &[1995]);
        assert!(g.bytes() * 4 < before, "{} against {before}", g.bytes());
        // And it is still a working plane afterwards, which is the part a
        // rebuild is easy to get wrong. This one also has to grow a run whose
        // capacity the sweep cut to exactly what it held.
        g.link(1995, 3000, FOLLOWS, 7);
        assert_eq!(
            sorted(g.neighbours(1995, FOLLOWS, Dir::Out)),
            vec![1996, 3000]
        );
        assert_eq!(sorted(g.neighbours(3000, FOLLOWS, Dir::In)), vec![1995]);
    }

    #[test]
    fn a_hot_run_costs_about_twelve_bytes_an_edge() {
        // A degree distribution with a tail, because a uniform one hides both
        // things that could go wrong: the run headers, which a graph of hubs
        // has too few of to notice, and the capacity slack, which a graph of
        // leaves never reaches.
        let mut g = Adjacency::out_only();
        let mut rng = Rng::new(0x9e3f);
        let nodes = 200_000u64;
        let mut edges = 0usize;
        for src in 0..nodes {
            let deg = match rng.next_u64() % 1000 {
                0..=799 => 1 + rng.next_u64() % 4,
                800..=979 => 5 + rng.next_u64() % 40,
                _ => 45 + rng.next_u64() % 600,
            };
            for _ in 0..deg {
                g.link(src, rng.next_u64() % nodes, FOLLOWS, 0);
                edges += 1;
            }
        }
        let per = g.bytes() as f64 / edges as f64;
        g.compact();
        let settled = g.bytes() as f64 / edges as f64;
        // Twelve is the payload and the rest is one 32 byte header per run
        // against an average degree in the teens, plus the capacity slack.
        // Both are the price of a structure that inserts and deletes in
        // constant time, and the cold form is where 8 bits an edge comes from.
        assert!(per < 19.0, "{per:.2} bytes an edge over {edges} edges");
        assert!(settled < 16.0, "{settled:.2} bytes an edge once swept");
    }

    #[test]
    fn a_two_hop_reaches_what_a_pair_of_one_hops_reaches() {
        let mut g = Adjacency::new();
        let mut rng = Rng::new(7);
        let nodes = 5000u64;
        for src in 0..nodes {
            for _ in 0..8 {
                g.link(src, rng.next_u64() % nodes, FOLLOWS, 0);
            }
        }
        let first = g.neighbours(0, FOLLOWS, Dir::Out).to_vec();
        for hop in &first {
            g.prefetch(*hop, FOLLOWS, Dir::Out);
        }
        let mut seen = Vec::new();
        for hop in &first {
            seen.extend_from_slice(g.neighbours(*hop, FOLLOWS, Dir::Out));
        }
        assert_eq!(seen.len(), 64);
        // Every edge is at both ends, so everything the walk reached agrees it
        // was reached from where the walk was standing.
        for (i, hop) in first.iter().enumerate() {
            for dst in &seen[i * 8..(i + 1) * 8] {
                assert!(g.neighbours(*dst, FOLLOWS, Dir::In).contains(hop));
            }
        }
    }

    #[test]
    fn the_plane_agrees_with_a_list_of_what_was_done_to_it() {
        // The reference is a plain vector per node, which is obviously right
        // and obviously too expensive, and the point is that the plane matches
        // it over a mix of links and unlinks that crosses every capacity size
        // in both directions.
        let mut g = Adjacency::new();
        let mut want: Vec<Vec<u64>> = vec![Vec::new(); 64];
        let mut rng = Rng::new(0xbeef);
        for _ in 0..200_000 {
            let src = rng.next_u64() % 64;
            let dst = rng.next_u64() % 64;
            if rng.next_u64().is_multiple_of(3) {
                if let Some(i) = want[src as usize].iter().position(|n| *n == dst) {
                    want[src as usize].swap_remove(i);
                    assert!(g.unlink(src, dst, FOLLOWS).is_some());
                } else {
                    assert_eq!(g.unlink(src, dst, FOLLOWS), None);
                }
            } else {
                want[src as usize].push(dst);
                g.link(src, dst, FOLLOWS, 0);
            }
        }
        let mut total = 0;
        for (src, list) in want.iter().enumerate() {
            assert_eq!(
                sorted(g.neighbours(src as u64, FOLLOWS, Dir::Out)),
                sorted(list),
                "node {src}"
            );
            total += list.len();
        }
        assert_eq!(g.edges(), total);
        // And the incoming side is the transpose of the outgoing one.
        let mut incoming: Vec<Vec<u64>> = vec![Vec::new(); 64];
        for (src, list) in want.iter().enumerate() {
            for dst in list {
                incoming[*dst as usize].push(src as u64);
            }
        }
        for (dst, list) in incoming.iter().enumerate() {
            assert_eq!(
                sorted(g.neighbours(dst as u64, FOLLOWS, Dir::In)),
                sorted(list),
                "into node {dst}"
            );
        }
    }
}
