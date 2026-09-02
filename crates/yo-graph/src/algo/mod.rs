//! The algorithms, over a [`crate::Snapshot`] (`11` section 8).
//!
//! Every one of these is a whole graph computation rather than a traversal: it
//! reads every node, several times, in an order it chooses. That is the
//! opposite of what the adjacency plane is built for, which is why they all
//! take a snapshot and none of them takes a [`crate::Graph`].
//!
//! # What is implemented and where it comes from
//!
//! These are not folk implementations. Each one is the published algorithm that
//! is currently the fastest single machine answer for its problem, and the
//! reference is named in the module that implements it, so a reader can check
//! the code against the paper rather than against a guess.
//!
//! [`bfs()`] is direction optimizing, from Beamer, Asanović and Patterson at SC12,
//! which is the same algorithm the GAP benchmark suite measures and the reason a
//! breadth first search over a social graph is not bound by the size of its
//! frontier.
//!
//! [`wcc()`] is Afforest, from Sutton, Ben-Nun and Barak at IPDPS 2018, which
//! finds the giant component out of a two neighbour sample and then only looks
//! at the edges of the nodes that are not in it.
//!
//! [`pagerank()`] is the pull form, which is the one the GAP suite measures, with
//! the mass that lands on a dead end handed back out rather than dropped, which
//! is what the 1999 paper describes and what GAP leaves out.
//!
//! [`triangle_count()`] is the ordered count from Schank and Wagner at WEA 2005
//! under the degree ordering Ortmann and Brandes recommend at ALENEX 2014, so a
//! hub is intersected against almost nothing rather than against everybody.
//!
//! [`sssp()`] is delta stepping, from Meyer and Sanders in the Journal of
//! Algorithms 2003, which settles a band of nodes at a time instead of one at a
//! time so the reads are independent of each other and a heap is not in the way.
//!
//! [`scc()`] is Tarjan from 1972, which is still the fastest single core answer
//! for strong components, written with its frames in a `Vec` so a long chain
//! does not take the process down with it.
//!
//! # Why they are deterministic
//!
//! Two of them sample, and both sample from [`yo_common::Rng`] with a fixed
//! seed. A caller who runs the same algorithm over the same snapshot twice gets
//! the same answer, including the same representative for a component and the
//! same nodes chosen for a sample. That is worth more than the entropy is: an
//! algorithm whose answer moves between runs cannot be tested against a
//! reference implementation and cannot be diffed between two versions of this
//! crate.

pub mod bfs;
pub mod pagerank;
pub mod scc;
pub mod sssp;
pub mod triangle;
pub mod wcc;

pub use bfs::{UNREACHED, bfs};
pub use pagerank::{Rank, pagerank, pagerank_with};
pub use scc::scc;
pub use sssp::{UNREACHABLE, sssp, sssp_with};
pub use triangle::triangle_count;
pub use wcc::wcc;

/// Which component each node is in.
///
/// What counts as a component is the algorithm's business. [`wcc()`] fills this
/// in with the weakly connected ones, where an edge joins its two ends whichever
/// way it points, and [`scc()`] with the strongly connected ones, where two
/// nodes are together only if each can be reached from the other. The shape of
/// the answer is the same either way, and so is the rule about the label.
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

/// A bit per node, which is how a frontier is held when it is big.
///
/// A frontier as a list of nodes costs four bytes a node and is read once. A
/// frontier as a bitmap costs one bit a node whether the node is in it or not,
/// and can be asked about a node without being searched. Which one is cheaper
/// depends on how full the frontier is, and a breadth first search over a real
/// graph goes from one being right to the other being right and back inside a
/// single search, so [`bfs()`] holds both and converts.
#[derive(Debug, Clone)]
pub(crate) struct Bits {
    words: Vec<u64>,
    len: u32,
}

impl Bits {
    /// Room for `len` bits, all clear.
    pub(crate) fn new(len: u32) -> Bits {
        Bits {
            words: vec![0; (len as usize).div_ceil(64)],
            len,
        }
    }

    /// Clears every bit, without giving the memory back.
    pub(crate) fn clear(&mut self) {
        self.words.fill(0);
    }

    #[inline]
    pub(crate) fn set(&mut self, at: u32) {
        self.words[at as usize / 64] |= 1 << (at % 64);
    }

    #[inline]
    pub(crate) fn unset(&mut self, at: u32) {
        self.words[at as usize / 64] &= !(1 << (at % 64));
    }

    #[inline]
    pub(crate) fn get(&self, at: u32) -> bool {
        self.words[at as usize / 64] >> (at % 64) & 1 == 1
    }

    /// Every set bit, in order.
    ///
    /// A word at a time and then a bit at a time inside a word that has
    /// anything in it, so an empty stretch of a sparse frontier costs one load
    /// and one compare per sixty four nodes rather than one per node.
    pub(crate) fn for_each(&self, mut f: impl FnMut(u32)) {
        for (i, word) in self.words.iter().enumerate() {
            let mut w = *word;
            while w != 0 {
                let at = i as u32 * 64 + w.trailing_zeros();
                if at >= self.len {
                    return;
                }
                f(at);
                w &= w - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bitmap_holds_what_was_put_in_it() {
        let mut b = Bits::new(200);
        for at in [0u32, 1, 63, 64, 65, 199] {
            b.set(at);
        }
        assert!(b.get(64));
        assert!(!b.get(66));

        let mut seen = Vec::new();
        b.for_each(|at| seen.push(at));
        assert_eq!(seen, vec![0, 1, 63, 64, 65, 199]);

        b.clear();
        assert!(!b.get(0));
        assert!(!b.get(199));
    }

    /// The last word has bits past the end of the graph in it, and nothing may
    /// hand one of them back as a node.
    #[test]
    fn a_bit_past_the_end_is_not_a_node() {
        let mut b = Bits::new(3);
        b.set(0);
        b.set(2);
        // Reaching into the spare bits of the last word, which only this test
        // can do, because nothing else knows a node it has not been given.
        b.words[0] |= 1 << 40;
        let mut seen = Vec::new();
        b.for_each(|at| seen.push(at));
        assert_eq!(seen, vec![0, 2]);
    }
}
