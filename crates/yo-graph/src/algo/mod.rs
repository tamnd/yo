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
pub mod triangle;
pub mod wcc;

pub use bfs::{UNREACHED, bfs};
pub use pagerank::{Rank, pagerank, pagerank_with};
pub use triangle::triangle_count;
pub use wcc::{Components, wcc};

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
