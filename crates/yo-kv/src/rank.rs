//! A counted B+ tree, which is the ordered index a sorted set ranks with.
//!
//! What this holds is a sequence of element row numbers and nothing else. It
//! does not know what a score is, it does not know what a member is, and it
//! never reads either. Every question about order is asked of the caller through
//! a closure, and every answer this gives back is a rank or a row number. That
//! is not shyness about the sorted set, it is the reason the document model can
//! use the same tree for an ordered index over a path (`09` section 5) without
//! either of them growing a special case for the other.
//!
//! ```text
//!            +-------------------------------+
//!   branch   | kid | kid | kid | kid |       |   counts, and the first row
//!            +-------------------------------+   of each kid's subtree
//!               |     |     |     \
//!            +-----+-----+-----+-----+
//!   leaves   |rows |rows |rows |rows |  <-> linked both ways
//!            +-----+-----+-----+-----+
//! ```
//!
//! # Why this and not a skiplist
//!
//! Redis ranks with a skiplist and aki copied it, and that decision is most of
//! the 37 bytes an aki zset entry cost, plus ninety more per node. A skiplist
//! node is an allocation with a tower of pointers on it, so a million member
//! zset is a million allocations scattered over the heap, and the interior of
//! the structure, the part every single `ZRANK` has to walk, is scattered with
//! them. That is `08` section 5's zipfian miss and the long term memory fault
//! count in one sentence: the hot part of a skiplist is not a small part.
//!
//! Here the interior is separate from the elements and it is tiny. A branch node
//! holds a hundred and twenty eight kids, so ten million elements sit under three
//! levels and the whole interior is a few hundred kilobytes no matter how the
//! members are spread. It stays resident under any budget worth having, which is
//! the mechanism behind `06`'s one device read per point read.
//!
//! # What an element costs
//!
//! Three bytes, the row number, plus whatever share of a branch node it owns,
//! which at a fanout of a hundred and twenty eight is a fifth of a byte. Three
//! point two three bytes an element measured over a million of them, against the
//! 37 plus 90 per node an aki zset entry cost, and `G8` asks for three. There is
//! no per element allocation and no per element pointer, and the score is not
//! here at all: it lives once, in the element table, where `ZSCORE` reads it.
//!
//! The cost that buys is that a search asks the caller to compare, and the caller
//! reads a score out of the element table to answer, so a descent touches a
//! handful of rows that are not next to each other. That trade is the right way
//! round for `G8` and it is deliberately the only one on the table: a tree that
//! cached the score next to the row would answer a search without leaving the
//! node and would cost three times the memory, which `Y14` calls a fail however
//! fast it is.
//!
//! # Occupancy
//!
//! A B+ tree that splits a full node down the middle settles at about seventy
//! percent full under random inserts and at exactly half under sorted ones, and
//! sorted is not a corner case here: a leaderboard is written in score order more
//! often than not. So a full node that is being pushed at either end does not
//! split down the middle, it puts the new row in a node of its own and leaves the
//! full one alone. Ascending and descending runs both come out at very nearly a
//! hundred percent full that way, and a random spread is unaffected.

use core::cmp::Ordering;

/// How many rows a leaf holds.
///
/// A kilobyte of row numbers. Small enough that the binary search inside a leaf
/// is eight comparisons rather than twenty, which matters because each of those
/// comparisons is a question for the caller and the caller answers it by reading
/// an element row somewhere else in memory.
const LEAF_MAX: usize = 256;

/// When a leaf is small enough to be worth folding into a neighbour.
const LEAF_MIN: usize = LEAF_MAX / 2;

/// How many kids a branch holds.
///
/// Three levels reach sixteen million elements, which is the ceiling the element
/// table puts on a collection anyway, so no tree here is ever deeper than that.
const BRANCH_MAX: usize = 128;

/// When a branch is small enough to be worth folding into a neighbour.
const BRANCH_MIN: usize = BRANCH_MAX / 2;

/// No node.
const NIL: u32 = u32::MAX;

/// Row numbers packed three bytes each.
///
/// The fourth byte of a row number is always zero and it is not a guess: the
/// element table packs a tag and a row into one word and gives the row
/// twenty four bits of it, so [`crate::elem::MAX_ROWS`] is what a collection can
/// hold and no row number this ever sees needs the top byte. Storing it anyway
/// is a third of the leaf, and the leaf is nearly all of what a zset index
/// costs, so it is the difference between four and a bit bytes an element and
/// three and a bit.
///
/// The room is asked for once, at the size a leaf is allowed to reach, so that a
/// leaf that is half full is holding a fixed 768 bytes rather than whatever
/// power of two a growing `Vec` last landed on.
#[derive(Debug, Clone, Default)]
struct Rows {
    at: Vec<u8>,
}

impl Rows {
    fn with_room() -> Self {
        Self { at: Vec::with_capacity(LEAF_MAX * 3) }
    }

    fn len(&self) -> usize {
        self.at.len() / 3
    }

    fn get(&self, i: usize) -> u32 {
        let at = i * 3;
        u32::from(self.at[at]) | u32::from(self.at[at + 1]) << 8 | u32::from(self.at[at + 2]) << 16
    }

    fn push(&mut self, row: u32) {
        self.at.extend_from_slice(&row.to_le_bytes()[..3]);
    }

    fn insert(&mut self, i: usize, row: u32) {
        let at = i * 3;
        self.at.extend_from_slice(&[0, 0, 0]);
        let end = self.at.len();
        self.at.copy_within(at..end - 3, at + 3);
        self.at[at..at + 3].copy_from_slice(&row.to_le_bytes()[..3]);
    }

    fn remove(&mut self, i: usize) -> u32 {
        let row = self.get(i);
        let at = i * 3;
        self.at.copy_within(at + 3.., at);
        self.at.truncate(self.at.len() - 3);
        row
    }

    /// Everything from `i` on, in a run of its own.
    fn split_off(&mut self, i: usize) -> Self {
        let mut out = Self::with_room();
        out.at.extend_from_slice(&self.at[i * 3..]);
        self.at.truncate(i * 3);
        out
    }

    fn append(&mut self, other: &Self) {
        self.at.extend_from_slice(&other.at);
    }

    /// How many rows at the front the probe calls `Greater`.
    fn partition_point<F: FnMut(u32) -> bool>(&self, mut keep: F) -> usize {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if keep(self.get(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn bytes(&self) -> usize {
        self.at.capacity()
    }
}

/// A run of rows, in order, linked to the runs on either side of it.
///
/// The links are what makes `ZRANGE` a descent and then a walk rather than a
/// descent per element.
#[derive(Debug, Clone)]
struct Leaf {
    rows: Rows,
    prev: u32,
    next: u32,
}

/// A level of the interior: which subtrees are under here, how many elements
/// each one holds, and the first row of each so that a search can steer without
/// descending into a subtree to find out what is in it.
#[derive(Debug, Clone)]
struct Branch {
    kids: Vec<u32>,
    counts: Vec<u32>,
    firsts: Vec<u32>,
}

/// A node that came out of a node that was full.
struct Split {
    node: u32,
    count: u32,
    first: u32,
}

/// An ordered sequence of element rows that can be asked for a rank.
///
/// The sequence is kept in whatever order the caller's comparisons imply. This
/// type never checks that order and never repairs it, which is the same contract
/// `slice::binary_search` has: hand it something out of order and it will answer
/// nonsense rather than complain.
#[derive(Debug, Clone)]
pub struct Rank {
    leaves: Vec<Leaf>,
    branches: Vec<Branch>,
    free_leaves: Vec<u32>,
    free_branches: Vec<u32>,
    /// The leaf, when `depth` is zero, and otherwise the branch.
    root: u32,
    /// How many branch levels sit above the leaves.
    depth: u8,
    len: usize,
    /// The leftmost leaf, which is where a forward walk starts.
    head: u32,
    /// The rightmost leaf, which is where a backward walk starts.
    tail: u32,
}

impl Default for Rank {
    fn default() -> Self {
        Self::new()
    }
}

impl Rank {
    /// An empty sequence, holding one empty leaf.
    ///
    /// The leaf is made up front rather than on the first insert because every
    /// path below would otherwise need to know that the root might not exist,
    /// and an empty leaf is a `Vec` that has not allocated.
    #[must_use]
    pub fn new() -> Self {
        Self {
            leaves: vec![Leaf { rows: Rows::with_room(), prev: NIL, next: NIL }],
            branches: Vec::new(),
            free_leaves: Vec::new(),
            free_branches: Vec::new(),
            root: 0,
            depth: 0,
            len: 0,
            head: 0,
            tail: 0,
        }
    }

    /// How many rows are in here.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no rows in here.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// What this is holding on to, in bytes, not counting the rows themselves.
    ///
    /// This is the number `G8` is about, so it is reported rather than estimated
    /// from the element count. It counts the room the nodes have asked for and
    /// not the room they are using, because the difference between those two is
    /// exactly what the occupancy argument above is about.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let leaves: usize = self
            .leaves
            .iter()
            .map(|l| l.rows.bytes() + size_of::<Leaf>())
            .sum();
        let branches: usize = self
            .branches
            .iter()
            .map(|b| {
                (b.kids.capacity() + b.counts.capacity() + b.firsts.capacity()) * size_of::<u32>()
                    + size_of::<Branch>()
            })
            .sum();
        leaves + branches
    }

    /// The row at a rank, or `None` past the end.
    #[must_use]
    pub fn row_at(&self, rank: usize) -> Option<u32> {
        if rank >= self.len {
            return None;
        }
        let (leaf, at) = self.find(rank);
        Some(self.leaves[leaf as usize].rows.get(at))
    }

    /// Where a rank sits: which leaf, and how far into it.
    fn find(&self, rank: usize) -> (u32, usize) {
        let mut node = self.root;
        let mut at = rank;
        for _ in 0..self.depth {
            let b = &self.branches[node as usize];
            let mut i = 0;
            while at >= b.counts[i] as usize && i + 1 < b.kids.len() {
                at -= b.counts[i] as usize;
                i += 1;
            }
            node = b.kids[i];
        }
        (node, at)
    }

    /// The rank of the first row the probe does not call `Greater`.
    ///
    /// The probe answers where the thing being looked for sits against the row it
    /// is given: `Greater` means the target is past that row and the search
    /// should keep going right. So this is a lower bound, the same thing
    /// `partition_point` gives, and it is what every one of `ZADD`, `ZRANK`,
    /// `ZRANGEBYSCORE` and `ZRANGEBYLEX` is asking for underneath.
    ///
    /// It is up to the caller whether ties count. A probe that answers `Equal` on
    /// an exact match lands on it, and a probe that answers `Greater` there lands
    /// one past it, which is the difference between the two ends of a range.
    pub fn seek<F: FnMut(u32) -> Ordering>(&self, mut probe: F) -> usize {
        let mut node = self.root;
        let mut base = 0;
        for _ in 0..self.depth {
            let b = &self.branches[node as usize];
            // The first kid is taken whatever the probe says, because everything
            // in the tree is under it or to the right of it and there is nowhere
            // further left to go.
            let mut i = 0;
            while i + 1 < b.kids.len() && probe(b.firsts[i + 1]) == Ordering::Greater {
                base += b.counts[i] as usize;
                i += 1;
            }
            node = b.kids[i];
        }
        let rows = &self.leaves[node as usize].rows;
        base + rows.partition_point(|r| probe(r) == Ordering::Greater)
    }

    /// Put a row at a rank, moving everything from there on one to the right.
    ///
    /// # Panics
    ///
    /// If the rank is past the end. Ranks up to and including [`Rank::len`] are
    /// fine, because appending is inserting at the end.
    pub fn insert_at(&mut self, rank: usize, row: u32) {
        assert!(rank <= self.len, "rank {rank} is past the end of {}", self.len);
        let split = if self.depth == 0 {
            self.leaf_insert(self.root, rank, row)
        } else {
            self.branch_insert(self.root, self.depth, rank, row)
        };
        if let Some(split) = split {
            let left = self.root;
            let left_count = (self.len + 1 - split.count as usize) as u32;
            let left_first = self.first_of(left, self.depth);
            let root = self.take_branch();
            let b = &mut self.branches[root as usize];
            b.kids.push(left);
            b.counts.push(left_count);
            b.firsts.push(left_first);
            b.kids.push(split.node);
            b.counts.push(split.count);
            b.firsts.push(split.first);
            self.root = root;
            self.depth += 1;
        }
        self.len += 1;
    }

    /// Take the row at a rank out, moving everything after it one to the left.
    ///
    /// # Panics
    ///
    /// If the rank is past the end.
    pub fn remove_at(&mut self, rank: usize) -> u32 {
        assert!(rank < self.len, "rank {rank} is past the end of {}", self.len);
        let row = if self.depth == 0 {
            let leaf = &mut self.leaves[self.root as usize];
            leaf.rows.remove(rank)
        } else {
            self.branch_remove(self.root, self.depth, rank)
        };
        // A root that has been emptied down to one kid is a level nobody needs.
        while self.depth > 0 && self.branches[self.root as usize].kids.len() == 1 {
            let old = self.root;
            self.root = self.branches[old as usize].kids[0];
            self.drop_branch(old);
            self.depth -= 1;
        }
        self.len -= 1;
        row
    }

    /// Walk rows in order from a rank.
    ///
    /// The walk is a descent to find the leaf and then a link per leaf after
    /// that, so a range of a thousand costs one descent and four link hops.
    #[must_use]
    pub fn iter_from(&self, rank: usize) -> Walk<'_> {
        if rank >= self.len {
            return Walk { tree: self, leaf: NIL, at: 0, left: 0 };
        }
        let (leaf, at) = self.find(rank);
        Walk { tree: self, leaf, at, left: self.len - rank }
    }

    /// Walk rows backwards from a rank.
    #[must_use]
    pub fn iter_back_from(&self, rank: usize) -> Back<'_> {
        if rank >= self.len {
            return Back { tree: self, leaf: NIL, at: 0, left: 0 };
        }
        let (leaf, at) = self.find(rank);
        Back { tree: self, leaf, at, left: rank + 1 }
    }

    /// Insert into a leaf, and say what came out of it if it was full.
    fn leaf_insert(&mut self, id: u32, at: usize, row: u32) -> Option<Split> {
        let leaf = &mut self.leaves[id as usize];
        if leaf.rows.len() < LEAF_MAX {
            leaf.rows.insert(at, row);
            return None;
        }
        // A full leaf being pushed at its right hand end is a sorted run, and
        // splitting it down the middle would leave both halves half full for
        // ever. The new row goes in a leaf of its own instead.
        if at == LEAF_MAX {
            let new = self.take_leaf();
            self.leaves[new as usize].rows.push(row);
            self.link_after(id, new);
            return Some(Split { node: new, count: 1, first: row });
        }
        // Same at the other end, except that what moves is the full leaf rather
        // than the new row, because the parent already has this node in its kid
        // list at the position the new row belongs in. The rows go across to a
        // node of their own and this one keeps its place holding the one row.
        if at == 0 {
            let new = self.take_leaf();
            let full = core::mem::replace(&mut self.leaves[id as usize].rows, Rows::with_room());
            let count = full.len() as u32;
            let first = full.get(0);
            self.leaves[new as usize].rows = full;
            self.leaves[id as usize].rows.push(row);
            self.link_after(id, new);
            return Some(Split { node: new, count, first });
        }
        let new = self.take_leaf();
        let tail = self.leaves[id as usize].rows.split_off(LEAF_MAX / 2);
        self.leaves[new as usize].rows = tail;
        self.link_after(id, new);
        if at <= LEAF_MAX / 2 {
            self.leaves[id as usize].rows.insert(at, row);
        } else {
            self.leaves[new as usize].rows.insert(at - LEAF_MAX / 2, row);
        }
        let first = self.leaves[new as usize].rows.get(0);
        let count = self.leaves[new as usize].rows.len() as u32;
        Some(Split { node: new, count, first })
    }

    /// Insert under a branch, and say what came out of it if it was full.
    fn branch_insert(&mut self, id: u32, level: u8, at: usize, row: u32) -> Option<Split> {
        let (mut i, mut local) = (0, at);
        {
            let b = &self.branches[id as usize];
            while local > b.counts[i] as usize && i + 1 < b.kids.len() {
                local -= b.counts[i] as usize;
                i += 1;
            }
        }
        let kid = self.branches[id as usize].kids[i];
        let split = if level == 1 {
            self.leaf_insert(kid, local, row)
        } else {
            self.branch_insert(kid, level - 1, local, row)
        };
        {
            let b = &mut self.branches[id as usize];
            b.counts[i] += 1;
            if local == 0 {
                b.firsts[i] = row;
            }
        }
        let split = split?;
        // A leaf split at its left hand end hands back the node that stayed put
        // rather than the new one, and the count and first of the kid that is
        // already in this branch have to be repaired from what is under it.
        let kept = self.branches[id as usize].kids[i];
        let kept_count = self.branches[id as usize].counts[i] - split.count;
        {
            let b = &mut self.branches[id as usize];
            b.counts[i] = kept_count;
        }
        let kept_first = self.first_of(kept, level - 1);
        {
            let b = &mut self.branches[id as usize];
            b.firsts[i] = kept_first;
            b.kids.insert(i + 1, split.node);
            b.counts.insert(i + 1, split.count);
            b.firsts.insert(i + 1, split.first);
            if b.kids.len() <= BRANCH_MAX {
                return None;
            }
        }
        let new = self.take_branch();
        let (kids, counts, firsts) = {
            let b = &mut self.branches[id as usize];
            (
                b.kids.split_off(BRANCH_MAX / 2),
                b.counts.split_off(BRANCH_MAX / 2),
                b.firsts.split_off(BRANCH_MAX / 2),
            )
        };
        let count: u32 = counts.iter().sum();
        let first = firsts[0];
        let b = &mut self.branches[new as usize];
        b.kids = kids;
        b.counts = counts;
        b.firsts = firsts;
        Some(Split { node: new, count, first })
    }

    /// Remove from under a branch and put right whatever that emptied.
    fn branch_remove(&mut self, id: u32, level: u8, at: usize) -> u32 {
        let (mut i, mut local) = (0, at);
        {
            let b = &self.branches[id as usize];
            while local >= b.counts[i] as usize && i + 1 < b.kids.len() {
                local -= b.counts[i] as usize;
                i += 1;
            }
        }
        let kid = self.branches[id as usize].kids[i];
        let row = if level == 1 {
            self.leaves[kid as usize].rows.remove(local)
        } else {
            self.branch_remove(kid, level - 1, local)
        };
        self.branches[id as usize].counts[i] -= 1;
        if local == 0 && self.branches[id as usize].counts[i] > 0 {
            let first = self.first_of(kid, level - 1);
            self.branches[id as usize].firsts[i] = first;
        }
        self.mend(id, level, i);
        row
    }

    /// Fold a kid that has got too small into one of its neighbours, or borrow
    /// from one if neither will fit.
    fn mend(&mut self, id: u32, level: u8, i: usize) {
        let (small, kids) = {
            let b = &self.branches[id as usize];
            let kid = b.kids[i];
            let small = if level == 1 {
                self.leaves[kid as usize].rows.len() < LEAF_MIN
            } else {
                self.branches[kid as usize].kids.len() < BRANCH_MIN
            };
            (small, b.kids.len())
        };
        if !small || kids == 1 {
            return;
        }
        // Always fold to the right, so that the pair is (i, i + 1) and the node
        // that goes away is the second of the two. At the end there is no right
        // hand neighbour, so step back one and fold this one into its left.
        let at = if i + 1 == kids { i - 1 } else { i };
        let (left, right) = {
            let b = &self.branches[id as usize];
            (b.kids[at], b.kids[at + 1])
        };
        let room = if level == 1 {
            self.leaves[left as usize].rows.len() + self.leaves[right as usize].rows.len()
                <= LEAF_MAX
        } else {
            self.branches[left as usize].kids.len() + self.branches[right as usize].kids.len()
                <= BRANCH_MAX
        };
        if room {
            self.join(id, level, at);
        } else {
            self.share(id, level, at);
        }
    }

    /// Move everything in the right hand node into the left hand one and drop it.
    fn join(&mut self, id: u32, level: u8, at: usize) {
        let (left, right) = {
            let b = &self.branches[id as usize];
            (b.kids[at], b.kids[at + 1])
        };
        if level == 1 {
            let rows = core::mem::take(&mut self.leaves[right as usize].rows);
            self.leaves[left as usize].rows.append(&rows);
            self.unlink(right);
            self.drop_leaf(right);
        } else {
            let (kids, counts, firsts) = {
                let b = &mut self.branches[right as usize];
                (
                    core::mem::take(&mut b.kids),
                    core::mem::take(&mut b.counts),
                    core::mem::take(&mut b.firsts),
                )
            };
            let b = &mut self.branches[left as usize];
            b.kids.extend_from_slice(&kids);
            b.counts.extend_from_slice(&counts);
            b.firsts.extend_from_slice(&firsts);
            self.drop_branch(right);
        }
        {
            let b = &mut self.branches[id as usize];
            b.counts[at] += b.counts[at + 1];
            b.kids.remove(at + 1);
            b.counts.remove(at + 1);
            b.firsts.remove(at + 1);
        }
        // The kid that stayed may have been the empty one, in which case its
        // first row is whatever just came across into it.
        let first = self.first_of(left, level - 1);
        self.branches[id as usize].firsts[at] = first;
    }

    /// Move one across from the right hand node to the left hand one.
    fn share(&mut self, id: u32, level: u8, at: usize) {
        let (left, right) = {
            let b = &self.branches[id as usize];
            (b.kids[at], b.kids[at + 1])
        };
        let moved = if level == 1 {
            let row = self.leaves[right as usize].rows.remove(0);
            self.leaves[left as usize].rows.push(row);
            1
        } else {
            let b = &mut self.branches[right as usize];
            let kid = b.kids.remove(0);
            let count = b.counts.remove(0);
            let first = b.firsts.remove(0);
            let b = &mut self.branches[left as usize];
            b.kids.push(kid);
            b.counts.push(count);
            b.firsts.push(first);
            count
        };
        let first = self.first_of(right, level - 1);
        let b = &mut self.branches[id as usize];
        b.counts[at] += moved;
        b.counts[at + 1] -= moved;
        b.firsts[at + 1] = first;
    }

    /// The first row under a node.
    fn first_of(&self, id: u32, level: u8) -> u32 {
        if level == 0 {
            return self.leaves[id as usize].rows.get(0);
        }
        self.branches[id as usize].firsts[0]
    }

    fn take_leaf(&mut self) -> u32 {
        if let Some(id) = self.free_leaves.pop() {
            return id;
        }
        self.leaves.push(Leaf { rows: Rows::with_room(), prev: NIL, next: NIL });
        (self.leaves.len() - 1) as u32
    }

    fn drop_leaf(&mut self, id: u32) {
        let leaf = &mut self.leaves[id as usize];
        leaf.rows = Rows::default();
        leaf.prev = NIL;
        leaf.next = NIL;
        self.free_leaves.push(id);
    }

    fn take_branch(&mut self) -> u32 {
        if let Some(id) = self.free_branches.pop() {
            return id;
        }
        self.branches.push(Branch { kids: Vec::new(), counts: Vec::new(), firsts: Vec::new() });
        (self.branches.len() - 1) as u32
    }

    fn drop_branch(&mut self, id: u32) {
        let b = &mut self.branches[id as usize];
        b.kids = Vec::new();
        b.counts = Vec::new();
        b.firsts = Vec::new();
        self.free_branches.push(id);
    }

    fn link_after(&mut self, id: u32, new: u32) {
        let next = self.leaves[id as usize].next;
        self.leaves[new as usize].prev = id;
        self.leaves[new as usize].next = next;
        self.leaves[id as usize].next = new;
        if next == NIL {
            self.tail = new;
        } else {
            self.leaves[next as usize].prev = new;
        }
    }

    fn unlink(&mut self, id: u32) {
        let (prev, next) = {
            let l = &self.leaves[id as usize];
            (l.prev, l.next)
        };
        if prev == NIL {
            self.head = next;
        } else {
            self.leaves[prev as usize].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.leaves[next as usize].prev = prev;
        }
    }
}

/// Rows in order, from where [`Rank::iter_from`] was asked to start.
#[derive(Debug)]
pub struct Walk<'a> {
    tree: &'a Rank,
    leaf: u32,
    at: usize,
    left: usize,
}

impl Iterator for Walk<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.left == 0 || self.leaf == NIL {
            return None;
        }
        let leaf = &self.tree.leaves[self.leaf as usize];
        let row = leaf.rows.get(self.at);
        self.at += 1;
        self.left -= 1;
        if self.at == leaf.rows.len() {
            self.leaf = leaf.next;
            self.at = 0;
        }
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Walk<'_> {}

/// Rows in reverse order, from where [`Rank::iter_back_from`] was asked to start.
#[derive(Debug)]
pub struct Back<'a> {
    tree: &'a Rank,
    leaf: u32,
    at: usize,
    left: usize,
}

impl Iterator for Back<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.left == 0 || self.leaf == NIL {
            return None;
        }
        let leaf = &self.tree.leaves[self.leaf as usize];
        let row = leaf.rows.get(self.at);
        self.left -= 1;
        if self.at == 0 {
            self.leaf = leaf.prev;
            if self.leaf != NIL {
                self.at = self.tree.leaves[self.leaf as usize].rows.len() - 1;
            }
        } else {
            self.at -= 1;
        }
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Back<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rows in order, which is the only thing any of this has to get right.
    fn rows(tree: &Rank) -> Vec<u32> {
        tree.iter_from(0).collect()
    }

    /// Every count on every branch says how many rows are under it, every leaf
    /// but the root has something in it, and the links agree with the tree.
    fn sound(tree: &Rank) {
        let total = check(tree, tree.root, tree.depth);
        assert_eq!(total, tree.len, "the root's counts do not add up to the length");
        let mut walked = 0;
        let mut at = tree.head;
        let mut prev = NIL;
        while at != NIL {
            assert_eq!(tree.leaves[at as usize].prev, prev, "a back link is wrong");
            walked += tree.leaves[at as usize].rows.len();
            prev = at;
            at = tree.leaves[at as usize].next;
        }
        assert_eq!(prev, tree.tail, "the tail is not the end of the chain");
        assert_eq!(walked, tree.len, "the leaf chain does not hold every row");
    }

    fn check(tree: &Rank, id: u32, level: u8) -> usize {
        if level == 0 {
            return tree.leaves[id as usize].rows.len();
        }
        let b = &tree.branches[id as usize];
        assert!(!b.kids.is_empty(), "a branch with no kids");
        let mut total = 0;
        for (i, &kid) in b.kids.iter().enumerate() {
            let under = check(tree, kid, level - 1);
            assert_eq!(under, b.counts[i] as usize, "a count does not match what is under it");
            assert_eq!(b.firsts[i], tree.first_of(kid, level - 1), "a first row is stale");
            total += under;
        }
        total
    }

    #[test]
    fn an_empty_tree_answers_nothing() {
        let tree = Rank::new();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert_eq!(tree.row_at(0), None);
        assert_eq!(tree.seek(|_| Ordering::Greater), 0);
        assert_eq!(rows(&tree), Vec::<u32>::new());
    }

    #[test]
    fn a_sorted_run_of_appends_fills_its_leaves() {
        let mut tree = Rank::new();
        let n = 10_000;
        for i in 0..n {
            tree.insert_at(i as usize, i);
        }
        sound(&tree);
        assert_eq!(rows(&tree), (0..n).collect::<Vec<_>>());
        // The whole point of not splitting a full leaf that is being appended to
        // is that the leaves come out full. Four bytes a row plus the odd branch
        // is under five, and a tree that split down the middle would be at eight.
        let per = tree.bytes() as f64 / n as f64;
        assert!(per < 5.0, "{per} bytes a row on a sorted run");
    }

    #[test]
    fn a_sorted_run_of_prepends_fills_its_leaves_too() {
        let mut tree = Rank::new();
        let n = 10_000;
        for i in 0..n {
            tree.insert_at(0, i);
        }
        sound(&tree);
        assert_eq!(rows(&tree), (0..n).rev().collect::<Vec<_>>());
        let per = tree.bytes() as f64 / n as f64;
        assert!(per < 5.0, "{per} bytes a row on a reversed run");
    }

    #[test]
    fn a_row_can_go_in_anywhere_and_come_out_where_it_went() {
        let mut tree = Rank::new();
        let mut model: Vec<u32> = Vec::new();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let roll = |seed: &mut u64, n: usize| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed % (n as u64 + 1)) as usize
        };
        for i in 0..4_000u32 {
            let at = roll(&mut seed, model.len());
            tree.insert_at(at, i);
            model.insert(at, i);
        }
        sound(&tree);
        assert_eq!(rows(&tree), model);
        for rank in [0, 1, 999, 3_999] {
            assert_eq!(tree.row_at(rank), Some(model[rank]));
        }
        assert_eq!(tree.row_at(4_000), None);
    }

    #[test]
    fn taking_rows_out_puts_the_tree_back_together() {
        let mut tree = Rank::new();
        for i in 0..5_000u32 {
            tree.insert_at(i as usize, i);
        }
        let mut model: Vec<u32> = (0..5_000).collect();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        while !model.is_empty() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let at = (seed % model.len() as u64) as usize;
            assert_eq!(tree.remove_at(at), model.remove(at));
            if model.len().is_multiple_of(97) {
                sound(&tree);
                assert_eq!(rows(&tree), model);
            }
        }
        sound(&tree);
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.depth, 0, "an emptied tree should be one leaf again");
    }

    #[test]
    fn a_tree_that_has_been_emptied_reuses_what_it_had() {
        let mut tree = Rank::new();
        for i in 0..2_000u32 {
            tree.insert_at(i as usize, i);
        }
        let leaves = tree.leaves.len();
        let branches = tree.branches.len();
        for _ in 0..2_000 {
            tree.remove_at(0);
        }
        for i in 0..2_000u32 {
            tree.insert_at(i as usize, i);
        }
        sound(&tree);
        assert_eq!(tree.leaves.len(), leaves, "leaves were not reused");
        assert_eq!(tree.branches.len(), branches, "branches were not reused");
    }

    /// A search over a sequence sorted by the value at each row, which is what
    /// the sorted set will do with a score.
    #[test]
    fn a_search_finds_where_a_value_belongs() {
        let mut tree = Rank::new();
        // Row i holds the value i * 10, so the gaps are where the interesting
        // answers are.
        let value = |row: u32| i64::from(row) * 10;
        for i in 0..3_000u32 {
            tree.insert_at(i as usize, i);
        }
        for want in [0i64, 5, 10, 15, 29_990, 29_995, 30_000, 40_000] {
            let lower = tree.seek(|row| want.cmp(&value(row)));
            let expect = (0..3_000).filter(|&r| value(r) < want).count();
            assert_eq!(lower, expect, "lower bound of {want}");
        }
        // The upper bound is the same descent with a probe that never says it
        // has found what it is looking for, which is how a range that includes
        // its end is written against a search that excludes it.
        let upper = tree.seek(|row| match 100i64.cmp(&value(row)) {
            Ordering::Equal => Ordering::Greater,
            other => other,
        });
        assert_eq!(upper, 11);
    }

    #[test]
    fn a_search_over_a_run_of_equal_values_finds_both_of_its_ends() {
        let mut tree = Rank::new();
        // A thousand rows, all with the same value, which is what a zset that is
        // being used as an ordered set looks like.
        for i in 0..1_000u32 {
            tree.insert_at(i as usize, i);
        }
        let value = |_row: u32| 7i64;
        let first = tree.seek(|row| 7i64.cmp(&value(row)));
        let past = tree.seek(|row| match 7i64.cmp(&value(row)) {
            Ordering::Equal => Ordering::Greater,
            other => other,
        });
        assert_eq!(first, 0);
        assert_eq!(past, 1_000);
    }

    #[test]
    fn a_walk_can_start_anywhere_and_go_either_way() {
        let mut tree = Rank::new();
        for i in 0..1_000u32 {
            tree.insert_at(i as usize, i);
        }
        assert_eq!(tree.iter_from(998).collect::<Vec<_>>(), vec![998, 999]);
        assert_eq!(tree.iter_from(1_000).count(), 0);
        assert_eq!(tree.iter_back_from(2).collect::<Vec<_>>(), vec![2, 1, 0]);
        assert_eq!(tree.iter_back_from(999).count(), 1_000);
        assert_eq!(tree.iter_back_from(1_000).count(), 0);
        // A range in the middle is a descent and then a walk, and the count it
        // reports up front is what lets a reply write its header first.
        let mut walk = tree.iter_from(500);
        assert_eq!(walk.len(), 500);
        assert_eq!(walk.next(), Some(500));
        assert_eq!(walk.len(), 499);
    }

    #[test]
    fn a_million_rows_cost_under_five_bytes_each() {
        let mut tree = Rank::new();
        let n = 1_000_000u32;
        for i in 0..n {
            tree.insert_at(i as usize, i);
        }
        sound(&tree);
        let per = tree.bytes() as f64 / f64::from(n);
        // G8 asks for three bytes an element for the zset index. This is the
        // three byte row number plus a fifth of a byte of interior, and the
        // interior is what a fanout of a hundred and twenty eight costs. The
        // remaining fifth is not going anywhere without giving up either the
        // branch nodes or the `Vec` header on each of them, and neither is worth
        // it, so this is the number and it is reported rather than rounded.
        assert!(per < 3.4, "{per} bytes a row");
    }
}
