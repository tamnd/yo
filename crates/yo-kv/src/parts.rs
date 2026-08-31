//! The partitioned band: one collection held as several element tables, with a
//! two level summary of their lengths in front so that finding the one that owns
//! a given position is not a walk over all of them.
//!
//! `05` section 4.3 is the shape. Above 262,144 elements a collection stops
//! being one [`Elements`] and becomes P of them, each an ordinary table, with a
//! member's partition picked out of its hash. Nothing about a partition is
//! special, which is the point: every path that already works on a table keeps
//! working, and what is new is only the arithmetic that says which table.
//!
//! ```text
//!  blocks   +--------------+--------------+  sums of 64 lens each
//!           |      b0      |      b1      |
//!           +--------------+--------------+
//!  lens     +--+--+--+--+--+--+--+--+--+--+  one count per partition
//!           |n0|n1|n2|n3|n4|n5|n6|n7|..|..|
//!           +--+--+--+--+--+--+--+--+--+--+
//!  tables   +-----+-----+-----+-----+-----+  one slot array and one
//!           | t0  | t1  | t2  | t3  | ... |  row array each
//!           +-----+-----+-----+-----+-----+
//! ```
//!
//! # Why partition at all
//!
//! Three reasons and none of them is the probe, which is already one load into
//! a slot array and does not care how large the array is.
//!
//! A merge is the first. `SINTER` and `SUNION` over a pair of million member
//! sets can walk per partition sorted arrays and merge them, which L12 measured
//! at 5.78 ms against 450 ms for sorting the whole thing per call. That work is
//! not here yet and this is the structure it needs.
//!
//! A growth is the second. One table at a million members doubles into a sixteen
//! megabyte slot array, and the doubling copies it and rehashes every row while
//! the shard is holding the key. P tables double one partition at a time, so the
//! pause is a P'th of the size.
//!
//! Reclaim is the third. The name blob compacts per partition, so the rewrite
//! that reclaims dead name bytes on a large collection is bounded rather than
//! proportional to the whole thing.
//!
//! # The descriptor cache, and what actually turned out to be slow
//!
//! `05` section 4.3 calls the descriptor cache mandatory and K10 is the number:
//! resolving a partition costs 9.9 to 11.3 ns with it and 177 to 275 ns without.
//! The operation it is talking about is the dense draw. `SRANDMEMBER` picks a
//! number under the total and has to work out which partition owns that
//! position, which means knowing how the elements are spread.
//!
//! The obvious reading of that is a locality story: the partitions are Vec
//! headers scattered a cache line apart, so reading P of their lengths is P
//! cache misses, and copying the lengths into one contiguous array fixes it.
//! That reading is wrong, and `benches/parts.rs` was written to check it before
//! it got baked in. An array of tables is itself contiguous, so walking their
//! lengths is a strided read the prefetcher handles, and a flat array of lengths
//! beside them measured no faster at any layout. Both were 26 ns at 64
//! partitions and both were past 300 ns at 1024.
//!
//! What is actually slow is the walk, not where it reads from. Adding P numbers
//! up is O(P) however tight the loop is, and at 1024 partitions that is a
//! thousand adds nobody can make free.
//!
//! So the cache here is two levels rather than one. The lower one is a
//! per-partition count, four bytes each and back to back, and the upper one is
//! the sum of every run of [`BLOCK`] of them. A resolve walks the blocks to
//! find the run and then the lengths inside it, which is `P/BLOCK + BLOCK`
//! numbers instead of P. `BLOCK` is 64, near the square root of the largest
//! layout this band allows, so the worst case is 96 reads over six cache lines
//! rather than 2,048 reads over 128. Measured, that is 33 ns at the ceiling
//! against 1.2 microseconds for the flat walk, and it barely moves across the
//! whole sweep: 3 ns at four partitions and 21 ns at a thousand.
//!
//! Both levels stay O(1) on the write path, which is the reason the counts are
//! per partition rather than cumulative. An insert is `lens[p] += 1`,
//! `blocks[p / BLOCK] += 1` and `total += 1`, where a prefix sum would have to
//! rewrite every entry above the one that changed. It is duplicated state and it
//! is duplicated on purpose, and the one rule is that every write goes through
//! the two methods that keep all three in step.
//!
//! # Which bits pick the partition
//!
//! Bits 32 and up, not the low ones. A table indexes its slot array with the low
//! bits of the same hash and tags a slot with the top byte, so partitioning on
//! the low bits would hand every member of a partition the same home slot and
//! turn the table into a linked list. Bits 32 and up are spoken for by nothing,
//! and a scan cursor can carry eleven of them, which is [`MAX_PARTS`].
//!
//! What matters for the cursor is not which bits they are but that growth reads
//! **one more** of them, so a partition splits in two and nothing moves between
//! any other pair. [`Cursor::rebase`] is that arithmetic and it was written and
//! tested before this module existed.
//!
//! # P is never 2
//!
//! L5, and it is counterintuitive enough to be worth a sentence rather than a
//! constant. Two partitions measured worse than one and worse than four, so the
//! floor is [`PART_MIN`] and there is no layout with two.

use yo_common::hash_key;

use crate::elem::{Elements, Full};
use crate::scan::{Cursor, MAX_PARTS};

/// Where a collection stops being one table and becomes several.
pub const PARTITION_AT: usize = 262_144;

/// The fewest partitions a partitioned collection has.
///
/// Four, because L5 measured two as the worst point of the whole sweep, worse
/// than not partitioning at all. Halving a table gives up the locality of one
/// and buys none of the parallelism of many.
pub const PART_MIN: u32 = 4;

/// How many elements a partition is aimed at holding.
///
/// The threshold divided by [`PART_MIN`], so that a collection arriving at the
/// band gets the floor and everything above it grows from there.
pub const PART_TARGET: usize = PARTITION_AT / PART_MIN as usize;

/// Where the partition number is taken from in a hash.
///
/// The low thirty two bits are a table's home slot and the top eight are its
/// slot tag. This is the first bit above the slot that is spoken for by nothing.
/// Public because anything that wants to sort or shard the same way a partition
/// does has to agree with this, and guessing at it is how two pieces of the same
/// structure end up disagreeing about where a member lives.
pub const PART_BIT: u32 = 32;

/// How many partitions one entry of the summary level covers.
///
/// Sixty four, which is about the square root of [`MAX_PARTS`] and so near the
/// fan out that makes the worst resolve as cheap as it can be: `P/BLOCK + BLOCK`
/// reads is smallest when `BLOCK` is the root of P. A power of two is worth more
/// than the last few reads, because it turns the divide on the write path into a
/// shift. It is also four cache lines of `u32`, which the prefetcher handles.
pub const BLOCK: usize = 64;

/// How many partitions a collection of `n` elements should have.
///
/// A power of two, at least [`PART_MIN`], at most [`MAX_PARTS`], aiming at
/// [`PART_TARGET`] elements each. A collection arriving at the band boundary
/// gets four partitions of 65,536, and it doubles from there.
#[must_use]
pub fn parts_for(n: usize) -> u32 {
    layout(u32::try_from(n.div_ceil(PART_TARGET)).unwrap_or(MAX_PARTS))
}

/// The nearest legal layout at or above `parts`.
///
/// The ceiling is checked before the rounding rather than after, because
/// rounding a number near the top of a `u32` up to a power of two overflows
/// rather than saturating.
const fn layout(parts: u32) -> u32 {
    if parts >= MAX_PARTS {
        return MAX_PARTS;
    }
    let want = parts.next_power_of_two();
    if want < PART_MIN { PART_MIN } else { want }
}

/// A collection held as several element tables.
#[derive(Debug, Clone)]
pub struct Parts<V> {
    /// The partitions, low to high. Always a power of two of them, never two.
    tables: Vec<Elements<V>>,
    /// How many elements each partition holds, back to back.
    ///
    /// The lower level of the descriptor cache. This is `tables[i].len()` kept a
    /// second time, and on its own it buys nothing, which the bench says out
    /// loud. It is here to be summarised.
    lens: Vec<u32>,
    /// How many elements each run of [`BLOCK`] partitions holds.
    ///
    /// The upper level, and the one that does the work. A resolve reads this to
    /// find the run and then reads the lower level inside it, which turns an
    /// O(P) walk into an O(sqrt P) one. Always `parts / BLOCK` entries, rounded
    /// up, so a layout at or under `BLOCK` has exactly one and the level costs a
    /// single compare.
    blocks: Vec<u32>,
    /// How many elements in total, so that `len` is not a sum.
    total: usize,
    /// One less than the number of partitions, for the partition select.
    mask: u32,
}

impl<V: Copy> Parts<V> {
    /// An empty collection with `parts` partitions.
    ///
    /// `parts` is rounded to a power of two inside [`PART_MIN`] and
    /// [`MAX_PARTS`], so a caller cannot ask for two of them.
    #[must_use]
    pub fn with_parts(parts: u32) -> Parts<V> {
        let parts = layout(parts);
        let n = parts as usize;
        Parts {
            tables: (0..n).map(|_| Elements::new()).collect(),
            lens: vec![0; n],
            blocks: vec![0; n.div_ceil(BLOCK)],
            total: 0,
            mask: parts - 1,
        }
    }

    /// Note that a partition gained an element.
    ///
    /// Both levels and the total, in one place, because the failure mode of a
    /// cache kept by hand is one write path that forgot a level.
    #[inline]
    fn gained(&mut self, at: usize) {
        self.lens[at] += 1;
        self.blocks[at / BLOCK] += 1;
        self.total += 1;
    }

    /// Note that a partition lost an element.
    #[inline]
    fn lost(&mut self, at: usize) {
        self.lens[at] -= 1;
        self.blocks[at / BLOCK] -= 1;
        self.total -= 1;
    }

    /// Spread one table's elements over `parts` partitions.
    ///
    /// This is the promotion, and it is the one place the whole collection is
    /// rehashed. It happens once per collection per partition count, on a write
    /// that has just taken the collection past a threshold, and the alternative
    /// is a layout that never partitions.
    #[must_use]
    pub fn from_table(table: &Elements<V>, parts: u32) -> Parts<V> {
        let mut p = Parts::with_parts(parts);
        for (name, value) in table.iter() {
            let h = hash_key(name);
            let at = p.part_of(h) as usize;
            // A name and a row count that one table accepted cannot be refused
            // by a smaller one, so this cannot fail.
            let _ = p.tables[at].insert_hashed(h, name, *value);
            p.gained(at);
        }
        p
    }

    /// How many partitions.
    #[inline]
    #[must_use]
    pub const fn parts(&self) -> u32 {
        self.mask + 1
    }

    /// Which partition a hash names.
    #[inline]
    const fn part_of(&self, h: u64) -> u32 {
        ((h >> PART_BIT) as u32) & self.mask
    }

    /// How many elements are here.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.total
    }

    /// Whether there are none.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// The hash of a name, for a caller about to ask several collections about
    /// it. The same function [`Elements::hash_of`] uses, deliberately, so one
    /// hash serves both bands.
    #[inline]
    #[must_use]
    pub fn hash_of(name: &[u8]) -> u64 {
        hash_key(name)
    }

    /// What is stored against this name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &[u8]) -> Option<&V> {
        self.get_hashed(hash_key(name), name)
    }

    /// What is stored against this name, with its hash already in hand.
    #[inline]
    #[must_use]
    pub fn get_hashed(&self, h: u64, name: &[u8]) -> Option<&V> {
        self.tables[self.part_of(h) as usize].get_hashed(h, name)
    }

    /// The payload, to be changed in place.
    #[inline]
    pub fn get_mut(&mut self, name: &[u8]) -> Option<&mut V> {
        let h = hash_key(name);
        let at = self.part_of(h) as usize;
        self.tables[at].get_hashed_mut(h, name)
    }

    /// Whether this name is here at all.
    #[inline]
    #[must_use]
    pub fn contains(&self, name: &[u8]) -> bool {
        self.get(name).is_some()
    }

    /// Whether this name is here, with its hash already in hand.
    #[inline]
    #[must_use]
    pub fn contains_hashed(&self, h: u64, name: &[u8]) -> bool {
        self.get_hashed(h, name).is_some()
    }

    /// Store `value` against `name`, and say what was there before.
    ///
    /// `None` means the element is new, the same as it does on one table.
    pub fn insert(&mut self, name: &[u8], value: V) -> Result<Option<V>, Full> {
        let h = hash_key(name);
        let at = self.part_of(h) as usize;
        let was = self.tables[at].insert_hashed(h, name, value)?;
        if was.is_none() {
            self.gained(at);
        }
        Ok(was)
    }

    /// Take an element out and hand back what it held.
    pub fn remove(&mut self, name: &[u8]) -> Option<V> {
        let h = hash_key(name);
        let at = self.part_of(h) as usize;
        let was = self.tables[at].remove_hashed(h, name)?;
        self.lost(at);
        Some(was)
    }

    /// Which partition holds the element at global position `idx`, and where in
    /// it, or `None` past the end.
    ///
    /// The descriptor cache walk, and the reason the cache exists. Partitions
    /// are laid end to end in partition order, so this is a running total, and
    /// the two levels are what stop it being a running total over all P of them.
    /// Neither loop can fall off the end once `idx` is known to be inside the
    /// collection, because the two levels are kept in step with the total.
    ///
    /// Public because a caller drawing several positions at once, which is what
    /// `SRANDMEMBER` with a count does, wants to resolve once and then read
    /// straight out of the partition.
    #[inline]
    #[must_use]
    pub fn locate(&self, idx: usize) -> Option<(usize, usize)> {
        if idx >= self.total {
            return None;
        }
        let mut seen = 0usize;
        let mut at = 0usize;
        for &n in &self.blocks {
            let n = n as usize;
            if idx < seen + n {
                break;
            }
            seen += n;
            at += BLOCK;
        }
        for &n in &self.lens[at..] {
            let n = n as usize;
            if idx < seen + n {
                return Some((at, idx - seen));
            }
            seen += n;
            at += 1;
        }
        None
    }

    /// The name and payload at a global position. The dense draw.
    #[inline]
    #[must_use]
    pub fn at(&self, idx: usize) -> Option<(&[u8], &V)> {
        let (at, within) = self.locate(idx)?;
        self.tables[at].at(within)
    }

    /// The payload at a global position, to be written over.
    #[inline]
    pub fn at_mut(&mut self, idx: usize) -> Option<&mut V> {
        let (at, within) = self.locate(idx)?;
        self.tables[at].at_mut(within)
    }

    /// Take the element at a global position out.
    pub fn remove_at(&mut self, idx: usize) -> Option<V> {
        let (at, within) = self.locate(idx)?;
        let was = self.tables[at].remove_at(within)?;
        self.lost(at);
        Some(was)
    }

    /// Take the element at a global position out and hand back its name too.
    pub fn take_at(&mut self, idx: usize) -> Option<(Vec<u8>, V)> {
        let (at, within) = self.locate(idx)?;
        let got = self.tables[at].take_at(within)?;
        self.lost(at);
        Some(got)
    }

    /// Every element, partition by partition.
    ///
    /// The order is not insertion order across the collection, only within a
    /// partition, and no command promises one. `SMEMBERS` and `HGETALL` are
    /// unordered in Redis and a client that depends on the order is already
    /// wrong on a table that has had one removal.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &V)> {
        self.tables.iter().flat_map(Elements::iter)
    }

    /// Every payload, to be changed in place.
    pub fn payloads_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.tables.iter_mut().flat_map(Elements::payloads_mut)
    }

    /// Walk part of the collection and say where to resume.
    ///
    /// Partitions go from the top down and rows inside a partition go from the
    /// top down, which is the same downward rule one table follows and holds the
    /// scan guarantee for the same reason: nothing ever moves an element up past
    /// the cursor. `crate::scan` has the argument.
    ///
    /// A cursor issued under a different partition count is rebased first, which
    /// is what makes a scan survive the collection doubling underneath it.
    pub fn scan<F>(&self, cursor: Cursor, count: usize, mut f: F) -> Cursor
    where
        F: FnMut(&[u8], &V),
    {
        if self.total == 0 {
            return Cursor::END;
        }
        let parts = self.parts();
        // A fresh cursor is zero, and zero is also the end, so the start has to
        // be read as the top of the highest partition rather than run through
        // the rebase.
        let (mut part, mut idx) = if cursor.is_end() {
            (parts - 1, None)
        } else {
            let here = cursor.rebase(parts);
            (here.part().min(parts - 1), here.idx())
        };
        let mut left = count.max(1);
        loop {
            let table = &self.tables[part as usize];
            let mut at = if table.is_empty() {
                None
            } else {
                let top = table.len() - 1;
                // A cursor from before a run of removals can name a row that is
                // gone. Everything above the end has been walked already.
                Some(match idx {
                    Some(i) => (i as usize).min(top),
                    None => top,
                })
            };
            while let Some(row) = at {
                if left == 0 {
                    return Cursor::at(parts, part, row as u64);
                }
                let (name, value) = table.at(row).expect("the row is inside the table");
                f(name, value);
                left -= 1;
                at = row.checked_sub(1);
            }
            if part == 0 {
                return Cursor::END;
            }
            part -= 1;
            idx = None;
            if left == 0 {
                return Cursor::top(parts, part);
            }
        }
    }

    /// Double the partition count, or more.
    ///
    /// Every partition splits by one further bit of the hash, so a member either
    /// stays where it is or moves to exactly one new partition, and no member
    /// crosses between two old partitions. That is the property
    /// [`Cursor::rebase`] leans on, and it is why the split reads a higher bit
    /// rather than rehashing into a different space.
    ///
    /// Answers whether anything happened, which is `false` once the layout is at
    /// [`MAX_PARTS`].
    pub fn grow_to(&mut self, parts: u32) -> bool {
        let parts = layout(parts);
        if parts <= self.parts() {
            return false;
        }
        let mut grown = Parts::with_parts(parts);
        for table in &self.tables {
            for (name, value) in table.iter() {
                let h = hash_key(name);
                let at = grown.part_of(h) as usize;
                let _ = grown.tables[at].insert_hashed(h, name, *value);
                grown.gained(at);
            }
        }
        *self = grown;
        true
    }

    /// Whether the collection has outgrown its partition count, and what it
    /// should be if so.
    ///
    /// The caller asks after a write rather than this deciding for itself,
    /// because growing is a rehash of everything and the caller is the one that
    /// knows whether it is in the middle of a bulk load.
    #[must_use]
    pub fn wants_parts(&self) -> Option<u32> {
        let want = parts_for(self.total);
        (want > self.parts()).then_some(want)
    }

    /// Throw everything away and keep the allocations and the layout.
    pub fn clear(&mut self) {
        for table in &mut self.tables {
            table.clear();
        }
        self.lens.fill(0);
        self.blocks.fill(0);
        self.total = 0;
    }

    /// What this collection costs, not counting anything a payload points at.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.tables
            .iter()
            .map(Elements::memory_bytes)
            .sum::<usize>()
            + (self.lens.len() + self.blocks.len()) * size_of::<u32>()
    }

    /// What the slot arrays cost, added up.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.tables.iter().map(Elements::slot_bytes).sum()
    }

    /// What the row arrays cost, added up.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.tables.iter().map(Elements::row_bytes).sum()
    }

    /// What the name blobs cost, added up.
    #[must_use]
    pub fn name_bytes(&self) -> usize {
        self.tables.iter().map(Elements::name_bytes).sum()
    }

    /// Name bytes no row points at any more, added up.
    #[must_use]
    pub fn dead_name_bytes(&self) -> usize {
        self.tables.iter().map(Elements::dead_name_bytes).sum()
    }

    /// How many elements each partition holds, for a test or a measurement.
    #[must_use]
    pub fn lengths(&self) -> &[u32] {
        &self.lens
    }

    /// How many elements each run of [`BLOCK`] partitions holds, for a test.
    #[must_use]
    pub fn block_lengths(&self) -> &[u32] {
        &self.blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A collection of `n` members named the way a benchmark would name them.
    fn filled(n: usize, parts: u32) -> Parts<u32> {
        let mut p = Parts::with_parts(parts);
        for i in 0..n {
            p.insert(format!("member:{i}").as_bytes(), i as u32)
                .expect("room");
        }
        p
    }

    #[test]
    fn the_layout_is_a_power_of_two_and_never_two() {
        assert_eq!(Parts::<()>::with_parts(0).parts(), PART_MIN);
        assert_eq!(Parts::<()>::with_parts(1).parts(), PART_MIN);
        assert_eq!(Parts::<()>::with_parts(2).parts(), PART_MIN);
        assert_eq!(Parts::<()>::with_parts(3).parts(), PART_MIN);
        assert_eq!(Parts::<()>::with_parts(4).parts(), 4);
        assert_eq!(Parts::<()>::with_parts(5).parts(), 8);
        assert_eq!(Parts::<()>::with_parts(u32::MAX).parts(), MAX_PARTS);
        for n in [0, 1, 100, PARTITION_AT - 1, PARTITION_AT] {
            assert_ne!(parts_for(n), 2, "there is no layout with two partitions");
        }
        assert_eq!(parts_for(PARTITION_AT), PART_MIN);
        assert_eq!(parts_for(PARTITION_AT + 1), 8);
        assert_eq!(parts_for(1_000_000), 16);
    }

    #[test]
    fn what_goes_in_comes_out_of_the_partition_it_went_into() {
        let p = filled(2_000, 8);
        assert_eq!(p.len(), 2_000);
        for i in 0..2_000 {
            let name = format!("member:{i}");
            assert_eq!(p.get(name.as_bytes()), Some(&(i as u32)));
        }
        assert!(!p.contains(b"member:2000"));
        assert_eq!(p.get(b"nothing"), None);
    }

    #[test]
    fn the_lengths_add_up_to_the_total_and_the_spread_is_even() {
        let p = filled(8_000, 8);
        assert_eq!(p.lengths().len(), 8);
        assert_eq!(
            p.lengths().iter().map(|&n| n as usize).sum::<usize>(),
            8_000
        );
        // A thousand each on average. A hash that put everything in one
        // partition would still pass every correctness test here, so the spread
        // is asserted rather than assumed.
        for &n in p.lengths() {
            assert!((700..1_300).contains(&n), "one partition holds {n}");
        }
    }

    #[test]
    fn writing_a_member_again_replaces_it_and_moves_no_length() {
        let mut p = filled(100, 4);
        let before = p.lengths().to_vec();
        assert_eq!(p.insert(b"member:7", 700), Ok(Some(7)));
        assert_eq!(p.len(), 100);
        assert_eq!(p.lengths(), before.as_slice());
        assert_eq!(p.get(b"member:7"), Some(&700));
    }

    #[test]
    fn removing_takes_it_out_of_one_partition_only() {
        let mut p = filled(1_000, 4);
        let before = p.lengths().to_vec();
        assert_eq!(p.remove(b"member:500"), Some(500));
        assert_eq!(p.len(), 999);
        assert!(!p.contains(b"member:500"));
        assert_eq!(p.remove(b"member:500"), None);
        let after = p.lengths();
        let moved: Vec<usize> = (0..4).filter(|&i| after[i] != before[i]).collect();
        assert_eq!(moved.len(), 1, "one partition changed length");
    }

    #[test]
    fn a_draw_reaches_every_member_exactly_once() {
        let p = filled(1_000, 8);
        let mut seen = vec![false; 1_000];
        for i in 0..p.len() {
            let (_, &v) = p.at(i).expect("inside the collection");
            assert!(!seen[v as usize], "position {i} handed back a repeat");
            seen[v as usize] = true;
        }
        assert!(seen.into_iter().all(|s| s));
        assert!(p.at(1_000).is_none());
    }

    #[test]
    fn a_collection_drained_one_draw_at_a_time_stays_correct() {
        let mut p = filled(500, 4);
        let mut seen = Vec::new();
        while !p.is_empty() {
            let n = p.len();
            // Always the last position, which is the partition boundary case:
            // the draw has to land in the highest non empty partition.
            let (name, _) = p.at(n - 1).expect("inside");
            seen.push(name.to_vec());
            assert!(p.remove_at(n - 1).is_some());
            assert_eq!(p.len(), n - 1);
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 500, "every member came out once");
        assert_eq!(p.lengths().iter().sum::<u32>(), 0);
    }

    /// The summary level is state kept by hand, so the thing worth testing is
    /// that no write path forgets it. Every method that moves an element is
    /// exercised and the two levels are checked against each other and against
    /// the partitions themselves after each one.
    #[test]
    fn both_levels_of_the_cache_agree_with_the_partitions() {
        fn agrees(p: &Parts<u32>) {
            for (at, table) in p.tables.iter().enumerate() {
                assert_eq!(p.lengths()[at] as usize, table.len(), "partition {at}");
            }
            for (b, &sum) in p.block_lengths().iter().enumerate() {
                let run = &p.lengths()[b * BLOCK..((b + 1) * BLOCK).min(p.lengths().len())];
                assert_eq!(sum, run.iter().sum::<u32>(), "block {b}");
            }
            assert_eq!(
                p.len(),
                p.block_lengths().iter().map(|&n| n as usize).sum::<usize>()
            );
        }

        let mut p = filled(300, 128);
        assert_eq!(p.block_lengths().len(), 2, "128 partitions is two blocks");
        agrees(&p);
        p.insert(b"fresh", 1).expect("room");
        agrees(&p);
        p.insert(b"fresh", 2).expect("room");
        agrees(&p);
        p.remove(b"fresh");
        agrees(&p);
        p.remove_at(0);
        agrees(&p);
        p.take_at(p.len() - 1);
        agrees(&p);
        p.grow_to(256);
        agrees(&p);
        p.clear();
        agrees(&p);
    }

    /// A layout at or under the fan out has one summary entry, which makes the
    /// upper level a single compare rather than a special case in the code.
    #[test]
    fn a_small_layout_has_exactly_one_block() {
        for parts in [PART_MIN, 8, 32, BLOCK as u32] {
            let p: Parts<()> = Parts::with_parts(parts);
            assert_eq!(p.block_lengths().len(), 1, "{parts} partitions");
        }
        let p: Parts<()> = Parts::with_parts(MAX_PARTS);
        assert_eq!(p.block_lengths().len(), MAX_PARTS as usize / BLOCK);
    }

    /// The resolve has two loops and either one falling off the end is a wrong
    /// answer rather than a panic, so every position is checked against the
    /// straightforward O(P) walk it replaces.
    #[test]
    fn the_two_level_resolve_agrees_with_the_flat_one() {
        let p = filled(5_000, 128);
        for idx in 0..p.len() {
            let mut seen = 0usize;
            let flat = p
                .lengths()
                .iter()
                .enumerate()
                .find_map(|(at, &n)| {
                    let n = n as usize;
                    if idx < seen + n {
                        Some((at, idx - seen))
                    } else {
                        seen += n;
                        None
                    }
                })
                .expect("inside the collection");
            assert_eq!(p.locate(idx), Some(flat), "position {idx}");
        }
        assert_eq!(p.locate(p.len()), None);
        assert_eq!(p.locate(usize::MAX), None);
    }

    #[test]
    fn taking_by_position_hands_back_the_name() {
        let mut p = filled(50, 4);
        let (name, value) = p.take_at(0).expect("inside");
        assert_eq!(p.len(), 49);
        assert!(!p.contains(&name));
        assert_eq!(p.get(&name), None);
        assert!(value < 50);
    }

    #[test]
    fn a_walk_sees_everything_once() {
        let p = filled(3_000, 8);
        let mut seen: Vec<u32> = p.iter().map(|(_, &v)| v).collect();
        assert_eq!(seen.len(), 3_000);
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 3_000);
    }

    #[test]
    fn payloads_can_be_rewritten_in_place() {
        let mut p = filled(200, 4);
        for v in p.payloads_mut() {
            *v += 1;
        }
        assert_eq!(p.get(b"member:0"), Some(&1));
        assert_eq!(p.get(b"member:199"), Some(&200));
        *p.at_mut(0).expect("inside") = 12345;
        assert_eq!(p.iter().filter(|&(_, &v)| v == 12345).count(), 1);
    }

    /// Every page size, because the interesting bugs are all at the seam where
    /// one partition runs out and the next one starts.
    #[test]
    fn a_scan_sees_everything_once_at_every_page_size() {
        let p = filled(1_000, 8);
        for page in [1, 2, 7, 64, 999, 1_000, 5_000] {
            let mut seen = Vec::new();
            let mut cursor = Cursor::START;
            let mut rounds = 0;
            loop {
                cursor = p.scan(cursor, page, |_, &v| seen.push(v));
                rounds += 1;
                assert!(rounds < 10_000, "the scan is not finishing");
                if cursor.is_end() {
                    break;
                }
            }
            seen.sort_unstable();
            let before = seen.len();
            seen.dedup();
            assert_eq!(before, 1_000, "page {page} returned {before}");
            assert_eq!(seen.len(), 1_000, "page {page} repeated a member");
        }
    }

    /// The regression for a bug the bench found and no test did. The scan cursor
    /// carries the partition count in twelve bits and zero is taken, so the
    /// largest layout it can name is 2,048 and not 4,096. A collection laid out
    /// on a count the cursor cannot hold has every cursor come back clamped,
    /// [`Cursor::rebase`] reads the clamp as a growth, and the scan walks back up
    /// instead of down and never finishes.
    #[test]
    fn a_scan_finishes_at_the_largest_layout() {
        let p = filled(8_000, MAX_PARTS);
        assert_eq!(p.parts(), MAX_PARTS);
        assert_eq!(
            Cursor::at(p.parts(), p.parts() - 1, 0).parts(),
            p.parts(),
            "a cursor has to be able to name the layout it was issued under"
        );
        let mut seen = Vec::new();
        let mut cursor = Cursor::START;
        let mut rounds = 0;
        loop {
            cursor = p.scan(cursor, 10, |_, &v| seen.push(v));
            rounds += 1;
            assert!(rounds < 20_000, "the scan is not finishing");
            if cursor.is_end() {
                break;
            }
        }
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, 8_000);
        assert_eq!(seen.len(), 8_000);
    }

    #[test]
    fn a_scan_of_an_empty_collection_is_over_at_once() {
        let p: Parts<u32> = Parts::with_parts(8);
        let mut hits = 0;
        assert!(p.scan(Cursor::START, 10, |_, _| hits += 1).is_end());
        assert_eq!(hits, 0);
    }

    #[test]
    fn a_page_of_zero_still_makes_progress() {
        let p = filled(4, 4);
        let mut hits = 0;
        let cursor = p.scan(Cursor::START, 0, |_, _| hits += 1);
        assert_eq!(hits, 1, "a zero page is read as one, not as none");
        assert!(!cursor.is_end() || p.len() == 1);
    }

    /// The guarantee that makes the cursor format worth carrying a partition
    /// count: every member that was there for the whole scan comes back, even
    /// though the collection doubled its layout halfway through.
    #[test]
    fn a_scan_survives_the_collection_growing_underneath_it() {
        let mut p = filled(4_000, 4);
        let mut seen = Vec::new();
        let mut cursor = p.scan(Cursor::START, 700, |_, &v| seen.push(v));
        assert!(!cursor.is_end());
        assert!(p.grow_to(16));
        assert_eq!(p.parts(), 16);
        assert_eq!(p.len(), 4_000);
        let mut rounds = 0;
        loop {
            cursor = p.scan(cursor, 700, |_, &v| seen.push(v));
            rounds += 1;
            assert!(rounds < 1_000, "the scan is not finishing");
            if cursor.is_end() {
                break;
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 4_000, "the growth lost a member");
    }

    /// The other half of the same guarantee. Members removed during a scan may
    /// or may not come back, but every member left at the end must.
    #[test]
    fn a_scan_under_removals_returns_everything_that_stayed() {
        let mut p = filled(2_000, 8);
        let mut seen = Vec::new();
        let mut cursor = p.scan(Cursor::START, 300, |_, &v| seen.push(v));
        for i in 0..500 {
            p.remove(format!("member:{i}").as_bytes());
        }
        let mut rounds = 0;
        loop {
            cursor = p.scan(cursor, 300, |_, &v| seen.push(v));
            rounds += 1;
            assert!(rounds < 1_000, "the scan is not finishing");
            if cursor.is_end() {
                break;
            }
        }
        seen.sort_unstable();
        seen.dedup();
        for v in 500..2_000u32 {
            assert!(
                seen.contains(&v),
                "member:{v} stayed and was never returned"
            );
        }
    }

    #[test]
    fn growing_splits_a_partition_and_moves_nothing_else() {
        let mut p = filled(4_000, 4);
        // Where every member sat before, by partition.
        let before: Vec<(Vec<u8>, usize)> = (0..4)
            .flat_map(|at| {
                p.tables[at]
                    .iter()
                    .map(move |(n, _)| (n.to_vec(), at))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(p.grow_to(8));
        for (name, was) in before {
            let h = hash_key(&name);
            let now = p.part_of(h) as usize;
            assert!(
                now == was || now == was + 4,
                "{name:?} moved from {was} to {now}, which is not a split"
            );
        }
        assert_eq!(p.len(), 4_000);
        assert_eq!(
            p.lengths().iter().map(|&n| n as usize).sum::<usize>(),
            4_000
        );
    }

    #[test]
    fn growing_stops_at_the_ceiling_and_never_shrinks() {
        let mut p: Parts<()> = Parts::with_parts(8);
        assert!(!p.grow_to(4), "a smaller layout is not a growth");
        assert!(!p.grow_to(8), "the same layout is not a growth");
        assert!(p.grow_to(MAX_PARTS));
        assert_eq!(p.parts(), MAX_PARTS);
        assert!(!p.grow_to(MAX_PARTS), "there is nowhere above the ceiling");
    }

    #[test]
    fn the_layout_is_asked_for_rather_than_decided() {
        let mut p: Parts<u32> = Parts::with_parts(PART_MIN);
        assert_eq!(p.wants_parts(), None);
        for i in 0..PART_TARGET * 5 {
            p.insert(format!("m{i}").as_bytes(), i as u32)
                .expect("room");
        }
        assert_eq!(p.wants_parts(), Some(8));
        assert!(p.grow_to(8));
        assert_eq!(p.wants_parts(), None);
    }

    #[test]
    fn promotion_carries_every_element_across() {
        let mut one = Elements::<u32>::new();
        for i in 0..5_000u32 {
            one.insert(format!("member:{i}").as_bytes(), i)
                .expect("room");
        }
        let p = Parts::from_table(&one, 8);
        assert_eq!(p.len(), 5_000);
        assert_eq!(p.parts(), 8);
        for i in 0..5_000u32 {
            assert_eq!(p.get(format!("member:{i}").as_bytes()), Some(&i));
        }
        assert_eq!(
            p.lengths().iter().map(|&n| n as usize).sum::<usize>(),
            5_000
        );
    }

    #[test]
    fn clearing_keeps_the_layout_and_forgets_the_elements() {
        let mut p = filled(1_000, 8);
        let bytes = p.memory_bytes();
        p.clear();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.parts(), 8);
        assert!(p.lengths().iter().all(|&n| n == 0));
        assert!(p.at(0).is_none());
        assert!(p.memory_bytes() <= bytes, "clearing does not allocate");
        p.insert(b"back", 1).expect("room");
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn the_memory_accounting_adds_the_partitions_up() {
        let p = filled(2_000, 8);
        assert_eq!(
            p.memory_bytes(),
            p.slot_bytes() + p.row_bytes() + p.name_bytes() + 9 * size_of::<u32>()
        );
        assert_eq!(p.dead_name_bytes(), 0);
        assert!(p.name_bytes() > 0);
    }
}
