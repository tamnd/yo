//! The scan cursor, and what it has to survive.
//!
//! `SCAN`, `SSCAN`, `HSCAN` and `ZSCAN` all hand the client an opaque number and
//! promise something quite specific about what happens when it comes back. Every
//! element that was there for the whole scan is returned at least once. An
//! element that arrived or left during the scan may or may not be. The same
//! element may be returned more than once, and the client is expected to cope.
//!
//! Redis buys that guarantee with reverse binary iteration over a bucket array,
//! because its table rehashes under the scan and buckets split. Ours is a
//! different structure and needs a different trick, which is K9's downward
//! cursor, `((P << 52) | (part << 40) | (idx + 1))`.
//!
//! # Why downward
//!
//! A collection is a dense array of rows in insertion order. Two things move a
//! row: an insert appends at the top, and a removal moves the top row down into
//! the hole it made. Nothing else moves anything.
//!
//! Walk that array downward and both of those are harmless. An insert lands
//! above the cursor, in the part already walked, so a member added during the
//! scan is simply not returned, which is allowed. A removal moves the top row,
//! which is also above the cursor and so already returned, down into the hole.
//! If the hole is below the cursor that member is returned a second time, which
//! is allowed. What cannot happen is a member below the cursor being lifted above
//! it, because nothing ever moves a row upward, and that is exactly the case the
//! guarantee forbids.
//!
//! Walking upward has none of that. A removal at the top would drop an unvisited
//! member into a visited position and it would never be returned, which is the
//! bug the guarantee exists to rule out.
//!
//! # Why the partition count is in the cursor
//!
//! Above 262,144 elements a collection is partitioned (`05` §4.3), and it can
//! gain partitions while a client is halfway through scanning it. A member lives
//! in the partition its hash's low bits name, so growing from `P` to `2P` splits
//! each old partition in two and moves nothing else. Carrying `P` in the cursor
//! is what lets the resume work out which of the new partitions the client has
//! already been through: everything whose low `log2(P)` bits are above the
//! partition it stopped in. See [`Cursor::rebase`], which is the whole of that
//! arithmetic and is written and tested here even though the partitioned band
//! itself lands later, because a cursor format that gets this wrong is a wire
//! format that has already shipped.

/// Where a scan stopped, as the client sees it.
///
/// Opaque to the client, and deliberately so, but not opaque in here: it is a
/// partition count, a partition, and a row index, packed the way `08` §4 names
/// them.
///
/// ```text
///  63    52 51    40 39                                   0
/// +--------+--------+--------------------------------------+
/// |   P    |  part  |               idx + 1                |
/// +--------+--------+--------------------------------------+
/// ```
///
/// Zero is both the start and the end, which is Redis's convention and is
/// unambiguous here because a real cursor always names a partition count and a
/// partition count is never zero.
///
/// An `idx + 1` of zero is not a row, it means the top of that partition,
/// whatever its length turns out to be. A resume needs to be able to say that
/// without knowing how long the partition is, because [`Cursor::rebase`] moves a
/// cursor into a partition it has never looked at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Cursor(u64);

/// Bits 40 to 51, the partition.
const PART_SHIFT: u32 = 40;
/// Bits 52 to 63, the partition count the cursor was issued under.
const PARTS_SHIFT: u32 = 52;
/// Twelve bits each for the count and the partition.
const PART_MASK: u64 = 0xFFF;
/// Forty bits for the row index, which is more than any one partition holds.
const IDX_MASK: u64 = (1 << PART_SHIFT) - 1;

/// The most partitions a cursor can name.
///
/// Twelve bits. The partitioned band starts at 262,144 elements and `P` is a
/// power of two that is never 2, so a collection would have to be enormous
/// before this mattered, and a cursor is a wire format where a spare bit is
/// worth more than a partition count nobody will reach.
pub const MAX_PARTS: u32 = PART_MASK as u32 + 1;

impl Cursor {
    /// Start at the beginning, which for a downward walk is the top.
    pub const START: Cursor = Cursor(0);

    /// Nothing left. The same value as [`Cursor::START`], which is what the
    /// protocol says and what every Redis client already loops on.
    pub const END: Cursor = Cursor(0);

    /// A cursor as the client sent it back.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: u64) -> Cursor {
        Cursor(raw)
    }

    /// The number to put on the wire.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Whether the scan is over.
    #[inline]
    #[must_use]
    pub const fn is_end(self) -> bool {
        self.0 == 0
    }

    /// Resume at a row.
    ///
    /// `parts` and `part` are clamped rather than rejected. A client can send
    /// any number back and Redis answers all of them, so a cursor that names a
    /// partition that does not exist has to mean something sane rather than be
    /// an error.
    #[must_use]
    pub const fn at(parts: u32, part: u32, idx: u64) -> Cursor {
        Cursor(pack(parts, part, (idx + 1) & IDX_MASK))
    }

    /// Resume at the top of a partition, without saying how long it is.
    #[must_use]
    pub const fn top(parts: u32, part: u32) -> Cursor {
        Cursor(pack(parts, part, 0))
    }

    /// How many partitions the collection had when this was issued.
    ///
    /// One for a cursor that has not been anywhere yet, which is also the
    /// truth for every collection below the partitioned band.
    #[inline]
    #[must_use]
    pub const fn parts(self) -> u32 {
        let p = ((self.0 >> PARTS_SHIFT) & PART_MASK) as u32;
        if p == 0 { 1 } else { p }
    }

    /// Which partition it stopped in.
    #[inline]
    #[must_use]
    pub const fn part(self) -> u32 {
        ((self.0 >> PART_SHIFT) & PART_MASK) as u32
    }

    /// The next row to read, or `None` for the top of the partition.
    #[inline]
    #[must_use]
    pub const fn idx(self) -> Option<u64> {
        let i = self.0 & IDX_MASK;
        if i == 0 { None } else { Some(i - 1) }
    }

    /// Move a cursor into the layout the collection has now.
    ///
    /// Growing from `P` to some larger power of two splits every partition and
    /// moves nothing between the halves, because a member's partition is the low
    /// bits of its hash and growing only reads more of them. So an old partition
    /// `part` becomes the new partitions whose low `log2(P)` bits are `part`, and
    /// every new partition whose low bits are above `part` has already been
    /// walked in full.
    ///
    /// Resuming at the top of the highest new partition with those low bits
    /// covers all of the work that is left, and walking down from there also
    /// passes back over some partitions that were already done. That is
    /// duplicate work and duplicates are allowed. What it never does is skip
    /// one, and it never restarts the whole scan either, which is the other easy
    /// answer and the one that turns a growth into a full second pass.
    ///
    /// The row index is dropped rather than carried across. A split redistributes
    /// the rows, so an index into the old partition's array names a different
    /// member in the new one, and resuming at the top of the partition it stopped
    /// in is the only thing that can be said honestly.
    ///
    /// Shrinking is the other direction and is not something the size ladder
    /// does under a live scan, so a cursor from a larger layout is answered by
    /// starting the current one at the top. A repeat is allowed. A miss is not.
    #[must_use]
    pub const fn rebase(self, parts_now: u32) -> Cursor {
        let was = self.parts();
        if was == parts_now || self.is_end() {
            return self;
        }
        if parts_now < was {
            return Cursor::top(parts_now, parts_now.saturating_sub(1));
        }
        Cursor::top(parts_now, self.part() + (parts_now - was))
    }
}

/// Pack the three fields, clamping the two that come from outside.
const fn pack(parts: u32, part: u32, idx_plus_one: u64) -> u64 {
    let parts = if parts as u64 > PART_MASK {
        PART_MASK
    } else {
        parts as u64
    };
    let part = if part as u64 > PART_MASK {
        PART_MASK
    } else {
        part as u64
    };
    (parts << PARTS_SHIFT) | (part << PART_SHIFT) | idx_plus_one
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_the_start_and_the_end() {
        assert!(Cursor::START.is_end());
        assert_eq!(Cursor::START, Cursor::END);
        assert_eq!(Cursor::START.parts(), 1);
        assert_eq!(Cursor::START.part(), 0);
        assert_eq!(Cursor::START.idx(), None);
    }

    #[test]
    fn the_three_fields_survive_the_wire() {
        let c = Cursor::at(8, 5, 1234);
        let back = Cursor::from_raw(c.raw());
        assert_eq!(back.parts(), 8);
        assert_eq!(back.part(), 5);
        assert_eq!(back.idx(), Some(1234));
        assert!(!back.is_end());
    }

    /// The layout is the one `08` section 4 writes down, and it is a wire format
    /// once a client has a cursor in its hand, so it is pinned here.
    #[test]
    fn the_layout_is_the_one_the_spec_names() {
        let c = Cursor::at(4, 3, 41);
        assert_eq!(c.raw(), (4 << 52) | (3 << 40) | 42);
    }

    #[test]
    fn the_top_of_a_partition_has_no_row_yet() {
        let c = Cursor::top(16, 9);
        assert_eq!(c.parts(), 16);
        assert_eq!(c.part(), 9);
        assert_eq!(c.idx(), None);
        assert!(!c.is_end(), "the top of a partition is not the end");
    }

    /// The point of carrying the partition count. A scan that stopped in
    /// partition 2 of 4 has already been through 3 of 4, so in a layout of 8 it
    /// has been through everything whose low two bits are 3, and resuming at 6
    /// covers the rest.
    #[test]
    fn growing_the_partitions_does_not_skip_any() {
        let stopped = Cursor::at(4, 2, 500);
        let now = stopped.rebase(8);
        assert_eq!(now.parts(), 8);
        assert_eq!(now.part(), 6);
        assert_eq!(
            now.idx(),
            None,
            "the split moved the rows, so start at the top"
        );

        // Every new partition the old cursor had not finished is at or below the
        // resume point, so walking down from there reaches all of them. Some
        // that were finished are below it too, and those are walked a second
        // time, which is the trade and is allowed.
        for n in 0..8u32 {
            let done = (n & 3) > 2;
            assert!(
                done || n <= now.part(),
                "new partition {n} would be skipped"
            );
        }
    }

    #[test]
    fn a_cursor_from_the_same_layout_is_left_alone() {
        let c = Cursor::at(8, 5, 77);
        assert_eq!(c.rebase(8), c);
        assert_eq!(Cursor::END.rebase(64), Cursor::END);
    }

    /// Not a case the ladder produces, but a client can send anything, and the
    /// answer has to be a repeat rather than a miss.
    #[test]
    fn a_cursor_from_a_bigger_layout_starts_again_at_the_top() {
        let c = Cursor::at(64, 40, 9).rebase(4);
        assert_eq!(c.parts(), 4);
        assert_eq!(c.part(), 3);
        assert_eq!(c.idx(), None);
    }

    #[test]
    fn a_partition_count_past_the_field_is_clamped_and_not_wrapped() {
        let c = Cursor::at(MAX_PARTS * 4, MAX_PARTS * 4, 1);
        assert!(c.parts() <= MAX_PARTS);
        assert!(c.part() < MAX_PARTS);
        assert_eq!(c.idx(), Some(1));
    }
}
