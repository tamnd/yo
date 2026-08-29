//! Walking the whole index while it is being written to.
//!
//! `KEYS` can walk the index in one go and be done with it. `SCAN` cannot: it
//! hands the client a number, the client goes away and does something else, and
//! then it comes back and expects the walk to carry on. Between those two calls
//! the index may have doubled its directory and split any number of segments,
//! and the promise `SCAN` makes has to survive all of it.
//!
//! The promise is one sided and worth stating exactly, because half of what
//! makes it implementable is what it does not say. A key that is there for the
//! whole walk is returned at least once. A key added or removed partway through
//! may or may not appear. A key may appear twice. That is Redis's contract and
//! it is the contract here.
//!
//! # The prefix does not move
//!
//! Redis walks a power of two table and doubles it by adding a bit at the top of
//! the bucket index, so a bucket that was `i` becomes `i` and `i + n`. That is
//! why its cursor counts in reverse binary: it is the only order in which the
//! two halves of a split bucket stay next to each other.
//!
//! This index doubles the other way round. [`Index::dir_index`] takes the top
//! `global_depth` bits below the tag, and doubling the directory copies each
//! entry to two neighbouring slots, so a directory index `d` becomes `2d` and
//! `2d + 1`. A bit is added at the bottom, not the top.
//!
//! That makes the cursor simple, because there is a number that does not move at
//! all. Take the full 48 bits the directory could ever use, left aligned, and
//! call it the prefix. It is a function of the key's hash and nothing else, so
//! it is the same number before a doubling and after one, and the directory
//! index at any depth is just the top `global_depth` bits of it. Walk in
//! increasing prefix order and the boundary between what has been seen and what
//! has not is a number that means the same thing in every version of the index
//! this walk will ever see.
//!
//! # What a split does
//!
//! A segment covers a contiguous run of prefixes. Splitting it cuts that run in
//! half and gives the top half to a new segment, which is a change to where the
//! keys live and not to any key's prefix.
//!
//! If the cursor is partway through a segment when it splits, the walk resumes
//! in the half that still holds the cursor's prefix, finishes it, and then
//! starts the other half from its first bucket. Keys in the top half that had
//! already been returned are returned again. That is the duplicate the contract
//! allows, and it is the price of never having to stop the world.
//!
//! # The shape of the number
//!
//! ```text
//!  63                              16 15      6 5        0
//! +----------------------------------+---------+----------+
//! |         directory prefix         |    0    |  bucket  |
//! +----------------------------------+---------+----------+
//! ```
//!
//! The bucket within a segment comes off the bottom of the hash and the prefix
//! comes off the top, so the two never overlap and a split cannot move a key
//! from one bucket to another. The ten bits in the middle are spare. They are
//! not padding for its own sake: a segment is 64 buckets today and the day it is
//! not, the field grows into them without the cursors clients are holding
//! meaning something different.
//!
//! Zero is both the start and the end, which is Redis's convention and is not an
//! ambiguity in practice: a walk that has finished says zero, and a client that
//! says zero is starting a new one.

/// How far a scan has got, and the number the client holds between calls.
///
/// It is a position in the keyspace and not a position in memory. Two calls a
/// week apart with the same cursor resume at the same place, even if every
/// segment in the index has split in between.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cursor(u64);

/// Bits the directory can ever use, which is the index's `MAX_DEPTH`.
pub(crate) const PREFIX_BITS: u32 = 48;

/// Where the prefix sits in the cursor, leaving room below it for the bucket.
pub(crate) const PREFIX_SHIFT: u32 = 16;

/// Bits of bucket index, which is `log2` of [`SEGMENT_BUCKETS`](super::SEGMENT_BUCKETS).
pub(crate) const BUCKET_BITS: u32 = 6;

impl Cursor {
    /// The start of a walk, and the same value as the end of one.
    pub const START: Cursor = Cursor(0);

    /// The cursor a client sent, whatever it sent.
    ///
    /// Any number is a valid cursor. A made up one resumes somewhere arbitrary
    /// and answers keys from there, which is what Redis does and is the only
    /// behaviour that does not require the server to remember every cursor it
    /// has ever handed out.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Cursor {
        Cursor(raw)
    }

    /// The number to hand the client.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Whether the walk is over.
    #[must_use]
    pub const fn is_end(self) -> bool {
        self.0 == 0
    }

    /// The prefix half, which says which segment.
    #[must_use]
    pub(crate) const fn prefix(self) -> u64 {
        (self.0 >> PREFIX_SHIFT) & ((1 << PREFIX_BITS) - 1)
    }

    /// The bucket half, which says where in the segment.
    #[must_use]
    pub(crate) const fn bucket(self) -> usize {
        (self.0 & ((1 << BUCKET_BITS) - 1)) as usize
    }

    /// Put the two halves back together.
    ///
    /// A prefix that has run off the top of its 48 bits means the last segment
    /// is done, which is the end of the walk and therefore zero.
    #[must_use]
    pub(crate) const fn at(prefix: u64, bucket: usize) -> Cursor {
        if prefix >= (1 << PREFIX_BITS) {
            return Cursor::START;
        }
        Cursor((prefix << PREFIX_SHIFT) | (bucket as u64 & ((1 << BUCKET_BITS) - 1)))
    }

    /// The prefix of a key, which is the part of its hash the directory reads.
    ///
    /// Left aligned into the full 48 bits rather than into `global_depth` of
    /// them, which is the whole trick: this number is the same before a doubling
    /// and after one.
    ///
    /// Only the tests need this. The walk itself never goes from a key to a
    /// cursor, it only ever goes forward from the cursor it was handed, so this
    /// is the statement of the invariant rather than a step in the code.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn prefix_of(hash: u64) -> u64 {
        (hash >> (super::index::DIR_BITS - PREFIX_BITS)) & ((1 << PREFIX_BITS) - 1)
    }
}

impl From<u64> for Cursor {
    fn from(raw: u64) -> Cursor {
        Cursor::from_raw(raw)
    }
}

impl From<Cursor> for u64 {
    fn from(c: Cursor) -> u64 {
        c.raw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_halves_survive_the_round_trip() {
        for prefix in [0u64, 1, 255, (1 << PREFIX_BITS) - 1] {
            for bucket in [0usize, 1, 63] {
                let c = Cursor::at(prefix, bucket);
                assert_eq!(c.prefix(), prefix, "prefix {prefix} bucket {bucket}");
                assert_eq!(c.bucket(), bucket, "prefix {prefix} bucket {bucket}");
            }
        }
    }

    #[test]
    fn a_prefix_past_the_end_is_the_end() {
        assert_eq!(Cursor::at(1 << PREFIX_BITS, 0), Cursor::START);
        assert!(Cursor::at(1 << PREFIX_BITS, 7).is_end());
        // And zero with a bucket in it is not the end, because a walk that has
        // done one bucket of the first segment has not finished.
        assert!(!Cursor::at(0, 1).is_end());
    }

    #[test]
    fn a_prefix_is_the_directory_index_at_every_depth() {
        // What the index does at depth g, spelled out here rather than reached
        // through a private method, so the two are checked against each other.
        let hash = 0x1234_5678_9abc_def0u64;
        let prefix = Cursor::prefix_of(hash);
        for g in 1..=16u32 {
            let dir_bits = super::super::index::DIR_BITS;
            let want = (hash >> (dir_bits - g)) & ((1 << g) - 1);
            assert_eq!(prefix >> (PREFIX_BITS - g), want, "depth {g}");
        }
    }

    #[test]
    fn the_cursor_a_client_holds_is_just_a_number() {
        let c = Cursor::from_raw(0x0001_0000_0000_002a);
        assert_eq!(u64::from(c), 0x0001_0000_0000_002a);
        assert_eq!(Cursor::from(7u64).bucket(), 7);
    }
}
