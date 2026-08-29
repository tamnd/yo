//! The element table, which is what a hash, a set and a sorted set are all
//! made of underneath.
//!
//! One structure serves every collection because the three of them ask the same
//! two questions. Is this member here, and what is stored against it. A hash
//! stores a value address and a TTL slot against a field name, a set stores
//! nothing at all against a member, and a sorted set stores a score. That is one
//! table with a payload type the caller picks, and it is `05` section 4.2's
//! element per row: a dense array of fixed size rows, plus a blob holding the
//! variable length names, plus an open addressed slot array in front of them
//! that turns a name into a row index.
//!
//! ```text
//!   slots            rows                      names
//! +--------+       +--------------+--------+  +--------------------------+
//! | tag|idx| ----> | name off,len | payload|  | fieldbytesmemberbytes... |
//! +--------+       +--------------+--------+  +--------------------------+
//!  one load         one load                   only touched on a tag hit
//! ```
//!
//! Three properties come out of that shape and all three are the reason for it.
//!
//! A probe is one load into the slot array and one into the row array. The top
//! byte of a slot is a tag taken from the hash, so a collision on the low bits
//! is thrown out without reading the name bytes at all, and the name is only
//! compared when the tag says it is worth comparing.
//!
//! A walk is sequential. `HGETALL`, `SMEMBERS` and `HSCAN` read the row array
//! front to back with no pointer chasing, which is the difference between the
//! 13.6 nanoseconds a field walk actually costs and the number a linked
//! structure would cost.
//!
//! A uniform draw is an index. `SPOP` and `SRANDMEMBER` pick a number under
//! [`Elements::len`] and read that row, because the row array has no holes in
//! it. That is K9 and it is the whole of aki's signature failure: `SPOP` came in
//! at 0.58x at pipeline 16 and 0.29x at pipeline 1 there, because a draw had to
//! remove from an ordered structure, and here there is no ordered structure to
//! remove from.
//!
//! # Removal
//!
//! Keeping the row array dense means a removal moves the last row into the hole
//! and fixes up the slot that pointed at it, which costs one extra probe. The
//! alternative is a free list and holes, and then a draw has to retry until it
//! lands on a live row, which is fine at 90 percent occupancy and unbounded on a
//! set that has been drained down to its last member.
//!
//! The slot the removed member sat in is closed by shifting the run behind it
//! back, not by leaving a tombstone. A tombstone is cheaper on the delete and it
//! is the wrong trade here: `SPOP` in a loop empties a set, and a table that
//! answers an empty set by walking a million tombstones is slower than the
//! structure it replaced.
//!
//! Neither the removal nor the shift reads a name, because every row carries the
//! slot it wanted. Four bytes a row for that, and it took a pop at a hundred
//! thousand members from 123 ns to 25.7 ns, which is the difference between a
//! random trip into the name blob per slot examined and no trip at all.
//!
//! # Names
//!
//! Names are interned per collection, which is `05` section 3's rule and the
//! reason a hash field costs 16 bytes and not 16 bytes plus its name on every
//! write. Writing the same field again is a row update and touches no name
//! bytes. It is per collection and not global because a global table is state
//! shared between shards and Y1 does not allow any.
//!
//! Removing a member leaves its bytes in the blob unreferenced. Those bytes come
//! back when the dead share crosses a half and there are at least a few thousand
//! of them, which is a rewrite of the blob and a walk over the rows to move
//! their offsets, and until then they are counted and reported rather than
//! pretended away. That accounting lives in [`crate::blob`], because a hash's
//! values want exactly the same thing and there should be one copy of it.

use yo_common::{hash_key, tag_of};

use crate::blob::Blob;
use crate::scan::Cursor;

/// The most rows one table holds.
///
/// A slot packs a tag and a row index into 32 bits, which leaves 24 bits for the
/// index. A collection past this belongs in the partitioned band of `05`
/// section 4.3, which is a set of these rather than a bigger one, and the band
/// boundary is 262,144, well under this.
pub const MAX_ROWS: usize = 0x00FF_FFFE;

/// The longest name this table stores.
///
/// Redis has no limit on a field name below the 512 MiB it puts on everything.
/// A name that long is a value that has been put in the wrong place, and holding
/// the ceiling at what fits in sixteen bits is what keeps a row at twelve bytes
/// plus its payload.
pub const NAME_MAX: usize = u16::MAX as usize;

/// An empty slot. No live slot can hold this, because a live row index is at
/// most [`MAX_ROWS`] and the tag occupies the byte above it.
const EMPTY: u32 = u32::MAX;

/// How full the slot array is allowed to get before it doubles.
///
/// Three quarters is where linear probing is still short and the array is not
/// mostly air. The run length at this load is under three on average, which is
/// inside one cache line of slots.
const LOAD_NUM: usize = 3;
const LOAD_DEN: usize = 4;

/// The smallest slot array, which is one cache line of slots.
const MIN_SLOTS: usize = 16;

/// Where a name sits in the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NameRef {
    at: u32,
    len: u16,
}

/// One element: where its name is, where it wanted to sit, and what is stored
/// against it.
///
/// `home` is the low 32 bits of the name's hash, which is where the slot for
/// this row would sit in an empty table. Four bytes to hold it, and what they
/// buy is that a removal and a growth never read a name and never hash one. Both
/// of those walk slots and ask each one where it wanted to be, and asking the
/// blob instead means a random cache miss per slot examined, on the two
/// operations where there is no reply to send that would have paid for it.
#[derive(Debug, Clone, Copy)]
struct Row<V> {
    name: NameRef,
    home: u32,
    value: V,
}

/// An open addressed table of elements, keyed by name, dense in insertion order.
///
/// The payload is whatever the collection needs. A set uses `()`, a hash uses
/// the value address and the TTL slot, a sorted set uses the score.
#[derive(Debug, Clone)]
pub struct Elements<V> {
    /// Tag in the top byte, row index in the low 24 bits, [`EMPTY`] for nothing.
    slots: Box<[u32]>,
    /// The rows, in insertion order, with no holes.
    rows: Vec<Row<V>>,
    /// Every live name, back to back, and some dead ones.
    ///
    /// The length stays in [`NameRef`] rather than in a [`crate::blob::Span`],
    /// because sixteen bits of it is what keeps a row at twelve bytes and a name
    /// that needs more than sixteen bits is a value someone put in the wrong
    /// place.
    names: Blob,
}

impl<V: Copy> Default for Elements<V> {
    fn default() -> Elements<V> {
        Elements::new()
    }
}

impl<V: Copy> Elements<V> {
    /// An empty table that has not allocated anything yet.
    ///
    /// A collection is created by its first write, so the empty case is the one
    /// that happens most often and it does not deserve an allocation.
    #[must_use]
    pub fn new() -> Elements<V> {
        Elements {
            slots: Box::new([]),
            rows: Vec::new(),
            names: Blob::new(),
        }
    }

    /// An empty table with room for `n` elements already taken.
    ///
    /// This is Y18's presize rule. `SINTERSTORE` knows the result is no larger
    /// than its smaller input, so it says so once instead of growing eight
    /// times on the way there.
    #[must_use]
    pub fn with_capacity(n: usize) -> Elements<V> {
        let mut e = Elements::new();
        if n > 0 {
            e.rows.reserve(n);
            e.grow_to(slots_for(n));
        }
        e
    }

    /// How many elements are here.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the collection is empty, which for Redis means it does not exist.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// What is stored against this name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &[u8]) -> Option<&V> {
        let at = self.find(name)?;
        Some(&self.rows[at].value)
    }

    /// The payload, to be changed in place.
    ///
    /// This is the `HINCRBY` and `ZINCRBY` path. Neither of them writes a name,
    /// so neither of them should pay for one.
    #[inline]
    pub fn get_mut(&mut self, name: &[u8]) -> Option<&mut V> {
        let at = self.find(name)?;
        Some(&mut self.rows[at].value)
    }

    /// Whether this name is here at all. `SISMEMBER` and `HEXISTS`.
    #[inline]
    #[must_use]
    pub fn contains(&self, name: &[u8]) -> bool {
        self.find(name).is_some()
    }

    /// Which row this name is in, for a caller keeping an array beside the rows.
    ///
    /// A hash's field deadlines are indexed by row position rather than by a
    /// number in the row (`crate::ttl` says why), so `HEXPIRE` needs the position
    /// the probe found rather than the payload it found there. That is the only
    /// caller, and it is why this is a position and not a payload.
    ///
    /// The position is only good until the next insert or remove, since a remove
    /// moves the last row into the hole.
    #[inline]
    #[must_use]
    pub fn index_of(&self, name: &[u8]) -> Option<usize> {
        self.find(name)
    }

    /// The hash of a name, for a caller about to ask several tables about it.
    ///
    /// `SINTER` over k sets asks the same question k times, and hashing the
    /// member once instead of k times is the difference between the hash being
    /// noise and it being most of the work. Pair it with
    /// [`Elements::contains_hashed`].
    #[inline]
    #[must_use]
    pub fn hash_of(name: &[u8]) -> u64 {
        hash(name)
    }

    /// Whether this name is here, with its hash already in hand.
    ///
    /// The hash must be [`Elements::hash_of`] of the same bytes. Anything else
    /// gives a wrong answer rather than an error, which is why this takes the
    /// name too and compares it: a caller cannot fake membership with a number.
    #[inline]
    #[must_use]
    pub fn contains_hashed(&self, h: u64, name: &[u8]) -> bool {
        self.find_hashed(h, name).is_some()
    }

    /// Store `value` against `name`, and say what was there before.
    ///
    /// `None` means the element is new, which is the number `SADD` and `HSET`
    /// report. A name over [`NAME_MAX`] or a table at [`MAX_ROWS`] is refused
    /// rather than truncated, and refusing is a `false` here and an error
    /// message from the layer above, which is the one that knows which command
    /// is being answered.
    pub fn insert(&mut self, name: &[u8], value: V) -> Result<Option<V>, Full> {
        if name.len() > NAME_MAX {
            return Err(Full::Name);
        }
        let h = hash(name);
        if let Some(at) = self.find_hashed(h, name) {
            return Ok(Some(std::mem::replace(&mut self.rows[at].value, value)));
        }
        if self.rows.len() >= MAX_ROWS {
            return Err(Full::Rows);
        }
        self.reserve_one();
        let at = u32::try_from(self.rows.len()).expect("MAX_ROWS is under u32::MAX");
        let name_ref = self.push_name(name);
        self.rows.push(Row {
            name: name_ref,
            home: h as u32,
            value,
        });
        self.put_slot(h, at);
        Ok(None)
    }

    /// Take an element out and hand back what it held.
    ///
    /// `SREM`, `HDEL` and the removing half of `SPOP`.
    pub fn remove(&mut self, name: &[u8]) -> Option<V> {
        let at = self.find(name)?;
        Some(self.remove_row(at))
    }

    /// Take the element at a position out, without looking its name up again.
    ///
    /// `SPOP` reads the name with [`Elements::at`], writes it into the reply,
    /// and then calls this. That way the name is copied once, into the buffer it
    /// was going to be copied into anyway, rather than into a `Vec` that exists
    /// only to be dropped after the reply is framed.
    pub fn remove_at(&mut self, idx: usize) -> Option<V> {
        if idx >= self.rows.len() {
            return None;
        }
        Some(self.remove_row(idx))
    }

    /// The name and payload of one row, by position.
    ///
    /// The dense draw. `SRANDMEMBER` picks a number under [`Elements::len`] and
    /// calls this, and that is the whole operation: no walk, no ordered
    /// structure, no allocation.
    #[inline]
    #[must_use]
    pub fn at(&self, idx: usize) -> Option<(&[u8], &V)> {
        let row = self.rows.get(idx)?;
        Some((self.name_of(row), &row.value))
    }

    /// The payload at `idx`, to be written over.
    ///
    /// The companion to [`Elements::index_of`], for a caller that has probed
    /// once and wants to use the position it found rather than probe again.
    #[inline]
    pub fn at_mut(&mut self, idx: usize) -> Option<&mut V> {
        self.rows.get_mut(idx).map(|r| &mut r.value)
    }

    /// Take the row at `idx` out and hand back its name and payload.
    ///
    /// The convenient form of a draw and a removal, for a caller that wants the
    /// name and does not have a buffer to put it in. It allocates. The path that
    /// answers a client uses [`Elements::at`] and then [`Elements::remove_at`]
    /// and allocates nothing.
    pub fn take_at(&mut self, idx: usize) -> Option<(Vec<u8>, V)> {
        if idx >= self.rows.len() {
            return None;
        }
        let name = self.name_of(&self.rows[idx]).to_vec();
        let value = self.remove_row(idx);
        Some((name, value))
    }

    /// Every element, in insertion order.
    ///
    /// The sequential walk. `HGETALL`, `SMEMBERS` and the scan cursor all read
    /// the row array front to back, which is one stream of cache lines and no
    /// pointer chasing.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &V)> {
        self.rows.iter().map(|r| (self.name_of(r), &r.value))
    }

    /// Every payload, to be changed in place, with no names in the way.
    ///
    /// A hash keeps its values in a blob of its own and the payload is where
    /// they are, so when that blob compacts every one of those references has to
    /// move. The names are deliberately not offered here: this borrows the rows
    /// mutably, and handing out a name at the same time would borrow the name
    /// blob as well for no caller that wants it.
    pub fn payloads_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.rows.iter_mut().map(|r| &mut r.value)
    }

    /// Walk part of the table and say where to resume.
    ///
    /// This is `SSCAN`, `HSCAN` and `ZSCAN`. It reads downward from the cursor,
    /// hands each element to `f`, and stops after `count` of them or at the
    /// bottom, whichever comes first. A returned cursor that is
    /// [`Cursor::is_end`] means the collection has been walked.
    ///
    /// Downward is what makes the guarantee hold while the collection is being
    /// written, and [`crate::scan`] is where the argument for that lives. `count`
    /// is a hint in Redis and a limit here, and a zero is read as one, because a
    /// scan that returns nothing and the same cursor is a client that never
    /// finishes.
    ///
    /// This band is one partition, so a cursor from a partitioned layout is
    /// rebased onto it before anything is read.
    pub fn scan<F>(&self, cursor: Cursor, count: usize, mut f: F) -> Cursor
    where
        F: FnMut(&[u8], &V),
    {
        if self.rows.is_empty() {
            return Cursor::END;
        }
        let here = cursor.rebase(1);
        let top = self.rows.len() - 1;
        // A cursor from before a run of removals can name a row that is no
        // longer there. Everything above the end has been walked already or was
        // never there, so the top is the honest place to carry on from.
        let mut at = match here.idx() {
            Some(idx) => (idx as usize).min(top),
            None => top,
        };
        for _ in 0..count.max(1) {
            let row = &self.rows[at];
            f(self.name_of(row), &row.value);
            if at == 0 {
                return Cursor::END;
            }
            at -= 1;
        }
        Cursor::at(1, 0, at as u64)
    }

    /// Throw everything away and keep the allocations.
    ///
    /// Emptying a collection usually means it is about to be filled again, which
    /// is `SINTERSTORE` over the same destination in a loop.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.names.clear();
        for slot in &mut self.slots {
            *slot = EMPTY;
        }
    }

    /// What this table costs, not counting anything the payload points at.
    ///
    /// The payload is the caller's, so a value that lives in the arena is
    /// counted by the arena and not twice here.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.slots.len() * size_of::<u32>()
            + self.rows.capacity() * size_of::<Row<V>>()
            + self.names.memory_bytes()
    }

    /// Name bytes no row points at any more.
    ///
    /// Reported rather than hidden, because a set that has been written and
    /// rewritten holds them and `INFO memory` should say so.
    #[inline]
    #[must_use]
    pub const fn dead_name_bytes(&self) -> usize {
        self.names.dead()
    }

    /// Row index for a name, or nothing.
    #[inline]
    fn find(&self, name: &[u8]) -> Option<usize> {
        self.find_hashed(hash(name), name)
    }

    /// The probe itself, with the hash already in hand.
    ///
    /// One load from the slot array. The tag in the top byte throws out a
    /// collision on the low bits without touching the row, so the name
    /// comparison below runs about once per hit and not once per probe.
    #[inline]
    fn find_hashed(&self, h: u64, name: &[u8]) -> Option<usize> {
        if self.rows.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let tag = tag_of(h);
        let mut at = (h as usize) & mask;
        loop {
            let slot = self.slots[at];
            if slot == EMPTY {
                return None;
            }
            if slot >> 24 == u32::from(tag) {
                let row = (slot & 0x00FF_FFFF) as usize;
                if self.name_of(&self.rows[row]) == name {
                    return Some(row);
                }
            }
            at = (at + 1) & mask;
        }
    }

    /// Put a row index in the first slot the probe reaches.
    fn put_slot(&mut self, h: u64, row: u32) {
        let mask = self.slots.len() - 1;
        let mut at = (h as usize) & mask;
        while self.slots[at] != EMPTY {
            at = (at + 1) & mask;
        }
        self.slots[at] = (u32::from(tag_of(h)) << 24) | row;
    }

    /// Take the row at `at` out, keeping the row array dense.
    fn remove_row(&mut self, at: usize) -> V {
        let last = self.rows.len() - 1;
        self.clear_slot(at);
        if at != last {
            // The last row moves into the hole, so the slot that pointed at the
            // end now has to point here. One extra probe, which is what a draw
            // being a single index costs.
            let moved = self.rows[last].home;
            self.repoint(moved, last, at);
            self.rows.swap(at, last);
        }
        let row = self.rows.pop().expect("the table was not empty");
        self.names.release(row.name.len as usize);
        self.maybe_compact_names();
        row.value
    }

    /// Close the slot holding `row`, and shift the run behind it back.
    ///
    /// Linear probing means a run of occupied slots is a chain, and punching a
    /// hole in the middle of one hides everything behind it. Shifting the run
    /// back keeps every probe correct and leaves no tombstone, which matters
    /// because `SPOP` in a loop deletes every element a set ever had.
    fn clear_slot(&mut self, row: usize) {
        let mask = self.slots.len() - 1;
        let mut at = (self.rows[row].home as usize) & mask;
        loop {
            let slot = self.slots[at];
            debug_assert!(slot != EMPTY, "the row being removed has a slot");
            if slot != EMPTY && (slot & 0x00FF_FFFF) as usize == row {
                break;
            }
            at = (at + 1) & mask;
        }
        self.slots[at] = EMPTY;

        // Everything behind the hole that probed past it has to move up, or it
        // becomes unreachable.
        let mut hole = at;
        let mut scan = (at + 1) & mask;
        while self.slots[scan] != EMPTY {
            let slot = self.slots[scan];
            let idx = (slot & 0x00FF_FFFF) as usize;
            let home = (self.rows[idx].home as usize) & mask;
            // True when the slot's home is at or behind the hole, meaning it
            // probed over the hole to get here and would not be found now.
            if (scan.wrapping_sub(home) & mask) >= (scan.wrapping_sub(hole) & mask) {
                self.slots[hole] = slot;
                self.slots[scan] = EMPTY;
                hole = scan;
            }
            scan = (scan + 1) & mask;
        }
    }

    /// Point the slot holding `from` at `to` instead.
    fn repoint(&mut self, home: u32, from: usize, to: usize) {
        let mask = self.slots.len() - 1;
        let mut at = (home as usize) & mask;
        loop {
            let slot = self.slots[at];
            debug_assert!(slot != EMPTY, "the row being moved has a slot");
            if slot != EMPTY && (slot & 0x00FF_FFFF) as usize == from {
                self.slots[at] =
                    (slot & 0xFF00_0000) | u32::try_from(to).expect("a row index fits in 24 bits");
                return;
            }
            at = (at + 1) & mask;
        }
    }

    /// Make sure there is room for one more before it is inserted.
    fn reserve_one(&mut self) {
        let want = self.rows.len() + 1;
        if want * LOAD_DEN > self.slots.len() * LOAD_NUM {
            self.grow_to(slots_for(want));
        }
    }

    /// Rebuild the slot array at a new size.
    ///
    /// The rows do not move and the names do not move. Only the slots are, and
    /// they are rebuilt from the old slot array rather than from the names: the
    /// tag is already in the old slot and the home is already in the row, so a
    /// growth reads two flat arrays and hashes nothing.
    fn grow_to(&mut self, slots: usize) {
        let slots = slots.max(MIN_SLOTS).next_power_of_two();
        let mask = slots - 1;
        let old = std::mem::replace(&mut self.slots, vec![EMPTY; slots].into_boxed_slice());
        for &slot in &old {
            if slot == EMPTY {
                continue;
            }
            let row = (slot & 0x00FF_FFFF) as usize;
            let mut at = (self.rows[row].home as usize) & mask;
            while self.slots[at] != EMPTY {
                at = (at + 1) & mask;
            }
            self.slots[at] = slot;
        }
    }

    /// Append a name to the blob.
    fn push_name(&mut self, name: &[u8]) -> NameRef {
        NameRef {
            at: self.names.push(name),
            len: u16::try_from(name.len()).expect("the caller checked NAME_MAX"),
        }
    }

    /// The bytes of one row's name.
    #[inline]
    fn name_of(&self, row: &Row<V>) -> &[u8] {
        self.names.read(row.name.at, row.name.len as usize)
    }

    /// Give the dead name bytes back once there are more of them than live ones.
    ///
    /// The line and the floor are the blob's, and walking in row order is what
    /// leaves a name walk sequential afterwards.
    fn maybe_compact_names(&mut self) {
        if !self.names.worth_compacting() {
            return;
        }
        let rows = &mut self.rows;
        self.names.compact(|keep| {
            for row in rows.iter_mut() {
                keep.moved(&mut row.name.at, row.name.len as usize);
            }
        });
    }
}

/// Why an insert was refused.
///
/// Two ways, both of them a limit of this band rather than of Redis, and both
/// turned into Redis's own error text by the command layer above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Full {
    /// The name is longer than [`NAME_MAX`].
    Name,
    /// The collection already holds [`MAX_ROWS`] elements.
    Rows,
}

/// The hash a name is filed under.
///
/// wyhash at the shard's seed, the same call the key index makes, because a
/// field name and a key are the same kind of short byte string and there is no
/// reason to have two hashes in one process.
#[inline]
fn hash(name: &[u8]) -> u64 {
    hash_key(name)
}

/// How many slots `n` elements need at the load factor.
fn slots_for(n: usize) -> usize {
    ((n * LOAD_DEN) / LOAD_NUM + 1)
        .max(MIN_SLOTS)
        .next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A set is this table with nothing stored against a member.
    type Set = Elements<()>;

    fn set(members: &[&[u8]]) -> Set {
        let mut s = Set::new();
        for m in members {
            s.insert(m, ()).expect("room");
        }
        s
    }

    #[test]
    fn an_empty_table_allocates_nothing() {
        let e = Set::new();
        assert_eq!(e.len(), 0);
        assert!(e.is_empty());
        assert_eq!(e.memory_bytes(), 0);
        assert!(!e.contains(b"anything"));
    }

    #[test]
    fn what_goes_in_comes_out() {
        let mut h: Elements<u64> = Elements::new();
        assert_eq!(h.insert(b"name", 7), Ok(None));
        assert_eq!(h.insert(b"age", 41), Ok(None));
        assert_eq!(h.get(b"name"), Some(&7));
        assert_eq!(h.get(b"age"), Some(&41));
        assert_eq!(h.get(b"missing"), None);
        assert_eq!(h.len(), 2);
    }

    /// The number `HSET` reports is how many fields were new, so an overwrite
    /// has to be distinguishable from an insert.
    #[test]
    fn writing_a_field_again_replaces_it_and_says_so() {
        let mut h: Elements<u64> = Elements::new();
        assert_eq!(h.insert(b"f", 1), Ok(None));
        assert_eq!(h.insert(b"f", 2), Ok(Some(1)));
        assert_eq!(h.len(), 1, "an overwrite is not a second element");
        assert_eq!(h.get(b"f"), Some(&2));
    }

    /// The name is written once. Rewriting a field is a row update and the blob
    /// does not move, which is what per collection interning is for.
    #[test]
    fn rewriting_a_field_does_not_write_its_name_again() {
        let mut h: Elements<u64> = Elements::new();
        h.insert(b"a-fairly-long-field-name", 1).expect("room");
        let after_first = h.memory_bytes();
        for i in 0..1000 {
            h.insert(b"a-fairly-long-field-name", i).expect("room");
        }
        assert_eq!(h.memory_bytes(), after_first);
        assert_eq!(h.dead_name_bytes(), 0);
    }

    #[test]
    fn removing_takes_the_element_out() {
        let mut s = set(&[b"a", b"b", b"c"]);
        assert_eq!(s.remove(b"b"), Some(()));
        assert_eq!(s.remove(b"b"), None);
        assert_eq!(s.len(), 2);
        assert!(s.contains(b"a"));
        assert!(s.contains(b"c"));
        assert!(!s.contains(b"b"));
    }

    /// The row array has no holes, so a draw is one index and never a retry.
    #[test]
    fn the_rows_stay_dense_through_removals() {
        let mut s = set(&[b"a", b"b", b"c", b"d", b"e"]);
        s.remove(b"a").expect("there");
        s.remove(b"c").expect("there");
        assert_eq!(s.len(), 3);
        let mut seen: Vec<Vec<u8>> = (0..s.len())
            .map(|i| s.at(i).expect("dense").0.to_vec())
            .collect();
        seen.sort();
        assert_eq!(seen, vec![b"b".to_vec(), b"d".to_vec(), b"e".to_vec()]);
        assert_eq!(s.at(3), None);
    }

    /// This is the case a tombstone would ruin, so it is the case with a test.
    /// Every member goes in, every member comes out one draw at a time, and the
    /// table answers correctly the whole way down.
    #[test]
    fn a_set_drained_one_draw_at_a_time_stays_correct() {
        let names: Vec<Vec<u8>> = (0..500u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut s = Set::new();
        for n in &names {
            s.insert(n, ()).expect("room");
        }
        let mut taken = Vec::new();
        // A fixed walk rather than a random one, because a test that draws
        // randomly and fails is a test nobody can rerun.
        while !s.is_empty() {
            let idx = (taken.len() * 7 + 3) % s.len();
            let (name, ()) = s.take_at(idx).expect("in range");
            assert!(!s.contains(&name), "it came out and stayed out");
            taken.push(name);
        }
        assert_eq!(taken.len(), names.len());
        taken.sort();
        let mut want = names;
        want.sort();
        assert_eq!(taken, want);
    }

    /// Everything still probes correctly after a removal from the middle of a
    /// linear probe run, which is what the backward shift is for.
    #[test]
    fn removals_do_not_hide_what_is_behind_them() {
        let mut s = Set::new();
        let names: Vec<Vec<u8>> = (0..200u32).map(|i| format!("k{i}").into_bytes()).collect();
        for n in &names {
            s.insert(n, ()).expect("room");
        }
        for n in names.iter().step_by(3) {
            assert_eq!(s.remove(n), Some(()));
        }
        for (i, n) in names.iter().enumerate() {
            assert_eq!(s.contains(n), i % 3 != 0, "member {i}");
        }
    }

    #[test]
    fn growth_keeps_everything_findable() {
        let names: Vec<Vec<u8>> = (0..5000u32)
            .map(|i| format!("member-number-{i}").into_bytes())
            .collect();
        let mut s = Set::new();
        for n in &names {
            s.insert(n, ()).expect("room");
        }
        assert_eq!(s.len(), names.len());
        for n in &names {
            assert!(s.contains(n));
        }
        assert!(!s.contains(b"member-number-5000"));
    }

    #[test]
    fn a_walk_reads_them_in_the_order_they_went_in() {
        let s = set(&[b"first", b"second", b"third"]);
        let seen: Vec<&[u8]> = s.iter().map(|(n, ())| n).collect();
        assert_eq!(seen, vec![&b"first"[..], &b"second"[..], &b"third"[..]]);
    }

    #[test]
    fn presizing_does_not_change_what_the_table_says() {
        let mut a = Set::with_capacity(1000);
        let mut b = Set::new();
        for i in 0..1000u32 {
            let n = format!("m{i}").into_bytes();
            a.insert(&n, ()).expect("room");
            b.insert(&n, ()).expect("room");
        }
        assert_eq!(a.len(), b.len());
        for i in 0..1000u32 {
            assert!(a.contains(format!("m{i}").as_bytes()));
        }
    }

    #[test]
    fn a_name_that_is_too_long_is_refused_and_not_truncated() {
        let mut s = Set::new();
        let long = vec![b'x'; NAME_MAX + 1];
        assert_eq!(s.insert(&long, ()), Err(Full::Name));
        assert!(s.is_empty());
        let ok = vec![b'x'; NAME_MAX];
        assert_eq!(s.insert(&ok, ()), Ok(None));
    }

    /// Dead name bytes are given back once there are more of them than live
    /// ones, and everything still reads correctly on the other side of it.
    #[test]
    fn dead_name_bytes_come_back() {
        let mut s = Set::new();
        let long: Vec<Vec<u8>> = (0..400u32)
            .map(|i| format!("{i:0>64}").into_bytes())
            .collect();
        for n in &long {
            s.insert(n, ()).expect("room");
        }
        let full = s.memory_bytes();
        for n in long.iter().take(390) {
            s.remove(n).expect("there");
        }
        assert!(
            s.memory_bytes() < full,
            "the blob shrank, {} against {full}",
            s.memory_bytes()
        );
        // Not zero. What is left is under the floor, which is the point of
        // having a floor: a few hundred bytes are not worth a copy.
        assert!(
            s.dead_name_bytes() < 4096,
            "{} bytes left dead",
            s.dead_name_bytes()
        );
        for n in long.iter().skip(390) {
            assert!(s.contains(n), "still findable after the blob moved");
        }
    }

    #[test]
    fn clearing_keeps_the_allocation_and_forgets_the_elements() {
        let mut s = set(&[b"a", b"b", b"c"]);
        let before = s.memory_bytes();
        s.clear();
        assert!(s.is_empty());
        assert!(!s.contains(b"a"));
        assert_eq!(s.memory_bytes(), before, "the room is kept for the refill");
        s.insert(b"a", ()).expect("room");
        assert!(s.contains(b"a"));
    }

    /// The empty marker is all ones, and a live slot cannot be all ones however
    /// the tag comes out, because a row index only ever occupies 24 bits and the
    /// table refuses to hold enough rows to fill them.
    #[test]
    fn no_live_slot_can_look_empty() {
        let mut s = Set::new();
        for i in 0..2000u32 {
            s.insert(format!("m{i}").as_bytes(), ()).expect("room");
        }
        assert_eq!(s.slots.iter().filter(|v| **v != EMPTY).count(), s.len());
        const { assert!(MAX_ROWS < 0x00FF_FFFF, "a row index is never all ones") }
    }

    /// Collect a whole scan, a page at a time, the way a client loops.
    fn scan_all(s: &Set, page: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut c = Cursor::START;
        loop {
            c = s.scan(c, page, |n, ()| out.push(n.to_vec()));
            if c.is_end() {
                return out;
            }
        }
    }

    #[test]
    fn a_scan_of_a_still_collection_returns_everything_once() {
        let names: Vec<Vec<u8>> = (0..300u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut s = Set::new();
        for n in &names {
            s.insert(n, ()).expect("room");
        }
        for page in [1, 7, 10, 1000] {
            let mut seen = scan_all(&s, page);
            assert_eq!(seen.len(), names.len(), "page {page} returned a duplicate");
            seen.sort();
            let mut want = names.clone();
            want.sort();
            assert_eq!(seen, want, "page {page}");
        }
    }

    #[test]
    fn scanning_an_empty_collection_is_over_immediately() {
        let s = Set::new();
        let mut hit = 0;
        assert!(s.scan(Cursor::START, 10, |_, ()| hit += 1).is_end());
        assert_eq!(hit, 0);
    }

    /// The guarantee, which is the only reason the walk goes downward. Members
    /// are removed while the scan is running, and every member that was there
    /// the whole time has to come back at least once. Duplicates are allowed and
    /// are not what this is checking.
    #[test]
    fn a_scan_never_misses_a_member_that_stayed() {
        let names: Vec<Vec<u8>> = (0..400u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut s = Set::new();
        for n in &names {
            s.insert(n, ()).expect("room");
        }

        // Every seventh member goes away, a few at a time, in the middle of the
        // scan. Removal moves the top row into the hole, so this is the case
        // that would break an upward walk.
        let doomed: Vec<Vec<u8>> = names.iter().step_by(7).cloned().collect();
        let mut gone = 0usize;
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut c = Cursor::START;
        loop {
            c = s.scan(c, 9, |n, ()| seen.push(n.to_vec()));
            for n in doomed.iter().skip(gone).take(3) {
                s.remove(n);
            }
            gone = (gone + 3).min(doomed.len());
            if c.is_end() {
                break;
            }
        }

        for n in &names {
            if doomed.contains(n) {
                continue;
            }
            assert!(
                seen.contains(n),
                "{} was there all along",
                String::from_utf8_lossy(n)
            );
        }
    }

    /// A cursor that names a row past the end, because the collection shrank
    /// under it, carries on rather than panicking or ending early.
    #[test]
    fn a_stale_cursor_is_answered_and_not_refused() {
        let s = set(&[b"a", b"b", b"c"]);
        let mut seen = Vec::new();
        let c = s.scan(Cursor::at(1, 0, 900), 2, |n, ()| seen.push(n.to_vec()));
        assert_eq!(seen, vec![b"c".to_vec(), b"b".to_vec()]);
        assert_eq!(c.idx(), Some(0));

        // And one from a layout this band does not have.
        let mut also = Vec::new();
        s.scan(Cursor::at(16, 9, 4), 99, |n, ()| also.push(n.to_vec()));
        assert_eq!(also.len(), 3);
    }

    /// The two ways to take an element out have to agree, because `SPOP` uses
    /// the one that does not allocate and `SREM` uses the one that looks a name
    /// up, and a set has to end up in the same state either way.
    #[test]
    fn taking_by_index_and_by_name_leave_the_same_table() {
        let mut by_index = set(&[b"a", b"b", b"c", b"d"]);
        let mut by_name = set(&[b"a", b"b", b"c", b"d"]);
        let name = by_index.at(1).expect("in range").0.to_vec();
        assert_eq!(by_index.remove_at(1), Some(()));
        assert_eq!(by_name.remove(&name), Some(()));
        assert_eq!(by_index.remove_at(99), None);

        let mut left: Vec<Vec<u8>> = by_index.iter().map(|(n, ())| n.to_vec()).collect();
        let mut also: Vec<Vec<u8>> = by_name.iter().map(|(n, ())| n.to_vec()).collect();
        left.sort();
        also.sort();
        assert_eq!(left, also);
        assert_eq!(left.len(), 3);
    }

    /// The row is the thing there are a million of, so its size is a decision
    /// and not an accident. Six bytes of name reference, four of home slot, and
    /// whatever the collection stores.
    #[test]
    fn a_row_is_twelve_bytes_plus_its_payload() {
        assert_eq!(size_of::<Row<()>>(), 12);
        assert_eq!(size_of::<Row<u32>>(), 16);
        assert_eq!(size_of::<Row<u64>>(), 24);
    }

    #[test]
    fn a_payload_can_be_changed_in_place() {
        let mut h: Elements<i64> = Elements::new();
        h.insert(b"counter", 1).expect("room");
        *h.get_mut(b"counter").expect("there") += 41;
        assert_eq!(h.get(b"counter"), Some(&42));
        assert_eq!(h.get_mut(b"nothing"), None);
    }
}
