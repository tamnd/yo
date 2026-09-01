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
//!   slots            rows            payloads   names
//! +--------+       +--------------+  +-------+  +--------------------------+
//! | tag|idx| ----> | name off,len |  | score |  | fieldbytesmemberbytes... |
//! +--------+       +--------------+  +-------+  +--------------------------+
//!  one load         one load          same idx   only touched on a tag hit
//! ```
//!
//! The payload sits beside the row rather than in it, so that a score's eight
//! byte alignment does not put four bytes of padding on every member.
//!
//! Three properties come out of that shape and all three are the reason for it.
//!
//! A probe is one load into the slot array and one into the row array. The top
//! byte of a slot is a tag taken from the hash, so a collision on the low bits
//! is thrown out without reading the name bytes at all, and the name is only
//! compared when the tag says it is worth comparing.
//!
//! Eight slots at a time, not one. The slot array is walked in aligned groups of
//! [`group::WIDTH`], and one group compare answers the tag question for all
//! eight at once and hands back a bit per slot. That is [`crate::group`], and
//! the reason it exists is that the memory a slot array costs and the length of
//! a probe through it are the same number seen twice: slots per element is one
//! over the load, and an unsuccessful linear probe costs about half of one plus
//! one over the square of one minus the load. One slot at a time, the array
//! could not be made smaller without being made slower. A group probe touches
//! about one group whatever the load is, which is what makes the array's size a
//! choice again.
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
//! The slot the removed member sat in is closed by writing a marker over it,
//! and the marker is the empty one rather than the dead one whenever it can
//! prove no probe ever ran past the slot. A probe stops at the first group with
//! an empty slot in it, so the proof is one question about the group the slot is
//! in: if it already has an empty, nothing ever went past it. Every marker in
//! that group goes back to empty at the same time and by the same argument, not
//! just the one being cleared, which is what makes a set filled and then drained
//! collect after itself.
//!
//! What that leaves behind is the groups that were completely full, because a
//! probe really did go past one of those and a marker in it is holding the path
//! open. At three quarters full that is about one group in ten, so a drained set
//! ends with a few percent of its slots marked rather than none of them. It used
//! to be none, when a removal shifted the run behind it back one slot at a time,
//! and a few percent is the price of the group probe. It is a small one: almost
//! every group still has an empty in it, so a miss against a drained set is
//! still one group.
//!
//! What is left over is counted and it counts against the load exactly as a live
//! member does, so a table churned in place rebuilds on the same schedule as one
//! that only grows and can never fill up with markers. That count is also what
//! bounds a probe, and the bound is easier than it looks: a removal turns a full
//! slot into a marker or into an empty one, so the two together never go up, so
//! a probe is never longer than it was at the moment the table was fullest.
//!
//! It is not free, and the case where it is not is a table drained and then read
//! from. Shifting the run back really did give the slot up, so a set emptied
//! from a million down to ten used to answer a miss like a table holding ten,
//! and now it answers like a table that once held a million, which is under
//! three slots looked at either way and is the whole of the trade.
//!
//! This used to shift the run behind the hole back instead, which leaves no
//! marker and costs a walk over that run with a home slot computed for every
//! slot in it. The marker is two writes and the shift was the single most
//! expensive thing on the removal path. Nothing here allocates, on either side
//! of the change, because a removal is on a command path and `cargo xtask alloc`
//! is the gate that says a command path allocates nothing.
//!
//! Neither the removal nor the marker reads a name, because every row carries
//! the slot it wanted. Three bytes a row for that, packed in beside the name
//! length, and it took a pop at a hundred thousand members from 123 ns to
//! 25.7 ns, which is the difference between a random trip into the name blob per
//! slot examined and no trip at all.
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

use yo_common::{bytes_eq, hash_key, tag_of};

use crate::blob::Blob;
use crate::group;
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
/// the ceiling at what fits in sixteen bits is what lets a long name carry its
/// own length in two bytes rather than four.
pub const NAME_MAX: usize = u16::MAX as usize;

/// The low twenty four bits of a slot, which are the row index.
const ROW: u32 = 0x00FF_FFFF;

/// A slot nothing has ever been written to. A probe stops here.
const EMPTY: u32 = 0xFFFF_FFFF;

/// A slot something was written to and then removed from. A probe keeps going.
///
/// Both markers have all twenty four row bits set and a live row index never
/// does, because [`MAX_ROWS`] is one short of that, so `slot & ROW == ROW` tells
/// a marker of either kind from a live slot in two instructions. Doing it that
/// way rather than by stealing the top bit is what keeps the tag a full eight
/// bits: a seven bit tag would double the rate at which a probe reads a row it
/// is about to reject, and the probe is the hottest path in the engine.
const TOMB: u32 = 0x00FF_FFFF;

/// How full the slot array is allowed to get before it doubles.
///
/// Three quarters is where linear probing is still short and the array is not
/// mostly air. The run length at this load is under three on average, which is
/// inside one cache line of slots. Markers count towards it, because a marker is
/// a slot a probe has to look at and step over.
const LOAD_NUM: usize = 3;
const LOAD_DEN: usize = 4;

/// The smallest slot array, which is one cache line of slots.
const MIN_SLOTS: usize = 16;

/// The shortest name that keeps its length in the blob instead of in its row.
///
/// A row holds the length in one byte, so a name this long or longer writes its
/// real length into the two bytes ahead of it and puts this sentinel in the
/// byte. Nothing on the probe path pays much for that: the prefix sits in the
/// cache line the name comparison was about to read anyway, and the branch is a
/// comparison against a constant that goes the same way on every element of
/// every collection anyone has ever measured.
const LONG_NAME: usize = 255;

/// How many bytes a long name's length prefix takes.
const PREFIX: usize = 2;

/// How many bits of the home slot a row keeps.
const HOME_BITS: u32 = 24;

/// One element: where its name is and where it wanted to sit.
///
/// Eight bytes, and the packing is what makes it eight rather than twelve. The
/// blob offset needs a whole `u32` because a large collection's names run to
/// megabytes. The other four hold the name's length in the low byte and the home
/// slot in the twenty four above it.
///
/// The home slot is where this row would sit in an empty table, and what it buys
/// is that a removal and a growth never read a name and never hash one. Both of
/// those walk slots and ask each one where it wanted to be, and asking the blob
/// instead means a random cache miss per slot examined, on the two operations
/// where there is no reply to send that would have paid for it.
///
/// Twenty four bits of it is every bit that matters until the slot array passes
/// sixteen million, which is a table holding twelve million elements. Past there
/// [`Elements::home_of`] hashes the name instead, and that is the right place for
/// the cost to land: the partitioned band splits a collection at a quarter of a
/// million, so a table that large is one partition of a set with two hundred
/// million members in it.
///
/// The payload is deliberately not in here. See [`Elements::vals`].
#[derive(Debug, Clone, Copy)]
struct Row {
    /// Where the name starts in the blob, or where its length prefix does.
    at: u32,
    /// The name's length in the low eight bits, its home slot in the top
    /// twenty four.
    packed: u32,
}

impl Row {
    /// The row for a name of `len` bytes that has just been pushed at `at`.
    #[inline]
    fn new(at: u32, len: usize, h: u64) -> Row {
        let len = u32::try_from(len.min(LONG_NAME)).expect("LONG_NAME is one byte");
        Row {
            at,
            packed: ((h as u32 & ((1 << HOME_BITS) - 1)) << 8) | len,
        }
    }

    /// The length byte, which is [`LONG_NAME`] when the real length is in the
    /// blob.
    #[inline]
    const fn len_byte(self) -> usize {
        (self.packed & 0xFF) as usize
    }

    /// The low [`HOME_BITS`] of the name's hash.
    #[inline]
    const fn home(self) -> usize {
        (self.packed >> 8) as usize
    }
}

/// An open addressed table of elements, keyed by name, dense in insertion order.
///
/// The payload is whatever the collection needs. A set uses `()`, a hash uses
/// the value address and the TTL slot, a sorted set uses the score.
#[derive(Debug, Clone)]
pub struct Elements<V> {
    /// Tag in the top byte, row index in the low 24 bits, or [`EMPTY`]/[`TOMB`].
    slots: Box<[u32]>,
    /// How many slots hold [`TOMB`].
    ///
    /// These count against the load exactly as live rows do, which is what stops
    /// a table written and removed from in place filling up with them, and it is
    /// also what a drained table watches to know when to rebuild.
    dead: usize,
    /// The rows, in insertion order, with no holes.
    rows: Vec<Row>,
    /// The payloads, one per row and at the same index.
    ///
    /// Beside the rows rather than inside them, because a payload with a
    /// stricter alignment than the row's four bytes pays for that alignment on
    /// every element. A sorted set's score is the case that matters: eight byte
    /// aligned, so a row holding one is twenty four bytes to carry twenty, and
    /// the four wasted bytes are per member. Split, the pair is twenty and there
    /// is no padding anywhere. A set pays nothing for this either way, because
    /// `Vec<()>` does not allocate.
    ///
    /// It costs the walks a second array, which is a second sequential stream
    /// and not a second random access, so the prefetcher covers it.
    vals: Vec<V>,
    /// Every live name, back to back, and some dead ones.
    ///
    /// The length stays in the row rather than beside the offset, because one
    /// byte of it is what keeps a row at eight bytes, and the names that do not
    /// fit in one byte carry their own length in the blob instead of widening
    /// every row that does.
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
            dead: 0,
            rows: Vec::new(),
            vals: Vec::new(),
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
        e.reserve(n);
        e
    }

    /// Room for `n` elements in a table that already exists.
    ///
    /// [`Elements::with_capacity`] for a table being reused rather than built.
    /// A scratch table that is cleared and refilled on every call keeps
    /// whatever it grew to last time, so this does nothing at all unless the
    /// run coming up is bigger than any run before it, which is what takes the
    /// allocator off a `SUNION` sent in a loop.
    ///
    /// The slot array is only rebuilt when it could not hold `n` at the load
    /// factor, rather than whenever a size is named. Rebuilding it to the size
    /// it already is would be an allocation asked for by a call whose whole
    /// point is to avoid one.
    pub fn reserve(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        self.rows.reserve(n.saturating_sub(self.rows.len()));
        self.vals.reserve(n.saturating_sub(self.vals.len()));
        if (n + self.dead) * LOAD_DEN > self.slots.len() * LOAD_NUM {
            self.grow_to(slots_for(n));
        }
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
        Some(&self.vals[at])
    }

    /// The payload, to be changed in place.
    ///
    /// This is the `HINCRBY` and `ZINCRBY` path. Neither of them writes a name,
    /// so neither of them should pay for one.
    #[inline]
    pub fn get_mut(&mut self, name: &[u8]) -> Option<&mut V> {
        let at = self.find(name)?;
        Some(&mut self.vals[at])
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

    /// Which row this name is in, with its hash already in hand.
    #[inline]
    #[must_use]
    pub fn index_of_hashed(&self, h: u64, name: &[u8]) -> Option<usize> {
        self.find_hashed(h, name)
    }

    /// What is stored against this name, with its hash already in hand.
    #[inline]
    #[must_use]
    pub fn get_hashed(&self, h: u64, name: &[u8]) -> Option<&V> {
        let at = self.find_hashed(h, name)?;
        Some(&self.vals[at])
    }

    /// The payload to be changed in place, with the hash already in hand.
    #[inline]
    pub fn get_hashed_mut(&mut self, h: u64, name: &[u8]) -> Option<&mut V> {
        let at = self.find_hashed(h, name)?;
        Some(&mut self.vals[at])
    }

    /// Store `value` against `name`, and say what was there before.
    ///
    /// `None` means the element is new, which is the number `SADD` and `HSET`
    /// report. A name over [`NAME_MAX`] or a table at [`MAX_ROWS`] is refused
    /// rather than truncated, and refusing is a `false` here and an error
    /// message from the layer above, which is the one that knows which command
    /// is being answered.
    pub fn insert(&mut self, name: &[u8], value: V) -> Result<Option<V>, Full> {
        self.insert_hashed(hash(name), name, value)
    }

    /// Store `value` against `name`, with its hash already in hand.
    ///
    /// The partitioned band hashes once to pick a partition and would otherwise
    /// hash again to place the row inside it, which on a short member is most of
    /// the write.
    pub fn insert_hashed(&mut self, h: u64, name: &[u8], value: V) -> Result<Option<V>, Full> {
        if name.len() > NAME_MAX {
            return Err(Full::Name);
        }
        if let Some(at) = self.find_hashed(h, name) {
            return Ok(Some(std::mem::replace(&mut self.vals[at], value)));
        }
        if self.rows.len() >= MAX_ROWS {
            return Err(Full::Rows);
        }
        self.reserve_one();
        let at = u32::try_from(self.rows.len()).expect("MAX_ROWS is under u32::MAX");
        let name_at = self.push_name(name);
        self.rows.push(Row::new(name_at, name.len(), h));
        self.vals.push(value);
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

    /// Take an element out, with its hash already in hand.
    #[inline]
    pub fn remove_hashed(&mut self, h: u64, name: &[u8]) -> Option<V> {
        let at = self.find_hashed(h, name)?;
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
        Some((self.name_of(row), &self.vals[idx]))
    }

    /// The payload at `idx`, to be written over.
    ///
    /// The companion to [`Elements::index_of`], for a caller that has probed
    /// once and wants to use the position it found rather than probe again.
    #[inline]
    pub fn at_mut(&mut self, idx: usize) -> Option<&mut V> {
        self.vals.get_mut(idx)
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
        self.rows
            .iter()
            .zip(&self.vals)
            .map(|(r, v)| (self.name_of(r), v))
    }

    /// Every payload, to be changed in place, with no names in the way.
    ///
    /// A hash keeps its values in a blob of its own and the payload is where
    /// they are, so when that blob compacts every one of those references has to
    /// move. The names are deliberately not offered here: this borrows the rows
    /// mutably, and handing out a name at the same time would borrow the name
    /// blob as well for no caller that wants it.
    pub fn payloads_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.vals.iter_mut()
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
            f(self.name_of(row), &self.vals[at]);
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
        self.vals.clear();
        self.names.clear();
        for slot in &mut self.slots {
            *slot = EMPTY;
        }
        self.dead = 0;
    }

    /// What this table costs, not counting anything the payload points at.
    ///
    /// The payload is the caller's, so a value that lives in the arena is
    /// counted by the arena and not twice here.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.slot_bytes() + self.row_bytes() + self.names.memory_bytes()
    }

    /// What the slot array costs on its own, for the memory measurements.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slots.len() * size_of::<u32>()
    }

    /// What the row array costs on its own, capacity and not length, because
    /// the slack a doubling `Vec` is holding is memory this table is using.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.rows.capacity() * size_of::<Row>() + self.vals.capacity() * size_of::<V>()
    }

    /// What the name blob costs on its own, live bytes and dead ones together.
    #[must_use]
    pub fn name_bytes(&self) -> usize {
        self.names.memory_bytes()
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
    ///
    /// The stop is [`EMPTY`] and only [`EMPTY`], because a [`TOMB`] means
    /// something used to be here and whatever probed past it is still behind it.
    /// A marker cannot be mistaken for a match: its row bits are all ones and no
    /// row index is, so the check that rejects it is on the arm the tag already
    /// agreed with, which is one comparison in two hundred and fifty six.
    #[inline]
    fn find_hashed(&self, h: u64, name: &[u8]) -> Option<usize> {
        if self.rows.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let tag = tag_of(h);
        let mut at = self.group_of(h as usize & mask);
        loop {
            let slots = &self.slots[at..at + group::WIDTH];
            let mut hits = group::tags(slots, tag);
            while hits != 0 {
                let i = hits.trailing_zeros() as usize;
                hits &= hits - 1;
                let row = slots[i] & ROW;
                // A marker cannot be a hit on a tag, because `tag_of` never
                // answers zero and a marker's top byte is zero, and the empty
                // one is caught here on the one tag in two hundred and fifty six
                // that is all ones.
                if row != ROW && bytes_eq(self.name_of(&self.rows[row as usize]), name) {
                    return Some(row as usize);
                }
            }
            if group::empty(slots, EMPTY) != 0 {
                return None;
            }
            at = (at + group::WIDTH) & mask;
        }
    }

    /// Put a row index in the first free slot the probe reaches.
    ///
    /// Free rather than empty, so an insert takes a marker back as soon as it
    /// meets one. That is correct because the caller has already probed for this
    /// name and not found it, and because a later probe for the same name walks
    /// these slots in this order and stops only at an [`EMPTY`], which is at or
    /// after wherever this lands.
    fn put_slot(&mut self, h: u64, row: u32) {
        let mask = self.slots.len() - 1;
        let mut at = self.group_of(h as usize & mask);
        let at = loop {
            let free = group::free(&self.slots[at..at + group::WIDTH], ROW);
            if free != 0 {
                break at + free.trailing_zeros() as usize;
            }
            at = (at + group::WIDTH) & mask;
        };
        if self.slots[at] == TOMB {
            self.dead -= 1;
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
            self.repoint(last, at);
            self.rows.swap(at, last);
            self.vals.swap(at, last);
        }
        let row = self.rows.pop().expect("the table was not empty");
        let value = self.vals.pop().expect("a payload per row");
        let gone = self.footprint(&row);
        self.names.release(gone);
        self.maybe_compact_names();
        value
    }

    /// Close the slot holding `row`.
    ///
    /// A slot whose neighbour is already [`EMPTY`] is a slot nothing ever probed
    /// past, because a probe stops at the first empty one, so it can go straight
    /// back to empty and cost nothing. Any run of markers directly behind it goes
    /// with it, since the same argument now holds for each of them in turn, and
    /// that is what makes a drain collect after itself.
    ///
    /// Otherwise the slot becomes a [`TOMB`], which says keep going and counts
    /// against the load until the next rebuild.
    fn clear_slot(&mut self, row: usize) {
        let at = self.slot_of(row);
        let group = at & !(group::WIDTH - 1);
        let slots = &self.slots[group..group + group::WIDTH];
        if group::empty(slots, EMPTY) == 0 {
            self.slots[at] = TOMB;
            self.dead += 1;
            return;
        }
        // This group already stops a probe, so nothing ever went past it and
        // every marker in it is holding a place nobody is coming back for. They
        // all go, not just the one being cleared, which is what makes a set
        // filled and then drained collect after itself.
        self.slots[at] = EMPTY;
        for i in group..group + group::WIDTH {
            if self.slots[i] == TOMB {
                self.slots[i] = EMPTY;
                self.dead -= 1;
            }
        }
    }

    /// Point the slot holding `from` at `to` instead.
    fn repoint(&mut self, from: usize, to: usize) {
        let at = self.slot_of(from);
        let to = u32::try_from(to).expect("a row index fits in 24 bits");
        self.slots[at] = (self.slots[at] & !ROW) | to;
    }

    /// Which slot holds `row`.
    ///
    /// The row says where it wanted to sit, so this walks the same slots the
    /// name would have walked without ever reading the name, and it matches on
    /// the row index rather than on the tag because the tag is the one thing a
    /// row does not keep. A marker cannot match, because its row bits are all
    /// ones and no row index is.
    fn slot_of(&self, row: usize) -> usize {
        let mask = self.slots.len() - 1;
        let want = u32::try_from(row).expect("a row index fits in 24 bits");
        let mut at = self.group_of(self.home_of(row, mask));
        loop {
            let slots = &self.slots[at..at + group::WIDTH];
            let hit = group::rows(slots, ROW, want);
            if hit != 0 {
                return at + hit.trailing_zeros() as usize;
            }
            debug_assert!(
                group::empty(slots, EMPTY) == 0,
                "the row being looked for has a slot at or before the first gap"
            );
            at = (at + group::WIDTH) & mask;
        }
    }

    /// Where the group holding slot `at` starts.
    ///
    /// The slot array is a power of two and never shorter than [`group::WIDTH`],
    /// so every group is aligned, there is no partial one at the end, and this
    /// is one and rather than a division.
    #[inline(always)]
    const fn group_of(&self, at: usize) -> usize {
        at & !(group::WIDTH - 1)
    }

    /// Make sure there is room for one more before it is inserted.
    ///
    /// The row array grows by [`crate::grow`]'s policy rather than by `Vec`'s,
    /// because a doubling row array on a large collection is the single largest
    /// piece of memory nobody asked for in the whole structure. The slot array
    /// keeps its power of two, which is not a policy, it is what makes the
    /// probe a mask instead of a division.
    fn reserve_one(&mut self) {
        let want = self.rows.len() + 1;
        crate::grow::reserve(&mut self.rows, 1);
        crate::grow::reserve(&mut self.vals, 1);
        // The markers are in here because they are what a probe has to walk
        // past, so a table churned in place rebuilds on the same schedule as one
        // that only grows. A rebuild the markers alone triggered comes back the
        // same size or smaller and clears every one of them.
        if (want + self.dead) * LOAD_DEN > self.slots.len() * LOAD_NUM {
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
        self.dead = 0;
        for &slot in &old {
            if slot & ROW == ROW {
                continue;
            }
            let row = (slot & ROW) as usize;
            let mut at = self.group_of(self.home_of(row, mask));
            let at = loop {
                let free = group::empty(&self.slots[at..at + group::WIDTH], EMPTY);
                if free != 0 {
                    break at + free.trailing_zeros() as usize;
                }
                at = (at + group::WIDTH) & mask;
            };
            self.slots[at] = slot;
        }
    }

    /// Where the row at `idx` wanted to sit, in a table with this `mask`.
    ///
    /// One comparison against a number the caller already had in a register, and
    /// then a field of a row it was going to read anyway. The other arm is for a
    /// table with more slots than a row has bits to name one, which costs a hash
    /// and a trip into the blob and is the reason the row is eight bytes rather
    /// than twelve for everybody else.
    ///
    /// The arms are split and the cold one is kept out of line because this is
    /// called once per slot of the run behind a removal. Left as one function it
    /// has a hash call in it, the call stops it being inlined into that loop, and
    /// a pop of a thousand members measured 52 percent slower.
    #[inline(always)]
    fn home_of(&self, idx: usize, mask: usize) -> usize {
        let row = self.rows[idx];
        if mask < 1 << HOME_BITS {
            row.home() & mask
        } else {
            self.home_by_hash(&row, mask)
        }
    }

    /// Where a row wanted to sit in a table too large for the packed bits.
    #[cold]
    #[inline(never)]
    fn home_by_hash(&self, row: &Row, mask: usize) -> usize {
        hash(self.name_of(row)) as usize & mask
    }

    /// Append a name to the blob and say where it went.
    ///
    /// A long one goes in behind its own length, because a row has one byte to
    /// say how long a name is and that is not enough for this one.
    fn push_name(&mut self, name: &[u8]) -> u32 {
        if name.len() < LONG_NAME {
            return self.names.push(name);
        }
        let len = u16::try_from(name.len()).expect("the caller checked NAME_MAX");
        let at = self.names.push(&len.to_le_bytes());
        self.names.push(name);
        at
    }

    /// The bytes of one row's name.
    #[inline(always)]
    fn name_of(&self, row: &Row) -> &[u8] {
        let len = row.len_byte();
        if len < LONG_NAME {
            self.names.read(row.at, len)
        } else {
            self.long_name(row.at)
        }
    }

    /// The bytes of a name too long to measure in a row.
    #[cold]
    #[inline(never)]
    fn long_name(&self, at: u32) -> &[u8] {
        self.names.read(at + PREFIX as u32, self.long_len(at))
    }

    /// The real length of a long name, from the bytes written ahead of it.
    #[inline]
    fn long_len(&self, at: u32) -> usize {
        let head = self.names.read(at, PREFIX);
        usize::from(u16::from_le_bytes([head[0], head[1]]))
    }

    /// How many blob bytes one row's name occupies, its prefix included.
    #[inline(always)]
    fn footprint(&self, row: &Row) -> usize {
        let len = row.len_byte();
        if len < LONG_NAME {
            len
        } else {
            PREFIX + self.long_len(row.at)
        }
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
                let len = row.len_byte();
                let take = if len < LONG_NAME {
                    len
                } else {
                    // The length is in the bytes rather than in the row, and the
                    // blob this would normally read it from is half rebuilt, so
                    // it comes off the old copy the rebuild is reading from.
                    let head = keep.peek(row.at, PREFIX);
                    PREFIX + usize::from(u16::from_le_bytes([head[0], head[1]]))
                };
                keep.moved(&mut row.at, take);
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

    /// A row says how long a name is in one byte, and a name that does not fit
    /// in one byte keeps its length in the blob instead. Everything either side
    /// of that line has to read back as what went in, and the line itself is
    /// where an off by one lives, so this walks across it.
    #[test]
    fn a_name_too_long_to_measure_in_a_row_reads_back_whole() {
        let lens = [0, 1, 2, 253, 254, 255, 256, 257, 1000, NAME_MAX];
        // Distinct bytes per name as well as distinct lengths, so a read that
        // lands on the wrong name is not hidden by every name being x's.
        let names: Vec<Vec<u8>> = lens
            .iter()
            .enumerate()
            .map(|(i, &n)| vec![b'a' + u8::try_from(i).expect("under 26"); n])
            .collect();

        let mut s = Set::new();
        for name in &names {
            assert_eq!(s.insert(name, ()), Ok(None), "length {}", name.len());
        }
        assert_eq!(s.len(), names.len(), "two of them collided into one row");
        for name in &names {
            assert!(s.contains(name), "length {} went missing", name.len());
        }
        let mut back: Vec<Vec<u8>> = s.iter().map(|(n, ())| n.to_vec()).collect();
        back.sort();
        let mut want = names.clone();
        want.sort();
        assert_eq!(back, want, "a walk gave back different bytes");

        // And out again, one at a time, because a removal reads the length to
        // give the blob its bytes back and moves the last row into the hole.
        for (i, name) in names.iter().enumerate() {
            assert_eq!(s.remove(name), Some(()), "length {}", name.len());
            for later in &names[i + 1..] {
                assert!(s.contains(later), "length {} lost", later.len());
            }
        }
        assert!(s.is_empty());
    }

    /// The same names through a blob rebuild, which is the one place that has to
    /// read a length out of bytes that are being moved underneath it.
    #[test]
    fn long_names_survive_the_blob_giving_its_dead_bytes_back() {
        let mut s = Set::new();
        let names: Vec<Vec<u8>> = (0..200u32)
            .map(|i| format!("{i:0>500}").into_bytes())
            .collect();
        for name in &names {
            s.insert(name, ()).expect("room");
        }
        let keep: Vec<Vec<u8>> = (0..100u32)
            .map(|i| format!("keep-{i:0>500}").into_bytes())
            .collect();
        for name in &keep {
            s.insert(name, ()).expect("room");
        }
        let before = s.name_bytes();
        // A hundred kilobytes of dead names against fifty of live ones, which is
        // over the floor and past the ratio, so the removals rebuild.
        for name in &names {
            assert_eq!(s.remove(name), Some(()));
        }
        assert!(
            s.name_bytes() * 2 < before,
            "the rebuild never ran, the blob went from {before} to {}",
            s.name_bytes()
        );
        assert_eq!(s.len(), keep.len());
        for name in &keep {
            assert!(s.contains(name), "a long name moved wrongly");
        }
        let mut back: Vec<Vec<u8>> = s.iter().map(|(n, ())| n.to_vec()).collect();
        back.sort();
        let mut want = keep.clone();
        want.sort();
        assert_eq!(back, want);
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

    /// Both markers have their row bits all ones and a live slot never does,
    /// however the tag comes out, because the table refuses to hold enough rows
    /// to fill twenty four bits.
    #[test]
    fn no_live_slot_can_look_like_a_marker() {
        let mut s = Set::new();
        for i in 0..2000u32 {
            s.insert(format!("m{i}").as_bytes(), ()).expect("room");
        }
        let live = s.slots.iter().filter(|v| **v & ROW != ROW).count();
        assert_eq!(live, s.len());
        assert_eq!(s.dead, 0, "nothing has been removed yet");
        const { assert!(MAX_ROWS < ROW as usize, "a row index is never all ones") }
    }

    /// Every live row is reachable by name and by row index, every slot is one
    /// of the three things a slot may be, and there is always somewhere for a
    /// probe to stop.
    fn check(s: &Set, names: &[Vec<u8>]) {
        assert_eq!(s.len(), names.len());
        let mut live = 0usize;
        let mut dead = 0usize;
        for &slot in &s.slots {
            if slot == EMPTY {
            } else if slot == TOMB {
                dead += 1;
            } else {
                assert!(slot & ROW != ROW, "a slot is live, empty or dead");
                assert!(((slot & ROW) as usize) < s.len(), "a live slot names a row");
                live += 1;
            }
        }
        assert_eq!(live, s.len(), "one live slot per row and no more");
        assert_eq!(dead, s.dead, "the dead count is what is in the array");
        assert!(
            s.len() + s.dead <= s.slots.len() * LOAD_NUM / LOAD_DEN,
            "there is always an empty slot left for a probe to stop at"
        );
        for (i, name) in names.iter().enumerate() {
            assert_eq!(
                s.index_of(name),
                Some(i),
                "{name:?} is not where it was put"
            );
            assert!(s.slot_of(i) < s.slots.len(), "row {i} has no slot");
        }
    }

    #[test]
    fn a_removal_leaves_the_table_whole() {
        let names: Vec<Vec<u8>> = (0..500u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut s = Set::new();
        for name in &names {
            s.insert(name, ()).expect("room");
        }

        // Every third one out, back to front, so the dense row array's swap
        // never moves something that has already been checked.
        let mut live = names.clone();
        for i in (0..live.len()).rev().step_by(3) {
            let gone = live.swap_remove(i);
            assert!(s.remove(&gone).is_some(), "{gone:?} was there");
        }
        check(&s, &live);
        for name in names.iter().filter(|n| !live.contains(n)) {
            assert!(!s.contains(name), "{name:?} came back");
        }
    }

    /// The case the marker count exists for. Without it this loop leaves an
    /// array with no empty slot in it and the next probe never stops.
    #[test]
    fn a_table_churned_in_place_does_not_fill_up_with_markers() {
        let mut s = Set::new();
        for i in 0..1000u32 {
            s.insert(format!("m{i}").as_bytes(), ()).expect("room");
        }
        let slots = s.slots.len();

        for i in 1000..100_000u32 {
            let gone = format!("m{}", i - 1000);
            assert!(s.remove(gone.as_bytes()).is_some());
            s.insert(format!("m{i}").as_bytes(), ()).expect("room");
            assert_eq!(s.len(), 1000);
        }
        assert_eq!(s.slots.len(), slots, "the array is the size it started at");
        let live: Vec<Vec<u8>> = (99_000..100_000u32)
            .map(|i| format!("m{i}").into_bytes())
            .collect();
        for name in &live {
            assert!(s.contains(name), "{name:?} is missing after the churn");
        }
    }

    /// A drain collects most of what it leaves behind.
    ///
    /// A removal from a group that has an empty slot in it clears every marker
    /// in that group, so as a set empties the markers go with it. What is left
    /// at the end is the groups that were completely full when the drain
    /// started, because a probe does go past one of those and a marker there is
    /// still holding the path open. At three quarters full that is about one
    /// group in ten, which is what the bound below is, and it is loose because
    /// it is checking an order of magnitude rather than a formula.
    ///
    /// It used to be none at all, when a removal shifted the run behind it back
    /// instead. That is the price of the group probe and it is a small one: a
    /// few percent of slots holding a marker means almost every group still has
    /// an empty in it, so a miss against the drained set is still one group.
    #[test]
    fn emptying_a_set_a_member_at_a_time_leaves_almost_nothing_behind() {
        let names: Vec<Vec<u8>> = (0..2000u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut s = Set::new();
        for name in &names {
            s.insert(name, ()).expect("room");
        }
        let slots = s.slots.len();
        for name in &names {
            assert!(s.remove(name).is_some(), "{name:?} was there");
        }
        assert!(s.is_empty());
        assert!(
            s.dead * 5 < slots,
            "the drain left {} markers in {slots} slots",
            s.dead
        );
        assert_eq!(
            s.dead,
            s.slots.iter().filter(|v| **v == TOMB).count(),
            "the count is what is actually in the array"
        );
        assert!(s.slots.iter().all(|v| *v == EMPTY || *v == TOMB));
        for name in &names {
            assert!(!s.contains(name), "{name:?} came back");
        }

        // And refilling reuses the array rather than growing past it.
        for name in &names {
            s.insert(name, ()).expect("room");
        }
        check(&s, &names);
        assert_eq!(s.slots.len(), slots, "the array is the size it was");
    }

    /// The invariant the probe bound rests on. Live plus dead never goes up on a
    /// removal, so an unsuccessful probe is never longer after one than before.
    #[test]
    fn a_removal_never_makes_the_array_fuller() {
        let names: Vec<Vec<u8>> = (0..3000u32).map(|i| format!("m{i}").into_bytes()).collect();
        let mut s = Set::new();
        for name in &names {
            s.insert(name, ()).expect("room");
        }
        let mut was = s.len() + s.dead;
        // Out of order, so the runs are broken up rather than eaten from one end.
        for i in (0..names.len()).rev().step_by(7) {
            s.remove(&names[i]).expect("was there");
            let now = s.len() + s.dead;
            assert!(
                now <= was,
                "{now} occupied against {was} before the removal"
            );
            was = now;
        }
    }

    #[test]
    fn a_rebuild_clears_the_markers() {
        let mut s = Set::new();
        for i in 0..1000u32 {
            s.insert(format!("m{i}").as_bytes(), ()).expect("room");
        }
        // Out of order, so most of these leave a marker rather than clearing one.
        for i in (0..1000u32).step_by(2) {
            s.remove(format!("m{i}").as_bytes()).expect("was there");
        }
        assert!(s.dead > 0, "some of those removals left a marker");
        s.grow_to(s.slots.len() * 2);
        assert_eq!(s.dead, 0, "and a rebuild took all of them");
        for i in (1..1000u32).step_by(2) {
            assert!(s.contains(format!("m{i}").as_bytes()));
        }
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
    /// and not an accident. Four bytes of blob offset, one of name length and
    /// three of home slot, with no padding anywhere in it.
    ///
    /// The payload is in an array of its own, so a score costs its eight bytes
    /// and not twelve. That is the whole reason for the split and it is worth a
    /// test, because putting the score back in the row would compile.
    #[test]
    fn a_row_is_eight_bytes_whatever_the_collection_stores() {
        assert_eq!(size_of::<Row>(), 8);
        assert_eq!(size_of::<Row>() + size_of::<()>(), 8, "a set member");
        assert_eq!(size_of::<Row>() + size_of::<f64>(), 16, "a sorted set");
        assert_eq!(size_of::<Row>() + size_of::<u32>(), 12, "a hash field");
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
