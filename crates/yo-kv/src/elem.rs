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
//! slot it wanted. Three bytes a row for that, packed in beside the name length,
//! and it took a pop at a hundred thousand members from 123 ns to 25.7 ns, which
//! is the difference between a random trip into the name blob per slot examined
//! and no trip at all.
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

/// How many bits of a name's hash a row keeps to say where it wanted to sit.
const HOME_BITS: u32 = 24;

/// Those bits on their own.
const HOME_MASK: usize = (1 << HOME_BITS) - 1;

/// The hash bits a row keeps.
#[inline(always)]
const fn home_bits(h: u64) -> usize {
    (h as usize) & HOME_MASK
}

/// Where a name lands in a table of `m` slots.
///
/// A multiply and a shift rather than a mask, which is what lets `m` be the size
/// the table actually needs instead of the next power of two above it. See the
/// note on [`slots_for`] for what that is worth, and it is worth a lot: at eight
/// hundred thousand members a masked table holds two million and ninety seven
/// thousand slots to serve one million and eighty thousand, and eight bytes a
/// member of the ten it spends are air.
///
/// The three cycles this costs over an `and` land in front of a load that misses
/// cache, which is the only reason the trade is available at all.
#[inline(always)]
const fn slot_of(bits: usize, m: usize) -> usize {
    ((bits as u64 * m as u64) >> HOME_BITS) as usize
}

/// How far forward from `from` to `to`, the long way round if it has to be.
///
/// The modular distance a mask used to give for nothing. Both arguments are
/// already inside the table, so this is a subtract and a conditional add.
#[inline(always)]
const fn ahead(from: usize, to: usize, m: usize) -> usize {
    if to >= from { to - from } else { to + m - from }
}

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
/// Twenty four bits is enough of the hash at every table size, because
/// [`slot_of`] scales those bits across the table rather than masking them down
/// to it. Every placement in this file goes through that one function, so the
/// answer is the same one the insert gave whatever the table is. Above sixteen
/// million slots the twenty four bits stop naming every slot and some of them are
/// never anybody's home, which costs a little clustering on a table holding
/// twelve million elements and nothing at all below it.
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
        let bits = u32::try_from(home_bits(h)).expect("HOME_BITS is under 32");
        Row {
            at,
            packed: (bits << 8) | len,
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
    /// Tag in the top byte, row index in the low 24 bits, [`EMPTY`] for nothing.
    slots: Box<[u32]>,
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
    /// The length stays in the row rather than in a [`crate::blob::Span`],
    /// because one byte of it is what keeps a row at eight bytes, and the names
    /// that do not fit in one byte carry their own length in the blob instead of
    /// widening every row that does.
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
        if n * LOAD_DEN > self.slots.len() * LOAD_NUM {
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
    #[inline]
    fn find_hashed(&self, h: u64, name: &[u8]) -> Option<usize> {
        if self.rows.is_empty() {
            return None;
        }
        let m = self.slots.len();
        let tag = tag_of(h);
        let mut at = slot_of(home_bits(h), m);
        loop {
            let slot = self.slots[at];
            if slot == EMPTY {
                return None;
            }
            if slot >> 24 == u32::from(tag) {
                let row = (slot & 0x00FF_FFFF) as usize;
                if bytes_eq(self.name_of(&self.rows[row]), name) {
                    return Some(row);
                }
            }
            at += 1;
            if at == m {
                at = 0;
            }
        }
    }

    /// Put a row index in the first slot the probe reaches.
    fn put_slot(&mut self, h: u64, row: u32) {
        let m = self.slots.len();
        let mut at = slot_of(home_bits(h), m);
        while self.slots[at] != EMPTY {
            at += 1;
            if at == m {
                at = 0;
            }
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

    /// Close the slot holding `row`, and shift the run behind it back.
    ///
    /// Linear probing means a run of occupied slots is a chain, and punching a
    /// hole in the middle of one hides everything behind it. Shifting the run
    /// back keeps every probe correct and leaves no tombstone, which matters
    /// because `SPOP` in a loop deletes every element a set ever had.
    fn clear_slot(&mut self, row: usize) {
        let m = self.slots.len();
        let mut at = self.home_of(row, m);
        loop {
            let slot = self.slots[at];
            debug_assert!(slot != EMPTY, "the row being removed has a slot");
            if slot != EMPTY && (slot & 0x00FF_FFFF) as usize == row {
                break;
            }
            at += 1;
            if at == m {
                at = 0;
            }
        }
        self.slots[at] = EMPTY;

        // Everything behind the hole that probed past it has to move up, or it
        // becomes unreachable.
        let mut hole = at;
        let mut scan = if at + 1 == m { 0 } else { at + 1 };
        while self.slots[scan] != EMPTY {
            let slot = self.slots[scan];
            let idx = (slot & 0x00FF_FFFF) as usize;
            let home = self.home_of(idx, m);
            // True when the slot's home is at or behind the hole, meaning it
            // probed over the hole to get here and would not be found now.
            if ahead(home, scan, m) >= ahead(hole, scan, m) {
                self.slots[hole] = slot;
                self.slots[scan] = EMPTY;
                hole = scan;
            }
            scan += 1;
            if scan == m {
                scan = 0;
            }
        }
    }

    /// Point the slot holding `from` at `to` instead.
    fn repoint(&mut self, from: usize, to: usize) {
        let m = self.slots.len();
        let mut at = self.home_of(from, m);
        loop {
            let slot = self.slots[at];
            debug_assert!(slot != EMPTY, "the row being moved has a slot");
            if slot != EMPTY && (slot & 0x00FF_FFFF) as usize == from {
                self.slots[at] =
                    (slot & 0xFF00_0000) | u32::try_from(to).expect("a row index fits in 24 bits");
                return;
            }
            at += 1;
            if at == m {
                at = 0;
            }
        }
    }

    /// Make sure there is room for one more before it is inserted.
    ///
    /// Both arrays grow by [`crate::grow`]'s policy rather than by `Vec`'s or by
    /// doubling, because on a large collection the slack in either of them is
    /// the largest piece of memory nobody asked for in the whole structure.
    ///
    /// [`slots_for`] says the smallest table that would hold `want` at the load
    /// factor, which at the moment this fires is the table there already, and
    /// then the growth policy is what actually decides the step. So the load
    /// runs between three fifths and three quarters rather than between three
    /// eighths and three quarters, and the slot array is rebuilt more often for
    /// a smaller step each time.
    fn reserve_one(&mut self) {
        let want = self.rows.len() + 1;
        crate::grow::reserve(&mut self.rows, 1);
        crate::grow::reserve(&mut self.vals, 1);
        if want * LOAD_DEN > self.slots.len() * LOAD_NUM {
            let need = slots_for(want);
            let next = (self.slots.len() + self.slots.len() / 2).max(need);
            self.grow_to(next);
        }
    }

    /// Rebuild the slot array at a new size.
    ///
    /// The rows do not move and the names do not move. Only the slots are, and
    /// they are rebuilt from the old slot array rather than from the names: the
    /// tag is already in the old slot and the home is already in the row, so a
    /// growth reads two flat arrays and hashes nothing.
    fn grow_to(&mut self, slots: usize) {
        let m = slots.max(MIN_SLOTS);
        let old = std::mem::replace(&mut self.slots, vec![EMPTY; m].into_boxed_slice());
        for &slot in &old {
            if slot == EMPTY {
                continue;
            }
            let row = (slot & 0x00FF_FFFF) as usize;
            let mut at = self.home_of(row, m);
            while self.slots[at] != EMPTY {
                at += 1;
                if at == m {
                    at = 0;
                }
            }
            self.slots[at] = slot;
        }
    }

    /// Where the row at `idx` wanted to sit, in a table of `m` slots.
    ///
    /// A field of a row this was going to read anyway, scaled across the table
    /// by the one function every placement in here goes through.
    #[inline(always)]
    fn home_of(&self, idx: usize, m: usize) -> usize {
        slot_of(self.rows[idx].home(), m)
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
///
/// The number itself and not the next power of two above it, which is the whole
/// point of [`slot_of`]. Rounding up to a power of two costs on average a third
/// of the slot array and at the worst point half of it, and the worst point is
/// not rare: eight hundred thousand members want one million and eighty thousand
/// slots and a masked table gives them two million and ninety seven thousand, so
/// the array sits at a load of 0.39 holding eight bytes a member of air.
fn slots_for(n: usize) -> usize {
    ((n * LOAD_DEN) / LOAD_NUM + 1).max(MIN_SLOTS)
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
        assert_eq!(
            size_of::<Row>() + size_of::<crate::blob::Span>(),
            16,
            "a hash field"
        );
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
