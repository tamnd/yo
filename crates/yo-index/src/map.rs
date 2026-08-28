//! The raw map: an index and an arena wired together.
//!
//! This is the smallest thing that is actually a key value store, and it is the
//! thing M0's exit gate measures against aki's `f1raw` numbers. There is no
//! record header yet beyond two lengths, no TTL, no type byte, no version. All
//! of that arrives in M1 and replaces [`Record`] without the index noticing,
//! which is the point of keeping the two crates apart.
//!
//! Layout of one record in the arena:
//!
//! ```text
//! +--------+--------+-----------+-------------+
//! | klen   | vlen   | key bytes | value bytes |
//! | u32 LE | u32 LE | klen      | vlen        |
//! +--------+--------+-----------+-------------+
//! ```
//!
//! Key and value live in one allocation so that a hit is one cache miss for the
//! bucket and one for the record, not three.

use crate::index::{Index, Keys};
use yo_arena::Arena;
use yo_common::{Addr, Space, wyhash};

/// Bytes of length prefix in front of a record.
const HDR: usize = 8;

struct Record;

impl Record {
    #[inline]
    fn lens(bytes: &[u8]) -> (usize, usize) {
        let k = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let v = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        (k, v)
    }
}

/// Arena backed record access, which is what the index probes through.
struct Records<'a> {
    arena: &'a Arena,
}

impl Keys for Records<'_> {
    #[inline]
    fn hash_at(&self, addr: Addr) -> u64 {
        let (klen, _) = Record::lens(self.arena.get(addr, HDR));
        let bytes = self.arena.get(addr, HDR + klen);
        wyhash(&bytes[HDR..], 0)
    }

    #[inline]
    fn eq_at(&self, addr: Addr, key: &[u8]) -> bool {
        let bytes = self.arena.get(addr, HDR);
        let (klen, _) = Record::lens(bytes);
        if klen != key.len() {
            return false;
        }
        let bytes = self.arena.get(addr, HDR + klen);
        &bytes[HDR..] == key
    }
}

/// A single shard's key value map: bytes in, bytes out, nothing else.
///
/// Not `Sync`, and deliberately so. One of these belongs to one shard thread
/// and is reached through `ShardLocal`, which is `05` section 1's whole
/// argument: single ownership means no atomics on the hot path.
///
/// ```
/// let mut m = yo_index::RawMap::new();
/// assert_eq!(m.set(b"k", b"v"), None);
/// assert_eq!(m.get(b"k"), Some(&b"v"[..]));
/// assert_eq!(m.set(b"k", b"w").is_some(), true);
/// assert_eq!(m.get(b"k"), Some(&b"w"[..]));
/// assert_eq!(m.del(b"k"), true);
/// assert_eq!(m.get(b"k"), None);
/// ```
pub struct RawMap {
    index: Index,
    arena: Arena,
}

impl RawMap {
    /// An empty map.
    pub fn new() -> RawMap {
        RawMap {
            index: Index::new(),
            arena: Arena::new(),
        }
    }

    /// How many keys are stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Throw everything away and give the memory back.
    ///
    /// A fresh index and a fresh arena rather than a walk that deletes each key
    /// in turn. Deleting one at a time would leave an arena the size of the
    /// data that used to be in it and an index still grown to fit it, and the
    /// one thing a client that has just said `FLUSHALL` is entitled to expect is
    /// the memory back.
    pub fn clear(&mut self) {
        *self = RawMap::new();
    }

    /// The hash this map files `key` under.
    ///
    /// Public because the batch walk in `04` section 3 hashes on the first walk
    /// and looks up on the second, and the alternative is hashing every key
    /// twice to keep the seed a private detail.
    #[inline]
    #[must_use]
    pub fn hash_of(key: &[u8]) -> u64 {
        wyhash(key, 0)
    }

    /// Ask the cache for the bucket `hash` will be looked up in.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        self.index.prefetch(hash);
    }

    /// The value stored under `key`.
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.get_hashed(Self::hash_of(key), key)
    }

    /// The value stored under `key`, whose hash the caller already has.
    ///
    /// The second walk's entry point. `hash` has to be [`RawMap::hash_of`] of
    /// this key: a hash from somewhere else is not unsafe, it just misses.
    #[inline]
    pub fn get_hashed(&self, hash: u64, key: &[u8]) -> Option<&[u8]> {
        let addr = self.index.get(hash, key, &Records { arena: &self.arena })?;
        Some(self.value_at(addr))
    }

    #[inline]
    fn value_at(&self, addr: Addr) -> &[u8] {
        let (klen, vlen) = Record::lens(self.arena.get(addr, HDR));
        &self.arena.get(addr, HDR + klen + vlen)[HDR + klen..]
    }

    /// The value stored under `key`, to be overwritten where it lies.
    ///
    /// The length cannot change, which is the whole reason this is safe to
    /// offer. `INCR` on an integer encoded string is a probe, an add and a
    /// store, and the store is eight bytes back into the record it came from
    /// (`08` section 2). Going through [`RawMap::set`] instead would write a
    /// fresh record and free the old one on every increment, which is an arena
    /// append and a dead byte per operation for a value whose size never moves.
    ///
    /// There is no reader to tear. A map belongs to one shard thread and is not
    /// `Sync`, so the only code that can observe a half written value is the
    /// code doing the writing. When a replica stream or a snapshot reader starts
    /// walking the arena from another thread, this becomes an epoch question and
    /// the write becomes an install rather than an overwrite.
    #[inline]
    pub fn value_mut(&mut self, key: &[u8]) -> Option<&mut [u8]> {
        self.value_mut_hashed(Self::hash_of(key), key)
    }

    /// [`RawMap::value_mut`] for a caller that already hashed the key.
    #[inline]
    pub fn value_mut_hashed(&mut self, hash: u64, key: &[u8]) -> Option<&mut [u8]> {
        let addr = self.index.get(hash, key, &Records { arena: &self.arena })?;
        let (klen, vlen) = Record::lens(self.arena.get(addr, HDR));
        Some(&mut self.arena.get_mut(addr, HDR + klen + vlen)[HDR + klen..])
    }

    /// Store `val` under `key`, returning the length of the value it replaced.
    pub fn set(&mut self, key: &[u8], val: &[u8]) -> Option<usize> {
        self.set_with(key, val.len(), |buf| buf.copy_from_slice(val))
    }

    /// The largest record this map can store, key and value and header together.
    ///
    /// A value past this belongs in the log region rather than the arena, which
    /// is `06` section 2's business and not this crate's.
    #[inline]
    #[must_use]
    pub const fn max_record() -> usize {
        yo_arena::MAX_ALLOC
    }

    /// Bytes of record header in front of the key.
    #[inline]
    #[must_use]
    pub const fn header_len() -> usize {
        HDR
    }

    /// Store a `vlen` byte value under `key`, written by `fill`.
    ///
    /// The same thing [`RawMap::set`] does, except that the caller writes
    /// straight into the record instead of building the value somewhere else
    /// first and having it copied in. A string with a one byte encoding tag in
    /// front of it would otherwise be assembled in a scratch buffer and then
    /// memcpy'd again, and two copies for one `SET` is one too many on a path
    /// that is trying to be ten times faster than Redis.
    ///
    /// `fill` is handed exactly `vlen` bytes of uninitialised-looking storage.
    /// It is arena memory that has been handed out before and freed, so its
    /// contents are arbitrary and every byte of it must be written.
    ///
    /// # Panics
    ///
    /// If the whole record would exceed [`RawMap::max_record`].
    pub fn set_with<F>(&mut self, key: &[u8], vlen: usize, fill: F) -> Option<usize>
    where
        F: FnOnce(&mut [u8]),
    {
        assert!(key.len() <= u32::MAX as usize, "key too long");
        assert!(vlen <= u32::MAX as usize, "value too long");
        let total = HDR + key.len() + vlen;
        let h = wyhash(key, 0);

        // A key that is already here, in a record exactly the size the new value
        // needs, is written over where it lies. No allocation, no dead bytes, no
        // index write, and nothing for compaction to collect later.
        //
        // This used to say the in place path had to wait for epochs, because a
        // reader that had already resolved the address would see a torn value.
        // That was never a rule this map kept: `value_mut` is the same write and
        // `INCR` has been doing it since the day it was written, for the same
        // reason given there. A map belongs to one shard thread and is not
        // `Sync`, so the only code that can see a half written value is the code
        // writing it. When a replica stream or a snapshot reader starts walking
        // the arena from another thread, both of these become an install rather
        // than an overwrite, together.
        //
        // Exactly the size and not merely small enough. A shorter value in a
        // longer record would leave the header disagreeing with the space the
        // record occupies, and compaction walks a segment by stepping over each
        // record by the length in its header, so the walk would land in the
        // middle of the next one.
        //
        // Overwriting a key with a value the same size as the last one is what
        // half of the world's caches do, and it is what every SET benchmark
        // does. On gamingpc it was 25 percent of SET throughput at pipeline 16
        // and 37 percent of MSET, all of it spent making garbage and then
        // collecting it.
        if let Some(addr) = self.index.get(h, key, &Records { arena: &self.arena }) {
            let (klen, old_vlen) = Record::lens(self.arena.get(addr, HDR));
            debug_assert_eq!(klen, key.len(), "the index matched a different key");
            if old_vlen == vlen {
                let rec = self.arena.get_mut(addr, total);
                fill(&mut rec[HDR + klen..]);
                return Some(vlen);
            }
        }

        let (addr, buf) = self
            .arena
            .alloc(total)
            .expect("record is larger than a segment");
        buf[0..4].copy_from_slice(&(key.len() as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(vlen as u32).to_le_bytes());
        // The arena hands back a run padded up to its alignment, so index to
        // `total` rather than to the end of the slice.
        buf[HDR..HDR + key.len()].copy_from_slice(key);
        fill(&mut buf[HDR + key.len()..total]);

        let old = {
            let recs = Records { arena: &self.arena };
            self.index.insert(h, key, addr, &recs)
        };
        match old {
            Some(prev) => {
                let (pk, pv) = Record::lens(self.arena.get(prev, HDR));
                self.arena.free(prev, HDR + pk + pv);
                Some(pv)
            }
            None => None,
        }
    }

    /// Remove `key`, returning whether it was there.
    pub fn del(&mut self, key: &[u8]) -> bool {
        let h = wyhash(key, 0);
        let addr = {
            let recs = Records { arena: &self.arena };
            self.index.remove(h, key, &recs)
        };
        match addr {
            Some(a) => {
                let (k, v) = Record::lens(self.arena.get(a, HDR));
                self.arena.free(a, HDR + k + v);
                true
            }
            None => false,
        }
    }

    /// Whether `key` is present.
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        let h = wyhash(key, 0);
        self.index.contains(h, key, &Records { arena: &self.arena })
    }

    /// The index, for stats and for compaction.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// The arena, for stats and for compaction.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Bytes held by index structure plus arena segments.
    pub fn memory_bytes(&self) -> usize {
        self.index.memory_bytes() + self.arena.reserved_bytes() as usize
    }

    /// Move every live record out of `seg` and into the current segment, then
    /// put the segment back on the arena's free list.
    ///
    /// Copy, rewrite the index entry, done. No forwarding pointers and no read
    /// barrier, which is the F2 shape from `05` section 3.2 and is what an
    /// allocation having exactly one referent buys.
    ///
    /// The walk is over the segment and not over the index. Both find the same
    /// records, and the index walk is the one written in the spec, but it reads
    /// the whole index to compact two megabytes: fine when this only ran in a
    /// test, wrong once the event loop calls it, because the pause would then
    /// grow with the size of the database rather than with the size of a
    /// segment. Walking the segment costs one index probe per record in it and
    /// does not care how many keys exist elsewhere.
    ///
    /// Records sit back to back from the header to the segment's bump, each one
    /// rounded up to the arena's alignment, and every arena allocation is a
    /// record, so the next one is always a known distance away. A record is
    /// live when the index still points at this copy of it, and dead when it
    /// points somewhere else or at nothing, which is exactly what an overwrite
    /// and a delete leave behind.
    ///
    /// The reclaim at the end is the part that makes the space usable again.
    /// Moving the records out only makes a segment empty, and an empty segment
    /// that nothing ever bumps through again is still two megabytes the process
    /// is holding.
    pub fn compact_segment(&mut self, seg: usize) -> usize {
        if seg == self.arena.current_segment() {
            // Its bump is a cursor, not a checkpoint, and reclaiming it would
            // take the ground out from under the next allocation.
            return 0;
        }
        let base = (seg as u64) << yo_arena::SEGMENT_SHIFT;
        let bump = self.arena.recorded_bump(seg) as usize;

        let mut moved = 0;
        let mut off = yo_arena::HEADER_SIZE;
        while off < bump {
            let old = Addr::new(Space::Arena, base + off as u64);
            let (klen, vlen) = Record::lens(self.arena.get(old, HDR));
            let total = HDR + klen + vlen;
            off += total.next_multiple_of(yo_arena::ALIGN);

            let hash = {
                let bytes = self.arena.get(old, HDR + klen);
                wyhash(&bytes[HDR..], 0)
            };
            let live = {
                let bytes = self.arena.get(old, HDR + klen);
                let key = &bytes[HDR..];
                let recs = Records { arena: &self.arena };
                self.index.get(hash, key, &recs) == Some(old)
            };
            if !live {
                continue;
            }

            let new = self.arena.copy_within(old, total);
            let bytes = self.arena.get(new, HDR + klen);
            let key = &bytes[HDR..];
            let recs = Records { arena: &self.arena };
            let ok = self.index.relocate(hash, key, new, &recs);
            debug_assert!(ok, "compaction lost an entry the index just handed us");
            self.arena.free(old, total);
            moved += 1;
        }
        self.arena.reclaim(seg);
        moved
    }

    /// Compact one segment if one is worth compacting, and say how many records
    /// moved.
    ///
    /// `None` means there was no candidate. It is not the same as `Some(0)`,
    /// which is a segment whose every record had already been overwritten: that
    /// one gave two megabytes back without moving anything, and a caller
    /// deciding whether to go round again needs to be told so.
    ///
    /// This is the whole maintenance contract: at most one segment per call, so
    /// a caller that runs it once per turn of the loop pays a bounded amount
    /// and never a full pass over the arena. Zero means there was nothing to
    /// do, and finding that out is one comparison against the running dead byte
    /// total.
    pub fn compact_step(&mut self) -> Option<usize> {
        let seg = self.arena.worst_candidate()?;
        Some(self.compact_segment(seg))
    }
}

impl Default for RawMap {
    fn default() -> RawMap {
        RawMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `key:` and the index zero padded to twelve digits.
    ///
    /// Written out by hand rather than with `format!`, which produces the same
    /// bytes. Formatting is a lot of machinery for twelve digits, and Miri pays
    /// per operation rather than per instruction, so under the interpreter one
    /// `format!` costs a couple of milliseconds. `grows_through_many_splits`
    /// calls this once per set, get, delete and contains, which is ten thousand
    /// calls on its own, and that is twenty seconds of the ninety five this
    /// crate's Miri shard used to take.
    fn key(i: usize) -> Vec<u8> {
        let mut k = *b"key:000000000000";
        let mut n = i;
        let mut p = k.len() - 1;
        while n > 0 {
            k[p] = b'0' + (n % 10) as u8;
            n /= 10;
            p -= 1;
        }
        k.to_vec()
    }

    /// `v` and the index, unpadded, which is what `format!("v{i}")` gives.
    fn val(i: usize) -> Vec<u8> {
        let mut v = vec![b'v'];
        if i == 0 {
            v.push(b'0');
            return v;
        }
        let start = v.len();
        let mut n = i;
        while n > 0 {
            v.push(b'0' + (n % 10) as u8);
            n /= 10;
        }
        v[start..].reverse();
        v
    }

    // Miri is a few hundred times slower than the machine, so the counts below
    // shrink under it. They stay large enough to force directory doublings,
    // segment splits and overflow chains, which is what these tests are for.
    // Only the scale goes away, not the coverage.
    // Three thousand and not fewer. `splits() > 4` is the assertion and the
    // splits go 1, 1, 2, 3, 3, 5 at 800, 1200, 1500, 2000, 2500 and 3000 keys,
    // so this is already the smallest count that grows the directory the number
    // of times the test asks about.
    #[cfg(miri)]
    const GROW_N: usize = 3_000;
    #[cfg(not(miri))]
    const GROW_N: usize = 200_000;

    #[cfg(miri)]
    const ADVERSARIAL_N: u64 = 1_000;
    #[cfg(not(miri))]
    const ADVERSARIAL_N: u64 = 50_000;

    // Big values so that a handful of records fills a 2 MiB segment and
    // compaction has something to do without a hundred thousand writes.
    #[cfg(miri)]
    const COMPACT_VAL: usize = 65_536;
    #[cfg(miri)]
    const COMPACT_N: usize = 200;
    #[cfg(not(miri))]
    const COMPACT_VAL: usize = 1024;
    #[cfg(not(miri))]
    const COMPACT_N: usize = 8_000;

    #[test]
    fn set_get_del() {
        let mut m = RawMap::new();
        assert!(m.is_empty());
        assert_eq!(m.set(b"a", b"1"), None);
        assert_eq!(m.get(b"a"), Some(&b"1"[..]));
        assert_eq!(m.len(), 1);
        assert_eq!(m.set(b"a", b"22"), Some(1));
        assert_eq!(m.get(b"a"), Some(&b"22"[..]));
        assert_eq!(m.len(), 1);
        assert!(m.del(b"a"));
        assert!(!m.del(b"a"));
        assert_eq!(m.get(b"a"), None);
        assert!(m.is_empty());
    }

    #[test]
    fn a_value_can_be_overwritten_where_it_lies() {
        let mut m = RawMap::new();
        m.set(b"n", &7u64.to_le_bytes());
        m.set(b"other", b"untouched");
        let before = m.arena().live_bytes();

        let v = m.value_mut(b"n").expect("the key is there");
        v.copy_from_slice(&8u64.to_le_bytes());

        assert_eq!(m.get(b"n"), Some(&8u64.to_le_bytes()[..]));
        assert_eq!(m.get(b"other"), Some(&b"untouched"[..]));
        // The point of the whole method: no second record and nothing dead.
        assert_eq!(m.arena().live_bytes(), before);
        assert_eq!(m.len(), 2);

        assert!(m.value_mut(b"missing").is_none());
    }

    /// A key overwritten with a value the same size stays in the record it is
    /// already in, and one overwritten with a different size does not.
    ///
    /// The first is the shape every SET benchmark and half the world's caches
    /// have: the same keys, the same value size, over and over. Writing a fresh
    /// record for each of those makes a dead one to go with it, and compaction
    /// then spends a quarter of the server's write throughput copying live
    /// records out from between them.
    #[test]
    fn an_overwrite_of_the_same_size_makes_no_garbage() {
        let mut m = RawMap::new();
        m.set(b"k", b"12345678");
        m.set(b"other", b"untouched");
        let live = m.arena().live_bytes();
        let dead = m.arena().dead_bytes_total();

        for i in 0..1000u32 {
            let v = format!("{i:08}");
            assert_eq!(m.set(b"k", v.as_bytes()), Some(8));
        }

        assert_eq!(m.get(b"k"), Some(&b"00000999"[..]));
        assert_eq!(m.get(b"other"), Some(&b"untouched"[..]));
        assert_eq!(m.len(), 2);
        assert_eq!(m.arena().live_bytes(), live, "a thousand writes, no growth");
        assert_eq!(m.arena().dead_bytes_total(), dead, "and nothing dead");

        // A different length cannot go in the same hole, because the record has
        // to be as long as its header says it is.
        assert_eq!(m.set(b"k", b"123456789"), Some(8));
        assert_eq!(m.get(b"k"), Some(&b"123456789"[..]));
        assert!(
            m.arena().dead_bytes_total() > dead,
            "the old record is dead"
        );
    }

    /// An expiring value and a plain one are different record lengths, so the
    /// one does not get written over the other.
    ///
    /// This is the case the in place path has to refuse rather than the case it
    /// is for, and it is the one that would corrupt a record if it took it: the
    /// value here is a `Strings` record, whose deadline is inside the value, so
    /// two values of the same visible length are two different record lengths.
    #[test]
    fn a_longer_value_moves_and_the_index_follows_it() {
        let mut m = RawMap::new();
        m.set(b"k", b"aaaa");
        let first = m
            .index()
            .get(RawMap::hash_of(b"k"), b"k", &Records { arena: m.arena() });

        m.set(b"k", b"aaaaaaaa");
        let second = m
            .index()
            .get(RawMap::hash_of(b"k"), b"k", &Records { arena: m.arena() });

        assert_ne!(first, second, "a longer value needs a new record");
        assert_eq!(m.get(b"k"), Some(&b"aaaaaaaa"[..]));
    }

    #[test]
    fn empty_key_and_empty_value() {
        let mut m = RawMap::new();
        m.set(b"", b"");
        assert_eq!(m.get(b""), Some(&b""[..]));
        m.set(b"x", b"");
        assert_eq!(m.get(b"x"), Some(&b""[..]));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn grows_through_many_splits() {
        let mut m = RawMap::new();
        const N: usize = GROW_N;
        for i in 0..N {
            m.set(&key(i), &val(i));
        }
        assert_eq!(m.len(), N);
        assert!(
            m.index().splits() > 4,
            "expected real growth, saw {} splits",
            m.index().splits()
        );
        for i in 0..N {
            assert_eq!(
                m.get(&key(i)),
                Some(val(i).as_slice()),
                "lost key {i} after {} splits",
                m.index().splits()
            );
        }
        for i in (0..N).step_by(3) {
            assert!(m.del(&key(i)), "delete missed key {i}");
        }
        for i in 0..N {
            assert_eq!(
                m.contains(&key(i)),
                i % 3 != 0,
                "wrong presence for key {i}"
            );
        }
    }

    #[test]
    fn compaction_preserves_everything() {
        let mut m = RawMap::new();
        // Enough to fill several arena segments with 1 KiB values.
        let val = vec![b'z'; COMPACT_VAL];
        const N: usize = COMPACT_N;
        for i in 0..N {
            m.set(&key(i), &val);
        }
        // Kill half, which pushes the early segments over the dead ratio.
        for i in (0..N).step_by(2) {
            m.del(&key(i));
        }
        let candidates = m.arena().compaction_candidates();
        assert!(
            !candidates.is_empty(),
            "expected at least one segment past the dead ratio"
        );
        for seg in candidates {
            m.compact_segment(seg);
        }
        for i in 0..N {
            let want = if i % 2 == 0 { None } else { Some(val.clone()) };
            assert_eq!(m.get(&key(i)).map(|v| v.to_vec()), want, "key {i}");
        }
    }

    /// The bug this exists for: overwriting a key writes a new record and only
    /// counts the old one dead, so without compaction a server that rewrites
    /// the same keys holds every version of every one of them forever. Measured
    /// on a real server before this, 400000 sets over 100000 keys came to 742
    /// bytes a key for 64 byte values.
    #[test]
    fn rewriting_the_same_keys_stops_growing() {
        let mut m = RawMap::new();
        let val = vec![b'z'; COMPACT_VAL];
        const N: usize = COMPACT_N;

        for i in 0..N {
            m.set(&key(i), &val);
            m.compact_step();
        }
        let after_first_pass = m.arena().reserved_bytes();

        // Nine more passes over the same keys, writing the same amount of data
        // nine more times and keeping exactly as much of it.
        for _ in 0..9 {
            for i in 0..N {
                m.set(&key(i), &val);
                m.compact_step();
            }
        }
        let after_ten = m.arena().reserved_bytes();

        assert!(
            after_ten <= after_first_pass * 2,
            "held {after_ten} after ten passes against {after_first_pass} after one, \
             which is the grow forever shape"
        );
        assert!(
            after_ten < m.arena().live_bytes() * 2,
            "held {after_ten} for {} live, which is more than the ratio allows",
            m.arena().live_bytes()
        );
        for i in 0..N {
            assert_eq!(
                m.get(&key(i)).map(<[u8]>::to_vec),
                Some(val.clone()),
                "key {i}"
            );
        }
    }

    /// A segment that compaction emptied is bumped through again rather than
    /// sitting there holding two megabytes.
    #[test]
    fn an_emptied_segment_is_used_again() {
        let mut m = RawMap::new();
        let val = vec![b'z'; COMPACT_VAL];
        const N: usize = COMPACT_N;
        for i in 0..N {
            m.set(&key(i), &val);
        }
        for i in (0..N).step_by(2) {
            m.del(&key(i));
        }

        let before = m.arena().segment_count();
        let seg = m.arena().worst_candidate().expect("half of it is dead");
        m.compact_segment(seg);
        assert_eq!(
            m.arena().free_segments(),
            1,
            "the segment did not come back"
        );

        // Write until the free segment has to be taken, and the count is where
        // it was rather than one higher.
        for i in N..N * 2 {
            m.set(&key(i), &val);
            if m.arena().free_segments() == 0 {
                break;
            }
        }
        assert_eq!(
            m.arena().segment_count(),
            before,
            "asked the system for memory while holding an empty segment"
        );
    }

    #[test]
    fn adversarial_keys_that_share_low_bits() {
        // Keys chosen so that many land in the same bucket index. The point is
        // that overflow chaining and splitting both still work when the hash is
        // not being kind.
        let mut m = RawMap::new();
        let mut inserted = Vec::new();
        for i in 0..ADVERSARIAL_N {
            let k = i.to_le_bytes().to_vec();
            m.set(&k, b"v");
            inserted.push(k);
        }
        for k in &inserted {
            assert_eq!(m.get(k), Some(&b"v"[..]));
        }
        assert_eq!(m.len(), inserted.len());
    }
}
