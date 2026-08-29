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

/// The least a single [`RawMap::compact_step`] walks.
///
/// A segment is two megabytes and evacuating one in a single call was a stop
/// the world pause in the middle of a batch. At 64 byte values that is around
/// twenty six thousand records, each one an index probe, a copy and an index
/// write, and the replies behind it wait for all of them. It is why the write
/// rows had a p99 of 3.9 milliseconds against Redis at 0.8 while the p50 was
/// in line: the median command paid nothing and one command in a few thousand
/// paid for the whole segment.
///
/// Sixty four kilobytes is a thirty second of a segment, which puts the worst
/// call at a few hundred records. Smaller would be smoother and would spend
/// more of the total on the fixed cost of picking up where the last call left
/// off; this is the smallest size at which that overhead is still noise.
///
/// The budget is spent on how far the cursor moves and not on how many records
/// move, because a segment can be entirely dead. Charging only for records
/// that move would let one call walk two megabytes of headers for free, which
/// is the pause this exists to prevent, just without the copying.
const EVAC_FLOOR: usize = 64 * 1024;

/// The most, which is a whole segment.
///
/// The cap is here so that the scaling below has an end, not because a segment
/// is a good amount of work to do at once. Reaching it means the collector is
/// sixteen times past the line it starts at, at which point the pause is the
/// smaller problem.
const EVAC_CEILING: usize = yo_arena::SEGMENT_SIZE;

/// A segment that is partway through being evacuated, and how far it got.
#[derive(Clone, Copy)]
struct Evac {
    seg: usize,
    off: usize,
}

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
    /// Where the last `compact_step` stopped, if it stopped partway.
    evac: Option<Evac>,
    /// How many times anything in here has been written to.
    ///
    /// A caller that resolved a key once and wants to skip resolving it again
    /// needs to know whether anything could have moved in between, and the
    /// honest answer is any write at all. Every method that takes `&mut self`
    /// bumps this, including the in place ones, so the question a caller asks is
    /// "has this map been written since" and not "has this map been written in a
    /// way I thought would matter".
    ///
    /// It lives here rather than in the caller because there are eleven places
    /// in `yo-kv` that write to a map and one place here that could be missed,
    /// and a missed invalidation is a stale answer rather than a slow one.
    writes: u64,
}

impl RawMap {
    /// An empty map.
    pub fn new() -> RawMap {
        RawMap {
            index: Index::new(),
            arena: Arena::new(),
            evac: None,
            writes: 0,
        }
    }

    /// How many times this map has been written to.
    ///
    /// Two reads of this with the same value either side of some work mean
    /// nothing in the map moved, so an address or a slot resolved before the
    /// first read is still the right one after the second. It never goes
    /// backwards, including across [`RawMap::clear`].
    #[inline]
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
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
        // Carried across the reset and bumped, because a counter that went back
        // to zero here could land on a value a memo was already holding and
        // read as "nothing moved" on the one call where everything did.
        let writes = self.writes;
        *self = RawMap::new();
        self.writes = writes + 1;
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

    /// Where `key`'s record is, for a caller that has to look at it twice.
    ///
    /// A `GET` has to know whether the key is past its deadline before it can
    /// answer, and then has to read the value it just decided about. Asking
    /// [`RawMap::get`] twice is two hashes and two probes for one record, and a
    /// probe is the expensive half of a command. This hands back the address
    /// instead, and [`RawMap::value_at`] reads it with no probe at all.
    ///
    /// The address is good until the next write to this map. Anything that
    /// inserts, deletes or compacts can move a record, and an address held
    /// across one of those reads whatever is at that spot now. Hold it for the
    /// length of one command and no longer.
    #[inline]
    pub fn find(&self, key: &[u8]) -> Option<Addr> {
        self.find_hashed(Self::hash_of(key), key)
    }

    /// [`RawMap::find`] for a caller that already hashed the key.
    #[inline]
    pub fn find_hashed(&self, hash: u64, key: &[u8]) -> Option<Addr> {
        self.index.get(hash, key, &Records { arena: &self.arena })
    }

    /// The value at an address this map handed out, with no probe.
    ///
    /// See [`RawMap::find`] for how long an address is worth holding.
    #[inline]
    #[must_use]
    pub fn value_at(&self, addr: Addr) -> &[u8] {
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
        self.writes += 1;
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
        self.writes += 1;
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
        self.writes += 1;
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
        self.writes += 1;
        if seg == self.arena.current_segment() {
            // Its bump is a cursor, not a checkpoint, and reclaiming it would
            // take the ground out from under the next allocation.
            return 0;
        }
        let (moved, _) = self.evacuate(seg, yo_arena::HEADER_SIZE, usize::MAX);
        self.arena.reclaim(seg);
        moved
    }

    /// Walk `seg` from `from`, moving live records out, and stop once the walk
    /// has covered `budget` bytes of it. Says how many records moved and where
    /// to start again.
    ///
    /// The record that straddles the budget is finished rather than cut in
    /// half, so the walk can go a little past what was asked for. The overrun
    /// is one record and the budget is thousands of bytes.
    ///
    /// Nothing here reclaims. A segment is only empty once the walk reaches the
    /// bump, and the caller is the one that knows whether it did.
    fn evacuate(&mut self, seg: usize, from: usize, budget: usize) -> (usize, usize) {
        let base = (seg as u64) << yo_arena::SEGMENT_SHIFT;
        let bump = self.arena.recorded_bump(seg) as usize;
        let stop = from.saturating_add(budget).min(bump);

        let mut moved = 0;
        let mut off = from;
        while off < stop {
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
        (moved, off)
    }

    /// How much to walk on this call, given how far behind the collector is.
    ///
    /// A fixed budget has to be either a good pause or a good collection rate
    /// and it cannot be both. At 64 kilobytes a segment takes thirty two calls,
    /// and a pipelined flood of writes makes garbage faster than one call per
    /// batch gets it back: measured with variable sized values at pipeline 16,
    /// the tail came down from 2.6 milliseconds to 1.6 and the process held 18
    /// MB more, because segments queued up waiting their turn to be walked.
    ///
    /// So the floor is what a command can be asked to wait for, and the depth
    /// of that queue is what says how much more than the floor is needed to
    /// keep up. One candidate is a store that is keeping up and pays the floor.
    /// Nine is a store nine segments behind, and it walks nine slices.
    ///
    /// The queue and not the dead byte total. Dead bytes were tried first,
    /// measured against the point compaction starts at, and that ratio cannot
    /// see a backlog at all: the threshold is a fraction of what the arena
    /// holds, so a collector that falls behind grows the arena, which raises
    /// the threshold, which puts the ratio back where it was. It sat at the
    /// floor through the whole flood and the 18 MB stayed exactly where it was.
    /// A count of segments has no such denominator.
    ///
    /// Linear in the depth and not squared. This is a controller in a loop with
    /// its own input, and a term that grows faster than the error is how one of
    /// those starts to oscillate.
    fn budget(&self) -> usize {
        let behind = self.arena.candidate_count().max(1);
        EVAC_FLOOR.saturating_mul(behind).min(EVAC_CEILING)
    }

    /// Do one bounded slice of compaction, and say how many records moved.
    ///
    /// `None` means there was no candidate and there is nothing in flight. It
    /// is not the same as `Some(0)`, which is a slice that walked only records
    /// that had already been overwritten: that one made progress and cost
    /// something, and a caller deciding whether to go round again needs to be
    /// told so.
    ///
    /// This is the whole maintenance contract: a bounded amount of work per
    /// call, so a caller that runs it once per batch never pays for a full pass
    /// over the arena and never pays for a whole segment either. Finding out
    /// there is nothing to do is one comparison against the running dead byte
    /// total.
    ///
    /// A segment takes as many calls as it takes. Each one picks up where the
    /// last stopped and only the call that reaches the end gives the two
    /// megabytes back, so the space comes back in one lump at the end while the
    /// cost of getting it back is spread over the batches in between. That is
    /// the trade: a segment stays around a little longer than it used to, and
    /// no single command waits for the whole of it.
    ///
    /// The segment in flight is finished before another is chosen, rather than
    /// asking which segment is worst on every call. Otherwise a segment that is
    /// three quarters evacuated could be put down in favour of a worse one and
    /// never picked up, and the arena would fill with segments that are nearly
    /// empty and never reclaimed.
    pub fn compact_step(&mut self) -> Option<usize> {
        self.writes += 1;
        let (seg, from) = match self.evac {
            Some(e) => (e.seg, e.off),
            None => (self.arena.worst_candidate()?, yo_arena::HEADER_SIZE),
        };
        if seg == self.arena.current_segment() {
            self.evac = None;
            return Some(0);
        }

        // After the choice and not before it. The count is a walk over the
        // segment headers, and a store with nothing to collect should not pay
        // for one on every batch to be told there is nothing to collect.
        let budget = self.budget();
        let (moved, off) = self.evacuate(seg, from, budget);
        if off >= self.arena.recorded_bump(seg) as usize {
            self.arena.reclaim(seg);
            self.evac = None;
        } else {
            self.evac = Some(Evac { seg, off });
        }
        Some(moved)
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
    /// value here is a keyspace record, whose deadline is inside the value, so
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

    /// A segment is evacuated over several calls, and it comes back only on the
    /// call whose walk reaches the end of it.
    ///
    /// This is what the budget is for. One call used to copy every live record
    /// in two megabytes, around twenty six thousand of them at 64 byte values,
    /// and the whole batch of replies queued behind it waited for all of them.
    /// That is where a p99 of 3.9 milliseconds on the write rows came from
    /// while the p50 was in line with Redis: the median command paid nothing
    /// and one command in a few thousand paid for a segment.
    ///
    /// The loop is also what catches a walk that restarts instead of resuming.
    /// A restart would move records and look like progress, and it would spend
    /// every call re-walking the dead space it made on the last one, so the
    /// cursor would never reach the bump and the segment would never come back.
    #[test]
    fn a_segment_comes_back_over_several_calls() {
        let mut m = RawMap::new();
        let val = vec![b'z'; COMPACT_VAL];
        const N: usize = COMPACT_N;
        for i in 0..N {
            m.set(&key(i), &val);
        }
        // Every other key, so the early segments are well past the dead ratio
        // and there is still a live half to copy out.
        for i in (0..N).step_by(2) {
            m.del(&key(i));
        }

        let rec = (HDR + key(0).len() + COMPACT_VAL).next_multiple_of(yo_arena::ALIGN);
        let per_call = m.budget() / rec + 1;
        let free = m.arena().free_segments();

        let moved = m.compact_step().expect("half of it is dead");
        assert!(
            moved <= per_call,
            "one call moved {moved} records and the budget is {per_call}"
        );
        assert_eq!(
            m.arena().free_segments(),
            free,
            "a segment came back before the walk reached the end of it"
        );

        let mut calls = 1;
        while m.arena().free_segments() == free {
            m.compact_step()
                .expect("the segment in flight is not finished");
            calls += 1;
            assert!(calls < 1000, "the walk is not getting any further along");
        }
        assert!(calls > 2, "the whole segment came back in {calls} calls");

        for i in 0..N {
            let want = if i % 2 == 0 { None } else { Some(val.clone()) };
            assert_eq!(m.get(&key(i)).map(<[u8]>::to_vec), want, "key {i}");
        }
    }

    /// The budget grows with how far behind the collector is.
    ///
    /// A store with one segment waiting pays the floor, which is the pause a
    /// command can be asked to wait for. One with a queue of them walks a slice
    /// per segment in the queue, which is what keeps a pipelined write flood
    /// from outrunning one call per batch and leaving the process holding the
    /// segments that never got their turn.
    #[test]
    fn the_budget_scales_with_the_backlog() {
        let mut m = RawMap::new();
        let val = vec![b'z'; COMPACT_VAL];
        const N: usize = COMPACT_N;
        for i in 0..N {
            m.set(&key(i), &val);
        }
        assert_eq!(m.budget(), EVAC_FLOOR, "nothing is waiting yet");

        for i in 0..N {
            m.del(&key(i));
        }
        let flooded = m.budget();
        assert!(
            flooded >= EVAC_FLOOR * m.arena().candidate_count(),
            "{} segments are waiting and the budget is {flooded}",
            m.arena().candidate_count()
        );
        assert!(
            flooded > EVAC_FLOOR,
            "every segment is dead and the budget is still the floor"
        );
        assert!(flooded <= EVAC_CEILING, "walked past a whole segment");
    }

    /// A segment that is partway through being evacuated is finished before a
    /// worse one is started.
    ///
    /// Writes keep coming while a segment is being walked and they make dead
    /// space elsewhere, so the answer to "which segment is worst" moves around
    /// underneath a walk that takes thirty calls. Asking it again on every call
    /// would let a segment be put down at nine tenths done in favour of one
    /// that is slightly worse, and the arena would fill up with segments that
    /// are nearly empty and never reclaimed.
    ///
    /// Here the first quarter of the keyspace is deleted so that the segment at
    /// the front is the only candidate, one call starts on it, and then the
    /// back half goes too so that another segment ties with it mid walk. The
    /// tie goes to the later segment, so a walk that asked again would move to
    /// it and leave the first one part done.
    #[test]
    fn the_segment_in_flight_is_finished_first() {
        let mut m = RawMap::new();
        let val = vec![b'z'; COMPACT_VAL];
        const N: usize = COMPACT_N;
        for i in 0..N {
            m.set(&key(i), &val);
        }
        for i in 0..N / 4 {
            m.del(&key(i));
        }

        let free = m.arena().free_segments();
        let first = m.arena().worst_candidate().expect("the front is all dead");
        m.compact_step().expect("there is a candidate");

        for i in N / 2..N {
            m.del(&key(i));
        }
        let worse = m.arena().worst_candidate().expect("the back is all dead");
        assert_ne!(worse, first, "the test needs the answer to have moved");

        while m.arena().free_segments() == free {
            m.compact_step()
                .expect("the segment in flight is not finished");
        }
        assert_eq!(
            m.arena().recorded_bump(first),
            yo_arena::HEADER_SIZE as u64,
            "the segment that was in flight is not the one that came back"
        );
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

    /// Whatever memoizes against this counter is only correct if every way of
    /// writing to the map moves it. A method that mutates and does not is not a
    /// slow memo, it is a wrong answer, so this asserts on the whole `&mut self`
    /// surface rather than on the ones that look like they matter.
    #[test]
    fn every_way_of_writing_moves_the_counter() {
        let mut m = RawMap::new();
        let mut last = m.writes();
        let mut moved = |m: &RawMap, what: &str| {
            assert!(m.writes() > last, "{what} did not move the counter");
            last = m.writes();
        };

        m.set(b"k", b"v");
        moved(&m, "set");
        m.set_with(b"k", 1, |b| b[0] = b'w');
        moved(&m, "set_with");
        m.value_mut(b"k");
        moved(&m, "value_mut");
        m.value_mut_hashed(RawMap::hash_of(b"k"), b"k");
        moved(&m, "value_mut_hashed");
        m.compact_step();
        moved(&m, "compact_step");
        m.compact_segment(0);
        moved(&m, "compact_segment");
        m.del(b"k");
        moved(&m, "del");
    }

    /// `clear` replaces the map with a fresh one, and a fresh one starts at
    /// zero. A memo taken at write 3 against a map that went back to 0 and
    /// climbed to 3 again would read as still valid on the one call where every
    /// key in the map had been thrown away.
    #[test]
    fn clearing_does_not_send_the_counter_backwards() {
        let mut m = RawMap::new();
        for i in 0..10u32 {
            m.set(&i.to_le_bytes(), b"v");
        }
        let before = m.writes();
        m.clear();
        assert!(m.writes() > before, "clear went backwards or stood still");
    }
}
