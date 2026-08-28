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

    /// Store `val` under `key`, returning the length of the value it replaced.
    ///
    /// Replacement always writes a fresh record rather than editing in place,
    /// even when the lengths match. Editing in place would be faster for the
    /// same length case, but it is only safe once epochs are in (M1), because a
    /// reader that has already resolved the address would see a torn value. The
    /// in place path lands with the epoch machinery, not before it.
    pub fn set(&mut self, key: &[u8], val: &[u8]) -> Option<usize> {
        assert!(key.len() <= u32::MAX as usize, "key too long");
        assert!(val.len() <= u32::MAX as usize, "value too long");
        let total = HDR + key.len() + val.len();
        let (addr, buf) = self.arena.alloc(total).expect("arena out of space");
        buf[0..4].copy_from_slice(&(key.len() as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(val.len() as u32).to_le_bytes());
        // The arena hands back a run padded up to its alignment, so index to
        // `total` rather than to the end of the slice.
        buf[HDR..HDR + key.len()].copy_from_slice(key);
        buf[HDR + key.len()..total].copy_from_slice(val);

        let h = wyhash(key, 0);
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
    /// hand the segment back for retirement.
    ///
    /// The F2 shape from `05` section 3.2: walk the index, not the arena,
    /// because an allocation has exactly one referent and that referent is an
    /// index entry. Copy, rewrite the entry, done. No forwarding pointers and
    /// no read barrier.
    pub fn compact_segment(&mut self, seg: usize) -> usize {
        let base = (seg as u64) << yo_arena::SEGMENT_SHIFT;
        let end = base + yo_arena::SEGMENT_SIZE as u64;
        let victims: Vec<Addr> = self
            .index
            .addresses()
            .filter(|a| a.space() == Some(Space::Arena))
            .filter(|a| a.offset() >= base && a.offset() < end)
            .collect();

        let mut moved = 0;
        for old in victims {
            let (klen, vlen) = Record::lens(self.arena.get(old, HDR));
            let total = HDR + klen + vlen;
            let bytes = self.arena.get(old, total).to_vec();
            let key = &bytes[HDR..HDR + klen];
            let h = wyhash(key, 0);
            let (new, buf) = self.arena.alloc(total).expect("arena out of space");
            buf[..total].copy_from_slice(&bytes);
            let recs = Records { arena: &self.arena };
            let ok = self.index.relocate(h, key, new, &recs);
            debug_assert!(ok, "compaction lost an entry the index just handed us");
            self.arena.free(old, total);
            moved += 1;
        }
        moved
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
