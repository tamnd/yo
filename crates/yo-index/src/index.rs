//! The index: a directory of segments, dashtable style growth, no stop the
//! world rehash.
//!
//! `05` section 2.2. The index is an array of segment pointers. Each segment is
//! a run of buckets with a local depth. When a bucket and its overflow chain
//! are both full, the segment splits: a new segment is allocated, entries are
//! redistributed by the next hash bit, and the directory entries that named the
//! old segment are repointed. If the local depth would pass the global depth,
//! the directory doubles first, which is one allocation and a memcpy of
//! pointers.
//!
//! A split touches one segment. Nothing else in the shard stops.
//!
//! # Which bits do what
//!
//! ```text
//!  63          56 55                    N                          0
//! +--------------+-----------------------+--------------------------+
//! |     tag      |  directory index      |   bucket within segment  |
//! +--------------+-----------------------+--------------------------+
//! ```
//!
//! The tag is the top eight bits, the directory takes the next `global_depth`
//! bits, and the bucket index comes off the bottom. They are disjoint on
//! purpose: if the directory used the top bits, every key in a segment would
//! share a tag prefix and the prefilter would stop filtering.

use crate::bucket::{Bucket, SLOTS};
use yo_common::{Addr, tag_of};

/// Buckets in one index segment. Sixty four buckets is 4 KiB, which is one page
/// and 448 entries.
pub const SEGMENT_BUCKETS: usize = 64;

/// Overflow buckets a chain may hold before the segment splits instead.
pub const MAX_CHAIN: usize = 2;

/// Bits available to the directory, below the tag.
const DIR_BITS: u32 = 56;

/// The deepest the directory may go. Past this the hash has no bits left to
/// discriminate on and the only honest answer is a longer chain.
const MAX_DEPTH: u8 = 48;

struct Segment {
    buckets: Vec<Bucket>,
    overflow: Vec<Bucket>,
    local_depth: u8,
}

impl Segment {
    fn new(local_depth: u8) -> Segment {
        Segment {
            buckets: vec![Bucket::EMPTY; SEGMENT_BUCKETS],
            overflow: Vec::new(),
            local_depth,
        }
    }
}

/// What the index needs to know about the records its addresses point at.
///
/// The index stores a tag and an address, not a key and not a hash. A split has
/// to recompute which side of the next bit each entry falls on, and a probe has
/// to confirm a tag match, so both need to reach the key bytes. Keeping that
/// behind a trait is what lets the index stay independent of the record format,
/// which changes in M1 and again when documents arrive.
pub trait Keys {
    /// The full hash of the key stored at `addr`.
    fn hash_at(&self, addr: Addr) -> u64;

    /// Whether the key stored at `addr` is `key`.
    fn eq_at(&self, addr: Addr, key: &[u8]) -> bool;
}

/// The shard's index.
#[derive(Debug)]
pub struct Index {
    dir: Vec<u32>,
    segs: Vec<Segment>,
    global_depth: u8,
    len: usize,
    splits: u64,
    doublings: u64,
}

impl core::fmt::Debug for Segment {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Segment")
            .field("local_depth", &self.local_depth)
            .field("overflow", &self.overflow.len())
            .finish()
    }
}

impl Index {
    /// A new index with one segment.
    pub fn new() -> Index {
        Index {
            dir: vec![0],
            segs: vec![Segment::new(0)],
            global_depth: 0,
            len: 0,
            splits: 0,
            doublings: 0,
        }
    }

    /// How many entries the index holds.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index holds nothing.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many segments exist.
    pub fn segment_count(&self) -> usize {
        self.segs.len()
    }

    /// The current global depth.
    pub fn global_depth(&self) -> u8 {
        self.global_depth
    }

    /// How many segment splits have happened over the life of this index.
    pub fn splits(&self) -> u64 {
        self.splits
    }

    /// How many times the directory has doubled.
    pub fn doublings(&self) -> u64 {
        self.doublings
    }

    /// Bytes of index structure, for `INFO memory`.
    pub fn memory_bytes(&self) -> usize {
        self.dir.len() * size_of::<u32>()
            + self
                .segs
                .iter()
                .map(|s| (s.buckets.len() + s.overflow.len()) * size_of::<Bucket>())
                .sum::<usize>()
    }

    #[inline(always)]
    fn dir_index(&self, hash: u64) -> usize {
        let d = self.global_depth as u32;
        if d == 0 {
            return 0;
        }
        ((hash >> (DIR_BITS - d)) & ((1u64 << d) - 1)) as usize
    }

    #[inline(always)]
    fn bucket_index(hash: u64) -> usize {
        (hash as usize) & (SEGMENT_BUCKETS - 1)
    }

    /// Find the address stored under `key`.
    ///
    /// The hot path, and the one M0's four nanosecond gate measures. One load
    /// of the bucket, one SWAR compare of seven tags, and a key comparison per
    /// surviving match, which is one comparison in the overwhelming majority of
    /// probes because a tag collision is a 1 in 256 event.
    #[inline]
    pub fn get<K: Keys>(&self, hash: u64, key: &[u8], keys: &K) -> Option<Addr> {
        let tag = tag_of(hash);
        let seg = &self.segs[self.dir[self.dir_index(hash)] as usize];
        let mut b = &seg.buckets[Self::bucket_index(hash)];
        loop {
            for i in b.match_tag(tag) {
                let addr = b.addr(i);
                if keys.eq_at(addr, key) {
                    return Some(addr);
                }
            }
            {
                let next = b.link()?;
                b = &seg.overflow[(next - 1) as usize]
            }
        }
    }

    /// Whether `key` is present.
    #[inline]
    pub fn contains<K: Keys>(&self, hash: u64, key: &[u8], keys: &K) -> bool {
        self.get(hash, key, keys).is_some()
    }

    /// Insert or replace the address stored under `key`.
    ///
    /// Returns the address that was there before, if any. The caller owns what
    /// that address points at, so freeing it is the caller's job. The index
    /// does not know how big a record is and will not guess.
    pub fn insert<K: Keys>(&mut self, hash: u64, key: &[u8], addr: Addr, keys: &K) -> Option<Addr> {
        debug_assert!(addr.is_some(), "the index cannot store the absent address");
        let tag = tag_of(hash);

        loop {
            let seg_idx = self.dir[self.dir_index(hash)] as usize;
            let bucket_idx = Self::bucket_index(hash);

            // Replace in place if the key is already here, and remember the
            // first free slot on the way through so that a miss does not walk
            // the chain twice.
            let mut free: Option<(usize, usize)> = None;
            let mut chain_len = 0usize;
            let mut cursor: Option<usize> = None;

            loop {
                let seg = &self.segs[seg_idx];
                let b = match cursor {
                    None => &seg.buckets[bucket_idx],
                    Some(o) => &seg.overflow[o],
                };
                for i in b.match_tag(tag) {
                    if keys.eq_at(b.addr(i), key) {
                        let old = b.addr(i);
                        let seg = &mut self.segs[seg_idx];
                        let b = match cursor {
                            None => &mut seg.buckets[bucket_idx],
                            Some(o) => &mut seg.overflow[o],
                        };
                        b.set_addr(i, addr);
                        return Some(old);
                    }
                }
                if free.is_none()
                    && let Some(i) = b.match_empty().first()
                {
                    free = Some((cursor.unwrap_or(usize::MAX), i));
                }
                match b.link() {
                    Some(next) => {
                        cursor = Some((next - 1) as usize);
                        chain_len += 1;
                    }
                    None => break,
                }
            }

            if let Some((where_, slot)) = free {
                let seg = &mut self.segs[seg_idx];
                let b = if where_ == usize::MAX {
                    &mut seg.buckets[bucket_idx]
                } else {
                    &mut seg.overflow[where_]
                };
                b.set(slot, tag, addr);
                self.len += 1;
                return None;
            }

            // Nothing free anywhere in the chain.
            if chain_len < MAX_CHAIN || self.segs[seg_idx].local_depth >= MAX_DEPTH {
                self.extend_chain(seg_idx, bucket_idx, cursor, tag, addr);
                self.len += 1;
                return None;
            }

            self.split(seg_idx, keys);
            // The directory moved under us, so start over rather than trying to
            // reason about where this key landed.
        }
    }

    fn extend_chain(
        &mut self,
        seg_idx: usize,
        bucket_idx: usize,
        tail: Option<usize>,
        tag: u8,
        addr: Addr,
    ) {
        let seg = &mut self.segs[seg_idx];
        let mut fresh = Bucket::EMPTY;
        fresh.set(0, tag, addr);
        yo_alloc::allow(|| seg.overflow.push(fresh));
        let new_idx = seg.overflow.len() - 1;
        let link = (new_idx + 1) as u64;
        match tail {
            None => seg.buckets[bucket_idx].set_link(link),
            Some(o) => seg.overflow[o].set_link(link),
        }
    }

    /// Remove `key`.
    ///
    /// Returns the address that was stored, if any. Tombstone free: the tag
    /// goes back to zero. A probe stops at the first empty tag in the chain
    /// rather than in the bucket, so nothing has to be pulled back.
    pub fn remove<K: Keys>(&mut self, hash: u64, key: &[u8], keys: &K) -> Option<Addr> {
        let tag = tag_of(hash);
        let seg_idx = self.dir[self.dir_index(hash)] as usize;
        let bucket_idx = Self::bucket_index(hash);
        let mut cursor: Option<usize> = None;

        loop {
            let seg = &self.segs[seg_idx];
            let b = match cursor {
                None => &seg.buckets[bucket_idx],
                Some(o) => &seg.overflow[o],
            };
            let mut hit = None;
            for i in b.match_tag(tag) {
                if keys.eq_at(b.addr(i), key) {
                    hit = Some((i, b.addr(i)));
                    break;
                }
            }
            if let Some((i, addr)) = hit {
                let seg = &mut self.segs[seg_idx];
                let b = match cursor {
                    None => &mut seg.buckets[bucket_idx],
                    Some(o) => &mut seg.overflow[o],
                };
                b.clear(i);
                self.len -= 1;
                return Some(addr);
            }
            let next = {
                let seg = &self.segs[seg_idx];
                let b = match cursor {
                    None => &seg.buckets[bucket_idx],
                    Some(o) => &seg.overflow[o],
                };
                b.link()
            };
            {
                let n = next?;
                cursor = Some((n - 1) as usize)
            }
        }
    }

    /// Split one segment by the next hash bit.
    fn split<K: Keys>(&mut self, seg_idx: usize, keys: &K) {
        let ld = self.segs[seg_idx].local_depth;
        if ld == self.global_depth {
            self.double_directory();
        }
        let gd = self.global_depth;
        debug_assert!(ld < gd);

        self.segs[seg_idx].local_depth = ld + 1;
        yo_alloc::allow(|| self.segs.push(Segment::new(ld + 1)));
        let new_idx = self.segs.len() - 1;
        self.splits += 1;

        // Repoint the half of the directory entries whose bit `ld`, counted
        // from the top of a `gd` bit index, is one.
        let shift = (gd - 1 - ld) as u32;
        for i in 0..self.dir.len() {
            if self.dir[i] as usize == seg_idx && ((i >> shift) & 1) == 1 {
                self.dir[i] = new_idx as u32;
            }
        }

        // Move every entry that now belongs to the new segment. The hash is
        // recomputed from the key rather than stored, which is the cost of
        // spending all 56 non tag bits on addressing instead of on a cached
        // hash. It is paid once per entry per split, and a split is a
        // `log2(n / segment capacity)` event.
        let mut moving: Vec<(u64, Addr)> = Vec::new();
        yo_alloc::allow(|| {
            let seg = &mut self.segs[seg_idx];
            let mut visit = |b: &mut Bucket| {
                for i in 0..SLOTS {
                    if b.tag(i) == crate::bucket::EMPTY {
                        continue;
                    }
                    let addr = b.addr(i);
                    let h = keys.hash_at(addr);
                    if ((h >> (DIR_BITS - gd as u32)) & ((1u64 << gd) - 1)) >> shift & 1 == 1 {
                        moving.push((h, addr));
                        b.clear(i);
                    }
                }
            };
            for b in seg.buckets.iter_mut() {
                visit(b);
            }
            for b in seg.overflow.iter_mut() {
                visit(b);
            }
        });

        for (h, addr) in moving {
            self.place_raw(new_idx, h, addr);
        }
    }

    /// Put an entry into a known segment without any lookup.
    ///
    /// Only correct during a split, where the entry is known to be absent from
    /// the destination because it was just removed from the source.
    fn place_raw(&mut self, seg_idx: usize, hash: u64, addr: Addr) {
        let tag = tag_of(hash);
        let bucket_idx = Self::bucket_index(hash);
        let mut cursor: Option<usize> = None;
        loop {
            let seg = &mut self.segs[seg_idx];
            let b = match cursor {
                None => &mut seg.buckets[bucket_idx],
                Some(o) => &mut seg.overflow[o],
            };
            if let Some(i) = b.match_empty().first() {
                b.set(i, tag, addr);
                return;
            }
            match b.link() {
                Some(n) => cursor = Some((n - 1) as usize),
                None => {
                    self.extend_chain(seg_idx, bucket_idx, cursor, tag, addr);
                    return;
                }
            }
        }
    }

    fn double_directory(&mut self) {
        assert!(
            self.global_depth < MAX_DEPTH,
            "the directory has run out of hash bits"
        );
        yo_alloc::allow(|| {
            let mut next = Vec::with_capacity(self.dir.len() * 2);
            for &s in &self.dir {
                next.push(s);
                next.push(s);
            }
            self.dir = next;
        });
        self.global_depth += 1;
        self.doublings += 1;
    }

    /// Every address in the index, in no particular order.
    ///
    /// For compaction, which walks the index rather than the arena because an
    /// allocation has exactly one referent and that referent is an index entry
    /// (`05` section 3.2).
    pub fn addresses(&self) -> impl Iterator<Item = Addr> + '_ {
        self.segs
            .iter()
            .enumerate()
            .flat_map(move |(si, seg)| {
                // A segment can be named by several directory entries, but it
                // is visited once here because we walk segments, not the
                // directory.
                let _ = si;
                seg.buckets.iter().chain(seg.overflow.iter())
            })
            .flat_map(|b| {
                (0..SLOTS).filter_map(move |i| {
                    if b.tag(i) == crate::bucket::EMPTY {
                        None
                    } else {
                        Some(b.addr(i))
                    }
                })
            })
    }

    /// Replace the address of an entry that is being moved by compaction.
    ///
    /// Since the shard owns both the arena and the index, rewriting an index
    /// entry is a store, which is the whole reason compaction is affordable.
    pub fn relocate<K: Keys>(&mut self, hash: u64, key: &[u8], to: Addr, keys: &K) -> bool {
        let tag = tag_of(hash);
        let seg_idx = self.dir[self.dir_index(hash)] as usize;
        let bucket_idx = Self::bucket_index(hash);
        let mut cursor: Option<usize> = None;
        loop {
            let found = {
                let seg = &self.segs[seg_idx];
                let b = match cursor {
                    None => &seg.buckets[bucket_idx],
                    Some(o) => &seg.overflow[o],
                };
                let mut hit = None;
                for i in b.match_tag(tag) {
                    if keys.eq_at(b.addr(i), key) {
                        hit = Some(i);
                        break;
                    }
                }
                (hit, b.link())
            };
            if let (Some(i), _) = found {
                let seg = &mut self.segs[seg_idx];
                let b = match cursor {
                    None => &mut seg.buckets[bucket_idx],
                    Some(o) => &mut seg.overflow[o],
                };
                b.set_addr(i, to);
                return true;
            }
            match found.1 {
                Some(n) => cursor = Some((n - 1) as usize),
                None => return false,
            }
        }
    }
}

impl Default for Index {
    fn default() -> Index {
        Index::new()
    }
}
