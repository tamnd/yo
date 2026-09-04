//! A cuckoo filter, byte for byte the one RedisBloom holds.
//!
//! # Why a copy and not a better one
//!
//! The same argument as `crate::bloom`, for the same reason: `CF.SCANDUMP` hands
//! a client the filter's actual bytes and `CF.LOADCHUNK` takes them back, so the
//! layout is the product. What is copied here is the geometry, the hash, the
//! fingerprint, the alternate bucket, the order the buckets are searched in and
//! the order a full bucket is kicked in, because all six of them decide which
//! byte an item ends up in and a client moving a filter between two servers is
//! entitled to find it in the same place.
//!
//! # What a cuckoo filter buys over a Bloom filter
//!
//! Deletion, which a Bloom filter cannot do, and a lookup that touches two cache
//! lines instead of eight scattered bits. The cost is that a filter can refuse an
//! item: there is no arrangement of the two buckets an item is allowed in that
//! leaves room, and no amount of kicking finds one. Fan, Andersen, Kaminsky and
//! Mitzenmacher's 2014 paper is the design and RedisBloom's is a direct
//! implementation of it, with a chain of filters bolted on the side so that a
//! refusal turns into a bigger filter rather than an error.
//!
//! # The chain
//!
//! One filter has a fixed number of buckets and a fixed number of slots in each,
//! and when an insert cannot find room the chain grows a filter `expansion` times
//! wider and puts the item there. A lookup walks every filter and a count adds
//! them up, so the read side pays for the growth exactly the way the Bloom chain
//! does.
//!
//! Deleting leaves a hole that only the item that hashed there can use, so the
//! chain also compacts: items in the newer filters are pulled back down into the
//! older ones and an emptied filter is dropped. That happens on its own once
//! more than a tenth of what is in the filter has been deleted, and `CF.COMPACT`
//! asks for it directly. Compaction is possible at all because every filter's
//! bucket count is a power of two, so an item's bucket in a smaller filter is
//! its bucket in a bigger one taken modulo the smaller count, and the original
//! hash is not needed to work it out.

use crate::hash;

/// The multiplier the alternate bucket is derived with, which is the low half of
/// the same constant the hash mixes with.
const ALT: u64 = 0x5bd1_e995;
/// What an unused slot holds. A fingerprint is never zero, because it is a
/// modulo 255 with a one added, which is what leaves this spare.
const EMPTY: u8 = 0;
/// Bytes of header a dump starts with, which is four counts and three widths.
/// A `CF.LOADCHUNK` needs this to tell a chunk that is not a header from one
/// that is a header and describes a filter nobody could build.
pub const HEADER: usize = 38;
/// How much of what is in the filter has to have been deleted before a delete
/// compacts the chain on its own.
const COMPACT_AT: f64 = 0.10;

/// The largest capacity `CF.RESERVE` accepts.
pub const MAX_CAPACITY: i64 = 1_073_741_824;
/// The largest number of slots in a bucket. One is a plain cuckoo hash table and
/// fills up at about half, four is where the occupancy stops improving, and the
/// default of two is RedisBloom's.
pub const MAX_BUCKET_SIZE: i64 = 255;
/// The most kicks an insert will do before it gives up and grows the chain.
pub const MAX_ITERATIONS: i64 = 65_535;
/// The largest growth factor.
pub const MAX_EXPANSION: i64 = 32_768;
/// How many filters a chain may have. At the default growth of one they are all
/// the same size, so this is the real capacity limit and the reason a chain that
/// was reserved too small answers an error eventually rather than growing for
/// ever.
pub const MAX_FILTERS: usize = 32;

/// What happened to an item that was offered to a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insert {
    /// It went in. A cuckoo filter takes the same item twice and counts it
    /// twice, so this is the answer for a duplicate too unless the caller asked
    /// for the unique form.
    Yes,
    /// Something already in the filter has the same fingerprint in the same
    /// bucket, and the caller asked for the unique form.
    Exists,
    /// There is no room and the chain was told not to grow.
    Full,
    /// There is no room and the chain has as many filters as it is allowed.
    MaxFilters,
}

/// One filter in the chain.
#[derive(Debug, Clone)]
struct Sub {
    /// `buckets * bucket_size` fingerprints, bucket by bucket.
    data: Box<[u8]>,
    /// How many buckets, always a power of two.
    buckets: u64,
}

impl Sub {
    /// An empty filter of `buckets` buckets.
    fn new(buckets: u64, bucket_size: u16) -> Sub {
        let len = buckets as usize * bucket_size as usize;
        Sub {
            data: vec![EMPTY; len].into_boxed_slice(),
            buckets,
        }
    }

    /// Where the bucket a hash lands in starts.
    fn at(&self, h: u64, bucket_size: usize) -> usize {
        (h % self.buckets) as usize * bucket_size
    }
}

/// The two buckets and the fingerprint an item is allowed to use.
///
/// Both hashes are kept whole rather than reduced to bucket numbers, because the
/// chain's filters have different bucket counts and every one of them is a power
/// of two, so one pair of hashes serves all of them.
#[derive(Debug, Clone, Copy)]
struct Look {
    h1: u64,
    h2: u64,
    fp: u8,
}

/// The fingerprint and the two buckets for `item`.
///
/// The hash is `MurmurHash64A` at seed zero, the fingerprint is the hash modulo
/// 255 with one added so that it is never the empty marker, and the second
/// bucket is the first exclusive ored with the fingerprint times a constant.
/// That last part is what makes a cuckoo filter work at all: the move from
/// either bucket to the other needs only the fingerprint, so an item can be
/// kicked out of a bucket without anybody remembering what it was.
fn look(item: &[u8]) -> Look {
    let h = hash::murmur64a(item, 0);
    let fp = (h % 255 + 1) as u8;
    Look {
        h1: h,
        h2: alt(h, fp),
        fp,
    }
}

/// The other bucket for a fingerprint, which is its own inverse.
fn alt(h: u64, fp: u8) -> u64 {
    h ^ u64::from(fp).wrapping_mul(ALT)
}

/// A chain of cuckoo filters under one name.
#[derive(Debug, Clone)]
pub struct Cuckoo {
    /// Oldest and smallest first.
    filters: Vec<Sub>,
    /// How many items are in the chain now, which is the adds less the deletes.
    items: u64,
    /// How many have been deleted since the last compaction.
    deletes: u64,
    /// The first filter's bucket count, which every later one is a multiple of
    /// and which the dumped header carries.
    buckets: u64,
    /// Slots per bucket.
    bucket_size: u16,
    /// How many kicks an insert will do before giving up.
    max_iterations: u16,
    /// How much wider each filter is than the one before it. Zero means the
    /// chain will not grow.
    expansion: u16,
}

impl Cuckoo {
    /// A chain of one filter with room for about `capacity` items.
    ///
    /// The bucket count is `capacity / bucket_size` rounded up to a power of
    /// two, so the real capacity is somewhere between the one asked for and
    /// twice it, and the rounding is what lets a bucket number be a mask rather
    /// than a division. `expansion` is rounded up to a power of two for the same
    /// reason, so an expansion of three is an expansion of four and `CF.INFO`
    /// says so.
    #[must_use]
    pub fn new(capacity: u64, bucket_size: u16, max_iterations: u16, expansion: u16) -> Cuckoo {
        let buckets = (capacity / u64::from(bucket_size))
            .max(2)
            .next_power_of_two();
        let expansion = match expansion {
            0 => 0,
            n => n.next_power_of_two(),
        };
        Cuckoo {
            filters: vec![Sub::new(buckets, bucket_size)],
            items: 0,
            deletes: 0,
            buckets,
            bucket_size,
            max_iterations,
            expansion,
        }
    }

    /// Slots per bucket, as an index.
    fn size(&self) -> usize {
        self.bucket_size as usize
    }

    /// How many items are in the chain.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.items
    }

    /// Whether nothing is in it, which is what makes a dump answer nothing at
    /// all rather than a header.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items == 0
    }

    /// How many items have been deleted since the last compaction.
    #[must_use]
    pub fn deleted(&self) -> u64 {
        self.deletes
    }

    /// How many filters the chain has.
    #[must_use]
    pub fn filters(&self) -> usize {
        self.filters.len()
    }

    /// The first filter's bucket count, which is what `CF.INFO` reports however
    /// far the chain has grown.
    #[must_use]
    pub fn buckets(&self) -> u64 {
        self.buckets
    }

    /// Slots per bucket.
    #[must_use]
    pub fn bucket_size(&self) -> u16 {
        self.bucket_size
    }

    /// The kick budget.
    #[must_use]
    pub fn max_iterations(&self) -> u16 {
        self.max_iterations
    }

    /// The growth factor, already rounded to a power of two.
    #[must_use]
    pub fn expansion(&self) -> u16 {
        self.expansion
    }

    /// What `CF.INFO` reports as the size, which is RedisBloom's own accounting
    /// of its chain struct, its per filter structs and the fingerprint arrays.
    #[must_use]
    pub fn reported_size(&self) -> u64 {
        40 + 16 * self.filters.len() as u64 + self.bytes()
    }

    /// What this actually holds.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        size_of::<Cuckoo>() + self.filters.capacity() * size_of::<Sub>() + self.bytes() as usize
    }

    /// Every filter's fingerprints added up.
    fn bytes(&self) -> u64 {
        self.filters.iter().map(|f| f.data.len() as u64).sum()
    }

    /// Whether the chain has seen `item`, allowing for the false positives a
    /// fingerprint of one byte gives.
    #[must_use]
    pub fn contains(&self, item: &[u8]) -> bool {
        let l = look(item);
        // Newest first, because the item most recently offered went there.
        (0..self.filters.len())
            .rev()
            .any(|f| self.slot(f, l.h1, l.fp).is_some() || self.slot(f, l.h2, l.fp).is_some())
    }

    /// How many copies of `item` the chain thinks it has.
    ///
    /// Both of an item's buckets are counted even when they are the same bucket,
    /// which happens whenever the fingerprint times the constant is a multiple
    /// of the bucket count, so a filter with one copy of such an item answers
    /// two. That is the reference's arithmetic and it is a bug, and it is copied
    /// because a client comparing counts between two servers would otherwise see
    /// them disagree.
    #[must_use]
    pub fn count(&self, item: &[u8]) -> u64 {
        let l = look(item);
        (0..self.filters.len())
            .map(|f| self.matches(f, l.h1, l.fp) + self.matches(f, l.h2, l.fp))
            .sum()
    }

    /// Take one copy of `item` out, compacting the chain if enough has been
    /// deleted to be worth it.
    pub fn remove(&mut self, item: &[u8]) -> bool {
        let l = look(item);
        for f in (0..self.filters.len()).rev() {
            let Some(at) = self
                .slot(f, l.h1, l.fp)
                .or_else(|| self.slot(f, l.h2, l.fp))
            else {
                continue;
            };
            self.filters[f].data[at] = EMPTY;
            self.items -= 1;
            self.deletes += 1;
            if self.filters.len() > 1 && self.deletes as f64 > self.items as f64 * COMPACT_AT {
                self.compact(false);
            }
            return true;
        }
        false
    }

    /// Put `item` in, whether or not it is already there.
    pub fn insert(&mut self, item: &[u8]) -> Insert {
        self.put(&look(item))
    }

    /// Put `item` in unless the chain already thinks it has it.
    pub fn insert_unique(&mut self, item: &[u8]) -> Insert {
        let l = look(item);
        let seen = (0..self.filters.len())
            .rev()
            .any(|f| self.slot(f, l.h1, l.fp).is_some() || self.slot(f, l.h2, l.fp).is_some());
        if seen {
            return Insert::Exists;
        }
        self.put(&l)
    }

    /// The insert proper.
    ///
    /// A free slot in either bucket of any filter takes it, newest filter first.
    /// Failing that the newest filter is kicked, which walks a chain of evicted
    /// fingerprints looking for one whose other bucket has room. Failing that
    /// the chain grows and the whole thing is tried again on the new filter,
    /// which is why this is a loop and not one pass.
    ///
    /// The filter limit is looked at before any of that rather than at the point
    /// the chain would grow, so a chain that has all the filters it is allowed
    /// refuses an item without looking for room for it, and it refuses one even
    /// when it is empty. That is the reference's order and it is visible from
    /// outside: a chain of four buckets that grows all the way stops at 125
    /// items with three slots still free, because the pass that added the last
    /// filter is the only one that ever gets to put anything in it.
    fn put(&mut self, l: &Look) -> Insert {
        if self.expansion != 0 && self.filters.len() >= MAX_FILTERS {
            return Insert::MaxFilters;
        }
        loop {
            for f in (0..self.filters.len()).rev() {
                if let Some(at) = self.free(f, l.h1).or_else(|| self.free(f, l.h2)) {
                    self.filters[f].data[at] = l.fp;
                    self.items += 1;
                    return Insert::Yes;
                }
            }
            if self.kick(l) {
                self.items += 1;
                return Insert::Yes;
            }
            if self.expansion == 0 {
                return Insert::Full;
            }
            if self.filters.len() >= MAX_FILTERS {
                return Insert::MaxFilters;
            }
            if !self.grow() {
                return Insert::Full;
            }
        }
    }

    /// Add a filter `expansion` times wider than the last one.
    ///
    /// False if the width would not fit in memory that could be addressed, which
    /// only an expansion of thousands can reach and which the reference answers
    /// by failing its allocation.
    fn grow(&mut self) -> bool {
        let last = self.filters.last().expect("a chain has a filter");
        let Some(buckets) = last.buckets.checked_mul(u64::from(self.expansion)) else {
            return false;
        };
        let Some(len) = buckets.checked_mul(u64::from(self.bucket_size)) else {
            return false;
        };
        if len > 64 << 30 || usize::try_from(len).is_err() {
            return false;
        }
        self.filters.push(Sub::new(buckets, self.bucket_size));
        true
    }

    /// Kick fingerprints around the newest filter looking for room.
    ///
    /// Each turn puts the fingerprint in hand into a slot, takes out whatever
    /// was there, and moves to that one's other bucket. If it has a free slot
    /// the walk is over, and if `max_iterations` turns go by without one the
    /// whole walk is undone step by step, so a failed insert leaves the filter
    /// exactly as it found it. The slot that gets evicted moves along by one
    /// each turn, which is what stops a bucket of two from swapping the same
    /// pair back and forth for ever.
    fn kick(&mut self, l: &Look) -> bool {
        let size = self.size();
        let last = self.filters.len() - 1;
        let buckets = self.filters[last].buckets;
        let mut fp = l.fp;
        let mut at = l.h1 % buckets;
        let mut victim = 0usize;
        let mut turns = 0u32;
        while turns < u32::from(self.max_iterations) {
            turns += 1;
            let i = at as usize * size + victim;
            core::mem::swap(&mut self.filters[last].data[i], &mut fp);
            at = alt(at, fp) % buckets;
            if let Some(free) = self.free(last, at) {
                self.filters[last].data[free] = fp;
                return true;
            }
            victim = (victim + 1) % size;
        }
        for _ in 0..turns {
            victim = (victim + size - 1) % size;
            at = alt(at, fp) % buckets;
            let i = at as usize * size + victim;
            core::mem::swap(&mut self.filters[last].data[i], &mut fp);
        }
        false
    }

    /// Pull items down out of the newer filters and drop the ones that empty.
    ///
    /// `all` is the difference between what `CF.COMPACT` asks for and what a
    /// delete does on its own: the command works its way down the whole chain,
    /// and the delete stops at the first filter that would not empty, since
    /// carrying on from there is work with nothing to show for it.
    pub fn compact(&mut self, all: bool) {
        let mut top = self.filters.len();
        while top > 1 {
            if !self.fold(top - 1) && !all {
                break;
            }
            top -= 1;
        }
        self.deletes = 0;
    }

    /// Move what it can out of filter `ix`, and drop the filter if it emptied.
    fn fold(&mut self, ix: usize) -> bool {
        let size = self.size();
        let mut left = false;
        for b in 0..self.filters[ix].buckets {
            for s in 0..size {
                let at = b as usize * size + s;
                let fp = self.filters[ix].data[at];
                if fp == EMPTY {
                    continue;
                }
                if self.relocate(ix, b, fp) {
                    self.filters[ix].data[at] = EMPTY;
                } else {
                    left = true;
                }
            }
        }
        // The walk comes down from the top, so a filter that empties is always
        // the last one and dropping it does not renumber anything.
        if left || ix + 1 != self.filters.len() {
            return false;
        }
        self.filters.pop();
        true
    }

    /// Find room for a fingerprint from bucket `bucket` of filter `ix` in one of
    /// the filters below it.
    ///
    /// The bucket number carries over: every filter's bucket count is a power of
    /// two and the smaller ones divide the bigger ones, so an item's bucket in a
    /// smaller filter is this bucket taken modulo that filter's count, and the
    /// alternate bucket comes off the fingerprint the same way it always does.
    /// Nothing here needs the item.
    fn relocate(&mut self, ix: usize, bucket: u64, fp: u8) -> bool {
        for f in 0..ix {
            let buckets = self.filters[f].buckets;
            let first = bucket % buckets;
            for at in [first, alt(first, fp) % buckets] {
                if let Some(free) = self.free(f, at) {
                    self.filters[f].data[free] = fp;
                    return true;
                }
            }
        }
        false
    }

    /// The first free slot in the bucket `h` lands in, if there is one.
    fn free(&self, f: usize, h: u64) -> Option<usize> {
        let size = self.size();
        let at = self.filters[f].at(h, size);
        (at..at + size).find(|&i| self.filters[f].data[i] == EMPTY)
    }

    /// The first slot in that bucket holding `fp`, if there is one.
    fn slot(&self, f: usize, h: u64, fp: u8) -> Option<usize> {
        let size = self.size();
        let at = self.filters[f].at(h, size);
        (at..at + size).find(|&i| self.filters[f].data[i] == fp)
    }

    /// How many slots in that bucket hold `fp`.
    fn matches(&self, f: usize, h: u64, fp: u8) -> u64 {
        let size = self.size();
        let at = self.filters[f].at(h, size);
        self.filters[f].data[at..at + size]
            .iter()
            .filter(|&&b| b == fp)
            .count() as u64
    }

    /// The header a `CF.SCANDUMP` at position zero answers with.
    ///
    /// Four counts and three widths, all little endian, which is the C struct
    /// written out. Everything else about the chain is derived from it: the
    /// filter widths are the base bucket count times the expansion to the power
    /// of the filter's position, so the header alone says how long every chunk
    /// that follows will be.
    #[must_use]
    pub fn header(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER);
        out.extend_from_slice(&self.items.to_le_bytes());
        out.extend_from_slice(&self.buckets.to_le_bytes());
        out.extend_from_slice(&self.deletes.to_le_bytes());
        out.extend_from_slice(&(self.filters.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.bucket_size.to_le_bytes());
        out.extend_from_slice(&self.max_iterations.to_le_bytes());
        out.extend_from_slice(&self.expansion.to_le_bytes());
        out
    }

    /// The bytes at `iter`, and the position to ask with next.
    ///
    /// `iter` is one past the offset into every filter's fingerprints laid end
    /// to end, so the first data chunk is asked for with the one the header came
    /// back with. A chunk never spans two filters. `(0, &[])` means there is
    /// nothing left.
    #[must_use]
    pub fn chunk(&self, iter: i64) -> (i64, &[u8]) {
        let Ok(offset) = u64::try_from(iter - 1) else {
            return (0, &[]);
        };
        let mut start = 0u64;
        for f in &self.filters {
            let end = start + f.data.len() as u64;
            if offset < end {
                let bytes = &f.data[(offset - start) as usize..];
                return (iter + bytes.len() as i64, bytes);
            }
            start = end;
        }
        (0, &[])
    }

    /// A chain shaped the way `header` says, with every slot empty.
    ///
    /// `None` for anything that is not a header this could have written. The
    /// widths have to be workable rather than merely present, because a chunk is
    /// about to be copied into them at an offset the header decided.
    #[must_use]
    pub fn from_header(header: &[u8]) -> Option<Cuckoo> {
        if header.len() != HEADER {
            return None;
        }
        // The length is exact, so every slice below is the width it has to be.
        let at8 = |at: usize| {
            let word: [u8; 8] = header[at..at + 8].try_into().expect("eight of them");
            u64::from_le_bytes(word)
        };
        let at2 = |at: usize| {
            let word: [u8; 2] = header[at..at + 2].try_into().expect("two of them");
            u16::from_le_bytes(word)
        };
        let items = at8(0);
        let buckets = at8(8);
        let deletes = at8(16);
        // The header carries the filter count in eight bytes and the reference
        // reads it into two, so a count of 100000 is a chain of 34464 and a
        // count of 65536 is no chain at all. That is a C assignment showing
        // through rather than a decision, and it is copied because a header is
        // either read the same way on both servers or it is not portable.
        let count = at8(24) as u16;
        let bucket_size = at2(32);
        let max_iterations = at2(34);
        let expansion = at2(36);
        if count == 0 || bucket_size == 0 || max_iterations == 0 {
            return None;
        }
        if bucket_size as i64 > MAX_BUCKET_SIZE || expansion as i64 > MAX_EXPANSION {
            return None;
        }
        // A loaded chain may have more filters than one that grew into place,
        // and its expansion need not be a power of two, because neither of
        // those is checked on the way in. The bucket count is, since a bucket
        // number in one filter is only a bucket number in a smaller one when
        // every count divides every other.
        if !buckets.is_power_of_two() {
            return None;
        }
        if count > 1 && expansion == 0 {
            return None;
        }
        let mut filters = Vec::with_capacity(count as usize);
        let mut width = buckets;
        let mut total = 0u64;
        for i in 0..count {
            if i > 0 {
                width = width.checked_mul(u64::from(expansion))?;
            }
            let len = width.checked_mul(u64::from(bucket_size))?;
            total = total.checked_add(len)?;
            if total > 64 << 30 || usize::try_from(total).is_err() {
                return None;
            }
            filters.push(Sub::new(width, bucket_size));
        }
        Some(Cuckoo {
            filters,
            items,
            deletes,
            buckets,
            bucket_size,
            max_iterations,
            expansion,
        })
    }

    /// Copy a chunk back in at the offset `iter` names.
    ///
    /// `iter` is what the dump answered with, so it is one past the end of the
    /// bytes rather than their start, and the start is worked out backwards from
    /// it. Unlike the Bloom filter's version there is one refusal rather than
    /// three, because the module has one sentence for all of them. So this
    /// answers whether the chunk went in rather than saying which way it did
    /// not: there is nothing on the wire that could tell the two apart.
    #[must_use]
    pub fn load(&mut self, iter: i64, data: &[u8]) -> bool {
        let Ok(offset) = u64::try_from(iter - data.len() as i64 - 1) else {
            return false;
        };
        let mut start = 0u64;
        for f in &mut self.filters {
            let end = start + f.data.len() as u64;
            if offset < end {
                let at = (offset - start) as usize;
                if at + data.len() > f.data.len() {
                    return false;
                }
                f.data[at..at + data.len()].copy_from_slice(data);
                return true;
            }
            start = end;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read off `CF.INFO` on a real Redis 8.10.1 with RedisBloom in it.
    #[test]
    fn the_geometry_is_the_reference_geometry() {
        // capacity, bucket size, then buckets and the size the module reports.
        let cases = [
            (1000, 2, 512, 1080),
            (4, 2, 2, 60),
            (5, 2, 2, 60),
            (8, 2, 4, 64),
            (100, 2, 64, 184),
            (1025, 2, 512, 1080),
            (100_000, 2, 65_536, 131_128),
            (100, 1, 128, 184),
            (100, 4, 32, 184),
            (2, 1, 2, 58),
        ];
        for (capacity, bucket_size, buckets, size) in cases {
            let c = Cuckoo::new(capacity, bucket_size, 20, 1);
            assert_eq!(
                (c.buckets(), c.reported_size()),
                (buckets, size),
                "capacity {capacity} in buckets of {bucket_size}"
            );
        }
    }

    /// An expansion is rounded up to a power of two, so three is four.
    #[test]
    fn the_expansion_is_rounded_the_way_the_reference_rounds_it() {
        let cases = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 4),
            (5, 8),
            (9, 16),
            (100, 128),
            (255, 256),
            (1000, 1024),
            (32_767, 32_768),
        ];
        for (asked, got) in cases {
            assert_eq!(Cuckoo::new(100, 2, 20, asked).expansion(), got, "{asked}");
        }
    }

    /// The bucket and the fingerprint a known item gets, which is the whole
    /// reason the hash is the one it is. Both came out of a `CF.SCANDUMP` on the
    /// reference with one item in the filter.
    #[test]
    fn an_item_lands_where_the_reference_puts_it() {
        let mut c = Cuckoo::new(1024, 1, 20, 1);
        c.insert(b"item0");
        let set: Vec<(usize, u8)> = c.filters[0]
            .data
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b != EMPTY)
            .map(|(i, &b)| (i, b))
            .collect();
        assert_eq!(set, vec![(389, 165)]);
    }

    /// The whole of a small filter after seven adds, byte for byte off the
    /// reference, which covers the kick and the slot it evicts.
    #[test]
    fn a_full_bucket_kicks_the_way_the_reference_kicks() {
        let mut c = Cuckoo::new(8, 1, 20, 0);
        let mut added = Vec::new();
        for i in 0..12 {
            added.push(c.insert(format!("i{i}").as_bytes()));
        }
        assert_eq!(
            &*c.filters[0].data,
            &[0xad, 0x70, 0xb6, 0xec, 0x4a, 0xd4, 0xbd, 0x7b]
        );
        let full = |i: usize| added[i] == Insert::Full;
        assert!(full(7) && full(8) && full(9) && full(11), "{added:?}");
        assert_eq!(added[10], Insert::Yes);
        assert_eq!(c.len(), 8);
    }

    #[test]
    fn what_went_in_comes_back_out() {
        let mut c = Cuckoo::new(1000, 2, 20, 1);
        for i in 0..1000 {
            assert_eq!(c.insert(format!("item{i}").as_bytes()), Insert::Yes);
        }
        for i in 0..1000 {
            assert!(c.contains(format!("item{i}").as_bytes()), "lost item{i}");
        }
        assert_eq!(c.len(), 1000);
    }

    /// The same item twice is two items, which is the difference between this
    /// and a Bloom filter and is what `CF.COUNT` is for.
    #[test]
    fn the_same_item_twice_is_two_items() {
        let mut c = Cuckoo::new(100, 2, 20, 1);
        assert_eq!(c.insert(b"x"), Insert::Yes);
        assert_eq!(c.insert(b"x"), Insert::Yes);
        assert_eq!(c.count(b"x"), 2);
        assert_eq!(c.insert_unique(b"x"), Insert::Exists);
        assert!(c.remove(b"x"));
        assert_eq!(c.count(b"x"), 1);
        assert!(c.remove(b"x"));
        assert!(!c.remove(b"x"));
        assert_eq!(c.len(), 0);
    }

    /// The chain grows by the expansion factor and the filters come out at the
    /// widths the reference dumps.
    #[test]
    fn a_full_chain_grows_by_the_expansion_factor() {
        let mut c = Cuckoo::new(8, 2, 20, 2);
        for i in 0..20 {
            assert_eq!(c.insert(format!("a{i}").as_bytes()), Insert::Yes);
        }
        assert_eq!(c.filters(), 3);
        assert_eq!(c.buckets(), 4);
        assert_eq!(c.reported_size(), 144);
        let widths: Vec<usize> = c.filters.iter().map(|f| f.data.len()).collect();
        assert_eq!(widths, vec![8, 16, 32]);
    }

    /// A chain that will not grow says no, and one that has grown as far as it
    /// is allowed says something else.
    #[test]
    fn a_chain_that_cannot_grow_says_which_wall_it_hit() {
        let mut fixed = Cuckoo::new(4, 1, 20, 0);
        for i in 0..4 {
            assert_eq!(fixed.insert(format!("f{i}").as_bytes()), Insert::Yes);
        }
        assert_eq!(fixed.insert(b"f4"), Insert::Full);

        let mut chain = Cuckoo::new(4, 1, 20, 1);
        let mut n = 0;
        loop {
            match chain.insert(format!("b{n}").as_bytes()) {
                Insert::Yes => n += 1,
                other => break assert_eq!(other, Insert::MaxFilters),
            }
        }
        assert_eq!((n, chain.filters()), (125, MAX_FILTERS));
        assert_eq!(chain.reported_size(), 680);

        // The wall is the filter count and not the room, so a chain that was
        // loaded with every filter it is allowed refuses an item straight into
        // an empty slot. The reference answers the same way.
        let mut loaded = Cuckoo::new(4, 1, 20, 1).header();
        loaded[24..32].copy_from_slice(&(MAX_FILTERS as u64).to_le_bytes());
        let mut empty = Cuckoo::from_header(&loaded).expect("a header it wrote");
        assert_eq!(empty.insert(b"b125"), Insert::MaxFilters);
    }

    /// Deleting past a tenth of what is in the chain pulls the newest filter
    /// back into the older ones, and the bytes afterwards are the reference's.
    #[test]
    fn enough_deletes_compact_the_chain_on_their_own() {
        let mut c = Cuckoo::new(8, 2, 20, 2);
        for i in 0..20 {
            c.insert(format!("a{i}").as_bytes());
        }
        assert!(c.remove(b"a0"));
        assert_eq!((c.filters(), c.deleted()), (3, 1));
        assert!(c.remove(b"a1"));
        assert_eq!((c.filters(), c.deleted(), c.len()), (2, 0, 18));
        let bytes: Vec<u8> = c
            .filters
            .iter()
            .flat_map(|f| f.data.iter().copied())
            .collect();
        assert_eq!(
            bytes,
            [
                0xc3, 0xf1, 0xbc, 0xea, 0x16, 0x96, 0x83, 0x47, 0x68, 0xe0, 0xfc, 0x00, 0xc4, 0xfc,
                0xc4, 0x00, 0x00, 0x00, 0xfb, 0x00, 0xe9, 0x00, 0x63, 0x33,
            ]
        );
        // And the command form keeps going down the chain where the delete
        // stopped, so an item that could not move a moment ago moves now.
        assert!(c.remove(b"a2"));
        c.compact(true);
        assert_eq!(c.filters[0].data[6], 0xc4);
        assert_eq!(c.deleted(), 0);
        for i in 3..20 {
            assert!(c.contains(format!("a{i}").as_bytes()), "lost a{i}");
        }
    }

    /// An item whose two buckets are the same bucket is counted twice, which is
    /// the reference's bug and is pinned here so that fixing it is a decision
    /// rather than an accident.
    #[test]
    fn an_item_in_one_bucket_twice_over_is_counted_twice() {
        let mut c = Cuckoo::new(8, 2, 20, 0);
        assert_eq!(c.insert(b"z0"), Insert::Yes);
        assert_eq!(c.count(b"z0"), 2);
        assert!(c.contains(b"z0"));
        assert!(c.remove(b"z0"));
        assert_eq!(c.count(b"z0"), 0);
    }

    #[test]
    fn a_dump_reloads_into_the_same_filter() {
        let mut c = Cuckoo::new(8, 2, 20, 2);
        for i in 0..20 {
            c.insert(format!("a{i}").as_bytes());
        }
        let mut copy = Cuckoo::from_header(&c.header()).expect("its own header");
        let mut iter = 1;
        loop {
            let (next, data) = c.chunk(iter);
            if next == 0 {
                break;
            }
            assert!(copy.load(next, data), "its own chunk");
            iter = next;
        }
        assert_eq!(copy.header(), c.header());
        for i in 0..20 {
            assert!(copy.contains(format!("a{i}").as_bytes()), "lost a{i}");
        }
        assert!(!copy.contains(b"nothing"));
    }

    /// The positions the reference answers for a three filter chain, which is
    /// what a client that already knows how to walk one depends on.
    #[test]
    fn the_position_is_the_running_byte_offset() {
        let mut c = Cuckoo::new(8, 2, 20, 2);
        for i in 0..20 {
            c.insert(format!("a{i}").as_bytes());
        }
        assert_eq!(c.header().len(), 38);
        let (next, data) = c.chunk(1);
        assert_eq!((next, data.len()), (9, 8));
        let (next, data) = c.chunk(next);
        assert_eq!((next, data.len()), (25, 16));
        let (next, data) = c.chunk(next);
        assert_eq!((next, data.len()), (57, 32));
        assert_eq!(c.chunk(next), (0, &[][..]));
    }

    #[test]
    fn a_header_that_was_not_written_by_a_filter_is_refused() {
        let good = Cuckoo::new(100, 2, 20, 2).header();
        assert!(Cuckoo::from_header(&good).is_some());
        assert!(Cuckoo::from_header(&[]).is_none());
        assert!(Cuckoo::from_header(&good[..37]).is_none());
        let mut none = good.clone();
        none[24..32].copy_from_slice(&0u64.to_le_bytes());
        assert!(Cuckoo::from_header(&none).is_none());
        // More filters than a chain could ever grow is fine on the way in, and
        // a count that is a multiple of the two bytes it is read into is not,
        // because it arrives as no filters at all. Both are the reference's.
        let mut many = Cuckoo::new(100, 2, 20, 1).header();
        many[24..32].copy_from_slice(&99u64.to_le_bytes());
        assert_eq!(Cuckoo::from_header(&many).map(|c| c.filters()), Some(99));
        let mut wrapped = good.clone();
        wrapped[24..32].copy_from_slice(&65_536u64.to_le_bytes());
        assert!(Cuckoo::from_header(&wrapped).is_none());
        let mut wide_growth = good.clone();
        wide_growth[36..38].copy_from_slice(&65_535u16.to_le_bytes());
        assert!(Cuckoo::from_header(&wide_growth).is_none());
        let mut odd = good.clone();
        odd[8..16].copy_from_slice(&100u64.to_le_bytes());
        assert!(Cuckoo::from_header(&odd).is_none());
        let mut wide = good.clone();
        wide[32..34].copy_from_slice(&0u16.to_le_bytes());
        assert!(Cuckoo::from_header(&wide).is_none());
    }

    /// The offsets the reference takes and the ones it refuses, on a chain of
    /// filters of eight, sixteen and thirty two bytes.
    #[test]
    fn a_chunk_that_does_not_fit_is_refused() {
        let mut c = Cuckoo::from_header(&Cuckoo::new(8, 2, 20, 2).header()).expect("header");
        while c.filters() < 3 {
            c.grow();
        }
        assert!(!c.load(2, &[0; 8]));
        assert!(!c.load(8, &[0; 8]));
        assert!(c.load(9, &[0; 8]));
        assert!(!c.load(10, &[0; 8]));
        assert!(c.load(24, &[0; 8]));
        assert!(c.load(25, &[0; 8]));
        assert!(c.load(56, &[0; 8]));
        assert!(c.load(57, &[0; 8]));
        assert!(!c.load(58, &[0; 8]));
        assert!(!c.load(100, &[0; 8]));
        assert!(!c.load(-5, &[0; 8]));
    }

    /// The false positive rate is near what a one byte fingerprint in buckets of
    /// two gives, which is the one property that is about the filter rather than
    /// about the format.
    #[test]
    fn the_error_rate_is_what_a_one_byte_fingerprint_gives() {
        let mut c = Cuckoo::new(16_384, 2, 20, 2);
        for i in 0..10_000 {
            c.insert(format!("in{i}").as_bytes());
        }
        let wrong = (0..10_000)
            .filter(|i| c.contains(format!("out{i}").as_bytes()))
            .count();
        assert!(wrong < 400, "{wrong} false positives in ten thousand");
    }
}
