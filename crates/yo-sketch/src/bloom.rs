//! A scaling Bloom filter, byte for byte the one RedisBloom holds.
//!
//! # Why a copy and not a better one
//!
//! There are better Bloom filters than this. Blocked filters touch one cache
//! line per lookup instead of nine scattered ones, and a register blocked
//! filter with SIMD beats this by a factor on every benchmark anyone has run
//! since 2019. None of that is worth having here, because the format is the
//! product.
//!
//! `BF.SCANDUMP` and `BF.LOADCHUNK` exist so that a filter can be moved between
//! servers, and the only reason to answer them at all is that the chunks are
//! interchangeable with a real Redis. That fixes the geometry, the growth rule,
//! the error tightening, the hash and the bit order all at once: change any of
//! them and a chunk written here is silently a different set over there, which
//! is worse than not answering the command. So the arithmetic below is
//! RedisBloom's arithmetic, down to the hardcoded `ln(2)^2` it divides by and
//! the truncation to an integer number of bits before the rounding up to eight
//! bytes.
//!
//! Where there is room to be faster without moving a bit, this takes it: the
//! two hashes are computed once for the whole chain rather than once per link,
//! and a lookup stops at the first link that answers.
//!
//! # The chain
//!
//! One filter has a capacity fixed when it is made, and its error rate is only
//! the promised one up to that capacity. Almeida's scalable filter is the usual
//! answer: when the last link fills, add another one that is `growth` times
//! bigger and half as wrong, and answer a lookup with the union. The halving is
//! what keeps the error of the whole chain bounded, since the sum of a
//! geometric series with ratio a half converges to twice the first term.
//!
//! An item goes into the newest link and nowhere else, so the chain never
//! rewrites what it has already written and `BF.ADD` stays a handful of
//! nanoseconds whatever the chain has grown to. A lookup walks every link, so
//! the read side is what the growth costs, which is the trade Redis made and
//! the reason `BF.RESERVE` with a capacity you actually expect is worth doing.

use crate::hash;

/// `ln(2)^2`, written out rather than computed.
///
/// RedisBloom has this literal in `calc_bpe` and the last digit of it decides
/// how many bits a filter has, so computing `2f64.ln().powi(2)` here would be
/// the more honest looking way to disagree with it.
const LN2_SQUARED: f64 = 0.480_453_013_918_201;
/// `ln(2)`, for the number of hash functions. Same argument.
///
/// Clippy is right that `std::f64::consts::LN_2` is this number and is more
/// digits of it, and using it would be wrong here for exactly that reason. The
/// literal ends at the fifteenth digit because RedisBloom's does, and a hash
/// count is a `ceil` of a product with it, so a sixteenth digit is one filter
/// somewhere with an extra hash function and a different set of bits.
#[expect(
    clippy::approx_constant,
    reason = "the reference's fifteen digits and not the true value"
)]
const LN2: f64 = 0.693_147_180_559_945;
/// What a new link's error rate is multiplied by, so the chain's total error
/// converges instead of adding up.
const TIGHTENING: f64 = 0.5;

/// The options word a filter this engine builds always carries.
///
/// RedisBloom passes `BLOOM_OPT_NOROUND | BLOOM_OPT_FORCE64`, which is the two
/// decisions the bit indexing depends on: sizes are not rounded up to a power of
/// two, so a position is a modulo rather than a mask, and the hash is the 64 bit
/// one rather than the 32 bit one whatever the filter's size. Both are in the
/// dumped header, so a reader can tell, and neither is configurable.
const OPT_BASE: u32 = 5;
/// The bit that says a full chain answers an error rather than growing.
const OPT_NO_SCALING: u32 = 8;

/// Bytes of chain header at the front of a dump.
const CHAIN_HEADER: usize = 20;
/// Bytes of per link header after it.
const LINK_HEADER: usize = 53;

/// The smallest capacity `BF.RESERVE` accepts.
pub const MIN_CAPACITY: i64 = 1;
/// The largest, which is a gigabyte of items and about 1.4 GB of filter at the
/// default error rate.
pub const MAX_CAPACITY: i64 = 1_073_741_824;
/// The largest growth factor, which would take a chain from one item to the
/// capacity of the whole server in three links.
pub const MAX_EXPANSION: i64 = 32_768;

/// What happened to an item that was offered to a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Added {
    /// It was not there and now it is.
    Yes,
    /// Something already hashed to those bits, so nothing was written.
    Already,
    /// The last link is full and the chain was told not to grow.
    Full,
}

/// Why a chunk did not go back in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Load {
    /// The iterator is smaller than the chunk it came with, so it could not be
    /// one past the end of it and nothing sensible can be worked out from the
    /// pair.
    BadData,
    /// The offset is not inside any link's bit array.
    NoLink,
    /// It starts inside one but runs off the end of it.
    TooBig,
}

/// What `BF.DEBUG` says about one link.
#[derive(Debug, Clone, Copy)]
pub struct LinkInfo {
    /// The bit array's length.
    pub bytes: u64,
    /// Eight times that, which is what the modulo is taken against.
    pub bits: u64,
    /// How many positions an item sets.
    pub hashes: u32,
    /// How many items this link was sized for.
    pub capacity: u64,
    /// How many are in it.
    pub size: u64,
    /// The error rate it was sized for, which is not the one the client asked
    /// for except on the first link of a non scaling chain.
    pub error: f64,
}

/// One filter in the chain.
#[derive(Debug, Clone)]
struct Link {
    /// The bit array, `bytes` long and indexed little end first inside a byte.
    bits: Box<[u8]>,
    /// How many positions an item sets, which is `ceil(ln(2) * bpe)`.
    hashes: u32,
    /// The capacity this link was sized for.
    entries: u64,
    /// The error rate it was sized for.
    error: f64,
    /// Bits per element, kept because the dump carries it and a reader is
    /// entitled to it rather than to a recomputation that might differ in the
    /// last place.
    bpe: f64,
    /// How many items have gone in.
    size: u64,
}

impl Link {
    /// A link sized for `entries` items at `error`.
    fn new(entries: u64, error: f64) -> Link {
        let bpe = -error.ln() / LN2_SQUARED;
        // Truncated, then rounded up to a whole number of eight byte words, and
        // only then turned back into a bit count. Doing the rounding in bits
        // rather than in bytes is what makes a one item filter 64 bits wide.
        let want = (entries as f64 * bpe) as u64;
        let bytes = if want.is_multiple_of(64) {
            want / 8
        } else {
            ((want / 64) + 1) * 8
        };
        Link {
            bits: vec![0u8; bytes as usize].into_boxed_slice(),
            hashes: (LN2 * bpe).ceil() as u32,
            entries,
            error,
            bpe,
            size: 0,
        }
    }

    /// How wide the bit array is.
    fn bits(&self) -> u64 {
        self.bits.len() as u64 * 8
    }

    /// The `i`th bit position for an item whose two hashes are `a` and `b`.
    ///
    /// The addition wraps at 64 bits before the modulo, which is not an
    /// accident anybody could have designed around: it is what the C does, and
    /// a filter that took the modulo first would set different bits for the
    /// same item.
    fn at(&self, (a, b): (u64, u64), i: u32) -> u64 {
        a.wrapping_add(u64::from(i).wrapping_mul(b)) % self.bits()
    }

    /// Whether every one of this link's positions is set.
    fn contains(&self, h: (u64, u64)) -> bool {
        (0..self.hashes).all(|i| {
            let x = self.at(h, i);
            self.bits[(x >> 3) as usize] & (1 << (x & 7)) != 0
        })
    }

    /// Set every one of them.
    fn insert(&mut self, h: (u64, u64)) {
        for i in 0..self.hashes {
            let x = self.at(h, i);
            self.bits[(x >> 3) as usize] |= 1 << (x & 7);
        }
    }
}

/// A chain of Bloom filters under one name.
#[derive(Debug, Clone)]
pub struct Bloom {
    /// Oldest first, so the newest is the last and is the one writes go to.
    links: Vec<Link>,
    /// The word the dump carries, which says how the bits are indexed and
    /// whether the chain may grow.
    options: u32,
    /// How much bigger each link is than the one before it.
    growth: u32,
    /// How many items have gone into the whole chain.
    size: u64,
}

impl Bloom {
    /// A chain of one link, sized for `capacity` items at `error`.
    ///
    /// `growth` of zero means the chain will not grow, and so does `nonscaling`.
    /// They are two spellings of the same thing to a client and stay two fields
    /// here because the dumped header carries both and a real Redis writes a
    /// growth of two beside the no scaling bit when the client said `NONSCALING`.
    ///
    /// The error rate the first link is built at is half the one asked for
    /// unless the chain cannot grow, because the rest of the chain's error is
    /// going to be added to it and the halving is what leaves room.
    #[must_use]
    pub fn new(capacity: u64, error: f64, growth: u32, nonscaling: bool) -> Bloom {
        let fixed = nonscaling || growth == 0;
        let options = if fixed {
            OPT_BASE | OPT_NO_SCALING
        } else {
            OPT_BASE
        };
        Bloom {
            links: vec![Link::new(
                capacity,
                if fixed { error } else { error * TIGHTENING },
            )],
            options,
            growth,
            size: 0,
        }
    }

    /// Whether the chain will add a link when the last one fills.
    #[must_use]
    pub fn scales(&self) -> bool {
        self.options & OPT_NO_SCALING == 0
    }

    /// The growth factor, or `None` for a chain that will not grow.
    ///
    /// `BF.INFO` reports the nil for both spellings of a fixed chain, so the
    /// two are one answer here rather than two.
    #[must_use]
    pub fn expansion(&self) -> Option<u32> {
        self.scales().then_some(self.growth)
    }

    /// How many items have gone in.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.size
    }

    /// Whether nothing has gone in yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// The capacities of every link added up, which is what `BF.INFO` calls the
    /// capacity and is not the capacity of anything: the chain will keep taking
    /// items past it, it just adds a link to do so.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.links.iter().map(|l| l.entries).sum()
    }

    /// How many links there are.
    #[must_use]
    pub fn filters(&self) -> usize {
        self.links.len()
    }

    /// What `BF.INFO` reports as the size.
    ///
    /// RedisBloom's own accounting: its chain struct, its link structs and the
    /// bit arrays. It is not what this holds, and it is the number a client
    /// compares against a real server, so it is computed rather than measured.
    #[must_use]
    pub fn reported_size(&self) -> u64 {
        32 + 64 * self.links.len() as u64
            + self.links.iter().map(|l| l.bits.len() as u64).sum::<u64>()
    }

    /// What this actually holds, for `MEMORY USAGE` and for `maxmemory`.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        size_of::<Bloom>()
            + self.links.capacity() * size_of::<Link>()
            + self.links.iter().map(|l| l.bits.len()).sum::<usize>()
    }

    /// What `BF.DEBUG` walks.
    pub fn links(&self) -> impl Iterator<Item = LinkInfo> + '_ {
        self.links.iter().map(|l| LinkInfo {
            bytes: l.bits.len() as u64,
            bits: l.bits(),
            hashes: l.hashes,
            capacity: l.entries,
            size: l.size,
            error: l.error,
        })
    }

    /// Whether the chain has seen `item`, allowing for the false positives it
    /// was sized for.
    #[must_use]
    pub fn contains(&self, item: &[u8]) -> bool {
        let h = hash::pair(item);
        self.has(h)
    }

    /// The same, for a pair of hashes that has already been taken.
    fn has(&self, h: (u64, u64)) -> bool {
        // Newest first, because a filter that is being filled right now is
        // where the item most recently offered went.
        self.links.iter().rev().any(|l| l.contains(h))
    }

    /// Put `item` in, growing the chain if the last link is full.
    pub fn add(&mut self, item: &[u8]) -> Added {
        let h = hash::pair(item);
        if self.has(h) {
            return Added::Already;
        }
        let last = self.links.len() - 1;
        if self.links[last].size >= self.links[last].entries {
            if !self.scales() {
                return Added::Full;
            }
            let (entries, error) = (
                self.links[last].entries * u64::from(self.growth),
                self.links[last].error * TIGHTENING,
            );
            self.links.push(Link::new(entries, error));
        }
        let last = self.links.len() - 1;
        self.links[last].insert(h);
        self.links[last].size += 1;
        self.size += 1;
        Added::Yes
    }

    /// The header a `BF.SCANDUMP` at iterator zero answers with.
    ///
    /// Twenty bytes of chain and fifty three per link, all little endian, which
    /// is the C structs written out in the order the compiler laid them out on
    /// x86-64. A reader gets the whole shape of the chain from this and then
    /// needs only the bit arrays.
    #[must_use]
    pub fn header(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CHAIN_HEADER + LINK_HEADER * self.links.len());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.extend_from_slice(&(self.links.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.options.to_le_bytes());
        out.extend_from_slice(&self.growth.to_le_bytes());
        for l in &self.links {
            out.extend_from_slice(&(l.bits.len() as u64).to_le_bytes());
            out.extend_from_slice(&l.bits().to_le_bytes());
            out.extend_from_slice(&l.size.to_le_bytes());
            out.extend_from_slice(&l.error.to_le_bytes());
            out.extend_from_slice(&l.bpe.to_le_bytes());
            out.extend_from_slice(&l.hashes.to_le_bytes());
            out.extend_from_slice(&l.entries.to_le_bytes());
            // `n2`, the shift a power of two sized filter would mask with. It is
            // zero for everything this engine or RedisBloom builds, because both
            // pass the option that says not to round the size up.
            out.push(0);
        }
        out
    }

    /// The bytes at `iter`, and the iterator to ask with next.
    ///
    /// `iter` is one past the offset into every link's bit array laid end to
    /// end, so the first data chunk is asked for with the one the header came
    /// back with. A chunk never spans two links, so a chain of `n` links takes
    /// `n + 2` calls to walk. `(0, &[])` means there is nothing left.
    #[must_use]
    pub fn chunk(&self, iter: i64) -> (i64, &[u8]) {
        let Ok(offset) = u64::try_from(iter - 1) else {
            return (0, &[]);
        };
        let mut start = 0u64;
        for l in &self.links {
            let end = start + l.bits.len() as u64;
            if offset < end {
                let bytes = &l.bits[(offset - start) as usize..];
                return (iter + bytes.len() as i64, bytes);
            }
            start = end;
        }
        (0, &[])
    }

    /// A chain shaped the way `header` says, with every bit clear.
    ///
    /// `None` for anything that is not a header this could have written. The
    /// length has to be exactly right, which is the check that catches a client
    /// sending the chunks in the wrong order or sending something else
    /// entirely.
    #[must_use]
    pub fn from_header(header: &[u8]) -> Option<Bloom> {
        if header.len() < CHAIN_HEADER {
            return None;
        }
        let size = u64::from_le_bytes(header[0..8].try_into().ok()?);
        let nfilters = u32::from_le_bytes(header[8..12].try_into().ok()?);
        let options = u32::from_le_bytes(header[12..16].try_into().ok()?);
        let growth = u32::from_le_bytes(header[16..20].try_into().ok()?);
        if nfilters == 0 || header.len() != CHAIN_HEADER + LINK_HEADER * nfilters as usize {
            return None;
        }
        let mut links = Vec::with_capacity(nfilters as usize);
        let mut total = 0u64;
        for i in 0..nfilters as usize {
            let at = CHAIN_HEADER + LINK_HEADER * i;
            let f = |lo: usize, hi: usize| -> Option<[u8; 8]> {
                header[at + lo..at + hi].try_into().ok()
            };
            let bytes = u64::from_le_bytes(f(0, 8)?);
            let bits = u64::from_le_bytes(f(8, 16)?);
            let used = u64::from_le_bytes(f(16, 24)?);
            let error = f64::from_le_bytes(f(24, 32)?);
            let bpe = f64::from_le_bytes(f(32, 40)?);
            let hashes = u32::from_le_bytes(header[at + 40..at + 44].try_into().ok()?);
            let entries = u64::from_le_bytes(f(44, 52)?);
            // Everything a lookup depends on has to hold together, or a chunk
            // that is loaded on top of it indexes off the end of the array and
            // the answers are somebody else's memory. `n2` is not checked, only
            // ignored: this indexes with a modulo whatever it says.
            if bits != bytes * 8 || hashes == 0 || bytes == 0 || used > entries {
                return None;
            }
            // A gigabyte of items at the tightest error rate anyone can ask for
            // is about 40 GB of filter, so anything past that is a length that
            // was never written by a filter.
            total = total.checked_add(bytes)?;
            if total > 64 << 30 {
                return None;
            }
            links.push(Link {
                bits: vec![0u8; bytes as usize].into_boxed_slice(),
                hashes,
                entries,
                error,
                bpe,
                size: used,
            });
        }
        Some(Bloom {
            links,
            options,
            growth,
            size,
        })
    }

    /// Copy a chunk back in at the offset `iter` names.
    ///
    /// `iter` is what the dump answered with, so it is one past the end of the
    /// bytes rather than their start. That is not a design, it is what the
    /// command sends, and working the start out backwards from it is what makes
    /// a chunk loadable without the client having tracked anything.
    ///
    /// The three ways it can go wrong are three different sentences on the
    /// wire, and which one a client sees is worth getting right because they
    /// mean different things. An iterator smaller than its own chunk cannot
    /// have come from a dump at all, an offset outside every link is a chunk
    /// from a differently shaped filter, and one that starts inside a link and
    /// runs off it is a chunk from a bigger version of this one.
    pub fn load(&mut self, iter: i64, data: &[u8]) -> Result<(), Load> {
        let len = data.len() as i64;
        if iter < len {
            return Err(Load::BadData);
        }
        // Exactly equal leaves an offset of minus one, which is a real offset
        // to the reference because it works in unsigned and lands nowhere.
        let Ok(offset) = u64::try_from(iter - len - 1) else {
            return Err(Load::NoLink);
        };
        let mut start = 0u64;
        for l in &mut self.links {
            let end = start + l.bits.len() as u64;
            if offset < end {
                let at = (offset - start) as usize;
                if at + data.len() > l.bits.len() {
                    return Err(Load::TooBig);
                }
                l.bits[at..at + data.len()].copy_from_slice(data);
                return Ok(());
            }
            start = end;
        }
        Err(Load::NoLink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The geometry of every filter the reference was asked to build, read back
    /// out of `BF.DEBUG` on a real Redis 8.10.1 with RedisBloom in it.
    #[test]
    fn the_geometry_is_the_reference_geometry() {
        // capacity, error as asked, nonscaling, then bytes, bits and hashes.
        let cases = [
            (100, 0.01, false, 144, 1152, 8),
            (10, 0.01, false, 16, 128, 8),
            (10, 0.01, true, 16, 128, 7),
            (100, 0.01, true, 120, 960, 7),
            (1, 0.01, false, 8, 64, 8),
            (3, 0.01, true, 8, 64, 7),
            (50, 0.001, false, 104, 832, 11),
            (10, 0.000_000_000_1, false, 64, 512, 35),
            (1_000_000, 0.01, false, 1_378_472, 11_027_776, 8),
        ];
        for (capacity, error, fixed, bytes, bits, hashes) in cases {
            let b = Bloom::new(capacity, error, 2, fixed);
            let l = b.links().next().expect("a chain has a first link");
            assert_eq!(
                (l.bytes, l.bits, l.hashes),
                (bytes, bits, hashes),
                "capacity {capacity} at {error}"
            );
        }
    }

    /// The bits a known item sets in a known filter, which is the whole reason
    /// the hash is the one it is.
    #[test]
    fn an_item_lands_where_the_reference_puts_it() {
        let mut b = Bloom::new(100, 0.01, 2, false);
        b.add(b"hello");
        let set: Vec<u64> = (0..1152u64)
            .filter(|x| {
                let l = &b.links[0];
                l.bits[(x >> 3) as usize] & (1 << (x & 7)) != 0
            })
            .collect();
        assert_eq!(set, vec![77, 97, 497, 517, 537, 789, 809, 957]);
    }

    #[test]
    fn what_went_in_comes_back_out() {
        let mut b = Bloom::new(1000, 0.001, 2, false);
        for i in 0..1000 {
            assert_eq!(b.add(format!("item{i}").as_bytes()), Added::Yes);
        }
        for i in 0..1000 {
            assert!(b.contains(format!("item{i}").as_bytes()), "lost item{i}");
        }
        assert_eq!(b.len(), 1000);
        assert_eq!(b.filters(), 1);
    }

    #[test]
    fn the_same_item_twice_is_one_item() {
        let mut b = Bloom::new(100, 0.01, 2, false);
        assert_eq!(b.add(b"x"), Added::Yes);
        assert_eq!(b.add(b"x"), Added::Already);
        assert_eq!(b.len(), 1);
    }

    /// The link sizes and the total capacity a growth of two produces, which is
    /// the sequence `BF.INFO` reports on the reference.
    #[test]
    fn a_full_chain_grows_by_the_expansion_factor() {
        let mut b = Bloom::new(10, 0.01, 2, false);
        for i in 0..25 {
            b.add(format!("i{i}").as_bytes());
        }
        assert_eq!(b.filters(), 2);
        assert_eq!(b.capacity(), 30);
        assert_eq!(b.len(), 25);
        assert_eq!(b.reported_size(), 208);
        let sizes: Vec<(u64, u64, u32)> =
            b.links().map(|l| (l.capacity, l.size, l.hashes)).collect();
        assert_eq!(sizes, vec![(10, 10, 8), (20, 15, 9)]);
    }

    /// The error rate halves per link, so the chain's total error stays under
    /// twice the first link's rather than growing without bound.
    #[test]
    fn each_link_is_half_as_wrong_as_the_one_before() {
        let mut b = Bloom::new(10, 0.01, 2, false);
        for i in 0..80 {
            b.add(format!("i{i}").as_bytes());
        }
        let errors: Vec<f64> = b.links().map(|l| l.error).collect();
        assert_eq!(errors, vec![0.005, 0.0025, 0.00125, 0.000625]);
    }

    #[test]
    fn a_fixed_chain_says_no_rather_than_growing() {
        let mut b = Bloom::new(2, 0.01, 2, true);
        assert_eq!(b.add(b"a"), Added::Yes);
        assert_eq!(b.add(b"b"), Added::Yes);
        // The membership check comes first, so an item that is already in a
        // full filter is still reported as a duplicate and not as a failure.
        assert_eq!(b.add(b"a"), Added::Already);
        assert_eq!(b.add(b"c"), Added::Full);
        assert_eq!(b.filters(), 1);
        assert!(!b.contains(b"c"));
    }

    #[test]
    fn an_expansion_of_zero_is_a_fixed_chain() {
        let b = Bloom::new(10, 0.01, 0, false);
        assert!(!b.scales());
        assert_eq!(b.expansion(), None);
        // And it is sized at the rate the client asked for, not half of it,
        // because there is no second link to leave room for.
        assert_eq!(b.links().next().expect("one link").error, 0.01);
    }

    #[test]
    fn a_dump_reloads_into_the_same_filter() {
        let mut b = Bloom::new(10, 0.01, 2, false);
        for i in 0..25 {
            b.add(format!("i{i}").as_bytes());
        }
        let mut copy = Bloom::from_header(&b.header()).expect("its own header");
        let mut iter = 1;
        loop {
            let (next, data) = b.chunk(iter);
            if next == 0 {
                break;
            }
            copy.load(next, data).expect("its own chunk");
            iter = next;
        }
        assert_eq!(copy.header(), b.header());
        for i in 0..25 {
            assert!(copy.contains(format!("i{i}").as_bytes()), "lost i{i}");
        }
        assert!(!copy.contains(b"nothing"));
    }

    /// A chain of two links answers the iterators the reference answers, which
    /// is what a client that already knows how to walk one depends on.
    #[test]
    fn the_iterator_is_the_running_byte_offset() {
        let mut b = Bloom::new(10, 0.01, 2, false);
        for i in 0..25 {
            b.add(format!("i{i}").as_bytes());
        }
        assert_eq!(b.header().len(), 126);
        let (next, data) = b.chunk(1);
        assert_eq!((next, data.len()), (17, 16));
        let (next, data) = b.chunk(next);
        assert_eq!((next, data.len()), (49, 32));
        assert_eq!(b.chunk(next), (0, &[][..]));
    }

    #[test]
    fn a_header_that_was_not_written_by_a_filter_is_refused() {
        let good = Bloom::new(10, 0.01, 2, false).header();
        assert!(Bloom::from_header(&good).is_some());
        assert!(Bloom::from_header(&[]).is_none());
        assert!(Bloom::from_header(&good[..good.len() - 1]).is_none());
        let mut long = good.clone();
        long.push(0);
        assert!(Bloom::from_header(&long).is_none());
        let mut many = good.clone();
        many[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(Bloom::from_header(&many).is_none());
        let mut none = good.clone();
        none[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert!(Bloom::from_header(&none).is_none());
        // Bits that do not match the byte count would index off the end.
        let mut skew = good.clone();
        skew[28..36].copy_from_slice(&1u64.to_le_bytes());
        assert!(Bloom::from_header(&skew).is_none());
    }

    #[test]
    fn a_chunk_that_does_not_fit_is_refused() {
        let mut b = Bloom::from_header(&Bloom::new(10, 0.01, 2, false).header()).expect("header");
        assert_eq!(b.load(99, &[0; 16]), Err(Load::NoLink));
        assert_eq!(b.load(65, &[0; 16]), Err(Load::NoLink));
        assert_eq!(b.load(25, &[0; 24]), Err(Load::TooBig));
        assert_eq!(b.load(17, &[0; 16]), Ok(()));
        assert_eq!(b.load(9, &[0; 8]), Ok(()));
    }

    /// The false positive rate is close to what was asked for, which is the one
    /// property that is about the filter rather than about the format.
    #[test]
    fn the_error_rate_is_near_the_one_that_was_asked_for() {
        let mut b = Bloom::new(10_000, 0.01, 2, false);
        for i in 0..10_000 {
            b.add(format!("in{i}").as_bytes());
        }
        let wrong = (0..10_000)
            .filter(|i| b.contains(format!("out{i}").as_bytes()))
            .count();
        assert!(wrong < 100, "{wrong} false positives in ten thousand");
    }
}
