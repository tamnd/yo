//! The hashes the sketches use, which are Austin Appleby's MurmurHash2 in both
//! of its widths.
//!
//! Neither is a hash chosen on its merits. They are the ones RedisBloom compiled
//! in, and a filter is only portable between two servers if both of them put a
//! given item in the same bits, so the hash is part of the file format rather
//! than an implementation detail. `wyhash` is faster and would make every
//! `BF.SCANDUMP` this engine writes unreadable by anything else, which is the
//! whole point of writing one.
//!
//! The Bloom and cuckoo filters take the 64 bit variant and the count min
//! sketch takes the 32 bit one, which is not a considered choice on the module's
//! part either: the two structures were written years apart by different people
//! against the same header. The 32 bit one is not the low half of the 64 bit
//! one and there is no relation between the two, so both are here.
//!
//! Both read the key a word at a time as a little endian integer, so they answer
//! differently on a big endian machine. RedisBloom has the same property and
//! nobody ships Redis on one, so the two agree everywhere either of them runs.

/// The mixing constant, which doubles as the seed of the first of the two
/// hashes a filter takes.
pub const M: u64 = 0xc6a4_a793_5bd1_e995;

/// `MurmurHash64A` over `key`, seeded with `seed`.
#[must_use]
pub fn murmur64a(key: &[u8], seed: u64) -> u64 {
    const R: u32 = 47;
    let mut h = seed ^ (key.len() as u64).wrapping_mul(M);

    let (words, tail) = key.as_chunks::<8>();
    for chunk in words {
        let mut k = u64::from_le_bytes(*chunk);
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }

    // The tail falls through in C, so the bytes are folded in from the top down
    // and the multiply happens once at the end of the run rather than per byte.
    if !tail.is_empty() {
        for (i, &b) in tail.iter().enumerate() {
            h ^= u64::from(b) << (8 * i);
        }
        h = h.wrapping_mul(M);
    }

    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// `MurmurHash2` in its 32 bit shape, seeded with `seed`.
///
/// The count min sketch calls this once per row with the row number as the
/// seed, which is the whole of its hashing: no double hashing, no second
/// function, just the same hash run `depth` times over the same bytes. That is
/// more work than the Bloom filter does for the same number of positions and it
/// is what the module does.
///
/// It is not the 64 bit function truncated. The mixing constant is the low half
/// of the other one's, the shifts are different, and the two answer different
/// things about the same key, which is why both are in this file.
#[must_use]
pub fn murmur2_32(key: &[u8], seed: u32) -> u32 {
    const M32: u32 = 0x5bd1_e995;
    const R: u32 = 24;
    // The length is mixed in as an `int` in the C, so a key longer than two
    // gibibytes folds in a negative number. The bits are the same either way.
    let mut h = seed ^ (key.len() as u32);

    let (words, tail) = key.as_chunks::<4>();
    for chunk in words {
        let mut k = u32::from_le_bytes(*chunk);
        k = k.wrapping_mul(M32);
        k ^= k >> R;
        k = k.wrapping_mul(M32);
        // The accumulator is multiplied before the word goes in here and after
        // it in the 64 bit function. Swapping the two lines produces a hash
        // that looks just as good and puts every item in the wrong cell.
        h = h.wrapping_mul(M32);
        h ^= k;
    }

    if !tail.is_empty() {
        for (i, &b) in tail.iter().enumerate() {
            h ^= u32::from(b) << (8 * i);
        }
        h = h.wrapping_mul(M32);
    }

    h ^= h >> 13;
    h = h.wrapping_mul(M32);
    h ^= h >> 15;
    h
}

/// The two hashes a filter derives all of its bit positions from.
///
/// Kirsch and Mitzenmacher's double hashing: the `i`th position is `a + i * b`,
/// so a filter with nine hash functions costs two hashes and nine multiplies
/// rather than nine hashes. The second seed is the first answer, which is what
/// RedisBloom does and is why the pair cannot be computed in parallel.
#[must_use]
pub fn pair(item: &[u8]) -> (u64, u64) {
    let a = murmur64a(item, M);
    let b = murmur64a(item, a);
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers came out of the C in RedisBloom's `murmur2/MurmurHash2.c`
    /// compiled and run against the same input, not out of this file.
    #[test]
    fn the_hash_pair_is_the_one_redisbloom_takes() {
        let (a, b) = pair(b"hello");
        assert_eq!(a, 6_603_887_449_968_207_513);
        assert_eq!(b, 12_093_380_876_958_745_252);
    }

    /// A tail of every length, so the fall through in the C is covered byte by
    /// byte rather than only at zero and seven.
    #[test]
    fn every_tail_length_lands_somewhere_different() {
        let mut seen = Vec::new();
        for n in 0..24usize {
            let key: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let h = murmur64a(&key, M);
            assert!(!seen.contains(&h), "{n} bytes collided with a shorter key");
            seen.push(h);
        }
    }

    /// The seed is mixed in rather than added at the end, so two seeds over the
    /// same key have nothing in common.
    #[test]
    fn the_seed_changes_the_answer() {
        assert_ne!(murmur64a(b"item", M), murmur64a(b"item", 0));
    }

    /// These came off a live Redis 8.10.1 with RedisBloom in it, by building a
    /// sixteen by four sketch and working backwards from which cells three
    /// hundred `CMS.INCRBY` calls landed in. Every one of the three hundred
    /// agreed, which is what pins the function rather than these four numbers
    /// on their own.
    #[test]
    fn the_thirty_two_bit_hash_is_the_one_the_sketch_takes() {
        assert_eq!(murmur2_32(b"hello", 0), 3_848_350_155);
        assert_eq!(murmur2_32(b"hello", 1), 2_788_266_382);
        assert_eq!(murmur2_32(b"abc", 7), 957_085_255);
        // An empty key with seed zero has nothing to mix, so the tail is
        // skipped and the finish runs over a zero.
        assert_eq!(murmur2_32(b"", 0), 0);
    }

    /// The two widths are unrelated, so the narrow one is not the wide one cut
    /// in half however tempting that would be to assume.
    #[test]
    fn the_narrow_hash_is_not_the_wide_one_truncated() {
        let wide = murmur64a(b"item", 0) as u32;
        assert_ne!(murmur2_32(b"item", 0), wide);
    }

    /// The same tail walk as the 64 bit one, three bytes at a time instead of
    /// seven.
    #[test]
    fn every_narrow_tail_length_lands_somewhere_different() {
        let mut seen = Vec::new();
        for n in 0..16usize {
            let key: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let h = murmur2_32(&key, 0);
            assert!(!seen.contains(&h), "{n} bytes collided with a shorter key");
            seen.push(h);
        }
    }
}
