//! The hash the bloom filters use, which is Austin Appleby's MurmurHash64A.
//!
//! This is not a hash chosen on its merits. It is the one RedisBloom compiled
//! in, and a filter is only portable between two servers if both of them put a
//! given item in the same bits, so the hash is part of the file format rather
//! than an implementation detail. `wyhash` is faster and would make every
//! `BF.SCANDUMP` this engine writes unreadable by anything else, which is the
//! whole point of writing one.
//!
//! The 64 bit variant reads the key eight bytes at a time as a little endian
//! integer, so it answers differently on a big endian machine. RedisBloom has
//! the same property and nobody ships Redis on one, so the two agree everywhere
//! either of them runs.

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
}
