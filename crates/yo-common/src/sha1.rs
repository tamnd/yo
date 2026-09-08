//! SHA-1, because a script is named by the digest of its body.
//!
//! Nobody would choose SHA-1 for anything today and nothing here is choosing
//! it. `SCRIPT LOAD` answers a name, `EVALSHA` takes that name, and clients
//! have those names written down, so the digest is part of the protocol rather
//! than a security decision that could be revisited. `redis.sha1hex` is the
//! same digest offered to the script, and a script that computes a name and
//! then calls it has to get the same forty characters a client would.
//!
//! The second caller is the keyspace digest behind `DEBUG DIGEST`, where the
//! same reasoning applies for the same reason: that number exists to be compared
//! against another server's, so it has to be the number the other one would have
//! computed.
//!
//! Sixty lines rather than a dependency. This is the whole of the algorithm and
//! it is a fixed one, so there is nothing here that will need updating and
//! nothing that a crate would do differently.

/// The eighty round constants, four of them repeated twenty times each.
const K: [u32; 4] = [0x5a82_7999, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6];

/// The five words the standard starts from.
const H: [u32; 5] = [
    0x6745_2301,
    0xefcd_ab89,
    0x98ba_dcfe,
    0x1032_5476,
    0xc3d2_e1f0,
];

/// The twenty byte digest of `data`.
#[must_use]
pub fn digest(data: &[u8]) -> [u8; 20] {
    let mut h = H;
    // The message, its terminator, the padding and the length, walked as
    // sixty four byte blocks without ever building the padded copy. The last
    // one or two blocks are the only ones that are not entirely message, so
    // they are the only ones written into the scratch buffer.
    let (blocks, tail) = data.as_chunks::<64>();
    for block in blocks {
        round(&mut h, block);
    }
    let bits = (data.len() as u64).wrapping_mul(8);
    let mut last = [0u8; 128];
    last[..tail.len()].copy_from_slice(tail);
    last[tail.len()] = 0x80;
    // Eight bytes for the length and one for the terminator, so a tail with
    // fifty six bytes or more in it needs a second block to put them in.
    let taken = if tail.len() < 56 { 64 } else { 128 };
    last[taken - 8..taken].copy_from_slice(&bits.to_be_bytes());
    for block in last[..taken].as_chunks::<64>().0 {
        round(&mut h, block);
    }

    let mut out = [0u8; 20];
    for (slot, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        slot.copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// The digest of `data` as the forty lowercase characters a client sees.
#[must_use]
pub fn hex(data: &[u8]) -> [u8; 40] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 40];
    for (pair, byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(digest(data)) {
        pair[0] = DIGITS[usize::from(byte >> 4)];
        pair[1] = DIGITS[usize::from(byte & 0xf)];
    }
    out
}

/// One sixty four byte block folded into the five words.
fn round(h: &mut [u32; 5], block: &[u8; 64]) {
    let mut w = [0u32; 80];
    for (word, chunk) in w[..16].iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*chunk);
    }
    for i in 16..80 {
        w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
    }

    let [mut a, mut b, mut c, mut d, mut e] = *h;
    for (i, word) in w.iter().enumerate() {
        // The four twenty round stretches, which differ only in how three of
        // the words are mixed and in which constant is added.
        let (mix, k) = match i / 20 {
            0 => ((b & c) | (!b & d), K[0]),
            1 => (b ^ c ^ d, K[1]),
            2 => ((b & c) | (b & d) | (c & d), K[2]),
            _ => (b ^ c ^ d, K[3]),
        };
        let next = a
            .rotate_left(5)
            .wrapping_add(mix)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(*word);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = next;
    }
    for (slot, add) in h.iter_mut().zip([a, b, c, d, e]) {
        *slot = slot.wrapping_add(add);
    }
}

#[cfg(test)]
mod tests {
    use super::hex;

    /// The three the standard names, and the empty one a client is most likely
    /// to send by accident.
    #[test]
    fn the_published_vectors_come_out_right() {
        for (given, want) in [
            (b"".as_slice(), "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
            (b"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
            ),
            (b"1", "356a192b7913b04c54574d18c28d46e6395428ab"),
            (b"return 1", "e0e1f9fabfc9d4800c877a703b823ac0578ff8db"),
        ] {
            assert_eq!(hex(given), want.as_bytes(), "{given:?}");
        }
    }

    /// The lengths either side of where the padding needs a second block, since
    /// that is the only branch in the whole of it.
    #[test]
    fn the_lengths_around_a_block_boundary_all_agree() {
        // A million a's is the fourth published vector and takes far too long
        // to be a test, so these are the boundaries and a run long enough to
        // need several blocks, checked against the digests Redis answers.
        for (len, want) in [
            (55usize, "c1c8bbdc22796e28c0e15163d20899b65621d65a"),
            (56, "c2db330f6083854c99d4b5bfb6e8f29f201be699"),
            (63, "03f09f5b158a7a8cdad920bddc29b81c18a551f5"),
            (64, "0098ba824b5c16427bd7a1122a5a442a25ec644d"),
            (65, "11655326c708d70319be2610e8a57d9a5b959d3b"),
            (119, "ee971065aaa017e0632a8ca6c77bb3bf8b1dfc56"),
            (120, "f34c1488385346a55709ba056ddd08280dd4c6d6"),
        ] {
            assert_eq!(hex(&vec![b'a'; len]), want.as_bytes(), "{len}");
        }
    }
}
