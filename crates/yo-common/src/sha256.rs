//! SHA-256, because an ACL password is kept as one and never as itself.
//!
//! A real server hashes a password the moment it is given one and keeps the
//! sixty four characters, so `ACL GETUSER` and `ACL LIST` and the rewritten
//! config file all name a user's passwords without any of them holding a
//! password. That is the whole reason this is here, and it is the reason the
//! digest has to be this one rather than a better one: `ACL SETUSER u #<hash>`
//! takes a hash a client computed somewhere else, and a config file written by
//! a real server has to load here and mean the same thing.
//!
//! Seventy lines rather than a dependency, for the reason [`crate::sha1`] gives:
//! this is the whole of a fixed algorithm, so there is nothing here that will
//! need updating and nothing a crate would do differently.
//!
//! This is not a password hashing function and a real server does not pretend
//! otherwise. A weak password behind one round of SHA-256 is a weak password,
//! which is what `ACL GENPASS` is for.

/// The sixty four round constants, the fractional parts of the cube roots of
/// the first sixty four primes.
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// The eight words the standard starts from.
const H: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// The thirty two byte digest of `data`.
#[must_use]
pub fn digest(data: &[u8]) -> [u8; 32] {
    let mut h = H;
    // The message, its terminator, the padding and the length, walked as sixty
    // four byte blocks without ever building the padded copy. The last one or
    // two blocks are the only ones that are not entirely message, so they are
    // the only ones written into the scratch buffer.
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

    let mut out = [0u8; 32];
    for (slot, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        slot.copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// The sixty four lowercase characters, which is the form every ACL command
/// writes and the form `ACL SETUSER u #<hash>` reads back.
#[must_use]
pub fn hex(data: &[u8]) -> [u8; 64] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 64];
    for (pair, byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(digest(data)) {
        pair[0] = DIGITS[usize::from(byte >> 4)];
        pair[1] = DIGITS[usize::from(byte & 0xf)];
    }
    out
}

/// One sixty four byte block through the compression function.
fn round(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (slot, bytes) in w[..16].iter_mut().zip(block.as_chunks::<4>().0) {
        *slot = u32::from_be_bytes(*bytes);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut i] = *h;
    for step in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let one = i
            .wrapping_add(s1)
            .wrapping_add(choose)
            .wrapping_add(K[step])
            .wrapping_add(w[step]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let most = (a & b) ^ (a & c) ^ (b & c);
        let two = s0.wrapping_add(most);
        i = g;
        g = f;
        f = e;
        e = d.wrapping_add(one);
        d = c;
        c = b;
        b = a;
        a = one.wrapping_add(two);
    }
    for (into, word) in h.iter_mut().zip([a, b, c, d, e, f, g, i]) {
        *into = into.wrapping_add(word);
    }
}

#[cfg(test)]
mod tests {
    use super::hex;

    /// The three vectors everybody checks a SHA-256 against, plus the two
    /// lengths that decide whether the padding needs a second block.
    #[test]
    fn the_published_vectors_come_out_right() {
        assert_eq!(
            &hex(b""),
            b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            &hex(b"abc"),
            b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            &hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            b"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// A message that fills the block it lands in, which is where a padding
    /// that forgets to start a second block goes wrong and nowhere else.
    #[test]
    fn a_message_that_leaves_no_room_for_its_length_takes_another_block() {
        // Fifty five bytes fit with their terminator and length, fifty six do
        // not, and the two either side of that line are the whole of the case.
        let long = vec![b'a'; 55];
        assert_eq!(
            &hex(&long),
            b"9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
        let longer = vec![b'a'; 56];
        assert_eq!(
            &hex(&longer),
            b"b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
        );
    }

    /// What the reference answers for the password used all over its own test
    /// suite, which is the number `ACL LIST` prints beside a `#`.
    #[test]
    fn the_hash_is_the_one_a_real_server_writes_down() {
        assert_eq!(
            &hex(b"pw"),
            b"30c952fab122c3f9759f02a6d95c3758b246b4fee239957b2d4fee46e26170c4"
        );
    }
}
