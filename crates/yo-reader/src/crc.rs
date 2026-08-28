//! CRC32C, written out from the polynomial rather than taken from anywhere.
//!
//! The engine computes this with the SSE 4.2 and NEON instructions, and falls
//! back to a slice by eight table when it has neither. Both of those are fast
//! and neither of them is obvious. This one is the definition: shift a bit out,
//! exclusive or the polynomial back in if it was set, eight times per byte.
//!
//! The table is built at first use from that same loop rather than pasted in as
//! a literal, which matters more than it looks. A pasted table is a copy of
//! whatever produced it, so a wrong table and a wrong reference implementation
//! agree with each other perfectly. A table derived here from the polynomial
//! can only be wrong if the polynomial is wrong, and the polynomial is one
//! constant that is easy to check against the standard.
//!
//! Castagnoli, reflected form, which is what iSCSI and ext4 and everything else
//! that says CRC32C means.

use std::sync::OnceLock;

/// The Castagnoli polynomial, reflected. `0x1edc6f41` read backwards.
const POLY: u32 = 0x82f6_3b78;

/// One byte's worth of the shift loop, for every possible byte.
fn table() -> &'static [u32; 256] {
    static T: OnceLock<[u32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                // The reflected form shifts right and tests the low bit, which
                // is the mirror of the textbook version. Getting this backwards
                // produces a checksum that is self consistent and wrong, so it
                // is worth saying out loud which one this is.
                c = if c & 1 == 1 { (c >> 1) ^ POLY } else { c >> 1 };
            }
            *slot = c;
        }
        t
    })
}

/// Continues a CRC32C over `bytes`, starting from `seed`.
///
/// Pass 0 for a fresh checksum. The seed is there so a checksum can be taken
/// over two pieces that are not next to each other in memory, which the record
/// trailer needs.
#[must_use]
pub fn crc32c(seed: u32, bytes: &[u8]) -> u32 {
    let t = table();
    let mut c = !seed;
    for &b in bytes {
        c = t[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// A CRC32C over `bytes` with the four bytes at `skip` treated as zero.
///
/// Every checksum in the format lives inside the run it covers, so this is how
/// all of them are computed: the field reads as zero while its own value is
/// being worked out. Anything shorter than the skip window is checksummed
/// whole, which only happens on a buffer that is already going to be rejected
/// for its length.
#[must_use]
pub fn crc32c_skipping(bytes: &[u8], skip: usize) -> u32 {
    if skip + 4 > bytes.len() {
        return crc32c(0, bytes);
    }
    let c = crc32c(0, &bytes[..skip]);
    let c = crc32c(c, &[0, 0, 0, 0]);
    crc32c(c, &bytes[skip + 4..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_check_value() {
        // Every CRC specification carries a check value: the checksum of the
        // nine ASCII digits. For CRC32C it is 0xe3069283. If this passes, the
        // polynomial and the reflection are both right, and those are the only
        // two things in here that can be wrong in a way the rest of the crate
        // would not notice.
        assert_eq!(crc32c(0, b"123456789"), 0xe306_9283);
    }

    #[test]
    fn an_empty_run_is_the_seed() {
        assert_eq!(crc32c(0, b""), 0);
        assert_eq!(crc32c(0x1234_5678, b""), 0x1234_5678);
    }

    #[test]
    fn seeding_is_the_same_as_one_pass() {
        let whole = crc32c(0, b"the quick brown fox");
        let a = crc32c(0, b"the quick ");
        let split = crc32c(a, b"brown fox");
        assert_eq!(whole, split);
    }

    #[test]
    fn skipping_ignores_what_is_in_the_window() {
        let mut a = *b"0123456789abcdef";
        let mut b = a;
        b[4..8].copy_from_slice(&[0xff; 4]);
        assert_eq!(crc32c_skipping(&a, 4), crc32c_skipping(&b, 4));
        // And it does not ignore anything else.
        a[9] = b'!';
        assert_ne!(crc32c_skipping(&a, 4), crc32c_skipping(&b, 4));
    }

    #[test]
    fn a_flipped_bit_changes_it() {
        let base = [7u8; 64];
        for bit in 0..64 * 8 {
            let mut v = base;
            v[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(crc32c(0, &v), crc32c(0, &base), "bit {bit}");
        }
    }
}
