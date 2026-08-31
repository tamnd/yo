//! Comparing two byte strings without leaving the function.
//!
//! `a == b` on two slices ends up in the platform's `memcmp`. That is the right
//! answer for a megabyte and the wrong one for a key: a profile of `SADD` on
//! the wire had seven percent of the command inside `_platform_memcmp` and the
//! stub that reaches it, comparing nineteen bytes. The call itself, the length
//! dispatch inside it and the return are most of that, and none of it is the
//! comparison.
//!
//! So this compares in machine words and stays inline. A key or a member is
//! almost always shorter than a cache line, and the shapes that matter are the
//! ones a benchmark and a real workload agree on: `key:000000000001` at
//! fourteen bytes, `member:000000000001` at nineteen, a session id at thirty
//! two.
//!
//! # The last word overlaps
//!
//! A comparison of nineteen bytes reads bytes 0..8, 8..16 and then 11..19,
//! which covers the whole string with three loads and no tail loop. The middle
//! five bytes are read twice, which costs nothing and is what keeps the shape
//! branch free. Below eight bytes the same trick runs on four byte words, and
//! below four it is a loop of at most three bytes, which is shorter than any
//! cleverness would be.
//!
//! # This is equality and not ordering
//!
//! There is no `cmp` here on purpose. Nothing on the hot path needs to know
//! which of two keys sorts first, and a word wise ordering would have to
//! byte swap on a little endian machine to get the answer right, which is the
//! sort of subtlety that is worth avoiding when nothing is asking for it.

/// The eight bytes at `at`, as a machine word.
#[inline(always)]
fn w8(s: &[u8], at: usize) -> u64 {
    u64::from_ne_bytes([
        s[at],
        s[at + 1],
        s[at + 2],
        s[at + 3],
        s[at + 4],
        s[at + 5],
        s[at + 6],
        s[at + 7],
    ])
}

/// The four bytes at `at`, as a half word.
#[inline(always)]
fn w4(s: &[u8], at: usize) -> u32 {
    u32::from_ne_bytes([s[at], s[at + 1], s[at + 2], s[at + 3]])
}

/// Whether `a` and `b` hold the same bytes.
///
/// The same answer as `a == b` and the same cost model as `memcmp` for a long
/// string, without the call for a short one.
#[inline]
#[must_use]
pub fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len();
    if n != b.len() {
        return false;
    }
    if n >= 8 {
        let mut at = 0;
        while at + 8 < n {
            if w8(a, at) != w8(b, at) {
                return false;
            }
            at += 8;
        }
        // The last whole word, which reaches back over bytes already compared
        // when the length is not a multiple of eight.
        return w8(a, n - 8) == w8(b, n - 8);
    }
    if n >= 4 {
        return w4(a, 0) == w4(b, 0) && w4(a, n - 4) == w4(b, n - 4);
    }
    let mut at = 0;
    while at < n {
        if a[at] != b[at] {
            return false;
        }
        at += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::bytes_eq;

    /// Every length from nothing to past two words, equal and then not equal at
    /// every position in turn.
    ///
    /// The overlapping last word is the part worth being thorough about: a
    /// difference in the bytes that get read twice has to be found, and so does
    /// one in the bytes that are only read by the overlapping load.
    #[test]
    fn it_agrees_with_the_slice_comparison_at_every_length_and_position() {
        for n in 0..40usize {
            let a: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            assert!(bytes_eq(&a, &a.clone()), "{n} bytes against itself");
            for at in 0..n {
                let mut b = a.clone();
                b[at] ^= 0x80;
                assert!(!bytes_eq(&a, &b), "{n} bytes differing at {at}");
                assert_eq!(bytes_eq(&a, &b), a == b);
            }
        }
    }

    #[test]
    fn a_different_length_is_never_equal() {
        for n in 0..40usize {
            let a = vec![b'x'; n];
            let b = vec![b'x'; n + 1];
            assert!(!bytes_eq(&a, &b));
            assert!(!bytes_eq(&b, &a));
        }
    }

    #[test]
    fn nothing_is_the_same_as_nothing() {
        assert!(bytes_eq(b"", b""));
        assert!(bytes_eq(&[], &[]));
    }
}
