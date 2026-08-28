//! The conditional write comparisons Redis 8.4 added.
//!
//! `SET` grew `IFEQ`, `IFNE`, `IFDEQ` and `IFDNE`, and `DELEX` is the same four
//! conditions in front of a delete. They exist for the read modify write that
//! nobody was doing correctly: a client reads a value, decides something about
//! it, and writes back, and another client can get to the key in between. The
//! alternative was `WATCH` and `MULTI`, which costs a round trip.
//!
//! The digest forms are the same conditions with the value replaced by its
//! [`DIGEST`](yo_common::xxh3), so a client comparing against a megabyte value
//! sends sixteen bytes instead of a megabyte.
//!
//! Whether a missing key satisfies a condition is not symmetric and it is not
//! arbitrary. A key that is not there is not equal to anything, so `IFEQ` fails
//! on it, and it is not equal to anything, so `IFNE` passes on it. That is what
//! a real 8.8 does: `SET k v IFNE other` on a key that does not exist stores.

use crate::value::Str;

/// What a conditional write compares the current value against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare<'a> {
    /// `IFEQ`: the value is exactly these bytes.
    Equal(&'a [u8]),
    /// `IFNE`: the value is not exactly these bytes, or there is no value.
    NotEqual(&'a [u8]),
    /// `IFDEQ`: the value's digest is this number.
    DigestEqual(u64),
    /// `IFDNE`: the value's digest is not this number, or there is no value.
    DigestNotEqual(u64),
}

impl Compare<'_> {
    /// Whether the condition holds against what is stored, where `None` is a
    /// key that is not there.
    #[must_use]
    pub fn holds(&self, current: Option<Str<'_>>) -> bool {
        match *self {
            Compare::Equal(want) => current.is_some_and(|v| v.eq_bytes(want)),
            Compare::NotEqual(want) => !current.is_some_and(|v| v.eq_bytes(want)),
            Compare::DigestEqual(want) => current.is_some_and(|v| v.digest() == want),
            Compare::DigestNotEqual(want) => !current.is_some_and(|v| v.digest() == want),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest of `hello`, which is the number a real Redis replies to
    /// `DIGEST` with and the number a client would send back as `IFDEQ`.
    const HELLO: u64 = 0x9555_e855_5c62_dcfd;

    #[test]
    fn equality_is_against_the_string_the_client_would_have_read() {
        let int = Some(Str::Int(42));
        assert!(Compare::Equal(b"42").holds(int));
        // Not `042`, which parses to the same number and is a different string.
        assert!(!Compare::Equal(b"042").holds(int));
        assert!(Compare::NotEqual(b"042").holds(int));
    }

    #[test]
    fn a_missing_key_fails_the_equal_forms_and_passes_the_not_equal_forms() {
        assert!(!Compare::Equal(b"").holds(None));
        assert!(!Compare::DigestEqual(HELLO).holds(None));
        assert!(Compare::NotEqual(b"anything").holds(None));
        assert!(Compare::DigestNotEqual(HELLO).holds(None));
    }

    #[test]
    fn the_digest_forms_agree_with_the_value_forms() {
        let v = Some(Str::Bytes(b"hello"));
        assert!(Compare::Equal(b"hello").holds(v));
        assert!(Compare::DigestEqual(HELLO).holds(v));
        assert!(!Compare::NotEqual(b"hello").holds(v));
        assert!(!Compare::DigestNotEqual(HELLO).holds(v));
    }

    #[test]
    fn an_int_encoded_value_digests_its_digits() {
        // A real 8.8 answers this for `SET n 42` followed by `DIGEST n`, so the
        // digits and not the eight bytes the record actually holds.
        assert_eq!(Str::Int(42).digest(), 0x1217_cb28_c0ef_2191);
        assert!(Compare::DigestEqual(0x1217_cb28_c0ef_2191).holds(Some(Str::Int(42))));
    }
}
