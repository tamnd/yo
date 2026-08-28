//! Numbers to text and back, written by hand.
//!
//! Every reply that carries a length or an integer goes through here, which is
//! every reply, so this is as hot as anything in the codec. The formatting
//! machinery in `core::fmt` would produce the same bytes and would be several
//! times slower for the two or three digits a bulk header usually needs, so it
//! is not used for integers.
//!
//! Parsing is deliberately strict and matches Redis's `string2ll` byte for
//! byte, including its refusal of leading zeros and of a leading `+`. The
//! protocol's own lengths are parsed with it, so a stricter or looser reading
//! here is a real difference in what the two servers accept.

use core::fmt::Write as _;

/// Every two digit pair, `00` through `99`, laid out end to end.
///
/// Two digits per pass rather than one halves the number of divisions, which is
/// the whole cost of this loop. Built at compile time rather than typed out,
/// because a two hundred character literal is a typo waiting to happen and the
/// compiler will do it for free.
const PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[i * 2] = b'0' + (i / 10) as u8;
        t[i * 2 + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

/// The most digits a `u64` can have, which is what `18446744073709551615` needs.
const U64_DIGITS: usize = 20;

/// Appends the decimal digits of `n`.
pub fn push_u64(out: &mut Vec<u8>, n: u64) {
    let mut buf = [0u8; U64_DIGITS];
    let mut i = U64_DIGITS;
    let mut n = n;
    while n >= 100 {
        let p = ((n % 100) as usize) * 2;
        n /= 100;
        i -= 2;
        buf[i] = PAIRS[p];
        buf[i + 1] = PAIRS[p + 1];
    }
    if n >= 10 {
        let p = (n as usize) * 2;
        i -= 2;
        buf[i] = PAIRS[p];
        buf[i + 1] = PAIRS[p + 1];
    } else {
        i -= 1;
        buf[i] = b'0' + n as u8;
    }
    out.extend_from_slice(&buf[i..]);
}

/// Appends the decimal digits of `n`, with a minus sign if it needs one.
pub fn push_i64(out: &mut Vec<u8>, n: i64) {
    if n < 0 {
        out.push(b'-');
    }
    // `unsigned_abs` rather than `-n`, which overflows on `i64::MIN`.
    push_u64(out, n.unsigned_abs());
}

/// The number of bytes [`push_i64`] would append.
///
/// Used to presize a reply buffer before anything is written to it, which is
/// the whole point of Y18: the buffer is sized once from what is about to go
/// into it rather than grown while it is being filled.
pub const fn i64_len(n: i64) -> usize {
    let mut len = if n < 0 { 1 } else { 0 };
    let mut v = n.unsigned_abs();
    loop {
        len += 1;
        v /= 10;
        if v == 0 {
            return len;
        }
    }
}

/// Parses a signed decimal integer the way Redis's `string2ll` does.
///
/// Returns `None` for anything it would reject, which includes an empty slice,
/// a leading `+`, a leading zero on a non zero number, any non digit anywhere,
/// and anything that does not fit in an `i64`. The protocol's array and bulk
/// lengths are parsed with this, so being looser here would mean accepting
/// frames that Redis rejects, and being stricter would mean the reverse.
pub fn parse_i64(s: &[u8]) -> Option<i64> {
    // The longest thing that can parse is `-9223372036854775808`, at twenty.
    if s.is_empty() || s.len() > 20 {
        return None;
    }
    let (negative, digits) = if s[0] == b'-' {
        (true, &s[1..])
    } else {
        (false, s)
    };
    if digits.is_empty() {
        return None;
    }
    // A leading zero is only ever a whole number zero, and only a positive one.
    // `007` is not seven here and it is not seven in Redis either, and `-0` is
    // not a number in either: `string2ll` tests its zero case against the length
    // of the whole string, so the minus sign puts `-0` past it and into the one
    // to nine gate, which it fails. That matters beyond parsing, because this is
    // also what decides whether a string is stored int encoded. Accepting `-0`
    // would store it as the integer zero, and `GET` would then hand the client
    // back `0` for a value it wrote as `-0`.
    if digits[0] == b'0' {
        return if digits.len() == 1 && !negative {
            Some(0)
        } else {
            None
        };
    }
    let mut v: u64 = 0;
    for &c in digits {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(u64::from(c - b'0'))?;
    }
    if negative {
        // One more magnitude is available going negative, and `i64::MIN`
        // reached through `wrapping_neg` is the one value that cannot be
        // written as a positive `i64` first.
        if v > (i64::MAX as u64) + 1 {
            None
        } else {
            Some((v as i64).wrapping_neg())
        }
    } else if v > i64::MAX as u64 {
        None
    } else {
        Some(v as i64)
    }
}

/// The largest magnitude Redis's `double2ll` will turn into an integer.
const DOUBLE_INT_LIMIT: f64 = 4_503_599_627_370_496.0; // 2^52

/// Appends a double the way Redis 8 writes one.
///
/// Redis stopped using `%.17g` in 7.0 and now writes a double in two cases: a
/// value that is exactly an integer within two to the fifty second is written
/// as an integer, and everything else goes through a shortest round trip
/// printer. Rust's own `Display` for `f64` is the same kind of shortest round
/// trip printer, so the bytes agree for every value either of them is likely to
/// be handed. The one place they can still differ is very large or very small
/// magnitudes, where the C printer switches to an exponent and Rust does not,
/// and that is recorded rather than worked around.
///
/// The infinities and NaN are written as bare words because that is what RESP3
/// says and what RESP2 clients have always been given.
pub fn push_double(out: &mut Vec<u8>, d: f64) {
    if d.is_nan() {
        out.extend_from_slice(b"nan");
        return;
    }
    if d.is_infinite() {
        out.extend_from_slice(if d > 0.0 { b"inf" } else { b"-inf" });
        return;
    }
    if d.fract() == 0.0 && d.abs() <= DOUBLE_INT_LIMIT {
        push_i64(out, d as i64);
        return;
    }
    // Writing through the sink puts the digits straight into the reply buffer.
    // `format!` would produce the same bytes and one throwaway allocation, and
    // a shard thread that allocates aborts.
    let mut sink = Utf8Sink(out);
    let _ = write!(sink, "{d}");
}

/// A `core::fmt::Write` that appends UTF-8 to a byte buffer.
///
/// The float printer only speaks `fmt::Write` and the reply buffer is bytes.
/// This is the whole adapter, and it exists so that no reply path anywhere ever
/// builds a `String` it immediately throws away.
struct Utf8Sink<'a>(&'a mut Vec<u8>);

impl core::fmt::Write for Utf8Sink<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(n: i64) -> String {
        let mut v = Vec::new();
        push_i64(&mut v, n);
        String::from_utf8(v).unwrap()
    }

    #[test]
    fn integers_round_trip_through_text() {
        for n in [
            0,
            1,
            9,
            10,
            99,
            100,
            -1,
            -9,
            -10,
            12345,
            -12345,
            i64::MAX,
            i64::MIN,
        ] {
            assert_eq!(text(n), n.to_string(), "writing {n}");
            assert_eq!(parse_i64(text(n).as_bytes()), Some(n), "reading {n}");
        }
    }

    #[test]
    fn the_length_is_known_before_the_digits_are_written() {
        for n in [0, 5, 42, -42, 999, 1000, i64::MAX, i64::MIN] {
            assert_eq!(i64_len(n), text(n).len(), "length of {n}");
        }
    }

    /// Every boundary of the two digit loop, since an off by one there is a
    /// wrong length header rather than a crash and would be found by a client.
    #[test]
    fn every_length_of_number_is_written_correctly() {
        let mut n: u64 = 0;
        for _ in 0..20 {
            for probe in [n, n + 1, n.saturating_sub(1)] {
                let mut v = Vec::new();
                push_u64(&mut v, probe);
                assert_eq!(v, probe.to_string().as_bytes(), "writing {probe}");
            }
            n = n.saturating_mul(10).max(9);
            if n == u64::MAX {
                break;
            }
        }
    }

    #[test]
    fn the_parser_refuses_what_redis_refuses() {
        for bad in [
            &b""[..],
            b"-",
            b"+1",
            b"01",
            b"-01",
            b" 1",
            b"1 ",
            b"1a",
            b"a",
            b"1.0",
            b"-0",
            b"-00",
            b"9223372036854775808",
            b"-9223372036854775809",
            b"99999999999999999999999",
        ] {
            assert_eq!(parse_i64(bad), None, "{:?} should not parse", bad);
        }
        // The one leading zero that is a number, and the one negative that only
        // exists going downwards.
        assert_eq!(parse_i64(b"0"), Some(0));
        assert_eq!(parse_i64(b"-9223372036854775808"), Some(i64::MIN));
    }

    #[test]
    fn doubles_are_written_the_way_redis_writes_them() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (-0.0, "0"),
            (3.0, "3"),
            (-3.0, "-3"),
            (3.5, "3.5"),
            (0.1, "0.1"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ];
        for &(d, want) in cases {
            let mut v = Vec::new();
            push_double(&mut v, d);
            assert_eq!(String::from_utf8(v).unwrap(), want, "writing {d}");
        }
    }
}
