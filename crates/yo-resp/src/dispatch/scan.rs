//! The two pieces every scan shares: reading a cursor and writing the reply.
//!
//! There are four scans on the wire, `SCAN`, `SSCAN`, `HSCAN` and `ZSCAN`, and
//! they differ in what they walk and in the one or two options they take. What
//! they do not differ in is the cursor they are given and the shape of what
//! they answer, and both of those were written out four times before this
//! module existed, which is four chances to disagree with Redis in three of
//! them. Two of the four did.
//!
//! # A cursor is not an integer
//!
//! Redis parses one with `strtoull` and then insists the whole argument was
//! consumed and that there was no minus, so `+0`, `  0` and `007` are all a
//! cursor of zero and `-1`, `0 `, `0abc` and a number past sixty four bits are
//! all `invalid cursor`. That is a sentence of its own and not the usual one
//! about integers, which matters because a client that has lost its place looks
//! for exactly this error to know it has to start again.
//!
//! Ours is unsigned for a second reason on top of Redis's: a yo cursor packs a
//! partition count into its top bits, so a large enough collection makes it
//! wider than an `i64` and it cannot go out through the integer path either.
//!
//! # The header cannot be written first
//!
//! A scan answers two things, the cursor to come back with and the elements it
//! found, and neither is known until the walk is over: `MATCH` decides how many
//! elements there are as it goes, and the cursor is what the walk produced. So
//! the elements are written into the buffer first, their header is written
//! behind them and moved in front, and then the cursor is written behind both
//! and moved in front of them. Nothing is copied twice and nothing is collected
//! into a `Vec` that would be an allocation per call.

use yo_common::{Code, Error, Result};
use yo_kv::{Cursor, KeyCursor};

use crate::reply::Out;

/// Where a walk got to, whichever kind of walk it was.
///
/// There are two cursor types and they are named apart on purpose, so that a
/// caller holding both cannot pass one where the other was meant. This trait is
/// what lets one parse and one reply serve both without giving that up.
pub(super) trait Resume: Copy {
    /// The cursor a client sent back.
    fn from_raw(raw: u64) -> Self;
    /// The number to send it.
    fn raw(self) -> u64;
}

impl Resume for Cursor {
    fn from_raw(raw: u64) -> Cursor {
        Cursor::from_raw(raw)
    }
    fn raw(self) -> u64 {
        Cursor::raw(self)
    }
}

impl Resume for KeyCursor {
    fn from_raw(raw: u64) -> KeyCursor {
        KeyCursor::from_raw(raw)
    }
    fn raw(self) -> u64 {
        KeyCursor::raw(self)
    }
}

/// What Redis says about a cursor it cannot read.
pub(super) const BAD_CURSOR: &str = "invalid cursor";

/// How many elements a scan walks when the client does not say.
///
/// Redis's default and a hint rather than a promise, the same as it is there: a
/// listpack collection comes back whole whatever this says.
pub(super) const COUNT: usize = 10;

/// A cursor as the client sent it back.
///
/// # Errors
///
/// [`BAD_CURSOR`] for anything that is not a whole unsigned number, which
/// includes a negative one, trailing anything, and overflow.
pub(super) fn parse_cursor<C: Resume>(arg: &[u8]) -> Result<C> {
    let bad = || Error::new(Code::Invalid, BAD_CURSOR);
    // Leading blanks then an optional plus, which is what strtoull skips and
    // what Redis therefore accepts without ever having decided to.
    let digits = arg.trim_ascii_start();
    let digits = digits.strip_prefix(b"+").unwrap_or(digits);
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return Err(bad());
    }
    let mut raw: u64 = 0;
    for b in digits {
        raw = raw
            .checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(b - b'0')))
            .ok_or_else(bad)?;
    }
    Ok(C::from_raw(raw))
}

/// Write the two element reply around a walk.
///
/// The walk writes its elements into `out` and says how many it wrote and where
/// the next cursor is. Everything else is this function.
///
/// # Errors
///
/// Whatever the walk fails with, which in practice is `WRONGTYPE`.
pub(super) fn reply<C: Resume>(
    out: &mut Out,
    walk: impl FnOnce(&mut Out) -> Result<(C, usize)>,
) -> Result<()> {
    out.array(2);
    let at = out.len();
    let (next, n) = walk(out)?;
    out.close_array(at, n);
    let body = out.len() - at;
    out.bulk_u64(next.raw());
    let cursor = out.len() - at - body;
    out.hoist(at, cursor);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every one of these was read off a running Redis 8.10.1, because the
    /// accepted shapes are not a decision anybody made, they are what strtoull
    /// happens to do.
    #[test]
    fn a_cursor_is_read_the_way_strtoull_reads_one() {
        for ok in [
            &b"0"[..],
            b"+0",
            b" 0",
            b"  +0",
            b"007",
            b"18446744073709551615",
        ] {
            assert!(
                parse_cursor::<Cursor>(ok).is_ok(),
                "{:?}",
                core::str::from_utf8(ok)
            );
        }
        for bad in [
            &b""[..],
            b"-1",
            b"0 ",
            b"0abc",
            b"abc",
            b"0x10",
            b"+ 0",
            b"18446744073709551616",
        ] {
            let got = parse_cursor::<Cursor>(bad).expect_err("this is not a cursor");
            assert_eq!(got.message(), BAD_CURSOR, "{:?}", core::str::from_utf8(bad));
        }
    }

    #[test]
    fn the_cursor_ends_up_in_front_of_the_elements() {
        let mut out = Out::new(crate::proto::Proto::Resp2);
        reply(&mut out, |out| {
            out.bulk(b"a");
            out.bulk(b"b");
            Ok((Cursor::from_raw(7), 2))
        })
        .expect("a walk that cannot fail");
        assert_eq!(
            core::str::from_utf8(out.as_slice()).expect("ascii"),
            "*2\r\n$1\r\n7\r\n*2\r\n$1\r\na\r\n$1\r\nb\r\n"
        );
    }
}
