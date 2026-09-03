//! The bytes a collection body turns into on its way out of memory.
//!
//! A string demotes by handing the bytes it already is to the tier. A set, a
//! hash, a list or a sorted set cannot do that, because its body is a table or a
//! band or a listpack sitting in a [`Slab`](crate::slab), and the pointers in it
//! mean nothing to anyone reading the file back. So it is written out as a form
//! byte and whatever that form needs, and read back into the same representation
//! it left in. This module is the plumbing both directions share.
//!
//! # Why not the RDB serialiser
//!
//! [`crate::rdb`] already turns a value into bytes and back, and it is the wrong
//! tool twice over. It writes Redis's format, so every blob carries a version
//! envelope and a crc64 that nothing here needs and that costs a pass over the
//! payload in each direction. Worse, it writes the simple shape: a set that is a
//! partitioned band comes back a table, and an intset past its ceiling comes back
//! something that answers a different word to `OBJECT ENCODING`. A value that
//! changes encoding because it was quiet long enough to be demoted is a value
//! whose behaviour depends on memory pressure, and that is not something a client
//! can be asked to reason about. The forms here carry enough to land back in the
//! exact representation that left, including the flags that only affect what the
//! encoding is called.
//!
//! # The encoding
//!
//! Unsigned numbers are LEB128, seven bits a byte, low group first. Signed
//! numbers are zigzag over the same thing, so a small negative is one byte rather
//! than ten. Byte strings are a length and then the bytes.
//!
//! There is no alignment and no padding anywhere, because the reader is a cursor
//! over a slice and never a cast. A frozen body is read exactly once, straight
//! into the structure it rebuilds, so the only thing the layout is tuned for is
//! being short.

/// What is wrong with a frozen body that will not parse.
///
/// Every arm means the same thing operationally, which is that the store handed
/// back bytes that are not what was written, and the value is gone. They are
/// separate so the error a client sees can say which check failed, because the
/// three causes are different bugs: a truncated read, a form written by a newer
/// version, and a payload whose own structure is broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Broken {
    /// It ended in the middle of something.
    Short,
    /// The form byte is not one this version writes.
    Form,
    /// The form was understood and the payload inside it was not.
    Body,
}

impl std::fmt::Display for Broken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Broken::Short => "the frozen body ends early",
            Broken::Form => "the frozen body has a form this version cannot read",
            Broken::Body => "the frozen body has a payload that does not parse",
        })
    }
}

impl std::error::Error for Broken {}

/// Append `n` as LEB128.
///
/// Most numbers that come through here are a member count or a byte length and
/// most of those are under 128, which is the case the loop is shaped for: one
/// comparison and one push.
#[inline]
pub fn put_uint(out: &mut Vec<u8>, mut n: u64) {
    while n >= 0x80 {
        out.push((n as u8) | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

/// Append `v` zigzagged, so that a small magnitude is a short encoding either
/// side of zero.
#[inline]
pub fn put_int(out: &mut Vec<u8>, v: i64) {
    put_uint(out, ((v << 1) ^ (v >> 63)) as u64);
}

/// Append `v` as its eight raw bytes, little endian.
///
/// Not LEB128 and not zigzag, because a double's bit pattern carries its
/// exponent in the high bits, so every ordinary score would take the full ten
/// groups plus the sign work. Eight flat bytes are shorter and cost nothing to
/// read back.
#[inline]
pub fn put_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Append `bytes` behind its length.
#[inline]
pub fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_uint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// A cursor over a frozen body.
///
/// Every read either moves the cursor forward and answers, or answers
/// [`Broken::Short`] and leaves the cursor wherever it was. There is no way to
/// go backwards, because nothing that reads one of these needs to: a form is
/// written in the order it is read.
pub struct Cut<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cut<'a> {
    /// A cursor at the start of `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Cut<'a> {
        Cut { bytes, at: 0 }
    }

    /// The next byte.
    #[inline]
    pub fn byte(&mut self) -> Result<u8, Broken> {
        let b = *self.bytes.get(self.at).ok_or(Broken::Short)?;
        self.at += 1;
        Ok(b)
    }

    /// The next LEB128 number.
    ///
    /// Ten groups at most, which is what a u64 takes, so a payload of nothing but
    /// continuation bits cannot spin here.
    #[inline]
    pub fn uint(&mut self) -> Result<u64, Broken> {
        let mut n = 0u64;
        let mut shift = 0;
        loop {
            let b = self.byte()?;
            n |= u64::from(b & 0x7f) << shift;
            if b < 0x80 {
                return Ok(n);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Broken::Body);
            }
        }
    }

    /// The next zigzagged number.
    #[inline]
    pub fn int(&mut self) -> Result<i64, Broken> {
        let n = self.uint()?;
        Ok(((n >> 1) as i64) ^ -((n & 1) as i64))
    }

    /// The next double, from its eight raw bytes.
    #[inline]
    pub fn f64(&mut self) -> Result<f64, Broken> {
        let s = self.take(8)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(f64::from_le_bytes(b))
    }

    /// The next `n` bytes.
    #[inline]
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], Broken> {
        let end = self.at.checked_add(n).ok_or(Broken::Short)?;
        let s = self.bytes.get(self.at..end).ok_or(Broken::Short)?;
        self.at = end;
        Ok(s)
    }

    /// The next length prefixed byte string.
    #[inline]
    pub fn bytes(&mut self) -> Result<&'a [u8], Broken> {
        let n = self.uint()?;
        // Through `usize` after the read rather than before, so that a length
        // larger than this machine's address space is a short body and not a
        // truncating cast that reads the wrong amount.
        let n = usize::try_from(n).map_err(|_| Broken::Short)?;
        self.take(n)
    }

    /// Everything from here to the end, which is what a form that ends in one
    /// blob wants.
    ///
    /// A blob at the end of a form needs no length, because the frozen body is
    /// its own length. That is the whole reason the two forms that hand back a
    /// structure Redis already knows how to lay out cost one byte of overhead.
    #[must_use]
    #[inline]
    pub const fn rest(&self) -> &'a [u8] {
        // Not a slice expression, because this is `const` and indexing is not.
        self.bytes.split_at(self.at).1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_number_comes_back_as_itself() {
        for n in [0u64, 1, 127, 128, 300, 16383, 16384, u64::MAX] {
            let mut out = Vec::new();
            put_uint(&mut out, n);
            assert_eq!(Cut::new(&out).uint(), Ok(n), "{n}");
        }
    }

    #[test]
    fn a_signed_number_comes_back_as_itself_and_a_small_one_is_short() {
        for v in [0i64, 1, -1, 63, -64, 1000, -1000, i64::MIN, i64::MAX] {
            let mut out = Vec::new();
            put_int(&mut out, v);
            assert_eq!(Cut::new(&out).int(), Ok(v), "{v}");
        }
        let mut out = Vec::new();
        put_int(&mut out, -5);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn a_body_that_ends_early_is_short_and_not_a_panic() {
        let mut out = Vec::new();
        put_bytes(&mut out, b"hello");
        out.truncate(3);
        assert_eq!(Cut::new(&out).bytes(), Err(Broken::Short));
        assert_eq!(Cut::new(&[]).byte(), Err(Broken::Short));
        assert_eq!(Cut::new(&[0x80]).uint(), Err(Broken::Short));
    }

    #[test]
    fn nothing_but_continuation_bits_stops() {
        assert_eq!(Cut::new(&[0x80; 32]).uint(), Err(Broken::Body));
    }

    #[test]
    fn the_rest_is_what_is_left() {
        let mut out = vec![7u8];
        out.extend_from_slice(b"the blob");
        let mut cut = Cut::new(&out);
        assert_eq!(cut.byte(), Ok(7));
        assert_eq!(cut.rest(), b"the blob");
    }
}
