//! Bit level kernels: counting, searching, combining and packed fields.
//!
//! Redis calls a string used this way a bitmap, and it is not a separate type:
//! `SETBIT`, `BITCOUNT`, `BITOP` and `BITFIELD` all work on the ordinary string
//! a `SET` would have left behind, which is why this file is kernels over byte
//! slices and nothing else. The keyspace side, which is where a key turns into
//! bytes and where growing a value is decided, is in
//! [`bitmaps`](crate::bitmaps).
//!
//! # Which end a bit is
//!
//! Bit zero is the top bit of byte zero. That is the convention every one of
//! these commands uses and it is the opposite of the one a language's shift
//! operators suggest, so it is worth being blunt about it: bit `i` is
//!
//! ```text
//! bytes[i / 8] & (0x80 >> (i % 8))
//! ```
//!
//! It falls out of wanting `BITPOS` over a bitmap of user ids to answer in the
//! order the ids were assigned, and it is why a `u64` loaded out of the middle
//! of a bitmap has to be read big endian for [`u64::leading_zeros`] to mean the
//! distance to the next set bit.
//!
//! # Counting
//!
//! [`count`] is four `u64` accumulators fed by `count_ones`. That is the same
//! shape as Redis's `redisPopcount`, which unrolls by four for the same reason:
//! `popcnt` has a three cycle latency and one per cycle throughput on every x86
//! since Nehalem, so a loop with one accumulator is latency bound at a third of
//! the rate and four independent chains fill the pipe. On aarch64 there is no
//! scalar popcount at all and LLVM turns the same loop into `cnt` over a vector
//! register plus a widening add tree, which is why this is written as an
//! ordinary loop rather than as intrinsics: the ordinary loop is what both
//! backends already do well, and an intrinsic version would be two more code
//! paths to keep right for no measured gain.
//!
//! # Combining
//!
//! [`combine`] does `BITOP`, including the four operations Redis 8.2 added:
//! `DIFF`, `DIFF1`, `ANDOR` and `ONE`. It works a block at a time over a fixed
//! stack buffer rather than allocating one accumulator per source, so a `BITOP`
//! over eight sources touches the same two kibibytes of stack whatever the
//! bitmaps weigh, and each block of each source is read once while it is warm.
//! The alternative, folding whole bitmaps one source at a time, walks the
//! destination once per source and that is where a `BITOP` over big bitmaps
//! spends its time.

/// How many bytes of each source a block pass works on at once.
///
/// Two of these live on the stack in the worst case, which is `ONE` and its
/// "seen more than once" mask, so the whole of `BITOP` is two kibibytes of
/// stack. Big enough that the per block overhead disappears against the byte
/// loops, small enough to sit in L1 next to a block of every source.
const BLOCK: usize = 1024;

/// How many bits are set.
#[must_use]
pub fn count(bytes: &[u8]) -> u64 {
    let (words, tail_bytes) = bytes.as_chunks::<32>();
    let (mut a, mut b, mut c, mut d) = (0u32, 0u32, 0u32, 0u32);
    for w in words {
        a += u64::from_le_bytes(w[0..8].try_into().expect("eight bytes")).count_ones();
        b += u64::from_le_bytes(w[8..16].try_into().expect("eight bytes")).count_ones();
        c += u64::from_le_bytes(w[16..24].try_into().expect("eight bytes")).count_ones();
        d += u64::from_le_bytes(w[24..32].try_into().expect("eight bytes")).count_ones();
    }
    let tail: u32 = tail_bytes.iter().map(|&x| x.count_ones()).sum();
    u64::from(a) + u64::from(b) + u64::from(c) + u64::from(d) + u64::from(tail)
}

/// How many bits are set in the half open bit range `from..to`.
///
/// Both ends are bit indexes and the caller has already clamped them to the
/// bitmap, which is where the negative index and the `BYTE` or `BIT` word are
/// dealt with. An empty or backwards range is zero.
#[must_use]
pub fn count_range(bytes: &[u8], from: u64, to: u64) -> u64 {
    let Some((head, whole, tail)) = split(bytes, from, to) else {
        return 0;
    };
    u64::from(head.count_ones()) + count(whole) + u64::from(tail.count_ones())
}

/// The first bit equal to `set` in the half open bit range `from..to`.
///
/// `None` when the range holds no such bit, which the command layer turns into
/// minus one or into the bit past the end depending on which of the two
/// questions was asked.
#[must_use]
pub fn find(bytes: &[u8], set: bool, from: u64, to: u64) -> Option<u64> {
    if from >= to || from >= (bytes.len() as u64) * 8 {
        return None;
    }
    let end = to.min((bytes.len() as u64) * 8);
    // A byte at a time over the ragged ends and a word at a time in the middle
    // would be three loops to get right. This is one loop over bytes with the
    // two ends masked, and the word scan below it only has to find the first
    // byte that is not uniform, which is where the time goes on a long bitmap.
    let (first, last) = ((from / 8) as usize, ((end - 1) / 8) as usize);
    let mut at = first;
    while at <= last {
        let mut byte = bytes[at];
        if !set {
            byte = !byte;
        }
        // Off the ends of the range, pretend the bits are not what we want.
        if at == first {
            byte &= 0xffu8 >> (from % 8);
        }
        if at == last && !end.is_multiple_of(8) {
            byte &= !(0xffu8 >> (end % 8));
        }
        if byte != 0 {
            return Some(at as u64 * 8 + u64::from(byte.leading_zeros()));
        }
        // Nothing in this byte, so skip whole words of nothing. The scan reads
        // eight bytes at a time and only past the first byte, so the masks
        // above never apply to what it skips.
        at += 1;
        let uniform = if set { 0 } else { u64::MAX };
        while at + 8 <= last {
            let w = u64::from_ne_bytes(bytes[at..at + 8].try_into().expect("eight bytes"));
            if w != uniform {
                break;
            }
            at += 8;
        }
    }
    None
}

/// The masked first byte, the whole bytes and the masked last byte of a range.
///
/// `None` for a range that holds nothing. The two ends come back as values
/// rather than as slices because they are masked copies and not what is in the
/// bitmap, and the middle comes back as a slice so that [`count`] can have it.
fn split(bytes: &[u8], from: u64, to: u64) -> Option<(u8, &[u8], u8)> {
    let bits = (bytes.len() as u64) * 8;
    let (from, to) = (from.min(bits), to.min(bits));
    if from >= to {
        return None;
    }
    let (first, last) = ((from / 8) as usize, ((to - 1) / 8) as usize);
    let low = 0xffu8 >> (from % 8);
    let high = if to % 8 == 0 {
        0xff
    } else {
        !(0xffu8 >> (to % 8))
    };
    if first == last {
        return Some((bytes[first] & low & high, &[], 0));
    }
    Some((
        bytes[first] & low,
        &bytes[first + 1..last],
        bytes[last] & high,
    ))
}

/// The operations `BITOP` takes.
///
/// The first four are Redis 2.6's and the last four went in with 8.2. They all
/// answer a bitmap as long as the longest source, since a source that is
/// shorter reads as zeros past its end, and `NOT` is the only one that will not
/// take more than one source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Bits set in every source.
    And,
    /// Bits set in any source.
    Or,
    /// Bits set in an odd number of sources.
    Xor,
    /// The complement of the one source.
    Not,
    /// Bits set in the first source and in none of the others.
    Diff,
    /// Bits set in one or more of the others and not in the first.
    Diff1,
    /// Bits set in the first source and in one or more of the others.
    AndOr,
    /// Bits set in exactly one source.
    One,
}

impl Op {
    /// The word a client sends, in any case.
    #[must_use]
    pub fn parse(word: &[u8]) -> Option<Op> {
        const NAMES: [(&[u8], Op); 8] = [
            (b"and", Op::And),
            (b"or", Op::Or),
            (b"xor", Op::Xor),
            (b"not", Op::Not),
            (b"diff", Op::Diff),
            (b"diff1", Op::Diff1),
            (b"andor", Op::AndOr),
            (b"one", Op::One),
        ];
        NAMES
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(word))
            .map(|&(_, op)| op)
    }

    /// The name, upper case, which is how the error sentences spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Op::And => "AND",
            Op::Or => "OR",
            Op::Xor => "XOR",
            Op::Not => "NOT",
            Op::Diff => "DIFF",
            Op::Diff1 => "DIFF1",
            Op::AndOr => "ANDOR",
            Op::One => "ONE",
        }
    }

    /// Whether this operation reads the first source differently from the rest.
    ///
    /// The three set difference shapes do, and it is the only reason `combine`
    /// keeps the first source apart from the fold over the others.
    const fn asymmetric(self) -> bool {
        matches!(self, Op::Diff | Op::Diff1 | Op::AndOr)
    }
}

/// How long the result of `op` over `srcs` will be.
///
/// As long as the longest source, since a shorter one reads as zeros past its
/// end. A caller sizes its destination with this and then fills it with
/// [`combine`].
///
/// # Panics
///
/// If `srcs` is empty, which is refused with a message on the wire.
pub fn width<'a, I>(srcs: I) -> usize
where
    I: Iterator<Item = &'a [u8]>,
{
    srcs.map(<[u8]>::len).max().expect("BITOP with no source")
}

/// Run `op` over `srcs`, filling `out`.
///
/// The sources are taken as an iterator that can be cloned rather than as a
/// slice, so that a caller with its sources end to end in one buffer does not
/// have to build a list of slices into it. The iterator is walked once per block
/// of the destination, which is why it has to be cloneable and why it should be
/// cheap to walk.
///
/// `out` is filled to whatever length it already has, and the caller gets that
/// length from [`width`]. Handing over a longer one is not wrong, it reads the
/// sources as zero padded out to there, which is the same rule that applies
/// inside the result anyway.
///
/// # Panics
///
/// If `srcs` is empty, or if it holds more than one source for [`Op::Not`].
/// Both are refused on the wire before this is called.
pub fn combine<'a, I>(op: Op, srcs: I, out: &mut [u8])
where
    I: Iterator<Item = &'a [u8]> + Clone,
{
    let mut count = srcs.clone();
    assert!(count.next().is_some(), "BITOP with no source");
    assert!(
        op != Op::Not || count.next().is_none(),
        "BITOP NOT with more"
    );
    let len = out.len();

    // One block of whichever source is being folded in, and one block held to
    // one side. Nothing needs the side buffer for two purposes at once: the
    // three difference shapes keep the first source there and `ONE` keeps its
    // "already seen once" mask there.
    let mut blk = [0u8; BLOCK];
    let mut side = [0u8; BLOCK];
    let mut at = 0;
    while at < len {
        let n = BLOCK.min(len - at);
        let acc = &mut out[at..at + n];
        let mut rest = srcs.clone();
        let first = rest.next().expect("a first source");
        // The first source seeds the accumulator for everything except the
        // three difference shapes, which need it again at the end and so keep
        // it to one side while the others are folded together.
        if op.asymmetric() {
            load(&mut side[..n], first, at);
            acc.fill(0);
        } else {
            load(acc, first, at);
            if op == Op::One {
                side[..n].fill(0);
            }
        }

        for src in rest {
            load(&mut blk[..n], src, at);
            let s = &blk[..n];
            match op {
                Op::And => fold(acc, s, |a, b| a & b),
                Op::Or | Op::Diff | Op::Diff1 | Op::AndOr => fold(acc, s, |a, b| a | b),
                Op::Xor => fold(acc, s, |a, b| a ^ b),
                Op::One => {
                    for (i, &b) in s.iter().enumerate() {
                        side[i] |= acc[i] & b;
                        acc[i] |= b;
                    }
                }
                Op::Not => unreachable!("NOT takes one source"),
            }
        }

        match op {
            Op::Not => {
                for a in acc.iter_mut() {
                    *a = !*a;
                }
            }
            // Set anywhere, minus set more than once.
            Op::One => fold(acc, &side[..n], |a, b| a & !b),
            // `side` still holds the first source's block, and `acc` holds the
            // others folded together with `OR`.
            Op::Diff => {
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = side[i] & !*a;
                }
            }
            Op::Diff1 => {
                for (i, a) in acc.iter_mut().enumerate() {
                    *a &= !side[i];
                }
            }
            Op::AndOr => {
                for (i, a) in acc.iter_mut().enumerate() {
                    *a &= side[i];
                }
            }
            Op::And | Op::Or | Op::Xor => {}
        }
        at += n;
    }
}

/// A block of `src` starting at `at`, zero padded past its end.
fn load(dst: &mut [u8], src: &[u8], at: usize) {
    let from = at.min(src.len());
    let take = (src.len() - from).min(dst.len());
    dst[..take].copy_from_slice(&src[from..from + take]);
    dst[take..].fill(0);
}

/// `acc[i] = f(acc[i], src[i])`, written so both backends vectorise it.
#[inline]
fn fold(acc: &mut [u8], src: &[u8], f: impl Fn(u8, u8) -> u8) {
    for (a, &b) in acc.iter_mut().zip(src) {
        *a = f(*a, b);
    }
}

// ------------------------------------------------------------ packed fields

/// One `BITFIELD` field type: `u8`, `i37` and so on.
///
/// Unsigned goes up to 63 bits and signed to 64, which is Redis's rule and not
/// an accident of the reply type: every value comes back as a RESP integer and
/// a RESP integer is signed, so a `u64` could not be reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field {
    /// Whether the top bit is a sign.
    signed: bool,
    /// How many bits wide, 1 to 64 signed and 1 to 63 unsigned.
    bits: u32,
}

impl Field {
    /// A field of `bits` bits, or `None` if that is not a width Redis takes.
    #[must_use]
    pub const fn new(signed: bool, bits: u32) -> Option<Field> {
        let top = if signed { 64 } else { 63 };
        if bits == 0 || bits > top {
            return None;
        }
        Some(Field { signed, bits })
    }

    /// The `u8` or `i37` a client sends.
    #[must_use]
    pub fn parse(word: &[u8]) -> Option<Field> {
        let (&kind, digits) = word.split_first()?;
        let signed = match kind {
            b'i' => true,
            b'u' => false,
            _ => return None,
        };
        if digits.is_empty() || digits.len() > 2 || !digits.iter().all(u8::is_ascii_digit) {
            return None;
        }
        let bits = digits
            .iter()
            .fold(0u32, |n, d| n * 10 + u32::from(d - b'0'));
        Field::new(signed, bits)
    }

    /// How wide.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Whether the top bit is a sign.
    #[must_use]
    pub const fn signed(self) -> bool {
        self.signed
    }

    /// The largest value it holds.
    #[must_use]
    pub const fn max(self) -> i64 {
        if self.signed {
            if self.bits == 64 {
                i64::MAX
            } else {
                (1i64 << (self.bits - 1)) - 1
            }
        } else if self.bits == 63 {
            i64::MAX
        } else {
            (1i64 << self.bits) - 1
        }
    }

    /// The smallest value it holds, which is zero when it is unsigned.
    #[must_use]
    pub const fn min(self) -> i64 {
        if !self.signed {
            0
        } else if self.bits == 64 {
            i64::MIN
        } else {
            -(1i64 << (self.bits - 1))
        }
    }

    /// The last bit a field at `at` touches, which is what a caller grows to.
    #[must_use]
    pub const fn last_bit(self, at: u64) -> u64 {
        at + self.bits as u64 - 1
    }

    /// The value truncated to this many bits, sign extended if it is signed.
    ///
    /// This is `OVERFLOW WRAP`, and it is the only one of the three that has to
    /// think about the width at all.
    #[must_use]
    const fn wrapped(self, n: i128) -> i64 {
        if self.bits == 64 {
            return n as i64;
        }
        let mask = (1i128 << self.bits) - 1;
        let low = n & mask;
        if self.signed && low > self.max() as i128 {
            (low - (1i128 << self.bits)) as i64
        } else {
            low as i64
        }
    }
}

/// What to do about a value that will not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overflow {
    /// Keep the low bits, which is what a counter that is allowed to lap does.
    #[default]
    Wrap,
    /// Stop at the end of the range.
    Sat,
    /// Do nothing and answer nothing.
    Fail,
}

impl Overflow {
    /// The word a client sends, in any case.
    #[must_use]
    pub fn parse(word: &[u8]) -> Option<Overflow> {
        if word.eq_ignore_ascii_case(b"wrap") {
            Some(Overflow::Wrap)
        } else if word.eq_ignore_ascii_case(b"sat") {
            Some(Overflow::Sat)
        } else if word.eq_ignore_ascii_case(b"fail") {
            Some(Overflow::Fail)
        } else {
            None
        }
    }
}

/// The field of `f` bits at bit `at`, reading past the end as zeros.
#[must_use]
pub fn get(bytes: &[u8], at: u64, f: Field) -> i64 {
    let raw = window(bytes, at, f.bits);
    if f.signed && f.bits < 64 && raw >= 1u64 << (f.bits - 1) {
        // The subtraction is done wide because at 63 bits the thing being taken
        // off is one past what an `i64` holds.
        (i128::from(raw) - (1i128 << f.bits)) as i64
    } else {
        raw as i64
    }
}

/// Write `val` into the field of `f` bits at bit `at`.
///
/// # Panics
///
/// If the slice does not reach the end of the field. Growing the value is the
/// caller's job, because only the caller knows whether it is allowed to.
pub fn set(bytes: &mut [u8], at: u64, f: Field, val: i64) {
    let last = ((at + u64::from(f.bits) - 1) / 8) as usize;
    assert!(last < bytes.len(), "the field runs off the end");
    let (byte, off) = ((at / 8) as usize, (at % 8) as u32);
    let span = ((off + f.bits).div_ceil(8)) as usize;
    // Nine bytes is the worst case, a 64 bit field starting one bit into a
    // byte, so the window is a u128 and never a wider read than that.
    let mut win: u128 = 0;
    for &b in &bytes[byte..byte + span] {
        win = (win << 8) | u128::from(b);
    }
    let shift = span as u32 * 8 - off - f.bits;
    let mask = ((1u128 << f.bits) - 1) << shift;
    win = (win & !mask) | ((u128::from(val as u64) << shift) & mask);
    for (i, b) in bytes[byte..byte + span].iter_mut().enumerate() {
        *b = (win >> ((span - 1 - i) * 8)) as u8;
    }
}

/// The raw bits of a field, as an unsigned number, zero past the end.
fn window(bytes: &[u8], at: u64, bits: u32) -> u64 {
    let (byte, off) = ((at / 8) as usize, (at % 8) as u32);
    let span = ((off + bits).div_ceil(8)) as usize;
    let mut win: u128 = 0;
    for i in 0..span {
        win = (win << 8) | u128::from(bytes.get(byte + i).copied().unwrap_or(0));
    }
    let shift = span as u32 * 8 - off - bits;
    let mask = (1u128 << bits) - 1;
    ((win >> shift) & mask) as u64
}

/// The value a `SET` of `val` should write, or `None` for `FAIL`.
///
/// A negative value into an unsigned field is the one that surprises people. It
/// is not clamped to zero: Redis reads the value as the unsigned number its
/// two's complement bits spell, which is enormous, so it overflows off the top
/// and `SAT` gives the field's maximum rather than nothing. Measured on 8.10.1:
/// `OVERFLOW SAT SET u8 0 -5` leaves 255 and `OVERFLOW WRAP` leaves 251.
#[must_use]
pub fn setting(f: Field, val: i64, on: Overflow) -> Option<i64> {
    let want = if f.signed {
        i128::from(val)
    } else {
        i128::from(val as u64)
    };
    fit(f, want, on)
}

/// The value an `INCRBY` of `by` should write, or `None` for `FAIL`.
#[must_use]
pub fn adding(f: Field, had: i64, by: i64, on: Overflow) -> Option<i64> {
    fit(f, i128::from(had) + i128::from(by), on)
}

/// `want` brought into the field's range the way `on` says to.
fn fit(f: Field, want: i128, on: Overflow) -> Option<i64> {
    if want > i128::from(f.max()) {
        return match on {
            Overflow::Wrap => Some(f.wrapped(want)),
            Overflow::Sat => Some(f.max()),
            Overflow::Fail => None,
        };
    }
    if want < i128::from(f.min()) {
        return match on {
            Overflow::Wrap => Some(f.wrapped(want)),
            Overflow::Sat => Some(f.min()),
            Overflow::Fail => None,
        };
    }
    Some(want as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The obvious version of everything in this file, one bit at a time.
    fn slow_bit(bytes: &[u8], at: u64) -> bool {
        let (byte, off) = ((at / 8) as usize, (at % 8) as u32);
        bytes.get(byte).is_some_and(|b| b & (0x80 >> off) != 0)
    }

    fn slow_count(bytes: &[u8], from: u64, to: u64) -> u64 {
        (from..to).filter(|&i| slow_bit(bytes, i)).count() as u64
    }

    fn slow_find(bytes: &[u8], set: bool, from: u64, to: u64) -> Option<u64> {
        (from..to.min((bytes.len() as u64) * 8)).find(|&i| slow_bit(bytes, i) == set)
    }

    fn slow_get(bytes: &[u8], at: u64, f: Field) -> i64 {
        let mut raw = 0u64;
        for i in 0..u64::from(f.bits()) {
            raw = (raw << 1) | u64::from(slow_bit(bytes, at + i));
        }
        if f.signed() && f.bits() < 64 && raw >= 1u64 << (f.bits() - 1) {
            (i128::from(raw) - (1i128 << f.bits())) as i64
        } else {
            raw as i64
        }
    }

    /// A repeatable spread of bytes, since none of this is about randomness.
    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn counting_agrees_with_counting_one_bit_at_a_time() {
        for len in [0usize, 1, 7, 8, 31, 32, 33, 100, 257] {
            let bytes = noise(len, len as u64 + 7);
            assert_eq!(count(&bytes), slow_count(&bytes, 0, len as u64 * 8));
        }
    }

    #[test]
    fn every_range_counts_what_a_bit_loop_counts() {
        let bytes = noise(37, 99);
        let bits = 37 * 8;
        for from in (0..bits).step_by(7) {
            for to in (from..bits + 16).step_by(5) {
                assert_eq!(
                    count_range(&bytes, from, to),
                    slow_count(&bytes, from, to.min(bits)),
                    "{from}..{to}"
                );
            }
        }
        // Backwards and empty ranges are nothing rather than a panic.
        assert_eq!(count_range(&bytes, 10, 10), 0);
        assert_eq!(count_range(&bytes, 20, 3), 0);
        assert_eq!(count_range(&[], 0, 64), 0);
    }

    #[test]
    fn finding_agrees_with_scanning_one_bit_at_a_time() {
        // Long enough that the word skip in the middle runs, and with runs of
        // all ones and all zeros in it so that it has something to skip.
        let mut bytes = noise(300, 5);
        bytes[64..128].fill(0);
        bytes[160..224].fill(0xff);
        let bits = bytes.len() as u64 * 8;
        for set in [true, false] {
            for from in (0..bits).step_by(11) {
                for to in [from, from + 1, from + 63, from + 700, bits, bits + 9] {
                    assert_eq!(
                        find(&bytes, set, from, to),
                        slow_find(&bytes, set, from, to),
                        "set={set} {from}..{to}"
                    );
                }
            }
        }
        assert_eq!(find(&[], true, 0, 64), None);
        assert_eq!(find(&[0xff], false, 0, 8), None);
        assert_eq!(find(&[0xff], true, 0, 8), Some(0));
    }

    #[test]
    fn a_bit_is_counted_from_the_top_of_the_first_byte() {
        assert_eq!(find(&[0x01], true, 0, 8), Some(7));
        assert_eq!(find(&[0x80], true, 0, 8), Some(0));
        assert_eq!(count(&[0x01]), 1);
    }

    /// The shapes measured on 8.10.1, which is where these came from.
    #[test]
    fn the_eight_operations_are_what_a_real_server_does() {
        let a: &[u8] = &[0xf0, 0x0f, 0xff];
        let b: &[u8] = &[0xff, 0x00];
        let c: &[u8] = &[0x0f];
        let mut out = Vec::new();
        let run = |op, srcs: &[&[u8]], out: &mut Vec<u8>| {
            out.clear();
            out.resize(width(srcs.iter().copied()), 0);
            combine(op, srcs.iter().copied(), out);
        };

        run(Op::And, &[a, b], &mut out);
        assert_eq!(out, vec![0xf0, 0x00, 0x00], "and, padded with zeros");
        run(Op::Or, &[a, b], &mut out);
        assert_eq!(out, vec![0xff, 0x0f, 0xff]);
        run(Op::Xor, &[a, b], &mut out);
        assert_eq!(out, vec![0x0f, 0x0f, 0xff]);
        run(Op::Not, &[a], &mut out);
        assert_eq!(out, vec![0x0f, 0xf0, 0x00]);
        run(Op::Diff, &[a, b], &mut out);
        assert_eq!(out, vec![0x00, 0x0f, 0xff], "in a and in nothing else");
        run(Op::Diff1, &[a, b], &mut out);
        assert_eq!(out, vec![0x0f, 0x00, 0x00], "in the others and not in a");
        run(Op::AndOr, &[a, b, c], &mut out);
        assert_eq!(out, vec![0xf0, 0x00, 0x00]);
        run(Op::One, &[a, b, c], &mut out);
        assert_eq!(out, vec![0x00, 0x0f, 0xff], "set in exactly one of them");

        // One source is a copy for everything that takes one.
        for op in [Op::And, Op::Or, Op::Xor, Op::One] {
            run(op, &[a], &mut out);
            assert_eq!(out, a, "{} of one source", op.name());
        }
    }

    /// The block loop is only exercised by something longer than a block.
    #[test]
    fn combining_crosses_the_block_boundary() {
        let a = noise(BLOCK * 2 + 37, 1);
        let b = noise(BLOCK + 3, 2);
        let c = noise(BLOCK * 3, 3);
        let srcs: [&[u8]; 3] = [&a, &b, &c];
        let mut out = Vec::new();

        let at = |s: &[u8], i: usize| s.get(i).copied().unwrap_or(0);
        for op in [
            Op::And,
            Op::Or,
            Op::Xor,
            Op::Diff,
            Op::Diff1,
            Op::AndOr,
            Op::One,
        ] {
            out.clear();
            out.resize(width(srcs.iter().copied()), 0);
            combine(op, srcs.iter().copied(), &mut out);
            assert_eq!(out.len(), c.len(), "{}", op.name());
            for (i, &got) in out.iter().enumerate() {
                let (x, y, z) = (at(&a, i), at(&b, i), at(&c, i));
                let want = match op {
                    Op::And => x & y & z,
                    Op::Or => x | y | z,
                    Op::Xor => x ^ y ^ z,
                    Op::Diff => x & !(y | z),
                    Op::Diff1 => (y | z) & !x,
                    Op::AndOr => x & (y | z),
                    Op::One => (x & !y & !z) | (y & !x & !z) | (z & !x & !y),
                    Op::Not => unreachable!(),
                };
                assert_eq!(got, want, "{} at byte {i}", op.name());
            }
        }
    }

    #[test]
    fn an_operation_is_named_in_any_case() {
        assert_eq!(Op::parse(b"AND"), Some(Op::And));
        assert_eq!(Op::parse(b"diff1"), Some(Op::Diff1));
        assert_eq!(Op::parse(b"AnDoR"), Some(Op::AndOr));
        assert_eq!(Op::parse(b"nope"), None);
        assert_eq!(Op::And.name(), "AND");
    }

    #[test]
    fn a_field_type_is_a_letter_and_a_width() {
        assert_eq!(Field::parse(b"u8").map(Field::bits), Some(8));
        assert_eq!(Field::parse(b"i64").map(Field::signed), Some(true));
        assert_eq!(Field::parse(b"u63").map(Field::bits), Some(63));
        // The two Redis refuses, for the reason it refuses them: a u64 would
        // not fit in the signed integer the reply is.
        assert_eq!(Field::parse(b"u64"), None);
        assert_eq!(Field::parse(b"i65"), None);
        assert_eq!(Field::parse(b"u0"), None);
        assert_eq!(Field::parse(b"x8"), None);
        assert_eq!(Field::parse(b"u"), None);
        assert_eq!(Field::parse(b""), None);
        assert_eq!(Field::parse(b"u008"), None);
    }

    #[test]
    fn a_field_knows_its_own_range() {
        let f = |s, b| Field::new(s, b).expect("a width");
        assert_eq!((f(false, 8).min(), f(false, 8).max()), (0, 255));
        assert_eq!((f(true, 8).min(), f(true, 8).max()), (-128, 127));
        assert_eq!((f(true, 1).min(), f(true, 1).max()), (-1, 0));
        assert_eq!((f(false, 1).min(), f(false, 1).max()), (0, 1));
        assert_eq!((f(true, 64).min(), f(true, 64).max()), (i64::MIN, i64::MAX));
        assert_eq!((f(false, 63).min(), f(false, 63).max()), (0, i64::MAX));
    }

    #[test]
    fn reading_and_writing_a_field_agrees_with_a_bit_loop() {
        let mut bytes = noise(64, 3);
        for bits in [1u32, 2, 7, 8, 9, 31, 32, 33, 63, 64] {
            for signed in [true, false] {
                let Some(f) = Field::new(signed, bits) else {
                    continue;
                };
                for at in 0..64u64 {
                    assert_eq!(get(&bytes, at, f), slow_get(&bytes, at, f), "{f:?} at {at}");
                }
            }
        }
        // A field that runs off the end reads the missing bytes as zeros, which
        // is what `BITFIELD GET` past the end of a string answers.
        let f = Field::new(false, 16).expect("a width");
        assert_eq!(get(&[0xff], 0, f), 0xff00);
        assert_eq!(get(&[], 0, f), 0);

        // And a write comes back out, wherever it is put.
        for bits in [1u32, 5, 8, 13, 32, 64] {
            for signed in [true, false] {
                let Some(f) = Field::new(signed, bits) else {
                    continue;
                };
                for at in [0u64, 1, 7, 8, 63, 100] {
                    // Both ends of the range, since an `i1` holds -1 and 0 and
                    // nothing an ordinary loop over small numbers would try.
                    for want in [f.min(), f.max()] {
                        set(&mut bytes, at, f, want);
                        assert_eq!(get(&bytes, at, f), want, "{f:?} at {at}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_write_leaves_the_bits_around_it_alone() {
        let mut bytes = [0xffu8; 4];
        let f = Field::new(false, 3).expect("a width");
        set(&mut bytes, 5, f, 0);
        assert_eq!(bytes, [0xf8, 0xff, 0xff, 0xff]);
        set(&mut bytes, 29, f, 0);
        assert_eq!(bytes, [0xf8, 0xff, 0xff, 0xf8]);
    }

    /// Every one of these was read off a running 8.10.1.
    #[test]
    fn overflow_is_what_a_real_server_does() {
        let u8f = Field::new(false, 8).expect("a width");
        let i8f = Field::new(true, 8).expect("a width");

        assert_eq!(setting(u8f, 300, Overflow::Wrap), Some(44));
        assert_eq!(setting(u8f, 300, Overflow::Sat), Some(255));
        assert_eq!(setting(u8f, 300, Overflow::Fail), None);
        // The negative into unsigned case, which is the one nobody guesses.
        assert_eq!(setting(u8f, -5, Overflow::Wrap), Some(251));
        assert_eq!(setting(u8f, -5, Overflow::Sat), Some(255));
        assert_eq!(setting(u8f, -5, Overflow::Fail), None);
        // Signed compares as signed, so it saturates at the near end.
        assert_eq!(setting(i8f, -200, Overflow::Sat), Some(-128));
        assert_eq!(setting(i8f, 200, Overflow::Sat), Some(127));
        assert_eq!(setting(i8f, -200, Overflow::Wrap), Some(56));

        assert_eq!(adding(u8f, 255, 10, Overflow::Wrap), Some(9));
        assert_eq!(adding(u8f, 255, 250, Overflow::Sat), Some(255));
        assert_eq!(adding(u8f, 255, 250, Overflow::Fail), None);
        assert_eq!(adding(u8f, 0, -1000, Overflow::Sat), Some(0));
        assert_eq!(adding(u8f, 0, -1, Overflow::Wrap), Some(255));

        let i64f = Field::new(true, 64).expect("a width");
        assert_eq!(adding(i64f, i64::MAX, 1, Overflow::Wrap), Some(i64::MIN));
        assert_eq!(adding(i64f, i64::MAX, 1, Overflow::Sat), Some(i64::MAX));
        assert_eq!(adding(i64f, i64::MIN, -1, Overflow::Sat), Some(i64::MIN));
        assert_eq!(adding(i64f, i64::MIN, -1, Overflow::Fail), None);

        let u1 = Field::new(false, 1).expect("a width");
        assert_eq!(adding(u1, 1, 5, Overflow::Sat), Some(1));
        assert_eq!(adding(u1, 1, 3, Overflow::Wrap), Some(0));
    }

    #[test]
    fn an_overflow_word_is_read_in_any_case() {
        assert_eq!(Overflow::parse(b"WRAP"), Some(Overflow::Wrap));
        assert_eq!(Overflow::parse(b"sat"), Some(Overflow::Sat));
        assert_eq!(Overflow::parse(b"Fail"), Some(Overflow::Fail));
        assert_eq!(Overflow::parse(b"nope"), None);
        assert_eq!(Overflow::default(), Overflow::Wrap);
    }
}
