//! The HyperLogLog sketch, byte for byte the one Redis writes.
//!
//! A HyperLogLog in Redis is a string, the same as a bitmap is, and the bytes of
//! that string are a documented format rather than an internal detail. A client
//! can `GET` a sketch out of Redis, `SET` it into us, and `PFCOUNT` it here, and
//! it has to answer the same number. That is the whole reason this file copies
//! the layout instead of picking a better one, and it is why every constant
//! below is Redis's constant.
//!
//! # The layout
//!
//! Sixteen header bytes, then the registers. The header is the four magic bytes
//! `HYLL`, an encoding byte, three unused bytes, and eight bytes of cached
//! cardinality, little endian, with the top bit of the last one meaning the
//! cache is stale. Every write sets that bit and [`Keyspace::pfcount`] clears it
//! again, which is why a read of a sketch is really a write.
//!
//! [`Keyspace::pfcount`]: crate::Keyspace::pfcount
//!
//! There are 16384 registers of six bits each, so the dense form is 12288 bytes
//! of registers and 12304 bytes altogether. The six bit fields are packed low
//! bits first, which is the opposite way round from the bit a `SETBIT` names,
//! and the two are unrelated: this packing is internal to the sketch.
//!
//! The sparse form is a run length encoding of the same registers, and a sketch
//! stays in it until it either needs a register larger than 32 or would grow
//! past [`SPARSE_MAX`] bytes. Three opcodes: `ZERO` is one byte and up to 64
//! empty registers, `XZERO` is two bytes and up to 16384 of them, and `VAL` is
//! one byte holding a value from 1 to 32 repeated up to four times. An empty
//! sketch is one `XZERO` covering all 16384 registers, which is 18 bytes, and
//! that is what `PFADD k` with no elements leaves behind.
//!
//! # The parts that had to be copied exactly
//!
//! The hash is MurmurHash64A with the seed `0xadc83b19`, and the register index
//! is the low fourteen bits of it. Checked against a running 8.10.1: `a` lands
//! in register 12711 with a count of 2, `b` in 15780 and `c` in 8436, and those
//! are the registers a real server has after `PFADD h a b c`.
//!
//! The estimator is Ertl's, the one Redis moved to in 5.0, out of "New
//! cardinality estimation algorithms for HyperLogLog sketches". It is not the
//! original bias corrected estimator and it is not LogLog-Beta, and using either
//! of those would answer a different number for the same bytes.
//!
//! [`set`] is the part with the most rules in it and all of them are Redis's:
//! which opcode a run splits into, when a split is written as `ZERO` rather than
//! `XZERO`, and the five opcode window that merges neighbouring `VAL` runs
//! afterwards. Getting any of those wrong still gives a working sketch that
//! counts correctly and does not give the same bytes, and the same bytes are the
//! point.

use yo_common::{Code, Error, Result};

/// The number of bits of the hash that pick a register.
pub const P: u32 = 14;
/// How many registers a sketch has.
pub const REGISTERS: usize = 1 << P;
/// How many bits of the hash are left to count zeros in.
pub const Q: u32 = 64 - P;
/// How many bits a dense register takes.
const BITS: usize = 6;
/// The largest value a register can hold, which is what six bits reach.
const REGISTER_MAX: u32 = 63;
/// The header, in bytes.
pub const HDR: usize = 16;
/// The length of a dense sketch, header included.
pub const DENSE: usize = HDR + REGISTERS * BITS / 8;
/// How large a sparse sketch is allowed to get before it turns dense.
///
/// Redis calls this `hll-sparse-max-bytes` and defaults it to 3000. Above it the
/// run length encoding stops paying for itself, both in space and in the walk
/// every read does over it.
pub const SPARSE_MAX: usize = 3000;

/// The four bytes every sketch starts with.
const MAGIC: [u8; 4] = *b"HYLL";
/// The encoding byte of a dense sketch.
const DENSE_TAG: u8 = 0;
/// The encoding byte of a sparse sketch.
const SPARSE_TAG: u8 = 1;
/// The seed Redis hashes with, and the reason our registers are its registers.
const SEED: u64 = 0xadc8_3b19;

/// The top bits of an `XZERO` opcode.
const XZERO_BIT: u8 = 0x40;
/// The top bit of a `VAL` opcode.
const VAL_BIT: u8 = 0x80;
/// The longest run a one byte `ZERO` opcode can hold.
const ZERO_MAX: usize = 64;
/// The largest value a `VAL` opcode can hold, and the promotion trigger.
const VAL_MAX: u8 = 32;
/// How many times a `VAL` opcode can repeat its value.
const VAL_MAX_LEN: usize = 4;

/// What Redis says about a string that is not a sketch.
///
/// The word `WRONGTYPE` is in the message rather than only in the prefix on a
/// real server, and the sentence has a full stop on the end, which the ordinary
/// wrong type error does not.
const NOT_HLL: &str = "Key is not a valid HyperLogLog string value.";
/// What Redis says about a sketch whose opcodes do not add up.
const CORRUPT: &str = "Corrupted HLL object detected";

/// Which of the two representations a sketch is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Every register written out, 12304 bytes whatever the cardinality.
    Dense,
    /// The registers run length encoded, which is smaller while most of them
    /// are still empty.
    Sparse,
}

impl Encoding {
    /// The word `PFDEBUG ENCODING` answers with.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Dense => "dense",
            Encoding::Sparse => "sparse",
        }
    }
}

/// The error a command answers for a string that is not a sketch.
#[must_use]
pub fn not_hll() -> Error {
    Error::new(Code::WrongType, NOT_HLL)
}

/// The error a command answers for a sketch whose opcodes do not add up.
///
/// The prefix a real server writes is neither `ERR` nor `WRONGTYPE`, it is
/// `INVALIDOBJ`, and [`Code::Corrupt`] is what the wire layer turns into that.
/// Nothing else on the command path answers that code, which is what makes the
/// mapping safe to make there rather than here.
#[must_use]
pub fn corrupt() -> Error {
    Error::new(Code::Corrupt, CORRUPT)
}

/// MurmurHash64A, the one Redis hashes elements with.
///
/// Not a hash anybody would choose today, and not one we use anywhere else. It
/// is here because the register an element lands in is part of the file format:
/// hash an element differently and the sketch is still a valid sketch and no
/// longer the same sketch a real server would have written.
#[must_use]
pub fn hash(ele: &[u8]) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;
    let mut h = SEED ^ (ele.len() as u64).wrapping_mul(M);
    let (blocks, tail) = ele.as_chunks::<8>();
    for block in blocks {
        let mut k = u64::from_le_bytes(*block);
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
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

/// Which register an element belongs to, and what it wants written there.
///
/// The low fourteen bits pick the register and the rest is scanned for its first
/// set bit, counting from one. A bit is forced in at position `Q` so that the
/// scan always terminates, which caps the answer at `Q + 1` and is why 51 is the
/// largest value a register ever holds.
#[must_use]
pub fn place(ele: &[u8]) -> (usize, u8) {
    let h = hash(ele);
    let index = (h & (REGISTERS as u64 - 1)) as usize;
    let rest = (h >> P) | (1 << Q);
    (index, rest.trailing_zeros() as u8 + 1)
}

/// Write an empty sketch, which is the header and one `XZERO` over everything.
pub fn empty(out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(&MAGIC);
    out.push(SPARSE_TAG);
    out.extend_from_slice(&[0; 3]);
    out.extend_from_slice(&[0; 8]);
    let mut left = REGISTERS;
    while left > 0 {
        let run = left.min(1 << P);
        out.extend_from_slice(&xzero_bytes(run));
        left -= run;
    }
}

/// Check that a string really is a sketch, and say which kind.
///
/// The rules are Redis's and the order matters as little as it looks: a string
/// shorter than the header, a bad magic, an encoding byte that is neither of the
/// two, and a dense sketch whose length is not exactly [`DENSE`] all answer the
/// same sentence.
pub fn check(bytes: &[u8]) -> Result<Encoding> {
    if bytes.len() < HDR || bytes[..4] != MAGIC {
        return Err(not_hll());
    }
    match bytes[4] {
        DENSE_TAG if bytes.len() == DENSE => Ok(Encoding::Dense),
        SPARSE_TAG => Ok(Encoding::Sparse),
        _ => Err(not_hll()),
    }
}

/// The cached cardinality, or `None` when a write has invalidated it.
#[must_use]
pub fn cached(bytes: &[u8]) -> Option<u64> {
    let card = u64::from_le_bytes(bytes[8..16].try_into().expect("eight bytes"));
    (card >> 63 == 0).then_some(card)
}

/// Write a freshly computed cardinality into the header, marking it good.
pub fn cache(bytes: &mut [u8], n: u64) {
    bytes[8..16].copy_from_slice(&(n & !(1 << 63)).to_le_bytes());
}

/// Mark the cached cardinality stale, which every write does.
pub fn invalidate(bytes: &mut [u8]) {
    bytes[15] |= 0x80;
}

/// Read one dense register.
///
/// The six bit fields are packed low bits first, so a field either sits inside
/// one byte or straddles two. Redis always reads the second byte and relies on
/// the string being null terminated to make the last register safe; we stop at
/// the end of the slice instead, which is the same zero.
#[must_use]
#[inline]
pub fn dense_get(regs: &[u8], index: usize) -> u8 {
    let bit = index * BITS;
    let (byte, shift) = (bit / 8, (bit % 8) as u32);
    let low = u32::from(regs[byte]);
    let high = regs.get(byte + 1).map_or(0, |&b| u32::from(b));
    (((low >> shift) | (high << (8 - shift))) & REGISTER_MAX) as u8
}

/// Write one dense register, answering whether it changed.
///
/// The second byte is only touched when the field really straddles two, which is
/// what keeps a write to the last register inside the slice.
#[inline]
pub fn dense_set(regs: &mut [u8], index: usize, val: u8) -> bool {
    if dense_get(regs, index) >= val {
        return false;
    }
    let bit = index * BITS;
    let (byte, shift) = (bit / 8, (bit % 8) as u32);
    let v = u32::from(val);
    regs[byte] = ((u32::from(regs[byte]) & !(REGISTER_MAX << shift)) | (v << shift)) as u8;
    if shift > 2 {
        let rest = 8 - shift;
        let high = &mut regs[byte + 1];
        *high = ((u32::from(*high) & !(REGISTER_MAX >> rest)) | (v >> rest)) as u8;
    }
    true
}

/// Whether the byte at `p` starts a `ZERO` opcode.
const fn is_zero(b: u8) -> bool {
    b & 0xc0 == 0
}

/// Whether the byte at `p` starts an `XZERO` opcode.
const fn is_xzero(b: u8) -> bool {
    b & 0xc0 == XZERO_BIT
}

/// Whether the byte at `p` is a `VAL` opcode.
const fn is_val(b: u8) -> bool {
    b & VAL_BIT != 0
}

/// How many registers a `ZERO` opcode covers.
const fn zero_len(b: u8) -> usize {
    (b & 0x3f) as usize + 1
}

/// How many registers an `XZERO` opcode covers.
const fn xzero_len(a: u8, b: u8) -> usize {
    (((a & 0x3f) as usize) << 8 | b as usize) + 1
}

/// The value a `VAL` opcode holds.
const fn val_value(b: u8) -> u8 {
    ((b >> 2) & 0x1f) + 1
}

/// How many registers a `VAL` opcode covers.
const fn val_len(b: u8) -> usize {
    (b & 3) as usize + 1
}

/// A `VAL` opcode holding `val` repeated `len` times.
const fn val_byte(val: u8, len: usize) -> u8 {
    ((val - 1) << 2) | (len as u8 - 1) | VAL_BIT
}

/// A `ZERO` opcode covering `len` registers.
const fn zero_byte(len: usize) -> u8 {
    (len - 1) as u8
}

/// An `XZERO` opcode covering `len` registers.
const fn xzero_bytes(len: usize) -> [u8; 2] {
    let n = len - 1;
    [((n >> 8) as u8) | XZERO_BIT, (n & 0xff) as u8]
}

/// How many registers the opcode at `at` covers, and how many bytes it is.
fn opcode(sparse: &[u8], at: usize) -> Option<(usize, usize)> {
    let b = *sparse.get(at)?;
    if is_zero(b) {
        Some((zero_len(b), 1))
    } else if is_xzero(b) {
        Some((xzero_len(b, *sparse.get(at + 1)?), 2))
    } else {
        Some((val_len(b), 1))
    }
}

/// Walk a sparse body, handing each run to `each` as a value and a length.
///
/// `false` when the runs do not add up to exactly [`REGISTERS`], which is the
/// only corruption any of the readers here can detect and the one Redis checks
/// for as well.
fn walk(sparse: &[u8], mut each: impl FnMut(u8, usize, usize)) -> bool {
    let mut at = 0;
    let mut index = 0;
    while at < sparse.len() {
        let b = sparse[at];
        if is_val(b) {
            let len = val_len(b);
            if index + len > REGISTERS {
                return false;
            }
            each(val_value(b), index, len);
            index += len;
            at += 1;
        } else if is_zero(b) {
            index += zero_len(b);
            at += 1;
        } else {
            let Some(&next) = sparse.get(at + 1) else {
                return false;
            };
            index += xzero_len(b, next);
            at += 2;
        }
    }
    index == REGISTERS
}

/// Turn a sparse sketch into a dense one in place.
///
/// `false` for a body whose opcodes do not add up, which leaves the buffer as it
/// was. The registers go through the stack rather than through a second buffer,
/// which is sixteen kibibytes and the same thing Redis does when it merges.
pub fn to_dense(buf: &mut Vec<u8>) -> bool {
    if buf[4] == DENSE_TAG {
        return true;
    }
    let mut regs = [0u8; REGISTERS];
    if !walk(&buf[HDR..], |val, at, len| {
        regs[at..at + len].fill(val);
    }) {
        return false;
    }
    buf.truncate(HDR);
    buf.resize(DENSE, 0);
    buf[4] = DENSE_TAG;
    let body = &mut buf[HDR..];
    for (i, &val) in regs.iter().enumerate() {
        if val != 0 {
            dense_set(body, i, val);
        }
    }
    true
}

/// Raise one register to `val`, answering whether anything changed.
///
/// `None` for a corrupted body. A sparse sketch turns dense here when the value
/// will not fit in a `VAL` opcode or when the rewrite would push it past
/// [`SPARSE_MAX`], and the caller does not have to know which happened.
pub fn set(buf: &mut Vec<u8>, index: usize, val: u8) -> Option<bool> {
    if buf[4] == DENSE_TAG {
        let changed = dense_set(&mut buf[HDR..], index, val);
        if changed {
            invalidate(buf);
        }
        return Some(changed);
    }
    if val > VAL_MAX {
        return promote(buf, index, val);
    }

    // Find the opcode covering the register, keeping the one before it, which is
    // where the merge pass at the end starts from.
    let (mut at, mut first, mut prev, mut span) = (HDR, 0usize, None, 0usize);
    while at < buf.len() {
        let (covers, bytes) = opcode(buf, at)?;
        span = covers;
        if index < first + span {
            break;
        }
        prev = Some(at);
        at += bytes;
        first += span;
    }
    if span == 0 || at >= buf.len() {
        return None;
    }

    let here = buf[at];
    let (zero, xzero, run) = if is_val(here) {
        (false, false, val_len(here))
    } else if is_zero(here) {
        (true, false, zero_len(here))
    } else {
        (false, true, xzero_len(here, *buf.get(at + 1)?))
    };

    // Two shapes need no rewriting at all. A run already holding a value this
    // large is the common case once a sketch has any size to it, and a single
    // register run is written over where it lies whatever it held.
    if is_val(here) {
        if val_value(here) >= val {
            return Some(false);
        }
        if run == 1 {
            buf[at] = val_byte(val, 1);
            return Some(finish(buf, prev));
        }
    }
    if zero && run == 1 {
        buf[at] = val_byte(val, 1);
        return Some(finish(buf, prev));
    }

    // Everything else splits the run into up to three opcodes, which is five
    // bytes in the worst case: an `XZERO` on each side of a one register `VAL`.
    let mut seq = [0u8; 5];
    let mut n = 0;
    let last = first + span - 1;
    let gap = |seq: &mut [u8; 5], n: &mut usize, len: usize| {
        if len > ZERO_MAX {
            seq[*n..*n + 2].copy_from_slice(&xzero_bytes(len));
            *n += 2;
        } else {
            seq[*n] = zero_byte(len);
            *n += 1;
        }
    };
    if zero || xzero {
        if index != first {
            gap(&mut seq, &mut n, index - first);
        }
        seq[n] = val_byte(val, 1);
        n += 1;
        if index != last {
            gap(&mut seq, &mut n, last - index);
        }
    } else {
        let had = val_value(here);
        if index != first {
            seq[n] = val_byte(had, index - first);
            n += 1;
        }
        seq[n] = val_byte(val, 1);
        n += 1;
        if index != last {
            seq[n] = val_byte(had, last - index);
            n += 1;
        }
    }

    // Put the new opcodes where the old one was. Growing past the sparse limit
    // is what turns a sketch dense in the ordinary case, long before any single
    // register needs a value larger than 32.
    let old = if xzero { 2 } else { 1 };
    let end = buf.len();
    if n > old && end + (n - old) > SPARSE_MAX {
        return promote(buf, index, val);
    }
    if n > old {
        buf.resize(end + (n - old), 0);
        buf.copy_within(at + old..end, at + n);
    } else if n < old {
        buf.copy_within(at + old..end, at + n);
        buf.truncate(end - (old - n));
    }
    buf[at..at + n].copy_from_slice(&seq[..n]);
    Some(finish(buf, prev))
}

/// Turn the sketch dense and write the register into it.
fn promote(buf: &mut Vec<u8>, index: usize, val: u8) -> Option<bool> {
    if !to_dense(buf) {
        return None;
    }
    let changed = dense_set(&mut buf[HDR..], index, val);
    invalidate(buf);
    Some(changed)
}

/// Tidy up after a write and mark the cache stale.
///
/// The tidying is Redis's five opcode window: a split can leave two `VAL`
/// opcodes holding the same value next to each other, and joining them back up
/// is what stops a sketch from growing a byte every time a register is written
/// twice. Five is Redis's number and it is enough, because a single write can
/// only ever produce three new opcodes.
fn finish(buf: &mut Vec<u8>, prev: Option<usize>) -> bool {
    let mut at = prev.unwrap_or(HDR);
    let mut left = 5;
    while at < buf.len() && left > 0 {
        left -= 1;
        let b = buf[at];
        if is_xzero(b) {
            at += 2;
            continue;
        }
        if is_zero(b) {
            at += 1;
            continue;
        }
        if let Some(&next) = buf.get(at + 1)
            && is_val(next)
            && val_value(b) == val_value(next)
        {
            let len = val_len(b) + val_len(next);
            if len <= VAL_MAX_LEN {
                buf[at + 1] = val_byte(val_value(b), len);
                let end = buf.len();
                buf.copy_within(at + 1..end, at);
                buf.truncate(end - 1);
                // Try the merged opcode against its new neighbour before moving
                // on, which is what lets four ones become one run.
                continue;
            }
        }
        at += 1;
    }
    invalidate(buf);
    true
}

/// How many registers hold each value, which is all the estimator needs.
///
/// `None` for a body whose opcodes do not add up.
fn histogram(bytes: &[u8], enc: Encoding) -> Option<[u32; 64]> {
    let mut hist = [0u32; 64];
    match enc {
        Encoding::Dense => {
            let regs = &bytes[HDR..];
            for i in 0..REGISTERS {
                hist[dense_get(regs, i) as usize] += 1;
            }
        }
        Encoding::Sparse => {
            let mut seen = 0;
            if !walk(&bytes[HDR..], |val, _, len| {
                hist[val as usize] += len as u32;
                seen += len as u32;
            }) {
                return None;
            }
            hist[0] = REGISTERS as u32 - seen;
        }
    }
    Some(hist)
}

/// Every register of a sketch, which is what a merge and `PFDEBUG GETREG` want.
///
/// `false` for a body whose opcodes do not add up. The registers are raised to
/// what the sketch holds rather than overwritten, so merging several sketches is
/// calling this once per sketch over the same array.
pub fn merge(max: &mut [u8; REGISTERS], bytes: &[u8], enc: Encoding) -> bool {
    match enc {
        Encoding::Dense => {
            let regs = &bytes[HDR..];
            for (i, slot) in max.iter_mut().enumerate() {
                *slot = (*slot).max(dense_get(regs, i));
            }
            true
        }
        Encoding::Sparse => walk(&bytes[HDR..], |val, at, len| {
            for slot in &mut max[at..at + len] {
                *slot = (*slot).max(val);
            }
        }),
    }
}

/// Ertl's tau, the correction for the registers that are already saturated.
fn tau(mut x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let mut y = 1.0;
    let mut z = 1.0 - x;
    loop {
        x = x.sqrt();
        let was = z;
        y *= 0.5;
        z -= (1.0 - x).powi(2) * y;
        if was == z {
            return z / 3.0;
        }
    }
}

/// Ertl's sigma, the correction for the registers that are still empty.
fn sigma(mut x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    let mut y = 1.0;
    let mut z = x;
    loop {
        x *= x;
        let was = z;
        z += x * y;
        y += y;
        if was == z {
            return z;
        }
    }
}

/// The cardinality a register histogram implies.
///
/// This is Ertl's estimator out of "New cardinality estimation algorithms for
/// HyperLogLog sketches", which is what Redis has used since 5.0. It replaced
/// the original bias corrected estimator and the switch to linear counting at
/// small cardinalities, and it is a single expression with no thresholds in it.
/// Writing a different estimator here would answer a different number for a
/// sketch a real server wrote, which is the one thing this file must not do.
#[must_use]
pub fn estimate(hist: &[u32; 64]) -> u64 {
    /// One over twice the natural log of two, which is the limit of the alpha
    /// correction as the register count grows. Redis writes it out as a literal
    /// rather than computing it and so do we, so the last bit agrees.
    const ALPHA_INF: f64 = 0.721_347_520_444_481_7;
    let m = REGISTERS as f64;
    let mut z = m * tau((m - f64::from(hist[Q as usize + 1])) / m);
    for j in (1..=Q as usize).rev() {
        z += f64::from(hist[j]);
        z *= 0.5;
    }
    z += m * sigma(f64::from(hist[0]) / m);
    (ALPHA_INF * m * m / z).round() as u64
}

/// The cardinality of one sketch, ignoring whatever the header has cached.
pub fn count(bytes: &[u8], enc: Encoding) -> Result<u64> {
    match histogram(bytes, enc) {
        Some(hist) => Ok(estimate(&hist)),
        None => Err(corrupt()),
    }
}

/// The sparse opcodes written out, which is what `PFDEBUG DECODE` answers.
///
/// The spelling is a real server's, checked against 8.10.1 on a sketch built by
/// hand to have one of each: lowercase `z` for a `ZERO` run, uppercase `Z` for
/// an `XZERO` one, and `v` for a value and how many times it repeats, the three
/// separated by single spaces. It goes into a byte buffer rather than a `String`
/// so that the caller can hand it the one it already has.
pub fn decode(bytes: &[u8], out: &mut Vec<u8>) {
    use std::io::Write;
    let sparse = &bytes[HDR..];
    let mut at = 0;
    while at < sparse.len() {
        if !out.is_empty() {
            out.push(b' ');
        }
        let b = sparse[at];
        if is_val(b) {
            let _ = write!(out, "v:{},{}", val_value(b), val_len(b));
            at += 1;
        } else if is_zero(b) {
            let _ = write!(out, "z:{}", zero_len(b));
            at += 1;
        } else {
            let Some(&next) = sparse.get(at + 1) else {
                return;
            };
            let _ = write!(out, "Z:{}", xzero_len(b, next));
            at += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hash is the file format, so it is pinned against a real server.
    ///
    /// These three registers were read out of a running 8.10.1 with `PFDEBUG
    /// GETREG` after `PFADD h a b c`, and they are the whole reason this file
    /// carries a hash function nothing else in the tree uses.
    #[test]
    fn an_element_lands_where_a_real_server_puts_it() {
        assert_eq!(place(b"a"), (12711, 2));
        assert_eq!(place(b"b"), (15780, 1));
        assert_eq!(place(b"c"), (8436, 1));
    }

    /// The bytes of an empty sketch, and of one with three elements in it.
    ///
    /// Both were read off a real server with `GET`. Both have the cache marked
    /// stale, and on the empty one that is not this function's doing: Redis
    /// creates the sketch with a valid cache of zero and `PFADD` invalidates it
    /// on the way out, even when it added nothing. The bytes a client can see
    /// are the ones that matter and they have the top bit set.
    #[test]
    fn a_sketch_is_the_bytes_a_real_server_writes() {
        let mut buf = Vec::new();
        empty(&mut buf);
        assert_eq!(buf.len(), 18);
        assert_eq!(&buf[..], b"HYLL\x01\0\0\0\0\0\0\0\0\0\0\0\x7f\xff");
        invalidate(&mut buf);
        assert_eq!(&buf[..], b"HYLL\x01\0\0\0\0\0\0\0\0\0\0\x80\x7f\xff");

        for ele in [&b"a"[..], b"b", b"c"] {
            let (index, val) = place(ele);
            assert_eq!(set(&mut buf, index, val), Some(true));
        }
        assert_eq!(
            &buf[..],
            b"HYLL\x01\0\0\0\0\0\0\0\0\0\0\x80\x60\xf3\x80\x50\xb1\x84\x4b\xfb\x80\x42\x5a"
        );
    }

    /// A second write of the same element changes nothing at all.
    #[test]
    fn writing_the_same_element_twice_is_not_a_change() {
        let mut buf = Vec::new();
        empty(&mut buf);
        let (index, val) = place(b"a");
        assert_eq!(set(&mut buf, index, val), Some(true));
        let before = buf.clone();
        assert_eq!(set(&mut buf, index, val), Some(false));
        assert_eq!(buf, before);
    }

    /// The dense packing, against the obvious slow version of itself.
    #[test]
    fn a_dense_register_is_six_bits_packed_from_the_bottom() {
        let mut regs = vec![0u8; REGISTERS * BITS / 8];
        let mut want = vec![0u8; REGISTERS];
        for (i, slot) in want.iter_mut().enumerate() {
            *slot = ((i * 7 + 1) % 52) as u8;
        }
        // Written in a shuffled order so that a write that spilled into its
        // neighbour would be caught rather than overwritten afterwards.
        for step in [1usize, 3, 5] {
            let mut i = 0;
            while i < REGISTERS {
                let val = want[i];
                if val > dense_get(&regs, i) {
                    assert!(dense_set(&mut regs, i, val));
                }
                i += step;
            }
        }
        for (i, &val) in want.iter().enumerate() {
            assert_eq!(dense_get(&regs, i), val, "register {i}");
        }
        // A write that would lower a register is refused, the way a sketch needs.
        assert!(!dense_set(&mut regs, 5, 0));
    }

    /// A sparse sketch and the dense one it turns into hold the same registers.
    #[test]
    fn turning_dense_keeps_every_register() {
        let mut buf = Vec::new();
        empty(&mut buf);
        let mut want = [0u8; REGISTERS];
        for i in 0..400 {
            let ele = format!("e:{i}");
            let (index, val) = place(ele.as_bytes());
            set(&mut buf, index, val).expect("a write");
            want[index] = want[index].max(val);
        }
        let sparse = count(&buf, Encoding::Sparse).expect("a count");

        assert!(to_dense(&mut buf));
        assert_eq!(buf.len(), DENSE);
        assert_eq!(check(&buf).expect("a sketch"), Encoding::Dense);
        for (i, &val) in want.iter().enumerate() {
            assert_eq!(dense_get(&buf[HDR..], i), val, "register {i}");
        }
        assert_eq!(count(&buf, Encoding::Dense).expect("a count"), sparse);
    }

    /// Enough elements to push a sketch over the sparse limit on its own.
    #[test]
    fn a_sketch_turns_dense_when_it_outgrows_the_sparse_form() {
        let mut buf = Vec::new();
        empty(&mut buf);
        for i in 0..2000 {
            let ele = format!("e:{i}");
            let (index, val) = place(ele.as_bytes());
            set(&mut buf, index, val).expect("a write");
            assert!(buf.len() <= SPARSE_MAX || buf.len() == DENSE);
        }
        assert_eq!(check(&buf).expect("a sketch"), Encoding::Dense);
    }

    /// A value larger than a `VAL` opcode can hold turns the sketch dense.
    #[test]
    fn a_large_register_turns_the_sketch_dense() {
        let mut buf = Vec::new();
        empty(&mut buf);
        assert_eq!(set(&mut buf, 100, VAL_MAX), Some(true));
        assert_eq!(check(&buf).expect("a sketch"), Encoding::Sparse);
        assert_eq!(set(&mut buf, 200, VAL_MAX + 1), Some(true));
        assert_eq!(check(&buf).expect("a sketch"), Encoding::Dense);
        assert_eq!(dense_get(&buf[HDR..], 100), VAL_MAX);
        assert_eq!(dense_get(&buf[HDR..], 200), VAL_MAX + 1);
    }

    /// Neighbouring runs of the same value are joined back up.
    ///
    /// Without the merge pass a sketch grows an opcode for every register
    /// written, and four ones in a row here would be four bytes instead of one.
    #[test]
    fn neighbouring_runs_of_the_same_value_are_joined() {
        let mut buf = Vec::new();
        empty(&mut buf);
        for i in 0..4 {
            set(&mut buf, 100 + i, 1).expect("a write");
        }
        let mut decoded = Vec::new();
        decode(&buf, &mut decoded);
        assert_eq!(decoded, b"Z:100 v:1,4 Z:16280");
    }

    /// One sketch with all three opcodes in it, against a real server.
    ///
    /// The bytes were written by hand, `SET` into 8.10.1, and read back through
    /// `PFDEBUG DECODE` and `PFCOUNT`. It is the only case that pins the
    /// lowercase `z`, since a sketch built by adding elements rarely has a gap
    /// short enough to need one.
    #[test]
    fn all_three_opcodes_decode_the_way_a_real_server_prints_them() {
        let mut buf = Vec::new();
        empty(&mut buf);
        buf.truncate(HDR);
        buf.extend_from_slice(&xzero_bytes(100));
        buf.push(val_byte(1, 4));
        buf.push(zero_byte(10));
        buf.push(val_byte(3, 2));
        buf.extend_from_slice(&xzero_bytes(REGISTERS - 100 - 4 - 10 - 2));

        let mut decoded = Vec::new();
        decode(&buf, &mut decoded);
        assert_eq!(decoded, b"Z:100 v:1,4 z:10 v:3,2 Z:16268");
        assert_eq!(count(&buf, Encoding::Sparse).expect("a count"), 6);
    }

    /// The counted answer against the sketch itself, over a range of sizes.
    ///
    /// A HyperLogLog is allowed to be wrong and this checks it is wrong by less
    /// than the two percent the parameters promise, which is what would catch a
    /// register being written in the wrong place.
    #[test]
    fn the_estimate_is_close_to_the_truth() {
        for n in [10usize, 100, 1000, 10_000, 100_000] {
            let mut buf = Vec::new();
            empty(&mut buf);
            for i in 0..n {
                let ele = format!("element:{i}");
                let (index, val) = place(ele.as_bytes());
                set(&mut buf, index, val).expect("a write");
            }
            let enc = check(&buf).expect("a sketch");
            let got = count(&buf, enc).expect("a count") as f64;
            let off = (got - n as f64).abs() / n as f64;
            assert!(off < 0.02, "{n} counted as {got}");
        }
    }

    /// The numbers a real 8.10.1 answered for the same elements.
    ///
    /// A sketch that counted correctly and not identically would be useless for
    /// the thing this is for, which is a client moving sketches between servers.
    #[test]
    fn the_estimate_is_the_number_a_real_server_gives() {
        for (n, want) in [(100usize, 100u64), (1000, 995), (10_000, 10_077)] {
            let mut buf = Vec::new();
            empty(&mut buf);
            for i in 0..n {
                let ele = format!("e:{i}");
                let (index, val) = place(ele.as_bytes());
                set(&mut buf, index, val).expect("a write");
            }
            let enc = check(&buf).expect("a sketch");
            assert_eq!(count(&buf, enc).expect("a count"), want, "{n} elements");
        }
    }

    /// The two sizes the milestone gate names, on the elements that reach them.
    #[test]
    fn the_two_sizes_are_the_ones_a_real_server_has() {
        let build = |n: usize| {
            let mut buf = Vec::new();
            empty(&mut buf);
            for i in 0..n {
                let ele = format!("e:{i}");
                let (index, val) = place(ele.as_bytes());
                set(&mut buf, index, val).expect("a write");
            }
            buf
        };
        assert_eq!(build(1000).len(), 1880);
        assert_eq!(build(10_000).len(), DENSE);
        assert_eq!(DENSE, 12304);
        const { assert!(1880 <= SPARSE_MAX) };
    }

    /// What is refused, and what a stale cache looks like.
    #[test]
    fn a_string_that_is_not_a_sketch_is_refused() {
        assert!(check(b"").is_err());
        assert!(check(b"HYLL").is_err());
        assert!(check(b"NOPE\x01\0\0\0\0\0\0\0\0\0\0\0\x7f\xff").is_err());
        assert!(check(b"HYLL\x02\0\0\0\0\0\0\0\0\0\0\0\x7f\xff").is_err());
        // A dense sketch has to be exactly the right length.
        assert!(check(b"HYLL\0\0\0\0\0\0\0\0\0\0\0\0\x7f\xff").is_err());

        let mut buf = Vec::new();
        empty(&mut buf);
        assert_eq!(cached(&buf), Some(0));
        cache(&mut buf, 12345);
        assert_eq!(cached(&buf), Some(12345));
        invalidate(&mut buf);
        assert_eq!(cached(&buf), None);
    }

    /// Runs that do not cover all 16384 registers are a corrupted sketch.
    #[test]
    fn a_body_that_does_not_add_up_is_corrupt() {
        let mut buf = Vec::new();
        empty(&mut buf);
        buf.truncate(buf.len() - 1);
        assert!(count(&buf, Encoding::Sparse).is_err());
        let mut short = Vec::new();
        empty(&mut short);
        short.pop();
        short.pop();
        assert!(count(&short, Encoding::Sparse).is_err());
    }

    /// Merging takes the larger of each register, whichever form it is in.
    #[test]
    fn merging_takes_the_larger_of_every_register() {
        let build = |from: usize, to: usize| {
            let mut buf = Vec::new();
            empty(&mut buf);
            for i in from..to {
                let ele = format!("e:{i}");
                let (index, val) = place(ele.as_bytes());
                set(&mut buf, index, val).expect("a write");
            }
            buf
        };
        let a = build(0, 500);
        let b = build(400, 900);
        let mut max = [0u8; REGISTERS];
        assert!(merge(&mut max, &a, Encoding::Sparse));
        assert!(merge(&mut max, &b, Encoding::Sparse));

        let both = build(0, 900);
        let mut want = [0u8; REGISTERS];
        assert!(merge(&mut want, &both, check(&both).expect("a sketch")));
        assert_eq!(max, want);
    }
}
