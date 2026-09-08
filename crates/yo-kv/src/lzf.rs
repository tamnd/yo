//! LZF, the one compression Redis puts inside an RDB payload.
//!
//! # The format
//!
//! Three shapes of control byte, and that is the whole thing:
//!
//! ```text
//! 000LLLLL <L+1 bytes>       a run of 1 to 32 literal bytes
//! LLLooooo oooooooo          a back reference, L+2 bytes, 1 to 8192 back
//! 111ooooo LLLLLLLL oooooooo a back reference, L+9 bytes, 1 to 8192 back
//! ```
//!
//! A back reference is allowed to overlap what it is producing, which is how a
//! long run of one byte compresses down to four, so the copy has to go a byte at
//! a time rather than through a slice copy.
//!
//! # Why the compressor is a transcription
//!
//! [`pack`] is liblzf 3.6's `lzf_compress` written out in Rust, with the same
//! constants Redis builds it with: a sixty five thousand slot table, the
//! `VERY_FAST` hash and skip rules, and no clearing of the table between runs.
//! It is a transcription and not an implementation, and that is on purpose.
//!
//! LZF is not a canonical format. The same input has many valid compressions and
//! which one you get is decided entirely by the search, so an honest reimplement
//! that picked better matches would produce a payload a real Redis reads back
//! perfectly and that is a different length and a different sequence of bytes.
//! `DUMP` output being byte identical to Redis's is a thing this project claims,
//! and once a string is compressed the only way to keep that claim is to make
//! the same choices in the same order. So the search here is theirs, down to the
//! unrolled sixteen byte compare that can run past the length it was given and
//! the two positions rehashed after a match rather than all of them.
//!
//! The one place this deliberately differs is the table. Redis leaves it
//! uninitialised, so a run can find a match through a slot left by the run
//! before it, which is safe, because every candidate is checked against the
//! input again before it is used, and not reproducible, because it depends on
//! what the stack happened to hold. This clears the slots each run touched
//! before the next run starts, so the same input always compresses to the same
//! bytes. That is a difference in the direction of being more deterministic than
//! the reference, and the only way to see it is a string whose compression
//! Redis improved by accident.
//!
//! # Why the table is kept rather than made
//!
//! The table is a quarter of a megabyte and every `DUMP` of a string over twenty
//! bytes wants one. Redis puts it on the stack and pays half a megabyte of stack
//! per call, which is a thing you can do in C and would be an odd thing to do
//! here. So it is one per thread, made on first use and kept, and what gets
//! cleared between runs is the list of slots the last run wrote rather than the
//! whole table. That list is at most as long as the input, so clearing it is
//! part of the same linear pass the compression already is.

use std::cell::RefCell;

/// The number of slots in the match table, which is Redis's `HLOG` of sixteen.
const SLOTS: usize = 1 << 16;

/// The longest run of literal bytes one control byte can introduce.
const MAX_LIT: usize = 1 << 5;

/// The furthest back a reference can point.
const MAX_OFF: usize = 1 << 13;

/// The longest run a reference can copy.
const MAX_REF: usize = (1 << 8) + (1 << 3);

/// One thread's match table, and the slots it last wrote.
struct Table {
    /// Where each three byte sequence was last seen, or nought for never.
    ///
    /// Nought doubles as the empty marker and as position nought, which loses
    /// nothing: a match at position nought is refused anyway, because the
    /// reference has to be strictly past the start of the input.
    slots: Box<[u32]>,
    /// Which slots the last run wrote, so the next run can undo just those.
    touched: Vec<u32>,
}

thread_local! {
    static TABLE: RefCell<Table> = RefCell::new(Table {
        slots: vec![0u32; SLOTS].into_boxed_slice(),
        touched: Vec::new(),
    });
}

/// The hash of a three byte sequence, which is liblzf's `IDX` of `NEXT`.
///
/// Everything here is a wrapping thirty two bit multiply and shift, because that
/// is what it is in C and the table it indexes is a shared format decision even
/// though the format does not mention it.
#[inline]
const fn slot(hval: u32) -> usize {
    ((hval >> 8).wrapping_sub(hval.wrapping_mul(5)) & (SLOTS as u32 - 1)) as usize
}

/// Compress `data`, or `None` when the result would not be smaller.
///
/// The output buffer is four bytes shorter than the input, which is Redis's
/// bound and not an arbitrary one: the header that says a string is compressed
/// costs at least four bytes over the header that says it is not, so a
/// compression that only just fits is not worth storing.
pub(crate) fn pack(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() <= 4 {
        return None;
    }
    TABLE.with(|table| {
        let mut table = table.borrow_mut();
        // Undone at the start rather than at the end, so a run that stops early
        // for any reason still leaves a clean table behind it.
        let Table { slots, touched } = &mut *table;
        for &at in touched.iter() {
            slots[at as usize] = 0;
        }
        touched.clear();
        compress(data, slots, touched)
    })
}

/// The search itself, with the table handed in.
///
/// Split out from [`pack`] so that the borrow of the thread local ends whatever
/// this returns, and so that a test can hand it a table of its own.
#[allow(clippy::needless_range_loop)]
fn compress(data: &[u8], slots: &mut [u32], touched: &mut Vec<u32>) -> Option<Vec<u8>> {
    let n = data.len();
    let room = n - 4;
    let mut out = vec![0u8; room];
    // How many bytes of `out` are spoken for. The byte at this position is often
    // one that has not been written yet, because a run of literals reserves its
    // control byte when it starts and fills it in when it stops.
    let mut op = 0usize;
    // How many literals the run in progress has, where nought means the control
    // byte is reserved and nothing has gone in after it.
    let mut lit = 0usize;
    let mut ip = 0usize;

    // Start the first run.
    op += 1;

    let mut hval = (u32::from(data[0]) << 8) | u32::from(data[1]);
    while ip + 2 < n {
        hval = (hval << 8) | u32::from(data[ip + 2]);
        let at = slot(hval);
        let reference = slots[at] as usize;
        if slots[at] == 0 {
            touched.push(at as u32);
        }
        slots[at] = ip as u32;

        // A slot holding nought is either empty or position nought, and the
        // second test rules out both. The offset is computed the way C computes
        // it, so a reference ahead of here wraps to something enormous and fails
        // the bound rather than needing a test of its own.
        let off = ip.wrapping_sub(reference).wrapping_sub(1);
        if off < MAX_OFF
            && reference > 0
            && data[reference + 2] == data[ip + 2]
            && data[reference] == data[ip]
            && data[reference + 1] == data[ip + 1]
        {
            let mut len = 2usize;
            let maxlen = (n - ip - len).min(MAX_REF);

            // The conservative test first and the exact one only when it fails,
            // which is liblzf's, and the exact one is what decides.
            if op + 3 + 1 >= room && op - usize::from(lit == 0) + 3 + 1 >= room {
                return None;
            }

            // Stop the run in progress, and take its control byte back if it
            // never got a literal.
            out[op - lit - 1] = (lit as u8).wrapping_sub(1);
            op -= usize::from(lit == 0);

            // The sixteen byte unrolled compare, which can leave `len` past
            // `maxlen` by up to one. That is liblzf and it is safe: reaching the
            // end of the unrolled block means at least eighteen bytes were
            // there to read, and the reference that comes out of it stops
            // exactly at the end of the input.
            let mut stopped = false;
            if maxlen > 16 {
                for _ in 0..16 {
                    len += 1;
                    if data[reference + len] != data[ip + len] {
                        stopped = true;
                        break;
                    }
                }
            }
            if !stopped {
                loop {
                    len += 1;
                    if !(len < maxlen && data[reference + len] == data[ip + len]) {
                        break;
                    }
                }
            }

            // `len` is now the number of bytes the reference copies, less two.
            len -= 2;
            ip += 1;
            if len < 7 {
                out[op] = ((off >> 8) as u8) + ((len as u8) << 5);
                op += 1;
            } else {
                out[op] = ((off >> 8) as u8) + (7 << 5);
                op += 1;
                out[op] = (len - 7) as u8;
                op += 1;
            }
            out[op] = off as u8;
            op += 1;

            lit = 0;
            op += 1;

            ip += len + 1;
            if ip + 2 >= n {
                break;
            }

            // Two positions inside the match are hashed and the rest are not,
            // which is the `VERY_FAST` rule. Skipping them is why liblzf is
            // fast and is also why it misses matches a slower search would find,
            // and both halves of that are part of the output being what it is.
            ip -= 2;
            hval = (u32::from(data[ip]) << 8) | u32::from(data[ip + 1]);
            for _ in 0..2 {
                hval = (hval << 8) | u32::from(data[ip + 2]);
                let at = slot(hval);
                if slots[at] == 0 {
                    touched.push(at as u32);
                }
                slots[at] = ip as u32;
                ip += 1;
            }
        } else {
            if op >= room {
                return None;
            }
            lit += 1;
            out[op] = data[ip];
            op += 1;
            ip += 1;
            if lit == MAX_LIT {
                out[op - lit - 1] = (lit - 1) as u8;
                lit = 0;
                op += 1;
            }
        }
    }

    // At most two bytes are left and at most one control byte goes with them.
    if op + 3 > room {
        return None;
    }
    while ip < n {
        lit += 1;
        out[op] = data[ip];
        op += 1;
        ip += 1;
        if lit == MAX_LIT {
            out[op - lit - 1] = (lit - 1) as u8;
            lit = 0;
            op += 1;
        }
    }
    out[op - lit - 1] = (lit as u8).wrapping_sub(1);
    op -= usize::from(lit == 0);

    out.truncate(op);
    Some(out)
}

/// Undo [`pack`], or `None` for a payload that does not describe `plain` bytes.
///
/// `plain` is the length the payload claims the result will be, and it is used
/// as the bound rather than trusted, so a payload claiming four bytes and
/// describing four gigabytes stops at four.
pub(crate) fn unpack(packed: &[u8], plain: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(plain.min(1 << 20));
    let mut i = 0;
    while i < packed.len() {
        let ctrl = usize::from(packed[i]);
        i += 1;
        if ctrl < 32 {
            let run = ctrl + 1;
            let end = i.checked_add(run)?;
            if end > packed.len() || out.len() + run > plain {
                return None;
            }
            out.extend_from_slice(&packed[i..end]);
            i = end;
        } else {
            let mut run = ctrl >> 5;
            if run == 7 {
                run += usize::from(*packed.get(i)?);
                i += 1;
            }
            let back = ((ctrl & 0x1f) << 8) + usize::from(*packed.get(i)?) + 1;
            i += 1;
            let run = run + 2;
            if back > out.len() || out.len() + run > plain {
                return None;
            }
            let from = out.len() - back;
            for at in from..from + run {
                out.push(out[at]);
            }
        }
    }
    (out.len() == plain).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::{pack, unpack};

    #[test]
    fn unpacks_a_literal_run() {
        // One control byte saying four literals, then the four.
        assert_eq!(
            unpack(&[3, b'a', b'b', b'c', b'd'], 4).as_deref(),
            Some(&b"abcd"[..])
        );
    }

    /// The case the byte at a time copy exists for: a back reference that reads
    /// bytes it is in the middle of writing.
    #[test]
    fn unpacks_an_overlapping_reference() {
        // One literal `a`, then a reference one byte back for five bytes. The
        // low five bits of the control byte and the byte after it are the
        // distance, and they are both zero because a distance is stored one
        // less than it is.
        let packed = [0u8, b'a', 3 << 5, 0];
        assert_eq!(unpack(&packed, 6).as_deref(), Some(&b"aaaaaa"[..]));
    }

    #[test]
    fn refuses_a_reference_to_nothing() {
        assert_eq!(unpack(&[(3 << 5), 0], 5), None);
        assert_eq!(unpack(&[3, b'a'], 4), None);
    }

    /// The bytes a real liblzf produces for a run of one byte.
    ///
    /// These came off liblzf 3.6 compiled out of the Redis tree and not off the
    /// format notes, and two of the three pieces are things the notes would not
    /// have told you. Two literals go out at the front rather than one, because
    /// a reference has to point strictly past the start of the input, so the
    /// three bytes hashed at position nought can never be matched against. Two
    /// go out at the back because the search stops three bytes from the end and
    /// the tail is copied as literals whatever it holds.
    #[test]
    fn packs_a_long_run_the_way_liblzf_does() {
        let packed = pack(&[b'a'; 200]).expect("a run of one byte compresses");
        assert_eq!(packed, [1, b'a', b'a', 0xe0, 187, 0, 1, b'a', b'a']);
        assert_eq!(unpack(&packed, 200).as_deref(), Some(&[b'a'; 200][..]));
    }

    /// Nothing repeats, so there is nothing to point back at.
    #[test]
    fn refuses_what_does_not_compress() {
        let mut data = Vec::new();
        for n in 0..=255u8 {
            data.push(n);
        }
        assert_eq!(pack(&data), None);
        // And the short cases, which never get as far as the search.
        assert_eq!(pack(b""), None);
        assert_eq!(pack(b"abcd"), None);
    }

    /// Every length up to a few hundred, of a shape that does compress.
    ///
    /// The point is the boundaries, which are where a transcription goes wrong:
    /// the run of thirty two literals that fills a control byte, the two byte
    /// tail the main loop never reaches, and the reference that lands exactly on
    /// the end of the input.
    #[test]
    fn every_length_comes_back_the_way_it_went_in() {
        let alphabet = b"abcdefgh";
        for n in 0..400usize {
            let mut data = Vec::with_capacity(n);
            for i in 0..n {
                data.push(alphabet[(i / 3) % alphabet.len()]);
            }
            if let Some(packed) = pack(&data) {
                assert!(packed.len() < data.len(), "{n} did not get smaller");
                assert_eq!(unpack(&packed, n).as_deref(), Some(&data[..]), "{n}");
            }
        }
    }

    /// The same input compresses to the same bytes however often it is asked.
    ///
    /// This is the one place this deliberately parts company with liblzf, which
    /// leaves the table alone between runs and so can compress the same string
    /// two ways depending on what came before it.
    #[test]
    fn the_same_input_packs_the_same_way_twice() {
        let one = vec![b'x'; 100];
        let two = b"the quick brown fox jumps over the lazy dog, the quick brown fox".to_vec();
        let first = pack(&one);
        pack(&two);
        let again = pack(&one);
        assert_eq!(first, again);
    }

    /// A mix of runs and noise, which is what a real value looks like.
    #[test]
    fn a_mixed_value_round_trips() {
        let mut data = Vec::new();
        for n in 0..500u32 {
            data.extend_from_slice(b"field");
            data.extend_from_slice(n.to_string().as_bytes());
            data.extend_from_slice(b":value:");
            data.extend_from_slice(&[b'z'; 7]);
        }
        let packed = pack(&data).expect("this compresses");
        assert!(packed.len() * 4 < data.len());
        assert_eq!(unpack(&packed, data.len()).as_deref(), Some(&data[..]));
    }
}
