//! Eight slots looked at in one go, which is what makes a full table cheap.
//!
//! The element table probes an open addressed slot array, and until now it did
//! that one slot at a time: a load, a compare against the tag, a branch, and
//! round again. The cost of that grows with how full the array is, and so does
//! the memory, because slots per element is one over the load factor. They are
//! the same number seen twice, which is why [`crate::elem`] could not be made
//! smaller without being made slower, and #175 is that sentence measured.
//!
//! A group probe breaks the tie. Eight slots go into two vector registers, both
//! are compared against the wanted tag at once, and the answer comes back as an
//! eight bit mask with one bit per slot. One load, a handful of arithmetic and
//! one branch, for eight slots instead of one.
//!
//! ```text
//!   slots  | 3f..  | 0f..  | a7..  | 3f..  | ffff  | 91..  | 3f..  | 02..  |
//!   tag 3f |   1   |   0   |   0   |   1   |   0   |   0   |   1   |   0   | -> 0b0100_1001
//!   empty  |   0   |   0   |   0   |   0   |   1   |   0   |   0   |   0   | -> 0b0001_0000
//! ```
//!
//! The stop condition falls out of the same work. A probe stops at the first
//! group with an empty slot in it, so the empty mask is both the answer to "is
//! it here" and the answer to "is there anywhere left to look". [`scan`] hands
//! back both off one pass, because the slots are already in registers by the
//! time the first one has been worked out and going round again for the second
//! is thirty two bytes read twice for nothing.
//!
//! How full the array is allowed to get is [`crate::elem`]'s decision and not
//! this module's, but the number a group probe cares about is not the load, it
//! is the chance that a group has an empty slot in it. Eight wide, three
//! quarters full, that is nine in ten and a miss is 1.1 groups. Seven eighths
//! full it is six in ten and a miss is over one and a half, which is why the
//! load this ships with is the first number and not the second.
//!
//! # Why eight and why aligned
//!
//! Eight `u32` slots are thirty two bytes, so a group that starts on a multiple
//! of eight is always inside one cache line and a probe never straddles two. It
//! is also two 128 bit registers, which is what both architectures this builds
//! for have without asking: SSE2 is part of x86-64 and NEON is part of aarch64,
//! so there is no runtime detection here and no build flag, and the scalar
//! fallback is a loop over the same eight slots that any other target compiles.
//!
//! Sixteen slots would be one group at almost any load, and it was not chosen
//! because sixteen `u32` is a whole cache line rather than half of one, so the
//! group a probe wants is twice as likely to be a line it does not have. That
//! is a trade to revisit with a measurement rather than a guess.
//!
//! # What a mask means
//!
//! Bit `i` is slot `i` of the group, counting from the one the group starts at.
//! Every function here returns bits and reads nothing but the eight slots it was
//! given, so the caller decides what a match is worth and this decides nothing.

/// How many slots one group covers.
///
/// A power of two, and the slot array is a whole number of these and never
/// fewer than one, so every group is aligned and there is no partial group at
/// the end.
pub const WIDTH: usize = 8;

/// The slots of one group, always [`WIDTH`] of them.
pub type Slots<'a> = &'a [u32];

/// Which slots carry `tag` in their top byte.
///
/// A hit here is not a hit in the table. The tag is eight bits, so one match in
/// two hundred and fifty six is a different name that happens to share it, and
/// the caller still has to read the row and compare the name. What it buys is
/// that the read only happens on a match rather than on every slot.
#[inline(always)]
#[must_use]
pub fn tags(slots: Slots<'_>, tag: u8) -> u8 {
    debug_assert_eq!(slots.len(), WIDTH);
    imp::eq_top(slots, tag)
}

/// The tag mask and the empty mask together, off one pass over the group.
///
/// A probe wants both of them every time round: which slots might be the name it
/// is looking for, and whether it is allowed to stop here. Asked for one at a
/// time they are two passes over the same thirty two bytes, and the second one
/// is pure waste because the first already had the slots in registers.
///
/// Answered in that order, tags first and empties second, which is the order the
/// caller uses them in.
#[inline(always)]
#[must_use]
pub fn scan(slots: Slots<'_>, tag: u8, empty: u32) -> (u8, u8) {
    debug_assert_eq!(slots.len(), WIDTH);
    imp::scan(slots, tag, empty)
}

/// Which slots have never been written to.
///
/// A probe stops at the first group where this is not zero, because an empty
/// slot means nothing ever probed past it.
#[inline(always)]
#[must_use]
pub fn empty(slots: Slots<'_>, empty: u32) -> u8 {
    debug_assert_eq!(slots.len(), WIDTH);
    imp::eq_all(slots, empty)
}

/// Which slots an insert may take, which is the empty ones and the markers.
///
/// Both markers have all the row bits set and a live row never does, so this is
/// one compare of the masked slot against the mask, and it does not need to know
/// which kind of marker it found.
#[inline(always)]
#[must_use]
pub fn free(slots: Slots<'_>, row_bits: u32) -> u8 {
    debug_assert_eq!(slots.len(), WIDTH);
    imp::eq_masked(slots, row_bits, row_bits)
}

/// Which slots point at row `want`.
///
/// Used where the row index is known and the name is not, which is a removal and
/// a growth. A marker cannot match, because its row bits are all ones and no row
/// index is.
#[inline(always)]
#[must_use]
pub fn rows(slots: Slots<'_>, row_bits: u32, want: u32) -> u8 {
    debug_assert_eq!(slots.len(), WIDTH);
    imp::eq_masked(slots, row_bits, want)
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_loadu_si128,
        _mm_movemask_ps, _mm_set1_epi32,
    };

    /// The four lane compare both halves of a group go through.
    ///
    /// `_mm_movemask_ps` takes the sign bit of each of the four lanes, and a
    /// lane of a `_mm_cmpeq_epi32` result is all ones or all zeros, so the sign
    /// bit is the answer.
    #[inline(always)]
    unsafe fn half(slots: &[u32], at: usize, and: __m128i, want: __m128i) -> u8 {
        // SAFETY: the caller has checked the slice is `WIDTH` long and `at` is
        // 0 or 4, so four `u32` from `at` are in bounds. The load is unaligned
        // by name and the slice's own alignment is enough for it either way.
        unsafe {
            let v = _mm_loadu_si128(slots.as_ptr().add(at).cast::<__m128i>());
            let hit = _mm_cmpeq_epi32(_mm_and_si128(v, and), want);
            _mm_movemask_ps(_mm_castsi128_ps(hit)) as u8
        }
    }

    #[inline(always)]
    fn both(slots: &[u32], and: u32, want: u32) -> u8 {
        // SAFETY: SSE2 is part of the x86-64 baseline, so these are always
        // available on this target and need no runtime check.
        unsafe {
            let and = _mm_set1_epi32(and as i32);
            let want = _mm_set1_epi32(want as i32);
            half(slots, 0, and, want) | (half(slots, 4, and, want) << 4)
        }
    }

    /// Both masks off one load of each half.
    #[inline(always)]
    unsafe fn pair(
        slots: &[u32],
        at: usize,
        top: __m128i,
        want: __m128i,
        none: __m128i,
    ) -> (u8, u8) {
        // SAFETY: as `half`, and the two compares read the same loaded lanes.
        unsafe {
            let v = _mm_loadu_si128(slots.as_ptr().add(at).cast::<__m128i>());
            let t = _mm_cmpeq_epi32(_mm_and_si128(v, top), want);
            let e = _mm_cmpeq_epi32(v, none);
            (
                _mm_movemask_ps(_mm_castsi128_ps(t)) as u8,
                _mm_movemask_ps(_mm_castsi128_ps(e)) as u8,
            )
        }
    }

    #[inline(always)]
    pub fn scan(slots: &[u32], tag: u8, empty: u32) -> (u8, u8) {
        // SAFETY: SSE2 is part of the x86-64 baseline.
        unsafe {
            let top = _mm_set1_epi32(0xFF00_0000u32 as i32);
            let want = _mm_set1_epi32((u32::from(tag) << 24) as i32);
            let none = _mm_set1_epi32(empty as i32);
            let (t0, e0) = pair(slots, 0, top, want, none);
            let (t1, e1) = pair(slots, 4, top, want, none);
            (t0 | (t1 << 4), e0 | (e1 << 4))
        }
    }

    #[inline(always)]
    pub fn eq_top(slots: &[u32], tag: u8) -> u8 {
        both(slots, 0xFF00_0000, u32::from(tag) << 24)
    }

    #[inline(always)]
    pub fn eq_all(slots: &[u32], want: u32) -> u8 {
        both(slots, u32::MAX, want)
    }

    #[inline(always)]
    pub fn eq_masked(slots: &[u32], and: u32, want: u32) -> u8 {
        both(slots, and, want)
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::{
        uint32x4_t, vaddvq_u32, vandq_u32, vceqq_u32, vdupq_n_u32, vld1q_u32,
    };

    /// One bit per lane, which NEON has no single instruction for.
    ///
    /// A compare leaves each lane all ones or all zeros, so anding with 1, 2, 4,
    /// 8 and adding the four lanes across gives the four bit mask. That is the
    /// usual way to do this on aarch64 and it is three instructions.
    #[inline(always)]
    unsafe fn half(slots: &[u32], at: usize, and: uint32x4_t, want: uint32x4_t) -> u8 {
        const BITS: [u32; 4] = [1, 2, 4, 8];
        // SAFETY: the caller has checked the slice is `WIDTH` long and `at` is
        // 0 or 4, so four `u32` from `at` are in bounds.
        unsafe {
            let v = vld1q_u32(slots.as_ptr().add(at));
            let hit = vceqq_u32(vandq_u32(v, and), want);
            vaddvq_u32(vandq_u32(hit, vld1q_u32(BITS.as_ptr()))) as u8
        }
    }

    #[inline(always)]
    fn both(slots: &[u32], and: u32, want: u32) -> u8 {
        // SAFETY: NEON is part of the aarch64 baseline, so these are always
        // available on this target and need no runtime check.
        unsafe {
            let and = vdupq_n_u32(and);
            let want = vdupq_n_u32(want);
            half(slots, 0, and, want) | (half(slots, 4, and, want) << 4)
        }
    }

    /// Both masks off one load of each half.
    #[inline(always)]
    unsafe fn pair(
        slots: &[u32],
        at: usize,
        top: uint32x4_t,
        want: uint32x4_t,
        none: uint32x4_t,
    ) -> (u8, u8) {
        const BITS: [u32; 4] = [1, 2, 4, 8];
        // SAFETY: as `half`, and the two compares read the same loaded lanes.
        unsafe {
            let bits = vld1q_u32(BITS.as_ptr());
            let v = vld1q_u32(slots.as_ptr().add(at));
            let t = vceqq_u32(vandq_u32(v, top), want);
            let e = vceqq_u32(v, none);
            (
                vaddvq_u32(vandq_u32(t, bits)) as u8,
                vaddvq_u32(vandq_u32(e, bits)) as u8,
            )
        }
    }

    #[inline(always)]
    pub fn scan(slots: &[u32], tag: u8, empty: u32) -> (u8, u8) {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let top = vdupq_n_u32(0xFF00_0000);
            let want = vdupq_n_u32(u32::from(tag) << 24);
            let none = vdupq_n_u32(empty);
            let (t0, e0) = pair(slots, 0, top, want, none);
            let (t1, e1) = pair(slots, 4, top, want, none);
            (t0 | (t1 << 4), e0 | (e1 << 4))
        }
    }

    #[inline(always)]
    pub fn eq_top(slots: &[u32], tag: u8) -> u8 {
        both(slots, 0xFF00_0000, u32::from(tag) << 24)
    }

    #[inline(always)]
    pub fn eq_all(slots: &[u32], want: u32) -> u8 {
        both(slots, u32::MAX, want)
    }

    #[inline(always)]
    pub fn eq_masked(slots: &[u32], and: u32, want: u32) -> u8 {
        both(slots, and, want)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod imp {
    /// The same answer without vectors, for a target that has none we can name.
    ///
    /// It is a fixed count loop over eight slots with no branch in the body, so
    /// a compiler that wants to vectorise it can, and one that does not still
    /// produces something no worse than the slot at a time probe this replaced.
    #[inline(always)]
    fn both(slots: &[u32], and: u32, want: u32) -> u8 {
        let mut bits = 0u8;
        for (i, &slot) in slots.iter().enumerate() {
            bits |= u8::from(slot & and == want) << i;
        }
        bits
    }

    #[inline(always)]
    pub fn scan(slots: &[u32], tag: u8, empty: u32) -> (u8, u8) {
        (eq_top(slots, tag), eq_all(slots, empty))
    }

    #[inline(always)]
    pub fn eq_top(slots: &[u32], tag: u8) -> u8 {
        both(slots, 0xFF00_0000, u32::from(tag) << 24)
    }

    #[inline(always)]
    pub fn eq_all(slots: &[u32], want: u32) -> u8 {
        both(slots, u32::MAX, want)
    }

    #[inline(always)]
    pub fn eq_masked(slots: &[u32], and: u32, want: u32) -> u8 {
        both(slots, and, want)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same answers a slot at a time, which is what the vectors have to
    /// agree with.
    fn scalar(slots: &[u32], and: u32, want: u32) -> u8 {
        let mut bits = 0u8;
        for (i, &slot) in slots.iter().enumerate() {
            bits |= u8::from(slot & and == want) << i;
        }
        bits
    }

    /// Eight slots holding every shape one can be in.
    fn sample() -> Vec<[u32; WIDTH]> {
        let live = |tag: u8, row: u32| (u32::from(tag) << 24) | row;
        vec![
            [0xFFFF_FFFF; WIDTH],
            [0x00FF_FFFF; WIDTH],
            [
                live(0x3F, 0),
                live(0x0F, 1),
                live(0xA7, 2),
                live(0x3F, 3),
                0xFFFF_FFFF,
                live(0x91, 4),
                live(0x3F, 5),
                live(0x02, 6),
            ],
            [
                0x00FF_FFFF,
                live(0xFF, 7),
                0xFFFF_FFFF,
                live(0x01, 0x00FF_FFFE),
                live(0x80, 8),
                0x00FF_FFFF,
                live(0xFF, 9),
                live(0x3F, 10),
            ],
        ]
    }

    #[test]
    fn a_group_answers_what_a_slot_at_a_time_would() {
        for group in sample() {
            for tag in [1u8, 0x02, 0x3F, 0x80, 0x91, 0xA7, 0xFF] {
                assert_eq!(
                    tags(&group, tag),
                    scalar(&group, 0xFF00_0000, u32::from(tag) << 24),
                    "tag {tag:#04x} over {group:08x?}"
                );
            }
            assert_eq!(
                empty(&group, 0xFFFF_FFFF),
                scalar(&group, u32::MAX, 0xFFFF_FFFF),
                "empty over {group:08x?}"
            );
            for tag in [1u8, 0x3F, 0xFF] {
                assert_eq!(
                    scan(&group, tag, 0xFFFF_FFFF),
                    (tags(&group, tag), empty(&group, 0xFFFF_FFFF)),
                    "one pass and two have to agree, tag {tag:#04x} over {group:08x?}"
                );
            }
            assert_eq!(
                free(&group, 0x00FF_FFFF),
                scalar(&group, 0x00FF_FFFF, 0x00FF_FFFF),
                "free over {group:08x?}"
            );
            for want in [0u32, 1, 6, 10, 0x00FF_FFFE] {
                assert_eq!(
                    rows(&group, 0x00FF_FFFF, want),
                    scalar(&group, 0x00FF_FFFF, want),
                    "row {want} over {group:08x?}"
                );
            }
        }
    }

    /// Bit `i` is slot `i`, which everything above reads off and nothing states.
    #[test]
    fn bit_i_is_slot_i() {
        for i in 0..WIDTH {
            let mut group = [0u32; WIDTH];
            group[i] = 0xFFFF_FFFF;
            assert_eq!(empty(&group, 0xFFFF_FFFF), 1 << i);
            assert_eq!(tags(&group, 0xFF), 1 << i);
        }
    }
}
