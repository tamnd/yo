//! The 64 byte index bucket.
//!
//! One cache line, seven entries, one link. Laid out exactly as `05` section
//! 2.1 specifies:
//!
//! | Offset | Size | Field   | Meaning                                  |
//! |--------|------|---------|------------------------------------------|
//! | 0      | 7x1  | `tag`   | 8 high bits of the hash, 0 means empty   |
//! | 7      | 1    | `flags` | bit 0 overflow present                   |
//! | 8      | 7x7  | `addr`  | 56 bit address, 4 bit space, 52 bit offset |
//! | 57     | 7    | `link`  | 56 bit address of the overflow bucket    |
//!
//! The probe is the reason for the shape. Seven tags fit in the first eight
//! bytes, so comparing all of them is one load and three arithmetic operations
//! with no branches and no SIMD intrinsics. aki measured the tag prefilter at
//! 3.31 ns against 4.77 ns without it (L13), and the whole point is that the
//! common case dereferences a key exactly once instead of up to seven times.

use yo_common::{Addr, CACHE_LINE};

/// Entries in one bucket.
pub const SLOTS: usize = 7;

/// A tag value of zero means the slot is empty.
pub const EMPTY: u8 = 0;

/// Bit 0 of `flags`: this bucket has an overflow bucket on its link.
const FLAG_OVERFLOW: u8 = 1;

/// Broadcast constant for the SWAR byte compare.
const ONES: u64 = 0x0101_0101_0101_0101;
/// Low seven bits of every lane.
const LOW7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
/// High bit of each of the seven tag lanes. Lane 7 is `flags` and is never a
/// match, which is what keeps a search for the empty tag from finding it.
const LANES: u64 = 0x0080_8080_8080_8080;

/// One index bucket.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Bucket {
    tags: [u8; SLOTS],
    flags: u8,
    addrs: [[u8; 7]; SLOTS],
    link: [u8; 7],
}

const _: () = {
    assert!(size_of::<Bucket>() == CACHE_LINE);
    assert!(align_of::<Bucket>() == CACHE_LINE);
};

impl Default for Bucket {
    fn default() -> Bucket {
        Bucket::EMPTY
    }
}

impl Bucket {
    /// A bucket with no entries.
    pub const EMPTY: Bucket = Bucket {
        tags: [EMPTY; SLOTS],
        flags: 0,
        addrs: [[0; 7]; SLOTS],
        link: [0; 7],
    };

    /// The seven tags and the flags byte, as one little endian word.
    #[inline(always)]
    fn tag_word(&self) -> u64 {
        // The pointer has to come from the whole bucket rather than from
        // `self.tags`. A pointer derived from a seven byte array carries
        // provenance over seven bytes, and reading eight through it is
        // undefined behaviour even though the eighth byte is the very next
        // field of the same struct. Miri catches it under stacked borrows,
        // nothing else does, and the two versions compile to the same load.
        let base: *const u8 = core::ptr::from_ref(self).cast();
        // SAFETY: `Bucket` is `repr(C)` with `tags` at offset 0 followed by
        // `flags`, so the first eight bytes are inside the bucket and are
        // initialised. An unaligned read is used because the compiler is free
        // to pick either and this makes the intent explicit.
        unsafe { base.cast::<u64>().read_unaligned().to_le() }
    }

    /// A bitmask over the seven slots whose tag equals `tag`.
    ///
    /// Bit `i` of the result is set when slot `i` matches. This is the whole
    /// prefilter: one load, an xor, a subtract, an and, and a shift.
    #[inline(always)]
    pub fn match_tag(&self, tag: u8) -> SlotMask {
        let word = self.tag_word();
        let x = word ^ (ONES.wrapping_mul(tag as u64));
        // Set the high bit of every lane whose byte is zero, meaning the tag
        // matched. The obvious `(x - ONES) & !x & LANES` is wrong here: a
        // borrow out of one lane walks into the next, so a slot holding 0x01
        // reports a match whenever the slot below it matched. That is fine in a
        // hash table whose control bytes have a reserved high bit, and it is
        // not fine here, where a tag is any of 256 values. So take the version
        // that cannot carry: masking off the high bits before the add keeps
        // every lane's sum under 0x100.
        let z = !((x & LOW7).wrapping_add(LOW7) | x | LOW7);
        SlotMask(z & LANES)
    }

    /// A bitmask over the seven slots that are empty.
    #[inline(always)]
    pub fn match_empty(&self) -> SlotMask {
        self.match_tag(EMPTY)
    }

    /// Whether every slot is occupied.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.match_empty().is_empty()
    }

    /// The tag in slot `i`.
    #[inline(always)]
    pub fn tag(&self, i: usize) -> u8 {
        self.tags[i]
    }

    /// The address in slot `i`.
    #[inline(always)]
    pub fn addr(&self, i: usize) -> Addr {
        Addr::from_bits(read56(&self.addrs[i]))
    }

    /// Put `addr` under `tag` in slot `i`.
    ///
    /// # Panics
    ///
    /// If `tag` is zero, which would make an occupied slot read as empty.
    #[inline(always)]
    pub fn set(&mut self, i: usize, tag: u8, addr: Addr) {
        assert_ne!(tag, EMPTY, "an occupied slot cannot carry the empty tag");
        self.tags[i] = tag;
        write56(&mut self.addrs[i], addr.to_bits());
    }

    /// Replace the address in an occupied slot, keeping its tag.
    #[inline(always)]
    pub fn set_addr(&mut self, i: usize, addr: Addr) {
        debug_assert_ne!(self.tags[i], EMPTY);
        write56(&mut self.addrs[i], addr.to_bits());
    }

    /// Empty slot `i`.
    ///
    /// Tombstone free (`05` section 2.3). The tag goes back to zero and the
    /// address is cleared so that a stale address cannot be followed by a
    /// probe that races a concurrent split in a future revision.
    #[inline(always)]
    pub fn clear(&mut self, i: usize) {
        self.tags[i] = EMPTY;
        self.addrs[i] = [0; 7];
    }

    /// The link to this bucket's overflow bucket, if it has one.
    #[inline(always)]
    pub fn link(&self) -> Option<u64> {
        if self.flags & FLAG_OVERFLOW == 0 {
            return None;
        }
        Some(read56(&self.link))
    }

    /// Whether this bucket has an overflow bucket.
    #[inline(always)]
    pub fn has_overflow(&self) -> bool {
        self.flags & FLAG_OVERFLOW != 0
    }

    /// Attach an overflow bucket.
    #[inline]
    pub fn set_link(&mut self, target: u64) {
        write56(&mut self.link, target);
        self.flags |= FLAG_OVERFLOW;
    }

    /// Detach the overflow bucket.
    #[inline]
    pub fn clear_link(&mut self) {
        self.link = [0; 7];
        self.flags &= !FLAG_OVERFLOW;
    }

    /// How many slots are occupied. Diagnostics only, not on the probe path.
    pub fn occupancy(&self) -> u32 {
        self.tags.iter().filter(|&&t| t != EMPTY).count() as u32
    }
}

impl core::fmt::Debug for Bucket {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Bucket")
            .field("tags", &self.tags)
            .field("occupancy", &self.occupancy())
            .field("link", &self.link())
            .finish()
    }
}

/// A set of matching slots, iterated lowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotMask(u64);

impl SlotMask {
    /// Whether nothing matched.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The lowest matching slot, or `None`.
    #[inline(always)]
    pub const fn first(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some((self.0.trailing_zeros() / 8) as usize)
        }
    }

    /// How many slots matched.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl Iterator for SlotMask {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let i = (self.0.trailing_zeros() / 8) as usize;
        self.0 &= self.0 - 1;
        Some(i)
    }
}

#[inline(always)]
fn read56(b: &[u8; 7]) -> u64 {
    // Assembled byte by byte rather than as a masked 8 byte read. The last
    // address in a bucket ends one byte before the link, so an 8 byte read from
    // the final slot would run into it, and the branch to avoid that costs more
    // than this does. The compiler turns this into a load and a shift on every
    // target that allows unaligned access.
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], 0])
}

#[inline(always)]
fn write56(b: &mut [u8; 7], v: u64) {
    let x = v.to_le_bytes();
    b.copy_from_slice(&x[..7]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Space;

    #[test]
    fn a_bucket_is_one_cache_line() {
        assert_eq!(size_of::<Bucket>(), 64);
        assert_eq!(align_of::<Bucket>(), 64);
    }

    #[test]
    fn field_offsets_match_the_specification() {
        let b = Bucket::EMPTY;
        let base = (&b as *const Bucket).addr();
        assert_eq!((&raw const b.tags).addr() - base, 0);
        assert_eq!((&raw const b.flags).addr() - base, 7);
        assert_eq!((&raw const b.addrs).addr() - base, 8);
        assert_eq!((&raw const b.link).addr() - base, 57);
    }

    #[test]
    fn an_empty_bucket_matches_nothing_and_is_all_free() {
        let b = Bucket::EMPTY;
        assert!(b.match_tag(1).is_empty());
        assert!(b.match_tag(255).is_empty());
        assert_eq!(b.match_empty().count(), SLOTS as u32);
        assert!(!b.is_full());
    }

    #[test]
    fn set_then_find() {
        let mut b = Bucket::EMPTY;
        let a = Addr::new(Space::Arena, 0x1234_5678);
        b.set(3, 0xAB, a);
        let m = b.match_tag(0xAB);
        assert_eq!(m.count(), 1);
        assert_eq!(m.first(), Some(3));
        assert_eq!(b.addr(3), a);
        assert_eq!(b.tag(3), 0xAB);
    }

    #[test]
    fn every_slot_round_trips_every_space() {
        for &space in Space::ALL {
            for i in 0..SLOTS {
                let mut b = Bucket::EMPTY;
                let a = Addr::new(space, yo_common::MAX_OFFSET);
                b.set(i, 0x5A, a);
                assert_eq!(b.addr(i), a, "slot {i} space {space:?}");
                assert_eq!(b.match_tag(0x5A).first(), Some(i));
            }
        }
    }

    /// The last address ends at byte 57 and the link starts there. A widened
    /// read of the final slot would pick up link bytes, so this is the test
    /// that catches it.
    #[test]
    fn the_last_slot_does_not_bleed_into_the_link() {
        let mut b = Bucket::EMPTY;
        let a = Addr::new(Space::Arena, 0xABCD);
        b.set(SLOTS - 1, 0x11, a);
        b.set_link(yo_common::MAX_OFFSET);
        assert_eq!(b.addr(SLOTS - 1), a);
        assert_eq!(b.link(), Some(yo_common::MAX_OFFSET));
    }

    /// And the reverse: writing the link must not disturb the last address.
    #[test]
    fn the_link_does_not_bleed_into_the_last_slot() {
        let mut b = Bucket::EMPTY;
        b.set_link(u64::MAX >> 8);
        let a = Addr::new(Space::Graph, 7);
        b.set(SLOTS - 1, 0x22, a);
        assert_eq!(b.link(), Some(u64::MAX >> 8));
        assert_eq!(b.addr(SLOTS - 1), a);
    }

    #[test]
    fn all_seven_slots_are_independent() {
        let mut b = Bucket::EMPTY;
        for i in 0..SLOTS {
            b.set(
                i,
                (i as u8) + 1,
                Addr::new(Space::Arena, (i as u64 + 1) * 16),
            );
        }
        assert!(b.is_full());
        for i in 0..SLOTS {
            assert_eq!(b.tag(i), (i as u8) + 1);
            assert_eq!(b.addr(i).offset(), (i as u64 + 1) * 16);
            assert_eq!(b.match_tag((i as u8) + 1).first(), Some(i));
        }
    }

    #[test]
    fn duplicate_tags_all_report() {
        let mut b = Bucket::EMPTY;
        b.set(1, 0x77, Addr::new(Space::Arena, 16));
        b.set(4, 0x77, Addr::new(Space::Arena, 32));
        b.set(6, 0x77, Addr::new(Space::Arena, 48));
        let m = b.match_tag(0x77);
        assert_eq!(m.count(), 3);
        assert_eq!(m.collect::<Vec<_>>(), vec![1, 4, 6]);
    }

    /// A tag equal to the flags byte must not produce a phantom eighth slot.
    /// This is why the lane mask covers seven lanes and not eight.
    #[test]
    fn the_flags_byte_is_never_a_match() {
        let mut b = Bucket::EMPTY;
        b.set_link(1); // sets flags to 1
        assert!(
            b.match_tag(1).is_empty(),
            "flags leaked into the tag search"
        );
        assert_eq!(b.match_empty().count(), SLOTS as u32);
        // And with flags at zero, searching for the empty tag must still find
        // seven slots rather than eight.
        let c = Bucket::EMPTY;
        assert_eq!(c.match_empty().count(), SLOTS as u32);
    }

    #[test]
    fn clear_frees_the_slot() {
        let mut b = Bucket::EMPTY;
        b.set(2, 0x99, Addr::new(Space::Arena, 64));
        assert_eq!(b.match_empty().count(), 6);
        b.clear(2);
        assert!(b.match_tag(0x99).is_empty());
        assert_eq!(b.match_empty().count(), 7);
        assert_eq!(b.addr(2), Addr::NONE);
    }

    #[test]
    fn links_attach_and_detach() {
        let mut b = Bucket::EMPTY;
        assert_eq!(b.link(), None);
        assert!(!b.has_overflow());
        b.set_link(4096);
        assert!(b.has_overflow());
        assert_eq!(b.link(), Some(4096));
        b.clear_link();
        assert_eq!(b.link(), None);
        assert!(!b.has_overflow());
    }

    #[test]
    #[should_panic(expected = "empty tag")]
    fn setting_the_empty_tag_is_refused() {
        let mut b = Bucket::EMPTY;
        b.set(0, EMPTY, Addr::new(Space::Arena, 16));
    }

    /// Exhaustive: for every occupancy pattern and every tag, the SWAR mask
    /// must agree with a byte by byte scan. 128 patterns times 256 tags is
    /// cheap and it covers every borrow interaction the subtract can produce.
    #[test]
    fn swar_agrees_with_a_plain_scan() {
        for pattern in 0u32..128 {
            let mut b = Bucket::EMPTY;
            for i in 0..SLOTS {
                if pattern & (1 << i) != 0 {
                    // Tags 1..=7 by slot, so patterns differ in content too.
                    b.set(i, (i as u8) + 1, Addr::new(Space::Arena, 16));
                }
            }
            // Every tag natively. Under Miri the tags that mean something plus
            // the ones that exercise the high bit of the SWAR word, which is
            // where a match either works or does not: a hundred and twenty
            // eight patterns against two hundred and fifty six tags is thirty
            // two thousand laps of an interpreter to prove something a couple
            // of thousand already proves.
            const MIRI_TAGS: [u8; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x40, 0x7f, 0x80, 0xff];
            let tags: Vec<u8> = if cfg!(miri) {
                MIRI_TAGS.to_vec()
            } else {
                (0..=255u8).collect()
            };
            for tag in tags {
                let want: Vec<usize> = (0..SLOTS).filter(|&i| b.tags[i] == tag).collect();
                let got: Vec<usize> = b.match_tag(tag).collect();
                assert_eq!(got, want, "pattern {pattern:#b} tag {tag}");
            }
        }
    }
}
