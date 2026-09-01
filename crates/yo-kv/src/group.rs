//! Eight slots of an element table looked at in one go.
//!
//! An open addressed table has one dial, not two. Slots per member is one over
//! the load factor, and an unsuccessful linear probe costs about half of one
//! plus one over the square of one minus the load, so the memory the slot array
//! costs and the length of a probe through it are the same number seen twice.
//! At the three quarters the table used to run at that is 2.3 slots looked at
//! per miss, and there is no way to spend less memory without looking at more.
//!
//! ```text
//!   one group, eight slots, thirty two bytes
//! +------+------+------+------+------+------+------+------+
//! | tag  | tag  | FREE | tag  | tag  | tag  | EMPTY| tag  |
//! +------+------+------+------+------+------+------+------+
//!    one compare against the wanted tag, one mask, eight answers
//! ```
//!
//! The way off the dial is to stop looking at slots one at a time. A slot is a
//! tag in the top byte and a row index in the low twenty four, which is what the
//! row packing already put there, so eight of them are eight tags in thirty two
//! bytes and one vector compare answers all eight at once. The load can then go
//! up, which is where the memory is, without the probe getting longer, because
//! the probe is counted in groups and a group is one instruction either way.
//!
//! # What a slot can be
//!
//! Three things, told apart by the top two bits of the word, which is why the
//! tag is seven bits and not eight.
//!
//! A full slot has its top bit clear, so a tag is `0x00` to `0x7F`. [`EMPTY`] is
//! all ones and means nothing has ever been here. [`TOMB`] has the top bit set
//! and the low twenty four all ones, and means something was here and is not any
//! more.
//!
//! The low bits of both sentinels are all ones on purpose. A removal finds a
//! row's slot by comparing row indices, `MAX_ROWS` is one short of all ones, and
//! so neither sentinel can be mistaken for a row.
//!
//! # Why there has to be a tombstone
//!
//! Linear probing can close a hole by shifting the run behind it back, and that
//! is what this table did, which is why it never needed a tombstone. Group
//! probing does not walk a run in slot order and there is nothing to shift, so a
//! removal leaves a marker that says keep going rather than stop here.
//!
//! Tombstones are counted and they count against the load, so a table that is
//! churned in place rebuilds itself on the same schedule as one that only grows.
//! A removal that can prove no probe ever ran past its group writes [`EMPTY`]
//! instead and costs nothing at all: if the group already held an empty slot
//! then no search ever left it and no insert ever passed it, because a search
//! stops at the first empty and an insert takes the first free slot it sees.

/// How many slots are looked at in one go.
pub const WIDTH: usize = 8;

/// The low bits of a slot, which are the row index.
pub const ROW: u32 = 0x00FF_FFFF;

/// A slot nothing has ever been written to.
pub const EMPTY: u32 = 0xFFFF_FFFF;

/// A slot something was written to and then removed from.
pub const TOMB: u32 = 0x80FF_FFFF;

/// The bit that is set on [`EMPTY`] and [`TOMB`] and clear on everything else.
pub const FREE: u32 = 0x8000_0000;

/// The seven bit tag a hash files a name under.
///
/// Off the top of the hash, where the group index is off the bottom, so a name
/// that collides on one has an even chance on the other. Seven bits rather than
/// eight because the eighth is what tells a full slot from a free one, and one
/// false positive in a hundred and twenty eight costs a row read to reject.
#[inline]
#[must_use]
pub const fn tag(h: u64) -> u32 {
    ((h >> 56) as u32) & 0x7F
}

/// A slot holding `row` under `tag`.
#[inline]
#[must_use]
pub const fn slot(tag: u32, row: u32) -> u32 {
    (tag << 24) | row
}

/// Which lanes of a group answered, lowest lane first.
///
/// It is an iterator over the lanes that matched, and it is also the answer to
/// whether any of them did, because both readings come off the same eight bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mask(u8);

impl Mask {
    /// Whether any lane answered.
    #[inline]
    #[must_use]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// The lowest lane that answered.
    #[inline]
    #[must_use]
    pub const fn lowest(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }
}

impl Iterator for Mask {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let at = self.lowest()?;
        self.0 &= self.0 - 1;
        Some(at)
    }
}

/// Eight slots held in whatever register this machine has for them.
#[derive(Debug, Clone, Copy)]
pub struct Group(imp::Raw);

impl Group {
    /// The eight slots starting at `at`.
    ///
    /// # Panics
    ///
    /// If there are not eight slots there, which means the caller lost the
    /// group alignment the table is built on.
    #[inline]
    #[must_use]
    pub fn load(slots: &[u32], at: usize) -> Group {
        assert!(at + WIDTH <= slots.len(), "a group is eight slots");
        // SAFETY: the bound above is the whole safety condition, and `at` is a
        // multiple of `WIDTH` by construction so the read stays inside one
        // allocation.
        Group(unsafe { imp::load(slots.as_ptr().add(at)) })
    }

    /// The lanes holding a full slot filed under `tag`.
    #[inline]
    #[must_use]
    pub fn tagged(self, tag: u32) -> Mask {
        Mask(imp::tagged(self.0, tag))
    }

    /// The lanes holding a full slot pointing at `row`.
    ///
    /// This looks at the low twenty four bits and nothing else, which is one
    /// instruction fewer than also proving the lane is full. It is allowed to
    /// be because both sentinels have those bits all ones and a row index never
    /// does, so a real row cannot collide with one.
    ///
    /// # Panics
    ///
    /// In debug, if asked about a row index that is not one, which is the
    /// assumption the shortcut above rests on.
    #[inline]
    #[must_use]
    pub fn rowed(self, row: u32) -> Mask {
        debug_assert!(row < ROW, "a row index is never all ones");
        Mask(imp::rowed(self.0, row))
    }

    /// The lanes nothing has ever been written to.
    #[inline]
    #[must_use]
    pub fn empty(self) -> Mask {
        Mask(imp::empty(self.0))
    }

    /// The lanes an insert may take, which is the empty ones and the dead ones.
    #[inline]
    #[must_use]
    pub fn free(self) -> Mask {
        Mask(imp::free(self.0))
    }
}

/// Where a probe is, counted in groups rather than in slots.
///
/// The step grows by one group each time, so the offsets from the first group
/// are the triangular numbers. Over a power of two count of groups those cover
/// every group exactly once, which a fixed step of one does not do any better
/// but which spreads a run of full groups out instead of walking along it.
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    /// The first slot of the group being looked at.
    at: usize,
    /// How far the next group is from this one, in slots.
    step: usize,
    /// One less than the number of slots, which is a power of two.
    mask: usize,
}

impl Probe {
    /// Start at the group holding `home`.
    #[inline]
    #[must_use]
    pub const fn from_slot(home: usize, slots: usize) -> Probe {
        Probe {
            at: home & !(WIDTH - 1),
            step: 0,
            mask: slots - 1,
        }
    }

    /// The first slot of the group being looked at.
    #[inline]
    #[must_use]
    pub const fn at(self) -> usize {
        self.at
    }

    /// Move on to the next group.
    #[inline]
    pub const fn next(&mut self) {
        self.step += WIDTH;
        self.at = (self.at + self.step) & self.mask;
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_loadu_si128,
        _mm_movemask_ps, _mm_set1_epi32, _mm_srli_epi32,
    };

    /// Two SSE2 registers, which every x86-64 has without asking.
    pub type Raw = (__m128i, __m128i);

    /// The four sign bits of a vector, which is what a comparison leaves behind.
    #[inline]
    fn bits(v: __m128i) -> u8 {
        // SAFETY: SSE2 is in the x86-64 baseline.
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(v)) as u8 }
    }

    /// The eight slots at `p`, which need no alignment past the four bytes a
    /// slot already has.
    #[inline]
    pub unsafe fn load(p: *const u32) -> Raw {
        // SAFETY: the caller has checked there are eight slots at `p`.
        unsafe { (_mm_loadu_si128(p.cast()), _mm_loadu_si128(p.add(4).cast())) }
    }

    /// Both halves put through `f` and their nibbles joined.
    #[inline]
    fn both(v: Raw, f: impl Fn(__m128i) -> __m128i) -> u8 {
        bits(f(v.0)) | (bits(f(v.1)) << 4)
    }

    #[inline]
    pub fn tagged(v: Raw, tag: u32) -> u8 {
        // SAFETY: SSE2 is in the x86-64 baseline.
        unsafe {
            let want = _mm_set1_epi32(tag as i32);
            both(v, |h| _mm_cmpeq_epi32(_mm_srli_epi32(h, 24), want))
        }
    }

    #[inline]
    pub fn rowed(v: Raw, row: u32) -> u8 {
        // SAFETY: SSE2 is in the x86-64 baseline.
        unsafe {
            let low = _mm_set1_epi32(super::ROW as i32);
            let want = _mm_set1_epi32(row as i32);
            both(v, |h| _mm_cmpeq_epi32(_mm_and_si128(h, low), want))
        }
    }

    #[inline]
    pub fn empty(v: Raw) -> u8 {
        // SAFETY: SSE2 is in the x86-64 baseline.
        unsafe {
            let want = _mm_set1_epi32(super::EMPTY as i32);
            both(v, |h| _mm_cmpeq_epi32(h, want))
        }
    }

    /// No comparison at all, because free is exactly the top bit and the top
    /// bit is what a float sign mask reads.
    #[inline]
    pub fn free(v: Raw) -> u8 {
        bits(v.0) | (bits(v.1) << 4)
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::{
        uint32x4_t, vaddvq_u32, vandq_u32, vceqq_u32, vcgeq_u32, vdupq_n_u32, vld1q_u32,
        vshrq_n_u32,
    };

    /// Two NEON registers, which every aarch64 has without asking.
    pub type Raw = (uint32x4_t, uint32x4_t);

    /// One bit per lane, out of a comparison that left all ones or all zeros.
    ///
    /// There is no sign mask instruction here, so the lanes are weighted one,
    /// two, four and eight and added across, which comes to the same nibble.
    #[inline]
    fn bits(cmp: uint32x4_t) -> u8 {
        // SAFETY: NEON is in the aarch64 baseline.
        unsafe {
            let weights = vld1q_u32([1u32, 2, 4, 8].as_ptr());
            vaddvq_u32(vandq_u32(cmp, weights)) as u8
        }
    }

    /// The eight slots at `p`, which need no alignment past the four bytes a
    /// slot already has.
    #[inline]
    pub unsafe fn load(p: *const u32) -> Raw {
        // SAFETY: the caller has checked there are eight slots at `p`.
        unsafe { (vld1q_u32(p), vld1q_u32(p.add(4))) }
    }

    /// Both halves put through `f` and their nibbles joined.
    #[inline]
    fn both(v: Raw, f: impl Fn(uint32x4_t) -> uint32x4_t) -> u8 {
        bits(f(v.0)) | (bits(f(v.1)) << 4)
    }

    #[inline]
    pub fn tagged(v: Raw, tag: u32) -> u8 {
        // SAFETY: NEON is in the aarch64 baseline.
        unsafe {
            let want = vdupq_n_u32(tag);
            both(v, |h| vceqq_u32(vshrq_n_u32(h, 24), want))
        }
    }

    #[inline]
    pub fn rowed(v: Raw, row: u32) -> u8 {
        // SAFETY: NEON is in the aarch64 baseline.
        unsafe {
            let low = vdupq_n_u32(super::ROW);
            let want = vdupq_n_u32(row);
            both(v, |h| vceqq_u32(vandq_u32(h, low), want))
        }
    }

    #[inline]
    pub fn empty(v: Raw) -> u8 {
        // SAFETY: NEON is in the aarch64 baseline.
        unsafe {
            let want = vdupq_n_u32(super::EMPTY);
            both(v, |h| vceqq_u32(h, want))
        }
    }

    /// The top bit on its own, which is every value at or above [`super::FREE`]
    /// read as unsigned.
    #[inline]
    pub fn free(v: Raw) -> u8 {
        // SAFETY: NEON is in the aarch64 baseline.
        unsafe {
            let want = vdupq_n_u32(super::FREE);
            both(v, |h| vcgeq_u32(h, want))
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use plain as imp;

/// The same eight answers worked out one lane at a time.
///
/// This is what runs on a machine with neither of the two vector units above,
/// and it is compiled everywhere whether it runs or not, because the tests check
/// it against whichever of the two is in use and a fallback that is only built
/// on the machine nobody has is a fallback nobody has ever run.
///
/// The shape is the same as the vector ones on purpose. A compiler that can see
/// eight independent comparisons over a fixed length array will usually find the
/// vector instructions itself, and where it cannot, eight comparisons with no
/// branch between them still beat eight probes with a dependent load and a
/// branch each.
// Nothing outside the tests calls it on a machine that has a vector unit, which
// is the point of it and is not something to be warned about.
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    allow(dead_code, reason = "the tests are the only caller on these two")
)]
mod plain {
    /// Eight slots in eight registers, or in whatever the compiler makes of it.
    pub type Raw = [u32; super::WIDTH];

    /// The eight slots at `p`.
    #[inline]
    pub unsafe fn load(p: *const u32) -> Raw {
        // SAFETY: the caller has checked there are eight slots at `p`.
        unsafe { std::ptr::read_unaligned(p.cast()) }
    }

    /// One bit per lane that `f` said yes to.
    #[inline]
    fn bits(v: Raw, f: impl Fn(u32) -> bool) -> u8 {
        let mut m = 0u8;
        for (i, &slot) in v.iter().enumerate() {
            m |= u8::from(f(slot)) << i;
        }
        m
    }

    #[inline]
    pub fn tagged(v: Raw, tag: u32) -> u8 {
        bits(v, |slot| slot >> 24 == tag)
    }

    #[inline]
    pub fn rowed(v: Raw, row: u32) -> u8 {
        bits(v, |slot| slot & super::ROW == row)
    }

    #[inline]
    pub fn empty(v: Raw) -> u8 {
        bits(v, |slot| slot == super::EMPTY)
    }

    #[inline]
    pub fn free(v: Raw) -> u8 {
        bits(v, |slot| slot & super::FREE != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A group with `full` under the tags given and the rest empty or dead.
    fn group(slots: &[u32]) -> Group {
        assert_eq!(slots.len(), WIDTH);
        Group::load(slots, 0)
    }

    /// Whatever this machine actually runs, checked against the plain version.
    ///
    /// The vector units cannot both be tested on one machine, so what is worth
    /// testing on either is that it agrees with the one implementation of this
    /// that is obviously right. Everything a group is asked, over every shape a
    /// group can be in.
    #[test]
    fn the_vector_answer_is_the_plain_answer() {
        // A cheap spread of slot values: full slots at every tag, both
        // sentinels, and the awkward ones next to them.
        let shapes: Vec<u32> = (0..0x80)
            .map(|t| slot(t, t * 37 % 1000))
            .chain([EMPTY, TOMB, FREE, ROW, 0, slot(0, 0), slot(0x7F, ROW - 1)])
            .collect();

        // A cheap deterministic walk over the shapes, so a group is a different
        // mixture every time round without a random source.
        let mut r = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..20_000 {
            let mut slots = [0u32; WIDTH];
            for lane in &mut slots {
                r = r.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *lane = shapes[(r >> 33) as usize % shapes.len()];
            }
            let v = Group::load(&slots, 0);
            // SAFETY: eight slots are there, which is the whole condition.
            let p = unsafe { plain::load(slots.as_ptr()) };

            assert_eq!(v.empty(), Mask(plain::empty(p)), "{slots:08x?}");
            assert_eq!(v.free(), Mask(plain::free(p)), "{slots:08x?}");
            for tag in [0u32, 1, 0x2A, 0x7F] {
                assert_eq!(v.tagged(tag), Mask(plain::tagged(p, tag)), "{slots:08x?}");
            }
            for row in [0u32, 1, 37, 999, ROW - 1] {
                assert_eq!(v.rowed(row), Mask(plain::rowed(p, row)), "{slots:08x?}");
            }
        }
    }

    #[test]
    fn a_tag_is_seven_bits_and_never_looks_free() {
        for h in [
            0u64,
            1,
            u64::MAX,
            0x7F00_0000_0000_0000,
            0x8000_0000_0000_0000,
        ] {
            let t = tag(h);
            assert!(t < 0x80, "a tag keeps its top bit clear");
            assert_eq!(slot(t, 0) & FREE, 0, "a full slot never looks free");
        }
        assert_eq!(tag(u64::MAX), 0x7F);
        assert_eq!(tag(0x2A00_0000_0000_0000), 0x2A);
    }

    #[test]
    fn neither_sentinel_can_be_mistaken_for_a_row() {
        assert_eq!(EMPTY & ROW, ROW);
        assert_eq!(TOMB & ROW, ROW);
        assert_ne!(EMPTY & FREE, 0);
        assert_ne!(TOMB & FREE, 0);
        assert_ne!(EMPTY, TOMB);
        assert!(
            crate::elem::MAX_ROWS < ROW as usize,
            "a row is never all ones"
        );
    }

    #[test]
    fn every_lane_answers_for_itself() {
        for lane in 0..WIDTH {
            let mut slots = [EMPTY; WIDTH];
            slots[lane] = slot(0x2A, 7);
            let g = group(&slots);

            assert_eq!(g.tagged(0x2A).collect::<Vec<_>>(), vec![lane]);
            assert_eq!(g.rowed(7).collect::<Vec<_>>(), vec![lane]);
            assert!(!g.tagged(0x2B).any(), "another tag matches nothing");
            assert!(!g.rowed(8).any(), "another row matches nothing");

            let rest: Vec<usize> = (0..WIDTH).filter(|&i| i != lane).collect();
            assert_eq!(g.empty().collect::<Vec<_>>(), rest);
            assert_eq!(g.free().collect::<Vec<_>>(), rest);
        }
    }

    #[test]
    fn a_dead_lane_is_free_without_being_empty() {
        let mut slots = [slot(3, 1); WIDTH];
        slots[2] = TOMB;
        slots[5] = EMPTY;
        let g = group(&slots);

        assert_eq!(g.empty().collect::<Vec<_>>(), vec![5]);
        assert_eq!(g.free().collect::<Vec<_>>(), vec![2, 5]);
        assert_eq!(g.free().lowest(), Some(2), "an insert takes the dead one");
        assert_eq!(g.tagged(3).collect::<Vec<_>>(), vec![0, 1, 3, 4, 6, 7]);
        assert_eq!(
            g.rowed(1).collect::<Vec<_>>(),
            vec![0, 1, 3, 4, 6, 7],
            "a sentinel has its row bits all ones and is never one"
        );
    }

    #[test]
    fn a_full_group_answers_nothing() {
        let slots: [u32; WIDTH] = std::array::from_fn(|i| slot(1, i as u32));
        let g = group(&slots);

        assert!(!g.empty().any());
        assert!(!g.free().any());
        assert_eq!(g.empty().lowest(), None);
        assert_eq!(g.tagged(1).count(), WIDTH);
        assert_eq!(g.rowed(3).collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn a_mask_is_read_lowest_lane_first() {
        let mut slots = [slot(9, 0); WIDTH];
        slots[1] = EMPTY;
        slots[6] = TOMB;
        let g = group(&slots);

        assert_eq!(g.free().lowest(), Some(1));
        assert_eq!(g.free().collect::<Vec<_>>(), vec![1, 6]);
        assert_eq!(g.empty().lowest(), Some(1));
    }

    #[test]
    fn a_probe_reaches_every_group_once() {
        for groups in [2usize, 4, 8, 16, 64] {
            let slots = groups * WIDTH;
            for home in [0usize, 1, 7, slots / 2, slots - 1] {
                let mut p = Probe::from_slot(home, slots);
                let mut seen = Vec::new();
                for _ in 0..groups {
                    assert_eq!(p.at() % WIDTH, 0, "a probe stays on a group");
                    seen.push(p.at());
                    p.next();
                }
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), groups, "every group, and none of them twice");
            }
        }
    }

    #[test]
    fn a_probe_starts_at_the_group_holding_its_home() {
        for home in 0..64usize {
            let p = Probe::from_slot(home, 64);
            assert_eq!(p.at(), home / WIDTH * WIDTH);
        }
    }
}
