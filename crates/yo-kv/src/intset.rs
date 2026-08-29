//! A set of integers as one sorted packed array, which is Redis's intset.
//!
//! A set whose members all parse as integers is held as the integers themselves,
//! sorted, in the narrowest width that covers the widest of them, with no hash
//! table and no per member allocation anywhere:
//!
//! ```text
//! +----------+----------+---------+---------+-----+
//! | u32 width| u32 count| member 0| member 1| ... |
//! +----------+----------+---------+---------+-----+
//!   2, 4 or 8   how many   sorted ascending, width bytes each
//! ```
//!
//! Eight bytes of header and then nothing but members. At two byte width that is
//! two bytes an element with no overhead at all, which is the number G8 asks for
//! from a set of integers, and it is why this exists as a third representation
//! rather than everything small going in a listpack.
//!
//! Measured, on a set of five hundred and twelve small integers, that is 2.0
//! bytes a member against the listpack's 3.0 and the element table's 24.0. The
//! header is the only thing between it and exactly two, and it is amortised away
//! by about sixty members.
//!
//! Both header fields are little endian whatever the machine is, because Redis
//! writes them that way: `intrev32ifbe` is a no-op on a little endian host and a
//! byte swap on a big endian one, so the bytes on the wire and in the file are
//! little endian from either. Getting that backwards would produce a file a real
//! server cannot read, on the one class of machine nobody tests on.
//!
//! # Why sorted, and what it costs
//!
//! Membership is a binary search, which is nine steps at the 512 member ceiling
//! against the element table's one probe. The reason to accept that is that 512
//! members is at most four kilobytes, so the search stays in cache and the steps
//! are not nine cache misses, and above the threshold the set converts and stops
//! paying it at all.
//!
//! That paragraph used to be an argument with no measurement behind it, which in
//! this project is a warning sign: L6 put a positional probe at 70 ns and it
//! measured 13, and K11's crossover does not exist. `benches/intset.rs` settled
//! it. Minimum per iteration on an M3 laptop, membership against a member that
//! is there, at the sizes either side of the ceiling:
//!
//! ```text
//!   members     intset     listpack     element table
//!         8     4.6 ns       6.2 ns            7.7 ns
//!        64     6.6 ns      29.5 ns           10.2 ns
//!       128     7.7 ns      60.7 ns           10.2 ns
//!       512    10.4 ns     239.3 ns            9.0 ns
//! ```
//!
//! So the search is affordable, and the number that makes the case is not the
//! one against the table. Doubling the set three times costs the intset about
//! 3 ns in total, which is what a search that stays in cache looks like. What
//! the intset is actually replacing below the ceiling is the listpack, and there
//! it is eight times quicker at 128 members and pulling away, because a listpack
//! walks and this does not.
//!
//! The crossover with the element table lands almost exactly on Redis's ceiling.
//! At 128 the intset wins by a quarter, at 512 the table wins by a seventh, and
//! 512 is where a real server gives up on the intset anyway. That is a better
//! outcome than the argument deserved, and it was not predicted here: the guess
//! was that the search would be affordable, not that the constant Redis picked
//! in 2011 would sit on the crossover.
//!
//! Sorted also means an insert memmoves the tail, and that turns out not to
//! matter at these sizes. A scattered fill, where every add lands in the middle,
//! measured 6.47 ns a member at 128 against an ascending fill's 6.46, and the
//! two only separate at 512 where scattered costs 5.26 against 4.44. Four
//! kilobytes is not a memmove worth avoiding. The reason the ascending case is
//! still worth having, and worth a test, is a shape argument and not a timing
//! one: a fill in ascending order hits the "greater than the last member" test in
//! front of the search, so it never searches and never moves anything, and
//! `an_ascending_fill_never_moves_anything` asserts that rather than timing it.
//!
//! # Widening
//!
//! Adding a member too wide for the current width rewrites every member into the
//! new width. That happens at most twice in a set's life, 2 to 4 and 4 to 8, and
//! the new member is known to sit at one end before the rewrite starts, because
//! being too wide is exactly what it means to be outside the range of everything
//! already there. Negative goes to the front and positive to the back.
//!
//! Removing never narrows the width back. Redis does not either, and a set that
//! narrowed on the way down would rewrite itself on every second operation for a
//! workload that adds and removes around a boundary.

/// Why an intset from somewhere else was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Malformed {
    /// Shorter than the eight byte header.
    Short,
    /// The width is not 2, 4 or 8.
    Width,
    /// The count and the width do not account for the bytes that arrived.
    Length,
    /// The members are not in ascending order, or one appears twice.
    Order,
}

/// The widths a member can be stored in, which are the widths of the three
/// signed integer types Redis uses and nothing else.
const W16: u32 = 2;
const W32: u32 = 4;
const W64: u32 = 8;

/// Width, then count.
const HEADER: usize = 8;

/// A sorted packed set of integers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intset {
    /// The header and the members, in Redis's own layout, so that handing this
    /// to an RDB writer is a copy.
    bytes: Vec<u8>,
}

impl Intset {
    /// An empty set at the narrowest width.
    #[must_use]
    pub fn new() -> Intset {
        let mut bytes = Vec::with_capacity(HEADER);
        bytes.extend_from_slice(&W16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        Intset { bytes }
    }

    /// An empty set with room for `n` members at the narrowest width.
    ///
    /// Only a hint. A member that needs a wider slot still widens the set, and
    /// the reservation is then short, which costs one growth and no correctness.
    #[must_use]
    pub fn with_capacity(n: usize) -> Intset {
        let mut s = Intset::new();
        s.bytes.reserve(n * W16 as usize);
        s
    }

    /// Read a blob written by us or by a real server.
    ///
    /// The order check is the one worth having. A truncated blob is caught by
    /// the length arithmetic, but a blob whose members are out of order reads
    /// as a perfectly valid set that silently answers no to members it holds,
    /// because every search here assumes the order.
    pub fn from_bytes(bytes: &[u8]) -> Result<Intset, Malformed> {
        if bytes.len() < HEADER {
            return Err(Malformed::Short);
        }
        let width = u32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
        if width != W16 && width != W32 && width != W64 {
            return Err(Malformed::Width);
        }
        let count = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes")) as usize;
        let want = count
            .checked_mul(width as usize)
            .and_then(|n| n.checked_add(HEADER))
            .ok_or(Malformed::Length)?;
        if bytes.len() != want {
            return Err(Malformed::Length);
        }
        let s = Intset {
            bytes: bytes.to_vec(),
        };
        for i in 1..count {
            if s.at(i - 1) >= s.at(i) {
                return Err(Malformed::Order);
            }
        }
        Ok(s)
    }

    /// The blob, header included, ready to write to a file.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// How many members.
    #[inline]
    pub fn len(&self) -> usize {
        u32::from_le_bytes(self.bytes[4..8].try_into().expect("four bytes")) as usize
    }

    /// Whether there are none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes a member occupies, which is 2, 4 or 8.
    #[inline]
    pub fn width(&self) -> usize {
        u32::from_le_bytes(self.bytes[0..4].try_into().expect("four bytes")) as usize
    }

    /// The blob's length, which is what `MEMORY USAGE` counts.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Bytes held, including whatever the vector has reserved and not used.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    /// The member at `index`, counting from the smallest.
    ///
    /// # Panics
    ///
    /// If `index` is not under [`Intset::len`]. Every caller here has already
    /// bounded it, and a draw for `SRANDMEMBER` bounds it by construction.
    #[inline]
    #[must_use]
    pub fn at(&self, index: usize) -> i64 {
        let w = self.width();
        let at = HEADER + index * w;
        let raw = &self.bytes[at..at + w];
        match w {
            2 => i64::from(i16::from_le_bytes(raw.try_into().expect("two bytes"))),
            4 => i64::from(i32::from_le_bytes(raw.try_into().expect("four bytes"))),
            _ => i64::from_le_bytes(raw.try_into().expect("eight bytes")),
        }
    }

    /// The member at `index`, or `None` past the end.
    #[inline]
    pub fn get(&self, index: usize) -> Option<i64> {
        (index < self.len()).then(|| self.at(index))
    }

    /// The smallest member, or `None` if there are none.
    #[inline]
    pub fn min(&self) -> Option<i64> {
        self.get(0)
    }

    /// The largest member, or `None` if there are none.
    #[inline]
    pub fn max(&self) -> Option<i64> {
        self.len().checked_sub(1).map(|last| self.at(last))
    }

    /// Whether `v` is a member.
    #[inline]
    pub fn contains(&self, v: i64) -> bool {
        // A value too wide for this set's members cannot be one of them, and
        // saying so costs a compare instead of a search.
        width_of(v) <= self.width() as u32 && self.search(v).is_ok()
    }

    /// Every member, smallest first.
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        (0..self.len()).map(|i| self.at(i))
    }

    /// Add `v`. Answers whether it was not already there.
    pub fn add(&mut self, v: i64) -> bool {
        let w = width_of(v);
        if w > self.width() as u32 {
            self.widen_and_add(v, w);
            return true;
        }
        match self.search(v) {
            Ok(_) => false,
            Err(at) => {
                self.insert_at(at, v);
                true
            }
        }
    }

    /// Remove `v`. Answers whether it was there.
    pub fn remove(&mut self, v: i64) -> bool {
        if width_of(v) > self.width() as u32 {
            return false;
        }
        let Ok(at) = self.search(v) else {
            return false;
        };
        let w = self.width();
        let from = HEADER + at * w;
        self.bytes.drain(from..from + w);
        self.set_len(self.len() - 1);
        true
    }

    /// Where `v` is, or where it would go.
    ///
    /// The two range tests in front of the binary search are Redis's and they
    /// are not an optimisation of the search, they are what makes an ascending
    /// fill linear: every add lands past the last member, answers in two loads,
    /// and appends with nothing to move.
    fn search(&self, v: i64) -> Result<usize, usize> {
        let n = self.len();
        if n == 0 {
            return Err(0);
        }
        if v > self.at(n - 1) {
            return Err(n);
        }
        if v < self.at(0) {
            return Err(0);
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo <= hi {
            let mid = lo.midpoint(hi);
            let cur = self.at(mid);
            if v > cur {
                lo = mid + 1;
            } else if v < cur {
                // `mid` is at least one here, because `v` is not under the
                // first member and so cannot be under member zero.
                hi = mid - 1;
            } else {
                return Ok(mid);
            }
        }
        Err(lo)
    }

    /// Rewrite every member at `w` bytes and put `v` at whichever end it belongs.
    ///
    /// Back to front, so that a member is read before the wider write that would
    /// have covered it. `v` is outside the range of everything here, which is
    /// what being too wide means, so it goes at the front if it is negative and
    /// at the back if it is not, with no search.
    fn widen_and_add(&mut self, v: i64, w: u32) {
        let n = self.len();
        let old = self.width();
        let neww = w as usize;
        self.bytes.resize(HEADER + (n + 1) * neww, 0);
        let ahead = usize::from(v < 0);
        for i in (0..n).rev() {
            let at = HEADER + i * old;
            let raw = &self.bytes[at..at + old];
            let val = match old {
                2 => i64::from(i16::from_le_bytes(raw.try_into().expect("two bytes"))),
                4 => i64::from(i32::from_le_bytes(raw.try_into().expect("four bytes"))),
                _ => i64::from_le_bytes(raw.try_into().expect("eight bytes")),
            };
            write_at(&mut self.bytes, HEADER + (i + ahead) * neww, neww, val);
        }
        let end = if ahead == 1 { 0 } else { n };
        write_at(&mut self.bytes, HEADER + end * neww, neww, v);
        self.bytes[0..4].copy_from_slice(&w.to_le_bytes());
        self.set_len(n + 1);
    }

    /// Open a slot at `at` and put `v` in it.
    fn insert_at(&mut self, at: usize, v: i64) {
        let w = self.width();
        let from = HEADER + at * w;
        // One growth and one move. `splice` over an empty range is the memmove
        // and the reserve in one call, and the zeros are overwritten below.
        self.bytes
            .splice(from..from, std::iter::repeat_n(0u8, w))
            .for_each(drop);
        write_at(&mut self.bytes, from, w, v);
        self.set_len(self.len() + 1);
    }

    #[inline]
    fn set_len(&mut self, n: usize) {
        let n = u32::try_from(n).expect("an intset never reaches four billion members");
        self.bytes[4..8].copy_from_slice(&n.to_le_bytes());
    }
}

impl Default for Intset {
    fn default() -> Intset {
        Intset::new()
    }
}

/// The narrowest width that holds `v`.
#[inline]
const fn width_of(v: i64) -> u32 {
    if v < i32::MIN as i64 || v > i32::MAX as i64 {
        W64
    } else if v < i16::MIN as i64 || v > i16::MAX as i64 {
        W32
    } else {
        W16
    }
}

/// Write `v` at `at` in `w` bytes, little endian.
#[inline]
fn write_at(bytes: &mut [u8], at: usize, w: usize, v: i64) {
    match w {
        2 => bytes[at..at + 2].copy_from_slice(&(v as i16).to_le_bytes()),
        4 => bytes[at..at + 4].copy_from_slice(&(v as i32).to_le_bytes()),
        _ => bytes[at..at + 8].copy_from_slice(&v.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(vals: &[i64]) -> Intset {
        let mut s = Intset::new();
        for &v in vals {
            assert!(s.add(v), "{v} was supposed to be new");
        }
        s
    }

    fn members(s: &Intset) -> Vec<i64> {
        s.iter().collect()
    }

    #[test]
    fn an_empty_set_is_eight_bytes_and_holds_nothing() {
        let s = Intset::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.width(), 2);
        assert_eq!(s.byte_len(), 8);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
        assert!(!s.contains(0));
        assert_eq!(s.as_bytes(), &[2, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn members_come_back_sorted_however_they_went_in() {
        let s = of(&[5, -3, 100, 0, -70, 42]);
        assert_eq!(members(&s), [-70, -3, 0, 5, 42, 100]);
        assert_eq!(s.min(), Some(-70));
        assert_eq!(s.max(), Some(100));
        assert_eq!(s.len(), 6);
    }

    #[test]
    fn adding_the_same_member_twice_says_so_and_changes_nothing() {
        let mut s = of(&[1, 2, 3]);
        assert!(!s.add(2));
        assert_eq!(members(&s), [1, 2, 3]);
        assert_eq!(s.byte_len(), 8 + 3 * 2);
    }

    #[test]
    fn a_small_set_of_integers_costs_two_bytes_each() {
        // G8's number for a set of integers, and the reason this representation
        // exists next to the listpack rather than instead of it.
        let s = of(&(0..512).collect::<Vec<i64>>());
        assert_eq!(s.width(), 2);
        assert_eq!(s.byte_len(), 8 + 512 * 2);
        assert_eq!((s.byte_len() - 8) / s.len(), 2);
    }

    #[test]
    fn the_width_follows_the_widest_member_and_never_comes_back_down() {
        let mut s = of(&[1, 2, 3]);
        assert_eq!(s.width(), 2);

        s.add(100_000);
        assert_eq!(s.width(), 4, "past an i16");
        assert_eq!(members(&s), [1, 2, 3, 100_000]);

        s.add(-5_000_000_000);
        assert_eq!(s.width(), 8, "past an i32");
        assert_eq!(members(&s), [-5_000_000_000, 1, 2, 3, 100_000]);

        assert!(s.remove(-5_000_000_000));
        assert!(s.remove(100_000));
        assert_eq!(s.width(), 8, "removing does not narrow it back");
        assert_eq!(members(&s), [1, 2, 3]);
    }

    #[test]
    fn widening_puts_a_negative_at_the_front_and_a_positive_at_the_back() {
        // The whole of `widen_and_add` turns on this: the new member is outside
        // the range of what is there, so it needs no search, and getting the end
        // wrong writes it over a member instead of next to one.
        let mut up = of(&[-2, -1, 0, 1, 2]);
        up.add(70_000);
        assert_eq!(members(&up), [-2, -1, 0, 1, 2, 70_000]);

        let mut down = of(&[-2, -1, 0, 1, 2]);
        down.add(-70_000);
        assert_eq!(members(&down), [-70_000, -2, -1, 0, 1, 2]);
    }

    #[test]
    fn widening_an_empty_set_still_works() {
        let mut s = Intset::new();
        assert!(s.add(i64::MIN));
        assert_eq!(s.width(), 8);
        assert_eq!(members(&s), [i64::MIN]);
    }

    #[test]
    fn the_extremes_of_every_width_land_in_the_width_they_belong_to() {
        assert_eq!(width_of(0), 2);
        assert_eq!(width_of(i64::from(i16::MAX)), 2);
        assert_eq!(width_of(i64::from(i16::MIN)), 2);
        assert_eq!(width_of(i64::from(i16::MAX) + 1), 4);
        assert_eq!(width_of(i64::from(i16::MIN) - 1), 4);
        assert_eq!(width_of(i64::from(i32::MAX)), 4);
        assert_eq!(width_of(i64::from(i32::MIN)), 4);
        assert_eq!(width_of(i64::from(i32::MAX) + 1), 8);
        assert_eq!(width_of(i64::from(i32::MIN) - 1), 8);
        assert_eq!(width_of(i64::MAX), 8);
        assert_eq!(width_of(i64::MIN), 8);

        let s = of(&[i64::MIN, i64::MAX, 0]);
        assert_eq!(members(&s), [i64::MIN, 0, i64::MAX]);
        assert!(s.contains(i64::MIN));
        assert!(s.contains(i64::MAX));
    }

    #[test]
    fn a_member_too_wide_for_the_set_is_not_in_it() {
        // Not merely absent, unrepresentable, and answering that without a
        // search is the point.
        let s = of(&[1, 2, 3]);
        assert!(!s.contains(100_000));
        assert!(!s.contains(i64::MAX));
    }

    #[test]
    fn removing_takes_out_the_right_one_and_only_that_one() {
        let mut s = of(&[10, 20, 30, 40, 50]);
        assert!(s.remove(30));
        assert_eq!(members(&s), [10, 20, 40, 50]);
        assert!(!s.remove(30), "gone already");
        assert!(s.remove(10), "the first");
        assert_eq!(members(&s), [20, 40, 50]);
        assert!(s.remove(50), "the last");
        assert_eq!(members(&s), [20, 40]);
        assert_eq!(s.byte_len(), 8 + 2 * 2, "and the blob shrank each time");
    }

    #[test]
    fn a_set_can_be_emptied_and_used_again() {
        let mut s = of(&[1, 2, 3]);
        for v in [1, 2, 3] {
            assert!(s.remove(v));
        }
        assert!(s.is_empty());
        assert_eq!(s.byte_len(), 8);
        assert!(s.add(9));
        assert_eq!(members(&s), [9]);
    }

    #[test]
    fn every_member_of_a_big_set_is_found_and_no_stranger_is() {
        // Enough members to make the binary search do real work, in an order
        // that is neither ascending nor descending so the two range tests in
        // front of it are not what is being exercised.
        let mut s = Intset::new();
        for i in 0..1000i64 {
            assert!(s.add((i * 7919) % 1000 * 2));
        }
        assert_eq!(s.len(), 1000);
        for i in 0..1000i64 {
            assert!(s.contains(i * 2), "{} is a member", i * 2);
            assert!(!s.contains(i * 2 + 1), "{} is not", i * 2 + 1);
        }
        assert_eq!(members(&s), (0..1000i64).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn a_blob_survives_a_round_trip_through_bytes() {
        for vals in [
            &[][..],
            &[0],
            &[1, 2, 3],
            &[-70_000, 5, 70_000],
            &[i64::MIN, 0, i64::MAX],
        ] {
            let s = of(vals);
            let back = Intset::from_bytes(s.as_bytes()).expect("we wrote it");
            assert_eq!(back, s);
            assert_eq!(members(&back), members(&s));
        }
    }

    #[test]
    fn a_blob_that_is_wrong_is_refused_rather_than_believed() {
        assert_eq!(Intset::from_bytes(&[]), Err(Malformed::Short));
        assert_eq!(
            Intset::from_bytes(&[2, 0, 0, 0, 0, 0, 0]),
            Err(Malformed::Short)
        );

        let mut bad = of(&[1, 2, 3]).as_bytes().to_vec();
        bad[0] = 3;
        assert_eq!(Intset::from_bytes(&bad), Err(Malformed::Width));

        let mut short = of(&[1, 2, 3]).as_bytes().to_vec();
        short.pop();
        assert_eq!(Intset::from_bytes(&short), Err(Malformed::Length));

        let mut over = of(&[1, 2, 3]).as_bytes().to_vec();
        over[4] = 9;
        assert_eq!(Intset::from_bytes(&over), Err(Malformed::Length));

        // The one that would otherwise be believed: valid arithmetic, members
        // out of order, and every search after that quietly wrong.
        let mut jumbled = of(&[1, 2, 3]).as_bytes().to_vec();
        jumbled[8..10].copy_from_slice(&9i16.to_le_bytes());
        assert_eq!(Intset::from_bytes(&jumbled), Err(Malformed::Order));

        let mut twice = of(&[1, 2, 3]).as_bytes().to_vec();
        twice[10..12].copy_from_slice(&1i16.to_le_bytes());
        assert_eq!(Intset::from_bytes(&twice), Err(Malformed::Order));
    }

    #[test]
    fn the_header_is_little_endian_on_every_machine() {
        // Redis writes it little endian from a big endian host too, so a blob
        // this code produces has to be readable by a real server whatever it is
        // running on. Written out as bytes rather than as a round trip, because
        // a round trip through our own reader agrees with itself either way.
        let s = of(&[1, 258]);
        assert_eq!(
            s.as_bytes(),
            &[
                2, 0, 0, 0, // width, u32 little endian
                2, 0, 0, 0, // count, u32 little endian
                1, 0, // 1 as an i16 little endian
                2, 1, // 258 as an i16 little endian
            ]
        );
    }

    #[test]
    fn an_ascending_fill_never_moves_anything() {
        // Not a timing claim, a shape claim: `search` answers past the end for
        // every one of these, which is the branch that makes the fill linear.
        let mut s = Intset::new();
        for i in 0..100i64 {
            assert_eq!(s.search(i), Err(i as usize), "{i} appends");
            s.add(i);
        }
        assert_eq!(s.len(), 100);
    }
}
