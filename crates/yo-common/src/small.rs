//! A list that stays on the stack until it does not fit.
//!
//! Every multi key command builds a handful of little vectors before it does any
//! work: the slot each key resolved to, the body each slot points at, the
//! operands sorted by size, a cursor per operand. Each of those is `k` long,
//! where `k` is the number of keys the command was given, and `k` is two or
//! three almost every time. A `SINTER` of two eight member sets does about two
//! hundred nanoseconds of real work and was paying five mallocs and five frees
//! on top of it.
//!
//! [`Small`] is those vectors without the allocator. Up to `N` elements it is an
//! array in the caller's frame, and past that it is a `Vec` and behaves exactly
//! as it did before, so a `SUNIONSTORE` over fifty keys is not made worse to
//! make the common one better.
//!
//! # Why `T: Copy` and why there is no unsafe here
//!
//! An inline buffer normally needs `MaybeUninit`, because `[T; N]` has to be
//! filled with something before the first element is written into it. That means
//! unsafe, and unsafe in a container means getting `Drop` and panic safety right
//! for a saving measured in nanoseconds.
//!
//! There is no need for any of it here. Everything this holds is `Copy`: a slot
//! number, a shared reference to a body, an index, a cursor over a sorted array.
//! So the buffer is filled with a copy of the first element and the elements
//! after `len` are that first element again, harmlessly. Nothing is ever read
//! out of them, nothing is ever dropped, and the whole type is ordinary safe
//! Rust.
//!
//! [`Small::Empty`] is a variant of its own for the same reason: an inline
//! buffer needs a value to fill itself with, and a list that never saw a `T` has
//! not got one.
//!
//! # What it is worth
//!
//! `yo-kv`'s `setops_small` bench, nanoseconds per operation over sets of eight
//! and sixty four members, before and after the three vectors inside
//! `yo_kv::setops` became this:
//!
//! ```text
//!                     before    after
//!   inter ints k=2     69.50    44.23
//!   inter ints k=3     88.34    58.28
//!   union ints k=2     80.39    69.45
//!   union ints k=3    122.08   114.71
//!   inter text k=2    160.58   154.84
//!   union text k=2    370.17   368.54
//! ```
//!
//! The integer intersection is the row that shows it, at about 1.5 times,
//! because a merge over small sorted arrays is a few dozen nanoseconds of real
//! work and three allocator round trips were most of what it was doing. The text
//! rows barely move, because those plans build a hash table sized by the members
//! and that is what they spend their time on.
//!
//! The bench calls `setops` directly, so it does not see the two more vectors
//! `Keyspace::set_slots` and `Keyspace::bodies_of` used to build per command. A
//! whole `SINTER` over three small sets went from eleven allocations to none.

use std::ops::{Deref, DerefMut};

/// A list of up to `N` elements on the stack, spilling to the heap past that.
#[derive(Debug, Clone)]
pub enum Small<T: Copy, const N: usize> {
    /// Nothing at all. See the module doc for why this is not `Inline` with a
    /// length of zero.
    Empty,
    /// The first `len` of `buf`. The rest are copies of the first element.
    Inline {
        /// The elements, and then padding that is never read.
        buf: [T; N],
        /// How many of `buf` are real.
        len: usize,
    },
    /// More than `N` of them, so the allocator was the right answer after all.
    Spilled(Vec<T>),
}

impl<T: Copy, const N: usize> Small<T, N> {
    /// An empty one.
    #[must_use]
    pub fn new() -> Small<T, N> {
        const { assert!(N > 0, "a Small with no inline room is just a Vec") };
        Small::Empty
    }

    /// Add one on the end.
    ///
    /// Crossing `N` copies what is already there into a `Vec` and never comes
    /// back, which is the whole of the spill and the reason [`Small::is_inline`]
    /// is a fact about the list rather than about its length.
    pub fn push(&mut self, v: T) {
        const { assert!(N > 0, "a Small with no inline room is just a Vec") };
        match self {
            Small::Empty => {
                *self = Small::Inline {
                    buf: [v; N],
                    len: 1,
                }
            }
            Small::Inline { buf, len } if *len < N => {
                buf[*len] = v;
                *len += 1;
            }
            Small::Inline { buf, len } => {
                let mut spill = Vec::with_capacity(N * 2);
                spill.extend_from_slice(&buf[..*len]);
                spill.push(v);
                *self = Small::Spilled(spill);
            }
            Small::Spilled(s) => s.push(v),
        }
    }

    /// Everything the iterator yields, on the stack if it fits.
    ///
    /// The spill is decided by the `N` and first element, so an iterator that
    /// yields `N + 1` copies once and moves everything already collected into a
    /// `Vec`. That copy is `N` elements of a `Copy` type and is not worth
    /// avoiding with a size hint that an iterator is allowed to lie about.
    pub fn collect<I: IntoIterator<Item = T>>(it: I) -> Small<T, N> {
        const { assert!(N > 0, "a Small with no inline room is just a Vec") };
        let mut it = it.into_iter();
        let Some(first) = it.next() else {
            return Small::Empty;
        };
        let mut buf = [first; N];
        let mut len = 1;
        while let Some(v) = it.next() {
            if len == N {
                let mut spill = Vec::with_capacity(N * 2);
                spill.extend_from_slice(&buf[..len]);
                spill.push(v);
                spill.extend(it);
                return Small::Spilled(spill);
            }
            buf[len] = v;
            len += 1;
        }
        Small::Inline { buf, len }
    }

    /// Everything in a slice, on the stack if it fits.
    ///
    /// [`Small::collect`] has to ask whether it has run out of room on every
    /// element, because an iterator is not obliged to say how many it has. A
    /// slice does say, so this asks once and then copies, which is a `memcpy`
    /// and not a loop. That is worth having wherever the elements are already
    /// laid out, and a fixed size key built in a local array is the case it was
    /// written for.
    #[must_use]
    pub fn from_slice(s: &[T]) -> Small<T, N> {
        const { assert!(N > 0, "a Small with no inline room is just a Vec") };
        let Some(&first) = s.first() else {
            return Small::Empty;
        };
        if s.len() > N {
            return Small::Spilled(s.to_vec());
        }
        let mut buf = [first; N];
        buf[..s.len()].copy_from_slice(s);
        Small::Inline { buf, len: s.len() }
    }

    /// Add a slice on the end, in one go.
    ///
    /// Pushing one at a time re-reads which variant this is on every element,
    /// and the compiler cannot keep the length in a register across it. This
    /// works out where everything is going first, so the common case is a
    /// `memcpy` into the inline buffer and the spill happens at most once.
    pub fn extend_from_slice(&mut self, s: &[T]) {
        const { assert!(N > 0, "a Small with no inline room is just a Vec") };
        if s.is_empty() {
            return;
        }
        match self {
            Small::Empty => *self = Small::from_slice(s),
            Small::Inline { buf, len } if *len + s.len() <= N => {
                buf[*len..*len + s.len()].copy_from_slice(s);
                *len += s.len();
            }
            Small::Inline { buf, len } => {
                let mut spill = Vec::with_capacity((*len + s.len()).max(N * 2));
                spill.extend_from_slice(&buf[..*len]);
                spill.extend_from_slice(s);
                *self = Small::Spilled(spill);
            }
            Small::Spilled(v) => v.extend_from_slice(s),
        }
    }

    /// The elements, in order.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Small::Empty => &[],
            Small::Inline { buf, len } => &buf[..*len],
            Small::Spilled(v) => v,
        }
    }

    /// The same, to be sorted or stepped through.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Small::Empty => &mut [],
            Small::Inline { buf, len } => &mut buf[..*len],
            Small::Spilled(v) => v,
        }
    }

    /// Whether this one is still on the stack, which is what the tests check
    /// and what nothing else has any business asking.
    #[must_use]
    pub fn is_inline(&self) -> bool {
        !matches!(self, Small::Spilled(_))
    }
}

// Written out rather than derived, whatever clippy thinks. `#[derive(Default)]`
// on an enum puts a `T: Default` bound on the whole thing, and the whole point
// of this type is holding references to bodies, which have no default.
#[allow(clippy::derivable_impls)]
impl<T: Copy, const N: usize> Default for Small<T, N> {
    fn default() -> Small<T, N> {
        Small::Empty
    }
}

impl<T: Copy, const N: usize> Deref for Small<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy, const N: usize> DerefMut for Small<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a Small<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> std::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a mut Small<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> std::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T: Copy, const N: usize> FromIterator<T> for Small<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Small<T, N> {
        Small::collect(it)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_one_is_an_empty_slice() {
        let s: Small<u32, 4> = Small::new();
        assert!(s.is_empty());
        assert_eq!(&*s, &[] as &[u32]);
        assert!(s.is_inline());
    }

    #[test]
    fn everything_up_to_n_stays_on_the_stack() {
        for n in 1..=4usize {
            let s: Small<u32, 4> = Small::collect(0..n as u32);
            assert!(s.is_inline(), "{n} elements spilled and should not have");
            assert_eq!(&*s, &(0..n as u32).collect::<Vec<_>>()[..]);
        }
    }

    #[test]
    fn one_past_n_spills_and_keeps_everything() {
        let s: Small<u32, 4> = Small::collect(0..5);
        assert!(!s.is_inline(), "five in a four did not spill");
        assert_eq!(&*s, &[0, 1, 2, 3, 4]);
    }

    /// The spill copies what it already had and then drains the rest of the
    /// iterator, which is the one place an element could go missing.
    #[test]
    fn a_long_spill_keeps_the_order() {
        let s: Small<u32, 4> = Small::collect(0..1_000);
        assert!(!s.is_inline());
        assert_eq!(s.len(), 1_000);
        assert!(s.iter().copied().eq(0..1_000));
    }

    /// The two slice forms have to agree with the iterator form at every
    /// length, either side of the spill, because they are the same list built a
    /// faster way and not a second type.
    #[test]
    fn the_slice_forms_agree_with_collecting() {
        for n in 0..12usize {
            let want: Vec<u32> = (0..n as u32).collect();

            let s: Small<u32, 4> = Small::from_slice(&want);
            assert_eq!(&*s, &want[..], "from_slice at {n}");
            assert_eq!(s.is_inline(), n <= 4, "from_slice spilled wrongly at {n}");

            for split in 0..=n {
                let mut s: Small<u32, 4> = Small::from_slice(&want[..split]);
                s.extend_from_slice(&want[split..]);
                assert_eq!(&*s, &want[..], "extend at {n} split at {split}");
            }
        }
    }

    /// Extending a list that has already spilled goes to the `Vec` and stays
    /// there, and extending by nothing at all leaves an empty one empty rather
    /// than making it inline with no elements.
    #[test]
    fn extending_past_the_spill_keeps_everything() {
        let mut s: Small<u32, 4> = Small::collect(0..6);
        s.extend_from_slice(&[6, 7, 8]);
        assert!(!s.is_inline());
        assert!(s.iter().copied().eq(0..9));

        let mut empty: Small<u32, 4> = Small::new();
        empty.extend_from_slice(&[]);
        assert!(empty.is_empty());
        assert!(matches!(empty, Small::Empty));
    }

    #[test]
    fn it_can_be_sorted_in_place_either_way_round() {
        let mut small: Small<u32, 4> = Small::collect([3, 1, 2]);
        small.sort_unstable();
        assert_eq!(&*small, &[1, 2, 3]);

        let mut big: Small<u32, 4> = Small::collect([9, 3, 1, 2, 7, 5]);
        big.sort_unstable();
        assert_eq!(&*big, &[1, 2, 3, 5, 7, 9]);
    }

    /// References are the point of the type, so they get their own case.
    #[test]
    fn it_holds_references() {
        let owned = [1u32, 2, 3];
        let s: Small<&u32, 4> = owned.iter().collect();
        assert_eq!(s.iter().copied().copied().collect::<Vec<_>>(), [1, 2, 3]);
    }

    /// The padding past `len` is a copy of the first element and is never read.
    /// Nothing depends on that being true, but it is worth pinning down that a
    /// short list does not accidentally expose it.
    #[test]
    fn the_padding_is_not_part_of_the_slice() {
        let s: Small<u32, 8> = Small::collect([7, 8]);
        assert_eq!(&*s, &[7, 8]);
        assert_eq!(s.len(), 2);
    }
}
