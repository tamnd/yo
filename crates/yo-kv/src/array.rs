//! A sparse array: a sequence indexed by a `u64`, with holes.
//!
//! This is the type behind the `AR*` commands Redis added in 8.9, and it is the
//! only collection here whose index is unsigned. A list is indexed from either
//! end and `-1` is the last element. An array is indexed by position in a space
//! that runs to `2^64 - 2`, so `-1` is not the end of anything, it is an error,
//! and most of that space is empty at any moment.
//!
//! ```text
//!   slices, sorted by id, binary searched
//! +---------+---------+-------------------+---------+
//! | id 0    | id 7    |        ...        | id 9e12 |
//! | sparse  | dense   |                   | sparse  |
//! +---------+---------+-------------------+---------+
//!      \                                        /
//!       \--- offsets and words -----------------/
//!                     |
//!             +-------------------------------+
//!             | one blob per array, for the   |
//!             | values too long to inline     |
//!             +-------------------------------+
//! ```
//!
//! # Two numbers that are not the same number
//!
//! [`Array::len`] is the highest populated index plus one and [`Array::count`]
//! is how many indices are populated. `ARSET k 1000000 x` gives a length of a
//! million and one and a count of one. Every other collection here has one
//! number for both and it is worth saying out loud, because a caller that
//! reaches for the wrong one gets an answer rather than an error.
//!
//! # Where the slices live
//!
//! Redis keeps a flat directory of slice pointers indexed by slice id, and then
//! a second structure over that for when the ids get far apart, because a flat
//! array indexed by `idx >> 12` is nine billion entries for an index of nine
//! trillion. Here it is one `Vec` of `(id, slice)` kept sorted and binary
//! searched, which covers the whole index space in one structure with no
//! second mode to get wrong, and costs a handful of compares on a get instead
//! of one load. A key with a thousand slices is ten compares, and a key with a
//! thousand slices is four million elements, so the compares are noise next to
//! what the caller is doing with the data.
//!
//! # Where the values live
//!
//! A value of eight bytes or more is a slice of one blob owned by the array, and
//! everything shorter is inlined in the word itself. Redis heap allocates each
//! of those, paying a malloc header and the rounding on every one. One blob per
//! key pays the bytes and nothing else, at the cost of having to compact when
//! enough of it is dead. See `Word` for the four things a word can be.

use std::cmp::Ordering;

use yo_common::num;
use yo_common::{Code, Error, Result};

use crate::frozen::{self, Broken};

/// How many indices one slice covers.
///
/// Redis's `AR_SLICE_SIZE_DEFAULT`, and unlike Redis it is not configurable,
/// because the two settings either side of the default were not worth a branch
/// in `slice_of` and nobody has ever reported tuning them.
pub const SLICE_SIZE: u64 = 4096;

/// `SLICE_SIZE` as a shift, so the divide is a shift.
const SLICE_BITS: u32 = SLICE_SIZE.trailing_zeros();

/// The most elements a slice holds while staying sparse.
///
/// Redis's `AR_SPARSE_KMAX_DEFAULT`. Above this a slice is worth an index, below
/// it the pairs are cheaper than the holes.
const SPARSE_MAX: usize = 10;

/// The fewest elements a dense slice keeps before going back to pairs.
///
/// Redis's `AR_SPARSE_KMIN_DEFAULT`. It is half of `SPARSE_MAX` rather than
/// equal to it so that a slice sitting on the line does not rebuild itself on
/// every other write.
const SPARSE_MIN: usize = 5;

/// The directory and its slices, which is the only form an array is written in.
const FORM_SLICES: u8 = 1;
/// On the form byte, that the insert cursor has been set and follows it.
const HAS_INSERT: u8 = 0x80;
/// A slice held as offsets and words.
const LAYOUT_SPARSE: u8 = 1;
/// A slice held as a window.
const LAYOUT_DENSE: u8 = 2;

/// The largest index an array will accept.
///
/// Redis reserves `UINT64_MAX` as "no insert has happened yet" in the cursor
/// that `ARINSERT` and `ARNEXT` share, so it is not a position anything can be
/// written to, and `ARSET k 18446744073709551615 v` is an error rather than a
/// write. Keeping the same ceiling here keeps that cursor able to mean the same
/// thing when it lands.
pub const INDEX_MAX: u64 = u64::MAX - 1;

/// Room for the longest text an [`Element`] can turn into.
///
/// The float is the long one, and it is the widest double plus the `.0` that
/// gets appended to one that came out looking like an integer.
pub const ELEMENT_MAX: usize = num::DOUBLE_MAX + 2;

/// What is stored at one index, in whichever form it was worth keeping.
///
/// Handing back the stored form rather than bytes is Y18: a value that went in
/// as `12345` is held as an `i64` and formatted once, into the reply buffer, at
/// the moment the reply is built. Use [`Element::text`] to get the bytes when
/// the caller has nowhere better to put them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Element<'a> {
    /// A value that was written as an integer and is held as one.
    Int(i64),
    /// A value that was written as a decimal and round trips as a double.
    Float(f64),
    /// A string long enough to live in the array's blob, borrowed from it.
    Str(&'a [u8]),
    /// A string of seven bytes or fewer, which was packed into the word and so
    /// has nowhere to be borrowed from.
    Short(Short),
}

impl<'a> Element<'a> {
    /// The bytes a client would see, written into `buf` if they are not stored
    /// anywhere already.
    ///
    /// A blob string is already bytes and comes back borrowed with `buf`
    /// untouched. Everything else has to be written somewhere, and it goes into
    /// the caller's stack buffer rather than a `Vec`, because the caller needs
    /// the length before it can write the bulk header and a shard thread that
    /// allocates aborts.
    pub fn text<'b>(&'b self, buf: &'b mut [u8; ELEMENT_MAX]) -> &'b [u8]
    where
        'a: 'b,
    {
        match *self {
            // The two cases with nothing to do, borrowed straight through.
            Element::Str(s) => s,
            Element::Short(ref s) => s.as_bytes(),
            Element::Int(i) => {
                let mut digits = [0u8; num::DIGITS_MAX];
                let text = num::i64_digits(&mut digits, i);
                let n = text.len();
                buf[..n].copy_from_slice(text);
                &buf[..n]
            }
            Element::Float(d) => {
                let mut wide = [0u8; num::DOUBLE_MAX];
                let text = num::write_double(&mut wide, d);
                let mut n = text.len();
                buf[..n].copy_from_slice(text);
                // Redis's `arFormatFloat`: a stored double that prints without a
                // dot or an exponent gets `.0` put back on, so that a value
                // written as `1.0` does not read back as `1`. Nothing that
                // reached this branch was written as `1`, because the integer
                // encoding takes those first.
                if !text.iter().any(|&c| c == b'.' || c == b'e' || c == b'E') {
                    buf[n] = b'.';
                    buf[n + 1] = b'0';
                    n += 2;
                }
                &buf[..n]
            }
        }
    }
}

/// A string short enough that it was stored inside the word.
///
/// It exists because an [`Element`] borrows and there is nothing to borrow
/// from: the bytes were in the eight bytes that were read, and unpacking them
/// somewhere is unavoidable. Carrying them by value in the element is seven
/// bytes on the caller's stack, which is cheaper than the alternative of making
/// every reader pass in a scratch buffer for the case where the value is short.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Short {
    buf: [u8; INLINE_MAX],
    len: u8,
}

impl Short {
    /// The bytes, as they were written.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..usize::from(self.len)]
    }
}

impl core::fmt::Debug for Short {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", String::from_utf8_lossy(self.as_bytes()))
    }
}

/// One stored value, in eight bytes, or the empty slot.
///
/// The low two bits are a tag and the rest is payload, which is Redis's tagged
/// pointer scheme with the pointer replaced by an offset into the array's own
/// blob. That replacement is the whole memory argument: Redis pays eight bytes
/// of pointer plus a malloc header plus rounding for every value of eight bytes
/// or more, and this pays eight bytes plus the payload.
///
/// ```text
///   tag 00  offset into the blob: length in bits 2..32, start in bits 32..64
///   tag 01  a signed integer in the top 62 bits
///   tag 10  an f64 with its low two bits cleared
///   tag 11  a string of up to seven bytes: length in bits 2..5, bytes from 8
/// ```
///
/// [`Word::EMPTY`] is all zeroes, which is unambiguous: a blob word always has a
/// length of at least eight so its payload is never zero, and the other three
/// tags are non zero by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Word(u64);

/// Values of this many bytes and up go in the blob, and shorter ones inline.
const INLINE_MAX: usize = 7;

/// The largest blob one array can hold, because a start has to fit 32 bits.
const BLOB_MAX: usize = u32::MAX as usize;

/// The longest single value, because a length has to fit the other 30 bits.
///
/// A gigabyte, and a client cannot send one anyway: `proto-max-bulk-len` caps a
/// bulk string at 512 megabytes, so nothing that gets here can be over this. It
/// is checked rather than assumed because the embedded API does not go through
/// the protocol and can hand over whatever it likes.
const VALUE_MAX: usize = (1 << 30) - 1;

const TAG_MASK: u64 = 0b11;
const TAG_BLOB: u64 = 0;
const TAG_INT: u64 = 1;
const TAG_FLOAT: u64 = 2;
const TAG_STR: u64 = 3;

/// The range of integers that fits the 62 bit payload, which is Redis's
/// `arIntFits`. Anything outside it is kept as the text it arrived as.
const INT_LO: i64 = -(1 << 61);
const INT_HI: i64 = (1 << 61) - 1;

impl Word {
    /// Nothing is stored here.
    const EMPTY: Word = Word(0);

    const fn is_empty(self) -> bool {
        self.0 == 0
    }

    const fn tag(self) -> u64 {
        self.0 & TAG_MASK
    }

    const fn from_int(i: i64) -> Word {
        Word(((i as u64) << 2) | TAG_INT)
    }

    const fn to_int(self) -> i64 {
        // Arithmetic shift, so the sign comes back with it.
        (self.0 as i64) >> 2
    }

    const fn from_float_bits(bits: u64) -> Word {
        Word((bits & !TAG_MASK) | TAG_FLOAT)
    }

    const fn to_float(self) -> f64 {
        f64::from_bits(self.0 & !TAG_MASK)
    }

    fn from_short(s: &[u8]) -> Word {
        let mut v = TAG_STR | ((s.len() as u64) << 2);
        for (i, &b) in s.iter().enumerate() {
            v |= u64::from(b) << (8 * (i + 1));
        }
        Word(v)
    }

    const fn short_len(self) -> usize {
        ((self.0 >> 2) & 0b111) as usize
    }

    fn to_short(self) -> Short {
        let n = self.short_len();
        let mut buf = [0u8; INLINE_MAX];
        for (i, out) in buf.iter_mut().take(n).enumerate() {
            *out = ((self.0 >> (8 * (i + 1))) & 0xff) as u8;
        }
        Short { buf, len: n as u8 }
    }

    const fn from_blob(start: usize, len: usize) -> Word {
        Word(((start as u64) << 32) | ((len as u64) << 2) | TAG_BLOB)
    }

    const fn blob_span(self) -> (usize, usize) {
        let start = (self.0 >> 32) as usize;
        let len = ((self.0 >> 2) & 0x3fff_ffff) as usize;
        (start, len)
    }
}

/// The one slice of the index space that holds anything, in whichever layout
/// fits it.
#[derive(Debug, Clone)]
struct Slice {
    /// How many of the words are populated. Never zero: an empty slice is
    /// dropped rather than kept, so that `len` does not have to walk past any.
    count: u16,
    layout: Layout,
}

/// The two ways a slice holds its words.
///
/// Both carry a `Vec<Word>`, which is what lets blob compaction walk every word
/// in the array without caring which layout it is looking at.
#[derive(Debug, Clone)]
enum Layout {
    /// Offsets and words in parallel, sorted by offset, binary searched.
    ///
    /// Ten bytes an element and no cost for the holes, which is what a slice
    /// with a handful of scattered elements wants.
    Sparse { offs: Vec<u16>, words: Vec<Word> },
    /// A window of consecutive positions, with `offset` the position of
    /// `words[0]`.
    ///
    /// Eight bytes a position including the holes, so this only wins when the
    /// populated positions are close together. The window is kept trimmed of
    /// leading and trailing empties, which is what makes the highest populated
    /// offset derivable rather than stored.
    Dense { offset: u16, words: Vec<Word> },
}

impl Slice {
    /// Every word in the slice, for the one walk that rewrites them all: blob
    /// compaction.
    fn words_mut(&mut self) -> &mut [Word] {
        match &mut self.layout {
            Layout::Sparse { words, .. } | Layout::Dense { words, .. } => words,
        }
    }

    /// The same walk, for the one that only reads: freezing the blob.
    fn words(&self) -> &[Word] {
        match &self.layout {
            Layout::Sparse { words, .. } | Layout::Dense { words, .. } => words,
        }
    }

    /// The highest populated offset. The slice is never empty, so there is one.
    fn high(&self) -> u16 {
        match &self.layout {
            Layout::Sparse { offs, .. } => *offs.last().expect("a slice is never empty"),
            // Trimmed, so the last word is populated.
            Layout::Dense { offset, words } => offset + (words.len() as u16) - 1,
        }
    }

    fn get(&self, off: u16) -> Word {
        match &self.layout {
            Layout::Sparse { offs, words } => match offs.binary_search(&off) {
                Ok(at) => words[at],
                Err(_) => Word::EMPTY,
            },
            Layout::Dense { offset, words } => {
                if off < *offset {
                    return Word::EMPTY;
                }
                let at = usize::from(off - offset);
                words.get(at).copied().unwrap_or(Word::EMPTY)
            }
        }
    }

    /// Writes a word, and answers with what was there before.
    fn put(&mut self, off: u16, w: Word) -> Word {
        let old = match &mut self.layout {
            Layout::Sparse { offs, words } => match offs.binary_search(&off) {
                Ok(at) => std::mem::replace(&mut words[at], w),
                Err(at) => {
                    offs.insert(at, off);
                    words.insert(at, w);
                    Word::EMPTY
                }
            },
            Layout::Dense { offset, words } => {
                if off < *offset {
                    // Growing downwards moves the window, which is a memmove
                    // bounded by the slice: 32 KiB at the very worst and zero on
                    // the ascending writes that are what an array is normally
                    // filled by.
                    let gap = usize::from(*offset - off);
                    words.splice(0..0, std::iter::repeat_n(Word::EMPTY, gap));
                    *offset = off;
                    std::mem::replace(&mut words[0], w)
                } else {
                    let at = usize::from(off - *offset);
                    if at >= words.len() {
                        words.resize(at + 1, Word::EMPTY);
                    }
                    std::mem::replace(&mut words[at], w)
                }
            }
        };
        if old.is_empty() {
            self.count += 1;
        }
        old
    }

    /// Clears a position, and answers with what was there.
    fn take(&mut self, off: u16) -> Word {
        let old = match &mut self.layout {
            Layout::Sparse { offs, words } => match offs.binary_search(&off) {
                Ok(at) => {
                    offs.remove(at);
                    words.remove(at)
                }
                Err(_) => Word::EMPTY,
            },
            Layout::Dense { offset, words } => {
                if off < *offset {
                    Word::EMPTY
                } else {
                    let at = usize::from(off - *offset);
                    match words.get_mut(at) {
                        Some(slot) => std::mem::replace(slot, Word::EMPTY),
                        None => Word::EMPTY,
                    }
                }
            }
        };
        if !old.is_empty() {
            self.count -= 1;
            self.trim();
        }
        old
    }

    /// Drops the empties off both ends of a dense window.
    ///
    /// This is what keeps [`Slice::high`] derivable, and it is also why a dense
    /// slice that has been emptied from the middle out does not keep paying for
    /// the positions nobody is using any more.
    fn trim(&mut self) {
        let Layout::Dense { offset, words } = &mut self.layout else {
            return;
        };
        while words.last().is_some_and(|w| w.is_empty()) {
            words.pop();
        }
        let lead = words.iter().take_while(|w| w.is_empty()).count();
        if lead > 0 {
            words.drain(..lead);
            *offset += lead as u16;
        }
    }

    /// The number of positions a dense window would have to cover.
    fn span(&self) -> usize {
        match &self.layout {
            Layout::Sparse { offs, .. } => match (offs.first(), offs.last()) {
                (Some(lo), Some(hi)) => usize::from(hi - lo) + 1,
                _ => 0,
            },
            Layout::Dense { words, .. } => words.len(),
        }
    }

    /// Moves the slice to whichever layout now fits it.
    ///
    /// Redis promotes on the count alone, at more than ten elements, which can
    /// put eleven elements in a thirty two kilobyte window. Promotion is a
    /// memory question and not a count question, so this asks about the span
    /// too: dense is eight bytes a position and sparse is ten bytes an element,
    /// so dense only wins while the positions are within about twice the count
    /// of each other.
    fn rebalance(&mut self) {
        let count = usize::from(self.count);
        match &self.layout {
            Layout::Sparse { .. } => {
                if count > SPARSE_MAX && self.span() <= count * 2 {
                    self.make_dense();
                }
            }
            Layout::Dense { .. } => {
                if count <= SPARSE_MIN || self.span() > count * 4 {
                    self.make_sparse();
                }
            }
        }
    }

    fn make_dense(&mut self) {
        let Layout::Sparse { offs, words } = &self.layout else {
            return;
        };
        let base = offs[0];
        let span = self.span();
        let mut window = vec![Word::EMPTY; span];
        for (&off, &w) in offs.iter().zip(words) {
            window[usize::from(off - base)] = w;
        }
        self.layout = Layout::Dense {
            offset: base,
            words: window,
        };
    }

    fn make_sparse(&mut self) {
        let Layout::Dense { offset, words } = &self.layout else {
            return;
        };
        let mut offs = Vec::with_capacity(usize::from(self.count));
        let mut vals = Vec::with_capacity(usize::from(self.count));
        for (i, &w) in words.iter().enumerate() {
            if !w.is_empty() {
                offs.push(offset + (i as u16));
                vals.push(w);
            }
        }
        self.layout = Layout::Sparse { offs, words: vals };
    }

    fn memory_bytes(&self) -> usize {
        match &self.layout {
            Layout::Sparse { offs, words } => {
                offs.capacity() * 2 + words.capacity() * size_of::<Word>()
            }
            Layout::Dense { words, .. } => words.capacity() * size_of::<Word>(),
        }
    }

    /// Hands every populated offset in `from..=to` to `f`, in whichever
    /// direction was asked for, and stops early when `f` says to.
    ///
    /// Both layouts cost what is in the window rather than how wide it is. A
    /// sparse slice binary searches for the first entry in range and walks its
    /// entries, and a dense one walks the part of its window that overlaps and
    /// skips the holes, which is the whole reason a scan of the entire index
    /// space over a key holding three elements is three visits.
    ///
    /// Answers whether the caller should carry on to the next slice.
    fn window<F>(&self, from: u16, to: u16, reverse: bool, f: &mut F) -> bool
    where
        F: FnMut(u16, Word) -> bool,
    {
        match &self.layout {
            Layout::Sparse { offs, words } => {
                // Sorted, and no entry is ever empty, so the two ends of the
                // window are two binary searches and everything between them is
                // a hit.
                let a = offs.partition_point(|&o| o < from);
                let b = offs.partition_point(|&o| o <= to);
                if reverse {
                    for i in (a..b).rev() {
                        if !f(offs[i], words[i]) {
                            return false;
                        }
                    }
                } else {
                    for i in a..b {
                        if !f(offs[i], words[i]) {
                            return false;
                        }
                    }
                }
            }
            Layout::Dense { offset, words } => {
                let base = *offset;
                let end = base + (words.len() as u16) - 1;
                if to < base || from > end {
                    return true;
                }
                let a = usize::from(from.max(base) - base);
                let b = usize::from(to.min(end) - base);
                let window = &words[a..=b];
                let at = |i: usize| base + ((a + i) as u16);
                if reverse {
                    for (i, w) in window.iter().enumerate().rev() {
                        if !w.is_empty() && !f(at(i), *w) {
                            return false;
                        }
                    }
                } else {
                    for (i, w) in window.iter().enumerate() {
                        if !w.is_empty() && !f(at(i), *w) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

/// A sparse array of values, indexed by a `u64`.
#[derive(Debug, Clone, Default)]
pub struct Array {
    /// Populated slices, sorted by slice id and binary searched. No slice in
    /// here is empty.
    slices: Vec<(u64, Slice)>,
    /// Every value of eight bytes or more, back to back.
    blob: Vec<u8>,
    /// How much of the blob is no longer pointed at by any word.
    dead: usize,
    /// How many indices are populated, kept rather than counted because
    /// `ARCOUNT` is documented as O(1).
    count: u64,
    /// The last index `ARINSERT` or `ARRING` wrote to, or none when neither has
    /// written yet.
    ///
    /// This is the array's own cursor and it is nothing to do with where the
    /// elements are. `ARSET` does not move it, so an array filled by `ARSET`
    /// and then appended to with `ARINSERT` gets its first append at index
    /// zero, on top of whatever was there. Redis holds the same thing as a
    /// `u64` with `UINT64_MAX` meaning not set, which is also why that index is
    /// not addressable.
    insert: Option<u64>,
}

/// How dead a blob has to get before it is worth rewriting.
///
/// Half, with a floor so that a small array does not compact on every overwrite.
/// Compaction is one pass over the words and one pass over the live bytes, and
/// paying that when a quarter of a kilobyte is dead would cost more than the
/// kilobyte.
const COMPACT_MIN: usize = 4096;

impl Array {
    /// A new, empty array.
    #[must_use]
    pub fn new() -> Array {
        Array::default()
    }

    /// The highest populated index plus one, which is `ARLEN`.
    ///
    /// Zero for an empty array, and note that this is not the number of
    /// elements. See [`Array::count`] for that.
    #[must_use]
    pub fn len(&self) -> u64 {
        match self.slices.last() {
            Some((id, slice)) => id * SLICE_SIZE + u64::from(slice.high()) + 1,
            None => 0,
        }
    }

    /// How many indices are populated, which is `ARCOUNT`.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Whether anything is stored at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The value at `idx`, or none if that position is a hole.
    ///
    /// A missing key and a hole are the same answer to a client, which is why
    /// there is one `None` here and not two.
    #[must_use]
    pub fn get(&self, idx: u64) -> Option<Element<'_>> {
        let (id, off) = split(idx);
        let at = self.find(id).ok()?;
        let w = self.slices[at].1.get(off);
        self.decode(w)
    }

    /// Writes `val` at `idx`, and answers whether that position was empty.
    ///
    /// The count of newly filled positions is what `ARSET` and `ARMSET` reply
    /// with, so the boolean is the useful return rather than the old value.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] when the blob of long values would pass four gigabytes,
    /// which is a recorded divergence: Redis heap allocates each of those and
    /// has no per key ceiling.
    pub fn set(&mut self, idx: u64, val: &[u8]) -> Result<bool> {
        let w = self.encode(val)?;
        let (id, off) = split(idx);
        let at = match self.find(id) {
            Ok(at) => at,
            Err(at) => {
                self.slices.insert(
                    at,
                    (
                        id,
                        Slice {
                            count: 0,
                            layout: Layout::Sparse {
                                offs: Vec::new(),
                                words: Vec::new(),
                            },
                        },
                    ),
                );
                at
            }
        };
        let old = self.slices[at].1.put(off, w);
        self.slices[at].1.rebalance();
        self.retire(old);
        self.maybe_compact();
        if old.is_empty() {
            self.count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clears `idx`, and answers whether anything was there.
    pub fn del(&mut self, idx: u64) -> bool {
        let (id, off) = split(idx);
        let Ok(at) = self.find(id) else {
            return false;
        };
        let old = self.slices[at].1.take(off);
        if old.is_empty() {
            return false;
        }
        self.retire(old);
        self.count -= 1;
        if self.slices[at].1.count == 0 {
            self.slices.remove(at);
        } else {
            self.slices[at].1.rebalance();
        }
        self.maybe_compact();
        true
    }

    /// Clears every populated index in `lo..=hi`, and answers how many there
    /// were.
    ///
    /// The cost is in the slices the range touches and not in the width of the
    /// range, so `ARDELRANGE k 0 18446744073709551614` on a key holding three
    /// elements is three deletes and not a walk of the index space.
    pub fn delete_range(&mut self, lo: u64, hi: u64) -> u64 {
        if lo > hi {
            return 0;
        }
        let (lo_id, lo_off) = split(lo);
        let (hi_id, hi_off) = split(hi);
        let first = match self.find(lo_id) {
            Ok(at) | Err(at) => at,
        };
        let mut gone = 0;
        let mut at = first;
        while at < self.slices.len() && self.slices[at].0 <= hi_id {
            let id = self.slices[at].0;
            // A slice strictly inside the range loses everything, and the two on
            // the ends lose the part that is in it.
            let from = if id == lo_id { lo_off } else { 0 };
            let to = if id == hi_id {
                hi_off
            } else {
                (SLICE_SIZE - 1) as u16
            };
            gone += self.clear_within(at, from, to);
            if self.slices[at].1.count == 0 {
                self.slices.remove(at);
            } else {
                self.slices[at].1.rebalance();
                at += 1;
            }
        }
        self.count -= gone;
        self.maybe_compact();
        self.maybe_compact_slices();
        gone
    }

    /// The index the next `ARINSERT` would write to, which is `ARNEXT`.
    ///
    /// None when the cursor has run out of space, which happens only after an
    /// `ARSEEK` to the very top: the next append would have nowhere to go, and
    /// Redis answers a null rather than an index it cannot honour.
    #[must_use]
    pub const fn next_index(&self) -> Option<u64> {
        match self.insert {
            None => Some(0),
            Some(i) if i >= INDEX_MAX => None,
            Some(i) => Some(i + 1),
        }
    }

    /// Points the cursor so that the next append lands on `idx`, which is
    /// `ARSEEK`.
    ///
    /// Seeking to zero is not the same as seeking to one less than one: it puts
    /// the cursor back in the state it was in before anything was appended,
    /// which is also the state `ARRING` reads as "do not reshape me".
    pub const fn seek(&mut self, idx: u64) {
        self.insert = if idx == 0 { None } else { Some(idx - 1) };
    }

    /// Appends `values` at consecutive indices from the cursor, which is
    /// `ARINSERT`, and answers where the last one landed.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] with [`INSERT_OVERFLOW`] when the batch would run off
    /// the top of the index space, checked before any of it is written so that
    /// a batch either lands whole or not at all.
    pub fn append<'v>(&mut self, values: impl Iterator<Item = &'v [u8]> + Clone) -> Result<u64> {
        let n = values.clone().count() as u64;
        let over = || Error::new(Code::Invalid, INSERT_OVERFLOW);
        let start = self.next_index().ok_or_else(over)?;
        if n == 0 {
            return Ok(self.insert.unwrap_or(0));
        }
        let last = start.checked_add(n - 1).filter(|l| *l <= INDEX_MAX);
        let last = last.ok_or_else(over)?;
        for (i, v) in values.enumerate() {
            self.set(start + i as u64, v)?;
        }
        self.insert = Some(last);
        Ok(last)
    }

    /// Writes `values` into a ring of `size` positions, which is `ARRING`, and
    /// answers where the last one landed.
    ///
    /// The ring is not a structure, it is an agreement about indices: writes go
    /// to the cursor plus one modulo the size, so a ring of ten holds indices
    /// zero to nine and the eleventh write goes back over the first. Changing
    /// the size between calls is the only expensive case, because the positions
    /// that survive have to be renumbered so that they stay in order, and that
    /// is the `O(N + M)` in the command's complexity.
    ///
    /// # Errors
    ///
    /// Whatever [`Array::set`] can fail with, which is the two size ceilings.
    pub fn ring<'v>(&mut self, size: u64, values: impl Iterator<Item = &'v [u8]>) -> Result<u64> {
        debug_assert!(size > 0, "the caller refuses a size of zero");
        let old_span = self.len();
        // A reshape is needed when the ring shrank, and when it grew after it
        // had already wrapped, because otherwise the next write would go back
        // over the oldest position instead of using the room that was just
        // added. An explicit seek to zero says where the next write goes, so it
        // is honoured rather than reshaped around.
        let keep = if old_span == 0 || size == old_span {
            0
        } else if size < old_span {
            size
        } else if self.insert.is_some() && self.next_cursor() < old_span {
            old_span
        } else {
            0
        };
        if keep > 0 {
            self.rework(old_span, keep)?;
        }

        let mut cursor = self.insert.unwrap_or(0);
        for v in values {
            cursor = self.next_cursor();
            if cursor >= size {
                cursor %= size;
            }
            self.set(cursor, v)?;
            self.insert = Some(cursor);
        }
        Ok(cursor)
    }

    /// Where the next ring write goes before the size is applied to it.
    ///
    /// Wrapping, because a cursor sitting on the last addressable index steps to
    /// the reserved one, and the modulo in [`Array::ring`] brings it back into
    /// the ring anyway. Redis relies on the same wrap.
    const fn next_cursor(&self) -> u64 {
        match self.insert {
            None => 0,
            Some(i) => i.wrapping_add(1),
        }
    }

    /// Renumbers the ring so the positions that survive a size change stay in
    /// order, oldest at zero.
    ///
    /// The walk goes backwards from the cursor and stops at the first hole, so
    /// a ring that somebody has been deleting out of keeps its newest unbroken
    /// run and not a scattering either side of a gap.
    fn rework(&mut self, old_span: u64, keep: u64) -> Result<()> {
        let anchor = match self.insert {
            None => old_span - 1,
            Some(i) => i % old_span,
        };
        let back = |i: u64| if i == 0 { old_span - 1 } else { i - 1 };
        let forward = |i: u64| if i + 1 == old_span { 0 } else { i + 1 };

        let mut kept = 0;
        let mut src = anchor;
        while kept < keep && self.get(src).is_some() {
            kept += 1;
            src = back(src);
        }
        // The walk stopped one past the oldest one it kept.
        src = forward(src);

        let mut fresh = Array::new();
        for dst in 0..kept {
            let mut buf = [0u8; ELEMENT_MAX];
            let el = self.get(src).expect("the walk stopped at the first hole");
            fresh.set(dst, el.text(&mut buf))?;
            src = forward(src);
        }
        fresh.insert = kept.checked_sub(1);
        *self = fresh;
        Ok(())
    }

    /// The last `count` positions from the cursor, which is `ARLASTITEMS`, and
    /// answers how many that turned out to be.
    ///
    /// Positions and not elements, so a hole inside the window is reported as
    /// one, and the walk wraps at the bottom of the array back to the top. `f`
    /// is called oldest first, or newest first when `newest_first`.
    pub fn last_items<F>(&self, count: u64, newest_first: bool, mut f: F) -> u64
    where
        F: FnMut(Option<Element<'_>>),
    {
        let steps = count.min(self.count);
        if steps == 0 {
            return 0;
        }
        let span = self.len();
        // With no cursor the tail of the array is the anchor, which is what
        // makes this answer something sensible for an array nobody has
        // appended to. A cursor past the end of the array is left where it is
        // rather than folded in, so the positions above the array read as the
        // holes they are.
        let anchor = self.insert.unwrap_or(span - 1);
        // The backwards walk is at most two descending runs: from the anchor
        // down towards zero, and then, if it ran out, from the top of the array
        // downwards. Naming the two runs is what lets this answer in
        // chronological order without collecting the positions first.
        let near = steps.min(anchor + 1);
        let wrapped = steps - near;
        let near_lo = anchor - (near - 1);
        let wrapped_lo = span - wrapped;

        let mut emit = |i: u64| f(self.get(i));
        if newest_first {
            (near_lo..=anchor).rev().for_each(&mut emit);
            (wrapped_lo..span).rev().for_each(&mut emit);
        } else {
            (wrapped_lo..span).for_each(&mut emit);
            (near_lo..=anchor).for_each(&mut emit);
        }
        steps
    }

    /// Hands every populated index in `start..=end` to `f`, which is `ARSCAN`.
    ///
    /// Low to high, or high to low when the two ends come the other way round.
    /// `f` answers whether to keep going, which is how `LIMIT` stops the walk
    /// without the walk knowing what a limit is. Holes are skipped rather than
    /// reported, which is the whole difference between this and `ARGETRANGE`,
    /// and it is why this one needs no cap: the cost is the elements it finds
    /// and the slices it has to look in, not the width of the range.
    pub fn scan<F>(&self, start: u64, end: u64, mut f: F)
    where
        F: FnMut(u64, Element<'_>) -> bool,
    {
        let reverse = start > end;
        let (lo, hi) = if reverse { (end, start) } else { (start, end) };
        let (lo_id, lo_off) = split(lo);
        let (hi_id, hi_off) = split(hi);
        let first = match self.find(lo_id) {
            Ok(at) | Err(at) => at,
        };
        let last = match self.find(hi_id) {
            Ok(at) => at + 1,
            Err(at) => at,
        };

        let mut visit = |at: usize| {
            let (id, slice) = &self.slices[at];
            let from = if *id == lo_id { lo_off } else { 0 };
            let to = if *id == hi_id {
                hi_off
            } else {
                (SLICE_SIZE - 1) as u16
            };
            let base = id * SLICE_SIZE;
            slice.window(from, to, reverse, &mut |off, w| {
                let el = self.decode(w).expect("a populated word decodes");
                f(base + u64::from(off), el)
            })
        };
        if reverse {
            for at in (first..last).rev() {
                if !visit(at) {
                    return;
                }
            }
        } else {
            for at in first..last {
                if !visit(at) {
                    return;
                }
            }
        }
    }

    /// What `ARINFO` says about the array.
    ///
    /// The per layout numbers cost a walk of the directory and are only filled
    /// in when `full`, which is the same split Redis makes and for the same
    /// reason: the seven cheap numbers are all read off fields.
    #[must_use]
    pub fn info(&self, full: bool) -> Info {
        let mut info = Info {
            count: self.count,
            len: self.len(),
            // The terminal cursor reports zero here rather than the null
            // `ARNEXT` gives, which is Redis's choice and not ours.
            next_insert: self.next_index().unwrap_or(0),
            slices: self.slices.len() as u64,
            directory_size: self.slices.capacity() as u64,
            slice_size: SLICE_SIZE,
            ..Info::default()
        };
        if !full {
            return info;
        }
        let (mut window, mut filled, mut room) = (0u64, 0u64, 0u64);
        for (_, slice) in &self.slices {
            match &slice.layout {
                Layout::Dense { words, .. } => {
                    info.dense_slices += 1;
                    window += words.len() as u64;
                    filled += u64::from(slice.count);
                }
                Layout::Sparse { offs, .. } => {
                    info.sparse_slices += 1;
                    room += offs.capacity() as u64;
                }
            }
        }
        let ratio = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        info.avg_dense_size = ratio(window, info.dense_slices);
        info.avg_dense_fill = ratio(filled, window);
        info.avg_sparse_size = ratio(room, info.sparse_slices);
        info
    }

    /// What the array is holding on the heap, for `MEMORY USAGE`.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.slices.capacity() * size_of::<(u64, Slice)>()
            + self
                .slices
                .iter()
                .map(|(_, s)| s.memory_bytes())
                .sum::<usize>()
            + self.blob.capacity()
    }

    /// Write this array out in a form a device can hold.
    ///
    /// The directory goes out as it stands, slice by slice and word by word,
    /// rather than as the index and value pairs a client would see. Rebuilding
    /// from pairs would go through [`Array::set`], and the layout a slice ends up
    /// in depends on the order it was written in as well as on what is in it, so
    /// a slice that had been filled and partly emptied would come back sparse
    /// where it went out dense. `ARINFO` reports that split, so an array whose
    /// layout changed because it was quiet long enough to be demoted would be an
    /// array whose answers depend on memory pressure.
    ///
    /// The blob is written live bytes only, in the order the words are walked in,
    /// so a demotion is also a compaction and the dead space does not reach the
    /// device. It goes in front of the directory because a word carries where its
    /// value starts, and reading the blob first is what lets every one of those
    /// be checked as it arrives rather than in a second pass.
    pub fn freeze(&self, out: &mut Vec<u8>) {
        out.push(match self.insert {
            Some(_) => FORM_SLICES | HAS_INSERT,
            None => FORM_SLICES,
        });
        if let Some(at) = self.insert {
            frozen::put_uint(out, at);
        }
        frozen::put_uint(out, self.count);

        // The live length is known without a walk, which is the same subtraction
        // `compact` sizes its fresh blob with.
        frozen::put_uint(out, (self.blob.len() - self.dead) as u64);
        for (_, slice) in &self.slices {
            for w in slice.words() {
                if !w.is_empty() && w.tag() == TAG_BLOB {
                    let (start, len) = w.blob_span();
                    out.extend_from_slice(&self.blob[start..start + len]);
                }
            }
        }

        frozen::put_uint(out, self.slices.len() as u64);
        // The same walk again, in the same order, so a value's new start is the
        // running total of what went before it and no table has to be kept.
        let mut at = 0usize;
        for (id, slice) in &self.slices {
            frozen::put_uint(out, *id);
            match &slice.layout {
                Layout::Sparse { offs, words } => {
                    out.push(LAYOUT_SPARSE);
                    frozen::put_uint(out, words.len() as u64);
                    for (&off, &w) in offs.iter().zip(words) {
                        frozen::put_uint(out, u64::from(off));
                        frozen::put_uint(out, moved(w, &mut at));
                    }
                }
                Layout::Dense { offset, words } => {
                    out.push(LAYOUT_DENSE);
                    frozen::put_uint(out, u64::from(*offset));
                    frozen::put_uint(out, words.len() as u64);
                    for &w in words {
                        frozen::put_uint(out, moved(w, &mut at));
                    }
                }
            }
        }
    }

    /// Read back what [`Array::freeze`] wrote.
    ///
    /// Everything the rest of this file takes for granted is checked here, since
    /// this is the one way a directory arrives without having been built by
    /// [`Array::set`]: offsets inside a slice go up and stay under
    /// [`SLICE_SIZE`], a sparse slice holds no holes, a dense window has a
    /// populated word at each end, the slice ids go up, the counts add up to the
    /// array's own, and every value in the blob is pointed at by exactly one
    /// word. A body that fails any of them is an error, because the alternative
    /// is an `ARGET` that reads off the end of the blob.
    pub fn thaw(bytes: &[u8]) -> core::result::Result<Array, Broken> {
        let mut cut = frozen::Cut::new(bytes);
        let tag = cut.byte()?;
        if tag & !HAS_INSERT != FORM_SLICES {
            return Err(Broken::Form);
        }
        let insert = if tag & HAS_INSERT != 0 {
            let at = cut.uint()?;
            if at > INDEX_MAX {
                return Err(Broken::Body);
            }
            Some(at)
        } else {
            None
        };
        let count = cut.uint()?;
        let blob = cut.bytes()?.to_vec();

        let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
        // A slice costs an id, a layout byte and a length, so a count larger
        // than what is left cannot be honest and is not worth an allocation.
        if n > cut.rest().len() {
            return Err(Broken::Body);
        }
        let mut slices: Vec<(u64, Slice)> = Vec::with_capacity(n);
        let mut used = 0usize;
        let mut seen = 0u64;
        for _ in 0..n {
            let id = cut.uint()?;
            if id > INDEX_MAX >> SLICE_BITS {
                return Err(Broken::Body);
            }
            if slices.last().is_some_and(|(last, _)| id <= *last) {
                return Err(Broken::Body);
            }
            let slice = read_slice(&mut cut, blob.len(), &mut used)?;
            seen += u64::from(slice.count);
            slices.push((id, slice));
        }
        if seen != count || used != blob.len() {
            return Err(Broken::Body);
        }
        Ok(Array {
            slices,
            blob,
            dead: 0,
            count,
            insert,
        })
    }

    /// Which entry holds slice `id`, or where it would be inserted.
    fn find(&self, id: u64) -> core::result::Result<usize, usize> {
        self.slices.binary_search_by(|(have, _)| {
            if *have < id {
                Ordering::Less
            } else if *have > id {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
    }

    /// Clears `from..=to` inside one slice, and answers how many went.
    ///
    /// The dead blob bytes are counted up here and added to the array's total
    /// afterwards, rather than collected into a list of words and retired one by
    /// one, because the list would be an allocation on the delete path and the
    /// only thing a retired word has left to say is how many bytes it held.
    fn clear_within(&mut self, at: usize, from: u16, to: u16) -> u64 {
        let slice = &mut self.slices[at].1;
        let mut gone = 0u64;
        let mut dead = 0usize;
        match &mut slice.layout {
            Layout::Sparse { offs, words } => {
                let lo = offs.partition_point(|&o| o < from);
                let hi = offs.partition_point(|&o| o <= to);
                // Every word a sparse slice holds is populated, so the range is
                // the count.
                for w in &words[lo..hi] {
                    gone += 1;
                    if w.tag() == TAG_BLOB {
                        dead += w.blob_span().1;
                    }
                }
                offs.drain(lo..hi);
                words.drain(lo..hi);
            }
            Layout::Dense { offset, words } => {
                let base = *offset;
                let lo = usize::from(from.saturating_sub(base));
                if to >= base && lo < words.len() {
                    let hi = usize::from(to - base).min(words.len() - 1);
                    for w in &mut words[lo..=hi] {
                        if !w.is_empty() {
                            gone += 1;
                            if w.tag() == TAG_BLOB {
                                dead += w.blob_span().1;
                            }
                            *w = Word::EMPTY;
                        }
                    }
                }
            }
        }
        slice.count -= gone as u16;
        slice.trim();
        self.dead += dead;
        gone
    }

    /// Gives back whatever blob space a word was using.
    fn retire(&mut self, w: Word) {
        if !w.is_empty() && w.tag() == TAG_BLOB {
            self.dead += w.blob_span().1;
        }
    }

    /// Turns bytes into the smallest word that holds them.
    fn encode(&mut self, val: &[u8]) -> Result<Word> {
        if let Some(i) = num::parse_i64(val)
            && (INT_LO..=INT_HI).contains(&i)
        {
            return Ok(Word::from_int(i));
        }
        if let Some(w) = float_word(val) {
            return Ok(w);
        }
        if val.len() <= INLINE_MAX {
            return Ok(Word::from_short(val));
        }
        if val.len() > VALUE_MAX {
            return Err(Error::new(Code::Full, VALUE_TOO_LONG));
        }
        if self.blob.len() + val.len() > BLOB_MAX {
            self.compact();
        }
        if self.blob.len() + val.len() > BLOB_MAX {
            return Err(Error::new(Code::Full, BLOB_TOO_LONG));
        }
        let start = self.blob.len();
        self.blob.extend_from_slice(val);
        Ok(Word::from_blob(start, val.len()))
    }

    fn decode(&self, w: Word) -> Option<Element<'_>> {
        if w.is_empty() {
            return None;
        }
        Some(match w.tag() {
            TAG_INT => Element::Int(w.to_int()),
            TAG_FLOAT => Element::Float(w.to_float()),
            TAG_STR => Element::Short(w.to_short()),
            _ => {
                let (start, len) = w.blob_span();
                Element::Str(&self.blob[start..start + len])
            }
        })
    }

    /// Rewrites the blob with the dead bytes gone, and points every word at
    /// where its value landed.
    fn compact(&mut self) {
        let mut fresh = Vec::with_capacity(self.blob.len() - self.dead);
        for (_, slice) in &mut self.slices {
            for w in slice.words_mut() {
                if w.is_empty() || w.tag() != TAG_BLOB {
                    continue;
                }
                let (start, len) = w.blob_span();
                let to = fresh.len();
                fresh.extend_from_slice(&self.blob[start..start + len]);
                *w = Word::from_blob(to, len);
            }
        }
        self.blob = fresh;
        self.dead = 0;
    }

    fn maybe_compact(&mut self) {
        if self.dead >= COMPACT_MIN && self.dead * 2 >= self.blob.len() {
            self.compact();
        }
    }

    /// Gives back the directory space a mass delete freed.
    ///
    /// `Vec::remove` leaves the capacity behind, and a key that had a million
    /// slices and now has one should not still be holding sixteen megabytes of
    /// directory.
    fn maybe_compact_slices(&mut self) {
        if self.slices.capacity() > 16 && self.slices.capacity() > self.slices.len() * 4 {
            self.slices.shrink_to_fit();
        }
    }
}

/// What `ARINFO` reports, which is the shape of the array and not its contents.
///
/// Three of these describe our directory rather than Redis's, which is D-20:
/// the slice count and the two directory numbers are about a sorted vector of
/// slices where Redis has a growable table and, past a point, a second level
/// above it. Everything else means the same thing in both.
#[derive(Debug, Default, Clone, Copy)]
pub struct Info {
    /// How many indices hold something.
    pub count: u64,
    /// The highest populated index plus one.
    pub len: u64,
    /// Where the next append would go, and zero when there is nowhere.
    pub next_insert: u64,
    /// How many slices the array is made of.
    pub slices: u64,
    /// How many slots the directory has room for.
    pub directory_size: u64,
    /// How many indices one slice covers.
    pub slice_size: u64,
    /// How many slices are holding a window of consecutive positions.
    pub dense_slices: u64,
    /// How many are holding offsets and words in parallel.
    pub sparse_slices: u64,
    /// The mean width of a dense window, in positions.
    pub avg_dense_size: f64,
    /// How much of that width is populated, between zero and one.
    pub avg_dense_fill: f64,
    /// The mean number of entries a sparse slice has room for.
    pub avg_sparse_size: f64,
}

/// The message when one key's long values pass four gigabytes in total.
pub const BLOB_TOO_LONG: &str = "array values exceed the four gigabyte per key limit";

/// The message when one value on its own passes a gigabyte.
pub const VALUE_TOO_LONG: &str = "array value exceeds the one gigabyte limit";

/// What `ARINSERT` says when the cursor has nowhere left to go.
///
/// Redis's words. This is not the same error `ARSET` gives for the same
/// underlying problem, which is Redis's doing and worth keeping: one of them is
/// about an index the client named and the other is about a cursor it did not.
pub const INSERT_OVERFLOW: &str = "insert index overflow";

/// A word as it should be written, with a blob value pointed at where it is
/// about to land rather than where it used to be.
///
/// `at` is the running length of the frozen blob, so this is the compaction
/// `Array::compact` does, run against a buffer that has already been written
/// instead of against a fresh `Vec`.
fn moved(w: Word, at: &mut usize) -> u64 {
    if w.is_empty() || w.tag() != TAG_BLOB {
        return w.0;
    }
    let (_, len) = w.blob_span();
    let start = *at;
    *at += len;
    Word::from_blob(start, len).0
}

/// A word read back, checked against the blob it may be pointing into.
///
/// `used` counts the blob bytes claimed so far, which the caller compares with
/// the blob's length at the end. Two words pointing at the same bytes, or a
/// value nothing points at, both fail that comparison.
fn read_word(
    cut: &mut frozen::Cut<'_>,
    blob: usize,
    used: &mut usize,
) -> core::result::Result<Word, Broken> {
    let w = Word(cut.uint()?);
    if !w.is_empty() && w.tag() == TAG_BLOB {
        let (start, len) = w.blob_span();
        // A blob word is only written for a value of `INLINE_MAX` and up, so a
        // shorter one is a word that was not written by `freeze`.
        if len <= INLINE_MAX || start + len > blob {
            return Err(Broken::Body);
        }
        *used += len;
    }
    Ok(w)
}

/// One slice read back, in whichever layout it says it is in.
fn read_slice(
    cut: &mut frozen::Cut<'_>,
    blob: usize,
    used: &mut usize,
) -> core::result::Result<Slice, Broken> {
    match cut.byte()? {
        LAYOUT_SPARSE => {
            let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
            // An offset and a word are a byte each at the very least, and a
            // slice with nothing in it is dropped rather than kept.
            if n == 0 || n > cut.rest().len() {
                return Err(Broken::Body);
            }
            let mut offs: Vec<u16> = Vec::with_capacity(n);
            let mut words = Vec::with_capacity(n);
            for _ in 0..n {
                let off = u16::try_from(cut.uint()?).map_err(|_| Broken::Body)?;
                if u64::from(off) >= SLICE_SIZE {
                    return Err(Broken::Body);
                }
                if offs.last().is_some_and(|last| off <= *last) {
                    return Err(Broken::Body);
                }
                let w = read_word(cut, blob, used)?;
                // Sparse holds what is there and nothing else, so an empty word
                // in here would make `count` and the entries disagree.
                if w.is_empty() {
                    return Err(Broken::Body);
                }
                offs.push(off);
                words.push(w);
            }
            Ok(Slice {
                count: n as u16,
                layout: Layout::Sparse { offs, words },
            })
        }
        LAYOUT_DENSE => {
            let offset = u16::try_from(cut.uint()?).map_err(|_| Broken::Body)?;
            let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
            if n == 0 || n > cut.rest().len() {
                return Err(Broken::Body);
            }
            if u64::from(offset) + n as u64 > SLICE_SIZE {
                return Err(Broken::Body);
            }
            let mut words = Vec::with_capacity(n);
            let mut live = 0u16;
            for _ in 0..n {
                let w = read_word(cut, blob, used)?;
                if !w.is_empty() {
                    live += 1;
                }
                words.push(w);
            }
            // Trimmed at both ends, which is what makes `Slice::high` derivable
            // rather than stored.
            if words[0].is_empty() || words[n - 1].is_empty() {
                return Err(Broken::Body);
            }
            Ok(Slice {
                count: live,
                layout: Layout::Dense { offset, words },
            })
        }
        _ => Err(Broken::Form),
    }
}

/// Splits an index into the slice that holds it and the offset inside.
#[inline]
const fn split(idx: u64) -> (u64, u16) {
    (idx >> SLICE_BITS, (idx & (SLICE_SIZE - 1)) as u16)
}

/// Whether these bytes round trip exactly through an inline double.
///
/// This is `arTryEncodeFloat`, and the round trip is the point rather than an
/// optimisation. `3.140` parses to the same double as `3.14` and prints back as
/// `3.14`, so storing it as a number would change what the client wrote. Only a
/// value that prints back byte for byte is allowed to become a number, and
/// everything else stays a string.
fn float_word(val: &[u8]) -> Option<Word> {
    // The cheap filter first: optional minus, then digits with exactly one dot.
    // Nothing else can survive the round trip, and this skips the parse for the
    // overwhelming majority of values that are not numbers at all.
    let body = match val.first() {
        Some(b'-') if val.len() > 1 => &val[1..],
        Some(_) => val,
        None => return None,
    };
    let mut dots = 0;
    for &c in body {
        match c {
            b'.' => dots += 1,
            b'0'..=b'9' => {}
            _ => return None,
        }
    }
    if dots != 1 {
        return None;
    }

    let d = num::parse_f64(val)?;
    if !d.is_finite() {
        return None;
    }
    // The low two bits of the payload are the tag, so the value that gets stored
    // is the input with those cleared, and it is that value that has to print
    // back to the input. Most decimals do not survive that, which is the design
    // working: `3.14` loses three units in the last place and prints back as
    // something else, so it stays a string.
    let trunc = f64::from_bits(d.to_bits() & !TAG_MASK);
    let mut buf = [0u8; ELEMENT_MAX];
    let el = Element::Float(trunc);
    if el.text(&mut buf) == val {
        Some(Word::from_float_bits(trunc.to_bits()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes a client would see for whatever is at `idx`.
    fn read(a: &Array, idx: u64) -> Option<Vec<u8>> {
        let el = a.get(idx)?;
        let mut buf = [0u8; ELEMENT_MAX];
        Some(el.text(&mut buf).to_vec())
    }

    fn set(a: &mut Array, idx: u64, val: &[u8]) -> bool {
        a.set(idx, val).expect("a value that fits")
    }

    /// Everything the scan finds, as index and bytes.
    fn scan(a: &Array, start: u64, end: u64, limit: usize) -> Vec<(u64, Vec<u8>)> {
        let mut got = Vec::new();
        a.scan(start, end, |i, el| {
            let mut buf = [0u8; ELEMENT_MAX];
            got.push((i, el.text(&mut buf).to_vec()));
            got.len() < limit
        });
        got
    }

    /// What `ARLASTITEMS` would reply, holes included.
    fn last(a: &Array, count: u64, newest_first: bool) -> Vec<Option<Vec<u8>>> {
        let mut got = Vec::new();
        let n = a.last_items(count, newest_first, |el| {
            got.push(el.map(|e| {
                let mut buf = [0u8; ELEMENT_MAX];
                e.text(&mut buf).to_vec()
            }));
        });
        assert_eq!(n as usize, got.len(), "the count is what it emitted");
        got
    }

    fn append(a: &mut Array, vals: &[&[u8]]) -> Result<u64> {
        a.append(vals.iter().copied())
    }

    fn ring(a: &mut Array, size: u64, vals: &[&[u8]]) -> u64 {
        a.ring(size, vals.iter().copied()).expect("values that fit")
    }

    #[test]
    fn a_value_comes_back_the_way_it_went_in() {
        let mut a = Array::new();
        assert!(set(&mut a, 0, b"hello"));
        assert!(set(&mut a, 1, b"a much longer value than fits in a word"));
        assert!(set(&mut a, 2, b"42"));
        assert!(set(&mut a, 3, b"1.5"));
        assert!(set(&mut a, 4, b""));

        assert_eq!(read(&a, 0).as_deref(), Some(&b"hello"[..]));
        assert_eq!(
            read(&a, 1).as_deref(),
            Some(&b"a much longer value than fits in a word"[..])
        );
        assert_eq!(read(&a, 2).as_deref(), Some(&b"42"[..]));
        assert_eq!(read(&a, 3).as_deref(), Some(&b"1.5"[..]));
        assert_eq!(read(&a, 4).as_deref(), Some(&b""[..]));
        assert_eq!(read(&a, 5), None);
    }

    /// The length and the count are different numbers, and this is the test
    /// that says so.
    #[test]
    fn the_length_is_the_high_water_mark_and_the_count_is_the_population() {
        let mut a = Array::new();
        assert_eq!(a.len(), 0);
        assert_eq!(a.count(), 0);
        assert!(a.is_empty());

        set(&mut a, 1_000_000, b"x");
        assert_eq!(a.len(), 1_000_001);
        assert_eq!(a.count(), 1);
        assert!(!a.is_empty());

        set(&mut a, 5, b"y");
        assert_eq!(a.len(), 1_000_001, "a lower index does not move the length");
        assert_eq!(a.count(), 2);

        a.del(1_000_000);
        assert_eq!(a.len(), 6, "and the length comes back down when it goes");
        assert_eq!(a.count(), 1);
    }

    /// Writing over a position is not a new position.
    #[test]
    fn an_overwrite_does_not_count_as_a_fill() {
        let mut a = Array::new();
        assert!(set(&mut a, 7, b"first"));
        assert!(!set(&mut a, 7, b"second"));
        assert_eq!(a.count(), 1);
        assert_eq!(read(&a, 7).as_deref(), Some(&b"second"[..]));
    }

    #[test]
    fn deleting_the_last_element_leaves_nothing_behind() {
        let mut a = Array::new();
        set(&mut a, 3, b"x");
        assert!(a.del(3));
        assert!(!a.del(3), "and a second delete finds nothing");
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
        assert!(a.slices.is_empty(), "the slice went with the last element");
    }

    /// The whole 64 bit index space, not just the part a `Vec` could index.
    #[test]
    fn the_index_space_runs_to_the_top() {
        let mut a = Array::new();
        set(&mut a, 0, b"low");
        set(&mut a, INDEX_MAX, b"high");
        assert_eq!(read(&a, INDEX_MAX).as_deref(), Some(&b"high"[..]));
        assert_eq!(a.count(), 2);
        assert_eq!(a.len(), u64::MAX, "the highest index plus one");
        // And it is two slices, not four thousand billion of them.
        assert_eq!(a.slices.len(), 2);
    }

    /// A slice earns a dense window by being full enough to want one, and gives
    /// it back when it is not.
    #[test]
    fn a_slice_changes_layout_when_the_shape_of_it_changes() {
        let mut a = Array::new();
        for i in 0..SPARSE_MAX as u64 {
            set(&mut a, i, b"x");
        }
        assert!(
            matches!(a.slices[0].1.layout, Layout::Sparse { .. }),
            "ten scattered elements do not want an index"
        );

        set(&mut a, 10, b"x");
        assert!(
            matches!(a.slices[0].1.layout, Layout::Dense { .. }),
            "eleven consecutive ones do"
        );

        // Spread the same elements out and the window stops being worth it.
        for i in 0..8 {
            a.del(i);
        }
        assert!(
            matches!(a.slices[0].1.layout, Layout::Sparse { .. }),
            "three left is under the floor"
        );
        assert_eq!(a.count(), 3);
        assert_eq!(read(&a, 10).as_deref(), Some(&b"x"[..]));
    }

    /// Eleven elements spread across a slice stay sparse, which is where this
    /// parts company with Redis.
    #[test]
    fn a_wide_slice_stays_sparse_however_many_elements_it_has() {
        let mut a = Array::new();
        for i in 0..40 {
            set(&mut a, i * 100, b"x");
        }
        assert!(
            matches!(a.slices[0].1.layout, Layout::Sparse { .. }),
            "forty elements over four thousand positions is not a window"
        );
        for i in 0..40 {
            assert_eq!(read(&a, i * 100).as_deref(), Some(&b"x"[..]), "at {i}");
        }
    }

    /// A dense window is filled from the top down as well as the bottom up.
    #[test]
    fn a_dense_window_grows_downwards_too() {
        let mut a = Array::new();
        for i in (0..20u64).rev() {
            set(&mut a, i, b"v");
        }
        assert!(matches!(a.slices[0].1.layout, Layout::Dense { .. }));
        for i in 0..20 {
            assert_eq!(read(&a, i).as_deref(), Some(&b"v"[..]), "at {i}");
        }
        assert_eq!(a.count(), 20);
        assert_eq!(a.len(), 20);
    }

    #[test]
    fn a_range_delete_costs_what_it_touches_and_not_what_it_spans() {
        let mut a = Array::new();
        set(&mut a, 1, b"a");
        set(&mut a, 500_000, b"b");
        set(&mut a, INDEX_MAX, b"c");

        // The widest range there is, against three elements.
        assert_eq!(a.delete_range(0, INDEX_MAX), 3);
        assert!(a.is_empty());
        assert!(a.slices.is_empty());
        assert_eq!(a.delete_range(0, INDEX_MAX), 0, "and again finds nothing");
    }

    #[test]
    fn a_range_delete_takes_the_ends_and_leaves_the_rest() {
        let mut a = Array::new();
        for i in 0..30_000u64 {
            set(&mut a, i, b"x");
        }
        assert_eq!(a.delete_range(100, 29_899), 29_800);
        assert_eq!(a.count(), 200);
        assert_eq!(read(&a, 99).as_deref(), Some(&b"x"[..]));
        assert_eq!(read(&a, 100), None);
        assert_eq!(read(&a, 29_899), None);
        assert_eq!(read(&a, 29_900).as_deref(), Some(&b"x"[..]));
        assert_eq!(a.len(), 30_000);
    }

    #[test]
    fn a_backwards_range_deletes_nothing() {
        let mut a = Array::new();
        set(&mut a, 5, b"x");
        assert_eq!(a.delete_range(9, 4), 0);
        assert_eq!(a.count(), 1);
    }

    /// A value that is an integer is held as one, and one that is not is not.
    ///
    /// This is the compatibility requirement rather than an implementation
    /// detail: `007` is not the number seven, because it does not print back as
    /// `007`, and an implementation that normalised it would hand a client
    /// different bytes than it was given.
    #[test]
    fn only_a_value_that_prints_back_the_same_becomes_a_number() {
        let cases: &[(&[u8], bool)] = &[
            (b"0", true),
            (b"42", true),
            (b"-42", true),
            (b"9007199254740993", true),
            (b"007", false),
            (b"+7", false),
            (b"-0", false),
            (b" 7", false),
            (b"7 ", false),
            (b"", false),
        ];
        for &(val, want) in cases {
            let mut a = Array::new();
            set(&mut a, 0, val);
            let is_int = matches!(a.get(0), Some(Element::Int(_)));
            assert_eq!(is_int, want, "{}", String::from_utf8_lossy(val));
            assert_eq!(read(&a, 0).as_deref(), Some(val), "round trip");
        }
    }

    /// The same rule for the doubles, and it rejects most of them.
    #[test]
    fn only_a_double_that_prints_back_the_same_is_stored_as_one() {
        let cases: &[(&[u8], bool)] = &[
            (b"1.0", true),
            (b"1.5", true),
            (b"-2.25", true),
            (b"0.0", true),
            // Three units in the last place go missing when the tag bits are
            // cleared, so this one prints back as something else.
            (b"3.14", false),
            (b"1.10", false),
            // Negative zero keeps its sign through the printer, so `-0` comes
            // back and `arFormatFloat` puts the `.0` on the end of it.
            (b"-0.0", true),
            (b"1.", false),
            (b".5", false),
            (b"1e5", false),
            (b"nan", false),
            (b"inf", false),
        ];
        for &(val, want) in cases {
            let mut a = Array::new();
            set(&mut a, 0, val);
            let is_float = matches!(a.get(0), Some(Element::Float(_)));
            assert_eq!(is_float, want, "{}", String::from_utf8_lossy(val));
            assert_eq!(read(&a, 0).as_deref(), Some(val), "round trip");
        }
    }

    /// Long values live in one blob, and the blob gets rewritten when enough of
    /// it is dead.
    #[test]
    fn the_blob_is_compacted_once_enough_of_it_is_dead() {
        let mut a = Array::new();
        let long = vec![b'a'; 64];
        for i in 0..1000 {
            set(&mut a, i, &long);
        }
        let full = a.blob.len();
        assert_eq!(full, 64_000);

        // Overwrite every one of them with a value that does not need the blob.
        for i in 0..1000 {
            set(&mut a, i, b"short");
        }
        assert!(a.blob.len() < full / 2, "{} bytes left", a.blob.len());
        assert_eq!(a.count(), 1000);
        for i in 0..1000 {
            assert_eq!(read(&a, i).as_deref(), Some(&b"short"[..]), "at {i}");
        }
    }

    /// Compaction moves the live bytes, so every word that pointed into the
    /// blob has to be moved with them.
    #[test]
    fn compaction_keeps_the_values_that_survive_it() {
        let mut a = Array::new();
        for i in 0..2000u64 {
            let val = format!("value number {i} padded out past the inline limit");
            set(&mut a, i, val.as_bytes());
        }
        // Kill the even ones, which is enough dead bytes to trigger a rewrite.
        for i in (0..2000u64).step_by(2) {
            a.del(i);
        }
        assert!(a.dead * 2 < a.blob.len(), "the blob was rewritten");
        for i in (1..2000u64).step_by(2) {
            let want = format!("value number {i} padded out past the inline limit");
            assert_eq!(read(&a, i).as_deref(), Some(want.as_bytes()), "at {i}");
        }
    }

    #[test]
    fn a_value_over_the_ceiling_is_an_error_and_not_a_panic() {
        let mut a = Array::new();
        let huge = vec![b'x'; VALUE_MAX + 1];
        let e = a.set(0, &huge).unwrap_err();
        assert_eq!(e.code(), Code::Full);
        assert_eq!(e.message(), VALUE_TOO_LONG);
        assert!(a.is_empty(), "and nothing was written");
    }

    /// Every word encoding, through the eight bytes and back.
    #[test]
    fn a_word_holds_what_it_was_given() {
        assert!(Word::EMPTY.is_empty());
        for i in [0i64, 1, -1, INT_LO, INT_HI, 12345, -99999] {
            let w = Word::from_int(i);
            assert!(!w.is_empty());
            assert_eq!(w.tag(), TAG_INT);
            assert_eq!(w.to_int(), i, "{i}");
        }
        for d in [0.0f64, 1.5, -2.25, 1e300] {
            let bits = d.to_bits() & !TAG_MASK;
            let w = Word::from_float_bits(bits);
            assert!(!w.is_empty());
            assert_eq!(w.tag(), TAG_FLOAT);
            assert_eq!(w.to_float().to_bits(), bits);
        }
        for s in [&b""[..], b"a", b"abc", b"1234567"] {
            let w = Word::from_short(s);
            assert!(!w.is_empty(), "{s:?}");
            assert_eq!(w.tag(), TAG_STR);
            assert_eq!(w.to_short().as_bytes(), s);
        }
        let w = Word::from_blob(4_000_000_000, 1_000_000);
        assert_eq!(w.tag(), TAG_BLOB);
        assert_eq!(w.blob_span(), (4_000_000_000, 1_000_000));
        assert!(!w.is_empty());
    }

    #[test]
    fn what_it_holds_is_what_it_says_it_holds() {
        let mut a = Array::new();
        assert_eq!(a.memory_bytes(), 0);
        for i in 0..1000u64 {
            set(&mut a, i * 7, b"a value past the inline limit");
        }
        let held = a.memory_bytes();
        assert!(held > 29_000, "{held} bytes for 29 kilobytes of values");
        a.delete_range(0, u64::MAX - 1);
        assert!(
            a.memory_bytes() < held / 2,
            "{} bytes left of {held}",
            a.memory_bytes()
        );
    }

    /// A thousand writes in a random order against a plain map, to catch the
    /// promotion, demotion, window and blob paths interacting.
    #[test]
    fn it_agrees_with_a_map_over_a_scramble_of_writes() {
        use std::collections::BTreeMap;

        let mut a = Array::new();
        let mut want: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for step in 0..20_000u64 {
            let idx = next() % 20_000;
            match step % 5 {
                0..=2 => {
                    let val = format!("v{step}");
                    let was_new = set(&mut a, idx, val.as_bytes());
                    assert_eq!(was_new, want.insert(idx, val.into_bytes()).is_none());
                }
                3 => {
                    assert_eq!(a.del(idx), want.remove(&idx).is_some());
                }
                _ => {
                    let hi = idx + (next() % 500);
                    let gone = a.delete_range(idx, hi);
                    let keys: Vec<u64> = want.range(idx..=hi).map(|(k, _)| *k).collect();
                    assert_eq!(gone, keys.len() as u64);
                    for k in keys {
                        want.remove(&k);
                    }
                }
            }
            assert_eq!(a.count(), want.len() as u64, "count after step {step}");
        }

        assert_eq!(
            a.len(),
            want.keys().next_back().map_or(0, |k| k + 1),
            "the high water mark"
        );
        for (&idx, val) in &want {
            assert_eq!(read(&a, idx).as_deref(), Some(&val[..]), "at {idx}");
        }
    }

    #[test]
    fn a_scan_finds_the_elements_and_steps_over_the_holes() {
        let mut a = Array::new();
        set(&mut a, 0, b"a");
        set(&mut a, 5, b"b");
        // Three slices apart, so this also proves the walk moves between them.
        set(&mut a, SLICE_SIZE * 2 + 7, b"c");

        let all = vec![
            (0, b"a".to_vec()),
            (5, b"b".to_vec()),
            (SLICE_SIZE * 2 + 7, b"c".to_vec()),
        ];
        // The whole index space costs three visits and not eighteen quintillion,
        // which is why this one needs no cap where ARGETRANGE does.
        assert_eq!(scan(&a, 0, INDEX_MAX, usize::MAX), all);
        let mut backwards = all.clone();
        backwards.reverse();
        assert_eq!(scan(&a, INDEX_MAX, 0, usize::MAX), backwards);

        // A window inside one slice, a window that lands on nothing, and a
        // limit that stops the walk early.
        assert_eq!(scan(&a, 1, 5, usize::MAX), all[1..2].to_vec());
        assert_eq!(scan(&a, 6, SLICE_SIZE, usize::MAX), Vec::new());
        assert_eq!(scan(&a, 0, INDEX_MAX, 2), all[..2].to_vec());
        assert_eq!(scan(&Array::new(), 0, INDEX_MAX, usize::MAX), Vec::new());
    }

    /// A dense slice has holes inside its window and a sparse one does not, so
    /// the walk has to be right in both layouts.
    #[test]
    fn a_scan_reads_both_layouts_the_same_way() {
        let mut a = Array::new();
        for i in 0..40u64 {
            set(&mut a, i, format!("v{i}").as_bytes());
        }
        for i in (0..40u64).step_by(2) {
            a.del(i);
        }
        let odd: Vec<(u64, Vec<u8>)> = (1..40u64)
            .step_by(2)
            .map(|i| (i, format!("v{i}").into_bytes()))
            .collect();
        assert_eq!(scan(&a, 0, 100, usize::MAX), odd);

        // Now the same twenty elements spread far enough apart that the slice
        // has to be sparse, and the answer is the same shape.
        let mut b = Array::new();
        for i in (1..40u64).step_by(2) {
            set(&mut b, i, format!("v{i}").as_bytes());
        }
        assert_eq!(scan(&b, 0, 100, usize::MAX), odd);
    }

    #[test]
    fn the_cursor_moves_only_when_something_appends_to_it() {
        let mut a = Array::new();
        assert_eq!(a.next_index(), Some(0));
        // A plain write does not move it, which is why the first append lands on
        // top of what ARSET put at zero.
        set(&mut a, 0, b"set");
        assert_eq!(a.next_index(), Some(0));
        assert_eq!(append(&mut a, &[b"x", b"y"]).expect("room"), 1);
        assert_eq!(read(&a, 0).as_deref(), Some(&b"x"[..]));
        assert_eq!(a.next_index(), Some(2));

        // A seek says where the next append goes, not where the cursor is.
        a.seek(100);
        assert_eq!(a.next_index(), Some(100));
        assert_eq!(append(&mut a, &[b"z"]).expect("room"), 100);
        assert_eq!(read(&a, 100).as_deref(), Some(&b"z"[..]));
        a.seek(0);
        assert_eq!(a.next_index(), Some(0));
    }

    #[test]
    fn an_append_that_would_run_off_the_top_writes_nothing() {
        let mut a = Array::new();
        a.seek(INDEX_MAX - 1);
        let e = append(&mut a, &[b"x", b"y", b"z"]).unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), INSERT_OVERFLOW);
        assert_eq!(a.count(), 0, "and none of the batch landed");

        // The last index is reachable, and the cursor is finished afterwards.
        assert_eq!(append(&mut a, &[b"x", b"y"]).expect("room"), INDEX_MAX);
        assert_eq!(a.next_index(), None);
        assert_eq!(
            append(&mut a, &[b"z"]).unwrap_err().message(),
            INSERT_OVERFLOW
        );
    }

    #[test]
    fn a_ring_wraps_round_at_its_size() {
        let mut a = Array::new();
        assert_eq!(ring(&mut a, 3, &[b"a", b"b", b"c"]), 2);
        assert_eq!(ring(&mut a, 3, &[b"d", b"e"]), 1);
        assert_eq!(a.len(), 3, "it never grows past the size it was given");
        assert_eq!(a.count(), 3);
        assert_eq!(read(&a, 0).as_deref(), Some(&b"d"[..]));
        assert_eq!(read(&a, 1).as_deref(), Some(&b"e"[..]));
        assert_eq!(read(&a, 2).as_deref(), Some(&b"c"[..]));
    }

    /// A ring that changes size keeps the newest run and renumbers it, so that
    /// reading it back in index order is still reading it in the order it
    /// arrived.
    #[test]
    fn a_ring_that_changes_size_is_renumbered_oldest_first() {
        let mut a = Array::new();
        ring(&mut a, 3, &[b"a", b"b", b"c", b"d", b"e"]);
        // Holding d e c at 0 1 2, so the newest three in order are c d e.
        assert_eq!(ring(&mut a, 5, &[b"f"]), 3);
        assert_eq!(
            (0..4).map(|i| read(&a, i)).collect::<Vec<_>>(),
            vec![
                Some(b"c".to_vec()),
                Some(b"d".to_vec()),
                Some(b"e".to_vec()),
                Some(b"f".to_vec())
            ]
        );

        // And shrinking drops the oldest of them.
        let mut b = Array::new();
        ring(&mut b, 3, &[b"a", b"b", b"c", b"d", b"e"]);
        assert_eq!(ring(&mut b, 2, &[b"f"]), 0);
        assert_eq!(b.count(), 2);
        assert_eq!(read(&b, 0).as_deref(), Some(&b"f"[..]));
        assert_eq!(read(&b, 1).as_deref(), Some(&b"e"[..]));
    }

    /// The rebuild stops at the first hole, so a ring somebody has deleted out
    /// of keeps its newest unbroken run rather than a scattering.
    #[test]
    fn a_hole_cuts_what_a_resize_keeps() {
        let mut a = Array::new();
        ring(&mut a, 4, &[b"a", b"b", b"c", b"d", b"e"]);
        // Holding e b c d with the cursor on 0, and now c is gone.
        a.del(2);
        assert_eq!(ring(&mut a, 8, &[b"f"]), 2);
        // The walk back from e reached the hole where c was, so b did not
        // survive it and d and e did.
        assert_eq!(
            (0..3).map(|i| read(&a, i)).collect::<Vec<_>>(),
            vec![
                Some(b"d".to_vec()),
                Some(b"e".to_vec()),
                Some(b"f".to_vec())
            ]
        );
    }

    #[test]
    fn the_last_items_walk_wraps_and_reports_the_holes() {
        let mut a = Array::new();
        ring(&mut a, 4, &[b"a", b"b", b"c", b"d", b"e"]);
        // Holding e b c d, with the cursor on 0, so the newest is e and the
        // walk has to wrap to find the three before it.
        assert_eq!(
            last(&a, 3, false),
            vec![
                Some(b"c".to_vec()),
                Some(b"d".to_vec()),
                Some(b"e".to_vec())
            ]
        );
        assert_eq!(
            last(&a, 3, true),
            vec![
                Some(b"e".to_vec()),
                Some(b"d".to_vec()),
                Some(b"c".to_vec())
            ]
        );
        // More than there is gets everything and no more.
        assert_eq!(last(&a, 99, false).len(), 4);
        assert_eq!(last(&a, 0, false), Vec::new());
        assert_eq!(last(&Array::new(), 5, false), Vec::new());

        // With no cursor the tail of the array is the anchor, and a position
        // inside the window that holds nothing is reported as a hole.
        let mut b = Array::new();
        set(&mut b, 0, b"a");
        set(&mut b, 2, b"c");
        assert_eq!(last(&b, 5, false), vec![None, Some(b"c".to_vec())]);
    }

    /// The cursor is part of the value, so a copy of an array is a copy of
    /// where it was up to.
    #[test]
    fn a_copy_of_an_array_remembers_the_cursor() {
        let mut a = Array::new();
        append(&mut a, &[b"x", b"y"]).expect("room");
        let mut b = a.clone();
        assert_eq!(b.next_index(), Some(2));
        assert_eq!(append(&mut b, &[b"z"]).expect("room"), 2);
        assert_eq!(a.next_index(), Some(2), "and the two do not share it");
    }

    /// Freeze an array, read it back, and check that nothing about it moved.
    fn round_trip(a: &Array) -> Array {
        let mut buf = Vec::new();
        a.freeze(&mut buf);
        let back = Array::thaw(&buf).expect("what freeze wrote");
        assert_eq!(back.count(), a.count(), "the population");
        assert_eq!(back.len(), a.len(), "the high water mark");
        assert_eq!(back.next_index(), a.next_index(), "the insert cursor");
        assert_eq!(back.slices.len(), a.slices.len(), "the slice count");
        for ((id, was), (back_id, now)) in a.slices.iter().zip(&back.slices) {
            assert_eq!(id, back_id, "the slice ids");
            assert_eq!(was.count, now.count, "slice {id} holds the same number");
            assert_eq!(
                matches!(was.layout, Layout::Dense { .. }),
                matches!(now.layout, Layout::Dense { .. }),
                "slice {id} came back in the layout it left in"
            );
        }
        assert_eq!(
            scan(&back, 0, u64::MAX, usize::MAX),
            scan(a, 0, u64::MAX, usize::MAX)
        );
        back
    }

    #[test]
    fn a_frozen_array_comes_back_with_every_value_it_held() {
        let mut a = Array::new();
        // One of each of the four things a word can be, and a long value that
        // has to live in the blob.
        set(&mut a, 0, b"12345");
        set(&mut a, 1, b"1.5");
        set(&mut a, 2, b"short");
        set(
            &mut a,
            3,
            b"a value well past the seven bytes a word can inline",
        );
        set(&mut a, 9_000_000_000_000, b"a long way up the index space");
        let back = round_trip(&a);
        assert_eq!(read(&back, 0).as_deref(), Some(&b"12345"[..]));
        assert_eq!(read(&back, 1).as_deref(), Some(&b"1.5"[..]));
        assert_eq!(read(&back, 2).as_deref(), Some(&b"short"[..]));
        assert_eq!(
            read(&back, 3).as_deref(),
            Some(&b"a value well past the seven bytes a word can inline"[..])
        );
        assert_eq!(
            read(&back, 9_000_000_000_000).as_deref(),
            Some(&b"a long way up the index space"[..])
        );
        assert_eq!(read(&back, 4), None, "and a hole is still a hole");
        assert_eq!(back.get(0), Some(Element::Int(12345)), "still an integer");
        assert_eq!(back.get(1), Some(Element::Float(1.5)), "still a double");

        round_trip(&Array::new());
    }

    #[test]
    fn both_layouts_come_back_in_the_layout_they_left_in() {
        // Dense, which is eleven consecutive positions.
        let mut dense = Array::new();
        for i in 0..=SPARSE_MAX as u64 {
            set(&mut dense, i, b"x");
        }
        assert!(matches!(dense.slices[0].1.layout, Layout::Dense { .. }));
        round_trip(&dense);

        // Dense with holes punched in the middle, which is the case a rebuild
        // through `set` would have brought back sparse.
        let mut holed = dense.clone();
        for i in 2..5 {
            holed.del(i);
        }
        assert!(matches!(holed.slices[0].1.layout, Layout::Dense { .. }));
        assert_eq!(holed.count(), 8);
        let back = round_trip(&holed);
        assert_eq!(read(&back, 1).as_deref(), Some(&b"x"[..]));
        assert_eq!(read(&back, 4), None);

        // Sparse, which is elements too far apart to be worth a window.
        let mut sparse = Array::new();
        for i in 0..40 {
            set(&mut sparse, i * 100, b"x");
        }
        assert!(matches!(sparse.slices[0].1.layout, Layout::Sparse { .. }));
        round_trip(&sparse);
    }

    #[test]
    fn freezing_an_array_leaves_the_dead_blob_bytes_behind() {
        let mut a = Array::new();
        let long = vec![b'v'; 200];
        // Written and overwritten enough times that most of the blob is dead,
        // and under the floor that would have compacted it in place.
        for _ in 0..8 {
            set(&mut a, 0, &long);
        }
        assert!(a.dead > 0, "there is dead space to leave behind");
        let mut buf = Vec::new();
        a.freeze(&mut buf);
        let back = Array::thaw(&buf).expect("what freeze wrote");
        assert_eq!(back.dead, 0, "a demotion is a compaction");
        assert_eq!(back.blob.len(), a.blob.len() - a.dead);
        assert_eq!(read(&back, 0).as_deref(), Some(&long[..]));
        assert!(
            buf.len() < a.blob.len(),
            "and the dead bytes never went out"
        );
    }

    #[test]
    fn a_frozen_array_keeps_the_insert_cursor() {
        let mut a = Array::new();
        append(&mut a, &[b"x", b"y", b"z"]).expect("room");
        let mut back = round_trip(&a);
        assert_eq!(back.next_index(), Some(3));
        assert_eq!(append(&mut back, &[b"w"]).expect("room"), 3);

        // And an array that nothing has appended to comes back without one, so
        // its first append still lands at zero.
        let mut untouched = Array::new();
        set(&mut untouched, 99, b"x");
        let mut back = round_trip(&untouched);
        assert_eq!(back.next_index(), Some(0), "a cursor nothing has moved");
        assert_eq!(append(&mut back, &[b"first"]).expect("room"), 0);
    }

    #[test]
    fn a_frozen_array_that_arrives_damaged_is_an_error_and_not_a_panic() {
        let mut a = Array::new();
        for i in 0..200u64 {
            set(
                &mut a,
                i * 7,
                format!("value:{i:04} and enough bytes to reach the blob").as_bytes(),
            );
        }
        let mut buf = Vec::new();
        a.freeze(&mut buf);
        assert!(Array::thaw(&buf).is_ok(), "the body it wrote reads back");

        assert!(Array::thaw(&[]).is_err(), "nothing at all");
        assert!(Array::thaw(&[99]).is_err(), "a form nobody wrote");
        for cut in 1..buf.len().min(96) {
            assert!(Array::thaw(&buf[..cut]).is_err(), "cut at {cut}");
        }
        // Every single byte flipped in the header and the first slice, which is
        // where a length, a layout byte and a word all live.
        for at in 0..buf.len().min(96) {
            for bit in 0..8 {
                let mut bad = buf.clone();
                bad[at] ^= 1 << bit;
                // Whatever it decides, it decides without reading off the end of
                // the blob and without a subtraction going backwards.
                let _ = Array::thaw(&bad);
            }
        }
    }
}
