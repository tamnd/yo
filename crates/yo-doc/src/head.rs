//! The four byte header that every value begins with, at every level.
//!
//! One word says what a value is and how big it is, and for a container it is
//! also the only thing a reader needs before it can index the entry table. It
//! is read unaligned, because a document is stored inside a record and a record
//! starts wherever the log put it.

/// What a value is, as a caller sees it.
///
/// The wire keeps object and array apart with a bit rather than a kind, since
/// they share a layout, but nobody outside this crate wants to write
/// `kind == Container && is_array()` so the two are separate here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool,
    /// A signed 64 bit integer.
    Int,
    /// A 64 bit float.
    Float,
    /// A UTF-8 string.
    Text,
    /// An ordered list of values.
    Array,
    /// A set of keys, each with a value.
    Object,
}

/// The wire tag, which is what the low three bits of a header hold.
///
/// The numbers are the format, so they are written out rather than derived from
/// declaration order, and 2 is missing on purpose: false and true are 1 and 3
/// so that the low bit of a boolean is the boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Tag {
    Null = 0,
    False = 1,
    True = 3,
    Int = 4,
    Float = 5,
    Text = 6,
    Container = 7,
}

impl Tag {
    /// The tag the low three bits of `head` name, or `None` for the one value
    /// in that range this version does not define.
    pub(crate) fn of(head: u32) -> Option<Tag> {
        match head & 0b111 {
            0 => Some(Tag::Null),
            1 => Some(Tag::False),
            3 => Some(Tag::True),
            4 => Some(Tag::Int),
            5 => Some(Tag::Float),
            6 => Some(Tag::Text),
            7 => Some(Tag::Container),
            _ => None,
        }
    }
}

/// Set on a container that is an array, clear on one that is an object.
pub(crate) const ARRAY: u32 = 1 << 3;

/// Set on an object whose members are in key order, which is every object this
/// version writes. A reader that finds it clear falls back to a linear scan
/// rather than refusing the document, because an object out of order is still
/// readable and only a lookup gets slower.
pub(crate) const SORTED: u32 = 1 << 4;

/// Set on a container whose entry table carries offsets.
///
/// Every container this version writes has one, and a reader requires it. It is
/// here so that a later version can store lengths instead for a small container
/// and say so, which is the kind of change the format freeze has to leave room
/// for.
pub(crate) const OFFSETS: u32 = 1 << 5;

/// Set on an object whose keys are two byte ids from the collection's intern
/// table rather than bytes in a key region.
pub(crate) const INTERNED: u32 = 1 << 6;

/// Where the count starts.
pub(crate) const COUNT_SHIFT: u32 = 8;

/// The largest count a header can hold, which caps a container at 16.7 M
/// elements and a scalar at 16 MiB.
pub const COUNT_MAX: usize = (1 << 24) - 1;

/// How deep a document may nest.
///
/// A reader walks a document with recursion, down the right hand edge to find a
/// length and down everything to check one, so the depth a writer will produce
/// and the depth a reader will accept have to be one number. It is the same
/// limit RedisJSON has, which means a document that was legal there stays legal
/// here.
pub const DEPTH_MAX: usize = 128;

/// A header built from its parts.
pub(crate) fn head(tag: Tag, flags: u32, count: usize) -> u32 {
    debug_assert!(count <= COUNT_MAX, "the caller checked the count");
    (tag as u32) | flags | ((count as u32) << COUNT_SHIFT)
}

/// The count a header carries: how many elements a container holds, and how
/// many bytes of payload a scalar has.
pub(crate) fn count(head: u32) -> usize {
    (head >> COUNT_SHIFT) as usize
}

/// The four bytes at `at`, or `None` if they are not all there.
pub(crate) fn read(b: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let raw = b.get(at..end)?;
    Some(u32::from_le_bytes(raw.try_into().expect("four bytes")))
}
