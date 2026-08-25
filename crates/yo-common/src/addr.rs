//! The 56 bit address: a 4 bit space and a 52 bit offset.
//!
//! This is the type that makes Y6 work. A vector's payload, a document's body,
//! a graph node's properties and a Redis string are all reached through one
//! index because the address says which world the offset lives in. None of the
//! four models is layered on another, they just share an address width.
//!
//! 52 bits of offset is 4 PiB, which is past the point where a single file is
//! the right answer, and 4 bits of space is 16 worlds against the 11 that exist.
//! Both were sized once, here, and the sizes are checked by the tests below so
//! that widening one later is a deliberate act.

use core::fmt;

/// Which world an [`Addr`] offset points into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum Space {
    /// The value is the 52 bits. Small integers and short strings live entirely
    /// inside the index entry and cost zero dereferences. This is what Redis
    /// calls `int` encoding and part of what it calls `embstr`.
    Inline = 0,
    /// A byte offset into the shard's arena segments (`05` section 3).
    Arena = 1,
    /// A byte offset into the shard's log region (`06` section 2).
    Log = 2,
    /// A hash, as an owner local element table.
    Hash = 3,
    /// A set, as an owner local element table or a dense member vector.
    Set = 4,
    /// A sorted set, as a counted B+ tree.
    ZSet = 5,
    /// A list, as a ring deque.
    List = 6,
    /// A stream, as a radix log.
    Stream = 7,
    /// A document, with its path indexes.
    Doc = 8,
    /// A vector partition set.
    Vector = 9,
    /// Graph adjacency.
    Graph = 10,
}

impl Space {
    /// Every space in numeric order.
    pub const ALL: &'static [Space] = &[
        Space::Inline,
        Space::Arena,
        Space::Log,
        Space::Hash,
        Space::Set,
        Space::ZSet,
        Space::List,
        Space::Stream,
        Space::Doc,
        Space::Vector,
        Space::Graph,
    ];

    /// The space for a raw 4 bit value, or `None` if nothing uses it yet.
    #[inline]
    pub const fn from_bits(bits: u8) -> Option<Space> {
        match bits {
            0 => Some(Space::Inline),
            1 => Some(Space::Arena),
            2 => Some(Space::Log),
            3 => Some(Space::Hash),
            4 => Some(Space::Set),
            5 => Some(Space::ZSet),
            6 => Some(Space::List),
            7 => Some(Space::Stream),
            8 => Some(Space::Doc),
            9 => Some(Space::Vector),
            10 => Some(Space::Graph),
            _ => None,
        }
    }

    /// The name that appears in `OBJECT ENCODING` style output and in errors.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Space::Inline => "inline",
            Space::Arena => "arena",
            Space::Log => "log",
            Space::Hash => "hash",
            Space::Set => "set",
            Space::ZSet => "zset",
            Space::List => "list",
            Space::Stream => "stream",
            Space::Doc => "doc",
            Space::Vector => "vector",
            Space::Graph => "graph",
        }
    }
}

/// Bits of offset in an address.
pub const OFFSET_BITS: u32 = 52;
/// Bits of space in an address.
pub const SPACE_BITS: u32 = 4;
/// Total bits an address occupies in an index bucket.
pub const ADDR_BITS: u32 = OFFSET_BITS + SPACE_BITS;

/// The largest representable offset.
pub const MAX_OFFSET: u64 = (1u64 << OFFSET_BITS) - 1;

const OFFSET_MASK: u64 = MAX_OFFSET;

/// A 56 bit address, held in the low 56 bits of a `u64`.
///
/// The zero address is reserved to mean "no entry" so that a cleared bucket
/// needs no separate occupancy bit. That costs the `Inline` space its zero
/// value, which is fine because an inline zero is stored as the integer 0 with
/// a type tag rather than as a bare address.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct Addr(u64);

impl Addr {
    /// The absent address. A bucket entry holding this has no value.
    pub const NONE: Addr = Addr(0);

    /// Build an address.
    ///
    /// # Panics
    ///
    /// If `offset` does not fit in 52 bits. An offset that large means the file
    /// has outgrown the address width, which is a design limit and not a
    /// runtime condition, so it is a panic rather than an error value.
    #[inline]
    pub const fn new(space: Space, offset: u64) -> Addr {
        assert!(offset <= MAX_OFFSET, "offset does not fit in 52 bits");
        Addr(((space as u64) << OFFSET_BITS) | offset)
    }

    /// Build an address without checking the offset width.
    ///
    /// # Safety
    ///
    /// `offset` must be at or below [`MAX_OFFSET`]. Passing a wider value
    /// silently corrupts the space bits, which sends a later read into the
    /// wrong world.
    #[inline]
    pub const unsafe fn new_unchecked(space: Space, offset: u64) -> Addr {
        Addr(((space as u64) << OFFSET_BITS) | offset)
    }

    /// The raw 56 bit value, as stored.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Rebuild from a raw 56 bit value.
    ///
    /// Bits above 56 are dropped rather than trusted, because this value comes
    /// off disk and a corrupt high byte should not become a wild pointer.
    #[inline]
    pub const fn from_bits(bits: u64) -> Addr {
        Addr(bits & ((1u64 << ADDR_BITS) - 1))
    }

    /// Whether this address points at anything.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// Whether this address points at something.
    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }

    /// The offset part.
    #[inline]
    pub const fn offset(self) -> u64 {
        self.0 & OFFSET_MASK
    }

    /// The raw space bits, before any check that they name a known space.
    #[inline]
    pub const fn space_bits(self) -> u8 {
        (self.0 >> OFFSET_BITS) as u8
    }

    /// The space, or `None` if the bits name a space this build does not know.
    ///
    /// A file written by a newer release can carry a space we have never heard
    /// of. That is a `VersionTooNew` condition for the caller to report, not a
    /// panic, so this returns an option rather than unwrapping.
    #[inline]
    pub const fn space(self) -> Option<Space> {
        Space::from_bits(self.space_bits())
    }
}

impl fmt::Debug for Addr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            return f.write_str("Addr(none)");
        }
        match self.space() {
            Some(s) => write!(f, "Addr({}+{:#x})", s.name(), self.offset()),
            None => write!(f, "Addr(space{}+{:#x})", self.space_bits(), self.offset()),
        }
    }
}

/// Which shard owns a slot. Small because shard count is bounded by core count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ShardId(pub u16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_widths_are_what_the_bucket_assumes() {
        // The index bucket packs seven addresses into 49 bytes and a link into
        // 7 more. If this ever stops being 56, `05` section 2.1 is wrong.
        assert_eq!(ADDR_BITS, 56);
        assert_eq!(ADDR_BITS % 8, 0);
    }

    #[test]
    fn round_trips_through_bits() {
        for &space in Space::ALL {
            for offset in [0u64, 1, 4096, MAX_OFFSET] {
                let a = Addr::new(space, offset);
                assert_eq!(a.offset(), offset);
                assert_eq!(a.space(), Some(space));
                assert_eq!(Addr::from_bits(a.to_bits()), a);
            }
        }
    }

    #[test]
    fn zero_is_absent() {
        assert!(Addr::NONE.is_none());
        assert!(Addr::new(Space::Arena, 0).is_some());
        assert!(Addr::new(Space::Inline, 1).is_some());
    }

    #[test]
    fn unknown_space_reports_rather_than_panics() {
        let a = Addr::from_bits(15u64 << OFFSET_BITS | 99);
        assert_eq!(a.space(), None);
        assert_eq!(a.space_bits(), 15);
        assert_eq!(a.offset(), 99);
    }

    #[test]
    fn high_byte_from_disk_is_dropped() {
        let a = Addr::from_bits(0xFF00_0000_0000_0000 | 7);
        assert_eq!(a.offset(), 7);
        assert!(a.space_bits() <= 15);
    }

    #[test]
    #[should_panic(expected = "52 bits")]
    fn oversized_offset_panics() {
        let _ = Addr::new(Space::Arena, MAX_OFFSET + 1);
    }
}
