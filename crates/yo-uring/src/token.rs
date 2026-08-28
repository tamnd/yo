//! What a submission is for, packed into the eight bytes io_uring gives back.
//!
//! `04` section 7 puts storage and network submissions on one ring and says
//! they are told apart by the user data tag. That tag is a `u64` and it is the
//! only thing a completion carries besides the result, so everything the
//! resume path needs has to fit in it.
//!
//! ```text
//!   63          56 55                     32 31                            0
//!  +--------------+------------------------+------------------------------+
//!  |     kind     |          slot          |          generation          |
//!  +--------------+------------------------+------------------------------+
//! ```
//!
//! The kind says which subsystem the completion belongs to, so a drain can
//! route without a lookup. The slot indexes the pending table. The generation
//! is what makes a stale completion detectable: a slot that has been reused
//! since a submission was cancelled comes back with a generation that no longer
//! matches, and the completion is dropped rather than applied to whatever is
//! living in that slot now. Without it a cancelled read lands on an unrelated
//! connection and nobody ever finds out why.

use core::fmt;

/// Which subsystem a submission belongs to.
///
/// The discriminants are part of the tag and are not rearranged. A value that
/// is not one of these comes back as [`Kind::Unknown`] rather than as a panic,
/// because the kernel will hand back whatever was submitted and a tag that
/// arrived corrupted is not a reason to take the shard down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Kind {
    /// A log page write. `06` section 3.
    Write = 1,
    /// A page read, which is compaction or recovery. `06` sections 4 and 5.
    Read = 2,
    /// The group commit boundary. `06` section 3.
    Fsync = 3,
    /// An inbound connection. `12`.
    Accept = 4,
    /// A read off a connection.
    Recv = 5,
    /// A reply flush. One per connection per batch, never one per reply.
    Send = 6,
    /// A connection going away.
    Close = 7,
    /// Submitted to keep an idle ring from sleeping past a deadline.
    Timeout = 8,
    /// A tag this build does not know. Never submitted, only ever decoded.
    Unknown = 0,
}

impl Kind {
    /// The kind a tag byte names.
    #[must_use]
    pub const fn from_u8(b: u8) -> Kind {
        match b {
            1 => Kind::Write,
            2 => Kind::Read,
            3 => Kind::Fsync,
            4 => Kind::Accept,
            5 => Kind::Recv,
            6 => Kind::Send,
            7 => Kind::Close,
            8 => Kind::Timeout,
            _ => Kind::Unknown,
        }
    }

    /// The tag byte.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Whether this is a storage submission rather than a network one.
    ///
    /// The two share a ring and are drained together, and this is the one bit
    /// of routing that happens before anything is looked up.
    #[must_use]
    pub const fn is_storage(self) -> bool {
        matches!(self, Kind::Write | Kind::Read | Kind::Fsync)
    }
}

/// The most slots a pending table can hold, which is what 24 bits allows.
pub const MAX_SLOT: u32 = (1 << 24) - 1;

/// A submission's user data.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(u64);

impl Token {
    /// Builds a tag.
    ///
    /// `slot` is masked to 24 bits rather than checked, because the only
    /// producer of a slot is the pending table and it already refuses to grow
    /// past [`MAX_SLOT`]. Masking here would hide a bug in a caller that built
    /// one by hand, so debug builds assert instead.
    #[must_use]
    pub const fn new(kind: Kind, slot: u32, generation: u32) -> Token {
        debug_assert!(slot <= MAX_SLOT, "a slot that does not fit in the tag");
        Token(
            ((kind.as_u8() as u64) << 56)
                | (((slot & MAX_SLOT) as u64) << 32)
                | (generation as u64),
        )
    }

    /// Reads a tag back off a completion.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Token {
        Token(raw)
    }

    /// The eight bytes to hand the kernel.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Which subsystem this belongs to.
    #[must_use]
    pub const fn kind(self) -> Kind {
        Kind::from_u8((self.0 >> 56) as u8)
    }

    /// Which pending slot this resumes.
    #[must_use]
    pub const fn slot(self) -> u32 {
        ((self.0 >> 32) as u32) & MAX_SLOT
    }

    /// Which occupant of that slot this belongs to.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.0 as u32
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Token")
            .field("kind", &self.kind())
            .field("slot", &self.slot())
            .field("generation", &self.generation())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_survives_the_round_trip_through_the_kernel() {
        for kind in [
            Kind::Write,
            Kind::Read,
            Kind::Fsync,
            Kind::Accept,
            Kind::Recv,
            Kind::Send,
            Kind::Close,
            Kind::Timeout,
        ] {
            for &slot in &[0u32, 1, 4095, MAX_SLOT] {
                for &generation in &[0u32, 1, u32::MAX] {
                    let t = Token::new(kind, slot, generation);
                    let back = Token::from_raw(t.raw());
                    assert_eq!(back.kind(), kind);
                    assert_eq!(back.slot(), slot, "{kind:?} slot");
                    assert_eq!(back.generation(), generation, "{kind:?} generation");
                }
            }
        }
    }

    /// The three fields have to be independent. A shift that is one bit out
    /// makes a write completion resume a connection, which is the kind of bug
    /// that shows up once a week under load and never in a test.
    #[test]
    fn the_three_fields_do_not_bleed_into_each_other() {
        let t = Token::new(Kind::Fsync, MAX_SLOT, u32::MAX);
        assert_eq!(t.kind(), Kind::Fsync);
        assert_eq!(t.slot(), MAX_SLOT);
        assert_eq!(t.generation(), u32::MAX);

        let t = Token::new(Kind::Write, 0, 0);
        assert_eq!(t.raw(), 1u64 << 56);
    }

    #[test]
    fn a_tag_that_arrived_corrupted_decodes_rather_than_panicking() {
        assert_eq!(Token::from_raw(u64::MAX).kind(), Kind::Unknown);
        assert_eq!(Token::from_raw(0).kind(), Kind::Unknown);
    }

    #[test]
    fn storage_and_network_are_told_apart_without_a_lookup() {
        assert!(Kind::Write.is_storage());
        assert!(Kind::Read.is_storage());
        assert!(Kind::Fsync.is_storage());
        assert!(!Kind::Accept.is_storage());
        assert!(!Kind::Recv.is_storage());
        assert!(!Kind::Send.is_storage());
        assert!(!Kind::Unknown.is_storage());
    }
}
