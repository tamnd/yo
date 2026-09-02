//! What a document record holds, which is the document and nothing around it.
//!
//! `06` section 2.1 gives kind 2 to a document and says nothing about what is
//! inside it. The answer is that a document record's value is the YOJB value
//! byte for byte, with no header of its own, and this module is where that
//! decision is written down and checked.
//!
//! ```text
//! +---------+----------------------------------------+
//! |  head   |            the rest of the value       |
//! |    4    |                                        |
//! +---------+----------------------------------------+
//! ```
//!
//! # Why there is no framing
//!
//! A vector record needs a header because a run of `f32` says nothing about
//! itself: the dimension and the element type have to come from somewhere, and
//! the record is the only place that a reader with no catalogue can get them.
//! YOJB is the opposite. Every value begins with a four byte header that carries
//! its kind, its flags and its count, offsets inside a container are relative to
//! that header, and the last entry is enough to work out the whole length. A
//! frame around it would be four to eight bytes on every document that repeat
//! what the first word already says, and it would be a second length to
//! disagree with the first one.
//!
//! So the record's value is the document, `DocumentBody::decode` is the check
//! that the first word is one this version understands, and the length a reader
//! gets back from the log is the length of the document.
//!
//! # What this checks and what it does not
//!
//! The record layer owns the framing and `yo-doc` owns the value. This checks
//! the head: that it is there, that the tag is one of the seven this version
//! defines, and, for a scalar, that the payload is exactly as long as the head
//! says and the right length for its type. It does not walk a container, because
//! walking a container means knowing where the entry tables are and how deep the
//! nesting is allowed to go, and there is one copy of that in `yo-doc` on
//! purpose. `Value::validate` is the deep check and `yodb check` is what calls
//! both.
//!
//! Getting this split wrong in the other direction would be worse than the
//! duplication it saves. A reader that has to understand documents to skip a
//! document record cannot skip a kind it does not know, and skipping is what
//! `07` section 9 requires of it.
//!
//! # Why the numbering is here as well as in `yo-doc`
//!
//! These are the bytes on disk, so they belong with the other frozen shapes,
//! and a reader that only wants to know whether a record is an object or an
//! array should not have to pull in the document model to find out. `yo-doc`
//! has the same numbers because it is the one that reads them, and a test in
//! that crate holds the two together, which is what `yo-kv` already does with
//! [`crate::ValueType`].
//!
//! # What is not in here
//!
//! The key table. An interned object stores two byte ids instead of key bytes,
//! and the names those ids stand for live in the collection rather than in any
//! one document. That is a collection chunk under a checkpoint, not a record
//! kind, and it is not written yet.
//!
//! Interning needs no generation number alongside it, which is worth saying
//! because it looks like it should. An id is the row a name sits at in a table
//! that never removes anything, so an id handed out at any point stays the same
//! name for the life of the collection, and a document interned against an
//! early state of the table reads correctly against every later one.

use crate::get_u32;
use yo_common::{Code, Error, Result};

/// The header every YOJB value begins with.
pub const DOC_HEADER_LEN: usize = 4;

/// Where the count starts in the header.
pub const DOC_COUNT_SHIFT: u32 = 8;

/// The largest count a header can hold, which caps a container at 16.7 M
/// elements and a scalar at 16 MiB of payload.
pub const DOC_COUNT_MAX: usize = (1 << 24) - 1;

/// The flag bits of a value header.
pub mod doc_flags {
    /// Set on a container that is an array, clear on one that is an object.
    pub const ARRAY: u32 = 1 << 3;
    /// Set on an object whose members are in key order, which is every object
    /// this version writes.
    pub const SORTED: u32 = 1 << 4;
    /// Set on a container whose entry table carries offsets, which is every
    /// container this version writes.
    pub const OFFSETS: u32 = 1 << 5;
    /// Set on an object whose keys are two byte ids from the collection's key
    /// table rather than bytes in a key region.
    pub const INTERNED: u32 = 1 << 6;
}

/// What the low three bits of a value header say the value is.
///
/// The numbers are the format, so they are written out rather than derived from
/// declaration order, and 2 is missing on purpose: false and true are 1 and 3 so
/// that the low bit of a boolean is the boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueTag {
    /// `null`.
    Null = 0,
    /// `false`.
    False = 1,
    /// `true`.
    True = 3,
    /// A signed integer, stored in one, two, four or eight bytes.
    Int = 4,
    /// A 64 bit float.
    Float = 5,
    /// A UTF-8 string.
    Text = 6,
    /// An object or an array, told apart by [`doc_flags::ARRAY`].
    Container = 7,
}

impl ValueTag {
    /// The byte that stands for this tag.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The tag for a byte, or `None` for the one value in the range this
    /// version does not define.
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<ValueTag> {
        match b {
            0 => Some(ValueTag::Null),
            1 => Some(ValueTag::False),
            3 => Some(ValueTag::True),
            4 => Some(ValueTag::Int),
            5 => Some(ValueTag::Float),
            6 => Some(ValueTag::Text),
            7 => Some(ValueTag::Container),
            _ => None,
        }
    }

    /// Whether this tag is a container rather than a scalar.
    #[must_use]
    pub const fn is_container(self) -> bool {
        matches!(self, ValueTag::Container)
    }
}

/// A document record's value, borrowed.
///
/// Decoding copies nothing. See the module note for what it checks.
#[derive(Debug, Clone, Copy)]
pub struct DocumentBody<'a> {
    head: u32,
    tag: ValueTag,
    bytes: &'a [u8],
}

impl<'a> DocumentBody<'a> {
    /// Reads a document record's value.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the value is shorter than a header, if the tag is
    /// one this version does not define, or if it is a scalar whose payload is
    /// not the length the header claims.
    pub fn decode(value: &'a [u8]) -> Result<DocumentBody<'a>> {
        if value.len() < DOC_HEADER_LEN {
            return Err(Error::new(
                Code::Corrupt,
                "a document record is shorter than a value header",
            )
            .with_detail(format!("len={}", value.len())));
        }
        let head = get_u32(value, 0);
        let Some(tag) = ValueTag::from_u8((head & 0b111) as u8) else {
            return Err(
                Error::new(Code::Corrupt, "a document record has an unknown tag")
                    .with_detail(format!("head={head:#010x}")),
            );
        };
        let count = (head >> DOC_COUNT_SHIFT) as usize;
        if !tag.is_container() {
            // A scalar is the header and its payload and nothing else, so its
            // length is knowable here and a short one is worth catching before
            // anybody reads eight bytes out of a four byte record.
            let want = DOC_HEADER_LEN + count;
            if value.len() != want {
                return Err(Error::new(
                    Code::Corrupt,
                    "a scalar document is not the length its header says",
                )
                .with_detail(format!("len={} want={want}", value.len())));
            }
            let ok = match tag {
                ValueTag::Null | ValueTag::False | ValueTag::True => count == 0,
                ValueTag::Int => matches!(count, 1 | 2 | 4 | 8),
                ValueTag::Float => count == 8,
                ValueTag::Text => true,
                ValueTag::Container => unreachable!("checked above"),
            };
            if !ok {
                return Err(Error::new(
                    Code::Corrupt,
                    "a scalar document has a payload its type cannot have",
                )
                .with_detail(format!("tag={tag:?} payload={count}")));
            }
        }
        Ok(DocumentBody {
            head,
            tag,
            bytes: value,
        })
    }

    /// The header word, for a reader that wants a flag this version has no name
    /// for.
    #[must_use]
    pub fn head(self) -> u32 {
        self.head
    }

    /// What the value is.
    #[must_use]
    pub fn tag(self) -> ValueTag {
        self.tag
    }

    /// How many elements a container holds, or how many bytes of payload a
    /// scalar has.
    #[must_use]
    pub fn count(self) -> usize {
        (self.head >> DOC_COUNT_SHIFT) as usize
    }

    /// Whether this is an array. False for an object and for every scalar.
    #[must_use]
    pub fn is_array(self) -> bool {
        self.tag.is_container() && self.head & doc_flags::ARRAY != 0
    }

    /// Whether this is an object whose keys are ids from the collection's key
    /// table.
    #[must_use]
    pub fn is_interned(self) -> bool {
        self.tag.is_container()
            && self.head & doc_flags::ARRAY == 0
            && self.head & doc_flags::INTERNED != 0
    }

    /// The value, which is the whole of the record's value.
    #[must_use]
    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header built the way a writer would, so the tests read as documents
    /// rather than as hex.
    fn head(tag: ValueTag, flags: u32, count: usize) -> [u8; 4] {
        (u32::from(tag.as_u8()) | flags | ((count as u32) << DOC_COUNT_SHIFT)).to_le_bytes()
    }

    #[test]
    fn a_scalar_is_its_header_and_its_payload() {
        let mut v = head(ValueTag::Int, 0, 8).to_vec();
        v.extend_from_slice(&41_920i64.to_le_bytes());
        let d = DocumentBody::decode(&v).unwrap();
        assert_eq!(d.tag(), ValueTag::Int);
        assert_eq!(d.count(), 8);
        assert!(!d.is_array());
        assert!(!d.is_interned());
        assert_eq!(d.bytes(), &v[..]);
    }

    #[test]
    fn a_scalar_that_lost_bytes_is_corrupt() {
        let mut v = head(ValueTag::Int, 0, 8).to_vec();
        v.extend_from_slice(&41_920i64.to_le_bytes());
        for cut in 1..=8 {
            let short = &v[..v.len() - cut];
            assert!(
                DocumentBody::decode(short).is_err(),
                "an int missing {cut} bytes was accepted"
            );
        }
    }

    #[test]
    fn a_scalar_of_a_length_its_type_cannot_have_is_corrupt() {
        // Three byte integers and four byte floats do not exist, and a header
        // that claims one is a corruption that lands inside the length check
        // rather than outside it.
        let mut v = head(ValueTag::Int, 0, 3).to_vec();
        v.extend_from_slice(&[1, 2, 3]);
        assert!(DocumentBody::decode(&v).is_err());

        let mut v = head(ValueTag::Float, 0, 4).to_vec();
        v.extend_from_slice(&[1, 2, 3, 4]);
        assert!(DocumentBody::decode(&v).is_err());

        // And null carries nothing at all.
        let mut v = head(ValueTag::Null, 0, 1).to_vec();
        v.push(0);
        assert!(DocumentBody::decode(&v).is_err());
    }

    #[test]
    fn a_container_is_not_walked_here() {
        // Nine bytes is not enough for an object of four members, and this
        // still decodes, because how much room four members need is the
        // layout's question and the layout lives in `yo-doc`.
        let mut v = head(
            ValueTag::Container,
            doc_flags::OFFSETS | doc_flags::SORTED,
            4,
        )
        .to_vec();
        v.extend_from_slice(&[0; 5]);
        let d = DocumentBody::decode(&v).unwrap();
        assert_eq!(d.tag(), ValueTag::Container);
        assert_eq!(d.count(), 4);
        assert!(!d.is_array());
    }

    #[test]
    fn an_array_and_an_interned_object_say_so() {
        let v = head(
            ValueTag::Container,
            doc_flags::ARRAY | doc_flags::OFFSETS,
            0,
        )
        .to_vec();
        let d = DocumentBody::decode(&v).unwrap();
        assert!(d.is_array());
        assert!(!d.is_interned(), "an array has no keys to intern");

        let v = head(
            ValueTag::Container,
            doc_flags::INTERNED | doc_flags::OFFSETS,
            0,
        )
        .to_vec();
        let d = DocumentBody::decode(&v).unwrap();
        assert!(!d.is_array());
        assert!(d.is_interned());
    }

    #[test]
    fn an_unknown_tag_is_corrupt_rather_than_a_guess() {
        // Two is the one value in the range this version does not define, and
        // it is the one a later version would use first.
        let v = 2u32.to_le_bytes().to_vec();
        assert!(DocumentBody::decode(&v).is_err());
        assert_eq!(ValueTag::from_u8(2), None);
    }

    #[test]
    fn a_value_shorter_than_a_header_is_corrupt() {
        for n in 0..DOC_HEADER_LEN {
            assert!(DocumentBody::decode(&vec![0u8; n]).is_err(), "{n} bytes");
        }
    }

    #[test]
    fn every_tag_round_trips_through_its_byte() {
        for tag in [
            ValueTag::Null,
            ValueTag::False,
            ValueTag::True,
            ValueTag::Int,
            ValueTag::Float,
            ValueTag::Text,
            ValueTag::Container,
        ] {
            assert_eq!(ValueTag::from_u8(tag.as_u8()), Some(tag));
        }
    }
}
