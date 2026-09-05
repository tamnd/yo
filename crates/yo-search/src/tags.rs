//! The tag index: what a field's bytes turn into on the way in, and which
//! documents ended up holding each value.
//!
//! ```
//! use yo_search::field::Tag;
//! use yo_search::tags::{Tags, values};
//!
//! let tag = Tag::default();
//! assert_eq!(values(b"Fiction, Crime", &tag), [b"fiction".to_vec(), b"crime".to_vec()]);
//!
//! let mut t = Tags::new();
//! t.index(1, b"Fiction, Crime", &tag);
//! t.index(2, b"CRIME", &tag);
//! assert_eq!(t.get(b"crime"), [1, 2]);
//! assert_eq!(t.get(b"Crime"), []);
//! ```
//!
//! # What a value goes through
//!
//! The whole field is cut at the first zero byte, because what holds it is a C
//! string and there is nothing after a zero in one of those. So `a\0b,c` is the
//! one value `a` and the `c` on the other side of the zero is not indexed at
//! all. What is left is split on the separator, which is a comma unless the
//! field named another one, and each piece is trimmed of the six bytes C calls
//! space. A piece with nothing left in it is dropped, so `one,,two` is two
//! values and a lone comma is none.
//!
//! Then it is folded to lower case, unless the field is `CASESENSITIVE`, in
//! which case `Crime` and `crime` are two values and both are kept. The folding
//! is [`crate::token::fold`], the same one a query uses, and it is not the one a
//! text document gets: a tag is not cut down to sixteen bits, so an emoji goes
//! in whole and `𐐀` folds to `𐐨` rather than to whatever those characters are
//! once sixteen bits is all they have.
//!
//! A backslash is not an escape here. A client writing `a\,b` in a query means
//! the one value `a,b`, and the same bytes in a document are the two values
//! `a\` and `b`, because the document side splits first and never looks at the
//! backslash. That is a real server's behaviour and it is worth knowing before
//! wondering why a tag written with an escape cannot be found.
//!
//! # Why an empty fold keeps the bytes it started with
//!
//! Folding stops at a character that comes out as zero, which is how `A\x80`
//! indexes as `a`. When that happens on the first character there is nothing
//! left, and a real server puts the value in unfolded rather than dropping it,
//! so a lone `\x80` is indexed as a lone `\x80`. It is not a rule anybody would
//! design, it is what a fold that reports its length back and a caller that
//! reads zero as nothing to do add up to, and a query folded the same way finds
//! the same thing, so copying it costs nothing and not copying it loses
//! documents.
//!
//! # What is not here
//!
//! Taking anything out, the same as [`crate::posts`]. A value's list only grows
//! at the end and a document that is written again is a new number, so the
//! number it had stays where it was and is skipped by whoever finds it missing
//! from the document table.

use std::collections::BTreeMap;

use crate::field::Tag;
use crate::posts::Id;
use crate::token::fold;

/// Whether a byte is trimmed off the ends of a value.
///
/// The six C calls space, which is the space itself and the run from tab to
/// carriage return. They are trimmed off the ends and kept in the middle, so
/// `a b c` is one value with two spaces in it.
#[must_use]
pub const fn spacing(b: u8) -> bool {
    b == b' ' || matches!(b, 0x09..=0x0d)
}

/// What one field of a document turns into, in the order it was written.
///
/// Duplicates are left in, because taking them out is the index's job and a
/// caller that only wants the values a client wrote should see what was there.
#[must_use]
pub fn values(raw: &[u8], tag: &Tag) -> Vec<Vec<u8>> {
    let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
    let mut out = Vec::new();
    for piece in raw[..end].split(|b| *b == tag.separator) {
        let piece = trimmed(piece);
        if piece.is_empty() {
            continue;
        }
        if tag.casesensitive {
            out.push(piece.to_vec());
            continue;
        }
        let folded = fold(piece);
        // Nothing survived the fold, so the value goes in as it arrived.
        out.push(if folded.is_empty() {
            piece.to_vec()
        } else {
            folded
        });
    }
    out
}

/// One value with the spacing taken off both ends.
fn trimmed(piece: &[u8]) -> &[u8] {
    let from = piece.iter().position(|b| !spacing(*b));
    let Some(from) = from else {
        return &[];
    };
    let to = piece
        .iter()
        .rposition(|b| !spacing(*b))
        .expect("there is a byte that is not spacing");
    &piece[from..=to]
}

/// Every value one tag field holds, over every document that has it.
#[derive(Debug, Clone, Default)]
pub struct Tags {
    by: BTreeMap<Vec<u8>, Vec<Id>>,
    last: Id,
}

impl Tags {
    /// An index with nothing in it.
    #[must_use]
    pub fn new() -> Tags {
        Tags::default()
    }

    /// Reads one document's field and records every value in it.
    pub fn index(&mut self, id: Id, raw: &[u8], tag: &Tag) {
        for value in values(raw, tag) {
            self.add(id, &value);
        }
    }

    /// Records that a document holds a value, which has already been folded.
    ///
    /// A document that names the same value twice is one entry and not two,
    /// which is why the list is checked at the end before anything is added to
    /// it. That is all the checking a list needs, because documents arrive in
    /// the order their numbers were handed out.
    pub fn add(&mut self, id: Id, value: &[u8]) {
        let ids = self.by.entry(value.to_vec()).or_default();
        if ids.last() != Some(&id) {
            ids.push(id);
        }
        self.last = self.last.max(id);
    }

    /// The documents that hold a value, in number order.
    #[must_use]
    pub fn get(&self, value: &[u8]) -> &[Id] {
        self.by.get(value).map_or(&[], Vec::as_slice)
    }

    /// Every value with its documents, in byte order.
    ///
    /// Which is the order a real server dumps them in and the order `FT.TAGVALS`
    /// answers in, and it comes free because that is how they are kept.
    pub fn all(&self) -> impl Iterator<Item = (&[u8], &[Id])> {
        self.by
            .iter()
            .map(|(value, ids)| (value.as_slice(), ids.as_slice()))
    }

    /// How many different values are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by.len()
    }

    /// Whether no document holds a value here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by.is_empty()
    }

    /// The largest document number seen, or zero when there is none.
    #[must_use]
    pub const fn last(&self) -> Id {
        self.last
    }

    /// How many bytes the values and their lists take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.by
            .iter()
            .map(|(value, ids)| value.len() + ids.len() * size_of::<Id>())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(raw: &[u8]) -> Vec<Vec<u8>> {
        values(raw, &Tag::default())
    }

    fn exact(raw: &[u8]) -> Vec<Vec<u8>> {
        values(
            raw,
            &Tag {
                casesensitive: true,
                ..Tag::default()
            },
        )
    }

    /// The whole of it: split on the comma, trimmed at both ends, folded down.
    #[test]
    fn a_value_is_split_and_trimmed_and_folded() {
        assert_eq!(
            plain(b"Fiction, Crime ,  Noir"),
            [b"fiction".to_vec(), b"crime".to_vec(), b"noir".to_vec()]
        );
        assert_eq!(plain(b"a b c"), [b"a b c".to_vec()]);
        assert_eq!(
            plain(b"\t\r\n\x0b\x0ctrimmed\x0c\x0b\n\r\t"),
            [b"trimmed".to_vec()]
        );
    }

    /// The separator is whatever the field named, and every other byte is an
    /// ordinary byte, the comma included.
    #[test]
    fn the_separator_is_the_one_the_field_named() {
        let tag = Tag {
            separator: b';',
            ..Tag::default()
        };
        assert_eq!(values(b"a,b;c", &tag), [b"a,b".to_vec(), b"c".to_vec()]);
    }

    /// A value with nothing in it is not a value, however it came to be empty.
    #[test]
    fn an_empty_value_is_not_a_value() {
        assert_eq!(plain(b"one,,two"), [b"one".to_vec(), b"two".to_vec()]);
        assert!(plain(b",").is_empty());
        assert!(plain(b"   ").is_empty());
        assert!(plain(b"\t\n").is_empty());
        assert!(plain(b"").is_empty());
        assert_eq!(plain(b"a, ,b"), [b"a".to_vec(), b"b".to_vec()]);
    }

    /// The field is a C string and stops at the first zero, which takes the
    /// separator after it out of service along with everything else.
    #[test]
    fn a_value_ends_at_the_first_zero_byte() {
        assert_eq!(plain(b"a\x00b,c"), [b"a".to_vec()]);
        assert!(plain(b"\x00").is_empty());
        assert_eq!(exact(b"A\x00B"), [b"A".to_vec()]);
    }

    /// A backslash means nothing on this side, so an escaped separator is still
    /// a separator and the backslash is an ordinary byte on the value.
    #[test]
    fn a_backslash_is_not_an_escape_here() {
        assert_eq!(plain(br"a\,b"), [br"a\".to_vec(), b"b".to_vec()]);
        assert_eq!(plain(br"a\\,b"), [br"a\\".to_vec(), b"b".to_vec()]);
        assert_eq!(plain(br"\ lead"), [br"\ lead".to_vec()]);
    }

    /// A case sensitive field folds nothing, so two spellings are two values,
    /// and the trimming and the zero byte still apply.
    #[test]
    fn a_case_sensitive_field_keeps_both_spellings() {
        assert_eq!(
            exact(b"Crime,crime"),
            [b"Crime".to_vec(), b"crime".to_vec()]
        );
        assert_eq!(exact(b"  Mixed Case  "), [b"Mixed Case".to_vec()]);
        assert_eq!(exact("É".as_bytes()), ["É".as_bytes().to_vec()]);
        // Nothing folds, so nothing ends at a character that folds to zero.
        assert_eq!(exact(b"A\x80"), [b"A\x80".to_vec()]);
    }

    /// The fold here is the query's fold and not the document's, so a character
    /// that does not fit in sixteen bits keeps the rest of itself. Every one of
    /// these came off a real server.
    #[test]
    fn folding_is_not_cut_down_to_sixteen_bits() {
        assert_eq!(plain("𐐀".as_bytes()), ["𐐨".as_bytes().to_vec()]);
        assert_eq!(plain("😀".as_bytes()), ["😀".as_bytes().to_vec()]);
        assert_eq!(plain("K".as_bytes()), ["k".as_bytes().to_vec()]);
        assert_eq!(plain("ẞ".as_bytes()), ["ß".as_bytes().to_vec()]);
        assert_eq!(plain("İ".as_bytes()), ["i\u{307}".as_bytes().to_vec()]);
        assert_eq!(plain("Ａ".as_bytes()), ["ａ".as_bytes().to_vec()]);
    }

    /// Bytes that are not characters are read as though they were, under the
    /// same lenient rule a word is read under, and a character that comes out as
    /// zero ends the value there.
    #[test]
    fn broken_bytes_are_read_rather_than_refused() {
        assert_eq!(plain(b"\xc3"), ["à".as_bytes().to_vec()]);
        assert_eq!(plain(b"\xe2\x82"), ["₀".as_bytes().to_vec()]);
        assert_eq!(plain(b"\xff"), [b"\xf7\x80\x80\x80".to_vec()]);
        assert_eq!(plain(b"\xfe"), [b"\xf6\x80\x80\x80".to_vec()]);
        assert_eq!(plain(b"\x80A"), [b"\x01".to_vec()]);
        assert_eq!(plain(b"\xc0A"), [b"\x01".to_vec()]);
        assert_eq!(plain(b"\xbf"), [b"\xdf\x80".to_vec()]);
        assert_eq!(plain(b"AB\x80CD"), [b"ab\x03d".to_vec()]);
        assert_eq!(plain(b"A\x80"), [b"a".to_vec()]);
        assert_eq!(plain(b"\xc3\x89\x80"), ["é".as_bytes().to_vec()]);
    }

    /// A fold with nothing left at the end of it is not a value that goes away,
    /// it is the value it started as.
    #[test]
    fn a_fold_that_comes_out_empty_keeps_what_it_was_given() {
        assert_eq!(plain(b"\x80"), [b"\x80".to_vec()]);
        assert_eq!(plain(b"\x80\x80"), [b"\x80\x80".to_vec()]);
        assert_eq!(plain(b" \x80 "), [b"\x80".to_vec()]);
        assert_eq!(plain(b"ABC,\x80"), [b"abc".to_vec(), b"\x80".to_vec()]);
        assert_eq!(plain(b"\x80,DEF"), [b"\x80".to_vec(), b"def".to_vec()]);
    }

    /// There is no length a value is cut at, which a real server was asked about
    /// a thousand characters at a time.
    #[test]
    fn a_long_value_is_kept_whole() {
        let long = vec![b'A'; 1000];
        assert_eq!(plain(&long), [vec![b'a'; 1000]]);
    }

    /// A value points at every document that has it, in the order the numbers
    /// were handed out, and a document that names it twice is in the list once.
    #[test]
    fn a_value_points_at_every_document_that_has_it() {
        let tag = Tag::default();
        let mut t = Tags::new();
        t.index(1, b"a,a", &tag);
        t.index(2, b"A,a,  a  ", &tag);
        t.index(3, b"b", &tag);
        t.index(4, b"a", &tag);
        assert_eq!(t.get(b"a"), [1, 2, 4]);
        assert_eq!(t.get(b"b"), [3]);
        assert_eq!(t.get(b"c"), []);
        assert_eq!(t.len(), 2);
        assert_eq!(t.last(), 4);
        assert!(t.bytes() > 0);
    }

    /// The values come out in byte order, which is the order they are dumped in
    /// and the order `FT.TAGVALS` answers in.
    #[test]
    fn the_values_come_out_in_byte_order() {
        let tag = Tag::default();
        let mut t = Tags::new();
        t.index(1, b"zebra,Ant,mule", &tag);
        t.index(2, b"ant", &tag);
        let seen: Vec<_> = t.all().map(|(value, ids)| (value.to_vec(), ids)).collect();
        assert_eq!(
            seen,
            [
                (b"ant".to_vec(), [1, 2].as_slice()),
                (b"mule".to_vec(), [1].as_slice()),
                (b"zebra".to_vec(), [1].as_slice())
            ]
        );
        let empty = Tags::new();
        assert!(empty.is_empty());
        assert_eq!(empty.last(), 0);
        assert_eq!(empty.bytes(), 0);
        assert_eq!(empty.all().count(), 0);
    }

    #[test]
    fn the_bytes_that_come_off_the_ends_are_the_six_c_calls_space() {
        for b in [b' ', 0x09, 0x0a, 0x0b, 0x0c, 0x0d] {
            assert!(spacing(b));
        }
        assert!(!spacing(0x00));
        assert!(!spacing(0x08));
        assert!(!spacing(0x0e));
        assert!(!spacing(b'a'));
        assert!(!spacing(0xa0));
    }
}
