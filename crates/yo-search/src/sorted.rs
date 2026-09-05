//! The value a document keeps beside itself for a field the schema calls
//! `SORTABLE`, so that sorting by that field does not have to read the key.
//!
//! ```
//! use yo_search::field::{Field, Kind, Text};
//! use yo_search::sorted::Sorted;
//!
//! let mut title = Field::new(b"title", Kind::Text(Text::default()));
//! title.sortable = true;
//! let kept = Sorted::read(&title, b"Banana Split");
//! assert_eq!(kept, Some(Sorted::Text((*b"banana split").into())));
//! ```
//!
//! # What is kept
//!
//! A number is kept as the number it parsed to, and everything else is kept as
//! bytes. The bytes are folded to lower case first, unless the client said `UNF`
//! or the field is a case sensitive tag, both of which keep what they were
//! written. The fold is the whole value at once and not word by word: nothing is
//! trimmed, nothing is split on a separator, an empty value stays empty and a
//! value that is one space stays one space. That is measured against a real
//! server, where a tag written `red,blue` sorts under `red,blue` and not under
//! either half of it.
//!
//! # What is compared
//!
//! A field the schema does not call sortable can still be sorted by, and a real
//! server does it by reading the value off the key and comparing that. So the
//! same rules are wanted in two places, which is why [`Sorted::read`] takes the
//! field rather than being a method on the store: a field nobody called sortable
//! is never folded, and a numeric one is still compared as a number.
//!
//! A document with nothing at the field sorts last whichever way the sort runs,
//! which is what [`order`] is for. It is not the same as sorting first
//! ascending and last descending, and a real server is measurably the second
//! thing: a descending answer is the ascending one reversed with the missing
//! rows left at the end.

use core::cmp::Ordering;

use crate::field::{Field, Kind};
use crate::reduce::number;
use crate::token::fold;

/// What one document holds at one field, as a sort compares it.
#[derive(Debug, Clone, PartialEq)]
pub enum Sorted {
    /// A `NUMERIC` field, as the number it parsed to.
    Number(f64),
    /// Everything else, as the bytes a sort compares.
    Text(Box<[u8]>),
}

impl Sorted {
    /// The value a field of this kind keeps for what the key was written with.
    ///
    /// `None` when there is nothing to keep, which is a numeric field holding
    /// something that is not a number. That cannot happen on the way in, since
    /// a document with a bad number in it is not indexed at all, and it can
    /// happen on a `NOINDEX` field, whose value nothing ever read.
    #[must_use]
    pub fn read(field: &Field, raw: &[u8]) -> Option<Sorted> {
        if field.kind == Kind::Numeric {
            return number(raw).map(Sorted::Number);
        }
        match folds(field) {
            true => Some(Sorted::Text(fold(raw).into())),
            false => Some(Sorted::Text(raw.into())),
        }
    }
}

/// Whether the copy of this field is folded on the way in.
///
/// A number has nothing to fold, which is why a real server reports `UNF` on
/// every sortable numeric field whether the client asked for it or not, and
/// [`Field::is_unf`] says so for us. A case sensitive tag is not folded either,
/// and it does not have to be marked `UNF` to keep its case: the two options
/// mean the same thing there and a real server takes either.
fn folds(field: &Field) -> bool {
    if !field.sortable || field.is_unf() {
        return false;
    }
    match &field.kind {
        Kind::Tag(tag) => !tag.casesensitive,
        _ => true,
    }
}

/// Two documents' values in the order a sort puts them in.
///
/// A document with nothing at the field is last in both directions, so it is
/// the one thing `desc` does not turn around. Everything else is compared as
/// what it is, numbers as numbers and bytes as bytes, and the caller breaks a
/// tie by document number.
#[must_use]
pub fn order(a: Option<&Sorted>, b: Option<&Sorted>, desc: bool) -> Ordering {
    let (Some(a), Some(b)) = (a, b) else {
        return match (a.is_some(), b.is_some()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        };
    };
    let held = match (a, b) {
        (Sorted::Number(a), Sorted::Number(b)) => a.total_cmp(b),
        (Sorted::Text(a), Sorted::Text(b)) => a.cmp(b),
        // Nothing writes two kinds of value into one field, since what is kept
        // is decided by the schema and the schema does not change under a
        // document. This is here so the comparison is total anyway.
        (Sorted::Number(_), Sorted::Text(_)) => Ordering::Less,
        (Sorted::Text(_), Sorted::Number(_)) => Ordering::Greater,
    };
    match desc {
        true => held.reverse(),
        false => held,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Tag, Text};

    fn text(sortable: bool, unf: bool) -> Field {
        let mut field = Field::new(b"t", Kind::Text(Text::default()));
        field.sortable = sortable;
        field.unf = unf;
        field
    }

    fn tag(casesensitive: bool) -> Field {
        let tag = Tag {
            casesensitive,
            ..Tag::default()
        };
        let mut field = Field::new(b"g", Kind::Tag(tag));
        field.sortable = true;
        field
    }

    /// The whole value folded, with nothing trimmed and nothing split, which is
    /// what a real server keeps and what a sort then compares.
    #[test]
    fn a_sortable_copy_is_the_whole_value_folded() {
        let field = text(true, false);
        assert_eq!(
            Sorted::read(&field, b"Banana Split"),
            Some(Sorted::Text((*b"banana split").into()))
        );
        assert_eq!(
            Sorted::read(&field, b"  Red  "),
            Some(Sorted::Text((*b"  red  ").into()))
        );
        assert_eq!(
            Sorted::read(&field, b""),
            Some(Sorted::Text(Box::default()))
        );
        assert_eq!(
            Sorted::read(&field, "Ã".as_bytes()),
            Some(Sorted::Text("ã".as_bytes().into()))
        );
    }

    /// `UNF` keeps the bytes it was given, and so does a case sensitive tag
    /// without anybody saying `UNF`, and so does a field nobody called sortable
    /// at all, whose value is read off the key when a sort asks for it.
    #[test]
    fn what_is_not_folded_is_kept_as_it_arrived() {
        assert_eq!(
            Sorted::read(&text(true, true), b"MiXed"),
            Some(Sorted::Text((*b"MiXed").into()))
        );
        assert_eq!(
            Sorted::read(&tag(true), b"Red,Blue"),
            Some(Sorted::Text((*b"Red,Blue").into()))
        );
        assert_eq!(
            Sorted::read(&tag(false), b"Red,Blue"),
            Some(Sorted::Text((*b"red,blue").into()))
        );
        assert_eq!(
            Sorted::read(&text(false, false), b"MiXed"),
            Some(Sorted::Text((*b"MiXed").into()))
        );
    }

    /// A number is kept as a number whether or not it is sortable, and a field
    /// that will not read as one keeps nothing.
    #[test]
    fn a_number_is_kept_as_a_number() {
        let mut field = Field::new(b"n", Kind::Numeric);
        field.sortable = true;
        assert_eq!(Sorted::read(&field, b"1e3"), Some(Sorted::Number(1000.0)));
        assert_eq!(Sorted::read(&field, b"-3"), Some(Sorted::Number(-3.0)));
        assert_eq!(Sorted::read(&field, b"7x"), None);
        assert_eq!(Sorted::read(&field, b""), None);
    }

    /// Missing is last both ways round, which is the one thing a descending
    /// sort does not turn over.
    #[test]
    fn nothing_at_all_sorts_last_whichever_way_the_sort_runs() {
        let one = Sorted::Number(1.0);
        let two = Sorted::Number(2.0);
        assert_eq!(order(Some(&one), Some(&two), false), Ordering::Less);
        assert_eq!(order(Some(&one), Some(&two), true), Ordering::Greater);
        for desc in [false, true] {
            assert_eq!(order(Some(&one), None, desc), Ordering::Less);
            assert_eq!(order(None, Some(&one), desc), Ordering::Greater);
            assert_eq!(order(None, None, desc), Ordering::Equal);
        }
    }

    /// Bytes are compared as bytes, so an upper case value that was folded on
    /// the way in sorts where its lower case form belongs.
    #[test]
    fn bytes_are_compared_as_bytes() {
        let field = text(true, false);
        let apple = Sorted::read(&field, b"Apple").expect("a value is kept");
        let banana = Sorted::read(&field, b"banana").expect("a value is kept");
        assert_eq!(order(Some(&apple), Some(&banana), false), Ordering::Less);
        // Unfolded, `Apple` would sort in front of everything lower case.
        let raw = text(true, true);
        let apple = Sorted::read(&raw, b"Apple").expect("a value is kept");
        assert_eq!(order(Some(&apple), Some(&banana), false), Ordering::Less);
        let zebra = Sorted::read(&raw, b"Zebra").expect("a value is kept");
        assert_eq!(order(Some(&zebra), Some(&banana), false), Ordering::Less);
    }
}
