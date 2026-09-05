//! The synonym groups one index keeps, which is `FT.SYNUPDATE` and
//! `FT.SYNDUMP`.
//!
//! ```
//! use yo_search::synonyms::Synonyms;
//!
//! let mut syn = Synonyms::new();
//! syn.update(b"g1", &[b"boy", b"child"]);
//! assert_eq!(syn.asked(b"child"), [b"g1".to_vec().into_boxed_slice()]);
//! assert!(syn.asked(b"nobody").is_empty());
//! ```
//!
//! # A group is a term of its own
//!
//! There is no comparing of words at query time and no list of the other words
//! in a group being walked. A group is a term the index holds like any other,
//! spelled with a `~` in front of the id so nothing a client can type collides
//! with it, and both sides of the search put it there:
//!
//! A document with a word in a group has `~g1` written into the index beside
//! the word, at the same place, in the same field and worth the same as the
//! word is. A query for a word in a group is a union of the word and `~g1`. So
//! `child` finds the document that only ever said `boy` by reading one posting
//! list, and a group of a hundred words costs a query exactly as much as a group
//! of two.
//!
//! That has consequences worth knowing. The group term is in the dictionary, so
//! `FT.INFO` counts it in `num_terms` and a term dump lists it. It is worth what
//! the word was worth, so a word in a field of weight three puts three into the
//! group term as well, and a document with two words of the same group in it has
//! the group term at a frequency of two. It counts towards the largest frequency
//! in the document and not towards the document's length, which is what a stem
//! does, so the scoring divides by the same length whether an index has synonyms
//! or not. Every one of those was measured rather than chosen.
//!
//! # It is written at index time, so the index has to be read again
//!
//! A group added after the documents were written does nothing for them until
//! they are read again, because the group term was not in the index when they
//! went in. So `FT.SYNUPDATE` rescans the whole index unless the client says
//! `SKIPINITIALSCAN`, and that rescan renumbers every document in it. A real
//! server does the same, and it does it even for an index that was created with
//! `SKIPINITIALSCAN`, which is measured: the two flags are not the same flag.
//!
//! # Two ways of folding one term
//!
//! A term is folded on the way in the way a query folds a word, which is what
//! `FT.SYNDUMP` answers with. A document folds a word further, cutting every
//! character down to sixteen bits, so the two forms differ for a term with a
//! character above that in it. Both are kept here, one map for each side, since
//! a query looks a word up as it typed it and a document looks it up as it
//! stored it.
//!
//! There is one thing that does not follow from that, and it is D-86. A document
//! word whose character was cut down onto a term in a group is matched here and
//! is not matched by a real server, which reads the word before the cut and
//! never sees the two as the same.

use std::collections::BTreeMap;

use crate::token::{fold, fold_stored};

/// The byte a group id is kept behind, so a group term is not a word.
pub const MARK: u8 = b'~';

/// The term a group is written into the index under.
#[must_use]
pub fn term(id: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(id.len() + 1);
    out.push(MARK);
    out.extend_from_slice(id);
    out
}

/// Every synonym group of one index, from both sides.
///
/// One of these per index rather than one per server, which is what a real
/// server does: two indexes over the same keys have their own groups and
/// dumping one says nothing about the other.
#[derive(Debug, Clone, Default)]
pub struct Synonyms {
    /// Term to the ids of the groups it is in, keyed as a query folds a word.
    ///
    /// The ids are in the order they were added to the term rather than in
    /// order, which is what the dump answers with and what the union a query
    /// expands into is built in.
    words: BTreeMap<Box<[u8]>, Vec<Box<[u8]>>>,
    /// The same, keyed as a document stores a word.
    stored: BTreeMap<Box<[u8]>, Vec<Box<[u8]>>>,
}

impl Synonyms {
    /// An index with no groups on it.
    #[must_use]
    pub fn new() -> Synonyms {
        Synonyms::default()
    }

    /// Whether there are none, which is the question the write path asks before
    /// it does anything at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// How many terms are in a group, counting a term in two groups once.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Puts terms into a group, making the group if it is not there.
    ///
    /// A term that is already in the group is left where it is rather than
    /// being listed twice, and a term already in another group joins this one
    /// as well and belongs to both. An empty term is a term, which is measured:
    /// it dumps as an empty string and no document word ever equals it.
    pub fn update(&mut self, id: &[u8], terms: &[&[u8]]) {
        for raw in terms {
            let word = fold(raw);
            let stored = fold_stored(&word, word.len());
            put(&mut self.words, word, id);
            put(&mut self.stored, stored, id);
        }
    }

    /// The groups a query word is in, as the query folded it.
    #[must_use]
    pub fn asked(&self, word: &[u8]) -> &[Box<[u8]>] {
        self.words.get(word).map_or(&[], Vec::as_slice)
    }

    /// The groups a document word is in, as the document stored it.
    #[must_use]
    pub fn held(&self, word: &[u8]) -> &[Box<[u8]>] {
        self.stored.get(word).map_or(&[], Vec::as_slice)
    }

    /// Every term and the groups it is in, in byte order.
    ///
    /// Byte order because a walk has to have one. A real server answers in
    /// whatever order its hash table holds them, which is D-87.
    pub fn dump(&self) -> impl Iterator<Item = (&[u8], &[Box<[u8]>])> {
        self.words.iter().map(|(term, ids)| (&**term, &**ids))
    }
}

/// Adds a group to a term's list, unless the term is already in that group.
fn put(map: &mut BTreeMap<Box<[u8]>, Vec<Box<[u8]>>>, key: Vec<u8>, id: &[u8]) {
    let ids = map.entry(key.into()).or_default();
    if !ids.iter().any(|had| **had == *id) {
        ids.push(id.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The terms and their groups, as strings, in the order a dump walks them.
    fn dump(syn: &Synonyms) -> Vec<(String, Vec<String>)> {
        syn.dump()
            .map(|(term, ids)| {
                let ids = ids
                    .iter()
                    .map(|id| String::from_utf8_lossy(id).into_owned())
                    .collect();
                (String::from_utf8_lossy(term).into_owned(), ids)
            })
            .collect()
    }

    /// A group id is any bytes at all, so it keeps its case where the terms do
    /// not, and the term it goes into the index as carries it whole.
    #[test]
    fn a_group_is_an_id_and_the_terms_that_are_in_it() {
        let mut syn = Synonyms::new();
        syn.update(b"G1", &[b"Boy", b"CHILD"]);
        assert_eq!(
            dump(&syn),
            [
                ("boy".to_owned(), vec!["G1".to_owned()]),
                ("child".to_owned(), vec!["G1".to_owned()])
            ]
        );
        assert_eq!(term(b"G1"), b"~G1");
        assert_eq!(syn.len(), 2);
        assert!(!syn.is_empty());
    }

    /// A term in two groups carries both, in the order they were added, and
    /// adding it to a group it is already in changes nothing.
    #[test]
    fn a_term_can_be_in_more_than_one_group() {
        let mut syn = Synonyms::new();
        syn.update(b"zz", &[b"boy"]);
        syn.update(b"aa", &[b"boy"]);
        syn.update(b"zz", &[b"boy"]);
        assert_eq!(
            dump(&syn),
            [("boy".to_owned(), vec!["zz".to_owned(), "aa".to_owned()])]
        );
    }

    /// The query side and the document side agree on an ordinary word, and a
    /// word nobody put in a group has no groups on either side.
    #[test]
    fn a_word_is_found_from_the_query_side_and_from_the_document_side() {
        let mut syn = Synonyms::new();
        syn.update(b"g1", &[b"boy"]);
        assert_eq!(syn.asked(b"boy").len(), 1);
        assert_eq!(syn.held(b"boy").len(), 1);
        assert!(syn.asked(b"BOY").is_empty(), "the caller folds first");
        assert!(syn.held(b"girl").is_empty());
        assert!(Synonyms::new().asked(b"boy").is_empty());
    }

    /// A term with a character too wide for an index is kept both ways: whole
    /// for the dump and for a query, and cut down for the documents, which
    /// store every character in sixteen bits.
    #[test]
    fn a_wide_character_is_kept_whole_and_kept_cut() {
        let mut syn = Synonyms::new();
        syn.update(b"g1", &["\u{1f600}".as_bytes()]);
        assert_eq!(
            dump(&syn),
            [("\u{1f600}".to_owned(), vec!["g1".to_owned()])]
        );
        assert_eq!(syn.asked("\u{1f600}".as_bytes()).len(), 1);
        assert_eq!(syn.held("\u{f600}".as_bytes()).len(), 1);
        assert!(syn.held("\u{1f600}".as_bytes()).is_empty());
    }

    /// An empty term goes in and dumps, which is measured, and a group with no
    /// terms at all leaves nothing behind to dump.
    #[test]
    fn an_empty_term_is_a_term_and_no_terms_are_nothing() {
        let mut syn = Synonyms::new();
        syn.update(b"g1", &[b""]);
        assert_eq!(dump(&syn), [(String::new(), vec!["g1".to_owned()])]);
        syn.update(b"g2", &[]);
        assert_eq!(syn.len(), 1);
    }
}
