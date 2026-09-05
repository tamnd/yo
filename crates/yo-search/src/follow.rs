//! Which indexes a key belongs to, and what happens to them when it changes.
//!
//! ```
//! use yo_search::{Definition, Field, Index, Kind, Registry, Source, Text};
//!
//! let mut r = Registry::new();
//! let mut d = Definition::default();
//! d.prefixes = vec![b"book:".to_vec().into_boxed_slice()];
//! let title = Field::new(b"title", Kind::Text(Text::default()));
//! r.create(Index::new(b"books", d, vec![title]))?;
//!
//! // A key under the prefix reaches the index, and one outside it does not.
//! r.wrote(Source::Hash, b"book:1", &[(&b"title"[..], &b"Running dogs"[..])]);
//! r.wrote(Source::Hash, b"other:1", &[(&b"title"[..], &b"Running dogs"[..])]);
//! assert_eq!(r.named(b"books").map(|i| i.held.docs.len()), Some(1));
//!
//! // And a key that goes takes its document with it.
//! r.went(b"book:1");
//! assert_eq!(r.named(b"books").map(|i| i.held.docs.len()), Some(0));
//! # Ok::<(), yo_search::Clash>(())
//! ```
//!
//! # A write is a whole rewrite
//!
//! There is no such thing as updating one field of a document. Any change to a
//! followed key throws its document away and reads the key again from nothing,
//! which is why a document gets a new number every time it is written. Measured
//! against 8.10.1, and it holds for changes that look like they could not
//! matter: writing a field the schema never named still rewrites, and writing a
//! field the value it already had still rewrites. What does not rewrite is a
//! command that changed nothing at all, so an `HSETNX` that found the field
//! there leaves the document alone. So the rule is not "this command was a
//! write", it is "the value under this key is not what it was".
//!
//! # The two ways of emptying a hash are not the same
//!
//! A key that ends up with no fields left is either a refusal or a document
//! with nothing in it, and which one it is depends on how it emptied. `HDEL` of
//! the last field sends the indexes to read a key that is not there, and that
//! is counted in `hash_indexing_failures` with the sentence about a key that
//! does not exist. A deadline that took the last field, which is `HEXPIRE key
//! 0` or `HGETDEL`, writes the document one more time with nothing in it before
//! taking it away, so it spends a number and is counted as nothing. There is no
//! reason for the difference other than that a real server has two code paths,
//! and no way to see it other than through `FT.INFO`, which is where anyone
//! would meet it.
//!
//! `HSETEX` with a deadline that has already passed is the third shape. It is
//! one command and two pieces of news, so `max_doc_id` moves twice and the
//! value it was handed never reaches the index at all, because the field is
//! dead before anything goes to read it.
//!
//! # A key of the wrong type is not a failure
//!
//! An index `ON HASH` passes over a string or a list sitting under its prefix
//! without a word. It is not indexed and it is not counted in
//! `hash_indexing_failures`, which is worth stating because the opposite is the
//! obvious guess and it is wrong.
//!
//! # One document per name, across every database
//!
//! An index follows a key by name and does not care which database it is in.
//! Writing `p:1` on database one replaces the document that `p:1` on database
//! zero had, and deleting either one takes the single document away while the
//! other key is still sitting there. This build says the same thing, because a
//! difference here is one that only turns up in somebody's failover. It follows
//! from the registry being per server rather than per database, which is itself
//! measured rather than chosen.
//!
//! The scan is the other half of that and does not match it. An index reads the
//! database it was created on and no other, so the same key on database one is
//! invisible to it until something writes to that key. Both halves are measured
//! and neither is a choice. It matters less than it reads, because a real
//! server refuses `FT.CREATE` anywhere but on database zero.
//!
//! # What the failures are for
//!
//! A document that cannot be read is counted twice, once against the index and
//! once against the field that could not be read, and both places remember the
//! sentence and the key. That is the `Index Errors` block `FT.INFO` answers
//! with. The counts only ever climb: putting the key right afterwards indexes
//! it but does not take the failure back, and the last error stays the last
//! error until another one replaces it.
//!
//! The check stops at the first field in schema order that will not read, so a
//! document with a bad number and a bad geo pair counts one failure and blames
//! whichever of the two the schema declared first.

use std::collections::BTreeMap;

use crate::english::English;
use crate::held::{Failed, VANISHED};
use crate::index::{Index, Source};
use crate::registry::Registry;

/// What a real server answers where no error has happened yet.
pub const NONE: &str = "N/A";

/// How many keys something refused, and which one it refused last.
///
/// One of these for the index as a whole and one for each field that has ever
/// refused something, which is the shape `FT.INFO` reports and the reason the
/// index count is kept rather than added up from the fields: they agree today
/// and a geo pair that fails two checks at once would make them disagree.
#[derive(Debug, Clone, Default)]
pub struct Errors {
    /// How many keys were refused.
    failures: u64,
    /// The sentence the last refusal produced.
    last: Option<Box<[u8]>>,
    /// The key the last refusal was about.
    key: Option<Box<[u8]>>,
}

impl Errors {
    /// Counts one refusal and remembers it.
    pub fn note(&mut self, key: &[u8], sentence: &str) {
        self.failures += 1;
        self.last = Some(sentence.as_bytes().into());
        self.key = Some(key.into());
    }

    /// How many keys were refused.
    #[must_use]
    pub const fn failures(&self) -> u64 {
        self.failures
    }

    /// The last sentence, or `None` when nothing has been refused.
    #[must_use]
    pub fn last(&self) -> Option<&[u8]> {
        self.last.as_deref()
    }

    /// The last key, or `None` when nothing has been refused.
    #[must_use]
    pub fn key(&self) -> Option<&[u8]> {
        self.key.as_deref()
    }

    /// The last sentence as a reply writes it, which is `N/A` for none.
    #[must_use]
    pub fn sentence(&self) -> &[u8] {
        self.last().unwrap_or(NONE.as_bytes())
    }

    /// The last key as a reply writes it, which is `N/A` for none.
    #[must_use]
    pub fn about(&self) -> &[u8] {
        self.key().unwrap_or(NONE.as_bytes())
    }
}

/// Every key one index has refused, by the index and by the field.
#[derive(Debug, Clone, Default)]
pub struct Trouble {
    /// The index as a whole.
    whole: Errors,
    /// The fields that have refused something, by the name a query calls them.
    ///
    /// Only the ones that have, because a schema of forty fields where one has
    /// ever failed should not carry thirty nine empty counters around. A field
    /// nobody has a row for reads as a fresh [`Errors`].
    fields: BTreeMap<Box<[u8]>, Errors>,
}

impl Trouble {
    /// Counts one refusal against the index and against the field it names.
    pub fn note(&mut self, key: &[u8], failed: &Failed) {
        let sentence = failed.sentence();
        self.whole.note(key, &sentence);
        self.fields
            .entry(failed.field.clone())
            .or_default()
            .note(key, &sentence);
    }

    /// The index's own counters.
    #[must_use]
    pub const fn whole(&self) -> &Errors {
        &self.whole
    }

    /// One field's counters, which are empty for a field that has never failed.
    #[must_use]
    pub fn field(&self, attribute: &[u8]) -> &Errors {
        static EMPTY: Errors = Errors {
            failures: 0,
            last: None,
            key: None,
        };
        self.fields.get(attribute).unwrap_or(&EMPTY)
    }
}

impl Index {
    /// Whether this index follows a key, on its name and its kind alone.
    ///
    /// The `FILTER` is not applied here. It is an expression over the value, it
    /// is not parsed yet, and an index carrying one currently follows everything
    /// its prefixes cover.
    #[must_use]
    pub fn follows(&self, source: Source, key: &[u8]) -> bool {
        self.definition.on == source && self.definition.covers(key)
    }

    /// Reads a key that has changed, and says whether it went in.
    ///
    /// A key that cannot be read is counted and left out, and whatever this
    /// index had for it before is gone either way.
    pub fn wrote(&mut self, english: &mut English, key: &[u8], fields: &[(&[u8], &[u8])]) -> bool {
        match self.write(english, key, fields) {
            Ok(_) => true,
            Err(failed) => {
                self.trouble.note(key, &failed);
                false
            }
        }
    }

    /// Counts a key that was not there when this index went to read it.
    ///
    /// Against the index and against no field, since there is no field in the
    /// schema to blame for a key that is not there.
    pub fn vanished(&mut self, key: &[u8]) {
        let sentence = format!(
            "{VANISHED} Key does not exist or is not a hash: {}",
            String::from_utf8_lossy(key)
        );
        self.trouble.whole.note(key, &sentence);
    }
}

impl Registry {
    /// Tells every index that follows a key that the key has changed.
    ///
    /// The fields are the whole of what is under the key now, not the ones the
    /// command touched, because a document is read again from nothing every
    /// time.
    pub fn wrote(&mut self, source: Source, key: &[u8], fields: &[(&[u8], &[u8])]) {
        let (indexes, english) = self.reading();
        for index in indexes {
            if index.follows(source, key) {
                index.wrote(english, key, fields);
            }
        }
    }

    /// Tells every index that a key is gone.
    ///
    /// Every index and not only the ones that follow it, because a key can stop
    /// being followed without being deleted. `RENAME` out of a prefix and a
    /// `HSET` that trips a filter both land here, and an index that never had
    /// the key has nothing to do about being told.
    pub fn went(&mut self, key: &[u8]) {
        for index in self.reading().0 {
            index.erase(key);
        }
    }

    /// Tells every index that follows a key that the key was not there when it
    /// went to read it, which is counted as a refusal.
    ///
    /// One command does this and it is `HDEL` taking the last field, which
    /// leaves nothing under the key to read. A real server counts that as an
    /// indexing failure, where the same key emptied by a deadline is not
    /// counted at all, and the two are worth telling apart because the counter
    /// is in `FT.INFO` and it never goes back down.
    ///
    /// No field is blamed, because there is no field to blame. A bad number
    /// names the field it was in and this names nothing, which is the shape a
    /// real server reports as well.
    pub fn vanished(&mut self, source: Source, key: &[u8]) {
        for index in self.reading().0 {
            if index.follows(source, key) {
                index.vanished(key);
            }
            index.erase(key);
        }
    }

    /// Whether reading this key back is worth doing at all.
    ///
    /// The question the write path asks before it goes and fetches a hash it
    /// has already written. A server with no indexes on it, which is nearly
    /// every server, answers no after one look at an empty vector.
    #[must_use]
    pub fn follows(&self, source: Source, key: &[u8]) -> bool {
        self.iter().any(|index| index.follows(source, key))
    }

    /// Whether any index would want to hear that a key is gone.
    #[must_use]
    pub fn watching(&self) -> bool {
        !self.is_empty()
    }

    /// Whether one named index wants a key, which is what the initial scan asks
    /// before it goes and reads one back.
    ///
    /// The exact name and never an alias, because the only caller is
    /// `FT.CREATE` handing back the name it just made.
    #[must_use]
    pub fn wants(&self, name: &[u8], source: Source, key: &[u8]) -> bool {
        self.named(name)
            .is_some_and(|index| index.follows(source, key))
    }

    /// Reads a key into one index and leaves every other index alone.
    ///
    /// The scan a fresh index runs over the keys that were already there.
    /// [`Registry::wrote`] is the wrong thing for it: an `FT.CREATE` over a
    /// prefix that another index already covers would renumber every document
    /// that one holds, and a real server does not do that to an index nobody
    /// touched.
    pub fn filled(
        &mut self,
        name: &[u8],
        source: Source,
        key: &[u8],
        fields: &[(&[u8], &[u8])],
    ) -> bool {
        let (mut indexes, english) = self.reading();
        let Some(index) = indexes.find(|index| &*index.name == name) else {
            return false;
        };
        index.follows(source, key) && index.wrote(english, key, fields)
    }

    /// Whether an index by this name is one the scan should run for.
    ///
    /// `SKIPINITIALSCAN` says no, and so does a name that is not there, which
    /// is what a failed create leaves behind.
    #[must_use]
    pub fn scanning(&self, name: &[u8]) -> bool {
        self.named(name)
            .is_some_and(|index| !index.definition.skip_initial_scan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field, Kind, Tag, Text};
    use crate::index::{Definition, Index};
    use crate::nums::Nums;
    use crate::tags::Tags;

    fn on(prefix: &[u8], fields: Vec<Field>) -> Index {
        let definition = Definition {
            prefixes: vec![prefix.into()],
            ..Definition::default()
        };
        Index::new(b"ix", definition, fields)
    }

    fn text() -> Field {
        Field::new(b"t", Kind::Text(Text::default()))
    }

    fn registry(index: Index) -> Registry {
        let mut r = Registry::new();
        r.create(index).expect("the name is free");
        r
    }

    /// The prefix decides, and the kind of key decides with it.
    #[test]
    fn an_index_follows_the_keys_its_prefix_covers() {
        let ix = on(b"p:", vec![text()]);
        assert!(ix.follows(Source::Hash, b"p:1"));
        assert!(ix.follows(Source::Hash, b"p:"));
        assert!(!ix.follows(Source::Hash, b"q:1"));
        assert!(!ix.follows(Source::Hash, b"p"));
        // An index over hashes has nothing to say about a document.
        assert!(!ix.follows(Source::Json, b"p:1"));
    }

    /// No prefix at all is one empty prefix, which every key starts with.
    #[test]
    fn an_index_with_no_prefix_follows_everything() {
        let ix = Index::new(b"ix", Definition::default(), vec![text()]);
        assert!(ix.follows(Source::Hash, b"anything"));
        assert!(ix.follows(Source::Hash, b""));
    }

    /// A write reaches the indexes that cover the key and no others.
    #[test]
    fn a_write_reaches_the_indexes_that_cover_it() {
        let mut r = Registry::new();
        r.create(on(b"p:", vec![text()])).expect("free");
        let mut other = on(b"q:", vec![text()]);
        other.name = b"jx".to_vec().into();
        r.create(other).expect("free");

        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(1));
        assert_eq!(r.named(b"jx").map(|i| i.held.docs.len()), Some(0));
        assert!(r.follows(Source::Hash, b"q:1"));
        assert!(!r.follows(Source::Hash, b"r:1"));
    }

    /// Writing a key again gives its document a new number, whatever changed,
    /// because a document is read from nothing every time.
    #[test]
    fn a_second_write_is_a_whole_rewrite() {
        let mut r = registry(on(b"p:", vec![text()]));
        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        assert_eq!(r.named(b"ix").and_then(|i| i.held.docs.id(b"p:1")), Some(1));

        // The same value it already had, and it still moves.
        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        assert_eq!(r.named(b"ix").and_then(|i| i.held.docs.id(b"p:1")), Some(2));

        // A field the schema never named, and it moves again.
        r.wrote(
            Source::Hash,
            b"p:1",
            &[(&b"t"[..], &b"alpha"[..]), (&b"other"[..], &b"x"[..])],
        );
        assert_eq!(r.named(b"ix").and_then(|i| i.held.docs.id(b"p:1")), Some(3));
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(1));
    }

    /// A key that goes is taken out of every index, whether or not it is one
    /// they still cover, since falling out of the prefix is how it usually
    /// happens.
    #[test]
    fn a_key_that_goes_is_taken_out_everywhere() {
        let mut r = registry(on(b"p:", vec![text()]));
        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        r.wrote(Source::Hash, b"p:2", &[(&b"t"[..], &b"beta"[..])]);
        assert!(r.watching());

        r.went(b"p:1");
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(1));
        assert_eq!(r.named(b"ix").and_then(|i| i.held.docs.id(b"p:1")), None);
        // Telling it about a key it never had is not an error.
        r.went(b"nothing");
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(1));
    }

    /// A bad number is counted against the index and against the field, and the
    /// document it would have been is not there.
    #[test]
    fn a_key_that_will_not_read_is_counted_twice() {
        let num = Field::new(b"n", Kind::Numeric);
        let mut r = registry(on(b"p:", vec![text(), num]));
        r.wrote(
            Source::Hash,
            b"p:1",
            &[(&b"t"[..], &b"alpha"[..]), (&b"n"[..], &b"notanumber"[..])],
        );

        let ix = r.named(b"ix").expect("the index is there");
        assert_eq!(ix.held.docs.len(), 0);
        // The number was never handed out, so the next document gets the first.
        assert_eq!(ix.held.docs.last(), 0);
        assert_eq!(ix.trouble.whole().failures(), 1);
        assert_eq!(
            ix.trouble.whole().sentence(),
            b"SEARCH_NUMERIC_VALUE_INVALID Invalid numeric value: 'notanumber'"
        );
        assert_eq!(ix.trouble.whole().about(), b"p:1");
        assert_eq!(ix.trouble.field(b"n").failures(), 1);
        assert_eq!(ix.trouble.field(b"t").failures(), 0);
        assert_eq!(ix.trouble.field(b"t").sentence(), NONE.as_bytes());
        assert_eq!(ix.trouble.field(b"t").about(), NONE.as_bytes());
    }

    /// A key that was good and goes bad loses the document it had, and putting
    /// it right afterwards does not take the failure back.
    #[test]
    fn a_key_that_goes_bad_loses_what_it_had_and_keeps_the_count() {
        let num = Field::new(b"n", Kind::Numeric);
        let mut r = registry(on(b"p:", vec![num]));
        r.wrote(Source::Hash, b"p:1", &[(&b"n"[..], &b"5"[..])]);
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(1));

        r.wrote(Source::Hash, b"p:1", &[(&b"n"[..], &b"bad"[..])]);
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(0));

        r.wrote(Source::Hash, b"p:1", &[(&b"n"[..], &b"7"[..])]);
        let ix = r.named(b"ix").expect("the index is there");
        assert_eq!(ix.held.docs.len(), 1);
        assert_eq!(ix.trouble.whole().failures(), 1);
        assert_eq!(ix.trouble.whole().about(), b"p:1");
        assert_eq!(
            ix.trouble.whole().sentence(),
            b"SEARCH_NUMERIC_VALUE_INVALID Invalid numeric value: 'bad'"
        );
    }

    /// Every refusal counts, including the same key refused twice, and the last
    /// one is the one that is remembered.
    #[test]
    fn every_refusal_counts_and_the_last_one_is_kept() {
        let num = Field::new(b"n", Kind::Numeric);
        let mut r = registry(on(b"p:", vec![num]));
        r.wrote(Source::Hash, b"p:1", &[(&b"n"[..], &b"bad"[..])]);
        r.wrote(Source::Hash, b"p:1", &[(&b"n"[..], &b"stillnot"[..])]);
        r.wrote(Source::Hash, b"p:2", &[(&b"n"[..], &b""[..])]);

        let ix = r.named(b"ix").expect("the index is there");
        assert_eq!(ix.trouble.whole().failures(), 3);
        assert_eq!(ix.trouble.field(b"n").failures(), 3);
        assert_eq!(ix.trouble.whole().about(), b"p:2");
        assert_eq!(
            ix.trouble.whole().sentence(),
            b"SEARCH_NUMERIC_VALUE_INVALID Invalid numeric value: ''"
        );
    }

    /// A document with a tag and a number lands in all three indexes, which is
    /// the whole point of the fan out.
    #[test]
    fn a_write_fills_every_index_the_schema_asked_for() {
        let tag = Field::new(b"g", Kind::Tag(Tag::default()));
        let num = Field::new(b"n", Kind::Numeric);
        let mut r = registry(on(b"p:", vec![text(), tag, num]));
        r.wrote(
            Source::Hash,
            b"p:1",
            &[
                (&b"t"[..], &b"running dogs"[..]),
                (&b"g"[..], &b"red,blue"[..]),
                (&b"n"[..], &b"42"[..]),
            ],
        );

        let ix = r.named(b"ix").expect("the index is there");
        assert_eq!(ix.held.terms().count(), 4);
        assert_eq!(ix.held.values(b"g").map(Tags::len), Some(2));
        assert_eq!(ix.held.numbers(b"n").map(Nums::len), Some(1));
        assert_eq!(ix.held.docs.get(1).map(|d| d.tokens), Some(2));
    }

    /// The two counts `FT.INFO` reports about the index, against the sequence a
    /// real server answered for the same five writes.
    ///
    /// The terms count the dictionary, stems included, so `running dogs` is
    /// four of them. The records count every entry in all three indexes, so a
    /// number is one, a two value tag is two, and a second document with a term
    /// somebody else already had is one more.
    #[test]
    fn the_terms_and_the_records_are_counted_the_way_a_real_server_counts_them() {
        let tag = Field::new(b"g", Kind::Tag(Tag::default()));
        let num = Field::new(b"n", Kind::Numeric);
        let mut r = registry(on(b"p:", vec![text(), num, tag]));
        let counts = |r: &Registry| {
            let ix = r.named(b"ix").expect("the index is there");
            (ix.held.docs.len(), ix.held.words(), ix.held.records())
        };

        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        assert_eq!(counts(&r), (1, 1, 1));
        r.wrote(Source::Hash, b"p:2", &[(&b"t"[..], &b"running dogs"[..])]);
        assert_eq!(counts(&r), (2, 5, 5));
        r.wrote(Source::Hash, b"p:3", &[(&b"n"[..], &b"5"[..])]);
        assert_eq!(counts(&r), (3, 5, 6));
        r.wrote(Source::Hash, b"p:4", &[(&b"g"[..], &b"red,blue"[..])]);
        assert_eq!(counts(&r), (4, 5, 8));
        r.wrote(Source::Hash, b"p:5", &[(&b"t"[..], &b"alpha"[..])]);
        assert_eq!(counts(&r), (5, 5, 9));
    }

    /// The scan fills the index that was just made and does not disturb the one
    /// that was already over the same prefix.
    #[test]
    fn a_scan_fills_one_index_and_leaves_the_others_where_they_were() {
        let mut r = registry(on(b"p:", vec![text()]));
        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        let mut second = on(b"p:", vec![text()]);
        second.name = b"jx".to_vec().into();
        r.create(second).expect("free");

        assert!(r.wants(b"jx", Source::Hash, b"p:1"));
        assert!(!r.wants(b"jx", Source::Hash, b"q:1"));
        assert!(!r.wants(b"nope", Source::Hash, b"p:1"));
        assert!(r.filled(b"jx", Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]));
        assert!(!r.filled(b"jx", Source::Hash, b"q:1", &[(&b"t"[..], &b"alpha"[..])]));
        assert!(!r.filled(b"nope", Source::Hash, b"p:1", &[]));

        // The new one has it, and the old one still has the number it had.
        assert_eq!(r.named(b"jx").and_then(|i| i.held.docs.id(b"p:1")), Some(1));
        assert_eq!(r.named(b"ix").and_then(|i| i.held.docs.id(b"p:1")), Some(1));
    }

    /// `SKIPINITIALSCAN` is the one thing that turns the scan off, and a name
    /// that is not there answers the same way.
    #[test]
    fn an_index_that_asked_to_be_left_alone_is_not_scanned() {
        let mut skipping = on(b"p:", vec![text()]);
        skipping.definition.skip_initial_scan = true;
        let mut r = registry(skipping);
        assert!(!r.scanning(b"ix"));
        assert!(!r.scanning(b"nope"));

        let mut second = on(b"p:", vec![text()]);
        second.name = b"jx".to_vec().into();
        r.create(second).expect("free");
        assert!(r.scanning(b"jx"));
    }

    /// A key that was not there when the index went to read it is counted
    /// against the index, is not counted against any field, and takes whatever
    /// document it had with it.
    #[test]
    fn a_key_that_is_not_there_is_counted_and_erased() {
        let mut r = registry(on(b"p:", vec![text()]));
        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        assert_eq!(r.named(b"ix").map(|i| i.held.docs.len()), Some(1));

        r.vanished(Source::Hash, b"p:1");
        let ix = r.named(b"ix").expect("the index is there");
        assert_eq!(ix.held.docs.len(), 0);
        assert_eq!(ix.trouble.whole().failures(), 1);
        assert_eq!(ix.trouble.whole().key(), Some(&b"p:1"[..]));
        assert_eq!(
            ix.trouble.whole().last(),
            Some(&b"SEARCH_QUERY_BAD Key does not exist or is not a hash: p:1"[..])
        );
        assert_eq!(ix.trouble.field(b"t").failures(), 0);

        // A key no index follows is erased and counted against nobody.
        r.vanished(Source::Hash, b"q:1");
        assert_eq!(
            r.named(b"ix").map(|i| i.trouble.whole().failures()),
            Some(1)
        );
    }

    /// An empty registry answers no to everything, which is what keeps the
    /// write path from paying for a feature nobody turned on.
    #[test]
    fn a_server_with_no_indexes_wants_nothing() {
        let mut r = Registry::new();
        assert!(!r.follows(Source::Hash, b"p:1"));
        assert!(!r.watching());
        r.wrote(Source::Hash, b"p:1", &[(&b"t"[..], &b"alpha"[..])]);
        r.went(b"p:1");
        assert!(r.is_empty());
    }
}
