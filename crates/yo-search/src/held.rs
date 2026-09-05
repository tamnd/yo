//! What an index has read: the documents, the terms, the numbers and the
//! values, and the one routine that puts a document into all four.
//!
//! ```
//! use yo_search::english::English;
//! use yo_search::field::{Field, Kind, Text};
//! use yo_search::index::{Definition, Index};
//!
//! let mut english = English::new();
//! let title = Field::new(b"title", Kind::Text(Text::default()));
//! let mut index = Index::new(b"books", Definition::default(), vec![title]);
//!
//! let id = index.write(&mut english, b"book:1", &[(b"title", b"Running dogs")])?;
//! assert_eq!(id, 1);
//! assert_eq!(index.held.docs.get(id).map(|d| d.tokens), Some(2));
//! // The word and the stem it was given, which is what makes a search for
//! // `run` find a document that only ever said `running`.
//! assert_eq!(index.held.terms().collect::<Vec<_>>(), [
//!     &b"+dog"[..],
//!     &b"+run"[..],
//!     &b"dogs"[..],
//!     &b"running"[..]
//! ]);
//! # Ok::<(), yo_search::held::Failed>(())
//! ```
//!
//! # What a document costs
//!
//! Every word of every text field is folded, stemmed, given a place and written
//! into the list for its term, and the document is given a number, a score, a
//! length and a largest frequency at the same time. The length is the sum of
//! what each term is worth in it, where a word in a field of weight three is
//! worth three, and the stems are not in it. That is the `num_tokens` a real
//! server reports and it is what the scoring divides by, so it has to be counted
//! here and not worked out later.
//!
//! Places run across the whole document rather than restarting per field, in the
//! order the schema declares the fields and not the order the hash holds them.
//! So a document with `alpha` in one field and `beta` in the next matches the
//! phrase `"alpha beta"`, which is easy to mistake for a bug and is what a real
//! server does.
//!
//! A stem is worth one wherever it is found, whatever the field weight is. A
//! word twice in a field of weight three is a frequency of six and its stem is a
//! frequency of two, both measured rather than assumed.
//!
//! # What a bad number costs
//!
//! The whole document. A numeric field that will not parse is not left out of
//! the document, it loses the document: nothing is indexed, no number is handed
//! out, and a reading that was there before is gone. A real server counts that
//! in `hash_indexing_failures` and remembers the sentence and the key, which is
//! why [`Failed`] carries both halves of it rather than being a bare `None`.
//!
//! Everything else is forgiving. A hash with none of the schema's fields in it
//! is a document of no tokens rather than a failure, an empty text field is the
//! same, and a score field that is not a number leaves the document with the
//! score the index gives everything else.

use std::collections::BTreeMap;

use crate::docs::Docs;
use crate::english::English;
use crate::field::Kind;
use crate::index::Index;
use crate::nums::Nums;
use crate::posts::{Id, Posts, Terms, adds, stemmed};
use crate::score::Facts;
use crate::sorted::Sorted;
use crate::tags::Tags;
use crate::words::{Words, stem};

/// The name a real server gives the error a bad number raises.
pub const NUMERIC: &str = "SEARCH_NUMERIC_VALUE_INVALID";

/// And the one it gives a key that was not there when it went to read it.
pub const VANISHED: &str = "SEARCH_QUERY_BAD";

/// Why a document was not indexed.
///
/// One reason so far, because a number is the only thing in a schema that can
/// be written down wrong. A geo pair is the next one and it is not read yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failed {
    /// The field that could not be read, by the name the schema gave it.
    pub field: Box<[u8]>,
    /// The value that could not be read.
    pub value: Box<[u8]>,
}

impl Failed {
    /// The sentence a real server records as the last indexing error.
    #[must_use]
    pub fn sentence(&self) -> String {
        format!(
            "{NUMERIC} Invalid numeric value: '{}'",
            String::from_utf8_lossy(&self.value)
        )
    }
}

/// Everything one index has read out of the keyspace.
///
/// The terms are one dictionary for the whole index, each with a posting list
/// carrying the field it came from as a bit, which is what lets `@title:dogs`
/// and `dogs` read the same list. The numbers and the values are per field,
/// because a range and a tag are asked about one field at a time and there is
/// nothing to gain by mixing them.
#[derive(Debug, Clone, Default)]
pub struct Held {
    /// The documents, by number and by key.
    pub docs: Docs,
    /// Every term in the index and the documents that hold it.
    terms: Terms,
    /// Every `NUMERIC` field's values, by the name a query calls the field.
    nums: BTreeMap<Box<[u8]>, Nums>,
    /// Every `TAG` field's values, by the name a query calls the field.
    tags: BTreeMap<Box<[u8]>, Tags>,
}

impl Held {
    /// An index with nothing read into it.
    #[must_use]
    pub fn new() -> Held {
        Held::default()
    }

    /// Every term, in byte order, which is the order a dump answers in.
    pub fn terms(&self) -> impl Iterator<Item = &[u8]> {
        self.terms.all()
    }

    /// The whole term dictionary, for a walk that needs the lists as well.
    #[must_use]
    pub const fn dictionary(&self) -> &Terms {
        &self.terms
    }

    /// The documents a term is in, or `None` when no document has it.
    #[must_use]
    pub fn posts(&self, term: &[u8]) -> Option<&Posts> {
        self.terms.get(term)
    }

    /// One field's numbers.
    #[must_use]
    pub fn numbers(&self, attribute: &[u8]) -> Option<&Nums> {
        self.nums.get(attribute)
    }

    /// One field's tag values.
    #[must_use]
    pub fn values(&self, attribute: &[u8]) -> Option<&Tags> {
        self.tags.get(attribute)
    }

    /// Folds every numeric field's fresh writes into its ordered list.
    ///
    /// Worth doing after a batch of documents and not after each one, which is
    /// the whole reason [`Nums::settle`] is a separate call.
    pub fn settle(&mut self) {
        for nums in self.nums.values_mut() {
            nums.settle();
        }
    }

    /// What the scoring needs to know about the index as a whole.
    #[must_use]
    pub fn facts(&self) -> Facts {
        Facts::new(self.docs.len() as u32, self.docs.tokens())
    }

    /// How many terms there are, stems counted apart from their words.
    ///
    /// The `num_terms` a real server reports, which counts the whole
    /// dictionary and so counts `dogs` and `+dog` as two.
    #[must_use]
    pub fn words(&self) -> usize {
        self.terms.len()
    }

    /// How many entries all three kinds of index hold between them.
    ///
    /// The `num_records` a real server reports. It is not the number of terms
    /// and it is not the number of documents, it is every place any of the
    /// three says a document has something: one per document per term, one per
    /// number, and one per tag value. Measured, because a hash with one text
    /// field, one number and a two value tag answers four the first time it is
    /// written and the arithmetic only works if all three are counted.
    #[must_use]
    pub fn records(&self) -> u64 {
        let terms: u64 = self
            .terms
            .all()
            .filter_map(|term| self.posts(term))
            .map(|posts| u64::from(posts.len()))
            .sum();
        let nums: u64 = self.nums.values().map(|n| n.len() as u64).sum();
        let tags: u64 = self.tags.values().map(|t| t.entries() as u64).sum();
        terms + nums + tags
    }

    /// Throws everything away, which is what dropping the index does.
    pub fn clear(&mut self) {
        self.docs = Docs::new();
        self.terms = Terms::new();
        self.nums.clear();
        self.tags.clear();
    }
}

/// One term as one document holds it, while the document is being read.
#[derive(Debug, Default)]
struct Entry {
    /// The sum of what every occurrence of it is worth.
    freq: u32,
    /// Which fields it was found in, one bit each.
    mask: u32,
    /// Where it was found, in order and counting from one.
    at: Vec<u32>,
    /// Whether this is a stem rather than a word, which is what keeps it out of
    /// the document's length.
    stem: bool,
}

impl Index {
    /// Reads a document into the index and gives back the number it got.
    ///
    /// The fields are the key's own, as pairs of a name and a value, and they
    /// are looked up by the identifier the schema declared rather than by the
    /// name a query uses, which are the same bytes unless the client said `AS`.
    /// A key that was read before is read again from nothing: its old number
    /// stops meaning anything and it gets a new one.
    ///
    /// # Errors
    ///
    /// [`Failed`] when a numeric field holds something that is not a number, in
    /// which case the document is not in the index at all and no number was
    /// handed out.
    pub fn write(
        &mut self,
        english: &mut English,
        key: &[u8],
        fields: &[(&[u8], &[u8])],
    ) -> Result<Id, Failed> {
        let Index {
            definition,
            schema,
            held,
            ..
        } = self;
        // The old reading goes first and it goes whatever happens next, because
        // a rewrite that cannot be indexed leaves the key out of the index
        // rather than leaving what was there before.
        held.docs.remove(key);
        // Every number is read before anything is written, since one that will
        // not parse loses the document and a half indexed document would leave
        // terms pointing at a number nobody handed out.
        let mut numbers = Vec::new();
        for field in schema.iter().filter(|f| !f.noindex) {
            if field.kind != Kind::Numeric {
                continue;
            }
            let Some(raw) = value(fields, &field.identifier) else {
                continue;
            };
            let Some(number) = number(raw) else {
                return Err(Failed {
                    field: field.attribute.clone(),
                    value: raw.into(),
                });
            };
            numbers.push((field.attribute.clone(), number));
        }

        let score = definition
            .score_field
            .as_deref()
            .and_then(|name| value(fields, name))
            .and_then(number)
            .unwrap_or(definition.score);
        let id = held.docs.add(key, score);
        if let Some(raw) = definition
            .payload_field
            .as_deref()
            .and_then(|name| value(fields, name))
        {
            held.docs.carry(id, raw);
        }
        for (attribute, number) in numbers {
            held.nums.entry(attribute).or_default().add(id, number);
        }
        // The copies a sort reads instead of reading the key back, one per
        // sortable field in the order the schema declares them. A `NOINDEX`
        // field is in here, unlike everywhere else in this routine, because
        // `NOINDEX SORTABLE` is a field that cannot be matched and can still be
        // sorted by and returned, which is measured.
        let sortable: Vec<Option<Sorted>> = schema
            .iter()
            .filter(|f| f.sortable)
            .map(|f| value(fields, &f.identifier).and_then(|raw| Sorted::read(f, raw)))
            .collect();
        held.docs.store(id, sortable);

        let stops = definition.stopwords.as_deref();
        let mut found: BTreeMap<Box<[u8]>, Entry> = BTreeMap::new();
        let mut place = 0;
        let mut texts = 0;
        for field in schema.iter().filter(|f| !f.noindex) {
            let text = match &field.kind {
                Kind::Text(text) => text,
                Kind::Tag(tag) => {
                    if let Some(raw) = value(fields, &field.identifier) {
                        held.tags
                            .entry(field.attribute.clone())
                            .or_default()
                            .index(id, raw, tag);
                    }
                    continue;
                }
                // A number is already in, and a coordinate pair, a shape and a
                // vector are not read yet.
                _ => continue,
            };
            // The bit a query means by this field. Counted over the text fields
            // and not over the schema, because they are the only ones a mask is
            // ever asked about.
            let mask = 1 << texts.min(31);
            texts += 1;
            let Some(raw) = value(fields, &field.identifier) else {
                continue;
            };
            let worth = adds(text.weight);
            let mut words = Words::new(raw, stops);
            for word in words.by_ref() {
                let at = place + word.at;
                if !text.nostem
                    && let Some(root) = stem(english, &word.text)
                {
                    let entry = found.entry(stemmed(&root).into()).or_default();
                    entry.stem = true;
                    // One wherever it is found, whatever the field is worth,
                    // which is not what the word itself gets.
                    entry.freq += 1;
                    entry.mask |= mask;
                    entry.at.push(at);
                }
                let entry = found.entry(word.text).or_default();
                entry.freq += worth;
                entry.mask |= mask;
                entry.at.push(at);
            }
            // The places a field took, including the ones taken by a word too
            // long to index, so the next field starts past all of them.
            place += words.seen();
        }

        for (term, entry) in found {
            if !entry.stem {
                held.docs.note(id, entry.freq);
            }
            held.terms.add(&term, id, entry.freq, entry.mask, &entry.at);
        }
        Ok(id)
    }

    /// Takes a key out of the index, giving back the number it had.
    ///
    /// The terms it was in are left alone, the same as a rewrite, because a
    /// number that means nothing is skipped by whoever walks past it and going
    /// back over every list to take it out is the cost this whole design avoids.
    pub fn erase(&mut self, key: &[u8]) -> Option<Id> {
        self.held.docs.remove(key)
    }
}

/// The value of one field of a document, by the name the schema reads it under.
///
/// A walk and not a lookup. A hash has a handful of fields, the schema has a
/// handful of columns, and building a map of one to look the other up in costs
/// more than the walk it saves.
fn value<'a>(fields: &[(&'a [u8], &'a [u8])], name: &[u8]) -> Option<&'a [u8]> {
    fields
        .iter()
        .find(|(field, _)| *field == name)
        .map(|(_, value)| *value)
}

/// A field read as a number, the way a real server reads one.
///
/// The whole value or nothing, so `2.5` and `1e3` and `inf` are numbers and
/// `7x` and an empty value and a value with a space around it are not. A NaN is
/// not one either, which is why a score field holding `nan` leaves the document
/// with the index's own score rather than a score nothing compares to.
fn number(raw: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(raw).ok()?;
    let number: f64 = text.parse().ok()?;
    (!number.is_nan()).then_some(number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field, Tag, Text};
    use crate::index::Definition;
    use crate::nums::Ends;
    use crate::posts::Post;
    use crate::score::{Facts, Found, Scorer, Term};

    /// A text field with a weight and nothing else said about it.
    fn text(name: &[u8], weight: f64) -> Field {
        Field::new(
            name,
            Kind::Text(Text {
                weight,
                ..Text::default()
            }),
        )
    }

    /// The two field index every measurement below was taken against, with the
    /// title worth three and the body worth one.
    fn index() -> Index {
        Index::new(
            b"ix",
            Definition::default(),
            vec![
                text(b"title", 3.0),
                text(b"body", 1.0),
                Field::new(b"n", Kind::Numeric),
                Field::new(b"g", Kind::Tag(Tag::default())),
            ],
        )
    }

    /// The whole entry for a term, as one document holds it.
    fn post(index: &Index, term: &[u8], want: Id) -> (Post, Vec<u32>) {
        let posts = index.held.posts(term).expect("the term is in the index");
        let mut reader = posts.read();
        let post = reader.seek(want).expect("the document holds it");
        let mut places = Vec::new();
        reader.places(&mut places);
        (post, places)
    }

    /// The two documents a real server was asked about, read the same way here.
    /// Its answers were 9 tokens and a largest frequency of 4 for the first and
    /// 5 and 4 for the second, with the score field read off the hash.
    #[test]
    fn a_document_is_as_long_as_what_is_worth_in_it() {
        let mut english = English::new();
        let mut ix = index();
        ix.definition.score_field = Some(b"sc".as_slice().into());
        let one = ix
            .write(
                &mut english,
                b"d:1",
                &[
                    (b"title", b"Running dogs"),
                    (b"body", b"the dogs are running fast"),
                    (b"n", b"42"),
                    (b"sc", b"0.5"),
                    (b"g", b"a,b"),
                ],
            )
            .expect("it indexes");
        let two = ix
            .write(
                &mut english,
                b"d:2",
                &[
                    (b"title", b"Cats"),
                    (b"body", b"cats and dogs"),
                    (b"n", b"7"),
                    (b"g", b"b"),
                ],
            )
            .expect("it indexes");
        assert_eq!((one, two), (1, 2));

        let doc = ix.held.docs.get(one).expect("it is there");
        assert_eq!(doc.tokens, 9);
        assert_eq!(doc.top, 4);
        assert!((doc.score - 0.5).abs() < f64::EPSILON);
        let doc = ix.held.docs.get(two).expect("it is there");
        assert_eq!(doc.tokens, 5);
        assert_eq!(doc.top, 4);
        assert!((doc.score - 1.0).abs() < f64::EPSILON);

        // The same seven terms a real server dumped for these two documents,
        // the stems among them under the plus they are kept behind.
        assert_eq!(
            ix.held.terms().collect::<Vec<_>>(),
            [
                &b"+cat"[..],
                &b"+dog"[..],
                &b"+run"[..],
                &b"cats"[..],
                &b"dogs"[..],
                &b"fast"[..],
                &b"running"[..]
            ]
        );
        assert!(ix.held.posts(b"cat").is_none(), "the word was never in");
    }

    /// The frequencies a real server answered with, read off it a term at a
    /// time under the scorer that gives the frequency and nothing else.
    #[test]
    fn a_word_is_worth_its_field_and_a_stem_is_worth_one() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(
            &mut english,
            b"s:1",
            &[(b"title", b"running running"), (b"body", b"x")],
        )
        .expect("it indexes");
        ix.write(
            &mut english,
            b"s:2",
            &[(b"title", b"walking"), (b"body", b"walking walking")],
        )
        .expect("it indexes");

        assert_eq!(post(&ix, b"running", 1).0.freq, 6);
        assert_eq!(post(&ix, b"+run", 1).0.freq, 2);
        assert_eq!(post(&ix, b"x", 1).0.freq, 1);
        assert_eq!(post(&ix, b"walking", 2).0.freq, 5);
        assert_eq!(post(&ix, b"+walk", 2).0.freq, 3);
        // The stems are not in the length, so the first is the six of its two
        // titles and the one of its body.
        assert_eq!(ix.held.docs.get(1).map(|d| d.tokens), Some(7));
        assert_eq!(ix.held.docs.get(2).map(|d| d.tokens), Some(5));
    }

    /// Places run on from one field into the next, in schema order, which is
    /// what makes a phrase match across two fields that have nothing to do with
    /// each other. A real server matched `"alpha beta"` here and not
    /// `"beta alpha"`, whichever order the hash held the fields in.
    #[test]
    fn places_run_across_the_fields_in_schema_order() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(
            &mut english,
            b"p:1",
            &[(b"body", b"beta"), (b"title", b"alpha")],
        )
        .expect("it indexes");
        assert_eq!(post(&ix, b"alpha", 1).1, [1]);
        assert_eq!(post(&ix, b"beta", 1).1, [2]);

        ix.write(
            &mut english,
            b"p:3",
            &[(b"title", b"one two"), (b"body", b"three")],
        )
        .expect("it indexes");
        assert_eq!(post(&ix, b"one", 2).1, [1]);
        assert_eq!(post(&ix, b"two", 2).1, [2]);
        assert_eq!(post(&ix, b"three", 2).1, [3]);
    }

    /// A stop word takes no place and a word too long to index takes one, so
    /// the field after it starts where it would have anyway.
    #[test]
    fn what_is_dropped_still_settles_where_the_next_field_starts() {
        let mut english = English::new();
        let mut ix = index();
        let long = String::from_utf8(vec![b'q'; 300]).expect("ascii");
        let title = format!("the ok {long}");
        ix.write(
            &mut english,
            b"k:1",
            &[(b"title", title.as_bytes()), (b"body", b"after")],
        )
        .expect("it indexes");
        // `the` is a stop word and took nothing, `ok` is first, the long word
        // took the second place without being indexed, so the next field is at
        // three.
        assert_eq!(post(&ix, b"ok", 1).1, [1]);
        assert_eq!(post(&ix, b"after", 1).1, [3]);
    }

    /// A query asks about one field by the bit the term carries, and a term in
    /// two fields carries both.
    #[test]
    fn a_term_remembers_which_fields_it_came_from() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(
            &mut english,
            b"m:1",
            &[(b"title", b"dogs"), (b"body", b"dogs and cats")],
        )
        .expect("it indexes");
        assert_eq!(post(&ix, b"dogs", 1).0.fields, 0b11);
        assert_eq!(post(&ix, b"cats", 1).0.fields, 0b10);
    }

    /// A number that will not parse loses the document, which a real server
    /// counts as an indexing failure and answers `DOCINFO` on with nothing.
    #[test]
    fn a_number_that_will_not_parse_loses_the_document() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(&mut english, b"f:1", &[(b"title", b"word"), (b"n", b"1")])
            .expect("it indexes");
        let failed = ix
            .write(&mut english, b"f:1", &[(b"title", b"word"), (b"n", b"bad")])
            .expect_err("it does not index");
        assert_eq!(&*failed.field, b"n");
        assert_eq!(&*failed.value, b"bad");
        assert_eq!(
            failed.sentence(),
            "SEARCH_NUMERIC_VALUE_INVALID Invalid numeric value: 'bad'"
        );
        // The reading that was there is gone and no number was handed out for
        // the one that failed, which is what the reference does.
        assert_eq!(ix.held.docs.id(b"f:1"), None);
        assert_eq!(ix.held.docs.last(), 1);
        assert!(ix.held.docs.is_empty());
    }

    /// A hash with none of the schema's fields in it is a document all the
    /// same, of no tokens, and so is one whose only field is empty.
    #[test]
    fn a_document_with_nothing_to_index_is_still_a_document() {
        let mut english = English::new();
        let mut ix = index();
        let bare = ix
            .write(&mut english, b"n:1", &[(b"other", b"stuff")])
            .expect("it indexes");
        let empty = ix
            .write(&mut english, b"n:4", &[(b"title", b"")])
            .expect("it indexes");
        assert_eq!((bare, empty), (1, 2));
        assert_eq!(ix.held.docs.get(bare).map(|d| d.tokens), Some(0));
        assert_eq!(ix.held.docs.get(empty).map(|d| d.tokens), Some(0));
        assert_eq!(ix.held.terms().count(), 0);
    }

    /// The score comes off the field the index named, and anything that is not
    /// a number leaves the document with the score the index gives everything.
    /// Every one of these was read back off a real server.
    #[test]
    fn a_score_field_is_read_as_a_whole_number_or_not_at_all() {
        let mut english = English::new();
        let mut ix = index();
        ix.definition.score_field = Some(b"sc".as_slice().into());
        for (raw, want) in [
            (b"2.5".as_slice(), 2.5),
            (b"abc", 1.0),
            (b"", 1.0),
            (b"-1", -1.0),
            (b"1e3", 1000.0),
            (b" 3 ", 1.0),
            (b"0", 0.0),
            (b"nan", 1.0),
            (b"7x", 1.0),
        ] {
            let id = ix
                .write(&mut english, b"q:1", &[(b"title", b"word"), (b"sc", raw)])
                .expect("it indexes");
            let got = ix.held.docs.get(id).expect("it is there").score;
            assert!(
                (got - want).abs() < f64::EPSILON,
                "score of {raw:?} was {got} and not {want}"
            );
        }
        let id = ix
            .write(
                &mut english,
                b"q:2",
                &[(b"title", b"word"), (b"sc", b"inf")],
            )
            .expect("it indexes");
        assert_eq!(ix.held.docs.get(id).map(|d| d.score), Some(f64::INFINITY));
    }

    /// The payload is carried on the document and read from the field the index
    /// named, and a document without that field carries nothing.
    #[test]
    fn the_payload_comes_off_the_field_the_index_named() {
        let mut english = English::new();
        let mut ix = index();
        ix.definition.payload_field = Some(b"pl".as_slice().into());
        let with = ix
            .write(&mut english, b"q:0", &[(b"title", b"word"), (b"pl", b"P0")])
            .expect("it indexes");
        let without = ix
            .write(&mut english, b"q:m", &[(b"title", b"word")])
            .expect("it indexes");
        let payload = |id| {
            ix.held
                .docs
                .get(id)
                .and_then(|doc| doc.payload.clone())
                .map(Vec::from)
        };
        assert_eq!(payload(with), Some(b"P0".to_vec()));
        assert_eq!(payload(without), None);
    }

    /// The numbers and the tag values go to their own field, and a range and a
    /// value both answer with the documents that held them.
    #[test]
    fn a_number_and_a_tag_go_to_the_field_that_declared_them() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(&mut english, b"d:1", &[(b"n", b"42"), (b"g", b"a,B")])
            .expect("it indexes");
        ix.write(&mut english, b"d:2", &[(b"n", b"7"), (b"g", b" b ")])
            .expect("it indexes");
        ix.held.settle();
        let nums = ix.held.numbers(b"n").expect("the field has numbers");
        assert_eq!(nums.range(Ends::shut(0.0, 10.0)), [2]);
        assert_eq!(nums.range(Ends::all()), [1, 2]);
        let tags = ix.held.values(b"g").expect("the field has values");
        assert_eq!(tags.get(b"b"), [1, 2]);
        assert_eq!(tags.get(b"a"), [1]);
        assert!(ix.held.numbers(b"nope").is_none());
        assert!(ix.held.values(b"nope").is_none());
    }

    /// A field the client said to leave out is left out of everything, and it
    /// takes no places either, so the field after it does not move.
    #[test]
    fn a_field_that_is_not_indexed_is_not_read() {
        let mut english = English::new();
        let mut ix = index();
        ix.schema[0].noindex = true;
        ix.schema[2].noindex = true;
        ix.write(
            &mut english,
            b"x:1",
            &[
                (b"title", b"skipped"),
                (b"body", b"kept"),
                (b"n", b"not a number"),
            ],
        )
        .expect("the number is not read either, so it cannot fail");
        assert!(ix.held.posts(b"skipped").is_none());
        assert_eq!(post(&ix, b"kept", 1).1, [1]);
        assert_eq!(post(&ix, b"kept", 1).0.fields, 0b1);
        assert!(ix.held.numbers(b"n").is_none());
    }

    /// A field that says not to stem does not, so the word goes in on its own.
    #[test]
    fn a_field_that_keeps_its_endings_has_no_stems() {
        let mut english = English::new();
        let mut ix = index();
        let Kind::Text(text) = &mut ix.schema[0].kind else {
            unreachable!("the first field is text")
        };
        text.nostem = true;
        ix.write(
            &mut english,
            b"s:1",
            &[(b"title", b"running"), (b"body", b"walking")],
        )
        .expect("it indexes");
        assert_eq!(
            ix.held.terms().collect::<Vec<_>>(),
            [&b"+walk"[..], &b"running"[..], &b"walking"[..]]
        );
    }

    /// Rewriting a key gives it a new number and the old number stays in the
    /// lists it was in, which is what makes a write cheap and a walk have to
    /// check.
    #[test]
    fn a_rewrite_leaves_the_number_it_had_behind() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(&mut english, b"r:1", &[(b"title", b"first")])
            .expect("it indexes");
        ix.write(&mut english, b"r:1", &[(b"title", b"second")])
            .expect("it indexes");
        assert_eq!(ix.held.docs.id(b"r:1"), Some(2));
        assert!(ix.held.docs.gone(1));
        assert_eq!(post(&ix, b"first", 1).0.id, 1);
        assert_eq!(ix.erase(b"r:1"), Some(2));
        assert_eq!(ix.erase(b"r:1"), None);
        assert!(ix.held.docs.is_empty());
        assert_eq!(post(&ix, b"second", 2).0.id, 2);
    }

    /// The facts the scoring asks the index for, which is the count of
    /// documents and the tokens they hold between them.
    #[test]
    fn the_index_can_say_what_the_scoring_needs() {
        let mut english = English::new();
        let mut ix = index();
        assert_eq!(ix.held.facts().docs, 0);
        ix.write(&mut english, b"b:1", &[(b"body", b"alpha beta")])
            .expect("it indexes");
        ix.write(&mut english, b"b:2", &[(b"body", b"alpha")])
            .expect("it indexes");
        let facts = ix.held.facts();
        assert_eq!((facts.docs, facts.tokens), (2, 3));
        assert!((facts.average() - 1.5).abs() < f64::EPSILON);

        // And the whole way through: a term, its document, its frequency and
        // the scorer that turns the three into a number.
        let doc = ix.held.docs.get(1).expect("it is there");
        let (post, _) = post(&ix, b"alpha", 1);
        let found = Found::Term(Term::new(post.freq, 1.0, 2));
        let mine = Scorer::Bm25.of(&facts, doc, &found, None);
        let same = Scorer::Bm25.of(&Facts::new(2, 3), doc, &found, None);
        assert!((mine - same).abs() < f64::EPSILON);
    }

    /// The whole way from two hashes to the number a real server answered
    /// `FT.SEARCH ... WITHSCORES` with, which is the only test here that says
    /// the lengths, the frequencies, the stem frequencies and the scoring are
    /// all right at once rather than one at a time.
    ///
    /// The query is the one word `dogs`, which is two terms in the index: the
    /// word and the stem it shares with `dog`, added together. 8.10.1 answered
    /// 0.4129047031380885 for the second document and 0.26302553225152586 for
    /// the first, which is worth two points and half a point respectively.
    #[test]
    fn a_search_for_one_word_scores_what_a_real_server_scored() {
        let mut english = English::new();
        let mut ix = index();
        ix.definition.score_field = Some(b"sc".as_slice().into());
        ix.write(
            &mut english,
            b"d:1",
            &[
                (b"title", b"Running dogs"),
                (b"body", b"the dogs are running fast"),
                (b"sc", b"0.5"),
            ],
        )
        .expect("it indexes");
        ix.write(
            &mut english,
            b"d:2",
            &[(b"title", b"Cats"), (b"body", b"cats and dogs")],
        )
        .expect("it indexes");

        let facts = ix.held.facts();
        assert_eq!((facts.docs, facts.tokens), (2, 14));
        let asked = |id| {
            let word = post(&ix, b"dogs", id).0.freq;
            let root = post(&ix, b"+dog", id).0.freq;
            Found::Any(vec![
                Found::Term(Term::new(word, 1.0, 2)),
                Found::Term(Term::new(root, 1.0, 2)),
            ])
        };
        let scored = |id, scorer: Scorer| {
            let doc = ix.held.docs.get(id).expect("it is there");
            scorer.of(&facts, doc, &asked(id), None)
        };
        assert_eq!(scored(1, Scorer::Bm25), 0.263_025_532_251_525_86);
        assert_eq!(scored(2, Scorer::Bm25), 0.412_904_703_138_088_5);
        // The union scorer that takes the larger of the two rather than adding
        // them, which is the frequency of the word itself in both documents.
        assert_eq!(scored(1, Scorer::DisMax), 4.0);
        assert_eq!(scored(2, Scorer::DisMax), 1.0);
    }

    /// Emptying an index leaves it as it started, which is what dropping one
    /// with its documents does.
    #[test]
    fn an_index_can_be_emptied() {
        let mut english = English::new();
        let mut ix = index();
        ix.write(&mut english, b"c:1", &[(b"title", b"word"), (b"g", b"a")])
            .expect("it indexes");
        ix.held.clear();
        assert!(ix.held.docs.is_empty());
        assert_eq!(ix.held.docs.last(), 0);
        assert_eq!(ix.held.terms().count(), 0);
        assert!(ix.held.values(b"g").is_none());
    }
}
