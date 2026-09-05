//! The search model: what an index is, what it reads, and where the server
//! keeps them (`09` section 5).
//!
//! `FT.*` is the one command family in this build that is not about a key. An
//! index is not stored under a name a client can `GET`, it does not show up in
//! `KEYS *`, and `TYPE` has nothing to say about it. It is a standing
//! instruction: follow every key with this prefix, read these paths out of it,
//! and let me ask questions about what you found. So this crate is the
//! instruction and the table of them, and the answering is above it.
//!
//! ```
//! use yo_search::{Definition, Field, Index, Kind, Registry, Text};
//!
//! let mut r = Registry::new();
//! let title = Field::new(b"title", Kind::Text(Text::default()));
//! r.create(Index::new(b"books", Definition::default(), vec![title]))?;
//! assert_eq!(r.get(b"books").map(|i| i.schema.len()), Some(1));
//! # Ok::<(), yo_search::Clash>(())
//! ```
//!
//! # What is here
//!
//! [`Field`] and [`Kind`] are one column of a schema. The six kinds are the six
//! a real server takes, and each one carries only the options that mean
//! something for it, which is why a weight lives on [`Text`] and a separator on
//! [`Tag`] rather than both living on every field and being ignored five times
//! out of six.
//!
//! [`Definition`] is which keys an index follows and how it reads them, and
//! [`Index`] is a definition with a schema under a name. [`Registry`] is every
//! index on the server plus the aliases pointing at them.
//!
//! [`query`] is the query language: the grammar a client writes, the tree it
//! parses into and the printout `FT.EXPLAIN` answers with. Both dialects are
//! there, because the one a client gets when it does not ask for one is still
//! the first and they parse the same bytes into different trees.
//!
//! [`token`] and [`words`] are the two halves of the tokenizer, the query side
//! and the document side, and [`posts`] is the inverted index they both feed:
//! every term in an index and, for each one, which documents have it, how often,
//! in which fields and where. [`nums`] is the same thing for a `NUMERIC` field,
//! a sorted run of values a range is cut out of. [`docs`] is the table the two
//! of them point into: which key a number was given to, what the client said it
//! was worth, and how long it turned out to be. [`tags`] is the third index, the
//! one a `TAG` field gets, where a value is matched whole rather than being
//! broken into words. [`geos`] is the fourth, and it is the numeric one again
//! with a point folded into the number, so a circle is a handful of ranges out
//! of it with the distance checked on whatever they turn up.
//!
//! [`score`] is what turns a walk over those lists into an ordered answer. All
//! nine scorers a real server takes are there, measured against one rather than
//! read off its manual, which is how the single precision `k1` in the default
//! one came to light.
//!
//! [`sorted`] is the other way of ordering an answer, by a field rather than by
//! a score. A field the schema calls `SORTABLE` has its value copied into the
//! document table as the sort will compare it, folded for text and parsed for a
//! number, so a `SORTBY` over it never has to go back to the key. A field it
//! does not is still sortable, just read off the key afterwards, and the same
//! comparison is used either way so both roads give the same answer.
//!
//! [`summary`] is what `SUMMARIZE` and `HIGHLIGHT` do to a value on its way
//! back out: cut it down to the parts the query matched and mark them. It runs
//! long after the walk did, off the value under the key rather than off the
//! index, which is why a field the schema does not know is cut but never
//! marked.
//!
//! [`held`] is the three of them under one index with the routine that fills
//! them, so a key and its fields go in one end and a document with a number, a
//! score, a length and a term in every list it belongs to comes out the other.
//!
//! [`follow`] is the fan out over that: which indexes a key belongs to, what
//! happens to each of them when it changes or goes, and the count of the keys
//! they would not read, which is the `Index Errors` block `FT.INFO` answers
//! with.
//!
//! [`walk`] is the other direction: a parsed tree goes in and every document
//! that answers it comes out, in number order and each with the shape of how it
//! answered, which is what a scorer turns into a number. [`expand`] is the part
//! of that which is not one term but many, a prefix, a suffix, an infix, a
//! pattern or a fuzzy word standing for every term in the dictionary it covers.
//!
//! # What is not here yet
//!
//! `FT.SEARCH` itself. A query can be parsed, walked and scored from Rust and
//! the answers agree with a real server's over sixty two queries on the same
//! corpus, but nothing on the wire asks for them yet, so a client still cannot
//! search.
//!
//! Two kinds of node are not walked. A phrase needs the places two terms were
//! found at and the rules for how far apart they may be. A vector query needs a
//! field the document reader does not read yet, so there is nothing in the
//! index for it to walk either way. Both answer nothing rather than answering
//! wrongly.
//!
//! `FILTER` is not parsed, so an index carrying one follows every key its
//! prefixes cover.
//!
//! # Why the registry is per server
//!
//! Every other collection in this build is per database and this one is not.
//! A real server keeps its indexes in the search module, the module has one
//! table, and an index created on database zero is listed by `FT._LIST` after
//! `SELECT 1`. [`Registry`] says the same, because the alternative is a
//! difference that shows up the first time somebody runs two databases.

#![deny(missing_docs)]

pub mod docs;
pub mod english;
pub mod expand;
pub mod expr;
pub mod field;
pub mod follow;
pub mod geos;
pub mod held;
pub mod index;
pub mod nums;
pub mod posts;
pub mod query;
pub mod reduce;
pub mod registry;
pub mod score;
pub mod sorted;
pub mod summary;
pub mod tags;
pub mod text;
pub mod token;
pub mod walk;
pub mod words;

pub use english::English;
pub use field::{Algo, Coords, Field, Kind, Tag, Text, Vector, Width};
pub use index::{Definition, Index, Options, Source};
pub use query::{Node, What};
pub use registry::{Clash, Registry};
