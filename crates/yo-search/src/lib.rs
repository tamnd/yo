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
//! # What is not here yet
//!
//! The inverted index and the scoring. This is the shape an index has, the
//! table it lives in and the queries that can be asked of it, which is what
//! `FT.CREATE`, `FT.ALTER`, `FT.INFO`, `FT.DROPINDEX`, `FT.EXPLAIN` and the
//! alias family need and all that they need. `FT.SEARCH` needs postings under
//! it and that is the next piece.
//!
//! # Why the registry is per server
//!
//! Every other collection in this build is per database and this one is not.
//! A real server keeps its indexes in the search module, the module has one
//! table, and an index created on database zero is listed by `FT._LIST` after
//! `SELECT 1`. [`Registry`] says the same, because the alternative is a
//! difference that shows up the first time somebody runs two databases.

#![deny(missing_docs)]

pub mod english;
pub mod field;
pub mod index;
pub mod query;
pub mod registry;
pub mod text;
pub mod token;

pub use english::English;
pub use field::{Algo, Coords, Field, Kind, Tag, Text, Vector, Width};
pub use index::{Definition, Index, Options, Source};
pub use query::{Node, What};
pub use registry::{Clash, Registry};
