//! The document model: YOJB, the encoding, and the collection it is stored in
//! (`09` sections 2 and 4).
//!
//! A document database is a binary JSON encoding plus secondary indexes over
//! paths into it. [`Value`] and [`Builder`] are the encoding, which is JSONB in
//! spirit, which is Postgres and CockroachDB in spirit, with the differences
//! that matter for an embedded engine written down below. [`Docs`] is the
//! collection: documents by id, with the [`Keys`] table that turns every object
//! key into two bytes and a [`PathIndex`] per path that is worth looking
//! documents up by, for equality, for ranges, for the elements of an array or
//! for the words of a string.
//!
//! ```
//! use yo_doc::{Builder, Kind, Value};
//!
//! let mut b = Builder::new();
//! b.begin_object()?;
//! b.key(b"id")?;
//! b.int(41_920)?;
//! b.key(b"name")?;
//! b.text("a wrench")?;
//! b.key(b"price")?;
//! b.float(12.5)?;
//! b.end_object()?;
//! let doc = b.finish()?.to_vec();
//!
//! let v = Value::new(&doc).unwrap();
//! assert_eq!(v.kind(), Kind::Object);
//! assert_eq!(v.get(b"name").unwrap().as_text(), Some("a wrench"));
//! assert_eq!(v.path("$.price")?.unwrap().as_float(), Some(12.5));
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # The shape of a value
//!
//! Every value, at every level, begins with a four byte header: three bits of
//! kind, a bit that tells an object from an array, four flags, and a
//! twenty four bit count that is an element count for a container and a payload
//! length for a scalar. A scalar is the header and its bytes. A container is
//! the header, an entry table, and then the elements.
//!
//! See [`layout`] for the container layout and why each piece is where it is.
//!
//! # Three differences from Postgres JSONB
//!
//! **Keys are interned per collection.** A typed collection assigns every field
//! name it has seen a two byte id, and an object written into it stores ids
//! rather than bytes. Document collections repeat the same twenty field names
//! on every document, so this is worth roughly forty percent of a collection's
//! size, and it turns a member lookup from a comparison of bytes into a
//! comparison of integers. [`Keys`] is the table that hands out the ids and
//! [`Docs::put`] is what applies it.
//!
//! **A container is capped at 16.7 M elements**, because the count shares a
//! word with the kind and the flags. That is one word of overhead per value
//! rather than Postgres's per entry scheme with a separate container header.
//!
//! **Nothing inside a value is compressed.** Compression is a record level flag
//! (`06` section 2.1), so a path read never has to decompress a document to
//! reach one field of it. A document model that stores a compressed blob and
//! calls the fields indexed is a document model that decompresses on every
//! read.
//!
//! [`query`] is the other half of the path grammar. [`Value::path`] answers one
//! value and refuses `[*]` and `..` because it has nowhere to put a second
//! answer, and [`Path`] is what reads those: a descent, a wildcard, a slice and
//! a union, which is RFC 9535 without its filter selector. The `JSON.*` surface
//! is written against sets rather than single values, so it needs both.
//!
//! [`edit`](mod@edit) is the write side. A path answers a set of places and an edit says
//! what happens at each of them, which is a replacement, a removal, a key put
//! into an object or a run of an array spliced. A document is rebuilt rather
//! than patched, and everything the edit did not name is a memcpy through
//! [`Builder::embed`], so the cost follows the size of the document and not the
//! number of changes.
//!
//! [`text`] is JSON text in and out. The typed API never touches it, since a
//! struct is serialized straight into this encoding and read straight back out
//! of it, but `JSON.SET` arrives with text and `JSON.GET` has to hand text
//! back, so the whole `JSON.*` surface stands on [`Builder::json`] and
//! [`Value::to_json`]. The parser takes RFC 8259 and nothing else, for the
//! reason spelled out there: every convenience a JSON parser adds is a document
//! that loads here and is refused by a real Redis.
//!
//! # What is not here
//!
//! The typed `Docs<T>` surface with its derive, which is `15`.
//!
//! # What is here now that was not
//!
//! [`VectorIndex`] puts an embedding under a path in the same collection the
//! document is in, so a nearest neighbour search hands back documents and the
//! filter over their other indexed fields runs inside the scan. See
//! [`vector`] for why that is not a [`PathIndex`] and why the
//! filter has to be inside.

#![deny(missing_docs)]

mod build;
mod docs;
pub mod edit;
mod filter;
mod head;
mod index;
mod keys;
pub mod layout;
mod path;
pub mod query;
mod read;
pub mod text;
pub mod vector;

pub use build::Builder;
pub use docs::{Doc, DocElems, DocMembers, Docs};
pub use edit::{Edit, edit};
pub use head::{COUNT_MAX, DEPTH_MAX, Kind};
pub use index::{IndexKind, KEY_MAX, Key, PathIndex, Ranged, RangedRev};
pub use keys::{KEYS_MAX, Keys};
pub use path::{Step, Steps};
pub use query::Path;
pub use read::{Elems, Members, Value, key_order};
pub use text::{Format, from_json};
pub use vector::VectorIndex;
pub use yo_kv::Cursor;
