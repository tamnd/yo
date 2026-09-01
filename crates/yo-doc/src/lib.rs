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
//! # What is not here
//!
//! Parsing JSON text. It arrives with the RESP surface, where it belongs: the
//! typed API never parses JSON, it serializes a struct straight into this
//! encoding, and text parsing is for `JSON.SET` and for bulk import.
//!
//! The vector index, which is `10`, and the typed `Docs<T>` surface with its
//! derive, which is `15`.

#![deny(missing_docs)]

mod build;
mod docs;
mod head;
mod index;
mod keys;
pub mod layout;
mod path;
mod read;

pub use build::Builder;
pub use docs::{Doc, DocElems, DocMembers, Docs};
pub use head::{COUNT_MAX, DEPTH_MAX, Kind};
pub use index::{IndexKind, KEY_MAX, Key, PathIndex, Ranged, RangedRev};
pub use keys::{KEYS_MAX, Keys};
pub use path::{Step, Steps};
pub use read::{Elems, Members, Value, key_order};
pub use yo_kv::Cursor;
