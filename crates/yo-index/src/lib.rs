//! The index plane: buckets, tag prefiltered probing, and dashtable style
//! growth.
//!
//! This crate is the part of the record plane that answers "where is this key".
//! It stores tags and addresses, never keys and never values, and it reaches
//! the key bytes through the [`Keys`] trait so that the record format can
//! change under it without the index changing at all.
//!
//! Three things carry the performance claim, and all three are in `05`:
//!
//! 1. A bucket is 64 bytes, which is one cache line, so a probe is one load.
//! 2. Seven one byte tags are compared in a single 64 bit SWAR operation, so a
//!    miss costs no key comparison at all and a hit costs one.
//! 3. Growth splits one segment at a time, so there is no rehash pause.
//!
//! ```
//! use yo_index::{Index, Keys};
//! use yo_common::Addr;
//!
//! // A toy record plane: the address is an offset into one flat buffer of
//! // length prefixed keys. M1 replaces this with the real record header.
//! struct Toy(Vec<Vec<u8>>);
//! impl Keys for Toy {
//!     fn hash_at(&self, addr: Addr) -> u64 {
//!         yo_common::wyhash(&self.0[addr.offset() as usize], 0)
//!     }
//!     fn eq_at(&self, addr: Addr, key: &[u8]) -> bool {
//!         self.0[addr.offset() as usize] == key
//!     }
//! }
//!
//! let mut recs = Toy(vec![b"greeting".to_vec()]);
//! let mut ix = Index::new();
//! let h = yo_common::wyhash(b"greeting", 0);
//! ix.insert(h, b"greeting", Addr::new(yo_common::Space::Arena, 0), &recs);
//! assert!(ix.contains(h, b"greeting", &recs));
//! ```

#![deny(missing_docs)]

mod bucket;
mod index;
mod map;
mod scan;
mod tagged;

pub use bucket::{Bucket, EMPTY, SLOTS, SlotMask};
pub use index::{Index, Keys, MAX_CHAIN, SEGMENT_BUCKETS};
pub use map::{Compaction, RawMap};
pub use scan::Cursor;
