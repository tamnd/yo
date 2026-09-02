//! The vector model: RaBitQ codes under partitions that update in place
//! (`10`).
//!
//! A vector index is two decisions, and as of 2026 only one of them is still
//! open. The quantiser is settled: RaBitQ won, and VectorChord, Lucene, which
//! calls it BBQ, CockroachDB, turbopuffer and Zvec all landed on it against five
//! different index structures inside eighteen months. [`Quantizer`] is that,
//! and it is what makes a ten million vector collection a 1 GB index rather
//! than a 30 GB one.
//!
//! The index is decided here by the update path. A graph index tombstones a
//! delete, degrades as the tombstones pile up, and gets better again only when
//! it is rebuilt. Redis shipped HNSW vector sets in 8.0 in May 2025 and they
//! were still beta three minor releases later, which is the vendor's own
//! evidence about how hard that path is. So this is partitions with the
//! centroids resident and the codes in flat postings, which SPFresh showed can
//! be split, merged and reassigned in place, and there is never a rebuild.
//!
//! # What is here so far
//!
//! The quantiser and the rotation it needs. [`Quantizer::encode`] writes the
//! searchable form of a vector against the centroid of the partition it belongs
//! to, and [`Quantizer::query`] prepares a query once so that measuring it
//! against the codes in a partition is a scan over contiguous bytes.
//!
//! ```
//! use yo_vector::{Bits, Quantizer};
//!
//! let q = Quantizer::new(128, Bits::One, 7);
//! assert_eq!(q.code_bytes(), 16);
//! ```
//!
//! A code is stored as bit planes rather than with each coordinate's bits next
//! to each other, and the query is quantised and transposed the same way, so
//! measuring one against the other is ANDs and popcounts rather than a float
//! multiply per dimension. That is 20 nanoseconds a vector at 768 dimensions
//! against a whole search budget of a millisecond, and 35 times what the same
//! estimator costs with the query left in floats. `benches/rabitq.rs` runs both
//! so the ratio is measured rather than remembered.
//!
//! The partitions themselves, the LIRE update protocol, filtered search, MUVERA
//! and the HNSW compatibility view are the rest of M6 and are not here yet.
//! Preparing a query is 10 microseconds and it happens once per partition
//! probed, which the partition store fixes by rotating its centroids once when
//! it builds them rather than rotating the query again for each of them.

#![deny(missing_docs)]

pub mod rabitq;
pub mod rotate;

pub use rabitq::{Bits, Coded, Quantizer, Query};
pub use rotate::Rotation;
