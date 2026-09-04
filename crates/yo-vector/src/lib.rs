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
//! [`Partitions`] is the index over those codes. A vector belongs to the
//! partition whose centroid it is nearest, a partition's members are a flat run
//! of codes, an insert is an append and a delete moves the last member into the
//! hole. A search ranks the centroids, scans the nearest few postings, and then
//! measures the best handful properly against the full precision vectors. It
//! splits, merges and reassigns in bounded steps as it goes, which is SPFresh's
//! LIRE, and it is why there is never a rebuild.
//!
//! The centroids are kept already rotated, which matters more than it sounds
//! like it should. Preparing a query is mostly the rotation and it happens once
//! per partition probed, so rotating the centroids once when they are built
//! turns tens of rotations a search into one.
//!
//! They are read in full on every search, which sounds like the obvious thing to
//! fix on a collection with thousands of them and is not. `src/rank.rs` is the
//! measurement: coding them the way their members are coded is three to nine
//! times slower than reading them, because reading them is already going at
//! memory speed and the estimator that would replace it is not.
//!
//! `src/probe.rs` is where that decision gets its context. It splits a query
//! into ranking the centroids, preparing the query against each partition
//! probed, and scanning the postings, and it is what says which of the three is
//! worth working on at a given dimension and collection size.
//!
//! [`Collection`] is the piece above that, and it is the one both doors reach.
//! [`Partitions`] deals in ids and knows nothing about the key a client wrote a
//! vector under, what metric the collection was opened with, or where the full
//! precision vectors live, because none of those are the index's business.
//! `db.vectors()` in a Rust program and `VADD` off a socket both need all three,
//! so they are answered once here rather than twice above.
//!
//! A filter runs inside the posting scan rather than after it. Every member
//! carries a tag word beside its code, and a scan that can reject a member
//! before it measures one can keep going into further partitions until it has
//! enough that pass. That widening is the whole point: a selective filter means
//! the nearest partitions may hold nothing the caller asked for, and a search
//! that does not go looking is a recall lottery. [`Signature`] packs arbitrary
//! attribute values into that one word, and it is allowed to say yes when it
//! should have said no but never the other way round, so the caller's own
//! predicate stays the authority. That predicate has a place to live too:
//! [`Filter::exact`] sees the member's id and runs only on members the tag let
//! through that are near enough to be ranked, which is what lets an expression
//! over a JSON string be the real answer without the scan ever reading one.
//!
//! [`muvera`] is late interaction retrieval on that same index. A ColBERT style
//! model gives a document one vector per token and scores a query against it
//! with Chamfer similarity, which normally means a second index over every
//! token of every document and a scoring pass on top of it. MUVERA maps a set
//! of token vectors to one fixed length vector whose dot product approximates
//! Chamfer, so it costs an encode at write time, the index that is already
//! here, and [`muvera::chamfer`] as the rerank. There is no second index.
//!
//! [`hnsw`] is the compatibility view. Clients pass `M`, `EF_CONSTRUCTION` and
//! `EF_RUNTIME` and expect them to do something, because against Redis and
//! valkey they do, and there is no graph here to point them at. So each one is
//! mapped onto whatever it was actually for: build effort becomes the posting
//! size, search beam becomes the probe and the rerank width, and `M` is the out
//! degree of a graph that does not exist, so it is echoed back and changes
//! nothing. A client that asked for HNSW and meant it can say so and be
//! refused rather than quietly served something else.
//!
//! [`image`] is how any of it survives a restart. A collection is written down
//! as the format's `10` section 2 says, one chain per partition under a
//! checkpoint, and read back without a single vector being requantised. The
//! vectors themselves are not in there, because they are already records of kind
//! 3 in the log, so a load takes the shape and the codes from the image and the
//! vectors from whatever the caller points it at.

#![deny(missing_docs)]

pub(crate) mod coarse;
pub mod collection;
pub(crate) mod dist;
pub mod hnsw;
pub mod image;
mod miss;
pub mod muvera;
mod narrow;
pub mod partition;
mod probe;
pub mod rabitq;
mod rank;
pub mod rotate;

pub use collection::{Collection, Match};
pub use image::{Restored, Stored};
pub use partition::{Any, Filter, Hit, Partitions, Signature, Tuning, Vectors, Work};
pub use rabitq::{Bits, Coded, Quantizer, Query};
pub use rotate::Rotation;
