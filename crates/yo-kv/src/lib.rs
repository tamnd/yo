//! The Redis data structures, as plain Rust types with no protocol attached.
//!
//! This crate is the answer to a question the design keeps asking: where does
//! `INCR` actually live. It does not live in the codec, because an embedded
//! caller never speaks RESP and should not have to. It does not live in the
//! typed API either, because the wire has to reach the same code the typed API
//! reaches, or there are two implementations of `INCR` and one of them is wrong
//! (Y23). So it lives here, one method per command, taking and returning
//! ordinary Rust values, with `yo-resp` above it turning frames into calls and
//! the typed API above it turning generics into calls.
//!
//! What that buys is worth being clear about. An embedded program calls
//! [`Keyspace::incr`] and gets an `i64` or a [`yo_common::Error`]. It does not
//! serialise a command, it does not cross a socket, and it does not parse a
//! reply. That is the whole point of P1 and it is why the sub 150 nanosecond
//! number in `bench/00` is measured through here and not through a client.
//!
//! # What is here so far
//!
//! The string type, which is the first row of M2, and all 26 of its commands.
//!
//! The hash, in both of its representations and with field TTL, which is the
//! `HEXPIRE` family and the third answer `OBJECT ENCODING` can give.
//!
//! The bitmaps, which are the same string values seen a bit at a time. [`bits`]
//! is the kernels, popcount and scan and the eight ways of combining two
//! bitmaps and the packed integer fields, and [`bitmaps`] is where a key turns
//! into bytes and where Redis's edges are kept. There is no bitmap type,
//! because in Redis there is not one either: `SET k A` then `GETBIT k 1`
//! answers one.
//!
//! The HyperLogLogs, which are those same string values again with a documented
//! layout inside them. [`hll`] is the sketch, the hash and the two
//! representations and Ertl's estimator, and [`hlls`] is where a key turns into
//! one. It is byte for byte Redis's format, on purpose: a client can `GET` a
//! sketch out of a real server and `SET` it into us, and it has to count the
//! same, so the hash function and the opcodes and the promotion threshold are
//! copied rather than improved on.
//!
//! The places, which are sorted sets seen as a map. [`geo`] is the arithmetic,
//! the 52 bit interleave of longitude and latitude that a score is, the
//! haversine, the eleven character geohash string and the boxes a search covers,
//! and [`geos`] is where a key turns into a search. There is no geo type either,
//! for the same reason there is no bitmap one: `GEOADD` writes a member with a
//! score and `ZSCORE` reads that score straight back out, so a place is a sorted
//! set entry that somebody has agreed to read as a coordinate. What the search
//! costs is nine score ranges, one for the box the centre falls in and one for
//! each neighbour, at a precision picked so the nine cover the shape, and every
//! candidate is measured properly afterwards so nothing outside the circle or
//! the rectangle reaches the caller.
//!
//! The set, which is the first row of M3, in all seventeen of its commands:
//! `SADD`, `SREM`, `SCARD`, `SISMEMBER`, `SMISMEMBER`, `SMEMBERS`, `SPOP`,
//! `SRANDMEMBER`, `SSCAN`, `SMOVE`, `SINTER`, `SINTERCARD`, `SUNION`, `SDIFF`
//! and the three store forms. A [`Set`] is one of three representations and
//! moves between them on the same rules Redis uses, with `OBJECT ENCODING`
//! saying which one it is on and the five `CONFIG` thresholds moving the lines.
//! [`Keyspace`] is where a key gets to be
//! something other than a string: the record holds a number, the number points
//! into a [`Slab`], and every path that deletes a key or writes over one frees
//! what it was pointing at. That last part is why `WRONGTYPE` exists as of this
//! milestone. There was no way to trigger it while a string was all there was.
//!
//! Lists, hashes and the sorted set follow the same shape and land in M3 too.
//!
//! [`orderkey`] is the allocator that decides what sort key a list element gets,
//! which is the piece that stops `LINSERT` from renumbering everything behind
//! the element it inserted. It is the variable width scheme Y19 settles on, it
//! reproduces K14's eight inserts per byte, and like aki's own first slice of
//! this it ships proven and wired to nothing: the representation that stores a
//! list by key rather than by position is the partitioned band below, and the
//! wiring is a later piece of M4 than either of them.
//!
//! [`Parts`] is the partitioned band, which is what a collection becomes once it
//! is too large to be one element table. It is P tables with a member's
//! partition taken out of its hash, and in front of them the descriptor cache
//! `05` calls mandatory, which is what lets an operation know how the elements
//! are spread without adding up all P of them. What partitioning is really for is
//! the merge, the growth and the reclaim, none of which want to touch a million
//! element table at once. The cache turned out not to be the locality problem it
//! reads as, and `benches/parts.rs` has the measurement and the module doc has
//! the argument.
//!
//! The pieces of M3 that are here already are the ones every collection shares.
//! [`Elements`] is the element table a hash, a set and a sorted set are all
//! built out of, and [`Cursor`] is the scan cursor they all hand back to a
//! client. Neither is a set or a hash on its own, and both are where the
//! decisions that make those fast were made.
//!
//! [`Listpack`] is the band underneath both of them. A collection of a few dozen
//! elements does not want an index at all, so up to a hundred and twenty eight
//! it is one packed blob walked linearly, in Redis's own byte layout so that an
//! RDB export is a copy and `OBJECT ENCODING` can honestly say `listpack`.
//!
//! The four commands Redis added in 8.4 and 8.8 are the interesting ones and
//! they were checked against a real 8.8 rather than written from the
//! documentation. [`Keyspace::digest`] is the XXH3 of a value, and it is bit for
//! bit Redis's number, which is what makes [`Compare::DigestEqual`] worth
//! anything: a client comparing against a large value sends eight bytes instead
//! of the value. [`Keyspace::increx`] is not the rate limiter it looks like at
//! first, it is a counter with a bound, a saturation policy and four things it
//! can do to the deadline, and the rate limiter is one setting of it.
//!
//! # Divergences
//!
//! Four, all recorded in `divergences.toml` rather than left to be discovered.
//!
//! A string is capped at [`strings::STRING_MAX`] rather than Redis's 512 MiB,
//! because a value lives in one arena segment until the log backed band lands in
//! M5. Expiry is lazy only: a key past its deadline is dropped when something
//! touches it, and the active cycle that would reclaim a key nobody ever touches
//! again is maintenance slice work in M5. And `LCS` refuses a table over
//! [`LCS_MAX_CELLS`], where Redis has no explicit limit and fails on the
//! allocation instead, which on a server that has overcommitted is a kill rather
//! than an error. And the float counters count in `f64` where Redis counts in
//! the C `long double`, which is eighty bit on x86-64 and a hundred and twenty
//! eight bit on aarch64, so Redis does not agree with itself across machines
//! and we agree with ourselves everywhere.

#![deny(missing_docs)]

pub mod access;
pub mod array;
pub mod arrays;
pub mod bitmaps;
pub mod bits;
pub mod blob;
pub mod chunk;
pub mod clock;
pub mod cold;
pub mod cond;
pub mod counter;
pub mod db;
pub mod demote;
pub mod elem;
pub mod evict;
pub mod expiry;
pub mod foreign;
pub mod frozen;
/// The geohash arithmetic, which lives one crate down so that the search
/// index and the geo commands cover a circle in exactly the same way.
pub use yo_common::geo;
pub mod geos;
pub mod grow;
pub mod hash;
pub mod hashes;
pub mod hll;
pub mod hlls;
pub mod intset;
pub mod keys;
pub mod keyspace;
pub mod lcs;
pub mod list;
pub mod listpack;
pub mod lists;
pub mod orderkey;
pub mod parts;
pub mod rank;
pub mod rdb;
pub mod scan;
pub mod set;
pub mod setops;
pub mod sets;
pub mod slab;
pub mod snapshot;
pub mod sort;
pub mod stream;
pub mod streams;
pub mod strings;
#[cfg(test)]
mod tally;
pub mod tier;
pub mod ttl;
pub mod value;
pub mod walk;
pub mod zset;
pub mod zsetops;
pub mod zsets;

pub use array::{Array, Element as ArrayElement, INDEX_MAX, SLICE_SIZE};
pub use blob::{Blob, Span};
pub use clock::Clock;
pub use cond::Compare;
pub use counter::{Counted, IncrEx, IncrExpire, Num};
pub use db::{Db, Holds, MAX_STRIPES};
pub use elem::{Elements, Full, MAX_ROWS, NAME_MAX};
pub use foreign::Foreign;
pub use hash::{Hash, Limits as HashLimits};
pub use intset::{Intset, Walk};
pub use keys::{Moved, Record};
pub use keyspace::Keyspace;
pub use lcs::{Idx as LcsIdx, LCS_MAX_CELLS, Match as LcsMatch};
pub use list::{Limits as ListLimits, List};
pub use listpack::{Entry, Listpack, Malformed};
pub use lists::{End, Movem, Order};
pub use parts::{PART_MIN, PARTITION_AT, Parts};
pub use rank::Rank;
pub use scan::{Cursor, MAX_PARTS};
pub use set::{Limits as SetLimits, Member, Set};
pub use setops::Plan;
pub use slab::{MAX_SLOTS, Slab};
pub use snapshot::Snapshot;
pub use strings::{Exists, Expire, KEY_MAX, STRING_MAX, SetOptions, SetOutcome};
pub use ttl::{Applied, Ask, Cond, Deadlines, MAX_AT};
pub use value::{EMBSTR_MAX, Encoding, Kind, Str};
// Where a keyspace walk has got to, which is a different number from the
// [`Cursor`] a collection walk hands back and is named apart from it so that a
// caller holding both cannot pass one where the other was meant.
pub use yo_index::Cursor as KeyCursor;
// What arena compaction has cost, which `INFO` reports and nothing in here
// reads, so it is only here to save the reporter a dependency on the index.
pub use yo_index::Compaction;
pub use zset::{Bound as ZBound, Lex, Limits as ZsetLimits, Zset};
pub use zsetops::{Aggregate, Op as ZOp, Operand};
pub use zsets::{By, Gate, Move, Query, Window, ZAdd};
// Which end of a sorted set a pop works from. Renamed on the way out because
// `From` is in every Rust prelude and a second one under that name would be a
// trap for every file that imports this crate with a glob.
pub use zsets::From as ZEnd;
