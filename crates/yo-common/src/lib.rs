//! Shared vocabulary for the whole engine: addresses, ids, the error model,
//! and the two hash families.
//!
//! Everything here is used on the hot path, so the rules are strict. No
//! allocation outside the error path, no dependencies, and every constant that
//! another crate assumes is asserted in a test here rather than assumed twice.

#![deny(missing_docs)]

pub mod addr;
pub mod crc;
pub mod error;
pub mod wyhash;

pub use addr::{ADDR_BITS, Addr, MAX_OFFSET, OFFSET_BITS, SPACE_BITS, ShardId, Space};
pub use crc::{SLOT_COUNT, crc16, crc32c, hash_tag, slot_of};
pub use error::{Code, Error, Result};
pub use wyhash::{hash_key, tag_of, wyhash};

/// The size of an index bucket, and the cache line the engine is built around.
///
/// Everything that claims to be one cache line asserts against this rather than
/// writing 64 twice.
pub const CACHE_LINE: usize = 64;

/// The largest batch the shard loop drains in one pass (`04` section 3).
///
/// This is also the C ABI's `BATCH_MAX` and the size of every `_many` form, so
/// it is here rather than in the shard crate.
pub const BATCH_MAX: usize = 64;
