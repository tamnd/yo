//! Shard per core runtime.
//!
//! One thread per core, each owning its slice of the keyspace outright. There
//! is no lock on the data path and no atomic on it either, because there is
//! only ever one thread that can reach a given piece of state. `05` section 1
//! makes the argument; this crate is the argument compiled.
//!
//! Three pieces:
//!
//! - [`ShardLocal`], which is `!Send` and `!Sync`, so shard state cannot escape
//!   the thread that owns it. Getting this wrong is a compile error rather than
//!   a rare corruption.
//! - [`spsc`], the single producer single consumer lane that work crosses on.
//!   One lane per submitter per shard, so no queue ever has two writers.
//! - [`Epochs`], one counter per shard, bumped once per batch, which is how
//!   memory gets reclaimed without a read barrier.
//!
//! ```
//! use yo_shard::Runtime;
//!
//! let rt: Runtime<u64> = yo_shard::builder().shards(4).pin(false).build(|_| 0);
//! let sub = rt.submitter();
//! for i in 0..100u64 {
//!     let shard = rt.shard_of(yo_common::wyhash(&i.to_le_bytes(), 0));
//!     sub.send(shard, move |ctx| *ctx.state += i);
//! }
//! let total: u64 = (0..rt.shards()).map(|s| sub.call(s, |ctx| *ctx.state)).sum();
//! assert_eq!(total, (0..100u64).sum::<u64>());
//! ```

#![deny(missing_docs)]

pub mod epoch;
mod shard;
pub mod spsc;
mod sync;

pub use epoch::{Epochs, Retired};
pub use shard::{Builder, LANE_CAPACITY, Rejected, Runtime, ShardCtx, ShardLocal, Submitter};

/// Start configuring a runtime.
///
/// Prefer this over `Runtime::builder`, which cannot infer the shard state type
/// until the `build` call and so needs a turbofish on the way in.
pub fn builder() -> Builder {
    Builder::new()
}
