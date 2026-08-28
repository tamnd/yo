//! The shard loop.
//!
//! `04` section 2 is nine lines of pseudocode and this crate is those nine
//! lines with the bookkeeping filled in. Six stages, in one order, forever:
//!
//! 1. hand the submission queue to the kernel, which is one syscall or none,
//! 2. drain up to [`BATCH_MAX`] commands out of the intake lanes,
//! 3. enter the epoch,
//! 4. walk the batch once for the prefetch and once for the work,
//! 5. leave the epoch, then flush one reply buffer per connection touched,
//! 6. pick up completions and spend a bounded slice on maintenance.
//!
//! There is no executor here, no future, no waker, no `.await` and no work
//! stealing. The loop is the scheduler. That is not a stylistic preference: the
//! VLDB 2026 io_uring ladder is 16.5 thousand transactions a second when every
//! submission is waited on and 546.5 thousand when the shard keeps going and
//! picks completions up later, and an async runtime in the middle is how a
//! system gives that back.
//!
//! ## What this crate is and is not
//!
//! It is the loop, the batch, the two walks, the epoch discipline and the
//! maintenance budget. It is not the commands, the connections or the parser.
//! Those arrive through [`Engine`], which is the seam a shard is plugged in
//! through, and the loop does not know what a command is beyond having a key
//! hash or not having one.
//!
//! ```
//! use yo_reactor::{Engine, Flow, Reactor};
//!
//! struct Counting(u64);
//!
//! impl Engine for Counting {
//!     type Work = u64;
//!     fn key_hash(&self, w: &u64) -> Option<u64> { Some(*w) }
//!     fn prefetch(&self, _w: &u64, _hash: u64) {}
//!     fn run(&mut self, w: u64, _hash: Option<u64>) -> Flow { self.0 += w; Flow::Next }
//!     fn flush(&mut self) {}
//! }
//!
//! let mut r = Reactor::inline(Counting(0));
//! for i in 1..=10 { r.execute(i); }
//! assert_eq!(r.engine().0, 55);
//! ```

#![deny(missing_docs)]

mod budget;
mod reactor;

pub use budget::{Budget, MAINTENANCE_UNITS};
pub use reactor::{BATCH_MAX, Engine, Flow, Reactor, Turn};
