//! Crash injection. The last piece of `06`, and the one the milestone is gated
//! on: a hundred thousand injected faults with zero silent corruptions.
//!
//! # What it does
//!
//! Writes a real log through a real [`Log`](yo_record::Log), keeping a ledger
//! of every record and the address it went to. Kills the store at a point the
//! seed chooses, in a way the seed chooses. Replays what is left and compares
//! it against the ledger, record by record.
//!
//! There is no sampling and no summary statistic. The oracle knows the exact
//! bytes that went in, so it can say exactly which record came back wrong, and
//! a failure prints the seed that reproduces it in a millisecond.
//!
//! # The two rules
//!
//! **A crash must not lose an acknowledged commit.** Whatever the caller was
//! told was durable is still there afterwards.
//!
//! **Nothing may come back wrong.** Whatever replay hands over is a record that
//! was really written, at the address it claims, with the bytes it was written
//! with. This is the one that holds under every fault, media rot included,
//! because losing data loudly is survivable and handing back data that was
//! never written is not.
//!
//! # Why the faults are what they are
//!
//! A crash is not "the last write did not happen". Writes tear at a sector
//! boundary, land out of order, and go missing individually. Each of those is
//! its own fault in [`fault`], because each has broken a storage engine that
//! only modelled the easy version. See that module for the argument.
//!
//! # What this does not cover
//!
//! The record plane, not the file layer. Faults are injected at the
//! [`PageSink`](yo_record::sink::PageSink) seam, which is where pages leave
//! memory, so this exercises page headers, record framing, group commit and
//! replay. The superblock flip is a different mechanism with a different
//! argument, it is dual slotted and ordered by two syncs, and it is covered by
//! its own tests in `yo-file` and by `yodb check`. A run that reports zero
//! violations here is a statement about `06` and not about the whole file.
//!
//! # Example
//!
//! ```
//! use yo_crash::{Shape, trial};
//!
//! let out = trial::run(12345, Shape::default())?;
//! assert!(out.passed(), "{:?}", out.violations);
//! # Ok::<(), yo_common::Error>(())
//! ```

pub mod fault;
pub mod reader;
pub mod rng;
pub mod sink;
pub mod trial;

pub use fault::Fault;
pub use rng::Rng;
pub use sink::{CrashSink, Image};
pub use trial::{Outcome, Shape, Violation};
