//! Time series storage, which is what yo answers the `TS.*` commands from.
//!
//! A [`Series`] is a list of chunks in timestamp order plus the settings the
//! commands hang off it: how far back to keep samples, how much room a chunk
//! gets, what to do about a repeated timestamp, the labels it can be found by,
//! and how close to the last sample a reading has to be before it is not worth
//! storing.
//!
//! Samples are squeezed with Gorilla by default, which for a series read at a
//! steady rate costs a couple of bits a sample rather than sixteen bytes. A
//! series whose values have nothing to do with each other can ask for the plain
//! layout instead and pay the sixteen bytes for a simpler walk.
//!
//! ```
//! use yo_series::{Sample, Series};
//!
//! let mut series = Series::new();
//! for i in 0..10 {
//!     series.add(Sample::new(1_700_000_000_000 + i * 1000, 21.5), None).unwrap();
//! }
//! assert_eq!(series.len(), 10);
//! assert_eq!(series.range(1_700_000_000_000, 1_700_000_002_000).count(), 3);
//! ```

mod bits;
mod chunk;
mod query;
mod sample;
mod series;

pub use chunk::{Chunk, Encoding, Samples};
pub use query::{Agg, Buckets, MAX_ROWS, Query, Rows, Stamp, Unread, bucket_start, group};
pub use sample::Sample;
pub use series::{DEFAULT_CHUNK_BYTES, Policy, Refused, Rule, Series};
