//! The hybrid log. `06` in the specification.
//!
//! One log per shard. Writes append to the tail, reads take an address and get
//! bytes back, and nothing in here knows what a key means or what a value is
//! for. That separation is the point: this crate is where durability and
//! ordering live, and every data structure above it gets both for free.
//!
//! Three things this crate is responsible for, and they are the three things
//! that are hard to bolt on afterwards.
//!
//! **Ordering.** A record's length is stored last, after a release fence, and
//! the four bytes past the tail are always zero. A reader arriving at any
//! moment, including a recovery pass reading a page off a store after a machine
//! lost power mid write, sees either a whole record or a zero. There is no
//! third case, and that is why replay does not need a write ahead log in front
//! of the log.
//!
//! **Group commit.** One `fsync` answers every commit in a page. That is the
//! difference between a few hundred durable commits a second and the two
//! hundred thousand this milestone is gated on, and it is why
//! [`Durability::Group`] is the default rather than an option somebody has to
//! find.
//!
//! **Lookup based compaction.** Garbage is collected by asking the index
//! whether a record is still the one it points at, one probe per record, rather
//! than by merging sorted runs. See [`compact`].
//!
//! # What this crate does not do
//!
//! It does not open files. Pages go to a [`PageSink`], and the file layer is
//! what knows about descriptors and io_uring. It does not chunk large values;
//! a record larger than a page is refused with [`Code::Full`](yo_common::Code)
//! and building the chain is `05`'s job. It does not hold an index; compaction
//! asks for one through [`compact::Index`].
//!
//! # Example
//!
//! ```
//! use yo_record::{CommitAction, Durability, Log, LogConfig};
//! use yo_record::sink::MemorySink;
//! use yo_format::{RecordHeader, RecordKind};
//!
//! # fn main() -> yo_common::Result<()> {
//! let cfg = LogConfig { page_len: 8192, durability: Durability::Group, ..LogConfig::default() };
//! let mut log = Log::new(cfg, MemorySink::new())?;
//!
//! let put = log.append(&RecordHeader::new(RecordKind::String), b"user:1", b"ada")?;
//! assert_eq!(log.read(put.addr)?.value, b"ada");
//!
//! // In group mode the caller does not reply until the commit is durable, and
//! // the maintenance slice is what gets it there when a page is not full yet.
//! if let CommitAction::WaitFor(at) = put.action {
//!     log.commit_pending()?;
//!     assert!(log.durable_upto() >= at);
//! }
//! # Ok(())
//! # }
//! ```

pub mod compact;
pub mod log;
pub mod replay;
pub mod sink;

pub use compact::{COMPACT_TRIGGER, CompactStats, Index};
pub use log::{
    Append, CommitAction, DEFAULT_MUTABLE_FRACTION, DEFAULT_RESIDENT_PAGES, Durability,
    FLUSH_BLOCK, Log, LogConfig, Region,
};
pub use replay::{ReplayReport, replay};
pub use sink::{MemorySink, NullSink, PageSink, PageSource, PageWrite};
