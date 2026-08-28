//! The `.yo` file.
//!
//! One file, and everything in it. Two superblock slots at the front, then
//! regions, and a region is one log page's worth of contiguous segments. Above
//! this crate everything talks in log addresses; below it everything is
//! `pread` and `pwrite` and one sync call. Nothing in here knows what a key is.
//!
//! There are three pieces:
//!
//! - [`Yo`] owns the descriptor and the superblock, and does the root flip.
//! - [`LogFile`] is one shard's log pages, and is what `yo-record` writes
//!   through and reads back.
//! - [`io`] is positioned reads and writes and the strongest sync the platform
//!   has, which on macOS is not the one called `fsync`.
//! - [`RingWriter`] is the same writes done asynchronously, which is what
//!   [`LogFile::use_ring`] switches on and what the two hundred thousand commits
//!   a second in `06` section 3 needs.
//!
//! **No memory mapping.** A mapping would make reads free and make every write
//! fault, and it would make a torn page an unrecoverable `SIGBUS` in the middle
//! of a shard rather than an error return. Positioned reads also work the same
//! way on Windows, which is one of the three platforms CI runs. The cost is a
//! syscall per read of a page that is not resident, and pages that are not
//! resident are the cold path by construction.
//!
//! **The file describes itself.** Every region begins with a header saying
//! which shard owns it and what log address its payload starts at, so opening a
//! file is reading 32 bytes per 32 MiB and nothing else. There is no allocation
//! table that could disagree with the data, because there is no allocation
//! table.

pub mod io;

mod file;
mod log_file;
mod ring;

pub use file::{Checkpoint, CreateOptions, REGION_LEN, Yo, region_offset};
pub use log_file::LogFile;
pub use ring::RingWriter;
pub use yo_uring::{RingConfig, SqPoll};
