//! An independent reader for `.yo` files.
//!
//! This crate reads the format and shares no code with the thing that writes
//! it. Not the checksum, not the layout constants, not the record parser, not a
//! byte order helper. It has no dependencies at all, and that is the point
//! rather than a boast about being lightweight.
//!
//! # Why a second reader exists
//!
//! The engine's own tests write a file and read it back. That catches a great
//! deal and it cannot catch the one thing that matters most here: a
//! misunderstanding of the format that the writer and the reader share. If
//! `yo-format` puts a field at the wrong offset, the engine writes it there,
//! reads it back from there, and every round trip test passes. The file is
//! wrong and nothing in the project says so.
//!
//! So the M1 exit gate is not "the engine can read its own files". It is that
//! this crate, written from `06` and `07` rather than from the engine, agrees
//! with the engine about every file the fuzzer can produce. Two independent
//! transcriptions that agree are evidence. One transcription checked against
//! itself is not.
//!
//! That is also why the tests here are the one place a dependency on the engine
//! is allowed: the agreement is the deliverable, and it needs both sides
//! present to be measured.
//!
//! # What it does and does not do
//!
//! It opens a file, picks the live superblock, and walks the regions and the
//! records in them. It never writes, never maps, and never repairs. It does not
//! replay, does not build an index and does not know what a value means. A
//! record is a kind byte and two byte strings as far as this crate is
//! concerned.
//!
//! It is also deliberately more forgiving than the engine in one direction and
//! less in another. More: a damaged region does not stop the walk, because
//! somebody running this is usually running it because something is damaged and
//! hiding the rest of the file from them helps nobody. Less: it refuses a
//! record whose checksum flag is clear rather than reading it, for the reason
//! written out in [`format::parse_record`].
//!
//! ```no_run
//! use std::path::Path;
//! use yo_reader::Reader;
//!
//! let r = Reader::open(Path::new("some.yo"))?;
//! println!("{} shards, sequence {}", r.superblock().shard_count, r.superblock().seq);
//! for region in r.regions() {
//!     let records = r.records(region)?;
//!     println!("region {} holds {} records", region.index, records.len());
//! }
//! # Ok::<(), yo_reader::Error>(())
//! ```

pub mod crc;
pub mod error;
pub mod format;
pub mod io;

mod reader;

pub use error::{Error, Result};
pub use format::{CheckpointEntry, PageHeader, Record, Superblock};
pub use reader::{Reader, Region, SlotStatus};
