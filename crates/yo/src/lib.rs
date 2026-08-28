//! The embedded API: one file, typed handles, and no query language
//! (`15` sections 1 and 2).
//!
//! Two lines get you a database, and there is no third line. No server to
//! start, no connection string, no schema migration to run first, and nothing
//! to parse at runtime that the compiler could have checked instead.
//!
//! ```
//! let db = yo::open(yo::MEMORY)?;
//! let hits = db.map::<String, u64>("hits")?;
//!
//! hits.set("home", &1)?;
//! assert_eq!(hits.get("home")?, Some(1));
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! # Why there is no query language
//!
//! A query language is a second language inside the first one, and it costs
//! what a second language costs: strings the compiler cannot check, a parser
//! and a planner on the hot path, types that are yours on one side of the
//! quote and the database's on the other, and errors that arrive at runtime in
//! production rather than at build time on a laptop. A `Map<String, u64>` is
//! the same idea with none of that. Your editor completes it, your compiler
//! checks it, and a lookup is a function call.
//!
//! What replaces the query language for the parts a map cannot do is more
//! handles rather than more syntax. `Doc`, `Vectors`, `Graph` and the rest of
//! the Redis shapes all arrive as types in this crate, and each of them is
//! read the way a collection in your own program is read.
//!
//! # The type is the schema
//!
//! The type parameters on a handle are not a convenience the compiler erases.
//! They are written into the collection when it is created, as a description
//! that six languages compute identically (`15` section 3), and an open with a
//! different type is refused with a message that says which field moved and
//! whether the change is additive or breaking.
//!
//! ```
//! let db = yo::open(yo::MEMORY)?;
//! let _hits = db.map::<String, u64>("hits")?;
//!
//! let e = db.map::<String, String>("hits").unwrap_err();
//! assert_eq!(e.code(), yo::Code::ShapeMismatch);
//! assert!(e.message().contains("the type changed from u64 to str"));
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! # Zero copy is available, never mandatory
//!
//! [`Map::get`] hands back an owned value because that is what most code
//! wants. [`Map::with`] hands the bytes over where they lie, which allocates
//! nothing and is where the point read budget in `bench/00` is spent. Same
//! collection, same key, and the choice is made per call rather than per
//! database (Y29).
//!
//! # What is not here yet
//!
//! A file. This build holds a database in memory, and a path that is not
//! [`MEMORY`] says so rather than pretending. The `.yo` format arrives in M5
//! and nothing on this page changes when it does, which is the reason the
//! front door is being built before the room behind it.
//!
//! Threads. The database runs in inline mode (`15` section 7), where the
//! calling thread is the shard and a point read is a call rather than a
//! message. The owned and served modes put this same API over `yo-shard`'s
//! runtime and arrive with it.
//!
//! `#[derive(Yo)]`. Until it lands a collection holds the primitives, strings
//! and byte strings, which is enough to measure and enough to use.

#![deny(missing_docs)]

pub mod db;
pub mod map;
pub mod store;

pub use db::{Db, MEMORY, open};
pub use map::Map;
pub use store::{Decode, Encode};
pub use yo_common::{Code, Error, Result};
pub use yo_shape::{Desc, Shape, Tag};
