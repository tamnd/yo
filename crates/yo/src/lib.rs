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
//! # The same store the wire talks to
//!
//! [`Db::strings`] is the Redis string keyspace and [`Db::sets`] is the set
//! commands over the same one. A program that calls `incr` here runs the same
//! code an `INCR` off a socket runs (Y23), without the socket, the parser or the
//! reply, so the embedded API and a Redis client are two doors into one store
//! rather than two stores that agree for now.
//!
//! ```
//! let db = yo::open(yo::MEMORY)?;
//! let hits = db.counter("hits");
//!
//! hits.incr()?;
//! assert_eq!(db.strings().get("hits")?.as_deref(), Some(&b"1"[..]));
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! Where a Redis command works on one key for its whole life, there is a handle
//! that holds the key: [`Db::counter`] for a counter and [`Db::set`] for a set.
//! Those are sugar and they are worth having, because a name spelled once is a
//! name that cannot be misspelled at the third call site.
//!
//! ```
//! let db = yo::open(yo::MEMORY)?;
//! let online = db.set("online");
//!
//! online.add("alice")?;
//! online.add("bob")?;
//! assert_eq!(online.len()?, 2);
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

pub mod counter;
pub mod db;
pub mod keyspace;
pub mod map;
pub mod sets;
pub mod store;

pub use counter::Counter;
pub use db::{Db, MEMORY, open};
pub use keyspace::Strings;
pub use map::Map;
pub use sets::{Set, Sets};
pub use store::{Decode, Encode};
pub use yo_common::{Code, Error, Result};
pub use yo_shape::{Desc, Shape, Tag};
// The two views a borrowing read hands to its closure. They were reachable
// before this and not nameable, so a caller could take one and could not write
// down the type of what they had taken.
pub use yo_kv::{Member, Str};
