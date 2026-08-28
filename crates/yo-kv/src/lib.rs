//! The Redis data structures, as plain Rust types with no protocol attached.
//!
//! This crate is the answer to a question the design keeps asking: where does
//! `INCR` actually live. It does not live in the codec, because an embedded
//! caller never speaks RESP and should not have to. It does not live in the
//! typed API either, because the wire has to reach the same code the typed API
//! reaches, or there are two implementations of `INCR` and one of them is wrong
//! (Y23). So it lives here, one method per command, taking and returning
//! ordinary Rust values, with `yo-resp` above it turning frames into calls and
//! the typed API above it turning generics into calls.
//!
//! What that buys is worth being clear about. An embedded program calls
//! [`Strings::incr`] and gets an `i64` or a [`yo_common::Error`]. It does not
//! serialise a command, it does not cross a socket, and it does not parse a
//! reply. That is the whole point of P1 and it is why the sub 150 nanosecond
//! number in `bench/00` is measured through here and not through a client.
//!
//! # What is here so far
//!
//! The string type, which is the first row of M2. Lists, sets, hashes and the
//! sorted set follow the same shape and land in M3.
//!
//! # Divergences
//!
//! Two, both recorded in `divergences.toml` rather than left to be discovered.
//! A string is capped at [`strings::STRING_MAX`] rather than Redis's 512 MiB,
//! because a value lives in one arena segment until the log backed band lands in
//! M5. And expiry is lazy only: a key past its deadline is dropped when
//! something touches it, and the active cycle that would reclaim a key nobody
//! ever touches again is maintenance slice work in M5.

#![deny(missing_docs)]

pub mod clock;
pub mod strings;
pub mod value;

pub use clock::Clock;
pub use strings::{Exists, Expire, KEY_MAX, STRING_MAX, SetOptions, SetOutcome, Strings};
pub use value::{EMBSTR_MAX, Encoding, Str};
