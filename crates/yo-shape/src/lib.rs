//! The shape tag: what makes "type safe" mean something across six languages
//! and across time (`15` section 3).
//!
//! A C ABI erases types and a file outlives every process that opens it, so
//! neither the host language's generics nor the ABI can tell a Python program
//! that the collection it just opened as a map of strings was written as a map
//! of integers by a Go program last March. Something in the file has to know.
//!
//! That something is a **canonical description**: a byte string that says what
//! the element type is, in a grammar no language owns, that every binding
//! computes identically. Its first 128 bits of BLAKE3 are the **tag**, and the
//! tag is what a collection stores and what an open compares.
//!
//! ```
//! use yo_shape::{Desc, Shape, Tag};
//!
//! struct Order;
//!
//! impl Shape for Order {
//!     fn describe(d: &mut Desc) {
//!         d.strukt("Order", &[("id", u64::describe), ("total", f64::describe)]);
//!     }
//! }
//!
//! // Same shape, same tag, on every machine and in every language.
//! assert_eq!(Tag::for_type::<Order>(), Desc::of::<Order>().tag());
//! assert_eq!(Desc::of::<Order>().tag().to_string().len(), 32);
//! ```
//!
//! # When the shapes differ
//!
//! A tag comparison alone would produce the worst error message a database can
//! give: something moved, and nothing about what. So the description is stored
//! next to the tag, and a mismatch renders both, underlines the difference,
//! names it in a sentence, and says whether it is additive or breaking.
//!
//! ```
//! use yo_shape::{Desc, Shape, check};
//!
//! let stored = Desc::of::<u32>();
//! let opening = Desc::of::<u64>();
//! let e = check("hits", &stored, &opening, None).unwrap_err();
//! assert!(e.message().contains("the type changed from u32 to u64"));
//! assert_eq!(e.detail(), Some("change=breaking"));
//! ```
//!
//! # What is not here
//!
//! The catalogue. Where a tag is stored, the list of prior tags that makes an
//! additive change open silently, and the creating SDK and version all belong
//! to the file (`07` section 5) and arrive with it. This crate computes,
//! compares and explains; it does not persist.
//!
//! The `#[derive(Yo)]` that writes [`Shape`] for you also comes later. Until
//! then a handful of lines per type is the price, and writing one by hand is
//! the best way to see that the description is not magic.

#![deny(missing_docs)]

pub mod desc;
pub mod diff;
pub mod parse;

pub use desc::{Bytes, Desc, Describe, Metric, Prim, Shape, Tag};
pub use diff::{Change, ChangeKind, Provenance, check, compare, mismatch};
pub use parse::{Type, parse};
