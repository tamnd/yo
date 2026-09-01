//! Where the pieces of a container sit, worked out in one place so that the
//! builder and the reader cannot drift apart.
//!
//! ```text
//! +--------+--------------+---------------+------------+--------------+
//! | header | key entries  | value entries | key region | value region |
//! | 4      | objects only | 8 per element |            |              |
//! +--------+--------------+---------------+------------+--------------+
//! ```
//!
//! A key entry is four bytes, an offset into the container, and a key's length
//! is the difference between its offset and the next one. An interned object's
//! key entries are two byte ids instead, padded up so that the value entries
//! stay on a four byte stride.
//!
//! A value entry is eight bytes: a copy of the element's own header, so that a
//! scan over the entries never touches the value region, and the offset the
//! element starts at. Offsets are relative to the container's header, so a
//! container can be copied anywhere without rewriting it.
//!
//! Values are stored in entry order and tile the value region, which is what
//! makes the last entry enough to work out the whole container's length.

use crate::head::{ARRAY, INTERNED};

/// How many bytes an object's key entries take, padding included.
pub(crate) fn keys_area(head: u32, count: usize) -> usize {
    if head & ARRAY != 0 {
        return 0;
    }
    if head & INTERNED != 0 {
        (count * 2).next_multiple_of(4)
    } else {
        count * 4
    }
}
