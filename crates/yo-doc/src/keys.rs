//! The per collection key table, which is what makes an interned object's keys
//! two bytes each (`09` section 4).
//!
//! A document collection stores the same field names on every document it
//! holds. A collection of a million orders with twenty fields each holds twenty
//! million copies of twenty strings, and that is most of what the collection
//! costs before anything useful is in it. Interning turns the twenty million
//! copies into twenty strings and twenty million two byte ids, and it turns a
//! member lookup from a comparison of bytes into a comparison of integers.
//!
//! The table is an [`Elements`] with nothing stored against a name, which is
//! the same structure a set is, and an id is a row index in it. That works
//! because a name is never taken out: see [`Keys`] for why not.
//!
//! ```
//! use yo_doc::Keys;
//!
//! let mut keys = Keys::new();
//! let id = keys.intern(b"customer").expect("room");
//! assert_eq!(keys.id(b"customer"), Some(id));
//! assert_eq!(keys.name(id), Some(&b"customer"[..]));
//! assert_eq!(keys.intern(b"customer"), Some(id), "a name gets one id, ever");
//! ```

use yo_kv::Elements;

/// How many names one collection can intern.
///
/// An id is two bytes because that is what an interned object's key entry is,
/// so the table stops at 65536 names. A collection that reaches it is not a
/// collection of documents any more, it is a collection of a schema per row,
/// and [`Keys::intern`] answers `None` so the writer can store that document
/// with its keys as bytes rather than fail the write.
pub const KEYS_MAX: usize = 1 << 16;

/// The names one collection has interned, and the ids it gave them.
///
/// # Why nothing is ever removed
///
/// An id is the row index the name sits at, which costs the table nothing at
/// all: no second array from id to row, no free list, no generation counter. It
/// holds only while the rows do not move, and [`Elements`] moves its last row
/// into the hole when something is taken out, so a removal here would silently
/// repoint every document that used the moved name.
///
/// Never removing is the right answer rather than a limitation being tolerated.
/// A field name that no document uses any more costs its bytes once and two
/// bytes of nothing in the row array, and the alternative is either an
/// indirection on every lookup forever or a scan of the whole collection to
/// find out whether a name is still wanted. The table is capped at
/// [`KEYS_MAX`] names, so the worst case is bounded and small.
#[derive(Debug, Clone, Default)]
pub struct Keys {
    names: Elements<()>,
}

impl Keys {
    /// An empty table that has not allocated anything yet.
    #[must_use]
    pub fn new() -> Keys {
        Keys {
            names: Elements::new(),
        }
    }

    /// An empty table with room for `n` names already taken.
    #[must_use]
    pub fn with_capacity(n: usize) -> Keys {
        Keys {
            names: Elements::with_capacity(n.min(KEYS_MAX)),
        }
    }

    /// The id of `name`, giving it one if it does not have one yet.
    ///
    /// `None` means the table is full or the name is longer than a name may be,
    /// and in both cases the caller writes the document with its keys as bytes
    /// instead. That is always safe, because the interned flag is per container
    /// and not per collection, so a collection can hold both kinds at once and
    /// everything already written stays readable.
    pub fn intern(&mut self, name: &[u8]) -> Option<u16> {
        if let Some(id) = self.id(name) {
            return Some(id);
        }
        if self.names.len() >= KEYS_MAX {
            return None;
        }
        let id = u16::try_from(self.names.len()).expect("the length is under KEYS_MAX");
        self.names.insert(name, ()).ok()?;
        Some(id)
    }

    /// The id `name` already has, without giving it one.
    ///
    /// This is the read path: a lookup by name against an interned document
    /// resolves the name here once and then searches the document by id.
    #[must_use]
    pub fn id(&self, name: &[u8]) -> Option<u16> {
        let at = self.names.index_of(name)?;
        u16::try_from(at).ok()
    }

    /// The name behind an id.
    #[must_use]
    pub fn name(&self, id: u16) -> Option<&[u8]> {
        self.names.at(usize::from(id)).map(|(name, ())| name)
    }

    /// How many names are interned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether nothing has been interned yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Whether the next new name would be refused.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.names.len() >= KEYS_MAX
    }

    /// Every name and its id, in id order.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], u16)> {
        self.names
            .iter()
            .enumerate()
            .map(|(at, (name, ()))| (name, at as u16))
    }

    /// What the table costs.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.names.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_keeps_the_id_it_was_given() {
        let mut keys = Keys::new();
        let a = keys.intern(b"alpha").expect("room");
        let b = keys.intern(b"beta").expect("room");
        assert_ne!(a, b);
        assert_eq!(keys.intern(b"alpha"), Some(a));
        assert_eq!(keys.intern(b"beta"), Some(b));
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn ids_come_out_in_the_order_they_were_handed_out() {
        let mut keys = Keys::new();
        for i in 0..64u16 {
            let name = format!("field{i}");
            assert_eq!(keys.intern(name.as_bytes()), Some(i));
        }
        for (i, (name, id)) in keys.iter().enumerate() {
            assert_eq!(id, i as u16);
            assert_eq!(name, format!("field{i}").as_bytes());
            assert_eq!(keys.name(id), Some(name));
        }
    }

    #[test]
    fn a_name_nobody_interned_has_no_id() {
        let mut keys = Keys::new();
        keys.intern(b"here").expect("room");
        assert_eq!(keys.id(b"not here"), None);
        assert_eq!(keys.name(1), None);
        assert_eq!(keys.len(), 1, "asking did not add it");
    }

    #[test]
    fn a_full_table_refuses_rather_than_failing_the_write() {
        let mut keys = Keys::with_capacity(KEYS_MAX);
        for i in 0..KEYS_MAX {
            let name = format!("f{i}");
            assert_eq!(keys.intern(name.as_bytes()), Some(i as u16));
        }
        assert!(keys.is_full());
        assert_eq!(keys.intern(b"one too many"), None);
        assert_eq!(
            keys.id(b"f0"),
            Some(0),
            "a full table still answers for what is in it"
        );
    }
}
