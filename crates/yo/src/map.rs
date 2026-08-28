//! `Map<K, V>`, the first typed handle (`15` section 2).

use core::marker::PhantomData;
use std::borrow::Borrow;

use yo_common::{Code, Error, Result};
use yo_index::RawMap;
use yo_shape::Tag;

use crate::db::Handle;
use crate::store::{Decode, Encode};

/// A map from `K` to `V`.
///
/// Cheap to clone and cheap to keep around: the handle is a pointer and an
/// index, so cloning one does not copy anything and every clone is the same
/// collection.
///
/// The type parameters are not decoration. They are the collection's shape
/// (`15` section 3), they are written down when it is created, and they are
/// what a later open is checked against.
///
/// # Borrowed lookups
///
/// A lookup takes a borrowed form of the key, the same way `HashMap` does, so
/// a `Map<String, u64>` is read with `map.get("home")` and not with a `String`
/// built for the length of one call.
///
/// ```
/// let db = yo::open(yo::MEMORY)?;
/// let hits = db.map::<String, u64>("hits")?;
///
/// hits.set("home", &1)?;
/// assert_eq!(hits.get("home")?, Some(1));
/// assert_eq!(hits.get("about")?, None);
/// # Ok::<(), yo::Error>(())
/// ```
pub struct Map<K, V> {
    db: Handle,
    at: usize,
    tag: Tag,
    /// `fn() -> (K, V)` rather than `(K, V)` so that the handle's auto traits
    /// and variance come from the handle rather than from what it holds.
    marker: PhantomData<fn() -> (K, V)>,
}

impl<K, V> Clone for Map<K, V> {
    fn clone(&self) -> Map<K, V> {
        Map {
            db: self.db.clone(),
            at: self.at,
            tag: self.tag,
            marker: PhantomData,
        }
    }
}

impl<K, V> core::fmt::Debug for Map<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = self
            .db
            .read(|inner| Ok(inner.collections[self.at].name.clone()))
            .unwrap_or_else(|_| "?".to_owned());
        f.debug_struct("Map").field("name", &name).finish()
    }
}

impl<K: Decode, V: Decode> Map<K, V> {
    pub(crate) fn new(db: Handle, at: usize, tag: Tag) -> Map<K, V> {
        Map {
            db,
            at,
            tag,
            marker: PhantomData,
        }
    }

    /// The name this collection was opened under.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn name(&self) -> Result<String> {
        self.read(|c| Ok(c.name.clone()))
    }

    /// This collection's shape tag.
    ///
    /// The same 128 bits that a file carries and that another language's
    /// binding computes for the same type. Carried in the handle rather than
    /// read out of the database, so it costs nothing and cannot fail.
    #[must_use]
    pub fn tag(&self) -> Tag {
        self.tag
    }

    /// Read a value, owned.
    ///
    /// One allocation for the value, and none for the key. When even that one
    /// is too many, [`Map::with`] hands over the bytes where they lie.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the stored bytes are not a `V`, which means the
    /// file disagrees with its own shape.
    pub fn get<Q>(&self, key: &Q) -> Result<Option<V>>
    where
        K: Borrow<Q>,
        Q: Encode + ?Sized,
    {
        self.read(|c| match key.encode(|k| c.data.get(k)) {
            // Decoded straight out of the arena rather than copied out and then
            // decoded, so a fixed width value costs no allocation at all and a
            // string costs exactly the one the caller asked for.
            Some(bytes) => V::decode(bytes).map(Some),
            None => Ok(None),
        })
    }

    /// Read a value without copying it, by handing the borrowed view to `f`.
    ///
    /// This is Y29 in one method: zero copy is always available and never
    /// mandatory. The view points into the arena, so nothing is allocated and
    /// nothing is decoded beyond checking that the bytes are a `V`. It is also
    /// where G6's point read budget is spent, which is why the closure takes
    /// the view rather than the value.
    ///
    /// `f` runs while the database is borrowed, so it cannot call back into
    /// the same database. One that tries gets [`Code::Invalid`] rather than a
    /// panic.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let names = db.map::<u64, String>("names")?;
    /// names.set(&7, "ada")?;
    ///
    /// // No String is built here, and no bytes are copied.
    /// let len = names.with(&7, str::len)?;
    /// assert_eq!(len, Some(3));
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the stored bytes are not a `V`.
    pub fn with<Q, R>(&self, key: &Q, f: impl FnOnce(V::Ref<'_>) -> R) -> Result<Option<R>>
    where
        K: Borrow<Q>,
        Q: Encode + ?Sized,
    {
        self.read(|c| match key.encode(|k| c.data.get(k)) {
            Some(bytes) => V::view(bytes).map(|view| Some(f(view))),
            None => Ok(None),
        })
    }

    /// Store a value, replacing whatever was there.
    ///
    /// The value is taken borrowed as well as the key, so a
    /// `Map<String, String>` is written with `map.set("k", "v")`.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] if the key and the value together are larger than
    /// [`Map::max_entry`]. A value that big belongs in the log region, which
    /// arrives with the file format in M5.
    pub fn set<Q, W>(&self, key: &Q, value: &W) -> Result<()>
    where
        K: Borrow<Q>,
        Q: Encode + ?Sized,
        V: Borrow<W>,
        W: Encode + ?Sized,
    {
        self.write(|c| {
            key.encode(|k| {
                value.encode(|v| {
                    let total = RawMap::header_len() + k.len() + v.len();
                    if total > RawMap::max_record() {
                        return Err(too_big(total));
                    }
                    c.data.set(k, v);
                    Ok(())
                })
            })
        })
    }

    /// Remove a key, returning whether it was there.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn del<Q>(&self, key: &Q) -> Result<bool>
    where
        K: Borrow<Q>,
        Q: Encode + ?Sized,
    {
        self.write(|c| Ok(key.encode(|k| c.data.del(k))))
    }

    /// Whether a key is present, without reading its value.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn contains<Q>(&self, key: &Q) -> Result<bool>
    where
        K: Borrow<Q>,
        Q: Encode + ?Sized,
    {
        self.read(|c| Ok(key.encode(|k| c.data.contains(k))))
    }

    /// How many keys are stored.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn len(&self) -> Result<usize> {
        self.read(|c| Ok(c.data.len()))
    }

    /// Whether the collection is empty.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn is_empty(&self) -> Result<bool> {
        self.read(|c| Ok(c.data.is_empty()))
    }

    /// The largest key and value this collection takes, the two together.
    #[must_use]
    pub const fn max_entry() -> usize {
        RawMap::max_record() - RawMap::header_len()
    }

    fn read<R>(&self, f: impl FnOnce(&crate::db::Collection) -> Result<R>) -> Result<R> {
        self.db.read(|inner| f(&inner.collections[self.at]))
    }

    fn write<R>(&self, f: impl FnOnce(&mut crate::db::Collection) -> Result<R>) -> Result<R> {
        self.db.write(|inner| f(&mut inner.collections[self.at]))
    }
}

fn too_big(total: usize) -> Error {
    Error::fmt(
        Code::Full,
        format_args!(
            "a key and value of {} bytes is larger than the {} a record holds. A value that size belongs in the log region, which arrives with the .yo format in M5",
            total - RawMap::header_len(),
            Map::<Vec<u8>, Vec<u8>>::max_entry()
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::{MEMORY, open};

    #[test]
    fn a_map_of_strings_to_numbers_reads_back_what_it_wrote() {
        let db = open(MEMORY).unwrap();
        let hits = db.map::<String, u64>("hits").unwrap();

        assert!(hits.is_empty().unwrap());
        hits.set("home", &1).unwrap();
        hits.set("about", &2).unwrap();

        assert_eq!(hits.get("home").unwrap(), Some(1));
        assert_eq!(hits.get("about").unwrap(), Some(2));
        assert_eq!(hits.get("nowhere").unwrap(), None);
        assert_eq!(hits.len().unwrap(), 2);
        assert!(hits.contains("home").unwrap());
        assert_eq!(hits.name().unwrap(), "hits");
    }

    #[test]
    fn a_write_replaces_and_a_delete_removes() {
        let db = open(MEMORY).unwrap();
        let hits = db.map::<String, u64>("hits").unwrap();

        hits.set("home", &1).unwrap();
        hits.set("home", &9).unwrap();
        assert_eq!(hits.get("home").unwrap(), Some(9));
        assert_eq!(hits.len().unwrap(), 1);

        assert!(hits.del("home").unwrap());
        assert!(!hits.del("home").unwrap());
        assert_eq!(hits.get("home").unwrap(), None);
        assert!(hits.is_empty().unwrap());
    }

    /// The DX claim in one test: neither side of a write needs an owned value
    /// built for the length of the call.
    #[test]
    fn neither_the_key_nor_the_value_has_to_be_owned() {
        let db = open(MEMORY).unwrap();
        let names = db.map::<String, String>("names").unwrap();

        names.set("7", "ada").unwrap();
        assert_eq!(names.get("7").unwrap().as_deref(), Some("ada"));
        assert_eq!(
            names.with("7", str::to_owned).unwrap().as_deref(),
            Some("ada")
        );
    }

    #[test]
    fn keys_can_be_numbers_and_values_can_be_bytes() {
        let db = open(MEMORY).unwrap();
        let blobs = db.map::<u64, Vec<u8>>("blobs").unwrap();

        blobs.set(&7, b"\x00\xff".as_slice()).unwrap();
        assert_eq!(blobs.get(&7).unwrap().as_deref(), Some(&b"\x00\xff"[..]));
        assert_eq!(blobs.with(&7, <[u8]>::len).unwrap(), Some(2));
        assert_eq!(blobs.with(&8, <[u8]>::len).unwrap(), None);
    }

    #[test]
    fn a_clone_of_a_handle_is_the_same_collection() {
        let db = open(MEMORY).unwrap();
        let hits = db.map::<String, u64>("hits").unwrap();
        let same = hits.clone();

        hits.set("home", &4).unwrap();
        assert_eq!(same.get("home").unwrap(), Some(4));
        assert_eq!(same.tag(), hits.tag());
        assert!(format!("{same:?}").contains("hits"));
    }

    /// A `with` closure that reaches back into the same database is a mistake
    /// the caller can fix, so it gets a sentence rather than a panic.
    #[test]
    fn calling_back_into_the_database_from_a_closure_is_an_error() {
        let db = open(MEMORY).unwrap();
        let hits = db.map::<String, u64>("hits").unwrap();
        hits.set("home", &1).unwrap();

        let inner = hits.clone();
        let e = hits
            .with("home", |_| inner.set("other", &2))
            .unwrap()
            .unwrap()
            .expect_err("a write inside a read is re-entrant");
        assert_eq!(e.code(), yo_common::Code::Invalid);
        assert!(e.message().contains("cannot call back"), "{e}");

        // A read inside a read is fine, because nothing is being moved.
        assert_eq!(
            hits.with("home", |_| inner.get("home").unwrap()).unwrap(),
            Some(Some(1))
        );
    }

    #[test]
    fn a_record_larger_than_the_arena_takes_is_full_rather_than_a_panic() {
        let db = open(MEMORY).unwrap();
        let blobs = db.map::<String, Vec<u8>>("blobs").unwrap();

        let huge = vec![0u8; super::Map::<String, Vec<u8>>::max_entry()];
        let e = blobs
            .set("k", huge.as_slice())
            .expect_err("one byte too far");
        assert_eq!(e.code(), yo_common::Code::Full);
        assert!(e.message().contains("belongs in the log region"), "{e}");

        // And the edge itself still fits, header and key included.
        let fits = vec![0u8; super::Map::<String, Vec<u8>>::max_entry() - 1];
        blobs.set("k", fits.as_slice()).unwrap();
        assert_eq!(blobs.with("k", <[u8]>::len).unwrap(), Some(fits.len()));
    }
}
