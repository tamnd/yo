//! The database, the one call that opens it, and the handle everything else
//! reaches it through.

use core::cell::RefCell;
use std::rc::Rc;

use yo_common::{Code, Error, Result};
use yo_index::RawMap;
use yo_shape::{Desc, Tag};

use crate::counter::Counter;
use crate::keyspace::Strings;
use crate::map::Map;
use crate::store::Decode;

/// The path that means "no file at all", which is a real path and not a flag
/// (`07` section 7).
pub const MEMORY: &str = ":memory:";

/// Open a database.
///
/// That is the whole setup. There is no database to build before a connection,
/// no configuration to pass, no engine to choose, and no pool. The engine is
/// inferred from the path, and [`MEMORY`] is a path.
///
/// # Errors
///
/// [`Code::Unsupported`] for a path on disk, until the file format lands in
/// M5. Everything else about the API is the same either way, which is the
/// point of putting the front door in before the file.
pub fn open(path: &str) -> Result<Db> {
    if path != MEMORY {
        return Err(Error::fmt(
            Code::Unsupported,
            format_args!(
                "this build holds a database in memory only, so the path has to be \"{MEMORY}\", not \"{path}\". A file backed database arrives with the .yo format in M5"
            ),
        ));
    }
    Ok(Db {
        db: Handle {
            inner: Rc::new(RefCell::new(Inner {
                collections: Vec::new(),
                strings: yo_kv::Keyspace::new(),
                deadlines: false,
            })),
        },
    })
}

/// An open database.
///
/// Cheap to clone, and every clone is the same database. A handle taken out of
/// it stays valid for as long as any clone lives.
///
/// This build runs in inline mode (`15` section 7): the calling thread is the
/// shard, which is what makes a point read a function call rather than a
/// message. That is also why a handle does not cross threads yet. The owned
/// and served modes put the same API on top of `yo-shard`'s runtime, and they
/// arrive with it.
#[derive(Clone)]
pub struct Db {
    db: Handle,
}

impl core::fmt::Debug for Db {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut d = f.debug_struct("Db");
        match self.collections() {
            Ok(names) => d.field("collections", &names).finish(),
            Err(_) => d.finish_non_exhaustive(),
        }
    }
}

/// The database itself, which one thread owns and reaches through [`Handle`].
pub(crate) struct Inner {
    pub(crate) collections: Vec<Collection>,
    pub(crate) strings: yo_kv::Keyspace,
    /// Whether any key has ever been given a deadline, which is exactly when
    /// the clock's answer can be observed. See `keyspace`'s module docs.
    pub(crate) deadlines: bool,
}

pub(crate) struct Collection {
    pub(crate) name: String,
    pub(crate) desc: Desc,
    pub(crate) data: RawMap,
}

/// A shared, cheap pointer to one database.
///
/// Every handle the user holds is one of these plus whatever names the thing
/// it points at, so a `Map` is two words and an index and a `Counter` is two
/// words and a key.
#[derive(Clone)]
pub(crate) struct Handle {
    inner: Rc<RefCell<Inner>>,
}

impl Handle {
    /// Run something against the database, with the clock brought up to date
    /// first if any deadline exists to compare against.
    pub(crate) fn run<R>(&self, f: impl FnOnce(&mut Inner) -> Result<R>) -> Result<R> {
        let mut inner = self.inner.try_borrow_mut().map_err(|_| reentrant())?;
        if inner.deadlines {
            inner.strings.clock_mut().refresh();
        }
        f(&mut inner)
    }

    /// The same, for something that is about to create a deadline.
    pub(crate) fn deadlines<R>(&self, f: impl FnOnce(&mut Inner) -> Result<R>) -> Result<R> {
        let mut inner = self.inner.try_borrow_mut().map_err(|_| reentrant())?;
        inner.deadlines = true;
        inner.strings.clock_mut().refresh();
        f(&mut inner)
    }

    /// A shared look at the database, which is what a read of a typed
    /// collection needs and nothing more.
    pub(crate) fn read<R>(&self, f: impl FnOnce(&Inner) -> Result<R>) -> Result<R> {
        let inner = self.inner.try_borrow().map_err(|_| reentrant())?;
        f(&inner)
    }

    /// A write that no deadline can be observed through, which is every write
    /// to a typed collection so far. The clock is left where it is.
    pub(crate) fn write<R>(&self, f: impl FnOnce(&mut Inner) -> Result<R>) -> Result<R> {
        let mut inner = self.inner.try_borrow_mut().map_err(|_| reentrant())?;
        f(&mut inner)
    }

    /// Whether two handles point at the same database.
    pub(crate) fn is(&self, other: &Handle) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

/// The error a call made from inside another call's callback gets.
///
/// A database that panics because of how the caller nested two of its own
/// methods is a database people stop trusting, so re-entrancy is an error
/// value with a sentence attached rather than a `RefCell` panic.
pub(crate) fn reentrant() -> Error {
    Error::new(
        Code::Invalid,
        "this database is already in use by the call above this one. A closure passed to with() or update() cannot call back into the same database, so read what you need first and write after the closure returns",
    )
}

impl Db {
    /// Open a map, creating it if this is the first time.
    ///
    /// The type is the collection's shape (`15` section 3), so opening the
    /// same name a second time with a different type is an error and not a
    /// surprise later: the shapes are compared, and a mismatch says which
    /// field moved and whether the change is additive or breaking.
    ///
    /// # Errors
    ///
    /// [`Code::ShapeMismatch`] when the name is already a collection of
    /// another shape.
    pub fn map<K: Decode, V: Decode>(&self, name: &str) -> Result<Map<K, V>> {
        let mut desc = Desc::new();
        desc.map(K::describe, V::describe);
        let tag = desc.tag();

        let at =
            self.db.write(
                |inner| match inner.collections.iter().position(|c| c.name == name) {
                    Some(at) => {
                        yo_shape::check(name, &inner.collections[at].desc, &desc, None)?;
                        Ok(at)
                    }
                    None => {
                        inner.collections.push(Collection {
                            name: name.to_owned(),
                            desc,
                            data: RawMap::new(),
                        });
                        Ok(inner.collections.len() - 1)
                    }
                },
            )?;
        Ok(Map::new(self.db.clone(), at, tag))
    }

    /// The Redis string keyspace.
    ///
    /// The same store a client reaches over RESP, reached without the socket,
    /// the parser or the reply (Y23). Not a named collection, because in Redis
    /// a string is not one: it is the keyspace itself.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// assert_eq!(db.strings().incr("hits")?, 1);
    /// # Ok::<(), yo::Error>(())
    /// ```
    #[must_use]
    pub fn strings(&self) -> Strings {
        Strings {
            db: self.db.clone(),
        }
    }

    /// A counter at one key, which is `15` section 2's `db.counter("hits")`.
    ///
    /// Sugar over [`Db::strings`] and worth having: a counter is the commonest
    /// thing a string key is, and a handle that holds the key means the key is
    /// spelled once rather than at every call site.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let hits = db.counter("hits");
    ///
    /// hits.incr()?;
    /// hits.add(9)?;
    /// assert_eq!(hits.get()?, 10);
    /// # Ok::<(), yo::Error>(())
    /// ```
    #[must_use]
    pub fn counter(&self, key: impl Into<Vec<u8>>) -> Counter {
        Counter {
            db: self.db.clone(),
            key: key.into(),
        }
    }

    /// The names of the typed collections in this database, in the order they
    /// were first opened.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn collections(&self) -> Result<Vec<String>> {
        self.db
            .read(|inner| Ok(inner.collections.iter().map(|c| c.name.clone()).collect()))
    }

    /// The shape of a collection, if it exists.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn shape(&self, name: &str) -> Result<Option<Tag>> {
        self.db.read(|inner| {
            Ok(inner
                .collections
                .iter()
                .find(|c| c.name == name)
                .map(|c| c.desc.tag()))
        })
    }

    /// What this database is holding, index and arena together, across the
    /// keyspace and every typed collection.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn memory_bytes(&self) -> Result<usize> {
        self.db.read(|inner| {
            Ok(inner.strings.memory_bytes()
                + inner
                    .collections
                    .iter()
                    .map(|c| c.data.memory_bytes())
                    .sum::<usize>())
        })
    }

    /// Whether this database reads the clock on the data path.
    ///
    /// False until something is given a deadline, because until then the
    /// clock's answer cannot change any reply. `04` section 5 is the reason
    /// this is worth a method: a clock read is tens of nanoseconds against a
    /// budget of a hundred and fifty.
    #[must_use]
    pub fn reads_the_clock(&self) -> bool {
        self.db.read(|inner| Ok(inner.deadlines)).unwrap_or(false)
    }

    /// Whether two databases are the same one.
    #[must_use]
    pub fn is(&self, other: &Db) -> bool {
        self.db.is(&other.db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_on_disk_says_which_build_would_take_it() {
        let e = open("app.yo").expect_err("no file format yet");
        assert_eq!(e.code(), Code::Unsupported);
        assert!(e.message().contains("M5"), "{e}");
    }

    #[test]
    fn opening_the_same_name_twice_gives_the_same_collection() {
        let db = open(MEMORY).unwrap();
        let a = db.map::<String, u64>("hits").unwrap();
        let b = db.map::<String, u64>("hits").unwrap();
        a.set("home", &1).unwrap();
        assert_eq!(b.get("home").unwrap(), Some(1));
        assert_eq!(db.collections().unwrap(), vec!["hits".to_owned()]);
    }

    /// The whole reason the tag exists, from the caller's side: the second
    /// open does not quietly hand back a map that reads other people's bytes
    /// as its own type.
    #[test]
    fn opening_the_same_name_with_another_type_is_a_shape_mismatch() {
        let db = open(MEMORY).unwrap();
        let _first = db.map::<String, u64>("hits").unwrap();
        let e = db
            .map::<String, String>("hits")
            .expect_err("that is a different shape");
        assert_eq!(e.code(), Code::ShapeMismatch);
        assert!(
            e.message().contains("the type changed from u64 to str"),
            "{e}"
        );
        assert_eq!(e.detail(), Some("change=breaking"));
    }

    #[test]
    fn two_collections_are_two_keyspaces() {
        let db = open(MEMORY).unwrap();
        let a = db.map::<String, u64>("a").unwrap();
        let b = db.map::<String, u64>("b").unwrap();
        a.set("k", &1).unwrap();
        b.set("k", &2).unwrap();
        assert_eq!(a.get("k").unwrap(), Some(1));
        assert_eq!(b.get("k").unwrap(), Some(2));
        assert_eq!(db.collections().unwrap().len(), 2);
    }

    /// A typed collection and the Redis keyspace do not see each other, which
    /// is what the catalogue in `07` section 5 says: a collection is a name,
    /// and the string type is the keyspace.
    #[test]
    fn a_typed_collection_and_the_keyspace_are_not_the_same_store() {
        let db = open(MEMORY).unwrap();
        let map = db.map::<String, u64>("hits").unwrap();
        map.set("home", &1).unwrap();
        db.strings().set("home", "elsewhere").unwrap();

        assert_eq!(map.get("home").unwrap(), Some(1));
        assert_eq!(
            db.strings().get("home").unwrap().as_deref(),
            Some(&b"elsewhere"[..])
        );
    }

    #[test]
    fn a_shape_can_be_read_back_and_an_unopened_name_has_none() {
        let db = open(MEMORY).unwrap();
        let map = db.map::<String, u64>("hits").unwrap();
        assert_eq!(db.shape("hits").unwrap(), Some(map.tag()));
        assert_eq!(db.shape("misses").unwrap(), None);
    }

    #[test]
    fn a_clone_is_the_same_database() {
        let db = open(MEMORY).unwrap();
        let map = db.map::<String, u64>("hits").unwrap();
        map.set("home", &3).unwrap();
        let same = db.clone();
        assert_eq!(
            same.map::<String, u64>("hits")
                .unwrap()
                .get("home")
                .unwrap(),
            Some(3)
        );
        assert!(db.is(&same));
        assert!(!db.is(&open(MEMORY).unwrap()));
        assert!(db.memory_bytes().unwrap() > 0);
        assert!(format!("{db:?}").contains("hits"));
    }
}
