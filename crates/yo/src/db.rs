//! The database and the one call that opens it.

use core::cell::RefCell;
use std::rc::Rc;

use yo_common::{Code, Error, Result};
use yo_index::RawMap;
use yo_shape::{Desc, Tag};

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
        inner: Rc::new(RefCell::new(Inner {
            collections: Vec::new(),
        })),
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
    inner: Rc<RefCell<Inner>>,
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

pub(crate) struct Inner {
    pub(crate) collections: Vec<Collection>,
}

pub(crate) struct Collection {
    pub(crate) name: String,
    pub(crate) desc: Desc,
    pub(crate) data: RawMap,
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
        let mut inner = self.borrow_mut()?;
        let at = match inner.collections.iter().position(|c| c.name == name) {
            Some(at) => {
                yo_shape::check(name, &inner.collections[at].desc, &desc, None)?;
                at
            }
            None => {
                inner.collections.push(Collection {
                    name: name.to_owned(),
                    desc,
                    data: RawMap::new(),
                });
                inner.collections.len() - 1
            }
        };
        drop(inner);
        Ok(Map::new(Rc::clone(&self.inner), at, tag))
    }

    /// The names of the collections in this database, in the order they were
    /// first opened.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn collections(&self) -> Result<Vec<String>> {
        Ok(self
            .borrow()?
            .collections
            .iter()
            .map(|c| c.name.clone())
            .collect())
    }

    /// The shape of a collection, if it exists.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn shape(&self, name: &str) -> Result<Option<Tag>> {
        Ok(self
            .borrow()?
            .collections
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.desc.tag()))
    }

    /// What this database is holding, index and arena together.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn memory_bytes(&self) -> Result<usize> {
        Ok(self
            .borrow()?
            .collections
            .iter()
            .map(|c| c.data.memory_bytes())
            .sum())
    }

    fn borrow(&self) -> Result<core::cell::Ref<'_, Inner>> {
        self.inner.try_borrow().map_err(|_| crate::map::reentrant())
    }

    fn borrow_mut(&self) -> Result<core::cell::RefMut<'_, Inner>> {
        self.inner
            .try_borrow_mut()
            .map_err(|_| crate::map::reentrant())
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
        assert!(db.memory_bytes().unwrap() > 0);
    }
}
