//! The database, the one call that opens it, and the handle everything else
//! reaches it through.

use core::cell::RefCell;
use std::rc::Rc;

use yo_common::{Code, Error, Result};
use yo_index::RawMap;
use yo_shape::{Desc, Shape, Tag};

use crate::counter::Counter;
use crate::doc::{Docs, Document, Documents};
use crate::graph::Graph;
use crate::keys::Keys;
use crate::keyspace::Strings;
use crate::map::Map;
use crate::sets::{Set, Sets};
use crate::store::Decode;
use crate::vector::Vectors;

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
    pub(crate) data: Data,
}

/// What a collection holds, which is decided by the handle it was opened
/// through and never changes afterwards.
///
/// One catalogue covers both kinds rather than two, because a name is a name:
/// opening `orders` as a map and then as a document collection has to be the
/// same refusal as opening it as a map of the wrong type, and it is, since the
/// shapes differ and [`yo_shape::check`] compares them before this is reached.
// The two variants are different sizes and that is the point. A map is the hot
// path and it stays where it is, so the one that would have made this enum wide
// is the one behind a pointer.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Data {
    Map(RawMap),
    /// Boxed because a document collection carries its indexes and its build
    /// buffer, and a map should not pay for the size of one.
    Docs(Box<Documents>),
    /// Boxed for the same reason, harder: a graph carries a plane, two document
    /// stores and the table that turns an id into a dense one.
    Graph(Box<crate::graph::Store>),
    /// Boxed for the same reason again: a vector collection carries the
    /// partitions, the codes under them and every full precision vector.
    Vectors(Box<yo_vector::Collection>),
}

impl Data {
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            Data::Map(m) => m.memory_bytes(),
            Data::Docs(d) => d.docs.memory_bytes(),
            Data::Graph(g) => g.memory_bytes(),
            Data::Vectors(v) => v.memory_bytes(),
        }
    }

    /// The map inside, for a handle that was handed out against one.
    ///
    /// # Panics
    ///
    /// Never, from outside: a `Map<K, V>` handle only exists for a collection
    /// whose shape is a map, and a shape cannot change under a name.
    #[track_caller]
    pub(crate) fn map(&self) -> &RawMap {
        match self {
            Data::Map(m) => m,
            _ => wrong_kind(),
        }
    }

    /// The same, for a write.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn map_mut(&mut self) -> &mut RawMap {
        match self {
            Data::Map(m) => m,
            _ => wrong_kind(),
        }
    }

    /// The documents inside, for a handle that was handed out against them.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn docs(&self) -> &yo_doc::Docs {
        match self {
            Data::Docs(d) => &d.docs,
            _ => wrong_kind(),
        }
    }

    /// The same, for a write.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn docs_mut(&mut self) -> &mut Documents {
        match self {
            Data::Docs(d) => d,
            _ => wrong_kind(),
        }
    }

    /// The graph inside, for a handle that was handed out against one.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn graph(&self) -> &crate::graph::Store {
        match self {
            Data::Graph(g) => g,
            _ => wrong_kind(),
        }
    }

    /// The same, for a write.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn graph_mut(&mut self) -> &mut crate::graph::Store {
        match self {
            Data::Graph(g) => g,
            _ => wrong_kind(),
        }
    }

    /// The vectors inside, for a handle that was handed out against them.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn vectors(&self) -> &yo_vector::Collection {
        match self {
            Data::Vectors(v) => v,
            _ => wrong_kind(),
        }
    }

    /// The same, for a write.
    ///
    /// # Panics
    ///
    /// The same as [`Data::map`].
    #[track_caller]
    pub(crate) fn vectors_mut(&mut self) -> &mut yo_vector::Collection {
        match self {
            Data::Vectors(v) => v,
            _ => wrong_kind(),
        }
    }
}

#[track_caller]
fn wrong_kind() -> ! {
    panic!(
        "this handle and the collection it names hold different things, which the shape check is there to make impossible. Please report this as a bug"
    )
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

/// Put the indexes `T` declares on a collection.
fn declare<T: Document>(data: &mut Documents) -> Result<()> {
    for (path, kind) in T::INDEXES {
        data.docs.create_index_bytes(path.as_bytes(), *kind)?;
    }
    Ok(())
}

/// The shape every graph collection has.
///
/// A graph's node and edge types register themselves under their labels as they
/// are used, so what the catalogue holds is only that this name is a graph and
/// not a map or a document collection. The per label shape check is in
/// `graph::Store`, and it is stricter than one tuple named at open time because
/// it catches a type that changed under a label it already used.
struct AGraph;

impl Shape for AGraph {
    fn describe(d: &mut Desc) {
        d.strukt("graph", &[]);
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
                            data: Data::Map(RawMap::new()),
                        });
                        Ok(inner.collections.len() - 1)
                    }
                },
            )?;
        Ok(Map::new(self.db.clone(), at, tag))
    }

    /// Open a collection of documents, creating it if this is the first time.
    ///
    /// `T` is the collection's shape, exactly as it is for [`Db::map`], and it
    /// also carries the indexes: every field the type marked with `#[yo(index)]`
    /// or one of its friends is declared here, so a collection cannot be opened
    /// without the indexes its queries need.
    ///
    /// ```
    /// use yo::Yo;
    ///
    /// #[derive(Yo)]
    /// struct Order {
    ///     #[yo(id)]
    ///     id: u64,
    ///     #[yo(index)]
    ///     status: String,
    /// }
    ///
    /// let db = yo::open(yo::MEMORY)?;
    /// let orders = db.docs::<Order>("orders")?;
    ///
    /// orders.put(&Order { id: 7, status: "open".to_owned() })?;
    /// assert_eq!(orders.find(Order::STATUS, "open")?.len(), 1);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Code::ShapeMismatch`] when the name is already a collection of another
    /// shape, and [`Code::Invalid`] for a declared index whose path is not one.
    pub fn docs<T: Document>(&self, name: &str) -> Result<Docs<T>> {
        let mut desc = Desc::new();
        T::describe(&mut desc);
        let tag = desc.tag();

        let at =
            self.db.write(
                |inner| match inner.collections.iter().position(|c| c.name == name) {
                    Some(at) => {
                        yo_shape::check(name, &inner.collections[at].desc, &desc, None)?;
                        // The shape matched, so the indexes are the ones already
                        // here and declaring them again is nothing at all. This runs
                        // anyway because it is what will create them on a collection
                        // read back off disk in M5.
                        declare::<T>(inner.collections[at].data.docs_mut())?;
                        Ok(at)
                    }
                    None => {
                        let mut data = Documents::new();
                        declare::<T>(&mut data)?;
                        inner.collections.push(Collection {
                            name: name.to_owned(),
                            desc,
                            data: Data::Docs(Box::new(data)),
                        });
                        Ok(inner.collections.len() - 1)
                    }
                },
            )?;
        Ok(Docs::new(self.db.clone(), at, tag))
    }

    /// Open a graph, creating it if this is the first time.
    ///
    /// The name has one shape like every other collection, and it is the same
    /// shape for every graph, because a graph's types are not fixed when it is
    /// opened. A node type or an edge type registers itself under its label the
    /// first time it is used, and the shape it registered is checked on every
    /// later use, so the check is per label rather than one tuple named up
    /// front. That way adding a node type to a program is not a schema change
    /// for the types already in the graph.
    ///
    /// ```
    /// use yo::{Edge, Node, Yo};
    ///
    /// #[derive(Yo)]
    /// struct Person { #[yo(id)] id: u64, name: String }
    ///
    /// #[derive(Yo)]
    /// struct Follows { since: i64 }
    ///
    /// impl Node for Person { const LABEL: &'static str = "Person"; }
    /// impl Edge for Follows {
    ///     type From = Person;
    ///     type To = Person;
    ///     const LABEL: &'static str = "FOLLOWS";
    /// }
    ///
    /// let db = yo::open(yo::MEMORY)?;
    /// let g = db.graph("social")?;
    ///
    /// let ada = g.add(&Person { id: 1, name: "ada".to_owned() })?;
    /// let grace = g.add(&Person { id: 2, name: "grace".to_owned() })?;
    /// g.link(ada, grace, &Follows { since: 2026 })?;
    ///
    /// assert_eq!(g.out::<Follows>(ada)?, vec![grace]);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Code::ShapeMismatch`] when the name is already a collection of another
    /// shape, which includes a name that is already a map or a document
    /// collection.
    pub fn graph(&self, name: &str) -> Result<Graph> {
        let mut desc = Desc::new();
        AGraph::describe(&mut desc);

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
                            data: Data::Graph(Box::new(crate::graph::Store::new())),
                        });
                        Ok(inner.collections.len() - 1)
                    }
                },
            )?;
        Ok(Graph::new(self.db.clone(), at))
    }

    /// Open a collection of vectors, creating it if this is the first time.
    ///
    /// Nearness is euclidean distance. [`Db::vectors_with`] is the same call
    /// with the metric spelled out, and cosine is the other one worth having.
    ///
    /// The dimension is part of the collection's shape exactly as a map's value
    /// type is, so a collection opened at 768 dimensions cannot later be opened
    /// at 1536 and quietly read the old vectors as half of a new one.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let v = db.vectors("passages", 3)?;
    ///
    /// v.put("a", &[1.0, 0.0, 0.0])?;
    /// v.put("b", &[0.0, 1.0, 0.0])?;
    ///
    /// assert_eq!(v.search(&[0.9, 0.1, 0.0], 1)?[0].key, b"a".to_vec());
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Code::ShapeMismatch`] when the name is already a collection of another
    /// shape, which includes the same name at another dimension or metric, and
    /// [`Code::Invalid`] for a dimension no collection can hold.
    pub fn vectors(&self, name: &str, dim: usize) -> Result<Vectors> {
        self.vectors_with(name, dim, yo_shape::Metric::L2)
    }

    /// The same, saying what nearness means.
    ///
    /// [`Metric::Cosine`](yo_shape::Metric::Cosine) stores the unit vector and
    /// reports one minus the cosine similarity, which is what a collection of
    /// text embeddings wants. See the [`vector`](crate::vector) module for what
    /// each metric does and why two of the four are refused.
    ///
    /// # Errors
    ///
    /// As [`Db::vectors`], and [`Code::Unsupported`] for a metric this build
    /// does not measure.
    pub fn vectors_with(
        &self,
        name: &str,
        dim: usize,
        metric: yo_shape::Metric,
    ) -> Result<Vectors> {
        let width = yo_vector::collection::width(dim)?;
        yo_vector::collection::check_metric(metric)?;

        let mut desc = Desc::new();
        desc.vector(width, metric);

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
                            data: Data::Vectors(Box::new(yo_vector::Collection::new(dim, metric)?)),
                        });
                        Ok(inner.collections.len() - 1)
                    }
                },
            )?;
        Ok(Vectors {
            db: self.db.clone(),
            at,
            dim,
            metric,
        })
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

    /// Every Redis set command, with the key as the first argument.
    ///
    /// The same store `SADD` off a socket reaches. Like [`Db::strings`] this is
    /// not a named collection, because in Redis a set is not one: it is a key in
    /// the keyspace that happens to hold a set.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// db.sets().add_many("online", &["alice", "bob"])?;
    /// assert_eq!(db.sets().len_of("online")?, 2);
    /// # Ok::<(), yo::Error>(())
    /// ```
    #[must_use]
    pub fn sets(&self) -> Sets {
        Sets {
            db: self.db.clone(),
        }
    }

    /// A set at one key, which is the same sugar [`Db::counter`] is.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let online = db.set("online");
    ///
    /// online.add("alice")?;
    /// assert!(online.contains("alice")?);
    /// # Ok::<(), yo::Error>(())
    /// ```
    #[must_use]
    pub fn set(&self, key: impl Into<Vec<u8>>) -> Set {
        Set {
            sets: self.sets(),
            key: key.into(),
        }
    }

    /// Every command that works on a key whatever the key holds.
    ///
    /// `DEL`, `EXISTS` and `TYPE`, and the whole expiry family. These are the
    /// ones that belong to the keyspace rather than to a type, which is why
    /// they are not on [`Db::strings`] or [`Db::sets`]: a deadline sits in the
    /// key's record and does not care what the record points at.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// let db = yo::open(yo::MEMORY)?;
    /// db.set("online").add("alice")?;
    /// db.keys().expire_in("online", Duration::from_secs(60))?;
    /// # Ok::<(), yo::Error>(())
    /// ```
    #[must_use]
    pub fn keys(&self) -> Keys {
        Keys {
            db: self.db.clone(),
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
