//! `Docs<T>`, the typed document handle, and the traits `#[derive(Yo)]` writes
//! (`15` sections 2 and 4).
//!
//! A document collection is your own struct, stored as your own struct. There
//! is no schema to declare, no JSON text to parse on either side, and no query
//! language: a struct goes in, the same struct comes out, and the fields worth
//! looking documents up by say so with an attribute.
//!
//! ```
//! use yo::Yo;
//!
//! #[derive(Yo, Debug, PartialEq)]
//! struct Order {
//!     #[yo(id)]
//!     id: u64,
//!     #[yo(index)]
//!     status: String,
//!     #[yo(ordered)]
//!     total: f64,
//! }
//!
//! let db = yo::open(yo::MEMORY)?;
//! let orders = db.docs::<Order>("orders")?;
//!
//! orders.put(&Order { id: 1, status: "open".to_owned(), total: 12.5 })?;
//! orders.put(&Order { id: 2, status: "shipped".to_owned(), total: 99.0 })?;
//!
//! assert_eq!(orders.get(&1)?.unwrap().total, 12.5);
//! assert_eq!(orders.find(Order::STATUS, "open")?.len(), 1);
//! assert_eq!(orders.count(Order::STATUS, "shipped")?, 1);
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! # The query is a constant, not a string
//!
//! `Order::STATUS` is a [`Path`] the derive wrote, and `Order::TOTAL` is an
//! [`Ordered`], which is what a `#[yo(ordered)]` field gets. A field that is not
//! indexed has no constant at all, so asking for one is a name that does not
//! exist rather than a query that quietly turns into a scan. [`Docs::range`]
//! takes an `Ordered` and nothing else, so asking an equality index for a range
//! is a type error at the call site.
//!
//! ```compile_fail
//! # use yo::Yo;
//! # #[derive(Yo)]
//! # struct Order { #[yo(id)] id: u64, #[yo(index)] status: String }
//! # let db = yo::open(yo::MEMORY).unwrap();
//! # let orders = db.docs::<Order>("orders").unwrap();
//! // The index on status answers equality, so there is no range to walk.
//! orders.range(Order::STATUS, "a".."z").unwrap();
//! ```
//!
//! The value side is typed too, so comparing a number field against a string is
//! the same kind of mistake and gets the same answer.
//!
//! ```compile_fail
//! # use yo::Yo;
//! # #[derive(Yo)]
//! # struct Order { #[yo(id)] id: u64, #[yo(ordered)] total: f64 }
//! # let db = yo::open(yo::MEMORY).unwrap();
//! # let orders = db.docs::<Order>("orders").unwrap();
//! orders.find(Order::TOTAL, "twelve").unwrap();
//! ```
//!
//! # A range over a string field takes a pair of bounds
//!
//! `orders.range(Order::NAME, "a".."m")` does not compile, and the reason is not
//! this crate. `Range<&str>` only implements `RangeBounds<str>` when `str` is
//! sized, which it is not, so the standard library's own
//! `BTreeMap<String, u8>::range("a".."m")` is rejected the same way. Writing the
//! two ends out is what works there and it is what works here.
//!
//! ```
//! # use std::ops::Bound;
//! # use yo::Yo;
//! # #[derive(Yo, Debug)]
//! # struct Order { #[yo(id)] id: u64, #[yo(ordered)] name: String }
//! # let db = yo::open(yo::MEMORY)?;
//! # let orders = db.docs::<Order>("orders")?;
//! # orders.put(&Order { id: 1, name: "banana".to_owned() })?;
//! # orders.put(&Order { id: 2, name: "quince".to_owned() })?;
//! let early = orders.range(Order::NAME, (Bound::Included("a"), Bound::Excluded("m")))?;
//! assert_eq!(early.len(), 1);
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! A range over a number field is written the way anyone would write it, because
//! the numbers are sized and `0.0..50.0` is a `RangeBounds<f64>` already.
//!
//! # What a field can be
//!
//! [`Field`] is the list, and it is the JSON types rather than the Rust ones,
//! because a document is JSON shaped whatever it was written from. The integers
//! and floats, `bool`, `String`, `Option<T>` for a field that may be absent,
//! `Vec<T>` for a list, and any other type that derives `Yo`, which nests.
//!
//! An integer is stored as an `i64`, which is the one number type JSON has, so
//! a `u64` above `i64::MAX` is refused on the way in rather than silently
//! rounded through a float.

use core::marker::PhantomData;
use core::ops::{Bound, RangeBounds};

use yo_common::{Code, Error, Result};
use yo_shape::{Shape, Tag};

use crate::db::Handle;

pub use yo_doc::{Builder, Doc, IndexKind, Key};

/// A type that can be a field of a document.
///
/// The encoding is YOJB and not a Rust layout, so what goes in the store is
/// what a document is: an object with named fields, readable by the RESP
/// surface and by another language's binding without either of them knowing
/// what Rust is.
pub trait Field: Shape + Sized {
    /// Write this value into the document being built.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for a value the document encoding cannot hold, which
    /// is a `u64` past `i64::MAX` and nothing else so far.
    fn write(&self, b: &mut Builder) -> Result<()>;

    /// Read this value back out.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] when the stored value is not this type, which means
    /// the collection disagrees with its own shape.
    fn read(d: Doc<'_>) -> Result<Self>;

    /// What to do when the field is not in the document at all.
    ///
    /// An error for everything except [`Option`], which is the whole point of
    /// having an `Option`: a field that may be absent says so in the type, and
    /// every other field being absent is a document that does not match the
    /// shape it was stored under.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`], unless the type is an `Option`.
    fn missing(name: &str) -> Result<Self> {
        Err(Error::fmt(
            Code::Corrupt,
            format_args!(
                "this document has no {name}, and the field is not an Option. Either the collection holds something written under another shape, or the field was added without a default"
            ),
        ))
    }
}

/// A value that can be an index key.
///
/// Separate from [`Field`] because a lookup takes the borrowed form, the same
/// way `HashMap::get` does, so a `String` field is searched with `&str` and not
/// with a `String` built for the length of one call.
pub trait Query {
    /// The key this value is filed under in an index of `kind`, or `None` if an
    /// index of that kind does not file this type at all.
    fn key(&self, kind: IndexKind) -> Option<Key>;
}

/// How a field's type is written in a query.
///
/// An associated type rather than a `Borrow` bound on the call, because a
/// `Borrow` bound leaves the compiler two ways to read `"a".."z"` and it picks
/// the wrong one. This way the borrowed form follows from the field's type and
/// there is nothing to infer.
pub trait Asked: Query {
    /// The borrowed form, which is `str` for a `String` and the type itself for
    /// everything else.
    type Ask: Query + ?Sized;
}

macro_rules! asks_for_itself {
    ($($t:ty),* $(,)?) => {
        $(impl Asked for $t {
            type Ask = $t;
        })*
    };
}

asks_for_itself!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);

impl Asked for String {
    type Ask = str;
}

/// A type that is a whole document: a [`Field`] with an id and its indexes.
///
/// Written by `#[derive(Yo)]` from the field marked `#[yo(id)]`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a document",
    label = "this type has no id",
    note = "add `#[derive(Yo)]` to it and mark one field `#[yo(id)]`, which is what a document is stored under"
)]
pub trait Document: Field + Indexed {
    /// The type of the field marked `#[yo(id)]`.
    type Id: Field + Asked;

    /// This document's id.
    fn id(&self) -> &Self::Id;
}

/// The indexes a type declares.
///
/// Written by `#[derive(Yo)]` for every type it is put on, whether or not the
/// type has an id, because an edge type declares indexes and has no id. That is
/// the whole reason this is a trait of its own rather than a constant on
/// [`Document`].
pub trait Indexed {
    /// The paths this type asks to be indexed, and how.
    const INDEXES: &'static [(&'static str, IndexKind)];
}

/// A path into a document, what its index can be asked, and the type of the
/// value that lives there.
///
/// Written by `#[derive(Yo)]` as a constant per indexed field, so a query names
/// the field rather than spelling a string the compiler cannot check. A field
/// marked `#[yo(ordered)]` gets an [`Ordered`] instead, which is the same thing
/// with ranges on it.
pub struct Path<T, V> {
    path: &'static str,
    kind: IndexKind,
    /// `fn() -> (T, V)` so the constant's auto traits do not come from what it
    /// points at, which lets it be a `const` in any type.
    marker: PhantomData<fn() -> (T, V)>,
}

impl<T, V> Clone for Path<T, V> {
    fn clone(&self) -> Path<T, V> {
        *self
    }
}

impl<T, V> Copy for Path<T, V> {}

impl<T, V> core::fmt::Debug for Path<T, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Path")
            .field("path", &self.path)
            .field("kind", &self.kind)
            .finish()
    }
}

impl<T, V> Path<T, V> {
    /// A path and what its index answers.
    ///
    /// The derive calls this. Calling it by hand is allowed and is how a path
    /// into a nested object is reached until the derive learns to follow one,
    /// but nothing checks that the collection has the index you named until the
    /// query runs.
    #[must_use]
    pub const fn new(path: &'static str, kind: IndexKind) -> Path<T, V> {
        Path {
            path,
            kind,
            marker: PhantomData,
        }
    }

    /// The path, as `$.status`.
    #[must_use]
    pub const fn path(&self) -> &'static str {
        self.path
    }

    /// What the index on this path can be asked.
    #[must_use]
    pub const fn kind(&self) -> IndexKind {
        self.kind
    }
}

/// A path whose index keeps its keys in order, so it answers ranges as well as
/// equality.
///
/// A separate type rather than a flag on [`Path`], because which questions a
/// path can answer is decided when the type is written and there is no reason
/// for the compiler not to know it. [`Docs::range`] takes one of these and
/// nothing else, so asking an equality index for a range is a type error at the
/// call site rather than a message at run time.
pub struct Ordered<T, V> {
    path: Path<T, V>,
}

impl<T, V> Clone for Ordered<T, V> {
    fn clone(&self) -> Ordered<T, V> {
        *self
    }
}

impl<T, V> Copy for Ordered<T, V> {}

impl<T, V> core::fmt::Debug for Ordered<T, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ordered")
            .field("path", &self.path.path)
            .finish()
    }
}

impl<T, V> Ordered<T, V> {
    /// A path whose index is ordered.
    ///
    /// The derive calls this.
    #[must_use]
    pub const fn new(path: &'static str) -> Ordered<T, V> {
        Ordered {
            path: Path::new(path, IndexKind::Ordered),
        }
    }

    /// The path, as `$.total`.
    #[must_use]
    pub const fn path(&self) -> &'static str {
        self.path.path
    }
}

/// An ordered path answers equality too, so everything that takes a [`Path`]
/// takes one of these.
impl<T, V> From<Ordered<T, V>> for Path<T, V> {
    fn from(o: Ordered<T, V>) -> Path<T, V> {
        o.path
    }
}

/// A collection of `T`.
///
/// Cheap to clone and cheap to keep around, the same way [`crate::Map`] is: the
/// handle is a pointer and an index, and every clone is the same collection.
pub struct Docs<T> {
    db: Handle,
    at: usize,
    tag: Tag,
    marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Docs<T> {
    fn clone(&self) -> Docs<T> {
        Docs {
            db: self.db.clone(),
            at: self.at,
            tag: self.tag,
            marker: PhantomData,
        }
    }
}

impl<T> core::fmt::Debug for Docs<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = self
            .db
            .read(|inner| Ok(inner.collections[self.at].name.clone()))
            .unwrap_or_else(|_| "?".to_owned());
        f.debug_struct("Docs").field("name", &name).finish()
    }
}

impl<T: Document> Docs<T> {
    pub(crate) fn new(db: Handle, at: usize, tag: Tag) -> Docs<T> {
        Docs {
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
        self.db
            .read(|inner| Ok(inner.collections[self.at].name.clone()))
    }

    /// This collection's shape tag.
    #[must_use]
    pub fn tag(&self) -> Tag {
        self.tag
    }

    /// Store a document, replacing whatever was under its id.
    ///
    /// Answers whether the id was new. Every index the type declares is brought
    /// up to date in the same call, and the old document is taken back out of
    /// them first, so an overwrite cannot leave a stale posting behind.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for an id that cannot be a key, and [`Code::Full`] for
    /// a value at an indexed path that is too long to be one.
    pub fn put(&self, doc: &T) -> Result<bool> {
        let id = key_of(doc.id(), IndexKind::Equality, "the id")?;
        self.write(|c| {
            c.scratch.clear();
            Field::write(doc, &mut c.scratch)?;
            let bytes = c.scratch.finish()?;
            c.docs.put_bytes(id.as_bytes(), bytes)
        })
    }

    /// Read a document by its id.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the stored document is not a `T`.
    pub fn get(&self, id: &<T::Id as Asked>::Ask) -> Result<Option<T>> {
        let id = key_of(id, IndexKind::Equality, "the id")?;
        self.read(|docs| match docs.get(id.as_bytes()) {
            Some(doc) => T::read(doc).map(Some),
            None => Ok(None),
        })
    }

    /// Whether an id is in the collection, without reading the document.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for an id that cannot be a key.
    pub fn contains(&self, id: &<T::Id as Asked>::Ask) -> Result<bool> {
        let id = key_of(id, IndexKind::Equality, "the id")?;
        self.read(|docs| Ok(docs.contains(id.as_bytes())))
    }

    /// Take a document out, answering whether it was there.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for an id that cannot be a key.
    pub fn remove(&self, id: &<T::Id as Asked>::Ask) -> Result<bool> {
        let id = key_of(id, IndexKind::Equality, "the id")?;
        self.write(|c| Ok(c.docs.remove(id.as_bytes())))
    }

    /// How many documents there are.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn len(&self) -> Result<usize> {
        self.read(|docs| Ok(docs.len()))
    }

    /// Whether the collection is empty.
    ///
    /// # Errors
    ///
    /// The same as [`Docs::len`].
    pub fn is_empty(&self) -> Result<bool> {
        self.read(|docs| Ok(docs.is_empty()))
    }

    /// Every document, in no particular order.
    ///
    /// A walk of the whole collection, which is what it says it is. The indexed
    /// calls are the ones with a cost model.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if any stored document is not a `T`.
    pub fn all(&self) -> Result<Vec<T>> {
        self.read(|docs| {
            let mut out = Vec::with_capacity(docs.len());
            for (_, doc) in docs.iter() {
                out.push(T::read(doc)?);
            }
            Ok(out)
        })
    }

    /// Every document whose value at `path` is `value`.
    ///
    /// One probe of the index and one probe of the primary table per document
    /// in the answer, so the cost is the size of the answer rather than the
    /// size of the collection.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the collection has no index on that path, because a
    /// query that quietly turns into a scan is the thing this API exists not to
    /// do.
    pub fn find<V: Asked>(&self, path: impl Into<Path<T, V>>, value: &V::Ask) -> Result<Vec<T>> {
        let path = path.into();
        let key = key_of(value, path.kind, path.path)?;
        self.read(|docs| {
            let mut out = Vec::new();
            let mut bad = Ok(());
            docs.find(path.path, &key, |_, doc| {
                if bad.is_ok() {
                    match T::read(doc) {
                        Ok(v) => out.push(v),
                        Err(e) => bad = Err(e),
                    }
                }
            })?;
            bad?;
            Ok(out)
        })
    }

    /// How many documents have `value` at `path`, without reading any of them.
    ///
    /// The number to sort filters by before intersecting them, and it is a
    /// probe rather than a walk.
    ///
    /// # Errors
    ///
    /// The same as [`Docs::find`].
    pub fn count<V: Asked>(&self, path: impl Into<Path<T, V>>, value: &V::Ask) -> Result<usize> {
        let path = path.into();
        let key = key_of(value, path.kind, path.path)?;
        self.read(|docs| docs.count(path.path, &key))
    }

    /// Every document whose value at `path` falls in `range`, smallest first.
    ///
    /// The bounds are the ordinary Rust range syntax, so `..`, `a..b`, `a..=b`
    /// and `..b` all work and mean what they say.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the collection has no index on that path. An index
    /// that answers equality only cannot get here at all, because [`Ordered`] is
    /// a different type from [`Path`] and this takes one of them.
    pub fn range<V: Asked, R: RangeBounds<V::Ask>>(
        &self,
        path: Ordered<T, V>,
        range: R,
    ) -> Result<Vec<T>> {
        let path = path.path();
        let (lo, hi) = bounds(&range, path)?;
        self.read(|docs| {
            let mut out = Vec::new();
            let mut bad = Ok(());
            docs.range(path, as_ref(&lo), as_ref(&hi), |_, doc| {
                if bad.is_ok() {
                    match T::read(doc) {
                        Ok(v) => out.push(v),
                        Err(e) => bad = Err(e),
                    }
                }
            })?;
            bad?;
            Ok(out)
        })
    }

    /// [`Docs::range`] backwards, largest value first.
    ///
    /// # Errors
    ///
    /// The same as [`Docs::range`].
    pub fn range_rev<V: Asked, R: RangeBounds<V::Ask>>(
        &self,
        path: Ordered<T, V>,
        range: R,
    ) -> Result<Vec<T>> {
        let path = path.path();
        let (lo, hi) = bounds(&range, path)?;
        self.read(|docs| {
            let mut out = Vec::new();
            let mut bad = Ok(());
            docs.range_rev(path, as_ref(&lo), as_ref(&hi), |_, doc| {
                if bad.is_ok() {
                    match T::read(doc) {
                        Ok(v) => out.push(v),
                        Err(e) => bad = Err(e),
                    }
                }
            })?;
            bad?;
            Ok(out)
        })
    }

    /// How many documents fall in `range` at `path`, without reading any.
    ///
    /// This reads the distinct values in the range rather than the documents,
    /// so a range covering a million documents under a hundred values costs a
    /// hundred.
    ///
    /// # Errors
    ///
    /// The same as [`Docs::range`].
    pub fn count_range<V: Asked, R: RangeBounds<V::Ask>>(
        &self,
        path: Ordered<T, V>,
        range: R,
    ) -> Result<usize> {
        let path = path.path();
        let (lo, hi) = bounds(&range, path)?;
        self.read(|docs| docs.count_range(path, as_ref(&lo), as_ref(&hi)))
    }

    /// What this collection is holding, documents and indexes together.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn memory_bytes(&self) -> Result<usize> {
        self.read(|docs| Ok(docs.memory_bytes()))
    }

    fn read<R>(&self, f: impl FnOnce(&yo_doc::Docs) -> Result<R>) -> Result<R> {
        self.db
            .read(|inner| f(inner.collections[self.at].data.docs()))
    }

    fn write<R>(&self, f: impl FnOnce(&mut Documents) -> Result<R>) -> Result<R> {
        self.db
            .write(|inner| f(inner.collections[self.at].data.docs_mut()))
    }
}

/// The documents of one collection, and the builder a write goes through.
///
/// The builder lives here rather than on the stack of [`Docs::put`] so that a
/// write reuses the buffer it filled last time and allocates nothing.
pub(crate) struct Documents {
    pub(crate) docs: yo_doc::Docs,
    pub(crate) scratch: Builder,
}

impl Documents {
    pub(crate) fn new() -> Documents {
        Documents {
            docs: yo_doc::Docs::new(),
            scratch: Builder::new(),
        }
    }
}

/// The key a value is filed under, or the sentence saying why it has none.
pub(crate) fn key_of<Q: Query + ?Sized>(value: &Q, kind: IndexKind, what: &str) -> Result<Key> {
    // The kind matters because a text index files words rather than whole
    // strings, so a query against one has to be folded the same way the write
    // was. Everything else asks its value for the key it always gives.
    let key = value.key(kind).ok_or_else(|| {
        let why = if kind == IndexKind::Text {
            "a text index holds one word at a time, and this is not one word"
        } else {
            "an index does not file this type, so it cannot be looked up"
        };
        Error::fmt(Code::Invalid, format_args!("{what}: {why}"))
    })?;
    if key.is_too_long() {
        return Err(Error::fmt(
            Code::Full,
            format_args!(
                "{what} is longer than {} bytes, which is as long as a key can be",
                yo_doc::KEY_MAX
            ),
        ));
    }
    Ok(key)
}

/// Turn a Rust range over the query type into the pair of key bounds the index
/// walks between.
fn bounds<Q, R>(range: &R, path: &str) -> Result<(Bound<Key>, Bound<Key>)>
where
    Q: Query + ?Sized,
    R: RangeBounds<Q>,
{
    Ok((
        one(range.start_bound(), path)?,
        one(range.end_bound(), path)?,
    ))
}

fn one<Q: Query + ?Sized>(b: Bound<&Q>, path: &str) -> Result<Bound<Key>> {
    Ok(match b {
        Bound::Included(v) => Bound::Included(key_of(v, IndexKind::Ordered, path)?),
        Bound::Excluded(v) => Bound::Excluded(key_of(v, IndexKind::Ordered, path)?),
        Bound::Unbounded => Bound::Unbounded,
    })
}

fn as_ref(b: &Bound<Key>) -> Bound<&Key> {
    match b {
        Bound::Included(k) => Bound::Included(k),
        Bound::Excluded(k) => Bound::Excluded(k),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// Read one field out of a document, which is what the derive calls per field.
///
/// # Errors
///
/// [`Code::Corrupt`] if the field is missing and its type is not an `Option`,
/// or if it is there and is the wrong type.
pub fn at<V: Field>(d: Doc<'_>, name: &str) -> Result<V> {
    match d.get(name.as_bytes()) {
        Some(at) => V::read(at),
        None => V::missing(name),
    }
}

/// Check that what is stored under this collection is an object at all, which
/// is what the derive calls before it reads the fields.
///
/// # Errors
///
/// [`Code::Corrupt`] for anything that is not an object.
pub fn expect_object(d: Doc<'_>, name: &str) -> Result<()> {
    if d.kind() == yo_doc::Kind::Object {
        return Ok(());
    }
    Err(Error::fmt(
        Code::Corrupt,
        format_args!("a {name} in this collection is stored as {:?}", d.kind()),
    ))
}

fn not_a(want: &str, d: Doc<'_>) -> Error {
    Error::fmt(
        Code::Corrupt,
        format_args!(
            "this field should be a {want} and is stored as {:?}",
            d.kind()
        ),
    )
}

macro_rules! ints {
    ($($t:ty),* $(,)?) => {
        $(
            impl Field for $t {
                fn write(&self, b: &mut Builder) -> Result<()> {
                    b.int(i64::from(*self))
                }

                fn read(d: Doc<'_>) -> Result<$t> {
                    let n = d.as_int().ok_or_else(|| not_a(stringify!($t), d))?;
                    <$t>::try_from(n).map_err(|_| {
                        Error::fmt(
                            Code::Corrupt,
                            format_args!("{n} does not fit in a {}", stringify!($t)),
                        )
                    })
                }
            }

            impl Query for $t {
                fn key(&self, _kind: IndexKind) -> Option<Key> {
                    Some(Key::int(i64::from(*self)))
                }
            }
        )*
    };
}

ints!(i8, i16, i32, i64, u8, u16, u32);

/// A `u64` is the one integer that does not fit, because JSON has one number
/// type and it is signed. Anything past `i64::MAX` is refused on the way in
/// rather than rounded through a float on the way out.
impl Field for u64 {
    fn write(&self, b: &mut Builder) -> Result<()> {
        match i64::try_from(*self) {
            Ok(n) => b.int(n),
            Err(_) => Err(Error::fmt(
                Code::Invalid,
                format_args!(
                    "{self} is past i64::MAX, and a document holds one number type, which is signed"
                ),
            )),
        }
    }

    fn read(d: Doc<'_>) -> Result<u64> {
        let n = d.as_int().ok_or_else(|| not_a("u64", d))?;
        u64::try_from(n).map_err(|_| {
            Error::fmt(
                Code::Corrupt,
                format_args!("{n} is negative and this field is a u64"),
            )
        })
    }
}

impl Query for u64 {
    fn key(&self, _kind: IndexKind) -> Option<Key> {
        i64::try_from(*self).ok().map(Key::int)
    }
}

macro_rules! floats {
    ($($t:ty),* $(,)?) => {
        $(
            impl Field for $t {
                fn write(&self, b: &mut Builder) -> Result<()> {
                    b.float(f64::from(*self))
                }

                fn read(d: Doc<'_>) -> Result<$t> {
                    // An integer reads back as a float, because a whole number
                    // written as a float is stored as an integer and refusing
                    // it here would make a round trip fail on 12.0.
                    match (d.as_float(), d.as_int()) {
                        (Some(v), _) => Ok(v as $t),
                        (None, Some(n)) => Ok(n as $t),
                        (None, None) => Err(not_a(stringify!($t), d)),
                    }
                }
            }

            impl Query for $t {
                fn key(&self, _kind: IndexKind) -> Option<Key> {
                    Some(Key::float(f64::from(*self)))
                }
            }
        )*
    };
}

floats!(f32, f64);

impl Field for bool {
    fn write(&self, b: &mut Builder) -> Result<()> {
        b.bool(*self)
    }

    fn read(d: Doc<'_>) -> Result<bool> {
        d.as_bool().ok_or_else(|| not_a("bool", d))
    }
}

impl Query for bool {
    fn key(&self, _kind: IndexKind) -> Option<Key> {
        Some(Key::bool(*self))
    }
}

impl Field for String {
    fn write(&self, b: &mut Builder) -> Result<()> {
        b.text(self)
    }

    fn read(d: Doc<'_>) -> Result<String> {
        d.as_text()
            .map(str::to_owned)
            .ok_or_else(|| not_a("string", d))
    }
}

impl Query for String {
    fn key(&self, kind: IndexKind) -> Option<Key> {
        self.as_str().key(kind)
    }
}

/// The borrowed form, so a `String` field is searched with a `&str`.
impl Query for str {
    fn key(&self, kind: IndexKind) -> Option<Key> {
        match kind {
            // A text index filed the words of the string, folded, so one word
            // is what can be asked for and a phrase is not a key at all.
            IndexKind::Text => Key::word(self),
            _ => Some(Key::text(self)),
        }
    }
}

/// A field that may be absent, which is the only type whose absence is not an
/// error. `None` is stored as null rather than left out, so a document always
/// has the fields its shape says it has.
impl<T: Field> Field for Option<T> {
    fn write(&self, b: &mut Builder) -> Result<()> {
        match self {
            Some(v) => v.write(b),
            None => b.null(),
        }
    }

    fn read(d: Doc<'_>) -> Result<Option<T>> {
        if d.is_null() {
            return Ok(None);
        }
        T::read(d).map(Some)
    }

    fn missing(_name: &str) -> Result<Option<T>> {
        Ok(None)
    }
}

impl<T: Field> Field for Vec<T> {
    fn write(&self, b: &mut Builder) -> Result<()> {
        b.begin_array()?;
        for v in self {
            v.write(b)?;
        }
        b.end_array()
    }

    fn read(d: Doc<'_>) -> Result<Vec<T>> {
        if d.kind() != yo_doc::Kind::Array {
            return Err(not_a("list", d));
        }
        let mut out = Vec::with_capacity(d.len());
        for elem in d.iter() {
            out.push(T::read(elem)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Yo, open};

    #[derive(Yo, Debug, Clone, PartialEq)]
    struct Order {
        #[yo(id)]
        id: u64,
        #[yo(index)]
        status: String,
        #[yo(ordered)]
        total: f64,
        #[yo(array)]
        tags: Vec<String>,
        #[yo(text)]
        note: String,
        sent: Option<String>,
    }

    fn order(id: u64, status: &str, total: f64) -> Order {
        Order {
            id,
            status: status.to_owned(),
            total,
            tags: Vec::new(),
            note: String::new(),
            sent: None,
        }
    }

    /// A collection holding the three orders most of these tests want.
    fn three() -> (crate::Db, Docs<Order>) {
        let db = open(crate::MEMORY).expect("a database in memory");
        let orders = db.docs::<Order>("orders").expect("a new collection");
        for o in [
            order(1, "open", 12.5),
            order(2, "shipped", 99.0),
            order(3, "open", 40.0),
        ] {
            orders.put(&o).expect("a document that fits");
        }
        (db, orders)
    }

    #[test]
    fn a_document_comes_back_as_the_struct_that_went_in() {
        let (_db, orders) = three();
        assert_eq!(
            orders.get(&1).expect("a read"),
            Some(order(1, "open", 12.5))
        );
        assert_eq!(orders.get(&9).expect("a read"), None);
        assert_eq!(orders.len().expect("a count"), 3);
        assert!(orders.contains(&2).expect("a read"));
    }

    #[test]
    fn every_field_kind_survives_the_round_trip() {
        let db = open(crate::MEMORY).expect("a database in memory");
        let orders = db.docs::<Order>("orders").expect("a new collection");
        let o = Order {
            id: 7,
            status: "open".to_owned(),
            total: -0.5,
            tags: vec!["red".to_owned(), "small".to_owned()],
            note: "A red kite".to_owned(),
            sent: Some("tuesday".to_owned()),
        };
        orders.put(&o).expect("a document that fits");
        assert_eq!(orders.get(&7).expect("a read"), Some(o));
    }

    #[test]
    fn putting_the_same_id_twice_replaces_it() {
        let (_db, orders) = three();
        assert!(!orders.put(&order(1, "shut", 1.0)).expect("a write"));
        assert_eq!(orders.len().expect("a count"), 3);
        assert_eq!(
            orders.get(&1).expect("a read").expect("it is there").status,
            "shut"
        );
        // And the old value is out of the index it was under.
        assert_eq!(orders.count(Order::STATUS, "open").expect("a count"), 1);
    }

    #[test]
    fn removing_a_document_takes_it_out_of_its_indexes() {
        let (_db, orders) = three();
        assert!(orders.remove(&1).expect("a write"));
        assert!(!orders.remove(&1).expect("a write"));
        assert_eq!(orders.len().expect("a count"), 2);
        assert_eq!(orders.count(Order::STATUS, "open").expect("a count"), 1);
        assert!(orders.find(Order::TOTAL, &12.5).expect("a read").is_empty());
    }

    #[test]
    fn an_equality_index_answers_with_the_documents() {
        let (_db, orders) = three();
        let mut open = orders.find(Order::STATUS, "open").expect("a read");
        open.sort_by_key(|o| o.id);
        assert_eq!(open, [order(1, "open", 12.5), order(3, "open", 40.0)]);
        assert_eq!(orders.count(Order::STATUS, "gone").expect("a count"), 0);
    }

    /// This is the query that was wrong before the numeric key encoding was
    /// fixed: 12.5 sorted after 99.0, so the range came back empty and the
    /// reverse walk came back ascending.
    #[test]
    fn a_range_over_a_float_field_is_in_numeric_order() {
        let (_db, orders) = three();
        let cheap = orders.range(Order::TOTAL, 0.0..50.0).expect("a read");
        assert_eq!(
            cheap.iter().map(|o| o.total).collect::<Vec<_>>(),
            [12.5, 40.0]
        );

        let all = orders.range(Order::TOTAL, ..).expect("a read");
        assert_eq!(
            all.iter().map(|o| o.total).collect::<Vec<_>>(),
            [12.5, 40.0, 99.0]
        );

        let down = orders.range_rev(Order::TOTAL, ..).expect("a read");
        assert_eq!(
            down.iter().map(|o| o.total).collect::<Vec<_>>(),
            [99.0, 40.0, 12.5]
        );

        assert_eq!(
            orders
                .count_range(Order::TOTAL, 12.5..=40.0)
                .expect("a count"),
            2
        );
    }

    #[test]
    fn an_ordered_path_can_still_be_asked_for_equality() {
        let (_db, orders) = three();
        assert_eq!(orders.find(Order::TOTAL, &40.0).expect("a read").len(), 1);
        assert_eq!(orders.count(Order::TOTAL, &99.0).expect("a count"), 1);
    }

    #[test]
    fn a_range_over_a_string_field_takes_a_pair_of_bounds() {
        let db = open(crate::MEMORY).expect("a database in memory");
        let names = db.docs::<Named>("names").expect("a new collection");
        for (id, name) in [(1u64, "banana"), (2, "apple"), (3, "quince")] {
            names
                .put(&Named {
                    id,
                    name: name.to_owned(),
                })
                .expect("a document that fits");
        }
        let early = names
            .range(Named::NAME, (Bound::Included("a"), Bound::Excluded("m")))
            .expect("a read");
        assert_eq!(
            early.iter().map(|n| n.name.as_str()).collect::<Vec<_>>(),
            ["apple", "banana"]
        );
    }

    #[derive(Yo, Debug, PartialEq)]
    struct Named {
        #[yo(id)]
        id: u64,
        #[yo(ordered)]
        name: String,
    }

    #[test]
    fn an_array_index_files_a_document_under_every_element() {
        let db = open(crate::MEMORY).expect("a database in memory");
        let orders = db.docs::<Order>("orders").expect("a new collection");
        let mut o = order(1, "open", 1.0);
        o.tags = vec!["red".to_owned(), "small".to_owned()];
        orders.put(&o).expect("a document that fits");

        assert_eq!(orders.find(Order::TAGS, "red").expect("a read").len(), 1);
        assert_eq!(orders.find(Order::TAGS, "small").expect("a read").len(), 1);
        assert_eq!(orders.count(Order::TAGS, "large").expect("a count"), 0);
    }

    #[test]
    fn a_text_index_files_a_document_under_every_word() {
        let db = open(crate::MEMORY).expect("a database in memory");
        let orders = db.docs::<Order>("orders").expect("a new collection");
        let mut o = order(1, "open", 1.0);
        o.note = "A red kite".to_owned();
        orders.put(&o).expect("a document that fits");

        // The case is folded on both sides, so the query does not have to match
        // how the document happened to be written.
        assert_eq!(orders.find(Order::NOTE, "RED").expect("a read").len(), 1);
        assert_eq!(orders.find(Order::NOTE, "kite").expect("a read").len(), 1);
        assert_eq!(orders.count(Order::NOTE, "blue").expect("a count"), 0);
    }

    #[test]
    fn asking_a_text_index_for_a_phrase_says_so() {
        let (_db, orders) = three();
        let e = orders
            .find(Order::NOTE, "red kite")
            .expect_err("not one word");
        assert_eq!(e.code(), crate::Code::Invalid);
        assert!(e.message().contains("one word"), "{}", e.message());
    }

    #[test]
    fn an_absent_field_reads_back_as_none() {
        let (_db, orders) = three();
        assert_eq!(
            orders.get(&1).expect("a read").expect("it is there").sent,
            None
        );
    }

    #[test]
    fn a_nested_struct_is_a_field() {
        #[derive(Yo, Debug, PartialEq)]
        struct Where {
            city: String,
            postcode: String,
        }

        #[derive(Yo, Debug, PartialEq)]
        struct Person {
            #[yo(id)]
            id: u64,
            home: Where,
        }

        let db = open(crate::MEMORY).expect("a database in memory");
        let people = db.docs::<Person>("people").expect("a new collection");
        let p = Person {
            id: 1,
            home: Where {
                city: "Hanoi".to_owned(),
                postcode: "100000".to_owned(),
            },
        };
        people.put(&p).expect("a document that fits");
        assert_eq!(people.get(&1).expect("a read"), Some(p));
    }

    #[test]
    fn all_walks_every_document() {
        let (_db, orders) = three();
        let mut ids: Vec<u64> = orders.all().expect("a read").iter().map(|o| o.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, [1, 2, 3]);
    }

    #[test]
    fn opening_a_collection_as_the_wrong_thing_is_refused() {
        let db = open(crate::MEMORY).expect("a database in memory");
        let _orders = db.docs::<Order>("orders").expect("a new collection");
        let e = db
            .map::<String, u64>("orders")
            .expect_err("a different shape");
        assert_eq!(e.code(), crate::Code::ShapeMismatch);
        let e = db.docs::<Named>("orders").expect_err("a different struct");
        assert_eq!(e.code(), crate::Code::ShapeMismatch);
    }

    #[test]
    fn reopening_a_collection_hands_back_the_same_documents() {
        let (db, orders) = three();
        let again = db.docs::<Order>("orders").expect("the same collection");
        assert_eq!(again.len().expect("a count"), 3);
        assert_eq!(again.count(Order::STATUS, "open").expect("a count"), 2);
        drop(orders);
    }

    #[test]
    fn a_u64_past_what_json_can_hold_is_refused() {
        let db = open(crate::MEMORY).expect("a database in memory");
        let orders = db.docs::<Order>("orders").expect("a new collection");
        let e = orders
            .put(&order(u64::MAX, "open", 1.0))
            .expect_err("too big");
        assert_eq!(e.code(), crate::Code::Invalid);
    }
}
