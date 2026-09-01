//! A document collection: the primary table and the key table that goes with
//! it (`09` section 4).
//!
//! The primary table is an element table keyed by document id with the
//! document's bytes stored behind its id in the same blob. That is not a family
//! resemblance to a hash's field table, it is the same code: [`Elements`] in
//! tailed mode, which is what `HSET` writes into. A document collection and a
//! hash differ in what the bytes behind the name mean and in nothing else, and
//! that is the point of R25.
//!
//! ```
//! use yo_doc::{Builder, Docs};
//!
//! let mut b = Builder::new();
//! b.begin_object()?;
//! b.key(b"customer")?;
//! b.int(7)?;
//! b.key(b"status")?;
//! b.text("open")?;
//! b.end_object()?;
//! let order = b.finish()?.to_vec();
//!
//! let mut docs = Docs::new();
//! assert!(docs.put_bytes(b"order:1", &order)?);
//! let got = docs.get(b"order:1").expect("stored");
//! assert_eq!(got.get(b"status").and_then(|v| v.as_text()), Some("open"));
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # What a write does
//!
//! [`Docs::put`] does not store the value it is given. It walks it once and
//! writes it again with every object key replaced by its id from the
//! collection's [`Keys`], which is where the forty percent that interning is
//! worth actually gets saved.
//!
//! If the key table fills part way through, the document is stored as it
//! arrived with its keys as bytes. Nothing about that is a fallback mode: the
//! interned flag sits in each container's header, so a collection holds both
//! kinds at once, a reader tells them apart per container, and documents
//! written before the table filled stay exactly as they were.
//!
//! # What a write does about the indexes
//!
//! A write takes one path lookup per declared index, not a comparison against
//! every index path at every node of the document. I had it the other way round
//! at first, on the argument that the interning walk is already touching every
//! node so the extraction may as well ride along on it. That is worse: with N
//! indexes it costs N comparisons at every node, where a lookup per index costs
//! N times the two or three binary searches a shallow path takes, and index
//! paths are shallow. It is also much simpler, and it is the same code the
//! backfill in [`Docs::create_index`] runs.
//!
//! The keys are worked out before anything is stored, so a value that is too
//! long to be an index key fails the write rather than leaving behind a document
//! that is in the collection and in none of its indexes. Then the old document
//! under that id is taken out of the indexes, then the new one is stored, then
//! it is filed. An overwrite and a removal both un-index through the same code,
//! because both make the old entries wrong.
//!
//! # Reading one back
//!
//! [`Docs::get`] answers a [`Doc`], which is a [`Value`] with the collection's
//! key table beside it. Everything that needs a name rather than an id goes
//! through the table: `get(b"status")` resolves the name to an id once and then
//! searches the document by id, which is a binary search over integers.

use core::ops::Bound;

use yo_common::{Code, Error, Result};
use yo_kv::{Cursor, Elements, Full};

use crate::head::{DEPTH_MAX, Kind};
use crate::index::{self, IndexKind, Key, PathIndex};
use crate::path::{Step, Steps};
use crate::{Builder, Keys, Value};

/// Documents by id, with the key table their keys are interned against.
#[derive(Debug)]
pub struct Docs {
    /// Document id to the document's bytes, the bytes behind the id.
    rows: Elements<()>,
    /// The names every interned object in this collection uses.
    keys: Keys,
    /// The buffer a write is re-encoded into, kept so a write does not allocate.
    build: Builder,
    /// One per indexed path, in the order they were declared.
    indexes: Vec<PathIndex>,
    /// The key each index takes from the document being written, one slot per
    /// index and empty where the document has nothing to file.
    ///
    /// Worked out before anything is stored, so a value that cannot be indexed
    /// fails the write rather than leaving a document behind that no query will
    /// ever find. Kept on the collection so a write allocates nothing.
    taken: Vec<Vec<u8>>,
}

impl Default for Docs {
    /// Not derived, because the primary table has to be the kind that keeps a
    /// document behind its id and an empty [`Elements`] is not.
    fn default() -> Docs {
        Docs::new()
    }
}

impl Docs {
    /// An empty collection that has not allocated anything yet.
    #[must_use]
    pub fn new() -> Docs {
        Docs {
            rows: Elements::tailed(0, 0),
            keys: Keys::new(),
            build: Builder::new(),
            indexes: Vec::new(),
            taken: Vec::new(),
        }
    }

    /// An empty collection with room for `n` documents of about `each` bytes.
    ///
    /// The ids and the documents share one blob, so the size asked for is the
    /// two of them together. Getting it wrong costs a growth, not a rewrite.
    #[must_use]
    pub fn with_capacity(n: usize, each: usize) -> Docs {
        Docs {
            rows: Elements::tailed(n, n.saturating_mul(each)),
            keys: Keys::new(),
            build: Builder::with_capacity(each),
            indexes: Vec::new(),
            taken: Vec::new(),
        }
    }

    /// Store `value` under `id`, and say whether the id is new.
    ///
    /// The value is re-encoded with this collection's interned keys on the way
    /// in. It may not already be interned: a document whose keys are ids
    /// belongs to whichever collection handed those ids out, and moving it to
    /// another one without the names is how a collection ends up reading the
    /// wrong field.
    pub fn put(&mut self, id: &[u8], value: Value<'_>) -> Result<bool> {
        self.write(id, value, None)
    }

    /// Store the document `doc` encodes under `id`, and say whether the id is
    /// new.
    ///
    /// The bytes are checked far enough to be readable and no further, the same
    /// as [`Value::new`]. A caller holding bytes it did not write should run
    /// [`Value::validate`] first.
    pub fn put_bytes(&mut self, id: &[u8], doc: &[u8]) -> Result<bool> {
        let value = Value::new(doc)
            .ok_or_else(|| Error::new(Code::Corrupt, "the document is not a readable value"))?;
        self.write(id, value, Some(doc))
    }

    /// The write both forms of put go through.
    ///
    /// `raw` is the caller's bytes when it had some, so that the path where the
    /// key table is full stores them directly instead of copying them through
    /// the builder to get back what it was already holding.
    ///
    /// The order is: work out the index keys, un-index whatever was under this
    /// id, store, index. Working the keys out first is what makes a write that
    /// cannot be indexed leave the collection exactly as it was, rather than
    /// storing a document that no query will find.
    fn write(&mut self, id: &[u8], value: Value<'_>, raw: Option<&[u8]>) -> Result<bool> {
        let Docs {
            rows,
            keys,
            build,
            indexes,
            taken,
        } = self;

        taken.resize(indexes.len(), Vec::new());
        for (slot, index) in taken.iter_mut().zip(indexes.iter()) {
            slot.clear();
            // The incoming value has its keys as bytes, since put refuses one
            // that does not, so its paths resolve without the key table.
            let Some(at) = value.path_bytes(index.path())? else {
                continue;
            };
            let Some(key) = Key::of(at) else {
                continue;
            };
            if key.is_too_long() {
                return Err(Error::fmt(
                    Code::Full,
                    format_args!(
                        "the value at {} is longer than {} bytes and cannot be indexed",
                        String::from_utf8_lossy(index.path()),
                        index::KEY_MAX
                    ),
                ));
            }
            slot.extend_from_slice(key.as_bytes());
        }

        unindex(rows, keys, indexes, id);

        build.clear();
        let fresh = if intern_into(keys, build, value, 0)? {
            store(rows, id, build.finish()?)?
        } else if let Some(raw) = raw {
            // The key table filled part way through, and the caller is holding
            // exactly what should be stored.
            store(rows, id, raw)?
        } else {
            build.clear();
            build.embed(&value)?;
            store(rows, id, build.finish()?)?
        };

        for (slot, index) in taken.iter().zip(indexes.iter_mut()) {
            if !slot.is_empty() {
                index.add(slot, id)?;
            }
        }
        Ok(fresh)
    }

    /// Start indexing `path` for equality, and file every document already here
    /// under it.
    ///
    /// Declaring the same path twice is not an error and does not rebuild
    /// anything, because a caller that opens a collection and declares its
    /// indexes on the way in should be able to do that every time it opens it.
    /// An ordered index that is already there stays ordered, since it answers
    /// equality as well.
    ///
    /// The backfill is a path lookup per document, so it costs the collection
    /// once. There is no background indexer and no window in which the index is
    /// declared and not yet true, which is Y3.
    pub fn create_index(&mut self, path: &str) -> Result<()> {
        self.create_index_bytes(path.as_bytes(), IndexKind::Equality)
    }

    /// Start indexing `path` for equality and for ranges.
    ///
    /// An ordered index is an equality index with a counted B+ tree over the
    /// rows of its key table, which is the same tree a sorted set ranks with.
    /// It costs about three bytes per distinct value on top of the equality
    /// index and a logarithmic search per new value, and it is what
    /// [`Docs::range`] needs.
    ///
    /// A path that is already indexed for equality is upgraded and rebuilt. The
    /// alternative is answering `Ok` and then having every range on it come back
    /// empty, which is a query that lies.
    pub fn create_ordered_index(&mut self, path: &str) -> Result<()> {
        self.create_index_bytes(path.as_bytes(), IndexKind::Ordered)
    }

    /// [`Docs::create_index`] and [`Docs::create_ordered_index`] for a path that
    /// is already bytes.
    pub fn create_index_bytes(&mut self, path: &[u8], kind: IndexKind) -> Result<()> {
        for step in Steps::new(path) {
            step?;
        }
        match self.indexes.iter().position(|i| i.path() == path) {
            // Equality on top of ordered is already answered, and equality on
            // top of equality is nothing at all.
            Some(_) if kind == IndexKind::Equality => return Ok(()),
            Some(at) if self.indexes[at].kind() == IndexKind::Ordered => return Ok(()),
            Some(at) => {
                self.indexes.remove(at);
                self.taken.truncate(self.indexes.len());
            }
            None => {}
        }
        let mut index = PathIndex::new(path, kind);
        for (id, bytes) in self.rows.pairs() {
            let Some(value) = Value::new(bytes) else {
                continue;
            };
            let doc = Doc {
                value,
                keys: &self.keys,
            };
            let Some(at) = doc.path_bytes(path)? else {
                continue;
            };
            let Some(key) = Key::of(at.value()) else {
                continue;
            };
            if key.is_too_long() {
                return Err(Error::fmt(
                    Code::Full,
                    format_args!(
                        "the value at {} in {} is longer than {} bytes and cannot be indexed",
                        String::from_utf8_lossy(path),
                        String::from_utf8_lossy(id),
                        index::KEY_MAX
                    ),
                ));
            }
            index.add(key.as_bytes(), id)?;
        }
        self.indexes.push(index);
        self.taken.push(Vec::new());
        Ok(())
    }

    /// Stop indexing `path`, and say whether it was indexed.
    pub fn drop_index(&mut self, path: &str) -> bool {
        self.drop_index_bytes(path.as_bytes())
    }

    /// [`Docs::drop_index`] for a path that is already bytes.
    pub fn drop_index_bytes(&mut self, path: &[u8]) -> bool {
        let Some(at) = self.indexes.iter().position(|i| i.path() == path) else {
            return false;
        };
        self.indexes.remove(at);
        self.taken.truncate(self.indexes.len());
        true
    }

    /// The indexes this collection keeps, in the order they were declared.
    #[must_use]
    pub fn indexes(&self) -> &[PathIndex] {
        &self.indexes
    }

    /// The index on `path`, if there is one.
    #[must_use]
    pub fn index(&self, path: &str) -> Option<&PathIndex> {
        self.indexes.iter().find(|i| i.path() == path.as_bytes())
    }

    /// Hand every document whose value at `path` is `key` to `f`, and say how
    /// many there were.
    ///
    /// One probe of the index and one probe of the primary table per document,
    /// which is the cost model `09` section 5 states rather than hides. A path
    /// with no index on it is an error and not a scan: a query that silently
    /// turns into a walk of the collection is the thing this API exists not to
    /// do.
    pub fn find(&self, path: &str, key: &Key, mut f: impl FnMut(&[u8], Doc<'_>)) -> Result<usize> {
        let index = self.index(path).ok_or_else(|| {
            Error::fmt(
                Code::Invalid,
                format_args!("there is no index on {path}, so this would be a scan"),
            )
        })?;
        let Some(set) = index.get(key) else {
            return Ok(0);
        };
        let mut n = 0usize;
        index::each_id(set, |id| {
            if let Some(doc) = self.get(id) {
                f(id, doc);
                n += 1;
            }
        });
        Ok(n)
    }

    /// How many documents have `key` at `path`, without reading any of them.
    ///
    /// The number a caller sorts its filters by before it intersects them, and
    /// it is a probe rather than a walk.
    pub fn count(&self, path: &str, key: &Key) -> Result<usize> {
        let index = self.index(path).ok_or_else(|| {
            Error::fmt(
                Code::Invalid,
                format_args!("there is no index on {path}, so this would be a scan"),
            )
        })?;
        Ok(index.count(key))
    }

    /// Hand every document whose value at `path` falls between `lo` and `hi` to
    /// `f`, smallest first, and say how many there were.
    ///
    /// One search of the tree and then a walk, so the cost is the size of the
    /// answer and not the size of the collection. The bounds are the ordinary
    /// [`Bound`], so a half open range, a range open at one end and a range open
    /// at both are all the same call.
    ///
    /// The path has to carry an ordered index. An equality index has no order to
    /// walk, and answering nothing would be a query that lies rather than a
    /// query that says no.
    pub fn range(
        &self,
        path: &str,
        lo: Bound<&Key>,
        hi: Bound<&Key>,
        mut f: impl FnMut(&[u8], Doc<'_>),
    ) -> Result<usize> {
        let index = self.ordered(path)?;
        let mut n = 0usize;
        for (_, set) in index.range(lo, hi) {
            index::each_id(set, |id| {
                if let Some(doc) = self.get(id) {
                    f(id, doc);
                    n += 1;
                }
            });
        }
        Ok(n)
    }

    /// [`Docs::range`] backwards, largest value first.
    pub fn range_rev(
        &self,
        path: &str,
        lo: Bound<&Key>,
        hi: Bound<&Key>,
        mut f: impl FnMut(&[u8], Doc<'_>),
    ) -> Result<usize> {
        let index = self.ordered(path)?;
        let mut n = 0usize;
        for (_, set) in index.range_rev(lo, hi) {
            index::each_id(set, |id| {
                if let Some(doc) = self.get(id) {
                    f(id, doc);
                    n += 1;
                }
            });
        }
        Ok(n)
    }

    /// How many documents fall between `lo` and `hi` at `path`, without reading
    /// any of them.
    ///
    /// This reads the distinct values in the range rather than the documents, so
    /// a range covering a million documents under a hundred values costs a
    /// hundred.
    pub fn count_range(&self, path: &str, lo: Bound<&Key>, hi: Bound<&Key>) -> Result<usize> {
        Ok(self.ordered(path)?.count_in(lo, hi))
    }

    /// The ordered index on `path`, or the error that says why there is not one.
    fn ordered(&self, path: &str) -> Result<&PathIndex> {
        match self.index(path) {
            Some(index) if index.kind() == IndexKind::Ordered => Ok(index),
            Some(_) => Err(Error::fmt(
                Code::Invalid,
                format_args!("the index on {path} answers equality and not ranges"),
            )),
            None => Err(Error::fmt(
                Code::Invalid,
                format_args!("there is no index on {path}, so this would be a scan"),
            )),
        }
    }

    /// The document stored under `id`.
    #[must_use]
    pub fn get(&self, id: &[u8]) -> Option<Doc<'_>> {
        let value = Value::new(self.rows.tail(id)?)?;
        Some(Doc {
            value,
            keys: &self.keys,
        })
    }

    /// The stored bytes of the document under `id`, as they sit in the blob.
    ///
    /// For a caller that is going to write them somewhere else rather than read
    /// them, which is `DUMP`, replication and the record plane.
    #[must_use]
    pub fn bytes(&self, id: &[u8]) -> Option<&[u8]> {
        self.rows.tail(id)
    }

    /// Whether there is a document under `id`.
    #[must_use]
    pub fn contains(&self, id: &[u8]) -> bool {
        self.rows.contains(id)
    }

    /// Take the document under `id` out, and say whether there was one.
    ///
    /// Every index the document was filed in loses it first, so a removal costs
    /// a path lookup per index on the way out.
    ///
    /// The key table is left alone. A name it interned stays interned even if
    /// this was the last document using it, which is [`Keys`]'s rule and the
    /// reason an id is a row index.
    pub fn remove(&mut self, id: &[u8]) -> bool {
        let Docs {
            rows,
            keys,
            indexes,
            ..
        } = self;
        unindex(rows, keys, indexes, id);
        rows.remove(id).is_some()
    }

    /// How many documents there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the collection holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The names this collection has interned.
    #[must_use]
    pub fn keys(&self) -> &Keys {
        &self.keys
    }

    /// Every document, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], Doc<'_>)> {
        let keys = &self.keys;
        self.rows.pairs().filter_map(move |(id, bytes)| {
            let value = Value::new(bytes)?;
            Some((id, Doc { value, keys }))
        })
    }

    /// Walk part of the collection and say where to resume, the same contract
    /// [`Elements::scan`] has.
    pub fn scan<F>(&self, cursor: Cursor, count: usize, mut f: F) -> Cursor
    where
        F: FnMut(&[u8], Doc<'_>),
    {
        let keys = &self.keys;
        self.rows.scan_pairs(cursor, count, |id, bytes| {
            if let Some(value) = Value::new(bytes) {
                f(id, Doc { value, keys });
            }
        })
    }

    /// Throw every document away and keep the key table and the allocations.
    ///
    /// The indexes stay declared and go empty, for the same reason the key table
    /// stays: a caller that empties a collection is refilling it, and an index
    /// that quietly disappeared when the last document did would turn the next
    /// query into an error.
    ///
    /// The key table stays because a collection that is emptied is usually a
    /// collection that is about to be refilled with the same shape of document,
    /// and relearning twenty names is work with nothing to show for it.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.build.clear();
        for index in &mut self.indexes {
            index.clear();
        }
    }

    /// What the collection costs, the key table and the indexes included.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.rows.memory_bytes()
            + self.keys.memory_bytes()
            + self
                .indexes
                .iter()
                .map(PathIndex::memory_bytes)
                .sum::<usize>()
    }
}

/// Take whatever is stored under `id` out of every index, leaving the primary
/// table alone.
///
/// Both an overwrite and a removal go through here, because both of them make
/// the old document's index entries wrong and neither of them can work out what
/// those entries were once the bytes are gone. Nothing here can fail: a document
/// that is no longer readable, or a path that no longer resolves, simply has
/// nothing filed under it, and refusing a removal because the thing being
/// removed is damaged is the wrong answer.
fn unindex(rows: &Elements<()>, keys: &Keys, indexes: &mut [PathIndex], id: &[u8]) {
    if indexes.is_empty() {
        return;
    }
    let Some(bytes) = rows.tail(id) else {
        return;
    };
    let Some(value) = Value::new(bytes) else {
        return;
    };
    let doc = Doc { value, keys };
    for index in indexes {
        let Ok(Some(at)) = doc.path_bytes(index.path()) else {
            continue;
        };
        let Some(key) = Key::of(at.value()) else {
            continue;
        };
        index.take(key.as_bytes(), id);
    }
}

/// Put `bytes` in the primary table under `id`, turning a refusal into the
/// error the layer above would have written anyway.
fn store(rows: &mut Elements<()>, id: &[u8], bytes: &[u8]) -> Result<bool> {
    match rows.set_tailed(id, bytes, ()) {
        Ok((_, fresh)) => Ok(fresh),
        Err(Full::Name) => Err(Error::fmt(
            Code::Full,
            format_args!("a document id is at most {} bytes", yo_kv::NAME_MAX),
        )),
        Err(Full::Rows) => Err(Error::fmt(
            Code::Full,
            format_args!("a collection holds at most {} documents", yo_kv::MAX_ROWS),
        )),
    }
}

/// Write `value` into `b` with every object key replaced by its id.
///
/// `Ok(false)` means the key table ran out of ids part way through, and the
/// caller stores the document with its keys as bytes instead. The builder is
/// left half open in that case, so the caller clears it.
///
/// The recursion is bounded by the builder: it refuses to open a container more
/// than [`DEPTH_MAX`] deep, so a document nested deeper than that, which only a
/// damaged one can be, stops with an error rather than with the stack.
fn intern_into(keys: &mut Keys, b: &mut Builder, value: Value<'_>, depth: usize) -> Result<bool> {
    let corrupt = || Error::new(Code::Corrupt, "the document is not readable at that point");
    match value.kind() {
        Kind::Null => b.null()?,
        Kind::Bool => b.bool(value.as_bool().ok_or_else(corrupt)?)?,
        Kind::Int => b.int(value.as_int().ok_or_else(corrupt)?)?,
        Kind::Float => b.float(value.as_float().ok_or_else(corrupt)?)?,
        Kind::Text => b.text_bytes(value.text_bytes().ok_or_else(corrupt)?)?,
        Kind::Array => {
            b.begin_array()?;
            for i in 0..value.len() {
                let child = value.at(i).ok_or_else(corrupt)?;
                if !intern_into(keys, b, child, depth + 1)? {
                    return Ok(false);
                }
            }
            b.end_array()?;
        }
        Kind::Object => {
            if value.is_interned() {
                return Err(Error::new(
                    Code::Invalid,
                    "this document's keys are ids from another collection's key table",
                ));
            }
            b.begin_object_interned()?;
            for i in 0..value.len() {
                let name = value.key_at(i).ok_or_else(corrupt)?;
                let Some(id) = keys.intern(name) else {
                    return Ok(false);
                };
                b.key_id(id)?;
                let child = value.at(i).ok_or_else(corrupt)?;
                if !intern_into(keys, b, child, depth + 1)? {
                    return Ok(false);
                }
            }
            b.end_object()?;
        }
    }
    debug_assert!(depth <= DEPTH_MAX, "the builder caps the depth");
    Ok(true)
}

/// A value with the key table its keys are interned against.
///
/// Everything a [`Value`] offers is here too, and the things that need a name
/// rather than an id, which are a lookup, a walk over the members and printing
/// the thing, go through the table. A document whose keys are bytes works the
/// same way and simply never asks the table anything, so a caller does not have
/// to know which kind it is holding.
#[derive(Clone, Copy)]
pub struct Doc<'a> {
    value: Value<'a>,
    keys: &'a Keys,
}

impl<'a> Doc<'a> {
    /// A view of `value` against `keys`.
    #[must_use]
    pub fn new(value: Value<'a>, keys: &'a Keys) -> Doc<'a> {
        Doc { value, keys }
    }

    /// The value underneath, for the accessors that never need a name.
    #[must_use]
    pub fn value(&self) -> Value<'a> {
        self.value
    }

    /// The key table this reads names out of.
    #[must_use]
    pub fn keys(&self) -> &'a Keys {
        self.keys
    }

    /// What this value is.
    #[must_use]
    pub fn kind(&self) -> Kind {
        self.value.kind()
    }

    /// Whether this is `null`.
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    /// The boolean this holds, if it holds one.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        self.value.as_bool()
    }

    /// The integer this holds, if it holds one.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        self.value.as_int()
    }

    /// The float this holds, if it holds one.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        self.value.as_float()
    }

    /// The string this holds, if it holds one and it is UTF-8.
    #[must_use]
    pub fn as_text(&self) -> Option<&'a str> {
        self.value.as_text()
    }

    /// The string this holds as it is stored, without the UTF-8 check.
    #[must_use]
    pub fn text_bytes(&self) -> Option<&'a [u8]> {
        self.value.text_bytes()
    }

    /// How many elements a container holds. Zero for anything else.
    #[must_use]
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Whether this is a container with nothing in it.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// The value stored under `key`.
    ///
    /// For an interned object this is a name to id lookup in the table and then
    /// a binary search over integers. A name the table has never seen cannot be
    /// in the document, so it answers `None` without touching the document at
    /// all.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<Doc<'a>> {
        let value = if self.value.is_interned() {
            self.value.get_id(self.keys.id(key)?)?
        } else {
            self.value.get(key)?
        };
        Some(Doc {
            value,
            keys: self.keys,
        })
    }

    /// Element `i` of a container, in the container's own order.
    #[must_use]
    pub fn at(&self, i: usize) -> Option<Doc<'a>> {
        Some(Doc {
            value: self.value.at(i)?,
            keys: self.keys,
        })
    }

    /// The name of member `i` of an object, whichever way the keys are stored.
    #[must_use]
    pub fn key_at(&self, i: usize) -> Option<&'a [u8]> {
        if self.value.is_interned() {
            self.keys.name(self.value.key_id_at(i)?)
        } else {
            self.value.key_at(i)
        }
    }

    /// Every member of an object, name first, in the order the document stores
    /// them.
    ///
    /// That order is by key id for an interned object and by key bytes for one
    /// whose keys are bytes, so it is stable for a given collection and it is
    /// not alphabetical. Sort it if the order is part of the answer.
    #[must_use]
    pub fn members(&self) -> DocMembers<'a> {
        DocMembers { d: *self, i: 0 }
    }

    /// Every element of a container, in the container's own order.
    #[must_use]
    pub fn iter(&self) -> DocElems<'a> {
        DocElems { d: *self, i: 0 }
    }

    /// The value at `path`, where a path names exactly one place.
    ///
    /// The same grammar [`Value::path`] takes, with the names resolved through
    /// the key table on the way down.
    pub fn path(&self, path: &str) -> Result<Option<Doc<'a>>> {
        self.path_bytes(path.as_bytes())
    }

    /// [`Doc::path`] for a path that is already bytes.
    pub fn path_bytes(&self, path: &[u8]) -> Result<Option<Doc<'a>>> {
        let mut at = *self;
        for step in Steps::new(path) {
            let next = match step? {
                Step::Key(k) => at.get(k),
                Step::Index(_) if at.kind() != Kind::Array => None,
                Step::Index(i) => {
                    let n = at.len();
                    let i = if i < 0 {
                        match n.checked_sub(i.unsigned_abs() as usize) {
                            Some(i) => i,
                            None => return Ok(None),
                        }
                    } else {
                        i as usize
                    };
                    at.at(i)
                }
            };
            let Some(next) = next else {
                return Ok(None);
            };
            at = next;
        }
        Ok(Some(at))
    }
}

impl core::fmt::Debug for Doc<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind() {
            Kind::Object => {
                let mut m = f.debug_map();
                for (k, v) in self.members() {
                    m.entry(&String::from_utf8_lossy(k), &v);
                }
                m.finish()
            }
            Kind::Array => f.debug_list().entries(self.iter()).finish(),
            _ => self.value.fmt(f),
        }
    }
}

/// Every member of an object, from [`Doc::members`].
#[derive(Clone)]
pub struct DocMembers<'a> {
    d: Doc<'a>,
    i: usize,
}

impl<'a> Iterator for DocMembers<'a> {
    type Item = (&'a [u8], Doc<'a>);

    fn next(&mut self) -> Option<(&'a [u8], Doc<'a>)> {
        let key = self.d.key_at(self.i)?;
        let val = self.d.at(self.i)?;
        self.i += 1;
        Some((key, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.d.len().saturating_sub(self.i);
        (left, Some(left))
    }
}

/// Every element of a container, from [`Doc::iter`].
#[derive(Clone)]
pub struct DocElems<'a> {
    d: Doc<'a>,
    i: usize,
}

impl<'a> Iterator for DocElems<'a> {
    type Item = Doc<'a>;

    fn next(&mut self) -> Option<Doc<'a>> {
        let out = self.d.at(self.i)?;
        self.i += 1;
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.d.len().saturating_sub(self.i);
        (left, Some(left))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An order, the shape `09` section 5 uses as its example.
    fn order(id: i64, status: &str, lines: usize) -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"id").expect("key");
        b.int(id).expect("value");
        b.key(b"customer").expect("key");
        b.int(id * 7).expect("value");
        b.key(b"status").expect("key");
        b.text(status).expect("value");
        b.key(b"lines").expect("key");
        b.begin_array().expect("open");
        for i in 0..lines {
            b.begin_object().expect("open");
            b.key(b"sku").expect("key");
            b.text(&format!("sku-{i}")).expect("value");
            b.key(b"qty").expect("key");
            b.int(i as i64 + 1).expect("value");
            b.end_object().expect("close");
        }
        b.end_array().expect("close");
        b.end_object().expect("close");
        b.finish().expect("finished").to_vec()
    }

    #[test]
    fn a_document_reads_back_the_way_it_went_in() {
        let mut docs = Docs::new();
        assert!(
            docs.put_bytes(b"order:1", &order(1, "open", 3))
                .expect("put")
        );
        assert!(
            !docs
                .put_bytes(b"order:1", &order(1, "shut", 3))
                .expect("put")
        );
        assert_eq!(docs.len(), 1);

        let d = docs.get(b"order:1").expect("stored");
        assert_eq!(d.get(b"id").and_then(|v| v.as_int()), Some(1));
        assert_eq!(d.get(b"status").and_then(|v| v.as_text()), Some("shut"));
        assert_eq!(d.get(b"lines").map(|v| v.len()), Some(3));
        assert_eq!(
            d.path("$.lines[1].sku")
                .expect("a path")
                .and_then(|v| v.as_text()),
            Some("sku-1")
        );
        assert_eq!(
            d.path("$.lines[-1].qty")
                .expect("a path")
                .and_then(|v| v.as_int()),
            Some(3)
        );
        assert!(d.get(b"missing").is_none());
    }

    #[test]
    fn the_keys_are_interned_and_the_names_come_back() {
        let mut docs = Docs::new();
        docs.put_bytes(b"order:1", &order(1, "open", 2))
            .expect("put");
        let names: Vec<String> = docs
            .keys()
            .iter()
            .map(|(n, _)| String::from_utf8_lossy(n).into_owned())
            .collect();
        names.iter().for_each(|n| assert!(!n.is_empty()));
        assert_eq!(
            docs.keys().len(),
            6,
            "id customer status lines sku qty: {names:?}"
        );

        let d = docs.get(b"order:1").expect("stored");
        assert!(d.value().is_interned());
        let mut got: Vec<&[u8]> = d.members().map(|(k, _)| k).collect();
        got.sort_unstable();
        assert_eq!(got, [&b"customer"[..], b"id", b"lines", b"status"]);
        let line = d.path("$.lines[0]").expect("a path").expect("there");
        assert!(line.value().is_interned());
        let mut inner: Vec<&[u8]> = line.members().map(|(k, _)| k).collect();
        inner.sort_unstable();
        assert_eq!(inner, [&b"qty"[..], b"sku"]);
    }

    /// Store 256 copies of a shape and say what fraction of the bytes survived.
    fn shrinkage(shape: impl Fn(i64) -> Vec<u8>) -> f64 {
        let mut docs = Docs::new();
        let mut plain = 0usize;
        for i in 0..256i64 {
            let bytes = shape(i);
            plain += bytes.len();
            docs.put_bytes(format!("d:{i}").as_bytes(), &bytes)
                .expect("put");
        }
        let stored: usize = (0..256i64)
            .map(|i| {
                docs.bytes(format!("d:{i}").as_bytes())
                    .expect("stored")
                    .len()
            })
            .sum();
        stored as f64 / plain as f64
    }

    #[test]
    fn interning_makes_a_collection_of_the_same_shape_smaller() {
        // The claim in `09` section 4 is that the same field names on every
        // document are most of what a document collection costs, and that
        // interning them is worth about forty percent. How much it is actually
        // worth depends on how much of a document is names, so both ends are
        // measured here rather than one number being asserted twice.
        //
        // A document that is mostly names, which is what a typed collection of
        // small records looks like, keeps a little over half its bytes.
        let names = shrinkage(|i| {
            let mut b = Builder::new();
            b.begin_object().expect("open");
            for f in 0..20 {
                b.key(format!("some_field_name_{f:02}").as_bytes())
                    .expect("key");
                b.int(i + f).expect("value");
            }
            b.end_object().expect("close");
            b.finish().expect("finished").to_vec()
        });
        assert!(names < 0.60, "a document of names kept {names}");

        // An order, which carries real payload as well, keeps about three
        // quarters. That is the honest floor for the claim and it is still a
        // fifth of the collection gone for nothing but a table of twenty
        // strings.
        let orders = shrinkage(|i| order(i, "open", 2));
        assert!(orders < 0.80, "an order collection kept {orders}");
    }

    #[test]
    fn a_document_whose_keys_are_already_ids_is_refused() {
        let mut b = Builder::new();
        b.begin_object_interned().expect("open");
        b.key_id(0).expect("key");
        b.int(1).expect("value");
        b.end_object().expect("close");
        let bytes = b.finish().expect("finished").to_vec();

        let mut docs = Docs::new();
        let err = docs.put_bytes(b"x", &bytes).expect_err("refused");
        assert_eq!(err.code(), Code::Invalid);
    }

    #[test]
    fn a_document_that_is_not_readable_is_refused() {
        let mut docs = Docs::new();
        let err = docs.put_bytes(b"x", &[2, 0, 0, 0]).expect_err("refused");
        assert_eq!(err.code(), Code::Corrupt);
        assert!(docs.is_empty());
    }

    #[test]
    fn a_removal_leaves_every_other_document_where_it_was() {
        let mut docs = Docs::new();
        for i in 0..64i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }
        for i in (0..64i64).step_by(3) {
            assert!(docs.remove(format!("order:{i}").as_bytes()));
        }
        assert_eq!(docs.len(), 64 - 22);
        for i in 0..64i64 {
            let id = format!("order:{i}");
            match docs.get(id.as_bytes()) {
                Some(d) => {
                    assert!(i % 3 != 0, "{id} was removed");
                    assert_eq!(d.get(b"id").and_then(|v| v.as_int()), Some(i));
                }
                None => assert!(i % 3 == 0, "{id} was not removed"),
            }
        }
        assert_eq!(docs.keys().len(), 6, "a removal does not un-intern a name");
    }

    #[test]
    fn a_walk_sees_every_document_once() {
        let mut docs = Docs::new();
        for i in 0..200i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }

        let mut seen: Vec<i64> = docs
            .iter()
            .map(|(_, d)| d.get(b"id").and_then(|v| v.as_int()).expect("an id"))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..200).collect::<Vec<i64>>());

        let mut scanned = Vec::new();
        let mut cursor = Cursor::START;
        loop {
            cursor = docs.scan(cursor, 16, |id, _| scanned.push(id.to_vec()));
            if cursor.is_end() {
                break;
            }
        }
        scanned.sort_unstable();
        scanned.dedup();
        assert_eq!(scanned.len(), 200);
    }

    #[test]
    fn an_empty_collection_answers_nothing_rather_than_failing() {
        let docs = Docs::new();
        assert!(docs.is_empty());
        assert!(docs.get(b"nothing").is_none());
        assert!(docs.bytes(b"nothing").is_none());
        assert!(!docs.contains(b"nothing"));
        assert_eq!(docs.iter().count(), 0);
    }

    #[test]
    fn a_document_prints_with_its_names_back_on() {
        let mut docs = Docs::new();
        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        let text = format!("{:?}", docs.get(b"order:1").expect("stored"));
        assert!(text.contains("\"status\": \"open\""), "{text}");
        assert!(text.contains("\"sku\": \"sku-0\""), "{text}");
    }

    /// The ids `find` answers for one key, sorted so a test can compare them.
    fn found(docs: &Docs, path: &str, key: &Key) -> Vec<String> {
        let mut out = Vec::new();
        let n = docs
            .find(path, key, |id, d| {
                assert!(d.get(b"id").is_some(), "the document came back whole");
                out.push(String::from_utf8_lossy(id).into_owned());
            })
            .expect("indexed");
        assert_eq!(n, out.len(), "the count is what the callback saw");
        out.sort();
        out
    }

    #[test]
    fn an_index_declared_after_the_documents_finds_them() {
        let mut docs = Docs::new();
        for i in 0..64i64 {
            let status = if i % 4 == 0 { "shut" } else { "open" };
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, status, 1))
                .expect("put");
        }
        docs.create_index("$.status").expect("indexed");
        assert_eq!(docs.index("$.status").expect("there").len(), 2);
        assert_eq!(docs.count("$.status", &Key::text("shut")).expect("i"), 16);
        assert_eq!(docs.count("$.status", &Key::text("open")).expect("i"), 48);
        assert_eq!(found(&docs, "$.status", &Key::text("shut")).len(), 16);
        assert!(found(&docs, "$.status", &Key::text("gone")).is_empty());

        // A document written after the index exists is filed by the write.
        docs.put_bytes(b"order:64", &order(64, "shut", 1))
            .expect("put");
        assert_eq!(docs.count("$.status", &Key::text("shut")).expect("i"), 17);
    }

    #[test]
    fn an_overwrite_moves_a_document_from_one_key_to_the_other() {
        let mut docs = Docs::new();
        docs.create_index("$.status").expect("indexed");
        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        assert_eq!(found(&docs, "$.status", &Key::text("open")), ["order:1"]);

        docs.put_bytes(b"order:1", &order(1, "shut", 1))
            .expect("put");
        assert!(
            found(&docs, "$.status", &Key::text("open")).is_empty(),
            "the old key kept it"
        );
        assert_eq!(found(&docs, "$.status", &Key::text("shut")), ["order:1"]);
        assert_eq!(docs.index("$.status").expect("there").postings(), 1);
    }

    #[test]
    fn a_removal_takes_a_document_out_of_every_index() {
        let mut docs = Docs::new();
        docs.create_index("$.status").expect("indexed");
        docs.create_index("$.customer").expect("indexed");
        for i in 0..8i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }
        assert!(docs.remove(b"order:3"));
        assert_eq!(found(&docs, "$.status", &Key::text("open")).len(), 7);
        assert_eq!(docs.count("$.customer", &Key::int(21)).expect("i"), 0);
        assert_eq!(docs.count("$.customer", &Key::int(28)).expect("i"), 1);
        for index in docs.indexes() {
            assert_eq!(index.postings(), 7);
        }

        assert!(!docs.remove(b"order:3"), "it is already gone");
        assert_eq!(docs.index("$.status").expect("there").postings(), 7);
    }

    #[test]
    fn a_path_that_names_a_container_or_nothing_is_simply_not_filed() {
        let mut docs = Docs::new();
        docs.create_index("$.lines").expect("indexed");
        docs.create_index("$.shipped").expect("indexed");
        docs.create_index("$.lines[0].qty").expect("indexed");
        for i in 0..4i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 2))
                .expect("put");
        }
        assert_eq!(docs.len(), 4);
        assert!(
            docs.index("$.lines").expect("there").is_empty(),
            "an array has no equality key"
        );
        assert!(
            docs.index("$.shipped").expect("there").is_empty(),
            "no document has that path"
        );
        assert_eq!(
            docs.count("$.lines[0].qty", &Key::int(1)).expect("i"),
            4,
            "a path through an array reaches a scalar"
        );
    }

    #[test]
    fn a_value_too_long_to_index_fails_the_write_and_stores_nothing() {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"status").expect("key");
        b.text(&"x".repeat(crate::KEY_MAX)).expect("value");
        b.end_object().expect("close");
        let huge = b.finish().expect("finished").to_vec();

        let mut docs = Docs::new();
        docs.create_index("$.status").expect("indexed");
        let err = docs.put_bytes(b"order:1", &huge).expect_err("refused");
        assert_eq!(err.code(), Code::Full);
        assert!(
            docs.is_empty(),
            "a write that cannot be indexed leaves nothing behind"
        );

        // Without the index it is an ordinary document and goes in fine.
        assert!(docs.drop_index("$.status"));
        docs.put_bytes(b"order:1", &huge).expect("put");
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn a_query_on_a_path_with_no_index_says_so_rather_than_scanning() {
        let mut docs = Docs::new();
        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        let err = docs
            .find("$.status", &Key::text("open"), |_, _| ())
            .expect_err("refused");
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(
            docs.count("$.status", &Key::text("open"))
                .expect_err("refused")
                .code(),
            Code::Invalid
        );
        assert!(docs.index("$.status").is_none());
        assert!(!docs.drop_index("$.status"));
    }

    #[test]
    fn declaring_the_same_index_twice_leaves_the_first_one_alone() {
        let mut docs = Docs::new();
        docs.create_index("$.status").expect("indexed");
        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        docs.create_index("$.status").expect("indexed again");
        assert_eq!(docs.indexes().len(), 1);
        assert_eq!(
            docs.index("$.status").expect("there").postings(),
            1,
            "a redeclaration did not double file anything"
        );
        assert!(docs.create_index("$.[").is_err(), "the path has to parse");
    }

    #[test]
    fn clearing_a_collection_empties_its_indexes_and_keeps_them() {
        let mut docs = Docs::new();
        docs.create_index("$.status").expect("indexed");
        for i in 0..8i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }
        docs.clear();
        assert!(docs.is_empty());
        assert!(docs.index("$.status").expect("still declared").is_empty());
        assert_eq!(docs.count("$.status", &Key::text("open")).expect("i"), 0);

        docs.put_bytes(b"order:9", &order(9, "open", 1))
            .expect("put");
        assert_eq!(found(&docs, "$.status", &Key::text("open")), ["order:9"]);
    }

    #[test]
    fn two_indexes_intersect_as_the_sets_they_are() {
        let mut docs = Docs::new();
        docs.create_index("$.status").expect("indexed");
        docs.create_index("$.customer").expect("indexed");
        for i in 0..32i64 {
            let status = if i % 2 == 0 { "open" } else { "shut" };
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i % 4, status, 1))
                .expect("put");
        }

        // What a planner does: probe both, walk the smaller, ask the larger.
        // That is `SINTER` and there is no code here that is not already the
        // set's.
        let open = Key::text("open");
        let customer = Key::int(14);
        let small = docs.count("$.customer", &customer).expect("indexed");
        let large = docs.count("$.status", &open).expect("indexed");
        assert_eq!((small, large), (8, 16));

        let small = docs.index("$.customer").expect("there").get(&customer);
        let large = docs.index("$.status").expect("there").get(&open);
        let (Some(small), Some(large)) = (small, large) else {
            panic!("both keys are filed");
        };
        let mut both = Vec::new();
        index::each_id(small, |id| {
            if large.contains(id) {
                both.push(String::from_utf8_lossy(id).into_owned());
            }
        });
        both.sort();
        assert_eq!(
            both,
            [
                "order:10", "order:14", "order:18", "order:2", "order:22", "order:26", "order:30",
                "order:6"
            ]
        );
    }

    /// The customer numbers a range answers, in the order it answered them.
    fn ranged(docs: &Docs, lo: Bound<&Key>, hi: Bound<&Key>) -> Vec<i64> {
        let mut out = Vec::new();
        let n = docs
            .range("$.customer", lo, hi, |_, d| {
                out.push(d.get(b"customer").and_then(|v| v.as_int()).expect("there"));
            })
            .expect("ordered");
        assert_eq!(n, out.len());

        let mut back = Vec::new();
        docs.range_rev("$.customer", lo, hi, |_, d| {
            back.push(d.get(b"customer").and_then(|v| v.as_int()).expect("there"));
        })
        .expect("ordered");
        back.reverse();
        assert_eq!(out, back, "backwards is forwards read the other way");
        assert_eq!(
            docs.count_range("$.customer", lo, hi).expect("ordered"),
            out.len()
        );
        out
    }

    #[test]
    fn an_ordered_index_answers_a_range_in_order() {
        let mut docs = Docs::new();
        docs.create_ordered_index("$.customer").expect("ordered");
        // Customer is seven times the id, so the values are 0, 7, 14 and on.
        for i in 0..64i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }

        assert_eq!(
            ranged(&docs, Bound::Unbounded, Bound::Unbounded),
            (0..64i64).map(|i| i * 7).collect::<Vec<i64>>()
        );
        let (lo, hi) = (Key::int(70), Key::int(105));
        assert_eq!(
            ranged(&docs, Bound::Included(&lo), Bound::Included(&hi)),
            [70, 77, 84, 91, 98, 105]
        );
        assert_eq!(
            ranged(&docs, Bound::Excluded(&lo), Bound::Excluded(&hi)),
            [77, 84, 91, 98]
        );
        // Bounds that fall between two values, which is the ordinary case.
        assert_eq!(
            ranged(
                &docs,
                Bound::Included(&Key::int(71)),
                Bound::Excluded(&Key::int(90))
            ),
            [77, 84]
        );
        assert!(ranged(&docs, Bound::Included(&Key::int(442)), Bound::Unbounded).is_empty());

        // Equality still works on the same index.
        assert_eq!(docs.count("$.customer", &Key::int(70)).expect("i"), 1);
        assert_eq!(
            docs.index("$.customer").expect("there").kind(),
            IndexKind::Ordered
        );
    }

    #[test]
    fn a_range_stays_right_through_writes_and_removals() {
        let mut docs = Docs::new();
        for i in 0..128i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }
        // Declared after the fact, so this is the backfill and not the write
        // path putting the tree together.
        docs.create_ordered_index("$.customer").expect("ordered");
        assert_eq!(ranged(&docs, Bound::Unbounded, Bound::Unbounded).len(), 128);

        // Every removal moves the key table's last row into the hole, so this is
        // the renumbering going through the whole collection.
        for i in (0..128i64).step_by(2) {
            assert!(docs.remove(format!("order:{i}").as_bytes()));
        }
        assert_eq!(
            ranged(&docs, Bound::Unbounded, Bound::Unbounded),
            (0..128i64)
                .filter(|i| i % 2 == 1)
                .map(|i| i * 7)
                .collect::<Vec<i64>>()
        );

        // And an overwrite that moves a document from one key to another.
        docs.put_bytes(b"order:1", &order(200, "open", 1))
            .expect("put");
        let after = ranged(&docs, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(after.first(), Some(&21), "seven is gone");
        assert_eq!(after.last(), Some(&1400), "and it came back at the top");
    }

    #[test]
    fn an_equality_index_refuses_a_range_rather_than_answering_nothing() {
        let mut docs = Docs::new();
        docs.create_index("$.customer").expect("indexed");
        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        let err = docs
            .range("$.customer", Bound::Unbounded, Bound::Unbounded, |_, _| ())
            .expect_err("refused");
        assert_eq!(err.code(), Code::Invalid);
        assert!(err.to_string().contains("equality"), "{err}");
        assert_eq!(
            docs.range("$.status", Bound::Unbounded, Bound::Unbounded, |_, _| ())
                .expect_err("refused")
                .code(),
            Code::Invalid
        );
    }

    #[test]
    fn asking_for_an_order_on_an_equality_index_upgrades_it() {
        let mut docs = Docs::new();
        docs.create_index("$.customer").expect("indexed");
        for i in 0..8i64 {
            docs.put_bytes(format!("order:{i}").as_bytes(), &order(i, "open", 1))
                .expect("put");
        }
        assert_eq!(
            docs.index("$.customer").expect("there").kind(),
            IndexKind::Equality
        );

        docs.create_ordered_index("$.customer").expect("upgraded");
        assert_eq!(docs.indexes().len(), 1, "it replaced rather than added");
        assert_eq!(ranged(&docs, Bound::Unbounded, Bound::Unbounded).len(), 8);

        // And going the other way leaves the order alone, because an ordered
        // index answers equality too.
        docs.create_index("$.customer").expect("already there");
        assert_eq!(
            docs.index("$.customer").expect("there").kind(),
            IndexKind::Ordered
        );
        assert_eq!(docs.indexes().len(), 1);
    }

    #[test]
    fn a_collection_whose_key_table_is_full_stores_the_rest_with_names() {
        // Fill the table with names no document below uses, then write one and
        // check it is stored whole rather than refused.
        let mut docs = Docs::new();
        for i in 0..crate::KEYS_MAX {
            let name = format!("filler{i}");
            assert!(docs.keys.intern(name.as_bytes()).is_some());
        }
        assert!(docs.keys().is_full());

        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        let d = docs.get(b"order:1").expect("stored");
        assert!(!d.value().is_interned(), "there were no ids left to use");
        assert_eq!(d.get(b"status").and_then(|v| v.as_text()), Some("open"));
        assert_eq!(
            d.path("$.lines[0].sku")
                .expect("a path")
                .and_then(|v| v.as_text()),
            Some("sku-0")
        );
    }
}
