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
//! A vector index at a path is read in the same pass and for the same reason: a
//! document whose embedding is the wrong shape fails the write rather than
//! landing in the collection with no vector in it. It is filed after the
//! document is stored, along with the tag that a filtered search tests, which is
//! the keys the other indexes just filed this document under. See
//! [`vector`](crate::vector).
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
use yo_shape::Metric;
use yo_vector::{Match, Signature};

use crate::head::{DEPTH_MAX, Kind};
use crate::index::{self, IndexKind, Key, PathIndex};
use crate::path::{Step, Steps};
use crate::vector::{self, VectorIndex};
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
    /// One per path holding an embedding, in the order they were declared.
    ///
    /// Beside the path indexes rather than among them, because nearness has no
    /// key to file under. See [`vector`](crate::vector).
    vectors: Vec<VectorIndex>,
    /// The vector each of those takes from the document being written, one slot
    /// per index and empty where the document has nothing at the path.
    ///
    /// Read before anything is stored, for the same reason the index keys are:
    /// a document whose embedding is the wrong shape fails the write rather than
    /// landing in the collection and in none of its vector indexes.
    drawn: Vec<Vec<f32>>,
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
            vectors: Vec::new(),
            drawn: Vec::new(),
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
            vectors: Vec::new(),
            drawn: Vec::new(),
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
            vectors,
            drawn,
        } = self;

        taken.resize(indexes.len(), Vec::new());
        for (slot, index) in taken.iter_mut().zip(indexes.iter()) {
            slot.clear();
            // The incoming value has its keys as bytes, since put refuses one
            // that does not, so its paths resolve without the key table.
            let Some(at) = value.path_bytes(index.path())? else {
                continue;
            };
            if index.keys_at(at, slot).is_err() {
                return Err(Error::fmt(
                    Code::Full,
                    format_args!(
                        "a value at {} is longer than {} bytes and cannot be indexed",
                        String::from_utf8_lossy(index.path()),
                        index::KEY_MAX
                    ),
                ));
            }
        }

        drawn.resize(vectors.len(), Vec::new());
        for (slot, index) in drawn.iter_mut().zip(vectors.iter()) {
            slot.clear();
            let Some(at) = value.path_bytes(index.path())? else {
                continue;
            };
            vector::coordinates(at, index.dim(), index.path(), slot)?;
        }
        // What a filtered search will meet in the posting scan: one bit per key
        // the other indexes file this document under, worked out here because
        // `taken` already holds exactly those keys.
        let tag = if vectors.is_empty() {
            0
        } else {
            vector::tag_of(
                indexes
                    .iter()
                    .map(PathIndex::path)
                    .zip(taken.iter().map(Vec::as_slice)),
            )
        };

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
            let mut filed = Ok(());
            index::each_key(slot, |key| {
                if filed.is_ok() {
                    filed = index.add(key, id);
                }
            });
            filed?;
        }
        // A document that no longer has anything at the path leaves the vector
        // index, rather than keeping the vector the last version of it had.
        for (slot, index) in drawn.iter().zip(vectors.iter_mut()) {
            if slot.is_empty() {
                index.collection_mut().remove(id);
            } else {
                index.collection_mut().put_tagged(id, slot, tag)?;
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

    /// Start indexing every element of the array at `path`.
    ///
    /// A document with `["red", "blue"]` there is filed under both, so a search
    /// for either finds it. A scalar at the path is an array of one, so a
    /// collection where some documents have a list of tags and some have a
    /// single tag works without the caller having to normalise it first.
    ///
    /// The lookup is [`Docs::find`] with the element as the key, unchanged. An
    /// array index costs what the document has at the path, so a document with
    /// ten elements costs ten postings and a document with none costs nothing.
    pub fn create_array_index(&mut self, path: &str) -> Result<()> {
        self.create_index_bytes(path.as_bytes(), IndexKind::Array)
    }

    /// Start indexing every word of the string at `path`.
    ///
    /// A document with `"A red bicycle"` there is filed under `a`, `red` and
    /// `bicycle`, and the lookup is [`Docs::find`] with [`Key::word`] as the
    /// key. Case is folded on both sides, so a search does not have to know how
    /// the document was written.
    ///
    /// This is a word index and not a search engine. There is no ranking, no
    /// stemming and no phrase matching, and a path that holds something other
    /// than a string files nothing. What it answers is which documents contain
    /// a word, which is a filter, and the ranking that belongs on top of it is
    /// `10`.
    pub fn create_text_index(&mut self, path: &str) -> Result<()> {
        self.create_index_bytes(path.as_bytes(), IndexKind::Text)
    }

    /// [`Docs::create_index`] and [`Docs::create_ordered_index`] for a path that
    /// is already bytes.
    pub fn create_index_bytes(&mut self, path: &[u8], kind: IndexKind) -> Result<()> {
        for step in Steps::new(path) {
            step?;
        }
        // The same kind again is nothing at all, and equality on top of ordered
        // is already answered. Every other pair means the path is being asked a
        // different question, so it gets rebuilt. The old one stays in place
        // until the new one is filled, so a backfill that fails leaves the
        // collection with the index it already had rather than with none.
        let old = match self.indexes.iter().position(|i| i.path() == path) {
            Some(at) if self.indexes[at].kind() == kind => return Ok(()),
            Some(at)
                if kind == IndexKind::Equality && self.indexes[at].kind() == IndexKind::Ordered =>
            {
                return Ok(());
            }
            found => found,
        };
        let mut index = PathIndex::new(path, kind);
        let mut list = Vec::new();
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
            list.clear();
            if index.keys_at(at.value(), &mut list).is_err() {
                return Err(Error::fmt(
                    Code::Full,
                    format_args!(
                        "a value at {} in {} is longer than {} bytes and cannot be indexed",
                        String::from_utf8_lossy(path),
                        String::from_utf8_lossy(id),
                        index::KEY_MAX
                    ),
                ));
            }
            let mut filed = Ok(());
            index::each_key(&list, |key| {
                if filed.is_ok() {
                    filed = index.add(key, id);
                }
            });
            filed?;
        }
        match old {
            Some(at) => self.indexes[at] = index,
            None => {
                self.indexes.push(index);
                self.taken.push(Vec::new());
            }
        }
        self.retag();
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
        self.retag();
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

    /// Start indexing the `dim` wide embedding at `path` for nearness, by
    /// cosine, and file every document already here.
    ///
    /// Cosine because that is what a text or image embedding is compared by, and
    /// a collection built for one measure and searched as if it were another
    /// gives wrong answers quietly. [`Docs::create_vector_index_with`] takes the
    /// other one.
    ///
    /// Declaring the same path at the same width and measure twice is not an
    /// error and rebuilds nothing, the same as [`Docs::create_index`], so a
    /// caller that declares its indexes every time it opens a collection can.
    /// Changing the width or the measure rebuilds, because there is nothing in
    /// the old collection that answers the new question.
    pub fn create_vector_index(&mut self, path: &str, dim: usize) -> Result<()> {
        self.create_vector_index_bytes(path.as_bytes(), dim, Metric::Cosine)
    }

    /// The same, saying what nearness means.
    pub fn create_vector_index_with(
        &mut self,
        path: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<()> {
        self.create_vector_index_bytes(path.as_bytes(), dim, metric)
    }

    /// [`Docs::create_vector_index_with`] for a path that is already bytes.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for a path that does not parse, a width of zero or past
    /// the format's limit, and for a document already here whose value at the
    /// path is not an array of `dim` numbers. [`Code::Unsupported`] for a
    /// measure this build does not do.
    pub fn create_vector_index_bytes(
        &mut self,
        path: &[u8],
        dim: usize,
        metric: Metric,
    ) -> Result<()> {
        for step in Steps::new(path) {
            step?;
        }
        // As with a path index, the one that is already here stays until the new
        // one is filled, so a document whose embedding is not the new width
        // leaves the collection with the index it had rather than with none.
        let old = match self.vectors.iter().position(|v| v.path() == path) {
            Some(at) if self.vectors[at].dim() == dim && self.vectors[at].metric() == metric => {
                return Ok(());
            }
            found => found,
        };
        let mut index = VectorIndex::new(path, dim, metric)?;
        let mut list = Vec::new();
        let mut v = Vec::new();
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
            vector::coordinates(at.value(), dim, path, &mut v)?;
            let tag = tag_for(&doc, &self.indexes, &mut list);
            index.collection_mut().put_tagged(id, &v, tag)?;
        }
        match old {
            Some(at) => self.vectors[at] = index,
            None => {
                self.vectors.push(index);
                self.drawn.push(Vec::new());
            }
        }
        Ok(())
    }

    /// Stop indexing the embedding at `path`, and say whether it was indexed.
    pub fn drop_vector_index(&mut self, path: &str) -> bool {
        self.drop_vector_index_bytes(path.as_bytes())
    }

    /// [`Docs::drop_vector_index`] for a path that is already bytes.
    pub fn drop_vector_index_bytes(&mut self, path: &[u8]) -> bool {
        let Some(at) = self.vectors.iter().position(|v| v.path() == path) else {
            return false;
        };
        self.vectors.remove(at);
        self.drawn.truncate(self.vectors.len());
        true
    }

    /// The vector indexes this collection keeps, in the order they were
    /// declared.
    #[must_use]
    pub fn vector_indexes(&self) -> &[VectorIndex] {
        &self.vectors
    }

    /// The vector index on `path`, if there is one.
    #[must_use]
    pub fn vector_index(&self, path: &str) -> Option<&VectorIndex> {
        self.vectors.iter().find(|v| v.path() == path.as_bytes())
    }

    /// The vector stored for `id` at `path`, as the index holds it.
    ///
    /// For a cosine index that is the normalised vector and not the one the
    /// document carries, because normalising once at write time is what makes
    /// every later comparison a dot product. The document itself still has what
    /// was written.
    #[must_use]
    pub fn embedding(&self, path: &str, id: &[u8]) -> Option<&[f32]> {
        self.vector_index(path)?.collection().get(id)
    }

    /// Hand the `k` documents nearest to `q` at `path` to `f`, nearest first,
    /// and say how many there were.
    ///
    /// The third argument to `f` is the distance, which for a cosine index is
    /// one minus the cosine so that nearer is smaller, and it is measured
    /// against the full precision vector rather than against the code.
    ///
    /// A path with no vector index on it is an error and not a scan, the same
    /// rule [`Docs::find`] follows.
    pub fn nearest(
        &self,
        path: &str,
        q: &[f32],
        k: usize,
        f: impl FnMut(&[u8], Doc<'_>, f32),
    ) -> Result<usize> {
        let hits = self.vector(path)?.collection().search(q, k, None)?;
        Ok(self.answer(&hits, f))
    }

    /// The same, over only the documents whose indexed fields hold every value
    /// in `want`.
    ///
    /// The filter runs inside the posting scan rather than over the answers, so
    /// a selective filter returns the nearest documents that match instead of
    /// whichever of the nearest happened to match. See
    /// [`vector`](crate::vector) for the encoding and for the one direction it
    /// is not exact in: a document can pass a filter it does not really match,
    /// never fail one it does, so a caller with a predicate of its own still
    /// gets every answer to check.
    ///
    /// Every path in `want` has to carry an ordinary index, because the bits the
    /// filter tests are the keys those indexes filed the document under. A path
    /// with no index is an error rather than a filter that matches nothing.
    pub fn nearest_where(
        &self,
        path: &str,
        q: &[f32],
        k: usize,
        want: &[(&str, Key)],
        f: impl FnMut(&[u8], Doc<'_>, f32),
    ) -> Result<usize> {
        let filter = self.wanted(want)?;
        let hits = self
            .vector(path)?
            .collection()
            .search_where(q, k, None, &filter)?;
        Ok(self.answer(&hits, f))
    }

    /// Hand the `k` documents most like the one under `id` to `f`, `id` itself
    /// left out.
    ///
    /// More like this, which is the query a document collection with embeddings
    /// in it is really for. A document with nothing at the path has nothing to
    /// be like, so it answers zero rather than an error.
    pub fn nearest_to(
        &self,
        path: &str,
        id: &[u8],
        k: usize,
        f: impl FnMut(&[u8], Doc<'_>, f32),
    ) -> Result<usize> {
        let index = self.vector(path)?;
        let Some(q) = index.collection().get(id) else {
            return Ok(0);
        };
        let hits = index.collection().search(q, k, Some(id))?;
        Ok(self.answer(&hits, f))
    }

    /// The vector index on `path`, or the error that says why there is not one.
    fn vector(&self, path: &str) -> Result<&VectorIndex> {
        self.vector_index(path).ok_or_else(|| {
            Error::fmt(
                Code::Invalid,
                format_args!("there is no vector index on {path}, so this would be a scan"),
            )
        })
    }

    /// Turn what a query requires into the signature the scan tests.
    fn wanted(&self, want: &[(&str, Key)]) -> Result<Signature> {
        let mut sig = Signature::default();
        for (path, key) in want {
            if self.index(path).is_none() {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("there is no index on {path}, so a search cannot filter on it"),
                ));
            }
            sig.insert(path, key.as_bytes());
        }
        Ok(sig)
    }

    /// Read the documents a search answered with, skipping any that have gone.
    fn answer(&self, hits: &[Match], mut f: impl FnMut(&[u8], Doc<'_>, f32)) -> usize {
        let mut n = 0usize;
        for hit in hits {
            if let Some(doc) = self.get(&hit.key) {
                f(&hit.key, doc, hit.distance);
                n += 1;
            }
        }
        n
    }

    /// Work out every document's tag again and write it back.
    ///
    /// A tag summarises the keys the other indexes filed the document under, so
    /// declaring or dropping one makes every tag wrong. This is one store per
    /// document per vector index, with no requantising and no maintenance, which
    /// is cheap enough to do on the spot and is the only alternative to a filter
    /// that used to work quietly answering nothing.
    fn retag(&mut self) {
        let Docs {
            rows,
            keys,
            indexes,
            vectors,
            ..
        } = self;
        if vectors.is_empty() {
            return;
        }
        let mut list = Vec::new();
        for (id, bytes) in rows.pairs() {
            let Some(value) = Value::new(bytes) else {
                continue;
            };
            let doc = Doc { value, keys };
            let tag = tag_for(&doc, indexes, &mut list);
            for index in vectors.iter_mut() {
                index.collection_mut().retag(id, tag);
            }
        }
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
            vectors,
            ..
        } = self;
        unindex(rows, keys, indexes, id);
        for index in vectors.iter_mut() {
            index.collection_mut().remove(id);
        }
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
        for index in &mut self.vectors {
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
            + self
                .vectors
                .iter()
                .map(VectorIndex::memory_bytes)
                .sum::<usize>()
    }
}

/// The tag a stored document carries, worked out from the indexes it is filed
/// in.
///
/// Nothing here can fail. A path that no longer resolves or a value that is too
/// long to be an index key contributes no bit, which is the same absence the
/// document has in that index.
fn tag_for(doc: &Doc<'_>, indexes: &[PathIndex], list: &mut Vec<u8>) -> u64 {
    let mut sig = Signature::default();
    for index in indexes {
        let Ok(Some(at)) = doc.path_bytes(index.path()) else {
            continue;
        };
        list.clear();
        let _ = index.keys_at(at.value(), list);
        vector::add_keys(&mut sig, index.path(), list);
    }
    sig.bits()
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
    let mut list = Vec::new();
    for index in indexes {
        let Ok(Some(at)) = doc.path_bytes(index.path()) else {
            continue;
        };
        list.clear();
        // The keys came out of this same code on the way in, so a key that was
        // refused then is not filed now and there is nothing to take out.
        let _ = index.keys_at(at.value(), &mut list);
        index::each_key(&list, |key| index.take(key, id));
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
                assert!(!d.is_empty(), "the document came back whole");
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

    /// A document with a list of tags at `$.tags` and a title at `$.title`.
    fn tagged(title: &str, tags: &[&str]) -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"title").expect("key");
        b.text(title).expect("value");
        b.key(b"tags").expect("key");
        b.begin_array().expect("open");
        for tag in tags {
            b.text(tag).expect("value");
        }
        b.end_array().expect("close");
        b.end_object().expect("close");
        b.finish().expect("finished").to_vec()
    }

    #[test]
    fn an_array_index_files_a_document_under_every_element() {
        let mut docs = Docs::new();
        docs.create_array_index("$.tags").expect("indexed");
        docs.put_bytes(b"a", &tagged("one", &["red", "blue"]))
            .expect("put");
        docs.put_bytes(b"b", &tagged("two", &["blue", "green"]))
            .expect("put");
        docs.put_bytes(b"c", &tagged("three", &[])).expect("put");

        assert_eq!(found(&docs, "$.tags", &Key::text("red")), ["a"]);
        assert_eq!(found(&docs, "$.tags", &Key::text("blue")), ["a", "b"]);
        assert_eq!(found(&docs, "$.tags", &Key::text("green")), ["b"]);
        assert!(found(&docs, "$.tags", &Key::text("puce")).is_empty());
        assert_eq!(
            docs.index("$.tags").expect("there").len(),
            3,
            "three distinct tags over two documents"
        );
    }

    #[test]
    fn an_array_index_takes_every_element_back_out_again() {
        let mut docs = Docs::new();
        docs.create_array_index("$.tags").expect("indexed");
        docs.put_bytes(b"a", &tagged("one", &["red", "blue"]))
            .expect("put");
        docs.put_bytes(b"b", &tagged("two", &["blue"]))
            .expect("put");

        // An overwrite drops one tag and gains another.
        docs.put_bytes(b"a", &tagged("one", &["blue", "green"]))
            .expect("put");
        assert!(found(&docs, "$.tags", &Key::text("red")).is_empty());
        assert_eq!(found(&docs, "$.tags", &Key::text("blue")), ["a", "b"]);
        assert_eq!(found(&docs, "$.tags", &Key::text("green")), ["a"]);

        assert!(docs.remove(b"a"));
        assert_eq!(found(&docs, "$.tags", &Key::text("blue")), ["b"]);
        assert!(found(&docs, "$.tags", &Key::text("green")).is_empty());
        assert_eq!(
            docs.index("$.tags").expect("there").len(),
            1,
            "a tag nobody has left is not a key any more"
        );
    }

    #[test]
    fn an_array_index_treats_one_value_as_a_list_of_one() {
        let mut docs = Docs::new();
        docs.create_array_index("$.status").expect("indexed");
        docs.put_bytes(b"order:1", &order(1, "open", 1))
            .expect("put");
        assert_eq!(found(&docs, "$.status", &Key::text("open")), ["order:1"]);
    }

    #[test]
    fn the_same_element_twice_is_one_posting() {
        let mut docs = Docs::new();
        docs.create_array_index("$.tags").expect("indexed");
        docs.put_bytes(b"a", &tagged("one", &["red", "red", "red"]))
            .expect("put");
        assert_eq!(found(&docs, "$.tags", &Key::text("red")), ["a"]);
        assert_eq!(docs.index("$.tags").expect("there").postings(), 1);

        // And taking it out once takes it out, rather than three times over.
        assert!(docs.remove(b"a"));
        assert_eq!(docs.index("$.tags").expect("there").postings(), 0);
        assert!(docs.index("$.tags").expect("there").is_empty());
    }

    #[test]
    fn a_text_index_files_a_document_under_every_word() {
        let mut docs = Docs::new();
        docs.create_text_index("$.title").expect("indexed");
        docs.put_bytes(b"a", &tagged("A red bicycle", &[]))
            .expect("put");
        docs.put_bytes(b"b", &tagged("The red car, and a bicycle!", &[]))
            .expect("put");

        assert_eq!(found(&docs, "$.title", &word("bicycle")), ["a", "b"]);
        assert_eq!(found(&docs, "$.title", &word("car")), ["b"]);
        assert_eq!(
            found(&docs, "$.title", &word("RED")),
            ["a", "b"],
            "a search folds case the same way the write did"
        );
        assert!(found(&docs, "$.title", &word("lorry")).is_empty());
    }

    #[test]
    fn a_text_index_follows_the_words_through_a_rewrite() {
        let mut docs = Docs::new();
        docs.create_text_index("$.title").expect("indexed");
        docs.put_bytes(b"a", &tagged("a red bicycle", &[]))
            .expect("put");
        docs.put_bytes(b"a", &tagged("a blue bicycle", &[]))
            .expect("put");
        assert!(found(&docs, "$.title", &word("red")).is_empty());
        assert_eq!(found(&docs, "$.title", &word("blue")), ["a"]);
        assert_eq!(found(&docs, "$.title", &word("bicycle")), ["a"]);

        assert!(docs.remove(b"a"));
        assert!(docs.index("$.title").expect("there").is_empty());
    }

    #[test]
    fn a_text_index_declared_after_the_documents_finds_them() {
        let mut docs = Docs::new();
        for i in 0..16i64 {
            let title = if i % 2 == 0 {
                "a red one"
            } else {
                "a blue one"
            };
            docs.put_bytes(format!("t:{i}").as_bytes(), &tagged(title, &[]))
                .expect("put");
        }
        docs.create_text_index("$.title").expect("indexed");
        assert_eq!(docs.count("$.title", &word("red")).expect("i"), 8);
        assert_eq!(docs.count("$.title", &word("one")).expect("i"), 16);
        assert_eq!(
            docs.index("$.title").expect("there").len(),
            4,
            "a, red, blue and one"
        );
    }

    #[test]
    fn changing_what_an_index_is_asked_rebuilds_it() {
        let mut docs = Docs::new();
        docs.create_index("$.tags").expect("indexed");
        docs.put_bytes(b"a", &tagged("one", &["red", "blue"]))
            .expect("put");
        assert!(
            found(&docs, "$.tags", &Key::text("red")).is_empty(),
            "an equality index over an array files nothing"
        );

        docs.create_array_index("$.tags").expect("rebuilt");
        assert_eq!(docs.indexes().len(), 1, "it replaced rather than added");
        assert_eq!(found(&docs, "$.tags", &Key::text("red")), ["a"]);

        docs.create_array_index("$.tags").expect("already there");
        assert_eq!(docs.indexes().len(), 1);
    }

    /// The key a text index files one word under.
    fn word(w: &str) -> Key {
        Key::word(w).expect("one word")
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

    // ---- the vector index

    /// A document with a language and an embedding, which is the shape `10`
    /// section 3 uses as its example.
    fn item(lang: &str, v: &[f32]) -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"lang").expect("key");
        b.text(lang).expect("value");
        b.key(b"embedding").expect("key");
        b.begin_array().expect("open");
        for x in v {
            b.float(f64::from(*x)).expect("value");
        }
        b.end_array().expect("close");
        b.end_object().expect("close");
        b.finish().expect("finished").to_vec()
    }

    /// The same document with no embedding in it at all.
    fn bare(lang: &str) -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"lang").expect("key");
        b.text(lang).expect("value");
        b.end_object().expect("close");
        b.finish().expect("finished").to_vec()
    }

    /// Eight coordinates that depend only on `n`, so a test that fails fails
    /// the same way twice.
    fn spread(n: u64) -> [f32; 8] {
        let mut s = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut v = [0.0f32; 8];
        for x in &mut v {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = (s >> 40) as f32 / 4096.0 - 1.0;
        }
        v
    }

    #[test]
    fn a_document_and_its_embedding_are_one_write() {
        let mut docs = Docs::new();
        docs.create_vector_index("$.embedding", 3).expect("index");
        for (id, v) in [
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0]),
            ("c", [0.0, 0.0, 1.0]),
        ] {
            docs.put_bytes(id.as_bytes(), &item("en", &v)).expect("put");
        }
        assert_eq!(docs.vector_index("$.embedding").expect("declared").len(), 3);

        // The answer is documents and not ids to go and look up somewhere else,
        // and it comes back nearest first.
        let mut got = Vec::new();
        let n = docs
            .nearest("$.embedding", &[0.9, 0.1, 0.0], 3, |id, doc, d| {
                let lang = doc
                    .get(b"lang")
                    .and_then(|v| v.as_text())
                    .map(str::to_owned);
                got.push((id.to_vec(), lang, d));
            })
            .expect("nearest");
        assert_eq!(n, 3);
        assert_eq!(got[0].0, b"a".to_vec());
        assert_eq!(got[0].1.as_deref(), Some("en"));
        assert!(got[0].2 <= got[1].2 && got[1].2 <= got[2].2);

        // A path with no vector index on it says so rather than scanning.
        assert!(
            docs.nearest("$.lang", &[1.0, 0.0, 0.0], 1, |_, _, _| {})
                .is_err()
        );
    }

    #[test]
    fn an_embedding_of_the_wrong_shape_fails_the_write_and_stores_nothing() {
        let mut docs = Docs::new();
        docs.create_vector_index("$.embedding", 3).expect("index");
        docs.put_bytes(b"a", &item("en", &[1.0, 0.0, 0.0]))
            .expect("put");

        for wrong in [vec![1.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]] {
            assert!(docs.put_bytes(b"b", &item("en", &wrong)).is_err());
        }
        assert!(docs.get(b"b").is_none(), "the write left nothing behind");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs.vector_index("$.embedding").expect("declared").len(), 1);
    }

    #[test]
    fn a_document_with_nothing_at_the_path_is_not_in_the_index() {
        let mut docs = Docs::new();
        docs.create_vector_index("$.embedding", 3).expect("index");
        docs.put_bytes(b"a", &item("en", &[1.0, 0.0, 0.0]))
            .expect("put");
        docs.put_bytes(b"b", &bare("en")).expect("put");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs.vector_index("$.embedding").expect("declared").len(), 1);

        // Rewriting a document without its embedding takes the old vector out
        // rather than leaving the one the last version had.
        docs.put_bytes(b"a", &bare("en")).expect("put");
        assert!(
            docs.vector_index("$.embedding")
                .expect("declared")
                .is_empty()
        );
        assert!(docs.get(b"a").is_some());

        // And a removal takes it with it.
        docs.put_bytes(b"a", &item("en", &[1.0, 0.0, 0.0]))
            .expect("put");
        assert!(docs.remove(b"a"));
        assert!(
            docs.vector_index("$.embedding")
                .expect("declared")
                .is_empty()
        );
    }

    #[test]
    fn a_filter_finds_the_nearest_match_and_not_the_nearest_that_matches() {
        let mut docs = Docs::new();
        docs.create_index("$.lang").expect("index");
        docs.create_vector_index("$.embedding", 8).expect("index");

        // Four hundred English documents spread about, and one French one that
        // sits nowhere near the query.
        for n in 0..400u64 {
            let id = format!("en:{n}");
            docs.put_bytes(id.as_bytes(), &item("en", &spread(n)))
                .expect("put");
        }
        docs.put_bytes(b"fr", &item("fr", &spread(9_999)))
            .expect("put");

        let q = spread(3);
        let mut top = Vec::new();
        docs.nearest("$.embedding", &q, 20, |id, _, _| top.push(id.to_vec()))
            .expect("nearest");
        assert_eq!(top[0], b"en:3".to_vec());
        assert!(
            !top.iter().any(|id| id == b"fr"),
            "searching and then filtering would have answered nothing"
        );

        // Filtering inside the scan finds it anyway.
        let french = [("$.lang", Key::text("fr"))];
        let mut found = Vec::new();
        docs.nearest_where("$.embedding", &q, 5, &french, |id, _, _| {
            found.push(id.to_vec())
        })
        .expect("nearest");
        assert_eq!(found, [b"fr".to_vec()]);

        // A path with no index on it cannot be filtered on, and says so.
        let nothing = [("$.topic", Key::text("finance"))];
        assert!(
            docs.nearest_where("$.embedding", &q, 5, &nothing, |_, _, _| {})
                .is_err()
        );
    }

    #[test]
    fn declaring_either_index_last_gives_the_same_answers() {
        let q = spread(11);
        let french = [("$.lang", Key::text("fr"))];

        // Vectors first, then the field the filter reads, so every tag was
        // written before there was anything to put in it.
        let mut late = Docs::new();
        late.create_vector_index("$.embedding", 8).expect("index");
        for n in 0..200u64 {
            let lang = if n % 50 == 0 { "fr" } else { "en" };
            let id = format!("{n}");
            late.put_bytes(id.as_bytes(), &item(lang, &spread(n)))
                .expect("put");
        }
        late.create_index("$.lang").expect("index");

        // The other way round, where every write already knew.
        let mut early = Docs::new();
        early.create_index("$.lang").expect("index");
        for n in 0..200u64 {
            let lang = if n % 50 == 0 { "fr" } else { "en" };
            let id = format!("{n}");
            early
                .put_bytes(id.as_bytes(), &item(lang, &spread(n)))
                .expect("put");
        }
        early.create_vector_index("$.embedding", 8).expect("index");

        let mut a = Vec::new();
        late.nearest_where("$.embedding", &q, 4, &french, |id, _, _| {
            a.push(id.to_vec())
        })
        .expect("nearest");
        let mut b = Vec::new();
        early
            .nearest_where("$.embedding", &q, 4, &french, |id, _, _| {
                b.push(id.to_vec())
            })
            .expect("nearest");
        assert_eq!(a.len(), 4, "there are four French documents to find");
        assert_eq!(a, b);

        // Dropping the field index and declaring it again leaves the tags right.
        assert!(late.drop_index("$.lang"));
        late.create_index("$.lang").expect("index");
        let mut again = Vec::new();
        late.nearest_where("$.embedding", &q, 4, &french, |id, _, _| {
            again.push(id.to_vec());
        })
        .expect("nearest");
        assert_eq!(again, a);
    }

    #[test]
    fn nearest_to_leaves_the_document_itself_out() {
        let mut docs = Docs::new();
        docs.create_vector_index("$.embedding", 3).expect("index");
        for (id, v) in [
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.9, 0.1, 0.0]),
            ("c", [0.0, 0.0, 1.0]),
        ] {
            docs.put_bytes(id.as_bytes(), &item("en", &v)).expect("put");
        }

        let mut like = Vec::new();
        docs.nearest_to("$.embedding", b"a", 2, |id, _, _| like.push(id.to_vec()))
            .expect("nearest");
        assert_eq!(like, [b"b".to_vec(), b"c".to_vec()]);

        // A document with no embedding has nothing to be like.
        docs.put_bytes(b"d", &bare("en")).expect("put");
        let mut none = 0;
        assert_eq!(
            docs.nearest_to("$.embedding", b"d", 2, |_, _, _| none += 1)
                .expect("nearest"),
            0
        );
    }

    #[test]
    fn declaring_the_same_vector_index_again_rebuilds_nothing() {
        let mut docs = Docs::new();
        for n in 0..8u64 {
            let id = format!("{n}");
            docs.put_bytes(id.as_bytes(), &item("en", &spread(n)))
                .expect("put");
        }
        docs.create_vector_index("$.embedding", 8).expect("index");
        assert_eq!(docs.vector_index("$.embedding").expect("declared").len(), 8);

        // The same declaration is nothing at all.
        docs.create_vector_index("$.embedding", 8).expect("again");
        assert_eq!(docs.vector_indexes().len(), 1);

        // A different width is a different question, so it is rebuilt, and
        // documents whose embedding is not that wide fail the declaration. The
        // index that was already there is the one that is still there.
        assert!(docs.create_vector_index("$.embedding", 4).is_err());
        let still = docs.vector_index("$.embedding").expect("still declared");
        assert_eq!(still.dim(), 8);
        assert_eq!(still.len(), 8);

        assert!(docs.drop_vector_index("$.embedding"));
        assert!(!docs.drop_vector_index("$.embedding"));
        assert!(docs.vector_indexes().is_empty());
        assert_eq!(docs.len(), 8, "the documents are untouched");
    }

    #[test]
    fn clearing_a_collection_empties_the_vector_index_and_keeps_it_declared() {
        let mut docs = Docs::new();
        docs.create_vector_index("$.embedding", 3).expect("index");
        docs.put_bytes(b"a", &item("en", &[1.0, 0.0, 0.0]))
            .expect("put");
        let full = docs.memory_bytes();

        docs.clear();
        assert!(docs.is_empty());
        assert!(
            docs.vector_index("$.embedding")
                .expect("declared")
                .is_empty()
        );
        assert!(docs.memory_bytes() < full);

        docs.put_bytes(b"b", &item("en", &[0.0, 1.0, 0.0]))
            .expect("put");
        let mut got = Vec::new();
        docs.nearest("$.embedding", &[0.0, 1.0, 0.0], 1, |id, _, _| {
            got.push(id.to_vec())
        })
        .expect("nearest");
        assert_eq!(got, [b"b".to_vec()]);
    }
}
