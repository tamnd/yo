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
//! worth actually gets saved. The walk is the same pass that will extract the
//! indexed paths when the path indexes land, so a document is read once on the
//! way in and not once per index.
//!
//! If the key table fills part way through, the document is stored as it
//! arrived with its keys as bytes. Nothing about that is a fallback mode: the
//! interned flag sits in each container's header, so a collection holds both
//! kinds at once, a reader tells them apart per container, and documents
//! written before the table filled stay exactly as they were.
//!
//! # Reading one back
//!
//! [`Docs::get`] answers a [`Doc`], which is a [`Value`] with the collection's
//! key table beside it. Everything that needs a name rather than an id goes
//! through the table: `get(b"status")` resolves the name to an id once and then
//! searches the document by id, which is a binary search over integers.

use yo_common::{Code, Error, Result};
use yo_kv::{Cursor, Elements, Full};

use crate::head::{DEPTH_MAX, Kind};
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
        self.build.clear();
        if intern_into(&mut self.keys, &mut self.build, value, 0)? {
            let bytes = self.build.finish()?;
            return store(&mut self.rows, id, bytes);
        }
        // The key table filled part way through. Store what arrived, keys and
        // all, which is a memcpy and needs no second walk.
        self.build.clear();
        self.build.embed(&value)?;
        let bytes = self.build.finish()?;
        store(&mut self.rows, id, bytes)
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
        self.build.clear();
        if intern_into(&mut self.keys, &mut self.build, value, 0)? {
            let bytes = self.build.finish()?;
            return store(&mut self.rows, id, bytes);
        }
        store(&mut self.rows, id, doc)
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
    /// The key table is left alone. A name it interned stays interned even if
    /// this was the last document using it, which is [`Keys`]'s rule and the
    /// reason an id is a row index.
    pub fn remove(&mut self, id: &[u8]) -> bool {
        self.rows.remove(id).is_some()
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
    /// The key table stays because a collection that is emptied is usually a
    /// collection that is about to be refilled with the same shape of document,
    /// and relearning twenty names is work with nothing to show for it.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.build.clear();
    }

    /// What the collection costs, the key table included.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.rows.memory_bytes() + self.keys.memory_bytes()
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
