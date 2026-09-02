//! The properties of nodes and of edges, which are documents (`11` section 3).
//!
//! A property graph is an adjacency structure and a pile of key value maps, and
//! the second half is the part that engines get wrong. Neo4j gives every
//! property its own store and pays a pointer chase per read. FalkorDB keeps a
//! matrix of attribute vectors. Both are a second data model built to hold what
//! the document model already holds, and both then need their own indexes,
//! their own encoding and their own answer to nested values.
//!
//! There is a document model here, so a node's properties are a document and an
//! edge's properties are a document. That is not a saving in lines of code, it
//! is what makes `#[yo(index)]` on a node's field mean the same thing as
//! `#[yo(index)]` on a document's field, and it is why a graph gets path
//! indexes, key interning and nested values without any of the three being
//! written twice.
//!
//! ```
//! use yo_doc::Builder;
//! use yo_graph::Props;
//!
//! let mut b = Builder::new();
//! b.begin_object()?;
//! b.key(b"name")?;
//! b.text("ada")?;
//! b.end_object()?;
//! let doc = b.finish()?.to_vec();
//!
//! let mut people = Props::new();
//! people.put(41_920, &doc)?;
//! let got = people.get(41_920).expect("stored");
//! assert_eq!(got.get(b"name").and_then(|n| n.as_text()), Some("ada"));
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # Interning is worth more here than anywhere else
//!
//! A document collection repeats its field names on every document, which is
//! what key interning is for. A graph repeats them harder: an edge property map
//! is two or three fields and there are ten to a hundred times as many edges as
//! nodes, so the names are most of what an edge property store weighs. Two byte
//! ids against a `weight` and a `since` on fifty million edges is the difference
//! between the properties fitting beside the adjacency and not.
//!
//! # Why the id is bytes
//!
//! [`Docs`] is keyed by bytes because that is what a document collection is
//! keyed by, and a node id here is a `u64`, so it becomes eight bytes. They are
//! big endian, which costs a byte swap that no lookup notices and buys the one
//! thing byte order can buy: if this ever grows an ordered scan, the keys sort
//! the way the numbers do.

use yo_common::Result;
use yo_doc::{Doc, Docs, IndexKind, Key, Keys, Value};

/// The properties of a set of nodes, or of a set of edges.
///
/// One of these holds documents under integer ids, with whatever path indexes
/// the caller declared. A [`crate::Graph`] has two: one keyed by node id and one
/// keyed by edge slot.
#[derive(Debug, Default)]
pub struct Props {
    docs: Docs,
}

/// A node id or an edge slot, as the bytes a document collection is keyed by.
///
/// Big endian, so that the keys sort the way the numbers do. Eight bytes for
/// both, because an edge slot is a `u32` and widening it here means the two
/// stores are the same shape and a later move to `u64` slots is not a format
/// change.
#[must_use]
pub fn id_key(id: u64) -> [u8; 8] {
    id.to_be_bytes()
}

impl Props {
    /// An empty store.
    pub fn new() -> Props {
        Props { docs: Docs::new() }
    }

    /// Stores `doc` under `id`, replacing whatever was there.
    ///
    /// Answers whether this was new, which is what a caller counting nodes
    /// wants and what an overwrite is not.
    ///
    /// # Errors
    ///
    /// Whatever [`Docs::put_bytes`] answers: the document is malformed, or a
    /// value that an index covers is too long to be an index key. In the second
    /// case nothing is stored, so a document is never in the collection and out
    /// of its own indexes.
    pub fn put(&mut self, id: u64, doc: &[u8]) -> Result<bool> {
        self.docs.put_bytes(&id_key(id), doc)
    }

    /// The same from a value that is already read.
    ///
    /// # Errors
    ///
    /// The same as [`Props::put`].
    pub fn put_value(&mut self, id: u64, value: Value<'_>) -> Result<bool> {
        self.docs.put(&id_key(id), value)
    }

    /// What is stored under `id`.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<Doc<'_>> {
        self.docs.get(&id_key(id))
    }

    /// The stored bytes, for a caller that is about to write them somewhere
    /// rather than read them.
    #[must_use]
    pub fn bytes(&self, id: u64) -> Option<&[u8]> {
        self.docs.bytes(&id_key(id))
    }

    /// Whether anything is stored under `id`.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.docs.contains(&id_key(id))
    }

    /// Takes out what is under `id`, and says whether there was anything.
    pub fn remove(&mut self, id: u64) -> bool {
        self.docs.remove(&id_key(id))
    }

    /// How many ids have properties.
    ///
    /// Not how many nodes the graph has. A node with no properties is a node,
    /// and it is not in here.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether nothing has properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Declares an index over `path`, and backfills it over what is already
    /// stored.
    ///
    /// # Errors
    ///
    /// Whatever [`Docs::create_index_bytes`] answers: the path does not parse,
    /// it is already indexed, or a document already here has a value under it
    /// that cannot be an index key.
    pub fn create_index(&mut self, path: &str, kind: IndexKind) -> Result<()> {
        self.docs.create_index_bytes(path.as_bytes(), kind)
    }

    /// Drops the index over `path`, and says whether there was one.
    pub fn drop_index(&mut self, path: &str) -> bool {
        self.docs.drop_index(path)
    }

    /// Calls `f` for every id whose document has `key` under `path`, and
    /// answers how many that was.
    ///
    /// # Errors
    ///
    /// [`yo_common::Code::Invalid`] if `path` is not indexed, because a query
    /// that would have to scan the whole store is a mistake rather than a slow
    /// answer.
    pub fn find(&self, path: &str, key: &Key, mut f: impl FnMut(u64, Doc<'_>)) -> Result<usize> {
        self.docs.find(path, key, |id, doc| {
            if let Some(id) = read_key(id) {
                f(id, doc);
            }
        })
    }

    /// How many ids have `key` under `path`, without reading any of them.
    ///
    /// # Errors
    ///
    /// The same as [`Props::find`].
    pub fn count(&self, path: &str, key: &Key) -> Result<usize> {
        self.docs.count(path, key)
    }

    /// Every id with properties, and what they are.
    pub fn iter(&self) -> impl Iterator<Item = (u64, Doc<'_>)> {
        self.docs
            .iter()
            .filter_map(|(id, doc)| read_key(id).map(|id| (id, doc)))
    }

    /// The key table, which is where the field names went.
    #[must_use]
    pub fn keys(&self) -> &Keys {
        self.docs.keys()
    }

    /// Takes everything out.
    pub fn clear(&mut self) {
        self.docs.clear();
    }

    /// What this store weighs.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.docs.memory_bytes()
    }

    /// The collection underneath, for the range and scan operations that are
    /// the document model's rather than the graph's.
    #[must_use]
    pub fn docs(&self) -> &Docs {
        &self.docs
    }
}

/// The id a key stands for, or `None` if it is not one this store wrote.
///
/// It cannot be anything else today, since every write goes through
/// [`Props::put`]. It is checked rather than asserted because the alternative is
/// a panic in a callback that a caller has no way to guard against, and a key of
/// the wrong length means the collection was handed to something that is not
/// this.
fn read_key(k: &[u8]) -> Option<u64> {
    let raw: [u8; 8] = k.try_into().ok()?;
    Some(u64::from_be_bytes(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_doc::Builder;

    fn person(name: &str, city: &str, age: i64) -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().unwrap();
        b.key(b"age").unwrap();
        b.int(age).unwrap();
        b.key(b"city").unwrap();
        b.text(city).unwrap();
        b.key(b"name").unwrap();
        b.text(name).unwrap();
        b.end_object().unwrap();
        b.finish().unwrap().to_vec()
    }

    #[test]
    fn a_node_keeps_its_properties() {
        let mut p = Props::new();
        assert!(p.put(1, &person("ada", "london", 36)).unwrap());
        assert!(
            !p.put(1, &person("ada", "turin", 37)).unwrap(),
            "an overwrite is not new"
        );
        assert_eq!(p.len(), 1);
        let got = p.get(1).expect("stored");
        assert_eq!(got.get(b"city").and_then(|c| c.as_text()), Some("turin"));
        assert_eq!(got.get(b"age").and_then(|a| a.as_int()), Some(37));
    }

    #[test]
    fn an_id_that_was_never_written_has_nothing() {
        let mut p = Props::new();
        p.put(1, &person("ada", "london", 36)).unwrap();
        assert!(p.get(2).is_none());
        assert!(!p.contains(2));
        assert!(!p.remove(2));
        assert!(p.remove(1));
        assert!(p.is_empty());
    }

    #[test]
    fn the_whole_range_of_a_u64_id_works() {
        // Zero, the top of a u32 either side, and the top of a u64. A key that
        // was truncated to four bytes or read at the wrong width would collide
        // two of these.
        let ids = [
            0u64,
            1,
            u64::from(u32::MAX) - 1,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            u64::MAX,
        ];
        let mut p = Props::new();
        for (i, id) in ids.into_iter().enumerate() {
            assert!(
                p.put(id, &person(&format!("n{i}"), "here", i as i64))
                    .unwrap()
            );
        }
        assert_eq!(p.len(), ids.len());
        for (i, id) in ids.into_iter().enumerate() {
            let got = p.get(id).unwrap_or_else(|| panic!("{id} is missing"));
            assert_eq!(
                got.get(b"name").and_then(|n| n.as_text()),
                Some(&*format!("n{i}"))
            );
        }
    }

    #[test]
    fn an_index_finds_nodes_by_a_property() {
        let mut p = Props::new();
        p.create_index("$.city", IndexKind::Equality).unwrap();
        p.put(1, &person("ada", "london", 36)).unwrap();
        p.put(2, &person("grace", "london", 45)).unwrap();
        p.put(3, &person("edsger", "austin", 51)).unwrap();

        let mut found = Vec::new();
        let n = p
            .find("$.city", &Key::text("london"), |id, _| found.push(id))
            .unwrap();
        assert_eq!(n, 2);
        found.sort_unstable();
        assert_eq!(found, vec![1, 2]);
        assert_eq!(p.count("$.city", &Key::text("austin")).unwrap(), 1);
    }

    #[test]
    fn an_index_declared_after_the_fact_backfills() {
        let mut p = Props::new();
        p.put(1, &person("ada", "london", 36)).unwrap();
        p.put(2, &person("grace", "london", 45)).unwrap();
        p.create_index("$.city", IndexKind::Equality).unwrap();
        assert_eq!(p.count("$.city", &Key::text("london")).unwrap(), 2);

        // And a removal takes the node out of the index, not only out of the
        // store, which is the failure that shows up as a query answering ids
        // that are not there any more.
        assert!(p.remove(1));
        assert_eq!(p.count("$.city", &Key::text("london")).unwrap(), 1);
    }

    #[test]
    fn the_field_names_are_stored_once() {
        let mut p = Props::new();
        for id in 0..100u64 {
            p.put(id, &person("someone", "london", id as i64)).unwrap();
        }
        // Three names, whatever the document count is. That is the whole point
        // of interning and it is worth an assertion rather than a comment.
        assert_eq!(p.keys().len(), 3);
        assert!(p.get(7).expect("stored").value().is_interned());
    }

    #[test]
    fn iterating_gives_back_the_ids_that_went_in() {
        let mut p = Props::new();
        for id in [5u64, 9, 41_920] {
            p.put(id, &person("someone", "london", 1)).unwrap();
        }
        let mut ids: Vec<u64> = p.iter().map(|(id, _)| id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![5, 9, 41_920]);
    }
}
