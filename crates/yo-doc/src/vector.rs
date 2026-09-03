//! A vector index over a path into a document (`10` section 3).
//!
//! An embedding is a field like any other. A document that carries one at
//! `$.embedding` should be one write, and a nearest neighbour search over the
//! collection should be one call that hands back documents, not ids to go and
//! look up somewhere else. Every other engine makes this two stores joined on
//! the id by the caller, and the join is where the two of them drift apart.
//!
//! ```
//! use yo_doc::{Builder, Docs, Key};
//!
//! let mut docs = Docs::new();
//! docs.create_index("$.lang")?;
//! docs.create_vector_index("$.embedding", 3)?;
//!
//! for (id, lang, v) in [
//!     ("a", "en", [1.0, 0.0, 0.0]),
//!     ("b", "fr", [0.9, 0.1, 0.0]),
//!     ("c", "en", [0.0, 0.0, 1.0]),
//! ] {
//!     let mut b = Builder::new();
//!     b.begin_object()?;
//!     b.key(b"lang")?;
//!     b.text(lang)?;
//!     b.key(b"embedding")?;
//!     b.begin_array()?;
//!     for x in v {
//!         b.float(x)?;
//!     }
//!     b.end_array()?;
//!     b.end_object()?;
//!     let bytes = b.finish()?.to_vec();
//!     docs.put_bytes(id.as_bytes(), &bytes)?;
//! }
//!
//! // Nearest overall, which is the French one.
//! let mut best = Vec::new();
//! docs.nearest("$.embedding", &[1.0, 0.05, 0.0], 2, |id, _, _| best.push(id.to_vec()))?;
//! assert_eq!(best[0], b"a".to_vec());
//! assert_eq!(best[1], b"b".to_vec());
//!
//! // Nearest among the English ones, decided inside the scan and not after it.
//! let mut found = Vec::new();
//! let english = [("$.lang", Key::text("en"))];
//! docs.nearest_where("$.embedding", &[1.0, 0.05, 0.0], 2, &english, |id, _, _| {
//!     found.push(id.to_vec())
//! })?;
//! assert_eq!(found, [b"a".to_vec(), b"c".to_vec()]);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # Why this is not a [`PathIndex`](crate::PathIndex)
//!
//! Every other index kind files a document under byte keys, and the lookup is
//! equality or a range over those bytes. Nearness is neither. There is no key a
//! query could ask for, the answer depends on all of the coordinates at once,
//! and the structure that answers it is a partitioned quantised index rather
//! than a table from key to posting list. So a vector index is a
//! [`Collection`], keyed by document id, held in its own list beside the path
//! indexes rather than pretending to be one.
//!
//! It is still the same [`Collection`] a vector set on the wire is, which is Y23
//! again: a document's embedding and a `VADD` land in the same code, so a
//! replace, a zero length vector and a dimension mismatch cannot be answered one
//! way here and another way there.
//!
//! # The filter is the point
//!
//! "The five nearest documents where the language is English" is the question
//! people actually have, and answering it by searching for fifty and then
//! throwing away the ones that are not English is a lottery. The more selective
//! the filter the worse the lottery, and it fails quietly: the answers that come
//! back are real, the ones that should have been there were never ranked.
//!
//! So the filter runs inside the posting scan. Every document carries a 64 bit
//! [`Signature`] over the keys its other indexes filed it under, which sits
//! beside the code in the posting and costs one instruction to test on a word
//! the scan has already loaded. [`nearest_where`](crate::Docs::nearest_where)
//! builds the same signature out of the values the query requires, and a
//! document is worth ranking when its bits cover the query's.
//!
//! Two values can land on the same bit, so the signature can let a document
//! through that does not really match. It can never reject one that does, which
//! is the direction that matters, and the caller's own predicate over the
//! answers settles the rest.
//!
//! Because the tag summarises the other indexes, it goes stale when the set of
//! indexes changes. Declaring or dropping an index therefore rewrites every tag,
//! which is one store per document per vector index with no requantising, rather
//! than leaving a filter that used to work quietly answering nothing.

use yo_common::{Code, Error, Result};
use yo_shape::Metric;
use yo_vector::{Collection, Signature};

use crate::head::Kind;
use crate::read::Value;

/// One path holding an embedding, and the collection its vectors live in.
///
/// The collection is keyed by document id, so an answer from it is a document id
/// and the caller never sees a second numbering.
#[derive(Debug)]
pub struct VectorIndex {
    path: Vec<u8>,
    c: Collection,
}

impl VectorIndex {
    /// An empty index over `path`, holding `dim` wide vectors measured by
    /// `metric`.
    pub(crate) fn new(path: &[u8], dim: usize, metric: Metric) -> Result<VectorIndex> {
        Ok(VectorIndex {
            path: path.to_vec(),
            c: Collection::new(dim, metric)?,
        })
    }

    /// The path the embedding is read from.
    #[must_use]
    pub fn path(&self) -> &[u8] {
        &self.path
    }

    /// How many coordinates a vector here has.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.c.dim()
    }

    /// What nearness means here.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.c.metric()
    }

    /// How many documents have a vector filed.
    ///
    /// A document with nothing at the path is not in here, so the difference
    /// between this and the collection's length is how many documents the index
    /// does not cover.
    #[must_use]
    pub fn len(&self) -> usize {
        self.c.len()
    }

    /// Whether none do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// What the index costs.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.c.memory_bytes()
    }

    /// The collection underneath, for a caller that wants the vector API rather
    /// than the document one.
    #[must_use]
    pub fn collection(&self) -> &Collection {
        &self.c
    }

    /// The same, to write through.
    pub(crate) fn collection_mut(&mut self) -> &mut Collection {
        &mut self.c
    }

    /// Start again with the same path, dimension and metric.
    ///
    /// A collection has no empty in place, because the quantiser's rotation and
    /// the partition layout are the collection. Building a new one is what
    /// emptying it means, and the dimension and metric were already checked when
    /// the index was declared.
    pub(crate) fn clear(&mut self) {
        self.c = Collection::new(self.c.dim(), self.c.metric())
            .expect("the dimension and the metric were accepted when the index was declared");
    }
}

/// Read the vector at `at` into `into`.
///
/// An array of `dim` numbers and nothing else. A path that holds something of
/// another shape fails the write rather than being skipped, because unlike a
/// scalar index there is no reading of "the document does not have one here":
/// the caller declared a path as an embedding and put something else there.
pub(crate) fn coordinates(
    at: Value<'_>,
    dim: usize,
    path: &[u8],
    into: &mut Vec<f32>,
) -> Result<()> {
    let wrong = || {
        Error::fmt(
            Code::Invalid,
            format_args!(
                "a value at {} is not an array of {dim} numbers",
                String::from_utf8_lossy(path)
            ),
        )
    };
    if at.kind() != Kind::Array || at.len() != dim {
        return Err(wrong());
    }
    into.clear();
    into.reserve(dim);
    for elem in at.iter() {
        match elem.kind() {
            Kind::Int => into.push(elem.as_int().ok_or_else(wrong)? as f32),
            Kind::Float => into.push(elem.as_float().ok_or_else(wrong)? as f32),
            _ => return Err(wrong()),
        }
    }
    Ok(())
}

/// The tag a document with these index key lists carries.
///
/// One bit per path and key pair, which is exactly what
/// [`Docs::nearest_where`](crate::Docs::nearest_where) builds on the other side.
/// A document with an array index files under several keys at one path and gets
/// a bit for each, so a query asking for any one of them still covers it.
pub(crate) fn tag_of<'a>(slots: impl Iterator<Item = (&'a [u8], &'a [u8])>) -> u64 {
    let mut sig = Signature::default();
    for (path, keys) in slots {
        add_keys(&mut sig, path, keys);
    }
    sig.bits()
}

/// Set the bit for every key in one index's key list.
///
/// The one place a path and a key become a bit, so that the tag a write puts on
/// a document, the tag a rebuild puts back, and the signature a query is filtered
/// by cannot drift apart.
pub(crate) fn add_keys(sig: &mut Signature, path: &[u8], keys: &[u8]) {
    crate::index::each_key(keys, |key| sig.insert_bytes(path, key));
}
