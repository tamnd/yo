//! `Vectors`, the collection of embeddings, and the search over it (`10` and
//! `15` section 2).
//!
//! A vector collection is a name, a dimension and a metric. Something goes in
//! under a key, the same thing comes back under that key, and a query vector
//! gets the nearest keys to it. There is no index to create, no probe list to
//! tune and no build step: the index is maintained as the collection is written
//! to, in bounded pieces, which is what `10` section 5 is about and is the whole
//! reason the index under here is partitions rather than a graph.
//!
//! ```
//! let db = yo::open(yo::MEMORY)?;
//! let v = db.vectors("passages", 3)?;
//!
//! v.put("a", &[1.0, 0.0, 0.0])?;
//! v.put("b", &[0.0, 1.0, 0.0])?;
//!
//! let hits = v.search(&[0.9, 0.1, 0.0], 1)?;
//! assert_eq!(hits[0].key, b"a".to_vec());
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! # What a search costs, and why it is exact at the end
//!
//! The searchable form of a vector is a RaBitQ code, which is a bit per
//! dimension, so a collection of 768 dimensional embeddings is 96 bytes a vector
//! to scan rather than 3072. The codes pick the candidates and then the full
//! precision vectors settle the order, so a hit's distance is the real distance
//! and not an estimate, and the only thing quantisation can cost is a near miss
//! that never made the shortlist. `yo-vector`'s recall tables are the measured
//! version of that sentence.
//!
//! # The metric decides what is stored
//!
//! [`Metric::L2`] stores the vector it was given. [`Metric::Cosine`] stores the
//! unit vector, because the index measures distance and on unit vectors the
//! nearest by distance is the nearest by angle, which means cosine costs one
//! normalisation on the way in rather than a different index. That is also what
//! comes back out of [`Vectors::get`], and it is the same answer Redis gives
//! for a cosine vector set.
//!
//! [`Metric::Ip`] and [`Metric::Hamming`] are refused rather than approximated.
//! Inner product is not a distance, so ordering by it is not ordering by
//! nearness and the partitions would be built around the wrong question, and
//! Hamming wants binary vectors that this collection does not hold yet.
//!
//! # Where the vectors live
//!
//! Beside the index, one flat run of floats per collection, which is the shape
//! `06` gives them: a vector is a record like any other and the rerank is a read
//! at an address the id already resolves to. This build holds that run in
//! memory, exactly as every other collection here is held in memory, and the
//! record kind it becomes on disk is already written down as
//! `yo_format::vector`. Nothing on this page changes when the file arrives.

use yo_common::{Code, Error, Result};
use yo_format::vector::MAX_DIM;
use yo_kv::Elements;
use yo_shape::Metric;
use yo_vector::{Bits, Partitions, Tuning, Vectors as Source};

use crate::db::Handle;

/// The seed the rotation is built from.
///
/// Fixed rather than random, because the codes in a collection are only
/// comparable to each other if they were rotated the same way, so this is a
/// property of the collection and it belongs in the catalogue when the file
/// lands. Until then a constant is the honest version of the same thing: two
/// runs of the same program build the same index.
const SEED: u64 = 0x596F_5F76_6563_0001;

/// How many vectors one write is willing to have maintenance touch.
///
/// Splits and merges run inside the write that made them necessary, because in
/// this build the calling thread is the only thread there is (`15` section 7).
/// A budget of four postings means a write that triggers a split pays for that
/// split and nothing more, and a collection that has fallen behind catches up
/// over the writes that follow rather than in one pause. The served mode has a
/// maintenance slice for this and it will call the same method with a bigger
/// number.
const BUDGET: usize = 1024;

/// One answer from a search.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The key the vector was put under.
    pub key: Vec<u8>,
    /// How far it is, measured against the full precision vector rather than
    /// against its code.
    ///
    /// For [`Metric::L2`] that is the euclidean distance. For
    /// [`Metric::Cosine`] it is one minus the cosine similarity, so 0 is the
    /// same direction and 2 is the opposite one. Both are distances, so nearer
    /// is smaller and the answers come back in that order.
    pub distance: f32,
}

/// A collection of vectors, reached by [`Db::vectors`](crate::Db::vectors).
///
/// Keys are byte strings the way the keyspace's are, so anything that is bytes
/// will do.
#[derive(Clone)]
pub struct Vectors {
    pub(crate) db: Handle,
    pub(crate) at: usize,
    pub(crate) dim: usize,
    pub(crate) metric: Metric,
}

impl core::fmt::Debug for Vectors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vectors")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .finish_non_exhaustive()
    }
}

impl Vectors {
    /// How many coordinates a vector in this collection has.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// What nearness means here.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Put a vector in under `key`, and say whether the key is new.
    ///
    /// Replacing is the same call, which is what makes re-embedding a document
    /// one line. The old code comes out of its partition and the new one goes
    /// into whichever partition it belongs to now, so nothing accumulates and
    /// there is no rebuild waiting at the end of it.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the vector is not [`Vectors::dim`] long, when it
    /// holds a coordinate that is not a number, or when a cosine collection is
    /// handed a vector of length zero, which has no direction to store.
    /// [`Code::Full`] for a key past the length limit.
    pub fn put(&self, key: impl AsRef<[u8]>, v: &[f32]) -> Result<bool> {
        let key = key.as_ref();
        self.db
            .write(|inner| inner.collections[self.at].data.vectors_mut().put(key, v))
    }

    /// The vector under `key`, if there is one.
    ///
    /// A cosine collection hands back the unit vector it stored rather than the
    /// one it was given. See the module note on the metric for why.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn get(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<f32>>> {
        self.with(key, <[f32]>::to_vec)
    }

    /// The same, handed to a closure where it lies, which copies nothing.
    ///
    /// The owned form is what most code wants and this is Y29's other half: the
    /// vectors are contiguous floats already, so a caller that only wants to
    /// measure one against something never has to allocate to do it.
    ///
    /// # Errors
    ///
    /// As [`Vectors::get`].
    pub fn with<R>(&self, key: impl AsRef<[u8]>, f: impl FnOnce(&[f32]) -> R) -> Result<Option<R>> {
        let key = key.as_ref();
        self.db
            .read(|inner| Ok(inner.collections[self.at].data.vectors().get(key).map(f)))
    }

    /// Whether the collection holds a vector under `key`.
    ///
    /// # Errors
    ///
    /// As [`Vectors::get`].
    pub fn contains(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        let key = key.as_ref();
        self.db
            .read(|inner| Ok(inner.collections[self.at].data.vectors().contains(key)))
    }

    /// Take a vector out, saying whether it was there.
    ///
    /// A delete here is a delete and not a tombstone: the member leaves its
    /// posting and the last member of that posting moves into the hole. That is
    /// the difference between this and a graph index, where deletes pile up
    /// until somebody rebuilds.
    ///
    /// # Errors
    ///
    /// As [`Vectors::get`].
    pub fn remove(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        let key = key.as_ref();
        self.db
            .write(|inner| Ok(inner.collections[self.at].data.vectors_mut().remove(key)))
    }

    /// How many vectors are in the collection.
    ///
    /// # Errors
    ///
    /// As [`Vectors::get`].
    pub fn len(&self) -> Result<usize> {
        self.db
            .read(|inner| Ok(inner.collections[self.at].data.vectors().len()))
    }

    /// Whether there are none.
    ///
    /// # Errors
    ///
    /// As [`Vectors::get`].
    pub fn is_empty(&self) -> Result<bool> {
        self.len().map(|n| n == 0)
    }

    /// The `k` nearest keys to `q`, nearest first.
    ///
    /// Fewer than `k` come back when the collection holds fewer than that, and
    /// nothing comes back from an empty one.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let v = db.vectors("words", 2)?;
    ///
    /// v.put("north", &[0.0, 1.0])?;
    /// v.put("east", &[1.0, 0.0])?;
    /// v.put("west", &[-1.0, 0.0])?;
    ///
    /// let hits = v.search(&[0.2, 0.9], 2)?;
    /// assert_eq!(hits[0].key, b"north".to_vec());
    /// assert_eq!(hits.len(), 2);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when `q` is not [`Vectors::dim`] long or holds a
    /// coordinate that is not a number.
    pub fn search(&self, q: &[f32], k: usize) -> Result<Vec<Match>> {
        self.db
            .read(|inner| inner.collections[self.at].data.vectors().search(q, k, None))
    }

    /// The `k` nearest keys to the vector already stored under `key`, which is
    /// the more-like-this search.
    ///
    /// `key` itself is never one of the answers, because it is always the
    /// nearest and nobody asked what a thing is most similar to itself.
    ///
    /// # Errors
    ///
    /// [`Code::NotFound`] when nothing is stored under `key`, because an empty
    /// answer would otherwise mean both "no such key" and "nothing near it".
    pub fn near(&self, key: impl AsRef<[u8]>, k: usize) -> Result<Vec<Match>> {
        let key = key.as_ref();
        self.db.read(|inner| {
            let store = inner.collections[self.at].data.vectors();
            let Some(q) = store.get(key) else {
                return Err(Error::fmt(
                    Code::NotFound,
                    format_args!(
                        "this collection has no vector under that key, so there is nothing to be near to"
                    ),
                ));
            };
            store.search(q, k, Some(key))
        })
    }
}

/// A vector collection: the index, the vectors it reranks against, and the keys
/// they are known by.
pub(crate) struct Store {
    /// The RaBitQ codes under partitions that split and merge in place.
    index: Partitions,
    /// The full precision vectors, which the last step of every search measures
    /// against and which the index itself never holds.
    raw: Raw,
    /// Key to the id the index knows it by.
    ids: Elements<u64>,
    metric: Metric,
}

impl Store {
    /// An empty collection that has allocated nothing yet.
    pub(crate) fn new(dim: usize, metric: Metric) -> Store {
        Store {
            // One bit rather than four. Four bits costs four times the scan and
            // on both public datasets it reaches the same recall to four
            // decimal places once the rerank is 16 wide, which says the
            // estimator is not what limits recall on real embeddings.
            index: Partitions::new(dim, Bits::One, SEED, Tuning::default()),
            raw: Raw {
                dim,
                data: Vec::new(),
                owner: Vec::new(),
                free: Vec::new(),
            },
            ids: Elements::new(),
            metric,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn contains(&self, key: &[u8]) -> bool {
        self.ids.contains(key)
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<&[f32]> {
        let id = *self.ids.get(key)?;
        Some(self.raw.at(id))
    }

    pub(crate) fn put(&mut self, key: &[u8], v: &[f32]) -> Result<bool> {
        let mut ready = self.check(v)?;
        if self.metric == Metric::Cosine {
            normalize(&mut ready)?;
        }

        let new = match self.ids.get(key) {
            Some(&id) => {
                self.raw.write(id, &ready);
                self.index.insert(id, &ready);
                false
            }
            None => {
                let id = self.raw.take(key, &ready);
                if self.ids.insert(key, id).is_err() {
                    self.raw.release(id);
                    return Err(Error::fmt(
                        Code::Full,
                        format_args!("that key is too long for a vector collection"),
                    ));
                }
                self.index.insert(id, &ready);
                true
            }
        };

        // Inside the write rather than after a threshold of them, because a
        // split that is owed is a partition that is already twice the size it
        // wants to be and every search until it happens reads all of it.
        if self.index.needs_maintenance() {
            self.index.maintain(&self.raw, BUDGET);
        }
        Ok(new)
    }

    pub(crate) fn remove(&mut self, key: &[u8]) -> bool {
        let Some(id) = self.ids.remove(key) else {
            return false;
        };
        self.index.remove(id);
        self.raw.release(id);
        if self.index.needs_maintenance() {
            self.index.maintain(&self.raw, BUDGET);
        }
        true
    }

    /// The `k` nearest keys to `q`, with `skip` left out of the answer.
    pub(crate) fn search(&self, q: &[f32], k: usize, skip: Option<&[u8]>) -> Result<Vec<Match>> {
        // The query is checked before the collection is looked at, so that a
        // query of the wrong length says so rather than answering nothing at
        // all while the collection happens to be empty.
        let mut ready = self.check(q)?;
        if k == 0 || self.index.is_empty() {
            return Ok(Vec::new());
        }
        if self.metric == Metric::Cosine {
            normalize(&mut ready)?;
        }

        // One more than asked for when a key is being left out, so that leaving
        // it out does not cost an answer. It is the nearest one and it is
        // therefore always in the shortlist.
        let want = if skip.is_some() { k + 1 } else { k };
        let hits = self.index.search(&ready, want, &self.raw);

        let mut out = Vec::with_capacity(hits.len().min(k));
        for hit in hits {
            let key = self.raw.owner(hit.id);
            if skip == Some(key) {
                continue;
            }
            out.push(Match {
                key: key.to_vec(),
                distance: self.report(hit.distance),
            });
            if out.len() == k {
                break;
            }
        }
        Ok(out)
    }

    /// What the collection is holding, the vectors and the codes and the keys.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.raw.memory_bytes() + self.index.code_bytes() + self.ids.memory_bytes()
    }

    /// A vector this collection can take, copied so the caller's is untouched.
    fn check(&self, v: &[f32]) -> Result<Vec<f32>> {
        if v.len() != self.raw.dim {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!(
                    "this collection holds {} dimensional vectors and was handed {}",
                    self.raw.dim,
                    v.len()
                ),
            ));
        }
        if let Some(at) = v.iter().position(|x| !x.is_finite()) {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!(
                    "coordinate {at} of that vector is {}, and a distance to it would be one too",
                    v[at]
                ),
            ));
        }
        Ok(v.to_vec())
    }

    /// The distance to report for the squared one the index measured.
    ///
    /// The index works in squared euclidean distance because a square root
    /// changes no ordering and costs one per candidate. The caller asked in the
    /// metric they opened the collection with, so the square root happens here,
    /// `k` times rather than once per candidate scanned.
    fn report(&self, sq: f32) -> f32 {
        match self.metric {
            // On unit vectors the squared distance is 2 - 2cos, so half of it
            // is one minus the cosine similarity, which is the number every
            // cosine API in the world reports.
            Metric::Cosine => (sq / 2.0).clamp(0.0, 2.0),
            _ => sq.max(0.0).sqrt(),
        }
    }
}

/// The full precision vectors, one flat run of floats with a slot per id.
///
/// A slot is `dim` floats at `id * dim` and an id is never anything but the slot
/// it names, so reading a vector for the rerank is an offset rather than a
/// lookup. Slots come back for reuse when a key is removed, which is what keeps
/// a collection that is rewritten forever from growing forever.
struct Raw {
    dim: usize,
    data: Vec<f32>,
    /// The key each slot holds, and `None` for a slot that is free.
    ///
    /// This is the second copy of a key, the first being the one in
    /// [`Store::ids`], and it is here because a search comes back with ids and
    /// has to answer in keys. A key is tens of bytes against a vector's
    /// thousands, so the copy is worth more than the indirection that would
    /// avoid it.
    owner: Vec<Option<Box<[u8]>>>,
    free: Vec<u64>,
}

impl Raw {
    /// The vector in slot `id`.
    fn at(&self, id: u64) -> &[f32] {
        let at = id as usize * self.dim;
        &self.data[at..at + self.dim]
    }

    /// The key slot `id` was taken for.
    fn owner(&self, id: u64) -> &[u8] {
        self.owner[id as usize]
            .as_deref()
            .expect("a live id has a key, and a search only answers with live ids")
    }

    /// Put `v` in a free slot for `key`, or in a new one, and say which.
    fn take(&mut self, key: &[u8], v: &[f32]) -> u64 {
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                self.data.resize(self.data.len() + self.dim, 0.0);
                self.owner.push(None);
                (self.owner.len() - 1) as u64
            }
        };
        self.owner[id as usize] = Some(key.into());
        self.write(id, v);
        id
    }

    /// Overwrite the vector in a slot that is already taken.
    fn write(&mut self, id: u64, v: &[f32]) {
        let at = id as usize * self.dim;
        self.data[at..at + self.dim].copy_from_slice(v);
    }

    /// Give a slot back.
    fn release(&mut self, id: u64) {
        self.owner[id as usize] = None;
        self.free.push(id);
    }

    fn memory_bytes(&self) -> usize {
        self.data.capacity() * size_of::<f32>()
            + self.owner.capacity() * size_of::<Option<Box<[u8]>>>()
            + self
                .owner
                .iter()
                .map(|k| k.as_ref().map_or(0, |k| k.len()))
                .sum::<usize>()
            + self.free.capacity() * size_of::<u64>()
    }
}

impl Source for Raw {
    fn get(&self, id: u64, into: &mut [f32]) -> bool {
        let Some(Some(_)) = self.owner.get(id as usize) else {
            return false;
        };
        into.copy_from_slice(self.at(id));
        true
    }
}

/// Turn a vector into the unit vector pointing the same way.
fn normalize(v: &mut [f32]) -> Result<()> {
    let norm = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>();
    if norm <= 0.0 {
        return Err(Error::new(
            Code::Invalid,
            "a cosine collection compares directions and a vector of length zero has none",
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let scale = norm.sqrt().recip() as f32;
    for x in v.iter_mut() {
        *x *= scale;
    }
    Ok(())
}

/// Whether `dim` is a dimension a collection can be opened with.
pub(crate) fn dimension(dim: usize) -> Result<u32> {
    if dim == 0 || dim > MAX_DIM {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!(
                "a vector collection holds between 1 and {MAX_DIM} dimensions, and {dim} is not one of them"
            ),
        ));
    }
    u32::try_from(dim).map_err(|_| Error::new(Code::Invalid, "that dimension does not fit"))
}

/// Whether `metric` is one this build can measure.
pub(crate) fn metric(metric: Metric) -> Result<()> {
    match metric {
        Metric::L2 | Metric::Cosine => Ok(()),
        Metric::Ip => Err(Error::new(
            Code::Unsupported,
            "inner product is not a distance, so a partition index cannot be built around it, and a collection that ordered by it would not be ordering by nearness. Normalise the vectors and open the collection with cosine, which is the same ranking",
        )),
        Metric::Hamming => Err(Error::new(
            Code::Unsupported,
            "hamming distance is for binary vectors and this collection holds floats",
        )),
    }
}

#[cfg(test)]
mod tests {
    use crate::db::{MEMORY, open};

    use super::*;

    /// Three vectors on the axes, which makes every answer obvious by eye.
    fn axes(v: &Vectors) {
        v.put("x", &[1.0, 0.0, 0.0]).unwrap();
        v.put("y", &[0.0, 1.0, 0.0]).unwrap();
        v.put("z", &[0.0, 0.0, 1.0]).unwrap();
    }

    #[test]
    fn a_vector_comes_back_the_way_it_went_in() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        assert!(v.is_empty().unwrap());

        assert!(v.put("x", &[1.0, 2.0, 3.0]).unwrap(), "the key is new");
        assert!(!v.put("x", &[1.0, 2.0, 3.0]).unwrap(), "and then it is not");
        assert_eq!(v.get("x").unwrap(), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(v.len().unwrap(), 1);
        assert!(v.contains("x").unwrap());
        assert_eq!(v.get("nobody").unwrap(), None);
        assert_eq!(v.with("x", |x| x.len()).unwrap(), Some(3));
    }

    #[test]
    fn the_nearest_answer_is_the_nearest_vector() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        axes(&v);

        let hits = v.search(&[0.9, 0.2, 0.1], 3).unwrap();
        let keys: Vec<&[u8]> = hits.iter().map(|h| h.key.as_slice()).collect();
        assert_eq!(keys, vec![&b"x"[..], &b"y"[..], &b"z"[..]]);
        assert!(hits[0].distance < hits[1].distance);
        // The exact euclidean distance and not the estimate the codes gave,
        // which is the whole point of reranking against the stored vector.
        let want = (0.01f32 + 0.04 + 0.01).sqrt();
        assert!((hits[0].distance - want).abs() < 1e-6, "{hits:?}");
    }

    #[test]
    fn asking_for_more_than_there_is_gets_what_there_is() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        assert!(v.search(&[1.0, 0.0, 0.0], 4).unwrap().is_empty());
        axes(&v);
        assert_eq!(v.search(&[1.0, 0.0, 0.0], 10).unwrap().len(), 3);
        assert!(v.search(&[1.0, 0.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn a_removed_vector_is_not_an_answer_and_its_slot_comes_back() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        axes(&v);

        assert!(v.remove("x").unwrap());
        assert!(!v.remove("x").unwrap(), "twice is not there twice");
        assert_eq!(v.len().unwrap(), 2);
        assert!(!v.contains("x").unwrap());

        let hits = v.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.key != b"x".to_vec()));

        v.put("w", &[1.0, 0.0, 0.0]).unwrap();
        let hits = v.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].key, b"w".to_vec(), "the reused slot answers as w");
    }

    /// Replacing has to take the old code out of its partition as well as
    /// writing the new vector, or the search answers with a key whose vector
    /// moved somewhere else.
    #[test]
    fn a_replaced_vector_is_searched_at_its_new_place() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        axes(&v);

        v.put("x", &[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(v.len().unwrap(), 3, "a replacement is not a second key");

        let hits = v.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].key, b"y".to_vec(), "x moved away from that corner");
        let hits = v.search(&[0.0, 0.0, 1.0], 2).unwrap();
        let keys: Vec<Vec<u8>> = hits.into_iter().map(|h| h.key).collect();
        assert!(keys.contains(&b"x".to_vec()) && keys.contains(&b"z".to_vec()));
    }

    #[test]
    fn more_like_this_leaves_the_thing_itself_out() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        axes(&v);
        v.put("x2", &[0.9, 0.1, 0.0]).unwrap();

        let hits = v.near("x", 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].key, b"x2".to_vec());
        assert!(hits.iter().all(|h| h.key != b"x".to_vec()));

        let e = v.near("nobody", 2).expect_err("no such key");
        assert_eq!(e.code(), Code::NotFound);
    }

    #[test]
    fn a_cosine_collection_stores_the_direction_and_reports_the_angle() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors_with("e", 2, Metric::Cosine).unwrap();

        v.put("east", &[7.0, 0.0]).unwrap();
        v.put("north", &[0.0, 3.0]).unwrap();
        v.put("west", &[-2.0, 0.0]).unwrap();
        assert_eq!(v.get("east").unwrap(), Some(vec![1.0, 0.0]));

        // Length is nothing to a cosine collection, so a long east and a short
        // east are the same vector and both are nearer than north.
        let hits = v.search(&[100.0, 0.0], 3).unwrap();
        assert_eq!(hits[0].key, b"east".to_vec());
        assert!(hits[0].distance.abs() < 1e-6, "{hits:?}");
        assert!(
            (hits[1].distance - 1.0).abs() < 1e-6,
            "north is a right angle"
        );
        assert!(
            (hits[2].distance - 2.0).abs() < 1e-6,
            "west is the opposite"
        );

        let e = v.put("nowhere", &[0.0, 0.0]).expect_err("no direction");
        assert_eq!(e.code(), Code::Invalid);
    }

    #[test]
    fn a_vector_of_the_wrong_length_or_shape_is_refused() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();

        let e = v.put("x", &[1.0, 2.0]).expect_err("two is not three");
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("3 dimensional"), "{e}");

        let e = v.put("x", &[1.0, f32::NAN, 2.0]).expect_err("not a number");
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("coordinate 1"), "{e}");

        let e = v.search(&[1.0], 1).expect_err("one is not three");
        assert_eq!(e.code(), Code::Invalid);
    }

    /// Enough vectors that the index has actually split, because everything
    /// above runs inside one partition and a one partition index is a scan.
    #[test]
    fn recall_holds_once_the_index_has_split() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 8).unwrap();

        let mut seed = 0x2026u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut all: Vec<Vec<f32>> = Vec::new();
        for i in 0..2000usize {
            let x: Vec<f32> = (0..8).map(|_| next()).collect();
            v.put(format!("k{i}"), &x).unwrap();
            all.push(x);
        }

        let mut found = 0;
        for (i, q) in all.iter().enumerate().step_by(50) {
            let hits = v.search(q, 1).unwrap();
            if hits[0].key == format!("k{i}").into_bytes() {
                found += 1;
            }
        }
        assert!(found >= 39, "{found} of 40 queries found their own vector");
        assert_eq!(v.len().unwrap(), 2000);
        assert!(db.memory_bytes().unwrap() > 2000 * 8 * 4);
    }

    #[test]
    fn a_dimension_or_a_metric_the_build_cannot_hold_is_refused_at_open() {
        let db = open(MEMORY).unwrap();

        let e = db.vectors("e", 0).expect_err("zero dimensions is nothing");
        assert_eq!(e.code(), Code::Invalid);
        let e = db.vectors("e", MAX_DIM + 1).expect_err("past the limit");
        assert_eq!(e.code(), Code::Invalid);

        let e = db
            .vectors_with("e", 8, Metric::Ip)
            .expect_err("not a distance");
        assert_eq!(e.code(), Code::Unsupported);
        assert!(e.message().contains("cosine"), "{e}");
        let e = db
            .vectors_with("e", 8, Metric::Hamming)
            .expect_err("not floats");
        assert_eq!(e.code(), Code::Unsupported);
    }

    /// The dimension and the metric are the collection's shape, so opening the
    /// same name with either of them changed is the same refusal a map gets for
    /// the wrong value type.
    #[test]
    fn the_dimension_and_the_metric_are_part_of_the_shape() {
        let db = open(MEMORY).unwrap();
        let v = db.vectors("e", 3).unwrap();
        v.put("x", &[1.0, 0.0, 0.0]).unwrap();

        let same = db.vectors("e", 3).unwrap();
        assert_eq!(same.len().unwrap(), 1, "the same name is the same store");

        let e = db.vectors("e", 4).expect_err("that is another collection");
        assert_eq!(e.code(), Code::ShapeMismatch);
        let e = db
            .vectors_with("e", 3, Metric::Cosine)
            .expect_err("and so is that");
        assert_eq!(e.code(), Code::ShapeMismatch);

        let e = db.map::<String, u64>("e").expect_err("and so is a map");
        assert_eq!(e.code(), Code::ShapeMismatch);
    }
}
