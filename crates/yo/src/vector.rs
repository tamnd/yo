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
//!
//! # What is on this page and what is not
//!
//! Only the handle. The collection itself is [`yo_vector::Collection`], one crate
//! down, because the vector commands on the wire need the same key table, the
//! same slab of floats and the same metric handling that this does. Two doors
//! into one store is Y23 and it is the reason `INCR` off a socket and
//! [`Db::counter`](crate::Db::counter) cannot drift apart either.

use yo_common::{Code, Error, Result};
use yo_shape::Metric;

use crate::db::Handle;

pub use yo_vector::Match;

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

#[cfg(test)]
mod tests {
    use yo_vector::collection::MAX_DIM;

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
