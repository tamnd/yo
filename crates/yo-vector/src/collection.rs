//! A collection of vectors under keys, which is what both doors reach.
//!
//! [`Partitions`] is the index and it knows about ids, dimensions and codes. It
//! does not know that a client calls a vector `doc:1`, that a collection has a
//! metric, or where the full precision vectors live, because none of those are
//! the index's business and putting them there would make one of the two callers
//! wrong. This is the piece above it that answers all three, so `db.vectors()`
//! and `VADD` off a socket are two doors into one store (Y23) rather than two
//! stores that agree for now.
//!
//! ```
//! use yo_vector::Collection;
//! use yo_shape::Metric;
//!
//! let mut c = Collection::new(3, Metric::L2)?;
//! c.put(b"a", &[1.0, 0.0, 0.0])?;
//! c.put(b"b", &[0.0, 1.0, 0.0])?;
//!
//! let hits = c.search(&[0.9, 0.1, 0.0], 1, None)?;
//! assert_eq!(hits[0].key, b"a");
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # The metric decides what is stored
//!
//! [`Metric::L2`] stores the vector it was given. [`Metric::Cosine`] stores the
//! unit vector, because the index measures distance and on unit vectors the
//! nearest by distance is the nearest by angle, so cosine costs one
//! normalisation on the way in rather than a second index. That is also what
//! comes back out of [`Collection::get`], and it is the same answer Redis gives
//! for a cosine vector set.
//!
//! [`Metric::Ip`] and [`Metric::Hamming`] are refused rather than approximated.
//! Inner product is not a distance, so ordering by it is not ordering by
//! nearness and the partitions would be built around the wrong question.
//! Hamming wants binary vectors that a collection of floats does not hold.
//!
//! # Where the vectors live
//!
//! Beside the index, one flat run of floats with a slot per id, which the last
//! step of every search reads and which the index itself never holds. An id is
//! never anything but the slot it names, so the rerank is an offset rather than
//! a lookup, and a slot comes back for reuse when its key is removed. `06` puts
//! them in
//! the record log at kind 3 and `yo_format::vector` is that record already
//! written down. Until the file lands the run is in memory, which is where every
//! other collection in this build is too, and nothing on this page changes when
//! it moves.

use yo_common::{Code, Error, Result};
use yo_kv::Elements;
use yo_shape::Metric;

use crate::partition::{Partitions, Tuning, Vectors};
use crate::rabitq::Bits;

/// The largest dimension a collection can hold.
///
/// Taken from the record format rather than picked again here, because a
/// collection that accepted a vector the log cannot hold would be one that works
/// until the file lands.
pub use yo_format::vector::MAX_DIM;

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
/// Splits and merges run inside the write that made them necessary, because a
/// build with no maintenance slice has nowhere else to run them. A budget of
/// four postings means a write that triggers a split pays for that split and
/// not for a backlog, and a collection that has fallen behind catches up over
/// the writes that follow rather than in one pause. A server with a maintenance
/// slice calls [`Collection::maintain`] with a bigger number and this stays as
/// the floor under it.
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

/// Vectors under keys: the index, the vectors it reranks against, and the table
/// that turns one into the other.
#[derive(Debug)]
pub struct Collection {
    /// The RaBitQ codes under partitions that split and merge in place.
    index: Partitions,
    /// The full precision vectors, which the last step of every search measures
    /// against and which the index itself never holds.
    raw: Raw,
    /// Key to the id the index knows it by.
    ids: Elements<u64>,
    metric: Metric,
}

impl Collection {
    /// An empty collection that has allocated nothing yet.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for a dimension of zero or past [`MAX_DIM`], and
    /// [`Code::Unsupported`] for a metric this build does not measure.
    pub fn new(dim: usize, metric: Metric) -> Result<Collection> {
        width(dim)?;
        check_metric(metric)?;
        Ok(Collection {
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
        })
    }

    /// How many coordinates a vector here has.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.raw.dim
    }

    /// What nearness means here.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// How many vectors are in the collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// How many partitions the collection has grown to.
    #[must_use]
    pub fn partitions(&self) -> usize {
        self.index.partitions()
    }

    /// The knobs the index is searched with.
    #[must_use]
    pub fn tuning(&self) -> Tuning {
        self.index.tuning()
    }

    /// Change them, which is what `EF_RUNTIME` on the wire means.
    pub fn retune(&mut self, tuning: Tuning) {
        self.index.retune(tuning);
    }

    /// Whether the collection holds a vector under `key`.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        self.ids.contains(key)
    }

    /// The vector under `key`, where it lies.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&[f32]> {
        let id = *self.ids.get(key)?;
        Some(self.raw.at(id))
    }

    /// Every key in the collection, in no order worth relying on.
    pub fn keys(&self) -> impl Iterator<Item = &[u8]> {
        self.ids.iter().map(|(key, _)| key)
    }

    /// The `n`th key, counting from zero, in that same order.
    ///
    /// For a caller that wants one member and not all of them, which is what
    /// `VRANDMEMBER` is and what walking the whole table to throw it away would
    /// be the wrong way to answer.
    #[must_use]
    pub fn key_at(&self, n: usize) -> Option<&[u8]> {
        self.ids.at(n).map(|(key, _)| key)
    }

    /// The id the index knows `key` by.
    ///
    /// An id is the slot the vector sits in. It is stable while the key is
    /// there, it changes if the key is removed and written again, and it is
    /// handed out so that a caller keeping something else per vector can key it
    /// by a small integer rather than by a second copy of the key. The attribute
    /// a vector set holds is the first of those and a pushed down filter's tag
    /// will be the next.
    #[must_use]
    pub fn id(&self, key: &[u8]) -> Option<u64> {
        self.ids.get(key).copied()
    }

    /// Put a vector in under `key`, and say whether the key is new.
    ///
    /// Replacing is the same call. The old code comes out of its partition and
    /// the new one goes into whichever partition it belongs to now, so nothing
    /// accumulates and there is no rebuild waiting at the end of it.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the vector is not [`Collection::dim`] long, when
    /// a coordinate is not a number, or when a cosine collection is handed a
    /// vector of length zero, which has no direction to store. [`Code::Full`]
    /// for a key past the length limit.
    pub fn put(&mut self, key: &[u8], v: &[f32]) -> Result<bool> {
        self.put_tagged(key, v, 0)
    }

    /// The same, with the tag a filtered search will meet in the posting scan.
    ///
    /// A tag is 64 bits and it travels beside the code rather than beside the
    /// vector, which is the whole reason a filter here costs nothing: the scan
    /// is already reading that cache line to get at the code, so testing the
    /// tag is one instruction on a word that has arrived anyway. See
    /// [`Signature`](crate::Signature) for how a set of field and value pairs
    /// becomes one, and [`Collection::search_where`] for the other end of it.
    ///
    /// A tag of zero passes no filter except [`Any`](crate::Any), which is
    /// what an untagged collection wants: [`Collection::put`] is this with a
    /// zero and every search over it is unfiltered.
    ///
    /// # Errors
    ///
    /// As [`Collection::put`].
    pub fn put_tagged(&mut self, key: &[u8], v: &[f32], tag: u64) -> Result<bool> {
        let ready = self.ready(v)?;

        let new = match self.ids.get(key) {
            Some(&id) => {
                self.raw.write(id, &ready);
                self.index.insert_tagged(id, &ready, tag);
                false
            }
            None => {
                let id = self.raw.take(key, &ready);
                if self.ids.insert(key, id).is_err() {
                    self.raw.release(id);
                    return Err(Error::new(
                        Code::Full,
                        "that key is too long for a vector collection",
                    ));
                }
                self.index.insert_tagged(id, &ready, tag);
                true
            }
        };

        self.catch_up();
        Ok(new)
    }

    /// The tag `key` was stored with, if it is here.
    #[must_use]
    pub fn tag(&self, key: &[u8]) -> Option<u64> {
        self.index.tag(*self.ids.get(key)?)
    }

    /// Change the tag under `key` without touching the vector, and say whether
    /// there was one.
    ///
    /// The tag summarises something outside the vector, so it can go stale while
    /// the vector is still right. Rewriting it is one store into the posting,
    /// with no requantisation and no maintenance, which is what makes it cheap
    /// enough to redo every tag in a collection when the summary changes.
    pub fn retag(&mut self, key: &[u8], tag: u64) -> bool {
        let Some(&id) = self.ids.get(key) else {
            return false;
        };
        self.index.retag(id, tag)
    }

    /// Take a vector out, saying whether it was there.
    ///
    /// A delete here is a delete and not a tombstone: the member leaves its
    /// posting and the last member of that posting moves into the hole.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        let Some(id) = self.ids.remove(key) else {
            return false;
        };
        self.index.remove(id);
        self.raw.release(id);
        self.catch_up();
        true
    }

    /// The `k` nearest keys to `q`, nearest first, with `skip` left out.
    ///
    /// `skip` is what makes a more-like-this search work: the vector already
    /// stored under a key is always nearest to itself, and nobody asked what a
    /// thing is most similar to itself.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when `q` is not [`Collection::dim`] long or holds a
    /// coordinate that is not a number.
    pub fn search(&self, q: &[f32], k: usize, skip: Option<&[u8]>) -> Result<Vec<Match>> {
        self.search_where(q, k, skip, &crate::Any)
    }

    /// The same, over only the members whose tag `filter` allows.
    ///
    /// The filter runs inside the posting scan and not on the answers, which is
    /// the difference between a filtered search and a search followed by a
    /// filter. A filter matching one member in a thousand, applied to the
    /// nearest ten, returns nothing almost every time; applied in the scan it
    /// keeps reading further partitions until it has `k` or until it has spent
    /// [`Tuning::widen`], so it returns the nearest ten that pass.
    ///
    /// It can still come back with fewer than `k`. That is the trade every
    /// engine makes here and it is the right one, because the alternative to
    /// giving up after a bounded widen is reading the whole collection for a
    /// query that was going to find nothing anyway.
    ///
    /// # Errors
    ///
    /// As [`Collection::search`].
    pub fn search_where(
        &self,
        q: &[f32],
        k: usize,
        skip: Option<&[u8]>,
        filter: &impl crate::Filter,
    ) -> Result<Vec<Match>> {
        // The query is checked before the collection is looked at, so that a
        // query of the wrong length says so rather than answering nothing at
        // all while the collection happens to be empty.
        let ready = self.ready(q)?;
        if k == 0 || self.index.is_empty() {
            return Ok(Vec::new());
        }

        // One more than asked for when a key is being left out, so that leaving
        // it out does not cost an answer. It is the nearest one and it is
        // therefore always in the shortlist.
        let want = if skip.is_some() { k + 1 } else { k };
        let hits = self.index.search_where(&ready, want, filter, &self.raw);

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

    /// The same answer, arrived at by measuring every vector in the collection.
    ///
    /// This is what the index is an approximation of, so it is the thing recall
    /// is measured against, and it is what `VSIM ... TRUTH` asks for. It reads
    /// no codes at all: the estimator exists to avoid this walk and there is
    /// nothing it can contribute to a walk that is happening anyway.
    ///
    /// Linear in the collection, which is the point. A client asking for it on a
    /// million vectors is asking for a million distances and should get them
    /// rather than a refusal, because the reason to ask is to find out what the
    /// index missed.
    ///
    /// # Errors
    ///
    /// As [`Collection::search`].
    pub fn search_exact(&self, q: &[f32], k: usize, skip: Option<&[u8]>) -> Result<Vec<Match>> {
        self.search_exact_where(q, k, skip, &crate::Any)
    }

    /// The same walk, over only the members the filter allows.
    ///
    /// There is no scan to push the filter into here, because there is no scan:
    /// this measures everything. It exists so that a client asking for the exact
    /// answer and asking for a filter gets the exact answer to the question it
    /// asked, rather than being told the two options do not go together.
    ///
    /// # Errors
    ///
    /// As [`Collection::search`].
    pub fn search_exact_where(
        &self,
        q: &[f32],
        k: usize,
        skip: Option<&[u8]>,
        filter: &impl crate::Filter,
    ) -> Result<Vec<Match>> {
        let ready = self.ready(q)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut hits: Vec<(f32, &[u8])> = Vec::with_capacity(self.ids.len());
        for (key, &id) in self.ids.iter() {
            if skip == Some(key) {
                continue;
            }
            if !filter.allows(self.index.tag(id).unwrap_or(0)) || !filter.exact(id) {
                continue;
            }
            hits.push((crate::dist::sqdist(&ready, self.raw.at(id)), key));
        }
        // By distance and then by key, so that vectors at the same distance come
        // back in an order that does not depend on where the table put them.
        hits.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
        hits.truncate(k);
        Ok(hits
            .into_iter()
            .map(|(sq, key)| Match {
                key: key.to_vec(),
                distance: self.report(sq),
            })
            .collect())
    }

    /// Do bounded maintenance, and say how many vectors it looked at.
    ///
    /// A caller with a maintenance slice runs this until it returns less than
    /// the budget. A caller without one gets what [`Collection::put`] does on
    /// its own, which is the same work in smaller pieces.
    pub fn maintain(&mut self, budget: usize) -> usize {
        self.index.maintain(&self.raw, budget)
    }

    /// What the collection is holding: the vectors, the codes and the keys.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.raw.memory_bytes() + self.index.code_bytes() + self.ids.memory_bytes()
    }

    /// The searchable size of the collection, which is the number the 32x claim
    /// is about.
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.index.code_bytes()
    }

    // -- what an image is made of -------------------------------------------
    //
    // [`crate::image`] is the other half of these: it writes a collection down
    // and reads it back, and it needs at the parts that nothing else outside
    // this file has any business touching. A load is not a sequence of writes,
    // so it cannot go through [`Collection::put`]: putting a vector back would
    // requantise it, and requantising every vector is the rebuild an image
    // exists to avoid.

    /// The index, for the half of an image that is codes and centroids.
    pub(crate) fn index(&self) -> &Partitions {
        &self.index
    }

    /// Key to id, which is the half of an image that is names.
    pub(crate) fn id_table(&self) -> &Elements<u64> {
        &self.ids
    }

    /// How far the ids go, which is not how many there are.
    ///
    /// An id is a slot, so a collection that has had members removed has holes
    /// and the live count says nothing about the highest id in use. This is the
    /// number an image writes so that a load can allocate the table once.
    pub(crate) fn slots(&self) -> usize {
        self.raw.owner.len()
    }

    /// Whether the index has a member under `id`.
    pub(crate) fn holds(&self, id: u64) -> bool {
        self.index.contains(id)
    }

    /// An empty collection around an index that is already built, with room for
    /// `slots` vectors and not one of them written yet.
    pub(crate) fn from_image(index: Partitions, metric: Metric, slots: usize) -> Collection {
        let dim = index.dim();
        Collection {
            index,
            raw: Raw {
                dim,
                data: vec![0.0; slots * dim],
                owner: vec![None; slots],
                free: Vec::new(),
            },
            ids: Elements::new(),
            metric,
        }
    }

    /// Put a vector back in the slot the image says it was in.
    ///
    /// The index already has the member, so this is the other two tables only:
    /// the vector into its slot and the key into the id table. No maintenance
    /// runs, because nothing moved.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] for a key the id table will not take, which for an image
    /// written by this build cannot happen and for a damaged one is the honest
    /// answer.
    pub(crate) fn restore(&mut self, key: &[u8], id: u64, v: &[f32]) -> Result<()> {
        self.raw.owner[id as usize] = Some(key.into());
        self.raw.write(id, v);
        self.ids
            .insert(key, id)
            .map_err(|_| Error::new(Code::Full, "that key is too long for a vector collection"))?;
        Ok(())
    }

    /// Drop a member the store could not produce a vector for.
    pub(crate) fn forget(&mut self, id: u64) {
        self.index.remove(id);
    }

    /// Say a load is over, so the free list can be built from what is left.
    ///
    /// In reverse, so that the lowest free slot is on top and the next write
    /// takes it. That is what an insert path that has been running normally
    /// leaves behind, and a collection that has just been loaded should not be
    /// distinguishable from one that has not.
    pub(crate) fn seal(&mut self) {
        self.raw.free.clear();
        for id in (0..self.raw.owner.len()).rev() {
            if self.raw.owner[id].is_none() {
                self.raw.free.push(id as u64);
            }
        }
    }

    /// A vector this collection can take, in the form it stores.
    fn ready(&self, v: &[f32]) -> Result<Vec<f32>> {
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
        let mut ready = v.to_vec();
        if self.metric == Metric::Cosine {
            normalize(&mut ready)?;
        }
        Ok(ready)
    }

    /// Run the splits and merges the last write owes, if it owes any.
    ///
    /// Inside the write rather than after a threshold of them, because a split
    /// that is owed is a partition already twice the size it wants to be and
    /// every search until it happens reads all of it.
    fn catch_up(&mut self) {
        if self.index.needs_maintenance() {
            self.index.maintain(&self.raw, BUDGET);
        }
    }

    /// The distance to report for the squared one the index measured.
    ///
    /// The index works in squared euclidean distance because a square root
    /// changes no ordering and costs one per candidate. The caller asked in the
    /// metric the collection was opened with, so the square root happens here,
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
#[derive(Debug)]
struct Raw {
    dim: usize,
    data: Vec<f32>,
    /// The key each slot holds, and `None` for a slot that is free.
    ///
    /// This is the second copy of a key, the first being the one in
    /// [`Collection::ids`], and it is here because a search comes back with ids
    /// and has to answer in keys. A key is tens of bytes against a vector's
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

impl Vectors for Raw {
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

/// `dim` as the shape grammar writes it, if it is a dimension a collection can
/// be opened with.
///
/// The width comes back rather than a bare yes, because every caller that has to
/// ask this also has to write the dimension into the collection's description,
/// and the check is what makes the conversion safe.
///
/// # Errors
///
/// [`Code::Invalid`], with the range in the message.
pub fn width(dim: usize) -> Result<u32> {
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
///
/// # Errors
///
/// [`Code::Unsupported`], saying what to do instead.
pub fn check_metric(metric: Metric) -> Result<()> {
    match metric {
        Metric::L2 | Metric::Cosine => Ok(()),
        Metric::Ip => Err(Error::new(
            Code::Unsupported,
            "inner product is not a distance, so a partition index cannot be built around it, and a collection that ordered by it would not be ordering by nearness. Normalise the vectors and use cosine, which is the same ranking",
        )),
        Metric::Hamming => Err(Error::new(
            Code::Unsupported,
            "hamming distance is for binary vectors and this collection holds floats",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Collection {
        let mut c = Collection::new(3, Metric::L2).unwrap();
        c.put(b"x", &[1.0, 0.0, 0.0]).unwrap();
        c.put(b"y", &[0.0, 1.0, 0.0]).unwrap();
        c.put(b"z", &[0.0, 0.0, 1.0]).unwrap();
        c
    }

    #[test]
    fn a_vector_comes_back_the_way_it_went_in() {
        let mut c = Collection::new(3, Metric::L2).unwrap();
        assert!(c.is_empty());
        assert!(c.put(b"x", &[1.0, 2.0, 3.0]).unwrap(), "the key is new");
        assert!(
            !c.put(b"x", &[1.0, 2.0, 3.0]).unwrap(),
            "and then it is not"
        );
        assert_eq!(c.get(b"x"), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(c.len(), 1);
        assert!(c.contains(b"x"));
        assert_eq!(c.get(b"nobody"), None);
        assert_eq!(c.keys().collect::<Vec<_>>(), vec![&b"x"[..]]);
        assert!(c.memory_bytes() > 0);
    }

    #[test]
    fn the_nearest_answer_is_the_nearest_vector() {
        let c = axes();
        let hits = c.search(&[0.9, 0.2, 0.1], 3, None).unwrap();
        let keys: Vec<&[u8]> = hits.iter().map(|h| h.key.as_slice()).collect();
        assert_eq!(keys, vec![&b"x"[..], &b"y"[..], &b"z"[..]]);
        // The exact euclidean distance and not the estimate the codes gave,
        // which is the point of reranking against the stored vector.
        let want = (0.01f32 + 0.04 + 0.01).sqrt();
        assert!((hits[0].distance - want).abs() < 1e-6, "{hits:?}");
    }

    #[test]
    fn a_removed_vector_is_not_an_answer_and_its_slot_comes_back() {
        let mut c = axes();
        assert!(c.remove(b"x"));
        assert!(!c.remove(b"x"), "twice is not there twice");
        assert_eq!(c.len(), 2);

        let hits = c.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.key != b"x"));

        c.put(b"w", &[1.0, 0.0, 0.0]).unwrap();
        let hits = c.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].key, b"w", "the reused slot answers as w");
    }

    /// Replacing has to take the old code out of its partition as well as
    /// writing the new vector, or a search answers with a key whose vector
    /// moved somewhere else.
    #[test]
    fn a_replaced_vector_is_searched_at_its_new_place() {
        let mut c = axes();
        c.put(b"x", &[0.0, 0.0, 1.0]).unwrap();
        assert_eq!(c.len(), 3, "a replacement is not a second key");

        let hits = c.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].key, b"y", "x moved away from that corner");
    }

    #[test]
    fn a_search_can_leave_one_key_out() {
        let mut c = axes();
        c.put(b"x2", &[0.9, 0.1, 0.0]).unwrap();
        let hits = c.search(&[1.0, 0.0, 0.0], 2, Some(b"x")).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].key, b"x2");
        assert!(hits.iter().all(|h| h.key != b"x"));
    }

    #[test]
    fn a_cosine_collection_stores_the_direction_and_reports_the_angle() {
        let mut c = Collection::new(2, Metric::Cosine).unwrap();
        c.put(b"east", &[7.0, 0.0]).unwrap();
        c.put(b"north", &[0.0, 3.0]).unwrap();
        c.put(b"west", &[-2.0, 0.0]).unwrap();
        assert_eq!(c.get(b"east"), Some(&[1.0, 0.0][..]));

        // Length is nothing to a cosine collection, so a long east and a short
        // east are the same vector and both are nearer than north.
        let hits = c.search(&[100.0, 0.0], 3, None).unwrap();
        assert_eq!(hits[0].key, b"east");
        assert!(hits[0].distance.abs() < 1e-6, "{hits:?}");
        assert!(
            (hits[1].distance - 1.0).abs() < 1e-6,
            "north is a right angle"
        );
        assert!(
            (hits[2].distance - 2.0).abs() < 1e-6,
            "west is the opposite"
        );

        let e = c.put(b"nowhere", &[0.0, 0.0]).expect_err("no direction");
        assert_eq!(e.code(), Code::Invalid);
    }

    #[test]
    fn a_vector_of_the_wrong_length_or_shape_is_refused() {
        let mut c = Collection::new(3, Metric::L2).unwrap();
        let e = c.put(b"x", &[1.0, 2.0]).expect_err("two is not three");
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("3 dimensional"), "{e}");

        let e = c
            .put(b"x", &[1.0, f32::NAN, 2.0])
            .expect_err("not a number");
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("coordinate 1"), "{e}");

        // On an empty collection too, where there is nothing to search and the
        // easy thing would be to answer nothing.
        let e = c.search(&[1.0], 1, None).expect_err("one is not three");
        assert_eq!(e.code(), Code::Invalid);
    }

    #[test]
    fn a_dimension_or_a_metric_the_build_cannot_hold_is_refused() {
        assert_eq!(
            Collection::new(0, Metric::L2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Collection::new(MAX_DIM + 1, Metric::L2).unwrap_err().code(),
            Code::Invalid
        );
        let e = Collection::new(8, Metric::Ip).unwrap_err();
        assert_eq!(e.code(), Code::Unsupported);
        assert!(e.message().contains("cosine"), "{e}");
        assert_eq!(
            Collection::new(8, Metric::Hamming).unwrap_err().code(),
            Code::Unsupported
        );
    }

    /// Everything above runs inside one partition, and one partition is a scan
    /// rather than an index, so this is the one that actually exercises the
    /// splits and the maintenance the writes pay for.
    #[test]
    fn recall_holds_once_the_index_has_split() {
        let mut c = Collection::new(8, Metric::L2).unwrap();
        let mut seed = 0x2026u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };

        let mut all: Vec<Vec<f32>> = Vec::new();
        for i in 0..2000usize {
            let x: Vec<f32> = (0..8).map(|_| next()).collect();
            c.put(format!("k{i}").as_bytes(), &x).unwrap();
            all.push(x);
        }
        assert!(c.partitions() > 1, "nothing ever split");

        let mut found = 0;
        for (i, q) in all.iter().enumerate().step_by(50) {
            let hits = c.search(q, 1, None).unwrap();
            if hits[0].key == format!("k{i}").into_bytes() {
                found += 1;
            }
        }
        assert!(found >= 39, "{found} of 40 queries found their own vector");
        assert_eq!(c.maintain(1 << 20), 0, "the writes left nothing owed");
    }
}
