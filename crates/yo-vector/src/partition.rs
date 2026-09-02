//! The partition index, and the in place update protocol that means it never
//! has to be rebuilt (`10` sections 2, 4 and 5).
//!
//! A vector index is two decisions and the quantiser was the easy one. This is
//! the other: what holds the codes, and what happens to it when the collection
//! changes.
//!
//! The answer is not a graph. Redis shipped HNSW vector sets in 8.0 in May 2025
//! and they were still beta three minor releases later, which is the vendor's
//! own evidence about how that goes. The reason is the update path: a graph
//! index tombstones a delete, degrades as the tombstones pile up, and only gets
//! better again when somebody rebuilds it, which on a collection anyone cares
//! about is an outage with a nicer name.
//!
//! So this is partitions. Every vector belongs to the partition whose centroid
//! it is nearest, the centroids are resident, and a partition's members are a
//! flat run of codes that the scan walks end to end. An insert is an append. A
//! delete takes a member out and moves the last one into the hole. Neither one
//! touches anything else.
//!
//! # Search
//!
//! Rank the centroids, take the nearest `probe` of them, scan those partitions
//! with the estimator, keep the best `rerank` candidates, and then look up the
//! full precision vectors for exactly those and measure them properly. Rerank
//! costs nothing structurally here, because the vector is already in the record
//! log at an address the id resolves to. Every other system that quantises has
//! to keep the raw vectors somewhere on purpose.
//!
//! The scan is the shape hardware likes, a linear walk over contiguous bytes,
//! and that is why an index of this family beats a graph on a modern core even
//! though it looks at more candidates.
//!
//! # It never rebuilds
//!
//! This is SPFresh's LIRE, and it is four bounded operations rather than a
//! background rebuild.
//!
//! A posting that grows past twice its target splits, by two means over its own
//! members. A posting that falls under a quarter of its target merges, by
//! handing its members to whichever centroid is nearest now. Both are bounded
//! work on one partition.
//!
//! The third is the one that matters and it is what LIRE actually contributes.
//! After a split, the members of the partitions *around* the one that split may
//! now be nearer one of the two new centroids than the one they are filed
//! under. Nobody told them, and a plain partitioned index just lets that drift,
//! which is why a plain partitioned index measures beautifully on a freshly
//! built corpus and badly after a week of writes. So a split is followed by a
//! sweep of the neighbouring partitions, and anything whose nearest centroid
//! has changed is moved. The test for this writes a stream and checks recall at
//! the end of it rather than on a fresh build, because a fresh build is exactly
//! the measurement that hides the problem.
//!
//! # Everything here is in rotated space
//!
//! The rotation is linear, so `rotate(v - c)` is `rotate(v) - rotate(c)`, and
//! distances and angles come through it unchanged. That means a centroid can be
//! stored already rotated and a query can be rotated once, and then meeting a
//! partition is a subtraction rather than another rotation. The rotation is the
//! expensive half of preparing a query, so on a search that probes tens of
//! partitions this is most of what preparation costs.
//!
//! # What is not here yet
//!
//! Ranking the centroids is a linear pass over all of them, which is fine while
//! there are hundreds and is not fine at the ten thousand a ten million vector
//! collection wants. The fix is the same one as everywhere else in this crate,
//! which is to quantise the centroids and scan them with popcounts too, and the
//! bench has the row that says when it starts to matter.
//!
//! Filters pushed into the scan, MUVERA, and writing any of this to a `.yo`
//! file are the rest of M6.

use std::collections::HashMap;

use crate::rabitq::{Bits, Coded, Quantizer};

/// Where the full precision vectors live.
///
/// A real collection answers this out of the record log, which already holds
/// the vector at an address the id resolves to. A test answers it out of a map.
/// Either way the index itself never stores a raw vector, which is the whole
/// point of quantising one.
pub trait Vectors {
    /// Write the vector `id` stands for into `into` and say so, or say that the
    /// id is gone.
    ///
    /// An id that is gone is dropped from the index the next time maintenance
    /// walks over it, so a collection that deletes from the log without telling
    /// the index heals rather than lying.
    fn get(&self, id: u64, into: &mut [f32]) -> bool;
}

/// The knobs, all of which have a defensible default and none of which anybody
/// should have to touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tuning {
    /// How many members a partition wants. It splits past twice this and merges
    /// under a quarter of it.
    ///
    /// This is what sets how many partitions a collection ends up with, and so
    /// it trades the cost of ranking centroids against the cost of scanning a
    /// posting. A few hundred is where those two are near enough even.
    pub posting: usize,
    /// How many partitions a search scans.
    pub probe: usize,
    /// How many candidates are reranked per answer asked for.
    ///
    /// Four is the number the recall table was measured at: one bit codes put
    /// the true ten inside the best forty better than 98 times in a hundred.
    pub rerank: usize,
    /// How many neighbouring partitions a split sweeps for members that should
    /// move.
    ///
    /// This is the cost of never drifting. Zero would make a split free and
    /// would make recall fall off over a long write stream, which is the thing
    /// this index exists to not do.
    pub sweep: usize,
    /// How much further than `probe` a filtered search will go looking when the
    /// filter is selective enough that the nearest partitions do not hold `k`
    /// members that pass, as a multiple of `probe`.
    ///
    /// This is the only knob here with a genuinely hard trade behind it. Too
    /// small and a filter matching one document in a thousand returns nothing
    /// while the answer sat two partitions further out. Too large and the same
    /// filter reads the whole collection to prove there is nothing there.
    pub widen: usize,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            posting: 256,
            probe: 8,
            rerank: 4,
            sweep: 4,
            widen: 8,
        }
    }
}

/// The fewest candidates a search will rerank, whatever `k` and `rerank`
/// multiply out to.
///
/// Four times `k` is the right ratio and it is the wrong number when `k` is
/// small: asking for one answer and reranking four candidates puts the whole
/// weight of the answer on the estimator getting its top four right, which is
/// not what the estimator is for. Reranking a few dozen costs a few dozen
/// squared distances, which is nothing next to the scan that produced them.
const FLOOR: usize = 32;

/// What decides whether the scan bothers with a member.
///
/// A filtered vector search is a recall lottery when the filter runs after the
/// search: ask for ten English passages, get the best forty by vector, find
/// three of them are English, and the other seven English passages that were
/// nearer never had a chance. The fix is to filter inside the scan, so that
/// only members that can be answers are ranked at all, and that means the thing
/// the filter reads has to sit next to the codes rather than behind a lookup
/// into somebody else's table.
///
/// So every member carries a `u64` tag, given at insert, and a filter is a
/// predicate on that tag. What the tag means is the caller's business. A
/// handful of low cardinality attributes pack into it exactly, one field each,
/// and the filter is then exact. Anything wider goes through [`Signature`],
/// which is exact in the direction that matters: it never rejects a member that
/// should have matched, so the caller's real predicate over the answers still
/// decides.
pub trait Filter {
    /// Whether a member with this tag is worth ranking.
    fn allows(&self, tag: u64) -> bool;
}

/// The filter that lets everything through, which is what an unfiltered search
/// runs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Any;

impl Filter for Any {
    fn allows(&self, _tag: u64) -> bool {
        true
    }
}

impl<F: Fn(u64) -> bool> Filter for F {
    fn allows(&self, tag: u64) -> bool {
        self(tag)
    }
}

/// A tag built by setting one bit per attribute value, so that a conjunction of
/// required values is a subset test.
///
/// Superimposed coding, which is old and still the right answer when the test
/// has to be one instruction on a value that is already in a register. Each
/// attribute and value pair hashes to one of 64 bits. A member's tag is the
/// bits for the values it has. A query's tag is the bits for the values it
/// requires. The member is worth ranking when it has all of the query's bits.
///
/// Two different values can land on the same bit, so a member can pass a filter
/// it does not really match. It can never fail one it does match, which is the
/// direction that matters: the answers are a superset of the truth and the
/// caller's own predicate cuts them down, where the other way round would lose
/// answers silently.
///
/// ```
/// use yo_vector::Signature;
///
/// // What a document is tagged with, and what a query asks for.
/// let doc = Signature::of(&[("lang", "en".as_bytes()), ("topic", "finance".as_bytes())]);
/// let english = Signature::of(&[("lang", "en".as_bytes())]);
///
/// assert!(doc.covers(english));
/// // The other way round only holds if the two bits happened to collide.
/// assert!(!english.covers(doc) || english.bits() == doc.bits());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Signature(u64);

impl Signature {
    /// The signature of a set of attribute and value pairs.
    #[must_use]
    pub fn of(values: &[(&str, &[u8])]) -> Signature {
        let mut bits = 0u64;
        for (attribute, value) in values {
            bits |= 1u64 << (hash(attribute.as_bytes(), value) % 64);
        }
        Signature(bits)
    }

    /// The signature as the tag to hand to [`Partitions::insert_tagged`].
    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// The signature of a tag that came back out of the index.
    #[must_use]
    pub fn from_bits(bits: u64) -> Signature {
        Signature(bits)
    }

    /// Whether this has every bit `want` has, which is the test the scan runs.
    #[must_use]
    pub fn covers(self, want: Signature) -> bool {
        self.0 & want.0 == want.0
    }
}

impl Filter for Signature {
    fn allows(&self, tag: u64) -> bool {
        Signature(tag).covers(*self)
    }
}

/// FNV over the attribute and then the value, which is small, has no state and
/// spreads a short value over the whole word well enough to pick a bit.
fn hash(attribute: &[u8], value: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for byte in attribute.iter().chain(b":").chain(value) {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// An answer: a document id and how far it really is, not how far it was
/// estimated to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The id that was inserted.
    pub id: u64,
    /// The exact squared distance, measured against the full precision vector.
    pub distance: f32,
}

/// Where a member sits.
#[derive(Debug, Clone, Copy)]
struct Slot {
    partition: u32,
    slot: u32,
}

/// One partition's members: the ids, their codes end to end, and what each code
/// needs beside it.
#[derive(Default)]
struct Posting {
    ids: Vec<u64>,
    /// One tag per member, in the same order, which is what a filter meets.
    ///
    /// Beside the ids rather than behind a pointer, because the whole point is
    /// that the scan can skip a member without touching anything that is not
    /// already in cache.
    tags: Vec<u64>,
    codes: Vec<u8>,
    meta: Vec<Coded>,
    /// The size at which a split was tried and found there was no cut, which
    /// happens when every member is the same vector. It is not tried again
    /// until the posting has grown past it.
    stuck: usize,
}

impl Posting {
    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// A collection of vectors, quantised, partitioned, and updated in place.
pub struct Partitions {
    quant: Quantizer,
    tuning: Tuning,
    /// The centroids, already rotated, `dim` floats each end to end.
    centroids: Vec<f32>,
    postings: Vec<Posting>,
    /// Which partition and which slot every id is in, which is what makes a
    /// delete a constant time operation rather than a search.
    at: HashMap<u64, Slot>,
}

impl Partitions {
    /// An empty collection of `dim` dimensional vectors.
    ///
    /// The first vector inserted becomes the first centroid, and the index
    /// grows by splitting from there, so there is no build step and no moment
    /// where the shape of the collection has to be known in advance.
    ///
    /// # Panics
    ///
    /// If `dim` is zero.
    #[must_use]
    pub fn new(dim: usize, bits: Bits, seed: u64, tuning: Tuning) -> Partitions {
        Partitions {
            quant: Quantizer::new(dim, bits, seed),
            tuning,
            centroids: Vec::new(),
            postings: Vec::new(),
            at: HashMap::new(),
        }
    }

    /// How many coordinates a vector here has.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.quant.dim()
    }

    /// How many vectors are in the collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.at.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// How many partitions the collection has grown to.
    #[must_use]
    pub fn partitions(&self) -> usize {
        self.postings.len()
    }

    /// The knobs.
    #[must_use]
    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// The quantiser, whose seed and width a catalogue entry has to record.
    #[must_use]
    pub fn quantizer(&self) -> &Quantizer {
        &self.quant
    }

    /// How many bytes the codes take, which is the searchable size of the
    /// collection and the number the 32x claim is about.
    #[must_use]
    pub fn code_bytes(&self) -> usize {
        self.postings.iter().map(|p| p.codes.len()).sum()
    }

    /// Put a vector in, replacing whatever was under `id`.
    ///
    /// Two appends and no locks: the code goes on the end of the nearest
    /// partition's posting, and the caller puts the full precision vector in the
    /// log. Nothing else in the index is touched, which is the difference
    /// between this and a graph.
    ///
    /// # Panics
    ///
    /// If `v` is not [`Partitions::dim`] long.
    pub fn insert(&mut self, id: u64, v: &[f32]) {
        self.insert_tagged(id, v, 0);
    }

    /// The same, with the tag a filter will meet in the scan.
    ///
    /// See [`Filter`] for what a tag is and [`Signature`] for the encoding to
    /// reach for when the attributes do not fit in one exactly.
    ///
    /// # Panics
    ///
    /// If `v` is not [`Partitions::dim`] long.
    pub fn insert_tagged(&mut self, id: u64, v: &[f32], tag: u64) {
        assert_eq!(
            v.len(),
            self.dim(),
            "this collection holds {} dimensional vectors and was handed {}",
            self.dim(),
            v.len()
        );
        self.remove(id);
        let x = self.quant.rotate(v);
        let p = if self.postings.is_empty() {
            // The first vector is the first centroid. There is nothing to
            // average it with yet, and the first split is what starts the
            // centroids being means rather than members.
            self.add_partition(&x)
        } else {
            self.nearest(&x)
        };
        self.place(p, id, tag, &x);
    }

    /// The tag `id` was inserted with, if it is still here.
    #[must_use]
    pub fn tag(&self, id: u64) -> Option<u64> {
        let at = self.at.get(&id)?;
        Some(self.postings[at.partition as usize].tags[at.slot as usize])
    }

    /// Take a vector out, saying whether it was there.
    ///
    /// The last member of the posting moves into the hole. There is no
    /// tombstone, so there is nothing to accumulate and nothing to compact.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(Slot { partition, slot }) = self.at.remove(&id) else {
            return false;
        };
        let moved = self.pull(partition as usize, slot as usize);
        if let Some(other) = moved {
            self.at.insert(other, Slot { partition, slot });
        }
        true
    }

    /// Whether `id` is in the collection.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.at.contains_key(&id)
    }

    /// The `k` nearest vectors to `q`, measured exactly.
    ///
    /// The codes pick the candidates and the log settles the order, so the
    /// answer is as exact as brute force whenever the candidates contained the
    /// truth, and the recall table is about how often they do.
    ///
    /// # Panics
    ///
    /// If `q` is not [`Partitions::dim`] long.
    #[must_use]
    pub fn search(&self, q: &[f32], k: usize, vectors: &impl Vectors) -> Vec<Hit> {
        self.search_where(q, k, &Any, vectors)
    }

    /// The `k` nearest vectors to `q` that a filter allows.
    ///
    /// The filter runs inside the scan, on the tag that sits next to the code,
    /// so a member the filter rejects is never ranked and never takes a place
    /// that an answer should have had. Filtering afterwards instead is what
    /// makes a filtered vector search a lottery, and the more selective the
    /// filter the worse a lottery it is.
    ///
    /// A selective filter also means the nearest few partitions may not hold `k`
    /// members that pass, so the scan keeps going into further partitions until
    /// it has enough or until it has spent [`Tuning::widen`]. A filter that
    /// matches almost nothing returns fewer answers rather than reading the
    /// whole collection, which is the trade every engine makes here and is worth
    /// saying out loud.
    ///
    /// # Panics
    ///
    /// If `q` is not [`Partitions::dim`] long.
    #[must_use]
    pub fn search_where(
        &self,
        q: &[f32],
        k: usize,
        filter: &impl Filter,
        vectors: &impl Vectors,
    ) -> Vec<Hit> {
        if k == 0 {
            return Vec::new();
        }
        let candidates = self.candidates_where(q, (k * self.tuning.rerank).max(FLOOR), filter);
        let mut buf = vec![0.0f32; self.dim()];
        let mut hits = Vec::with_capacity(candidates.len());
        for (id, _) in candidates {
            if vectors.get(id, &mut buf) {
                hits.push(Hit {
                    id,
                    distance: sqdist(q, &buf),
                });
            }
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits.truncate(k);
        hits
    }

    /// The `want` best candidates by the estimator, without rerank.
    ///
    /// This is what a filter will eventually push into, and it is what the
    /// recall of the codes alone is measured on.
    ///
    /// # Panics
    ///
    /// If `q` is not [`Partitions::dim`] long.
    #[must_use]
    pub fn candidates(&self, q: &[f32], want: usize) -> Vec<(u64, f32)> {
        self.candidates_where(q, want, &Any)
    }

    /// The same, with the filter run inside the scan.
    ///
    /// # Panics
    ///
    /// If `q` is not [`Partitions::dim`] long.
    #[must_use]
    pub fn candidates_where(
        &self,
        q: &[f32],
        want: usize,
        filter: &impl Filter,
    ) -> Vec<(u64, f32)> {
        assert_eq!(
            q.len(),
            self.dim(),
            "this collection holds {} dimensional vectors and was handed {}",
            self.dim(),
            q.len()
        );
        if want == 0 || self.postings.is_empty() {
            return Vec::new();
        }
        // Rotated once here and never again, which is what lets a search probe
        // tens of partitions without paying for tens of rotations.
        let u = self.quant.rotate(q);
        let width = self.quant.code_bytes();
        let mut found = Vec::new();
        let reach = self.tuning.probe.saturating_mul(self.tuning.widen.max(1));
        for (n, p) in self.near_partitions(&u, reach).into_iter().enumerate() {
            // Past the partitions an unfiltered search would have read, keep
            // going only while there is still not enough to answer with. An
            // unfiltered search never gets here, because the first `probe`
            // partitions of a collection worth probing hold more than `want`.
            if n >= self.tuning.probe && found.len() >= want {
                break;
            }
            let prepared = self.quant.query_rotated(&u, self.centroid(p));
            let posting = &self.postings[p];
            for (i, &id) in posting.ids.iter().enumerate() {
                if !filter.allows(posting.tags[i]) {
                    continue;
                }
                let code = &posting.codes[i * width..(i + 1) * width];
                found.push((id, prepared.distance(code, &posting.meta[i])));
            }
        }
        if found.len() > want {
            found.select_nth_unstable_by(want, |a, b| a.1.total_cmp(&b.1));
            found.truncate(want);
        }
        found.sort_by(|a, b| a.1.total_cmp(&b.1));
        found
    }

    /// Whether there is a split or a merge waiting.
    #[must_use]
    pub fn needs_maintenance(&self) -> bool {
        self.job().is_some()
    }

    /// Do bounded maintenance, and say how many vectors it looked at.
    ///
    /// `budget` is in vectors touched rather than in time, because time is not
    /// something a storage engine gets to measure cheaply and a vector is the
    /// unit all of this work is actually made of. Call it from a maintenance
    /// slice until it returns less than the budget, which means there was
    /// nothing left to do.
    pub fn maintain(&mut self, vectors: &impl Vectors, budget: usize) -> usize {
        let mut done = 0;
        while done < budget {
            let Some(job) = self.job() else { break };
            done += match job {
                Job::Split(p) => self.split(p, vectors),
                Job::Merge(p) => self.merge(p, vectors),
            };
        }
        done
    }

    /// The next thing worth doing, biggest problem first.
    fn job(&self) -> Option<Job> {
        let big = (0..self.postings.len())
            .filter(|&p| self.postings[p].len() > self.postings[p].stuck)
            .max_by_key(|&p| self.postings[p].len());
        if let Some(big) = big
            && self.postings[big].len() > self.tuning.posting * 2
        {
            return Some(Job::Split(big));
        }
        if self.postings.len() > 1 {
            let small = (0..self.postings.len()).min_by_key(|&p| self.postings[p].len())?;
            if self.postings[small].len() * 4 < self.tuning.posting {
                return Some(Job::Merge(small));
            }
        }
        None
    }

    /// Cut a partition in two by two means over its own members, then sweep the
    /// neighbours for anything that should have come along.
    fn split(&mut self, p: usize, vectors: &impl Vectors) -> usize {
        let (members, xs) = self.take(p, vectors);
        let dim = self.dim();
        if members.len() < 2 {
            for (i, m) in members.iter().enumerate() {
                self.place(p, m.id, m.tag, &xs[i * dim..(i + 1) * dim]);
            }
            return members.len();
        }
        let (a, b) = two_means(&xs, dim);
        let sides: Vec<bool> = (0..members.len())
            .map(|i| {
                sqdist(&xs[i * dim..(i + 1) * dim], &a) <= sqdist(&xs[i * dim..(i + 1) * dim], &b)
            })
            .collect();
        // A thousand copies of the same vector is one point as far as two means
        // is concerned, and there is no cut that divides it. Put them back, and
        // do not come back until the posting has doubled, so that a collection
        // that really is all one vector costs a re-encode of it a logarithmic
        // number of times rather than once per insert.
        if sides.iter().all(|&s| s) || sides.iter().all(|&s| !s) {
            for (i, m) in members.iter().enumerate() {
                self.place(p, m.id, m.tag, &xs[i * dim..(i + 1) * dim]);
            }
            self.postings[p].stuck = members.len() * 2;
            return members.len();
        }
        self.centroids[p * dim..(p + 1) * dim].copy_from_slice(&a);
        let q = self.add_partition(&b);
        for (i, m) in members.iter().enumerate() {
            let to = if sides[i] { p } else { q };
            self.place(to, m.id, m.tag, &xs[i * dim..(i + 1) * dim]);
        }
        members.len() + self.sweep(&[p, q], vectors)
    }

    /// Hand a partition's members to whoever is nearest now, and drop it.
    fn merge(&mut self, p: usize, vectors: &impl Vectors) -> usize {
        let (members, xs) = self.take(p, vectors);
        let dim = self.dim();
        self.drop_partition(p);
        for (i, m) in members.iter().enumerate() {
            let x = &xs[i * dim..(i + 1) * dim];
            let to = self.nearest(x);
            self.place(to, m.id, m.tag, x);
        }
        members.len()
    }

    /// LIRE: after the centroids move, anything nearby that is now filed under
    /// the wrong one gets moved.
    ///
    /// Only the partitions near the ones that just changed are looked at,
    /// because those are the only ones whose members can have a new nearest
    /// centroid, and looking at all of them would be the rebuild this index
    /// exists to avoid.
    fn sweep(&mut self, changed: &[usize], vectors: &impl Vectors) -> usize {
        let dim = self.dim();
        let mut look: Vec<usize> = Vec::new();
        for &p in changed {
            let centre = self.centroid(p).to_vec();
            for q in self.near_partitions(&centre, self.tuning.sweep) {
                if !changed.contains(&q) && !look.contains(&q) {
                    look.push(q);
                }
            }
        }
        let mut seen = 0;
        let mut buf = vec![0.0f32; dim];
        for p in look {
            // Backwards, because taking a member out moves the last one into
            // its slot and a backwards walk never steps over the one that moved.
            for i in (0..self.postings[p].len()).rev() {
                seen += 1;
                let id = self.postings[p].ids[i];
                let tag = self.postings[p].tags[i];
                if !vectors.get(id, &mut buf) {
                    self.pull_and_forget(p, i);
                    continue;
                }
                let x = self.quant.rotate(&buf);
                let to = self.nearest(&x);
                if to != p {
                    self.pull_and_forget(p, i);
                    self.place(to, id, tag, &x);
                }
            }
        }
        seen
    }

    /// Empty a partition out, handing back its members and their rotated
    /// vectors. Ids the source has forgotten are dropped.
    fn take(&mut self, p: usize, vectors: &impl Vectors) -> (Vec<Member>, Vec<f32>) {
        let dim = self.dim();
        let ids = std::mem::take(&mut self.postings[p].ids);
        let tags = std::mem::take(&mut self.postings[p].tags);
        self.postings[p].codes.clear();
        self.postings[p].meta.clear();
        let mut kept = Vec::with_capacity(ids.len());
        let mut xs = Vec::with_capacity(ids.len() * dim);
        let mut buf = vec![0.0f32; dim];
        for (id, tag) in ids.into_iter().zip(tags) {
            self.at.remove(&id);
            if vectors.get(id, &mut buf) {
                xs.extend_from_slice(&self.quant.rotate(&buf));
                kept.push(Member { id, tag });
            }
        }
        (kept, xs)
    }

    /// The `n` partitions whose centroids are nearest `x`, nearest first.
    fn near_partitions(&self, x: &[f32], n: usize) -> Vec<usize> {
        let mut by: Vec<(usize, f32)> = (0..self.postings.len())
            .map(|p| (p, sqdist(x, self.centroid(p))))
            .collect();
        let n = n.min(by.len());
        by.select_nth_unstable_by(n.saturating_sub(1), |a, b| a.1.total_cmp(&b.1));
        by.truncate(n);
        by.sort_by(|a, b| a.1.total_cmp(&b.1));
        by.into_iter().map(|(p, _)| p).collect()
    }

    /// The partition `x` belongs to.
    fn nearest(&self, x: &[f32]) -> usize {
        (0..self.postings.len())
            .map(|p| (p, sqdist(x, self.centroid(p))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(p, _)| p)
    }

    fn centroid(&self, p: usize) -> &[f32] {
        let dim = self.dim();
        &self.centroids[p * dim..(p + 1) * dim]
    }

    /// A new empty partition around `centroid`, which is already rotated.
    fn add_partition(&mut self, centroid: &[f32]) -> usize {
        self.centroids.extend_from_slice(centroid);
        self.postings.push(Posting::default());
        self.postings.len() - 1
    }

    /// Drop an empty partition, moving the last one into its place.
    fn drop_partition(&mut self, p: usize) {
        debug_assert_eq!(self.postings[p].len(), 0, "a partition is emptied first");
        let dim = self.dim();
        let last = self.postings.len() - 1;
        self.postings.swap_remove(p);
        for i in 0..dim {
            self.centroids[p * dim + i] = self.centroids[last * dim + i];
        }
        self.centroids.truncate(last * dim);
        if p != last {
            // The partition that used to be last is at `p` now, so everything
            // filed under it has to be told.
            for &id in &self.postings[p].ids {
                if let Some(slot) = self.at.get_mut(&id) {
                    slot.partition = p as u32;
                }
            }
        }
    }

    /// Append a member to a partition. `x` is rotated.
    fn place(&mut self, p: usize, id: u64, tag: u64, x: &[f32]) {
        let dim = self.dim();
        let width = self.quant.code_bytes();
        let slot = self.postings[p].len();
        self.postings[p].codes.resize((slot + 1) * width, 0);
        let centroid = &self.centroids[p * dim..(p + 1) * dim];
        let coded = self.quant.encode_rotated(
            x,
            centroid,
            &mut self.postings[p].codes[slot * width..(slot + 1) * width],
        );
        self.postings[p].ids.push(id);
        self.postings[p].tags.push(tag);
        self.postings[p].meta.push(coded);
        self.at.insert(
            id,
            Slot {
                partition: p as u32,
                slot: slot as u32,
            },
        );
    }

    /// Take slot `s` out of partition `p`, returning the id that moved into it.
    fn pull(&mut self, p: usize, s: usize) -> Option<u64> {
        let width = self.quant.code_bytes();
        let posting = &mut self.postings[p];
        let last = posting.len() - 1;
        posting.ids.swap_remove(s);
        posting.tags.swap_remove(s);
        posting.meta.swap_remove(s);
        if s != last {
            let (head, tail) = posting.codes.split_at_mut(last * width);
            head[s * width..(s + 1) * width].copy_from_slice(&tail[..width]);
        }
        posting.codes.truncate(last * width);
        (s != last).then(|| posting.ids[s])
    }

    /// The same, keeping the map straight, for the paths that are about to put
    /// the member somewhere else.
    fn pull_and_forget(&mut self, p: usize, s: usize) {
        let id = self.postings[p].ids[s];
        self.at.remove(&id);
        if let Some(moved) = self.pull(p, s) {
            self.at.insert(
                moved,
                Slot {
                    partition: p as u32,
                    slot: s as u32,
                },
            );
        }
    }
}

/// A member on its way from one partition to another, which is the only time
/// its id and its tag travel together without a posting around them.
#[derive(Clone, Copy)]
struct Member {
    id: u64,
    tag: u64,
}

enum Job {
    Split(usize),
    Merge(usize),
}

/// Two means over a set of vectors laid out end to end.
///
/// The seeds are the member furthest from the middle and then the member
/// furthest from that one, which is deterministic, needs no generator, and
/// starts on the axis the cloud is actually longest along. Eight rounds is
/// past where this stops moving on anything shaped like an embedding.
fn two_means(xs: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = xs.len() / dim;
    let mut middle = vec![0.0f32; dim];
    for i in 0..n {
        for (m, c) in middle.iter_mut().zip(&xs[i * dim..(i + 1) * dim]) {
            *m += c;
        }
    }
    for m in &mut middle {
        *m /= n as f32;
    }
    let far = |from: &[f32]| {
        (0..n)
            .max_by(|&i, &j| {
                sqdist(from, &xs[i * dim..(i + 1) * dim])
                    .total_cmp(&sqdist(from, &xs[j * dim..(j + 1) * dim]))
            })
            .unwrap_or(0)
    };
    let i = far(&middle);
    let mut a = xs[i * dim..(i + 1) * dim].to_vec();
    let j = far(&a);
    let mut b = xs[j * dim..(j + 1) * dim].to_vec();

    for _ in 0..8 {
        let mut sums = (vec![0.0f32; dim], vec![0.0f32; dim]);
        let mut counts = (0usize, 0usize);
        for i in 0..n {
            let x = &xs[i * dim..(i + 1) * dim];
            if sqdist(x, &a) <= sqdist(x, &b) {
                for (s, c) in sums.0.iter_mut().zip(x) {
                    *s += c;
                }
                counts.0 += 1;
            } else {
                for (s, c) in sums.1.iter_mut().zip(x) {
                    *s += c;
                }
                counts.1 += 1;
            }
        }
        // A side that ended up with nothing keeps the seed it had, because a
        // mean of no points is not a place and the next round would put every
        // member on the other side for ever.
        if counts.0 > 0 {
            for (m, s) in a.iter_mut().zip(&sums.0) {
                *m = s / counts.0 as f32;
            }
        }
        if counts.1 > 0 {
            for (m, s) in b.iter_mut().zip(&sums.1) {
                *m = s / counts.1 as f32;
            }
        }
    }
    (a, b)
}

/// The squared distance, because the square root is monotone and nothing here
/// needs it.
///
/// Written with eight running totals rather than one, and that is not a
/// flourish. Adding floats is not associative, so a compiler is not allowed to
/// turn a single accumulator into a vector of them, and the obvious one line
/// version is a chain of dependent adds four cycles apart. It is the hottest
/// loop in the file, because every insert measures a query against every
/// centroid and every search does it twice over, and writing it this way was
/// worth two thirds of what an insert cost at 768 dimensions.
///
/// The eight totals are summed in a fixed order at the end, so the answer is
/// deterministic. It is a different answer from the one line version, by the
/// last bit or so, in the same way that any two orderings of a float sum are.
fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    let mut totals = [0.0f32; 8];
    let mut i = 0;
    while i + 8 <= a.len() {
        for (k, total) in totals.iter_mut().enumerate() {
            let d = a[i + k] - b[i + k];
            *total += d * d;
        }
        i += 8;
    }
    let mut sum = 0.0f32;
    for total in totals {
        sum += total;
    }
    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// The record log, for a test: every vector by id, where the id is where it
    /// sits.
    struct Store(Vec<Vec<f32>>);

    impl Vectors for Store {
        fn get(&self, id: u64, into: &mut [f32]) -> bool {
            match self.0.get(id as usize) {
                Some(v) => {
                    into.copy_from_slice(v);
                    true
                }
                None => false,
            }
        }
    }

    /// A store with a hole in it, for the case where the log forgot something
    /// the index still thinks it has.
    struct Holey(Vec<Vec<f32>>, u64);

    impl Vectors for Holey {
        fn get(&self, id: u64, into: &mut [f32]) -> bool {
            if id == self.1 {
                return false;
            }
            match self.0.get(id as usize) {
                Some(v) => {
                    into.copy_from_slice(v);
                    true
                }
                None => false,
            }
        }
    }

    /// Vectors with the two things real embeddings have and uniform noise does
    /// not: a few coordinates carrying most of the energy, so that two vectors
    /// are genuinely near each other rather than all being equally far apart,
    /// and clusters, so that the partitioning has something to get right.
    ///
    /// A corpus without the first of those is not a hard test, it is an
    /// impossible one. The nearest ten of three hundred uniform points are
    /// arbitrary, no quantiser can pick them out, and the recall it measures
    /// says nothing about the index.
    fn corpus(dim: usize, n: usize, clusters: usize, seed: u64) -> Store {
        let mut rng = Rng::new(seed);
        let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(dim, &mut rng)).collect();
        Store(
            (0..n)
                .map(|i| {
                    let off = draw(dim, &mut rng);
                    let mut v: Vec<f32> = centres[i % clusters]
                        .iter()
                        .zip(&off)
                        .map(|(c, o)| c + o * 0.7)
                        .collect();
                    unit(&mut v);
                    v
                })
                .collect(),
        )
    }

    /// One vector of the shape above, unit length.
    fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| {
                let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
                let heavy = if i < dim / 16 { 6.0 } else { 1.0 };
                (u * 2.0 - 1.0) * heavy
            })
            .collect();
        unit(&mut v);
        v
    }

    fn unit(v: &mut [f32]) {
        let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
        for c in v {
            *c /= len;
        }
    }

    fn truth(store: &Store, q: &[f32], k: usize) -> Vec<u64> {
        let mut all: Vec<(u64, f32)> = store
            .0
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, sqdist(q, v)))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.truncate(k);
        all.into_iter().map(|(i, _)| i).collect()
    }

    /// Build an index over a whole store, running maintenance as it goes the way
    /// a maintenance slice would.
    fn build(store: &Store, dim: usize, tuning: Tuning) -> Partitions {
        let mut ix = Partitions::new(dim, Bits::One, 7, tuning);
        for (i, v) in store.0.iter().enumerate() {
            ix.insert(i as u64, v);
            if i % 64 == 0 {
                ix.maintain(store, 4096);
            }
        }
        ix.maintain(store, 1 << 20);
        ix
    }

    /// How often the true `k` nearest come back.
    fn recall(ix: &Partitions, store: &Store, k: usize, queries: usize) -> f32 {
        let mut hits = 0usize;
        for i in 0..queries {
            // A query near a real vector rather than anywhere at all, because
            // that is what a search looks like.
            let q = &store.0[i * 7 % store.0.len()];
            let want = truth(store, q, k);
            let got: Vec<u64> = ix.search(q, k, store).into_iter().map(|h| h.id).collect();
            hits += want.iter().filter(|id| got.contains(id)).count();
        }
        hits as f32 / (queries * k) as f32
    }

    /// Everything the index believes about itself, checked.
    fn consistent(ix: &Partitions) {
        assert_eq!(ix.centroids.len(), ix.postings.len() * ix.dim());
        let width = ix.quant.code_bytes();
        let mut seen = 0usize;
        for (p, posting) in ix.postings.iter().enumerate() {
            assert_eq!(posting.codes.len(), posting.len() * width, "partition {p}");
            assert_eq!(posting.meta.len(), posting.len(), "partition {p}");
            assert_eq!(posting.tags.len(), posting.len(), "partition {p}");
            for (s, id) in posting.ids.iter().enumerate() {
                let at = ix.at.get(id).expect("every member is in the map");
                assert_eq!(at.partition as usize, p, "id {id}");
                assert_eq!(at.slot as usize, s, "id {id}");
                seen += 1;
            }
        }
        assert_eq!(seen, ix.at.len(), "the map has entries with no member");
    }

    /// Build an index where every vector carries a tag, so the filter has
    /// something to meet.
    fn build_tagged(
        store: &Store,
        dim: usize,
        tuning: Tuning,
        tag: impl Fn(u64) -> u64,
    ) -> Partitions {
        let mut ix = Partitions::new(dim, Bits::One, 7, tuning);
        for (i, v) in store.0.iter().enumerate() {
            ix.insert_tagged(i as u64, v, tag(i as u64));
            if i % 64 == 0 {
                ix.maintain(store, 4096);
            }
        }
        ix.maintain(store, 1 << 20);
        ix
    }

    /// The whole point of pushing a filter into the scan, measured against the
    /// thing it replaces.
    ///
    /// One document in fifty carries the tag. Filtering inside the scan finds
    /// the true ten of those. Taking the best forty by vector and then throwing
    /// away the ones that do not match, which is what a search that cannot push
    /// a filter down has to do, finds almost none of them, and the ones it
    /// misses were nearer than the ones it kept.
    #[test]
    fn a_filter_in_the_scan_finds_what_a_filter_after_it_cannot() {
        let dim = 96;
        let store = corpus(dim, 3000, 12, 47);
        let tuning = Tuning {
            posting: 64,
            ..Tuning::default()
        };
        let wanted = |id: u64| id.is_multiple_of(50);
        let ix = build_tagged(&store, dim, tuning, |id| u64::from(wanted(id)));

        let (mut pushed, mut after) = (0usize, 0usize);
        let k = 10;
        for i in 0..40 {
            let q = &store.0[i * 71 % store.0.len()];

            // What the answer is: brute force over the members that match.
            let mut all: Vec<(u64, f32)> = store
                .0
                .iter()
                .enumerate()
                .filter(|(id, _)| wanted(*id as u64))
                .map(|(id, v)| (id as u64, sqdist(q, v)))
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: Vec<u64> = all[..k].iter().map(|(id, _)| *id).collect();

            let got: Vec<u64> = ix
                .search_where(q, k, &|tag: u64| tag == 1, &store)
                .into_iter()
                .map(|h| h.id)
                .collect();
            pushed += want.iter().filter(|id| got.contains(id)).count();

            let late: Vec<u64> = ix
                .search(q, k * tuning.rerank, &store)
                .into_iter()
                .map(|h| h.id)
                .filter(|id| wanted(*id))
                .take(k)
                .collect();
            after += want.iter().filter(|id| late.contains(id)).count();
        }
        let (pushed, after) = (pushed as f32 / 400.0, after as f32 / 400.0);
        assert!(pushed >= 0.95, "pushing the filter down gave {pushed}");
        assert!(
            after < pushed / 2.0,
            "filtering afterwards gave {after} against {pushed}, which is not the point being made"
        );
    }

    #[test]
    fn a_filter_that_matches_nothing_answers_nothing() {
        let dim = 64;
        let store = corpus(dim, 500, 4, 53);
        let ix = build_tagged(&store, dim, Tuning::default(), |_| 1);
        assert!(
            ix.search_where(&store.0[0], 10, &|tag: u64| tag == 2, &store)
                .is_empty()
        );
        // And the same filter matching everything is the unfiltered answer.
        let all = ix.search_where(&store.0[0], 10, &Any, &store);
        assert_eq!(all, ix.search(&store.0[0], 10, &store));
    }

    #[test]
    fn a_tag_survives_a_split_and_a_merge() {
        let dim = 64;
        let store = corpus(dim, 800, 6, 59);
        let tuning = Tuning {
            posting: 24,
            ..Tuning::default()
        };
        let mut ix = build_tagged(&store, dim, tuning, |id| id * 7 + 1);
        assert!(ix.partitions() > 4, "it never split");
        for id in 0..800u64 {
            assert_eq!(ix.tag(id), Some(id * 7 + 1), "id {id} after the splits");
        }

        // Now shrink it until partitions merge, and the survivors keep theirs.
        for id in 0..760u64 {
            ix.remove(id);
        }
        ix.maintain(&store, 1 << 20);
        consistent(&ix);
        for id in 760..800u64 {
            assert_eq!(ix.tag(id), Some(id * 7 + 1), "id {id} after the merges");
        }
        assert_eq!(ix.tag(0), None);
    }

    /// A selective filter means the answers are not in the nearest partitions,
    /// and a search that will not look further returns fewer than it should.
    #[test]
    fn a_selective_filter_makes_the_search_look_further() {
        let dim = 64;
        let store = corpus(dim, 2000, 10, 61);
        let tuning = Tuning {
            posting: 32,
            ..Tuning::default()
        };
        let tag = |id: u64| u64::from(id.is_multiple_of(100));
        let ix = build_tagged(&store, dim, tuning, tag);
        let narrow = build_tagged(&store, dim, Tuning { widen: 1, ..tuning }, tag);

        let mut wide_found = 0usize;
        let mut narrow_found = 0usize;
        for i in 0..20 {
            let q = &store.0[i * 91 % store.0.len()];
            wide_found += ix.search_where(q, 10, &|t: u64| t == 1, &store).len();
            narrow_found += narrow.search_where(q, 10, &|t: u64| t == 1, &store).len();
        }
        assert_eq!(
            wide_found, 200,
            "one in a hundred of two thousand is twenty"
        );
        assert!(
            narrow_found < wide_found,
            "not widening found {narrow_found} of {wide_found}"
        );
    }

    #[test]
    fn a_signature_never_rejects_something_it_should_have_matched() {
        let english = Signature::of(&[("lang", b"en")]);
        let doc = Signature::of(&[("lang", b"en"), ("topic", b"finance"), ("year", b"2026")]);
        assert!(doc.covers(english));
        assert!(english.allows(doc.bits()));
        assert_eq!(Signature::from_bits(doc.bits()), doc);

        // And over a lot of values, nothing that matches is ever turned away.
        for i in 0..500u32 {
            let value = i.to_string();
            let one = Signature::of(&[("id", value.as_bytes())]);
            let with = Signature::of(&[("id", value.as_bytes()), ("kind", b"page")]);
            assert!(with.covers(one), "value {value}");
        }
    }

    #[test]
    fn an_empty_index_answers_nothing() {
        let ix = Partitions::new(32, Bits::One, 1, Tuning::default());
        let store = Store(Vec::new());
        assert!(ix.is_empty());
        assert_eq!(ix.partitions(), 0);
        assert!(ix.search(&[0.0; 32], 10, &store).is_empty());
        assert!(!ix.needs_maintenance());
    }

    #[test]
    fn the_first_vector_is_the_first_partition() {
        let store = corpus(32, 1, 1, 3);
        let mut ix = Partitions::new(32, Bits::One, 1, Tuning::default());
        ix.insert(0, &store.0[0]);
        assert_eq!(ix.partitions(), 1);
        assert_eq!(ix.len(), 1);
        let hits = ix.search(&store.0[0], 5, &store);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance < 1e-6, "{}", hits[0].distance);
        consistent(&ix);
    }

    #[test]
    fn a_search_finds_what_brute_force_finds() {
        let dim = 128;
        let store = corpus(dim, 2000, 12, 5);
        let ix = build(&store, dim, Tuning::default());
        assert!(ix.partitions() > 1, "it never split");
        consistent(&ix);
        let r = recall(&ix, &store, 10, 50);
        assert!(r >= 0.95, "recall at 10 was {r}");
    }

    #[test]
    fn a_posting_that_grows_too_big_splits() {
        let dim = 64;
        let tuning = Tuning {
            posting: 32,
            ..Tuning::default()
        };
        let store = corpus(dim, 600, 6, 9);
        let ix = build(&store, dim, tuning);
        assert!(
            ix.partitions() >= 600 / (32 * 2),
            "600 vectors in {} partitions",
            ix.partitions()
        );
        for posting in &ix.postings {
            assert!(
                posting.len() <= 32 * 2,
                "a posting is {} long",
                posting.len()
            );
        }
        consistent(&ix);
    }

    #[test]
    fn a_posting_that_shrinks_merges() {
        let dim = 64;
        let tuning = Tuning {
            posting: 32,
            ..Tuning::default()
        };
        let store = corpus(dim, 600, 6, 9);
        let mut ix = build(&store, dim, tuning);
        let grown = ix.partitions();
        assert!(grown > 4);

        // Take away almost everything and let maintenance settle.
        for id in 0..570u64 {
            assert!(ix.remove(id));
        }
        ix.maintain(&store, 1 << 20);
        consistent(&ix);
        assert_eq!(ix.len(), 30);
        assert!(
            ix.partitions() < grown,
            "{} partitions for 30 vectors, was {grown}",
            ix.partitions()
        );
        // And it still answers.
        let hits = ix.search(&store.0[599], 1, &store);
        assert_eq!(hits[0].id, 599);
    }

    #[test]
    fn a_removed_vector_stops_coming_back() {
        let dim = 64;
        let store = corpus(dim, 400, 4, 11);
        let mut ix = build(&store, dim, Tuning::default());
        let q = store.0[7].clone();
        assert_eq!(ix.search(&q, 1, &store)[0].id, 7);

        assert!(ix.remove(7));
        assert!(!ix.remove(7), "removing it twice should say so");
        assert!(!ix.contains(7));
        assert_eq!(ix.len(), 399);
        consistent(&ix);
        assert!(ix.search(&q, 5, &store).iter().all(|h| h.id != 7));
    }

    #[test]
    fn inserting_the_same_id_twice_replaces_it() {
        let dim = 64;
        let store = corpus(dim, 200, 2, 13);
        let mut ix = build(&store, dim, Tuning::default());
        let before = ix.len();
        ix.insert(3, &store.0[3]);
        assert_eq!(ix.len(), before);
        consistent(&ix);
        assert_eq!(ix.search(&store.0[3], 1, &store)[0].id, 3);
    }

    /// A collection of copies of one vector has no cut in it, and maintenance
    /// has to notice that rather than try the same split for ever.
    ///
    /// This is not a corner case anybody has to go looking for. It is what a
    /// collection looks like when a pipeline embeds the same document a thousand
    /// times, and getting it wrong is a hang rather than a wrong answer.
    #[test]
    fn a_thousand_copies_of_one_vector_do_not_spin() {
        let dim = 32;
        let one = corpus(dim, 1, 1, 41).0.pop().expect("one vector");
        let store = Store(vec![one; 1000]);
        let tuning = Tuning {
            posting: 16,
            ..Tuning::default()
        };
        let mut ix = Partitions::new(dim, Bits::One, 7, tuning);
        for (i, v) in store.0.iter().enumerate() {
            ix.insert(i as u64, v);
            ix.maintain(&store, 4096);
        }
        ix.maintain(&store, 1 << 20);
        consistent(&ix);
        assert_eq!(ix.len(), 1000);
        assert!(!ix.needs_maintenance(), "it still thinks there is work");
        // And it still answers, with the exact distance rather than an estimate.
        let hits = ix.search(&store.0[0], 5, &store);
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|h| h.distance < 1e-6));
    }

    /// G13's actual claim. Recall is measured at the end of a long stream of
    /// writes and deletes rather than on a fresh build, because a fresh build is
    /// the measurement that hides drift.
    #[test]
    fn recall_holds_over_a_write_stream_with_no_rebuild() {
        let dim = 96;
        let store = corpus(dim, 3000, 15, 17);
        let tuning = Tuning {
            posting: 64,
            ..Tuning::default()
        };
        let mut ix = Partitions::new(dim, Bits::One, 7, tuning);

        // Write everything, and churn a tenth of it as we go, which is what
        // moves the centroids around under the members that are already filed.
        let mut rng = Rng::new(23);
        for (i, v) in store.0.iter().enumerate() {
            ix.insert(i as u64, v);
            if i > 100 && i % 10 == 0 {
                let victim = rng.below(i) as u64;
                ix.remove(victim);
                ix.insert(victim, &store.0[victim as usize]);
            }
            ix.maintain(&store, 512);
        }
        ix.maintain(&store, 1 << 20);
        consistent(&ix);
        assert_eq!(ix.len(), store.0.len());

        let r = recall(&ix, &store, 10, 60);
        assert!(r >= 0.95, "recall at 10 after the stream was {r}");
    }

    /// What the sweep is for, measured as the thing it actually fixes rather
    /// than through recall.
    ///
    /// Drift is members filed under a partition that is no longer their nearest,
    /// which is what a split leaves behind in the partitions around it. It shows
    /// up in recall eventually, but recall is a blunt instrument here and moves
    /// by a percent for reasons that have nothing to do with this, so the
    /// straight count is the honest measurement.
    #[test]
    fn the_sweep_is_what_keeps_members_under_their_nearest_centroid() {
        let dim = 96;
        let store = corpus(dim, 2000, 10, 29);
        let tuning = Tuning {
            posting: 48,
            ..Tuning::default()
        };
        let with = misfiled(&build(&store, dim, tuning), &store);
        let without = misfiled(&build(&store, dim, Tuning { sweep: 0, ..tuning }), &store);
        assert!(
            with * 4 < without,
            "sweeping left {with} members drifted and not sweeping left {without}"
        );
    }

    /// How many members are filed under something that is not their nearest
    /// centroid.
    fn misfiled(ix: &Partitions, store: &Store) -> usize {
        let mut buf = vec![0.0f32; ix.dim()];
        let mut wrong = 0;
        for (p, posting) in ix.postings.iter().enumerate() {
            for &id in &posting.ids {
                assert!(store.get(id, &mut buf));
                if ix.nearest(&ix.quant.rotate(&buf)) != p {
                    wrong += 1;
                }
            }
        }
        wrong
    }

    #[test]
    fn a_vector_the_log_forgot_is_dropped_rather_than_returned() {
        let dim = 64;
        let store = corpus(dim, 400, 4, 31);
        let tuning = Tuning {
            posting: 24,
            ..Tuning::default()
        };
        let mut ix = build(&store, dim, tuning);
        assert!(ix.contains(11));

        // The log loses one without telling the index, which is the state a
        // crash between two appends leaves behind.
        let holey = Holey(store.0.clone(), 11);
        assert!(
            ix.search(&store.0[11], 5, &holey)
                .iter()
                .all(|h| h.id != 11)
        );

        // And maintenance walking over it takes it out for good.
        for id in 0..300u64 {
            ix.remove(id);
        }
        ix.maintain(&holey, 1 << 20);
        consistent(&ix);
        assert!(!ix.contains(11));
    }

    #[test]
    fn rotating_first_is_the_same_as_rotating_inside() {
        // The whole index rests on the rotation being linear, so this is the
        // property, not an implementation detail.
        let dim = 128;
        let q = Quantizer::new(dim, Bits::One, 5);
        let store = corpus(dim, 2, 1, 37);
        let (v, c) = (&store.0[0], &store.0[1]);

        let mut a = vec![0u8; q.code_bytes()];
        let one = q.encode(v, c, &mut a);
        let mut b = vec![0u8; q.code_bytes()];
        let two = q.encode_rotated(&q.rotate(v), &q.rotate(c), &mut b);

        assert_eq!(a, b, "the two ways round should write the same code");
        assert!((one.norm - two.norm).abs() < 1e-4);
        assert!((one.correction - two.correction).abs() < 1e-4);
    }

    #[test]
    fn two_means_splits_two_clouds_apart() {
        let dim = 8;
        let mut xs = Vec::new();
        for i in 0..40 {
            let far = if i % 2 == 0 { 0.0 } else { 10.0 };
            for d in 0..dim {
                xs.push(far + (i as f32 + d as f32) * 0.01);
            }
        }
        let (a, b) = two_means(&xs, dim);
        let (near, away) = if a[0] < b[0] { (a, b) } else { (b, a) };
        assert!(near[0] < 1.0, "{near:?}");
        assert!(away[0] > 9.0, "{away:?}");
    }
}
