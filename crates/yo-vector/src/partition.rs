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
//! # What it costs to build
//!
//! `examples/ingest.rs` measures the rate at every doubling and splits it
//! between the insert and the maintenance, because a rate that falls as the
//! collection grows and a rate that is just low need different work and a single
//! number cannot tell them apart. On 128 dimensional vectors on one core of a
//! 13th Gen Intel Core i9-13900K with nothing else running:
//!
//! ```text
//!         at  partitions    a second      insert    maintain   touched
//!      12500          36      132060       20.7%       79.3%       5.3
//!      50000         132       94166       31.0%       69.0%       6.9
//!     200000         595       67764       49.1%       50.9%       5.5
//!     800000        2141       64107       52.8%       47.2%       4.4
//!    1600000        4337       59040       53.7%       46.3%       4.4
//! ```
//!
//! `touched` is how many vectors maintenance moved or looked at per vector
//! inserted. It is flat, and that is the number which says the update protocol
//! is doing bounded work rather than quietly turning into a rebuild.
//!
//! Five fixes got it there and they were five different problems.
//! Maintenance was 80 percent of the time and most of it was `sweep` measuring
//! every member it looked at against every centroid in the collection, which is
//! not what LIRE says and is several full scans per vector inserted. The insert
//! was the other half and it was a scan over every centroid by definition, which
//! is why the coarse layer in `src/coarse.rs` is there, and that file is where
//! the reasoning about it lives. Before either fix, the rate halved on every
//! doubling and was 13563 a second by 800 thousand.
//!
//! The third was the squared distance itself, which by then was half of an
//! entire ingest across the two copies of it that existed, and it was slow for
//! a reason that had nothing to do with the index: a bounds check the compiler
//! could not remove was stopping the loop vectorising. There is now one copy in
//! `src/dist.rs` and that file is where the reasoning lives. It doubled the
//! rate on its own, 21183 a second to 40318 over the whole 1.6 million on the
//! same machine, and it cut the insert half by three times, 45.9 seconds to
//! 15.3, which is why insert and maintenance have swapped places in that table.
//!
//! The fourth was `Partitions::job`, which used to ask its two questions by
//! walking every partition twice, once per vector inserted. That is the same
//! quadratic the coarse layer exists to remove, hiding one level up, and by 1.6
//! million vectors it was around a fifth of an ingest spent deciding there was
//! nothing to do. It is the two candidate lists now.
//!
//! The fifth was the rotation, in `src/rotate.rs`, which unpacked a sign bit
//! with a shift and a mask inside the loop and then branched on a random bit
//! per pair. Turning a pair is the same as flipping the sign of its second
//! coordinate, so the branch folds into the sign table at build time. Those two
//! together took the whole 1.6 million from 40318 a second to 64647, and the
//! maintenance half from 24.4 seconds to 11.6.
//!
//! So G13's fifty thousand a second per core is met on the machine it is called
//! on, and it is met at every size in the table rather than only at the small
//! end. The two halves are close to even now, 53.7 percent insert against 46.3
//! percent maintenance at the far end, so neither one is the obvious next thing
//! to go and look at.
//!
//! # What is not here yet
//!
//! A checkpoint that writes the image out. [`crate::image`] is the layout and
//! the two halves of the round trip, and the seam it comes back through is a
//! pair of crate private calls further down this file, so an index survives a
//! restart without requantising anything. What is still missing is the shard
//! side: deciding when to write one, and pointing a checkpoint entry at it.
//!
//! MS-MARCO-v2. SIFT1M on a 13900K now gets recall 0.9597 at probe 64 rerank
//! 16 with p50 at 638 us and p99 at 776 us, so both halves of G12 are met on
//! that dataset, and the same run before the `src/dist.rs` change was 808 us
//! and 996 us for the same recall, which was inside the millisecond by so
//! little that nobody should have called it. The other dataset the gate names
//! has not been run.
//!
//! `examples/search.rs` is the breakdown of where the remaining time goes and
//! the answer is that two thirds of it is the estimator meeting one code at a
//! time.
//!
//! The commands that put all of this on the wire are the rest of M6.

use std::collections::{HashMap, HashSet};

use yo_common::{Code, Error, Result};

use crate::coarse::Coarse;
use crate::dist::sqdist;
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// How many partitions in a row may add nothing to the answer before the
    /// search stops reading, once it has enough candidates to answer with.
    ///
    /// [`Tuning::probe`] is a budget every query spends whether it needs to or
    /// not, and queries do not need the same amount. A query sitting deep inside
    /// one partition has found everything it is going to find after two or three
    /// of them, and a query on a boundary is still turning up better answers
    /// forty partitions in. This is what lets one search cost what it needs
    /// rather than what the slowest query needs, and it is the whole of the
    /// difference between a mean probe depth and a fixed one.
    ///
    /// It keys off the answer rather than off the geometry, which is deliberate.
    /// The obvious rule is to stop once the next centroid is more than some
    /// fraction further away than the nearest one, and that rule is useless
    /// here: see [`Partitions::spill_into`] for the measurement, but the short
    /// version is that distances concentrate, every centroid a query can see is
    /// within a few percent of every other, and there is no setting of the
    /// fraction between pruning nothing and pruning everything.
    ///
    /// A partition counts as adding nothing when not one of its members was good
    /// enough to displace an answer already held. The count resets the moment one
    /// is, so a run of empty partitions followed by a good one buys the search
    /// its patience back. Zero switches this off and every search reads `probe`
    /// partitions.
    ///
    /// It cannot change how many candidates come back, only which ones, because
    /// it is only allowed to fire once there are already enough. A filtered
    /// search that is widening because it does not have enough is never cut off
    /// by it.
    pub patience: usize,
    /// How many partitions one vector may be written into, at most.
    ///
    /// One is no replication and is what the index did before this existed. It
    /// is not the default, and `src/miss.rs` is why.
    ///
    /// A vector belongs to the partition whose centroid it is nearest, and on
    /// some data that is a much weaker statement than it sounds. Measured on a
    /// million MS-MARCO passage embeddings, only 0.8952 of the true nearest
    /// neighbours of a query sat in one of the 128 partitions the search reads,
    /// and the recall the search actually returned was 0.8942, so the whole of
    /// the miss was neighbours nobody looked at rather than anything the
    /// estimator did. A vector near the boundary between two partitions is one
    /// query away from being in the wrong one, and no amount of scanning fixes
    /// that because the scan never gets there.
    ///
    /// So a vector near a boundary goes in both, which is SPANN's answer.
    /// Raising this raises recall and costs memory and scan time in proportion
    /// to how many vectors actually qualify, which is what [`Tuning::slack`]
    /// controls.
    pub spill: usize,
    /// How much further than the nearest centroid a vector will still be copied
    /// into, as a fraction.
    ///
    /// A vector goes into every one of its [`Tuning::spill`] nearest partitions
    /// whose centroid is within `1 + slack` of the nearest one, so zero is no
    /// replication whatever `spill` says and a large value replicates
    /// everything into everything. It is a distance ratio rather than a count
    /// because the thing being asked is whether a vector is genuinely near a
    /// boundary, and a vector sitting squarely inside its partition should cost
    /// one copy however large `spill` is.
    pub slack: f32,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            posting: 256,
            probe: 8,
            rerank: 4,
            sweep: 4,
            widen: 8,
            spill: 4,
            slack: 0.10,
            patience: 0,
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
/// A tag that is only a summary needs a second test somewhere, and the place
/// for it is [`Filter::exact`], which sees the member's id and can go and read
/// whatever the caller keyed by it. That runs only for members the tag let
/// through that are also near enough to be ranked, which is why it is allowed to
/// be the expensive one: an expression over a JSON string is fine there and
/// would not be fine in the scan.
pub trait Filter {
    /// Whether a member with this tag is worth ranking.
    fn allows(&self, tag: u64) -> bool;

    /// The second test, on the member's id rather than on its tag.
    ///
    /// Everything lets everything through by default, because for a filter whose
    /// tag says the whole truth there is nothing left to ask. Override it when
    /// the tag is a summary and the real predicate lives in a table of the
    /// caller's, and keep the tag test as the cheap superset of it: a member the
    /// tag rejects never reaches here.
    fn exact(&self, _id: u64) -> bool {
        true
    }
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
        let mut got = Signature(0);
        for (attribute, value) in values {
            got.insert(attribute, value);
        }
        got
    }

    /// Add one attribute and value pair to what this signature covers.
    ///
    /// For a caller that meets the pairs one at a time rather than holding them
    /// all in a slice, which is what building a tag out of a document's indexed
    /// fields looks like.
    pub fn insert(&mut self, attribute: &str, value: &[u8]) {
        self.insert_bytes(attribute.as_bytes(), value);
    }

    /// The same for an attribute that is already bytes, which is what a document
    /// path is.
    pub fn insert_bytes(&mut self, attribute: &[u8], value: &[u8]) {
        self.0 |= 1u64 << (hash(attribute, value) % 64);
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

/// What one search actually read.
///
/// [`Tuning::probe`] is a budget rather than a bill, and with
/// [`Tuning::patience`] set the two are different for most queries. This is the
/// bill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Work {
    /// Partitions read.
    pub probed: usize,
    /// Coded members the estimator was run over, which is the number that
    /// actually tracks the time a search took. Partitions are not the same size
    /// as each other and a count of them hides that.
    pub scanned: usize,
}

/// Where one copy of a member sits, and where the next copy of it is.
///
/// A vector is in one posting most of the time and in several when it sits near
/// the boundary between them, which is what [`Tuning::spill`] is for. So an id
/// does not map to a place, it maps to a chain of them, threaded through an
/// arena so that a chain of one costs what the single slot used to cost and a
/// replicated id costs twelve more bytes per copy rather than an allocation.
#[derive(Debug, Clone, Copy)]
struct Place {
    partition: u32,
    slot: u32,
    /// The next copy of the same vector, or [`END`]. Also the free list.
    next: u32,
}

/// The end of a chain, and the empty free list.
const END: u32 = u32::MAX;

/// One partition's members: the ids, their codes end to end, and what each code
/// needs beside it.
#[derive(Debug, Default)]
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
#[derive(Debug)]
pub struct Partitions {
    quant: Quantizer,
    tuning: Tuning,
    /// The centroids, already rotated, `dim` floats each end to end.
    centroids: Vec<f32>,
    postings: Vec<Posting>,
    /// The head of every id's chain of placements, which is what makes a delete
    /// a constant time operation rather than a search.
    at: HashMap<u64, u32>,
    /// The placements themselves, and the free list through their `next`.
    ///
    /// One arena rather than a list per id, because most ids have exactly one
    /// placement and a `Vec` each would be a million allocations on a million
    /// vectors to hold one entry apiece.
    places: Vec<Place>,
    free: u32,
    /// The index over the centroids. See [`crate::coarse`].
    coarse: Coarse,
    /// The shortlist a placement fills in, kept here so that placing a vector
    /// does not allocate.
    scratch: Vec<u32>,
    /// One member's copies, and the partitions an insert is about to spill
    /// into, kept for the same reason `scratch` is.
    spare: Vec<Place>,
    spill: Vec<(usize, f32)>,
    /// Partitions that may be over the split threshold, and partitions that may
    /// be under the merge threshold.
    ///
    /// Deciding what to maintain next used to be two passes over every
    /// partition, once for the largest and once for the smallest, and it ran
    /// once per vector inserted. That is linear in the partition count and the
    /// partition count grows with the collection, which is the same quadratic
    /// the coarse layer was built to remove, just hiding somewhere else. By 1.6
    /// million vectors it was a fifth of an ingest and it was doing nothing at
    /// all almost every time it ran.
    ///
    /// A partition can only cross a threshold when its own length changes, and
    /// there are five places a length changes, so the crossings can be recorded
    /// as they happen instead of looked for afterwards. These two lists hold
    /// every partition that qualifies and usually nothing else. They are
    /// allowed to hold stale entries, because [`Partitions::job`] checks the
    /// real length before it returns anything and drops what no longer
    /// qualifies, and they are kept in partition order so that a tie is broken
    /// the same way the two passes broke it.
    big: Vec<u32>,
    small: Vec<u32>,
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
            places: Vec::new(),
            free: END,
            coarse: Coarse::default(),
            scratch: Vec::new(),
            spare: Vec::new(),
            spill: Vec::new(),
            big: Vec::new(),
            small: Vec::new(),
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

    /// How many coded members the postings hold between them.
    ///
    /// The same as [`Partitions::len`] until [`Tuning::spill`] puts a vector
    /// near a boundary into more than one partition, and the ratio of the two
    /// is what replication is costing in memory and in scan time.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.postings.iter().map(Posting::len).sum()
    }

    /// The knobs.
    #[must_use]
    pub fn tuning(&self) -> Tuning {
        self.tuning
    }

    /// Change the knobs on a collection that already has vectors in it.
    ///
    /// [`Tuning::probe`], [`Tuning::rerank`] and [`Tuning::widen`] are read by
    /// each search, so they take effect on the next one. That is what makes a
    /// recall against latency curve measurable on one built index rather than on
    /// one built per row, and it is what `EF_RUNTIME` means to a client that
    /// thinks it is talking to a graph.
    ///
    /// [`Tuning::posting`] and [`Tuning::sweep`] are what maintenance aims at,
    /// so lowering `posting` does not split anything by itself. The partitions
    /// move towards the new size as [`Partitions::maintain`] gets called, which
    /// is the same way they got to the old one.
    pub fn retune(&mut self, tuning: Tuning) {
        self.tuning = tuning;
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
        if self.postings.is_empty() {
            // The first vector is the first centroid. There is nothing to
            // average it with yet, and the first split is what starts the
            // centroids being means rather than members.
            let p = self.add_partition(&x);
            self.place(p, id, tag, &x);
            return;
        }
        let mut into = core::mem::take(&mut self.spill);
        self.spill_into(&x, &mut into);
        for &(p, _) in &into {
            self.place(p, id, tag, &x);
        }
        self.spill = into;
    }

    /// The partitions a vector goes into, nearest first.
    ///
    /// The first is the one it belongs to and there is always exactly one of
    /// those. The rest are the boundary copies [`Tuning::spill`] is about: every
    /// further partition whose centroid is within [`Tuning::slack`] of the
    /// nearest one, up to `spill` of them in total.
    ///
    /// # The rule that is not here
    ///
    /// SPANN has a third condition, and it was written, measured and taken out
    /// again. It drops a candidate if some partition already chosen is nearer to
    /// it than the vector is, on the grounds that a candidate on the far side of
    /// one already taken adds a copy in a direction that is already covered.
    /// That is the rule that keeps SPANN's replication factor down.
    ///
    /// It rejects every candidate there is at a thousand dimensions. On two
    /// hundred thousand generated 1024 dimensional vectors in 528 partitions,
    /// the copy rate with the rule in is 1.0000 at every setting of `spill` and
    /// `slack` that was tried, and without it 2.85 at a `spill` of 4 and 5.08 at
    /// 8. A million MS-MARCO passages say the same thing from the other end:
    /// 1.000 copies a vector at `spill` 4 and 1.001 at `spill` 8 with `slack` at
    /// 0.60, which is a feature that is switched on and doing nothing.
    ///
    /// The reason is the one [`coarse`](crate::coarse) already ran into.
    /// Distances concentrate, and a centroid is the mean of a few hundred
    /// members so it sits well inside a cloud whose radius is most of the
    /// distance to the next centroid. Two neighbouring centroids are therefore
    /// much closer to each other than any of their members is to either, the
    /// condition holds for every pair, and nothing is ever copied. Keeping a
    /// rule that fires on nothing would have left the whole feature switched on
    /// and inert, which is worse than not having it.
    ///
    /// What is left holds the replication factor down instead: `spill` caps it
    /// outright and `slack` cuts off candidates that are not really boundary
    /// cases. On this data `spill` is what binds, because `slack` at 0.15 and at
    /// 0.35 give copy rates of 2.8500 and 2.8452, which is the same
    /// concentration seen from the other side.
    fn spill_into(&mut self, x: &[f32], into: &mut Vec<(usize, f32)>) {
        into.clear();
        let dim = self.dim();
        let want = self.tuning.spill.max(1);
        // The coarse layer's shortlist rather than every centroid, which is what
        // keeps an insert from costing what a search costs. It is at least 256
        // partitions wide, so the nearest handful of them are in there.
        let mut short = core::mem::take(&mut self.scratch);
        if want == 1 || self.tuning.slack <= 0.0 {
            let p = self.roughly_nearest(x, &mut short);
            self.scratch = short;
            into.push((p, 0.0));
            return;
        }
        self.coarse.shortlist(x, dim, &mut short);
        let mut near: Vec<(usize, f32)> = if short.is_empty() {
            (0..self.postings.len())
                .map(|p| (p, sqdist(x, self.centroid(p))))
                .collect()
        } else {
            short
                .iter()
                .map(|&p| (p as usize, sqdist(x, self.centroid(p as usize))))
                .collect()
        };
        self.scratch = short;
        near.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let Some(&(first, best)) = near.first() else {
            return;
        };
        into.push((first, best));
        // Squared distances throughout, so the ratio on distances is the square
        // of it here. Comparing the squares directly and skipping two square
        // roots per candidate is worth it on a path that runs once per insert.
        let ceiling = best * (1.0 + self.tuning.slack) * (1.0 + self.tuning.slack);
        for &(q, d) in near.iter().skip(1) {
            if into.len() >= want {
                break;
            }
            if d > ceiling {
                break;
            }
            into.push((q, d));
        }
    }

    /// The tag `id` was inserted with, if it is still here.
    #[must_use]
    pub fn tag(&self, id: u64) -> Option<u64> {
        // Any copy will do. Every copy of a member carries the same tag, which
        // is what [`Partitions::retag`] is for.
        let at = self.any_place(id)?;
        Some(self.postings[at.partition as usize].tags[at.slot as usize])
    }

    /// Change the tag `id` carries, saying whether it was there.
    ///
    /// The tag sits beside the code and nothing about the placement depends on
    /// it, so this is one write and no maintenance. That is what makes it
    /// affordable to recompute every tag in a collection when the thing the tag
    /// summarises changes, which for a document index is a field being indexed
    /// or stopping being indexed.
    pub fn retag(&mut self, id: u64, tag: u64) -> bool {
        let mut walk = self.at.get(&id).copied().unwrap_or(END);
        let mut found = false;
        while walk != END {
            let place = self.places[walk as usize];
            self.postings[place.partition as usize].tags[place.slot as usize] = tag;
            found = true;
            walk = place.next;
        }
        found
    }

    /// Take a vector out, saying whether it was there.
    ///
    /// The last member of the posting moves into the hole. There is no
    /// tombstone, so there is nothing to accumulate and nothing to compact.
    pub fn remove(&mut self, id: u64) -> bool {
        let mut copies = core::mem::take(&mut self.spare);
        self.every_place(id, &mut copies);
        if copies.is_empty() {
            self.spare = copies;
            return false;
        }
        self.detach_all(id);
        // Highest slot first inside a partition, because pulling a member moves
        // the last one into its slot, and a copy of this same id sitting at a
        // higher slot in the same posting would have its recorded slot go stale.
        // There is at most one copy per partition so this only matters across
        // them, but the order costs nothing and the alternative is a rule that
        // has to stay true.
        copies.sort_unstable_by_key(|c| core::cmp::Reverse(c.slot));
        for copy in &copies {
            let p = copy.partition as usize;
            let s = copy.slot as usize;
            if let Some(moved) = self.pull(p, s) {
                self.reslot(moved, p, s);
            }
        }
        self.spare = copies;
        true
    }

    /// Whether `id` is in the collection.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.at.contains_key(&id)
    }

    // -- what an image is made of -------------------------------------------
    //
    // [`crate::image`] writes an index down and reads it back, and it lives in
    // its own file because the layout it writes is the format's business rather
    // than the index's. These four are the seam between the two: everything
    // above is private on purpose and none of it is worth making public just so
    // that a sibling module can copy it into a buffer.

    /// Every centroid, already rotated, `dim` floats each end to end.
    pub(crate) fn all_centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// One partition's four parallel arrays, and the size at which its last
    /// split gave up.
    pub(crate) fn posting_parts(&self, p: usize) -> (&[u64], &[u64], &[u8], &[Coded], usize) {
        let posting = &self.postings[p];
        (
            &posting.ids,
            &posting.tags,
            &posting.codes,
            &posting.meta,
            posting.stuck,
        )
    }

    /// Put a whole partition back, centroid and members together.
    ///
    /// The centroid goes on the end of the run and the members go into a new
    /// posting, so partitions come back in the order they were written and an
    /// id keeps the partition number it had. Nothing is requantised and nothing
    /// is measured: an image holds the codes, and recomputing them from the
    /// vectors would be the rebuild this whole index exists to not do.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the four arrays do not describe the same members or
    /// if an id is already in the index.
    pub(crate) fn absorb(
        &mut self,
        centroid: &[f32],
        ids: Vec<u64>,
        tags: Vec<u64>,
        codes: Vec<u8>,
        meta: Vec<Coded>,
        stuck: usize,
    ) -> Result<()> {
        let width = self.quant.code_bytes();
        if centroid.len() != self.dim()
            || tags.len() != ids.len()
            || meta.len() != ids.len()
            || codes.len() != ids.len() * width
        {
            return Err(Error::new(
                Code::Corrupt,
                "the parts of a partition do not describe the same members",
            )
            .with_detail(format!(
                "centroid={} ids={} tags={} codes={} meta={}",
                centroid.len(),
                ids.len(),
                tags.len(),
                codes.len(),
                meta.len()
            )));
        }
        let p = self.postings.len();
        for (slot, &id) in ids.iter().enumerate() {
            // An id in two partitions is a replicated member and is what an
            // image of a spilled collection looks like. An id twice in one
            // partition is not, and `attach` is where that is caught, because
            // it is the shape a delete cannot undo.
            if !self.attach(id, p, slot) {
                return Err(Error::new(
                    Code::Corrupt,
                    "an id is twice in one partition of an image",
                )
                .with_detail(format!("id={id} partition={p}")));
            }
        }
        self.centroids.extend_from_slice(centroid);
        self.postings.push(Posting {
            ids,
            tags,
            codes,
            meta,
            stuck,
        });
        Ok(())
    }

    /// Say that a load is over, so the derived parts can be built once.
    ///
    /// The coarse layer and the two maintenance candidate lists are the whole of
    /// what an image does not carry, because both are decided by the centroids
    /// and the posting lengths that it does carry. Building them here is one
    /// pass rather than the running updates the insert path makes, which is the
    /// difference between a load being linear and being quadratic.
    pub(crate) fn finish_image(&mut self) {
        let dim = self.quant.dim();
        let n = self.postings.len();
        self.coarse.rebuild(&self.centroids, dim, n);
        self.big.clear();
        self.small.clear();
        for p in 0..n {
            if self.over(p) {
                self.big.push(p as u32);
            }
            if self.under(p) {
                self.small.push(p as u32);
            }
        }
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
        self.search_costed(q, k, filter, vectors).0
    }

    /// The same again, and what the scan behind it cost.
    ///
    /// See [`Work`]. Worth having in front of a caller rather than behind a
    /// feature flag, because with [`Tuning::patience`] set the cost of a search
    /// is a property of the query and not of the settings, and a tuner that
    /// cannot see it is guessing.
    ///
    /// # Panics
    ///
    /// If `q` is not [`Partitions::dim`] long.
    #[must_use]
    pub fn search_costed(
        &self,
        q: &[f32],
        k: usize,
        filter: &impl Filter,
        vectors: &impl Vectors,
    ) -> (Vec<Hit>, Work) {
        if k == 0 {
            return (Vec::new(), Work::default());
        }
        let (candidates, work) =
            self.candidates_costed(q, (k * self.tuning.rerank).max(FLOOR), filter);
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
        (hits, work)
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
        self.candidates_costed(q, want, filter).0
    }

    /// The same again, and what reading them cost.
    ///
    /// The cost is here because [`Tuning::patience`] makes it vary from one
    /// query to the next, and a knob whose whole point is that different queries
    /// pay different amounts is not one anybody can set without being able to see
    /// what it did. It is also the honest way to compare two settings: recall
    /// against partitions actually read, rather than recall against the budget
    /// neither of them spent.
    ///
    /// # Panics
    ///
    /// If `q` is not [`Partitions::dim`] long.
    #[must_use]
    pub fn candidates_costed(
        &self,
        q: &[f32],
        want: usize,
        filter: &impl Filter,
    ) -> (Vec<(u64, f32)>, Work) {
        assert_eq!(
            q.len(),
            self.dim(),
            "this collection holds {} dimensional vectors and was handed {}",
            self.dim(),
            q.len()
        );
        if want == 0 || self.postings.is_empty() {
            return (Vec::new(), Work::default());
        }
        // Rotated once here and never again, which is what lets a search probe
        // tens of partitions without paying for tens of rotations.
        let u = self.quant.rotate(q);
        let mut best = Bounded::new(want);
        // One buffer for the whole search rather than one per partition, and
        // grown rather than cleared, because every partition after the first
        // wants the same room the one before it did.
        let mut scores: Vec<f32> = Vec::new();
        let reach = self.tuning.probe.saturating_mul(self.tuning.widen.max(1));
        let mut work = Work::default();
        // How many partitions in a row have gone by without one of their members
        // being good enough to displace an answer. See [`Tuning::patience`].
        let mut quiet = 0;
        for (n, p) in self.near_partitions(&u, reach).into_iter().enumerate() {
            // Two reasons to stop, and both of them need enough answers in hand
            // first. Past the partitions an unfiltered search would have read,
            // keep going only while there is still not enough to answer with; an
            // unfiltered search never gets here, because the first `probe`
            // partitions of a collection worth probing hold more than `want`.
            // Inside them, stop once the last few have added nothing.
            if best.full()
                && (n >= self.tuning.probe
                    || (self.tuning.patience > 0 && quiet >= self.tuning.patience))
            {
                break;
            }
            let prepared = self.quant.query_rotated(&u, self.centroid(p));
            let posting = &self.postings[p];
            let held = posting.ids.len();
            if scores.len() < held {
                scores.resize(held, 0.0);
            }
            // The whole posting at once, so the estimator's inner loops know
            // how wide a code is. Then a second pass, which for most members is
            // one comparison against the worst answer so far and no more, and
            // which does not read the id or the tag of a member that lost.
            prepared.scan(&posting.codes, &posting.meta, &mut scores[..held]);
            work.probed += 1;
            work.scanned += held;
            let mut took = 0;
            for (i, &at) in scores[..held].iter().enumerate() {
                if !best.wants(at) {
                    continue;
                }
                if !filter.allows(posting.tags[i]) {
                    continue;
                }
                if !filter.exact(posting.ids[i]) {
                    continue;
                }
                best.put(posting.ids[i], at);
                took += 1;
            }
            // A partition that displaced something buys the search its patience
            // back, because a run of empty ones followed by a good one is the
            // shape of a query whose neighbourhood is spread out rather than a
            // query that has finished.
            quiet = if took == 0 { quiet + 1 } else { 0 };
        }
        let mut out = best.sorted();
        // A replicated member is in more than one posting and a search can read
        // more than one of them, so the same id can be ranked twice, with two
        // different estimates because each copy is coded against its own
        // centroid. The near duplicates are not adjacent for that reason, so
        // this is a pass with a set rather than a `dedup`.
        //
        // Only when there is replication to undo. With `spill` at one there can
        // be no duplicate, and a search that pays for proving it every time is
        // charging every collection for a feature some of them do not use.
        if self.tuning.spill > 1 {
            let mut seen = HashSet::with_capacity(out.len());
            out.retain(|&(id, _)| seen.insert(id));
        }
        (out, work)
    }

    /// Whether there is a split or a merge waiting.
    #[must_use]
    pub fn needs_maintenance(&self) -> bool {
        // The same two questions [`Partitions::job`] asks, without the pruning,
        // so that asking does not need the collection mutably.
        self.big.iter().any(|&p| {
            let p = p as usize;
            self.over(p) && self.postings[p].len() > self.postings[p].stuck
        }) || (self.postings.len() > 1 && self.small.iter().any(|&p| self.under(p as usize)))
    }

    /// Whether partition `p` is big enough to split. False for an index that is
    /// no longer there, which is what a stale candidate looks like.
    fn over(&self, p: usize) -> bool {
        self.postings
            .get(p)
            .is_some_and(|posting| posting.len() > self.tuning.posting * 2)
    }

    /// Whether partition `p` is small enough to merge away.
    fn under(&self, p: usize) -> bool {
        self.postings
            .get(p)
            .is_some_and(|posting| posting.len() * 4 < self.tuning.posting)
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

    /// Note that partition `p`'s length has changed, so it may have crossed a
    /// threshold in either direction.
    ///
    /// Both lists are checked, rather than only the one the direction of the
    /// change could have reached, because every caller would otherwise have to
    /// know which way it moved the length and one of them moves it both ways.
    /// They are short enough that looking is free.
    fn note(&mut self, p: usize) {
        let (over, under) = (self.over(p), self.under(p));
        let p = p as u32;
        if over && !self.big.contains(&p) {
            self.big.push(p);
        }
        if under && !self.small.contains(&p) {
            self.small.push(p);
        }
    }

    /// The next thing worth doing, biggest problem first.
    ///
    /// This used to walk every partition twice. It now walks the two candidate
    /// lists, which hold every partition that qualifies and are usually empty,
    /// and drops the entries that have stopped qualifying on the way past. The
    /// answer is the same one the walk gave, including which partition is
    /// picked when two are the same size, because the lists are in partition
    /// order and a maximum takes the last of equals where a minimum takes the
    /// first.
    fn job(&mut self) -> Option<Job> {
        let mut big = std::mem::take(&mut self.big);
        big.retain(|&p| self.over(p as usize));
        big.sort_unstable();
        let split = big
            .iter()
            .map(|&p| p as usize)
            .filter(|&p| self.postings[p].len() > self.postings[p].stuck)
            .max_by_key(|&p| self.postings[p].len());
        self.big = big;
        if let Some(split) = split {
            return Some(Job::Split(split));
        }

        if self.postings.len() > 1 {
            let mut small = std::mem::take(&mut self.small);
            small.retain(|&p| self.under(p as usize));
            small.sort_unstable();
            let merge = small
                .iter()
                .map(|&p| p as usize)
                .min_by_key(|&p| self.postings[p].len());
            self.small = small;
            if let Some(merge) = merge {
                return Some(Job::Merge(merge));
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
        self.coarse.moved(p, &a, dim);
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
    ///
    /// # Why a member is only measured against what changed
    ///
    /// Every member is already filed under the centroid it was nearest to, and a
    /// split moves one centroid and adds one. Nothing else moved, so for a member
    /// of some other partition the nearest of all the centroids that did not
    /// change is still the one it is already under, and the only way it can have
    /// a new answer is if one of the two new centroids beats that. That is a
    /// comparison against two, not a search over all of them.
    ///
    /// This is not a shortcut, it is what LIRE says, and getting it wrong is
    /// expensive in a way that is easy to miss. A sweep after a split walks about
    /// four partitions' worth of members, and a split happens every posting's
    /// worth of inserts, so a full centroid scan per member works out at several
    /// scans of every centroid in the collection per vector inserted. That is the
    /// whole ingest cost at any size worth talking about: measured on 128
    /// dimensional vectors it was 74 thousand a second at twelve thousand vectors
    /// and 13 thousand at two hundred thousand, with maintenance three quarters
    /// of it, and `examples/ingest.rs` is the harness that says so.
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
        // Copied out because placing a member borrows the index, and safe to
        // copy because nothing below here moves a centroid: `place` appends a
        // code to a posting and leaves the centroids alone.
        let fresh: Vec<(usize, Vec<f32>)> = changed
            .iter()
            .map(|&p| (p, self.centroid(p).to_vec()))
            .collect();
        let mut seen = 0;
        let mut buf = vec![0.0f32; dim];
        for p in look {
            let here = self.centroid(p).to_vec();
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
                let mut best = (p, sqdist(&x, &here));
                for (q, centre) in &fresh {
                    let d = sqdist(&x, centre);
                    if d < best.1 {
                        best = (*q, d);
                    }
                }
                if best.0 != p {
                    self.pull_and_forget(p, i);
                    self.place(best.0, id, tag, &x);
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
        self.note(p);
        let mut kept = Vec::with_capacity(ids.len());
        let mut xs = Vec::with_capacity(ids.len() * dim);
        let mut buf = vec![0.0f32; dim];
        for (id, tag) in ids.into_iter().zip(tags) {
            // Only this partition's copy. A member replicated into a partition
            // that is not the one being emptied keeps the copy it has there.
            self.detach(id, p);
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

    /// The partitions in the order a search would probe them, nearest centroid
    /// first, all of them.
    ///
    /// Test only, and it exists for [`miss`](crate::miss), which asks how far
    /// down this order a query's true neighbours sit. That is the measurement
    /// that says whether the recall gate wants better partitions or a better
    /// estimator, and it cannot be taken from outside the crate because the
    /// probe order is not something a caller has any business seeing.
    #[cfg(test)]
    pub(crate) fn probe_order(&self, q: &[f32], into: &mut Vec<usize>) {
        let u = self.quant.rotate(q);
        *into = self.near_partitions(&u, self.postings.len());
    }

    /// Which partition holds `id`, if any.
    #[cfg(test)]
    pub(crate) fn holder(&self, id: u64) -> Option<usize> {
        self.any_place(id).map(|s| s.partition as usize)
    }

    /// The partition `x` belongs to, as far as the coarse layer can tell.
    fn roughly_nearest(&self, x: &[f32], short: &mut Vec<u32>) -> usize {
        if !self.coarse.ready() {
            return self.nearest(x);
        }
        self.coarse.shortlist(x, self.dim(), short);
        short
            .iter()
            .map(|&p| (p as usize, sqdist(x, self.centroid(p as usize))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(p, _)| p)
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
        let dim = self.quant.dim();
        self.centroids.extend_from_slice(centroid);
        self.postings.push(Posting::default());
        let p = self.postings.len() - 1;
        self.coarse.added(p, centroid, dim);
        self.note(p);
        self.refresh_coarse();
        p
    }

    /// Rebuild the coarse layer if the partition count has moved far enough
    /// since the anchors were last chosen.
    fn refresh_coarse(&mut self) {
        let n = self.postings.len();
        if self.coarse.stale(n) {
            let dim = self.quant.dim();
            self.coarse.rebuild(&self.centroids, dim, n);
        }
    }

    /// Drop an empty partition, moving the last one into its place.
    fn drop_partition(&mut self, p: usize) {
        debug_assert_eq!(self.postings[p].len(), 0, "a partition is emptied first");
        let dim = self.dim();
        let last = self.postings.len() - 1;
        self.coarse.dropped(p);
        self.postings.swap_remove(p);
        for i in 0..dim {
            self.centroids[p * dim + i] = self.centroids[last * dim + i];
        }
        self.centroids.truncate(last * dim);
        if p != last {
            // The partition that used to be last is at `p` now, so everything
            // filed under it has to be told. It is the copy in `last` that
            // moves, not the member, so a replicated id keeps its other copies
            // pointing where they already point.
            for i in 0..self.postings[p].len() {
                let id = self.postings[p].ids[i];
                if let Some(at) = self.placed_at(id, last) {
                    self.places[at as usize].partition = p as u32;
                }
            }
            self.note(p);
        }
        self.refresh_coarse();
    }

    /// Append a member to a partition. `x` is rotated.
    ///
    /// A partition that already holds a copy of `id` keeps the one it has, so
    /// that the two maintenance paths that can hand the same member to the same
    /// partition twice, a merge into a partition the member was replicated into
    /// and a sweep that moves it there, cannot produce a posting with the same
    /// id in it twice.
    fn place(&mut self, p: usize, id: u64, tag: u64, x: &[f32]) {
        let dim = self.dim();
        let width = self.quant.code_bytes();
        let slot = self.postings[p].len();
        if !self.attach(id, p, slot) {
            return;
        }
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
        self.note(p);
    }

    // -- the placement chain -------------------------------------------------
    //
    // Every site that used to write `self.at` goes through one of these, because
    // with replication the question is almost never about an id. It is about one
    // copy of an id, the one in a particular partition, and the difference only
    // shows up as a corrupt index a long way from where it was caused.

    /// Record that `id` has a copy at `(p, slot)`, saying whether it is new.
    ///
    /// A partition already holding a copy is left alone rather than given a
    /// second one. Nothing on the insert path asks for that, but the maintenance
    /// paths can: a member replicated into two partitions that are then merged
    /// into each other would otherwise arrive twice, and a duplicate inside one
    /// posting is the one shape the rest of this cannot cope with, because a
    /// delete would take out one copy and leave the other.
    fn attach(&mut self, id: u64, p: usize, slot: usize) -> bool {
        let head = self.at.get(&id).copied().unwrap_or(END);
        let mut walk = head;
        while walk != END {
            if self.places[walk as usize].partition as usize == p {
                return false;
            }
            walk = self.places[walk as usize].next;
        }
        let place = Place {
            partition: p as u32,
            slot: slot as u32,
            next: head,
        };
        let at = if self.free == END {
            self.places.push(place);
            (self.places.len() - 1) as u32
        } else {
            let at = self.free;
            self.free = self.places[at as usize].next;
            self.places[at as usize] = place;
            at
        };
        self.at.insert(id, at);
        true
    }

    /// Forget the copy of `id` in partition `p`, saying whether there was one.
    fn detach(&mut self, id: u64, p: usize) -> bool {
        let Some(&head) = self.at.get(&id) else {
            return false;
        };
        let mut prev = END;
        let mut walk = head;
        while walk != END {
            let this = self.places[walk as usize];
            if this.partition as usize == p {
                if prev == END {
                    if this.next == END {
                        self.at.remove(&id);
                    } else {
                        self.at.insert(id, this.next);
                    }
                } else {
                    self.places[prev as usize].next = this.next;
                }
                self.places[walk as usize].next = self.free;
                self.free = walk;
                return true;
            }
            prev = walk;
            walk = this.next;
        }
        false
    }

    /// Forget every copy of `id`, saying whether there were any.
    fn detach_all(&mut self, id: u64) -> bool {
        let Some(head) = self.at.remove(&id) else {
            return false;
        };
        let mut walk = head;
        while walk != END {
            let next = self.places[walk as usize].next;
            self.places[walk as usize].next = self.free;
            self.free = walk;
            walk = next;
        }
        true
    }

    /// Where the copy of `id` in partition `p` is, if there is one.
    fn placed_at(&self, id: u64, p: usize) -> Option<u32> {
        let mut walk = self.at.get(&id).copied().unwrap_or(END);
        while walk != END {
            if self.places[walk as usize].partition as usize == p {
                return Some(walk);
            }
            walk = self.places[walk as usize].next;
        }
        None
    }

    /// Say that the copy of `id` in partition `p` is at slot `s` now, which is
    /// what a pull leaves behind when it moves the last member into a hole.
    fn reslot(&mut self, id: u64, p: usize, s: usize) {
        if let Some(at) = self.placed_at(id, p) {
            self.places[at as usize].slot = s as u32;
        } else {
            debug_assert!(
                false,
                "id {id} is in partition {p} and the map does not say so"
            );
        }
    }

    /// How many partitions hold a copy of `id`.
    #[cfg(test)]
    fn placements_of(&self, id: u64) -> usize {
        let mut walk = self.at.get(&id).copied().unwrap_or(END);
        let mut n = 0;
        while walk != END {
            n += 1;
            walk = self.places[walk as usize].next;
        }
        n
    }

    /// Any one copy of `id`, for the questions that do not care which.
    fn any_place(&self, id: u64) -> Option<Place> {
        self.at.get(&id).map(|&at| self.places[at as usize])
    }

    /// Every copy of `id`, collected because the callers that want them all are
    /// about to borrow the index mutably.
    fn every_place(&self, id: u64, into: &mut Vec<Place>) {
        into.clear();
        let mut walk = self.at.get(&id).copied().unwrap_or(END);
        while walk != END {
            let place = self.places[walk as usize];
            into.push(place);
            walk = place.next;
        }
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
        let moved = (s != last).then(|| posting.ids[s]);
        self.note(p);
        moved
    }

    /// The same, keeping the map straight, for the paths that are about to put
    /// the member somewhere else.
    fn pull_and_forget(&mut self, p: usize, s: usize) {
        let id = self.postings[p].ids[s];
        self.detach(id, p);
        if let Some(moved) = self.pull(p, s) {
            self.reslot(moved, p, s);
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

#[derive(Debug, PartialEq, Eq)]
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

/// One candidate, ordered by its estimated distance.
///
/// The tie break on the id is not decoration. Two members of the same partition
/// can get the same estimate out of codes that are 16 bytes wide, and without a
/// tie break which of them survives depends on the order the heap happened to
/// be in, which makes a search answer depend on the insertion history of the
/// collection rather than on the collection.
#[derive(PartialEq)]
struct Ranked {
    at: f32,
    id: u64,
}

impl Eq for Ranked {}

impl Ord for Ranked {
    fn cmp(&self, other: &Ranked) -> std::cmp::Ordering {
        self.at.total_cmp(&other.at).then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Ranked) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The best `want` candidates seen so far, and nothing else.
///
/// The scan used to push every member of every partition it read into one
/// vector and then select from it, which at probe 64 is 24 thousand entries
/// pushed and 24 thousand selected over to keep 160. That is 384 kilobytes of
/// writes per search and it was a tenth of the search's time.
///
/// A bounded heap makes the common case one comparison. Once `want` candidates
/// are in, a member is only touched further if it beats the worst of them,
/// which after the first partition or two is a small fraction of them, and a
/// member that loses never has its id or its tag read at all.
struct Bounded {
    want: usize,
    heap: std::collections::BinaryHeap<Ranked>,
}

impl Bounded {
    fn new(want: usize) -> Bounded {
        Bounded {
            want,
            heap: std::collections::BinaryHeap::with_capacity(want + 1),
        }
    }

    /// Whether there are already `want` answers, which is what says a search
    /// that was widening for a filter can stop widening.
    fn full(&self) -> bool {
        self.heap.len() >= self.want
    }

    /// Whether `at` could still be one of the answers.
    #[inline]
    fn wants(&self, at: f32) -> bool {
        match self.heap.peek() {
            Some(worst) if self.heap.len() >= self.want => at < worst.at,
            _ => true,
        }
    }

    fn put(&mut self, id: u64, at: f32) {
        if self.heap.len() >= self.want {
            self.heap.pop();
        }
        self.heap.push(Ranked { at, id });
    }

    /// The answers, nearest first.
    fn sorted(self) -> Vec<(u64, f32)> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|r| (r.id, r.at))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// What `job` used to be: two passes over every partition. The candidate
    /// lists are only worth having if they give the same answer, so the slow
    /// version stays here as the thing the fast one is checked against.
    fn slow_job(ix: &Partitions) -> Option<Job> {
        let big = (0..ix.postings.len())
            .filter(|&p| ix.postings[p].len() > ix.postings[p].stuck)
            .max_by_key(|&p| ix.postings[p].len());
        if let Some(big) = big
            && ix.postings[big].len() > ix.tuning.posting * 2
        {
            return Some(Job::Split(big));
        }
        if ix.postings.len() > 1 {
            let small = (0..ix.postings.len()).min_by_key(|&p| ix.postings[p].len())?;
            if ix.postings[small].len() * 4 < ix.tuning.posting {
                return Some(Job::Merge(small));
            }
        }
        None
    }

    /// Every partition that qualifies has to be on a list, or maintenance stops
    /// happening and the index quietly rots. Deletes are in here on purpose,
    /// because a delete is the only thing that pushes a partition down through
    /// the merge threshold and it is also what makes a partition disappear and
    /// renumber the one that was last.
    #[test]
    fn the_candidate_lists_answer_what_the_two_passes_answered() {
        let store = corpus(16, 3000, 12, 0x105E);
        let mut ix = Partitions::new(16, Bits::One, 7, Tuning::default());
        let mut rng = Rng::new(0x105F);
        let mut live: Vec<u64> = Vec::new();
        for id in 0..3000u64 {
            ix.insert(id, &store.0[id as usize]);
            live.push(id);
            if id % 7 == 3 && !live.is_empty() {
                let at = rng.below(live.len());
                let gone = live.swap_remove(at);
                ix.remove(gone);
            }
            // Once before maintenance runs and once after, because the lists
            // are written by both and the state in between is the one a stale
            // entry would survive in.
            assert_eq!(
                ix.job(),
                slow_job(&ix),
                "after {id} inserts, before maintaining"
            );
            ix.maintain(&store, 8);
            assert_eq!(
                ix.job(),
                slow_job(&ix),
                "after {id} inserts, after maintaining"
            );
            assert_eq!(ix.needs_maintenance(), slow_job(&ix).is_some());
        }
        assert!(ix.postings.len() > 5, "the test never split anything");
    }

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
            let mut here = HashSet::new();
            for (s, id) in posting.ids.iter().enumerate() {
                assert!(here.insert(*id), "id {id} is twice in partition {p}");
                let at = ix
                    .placed_at(*id, p)
                    .expect("every member is in the map, under the partition holding it");
                assert_eq!(ix.places[at as usize].slot as usize, s, "id {id}");
                seen += 1;
            }
        }
        // Every placement points at a member, as many placements as there are
        // members, and the free list accounts for the rest of the arena. A
        // replicated id makes the first of those the interesting one: a chain
        // that kept an entry for a copy that was pulled would still look right
        // from the posting's side, and the count is what catches it.
        let mut held = 0usize;
        for (&id, &head) in &ix.at {
            let mut walk = head;
            let mut mine = HashSet::new();
            while walk != END {
                let place = ix.places[walk as usize];
                let p = place.partition as usize;
                assert!(mine.insert(p), "id {id} is filed twice under partition {p}");
                assert!(
                    p < ix.postings.len(),
                    "id {id} is filed under partition {p}"
                );
                assert_eq!(
                    ix.postings[p].ids[place.slot as usize], id,
                    "id {id} is filed at a slot holding something else"
                );
                held += 1;
                walk = place.next;
            }
        }
        assert_eq!(seen, held, "the map and the postings disagree on the count");
        let mut spare = 0usize;
        let mut walk = ix.free;
        while walk != END {
            spare += 1;
            assert!(spare <= ix.places.len(), "the free list has a cycle in it");
            walk = ix.places[walk as usize].next;
        }
        assert_eq!(held + spare, ix.places.len(), "the arena has leaked");
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

    /// The second test decides, and the scan keeps widening until it has `k` of
    /// what the second test wants rather than `k` of what the tag wants.
    ///
    /// This is what a filter whose tag is only a summary looks like: the tag
    /// lets a superset through, so an answer that passes it and fails the exact
    /// test must not have cost an answer that passes both.
    #[test]
    fn the_exact_test_decides_and_the_scan_widens_for_it() {
        struct Summary;

        impl Filter for Summary {
            fn allows(&self, tag: u64) -> bool {
                // One in ten, which is what a bit that several values landed on
                // looks like from inside the scan.
                tag == 1
            }

            fn exact(&self, id: u64) -> bool {
                // One in fifty, and a subset of what the tag allowed, which is
                // the direction a summary is allowed to be wrong in.
                id.is_multiple_of(50)
            }
        }

        let dim = 64;
        let store = corpus(dim, 3000, 9, 71);
        let tuning = Tuning {
            posting: 64,
            ..Tuning::default()
        };
        let ix = build_tagged(&store, dim, tuning, |id| u64::from(id.is_multiple_of(10)));

        let k = 10;
        let mut found = 0usize;
        for i in 0..20 {
            let q = &store.0[i * 131 % store.0.len()];
            let mut all: Vec<(u64, f32)> = store
                .0
                .iter()
                .enumerate()
                .map(|(id, v)| (id as u64, sqdist(q, v)))
                .filter(|(id, _)| id.is_multiple_of(50))
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: Vec<u64> = all[..k].iter().map(|(id, _)| *id).collect();

            let got: Vec<u64> = ix
                .search_where(q, k, &Summary, &store)
                .into_iter()
                .map(|h| h.id)
                .collect();
            assert!(
                got.iter().all(|id| id.is_multiple_of(50)),
                "the exact test did not decide: {got:?}"
            );
            found += want.iter().filter(|id| got.contains(id)).count();
        }
        let recall = found as f32 / (20.0 * k as f32);
        assert!(recall >= 0.9, "two stage filtering gave {recall}");
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

    /// How many copies of its members a collection is holding, which is what
    /// replication costs and what it has to be paid for in recall.
    fn copies(ix: &Partitions) -> f32 {
        let held: usize = ix.postings.iter().map(Posting::len).sum();
        held as f32 / ix.len() as f32
    }

    /// The knob does what it says: off means one copy of everything, and on
    /// means more than one copy of some things and not of everything.
    #[test]
    fn spilling_puts_boundary_vectors_in_more_than_one_partition() {
        let dim = 32;
        let store = corpus(dim, 3000, 12, 5);
        let off = Tuning {
            spill: 1,
            ..Tuning::default()
        };
        let none = build(&store, dim, off);
        consistent(&none);
        assert_eq!(copies(&none), 1.0, "spill of one is one copy of everything");

        let on = build(&store, dim, Tuning::default());
        consistent(&on);
        let rate = copies(&on);
        assert!(rate > 1.0, "spilling should make copies, made {rate}");
        assert!(
            rate < Tuning::default().spill as f32,
            "slack should stop short of copying everything into everything, made {rate}"
        );
        assert_eq!(on.len(), store.0.len(), "a copy is not a member");
    }

    /// The whole point of it, stated as the thing that is actually true rather
    /// than as a recall number.
    ///
    /// A copy of a member in a second partition means a search that reads that
    /// partition finds the member, without widening and without the member's own
    /// partition being anywhere near the query. So take a member that got
    /// copied, take a different member of the partition it was copied into, and
    /// search from that one with a probe of exactly one. The scan reads one
    /// posting, and the copy is why the answer is in it.
    ///
    /// Recall is deliberately not what this asserts. Whether copies pay for
    /// themselves end to end is a question about the shape of the data, and on
    /// generated vectors the answer is no by a hair, because a tight cluster has
    /// no boundary members worth copying and the copies that do get made push
    /// the partition count up and the share of the index a fixed probe reads
    /// down. `examples/recall.rs` is where that gets answered, on data somebody
    /// else made.
    #[test]
    fn a_copy_is_found_from_the_partition_it_was_copied_into() {
        let dim = 32;
        let store = corpus(dim, 3000, 12, 5);
        let t = Tuning {
            slack: 0.25,
            ..Tuning::default()
        };
        let mut ix = build(&store, dim, t);
        let (id, copies) = (0..3000u64)
            .filter_map(|id| {
                let mut places = Vec::new();
                ix.every_place(id, &mut places);
                (places.len() > 1).then_some((id, places))
            })
            .next()
            .expect("some member near a boundary got copied");

        let mut narrow = t;
        narrow.probe = 1;
        narrow.widen = 1;
        ix.retune(narrow);
        for place in &copies {
            let p = place.partition as usize;
            // A different member of the same posting, so the query lands there
            // rather than where the copied member belongs.
            let neighbour = ix.postings[p]
                .ids
                .iter()
                .copied()
                .find(|&other| other != id)
                .expect("the partition holds more than the copy");
            let got = ix.candidates(&store.0[neighbour as usize], ix.postings[p].len());
            assert!(
                got.iter().any(|&(seen, _)| seen == id),
                "member {id} has a copy in partition {p} and a search of it did not find it"
            );
        }
    }

    /// The knob is only worth having if the searches it cuts short were reading
    /// partitions that had stopped paying, so the two things to show are that it
    /// reads fewer of them and that the answers survive it.
    #[test]
    fn patience_reads_fewer_partitions_and_keeps_the_answers() {
        let dim = 32;
        let store = corpus(dim, 4000, 16, 77);
        let wide = Tuning {
            probe: 64,
            ..Tuning::default()
        };
        let mut ix = build(&store, dim, wide);
        let queries = 100;
        let full = recall(&ix, &store, 10, queries);
        let cost = |ix: &Partitions| -> f64 {
            (0..queries)
                .map(|i| {
                    let q = &store.0[i * 7 % store.0.len()];
                    ix.search_costed(q, 10, &Any, &store).1.probed
                })
                .sum::<usize>() as f64
                / queries as f64
        };
        let spent = cost(&ix);

        ix.retune(Tuning {
            probe: 64,
            patience: 2,
            ..Tuning::default()
        });
        let cut = cost(&ix);
        assert!(
            cut < spent * 0.75,
            "patience of two read {cut:.1} partitions a query against {spent:.1}, which is not a saving worth the knob"
        );
        let after = recall(&ix, &store, 10, queries);
        assert!(
            after >= full - 0.02,
            "recall went from {full} to {after}, which is more than giving up early is allowed to cost"
        );
    }

    /// The rule is written as "once there is enough to answer with", and the
    /// case that proves it is the one where there is not. A filter that almost
    /// nothing passes is why `widen` exists, and a search that gave up on it
    /// after two quiet partitions would return nothing at all.
    #[test]
    fn patience_does_not_cut_off_a_filter_that_is_still_short() {
        let dim = 32;
        let store = corpus(dim, 4000, 16, 91);
        let mut ix = build(
            &store,
            dim,
            Tuning {
                patience: 1,
                ..Tuning::default()
            },
        );
        // One member in fifty, spread over every partition, so the answers are
        // certainly not all in the first few.
        for id in 0..4000u64 {
            ix.retag(id, u64::from(id % 50 == 0));
        }
        struct Rare;
        impl Filter for Rare {
            fn allows(&self, tag: u64) -> bool {
                tag == 1
            }
        }
        let q = &store.0[3];
        let got = ix.search_where(q, 10, &Rare, &store);
        assert_eq!(got.len(), 10, "the filtered search came back short");
        for hit in &got {
            assert!(hit.id.is_multiple_of(50), "{} is not a match", hit.id);
        }
    }

    /// A replicated member is scanned twice by a search that reads both of its
    /// partitions, and an answer list with the same id in it twice is a bug the
    /// caller sees.
    #[test]
    fn a_replicated_member_comes_back_once() {
        let dim = 32;
        let store = corpus(dim, 2000, 8, 31);
        // Every partition, so that every copy of every member is read and the
        // duplicates are certain rather than likely.
        let t = Tuning {
            probe: 1 << 20,
            ..Tuning::default()
        };
        let ix = build(&store, dim, t);
        for i in 0..50 {
            let q = &store.0[i * 37 % store.0.len()];
            let got: Vec<u64> = ix.search(q, 20, &store).into_iter().map(|h| h.id).collect();
            let mut once = got.clone();
            once.sort_unstable();
            once.dedup();
            assert_eq!(got.len(), once.len(), "a duplicate answer for query {i}");
        }
    }

    /// Every copy has to go, and the arena has to come back. Removing under
    /// replication is the path where a leak or a stale placement would show up,
    /// and `consistent` is what says it did not.
    #[test]
    fn removing_a_replicated_member_takes_every_copy() {
        let dim = 32;
        let store = corpus(dim, 1500, 6, 41);
        let mut ix = build(&store, dim, Tuning::default());
        let before: usize = ix.postings.iter().map(Posting::len).sum();
        let mut gone = 0usize;
        for id in (0..1500u64).step_by(3) {
            gone += ix.placements_of(id);
            assert!(ix.remove(id));
            assert!(!ix.contains(id));
        }
        consistent(&ix);
        let after: usize = ix.postings.iter().map(Posting::len).sum();
        assert_eq!(before - after, gone, "a copy was left behind");
        assert_eq!(ix.len(), 1000);
        for id in (0..1500u64).step_by(3) {
            let q = &store.0[id as usize];
            assert!(ix.search(q, 5, &store).iter().all(|h| h.id != id));
        }
    }

    /// A retag has to reach every copy, because a scan meets whichever one it
    /// reads first and a filter that sees a stale tag in one partition and a
    /// fresh one in another is the worst kind of wrong.
    #[test]
    fn retagging_a_replicated_member_reaches_every_copy() {
        let dim = 32;
        let store = corpus(dim, 1200, 6, 47);
        let mut ix = Partitions::new(dim, Bits::One, 7, Tuning::default());
        for (i, v) in store.0.iter().enumerate() {
            ix.insert_tagged(i as u64, v, 1);
            if i % 64 == 0 {
                ix.maintain(&store, 4096);
            }
        }
        ix.maintain(&store, 1 << 20);
        let spread = (0..1200u64).find(|&id| ix.placements_of(id) > 1);
        let id = spread.expect("some member is in more than one partition");
        assert!(ix.retag(id, 9));
        let mut copies = Vec::new();
        ix.every_place(id, &mut copies);
        for place in &copies {
            assert_eq!(
                ix.postings[place.partition as usize].tags[place.slot as usize], 9,
                "a copy kept the old tag"
            );
        }
        consistent(&ix);
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
    /// Asked once per member rather than once per posting entry, because a
    /// boundary copy sits in a partition that is not the member's nearest on
    /// purpose, and counting one as drift would read replication as the very
    /// thing the sweep exists to undo. A member has drifted when none of the
    /// partitions holding it is its nearest.
    fn misfiled(ix: &Partitions, store: &Store) -> usize {
        let mut buf = vec![0.0f32; ix.dim()];
        let mut wrong = 0;
        for id in 0..store.0.len() as u64 {
            if !ix.contains(id) {
                continue;
            }
            assert!(store.get(id, &mut buf));
            let near = ix.nearest(&ix.quant.rotate(&buf));
            if ix.placed_at(id, near).is_none() {
                wrong += 1;
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
        assert!((one.scale - two.scale).abs() < 1e-4);
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
