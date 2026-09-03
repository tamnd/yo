//! Writing a collection down, and reading it back without rebuilding it.
//!
//! `yo_format::image` is the layout and this is the part that fills it in. The
//! split is the same one the rest of the build keeps: the format crate knows
//! where a byte goes and nothing else, and the crate that owns the structure
//! knows what the byte means.
//!
//! # Why an index is written down at all
//!
//! The records give back the vectors. They do not give back the shape. A
//! collection of a million vectors is a few thousand partitions that arrived at
//! their centroids through a long sequence of splits, merges and sweeps, and
//! rebuilding that on open is a million quantisations and a lot of two means:
//! minutes, on a machine that is meant to be answering. Every other index in
//! this family has the same problem and most of them solve it by not solving it,
//! which is what "the index warms up" means when a vendor says it.
//!
//! So a checkpoint writes the index down and an open reads it back. Nothing is
//! requantised on the way in, because requantising is the rebuild.
//!
//! # The two halves of a load
//!
//! An image holds the codes and does not hold the vectors, and that is on
//! purpose: the vectors are records of kind 3 already, at addresses the log
//! resolves, and G8's budget is 96 bytes of index for a 768 dimensional vector
//! with the raw copy in the log. An image that carried them too would write
//! every vector twice to save a walk.
//!
//! So the caller brings them. [`Stored`] is that: something that can produce the
//! vector a key was stored under, which for the engine is the log and for a test
//! is a map. A key the store cannot produce is dropped rather than refused,
//! which is the same answer [`Vectors`](crate::Vectors) gives for an id the log
//! forgot: an index that heals is worth more than an index that is right about
//! being unable to open.
//!
//! What comes back out of the store has to be what went in, which for a cosine
//! collection is the unit vector rather than whatever the client sent, because
//! that is what the collection stored and what its codes were measured against.
//! [`Collection::get`] returns the same thing, so a store built out of one
//! collection reloads another exactly.
//!
//! # The order things are written in
//!
//! Sections first, root last. A chain writes its chunks before its directory, so
//! a directory that is readable has readable chunks, and this is the same rule
//! one level up: a root that is readable points at sections that are all there.
//! A crash between the two leaves chunks nobody points at, which is what
//! compaction is for.

use yo_common::{Addr, Code, Error, Result};
use yo_format::image::{
    Chain, ImageHeader, Keys, PostingHeader, get_floats, image_kind, image_len, key_entry_len,
    metric, posting_len, put_floats, put_key, put_partition,
};
use yo_format::{get_f32, get_u64, put_u64};
use yo_kv::cold::{self, Blocks, Scratch};
use yo_shape::Metric;

use crate::collection::{Collection, check_metric};
use crate::partition::{Partitions, Tuning};
use crate::rabitq::{Bits, Coded};

/// Where the full precision vectors come back from when an image is loaded.
///
/// The engine answers this out of the record log. A test answers it out of a
/// map. Either way it is asked once per key in the image and never again, and
/// what it gives back has to be the stored form: see the note at the top of this
/// module about cosine.
pub trait Stored {
    /// Write the vector stored under `key` into `into` and say so, or say that
    /// the key is gone.
    fn get(&self, key: &[u8], into: &mut [f32]) -> bool;
}

/// What came back from an image.
#[derive(Debug)]
pub struct Restored {
    /// The collection.
    pub collection: Collection,
    /// How many keys the image named that the store could not produce.
    ///
    /// Zero on any pair of an image and a log that were written together. A
    /// number here is a log that was compacted or truncated past the checkpoint
    /// the image belongs to, and it is worth reporting rather than swallowing,
    /// because it is the difference between a collection that is smaller than it
    /// was and a collection that is smaller than it should be.
    pub missing: usize,
}

impl Collection {
    /// Write the collection down and say where the root went.
    ///
    /// The root's address and length are what a checkpoint entry records, so
    /// this returns the pair rather than putting it anywhere: which checkpoint
    /// this belongs to is the shard's business.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] if a section is longer than a chain holds, which for the
    /// centroids means a collection with more partitions than 512 MiB of them,
    /// and whatever the store returns while it is being written to.
    pub fn save<B: Blocks>(&self, blocks: &mut B, scratch: &mut Scratch) -> Result<Chain> {
        let index = self.index();
        let dim = index.dim();
        let width = index.quantizer().code_bytes();
        let count = index.partitions();

        let mut buf = Vec::new();
        let mut root = vec![0u8; image_len(as_u32(count)?)?];

        // Every partition first, each its own chain, because a partition is the
        // unit that can be brought back on its own and `10` section 2 says so.
        for p in 0..count {
            let (ids, tags, codes, meta, stuck) = index.posting_parts(p);
            let head = PostingHeader {
                count: as_u32(ids.len())?,
                code_bytes: as_u32(width)?,
                stuck: as_u32(stuck)?,
            };
            buf.clear();
            buf.resize(posting_len(head.count, head.code_bytes)?, 0);
            head.encode(&mut buf)?;
            for (i, &id) in ids.iter().enumerate() {
                put_u64(&mut buf, head.ids_at() + i * 8, id);
            }
            for (i, &tag) in tags.iter().enumerate() {
                put_u64(&mut buf, head.tags_at() + i * 8, tag);
            }
            let at = head.codes_at();
            buf[at..at + codes.len()].copy_from_slice(codes);
            for (i, m) in meta.iter().enumerate() {
                let at = head.meta_at() + i * 16;
                put_floats(&mut buf[at..], &[m.norm, m.scale, m.lo, m.delta])?;
            }
            put_partition(&mut root, as_u32(p)?, write(blocks, &buf, scratch)?)?;
        }

        // The centroids, one run of floats, and the key table, one run of
        // entries. Both are read whole at open, so neither is cut up further.
        buf.clear();
        buf.resize(index.all_centroids().len() * 4, 0);
        put_floats(&mut buf, index.all_centroids())?;
        let centroids = write(blocks, &buf, scratch)?;

        buf.clear();
        for (key, &id) in self.id_table().iter() {
            let at = buf.len();
            buf.resize(at + key_entry_len(key.len())?, 0);
            put_key(&mut buf[at..], id, key)?;
        }
        let keys = write(blocks, &buf, scratch)?;

        let tuning = index.tuning();
        let head = ImageHeader {
            kind: image_kind::VECTOR,
            bits: as_u32(index.quantizer().bits().count())? as u8,
            metric: metric_byte(self.metric()),
            dim: as_u32(dim)?,
            partitions: as_u32(count)?,
            seed: index.quantizer().seed(),
            members: self.len() as u64,
            slots: as_u32(self.slots())?,
            posting: as_u32(tuning.posting)?,
            probe: as_u32(tuning.probe)?,
            rerank: as_u32(tuning.rerank)?,
            sweep: as_u32(tuning.sweep)?,
            widen: as_u32(tuning.widen)?,
            spill: as_u32(tuning.spill)?,
            slack: tuning.slack,
            centroids,
            keys,
        };
        head.encode(&mut root)?;
        write(blocks, &root, scratch)
    }

    /// Read a collection back out of an image, taking the vectors from `stored`.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] for an image that does not describe a collection this
    /// build can hold: a kind or a metric it does not know, sections that
    /// disagree with the header that named them, an id in two partitions, or an
    /// id in a partition that the key table does not have.
    pub fn load<B: Blocks>(blocks: &mut B, at: Chain, stored: &impl Stored) -> Result<Restored> {
        let mut buf = Vec::new();
        read(blocks, at, &mut buf)?;
        let head = ImageHeader::decode(&buf)?;
        if head.kind != image_kind::VECTOR {
            return Err(
                Error::new(Code::Corrupt, "that image is not a vector index")
                    .with_detail(format!("kind={}", head.kind)),
            );
        }
        let bits = match head.bits {
            1 => Bits::One,
            _ => Bits::Four,
        };
        let metric = metric_of(head.metric)?;
        check_metric(metric)?;
        let dim = head.dim as usize;
        let root = std::mem::take(&mut buf);

        let mut index = Partitions::new(
            dim,
            bits,
            head.seed,
            Tuning {
                posting: head.posting as usize,
                probe: head.probe as usize,
                rerank: head.rerank as usize,
                sweep: head.sweep as usize,
                widen: head.widen as usize,
                spill: head.spill as usize,
                slack: head.slack,
            },
        );
        let width = index.quantizer().code_bytes();

        blocks.release();
        read(blocks, head.centroids, &mut buf)?;
        let mut centroids = vec![0f32; head.partitions as usize * dim];
        get_floats(&buf, &mut centroids)?;

        let mut members = Vec::new();
        for p in 0..head.partitions {
            blocks.release();
            read(blocks, yo_format::image::get_partition(&root, p)?, &mut buf)?;
            let post = PostingHeader::decode(&buf)?;
            if post.code_bytes as usize != width {
                return Err(
                    Error::new(Code::Corrupt, "a partition's codes are the wrong width")
                        .with_detail(format!("code_bytes={} want={width}", post.code_bytes)),
                );
            }
            let n = post.count as usize;
            let mut ids = Vec::with_capacity(n);
            let mut tags = Vec::with_capacity(n);
            let mut meta = Vec::with_capacity(n);
            for i in 0..n {
                let id = get_u64(&buf, post.ids_at() + i * 8);
                if id >= u64::from(head.slots) {
                    return Err(Error::new(Code::Corrupt, "a member's id is past the table")
                        .with_detail(format!("id={id} slots={}", head.slots)));
                }
                ids.push(id);
                tags.push(get_u64(&buf, post.tags_at() + i * 8));
                let at = post.meta_at() + i * 16;
                meta.push(Coded {
                    norm: get_f32(&buf, at),
                    scale: get_f32(&buf, at + 4),
                    lo: get_f32(&buf, at + 8),
                    delta: get_f32(&buf, at + 12),
                });
            }
            let codes = buf[post.codes_at()..post.meta_at()].to_vec();
            let at = p as usize * dim;
            index.absorb(
                &centroids[at..at + dim],
                ids,
                tags,
                codes,
                meta,
                post.stuck as usize,
            )?;
            members.push(n);
        }
        index.finish_image();

        blocks.release();
        read(blocks, head.keys, &mut buf)?;
        let mut collection = Collection::from_image(index, metric, head.slots as usize);
        let mut vector = vec![0f32; dim];
        let mut missing = 0;
        let mut named = 0u64;
        let mut walk = Keys::new(&buf);
        for (id, key) in walk.by_ref() {
            named += 1;
            if !collection.holds(id) {
                return Err(Error::new(
                    Code::Corrupt,
                    "the key table names an id no partition has",
                )
                .with_detail(format!("id={id}")));
            }
            if stored.get(key, &mut vector) {
                collection.restore(key, id, &vector)?;
            } else {
                collection.forget(id);
                missing += 1;
            }
        }
        if !walk.done() {
            return Err(Error::new(Code::Corrupt, "the key table ends mid entry"));
        }
        if named != head.members {
            return Err(Error::new(
                Code::Corrupt,
                "the key table is not the length the header says",
            )
            .with_detail(format!("keys={named} members={}", head.members)));
        }
        collection.seal();
        Ok(Restored {
            collection,
            missing,
        })
    }
}

/// One section, through the chunk chain, in the format's terms.
fn write<B: Blocks>(blocks: &mut B, bytes: &[u8], scratch: &mut Scratch) -> Result<Chain> {
    let chain = cold::write(blocks, bytes, scratch)?;
    Ok(Chain {
        at: chain.at.to_bits(),
        len: chain.len,
    })
}

/// The same section back, whole.
fn read<B: Blocks>(blocks: &B, at: Chain, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    let reader = cold::Reader::open(
        blocks,
        cold::Chain {
            at: Addr::from_bits(at.at),
            len: at.len,
        },
    )?;
    for piece in reader.range(0, at.len) {
        out.extend_from_slice(piece?);
    }
    Ok(())
}

/// The byte an image writes a metric as.
fn metric_byte(m: Metric) -> u8 {
    match m {
        Metric::L2 => metric::L2,
        Metric::Cosine => metric::COSINE,
        Metric::Ip => metric::IP,
        Metric::Hamming => metric::HAMMING,
    }
}

/// And back, for the four this version has a name for.
fn metric_of(b: u8) -> Result<Metric> {
    match b {
        metric::L2 => Ok(Metric::L2),
        metric::COSINE => Ok(Metric::Cosine),
        metric::IP => Ok(Metric::Ip),
        metric::HAMMING => Ok(Metric::Hamming),
        _ => Err(Error::new(Code::Corrupt, "unknown metric in an image")
            .with_detail(format!("metric={b}"))),
    }
}

/// A count the format writes as a `u32`, refused rather than truncated.
fn as_u32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| {
        Error::new(Code::Full, "that count does not fit in an image").with_detail(format!("n={n}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A store that keeps blobs in memory and hands back the index as the
    /// address, which is enough to exercise every path here without a file.
    struct Mem {
        blobs: Vec<Vec<u8>>,
    }

    impl Mem {
        fn new() -> Mem {
            Mem { blobs: Vec::new() }
        }
    }

    impl Blocks for Mem {
        fn put(&mut self, bytes: &[u8]) -> Result<Addr> {
            self.blobs.push(bytes.to_vec());
            Ok(Addr::new(
                yo_common::Space::Log,
                (self.blobs.len() - 1) as u64,
            ))
        }

        fn get(&self, at: Addr) -> Result<&[u8]> {
            self.blobs
                .get(at.offset() as usize)
                .map(Vec::as_slice)
                .ok_or_else(|| Error::new(Code::NotFound, "no such block"))
        }

        fn bytes(&self) -> u64 {
            self.blobs.iter().map(|b| b.len() as u64).sum()
        }
    }

    /// Everything the collection holds, keyed the way an image names it.
    struct Table(HashMap<Vec<u8>, Vec<f32>>);

    impl Table {
        fn of(c: &Collection) -> Table {
            let mut m = HashMap::new();
            for key in c.keys() {
                m.insert(
                    key.to_vec(),
                    c.get(key).expect("a key it just named").to_vec(),
                );
            }
            Table(m)
        }

        fn without(mut self, key: &[u8]) -> Table {
            self.0.remove(key);
            self
        }
    }

    impl Stored for Table {
        fn get(&self, key: &[u8], into: &mut [f32]) -> bool {
            let Some(v) = self.0.get(key) else {
                return false;
            };
            into.copy_from_slice(v);
            true
        }
    }

    /// A deterministic spread of vectors, so that a corpus is the same on every
    /// machine and a recall number means something when it is compared.
    fn corpus(dim: usize, n: usize, seed: u64) -> Vec<(Vec<u8>, Vec<f32>)> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        (0..n)
            .map(|i| {
                let key = format!("k{i}").into_bytes();
                let v: Vec<f32> = (0..dim).map(|_| next()).collect();
                (key, v)
            })
            .collect()
    }

    fn built(dim: usize, n: usize, metric: Metric) -> Collection {
        let mut c = Collection::new(dim, metric).expect("a collection");
        for (i, (key, v)) in corpus(dim, n, 42).into_iter().enumerate() {
            c.put_tagged(&key, &v, 1 << (i % 8)).expect("put");
        }
        c
    }

    fn round_trip(c: &Collection, stored: &impl Stored) -> Restored {
        let mut mem = Mem::new();
        let mut scratch = Scratch::new();
        let at = c.save(&mut mem, &mut scratch).expect("saved");
        Collection::load(&mut mem, at, stored).expect("loaded")
    }

    #[test]
    fn a_collection_comes_back_answering_the_same_questions() {
        let c = built(32, 900, Metric::L2);
        let back = round_trip(&c, &Table::of(&c)).collection;

        assert_eq!(back.len(), c.len());
        assert_eq!(back.dim(), c.dim());
        assert_eq!(back.metric(), c.metric());
        assert!(
            c.partitions() > 1,
            "a collection that never split proves nothing"
        );
        assert_eq!(
            back.partitions(),
            c.partitions(),
            "the shape is the thing an image exists to keep"
        );
        assert_eq!(back.tuning(), c.tuning());

        // Identical answers rather than close ones: the codes were not
        // recomputed, so the candidates are the same candidates and the rerank
        // measures the same vectors. Queries the collection has never seen, so
        // that this is a search and not a lookup.
        for (_, q) in corpus(32, 50, 7) {
            assert_eq!(
                back.search(&q, 10, None).expect("search"),
                c.search(&q, 10, None).expect("search"),
                "a query came back differently after a round trip"
            );
            assert_eq!(
                back.search_where(&q, 10, None, &crate::Signature::from_bits(1 << 3))
                    .expect("search"),
                c.search_where(&q, 10, None, &crate::Signature::from_bits(1 << 3))
                    .expect("search"),
                "a filtered query came back differently, so a tag moved"
            );
        }
    }

    #[test]
    fn every_vector_and_every_tag_survives() {
        let c = built(16, 400, Metric::Cosine);
        let back = round_trip(&c, &Table::of(&c)).collection;
        for key in c.keys() {
            assert_eq!(
                back.get(key).map(<[f32]>::to_vec),
                c.get(key).map(<[f32]>::to_vec),
                "the vector under a key changed, bit for bit"
            );
            assert_eq!(back.tag(key), c.tag(key), "a tag was lost");
        }
    }

    #[test]
    fn a_reloaded_collection_takes_writes_where_it_left_off() {
        let mut c = built(16, 300, Metric::L2);
        let mut back = round_trip(&c, &Table::of(&c)).collection;

        // The free list is the part of a load that is derived rather than
        // stored, and the way to find out it is wrong is to allocate from it.
        assert!(c.remove(b"k7"));
        assert!(back.remove(b"k7"));
        for (key, v) in corpus(16, 40, 99) {
            let key = [b"new-".as_slice(), &key].concat();
            c.put(&key, &v).expect("put");
            back.put(&key, &v).expect("put");
        }
        assert_eq!(back.len(), c.len());
        for key in c.keys() {
            assert!(back.contains(key), "a key written after a load is missing");
        }
        let q = c.get(b"k1").expect("a vector").to_vec();
        assert_eq!(
            back.search(&q, 5, None).expect("search"),
            c.search(&q, 5, None).expect("search")
        );
    }

    /// `stuck` is the one thing an image carries that could have been derived
    /// from the vectors and cannot be derived from the codes: it is the size at
    /// which a split was tried and there was no cut to make. A thousand copies
    /// of one vector is the case that produces it, and an image that dropped it
    /// would have the first write after every open try that split again.
    #[test]
    fn a_partition_that_gave_up_splitting_does_not_try_again_after_a_load() {
        let mut c = Collection::new(8, Metric::L2).expect("a collection");
        for i in 0..1000 {
            c.put(format!("same{i}").as_bytes(), &[0.5; 8])
                .expect("put");
        }
        assert_eq!(c.maintain(1 << 20), 0, "it has already given up");

        let mut back = round_trip(&c, &Table::of(&c)).collection;
        assert_eq!(
            back.maintain(1 << 20),
            0,
            "the load forgot that the split was hopeless and went looking again"
        );
        assert_eq!(back.partitions(), c.partitions());
    }

    #[test]
    fn an_empty_collection_is_an_image_too() {
        let c = Collection::new(8, Metric::L2).expect("a collection");
        let back = round_trip(&c, &Table::of(&c)).collection;
        assert!(back.is_empty());
        assert_eq!(back.partitions(), 0);
        assert!(back.search(&[0.0; 8], 4, None).expect("search").is_empty());
    }

    /// A section longer than a chunk is a directory and a run of chunks rather
    /// than one record, and the key table is the section that gets there first:
    /// five thousand short keys is already past 64 KiB. Nothing above this
    /// module knows the difference, which is the thing being checked.
    #[test]
    fn a_section_longer_than_a_chunk_is_still_one_section() {
        let mut c = Collection::new(8, Metric::L2).expect("a collection");
        for (i, (_, v)) in corpus(8, 5000, 11).into_iter().enumerate() {
            c.put(format!("key{i}").as_bytes(), &v).expect("put");
        }

        let mut mem = Mem::new();
        let mut scratch = Scratch::new();
        let at = c.save(&mut mem, &mut scratch).expect("saved");
        assert!(
            mem.blobs.len() > c.partitions() + 3,
            "no section was cut up, so this proves nothing about chains"
        );

        let back = Collection::load(&mut mem, at, &Table::of(&c))
            .expect("loaded")
            .collection;
        assert_eq!(back.len(), c.len());
        for key in c.keys() {
            assert!(
                back.contains(key),
                "a key on the far side of a chunk is gone"
            );
        }
    }

    #[test]
    fn a_key_the_store_cannot_produce_is_dropped_and_counted() {
        let c = built(16, 200, Metric::L2);
        let restored = round_trip(&c, &Table::of(&c).without(b"k5"));
        assert_eq!(restored.missing, 1);
        let back = restored.collection;
        assert_eq!(back.len(), c.len() - 1);
        assert!(!back.contains(b"k5"));
        // And the collection is whole afterwards rather than merely smaller: the
        // slot is free, so the next write takes it.
        let mut back = back;
        back.put(b"k5", &[1.0; 16]).expect("put");
        assert_eq!(back.len(), c.len());
        assert_eq!(back.get(b"k5"), Some([1.0f32; 16].as_slice()));
    }

    #[test]
    fn an_image_that_says_something_impossible_is_refused() {
        let c = built(8, 120, Metric::L2);
        let mut mem = Mem::new();
        let mut scratch = Scratch::new();
        let at = c.save(&mut mem, &mut scratch).expect("saved");
        assert!(Collection::load(&mut mem, at, &Table::of(&c)).is_ok());

        // The root is the last thing written, so it is the last blob, and every
        // field in it is one a corrupt file could disagree about.
        let root = mem.blobs.len() - 1;
        for (at_byte, to) in [(4usize, 9u8), (5, 2), (6, 9), (7, 1)] {
            let mut broken = Mem {
                blobs: mem.blobs.clone(),
            };
            broken.blobs[root][at_byte] = to;
            assert!(
                Collection::load(&mut broken, at, &Table::of(&c)).is_err(),
                "byte {at_byte} of the root was believed"
            );
        }

        // A member count that does not match the key table, which is what a
        // half written image would look like if the root went down first.
        let mut lying = Mem {
            blobs: mem.blobs.clone(),
        };
        put_u64(&mut lying.blobs[root], 24, 3);
        assert!(Collection::load(&mut lying, at, &Table::of(&c)).is_err());
    }
}
