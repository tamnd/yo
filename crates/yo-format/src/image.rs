//! The index image: what a checkpoint points at so an index does not have to be
//! rebuilt from the records it was built from.
//!
//! [`CheckpointEntry::index_image_addr`](crate::superblock::CheckpointEntry)
//! has been in the superblock since the format was first written down and
//! nothing has pointed at anything yet. This is the first thing it points at,
//! and it is the vector index, because the vector index is the one that costs
//! real money to rebuild: a million vectors is a million quantisations and a few
//! thousand partitions that arrived at their shape through a long sequence of
//! splits, merges and sweeps. Replaying the records gives back the vectors. It
//! does not give back the shape, and rebuilding the shape on open is the outage
//! the whole update protocol exists to avoid.
//!
//! # It is chunks, not a record kind
//!
//! `06` fixes the record kinds and there is no kind for an index, which is
//! deliberate: an index is derived, so a reader that has never heard of it must
//! be able to walk straight past it. So an image is written as
//! [`CollectionChunk`](crate::RecordKind::CollectionChunk) records through the
//! chain in `yo-kv`, exactly the way a demoted collection is, and the only thing
//! that knows the chunks mean an index is the checkpoint entry that points at
//! them. A reader that ignores the checkpoint sees a run of chunks nobody claims
//! and compaction drops them.
//!
//! `10` section 2 is where the shape comes from: a partition is a natural chunk.
//! At the default posting size and 768 dimensions a partition is about 32 KiB,
//! which is half of one chunk, so a partition is one chunk and one read almost
//! always, and a partition that has grown past a chunk is a chain of its own
//! rather than a special case.
//!
//! ```text
//!   root                        one per partition
//! +------------------+        +---------------------------+
//! | header, 88 bytes |        | count | code_bytes | stuck |
//! +------------------+        +---------------------------+
//! | centroid chain   |        | ids      count * 8        |
//! | key chain        |        | tags     count * 8        |
//! +------------------+        | codes    count * width    |
//! | partition 0      | -----> | meta     count * 16       |
//! | partition 1      |        +---------------------------+
//! | ...              |
//! +------------------+
//! ```
//!
//! The four arrays inside a partition are separate runs rather than one run of
//! structures, and that is the same reason the posting itself is laid out that
//! way in memory: a scan that only wants the tags reads only the tags. A cold
//! partition can be brought in one array at a time for the same reason.
//!
//! # What is not in here
//!
//! The vectors. They are records of kind 3 already (`crate::vector`), at
//! addresses the log resolves, and G8's budget is 96 bytes of index for a 768
//! dimensional vector with the raw copy in the log. An image that carried them
//! as well would write every vector twice and spend the whole gate to save a
//! walk. So loading an image is two halves: the image gives the shape and the
//! codes, and the log gives the vectors back under the keys the image names.
//!
//! No checksum either, and that is not an omission. Every chunk of an image is a
//! record, every record carries a CRC32C over its own bytes, and the chain's
//! directory is a record too, so a second checksum inside the image would cover
//! bytes that are already covered. What a checksum cannot catch is an image that
//! is intact and stale, and that is what the checkpoint's log addresses are for.
//!
//! # The freeze
//!
//! This layout is frozen with the rest of the format at the end of M6. After
//! that the only lever is `min_reader_version` (`07` section 9), so the fields
//! that exist to be changed later exist now: [`ImageHeader::kind`] so that a
//! document or graph index can have an image beside this one, `flags` in both
//! headers so that a section can be added to an image a version one reader then
//! refuses one image at a time, and `bits` and `metric` as their own bytes
//! rather than as something a reader has to infer.
//!
//! An image is a cache in the end, which is the safety net under all of it: a
//! reader that does not like an image can throw it away and rebuild from the
//! records, slowly and correctly.

use crate::{get_u8, get_u16, get_u32, get_u64, put_u8, put_u16, put_u32, put_u64};
use yo_common::{Code, Error, Result};

/// The four bytes an image starts with, so that a stray chunk is not read as
/// one.
pub const IMAGE_TAG: u32 = u32::from_le_bytes(*b"YOIX");

/// The fixed part at the front of an image root.
pub const IMAGE_HEADER_LEN: usize = 88;

/// One line of the partition table that follows the root header.
pub const PARTITION_ENTRY_LEN: usize = 16;

/// The fixed part at the front of one partition's image.
pub const POSTING_HEADER_LEN: usize = 16;

/// What a code needs beside it: `norm`, `scale`, `lo` and `delta`, four `f32`.
pub const META_LEN: usize = 16;

/// What kind of index an image holds.
///
/// One value today and a byte for it, because the vector index is the first
/// thing worth writing down and it will not be the last. A reader that meets a
/// kind it does not know throws the image away and rebuilds, which is always
/// available and is why this is a byte rather than a format version.
pub mod image_kind {
    /// The partition index over RaBitQ codes (`10`).
    pub const VECTOR: u8 = 1;
}

/// How a vector is compared, as the byte the image stores.
///
/// The same four values `yo_shape::Metric` has, written down here because this
/// crate is the format and it does not depend on the shape crate. A collection
/// built for cosine and searched as L2 answers wrongly and quietly, so the
/// metric travels with the image rather than being taken on trust from whatever
/// opened it.
pub mod metric {
    /// Euclidean distance.
    pub const L2: u8 = 0;
    /// Cosine similarity, stored as unit vectors.
    pub const COSINE: u8 = 1;
    /// Inner product.
    pub const IP: u8 = 2;
    /// Hamming distance.
    pub const HAMMING: u8 = 3;

    /// Whether `b` is a metric this version has a name for.
    #[must_use]
    pub const fn is_known(b: u8) -> bool {
        matches!(b, L2 | COSINE | IP | HAMMING)
    }
}

/// The root of an image: everything the index is, apart from the members.
///
/// The two chains here and the partition table after it are addresses into the
/// log, so this is small whatever the collection's size: a few dozen bytes plus
/// sixteen per partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageHeader {
    /// Which index this is an image of. See [`image_kind`].
    pub kind: u8,
    /// How wide one coordinate is written in a code, which is 1 or 4.
    pub bits: u8,
    /// What nearness means here. See [`metric`].
    pub metric: u8,
    /// How many coordinates a vector has.
    pub dim: u32,
    /// How many partitions the index has grown to.
    pub partitions: u32,
    /// What the rotation was built from. Without it the codes are noise.
    pub seed: u64,
    /// How many vectors the collection held, for a check after loading.
    pub members: u64,
    /// How many slots the vector table had, which is the largest id plus one.
    ///
    /// An id is a slot rather than a name, and a collection that has had
    /// members removed has holes in it, so the count of live members does not
    /// say how far the ids go. Writing it down means a loader allocates the
    /// table once and can refuse an id that is past the end of it before it is
    /// used to index anything.
    pub slots: u32,
    /// The size a partition wants.
    pub posting: u32,
    /// How many partitions a search scans.
    pub probe: u32,
    /// How many candidates are reranked per answer.
    pub rerank: u32,
    /// How many neighbours a split sweeps.
    pub sweep: u32,
    /// How much further a filtered search looks.
    pub widen: u32,
    /// Where the centroids are, and how many bytes of them there are.
    pub centroids: Chain,
    /// Where the key table is, and how long it is.
    pub keys: Chain,
}

/// Where a section went and how long it is.
///
/// The same pair `yo_kv::cold::Chain` carries, written here as two `u64` because
/// this crate is the layout and does not depend on the crate that walks the
/// chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Chain {
    /// The address of the single chunk, or of the directory.
    pub at: u64,
    /// The section's length in bytes, which is what says which of those it is.
    pub len: u64,
}

/// How long an image root with this many partitions is.
///
/// # Errors
///
/// [`Code::Invalid`] if the partition count is large enough that the root would
/// not fit in a `usize`, which on a 64 bit machine it never is and on a 32 bit
/// one is a corrupt count rather than a real index.
pub fn image_len(partitions: u32) -> Result<usize> {
    (partitions as usize)
        .checked_mul(PARTITION_ENTRY_LEN)
        .and_then(|n| n.checked_add(IMAGE_HEADER_LEN))
        .ok_or_else(|| {
            Error::new(Code::Invalid, "that many partitions do not fit in an image")
                .with_detail(format!("partitions={partitions}"))
        })
}

impl ImageHeader {
    /// Writes the header into the first [`IMAGE_HEADER_LEN`] bytes of `into`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if `into` is shorter than a whole root, header and
    /// partition table both, because a header written into a buffer that cannot
    /// hold the table it describes is a root nobody can read.
    pub fn encode(&self, into: &mut [u8]) -> Result<usize> {
        let need = image_len(self.partitions)?;
        if into.len() < need {
            return Err(
                Error::new(Code::Invalid, "buffer is shorter than the image root")
                    .with_detail(format!("have={} need={need}", into.len())),
            );
        }
        put_u32(into, 0, IMAGE_TAG);
        put_u8(into, 4, self.kind);
        put_u8(into, 5, self.bits);
        put_u8(into, 6, self.metric);
        put_u8(into, 7, 0);
        put_u32(into, 8, self.dim);
        put_u32(into, 12, self.partitions);
        put_u64(into, 16, self.seed);
        put_u64(into, 24, self.members);
        put_u32(into, 32, self.posting);
        put_u32(into, 36, self.probe);
        put_u32(into, 40, self.rerank);
        put_u32(into, 44, self.sweep);
        put_u32(into, 48, self.widen);
        put_u32(into, 52, self.slots);
        put_u64(into, 56, self.centroids.at);
        put_u64(into, 64, self.centroids.len);
        put_u64(into, 72, self.keys.at);
        put_u64(into, 80, self.keys.len);
        Ok(need)
    }

    /// Reads a header back and checks that the bytes behind it are a whole
    /// partition table.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the tag is wrong, if a reserved field is set, if the
    /// dimension or the code width is not one this version writes, or if the
    /// buffer is not as long as the partition count says it is.
    pub fn decode(bytes: &[u8]) -> Result<ImageHeader> {
        if bytes.len() < IMAGE_HEADER_LEN {
            return Err(Error::new(Code::Corrupt, "shorter than an image header")
                .with_detail(format!("len={}", bytes.len())));
        }
        let tag = get_u32(bytes, 0);
        if tag != IMAGE_TAG {
            return Err(Error::new(Code::Corrupt, "not an index image")
                .with_detail(format!("tag={tag:#010x}")));
        }
        // Reserved bytes are checked rather than ignored, as everywhere else in
        // this crate: anything in them was written by something that did not
        // agree with this layout.
        if get_u8(bytes, 7) != 0 {
            return Err(Error::new(
                Code::Corrupt,
                "reserved image header bytes are set",
            ));
        }
        let h = ImageHeader {
            kind: get_u8(bytes, 4),
            bits: get_u8(bytes, 5),
            metric: get_u8(bytes, 6),
            dim: get_u32(bytes, 8),
            partitions: get_u32(bytes, 12),
            seed: get_u64(bytes, 16),
            members: get_u64(bytes, 24),
            posting: get_u32(bytes, 32),
            probe: get_u32(bytes, 36),
            rerank: get_u32(bytes, 40),
            sweep: get_u32(bytes, 44),
            widen: get_u32(bytes, 48),
            slots: get_u32(bytes, 52),
            centroids: Chain {
                at: get_u64(bytes, 56),
                len: get_u64(bytes, 64),
            },
            keys: Chain {
                at: get_u64(bytes, 72),
                len: get_u64(bytes, 80),
            },
        };
        if h.dim == 0 || h.dim as usize > crate::vector::MAX_DIM {
            return Err(Error::new(Code::Corrupt, "image dimension out of range")
                .with_detail(format!("dim={}", h.dim)));
        }
        if h.bits != 1 && h.bits != 4 {
            return Err(Error::new(Code::Corrupt, "unknown code width")
                .with_detail(format!("bits={}", h.bits)));
        }
        if h.members > u64::from(h.slots) {
            return Err(
                Error::new(Code::Corrupt, "more members than the table has slots")
                    .with_detail(format!("members={} slots={}", h.members, h.slots)),
            );
        }
        if !metric::is_known(h.metric) {
            return Err(Error::new(Code::Corrupt, "unknown metric")
                .with_detail(format!("metric={}", h.metric)));
        }
        let need = image_len(h.partitions)?;
        if bytes.len() != need {
            return Err(
                Error::new(Code::Corrupt, "the image root is not the length it says")
                    .with_detail(format!("len={} need={need}", bytes.len())),
            );
        }
        // The centroids are the one section whose size the header already
        // implies, so a disagreement there is worth catching before anything
        // tries to cut the section into partitions.
        let want = u64::from(h.partitions) * u64::from(h.dim) * 4;
        if h.centroids.len != want {
            return Err(
                Error::new(Code::Corrupt, "the centroid section is the wrong size")
                    .with_detail(format!("len={} want={want}", h.centroids.len)),
            );
        }
        Ok(h)
    }
}

/// Writes partition `i`'s chain into an encoded root.
///
/// # Errors
///
/// [`Code::Invalid`] if `i` is past the table.
pub fn put_partition(root: &mut [u8], i: u32, chain: Chain) -> Result<()> {
    let at =
        partition_offset(root.len(), i).ok_or_else(|| missing(i, root.len(), Code::Invalid))?;
    put_u64(root, at, chain.at);
    put_u64(root, at + 8, chain.len);
    Ok(())
}

/// Reads partition `i`'s chain back out of an encoded root.
///
/// # Errors
///
/// [`Code::Corrupt`] if `i` is past the table.
pub fn get_partition(root: &[u8], i: u32) -> Result<Chain> {
    let at =
        partition_offset(root.len(), i).ok_or_else(|| missing(i, root.len(), Code::Corrupt))?;
    Ok(Chain {
        at: get_u64(root, at),
        len: get_u64(root, at + 8),
    })
}

fn partition_offset(root_len: usize, i: u32) -> Option<usize> {
    let at = IMAGE_HEADER_LEN + (i as usize) * PARTITION_ENTRY_LEN;
    (at + PARTITION_ENTRY_LEN <= root_len).then_some(at)
}

/// A partition the table does not have, which is a caller's mistake on the way
/// in and a broken root on the way out.
fn missing(i: u32, root_len: usize, code: Code) -> Error {
    Error::new(code, "no such partition in the image")
        .with_detail(format!("partition={i} root={root_len}"))
}

/// The fixed part at the front of one partition's image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PostingHeader {
    /// How many members the partition holds.
    pub count: u32,
    /// How many bytes one code takes.
    ///
    /// Derivable from the root's `dim` and `bits`, and stored anyway for the
    /// same reason `dim` is stored in a vector record: a partition can then be
    /// cut into its four arrays without the root, which is what a reader
    /// checking a file rather than opening one has.
    pub code_bytes: u32,
    /// The size at which a split was tried and found there was no cut.
    ///
    /// Part of the image because it is not derivable: it is the memory of a
    /// split that failed, and an image that dropped it would have every
    /// partition of identical vectors try to split again on the first write
    /// after an open.
    pub stuck: u32,
}

/// How long one partition's image is.
///
/// # Errors
///
/// [`Code::Invalid`] if the arithmetic overflows, which means a corrupt count
/// rather than a real partition.
pub fn posting_len(count: u32, code_bytes: u32) -> Result<usize> {
    let count = count as usize;
    let per = 8usize
        .checked_add(8)
        .and_then(|n| n.checked_add(code_bytes as usize))
        .and_then(|n| n.checked_add(META_LEN))
        .ok_or_else(|| Error::new(Code::Invalid, "a member of that size does not fit"))?;
    count
        .checked_mul(per)
        .and_then(|n| n.checked_add(POSTING_HEADER_LEN))
        .ok_or_else(|| {
            Error::new(Code::Invalid, "that many members do not fit in a partition")
                .with_detail(format!("count={count} code_bytes={code_bytes}"))
        })
}

impl PostingHeader {
    /// Writes the header into the front of `into`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if `into` is not long enough for the whole partition.
    pub fn encode(&self, into: &mut [u8]) -> Result<usize> {
        let need = posting_len(self.count, self.code_bytes)?;
        if into.len() < need {
            return Err(
                Error::new(Code::Invalid, "buffer is shorter than the posting")
                    .with_detail(format!("have={} need={need}", into.len())),
            );
        }
        put_u32(into, 0, self.count);
        put_u32(into, 4, self.code_bytes);
        put_u32(into, 8, self.stuck);
        put_u32(into, 12, 0);
        Ok(need)
    }

    /// Reads a partition's header back and checks the bytes behind it.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if a reserved field is set or if the buffer is not
    /// exactly the length the header describes.
    pub fn decode(bytes: &[u8]) -> Result<PostingHeader> {
        if bytes.len() < POSTING_HEADER_LEN {
            return Err(Error::new(Code::Corrupt, "shorter than a posting header")
                .with_detail(format!("len={}", bytes.len())));
        }
        if get_u32(bytes, 12) != 0 {
            return Err(Error::new(
                Code::Corrupt,
                "reserved posting header bytes are set",
            ));
        }
        let h = PostingHeader {
            count: get_u32(bytes, 0),
            code_bytes: get_u32(bytes, 4),
            stuck: get_u32(bytes, 8),
        };
        let Ok(need) = posting_len(h.count, h.code_bytes) else {
            return Err(
                Error::new(Code::Corrupt, "that many members do not fit in a partition")
                    .with_detail(format!("count={} code_bytes={}", h.count, h.code_bytes)),
            );
        };
        if bytes.len() != need {
            return Err(
                Error::new(Code::Corrupt, "the posting is not the length it says")
                    .with_detail(format!("len={} need={need}", bytes.len())),
            );
        }
        Ok(h)
    }

    /// Where the ids start.
    #[must_use]
    pub const fn ids_at(&self) -> usize {
        POSTING_HEADER_LEN
    }

    /// Where the tags start.
    #[must_use]
    pub const fn tags_at(&self) -> usize {
        self.ids_at() + self.count as usize * 8
    }

    /// Where the codes start.
    #[must_use]
    pub const fn codes_at(&self) -> usize {
        self.tags_at() + self.count as usize * 8
    }

    /// Where the meta starts.
    #[must_use]
    pub const fn meta_at(&self) -> usize {
        self.codes_at() + self.count as usize * self.code_bytes as usize
    }
}

/// Writes floats into `into` end to end and says how many bytes that took.
///
/// Every float in an image goes through here: the centroid section is one long
/// run of them, and the four numbers beside a code are a run of four. Both are
/// bit for bit, because a centroid that comes back nearly right puts members
/// under the wrong partition and a code's scale that comes back nearly right
/// reorders the answers.
///
/// # Errors
///
/// [`Code::Invalid`] if `into` is too short.
pub fn put_floats(into: &mut [u8], values: &[f32]) -> Result<usize> {
    let need = values.len() * 4;
    if into.len() < need {
        return Err(
            Error::new(Code::Invalid, "buffer is shorter than the floats")
                .with_detail(format!("have={} need={need}", into.len())),
        );
    }
    for (i, v) in values.iter().enumerate() {
        crate::put_f32(into, i * 4, *v);
    }
    Ok(need)
}

/// Reads floats back out of `bytes` into `out`, which says how many.
///
/// # Errors
///
/// [`Code::Corrupt`] if there are not that many floats there, which is what a
/// section that disagrees with the header it was described by looks like.
pub fn get_floats(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let need = out.len() * 4;
    if bytes.len() < need {
        return Err(
            Error::new(Code::Corrupt, "the section is shorter than its floats")
                .with_detail(format!("len={} need={need}", bytes.len())),
        );
    }
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = crate::get_f32(bytes, i * 4);
    }
    Ok(())
}

/// How many bytes a key of `klen` takes in the key table.
///
/// # Errors
///
/// [`Code::Invalid`] if the key is longer than a record's key can be, which is
/// the same limit for the same reason: a key is a thing you look up by.
pub fn key_entry_len(klen: usize) -> Result<usize> {
    if klen > crate::record::MAX_KEY_LEN {
        return Err(
            Error::new(Code::Invalid, "the key is longer than 65535 bytes")
                .with_detail(format!("klen={klen}")),
        );
    }
    Ok(10 + klen)
}

/// Writes one key table entry and says how long it was.
///
/// The table is a run of `id`, `klen`, key, with no padding and no order worth
/// relying on, because the only thing that reads it reads all of it. A key is
/// tens of bytes and the alignment would cost more than the sequential read
/// saves.
///
/// # Errors
///
/// [`Code::Invalid`] if the key is too long or `into` is too short.
pub fn put_key(into: &mut [u8], id: u64, key: &[u8]) -> Result<usize> {
    let need = key_entry_len(key.len())?;
    if into.len() < need {
        return Err(
            Error::new(Code::Invalid, "buffer is shorter than the key entry")
                .with_detail(format!("have={} need={need}", into.len())),
        );
    }
    put_u64(into, 0, id);
    put_u16(into, 8, key.len() as u16);
    into[10..need].copy_from_slice(key);
    Ok(need)
}

/// The key table, one entry at a time.
///
/// Stops at the first entry that does not fit, which is what a truncated
/// section looks like, and the caller compares the count it got against the
/// member count in the header rather than being told twice.
#[derive(Debug, Clone)]
pub struct Keys<'a> {
    rest: &'a [u8],
}

impl<'a> Keys<'a> {
    /// Walks the entries in `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Keys<'a> {
        Keys { rest: bytes }
    }

    /// Whether every byte handed in was accounted for.
    ///
    /// False after a short entry, which is the one thing walking the table
    /// cannot tell a caller by ending.
    #[must_use]
    pub const fn done(&self) -> bool {
        self.rest.is_empty()
    }
}

impl<'a> Iterator for Keys<'a> {
    type Item = (u64, &'a [u8]);

    fn next(&mut self) -> Option<(u64, &'a [u8])> {
        if self.rest.len() < 10 {
            return None;
        }
        let id = get_u64(self.rest, 0);
        let klen = get_u16(self.rest, 8) as usize;
        let end = 10 + klen;
        if self.rest.len() < end {
            return None;
        }
        let key = &self.rest[10..end];
        self.rest = &self.rest[end..];
        Some((id, key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(partitions: u32, dim: u32) -> ImageHeader {
        ImageHeader {
            kind: image_kind::VECTOR,
            bits: 1,
            metric: metric::COSINE,
            dim,
            partitions,
            seed: 0x0102_0304_0506_0708,
            members: 9999,
            slots: 10_000,
            posting: 256,
            probe: 8,
            rerank: 4,
            sweep: 4,
            widen: 8,
            centroids: Chain {
                at: 4096,
                len: u64::from(partitions) * u64::from(dim) * 4,
            },
            keys: Chain { at: 8192, len: 123 },
        }
    }

    #[test]
    fn a_root_comes_back_field_for_field() {
        let h = header(3, 128);
        let mut buf = vec![0u8; image_len(3).unwrap()];
        let wrote = h.encode(&mut buf).unwrap();
        assert_eq!(wrote, buf.len());
        assert_eq!(ImageHeader::decode(&buf).unwrap(), h);
    }

    #[test]
    fn the_partition_table_is_addressed_and_not_walked() {
        let h = header(4, 8);
        let mut buf = vec![0u8; image_len(4).unwrap()];
        h.encode(&mut buf).unwrap();
        for i in 0..4 {
            let chain = Chain {
                at: 1000 + u64::from(i),
                len: 64 + u64::from(i),
            };
            put_partition(&mut buf, i, chain).unwrap();
        }
        for i in 0..4 {
            assert_eq!(
                get_partition(&buf, i).unwrap(),
                Chain {
                    at: 1000 + u64::from(i),
                    len: 64 + u64::from(i)
                }
            );
        }
        assert!(get_partition(&buf, 4).is_err(), "there is no fifth");
        assert!(put_partition(&mut buf, 9, Chain::default()).is_err());
    }

    #[test]
    fn a_root_that_is_not_a_root_is_refused() {
        let h = header(2, 16);
        let mut buf = vec![0u8; image_len(2).unwrap()];
        h.encode(&mut buf).unwrap();
        assert!(ImageHeader::decode(&buf).is_ok());

        let mut wrong = buf.clone();
        put_u32(&mut wrong, 0, 0xdead_beef);
        assert_eq!(
            ImageHeader::decode(&wrong).unwrap_err().code(),
            Code::Corrupt,
            "a chunk that is not an image was read as one"
        );

        let mut set = buf.clone();
        set[7] = 1;
        assert!(
            ImageHeader::decode(&set).is_err(),
            "byte 7 is reserved and a writer that set it disagreed with this layout"
        );

        let mut short = buf.clone();
        put_u32(&mut short, 52, 3);
        assert!(
            ImageHeader::decode(&short).is_err(),
            "a table with fewer slots than members cannot hold them"
        );

        let mut bits = buf.clone();
        put_u8(&mut bits, 5, 2);
        assert!(ImageHeader::decode(&bits).is_err(), "no two bit codes");

        let mut met = buf.clone();
        put_u8(&mut met, 6, 9);
        assert!(ImageHeader::decode(&met).is_err(), "no ninth metric");

        let mut dim = buf.clone();
        put_u32(&mut dim, 8, 0);
        assert!(ImageHeader::decode(&dim).is_err(), "no zero dimension");

        // The count and the buffer have to agree, because a count that is
        // believed on its own is a count that indexes past the table.
        let mut count = buf.clone();
        put_u32(&mut count, 12, 99);
        assert!(ImageHeader::decode(&count).is_err());

        let mut cent = buf.clone();
        put_u64(&mut cent, 64, 7);
        assert!(
            ImageHeader::decode(&cent).is_err(),
            "the centroid section has to be partitions times dim floats"
        );

        for len in 0..buf.len() {
            assert!(
                ImageHeader::decode(&buf[..len]).is_err(),
                "{len} bytes decoded as a two partition image"
            );
        }
    }

    #[test]
    fn a_partition_is_four_runs_that_do_not_overlap() {
        let h = PostingHeader {
            count: 5,
            code_bytes: 16,
            stuck: 12,
        };
        let mut buf = vec![0u8; posting_len(5, 16).unwrap()];
        let wrote = h.encode(&mut buf).unwrap();
        assert_eq!(wrote, buf.len());
        assert_eq!(PostingHeader::decode(&buf).unwrap(), h);

        assert_eq!(h.ids_at(), POSTING_HEADER_LEN);
        assert_eq!(h.tags_at(), h.ids_at() + 40);
        assert_eq!(h.codes_at(), h.tags_at() + 40);
        assert_eq!(h.meta_at(), h.codes_at() + 80);
        assert_eq!(h.meta_at() + 5 * META_LEN, buf.len());
    }

    #[test]
    fn an_empty_partition_is_a_header_and_nothing_else() {
        let h = PostingHeader {
            count: 0,
            code_bytes: 96,
            stuck: 0,
        };
        let mut buf = vec![0u8; POSTING_HEADER_LEN];
        h.encode(&mut buf).unwrap();
        assert_eq!(PostingHeader::decode(&buf).unwrap(), h);
        assert_eq!(h.meta_at(), buf.len());
    }

    #[test]
    fn a_posting_that_is_not_the_length_it_claims_is_refused() {
        let h = PostingHeader {
            count: 3,
            code_bytes: 8,
            stuck: 0,
        };
        let mut buf = vec![0u8; posting_len(3, 8).unwrap()];
        h.encode(&mut buf).unwrap();
        for len in 0..buf.len() {
            assert!(
                PostingHeader::decode(&buf[..len]).is_err(),
                "{len} bytes decoded as three members"
            );
        }
        let mut set = buf.clone();
        set[12] = 1;
        assert!(PostingHeader::decode(&set).is_err(), "reserved");

        // A count nobody could have written, which is what a corrupt header
        // looks like, and it has to be refused before anything multiplies it out.
        let mut huge = buf.clone();
        put_u32(&mut huge, 0, u32::MAX);
        put_u32(&mut huge, 4, u32::MAX);
        assert!(PostingHeader::decode(&huge).is_err());
    }

    #[test]
    fn floats_go_down_and_come_back_bit_for_bit() {
        let values = [0.0f32, -0.0, 1.5, -2.25, 1e-38, 3.4e38];
        let mut buf = vec![0u8; values.len() * 4];
        assert_eq!(put_floats(&mut buf, &values).unwrap(), buf.len());
        let mut back = vec![0f32; values.len()];
        get_floats(&buf, &mut back).unwrap();
        for (a, b) in values.iter().zip(&back) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a} came back as {b}");
        }
        assert!(get_floats(&buf[..4], &mut back).is_err(), "not that many");
        assert!(put_floats(&mut buf[..4], &values).is_err());
    }

    #[test]
    fn the_key_table_walks_back_in_order() {
        let entries: Vec<(u64, &[u8])> = vec![
            (0, b"a".as_slice()),
            (7, b"".as_slice()),
            (3, b"a rather longer key than the first one".as_slice()),
        ];
        let mut buf = Vec::new();
        for (id, key) in &entries {
            let mut one = vec![0u8; key_entry_len(key.len()).unwrap()];
            let wrote = put_key(&mut one, *id, key).unwrap();
            assert_eq!(wrote, one.len());
            buf.extend_from_slice(&one);
        }
        let mut walk = Keys::new(&buf);
        let got: Vec<(u64, &[u8])> = walk.by_ref().collect();
        assert_eq!(got, entries);
        assert!(walk.done(), "the walk left bytes behind");
    }

    #[test]
    fn a_truncated_key_table_stops_rather_than_reading_past_it() {
        let mut buf = vec![0u8; key_entry_len(4).unwrap()];
        put_key(&mut buf, 1, b"abcd").unwrap();
        assert!(Keys::new(&[]).done(), "no bytes is an empty table");
        for len in 1..buf.len() {
            let mut walk = Keys::new(&buf[..len]);
            assert_eq!(walk.by_ref().count(), 0, "{len} bytes gave a whole key");
            assert!(!walk.done(), "a short entry is not a finished table");
        }
        assert!(key_entry_len(70_000).is_err());
    }

    #[test]
    fn the_layout_is_the_one_written_down() {
        // The numbers in the module diagram, so that a change to any of them is
        // a change to a test rather than a silent change to the format.
        assert_eq!(IMAGE_HEADER_LEN, 88);
        assert_eq!(PARTITION_ENTRY_LEN, 16);
        assert_eq!(POSTING_HEADER_LEN, 16);
        assert_eq!(META_LEN, 16);
        assert_eq!(IMAGE_TAG, u32::from_le_bytes(*b"YOIX"));
        assert_eq!(image_len(0).unwrap(), IMAGE_HEADER_LEN);
        assert_eq!(image_len(1).unwrap(), IMAGE_HEADER_LEN + 16);
        // A partition at the default posting size and 768 dimensions, which is
        // the case the chunk size was chosen for.
        assert!(
            posting_len(256, 96).unwrap() < 64 * 1024,
            "a partition should be one chunk at the sizes it is tuned for"
        );
    }
}
