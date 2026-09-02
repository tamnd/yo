//! A value that is too big to hold, cut into chunks that are addressed rather
//! than walked.
//!
//! `05` section 4.4 and the bottom row of section 5. Two different problems get
//! the same answer here. One is a value of a megabyte, which must never be a
//! single arena allocation that then gets copied twice, because that shape is
//! what killed aki's `HGETALL` (L22). The other is a collection body that has
//! left memory, where a membership test should fault one chunk and a full
//! enumeration should stream them.
//!
//! Both want the same thing: fixed size pieces in the log, and a way to get to
//! piece `i` without reading pieces `0` through `i - 1`.
//!
//! # The layout
//!
//! A value of at most one chunk is one record and nothing else. That is the
//! common case for a demoted string and it costs one read, which matters,
//! because G9 is a gate on device reads per point read and a value that needs
//! two reads has already spent it.
//!
//! Anything longer is `n` chunk records plus a directory record holding their
//! addresses in order, eight bytes each:
//!
//! ```text
//!   directory                chunks
//! +----------+----------+   +--------- 64 KiB ---------+
//! | addr 0   | addr 1   |...| bytes 0 .. 65536         |
//! +----------+----------+   +--------------------------+
//!                           | bytes 65536 .. 131072    |
//!                           +--------------------------+
//! ```
//!
//! There is no second directory level and there does not need to be one. A
//! directory is itself at most one chunk, which is 65536 bytes, which is 8192
//! addresses, which is 8192 chunks, which is 512 MiB. That is exactly the
//! largest string Redis will hold, so the arithmetic closes and a chain is
//! always either one read or two.
//!
//! # What this module is not
//!
//! It does not decide when a value should be chunked, it does not own the log,
//! and it does not know what a key is. [`Blocks`] is the whole of its contact
//! with storage: somewhere to put bytes that hands back an address, and
//! somewhere to read an address back from. The log in `yo-record` is one
//! implementation of that and a vector in a test is another, which is what
//! keeps this testable without a file.

use yo_common::{Addr, Code, Error, Result, Space};

/// One chunk, and the unit everything here counts in.
///
/// 64 KiB is `05` section 4.4's number. It is large enough that a megabyte is
/// sixteen reads rather than two hundred and fifty, and small enough that a
/// membership test on a spilled collection faults kilobytes rather than
/// megabytes.
pub const CHUNK: usize = 64 * 1024;

/// How many addresses fit in a directory, which is how many chunks a chain can
/// have.
pub const FANOUT: usize = CHUNK / 8;

/// The longest value a chain can hold, which is one directory's worth of chunks.
///
/// 512 MiB, and not a coincidence: it is also the largest string Redis accepts,
/// so nothing that fits in the protocol fails to fit in a chain.
pub const MAX_LEN: u64 = (FANOUT * CHUNK) as u64;

/// Somewhere chunks go, and come back from.
///
/// Deliberately two methods. Everything this module needs from a log is an
/// append that hands back an address and a read that takes one, and writing the
/// trait that small is what lets the tests run against a vector instead of a
/// file.
pub trait Blocks {
    /// Put these bytes somewhere and say where they went.
    fn put(&mut self, bytes: &[u8]) -> Result<Addr>;

    /// Read back what was put at `at`.
    ///
    /// The length is not passed in because the store knows it. A log record
    /// carries its own length, and a caller that had to remember it would be
    /// keeping a second copy of something that is already written down.
    fn get(&self, at: Addr) -> Result<&[u8]>;

    /// How many bytes the store is holding, for the storage limit.
    ///
    /// The store's own size and not the sum of what was put in it. A log that
    /// has been written to and compacted knows what it occupies and nothing
    /// above it does, and `maxstore` is a limit on the file rather than on the
    /// payload that went into it.
    fn bytes(&self) -> u64;
}

/// So that a store can be chosen at run time rather than at compile time.
///
/// [`Keyspace`](crate::Keyspace) holds its tier behind this box, and the reason
/// is that the alternative is a type parameter on `Keyspace`, which would spread
/// to `yo-resp` and to every caller of either, all to name a type that only the
/// code opening the file knows. The dispatch it costs is one indirect call on a
/// path that is about to read a device, and nothing at all on a warm read, which
/// never reaches this trait.
impl Blocks for Box<dyn Blocks> {
    fn put(&mut self, bytes: &[u8]) -> Result<Addr> {
        (**self).put(bytes)
    }

    fn get(&self, at: Addr) -> Result<&[u8]> {
        (**self).get(at)
    }

    fn bytes(&self) -> u64 {
        (**self).bytes()
    }
}

/// Where a value went, and how much of it there is.
///
/// Twelve bytes, which is what [`value::write_cold_record`](crate::value) puts
/// in a demoted record. `at` is the single chunk when the value fits in one and
/// the directory when it does not, and `len` is what says which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chain {
    /// The one chunk, or the directory.
    pub at: Addr,
    /// The value's length in bytes, not the chain's.
    pub len: u64,
}

/// How many chunks a value of this length needs.
///
/// One for an empty value, because a chain always points at something. A zero
/// length value that pointed at nothing would need a third case in every reader
/// and there is nothing to gain from it.
#[must_use]
pub const fn chunks_for(len: u64) -> u64 {
    if len == 0 {
        1
    } else {
        len.div_ceil(CHUNK as u64)
    }
}

/// A place to build a directory, owned by the shard and used again every time.
///
/// Y7 says a command path does not allocate, and a chain of the largest value
/// Redis takes has a 64 KiB directory that has to be laid out somewhere before
/// it is written. So the shard keeps one of these and hands it in.
pub struct Scratch {
    dir: Vec<u8>,
}

impl Scratch {
    /// One directory's worth of room, allocated once.
    #[must_use]
    pub fn new() -> Scratch {
        Scratch {
            dir: Vec::with_capacity(CHUNK),
        }
    }

    /// What it costs to keep, for the memory report.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.dir.capacity()
    }
}

impl Default for Scratch {
    fn default() -> Scratch {
        Scratch::new()
    }
}

/// Write a value as a chain and say where it went.
///
/// The chunks go down before the directory does, so a directory that is
/// readable is a directory whose chunks are all readable. A reader that arrives
/// after a crash either finds no directory, in which case the chunks are
/// garbage that compaction will notice nobody points at, or finds one and can
/// trust every address in it.
pub fn write<B: Blocks>(blocks: &mut B, value: &[u8], scratch: &mut Scratch) -> Result<Chain> {
    let len = value.len() as u64;
    if len > MAX_LEN {
        return Err(Error::fmt(
            Code::Full,
            format_args!("a value of {len} bytes is longer than a chain holds"),
        ));
    }

    if value.len() <= CHUNK {
        return Ok(Chain {
            at: blocks.put(value)?,
            len,
        });
    }

    scratch.dir.clear();
    for piece in value.chunks(CHUNK) {
        let at = blocks.put(piece)?;
        scratch.dir.extend_from_slice(&at.to_bits().to_le_bytes());
    }
    let at = blocks.put(&scratch.dir)?;
    Ok(Chain { at, len })
}

/// A chain opened for reading.
///
/// Holding one means the directory has been read, so every chunk after that is
/// a single fetch. That is the whole reason this is a type rather than a
/// function taking a [`Chain`]: an enumeration that walks a spilled collection
/// reads the directory once and not once per chunk.
pub struct Reader<'a, B: Blocks> {
    blocks: &'a B,
    len: u64,
    /// The directory, or the address of the only chunk when there is no
    /// directory to have.
    dir: Dir<'a>,
}

enum Dir<'a> {
    One(Addr),
    Many(&'a [u8]),
}

impl<'a, B: Blocks> Reader<'a, B> {
    /// Read the directory, if there is one, and get ready to fetch chunks.
    pub fn open(blocks: &'a B, chain: Chain) -> Result<Reader<'a, B>> {
        let want = chunks_for(chain.len);
        let dir = if want == 1 {
            Dir::One(chain.at)
        } else {
            let bytes = blocks.get(chain.at)?;
            if bytes.len() as u64 != want * 8 {
                return Err(Error::fmt(
                    Code::Corrupt,
                    format_args!(
                        "a chain of {} bytes wants {want} addresses and its directory has {}",
                        chain.len,
                        bytes.len() / 8
                    ),
                ));
            }
            Dir::Many(bytes)
        };
        Ok(Reader {
            blocks,
            len: chain.len,
            dir,
        })
    }

    /// The value's length in bytes.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Whether the value has no bytes in it, which is not the same as having no
    /// chunks.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many chunks the value is in.
    #[must_use]
    pub const fn chunks(&self) -> u64 {
        chunks_for(self.len)
    }

    /// Fetch chunk `i`, which is one read and never a walk.
    pub fn chunk(&self, i: u64) -> Result<&'a [u8]> {
        let at = match self.dir {
            Dir::One(at) if i == 0 => at,
            Dir::One(_) => {
                return Err(Error::new(Code::Invalid, "there is only one chunk"));
            }
            Dir::Many(bytes) => {
                let start = (i as usize)
                    .checked_mul(8)
                    .filter(|s| s + 8 <= bytes.len())
                    .ok_or_else(|| Error::new(Code::Invalid, "no such chunk"))?;
                let mut bits = [0u8; 8];
                bits.copy_from_slice(&bytes[start..start + 8]);
                Addr::from_bits(u64::from_le_bytes(bits))
            }
        };
        if at.space() != Some(Space::Log) {
            return Err(Error::fmt(
                Code::Corrupt,
                format_args!("chunk {i} is not in the log"),
            ));
        }
        self.blocks.get(at)
    }

    /// Walk the pieces of `from .. to`, in order, without touching a chunk the
    /// range does not reach.
    ///
    /// This is what `GETRANGE` on a spilled value wants, and what a membership
    /// test on a spilled collection wants, and they are the same walk. A range
    /// inside one chunk is one fetch whatever the value's size is.
    pub fn range(&self, from: u64, to: u64) -> Pieces<'a, '_, B> {
        let to = to.min(self.len);
        let from = from.min(to);
        Pieces {
            reader: self,
            at: from,
            end: to,
        }
    }
}

/// The pieces of a byte range, one chunk at a time.
pub struct Pieces<'a, 'r, B: Blocks> {
    reader: &'r Reader<'a, B>,
    at: u64,
    end: u64,
}

impl<'a, B: Blocks> Iterator for Pieces<'a, '_, B> {
    type Item = Result<&'a [u8]>;

    fn next(&mut self) -> Option<Result<&'a [u8]>> {
        if self.at >= self.end {
            return None;
        }
        let chunk = self.at / CHUNK as u64;
        let start = (self.at % CHUNK as u64) as usize;
        let take = (self.end - self.at).min(CHUNK as u64 - start as u64) as usize;
        self.at += take as u64;
        Some(match self.reader.chunk(chunk) {
            Ok(bytes) if start + take <= bytes.len() => Ok(&bytes[start..start + take]),
            Ok(bytes) => Err(Error::fmt(
                Code::Corrupt,
                format_args!(
                    "chunk {chunk} is {} bytes and the range wants {}",
                    bytes.len(),
                    start + take
                ),
            )),
            Err(e) => Err(e),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store that keeps what it is given in memory and hands back the index
    /// as the address. Enough to exercise every path in here without a file,
    /// and it counts its reads, which is the number the gate is about.
    struct Mem {
        blobs: Vec<Vec<u8>>,
        reads: std::cell::Cell<usize>,
    }

    impl Mem {
        fn new() -> Mem {
            Mem {
                blobs: Vec::new(),
                reads: std::cell::Cell::new(0),
            }
        }

        fn reads(&self) -> usize {
            self.reads.get()
        }
    }

    impl Blocks for Mem {
        fn put(&mut self, bytes: &[u8]) -> Result<Addr> {
            self.blobs.push(bytes.to_vec());
            Ok(Addr::new(Space::Log, (self.blobs.len() - 1) as u64))
        }

        fn get(&self, at: Addr) -> Result<&[u8]> {
            self.reads.set(self.reads.get() + 1);
            self.blobs
                .get(at.offset() as usize)
                .map(Vec::as_slice)
                .ok_or_else(|| Error::new(Code::NotFound, "no such block"))
        }

        fn bytes(&self) -> u64 {
            self.blobs.iter().map(|b| b.len() as u64).sum()
        }
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn whole<B: Blocks>(r: &Reader<'_, B>) -> Vec<u8> {
        let mut out = Vec::new();
        for piece in r.range(0, r.len()) {
            out.extend_from_slice(piece.expect("a piece the value has"));
        }
        out
    }

    #[test]
    fn a_value_that_fits_in_one_chunk_has_no_directory() {
        let mut m = Mem::new();
        let value = pattern(1000);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        assert_eq!(
            m.blobs.len(),
            1,
            "a directory was written and should not be"
        );
        assert_eq!(chain.len, 1000);

        let r = Reader::open(&m, chain).expect("opened");
        assert_eq!(r.chunks(), 1);
        assert_eq!(whole(&r), value);
        // The gate is device reads per point read, so this is the number that
        // matters and it is one.
        assert_eq!(
            m.reads(),
            1,
            "reading a short value took more than one read"
        );
    }

    #[test]
    fn exactly_one_chunk_still_has_no_directory() {
        let mut m = Mem::new();
        let value = pattern(CHUNK);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        assert_eq!(m.blobs.len(), 1);
        let r = Reader::open(&m, chain).expect("opened");
        assert_eq!(r.chunks(), 1);
        assert_eq!(whole(&r), value);
    }

    #[test]
    fn one_byte_more_than_a_chunk_is_two_chunks_and_a_directory() {
        let mut m = Mem::new();
        let value = pattern(CHUNK + 1);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        assert_eq!(m.blobs.len(), 3, "two chunks and a directory");
        let r = Reader::open(&m, chain).expect("opened");
        assert_eq!(r.chunks(), 2);
        assert_eq!(r.chunk(1).expect("the second chunk").len(), 1);
        assert_eq!(whole(&r), value);
    }

    #[test]
    fn an_empty_value_is_one_empty_chunk() {
        let mut m = Mem::new();
        let chain = write(&mut m, b"", &mut Scratch::new()).expect("written");
        let r = Reader::open(&m, chain).expect("opened");
        assert!(r.is_empty());
        assert_eq!(r.chunks(), 1, "a chain always points at something");
        assert_eq!(whole(&r), b"");
    }

    #[test]
    fn a_range_inside_one_chunk_only_fetches_that_chunk() {
        let mut m = Mem::new();
        let value = pattern(10 * CHUNK);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");

        let r = Reader::open(&m, chain).expect("opened");
        let before = m.reads();
        let mut got = Vec::new();
        // Somewhere in the middle of chunk 7, which is the case that would go
        // wrong if the walk started at the beginning.
        let (from, to) = (7 * CHUNK as u64 + 100, 7 * CHUNK as u64 + 300);
        for piece in r.range(from, to) {
            got.extend_from_slice(piece.expect("a piece"));
        }
        assert_eq!(got, value[from as usize..to as usize]);
        assert_eq!(
            m.reads() - before,
            1,
            "a range inside one chunk of a ten chunk value should be one fetch"
        );
    }

    #[test]
    fn a_range_across_a_boundary_comes_back_in_two_pieces() {
        let mut m = Mem::new();
        let value = pattern(3 * CHUNK);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        let r = Reader::open(&m, chain).expect("opened");

        let (from, to) = (CHUNK as u64 - 5, CHUNK as u64 + 5);
        let pieces: Vec<usize> = r
            .range(from, to)
            .map(|p| p.expect("a piece").len())
            .collect();
        assert_eq!(
            pieces,
            vec![5, 5],
            "the boundary was not where it should be"
        );
    }

    #[test]
    fn a_range_past_the_end_stops_at_the_end() {
        let mut m = Mem::new();
        let value = pattern(100);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        let r = Reader::open(&m, chain).expect("opened");
        let mut got = Vec::new();
        for piece in r.range(50, 1_000_000) {
            got.extend_from_slice(piece.expect("a piece"));
        }
        assert_eq!(got, value[50..]);
        assert_eq!(r.range(200, 300).count(), 0, "there is nothing out there");
        assert_eq!(r.range(80, 20).count(), 0, "a backwards range is empty");
    }

    #[test]
    fn every_chunk_but_the_last_is_full() {
        let mut m = Mem::new();
        let value = pattern(2 * CHUNK + 7);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        let r = Reader::open(&m, chain).expect("opened");
        assert_eq!(r.chunks(), 3);
        assert_eq!(r.chunk(0).expect("chunk 0").len(), CHUNK);
        assert_eq!(r.chunk(1).expect("chunk 1").len(), CHUNK);
        assert_eq!(r.chunk(2).expect("chunk 2").len(), 7);
        assert!(r.chunk(3).is_err(), "there is no fourth chunk");
    }

    #[test]
    fn a_multi_chunk_value_is_two_reads_and_not_more() {
        let mut m = Mem::new();
        let value = pattern(5 * CHUNK);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        let before = m.reads();
        let r = Reader::open(&m, chain).expect("opened");
        // Opening read the directory. Any one chunk after that is one more,
        // whatever the value's size, which is the property the whole layout
        // exists for.
        assert_eq!(m.reads() - before, 1);
        r.chunk(4).expect("the last chunk");
        assert_eq!(m.reads() - before, 2);
    }

    #[test]
    fn a_directory_that_does_not_match_the_length_is_refused() {
        let mut m = Mem::new();
        let value = pattern(2 * CHUNK);
        let chain = write(&mut m, &value, &mut Scratch::new()).expect("written");
        // Say the value is longer than the directory can account for, which is
        // what a torn write or a stale address looks like from here.
        let lying = Chain {
            at: chain.at,
            len: 9 * CHUNK as u64,
        };
        assert!(
            Reader::open(&m, lying).is_err(),
            "a directory that is the wrong size was accepted"
        );
    }

    #[test]
    fn a_value_longer_than_a_chain_holds_is_refused_rather_than_truncated() {
        // Not by building one, which would be half a gigabyte of test. The
        // arithmetic is the thing being checked: one directory is one chunk, so
        // a chain tops out at exactly the largest string the protocol carries.
        assert_eq!(MAX_LEN, 512 * 1024 * 1024);
        assert_eq!(chunks_for(MAX_LEN), FANOUT as u64);
        assert_eq!(chunks_for(MAX_LEN + 1), FANOUT as u64 + 1);
    }

    #[test]
    fn the_scratch_is_reused_and_does_not_grow_with_every_write() {
        let mut m = Mem::new();
        let mut scratch = Scratch::new();
        let value = pattern(4 * CHUNK);
        for _ in 0..8 {
            write(&mut m, &value, &mut scratch).expect("written");
        }
        assert_eq!(
            scratch.memory_bytes(),
            CHUNK,
            "the directory buffer grew, so a command path is allocating"
        );
    }

    #[test]
    fn what_went_in_comes_back_at_every_awkward_size() {
        let mut m = Mem::new();
        let mut scratch = Scratch::new();
        for len in [
            0,
            1,
            CHUNK - 1,
            CHUNK,
            CHUNK + 1,
            2 * CHUNK - 1,
            2 * CHUNK,
            2 * CHUNK + 1,
            3 * CHUNK + 123,
        ] {
            let value = pattern(len);
            let chain = write(&mut m, &value, &mut scratch).expect("written");
            let r = Reader::open(&m, chain).expect("opened");
            assert_eq!(r.len(), len as u64);
            assert_eq!(whole(&r), value, "a value of {len} bytes came back wrong");
        }
    }
}
