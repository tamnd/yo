//! The chunk chain, run against the log it is meant for rather than a vector.
//!
//! `yo_kv::cold` talks to storage through [`Blocks`], which is two methods, and
//! its unit tests implement that trait over a `Vec<Vec<u8>>`. That proves the
//! chaining arithmetic and proves nothing about whether the seam fits the log
//! this is going to be bolted to. So this file implements the same trait over a
//! real `yo_record::Log` and puts a value through it.
//!
//! A dev dependency and not a real one, on purpose. Which crate owns the wiring
//! between the memory engine and the record plane is not settled, and answering
//! it by adding an edge in a test file would be answering it by accident.

use yo_common::{Addr, Code, Error, Result, Space};
use yo_format::{RecordHeader, RecordKind};
use yo_kv::cold::{self, Blocks, CHUNK, Chain, Reader, Scratch};
use yo_record::{Log, LogConfig, MemorySink};

/// The log, wearing the interface `cold` asks for.
///
/// The whole adapter is nine lines, which is the thing being checked. A seam
/// that needed a translation layer would be a seam in the wrong place.
struct LogBlocks {
    log: Log<MemorySink>,
}

impl Blocks for LogBlocks {
    fn put(&mut self, bytes: &[u8]) -> Result<Addr> {
        // The kind the format already has a name for: one chunk of a
        // collection, or of a value too large for a page.
        let h = RecordHeader::new(RecordKind::CollectionChunk);
        let a = self.log.append(&h, b"", bytes)?;
        Ok(Addr::new(Space::Log, a.addr))
    }

    fn get(&self, at: Addr) -> Result<&[u8]> {
        if at.space() != Some(Space::Log) {
            return Err(Error::new(Code::Invalid, "that address is not in the log"));
        }
        Ok(self.log.read(at.offset())?.value)
    }
}

fn log() -> LogBlocks {
    // A megabyte page rather than the production 32 MiB, because the test wants
    // a few hundred kilobytes and not a few hundred megabytes of zeroed buffer.
    // Four resident pages so that nothing written here is evicted before it is
    // read, which is a property of the log and not of the chain.
    let cfg = LogConfig {
        page_len: 1024 * 1024,
        resident_pages: 4,
        ..LogConfig::default()
    };
    LogBlocks {
        log: Log::new(cfg, MemorySink::new()).expect("a log with a workable shape"),
    }
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn whole(r: &Reader<'_, LogBlocks>) -> Vec<u8> {
    let mut out = Vec::new();
    for piece in r.range(0, r.len()) {
        out.extend_from_slice(piece.expect("a piece the value has"));
    }
    out
}

#[test]
fn a_value_bigger_than_a_record_goes_through_the_log_and_comes_back() {
    let mut blocks = log();
    let value = pattern(3 * CHUNK + 41);
    let chain = cold::write(&mut blocks, &value, &mut Scratch::new()).expect("written");

    let r = Reader::open(&blocks, chain).expect("opened");
    assert_eq!(r.chunks(), 4);
    assert_eq!(whole(&r), value);
}

#[test]
fn a_short_value_is_one_record_with_no_directory_in_front_of_it() {
    let mut blocks = log();
    let value = pattern(900);
    let chain = cold::write(&mut blocks, &value, &mut Scratch::new()).expect("written");

    // The address the chain kept is the value itself, so a demoted string is
    // one read and the directory case never happens for it.
    assert_eq!(
        blocks.get(chain.at).expect("the record"),
        &value[..],
        "a short value went through a directory it did not need"
    );
}

#[test]
fn a_chunk_in_the_middle_is_reachable_without_reading_the_ones_before_it() {
    let mut blocks = log();
    let value = pattern(4 * CHUNK);
    let chain = cold::write(&mut blocks, &value, &mut Scratch::new()).expect("written");

    let r = Reader::open(&blocks, chain).expect("opened");
    let third = r.chunk(2).expect("the third chunk");
    assert_eq!(third, &value[2 * CHUNK..3 * CHUNK]);
}

#[test]
fn an_address_from_another_space_is_refused_rather_than_read() {
    let blocks = log();
    // What a stale or scrambled index entry looks like from here: a plausible
    // offset carrying the wrong space.
    let wrong = Chain {
        at: Addr::new(Space::Arena, 0),
        len: 10,
    };
    let r = Reader::open(&blocks, wrong).expect("a single chunk chain opens without a read");
    assert!(
        r.chunk(0).is_err(),
        "an arena address was followed into the log"
    );
}

#[test]
fn several_values_in_one_log_do_not_read_each_others_chunks() {
    let mut blocks = log();
    let mut scratch = Scratch::new();
    let mut written = Vec::new();
    for n in 1..=3usize {
        let value = pattern(n * CHUNK + n);
        let chain = cold::write(&mut blocks, &value, &mut scratch).expect("written");
        written.push((value, chain));
    }
    for (value, chain) in &written {
        let r = Reader::open(&blocks, *chain).expect("opened");
        assert_eq!(whole(&r), *value, "a value came back as somebody else's");
    }
}
