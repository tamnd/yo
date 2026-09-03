//! Every string command against a key whose value is not in memory.
//!
//! The point of value separation is that nothing above it can tell. A key that
//! has been demoted answers `GET`, `APPEND`, `SETRANGE`, `GETRANGE`, `SETBIT`,
//! `PFCOUNT`, `DUMP` and the rest exactly as it did before, and the commands
//! that can answer out of the record keep doing that without touching the
//! device at all.
//!
//! The store here is a vector rather than a log, for the reason the unit tests
//! use one: it counts its reads, and reads per command is what G9 is a gate on.
//! `chain_on_the_log.rs` is the same layer against a real log.

use std::cell::Cell;
use std::rc::Rc;

use yo_common::{Addr, Code, Error, Result, Space};
use yo_kv::cold::Blocks;
use yo_kv::{Encoding, Expire, Keyspace, Str};

/// A store that remembers how many times it was read.
///
/// The counter is shared out rather than reached through the database, because
/// the database owns the store and a caller wanting the number while holding a
/// mutable borrow cannot have both.
struct Mem {
    blobs: Vec<Vec<u8>>,
    reads: Rc<Cell<usize>>,
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
            .ok_or_else(|| Error::new(Code::Corrupt, "no chunk at that address"))
    }

    fn bytes(&self) -> u64 {
        self.blobs.iter().map(|b| b.len() as u64).sum()
    }
}

/// A database with somewhere to put values, and the read counter.
fn db() -> (Keyspace, Rc<Cell<usize>>) {
    let reads = Rc::new(Cell::new(0));
    let mut k = Keyspace::new();
    k.attach(Box::new(Mem {
        blobs: Vec::new(),
        reads: Rc::clone(&reads),
    }));
    (k, reads)
}

/// A value long enough to be worth demoting, and long enough that twelve bytes
/// of address read as a value would be obvious.
fn value() -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog, at length and repeatedly"
        .repeat(4)
        .to_vec()
}

/// A database holding `key` with its value already on the file.
fn cold(key: &[u8], val: &[u8]) -> (Keyspace, Rc<Cell<usize>>) {
    let (mut k, reads) = db();
    k.set_plain(key, val).expect("stored");
    assert!(
        k.demote(key).expect("demoted"),
        "the value should have gone"
    );
    reads.set(0);
    (k, reads)
}

#[test]
fn get_answers_the_value_that_was_stored() {
    let val = value();
    let (mut k, reads) = cold(b"k", &val);
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
    assert_eq!(reads.get(), 1, "one chunk is one read");
}

#[test]
fn the_first_read_does_not_bring_it_back_and_the_second_one_does() {
    // The doorkeeper. A single read of a cold key is what a scan looks like,
    // and a scan must not be able to fill memory with what it walked past.
    let val = value();
    let (mut k, reads) = cold(b"k", &val);

    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
    let stats = k.tier().expect("attached").stats();
    assert_eq!(stats.served, 1, "the first read should have been served");
    assert_eq!(stats.promoted, 0, "and should not have promoted");

    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
    assert_eq!(k.tier().expect("attached").stats().promoted, 1);

    // And now it is in memory, so nothing else goes to the device.
    let before = reads.get();
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
    assert_eq!(reads.get(), before, "a warm read is not a device read");
}

#[test]
fn the_header_still_answers_without_touching_the_device() {
    // What stays in memory when a value leaves is the argument for demoting at
    // all. Every one of these would be a device read in a design that moved the
    // whole record.
    let val = value();
    let (mut k, reads) = cold(b"k", &val);
    let when = k.clock().now_ms() + 100_000;
    k.expire(b"k", when, yo_kv::Cond::Always);

    assert_eq!(k.strlen(b"k").expect("length"), val.len());
    assert!(k.exists(b"k"));
    assert_eq!(k.type_name(b"k"), Some("string"));
    assert_eq!(k.encoding(b"k"), Some(Encoding::Raw));
    assert!(matches!(k.deadline_of(b"k"), yo_kv::Ask::At(w) if w == when));
    assert_eq!(reads.get(), 0, "the record answered all of it");
}

#[test]
fn a_deadline_moves_without_reading_the_value_back() {
    // A deadline lives in front of the address, not in the value, so `EXPIRE`
    // on a demoted key rewrites twelve bytes and leaves the chunks alone.
    let val = value();
    let (mut k, reads) = cold(b"k", &val);
    let when = k.clock().now_ms() + 100_000;
    assert!(k.set_expiry(b"k", Some(when)));
    assert_eq!(reads.get(), 0, "no chunk should have been read");
    assert_eq!(k.expire_at(b"k"), Some(when));
    assert_eq!(k.strlen(b"k").expect("length"), val.len());
    // And the value is still on the file and still readable.
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
    assert_eq!(reads.get(), 1);
}

#[test]
fn getrange_reads_the_window_it_was_asked_for() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    let got = k.getrange(b"k", 4, 8).expect("range");
    assert_eq!(&got[..], &val[4..9]);
}

#[test]
fn append_brings_the_value_back_and_adds_to_it() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    let len = k.append(b"k", b"!!").expect("appended");
    assert_eq!(len, val.len() + 2);

    let mut want = val.clone();
    want.extend_from_slice(b"!!");
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&want)));
    // A read modify write does not ask the doorkeeper, because the record it
    // leaves behind is resident whatever the answer would have been.
    assert_eq!(k.tier().expect("attached").stats().promoted, 1);
}

#[test]
fn setrange_writes_into_a_value_that_was_on_the_file() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    k.setrange(b"k", 0, b"THE").expect("wrote");

    let mut want = val.clone();
    want[..3].copy_from_slice(b"THE");
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&want)));
}

#[test]
fn setbit_and_getbit_see_the_same_bitmap() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    let before = k.getbit(b"k", 3).expect("read a bit");
    assert_eq!(k.setbit(b"k", 3, !before).expect("wrote a bit"), before);
    assert_eq!(k.getbit(b"k", 3).expect("read a bit"), !before);
    assert_eq!(k.strlen(b"k").expect("length"), val.len());
}

#[test]
fn a_sketch_still_counts_after_it_has_been_demoted() {
    let (mut k, _) = db();
    for i in 0..500u32 {
        k.pfadd(b"h", std::iter::once(&i.to_le_bytes()[..]))
            .expect("added");
    }
    let want = k.pfcount(std::iter::once(&b"h"[..])).expect("counted");
    assert!(k.demote(b"h").expect("demoted"), "a sketch is a string");
    assert_eq!(
        k.pfcount(std::iter::once(&b"h"[..])).expect("counted"),
        want
    );
}

#[test]
fn dump_reads_the_value_without_putting_it_back() {
    // A backup walks every key, and a backup that promoted everything it read
    // would be the worst thing that can happen to a tiered store.
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    let dumped = k.export(b"k").expect("exported");
    // The record is opaque out here, so it goes back in under another name and
    // is read through the front door, which is what `COPY` does with it anyway.
    k.import(b"copy", dumped);
    assert_eq!(k.get(b"copy").expect("read"), Some(Str::Bytes(&val)));
    assert_eq!(k.tier().expect("attached").stats().promoted, 0);
}

#[test]
fn getdel_hands_the_value_over_and_takes_the_key_with_it() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    assert_eq!(k.getdel(b"k").expect("deleted"), Some(val));
    assert!(!k.exists(b"k"));
    assert_eq!(
        k.tier().expect("attached").stats().promoted,
        0,
        "a key on its way out does not need to come back first"
    );
}

#[test]
fn incr_on_a_demoted_string_says_it_is_not_a_number() {
    // Not the interesting answer, but the interesting failure: reading twelve
    // bytes of address as digits would answer something.
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    assert!(k.incr(b"k").is_err());
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
}

#[test]
fn set_over_a_demoted_key_costs_no_device_read() {
    // Overwriting does not need what was there. This is the case that would be
    // a wasted read in a design that always faulted before writing.
    let val = value();
    let (mut k, reads) = cold(b"k", &val);
    k.set_plain(b"k", b"small").expect("stored");
    assert_eq!(reads.get(), 0);
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(b"small")));
}

#[test]
fn set_with_get_hands_back_what_was_on_the_file() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    let out = k
        .set(
            b"k",
            b"next",
            yo_kv::SetOptions {
                get: true,
                ..yo_kv::SetOptions::PLAIN
            },
        )
        .expect("stored");
    assert_eq!(out.previous, Some(val));
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(b"next")));
}

#[test]
fn getex_reads_and_can_leave_it_where_it_is() {
    let val = value();
    let (mut k, _) = cold(b"k", &val);
    assert_eq!(
        k.getex(b"k", Expire::Keep).expect("read"),
        Some(Str::Bytes(&val))
    );
    assert_eq!(
        k.tier().expect("attached").stats().promoted,
        0,
        "plain GETEX is a read and the doorkeeper gets its vote"
    );
}

#[test]
fn lcs_reads_two_demoted_values_one_after_the_other() {
    // Two keys and one buffer, which is the case that would quietly compare a
    // value against itself if the second fault overwrote the first.
    let (mut k, _) = db();
    let a = b"the quick brown fox jumps over the lazy dog".repeat(2);
    let b = b"the quick brown cat jumps over the lazy dog".repeat(2);
    k.set_plain(b"a", &a).expect("stored");
    k.set_plain(b"b", &b).expect("stored");
    assert!(k.demote(b"a").expect("demoted"));
    assert!(k.demote(b"b").expect("demoted"));

    let want = {
        let (mut plain, _) = (Keyspace::new(), ());
        plain.set_plain(b"a", &a).expect("stored");
        plain.set_plain(b"b", &b).expect("stored");
        plain.lcs(b"a", b"b").expect("compared")
    };
    assert_eq!(k.lcs(b"a", b"b").expect("compared"), want);
}

#[test]
fn a_sweep_moves_the_whole_database_and_every_key_still_answers() {
    let (mut k, _) = db();
    let val = value();
    for i in 0..200u32 {
        k.set_plain(&i.to_le_bytes(), &val).expect("stored");
    }
    let swept = k.relieve(usize::MAX).expect("swept");
    assert!(swept.moved > 0, "nothing was moved");

    for i in 0..200u32 {
        assert_eq!(
            k.get(&i.to_le_bytes()).expect("read"),
            Some(Str::Bytes(&val)),
            "key {i} did not read back"
        );
    }
    assert_eq!(k.len(), 200, "a sweep moves values and not keys");
}

#[test]
fn a_database_with_nothing_attached_never_goes_cold() {
    let mut k = Keyspace::new();
    let val = value();
    k.set_plain(b"k", &val).expect("stored");
    assert!(
        !k.demote(b"k").expect("asked"),
        "there is nowhere to put it"
    );
    assert_eq!(k.relieve(usize::MAX).expect("swept").moved, 0);
    assert_eq!(k.get(b"k").expect("read"), Some(Str::Bytes(&val)));
}
