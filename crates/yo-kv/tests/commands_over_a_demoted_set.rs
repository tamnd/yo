//! Set commands against a key whose body is not in memory.
//!
//! The same promise `commands_over_a_demoted_value.rs` makes about strings, for
//! the type whose body lives in a slab rather than in the record. Nothing above
//! the keyspace can tell that a set was on the device a moment ago: it answers
//! `SCARD`, `SISMEMBER`, `SADD`, `SMEMBERS` and the set algebra with the members
//! it had, in the representation it had them in.
//!
//! The store is a vector that counts its reads, as in the string file and for
//! the same reason: reads per command is what G9 is a gate on, and a set that
//! came back in one read is the number this milestone is about.

use std::cell::Cell;
use std::rc::Rc;

use yo_common::{Addr, Code, Error, Result, Space};
use yo_kv::cold::Blocks;
use yo_kv::{Keyspace, set};

/// A store that remembers how many times it was read.
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

fn db() -> (Keyspace, Rc<Cell<usize>>) {
    let reads = Rc::new(Cell::new(0));
    let mut k = Keyspace::new();
    k.attach(Box::new(Mem {
        blobs: Vec::new(),
        reads: Rc::clone(&reads),
    }));
    (k, reads)
}

/// Members that put the set past the listpack band and into a table, so the
/// body being moved is the one with something to save.
fn members(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| format!("member:{i:05}").into_bytes())
        .collect()
}

/// A database holding `key` as a set with its body already on the file.
fn cold(key: &[u8], of: &[Vec<u8>]) -> (Keyspace, Rc<Cell<usize>>) {
    let (mut k, reads) = db();
    k.sadd(key, of.iter().map(Vec::as_slice)).expect("added");
    assert!(k.demote(key).expect("demoted"), "the body should have gone");
    reads.set(0);
    (k, reads)
}

#[test]
fn a_demoted_set_answers_scard_with_what_it_held() {
    let all = members(400);
    let (mut k, _) = cold(b"s", &all);
    assert_eq!(k.scard(b"s").expect("counted"), 400);
}

#[test]
fn a_demoted_set_still_holds_every_member() {
    let all = members(400);
    let (mut k, _) = cold(b"s", &all);
    for m in &all {
        assert!(k.sismember(b"s", m).expect("asked"), "{m:?} is in it");
    }
    assert!(!k.sismember(b"s", b"member:99999").expect("asked"));
}

#[test]
fn bringing_a_set_back_costs_one_pass_over_its_chunks_and_then_nothing() {
    let all = members(400);
    let (mut k, reads) = cold(b"s", &all);
    assert_eq!(k.scard(b"s").expect("counted"), 400);
    let first = reads.get();
    assert!(first > 0, "the body was on the file");
    // Every command after the first is answered out of the slab. A collection
    // does not have a served path: it is in memory or it is not readable, so one
    // fault is the whole cost and there is not a second one.
    for _ in 0..50 {
        k.scard(b"s").expect("counted");
        k.sismember(b"s", b"member:00007").expect("asked");
    }
    assert_eq!(reads.get(), first, "nothing went back to the device");
}

#[test]
fn a_demoted_set_can_be_added_to_and_the_new_member_stays() {
    let all = members(400);
    let (mut k, _) = cold(b"s", &all);
    assert_eq!(
        k.sadd(b"s", [b"a new one".as_slice()].into_iter())
            .expect("added"),
        1
    );
    assert_eq!(k.scard(b"s").expect("counted"), 401);
    assert!(k.sismember(b"s", b"a new one").expect("asked"));
    assert!(k.sismember(b"s", b"member:00000").expect("asked"));
}

#[test]
fn a_demoted_set_keeps_the_word_object_encoding_answers() {
    // A table, which is what four hundred string members make.
    let (mut k, _) = cold(b"table", &members(400));
    assert_eq!(k.set_encoding(b"table"), Some(set::Encoding::Hashtable));

    // An intset, which has a Redis layout of its own and goes out as one.
    let ints: Vec<Vec<u8>> = (0..300i64).map(|i| i.to_string().into_bytes()).collect();
    let (mut k, _) = cold(b"ints", &ints);
    assert_eq!(k.set_encoding(b"ints"), Some(set::Encoding::Intset));
    assert_eq!(k.scard(b"ints").expect("counted"), 300);

    // An intset past `set-max-intset-entries`, which answers `hashtable` while
    // its body is still an intset. That flag rides in the frozen form's first
    // byte and losing it would change what a client sees.
    let big: Vec<Vec<u8>> = (0..1000i64).map(|i| i.to_string().into_bytes()).collect();
    let (mut k, _) = cold(b"big", &big);
    assert_eq!(k.set_encoding(b"big"), Some(set::Encoding::Hashtable));
    assert_eq!(k.scard(b"big").expect("counted"), 1000);
}

#[test]
fn a_deadline_on_a_demoted_set_costs_no_device_read() {
    let all = members(400);
    let (mut k, reads) = cold(b"s", &all);
    // Far enough out that the key is alive for the rest of this test whenever it
    // runs, which is what a deadline in milliseconds since the epoch has to be.
    assert!(k.set_expiry(b"s", Some(9_000_000_000_000)));
    assert_eq!(reads.get(), 0, "the deadline is in front of the address");
    // And the set is still there and still whole afterwards, which is the part
    // that would break if the record had been written as a slot record.
    assert_eq!(k.scard(b"s").expect("counted"), 400);
    assert!(k.sismember(b"s", b"member:00399").expect("asked"));
}

#[test]
fn deleting_a_demoted_set_does_not_free_somebody_elses_slab_slot() {
    let (mut k, _) = db();
    k.sadd(b"cold", members(400).iter().map(Vec::as_slice))
        .expect("added");
    k.sadd(b"warm", members(400).iter().map(Vec::as_slice))
        .expect("added");
    assert!(k.demote(b"cold").expect("demoted"));
    // The four bytes where a slot number would be are the front of an address,
    // and freeing what they read as would take the other set's slot.
    k.del(b"cold");
    assert_eq!(k.scard(b"warm").expect("counted"), 400);
    assert!(k.sismember(b"warm", b"member:00042").expect("asked"));
    assert_eq!(k.scard(b"cold").expect("counted"), 0);
}

#[test]
fn renaming_and_copying_a_demoted_set_brings_the_whole_set_with_it() {
    let all = members(400);
    let (mut k, _) = cold(b"s", &all);
    k.rename(b"s", b"moved", false);
    assert_eq!(k.scard(b"moved").expect("counted"), 400);
    assert_eq!(k.scard(b"s").expect("counted"), 0);

    let (mut k, _) = cold(b"s", &all);
    k.copy(b"s", b"twin", false);
    assert_eq!(k.scard(b"twin").expect("counted"), 400);
    assert_eq!(k.scard(b"s").expect("counted"), 400);
    // Two sets now, and adding to one must not show up in the other.
    k.sadd(b"twin", [b"only here".as_slice()].into_iter())
        .expect("added");
    assert!(k.sismember(b"twin", b"only here").expect("asked"));
    assert!(!k.sismember(b"s", b"only here").expect("asked"));
}

#[test]
fn dumping_and_restoring_a_demoted_set_gives_the_same_set_back() {
    let all = members(400);
    let (mut k, _) = cold(b"s", &all);
    let blob = k.dump(b"s").expect("dumped");
    assert!(k.restore(b"back", &blob, None, false).is_ok());
    assert_eq!(k.scard(b"back").expect("counted"), 400);
    for m in &all {
        assert!(k.sismember(b"back", m).expect("asked"), "{m:?} is in it");
    }
}

#[test]
fn a_sweep_moves_collection_bodies_and_the_memory_goes_down() {
    let (mut k, _) = db();
    for i in 0..200 {
        let key = format!("set:{i}");
        k.sadd(key.as_bytes(), members(200).iter().map(Vec::as_slice))
            .expect("added");
    }
    let before = k.memory_bytes();
    let relief = k.relieve(usize::MAX).expect("swept");
    assert!(relief.moved > 0, "something moved");
    let after = k.memory_bytes();
    // The arena grows on this path, because a twenty byte pointer replaces an
    // eight byte slot number. It is the sum over the arena and the slabs that
    // goes down, which is what the sweep is measured against and is the reason
    // its loop is in the keyspace rather than in the tier.
    assert!(after < before, "{after} is not under {before}");

    // And every set is still readable, in full, afterwards.
    for i in 0..200 {
        let key = format!("set:{i}");
        assert_eq!(k.scard(key.as_bytes()).expect("counted"), 200, "{key}");
    }
}

#[test]
fn a_sweep_over_a_keyspace_of_tiny_sets_still_ends() {
    let (mut k, _) = db();
    for i in 0..500 {
        let key = format!("s:{i}");
        k.sadd(key.as_bytes(), [b"one".as_slice()].into_iter())
            .expect("added");
    }
    // Nothing here is worth moving twice, and the sweep has to notice that and
    // stop rather than demote every set and then look for more.
    let relief = k.relieve(usize::MAX).expect("swept");
    let _ = relief;
    for i in 0..500 {
        let key = format!("s:{i}");
        assert_eq!(k.scard(key.as_bytes()).expect("counted"), 1, "{key}");
    }
}
