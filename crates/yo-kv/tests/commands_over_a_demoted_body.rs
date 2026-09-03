//! Collection commands against a key whose body is not in memory.
//!
//! The same promise `commands_over_a_demoted_value.rs` makes about strings, for
//! the types whose body lives in a slab rather than in the record. Nothing above
//! the keyspace can tell that a set was on the device a moment ago: it answers
//! `SCARD`, `SISMEMBER`, `SADD`, `SMEMBERS` and the set algebra with the members
//! it had, in the representation it had them in, the hash answers `HGET`, `HLEN`
//! and the `HEXPIRE` family the same way, the list answers `LLEN`, `LINDEX` and
//! a push at either end, the sorted set answers `ZSCORE` and `ZRANK` with the
//! order it went out in, and the array answers `ARGET` and `ARINFO` with the
//! values and the shape it had.
//!
//! The store is a vector that counts its reads, as in the string file and for
//! the same reason: reads per command is what G9 is a gate on, and a set that
//! came back in one read is the number this milestone is about.

use std::cell::Cell;
use std::rc::Rc;

use yo_common::{Addr, Code, Error, Result, Space};
use yo_kv::cold::Blocks;
use yo_kv::ttl::{Ask, Cond};
use yo_kv::zsets::{Query, ZAdd};
use yo_kv::{End, Keyspace, array, hash, list, set, zset};

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

/// Field and value pairs, past the listpack band and into a table.
fn fields(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..n)
        .map(|i| {
            (
                format!("field:{i:05}").into_bytes(),
                format!("value:{i:05}").into_bytes(),
            )
        })
        .collect()
}

/// A database holding `key` as a hash with its body already on the file.
fn cold_hash(key: &[u8], of: &[(Vec<u8>, Vec<u8>)]) -> (Keyspace, Rc<Cell<usize>>) {
    let (mut k, reads) = db();
    k.hset(key, of.iter().map(|(f, v)| (f.as_slice(), v.as_slice())))
        .expect("set");
    assert!(k.demote(key).expect("demoted"), "the body should have gone");
    reads.set(0);
    (k, reads)
}

/// Elements long enough that `n` of them are a ring of chunks and not one blob.
fn elements(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| format!("element:{i:05}:{}", "p".repeat(400)).into_bytes())
        .collect()
}

/// A database holding `key` as a list with its body already on the file.
fn cold_list(key: &[u8], of: &[Vec<u8>]) -> (Keyspace, Rc<Cell<usize>>) {
    let (mut k, reads) = db();
    k.push(key, End::Right, of.iter().map(Vec::as_slice))
        .expect("pushed");
    assert!(k.demote(key).expect("demoted"), "the body should have gone");
    reads.set(0);
    (k, reads)
}

/// What `LINDEX` answers, as bytes.
fn lindex(k: &mut Keyspace, key: &[u8], at: i64) -> Option<Vec<u8>> {
    k.lindex(key, at).expect("asked").map(|e| e.to_vec())
}

/// Members and the scores they go in with, one score a member so the order is
/// the order the names are in.
fn scored(n: usize) -> Vec<(f64, Vec<u8>)> {
    (0..n)
        .map(|i| (i as f64, format!("member:{i:05}").into_bytes()))
        .collect()
}

/// A database holding `key` as a sorted set with its body already on the file.
fn cold_zset(key: &[u8], of: &[(f64, Vec<u8>)]) -> (Keyspace, Rc<Cell<usize>>) {
    let (mut k, reads) = db();
    k.zadd(
        key,
        of.iter().map(|(s, m)| (*s, m.as_slice())),
        ZAdd::default(),
    )
    .expect("added");
    assert!(k.demote(key).expect("demoted"), "the body should have gone");
    reads.set(0);
    (k, reads)
}

/// Every member a query covers, in the order the walk gives them.
fn zrange(k: &mut Keyspace, key: &[u8], q: &Query<'_>) -> Vec<Vec<u8>> {
    let w = k.zwindow(key, q).expect("windowed");
    let mut out = Vec::new();
    k.zwalk(key, w, |m, _| {
        let mut bytes = Vec::new();
        m.write_to(&mut bytes);
        out.push(bytes);
    })
    .expect("walked");
    out
}

/// A database holding `key` as an array with `n` values in it, one every
/// seventh index so the slices are sparse, already on the file.
fn cold_array(key: &[u8], n: u64) -> (Keyspace, Rc<Cell<usize>>) {
    let (mut k, reads) = db();
    for i in 0..n {
        let val = format!("value:{i:05} and long enough to reach the blob").into_bytes();
        k.arset(key, i * 7, [val.as_slice()].into_iter())
            .expect("set");
    }
    assert!(k.demote(key).expect("demoted"), "the body should have gone");
    reads.set(0);
    (k, reads)
}

/// What `ARGET` answers, as bytes.
fn arget(k: &mut Keyspace, key: &[u8], at: u64) -> Option<Vec<u8>> {
    let mut got = None;
    k.arget_into(key, [at].into_iter(), |e| {
        got = e.map(|v| {
            let mut buf = [0u8; array::ELEMENT_MAX];
            v.text(&mut buf).to_vec()
        });
    })
    .expect("asked");
    got
}

/// What `HGET` answers, as bytes.
fn hget(k: &mut Keyspace, key: &[u8], field: &[u8]) -> Option<Vec<u8>> {
    k.hget(key, field, |v| v.map(|t| t.to_vec()))
        .expect("asked")
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
fn a_demoted_hash_answers_hlen_and_hget_with_what_it_held() {
    let all = fields(600);
    let (mut k, _) = cold_hash(b"h", &all);
    assert_eq!(k.hlen(b"h").expect("counted"), 600);
    for (f, v) in &all {
        assert_eq!(hget(&mut k, b"h", f).as_ref(), Some(v), "{f:?}");
    }
    assert_eq!(hget(&mut k, b"h", b"field:99999"), None);
}

#[test]
fn bringing_a_hash_back_costs_one_pass_and_then_nothing() {
    let all = fields(600);
    let (mut k, reads) = cold_hash(b"h", &all);
    assert_eq!(k.hlen(b"h").expect("counted"), 600);
    let first = reads.get();
    assert!(first > 0, "the body was on the file");
    for _ in 0..50 {
        k.hlen(b"h").expect("counted");
        hget(&mut k, b"h", b"field:00007");
    }
    assert_eq!(reads.get(), first, "nothing went back to the device");
}

#[test]
fn a_demoted_hash_can_be_written_to_and_the_new_field_stays() {
    let all = fields(600);
    let (mut k, _) = cold_hash(b"h", &all);
    assert_eq!(
        k.hset(b"h", [(b"fresh".as_slice(), b"one".as_slice())].into_iter())
            .expect("set"),
        1
    );
    assert_eq!(k.hlen(b"h").expect("counted"), 601);
    assert_eq!(hget(&mut k, b"h", b"fresh"), Some(b"one".to_vec()));
    assert_eq!(
        hget(&mut k, b"h", b"field:00000"),
        Some(b"value:00000".to_vec())
    );
}

#[test]
fn a_demoted_hash_keeps_the_word_object_encoding_answers() {
    let (mut k, _) = cold_hash(b"table", &fields(600));
    assert_eq!(k.hash_encoding(b"table"), Some(hash::Encoding::Hashtable));

    // A hash small enough to still be a listpack, which has to be demotable and
    // has to come back on the band it left.
    let (mut k, _) = cold_hash(b"small", &fields(60));
    assert_eq!(k.hash_encoding(b"small"), Some(hash::Encoding::Listpack));
    assert_eq!(k.hlen(b"small").expect("counted"), 60);
}

#[test]
fn a_field_deadline_survives_a_demoted_hash_coming_back() {
    let all = fields(600);
    let (mut k, _) = cold_hash(b"h", &all);
    let mut got = Vec::new();
    k.hexpire(
        b"h",
        9_000_000_000_000,
        Cond::Always,
        [b"field:00003".as_slice()].into_iter(),
        |a| got.push(a),
    )
    .expect("set a deadline");
    assert_eq!(got.len(), 1);

    // Out again, so the deadline has to go through the frozen form and back.
    assert!(k.demote(b"h").expect("demoted"));
    let mut asked = Vec::new();
    k.httl(
        b"h",
        [b"field:00003".as_slice(), b"field:00004".as_slice()].into_iter(),
        |a| asked.push(a),
    )
    .expect("asked");
    assert_eq!(asked, [Ask::At(9_000_000_000_000), Ask::NoDeadline]);
    assert_eq!(k.hlen(b"h").expect("counted"), 600, "and nothing was lost");
}

#[test]
fn a_listpack_hash_with_a_field_deadline_comes_back_widened() {
    let (mut k, _) = cold_hash(b"h", &fields(60));
    let mut got = Vec::new();
    k.hexpire(
        b"h",
        9_000_000_000_000,
        Cond::Always,
        [b"field:00003".as_slice()].into_iter(),
        |a| got.push(a),
    )
    .expect("set a deadline");
    assert_eq!(k.hash_encoding(b"h"), Some(hash::Encoding::ListpackEx));

    assert!(k.demote(b"h").expect("demoted"));
    assert_eq!(
        k.hash_encoding(b"h"),
        Some(hash::Encoding::ListpackEx),
        "widening is one way and a trip to the device is not a way back"
    );
    let mut asked = Vec::new();
    k.httl(b"h", [b"field:00003".as_slice()].into_iter(), |a| {
        asked.push(a);
    })
    .expect("asked");
    assert_eq!(asked, [Ask::At(9_000_000_000_000)]);
}

#[test]
fn deleting_a_demoted_hash_does_not_free_somebody_elses_slab_slot() {
    let (mut k, _) = db();
    let all = fields(600);
    for name in [b"cold".as_slice(), b"warm".as_slice()] {
        k.hset(name, all.iter().map(|(f, v)| (f.as_slice(), v.as_slice())))
            .expect("set");
    }
    assert!(k.demote(b"cold").expect("demoted"));
    k.del(b"cold");
    assert_eq!(k.hlen(b"warm").expect("counted"), 600);
    assert_eq!(
        hget(&mut k, b"warm", b"field:00042"),
        Some(b"value:00042".to_vec())
    );
    assert_eq!(k.hlen(b"cold").expect("counted"), 0);
}

#[test]
fn dumping_and_restoring_a_demoted_hash_gives_the_same_hash_back() {
    let all = fields(600);
    let (mut k, _) = cold_hash(b"h", &all);
    let blob = k.dump(b"h").expect("dumped");
    assert!(k.restore(b"back", &blob, None, false).is_ok());
    assert_eq!(k.hlen(b"back").expect("counted"), 600);
    for (f, v) in &all {
        assert_eq!(hget(&mut k, b"back", f).as_ref(), Some(v), "{f:?}");
    }
}

#[test]
fn a_demoted_list_answers_llen_and_lindex_with_what_it_held() {
    let all = elements(200);
    let (mut k, _) = cold_list(b"l", &all);
    assert_eq!(k.llen(b"l").expect("counted"), 200);
    assert_eq!(lindex(&mut k, b"l", 0).as_ref(), Some(&all[0]));
    assert_eq!(lindex(&mut k, b"l", 199).as_ref(), Some(&all[199]));
    assert_eq!(lindex(&mut k, b"l", -1).as_ref(), Some(&all[199]));
    assert_eq!(lindex(&mut k, b"l", 200), None);
    let got: Vec<Vec<u8>> = k
        .lrange(b"l", 0, -1)
        .expect("ranged")
        .map(|e| e.to_vec())
        .collect();
    assert_eq!(got, all, "and in the order they went in");
}

#[test]
fn bringing_a_list_back_costs_one_pass_and_then_nothing() {
    let all = elements(200);
    let (mut k, reads) = cold_list(b"l", &all);
    assert_eq!(k.llen(b"l").expect("counted"), 200);
    let first = reads.get();
    assert!(first > 0, "the body was on the file");
    for _ in 0..50 {
        k.llen(b"l").expect("counted");
        lindex(&mut k, b"l", 7);
    }
    assert_eq!(reads.get(), first, "nothing went back to the device");
}

#[test]
fn a_demoted_list_takes_elements_at_both_ends_again() {
    let all = elements(200);
    let (mut k, _) = cold_list(b"l", &all);
    k.push(b"l", End::Left, [b"first".as_slice()].into_iter())
        .expect("pushed");
    k.push(b"l", End::Right, [b"last".as_slice()].into_iter())
        .expect("pushed");
    assert_eq!(k.llen(b"l").expect("counted"), 202);
    assert_eq!(lindex(&mut k, b"l", 0), Some(b"first".to_vec()));
    assert_eq!(lindex(&mut k, b"l", -1), Some(b"last".to_vec()));
    assert_eq!(lindex(&mut k, b"l", 1).as_ref(), Some(&all[0]));

    assert_eq!(
        k.pop(b"l", End::Left).expect("popped"),
        Some(b"first".to_vec())
    );
    assert_eq!(
        k.pop(b"l", End::Right).expect("popped"),
        Some(b"last".to_vec())
    );
    assert_eq!(k.llen(b"l").expect("counted"), 200);
}

#[test]
fn a_demoted_list_keeps_the_word_object_encoding_answers() {
    let (mut k, _) = cold_list(b"ring", &elements(200));
    assert_eq!(k.list_encoding(b"ring"), Some(list::Encoding::Quicklist));

    // A list small enough to still be one blob, which has to come back one.
    let short: Vec<Vec<u8>> = (0..5).map(|i| format!("e{i}").into_bytes()).collect();
    let (mut k, _) = cold_list(b"blob", &short);
    assert_eq!(k.list_encoding(b"blob"), Some(list::Encoding::Listpack));
    assert_eq!(k.llen(b"blob").expect("counted"), 5);
}

#[test]
fn deleting_a_demoted_list_does_not_free_somebody_elses_slab_slot() {
    let (mut k, _) = db();
    let all = elements(200);
    for name in [b"cold".as_slice(), b"warm".as_slice()] {
        k.push(name, End::Right, all.iter().map(Vec::as_slice))
            .expect("pushed");
    }
    assert!(k.demote(b"cold").expect("demoted"));
    k.del(b"cold");
    assert_eq!(k.llen(b"warm").expect("counted"), 200);
    assert_eq!(lindex(&mut k, b"warm", 42).as_ref(), Some(&all[42]));
    assert_eq!(k.llen(b"cold").expect("counted"), 0);
}

#[test]
fn dumping_and_restoring_a_demoted_list_gives_the_same_list_back() {
    let all = elements(200);
    let (mut k, _) = cold_list(b"l", &all);
    let blob = k.dump(b"l").expect("dumped");
    assert!(k.restore(b"back", &blob, None, false).is_ok());
    assert_eq!(k.llen(b"back").expect("counted"), 200);
    let got: Vec<Vec<u8>> = k
        .lrange(b"back", 0, -1)
        .expect("ranged")
        .map(|e| e.to_vec())
        .collect();
    assert_eq!(got, all);
}

#[test]
fn a_demoted_sorted_set_answers_zcard_zscore_and_zrank_with_what_it_held() {
    let all = scored(400);
    let (mut k, _) = cold_zset(b"z", &all);
    assert_eq!(k.zcard(b"z").expect("counted"), 400);
    assert_eq!(k.zscore(b"z", b"member:00000").expect("asked"), Some(0.0));
    assert_eq!(k.zscore(b"z", b"member:00399").expect("asked"), Some(399.0));
    assert_eq!(k.zscore(b"z", b"nobody").expect("asked"), None);
    assert_eq!(
        k.zrank(b"z", b"member:00042", false).expect("ranked"),
        Some((42, 42.0))
    );
    assert_eq!(
        k.zrank(b"z", b"member:00042", true).expect("ranked"),
        Some((357, 42.0))
    );
    let got = zrange(&mut k, b"z", &Query::rank(0, -1));
    let want: Vec<Vec<u8>> = all.iter().map(|(_, m)| m.clone()).collect();
    assert_eq!(got, want, "and in the order the scores put them in");
}

#[test]
fn bringing_a_sorted_set_back_costs_one_pass_and_then_nothing() {
    let all = scored(400);
    let (mut k, reads) = cold_zset(b"z", &all);
    assert_eq!(k.zcard(b"z").expect("counted"), 400);
    let first = reads.get();
    assert!(first > 0, "the body was on the file");
    for _ in 0..50 {
        k.zcard(b"z").expect("counted");
        k.zscore(b"z", b"member:00007").expect("asked");
        k.zrank(b"z", b"member:00007", false).expect("ranked");
    }
    assert_eq!(reads.get(), first, "nothing went back to the device");
}

#[test]
fn a_demoted_sorted_set_can_be_added_to_and_the_ranks_still_come_out_right() {
    let all = scored(400);
    let (mut k, _) = cold_zset(b"z", &all);
    assert_eq!(
        k.zadd(
            b"z",
            [(-1.0, b"first".as_slice()), (1000.0, b"last".as_slice())].into_iter(),
            ZAdd::default(),
        )
        .expect("added"),
        2
    );
    assert_eq!(k.zcard(b"z").expect("counted"), 402);
    assert_eq!(
        k.zrank(b"z", b"first", false).expect("ranked"),
        Some((0, -1.0))
    );
    assert_eq!(
        k.zrank(b"z", b"last", false).expect("ranked"),
        Some((401, 1000.0))
    );
    assert_eq!(
        k.zrank(b"z", b"member:00000", false).expect("ranked"),
        Some((1, 0.0))
    );
    // And the score of a member that was already in there still moves it.
    k.zadd(
        b"z",
        [(500.0, b"member:00000".as_slice())].into_iter(),
        ZAdd::default(),
    )
    .expect("moved");
    assert_eq!(
        k.zrank(b"z", b"member:00000", false).expect("ranked"),
        Some((400, 500.0))
    );
}

#[test]
fn a_demoted_sorted_set_keeps_the_word_object_encoding_answers() {
    // The table, which is what four hundred members make.
    let (mut k, _) = cold_zset(b"table", &scored(400));
    assert_eq!(k.zset_encoding(b"table"), Some(zset::Encoding::Skiplist));

    // And one small enough to still be one packed blob, which has to come back
    // one, because the word is a property of the body and not of the record.
    let (mut k, _) = cold_zset(b"blob", &scored(40));
    assert_eq!(k.zset_encoding(b"blob"), Some(zset::Encoding::Listpack));
    assert_eq!(k.zcard(b"blob").expect("counted"), 40);
    assert_eq!(
        k.zscore(b"blob", b"member:00039").expect("asked"),
        Some(39.0)
    );
}

#[test]
fn deleting_a_demoted_sorted_set_does_not_free_somebody_elses_slab_slot() {
    let (mut k, _) = db();
    let all = scored(400);
    for name in [b"cold".as_slice(), b"warm".as_slice()] {
        k.zadd(
            name,
            all.iter().map(|(s, m)| (*s, m.as_slice())),
            ZAdd::default(),
        )
        .expect("added");
    }
    assert!(k.demote(b"cold").expect("demoted"));
    k.del(b"cold");
    assert_eq!(k.zcard(b"warm").expect("counted"), 400);
    assert_eq!(
        k.zrank(b"warm", b"member:00042", false).expect("ranked"),
        Some((42, 42.0))
    );
    assert_eq!(k.zcard(b"cold").expect("counted"), 0);
}

#[test]
fn dumping_and_restoring_a_demoted_sorted_set_gives_the_same_one_back() {
    let all = scored(400);
    let (mut k, _) = cold_zset(b"z", &all);
    let blob = k.dump(b"z").expect("dumped");
    assert!(k.restore(b"back", &blob, None, false).is_ok());
    assert_eq!(k.zcard(b"back").expect("counted"), 400);
    let got = zrange(&mut k, b"back", &Query::rank(0, -1));
    let want: Vec<Vec<u8>> = all.iter().map(|(_, m)| m.clone()).collect();
    assert_eq!(got, want);
    assert_eq!(
        k.zscore(b"back", b"member:00399").expect("asked"),
        Some(399.0)
    );
}

#[test]
fn a_demoted_array_answers_arlen_arcount_and_arget_with_what_it_held() {
    let (mut k, _) = cold_array(b"a", 300);
    assert_eq!(k.arcount(b"a").expect("counted"), 300);
    assert_eq!(k.arlen(b"a").expect("measured"), 299 * 7 + 1);
    for i in [0u64, 1, 42, 299] {
        assert_eq!(
            arget(&mut k, b"a", i * 7),
            Some(format!("value:{i:05} and long enough to reach the blob").into_bytes()),
            "index {i}"
        );
    }
    assert_eq!(arget(&mut k, b"a", 1), None, "and a hole is still a hole");
}

#[test]
fn bringing_an_array_back_costs_one_pass_and_then_nothing() {
    let (mut k, reads) = cold_array(b"a", 300);
    assert_eq!(k.arcount(b"a").expect("counted"), 300);
    let first = reads.get();
    assert!(first > 0, "the body was on the file");
    for _ in 0..50 {
        k.arcount(b"a").expect("counted");
        arget(&mut k, b"a", 49);
    }
    assert_eq!(reads.get(), first, "nothing went back to the device");
}

#[test]
fn a_demoted_array_can_be_written_to_and_the_new_value_stays() {
    let (mut k, _) = cold_array(b"a", 300);
    assert_eq!(
        k.arset(b"a", 1, [b"in a hole".as_slice()].into_iter())
            .expect("set"),
        1
    );
    assert_eq!(
        k.arset(b"a", 0, [b"over the top".as_slice()].into_iter())
            .expect("set"),
        0,
        "an overwrite does not fill a new position"
    );
    assert_eq!(k.arcount(b"a").expect("counted"), 301);
    assert_eq!(arget(&mut k, b"a", 1), Some(b"in a hole".to_vec()));
    assert_eq!(arget(&mut k, b"a", 0), Some(b"over the top".to_vec()));
    assert_eq!(
        arget(&mut k, b"a", 7),
        Some(b"value:00001 and long enough to reach the blob".to_vec())
    );
}

#[test]
fn a_demoted_array_keeps_the_shape_arinfo_reports() {
    let (mut k, _) = db();
    // Consecutive indices, which is what makes a slice dense.
    for i in 0..300u64 {
        k.arset(b"a", i, [b"x".as_slice()].into_iter())
            .expect("set");
    }
    let was = k.arinfo(b"a", true).expect("asked");
    assert!(was.dense_slices > 0, "consecutive writes make a window");
    assert!(k.demote(b"a").expect("demoted"));
    let now = k.arinfo(b"a", true).expect("asked");
    assert_eq!(now.count, was.count);
    assert_eq!(now.len, was.len);
    assert_eq!(now.slices, was.slices);
    assert_eq!(
        now.dense_slices, was.dense_slices,
        "and it is still a window"
    );
    assert_eq!(now.sparse_slices, was.sparse_slices);
}

#[test]
fn deleting_a_demoted_array_does_not_free_somebody_elses_slab_slot() {
    let (mut k, _) = cold_array(b"cold", 300);
    for i in 0..300u64 {
        let val = format!("value:{i:05} and long enough to reach the blob").into_bytes();
        k.arset(b"warm", i * 7, [val.as_slice()].into_iter())
            .expect("set");
    }
    k.del(b"cold");
    assert_eq!(k.arcount(b"warm").expect("counted"), 300);
    assert_eq!(
        arget(&mut k, b"warm", 294),
        Some(b"value:00042 and long enough to reach the blob".to_vec())
    );
    assert_eq!(k.arcount(b"cold").expect("counted"), 0);
}

#[test]
fn a_sweep_moves_collection_bodies_and_the_memory_goes_down() {
    let (mut k, _) = db();
    let pairs = fields(200);
    let rows = elements(60);
    let ranked = scored(200);
    for i in 0..200 {
        let key = format!("set:{i}");
        k.sadd(key.as_bytes(), members(200).iter().map(Vec::as_slice))
            .expect("added");
        let key = format!("hash:{i}");
        k.hset(
            key.as_bytes(),
            pairs.iter().map(|(f, v)| (f.as_slice(), v.as_slice())),
        )
        .expect("set");
        let key = format!("list:{i}");
        k.push(key.as_bytes(), End::Right, rows.iter().map(Vec::as_slice))
            .expect("pushed");
        let key = format!("zset:{i}");
        k.zadd(
            key.as_bytes(),
            ranked.iter().map(|(s, m)| (*s, m.as_slice())),
            ZAdd::default(),
        )
        .expect("added");
        let key = format!("array:{i}");
        for (at, (_, m)) in ranked.iter().enumerate() {
            k.arset(key.as_bytes(), at as u64 * 7, [m.as_slice()].into_iter())
                .expect("set");
        }
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

    // And every body is still readable, in full, afterwards.
    for i in 0..200 {
        let key = format!("set:{i}");
        assert_eq!(k.scard(key.as_bytes()).expect("counted"), 200, "{key}");
        let key = format!("hash:{i}");
        assert_eq!(k.hlen(key.as_bytes()).expect("counted"), 200, "{key}");
        let key = format!("list:{i}");
        assert_eq!(k.llen(key.as_bytes()).expect("counted"), 60, "{key}");
        let key = format!("zset:{i}");
        assert_eq!(k.zcard(key.as_bytes()).expect("counted"), 200, "{key}");
        let key = format!("array:{i}");
        assert_eq!(k.arcount(key.as_bytes()).expect("counted"), 200, "{key}");
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
