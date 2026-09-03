//! What reaching one field of a document costs, and what putting one together
//! costs.
//!
//! The claim the encoding is built on is that a path read is a few header reads
//! and a seek, and that the size of the document does not come into it. That is
//! the claim G15 turns into a gate: an indexed path equality lookup in the same
//! cost class as `HGET`.
//!
//! The rows to watch:
//!
//!   - `get` against the member count. A binary search is logarithmic, so this
//!     should climb slowly and never step.
//!   - `get` against the payload size at a fixed member count. This should be
//!     flat, because a lookup reads the entry table and the keys and never the
//!     value region.
//!   - `path` at four levels against a one level `get`. Four times a bit more
//!     than one, and no allocation either way.
//!   - `index/find` against `index/scan` at both collection sizes. `find` costs
//!     the same per matching document whatever the collection holds and `scan`
//!     has to read all of it, so the gap between the two rows should widen by
//!     the same factor the collection grows by.
//!   - `index/range` against `index/range_scan`, the same shape one level up.
//!     The window is 256 documents at both collection sizes, so `range` should
//!     be the same number twice and the scan should not be.
//!   - `index/word` and `index/tag` against `index/word_scan`. Filing a document
//!     under eight keys instead of one does not change the shape: the probe
//!     reads what matches and the scan reads everything.
//!   - `index/put` at zero, one and four indexes. The gap is a path lookup and a
//!     set write per index, and it should be a fixed cost per index rather than
//!     something that grows with the collection.
//!   - `index/put_equality` against `index/put_array` and `index/put_text`. The
//!     gap there is what filing eight keys costs over filing one, so it should
//!     be about eight set writes and not more.
//!   - `hget/find` against `hget/hget`, which is the gate itself. Both are the
//!     same records at the same size and the ratio between them is the answer.
//!     `hget/count` is the same probe without reading the document, so the two
//!     halves of `find` can be told apart.
//!
//! # Reading these on a machine someone else is using
//!
//! Same rule as `yo-kv`'s benches: criterion's mean picks up whatever else the
//! box is doing, so the comparable number is the minimum per iteration across
//! samples, out of `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::ops::Bound;
use yo_doc::{Builder, Docs, IndexKind, Key, Value};
use yo_kv::{Hash, HashLimits};

/// An object of `n` members, each holding a string of `pad` bytes.
fn object(n: usize, pad: usize) -> (Vec<u8>, Vec<String>) {
    let names: Vec<String> = (0..n).map(|i| format!("field{i:04}")).collect();
    let filler = "x".repeat(pad);
    let mut b = Builder::new();
    b.begin_object().expect("open");
    for name in &names {
        b.key(name.as_bytes()).expect("key");
        b.text(&filler).expect("value");
    }
    b.end_object().expect("close");
    (b.finish().expect("finished").to_vec(), names)
}

/// A lookup against the number of members, at a payload small enough that the
/// whole document is in cache.
fn bench_get(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/get");
    for n in [4usize, 16, 64, 256, 1024] {
        let (bytes, names) = object(n, 8);
        let v = Value::new(&bytes).expect("readable");
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(v.get(black_box(names[i].as_bytes())))
            });
        });
    }
    g.finish();
}

/// The same lookup against the size of the values, at a fixed member count.
///
/// The whole point of an entry table with a header copy in it is that this row
/// is flat. If it climbs, a lookup is touching the value region and the layout
/// is wrong.
fn bench_get_over_payload(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/get_padded");
    for pad in [8usize, 64, 512, 4096] {
        let (bytes, names) = object(32, pad);
        let v = Value::new(&bytes).expect("readable");
        g.bench_with_input(BenchmarkId::from_parameter(pad), &pad, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % names.len();
                black_box(v.get(black_box(names[i].as_bytes())))
            });
        });
    }
    g.finish();
}

/// A four level path against the one level lookup it is made of.
fn bench_path(c: &mut Criterion) {
    let mut b = Builder::new();
    b.begin_object().expect("open");
    b.key(b"order").expect("key");
    b.begin_object().expect("open");
    b.key(b"lines").expect("key");
    b.begin_array().expect("open");
    for i in 0..16i64 {
        b.begin_object().expect("open");
        b.key(b"sku").expect("key");
        b.int(i).expect("value");
        b.key(b"note").expect("key");
        b.text("a line with enough text on it to move the offsets about")
            .expect("value");
        b.end_object().expect("close");
    }
    b.end_array().expect("close");
    b.end_object().expect("close");
    b.key(b"id").expect("key");
    b.int(99).expect("value");
    b.end_object().expect("close");
    let bytes = b.finish().expect("finished").to_vec();
    let v = Value::new(&bytes).expect("readable");

    let mut g = c.benchmark_group("yojb/path");
    g.bench_function("one_level", |b| {
        b.iter(|| black_box(v.path(black_box("$.id"))));
    });
    g.bench_function("four_levels", |b| {
        b.iter(|| black_box(v.path(black_box("$.order.lines[9].sku"))));
    });
    g.finish();
}

/// Putting a document together, which is what a write costs before it reaches
/// the log.
fn bench_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/build");
    for n in [4usize, 16, 64, 256] {
        let names: Vec<String> = (0..n).map(|i| format!("field{i:04}")).collect();
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut builder = Builder::new();
            b.iter(|| {
                builder.clear();
                builder.begin_object().expect("open");
                for (i, name) in names.iter().enumerate() {
                    builder.key(name.as_bytes()).expect("key");
                    builder.int(i as i64).expect("value");
                }
                builder.end_object().expect("close");
                black_box(builder.finish().expect("finished").len())
            });
        });
    }
    g.finish();
}

/// Checking a whole document, which is what a read of bytes nobody trusts
/// costs.
fn bench_validate(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/validate");
    for n in [16usize, 256] {
        let (bytes, _) = object(n, 16);
        let v = Value::new(&bytes).expect("readable");
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(v.validate()));
        });
    }
    g.finish();
}

/// Storing a document and reading one back, which is the collection rather
/// than the encoding.
///
/// `put` carries the interning walk and the copy into the primary table, so it
/// is strictly more than `build` at the same field count and the gap is what
/// interning costs on the write side. `get` is a probe of the primary table and
/// then a field lookup, which is the whole of `HGET`'s shape with a document in
/// the middle of it, and `field` is that lookup on its own against the same
/// lookup on a document whose keys are bytes.
fn bench_docs(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/docs");
    for n in [4usize, 16, 64] {
        let (bytes, names) = object(n, 16);
        let ids: Vec<String> = (0..1024).map(|i| format!("doc:{i:06}")).collect();

        g.bench_with_input(BenchmarkId::new("put", n), &n, |b, _| {
            let mut docs = Docs::with_capacity(ids.len(), bytes.len());
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % ids.len();
                black_box(
                    docs.put_bytes(black_box(ids[i].as_bytes()), black_box(&bytes))
                        .expect("put"),
                )
            });
        });

        let mut docs = Docs::with_capacity(ids.len(), bytes.len());
        for id in &ids {
            docs.put_bytes(id.as_bytes(), &bytes).expect("put");
        }
        g.bench_with_input(BenchmarkId::new("get", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % ids.len();
                let d = docs.get(black_box(ids[i].as_bytes())).expect("stored");
                black_box(d.get(black_box(names[i % n].as_bytes())))
            });
        });

        let doc = docs.get(ids[0].as_bytes()).expect("stored");
        g.bench_with_input(BenchmarkId::new("field", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(doc.get(black_box(names[i].as_bytes())))
            });
        });
    }
    g.finish();
}

/// A record with four fields worth indexing and a bit of payload.
fn record(i: usize) -> Vec<u8> {
    let mut b = Builder::new();
    b.begin_object().expect("open");
    b.key(b"status").expect("key");
    b.text(&format!("s{:03}", i % 64)).expect("value");
    b.key(b"region").expect("key");
    b.text(&format!("r{:03}", i % 16)).expect("value");
    b.key(b"tier").expect("key");
    b.int((i % 4) as i64).expect("value");
    b.key(b"seq").expect("key");
    b.int(i as i64).expect("value");
    b.key(b"payload").expect("key");
    b.text(&"x".repeat(64)).expect("value");
    b.end_object().expect("close");
    b.finish().expect("finished").to_vec()
}

/// A collection of `n` records, with the first `indexes` of the four paths
/// declared before anything is written.
fn filled(n: usize, indexes: usize) -> Docs {
    let mut docs = Docs::with_capacity(n, 160);
    for path in ["$.status", "$.region", "$.tier", "$.seq"]
        .iter()
        .take(indexes)
    {
        docs.create_index(path).expect("indexed");
    }
    for i in 0..n {
        docs.put_bytes(format!("d:{i:08}").as_bytes(), &record(i))
            .expect("put");
    }
    docs
}

/// A record with eight tags and a sentence of eight words, for the kinds that
/// file a document under more than one key.
///
/// Eight out of a vocabulary of 512, in blocks, so one word lands on one
/// document in sixty four. That is the selectivity the equality rows above run
/// at, which is what makes the two comparable.
fn wordy_record(i: usize) -> Vec<u8> {
    let mut b = Builder::new();
    b.begin_object().expect("open");
    b.key(b"about").expect("key");
    let words: Vec<String> = (0..8)
        .map(|k| format!("w{:03}", (i * 8 + k) % 512))
        .collect();
    b.text(&words.join(" ")).expect("value");
    b.key(b"tags").expect("key");
    b.begin_array().expect("open");
    for k in 0..8 {
        b.text(&format!("t{:03}", (i * 8 + k) % 512))
            .expect("value");
    }
    b.end_array().expect("close");
    b.end_object().expect("close");
    b.finish().expect("finished").to_vec()
}

/// A collection of `n` wordy records with one index of `kind` over the path
/// that kind is for.
///
/// Equality goes on the sentence rather than the tags, because an equality
/// index over an array files nothing and a row that files nothing is not a
/// write path measurement. On the sentence it files the whole string as one
/// key, which is exactly the one against eight the put rows are comparing.
fn wordy(n: usize, kind: IndexKind) -> Docs {
    let mut docs = Docs::with_capacity(n, 128);
    let path: &[u8] = match kind {
        IndexKind::Array => b"$.tags",
        _ => b"$.about",
    };
    docs.create_index_bytes(path, kind).expect("indexed");
    for i in 0..n {
        docs.put_bytes(format!("d:{i:08}").as_bytes(), &wordy_record(i))
            .expect("put");
    }
    docs
}

fn bench_index(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/index");

    // The point of an index, stated as a ratio rather than as a claim: the same
    // answer, found through the index and found by walking the collection. The
    // scan reads every document and the probe reads the ones that match, so the
    // gap is whatever the selectivity is, and status is one in sixty four.
    for n in [1024usize, 16_384] {
        let docs = filled(n, 1);
        let key = Key::text("s007");
        assert_eq!(docs.count("$.status", &key).expect("indexed"), n / 64);

        g.bench_with_input(BenchmarkId::new("find", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0i64;
                docs.find("$.status", black_box(&key), |_, d| {
                    sum += d.get(b"seq").and_then(|v| v.as_int()).expect("a seq");
                })
                .expect("indexed");
                black_box(sum)
            });
        });

        g.bench_with_input(BenchmarkId::new("scan", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0i64;
                for (_, d) in docs.iter() {
                    if d.get(b"status").and_then(|v| v.as_text()) == Some("s007") {
                        sum += d.get(b"seq").and_then(|v| v.as_int()).expect("a seq");
                    }
                }
                black_box(sum)
            });
        });

        g.bench_with_input(BenchmarkId::new("count", n), &n, |b, _| {
            b.iter(|| black_box(docs.count("$.status", black_box(&key)).expect("indexed")));
        });
    }

    // A window of 256 out of the collection, through the tree and by walking.
    // `seq` is unique, so the answer is the same size at both collection sizes
    // and the `range` row should be flat while the `range_scan` row is not.
    for n in [1024usize, 16_384] {
        let mut docs = filled(n, 0);
        docs.create_ordered_index("$.seq").expect("ordered");
        let (lo, hi) = (Key::int(100), Key::int(356));
        let (lo, hi) = (Bound::Included(&lo), Bound::Excluded(&hi));
        assert_eq!(docs.count_range("$.seq", lo, hi).expect("ordered"), 256);

        g.bench_with_input(BenchmarkId::new("range", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0i64;
                docs.range("$.seq", black_box(lo), black_box(hi), |_, d| {
                    sum += d.get(b"tier").and_then(|v| v.as_int()).expect("a tier");
                })
                .expect("ordered");
                black_box(sum)
            });
        });

        g.bench_with_input(BenchmarkId::new("range_scan", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0i64;
                for (_, d) in docs.iter() {
                    let seq = d.get(b"seq").and_then(|v| v.as_int()).expect("a seq");
                    if (100..356).contains(&seq) {
                        sum += d.get(b"tier").and_then(|v| v.as_int()).expect("a tier");
                    }
                }
                black_box(sum)
            });
        });

        g.bench_with_input(BenchmarkId::new("count_range", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    docs.count_range("$.seq", black_box(lo), black_box(hi))
                        .expect("ordered"),
                )
            });
        });
    }

    // The two kinds that file a document under more than one key, against the
    // scan that gets the same answer. A tag or a word costs a posting each, so
    // the probe is flat in the collection and the scan is not.
    for n in [1024usize, 16_384] {
        let words = wordy(n, IndexKind::Text);
        let tags = wordy(n, IndexKind::Array);
        let word = Key::word("w007").expect("one word");
        let tag = Key::text("t007");
        assert_eq!(words.count("$.about", &word).expect("indexed"), n / 64);
        assert_eq!(tags.count("$.tags", &tag).expect("indexed"), n / 64);

        g.bench_with_input(BenchmarkId::new("word", n), &n, |b, _| {
            b.iter(|| {
                let mut hits = 0usize;
                words
                    .find("$.about", black_box(&word), |_, _| hits += 1)
                    .expect("indexed");
                black_box(hits)
            });
        });

        g.bench_with_input(BenchmarkId::new("word_scan", n), &n, |b, _| {
            b.iter(|| {
                let mut hits = 0usize;
                for (_, d) in words.iter() {
                    let about = d.get(b"about").and_then(|v| v.as_text()).expect("about");
                    if about.split(' ').any(|w| w == "w007") {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        g.bench_with_input(BenchmarkId::new("tag", n), &n, |b, _| {
            b.iter(|| {
                let mut hits = 0usize;
                tags.find("$.tags", black_box(&tag), |_, _| hits += 1)
                    .expect("indexed");
                black_box(hits)
            });
        });
    }

    // What each kind costs the write path. Equality is one key a document and
    // the other two are one key per element or per word, so the gap is what
    // filing more than one key costs rather than what the kind costs.
    for kind in [IndexKind::Equality, IndexKind::Array, IndexKind::Text] {
        let name = format!("put_{kind:?}").to_lowercase();
        g.bench_with_input(BenchmarkId::new(name, 8), &kind, |b, &kind| {
            let mut docs = wordy(1024, kind);
            let ids: Vec<String> = (0..1024).map(|i| format!("d:{i:08}")).collect();
            let bytes = wordy_record(7);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % ids.len();
                black_box(
                    docs.put_bytes(black_box(ids[i].as_bytes()), black_box(&bytes))
                        .expect("put"),
                )
            });
        });
    }

    // What an index costs the write path, which is one path lookup each.
    for indexes in [0usize, 1, 4] {
        g.bench_with_input(BenchmarkId::new("put", indexes), &indexes, |b, &indexes| {
            let mut docs = filled(1024, indexes);
            let ids: Vec<String> = (0..1024).map(|i| format!("d:{i:08}")).collect();
            let bytes = record(7);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % ids.len();
                black_box(
                    docs.put_bytes(black_box(ids[i].as_bytes()), black_box(&bytes))
                        .expect("put"),
                )
            });
        });
    }
    g.finish();
}

/// The same `n` records twice, once as a collection with a unique valued index
/// and once as a hash keyed by the same value.
///
/// `$.seq` is the path because it is the one no two documents share, so a probe
/// answers exactly one document and the row is a probe rather than a probe plus
/// a walk of a posting list. The hash holds the whole record as its value, so
/// both sides hand back the same bytes for the same lookup and the comparison
/// is not one side doing less work.
fn parity(n: usize) -> (Docs, Hash) {
    let mut docs = Docs::with_capacity(n, 160);
    docs.create_index("$.seq").expect("indexed");
    let mut hash = Hash::with_hint(n, &HashLimits::DEFAULT);
    for i in 0..n {
        let bytes = record(i);
        docs.put_bytes(format!("d:{i:08}").as_bytes(), &bytes)
            .expect("put");
        hash.set(format!("{i}").as_bytes(), &bytes, &HashLimits::DEFAULT);
    }
    (docs, hash)
}

/// G15 stated as a ratio: an indexed path equality against `HGET`.
///
/// The gate is that finding a document by a value at an indexed path costs what
/// getting a field out of a hash costs, because underneath it is the same
/// element table being probed once. So this runs both against the same records
/// at the same collection sizes and the number to read is `find` over `hget`.
///
/// The `find` side is doing three things the `hget` side is not, and all three
/// are on purpose because all three are what a caller actually pays. It looks
/// the path up by name, it encodes the value into an index key, and it takes
/// the id it gets back and reads the document out of the primary table. That
/// last one is the second probe, and it is the reason the honest claim here is
/// the same cost class rather than the same number.
fn bench_hget(c: &mut Criterion) {
    let mut g = c.benchmark_group("yojb/hget");
    for n in [1024usize, 16_384] {
        let (docs, hash) = parity(n);
        assert_eq!(docs.count("$.seq", &Key::int(7)).expect("indexed"), 1);
        let fields: Vec<Vec<u8>> = (0..n).map(|i| format!("{i}").into_bytes()).collect();

        g.bench_with_input(BenchmarkId::new("hget", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(hash.get(black_box(&fields[i])))
            });
        });

        g.bench_with_input(BenchmarkId::new("find", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                let mut len = 0usize;
                docs.find("$.seq", black_box(&Key::int(i as i64)), |_, d| {
                    len = d.len()
                })
                .expect("indexed");
                black_box(len)
            });
        });

        // The probe on its own, without reading the document it points at, so
        // the two probes in `find` can be told apart when the ratio moves.
        g.bench_with_input(BenchmarkId::new("count", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(
                    docs.count("$.seq", black_box(&Key::int(i as i64)))
                        .expect("indexed"),
                )
            });
        });

        // The probe again with the path already resolved, which is what a
        // prepared query would pay, and the key encoding on its own. Between
        // them and `count` there is nowhere for a nanosecond to hide.
        let index = docs.index("$.seq").expect("indexed");
        g.bench_with_input(BenchmarkId::new("probe", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(index.count(black_box(&Key::int(i as i64))))
            });
        });

        g.bench_with_input(BenchmarkId::new("key", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(Key::int(black_box(i as i64)))
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_get,
    bench_get_over_payload,
    bench_path,
    bench_build,
    bench_validate,
    bench_docs,
    bench_index,
    bench_hget
);
criterion_main!(benches);
