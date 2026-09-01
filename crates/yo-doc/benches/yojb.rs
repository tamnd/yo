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
//!   - `index/put` at zero, one and four indexes. The gap is a path lookup and a
//!     set write per index, and it should be a fixed cost per index rather than
//!     something that grows with the collection.
//!
//! # Reading these on a machine someone else is using
//!
//! Same rule as `yo-kv`'s benches: criterion's mean picks up whatever else the
//! box is doing, so the comparable number is the minimum per iteration across
//! samples, out of `target/criterion/<group>/<id>/new/sample.json`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_doc::{Builder, Docs, Key, Value};

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

criterion_group!(
    benches,
    bench_get,
    bench_get_over_payload,
    bench_path,
    bench_build,
    bench_validate,
    bench_docs,
    bench_index
);
criterion_main!(benches);
