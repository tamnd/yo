//! What inline execution mode costs, per command on the gate list.
//!
//! `bench/00` gives a point read 150 ns in process, and the number is only
//! meaningful if it is measured where a program actually calls (Y23). So every
//! group here has a `store/` row calling [`yo_kv::Keyspace`] straight and an
//! `api/` row calling the same command through `db.strings()`. The gap between
//! them is everything the embedded API adds: the handle, the borrow, the
//! `AsRef` and the clock policy. It is meant to be nothing.
//!
//! The `clock` group is the one that pays for itself. `04` section 5 keeps the
//! clock off the data path, and inline mode has no loop to refresh it in, so a
//! keyspace reads the clock only once something in it has a deadline. Those two
//! rows are what that rule is worth.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use yo::MEMORY;

fn key(i: usize) -> String {
    format!("key:{i:012}")
}

/// Enough keys to be past the caches without making the fill the slow part.
const KEYS: usize = 100_000;

fn keys() -> usize {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        1_000
    } else {
        KEYS
    }
}

/// The same keyspace filled twice, once through each door.
fn filled(n: usize) -> (yo_kv::Keyspace, yo::Db) {
    let mut store = yo_kv::Keyspace::new();
    let db = yo::open(MEMORY).expect("in memory always opens");
    let keys = db.strings();
    for i in 0..n {
        let k = key(i);
        store
            .set_plain(k.as_bytes(), k.as_bytes())
            .expect("room for a record");
        keys.set(&k, &k).expect("room for a record");
    }
    (store, db)
}

fn bench_get(c: &mut Criterion) {
    let n = keys();
    let (mut store, db) = filled(n);
    let api = db.strings();
    let sample: Vec<String> = (0..64).map(|i| key(i * (n / 64).max(1))).collect();

    let mut g = c.benchmark_group("get");
    g.throughput(Throughput::Elements(1));

    let mut at = 0usize;
    g.bench_function("store", |b| {
        b.iter(|| {
            at = (at + 1) & 63;
            black_box(
                store
                    .get(sample[at].as_bytes())
                    .expect("a string")
                    .map(|v| v.len()),
            )
        });
    });

    g.bench_function("api/with", |b| {
        b.iter(|| {
            at = (at + 1) & 63;
            black_box(api.with(&sample[at], |v| v.len()))
        });
    });

    g.bench_function("api/get", |b| {
        b.iter(|| {
            at = (at + 1) & 63;
            black_box(api.get(&sample[at]))
        });
    });

    g.finish();
}

fn bench_set(c: &mut Criterion) {
    let n = keys();
    let sample: Vec<String> = (0..n).map(key).collect();

    let mut g = c.benchmark_group("set");
    g.throughput(Throughput::Elements(1));

    g.bench_function("store", |b| {
        let mut store = yo_kv::Keyspace::new();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % n;
            store
                .set_plain(sample[i].as_bytes(), sample[i].as_bytes())
                .expect("room");
        });
    });

    g.bench_function("api", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let keys = db.strings();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % n;
            keys.set(&sample[i], &sample[i]).expect("room");
        });
    });

    g.finish();
}

fn bench_incr(c: &mut Criterion) {
    let mut g = c.benchmark_group("incr");
    g.throughput(Throughput::Elements(1));

    g.bench_function("store", |b| {
        let mut store = yo_kv::Keyspace::new();
        b.iter(|| black_box(store.incr(b"hits").expect("a counter")));
    });

    g.bench_function("api", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let keys = db.strings();
        b.iter(|| black_box(keys.incr("hits").expect("a counter")));
    });

    // The handle that holds the key, which is what most callers will reach for.
    g.bench_function("api/counter", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let hits = db.counter("hits");
        b.iter(|| black_box(hits.incr().expect("a counter")));
    });

    g.finish();
}

fn bench_mset(c: &mut Criterion) {
    let pairs: Vec<(String, String)> = (0..8).map(|i| (key(i), key(i))).collect();

    let mut g = c.benchmark_group("mset");
    // Eight pairs a call, which is what a client pipelining an MSET sends.
    g.throughput(Throughput::Elements(pairs.len() as u64));

    g.bench_function("store", |b| {
        let mut store = yo_kv::Keyspace::new();
        b.iter(|| {
            store
                .mset(pairs.iter().map(|(k, v)| (k.as_bytes(), v.as_bytes())))
                .expect("room");
        });
    });

    g.bench_function("api", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let keys = db.strings();
        b.iter(|| keys.set_many(&pairs).expect("room"));
    });

    g.finish();
}

/// What the clock policy saves, measured rather than asserted.
///
/// Both rows are the same `GET` on the same store. The second one has had one
/// key given a deadline, which is what turns the per call refresh on for the
/// whole keyspace.
fn bench_clock(c: &mut Criterion) {
    let mut g = c.benchmark_group("clock");
    g.throughput(Throughput::Elements(1));

    g.bench_function("off", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let keys = db.strings();
        keys.set("k", "v").expect("room");
        assert!(!db.reads_the_clock());
        b.iter(|| black_box(keys.with("k", |v| v.len())));
    });

    g.bench_function("on", |b| {
        let db = yo::open(MEMORY).expect("in memory always opens");
        let keys = db.strings();
        keys.set("k", "v").expect("room");
        keys.set_for("ttl", "v", Duration::from_secs(3_600))
            .expect("room");
        assert!(db.reads_the_clock());
        b.iter(|| black_box(keys.with("k", |v| v.len())));
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_get,
    bench_set,
    bench_incr,
    bench_mset,
    bench_clock
);
criterion_main!(benches);
