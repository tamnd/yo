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

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
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

/// A set of `n` members under `s`, for the rows that empty one.
fn set_of(n: usize) -> yo_kv::Keyspace {
    let mut store = yo_kv::Keyspace::new();
    let members: Vec<String> = (0..n).map(|i| format!("m{i}")).collect();
    store
        .sadd(b"s", members.iter().map(|m| m.as_bytes()))
        .expect("a set");
    store
}

/// [`set_of`], with a keyspace around it.
///
/// The difference matters more than it looks. A store holding one key has its
/// whole index in L1, so looking that key up costs nothing and a row measured
/// there says nothing about what a lookup costs. Put the same set in a database
/// with a hundred thousand other keys in it and the bucket is a cache miss,
/// which is what a lookup actually is on any real keyspace and what the memo in
/// `Keyspace` exists to skip.
fn crowded_set_of(n: usize, others: usize) -> yo_kv::Keyspace {
    let mut store = set_of(n);
    for i in 0..others {
        let k = key(i);
        store
            .set_plain(k.as_bytes(), k.as_bytes())
            .expect("room for a record");
    }
    store
}

/// The same two, through the embedded set API.
///
/// `set_of` and `crowded_set_of` built by hand, so the `api/` rows measure the
/// same database the `store/` rows do and the only difference between the two
/// numbers is the handle, the borrow and the `AsRef`.
fn api_set_of(n: usize, others: usize) -> (yo::Db, yo::Set) {
    let db = yo::open(MEMORY).expect("in memory always opens");
    let s = db.set("s");
    let members: Vec<String> = (0..n).map(|i| format!("m{i}")).collect();
    s.add_many(&members).expect("a set");
    let keys = db.strings();
    for i in 0..others {
        let k = key(i);
        keys.set(&k, &k).expect("room for a record");
    }
    (db, s)
}

/// The two set shapes on the hot key gate row, `SADD` onto one key and `SPOP`
/// off it.
///
/// `sadd/hot` is a single member onto one key over and over, which is the shape
/// with no spread to exploit and the one aki came in at 0.82x on. The set is
/// held at a thousand members so the row measures the lookup and the membership
/// check rather than the growth of an ever larger table.
///
/// The two `spop` rows are the two draws: the copying one an embedded caller
/// gets, which builds a `Vec` per member, and the borrowing one the wire takes,
/// which writes each member where the caller wants it and allocates nothing.
/// The gap between them is what `SPOP` used to pay on every reply.
fn bench_sets(c: &mut Criterion) {
    let mut g = c.benchmark_group("sadd");
    g.throughput(Throughput::Elements(1));

    g.bench_function("store/hot", |b| {
        let mut store = set_of(1_000);
        // Built up front, because a `format!` inside the timed loop would be an
        // allocation a call and this row exists to measure the ones inside.
        let members: Vec<String> = (0..1_024).map(|i| format!("m{i}")).collect();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) & 1_023;
            black_box(
                store
                    .sadd(b"s", std::iter::once(members[i].as_bytes()))
                    .expect("a set"),
            )
        });
    });

    // The same row with a hundred thousand other keys in the database, which is
    // where a lookup stops being free. Everything between the two numbers is
    // the probe the memo skips.
    g.bench_function("store/hot-crowded", |b| {
        let mut store = crowded_set_of(1_000, 100_000);
        let members: Vec<String> = (0..1_024).map(|i| format!("m{i}")).collect();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) & 1_023;
            black_box(
                store
                    .sadd(b"s", std::iter::once(members[i].as_bytes()))
                    .expect("a set"),
            )
        });
    });

    // The same two rows through `db.set("s")`. The gap against the `store/`
    // rows above is what the embedded API costs, and it is meant to be nothing.
    g.bench_function("api/hot", |b| {
        let (_db, s) = api_set_of(1_000, 0);
        let members: Vec<String> = (0..1_024).map(|i| format!("m{i}")).collect();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) & 1_023;
            black_box(s.add(&members[i]).expect("a set"))
        });
    });

    g.bench_function("api/hot-crowded", |b| {
        let (_db, s) = api_set_of(1_000, 100_000);
        let members: Vec<String> = (0..1_024).map(|i| format!("m{i}")).collect();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) & 1_023;
            black_box(s.add(&members[i]).expect("a set"))
        });
    });

    // Membership, which is the read half of the hot key shape and the one an
    // embedded caller reaches for most.
    g.bench_function("api/contains", |b| {
        let (_db, s) = api_set_of(1_000, 100_000);
        let members: Vec<String> = (0..1_024).map(|i| format!("m{i}")).collect();
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) & 1_023;
            black_box(s.contains(&members[i]).expect("a set"))
        });
    });
    g.finish();

    // A thousand members a call, so the setup that refills the set is not what
    // is being timed.
    const POP: usize = 1_000;
    let mut g = c.benchmark_group("spop");
    g.throughput(Throughput::Elements(POP as u64));

    g.bench_function("store/copy", |b| {
        b.iter_batched_ref(
            || set_of(POP),
            |store| black_box(store.spop_n(b"s", POP).expect("a set").len()),
            BatchSize::LargeInput,
        );
    });

    g.bench_function("store/into", |b| {
        b.iter_batched_ref(
            || set_of(POP),
            |store| {
                let mut bytes = 0;
                store
                    .spop_into(b"s", POP, |m| bytes += m.byte_len())
                    .expect("a set");
                black_box(bytes)
            },
            BatchSize::LargeInput,
        );
    });

    // The embedded draw, which is the copying one plus a `Vec` of them. There
    // is no borrowing form of this on the API and there should not be: a pop
    // hands back the member it just took out of the structure that was holding
    // it, so there is nothing left to borrow from.
    g.bench_function("api/pop_n", |b| {
        b.iter_batched_ref(
            || api_set_of(POP, 0),
            |(_db, s)| black_box(s.pop_n(POP).expect("a set").len()),
            BatchSize::LargeInput,
        );
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
    bench_sets,
    bench_clock
);
criterion_main!(benches);
