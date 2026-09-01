//! What `DUMP` costs, with the blob copy and without it.
//!
//! `rdb.rs` used to serialise every value by walking it, decoding one element at
//! a time and encoding it again, and it now copies the blob whole for the three
//! kinds that are already sitting in Redis's own layout. That is a claim about
//! time, so this is the file that puts a number on it rather than leaving it as
//! an argument about how obviously a memcpy is faster than a loop.
//!
//! The rows are paired on purpose. Each packed kind sits next to a value of the
//! same kind that is one member past the band, which is the same command on the
//! same data taking the walk instead, so the pair says what the copy is worth
//! and the second half of it also says that the walk did not get slower.
//!
//! `restore` is here because a payload is only useful if somebody loads it, and
//! the load side did not change: it accepted every one of these encodings before
//! this and it accepts them now. The row exists so that a later change to the
//! reader has something to move against.
//!
//! # Reading these on a machine someone else is using
//!
//! The same warning `benches/intset.rs` carries. Criterion's mean picks up
//! whatever else the box is doing, so the number to read is the minimum per
//! iteration across samples, out of `target/criterion/<group>/<id>/new/sample.json`
//! as `min(times[i]/iters[i])`, and the mean is not stable run to run.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::Keyspace;

/// The four kinds that have a packed band, and how many members put a value on
/// it or one past it.
///
/// The packed counts are inside the default thresholds, which are 128 for a set
/// listpack and a hash listpack and 128 for a sorted set, and 512 for an intset.
/// The walked counts are far enough past that nothing about the band is left.
const PACKED: usize = 100;
const WALKED: usize = 1000;

fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &[PACKED]
    } else {
        &[PACKED, WALKED]
    }
}

/// A keyspace holding one key of each kind at `n` elements.
///
/// Built once per row and dumped over and over, because `DUMP` does not change
/// what it reads and building it inside the loop would measure `SADD`.
fn filled(n: usize) -> Keyspace {
    let mut ks = Keyspace::new();
    let text: Vec<Vec<u8>> = (0..n).map(|i| format!("member:{i:0>8}").into()).collect();
    let ints: Vec<Vec<u8>> = (0..n).map(|i| format!("{}", i * 7 + 3).into()).collect();

    ks.sadd(b"text", text.iter().map(Vec::as_slice))
        .expect("a fresh set takes members");
    ks.sadd(b"ints", ints.iter().map(Vec::as_slice))
        .expect("a fresh set takes members");
    ks.hset(
        b"hash",
        text.iter().map(|f| (f.as_slice(), b"v".as_slice())),
    )
    .expect("a fresh hash takes fields");
    ks.zadd(
        b"zset",
        text.iter()
            .enumerate()
            .map(|(i, m)| (i as f64, m.as_slice())),
        yo_kv::ZAdd::default(),
    )
    .expect("a fresh sorted set takes members");
    ks
}

/// Every key `filled` puts in, in the order the rows report them.
const KEYS: [&[u8]; 4] = [b"text", b"ints", b"hash", b"zset"];

fn bench_dump(c: &mut Criterion) {
    let mut g = c.benchmark_group("rdb");
    for &n in sizes() {
        let mut ks = filled(n);
        for key in KEYS {
            let name = String::from_utf8_lossy(key).into_owned();
            // Per element, because the walk is linear in the members and the
            // copy is linear in the bytes, so the two only compare at a size.
            g.throughput(Throughput::Elements(n as u64));
            g.bench_with_input(BenchmarkId::new(format!("dump/{name}"), n), &n, |b, _| {
                b.iter(|| black_box(ks.dump(black_box(key))));
            });
        }
    }
    g.finish();
}

fn bench_restore(c: &mut Criterion) {
    let mut g = c.benchmark_group("rdb");
    for &n in sizes() {
        let mut ks = filled(n);
        for key in KEYS {
            let name = String::from_utf8_lossy(key).into_owned();
            let payload = ks.dump(key).expect("the key is there and has a shape");
            g.throughput(Throughput::Elements(n as u64));
            g.bench_with_input(
                BenchmarkId::new(format!("restore/{name}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        // Into a name nothing else uses, with `REPLACE`, so that
                        // every iteration does the same work as the first.
                        black_box(ks.restore(b"landing", black_box(&payload), None, true))
                    });
                },
            );
        }
    }
    g.finish();
}

criterion_group!(benches, bench_dump, bench_restore);
criterion_main!(benches);
