//! What an element costs.
//!
//! Four numbers matter here and they are the four the design was chosen for.
//!
//! `probe` is a membership question, which is `SISMEMBER`, `HEXISTS` and the
//! read half of every other command. One slot load and one row load, with the
//! name only compared when the tag says it is worth comparing.
//!
//! `draw` is the K9 row. `SRANDMEMBER` picks an index and reads it, and the
//! target from the spec is 4.8 ns at 100 K members and 12.2 ns at 1 M, where the
//! difference between the two is the cache and not the code. This is the
//! operation aki lost on, at 0.29x at pipeline 1, so it gets its own number
//! rather than being folded into a set benchmark later.
//!
//! `pop` is a draw and a removal together, which is `SPOP`. It carries the extra
//! probe that keeps the row array dense, and the question this answers is
//! whether that probe costs more than the retry loop a table with holes would
//! have paid instead.
//!
//! `walk` is `SMEMBERS` and `HGETALL`, per element, over a table that is far too
//! large for the cache. It is the one number that should be close to the memory
//! system's sequential read and nothing else.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::Elements;

/// A set: an element table with nothing stored against a member.
type Set = Elements<()>;

/// A hash: eight bytes of payload standing in for the value address and the TTL
/// slot the real hash will put there.
type Hash = Elements<u64>;

/// A sorted set: a score against a member.
///
/// This is the payload that made the split worth doing, because it is the one
/// with an alignment stricter than the row's, and it is the one that has to be
/// checked for what the split costs. A member and its score are in two arrays
/// now rather than one, so a walk that reads both is two sequential streams.
type Scored = Elements<f64>;

fn name(i: usize) -> Vec<u8> {
    format!("member:{i:012}").into_bytes()
}

/// At 1e3 the table is in L2, at 1e5 it is around L3, and at 1e6 a probe is a
/// trip to DRAM. The same ladder the index benchmark uses, so the two are
/// readable side by side.
const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

/// CI runs every benchmark once to check it still builds and still runs, and
/// gets nothing from filling a million member set to look in it once.
fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &SIZES[..1]
    } else {
        &SIZES
    }
}

fn filled(n: usize) -> Set {
    let mut s = Set::with_capacity(n);
    for i in 0..n {
        s.insert(&name(i), ()).expect("room");
    }
    s
}

fn bench_probe(c: &mut Criterion) {
    let mut g = c.benchmark_group("elem");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let s = filled(n);
        // A thousand and twenty four names, spread over the table by a stride
        // that is coprime with every size here, so the run does not walk the
        // rows in order and get a prefetch it would not get in production.
        let hits: Vec<Vec<u8>> = (0..1024).map(|i| name(i * 7919 % n)).collect();
        let misses: Vec<Vec<u8>> = (0..1024).map(|i| name(n + i)).collect();

        g.bench_with_input(BenchmarkId::new("probe_hit", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) & 1023;
                black_box(s.contains(black_box(&hits[i])))
            })
        });

        g.bench_with_input(BenchmarkId::new("probe_miss", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) & 1023;
                black_box(s.contains(black_box(&misses[i])))
            })
        });

        // The draw. An index into the row array and a read, which is the whole
        // operation, and the reason there is no ordered structure here.
        g.bench_with_input(BenchmarkId::new("draw", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i * 7919 + 13) % n;
                black_box(s.at(black_box(i)))
            })
        });

        // The walk, per element, over a table nothing fits in. The throughput is
        // the whole table rather than one element, so criterion divides by the
        // element count and the number reads as nanoseconds per element like
        // every other row here.
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("walk", n), &n, |b, _| {
            b.iter(|| {
                // The name bytes are read and not only their length. A walk that
                // sums lengths never touches the blob, and `HGETALL` copies
                // every byte of every name onto the wire.
                let mut got = 0usize;
                for (m, ()) in s.iter() {
                    got += m.len() + m[0] as usize + m[m.len() - 1] as usize;
                }
                got
            })
        });
        g.throughput(Throughput::Elements(1));
    }

    // A field write against a table already at size, which is what a hash
    // actually gets most of the time: the name is already interned, so this is a
    // probe and a row update and no name bytes at all.
    for &n in sizes() {
        let mut h = Hash::with_capacity(n);
        for i in 0..n {
            h.insert(&name(i), i as u64).expect("room");
        }
        let over: Vec<Vec<u8>> = (0..1024).map(|i| name(i * 7919 % n)).collect();
        g.bench_with_input(BenchmarkId::new("write_over", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) & 1023;
                black_box(h.insert(black_box(&over[i]), 7)).ok()
            })
        });
    }

    // A walk that reads the name and the payload together, which is `ZRANGE`
    // with scores and the one shape the split payload could plausibly hurt: the
    // member and its score are in two arrays now instead of one. Two sequential
    // streams rather than one, which a prefetcher covers, against four bytes an
    // element of padding that the single array was paying.
    for &n in sizes() {
        let mut z = Scored::with_capacity(n);
        for i in 0..n {
            z.insert(&name(i), i as f64).expect("room");
        }
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("score_walk", n), &n, |b, _| {
            b.iter(|| {
                let mut got = 0.0f64;
                for (m, &sc) in z.iter() {
                    got += sc + m.len() as f64;
                }
                got
            })
        });
        g.throughput(Throughput::Elements(1));
    }

    g.finish();
}

fn bench_churn(c: &mut Criterion) {
    let mut g = c.benchmark_group("elem");

    // Filling from empty, per element, including every growth on the way up.
    // A separate number from `write_over` because a write that grows the table
    // and a write that lands in it are different operations and one average
    // over both describes neither.
    for &n in sizes() {
        let names: Vec<Vec<u8>> = (0..n).map(name).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("fill", n), &n, |b, _| {
            b.iter_batched_ref(
                Set::new,
                |s| {
                    for name in &names {
                        s.insert(black_box(name), ()).expect("room");
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    // A draw and a removal together, which is SPOP, measured over a set that is
    // being emptied rather than over a steady one, because that is the shape
    // SPOP in a loop actually has and the shape a tombstone table degrades in.
    for &n in sizes() {
        g.throughput(Throughput::Elements((n / 10).max(1) as u64));
        g.bench_with_input(BenchmarkId::new("pop", n), &n, |b, _| {
            b.iter_batched_ref(
                || filled(n),
                |s| {
                    let mut taken = 0usize;
                    // A tenth of the set per batch. Enough to leave the table in
                    // a churned state and not so much that the last few draws
                    // are measuring a table of ten members.
                    //
                    // Read the name, then remove by index. That is the shape the
                    // command takes, because the name is going into a reply
                    // buffer either way and copying it into a `Vec` first would
                    // be measuring an allocator.
                    for i in 0..n / 10 {
                        let at = (i * 7919 + 13) % s.len();
                        taken += s.at(black_box(at)).expect("in range").0.len();
                        s.remove_at(at);
                    }
                    taken
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    g.finish();
}

criterion_group!(benches, bench_probe, bench_churn);
criterion_main!(benches);
