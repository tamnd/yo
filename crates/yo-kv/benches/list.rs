//! What a list operation costs, and where in the list it costs it.
//!
//! Six of M4's exit gate rows are list rows and none of them had a number here
//! before this file. The gate cells come from aki, and they are the reason the
//! chunked deque was built rather than a dense positional list: `LINSERT` at
//! 0.01x, `LPOS` at 0.49x, `RPOPLPUSH` at 0.84x, `LPOP` at 1.86x.
//!
//! What makes a list different from every other collection here is that the
//! cost depends on where you touch it and not only on how big it is. A push and
//! a pop work on a chunk that is already in cache because the last push put it
//! there. A `LINDEX` in the middle has to find which chunk holds that position,
//! and today that is a walk over the chunk ring from whichever end is closer,
//! which is the walk `08` section 5's descriptor cache is supposed to replace.
//! So every read row here is measured at both ends and in the middle, and the
//! gap between those two numbers is the whole argument for building the index.
//!
//! `find` is `LPOS` and it is a linear scan by definition, so it is measured
//! against a value that is genuinely at the far end rather than against one the
//! scan trips over immediately. That is the honest version of the row and it is
//! also the one aki lost.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::{List, ListLimits};

/// Redis's defaults, so the chunk size is the one a real server runs with.
fn limits() -> ListLimits {
    ListLimits::default()
}

/// Sixteen bytes, which is long enough to be a real element and short enough
/// that a chunk holds a realistic number of them.
fn value(i: usize) -> Vec<u8> {
    format!("element:{i:08}").into_bytes()
}

/// At a thousand the list is still one packed blob, at a hundred thousand it is
/// a few hundred chunks, and at a million it is a few thousand. The middle of
/// the last one is where a walk over the ring shows up.
const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &SIZES[..1]
    } else {
        &SIZES
    }
}

fn filled(n: usize) -> List {
    let lim = limits();
    let mut l = List::new();
    for i in 0..n {
        l.push_back(&value(i), &lim);
    }
    l
}

fn bench_ends(c: &mut Criterion) {
    let mut g = c.benchmark_group("list");
    let lim = limits();

    // Filling from the back, per element. This is `RPUSH` in a loop and it is
    // the row every other list number is read against.
    for &n in sizes() {
        let values: Vec<Vec<u8>> = (0..n).map(value).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("push_back", n), &n, |b, _| {
            b.iter_batched_ref(
                List::new,
                |l| {
                    for v in &values {
                        l.push_back(black_box(v), &lim);
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // The same from the front, which is the head chunk rather than the tail
        // chunk and is the reason the two ends are independent.
        g.bench_with_input(BenchmarkId::new("push_front", n), &n, |b, _| {
            b.iter_batched_ref(
                List::new,
                |l| {
                    for v in &values {
                        l.push_front(black_box(v), &lim);
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // `LPOP` over a list that is being drained, which is the shape a queue
        // has and the shape aki came in at 1.86x on.
        g.throughput(Throughput::Elements((n / 10).max(1) as u64));
        g.bench_with_input(BenchmarkId::new("pop_front", n), &n, |b, _| {
            b.iter_batched_ref(
                || filled(n),
                |l| {
                    let mut taken = 0usize;
                    for _ in 0..(n / 10).max(1) {
                        taken += l.pop_front(&lim).map_or(0, |v| v.len());
                    }
                    taken
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    g.finish();
}

fn bench_reach(c: &mut Criterion) {
    let mut g = c.benchmark_group("list");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let l = filled(n);

        // `LINDEX` near the front, which the ring walk reaches in one chunk.
        g.bench_with_input(BenchmarkId::new("index_near", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) & 63;
                black_box(l.get(black_box(i)).map(|e| e.byte_len()))
            })
        });

        // `LINDEX` in the middle, which is the same call and the far end of the
        // walk. The gap between this row and the one above it is what the
        // descriptor cache would close, and it is the number that says whether
        // the cache is worth building.
        //
        // Over the middle half rather than sixty four consecutive positions,
        // because consecutive positions all land in one chunk at one offset
        // inside it, and a chunk holds four hundred odd elements. That measures
        // whichever offset the midpoint happened to fall on. The stride is a
        // prime that is not a factor of any chunk count here, so the position
        // inside the chunk moves as well as the chunk does.
        g.bench_with_input(BenchmarkId::new("index_middle", n), &n, |b, _| {
            let half = (n / 2).max(1);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 7919) % half;
                black_box(l.get(black_box(n / 4 + i)).map(|e| e.byte_len()))
            })
        });

        // `LPOS` for something at the far end, which is the whole list read.
        // Per element rather than per call, so it reads as nanoseconds an
        // element like the walk rows in the other benchmarks.
        let last = value(n - 1);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("find_far", n), &n, |b, _| {
            b.iter(|| black_box(l.find(black_box(&last))))
        });

        // The same distance as `LPOS`, which is the whole list read and one
        // answer handed back. It is a separate row from `find_far` because it
        // carries a rank, a count and a `MAXLEN` budget that the pivot search
        // does not, and the question is whether carrying them costs anything.
        g.bench_with_input(BenchmarkId::new("lpos_far", n), &n, |b, _| {
            b.iter(|| {
                let mut at = 0usize;
                l.positions(black_box(&last), 1, 1, 0, &mut |p| at = p);
                black_box(at)
            })
        });

        // And `LPOS` with a negative rank for something at the front, which is
        // the whole list read the other way. The backward walk steps by back
        // lengths rather than by headers, so this is the row that says whether
        // the two directions cost the same.
        let first = value(0);
        g.bench_with_input(BenchmarkId::new("lpos_far_back", n), &n, |b, _| {
            b.iter(|| {
                let mut at = 0usize;
                l.positions(black_box(&first), -1, 1, 0, &mut |p| at = p);
                black_box(at)
            })
        });

        // `LRANGE` over a window in the middle, which is a locate and then a
        // sequential read, and is the shape a paging client actually sends. The
        // start moves over the middle half for the same reason the row above
        // does.
        g.throughput(Throughput::Elements(100));
        g.bench_with_input(BenchmarkId::new("range_middle", n), &n, |b, _| {
            let half = (n / 2).max(1);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 7919) % half;
                let mut got = 0usize;
                for e in l.range(black_box(n / 4 + i), 100) {
                    got += e.byte_len();
                }
                got
            })
        });
        g.throughput(Throughput::Elements(1));
    }

    g.finish();
}

fn bench_middle(c: &mut Criterion) {
    let mut g = c.benchmark_group("list");
    let lim = limits();
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        // An insert at a known position, which is `LINSERT` with the pivot
        // search taken out. This is the mechanism the milestone is about: a
        // middle insert should rewrite one chunk and split it at worst, and
        // should not touch the rest of the list at all.
        g.bench_with_input(BenchmarkId::new("insert_middle", n), &n, |b, _| {
            b.iter_batched_ref(
                || filled(n),
                |l| {
                    // A hundred inserts around the midpoint per batch, so the
                    // setup cost is spread and the list does not drift far from
                    // the size being measured.
                    for i in 0..100 {
                        l.insert(l.len() / 2 + i, black_box(b"inserted"), &lim);
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // And the whole command, pivot search included, against a pivot in the
        // middle. This is the 0.01x cell.
        let pivot = value(n / 2);
        g.bench_with_input(BenchmarkId::new("linsert_middle", n), &n, |b, _| {
            b.iter_batched_ref(
                || filled(n),
                |l| {
                    l.insert_at_pivot(black_box(&pivot), b"inserted", true, &lim);
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    g.finish();
}

criterion_group!(benches, bench_ends, bench_reach, bench_middle);
criterion_main!(benches);
