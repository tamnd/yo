//! The arena budget.
//!
//! M0's exit gate is a bump at or under 2 ns. The comparison row is the system
//! allocator doing the same job, because "the arena is faster than malloc" is
//! the claim `05` section 3.3 makes and it should be a number rather than an
//! assertion.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_arena::{Arena, SEGMENT_SIZE};

fn bench_bump(c: &mut Criterion) {
    let mut g = c.benchmark_group("arena");

    for size in [16usize, 64, 256] {
        g.bench_with_input(BenchmarkId::new("bump", size), &size, |b, &size| {
            // A fresh arena per batch, so that this measures the bump and not
            // the segment growth that a long run would eventually dominate.
            b.iter_batched_ref(
                Arena::new,
                |a| {
                    for _ in 0..1024 {
                        black_box(a.alloc(black_box(size)));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });

        g.bench_with_input(BenchmarkId::new("malloc", size), &size, |b, &size| {
            b.iter(|| {
                let mut keep: Vec<Vec<u8>> = Vec::with_capacity(1024);
                for _ in 0..1024 {
                    keep.push(black_box(vec![0u8; black_box(size)]));
                }
                black_box(keep)
            })
        });
    }
    g.finish();
}

fn bench_put(c: &mut Criterion) {
    let mut g = c.benchmark_group("arena_put");
    // 64 B is the gate value size, 1 KiB and 64 KiB are the other two bolded
    // rows in bench/00 section 2.
    for size in [64usize, 1024, 65536] {
        let data = vec![0xabu8; size];
        g.throughput(criterion::Throughput::Bytes(size as u64));
        g.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            let n = (SEGMENT_SIZE / (data.len() + 16)).max(1);
            b.iter_batched_ref(
                Arena::new,
                |a| {
                    for _ in 0..n {
                        black_box(a.put(black_box(data)));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench_bump, bench_put);
criterion_main!(benches);
