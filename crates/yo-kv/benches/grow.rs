//! What a growth factor actually costs, which is not what counting copies says.
//!
//! `grow.rs` picks the factor every large array in the engine grows by, and for
//! a long time it picked a quarter on an argument rather than a measurement: an
//! element is copied about five times over the life of an array that grows by a
//! quarter and about twice for one that doubles, so a quarter buys memory and
//! spends a little throughput. That count is wrong for exactly the arrays the
//! module's threshold selects, because a growing array is grown with `realloc`,
//! and past the system allocator's mmap threshold `realloc` on a bigger size is
//! a page table edit rather than a walk over the bytes. Whether that actually
//! happens depends on the allocator, on how much of the address space after the
//! mapping is free, and on glibc raising its own mmap threshold as it goes, so
//! it is a thing to measure and not a thing to assert.
//!
//! So these fill an array one element at a time through [`grow::reserve`] and
//! report the cost per element. The number that matters is not the absolute one,
//! it is the ratio between two branches with different factors in them, run back
//! to back on the same machine.
//!
//! `presized` is the control. It asks for the whole array up front, so it pays
//! for one allocation and no growth at all, and the gap between it and the
//! others is the entire cost of the growth policy. If a fill is within a few
//! percent of its control then the factor is free at that size whatever the copy
//! count says.
//!
//! Two element sizes, because the answer is allowed to differ. Eight bytes is a
//! row of an element table, and one byte is the name blob under it, which
//! reaches the mmap threshold at eight times the element count.
//!
//! # Reading these on a machine someone else is using
//!
//! The same rule the other benches in this crate carry. Criterion's mean picks
//! up whatever else the box is doing, so on a shared machine read the minimum
//! per iteration across samples out of `target/criterion/<group>/<id>/new/
//! sample.json` as `min(times[i]/iters[i])`. On this laptop the mean of a fill
//! moved forty five percent between two runs of the same commit, so a difference
//! under about fifteen percent measured here means nothing at all. gamingpc
//! under `taskset` is where these get read.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::grow;

/// Sizes either side of where the policy changes and where the allocator does.
///
/// The threshold in `grow.rs` is sixty four kilobytes, which is eight thousand
/// eight byte elements, and the mmap threshold on glibc starts at a hundred and
/// twenty eight kilobytes. So the small size is under both, the middle is over
/// the first and around the second, and the large one is far enough over both
/// that every growth on the way there should be a remap.
const SIZES: [usize; 3] = [4_000, 100_000, 4_000_000];

/// Push `n` elements through the policy, the way a row array is actually built.
fn fill<T: Copy + Default>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::new();
    for _ in 0..n {
        grow::reserve(&mut v, 1);
        v.push(T::default());
    }
    v
}

/// The same fill with the whole array asked for up front.
fn presized<T: Copy + Default>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(T::default());
    }
    v
}

fn rows(c: &mut Criterion) {
    let mut g = c.benchmark_group("grow");
    for n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("rows", n), &n, |b, &n| {
            b.iter(|| black_box(fill::<u64>(n).len()));
        });
        g.bench_with_input(BenchmarkId::new("rows_presized", n), &n, |b, &n| {
            b.iter(|| black_box(presized::<u64>(n).len()));
        });
    }
    g.finish();
}

fn names(c: &mut Criterion) {
    let mut g = c.benchmark_group("grow");
    for n in SIZES {
        // A member of an element table is sixteen bytes, so the blob under a
        // table of n rows is sixteen n. Kept to the same element counts as the
        // rows so the two rows of the table line up.
        let bytes = n * 16;
        g.throughput(Throughput::Bytes(bytes as u64));
        g.bench_with_input(BenchmarkId::new("names", n), &bytes, |b, &bytes| {
            b.iter(|| black_box(fill::<u8>(bytes).len()));
        });
        g.bench_with_input(
            BenchmarkId::new("names_presized", n),
            &bytes,
            |b, &bytes| {
                b.iter(|| black_box(presized::<u8>(bytes).len()));
            },
        );
    }
    g.finish();
}

/// What the policy is holding at the end of a fill, which is the other half.
///
/// Not a benchmark, but it belongs beside these numbers rather than in a
/// separate place, because the factor is a trade and quoting one side of a trade
/// is how #175 went wrong. Criterion runs it once and prints.
fn slack(c: &mut Criterion) {
    for n in SIZES {
        let v = fill::<u64>(n);
        println!(
            "grow/slack rows n={n:<9} cap={:<10} slack={:.2}%  bytes_per_element={:.2}",
            v.capacity(),
            (v.capacity() - v.len()) as f64 * 100.0 / v.len() as f64,
            v.capacity() as f64 * 8.0 / v.len() as f64
        );
    }
    // Nothing to register, but criterion wants a group to exist.
    c.benchmark_group("grow").finish();
}

criterion_group!(benches, rows, names, slack);
criterion_main!(benches);
