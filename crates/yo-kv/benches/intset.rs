//! What a binary search costs on a set of integers, against the two structures
//! it sits between.
//!
//! `intset.rs` claims a binary search is affordable up to the 512 member
//! threshold because the whole set is a page and most of the search lands in
//! cache. That is a claim with a number in it, and this file is here because the
//! last two claims of that shape in this project were both wrong: L6 put a
//! positional probe at 70 ns and it measured 13, and K11 put a crossover at k
//! equals seven and there is no crossover. So the intset is measured against the
//! listpack it replaces below the threshold and against the element table it
//! promotes into above it, on the same members, rather than argued about.
//!
//! The comparison is not apples to apples on purpose. The listpack and the table
//! hold members as bytes and the intset holds them as integers, which means the
//! intset never parses and the other two never binary search. That difference is
//! the whole design, so measuring around it would measure nothing.
//!
//! # Reading these on a machine someone else is using
//!
//! Criterion's mean picks up whatever else the box is doing, and this laptop
//! usually has another agent building something on it. Contention only ever adds
//! time, so the number to read is the minimum per iteration across samples, out
//! of `target/criterion/<group>/<id>/new/sample.json` as `min(times[i]/iters[i])`.
//! That is stable run to run where the mean is not.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::{Elements, Intset, Listpack};

/// Across the band and one size past the 512 threshold, so that what happens
/// after promotion is visible rather than inferred.
const SIZES: [usize; 5] = [8, 64, 128, 512, 4096];

fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &SIZES[..1]
    } else {
        &SIZES
    }
}

/// Members spread over a range wider than the count, so the values are not the
/// indices and a search cannot be a subtraction.
fn members(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i * 7 + 3).collect()
}

/// A value inside the range of the members and not one of them, which is the
/// miss that costs the most: the two range tests in front of the search both
/// pass and the whole binary search runs.
fn interior_miss(n: usize) -> i64 {
    (n as i64 / 2) * 7 + 4
}

fn bench_contains(c: &mut Criterion) {
    let mut g = c.benchmark_group("intset");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let vals = members(n);
        let mut is = Intset::new();
        let mut lp = Listpack::new();
        let mut el = Elements::<()>::new();
        for &v in &vals {
            is.add(v);
            lp.push(v.to_string().as_bytes());
            el.insert(v.to_string().as_bytes(), ()).expect("room");
        }
        let text: Vec<Vec<u8>> = vals.iter().map(|v| v.to_string().into_bytes()).collect();
        let miss = interior_miss(n);
        let miss_text = miss.to_string().into_bytes();

        // Every member in turn rather than one member over and over, so a hit is
        // measured at every depth of the search and not only at the root.
        g.bench_with_input(BenchmarkId::new("intset_hit", n), &n, |b, _| {
            let mut i = 0;
            b.iter(|| {
                i = (i + 1) % vals.len();
                black_box(is.contains(black_box(vals[i])))
            });
        });
        g.bench_with_input(BenchmarkId::new("intset_miss", n), &n, |b, _| {
            b.iter(|| black_box(is.contains(black_box(miss))));
        });
        g.bench_with_input(BenchmarkId::new("listpack_hit", n), &n, |b, _| {
            let mut i = 0;
            b.iter(|| {
                i = (i + 1) % text.len();
                black_box(lp.find(black_box(&text[i]), 1))
            });
        });
        g.bench_with_input(BenchmarkId::new("listpack_miss", n), &n, |b, _| {
            b.iter(|| black_box(lp.find(black_box(&miss_text), 1)));
        });
        g.bench_with_input(BenchmarkId::new("table_hit", n), &n, |b, _| {
            let mut i = 0;
            b.iter(|| {
                i = (i + 1) % text.len();
                black_box(el.get(black_box(&text[i])))
            });
        });
        g.bench_with_input(BenchmarkId::new("table_miss", n), &n, |b, _| {
            b.iter(|| black_box(el.get(black_box(&miss_text))));
        });
    }
    g.finish();
}

fn bench_fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("intset_fill");

    for &n in sizes() {
        let vals = members(n);
        // The other order the same members can arrive in. Ascending appends and
        // moves nothing, and this is what it costs when every add lands in the
        // middle and memmoves the tail.
        let mut shuffled = vals.clone();
        shuffled.sort_by_key(|v| v.wrapping_mul(2_654_435_761));

        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("intset_ascending", n), &n, |b, _| {
            b.iter(|| {
                let mut is = Intset::with_capacity(n);
                for &v in &vals {
                    is.add(black_box(v));
                }
                black_box(is.len())
            });
        });
        g.bench_with_input(BenchmarkId::new("intset_scattered", n), &n, |b, _| {
            b.iter(|| {
                let mut is = Intset::with_capacity(n);
                for &v in &shuffled {
                    is.add(black_box(v));
                }
                black_box(is.len())
            });
        });
        g.bench_with_input(BenchmarkId::new("table", n), &n, |b, _| {
            let text: Vec<Vec<u8>> = shuffled
                .iter()
                .map(|v| v.to_string().into_bytes())
                .collect();
            b.iter(|| {
                let mut el = Elements::<()>::with_capacity(n);
                for m in &text {
                    el.insert(black_box(m), ()).expect("room");
                }
                black_box(el.len())
            });
        });
    }
    g.finish();
}

fn bench_memory(c: &mut Criterion) {
    // Not a timing. Criterion is the harness that is already here and the row is
    // printed once per size, which is what makes the ratio easy to read next to
    // the timed rows above rather than in a separate note nobody opens.
    let mut g = c.benchmark_group("intset_memory");
    g.sample_size(10);

    for &n in sizes() {
        let vals = members(n);
        let mut is = Intset::new();
        let mut lp = Listpack::new();
        let mut el = Elements::<()>::new();
        for &v in &vals {
            is.add(v);
            lp.push(v.to_string().as_bytes());
            el.insert(v.to_string().as_bytes(), ()).expect("room");
        }
        println!(
            "  memory at {n}: intset {:.1} B/member, listpack {:.1}, table {:.1}",
            is.byte_len() as f64 / n as f64,
            lp.byte_len() as f64 / n as f64,
            el.memory_bytes() as f64 / n as f64,
        );
        g.bench_with_input(BenchmarkId::new("bytes_per_member", n), &n, |b, _| {
            b.iter(|| black_box(is.byte_len()));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_contains, bench_fill, bench_memory);
criterion_main!(benches);
