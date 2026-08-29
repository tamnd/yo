//! Where the inline band stops paying.
//!
//! The size ladder in `05` section 4 says a collection under a hundred and
//! twenty eight elements is a blob walked linearly and above it is an element
//! table with an index, and it argues for that on speed, citing L6's fifty times
//! gap between a 70 ns positional probe and a 1 to 2 ns listpack walk.
//!
//! That is a claim with a number in it, so it is measured here rather than
//! repeated, against our own element table and not against whatever L6 measured.
//! The answer is that the walk number holds and the probe number does not: ours
//! probes in 13 ns, and there is no crossover anywhere in the band. The blob
//! loses every timed row at every size and wins on memory by 2.4 times, which is
//! what actually keeps it. The band stays because of `memory` down at the bottom
//! of this file and because `OBJECT ENCODING` has to agree with Redis, not
//! because of `find`.
//!
//! Leave the timed rows in anyway. They are what would tell us if a change to
//! either structure moved the crossover into the band, and they are the evidence
//! for a design note that currently reads against the spec.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::{Elements, Listpack};

/// The band, and one size past the top of it so the crossover is visible rather
/// than inferred.
const SIZES: [usize; 5] = [8, 32, 64, 128, 512];

/// Long enough that a name is not free to compare and short enough to be the
/// six bit length encoding, which is what a real field name looks like.
fn name(i: usize) -> Vec<u8> {
    format!("member:{i:04}").into_bytes()
}

fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &SIZES[..1]
    } else {
        &SIZES
    }
}

fn bench_find(c: &mut Criterion) {
    let mut g = c.benchmark_group("inline");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let names: Vec<Vec<u8>> = (0..n).map(name).collect();
        let mut lp = Listpack::new();
        let mut el = Elements::<()>::new();
        for m in &names {
            lp.push(m);
            el.insert(m, ()).expect("room");
        }
        let miss = name(n + 1);

        // The average hit, which for a walk is halfway through and for a probe
        // is one slot and one row wherever it lands. Every member in turn, so
        // the walk is not measured only on the lucky end.
        g.bench_with_input(BenchmarkId::new("listpack_find_hit", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(lp.find(black_box(&names[i]), 1))
            })
        });
        g.bench_with_input(BenchmarkId::new("elements_find_hit", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % n;
                black_box(el.contains(black_box(&names[i])))
            })
        });

        // The miss is the worst case for the walk, since it reads every element
        // before it can answer, and the best case for the probe, since the tag
        // usually settles it without reading a name at all. If the walk still
        // wins here the threshold has room.
        g.bench_with_input(BenchmarkId::new("listpack_find_miss", n), &n, |b, _| {
            b.iter(|| black_box(lp.find(black_box(&miss), 1)))
        });
        g.bench_with_input(BenchmarkId::new("elements_find_miss", n), &n, |b, _| {
            b.iter(|| black_box(el.contains(black_box(&miss))))
        });

        // The reply path. Per element, so it reads the same way the walk in the
        // element table benchmark does.
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("listpack_walk", n), &n, |b, _| {
            b.iter(|| {
                let mut got = 0usize;
                for e in lp.iter() {
                    match e {
                        yo_kv::Entry::Str(s) => got += s.len() + s[0] as usize,
                        yo_kv::Entry::Int(v) => got += v as usize,
                    }
                }
                got
            })
        });
        g.bench_with_input(BenchmarkId::new("elements_walk", n), &n, |b, _| {
            b.iter(|| {
                let mut got = 0usize;
                for (m, ()) in el.iter() {
                    got += m.len() + m[0] as usize;
                }
                got
            })
        });
        g.throughput(Throughput::Elements(1));
    }

    g.finish();
}

fn bench_fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("inline");

    // Building the collection from nothing, per element. The blob reallocates as
    // it grows and copies its tail on every insert that is not an append, and
    // the table rehashes, so this is where the blob is meant to look worst.
    for &n in sizes() {
        let names: Vec<Vec<u8>> = (0..n).map(name).collect();
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("listpack_fill", n), &n, |b, _| {
            b.iter_batched_ref(
                Listpack::new,
                |lp| {
                    for m in &names {
                        lp.push(black_box(m));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });

        g.bench_with_input(BenchmarkId::new("elements_fill", n), &n, |b, _| {
            b.iter_batched_ref(
                Elements::<()>::new,
                |el| {
                    for m in &names {
                        el.insert(black_box(m), ()).expect("room");
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    g.finish();
}

/// What the two shapes cost to hold, which is the other half of the reason the
/// small band exists. Not timed, so it runs once and prints.
fn bench_memory(c: &mut Criterion) {
    let g = c.benchmark_group("inline");
    for &n in sizes() {
        let names: Vec<Vec<u8>> = (0..n).map(name).collect();
        let mut lp = Listpack::new();
        let mut el = Elements::<()>::new();
        for m in &names {
            lp.push(m);
            el.insert(m, ()).expect("room");
        }
        println!(
            "inline/memory/{n}: listpack {} bytes ({:.1} per element), elements {} bytes ({:.1} per element)",
            lp.byte_len(),
            lp.byte_len() as f64 / n as f64,
            el.memory_bytes(),
            el.memory_bytes() as f64 / n as f64,
        );
    }
    g.finish();
}

criterion_group!(benches, bench_find, bench_fill, bench_memory);
criterion_main!(benches);
