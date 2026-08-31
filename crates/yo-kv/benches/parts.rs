//! What the partitioned band costs, and what the descriptor cache buys.
//!
//! The number this file exists for is K10. `05` section 4.3 says resolving a
//! partition costs 9.9 to 11.3 ns with the descriptor cache and 177 to 275 ns
//! without, and that gap is the entire argument for keeping every partition's
//! length written down a second time. A number carried over from another
//! codebase is a claim, so both sides are built here and measured against each
//! other on the same data.
//!
//! The two arms are the same collection reached two ways. `cached` is
//! [`Parts::at`], which accumulates over a contiguous `[u32]`. `headers` is the
//! same walk with the lengths read out of the partitions themselves, which is
//! what the structure would do with no cache in front of it. Both end in the
//! same table lookup, so the difference between them is the resolve and nothing
//! else.
//!
//! # Reading these on a machine someone else is using
//!
//! The same rule as everywhere else in this crate: take the minimum per
//! iteration across samples out of
//! `target/criterion/<group>/<id>/new/sample.json` rather than criterion's mean,
//! because contention only ever adds time.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::hash_key;
use yo_kv::elem::Elements;
use yo_kv::parts::{PART_BIT, Parts};

/// The layouts to sweep, from the floor to the ceiling. The uncached resolve is
/// O(P) and should climb straight through them. The cached one is O(sqrt P) and
/// should barely move.
const LAYOUTS: [u32; 6] = [4, 16, 64, 256, 1_024, 2_048];

/// How many elements go in. Enough that a partition is a real table rather than
/// a handful of rows, and small enough that the whole sweep builds in seconds.
const MEMBERS: usize = 32_768;

fn layouts() -> &'static [u32] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &LAYOUTS[..1]
    } else {
        &LAYOUTS
    }
}

/// The same members, held as a partitioned band.
fn band(parts: u32) -> Parts<u32> {
    let mut p = Parts::with_parts(parts);
    for i in 0..MEMBERS {
        p.insert(format!("member:{i}").as_bytes(), i as u32)
            .expect("room");
    }
    p
}

/// The same members again, held as a bare array of tables with no cache in front
/// of them. This is what the band would be without the thing being measured.
fn bare(parts: u32) -> Vec<Elements<u32>> {
    let mut tables: Vec<Elements<u32>> = (0..parts).map(|_| Elements::new()).collect();
    let mask = parts - 1;
    for i in 0..MEMBERS {
        let name = format!("member:{i}");
        let h = hash_key(name.as_bytes());
        let at = ((h >> PART_BIT) as u32) & mask;
        tables[at as usize]
            .insert_hashed(h, name.as_bytes(), i as u32)
            .expect("room");
    }
    tables
}

/// Which partition holds global position `idx`, with the lengths read out of the
/// partitions themselves. This is the structure with no cache in front of it: an
/// O(P) walk, which is what the two level cache is there to replace.
fn locate_uncached(tables: &[Elements<u32>], idx: usize) -> Option<(usize, usize)> {
    let mut seen = 0usize;
    for (at, table) in tables.iter().enumerate() {
        let n = table.len();
        if idx < seen + n {
            return Some((at, idx - seen));
        }
        seen += n;
    }
    None
}

fn bench_locate(c: &mut Criterion) {
    let mut g = c.benchmark_group("parts_locate");
    g.throughput(Throughput::Elements(1));

    for &parts in layouts() {
        let p = band(parts);
        let tables = bare(parts);
        let n = p.len();
        // A different position every iteration and one that walks the whole
        // range, so neither arm gets to keep one partition hot. A stride coprime
        // with the length visits every position before it repeats.
        let stride = 7919;

        g.bench_with_input(BenchmarkId::new("cached", parts), &parts, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + stride) % n;
                black_box(p.locate(black_box(i)))
            });
        });

        g.bench_with_input(BenchmarkId::new("headers", parts), &parts, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + stride) % n;
                black_box(locate_uncached(black_box(&tables), black_box(i)))
            });
        });

        // The resolve with the fetch on the end of it, which is what
        // `SRANDMEMBER` actually pays.
        g.bench_with_input(BenchmarkId::new("draw", parts), &parts, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + stride) % n;
                black_box(p.at(black_box(i)))
            });
        });
    }
    g.finish();
}

fn bench_access(c: &mut Criterion) {
    let mut g = c.benchmark_group("parts");
    g.throughput(Throughput::Elements(1));

    let names: Vec<Vec<u8>> = (0..MEMBERS)
        .map(|i| format!("member:{i}").into_bytes())
        .collect();

    // One table against the band on the read path. The band pays one extra shift
    // and mask and one extra indirection, and the partitions are smaller, so
    // this is where a partitioned read is shown not to have got worse.
    let mut one = Elements::<u32>::new();
    for (i, name) in names.iter().enumerate() {
        one.insert(name, i as u32).expect("room");
    }
    g.bench_function("get_one_table", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % MEMBERS;
            black_box(one.get(black_box(&names[i])))
        });
    });

    for &parts in layouts() {
        let p = band(parts);
        g.bench_with_input(BenchmarkId::new("get", parts), &parts, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) % MEMBERS;
                black_box(p.get(black_box(&names[i])))
            });
        });
    }
    g.finish();
}

fn bench_scan(c: &mut Criterion) {
    let mut g = c.benchmark_group("parts_scan");
    g.throughput(Throughput::Elements(MEMBERS as u64));

    for &parts in layouts() {
        let p = band(parts);
        // A whole scan in pages of ten, which is the default `COUNT`, so the
        // per partition seam is crossed once per partition and paid for here
        // rather than hidden inside one big page.
        g.bench_with_input(BenchmarkId::new("whole", parts), &parts, |b, _| {
            b.iter(|| {
                let mut cursor = yo_kv::Cursor::START;
                let mut seen = 0usize;
                loop {
                    cursor = p.scan(cursor, 10, |_, _| seen += 1);
                    if cursor.is_end() {
                        break;
                    }
                }
                black_box(seen)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_locate, bench_access, bench_scan);
criterion_main!(benches);
