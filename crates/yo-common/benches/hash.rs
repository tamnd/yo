//! The hash budget.
//!
//! M0's exit gate wants the index probe at or under 4 ns and aki measured
//! wyhash at 1.95 ns against fnv1a at 5.39 ns on the same keys. Since the hash
//! is the first half of a probe, it gets its own row so that a probe regression
//! can be attributed to the hash or to the bucket walk rather than to "the
//! index".
//!
//! fnv1a is here purely as the reference point aki measured against. It is not
//! used by the engine and it is not going to be.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::{crc32c, slot_of, wyhash::wyhash};

/// The hash aki replaced, kept only so the 1.95 against 5.39 comparison can be
/// rerun on whatever box is in front of us rather than quoted from a document.
fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn keys(n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut k = format!("key:{i}").into_bytes();
            k.resize(len, b'x');
            k
        })
        .collect()
}

fn bench_hash(c: &mut Criterion) {
    let mut g = c.benchmark_group("hash");

    // 16 bytes is the common Redis key. 64 and 256 cover the tail. Each length
    // is a separate row because wyhash changes shape at 16 and 48.
    for len in [8usize, 16, 64, 256] {
        let ks = keys(1024, len);
        g.throughput(Throughput::Bytes(len as u64));

        g.bench_with_input(BenchmarkId::new("wyhash", len), &ks, |b, ks| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) & 1023;
                black_box(wyhash(black_box(&ks[i]), 0))
            })
        });

        g.bench_with_input(BenchmarkId::new("fnv1a", len), &ks, |b, ks| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 1) & 1023;
                black_box(fnv1a(black_box(&ks[i])))
            })
        });
    }
    g.finish();
}

fn bench_slot(c: &mut Criterion) {
    let ks = keys(1024, 16);
    c.bench_function("slot_of/16B", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) & 1023;
            black_box(slot_of(black_box(&ks[i])))
        })
    });
}

fn bench_crc32c(c: &mut Criterion) {
    let mut g = c.benchmark_group("crc32c");
    // 16 KiB is the default page, which is the size that actually matters for
    // `07`'s checksum on the write path.
    for len in [64usize, 4096, 16384] {
        let buf: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        g.throughput(Throughput::Bytes(len as u64));
        g.bench_with_input(BenchmarkId::from_parameter(len), &buf, |b, buf| {
            b.iter(|| black_box(crc32c(0, black_box(buf))))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_hash, bench_slot, bench_crc32c);
criterion_main!(benches);
