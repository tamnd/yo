//! What an append costs, with the store taken out of the picture.
//!
//! These numbers are the append path only: format the record, publish the
//! length, move the tail. A [`NullSink`] means no write ever happens, which is
//! the point. Durability is measured separately, on a real device, because a
//! number that mixes the two tells you about the device.
//!
//! The bar is that an append is in the same range as a bump allocation plus a
//! memcpy of the value, because that is all it is allowed to be. Anything above
//! that is bookkeeping that should not exist.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_format::{RecordHeader, RecordKind};
use yo_record::sink::{MemorySink, NullSink};
use yo_record::{Durability, Log, LogConfig};

/// A key buffer on the stack. Formatting into a `String` inside the loop
/// measures the allocator, which is a mistake this project has already made
/// once and does not intend to make twice.
struct KeyBuf([u8; 24]);

impl KeyBuf {
    fn new() -> KeyBuf {
        KeyBuf(*b"key:00000000000000000000")
    }

    fn set(&mut self, mut n: u64) -> &[u8] {
        for b in self.0[4..].iter_mut().rev() {
            *b = b'0' + (n % 10) as u8;
            n /= 10;
        }
        &self.0
    }
}

fn cfg(durability: Durability) -> LogConfig {
    LogConfig {
        page_len: 1 << 20,
        resident_pages: 3,
        mutable_fraction: 0.40,
        durability,
        ..LogConfig::default()
    }
}

fn append(c: &mut Criterion) {
    let mut g = c.benchmark_group("append");
    let header = RecordHeader::new(RecordKind::String);

    for vlen in [8usize, 64, 256, 1024] {
        let value = vec![b'v'; vlen];
        g.throughput(Throughput::Bytes(vlen as u64));
        g.bench_with_input(BenchmarkId::new("null_sink", vlen), &vlen, |b, _| {
            let mut log = Log::new(cfg(Durability::None), NullSink).unwrap();
            let mut key = KeyBuf::new();
            let mut n = 0u64;
            b.iter(|| {
                n += 1;
                let k = key.set(n);
                black_box(
                    log.append(&header, black_box(k), black_box(&value))
                        .unwrap(),
                )
            });
        });
    }
    g.finish();
}

/// The same append, but the bytes actually go somewhere, so this includes the
/// page copy the real sink does. The gap between this and the null sink is the
/// cost of moving a page out of the resident window.
fn append_with_pages(c: &mut Criterion) {
    let mut g = c.benchmark_group("append_paged");
    let header = RecordHeader::new(RecordKind::String);
    let value = vec![b'v'; 64];

    for mode in [Durability::None, Durability::Os, Durability::Group] {
        g.bench_function(mode.as_str(), |b| {
            b.iter_batched(
                || Log::new(cfg(mode), MemorySink::new()).unwrap(),
                |mut log| {
                    let mut key = KeyBuf::new();
                    for n in 0..2000u64 {
                        let k = key.set(n);
                        black_box(log.append(&header, k, &value).unwrap());
                    }
                    log
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

/// Reading a record back out of the resident window, which is what every hit
/// that is not in the index's own inline value does.
fn read(c: &mut Criterion) {
    let mut g = c.benchmark_group("read");
    let header = RecordHeader::new(RecordKind::String);
    let value = vec![b'v'; 64];

    let mut log = Log::new(cfg(Durability::None), NullSink).unwrap();
    let mut key = KeyBuf::new();
    let mut addrs = Vec::new();
    for n in 0..20_000u64 {
        let k = key.set(n);
        addrs.push(log.append(&header, k, &value).unwrap().addr);
    }
    // Only the ones still resident. The rest would fault a page, which is the
    // file layer's benchmark and not this one.
    addrs.retain(|a| *a >= log.head());

    g.bench_function("resident", |b| {
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % addrs.len();
            black_box(log.read(black_box(addrs[i])).unwrap().value.len())
        });
    });
    g.finish();
}

criterion_group!(benches, append, append_with_pages, read);
criterion_main!(benches);
