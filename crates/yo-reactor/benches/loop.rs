//! What the loop costs, and what the two walks buy.
//!
//! Three things are measured, and they answer three different questions.
//!
//! `inline/*` is one command through [`Reactor::execute`], which is the number
//! G6 is about: in process p50 under 150 ns through the public path, epoch and
//! all. There is no batch here, so there is no prefetch to help, and this is
//! the honest cost of a single embedded call.
//!
//! `batch/*` is 64 commands through [`Reactor::execute_all`], with the engine's
//! prefetch either doing its job or turned into a no-op. The gap between the
//! two arms is the entire argument for walking a batch twice, and it should
//! open up as the map outgrows the caches: at 100 thousand keys the index is
//! around L3 and there is not much to hide, at 10 million every probe is a trip
//! to DRAM and the misses have somewhere to overlap.
//!
//! `loop/*` is the same 64 commands through [`Reactor::tick`], so it carries
//! the intake queue, the epoch, the flush and the maintenance check on top. The
//! difference between this and `batch/warm` is what the server path pays over
//! the embedded one, and Y23 says that difference has to stay small.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use yo_index::RawMap;
use yo_reactor::{BATCH_MAX, Engine, Flow, Reactor};
use yo_shard::Epochs;
use yo_shard::spsc::{Sender, lane};

/// Sizes chosen against the cache hierarchy, same as `yo-index` uses. Prefetch
/// cannot help what is already in L2, so a bench that only ran the small size
/// would conclude the second walk is dead weight.
const SIZES: [usize; 2] = [100_000, 10_000_000];

/// The sizes to actually run.
///
/// CI runs every benchmark once to check that it still builds and still runs,
/// and gets nothing out of filling a ten million key map three times to look
/// something up in it once. `YO_BENCH_SMOKE` is that run, and anything
/// measuring anything leaves it unset.
fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &[1_000]
    } else {
        &SIZES
    }
}

/// Commands per measured iteration, which is one full batch.
const BATCH: usize = BATCH_MAX;

fn key(i: usize) -> Vec<u8> {
    format!("key:{i:012}").into_bytes()
}

/// A shard that does one thing: look a key up and count whether it was there.
///
/// Work is an index into the key table rather than a key, so the benchmark is
/// not measuring its own allocator. The hash is precomputed for the same
/// reason, and because that is what the real path does too: the parser has the
/// key, the loop hashes it once on the first walk.
struct Lookup {
    map: RawMap,
    keys: Vec<Vec<u8>>,
    hashes: Vec<u64>,
    /// Whether the first walk really asks for the line.
    warm: bool,
    found: u64,
    flushes: u64,
}

impl Lookup {
    fn new(n: usize, warm: bool) -> Lookup {
        let mut map = RawMap::new();
        for i in 0..n {
            map.set(&key(i), b"0123456789abcdef0123456789abcdef");
        }
        // A stride that is coprime with the size, so the batch walks the map
        // rather than one corner of it, and no two commands in a batch share a
        // bucket by accident.
        let keys: Vec<Vec<u8>> = (0..1024).map(|i| key(i * 7919 % n)).collect();
        let hashes = keys.iter().map(|k| RawMap::hash_of(k)).collect();
        Lookup {
            map,
            keys,
            hashes,
            warm,
            found: 0,
            flushes: 0,
        }
    }
}

impl Engine for Lookup {
    type Work = u32;

    fn key_hash(&self, work: &u32) -> Option<u64> {
        Some(self.hashes[*work as usize])
    }

    fn prefetch(&self, hash: u64) {
        if self.warm {
            self.map.prefetch(hash);
        }
    }

    fn run(&mut self, work: u32, hash: Option<u64>) -> Flow {
        let k = &self.keys[work as usize];
        if self.map.get_hashed(hash.unwrap(), k).is_some() {
            self.found += 1;
        }
        Flow::Next
    }

    fn flush(&mut self) {
        self.flushes += 1;
    }
}

/// The batch of work indices, spread across the 1024 probe keys.
fn batch(round: usize) -> Vec<u32> {
    (0..BATCH)
        .map(|i| ((round * BATCH + i * 13) % 1024) as u32)
        .collect()
}

fn bench_inline(c: &mut Criterion) {
    let mut g = c.benchmark_group("inline");
    g.throughput(Throughput::Elements(1));

    for &n in sizes() {
        let mut r = Reactor::inline(Lookup::new(n, true));
        g.bench_with_input(BenchmarkId::new("execute", n), &n, |bench, _| {
            let mut i = 0u32;
            bench.iter(|| {
                i = (i + 1) & 1023;
                black_box(r.execute(black_box(i)))
            });
        });
    }

    g.finish();
}

fn bench_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("batch");
    g.throughput(Throughput::Elements(BATCH as u64));

    for &n in sizes() {
        // One map for both arms. Filling ten million keys twice to change one
        // boolean is a minute of a benchmark run spent on nothing.
        let mut r = Reactor::inline(Lookup::new(n, true));
        let rounds: Vec<Vec<u32>> = (0..16).map(batch).collect();
        for (name, warm) in [("warm", true), ("cold", false)] {
            r.engine_mut().warm = warm;
            g.bench_with_input(BenchmarkId::new(name, n), &n, |bench, _| {
                let mut round = 0usize;
                bench.iter(|| {
                    round = (round + 1) & 15;
                    black_box(r.execute_all(rounds[round].iter().copied()))
                });
            });
        }
    }

    g.finish();
}

fn bench_loop(c: &mut Criterion) {
    let mut g = c.benchmark_group("loop");
    g.throughput(Throughput::Elements(BATCH as u64));

    for &n in sizes() {
        // The lane is deep enough that a push never fails, and the batch is
        // pushed inside the measured closure because on the server path
        // somebody always pays for the handoff. It is the same cost in every
        // arm, so it does not move the comparison.
        let (tx, rx): (Sender<u32>, _) = lane(4096);
        let epochs = Epochs::new(1);
        let mut r = Reactor::new(Lookup::new(n, true), 0, Arc::clone(&epochs), vec![rx]);
        let rounds: Vec<Vec<u32>> = (0..16).map(batch).collect();

        g.bench_with_input(BenchmarkId::new("tick", n), &n, |bench, _| {
            let mut round = 0usize;
            bench.iter(|| {
                round = (round + 1) & 15;
                for w in &rounds[round] {
                    tx.push(*w).unwrap();
                }
                black_box(r.tick().unwrap())
            });
        });
    }

    g.finish();
}

criterion_group!(benches, bench_inline, bench_batch, bench_loop);
criterion_main!(benches);
