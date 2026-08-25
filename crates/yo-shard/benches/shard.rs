//! Scaling, which is the half of M0's gate that a single threaded benchmark
//! cannot show.
//!
//! The comparison row is aki's raw map: Get 46.5 ns at one core, 20.6 ns at ten
//! cores, Set 49.7 ns. Those are mean per operation times with every core
//! working at once, so that is what is measured here: wall clock for the whole
//! fan out divided by the operations one shard performed.
//!
//! Two shapes.
//!
//! `owned/*` hands each shard a batch of work large enough that the dispatch
//! cost disappears, so what is left is the map. This is the shard per core model
//! with no crossing, and it is the number that should stay flat as cores are
//! added. If it does not stay flat, the memory system is the reason, not the
//! design.
//!
//! `routed/*` pays for the lane and the wakeup on every batch. The gap between
//! the two is the price of crossing a shard boundary, which is worth measuring
//! rather than assuming.
//!
//! Note that a `RawMap` is not `Send`, so there is no way to write this
//! benchmark by handing maps to threads. That is the design working: the only
//! way to reach a shard's map is to send that shard a job.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use yo_index::RawMap;
use yo_shard::Runtime;

const VALUE: &[u8] = b"0123456789abcdef0123456789abcdef";

/// Keys per shard. At 32 bytes of value this is well past any cache, which is
/// the case that matters. Anything smaller measures the L2 and flatters us.
const KEYS: usize = 1_000_000;

/// Operations one shard performs per measured round. Big enough that one job
/// dispatch, which is a few hundred nanoseconds, is under a thousandth of the
/// total.
const ROUND: usize = 100_000;

/// Stride through the keyspace. Coprime with `KEYS`, so it visits everything,
/// and large enough that the hardware prefetcher gets nothing out of it.
const STRIDE: usize = 7919;

fn key(i: usize) -> Vec<u8> {
    let mut buf = KeyBuf::new();
    buf.set(i);
    buf.as_bytes().to_vec()
}

/// A key on the stack, written digit by digit.
///
/// The obvious version of this is `format!("key:{i:012}")` and it was what this
/// file did first, which turned out to measure the formatter and the allocator
/// rather than the map. One `format!` plus one `Vec` is over a hundred
/// nanoseconds, and the operation underneath it is supposed to cost forty. The
/// hot loops fill one of these instead, so the only thing between the clock and
/// the map is the twelve stores below.
struct KeyBuf([u8; 16]);

impl KeyBuf {
    const fn new() -> Self {
        Self(*b"key:000000000000")
    }

    /// Overwrite the twelve digits with `i`, least significant first.
    #[inline]
    fn set(&mut self, i: usize) {
        let mut n = i;
        let mut p = 15;
        while p >= 4 {
            self.0[p] = b'0' + (n % 10) as u8;
            n /= 10;
            p -= 1;
        }
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

fn thread_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut v: Vec<usize> = [1usize, 2, 4, 8, 10, 16, 32]
        .into_iter()
        .filter(|&n| n <= max)
        .collect();
    if v.is_empty() {
        v.push(1);
    }
    v
}

/// A runtime with `shards` shards, each holding `KEYS` keys of its own.
///
/// Every shard gets the same key range on purpose. The point of `owned` is to
/// measure the map without any crossing, so which shard holds which key does
/// not matter, and giving them identical ranges keeps every shard's working set
/// the same size.
fn filled(shards: usize) -> Runtime<RawMap> {
    let rt: Runtime<RawMap> = yo_shard::builder()
        .shards(shards)
        .submitters(shards + 4)
        .build(|_| RawMap::new());
    let sub = rt.submitter();
    for s in 0..shards {
        sub.send(s, |ctx| {
            for i in 0..KEYS {
                ctx.state.set(&key(i), VALUE);
            }
        });
    }
    for s in 0..shards {
        assert_eq!(sub.call(s, |ctx| ctx.state.len()), KEYS);
    }
    rt.release(sub);
    rt
}

/// Give every shard `rounds` rounds of `ROUND` operations and wait for all of
/// them. Returns wall clock for the slowest shard.
fn fan_out<F>(rt: &Runtime<RawMap>, shards: usize, rounds: usize, body: F) -> Duration
where
    F: Fn(&mut RawMap, usize) + Copy + Send + 'static,
{
    let sub = rt.submitter();
    let (tx, rx) = mpsc::channel::<Duration>();
    for s in 0..shards {
        let tx = tx.clone();
        sub.send(s, move |ctx| {
            let start = Instant::now();
            for r in 0..rounds {
                body(&mut ctx.state, s.wrapping_mul(STRIDE).wrapping_add(r));
            }
            let _ = tx.send(start.elapsed());
        });
    }
    drop(tx);
    let slowest = rx.iter().max().unwrap_or_default();
    rt.release(sub);
    slowest
}

fn bench_owned(c: &mut Criterion) {
    let mut g = c.benchmark_group("owned");
    g.throughput(Throughput::Elements(ROUND as u64));
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(10));

    let max = thread_counts().last().copied().unwrap_or(1);
    let rt = filled(max);

    for shards in thread_counts() {
        g.bench_with_input(BenchmarkId::new("get", shards), &shards, |b, &shards| {
            b.iter_custom(|iters| {
                fan_out(&rt, shards, iters as usize, |m, seed| {
                    let mut i = seed % KEYS;
                    let mut k = KeyBuf::new();
                    for _ in 0..ROUND {
                        i += STRIDE;
                        if i >= KEYS {
                            i -= KEYS;
                        }
                        k.set(i);
                        black_box(m.get(black_box(k.as_bytes())));
                    }
                })
            })
        });

        g.bench_with_input(BenchmarkId::new("set", shards), &shards, |b, &shards| {
            b.iter_custom(|iters| {
                fan_out(&rt, shards, iters as usize, |m, seed| {
                    let mut i = seed % KEYS;
                    let mut k = KeyBuf::new();
                    for _ in 0..ROUND {
                        i += STRIDE;
                        if i >= KEYS {
                            i -= KEYS;
                        }
                        k.set(i);
                        black_box(m.set(black_box(k.as_bytes()), VALUE));
                    }
                })
            })
        });
    }
    g.finish();
}

fn bench_routed(c: &mut Criterion) {
    let mut g = c.benchmark_group("routed");
    g.sample_size(10);

    let shards = thread_counts().last().copied().unwrap_or(1);
    let rt = Arc::new(filled(shards));

    // Batched, because one job per command is exactly what `05` section 1.4
    // says not to do.
    const BATCH: usize = yo_common::BATCH_MAX;
    g.throughput(Throughput::Elements(BATCH as u64));

    for threads in thread_counts() {
        g.bench_with_input(BenchmarkId::new("get", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let workers: Vec<_> = (0..threads)
                    .map(|t| {
                        let rt = Arc::clone(&rt);
                        std::thread::spawn(move || {
                            let sub = rt.submitter();
                            let start = Instant::now();
                            let mut i = t * STRIDE;
                            for _ in 0..iters {
                                // One flat array rather than a vector of
                                // vectors. Sixty five allocations per batch
                                // would cost more than the crossing this is
                                // trying to measure.
                                let mut batch = [0u8; 16 * BATCH];
                                let mut k = KeyBuf::new();
                                for slot in 0..BATCH {
                                    i = (i + STRIDE) % KEYS;
                                    k.set(i);
                                    batch[slot * 16..slot * 16 + 16].copy_from_slice(k.as_bytes());
                                }
                                let shard = (t + i) % rt.shards();
                                black_box(sub.call(shard, move |ctx| {
                                    let mut hits = 0u32;
                                    for slot in 0..BATCH {
                                        let k = &batch[slot * 16..slot * 16 + 16];
                                        hits += ctx.state.get(k).is_some() as u32;
                                    }
                                    hits
                                }));
                            }
                            let took = start.elapsed();
                            rt.release(sub);
                            took
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|w| w.join().unwrap())
                    .max()
                    .unwrap_or_default()
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_owned, bench_routed);
criterion_main!(benches);
