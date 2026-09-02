//! O-G1 and P-8, the pre-registered experiment about what a cross shard
//! traversal actually costs.
//!
//! `11` section 4 states the cost of a multi hop traversal as one batch period
//! per hop, because the frontier is exchanged through the same intake queues
//! commands use and every hop is a barrier. O-G1 asks the obvious follow up:
//! whether the barrier can be dropped, so that a shard which has finished its
//! share of hop `k` starts hop `k+1` on the fragment it just produced instead
//! of waiting for the slowest shard. P-8 predicts that dropping it does not
//! help below four hops.
//!
//! Both strategies are here, on the same graph, doing the same expansions.
//!
//! `barriered/*` is the design as written. The driver holds the frontier,
//! buckets it by owning shard, sends one job per shard, waits for every reply,
//! merges, and goes again. Every hop is a fan out and a join.
//!
//! `wavefront/*` has no driver in the loop and no join. A shard expands the
//! fragment it was handed, buckets what came out by owning shard, and forwards
//! each piece straight down the lane to that shard, which starts on it as soon
//! as it gets to it. Shards are at different hops at the same time on purpose.
//! An atomic counts fragments in flight and the last one out wakes the caller.
//!
//! Dedup is exact in both, and for the same reason: a node lives on exactly one
//! shard, so the shard that owns it is the only place its visited stamp can be,
//! and a per shard stamp is a global one. What differs is that the barrier is
//! not only synchronisation. It is also what makes the walk breadth first. Drop
//! it and a node can be claimed by a wave that is a hop deeper before the
//! shallower wave arrives, and then everything behind that node is explored one
//! hop short, which is a different answer rather than a slower one. The first
//! version of this benchmark did exactly that and the two strategies disagreed
//! at four hops. So the wavefront keeps the remaining hop count next to the
//! stamp and expands a node again when a shallower claim turns up. That
//! repeated work is not overhead this benchmark could have avoided, it is what
//! dropping the barrier costs.
//!
//! Both are checked against each other on reach before either is timed.
//!
//! `roundtrip` is the batch period on this box, measured as an empty job sent
//! to a shard and waited for. It is the unit the two strategies should be read
//! in.
//!
//! Neither strategy pools its fragments, so both allocate a vector per shard
//! per hop. That is the same tax on both sides and it is well under the lane
//! crossing, but it is the first thing to fix if this ever becomes a hot path
//! rather than a measurement.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::Thread;
use std::time::Duration;
use yo_graph::{Adjacency, Dir};
use yo_shard::{Runtime, ShardCtx, Submitter};

/// The one edge label in the graph. A traversal that filters by label is the
/// same walk with a smaller run, and the run lookup is what is being timed.
const LABEL: u32 = 1;

/// Whether this is CI checking the benchmark still runs rather than a box
/// measuring something.
fn smoke() -> bool {
    std::env::var_os("YO_BENCH_SMOKE").is_some()
}

/// Nodes in the graph.
///
/// A quarter of a million at degree eight is two million edges, which is around
/// twenty five megabytes of adjacency spread over the shards. Big enough that
/// the runs are not sitting in L2 and small enough that a six hop walk from one
/// node reaches all of it, which is what makes the last hop the interesting
/// one.
fn nodes() -> u64 {
    if smoke() { 4_096 } else { 1 << 18 }
}

/// Out degree, uniform. Eight so the frontier multiplies by eight a hop and
/// saturates the graph at six, which puts a narrow hop and a wide one in the
/// same sweep.
fn degree() -> u64 {
    8
}

/// Hop counts to measure. P-8 puts its line at four, so the range has to have
/// both sides of it in it.
fn hop_counts() -> [usize; 6] {
    [1, 2, 3, 4, 5, 6]
}

/// Shard counts to measure. One is the control: it runs the same driver with
/// nothing to cross, so the difference is the crossing.
fn shard_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut v: Vec<usize> = [1usize, 4, 8].into_iter().filter(|&n| n <= max).collect();
    if v.is_empty() {
        v.push(1);
    }
    v
}

/// The `i`th neighbour of `node`, in a graph of `n` nodes.
///
/// Generated rather than stored, so every shard can fill its own slice without
/// anything being shipped to it, and so the graph is the same graph whatever
/// the shard count is. That last part is what makes the shard counts
/// comparable. It is splitmix over the pair, which is more mixing than a random
/// graph needs but costs nothing at build time.
fn neighbour(node: u64, i: u64, n: u64) -> u64 {
    let mut x = node
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(i.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 29;
    x % n
}

/// One shard's slice of the graph.
///
/// Node `id` belongs to shard `id % shards`, so the slice is a stride rather
/// than a range. A stride is the unfriendly case on purpose: a range would put
/// most of a random walk's neighbours on the shard it started from and the
/// benchmark would be measuring a local traversal wearing a shard's clothes.
struct Part {
    shards: u64,
    adj: Adjacency,
    /// Visited stamp per owned node, indexed by `id / shards`. A generation
    /// number rather than a bit, so a new traversal costs nothing to start
    /// instead of clearing a bitset the size of the shard.
    seen: Vec<u32>,
    /// Hops still to go when this node was last expanded, alongside the stamp.
    ///
    /// The barriered walk does not need this, because it takes the whole
    /// frontier a level at a time and so always reaches a node at its shallowest
    /// depth first. The wavefront walk does, and it is the whole reason the two
    /// are not the same algorithm with a different barrier. Without a barrier a
    /// node can be claimed by a deeper wave before a shallower one arrives, and
    /// then everything behind it is explored one hop short. Recording the depth
    /// and expanding again when a shallower claim turns up is what makes the two
    /// return the same answer, and the repeated work is the price of dropping
    /// the barrier rather than an artefact of this benchmark.
    left: Vec<u8>,
    /// This shard's own lanes, installed once the runtime exists. Only the
    /// wavefront strategy uses it, and only to forward a fragment onwards.
    out: Option<Submitter<Part>>,
}

impl Part {
    fn new(shards: u64, n: u64) -> Part {
        let owned = n.div_ceil(shards) as usize;
        Part {
            shards,
            adj: Adjacency::out_only(),
            seen: vec![0; owned],
            left: vec![0; owned],
            out: None,
        }
    }

    /// Fill in the runs for the nodes this shard owns.
    fn fill(&mut self, me: u64, n: u64, d: u64) {
        let mut node = me;
        while node < n {
            for i in 0..d {
                self.adj.link(node, neighbour(node, i, n), LABEL, 0);
            }
            node += self.shards;
        }
    }

    /// Expand `frag`, which has `left` hops still to go, and bucket what comes
    /// out by the shard that owns it.
    ///
    /// A node is expanded if this walk has not reached it, or if it has but
    /// with fewer hops in hand than this fragment carries. Returns how many
    /// nodes were reached for the first time, which is the reach of the walk
    /// and does not count a node that had to be expanded twice.
    fn expand(&mut self, walk: u32, left: usize, frag: &[u64]) -> (usize, Vec<Vec<u64>>) {
        let mut out = vec![Vec::new(); self.shards as usize];
        let mut fresh = 0;
        for &node in frag {
            let slot = (node / self.shards) as usize;
            let first = self.seen[slot] != walk;
            if !first && usize::from(self.left[slot]) >= left {
                continue;
            }
            self.seen[slot] = walk;
            self.left[slot] = left as u8;
            fresh += usize::from(first);
            for &next in self.adj.neighbours(node, LABEL, Dir::Out) {
                out[(next % self.shards) as usize].push(next);
            }
        }
        (fresh, out)
    }
}

/// The traversal generation, bumped once per walk so no stamp is ever reused.
static GENERATION: AtomicU32 = AtomicU32::new(1);

fn next_walk() -> u32 {
    GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// A graph of [`nodes`] nodes at [`degree`] out edges, spread over `shards`
/// shards, with every shard holding a submitter of its own.
fn build(shards: usize) -> Runtime<Part> {
    let (n, d) = (nodes(), degree());
    let rt: Runtime<Part> = yo_shard::builder()
        .shards(shards)
        .submitters(2 * shards + 4)
        .build(move |_| Part::new(shards as u64, n));

    let sub = rt.submitter();
    for s in 0..shards {
        sub.send(s, move |ctx| ctx.state.fill(s as u64, n, d));
    }
    // One submitter per shard, so every lane keeps its single producer. They
    // are never released, which is why the pool is built oversized above.
    for s in 0..shards {
        let mine = rt.submitter();
        sub.send(s, move |ctx| ctx.state.out = Some(mine));
    }
    let edges: usize = (0..shards)
        .map(|s| sub.call(s, |ctx| ctx.state.adj.edges()))
        .sum();
    assert_eq!(edges as u64, n * d);
    rt.release(sub);
    rt
}

/// The design as written: fan out per hop, join per hop.
fn barriered(sub: &Submitter<Part>, start: u64, hops: usize) -> usize {
    let shards = sub.shards();
    let walk = next_walk();
    let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); shards];
    buckets[(start % shards as u64) as usize].push(start);

    let mut visited = 0;
    for hop in 0..hops {
        let left = hops - hop;
        let (tx, rx) = mpsc::channel();
        let mut live = 0;
        for (shard, bucket) in buckets.iter_mut().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let frag = std::mem::take(bucket);
            let tx = tx.clone();
            sub.send(shard, move |ctx| {
                let _ = tx.send(ctx.state.expand(walk, left, &frag));
            });
            live += 1;
        }
        drop(tx);
        if live == 0 {
            break;
        }
        for (fresh, out) in rx {
            visited += fresh;
            for (dest, part) in out.into_iter().enumerate() {
                buckets[dest].extend(part);
            }
        }
    }
    visited
}

/// What one wavefront walk has left to do, and who to wake when it is nothing.
struct Wave {
    /// Fragments either in a lane or being expanded. The walk is over when this
    /// reaches zero, and it can only reach zero once, because a fragment adds
    /// its successors before it subtracts itself.
    left: AtomicUsize,
    visited: AtomicUsize,
    waiter: Thread,
}

/// No driver in the loop: a shard forwards straight to the next shard.
fn wavefront(sub: &Submitter<Part>, start: u64, hops: usize) -> usize {
    let shards = sub.shards();
    let walk = next_walk();
    let wave = Arc::new(Wave {
        left: AtomicUsize::new(1),
        visited: AtomicUsize::new(0),
        waiter: std::thread::current(),
    });

    let first = (start % shards as u64) as usize;
    let seed = Arc::clone(&wave);
    sub.send(first, move |ctx| step(ctx, walk, vec![start], hops, seed));

    while wave.left.load(Ordering::Acquire) != 0 {
        std::thread::park();
    }
    wave.visited.load(Ordering::Relaxed)
}

/// One fragment, expanded and forwarded, on the shard that owns it.
fn step(ctx: &mut ShardCtx<Part>, walk: u32, frag: Vec<u64>, left: usize, wave: Arc<Wave>) {
    let (fresh, out) = ctx.state.expand(walk, left, &frag);
    wave.visited.fetch_add(fresh, Ordering::Relaxed);

    if left > 1 {
        // Count before sending. A successor that finishes early must not find
        // the counter at zero, and this fragment has not subtracted itself yet.
        let live = out.iter().filter(|v| !v.is_empty()).count();
        if live > 0 {
            wave.left.fetch_add(live, Ordering::AcqRel);
            let lanes = ctx.state.out.as_ref().expect("the shard has no submitter");
            for (dest, part) in out.into_iter().enumerate() {
                if part.is_empty() {
                    continue;
                }
                let wave = Arc::clone(&wave);
                lanes.send(dest, move |ctx| step(ctx, walk, part, left - 1, wave));
            }
        }
    }

    if wave.left.fetch_sub(1, Ordering::AcqRel) == 1 {
        wave.waiter.unpark();
    }
}

/// A start node on shard zero for every run, so no strategy gets a head start
/// from where its walk began.
fn start_at(i: u64) -> u64 {
    i.wrapping_mul(2_654_435_761) % nodes()
}

fn bench_hops(c: &mut Criterion) {
    for shards in shard_counts() {
        let rt = build(shards);
        let sub = rt.submitter();

        let mut g = c.benchmark_group(format!("hop/{shards}"));
        g.sample_size(20);
        g.measurement_time(Duration::from_secs(if smoke() { 1 } else { 8 }));

        for hops in hop_counts() {
            // The two strategies have to agree on the reach of the walk before
            // either of their times means anything. This is the only place the
            // dedup argument gets checked rather than asserted in prose.
            let (a, b) = (
                barriered(&sub, start_at(hops as u64), hops),
                wavefront(&sub, start_at(hops as u64), hops),
            );
            assert_eq!(a, b, "the two strategies disagree at {hops} hops");

            let mut i = 0u64;
            g.bench_with_input(BenchmarkId::new("barriered", hops), &hops, |b, &hops| {
                b.iter(|| {
                    i += 1;
                    black_box(barriered(&sub, start_at(i), hops))
                })
            });
            let mut i = 0u64;
            g.bench_with_input(BenchmarkId::new("wavefront", hops), &hops, |b, &hops| {
                b.iter(|| {
                    i += 1;
                    black_box(wavefront(&sub, start_at(i), hops))
                })
            });
        }
        g.finish();
        rt.release(sub);
    }
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut g = c.benchmark_group("roundtrip");
    g.sample_size(50);
    for shards in shard_counts() {
        let rt = build(shards);
        let sub = rt.submitter();
        g.bench_with_input(BenchmarkId::from_parameter(shards), &shards, |b, _| {
            b.iter(|| black_box(sub.call(0, |ctx| ctx.state.shards)))
        });
        rt.release(sub);
    }
    g.finish();
}

criterion_group!(benches, bench_hops, bench_roundtrip);
criterion_main!(benches);
