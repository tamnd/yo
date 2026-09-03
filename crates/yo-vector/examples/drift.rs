//! Whether recall holds up under a write stream that never stops.
//!
//! This is the half of M6's ingest gate that a rate does not answer. The claim
//! the partition index makes is not that it is fast to build, it is that it is
//! never rebuilt, and the way an index of this family fails is not by falling
//! over. It measures beautifully on a freshly built corpus and then quietly
//! loses recall over a week of writes as members end up filed under centroids
//! that have moved away from them. A build and a single measurement cannot see
//! that at all, because the measurement happens at the one moment the index is
//! in perfect shape.
//!
//! So this builds SIFT1M once and then rewrites it from the inside for as long
//! as it is told to, measuring recall against the same ground truth every so
//! often. The stream deletes a member and puts it straight back, which means
//! the set of vectors in the collection never changes and the published ground
//! truth stays true, while the index underneath is churned completely: the
//! vector goes to whichever partition is nearest now, partitions fill and split
//! and empty and merge, and LIRE's sweep runs over and over. Nothing is ever
//! rebuilt and nothing is ever reloaded.
//!
//! ```text
//! cargo run --release -p yo-vector --example drift -- sift 24
//! ```
//!
//! The second argument is hours and it defaults to 24, which is what the gate
//! asks for. The row it prints every ten minutes is the whole output: recall
//! at 10 with the gate's probe and rerank, the latencies, how many writes have
//! gone through, and how many partitions there are. Recall is the column that
//! matters and flat is the answer. A slow slide is the failure this is here to
//! catch, and a partition count that climbs without end is the same failure
//! showing up earlier.

use std::time::{Duration, Instant};
use yo_common::Rng;
use yo_vector::{Bits, Partitions, Tuning, Vectors};

/// The base vectors, kept in full precision, which is what the rerank reads and
/// what a real collection would keep in its document log.
struct Base {
    dim: usize,
    data: Vec<f32>,
}

impl Vectors for Base {
    fn get(&self, id: u64, into: &mut [f32]) -> bool {
        let at = id as usize * self.dim;
        let Some(v) = self.data.get(at..at + self.dim) else {
            return false;
        };
        into.copy_from_slice(v);
        true
    }
}

/// How often a row is printed. Ten minutes is short enough that 24 hours is a
/// readable table and long enough that the measuring is a rounding error
/// against the writing.
const EVERY: Duration = Duration::from_secs(600);

/// The configuration the gate is written in.
const PROBE: usize = 64;
const RERANK: usize = 16;
const K: usize = 10;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: drift <sift directory> [hours] [queries]");
        std::process::exit(2);
    };
    let hours: f64 = args.next().map_or(24.0, |a| {
        a.parse().expect("the second argument is a number of hours")
    });
    // A thousand SIFT queries is recall to about half a percent, and this asks
    // the question a hundred and forty odd times, so a slide would show up in
    // the shape of the column long before any one row could be argued with.
    let queries: usize = args.next().map_or(1_000, |a| {
        a.parse()
            .expect("the third argument is a number of queries")
    });

    let t = Instant::now();
    let (dim, data) = read_vecs::<f32>(&format!("{dir}/sift_base.fvecs"));
    let (qdim, query) = read_vecs::<f32>(&format!("{dir}/sift_query.fvecs"));
    let (gdim, truth) = read_vecs::<i32>(&format!("{dir}/sift_groundtruth.ivecs"));
    assert_eq!(dim, qdim, "base and query dimensions differ");
    let n = data.len() / dim;
    let queries = queries.min(query.len() / dim);
    let base = Base { dim, data };
    println!(
        "{dir}: {n} base, {queries} of the queries, read in {:?}",
        t.elapsed()
    );
    println!("holding recall at {K} with probe {PROBE} rerank {RERANK} for {hours} hours");

    let tuning = Tuning {
        probe: PROBE,
        rerank: RERANK,
        ..Tuning::default()
    };
    let mut ix = Partitions::new(dim, Bits::One, 0x51f7, tuning);
    let mut buf = vec![0f32; dim];

    let t = Instant::now();
    for id in 0..n as u64 {
        base.get(id, &mut buf);
        ix.insert(id, &buf);
        if ix.needs_maintenance() {
            ix.maintain(&base, 4);
        }
    }
    println!("built in {:?}, {} partitions", t.elapsed(), ix.partitions());
    println!(
        "{:>8}{:>14}{:>12}{:>12}{:>11}{:>11}",
        "hours", "writes", "partitions", "recall@10", "p50", "p99"
    );

    let start = Instant::now();
    let stop = Duration::from_secs_f64(hours * 3600.0);
    let mut rng = Rng::new(0xD81F7);
    let mut writes = 0u64;
    let mut next = Duration::ZERO;
    loop {
        let now = start.elapsed();
        // The last row is the one the answer is read off, so it is printed
        // whether or not the clock happened to land on a ten minute mark, and a
        // run shorter than one interval still says something.
        let last = now >= stop;
        if now >= next || last {
            row(&ix, &base, &query, &truth, gdim, dim, queries, now, writes);
            next = now + EVERY;
        }
        if last {
            break;
        }
        // A batch between clock reads, because asking the clock is not free and
        // the point of the loop is the writing.
        for _ in 0..10_000 {
            let id = rng.below(n) as u64;
            base.get(id, &mut buf);
            ix.remove(id);
            ix.insert(id, &buf);
            if ix.needs_maintenance() {
                ix.maintain(&base, 4);
            }
        }
        writes += 10_000;
    }
}

/// One line of the table.
#[allow(clippy::too_many_arguments)]
fn row(
    ix: &Partitions,
    base: &Base,
    query: &[f32],
    truth: &[i32],
    gdim: usize,
    dim: usize,
    queries: usize,
    at: Duration,
    writes: u64,
) {
    let mut hit = 0usize;
    let mut took = Vec::with_capacity(queries);
    for q in 0..queries {
        let v = &query[q * dim..(q + 1) * dim];
        let t = Instant::now();
        let got = ix.search(v, K, base);
        took.push(t.elapsed().as_secs_f64() * 1e6);
        let want = &truth[q * gdim..q * gdim + K];
        hit += got.iter().filter(|h| want.contains(&(h.id as i32))).count();
    }
    took.sort_by(f64::total_cmp);
    let quantile = |p: f64| took[((took.len() - 1) as f64 * p) as usize];
    println!(
        "{:>8.2}{writes:>14}{:>12}{:>12.4}{:>9.0} us{:>9.0} us",
        at.as_secs_f64() / 3600.0,
        ix.partitions(),
        hit as f64 / (queries * K) as f64,
        quantile(0.50),
        quantile(0.99)
    );
}

/// One `fvecs` or `ivecs` file, as the dimension and every vector end to end.
///
/// The dimension is repeated in front of every record in this format, and the
/// only sane thing to do with the repeats is check them and drop them. This is
/// the same reader `recall.rs` has, kept separate because an example is meant
/// to be read on its own.
fn read_vecs<T: Le>(path: &str) -> (usize, Vec<T>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    assert!(bytes.len() >= 4, "{path} is too short to hold a dimension");
    let dim = i32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let record = 4 + dim * 4;
    assert!(
        dim > 0 && bytes.len().is_multiple_of(record),
        "{path} is not {dim} dimensional records of {record} bytes"
    );

    let mut out = Vec::with_capacity(bytes.len() / record * dim);
    for (i, rec) in bytes.chunks_exact(record).enumerate() {
        let d = i32::from_le_bytes(rec[..4].try_into().unwrap()) as usize;
        assert_eq!(d, dim, "{path} record {i} has {d} dimensions, not {dim}");
        out.extend(rec[4..].as_chunks::<4>().0.iter().copied().map(T::le));
    }
    (dim, out)
}

/// The four byte little endian element of one of these files.
trait Le {
    fn le(bytes: [u8; 4]) -> Self;
}

impl Le for f32 {
    fn le(bytes: [u8; 4]) -> f32 {
        f32::from_le_bytes(bytes)
    }
}

impl Le for i32 {
    fn le(bytes: [u8; 4]) -> i32 {
        i32::from_le_bytes(bytes)
    }
}
