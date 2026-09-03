//! Where a query's time goes, split into the three things a query is.
//!
//! Nothing here is on the search path. It is a measurement, kept for the same
//! reason [`rank`](crate::rank) and [`miss`](crate::miss) are: G12 is a latency
//! gate as well as a recall one, the query has three parts that grow at
//! different rates, and picking which one to work on by reading the code rather
//! than by timing it is how a week goes into the part that was already fast.
//!
//! # The three parts
//!
//! Ranking the centroids. Every partition's centroid is measured against the
//! query, all of them, every time. It does not depend on the probe count at
//! all, so it is a fixed cost per query, and it grows with the collection
//! because the partition count does. [`coarse`](crate::coarse) says why a tree
//! over the centroids is not the answer and [`rank`](crate::rank) says why
//! coding them is not either.
//!
//! Preparing the query, once for every partition probed. A code is written
//! against the centroid of the partition it belongs to, so the query has to be
//! expressed against that same centroid before the two can be compared, and
//! that is a residual, a normalisation and a transpose into bit planes. It is
//! linear in the probe count and linear in the dimension.
//!
//! Scanning the postings. The estimator against every member of every partition
//! probed, which is linear in the probe count and in the posting size and in
//! the dimension. This is the part everyone assumes is the whole query.
//!
//! # The answer
//!
//! Per query over 200 queries at a probe of 32, with the parts measured by
//! adding one to the last rather than in isolation, because a search does them
//! in that order and the caches see them in that order:
//!
//! ```text
//!                       768 dims, 60k    1024 dims, 200k
//!  ranking centroids          13.8 us            52.0 us
//!  preparing the query        54.2 us            75.0 us
//!  scanning the postings     144.6 us           195.9 us
//!  the whole search          217.0 us           311.1 us
//! ```
//!
//! Those are off a busy laptop and the absolute numbers are worth nothing. The
//! split between the three rows is what the file is for and it holds up: the
//! same run before the change below had preparing the query at 3.5 microseconds
//! a partition rather than 1.6, and the whole search at 268 microseconds rather
//! than 218.
//!
//! Preparing the query was a quarter of the whole thing, which is more than it
//! has any business being: it is one pass over a few kilobytes against a scan
//! that reads a hundred times as much.
//!
//! What was wrong with it was the order of one loop. Quantising the query
//! writes each coordinate's level down as a bit in each of four planes, and the
//! planes are one word of the query apart, so doing it a coordinate at a time
//! is a read, an or and a write back into four separate places for every
//! coordinate and nothing stays in a register. Sixty four coordinates at a time
//! keeps those four words in registers for the whole run and stores each once.
//! `level_code` on the encode path had always done it that way and the query
//! path had not.
//!
//! # What is left
//!
//! The scan is two thirds of a query and it measures out at 12 nanoseconds a
//! member at 768 dimensions and 18 at 1024, which is what `benches/rabitq.rs`
//! says one code costs, so there is no overhead hiding in the loop around it.
//! Making a query faster from here is a question of scanning fewer members
//! rather than of scanning them faster, which is what the probe count and the
//! posting size are, and what boundary replication is trying to buy.
//!
//! # Running it
//!
//! The shapes above are generated, which is fine here in a way it is not for
//! recall: how long a scan takes does not depend on whether the vectors mean
//! anything. Point it at a real collection with `YO_DATASET` when the question
//! is about a real one.
//!
//! ```text
//! cargo test --release -p yo-vector --lib probe:: -- --ignored --nocapture
//! YO_DATASET=$HOME/data/msmarco cargo test --release -p yo-vector \
//!     --lib probe:: -- --ignored --nocapture
//! ```

#![cfg(test)]

use crate::dist::sqdist;
use crate::partition::{Partitions, Tuning, Vectors};
use crate::rabitq::Bits;
use std::time::Instant;
use yo_common::Rng;

/// How many partitions a query reads here.
const PROBE: usize = 32;

/// How many queries each number is the mean of.
const QUERIES: usize = 200;

/// The full precision vectors, which the rerank reads and the build sweeps.
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

/// Clustered vectors with a few coordinates carrying more than their share,
/// which is the shape that makes a partitioning do any work. The timings do not
/// care much, but a corpus that lands everything in one partition would measure
/// a search nobody runs.
fn generated(dim: usize, n: usize, clusters: usize, seed: u64) -> Base {
    let mut rng = Rng::new(seed);
    let draw = |rng: &mut Rng| -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
                let heavy = if i < dim / 16 { 6.0 } else { 1.0 };
                (u * 2.0 - 1.0) * heavy
            })
            .collect()
    };
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(&mut rng)).collect();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let off = draw(&mut rng);
        for (c, o) in centres[i % clusters].iter().zip(&off) {
            data.push(c + o * 0.7);
        }
    }
    Base { dim, data }
}

/// The index over a base, built the way a shard builds one.
fn build(base: &Base) -> Partitions {
    let n = base.data.len() / base.dim;
    let tuning = Tuning {
        probe: PROBE,
        ..Tuning::default()
    };
    let mut ix = Partitions::new(base.dim, Bits::One, 0x51f7, tuning);
    let mut buf = vec![0f32; base.dim];
    for id in 0..n as u64 {
        base.get(id, &mut buf);
        ix.insert(id, &buf);
        if ix.needs_maintenance() {
            ix.maintain(base, 4);
        }
    }
    ix
}

/// The table in the module doc.
#[test]
#[ignore = "prints a table rather than asserting anything"]
fn where_a_query_spends_its_time() {
    let bases: Vec<Base> = match std::env::var("YO_DATASET") {
        Ok(dir) => vec![from_dataset(dir.trim())],
        Err(_) => vec![
            generated(768, 60_000, 24, 5),
            generated(1024, 200_000, 24, 5),
        ],
    };
    for base in &bases {
        let dim = base.dim;
        let n = base.data.len() / dim;
        let at = Instant::now();
        let ix = build(base);
        let parts = ix.partitions();
        let held: usize = (0..parts).map(|p| ix.posting_parts(p).0.len()).sum();
        println!(
            "{dim} dims, {n} vectors, {parts} partitions, {held} entries, built in {:?}",
            at.elapsed()
        );

        // Queries the collection has seen, because what is being timed is the
        // work and not the answers, and a query drawn from the corpus lands in
        // a populated part of it the way a real one does.
        let queries: Vec<&[f32]> = (0..QUERIES)
            .map(|i| {
                let at = (i * 7919 % n) * dim;
                &base.data[at..at + dim]
            })
            .collect();
        let each = |d: std::time::Duration| d.as_nanos() as f64 / QUERIES as f64 / 1000.0;
        let mut sink = 0f32;

        let mut order = Vec::new();
        let at = Instant::now();
        for q in &queries {
            ix.probe_order(q, &mut order);
            sink += order[0] as f32;
        }
        let rank = each(at.elapsed());

        let centroids = ix.all_centroids();
        let at = Instant::now();
        for q in &queries {
            ix.probe_order(q, &mut order);
            let u = ix.quantizer().rotate(q);
            for &p in order.iter().take(PROBE) {
                let prepared = ix
                    .quantizer()
                    .query_rotated(&u, &centroids[p * dim..(p + 1) * dim]);
                let (_, _, codes, meta, _) = ix.posting_parts(p);
                sink += prepared.distance(&codes[..ix.quantizer().code_bytes()], &meta[0]);
            }
        }
        let prep = each(at.elapsed());

        let mut scores: Vec<f32> = Vec::new();
        let at = Instant::now();
        for q in &queries {
            ix.probe_order(q, &mut order);
            let u = ix.quantizer().rotate(q);
            for &p in order.iter().take(PROBE) {
                let prepared = ix
                    .quantizer()
                    .query_rotated(&u, &centroids[p * dim..(p + 1) * dim]);
                let (_, _, codes, meta, _) = ix.posting_parts(p);
                if scores.len() < meta.len() {
                    scores.resize(meta.len(), 0.0);
                }
                prepared.scan(codes, meta, &mut scores[..meta.len()]);
                sink += scores[0];
            }
        }
        let scan = each(at.elapsed());

        let members = PROBE * held / parts;
        println!("  ranking centroids   {rank:8.1} us");
        println!(
            "  preparing the query {:8.1} us  ({:.1} us a partition)",
            prep - rank,
            (prep - rank) / PROBE as f64
        );
        println!(
            "  scanning the postings {:6.1} us  ({:.1} ns a member over {members})",
            scan - prep,
            (scan - prep) * 1000.0 / members as f64
        );
        println!("  and the whole search  {:6.1} us  ({sink})", {
            let at = Instant::now();
            for q in &queries {
                sink += ix.candidates(q, 10)[0].1;
            }
            each(at.elapsed())
        });
    }
}

/// A dataset directory holding `<name>_base.fvecs`, read the way
/// [`miss`](crate::miss) reads one.
fn from_dataset(dir: &str) -> Base {
    let set = dir
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(dir);
    let path = format!("{dir}/{set}_base.fvecs");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    assert!(bytes.len() >= 4, "{path} is too short to hold a vector");
    let dim = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let stride = 4 + dim * 4;
    assert_eq!(bytes.len() % stride, 0, "{path} is not whole vectors");
    let mut data = Vec::with_capacity(bytes.len() / stride * dim);
    for v in 0..bytes.len() / stride {
        for d in 0..dim {
            let at = v * stride + 4 + d * 4;
            data.push(f32::from_le_bytes([
                bytes[at],
                bytes[at + 1],
                bytes[at + 2],
                bytes[at + 3],
            ]));
        }
    }
    Base { dim, data }
}

/// The centroid ranking on its own, so the fixed cost in the table above can be
/// read against the arithmetic it is: one squared distance per partition.
#[test]
#[ignore = "prints a number rather than asserting anything"]
fn what_one_centroid_costs() {
    for &dim in &[128usize, 768, 1024] {
        let base = generated(dim, 4000, 24, 9);
        let n = base.data.len() / dim;
        let q = &base.data[..dim];
        let at = Instant::now();
        let mut sink = 0f32;
        for _ in 0..100 {
            for p in 0..n {
                sink += sqdist(q, &base.data[p * dim..(p + 1) * dim]);
            }
        }
        let each = at.elapsed().as_nanos() as f64 / 100.0 / n as f64;
        println!("{dim:5} dimensions {each:6.1} ns a centroid  ({sink})");
    }
}
