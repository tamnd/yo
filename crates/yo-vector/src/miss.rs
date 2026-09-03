//! Where a missed neighbour went, which decides what the recall gate needs.
//!
//! Nothing here is on the search path. It is a measurement, kept for the same
//! reason [`rank`](crate::rank) is: the recall gate has more than one plausible
//! fix, they cost very different amounts, and picking between them by argument
//! rather than by measurement is how a month goes missing.
//!
//! # The question
//!
//! G12 wants recall at 10 of 0.95. SIFT1M clears it. MS-MARCO, a million
//! passage embeddings at 1024 dimensions, reaches 0.8942 at probe 128 and is
//! still climbing, so a table that stops there is reporting the end of the loop.
//! The question is what the other 0.1 is made of, and there are exactly two
//! things it can be.
//!
//! A true neighbour that did not come back is either in a partition the search
//! never looked at, or in one it did look at and the estimator ranked it out.
//! Those are different problems. The first is partition quality, and the answer
//! to it is SPANN's boundary replication: a vector near the edge between two
//! partitions gets written into both, so a query that probes either one finds
//! it. The second is estimator quality, and the answer to it is more bits or a
//! wider rerank. Doing the wrong one is a lot of work for nothing, and boundary
//! replication in particular is a change to [`Partitions::at`] and to every path
//! that assumes an id lives in one posting.
//!
//! # What it costs to ask
//!
//! For a query, rank every partition by centroid distance, which is the probe
//! order a search would use. For each of that query's true ten, look up the
//! partition holding it and read off where that partition sits in the order. A
//! neighbour at rank 40 is found by any probe of 64. A neighbour at rank 1200 is
//! not found by any probe anyone would run, and no amount of rerank reaches it.
//!
//! That is one full centroid ranking per query, which is the same pass a search
//! already does, so the whole diagnostic is about as expensive as running the
//! queries once.
//!
//! # Running it
//!
//! It needs a real dataset, because the whole question is about the shape of
//! real data. Synthetic vectors have whatever partition structure the generator
//! gave them and would only measure that.
//!
//! ```text
//! YO_DATASET=$HOME/data/msmarco cargo test --release -p yo-vector \
//!     --lib miss:: -- --ignored --nocapture
//! ```

#![cfg(test)]

use crate::partition::{Partitions, Tuning, Vectors};
use crate::rabitq::Bits;
use std::time::Instant;

/// The base vectors in full precision, which is what the rerank reads.
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

/// A little endian element of one of the two file formats.
trait Le: Copy {
    fn read(b: &[u8]) -> Self;
}

impl Le for f32 {
    fn read(b: &[u8]) -> f32 {
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

impl Le for i32 {
    fn read(b: &[u8]) -> i32 {
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

/// `fvecs` and `ivecs`: per vector, a little endian `i32` dimension and then
/// that many elements. No header and no count, so the length says how many.
fn read_vecs<T: Le>(path: &str) -> (usize, Vec<T>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    assert!(bytes.len() >= 4, "{path} is too short to hold a vector");
    let dim = i32::read(&bytes[..4]) as usize;
    let stride = 4 + dim * 4;
    assert_eq!(
        bytes.len() % stride,
        0,
        "{path} is not a whole number of {dim} dimensional vectors"
    );
    let n = bytes.len() / stride;
    let mut out = Vec::with_capacity(n * dim);
    for v in 0..n {
        let at = v * stride + 4;
        for d in 0..dim {
            out.push(T::read(&bytes[at + d * 4..at + d * 4 + 4]));
        }
    }
    (dim, out)
}

/// The dataset directory names its own files, so `sift/` holds `sift_base.fvecs`.
///
/// Both separators, because this gets run on gamingpc as often as anywhere and
/// `C:\Users\gopher\data\msmarco` has no forward slash in it to split on.
fn prefix(dir: &str) -> &str {
    let dir = dir.trim_end_matches(['/', '\\']);
    dir.rsplit(['/', '\\']).next().unwrap_or(dir)
}

/// Where every true neighbour sits in the probe order, over a slice of the
/// queries. This is the table in the module doc.
#[test]
#[ignore = "needs a dataset in YO_DATASET and prints a table rather than asserting"]
fn how_far_down_the_probe_order_the_answers_are() {
    let Ok(dir) = std::env::var("YO_DATASET") else {
        panic!("set YO_DATASET to a directory holding <name>_base.fvecs and the two beside it");
    };
    // `set YO_DATASET=x && cargo test` on Windows puts the space before the
    // ampersand inside the variable, and the error that causes names a path
    // with a space in the middle of it and takes a while to read.
    let dir = dir.trim().to_string();
    let queries: usize = std::env::var("YO_QUERIES")
        .ok()
        .and_then(|q| q.parse().ok())
        .unwrap_or(500);
    let set = prefix(&dir);

    let t = Instant::now();
    let (dim, data) = read_vecs::<f32>(&format!("{dir}/{set}_base.fvecs"));
    let (qdim, query) = read_vecs::<f32>(&format!("{dir}/{set}_query.fvecs"));
    let (gdim, truth) = read_vecs::<i32>(&format!("{dir}/{set}_groundtruth.ivecs"));
    assert_eq!(dim, qdim, "base and query dimensions differ");
    let n = data.len() / dim;
    let queries = queries.min(query.len() / dim);
    println!(
        "{dir}: {n} base at {dim} dimensions, {queries} queries, read in {:?}",
        t.elapsed()
    );

    let base = Base { dim, data };
    let t = Instant::now();
    let mut ix = Partitions::new(dim, Bits::One, 0x51f7, Tuning::default());
    let mut buf = vec![0f32; dim];
    for id in 0..n as u64 {
        base.get(id, &mut buf);
        ix.insert(id, &buf);
        if ix.needs_maintenance() {
            ix.maintain(&base, 4);
        }
    }
    let parts = ix.partitions();
    println!("{parts} partitions, built in {:?}", t.elapsed());

    // Where each of the true ten sits in the probe order, bucketed by the probe
    // count that would first reach it. The last bucket is the one that matters:
    // those are the neighbours no probe anyone would run can find, and they are
    // the ones boundary replication is for.
    const K: usize = 10;
    let edges = [8usize, 16, 32, 64, 128, 256, 512, 1024];
    let mut bucket = vec![0usize; edges.len() + 1];
    let mut worst = 0usize;
    let mut total = 0usize;
    let mut order = Vec::new();
    let mut place = vec![0usize; parts];
    for q in 0..queries {
        let v = &query[q * dim..(q + 1) * dim];
        ix.probe_order(v, &mut order);
        for (rank, &p) in order.iter().enumerate() {
            place[p] = rank;
        }
        for &id in &truth[q * gdim..q * gdim + K] {
            let Some(p) = ix.holder(id as u64) else {
                continue;
            };
            let rank = place[p] + 1;
            worst = worst.max(rank);
            total += 1;
            let at = edges.iter().position(|&e| rank <= e).unwrap_or(edges.len());
            bucket[at] += 1;
        }
    }

    println!();
    println!("{:>10}{:>12}{:>12}", "found by", "share", "running");
    let mut running = 0usize;
    for (i, &count) in bucket.iter().enumerate() {
        running += count;
        let name = edges
            .get(i)
            .map_or("beyond".to_string(), |e| format!("probe {e}"));
        println!(
            "{name:>10}{:>11.4}{:>12.4}",
            count as f64 / total as f64,
            running as f64 / total as f64
        );
    }
    println!();
    println!("furthest a true neighbour sat down the order: {worst} of {parts}");
}
