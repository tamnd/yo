//! What the vector index recalls on a dataset somebody else made.
//!
//! The recall table in `partition.rs` is measured on vectors this repo
//! generates, and the cold form in `yo-graph` just taught me what that is worth:
//! the synthetic number said 9.89 bits an edge and the public graph said 19.62.
//! So the M6 exit gate, recall at 10 of 0.95 or better with p99 at or under a
//! millisecond, gets measured on SIFT1M, which is the dataset every vector index
//! published since 2010 reports against.
//!
//! ```text
//! curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//! tar xf sift.tar.gz
//! cargo run --release -p yo-vector --example recall -- sift
//! ```
//!
//! The directory wants `sift_base.fvecs`, `sift_query.fvecs` and
//! `sift_groundtruth.ivecs` in it, which is what the tarball unpacks to. That is
//! a million base vectors at 128 dimensions, ten thousand queries, and the true
//! hundred nearest neighbours of each query by exact L2, which is what recall is
//! measured against.
//!
//! The three names come from the directory, so anything else published this way
//! runs without a change. GIST1M is the same tarball one directory along, a
//! million vectors at 960 dimensions, and it is the harder one: a quantiser
//! that holds at 128 dimensions of SIFT descriptors is not thereby known to
//! hold at 960.
//!
//! `fvecs` and `ivecs` are the same layout with a different element type: for
//! each vector, a little endian `i32` dimension followed by that many `f32` or
//! `i32`. There is no header and no count, so the file length divided by the
//! record length is the number of vectors.
//!
//! Both knobs that matter are swept rather than picked, because the useful
//! output is the curve. `probe` is how many partitions a search scans and it is
//! the one that buys recall; `rerank` is how many candidates get measured
//! properly per answer. A single row would only say whether the default happens
//! to clear the gate, and the curve says how much headroom there is and what it
//! costs.
//!
//! On a sanity run over 32 dimensional Gaussian noise, which is the worst thing
//! you can hand a quantiser, the two knobs behave completely differently. One
//! bit recall goes up with `rerank` and does not go up with `probe` at all: 0.43
//! at probe 8 and 0.45 at probe 128, against 0.53 at rerank 8. That is the
//! estimator being the ceiling rather than the scan, and it is why `rerank` is
//! swept as far as it is here. Four bit on the same data is at 0.97 by probe 32
//! and identical at every `rerank`, which is the same fact from the other side:
//! once the ranking is nearly exact, widening the rerank only adds candidates
//! that were never going to place.
//!
//! The probe sweep stops at 128 unless a third argument says otherwise, and that
//! argument exists because of MS-MARCO. On SIFT the curve has flattened by 128
//! and the top of the loop is past the answer, so nothing was ever lost by
//! ending there. On a million MS-MARCO passage embeddings at 1024 dimensions
//! recall is still climbing at 128, and a table that stops there reports where
//! the loop ended and calls it a ceiling. So:
//!
//! ```text
//! cargo run --release -p yo-vector --example recall -- msmarco 500 1024
//! ```

use std::time::Instant;
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

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: recall <dataset directory> [queries] [highest probe]");
        std::process::exit(2);
    };
    // Ten thousand queries at a millisecond each is ten seconds a row and there
    // are a lot of rows, so the default is a slice of them. Recall over a
    // thousand SIFT queries is stable to about half a percent.
    let queries: usize = args.next().map_or(1_000, |a| {
        a.parse()
            .expect("the second argument is a number of queries")
    });
    // Where the sweep stops. A dataset the index finds easy is finished well
    // before 128 probes and one it does not is only getting interesting there,
    // so the top is an argument rather than a number in the loop. MS-MARCO is
    // what made it one: recall on it is still climbing at 128, and reporting the
    // end of the sweep as the ceiling would have been reporting the loop.
    let top: usize = args.next().map_or(128, |a| {
        a.parse()
            .expect("the third argument is the highest probe count")
    });
    let set = prefix(&dir);

    let t = Instant::now();
    let (dim, base) = read_vecs::<f32>(&format!("{dir}/{set}_base.fvecs"));
    let (qdim, query) = read_vecs::<f32>(&format!("{dir}/{set}_query.fvecs"));
    let (gdim, truth) = read_vecs::<i32>(&format!("{dir}/{set}_groundtruth.ivecs"));
    assert_eq!(dim, qdim, "base and query dimensions differ");
    let n = base.len() / dim;
    let queries = queries.min(query.len() / dim);
    println!(
        "{dir}: {n} base, {} query, {gdim} true neighbours each, read in {:?}",
        query.len() / dim,
        t.elapsed()
    );

    let bench = Bench {
        base: Base { dim, data: base },
        query,
        truth,
        gdim,
        dim,
        queries,
    };
    let base = &bench.base;

    for bits in [Bits::One, Bits::Four] {
        let t = Instant::now();
        let mut ix = Partitions::new(dim, bits, 0x51f7, Tuning::default());
        let mut buf = vec![0f32; dim];
        for id in 0..n as u64 {
            base.get(id, &mut buf);
            ix.insert(id, &buf);
            // The index is asked to keep itself in shape as it goes, which is
            // the whole SPFresh claim and the thing a build that batches at the
            // end would quietly skip.
            if ix.needs_maintenance() {
                ix.maintain(base, 4);
            }
        }
        let built = t.elapsed();
        let rate = n as f64 / built.as_secs_f64();
        println!();
        println!(
            "{bits:?} bit, {} partitions, built in {built:?}, {:.0} vectors a second on one core",
            ix.partitions(),
            rate
        );
        println!(
            "{:>6}{:>8}{:>12}{:>11}{:>11}",
            "probe", "rerank", "recall@10", "p50", "p99"
        );

        let mut probe = 8;
        while probe <= top {
            for rerank in [4usize, 8, 16, 32] {
                let mut t = ix.tuning();
                t.probe = probe;
                t.rerank = rerank;
                ix.retune(t);
                measure(&ix, &bench, probe, rerank);
            }
            probe *= 2;
        }
    }
}

/// The dataset, so that one row of the sweep is a call rather than nine
/// arguments.
struct Bench {
    base: Base,
    query: Vec<f32>,
    truth: Vec<i32>,
    /// How many true neighbours the ground truth lists per query, which is a
    /// hundred for SIFT and is not the `k` anything is measured at.
    gdim: usize,
    dim: usize,
    queries: usize,
}

fn measure(ix: &Partitions, b: &Bench, probe: usize, rerank: usize) {
    const K: usize = 10;
    let mut hit = 0usize;
    let mut took = Vec::with_capacity(b.queries);
    for q in 0..b.queries {
        let v = &b.query[q * b.dim..(q + 1) * b.dim];
        let t = Instant::now();
        let got = ix.search(v, K, &b.base);
        took.push(t.elapsed().as_secs_f64() * 1e6);
        // Recall at ten is how many of the true ten came back, and the true ten
        // are the first ten of the hundred the ground truth lists.
        let want = &b.truth[q * b.gdim..q * b.gdim + K];
        hit += got.iter().filter(|h| want.contains(&(h.id as i32))).count();
    }
    took.sort_by(f64::total_cmp);
    let at = |p: f64| took[((took.len() - 1) as f64 * p) as usize];
    println!(
        "{probe:>6}{rerank:>8}{:>12.4}{:>9.0} us{:>9.0} us",
        hit as f64 / (b.queries * K) as f64,
        at(0.50),
        at(0.99)
    );
}

/// What the three files in a directory are called, which every dataset
/// published in this format names after itself.
///
/// `sift/` holds `sift_base.fvecs` and `gist/` holds `gist_base.fvecs`, so the
/// last component of the path is the prefix and there is nothing to pass. A
/// trailing slash is dropped rather than turned into an empty prefix, because
/// tab completion puts one there.
fn prefix(dir: &str) -> &str {
    dir.trim_end_matches('/')
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(dir)
}

/// One `fvecs` or `ivecs` file, as the dimension and every vector end to end.
///
/// The dimension is repeated in front of every record in this format, and the
/// only sane thing to do with the repeats is check them and drop them.
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
///
/// Both types are read the same way and neither is worth an unaligned pointer
/// cast to save, because this runs once at startup and the file is on a disk.
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
