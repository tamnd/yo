//! The exact nearest neighbours of every query, worked out the slow way.
//!
//! `examples/recall.rs` measures recall against a ground truth file, and SIFT
//! and GIST ship one. Most datasets do not. An embedding corpus published for
//! retrieval comes with human relevance judgements, which are a different thing
//! entirely: they say which passage answers the question, not which vector is
//! nearest, and an approximate index is only ever going to be measured against
//! what the exact search would have returned.
//!
//! So this is the exact search. It reads the base and query files a dataset
//! directory holds and writes the `groundtruth.ivecs` next to them, in the same
//! format the published ones use, after which `recall.rs` and `drift.rs` cannot
//! tell the difference.
//!
//! ```text
//! cargo run --release -p yo-vector --example truth -- msmarco 100
//! ```
//!
//! The second argument is how many neighbours to record and defaults to 100,
//! which is what SIFT and GIST list. The third is how many threads to use and
//! defaults to every core the machine admits to, because this is the one thing
//! here that is embarrassingly parallel and there is no reason to sit through it
//! on one core.
//!
//! # What it costs
//!
//! Every query against every base vector, so a thousand queries over a million
//! 1024 dimensional vectors is about a trillion multiply adds. That is minutes
//! on a many core machine and it happens once per dataset, which is the trade
//! this makes: pay it here, and every recall run afterwards is comparing against
//! an answer that is exactly right rather than one that is approximately right
//! for reasons nobody wrote down.
//!
//! It refuses to overwrite a ground truth that is already there. Recomputing one
//! is fine and rewriting the published one by accident is not, because the
//! failure that causes is a recall number that looks plausible and is measured
//! against the wrong thing.

use std::io::Write;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: truth <dataset directory> [neighbours] [threads]");
        std::process::exit(2);
    };
    let k: usize = args.next().map_or(100, |a| {
        a.parse().expect("the second argument is a neighbour count")
    });
    let threads: usize = args.next().map_or_else(
        || {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
        },
        |a| a.parse().expect("the third argument is a thread count"),
    );
    let set = prefix(&dir);
    let out = format!("{dir}/{set}_groundtruth.ivecs");
    if std::fs::metadata(&out).is_ok() {
        eprintln!("{out} is already there, and this will not overwrite one");
        std::process::exit(1);
    }

    let t = Instant::now();
    let (dim, base) = read_fvecs(&format!("{dir}/{set}_base.fvecs"));
    let (qdim, query) = read_fvecs(&format!("{dir}/{set}_query.fvecs"));
    assert_eq!(dim, qdim, "base and query dimensions differ");
    let n = base.len() / dim;
    let queries = query.len() / dim;
    assert!(k <= n, "asked for {k} neighbours out of {n} vectors");
    println!(
        "{dir}: {n} base at {dim} dimensions, {queries} query, read in {:?}",
        t.elapsed()
    );
    println!("{k} neighbours each on {threads} threads");

    // One contiguous block of answers, handed out to the threads as disjoint
    // slices, so nothing has to be collected or put back in order afterwards.
    let mut found = vec![0i32; queries * k];
    let t = Instant::now();
    let per = queries.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, rows) in found.chunks_mut(per * k).enumerate() {
            let (base, query) = (&base, &query);
            scope.spawn(move || {
                let mut near = Nearest::new(k);
                for (q, row) in rows.chunks_mut(k).enumerate() {
                    let q = t * per + q;
                    let v = &query[q * dim..(q + 1) * dim];
                    near.clear();
                    for id in 0..n {
                        near.offer(id as i32, sqdist(v, &base[id * dim..(id + 1) * dim]));
                    }
                    near.take_into(row);
                }
            });
        }
    });
    println!("searched in {:?}", t.elapsed());

    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("could not write"));
    let width = (k as i32).to_le_bytes();
    for row in found.chunks(k) {
        file.write_all(&width).expect("write");
        for id in row {
            file.write_all(&id.to_le_bytes()).expect("write");
        }
    }
    file.flush().expect("flush");
    println!("wrote {out}");
}

/// The `k` smallest distances seen so far, worst first.
///
/// A heap would be the textbook answer and is the wrong one at this size. `k` is
/// a hundred, the list is scanned a million times per query, and almost every
/// candidate loses to the worst one already in it and is rejected by a single
/// comparison. So the list is kept sorted and an insert is a linear shift, which
/// happens rarely and touches a hundred contiguous entries when it does.
struct Nearest {
    k: usize,
    /// Distance and id, ordered by distance, nearest first.
    have: Vec<(f32, i32)>,
}

impl Nearest {
    fn new(k: usize) -> Nearest {
        Nearest {
            k,
            have: Vec::with_capacity(k + 1),
        }
    }

    fn clear(&mut self) {
        self.have.clear();
    }

    fn offer(&mut self, id: i32, d: f32) {
        if self.have.len() == self.k && d >= self.have[self.k - 1].0 {
            return;
        }
        let at = self.have.partition_point(|&(had, _)| had < d);
        self.have.insert(at, (d, id));
        self.have.truncate(self.k);
    }

    fn take_into(&self, row: &mut [i32]) {
        for (slot, &(_, id)) in row.iter_mut().zip(&self.have) {
            *slot = id;
        }
    }
}

/// The squared distance between two vectors of the same length.
///
/// Chunked rather than indexed for the reason `src/dist.rs` sets out at length:
/// indexing two slices by one counter leaves a bounds check in the loop body,
/// and a branch in there stops the whole thing vectorising. This is the only
/// arithmetic in the program and it runs a trillion times, so it is worth the
/// four extra lines here rather than a use of an internal module.
fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let (xs, x_tail) = a.as_chunks::<LANES>();
    let (ys, y_tail) = b.as_chunks::<LANES>();

    let mut totals = [0.0f32; LANES];
    for (x, y) in xs.iter().zip(ys) {
        for k in 0..LANES {
            let d = x[k] - y[k];
            totals[k] += d * d;
        }
    }

    let mut sum = 0.0f32;
    for total in totals {
        sum += total;
    }
    for (x, y) in x_tail.iter().zip(y_tail) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// What the files in a directory are called, which every dataset published in
/// this format names after itself.
fn prefix(dir: &str) -> &str {
    dir.trim_end_matches('/')
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(dir)
}

/// One `fvecs` file, as the dimension and every vector end to end.
fn read_fvecs(path: &str) -> (usize, Vec<f32>) {
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
        out.extend(
            rec[4..]
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes),
        );
    }
    (dim, out)
}
