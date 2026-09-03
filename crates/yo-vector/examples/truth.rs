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
//!
//! # Checking it against one somebody else made
//!
//! The way to trust this is to run it on a dataset that already has an answer
//! and compare, so it was run on SIFT1M under a different prefix and the result
//! held up against `sift_groundtruth.ivecs`. What that comparison says is worth
//! writing down, because the obvious way to read it is wrong.
//!
//! Only 4446 of the ten thousand rows are the same hundred ids in the same
//! order. That number looks like a bug and is not one. Work the distances out
//! properly and all ten thousand rows carry the same hundred distances as the
//! published file, in the same order, every time, with nothing further away
//! than it could be on either side. The two files differ only in which of two
//! vectors at exactly the same distance gets written down first.
//!
//! That comparison is what `--check` does, on a dataset that already has a
//! ground truth next to it:
//!
//! ```text
//! cargo run --release -p yo-vector --example truth -- sift 100 --check
//! ```
//!
//! There is a lot of that in SIFT because the descriptors are byte valued, so
//! distances are small integers and equal ones are common rather than a freak
//! event. This program breaks a tie towards the lower id, since candidates are
//! offered in id order and an equal distance does not displace one already
//! held. Whatever produced the published file broke it some other way. Both
//! answers are exactly right, which is the point: a ground truth is a set of
//! distances, and comparing two of them by id is comparing the tie break.
//!
//! So the check to run on a new dataset is the distance one, and the id
//! agreement is not a number to chase. Recall is unaffected either way, because
//! a search that returns a vector at the true nearest distance has found a true
//! nearest neighbour no matter which id the file happens to list.

use std::io::Write;
use std::time::Instant;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let check = args.iter().any(|a| a == "--check");
    args.retain(|a| a != "--check");
    let mut args = args.into_iter();
    let Some(dir) = args.next() else {
        eprintln!("usage: truth <dataset directory> [neighbours] [threads] [--check]");
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
    let there = std::fs::metadata(&out).is_ok();
    if there && !check {
        eprintln!("{out} is already there, and this will not overwrite one");
        eprintln!("pass --check to compare against it instead");
        std::process::exit(1);
    }
    if check && !there {
        eprintln!("{out} is not there, so there is nothing to check against");
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

    if check {
        compare(&out, &found, &base, &query, dim, k);
        return;
    }

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

/// What a ground truth already on disk says, against what this just worked out.
///
/// The comparison is on distances and not on ids, for the reason the module
/// header gives at length: two exact answers to the same question differ
/// wherever two vectors sit at exactly the same distance, and on a dataset with
/// integer coordinates that is most rows. So the id agreement is printed
/// because it is the number everybody reaches for first, and then the question
/// that actually decides it is asked underneath.
///
/// A row passes when the file's hundred distances are the same hundred numbers
/// in the same order as this program's. A row fails when they are not, and then
/// which way it fails is the interesting part. A file listing something further
/// away than it could have is a file that was built with an approximate search,
/// and a file listing something nearer means this program has a bug.
fn compare(out: &str, found: &[i32], base: &[f32], query: &[f32], dim: usize, k: usize) {
    let (fk, theirs) = read_ivecs(out);
    assert!(
        fk >= k,
        "{out} lists {fk} neighbours a query and this worked out {k}"
    );
    let queries = found.len() / k;
    assert_eq!(
        theirs.len() / fk,
        queries,
        "{out} covers a different number of queries"
    );

    let at = |row: &[i32], q: usize| -> Vec<f64> {
        let v = &query[q * dim..(q + 1) * dim];
        row.iter()
            .map(|&id| {
                let id = id as usize;
                sqdist(v, &base[id * dim..(id + 1) * dim])
            })
            .collect()
    };

    let (mut same_ids, mut same_dist, mut theirs_worse, mut mine_worse) = (0, 0, 0, 0);
    let mut first = None;
    for q in 0..queries {
        let mine = &found[q * k..(q + 1) * k];
        let file = &theirs[q * fk..q * fk + k];
        if mine == file {
            same_ids += 1;
            same_dist += 1;
            continue;
        }
        let (a, b) = (at(mine, q), at(file, q));
        if a == b {
            same_dist += 1;
        } else {
            if first.is_none() {
                first = Some(q);
            }
            // The first place they part company is the one that says which
            // list is wrong, because both are sorted and everything before it
            // agrees.
            let split = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(0);
            if a[split] < b[split] {
                theirs_worse += 1;
            } else {
                mine_worse += 1;
            }
        }
    }

    println!("{queries} queries against {out}, at {k} neighbours each");
    println!("  same ids in the same order: {same_ids}");
    println!("  same distances in the same order: {same_dist}");
    println!("  the file has something further away than it could: {theirs_worse}");
    println!("  this program has something nearer than the file: {mine_worse}");
    if let Some(q) = first {
        println!("  first query whose distances differ: {q}");
    }
    if same_dist == queries {
        println!("agreed on every query, and the ids differ only where the distances tie");
    } else {
        std::process::exit(1);
    }
}

/// The `k` smallest distances seen so far, nearest first.
///
/// A heap would be the textbook answer and is the wrong one at this size. `k` is
/// a hundred, the list is scanned a million times per query, and almost every
/// candidate loses to the worst one already in it and is rejected by a single
/// comparison. So the list is kept sorted and an insert is a linear shift, which
/// happens rarely and touches a hundred contiguous entries when it does.
struct Nearest {
    k: usize,
    /// Distance and id, ordered by distance, nearest first.
    have: Vec<(f64, i32)>,
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

    /// Offer a candidate, which is kept if it is near enough and dropped if it
    /// is not.
    ///
    /// Both comparisons hold the same line on a tie, which is that whatever is
    /// already in the list stays in front of an equal newcomer. Candidates
    /// arrive in id order, so that comes out as the lower id winning, and it is
    /// worth being deliberate about because equal distances are common on any
    /// dataset with integer coordinates and the tie break is the only thing
    /// that ever separates one correct ground truth from another.
    fn offer(&mut self, id: i32, d: f64) {
        if self.have.len() == self.k && d >= self.have[self.k - 1].0 {
            return;
        }
        let at = self.have.partition_point(|&(had, _)| had <= d);
        self.have.insert(at, (d, id));
        self.have.truncate(self.k);
    }

    fn take_into(&self, row: &mut [i32]) {
        for (slot, &(_, id)) in row.iter_mut().zip(&self.have) {
            *slot = id;
        }
    }
}

/// The squared distance between two vectors of the same length, in double.
///
/// The index measures in single precision and that is the right call there,
/// because a distance is only ever compared with another one and the answer
/// almost never turns on the last bit. A ground truth is the one place it can,
/// so this pays for the wider accumulator once per dataset and stops thinking
/// about it. The difference of two `f32` and its square are both exact in
/// `f64`, so the only error left is in the summation, and over a thousand
/// dimensions of real embedding values that is the error worth removing.
///
/// It makes no difference on SIFT, and it is worth knowing why, because SIFT is
/// what this gets validated against. SIFT descriptors are byte valued, the
/// largest coordinate in the million is 218, and the largest squared distance a
/// pair of them can reach is under 2^23. Every distance in that dataset is a
/// small integer, `f32` holds it exactly, and both accumulators give bit
/// identical answers. A single precision run and a double precision run of this
/// program produce the same ten thousand rows.
///
/// See the module header for what those rows do and do not have in common with
/// the published file.
///
/// Chunked rather than indexed for the reason `src/dist.rs` sets out at length:
/// indexing two slices by one counter leaves a bounds check in the loop body,
/// and a branch in there stops the whole thing vectorising. This is the only
/// arithmetic in the program and it runs a trillion times, so it is worth the
/// four extra lines here rather than a use of an internal module.
fn sqdist(a: &[f32], b: &[f32]) -> f64 {
    const LANES: usize = 8;
    let (xs, x_tail) = a.as_chunks::<LANES>();
    let (ys, y_tail) = b.as_chunks::<LANES>();

    let mut totals = [0.0f64; LANES];
    for (x, y) in xs.iter().zip(ys) {
        for k in 0..LANES {
            let d = f64::from(x[k]) - f64::from(y[k]);
            totals[k] += d * d;
        }
    }

    let mut sum = 0.0f64;
    for total in totals {
        sum += total;
    }
    for (x, y) in x_tail.iter().zip(y_tail) {
        let d = f64::from(*x) - f64::from(*y);
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

/// One `ivecs` file, which is the same layout holding ids instead of
/// coordinates and is only read by `--check`.
///
/// The records are parsed by the reader above and the four bytes of each
/// element are then read the other way round. `to_bits` hands back exactly the
/// bytes `from_le_bytes` was given, so this is the same file read twice and not
/// a conversion, and it beats a second copy of the same parse loop differing in
/// one line.
fn read_ivecs(path: &str) -> (usize, Vec<i32>) {
    let (dim, floats) = read_fvecs(path);
    let ids = floats.into_iter().map(|f| f.to_bits() as i32).collect();
    (dim, ids)
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
