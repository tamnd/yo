//! Making the centroid ranking pass cheaper, because it is what a query pays
//! before it has looked at anything.
//!
//! Nothing here is on the search path yet. It is a measurement, kept the way
//! [`rank`](crate::rank) and [`miss`](crate::miss) are kept, and it exists
//! because the cost it is about only shows up at a size the small runs never
//! reach.
//!
//! # The cost
//!
//! A search ranks every centroid on every query. [`coarse`](crate::coarse) says
//! why a tree over them is not the answer and [`rank`](crate::rank) says why
//! coding them the way the members are coded is not either, so the pass stays,
//! and it is a flat vectorised read of the whole table. That was cheap while the
//! table was small. On a million MS-MARCO passages the table is 7719 centroids
//! at 768 dimensions, and ranking it is about half a millisecond on a 13900K
//! against a G12 budget of one millisecond for the entire query. It does not
//! depend on the probe count, so it is there even at probe 8, and it is most of
//! why an MS-MARCO query has a floor of about two milliseconds under it however
//! few partitions it reads.
//!
//! # What the cost is not
//!
//! [`rank`](crate::rank) reads the pass as a bandwidth problem and says it is
//! already going as fast as the memory will go. That is the right conclusion
//! from the wrong reason, and the wrong reason is what sent this module down a
//! blind alley first, so both are written down.
//!
//! The table times the scan on its own, against a table held warm and against a
//! run of copies far too big for any cache, on this laptop:
//!
//! ```text
//!                                  warm scan          cold scan   prefix
//! centroids   dims       MB    full  bf16    i8    full  bf16    i8   1/6  +rerank    head
//!      1930    768      5.7     107   112   145     111   120   148    19       63       4
//!      2963   1024     11.6     236   242   292     237   247   305    40      112       5
//!      7719    768     22.6     438   451   608     436   451   573    76      147      11
//!     30000    768     87.9    1772  1816  2343    1717  1799  2299   306      403      40
//! ```
//!
//! Cold is warm, at every size, including one that cannot possibly have stayed
//! resident. The pass reads in a straight line, so the prefetcher hides the
//! whole latency and the loop runs at the speed the arithmetic allows and not
//! the speed the memory allows. Twenty two megabytes in 438 microseconds reads
//! as fifty gigabytes a second, which is not a number a single core gets out of
//! DRAM, and that is the tell.
//!
//! So making a centroid smaller does nothing. bfloat16 halves the bytes and is
//! not faster. An i8 quarters them and is slower, because unpacking a byte into
//! a float is work and work is the thing there is too much of. Both are kept
//! below as the record of it.
//!
//! # What does work, and why it is still not worth doing
//!
//! If the pass is a multiply and an add a coordinate, the only way to cut it is
//! fewer coordinates rather than fewer bytes each. The centroids are already
//! stored rotated, and a random rotation spreads a vector's length evenly over
//! its coordinates, so any fixed subset of them is a random projection with
//! nothing to train and nothing extra to store: the first `keep` columns of the
//! table that is already there. What comes back is the true distance scaled by
//! `keep / dim` plus noise, and a scale does not change an ordering, so nothing
//! needs correcting before the head is picked.
//!
//! A sixth of the coordinates on its own is too rough, at five per cent further
//! on a head of 128 at 768 dimensions. Taking four times the head that way and
//! putting those in order on the full coordinates brings it to one per cent
//! further and still runs in 147 microseconds against 449, which is a real three
//! times. The rerank is the part [`rank`](crate::rank) turned down for reading
//! scattered rows, and that was the bandwidth argument again, so it was turned
//! down for a reason that does not hold.
//!
//! It is still not worth doing, and this is the useful part. The fixed cost
//! only looks large next to a small posting scan, and a small posting scan is
//! the configuration that does not reach the recall the gate wants. On
//! MS-MARCO at 7719 partitions the ranking is about half a millisecond of a
//! query that tops out at 0.905 recall however long it is given. The
//! configuration that does reach the gate, 1809 partitions with a posting
//! target of 1024, hits 0.9504 at probe 128, and there the whole ranking pass is
//! about a tenth of a millisecond out of sixteen. Three times cheaper on a
//! tenth of a millisecond is not what stands between here and a p99 of one
//! millisecond. The posting scan is.
//!
//! # What it costs to be wrong
//!
//! No approximation of a centroid can lose a neighbour unless it moves the head.
//! So the measure is not the distances, it is which partitions come back in the
//! first `probe` of them, and it is specifically not how many of them the exact
//! pass would also have picked. Centroids in a cluster sit at nearly the same
//! distance, ties are everywhere, and swapping two partitions that are equally
//! near costs nothing: the search reads one instead of the other and what it was
//! looking for is as likely to be in either. What costs something is a head that
//! is further away than it needed to be. That is what
//! [`a_rougher_centroid_pass_picks_a_head_that_is_just_as_near`] asserts, and
//! [`what_a_rougher_centroid_pass_gives_away`] asks the same question of a real
//! collection in the only terms recall can see, which is
//! [`miss`](crate::miss)'s ceiling.
//!
//! # Where this leaves it
//!
//! The prefix and rerank is written out and measured here rather than wired in,
//! because it is a three times saving on a pass that is a fraction of the query
//! at the size that matters and it would be complexity bought with nothing. It
//! is worth coming back to if the partition count ever climbs again, which the
//! 30000 row shows costs 1.8 milliseconds by itself, and the two numbers to come
//! back for are in the table above.
//!
//! # Running it
//!
//! The head quality half runs anywhere. The timing half wants a quiet machine.
//!
//! ```text
//! cargo test --release -p yo-vector --lib narrow:: -- --ignored --nocapture
//! YO_DATASET=$HOME/data/sift cargo test --release -p yo-vector \
//!     --lib narrow:: -- --ignored --nocapture
//! ```

#![cfg(test)]

use crate::dist::sqdist;
use std::time::Instant;
use yo_common::Rng;

/// How many differences are accumulated side by side, which is
/// [`dist`](crate::dist)'s reason and the same eight.
const LANES: usize = 8;

// ------------------------------------------------------- fewer bytes, rejected

/// The centroid table with the bottom half of every float dropped.
///
/// Rounded to nearest rather than truncated. Truncation is one instruction
/// cheaper and biases every coordinate towards zero, which is a bias the sum
/// does not average out because it is the same sign in every term.
fn to_bf16(centroids: &[f32]) -> Vec<u16> {
    centroids
        .iter()
        .map(|&x| {
            // The rounding can carry into the exponent, which is right, the way
            // rounding 9.99 to one place carries into the units. It wraps rather
            // than adds because the very top of the range is a NaN, which a
            // centroid never is, and a panic there would be a strange way to
            // find that out.
            let bits = x.to_bits();
            let up = bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1);
            (up >> 16) as u16
        })
        .collect()
}

/// The squared distance between a full precision query and a bfloat16 centroid.
fn bf16_sqdist(q: &[f32], c: &[u16]) -> f32 {
    let n = q.len().min(c.len());
    let (xs, x_tail) = q[..n].as_chunks::<LANES>();
    let (ys, _) = c[..n].as_chunks::<LANES>();

    let mut totals = [0.0f32; LANES];
    for (x, y) in xs.iter().zip(ys) {
        for k in 0..LANES {
            let d = x[k] - f32::from_bits(u32::from(y[k]) << 16);
            totals[k] += d * d;
        }
    }

    let mut sum = 0.0f32;
    for total in totals {
        sum += total;
    }
    let from = n - x_tail.len();
    for (k, x) in x_tail.iter().enumerate() {
        let d = x - f32::from_bits(u32::from(c[from + k]) << 16);
        sum += d * d;
    }
    sum
}

/// The centroid table as one signed byte a coordinate, with the frame it was
/// written against.
///
/// Per dimension, because a centroid table's spread is not the same in every
/// dimension and a code is worth more where the spread is wider.
struct Bytes {
    dim: usize,
    /// The middle of the spread in each dimension.
    mid: Vec<f32>,
    /// What one step of a code is worth in each dimension.
    step: Vec<f32>,
    codes: Vec<i8>,
}

impl Bytes {
    fn build(centroids: &[f32], dim: usize) -> Bytes {
        let n = centroids.len() / dim;
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for p in 0..n {
            for (d, &x) in centroids[p * dim..(p + 1) * dim].iter().enumerate() {
                lo[d] = lo[d].min(x);
                hi[d] = hi[d].max(x);
            }
        }
        let mid: Vec<f32> = lo.iter().zip(&hi).map(|(a, b)| (a + b) / 2.0).collect();
        // A dimension where every centroid agrees has no spread to divide by,
        // and the smallest step there is right: every code comes out nought and
        // reconstructs to the middle, which is the exact value.
        let step: Vec<f32> = lo
            .iter()
            .zip(&hi)
            .map(|(a, b)| ((b - a) / 254.0).max(f32::MIN_POSITIVE))
            .collect();
        let mut codes = Vec::with_capacity(n * dim);
        for p in 0..n {
            for (d, &x) in centroids[p * dim..(p + 1) * dim].iter().enumerate() {
                let at = ((x - mid[d]) / step[d]).round().clamp(-127.0, 127.0);
                #[expect(clippy::cast_possible_truncation, reason = "clamped on the line above")]
                codes.push(at as i8);
            }
        }
        Bytes {
            dim,
            mid,
            step,
            codes,
        }
    }

    /// The query in the frame the codes are in, so the inner loop is a weighted
    /// difference of two numbers on the same scale.
    ///
    /// The weight is the step squared, because one code of difference in a
    /// dimension is one step of difference in the value and the distance wants
    /// the square of it.
    fn prepare(&self, q: &[f32], at: &mut Vec<f32>, weight: &mut Vec<f32>) {
        at.clear();
        weight.clear();
        for ((x, mid), step) in q.iter().zip(&self.mid).zip(&self.step) {
            at.push((x - mid) / step);
            weight.push(step * step);
        }
    }

    fn sqdist(&self, at: &[f32], weight: &[f32], p: usize) -> f32 {
        let c = &self.codes[p * self.dim..(p + 1) * self.dim];
        let (xs, x_tail) = at.as_chunks::<LANES>();
        let (ws, _) = weight.as_chunks::<LANES>();
        let (ys, _) = c.as_chunks::<LANES>();

        let mut totals = [0.0f32; LANES];
        for ((x, w), y) in xs.iter().zip(ws).zip(ys) {
            for k in 0..LANES {
                let d = x[k] - f32::from(y[k]);
                totals[k] += d * d * w[k];
            }
        }

        let mut sum = 0.0f32;
        for total in totals {
            sum += total;
        }
        let from = self.dim - x_tail.len();
        for (k, x) in x_tail.iter().enumerate() {
            let d = x - f32::from(c[from + k]);
            sum += d * d * weight[from + k];
        }
        sum
    }
}

// ------------------------------------------------------------ fewer dimensions

/// The first `keep` coordinates of every centroid, packed together.
///
/// Packed rather than read out of the full table with a stride, because a
/// strided read still pulls in the whole cache line and would save none of the
/// traffic. It saves the arithmetic either way, and the traffic is not the
/// binding cost, but there is no reason to give up the one that is free.
struct Prefix {
    keep: usize,
    rows: Vec<f32>,
}

impl Prefix {
    fn build(centroids: &[f32], dim: usize, keep: usize) -> Prefix {
        let keep = keep.min(dim);
        let n = centroids.len() / dim;
        let mut rows = Vec::with_capacity(n * keep);
        for p in 0..n {
            rows.extend_from_slice(&centroids[p * dim..p * dim + keep]);
        }
        Prefix { keep, rows }
    }

    fn sqdist(&self, q: &[f32], p: usize) -> f32 {
        sqdist(
            &q[..self.keep],
            &self.rows[p * self.keep..(p + 1) * self.keep],
        )
    }

    fn count(&self) -> usize {
        self.rows.len() / self.keep
    }

    /// The `want` nearest, on the prefix alone.
    fn head(&self, q: &[f32], want: usize) -> Vec<usize> {
        head(self.count(), want, |p| self.sqdist(q, p))
    }

    /// The `want` nearest, taking `widen` times as many on the prefix and then
    /// putting those in the right order on the full coordinates.
    ///
    /// [`rank`](crate::rank) turned down a shortlist and a rerank because the
    /// rerank reads scattered rows, which was a bandwidth argument and the
    /// bandwidth is not the constraint. What the rerank actually costs is its
    /// own arithmetic, and there is little of it, because the shortlist is a
    /// few hundred rows out of thousands.
    fn head_reranked(
        &self,
        q: &[f32],
        centroids: &[f32],
        dim: usize,
        want: usize,
        widen: usize,
    ) -> Vec<usize> {
        let short = self.head(q, want * widen);
        let mut by: Vec<(usize, f32)> = short
            .into_iter()
            .map(|p| (p, sqdist(q, &centroids[p * dim..(p + 1) * dim])))
            .collect();
        by.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        by.truncate(want);
        by.into_iter().map(|(p, _)| p).collect()
    }
}

// ------------------------------------------------------------------- the head

/// The `want` nearest partitions, nearest first, by whatever `score` says a
/// partition is worth.
fn head(n: usize, want: usize, score: impl Fn(usize) -> f32) -> Vec<usize> {
    let order = |a: &(usize, f32), b: &(usize, f32)| a.1.total_cmp(&b.1);
    let mut by: Vec<(usize, f32)> = (0..n).map(|p| (p, score(p))).collect();
    let want = want.min(n);
    by.select_nth_unstable_by(want.saturating_sub(1), order);
    by.truncate(want);
    by.sort_unstable_by(order);
    by.into_iter().map(|(p, _)| p).collect()
}

/// Centroids the way a real collection makes them, which is
/// [`rank`](crate::rank)'s generator and the same reasoning: uniform noise has
/// no cluster structure and cluster structure is what makes ranking hard.
fn clumped(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut unit = || (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
    let groups = 24;
    let hubs: Vec<f32> = (0..groups * dim).map(|_| unit()).collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let h = (i % groups) * dim;
        for d in 0..dim {
            out.push(hubs[h + d] + (unit() - 0.5) * 0.2);
        }
    }
    out
}

/// How long a run took, in microseconds a query.
fn timed(queries: usize, run: impl FnOnce()) -> f64 {
    let t = Instant::now();
    run();
    t.elapsed().as_nanos() as f64 / queries as f64 / 1000.0
}

// ----------------------------------------------------------------- the answers

/// Whether a rougher pass picks a head as near as the exact one does.
///
/// The ratio is the true full precision distance summed over the partitions the
/// rough pass chose, against the same sum over the partitions the exact pass
/// chose. One means the two heads are the same distance away and the search can
/// have either. How many partitions they have in common is printed beside it to
/// show how little it has to do with the answer.
#[test]
fn a_rougher_centroid_pass_picks_a_head_that_is_just_as_near() {
    for dim in [128usize, 768] {
        let n = 3000;
        let centroids = clumped(n, dim, 2);
        let queries = clumped(60, dim, 3);
        let halves = to_bf16(&centroids);
        let bytes = Bytes::build(&centroids, dim);
        let prefix = Prefix::build(&centroids, dim, dim / 6);
        let (mut at, mut weight) = (Vec::new(), Vec::new());

        // How far the prefix has to be widened before reranking takes back what
        // it gave away, which is the number that decides whether any of it is
        // worth having.
        for keep in [dim / 6, dim / 3, dim / 2] {
            let rough = Prefix::build(&centroids, dim, keep);
            for widen in [1usize, 4, 16] {
                let (mut far, mut base) = (0.0f64, 0.0f64);
                for i in 0..60 {
                    let q = &queries[i * dim..(i + 1) * dim];
                    let truly =
                        |p: &usize| f64::from(sqdist(q, &centroids[p * dim..(p + 1) * dim]));
                    let full = head(n, 128, |p| sqdist(q, &centroids[p * dim..(p + 1) * dim]));
                    let got = if widen == 1 {
                        rough.head(q, 128)
                    } else {
                        rough.head_reranked(q, &centroids, dim, 128, widen)
                    };
                    far += got.iter().map(truly).sum::<f64>();
                    base += full.iter().map(truly).sum::<f64>();
                }
                println!(
                    "  {dim:4} dimensions, keep {keep:4}, widen {widen:2}: {:.4} as far",
                    far / base
                );
            }
        }

        for want in [32usize, 128] {
            let mut far = [0.0f64; 5];
            let mut kept = [0usize; 5];
            let mut total = 0usize;
            for i in 0..60 {
                let q = &queries[i * dim..(i + 1) * dim];
                let truly = |p: &usize| f64::from(sqdist(q, &centroids[p * dim..(p + 1) * dim]));
                let full = head(n, want, |p| sqdist(q, &centroids[p * dim..(p + 1) * dim]));
                let half = head(n, want, |p| bf16_sqdist(q, &halves[p * dim..(p + 1) * dim]));
                bytes.prepare(q, &mut at, &mut weight);
                let byte = head(n, want, |p| bytes.sqdist(&at, &weight, p));
                let short = prefix.head(q, want);
                let two = prefix.head_reranked(q, &centroids, dim, want, 4);
                for (k, picked) in [&full, &half, &byte, &short, &two].into_iter().enumerate() {
                    far[k] += picked.iter().map(truly).sum::<f64>();
                    kept[k] += picked.iter().filter(|p| full.contains(p)).count();
                }
                total += want;
            }
            let cost = |k: usize| far[k] / far[0];
            let common = |k: usize| kept[k] as f64 / total as f64;
            println!(
                "{dim:4} dimensions, head of {want:3}: \
                 bf16 {:.4}/{:.3}  i8 {:.4}/{:.3}  \
                 prefix {:.4}/{:.3}  prefix and rerank {:.4}/{:.3}  (as far / in common)",
                cost(1),
                common(1),
                cost(2),
                common(2),
                cost(3),
                common(3),
                cost(4),
                common(4)
            );
            // Fewer bytes a coordinate keeps the head exactly, which is the
            // half of this that works and buys nothing, because the pass is not
            // short of bytes.
            assert!(cost(1) < 1.001, "bf16 head is {:.4} as far", cost(1));
            assert!(cost(2) < 1.01, "i8 head is {:.4} as far", cost(2));
            // Fewer coordinates is a real approximation and a loose one. It is
            // asserted loosely on purpose: the point of the number is that it is
            // not near one, and that a rerank narrow enough to be worth having
            // does not bring it back to one either.
            assert!(
                cost(3) > 1.02,
                "a sixth of the coordinates was expected to be rough"
            );
            assert!(cost(3) < 1.30, "prefix head is {:.4} as far", cost(3));
            assert!(
                cost(4) < cost(3),
                "reranking a prefix head did not make it nearer"
            );
        }
    }
}

/// What each pass costs to rank the same centroids, back to back on one machine.
/// This is the table in the module doc.
///
/// Three things are kept apart, because running them together is what makes the
/// answer look like a memory question when it is not one.
///
/// The scan is the distance to every centroid and nothing else, which is the
/// part any of this can help. Picking the head out of the scores afterwards is
/// the same work whatever the scan was, so it is timed once, and it is a floor
/// none of this gets under. Cold is the same scan over a run of copies far too
/// big to have stayed in any cache, and warm is one table read over and over. A
/// real query is nearer warm, because the centroids are the hottest thing it
/// touches, but cold is the one that says whether the pass is bandwidth bound.
#[test]
#[ignore = "prints a table rather than asserting anything"]
fn what_a_rougher_centroid_pass_costs() {
    let runs = 200;
    let want = 128;
    let mut sink = 0.0f32;
    let mut kept = 0usize;

    println!("                                 warm scan          cold scan   prefix");
    println!(
        "centroids   dims       MB    full  bf16    i8    full  bf16    i8   1/6  +rerank    head"
    );
    for &(n, dim) in &[
        (1930usize, 768usize),
        (2963, 1024),
        (7719, 768),
        (30000, 768),
    ] {
        let mb = (n * dim * 4) as f64 / (1 << 20) as f64;
        let centroids = clumped(n, dim, 2);
        let queries = clumped(runs, dim, 3);
        let halves = to_bf16(&centroids);
        let bytes = Bytes::build(&centroids, dim);
        let prefix = Prefix::build(&centroids, dim, dim / 6);
        let (mut at, mut weight) = (Vec::new(), Vec::new());

        // Enough copies to be past any last level cache going, so that reading
        // them in turn never finds one still resident.
        let copies = (256.0 / mb).ceil() as usize + 1;
        let cold: Vec<Vec<f32>> = (0..copies)
            .map(|s| clumped(n, dim, 20 + s as u64))
            .collect();
        let cold_half: Vec<Vec<u16>> = cold.iter().map(|c| to_bf16(c)).collect();
        let cold_byte: Vec<Bytes> = cold.iter().map(|c| Bytes::build(c, dim)).collect();

        let each = |i: usize| &queries[i * dim..(i + 1) * dim];

        // A warm pass over everything first, because the first read of a page
        // pays for the page and that is not what is being measured.
        for i in 0..runs {
            let q = each(i);
            for p in 0..n {
                sink += sqdist(q, &centroids[p * dim..(p + 1) * dim]);
                sink += bf16_sqdist(q, &halves[p * dim..(p + 1) * dim]);
                sink += prefix.sqdist(q, p);
            }
            bytes.prepare(q, &mut at, &mut weight);
            sink += bytes.sqdist(&at, &weight, i % n);
        }

        let warm_full = timed(runs, || {
            for i in 0..runs {
                let q = each(i);
                for p in 0..n {
                    sink += sqdist(q, &centroids[p * dim..(p + 1) * dim]);
                }
            }
        });
        let warm_half = timed(runs, || {
            for i in 0..runs {
                let q = each(i);
                for p in 0..n {
                    sink += bf16_sqdist(q, &halves[p * dim..(p + 1) * dim]);
                }
            }
        });
        let warm_byte = timed(runs, || {
            for i in 0..runs {
                let q = each(i);
                bytes.prepare(q, &mut at, &mut weight);
                for p in 0..n {
                    sink += bytes.sqdist(&at, &weight, p);
                }
            }
        });

        let cold_f = timed(runs, || {
            for i in 0..runs {
                let (q, c) = (each(i), &cold[i % copies]);
                for p in 0..n {
                    sink += sqdist(q, &c[p * dim..(p + 1) * dim]);
                }
            }
        });
        let cold_h = timed(runs, || {
            for i in 0..runs {
                let (q, c) = (each(i), &cold_half[i % copies]);
                for p in 0..n {
                    sink += bf16_sqdist(q, &c[p * dim..(p + 1) * dim]);
                }
            }
        });
        let cold_b = timed(runs, || {
            for i in 0..runs {
                let (q, c) = (each(i), &cold_byte[i % copies]);
                c.prepare(q, &mut at, &mut weight);
                for p in 0..n {
                    sink += c.sqdist(&at, &weight, p);
                }
            }
        });

        // The two prefix passes are whole answers rather than scans, so they
        // carry their own head and their own rerank and are comparable to a
        // full scan plus the head column beside it.
        let short = timed(runs, || {
            for i in 0..runs {
                kept += prefix.head(each(i), want).len();
            }
        });
        let two = timed(runs, || {
            for i in 0..runs {
                kept += prefix
                    .head_reranked(each(i), &centroids, dim, want, 4)
                    .len();
            }
        });

        let mut scores = vec![0f32; n];
        for (p, s) in scores.iter_mut().enumerate() {
            *s = sqdist(each(0), &centroids[p * dim..(p + 1) * dim]);
        }
        let pick = timed(runs, || {
            for _ in 0..runs {
                kept += head(n, want, |p| scores[p]).len();
            }
        });

        println!(
            "{n:9} {dim:6} {mb:8.1} {warm_full:7.0} {warm_half:5.0} {warm_byte:5.0} \
             {cold_f:7.0} {cold_h:5.0} {cold_b:5.0} {short:5.0} {two:8.0} {pick:7.0}"
        );
    }
    println!("microseconds a query, {runs} queries, head of {want} ({sink} {kept})");
}

/// The same question against a real collection: how much of the ceiling a
/// rougher ranking pass gives away.
///
/// The ceiling is [`miss`](crate::miss)'s number, the share of true neighbours
/// sitting in a partition the probe head reaches, which is the most recall a
/// search at that probe count could have. If the passes answer the same ceiling
/// then the ranking can be made rougher and recall cannot tell.
#[test]
#[ignore = "needs a dataset in YO_DATASET and prints a table rather than asserting"]
fn what_a_rougher_centroid_pass_gives_away() {
    use crate::partition::{Partitions, Tuning, Vectors};
    use crate::rabitq::Bits;

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

    /// A vector file, read the way [`miss`](crate::miss) reads one.
    fn read(path: &str, ints: bool) -> (usize, Vec<f32>) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        let dim = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let stride = 4 + dim * 4;
        assert_eq!(bytes.len() % stride, 0, "{path} is not whole vectors");
        let mut out = Vec::with_capacity(bytes.len() / stride * dim);
        for v in 0..bytes.len() / stride {
            for d in 0..dim {
                let at = v * stride + 4 + d * 4;
                let word = [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
                out.push(if ints {
                    i32::from_le_bytes(word) as f32
                } else {
                    f32::from_le_bytes(word)
                });
            }
        }
        (dim, out)
    }

    let dir = std::env::var("YO_DATASET")
        .expect("set YO_DATASET to a directory holding <name>_base.fvecs and the two beside it");
    let dir = dir.trim().to_string();
    let set = dir
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(&dir)
        .to_string();
    let queries: usize = std::env::var("YO_QUERIES")
        .ok()
        .and_then(|q| q.parse().ok())
        .unwrap_or(500);

    let (dim, data) = read(&format!("{dir}/{set}_base.fvecs"), false);
    let (_, query) = read(&format!("{dir}/{set}_query.fvecs"), false);
    let (gdim, truth) = read(&format!("{dir}/{set}_groundtruth.ivecs"), true);
    let n = data.len() / dim;
    let queries = queries.min(query.len() / dim);
    println!("{dir}: {n} base at {dim} dimensions, {queries} queries");

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

    let centroids = ix.all_centroids().to_vec();
    let halves = to_bf16(&centroids);
    let bytes = Bytes::build(&centroids, dim);
    let keeps = [dim / 12, dim / 6, dim / 3];
    let prefixes: Vec<Prefix> = keeps
        .iter()
        .map(|&k| Prefix::build(&centroids, dim, k))
        .collect();
    let (mut at, mut weight) = (Vec::new(), Vec::new());

    const K: usize = 10;
    let probes = [8usize, 32, 128, 512];
    let ways = 3 + keeps.len() * 2;
    let mut found = vec![vec![0usize; ways]; probes.len()];
    let mut total = 0usize;
    for q in 0..queries {
        let v = &query[q * dim..(q + 1) * dim];
        // Rotated, because the centroids are stored rotated and the search
        // rotates the query before it meets one, so a prefix here is a prefix
        // of the same coordinates the search would be looking at.
        let u = ix.quantizer().rotate(v);
        bytes.prepare(&u, &mut at, &mut weight);
        let holders: Vec<usize> = truth[q * gdim..q * gdim + K]
            .iter()
            .filter_map(|&id| ix.holder(id as u64))
            .collect();
        total += holders.len();
        for (i, &probe) in probes.iter().enumerate() {
            let mut picked = vec![
                head(parts, probe, |p| {
                    sqdist(&u, &centroids[p * dim..(p + 1) * dim])
                }),
                head(parts, probe, |p| {
                    bf16_sqdist(&u, &halves[p * dim..(p + 1) * dim])
                }),
                head(parts, probe, |p| bytes.sqdist(&at, &weight, p)),
            ];
            for p in &prefixes {
                picked.push(p.head(&u, probe));
            }
            for p in &prefixes {
                picked.push(p.head_reranked(&u, &centroids, dim, probe, 4));
            }
            for (k, way) in picked.iter().enumerate() {
                for h in &holders {
                    found[i][k] += usize::from(way.contains(h));
                }
            }
        }
    }

    let mut names = vec!["full".to_string(), "bf16".to_string(), "i8".to_string()];
    names.extend(keeps.iter().map(|k| format!("{k}d")));
    names.extend(keeps.iter().map(|k| format!("{k}d+rr")));
    println!();
    print!("{:>12}", "ceiling at");
    for name in &names {
        print!("{name:>9}");
    }
    println!();
    for (row, &probe) in found.iter().zip(&probes) {
        print!("{:>12}", format!("probe {probe}"));
        for reached in row.iter().take(ways) {
            print!("{:>9.4}", *reached as f64 / total as f64);
        }
        println!();
    }
}
