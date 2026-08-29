//! Where probing stops beating accumulating, which turns out to be nowhere.
//!
//! K11 pre-registers the crossover at k around 7: with seven sets or fewer, probe
//! the smallest against the others, and above that, merge. `08` section 4 puts a
//! probe question at about 40 ns on a DRAM miss, which is what would make merging
//! worth the extra reading.
//!
//! This runs both plans over the same sets at k from 2 to 16 and lets them
//! disagree with that. Probe wins at every k in both shapes. The full numbers and
//! what they mean are in the `setops` module doc, and the short version is that a
//! probe question and an accumulate touch both measure about 25 ns, because an
//! accumulate touch also hashes and also lands in a table at random. Two equal
//! costs cannot trade against each other, so the lines converge without crossing.
//!
//! Two shapes, because they answer different questions.
//!
//! `dense` is every set the same size with heavy overlap, which is the shape the
//! crossover claim is about: probe cannot exit early because nearly every member
//! is in nearly every set, so it pays its full `k - 1` questions.
//!
//! `sparse` is the same sets with little overlap, where probe usually fails on
//! the first question it asks and its cost barely grows with k at all. It is here
//! as a control, and it behaves: probe stays flat near 5 ms from k of 2 to 16
//! while accumulate goes from 18 ms to 80 ms.
//!
//! # Reading these on a machine someone else is using
//!
//! The laptop these were taken on runs at a load average of 4 or 5 from other
//! work, and the first run of this benchmark came out non monotonic in k, which
//! is not something the code can do. Contention only ever adds time, so the
//! number to read is the minimum per iteration across the samples rather than the
//! mean criterion prints. That is in `target/criterion/setops/*/new/sample.json`
//! as `times` and `iters`, and the minimum of one over the other is stable run to
//! run where the mean is not. A quiet box would not need this.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::setops::{self, Plan, Table};

/// Big enough that a set does not sit in L2, so a probe is a real miss and not a
/// cache hit dressed as one.
const N: usize = 200_000;

/// Two, seven and either side of it, and out to a k no real query reaches, so the
/// slope past the crossover is visible rather than assumed from one point.
const KS: [usize; 7] = [2, 4, 6, 7, 8, 12, 16];

fn ks() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &KS[..2]
    } else {
        &KS
    }
}

/// `keep` sets how much of each set the sets have in common. Every set gets the
/// same first `keep` members and then its own, so the overlap is exact and the
/// same at every k.
fn build(k: usize, keep: usize) -> Vec<Table> {
    (0..k)
        .map(|s| {
            let mut set = Table::with_capacity(N);
            for i in 0..keep {
                set.insert(format!("shared:{i:012}").as_bytes(), ())
                    .expect("room");
            }
            for i in keep..N {
                set.insert(format!("own{s}:{i:012}").as_bytes(), ())
                    .expect("room");
            }
            set
        })
        .collect()
}

fn bench_crossover(c: &mut Criterion) {
    let mut g = c.benchmark_group("setops");
    // One intersection is one measurement. Criterion's per-element throughput
    // would divide by a member count that is not the same between the two plans,
    // which would flatter whichever one returns less.
    g.sample_size(10);

    for (shape, keep) in [("dense", N * 9 / 10), ("sparse", N / 100)] {
        for &k in ks() {
            let sets = build(k, keep);
            let refs: Vec<&Table> = sets.iter().collect();

            for (name, how) in [("probe", Plan::Probe), ("accumulate", Plan::Accumulate)] {
                g.bench_with_input(
                    BenchmarkId::new(format!("{shape}_{name}"), k),
                    &k,
                    |b, _| {
                        b.iter(|| {
                            let mut n = 0usize;
                            setops::inter_with(how, black_box(&refs), 0, |_| n += 1);
                            n
                        })
                    },
                );
            }
        }
    }

    g.finish();
}

/// The other two, which have no plan to choose but do have the gates on them.
///
/// `SUNION` ran at 2.56x on aki and `SDIFF` is in the same family, and both are
/// wanted at ten times, so they get a number here rather than being inferred from
/// the intersection.
fn bench_union_and_diff(c: &mut Criterion) {
    let mut g = c.benchmark_group("setops");
    g.sample_size(10);

    for &k in ks() {
        let sets = build(k, N / 2);
        let refs: Vec<&Table> = sets.iter().collect();

        g.bench_with_input(BenchmarkId::new("union", k), &k, |b, _| {
            b.iter(|| {
                let mut n = 0usize;
                setops::union(black_box(&refs), |_| n += 1);
                n
            })
        });

        g.bench_with_input(BenchmarkId::new("diff", k), &k, |b, _| {
            b.iter(|| {
                let mut n = 0usize;
                setops::diff(black_box(&refs), |_| n += 1);
                n
            })
        });
    }

    g.finish();
}

/// `SINTERSTORE`, which is the row aki lost worst, at 0.30x.
///
/// The presized build against the same intersection, so the difference between
/// this and `dense_probe` at the same k is what storing the result costs.
fn bench_store(c: &mut Criterion) {
    let mut g = c.benchmark_group("setops");
    g.sample_size(10);

    for &k in ks() {
        let sets = build(k, N * 9 / 10);
        let refs: Vec<&Table> = sets.iter().collect();
        let upper = refs.iter().map(|s| s.len()).min().unwrap_or(0);

        g.bench_with_input(BenchmarkId::new("interstore", k), &k, |b, _| {
            b.iter(|| {
                setops::collect(upper, |f| {
                    setops::inter(black_box(&refs), 0, f);
                })
                .len()
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_crossover, bench_union_and_diff, bench_store);
criterion_main!(benches);
