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
//! # The other group, which is the merge
//!
//! `setops_ints` is the same thing over integer sets, where a third plan is
//! available because an intset is a sorted array. It has a shape the text group
//! does not, `striped`, and that shape is the reason it is worth reading the two
//! groups separately. A merge's whole advantage is what it can skip, so a
//! benchmark that only ever lays the sets out in a way that skips well is a
//! benchmark that will agree with whatever the merge does. `striped` is the
//! layout that cannot skip at all, and the first version of the merge lost to
//! the probe on it by nine times.
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
use yo_kv::setops::{self, Plan};
use yo_kv::{Set, SetLimits};

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
fn build(k: usize, keep: usize) -> Vec<Set> {
    (0..k)
        .map(|s| {
            // Hinted straight to the table band, because the point of this
            // benchmark is the plan and not the promotion path, and a set of two
            // hundred thousand is a table whatever it started as.
            let mut set = Set::with_hint(b"shared:000000000000", N, &SetLimits::DEFAULT);
            for i in 0..keep {
                set.add(format!("shared:{i:012}").as_bytes(), &SetLimits::DEFAULT);
            }
            for i in keep..N {
                set.add(format!("own{s}:{i:012}").as_bytes(), &SetLimits::DEFAULT);
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
            let refs: Vec<&Set> = sets.iter().collect();

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
        let refs: Vec<&Set> = sets.iter().collect();

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
        let refs: Vec<&Set> = sets.iter().collect();
        let upper = refs.iter().map(|s| s.len()).min().unwrap_or(0);

        g.bench_with_input(BenchmarkId::new("interstore", k), &k, |b, _| {
            b.iter(|| {
                setops::collect(upper, &SetLimits::DEFAULT, |f| {
                    setops::inter(black_box(&refs), 0, f);
                })
                .map_or(0, |s| s.len())
            })
        });
    }

    g.finish();
}

/// Where each set's unshared members sit relative to the other sets'.
///
/// This is the whole difference between a merge that is much faster and a merge
/// that is a little faster, so it is a parameter rather than a decision made once
/// and forgotten.
#[derive(Clone, Copy)]
enum Spread {
    /// Each set's own members in its own range, above every other set's.
    ///
    /// The best case for a seek: a cursor that lands in another set's range can
    /// step over the whole range in one binary search, so the merge never reads
    /// most of the members at all.
    Banded,
    /// Every set's own members drawn from one range, striped so they never
    /// collide.
    ///
    /// The worst case for a seek. Every member of every set lies between two
    /// members of every other set, so there is nothing to skip and the merge
    /// pays a step per member. Here so that the banded row cannot be read as the
    /// only answer.
    Striped,
}

/// Integer sets of `n` members, sharing their first `keep`.
///
/// The same shape as [`build`] and the same overlap, with the members as numbers
/// instead of text so that every set is an intset and there is something to
/// merge. The shared members are the low numbers and the rest are laid out by
/// `how`.
fn build_ints(k: usize, n: usize, keep: usize, how: Spread) -> Vec<Set> {
    (0..k)
        .map(|s| {
            let mut set = Set::new();
            for i in 0..keep {
                set.add(i.to_string().as_bytes(), &SetLimits::DEFAULT);
            }
            for i in keep..n {
                let m = match how {
                    Spread::Banded => (s + 1) * 10_000_000 + i,
                    // Widened by `k` rather than a constant so the stripes stay
                    // disjoint however many sets there are.
                    Spread::Striped => 10_000_000 + i * k + s,
                };
                set.add(m.to_string().as_bytes(), &SetLimits::DEFAULT);
            }
            assert!(set.ints().is_some(), "every operand has to be an intset");
            set
        })
        .collect()
}

/// The merge against the probe, on the operands the merge is possible on.
///
/// The claim is that a touch which is a pointer step and a comparison is a
/// different order of cost from one that hashes and lands in a table at random,
/// and this is where that stops being a claim. Same overlap, same counts and the
/// same `k` as the text rows above, so the two groups can be read against each
/// other.
///
/// `skewed` is the row the leapfrog exists for: a small set against a large one,
/// where the merge should touch a few members of the big set and skip the rest
/// while the probe reads every member of the small set and hashes it.
///
/// `striped` is the same sets as `sparse` with nothing to skip, which is the
/// merge's worst case and the honest floor under the other rows. Without it the
/// group would only ever measure the layout the merge likes.
fn bench_merge(c: &mut Criterion) {
    let mut g = c.benchmark_group("setops_ints");
    g.sample_size(10);

    for (shape, sizes, keep, spread) in [
        ("dense", (N, N), N * 9 / 10, Spread::Banded),
        ("sparse", (N, N), N / 100, Spread::Banded),
        ("striped", (N, N), N / 100, Spread::Striped),
        ("skewed", (N / 1_000, N), N / 2_000, Spread::Banded),
    ] {
        for &k in ks() {
            let mut sets = build_ints(k, sizes.1, keep, spread);
            // The first operand is the small one in the skewed row, which is
            // the set both plans start from.
            if sizes.0 != sizes.1 {
                sets[0] = build_ints(1, sizes.0, keep, spread).remove(0);
            }
            let refs: Vec<&Set> = sets.iter().collect();

            for (name, how) in [
                ("merge", Plan::Merge),
                ("probe", Plan::Probe),
                ("accumulate", Plan::Accumulate),
            ] {
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

    // The other two, each plan against the other over the same sets, because a
    // merge measured on integers and a table measured on text would be two
    // different inputs and no comparison at all.
    for &k in ks() {
        let sets = build_ints(k, N, N / 2, Spread::Striped);
        let refs: Vec<&Set> = sets.iter().collect();

        // Named for what each one actually is: the union falls back to a
        // counting table and the difference falls back to a probe.
        for (name, how) in [("union_merge", Plan::Merge), ("union_table", Plan::Probe)] {
            g.bench_with_input(BenchmarkId::new(name, k), &k, |b, _| {
                b.iter(|| {
                    let mut n = 0usize;
                    setops::union_with(how, black_box(&refs), |_| n += 1);
                    n
                })
            });
        }
        for (name, how) in [("diff_merge", Plan::Merge), ("diff_probe", Plan::Probe)] {
            g.bench_with_input(BenchmarkId::new(name, k), &k, |b, _| {
                b.iter(|| {
                    let mut n = 0usize;
                    setops::diff_with(how, black_box(&refs), |_| n += 1);
                    n
                })
            });
        }
    }

    // `SINTERSTORE` on integers, which is the row aki lost worst at 0.30x and
    // the row where the two halves help each other: a merge hands its members
    // over ascending, and ascending is the one order an intset takes at the tail
    // of its last run instead of memmoving something.
    for &k in ks() {
        let sets = build_ints(k, N, N * 9 / 10, Spread::Banded);
        let refs: Vec<&Set> = sets.iter().collect();
        let upper = refs.iter().map(|s| s.len()).min().unwrap_or(0);

        for (name, how) in [
            ("interstore_merge", Plan::Merge),
            ("interstore_probe", Plan::Probe),
        ] {
            g.bench_with_input(BenchmarkId::new(name, k), &k, |b, _| {
                b.iter(|| {
                    setops::collect(upper, &SetLimits::DEFAULT, |f| {
                        setops::inter_with(how, black_box(&refs), 0, f);
                    })
                    .map_or(0, |s| s.len())
                })
            });
        }
    }

    g.finish();
}

/// The small end, where the per key bookkeeping was most of the command.
///
/// Every row above is two hundred thousand members a set, which is a scale where
/// a handful of mallocs disappears into the walk. The commands people actually
/// send are `SINTER tag:a tag:b` over a few dozen members each, and there the
/// five little vectors a set operation used to build before it started were
/// comparable to the work itself.
///
/// Both representations, because they take different plans: integers merge and
/// text probes, and only one of them is free of a hash table.
fn bench_small(c: &mut Criterion) {
    let mut g = c.benchmark_group("setops_small");

    for &n in &[8usize, 64] {
        for &k in &[2usize, 3] {
            let ints = build_ints(k, n, n / 2, Spread::Banded);
            let int_refs: Vec<&Set> = ints.iter().collect();
            let text = build_small_text(k, n, n / 2);
            let text_refs: Vec<&Set> = text.iter().collect();

            for (what, refs) in [("ints", &int_refs), ("text", &text_refs)] {
                let id = format!("{what}/k{k}");
                g.bench_with_input(BenchmarkId::new("inter", &id), &n, |b, _| {
                    b.iter(|| {
                        let mut c = 0usize;
                        setops::inter(black_box(refs), 0, |_| c += 1);
                        c
                    })
                });
                g.bench_with_input(BenchmarkId::new("union", &id), &n, |b, _| {
                    b.iter(|| {
                        let mut c = 0usize;
                        setops::union(black_box(refs), |_| c += 1);
                        c
                    })
                });
            }
        }
    }

    g.finish();
}

/// [`build`] at a size small enough that the sets stay packed rather than
/// becoming tables, which is what a real tag set looks like.
fn build_small_text(k: usize, n: usize, keep: usize) -> Vec<Set> {
    (0..k)
        .map(|s| {
            let mut set = Set::new();
            for i in 0..keep {
                set.add(format!("shared:{i}").as_bytes(), &SetLimits::DEFAULT);
            }
            for i in keep..n {
                set.add(format!("own:{s}:{i}").as_bytes(), &SetLimits::DEFAULT);
            }
            set
        })
        .collect()
}

criterion_group!(
    benches,
    bench_crossover,
    bench_union_and_diff,
    bench_store,
    bench_merge,
    bench_small
);
criterion_main!(benches);
