//! What a stream operation costs, and what the node layout buys.
//!
//! M7's exit gate asks for stream commands at ten times a rival, and before
//! this file there was no number here at all. The rows are the five things a
//! stream is actually used for: appending, reading a window, reading it
//! backwards, deleting one entry and trimming the old end away.
//!
//! Two rows exist only to price the node layout rather than an operation.
//! `append` and `append/own fields` run the same entries through the same call
//! and differ only in whether the field names repeat, so the gap between them
//! is what storing the names once for the node is worth. It shows up in time
//! because the names are bytes that have to be copied, and it shows up much
//! harder in memory, which the module's tests check separately.
//!
//! `range/window` is the row that says whether the binary search over node
//! masters was the right call. It reads a hundred entries starting from a
//! moving point in the middle, so every iteration pays a fresh locate and then
//! a sequential read, which is the shape `XRANGE` has when a client is paging.
//! The start moves by a prime so it does not sit at the same offset inside the
//! same node every time.
//!
//! `lookup` is that locate on its own, one entry by ID and nothing read after
//! it. Between the two rows the cost of finding a node and the cost of walking
//! one are separated, which is what has to happen before anyone argues for a
//! radix tree here.
//!
//! The `group/` rows are the consumer group and its pending list. `group/read`
//! is the same walk as `range/all` with a pending entry written per delivery,
//! so the gap between the two is what keeping the ledger costs and nothing
//! else. The two ack rows differ only in the order the acks arrive, in order
//! against scattered, because the pending list is a B-tree partly on the bet
//! that the sorted case is the common one and the gap is what that bet is
//! worth.
//!
//! The numbers worth quoting are from gamingpc, a quiet 13900K, because this
//! laptop's load average sits above sixty and an absolute time taken on it is
//! not comparable with the next one. At a million entries a locate is 596
//! nanoseconds, a hundred entry window forwards is 1.44 microseconds and the
//! same window backwards is 4.92. The locate hardly moves between a thousand
//! entries and a million, 468 nanoseconds against 596, which is the binary
//! search doing the thing it was picked for. Reading the whole stream is 9.01
//! milliseconds at a million, so 111 million entries a second sequential, and
//! appending a million is 368 milliseconds, so 2.7 million appends a second on
//! one core.
//!
//! The memory side does not need a timer. These entries, `sensor` and `reading`
//! a millisecond apart, cost 20.95 bytes each at a thousand, 22.94 at a hundred
//! thousand and 23.94 at a million. The same entries with a field name that
//! changes every time cost 42.70, 46.72 and 48.74, so sharing the names is
//! worth a little over half the stream. The number creeps up with size because
//! the ID differences widen as a node fills, and it settles once they need two
//! bytes rather than one.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_kv::stream::{Filter, Id, Limits, Stream};

/// Redis's defaults, so the nodes are the size a real server's are.
fn limits() -> Limits {
    Limits::default()
}

/// A reading off a sensor, which is the shape a stream almost always has.
fn value(i: usize) -> Vec<u8> {
    format!("{:.3}", i as f64 / 7.0).into_bytes()
}

/// At a thousand the stream is ten nodes, at a hundred thousand a thousand, and
/// at a million ten thousand, which is where the binary search has fourteen
/// steps to take and a walk would have ten thousand.
const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

fn sizes() -> &'static [usize] {
    if std::env::var_os("YO_BENCH_SMOKE").is_some() {
        &SIZES[..1]
    } else {
        &SIZES
    }
}

/// One entry a millisecond, which is a busy stream and keeps the ID differences
/// in the two byte range they are in in practice.
fn id(i: usize) -> Id {
    Id::new(i as u64 + 1, 0)
}

fn filled(n: usize) -> Stream {
    let lim = limits();
    let mut s = Stream::new();
    for i in 0..n {
        let v = value(i);
        let fields: [(&[u8], &[u8]); 2] = [(b"sensor", b"a4"), (b"reading", &v)];
        s.append(id(i), &fields, lim).expect("an append");
    }
    s
}

fn bench_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("stream");
    let lim = limits();

    for &n in sizes() {
        let values: Vec<Vec<u8>> = (0..n).map(value).collect();
        g.throughput(Throughput::Elements(n as u64));

        // `XADD` in a loop, every entry carrying the node's master fields. This
        // is the row every other stream number is read against.
        g.bench_with_input(BenchmarkId::new("append", n), &n, |b, _| {
            b.iter_batched_ref(
                Stream::new,
                |s| {
                    for (i, v) in values.iter().enumerate() {
                        let fields: [(&[u8], &[u8]); 2] = [(b"sensor", b"a4"), (b"reading", v)];
                        s.append(id(i), black_box(&fields), lim).expect("an append");
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // The same entries with a field name that changes every time, so no
        // entry can share the master's fields and every one stores its own
        // names. Nobody writes a stream this way on purpose. The row is here to
        // price the trick, not the workload.
        let names: Vec<Vec<u8>> = (0..n).map(|i| format!("reading{i}").into_bytes()).collect();
        g.bench_with_input(BenchmarkId::new("append/own fields", n), &n, |b, _| {
            b.iter_batched_ref(
                Stream::new,
                |s| {
                    for (i, v) in values.iter().enumerate() {
                        let fields: [(&[u8], &[u8]); 2] = [(b"sensor", b"a4"), (&names[i], v)];
                        s.append(id(i), black_box(&fields), lim).expect("an append");
                    }
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // `XDEL` over a tenth of the stream, spread out so the deletes land in
        // different nodes. Each one is a locate and a flag flip, and no bytes
        // move, which is the whole reason deleting is cheap here.
        let step = 10usize;
        g.throughput(Throughput::Elements((n / step).max(1) as u64));
        g.bench_with_input(BenchmarkId::new("delete", n), &n, |b, _| {
            b.iter_batched_ref(
                || filled(n),
                |s| {
                    let mut gone = 0usize;
                    for i in (0..n).step_by(step) {
                        gone += usize::from(s.delete(black_box(id(i))));
                    }
                    gone
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // `XTRIM MAXLEN` taking nine tenths of the stream away, which is whole
        // nodes popped off the front and is what a capped stream does on every
        // write once it is full.
        g.bench_with_input(BenchmarkId::new("trim", n), &n, |b, _| {
            b.iter_batched_ref(
                || filled(n),
                |s| black_box(s.trim_maxlen((n / 10) as u64, true)),
                criterion::BatchSize::LargeInput,
            )
        });
    }

    g.finish();
}

fn bench_read(c: &mut Criterion) {
    let mut g = c.benchmark_group("stream");

    for &n in sizes() {
        let s = filled(n);

        // `XRANGE - +`, the whole stream read. Per entry, so it reads as
        // nanoseconds an entry like the walk rows elsewhere.
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("range/all", n), &n, |b, _| {
            b.iter(|| {
                let mut got = 0usize;
                s.range(Id::MIN, Id::MAX, None, |_, fields| {
                    got += fields.len();
                    true
                });
                black_box(got)
            })
        });

        // A hundred entries from a moving point in the middle, which is a
        // locate and then a sequential read and is what paging looks like.
        g.throughput(Throughput::Elements(100));
        g.bench_with_input(BenchmarkId::new("range/window", n), &n, |b, _| {
            let half = (n / 2).max(1);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 7919) % half;
                let mut got = 0usize;
                s.range(black_box(id(n / 4 + i)), Id::MAX, Some(100), |_, fields| {
                    got += fields.len();
                    true
                });
                black_box(got)
            })
        });

        // A hundred entries ending at a moving point and handed back newest
        // first, which is `XREVRANGE`. The ends stay in sorted order because
        // that is the order this call takes them in. The gap between this row
        // and the one above it is what reading a stream in reverse costs.
        g.bench_with_input(BenchmarkId::new("rev range/window", n), &n, |b, _| {
            let half = (n / 2).max(1);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 7919) % half;
                let mut got = 0usize;
                s.rev_range(Id::MIN, black_box(id(n / 4 + i)), Some(100), |_, fields| {
                    got += fields.len();
                    true
                });
                black_box(got)
            })
        });

        // One entry by ID and nothing after it, which is the locate on its own:
        // a binary search over the node masters and then a walk through one
        // node to the entry.
        g.throughput(Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::new("lookup", n), &n, |b, _| {
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 7919) % n;
                let mut got = 0usize;
                s.range(black_box(id(i)), Id::MAX, Some(1), |_, fields| {
                    got += fields.len();
                    true
                });
                black_box(got)
            })
        });
    }

    g.finish();
}

/// A stream with one group that has read everything into one consumer.
///
/// The pending list is then the whole stream, which is the shape a group takes
/// when its consumers have stopped acknowledging, and it is the shape every
/// interesting question about the PEL is asked in.
fn behind(n: usize) -> Stream {
    let mut s = filled(n);
    s.create_group(b"workers", Id::MIN, Some(0));
    s.read_group(b"workers", b"alice", None, 1, |_, _| true)
        .expect("the group");
    s
}

fn bench_groups(c: &mut Criterion) {
    let mut g = c.benchmark_group("stream");

    for &n in sizes() {
        g.throughput(Throughput::Elements(n as u64));

        // `XREADGROUP >` over the whole stream, which is the log walk plus a
        // pending entry written per delivery. Read against `range/all`, the gap
        // is what the ledger costs.
        g.bench_with_input(BenchmarkId::new("group/read", n), &n, |b, _| {
            b.iter_batched_ref(
                || {
                    let mut s = filled(n);
                    s.create_group(b"workers", Id::MIN, Some(0));
                    s
                },
                |s| {
                    let mut got = 0usize;
                    s.read_group(b"workers", b"alice", None, 1, |_, fields| {
                        got += fields.len();
                        true
                    });
                    black_box(got)
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // `XACK` over the whole pending list, oldest first, which is what a
        // consumer that is keeping up does. Every ack takes the first key of
        // the map and the first of the consumer's set.
        g.bench_with_input(BenchmarkId::new("group/ack in order", n), &n, |b, _| {
            b.iter_batched_ref(
                || behind(n),
                |s| {
                    let group = s.group_mut(b"workers").expect("the group");
                    let mut gone = 0usize;
                    for i in 0..n {
                        gone += usize::from(group.ack(black_box(id(i))));
                    }
                    gone
                },
                criterion::BatchSize::LargeInput,
            )
        });

        // The same acks in a scattered order, which is what a pool of workers
        // finishing at different speeds actually produces. The gap between this
        // row and the one above it is what the B-tree costs when the access is
        // not the sorted one it was chosen partly for.
        g.bench_with_input(BenchmarkId::new("group/ack scattered", n), &n, |b, _| {
            b.iter_batched_ref(
                || behind(n),
                |s| {
                    let group = s.group_mut(b"workers").expect("the group");
                    let mut gone = 0usize;
                    let mut at = 0usize;
                    for _ in 0..n {
                        at = (at + 7919) % n;
                        gone += usize::from(group.ack(black_box(id(at))));
                    }
                    gone
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    for &n in sizes() {
        let s = behind(n);
        let group = s.group(b"workers").expect("the group");

        // `XPENDING` over a hundred entry window from a moving point, which is
        // the paging shape again and the row that says whether the ordered walk
        // is worth what the map costs.
        g.throughput(Throughput::Elements(100));
        g.bench_with_input(BenchmarkId::new("group/pending window", n), &n, |b, _| {
            let half = (n / 2).max(1);
            let mut i = 0usize;
            b.iter(|| {
                i = (i + 7919) % half;
                let mut got = 0usize;
                let want = Filter {
                    start: black_box(id(n / 4 + i)),
                    count: Some(100),
                    ..Filter::default()
                };
                group.pending_range(want, 1_000, |_, _, _| {
                    got += 1;
                    true
                });
                black_box(got)
            })
        });

        // The `XAUTOCLAIM` scan, a hundred entries deep from a moving point.
        // Nothing is claimed, so this is the cost of finding the stale ones and
        // not of moving them.
        g.bench_with_input(BenchmarkId::new("group/claimable", n), &n, |b, _| {
            let half = (n / 2).max(1);
            let mut i = 0usize;
            let mut out = Vec::new();
            b.iter(|| {
                i = (i + 7919) % half;
                out.clear();
                black_box(group.claimable(black_box(id(n / 4 + i)), 0, 1_000, 100, &mut out))
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_write, bench_read, bench_groups);
criterion_main!(benches);
