//! What a durable commit costs on a real device.
//!
//! The M1 gate is 200 thousand durable commits a second in `group` mode. This
//! is the benchmark that says whether that is true, and it is the one number in
//! the project that is about the disk rather than about the code. Run it on the
//! device you care about, because the answer on an NVMe drive, on a network
//! mount and inside a virtual machine are three different answers and only one
//! of them is interesting.
//!
//! Three modes are measured and the comparison between them is the point:
//!
//! - `none` never syncs, and is the append path plus one `pwrite` per page. It
//!   is the ceiling.
//! - `group` syncs once per batch of commits, which is what the maintenance
//!   slice does. The whole claim of `06` section 3 is that this lands close to
//!   `none` while still being durable.
//! - `sync` syncs once per commit. It is here to show what group commit is
//!   buying, and it is expected to be one or two orders of magnitude worse.
//!
//! A caveat that belongs on every number this prints: on macOS a sync is
//! `F_FULLFSYNC`, which flushes the drive's own cache. On Linux it is
//! `fdatasync`, which does not, unless the drive has no volatile cache or has
//! been told to write through. The Linux number is therefore the optimistic one
//! and the macOS number is the honest one, and they are not comparable.
//!
//! **Set `YO_BENCH_DIR`.** The default is [`std::env::temp_dir`], and on plenty
//! of machines that is a tmpfs, where a sync returns without going anywhere. The
//! first run of this benchmark measured a ramdisk and reported nine million
//! durable commits a second against a two hundred thousand gate, which is the
//! sort of number that should be an obvious lie and was not obvious at all. So
//! the directory is printed on every run, next to the device it sits on, and
//! anyone quoting a number from here is expected to quote those too.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use yo_file::{CreateOptions, LogFile, RingConfig, Yo};
use yo_format::{RecordHeader, RecordKind};
use yo_record::{Durability, Log, LogConfig};

/// Where the benchmark file goes, from `YO_BENCH_DIR` or the temporary
/// directory, and loudly enough that nobody reads the result without it.
fn bench_dir() -> PathBuf {
    match std::env::var_os("YO_BENCH_DIR") {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!(
                "warning: YO_BENCH_DIR is not set, so this is measuring {}, \
                 which may be a ramdisk. Do not quote these numbers.",
                std::env::temp_dir().display()
            );
            std::env::temp_dir()
        }
    }
}

/// How many commits go between two syncs in `group` mode. The maintenance
/// slice picks this by time rather than by count, but a count is what a
/// benchmark can hold still.
const BATCH: usize = 64;

/// A key buffer on the stack, because formatting into a `String` inside the
/// loop measures the allocator.
struct KeyBuf([u8; 24]);

impl KeyBuf {
    fn new() -> KeyBuf {
        KeyBuf(*b"key:00000000000000000000")
    }

    fn set(&mut self, mut n: u64) -> &[u8] {
        for b in self.0[4..].iter_mut().rev() {
            *b = b'0' + (n % 10) as u8;
            n /= 10;
        }
        &self.0
    }
}

/// A fresh file, and a log for shard 0 over it.
struct Fixture {
    path: PathBuf,
    log: Log<LogFile>,
}

impl Fixture {
    fn new(name: &str, durability: Durability) -> Fixture {
        Fixture::build(name, durability, None)
    }

    fn ringed(name: &str, durability: Durability, config: &RingConfig) -> Fixture {
        Fixture::build(name, durability, Some(config))
    }

    fn build(name: &str, durability: Durability, ring: Option<&RingConfig>) -> Fixture {
        let mut path = bench_dir();
        std::fs::create_dir_all(&path).expect("the benchmark directory");
        path.push(format!("yo-bench-commit-{name}-{}.yo", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut db = Yo::create(&path, &CreateOptions::default()).expect("create");
        let mut sink = db.log(0).expect("a log for shard 0");
        if let Some(config) = ring {
            sink.use_ring(config).expect("a ring");
        }
        let log = Log::new(
            LogConfig {
                shard: 0,
                durability,
                ..LogConfig::default()
            },
            sink,
        )
        .expect("a log");
        Fixture { path, log }
    }

    /// Ends a batch and does not return until it is durable.
    ///
    /// The whole point of the ring is that `commit_pending` stops being the
    /// place durability happens, so a benchmark that stopped there would be
    /// timing a queue insertion and calling it a durable commit rate. The wait
    /// is what makes the two columns the same measurement.
    fn commit_durable(&mut self) {
        self.log.commit_pending().expect("commit");
        self.log.sink_mut().drain().expect("drain");
        debug_assert!(self.log.durable_upto() >= self.log.tail());
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn commit(c: &mut Criterion) {
    let h = RecordHeader::new(RecordKind::String);
    let value = [b'v'; 64];

    eprintln!("commit benchmark writing to {}", bench_dir().display());

    let mut g = c.benchmark_group("commit");
    g.throughput(Throughput::Elements(BATCH as u64));
    for durability in [Durability::None, Durability::Group, Durability::Sync] {
        g.bench_with_input(
            BenchmarkId::from_parameter(durability.as_str()),
            &durability,
            |b, d| {
                let mut f = Fixture::new(d.as_str(), *d);
                let mut keys = KeyBuf::new();
                let mut n = 0u64;
                b.iter(|| {
                    for _ in 0..BATCH {
                        n += 1;
                        let a = f.log.append(&h, keys.set(n), &value).expect("append").addr;
                        black_box(a);
                    }
                    // What ends the batch. `commit_pending` flushes and syncs
                    // whatever the mode, so `none` gets the flush on its own or
                    // it would be measuring a sync it never asked for, which is
                    // how `none` and `group` came out identical the first time
                    // this was run. In `sync` the appends above have each
                    // already paid and this finds nothing left to do.
                    match d {
                        Durability::None => f.log.flush().expect("flush"),
                        _ => f.log.commit_pending().expect("commit"),
                    }
                });
            },
        );
    }
    g.finish();
}

/// Group commit against batch size, which is the shape the gate actually asks
/// about.
///
/// The comparison above holds the batch at 64, and on a device where a sync
/// costs two milliseconds that caps the whole thing at 32 thousand commits a
/// second no matter how good the code is. That is not a fact about yo, it is 64
/// divided by the sync cost, and a fixed batch turns the gate into a question
/// about the drive alone.
///
/// The maintenance slice does not work that way. It batches by time, so on a
/// slow device it naturally gathers more commits per sync, and on a fast one it
/// syncs sooner. What this sweep gives is the curve: commits per second against
/// how many of them share a sync. Read the batch size where the line crosses
/// two hundred thousand, and compare it against how many commits a shard really
/// has in flight in the window the slice waits.
fn commit_batch(c: &mut Criterion) {
    let h = RecordHeader::new(RecordKind::String);
    let value = [b'v'; 64];

    let mut g = c.benchmark_group("commit_batch");
    for batch in [8usize, 64, 512, 4096] {
        g.throughput(Throughput::Elements(batch as u64));
        g.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch| {
            let mut f = Fixture::new(&format!("batch{batch}"), Durability::Group);
            let mut keys = KeyBuf::new();
            let mut n = 0u64;
            b.iter(|| {
                for _ in 0..batch {
                    n += 1;
                    let a = f.log.append(&h, keys.set(n), &value).expect("append").addr;
                    black_box(a);
                }
                f.log.commit_pending().expect("commit");
            });
        });
    }
    g.finish();
}

/// The synchronous write path against the ring, in group mode, over the batch
/// sizes that matter.
///
/// This is the M1 gate row and the reason `yo-uring` exists. The claim is that
/// handing the bytes over and carrying on beats stopping the shard for every
/// `pwrite`, and that the gap widens with the batch, because a batch of 4096 is
/// 4096 chances to be stopped.
///
/// Read the two columns together with what the run says about itself. On Linux
/// the ring column is io_uring. On macOS and Windows it is the portable backend
/// from `04` section 7, which does the same writes synchronously behind the same
/// state machine, so the two columns there differ by a memcpy and some
/// bookkeeping and the ring one should be slightly worse. That is the expected
/// result off Linux and it is not a regression, it is the platform. Only a
/// Linux run is a gate row, which is why the mode is printed rather than
/// assumed.
fn commit_ring(c: &mut Criterion) {
    let h = RecordHeader::new(RecordKind::String);
    let value = [b'v'; 64];

    // Once, before anything is measured. Inside the bench closure this would
    // print on every sample and shred criterion's own output.
    {
        let probe = Fixture::ringed("probe", Durability::Group, &RingConfig::plain());
        eprintln!(
            "commit_ring: ring backend is {}",
            if probe.log.sink().is_uring() {
                "io_uring"
            } else {
                "the portable one, so the ring column here is not a gate row"
            }
        );
    }

    let mut g = c.benchmark_group("commit_ring");
    for batch in [64usize, 512, 4096] {
        g.throughput(Throughput::Elements(batch as u64));
        for ring in [false, true] {
            let name = if ring { "ring" } else { "pwrite" };
            g.bench_with_input(
                BenchmarkId::new(name, batch),
                &(batch, ring),
                |b, &(batch, ring)| {
                    let tag = format!("{name}{batch}");
                    let mut f = if ring {
                        Fixture::ringed(&tag, Durability::Group, &RingConfig::plain())
                    } else {
                        Fixture::new(&tag, Durability::Group)
                    };
                    let mut keys = KeyBuf::new();
                    let mut n = 0u64;
                    b.iter(|| {
                        for _ in 0..batch {
                            n += 1;
                            let a = f.log.append(&h, keys.set(n), &value).expect("append").addr;
                            black_box(a);
                        }
                        f.commit_durable();
                    });
                },
            );
        }
    }
    g.finish();
}

criterion_group!(benches, commit, commit_batch, commit_ring);
criterion_main!(benches);
