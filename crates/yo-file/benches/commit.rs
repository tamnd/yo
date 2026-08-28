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
use yo_file::{CreateOptions, LogFile, Yo};
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
        let mut path = bench_dir();
        std::fs::create_dir_all(&path).expect("the benchmark directory");
        path.push(format!("yo-bench-commit-{name}-{}.yo", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut db = Yo::create(&path, &CreateOptions::default()).expect("create");
        let sink = db.log(0).expect("a log for shard 0");
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

criterion_group!(benches, commit, commit_batch);
criterion_main!(benches);
