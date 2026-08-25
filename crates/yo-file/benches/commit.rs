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

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use yo_file::{CreateOptions, LogFile, Yo};
use yo_format::{RecordHeader, RecordKind};
use yo_record::{Durability, Log, LogConfig};

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
        let mut path = std::env::temp_dir();
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
                    // What makes the batch durable. In `none` this is a write
                    // with no sync behind it, and in `sync` the appends above
                    // have each already paid, so the shape of the loop is the
                    // same in all three and only the cost moves.
                    f.log.commit_pending().expect("commit");
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, commit);
criterion_main!(benches);
