//! What it costs to open a `.yo` file, as a function of how big it is.
//!
//! The M1 gate is that a ten gigabyte file opens in under a hundred
//! milliseconds. Ten gigabytes is 320 regions, so the whole of open time is two
//! sixteen kilobyte slot reads plus 320 reads of 32 bytes each. That is the
//! design being measured rather than the disk: nothing here scans records,
//! nothing rebuilds an allocation table, and nothing grows with the number of
//! keys.
//!
//! The files are sparse. Each region is grown into existence and then has its
//! 32 byte header written, so a "ten gigabyte" file here occupies about 1.3 MB
//! of blocks. That is honest for what is being measured, because open only ever
//! touches the headers. A benchmark that wrote ten real gigabytes would be
//! measuring the page cache.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use yo_file::io as fio;
use yo_file::{CreateOptions, REGION_LEN, Yo, region_offset};
use yo_format::{PAGE_HEADER_LEN, PageHeader};

/// Where the fixtures go, from `YO_BENCH_DIR` or the temporary directory. Open
/// time is mostly the syscalls rather than the device, but a ramdisk still
/// flatters it, so the knob is the same one the commit benchmark uses.
fn bench_dir() -> PathBuf {
    std::env::var_os("YO_BENCH_DIR").map_or_else(std::env::temp_dir, PathBuf::from)
}

/// A file with `regions` written regions, and its own path.
struct Fixture(PathBuf);

impl Fixture {
    fn new(name: &str, regions: u64, shards: u32) -> Fixture {
        let mut p = bench_dir();
        std::fs::create_dir_all(&p).expect("the benchmark directory");
        p.push(format!("yo-bench-open-{name}-{}.yo", std::process::id()));
        let _ = std::fs::remove_file(&p);

        {
            drop(
                Yo::create(
                    &p,
                    &CreateOptions {
                        shard_count: shards,
                        ..CreateOptions::default()
                    },
                )
                .expect("create"),
            );
            let f = std::fs::OpenOptions::new()
                .write(true)
                .open(&p)
                .expect("reopen for the fixture");
            fio::grow_to(&f, region_offset(regions)).expect("grow");
            let mut head = [0u8; PAGE_HEADER_LEN];
            for i in 0..regions {
                PageHeader {
                    shard: (i % u64::from(shards)) as u32,
                    // Each shard numbers its own pages from zero, so this is the
                    // same shape a real file has.
                    page_addr: (i / u64::from(shards)) * REGION_LEN,
                    used: 1024,
                    dead_bytes: 0,
                    epoch: 1,
                }
                .encode(&mut head);
                fio::write_at(&f, region_offset(i), &head).expect("write a header");
            }
            fio::sync_all(&f).expect("sync");
        }
        Fixture(p)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn open(c: &mut Criterion) {
    let mut g = c.benchmark_group("open");
    // 320 regions is ten gigabytes, which is the number in the gate. The others
    // are there to show the shape, which should be a straight line through the
    // origin with a very small slope.
    for regions in [0u64, 32, 320, 1024] {
        let gib = regions * REGION_LEN / (1024 * 1024 * 1024);
        let f = Fixture::new(&format!("r{regions}"), regions, 8);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{regions}_regions_{gib}GiB")),
            &f.0,
            |b, path| {
                b.iter(|| {
                    let db = Yo::open(path).expect("open");
                    black_box(db.superblock().seq)
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, open);
criterion_main!(benches);
