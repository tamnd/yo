//! What the ring costs when the device is not the thing being measured.
//!
//! None of these is a gate number. The gate is two hundred thousand durable
//! commits a second in group mode on NVMe with SQPoll, and that is `bench/`, on
//! a qualified box, against a real device. What is here is the layer above the
//! device: the tag, the pending table and the submission path, all of which are
//! on the critical path of every commit and none of which should be visible next
//! to a write.
//!
//! The writes go to a file in the temp directory, so on a laptop this measures
//! the page cache. That is deliberate. A page cache write is the fastest thing a
//! write can be, which makes it the harshest backdrop for the overhead this is
//! actually measuring.

use std::fs::OpenOptions;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use yo_uring::{Kind, Pending, Ring, RingConfig};

fn temp(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("yo-uring-bench-{}-{name}", std::process::id()));
    p
}

#[cfg(unix)]
fn raw(f: &std::fs::File) -> yo_uring::Fd {
    use std::os::fd::AsRawFd;
    f.as_raw_fd()
}

#[cfg(windows)]
fn raw(f: &std::fs::File) -> yo_uring::Fd {
    use std::os::windows::io::AsRawHandle;
    f.as_raw_handle()
}

/// The tag and the table, with no kernel anywhere near them. This is the part
/// that runs once per submission on top of whatever the submission costs, so it
/// is the part that has to be nothing.
fn park_and_take(c: &mut Criterion) {
    let mut g = c.benchmark_group("pending");
    g.bench_function("park and take, one at a time", |b| {
        let mut p: Pending<u64> = Pending::with_capacity(4096);
        b.iter(|| {
            let t = p.park(Kind::Write, black_box(7)).expect("room");
            black_box(p.take(t))
        });
    });
    g.bench_function("park sixty four then take sixty four", |b| {
        let mut p: Pending<u64> = Pending::with_capacity(4096);
        b.iter(|| {
            let mut tokens = [None; 64];
            for (i, slot) in tokens.iter_mut().enumerate() {
                *slot = Some(p.park(Kind::Write, i as u64).expect("room"));
            }
            for slot in &tokens {
                black_box(p.take(slot.expect("parked"))).expect("state");
            }
        });
    });
    g.finish();
}

/// One turn of the loop, `04` section 2 shaped: park a batch, submit once,
/// drain. The number that matters is the per submission cost, which is this
/// divided by sixty four, and the thing to watch is that it does not move when
/// the batch grows.
fn a_turn_of_the_loop(c: &mut Criterion) {
    let path = temp("turn");
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(&path)
        .expect("a file in the temp directory");
    let page = vec![0x5au8; 4096];

    let mut g = c.benchmark_group("ring");
    for batch in [1usize, 16, 64] {
        g.throughput(criterion::Throughput::Elements(batch as u64));
        g.bench_function(format!("{batch} writes, one submit"), |b| {
            let mut ring = Ring::new(&RingConfig::plain()).expect("a ring");
            let mut pending: Pending<usize> = Pending::with_capacity(ring.entries());
            b.iter(|| {
                for i in 0..batch {
                    let t = pending.park(Kind::Write, i).expect("room in the table");
                    // SAFETY: `page` outlives the whole benchmark and is never
                    // moved, and `file` stays open for just as long. Every
                    // write goes to its own offset.
                    unsafe { ring.write_at(raw(&file), page.as_ptr(), 4096, (i as u64) * 4096, t) }
                        .expect("room in the queue");
                }
                ring.submit_and_wait(batch as u32).expect("submitted");
                let mut done = 0;
                while done < batch {
                    done += ring.drain(|comp| {
                        pending.take(comp.token).expect("the state that was parked");
                    }) as usize;
                }
            });
        });
    }
    g.finish();
    let _ = std::fs::remove_file(&path);
}

criterion_group!(benches, park_and_take, a_turn_of_the_loop);
criterion_main!(benches);
