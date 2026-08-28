//! What a command costs end to end, without a socket.
//!
//! `codec` measures the parse and the reply on their own, and `yo`'s `inline`
//! bench measures the store on its own. This is the two of them plus the part
//! in between: framing, the connection state, dispatch, and one write per
//! batch. Everything a real `SET` off the wire pays for except the kernel, so
//! the gap between this and a memtier number is the network stack and nothing
//! else.
//!
//! The pipeline depths are `bench/00`'s: P1 is the latency shape and P16 is the
//! throughput shape, which is also the shape that tells you whether the batch
//! is doing its job.
//!
//! Run one with:
//!
//!     cargo bench -p yo-resp -- engine/set/p16

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use yo_reactor::Reactor;
use yo_resp::engine::{Cmd, ConnId, Sink, Wire, pump};

/// A sink that takes everything and keeps nothing.
///
/// The bench is about the engine, not about a `Vec`, so the bytes are counted
/// and dropped. A real sink hands them to the ring, which is a memcpy this
/// stands in for.
#[derive(Default)]
struct Null {
    bytes: usize,
}

impl Sink for Null {
    fn write(&mut self, _conn: ConnId, bytes: &[u8]) -> usize {
        self.bytes += bytes.len();
        bytes.len()
    }
}

/// The wire bytes for one command.
fn wire(args: &[&[u8]]) -> Vec<u8> {
    let mut b = format!("*{}\r\n", args.len()).into_bytes();
    for a in args {
        b.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
        b.extend_from_slice(a);
        b.extend_from_slice(b"\r\n");
    }
    b
}

/// `depth` copies of one command, the way a pipelining client sends them.
fn pipelined(args: &[&[u8]], depth: usize) -> Vec<u8> {
    let one = wire(args);
    let mut all = Vec::with_capacity(one.len() * depth);
    for _ in 0..depth {
        all.extend_from_slice(&one);
    }
    all
}

fn ready(sink: Null) -> (Reactor<Wire<Null>>, ConnId, Vec<Cmd>) {
    let mut r = Reactor::inline(Wire::new(sink));
    let conn = r.engine_mut().accept();
    // Warm the buffers and the decoder pool, so the measured runs are the
    // steady state rather than the first one.
    let warm = pipelined(&[b"SET", b"warm", b"warm"], 64);
    let mut batch = Vec::new();
    for _ in 0..4 {
        r.engine_mut().feed(conn, &warm);
        pump(&mut r, &mut batch);
    }
    (r, conn, batch)
}

fn bench_command(c: &mut Criterion, name: &str, args: &[&[u8]]) {
    let mut g = c.benchmark_group(format!("engine/{name}"));

    for depth in [1usize, 16, 64] {
        let stream = pipelined(args, depth);
        g.throughput(Throughput::Elements(depth as u64));
        g.bench_function(format!("p{depth}"), |b| {
            let (mut r, conn, mut batch) = ready(Null::default());
            b.iter(|| {
                r.engine_mut().feed(conn, black_box(&stream));
                black_box(pump(&mut r, &mut batch))
            });
        });
    }

    g.finish();
}

fn bench_set(c: &mut Criterion) {
    bench_command(
        c,
        "set",
        &[b"SET", b"key:000000000001", b"value00000000001"],
    );
}

fn bench_get(c: &mut Criterion) {
    bench_command(c, "get", &[b"GET", b"key:000000000001"]);
}

fn bench_incr(c: &mut Criterion) {
    bench_command(c, "incr", &[b"INCR", b"hits"]);
}

/// Sixteen connections with one command each against one connection with
/// sixteen.
///
/// Same number of commands, same batch, and the difference is the flush: one
/// write against sixteen. It is the number that says whether a fan of idle
/// clients costs what a pipelining one costs.
fn bench_fanout(c: &mut Criterion) {
    let one = wire(&[b"GET", b"key:000000000001"]);
    let mut g = c.benchmark_group("engine/fanout");
    g.throughput(Throughput::Elements(16));

    g.bench_function("16x1", |b| {
        let mut r = Reactor::inline(Wire::new(Null::default()));
        let conns: Vec<ConnId> = (0..16).map(|_| r.engine_mut().accept()).collect();
        let mut batch = Vec::new();
        for _ in 0..4 {
            for &conn in &conns {
                r.engine_mut().feed(conn, &one);
            }
            pump(&mut r, &mut batch);
        }
        b.iter(|| {
            for &conn in &conns {
                r.engine_mut().feed(conn, black_box(&one));
            }
            black_box(pump(&mut r, &mut batch))
        });
    });

    g.bench_function("1x16", |b| {
        let stream = pipelined(&[b"GET", b"key:000000000001"], 16);
        let (mut r, conn, mut batch) = ready(Null::default());
        b.iter(|| {
            r.engine_mut().feed(conn, black_box(&stream));
            black_box(pump(&mut r, &mut batch))
        });
    });

    g.finish();
}

criterion_group!(benches, bench_set, bench_get, bench_incr, bench_fanout);
criterion_main!(benches);
