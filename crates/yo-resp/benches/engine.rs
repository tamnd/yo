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
    bench_over(c, name, args, |_, _| {});
}

/// The same, with a chance to put something in the database first.
///
/// `fill` is handed the engine and the connection before the measured runs
/// start, and whatever it feeds is pumped and thrown away. The set rows need
/// this because a draw from a set nobody has added to measures the reply for an
/// empty key, which is not the row anyone means.
fn bench_over(
    c: &mut Criterion,
    name: &str,
    args: &[&[u8]],
    fill: impl Fn(&mut Reactor<Wire<Null>>, ConnId),
) {
    let mut g = c.benchmark_group(format!("engine/{name}"));

    for depth in [1usize, 16, 64] {
        let stream = pipelined(args, depth);
        g.throughput(Throughput::Elements(depth as u64));
        g.bench_function(format!("p{depth}"), |b| {
            let (mut r, conn, mut batch) = ready(Null::default());
            fill(&mut r, conn);
            b.iter(|| {
                r.engine_mut().feed(conn, black_box(&stream));
                black_box(pump(&mut r, &mut batch))
            });
        });
    }

    g.finish();
}

/// How many members the hot set holds before the set rows start.
///
/// Past every representation boundary, so the row measures the shape a hot key
/// actually has in a benchmark that ran for ten seconds rather than the
/// listpack it passed through on the way there.
const HOT: usize = 100_000;

/// Fill one set with [`HOT`] members, a batch at a time.
fn fill_named(r: &mut Reactor<Wire<Null>>, conn: ConnId, key: &[u8]) {
    let mut batch = Vec::new();
    for chunk in 0..HOT / 64 {
        let mut stream = Vec::new();
        for i in 0..64 {
            let m = format!("member:{:012}", chunk * 64 + i);
            stream.extend_from_slice(&wire(&[b"SADD", key, m.as_bytes()]));
        }
        r.engine_mut().feed(conn, &stream);
        pump(r, &mut batch);
    }
}

/// Fill `set:hot` with [`HOT`] members.
fn fill_hot(r: &mut Reactor<Wire<Null>>, conn: ConnId) {
    fill_named(r, conn, b"set:hot");
}

/// Fill `set:hot` and `set:alt`, both to [`HOT`] members.
fn fill_both(r: &mut Reactor<Wire<Null>>, conn: ConnId) {
    fill_named(r, conn, b"set:hot");
    fill_named(r, conn, b"set:alt");
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

/// The hot key shape: every command in the batch on the same set.
///
/// The member is one that is already there, which is the steady state of the
/// gate row. memtier draws its members from a fixed range, so a run long enough
/// to be measured spends most of itself adding members the set already holds,
/// and the insert is the rare case rather than the common one.
fn bench_sadd(c: &mut Criterion) {
    bench_over(
        c,
        "sadd",
        &[b"SADD", b"set:hot", b"member:000000000001"],
        fill_hot,
    );
}

/// The same batch, alternating between two sets instead of hammering one.
///
/// This is the A/B for Y13. `engine/sadd` is the escalated shape: every command
/// in the batch names the key the command in front of it named, so the memo in
/// the keyspace answers where the body is and nothing hashes or probes. This row
/// alternates between two keys, which defeats the memo on every single command
/// while leaving both sets, both records and both slabs exactly as warm in the
/// cache as the one set was. So the difference between the two rows is the memo
/// and close to nothing else, which is not true of the obvious alternative of
/// spreading over sixty four keys: that one would be measuring cache misses as
/// well and would flatter the claim.
///
/// What it says, on an Apple M4 in release, as a development measurement and not
/// a gate number: 73.5 ns a command escalated against 92.0 alternating at P64,
/// 77.2 against 93.1 at P16, and 128.6 against 138.5 at P1. So the memo is worth
/// about nineteen nanoseconds a command, which is 1.25x at P64 and 1.20x at P16.
/// The P1 rows are the pair this bench can say least about. A batch of one has
/// no command in front of it inside the same batch, so the only hit available is
/// against the batch before, and the maintenance slice runs in between. Ten
/// nanoseconds is what the two rows differ by there and this row cannot tell you
/// how much of it is the memo.
///
/// The member is one that is already there, for the reason [`bench_sadd`] gives.
fn bench_sadd_alternating(c: &mut Criterion) {
    let hot = wire(&[b"SADD", b"set:hot", b"member:000000000001"]);
    let alt = wire(&[b"SADD", b"set:alt", b"member:000000000001"]);
    let mut g = c.benchmark_group("engine/sadd-alternating");

    for depth in [1usize, 16, 64] {
        let mut stream = Vec::new();
        for i in 0..depth {
            stream.extend_from_slice(if i % 2 == 0 { &hot } else { &alt });
        }
        g.throughput(Throughput::Elements(depth as u64));
        g.bench_function(format!("p{depth}"), |b| {
            let (mut r, conn, mut batch) = ready(Null::default());
            fill_both(&mut r, conn);
            b.iter(|| {
                r.engine_mut().feed(conn, black_box(&stream));
                black_box(pump(&mut r, &mut batch))
            });
        });
    }

    g.finish();
}

/// The draw, which is the read half of the same shape.
///
/// `SPOP` is not here because it consumes what it reads, so a bench that runs
/// until the numbers settle would spend the end of itself measuring how fast we
/// say the set is empty. `yo`'s `inline` bench pops against a fixture rebuilt
/// per batch, and this row is the same walk without the removal.
fn bench_srandmember(c: &mut Criterion) {
    bench_over(c, "srandmember", &[b"SRANDMEMBER", b"set:hot"], fill_hot);
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

criterion_group!(
    benches,
    bench_set,
    bench_get,
    bench_incr,
    bench_sadd,
    bench_sadd_alternating,
    bench_srandmember,
    bench_fanout
);
criterion_main!(benches);
