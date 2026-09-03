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
use std::time::{Duration, Instant};
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

/// A command from the back of the table, on a key that is not there.
///
/// `set` and `get` are the first two entries in the command table and `exists`
/// is the hundred and forty ninth, so the pair of them together says whether
/// finding a command costs anything, separately from what the command then does.
/// Deliberately a miss: `EXISTS` on a key nobody wrote does almost nothing after
/// the lookup, which is what leaves the lookup visible.
///
/// It is `EXISTS` and not `DEL` because a `DEL` that finds something is a
/// different command the second time round, and a row that changes shape between
/// the first iteration and the rest is not measuring one thing.
fn bench_exists(c: &mut Criterion) {
    bench_command(c, "exists", &[b"EXISTS", b"key:nothing"]);
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

/// How many entries the hot stream holds before the stream rows start.
///
/// Enough to be many listpack nodes rather than one, because a stream that fits
/// in a single node has no node walk in it and the node walk is most of what a
/// range costs.
const ENTRIES: u64 = 100_000;

/// The cap the stream rows keep the hot stream under.
///
/// A bench that appends for five seconds and never trims is a bench that ends up
/// measuring an allocator against a stream nobody would run. The tilde makes the
/// trim node at a time, which is what a real `MAXLEN ~` does and is why the trim
/// is close to free most of the time it is asked for.
const CAP: &[u8] = b"100000";

/// Fill `stream:hot` with [`ENTRIES`] entries and give it a consumer group.
///
/// The ids are explicit and consecutive so that the rows which have to name an
/// entry can work out what to name without asking.
fn fill_stream(r: &mut Reactor<Wire<Null>>, conn: ConnId) {
    fill_stream_of(r, conn, ENTRIES);
}

/// The same with the entry count named.
///
/// The group rows use a short stream. They read from the tail and they measure
/// the group lookup, the delivery and the pending list, none of which knows how
/// many nodes are sitting behind the entry being handed over. They also rebuild
/// the fixture once a sample rather than once a row, so a hundred thousand
/// entries there would be a minute of setup for a number that does not move.
fn fill_stream_of(r: &mut Reactor<Wire<Null>>, conn: ConnId, entries: u64) {
    let mut batch = Vec::new();
    let mut stream = Vec::new();
    for i in 1..=entries {
        let id = format!("{i}-0");
        stream.extend_from_slice(&wire(&[b"XADD", b"stream:hot", id.as_bytes(), b"f", b"v"]));
        if i % 64 == 0 {
            r.engine_mut().feed(conn, &stream);
            pump(r, &mut batch);
            stream.clear();
        }
    }
    r.engine_mut().feed(conn, &stream);
    pump(r, &mut batch);
    // `$` and not `0`, so the group starts at the end and the rows that read
    // with `>` see the entries they added themselves and nothing else.
    r.engine_mut().feed(
        conn,
        &wire(&[b"XGROUP", b"CREATE", b"stream:hot", b"g", b"$"]),
    );
    pump(r, &mut batch);
}

/// The append, capped so the stream stays the size a running one would be.
fn bench_xadd(c: &mut Criterion) {
    bench_over(
        c,
        "xadd",
        &[
            b"XADD",
            b"stream:hot",
            b"MAXLEN",
            b"~",
            CAP,
            b"*",
            b"f",
            b"v",
        ],
        fill_stream,
    );
}

/// The length, which is a counter and no walk at all.
///
/// It is here for the same reason `exists` is: it is the row that says what a
/// command costs when the command itself does nothing, so the other stream rows
/// can be read as the walk plus this.
fn bench_xlen(c: &mut Criterion) {
    bench_over(c, "xlen", &[b"XLEN", b"stream:hot"], fill_stream);
}

/// Ten entries off the front, which is the shape a reader polling a log has.
fn bench_xrange(c: &mut Criterion) {
    bench_over(
        c,
        "xrange",
        &[b"XRANGE", b"stream:hot", b"-", b"+", b"COUNT", b"10"],
        fill_stream,
    );
}

/// How many entries the hot stream holds before the group rows start.
const GROUP_ENTRIES: u64 = 1024;

/// Add entries to the hot stream, untimed, and answer the ids they got.
fn add_entries(r: &mut Reactor<Wire<Null>>, conn: ConnId, next: &mut u64, n: usize) -> Vec<u64> {
    let mut batch = Vec::new();
    let mut stream = Vec::new();
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        *next += 1;
        ids.push(*next);
        let id = format!("{next}-0");
        stream.extend_from_slice(&wire(&[
            b"XADD",
            b"stream:hot",
            b"MAXLEN",
            b"~",
            CAP,
            id.as_bytes(),
            b"f",
            b"v",
        ]));
    }
    r.engine_mut().feed(conn, &stream);
    pump(r, &mut batch);
    ids
}

/// The delivery, with the entries it delivers added outside the clock.
///
/// This row cannot be written the way the others are. `XREADGROUP` with `>`
/// consumes what it reads, so a batch fed to a stream that has run out is
/// measuring how fast we say there is nothing new, which is not the row anyone
/// means. So the entries are added before each timed batch and the clock only
/// covers the read, which is what `iter_custom` is for. Everything the setup
/// costs stays out of the number.
fn bench_xreadgroup(c: &mut Criterion) {
    let mut g = c.benchmark_group("engine/xreadgroup");

    for depth in [1usize, 16, 64] {
        let stream = pipelined(
            &[
                b"XREADGROUP",
                b"GROUP",
                b"g",
                b"c1",
                b"COUNT",
                b"1",
                b"STREAMS",
                b"stream:hot",
                b">",
            ],
            depth,
        );
        g.throughput(Throughput::Elements(depth as u64));
        g.bench_function(format!("p{depth}"), |b| {
            b.iter_custom(|iters| {
                let (mut r, conn, mut batch) = ready(Null::default());
                fill_stream_of(&mut r, conn, GROUP_ENTRIES);
                let mut next = GROUP_ENTRIES;
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // One entry a command, so every command in the batch has
                    // something of its own to deliver.
                    add_entries(&mut r, conn, &mut next, depth);
                    let at = Instant::now();
                    r.engine_mut().feed(conn, black_box(&stream));
                    black_box(pump(&mut r, &mut batch));
                    total += at.elapsed();
                }
                total
            });
        });
    }

    g.finish();
}

/// The acknowledgement, on entries that really are pending.
///
/// Same problem as [`bench_xreadgroup`] and the same answer. An `XACK` of an id
/// nobody is holding is a lookup and a miss, and the row that matters is the one
/// where the id comes out of the pending list.
fn bench_xack(c: &mut Criterion) {
    let mut g = c.benchmark_group("engine/xack");

    for depth in [1usize, 16, 64] {
        let read = pipelined(
            &[
                b"XREADGROUP",
                b"GROUP",
                b"g",
                b"c1",
                b"COUNT",
                b"1",
                b"STREAMS",
                b"stream:hot",
                b">",
            ],
            depth,
        );
        g.throughput(Throughput::Elements(depth as u64));
        g.bench_function(format!("p{depth}"), |b| {
            b.iter_custom(|iters| {
                let (mut r, conn, mut batch) = ready(Null::default());
                fill_stream_of(&mut r, conn, GROUP_ENTRIES);
                let mut next = GROUP_ENTRIES;
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let ids = add_entries(&mut r, conn, &mut next, depth);
                    r.engine_mut().feed(conn, &read);
                    pump(&mut r, &mut batch);
                    let mut acks = Vec::new();
                    for id in &ids {
                        let id = format!("{id}-0");
                        acks.extend_from_slice(&wire(&[
                            b"XACK",
                            b"stream:hot",
                            b"g",
                            id.as_bytes(),
                        ]));
                    }
                    let at = Instant::now();
                    r.engine_mut().feed(conn, black_box(&acks));
                    black_box(pump(&mut r, &mut batch));
                    total += at.elapsed();
                }
                total
            });
        });
    }

    g.finish();
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
    bench_exists,
    bench_sadd,
    bench_sadd_alternating,
    bench_srandmember,
    bench_xadd,
    bench_xlen,
    bench_xrange,
    bench_xreadgroup,
    bench_xack,
    bench_fanout
);
criterion_main!(benches);
