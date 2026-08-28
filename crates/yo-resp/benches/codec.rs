//! What the codec costs per command, in and out.
//!
//! These are the two numbers M2's in-process gate is built on top of. A `SET`
//! that has to come in under 150 ns end to end cannot spend most of that in the
//! parser, so the parse of a three argument command and the write of a `+OK`
//! are measured on their own here, before anything else is in the way.
//!
//! Run one with:
//!
//!     cargo bench -p yo-resp -- decode/set

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_resp::{Argv, Limits, Out, Proto, Step};

fn decode(c: &mut Criterion) {
    let limits = Limits::default();
    let mut g = c.benchmark_group("decode");

    // The command the gate is measured on, at the size the gate uses.
    let set = b"*3\r\n$3\r\nSET\r\n$16\r\nkey:000000000001\r\n$16\r\nvalue00000000001\r\n";
    g.bench_function("set", |b| {
        let mut argv = Argv::new();
        b.iter(|| {
            let step = argv.decode(black_box(&set[..]), &limits).unwrap();
            debug_assert!(matches!(step, Step::Command { .. }));
            black_box(step)
        });
    });

    let get = b"*2\r\n$3\r\nGET\r\n$16\r\nkey:000000000001\r\n";
    g.bench_function("get", |b| {
        let mut argv = Argv::new();
        b.iter(|| {
            let step = argv.decode(black_box(&get[..]), &limits).unwrap();
            black_box(step)
        });
    });

    // A pipeline of sixteen, which is the depth `bench/00` calls P16. This is
    // the shape a benchmark client actually sends and the one the reactor will
    // see, so a per command cost that only holds in isolation is not useful.
    let mut pipeline = Vec::new();
    for _ in 0..16 {
        pipeline.extend_from_slice(get);
    }
    g.bench_function("get_pipelined_16", |b| {
        let mut argv = Argv::new();
        b.iter(|| {
            let buf = black_box(&pipeline[..]);
            let mut at = 0;
            let mut n = 0;
            while let Ok(Step::Command { consumed }) = argv.decode(&buf[at..], &limits) {
                if consumed == 0 {
                    break;
                }
                at += consumed;
                n += 1;
            }
            black_box(n)
        });
    });

    // The telnet path, which is not a hot path and is here so that a change
    // that makes it accidentally quadratic shows up as a number.
    g.bench_function("inline", |b| {
        let mut argv = Argv::new();
        b.iter(|| {
            let step = argv
                .decode(black_box(&b"SET key:000000000001 value\r\n"[..]), &limits)
                .unwrap();
            black_box(step)
        });
    });

    g.finish();
}

fn encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("encode");

    // What a `SET` replies. This is as small as a reply gets and it is the one
    // written most often.
    g.bench_function("ok", |b| {
        let mut out = Out::new(Proto::Resp2);
        b.iter(|| {
            out.clear();
            out.ok();
            black_box(out.len())
        });
    });

    // What a `GET` replies.
    let value = vec![b'v'; 64];
    g.bench_function("bulk_64", |b| {
        let mut out = Out::new(Proto::Resp2);
        b.iter(|| {
            out.clear();
            out.bulk(black_box(&value));
            black_box(out.len())
        });
    });

    // The integer path, which is `INCR` and every `*LEN` command, and the one
    // place the hand written digits earn their comment.
    g.bench_function("int", |b| {
        let mut out = Out::new(Proto::Resp2);
        b.iter(|| {
            out.clear();
            out.int(black_box(1_234_567_890));
            black_box(out.len())
        });
    });

    // A ten field map in both protocols. Same calls, two different sets of
    // bytes, and the downgrade should not cost anything measurable.
    for (name, proto) in [
        ("map_10_resp2", Proto::Resp2),
        ("map_10_resp3", Proto::Resp3),
    ] {
        g.bench_function(name, |b| {
            let mut out = Out::new(proto);
            b.iter(|| {
                out.clear();
                out.map(10);
                for i in 0..10i64 {
                    out.bulk(b"field");
                    out.int(black_box(i));
                }
                black_box(out.len())
            });
        });
    }

    g.finish();
}

criterion_group!(benches, decode, encode);
criterion_main!(benches);
