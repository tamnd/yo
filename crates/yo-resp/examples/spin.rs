//! A profiling target: one command, pumped through the engine forever.
//!
//! The bench file measures. This one gives a profiler something to attach to,
//! which is the only thing criterion cannot do. Run it and sample it:
//!
//!     cargo build -p yo-resp --release --example spin
//!     ./target/release/examples/spin INCR hits &
//!     sample $! 5 -f /tmp/spin.txt
//!
//! The arguments after the program name are the command, so `spin GET k` and
//! `spin INCR hits` profile different paths through the same loop. The depth
//! comes from `YO_SPIN_DEPTH` and defaults to the batch shape, 64.
//!
//! `YO_SPIN_FILL` puts that many members into the key before the loop starts,
//! which is what makes the set commands measure the shape they have on a
//! benchmark that has been running for ten seconds rather than the listpack
//! they passed through on the way there. `YO_SPIN_FILL=100000 spin SADD set:hot
//! member:000000000001` is the hot key row: every command in the batch on one
//! key, adding a member that is already there.

use yo_reactor::Reactor;
use yo_resp::engine::{Cmd, ConnId, Sink, Wire, pump};

/// Counts bytes and keeps none, the same stand-in the bench uses.
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd: Vec<&[u8]> = if args.is_empty() {
        vec![b"INCR", b"hits"]
    } else {
        args.iter().map(|a| a.as_bytes()).collect()
    };
    let depth: usize = std::env::var("YO_SPIN_DEPTH")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(64);

    let mut stream = Vec::new();
    for _ in 0..depth {
        stream.extend_from_slice(format!("*{}\r\n", cmd.len()).as_bytes());
        for a in &cmd {
            stream.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
            stream.extend_from_slice(a);
            stream.extend_from_slice(b"\r\n");
        }
    }

    let mut r = Reactor::inline(Wire::new(Null::default()));
    let conn = r.engine_mut().accept();
    let mut batch: Vec<Cmd> = Vec::new();
    let fill: usize = std::env::var("YO_SPIN_FILL")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(0);

    if fill > 0 {
        let key = cmd.get(1).copied().unwrap_or(b"set:hot");
        let mut sadd = Vec::new();
        for i in 0..fill {
            let m = format!("member:{i:012}");
            sadd.extend_from_slice(b"*3\r\n$4\r\nSADD\r\n");
            sadd.extend_from_slice(format!("${}\r\n", key.len()).as_bytes());
            sadd.extend_from_slice(key);
            sadd.extend_from_slice(format!("\r\n${}\r\n{m}\r\n", m.len()).as_bytes());
            if i % 64 == 63 {
                r.engine_mut().feed(conn, &sadd);
                pump(&mut r, &mut batch);
                sadd.clear();
            }
        }
        r.engine_mut().feed(conn, &sadd);
        pump(&mut r, &mut batch);
    }
    let rounds: u64 = std::env::var("YO_SPIN_ROUNDS")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(0);

    if rounds == 0 {
        loop {
            r.engine_mut().feed(conn, &stream);
            pump(&mut r, &mut batch);
        }
    }

    // A timed run instead, for when the question is how much a change moved
    // rather than where the time goes.
    for _ in 0..rounds / 10 {
        r.engine_mut().feed(conn, &stream);
        pump(&mut r, &mut batch);
    }
    let t = std::time::Instant::now();
    for _ in 0..rounds {
        r.engine_mut().feed(conn, &stream);
        pump(&mut r, &mut batch);
    }
    let each = t.elapsed().as_secs_f64() / (rounds as f64 * depth as f64);
    println!("{:>6.1} ns/cmd  {}", each * 1e9, args.join(" "));
}
