//! Where the bytes actually go, split into index and arena.
//!
//! `INFO memory` on a running server gives one number, and one number cannot
//! tell you whether a store is holding too much because its records are fat or
//! because nothing ever came back for the ones it overwrote. This writes the
//! same keys as many times as you ask and prints both.
//!
//! ```text
//! cargo run --release -p yo-kv --example memprobe -- 100000 64 4
//! ```
//!
//! Keys, value length, and how many times to write the whole set. The live
//! number should not move between one pass and ten. The reserved number moving
//! between one pass and ten is the arena holding what it has already replaced.
use yo_kv::{Keyspace, SetOptions};

fn arg(n: usize, default: usize) -> usize {
    std::env::args()
        .nth(n)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let keys = arg(1, 100_000);
    let vlen = arg(2, 64);
    let rounds = arg(3, 1);

    let val = vec![b'x'; vlen];
    let mut s = Keyspace::new();
    for _ in 0..rounds {
        for i in 0..keys {
            let key = format!("key:{i:012}");
            s.set(key.as_bytes(), &val, SetOptions::default())
                .expect("set");
            // What the event loop does once a turn, done here once a command so
            // that a run of this is the steady state and not a snapshot taken
            // before maintenance has had a chance.
            s.compact_step();
        }
    }

    let per = |bytes: usize| bytes as f64 / keys as f64;
    let index = s.map().index().memory_bytes();
    let live = s.map().arena().live_bytes() as usize;
    let held = s.map().arena().reserved_bytes() as usize;
    let total = s.memory_bytes();

    println!(
        "keys       {} of {vlen} bytes, written {rounds} times",
        s.len()
    );
    println!("index      {index:>12}  {:>7.1} a key", per(index));
    println!("arena live {live:>12}  {:>7.1} a key", per(live));
    println!("arena held {held:>12}  {:>7.1} a key", per(held));
    println!("segments   {}", s.map().arena().segment_count());
    println!("total      {total:>12}  {:>7.1} a key", per(total));
}
