//! The arena against a shadow copy of every live run.
//!
//! Covers the bump pointer, the segment switch, and `resolve`, which are the
//! unsafe blocks in yo-arena. The model keeps its own copy of every allocation's
//! bytes, so any pointer arithmetic that lands one run on top of another shows
//! up as a byte mismatch instead of as a rare corruption in production.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use yo_arena::{Arena, MAX_ALLOC};

#[derive(Arbitrary, Debug)]
enum Op {
    /// Allocate and fill with a repeating byte.
    Alloc { len: u32, fill: u8 },
    /// Copy some bytes in.
    Put { data: Vec<u8> },
    /// Read one of the live runs back.
    Check { which: u16 },
    /// Free one of the live runs.
    Free { which: u16 },
    /// Overwrite a live run in place.
    Overwrite { which: u16, fill: u8 },
}

fuzz_target!(|ops: Vec<Op>| {
    let mut arena = Arena::new();
    // (address, expected bytes)
    let mut live: Vec<(yo_common::Addr, Vec<u8>)> = Vec::new();

    for op in ops {
        match op {
            Op::Alloc { len, fill } => {
                // Bounded so a single case cannot ask for gigabytes and turn
                // the fuzzer into a memory test.
                let len = (len as usize) % (MAX_ALLOC / 16);
                if let Some((addr, buf)) = arena.alloc(len) {
                    buf[..len].fill(fill);
                    live.push((addr, vec![fill; len]));
                }
            }
            Op::Put { data } => {
                if data.len() <= MAX_ALLOC
                    && let Some(addr) = arena.put(&data)
                {
                    live.push((addr, data));
                }
            }
            Op::Check { which } => {
                if !live.is_empty() {
                    let (addr, want) = &live[(which as usize) % live.len()];
                    assert_eq!(arena.get(*addr, want.len()), &want[..], "run at {addr:?}");
                }
            }
            Op::Free { which } => {
                if !live.is_empty() {
                    let i = (which as usize) % live.len();
                    let (addr, want) = live.remove(i);
                    arena.free(addr, want.len());
                }
            }
            Op::Overwrite { which, fill } => {
                if !live.is_empty() {
                    let i = (which as usize) % live.len();
                    let len = live[i].1.len();
                    arena.get_mut(live[i].0, len).fill(fill);
                    live[i].1.fill(fill);
                }
            }
        }
    }

    // Nothing may have moved. The arena never relocates on its own; only
    // compaction does, and compaction is driven from above.
    for (addr, want) in &live {
        assert_eq!(arena.get(*addr, want.len()), &want[..], "run at {addr:?} at the end");
    }
});
