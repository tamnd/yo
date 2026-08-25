//! The SPSC lane against a `VecDeque`, on one thread.
//!
//! Loom covers the interleavings. This covers the index arithmetic: the wrap,
//! the full check, the empty check, and the cached opposite index, all of which
//! are plain arithmetic that a model checker will not think to stress with a
//! hundred thousand operations. It also runs every element through drop, which
//! is where a leak or a double free would show up.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::VecDeque;
use yo_shard::spsc;

#[derive(Arbitrary, Debug)]
enum Op {
    Push { value: u32 },
    Pop,
    Len,
}

#[derive(Arbitrary, Debug)]
struct Input {
    capacity: u8,
    ops: Vec<Op>,
}

/// Boxed so that a leak or a double free is a real allocator event rather than
/// a copy of four bytes nobody would notice.
type Item = Box<u32>;

fuzz_target!(|input: Input| {
    let cap = ((input.capacity as usize) % 64) + 1;
    let (tx, rx) = spsc::lane::<Item>(cap);
    let real_cap = tx.capacity();
    let mut model: VecDeque<u32> = VecDeque::new();

    for op in input.ops {
        match op {
            Op::Push { value } => {
                let got = tx.push(Box::new(value));
                if model.len() < real_cap {
                    assert!(got.is_ok(), "lane refused with room to spare");
                    model.push_back(value);
                } else {
                    assert!(got.is_err(), "lane accepted past its capacity");
                }
            }
            Op::Pop => {
                assert_eq!(rx.pop().map(|b| *b), model.pop_front(), "wrong item out");
            }
            Op::Len => {
                assert_eq!(rx.len(), model.len());
                assert_eq!(rx.is_empty(), model.is_empty());
            }
        }
    }

    // Whatever is left is dropped by the ring, not by us, so this exercises the
    // partial drop path with a non empty ring at an arbitrary offset.
});
