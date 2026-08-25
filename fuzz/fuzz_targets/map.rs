//! The whole map against a `HashMap`.
//!
//! This is the one that finds real bugs. It drives set, get, delete and
//! compaction against the standard library's map and demands they agree at
//! every step, which covers directory doubling, segment splits, overflow chains
//! and the arena underneath all at once.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;
use yo_index::RawMap;

#[derive(Arbitrary, Debug)]
enum Op {
    Set { key: Vec<u8>, val: Vec<u8> },
    /// A key from a small pool, so collisions and overwrites actually happen.
    SetSmall { key: u16, val: u8 },
    Get { key: Vec<u8> },
    GetSmall { key: u16 },
    Del { key: Vec<u8> },
    DelSmall { key: u16 },
    Compact,
}

fn small(k: u16) -> Vec<u8> {
    format!("k{k}").into_bytes()
}

fuzz_target!(|ops: Vec<Op>| {
    let mut m = RawMap::new();
    let mut model: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

    for op in ops {
        match op {
            Op::Set { key, val } => {
                if key.len() > 4096 || val.len() > 65536 {
                    continue;
                }
                let got = m.set(&key, &val);
                let want = model.insert(key, val).map(|v| v.len());
                assert_eq!(got, want, "set returned the wrong old length");
            }
            Op::SetSmall { key, val } => {
                let k = small(key);
                let v = vec![val; (val as usize) % 300];
                let got = m.set(&k, &v);
                let want = model.insert(k, v).map(|v| v.len());
                assert_eq!(got, want);
            }
            Op::Get { key } => {
                assert_eq!(m.get(&key), model.get(&key).map(|v| &v[..]), "get {key:?}");
            }
            Op::GetSmall { key } => {
                let k = small(key);
                assert_eq!(m.get(&k), model.get(&k).map(|v| &v[..]));
            }
            Op::Del { key } => {
                assert_eq!(m.del(&key), model.remove(&key).is_some(), "del {key:?}");
            }
            Op::DelSmall { key } => {
                let k = small(key);
                assert_eq!(m.del(&k), model.remove(&k).is_some());
            }
            Op::Compact => {
                for seg in m.arena().compaction_candidates() {
                    m.compact_segment(seg);
                }
            }
        }
        assert_eq!(m.len(), model.len(), "length drifted");
    }

    // Full sweep at the end, because a single wrong probe can hide behind
    // operations that never touch the key it broke.
    for (k, v) in &model {
        assert_eq!(m.get(k), Some(&v[..]), "final sweep lost {k:?}");
    }
});
