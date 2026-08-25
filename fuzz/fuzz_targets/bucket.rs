//! The bucket against a plain array.
//!
//! Covers the SWAR tag compare and the seven byte address packing, which is the
//! unsafe block in `tag_word` plus the hand rolled `read56` and `write56`. The
//! model is seven optional pairs and a link. If the two ever disagree, the
//! packing has bled between fields, which is the one bug this layout can have.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use yo_common::{Addr, Space};
use yo_index::{Bucket, SLOTS};

#[derive(Arbitrary, Debug)]
enum Op {
    Set { slot: u8, tag: u8, space: u8, offset: u64 },
    SetAddr { slot: u8, space: u8, offset: u64 },
    Clear { slot: u8 },
    Link { target: u64 },
    Unlink,
    MatchTag { tag: u8 },
    MatchEmpty,
}

fn space_of(b: u8) -> Space {
    Space::ALL[(b as usize) % Space::ALL.len()]
}

fuzz_target!(|ops: Vec<Op>| {
    let mut b = Bucket::EMPTY;
    let mut model: [Option<(u8, Addr)>; SLOTS] = [None; SLOTS];
    let mut link: Option<u64> = None;

    for op in ops {
        match op {
            Op::Set { slot, tag, space, offset } => {
                let i = (slot as usize) % SLOTS;
                // Zero means empty, so it is not a legal tag and `set` panics
                // on it by contract. Fold it rather than skipping, so the case
                // still gets exercised.
                let tag = if tag == 0 { 1 } else { tag };
                let addr = Addr::new(space_of(space), offset & yo_common::MAX_OFFSET);
                b.set(i, tag, addr);
                model[i] = Some((tag, addr));
            }
            Op::SetAddr { slot, space, offset } => {
                let i = (slot as usize) % SLOTS;
                if let Some((tag, _)) = model[i] {
                    let addr = Addr::new(space_of(space), offset & yo_common::MAX_OFFSET);
                    b.set_addr(i, addr);
                    model[i] = Some((tag, addr));
                }
            }
            Op::Clear { slot } => {
                let i = (slot as usize) % SLOTS;
                b.clear(i);
                model[i] = None;
            }
            Op::Link { target } => {
                let t = target & yo_common::MAX_OFFSET;
                b.set_link(t);
                link = Some(t);
            }
            Op::Unlink => {
                b.clear_link();
                link = None;
            }
            Op::MatchTag { tag } => {
                let mut want: Vec<usize> = Vec::new();
                for (i, m) in model.iter().enumerate() {
                    let t = m.map_or(0u8, |(t, _)| t);
                    if t == tag {
                        want.push(i);
                    }
                }
                let got: Vec<usize> = b.match_tag(tag).collect();
                assert_eq!(got, want, "tag {tag} disagreed with the model");
            }
            Op::MatchEmpty => {
                let want: Vec<usize> = (0..SLOTS).filter(|&i| model[i].is_none()).collect();
                let got: Vec<usize> = b.match_empty().collect();
                assert_eq!(got, want, "empty slots disagreed with the model");
            }
        }

        // Every field, every time. The whole risk in this layout is one field
        // running into the next, and checking only the field just written would
        // miss exactly that.
        for (i, m) in model.iter().enumerate() {
            match m {
                Some((tag, addr)) => {
                    assert_eq!(b.tag(i), *tag, "slot {i} tag");
                    assert_eq!(b.addr(i), *addr, "slot {i} address");
                }
                None => assert_eq!(b.tag(i), 0, "slot {i} should read empty"),
            }
        }
        assert_eq!(b.link(), link, "link");
    }
});
