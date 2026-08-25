//! What a machine losing power actually does to a file.
//!
//! It is tempting to model a crash as "the last write did not happen". Real
//! devices are worse than that in three specific ways, and each one is a fault
//! here because each one has broken a storage engine that only tested the easy
//! version.
//!
//! **A write is not atomic.** The unit the device promises is a sector, and a
//! 4 KiB write is eight of them. A crash in the middle leaves some sectors new
//! and some old, in any combination. That is [`Fault::Tear`] and
//! [`Fault::Scatter`].
//!
//! **Writes are not ordered.** Without a barrier, two writes issued in order
//! can land in either order, or the second can land and the first not. That is
//! [`Fault::Reorder`] and [`Fault::LosePrefix`], and it is why a log that
//! stores a record's length last only works if the length write is separated
//! from the body write by something the device respects.
//!
//! **Bytes rot after they land.** Not a crash at all, and included because the
//! answer has to be different. [`Fault::RotBit`] flips a bit in data that was
//! already durable. Losing that data is allowed. Handing it back as if it were
//! fine is not, and that is the one thing this harness exists to catch.

use crate::rng::Rng;
use crate::sink::{CrashSink, Image};

/// The unit a device promises not to tear.
///
/// 512 is the pessimistic answer. Every drive worth using has a 4 KiB internal
/// block and most of them will not tear inside it, but "most of them" is not a
/// guarantee anybody wrote down, and picking the smaller number means the
/// engine is tested against the worst device it might be deployed on rather
/// than the best one on the developer's desk.
pub const SECTOR: usize = 512;

/// One way for a file to be hurt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// Nothing that had not reached the device is there.
    ///
    /// The common case, and the one everybody tests. It is in the list because
    /// a harness whose easiest fault fails is a harness that found something
    /// early, which is worth more than one that only fails on the exotic ones.
    LoseAll,
    /// The first `n` writes since the last sync landed, and nothing after.
    ///
    /// A crash partway through a batch, with the device honouring order.
    LosePrefix(usize),
    /// Every pending write landed, in a shuffled order.
    ///
    /// The device honoured nothing. Only visible when two writes touch the same
    /// bytes, which is exactly what a page being appended to in place does, so
    /// this is not as theoretical as it sounds.
    Reorder,
    /// One pending write landed as a subset of its sectors, and everything
    /// after it was lost.
    ///
    /// The realistic single crash: the drive was partway through one command
    /// when the power went.
    Tear {
        /// Which pending write, counted from the last sync.
        which: usize,
        /// A bit per sector of that write, low bit first. A set bit landed.
        sectors: u64,
    },
    /// Every pending write landed as a subset of its sectors.
    ///
    /// Not physically realistic for one crash. It is here because it is a
    /// superset of the realistic ones and it explores the state space faster
    /// than waiting for the right tear to come up at random.
    Scatter,
    /// A bit flipped in bytes that had already reached the device.
    ///
    /// Media rot, not a crash. Judged by a different rule: the data is allowed
    /// to be gone, and is not allowed to come back wrong.
    RotBit {
        /// Which page, by position in address order.
        page: usize,
        /// Which byte of it.
        byte: usize,
        /// Which bit of that byte.
        bit: u8,
    },
}

impl Fault {
    /// Whether this is a crash, as opposed to the device eating something it
    /// had already promised to keep.
    ///
    /// The two get different verdicts, so nothing may guess which one it is
    /// looking at.
    #[must_use]
    pub const fn is_crash(&self) -> bool {
        !matches!(self, Fault::RotBit { .. })
    }

    /// A short name, for counting how much of the model a run actually reached.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Fault::LoseAll => "lose-all",
            Fault::LosePrefix(_) => "lose-prefix",
            Fault::Reorder => "reorder",
            Fault::Tear { .. } => "tear",
            Fault::Scatter => "scatter",
            Fault::RotBit { .. } => "rot-bit",
        }
    }

    /// Picks a fault that suits the state this sink is actually in.
    ///
    /// Weighted rather than uniform, and the weights are the point. A run made
    /// mostly of `LoseAll` proves very little, so the tearing faults get most of
    /// the budget. Rot gets a tenth, because it is testing a different claim and
    /// a tenth of a hundred thousand is still ten thousand of them.
    #[must_use]
    pub fn pick(sink: &CrashSink, rng: &mut Rng) -> Fault {
        let pending = sink.pending().len();

        // Nothing in flight means there is no crash to model, so the only fault
        // with anything to say is rot. Returning `LoseAll` here would be a trial
        // that tests nothing and still counts towards the hundred thousand.
        if pending == 0 {
            return Fault::rot(sink, rng).unwrap_or(Fault::LoseAll);
        }

        match rng.below(100) {
            0..=9 => Fault::LoseAll,
            10..=24 => Fault::LosePrefix(rng.below(pending)),
            25..=34 => Fault::Reorder,
            35..=64 => {
                let which = rng.below(pending);
                let n = sink.pending()[which].bytes.len().div_ceil(SECTOR).min(64);
                Fault::Tear {
                    which,
                    // Not a uniform mask. All ones and all zeroes are the two
                    // most likely real outcomes and a uniform draw over 64 bits
                    // reaches neither in any run this size, so they get their
                    // own arms.
                    sectors: match rng.below(4) {
                        0 => 0,
                        1 => prefix_mask(rng.below(n + 1)),
                        2 => u64::MAX,
                        _ => rng.next_u64(),
                    },
                }
            }
            65..=89 => Fault::Scatter,
            _ => Fault::rot(sink, rng).unwrap_or(Fault::Scatter),
        }
    }

    /// A bit flip somewhere in the durable image, or `None` if it is blank.
    fn rot(sink: &CrashSink, rng: &mut Rng) -> Option<Fault> {
        let pages = sink.durable().pages();
        if pages.is_empty() {
            return None;
        }
        let page = rng.below(pages.len());
        let len = pages[page].1.len();
        if len == 0 {
            return None;
        }
        Some(Fault::RotBit {
            page,
            byte: rng.below(len),
            bit: (rng.below(8)) as u8,
        })
    }

    /// Builds the image the device would be left holding.
    ///
    /// Starts from everything a sync covered, which a crash never touches, and
    /// puts back whichever pending bytes this fault lets through.
    #[must_use]
    pub fn apply(&self, sink: &CrashSink, rng: &mut Rng) -> Image {
        let mut img = sink.crash_base();
        let pending = sink.pending();

        match self {
            Fault::LoseAll => {}

            Fault::LosePrefix(n) => {
                for p in pending.iter().take(*n) {
                    img.apply(p.page_addr, p.offset, &p.bytes);
                }
            }

            Fault::Reorder => {
                let mut order: Vec<usize> = (0..pending.len()).collect();
                // Fisher-Yates, downwards, so every permutation is reachable.
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.below(i + 1));
                }
                for i in order {
                    let p = &pending[i];
                    img.apply(p.page_addr, p.offset, &p.bytes);
                }
            }

            Fault::Tear { which, sectors } => {
                // Everything before the torn write got through, because the
                // device was working on them first and finished.
                for p in pending.iter().take(*which) {
                    img.apply(p.page_addr, p.offset, &p.bytes);
                }
                if let Some(p) = pending.get(*which) {
                    land_sectors(&mut img, p.page_addr, p.offset, &p.bytes, *sectors);
                }
                // And nothing after it, because the power went.
            }

            Fault::Scatter => {
                for p in pending {
                    land_sectors(&mut img, p.page_addr, p.offset, &p.bytes, rng.next_u64());
                }
            }

            Fault::RotBit { page, byte, bit } => {
                // Everything pending landed. Rot is not a crash, so the run
                // finished normally and then the device went bad underneath it.
                for p in pending {
                    img.apply(p.page_addr, p.offset, &p.bytes);
                }
                let addr = img.pages().get(*page).map(|(a, _)| *a);
                if let Some(addr) = addr
                    && let Some(buf) = img.page_mut(addr)
                    && let Some(b) = buf.get_mut(*byte)
                {
                    *b ^= 1 << (bit % 8);
                }
            }
        }

        img
    }
}

/// A mask with the low `n` bits set, saturating at 64.
const fn prefix_mask(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// Lands the sectors of one write whose bit is set in `mask`.
///
/// Sector boundaries are counted from the start of the write rather than from
/// the start of the page. That is slightly wrong, since a real device aligns
/// them to the device, and it is wrong in the harsher direction: a write that
/// begins mid sector gets torn at a boundary a real drive would not tear at.
/// Being harsher than the hardware is the right way to be wrong here.
fn land_sectors(img: &mut Image, page_addr: u64, offset: usize, bytes: &[u8], mask: u64) {
    for (i, chunk) in bytes.chunks(SECTOR).enumerate() {
        if i < 64 && (mask >> i) & 1 == 1 {
            img.apply(page_addr, offset + i * SECTOR, chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_record::sink::{PageSink, PageSource, PageWrite};

    fn sink_with(durable: &[u8], pending: &[&[u8]]) -> CrashSink {
        let mut s = CrashSink::new();
        s.write(PageWrite {
            page_addr: 0,
            offset: 0,
            bytes: durable,
            covers_upto: durable.len() as u64,
        })
        .unwrap();
        s.sync().unwrap();
        let mut at = durable.len();
        for p in pending {
            s.write(PageWrite {
                page_addr: 0,
                offset: at,
                bytes: p,
                covers_upto: (at + p.len()) as u64,
            })
            .unwrap();
            at += p.len();
        }
        s
    }

    #[test]
    fn losing_everything_leaves_exactly_what_was_synced() {
        let s = sink_with(b"kept", &[b"one", b"two"]);
        let img = Fault::LoseAll.apply(&s, &mut Rng::new(1));
        assert_eq!(img.page_bytes(0).unwrap(), b"kept");
    }

    #[test]
    fn a_prefix_lands_and_the_rest_does_not() {
        let s = sink_with(b"kept", &[b"one", b"two"]);
        let img = Fault::LosePrefix(1).apply(&s, &mut Rng::new(1));
        assert_eq!(img.page_bytes(0).unwrap(), b"keptone");
    }

    #[test]
    fn a_prefix_of_zero_is_the_same_as_losing_everything() {
        let s = sink_with(b"kept", &[b"one"]);
        let img = Fault::LosePrefix(0).apply(&s, &mut Rng::new(1));
        assert_eq!(img.page_bytes(0).unwrap(), b"kept");
    }

    #[test]
    fn a_tear_with_no_sectors_loses_the_write_and_everything_after() {
        let s = sink_with(b"kept", &[b"one", b"two"]);
        let img = Fault::Tear {
            which: 0,
            sectors: 0,
        }
        .apply(&s, &mut Rng::new(1));
        assert_eq!(img.page_bytes(0).unwrap(), b"kept");
    }

    #[test]
    fn a_tear_with_every_sector_still_stops_the_writes_behind_it() {
        let s = sink_with(b"kept", &[b"one", b"two"]);
        let img = Fault::Tear {
            which: 0,
            sectors: u64::MAX,
        }
        .apply(&s, &mut Rng::new(1));
        assert_eq!(img.page_bytes(0).unwrap(), b"keptone");
    }

    #[test]
    fn a_tear_lands_the_sectors_its_mask_names_and_zero_fills_the_rest() {
        let big = vec![b'x'; SECTOR * 3];
        let s = sink_with(b"", &[&big]);
        // Sectors 0 and 2, not 1.
        let img = Fault::Tear {
            which: 0,
            sectors: 0b101,
        }
        .apply(&s, &mut Rng::new(1));
        let p = img.page_bytes(0).unwrap();
        assert_eq!(p.len(), SECTOR * 3);
        assert!(p[..SECTOR].iter().all(|&b| b == b'x'));
        assert!(p[SECTOR..SECTOR * 2].iter().all(|&b| b == 0), "torn out");
        assert!(p[SECTOR * 2..].iter().all(|&b| b == b'x'));
    }

    #[test]
    fn reordering_lands_all_of_them() {
        let s = sink_with(b"", &[b"aaaa", b"bbbb"]);
        let img = Fault::Reorder.apply(&s, &mut Rng::new(1));
        // Different offsets, so order does not change the result. What is being
        // checked is that nothing went missing.
        assert_eq!(img.page_bytes(0).unwrap(), b"aaaabbbb");
    }

    #[test]
    fn reordering_two_writes_to_the_same_bytes_can_land_either_way() {
        let mut first = 0;
        let mut second = 0;
        for seed in 0..200 {
            let mut s = CrashSink::new();
            s.write(PageWrite {
                page_addr: 0,
                offset: 0,
                bytes: b"AAAA",
                covers_upto: 4,
            })
            .unwrap();
            s.write(PageWrite {
                page_addr: 0,
                offset: 0,
                bytes: b"BBBB",
                covers_upto: 4,
            })
            .unwrap();
            let img = Fault::Reorder.apply(&s, &mut Rng::new(seed));
            match img.page_bytes(0).unwrap() {
                b"BBBB" => second += 1,
                b"AAAA" => first += 1,
                other => panic!("neither write landed whole: {other:?}"),
            }
        }
        assert!(first > 20 && second > 20, "{first} vs {second}");
    }

    #[test]
    fn rot_flips_exactly_one_bit_of_a_file_that_is_otherwise_whole() {
        let s = sink_with(b"kept", &[b"more"]);
        let img = Fault::RotBit {
            page: 0,
            byte: 2,
            bit: 3,
        }
        .apply(&s, &mut Rng::new(1));
        let p = img.page_bytes(0).unwrap();
        assert_eq!(p.len(), 8, "rot is not a crash, so everything landed");
        assert_eq!(&p[..2], b"ke");
        assert_eq!(p[2], b'p' ^ 0b1000);
        assert_eq!(&p[3..], b"tmore");
    }

    #[test]
    fn rot_is_not_a_crash_and_the_others_are() {
        assert!(Fault::LoseAll.is_crash());
        assert!(Fault::Scatter.is_crash());
        assert!(
            !Fault::RotBit {
                page: 0,
                byte: 0,
                bit: 0
            }
            .is_crash()
        );
    }

    #[test]
    fn picking_reaches_every_kind() {
        let s = sink_with(b"kept", &[b"one", b"two", b"three"]);
        let mut rng = Rng::new(2024);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            seen.insert(Fault::pick(&s, &mut rng).kind());
        }
        for want in [
            "lose-all",
            "lose-prefix",
            "reorder",
            "tear",
            "scatter",
            "rot-bit",
        ] {
            assert!(seen.contains(want), "never picked {want}: {seen:?}");
        }
    }

    #[test]
    fn picking_with_nothing_in_flight_still_finds_something_to_do() {
        let s = sink_with(b"kept", &[]);
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            // A trial with no pending writes and a LoseAll fault tests nothing,
            // so it has to come back as rot instead.
            assert_eq!(Fault::pick(&s, &mut rng).kind(), "rot-bit");
        }
    }

    #[test]
    fn a_prefix_mask_covers_what_it_says() {
        assert_eq!(prefix_mask(0), 0);
        assert_eq!(prefix_mask(1), 1);
        assert_eq!(prefix_mask(8), 0xff);
        assert_eq!(prefix_mask(64), u64::MAX);
        assert_eq!(prefix_mask(9999), u64::MAX);
    }
}
