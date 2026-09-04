//! A run of samples, either laid out plainly or squeezed.
//!
//! A series is a list of these rather than one long buffer, because everything
//! a series is asked to do is bounded by a chunk: an insert in the middle only
//! rewrites the chunk it lands in, a range read only walks the chunks it
//! overlaps, and dropping what has aged out only drops whole chunks until it
//! reaches the one that straddles the cutoff.
//!
//! # The squeezed layout
//!
//! Timestamps and values are both written the way Gorilla writes them, which is
//! Facebook's 2015 paper and is what the module does too. A timestamp is stored
//! as the change in the gap since the last one, so a series sampled every ten
//! seconds writes one bit per sample however large the timestamps are, and a
//! value is stored as its exclusive or with the one before it, which for a
//! reading that moves slowly is a handful of bits in the middle of the double
//! with a run of zeros on either side. A sample that is a repeat of the one
//! before it costs two bits in total.
//!
//! The worst case is a sample whose gap and value are both unrelated to the one
//! before, at 145 bits, which is still under two doubles.

use crate::bits::{Reader, Writer};
use crate::sample::Sample;

/// How a chunk stores what it holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Encoding {
    /// Gorilla, which is what a series gets unless it asks otherwise.
    Compressed,
    /// A timestamp and a double each, which is what a series with values that
    /// have nothing to do with each other is better off with, and what a series
    /// that is rewritten in the middle a lot is better off with.
    Uncompressed,
}

impl Encoding {
    /// The word `TS.INFO` reports this as.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Compressed => "compressed",
            Self::Uncompressed => "uncompressed",
        }
    }
}

/// What a plain sample takes up, a timestamp and a double.
const PLAIN_BYTES: usize = 16;

/// The most bits one squeezed sample can take: four for the widest gap tag,
/// sixty four for the gap, two for the widest value tag, five and six for the
/// window it names, and sixty four for the window itself.
const PACKED_MAX_BITS: usize = 4 + 64 + 2 + 5 + 6 + 64;

/// Where the value encoder had got to, which is what the next sample is written
/// against.
#[derive(Clone, Copy, Debug)]
struct State {
    /// The last timestamp written.
    at: i64,
    /// The gap between the last two, which the next gap is written against.
    gap: i64,
    /// The last value, as bits.
    value: u64,
    /// How many zero bits the last value's window had in front of it, or
    /// [`NO_WINDOW`] if there has not been a window yet.
    leading: u32,
    /// How many it had behind it.
    trailing: u32,
}

/// The leading count that means no window has been named yet, which no real
/// window can be because a double has sixty four bits.
const NO_WINDOW: u32 = u32::MAX;

/// The samples themselves.
#[derive(Clone, Debug)]
enum Body {
    /// One `(i64, f64)` per sample.
    Plain(Vec<Sample>),
    /// Gorilla, with the encoder's state alongside so that appending does not
    /// have to walk the stream to find out where it was.
    Packed {
        /// The stream, with up to seven bits of padding at the end.
        bytes: Vec<u8>,
        /// How many bits of it are real.
        used: usize,
        /// What the next sample is written against.
        state: State,
    },
}

/// A run of samples in timestamp order, with a ceiling on how much room it may
/// take up.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// How the samples are stored.
    body: Body,
    /// The most bytes the samples may take up. A chunk always accepts its first
    /// sample, so a ceiling smaller than one sample gives chunks of one.
    room: usize,
    /// How many samples are in it.
    count: usize,
    /// The first timestamp, which is meaningless when the chunk is empty.
    first: i64,
    /// The last one.
    last: i64,
}

impl Chunk {
    /// An empty chunk that will hold about `room` bytes of samples.
    #[must_use]
    pub fn new(encoding: Encoding, room: usize) -> Self {
        let body = match encoding {
            Encoding::Compressed => Body::Packed {
                bytes: Vec::new(),
                used: 0,
                state: State {
                    at: 0,
                    gap: 0,
                    value: 0,
                    leading: NO_WINDOW,
                    trailing: 0,
                },
            },
            Encoding::Uncompressed => Body::Plain(Vec::new()),
        };
        Self {
            body,
            room,
            count: 0,
            first: 0,
            last: 0,
        }
    }

    /// How the samples are stored.
    #[must_use]
    pub fn encoding(&self) -> Encoding {
        match self.body {
            Body::Plain(_) => Encoding::Uncompressed,
            Body::Packed { .. } => Encoding::Compressed,
        }
    }

    /// How many samples are in it.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether it holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The first timestamp, or `None` when there are no samples.
    #[must_use]
    pub fn first(&self) -> Option<i64> {
        (self.count > 0).then_some(self.first)
    }

    /// The last one.
    #[must_use]
    pub fn last(&self) -> Option<i64> {
        (self.count > 0).then_some(self.last)
    }

    /// What the samples take up, not counting the chunk's own fields.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.body {
            Body::Plain(samples) => samples.capacity() * PLAIN_BYTES,
            Body::Packed { bytes, .. } => bytes.capacity(),
        }
    }

    /// Whether one more sample would fit.
    ///
    /// The answer is about the worst case rather than this particular sample,
    /// so a chunk stops a little short of its ceiling rather than a little
    /// past it. Stopping past it would mean a chunk size that is a promise
    /// about nothing.
    #[must_use]
    pub fn has_room(&self) -> bool {
        if self.count == 0 {
            return true;
        }
        match &self.body {
            Body::Plain(samples) => (samples.len() + 1) * PLAIN_BYTES <= self.room,
            Body::Packed { used, .. } => (used + PACKED_MAX_BITS).div_ceil(8) <= self.room,
        }
    }

    /// Adds a sample that comes after everything already here.
    ///
    /// The caller has to have asked [`Chunk::has_room`] first, and has to hand
    /// samples over in timestamp order, both of which the series does.
    pub fn append(&mut self, sample: Sample) {
        debug_assert!(
            self.count == 0 || sample.at > self.last,
            "a chunk is appended to in order"
        );
        if self.count == 0 {
            self.first = sample.at;
        }
        self.last = sample.at;
        self.count += 1;
        match &mut self.body {
            Body::Plain(samples) => samples.push(sample),
            Body::Packed { bytes, used, state } => {
                let mut w = Writer::resume(core::mem::take(bytes), *used);
                encode(&mut w, state, sample, self.count == 1);
                let (out, at) = w.finish();
                *bytes = out;
                *used = at;
            }
        }
    }

    /// Every sample in it, in order.
    #[must_use]
    pub fn samples(&self) -> Samples<'_> {
        let walk = match &self.body {
            Body::Plain(samples) => Walk::Plain(samples.iter()),
            Body::Packed { bytes, used, .. } => Walk::Packed {
                reader: Reader::new(bytes, *used),
                state: State {
                    at: 0,
                    gap: 0,
                    value: 0,
                    leading: NO_WINDOW,
                    trailing: 0,
                },
                first: true,
            },
        };
        Samples(walk)
    }

    /// Replaces everything in the chunk with `samples`, and answers the ones
    /// that did not fit.
    ///
    /// An insert in the middle and a delete both come through here, because
    /// both of them mean the stream after the point they touch has to be
    /// written again anyway. What comes back is a tail for the caller to put in
    /// a chunk of its own, which only happens when a chunk that was already
    /// near its ceiling had something inserted into it.
    pub fn rewrite<'s>(&mut self, samples: &'s [Sample]) -> &'s [Sample] {
        let room = self.room;
        *self = Self::new(self.encoding(), room);
        let mut taken = 0;
        for &sample in samples {
            if !self.has_room() {
                break;
            }
            self.append(sample);
            taken += 1;
        }
        &samples[taken..]
    }
}

/// Writes one sample against what came before it.
fn encode(w: &mut Writer, state: &mut State, sample: Sample, first: bool) {
    let bits = sample.value.to_bits();
    if first {
        w.put(sample.at as u64, 64);
        w.put(bits, 64);
        *state = State {
            at: sample.at,
            gap: 0,
            value: bits,
            leading: NO_WINDOW,
            trailing: 0,
        };
        return;
    }

    // The gap, written as how much it changed. A series sampled at a steady
    // rate has a change of zero every time, which is the one bit case.
    let gap = sample.at.wrapping_sub(state.at);
    let change = gap.wrapping_sub(state.gap);
    match change {
        0 => w.put_bit(false),
        -63..=64 => {
            w.put(0b10, 2);
            w.put((change + 63) as u64, 7);
        }
        -255..=256 => {
            w.put(0b110, 3);
            w.put((change + 255) as u64, 9);
        }
        -2047..=2048 => {
            w.put(0b1110, 4);
            w.put((change + 2047) as u64, 12);
        }
        _ => {
            w.put(0b1111, 4);
            w.put(change as u64, 64);
        }
    }
    state.at = sample.at;
    state.gap = gap;

    // The value, written as what changed in it.
    let xor = bits ^ state.value;
    state.value = bits;
    if xor == 0 {
        w.put_bit(false);
        return;
    }
    w.put_bit(true);
    let leading = xor.leading_zeros().min(31);
    let trailing = xor.trailing_zeros();
    if state.leading != NO_WINDOW && leading >= state.leading && trailing >= state.trailing {
        // The bits that changed sit inside the window the last value named, so
        // the window does not have to be named again.
        w.put_bit(false);
        let width = 64 - state.leading - state.trailing;
        w.put(xor >> state.trailing, width);
    } else {
        w.put_bit(true);
        let width = 64 - leading - trailing;
        w.put(u64::from(leading), 5);
        w.put(u64::from(width - 1), 6);
        w.put(xor >> trailing, width);
        state.leading = leading;
        state.trailing = trailing;
    }
}

/// Every sample in a chunk, in order.
#[derive(Debug)]
pub struct Samples<'a>(Walk<'a>);

/// Which of the two layouts a [`Samples`] is walking.
#[derive(Debug)]
enum Walk<'a> {
    /// A walk over the plain layout, which is a slice walk.
    Plain(core::slice::Iter<'a, Sample>),
    /// A walk over the squeezed one, which has to decode as it goes.
    Packed {
        /// Where the walk has got to in the stream.
        reader: Reader<'a>,
        /// What the next sample is read against.
        state: State,
        /// Whether the next sample is the one written in full.
        first: bool,
    },
}

impl Iterator for Samples<'_> {
    type Item = Sample;

    fn next(&mut self) -> Option<Sample> {
        match &mut self.0 {
            Walk::Plain(it) => it.next().copied(),
            Walk::Packed {
                reader,
                state,
                first,
            } => {
                if reader.done() {
                    return None;
                }
                if *first {
                    *first = false;
                    let at = reader.take(64)? as i64;
                    let bits = reader.take(64)?;
                    state.at = at;
                    state.gap = 0;
                    state.value = bits;
                    return Some(Sample {
                        at,
                        value: f64::from_bits(bits),
                    });
                }
                // The gap tag is a run of ones ending in a zero, up to four.
                let mut tag = 0;
                while tag < 4 && reader.take_bit()? {
                    tag += 1;
                }
                let change = match tag {
                    0 => 0,
                    1 => reader.take(7)? as i64 - 63,
                    2 => reader.take(9)? as i64 - 255,
                    3 => reader.take(12)? as i64 - 2047,
                    _ => reader.take(64)? as i64,
                };
                state.gap = state.gap.wrapping_add(change);
                state.at = state.at.wrapping_add(state.gap);

                if reader.take_bit()? {
                    if reader.take_bit()? {
                        state.leading = reader.take(5)? as u32;
                        let width = reader.take(6)? as u32 + 1;
                        state.trailing = 64 - state.leading - width;
                    }
                    let width = 64 - state.leading - state.trailing;
                    let xor = reader.take(width)? << state.trailing;
                    state.value ^= xor;
                }
                Some(Sample {
                    at: state.at,
                    value: f64::from_bits(state.value),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(encoding: Encoding, samples: &[Sample]) {
        let mut chunk = Chunk::new(encoding, 1 << 20);
        for &sample in samples {
            assert!(chunk.has_room());
            chunk.append(sample);
        }
        let back: Vec<Sample> = chunk.samples().collect();
        assert_eq!(back, samples, "{encoding:?}");
        assert_eq!(chunk.len(), samples.len());
        assert_eq!(chunk.first(), samples.first().map(|s| s.at));
        assert_eq!(chunk.last(), samples.last().map(|s| s.at));
    }

    fn both(samples: &[Sample]) {
        round_trip(Encoding::Compressed, samples);
        round_trip(Encoding::Uncompressed, samples);
    }

    #[test]
    fn an_empty_chunk_has_nothing_in_it() {
        both(&[]);
    }

    #[test]
    fn a_steady_series_comes_back_as_it_went_in() {
        let samples: Vec<Sample> = (0..500)
            .map(|i| Sample {
                at: 1_700_000_000_000 + i * 10_000,
                value: 21.5 + f64::from(i as i32 % 7) / 10.0,
            })
            .collect();
        both(&samples);
    }

    #[test]
    fn awkward_gaps_and_values_come_back_too() {
        let gaps = [0i64, 1, 2, 63, 64, 65, 255, 256, 257, 2047, 2048, 100_000];
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e-300,
            0.1,
        ];
        let mut samples = Vec::new();
        let mut at = 0i64;
        for (i, gap) in gaps.iter().cycle().take(120).enumerate() {
            at += gap;
            at += 1;
            samples.push(Sample {
                at,
                value: values[i % values.len()],
            });
        }
        // A nan does not compare equal to itself, so this one is checked by
        // bits rather than by the helper above.
        let mut chunk = Chunk::new(Encoding::Compressed, 1 << 20);
        for &sample in &samples {
            chunk.append(sample);
        }
        let back: Vec<Sample> = chunk.samples().collect();
        assert_eq!(back.len(), samples.len());
        for (a, b) in back.iter().zip(&samples) {
            assert_eq!(a.at, b.at);
            assert_eq!(a.value.to_bits(), b.value.to_bits());
        }
    }

    #[test]
    fn a_chunk_fills_up_and_says_so() {
        let mut chunk = Chunk::new(Encoding::Uncompressed, 64);
        for i in 0..4 {
            assert!(chunk.has_room(), "sample {i}");
            chunk.append(Sample { at: i, value: 1.0 });
        }
        assert!(!chunk.has_room());
        assert_eq!(chunk.len(), 4);
    }

    #[test]
    fn a_rewrite_hands_back_what_did_not_fit() {
        let samples: Vec<Sample> = (0..8)
            .map(|i| Sample {
                at: i,
                value: i as f64,
            })
            .collect();
        let mut chunk = Chunk::new(Encoding::Uncompressed, 64);
        let left = chunk.rewrite(&samples);
        assert_eq!(chunk.len(), 4);
        assert_eq!(left.len(), 4);
        assert_eq!(left[0].at, 4);
    }

    #[test]
    fn squeezing_a_steady_series_beats_storing_it() {
        let mut chunk = Chunk::new(Encoding::Compressed, 1 << 20);
        for i in 0..1000 {
            chunk.append(Sample {
                at: 1_700_000_000_000 + i * 1000,
                value: 42.0,
            });
        }
        // Sixteen bytes a sample plainly, and two bits a sample here.
        assert!(chunk.memory_bytes() < 1000, "{}", chunk.memory_bytes());
    }
}
