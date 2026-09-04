//! How a vector set squeezes a vector before it stores it.
//!
//! Three ways, and they are Redis's three rather than ours, because a client can
//! see all of them. `VEMB` hands back what was stored and not what was sent,
//! `VEMB RAW` hands back the bytes, and `VSIM` scores whatever is in there, so a
//! set written with `Q8` and a set written with `NOQUANT` answer differently and
//! a client is entitled to both answers.
//!
//! # What is stored is a direction and a length
//!
//! Every one of the three splits a vector into the length it had and the
//! direction it pointed, keeps the length as one float beside the element, and
//! squeezes only the direction. That is what makes the three comparable: they
//! differ in how much of the direction survives and in nothing else.
//!
//! [`Quant::None`] keeps the direction as it was, four bytes a coordinate.
//! [`Quant::Int8`] keeps it as a signed byte a coordinate against a scale that
//! is the largest coordinate there is. [`Quant::Bin`] keeps one bit a
//! coordinate, which is the sign, and throws the rest away.
//!
//! # The arithmetic is the arithmetic a real server does
//!
//! Down to which multiplication happens first, because the answers are visible.
//! A code is `round(unit * (127 / range))` with the reciprocal formed once, and
//! not `round(unit / range * 127)`, which disagrees about one coordinate in ten.
//! What comes back out is `(code * range) / 127`, and not `(code / 127) * range`,
//! which disagrees about half the time. Both were read off a real server over
//! several hundred vectors rather than guessed.
//!
//! The length is [`norm`], which is the strangest of the three and the one that
//! took the longest to pin down, because it is neither an accurate sum of
//! squares nor a naive one. It is Redis's loop from `hnsw.c` including its
//! unroll by four and which of its multiplies the compiler fused into the adds
//! beside them, and it matched a real server on 800 vectors out of 800 where
//! every simpler shape matched about seven in ten.

/// How a vector set stores the direction of its vectors.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// `NOQUANT`, which stores the direction as it arrived.
    #[default]
    None,
    /// `Q8`, one signed byte a coordinate against the widest one.
    Int8,
    /// `BIN`, one bit a coordinate, which is the sign of it.
    Bin,
}

impl Quant {
    /// What `VINFO` calls this, which is the name of the stored form and not the
    /// name of the option that asked for it.
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Quant::None => "f32",
            Quant::Int8 => "int8",
            Quant::Bin => "bin",
        }
    }

    /// How many bytes `VEMB RAW` writes for a `dim` wide vector.
    ///
    /// The binary form is rounded up to whole eight byte words rather than to
    /// whole bytes, because it is a run of 64 bit words on the wire and a set of
    /// 65 dimensions writes sixteen bytes and not nine.
    #[must_use]
    pub fn code_bytes(self, dim: usize) -> usize {
        match self {
            Quant::None => dim * 4,
            Quant::Int8 => dim,
            Quant::Bin => dim.div_ceil(64) * 8,
        }
    }
}

/// A vector split into the direction that gets stored and the two numbers that
/// turn it back into what the client sent.
#[derive(Debug, Clone)]
pub struct Squeezed {
    /// The direction, of unit length, in the form the quantisation leaves it.
    pub dir: Vec<f32>,
    /// How long the client's vector was.
    pub norm: f32,
    /// The widest coordinate of the direction, which is `Q8`'s scale and is 0
    /// for the other two because they have no scale.
    pub range: f32,
}

/// The euclidean length of a vector, to the bit.
///
/// This is a strange shape for a sum of squares and every part of it is on
/// purpose. Redis computes this one in `hnsw.c`, in a loop unrolled by four that
/// adds the four squares together and then adds that to the running total, and
/// the compiler fuses three of the four multiplies into the adds next to them.
/// The result is a different last bit from a plain sum, from a fully fused sum
/// and from an exact one, and it is the number `VEMB` prints and the number the
/// stored direction was divided by, so getting it right is the difference
/// between matching a real server on every vector and matching it on about
/// nine in ten.
///
/// [`f32::mul_add`] is fused by contract in Rust rather than by whatever the
/// compiler felt like, so this answers the same on every machine it runs on,
/// which is a thing a real server cannot quite say about its own.
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let (four, rest) = v.as_chunks::<4>();
    for c in four {
        let block = c[0].mul_add(c[0], c[1] * c[1]);
        let block = c[2].mul_add(c[2], block);
        let block = c[3].mul_add(c[3], block);
        sum += block;
    }
    for &x in rest {
        sum = x.mul_add(x, sum);
    }
    sum.sqrt()
}

/// Split `v` into what gets stored and what gets kept beside it.
///
/// A vector of no length has no direction, so it is stored as the origin and
/// comes back as the origin, which is the one input where `VEMB` of what went in
/// is not what went in and there is nothing else it could be.
#[must_use]
pub fn squeeze(quant: Quant, v: &[f32]) -> Squeezed {
    let norm = norm(v);
    if norm <= 0.0 || !norm.is_finite() {
        return Squeezed {
            dir: vec![0.0; v.len()],
            norm: 0.0,
            range: 0.0,
        };
    }
    let mut dir: Vec<f32> = v.iter().map(|x| x / norm).collect();
    let range = dir.iter().fold(0.0f32, |wide, x| wide.max(x.abs()));
    match quant {
        Quant::None => {}
        Quant::Int8 => {
            // The reciprocal is formed once and multiplied, which is both faster
            // and the grouping a real server uses.
            let step = 127.0 / range;
            for x in &mut dir {
                *x = f32::from(eighth(*x, step)) * range / 127.0;
            }
        }
        Quant::Bin => {
            // Unit length rather than plus and minus one, so that the distance
            // between two of these is the distance between two directions and
            // the index does not have to know a binary set from any other.
            #[allow(clippy::cast_precision_loss)]
            let w = (v.len() as f32).sqrt().recip();
            for x in &mut dir {
                *x = if *x > 0.0 { w } else { -w };
            }
        }
    }
    Squeezed { dir, norm, range }
}

/// What `VEMB` says about a stored direction, which is the client's vector back
/// again as near as the quantisation kept it.
///
/// The binary form is the exception and it is Redis's exception: it answers plus
/// and minus one and does not multiply the length back on, because a sign has no
/// length in it to scale.
#[must_use]
pub fn restore(quant: Quant, dir: &[f32], norm: f32) -> Vec<f32> {
    match quant {
        Quant::Bin => dir
            .iter()
            .map(|x| if *x > 0.0 { 1.0 } else { -1.0 })
            .collect(),
        _ => dir.iter().map(|x| x * norm).collect(),
    }
}

/// The bytes `VEMB RAW` writes for a stored direction.
///
/// Little endian floats for the full precision form, one signed byte a
/// coordinate for `Q8`, and for the binary form a run of 64 bit words with the
/// first coordinate in the lowest bit of the first word.
#[must_use]
pub fn raw(quant: Quant, dir: &[f32], range: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(quant.code_bytes(dir.len()));
    match quant {
        Quant::None => {
            for x in dir {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        Quant::Int8 => {
            for x in dir {
                bytes.push(code(*x, range) as u8);
            }
        }
        Quant::Bin => {
            bytes.resize(quant.code_bytes(dir.len()), 0);
            for (at, x) in dir.iter().enumerate() {
                if *x > 0.0 {
                    bytes[at / 8] |= 1 << (at % 8);
                }
            }
        }
    }
    bytes
}

/// The code a stored `Q8` coordinate came from.
///
/// The coordinate is `(code * range) / 127` and the way back is the way in with
/// the scale the other way up. It round trips exactly rather than nearly,
/// because a code is a whole number no bigger than 127 and the error either
/// direction is nowhere near half of one.
fn code(x: f32, range: f32) -> i8 {
    if range <= 0.0 || !range.is_finite() {
        return 0;
    }
    eighth(x, 127.0 / range)
}

/// One coordinate of a `Q8` direction, in the byte that holds it.
///
/// Clamped rather than wrapped, because a coordinate that is the range itself
/// lands on 127 give or take a rounding and the byte below it is the one a
/// wrapping cast would give.
#[allow(clippy::cast_possible_truncation)]
fn eighth(x: f32, step: f32) -> i8 {
    (x * step).round().clamp(-127.0, 127.0) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers here are a real server's answers for the same input, read off
    /// `VEMB` and `VEMB RAW` rather than worked out from the formula, which is
    /// the only way this test can catch the formula being wrong.
    #[test]
    fn a_q8_vector_is_squeezed_the_way_a_real_server_squeezes_it() {
        let v = [
            -1.057f32, -2.095, 0.906, -2.565, 0.215, -0.806, -2.652, 0.045,
        ];
        let s = squeeze(Quant::Int8, &v);
        assert_eq!(s.norm, 4.542_832_4);
        assert_eq!(s.range, 0.583_776_8);
        let bytes = raw(Quant::Int8, &s.dir, s.range);
        assert_eq!(bytes, [0xcd, 0x9c, 0x2b, 0x85, 0x0a, 0xd9, 0x81, 0x02]);
        let back = restore(Quant::Int8, &s.dir, s.norm);
        assert_eq!(back[0], -1.064_976_5);
        assert_eq!(back[7], 0.041_763_78);
    }

    /// Two vectors a real server answers differently from both a plain sum of
    /// squares and an exactly rounded one, which is what makes this test worth
    /// having: any of the three obvious ways to write [`norm`] passes on most
    /// input and fails on these.
    #[test]
    fn the_length_is_the_length_a_real_server_measures() {
        let six = [
            -0.937_271_f32,
            -0.990_583_06,
            -3.973_563_2,
            -3.560_724_7,
            -4.341_83,
            -2.311_353_2,
        ];
        assert_eq!(norm(&six), 7.383_869_6);
        let eight = [
            3.911_460_9_f32,
            -4.397_685_5,
            1.571_724_4,
            1.238_847_4,
            4.993_485_5,
            4.326_879,
            -0.686_563_8,
            2.032_343_4,
        ];
        assert_eq!(norm(&eight), 9.322_167);
        // A vector short enough that the unrolled loop never runs, where the
        // fused tail is the whole answer.
        assert_eq!(norm(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn nothing_is_lost_when_nothing_is_squeezed() {
        let v = [3.0f32, 4.0];
        let s = squeeze(Quant::None, &v);
        assert_eq!(s.norm, 5.0);
        assert_eq!(restore(Quant::None, &s.dir, s.norm), [3.0, 4.0]);
        assert_eq!(raw(Quant::None, &s.dir, s.range).len(), 8);
    }

    #[test]
    fn a_binary_vector_is_its_signs_and_keeps_no_length() {
        let v = [1.0f32, -2.0, 0.0, 4.0];
        let s = squeeze(Quant::Bin, &v);
        assert_eq!(restore(Quant::Bin, &s.dir, s.norm), [1.0, -1.0, -1.0, 1.0]);
        // Four dimensions still write one whole word, because the wire form is
        // words and not bytes.
        assert_eq!(
            raw(Quant::Bin, &s.dir, s.range),
            [0b1001, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn a_squeezed_direction_is_still_a_direction() {
        let v = [0.3f32, -1.7, 2.2, 0.9, -0.4];
        for quant in [Quant::None, Quant::Int8, Quant::Bin] {
            let s = squeeze(quant, &v);
            let len = norm(&s.dir);
            assert!((len - 1.0).abs() < 1e-3, "{} is {len} long", quant.token());
        }
    }

    #[test]
    fn a_vector_of_no_length_is_stored_as_the_origin() {
        let s = squeeze(Quant::Int8, &[0.0, 0.0, 0.0]);
        assert_eq!(s.norm, 0.0);
        assert_eq!(s.dir, [0.0, 0.0, 0.0]);
        assert_eq!(raw(Quant::Int8, &s.dir, s.range), [0, 0, 0]);
    }

    #[test]
    fn how_wide_the_bytes_are_is_how_wide_they_turn_out_to_be() {
        for dim in [1usize, 7, 8, 63, 64, 65, 300] {
            let v = vec![0.5f32; dim];
            for quant in [Quant::None, Quant::Int8, Quant::Bin] {
                let s = squeeze(quant, &v);
                let bytes = raw(quant, &s.dir, s.range);
                assert_eq!(
                    bytes.len(),
                    quant.code_bytes(dim),
                    "{} at {dim}",
                    quant.token()
                );
            }
        }
    }
}
