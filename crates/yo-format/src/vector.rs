//! What a vector record holds, which is the vector itself at full precision.
//!
//! `06` section 2.1 gives kind 3 to a vector and says nothing about what is
//! inside it, and `10` section 2 says why there is anything inside it at all:
//! the searchable form of a vector is a RaBitQ code in a posting, and the code
//! is lossy, so the last step of a search measures the best few candidates
//! against the real thing. Every other engine that quantises has to keep the
//! raw vectors somewhere on purpose. Here a vector is a record like any other
//! and the rerank is a read at an address the id already resolves to.
//!
//! ```text
//! +---------+---------+---------+---------+----------------------+
//! |   dim   | element |  flags  | reserved|  dim * width bytes   |
//! |    4    |    1    |    1    |    2    |                      |
//! +---------+---------+---------+---------+----------------------+
//! ```
//!
//! # Why the dimension is in the record
//!
//! The collection knows its own dimension, it is in the catalogue, and a
//! reader that has the catalogue could work the count out from the record
//! length. Storing it anyway costs four bytes and buys two things. A record is
//! checkable on its own, so `yodb check` and the independent reader can say
//! that a vector record is malformed without loading the catalogue for the
//! collection it belongs to, and a collection whose dimension was changed under
//! it produces a record that disagrees with the catalogue rather than a vector
//! that is silently reinterpreted at a different length.
//!
//! # Why there is an element byte when there is one element type
//!
//! Everything writes and reads [`Element::F32`] today. The byte is there
//! because after the freeze at the end of M6 the only lever left is
//! `min_reader_version`, and half precision storage is the change most likely
//! to be wanted: it halves what the log holds for a vector collection, which is
//! most of what a vector collection is. With the byte here that lands as a new
//! element value which old readers refuse one record at a time. Without it, it
//! would need a new record kind or a format version, and both of those refuse
//! the whole file.
//!
//! An unknown element value is [`Code::Corrupt`], not a record to skip. Skipping
//! is the right answer for an unknown `kind`, because a kind a reader has never
//! heard of is a thing it was never meant to understand. An element it cannot
//! read inside a kind it can is different: the caller asked for this vector, and
//! quietly returning nothing would look like a vector that is not there.
//!
//! # What is not in here
//!
//! The codes, the centroids and the postings. Those are the index, the index is
//! derived from these records, and `10` section 2 keeps them resident or in the
//! cold tier rather than in the log. When they do get written down it will be
//! as collection chunks under a checkpoint, because the record kinds are fixed
//! by `06` and there is no kind for an index.
//!
//! The id, because that is the record's key, and the tag a filtered scan reads,
//! because that is derived from the document the vector belongs to.

use crate::{get_u8, get_u16, get_u32, put_u8, put_u16, put_u32};
use yo_common::{Code, Error, Result};

/// The fixed part at the front of a vector record's value.
pub const VECTOR_HEADER_LEN: usize = 8;

/// The largest dimension this version will write or read.
///
/// Sixty five thousand is far past every published embedding family, which
/// stops at a few thousand, and it is small enough that a corrupt `dim` cannot
/// ask a reader for a gigabyte. The real limit is lower and it is not this: a
/// record has to fit a page, so at f32 a vector is bounded by the page size,
/// and a collection of vectors this long would want the chunked form (`05`
/// section 4.4) instead.
pub const MAX_DIM: usize = 65_536;

/// How one coordinate is stored.
///
/// The values are part of the format. See the module note on why there is a
/// byte for this when there is only one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Element {
    /// Little endian `f32`, four bytes.
    F32 = 0,
}

impl Element {
    /// The byte that stands for this element type.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The element type for a byte, or `None` if this version has not heard of
    /// it.
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<Element> {
        match b {
            0 => Some(Element::F32),
            _ => None,
        }
    }

    /// How many bytes one coordinate takes.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Element::F32 => 4,
        }
    }
}

/// How long the value of a vector record with `dim` coordinates is.
///
/// # Errors
///
/// [`Code::Invalid`] if `dim` is zero or past [`MAX_DIM`]. A zero dimensional
/// vector is not a small vector, it is a mistake somewhere upstream, and it
/// would sort as equidistant from everything.
pub fn vector_len(dim: usize, of: Element) -> Result<usize> {
    if dim == 0 || dim > MAX_DIM {
        return Err(Error::new(Code::Invalid, "dimension out of range")
            .with_detail(format!("dim={dim} max={MAX_DIM}")));
    }
    Ok(VECTOR_HEADER_LEN + dim * of.width())
}

/// A vector record's value, borrowed.
///
/// Decoding does not copy the coordinates and does not check them. What it
/// checks is that the header is one this version understands and that the
/// coordinates that the header claims are all there, which is what stops a
/// short or corrupt record from being read as a vector of the wrong length.
#[derive(Debug, Clone, Copy)]
pub struct VectorBody<'a> {
    dim: usize,
    element: Element,
    values: &'a [u8],
}

impl<'a> VectorBody<'a> {
    /// Writes `values` into `into` and says how many bytes that took.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the dimension is out of range, if `into` is too
    /// short, or if any coordinate is not finite.
    ///
    /// The last one is a refusal at the boundary rather than a thing to sort
    /// out later. A NaN coordinate makes every distance involving that vector a
    /// NaN, a NaN compares false against everything, and the result is not an
    /// error anywhere: the vector simply never wins and never loses, and it
    /// quietly distorts the centroid of whatever partition it lands in. That is
    /// a bug report about recall six months later, and it costs one pass over a
    /// buffer that is being copied anyway to make it impossible.
    pub fn encode(values: &[f32], into: &mut [u8]) -> Result<usize> {
        let need = vector_len(values.len(), Element::F32)?;
        if into.len() < need {
            return Err(
                Error::new(Code::Invalid, "buffer is shorter than the vector")
                    .with_detail(format!("have={} need={need}", into.len())),
            );
        }
        if let Some(at) = values.iter().position(|v| !v.is_finite()) {
            return Err(Error::new(Code::Invalid, "a coordinate is not a number")
                .with_detail(format!("at={at} value={}", values[at])));
        }
        put_u32(into, 0, values.len() as u32);
        put_u8(into, 4, Element::F32.as_u8());
        put_u8(into, 5, 0);
        put_u16(into, 6, 0);
        for (i, v) in values.iter().enumerate() {
            let at = VECTOR_HEADER_LEN + i * 4;
            into[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(need)
    }

    /// Reads the header back and borrows the coordinates.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the header is not one this version understands or
    /// if the record is not as long as its own header says it is.
    pub fn decode(bytes: &'a [u8]) -> Result<VectorBody<'a>> {
        if bytes.len() < VECTOR_HEADER_LEN {
            return Err(Error::new(Code::Corrupt, "shorter than a vector header")
                .with_detail(format!("len={}", bytes.len())));
        }
        let dim = get_u32(bytes, 0) as usize;
        let raw = get_u8(bytes, 4);
        let Some(element) = Element::from_u8(raw) else {
            return Err(Error::new(Code::Corrupt, "unknown vector element type")
                .with_detail(format!("element={raw}")));
        };
        // Reserved bytes are checked rather than ignored. A version that gives
        // them a meaning will say so with `min_reader_version`, and until then
        // a record with anything in them was written by something that did not
        // agree with this layout.
        let flags = get_u8(bytes, 5);
        let reserved = get_u16(bytes, 6);
        if flags != 0 || reserved != 0 {
            return Err(
                Error::new(Code::Corrupt, "reserved vector header bytes are set")
                    .with_detail(format!("flags={flags:#04x} reserved={reserved:#06x}")),
            );
        }
        // The same range as `vector_len` and a different code, because a
        // dimension a caller passed in is a mistake and one that came off a
        // disk is a broken record.
        if dim == 0 || dim > MAX_DIM {
            return Err(Error::new(Code::Corrupt, "vector dimension out of range")
                .with_detail(format!("dim={dim} max={MAX_DIM}")));
        }
        let need = VECTOR_HEADER_LEN + dim * element.width();
        if bytes.len() < need {
            return Err(
                Error::new(Code::Corrupt, "vector record is shorter than its dimension")
                    .with_detail(format!("len={} need={need} dim={dim}", bytes.len())),
            );
        }
        Ok(VectorBody {
            dim,
            element,
            values: &bytes[VECTOR_HEADER_LEN..need],
        })
    }

    /// How many coordinates the vector has.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// How the coordinates are stored.
    #[must_use]
    pub const fn element(&self) -> Element {
        self.element
    }

    /// Copies the coordinates into `out`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if `out` is not exactly [`dim`](Self::dim) long. Not a
    /// prefix and not a longer buffer, because both of those are a caller that
    /// thinks the collection has a different dimension than it does, and the
    /// only useful moment to say so is here.
    pub fn read_into(&self, out: &mut [f32]) -> Result<()> {
        if out.len() != self.dim {
            return Err(
                Error::new(Code::Invalid, "buffer is not the vector's length")
                    .with_detail(format!("have={} dim={}", out.len(), self.dim)),
            );
        }
        match self.element {
            Element::F32 => {
                for (i, slot) in out.iter_mut().enumerate() {
                    let at = i * 4;
                    let b = &self.values[at..at + 4];
                    *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
        Ok(())
    }

    /// One coordinate, or `None` past the end.
    #[must_use]
    pub fn at(&self, i: usize) -> Option<f32> {
        if i >= self.dim {
            return None;
        }
        match self.element {
            Element::F32 => {
                let b = self.values.get(i * 4..i * 4 + 4)?;
                Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
        }
    }

    /// The coordinates as they are stored, for a caller that is copying a
    /// record rather than reading it.
    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(values: &[f32]) -> Vec<f32> {
        let mut buf = vec![0u8; vector_len(values.len(), Element::F32).unwrap()];
        let wrote = VectorBody::encode(values, &mut buf).unwrap();
        assert_eq!(
            wrote,
            buf.len(),
            "encode wrote a different length than it asked for"
        );
        let body = VectorBody::decode(&buf).unwrap();
        assert_eq!(body.dim(), values.len());
        assert_eq!(body.element(), Element::F32);
        let mut out = vec![0f32; body.dim()];
        body.read_into(&mut out).unwrap();
        out
    }

    #[test]
    fn a_vector_comes_back_bit_for_bit() {
        // Exactly, not nearly. Rerank is the step that decides the final
        // ordering, so a vector that comes back close enough is a vector that
        // reorders results for no reason anybody could find.
        let values = [0.0, -0.0, 1.0, -1.0, 1e-38, 3.4e38, 0.1, 2.5];
        assert_eq!(round_trip(&values), values);
    }

    #[test]
    fn a_long_vector_is_fine() {
        let values: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        assert_eq!(round_trip(&values), values);
    }

    #[test]
    fn coordinates_can_be_read_one_at_a_time() {
        let values = [3.0f32, 1.0, 4.0, 1.5];
        let mut buf = vec![0u8; vector_len(4, Element::F32).unwrap()];
        VectorBody::encode(&values, &mut buf).unwrap();
        let body = VectorBody::decode(&buf).unwrap();
        for (i, want) in values.iter().enumerate() {
            assert_eq!(body.at(i), Some(*want));
        }
        assert_eq!(body.at(4), None, "past the end is not a coordinate");
    }

    #[test]
    fn a_vector_that_is_not_a_vector_is_refused() {
        let mut buf = vec![0u8; 64];
        assert!(VectorBody::encode(&[], &mut buf).is_err(), "no dimension");
        assert!(
            VectorBody::encode(&[f32::NAN, 1.0], &mut buf).is_err(),
            "a NaN coordinate poisons every distance it takes part in"
        );
        assert!(VectorBody::encode(&[f32::INFINITY], &mut buf).is_err());
        let mut tiny = [0u8; 8];
        assert!(
            VectorBody::encode(&[1.0, 2.0], &mut tiny).is_err(),
            "the header fits and the coordinates do not"
        );
    }

    #[test]
    fn a_record_shorter_than_it_claims_is_corrupt() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let mut buf = vec![0u8; vector_len(4, Element::F32).unwrap()];
        VectorBody::encode(&values, &mut buf).unwrap();
        for len in 0..buf.len() {
            assert!(
                VectorBody::decode(&buf[..len]).is_err(),
                "{len} bytes decoded as a four dimensional vector"
            );
        }
        assert!(VectorBody::decode(&buf).is_ok());
    }

    #[test]
    fn an_element_type_this_version_does_not_know_is_refused() {
        let mut buf = vec![0u8; vector_len(2, Element::F32).unwrap()];
        VectorBody::encode(&[1.0, 2.0], &mut buf).unwrap();
        buf[4] = 1;
        let e = VectorBody::decode(&buf).unwrap_err();
        assert_eq!(e.code(), Code::Corrupt);
    }

    #[test]
    fn reserved_bytes_have_to_be_zero() {
        let values = [1.0f32, 2.0];
        for at in [5usize, 6, 7] {
            let mut buf = vec![0u8; vector_len(2, Element::F32).unwrap()];
            VectorBody::encode(&values, &mut buf).unwrap();
            buf[at] = 1;
            assert!(
                VectorBody::decode(&buf).is_err(),
                "byte {at} is reserved and a set bit in it means the writer disagreed with this layout"
            );
        }
    }

    #[test]
    fn a_dimension_that_could_not_fit_anywhere_is_refused_before_it_is_believed() {
        let mut buf = vec![0u8; vector_len(2, Element::F32).unwrap()];
        VectorBody::encode(&[1.0, 2.0], &mut buf).unwrap();
        put_u32(&mut buf, 0, u32::MAX);
        let e = VectorBody::decode(&buf).unwrap_err();
        assert_eq!(e.code(), Code::Corrupt);
        assert!(vector_len(MAX_DIM + 1, Element::F32).is_err());
    }

    #[test]
    fn reading_into_the_wrong_length_says_so() {
        let mut buf = vec![0u8; vector_len(3, Element::F32).unwrap()];
        VectorBody::encode(&[1.0, 2.0, 3.0], &mut buf).unwrap();
        let body = VectorBody::decode(&buf).unwrap();
        assert!(body.read_into(&mut [0.0; 2]).is_err(), "too short");
        assert!(body.read_into(&mut [0.0; 4]).is_err(), "too long");
        assert!(body.read_into(&mut [0.0; 3]).is_ok());
    }

    #[test]
    fn the_element_byte_maps_both_ways() {
        assert_eq!(Element::from_u8(Element::F32.as_u8()), Some(Element::F32));
        assert_eq!(Element::from_u8(1), None);
        assert_eq!(Element::F32.width(), 4);
    }
}
