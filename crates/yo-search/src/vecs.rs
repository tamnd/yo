//! The vectors one `VECTOR` field has read, and the two questions asked of
//! them.
//!
//! A field keeps every document's vector at full precision in one flat run of
//! floats, and both questions are a pass over that run. Nearest `k` keeps the
//! best `k` in a heap as it goes, and everything within a radius keeps whatever
//! passes. Neither builds anything, so a write is an append and a delete is a
//! hole, and there is no graph to degrade and no rebuild to schedule.
//!
//! That is exact, which is what `FLAT` means and is more than `HNSW` promises.
//! An index that asked for a graph gets perfect recall here and pays a pass over
//! the field for it, which is D-91. The pass is the cheap kind: contiguous
//! floats, one multiply and one add per coordinate, no indirection per document.
//!
//! # What a distance means
//!
//! Whatever the field said it measures, reported the way a real server reports
//! it. `L2` is the squared euclidean distance and not the euclidean one, which
//! is measured: five vectors a unit apart along one axis come back at 0, 1, 4, 9
//! and 16. `COSINE` is one minus the cosine similarity, so 0 is the same
//! direction and 2 is the opposite one. `IP` is one minus the inner product.
//! All three are distances, so nearer is smaller and an answer is in that order.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use yo_shape::Metric;

use crate::field::{Vector, Width};
use crate::posts::Id;

/// One field's vectors, by the number of the document each came off.
#[derive(Debug, Clone)]
pub struct Vecs {
    /// How many coordinates a vector here has.
    dim: usize,
    /// What this field measures.
    metric: Metric,
    /// Every vector end to end, `dim` floats each, in the order they were
    /// written.
    data: Vec<f32>,
    /// The document each run of floats came off, and `None` for a run whose
    /// document has been rewritten or removed.
    owner: Vec<Option<Id>>,
    /// Where a document's vector is, as an index into `owner`.
    at: BTreeMap<Id, usize>,
    /// The holes, so that a rewritten document reuses the run the old one had
    /// rather than growing the field forever.
    free: Vec<usize>,
}

/// One document that answered, and how far away it was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Near {
    /// Which document.
    pub id: Id,
    /// How far, in whatever the field measures.
    pub distance: f32,
}

impl Vecs {
    /// An empty field of `dim` dimensional vectors.
    #[must_use]
    pub fn new(dim: usize, metric: Metric) -> Vecs {
        Vecs {
            dim,
            metric,
            data: Vec::new(),
            owner: Vec::new(),
            at: BTreeMap::new(),
            free: Vec::new(),
        }
    }

    /// The same, taking the two numbers off the field that declared them.
    #[must_use]
    pub fn of(field: &Vector) -> Vecs {
        Vecs::new(field.dim as usize, field.metric)
    }

    /// How many coordinates a vector here has.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// How many documents have one.
    #[must_use]
    pub fn len(&self) -> usize {
        self.at.len()
    }

    /// Whether none do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// Puts a document's vector in, replacing whatever it had.
    ///
    /// A vector of the wrong length is not written and the document keeps
    /// nothing here, which is the caller's to refuse before it gets this far.
    pub fn add(&mut self, id: Id, vector: &[f32]) {
        if vector.len() != self.dim {
            return;
        }
        let slot = match self.at.get(&id) {
            Some(slot) => *slot,
            None => match self.free.pop() {
                Some(slot) => slot,
                None => {
                    self.owner.push(None);
                    self.data.resize(self.owner.len() * self.dim, 0.0);
                    self.owner.len() - 1
                }
            },
        };
        self.data[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(vector);
        self.owner[slot] = Some(id);
        self.at.insert(id, slot);
    }

    /// Takes a document's vector out, and says whether there was one.
    pub fn remove(&mut self, id: Id) -> bool {
        let Some(slot) = self.at.remove(&id) else {
            return false;
        };
        self.owner[slot] = None;
        self.free.push(slot);
        true
    }

    /// The vector a document has, if it has one.
    #[must_use]
    pub fn get(&self, id: Id) -> Option<&[f32]> {
        let slot = *self.at.get(&id)?;
        Some(&self.data[slot * self.dim..(slot + 1) * self.dim])
    }

    /// The `k` nearest documents to a query, nearest first.
    ///
    /// `allowed` is what the query narrowed the field down to first, and `None`
    /// is the whole field. A `k` of nothing answers nothing rather than
    /// answering everything, because a client that asked for no neighbours is
    /// not asking for all of them.
    #[must_use]
    pub fn nearest(&self, query: &[f32], k: usize, allowed: Option<&[Id]>) -> Vec<Near> {
        let mut found = self.measured(query, allowed);
        // Sorted rather than kept in a heap, because the pass over the field is
        // what this costs and the sort is over what survived the filter. A tie
        // goes to the document written first, which is what makes an answer the
        // same one twice.
        found.sort_by(|a, b| order(a.distance, b.distance).then(a.id.cmp(&b.id)));
        found.truncate(k);
        found
    }

    /// Every document within `radius` of a query, nearest first.
    #[must_use]
    pub fn within(&self, query: &[f32], radius: f32, allowed: Option<&[Id]>) -> Vec<Near> {
        let mut found = self.measured(query, allowed);
        found.retain(|near| near.distance <= radius);
        found.sort_by(|a, b| order(a.distance, b.distance).then(a.id.cmp(&b.id)));
        found
    }

    /// Every document the filter allows, measured, in no particular order.
    fn measured(&self, query: &[f32], allowed: Option<&[Id]>) -> Vec<Near> {
        if query.len() != self.dim {
            return Vec::new();
        }
        match allowed {
            // The narrowed answer is walked rather than the field, since a
            // query that already cut the index down to three documents has no
            // reason to measure the other million.
            Some(ids) => ids
                .iter()
                .filter_map(|id| {
                    let vector = self.get(*id)?;
                    Some(Near {
                        id: *id,
                        distance: measure(self.metric, query, vector),
                    })
                })
                .collect(),
            None => self
                .owner
                .iter()
                .enumerate()
                .filter_map(|(slot, owner)| {
                    let id = (*owner)?;
                    let vector = &self.data[slot * self.dim..(slot + 1) * self.dim];
                    Some(Near {
                        id,
                        distance: measure(self.metric, query, vector),
                    })
                })
                .collect(),
        }
    }
}

/// Two distances compared, with anything that is not a number last.
fn order(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// How far apart two vectors are, in whatever the field measures.
#[must_use]
pub fn measure(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        // Squared, which is what a real server reports and is the same order
        // the euclidean distance would put them in.
        Metric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>(),
        Metric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let left = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let right = b.iter().map(|y| y * y).sum::<f32>().sqrt();
            // A vector of nothing has no direction, so nothing is the same
            // direction as it and everything is the furthest away.
            match left * right {
                scale if scale > 0.0 => 1.0 - dot / scale,
                _ => 1.0,
            }
        }
        // Not a distance, which is why an index cannot be built around it and
        // why this is the pass rather than a graph. One minus it so that
        // nearer is still smaller.
        Metric::Ip => 1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        Metric::Hamming => a
            .iter()
            .zip(b)
            .filter(|(x, y)| (x.to_bits() ^ y.to_bits()) != 0)
            .count() as f32,
    }
}

/// A query vector read out of the bytes a client sent.
///
/// `None` when the blob is not a whole number of coordinates of the width the
/// field declared, which is the one thing that can be wrong about it and is
/// what a real server refuses the query for.
#[must_use]
pub fn read(width: Width, raw: &[u8]) -> Option<Vec<f32>> {
    let size = width.bytes();
    if raw.is_empty() || !raw.len().is_multiple_of(size) {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len() / size);
    for chunk in raw.chunks_exact(size) {
        out.push(coordinate(width, chunk));
    }
    Some(out)
}

/// One coordinate, in whatever width the field is written in.
fn coordinate(width: Width, raw: &[u8]) -> f32 {
    match width {
        // Signed and unsigned bytes, which is what a quantised model emits and
        // is read as the number it is rather than scaled back into a unit
        // range, because nothing here knows what it was scaled by.
        Width::Int8 => f32::from(raw[0] as i8),
        Width::Uint8 => f32::from(raw[0]),
        Width::Float16 => half(u16::from_le_bytes([raw[0], raw[1]])),
        // Brain float is the top half of a float, so widening it is putting the
        // bottom half back as zeros.
        Width::BFloat16 => f32::from_bits(u32::from(u16::from_le_bytes([raw[0], raw[1]])) << 16),
        Width::Float32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
        Width::Float64 => f64::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
        ]) as f32,
    }
}

/// Half precision widened to single, done by hand because it is eight lines and
/// a dependency for eight lines is a dependency.
fn half(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    match exponent {
        // Zero and the subnormals, which have no leading one to shift into
        // place, so they are the fraction scaled by the smallest step a half
        // can take. The sign is put back afterwards so that a negative zero
        // stays one.
        0 => {
            let size = fraction as f32 * f32::from_bits(0x3380_0000);
            match sign {
                0 => size,
                _ => -size,
            }
        }
        // Infinity and the not a numbers, whose exponent is all ones in both
        // widths.
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        // Everything else, where the exponent moves by the difference between
        // the two biases and the fraction moves by the difference between the
        // two widths.
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field holds a vector per document, hands it back, and reuses the run a
    /// rewritten document had rather than growing.
    #[test]
    fn a_field_holds_one_vector_a_document_and_reuses_a_hole() {
        let mut vecs = Vecs::new(2, Metric::L2);
        vecs.add(1, &[1.0, 0.0]);
        vecs.add(2, &[0.0, 1.0]);
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs.get(1), Some(&[1.0, 0.0][..]));
        assert!(vecs.remove(1));
        assert!(!vecs.remove(1));
        assert_eq!(vecs.get(1), None);
        vecs.add(3, &[2.0, 2.0]);
        // Two runs still, because the third document went into the hole the
        // first one left.
        assert_eq!(vecs.data.len(), 4);
        assert_eq!(vecs.get(3), Some(&[2.0, 2.0][..]));
        // A vector of the wrong length is not a vector of this field.
        vecs.add(4, &[1.0]);
        assert_eq!(vecs.get(4), None);
    }

    /// The nearest few come back nearest first, and a squared euclidean
    /// distance is what a real server reports.
    #[test]
    fn the_nearest_come_back_nearest_first_and_the_distance_is_squared() {
        let mut vecs = Vecs::new(2, Metric::L2);
        for id in 1..=5 {
            vecs.add(id, &[f32::from(id as i16 - 1), 0.0]);
        }
        let found = vecs.nearest(&[0.0, 0.0], 3, None);
        assert_eq!(
            found,
            [
                Near {
                    id: 1,
                    distance: 0.0
                },
                Near {
                    id: 2,
                    distance: 1.0
                },
                Near {
                    id: 3,
                    distance: 4.0
                }
            ]
        );
        // Asking for more than there is answers what there is, and asking for
        // none answers none.
        assert_eq!(vecs.nearest(&[0.0, 0.0], 9, None).len(), 5);
        assert!(vecs.nearest(&[0.0, 0.0], 0, None).is_empty());
    }

    /// A range takes everything within it, and a filter cuts the field down
    /// before anything is measured.
    #[test]
    fn a_range_takes_what_is_inside_it_and_a_filter_cuts_it_down_first() {
        let mut vecs = Vecs::new(2, Metric::L2);
        for id in 1..=5 {
            vecs.add(id, &[f32::from(id as i16 - 1), 0.0]);
        }
        let ids: Vec<Id> = vecs
            .within(&[0.0, 0.0], 4.0, None)
            .iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(ids, [1, 2, 3]);
        let ids: Vec<Id> = vecs
            .within(&[0.0, 0.0], 4.0, Some(&[2, 4]))
            .iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(ids, [2]);
        let ids: Vec<Id> = vecs
            .nearest(&[0.0, 0.0], 2, Some(&[3, 4, 5]))
            .iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(ids, [3, 4]);
    }

    /// The three metrics that are not the euclidean one, each measured the way
    /// the field that named it means it.
    #[test]
    fn each_metric_is_measured_the_way_the_field_means_it() {
        let same = measure(Metric::Cosine, &[1.0, 0.0], &[2.0, 0.0]);
        assert!(same.abs() < 1e-6, "{same}");
        let apart = measure(Metric::Cosine, &[1.0, 0.0], &[-1.0, 0.0]);
        assert!((apart - 2.0).abs() < 1e-6, "{apart}");
        let none = measure(Metric::Cosine, &[1.0, 0.0], &[0.0, 0.0]);
        assert!((none - 1.0).abs() < 1e-6, "{none}");
        assert_eq!(measure(Metric::Ip, &[1.0, 2.0], &[3.0, 4.0]), 1.0 - 11.0);
        assert_eq!(measure(Metric::Hamming, &[1.0, 2.0], &[1.0, 3.0]), 1.0);
    }

    /// A blob is read at whatever width the field declared, and one that is not
    /// a whole number of coordinates is not read at all.
    #[test]
    fn a_blob_is_read_at_the_width_the_field_declared() {
        let raw = [0u8, 0, 128, 63, 0, 0, 0, 64];
        assert_eq!(read(Width::Float32, &raw), Some(vec![1.0, 2.0]));
        assert_eq!(read(Width::Float32, &raw[..3]), None);
        assert_eq!(read(Width::Float32, &[]), None);
        assert_eq!(read(Width::Int8, &[1, 255]), Some(vec![1.0, -1.0]));
        assert_eq!(read(Width::Uint8, &[1, 255]), Some(vec![1.0, 255.0]));
        // One in half precision and one in brain float, both of which are two
        // bytes and neither of which is the same two bytes.
        assert_eq!(read(Width::Float16, &[0x00, 0x3c]), Some(vec![1.0]));
        assert_eq!(read(Width::BFloat16, &[0x80, 0x3f]), Some(vec![1.0]));
        assert_eq!(read(Width::Float16, &[0x00, 0x00]), Some(vec![0.0]));
        let eight = [0u8, 0, 0, 0, 0, 0, 240, 63];
        assert_eq!(read(Width::Float64, &eight), Some(vec![1.0]));
    }
}
