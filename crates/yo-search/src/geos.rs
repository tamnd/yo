//! The geo index: which documents hold a point, and which of those fall inside
//! a circle.
//!
//! ```
//! use yo_search::geos::Geos;
//!
//! let mut g = Geos::new();
//! g.add(1, -0.1278, 51.5074);
//! g.add(2, 2.3522, 48.8566);
//! g.settle();
//!
//! assert_eq!(g.circle(-0.1278, 51.5074, 10.0, b"km"), Some(vec![1]));
//! assert_eq!(g.circle(-0.1278, 51.5074, 1000.0, b"km"), Some(vec![1, 2]));
//! assert_eq!(g.circle(0.0, 0.0, 1.0, b"yards"), None);
//! ```
//!
//! # Why this is the numeric index again
//!
//! A point is a 52 bit interleave of its longitude and latitude, which is a
//! number an `f64` holds exactly, and two points that are near each other on
//! the ground have numbers that are near each other most of the time. So the
//! whole index is [`crate::nums::Nums`] with the coordinates folded into the
//! value, and a circle is nine ranges out of it, one for the box the centre
//! falls in and one for each box around it, at a precision picked so the nine
//! cover the circle. That is what Redis does for `GEOSEARCH` and it is what
//! RediSearch does for a `GEO` field, so agreeing with both is a matter of
//! sharing one copy of the arithmetic rather than of matching two.
//!
//! # Where the boundary is
//!
//! A stored point is the middle of the smallest box the 52 bits name, which is
//! under a metre across, so the distance a filter measures is the distance to
//! that middle and not to the coordinates the client wrote. Both servers round
//! the same way, so a document either side of the radius lands the same way in
//! both, which is the only thing that matters here.

use crate::nums::{Ends, Nums};
use crate::posts::Id;
use yo_common::geo::{self, Kind, Shape, Unit};

/// Every point one field holds, over every document that has it.
#[derive(Debug, Clone, Default)]
pub struct Geos {
    points: Nums,
}

impl Geos {
    /// An index with nothing in it.
    #[must_use]
    pub fn new() -> Geos {
        Geos::default()
    }

    /// Records that a document holds a point, in degrees.
    ///
    /// A point outside the limits is dropped rather than clamped, though the
    /// reader refuses one before it gets this far.
    pub fn add(&mut self, id: Id, lon: f64, lat: f64) {
        if !geo::in_range(lon, lat) {
            return;
        }
        let (lon, lat) = inside(lon, lat);
        if let Some(score) = geo::score(lon, lat) {
            self.points.add(id, score as f64);
        }
    }

    /// Folds everything that has arrived into the ordered list.
    pub fn settle(&mut self) {
        self.points.settle();
    }

    /// The documents inside the circle, in order and each once.
    ///
    /// `None` when the unit is not one of the four, which is the one thing here
    /// a caller has to report rather than answer.
    #[must_use]
    pub fn circle(&self, lon: f64, lat: f64, radius: f64, unit: &[u8]) -> Option<Vec<Id>> {
        let unit = Unit::parse(unit)?;
        let (lon, lat) = inside(lon, lat);
        let shape = Shape {
            lon,
            lat,
            kind: Kind::Circle { radius },
            unit,
        };
        let mut out = Vec::new();
        let search = geo::areas(&shape);
        for hash in search.boxes {
            if hash.bits == 0 && hash.step == 0 {
                continue;
            }
            let (low, high) = geo::range(hash);
            // The top of a box is the bottom of the next one, so it is left out
            // here and picked up by whichever box owns it.
            let ends = Ends::shut(low as f64, high as f64).top_open();
            self.points.within(ends, |id, raw| {
                if let Some((lon, lat)) = geo::decode(raw)
                    && shape.covers(lon, lat).is_some()
                {
                    out.push(id);
                }
            });
        }
        out.sort_unstable();
        out.dedup();
        Some(out)
    }

    /// How many points are held, counting a document twice if it holds two.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether no document holds a point here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The largest document number seen, or zero when there is none.
    #[must_use]
    pub const fn last(&self) -> Id {
        self.points.last()
    }

    /// How many bytes the entries take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.points.bytes()
    }
}

/// How far inside the top of an axis a point on the edge is pulled.
///
/// A nanodegree, which is a tenth of a millimetre on the ground and a whole
/// bucket on a twenty six bit axis.
const EDGE: f64 = 1e-9;

/// A point pulled a hair inside the top of each axis.
///
/// Longitude 180 and the top latitude scale to one bucket past their axis, and
/// the score that comes out carries a fifty third or a fifty fourth bit that no
/// ordinary box covers. Redis stores a place that way and its own `GEOSEARCH`
/// then cannot find it from anywhere except that exact spot, which is measured:
/// `GEOSEARCH` around 179.99 misses a member at 180 and a search index does
/// not. So the top edge lands in the last bucket of its axis here rather than
/// one past it, on the way in and on the way out both, and a point that is not
/// on an edge is untouched.
fn inside(lon: f64, lat: f64) -> (f64, f64) {
    (lon.min(geo::LON_MAX - EDGE), lat.min(geo::LAT_MAX - EDGE))
}

/// A field read as a point, the way a real server reads one.
///
/// The value is cut at the first space or comma and both halves are read as
/// numbers, so `12,34` and `12 34` and `12, 34` are all the same point and
/// ` 12,34` is not a point at all: the leading space is the separator, which
/// leaves nothing in front of it to read. A half is allowed space around it and
/// nothing else, so `12,34,56` is refused rather than read as the first two.
/// Every one of those is measured rather than reasoned about.
#[must_use]
pub fn point(raw: &[u8]) -> Option<(f64, f64)> {
    let at = raw.iter().position(|b| *b == b' ' || *b == b',')?;
    let lon = half(&raw[..at])?;
    let lat = half(&raw[at + 1..])?;
    geo::in_range(lon, lat).then_some((lon, lat))
}

/// One side of a point, with the space around it taken off.
fn half(raw: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(raw).ok()?.trim_ascii();
    (!text.is_empty()).then_some(())?;
    text.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_is_read_the_way_a_real_server_reads_one() {
        assert_eq!(point(b"0,0"), Some((0.0, 0.0)));
        assert_eq!(point(b"0 0"), Some((0.0, 0.0)));
        assert_eq!(point(b"0, 0"), Some((0.0, 0.0)));
        assert_eq!(point(b"0,0 "), Some((0.0, 0.0)));
        assert_eq!(point(b"1e1,1e1"), Some((10.0, 10.0)));
        assert_eq!(point(b"+0,-0"), Some((0.0, -0.0)));
        assert_eq!(point(b"0,85.05112878"), Some((0.0, 85.051_128_78)));
        // The leading space is the separator, so there is no longitude.
        assert_eq!(point(b" 0,0"), None);
        assert_eq!(point(b"0,0,0"), None);
        assert_eq!(point(b"0;0"), None);
        assert_eq!(point(b""), None);
        assert_eq!(point(b"x"), None);
        assert_eq!(point(b"200,0"), None);
        assert_eq!(point(b"0,100"), None);
        assert_eq!(point(b"0,85.06"), None);
        assert_eq!(point(b"nan,0"), None);
        assert_eq!(point(b"inf,0"), None);
    }

    #[test]
    fn a_circle_holds_what_is_inside_it_and_nothing_else() {
        let mut g = Geos::new();
        // London, three hundred and fifty metres off it, five kilometres off
        // it, and Paris.
        g.add(1, -0.1278, 51.5074);
        g.add(2, -0.1300, 51.5100);
        g.add(3, -0.2000, 51.5300);
        g.add(4, 2.3522, 48.8566);
        g.settle();
        let at = |radius, unit| g.circle(-0.1278, 51.5074, radius, unit);
        assert_eq!(at(1.0, b"m"), Some(vec![1]));
        assert_eq!(at(1.0, b"km"), Some(vec![1, 2]));
        assert_eq!(at(1.0, b"mi"), Some(vec![1, 2]));
        assert_eq!(at(3000.0, b"ft"), Some(vec![1, 2]));
        assert_eq!(at(10.0, b"km"), Some(vec![1, 2, 3]));
        assert_eq!(at(1000.0, b"km"), Some(vec![1, 2, 3, 4]));
        assert_eq!(at(1.0, b"yd"), None);
    }

    #[test]
    fn a_circle_over_the_date_line_holds_both_sides_of_it() {
        let mut g = Geos::new();
        g.add(1, 180.0, 0.0);
        g.add(2, -180.0, 0.0);
        g.add(3, 0.0, 0.0);
        g.settle();
        // The two ends of the range are the same place, so both are one point.
        assert_eq!(g.circle(180.0, 0.0, 1.0, b"km"), Some(vec![1, 2]));
        assert_eq!(g.circle(-180.0, 0.0, 1.0, b"km"), Some(vec![1, 2]));
        assert_eq!(g.circle(179.99, 0.0, 10.0, b"km"), Some(vec![1, 2]));
        assert_eq!(g.circle(0.0, 0.0, 1.0, b"km"), Some(vec![3]));
    }

    #[test]
    fn a_point_that_is_not_one_is_not_held() {
        let mut g = Geos::new();
        g.add(1, 0.0, 0.0);
        g.add(2, 200.0, 0.0);
        g.settle();
        assert_eq!(g.len(), 1);
        assert_eq!(g.circle(0.0, 0.0, 20000.0, b"km"), Some(vec![1]));
    }
}
