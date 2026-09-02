//! The geospatial kernels, which are Redis's geohash arithmetic.
//!
//! A geo key in Redis is not a type. It is a sorted set whose scores happen to
//! be 52 bit interleaved geohashes, and every geo command is a sorted set
//! command with some arithmetic in front of it. `ZSCORE` on a geo key answers
//! the raw hash, `ZRANGE` works, `TYPE` says `zset`, and `OBJECT ENCODING` says
//! `listpack` or `skiplist`. That is not an accident of the implementation, it
//! is the documented behaviour, so it is what we do too.
//!
//! # The score
//!
//! Longitude is mapped from [-180, 180] and latitude from [-85.05112878,
//! 85.05112878], which are the EPSG:900913 limits and the reason you cannot
//! store a point at either pole. Each is scaled to 26 bits and the two are
//! interleaved, latitude in the even positions and longitude in the odd ones,
//! giving 52 bits, which is exactly what an `f64` holds without losing anything.
//! That last part is why the score can be a float at all.
//!
//! A shorter hash is the same thing with fewer bits, and it names a box rather
//! than a point. [`struct@Hash`] carries the step so the two cannot be confused, and
//! [`align`] is what turns a box into the score range that covers it.
//!
//! # The search
//!
//! A radius search is nine range queries. Work out how many bits of hash make a
//! box about the size of the search area, find the box the centre is in, take
//! its eight neighbours, and ask the sorted set for every member whose score
//! falls in one of those nine ranges. Then throw away the ones that are in a box
//! but outside the actual circle. The boxes are a filter and the distance is the
//! answer.
//!
//! Two adjustments in [`areas`] are what make that correct rather than nearly
//! correct, and both are Redis's. The step estimate can be one too coarse near
//! the edge of a box, so the four side neighbours are decoded and the step is
//! dropped by one if any of them fails to reach past the bounding box. And a
//! neighbour that is entirely outside the bounding box is zeroed rather than
//! searched, which is three of the nine gone in the common case.
//!
//! # The distance
//!
//! Haversine on a sphere of radius 6372797.560856 metres, which is the WGS-84
//! quadratic mean radius. Not Vincenty, not the ellipsoid, and not accurate to
//! better than about half a percent at continental distances. It is the number
//! Redis answers and a client comparing our `GEODIST` against its own is
//! comparing against this, so a better formula would read as a bug.
//!
//! The one shortcut in it is Redis's too: when the two longitudes are exactly
//! equal the haversine collapses to `asin(sin(x))`, which is `x` over the
//! latitude range, so the arc is computed directly and the trigonometry is
//! skipped.

use core::f64::consts::PI;

/// How many bits of hash a full precision point uses on each axis.
pub const STEP_MAX: u8 = 26;
/// The lowest longitude that can be stored.
pub const LON_MIN: f64 = -180.0;
/// The highest longitude that can be stored.
pub const LON_MAX: f64 = 180.0;
/// The lowest latitude that can be stored.
///
/// Not -90. The Mercator projection the hash is built on does not reach the
/// poles, and this is where it is cut off.
pub const LAT_MIN: f64 = -85.051_128_78;
/// The highest latitude that can be stored.
pub const LAT_MAX: f64 = 85.051_128_78;

/// The radius of the earth the distances are computed on, in metres.
///
/// The WGS-84 quadratic mean radius. Redis's constant, to every digit it writes.
const EARTH_RADIUS: f64 = 6_372_797.560_856;
/// Half the circumference of the Mercator projection, in metres.
const MERCATOR_MAX: f64 = 20_037_726.37;
/// Radians in a degree.
const DEG_TO_RAD: f64 = PI / 180.0;

/// The alphabet a `GEOHASH` string is written in.
///
/// Base 32 with `a`, `i`, `l` and `o` left out, which is the standard geohash
/// alphabet and not one of the base 32 alphabets anything else uses.
const ALPHABET: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// How many characters a `GEOHASH` reply has.
pub const HASH_CHARS: usize = 11;

/// What a distance is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Metres, and the default when a command leaves the unit off.
    M,
    /// Kilometres.
    Km,
    /// Feet.
    Ft,
    /// Miles.
    Mi,
}

impl Unit {
    /// How many metres one of these is.
    ///
    /// A mile is 1609.34 rather than 1609.344, which is wrong by about two
    /// millimetres and is the number Redis divides by, so it is the number a
    /// client's own conversion has been checked against.
    #[must_use]
    pub const fn metres(self) -> f64 {
        match self {
            Unit::M => 1.0,
            Unit::Km => 1000.0,
            Unit::Ft => 0.3048,
            Unit::Mi => 1609.34,
        }
    }

    /// Read a unit the way a command spells it, in any case.
    #[must_use]
    pub fn parse(word: &[u8]) -> Option<Unit> {
        let mut buf = [0u8; 2];
        if word.len() > 2 {
            return None;
        }
        for (i, b) in word.iter().enumerate() {
            buf[i] = b.to_ascii_lowercase();
        }
        match &buf[..word.len()] {
            b"m" => Some(Unit::M),
            b"km" => Some(Unit::Km),
            b"ft" => Some(Unit::Ft),
            b"mi" => Some(Unit::Mi),
            _ => None,
        }
    }
}

/// A box of the world, named by however many bits of hash it took to name it.
///
/// At [`STEP_MAX`] this is a point as far as anything can tell. Below that it is
/// an area, and [`align`] turns it into the range of scores inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hash {
    /// The interleaved bits, `step * 2` of them, low aligned.
    pub bits: u64,
    /// How many bits of each axis are in there.
    pub step: u8,
}

/// The corners of a box, in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Area {
    /// The lowest and highest longitude the box covers.
    pub lon: (f64, f64),
    /// The lowest and highest latitude the box covers.
    pub lat: (f64, f64),
}

impl Area {
    /// The middle of the box, which is what a stored point decodes to.
    #[must_use]
    pub fn centre(&self) -> (f64, f64) {
        let lon = ((self.lon.0 + self.lon.1) / 2.0).clamp(LON_MIN, LON_MAX);
        let lat = ((self.lat.0 + self.lat.1) / 2.0).clamp(LAT_MIN, LAT_MAX);
        (lon, lat)
    }
}

/// What is being searched for, and around where.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// The centre longitude.
    pub lon: f64,
    /// The centre latitude.
    pub lat: f64,
    /// A circle or a rectangle, with its size in `unit`.
    pub kind: Kind,
    /// What `kind`'s numbers are measured in.
    pub unit: Unit,
}

/// A circle or a rectangle.
#[derive(Debug, Clone, Copy)]
pub enum Kind {
    /// `BYRADIUS`, and every `GEORADIUS` form.
    Circle {
        /// The radius, in the shape's unit.
        radius: f64,
    },
    /// `BYBOX`, which is an axis aligned rectangle and not a box on the hash
    /// grid. The two have nothing to do with each other.
    Rect {
        /// The full width, in the shape's unit.
        width: f64,
        /// The full height, in the shape's unit.
        height: f64,
    },
}

impl Shape {
    /// The radius, or the half diagonal of the rectangle, in metres.
    ///
    /// The half diagonal and not the half width, because the step estimate has
    /// to cover the corners of the rectangle and not just its sides.
    #[must_use]
    pub fn reach(&self) -> f64 {
        let m = self.unit.metres();
        match self.kind {
            Kind::Circle { radius } => radius * m,
            Kind::Rect { width, height } => {
                let (w, h) = (width / 2.0 * m, height / 2.0 * m);
                (w * w + h * h).sqrt()
            }
        }
    }

    /// The half width and half height of the bounding box, in metres.
    #[must_use]
    fn half(&self) -> (f64, f64) {
        let m = self.unit.metres();
        match self.kind {
            Kind::Circle { radius } => (radius * m, radius * m),
            Kind::Rect { width, height } => (width / 2.0 * m, height / 2.0 * m),
        }
    }

    /// The smallest longitude and latitude rectangle that contains the shape.
    ///
    /// The left and right edges of a shape on the sphere are curved, so the
    /// widest part of it is the edge nearer the equator. That is why the two
    /// hemispheres take their width from different edges rather than both from
    /// the centre.
    #[must_use]
    pub fn bounds(&self) -> Area {
        let (width, height) = self.half();
        let lat_delta = (height / EARTH_RADIUS) / DEG_TO_RAD;
        let top = (width / EARTH_RADIUS / ((self.lat + lat_delta) * DEG_TO_RAD).cos()) / DEG_TO_RAD;
        let bottom =
            (width / EARTH_RADIUS / ((self.lat - lat_delta) * DEG_TO_RAD).cos()) / DEG_TO_RAD;
        let lon_delta = if self.lat < 0.0 { bottom } else { top };
        Area {
            lon: (self.lon - lon_delta, self.lon + lon_delta),
            lat: (self.lat - lat_delta, self.lat + lat_delta),
        }
    }

    /// Whether a point is inside the shape, and how far away it is in metres.
    ///
    /// Nothing at all if it is outside, so the caller never has to compare a
    /// distance against a radius itself and get the boundary case wrong. The
    /// boundary is inclusive on both shapes.
    #[must_use]
    pub fn covers(&self, lon: f64, lat: f64) -> Option<f64> {
        match self.kind {
            Kind::Circle { radius } => {
                let d = distance(self.lon, self.lat, lon, lat);
                (d <= radius * self.unit.metres()).then_some(d)
            }
            Kind::Rect { width, height } => {
                let m = self.unit.metres();
                // Latitude first, because it is the cheap one: a difference of
                // latitude is an arc and needs no trigonometry at all.
                if lat_distance(lat, self.lat) > height * m / 2.0 {
                    return None;
                }
                if distance(lon, lat, self.lon, lat) > width * m / 2.0 {
                    return None;
                }
                Some(distance(self.lon, self.lat, lon, lat))
            }
        }
    }
}

/// The nine boxes a search has to look in, and where the search area is.
#[derive(Debug, Clone, Copy)]
pub struct Search {
    /// The nine boxes, centre first. A box with no bits in it was ruled out and
    /// is not to be searched.
    pub boxes: [Hash; 9],
    /// The rectangle the boxes are covering, which is what a candidate is
    /// finally tested against.
    pub bounds: Area,
}

/// Spread the low 32 bits of each argument into alternating positions.
///
/// `x` lands in the even bits and `y` in the odd ones. Redis calls with latitude
/// first, so latitude is the even half of every score in the wild.
#[must_use]
fn interleave(x: u32, y: u32) -> u64 {
    const B: [u64; 5] = [
        0x5555_5555_5555_5555,
        0x3333_3333_3333_3333,
        0x0f0f_0f0f_0f0f_0f0f,
        0x00ff_00ff_00ff_00ff,
        0x0000_ffff_0000_ffff,
    ];
    let mut x = u64::from(x);
    let mut y = u64::from(y);
    for (shift, mask) in [(16, B[4]), (8, B[3]), (4, B[2]), (2, B[1]), (1, B[0])] {
        x = (x | (x << shift)) & mask;
        y = (y | (y << shift)) & mask;
    }
    x | (y << 1)
}

/// Pull the two halves of an interleaved value back apart.
#[must_use]
fn deinterleave(bits: u64) -> (u32, u32) {
    const B: [u64; 6] = [
        0x5555_5555_5555_5555,
        0x3333_3333_3333_3333,
        0x0f0f_0f0f_0f0f_0f0f,
        0x00ff_00ff_00ff_00ff,
        0x0000_ffff_0000_ffff,
        0x0000_0000_ffff_ffff,
    ];
    let mut x = bits;
    let mut y = bits >> 1;
    for (shift, mask) in [
        (0, B[0]),
        (1, B[1]),
        (2, B[2]),
        (4, B[3]),
        (8, B[4]),
        (16, B[5]),
    ] {
        x = (x | (x >> shift)) & mask;
        y = (y | (y >> shift)) & mask;
    }
    (x as u32, y as u32)
}

/// Whether a point is somewhere the hash can name.
#[must_use]
pub fn in_range(lon: f64, lat: f64) -> bool {
    (LON_MIN..=LON_MAX).contains(&lon) && (LAT_MIN..=LAT_MAX).contains(&lat)
}

/// The hash of a point at a given precision.
///
/// Nothing at all for a point outside the projection, which is the only way this
/// fails. `step` is between 1 and [`STEP_MAX`].
#[must_use]
pub fn encode(lon: f64, lat: f64, step: u8) -> Option<Hash> {
    encode_in(lon, lat, step, (LON_MIN, LON_MAX), (LAT_MIN, LAT_MAX))
}

/// The same, over ranges the caller picks.
///
/// This exists for one caller. A `GEOHASH` reply is a standard geohash string,
/// and the standard runs latitude from -90 to 90 where we store it from
/// -85.05112878, so the reply is the stored point decoded and then encoded again
/// over the wider range. Everything else uses [`encode`].
#[must_use]
pub fn encode_in(
    lon: f64,
    lat: f64,
    step: u8,
    lon_range: (f64, f64),
    lat_range: (f64, f64),
) -> Option<Hash> {
    if step == 0 || step > 32 || !in_range(lon, lat) {
        return None;
    }
    let lat_offset = (lat - lat_range.0) / (lat_range.1 - lat_range.0);
    let lon_offset = (lon - lon_range.0) / (lon_range.1 - lon_range.0);
    let scale = (1u64 << step) as f64;
    let bits = interleave((lat_offset * scale) as u32, (lon_offset * scale) as u32);
    Some(Hash { bits, step })
}

/// The box a hash names.
#[must_use]
pub fn area(hash: Hash) -> Area {
    let (ilat, ilon) = deinterleave(hash.bits);
    let scale = (1u64 << hash.step) as f64;
    let lon_scale = LON_MAX - LON_MIN;
    let lat_scale = LAT_MAX - LAT_MIN;
    Area {
        lon: (
            LON_MIN + (f64::from(ilon) / scale) * lon_scale,
            LON_MIN + (f64::from(ilon + 1) / scale) * lon_scale,
        ),
        lat: (
            LAT_MIN + (f64::from(ilat) / scale) * lat_scale,
            LAT_MIN + (f64::from(ilat + 1) / scale) * lat_scale,
        ),
    }
}

/// A box's bits pushed up to where a full precision score keeps them.
///
/// A search asks the sorted set for scores between the aligned box and the
/// aligned box after it, which is every point inside it.
#[must_use]
pub const fn align(hash: Hash) -> u64 {
    hash.bits << (52 - hash.step * 2)
}

/// The score range a box covers, low inclusive and high exclusive.
#[must_use]
pub const fn range(hash: Hash) -> (u64, u64) {
    let low = align(hash);
    let high = align(Hash {
        bits: hash.bits + 1,
        step: hash.step,
    });
    (low, high)
}

/// The score a point is stored under.
#[must_use]
pub fn score(lon: f64, lat: f64) -> Option<u64> {
    encode(lon, lat, STEP_MAX).map(align)
}

/// The point a score decodes to, which is the middle of the box it names.
///
/// A stored score is always 52 bits, so this never fails on anything that came
/// out of [`score`]. It takes an `f64` because that is what the sorted set holds
/// and a score that is not a whole number in range was not written by us.
#[must_use]
pub fn decode(raw: f64) -> Option<(f64, f64)> {
    if !raw.is_finite() || raw < 0.0 || raw >= (1u64 << 52) as f64 {
        return None;
    }
    let bits = raw as u64;
    if bits == 0 {
        // Redis treats an all zero hash as undecodable, because zero is also
        // what its "no such box" marker looks like. The point it would decode
        // to is the far south west corner, which nothing real is at.
        return None;
    }
    Some(
        area(Hash {
            bits,
            step: STEP_MAX,
        })
        .centre(),
    )
}

/// The arc between two latitudes, in metres.
///
/// The haversine with no longitude difference is `asin(sin(x))`, and latitude
/// stays inside the range where that is just `x`, so this is the whole formula
/// rather than a special case of it.
#[must_use]
pub fn lat_distance(lat1: f64, lat2: f64) -> f64 {
    EARTH_RADIUS * ((lat2 - lat1) * DEG_TO_RAD).abs()
}

/// The great circle distance between two points, in metres.
#[must_use]
pub fn distance(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let v = ((lon2 * DEG_TO_RAD - lon1 * DEG_TO_RAD) / 2.0).sin();
    if v == 0.0 {
        return lat_distance(lat1, lat2);
    }
    let (lat1r, lat2r) = (lat1 * DEG_TO_RAD, lat2 * DEG_TO_RAD);
    let u = ((lat2r - lat1r) / 2.0).sin();
    let a = u * u + lat1r.cos() * lat2r.cos() * v * v;
    2.0 * EARTH_RADIUS * a.sqrt().asin()
}

/// How many bits of hash make a box roughly the size of a search.
///
/// Doubling the range until it covers the world counts the halvings, and then
/// two are given back so the box is comfortably larger than the search rather
/// than the same size as it. Near the poles a degree of longitude is short, so
/// one or two more bits come off there.
#[must_use]
pub fn steps_for(mut metres: f64, lat: f64) -> u8 {
    if metres == 0.0 {
        return STEP_MAX;
    }
    let mut step = 1i32;
    while metres < MERCATOR_MAX {
        metres *= 2.0;
        step += 1;
    }
    step -= 2;
    if !(-66.0..=66.0).contains(&lat) {
        step -= 1;
        if !(-80.0..=80.0).contains(&lat) {
            step -= 1;
        }
    }
    step.clamp(1, i32::from(STEP_MAX)) as u8
}

/// Move a box one step east or west.
///
/// Longitude is the odd bits, so adding one to it means adding one at bit one
/// with the even bits masked out of the way. Going the other way is the same
/// trick with a borrow. It wraps at the edge of the world, which is what makes a
/// search across the date line work without a special case.
#[must_use]
fn move_lon(hash: Hash, dir: i8) -> Hash {
    if dir == 0 {
        return hash;
    }
    let width = 64 - u32::from(hash.step) * 2;
    let mut x = hash.bits & 0xaaaa_aaaa_aaaa_aaaa;
    let y = hash.bits & 0x5555_5555_5555_5555;
    let zz = 0x5555_5555_5555_5555u64 >> width;
    if dir > 0 {
        x = x.wrapping_add(zz + 1);
    } else {
        x |= zz;
        x = x.wrapping_sub(zz + 1);
    }
    x &= 0xaaaa_aaaa_aaaa_aaaau64 >> width;
    Hash {
        bits: x | y,
        step: hash.step,
    }
}

/// Move a box one step north or south, which is the even bits.
#[must_use]
fn move_lat(hash: Hash, dir: i8) -> Hash {
    if dir == 0 {
        return hash;
    }
    let width = 64 - u32::from(hash.step) * 2;
    let x = hash.bits & 0xaaaa_aaaa_aaaa_aaaa;
    let mut y = hash.bits & 0x5555_5555_5555_5555;
    let zz = 0xaaaa_aaaa_aaaa_aaaau64 >> width;
    if dir > 0 {
        y = y.wrapping_add(zz + 1);
    } else {
        y |= zz;
        y = y.wrapping_sub(zz + 1);
    }
    y &= 0x5555_5555_5555_5555u64 >> width;
    Hash {
        bits: x | y,
        step: hash.step,
    }
}

/// The eight boxes around one, in the order a search walks them.
///
/// North, south, east, west, then the four corners, which is the order Redis
/// uses and therefore the order an unsorted `GEOSEARCH` hands its results back
/// in. Nobody should depend on that order and clients do, so it is kept.
#[must_use]
fn neighbours(hash: Hash) -> [Hash; 8] {
    [
        move_lat(hash, 1),
        move_lat(hash, -1),
        move_lon(hash, 1),
        move_lon(hash, -1),
        move_lon(move_lat(hash, 1), 1),
        move_lon(move_lat(hash, 1), -1),
        move_lon(move_lat(hash, -1), 1),
        move_lon(move_lat(hash, -1), -1),
    ]
}

/// The nine boxes a search over this shape has to look in.
///
/// Two corrections happen here and both matter. The estimated step can leave the
/// side neighbours too small to reach past the search area when the centre sits
/// near the edge of its own box, so the four sides are checked and the step
/// drops by one if any of them falls short. And once the boxes are settled, a
/// neighbour that the bounding box does not reach into at all is zeroed, which
/// is usually three of the eight and is three range queries not run.
#[must_use]
pub fn areas(shape: &Shape) -> Search {
    let bounds = shape.bounds();
    let mut step = steps_for(shape.reach(), shape.lat);
    let Some(mut hash) = encode(shape.lon, shape.lat, step) else {
        return Search {
            boxes: [Hash { bits: 0, step: 0 }; 9],
            bounds,
        };
    };
    let mut near = neighbours(hash);

    let short = area(near[0]).lat.1 < bounds.lat.1
        || area(near[1]).lat.0 > bounds.lat.0
        || area(near[2]).lon.1 < bounds.lon.1
        || area(near[3]).lon.0 > bounds.lon.0;
    if step > 1 && short {
        step -= 1;
        hash = encode(shape.lon, shape.lat, step).unwrap_or(hash);
        near = neighbours(hash);
    }

    // The order is fixed by `neighbours`: north, south, east, west, north east,
    // north west, south east, south west.
    if step >= 2 {
        let own = area(hash);
        let zero = Hash { bits: 0, step: 0 };
        if own.lat.0 < bounds.lat.0 {
            near[1] = zero;
            near[7] = zero;
            near[6] = zero;
        }
        if own.lat.1 > bounds.lat.1 {
            near[0] = zero;
            near[4] = zero;
            near[5] = zero;
        }
        if own.lon.0 < bounds.lon.0 {
            near[3] = zero;
            near[7] = zero;
            near[5] = zero;
        }
        if own.lon.1 > bounds.lon.1 {
            near[2] = zero;
            near[6] = zero;
            near[4] = zero;
        }
    }

    Search {
        boxes: [
            hash, near[0], near[1], near[2], near[3], near[4], near[5], near[6], near[7],
        ],
        bounds,
    }
}

/// The eleven character geohash string for a point.
///
/// The string is the standard one, so latitude runs from -90 to 90 here rather
/// than from the Mercator limit the score uses. Eleven characters is 55 bits and
/// there are only 52, so the last character is always `0`. Redis has written it
/// that way since the command existed and a client that parses the string back
/// has to see the same thing.
#[must_use]
pub fn geohash(lon: f64, lat: f64) -> Option<[u8; HASH_CHARS]> {
    let hash = encode_in(lon, lat, STEP_MAX, (LON_MIN, LON_MAX), (-90.0, 90.0))?;
    let mut out = [b'0'; HASH_CHARS];
    for (i, slot) in out.iter_mut().enumerate().take(HASH_CHARS - 1) {
        let idx = (hash.bits >> (52 - (i + 1) * 5)) & 0x1f;
        *slot = ALPHABET[idx as usize];
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two scores in every piece of Redis documentation, and the ones a real
    /// 8.10.1 answers for `ZSCORE` after the `GEOADD` from its own manual page.
    const PALERMO: (f64, f64, u64) = (13.361389, 38.115556, 3_479_099_956_230_698);
    const CATANIA: (f64, f64, u64) = (15.087269, 37.502669, 3_479_447_370_796_909);

    #[test]
    fn a_point_scores_what_a_real_server_scores_it() {
        assert_eq!(score(PALERMO.0, PALERMO.1), Some(PALERMO.2));
        assert_eq!(score(CATANIA.0, CATANIA.1), Some(CATANIA.2));
    }

    #[test]
    fn a_score_decodes_back_to_where_the_point_nearly_was() {
        // Not exactly where it was. Twenty six bits of latitude is about two
        // metres, so a stored point comes back as the middle of the box it
        // landed in, and these are the digits a real server prints.
        let (lon, lat) = decode(PALERMO.2 as f64).expect("a real score decodes");
        assert_eq!(format!("{lon}"), "13.361389338970184");
        assert_eq!(format!("{lat}"), "38.1155563954963");
        let (lon, lat) = decode(CATANIA.2 as f64).expect("a real score decodes");
        assert_eq!(format!("{lon}"), "15.087267458438873");
        assert_eq!(format!("{lat}"), "37.50266842333162");
    }

    #[test]
    fn a_point_outside_the_projection_has_no_score() {
        assert_eq!(score(181.0, 38.0), None);
        assert_eq!(score(13.0, 86.0), None);
        assert_eq!(score(-180.1, 0.0), None);
        // The limit itself is inside.
        assert!(score(180.0, LAT_MAX).is_some());
        assert!(score(-180.0, LAT_MIN).is_some());
    }

    #[test]
    fn interleaving_is_its_own_inverse() {
        for (x, y) in [(0u32, 0u32), (1, 0), (0, 1), (0x03ff_ffff, 0x0155_5555)] {
            assert_eq!(deinterleave(interleave(x, y)), (x, y));
        }
    }

    #[test]
    fn the_distances_are_the_ones_a_real_server_answers() {
        let d = distance(PALERMO.0, PALERMO.1, CATANIA.0, CATANIA.1);
        // Redis answers 166274.1516 for these two, from the decoded positions
        // rather than the ones that were sent, which is what this measures.
        let (a, b) = (
            decode(PALERMO.2 as f64).expect("a score"),
            decode(CATANIA.2 as f64).expect("a score"),
        );
        let stored = distance(a.0, a.1, b.0, b.1);
        assert_eq!(format!("{stored:.4}"), "166274.1516");
        assert_eq!(format!("{:.4}", stored / Unit::Km.metres()), "166.2742");
        assert_eq!(format!("{:.4}", stored / Unit::Mi.metres()), "103.3182");
        assert_eq!(format!("{:.4}", stored / Unit::Ft.metres()), "545518.8700");
        // The sent positions are within a couple of metres of the stored ones,
        // which is the whole error budget of a 52 bit hash.
        assert!((d - stored).abs() < 3.0, "{d} against {stored}");
    }

    #[test]
    fn two_points_on_one_meridian_take_the_short_path() {
        // Both points store under the same longitude, so the shortcut is the
        // one that runs and it has to answer what a real server answers for the
        // same pair, which is 111226.3808 metres.
        let a = decode(score(10.0, 40.0).expect("in range") as f64).expect("a score");
        let b = decode(score(10.0, 41.0).expect("in range") as f64).expect("a score");
        assert_eq!(a.0, b.0);
        let d = distance(a.0, a.1, b.0, b.1);
        assert_eq!(format!("{d:.4}"), "111226.3808");
        assert_eq!(lat_distance(a.1, b.1), d);
    }

    #[test]
    fn the_geohash_strings_are_the_ones_a_real_server_writes() {
        let (lon, lat) = decode(PALERMO.2 as f64).expect("a score");
        assert_eq!(&geohash(lon, lat).expect("in range"), b"sqc8b49rny0");
        let (lon, lat) = decode(CATANIA.2 as f64).expect("a score");
        assert_eq!(&geohash(lon, lat).expect("in range"), b"sqdtr74hyu0");
    }

    #[test]
    fn a_unit_is_read_in_any_case_and_nothing_else_is() {
        assert_eq!(Unit::parse(b"m"), Some(Unit::M));
        assert_eq!(Unit::parse(b"KM"), Some(Unit::Km));
        assert_eq!(Unit::parse(b"Ft"), Some(Unit::Ft));
        assert_eq!(Unit::parse(b"mI"), Some(Unit::Mi));
        assert_eq!(Unit::parse(b"yd"), None);
        assert_eq!(Unit::parse(b"meters"), None);
        assert_eq!(Unit::parse(b""), None);
    }

    #[test]
    fn a_box_covers_the_scores_of_everything_inside_it() {
        // Every point in the box has a score in the range, and the range is half
        // open, so the box after it starts exactly where this one ends.
        let hash = encode(13.0, 38.0, 10).expect("in range");
        let (low, high) = range(hash);
        let inside = score(13.0, 38.0).expect("in range");
        assert!(low <= inside && inside < high);
        let next = range(Hash {
            bits: hash.bits + 1,
            step: hash.step,
        });
        assert_eq!(high, next.0);
    }

    #[test]
    fn the_step_estimate_shrinks_the_box_as_the_radius_grows() {
        // A tiny radius gets the finest boxes and a global one gets the coarsest.
        assert_eq!(steps_for(0.0, 0.0), STEP_MAX);
        assert!(steps_for(1.0, 0.0) > steps_for(1000.0, 0.0));
        assert!(steps_for(1000.0, 0.0) > steps_for(1_000_000.0, 0.0));
        assert_eq!(steps_for(40_000_000.0, 0.0), 1);
        // Nearer the poles the boxes are coarser for the same radius, because a
        // degree of longitude is shorter there.
        assert_eq!(steps_for(1000.0, 70.0), steps_for(1000.0, 0.0) - 1);
        assert_eq!(steps_for(1000.0, 85.0), steps_for(1000.0, 0.0) - 2);
    }

    #[test]
    fn the_neighbours_of_a_box_are_the_eight_boxes_around_it() {
        let hash = encode(13.0, 38.0, 10).expect("in range");
        let own = area(hash);
        let near = neighbours(hash);
        // North is the same longitude one step up in latitude, and the two boxes
        // meet with no gap between them.
        assert_eq!(area(near[0]).lat.0, own.lat.1);
        assert_eq!(area(near[1]).lat.1, own.lat.0);
        assert_eq!(area(near[2]).lon.0, own.lon.1);
        assert_eq!(area(near[3]).lon.1, own.lon.0);
        // The corners agree with the two sides they came from.
        assert_eq!(area(near[4]).lat.0, own.lat.1);
        assert_eq!(area(near[4]).lon.0, own.lon.1);
    }

    #[test]
    fn the_boxes_around_the_date_line_wrap_rather_than_run_out() {
        let hash = encode(179.99, 0.0, 6).expect("in range");
        let east = neighbours(hash)[2];
        // East of the last box is the first one, which is what makes a search
        // across the date line find anything at all.
        assert!(area(east).lon.0 < area(hash).lon.0);
    }

    #[test]
    fn a_search_keeps_the_boxes_the_area_reaches_and_drops_the_rest() {
        let shape = Shape {
            lon: 15.0,
            lat: 37.0,
            kind: Kind::Circle { radius: 200.0 },
            unit: Unit::Km,
        };
        let search = areas(&shape);
        // The centre box is always searched, and at least one neighbour was
        // ruled out, since a circle cannot reach into all eight.
        assert_ne!(search.boxes[0].bits, 0);
        assert!(search.boxes[1..].iter().any(|h| h.bits == 0));
        // Every box that survived overlaps the bounding rectangle.
        for h in &search.boxes {
            if h.bits == 0 && h.step == 0 {
                continue;
            }
            let a = area(*h);
            assert!(a.lon.1 >= search.bounds.lon.0 && a.lon.0 <= search.bounds.lon.1);
            assert!(a.lat.1 >= search.bounds.lat.0 && a.lat.0 <= search.bounds.lat.1);
        }
    }

    #[test]
    fn a_shape_covers_what_is_inside_it_and_nothing_else() {
        let circle = Shape {
            lon: 15.0,
            lat: 37.0,
            kind: Kind::Circle { radius: 100.0 },
            unit: Unit::Km,
        };
        assert!(circle.covers(15.0, 37.0).is_some());
        assert!(circle.covers(15.0, 37.5).is_some());
        assert!(circle.covers(15.0, 39.0).is_none());
        // A rectangle is not the circle that contains it, so the corner of the
        // bounding square is outside the circle and inside the box.
        let rect = Shape {
            lon: 15.0,
            lat: 37.0,
            kind: Kind::Rect {
                width: 200.0,
                height: 200.0,
            },
            unit: Unit::Km,
        };
        let corner = (15.0 + 1.1, 37.0 + 0.85);
        assert!(rect.covers(corner.0, corner.1).is_some());
        assert!(circle.covers(corner.0, corner.1).is_none());
    }

    #[test]
    fn a_score_nothing_wrote_does_not_decode() {
        assert_eq!(decode(-1.0), None);
        assert_eq!(decode(0.0), None);
        assert_eq!(decode(f64::NAN), None);
        assert_eq!(decode(f64::INFINITY), None);
        assert_eq!(decode((1u64 << 52) as f64), None);
        // A score with a fraction in it is truncated rather than refused, the
        // same as Redis's cast does, so `ZADD k 1.5 m` then `GEOPOS k m` answers
        // the point score 1 names rather than an error.
        assert!(decode(1.5).is_some());
    }
}
