//! The geospatial commands.
//!
//! Ten of them, and every one is a sorted set command underneath, because that
//! is what a geo key is. `GEOADD` is `ZADD` with the coordinates turned into a
//! score, `GEOPOS` and `GEOHASH` and `GEODIST` are `ZSCORE` with arithmetic on
//! the answer, and the six search forms are nine `ZRANGEBYSCORE` calls with a
//! distance filter over the results. Nothing here holds any state a sorted set
//! does not already hold, which is why `TYPE` on a geo key says `zset` and why a
//! client can `ZREM` a place out of one.
//!
//! # The six search commands are one command
//!
//! `GEOSEARCH`, `GEOSEARCHSTORE`, `GEORADIUS`, `GEORADIUS_RO`,
//! `GEORADIUSBYMEMBER` and `GEORADIUSBYMEMBER_RO` differ in how they spell where
//! the centre is and whether they are allowed to write. Once the centre, the
//! shape and the options are parsed there is one search, and it lives in
//! [`Keyspace::geosearch`]. The store forms run the same search and put the
//! results in a key instead of handing them back.
//!
//! # Why the results are held rather than streamed
//!
//! A search cannot answer in order as it goes. The nine boxes are walked in hash
//! order and the reply is in distance order, so every candidate has to be in
//! hand before the first one can be written. The wire needs the count before the
//! members for the same reason every other range command does.
//!
//! So there is a [`Scratch`] on the keyspace holding the hits and one byte
//! buffer with every member's name in it, cleared and refilled per search rather
//! than allocated per search. A search that found a million points holds a
//! million until the next one, which is the same trade [`crate::setops`] makes
//! and for the same reason: the buffer had to exist for the length of the
//! command anyway.
//!
//! # Errors
//!
//! `WRONGTYPE` for a key holding something that is not a sorted set, and a
//! missing key is an empty one everywhere except `GEODIST`, which answers nil
//! for a key that is not there rather than for a member that is not there, and
//! `GEORADIUSBYMEMBER`, which needs a member to take its centre from and says so
//! when it cannot find one.

use yo_common::num::DIGITS_MAX;
use yo_common::{Code, Error, Result};

use crate::elem::Elements;
use crate::geo::{self, Kind, Shape, Unit};
use crate::keyspace::Keyspace;
use crate::strings;
use crate::zset::{Bound, Zset};
use crate::zsets::{ZAdd, member_bytes};

/// What Redis says when a search is asked to start from a member it cannot read
/// a position out of.
const NO_MEMBER: &str = "could not decode requested zset member";

/// The error for a point the projection does not reach.
///
/// The coordinates are printed back with six decimal places, which is `%f` and
/// is what Redis formats them with, so `GEOADD k 181 38 x` complains about
/// `181.000000,38.000000` rather than about `181,38`.
#[must_use]
pub fn out_of_range(lon: f64, lat: f64) -> Error {
    yo_alloc::allow(|| {
        Error::fmt(
            Code::Invalid,
            format_args!("invalid longitude,latitude pair {lon:.6},{lat:.6}"),
        )
    })
}

/// The error for a member a search cannot take its centre from.
#[must_use]
pub fn no_member() -> Error {
    Error::new(Code::Invalid, NO_MEMBER)
}

/// Which way a search orders what it found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sort {
    /// Nearest first. `ASC`.
    Near,
    /// Furthest first. `DESC`.
    Far,
}

/// What a search was asked for beyond its shape.
#[derive(Debug, Clone, Copy, Default)]
pub struct Limit {
    /// `ASC` or `DESC`, or nothing, which leaves the results in the order the
    /// boxes were walked in.
    pub sort: Option<Sort>,
    /// `COUNT`, or nothing for all of them.
    pub count: Option<usize>,
    /// `ANY`, which stops the walk as soon as `count` are in hand rather than
    /// finding them all and keeping the nearest.
    pub any: bool,
}

impl Limit {
    /// The ordering the search actually runs with.
    ///
    /// A `COUNT` with no `ASC` or `DESC` means the nearest ones, so it implies
    /// `ASC`. `ANY` is the exception: it says the caller does not care which
    /// ones, only how many, and sorting would undo the whole point of it.
    #[must_use]
    fn ordering(&self) -> Option<Sort> {
        match self.sort {
            Some(s) => Some(s),
            None if self.count.is_some() && !self.any => Some(Sort::Near),
            None => None,
        }
    }

    /// How many the walk may stop at, or nothing if it has to find them all.
    #[must_use]
    fn cap(&self) -> Option<usize> {
        self.any.then_some(self.count).flatten()
    }
}

/// One member a search found.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    /// Where the member's name starts in the scratch buffer.
    at: usize,
    /// How long the name is.
    len: usize,
    /// The raw score, which is what `WITHHASH` answers.
    pub score: u64,
    /// The stored longitude.
    pub lon: f64,
    /// The stored latitude.
    pub lat: f64,
    /// How far from the centre of the search, in metres.
    pub metres: f64,
}

/// The buffers a search fills, kept rather than built per call.
///
/// Two of them: the hits, and one run of bytes with every member's name in it.
/// A name is a slice of the second, which is why [`Hit`] carries an offset and a
/// length rather than a `Vec<u8>` each. A search over ten thousand points is one
/// buffer that grows once instead of ten thousand small allocations.
#[derive(Debug, Default)]
pub struct Scratch {
    /// What the search found, in the order the boxes were walked unless it was
    /// sorted afterwards.
    hits: Vec<Hit>,
    /// The names, end to end.
    names: Vec<u8>,
}

impl Scratch {
    /// Everything the last search found, longest to say and cheapest to read.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &Hit)> {
        self.hits
            .iter()
            .map(|h| (&self.names[h.at..h.at + h.len], h))
    }

    /// How many hits the last search left here.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hits.len()
    }

    /// Whether the last search found nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Forget everything, keeping the space.
    fn clear(&mut self) {
        self.hits.clear();
        self.names.clear();
    }

    /// Remember one point.
    fn push(&mut self, name: &[u8], score: u64, lon: f64, lat: f64, metres: f64) {
        let at = self.names.len();
        self.names.extend_from_slice(name);
        self.hits.push(Hit {
            at,
            len: name.len(),
            score,
            lon,
            lat,
            metres,
        });
    }

    /// Put the hits in distance order and cut them down to `count`.
    ///
    /// The cut comes first when there is one, through a selection rather than a
    /// full sort, because a search over a city with `COUNT 10` should not pay to
    /// order the other side of the city. Ties between equal distances land in
    /// whatever order the selection leaves them, which is what a real server's
    /// `qsort` does too.
    fn order(&mut self, limit: Limit) {
        let Some(sort) = limit.ordering() else {
            self.hits.truncate(limit.count.unwrap_or(usize::MAX));
            return;
        };
        let near = |a: &Hit, b: &Hit| a.metres.total_cmp(&b.metres);
        let far = |a: &Hit, b: &Hit| b.metres.total_cmp(&a.metres);
        let want = limit.count.unwrap_or(self.hits.len()).min(self.hits.len());
        if want < self.hits.len() {
            match sort {
                Sort::Near => self.hits.select_nth_unstable_by(want, near),
                Sort::Far => self.hits.select_nth_unstable_by(want, far),
            };
            self.hits.truncate(want);
        }
        match sort {
            Sort::Near => self.hits.sort_unstable_by(near),
            Sort::Far => self.hits.sort_unstable_by(far),
        }
    }
}

impl Keyspace {
    /// `GEOADD key [NX|XX] [CH] longitude latitude member [...]`.
    ///
    /// Answers what the `ZADD` underneath answers, which is how many members
    /// were added, or how many were added or moved with `CH`.
    ///
    /// Every coordinate is checked before anything is stored, so a call with one
    /// bad pair in the middle of it leaves the key exactly as it was. Redis does
    /// the same, and it matters more here than it looks: `GEOADD` is how a whole
    /// dataset gets loaded, and a partial load with no way to tell where it
    /// stopped is worse than a refusal.
    pub fn geoadd<'m, I>(&mut self, key: &[u8], points: I, opts: ZAdd) -> Result<usize>
    where
        I: Iterator<Item = (f64, f64, &'m [u8])> + Clone,
    {
        for (lon, lat, member) in points.clone() {
            strings::check_len(key, member.len())?;
            if geo::score(lon, lat).is_none() {
                return Err(out_of_range(lon, lat));
            }
        }
        self.zadd(
            key,
            points.map(|(lon, lat, m)| {
                // Checked in the pass above, and nothing between the two passes
                // can change what a pair of coordinates hashes to.
                (geo::score(lon, lat).expect("checked") as f64, m)
            }),
            opts,
        )
    }

    /// `GEOPOS key member [member ...]`.
    ///
    /// Hands each position over as it is found rather than collecting them,
    /// because the reply is as long as the argument list and the wire already
    /// knows that number. A member that is not there, or whose score is not a
    /// position anything wrote, gets nothing.
    pub fn geopos<'m, F>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Option<(f64, f64)>),
    {
        let Some(at) = self.zset_slot(key)? else {
            members.for_each(|_| f(None));
            return Ok(());
        };
        let z = self.zset_at(at);
        for m in members {
            f(z.score(m).and_then(geo::decode));
        }
        Ok(())
    }

    /// `GEOHASH key member [member ...]`, the same shape one step further on.
    pub fn geohash<'m, F>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Option<&[u8]>),
    {
        let Some(at) = self.zset_slot(key)? else {
            members.for_each(|_| f(None));
            return Ok(());
        };
        let z = self.zset_at(at);
        for m in members {
            let text = z
                .score(m)
                .and_then(geo::decode)
                .and_then(|(lon, lat)| geo::geohash(lon, lat));
            match text {
                Some(bytes) => f(Some(&bytes)),
                None => f(None),
            }
        }
        Ok(())
    }

    /// `GEODIST key member1 member2 [unit]`, in metres whatever the unit was.
    ///
    /// The caller divides, because the unit is a wire concern and the number
    /// this hands back is the one every other distance in the crate is in.
    ///
    /// Nothing at all if either member is missing, and nothing for a key that is
    /// not there, which are the same nil on the wire.
    pub fn geodist(&mut self, key: &[u8], a: &[u8], b: &[u8]) -> Result<Option<f64>> {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(None);
        };
        let z = self.zset_at(at);
        let (Some(sa), Some(sb)) = (z.score(a), z.score(b)) else {
            return Ok(None);
        };
        let (Some(pa), Some(pb)) = (geo::decode(sa), geo::decode(sb)) else {
            return Ok(None);
        };
        Ok(Some(geo::distance(pa.0, pa.1, pb.0, pb.1)))
    }

    /// Where a member is, for a search that takes its centre from one.
    ///
    /// `FROMMEMBER`, and the two `GEORADIUSBYMEMBER` forms. A key that is not
    /// there answers nothing, because those commands have their own reply for
    /// that and it is not the error a missing member gets.
    pub fn geocentre(&mut self, key: &[u8], member: &[u8]) -> Result<Option<(f64, f64)>> {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(None);
        };
        match self.zset_at(at).score(member).and_then(geo::decode) {
            Some(xy) => Ok(Some(xy)),
            None => Err(no_member()),
        }
    }

    /// Run a search and leave what it found on the keyspace.
    ///
    /// Answers how many hits there are, which is what the wire needs before it
    /// can write the array header. The hits themselves come from
    /// [`Keyspace::geohits`], which borrows rather than copies.
    ///
    /// The nine boxes are walked in Redis's order and a box that a previous one
    /// already covered is skipped, which is not an optimisation: at a radius of
    /// a few thousand kilometres the step is small enough that neighbouring
    /// boxes come out identical, and walking one twice would report every member
    /// in it twice.
    pub fn geosearch(&mut self, key: &[u8], shape: &Shape, limit: Limit) -> Result<usize> {
        let mut found = std::mem::take(&mut self.geo);
        found.clear();
        let outcome = match self.zset_slot(key) {
            Err(e) => Err(e),
            Ok(None) => Ok(()),
            Ok(Some(at)) => {
                collect(self.zset_at(at), shape, limit, &mut found);
                Ok(())
            }
        };
        found.order(limit);
        let n = found.hits.len();
        self.geo = found;
        outcome.map(|()| n)
    }

    /// What the last [`Keyspace::geosearch`] found.
    #[must_use]
    pub fn geohits(&self) -> &Scratch {
        &self.geo
    }

    /// `GEOSEARCHSTORE`, and the `STORE` and `STOREDIST` forms of `GEORADIUS`.
    ///
    /// Answers how many members went into the destination. A search that found
    /// nothing deletes the destination rather than leaving an empty sorted set
    /// or leaving the old contents, which is the rule every store form follows.
    ///
    /// `dist` is `STOREDIST`, which stores the distance in the shape's unit as
    /// the score instead of the geohash. The two are not interchangeable: a key
    /// written with `STOREDIST` is a sorted set of distances and is not a geo
    /// key any more, and `GEOPOS` on it answers positions somewhere off the
    /// coast of Africa rather than an error.
    pub fn geosearchstore(
        &mut self,
        dest: &[u8],
        src: &[u8],
        shape: &Shape,
        limit: Limit,
        dist: bool,
    ) -> Result<usize> {
        let n = self.geosearch(src, shape, limit)?;
        let found = std::mem::take(&mut self.geo);
        let mut got = Elements::with_capacity(n.max(16));
        for (name, hit) in found.iter() {
            let score = if dist {
                hit.metres / shape.unit.metres()
            } else {
                hit.score as f64
            };
            let _ = got.insert(name, score);
        }
        self.geo = found;
        let limits = self.zset_limits;
        let built = Zset::from_elements(got, &limits);
        Ok(self.put_zset(dest, built))
    }
}

/// Walk the nine boxes and keep every member the shape covers.
///
/// Split out of [`Keyspace::geosearch`] so that the sorted set borrow and the
/// scratch buffer are two arguments rather than two borrows of the same
/// keyspace, which they cannot both be.
fn collect(z: &Zset, shape: &Shape, limit: Limit, out: &mut Scratch) {
    let search = geo::areas(shape);
    let cap = limit.cap();
    let mut digits = [0u8; DIGITS_MAX];
    // Redis compares each box against the last one it actually walked, and it
    // starts that at index zero, which means the centre box is never the thing a
    // neighbour is compared against. Keeping the quirk keeps the duplicate
    // behaviour identical at radii large enough for the boxes to collide.
    let mut last = 0usize;
    for i in 0..search.boxes.len() {
        let hash = search.boxes[i];
        if hash.bits == 0 && hash.step == 0 {
            continue;
        }
        if last != 0 && hash == search.boxes[last] {
            continue;
        }
        if cap.is_some_and(|n| out.hits.len() >= n) {
            break;
        }
        let (low, high) = geo::range(hash);
        let window = z.window_by_score(Bound::closed(low as f64), Bound::open(high as f64));
        z.walk(window.start, window.len(), false, |m, raw| {
            if cap.is_some_and(|n| out.hits.len() >= n) {
                return;
            }
            let Some((lon, lat)) = geo::decode(raw) else {
                return;
            };
            let Some(metres) = shape.covers(lon, lat) else {
                return;
            };
            out.push(member_bytes(m, &mut digits), raw as u64, lon, lat, metres);
        });
        last = i;
    }
}

/// A circle around a point, which is what five of the six search forms want.
#[must_use]
pub fn circle(lon: f64, lat: f64, radius: f64, unit: Unit) -> Shape {
    Shape {
        lon,
        lat,
        kind: Kind::Circle { radius },
        unit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three places every Redis geo example uses, and one more far enough
    /// away to be outside every search here.
    const PLACES: [(f64, f64, &[u8]); 4] = [
        (13.361389, 38.115556, b"Palermo"),
        (15.087269, 37.502669, b"Catania"),
        (12.758489, 38.788135, b"edge"),
        (2.352222, 48.856613, b"Paris"),
    ];

    fn ks() -> Keyspace {
        let mut db = Keyspace::new();
        let opts = ZAdd::default();
        db.geoadd(b"g", PLACES.iter().copied(), opts)
            .expect("the places are all in range");
        db
    }

    fn names(db: &Keyspace) -> Vec<Vec<u8>> {
        db.geohits().iter().map(|(n, _)| n.to_vec()).collect()
    }

    #[test]
    fn a_geo_key_is_a_sorted_set_of_hashes() {
        let mut db = ks();
        assert_eq!(db.zcard(b"g").expect("a zset"), 4);
        // The scores a real 8.10.1 has after the same `GEOADD`.
        assert_eq!(
            db.zscore(b"g", b"Palermo").expect("a zset"),
            Some(3_479_099_956_230_698.0)
        );
        assert_eq!(
            db.zscore(b"g", b"Catania").expect("a zset"),
            Some(3_479_447_370_796_909.0)
        );
    }

    #[test]
    fn nothing_is_stored_when_one_pair_is_out_of_range() {
        let mut db = Keyspace::new();
        let bad: [(f64, f64, &[u8]); 2] = [(13.0, 38.0, b"good"), (13.0, 86.0, b"bad")];
        let err = db
            .geoadd(b"g", bad.iter().copied(), ZAdd::default())
            .expect_err("86 is past the projection");
        assert_eq!(
            err.message(),
            "invalid longitude,latitude pair 13.000000,86.000000"
        );
        assert_eq!(db.zcard(b"g").expect("a zset"), 0);
    }

    #[test]
    fn a_position_comes_back_where_it_went_in_give_or_take_two_metres() {
        let mut db = ks();
        let mut got = Vec::new();
        db.geopos(b"g", [&b"Palermo"[..], b"nope"].into_iter(), |p| {
            got.push(p)
        })
        .expect("a zset");
        let (lon, lat) = got[0].expect("Palermo is there");
        assert_eq!(format!("{lon}"), "13.361389338970184");
        assert_eq!(format!("{lat}"), "38.1155563954963");
        assert_eq!(got[1], None);
    }

    #[test]
    fn the_distance_between_two_members_is_the_one_a_real_server_answers() {
        let mut db = ks();
        let d = db
            .geodist(b"g", b"Palermo", b"Catania")
            .expect("a zset")
            .expect("both are there");
        assert_eq!(format!("{d:.4}"), "166274.1516");
        assert_eq!(format!("{:.4}", d / Unit::Km.metres()), "166.2742");
        // One member missing is the same nil as the whole key missing.
        assert_eq!(db.geodist(b"g", b"Palermo", b"nope").expect("a zset"), None);
        assert_eq!(db.geodist(b"nope", b"a", b"b").expect("no key"), None);
    }

    #[test]
    fn a_hash_string_is_eleven_characters_and_ends_in_a_zero() {
        let mut db = ks();
        let mut got: Vec<Option<Vec<u8>>> = Vec::new();
        db.geohash(
            b"g",
            [&b"Palermo"[..], b"Catania", b"nope"].into_iter(),
            |h| {
                got.push(h.map(<[u8]>::to_vec));
            },
        )
        .expect("a zset");
        assert_eq!(got[0].as_deref(), Some(&b"sqc8b49rny0"[..]));
        assert_eq!(got[1].as_deref(), Some(&b"sqdtr74hyu0"[..]));
        assert_eq!(got[2], None);
    }

    #[test]
    fn a_radius_search_finds_what_is_inside_it_nearest_first() {
        let mut db = ks();
        let shape = circle(15.0, 37.0, 200.0, Unit::Km);
        let limit = Limit {
            sort: Some(Sort::Near),
            ..Limit::default()
        };
        assert_eq!(db.geosearch(b"g", &shape, limit).expect("a zset"), 2);
        assert_eq!(names(&db), [b"Catania".to_vec(), b"Palermo".to_vec()]);
        // The distances a real server prints for the same search.
        let hits: Vec<f64> = db.geohits().iter().map(|(_, h)| h.metres).collect();
        assert_eq!(format!("{:.4}", hits[0] / 1000.0), "56.4413");
        assert_eq!(format!("{:.4}", hits[1] / 1000.0), "190.4424");
    }

    #[test]
    fn a_box_search_reaches_the_corners_a_circle_does_not() {
        let mut db = ks();
        let shape = Shape {
            lon: 13.361389,
            lat: 38.115556,
            kind: Kind::Rect {
                width: 400.0,
                height: 400.0,
            },
            unit: Unit::Km,
        };
        let limit = Limit {
            sort: Some(Sort::Near),
            ..Limit::default()
        };
        assert_eq!(db.geosearch(b"g", &shape, limit).expect("a zset"), 3);
        assert_eq!(
            names(&db),
            [b"Palermo".to_vec(), b"edge".to_vec(), b"Catania".to_vec()]
        );
    }

    #[test]
    fn a_count_takes_the_nearest_and_desc_takes_the_furthest() {
        let mut db = ks();
        let shape = circle(15.0, 37.0, 200.0, Unit::Km);
        // A `COUNT` on its own means the nearest, with no `ASC` written.
        let limit = Limit {
            count: Some(1),
            ..Limit::default()
        };
        assert_eq!(db.geosearch(b"g", &shape, limit).expect("a zset"), 1);
        assert_eq!(names(&db), [b"Catania".to_vec()]);
        let limit = Limit {
            sort: Some(Sort::Far),
            count: Some(1),
            ..Limit::default()
        };
        assert_eq!(db.geosearch(b"g", &shape, limit).expect("a zset"), 1);
        assert_eq!(names(&db), [b"Palermo".to_vec()]);
    }

    #[test]
    fn any_stops_at_the_count_rather_than_finding_the_nearest() {
        let mut db = ks();
        let shape = circle(15.0, 37.0, 200.0, Unit::Km);
        let limit = Limit {
            count: Some(1),
            any: true,
            ..Limit::default()
        };
        assert_eq!(db.geosearch(b"g", &shape, limit).expect("a zset"), 1);
        // Whichever box came first, which is not necessarily the nearest, and
        // that is the whole point of the option.
        assert_eq!(db.geohits().len(), 1);
    }

    #[test]
    fn a_search_that_finds_nothing_is_not_an_error() {
        let mut db = ks();
        let shape = circle(0.0, 0.0, 1.0, Unit::M);
        assert_eq!(
            db.geosearch(b"g", &shape, Limit::default())
                .expect("a zset"),
            0
        );
        assert!(db.geohits().is_empty());
        // Nor is a key that is not there.
        assert_eq!(
            db.geosearch(b"nope", &shape, Limit::default())
                .expect("no key"),
            0
        );
    }

    #[test]
    fn a_search_around_a_member_is_a_search_around_where_it_is() {
        let mut db = ks();
        let (lon, lat) = db
            .geocentre(b"g", b"Palermo")
            .expect("a zset")
            .expect("Palermo is there");
        let shape = circle(lon, lat, 200.0, Unit::Km);
        let limit = Limit {
            sort: Some(Sort::Near),
            ..Limit::default()
        };
        assert_eq!(db.geosearch(b"g", &shape, limit).expect("a zset"), 3);
        assert_eq!(names(&db)[0], b"Palermo".to_vec());
        // A member that is not there is the error and not a nil, which is what
        // separates it from a key that is not there.
        assert!(db.geocentre(b"g", b"nope").is_err());
        assert_eq!(db.geocentre(b"nope", b"nope").expect("no key"), None);
    }

    #[test]
    fn a_store_keeps_the_hashes_and_a_storedist_keeps_the_distances() {
        let mut db = ks();
        let shape = circle(15.0, 37.0, 200.0, Unit::Km);
        let limit = Limit {
            sort: Some(Sort::Near),
            ..Limit::default()
        };
        assert_eq!(
            db.geosearchstore(b"d", b"g", &shape, limit, false)
                .expect("a zset"),
            2
        );
        assert_eq!(
            db.zscore(b"d", b"Catania").expect("a zset"),
            Some(3_479_447_370_796_909.0)
        );
        // A stored search is still a geo key, so the round trip works.
        assert_eq!(
            db.geodist(b"d", b"Palermo", b"Catania")
                .expect("a zset")
                .map(|d| format!("{d:.4}")),
            Some("166274.1516".to_string())
        );
        assert_eq!(
            db.geosearchstore(b"e", b"g", &shape, limit, true)
                .expect("a zset"),
            2
        );
        let d = db
            .zscore(b"e", b"Catania")
            .expect("a zset")
            .expect("stored");
        assert_eq!(format!("{d:.4}"), "56.4413");
    }

    #[test]
    fn a_store_that_finds_nothing_deletes_what_was_there() {
        let mut db = ks();
        let shape = circle(15.0, 37.0, 200.0, Unit::Km);
        assert_eq!(
            db.geosearchstore(b"d", b"g", &shape, Limit::default(), false)
                .expect("a zset"),
            2
        );
        let empty = circle(0.0, 0.0, 1.0, Unit::M);
        assert_eq!(
            db.geosearchstore(b"d", b"g", &empty, Limit::default(), false)
                .expect("a zset"),
            0
        );
        assert_eq!(db.zcard(b"d").expect("no key"), 0);
    }

    #[test]
    fn a_search_across_the_date_line_finds_both_sides_of_it() {
        let mut db = Keyspace::new();
        let pair: [(f64, f64, &[u8]); 2] = [(179.9, 0.0, b"west"), (-179.9, 0.0, b"east")];
        db.geoadd(b"d", pair.iter().copied(), ZAdd::default())
            .expect("both are in range");
        // The two are 22.2454 kilometres apart going the short way, and a search
        // that treated longitude as a plain number would find one of them.
        let d = db
            .geodist(b"d", b"west", b"east")
            .expect("a zset")
            .expect("both are there");
        assert_eq!(format!("{:.4}", d / Unit::Km.metres()), "22.2454");
        let shape = circle(179.95, 0.0, 50.0, Unit::Km);
        assert_eq!(
            db.geosearch(b"d", &shape, Limit::default())
                .expect("a zset"),
            2
        );
        // Centred exactly on 180 it finds only the western one, because 180 is
        // the last longitude the projection has and there is no box east of it
        // to be a neighbour. A real server answers the same one member, so this
        // is pinned rather than fixed.
        let shape = circle(180.0, 0.0, 50.0, Unit::Km);
        assert_eq!(
            db.geosearch(b"d", &shape, Limit::default())
                .expect("a zset"),
            1
        );
        assert_eq!(names(&db), [b"west".to_vec()]);
    }

    #[test]
    fn a_key_holding_something_else_is_refused_everywhere() {
        let mut db = Keyspace::new();
        db.set(b"s", b"v", strings::SetOptions::default())
            .expect("a fresh key");
        let shape = circle(0.0, 0.0, 1.0, Unit::Km);
        assert_eq!(
            db.geoadd(b"s", PLACES.iter().copied(), ZAdd::default())
                .expect_err("a string")
                .code(),
            Code::WrongType
        );
        assert!(db.geopos(b"s", [&b"x"[..]].into_iter(), |_| {}).is_err());
        assert!(db.geohash(b"s", [&b"x"[..]].into_iter(), |_| {}).is_err());
        assert!(db.geodist(b"s", b"a", b"b").is_err());
        assert!(db.geosearch(b"s", &shape, Limit::default()).is_err());
        assert!(
            db.geosearchstore(b"d", b"s", &shape, Limit::default(), false)
                .is_err()
        );
    }
}
