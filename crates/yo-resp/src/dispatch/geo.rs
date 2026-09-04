//! The geospatial commands, on the wire.
//!
//! Ten names and five of them are the same command. `GEOSEARCH`,
//! `GEOSEARCHSTORE`, `GEORADIUS`, `GEORADIUS_RO`, `GEORADIUSBYMEMBER` and
//! `GEORADIUSBYMEMBER_RO` differ in where the centre is written, whether a
//! shape is named or assumed to be a circle, and whether the result may be
//! stored. Everything after that is one option loop and one search, which is
//! how Redis writes it and is the only way the six stay in step: six parsers
//! would be six chances to accept an option in one spelling and refuse it in
//! another.
//!
//! # The order the arguments are read in is part of the contract
//!
//! A client sees which error it gets, so the order matters as much as the
//! sentences do. The source key is looked up first, before a single argument is
//! read, so `GEORADIUS somestring 1 2 x km` answers `WRONGTYPE` and not `ERR
//! need numeric radius`. A search whose source key does not exist still parses
//! everything, because whether the reply is an empty array or a zero depends on
//! an option that comes later, and a bad option in that command is still an
//! error rather than an empty array. And a `GEORADIUSBYMEMBER` on a key that is
//! not there never looks the member up, so it answers the empty array rather
//! than complaining about a member that was never going to be found.
//!
//! # Distances are written with four places
//!
//! `GEODIST` and `WITHDIST` go through [`Out::distance`], which is a fixed
//! point conversion and not the double formatter every other number in the
//! engine uses. It is a bulk string on RESP3 as well as on RESP2, unlike the
//! coordinates `WITHCOORD` writes, which are doubles on RESP3. Redis draws the
//! line in that odd place and a client reading a search reply has been written
//! against it.
//!
//! # What is not here
//!
//! No sorting, no distance arithmetic and no box algebra. Those are
//! [`yo_kv::geo`] and [`yo_kv::geos`], and this file turns arguments into one
//! call on the keyspace and the answer into bytes.

use yo_common::num::parse_f64;
use yo_common::{Code, Error, Result};
use yo_kv::geo::{self, Kind, Shape, Unit};
use yo_kv::geos::{self, Limit, Scratch, Sort};
use yo_kv::{Db, Gate, Keyspace, ZAdd};

use super::args::{self, Args, is};
use super::table::Spec;
use crate::reply::Out;

/// What Redis says about a unit it does not know, capitals and all.
const BAD_UNIT: &str = "unsupported unit provided. please use M, KM, FT, MI";
/// What it says about a `COUNT` of zero, and about a negative one.
const COUNT_POSITIVE: &str = "COUNT must be > 0";
/// What it says about `ANY` on its own.
const ANY_NEEDS_COUNT: &str = "the ANY argument requires COUNT argument";
/// The radius sentences. Both are the radius's own rather than the shared float
/// message, which is the one place a geo command overrides it.
const NUMERIC_RADIUS: &str = "need numeric radius";
/// What a negative radius gets, which is its own sentence and not a syntax
/// error.
const NEGATIVE_RADIUS: &str = "radius cannot be negative";
/// The `BYBOX` width, which has its own message where the height has another.
const NUMERIC_WIDTH: &str = "need numeric width";
/// And the height.
const NUMERIC_HEIGHT: &str = "need numeric height";
/// One sentence for a negative width and a negative height both, in that order
/// of words whichever of the two it was.
const NEGATIVE_BOX: &str = "height or width cannot be negative";
/// The first half of the sentence about a search with no centre.
const FROM_ONE: &str = "FROMMEMBER or FROMLONLAT";
/// And about one with no shape, which spells its two the other way round.
const BY_ONE: &str = "BYRADIUS and BYBOX";

/// Where a search reads its centre and its shape from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Centre {
    /// Two fixed arguments and then a radius, which is `GEORADIUS`.
    Coords,
    /// A member and then a radius, which is `GEORADIUSBYMEMBER`.
    Member,
    /// `FROMMEMBER` or `FROMLONLAT`, and `BYRADIUS` or `BYBOX`, anywhere in the
    /// options. The two `GEOSEARCH` forms.
    Options,
}

/// Whether a search may write its results somewhere and how it says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Store {
    /// It may not. The two `_RO` forms, and `GEOSEARCH`.
    No,
    /// `STORE key` or `STOREDIST key`, among the options. The two writable
    /// `GEORADIUS` forms.
    Option,
    /// The destination is argument one and `STOREDIST` is a bare flag, which is
    /// `GEOSEARCHSTORE`.
    Argument,
}

/// Which of the six spellings arrived.
#[derive(Debug, Clone, Copy)]
struct Form {
    /// The argument the source key is at.
    src: usize,
    /// The first argument that could be an option.
    base: usize,
    /// Where the centre and the shape are written.
    centre: Centre,
    /// Where a destination may be written, if anywhere.
    store: Store,
}

impl Form {
    /// The form a command name is.
    ///
    /// The base is Redis's `base_args` and the numbers are its numbers: six for
    /// `GEORADIUS`, which spends four arguments on a centre and a radius, five
    /// for the by member forms, which spend three, three for
    /// `GEOSEARCHSTORE`, which spends one on the destination, and two for
    /// `GEOSEARCH`, which spends none.
    fn of(name: &str) -> Form {
        match name {
            "georadius" => Form {
                src: 1,
                base: 6,
                centre: Centre::Coords,
                store: Store::Option,
            },
            "georadius_ro" => Form {
                src: 1,
                base: 6,
                centre: Centre::Coords,
                store: Store::No,
            },
            "georadiusbymember" => Form {
                src: 1,
                base: 5,
                centre: Centre::Member,
                store: Store::Option,
            },
            "georadiusbymember_ro" => Form {
                src: 1,
                base: 5,
                centre: Centre::Member,
                store: Store::No,
            },
            "geosearchstore" => Form {
                src: 2,
                base: 3,
                centre: Centre::Options,
                store: Store::Argument,
            },
            _ => Form {
                src: 1,
                base: 2,
                centre: Centre::Options,
                store: Store::No,
            },
        }
    }
}

/// Run one geospatial command.
pub(super) fn execute(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    // The four that read or write one key are handed the stripe that key is on.
    // The search forms take the database, because a store form names a second
    // key and the two of them can be anywhere.
    let key = args.get(1);
    match spec.name {
        "geoadd" => add(&mut db.hold(key), args, out),
        "geopos" => pos(&mut db.hold(key), args, out),
        "geohash" => hash(&mut db.hold(key), args, out),
        "geodist" => dist(&mut db.hold(key), args, out),
        _ => search(db, spec, args, out),
    }
}

/// `GEOADD key [NX|XX] [CH] longitude latitude member [...]`.
///
/// Every coordinate is parsed before the first one is stored, which is the same
/// rule the keyspace layer applies to the range check and is there for the same
/// reason: a bulk load that stops halfway with no way to tell where is worse
/// than one that refuses.
fn add(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (mut nx, mut xx) = (false, false);
    let mut opts = ZAdd::default();
    let mut at = 2;
    while at < args.len() {
        let arg = args.get(at);
        if is(arg, b"nx") {
            nx = true;
        } else if is(arg, b"xx") {
            xx = true;
        } else if is(arg, b"ch") {
            opts.changed = true;
        } else {
            break;
        }
        at += 1;
    }
    // Redis checks the count and then the two gates, and a call with no triples
    // at all reaches its `ZADD` and comes back with that command's syntax error
    // rather than an arity complaint about this one. The three land on the same
    // sentence here.
    let left = args.len() - at;
    if left == 0 || !left.is_multiple_of(3) || (nx && xx) {
        return Err(args::syntax());
    }
    opts.gate = if nx {
        Gate::IfMissing
    } else if xx {
        Gate::IfPresent
    } else {
        Gate::Always
    };
    for i in (at..args.len()).step_by(3) {
        args.float(i)?;
        args.float(i + 1)?;
    }
    let points = (at..args.len()).step_by(3).map(|i| {
        // Parsed in the pass above, all of them, before anything is stored.
        (
            parse_f64(args.get(i)).expect("checked"),
            parse_f64(args.get(i + 1)).expect("checked"),
            args.get(i + 2),
        )
    });
    out.uint(db.geoadd(args.get(1), points, opts)? as u64);
    Ok(())
}

/// `GEOPOS key member [member ...]`.
///
/// The array header goes out before the lookup, because its length is the
/// argument count and nothing else about it can change. A `WRONGTYPE` from the
/// lookup takes the header back out with it: the dispatcher truncates the reply
/// to where the command started before it writes an error line.
fn pos(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    out.array(args.len() - 2);
    let members = (2..args.len()).map(|i| args.get(i));
    db.geopos(args.get(1), members, |found| match found {
        Some((lon, lat)) => {
            out.array(2);
            out.double(lon);
            out.double(lat);
        }
        // The array null and not the string one, which matters on RESP2 where
        // the two are different bytes.
        None => out.nil_array(),
    })
}

/// `GEOHASH key member [member ...]`, the same shape and the string null.
fn hash(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    out.array(args.len() - 2);
    let members = (2..args.len()).map(|i| args.get(i));
    db.geohash(args.get(1), members, |found| match found {
        Some(text) => out.bulk(text),
        None => out.nil(),
    })
}

/// `GEODIST key member1 member2 [M|KM|FT|MI]`.
///
/// A missing key and a missing member are the same nil, and a unit that will
/// not parse is an error even when the key is not there, because the unit is
/// read before the key is looked up.
fn dist(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() > 5 {
        return Err(args::syntax());
    }
    let unit = match args.opt(4) {
        Some(word) => parse_unit(word)?,
        None => Unit::M,
    };
    match db.geodist(args.get(1), args.get(2), args.get(3))? {
        Some(metres) => out.distance(metres / unit.metres()),
        None => out.nil(),
    }
    Ok(())
}

/// The six search forms, which are one command with six front ends.
fn search(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    let form = Form::of(spec.name);
    let key = args.get(form.src);
    // Before any argument is read, because that is where Redis looks it up and
    // a wrong type has to win over a bad radius. It answers the existence
    // question at the same time, which two later decisions need.
    let here = db.hold(key).zcard(key)? != 0;

    let mut dest = (form.store == Store::Argument).then(|| args.get(1));
    let mut storedist = false;
    let (mut lon, mut lat) = (0.0, 0.0);
    // The member the centre comes from, kept for the second reading of it below.
    // The one during the parse settles which error a bad command answers, and
    // the one under the hold settles where the search actually starts from.
    let mut from = None;
    let mut kind = Kind::Circle { radius: 0.0 };
    let mut unit = Unit::M;

    match form.centre {
        Centre::Coords => {
            (lon, lat) = coords(args, 2)?;
            (kind, unit) = radius(args, 4)?;
        }
        Centre::Member => {
            // Only when the key is there. A key that is not there has its own
            // reply and it is not the error a member that is not there gets.
            from = Some(args.get(2));
            if here {
                (lon, lat) = centre(db, key, args.get(2))?;
            }
            (kind, unit) = radius(args, 3)?;
        }
        Centre::Options => {}
    }

    let (mut withdist, mut withhash, mut withcoord) = (false, false, false);
    let (mut from_member, mut from_lonlat) = (false, false);
    let (mut by_radius, mut by_box) = (false, false);
    let mut sort = None;
    let mut count = None;
    let mut any = false;

    let options = form.centre == Centre::Options;
    let mut at = form.base;
    while at < args.len() {
        let arg = args.get(at);
        // How many arguments follow this one, which is what decides whether an
        // option that takes a value is that option or a stray word. Redis lets
        // `GEOSEARCH k FROMMEMBER` fall through to the syntax error rather than
        // reading past the end of its own argument list.
        let after = args.len() - at - 1;
        if is(arg, b"withdist") {
            withdist = true;
        } else if is(arg, b"withhash") {
            withhash = true;
        } else if is(arg, b"withcoord") {
            withcoord = true;
        } else if is(arg, b"any") {
            any = true;
        } else if is(arg, b"asc") {
            sort = Some(Sort::Near);
        } else if is(arg, b"desc") {
            sort = Some(Sort::Far);
        } else if is(arg, b"count") && after >= 1 {
            let want = args.int(at + 1)?;
            if want <= 0 {
                return Err(plain(COUNT_POSITIVE));
            }
            count = Some(want as usize);
            at += 1;
        } else if (is(arg, b"store") || is(arg, b"storedist"))
            && after >= 1
            && form.store == Store::Option
        {
            dest = Some(args.get(at + 1));
            storedist = is(arg, b"storedist");
            at += 1;
        } else if is(arg, b"storedist") && form.store == Store::Argument {
            storedist = true;
        } else if is(arg, b"frommember") && after >= 1 && options && !from_lonlat {
            from = Some(args.get(at + 1));
            if here {
                (lon, lat) = centre(db, key, args.get(at + 1))?;
            }
            from_member = true;
            at += 1;
        } else if is(arg, b"fromlonlat") && after >= 2 && options && !from_member {
            (lon, lat) = coords(args, at + 1)?;
            from_lonlat = true;
            at += 2;
        } else if is(arg, b"byradius") && after >= 2 && options && !by_box {
            (kind, unit) = radius(args, at + 1)?;
            by_radius = true;
            at += 2;
        } else if is(arg, b"bybox") && after >= 3 && options && !by_radius {
            (kind, unit) = rectangle(args, at + 1)?;
            by_box = true;
            at += 3;
        } else {
            return Err(args::syntax());
        }
        at += 1;
    }

    if dest.is_some() && (withdist || withhash || withcoord) {
        return Err(store_clash(form.store == Store::Argument));
    }
    if options && !(from_member || from_lonlat) {
        return Err(exactly_one(FROM_ONE, args.name()));
    }
    if options && !(by_radius || by_box) {
        return Err(exactly_one(BY_ONE, args.name()));
    }
    if any && count.is_none() {
        return Err(plain(ANY_NEEDS_COUNT));
    }

    let mut shape = Shape {
        lon,
        lat,
        kind,
        unit,
    };
    let limit = Limit { sort, count, any };
    // A source key that is not there finds nothing, and finding nothing is
    // already the empty array in one form and the zero with the destination
    // deleted in the other, so there is no early return for it.
    match dest {
        Some(into) => {
            out.uint(db.geosearchstore(into, key, from, &shape, limit, storedist)? as u64);
        }
        None => {
            let mut stripe = db.hold(key);
            // Where the member is, read again now that the stripe is held, so
            // that the centre and the members it is measured against are the
            // same key at the same moment rather than two of them.
            if let Some(member) = from
                && let Some(centre) = stripe.geocentre(key, member)?
            {
                (shape.lon, shape.lat) = centre;
            }
            stripe.geosearch(key, &shape, limit)?;
            found(stripe.geohits(), unit, [withdist, withhash, withcoord], out);
        }
    }
    Ok(())
}

/// Write out what a search found.
///
/// A member with no options asked for is a bulk string on its own, and one with
/// any of the three is an array of the member and then the distance, the hash
/// and the coordinates in that order. Redis counts the options and nests only
/// when the count is not zero, so a plain search is a flat array of names and
/// nothing has to be unwrapped to read it.
fn found(hits: &Scratch, unit: Unit, with: [bool; 3], out: &mut Out) {
    let [dist, hash, coord] = with;
    let extra = usize::from(dist) + usize::from(hash) + usize::from(coord);
    out.array(hits.len());
    for (member, hit) in hits.iter() {
        if extra != 0 {
            out.array(extra + 1);
        }
        out.bulk(member);
        if dist {
            out.distance(hit.metres / unit.metres());
        }
        if hash {
            out.int(hit.score as i64);
        }
        if coord {
            out.array(2);
            out.double(hit.lon);
            out.double(hit.lat);
        }
    }
}

/// The longitude at `at` and the latitude after it.
fn coords(args: Args<'_>, at: usize) -> Result<(f64, f64)> {
    let lon = args.float(at)?;
    let lat = args.float(at + 1)?;
    if !geo::in_range(lon, lat) {
        return Err(geos::out_of_range(lon, lat));
    }
    Ok((lon, lat))
}

/// The radius at `at` and the unit after it.
fn radius(args: Args<'_>, at: usize) -> Result<(Kind, Unit)> {
    let radius = parse_f64(args.get(at)).ok_or_else(|| plain(NUMERIC_RADIUS))?;
    if radius < 0.0 {
        return Err(plain(NEGATIVE_RADIUS));
    }
    Ok((Kind::Circle { radius }, parse_unit(args.get(at + 1))?))
}

/// The width at `at`, the height after it and the unit after that.
fn rectangle(args: Args<'_>, at: usize) -> Result<(Kind, Unit)> {
    let width = parse_f64(args.get(at)).ok_or_else(|| plain(NUMERIC_WIDTH))?;
    let height = parse_f64(args.get(at + 1)).ok_or_else(|| plain(NUMERIC_HEIGHT))?;
    if width < 0.0 || height < 0.0 {
        return Err(plain(NEGATIVE_BOX));
    }
    Ok((Kind::Rect { width, height }, parse_unit(args.get(at + 2))?))
}

/// Where a member is, for the forms that take their centre from one.
fn centre(db: &Db, key: &[u8], member: &[u8]) -> Result<(f64, f64)> {
    // The key was there a moment ago and nothing has run since, so the `None`
    // is unreachable rather than being another way to say the member is
    // missing. Both answer the same sentence anyway.
    db.hold(key)
        .geocentre(key, member)?
        .ok_or_else(geos::no_member)
}

/// A unit word, or Redis's complaint about the ones it does not know.
fn parse_unit(word: &[u8]) -> Result<Unit> {
    Unit::parse(word).ok_or_else(|| plain(BAD_UNIT))
}

/// One of the fixed sentences, as an error.
fn plain(msg: &'static str) -> Error {
    Error::new(Code::Invalid, msg)
}

/// What a store form says when it was also asked for the distance.
///
/// Two spellings of one sentence, and the difference is which command it was:
/// `GEOSEARCHSTORE` names itself and the `GEORADIUS` pair name the option they
/// were given rather than themselves.
fn store_clash(named: bool) -> Error {
    let who = if named {
        "GEOSEARCHSTORE"
    } else {
        "STORE option in GEORADIUS"
    };
    Error::fmt(
        Code::Invalid,
        format_args!("{who} is not compatible with WITHDIST, WITHHASH and WITHCOORD options"),
    )
}

/// What a `GEOSEARCH` with no centre, or with no shape, says.
///
/// The command is quoted as the client spelled it and not as the table has it,
/// so a lower case `geosearch` comes back lower case, which is one of the few
/// places Redis echoes the spelling rather than its own name.
fn exactly_one(which: &str, name: &[u8]) -> Error {
    yo_alloc::allow(|| {
        Error::fmt(
            Code::Invalid,
            format_args!(
                "exactly one of {which} can be specified for {}",
                String::from_utf8_lossy(name)
            ),
        )
    })
}
