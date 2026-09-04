//! The vector set commands, on the wire (`10` section 9).
//!
//! Thirteen of them, and they are Redis's own names and argument order rather
//! than ours, because a client that already speaks to a vector set should not
//! have to learn anything to speak to this one. What is underneath is not
//! Redis's HNSW graph, and where that shows through it says so in
//! `divergences.toml` instead of pretending.
//!
//! # The names are upper case and the commands are in no ACL category
//!
//! Both are the module's own doing and both are copied because a client can see
//! them. The thirteen are registered in upper case, so `COMMAND INFO vadd`
//! answers `VADD` and an arity error quotes `VADD`, which is the only group here
//! that is not lower case. Nothing else in the table minds, because a lookup
//! compares without regard to case.
//!
//! None of them is in any ACL category either. There is no `vectorset` category
//! on a real server, `ACL CAT vectorset` is an unknown category there, and
//! `COMMAND LIST FILTERBY ACLCAT read` does not answer `VSIM`. Inventing one
//! would make a rule written against this server mean something it does not
//! mean against a real one, in the direction that grants access rather than the
//! one that refuses it.
//!
//! # What holds the vectors
//!
//! [`VectorBody`], through [`yo_kv::Foreign`], which is exactly how a graph
//! sits under a key. `yo-kv` is the crate every engine above it is built out
//! of, so it cannot name [`yo_vector::Collection`] without a cycle, and the
//! escape in the record tag is there so it does not have to. The payoff is that
//! `DEL`, `EXISTS`, `TYPE`, `KEYS`, `SCAN`, `RANDOMKEY`, `EXPIRE`, `DBSIZE`,
//! `FLUSHDB` and `MEMORY USAGE` all work on a vector set key without one line
//! here.
//!
//! The collection itself is the same one `Db::vectors` in the embedded API
//! hands out, which is Y23 and is the reason a replace, a zero length vector and
//! a dimension mismatch cannot be answered one way here and another way there.
//! That crate sits above this one, so the link only goes one way and this file
//! never names it.
//!
//! # A vector set is cosine, and the first VADD decides how wide
//!
//! Both are Redis's decisions and both are right. A vector set normalises on
//! the way in and reports a similarity, so the metric is not a parameter, and
//! since there is no create command the dimension is whatever the first vector
//! is and every one after it has to match.
//!
//! A similarity is not a distance. The collection measures one minus the cosine,
//! which is 0 for the same direction and 2 for the opposite one, and the wire
//! wants 1 for the same direction and 0 for the opposite one, so the reply is
//! `1 - distance / 2` and both ends of that are exact.
//!
//! # What a client sent and what is stored
//!
//! The unit vector, because that is what a cosine collection stores. `VEMB` is
//! supposed to hand back roughly what went in, so the length of the original is
//! kept beside the element and multiplied back on the way out. That is what
//! Redis does and for the same reason, and it is why `VEMB` of a vector that
//! went in as `[3, 4]` says `[3, 4]` and not `[0.6, 0.8]`.
//!
//! An element's attribute string lives here too, as the bytes the client sent.
//! `FILTER` reads JSON out of it, and a string that is not JSON is a string
//! whose fields are all missing, which is an element no filter matches. That is
//! the same answer refusing the write would eventually give and it is one a
//! client can get to without having its writes refused.
//!
//! Both are held in a slice indexed by the id the collection gave the element,
//! rather than in a second table keyed by the element name. An id is a small
//! integer that is already being computed, and a vector set with a million
//! elements should not hold a million extra copies of their names to answer
//! `VGETATTR`.
//!
//! # Filtering happens inside the search
//!
//! `VSIM ... FILTER '.year > 1980'` is answered by deciding the filter while the
//! search is still choosing what to rank, which is [`vfilter`](super::vfilter)
//! and [`yo_vector::Filter`]. Filtering the ten nearest afterwards would answer a
//! filter that matches one element in a thousand with nothing at all, almost
//! every time, and the client could not tell that from a real no match.
//!
//! The element's attributes are summarised into the tag the collection stores
//! beside its code, one bit per field and string value, so the scan's first test
//! is a subset test on a word it has already loaded and the expression itself
//! only ever runs on what survives that. The tag is rewritten whenever the
//! attribute is, which is what `VSETATTR` costs beyond the store.
//!
//! # The two things a client asks for that are not here
//!
//! `REDUCE` projects a vector onto fewer dimensions on the way in. It is a
//! refusal (D-31) and not silently ignored, because a client that asked for 100
//! dimensions and got 300 would be told the wrong thing by `VDIM` and would pay
//! three times the memory it budgeted for.
//!
//! `NOQUANT`, `BIN` and `Q8` choose how a vector is stored. This stores the full
//! precision vector and a one bit code beside it either way, which is at least
//! as accurate as the most accurate of the three, so they are recorded for
//! `VINFO` and change nothing (D-32).
//!
//! `FILTER-EF` bounds how much work Redis will do before giving up on a
//! selective filter. The scan here widens on its own until it has enough answers
//! or has spent its budget, so the number raises the effort rather than capping
//! it, and a filter that matches almost nothing comes back with fewer answers
//! than `COUNT` rather than reading the whole set. `TRUTH` is the way to ask for
//! all of them (D-33).

use yo_common::{Code, Error, Result, parse_i64};
use yo_kv::{Foreign, Keyspace};
use yo_shape::Metric;
use yo_vector::hnsw::Requested;
use yo_vector::{Collection, Match, Signature};

use super::args::{self, Args};
use super::table::Spec;
use super::vfilter;
use crate::reply::Out;

/// What a key holding anything else answers.
const NOT_A_VECTOR_SET: &str = "Operation against a key holding the wrong kind of value";
/// What a vector that is not one gets.
const BAD_VECTOR: &str = "invalid vector specification";
/// What a count that is not a positive number gets.
const BAD_COUNT: &str = "COUNT must be a positive integer";
/// What `EF` and `FILTER-EF` get for the same.
const BAD_EF: &str = "EF must be a positive integer";
/// What `M` gets.
const BAD_M: &str = "M must be a positive integer";
/// What a `VRANGE` low end that is not a range gets.
const BAD_START: &str = "invalid start range format";
/// What its high end gets.
const BAD_END: &str = "invalid end range format";
/// What a `VRANGE` written the other way round gets.
const BACKWARDS: &str = "'-' can only be used as first argument, '+' only as second";
/// What a `VRANGE` count that is not a number gets, which is not the sentence
/// `VSIM` uses for the same word because a real server does not use one either.
const BAD_COUNT_VALUE: &str = "invalid COUNT value";
/// How many answers `VSIM` gives when nobody said, which is Redis's.
const COUNT: usize = 10;

/// A vector set under a key: the collection, what the client said about it, and
/// the two things a vector set holds that a collection does not.
#[derive(Debug)]
pub(super) struct VectorBody {
    /// The vectors, the index over them and the element names, which is the
    /// same type the embedded API hands out.
    c: Collection,
    /// `M`, `EF_CONSTRUCTION` and `EF_RUNTIME` as the client sent them, for
    /// `VINFO` to answer with. Two of the three changed the tuning on the way
    /// through and `M` did not, which is `10` section 7.
    asked: Requested,
    /// Which of `NOQUANT`, `BIN` and `Q8` the client asked for, recorded and
    /// not applied.
    quant: &'static str,
    /// The length of the vector each element went in as, and its attribute
    /// string, indexed by the collection's id for that element.
    side: Vec<Side>,
}

/// What a vector set holds for an element besides the vector itself.
#[derive(Debug, Default, Clone)]
struct Side {
    /// What the client's vector was long before it was made a unit vector.
    ///
    /// One rather than zero for a vector that never went through here, so that
    /// multiplying by it is the identity and an element with no recorded length
    /// comes back as the unit vector rather than as the origin.
    norm: f32,
    /// The attribute string, which is bytes here and JSON to a client.
    attr: Option<Box<[u8]>>,
}

impl Foreign for VectorBody {
    fn type_name(&self) -> &'static str {
        "vectorset"
    }

    fn encoding(&self) -> &'static str {
        // What the searchable form is, which is the one thing about the storage
        // a client can act on: it is why the index is a thirty second of the
        // vectors it indexes.
        "rabitq"
    }

    fn memory_bytes(&self) -> usize {
        let attrs: usize = self
            .side
            .iter()
            .map(|s| s.attr.as_ref().map_or(0, |a| a.len()))
            .sum();
        self.c.memory_bytes() + self.side.capacity() * size_of::<Side>() + attrs
    }

    fn is_empty(&self) -> bool {
        self.c.is_empty()
    }
}

impl VectorBody {
    /// An empty vector set of `dim` dimensions.
    fn new(dim: usize, asked: Requested) -> Result<VectorBody> {
        let mut c = Collection::new(dim, Metric::Cosine)?;
        c.retune(asked.tuning());
        Ok(VectorBody {
            c,
            asked,
            quant: "f32",
            side: Vec::new(),
        })
    }

    /// What an element went in as, which is the unit vector scaled back up.
    fn embedding(&self, key: &[u8]) -> Option<Vec<f32>> {
        let unit = self.c.get(key)?;
        let norm = self.norm(key);
        Some(unit.iter().map(|x| x * norm).collect())
    }

    /// The length the element's vector had when it arrived.
    fn norm(&self, key: &[u8]) -> f32 {
        match self.c.id(key).and_then(|id| self.side.get(id as usize)) {
            Some(s) if s.norm > 0.0 => s.norm,
            _ => 1.0,
        }
    }

    /// The element's attribute string, if it has one.
    fn attr(&self, key: &[u8]) -> Option<&[u8]> {
        let id = self.c.id(key)?;
        self.side.get(id as usize)?.attr.as_deref()
    }

    /// Room for the element's id in the side table, whatever it turned out to
    /// be.
    fn side_mut(&mut self, key: &[u8]) -> &mut Side {
        let id = self.c.id(key).expect("the element was just written") as usize;
        if self.side.len() <= id {
            self.side.resize(id + 1, Side::default());
        }
        &mut self.side[id]
    }

    /// Put the element's attributes back into the tag the scan reads.
    ///
    /// The tag is a summary of a string the client can rewrite at any time, so
    /// this runs on every write that can have changed either the attribute or
    /// the element's place in the index. It is one store into the posting, with
    /// no requantising and no maintenance behind it.
    fn retag(&mut self, key: &[u8]) {
        let tag = self.attr(key).map_or(0, vfilter::tag);
        self.c.retag(key, tag);
    }

    /// How many elements carry an attribute, which is what `VINFO` reports.
    fn attributes(&self) -> usize {
        self.side.iter().filter(|s| s.attr.is_some()).count()
    }
}

/// The similarity a client expects, from the distance the collection measured.
///
/// The collection reports one minus the cosine, in 0 for the same direction to 2
/// for the opposite one. A vector set reports the other way round and on a scale
/// of one, so this is `1 - d / 2`, which puts an identical vector at exactly 1
/// and an opposite one at exactly 0.
fn similarity(distance: f32) -> f64 {
    f64::from(1.0 - distance / 2.0).clamp(0.0, 1.0)
}

pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "VADD" => vadd(db, args, out),
        "VSIM" => vsim(db, args, out),
        "VREM" => vrem(db, args, out),
        "VCARD" => vcard(db, args, out),
        "VDIM" => vdim(db, args, out),
        "VEMB" => vemb(db, args, out),
        "VINFO" => vinfo(db, args, out),
        "VISMEMBER" => vismember(db, args, out),
        "VRANDMEMBER" => vrandmember(db, args, out),
        "VLINKS" => vlinks(db, args, out),
        "VSETATTR" => vsetattr(db, args, out),
        "VGETATTR" => vgetattr(db, args, out),
        "VRANGE" => vrange(db, args, out),
        other => unreachable!("{other} is not a vector set command"),
    }
}

/// `VADD key [REDUCE dim] FP32 blob | VALUES n v... element [options]`.
///
/// Answers 1 for an element that was not there and 0 for one whose vector was
/// replaced, which is what lets an ingest count what it created.
fn vadd(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.opt(2).is_some_and(|a| args::is(a, b"reduce")) {
        return Err(Error::new(
            Code::Unsupported,
            "REDUCE is not supported. The vector is stored at the dimension it arrives at, and a projection that quietly changed what VDIM says would be worse than saying so",
        ));
    }
    let (v, next) = vector(args, 2)?;
    let element = args.opt(next).ok_or_else(args::syntax)?;
    let opts = Add::parse(args, next + 1)?;

    // Everything is parsed before the key is touched, so a VADD with a bad
    // option creates nothing rather than creating the set and then failing.
    let body = open(db, args.get(1), v.len(), opts.asked)?;
    let new = body.c.put(element, &v)?;
    body.quant = opts.quant;
    let norm = norm(&v);
    let side = body.side_mut(element);
    side.norm = norm;
    if let Some(attr) = opts.attr {
        side.attr = Some(attr.into());
    }
    // Always, and not only when SETATTR was sent. A VADD over an element that is
    // already there gives it a new vector and therefore a new place in the
    // index, carrying whatever tag the insert put on it, which is none.
    body.retag(element);
    out.int(i64::from(new));
    Ok(())
}

/// `VSIM key ELE e | FP32 blob | VALUES n v... [options]`.
fn vsim(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    // The query is read before the key, because ELE names an element of this
    // set and the answer to `VSIM missing ELE x` is an empty array either way.
    let (query, next) = if args::is(args.get(2), b"ele") {
        (Query::Element(args.opt(3).ok_or_else(args::syntax)?), 4)
    } else {
        let (v, next) = vector(args, 2)?;
        (Query::Vector(v), next)
    };
    let opts = Sim::parse(args, next)?;

    let Some(body) = read(db, args.get(1))? else {
        out.array(0);
        return Ok(());
    };
    let mut hits = match query {
        // A search from an element leaves that element out, because it is
        // always the nearest and nobody asked what a thing is most like itself.
        Query::Element(e) => {
            let Some(q) = body.c.get(e) else {
                out.array(0);
                return Ok(());
            };
            let q = q.to_vec();
            search(body, &q, opts.effort, Some(e), &opts)?
        }
        Query::Vector(v) => search(body, &v, opts.effort, None, &opts)?,
    };
    // Whatever `EF` widened the search to, the client asked for `COUNT`.
    hits.truncate(opts.count);
    answer(body, &hits, &opts, out);
    Ok(())
}

/// `VREM key element`, which answers 1 if the element was there.
fn vrem(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = write(db, args.get(1))? else {
        out.int(0);
        return Ok(());
    };
    let element = args.get(2);
    // The id is read before the removal, because afterwards there is no id to
    // read and the slot it names is free for the next element to be given.
    let id = body.c.id(element);
    let gone = body.c.remove(element);
    if let Some(id) = id.filter(|_| gone)
        && let Some(side) = body.side.get_mut(id as usize)
    {
        *side = Side::default();
    }
    out.int(i64::from(gone));
    // A vector set whose last element has gone takes its key with it, which is
    // what every other collection here does.
    db.reap_foreign(args.get(1));
    Ok(())
}

/// `VCARD key`, which is how many elements it holds.
fn vcard(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let n = read(db, args.get(1))?.map_or(0, |b| b.c.len());
    out.uint(n as u64);
    Ok(())
}

/// `VDIM key`, which is how wide its vectors are.
fn vdim(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    match read(db, args.get(1))? {
        Some(body) => out.uint(body.c.dim() as u64),
        // A dimension the set has not been given yet is not zero, because zero
        // is a number a client could act on.
        None => return Err(Error::new(Code::NotFound, "key does not exist")),
    }
    Ok(())
}

/// `VEMB key element [RAW]`, which is roughly what the client sent.
fn vemb(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let raw = match args.len() {
        3 => false,
        4 if args::is(args.get(3), b"raw") => true,
        _ => return Err(args::syntax()),
    };
    let Some(body) = read(db, args.get(1))? else {
        out.nil_array();
        return Ok(());
    };
    let element = args.get(2);
    let Some(v) = body.embedding(element) else {
        out.nil_array();
        return Ok(());
    };
    if raw {
        // The stored form and the number that turns it back into the client's,
        // which is what RAW is for. Ours is the unit vector as float32 in the
        // machine's own order, and there is no quantisation range to report
        // because the bytes are not quantised.
        let unit = body.c.get(element).expect("the element is there");
        let mut bytes = Vec::with_capacity(unit.len() * 4);
        for x in unit {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        out.array(3);
        out.bulk(b"f32");
        out.bulk(&bytes);
        out.double(f64::from(body.norm(element)));
        return Ok(());
    }
    out.array(v.len());
    for x in v {
        out.double(f64::from(x));
    }
    Ok(())
}

/// `VINFO key`, which is what the index really is.
fn vinfo(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        out.nil();
        return Ok(());
    };
    let t = body.c.tuning();
    let fields: [(&[u8], u64); 8] = [
        (b"vector-dim", body.c.dim() as u64),
        (b"size", body.c.len() as u64),
        (b"attributes-count", body.attributes() as u64),
        (b"hnsw-m", body.asked.m as u64),
        (b"ef-construction", body.asked.ef_construction as u64),
        (b"ef-runtime", body.asked.ef_runtime as u64),
        (b"partitions", body.c.partitions() as u64),
        (b"code-bytes", body.c.code_bytes() as u64),
    ];
    out.map(fields.len() + 4);
    // The index first, because it is the field that says the rest of this reply
    // is not describing a graph.
    out.bulk(b"index-type");
    out.bulk(b"partition");
    out.bulk(b"quant-type");
    out.bulk(body.quant.as_bytes());
    for (name, value) in fields {
        out.bulk(name);
        out.uint(value);
    }
    out.bulk(b"probe");
    out.uint(t.probe as u64);
    out.bulk(b"rerank");
    out.uint(t.rerank as u64);
    Ok(())
}

/// `VISMEMBER key element`.
fn vismember(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let there = read(db, args.get(1))?.is_some_and(|b| b.c.contains(args.get(2)));
    out.int(i64::from(there));
    Ok(())
}

/// `VRANDMEMBER key [count]`, which is `SRANDMEMBER` over the element names.
fn vrandmember(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let count = match args.len() {
        2 => None,
        3 => Some(args.int(2)?),
        _ => return Err(args::syntax()),
    };
    // The draws are taken from the database's own stream before the body is
    // borrowed, because both want the keyspace and only one of them can have
    // it. A negative count draws with repeats and a positive one draws
    // positions to skip, which is the same two shapes SRANDMEMBER has.
    let len = read(db, args.get(1))?.map_or(0, |b| b.c.len());
    let Some(count) = count else {
        if len == 0 {
            out.nil();
            return Ok(());
        }
        let pick = (db.random() % len as u64) as usize;
        let body = read(db, args.get(1))?.expect("the set is still there");
        out.bulk(body.c.key_at(pick).expect("the draw was under the length"));
        return Ok(());
    };
    if len == 0 {
        out.array(0);
        return Ok(());
    }
    let picks = draws(db, count, len);
    let body = read(db, args.get(1))?.expect("the set is still there");
    out.array(picks.len());
    for at in picks {
        out.bulk(body.c.key_at(at).expect("the draw was under the length"));
    }
    Ok(())
}

/// `VLINKS key element [WITHSCORES]`, which has no graph to walk (D-2).
///
/// The question it is asking is which elements this one is stored next to, and
/// there is an honest answer to that here: the members of its own partition, in
/// order of how near they are to it. A client that draws a neighbour graph from
/// this gets a graph of the index that exists rather than of one that does not.
fn vlinks(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let withscores = match args.len() {
        3 => false,
        4 if args::is(args.get(3), b"withscores") => true,
        _ => return Err(args::syntax()),
    };
    let Some(body) = read(db, args.get(1))? else {
        out.nil_array();
        return Ok(());
    };
    let element = args.get(2);
    let Some(q) = body.c.get(element) else {
        out.nil_array();
        return Ok(());
    };
    let q = q.to_vec();
    let near = body.c.search(&q, COUNT + 1, Some(element))?;

    // One layer, because there is one. A client walking layer by layer sees a
    // graph one deep rather than a reply it cannot parse.
    out.array(1);
    write_hits(&near, withscores, out);
    Ok(())
}

/// `VSETATTR key element json`, which answers 1 if the element was there.
///
/// An empty string takes the attribute off, which is Redis's spelling of a
/// removal and the reason this is not two commands.
fn vsetattr(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = write(db, args.get(1))? else {
        out.int(0);
        return Ok(());
    };
    let element = args.get(2);
    if !body.c.contains(element) {
        out.int(0);
        return Ok(());
    }
    let value = args.get(3);
    let side = body.side_mut(element);
    side.attr = if value.is_empty() {
        None
    } else {
        Some(value.into())
    };
    body.retag(element);
    out.int(1);
    Ok(())
}

/// `VGETATTR key element`.
fn vgetattr(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    match read(db, args.get(1))?.and_then(|b| b.attr(args.get(2))) {
        Some(attr) => out.bulk(attr),
        None => out.nil(),
    }
    Ok(())
}

/// `VRANGE key start end [count]`, which is the element names in a range.
///
/// The only command here that has nothing to do with vectors. It reads the
/// names as names, in the order bytes come in, which is what makes it the way to
/// page over a set that is being written to without a cursor and without the
/// repeats and misses a random draw gives.
fn vrange(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let count = match args.len() {
        4 => None,
        5 => Some(args.get(4)),
        _ => return Err(args::wrong_arity("VRANGE")),
    };
    // The count is read before the ends, which is the order a real server reads
    // them in and is worth keeping because it is the order a client sees: a
    // request with both a bad range and a bad count is told about the count.
    //
    // Zero is not the same as leaving it out. A client that asked for no
    // elements is given none, and only a negative number means every one.
    let count = match count {
        None => usize::MAX,
        Some(arg) => match parse_i64(arg) {
            Some(n) if n < 0 => usize::MAX,
            Some(n) => usize::try_from(n).unwrap_or(usize::MAX),
            None => return Err(Error::new(Code::Invalid, BAD_COUNT_VALUE)),
        },
    };
    // Both ends are read before either is placed, because a client that wrote a
    // range backwards has two things wrong with it and the one it is told about
    // is the spelling and not the direction.
    let start = bound(args.get(2), BAD_START)?;
    let end = bound(args.get(3), BAD_END)?;
    if matches!(start, Bound::Above) || matches!(end, Bound::Below) {
        return Err(Error::new(Code::Invalid, BACKWARDS));
    }
    let Some(body) = read(db, args.get(1))? else {
        out.array(0);
        return Ok(());
    };
    // Filtered first and sorted after, so the sort is over what the range
    // covers rather than over the whole set, which is the difference between
    // reading a page off a million element set and sorting a million names to
    // hand back ten.
    let mut names: Vec<&[u8]> = (0..body.c.len())
        .filter_map(|i| body.c.key_at(i))
        .filter(|name| start.holds_start(name) && end.holds_end(name))
        .collect();
    names.sort_unstable();
    names.truncate(count);
    out.array(names.len());
    for name in names {
        out.bulk(name);
    }
    Ok(())
}

/// One end of the range `VRANGE` reads.
enum Bound<'a> {
    /// `-`, which is before every name there could be.
    Below,
    /// `+`, which is after every name there could be.
    Above,
    /// `[name`, which is that name and everything past it.
    In(&'a [u8]),
    /// `(name`, which is everything past that name and not the name.
    Out(&'a [u8]),
}

impl Bound<'_> {
    /// Whether `name` is at or past this end read as the low one.
    fn holds_start(&self, name: &[u8]) -> bool {
        match self {
            Bound::Below => true,
            Bound::Above => false,
            Bound::In(at) => name >= *at,
            Bound::Out(at) => name > *at,
        }
    }

    /// Whether `name` is at or before this end read as the high one.
    fn holds_end(&self, name: &[u8]) -> bool {
        match self {
            Bound::Below => false,
            Bound::Above => true,
            Bound::In(at) => name <= *at,
            Bound::Out(at) => name < *at,
        }
    }
}

/// One end of a `VRANGE` as the client spelled it.
///
/// A bracket on its own is not a name, which is the one place this differs from
/// the lex range `ZRANGEBYLEX` reads, where a bracket on its own is the empty
/// name. An element can be called the empty string, so the two answer
/// differently about a real element, and this is the answer a vector set gives.
fn bound<'a>(arg: &'a [u8], bad: &'static str) -> Result<Bound<'a>> {
    match arg {
        b"-" => Ok(Bound::Below),
        b"+" => Ok(Bound::Above),
        _ if arg.len() < 2 => Err(Error::new(Code::Invalid, bad)),
        _ => match arg[0] {
            b'[' => Ok(Bound::In(&arg[1..])),
            b'(' => Ok(Bound::Out(&arg[1..])),
            _ => Err(Error::new(Code::Invalid, bad)),
        },
    }
}

/// What `VSIM` was asked about.
enum Query<'a> {
    /// An element of this set, whose own vector is the query.
    Element(&'a [u8]),
    /// A vector the client sent.
    Vector(Vec<f32>),
}

/// The options on a `VADD`.
struct Add<'a> {
    asked: Requested,
    quant: &'static str,
    attr: Option<&'a [u8]>,
}

impl<'a> Add<'a> {
    fn parse(args: Args<'a>, from: usize) -> Result<Add<'a>> {
        let mut got = Add {
            asked: Requested::default(),
            quant: "f32",
            attr: None,
        };
        let mut i = from;
        while i < args.len() {
            let arg = args.get(i);
            let rest = args.len() - i;
            if args::is(arg, b"cas") {
                // A vector set does the insert on a background thread and CAS
                // asks for the check to be redone when it lands. One thread
                // here, so the check never went stale and there is nothing to
                // redo.
                i += 1;
            } else if args::is(arg, b"noquant") {
                got.quant = "f32";
                i += 1;
            } else if args::is(arg, b"bin") {
                got.quant = "bin";
                i += 1;
            } else if args::is(arg, b"q8") {
                got.quant = "int8";
                i += 1;
            } else if args::is(arg, b"ef") && rest >= 2 {
                got.asked.ef_construction = positive(args.get(i + 1), BAD_EF)?;
                i += 2;
            } else if args::is(arg, b"m") && rest >= 2 {
                got.asked.m = positive(args.get(i + 1), BAD_M)?;
                i += 2;
            } else if args::is(arg, b"setattr") && rest >= 2 {
                got.attr = Some(args.get(i + 1));
                i += 2;
            } else {
                return Err(args::syntax());
            }
        }
        Ok(got)
    }
}

/// The options on a `VSIM`.
struct Sim {
    /// How many answers the client gets.
    count: usize,
    /// How many the search is asked for, which is the count or more.
    effort: usize,
    withscores: bool,
    withattribs: bool,
    truth: bool,
    /// The expression the elements have to match, if there is one.
    filter: Option<vfilter::Filter>,
}

impl Sim {
    fn parse(args: Args<'_>, from: usize) -> Result<Sim> {
        let mut got = Sim {
            count: COUNT,
            effort: COUNT,
            withscores: false,
            withattribs: false,
            truth: false,
            filter: None,
        };
        let mut ef = None;
        let mut i = from;
        while i < args.len() {
            let arg = args.get(i);
            let rest = args.len() - i;
            if args::is(arg, b"withscores") {
                got.withscores = true;
                i += 1;
            } else if args::is(arg, b"withattribs") {
                got.withattribs = true;
                i += 1;
            } else if args::is(arg, b"truth") {
                got.truth = true;
                i += 1;
            } else if args::is(arg, b"nothread") {
                // The search runs on the calling thread, so this is what
                // happens anyway.
                i += 1;
            } else if args::is(arg, b"count") && rest >= 2 {
                got.count = positive(args.get(i + 1), BAD_COUNT)?;
                i += 2;
            } else if args::is(arg, b"ef") && rest >= 2 {
                ef = Some(positive(args.get(i + 1), BAD_EF)?);
                i += 2;
            } else if args::is(arg, b"filter") && rest >= 2 {
                got.filter = Some(vfilter::Filter::parse(args.get(i + 1))?);
                i += 2;
            } else if args::is(arg, b"filter-ef") && rest >= 2 {
                // Redis reads this as a ceiling on the work a selective filter
                // may cost. The scan here widens until it has enough answers and
                // then stops on its own, so what a client can usefully move is
                // how much is asked for, which is the same knob EF turns (D-33).
                // Zero is Redis's word for no limit and is the default here, so
                // it changes nothing rather than being a syntax error.
                let asked = match parse_i64(args.get(i + 1)) {
                    Some(n) => usize::try_from(n).unwrap_or(0),
                    None => return Err(Error::new(Code::Invalid, BAD_EF)),
                };
                ef = Some(ef.unwrap_or(0).max(asked));
                i += 2;
            } else {
                return Err(args::syntax());
            }
        }
        // `EF` widens how much of the index is read and does not change how
        // many answers come back, so it raises the effort and leaves the count
        // where the client put it. `10` section 7 maps it onto the rerank
        // breadth, which here is how wide the search itself is asked to be, and
        // the extra answers are dropped before they are written. The cap is
        // there because the effort is a capacity and a client that sends four
        // billion should get a wide search rather than an allocation failure.
        got.effort = got.count.max(ef.unwrap_or(0).min(1 << 16));
        Ok(got)
    }
}

/// The `k` nearest, through the index or through every vector there is.
fn search(
    body: &VectorBody,
    q: &[f32],
    k: usize,
    skip: Option<&[u8]>,
    opts: &Sim,
) -> Result<Vec<Match>> {
    let Some(expr) = &opts.filter else {
        return if opts.truth {
            body.c.search_exact(q, k, skip)
        } else {
            body.c.search(q, k, skip)
        };
    };
    let want = Filtered {
        expr,
        want: expr.signature(),
        side: &body.side,
    };
    if opts.truth {
        body.c.search_exact_where(q, k, skip, &want)
    } else {
        body.c.search_where(q, k, skip, &want)
    }
}

/// What a `FILTER` becomes inside the scan.
///
/// Two tests in the order they cost. The tag is a summary of the element's
/// attributes that the scan can read in one instruction, and it can only ever
/// let through an element the expression will reject, never keep out one it
/// would have kept. The expression itself then runs on what is left, which is
/// the elements that are near enough to be ranked and passed the summary.
struct Filtered<'a> {
    expr: &'a vfilter::Filter,
    want: Signature,
    side: &'a [Side],
}

impl yo_vector::Filter for Filtered<'_> {
    fn allows(&self, tag: u64) -> bool {
        Signature::from_bits(tag).covers(self.want)
    }

    fn exact(&self, id: u64) -> bool {
        let attr = self.side.get(id as usize).and_then(|s| s.attr.as_deref());
        self.expr.matches(attr)
    }
}

/// Write what `VSIM` found, in whichever of the four shapes was asked for.
fn answer(body: &VectorBody, hits: &[Match], opts: &Sim, out: &mut Out) {
    // A plain VSIM is a flat array of names, and the option shapes are Redis's
    // rather than this codebase's ZRANGE shape, because a client that already
    // parses one of these should not have to tell the two servers apart.
    if !opts.withscores && !opts.withattribs {
        out.array(hits.len());
        for hit in hits {
            out.bulk(&hit.key);
        }
        return;
    }
    let extras = usize::from(opts.withscores) + usize::from(opts.withattribs);
    if out.proto().is_resp3() {
        out.map(hits.len());
        for hit in hits {
            out.bulk(&hit.key);
            if extras > 1 {
                out.array(extras);
            }
            if opts.withscores {
                out.double(similarity(hit.distance));
            }
            if opts.withattribs {
                attribute(body, &hit.key, out);
            }
        }
        return;
    }
    out.array(hits.len() * (1 + extras));
    for hit in hits {
        out.bulk(&hit.key);
        if opts.withscores {
            out.double(similarity(hit.distance));
        }
        if opts.withattribs {
            attribute(body, &hit.key, out);
        }
    }
}

/// An element's attribute, or a null where it has none.
fn attribute(body: &VectorBody, key: &[u8], out: &mut Out) {
    match body.attr(key) {
        Some(attr) => out.bulk(attr),
        None => out.nil(),
    }
}

/// A flat list of elements, with their similarities beside them if asked.
fn write_hits(hits: &[Match], withscores: bool, out: &mut Out) {
    out.array(hits.len() * (1 + usize::from(withscores)));
    for hit in hits {
        out.bulk(&hit.key);
        if withscores {
            out.double(similarity(hit.distance));
        }
    }
}

/// `FP32 blob` or `VALUES n v1 .. vn` at `at`, and where the arguments go on.
fn vector(args: Args<'_>, at: usize) -> Result<(Vec<f32>, usize)> {
    let spec = args.opt(at).ok_or_else(args::syntax)?;
    if args::is(spec, b"fp32") {
        let blob = args.opt(at + 1).ok_or_else(args::syntax)?;
        if blob.is_empty() || blob.len() % 4 != 0 {
            return Err(Error::new(Code::Invalid, BAD_VECTOR));
        }
        let v = blob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        return Ok((v, at + 2));
    }
    if args::is(spec, b"values") {
        let n = positive(args.opt(at + 1).ok_or_else(args::syntax)?, BAD_VECTOR)?;
        // The count is checked against what actually arrived before anything is
        // reserved, so `VALUES 4000000000 1` allocates nothing.
        if args.len() < at + 2 + n {
            return Err(args::syntax());
        }
        let mut v = Vec::with_capacity(n);
        for j in 0..n {
            let x = args.float(at + 2 + j)?;
            #[allow(clippy::cast_possible_truncation)]
            v.push(x as f32);
        }
        return Ok((v, at + 2 + n));
    }
    Err(Error::new(Code::Invalid, BAD_VECTOR))
}

/// The euclidean length of a vector, which is what `VEMB` multiplies back.
fn norm(v: &[f32]) -> f32 {
    let sum = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>();
    #[allow(clippy::cast_possible_truncation)]
    let norm = sum.sqrt() as f32;
    norm
}

/// The positions `VRANDMEMBER count` draws, in the two shapes it has.
fn draws(db: &mut Keyspace, count: i64, len: usize) -> Vec<usize> {
    let Ok(want) = usize::try_from(count) else {
        // Negative: exactly that many, drawn one at a time, repeats allowed.
        let repeats = usize::try_from(count.unsigned_abs()).unwrap_or(usize::MAX);
        return (0..repeats.min(1 << 20))
            .map(|_| (db.random() % len as u64) as usize)
            .collect();
    };
    // Positive: distinct, at most as many as there are. A shuffle of the
    // positions and then the front of it, because the alternative is drawing
    // and rejecting, which takes longer the closer the count gets to the size.
    let mut all: Vec<usize> = (0..len).collect();
    for i in (1..len).rev() {
        all.swap(i, (db.random() % (i as u64 + 1)) as usize);
    }
    all.truncate(want);
    all
}

/// The vector set under `key`, making one of `dim` dimensions if the key is
/// free.
///
/// An error for a key holding anything else, including a foreign body that is
/// not a vector set, which is the one case the keyspace cannot decide on its own
/// because only this file knows which foreign body it wanted.
fn open<'d>(
    db: &'d mut Keyspace,
    key: &[u8],
    dim: usize,
    asked: Requested,
) -> Result<&'d mut VectorBody> {
    if db.kind_of(key).is_none() {
        db.put_foreign(key, Box::new(VectorBody::new(dim, asked)?));
    }
    match write(db, key)? {
        Some(body) => {
            if body.c.dim() != dim {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!(
                        "Vector dimension mismatch - got {dim} but set has {}",
                        body.c.dim()
                    ),
                ));
            }
            Ok(body)
        }
        // The key was made a line ago and nothing between here and there can
        // have taken it away, so this is the assertion rather than a case.
        None => unreachable!("the vector set was just created"),
    }
}

/// The vector set under `key` for writing, or `None` if the key is not there.
fn write<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d mut VectorBody>> {
    match db.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<VectorBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, NOT_A_VECTOR_SET)),
        },
        None => Ok(None),
    }
}

/// The same, for reading.
fn read<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d VectorBody>> {
    match db.foreign(key)? {
        Some(body) => match body.downcast_ref::<VectorBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, NOT_A_VECTOR_SET)),
        },
        None => Ok(None),
    }
}

/// An argument that has to be a positive number, with its own sentence.
fn positive(arg: &[u8], msg: &'static str) -> Result<usize> {
    match parse_i64(arg) {
        Some(n) if n > 0 => Ok(usize::try_from(n).unwrap_or(usize::MAX)),
        _ => Err(Error::new(Code::Invalid, msg)),
    }
}
