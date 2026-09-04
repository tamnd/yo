//! The graph commands, on the wire.
//!
//! Ten of them and no query language. `11` section 7 says that out loud and it
//! is worth repeating here, because the absence is the design and not a gap:
//! there is no `G.QUERY`, a traversal is a command that says which way it is
//! going, and anything that needs a plan is written as typed methods against
//! the embedded API where the compiler can check it.
//!
//! # What holds the graph
//!
//! [`GraphBody`], through [`yo_kv::Foreign`]. `yo-kv` is the crate every engine
//! above it is built out of, so it cannot name `yo_graph::Graph` without a
//! cycle, and the escape in the record tag exists so it does not have to. The
//! payoff is that `DEL`, `EXISTS`, `TYPE`, `KEYS`, `SCAN`, `RANDOMKEY`,
//! `EXPIRE`, `DBSIZE`, `FLUSHDB` and `MEMORY USAGE` all work on a graph key
//! without one line here, because the keyspace owns the key and always did.
//!
//! # Two things a client says that the plane does not
//!
//! The plane is keyed by a dense `u64` and labels its edges with a `u32`, and a
//! client has neither. It writes `G.EADD social ada grace FOLLOWS`, so this
//! holds two interning tables: node ids to dense ids, and label names to label
//! numbers. Both are one lookup at the entry to a command and never again
//! inside it, which is the same discipline the typed surface follows and the
//! reason a hop stays a probe and a sequential read.
//!
//! A node's client id is kept twice, once as the key of the table and once in
//! the vector that turns a dense id back into it, because `G.OUT` has to answer
//! in the ids the client wrote. The typed surface avoids the second copy by
//! decoding the node back into its struct and recomputing the key, which is
//! only possible because there is a struct. Here there is not.
//!
//! # Two decisions worth arguing with
//!
//! `G.EADD` creates either end that is not there, as a node with no properties.
//! The typed surface refuses the same thing, and the two are not inconsistent:
//! there an `Id<N>` is a handle a caller can only have got from `add`, so an
//! unknown one is a bug, and here the id is the client's own bytes and creating
//! the node is well defined. Bulk edge ingest is most of what a graph does and
//! making it two commands an edge to say something the server can work out is
//! the wrong trade.
//!
//! `G.OUT` and `G.IN` take a cursor that is a position in the run. A run is
//! edited by moving its last entry into the hole a removal made, so a cursor
//! held across an edit can skip an entry or repeat one. That is the same
//! guarantee `SCAN` gives and for the same reason: the alternative is holding a
//! snapshot per client.

use std::collections::HashMap;

use yo_common::{Code, Error, Result, parse_i64};
use yo_doc::{Builder, Doc};
use yo_graph::{Dir, Graph};
use yo_kv::{Foreign, Keyspace};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What `SCAN` and `MEMORY USAGE` and the rest see under a graph key.
const NOT_A_GRAPH: &str = "Operation against a key holding the wrong kind of value";
/// What a depth or a count that is not a number gets.
const BAD_DEPTH: &str = "DEPTH must be a positive integer";
const BAD_COUNT: &str = "COUNT must be a positive integer";
/// The default frontier and page size, which is `SCAN`'s ten for the same
/// reason: a client that did not say wants an answer back rather than a graph.
const COUNT: usize = 10;
/// The most hops `G.NEIGH` and `G.PATH` will take without being asked.
///
/// A frontier grows as the product of the degrees, so an unbounded depth on a
/// social graph is the whole graph by hop four. The bound is the client's to
/// raise and this is what it gets for not saying.
const DEPTH: usize = 2;
/// Where `G.PATH` gives up when nobody said.
const MAXDEPTH: usize = 6;

/// A graph under a key, with the two tables a client needs and the plane does
/// not.
#[derive(Debug, Default)]
pub(super) struct GraphBody {
    g: Graph,
    /// The id the client wrote, to the dense id the plane is keyed by.
    ids: HashMap<Box<[u8]>, u64>,
    /// The other way, so a traversal can answer in the client's ids.
    ///
    /// An empty entry is a dense id whose node has been removed. The slot is
    /// not handed out again, so a client holding an id from an earlier reply
    /// cannot have it silently mean a different node.
    of: Vec<Box<[u8]>>,
    /// A label's name, to the number the plane uses for it.
    labels: HashMap<Box<[u8]>, u32>,
    /// The other way, for a reply that has to name a label.
    names: Vec<Box<[u8]>>,
    /// The buffer a property document is built in, kept so a write does not
    /// allocate.
    build: Builder,
    /// The frontier a walk is standing on, kept for the same reason.
    frontier: Vec<u64>,
    /// The frontier being built, swapped with the one above each hop.
    next: Vec<u64>,
}

impl Foreign for GraphBody {
    fn type_name(&self) -> &'static str {
        "graph"
    }

    fn encoding(&self) -> &'static str {
        // One encoding, since the cold form is a tier under this one rather
        // than a second shape a key can be in.
        "adjacency"
    }

    fn memory_bytes(&self) -> usize {
        let ids: usize = self
            .of
            .iter()
            .map(|id| id.len() + std::mem::size_of::<Box<[u8]>>())
            .sum();
        let names: usize = self
            .names
            .iter()
            .map(|n| n.len() + std::mem::size_of::<Box<[u8]>>())
            .sum();
        // The map is counted at its own entries rather than at its capacity,
        // which is the same approximation the rest of the keyspace makes, plus
        // a second copy of each id because the map holds one too.
        self.g.memory_bytes() + ids * 2 + names * 2
    }

    fn is_empty(&self) -> bool {
        self.g.nodes() == 0
    }
}

impl GraphBody {
    /// The dense id for a client's id, or `None` if there is no such node.
    fn dense(&self, id: &[u8]) -> Option<u64> {
        self.ids.get(id).copied()
    }

    /// The dense id for a client's id, making the node if it is not there.
    ///
    /// The node is created with no properties, which is four bytes: an empty
    /// object. See [`Graph::add_node`], where an isolated node being a document
    /// is what makes the property store double as the node table.
    fn dense_or_add(&mut self, id: &[u8]) -> Result<u64> {
        if let Some(at) = self.ids.get(id) {
            return Ok(*at);
        }
        let at = self.of.len() as u64;
        self.g.add_node(at)?;
        self.of.push(id.into());
        self.ids.insert(id.into(), at);
        Ok(at)
    }

    /// The label number for a name, or `None` if nothing has used it.
    fn label(&self, name: &[u8]) -> Option<u32> {
        self.labels.get(name).copied()
    }

    /// The label number for a name, minting one the first time it is seen.
    fn label_or_add(&mut self, name: &[u8]) -> u32 {
        if let Some(n) = self.labels.get(name) {
            return *n;
        }
        let n = self.names.len() as u32;
        self.names.push(name.into());
        self.labels.insert(name.into(), n);
        n
    }

    /// The client's id for a dense one.
    ///
    /// Only called with a dense id that came out of the plane a moment ago, so
    /// a miss would mean the plane holds an edge to a node the tables never
    /// saw, which is the one inconsistency this file exists to prevent.
    fn client(&self, at: u64) -> &[u8] {
        self.of
            .get(at as usize)
            .expect("the plane only holds nodes these tables made")
    }

    /// Build a property document out of alternating field and value arguments.
    ///
    /// Every value is text. A graph property is whatever the client sent and
    /// there is nothing on the wire that says which of `5` and `"5"` it meant,
    /// so guessing would make `G.NGET` hand back something the client did not
    /// write. The embedded API has types because it has a struct to read them
    /// off.
    fn document(&mut self, args: Args<'_>, from: usize) -> Result<&[u8]> {
        self.build.clear();
        self.build.begin_object()?;
        let mut i = from;
        while i + 1 < args.len() {
            self.build.key(args.get(i))?;
            self.build.text_bytes(args.get(i + 1))?;
            i += 2;
        }
        self.build.end_object()?;
        self.build.finish()
    }

    /// The slot of the edge from `src` to `dst` under `label`, if there is one.
    ///
    /// The first one, when there are several. The plane allows parallel edges
    /// and the wire does not name them, so `G.EADD` on a pair that already has
    /// two updates the older of them and `G.EDEL` takes it away. A client that
    /// wants both writes both and reads them back with `G.OUT`, which does
    /// answer a repeated neighbour once per edge.
    fn slot(&self, src: u64, dst: u64, label: u32) -> Option<u32> {
        self.g
            .hop(src, label, Dir::Out)
            .find(|(to, _)| *to == dst)
            .map(|(_, slot)| slot)
    }
}

/// Run one graph command.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "g.nadd" => nadd(db, args, out),
        "g.nget" => nget(db, args, out),
        "g.ndel" => ndel(db, args, out),
        "g.eadd" => eadd(db, args, out),
        "g.edel" => edel(db, args, out),
        "g.out" => step(db, args, Dir::Out, out),
        "g.in" => step(db, args, Dir::In, out),
        "g.deg" => deg(db, args, out),
        "g.neigh" => neigh(db, args, out),
        "g.path" => path(db, args, out),
        other => unreachable!("{other} is not a graph command"),
    }
}

/// `G.NADD key id [field value ...]`, which answers 1 for a new node.
fn nadd(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    pairs(args, 3)?;
    let key = args.get(1);
    let body = open(db, key)?;
    let fresh = body.dense(args.get(2)).is_none();
    let at = body.dense_or_add(args.get(2))?;
    // The document is built into the body's own buffer and then handed to the
    // store, so a node with ten fields is one encode and no allocation.
    let doc = body.document(args, 3)?;
    // The borrow of the buffer has to end before the store can be written to,
    // which is why this goes through a slice the graph copies rather than
    // through a closure.
    let doc = doc.to_vec();
    body.g.put_node(at, &doc)?;
    out.int(i64::from(fresh));
    Ok(())
}

/// `G.NGET key id`, which answers a map of the node's fields.
fn nget(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = read(db, args.get(1))? else {
        out.nil();
        return Ok(());
    };
    let Some(at) = body.dense(args.get(2)) else {
        out.nil();
        return Ok(());
    };
    match body.g.node(at) {
        Some(doc) => fields(&doc, out),
        None => out.nil(),
    }
    Ok(())
}

/// `G.NDEL key id`, which answers 1 if the node was there.
fn ndel(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = write(db, args.get(1))? else {
        out.int(0);
        return Ok(());
    };
    let Some(at) = body.dense(args.get(2)) else {
        out.int(0);
        return Ok(());
    };
    let gone = body.g.remove_node(at)?;
    if gone {
        // Both tables, and in this order, so there is never a moment where the
        // vector says a node exists and the map does not.
        body.ids.remove(args.get(2));
        body.of[at as usize] = Box::default();
    }
    out.int(i64::from(gone));
    // A graph whose last node has gone takes its key with it, which is what
    // every other collection here does.
    db.reap_foreign(args.get(1));
    Ok(())
}

/// `G.EADD key src dst label [field value ...]`, which answers 1 for a new edge.
fn eadd(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    pairs(args, 5)?;
    let body = open(db, args.get(1))?;
    let src = body.dense_or_add(args.get(2))?;
    let dst = body.dense_or_add(args.get(3))?;
    let label = body.label_or_add(args.get(4));
    let doc = body.document(args, 5)?.to_vec();
    match body.slot(src, dst, label) {
        Some(slot) => {
            body.g.put_edge(slot, &doc)?;
            out.int(0);
        }
        None => {
            body.g.link(src, dst, label, &doc)?;
            out.int(1);
        }
    }
    Ok(())
}

/// `G.EDEL key src dst label`, which answers 1 if the edge was there.
fn edel(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(body) = write(db, args.get(1))? else {
        out.int(0);
        return Ok(());
    };
    let (Some(src), Some(dst), Some(label)) = (
        body.dense(args.get(2)),
        body.dense(args.get(3)),
        body.label(args.get(4)),
    ) else {
        out.int(0);
        return Ok(());
    };
    out.int(i64::from(body.g.unlink(src, dst, label).is_some()));
    Ok(())
}

/// `G.OUT key id label [COUNT n] [CURSOR c]` and `G.IN`, which are one command
/// with the direction in the name.
///
/// The reply is the next cursor and then the neighbours, which is `SCAN`'s
/// shape, so a client that can page one can page the other. A cursor of zero
/// coming back is the end of the run.
fn step(db: &mut Keyspace, args: Args<'_>, dir: Dir, out: &mut Out) -> Result<()> {
    let (count, cursor) = page(args, 4)?;
    let Some(body) = read(db, args.get(1))? else {
        return empty_page(out);
    };
    let (Some(at), Some(label)) = (body.dense(args.get(2)), body.label(args.get(3))) else {
        return empty_page(out);
    };
    let run = body.g.neighbours(at, label, dir);
    let from = cursor.min(run.len());
    let to = from.saturating_add(count).min(run.len());
    let next = if to < run.len() { to } else { 0 };

    out.array(2);
    out.bulk_u64(next as u64);
    out.array(to - from);
    for hop in &run[from..to] {
        out.bulk(body.client(*hop));
    }
    Ok(())
}

/// `G.DEG key id label [IN|OUT|BOTH]`, which answers a count.
fn deg(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let dir = match args.opt(4) {
        None => Dir::Out,
        Some(w) if args::is(w, b"out") => Dir::Out,
        Some(w) if args::is(w, b"in") => Dir::In,
        Some(w) if args::is(w, b"both") => Dir::Out,
        Some(_) => return Err(args::syntax()),
    };
    let both = args.opt(4).is_some_and(|w| args::is(w, b"both"));
    let Some(body) = read(db, args.get(1))? else {
        out.int(0);
        return Ok(());
    };
    let (Some(at), Some(label)) = (body.dense(args.get(2)), body.label(args.get(3))) else {
        out.int(0);
        return Ok(());
    };
    // BOTH counts a self loop twice, because it is two entries in the plane and
    // saying otherwise would mean walking the run to look for them.
    let mut n = body.g.degree(at, label, dir);
    if both {
        n += body.g.degree(at, label, Dir::In);
    }
    out.int(i64::try_from(n).unwrap_or(i64::MAX));
    Ok(())
}

/// `G.NEIGH key id label DEPTH d [COUNT n]`, the frontier `d` hops out.
///
/// Everything reachable in exactly up to `d` hops, the start excluded, each
/// node once however many ways there are to reach it. A walk asks which nodes
/// it can get to and not by how many routes, which is the same call the typed
/// surface makes and the reason the frontier is deduplicated between hops
/// rather than at the end: a frontier that keeps every path grows as the
/// product of the degrees.
fn neigh(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut depth = DEPTH;
    let mut count = usize::MAX;
    let mut i = 4;
    while i < args.len() {
        let rest = args.len() - i;
        if args::is(args.get(i), b"depth") && rest >= 2 {
            depth = positive(args.get(i + 1), BAD_DEPTH)?;
        } else if args::is(args.get(i), b"count") && rest >= 2 {
            count = positive(args.get(i + 1), BAD_COUNT)?;
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }

    let Some(body) = write(db, args.get(1))? else {
        out.array(0);
        return Ok(());
    };
    let (Some(at), Some(label)) = (body.dense(args.get(2)), body.label(args.get(3))) else {
        out.array(0);
        return Ok(());
    };

    // Two vectors owned by the body and swapped each hop, so a walk of any
    // depth allocates nothing after the first one that grew them.
    let mut seen = vec![at];
    let mut frontier = std::mem::take(&mut body.frontier);
    let mut next = std::mem::take(&mut body.next);
    frontier.clear();
    frontier.push(at);
    let start = out.len();
    let mut written = 0;
    for _ in 0..depth {
        next.clear();
        // Every header is asked for before any of them is read, which is the
        // 37 percent the bench measures on a two hop walk. It is worth doing
        // here and not only in the typed surface, because a wire client has no
        // other way to get it.
        for node in &frontier {
            body.g.prefetch(*node, label, Dir::Out);
        }
        for node in &frontier {
            next.extend_from_slice(body.g.neighbours(*node, label, Dir::Out));
        }
        next.sort_unstable();
        next.dedup();
        frontier.clear();
        for node in &next {
            if seen.binary_search(node).is_ok() {
                continue;
            }
            frontier.push(*node);
        }
        if frontier.is_empty() {
            break;
        }
        for node in &frontier {
            if written == count {
                break;
            }
            out.bulk(body.client(*node));
            written += 1;
        }
        seen.extend_from_slice(&frontier);
        seen.sort_unstable();
        if written == count {
            break;
        }
    }
    body.frontier = frontier;
    body.next = next;
    out.close_array(start, written);
    Ok(())
}

/// `G.PATH key src dst [MAXDEPTH d]`, the nodes along a shortest path.
///
/// A search from both ends at once, which is the whole reason this is a command
/// and not something a client builds out of `G.OUT`. A one sided search over a
/// graph of branching factor `b` touches `b^d` nodes and two half searches
/// touch `2 * b^(d/2)`, so at a branching factor of 30 and a distance of 6 that
/// is 729 million against 54 thousand.
///
/// The reply is the nodes from `src` to `dst` inclusive, or an empty array when
/// there is no path within the depth. `src` equal to `dst` is a path of one.
fn path(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut max = MAXDEPTH;
    if let Some(word) = args.opt(4) {
        if args.len() != 6 || !args::is(word, b"maxdepth") {
            return Err(args::syntax());
        }
        max = positive(args.get(5), BAD_DEPTH)?;
    }
    let Some(body) = read(db, args.get(1))? else {
        out.array(0);
        return Ok(());
    };
    let (Some(src), Some(dst)) = (body.dense(args.get(2)), body.dense(args.get(3))) else {
        out.array(0);
        return Ok(());
    };

    match search(body, src, dst, max) {
        Some(nodes) => {
            out.array(nodes.len());
            for node in nodes {
                out.bulk(body.client(node));
            }
        }
        None => out.array(0),
    }
    Ok(())
}

/// The two sided search behind `G.PATH`, over every label at once.
///
/// Every label, because `G.PATH` does not take one. A shortest path question
/// asked of a graph with a FOLLOWS edge and a WORKS_AT edge means either, and a
/// client that wants one kind of hop says so with `G.NEIGH`.
///
/// Each side keeps who it came from, so the path is read back by walking the
/// two parents from the node where they met. That is one `u64` per node
/// touched, against a vector per path if the frontier carried whole paths.
fn search(body: &GraphBody, src: u64, dst: u64, max: usize) -> Option<Vec<u64>> {
    if src == dst {
        return Some(vec![src]);
    }
    let mut from_src: HashMap<u64, u64> = HashMap::from([(src, src)]);
    let mut from_dst: HashMap<u64, u64> = HashMap::from([(dst, dst)]);
    let mut a = vec![src];
    let mut b = vec![dst];
    for _ in 0..max {
        // The smaller side is expanded, which is what keeps a search between a
        // popular node and an obscure one from paying the popular one's degree
        // on every level.
        let (near, far, seen, other, dir) = if a.len() <= b.len() {
            (&mut a, &mut b, &mut from_src, &from_dst, Dir::Out)
        } else {
            (&mut b, &mut a, &mut from_dst, &from_src, Dir::In)
        };
        let mut grown = Vec::new();
        for node in near.iter() {
            for label in body.g.labels() {
                for to in body.g.neighbours(*node, *label, dir) {
                    if seen.contains_key(to) {
                        continue;
                    }
                    seen.insert(*to, *node);
                    if other.contains_key(to) {
                        return Some(join(&from_src, &from_dst, *to, src, dst));
                    }
                    grown.push(*to);
                }
            }
        }
        if grown.is_empty() {
            return None;
        }
        *near = grown;
        let _ = far;
    }
    None
}

/// The path through `meet`, read out of the two parent maps.
fn join(
    from_src: &HashMap<u64, u64>,
    from_dst: &HashMap<u64, u64>,
    meet: u64,
    src: u64,
    dst: u64,
) -> Vec<u64> {
    let mut head = vec![meet];
    let mut at = meet;
    while at != src {
        at = from_src[&at];
        head.push(at);
    }
    head.reverse();
    let mut at = meet;
    while at != dst {
        at = from_dst[&at];
        head.push(at);
    }
    head
}

/// The graph under `key`, making one if the key is free.
///
/// An error for a key holding anything else, including a foreign body that is
/// not a graph, which is the one case the keyspace cannot decide on its own
/// because only this file knows which foreign body it wanted.
fn open<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<&'d mut GraphBody> {
    if db.kind_of(key).is_none() {
        db.put_foreign(key, Box::new(GraphBody::default()));
    }
    match write(db, key)? {
        Some(body) => Ok(body),
        // The key was made a line ago and nothing between here and there can
        // have taken it away, so this is the assertion rather than a case.
        None => unreachable!("the graph was just created"),
    }
}

/// The graph under `key` for writing, or `None` if the key is not there.
fn write<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d mut GraphBody>> {
    match db.foreign_mut(key)? {
        Some(body) => match body.downcast_mut::<GraphBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, NOT_A_GRAPH)),
        },
        None => Ok(None),
    }
}

/// The same, for reading.
fn read<'d>(db: &'d mut Keyspace, key: &[u8]) -> Result<Option<&'d GraphBody>> {
    match db.foreign(key)? {
        Some(body) => match body.downcast_ref::<GraphBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, NOT_A_GRAPH)),
        },
        None => Ok(None),
    }
}

/// A document's fields, as the map a client reads.
///
/// Every value goes out as a bulk string, because every value went in as one.
/// A node written by the embedded API can hold an integer or a nested object,
/// and those come back as their text and as an empty string, which is the
/// honest answer for a shape this protocol has no way to carry.
fn fields(doc: &Doc<'_>, out: &mut Out) {
    out.map(doc.len());
    for (name, value) in doc.members() {
        out.bulk(name);
        match value.text_bytes() {
            Some(text) => out.bulk(text),
            None => match value.as_int() {
                Some(n) => out.bulk_int(n),
                None => out.bulk(b""),
            },
        }
    }
}

/// `COUNT n` and `CURSOR c`, in either order, from `from` on.
fn page(args: Args<'_>, from: usize) -> Result<(usize, usize)> {
    let mut count = COUNT;
    let mut cursor = 0;
    let mut i = from;
    while i < args.len() {
        let rest = args.len() - i;
        if args::is(args.get(i), b"count") && rest >= 2 {
            count = positive(args.get(i + 1), BAD_COUNT)?;
        } else if args::is(args.get(i), b"cursor") && rest >= 2 {
            cursor = match parse_i64(args.get(i + 1)) {
                Some(n) if n >= 0 => usize::try_from(n).unwrap_or(usize::MAX),
                _ => return Err(args::syntax()),
            };
        } else {
            return Err(args::syntax());
        }
        i += 2;
    }
    Ok((count, cursor))
}

/// The cursor and the empty page a missing key answers.
fn empty_page(out: &mut Out) -> Result<()> {
    out.array(2);
    out.bulk_u64(0);
    out.array(0);
    Ok(())
}

/// An argument that has to be a positive number, with its own sentence.
fn positive(arg: &[u8], msg: &'static str) -> Result<usize> {
    match parse_i64(arg) {
        Some(n) if n > 0 => Ok(usize::try_from(n).unwrap_or(usize::MAX)),
        _ => Err(Error::new(Code::Invalid, msg)),
    }
}

/// Check that the arguments from `from` on come in pairs.
///
/// Checked before the key is touched, so a `G.NADD` with a field and no value
/// creates nothing rather than creating the node and then failing.
fn pairs(args: Args<'_>, from: usize) -> Result<()> {
    if (args.len() - from).is_multiple_of(2) {
        Ok(())
    } else {
        Err(args::syntax())
    }
}
