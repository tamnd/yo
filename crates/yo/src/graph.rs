//! The typed graph surface, where a traversal that does not make sense does not
//! compile (`11` section 6).
//!
//! A node type is a struct with an id and a label, an edge type is a struct that
//! says which node type it goes from and which it goes to, and a walk is a chain
//! of method calls. There is no query language, and the argument for not having
//! one is this file: what a Cypher engine finds out at run time, this finds out
//! at compile time.
//!
//! ```
//! use yo::{Edge, Node, Yo};
//!
//! #[derive(Yo, Debug, PartialEq)]
//! struct Person {
//!     #[yo(id)]
//!     id: u64,
//!     #[yo(index)]
//!     city: String,
//! }
//!
//! #[derive(Yo, Debug, PartialEq)]
//! struct Follows {
//!     since: i64,
//! }
//!
//! impl Node for Person {
//!     const LABEL: &'static str = "Person";
//! }
//!
//! impl Edge for Follows {
//!     type From = Person;
//!     type To = Person;
//!     const LABEL: &'static str = "FOLLOWS";
//! }
//!
//! let db = yo::open(yo::MEMORY)?;
//! let g = db.graph("social")?;
//!
//! let ada = g.add(&Person { id: 1, city: "london".to_owned() })?;
//! let grace = g.add(&Person { id: 2, city: "london".to_owned() })?;
//! let edsger = g.add(&Person { id: 3, city: "austin".to_owned() })?;
//!
//! g.link(ada, grace, &Follows { since: 2024 })?;
//! g.link(grace, edsger, &Follows { since: 2026 })?;
//!
//! // Who does the person I follow follow.
//! let two = g.walk(ada).out::<Follows>()?.out::<Follows>()?.nodes()?;
//! assert_eq!(two, vec![Person { id: 3, city: "austin".to_owned() }]);
//!
//! // And a walk can start at an index rather than at an id.
//! assert_eq!(g.find(Person::CITY, "london")?.len(), 2);
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! # What does not compile
//!
//! An edge carries where it goes from and where it goes to in its type, so
//! `out::<Follows>()` on a walk that is standing on a `Company` is a type error
//! and not an empty result.
//!
//! ```compile_fail
//! # use yo::{Edge, Node, Yo};
//! # #[derive(Yo)] struct Person { #[yo(id)] id: u64 }
//! # #[derive(Yo)] struct Company { #[yo(id)] id: u64 }
//! # #[derive(Yo)] struct Follows { since: i64 }
//! # impl Node for Person { const LABEL: &'static str = "Person"; }
//! # impl Node for Company { const LABEL: &'static str = "Company"; }
//! # impl Edge for Follows { type From = Person; type To = Person; const LABEL: &'static str = "FOLLOWS"; }
//! # let db = yo::open(yo::MEMORY).unwrap();
//! # let g = db.graph("social").unwrap();
//! let acme = g.add(&Company { id: 100 }).unwrap();
//! // Follows starts at a Person, so there is no such hop from a Company.
//! g.walk(acme).out::<Follows>().unwrap();
//! ```
//!
//! Linking is the same. An edge's ends are its own types, so putting a company
//! on the wrong end of a `Follows` is a type error at the call site.
//!
//! ```compile_fail
//! # use yo::{Edge, Node, Yo};
//! # #[derive(Yo)] struct Person { #[yo(id)] id: u64 }
//! # #[derive(Yo)] struct Company { #[yo(id)] id: u64 }
//! # #[derive(Yo)] struct Follows { since: i64 }
//! # impl Node for Person { const LABEL: &'static str = "Person"; }
//! # impl Node for Company { const LABEL: &'static str = "Company"; }
//! # impl Edge for Follows { type From = Person; type To = Person; const LABEL: &'static str = "FOLLOWS"; }
//! # let db = yo::open(yo::MEMORY).unwrap();
//! # let g = db.graph("social").unwrap();
//! # let ada = g.add(&Person { id: 1 }).unwrap();
//! let acme = g.add(&Company { id: 100 }).unwrap();
//! g.link(ada, acme, &Follows { since: 2026 }).unwrap();
//! ```
//!
//! # An [`Id`] is not the id you wrote
//!
//! [`Graph::add`] hands back an `Id<Person>`, which is a handle into this graph
//! and not the `1` in the struct. The adjacency plane is keyed by a dense `u64`
//! and that is what makes a hop a probe and a sequential read, so the id in the
//! struct is looked up once on the way in and never again. Everything a walk
//! touches is already dense.
//!
//! That is the trade and it is worth stating plainly. An entry point costs a
//! hash lookup, and a hop costs nothing extra at all. A graph engine that keyed
//! its adjacency by whatever the user's id happened to be would pay that lookup
//! on every hop of every walk instead.
//!
//! Because the id is a handle, [`Graph::id_of`] is how you get back to one from
//! the id you wrote, and it is the only call that costs the lookup.
//!
//! # One store, several node types
//!
//! Every node type shares one document collection and one adjacency plane, and
//! the label keeps them apart. That costs four bytes a node, which is the label
//! beside each dense id, and it buys the thing that matters: a two hop walk that
//! crosses from `Person` to `Company` is one plane and one contiguous run, not a
//! join between two stores.
//!
//! An index is declared per path rather than per type, so two node types that
//! both index `$.name` share one index and [`Graph::find`] filters the answer by
//! label. That over-reads when two types share a path and share values, and it
//! is written down here rather than found later. Two types that index the same
//! path for different kinds is a conflict and is refused.

use core::marker::PhantomData;
use std::collections::HashMap;

use yo_common::{Code, Error, Result};
use yo_doc::Builder;
use yo_graph::Dir;
use yo_shape::{Desc, Shape, Tag};

use crate::db::Handle;
use crate::doc::{Asked, Document, Field, IndexKind, Indexed, Path, key_of};

/// A type that is a node in a graph.
///
/// A [`Document`] with a label. The label is what keeps two node types apart in
/// one store, and it is a string rather than a number so that a graph read back
/// off disk knows what it is holding.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a node type",
    label = "this type has no label",
    note = "give it a label with an `impl Node` block naming a `const LABEL`, and make sure it derives Yo with one field marked `#[yo(id)]`"
)]
pub trait Node: Document {
    /// What this type is called in the graph.
    const LABEL: &'static str;
}

/// A type that is an edge in a graph.
///
/// The two ends are in the type, which is the whole point: a hop that a node
/// type cannot start is a type error rather than an empty answer, and an edge
/// put between the wrong pair does not compile.
///
/// An edge has no id, because an edge is identified by where it is rather than
/// by a field, so this is a [`Field`] and an [`Indexed`] rather than a
/// [`Document`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an edge type",
    label = "this type does not say where it goes",
    note = "say where it goes with an `impl Edge` block naming a `From`, a `To` and a `const LABEL`"
)]
pub trait Edge: Field + Indexed {
    /// The node type an edge of this kind starts at.
    type From: Node;
    /// The node type an edge of this kind ends at.
    type To: Node;
    /// What this type is called in the graph.
    const LABEL: &'static str;
}

/// A node in a graph.
///
/// Handed out by [`Graph::add`] and by [`Graph::id_of`], and taken by every call
/// that starts somewhere. The type parameter is what makes a wrong hop a compile
/// error, so it is not a `u64` you can build by hand.
pub struct Id<N> {
    raw: u64,
    /// `fn() -> N` so that an `Id` is `Copy` and `Send` whatever `N` is.
    marker: PhantomData<fn() -> N>,
}

impl<N> Clone for Id<N> {
    fn clone(&self) -> Id<N> {
        *self
    }
}

impl<N> Copy for Id<N> {}

impl<N> PartialEq for Id<N> {
    fn eq(&self, other: &Id<N>) -> bool {
        self.raw == other.raw
    }
}

impl<N> Eq for Id<N> {}

impl<N> core::hash::Hash for Id<N> {
    fn hash<H: core::hash::Hasher>(&self, h: &mut H) {
        self.raw.hash(h);
    }
}

impl<N: Node> core::fmt::Debug for Id<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}#{}", N::LABEL, self.raw)
    }
}

impl<N> Id<N> {
    fn new(raw: u64) -> Id<N> {
        Id {
            raw,
            marker: PhantomData,
        }
    }
}

/// One edge in a graph, which is what [`Graph::link`] answers with.
///
/// Two edges of the same kind between the same pair are two edges, so this
/// names one of them and reading an edge's fields needs it.
pub struct EdgeId<E> {
    slot: u32,
    marker: PhantomData<fn() -> E>,
}

impl<E> Clone for EdgeId<E> {
    fn clone(&self) -> EdgeId<E> {
        *self
    }
}

impl<E> Copy for EdgeId<E> {}

impl<E> PartialEq for EdgeId<E> {
    fn eq(&self, other: &EdgeId<E>) -> bool {
        self.slot == other.slot
    }
}

impl<E> Eq for EdgeId<E> {}

impl<E: Edge> core::fmt::Debug for EdgeId<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}#{}", E::LABEL, self.slot)
    }
}

/// One step across an edge: where it went, and which edge it was.
///
/// A pair with names on it rather than a tuple, because the whole point of the
/// second half is that reading the edge's fields needs a handle and a caller
/// should not have to remember which end of a tuple that is.
pub struct Hop<E: Edge> {
    /// The node at the far end.
    pub to: Id<E::To>,
    /// The edge that got there, which [`Graph::edge`] reads.
    pub edge: EdgeId<E>,
}

impl<E: Edge> Clone for Hop<E> {
    fn clone(&self) -> Hop<E> {
        *self
    }
}

impl<E: Edge> Copy for Hop<E> {}

impl<E: Edge> core::fmt::Debug for Hop<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} by {:?}", self.to, self.edge)
    }
}

/// What a graph collection holds.
///
/// The plane and the documents are [`yo_graph::Graph`]. Everything else here is
/// the typing: which labels are in use, which shape each was first opened with,
/// and the one lookup that turns the id in a struct into the dense id the plane
/// is keyed by.
pub(crate) struct Store {
    g: yo_graph::Graph,
    /// Node labels, in the order they were first used. The position is the
    /// label id, which is what `of` holds.
    nodes: Vec<Kind>,
    /// Edge labels, likewise, and the position is the label the plane is given.
    edges: Vec<Kind>,
    /// The id in a struct, tagged with its node label, to the dense id.
    ids: HashMap<Box<[u8]>, u64>,
    /// The label of each dense id, or [`GONE`] for one that was removed.
    of: Vec<u32>,
    /// The next dense id never handed out.
    next: u64,
    /// Which paths are indexed on each side, and how, so that two types asking
    /// for the same path share one index and two types asking for it in two
    /// different ways is refused.
    node_paths: HashMap<&'static str, IndexKind>,
    edge_paths: HashMap<&'static str, IndexKind>,
    scratch: Builder,
}

/// A label that was removed, in `of`.
const GONE: u32 = u32::MAX;

/// A label in use, and the shape it was first used with.
struct Kind {
    label: &'static str,
    shape: Tag,
    live: usize,
}

impl Store {
    pub(crate) fn new() -> Store {
        Store {
            g: yo_graph::Graph::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            ids: HashMap::new(),
            of: Vec::new(),
            next: 0,
            node_paths: HashMap::new(),
            edge_paths: HashMap::new(),
            scratch: Builder::new(),
        }
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.g.memory_bytes()
            + self.of.capacity() * size_of::<u32>()
            + self.ids.capacity() * (size_of::<Box<[u8]>>() + size_of::<u64>() + 16)
    }

    /// The label id of a node type, registering it the first time.
    fn node_kind<N: Node>(&mut self) -> Result<u32> {
        let want = tag_of::<N>();
        let at = register(&mut self.nodes, N::LABEL, want)?;
        declare(&mut self.node_paths, N::INDEXES, N::LABEL, |path, kind| {
            self.g.index_nodes(path, kind)
        })?;
        Ok(at)
    }

    /// The label id of an edge type, registering it the first time.
    fn edge_kind<E: Edge>(&mut self) -> Result<u32> {
        let want = tag_of::<E>();
        let at = register(&mut self.edges, E::LABEL, want)?;
        declare(&mut self.edge_paths, E::INDEXES, E::LABEL, |path, kind| {
            self.g.index_edges(path, kind)
        })?;
        Ok(at)
    }

    /// The label id of an edge type without registering it, for a read that
    /// should answer nothing rather than create anything.
    fn edge_seen<E: Edge>(&self) -> Option<u32> {
        seen(&self.edges, E::LABEL)
    }

    fn node_seen<N: Node>(&self) -> Option<u32> {
        seen(&self.nodes, N::LABEL)
    }

    /// Whether a dense id is a node of `kind`.
    fn is(&self, raw: u64, kind: u32) -> bool {
        usize::try_from(raw).is_ok_and(|i| self.of.get(i).copied() == Some(kind))
    }
}

fn seen(kinds: &[Kind], label: &'static str) -> Option<u32> {
    kinds
        .iter()
        .position(|k| k.label == label)
        .map(|at| at as u32)
}

/// Find a label or add it, and check the shape has not changed under it.
fn register(kinds: &mut Vec<Kind>, label: &'static str, want: Tag) -> Result<u32> {
    if let Some(at) = kinds.iter().position(|k| k.label == label) {
        if kinds[at].shape != want {
            return Err(Error::fmt(
                Code::ShapeMismatch,
                format_args!(
                    "this graph already holds {label} under another shape, so the two types cannot share the label"
                ),
            ));
        }
        return Ok(at as u32);
    }
    if kinds.len() >= u32::MAX as usize {
        return Err(Error::new(Code::Full, "this graph has no labels left"));
    }
    kinds.push(Kind {
        label,
        shape: want,
        live: 0,
    });
    Ok((kinds.len() - 1) as u32)
}

/// Declare a type's indexes, sharing one per path across the types that ask for
/// it and refusing two types that want the same path indexed differently.
fn declare(
    have: &mut HashMap<&'static str, IndexKind>,
    want: &'static [(&'static str, IndexKind)],
    label: &'static str,
    mut create: impl FnMut(&str, IndexKind) -> Result<()>,
) -> Result<()> {
    for (path, kind) in want {
        match have.get(path) {
            Some(already) if already == kind => {}
            Some(already) => {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!(
                        "{label} asks for {path} to be indexed for {kind:?}, and another type in this graph already indexes it for {already:?}. One path is one index, so the two types have to agree"
                    ),
                ));
            }
            None => {
                create(path, *kind)?;
                have.insert(path, *kind);
            }
        }
    }
    Ok(())
}

fn tag_of<T: Shape>() -> Tag {
    let mut d = Desc::new();
    T::describe(&mut d);
    d.tag()
}

/// A graph.
///
/// Cheap to clone and cheap to keep around, the same way [`crate::Docs`] is: the
/// handle is a pointer and an index, and every clone is the same graph.
#[derive(Clone)]
pub struct Graph {
    db: Handle,
    at: usize,
}

impl core::fmt::Debug for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = self
            .db
            .read(|inner| Ok(inner.collections[self.at].name.clone()))
            .unwrap_or_else(|_| "?".to_owned());
        f.debug_struct("Graph").field("name", &name).finish()
    }
}

impl Graph {
    pub(crate) fn new(db: Handle, at: usize) -> Graph {
        Graph { db, at }
    }

    /// The name this graph was opened under.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn name(&self) -> Result<String> {
        self.db
            .read(|inner| Ok(inner.collections[self.at].name.clone()))
    }

    fn write<R>(&self, f: impl FnOnce(&mut Store) -> Result<R>) -> Result<R> {
        self.db
            .write(|inner| f(inner.collections[self.at].data.graph_mut()))
    }

    fn read<R>(&self, f: impl FnOnce(&Store) -> Result<R>) -> Result<R> {
        self.db
            .read(|inner| f(inner.collections[self.at].data.graph()))
    }

    /// Put a node in, replacing whatever was under its id.
    ///
    /// The [`Id`] that comes back is this graph's handle on the node and is the
    /// same one every time for the same struct id, so adding a node twice
    /// updates it rather than making a second one.
    ///
    /// # Errors
    ///
    /// [`Code::ShapeMismatch`] when another type already uses this label, and
    /// [`Code::Invalid`] for an id that cannot be a key.
    pub fn add<N: Node>(&self, node: &N) -> Result<Id<N>> {
        let key = key_of(node.id(), IndexKind::Equality, "the id")?;
        self.write(|s| {
            let kind = s.node_kind::<N>()?;
            let tagged = tagged(kind, key.as_bytes());
            let raw = match s.ids.get(&tagged[..]) {
                Some(raw) => *raw,
                None => {
                    let raw = s.next;
                    s.next += 1;
                    s.ids.insert(tagged.into_boxed_slice(), raw);
                    s.of.push(kind);
                    s.nodes[kind as usize].live += 1;
                    raw
                }
            };
            s.scratch.clear();
            Field::write(node, &mut s.scratch)?;
            let bytes = s.scratch.finish()?;
            s.g.put_node(raw, bytes)?;
            Ok(Id::new(raw))
        })
    }

    /// This graph's handle on the node with the id you wrote, if it has one.
    ///
    /// The one call that pays the lookup from a struct id to a dense id, which
    /// is what the module docs are about.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for an id that cannot be a key.
    pub fn id_of<N: Node>(&self, id: &<N::Id as Asked>::Ask) -> Result<Option<Id<N>>> {
        let key = key_of(id, IndexKind::Equality, "the id")?;
        self.read(|s| {
            let Some(kind) = s.node_seen::<N>() else {
                return Ok(None);
            };
            Ok(s.ids
                .get(&tagged(kind, key.as_bytes())[..])
                .map(|raw| Id::new(*raw)))
        })
    }

    /// Read a node back.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the stored node is not an `N`, which is a graph that
    /// disagrees with its own labels.
    pub fn get<N: Node>(&self, id: Id<N>) -> Result<Option<N>> {
        self.read(|s| match s.g.node(id.raw) {
            Some(doc) => N::read(doc).map(Some),
            None => Ok(None),
        })
    }

    /// Whether this graph still has that node.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn has<N: Node>(&self, id: Id<N>) -> Result<bool> {
        self.read(|s| Ok(s.g.has_node(id.raw)))
    }

    /// Take a node out, along with every edge at either end of it.
    ///
    /// Answers whether the node was there.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the stored node is not an `N`.
    pub fn remove<N: Node>(&self, id: Id<N>) -> Result<bool> {
        self.write(|s| {
            let Some(doc) = s.g.node(id.raw) else {
                return Ok(false);
            };
            // The struct is read back so that its id can be turned into the key
            // the lookup table is holding. Storing the key a second time would
            // be a copy of every id in the graph to save this one decode on a
            // path that is already removing edges.
            let node = N::read(doc)?;
            let key = key_of(node.id(), IndexKind::Equality, "the id")?;
            let Some(kind) = s.node_seen::<N>() else {
                return Ok(false);
            };
            s.ids.remove(&tagged(kind, key.as_bytes())[..]);
            if let Ok(i) = usize::try_from(id.raw)
                && let Some(slot) = s.of.get_mut(i)
            {
                *slot = GONE;
            }
            s.nodes[kind as usize].live -= 1;
            s.g.remove_node(id.raw)
        })
    }

    /// How many nodes of this type the graph holds.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn count<N: Node>(&self) -> Result<usize> {
        self.read(|s| {
            Ok(s.node_seen::<N>()
                .map_or(0, |kind| s.nodes[kind as usize].live))
        })
    }

    /// How many nodes of every type the graph holds.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn nodes(&self) -> Result<usize> {
        self.read(|s| Ok(s.g.nodes()))
    }

    /// How many edges of every type the graph holds, counting two edges between
    /// the same pair as two.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn edges(&self) -> Result<usize> {
        self.read(|s| Ok(s.g.edges()))
    }

    /// Put an edge between two nodes.
    ///
    /// Linking the same pair twice leaves two edges, each with its own fields,
    /// because that is what a property graph means by a multigraph and because
    /// two ratings of the same film on two dates is the case rather than the
    /// corner case.
    ///
    /// # Errors
    ///
    /// [`Code::NotFound`] if either end is not in the graph, because an edge
    /// hanging off an id that was never added is a dangling reference that every
    /// later read would have to guard against.
    pub fn link<E: Edge>(&self, from: Id<E::From>, to: Id<E::To>, edge: &E) -> Result<EdgeId<E>> {
        self.write(|s| {
            let label = s.edge_kind::<E>()?;
            let from_kind = s.node_kind::<E::From>()?;
            let to_kind = s.node_kind::<E::To>()?;
            if !s.is(from.raw, from_kind) {
                return Err(gone::<E::From>(from.raw));
            }
            if !s.is(to.raw, to_kind) {
                return Err(gone::<E::To>(to.raw));
            }
            s.scratch.clear();
            Field::write(edge, &mut s.scratch)?;
            let bytes = s.scratch.finish()?;
            let slot = s.g.link(from.raw, to.raw, label, bytes)?;
            Ok(EdgeId {
                slot,
                marker: PhantomData,
            })
        })
    }

    /// Take one edge of this kind out from between two nodes.
    ///
    /// Answers whether there was one. With two edges between the same pair it
    /// takes one of them, and which one is whatever the run's order left.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn unlink<E: Edge>(&self, from: Id<E::From>, to: Id<E::To>) -> Result<bool> {
        self.write(|s| {
            let Some(label) = s.edge_seen::<E>() else {
                return Ok(false);
            };
            Ok(s.g.unlink(from.raw, to.raw, label).is_some())
        })
    }

    /// Read an edge's fields.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the stored edge is not an `E`.
    pub fn edge<E: Edge>(&self, id: EdgeId<E>) -> Result<Option<E>> {
        self.read(|s| match s.g.edge(id.slot) {
            Some(doc) => E::read(doc).map(Some),
            None => Ok(None),
        })
    }

    /// Where an edge of this kind goes from this node.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn out<E: Edge>(&self, from: Id<E::From>) -> Result<Vec<Id<E::To>>> {
        self.step::<E>(from.raw, Dir::Out).map(ids)
    }

    /// Where an edge of this kind comes into this node from.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn incoming<E: Edge>(&self, to: Id<E::To>) -> Result<Vec<Id<E::From>>> {
        self.step::<E>(to.raw, Dir::In).map(ids)
    }

    /// The same as [`Graph::out`], with the edge that got to each one.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn out_edges<E: Edge>(&self, from: Id<E::From>) -> Result<Vec<Hop<E>>> {
        self.read(|s| {
            let Some(label) = s.edge_seen::<E>() else {
                return Ok(Vec::new());
            };
            Ok(s.g
                .hop(from.raw, label, Dir::Out)
                .map(|(node, slot)| Hop {
                    to: Id::new(node),
                    edge: EdgeId {
                        slot,
                        marker: PhantomData,
                    },
                })
                .collect())
        })
    }

    /// How many edges of this kind leave this node.
    ///
    /// Read off the run header, so it does not touch the run.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn degree<E: Edge>(&self, from: Id<E::From>) -> Result<usize> {
        self.read(|s| {
            Ok(s.edge_seen::<E>()
                .map_or(0, |label| s.g.degree(from.raw, label, Dir::Out)))
        })
    }

    /// Start a walk at a node.
    #[must_use]
    pub fn walk<N: Node>(&self, from: Id<N>) -> Walk<'_, N> {
        Walk {
            g: self,
            at: vec![from.raw],
            marker: PhantomData,
        }
    }

    /// Start a walk at everything a path index answers.
    ///
    /// # Errors
    ///
    /// The same as [`Graph::find`].
    pub fn walk_from<N: Node, V: Asked>(
        &self,
        path: impl Into<Path<N, V>>,
        value: &V::Ask,
    ) -> Result<Walk<'_, N>> {
        let at = self.matching::<N, V>(path.into(), value)?;
        Ok(Walk {
            g: self,
            at,
            marker: PhantomData,
        })
    }

    /// Every node of this type with `value` at `path`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the path is not indexed, and [`Code::Corrupt`] if a
    /// stored node is not an `N`.
    pub fn find<N: Node, V: Asked>(
        &self,
        path: impl Into<Path<N, V>>,
        value: &V::Ask,
    ) -> Result<Vec<N>> {
        let path = path.into();
        let key = key_of(value, path.kind(), "this value")?;
        self.read(|s| {
            let Some(kind) = s.node_seen::<N>() else {
                return Ok(Vec::new());
            };
            let mut out = Vec::new();
            let mut bad = None;
            s.g.find_nodes(path.path(), &key, |raw, doc| {
                // The index covers every type that asked for this path, so the
                // label is what says whether this row is an `N`.
                if !s.is(raw, kind) {
                    return;
                }
                match N::read(doc) {
                    Ok(node) => out.push(node),
                    Err(e) => bad = bad.take().or(Some(e)),
                }
            })?;
            match bad {
                Some(e) => Err(e),
                None => Ok(out),
            }
        })
    }

    /// How many nodes of this type have `value` at `path`.
    ///
    /// # Errors
    ///
    /// The same as [`Graph::find`], without the decode.
    pub fn count_at<N: Node, V: Asked>(
        &self,
        path: impl Into<Path<N, V>>,
        value: &V::Ask,
    ) -> Result<usize> {
        Ok(self.matching::<N, V>(path.into(), value)?.len())
    }

    /// What this graph weighs.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn memory_bytes(&self) -> Result<usize> {
        self.read(|s| Ok(s.memory_bytes()))
    }

    /// The dense ids of every node of this type at `path`.
    fn matching<N: Node, V: Asked>(&self, path: Path<N, V>, value: &V::Ask) -> Result<Vec<u64>> {
        let key = key_of(value, path.kind(), "this value")?;
        self.read(|s| {
            let Some(kind) = s.node_seen::<N>() else {
                return Ok(Vec::new());
            };
            let mut out = Vec::new();
            s.g.find_nodes(path.path(), &key, |raw, _| {
                if s.is(raw, kind) {
                    out.push(raw);
                }
            })?;
            Ok(out)
        })
    }

    /// One hop, in dense ids.
    fn step<E: Edge>(&self, from: u64, dir: Dir) -> Result<Vec<u64>> {
        self.read(|s| {
            Ok(s.edge_seen::<E>()
                .map_or_else(Vec::new, |label| s.g.neighbours(from, label, dir).to_vec()))
        })
    }

    /// A whole frontier hopped at once, deduplicated.
    fn frontier<E: Edge>(&self, at: &[u64], dir: Dir) -> Result<Vec<u64>> {
        self.read(|s| {
            let Some(label) = s.edge_seen::<E>() else {
                return Ok(Vec::new());
            };
            // The headers of the whole frontier are asked for before any of
            // them is read, because the frontier is known as soon as the last
            // hop finished and there is no reason for these probes to be
            // serial. It is worth about a fifth of a two hop walk.
            for node in at {
                s.g.prefetch(*node, label, dir);
            }
            let mut out = Vec::new();
            for node in at {
                out.extend_from_slice(s.g.neighbours(*node, label, dir));
            }
            // A frontier that keeps every path to a node grows as the product
            // of the degrees, and a walk is asking which nodes it can reach and
            // not by how many routes, so it is the set that goes on.
            out.sort_unstable();
            out.dedup();
            Ok(out)
        })
    }
}

/// A walk standing on a set of nodes of one type.
///
/// Each hop is a whole frontier at a time, and the frontier is a set, so a
/// diamond in the graph does not turn into two copies of the node at the far
/// end. See [`Graph::walk`].
pub struct Walk<'a, N> {
    g: &'a Graph,
    at: Vec<u64>,
    marker: PhantomData<fn() -> N>,
}

impl<N: Node> core::fmt::Debug for Walk<'_, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Walk")
            .field("on", &N::LABEL)
            .field("len", &self.at.len())
            .finish()
    }
}

impl<'a, N: Node> Walk<'a, N> {
    /// Follow every edge of this kind forwards.
    ///
    /// `E::From` is `N`, which is what makes a hop the graph does not have a
    /// compile error rather than an empty answer.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn out<E: Edge<From = N>>(self) -> Result<Walk<'a, E::To>> {
        let at = self.g.frontier::<E>(&self.at, Dir::Out)?;
        Ok(Walk {
            g: self.g,
            at,
            marker: PhantomData,
        })
    }

    /// Follow every edge of this kind backwards.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback holding this database.
    pub fn incoming<E: Edge<To = N>>(self) -> Result<Walk<'a, E::From>> {
        let at = self.g.frontier::<E>(&self.at, Dir::In)?;
        Ok(Walk {
            g: self.g,
            at,
            marker: PhantomData,
        })
    }

    /// Keep only the nodes that pass.
    ///
    /// The node is read to be looked at, so this is the expensive filter and it
    /// belongs at the end of a walk rather than in the middle of one.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if a node on the walk is not an `N`.
    pub fn filter(self, mut keep: impl FnMut(&N) -> bool) -> Result<Walk<'a, N>> {
        let mut at = Vec::with_capacity(self.at.len());
        self.g.read(|s| {
            for raw in &self.at {
                if let Some(doc) = s.g.node(*raw)
                    && keep(&N::read(doc)?)
                {
                    at.push(*raw);
                }
            }
            Ok(())
        })?;
        Ok(Walk {
            g: self.g,
            at,
            marker: PhantomData,
        })
    }

    /// How many nodes the walk is standing on.
    #[must_use]
    pub fn len(&self) -> usize {
        self.at.len()
    }

    /// Whether the walk reached nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// The nodes the walk reached, as handles.
    #[must_use]
    pub fn ids(self) -> Vec<Id<N>> {
        ids(self.at)
    }

    /// The nodes the walk reached, read out.
    ///
    /// A probe per node, which is what `11` section 4 prices this at and the
    /// reason it is a separate call rather than what a hop hands back.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if a node on the walk is not an `N`.
    pub fn nodes(self) -> Result<Vec<N>> {
        self.g.read(|s| {
            let mut out = Vec::with_capacity(self.at.len());
            for raw in &self.at {
                if let Some(doc) = s.g.node(*raw) {
                    out.push(N::read(doc)?);
                }
            }
            Ok(out)
        })
    }
}

fn ids<N>(raw: Vec<u64>) -> Vec<Id<N>> {
    raw.into_iter().map(Id::new).collect()
}

/// A node's key in the lookup table: its label and then the id in the struct.
///
/// The label is in front because two node types can hold the same id and they
/// are two nodes.
fn tagged(kind: u32, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + key.len());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(key);
    out
}

fn gone<N: Node>(raw: u64) -> Error {
    Error::fmt(
        Code::NotFound,
        format_args!(
            "{}#{raw} is not in this graph, so there is nothing to put an edge on. Add the node first, or use the id that add() answered with",
            N::LABEL
        ),
    )
}
