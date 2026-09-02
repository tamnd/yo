//! The graph model: adjacency runs a traversal can read at memory speed
//! (`11`).
//!
//! The embedded graph space had two things happen to it and neither was about
//! traversal speed. Kuzu, which would otherwise be the comparison, was acquired
//! in October 2025 and archived the next day. FalkorDB spent 2025 and 2026
//! rewriting in Rust and pointing at GraphRAG. Both events are about who is
//! willing to keep an embedded graph engine alive, which is why the format
//! being documented byte for byte and read by something other than this code is
//! a first class decision here rather than a nicety.
//!
//! What does not come along is the query processor. There is no Cypher, no GQL
//! and no factorised execution, because a graph database without a query
//! language is an adjacency structure with good ergonomics, and that is what an
//! agent memory or a recommendation workload actually calls. A traversal is an
//! iterator the caller drives, and its cost is written down rather than hidden
//! behind a planner.
//!
//! # What is here so far
//!
//! [`Adjacency`], the hot form of the adjacency plane. A run is the neighbours
//! of one node under one label in one direction, it is contiguous, and it is
//! appended to and deleted from in place. A one hop is a probe and a sequential
//! read; a two hop is a probe per neighbour over runs that can be prefetched
//! before any of them is read.
//!
//! ```
//! use yo_graph::{Adjacency, Dir};
//!
//! const FOLLOWS: u32 = 1;
//!
//! let mut g = Adjacency::new();
//! g.link(1, 2, FOLLOWS, 0);
//! g.link(2, 3, FOLLOWS, 1);
//!
//! let hop = g.neighbours(1, FOLLOWS, Dir::Out).to_vec();
//! let two: Vec<u64> = hop.iter().flat_map(|n| g.neighbours(*n, FOLLOWS, Dir::Out)).copied().collect();
//! assert_eq!(two, vec![3]);
//! ```
//!
//! Twelve bytes an edge is the payload, and the run headers and the capacity
//! slack take a graph shaped like LiveJournal to around 15 once it has settled.
//! That is the price of a structure where every operation is O(1). The cold
//! form, zu's node group CSR, is where 8 bits an edge comes from, and it is the
//! next thing here.
//!
//! The node table, the edge records, the typed `Graph<N, E>` surface, the ten
//! command `G.*` family and the algorithms are the rest of M7.

#![deny(missing_docs)]

pub mod adjacency;

pub use adjacency::{Adjacency, Dir};
