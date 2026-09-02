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
//! That is the price of a structure where every operation is O(1).
//!
//! [`Csr`], the cold form, which is the same adjacency once nothing is changing
//! it: node grouped, gap coded and bit packed, read only, and about an order of
//! magnitude smaller. On an R-MAT graph it is 12.63 bits an edge as the ids
//! come, and 9.89 after [`csr::order_by_degree`] gives the hubs the small ids.
//! On a uniformly random graph it is 15.98 against a floor of 13.44, and the
//! ordering pass moves that by nothing, which is what says the difference
//! between the two graphs is the graph rather than the encoder.
//!
//! On soc-LiveJournal1, which is a real graph and the one the target should be
//! judged on, it is 19.62 degree ordered, and the gap code is within 1.18 bits
//! of the entropy of the gaps it is coding. The bits that are left are in the
//! numbering rather than in the code, so the next thing this needs is layered
//! label propagation. [`csr`] has the full breakdown.
//!
//! ```
//! use yo_graph::{Csr, csr};
//!
//! let mut edges = vec![(0u32, 3u32), (0, 1), (2, 0), (0, 9)];
//! let to = csr::order_by_degree(10, &edges);
//! csr::renumber(&mut edges, &to);
//!
//! let cold = Csr::build(10, &mut edges);
//! assert_eq!(cold.degree(to[0]), 3);
//! ```
//!
//! [`Graph`] is the two of them together with a document behind every node and
//! every edge. The typed `Graph<N, E>` surface, the ten command `G.*` family and
//! the algorithms are the rest of M7.

#![deny(missing_docs)]

pub mod adjacency;
pub mod algo;
pub mod csr;
pub mod graph;
pub mod props;
pub mod snapshot;

pub use adjacency::{Adjacency, Dir};
pub use csr::Csr;
pub use graph::{Graph, NO_PROPS};
pub use props::{Props, id_key};
pub use snapshot::Snapshot;
