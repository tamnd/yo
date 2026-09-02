//! The cold form: the same adjacency at bits an edge rather than bytes.
//!
//! [`Adjacency`] is what a graph being written looks like,
//! twelve bytes an edge and every operation O(1). [`Csr`] is what the settled
//! part of it looks like once nothing is changing it: read only, node grouped,
//! and about an order of magnitude smaller. Spec `11` section 2 calls these the
//! hot form and the cold form and expects a graph to be mostly cold, because a
//! graph that is being traversed is very rarely being edited at the same rate.
//!
//! # The shape
//!
//! Nodes are dense `u32` ids, cut into groups of [`GROUP`] consecutive ids.
//! Everything that varies is chosen per group, so one hub does not set the
//! width of the whole graph. A group carries:
//!
//!   - a table of one bit offset per node, at the width that group's stream
//!     needs, pointing at where that node's run starts,
//!   - the runs themselves, each of them a degree, then the first neighbour
//!     written against the group's smallest neighbour, then the gaps between
//!     the rest.
//!
//! The gaps go out in blocks of [`BLOCK`] with a width per block rather than a
//! width per run. That is the one idea from BtrBlocks and FastLanes that is
//! worth taking here: a run of ten thousand where one gap is enormous and the
//! rest are small would otherwise pay the enormous one ten thousand times, and
//! a block only ever pays it thirty two times. It costs seven bits a block,
//! which is under a quarter of a bit an edge, and no extra offsets at all,
//! because a run is decoded from its start and the blocks come in order.
//!
//! A block whose width does not fit everything in it leaves the ones that did
//! not fit behind as patches, written again at the end of the block with their
//! positions, which is what PFOR does. The encoder tries every width and keeps
//! the cheapest, so a block only patches when patching is cheaper than widening.
//! This was measured before and rejected, and what changed is the numbering
//! rather than the code: under a degree ordering the gaps in one block are all
//! about the same size and there is nothing to patch, and under the community
//! numbering in [`bisect`](crate::bisect) a block is mostly ones with a jump to
//! another community in it, which is exactly the shape patching is for. It is
//! worth 0.65 bits an edge on R-MAT, 2.36 on soc-LiveJournal1 as its ids arrive,
//! and 5.41 on a bisected web-Google. On a uniform graph it is a wash, 15.98
//! before and 15.96 after, because there the widths in a block already agree and
//! the few blocks that do patch save about what the headers cost. It is worth
//! about a tenth of the decode, which is the trade being made.
//!
//! Elias gamma, which needs no width at all, was 1.7 bits an edge worse on R-MAT
//! and 7.7 worse on a uniform graph, and is not here.
//!
//! # What it costs
//!
//! Two things, and they pull in opposite directions. The payload is the gaps,
//! and how small they are is a property of the graph rather than of the
//! encoder: a uniformly random graph on `n` nodes with `m` edges cannot be
//! stored below about `log2(n * n / m) + 1.44` bits an edge no matter what
//! anybody does. The `8` bits an edge in `11` is not a claim about random
//! graphs. It is paid for entirely by real graphs having hubs and communities,
//! so neighbours cluster and the gaps between them are small.
//!
//! The overhead is the node offset table plus the per run header, and it is per
//! node rather than per edge, so it is loud at degree one and inaudible at
//! degree a hundred.
//!
//! The test at the bottom of this file measures both a uniformly random graph
//! and an R-MAT graph, which is the standard synthetic social graph and the one
//! Graph500 uses. At 65536 nodes and an average degree of 16:
//!
//! ```text
//!                          total    table     head     gaps
//! uniform                  15.96     1.06     1.31    13.58
//! R-MAT                    11.98     1.00     1.14     9.83
//! R-MAT, degree ordered     9.38     0.69     0.78     7.90
//! ```
//!
//! The uniform row is 15.96 against a floor of 13.44, so the encoder is within
//! a fifth of what is provably possible on the case where nothing can help. The
//! whole of the rest is the graph: R-MAT is 3.98 bits an edge cheaper for no
//! other reason than that it has hubs, and [`order_by_degree`] takes another
//! 2.60 off by giving those hubs the small ids. The same ordering pass moves a
//! uniform graph by nothing at all, which is the control that says it is the
//! structure being exploited rather than the measurement.
//!
//! # What it costs on a graph nobody here generated
//!
//! R-MAT is a stand in and it is known to cluster less than the social graphs it
//! stands in for, so a real one should come out under 9.38. It does not.
//! soc-LiveJournal1 from SNAP, 4847571 nodes and 68993773 edges, on server3,
//! through `examples/compress.rs`:
//!
//! ```text
//!                       total  offsets  degrees   firsts  widths   gaps  patches
//! cold                  17.99     1.18     0.49     1.43    0.82  10.58     3.47
//! cold, degree ordered  19.00     1.13     0.27     1.43    0.74  14.57     0.84
//! cold, bisected        15.00     1.14     0.42     1.44    0.91   7.53     3.55
//! ```
//!
//! Three things in that table are worth saying out loud.
//!
//! The best number is 15.00 and it takes the numbering to get there. Before the
//! patches and [`bisect`](crate::bisect) the same three rows were 20.35, 19.62
//! and nothing, so this graph is a quarter smaller than it was and the whole of
//! the difference is community structure that was there all along.
//!
//! Degree ordering now makes this graph bigger, 19.00 against 17.99. SNAP's ids
//! are roughly the order the crawl found the accounts in, which already puts
//! friends near each other, and sorting by degree throws that away. It was still
//! the right call when the code could not exploit locality, and it stopped being
//! the right call the moment the code could.
//!
//! The gaps are no longer most of the file. Under bisection the per node fields
//! are 3.00 of the 15.00 and the payload is 11.99, against a floor, quoted by
//! `--codes`, of 10.46 bits a gap or about 9.81 an edge. So the code is now 2.2
//! bits over what any code that prices each gap on its own could do, where under
//! degree ordering it was 1.18 over a floor that was itself six bits worse.
//!
//! ```text
//! gaps 64685321, 24.9% of them 1, 16.6% of neighbours in a run of 4 or more
//! by length floor       10.46 bits a gap
//! block of 32           17.06   (unpatched, which is what this used to be)
//! block of 8            13.99
//! block of 32, patched  14.06   (the model, and the real one does better)
//! elias delta           12.14
//! intervals, block 8    13.78
//! ```
//!
//! The 8 bits an edge in `11` is still not met and this is 15.00. What would
//! close it is not another block code: the three that are left are all within a
//! bit or two of each other and of the floor. It is a better numbering, and the
//! measurement that says so is that bisection's own objective, a plain log gap
//! code, prices this ordering at 9.29 bits a gap where it prices the degree
//! ordering at 14.87. The numbering has found structure the encoder is only half
//! spending. Interval encoding, which was dead on arrival under a degree
//! ordering because only 0.9 percent of neighbours were consecutive, is now
//! looking at 16.6 percent and is the first thing to try.
//!
//! # What this does not do
//!
//! It does not change. There is no link and no unlink, because every edge after
//! the first in a run is written against the one before it, so touching one is
//! rewriting the group. New edges go into the hot form and a later sweep folds
//! them in, which is the promotion in `11` section 2 and is why O-G2, when the
//! sweep should run, is still open.
//!
//! It does not renumber behind the caller's back. [`order_by_degree`] hands
//! back a numbering and [`renumber`] applies it, and the caller keeps it,
//! because the mapping from a caller's node id to a dense one is the node
//! table's and there is no way to read the encoded graph without it.
//!
//! It does not hold the edge slots. A traversal wants to know where it can go
//! next and nothing else, and the payload for that answer is what is packed
//! here. Edge records are found by their own ids, which is the next piece.

use crate::{Adjacency, Dir};

/// Nodes to a group.
///
/// Five hundred and twelve, which is where the two costs cross. The offset
/// table costs the log of the group's stream length per node, so halving the
/// group saves about a bit a node; the group record costs sixteen bytes however
/// many nodes are in it, so halving the group costs about one and a half bits a
/// node. Anywhere from 256 to 1024 is within a few hundredths of a bit an edge
/// of the best, and 512 is the middle of that.
pub const GROUP: u32 = 512;

/// Gaps to a block, each block with its own width.
pub const BLOCK: usize = 32;

/// One group's worth of what has to be known before its stream can be read.
///
/// Sixteen bytes, one per five hundred and twelve nodes, so a quarter of a bit
/// a node and not worth thinking about again.
#[derive(Debug, Clone, Copy, Default)]
struct Group {
    /// Bit offset of this group's node offset table in the word array.
    at: u64,
    /// Width of one entry in the node offset table.
    ow: u8,
    /// Width of a degree.
    dw: u8,
    /// The smallest neighbour id anything in this group points at. Every first
    /// neighbour is written against this, which is most of what makes a
    /// clustered graph cheaper than a random one.
    ///
    /// Writing it against the node's own id instead, which under a community
    /// numbering is the better guess, was measured and is worse: the width is a
    /// group wide maximum, one node in five hundred and twelve points a long
    /// way off, and the sign bit is paid by all of them. It cost 0.11 bits an
    /// edge on web-Google under every ordering.
    base: u32,
    /// Width of a first neighbour, once `base` is taken off it.
    nw: u8,
}

/// A read only adjacency, node grouped and bit packed.
///
/// Built from an edge list over dense `u32` ids, or from the hot form through
/// an id mapping. See the module documentation for the layout and for what it
/// costs.
///
/// ```
/// use yo_graph::Csr;
///
/// let mut edges = vec![(0u32, 3u32), (0, 1), (2, 0), (0, 9)];
/// let cold = Csr::build(10, &mut edges);
///
/// assert_eq!(cold.degree(0), 3);
/// assert_eq!(cold.neighbours(0), vec![1, 3, 9]);
/// assert_eq!(cold.neighbours(1), Vec::<u32>::new());
/// ```
#[derive(Debug, Default)]
pub struct Csr {
    nodes: u32,
    edges: u64,
    groups: Vec<Group>,
    words: Vec<u64>,
    cost: Cost,
}

/// Where the bits went, in bits.
///
/// A compressed structure that cannot say which part of itself is expensive is
/// very hard to improve, and the answer moves a lot between graphs: on a graph
/// with an average degree of sixteen the per node fields are a fifth of the
/// total, and on one with an average degree of three they are most of it.
/// [`Csr::cost`] returns this and the `compress` example prints it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cost {
    /// The per node bit offset tables.
    pub offsets: u64,
    /// One degree per node, including the nodes that have no edges.
    pub degrees: u64,
    /// One first neighbour per node that has any.
    pub firsts: u64,
    /// The header in front of every block of gaps: the width, whether anything
    /// did not fit it, and how much of it did not.
    pub widths: u64,
    /// The gaps, which is the only part that is really the graph.
    pub gaps: u64,
    /// The gaps that did not fit their block's width, written again at the end
    /// of the block with the position they belong at.
    pub patches: u64,
    /// The fixed group records, which are not in the bit stream at all.
    pub groups: u64,
    /// Whatever rounding the stream up to whole words left over.
    pub slack: u64,
}

impl Cost {
    /// Everything, which is [`Csr::bytes`] in bits and to the bit.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.offsets
            + self.degrees
            + self.firsts
            + self.widths
            + self.gaps
            + self.patches
            + self.groups
            + self.slack
    }
}

impl Csr {
    /// Encode an edge list.
    ///
    /// The list is sorted in place, which is the only reason it is taken by
    /// mutable reference. Every id has to be under `nodes`. Parallel edges are
    /// kept rather than merged, because whether two links between the same pair
    /// are one edge or two is the caller's question and the answer costs a zero
    /// gap either way.
    #[must_use]
    pub fn build(nodes: u32, edges: &mut [(u32, u32)]) -> Csr {
        assert!(
            edges.iter().all(|(s, d)| *s < nodes && *d < nodes),
            "an edge names a node outside the graph"
        );
        edges.sort_unstable();
        Csr::encode(nodes, edges)
    }

    /// Encode one label and direction of a hot plane.
    ///
    /// `id` maps the caller's node ids onto the dense `u32` ids the cold form
    /// is over. That mapping is the node table's job and the node table does
    /// not exist yet, so for now it is the caller's, which also means a caller
    /// whose ids are already dense can pass a cast and pay nothing.
    #[must_use]
    pub fn from_hot(
        hot: &Adjacency,
        label: u32,
        dir: Dir,
        nodes: u32,
        id: impl Fn(u64) -> u32,
    ) -> Csr {
        let mut edges = Vec::with_capacity(hot.edges());
        hot.for_each_run(label, dir, |node, ns, _| {
            let src = id(node);
            edges.extend(ns.iter().map(|n| (src, id(*n))));
        });
        Csr::build(nodes, &mut edges)
    }

    /// How many node ids this was built over, whether or not they have edges.
    #[must_use]
    pub fn nodes(&self) -> u32 {
        self.nodes
    }

    /// How many edges are packed in here.
    #[must_use]
    pub fn edges(&self) -> u64 {
        self.edges
    }

    /// Whether there is nothing here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges == 0
    }

    /// The degree of `node`, without decoding its run.
    ///
    /// Two dependent loads, the offset and then the degree that starts the run,
    /// and neither of them touches a gap.
    #[must_use]
    pub fn degree(&self, node: u32) -> u32 {
        let Some((g, i, count)) = self.locate(node) else {
            return 0;
        };
        let s = self.groups[g];
        let run = self.run_at(s, i, count);
        read(&self.words, run, s.dw.into()) as u32
    }

    /// Decode the neighbours of `node` into `out`, ascending, replacing
    /// whatever was in it.
    ///
    /// The buffer is the point: a walk decodes run after run and there is no
    /// reason for any of them but the first to allocate.
    pub fn neighbours_into(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        let Some((g, i, count)) = self.locate(node) else {
            return;
        };
        let s = self.groups[g];
        let mut at = self.run_at(s, i, count);
        let deg = read(&self.words, at, s.dw.into()) as usize;
        at += u64::from(s.dw);
        if deg == 0 {
            return;
        }
        out.reserve(deg);
        let mut cur = s.base + read(&self.words, at, s.nw.into()) as u32;
        at += u64::from(s.nw);
        out.push(cur);
        let mut left = deg - 1;
        // The block is read into here first, because a gap that was patched is
        // only whole once its patch has been put back and the running sum cannot
        // start until it is.
        let mut block = [0u32; BLOCK];
        while left > 0 {
            let n = left.min(BLOCK);
            let w = read(&self.words, at, 6) as u32;
            at += 6;
            let patched = read(&self.words, at, 1) == 1;
            at += 1;
            let (mut x, mut ew) = (0u64, 0u32);
            if patched {
                x = read(&self.words, at, POS) + 1;
                at += u64::from(POS);
                ew = read(&self.words, at, 6) as u32;
                at += 6;
            }
            for slot in &mut block[..n] {
                *slot = read(&self.words, at, w) as u32;
                at += u64::from(w);
            }
            for _ in 0..x {
                let pos = read(&self.words, at, POS) as usize;
                at += u64::from(POS);
                let high = read(&self.words, at, ew);
                at += u64::from(ew);
                block[pos] |= (high << w) as u32;
            }
            for gap in &block[..n] {
                cur += gap;
                out.push(cur);
            }
            left -= n;
        }
    }

    /// The neighbours of `node`, ascending, in a fresh vector.
    ///
    /// The convenient one. A traversal should use
    /// [`neighbours_into`](Csr::neighbours_into) and keep its buffer.
    #[must_use]
    pub fn neighbours(&self, node: u32) -> Vec<u32> {
        let mut out = Vec::new();
        self.neighbours_into(node, &mut out);
        out
    }

    /// Ask the cache for the word a node's offset will be found in.
    ///
    /// Same reason as the hot form's: a frontier is known before any of it is
    /// read, and the loads that decode it are dependent, so the only way to
    /// make them cheap is to stop them being serial.
    pub fn prefetch(&self, node: u32) {
        let Some((g, i, _)) = self.locate(node) else {
            return;
        };
        let s = self.groups[g];
        let bit = s.at + i * u64::from(s.ow);
        yo_common::prefetch(&self.words[(bit / 64) as usize]);
    }

    /// Resident bytes, everything included.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.words.capacity() * size_of::<u64>() + self.groups.capacity() * size_of::<Group>()
    }

    /// Where the bits went. See [`Cost`].
    #[must_use]
    pub fn cost(&self) -> Cost {
        self.cost
    }

    /// [`bytes`](Csr::bytes) said the way the target in `11` is written, and
    /// zero for a graph with no edges.
    #[must_use]
    pub fn bits_per_edge(&self) -> f64 {
        if self.edges == 0 {
            return 0.0;
        }
        self.bytes() as f64 * 8.0 / self.edges as f64
    }

    /// The group a node is in, its index within it, and how many nodes that
    /// group holds, which is fewer than [`GROUP`] for the last one.
    #[inline]
    fn locate(&self, node: u32) -> Option<(usize, u64, u64)> {
        if node >= self.nodes {
            return None;
        }
        let g = node / GROUP;
        let lo = g * GROUP;
        Some((
            g as usize,
            u64::from(node - lo),
            u64::from((lo + GROUP).min(self.nodes) - lo),
        ))
    }

    /// Where a node's run starts, which is one read of the offset table.
    #[inline]
    fn run_at(&self, s: Group, i: u64, count: u64) -> u64 {
        let table = s.at + i * u64::from(s.ow);
        s.at + count * u64::from(s.ow) + read(&self.words, table, s.ow.into())
    }

    /// The whole encoder, over an edge list already sorted by source and then
    /// by destination.
    fn encode(nodes: u32, edges: &[(u32, u32)]) -> Csr {
        let mut w = Writer::default();
        let mut groups = Vec::with_capacity(nodes.div_ceil(GROUP) as usize);
        let mut runs: Vec<(usize, usize)> = Vec::with_capacity(GROUP as usize);
        let mut offs: Vec<u64> = Vec::with_capacity(GROUP as usize);
        let mut gaps: Vec<u32> = Vec::new();
        let mut e = 0usize;
        let mut cost = Cost::default();

        for lo in (0..nodes).step_by(GROUP as usize) {
            let hi = (lo + GROUP).min(nodes);
            let count = (hi - lo) as usize;

            // Cut the group's edges into one run per node and learn the three
            // widths the group needs before anything is written.
            runs.clear();
            let (mut base, mut top, mut maxdeg) = (u32::MAX, 0u32, 0u32);
            for node in lo..hi {
                let s = e;
                while e < edges.len() && edges[e].0 == node {
                    e += 1;
                }
                runs.push((s, e));
                maxdeg = maxdeg.max((e - s) as u32);
                if e > s {
                    base = base.min(edges[s].1);
                    top = top.max(edges[e - 1].1);
                }
            }
            let base = if base == u32::MAX { 0 } else { base };
            let dw = width(u64::from(maxdeg));
            let nw = width(u64::from(top.saturating_sub(base)));

            // Then how long every run is, because the offset table is written
            // before the runs are and its own width comes from their total.
            offs.clear();
            let mut total = 0u64;
            for (s, t) in &runs {
                offs.push(total);
                gaps_of(&edges[*s..*t], &mut gaps);
                total += run_bits(&gaps, *t > *s, dw, nw);
            }
            let ow = width(total);

            let at = w.bits();
            w.skip(count as u64 * u64::from(ow));
            cost.offsets += count as u64 * u64::from(ow);
            for (i, (s, t)) in runs.iter().enumerate() {
                w.put_at(at + i as u64 * u64::from(ow), offs[i], ow);
                let run = &edges[*s..*t];
                w.put(run.len() as u64, dw);
                cost.degrees += u64::from(dw);
                if run.is_empty() {
                    continue;
                }
                w.put(u64::from(run[0].1 - base), nw);
                cost.firsts += u64::from(nw);
                gaps_of(run, &mut gaps);
                for block in gaps.chunks(BLOCK) {
                    let p = Plan::best(block);
                    w.put(u64::from(p.w), 6);
                    w.put(u64::from(p.x != 0), 1);
                    cost.widths += 7;
                    if p.x != 0 {
                        w.put(u64::from(p.x - 1), POS);
                        w.put(u64::from(p.ew), 6);
                        cost.widths += 11;
                    }
                    cost.gaps += block.len() as u64 * u64::from(p.w);
                    for gap in block {
                        w.put(u64::from(*gap) & mask(p.w), p.w);
                    }
                    // The part of a gap that did not fit, at its position in the
                    // block, which the reader puts back on top of what it read.
                    for (at, gap) in block.iter().enumerate() {
                        if u64::from(*gap) >> p.w != 0 {
                            w.put(at as u64, POS);
                            w.put(u64::from(*gap) >> p.w, p.ew);
                            cost.patches += u64::from(POS + p.ew);
                        }
                    }
                }
            }
            groups.push(Group {
                at,
                base,
                ow: ow as u8,
                dw: dw as u8,
                nw: nw as u8,
            });
        }

        // One spare word, so a field that ends flush against the last one can
        // still be read by the two word path without a bounds check for it.
        w.words.push(0);
        w.words.shrink_to_fit();
        groups.shrink_to_fit();
        // The spare word and the rounding, so the total is the resident size to
        // the bit rather than to the field. Taken before the group records go
        // in, because those are not in the stream.
        cost.slack = w.words.capacity() as u64 * 64 - cost.total();
        cost.groups = groups.capacity() as u64 * size_of::<Group>() as u64 * 8;
        Csr {
            nodes,
            edges: edges.len() as u64,
            groups,
            words: w.words,
            cost,
        }
    }
}

/// The distances between one run's neighbours, into a buffer the encoder keeps
/// across runs so a hub does not allocate.
fn gaps_of(run: &[(u32, u32)], into: &mut Vec<u32>) {
    into.clear();
    into.extend(run.windows(2).map(|p| p[1].1 - p[0].1));
}

/// A node numbering that makes the graph smaller, busiest node first.
///
/// Returns the new id of every old id, so `out[old]` is `new`. Apply it with
/// [`renumber`] and encode the result.
///
/// This is the cheapest ordering there is, one count and one sort, and on a
/// power law graph it is worth about a fifth of the whole size. The reason is
/// that in a graph with hubs, almost every neighbour list contains some of the
/// hubs, and giving the hubs the smallest ids puts that shared part of every
/// list at the front where the gaps between its members are single digits. On
/// the R-MAT graph in the test below it takes 11.98 bits an edge to 9.38.
///
/// The control matters as much as the result. On a uniformly random graph the
/// same pass changes nothing at all, 15.96 bits an edge before and after, which
/// is what says it is exploiting structure rather than being an artefact of the
/// encoder. Breadth first from the busiest node, taking each frontier in degree
/// order, was measured against this and came out slightly worse at 10.15, so it
/// is not here.
///
/// This is the cheap ordering and it is not the good one. On a real graph the
/// numbering that pays is [`bisect::order`](crate::bisect::order), which finds
/// communities rather than hubs and is minutes rather than one sort. It beats
/// this by 5.2 bits an edge on web-Google. On R-MAT it does not beat this at
/// all, because R-MAT's structure is its hubs and this pass is already the right
/// answer for those.
#[must_use]
pub fn order_by_degree(nodes: u32, edges: &[(u32, u32)]) -> Vec<u32> {
    let mut deg = vec![0u32; nodes as usize];
    for (s, d) in edges {
        deg[*s as usize] += 1;
        deg[*d as usize] += 1;
    }
    let mut order: Vec<u32> = (0..nodes).collect();
    // Ties by the old id, so the same graph always numbers the same way.
    order.sort_unstable_by_key(|n| (core::cmp::Reverse(deg[*n as usize]), *n));
    let mut to = vec![0u32; nodes as usize];
    for (new, old) in order.iter().enumerate() {
        to[*old as usize] = new as u32;
    }
    to
}

/// Rewrite an edge list under a numbering, in place.
pub fn renumber(edges: &mut [(u32, u32)], to: &[u32]) {
    for e in edges {
        *e = (to[e.0 as usize], to[e.1 as usize]);
    }
}

/// How one block of gaps goes out: a width every gap in it is written at, and
/// the ones that did not fit written again at the end.
///
/// A block of thirty two gaps under a community numbering is mostly ones with
/// the occasional jump to another community in it, and a width that has to hold
/// the jump charges all thirty two of them for it. Leaving the jump behind as a
/// patch is what stops that, and it is only worth doing because the numbering
/// makes the distribution that shape. Under a degree ordering the same thing
/// costs 0.08 bits an edge, which is why it was not here before.
#[derive(Debug, Clone, Copy, Default)]
struct Plan {
    /// The width every gap in the block is written at.
    w: u32,
    /// How many of them did not fit.
    x: u32,
    /// The width the part that did not fit is written at, which is the widest
    /// gap in the block less `w`.
    ew: u32,
}

/// Bits of position in a patch, which is what indexes a block.
const POS: u32 = 5;

impl Plan {
    /// The cheapest way to write one block.
    ///
    /// Walking the candidate width down from the widest gap, `over` is how many
    /// gaps do not fit it. The widest gap is an exception at every width below
    /// its own, so the width the patches need is always the widest less the
    /// candidate, and the whole search is a walk over a histogram rather than a
    /// pass over the block per candidate.
    fn best(block: &[u32]) -> Plan {
        let mut hist = [0u32; 33];
        let mut top = 0u32;
        for g in block {
            let b = width(u64::from(*g));
            hist[b as usize] += 1;
            top = top.max(b);
        }
        let n = block.len() as u64;
        let mut best = Plan {
            w: top,
            x: 0,
            ew: 0,
        };
        let mut cost = 7 + n * u64::from(top);
        let mut over = 0u32;
        for w in (0..top).rev() {
            over += hist[w as usize + 1];
            let plan = Plan {
                w,
                x: over,
                ew: top - w,
            };
            let bits = plan.bits(n);
            if bits < cost {
                (best, cost) = (plan, bits);
            }
        }
        best
    }

    /// What this block costs, header and payload and patches.
    fn bits(&self, n: u64) -> u64 {
        let head = if self.x == 0 { 7 } else { 7 + 11 };
        head + n * u64::from(self.w) + u64::from(self.x) * u64::from(POS + self.ew)
    }
}

/// How many bits one run takes, which has to agree exactly with what the
/// encoder then writes.
fn run_bits(gaps: &[u32], any: bool, dw: u32, nw: u32) -> u64 {
    if !any {
        return u64::from(dw);
    }
    let mut bits = u64::from(dw) + u64::from(nw);
    for block in gaps.chunks(BLOCK) {
        bits += Plan::best(block).bits(block.len() as u64);
    }
    bits
}

/// How many bits it takes to hold `v`, and none for zero.
#[inline]
fn width(v: u64) -> u32 {
    64 - v.leading_zeros()
}

#[inline]
fn mask(w: u32) -> u64 {
    if w == 64 { u64::MAX } else { (1u64 << w) - 1 }
}

/// `w` bits starting at bit `at`, out of at most two words.
///
/// A field only ever crosses a word boundary when it did not start on one, so
/// the shift in the second branch is never sixty four.
#[inline]
fn read(words: &[u64], at: u64, w: u32) -> u64 {
    if w == 0 {
        return 0;
    }
    let i = (at / 64) as usize;
    let off = (at % 64) as u32;
    let lo = words[i] >> off;
    let got = 64 - off;
    if got >= w {
        lo & mask(w)
    } else {
        (lo | (words[i + 1] << got)) & mask(w)
    }
}

/// A bit stream being written, forwards, with the ability to go back and fill
/// in a hole it left on purpose.
#[derive(Debug, Default)]
struct Writer {
    words: Vec<u64>,
    bits: u64,
}

impl Writer {
    #[inline]
    fn bits(&self) -> u64 {
        self.bits
    }

    /// Leave `n` bits alone, for something that is not known yet. They are
    /// zero, which is what [`Writer::put_at`] needs them to be.
    fn skip(&mut self, n: u64) {
        self.bits += n;
        self.room(self.bits);
    }

    fn put(&mut self, v: u64, w: u32) {
        self.put_at(self.bits, v, w);
        self.bits += u64::from(w);
    }

    /// Write into bits that are still zero, anywhere in the stream.
    fn put_at(&mut self, at: u64, v: u64, w: u32) {
        if w == 0 {
            return;
        }
        self.room(at + u64::from(w));
        let i = (at / 64) as usize;
        let off = (at % 64) as u32;
        let v = v & mask(w);
        debug_assert!(w == 64 || v >> w == 0, "a field wider than it was given");
        self.words[i] |= v << off;
        if off + w > 64 {
            self.words[i + 1] |= v >> (64 - off);
        }
    }

    fn room(&mut self, upto: u64) {
        let need = (upto as usize).div_ceil(64) + 1;
        if self.words.len() < need {
            self.words.resize(need, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// The reference every round trip test is checked against: what the edge
    /// list plainly says, sorted, one vector a node.
    fn reference(nodes: u32, edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); nodes as usize];
        for (s, d) in edges {
            out[*s as usize].push(*d);
        }
        for v in &mut out {
            v.sort_unstable();
        }
        out
    }

    fn agrees(nodes: u32, mut edges: Vec<(u32, u32)>) -> Csr {
        let want = reference(nodes, &edges);
        let cold = Csr::build(nodes, &mut edges);
        let mut got = Vec::new();
        for node in 0..nodes {
            cold.neighbours_into(node, &mut got);
            assert_eq!(got, want[node as usize], "node {node}");
            assert_eq!(
                cold.degree(node),
                want[node as usize].len() as u32,
                "degree of {node}"
            );
        }
        cold
    }

    /// A uniformly random graph, which is the case no encoder can win.
    fn uniform(nodes: u32, degree: u32, seed: u64) -> Vec<(u32, u32)> {
        let mut rng = Rng::new(seed);
        let mut edges = Vec::with_capacity((nodes as usize) * (degree as usize));
        for src in 0..nodes {
            for _ in 0..degree {
                edges.push((src, (rng.next_u64() % u64::from(nodes)) as u32));
            }
        }
        edges
    }

    /// R-MAT with the Graph500 probabilities, which is the standard synthetic
    /// stand in for a social graph and the case where community structure is
    /// what pays for the compression.
    fn rmat(scale: u32, degree: u32, seed: u64) -> Vec<(u32, u32)> {
        let nodes = 1u32 << scale;
        let mut rng = Rng::new(seed);
        let mut edges = Vec::with_capacity((nodes as usize) * (degree as usize));
        for _ in 0..(nodes as u64) * u64::from(degree) {
            let (mut r, mut c) = (0u32, 0u32);
            for level in 0..scale {
                let bit = 1u32 << (scale - 1 - level);
                let p = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                if p < 0.57 {
                } else if p < 0.76 {
                    c |= bit;
                } else if p < 0.95 {
                    r |= bit;
                } else {
                    r |= bit;
                    c |= bit;
                }
            }
            edges.push((r, c));
        }
        edges
    }

    #[test]
    fn a_field_comes_back_out_the_way_it_went_in() {
        let mut w = Writer::default();
        let mut rng = Rng::new(7);
        let mut wrote = Vec::new();
        for _ in 0..5000 {
            let bits = (rng.next_u64() % 33) as u32;
            let v = rng.next_u64() & mask(bits);
            wrote.push((w.bits(), v, bits));
            w.put(v, bits);
        }
        w.words.push(0);
        for (at, v, bits) in wrote {
            assert_eq!(read(&w.words, at, bits), v, "at {at} wide {bits}");
        }
    }

    #[test]
    fn a_hole_left_on_purpose_can_be_filled_in_later() {
        let mut w = Writer::default();
        let at = w.bits();
        w.skip(40);
        w.put(0xabcd, 32);
        w.put_at(at, 0x9f_ffff_ffff, 40);
        w.words.push(0);
        assert_eq!(read(&w.words, at, 40), 0x9f_ffff_ffff);
        assert_eq!(read(&w.words, at + 40, 32), 0xabcd);
    }

    #[test]
    fn the_bits_add_up_to_the_bytes() {
        let mut edges = rmat(12, 8, 0xadd);
        let cold = Csr::build(1 << 12, &mut edges);
        let c = cold.cost();
        assert_eq!(c.total(), cold.bytes() as u64 * 8, "{c:?}");
        assert!(
            c.gaps > c.offsets,
            "the gaps should be the biggest part here"
        );
    }

    #[test]
    fn an_empty_graph_is_a_graph() {
        let cold = Csr::build(0, &mut []);
        assert!(cold.is_empty());
        assert_eq!(cold.degree(0), 0);
        assert_eq!(cold.neighbours(0), Vec::<u32>::new());
        assert_eq!(cold.bits_per_edge(), 0.0);
    }

    #[test]
    fn nodes_with_no_edges_are_still_nodes() {
        let cold = agrees(1000, vec![(500, 1), (500, 2)]);
        assert_eq!(cold.nodes(), 1000);
        assert_eq!(cold.edges(), 2);
        assert_eq!(
            cold.degree(1000),
            0,
            "and past the end is nothing rather than a panic"
        );
    }

    #[test]
    fn a_run_comes_back_ascending_however_it_went_in() {
        agrees(64, vec![(3, 40), (3, 1), (3, 63), (3, 0), (3, 17)]);
    }

    #[test]
    fn parallel_edges_survive_as_the_zero_gaps_they_are() {
        let cold = agrees(16, vec![(1, 2), (1, 2), (1, 2), (1, 9)]);
        assert_eq!(cold.degree(1), 4);
        assert_eq!(cold.neighbours(1), vec![2, 2, 2, 9]);
    }

    #[test]
    fn a_self_loop_is_an_edge_like_any_other() {
        agrees(8, vec![(4, 4), (4, 0)]);
    }

    #[test]
    fn the_last_group_can_be_a_partial_one() {
        // Deliberately not a multiple of the group size, and with edges in the
        // stub group, because an off by one there reads another group's table.
        let nodes = GROUP * 2 + 5;
        let mut edges = Vec::new();
        for src in 0..nodes {
            edges.push((src, (src * 7) % nodes));
        }
        agrees(nodes, edges);
    }

    #[test]
    fn a_hub_spans_as_many_blocks_as_it_needs() {
        // Two hundred thousand edges out of one node is over six thousand
        // blocks, and one neighbour placed far away so the run is not uniform.
        let mut edges: Vec<(u32, u32)> = (0..200_000u32).map(|i| (1, i)).collect();
        edges.push((1, 999_999));
        let cold = agrees(1_000_000, edges);
        assert_eq!(cold.degree(1), 200_001);
    }

    #[test]
    fn a_block_of_one_enormous_gap_does_not_price_the_rest() {
        // The whole reason the width is per block. Fifty thousand edges one
        // apart and a single jump across the graph. Per run widths would charge
        // all fifty thousand of them twenty bits; per block widths charge
        // thirty two of them.
        let mut edges: Vec<(u32, u32)> = (0..50_000u32).map(|i| (7, i)).collect();
        edges.push((7, 99_999));
        let cold = Csr::build(100_000, &mut edges.clone());
        agrees(100_000, edges);
        assert!(
            cold.bits_per_edge() < 4.0,
            "one far neighbour priced the whole run at {:.2} bits an edge",
            cold.bits_per_edge()
        );
    }

    #[test]
    fn the_cold_form_agrees_with_a_graph_someone_made_up() {
        let mut rng = Rng::new(0x51de);
        let nodes = 5000u32;
        let mut edges = Vec::new();
        for _ in 0..60_000 {
            // A degree distribution with a tail, so groups differ in every
            // width they choose and the partial and empty runs are both hit.
            let src = if rng.next_u64().is_multiple_of(10) {
                (rng.next_u64() % 20) as u32
            } else {
                (rng.next_u64() % u64::from(nodes)) as u32
            };
            edges.push((src, (rng.next_u64() % u64::from(nodes)) as u32));
        }
        agrees(nodes, edges);
    }

    #[test]
    fn promotion_reads_what_the_hot_plane_holds() {
        const FOLLOWS: u32 = 1;
        const BLOCKS: u32 = 2;
        let mut hot = Adjacency::new();
        let mut rng = Rng::new(0x40ce);
        let mut want: Vec<Vec<u32>> = vec![Vec::new(); 4000];
        for _ in 0..40_000 {
            let (s, d) = (rng.next_u64() % 4000, rng.next_u64() % 4000);
            hot.link(s, d, FOLLOWS, 0);
            want[s as usize].push(d as u32);
        }
        // Another label, which promotion has to leave behind entirely.
        for _ in 0..1000 {
            hot.link(rng.next_u64() % 4000, rng.next_u64() % 4000, BLOCKS, 0);
        }
        for v in &mut want {
            v.sort_unstable();
        }

        let cold = Csr::from_hot(&hot, FOLLOWS, Dir::Out, 4000, |n| n as u32);
        assert_eq!(cold.edges(), 40_000);
        let mut got = Vec::new();
        for node in 0..4000u32 {
            cold.neighbours_into(node, &mut got);
            assert_eq!(got, want[node as usize], "node {node}");
        }

        // And the transpose, which the hot plane indexes and which has to come
        // out as the mirror of what went in.
        let mut mirror: Vec<Vec<u32>> = vec![Vec::new(); 4000];
        for (s, ds) in want.iter().enumerate() {
            for d in ds {
                mirror[*d as usize].push(s as u32);
            }
        }
        for v in &mut mirror {
            v.sort_unstable();
        }
        let back = Csr::from_hot(&hot, FOLLOWS, Dir::In, 4000, |n| n as u32);
        for node in 0..4000u32 {
            back.neighbours_into(node, &mut got);
            assert_eq!(got, mirror[node as usize], "incoming to {node}");
        }
    }

    /// The number the target in `11` is about, on the two graphs that bracket
    /// it: one where nothing can help and one shaped like the graphs the target
    /// was written for.
    #[test]
    fn what_a_random_graph_costs_and_what_a_real_one_saves() {
        let nodes = 1u32 << 16;
        let degree = 16u32;

        let mut random = uniform(nodes, degree, 0xbeef);
        let random = Csr::build(nodes, &mut random);

        // The floor for a uniformly random graph, log2(n * n / m) + 1.44 bits
        // an edge, which here is about 13.4. Nothing beats it, so the only
        // question is how close the encoder gets.
        let floor =
            ((f64::from(nodes) * f64::from(nodes)) / f64::from(nodes * degree)).log2() + 1.44;
        let got = random.bits_per_edge();
        assert!(
            got > floor - 0.5,
            "random graph at {got:.2} bits an edge is under its {floor:.2} bit floor, so something is not being counted"
        );
        assert!(
            got < floor * 1.35,
            "random graph at {got:.2} bits an edge against a floor of {floor:.2}, so the encoder is wasting a third of itself"
        );

        // The same shape of graph with hubs in it. Everything below the uniform
        // number is the structure rather than the encoder.
        let mut social = rmat(16, degree, 0xf00d);
        let flat = Csr::build(nodes, &mut social.clone()).bits_per_edge();
        assert!(
            flat < got - 2.5,
            "R-MAT at {flat:.2} bits an edge against uniform at {got:.2}, so having hubs is buying nothing"
        );

        // And the ordering pass on top, which is the last cheap lever.
        let to = order_by_degree(nodes, &social);
        renumber(&mut social, &to);
        let ordered = Csr::build(nodes, &mut social).bits_per_edge();
        assert!(
            ordered < flat - 2.0,
            "degree ordering took R-MAT from {flat:.2} to {ordered:.2} bits an edge, which is not the fifth it was measured at"
        );
        assert!(
            ordered < 10.5,
            "R-MAT degree ordered at {ordered:.2} bits an edge, against the 9.89 this was measured at"
        );
    }

    /// The control on the ordering pass. It has to be worth nothing at all on a
    /// graph with no structure in it, because if it moves this number then it
    /// is not doing what it says it is doing.
    #[test]
    fn ordering_a_graph_with_no_structure_saves_nothing() {
        let nodes = 1u32 << 16;
        let mut edges = uniform(nodes, 16, 0xbeef);
        let before = Csr::build(nodes, &mut edges.clone()).bits_per_edge();
        let to = order_by_degree(nodes, &edges);
        renumber(&mut edges, &to);
        let after = Csr::build(nodes, &mut edges).bits_per_edge();
        assert!(
            (after - before).abs() < 0.1,
            "degree ordering moved a uniform graph from {before:.2} to {after:.2} bits an edge"
        );
    }

    #[test]
    fn a_numbering_is_a_permutation_and_the_graph_survives_it() {
        let mut edges = vec![(0u32, 1u32), (0, 2), (0, 3), (5, 0), (5, 1), (9, 0)];
        let to = order_by_degree(10, &edges);
        let mut seen = to.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<u32>>(), "not a permutation");
        assert_eq!(to[0], 0, "the busiest node did not get the smallest id");

        // Every edge is still the same edge, just under other names.
        let before: Vec<(u32, u32)> = edges.clone();
        renumber(&mut edges, &to);
        let mapped: Vec<(u32, u32)> = before
            .iter()
            .map(|(s, d)| (to[*s as usize], to[*d as usize]))
            .collect();
        assert_eq!(edges, mapped);

        let cold = Csr::build(10, &mut edges);
        assert_eq!(cold.edges(), 6);
        assert_eq!(cold.degree(to[0] as u32), 3);
        assert_eq!(cold.degree(to[5] as u32), 2);
    }

    /// The cold form against the hot one, which is the whole reason it exists.
    #[test]
    fn the_cold_form_is_an_order_of_magnitude_under_the_hot_one() {
        let nodes = 1u32 << 16;
        let mut edges = rmat(16, 16, 0x0117);
        let mut hot = Adjacency::out_only();
        for (s, d) in &edges {
            hot.link(u64::from(*s), u64::from(*d), 1, 0);
        }
        hot.compact();
        let cold = Csr::build(nodes, &mut edges);

        let ratio = hot.bytes() as f64 / cold.bytes() as f64;
        assert!(
            ratio > 8.0,
            "the cold form is only {ratio:.1} times smaller than the hot one, at {:.2} bits an edge against {:.1} bytes",
            cold.bits_per_edge(),
            hot.bytes() as f64 / hot.edges() as f64
        );
    }
}
