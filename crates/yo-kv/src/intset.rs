//! A set of integers as sorted packed arrays, which is Redis's intset in runs.
//!
//! A set whose members all parse as integers is held as the integers themselves,
//! sorted, in the narrowest width that covers the widest of them, with no hash
//! table and no per member allocation anywhere. One run looks exactly like a
//! Redis intset, because it is one:
//!
//! ```text
//! +----------+----------+---------+---------+-----+
//! | u32 width| u32 count| member 0| member 1| ... |
//! +----------+----------+---------+---------+-----+
//!   2, 4 or 8   how many   sorted ascending, width bytes each
//! ```
//!
//! Eight bytes of header and then nothing but members. At two byte width that is
//! two bytes an element with no overhead at all, which is the number G8 asks for
//! from a set of integers, and it is why this exists as a third representation
//! rather than everything small going in a listpack.
//!
//! Two byte width is not only for sets of small numbers. Past one run a run
//! stores its members as distances from a base of its own rather than as
//! themselves, so a set of billions is still two bytes a member. That is the
//! frame of reference, and the `Run` type is where it is explained.
//!
//! Measured, on a set of five hundred and twelve small integers, that is 2.0
//! bytes a member against the listpack's 3.0 and the element table's 24.0. The
//! header is the only thing between it and exactly two, and it is amortised away
//! by about sixty members.
//!
//! Both header fields are little endian whatever the machine is, because Redis
//! writes them that way: `intrev32ifbe` is a no-op on a little endian host and a
//! byte swap on a big endian one, so the bytes on the wire and in the file are
//! little endian from either. Getting that backwards would produce a file a real
//! server cannot read, on the one class of machine nobody tests on.
//!
//! # Why there is more than one array
//!
//! Redis gives up on the intset at five hundred and twelve members and rehashes
//! the set into a dictionary, which measured here at 24.60 to 30.92 bytes a
//! member against the intset's 4.00. That is a twelvefold jump in memory for a
//! set that got one member bigger, and it is worth asking what forced it.
//!
//! What forced it is that Redis has one array. An insert into the middle of one
//! sorted array memmoves the tail, so a million member set moves half a megabyte
//! per `SADD`, and no ceiling on the memory saves you from that. The conversion
//! is a fix for the memmove and the memory is what it costs.
//!
//! That is an argument against having one array. It is not an argument for
//! giving up two bytes a member. So the members here live in a list of runs,
//! each one a complete intset in Redis's own layout, holding disjoint ranges of
//! values in ascending order. A run is capped at `RUN_MAX` members, so the
//! memmove an insert pays is bounded by the run and not by the set: a thousand
//! bytes at two byte width, whether the set holds a thousand members or a
//! hundred million. Membership is a binary search over the run maxima to pick
//! the run and a binary search inside it, so it stays logarithmic in the whole
//! set with both searches in cache.
//!
//! `RUN_MAX` is five hundred and twelve on purpose. A set that a default
//! configured Redis would still call an intset is exactly one run here, so its
//! bytes are still one array and [`Intset::as_bytes`] still answers with a blob
//! a real server can read. The runs only appear past the point where Redis has
//! stopped having an intset at all.
//!
//! # Finding the member at a position
//!
//! `SRANDMEMBER` and `SPOP` need the member at an index, which one array answers
//! by multiplying and a list of runs does not. Adding up run lengths would be
//! linear in the number of runs, which is 3906 of them at a million members, and
//! `SRANDMEMBER key 100` would walk that four thousand entry array a hundred
//! times.
//!
//! So the run lengths are kept in a Fenwick tree, which answers "which run holds
//! position `k`, and how far into it" in a walk down the tree rather than a walk
//! along the runs. An add or a remove that leaves the run structure alone is one
//! more walk down the tree, and a split or a merge rebuilds it, which is linear
//! in the number of runs and happens once per couple of hundred writes. Measured
//! at 11.2 ns for the member at a position on a set of a million.
//!
//! # What the runs cost, measured
//!
//! The whole change is a memory argument, so the memory row is the one to read
//! first. Per member, on an all integer set, before the runs against after, from
//! `measure_bytes_per_member`:
//!
//! ```text
//!   members     one array          the runs     and the frame
//!       512     4.00  intset       2.09         2.11
//!     1,000    24.60  hashtable    2.22         2.25
//!   100,000    30.92  hashtable    3.57         2.26
//! 1,000,000    29.19  hashtable    4.11         2.21
//! ```
//!
//! The four bytes a member the larger sizes used to cost was not slack, it was
//! the width: values up to a million need four byte slots, so four bytes was the
//! floor and the overhead above it was eight hundredths of a byte. Getting under
//! it meant making the width smaller rather than the overhead, which is what the
//! frame of reference does, and it is the whole of the last column.
//!
//! The frame costs eight bytes a run, which is the base sitting in the `Run`
//! next to the buffer, and that is the two hundredths of a byte the first two
//! rows go backwards by. It is a bad trade on a set of small integers, which had
//! no width to save, and it pays for itself several hundred times over on
//! anything bigger.
//!
//! Filled in scattered order rather than ascending the answer is much the same,
//! 2.11, 2.24, 2.23 and 2.28, against 4.33 at a million before the frame. That
//! matters more than the ascending row does, because a run whose members arrive
//! out of order is the one that has to widen, and it is the shape a real
//! keyspace has.
//!
//! What it costs in time, from `intset_runs` in `benches/intset.rs`:
//!
//! ```text
//!                    4,096     100,000     1,000,000
//!   contains hit   11.4 ns     13.0 ns       14.9 ns
//!   contains miss   9.3 ns     10.3 ns       12.7 ns
//!   member at k     3.5 ns      7.2 ns       10.4 ns
//!   runs                15         390          3906
//! ```
//!
//! Against the same benchmark before the frame that is 3 to 5 percent slower at
//! 4,096 and 3 to 12 percent quicker at 100,000 and a million. The slower end is
//! the subtract the frame adds. The quicker end is the set being half the size
//! it was, which at 4,096 buys nothing because both fit in cache anyway, and at
//! a million buys more than the subtract costs.
//!
//! End to end through [`crate::Set`], a set of a million integers filled in
//! scattered order went from 30.71 bytes a member to 2.28, and `SADD` went from
//! 72.6 ns to 49.8. Membership went the other way, 13.6 ns to about 15, which is
//! the price of the two searches and is what the memory bought.
//!
//! That last number was 40.5 ns at first, which would not have been a trade
//! worth making, and the fix is the maxima array: picking the run by asking
//! each one for its own largest member is a pointer chase into a separate heap
//! buffer at every step of the binary search, and the same search over a
//! contiguous array of the maxima is not.
//!
//! # Why sorted, and what it costs
//!
//! Membership is a binary search, which is nine steps at the 512 member ceiling
//! against the element table's one probe. The reason to accept that is that 512
//! members is at most four kilobytes, so the search stays in cache and the steps
//! are not nine cache misses.
//!
//! That paragraph used to be an argument with no measurement behind it, which in
//! this project is a warning sign: L6 put a positional probe at 70 ns and it
//! measured 13, and K11's crossover does not exist. `benches/intset.rs` settled
//! it. Minimum per iteration on an M3 laptop, membership against a member that
//! is there, at the sizes either side of the ceiling:
//!
//! ```text
//!   members     intset     listpack     element table
//!         8     4.6 ns       6.2 ns            7.7 ns
//!        64     6.6 ns      29.5 ns           10.2 ns
//!       128     7.7 ns      60.7 ns           10.2 ns
//!       512    10.4 ns     239.3 ns            9.0 ns
//! ```
//!
//! So the search is affordable, and the number that makes the case is not the
//! one against the table. Doubling the set three times costs the intset about
//! 3 ns in total, which is what a search that stays in cache looks like. What
//! the intset is actually replacing below the ceiling is the listpack, and there
//! it is eight times quicker at 128 members and pulling away, because a listpack
//! walks and this does not.
//!
//! The crossover with the element table lands almost exactly on Redis's ceiling.
//! At 128 the intset wins by a quarter, at 512 the table wins by a seventh. That
//! is a better outcome than the argument deserved, and it was not predicted here:
//! the guess was that the search would be affordable, not that the constant Redis
//! picked in 2011 would sit on the crossover.
//!
//! Sorted also means an insert memmoves the tail, and that turns out not to
//! matter at these sizes. A scattered fill, where every add lands in the middle,
//! measured 6.47 ns a member at 128 against an ascending fill's 6.46, and the
//! two only separate at 512 where scattered costs 5.26 against 4.44. Four
//! kilobytes is not a memmove worth avoiding, which is the whole reason a run is
//! allowed to be that big. The reason the ascending case is still worth having,
//! and worth a test, is a shape argument and not a timing one: a fill in
//! ascending order hits the "greater than the last member" test in front of the
//! search, so it never searches and never moves anything, and
//! `an_ascending_fill_never_moves_anything` asserts that rather than timing it.
//!
//! # Widening
//!
//! Adding a member too wide for the current width rewrites every member of that
//! run into the new width. That happens at most twice in a run's life, 2 to 4 and
//! 4 to 8, and the new member is known to sit at one end before the rewrite
//! starts, because being too wide is exactly what it means to be outside the
//! range of everything already there. Negative goes to the front and positive to
//! the back.
//!
//! Width is per run and not per set, which is a small win Redis cannot have. A
//! set holding a million small integers and one huge one keeps every run but the
//! last at two byte width, where one array would have rewritten all million
//! members to eight.
//!
//! Removing never narrows the width back. Redis does not either, and a set that
//! narrowed on the way down would rewrite itself on every second operation for a
//! workload that adds and removes around a boundary.

/// Why an intset from somewhere else was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Malformed {
    /// Shorter than the eight byte header.
    Short,
    /// The width is not 2, 4 or 8.
    Width,
    /// The count and the width do not account for the bytes that arrived.
    Length,
    /// The members are not in ascending order, or one appears twice.
    Order,
}

/// The widths a member can be stored in, which are the widths of the three
/// signed integer types Redis uses and nothing else.
const W16: u32 = 2;
const W32: u32 = 4;
const W64: u32 = 8;

/// Width, then count.
const HEADER: usize = 8;

/// Members a run holds before it splits in two.
///
/// Five hundred and twelve, which is Redis's default `set-max-intset-entries`,
/// and matching it is deliberate rather than a coincidence. Every set a default
/// configured server would call an intset is one run here, so it is still one
/// blob in Redis's own layout, and the runs only start once Redis has given up
/// on the encoding entirely.
///
/// It is also about where the measurements put the ceiling on a memmove that is
/// still free. A scattered insert into a 512 member run costs 5.26 ns a member
/// against an ascending one's 4.44, so the tail move is under a nanosecond at
/// this size and does not need to be smaller.
const RUN_MAX: usize = 512;

/// Members a run falls to before it is folded into a neighbour.
///
/// A quarter of the ceiling. Runs that split leave two halves at half the
/// ceiling, so the gap between this and that is what stops a set sitting on a
/// boundary from splitting and merging the same run on alternate writes.
const RUN_MIN: usize = RUN_MAX / 4;

/// Members a run's buffer grows by when it fills.
///
/// A `Vec` grows by doubling, which is right for a buffer whose final size
/// nobody knows and wrong for one that is never allowed past [`RUN_MAX`]
/// members. A run one member over a power of two would hold twice the bytes it
/// needs, and the bytes are the entire point of this representation. Growing in
/// fixed steps leaves at most this many members of slack whatever the run's
/// size, which is 64 bytes at two byte width against a run of a few hundred.
const STEP: usize = 32;

/// A sorted packed set of integers, in one or more runs.
#[derive(Debug, Clone)]
pub struct Intset {
    /// The runs, in ascending order of the values they hold, with disjoint
    /// ranges. Never empty: a set with no members is one empty run, so that
    /// every lookup has a run to land in without a special case.
    runs: Vec<Run>,
    /// Members across every run.
    total: usize,
    /// The largest member of each run, so that picking the run is a search over
    /// one contiguous array.
    ///
    /// It is a copy of something the runs already know, and it earns its eight
    /// bytes a run several times over. Asking each run for its own largest
    /// member walks a binary search over a list of pointers into separate heap
    /// buffers, which is a cache miss a step and was measured at 40.5 ns for a
    /// membership test on a set of a million. The same search over this array
    /// touches a few kilobytes that stay in L2.
    ///
    /// An empty run holds [`i64::MAX`], so it sorts last and every value lands
    /// in it, which is what an empty set needs and is the only time a run is
    /// empty at all.
    maxima: Vec<i64>,
    /// Run lengths as a Fenwick tree, one indexed, so that the member at a
    /// position is found without adding up run lengths. See [`Intset::select`].
    fen: Vec<u32>,
}

impl Intset {
    /// An empty set at the narrowest width.
    #[must_use]
    pub fn new() -> Intset {
        Intset {
            runs: vec![Run::new()],
            total: 0,
            maxima: vec![i64::MAX],
            fen: vec![0, 0],
        }
    }

    /// An empty set with room for `n` members at the narrowest width.
    ///
    /// Only a hint. A member that needs a wider slot still widens the run it
    /// lands in, and the reservation is then short, which costs one growth and
    /// no correctness.
    #[must_use]
    pub fn with_capacity(n: usize) -> Intset {
        let mut s = Intset::new();
        s.runs[0].reserve_members(n.min(RUN_MAX));
        if n > RUN_MAX {
            // A split leaves two runs of half the ceiling, so that is what the
            // expected count divides by rather than the ceiling itself.
            s.runs.reserve(n / (RUN_MAX / 2));
            s.maxima.reserve(n / (RUN_MAX / 2));
        }
        s
    }

    /// Read a blob written by us or by a real server.
    ///
    /// The order check is the one worth having. A truncated blob is caught by
    /// the length arithmetic, but a blob whose members are out of order reads
    /// as a perfectly valid set that silently answers no to members it holds,
    /// because every search here assumes the order.
    pub fn from_bytes(bytes: &[u8]) -> Result<Intset, Malformed> {
        let run = Run::from_bytes(bytes)?;
        let total = run.len();
        let mut s = Intset {
            runs: vec![run],
            total,
            maxima: Vec::new(),
            fen: Vec::new(),
        };
        s.maxima.push(top_of(&s.runs[0]));
        s.rebuild_ranks();
        Ok(s)
    }

    /// The blob, header included, ready to write to a file, when there is one.
    ///
    /// `None` once the set has split, because Redis's format is one array and
    /// carries nothing that could say otherwise. Nothing is lost by that: the
    /// split happens at `RUN_MAX` members, which is where a default configured
    /// server has already stopped storing the set as an intset, so a set that
    /// could have been written as one still is.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self.runs.as_slice() {
            // The base check is a backstop rather than a case that comes up on
            // the way here. A run only takes a base once the set has more than
            // one of them, and a set drained back down to one run gives its
            // frame up on the way, so a one run set with a base should not
            // exist. This is cheaper than being sure of that.
            [run] if run.base == 0 => Some(run.as_bytes()),
            _ => None,
        }
    }

    /// How many members.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.total
    }

    /// Whether there are none.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// How many runs the members are spread over, which is one until the set
    /// passes `RUN_MAX` members.
    #[inline]
    #[must_use]
    pub fn runs(&self) -> usize {
        self.runs.len()
    }

    /// Bytes a member occupies in the widest run, which is 2, 4 or 8.
    ///
    /// The widest and not one number for the set, because width is per run here.
    /// This is what a caller asking "how wide did this set have to get" means,
    /// and no search uses it.
    ///
    /// It is the width of the stored offset and not of the value, so a set of
    /// integers around a billion reports two once it has split into runs. That
    /// is the point of the frame of reference the runs are packed against.
    #[must_use]
    pub fn width(&self) -> usize {
        self.runs
            .iter()
            .map(Run::width)
            .max()
            .unwrap_or(W16 as usize)
    }

    /// The bytes the members occupy, which is what `MEMORY USAGE` counts.
    #[inline]
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.runs.iter().map(Run::byte_len).sum()
    }

    /// Bytes held, including whatever the vectors have reserved and not used.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let runs: usize = self.runs.iter().map(Run::memory_bytes).sum();
        runs + self.runs.capacity() * size_of::<Run>()
            + self.maxima.capacity() * size_of::<i64>()
            + self.fen.capacity() * size_of::<u32>()
    }

    /// The member at `index`, counting from the smallest.
    ///
    /// # Panics
    ///
    /// If `index` is not under [`Intset::len`]. Every caller here has already
    /// bounded it, and a draw for `SRANDMEMBER` bounds it by construction.
    #[inline]
    #[must_use]
    pub fn at(&self, index: usize) -> i64 {
        assert!(index < self.total, "index {index} is past the set");
        let (run, offset) = self.select(index);
        self.runs[run].at(offset)
    }

    /// The member at `index`, or `None` past the end.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<i64> {
        (index < self.total).then(|| self.at(index))
    }

    /// The smallest member, or `None` if there are none.
    #[inline]
    #[must_use]
    pub fn min(&self) -> Option<i64> {
        self.runs.first().and_then(Run::min)
    }

    /// The largest member, or `None` if there are none.
    #[inline]
    #[must_use]
    pub fn max(&self) -> Option<i64> {
        self.runs.last().and_then(Run::max)
    }

    /// Whether `v` is a member.
    #[inline]
    #[must_use]
    pub fn contains(&self, v: i64) -> bool {
        self.runs[self.run_for(v)].contains(v)
    }

    /// Every member, smallest first.
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.runs.iter().flat_map(Run::iter)
    }

    /// A cursor on the smallest member, for a merge. See [`Walk`].
    #[inline]
    #[must_use]
    pub fn walk(&self) -> Walk<'_> {
        Walk::new(self)
    }

    /// Add `v`. Answers whether it was not already there.
    pub fn add(&mut self, v: i64) -> bool {
        let i = self.run_for(v);
        // A one run set never moves its base, so its bytes stay a Redis intset
        // and `as_bytes` stays a borrow. See [`Run`].
        let rebase = self.runs.len() > 1;
        if !self.runs[i].add(v, rebase) {
            return false;
        }
        self.total += 1;
        if self.runs[i].len() > RUN_MAX {
            self.split(i);
        } else {
            self.maxima[i] = top_of(&self.runs[i]);
            self.bump(i, 1);
        }
        true
    }

    /// Remove `v`. Answers whether it was there.
    pub fn remove(&mut self, v: i64) -> bool {
        let i = self.run_for(v);
        if !self.runs[i].remove(v) {
            return false;
        }
        self.total -= 1;
        self.maxima[i] = top_of(&self.runs[i]);
        if self.runs.len() > 1 && self.runs[i].len() < RUN_MIN {
            self.shrink(i);
        } else {
            self.bump(i, -1);
        }
        true
    }

    /// Which run holds `v`, or would hold it.
    ///
    /// The runs cover disjoint ranges in ascending order, so the first one whose
    /// largest member is not under `v` is the only one that can hold it. A value
    /// above every run belongs at the end of the last one, which is what the
    /// clamp says, and it is the ascending fill: every add lands in the last run
    /// and appends inside it.
    #[inline]
    fn run_for(&self, v: i64) -> usize {
        let i = self.maxima.partition_point(|&m| m < v);
        i.min(self.runs.len() - 1)
    }

    /// Cut run `i` in half, because it has passed [`RUN_MAX`].
    ///
    /// The upper half moves into a new run built at whatever width its own
    /// members need, which is how a set of small integers with one huge member
    /// keeps most of its runs at two bytes.
    fn split(&mut self, i: usize) {
        let n = self.runs[i].len();
        let half = n / 2;
        let src = &self.runs[i];
        // The members are ascending, so the two ends bound the frame of
        // everything between them and there is nothing to scan.
        let (base, w) = frame(src.at(half), src.at(n - 1));
        let mut hi = Run::with_base(base, w, n - half);
        for k in half..n {
            hi.push_back(src.at(k));
        }
        self.runs[i].truncate(half);
        // Both halves cover a narrower range than the run they came out of, and
        // the upper one was built knowing that. This is where the lower one
        // finds out, and it is the whole reason a split is where the frames get
        // tight: an ascending fill splits every run exactly once.
        self.runs[i].rebase();
        // The lower half keeps the buffer the whole run had, which is twice
        // what it now holds, and an ascending fill splits every run exactly
        // once and then never touches the lower half again. Left alone that is
        // two bytes a member of pure slack on the commonest fill there is, so
        // the buffer is handed back here and the next insert into it reserves a
        // step like any other.
        self.runs[i].tighten();
        self.runs.insert(i + 1, hi);
        // The maxima are patched rather than recomputed. Recomputing them means
        // asking every run for its own last member, which is a pointer chase
        // per run into a separate heap buffer, and it measured at a sixth of
        // the cost of a whole scattered fill of a million members. Moving eight
        // bytes a run along an array is nothing next to that.
        self.maxima[i] = top_of(&self.runs[i]);
        let top = top_of(&self.runs[i + 1]);
        self.maxima.insert(i + 1, top);
        self.rebuild_ranks();
    }

    /// Fold run `i` into a neighbour, because it has fallen under [`RUN_MIN`].
    ///
    /// An empty run simply goes. Otherwise it merges with whichever neighbour
    /// the two of them fit inside one run, preferring the one on the left so
    /// that a set being drained from the front collapses rather than leaving a
    /// trail of short runs. Two neighbours that are both too full to take it is
    /// not a problem to solve: the run stays short and costs one entry in the
    /// tree.
    fn shrink(&mut self, i: usize) {
        if self.runs[i].is_empty() {
            self.runs.remove(i);
            self.maxima.remove(i);
            self.rebuild_ranks();
            return;
        }
        let fits = |a: usize, b: usize| self.runs[a].len() + self.runs[b].len() <= RUN_MAX;
        let (lo, hi) = if i > 0 && fits(i - 1, i) {
            (i - 1, i)
        } else if i + 1 < self.runs.len() && fits(i, i + 1) {
            (i, i + 1)
        } else {
            self.bump(i, -1);
            return;
        };
        let src = self.runs.remove(hi);
        self.maxima.remove(hi);
        self.runs[lo].append(&src);
        // A set drained back down to one run gives up its frame, so that it is
        // a Redis intset again and [`Intset::as_bytes`] is a borrow rather than
        // a rebuild. See [`Run`] for why one run never carries a base.
        if self.runs.len() == 1 {
            self.runs[0].unframe();
        }
        self.maxima[lo] = top_of(&self.runs[lo]);
        self.rebuild_ranks();
    }

    /// Which run holds position `k`, and how far into it.
    ///
    /// The standard Fenwick descent: walk the powers of two downward, taking a
    /// step whenever the members it covers are all still behind `k`. What is
    /// left over when the steps run out is the offset inside the run.
    fn select(&self, k: usize) -> (usize, usize) {
        let n = self.runs.len();
        let mut pos = 0usize;
        let mut rem = k;
        let mut step = 1usize << (usize::BITS - 1 - n.leading_zeros());
        while step > 0 {
            let next = pos + step;
            if next <= n {
                let covered = self.fen[next] as usize;
                if covered <= rem {
                    pos = next;
                    rem -= covered;
                }
            }
            step >>= 1;
        }
        (pos, rem)
    }

    /// Tell the tree that run `i` gained or lost one member.
    fn bump(&mut self, i: usize, delta: i32) {
        let n = self.runs.len();
        let mut at = i + 1;
        while at <= n {
            if delta > 0 {
                self.fen[at] += 1;
            } else {
                self.fen[at] -= 1;
            }
            at += at & at.wrapping_neg();
        }
    }

    /// Rebuild the tree from the run lengths.
    ///
    /// What a split or a merge needs, because both of them renumber every run
    /// after the one they touched and a Fenwick tree is not a thing you patch
    /// in the middle. It is linear in the number of runs, over one array that
    /// is read and written straight through, and it happens once per couple of
    /// hundred writes.
    fn rebuild_ranks(&mut self) {
        let n = self.runs.len();
        self.fen.clear();
        self.fen.resize(n + 1, 0);
        for i in 1..=n {
            let len = u32::try_from(self.runs[i - 1].len()).expect("a run is under RUN_MAX");
            self.fen[i] += len;
            let parent = i + (i & i.wrapping_neg());
            if parent <= n {
                let carry = self.fen[i];
                self.fen[parent] += carry;
            }
        }
    }
}

impl Default for Intset {
    fn default() -> Intset {
        Intset::new()
    }
}

/// A cursor over the members that only ever moves forward.
///
/// [`Intset::iter`] is enough to read a set out, and it is not enough to merge
/// two of them, because a merge needs to skip. Intersecting a set of ten with a
/// set of a million should touch ten members of the big one and not a million,
/// and that is [`Walk::seek`], which jumps to the first member at or past a
/// value instead of stepping to it.
///
/// Forward only, and that is the whole reason it is worth having. A cursor that
/// could go backwards would have to binary search the entire set on every seek.
/// This one searches from where it already is, so a merge that walks two sets in
/// lockstep pays one comparison a member in the common case and only searches
/// when it actually skipped something.
///
/// Stepping is a pointer step and nothing else, which is what makes a merge a
/// different order of cost from a probe. `setops.rs` explains what that buys and
/// has the numbers.
#[derive(Debug, Clone, Copy)]
pub struct Walk<'a> {
    set: &'a Intset,
    /// Which run. Equal to the run count once the cursor is past the end.
    run: usize,
    /// How far into that run. Always under the run's length except when the
    /// cursor is past the end, where the pair is `(runs.len(), 0)`.
    off: usize,
}

impl<'a> Walk<'a> {
    /// A cursor on the smallest member.
    fn new(set: &'a Intset) -> Walk<'a> {
        let mut w = Walk {
            set,
            run: 0,
            off: 0,
        };
        w.settle();
        w
    }

    /// The member the cursor is on, or `None` past the end.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<i64> {
        (self.run < self.set.runs.len()).then(|| self.set.runs[self.run].at(self.off))
    }

    /// Move to the next member.
    #[inline]
    pub fn bump(&mut self) {
        self.off += 1;
        self.settle();
    }

    /// Move to the first member that is not under `v`, without going backwards.
    ///
    /// A seek to a value the cursor is already at or past does nothing, which is
    /// what makes this safe to call in a loop that does not know whether it has
    /// moved.
    pub fn seek(&mut self, v: i64) {
        match self.peek() {
            Some(cur) if cur < v => {}
            // Already there, or there is nothing left to seek to.
            _ => return,
        }
        // The runs hold disjoint ranges in ascending order, so the run is the
        // first one at or after this one whose largest member is not under `v`.
        // Searching from `run + 1` rather than from the start is what keeps a
        // seek near the cursor cheap: a merge that steps through both sets
        // together never leaves its current run.
        if self.set.maxima[self.run] < v {
            let after = &self.set.maxima[self.run + 1..];
            let hop = after.partition_point(|&m| m < v);
            self.run += 1 + hop;
            if self.run >= self.set.runs.len() {
                self.run = self.set.runs.len();
                self.off = 0;
                return;
            }
            self.off = 0;
        }
        self.off = self.set.runs[self.run].lower_bound(v, self.off);
        self.settle();
    }

    /// Step off the end of a run onto the next one.
    ///
    /// A loop rather than a test because an empty set is one empty run, and that
    /// is the only time two runs in a row have nothing to land on.
    #[inline]
    fn settle(&mut self) {
        while self.run < self.set.runs.len() && self.off >= self.set.runs[self.run].len() {
            self.run += 1;
            self.off = 0;
        }
    }
}

/// Two sets are equal when they hold the same members.
///
/// Written out rather than derived, because where the run boundaries fell is an
/// artefact of the order the members arrived in and not something a caller has
/// any business seeing. A set filled ascending and the same set filled scattered
/// are the same set.
impl PartialEq for Intset {
    fn eq(&self, other: &Intset) -> bool {
        self.total == other.total && self.iter().eq(other.iter())
    }
}

impl Eq for Intset {}

/// One run: a complete intset in Redis's own layout, offset from a base.
///
/// The base is what takes a run of large integers down to two bytes a member.
/// A run holds at most [`RUN_MAX`] members out of a set that may hold millions,
/// so the values inside one run are close together whatever the set as a whole
/// spans: a million members scattered over sixteen million values leave every
/// run covering a few thousand of them. Stored as themselves those need four
/// bytes each, and stored as their distance from the middle of the run's own
/// range they need two.
///
/// The base is the middle of that range rather than the bottom of it, which is
/// worth a sentence because it is not obvious. The stored offsets are read back
/// through the same signed readers Redis uses, so a base at the bottom would
/// only ever use the positive half of the width and hold a span of thirty two
/// thousand at two bytes. Centred, the offsets run either side of zero and the
/// same two bytes hold a span of sixty five thousand. It also leaves room on
/// both sides for the members still to arrive rather than only above.
///
/// A base of zero is a run that is byte for byte a Redis intset, and a run only
/// takes a base once the set it belongs to has more than one of them. That is
/// deliberate: a set a default configured server would still call an intset
/// stays one array in Redis's own layout, so [`Intset::as_bytes`] is still a
/// borrow rather than a rebuild, and the frame only appears past the point
/// where Redis has stopped having an intset at all.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Run {
    /// The header and the members, in Redis's own layout, so that handing this
    /// to an RDB writer is a copy when the base is zero.
    bytes: Vec<u8>,
    /// What every stored member is measured from.
    base: i64,
}

impl Run {
    /// An empty run at the narrowest width.
    fn new() -> Run {
        Run::with_base(0, W16 as usize, 0)
    }

    /// An empty run against `base`, `w` bytes a member, with room for `n`.
    fn with_base(base: i64, w: usize, n: usize) -> Run {
        let mut bytes = Vec::with_capacity(HEADER + n * w);
        bytes.extend_from_slice(&(w as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        Run { bytes, base }
    }

    /// Read a blob written by us or by a real server. See [`Intset::from_bytes`].
    fn from_bytes(bytes: &[u8]) -> Result<Run, Malformed> {
        if bytes.len() < HEADER {
            return Err(Malformed::Short);
        }
        let width = u32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
        if width != W16 && width != W32 && width != W64 {
            return Err(Malformed::Width);
        }
        let count = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes")) as usize;
        let want = count
            .checked_mul(width as usize)
            .and_then(|n| n.checked_add(HEADER))
            .ok_or(Malformed::Length)?;
        if bytes.len() != want {
            return Err(Malformed::Length);
        }
        let s = Run {
            bytes: bytes.to_vec(),
            base: 0,
        };
        for i in 1..count {
            if s.at(i - 1) >= s.at(i) {
                return Err(Malformed::Order);
            }
        }
        Ok(s)
    }

    /// The blob, header included.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// How many members.
    #[inline]
    fn len(&self) -> usize {
        u32::from_le_bytes(self.bytes[4..8].try_into().expect("four bytes")) as usize
    }

    /// Whether there are none.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes a member occupies, which is 2, 4 or 8.
    #[inline]
    fn width(&self) -> usize {
        u32::from_le_bytes(self.bytes[0..4].try_into().expect("four bytes")) as usize
    }

    /// The blob's length.
    #[inline]
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Bytes held, including whatever the vector has reserved and not used.
    #[inline]
    fn memory_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    /// The member at `index`, counting from the smallest.
    #[inline]
    fn at(&self, index: usize) -> i64 {
        self.base + self.raw(index)
    }

    /// What is stored at `index`, which is the member less the base.
    #[inline]
    fn raw(&self, index: usize) -> i64 {
        self.raw_w(index, self.width())
    }

    /// [`Run::raw`] for a caller that already knows the width.
    ///
    /// The width lives in the buffer, so reading it is a load, and a binary
    /// search that reads it at every step reads the same four bytes nine times.
    /// Out here it is read once and the search compares stored offsets against
    /// a stored offset rather than adding the base back nine times.
    #[inline]
    fn raw_w(&self, index: usize, w: usize) -> i64 {
        let at = HEADER + index * w;
        let raw = &self.bytes[at..at + w];
        match w {
            2 => i64::from(i16::from_le_bytes(raw.try_into().expect("two bytes"))),
            4 => i64::from(i32::from_le_bytes(raw.try_into().expect("four bytes"))),
            _ => i64::from_le_bytes(raw.try_into().expect("eight bytes")),
        }
    }

    /// Whether `v` is inside the frame this run is packed against.
    ///
    /// A value outside it is not a member, because every member is inside it,
    /// and saying so costs a subtract and a compare instead of a search.
    #[inline]
    fn framed(&self, v: i64) -> bool {
        v.checked_sub(self.base)
            .is_some_and(|off| width_of(off) as usize <= self.width())
    }

    /// The smallest member, or `None` if there are none.
    #[inline]
    fn min(&self) -> Option<i64> {
        (!self.is_empty()).then(|| self.at(0))
    }

    /// The largest member, or `None` if there are none.
    #[inline]
    fn max(&self) -> Option<i64> {
        self.len().checked_sub(1).map(|last| self.at(last))
    }

    /// Whether `v` is a member.
    #[inline]
    fn contains(&self, v: i64) -> bool {
        self.framed(v) && self.search(v).is_ok()
    }

    /// Every member, smallest first.
    fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        (0..self.len()).map(|i| self.at(i))
    }

    /// Room for `n` more members without a growth.
    fn reserve_members(&mut self, n: usize) {
        self.bytes.reserve_exact(n * self.width());
    }

    /// Add `v`. Answers whether it was not already there.
    ///
    /// `rebase` is whether this run is allowed to move its base, which it is
    /// only once the set has more than one run. See [`Run`].
    fn add(&mut self, v: i64, rebase: bool) -> bool {
        if !self.framed(v) {
            self.refit_and_add(v, rebase);
            return true;
        }
        match self.search(v) {
            Ok(_) => false,
            Err(at) => {
                self.insert_at(at, v);
                true
            }
        }
    }

    /// Put `v` on the end, where it is already known to belong.
    ///
    /// Only used when one run is being built out of another, so the caller has
    /// the members in ascending order and the width is already wide enough.
    fn push_back(&mut self, v: i64) {
        let w = self.width();
        let at = self.bytes.len();
        self.grow_by(w);
        write_at(&mut self.bytes, at, w, v - self.base);
        self.set_len(self.len() + 1);
    }

    /// Put every member of `other` on the end, where they all belong.
    fn append(&mut self, other: &Run) {
        self.reserve_members(other.len());
        for v in other.iter() {
            // Through `add` and not `push_back`, because `other` may hold
            // members outside this run's frame and refitting is `add`'s job.
            // Every one of them is past the last member, so the range test in
            // front of the search answers and nothing moves.
            self.add(v, true);
        }
    }

    /// Drop everything from `n` onward.
    fn truncate(&mut self, n: usize) {
        self.bytes.truncate(HEADER + n * self.width());
        self.set_len(n);
    }

    /// Hand back whatever the buffer is holding and not using.
    fn tighten(&mut self) {
        self.bytes.shrink_to_fit();
    }

    /// Remove `v`. Answers whether it was there.
    fn remove(&mut self, v: i64) -> bool {
        if !self.framed(v) {
            return false;
        }
        let Ok(at) = self.search(v) else {
            return false;
        };
        let w = self.width();
        let from = HEADER + at * w;
        self.bytes.drain(from..from + w);
        self.set_len(self.len() - 1);
        true
    }

    /// The first member at or past `v`, searching only from `from`.
    ///
    /// [`Walk::seek`]'s inner half, and the reason it takes a lower bound rather
    /// than reusing [`Run::search`]: a cursor never goes backwards, so
    /// everything before where it already is has been ruled out and searching it
    /// again is work with a known answer. On a merge that steps through two sets
    /// together the range is one or two members wide.
    fn lower_bound(&self, v: i64, from: usize) -> usize {
        let w = self.width();
        let off = self.offset_of(v);
        let (mut lo, mut hi) = (from, self.len());
        while lo < hi {
            let mid = lo.midpoint(hi);
            if self.raw_w(mid, w) < off {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `v` as this run would store it, saturating rather than wrapping.
    ///
    /// A search compares stored offsets against a stored offset, so the base
    /// comes off the value it is looking for once instead of going back onto
    /// every member the search touches. Saturating is right here and not just
    /// convenient: a value too far from the base to subtract at all is a value
    /// past every member on that side, and the saturated offset is past every
    /// stored offset on that side too, so the search lands where it should.
    #[inline]
    fn offset_of(&self, v: i64) -> i64 {
        v.checked_sub(self.base)
            .unwrap_or(if v < self.base { i64::MIN } else { i64::MAX })
    }

    /// Where `v` is, or where it would go.
    ///
    /// The two range tests in front of the binary search are Redis's and they
    /// are not an optimisation of the search, they are what makes an ascending
    /// fill linear: every add lands past the last member, answers in two loads,
    /// and appends with nothing to move.
    fn search(&self, v: i64) -> Result<usize, usize> {
        let n = self.len();
        if n == 0 {
            return Err(0);
        }
        let w = self.width();
        let off = self.offset_of(v);
        if off > self.raw_w(n - 1, w) {
            return Err(n);
        }
        if off < self.raw_w(0, w) {
            return Err(0);
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo <= hi {
            let mid = lo.midpoint(hi);
            let cur = self.raw_w(mid, w);
            if off > cur {
                lo = mid + 1;
            } else if off < cur {
                // `mid` is at least one here, because `v` is not under the
                // first member and so cannot be under member zero.
                hi = mid - 1;
            } else {
                return Ok(mid);
            }
        }
        Err(lo)
    }

    /// Rewrite every member against a frame that holds `v` too, and add `v`.
    ///
    /// `v` is outside the frame everything here is packed against, so it is not
    /// a member and it is not in the middle: it is under the smallest or over
    /// the largest, and either way there is no search.
    ///
    /// The members go out to a fixed buffer and come back rather than being
    /// shuffled where they lie. The old code could move them in place because a
    /// widen only ever moved a member to a higher offset, so back to front was
    /// safe. A refit can narrow as well as widen, and it can shift the members
    /// up by one at the same time, and there is no single direction that is
    /// safe for all of those. A run is capped at [`RUN_MAX`] members so the
    /// buffer is a known four kilobytes, and this runs once per couple of
    /// hundred inserts and never on the ascending fill.
    fn refit_and_add(&mut self, v: i64, rebase: bool) {
        let n = self.len();
        let mut held = [0i64; RUN_MAX + 2];
        for (i, slot) in held.iter_mut().enumerate().take(n) {
            *slot = self.at(i);
        }
        let ahead = usize::from(n > 0 && v < held[0]);
        if ahead == 1 {
            held.copy_within(0..n, 1);
        }
        held[if ahead == 1 { 0 } else { n }] = v;
        self.repack(&held[..n + 1], rebase);
    }

    /// Repack against the tightest frame for the members that are here.
    ///
    /// Called after a split, where both halves cover a narrower range than the
    /// run they came out of and neither of them knows it yet.
    fn rebase(&mut self) {
        let n = self.len();
        let mut held = [0i64; RUN_MAX + 2];
        for (i, slot) in held.iter_mut().enumerate().take(n) {
            *slot = self.at(i);
        }
        self.repack(&held[..n], true);
    }

    /// Give up the frame and store the members as themselves.
    ///
    /// Called when a set shrinks back to one run, which is the one shape that
    /// has to stay byte for byte a Redis intset. It costs whatever the wider
    /// width costs, on a set small enough that the difference is a few hundred
    /// bytes, and it buys back a borrow on every save.
    fn unframe(&mut self) {
        if self.base == 0 {
            return;
        }
        let n = self.len();
        let mut held = [0i64; RUN_MAX + 2];
        for (i, slot) in held.iter_mut().enumerate().take(n) {
            *slot = self.at(i);
        }
        self.repack(&held[..n], false);
    }

    /// Write `members` out against the tightest frame that holds them.
    fn repack(&mut self, members: &[i64], rebase: bool) {
        let (base, w) = match members {
            [] => (0, W16 as usize),
            [only] => (if rebase { *only } else { 0 }, {
                let off = if rebase { 0 } else { *only };
                width_of(off) as usize
            }),
            [lo, .., hi] if rebase => frame(*lo, *hi),
            [lo, .., hi] => (0, width_of(*lo).max(width_of(*hi)) as usize),
        };
        self.base = base;
        self.bytes.resize(HEADER + members.len() * w, 0);
        self.bytes[0..4].copy_from_slice(&(w as u32).to_le_bytes());
        for (i, &v) in members.iter().enumerate() {
            write_at(&mut self.bytes, HEADER + i * w, w, v - base);
        }
        self.set_len(members.len());
    }

    /// Open a slot at `at` and put `v` in it.
    fn insert_at(&mut self, at: usize, v: i64) {
        let w = self.width();
        let from = HEADER + at * w;
        let old = self.bytes.len();
        self.grow_by(w);
        self.bytes.copy_within(from..old, from + w);
        write_at(&mut self.bytes, from, w, v - self.base);
        self.set_len(self.len() + 1);
    }

    /// Make the blob `w` bytes longer without letting the vector double.
    ///
    /// [`STEP`] says why this is not just a `resize`. The reserve is skipped
    /// when the capacity already covers it, so a run built with room for what is
    /// about to go in it never calls the allocator at all.
    #[inline]
    fn grow_by(&mut self, w: usize) {
        let want = self.bytes.len() + w;
        if want > self.bytes.capacity() {
            // `yo_alloc::for_the_data` and not a fix. A run that has taken its
            // ten thousandth member has grown along the way, and this is the
            // only place in the intset that ever asks the allocator for
            // anything.
            yo_alloc::for_the_data(|| self.bytes.reserve_exact(STEP * w));
        }
        self.bytes.resize(want, 0);
    }

    #[inline]
    fn set_len(&mut self, n: usize) {
        let n = u32::try_from(n).expect("a run never reaches four billion members");
        self.bytes[4..8].copy_from_slice(&n.to_le_bytes());
    }
}

/// A run's largest member, or [`i64::MAX`] when it has none.
///
/// The sentinel is what puts an empty run last in the maxima and sends every
/// value into it, which is the empty set and nothing else.
#[inline]
fn top_of(r: &Run) -> i64 {
    r.max().unwrap_or(i64::MAX)
}

/// The base and width that hold every value from `lo` to `hi`.
///
/// The base is the middle of the range and not the bottom of it, because the
/// offsets are read back through signed readers: from the bottom they would only
/// use the positive half of the width and two bytes would hold a span of thirty
/// two thousand, and from the middle they use both halves and two bytes hold
/// sixty five thousand.
///
/// A range too wide to subtract at all is a run holding both ends of the
/// sixty four bit line, which no frame helps with, so it gets no base and the
/// widest width.
#[inline]
fn frame(lo: i64, hi: i64) -> (i64, usize) {
    let Some(span) = hi.checked_sub(lo) else {
        return (0, W64 as usize);
    };
    let base = lo + span / 2;
    let w = width_of(lo - base).max(width_of(hi - base));
    (base, w as usize)
}

/// The narrowest width that holds `v`.
#[inline]
const fn width_of(v: i64) -> u32 {
    if v < i32::MIN as i64 || v > i32::MAX as i64 {
        W64
    } else if v < i16::MIN as i64 || v > i16::MAX as i64 {
        W32
    } else {
        W16
    }
}

/// Write `v` at `at` in `w` bytes, little endian.
#[inline]
fn write_at(bytes: &mut [u8], at: usize, w: usize, v: i64) {
    match w {
        2 => bytes[at..at + 2].copy_from_slice(&(v as i16).to_le_bytes()),
        4 => bytes[at..at + 4].copy_from_slice(&(v as i32).to_le_bytes()),
        _ => bytes[at..at + 8].copy_from_slice(&v.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(vals: &[i64]) -> Intset {
        let mut s = Intset::new();
        for &v in vals {
            assert!(s.add(v), "{v} was supposed to be new");
        }
        s
    }

    fn members(s: &Intset) -> Vec<i64> {
        s.iter().collect()
    }

    /// The frame is the whole memory argument, so this is the row that says it
    /// worked. A billion apart is far outside two byte range and the set still
    /// stores its members in two bytes each, because no one run spans more than
    /// a few thousand of them.
    #[test]
    fn a_set_of_large_integers_still_stores_them_in_two_bytes() {
        let mut s = Intset::new();
        for i in 0..10_000i64 {
            s.add(1_000_000_000 + i * 3);
        }
        assert_eq!(s.width(), W16 as usize, "every run is two bytes a member");
        assert!(
            s.byte_len() < 10_000 * 2 + s.runs() * 16,
            "{} bytes for ten thousand members over {} runs",
            s.byte_len(),
            s.runs()
        );
        for i in 0..10_000i64 {
            assert!(s.contains(1_000_000_000 + i * 3), "member {i}");
            assert!(!s.contains(1_000_000_000 + i * 3 + 1), "gap after {i}");
        }
        assert_eq!(s.len(), 10_000);
    }

    /// A member arriving under a run's smallest moves the frame down rather
    /// than widening it, which is the direction the old widen path never had to
    /// think about.
    #[test]
    fn a_member_under_the_frame_moves_it_instead_of_widening_it() {
        let mut s = Intset::new();
        // Past the ceiling, so the runs and the frames exist at all.
        for i in 0..2_000i64 {
            s.add(500_000 + i * 100);
        }
        let before = s.width();
        for i in 0..50i64 {
            assert!(s.add(500_000 - 1 - i), "{i} is new and under everything");
        }
        assert_eq!(s.width(), before, "still two bytes a member");
        assert_eq!(s.min(), Some(500_000 - 50));
        assert_eq!(s.len(), 2_050);
        for i in 0..50i64 {
            assert!(s.contains(500_000 - 1 - i));
        }
    }

    /// Negative members, which is where a centred base and a signed reader
    /// could disagree with each other and nothing else would notice.
    #[test]
    fn the_frame_holds_negative_members_too() {
        let mut s = Intset::new();
        for i in 0..3_000i64 {
            s.add(-2_000_000_000 + i * 7);
        }
        assert_eq!(s.width(), W16 as usize);
        assert_eq!(s.min(), Some(-2_000_000_000));
        assert_eq!(s.max(), Some(-2_000_000_000 + 2_999 * 7));
        for i in 0..3_000i64 {
            assert!(s.contains(-2_000_000_000 + i * 7), "member {i}");
        }
        assert_eq!(members(&s).len(), 3_000);
    }

    /// A run holding both ends of the sixty four bit line, which no frame helps
    /// with and which the subtraction cannot even be done on.
    #[test]
    fn a_span_too_wide_to_subtract_gets_no_frame() {
        assert_eq!(frame(i64::MIN, i64::MAX), (0, W64 as usize));
        let mut s = Intset::new();
        for i in 0..600i64 {
            s.add(i);
        }
        s.add(i64::MIN);
        s.add(i64::MAX);
        assert_eq!(s.width(), W64 as usize, "the widest run holds both ends");
        assert!(s.contains(i64::MIN) && s.contains(i64::MAX) && s.contains(300));
        assert_eq!(s.len(), 602);
    }

    /// A set small enough for a real server to call it an intset hands over the
    /// same bytes it always did, whatever its members are, because a one run set
    /// never takes a base.
    #[test]
    fn a_one_run_set_is_still_a_redis_intset() {
        let s = of(&[1_000_000_000, 1_000_000_001, 2_000_000_000]);
        let bytes = s.as_bytes().expect("one run");
        assert_eq!(
            Intset::from_bytes(bytes).expect("a real server could read this"),
            s
        );
        assert_eq!(s.width(), W32 as usize, "no base, so the values decide");
    }

    #[test]
    fn a_set_drained_back_to_one_run_is_a_redis_intset_again() {
        let mut s = Intset::new();
        for i in 0..4_000i64 {
            s.add(1_000_000_000 + i * 3);
        }
        assert!(s.runs.len() > 1, "several runs to start with");
        assert!(s.as_bytes().is_none(), "framed, so not a Redis intset");
        for i in 100..4_000i64 {
            s.remove(1_000_000_000 + i * 3);
        }
        sound(&s);
        assert_eq!(s.runs.len(), 1, "the merges took it back to one run");
        let bytes = s.as_bytes().expect("one run, so the frame is gone");
        assert_eq!(
            Intset::from_bytes(bytes).expect("a real server could read this"),
            s
        );
        assert_eq!(s.len(), 100);
    }

    /// Everything the runs have to keep true, checked in one place so that a
    /// test only has to call this rather than remember all four.
    fn sound(s: &Intset) {
        assert!(!s.runs.is_empty(), "there is always a run to land in");
        assert_eq!(s.maxima.len(), s.runs.len(), "one maximum per run");
        let mut seen = 0usize;
        let mut last: Option<i64> = None;
        for (i, r) in s.runs.iter().enumerate() {
            assert!(
                !r.is_empty() || s.runs.len() == 1,
                "run {i} is empty and is not the only one"
            );
            assert!(r.len() <= RUN_MAX, "run {i} holds {} members", r.len());
            // A stale maximum is the one thing that sends a lookup to the wrong
            // run, and it fails silently: the member is simply not found.
            assert_eq!(s.maxima[i], top_of(r), "the maximum of run {i} is stale");
            for v in r.iter() {
                if let Some(prev) = last {
                    assert!(prev < v, "{prev} then {v} is not ascending");
                }
                last = Some(v);
            }
            seen += r.len();
        }
        assert_eq!(seen, s.len(), "the runs and the count disagree");
        // The tree has to agree with the runs at every position, which is the
        // one thing a wrong `bump` breaks silently.
        let mut at = 0usize;
        for (i, r) in s.runs.iter().enumerate() {
            for k in 0..r.len() {
                assert_eq!(s.select(at), (i, k), "position {at}");
                at += 1;
            }
        }
    }

    #[test]
    fn an_empty_set_is_eight_bytes_and_holds_nothing() {
        let s = Intset::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.width(), 2);
        assert_eq!(s.byte_len(), 8);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
        assert!(!s.contains(0));
        assert_eq!(s.as_bytes(), Some(&[2, 0, 0, 0, 0, 0, 0, 0][..]));
    }

    #[test]
    fn members_come_back_sorted_however_they_went_in() {
        let s = of(&[5, -3, 100, 0, -70, 42]);
        assert_eq!(members(&s), [-70, -3, 0, 5, 42, 100]);
        assert_eq!(s.min(), Some(-70));
        assert_eq!(s.max(), Some(100));
        assert_eq!(s.len(), 6);
        sound(&s);
    }

    #[test]
    fn adding_the_same_member_twice_says_so_and_changes_nothing() {
        let mut s = of(&[1, 2, 3]);
        assert!(!s.add(2));
        assert_eq!(members(&s), [1, 2, 3]);
        assert_eq!(s.byte_len(), 8 + 3 * 2);
    }

    #[test]
    fn a_small_set_of_integers_costs_two_bytes_each() {
        // G8's number for a set of integers, and the reason this representation
        // exists next to the listpack rather than instead of it.
        let s = of(&(0..512).collect::<Vec<i64>>());
        assert_eq!(s.runs(), 1, "512 is still one run");
        assert_eq!(s.width(), 2);
        assert_eq!(s.byte_len(), 8 + 512 * 2);
        assert_eq!((s.byte_len() - 8) / s.len(), 2);
    }

    #[test]
    fn the_width_follows_the_widest_member_and_never_comes_back_down() {
        let mut s = of(&[1, 2, 3]);
        assert_eq!(s.width(), 2);

        s.add(100_000);
        assert_eq!(s.width(), 4, "past an i16");
        assert_eq!(members(&s), [1, 2, 3, 100_000]);

        s.add(-5_000_000_000);
        assert_eq!(s.width(), 8, "past an i32");
        assert_eq!(members(&s), [-5_000_000_000, 1, 2, 3, 100_000]);

        assert!(s.remove(-5_000_000_000));
        assert!(s.remove(100_000));
        assert_eq!(s.width(), 8, "removing does not narrow it back");
        assert_eq!(members(&s), [1, 2, 3]);
    }

    #[test]
    fn widening_puts_a_negative_at_the_front_and_a_positive_at_the_back() {
        // The whole of `widen_and_add` turns on this: the new member is outside
        // the range of what is there, so it needs no search, and getting the end
        // wrong writes it over a member instead of next to one.
        let mut up = of(&[-2, -1, 0, 1, 2]);
        up.add(70_000);
        assert_eq!(members(&up), [-2, -1, 0, 1, 2, 70_000]);

        let mut down = of(&[-2, -1, 0, 1, 2]);
        down.add(-70_000);
        assert_eq!(members(&down), [-70_000, -2, -1, 0, 1, 2]);
    }

    #[test]
    fn widening_an_empty_set_still_works() {
        let mut s = Intset::new();
        assert!(s.add(i64::MIN));
        assert_eq!(s.width(), 8);
        assert_eq!(members(&s), [i64::MIN]);
    }

    #[test]
    fn the_extremes_of_every_width_land_in_the_width_they_belong_to() {
        assert_eq!(width_of(0), 2);
        assert_eq!(width_of(i64::from(i16::MAX)), 2);
        assert_eq!(width_of(i64::from(i16::MIN)), 2);
        assert_eq!(width_of(i64::from(i16::MAX) + 1), 4);
        assert_eq!(width_of(i64::from(i16::MIN) - 1), 4);
        assert_eq!(width_of(i64::from(i32::MAX)), 4);
        assert_eq!(width_of(i64::from(i32::MIN)), 4);
        assert_eq!(width_of(i64::from(i32::MAX) + 1), 8);
        assert_eq!(width_of(i64::from(i32::MIN) - 1), 8);
        assert_eq!(width_of(i64::MAX), 8);
        assert_eq!(width_of(i64::MIN), 8);

        let s = of(&[i64::MIN, i64::MAX, 0]);
        assert_eq!(members(&s), [i64::MIN, 0, i64::MAX]);
        assert!(s.contains(i64::MIN));
        assert!(s.contains(i64::MAX));
    }

    #[test]
    fn a_member_too_wide_for_the_set_is_not_in_it() {
        // Not merely absent, unrepresentable, and answering that without a
        // search is the point.
        let s = of(&[1, 2, 3]);
        assert!(!s.contains(100_000));
        assert!(!s.contains(i64::MAX));
    }

    #[test]
    fn removing_takes_out_the_right_one_and_only_that_one() {
        let mut s = of(&[10, 20, 30, 40, 50]);
        assert!(s.remove(30));
        assert_eq!(members(&s), [10, 20, 40, 50]);
        assert!(!s.remove(30), "gone already");
        assert!(s.remove(10), "the first");
        assert_eq!(members(&s), [20, 40, 50]);
        assert!(s.remove(50), "the last");
        assert_eq!(members(&s), [20, 40]);
        assert_eq!(s.byte_len(), 8 + 2 * 2, "and the blob shrank each time");
    }

    #[test]
    fn a_set_can_be_emptied_and_used_again() {
        let mut s = of(&[1, 2, 3]);
        for v in [1, 2, 3] {
            assert!(s.remove(v));
        }
        assert!(s.is_empty());
        assert_eq!(s.byte_len(), 8);
        assert!(s.add(9));
        assert_eq!(members(&s), [9]);
        sound(&s);
    }

    #[test]
    fn every_member_of_a_big_set_is_found_and_no_stranger_is() {
        // Enough members to make the binary search do real work, in an order
        // that is neither ascending nor descending so the two range tests in
        // front of it are not what is being exercised.
        let mut s = Intset::new();
        for i in 0..1000i64 {
            assert!(s.add((i * 7919) % 1000 * 2));
        }
        assert_eq!(s.len(), 1000);
        for i in 0..1000i64 {
            assert!(s.contains(i * 2), "{} is a member", i * 2);
            assert!(!s.contains(i * 2 + 1), "{} is not", i * 2 + 1);
        }
        assert_eq!(members(&s), (0..1000i64).map(|i| i * 2).collect::<Vec<_>>());
        sound(&s);
    }

    #[test]
    fn a_blob_survives_a_round_trip_through_bytes() {
        for vals in [
            &[][..],
            &[0],
            &[1, 2, 3],
            &[-70_000, 5, 70_000],
            &[i64::MIN, 0, i64::MAX],
        ] {
            let s = of(vals);
            let back = Intset::from_bytes(s.as_bytes().expect("one run")).expect("we wrote it");
            assert_eq!(back, s);
            assert_eq!(members(&back), members(&s));
        }
    }

    #[test]
    fn a_blob_that_is_wrong_is_refused_rather_than_believed() {
        assert_eq!(Intset::from_bytes(&[]), Err(Malformed::Short));
        assert_eq!(
            Intset::from_bytes(&[2, 0, 0, 0, 0, 0, 0]),
            Err(Malformed::Short)
        );

        let good = |vals: &[i64]| of(vals).as_bytes().expect("one run").to_vec();

        let mut bad = good(&[1, 2, 3]);
        bad[0] = 3;
        assert_eq!(Intset::from_bytes(&bad), Err(Malformed::Width));

        let mut short = good(&[1, 2, 3]);
        short.pop();
        assert_eq!(Intset::from_bytes(&short), Err(Malformed::Length));

        let mut over = good(&[1, 2, 3]);
        over[4] = 9;
        assert_eq!(Intset::from_bytes(&over), Err(Malformed::Length));

        // The one that would otherwise be believed: valid arithmetic, members
        // out of order, and every search after that quietly wrong.
        let mut jumbled = good(&[1, 2, 3]);
        jumbled[8..10].copy_from_slice(&9i16.to_le_bytes());
        assert_eq!(Intset::from_bytes(&jumbled), Err(Malformed::Order));

        let mut twice = good(&[1, 2, 3]);
        twice[10..12].copy_from_slice(&1i16.to_le_bytes());
        assert_eq!(Intset::from_bytes(&twice), Err(Malformed::Order));
    }

    #[test]
    fn the_header_is_little_endian_on_every_machine() {
        // Redis writes it little endian from a big endian host too, so a blob
        // this code produces has to be readable by a real server whatever it is
        // running on. Written out as bytes rather than as a round trip, because
        // a round trip through our own reader agrees with itself either way.
        let s = of(&[1, 258]);
        assert_eq!(
            s.as_bytes(),
            Some(
                &[
                    2, 0, 0, 0, // width, u32 little endian
                    2, 0, 0, 0, // count, u32 little endian
                    1, 0, // 1 as an i16 little endian
                    2, 1, // 258 as an i16 little endian
                ][..]
            )
        );
    }

    #[test]
    fn an_ascending_fill_never_moves_anything() {
        // Not a timing claim, a shape claim: `search` answers past the end for
        // every one of these, which is the branch that makes the fill linear.
        let mut r = Run::new();
        for i in 0..100i64 {
            assert_eq!(r.search(i), Err(i as usize), "{i} appends");
            r.add(i, false);
        }
        assert_eq!(r.len(), 100);
    }

    #[test]
    fn a_set_splits_at_the_ceiling_and_the_client_cannot_tell() {
        let mut s = Intset::new();
        for i in 0..RUN_MAX as i64 {
            s.add(i);
        }
        assert_eq!(s.runs(), 1, "at the ceiling it is still one array");
        assert!(s.as_bytes().is_some());

        s.add(RUN_MAX as i64);
        assert_eq!(s.runs(), 2, "one past it splits");
        assert_eq!(s.as_bytes(), None, "and there is no single blob any more");
        assert_eq!(s.len(), RUN_MAX + 1);
        assert_eq!(
            members(&s),
            (0..=RUN_MAX as i64).collect::<Vec<_>>(),
            "and every member is still there in order"
        );
        sound(&s);
    }

    #[test]
    fn a_scattered_fill_past_the_ceiling_stays_sorted_and_whole() {
        // Scattered, so the splits land in the middle of runs rather than at
        // the end of the last one, which is the case an ascending fill never
        // reaches.
        let n = 20_000i64;
        let mut s = Intset::new();
        for i in 0..n {
            assert!(s.add((i * 7919) % n), "{i}");
        }
        assert_eq!(s.len(), n as usize);
        assert!(s.runs() > 30, "it really did split, {} runs", s.runs());
        sound(&s);
        for i in 0..n {
            assert!(s.contains(i), "{i} is a member");
            assert_eq!(s.at(i as usize), i, "position {i}");
        }
        assert!(!s.contains(n));
        assert!(!s.contains(-1));
    }

    #[test]
    fn draining_a_split_set_folds_the_runs_back_together() {
        let n = 5_000i64;
        let mut s = Intset::new();
        for i in 0..n {
            s.add(i);
        }
        let split = s.runs();
        assert!(split > 5, "{split} runs to start with");
        // Out from the middle, so runs empty out in the middle of the list and
        // the merge has a neighbour on both sides to choose between.
        for i in (0..n).map(|i| (i * 7919) % n) {
            assert!(s.remove(i), "{i}");
        }
        assert!(s.is_empty());
        assert_eq!(s.runs(), 1, "back to one run, not {} empty ones", s.runs());
        sound(&s);
        assert!(s.add(1));
        assert_eq!(members(&s), [1]);
    }

    #[test]
    fn adds_and_removes_in_any_order_leave_the_runs_sound() {
        // The one that catches a wrong `bump` or a merge that loses a member,
        // by mirroring the whole thing against a `BTreeSet` and checking the
        // invariants after every write.
        use std::collections::BTreeSet;
        let mut s = Intset::new();
        let mut want = BTreeSet::new();
        let mut x = 12_345i64;
        for step in 0..12_000 {
            // A cheap deterministic spread, so the run boundaries move around
            // rather than the whole thing filling in one direction.
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let v = (x >> 33) % 4_000;
            if step % 3 == 2 {
                assert_eq!(s.remove(v), want.remove(&v), "removing {v} at {step}");
            } else {
                assert_eq!(s.add(v), want.insert(v), "adding {v} at {step}");
            }
            assert_eq!(s.len(), want.len(), "at {step}");
        }
        sound(&s);
        assert_eq!(members(&s), want.iter().copied().collect::<Vec<_>>());
    }

    #[test]
    fn a_run_only_widens_the_members_it_holds() {
        // One array would have rewritten every member to eight bytes. Here only
        // the run the big member lands in pays for it, which is the one thing
        // this layout gives that Redis's cannot.
        let mut s = Intset::new();
        for i in 0..5_000i64 {
            s.add(i);
        }
        s.add(i64::MAX);
        assert_eq!(s.width(), 8, "the widest run is eight");
        let narrow = s.runs.iter().filter(|r| r.width() == 2).count();
        assert!(narrow > 5, "only {narrow} runs stayed narrow");
        assert_eq!(s.max(), Some(i64::MAX));
        sound(&s);
    }

    #[test]
    fn a_run_never_holds_much_more_than_it_uses() {
        // The whole point of the representation. A `Vec` that doubled would put
        // this near four bytes a member at two byte width, and `STEP` is what
        // stops it.
        let mut s = Intset::new();
        for i in 0..100_000i64 {
            s.add(i);
        }
        let per = s.memory_bytes() as f64 / s.len() as f64;
        // Four byte members, because a hundred thousand is past an i16, plus
        // the run headers and the run list and the tree.
        assert!(per < 4.6, "{per:.2} bytes a member");
    }

    #[test]
    fn the_member_at_a_position_is_the_same_one_a_walk_would_reach() {
        // `at` goes down the tree and `iter` goes along the runs, and the two
        // of them agreeing at every position either side of a run boundary is
        // what makes `SRANDMEMBER` on a split set draw uniformly.
        let n = 3_000usize;
        let mut s = Intset::new();
        for i in 0..n as i64 {
            s.add(i * 3);
        }
        let walked: Vec<i64> = s.iter().collect();
        assert_eq!(walked.len(), n);
        for (i, &v) in walked.iter().enumerate() {
            assert_eq!(s.at(i), v, "position {i}");
        }
        assert_eq!(s.get(n), None, "past the end");
    }
}
