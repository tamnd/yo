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
//!   members     before                after
//!       512     4.00  intset          2.08  intset
//!     1,000    24.60  hashtable       2.19  intset
//!   100,000    30.92  hashtable       3.53  intset
//! 1,000,000    29.19  hashtable       4.08  intset
//! ```
//!
//! The 4.08 at a million is not slack, it is the width. Values up to a million
//! need four byte slots, so four bytes a member is the floor there and the
//! overhead above it is eight hundredths of a byte. Two bytes a member is only
//! reachable by a set whose members fit an `i16`, and the 2.08 at 512 is that
//! case.
//!
//! What it costs in time, from `intset_runs` in `benches/intset.rs`:
//!
//! ```text
//!                    4,096     100,000     1,000,000
//!   contains hit   10.8 ns     13.2 ns       14.7 ns
//!   contains miss   9.0 ns     10.8 ns       13.0 ns
//!   member at k     3.2 ns      8.3 ns       11.2 ns
//!   runs                15         390          3906
//!   bytes a member    2.17        4.13          4.17
//! ```
//!
//! End to end through [`crate::Set`], a set of a million integers filled in
//! scattered order went from 30.71 bytes a member to 4.17, and `SADD` went from
//! 72.6 ns to 49.8. Membership went the other way, 13.6 ns to 14.7, which is the
//! price of the two searches and is what the memory bought.
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
            [run] => Some(run.as_bytes()),
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
        if !self.runs[i].add(v) {
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
        // The members are ascending, so the two ends bound the width of
        // everything between them and there is nothing to scan.
        let w = width_of(src.at(half)).max(width_of(src.at(n - 1)));
        let mut hi = Run::with_width(w, n - half);
        for k in half..n {
            hi.push_back(src.at(k));
        }
        self.runs[i].truncate(half);
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

/// One run: a complete intset in Redis's own layout.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Run {
    /// The header and the members, in Redis's own layout, so that handing this
    /// to an RDB writer is a copy.
    bytes: Vec<u8>,
}

impl Run {
    /// An empty run at the narrowest width.
    fn new() -> Run {
        let mut bytes = Vec::with_capacity(HEADER);
        bytes.extend_from_slice(&W16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        Run { bytes }
    }

    /// An empty run at `w` bytes a member with room for `n` of them.
    fn with_width(w: u32, n: usize) -> Run {
        let mut bytes = Vec::with_capacity(HEADER + n * w as usize);
        bytes.extend_from_slice(&w.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        Run { bytes }
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
        let w = self.width();
        let at = HEADER + index * w;
        let raw = &self.bytes[at..at + w];
        match w {
            2 => i64::from(i16::from_le_bytes(raw.try_into().expect("two bytes"))),
            4 => i64::from(i32::from_le_bytes(raw.try_into().expect("four bytes"))),
            _ => i64::from_le_bytes(raw.try_into().expect("eight bytes")),
        }
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
        // A value too wide for this run's members cannot be one of them, and
        // saying so costs a compare instead of a search.
        width_of(v) <= self.width() as u32 && self.search(v).is_ok()
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
    fn add(&mut self, v: i64) -> bool {
        let w = width_of(v);
        if w > self.width() as u32 {
            self.widen_and_add(v, w);
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
        write_at(&mut self.bytes, at, w, v);
        self.set_len(self.len() + 1);
    }

    /// Put every member of `other` on the end, where they all belong.
    fn append(&mut self, other: &Run) {
        self.reserve_members(other.len());
        for v in other.iter() {
            // Through `add` and not `push_back`, because `other` may hold wider
            // members than this run has room for and widening is `add`'s job.
            // Every one of them is past the last member, so the range test in
            // front of the search answers and nothing moves.
            self.add(v);
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
        if width_of(v) > self.width() as u32 {
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
        let (mut lo, mut hi) = (from, self.len());
        while lo < hi {
            let mid = lo.midpoint(hi);
            if self.at(mid) < v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
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
        if v > self.at(n - 1) {
            return Err(n);
        }
        if v < self.at(0) {
            return Err(0);
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo <= hi {
            let mid = lo.midpoint(hi);
            let cur = self.at(mid);
            if v > cur {
                lo = mid + 1;
            } else if v < cur {
                // `mid` is at least one here, because `v` is not under the
                // first member and so cannot be under member zero.
                hi = mid - 1;
            } else {
                return Ok(mid);
            }
        }
        Err(lo)
    }

    /// Rewrite every member at `w` bytes and put `v` at whichever end it belongs.
    ///
    /// Back to front, so that a member is read before the wider write that would
    /// have covered it. `v` is outside the range of everything here, which is
    /// what being too wide means, so it goes at the front if it is negative and
    /// at the back if it is not, with no search.
    fn widen_and_add(&mut self, v: i64, w: u32) {
        let n = self.len();
        let old = self.width();
        let neww = w as usize;
        self.bytes.resize(HEADER + (n + 1) * neww, 0);
        let ahead = usize::from(v < 0);
        for i in (0..n).rev() {
            let at = HEADER + i * old;
            let raw = &self.bytes[at..at + old];
            let val = match old {
                2 => i64::from(i16::from_le_bytes(raw.try_into().expect("two bytes"))),
                4 => i64::from(i32::from_le_bytes(raw.try_into().expect("four bytes"))),
                _ => i64::from_le_bytes(raw.try_into().expect("eight bytes")),
            };
            write_at(&mut self.bytes, HEADER + (i + ahead) * neww, neww, val);
        }
        let end = if ahead == 1 { 0 } else { n };
        write_at(&mut self.bytes, HEADER + end * neww, neww, v);
        self.bytes[0..4].copy_from_slice(&w.to_le_bytes());
        self.set_len(n + 1);
    }

    /// Open a slot at `at` and put `v` in it.
    fn insert_at(&mut self, at: usize, v: i64) {
        let w = self.width();
        let from = HEADER + at * w;
        let old = self.bytes.len();
        self.grow_by(w);
        self.bytes.copy_within(from..old, from + w);
        write_at(&mut self.bytes, from, w, v);
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
            r.add(i);
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
