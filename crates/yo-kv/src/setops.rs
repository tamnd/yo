//! Set algebra, and the choice between probing and merging.
//!
//! `SINTER`, `SUNION`, `SDIFF`, `SINTERCARD` and the `*STORE` forms. This is the
//! family aki lost worst on, at 0.75x for `SINTER` and 0.30x to 0.55x for the
//! `*STORE` forms, and `08` section 4 sets the gate at ten times for all of them.
//!
//! # Two ways to do it
//!
//! **Probe.** Take the smallest set, and for each of its members ask every other
//! set whether it has it. Work is `|smallest| * (k - 1)` questions in the worst
//! case, and far fewer in practice because a member that is missing from the
//! second set is never asked about the third. Every question is a random access
//! into a different table.
//!
//! **Accumulate.** Walk every member of every set once, into one table that
//! counts how many sets each member appeared in, and then read the answer off
//! the counts. Work is `sum(|set|)` insertions, all of them into the same table.
//!
//! K11 pre-registers a crossover at k around 7: below that probe, above it merge.
//! It does not reproduce, and it is worth being exact about why, because the
//! reason is not that the number is a little out.
//!
//! # The crossover is not at seven and there is not one
//!
//! `benches/setops.rs` runs both plans over the same sets at k from 2 to 16, with
//! sets of two hundred thousand and nine tenths of every set shared, which is the
//! shape that gives probe the least help. Probe wins at every k. The gap narrows
//! as k grows, from 2.95 times at k equals 2 to 1.24 times at k equals 16, and it
//! narrows towards parity rather than towards a crossing.
//!
//! The arithmetic says the same thing once the cost of an operation is measured
//! instead of assumed. Probe does `n * (k - 1)` table operations. Accumulate does
//! `n * (k + 1)`, being one seeding insert and one count raise per member plus the
//! read back. Those are 2.7 and 3.4 million at k equals 16, a ratio of 1.26
//! against a measured 1.24. Probe does less work at every k and the ratio tends to
//! one from above, so these two never cross.
//!
//! The pre-registered number assumed a probe question is much dearer than an
//! accumulate touch, because a question is a random access into a table this
//! operation has not otherwise touched and `08` section 4 floors that at about 40
//! ns on a DRAM miss. Both come out at about 25 ns here. An accumulate touch is
//! not the cheap sequential thing the model had in mind: it hashes the member and
//! makes its own random access, into the counting table. Two random accesses that
//! cost the same cannot trade off against each other, however many of them there
//! are. This is L6's 70 ns positional probe again, which measured 13.
//!
//! # The third plan, which does change it
//!
//! `08` section 4 describes a merge that is neither of the two above: sorted
//! arrays walked in lockstep, where a touch is a pointer step and a comparison
//! with no hash anywhere. That genuinely is much cheaper than a probe question,
//! and against it a crossover can exist. It was written down as needing the
//! partitioned band and was therefore out of reach.
//!
//! It is in reach now, from the other direction. An all integer set is an
//! [`Intset`], which is exactly a sorted array, and since #148 it stays one
//! however big it gets rather than turning into a table at five hundred and
//! twelve members. So whenever every operand is an intset there is something to
//! merge, and that is most of what `SINTERSTORE` is called with: identifier
//! sets, bitmap style tag sets, anything a numeric primary key went into.
//!
//! [`Plan::Merge`] is that, over [`Walk`], and it is why `plan_for` is a
//! chooser with something to choose. The intersection is a leapfrog driven from
//! the smallest set: take the value that set is on, pull the others up to it
//! with [`Walk::seek`], and if they all land on it then it is in all of them.
//! The seek is what makes the asymmetric case cheap, because a set of ten
//! against a set of a million touches ten members of the big one and skips the
//! rest.
//!
//! The counting plan stays reachable through [`inter_with`] and the benchmark
//! keeps measuring it, because it is the control the merge has to beat.
//!
//! # What the merge is worth, measured
//!
//! `benches/setops.rs` builds the same four shapes as integer sets and runs
//! every plan over them. Milliseconds per intersection, minimum per iteration,
//! two hundred thousand members a set:
//!
//! ```text
//!                        k=2      k=4      k=8     k=16
//!   dense    merge      3.95     5.15    11.06    23.91
//!            probe      6.93    11.93    22.63    41.95
//!            count     16.32    25.91    45.03    84.90
//!   sparse   merge      0.04     0.06     0.12     0.26
//!            probe      6.17     6.21     6.38     6.54
//!   striped  merge      4.70     5.07     5.13     5.27
//!            probe      7.42     6.43     7.63     7.80
//!   skewed   merge     0.002    0.003    0.007    0.015
//!            probe     0.004    0.007    0.013    0.023
//! ```
//!
//! And the other three commands, where the merge's opposite number is the table
//! for the union and the probe for the other two:
//!
//! ```text
//!                        k=2      k=4      k=8     k=16
//!   union    merge      1.47     4.35    14.76    54.49
//!            table     13.30    31.21    69.84   149.82
//!   diff     merge      4.40     5.16     8.60    15.98
//!            probe      6.94    10.02    16.65    29.07
//!   store    merge      7.17    10.36    16.59    29.79
//!            probe     13.88    23.95    36.69    63.24
//! ```
//!
//! The merge wins every row at every k. The narrowest is 1.27 times and the
//! widest is 141, which is a spread wide enough to be worth explaining rather
//! than averaging.
//!
//! # Where the spread comes from, and the shape that nearly broke it
//!
//! `sparse` and `striped` hold the same sets with the same one percent overlap
//! and differ only in where each set's unshared members sit. In `sparse` they
//! are in a range of their own, so a cursor that lands in another set's range
//! steps over the whole range in one binary search. In `striped` they are
//! interleaved one for one, so there is nothing to skip and a step is worth a
//! single member. That is 141 times against 1.6, on data that is identical by
//! every summary statistic an optimiser could look at. Real data lies between
//! the two and the number to quote is the striped one.
//!
//! Getting that row right took two goes and it is the reason the shape is in the
//! benchmark. The first merge was symmetric: no set in charge, the largest value
//! any cursor held as the target, every cursor visited in turn. On `striped` it
//! was nine times slower than the probe at k of 16, and it deserved to be. A
//! symmetric leapfrog costs a step per member of the union of every operand,
//! because proving that nothing matches means looking at everything, and the
//! union is `k` times the smallest set. The probe reads the smallest set once
//! and fails on its first question, so it is flat in k, which is exactly what
//! `sparse_probe` and `striped_probe` do at about 6 to 8 ms across the range.
//!
//! Driving the leapfrog from the smallest set fixes it, because it puts the
//! merge on the probe's own bound: a step per member of the smallest operand,
//! plus one per overshoot, over a step that is cheaper than a hash and a random
//! access. `striped_merge` is 4.70 ms at k of 2 and 5.27 at k of 16, which is
//! the same flatness the probe has with a smaller constant.
//!
//! So the honest claim is not that the merge is a different order of cost. It is
//! that the merge is never worse than the probe by more than its constant and is
//! sometimes better by two orders, and that the plan is free to take because the
//! representation already sorted the data.
//!
//! The one row with a slope worth watching is the union, which finds the
//! smallest value by looking at every cursor and is therefore quadratic in k
//! where the table is linear. It wins by 9.1 times at k of 2 and 2.75 at k of 16,
//! and extrapolating the two slopes they would cross somewhere past k of 50. A
//! heap would make it `log k` at the cost of a comparison per push, and there is
//! no point paying that until a `SUNION` with fifty keys turns up.
//!
//! # Ordering
//!
//! A probe or a count returns members in the order the first relevant set holds
//! them, which is insertion order for a listpack or a table and ascending for an
//! intset. A merge returns them ascending. Redis makes no ordering promise for
//! any of these, and picking the order the data is already in means the walk is
//! sequential and there is nothing to sort.
//!
//! For the intersection and the difference the two agree, because the plan only
//! changes when every operand is an intset and the set being walked is then
//! ascending either way. For the union they do not: the table walks the sets in
//! turn and the merge interleaves them. That is the one place a plan is visible
//! from outside, and it is visible only to a client that was relying on
//! something Redis never promised.
//!
//! # The three representations
//!
//! The operand is a [`Set`], which is one of three things, and not the element
//! table it used to be. The walked set gives up members through the same
//! [`Set::iter`] everything else uses, and the questioned sets answer through
//! [`Set::has`], which is [`Set::contains`] with the parse and the hash lifted
//! out into a [`Needle`] so they happen once per member rather than once per
//! question.
//!
//! What that buys is that the algebra never has to know what it is holding. It
//! also means the members cross between representations correctly, which is not
//! automatic: an intset member is a number that has no digits anywhere, and a
//! table stores that same member as its digits, so `SINTER ints table` only
//! finds anything because the needle carries both forms.
//!
//! # Presizing
//!
//! The `*STORE` forms hand the destination a size before they start filling it,
//! taken from the smallest input, which is Y18's rule and an upper bound on any
//! intersection. `05` section 3.1 wants that to be one arena bump. Until the
//! arena is under this, the destination's own hint is the same promise with a
//! different allocator behind it.

use yo_common::num::DIGITS_MAX;

use crate::intset::Walk;
use crate::set::{Limits, Needle, Set};
use crate::{Elements, Intset};

/// How to answer a set operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Walk the smallest set and question the others about each member.
    Probe,
    /// Walk everything once into one counting table.
    Accumulate,
    /// Step through every set at once, in order, comparing and never hashing.
    ///
    /// Only possible when every operand is an intset, because that is the only
    /// representation that holds its members in order. [`inter_with`] will
    /// refuse this plan for anything else rather than answer wrongly.
    Merge,
}

/// The members every set has, in the order the smallest set holds them.
///
/// `limit` is `SINTERCARD`'s, and zero means no limit. The count comes back
/// whether or not the caller collected anything, so `SINTERCARD` is this
/// function with a callback that does nothing.
///
/// An empty input, or any empty set, is an empty intersection, which is what
/// Redis says and is also the only sane reading.
///
/// A merge when every operand is an intset and a probe otherwise, which is a
/// chooser with something to choose. See `plan_for`.
pub fn inter<F>(sets: &[&Set], limit: usize, f: F) -> usize
where
    F: FnMut(&[u8]),
{
    inter_with(plan_for(sets), sets, limit, f)
}

/// Which plan the operands allow and deserve.
///
/// The merge is not a preference, it is a fact about the representation: two
/// sorted arrays can be stepped through together and a table cannot, so a
/// mixture of the two has nothing to merge and probes.
///
/// There is nothing to choose beyond that. A cost model that guessed at the
/// overlap would be the obvious next thing to build and it is not needed,
/// because the merge is driven from the smallest set and therefore carries the
/// probe's own bound: it wins every shape in the benchmark, including the one
/// laid out so that nothing can be skipped, and its worst row is still 1.27
/// times ahead. See the module doc.
fn plan_for(sets: &[&Set]) -> Plan {
    if sets.iter().all(|s| s.ints().is_some()) {
        Plan::Merge
    } else {
        Plan::Probe
    }
}

/// Every operand as an intset, or `None` if any of them is something else.
fn as_ints<'a>(sets: &[&'a Set]) -> Option<Vec<&'a Intset>> {
    sets.iter().map(|s| s.ints()).collect()
}

/// The same, with the plan named rather than assumed.
///
/// This is how the benchmark runs every plan over the same sets, which is the
/// only way to find out where they cross and the only way to check that they
/// agree on the answer. It is public because a caller that knows the shape of its
/// own data knows more about it than [`inter`] can see from the sets alone.
///
/// [`Plan::Merge`] falls back to a probe when the operands are not all intsets,
/// because a caller asking for it has stated a preference and not a fact, and
/// the fact wins.
pub fn inter_with<F>(how: Plan, sets: &[&Set], limit: usize, f: F) -> usize
where
    F: FnMut(&[u8]),
{
    if sets.is_empty() || sets.iter().any(|s| s.is_empty()) {
        return 0;
    }
    match how {
        Plan::Merge => match as_ints(sets) {
            Some(ints) => inter_merge(&ints, limit, f),
            None => inter_probe(sets, limit, f),
        },
        Plan::Probe => inter_probe(sets, limit, f),
        Plan::Accumulate => inter_accumulate(sets, limit, f),
    }
}

/// Step through every set at once, in order, and take what they all agree on.
///
/// Leapfrog, driven from the smallest set. Take the value that set is on, pull
/// every other cursor up to it with [`Walk::seek`], and if they all land on it
/// then it is in all of them. A cursor that lands past it has just proved that
/// nothing between the two values is in the answer, so that value becomes the
/// target and the driver is seeked to it as well, which is what lets a set of
/// ten against a set of a million touch ten members of the big one rather than
/// a million.
///
/// The others are seeked smallest first, and the loop restarts from the first of
/// them the moment one of them overshoots, which is the probe's early exit in a
/// different spelling: a member that is going to fail usually fails against the
/// smallest of the others and never gets asked about the rest.
///
/// # Why it is driven rather than symmetric
///
/// The first version of this was symmetric. It held the largest value any cursor
/// was on and went round them in turn, with no set in charge and no early exit,
/// and on the shape that has nothing to skip it was nine times slower than the
/// probe at k of 16 where this one is level with it. `benches/setops.rs` has the
/// `striped` row that found it and the module doc has what it means.
///
/// The reason is that a symmetric leapfrog costs one step per member of the
/// union of all the operands, because proving nothing matches means looking at
/// everything. A driven one costs one step per member of the smallest operand
/// plus one per overshoot, which is the same bound the probe has, over a step
/// that is cheaper than the probe's. So it cannot lose by much and it can win by
/// a lot.
///
/// The order is ascending, which is the order the probe plan produces on these
/// same operands, since the set it walks is an intset and holds its members that
/// way. So the answer does not change shape when the plan does.
fn inter_merge<F>(sets: &[&Intset], limit: usize, mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    let mut order: Vec<usize> = (0..sets.len()).collect();
    order.sort_unstable_by_key(|&i| sets[i].len());
    let mut driver = sets[order[0]].walk();
    let mut others: Vec<Walk<'_>> = order[1..].iter().map(|&i| sets[i].walk()).collect();

    let mut digits = [0u8; DIGITS_MAX];
    let mut found = 0usize;
    'members: while let Some(target) = driver.peek() {
        for w in &mut others {
            w.seek(target);
            match w.peek() {
                // This set has nothing left, so neither has the answer.
                None => break 'members,
                Some(v) if v > target => {
                    // Everything from `target` up to `v` is missing from this
                    // set, so the driver can skip all of it in one search.
                    driver.seek(v);
                    continue 'members;
                }
                Some(_) => {}
            }
        }
        f(yo_common::num::i64_digits(&mut digits, target));
        found += 1;
        if limit != 0 && found == limit {
            break;
        }
        driver.bump();
    }
    found
}

/// Walk the smallest set, question the rest.
///
/// The other sets are asked smallest first. That is not tidiness: a member that
/// is going to fail will usually fail against the smallest of the others, and
/// asking that one first is what turns `k - 1` questions per member into closer
/// to one.
fn inter_probe<F>(sets: &[&Set], limit: usize, mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    let mut order: Vec<usize> = (0..sets.len()).collect();
    order.sort_unstable_by_key(|&i| sets[i].len());
    let (&first, rest) = order.split_first().expect("not empty");

    let mut digits = [0u8; DIGITS_MAX];
    let mut found = 0usize;
    for m in sets[first].iter() {
        // Parsed and hashed once, asked k-1 times. Without this both are paid
        // per question about the same member. See [`Needle`].
        let needle = Needle::of(m, &mut digits);
        if rest.iter().all(|&i| sets[i].has(&needle)) {
            f(needle.bytes());
            found += 1;
            if limit != 0 && found == limit {
                break;
            }
        }
    }
    found
}

/// Walk everything once into one counting table.
///
/// A member of the first set starts at one and every later set that has it
/// raises it, so a member with the full count is in all of them. Members that
/// are not in the first set are never entered at all, which keeps the table no
/// bigger than the first set and is why the first set is the smallest one.
fn inter_accumulate<F>(sets: &[&Set], limit: usize, mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    let mut order: Vec<usize> = (0..sets.len()).collect();
    order.sort_unstable_by_key(|&i| sets[i].len());
    let (&first, rest) = order.split_first().expect("not empty");

    let mut digits = [0u8; DIGITS_MAX];
    let mut seen = Elements::<u32>::with_capacity(sets[first].len());
    for m in sets[first].iter() {
        seen.insert(text(m, &mut digits), 1)
            .expect("no larger than its source");
    }
    for &i in rest {
        for m in sets[i].iter() {
            if let Some(count) = seen.get_mut(text(m, &mut digits)) {
                *count += 1;
            }
        }
    }

    // Read the answer off the first set rather than off the counting table, so
    // the order the caller sees does not depend on which plan ran.
    let k = sets.len() as u32;
    let mut found = 0usize;
    for m in sets[first].iter() {
        let name = text(m, &mut digits);
        if seen.get(name) == Some(&k) {
            f(name);
            found += 1;
            if limit != 0 && found == limit {
                break;
            }
        }
    }
    found
}

/// A member as the bytes a table keys on.
///
/// The counting plan and the union never ask another set a question, so they
/// want a member's bytes and nothing else. Going through a [`Needle`] would
/// parse and hash for nobody, since the table they are about to touch hashes it
/// again on the way in.
#[inline]
fn text<'a>(m: crate::set::Member<'a>, digits: &'a mut [u8; DIGITS_MAX]) -> &'a [u8] {
    match m {
        crate::set::Member::Str(s) => s,
        crate::set::Member::Int(n) => yo_common::num::i64_digits(digits, n),
    }
}

/// Every member of any of the sets, each once.
///
/// A union has to read every member of every set whatever it does, so the only
/// question is what it does with each one. Against a mixture of representations
/// the answer is one insertion into a table that is also the duplicate check,
/// and against intsets it is a merge, where the duplicate check is that two
/// cursors are on the same value and costs a comparison rather than a hash.
///
/// The order differs between the two, and that is the one place a plan is
/// visible from outside. The table walks the sets in turn, so it answers in the
/// order each set holds its members, and the merge answers in ascending order
/// across all of them. Redis promises neither.
pub fn union<F>(sets: &[&Set], f: F) -> usize
where
    F: FnMut(&[u8]),
{
    union_with(plan_for(sets), sets, f)
}

/// The same, with the plan named rather than assumed.
///
/// [`inter_with`]'s reason for existing, applied here: the benchmark has to be
/// able to run the table over the very sets the merge is fastest on, or the
/// claim that the merge is worth having is a claim about two different inputs.
///
/// There are only two plans here, so anything that is not [`Plan::Merge`] is the
/// table, and a merge asked for over operands that cannot merge is the table too.
pub fn union_with<F>(how: Plan, sets: &[&Set], f: F) -> usize
where
    F: FnMut(&[u8]),
{
    match (how, as_ints(sets)) {
        (Plan::Merge, Some(ints)) if !ints.is_empty() => union_merge(&ints, f),
        _ => union_table(sets, f),
    }
}

/// Step through every set at once and take the smallest value each round.
///
/// The smallest is found by looking at every cursor, which is `k` comparisons a
/// member and no hashing at all. That makes this quadratic in `k` where the
/// table is linear, so the win narrows from 9.1 times at k of 2 to 2.75 at k of
/// 16 and the two would cross somewhere past k of 50. A heap would turn the scan
/// into `log k` at the cost of a comparison per push, and it is not worth paying
/// for a `SUNION` nobody writes.
fn union_merge<F>(sets: &[&Intset], mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    let mut walks: Vec<Walk<'_>> = sets.iter().map(|s| s.walk()).collect();
    let mut digits = [0u8; DIGITS_MAX];
    let mut found = 0usize;
    while let Some(low) = walks.iter().filter_map(Walk::peek).min() {
        f(yo_common::num::i64_digits(&mut digits, low));
        found += 1;
        // Every cursor sitting on it, because the same member in two sets is
        // one member and this is where that is decided.
        for w in &mut walks {
            if w.peek() == Some(low) {
                w.bump();
            }
        }
    }
    found
}

/// Walk everything into one table, where the table is the duplicate check.
fn union_table<F>(sets: &[&Set], mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    // The result is at most everything, and presizing to the largest input is
    // the cheap half of that bound without pretending to know the overlap.
    let biggest = sets.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut digits = [0u8; DIGITS_MAX];
    let mut seen = Elements::<()>::with_capacity(biggest);
    let mut found = 0usize;
    for s in sets {
        for m in s.iter() {
            // The bytes are the duplicate check, which is what makes the same
            // member found in two representations one member: an intset's 42
            // and a table's `42` key the same, and `042` keys as itself,
            // because that is the same rule that decided how each was stored.
            let name = text(m, &mut digits);
            if seen.insert(name, ()).is_ok_and(|was| was.is_none()) {
                f(name);
                found += 1;
            }
        }
    }
    found
}

/// The members of the first set that no later set has.
///
/// The first set is the one being walked whether we like it or not, so the only
/// choice is how each member is checked. Against a mixture that is a question
/// per member, asked smallest set first because a member that is going to be
/// found will usually be found there, and a member that is in the second set is
/// never asked about the third. Against intsets it is a merge, and the order is
/// the same either way because both walk the first set and the first set is
/// ascending.
pub fn diff<F>(sets: &[&Set], f: F) -> usize
where
    F: FnMut(&[u8]),
{
    diff_with(plan_for(sets), sets, f)
}

/// The same, with the plan named rather than assumed. See [`union_with`].
pub fn diff_with<F>(how: Plan, sets: &[&Set], f: F) -> usize
where
    F: FnMut(&[u8]),
{
    match (how, as_ints(sets)) {
        (Plan::Merge, Some(ints)) if !ints.is_empty() => diff_merge(&ints, f),
        _ => diff_probe(sets, f),
    }
}

/// Walk the first set, dragging a cursor through each of the others behind it.
///
/// The cursors only ever move forward, so the whole operation costs one pass
/// over the first set and at most one pass over each of the others, however many
/// members are in the answer. A probe pays a hash and a random access per member
/// per set instead.
fn diff_merge<F>(sets: &[&Intset], mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    let (first, rest) = sets.split_first().expect("not empty");
    let mut walk = first.walk();
    let mut others: Vec<Walk<'_>> = rest.iter().map(|s| s.walk()).collect();
    let mut digits = [0u8; DIGITS_MAX];
    let mut found = 0usize;
    while let Some(v) = walk.peek() {
        let mut anyone = false;
        for w in &mut others {
            w.seek(v);
            if w.peek() == Some(v) {
                anyone = true;
                break;
            }
        }
        if !anyone {
            f(yo_common::num::i64_digits(&mut digits, v));
            found += 1;
        }
        walk.bump();
    }
    found
}

/// Walk the first set and ask the others about every member.
fn diff_probe<F>(sets: &[&Set], mut f: F) -> usize
where
    F: FnMut(&[u8]),
{
    let Some((first, rest)) = sets.split_first() else {
        return 0;
    };
    let mut order: Vec<usize> = (0..rest.len()).collect();
    order.sort_unstable_by_key(|&i| rest[i].len());

    let mut digits = [0u8; DIGITS_MAX];
    let mut found = 0usize;
    for m in first.iter() {
        let needle = Needle::of(m, &mut digits);
        if !order.iter().any(|&i| rest[i].has(&needle)) {
            f(needle.bytes());
            found += 1;
        }
    }
    found
}

/// The `*STORE` forms: run the operation and build the result as a set.
///
/// Presized once from `upper`, which the caller takes from the smallest input
/// for an intersection or a difference and the sum for a union. Y18's rule, and
/// the thing that stopped aki's `*STORE` family at 0.30x was that it was not
/// applied.
///
/// Nothing comes back when nothing was found, because an empty set is not a
/// thing that can exist. That is not a tidy up either: `SINTERSTORE d a b` with
/// an empty intersection deletes `d` and answers zero, so the caller needs the
/// difference between a set of no members and no set, and this is where it is.
///
/// The result picks its own representation from its first member and `upper`,
/// through the same [`Set::with_hint`] `SADD` uses, so intersecting two intsets
/// stores an intset rather than storing a table that happens to hold digits.
/// The members arrive as bytes and that is enough to decide it, because the
/// rule that made a member an integer on the way in is the rule that reads it
/// as one on the way out.
pub fn collect(
    upper: usize,
    limits: &Limits,
    run: impl FnOnce(&mut dyn FnMut(&[u8])),
) -> Option<Set> {
    let mut out: Option<Set> = None;
    run(&mut |name| match &mut out {
        Some(s) => {
            s.add(name, limits);
        }
        None => {
            let mut s = Set::with_hint(name, upper, limits);
            s.add(name, limits);
            out = Some(s);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set::Encoding;

    /// A set holding these members, in whatever representation it picks.
    fn set(members: &[&str]) -> Set {
        of(members.iter().map(|m| m.as_bytes()))
    }

    /// The same from bytes, for the members that are not text.
    fn of<'a>(members: impl IntoIterator<Item = &'a [u8]>) -> Set {
        let mut s = Set::new();
        for m in members {
            s.add(m, &Limits::DEFAULT);
        }
        s
    }

    /// A named way of building a set in one particular representation.
    type Band = (&'static str, fn(&[&str]) -> Set);

    /// A set forced past a band, so that a test can pick which representation
    /// its operands are in rather than take whatever the member count gives.
    fn banded(members: &[&str], limits: &Limits) -> Set {
        let mut s = Set::new();
        for m in members {
            s.add(m.as_bytes(), limits);
        }
        s
    }

    /// Limits that put a set of any size in each of the three bands.
    const AS_INTSET: Limits = Limits {
        max_intset_entries: usize::MAX,
        max_listpack_entries: usize::MAX,
        max_listpack_value: usize::MAX,
    };
    const AS_LISTPACK: Limits = Limits {
        max_intset_entries: 0,
        max_listpack_entries: usize::MAX,
        max_listpack_value: usize::MAX,
    };
    const AS_TABLE: Limits = Limits {
        max_intset_entries: 0,
        max_listpack_entries: 0,
        max_listpack_value: 0,
    };

    /// A table holding these members, whatever they are.
    ///
    /// [`AS_TABLE`] is not enough on its own any more. Since #148 an all integer
    /// set stays an intset past every ceiling and only changes the word
    /// `OBJECT ENCODING` answers, so no configuration puts one in a table. What
    /// still does is a member that is not an integer, and taking it out again
    /// leaves the table behind, because every promotion here is one way.
    fn tabled(members: &[&str]) -> Set {
        let mut s = Set::new();
        s.add(b"not a number", &AS_TABLE);
        for m in members {
            s.add(m.as_bytes(), &AS_TABLE);
        }
        s.remove(b"not a number");
        assert_eq!(s.encoding(), Encoding::Hashtable);
        assert!(s.ints().is_none(), "and a table underneath the word");
        s
    }

    fn run<F>(op: F) -> Vec<String>
    where
        F: FnOnce(&mut dyn FnMut(&[u8])) -> usize,
    {
        let mut got = Vec::new();
        let n = op(&mut |m| got.push(String::from_utf8_lossy(m).into_owned()));
        assert_eq!(n, got.len(), "the count and the members disagree");
        got
    }

    #[test]
    fn an_intersection_is_what_they_all_have() {
        let a = set(&["a", "b", "c", "d"]);
        let b = set(&["b", "c", "d", "e"]);
        let c = set(&["c", "d", "e", "f"]);
        let got = run(|f| inter(&[&a, &b, &c], 0, f));
        assert_eq!(got, vec!["c", "d"]);
    }

    #[test]
    fn an_intersection_of_one_set_is_that_set() {
        let a = set(&["x", "y"]);
        assert_eq!(run(|f| inter(&[&a], 0, f)), vec!["x", "y"]);
    }

    #[test]
    fn an_empty_set_anywhere_empties_the_intersection() {
        let a = set(&["a", "b"]);
        let empty = set(&[]);
        assert_eq!(run(|f| inter(&[&a, &empty], 0, f)), Vec::<String>::new());
        assert_eq!(run(|f| inter(&[&empty, &a], 0, f)), Vec::<String>::new());
        assert_eq!(run(|f| inter(&[], 0, f)), Vec::<String>::new());
    }

    /// `SINTERCARD` stops as soon as it has enough, and stopping early must not
    /// change the members it already handed over.
    #[test]
    fn a_limit_stops_the_intersection_early() {
        let a = set(&["a", "b", "c", "d", "e"]);
        let b = set(&["a", "b", "c", "d", "e"]);
        assert_eq!(run(|f| inter(&[&a, &b], 2, f)), vec!["a", "b"]);
        assert_eq!(run(|f| inter(&[&a, &b], 99, f)).len(), 5);
        assert_eq!(run(|f| inter(&[&a, &b], 0, f)).len(), 5, "zero is no limit");
    }

    /// The two plans are two ways to compute the same thing, so they have to
    /// agree on the members and on the order, or a client sees the answer change
    /// when a set grows past a threshold it cannot see.
    #[test]
    fn both_plans_give_the_same_answer_in_the_same_order() {
        let sets: Vec<Set> = (0..9)
            .map(|s| {
                let members: Vec<String> = (0..200)
                    .filter(|i| i % (s + 2) != 1)
                    .map(|i| format!("m{i}"))
                    .collect();
                set(&members.iter().map(String::as_str).collect::<Vec<_>>())
            })
            .collect();
        let refs: Vec<&Set> = sets.iter().collect();

        let probed = run(|f| inter_with(Plan::Probe, &refs, 0, f));
        let piled = run(|f| inter_with(Plan::Accumulate, &refs, 0, f));
        assert_eq!(probed, piled);
        assert!(!probed.is_empty(), "the fixture should overlap");
        assert_eq!(
            run(|f| inter(&refs, 0, f)),
            probed,
            "and so does the chooser"
        );
    }

    #[test]
    fn a_union_has_everything_once() {
        let a = set(&["a", "b"]);
        let b = set(&["b", "c"]);
        let c = set(&["c", "d"]);
        assert_eq!(run(|f| union(&[&a, &b, &c], f)), vec!["a", "b", "c", "d"]);
        assert_eq!(run(|f| union(&[], f)), Vec::<String>::new());
    }

    #[test]
    fn a_difference_takes_the_others_out_of_the_first() {
        let a = set(&["a", "b", "c", "d"]);
        let b = set(&["b"]);
        let c = set(&["d", "e"]);
        assert_eq!(run(|f| diff(&[&a, &b, &c], f)), vec!["a", "c"]);
        assert_eq!(run(|f| diff(&[&a], f)), vec!["a", "b", "c", "d"]);
        assert_eq!(run(|f| diff(&[], f)), Vec::<String>::new());
    }

    /// Ten sets all holding the same members is the shape that gives probe the
    /// least help, because nothing fails early and every member is asked about by
    /// every other set. It is the shape the benchmark measures and the one K11's
    /// number was about, so the plans have to agree on it in particular.
    #[test]
    fn the_plans_agree_where_every_set_holds_everything() {
        let members: Vec<String> = (0..100).map(|i| format!("m{i}")).collect();
        let names: Vec<&str> = members.iter().map(String::as_str).collect();
        let sets: Vec<Set> = (0..10).map(|_| set(&names)).collect();
        let refs: Vec<&Set> = sets.iter().collect();

        let probed = run(|f| inter_with(Plan::Probe, &refs, 0, f));
        assert_eq!(probed, members, "everything is in all ten");
        assert_eq!(run(|f| inter_with(Plan::Accumulate, &refs, 0, f)), probed);
        assert_eq!(run(|f| inter(&refs, 0, f)), probed);
    }

    #[test]
    fn a_store_form_builds_a_set_of_the_result() {
        let a = set(&["a", "b", "c"]);
        let b = set(&["b", "c", "d"]);
        let out = collect(a.len().min(b.len()), &Limits::DEFAULT, |f| {
            inter(&[&a, &b], 0, f);
        })
        .expect("two members is a set");
        assert_eq!(out.len(), 2);
        assert!(out.contains(b"b") && out.contains(b"c"));
        assert!(!out.contains(b"a"));
    }

    /// A result of nothing is no set at all, which is the difference the STORE
    /// forms need: an empty intersection deletes the destination rather than
    /// leaving an empty set behind that EXISTS would answer one for.
    #[test]
    fn a_store_form_of_nothing_is_nothing() {
        let a = set(&["a"]);
        let b = set(&["b"]);
        assert!(
            collect(1, &Limits::DEFAULT, |f| {
                inter(&[&a, &b], 0, f);
            })
            .is_none()
        );
    }

    /// The destination picks its own representation from what went into it, so
    /// intersecting two intsets stores an intset and not a table of digits.
    #[test]
    fn a_store_form_keeps_the_representation_its_members_deserve() {
        let a = set(&["1", "2", "3"]);
        let b = set(&["2", "3", "4"]);
        assert_eq!(a.encoding(), Encoding::Intset);
        let out = collect(3, &Limits::DEFAULT, |f| {
            inter(&[&a, &b], 0, f);
        })
        .expect("two members");
        assert_eq!(out.encoding(), Encoding::Intset);
        assert!(out.contains(b"2") && out.contains(b"3"));

        // And a union with one string in it does not, because one member that
        // is not a number is all it takes.
        let c = set(&["x"]);
        let out = collect(4, &Limits::DEFAULT, |f| {
            union(&[&a, &c], f);
        })
        .expect("four members");
        assert_ne!(out.encoding(), Encoding::Intset);
        assert!(out.contains(b"1") && out.contains(b"x"));
    }

    /// The one that could not have worked before this: an intset member is a
    /// number with no digits anywhere and a table stores that same member as
    /// its digits, so every pairing of the three representations has to agree
    /// about what a member is or the answers come back empty.
    #[test]
    fn the_three_representations_intersect_each_other() {
        let names = ["1", "2", "3", "4"];
        let others = ["3", "4", "5", "6"];
        // The table is built rather than configured, because no ceiling puts an
        // all integer set in one any more. See [`tabled`].
        let bands: [Band; 3] = [
            ("intset", |m| banded(m, &AS_INTSET)),
            ("listpack", |m| banded(m, &AS_LISTPACK)),
            ("table", tabled),
        ];
        for (ln, left) in bands {
            for (rn, right) in bands {
                let a = left(&names);
                let b = right(&others);
                let mut got = run(|f| inter(&[&a, &b], 0, f));
                got.sort();
                assert_eq!(got, ["3", "4"], "{ln} against {rn}");

                let mut got = run(|f| union(&[&a, &b], f));
                got.sort();
                assert_eq!(got, ["1", "2", "3", "4", "5", "6"], "{ln} with {rn}");

                let mut got = run(|f| diff(&[&a, &b], f));
                got.sort();
                assert_eq!(got, ["1", "2"], "{ln} without {rn}");
            }
        }
    }

    /// A member that looks like a number and a member that does not quite are
    /// two different members, and which one a set stored is decided by the same
    /// rule the algebra reads it back by.
    #[test]
    fn a_number_and_its_untidy_spelling_stay_two_members() {
        let a = banded(&["42", "042", "-0"], &AS_LISTPACK);
        let b = banded(&["42"], &AS_INTSET);
        assert_eq!(run(|f| inter(&[&a, &b], 0, f)), vec!["42"]);
        let mut got = run(|f| diff(&[&a, &b], f));
        got.sort();
        assert_eq!(got, ["-0", "042"]);
        let mut got = run(|f| union(&[&a, &b], f));
        got.sort();
        assert_eq!(
            got,
            ["-0", "042", "42"],
            "and the union does not merge them"
        );
    }

    /// A set of the given integers, which is an intset and so is mergeable.
    fn ints(vals: &[i64]) -> Set {
        let mut s = Set::new();
        for v in vals {
            s.add(v.to_string().as_bytes(), &AS_INTSET);
        }
        assert_eq!(s.encoding(), Encoding::Intset);
        s
    }

    /// Integers from a cheap scrambler, so the sets are not runs of consecutive
    /// values and the cursors have something to skip over.
    fn scattered(n: usize, seed: i64, span: i64) -> Vec<i64> {
        (0..n as i64)
            .map(|i| (i.wrapping_add(seed).wrapping_mul(2_654_435_761)).rem_euclid(span))
            .collect()
    }

    /// The merge and the probe are two ways to compute the same thing, so they
    /// have to agree member for member and in order, on every shape.
    ///
    /// The shapes matter more than the count. Two sets of the same size that
    /// mostly overlap is what the seek never gets to help with, a small set
    /// against a huge one is what it exists for, and disjoint ranges are where a
    /// single seek is meant to cross the whole of the other set at once.
    #[test]
    fn the_merge_and_the_probe_agree_on_every_shape() {
        let shapes: [(&str, Vec<Vec<i64>>); 5] = [
            (
                "same size, mostly shared",
                vec![scattered(4_000, 0, 5_000), scattered(4_000, 7, 5_000)],
            ),
            (
                "ten against a hundred thousand",
                vec![scattered(10, 3, 100_000), scattered(100_000, 0, 200_000)],
            ),
            (
                "disjoint ranges",
                vec![(0..2_000).collect(), (900_000..902_000).collect()],
            ),
            (
                "five sets",
                vec![
                    scattered(3_000, 1, 4_000),
                    scattered(3_000, 2, 4_000),
                    scattered(3_000, 3, 4_000),
                    scattered(3_000, 4, 4_000),
                    scattered(3_000, 5, 4_000),
                ],
            ),
            (
                "negatives and a member too wide for a narrow run",
                vec![
                    vec![-9_000_000_000, -3, -2, -1, 0, 1, 2, 9_000_000_000],
                    vec![-9_000_000_000, -2, 0, 2, 4, 9_000_000_000],
                ],
            ),
        ];

        for (what, vals) in shapes {
            let sets: Vec<Set> = vals.iter().map(|v| ints(v)).collect();
            let refs: Vec<&Set> = sets.iter().collect();
            assert_eq!(plan_for(&refs), Plan::Merge, "{what}");

            let probed = run(|f| inter_with(Plan::Probe, &refs, 0, f));
            assert_eq!(run(|f| inter(&refs, 0, f)), probed, "intersect {what}");
            assert_eq!(
                run(|f| inter_with(Plan::Accumulate, &refs, 0, f)),
                probed,
                "and the count agrees, {what}"
            );

            let subbed = diff_the_slow_way(&vals);
            assert_eq!(run(|f| diff(&refs, f)), subbed, "sub {what}");
            assert_eq!(
                run(|f| diff_with(Plan::Probe, &refs, f)),
                subbed,
                "and the probe agrees, {what}"
            );

            let mut piled: Vec<String> = union_the_slow_way(&vals);
            piled.sort();
            for how in [Plan::Merge, Plan::Probe] {
                let mut got = run(|f| union_with(how, &refs, f));
                got.sort();
                assert_eq!(got, piled, "union {what} by {how:?}");
            }
        }
    }

    /// The difference worked out with a `BTreeSet`, which is the answer the
    /// merge has to match and shares no code with it.
    fn diff_the_slow_way(vals: &[Vec<i64>]) -> Vec<String> {
        let (first, rest) = vals.split_first().expect("not empty");
        let others: std::collections::BTreeSet<i64> =
            rest.iter().flat_map(|v| v.iter().copied()).collect();
        let mut left: Vec<i64> = first
            .iter()
            .copied()
            .filter(|v| !others.contains(v))
            .collect();
        left.sort_unstable();
        left.dedup();
        left.iter().map(i64::to_string).collect()
    }

    fn union_the_slow_way(vals: &[Vec<i64>]) -> Vec<String> {
        let all: std::collections::BTreeSet<i64> =
            vals.iter().flat_map(|v| v.iter().copied()).collect();
        all.iter().map(i64::to_string).collect()
    }

    /// A merged intersection comes back smallest first, which is the order the
    /// probe already produced on these operands, so a client cannot tell which
    /// plan ran.
    #[test]
    fn a_merged_intersection_is_ascending_and_so_was_the_probe() {
        let a = ints(&[900, 5, 40, 7, 1000, 3]);
        let b = ints(&[1000, 3, 900, 8, 5]);
        let got = run(|f| inter(&[&a, &b], 0, f));
        assert_eq!(got, vec!["3", "5", "900", "1000"]);
        assert_eq!(run(|f| inter_with(Plan::Probe, &[&a, &b], 0, f)), got);
    }

    /// `SINTERCARD` stops early on the merge too, and stopping early does not
    /// change what it had already handed over.
    #[test]
    fn a_limit_stops_a_merged_intersection_early() {
        let vals: Vec<i64> = (0..2_000).collect();
        let a = ints(&vals);
        let b = ints(&vals);
        assert_eq!(plan_for(&[&a, &b]), Plan::Merge);
        assert_eq!(run(|f| inter(&[&a, &b], 3, f)), vec!["0", "1", "2"]);
        assert_eq!(run(|f| inter(&[&a, &b], 0, f)).len(), 2_000);
        assert_eq!(run(|f| inter(&[&a], 3, f)), vec!["0", "1", "2"]);
    }

    /// One set that is not an intset takes the whole operation back to a probe,
    /// because there is nothing to walk in step with a table.
    #[test]
    fn one_unsorted_operand_takes_everything_back_to_a_probe() {
        let a = ints(&[1, 2, 3]);
        let b = tabled(&["2", "3", "4"]);
        assert_eq!(plan_for(&[&a, &b]), Plan::Probe);
        assert_eq!(run(|f| inter(&[&a, &b], 0, f)), vec!["2", "3"]);
        // And asking for the merge anyway gets the right answer rather than a
        // wrong one, because the fact beats the preference.
        assert_eq!(
            run(|f| inter_with(Plan::Merge, &[&a, &b], 0, f)),
            vec!["2", "3"]
        );
    }

    /// A set past `set-max-intset-entries` is still an intset here, so it still
    /// merges. Before #148 it was a table by this size and this test would have
    /// been measuring the probe.
    #[test]
    fn a_set_past_the_intset_ceiling_still_merges() {
        let a: Set = {
            let mut s = Set::new();
            for i in 0..5_000i64 {
                s.add(i.to_string().as_bytes(), &Limits::DEFAULT);
            }
            s
        };
        assert_eq!(a.encoding(), Encoding::Hashtable, "the word a server uses");
        assert!(a.ints().is_some(), "and an intset underneath it");
        let b = ints(&[4_998, 4_999, 5_000]);
        assert_eq!(plan_for(&[&a, &b]), Plan::Merge);
        assert_eq!(run(|f| inter(&[&a, &b], 0, f)), vec!["4998", "4999"]);
    }

    /// The members are the same bytes whatever they contain, and a set holds
    /// arbitrary bytes rather than text.
    #[test]
    fn members_that_are_not_text_work_the_same() {
        let a = of([&b"\x00\xff"[..], b"\xc3\x28", b""]);
        let b = of([&b"\xc3\x28"[..], b""]);
        let mut got: Vec<Vec<u8>> = Vec::new();
        let n = inter(&[&a, &b], 0, |m| got.push(m.to_vec()));
        assert_eq!(n, 2);
        assert_eq!(got, vec![b"\xc3\x28".to_vec(), b"".to_vec()]);
    }
}
