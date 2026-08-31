//! Sorted set algebra, and how a result gets into order without a descent per
//! member.
//!
//! `ZUNION`, `ZINTER`, `ZDIFF`, `ZINTERCARD` and the three `*STORE` forms. The
//! set algebra next door is about which members come out. This one is about that
//! and about what score each of them comes out with, which is where the work is:
//! a member in three of the inputs has three scores, and `WEIGHTS` and
//! `AGGREGATE` say how those become one.
//!
//! # A set is a sorted set where every score is one
//!
//! `ZUNIONSTORE d 2 zs plain` is legal and every one of these commands takes
//! either type. That is not a special case bolted on, it is Redis's rule and it
//! falls out of an [`Operand`] which answers the two questions the algebra asks,
//! what members are in you and what score do you give this one, whichever it is
//! holding.
//!
//! # Order comes last, once, and costs nothing extra
//!
//! The obvious way to build a union is to add each member to a result sorted set
//! as it is found, which is a tree descent per member and a second one every time
//! a later input raises a score that is already in there. For a union of four
//! sets of a hundred thousand that is somewhere over half a million descents to
//! produce four hundred thousand members.
//!
//! So nothing is ordered while it is being worked out. Every operation
//! accumulates into an [`Elements<f64>`], which is the same member to score table
//! a sorted set is half made of and which knows nothing about order, so a member
//! appearing again is a hash probe and a float operation and no more than that.
//! Ordering happens once at the end, in [`Zset::from_elements`], which sorts row
//! numbers and then fills the tree by appending, and the append case is the one
//! the tree is fastest at.
//!
//! The part worth noticing is that the accumulator is not copied into the result.
//! It becomes the result. The member bytes are written once, when the first input
//! that has that member is walked, and they are never moved again.
//!
//! # Probe or accumulate
//!
//! `SINTER` measured probing as faster at every number of inputs and the same
//! argument holds here, so [`gather`] walks the smallest input and asks the
//! others. `ZDIFF` walks the first and asks the others, because the first is the
//! only one whose members can be in the answer. `ZUNION` has no choice: every
//! member of every input is in the answer, so all of them are walked.
//!
//! # NaN
//!
//! Redis turns one into a zero rather than refusing the command, which is worth
//! knowing because there are two ways to get one and both are reachable from
//! ordinary arguments. A weight of zero against a score of infinity is a NaN, and
//! so is a sum of two infinities of opposite sign. Neither can be stored, since a
//! NaN compares equal to nothing including itself, so both become zero here.

use yo_common::num::DIGITS_MAX;

use crate::elem::Elements;
use crate::listpack::Entry;
use crate::set::Set;
use crate::zset::Zset;

/// How the scores of a member that is in more than one input become one score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregate {
    /// Add them up. `AGGREGATE SUM`, and the default.
    #[default]
    Sum,
    /// Keep the lowest. `AGGREGATE MIN`.
    Min,
    /// Keep the highest. `AGGREGATE MAX`.
    Max,
}

/// Which algebra is being done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Every member of every input. `ZUNION`.
    Union,
    /// Only members in all of them. `ZINTER`.
    Inter,
    /// Members in the first and in none of the rest. `ZDIFF`.
    Diff,
}

/// One input to a sorted set operation.
///
/// A plain set is an operand because Redis says it is, and it behaves as a
/// sorted set in which every member scores one.
#[derive(Debug, Clone, Copy)]
pub enum Operand<'a> {
    /// A sorted set, with the scores it holds.
    Zset(&'a Zset),
    /// A plain set, where every member scores one.
    Set(&'a Set),
    /// A key that is not there, which is an empty input and not an error.
    Missing,
}

impl Operand<'_> {
    /// How many members are in this input.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Operand::Zset(z) => z.len(),
            Operand::Set(s) => s.len(),
            Operand::Missing => 0,
        }
    }

    /// Whether this input has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The score this input gives a member, or `None` if it does not have it.
    fn score(&self, member: &[u8]) -> Option<f64> {
        match self {
            Operand::Zset(z) => z.score(member),
            Operand::Set(s) => s.contains(member).then_some(1.0),
            Operand::Missing => None,
        }
    }

    /// Hand every member and its score over, in whatever order is cheapest.
    fn walk<F: FnMut(&[u8], f64)>(&self, mut f: F) {
        let mut digits = [0u8; DIGITS_MAX];
        match self {
            Operand::Zset(z) => z.walk(0, z.len(), false, |m, s| f(bytes(m, &mut digits), s)),
            Operand::Set(s) => {
                for m in s.iter() {
                    f(bytes(m, &mut digits), 1.0);
                }
            }
            Operand::Missing => {}
        }
    }
}

/// The bytes of a member, which for one stored as an integer are the buffer.
#[inline]
fn bytes<'a>(m: Entry<'a>, digits: &'a mut [u8; DIGITS_MAX]) -> &'a [u8] {
    match m {
        Entry::Str(s) => s,
        Entry::Int(n) => yo_common::num::i64_digits(digits, n),
    }
}

/// A score with its weight applied, with Redis's rule for a NaN.
///
/// A weight of zero against an infinite score is the reachable case, and it is
/// reachable from `ZUNIONSTORE d 2 a b WEIGHTS 0 1` against a set holding an
/// infinite score, which is not an exotic thing to write.
#[inline]
fn weighted(score: f64, weight: f64) -> f64 {
    let v = score * weight;
    if v.is_nan() { 0.0 } else { v }
}

/// Fold a second score into one already held.
#[inline]
fn fold(now: f64, next: f64, agg: Aggregate) -> f64 {
    match agg {
        Aggregate::Sum => {
            let v = now + next;
            if v.is_nan() { 0.0 } else { v }
        }
        Aggregate::Min => {
            if next < now {
                next
            } else {
                now
            }
        }
        Aggregate::Max => {
            if next > now {
                next
            } else {
                now
            }
        }
    }
}

/// Work out an operation and answer the member to score table it produced.
///
/// The table is unordered, because ordering it costs a sort and no operation
/// needs one until it is about to be read. [`Zset::from_elements`] is what turns
/// it into something with ranks.
///
/// `weights` is either empty, meaning every input counts once, or one weight per
/// input. A shorter list than that is the caller's bug and the missing ones count
/// as one.
#[must_use]
pub fn gather(op: Op, inputs: &[Operand<'_>], weights: &[f64], agg: Aggregate) -> Elements<f64> {
    let weight = |i: usize| weights.get(i).copied().unwrap_or(1.0);
    match op {
        Op::Union => {
            let mut out = Elements::with_capacity(hint(inputs, op));
            for (i, input) in inputs.iter().enumerate() {
                let w = weight(i);
                input.walk(|member, score| {
                    let v = weighted(score, w);
                    // The first input to hold a member writes its bytes. Every
                    // later one that holds it touches the score and nothing else.
                    match out.get_mut(member) {
                        Some(now) => *now = fold(*now, v, agg),
                        None => {
                            let _ = out.insert(member, v);
                        }
                    }
                });
            }
            out
        }
        Op::Inter => {
            // The smallest input, because a member that is not in it cannot be
            // in the answer and walking any larger one asks more questions for
            // the same result.
            let Some(small) = (0..inputs.len()).min_by_key(|&i| inputs[i].len()) else {
                return Elements::with_capacity(0);
            };
            let mut out = Elements::with_capacity(inputs[small].len().clamp(16, 1 << 16));
            if inputs.iter().any(Operand::is_empty) {
                return out;
            }
            inputs[small].walk(|member, score| {
                let mut total = weighted(score, weight(small));
                for (i, other) in inputs.iter().enumerate() {
                    if i == small {
                        continue;
                    }
                    // A member missing from any input ends the questions for
                    // that member rather than the ones for the rest of them.
                    let Some(s) = other.score(member) else { return };
                    total = fold(total, weighted(s, weight(i)), agg);
                }
                let _ = out.insert(member, total);
            });
            out
        }
        Op::Diff => {
            let Some((first, rest)) = inputs.split_first() else {
                return Elements::with_capacity(0);
            };
            let mut out = Elements::with_capacity(first.len().clamp(16, 1 << 16));
            first.walk(|member, score| {
                if rest.iter().any(|o| o.score(member).is_some()) {
                    return;
                }
                // No weight and no aggregate. `ZDIFF` takes neither, because
                // every member in its answer came from exactly one input.
                let _ = out.insert(member, score);
            });
            out
        }
    }
}

/// `ZINTERCARD numkeys key [key ...] [LIMIT limit]`.
///
/// Counting only, so nothing is stored and no score is worked out. A limit stops
/// the walk as soon as it is reached, which is the only reason the command exists
/// separately from `ZINTER` with the members thrown away.
#[must_use]
pub fn intercard(inputs: &[Operand<'_>], limit: usize) -> usize {
    if inputs.is_empty() || inputs.iter().any(Operand::is_empty) {
        return 0;
    }
    let small = (0..inputs.len())
        .min_by_key(|&i| inputs[i].len())
        .expect("not empty");
    let stop = if limit == 0 { usize::MAX } else { limit };
    let mut found = 0;
    inputs[small].walk(|member, _| {
        if found >= stop {
            return;
        }
        if inputs
            .iter()
            .enumerate()
            .all(|(i, o)| i == small || o.score(member).is_some())
        {
            found += 1;
        }
    });
    found
}

/// How much room a result is likely to want.
///
/// A union is at most every member of every input and usually far fewer, so this
/// is an over estimate that costs one allocation to be wrong about, against a
/// growth every time it doubles if it is under.
fn hint(inputs: &[Operand<'_>], op: Op) -> usize {
    let total: usize = match op {
        Op::Union => inputs.iter().map(Operand::len).sum(),
        _ => inputs.first().map_or(0, Operand::len),
    };
    total.clamp(16, 1 << 20)
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use super::*;
    use crate::set::Limits as SetLimits;
    use crate::zset::Limits;

    /// The order a sorted set is in: score first, then the member bytes.
    ///
    /// The tests work the answer out the slow way and then put it in this order
    /// to compare, which is the only place in this file that needs it. The
    /// module itself never orders anything, because that is
    /// [`Zset::from_elements`]'s job and doing it here as well would be two
    /// implementations of one rule.
    fn cmp_key(a: (f64, &[u8]), b: (f64, &[u8])) -> Ordering {
        match a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.1.cmp(b.1),
            other => other,
        }
    }

    fn zs(pairs: &[(&str, f64)]) -> Zset {
        let mut z = Zset::new();
        for (m, s) in pairs {
            z.add(m.as_bytes(), *s, &Limits::DEFAULT);
        }
        z
    }

    fn plain(members: &[&str]) -> Set {
        let mut s = Set::new();
        for m in members {
            s.add(m.as_bytes(), &SetLimits::DEFAULT);
        }
        s
    }

    /// The result in rank order, which is what every caller of this actually
    /// wants and is the only way to compare two of them.
    fn ordered(got: Elements<f64>) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = (0..got.len())
            .map(|i| got.at(i).expect("in range"))
            .map(|(n, s)| (String::from_utf8(n.to_vec()).unwrap(), *s))
            .collect();
        out.sort_by(|a, b| cmp_key((a.1, a.0.as_bytes()), (b.1, b.0.as_bytes())));
        out
    }

    fn named(got: Vec<(String, f64)>) -> Vec<String> {
        got.into_iter().map(|(m, _)| m).collect()
    }

    #[test]
    fn a_union_adds_the_scores_of_a_member_in_both() {
        let a = zs(&[("x", 1.0), ("y", 2.0)]);
        let b = zs(&[("y", 3.0), ("z", 4.0)]);
        let got = ordered(gather(
            Op::Union,
            &[Operand::Zset(&a), Operand::Zset(&b)],
            &[],
            Aggregate::Sum,
        ));
        assert_eq!(
            got,
            [("x".into(), 1.0), ("z".into(), 4.0), ("y".into(), 5.0)]
        );
    }

    #[test]
    fn min_and_max_keep_one_score_rather_than_adding_them() {
        let a = zs(&[("y", 2.0)]);
        let b = zs(&[("y", 7.0)]);
        let ops = [Operand::Zset(&a), Operand::Zset(&b)];
        assert_eq!(
            ordered(gather(Op::Union, &ops, &[], Aggregate::Min)),
            [("y".to_string(), 2.0)]
        );
        assert_eq!(
            ordered(gather(Op::Union, &ops, &[], Aggregate::Max)),
            [("y".to_string(), 7.0)]
        );
    }

    #[test]
    fn weights_multiply_before_anything_is_aggregated() {
        let a = zs(&[("x", 1.0), ("y", 2.0)]);
        let b = zs(&[("y", 3.0)]);
        let ops = [Operand::Zset(&a), Operand::Zset(&b)];
        let got = ordered(gather(Op::Union, &ops, &[2.0, 10.0], Aggregate::Sum));
        assert_eq!(got, [("x".into(), 2.0), ("y".into(), 34.0)]);
        // MIN sees the weighted scores and not the raw ones, so the input with
        // the larger raw score can still be the one that wins.
        let got = ordered(gather(Op::Union, &ops, &[2.0, 0.5], Aggregate::Min));
        assert_eq!(got, [("y".into(), 1.5), ("x".into(), 2.0)]);
    }

    #[test]
    fn an_intersection_only_keeps_what_every_input_has() {
        let a = zs(&[("x", 1.0), ("y", 2.0), ("z", 3.0)]);
        let b = zs(&[("y", 10.0), ("z", 20.0)]);
        let c = zs(&[("z", 100.0)]);
        let ops = [Operand::Zset(&a), Operand::Zset(&b), Operand::Zset(&c)];
        assert_eq!(
            ordered(gather(Op::Inter, &ops, &[], Aggregate::Sum)),
            [("z".to_string(), 123.0)]
        );
        assert_eq!(intercard(&ops, 0), 1);
        // An empty input anywhere is an empty intersection.
        let ops = [Operand::Zset(&a), Operand::Missing];
        assert!(ordered(gather(Op::Inter, &ops, &[], Aggregate::Sum)).is_empty());
        assert_eq!(intercard(&ops, 0), 0);
    }

    #[test]
    fn a_difference_keeps_the_first_input_scores() {
        let a = zs(&[("x", 1.0), ("y", 2.0), ("z", 3.0)]);
        let b = zs(&[("y", 99.0)]);
        let ops = [Operand::Zset(&a), Operand::Zset(&b)];
        assert_eq!(
            ordered(gather(Op::Diff, &ops, &[], Aggregate::Sum)),
            [("x".into(), 1.0), ("z".into(), 3.0)]
        );
        // A first input that is not there is an empty answer whatever the rest
        // hold, and a later one that is not there takes nothing away.
        assert!(
            ordered(gather(
                Op::Diff,
                &[Operand::Missing, Operand::Zset(&a)],
                &[],
                Aggregate::Sum
            ))
            .is_empty()
        );
        let ops = [Operand::Zset(&a), Operand::Missing];
        assert_eq!(
            named(ordered(gather(Op::Diff, &ops, &[], Aggregate::Sum))),
            ["x", "y", "z"]
        );
    }

    #[test]
    fn a_plain_set_counts_as_a_sorted_set_where_every_score_is_one() {
        let a = zs(&[("x", 5.0), ("y", 6.0)]);
        let b = plain(&["y", "z"]);
        let ops = [Operand::Zset(&a), Operand::Set(&b)];
        let got = ordered(gather(Op::Union, &ops, &[], Aggregate::Sum));
        assert_eq!(
            got,
            [("z".into(), 1.0), ("x".into(), 5.0), ("y".into(), 7.0)]
        );
        assert_eq!(
            ordered(gather(Op::Inter, &ops, &[], Aggregate::Sum)),
            [("y".to_string(), 7.0)]
        );
        assert_eq!(
            named(ordered(gather(Op::Diff, &ops, &[], Aggregate::Sum))),
            ["x"]
        );
    }

    /// An intset holds numbers and a table holds their digits, so a member only
    /// crosses between the two if the walk hands over bytes either way.
    #[test]
    fn an_integer_member_crosses_between_a_set_and_a_sorted_set() {
        let a = zs(&[("17", 5.0), ("42", 6.0)]);
        let b = plain(&["42", "99"]);
        assert_eq!(b.encoding().name(), "intset");
        let ops = [Operand::Zset(&a), Operand::Set(&b)];
        assert_eq!(
            ordered(gather(Op::Inter, &ops, &[], Aggregate::Sum)),
            [("42".to_string(), 7.0)]
        );
        assert_eq!(intercard(&ops, 0), 1);
        assert_eq!(
            named(ordered(gather(Op::Union, &ops, &[], Aggregate::Sum))),
            ["99", "17", "42"]
        );
    }

    #[test]
    fn a_score_that_would_be_a_nan_becomes_a_zero() {
        // A weight of zero against an infinity.
        let a = zs(&[("x", f64::INFINITY)]);
        let ops = [Operand::Zset(&a)];
        assert_eq!(
            ordered(gather(Op::Union, &ops, &[0.0], Aggregate::Sum)),
            [("x".to_string(), 0.0)]
        );
        // Two infinities of opposite sign, added.
        let b = zs(&[("x", f64::NEG_INFINITY)]);
        let ops = [Operand::Zset(&a), Operand::Zset(&b)];
        assert_eq!(
            ordered(gather(Op::Union, &ops, &[], Aggregate::Sum)),
            [("x".to_string(), 0.0)]
        );
    }

    #[test]
    fn a_limit_stops_a_cardinality_count_where_it_was_told_to() {
        let a = zs(&[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]);
        let b = zs(&[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]);
        let ops = [Operand::Zset(&a), Operand::Zset(&b)];
        assert_eq!(intercard(&ops, 0), 4);
        assert_eq!(intercard(&ops, 2), 2);
        assert_eq!(intercard(&ops, 99), 4);
        assert_eq!(intercard(&[], 0), 0);
    }

    #[test]
    fn a_union_of_thousands_agrees_with_the_slow_way_of_working_it_out() {
        let one: Vec<(String, f64)> = (0..3_000)
            .map(|i| (format!("m{i:05}"), f64::from(i)))
            .collect();
        let two: Vec<(String, f64)> = (1_500..4_500)
            .map(|i| (format!("m{i:05}"), f64::from(i) * 2.0))
            .collect();
        let mut a = Zset::new();
        for (m, s) in &one {
            a.add(m.as_bytes(), *s, &Limits::DEFAULT);
        }
        let mut b = Zset::new();
        for (m, s) in &two {
            b.add(m.as_bytes(), *s, &Limits::DEFAULT);
        }
        let ops = [Operand::Zset(&a), Operand::Zset(&b)];

        let mut want: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        for (m, s) in one.iter().chain(two.iter()) {
            *want.entry(m.clone()).or_insert(0.0) += s;
        }
        let mut want: Vec<(String, f64)> = want.into_iter().collect();
        want.sort_by(|x, y| cmp_key((x.1, x.0.as_bytes()), (y.1, y.0.as_bytes())));
        assert_eq!(ordered(gather(Op::Union, &ops, &[], Aggregate::Sum)), want);
        assert_eq!(intercard(&ops, 0), 1_500);
    }
}
