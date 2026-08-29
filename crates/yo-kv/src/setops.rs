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
//! # What would actually change it
//!
//! `08` section 4's merge is not this. It is per-partition sorted arrays walked in
//! lockstep, measured at 12 ms against 450 ms for sorting per call, and its touch
//! is a pointer step and a comparison with no hash anywhere. That is a touch which
//! genuinely is much cheaper than a probe question, and against that a crossover
//! can exist. It needs the partitioned band, where partitions are held in order,
//! and that band is not built yet.
//!
//! So [`inter`] probes, with no chooser in front of it. A chooser that always
//! returns the same answer is worse than no chooser, and one written now would be
//! fitted to a plan that is not the plan it will eventually choose between.
//! [`inter_with`] keeps the counting plan reachable and the benchmark keeps
//! measuring it, so that the sorted merge has a control to beat when it lands.
//!
//! # Ordering
//!
//! Every operation returns members in the order the first relevant set holds
//! them, which is insertion order for a listpack or a table and ascending for an
//! intset. Redis makes no ordering promise for any of these, and picking the
//! order the data is already in means the walk is sequential and there is
//! nothing to sort.
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

use crate::Elements;
use crate::set::{Limits, Needle, Set};

/// How to answer a set operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Walk the smallest set and question the others about each member.
    Probe,
    /// Walk everything once into one counting table.
    Accumulate,
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
/// Always a probe, for the reason in the module doc. When the partitioned band
/// gives us something sorted to merge there will be a choice to make here, and
/// until then there is not one.
pub fn inter<F>(sets: &[&Set], limit: usize, f: F) -> usize
where
    F: FnMut(&[u8]),
{
    inter_with(Plan::Probe, sets, limit, f)
}

/// The same, with the plan named rather than assumed.
///
/// This is how the benchmark runs both plans over the same sets, which is the
/// only way to find out where they cross and the only way to check that they
/// agree on the answer. It is public because a caller that knows the shape of its
/// own data knows more about it than [`inter`] can see from the sets alone.
pub fn inter_with<F>(how: Plan, sets: &[&Set], limit: usize, f: F) -> usize
where
    F: FnMut(&[u8]),
{
    if sets.is_empty() || sets.iter().any(|s| s.is_empty()) {
        return 0;
    }
    match how {
        Plan::Probe => inter_probe(sets, limit, f),
        Plan::Accumulate => inter_accumulate(sets, limit, f),
    }
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

/// Every member of any of the sets, each once, in the order they are met.
///
/// There is no plan to choose here. A union has to read every member of every
/// set whatever it does, so the only question is what it does with each one, and
/// the answer is one insertion into a table that is also the duplicate check.
pub fn union<F>(sets: &[&Set], mut f: F) -> usize
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
/// Always a probe, and never anything else. The first set is the one being
/// walked whether we like it or not, so the only choice is how each member is
/// checked, and a member that is in the second set is never asked about the
/// third.
pub fn diff<F>(sets: &[&Set], mut f: F) -> usize
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
        let bands = [
            ("intset", AS_INTSET),
            ("listpack", AS_LISTPACK),
            ("table", AS_TABLE),
        ];
        for (ln, left) in bands {
            for (rn, right) in bands {
                let a = banded(&names, &left);
                let b = banded(&others, &right);
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
