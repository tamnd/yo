//! `SORT` and `SORT_RO`, the one keyspace command that reads keys nobody named.
//!
//! Everything else in [`crate::keyspace`] asks a question about the key in front
//! of it. `SORT mylist BY weight_* GET data_*` reads `mylist`, then reads a
//! different key for every element in it to decide the order, then reads another
//! one for every element to decide what to answer with. That is why it is the
//! command Redis marks as not deterministic, why `SORT_RO` exists at all, and
//! why it lives in a file of its own instead of as another arm of the keyspace
//! match.
//!
//! # What it sorts
//!
//! A list, a set or a sorted set. Anything else is `WRONGTYPE` and a missing key
//! is an empty answer, not an error.
//!
//! # The four ways it can order things
//!
//! Numerically on the elements, which is the default and which fails the whole
//! command if any element is not a number. Numerically on a `BY` lookup, where
//! an element whose lookup missed scores zero. Alphabetically on the elements.
//! Alphabetically on a `BY` lookup, where a miss sorts before every hit.
//!
//! Ties never fall through to whatever order the elements arrived in. Two equal
//! scores are broken by comparing the elements themselves, which is what makes
//! the answer the same on two servers holding the same set, and a set has no
//! order of its own to fall back on anyway.
//!
//! # Not sorting
//!
//! A `BY` pattern with no `*` in it cannot name a different key per element, so
//! Redis reads it as an instruction not to sort rather than as a pattern that
//! resolves to one key for everything. `BY nosort` is the idiom and there is
//! nothing special about the word.
//!
//! There is one hole in that, and it is the reason the `store` flag is in
//! [`Sort`] rather than being left to the caller. A set has no order, so
//! `SORT myset BY nosort` may answer its members in any order at all, which is
//! fine for a client that asked for exactly that and is not fine for a `STORE`,
//! because then two servers that agree about the set disagree about the list
//! they wrote. So a `BY nosort` over a set with a `STORE` sorts alphabetically
//! after all. Redis does the same thing for the same reason, and also does it
//! for a call from a script, which we do not, because a script here reaches the
//! same method any other caller does.
//!
//! # Patterns
//!
//! A pattern is not a glob. The first `*` in it is replaced with the element and
//! the result is a key name, so `weight_*` on the element `a` reads `weight_a`.
//! A pattern with no `*` looks up nothing and misses every time. `#` on its own
//! means the element itself, which is only useful in a `GET`.
//!
//! A pattern may reach into a hash instead of at a string, with `->` after the
//! `*`: `h_*->field` reads the field `field` of the hash `h_<element>`. The
//! split is on the first `->` that comes after the `*` and there has to be at
//! least one byte after it, so a pattern ending in `->` is a key name with a
//! `->` on the end and not a hash lookup with an empty field.
//!
//! A lookup that lands on a key of the wrong type is a miss and not an error.
//! `BY h_*` over keys holding lists gives every element a weight of nothing,
//! which sorts them all equal and lets the tie break do the work.
//!
//! # Stripes
//!
//! This is the only thing in the store that takes a whole database rather than
//! one keyspace, and it is why. The key it sorts is on one stripe, the key a
//! `BY` names for an element is on whichever stripe that name lands on, the key
//! a `GET` names is on another, and a `STORE` destination is on a fourth. None
//! of those can be worked out before the command runs, because the names are
//! built out of the elements, so there is nothing here to route once at the top
//! the way every other command is routed.
//!
//! So the stripes are taken before the command starts rather than one at a
//! time as the names appear. A sort with a pattern in it takes every stripe of
//! the database, because the names are built out of the elements and the
//! stripes those land on are, between them, all of them. A sort without one
//! takes the one stripe its key is on, and the two stripes its key and its
//! destination are on when there is a `STORE`.
//!
//! # Divergence
//!
//! Redis compares strings with `strcoll` when there is no `STORE`, and with a
//! byte compare when there is, because a stored result has to be the same on
//! every replica and `strcoll` answers to `LC_COLLATE`. Redis calls
//! `setlocale(LC_COLLATE, "")` at startup, so the order `SORT ... ALPHA` puts
//! two strings in depends on the environment the server was started with, and
//! the same server started twice can answer differently.
//!
//! This compares bytes always. That is the `STORE` behaviour applied everywhere,
//! it is what a client gets from Redis under the C locale, and it is the only
//! choice that makes the answer a property of the data. It also means a member
//! with a zero byte in it sorts on all of its bytes, where `strcoll` stops at
//! the zero. Registered in `divergences.toml`.

use crate::db::{Db, Holds};
use crate::keyspace::wrong_type;
use crate::value::Kind;
use crate::zsets::Window;
use std::cmp::Ordering;
use yo_common::{Code, Error, Result, num};

/// What Redis says when a numeric sort meets an element that is not a number.
const NOT_A_DOUBLE: &str = "One or more scores can't be converted into double";

/// What `SORT` was asked to do, everything except the key and the destination.
///
/// Borrowed from the caller's arguments rather than owned, because on the wire
/// path every field of this is a slice of the command that is already in memory
/// and copying them would be copying the command.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sort<'a> {
    /// The `BY` pattern, if there was one.
    pub by: Option<&'a [u8]>,
    /// The `GET` patterns, in the order they were given, `#` included.
    pub get: &'a [&'a [u8]],
    /// `LIMIT offset count`, if there was one.
    ///
    /// Both halves are signed because both halves are on the wire as integers
    /// and Redis takes them without complaint. A negative offset is clamped to
    /// the front and a negative count means everything from the offset on.
    pub limit: Option<(i64, i64)>,
    /// `DESC`, which reverses whatever order the rest of this produced.
    pub desc: bool,
    /// `ALPHA`, which compares bytes where the default compares numbers.
    pub alpha: bool,
    /// Whether the answer is going into a key rather than back to the caller.
    ///
    /// Set by [`Db::sort_store`] and not by the caller. It is in here
    /// rather than a separate argument because it changes the ordering and not
    /// just what happens to the result. See the module doc.
    pub store: bool,
}

/// One element on its way through the sort, with whatever it is ordered by.
///
/// The weight is one of the two, never both, and which one is decided once for
/// the whole command rather than per element.
struct Weighted {
    /// The element itself, which is also the tie break and may be the answer.
    elem: Vec<u8>,
    /// The number to sort on, under a numeric sort.
    score: f64,
    /// The bytes to sort on, under an alphabetic sort with a `BY`. `None` is a
    /// lookup that missed and sorts before every hit.
    text: Option<Vec<u8>>,
}

impl Db {
    /// `SORT key [BY pattern] [LIMIT offset count] [GET pattern ...] [ASC|DESC]
    /// [ALPHA]`, and the whole of `SORT_RO`.
    ///
    /// One row per element that survived the `LIMIT`, or one row per `GET`
    /// pattern per element if there were any. A row is `None` where a `GET`
    /// pattern missed, which is a nil on the wire, and `GET #` never misses.
    ///
    /// This allocates the elements, which nothing else on the read path does. It
    /// has to: the ordering is decided by reading other keys, and reading
    /// another key needs the database that the elements are borrowed from. Redis
    /// has the same problem and solves it by holding refcounted pointers, which
    /// is the same copy with the copy moved to whoever wrote the value. The copy
    /// is also what lets a `BY` or a `GET` reach a stripe other than the one the
    /// elements came from.
    pub fn sort(&self, key: &[u8], opts: &Sort<'_>) -> Result<Vec<Option<Vec<u8>>>> {
        let mut opts = *opts;
        opts.store = false;
        let mut held = self.reach(key, None, &opts);
        self.sorted(&mut held, key, &opts)
    }

    /// `SORT key ... STORE destination`, which answers the length of the list it
    /// wrote.
    ///
    /// An empty result deletes the destination rather than leaving an empty list
    /// behind, because a list is never empty and a key that holds one that is
    /// would be a key `TYPE` answers `list` for and `LLEN` answers zero for.
    ///
    /// A `GET` pattern that missed stores an empty string, where the same miss
    /// sent to a client is a nil. There is no nil in a list, so this is the only
    /// thing it could be, and it is what Redis stores.
    ///
    /// The destination is on its own stripe, which is not generally the stripe
    /// the elements came from, and it is held from the start alongside the rest
    /// rather than reached for once the sort is done, so that nothing can write
    /// into it between the read and the store.
    pub fn sort_store(&self, key: &[u8], dest: &[u8], opts: &Sort<'_>) -> Result<usize> {
        let mut opts = *opts;
        opts.store = true;
        let mut held = self.reach(key, Some(dest), &opts);
        let rows = self.sorted(&mut held, key, &opts)?;
        let onto = self.stripe_of(dest);
        if rows.is_empty() {
            held.stripe_mut(onto).del(dest);
            return Ok(0);
        }
        // Written into a fresh key rather than appended to whatever was there,
        // and the delete comes first so that `SORT k STORE k` reads k, then
        // throws it away, then writes the answer. Redis is the same, and it is
        // the reason the elements had to be copied out before any of this.
        held.stripe_mut(onto).del(dest);
        let owned: Vec<Vec<u8>> = rows.into_iter().map(Option::unwrap_or_default).collect();
        held.stripe_mut(onto).push(
            dest,
            crate::lists::End::Right,
            owned.iter().map(Vec::as_slice),
        )
    }

    /// Every stripe this sort can touch, held, in stripe order.
    ///
    /// A `BY` or a `GET` with a `*` in it builds a key name out of each element,
    /// and which stripes those names land on cannot be known before the elements
    /// have been read. So a sort with a pattern in it takes the whole database
    /// and a sort without one takes the one or two stripes it does name. The
    /// wide form is the price of a command whose keys it cannot know in advance,
    /// and it is no wider than what a client already gets from Redis, where a
    /// sort holds the whole server for its whole run.
    fn reach(&self, key: &[u8], dest: Option<&[u8]>, opts: &Sort<'_>) -> Holds<'_> {
        if patterned(opts) {
            return self.hold_many(0..self.width());
        }
        let named = std::iter::once(self.stripe_of(key));
        self.hold_many(named.chain(dest.map(|d| self.stripe_of(d))))
    }

    /// The body both of them share.
    fn sorted(
        &self,
        held: &mut Holds<'_>,
        key: &[u8],
        opts: &Sort<'_>,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        let kind = held.stripe_mut(self.stripe_of(key)).kind_of(key);
        let elems = self.elements(held, key, kind)?;

        // A `BY` with no `*` cannot name a key per element, so it is an order to
        // leave things alone. The exception is the one the module doc explains.
        let mut alpha = opts.alpha;
        let mut by = opts.by;
        let mut dontsort = by.is_some_and(|p| !p.contains(&b'*'));
        if dontsort && kind == Some(Kind::Set) && opts.store {
            dontsort = false;
            alpha = true;
            by = None;
        }

        let ordered = if dontsort {
            // The natural order of whatever it is, backwards if `DESC` was
            // given. A list is head to tail, a sorted set is by score, and a set
            // has no order to reverse but reversing it costs nothing and keeps
            // this one branch instead of two.
            let mut e = elems;
            if opts.desc {
                e.reverse();
            }
            e
        } else {
            self.order(held, elems, by, alpha, opts.desc)?
        };

        let window = limit(ordered.len(), opts.limit);
        self.emit(held, &ordered[window], opts.get)
    }

    /// Copy the elements out of whatever holds them.
    ///
    /// Owned, for the reason [`Db::sort`] gives. A missing key is an empty
    /// list and not an error, so `SORT nosuchkey` answers nothing.
    fn elements(
        &self,
        held: &mut Holds<'_>,
        key: &[u8],
        kind: Option<Kind>,
    ) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        // One stripe for the whole of this, since it is all the same key.
        let db = held.stripe_mut(self.stripe_of(key));
        match kind {
            None => {}
            Some(Kind::List) => {
                for e in db.lrange(key, 0, -1)? {
                    let mut v = Vec::new();
                    e.write_to(&mut v);
                    out.push(v);
                }
            }
            Some(Kind::Set) => {
                if let Some(members) = db.smembers(key)? {
                    for m in members {
                        let mut v = Vec::new();
                        m.write_to(&mut v);
                        out.push(v);
                    }
                }
            }
            Some(Kind::Zset) => {
                let n = db.zcard(key)?;
                let w = Window {
                    from: 0,
                    count: n,
                    rev: false,
                };
                out.reserve(n);
                db.zwalk(key, w, |m, _| {
                    let mut v = Vec::new();
                    m.write_to(&mut v);
                    out.push(v);
                })?;
            }
            Some(_) => return Err(wrong_type()),
        }
        Ok(out)
    }

    /// Weigh every element and put them in order.
    fn order(
        &self,
        held: &mut Holds<'_>,
        elems: Vec<Vec<u8>>,
        by: Option<&[u8]>,
        alpha: bool,
        desc: bool,
    ) -> Result<Vec<Vec<u8>>> {
        let mut weighed = Vec::with_capacity(elems.len());
        for elem in elems {
            let looked = match by {
                Some(pattern) => self.by_pattern(held, pattern, &elem),
                None => None,
            };
            let (score, text) = if alpha {
                // Without a `BY` the element is its own sort key, and the tie
                // break already compares elements, so there is nothing to carry.
                (0.0, if by.is_some() { looked } else { None })
            } else {
                let raw = match by {
                    // A lookup that missed weighs nothing. Redis leaves the
                    // score at zero rather than failing, which means a numeric
                    // `BY` over keys that do not exist is a pure tie break.
                    Some(_) => match looked {
                        Some(v) => v,
                        None => {
                            weighed.push(Weighted {
                                elem,
                                score: 0.0,
                                text: None,
                            });
                            continue;
                        }
                    },
                    None => elem.clone(),
                };
                let n =
                    num::parse_f64(&raw).ok_or_else(|| Error::new(Code::Invalid, NOT_A_DOUBLE))?;
                if n.is_nan() {
                    return Err(Error::new(Code::Invalid, NOT_A_DOUBLE));
                }
                (n, None)
            };
            weighed.push(Weighted { elem, score, text });
        }

        // Stable is not needed, since the tie break is total, but it is what
        // `sort_by` gives and asking for the unstable one to save nothing would
        // be trading a guarantee for no gain.
        weighed.sort_by(|a, b| {
            let cmp = if alpha {
                match (&a.text, &b.text) {
                    // Both missing, or no `BY` at all, so the elements decide.
                    (None, None) => a.elem.cmp(&b.elem),
                    // A miss sorts before a hit.
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                    (Some(x), Some(y)) => x.cmp(y).then_with(|| a.elem.cmp(&b.elem)),
                }
            } else {
                // No NaN can reach here, so the partial compare is total.
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.elem.cmp(&b.elem))
            };
            if desc { cmp.reverse() } else { cmp }
        });
        Ok(weighed.into_iter().map(|w| w.elem).collect())
    }

    /// Build the answer, which is the elements themselves or a `GET` per element.
    fn emit(
        &self,
        held: &mut Holds<'_>,
        elems: &[Vec<u8>],
        get: &[&[u8]],
    ) -> Result<Vec<Option<Vec<u8>>>> {
        if get.is_empty() {
            return Ok(elems.iter().map(|e| Some(e.clone())).collect());
        }
        let mut out = Vec::with_capacity(elems.len() * get.len());
        for elem in elems {
            for pattern in get {
                if *pattern == b"#" {
                    out.push(Some(elem.clone()));
                } else {
                    out.push(self.by_pattern(held, pattern, elem));
                }
            }
        }
        Ok(out)
    }

    /// Read the key a pattern names for one element.
    ///
    /// `None` for a pattern with no `*`, for a key that is not there, and for a
    /// key that is there and holds the wrong type. The last one is a miss rather
    /// than an error on purpose: a pattern is a guess about a naming convention
    /// and one key that does not fit the convention should not fail a command
    /// over ten thousand elements.
    ///
    /// The name is built out of the element, so the stripe it lands on is not
    /// known until here and two elements of the same key are read from two
    /// different stripes as often as not. Every stripe is already held by then,
    /// which is what [`Db::reach`] is for.
    fn by_pattern(&self, held: &mut Holds<'_>, pattern: &[u8], elem: &[u8]) -> Option<Vec<u8>> {
        let star = pattern.iter().position(|&c| c == b'*')?;
        // The field split is looked for after the `*`, so a `->` in the prefix
        // is part of the key name. And there has to be something after it, so a
        // pattern ending in `->` names a key whose name ends in `->`.
        let arrow = pattern[star + 1..]
            .windows(2)
            .position(|w| w == b"->")
            .map(|i| star + 1 + i)
            .filter(|&i| i + 2 < pattern.len());

        let (key_part, field) = match arrow {
            Some(i) => (&pattern[..i], Some(&pattern[i + 2..])),
            None => (pattern, None),
        };

        let mut key = Vec::with_capacity(key_part.len() + elem.len());
        key.extend_from_slice(&key_part[..star]);
        key.extend_from_slice(elem);
        key.extend_from_slice(&key_part[star + 1..]);

        let stripe = held.stripe_mut(self.stripe_of(&key));
        match field {
            Some(f) => stripe
                .hget(&key, f, |t| {
                    t.map(|t| {
                        let mut v = Vec::new();
                        t.write_to(&mut v);
                        v
                    })
                })
                .unwrap_or(None),
            None => stripe.get(&key).ok().flatten().map(|s| s.to_vec()),
        }
    }
}

/// Whether this sort can name a key that was not given on the wire.
///
/// A `BY` with no `*` names nothing, and neither does a `GET #`, which is the
/// element itself. Anything else with a `*` in it is a key per element.
fn patterned(opts: &Sort<'_>) -> bool {
    opts.by.is_some_and(|p| p.contains(&b'*')) || opts.get.iter().any(|&p| p != b"#")
}

/// Which slice of the sorted elements the `LIMIT` asked for.
///
/// Redis clamps rather than complains at every edge: a negative offset is the
/// front, a negative count is everything left, an offset past the end is an
/// empty answer and a count that runs past the end stops at it.
fn limit(len: usize, limit: Option<(i64, i64)>) -> std::ops::Range<usize> {
    let Some((offset, count)) = limit else {
        return 0..len;
    };
    let start = usize::try_from(offset).unwrap_or(0).min(len);
    let end = if count < 0 {
        len
    } else {
        start
            .saturating_add(usize::try_from(count).unwrap_or(0))
            .min(len)
    };
    start..end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lists::End;
    use crate::strings::SetOptions;
    use crate::zsets::ZAdd;

    /// The answer as flat bytes, with a missed `GET` written as the word `nil`,
    /// which no test here stores as a value.
    fn flat(rows: Vec<Option<Vec<u8>>>) -> Vec<String> {
        rows.into_iter()
            .map(|r| match r {
                Some(v) => String::from_utf8_lossy(&v).into_owned(),
                None => "nil".to_string(),
            })
            .collect()
    }

    /// One list element as a string, for reading back what a `STORE` wrote.
    fn text(e: crate::listpack::Entry<'_>) -> String {
        let mut v = Vec::new();
        e.write_to(&mut v);
        String::from_utf8_lossy(&v).into_owned()
    }

    fn list(db: &mut Db, key: &[u8], items: &[&str]) {
        db.at(key)
            .push(key, End::Right, items.iter().map(|s| s.as_bytes()))
            .expect("a fresh list takes elements");
    }

    #[test]
    fn numbers_sort_as_numbers_and_not_as_text() {
        let mut db = Db::new();
        list(&mut db, b"l", &["10", "9", "100", "1"]);
        let opts = Sort::default();
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["1", "9", "10", "100"]);
        let alpha = Sort {
            alpha: true,
            ..Sort::default()
        };
        assert_eq!(
            flat(db.sort(b"l", &alpha).unwrap()),
            ["1", "10", "100", "9"]
        );
    }

    #[test]
    fn an_element_that_is_not_a_number_fails_the_whole_command() {
        let mut db = Db::new();
        list(&mut db, b"l", &["1", "two", "3"]);
        let err = db.sort(b"l", &Sort::default()).unwrap_err();
        assert_eq!(err.message(), NOT_A_DOUBLE);
        // And the same elements under ALPHA are fine, which is the whole point
        // of the option.
        let alpha = Sort {
            alpha: true,
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &alpha).unwrap()), ["1", "3", "two"]);
    }

    #[test]
    fn desc_reverses_and_limit_takes_a_window_of_what_is_left() {
        let mut db = Db::new();
        list(&mut db, b"l", &["3", "1", "5", "2", "4"]);
        let opts = Sort {
            desc: true,
            limit: Some((1, 2)),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["4", "3"]);
        // A negative count is everything from the offset on, and an offset past
        // the end is nothing at all.
        let rest = Sort {
            limit: Some((3, -1)),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &rest).unwrap()), ["4", "5"]);
        let past = Sort {
            limit: Some((99, 5)),
            ..Sort::default()
        };
        assert!(db.sort(b"l", &past).unwrap().is_empty());
        let before = Sort {
            limit: Some((-4, 2)),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &before).unwrap()), ["1", "2"]);
    }

    #[test]
    fn by_reads_a_key_for_every_element() {
        let mut db = Db::new();
        list(&mut db, b"l", &["a", "b", "c"]);
        db.at(b"w_a").set(b"w_a", b"3", SetOptions::PLAIN).unwrap();
        db.at(b"w_b").set(b"w_b", b"1", SetOptions::PLAIN).unwrap();
        db.at(b"w_c").set(b"w_c", b"2", SetOptions::PLAIN).unwrap();
        let opts = Sort {
            by: Some(b"w_*"),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["b", "c", "a"]);
    }

    #[test]
    fn a_by_lookup_that_missed_weighs_nothing_and_the_element_breaks_the_tie() {
        let mut db = Db::new();
        list(&mut db, b"l", &["c", "a", "b"]);
        db.at(b"w_b").set(b"w_b", b"5", SetOptions::PLAIN).unwrap();
        let opts = Sort {
            by: Some(b"w_*"),
            ..Sort::default()
        };
        // `a` and `c` both weigh zero, so they come first in element order, and
        // `b` weighs five and comes last.
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["a", "c", "b"]);
    }

    #[test]
    fn under_alpha_a_missed_by_sorts_before_every_hit() {
        let mut db = Db::new();
        list(&mut db, b"l", &["c", "a", "b"]);
        db.at(b"w_b")
            .set(b"w_b", b"zzz", SetOptions::PLAIN)
            .unwrap();
        db.at(b"w_c")
            .set(b"w_c", b"aaa", SetOptions::PLAIN)
            .unwrap();
        let opts = Sort {
            by: Some(b"w_*"),
            alpha: true,
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["a", "c", "b"]);
    }

    #[test]
    fn a_pattern_can_reach_into_a_hash() {
        let mut db = Db::new();
        list(&mut db, b"l", &["a", "b"]);
        db.at(b"h_a")
            .hset(b"h_a", [(&b"w"[..], &b"2"[..])].into_iter())
            .unwrap();
        db.at(b"h_b")
            .hset(b"h_b", [(&b"w"[..], &b"1"[..])].into_iter())
            .unwrap();
        let opts = Sort {
            by: Some(b"h_*->w"),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["b", "a"]);
        // And a pattern that ends in an arrow is a key name, not a hash lookup
        // with no field, so it reads a string key called `h_a->`.
        db.at(b"h_a->")
            .set(b"h_a->", b"9", SetOptions::PLAIN)
            .unwrap();
        let trailing = Sort {
            by: Some(b"h_*->"),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &trailing).unwrap()), ["b", "a"]);
    }

    #[test]
    fn get_answers_other_keys_and_a_hash_of_them() {
        let mut db = Db::new();
        list(&mut db, b"l", &["2", "1"]);
        db.at(b"d_1")
            .set(b"d_1", b"one", SetOptions::PLAIN)
            .unwrap();
        db.at(b"d_2")
            .set(b"d_2", b"two", SetOptions::PLAIN)
            .unwrap();
        let get: [&[u8]; 2] = [b"#", b"d_*"];
        let opts = Sort {
            get: &get,
            ..Sort::default()
        };
        assert_eq!(
            flat(db.sort(b"l", &opts).unwrap()),
            ["1", "one", "2", "two"]
        );
        // A miss is a nil and not a skipped row, because the reply is positional.
        db.at(b"d_2").del(b"d_2");
        assert_eq!(
            flat(db.sort(b"l", &opts).unwrap()),
            ["1", "one", "2", "nil"]
        );
    }

    #[test]
    fn a_lookup_at_the_wrong_type_is_a_miss_and_not_an_error() {
        let mut db = Db::new();
        list(&mut db, b"l", &["a"]);
        list(&mut db, b"d_a", &["x"]);
        let get: [&[u8]; 1] = [b"d_*"];
        let opts = Sort {
            get: &get,
            alpha: true,
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["nil"]);
    }

    #[test]
    fn by_without_a_star_leaves_the_order_alone() {
        let mut db = Db::new();
        list(&mut db, b"l", &["3", "1", "2"]);
        let opts = Sort {
            by: Some(b"nosort"),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &opts).unwrap()), ["3", "1", "2"]);
        // And DESC still reverses it, because there is an order to reverse.
        let back = Sort {
            by: Some(b"nosort"),
            desc: true,
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"l", &back).unwrap()), ["2", "1", "3"]);
    }

    #[test]
    fn a_set_stored_without_a_sort_is_sorted_anyway() {
        let mut db = Db::new();
        for m in ["c", "a", "b"] {
            db.at(b"s").sadd(b"s", [m.as_bytes()].into_iter()).unwrap();
        }
        let opts = Sort {
            by: Some(b"nosort"),
            ..Sort::default()
        };
        assert_eq!(db.sort_store(b"s", b"out", &opts).unwrap(), 3);
        let got: Vec<String> = db
            .at(b"out")
            .lrange(b"out", 0, -1)
            .unwrap()
            .map(text)
            .collect();
        assert_eq!(got, ["a", "b", "c"]);
    }

    #[test]
    fn a_sorted_set_comes_out_in_score_order_when_nothing_says_otherwise() {
        let mut db = Db::new();
        db.at(b"z")
            .zadd(
                b"z",
                [(3.0, &b"c"[..]), (1.0, &b"a"[..]), (2.0, &b"b"[..])].into_iter(),
                ZAdd::default(),
            )
            .unwrap();
        let opts = Sort {
            by: Some(b"nosort"),
            ..Sort::default()
        };
        assert_eq!(flat(db.sort(b"z", &opts).unwrap()), ["a", "b", "c"]);
    }

    #[test]
    fn storing_an_empty_result_removes_the_destination() {
        let mut db = Db::new();
        list(&mut db, b"out", &["stale"]);
        assert_eq!(
            db.sort_store(b"missing", b"out", &Sort::default()).unwrap(),
            0
        );
        assert!(!db.at(b"out").exists(b"out"));
    }

    #[test]
    fn a_stored_get_that_missed_is_an_empty_string() {
        let mut db = Db::new();
        list(&mut db, b"l", &["1"]);
        let get: [&[u8]; 1] = [b"d_*"];
        let opts = Sort {
            get: &get,
            ..Sort::default()
        };
        assert_eq!(db.sort_store(b"l", b"out", &opts).unwrap(), 1);
        assert_eq!(db.at(b"out").llen(b"out").unwrap(), 1);
    }

    #[test]
    fn a_missing_key_is_empty_and_a_wrong_type_is_an_error() {
        let mut db = Db::new();
        assert!(db.sort(b"nosuchkey", &Sort::default()).unwrap().is_empty());
        db.at(b"str").set(b"str", b"x", SetOptions::PLAIN).unwrap();
        assert_eq!(
            db.sort(b"str", &Sort::default()).unwrap_err().code(),
            Code::WrongType
        );
    }

    #[test]
    fn sorting_into_the_key_being_sorted_works() {
        let mut db = Db::new();
        list(&mut db, b"l", &["3", "1", "2"]);
        assert_eq!(db.sort_store(b"l", b"l", &Sort::default()).unwrap(), 3);
        let got: Vec<String> = db.at(b"l").lrange(b"l", 0, -1).unwrap().map(text).collect();
        assert_eq!(got, ["1", "2", "3"]);
    }
}
