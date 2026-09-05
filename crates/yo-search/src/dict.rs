//! The word lists a client keeps beside the indexes, which is `FT.DICT*`.
//!
//! A dictionary is a name and a set of terms and nothing else. It is not a key:
//! `TYPE` says `none` for one, `KEYS` never lists it and `EXISTS` answers zero,
//! which is measured rather than assumed. It is not attached to an index
//! either, so the same name can be handed to `FT.SPELLCHECK` for one index
//! today and another tomorrow.
//!
//! The terms come back sorted by their bytes rather than by when they were
//! added, so `ALPHA` sorts in front of `Beta` and `Beta` in front of `alpha`.
//! Nothing folds case on the way in, which means a dictionary can hold `alpha`
//! and `ALPHA` at once and counts them as two.
//!
//! An empty term is dropped rather than stored, on the way in and on the way
//! out: `FT.DICTADD d ""` answers zero and `FT.DICTDEL d ""` answers zero, and
//! neither of them changes what is there.

use std::collections::{BTreeMap, BTreeSet};

/// One dictionary, its terms in byte order.
type Words = BTreeSet<Box<[u8]>>;

/// Every dictionary on the server.
///
/// One of these per server the same way [`Registry`](crate::Registry) is, and
/// for the same reason: a real server keeps them in the search module rather
/// than in a database, so a dictionary added while `SELECT 1` is in force is
/// dumped just as happily after `SELECT 0`. Emptying the keyspace does empty
/// them, which is also measured.
#[derive(Debug, Default)]
pub struct Dicts {
    /// Name to terms, the names in byte order so a walk over them is stable.
    by_name: BTreeMap<Box<[u8]>, Words>,
}

impl Dicts {
    /// A server with no dictionaries on it.
    #[must_use]
    pub fn new() -> Dicts {
        Dicts::default()
    }

    /// Puts terms in, and answers how many of them were not already there.
    ///
    /// A term the dictionary already holds is not counted, so adding the same
    /// word twice answers one and then zero. An empty term is not a term and is
    /// dropped without being counted.
    pub fn add(&mut self, name: &[u8], terms: &[&[u8]]) -> usize {
        let words = self.by_name.entry(name.into()).or_default();
        let mut added = 0;
        for term in terms {
            if !term.is_empty() && words.insert((*term).into()) {
                added += 1;
            }
        }
        // A name that was reached for and given nothing is not a dictionary, so
        // it does not linger in the table waiting to be dumped as empty.
        if words.is_empty() {
            self.by_name.remove(name);
        }
        added
    }

    /// Takes terms out, and answers how many of them were there.
    ///
    /// A dictionary that is not there answers zero rather than complaining,
    /// which is the same answer a dictionary that is there but does not hold
    /// the word gives.
    pub fn del(&mut self, name: &[u8], terms: &[&[u8]]) -> usize {
        let Some(words) = self.by_name.get_mut(name) else {
            return 0;
        };
        let mut gone = 0;
        for term in terms {
            if words.remove(*term) {
                gone += 1;
            }
        }
        // The last word out takes the dictionary with it, which nothing on the
        // wire can tell apart from an empty one that stayed.
        if words.is_empty() {
            self.by_name.remove(name);
        }
        gone
    }

    /// Every term in a dictionary, in byte order.
    ///
    /// A name nobody has added to walks nothing, since a dictionary that was
    /// never made and one that has been emptied answer the same way.
    pub fn dump(&self, name: &[u8]) -> impl Iterator<Item = &[u8]> {
        self.by_name
            .get(name)
            .into_iter()
            .flat_map(|words| words.iter().map(|w| &**w))
    }

    /// How many terms a dictionary holds.
    #[must_use]
    pub fn len(&self, name: &[u8]) -> usize {
        self.by_name.get(name).map_or(0, BTreeSet::len)
    }

    /// Whether there are no dictionaries at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Every dictionary name, in byte order.
    pub fn names(&self) -> impl Iterator<Item = &[u8]> {
        self.by_name.keys().map(|n| &**n)
    }

    /// Throws them all away, which is what emptying the keyspace does.
    pub fn clear(&mut self) {
        self.by_name.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(d: &Dicts, name: &[u8]) -> Vec<String> {
        d.dump(name)
            .map(|w| String::from_utf8_lossy(w).into_owned())
            .collect()
    }

    #[test]
    fn a_word_is_counted_the_first_time_and_not_the_second() {
        let mut d = Dicts::new();
        assert_eq!(d.add(b"d", &[b"zeta", b"alpha", b"Beta", b"alpha"]), 3);
        assert_eq!(d.add(b"d", &[b"alpha"]), 0);
        assert_eq!(d.len(b"d"), 3);
    }

    /// Byte order and not folded order, so the capital sorts in front.
    #[test]
    fn the_terms_come_back_sorted_by_their_bytes() {
        let mut d = Dicts::new();
        d.add(b"d", &[b"zeta", b"alpha", b"Beta", b"ALPHA"]);
        assert_eq!(words(&d, b"d"), ["ALPHA", "Beta", "alpha", "zeta"]);
    }

    #[test]
    fn an_empty_term_is_not_a_term() {
        let mut d = Dicts::new();
        assert_eq!(d.add(b"d", &[b""]), 0);
        assert!(d.is_empty());
        assert_eq!(d.add(b"d", &[b"a", b""]), 1);
        assert_eq!(d.del(b"d", &[b"a", b""]), 1);
        assert!(d.is_empty());
    }

    #[test]
    fn a_dictionary_nobody_made_dumps_nothing_and_deletes_nothing() {
        let mut d = Dicts::new();
        assert_eq!(words(&d, b"nope"), Vec::<String>::new());
        assert_eq!(d.del(b"nope", &[b"a"]), 0);
        assert_eq!(d.len(b"nope"), 0);
    }

    #[test]
    fn the_last_word_out_takes_the_dictionary_with_it() {
        let mut d = Dicts::new();
        d.add(b"d", &[b"a", b"b"]);
        assert_eq!(d.names().count(), 1);
        assert_eq!(d.del(b"d", &[b"a", b"b", b"nope"]), 2);
        assert!(d.is_empty());
    }

    #[test]
    fn emptying_the_keyspace_empties_this_too() {
        let mut d = Dicts::new();
        d.add(b"d", &[b"a"]);
        d.clear();
        assert!(d.is_empty());
    }
}
