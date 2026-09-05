//! The suggestion dictionaries behind `FT.SUG*`, which is a trie under a key.
//!
//! This is the one thing the search module puts in the keyspace. An index is
//! not a key and a `FT.DICT*` word list is not a key, but a suggestion
//! dictionary is: `TYPE` answers `trietype0` for one, `EXISTS` sees it, `KEYS`
//! lists it and `EXPIRE` works on it. So the storage is here and the body that
//! holds it is up in the dispatch beside the other module types.
//!
//! # What a lookup answers with
//!
//! Not the score the client stored. The number that comes back is
//!
//! ```text
//! stored / exp(2 * distance) / sqrt(1 + abs(runes(term) - bytes(prefix)))
//! ```
//!
//! worked out in single precision, which is where all the odd looking digits in
//! the replies come from. `distance` is the smallest number of edits between
//! the prefix the client asked with and any prefix of the term, and it is zero
//! unless `FUZZY` was asked for, where it is at most one.
//!
//! The two lengths in that formula are not measured the same way, and that is
//! not a slip on the way in here. The term is counted in runes and the prefix
//! the client sent is counted in bytes, which a real server also does, so
//! asking for `ec` against `École` divides by the square root of three rather
//! than of four. It only shows on a term or a prefix that is not plain ASCII,
//! and it is the difference between a `strlen` and a rune count in the module's
//! C.
//!
//! A term the module calls an exact match is scored as though the client had
//! stored two to the thirty first, so it sorts in front of everything. It is
//! compared against the lowered prefix rather than against what the client
//! typed, which is why storing `ABC` and asking for `ABC` is not exact but
//! storing `abc` and asking for `ABC` is.
//!
//! Only the first half of it is compared, though, and that is the module's own
//! reading rather than a shortcut taken here. It hands a rune count to a
//! comparison that counts bytes, a rune is two bytes wide, so a five rune query
//! is compared over five bytes and everything past the third rune is taken on
//! trust. `abcdX` is an exact match for `abcde` and `aXcde` is not. The
//! comparison itself has the whole of it written down beside it.
//!
//! # Case
//!
//! Matching does not care about case and storing does. `abc` and `ABC` are two
//! terms with two scores that both come back from a lookup for `aB`. That is
//! done here by keying the trie on the lowered runes and letting a node hold
//! more than one spelling, where a real server keeps two paths and folds as it
//! compares. Nothing on the wire tells the two arrangements apart except the
//! order two terms with the same score come out in, which is written down.
//!
//! # A term with a score of minus infinity is never answered
//!
//! It is stored, it is counted by `FT.SUGLEN` and it is deleted by
//! `FT.SUGDEL`, but no lookup will ever return it. A candidate has to beat the
//! worst score kept so far to get in, that floor starts at minus infinity, and
//! minus infinity does not beat minus infinity. It reads like a bug and it is
//! copied because a client that stores such a score is relying on the term
//! staying out of the way.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// What a term that is exactly the prefix is scored as, whatever was stored.
const EXACT: f32 = 2_147_483_648.0;

/// How many edits `FUZZY` allows.
const MAX_DIST: u32 = 1;

/// One spelling under a node: what the client stored and what it hung on it.
#[derive(Debug)]
struct Term {
    /// The bytes the client added, which is what a lookup replies with.
    text: Box<[u8]>,
    /// How many runes that is, which is one half of the length divisor.
    runes: u32,
    /// The score as stored, in the single precision a real server keeps it in,
    /// which is why adding `1e39` reads back as infinity.
    score: f32,
    /// The payload, or nothing. An empty payload is nothing, since that is what
    /// a real server answers a null for.
    payload: Option<Box<[u8]>>,
}

/// One node of the radix trie, labelled with the lowered runes it adds.
#[derive(Debug, Default)]
struct Node {
    /// What this node adds to its parent's path. Empty only at the root.
    label: Box<[char]>,
    /// Children, kept in rune order so a descent is a binary search.
    kids: Vec<Node>,
    /// The spellings that end here, in byte order. Usually one.
    terms: Vec<Term>,
}

/// A suggestion dictionary.
///
/// The terms are held once each, so a lookup borrows them rather than copying
/// them out, and a walk that finds nothing worth keeping does not allocate.
#[derive(Debug, Default)]
pub struct Suggestions {
    /// The empty node everything hangs off.
    root: Node,
    /// How many spellings are in there, which is what `FT.SUGLEN` answers.
    count: usize,
}

/// One answer from a lookup.
#[derive(Debug, Clone, Copy)]
pub struct Hit<'a> {
    /// The term as the client stored it.
    pub term: &'a [u8],
    /// What the lookup worked out, not what the client stored.
    pub score: f64,
    /// The payload, if there is one.
    pub payload: Option<&'a [u8]>,
}

impl Suggestions {
    /// An empty dictionary.
    #[must_use]
    pub fn new() -> Suggestions {
        Suggestions::default()
    }

    /// How many terms are in here.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether there are none, which is when the key goes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Puts a term in, or changes the one that is there.
    ///
    /// `incr` adds to the score that is already stored rather than replacing
    /// it. A payload replaces what was there only when one is given, so a plain
    /// add over a term that has one keeps it. An empty term is not stored,
    /// which is why a real server answers such an add with an unchanged length.
    pub fn add(&mut self, text: &[u8], score: f64, incr: bool, payload: Option<&[u8]>) {
        if text.is_empty() {
            return;
        }
        let key = lower(text);
        let runes = key.len() as u32;
        let terms = self.root.slot(&key);
        let score = score as f32;
        match terms.binary_search_by(|t| (*t.text).cmp(text)) {
            Ok(at) => {
                let term = &mut terms[at];
                term.score = if incr { term.score + score } else { score };
                if let Some(bytes) = payload {
                    term.payload = keep(bytes);
                }
            }
            Err(at) => {
                terms.insert(
                    at,
                    Term {
                        text: text.into(),
                        runes,
                        score,
                        payload: payload.and_then(keep),
                    },
                );
                self.count += 1;
            }
        }
    }

    /// Takes a term out, and answers whether it was there.
    ///
    /// The spelling has to match byte for byte, so deleting `ABC` leaves `abc`
    /// alone even though a lookup for either finds both.
    pub fn remove(&mut self, text: &[u8]) -> bool {
        let key = lower(text);
        let gone = self.root.cut(&key, text);
        if gone {
            self.count -= 1;
        }
        gone
    }

    /// The best terms under `prefix`, best first and at most `max` of them.
    ///
    /// `fuzzy` allows one edit between the prefix and the front of the term.
    /// The order among terms that score the same is by their bytes, which a
    /// real server does not promise and does not give.
    #[must_use]
    pub fn best(&self, prefix: &[u8], fuzzy: bool, max: usize) -> Vec<Hit<'_>> {
        let query = lower(prefix);
        let ask = Ask {
            folded: &query,
            bytes: prefix.len(),
        };
        let mut keep = Keep::new(max);
        if fuzzy && query.is_empty() {
            // An empty prefix asked for fuzzily puts every term one edit away
            // rather than none, which is measured off a real server and does
            // not follow from anything else here.
            gather(&self.root, MAX_DIST, &ask, &mut keep);
        } else if fuzzy {
            // The first row is the cost of turning the empty path into each
            // front of the query, which is one insertion each.
            let row: Vec<u32> = (0..=query.len() as u32).collect();
            walk(&self.root, &query, &row, u32::MAX, &ask, &mut keep);
        } else if let Some(node) = under(&self.root, &query) {
            gather(node, 0, &ask, &mut keep);
        }
        keep.finish()
    }

    /// What this dictionary has allocated.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.root.memory_bytes()
    }
}

impl Node {
    /// The terms that end at `key`, making the path if it is not there.
    ///
    /// Recursive, and the depth is the number of nodes on the path rather than
    /// the length of the term, since a run of runes nothing branches off is one
    /// node. Reaching a deep trie needs terms that nest inside each other, so
    /// the depth is around the square root of everything ever added.
    fn slot(&mut self, key: &[char]) -> &mut Vec<Term> {
        if key.is_empty() {
            return &mut self.terms;
        }
        match self.kids.binary_search_by_key(&key[0], |k| k.label[0]) {
            Ok(at) => {
                let same = shared(&self.kids[at].label, key);
                if same < self.kids[at].label.len() {
                    self.kids[at].split(same);
                }
                self.kids[at].slot(&key[same..])
            }
            Err(at) => {
                self.kids.insert(
                    at,
                    Node {
                        label: key.into(),
                        ..Node::default()
                    },
                );
                &mut self.kids[at].terms
            }
        }
    }

    /// Cuts this node in two at `at`, so what is below it hangs off a new child
    /// and this node keeps the runes the two paths share.
    fn split(&mut self, at: usize) {
        let below = Node {
            label: self.label[at..].into(),
            kids: std::mem::take(&mut self.kids),
            terms: std::mem::take(&mut self.terms),
        };
        self.label = self.label[..at].into();
        self.kids.push(below);
    }

    /// Takes the spelling `text` out from under here, and tidies up after it.
    fn cut(&mut self, key: &[char], text: &[u8]) -> bool {
        if key.is_empty() {
            let Ok(at) = self.terms.binary_search_by(|t| (*t.text).cmp(text)) else {
                return false;
            };
            self.terms.remove(at);
            return true;
        }
        let Ok(at) = self.kids.binary_search_by_key(&key[0], |k| k.label[0]) else {
            return false;
        };
        if !key.starts_with(&self.kids[at].label) {
            return false;
        }
        let past = self.kids[at].label.len();
        if !self.kids[at].cut(&key[past..], text) {
            return false;
        }
        // A node with nothing in it and nowhere to go is not on anybody's path,
        // and a node with one way down and no terms of its own is a run that
        // was split for a term that has now gone.
        let kid = &mut self.kids[at];
        if kid.terms.is_empty() {
            if kid.kids.is_empty() {
                self.kids.remove(at);
            } else if kid.kids.len() == 1 {
                kid.join();
            }
        }
        true
    }

    /// Folds an only child into this node, which is what keeps the trie radix.
    fn join(&mut self) {
        let below = self.kids.pop().expect("one child and no more");
        let mut label = self.label.to_vec();
        label.extend_from_slice(&below.label);
        self.label = label.into();
        self.kids = below.kids;
        self.terms = below.terms;
    }

    /// What this node and everything under it has allocated.
    fn memory_bytes(&self) -> usize {
        let mut total = size_of::<Node>() + self.label.len() * size_of::<char>();
        total += self.kids.capacity() * size_of::<Node>();
        total += self.terms.capacity() * size_of::<Term>();
        for term in &self.terms {
            total += term.text.len();
            total += term.payload.as_ref().map_or(0, |p| p.len());
        }
        for kid in &self.kids {
            total += kid.memory_bytes() - size_of::<Node>();
        }
        total
    }
}

/// The read time constants of one lookup.
struct Ask<'a> {
    /// The prefix with its runes lowered, which is what an exact match is
    /// measured against.
    folded: &'a [char],
    /// How many bytes the client sent, which is the length the divisor uses.
    bytes: usize,
}

impl Ask<'_> {
    /// One term as a lookup sees it.
    fn hit<'t>(&self, term: &'t Term, dist: u32) -> Hit<'t> {
        let stored = if self.exact(term) { EXACT } else { term.score };
        Hit {
            term: &term.text,
            score: rank(stored, term.runes, dist, self.bytes),
            payload: term.payload.as_deref(),
        }
    }

    /// Is this the term the module calls an exact match?
    ///
    /// Not what the name suggests, and this is copied rather than tidied
    /// because the difference is worth two hundred and ninety million on the
    /// score. The module means to compare the whole term with the query, and it
    /// does compare them, but it hands the number of runes to a comparison that
    /// counts bytes. A rune is two bytes wide in there, so a query of five
    /// runes is compared over five bytes, which is the first two runes and the
    /// lower half of the third. That is why `abcdX` scores as an exact match
    /// for `abcde` and `aXcde` does not, and why `aǩx` counts as `aéx`: the
    /// two spellings share a lower byte and the comparison never reads the
    /// upper one. The rune count itself is compared properly, so a term that is
    /// longer or shorter is never exact however far it agrees.
    fn exact(&self, term: &Term) -> bool {
        if term.runes as usize != self.folded.len() {
            return false;
        }
        let Ok(text) = core::str::from_utf8(&term.text) else {
            return false;
        };
        let mut runes = text.chars();
        let mut query = self.folded.iter().copied();
        for _ in 0..self.folded.len() / 2 {
            if runes.next() != query.next() {
                return false;
            }
        }
        if self.folded.len().is_multiple_of(2) {
            return true;
        }
        // The odd rune out, of which the count only reaches the lower byte.
        match (runes.next(), query.next()) {
            (Some(a), Some(b)) => a as u32 & 0xff == b as u32 & 0xff,
            _ => false,
        }
    }
}

/// What a lookup answers with, in the single precision it is worked out in.
///
/// Two divisions and a rounding after each, because the module holds the
/// running value in a float and both divisors are doubles, so every step is
/// worked out wide and put back narrow.
fn rank(stored: f32, runes: u32, dist: u32, bytes: usize) -> f64 {
    let gap = (runes as usize).abs_diff(bytes);
    let step = (f64::from(stored) / (2.0 * f64::from(dist)).exp()) as f32;
    f64::from((f64::from(step) / (1.0 + gap as f64).sqrt()) as f32)
}

/// The node covering `key`, or nothing if no term starts with it.
fn under<'a>(node: &'a Node, key: &[char]) -> Option<&'a Node> {
    if key.is_empty() {
        return Some(node);
    }
    let at = node
        .kids
        .binary_search_by_key(&key[0], |k| k.label[0])
        .ok()?;
    let kid = &node.kids[at];
    if kid.label.len() >= key.len() {
        // The prefix runs out inside this node's label, so everything below it
        // matches or nothing does.
        return kid.label.starts_with(key).then_some(kid);
    }
    key.starts_with(&kid.label)
        .then(|| under(kid, &key[kid.label.len()..]))
        .flatten()
}

/// Offers every term at or under `node` at the same distance.
///
/// An explicit stack rather than recursion, because this one walks a whole
/// subtree and a client chooses how deep that is. The children go on the stack
/// backwards so they come off it in rune order, which is what decides who is
/// kept when a run of terms all score the same.
fn gather<'a>(node: &'a Node, dist: u32, ask: &Ask<'_>, keep: &mut Keep<'a>) {
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        for term in &node.terms {
            keep.offer(ask.hit(term, dist));
        }
        stack.extend(node.kids.iter().rev());
    }
}

/// The fuzzy walk: a depth first descent carrying a Levenshtein row.
///
/// `row` is the cost of turning the path so far into each front of the query,
/// and `best` is the smallest cost any front of the path has reached, which is
/// the distance a term down here would be scored at. The row only ever grows,
/// so once its smallest entry is past the limit no descendant can do better
/// than `best` and the rest of the subtree is either taken whole or dropped
/// whole. That bounds this recursion to about the length of the query.
fn walk<'a>(
    node: &'a Node,
    query: &[char],
    row: &[u32],
    best: u32,
    ask: &Ask<'_>,
    keep: &mut Keep<'a>,
) {
    let mut here = row.to_vec();
    let mut next = Vec::with_capacity(row.len());
    let mut best = best.min(*here.last().expect("a row has the empty front in it"));
    for &rune in &node.label {
        step(&here, query, rune, &mut next);
        std::mem::swap(&mut here, &mut next);
        best = best.min(*here.last().expect("a row has the empty front in it"));
    }
    if best <= MAX_DIST {
        for term in &node.terms {
            keep.offer(ask.hit(term, best));
        }
    }
    let floor = *here.iter().min().expect("a row is never empty");
    if floor > MAX_DIST {
        // Nothing below can come back towards the query, so the answer for the
        // whole subtree is settled.
        if best <= MAX_DIST {
            for kid in &node.kids {
                gather(kid, best, ask, keep);
            }
        }
        return;
    }
    for kid in &node.kids {
        walk(kid, query, &here, best, ask, keep);
    }
}

/// One row of the edit distance table, for the path grown by one rune.
fn step(prev: &[u32], query: &[char], rune: char, out: &mut Vec<u32>) {
    out.clear();
    out.push(prev[0] + 1);
    for j in 1..prev.len() {
        let swap = prev[j - 1] + u32::from(query[j - 1] != rune);
        out.push(swap.min(out[j - 1] + 1).min(prev[j] + 1));
    }
}

/// The best `max` hits, kept as the walk finds them.
///
/// A candidate has to beat the worst one kept, strictly, which is where the
/// two visible oddities come from: a term scoring minus infinity never gets in
/// because the floor starts there, and once the heap is full a term that ties
/// the worst is dropped rather than replacing it.
struct Keep<'a> {
    /// How many to keep.
    max: usize,
    /// The ones kept, worst on top so it is the one that goes.
    heap: BinaryHeap<Worst<'a>>,
    /// The score to beat.
    floor: f64,
}

impl<'a> Keep<'a> {
    /// A collector for `max` hits.
    fn new(max: usize) -> Keep<'a> {
        Keep {
            max,
            heap: BinaryHeap::new(),
            floor: f64::NEG_INFINITY,
        }
    }

    /// Takes a hit if it is good enough to keep.
    fn offer(&mut self, hit: Hit<'a>) {
        if hit.score <= self.floor || self.max == 0 {
            return;
        }
        self.heap.push(Worst(hit));
        if self.heap.len() > self.max {
            self.heap.pop();
            self.floor = self.heap.peek().map_or(f64::NEG_INFINITY, |w| w.0.score);
        }
    }

    /// The hits, best first.
    fn finish(self) -> Vec<Hit<'a>> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|w| w.0)
            .collect()
    }
}

/// A hit ordered so that the worst one is the greatest, which is what a binary
/// heap hands back first.
struct Worst<'a>(Hit<'a>);

impl Ord for Worst<'_> {
    fn cmp(&self, other: &Worst<'_>) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.term.cmp(other.0.term))
    }
}

impl PartialOrd for Worst<'_> {
    fn partial_cmp(&self, other: &Worst<'_>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Worst<'_> {
    fn eq(&self, other: &Worst<'_>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Worst<'_> {}

/// The runes of `text`, lowered, which is what the trie is keyed on.
///
/// Bytes that are not UTF-8 come through as the replacement rune, so two terms
/// that are both nonsense can land on the same path. Nothing is lost by that,
/// since a node holds every spelling that reached it and each keeps its own
/// bytes.
fn lower(text: &[u8]) -> Vec<char> {
    String::from_utf8_lossy(text)
        .chars()
        .map(|rune| rune.to_lowercase().next().unwrap_or(rune))
        .collect()
}

/// How many runes two paths share at the front.
fn shared(label: &[char], key: &[char]) -> usize {
    label.iter().zip(key).take_while(|(a, b)| a == b).count()
}

/// A payload as it is stored, where an empty one is no payload at all.
fn keep(bytes: &[u8]) -> Option<Box<[u8]>> {
    (!bytes.is_empty()).then(|| bytes.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The terms a lookup answers with, in the order they come back.
    fn terms(s: &Suggestions, prefix: &str, fuzzy: bool, max: usize) -> Vec<String> {
        s.best(prefix.as_bytes(), fuzzy, max)
            .into_iter()
            .map(|h| String::from_utf8_lossy(h.term).into_owned())
            .collect()
    }

    fn add(s: &mut Suggestions, text: &str, score: f64) {
        s.add(text.as_bytes(), score, false, None);
    }

    #[test]
    fn a_term_is_counted_once_however_often_it_is_added() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        add(&mut s, "one", 5.0);
        assert_eq!(s.len(), 1);
        assert_eq!(s.best(b"on", false, 5)[0].score, 3.535533905029297);
    }

    #[test]
    fn an_empty_term_is_not_stored() {
        let mut s = Suggestions::new();
        add(&mut s, "", 1.0);
        assert!(s.is_empty());
    }

    /// Every number in here was read off a real server rather than worked out,
    /// which is the only way the single precision in the middle of the formula
    /// would have been believed.
    #[test]
    fn the_score_falls_off_with_how_much_longer_the_term_is() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        add(&mut s, "only", 2.0);
        add(&mut s, "ontario", 3.0);
        let scores: Vec<f64> = s.best(b"on", false, 5).iter().map(|h| h.score).collect();
        assert_eq!(
            scores,
            [1.2247449159622192, 1.154700517654419, 0.7071067690849304]
        );
    }

    /// The one measurement that pins the two lengths down as different things.
    #[test]
    fn the_term_is_counted_in_runes_and_the_prefix_in_bytes() {
        let mut s = Suggestions::new();
        add(&mut s, "École", 4.0);
        // Five runes against three bytes, so the gap is two and not three,
        // and four over the square root of three is what comes back.
        assert_eq!(
            s.best("éc".as_bytes(), false, 5)[0].score,
            2.309401035308838
        );
        // Five runes against six bytes is a gap of one either way round.
        assert_eq!(
            s.best("école".as_bytes(), false, 5)[0].score,
            2.8284270763397217
        );
    }

    #[test]
    fn a_term_that_is_the_lowered_prefix_sorts_in_front_of_everything() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        add(&mut s, "oneself", 1000.0);
        assert_eq!(terms(&s, "ONE", false, 5), ["one", "oneself"]);
        assert_eq!(s.best(b"ONE", false, 5)[0].score, f64::from(EXACT));
    }

    /// The exact one is still divided by the length gap, so the sentinel is
    /// only ever seen whole on a term whose runes and bytes come to the same
    /// number. This is the reading a real server gives for a five rune term
    /// asked for with six bytes.
    #[test]
    fn the_exact_score_is_divided_like_any_other() {
        let mut s = Suggestions::new();
        add(&mut s, "école", 4.0);
        assert_eq!(s.best("École".as_bytes(), false, 5)[0].score, 1518500224.0);
    }

    /// Storing the capital and asking for the capital is not exact, because the
    /// comparison is against the prefix after it has been lowered.
    #[test]
    fn a_term_stored_in_capitals_is_never_the_exact_one() {
        let mut s = Suggestions::new();
        add(&mut s, "ABC", 1.0);
        assert_eq!(s.best(b"ABC", false, 5)[0].score, 1.0);
        assert_eq!(s.best(b"abc", false, 5)[0].score, 1.0);
    }

    /// The comparison behind the exact score reads a rune count as a byte
    /// count, so it only ever looks at the front half of the word. Every number
    /// here is a real server's, and the three at the bottom are the ordinary
    /// score for one edit away.
    #[test]
    fn only_the_front_half_of_the_query_is_compared_for_an_exact_match() {
        let mut s = Suggestions::new();
        for text in ["abcde", "abcXe", "abcdX", "abXde", "aXcde", "Xbcde"] {
            add(&mut s, text, 1.0);
        }
        let hits = s.best(b"abcde", true, 10);
        let seen: Vec<(String, f64)> = hits
            .iter()
            .map(|h| (String::from_utf8_lossy(h.term).into_owned(), h.score))
            .collect();
        let mut want = vec![
            ("abcde".to_owned(), 2147483648.0),
            ("abcXe".to_owned(), 290630304.0),
            ("abcdX".to_owned(), 290630304.0),
            ("abXde".to_owned(), 0.1353352814912796),
            ("aXcde".to_owned(), 0.1353352814912796),
            ("Xbcde".to_owned(), 0.1353352814912796),
        ];
        let mut seen = seen;
        seen.sort_by(|a, b| a.0.cmp(&b.0));
        want.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(seen, want);
    }

    /// A query with an odd number of runes has its middle one compared on the
    /// lower of its two bytes and nothing else, so `ǩ` counts as `é`. Both
    /// numbers are a real server's.
    #[test]
    fn the_odd_rune_out_is_compared_on_its_lower_byte_alone() {
        let mut s = Suggestions::new();
        for text in ["aéx", "aǩx", "aèx"] {
            add(&mut s, text, 1.0);
        }
        let scores: Vec<f64> = s
            .best("aéx".as_bytes(), true, 10)
            .iter()
            .map(|h| h.score)
            .collect();
        // Three runes against four bytes, so even the exact one is divided.
        assert_eq!(scores, [1518500224.0, 205506656.0, 0.09569649398326874]);
    }

    #[test]
    fn two_spellings_are_two_terms_and_both_answer_one_lookup() {
        let mut s = Suggestions::new();
        add(&mut s, "ABC", 1.0);
        add(&mut s, "abc", 2.0);
        assert_eq!(s.len(), 2);
        assert_eq!(terms(&s, "aB", false, 5), ["abc", "ABC"]);
        assert!(s.remove(b"ABC"));
        assert_eq!(terms(&s, "aB", false, 5), ["abc"]);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn one_edit_is_allowed_when_fuzzy_is_asked_for_and_not_otherwise() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        add(&mut s, "only", 2.0);
        add(&mut s, "ontario", 3.0);
        assert_eq!(terms(&s, "one", false, 5), ["one"]);
        let scores: Vec<f64> = s.best(b"one", true, 5).iter().map(|h| h.score).collect();
        assert_eq!(
            scores,
            [2147483648.0, 0.19139298796653748, 0.1815713346004486]
        );
    }

    /// The edits are counted in runes and not in bytes, so the two bytes of an
    /// i with a diaeresis are one edit and not two.
    #[test]
    fn an_edit_is_a_rune_and_not_a_byte() {
        let mut s = Suggestions::new();
        add(&mut s, "naïve", 4.0);
        assert_eq!(s.best(b"naive", true, 5)[0].score, 0.5413411259651184);
    }

    #[test]
    fn two_edits_are_too_many() {
        let mut s = Suggestions::new();
        add(&mut s, "onward", 2.0);
        assert!(terms(&s, "xyz", true, 5).is_empty());
        assert_eq!(terms(&s, "onwarx", true, 5), ["onward"]);
    }

    /// A prefix of nothing is a prefix of everything, and the terms are then
    /// ordered by what a lookup works out rather than by what was stored.
    #[test]
    fn an_empty_prefix_matches_every_term() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        add(&mut s, "only", 2.0);
        add(&mut s, "ontario", 3.0);
        assert_eq!(terms(&s, "", false, 5), ["ontario", "only", "one"]);
        let scores: Vec<f64> = s.best(b"", false, 5).iter().map(|h| h.score).collect();
        assert_eq!(scores, [1.0606601238250732, 0.8944271802902222, 0.5]);
        // And asked for fuzzily it is every term at one edit, which is the
        // same order a long way further down.
        let fuzzy: Vec<f64> = s.best(b"", true, 5).iter().map(|h| h.score).collect();
        assert_eq!(fuzzy[2], 0.0676676407456398);
    }

    #[test]
    fn a_term_scoring_minus_infinity_is_stored_and_never_answered() {
        let mut s = Suggestions::new();
        add(&mut s, "aaa", f64::NEG_INFINITY);
        add(&mut s, "aab", -5.0);
        assert_eq!(s.len(), 2);
        assert_eq!(terms(&s, "aa", false, 5), ["aab"]);
    }

    /// The first few in trie order and not the last few, because a tie does not
    /// beat what is already kept.
    #[test]
    fn a_tie_does_not_push_out_what_is_already_kept() {
        let mut s = Suggestions::new();
        for name in ["aaa", "aab", "aac", "aad", "aae", "aaf", "aag"] {
            add(&mut s, name, 1.0);
        }
        assert_eq!(terms(&s, "aa", false, 3), ["aaa", "aab", "aac"]);
    }

    /// Three tenths added a tenth at a time, which is the reading that shows
    /// the score is held in single precision rather than double.
    #[test]
    fn the_score_can_be_added_to_rather_than_replaced() {
        let mut s = Suggestions::new();
        for _ in 0..3 {
            s.add(b"xxx", 0.1, true, None);
        }
        assert_eq!(s.len(), 1);
        assert_eq!(s.best(b"xx", false, 5)[0].score, 0.2121320366859436);
    }

    /// A score too large for a float is stored as infinity, which is how a
    /// real server answers it back.
    #[test]
    fn a_score_that_does_not_fit_a_float_becomes_infinity() {
        let mut s = Suggestions::new();
        add(&mut s, "aaa", core::f64::consts::PI);
        add(&mut s, "bbb", 1e39);
        assert_eq!(s.best(b"aa", false, 5)[0].score, 2.2214415073394775);
        assert_eq!(s.best(b"bb", false, 5)[0].score, f64::INFINITY);
    }

    #[test]
    fn a_negative_score_comes_back_negative() {
        let mut s = Suggestions::new();
        add(&mut s, "nnn", -5.0);
        assert_eq!(s.best(b"n", false, 5)[0].score, -2.886751413345337);
    }

    #[test]
    fn a_payload_stays_until_another_one_replaces_it() {
        let mut s = Suggestions::new();
        s.add(b"x", 1.0, false, Some(b"first"));
        assert_eq!(s.best(b"xx", true, 5)[0].payload, Some(&b"first"[..]));
        s.add(b"x", 2.0, false, None);
        assert_eq!(s.best(b"xx", true, 5)[0].payload, Some(&b"first"[..]));
        s.add(b"x", 2.0, true, Some(b"second"));
        assert_eq!(s.best(b"xx", true, 5)[0].payload, Some(&b"second"[..]));
    }

    #[test]
    fn an_empty_payload_is_no_payload() {
        let mut s = Suggestions::new();
        s.add(b"x", 1.0, false, Some(b""));
        assert_eq!(s.best(b"xx", true, 5)[0].payload, None);
    }

    #[test]
    fn a_term_that_was_never_added_is_not_deleted() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        assert!(!s.remove(b"two"));
        assert!(!s.remove(b"on"));
        assert!(s.remove(b"one"));
        assert!(s.is_empty());
    }

    /// The shape of the trie after a run has been split and put back together
    /// again, which is the part a walk quietly depends on.
    #[test]
    fn a_run_that_was_split_is_joined_up_again_when_the_split_goes() {
        let mut s = Suggestions::new();
        add(&mut s, "abcdef", 1.0);
        add(&mut s, "abcxyz", 2.0);
        add(&mut s, "abc", 3.0);
        assert_eq!(terms(&s, "abc", false, 10), ["abc", "abcxyz", "abcdef"]);
        assert!(s.remove(b"abc"));
        assert!(s.remove(b"abcxyz"));
        assert_eq!(terms(&s, "abcd", false, 10), ["abcdef"]);
        assert_eq!(terms(&s, "abcx", false, 10), Vec::<String>::new());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_lookup_for_something_that_is_not_there_answers_nothing() {
        let mut s = Suggestions::new();
        add(&mut s, "one", 1.0);
        assert_eq!(terms(&s, "two", false, 5), Vec::<String>::new());
        assert_eq!(terms(&s, "oneone", false, 5), Vec::<String>::new());
    }

    #[test]
    fn a_dictionary_takes_room_and_an_empty_one_takes_almost_none() {
        let mut s = Suggestions::new();
        let bare = s.memory_bytes();
        add(&mut s, "something reasonably long", 1.0);
        assert!(s.memory_bytes() > bare);
    }
}
