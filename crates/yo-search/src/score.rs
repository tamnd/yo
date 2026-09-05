//! Scoring: turning a document that matched into a number, the nine ways a
//! client can ask for it, and the constants each one uses.
//!
//! ```
//! use yo_search::docs::Doc;
//! use yo_search::score::{Facts, Found, Scorer, Term};
//!
//! // Four documents, two of them with the term, ten tokens between them.
//! let facts = Facts::new(4, 10);
//! let mut doc = Doc::new(b"book:1", 1.0);
//! doc.tokens = 1;
//! let found = Found::Term(Term::new(1, 1.0, 2));
//! let score = Scorer::Bm25.of(&facts, &doc, &found, None, 1);
//! assert!((score - 0.9186287989106708).abs() < 1e-12);
//! ```
//!
//! # Where the numbers come from
//!
//! Every constant here was read off a real server rather than taken from a
//! paper. The default scorer is BM25 with k1 of 1.2 and b of 0.75 and the usual
//! probabilistic idf, and it agrees with a real server to the last digit of a
//! double, which is worth knowing because the other eight do not all agree with
//! their own names.
//!
//! The scorer a client gets by asking for `BM25` is not the same as the one it
//! gets by asking for nothing. It uses a different idf, the whole number part of
//! a base two logarithm rather than a natural one, it drops the k1 plus one from
//! the top of the fraction, and it puts the average document length where the
//! document's own length belongs, so every document of the same term scores the
//! same however long it is. That last one looks like a slip in the original and
//! it is reproduced here on purpose, because a client that asks for that scorer
//! is asking for that behaviour. It also folds its constant in single precision,
//! which moves the eighth digit of the answer, so that is folded in single
//! precision here too.
//!
//! The rest are short. `TFIDF` divides the frequency by the largest frequency in
//! the document, `TFIDF.DOCNORM` divides it by the document's length instead,
//! `DISMAX` is the frequency and nothing else and is the only one that ignores
//! what the client said the document is worth, `DOCSCORE` is that worth on its
//! own, and `HAMMING` is about the payload rather than the terms.

use crate::docs::Doc;

/// How much a repeated term counts for before it stops mattering.
///
/// Single precision on purpose. A real server writes this constant as a float
/// and widens it, so the value that goes into the sum is a little over 1.2 and
/// the answers come out a few parts in a billion away from what the same sum in
/// double would give. Every other number in these scorers is a double, and this
/// one is what makes the answers here agree with a real server to the last digit
/// rather than to the eighth.
pub const K1: f32 = 1.2;

/// How much of the length correction to apply, from none at zero to all of it.
pub const B: f64 = 0.75;

/// The same knob in the `BM25` scorer, which uses half rather than three
/// quarters and applies it to the wrong length.
pub const OLD_B: f64 = 0.5;

/// How far apart two payloads can be before `HAMMING` calls them unalike.
///
/// At this distance and beyond the answer is zero rather than a small number,
/// so the scorer is a near neighbour test and not a distance.
pub const CLOSE: u32 = 8;

/// What `BM25STD.TANH` divides by before it takes the hyperbolic tangent.
pub const TANH: f64 = 4.0;

/// What the index as a whole knows, which the length correction needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Facts {
    /// How many documents are in the index.
    pub docs: u32,
    /// How many tokens they hold between them, weighted as they were indexed.
    pub tokens: u64,
    /// What `BM25STD.TANH` divides by, which a real server lets you set.
    pub tanh: f64,
}

impl Facts {
    /// The facts of an index with this many documents and this many tokens.
    #[must_use]
    pub fn new(docs: u32, tokens: u64) -> Facts {
        Facts {
            docs,
            tokens,
            tanh: TANH,
        }
    }

    /// The same facts with a different tangent factor.
    #[must_use]
    pub fn tanh(self, tanh: f64) -> Facts {
        Facts { tanh, ..self }
    }

    /// The average document length, which is what the correction compares to.
    ///
    /// Zero when the index is empty, and the scorers treat a zero average as no
    /// correction at all rather than dividing by it.
    #[must_use]
    pub fn average(&self) -> f64 {
        if self.docs == 0 {
            return 0.0;
        }
        self.tokens as f64 / f64::from(self.docs)
    }
}

/// One term of a query as one document answered it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term<'a> {
    /// How often the term is in the document, weighted by its field.
    pub freq: u32,
    /// What the query said this term is worth, one unless it said otherwise.
    pub weight: f64,
    /// How many documents in the index have the term at all.
    pub docs: u32,
    /// What the term is called, borrowed from wherever it is stored.
    ///
    /// Nothing to do with the score and everything to do with explaining it,
    /// which is the one place a real server says which word it was. The name is
    /// the dictionary's own spelling rather than the query's, so a stem is
    /// already written `+run` and a tag is written the way it was folded.
    pub name: Option<&'a [u8]>,
}

impl<'a> Term<'a> {
    /// A term with a frequency, a query weight and a document count.
    #[must_use]
    pub fn new(freq: u32, weight: f64, docs: u32) -> Term<'a> {
        Term {
            freq,
            weight,
            docs,
            name: None,
        }
    }

    /// The same term, knowing what it is called.
    #[must_use]
    pub fn about(self, name: &'a [u8]) -> Term<'a> {
        Term {
            name: Some(name),
            ..self
        }
    }

    /// The idf the default scorer uses, which is the usual probabilistic one.
    #[must_use]
    pub fn idf(&self, docs: u32) -> f64 {
        let n = f64::from(docs);
        let df = f64::from(self.docs.min(docs));
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// The idf the older scorers use, which is a whole number of bits.
    ///
    /// A term in one document out of sixteen is worth four, one in every
    /// document is worth one, and nothing is ever worth nothing, because the
    /// logarithm is of one plus the ratio rather than of the ratio.
    #[must_use]
    pub fn bits(&self, docs: u32) -> f64 {
        let df = f64::from(self.docs.max(1));
        (1.0 + f64::from(docs) / df).log2().floor()
    }
}

/// What matched, shaped the way the query was.
///
/// A scorer walks this rather than a flat list because the same numbers add up
/// differently depending on the shape. Everything sums both kinds of branch
/// except `DISMAX`, which takes the best branch of a union and the sum of an
/// intersection, so a document that answers one half of an or is worth what that
/// half is worth and not more.
///
/// A branch carries its own weight rather than having it pushed down into the
/// leaves under it. The two come to the same number, because a sum times a
/// weight is the sum of the weighted parts, and they do not come to the same
/// explanation: `(fox dog) => { $weight: 2.0 }` is measured to print the two on
/// the branch and one on each leaf.
#[derive(Debug, Clone, PartialEq)]
pub enum Found<'a> {
    /// One term in one document.
    Term(Term<'a>),
    /// Every branch matched, as in an intersection or a phrase.
    All {
        /// What the query said the branch as a whole is worth.
        weight: f64,
        /// The branches that matched.
        under: Vec<Found<'a>>,
    },
    /// Any branch could have matched, as in a union.
    Any {
        /// What the query said the branch as a whole is worth.
        weight: f64,
        /// The branches that matched, which is not every branch there is.
        under: Vec<Found<'a>>,
    },
    /// The document matched because everything does, which is a bare `*`.
    ///
    /// There is no term in a wildcard, so there is nothing to weigh by how rare
    /// it is, and a real server scores it as one occurrence of a term whose idf
    /// is one: the length correction and nothing else. Measured, on an index of
    /// three documents holding seven tokens between them, where the two lengths
    /// come back as two different scores and both agree to the last digit with
    /// the same sum with the rarity taken out. The three standard scorers name
    /// it `*` and give it an idf of one where they give a match with no word in
    /// it nothing at all, which is the whole reason it is not a [`Found::Blank`].
    Every,
    /// A match with no word in it, which is a weight and a count and no more.
    ///
    /// Three things come to this and all three are measured. A filter, which is
    /// a numeric range or a geo circle, weighs one and counts one. An optional
    /// branch the document did not answer weighs nothing and counts one. A
    /// negation the document answered weighs one and counts nothing.
    ///
    /// The three standard scorers give all three of them nothing, because they
    /// count words and there is no word here. The rest give them a weight times
    /// a frequency with every rarity taken as one, which is not a rule anybody
    /// would guess and is measured: under `TFIDF` the query `alpha @n:[1 2]`
    /// scores every document exactly twice what `alpha` scores it, and under
    /// `BM25STD` it scores it the same.
    Blank {
        /// What the query said this is worth, which is nothing for an optional
        /// branch that did not answer.
        weight: f64,
        /// How many times it counts, which is nothing for a negation.
        freq: u32,
    },
}

impl<'a> Found<'a> {
    /// A filter that answered, which weighs one and counts one.
    #[must_use]
    pub fn filter() -> Found<'a> {
        Found::Blank {
            weight: 1.0,
            freq: 1,
        }
    }

    /// A negation the document answered, which counts nothing.
    #[must_use]
    pub fn missing() -> Found<'a> {
        Found::Blank {
            weight: 1.0,
            freq: 0,
        }
    }

    /// An optional branch the document did not answer, which is worth nothing.
    #[must_use]
    pub fn skipped() -> Found<'a> {
        Found::Blank {
            weight: 0.0,
            freq: 1,
        }
    }

    /// An intersection or a phrase, at the weight the query gave it.
    #[must_use]
    pub fn all(weight: f64, under: Vec<Found<'a>>) -> Found<'a> {
        Found::All { weight, under }
    }

    /// A union, at the weight the query gave it.
    #[must_use]
    pub fn any(weight: f64, under: Vec<Found<'a>>) -> Found<'a> {
        Found::Any { weight, under }
    }
}

/// One of the nine ways of turning a match into a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scorer {
    /// `BM25STD`, the one a client gets without asking, and plain BM25.
    Bm25,
    /// `BM25STD.NORM`, the same divided by the best score in the answer.
    Norm,
    /// `BM25STD.TANH`, the same pushed into nought to one by a tangent.
    Tanh,
    /// `BM25`, the older one, with a different idf and no length correction.
    Old,
    /// `TFIDF`, the frequency over the largest frequency in the document.
    TfIdf,
    /// `TFIDF.DOCNORM`, the frequency over the document's length.
    Length,
    /// `DISMAX`, the frequency and nothing else.
    DisMax,
    /// `DOCSCORE`, what the client said the document is worth and no more.
    Worth,
    /// `HAMMING`, how close the document's payload is to the query's.
    Hamming,
}

impl Scorer {
    /// The scorer a client gets when it does not name one.
    #[must_use]
    pub const fn default_scorer() -> Scorer {
        Scorer::Bm25
    }

    /// The scorer under a name, or nothing when there is no such scorer.
    ///
    /// The names are matched exactly, because a real server refuses a lowercase
    /// spelling of one of its own scorers rather than taking it.
    #[must_use]
    pub fn named(name: &[u8]) -> Option<Scorer> {
        Some(match name {
            b"BM25STD" => Scorer::Bm25,
            b"BM25STD.NORM" => Scorer::Norm,
            b"BM25STD.TANH" => Scorer::Tanh,
            b"BM25" => Scorer::Old,
            b"TFIDF" => Scorer::TfIdf,
            b"TFIDF.DOCNORM" => Scorer::Length,
            b"DISMAX" => Scorer::DisMax,
            b"DOCSCORE" => Scorer::Worth,
            b"HAMMING" => Scorer::Hamming,
            _ => return None,
        })
    }

    /// The name a client asks for this scorer by.
    #[must_use]
    pub const fn name(self) -> &'static [u8] {
        match self {
            Scorer::Bm25 => b"BM25STD",
            Scorer::Norm => b"BM25STD.NORM",
            Scorer::Tanh => b"BM25STD.TANH",
            Scorer::Old => b"BM25",
            Scorer::TfIdf => b"TFIDF",
            Scorer::Length => b"TFIDF.DOCNORM",
            Scorer::DisMax => b"DISMAX",
            Scorer::Worth => b"DOCSCORE",
            Scorer::Hamming => b"HAMMING",
        }
    }

    /// What one document that matched is worth.
    ///
    /// `want` is the payload the query carried, which only `HAMMING` reads, and
    /// `slop` is how far apart the query's words landed in this document, which
    /// only the three older scorers divide by. One is the slop of a query with
    /// nothing to measure, so a caller that does not work it out passes one.
    #[must_use]
    pub fn of(
        self,
        facts: &Facts,
        doc: &Doc,
        found: &Found<'_>,
        want: Option<&[u8]>,
        slop: u32,
    ) -> f64 {
        match self {
            Scorer::Worth => doc.score,
            Scorer::Hamming => near(doc.payload.as_deref(), want),
            // The only scorer that does not care what the client said the
            // document is worth. A document scored zero still scores here.
            Scorer::DisMax => self.walk(facts, doc, found),
            Scorer::Tanh => (doc.score * self.walk(facts, doc, found) / facts.tanh).tanh(),
            _ => match self.divisor(doc, slop) {
                // Nothing to divide by is a document with no words in it under
                // a scorer that divides by how many it has, and it scores
                // nothing rather than answering with an infinity.
                divisor if divisor <= 0.0 => 0.0,
                divisor => doc.score * self.walk(facts, doc, found) / divisor,
            },
        }
    }

    /// What this scorer divides the whole document's worth by at the end.
    ///
    /// Measured, and the reason it is here rather than inside a leaf: a real
    /// server prints every leaf undivided and divides once at the top, so the
    /// two halves of `TFIDF` come out as `words TFIDF 9.00 / norm 3 / slop 1`
    /// rather than as three ninths. A sum divided is the sum of the divided
    /// parts, so the number is the same either way and only the explanation
    /// tells them apart.
    #[must_use]
    pub fn divisor(self, doc: &Doc, slop: u32) -> f64 {
        let slop = f64::from(slop.max(1));
        match self {
            Scorer::TfIdf => f64::from(doc.top) * slop,
            Scorer::Length => f64::from(doc.tokens) * slop,
            Scorer::Old => slop,
            _ => 1.0,
        }
    }

    /// Whether how far apart the words landed changes what this scorer says.
    ///
    /// Only the three that divide by it, so a caller can skip working the slop
    /// out for the six that do not, which is most of the traffic because the
    /// scorer nobody names is one of the six.
    #[must_use]
    pub fn divides(self) -> bool {
        matches!(self, Scorer::TfIdf | Scorer::Length | Scorer::Old)
    }

    /// Whether this scorer has to see the whole answer before any of it is
    /// worth reading, which is the one thing that changes how an aggregation
    /// hands its rows back.
    pub fn settles(self) -> bool {
        self == Scorer::Norm
    }

    /// Divides a whole answer by its best score, for the scorer that wants it.
    ///
    /// This is the one thing a scorer cannot do a document at a time, so it
    /// happens once the answer is gathered. Every other scorer leaves it alone,
    /// and so does this one when the best score is not above zero, because a
    /// page of zeroes stays a page of zeroes rather than becoming ones.
    pub fn settle(self, scores: &mut [f64]) {
        if self != Scorer::Norm {
            return;
        }
        let top = scores.iter().copied().fold(0.0_f64, f64::max);
        if top <= 0.0 {
            return;
        }
        for score in scores {
            *score /= top;
        }
    }

    /// Adds up a match, which is a sum except where `DISMAX` takes the best.
    ///
    /// What comes back is undivided, so it is the number a real server prints
    /// after `words TFIDF` or `children BM25` rather than the document's score.
    /// [`Scorer::of`] is the one that divides.
    #[must_use]
    pub fn walk(self, facts: &Facts, doc: &Doc, found: &Found<'_>) -> f64 {
        match found {
            Found::Term(term) => self.one(facts, doc, term.weight, term.freq, Some(term)),
            Found::Every => self.one(facts, doc, 1.0, 1, None),
            // The three standard scorers count words and there is no word in
            // any of these, so they add nothing to them. The rest count a
            // match, and a filter is one, so it is worth to them what a
            // wildcard is worth and a negation is worth nothing either way.
            Found::Blank { weight, freq } => match self {
                Scorer::Bm25 | Scorer::Norm | Scorer::Tanh => 0.0,
                _ => self.one(facts, doc, *weight, *freq, None),
            },
            // Added up from nought rather than summed, because the sum of no
            // doubles at all is negative zero and a real server never answers
            // a score with a sign on the front of it.
            Found::All { weight, under } => {
                weight
                    * under
                        .iter()
                        .fold(0.0_f64, |sum, f| sum + self.walk(facts, doc, f))
            }
            Found::Any { weight, under } if self == Scorer::DisMax => {
                weight
                    * under
                        .iter()
                        .map(|f| self.walk(facts, doc, f))
                        .fold(0.0_f64, f64::max)
            }
            Found::Any { weight, under } => {
                weight
                    * under
                        .iter()
                        .fold(0.0_f64, |sum, f| sum + self.walk(facts, doc, f))
            }
        }
    }

    /// What one leaf of a match is worth to this scorer, undivided.
    ///
    /// No term at all is a wildcard or one of the three blanks, and either way
    /// there is no word to weigh by how rare it is, so every rarity is one and
    /// what is left is the weight, the count and whatever the scorer does with
    /// the length.
    #[must_use]
    pub fn one(
        self,
        facts: &Facts,
        doc: &Doc,
        weight: f64,
        count: u32,
        term: Option<&Term<'_>>,
    ) -> f64 {
        let freq = f64::from(count);
        let idf = || term.map_or(1.0, |term| term.idf(facts.docs));
        let bits = || term.map_or(1.0, |term| term.bits(facts.docs));
        weight
            * match self {
                Scorer::Bm25 | Scorer::Norm | Scorer::Tanh => {
                    let k1 = f64::from(K1);
                    let long = long(f64::from(doc.tokens), facts.average());
                    idf() * (freq * (k1 + 1.0)) / (freq + k1 * long)
                }
                // The average length where the document's own length belongs,
                // which is the slip this scorer is kept around for.
                Scorer::Old => {
                    let long = f64::from(K1) * (1.0 - OLD_B + OLD_B * facts.average());
                    bits() * freq / (freq + long)
                }
                // The two of these divide by the document rather than by the
                // term, and [`Scorer::divisor`] is where that happens now.
                Scorer::TfIdf | Scorer::Length => bits() * freq,
                Scorer::DisMax => freq,
                Scorer::Worth | Scorer::Hamming => 0.0,
            }
    }
}

/// The length correction, which is one for a document of average length.
///
/// An empty index has no average to compare against, and rather than divide by
/// nothing it corrects by nothing, which leaves the score to the idf.
fn long(tokens: f64, average: f64) -> f64 {
    if average <= 0.0 {
        return 1.0;
    }
    1.0 - B + B * tokens / average
}

/// How far apart two payloads are, as a length and a count of bits.
///
/// Nothing without a payload on either side, nothing when the two are different
/// lengths, and nothing once they differ in [`CLOSE`] bits or more, so this
/// answers about payloads that are nearly the same and says nothing at all about
/// the rest. A real server tells all three of those apart in the score and not
/// in what it says about it, and prints the same line about a payload eight bits
/// away as it does about one of the wrong length. That is measured.
#[must_use]
pub fn apart(payload: Option<&[u8]>, want: Option<&[u8]>) -> Option<(usize, u32)> {
    let (payload, want) = (payload?, want?);
    if payload.len() != want.len() || want.is_empty() {
        return None;
    }
    let mut bits = 0;
    for (a, b) in payload.iter().zip(want) {
        bits += (a ^ b).count_ones();
        if bits >= CLOSE {
            return None;
        }
    }
    Some((want.len(), bits))
}

/// How close two payloads are, as one over one plus the bits they differ in.
fn near(payload: Option<&[u8]>, want: Option<&[u8]>) -> f64 {
    apart(payload, want).map_or(0.0, |(_, bits)| 1.0 / f64::from(bits + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How close two doubles have to be to count as the same answer.
    ///
    /// The measured values are what a real server printed, so they are exact to
    /// the last digit a double holds and the check can be tight.
    fn same(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= f64::EPSILON * want.abs().max(1.0) * 4.0,
            "got {got:?} want {want:?}"
        );
    }

    fn doc(score: f64, tokens: u32, top: u32) -> Doc {
        let mut d = Doc::new(b"k", score);
        d.tokens = tokens;
        d.top = top;
        d
    }

    /// Two documents of different lengths, one term in each, against the four
    /// documents and ten tokens of the index they are in. Both numbers are what
    /// a real server answered for the same corpus.
    #[test]
    fn the_default_scorer_is_bm25_to_the_last_digit() {
        let facts = Facts::new(4, 10);
        let found = Found::Term(Term::new(1, 1.0, 2));
        let short = Scorer::Bm25.of(&facts, &doc(1.0, 1, 1), &found, None, 1);
        let long = Scorer::Bm25.of(&facts, &doc(1.0, 7, 1), &found, None, 1);
        same(short, 0.9186287989106708);
        same(long, 0.39919470825949166);
    }

    /// A term in a field of weight three is three times as frequent, and that is
    /// the whole of what a weight does to a score. Both of these came off a real
    /// server, from an index of three documents holding thirty six tokens.
    #[test]
    fn a_field_weight_is_just_more_of_the_term() {
        let facts = Facts::new(3, 36);
        let d = doc(1.0, 16, 3);
        let plain = Found::Term(Term::new(1, 1.0, 2));
        let heavy = Found::Term(Term::new(3, 1.0, 2));
        same(
            Scorer::Bm25.of(&facts, &d, &plain, None, 1),
            0.4136031928397866,
        );
        same(
            Scorer::Bm25.of(&facts, &d, &heavy, None, 1),
            0.6893386620374727,
        );
    }

    /// A weight on the query multiplies the term's share, so asking for a term
    /// five times over is worth five times as much.
    #[test]
    fn a_query_weight_multiplies_what_the_term_is_worth() {
        let facts = Facts::new(3, 36);
        let d = doc(1.0, 16, 3);
        let five = Found::Term(Term::new(1, 5.0, 2));
        same(
            Scorer::Bm25.of(&facts, &d, &five, None, 1),
            2.0680159641989326,
        );
    }

    /// The older scorer ignores how long the document is, which is the point of
    /// pinning it: two documents of wildly different lengths score the same.
    /// The exact digits are a real server's, single precision fold and all.
    #[test]
    fn the_older_scorer_does_not_care_how_long_a_document_is() {
        let facts = Facts::new(4, 10);
        let found = Found::Term(Term::new(1, 1.0, 2));
        let short = Scorer::Old.of(&facts, &doc(1.0, 1, 1), &found, None, 1);
        let long = Scorer::Old.of(&facts, &doc(1.0, 7, 1), &found, None, 1);
        same(short, 0.3225806364779916);
        same(long, 0.3225806364779916);

        // A second corpus, three documents and thirty six tokens, where the same
        // term is once in one document and three times in another.
        let facts = Facts::new(3, 36);
        let d = doc(1.0, 16, 3);
        let once = Found::Term(Term::new(1, 1.0, 2));
        let thrice = Found::Term(Term::new(3, 1.0, 2));
        same(
            Scorer::Old.of(&facts, &d, &once, None, 1),
            0.11363635963398577,
        );
        same(
            Scorer::Old.of(&facts, &d, &thrice, None, 1),
            0.27777776980596336,
        );
    }

    /// The older idf is a whole number of bits, so a term in three documents out
    /// of sixteen and a term in five of them are worth the same.
    #[test]
    fn the_older_idf_is_a_whole_number_of_bits() {
        let bits = |df: u32, n: u32| Term::new(1, 1.0, df).bits(n);
        assert!((bits(1, 16) - 4.0).abs() < f64::EPSILON);
        assert!((bits(2, 16) - 3.0).abs() < f64::EPSILON);
        assert!((bits(3, 16) - 2.0).abs() < f64::EPSILON);
        assert!((bits(5, 16) - 2.0).abs() < f64::EPSILON);
        assert!((bits(6, 16) - 1.0).abs() < f64::EPSILON);
        assert!((bits(16, 16) - 1.0).abs() < f64::EPSILON);
        assert!((bits(7, 100) - 3.0).abs() < f64::EPSILON);
        assert!((bits(100, 100) - 1.0).abs() < f64::EPSILON);
        // A term in no documents cannot be asked about, but the division still
        // has to answer rather than divide by nothing.
        assert!(bits(0, 16).is_finite());
    }

    /// One divides the frequency by the largest frequency in the document and
    /// the other divides it by the document's length, and both of them fall back
    /// to nothing rather than dividing by nothing.
    #[test]
    fn the_two_older_normalisations_divide_by_different_things() {
        let facts = Facts::new(3, 12);
        let found = Found::Term(Term::new(1, 1.0, 2));
        // One document has the term once among three of something else, the
        // other has it twice among five.
        let a = doc(1.0, 4, 3);
        let b = doc(1.0, 7, 5);
        let twice = Found::Term(Term::new(2, 1.0, 2));
        same(Scorer::TfIdf.of(&facts, &a, &found, None, 1), 1.0 / 3.0);
        same(Scorer::TfIdf.of(&facts, &b, &twice, None, 1), 0.4);
        same(Scorer::Length.of(&facts, &a, &found, None, 1), 0.25);
        same(Scorer::Length.of(&facts, &b, &twice, None, 1), 2.0 / 7.0);
        let empty = doc(1.0, 0, 0);
        assert_eq!(Scorer::TfIdf.of(&facts, &empty, &found, None, 1), 0.0);
        assert_eq!(Scorer::Length.of(&facts, &empty, &found, None, 1), 0.0);
    }

    /// What the client says a document is worth multiplies every scorer except
    /// the one that only looks at frequencies, and a document worth nothing
    /// scores nothing however well it matched.
    #[test]
    fn what_a_document_is_worth_multiplies_all_but_one() {
        let facts = Facts::new(4, 16);
        let found = Found::Term(Term::new(1, 1.0, 4));
        let plain = Scorer::Bm25.of(&facts, &doc(1.0, 4, 1), &found, None, 1);
        same(plain, 0.10536051565782635);
        same(
            Scorer::Bm25.of(&facts, &doc(2.0, 4, 1), &found, None, 1),
            plain * 2.0,
        );
        same(
            Scorer::Bm25.of(&facts, &doc(0.5, 4, 1), &found, None, 1),
            plain * 0.5,
        );
        assert_eq!(
            Scorer::Bm25.of(&facts, &doc(0.0, 4, 1), &found, None, 1),
            0.0
        );
        // A half that is not quite a half, because the older scorer folds its
        // constant in single precision and this is what a real server answers.
        same(
            Scorer::Old.of(&facts, &doc(2.0, 4, 1), &found, None, 1),
            0.49999998509883925,
        );
        same(
            Scorer::TfIdf.of(&facts, &doc(2.0, 4, 1), &found, None, 1),
            2.0,
        );
        same(
            Scorer::Worth.of(&facts, &doc(2.0, 4, 1), &found, None, 1),
            2.0,
        );
        // Every document is worth the same to this one whatever it was told.
        for worth in [0.0, 0.5, 1.0, 2.0] {
            same(
                Scorer::DisMax.of(&facts, &doc(worth, 4, 1), &found, None, 1),
                1.0,
            );
        }
    }

    /// Two terms add up when both had to match and when either could, except
    /// that the best branch is all a union is worth to `DISMAX`.
    #[test]
    fn a_union_adds_up_except_for_the_one_that_takes_the_best() {
        let facts = Facts::new(4, 16);
        let d = doc(1.0, 4, 1);
        let one = Term::new(1, 1.0, 2);
        let both = vec![Found::Term(one), Found::Term(one)];
        let each = Scorer::Bm25.of(&facts, &d, &Found::Term(one), None, 1);
        // A term in half of four documents, in a document of average length, is
        // worth its idf and nothing more, and that idf is the log of two.
        same(each, std::f64::consts::LN_2);
        same(
            Scorer::Bm25.of(&facts, &d, &Found::all(1.0, both.clone()), None, 1),
            each * 2.0,
        );
        same(
            Scorer::Bm25.of(&facts, &d, &Found::any(1.0, both.clone()), None, 1),
            each * 2.0,
        );
        same(
            Scorer::DisMax.of(&facts, &d, &Found::all(1.0, both.clone()), None, 1),
            2.0,
        );
        same(
            Scorer::DisMax.of(&facts, &d, &Found::any(1.0, both), None, 1),
            1.0,
        );
    }

    /// Nothing matched is nothing, which is what a document pulled in by a
    /// negative or an optional clause scores.
    #[test]
    fn a_document_that_matched_nothing_scores_nothing() {
        let facts = Facts::new(4, 16);
        let d = doc(1.0, 4, 1);
        for scorer in [
            Scorer::Bm25,
            Scorer::Old,
            Scorer::TfIdf,
            Scorer::Length,
            Scorer::DisMax,
        ] {
            assert_eq!(
                scorer.of(&facts, &d, &Found::all(1.0, Vec::new()), None, 1),
                0.0
            );
            assert_eq!(
                scorer.of(&facts, &d, &Found::any(1.0, Vec::new()), None, 1),
                0.0
            );
        }
    }

    /// The tangent one is the default one pushed into nought to one, and the
    /// factor it divides by is a setting rather than a constant.
    #[test]
    fn the_tangent_scorer_flattens_the_default_one() {
        let facts = Facts::new(4, 10);
        let found = Found::Term(Term::new(1, 1.0, 2));
        let d = doc(1.0, 1, 1);
        same(
            Scorer::Tanh.of(&facts, &d, &found, None, 1),
            0.22570304007310169,
        );
        let two = facts.tanh(2.0);
        same(
            Scorer::Tanh.of(&two, &d, &found, None, 1),
            0.42952526332812657,
        );
        // What the document is worth goes in before the tangent, not after.
        same(
            Scorer::Tanh.of(&facts, &doc(2.0, 1, 1), &found, None, 1),
            0.42952526332812657,
        );
    }

    /// The normalising one divides the whole answer by its best score, which
    /// cannot be done a document at a time, and it leaves an answer of nothing
    /// alone rather than turning it into ones.
    #[test]
    fn normalising_happens_over_the_whole_answer() {
        let mut scores = vec![0.9186287989106708, 0.44072941826638706];
        Scorer::Norm.settle(&mut scores);
        same(scores[0], 1.0);
        same(scores[1], 0.47976878015256347);

        let mut zeroes = vec![0.0, 0.0];
        Scorer::Norm.settle(&mut zeroes);
        assert_eq!(zeroes, [0.0, 0.0]);

        // Every other scorer leaves the answer as it found it.
        let mut left = vec![2.0, 1.0];
        Scorer::Bm25.settle(&mut left);
        assert_eq!(left, [2.0, 1.0]);
    }

    /// The payload one answers about payloads that are nearly the same, and
    /// nothing at all about payloads that are not.
    #[test]
    fn the_payload_scorer_is_a_near_neighbour_test() {
        let facts = Facts::new(4, 16);
        let found = Found::all(1.0, Vec::new());
        let mut d = doc(1.0, 4, 1);
        d.payload = Some(Box::from(&b"\x00\x00"[..]));
        let of = |want: &[u8], d: &Doc| Scorer::Hamming.of(&facts, d, &found, Some(want), 1);
        same(of(b"\x00\x00", &d), 1.0);
        same(of(b"\x00\x01", &d), 0.5);
        same(of(b"\x00\x03", &d), 1.0 / 3.0);
        same(of(b"\x00\x7f", &d), 0.125);
        // Eight bits apart is where it stops answering.
        same(of(b"\x00\xff", &d), 0.0);
        same(of(b"\xff\xff", &d), 0.0);
        // A different length is not a distance, and neither is no payload.
        same(of(b"\x00", &d), 0.0);
        same(of(b"", &d), 0.0);
        same(Scorer::Hamming.of(&facts, &d, &found, None, 1), 0.0);
        same(of(b"\x00\x00", &doc(1.0, 4, 1)), 0.0);
    }

    /// The names are the names a real server takes, exactly as it takes them,
    /// and it refuses its own scorer spelled in lowercase.
    #[test]
    fn the_names_are_matched_as_they_are_written() {
        for scorer in [
            Scorer::Bm25,
            Scorer::Norm,
            Scorer::Tanh,
            Scorer::Old,
            Scorer::TfIdf,
            Scorer::Length,
            Scorer::DisMax,
            Scorer::Worth,
            Scorer::Hamming,
        ] {
            assert_eq!(Scorer::named(scorer.name()), Some(scorer));
        }
        assert_eq!(Scorer::named(b"bm25std"), None);
        assert_eq!(Scorer::named(b"Tfidf"), None);
        assert_eq!(Scorer::named(b""), None);
        assert_eq!(Scorer::default_scorer(), Scorer::Bm25);
    }

    /// An empty index has no average length to correct against and answers with
    /// a number anyway rather than dividing by nothing.
    #[test]
    fn an_empty_index_still_answers() {
        let facts = Facts::new(0, 0);
        assert_eq!(facts.average(), 0.0);
        let d = doc(1.0, 0, 0);
        let found = Found::Term(Term::new(1, 1.0, 0));
        for scorer in [Scorer::Bm25, Scorer::Old, Scorer::TfIdf, Scorer::Length] {
            assert!(scorer.of(&facts, &d, &found, None, 1).is_finite());
        }
    }

    /// A term said to be in more documents than the index holds is nonsense a
    /// caller should not produce, and the idf still has to come back with a
    /// number rather than the logarithm of something negative.
    #[test]
    fn an_impossible_document_count_still_gives_a_number() {
        let term = Term::new(1, 1.0, 100);
        assert!(term.idf(4).is_finite());
        assert!(term.idf(4) > 0.0);
    }
}
