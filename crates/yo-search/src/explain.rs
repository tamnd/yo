//! Explaining a score: the same sum a scorer does, written out as the tree of
//! strings `EXPLAINSCORE` sends back.
//!
//! ```
//! use yo_search::docs::Doc;
//! use yo_search::explain::{Note, Why};
//! use yo_search::score::{Facts, Found, Scorer, Term};
//!
//! let facts = Facts::new(4, 10);
//! let mut doc = Doc::new(b"book:1", 1.0);
//! doc.tokens = 1;
//! doc.top = 1;
//! let found = Found::Term(Term::new(1, 1.0, 2).about(b"fox"));
//! let note = Why::new(Scorer::TfIdf, facts).note(&doc, &found, 1);
//! let Note::Under(head, under) = note else { panic!("a top and its child") };
//! assert_eq!(head, "Final TFIDF : words TFIDF 1.00 * document score 1.00 / norm 1 / slop 1");
//! assert_eq!(under[0], Note::Line("(TFIDF 1.00 = Weight 1.00 * TF 1 * IDF 1.00)".into()));
//! ```
//!
//! # Every string here is measured
//!
//! None of this is a description of the sum in words that somebody thought
//! read well. Every line is the format string a real server uses, down to the
//! spacing, the two decimal places, which numbers are printed whole, and the
//! constants that are written out as text rather than filled in. `k1 1.2` and
//! `b 0.75` are literal, so a server that was built with a different k1 would
//! still print 1.2, and that is reproduced here because a client that reads
//! these lines is reading them as strings.
//!
//! Two of them are odd enough to be worth pointing at. `BM25STD.NORM` writes
//! its head with no space before the colon where every other one has a space
//! either side. And the older `BM25` leaves the whole `IDF` term out of its
//! line rather than printing an idf of one, when there is no word to be rare.
//!
//! # Where the shape comes from
//!
//! The tree is the shape of what matched and not the shape of the query, so a
//! branch that no document answered is not in it, and a union that stands for
//! several terms prints only the ones this document holds. Two of the nine
//! scorers wrap the whole thing in another layer: `BM25STD.TANH` puts its own
//! top over the ordinary `BM25STD` top, and `BM25STD.NORM` puts its top in
//! place of it, so the tangent has one more level than the normalisation does
//! for the same query.
//!
//! `DOCSCORE` and `HAMMING` never recurse at all, because neither of them looks
//! at what matched, and `DISMAX` has no top layer, so a one word query under it
//! answers a bare string where the other eight answer a pair.

use crate::docs::Doc;
use crate::score::{Facts, Found, Scorer, Term, apart};

/// One line of an explanation, with whatever hangs under it.
///
/// This is the reply shape rather than a shape of its own: a line goes on the
/// wire as a string and a line with children goes as an array of two, the head
/// and the list. Nothing else is in it, which is why there is no room here for
/// a value or a scorer. What a client gets is text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Note {
    /// A line with nothing under it, which is a leaf or a whole explanation
    /// from one of the three scorers that do not recurse.
    Line(String),
    /// A line and the lines that make it up.
    Under(String, Vec<Note>),
}

/// What a leaf has to say for itself, which is not the same in all three cases.
enum Leaf<'a> {
    /// A term the index knows, which is the only kind with a name and a rarity.
    Word(&'a Term<'a>),
    /// A wildcard, which the newer scorers name `*` and treat as one
    /// occurrence of a term that is in every document.
    Every,
    /// A match with no word in it, which is a filter, an optional branch that
    /// found nothing, or a negation that found nothing.
    Blank {
        /// What the query said this is worth.
        weight: f64,
        /// How many times it counts.
        freq: u32,
    },
}

impl Leaf<'_> {
    /// What the query said this is worth.
    fn weight(&self) -> f64 {
        match self {
            Leaf::Word(term) => term.weight,
            Leaf::Every => 1.0,
            Leaf::Blank { weight, .. } => *weight,
        }
    }

    /// How many times it counts.
    fn freq(&self) -> u32 {
        match self {
            Leaf::Word(term) => term.freq,
            Leaf::Every | Leaf::Blank { freq: 1, .. } => 1,
            Leaf::Blank { freq, .. } => *freq,
        }
    }

    /// The term behind it, for the scorers that weigh a term by how rare it is.
    fn term(&self) -> Option<&Term<'_>> {
        match self {
            Leaf::Word(term) => Some(term),
            _ => None,
        }
    }
}

/// Everything an explanation needs beside the document and what matched.
///
/// The same three things a score needs, since an explanation is a score with
/// the working shown, plus the best score in the answer, which only
/// `BM25STD.NORM` reads and which nobody can know until the whole answer is
/// gathered.
#[derive(Debug, Clone, Copy)]
pub struct Why<'a> {
    scorer: Scorer,
    facts: Facts,
    want: Option<&'a [u8]>,
    best: f64,
}

impl<'a> Why<'a> {
    /// An explanation from one scorer over one index.
    #[must_use]
    pub fn new(scorer: Scorer, facts: Facts) -> Why<'a> {
        Why {
            scorer,
            facts,
            want: None,
            best: 0.0,
        }
    }

    /// The same, knowing the payload the query carried, which `HAMMING` reads.
    #[must_use]
    pub fn about(self, want: Option<&'a [u8]>) -> Why<'a> {
        Why { want, ..self }
    }

    /// The same, knowing the best score in the answer.
    #[must_use]
    pub fn over(self, best: f64) -> Why<'a> {
        Why { best, ..self }
    }

    /// Why this document scored what it scored.
    ///
    /// `slop` is how far apart the query's words landed, the same number
    /// [`Scorer::of`] divides by, and it is printed even by the scorers that do
    /// not divide by it in the one place they print it at all.
    #[must_use]
    pub fn note(&self, doc: &Doc, found: &Found<'_>, slop: u32) -> Note {
        let words = self.scorer.walk(&self.facts, doc, found);
        let slop = slop.max(1);
        match self.scorer {
            Scorer::Worth => Note::Line(format!("Document's score is {:.2}", doc.score)),
            Scorer::Hamming => Note::Line(self.payload(doc)),
            // The one with no top layer at all, so a query of one word answers
            // one line and not a line with a line under it.
            Scorer::DisMax => self.body(doc, found),
            // A layer over the whole of `BM25STD`, its own top line included.
            Scorer::Tanh => {
                let plain = Why {
                    scorer: Scorer::Bm25,
                    ..*self
                };
                let whole = doc.score * words;
                Note::Under(
                    format!(
                        "Final Normalized BM25 : tanh(stretch factor 1/{} * Final BM25 {whole:.2})",
                        self.facts.tanh
                    ),
                    vec![plain.note(doc, found, slop)],
                )
            }
            // A layer in place of the top of `BM25STD` rather than over it, so
            // the terms hang straight off this line.
            Scorer::Norm => {
                let whole = doc.score * words;
                // An answer whose best score is nothing is left alone rather
                // than divided, which is what [`Scorer::settle`] does with it,
                // and it prints as the nothing it is.
                let norm = if self.best > 0.0 {
                    whole / self.best
                } else {
                    whole
                };
                Note::Under(
                    format!(
                        "Final BM25STD.NORM: {norm:.2} = Original Score: {whole:.2} / Max Score: {:.2}",
                        self.best
                    ),
                    vec![self.body(doc, found)],
                )
            }
            Scorer::Bm25 => Note::Under(
                format!(
                    "Final BM25 : words BM25 {words:.2} * document score {:.2}",
                    doc.score
                ),
                vec![self.body(doc, found)],
            ),
            Scorer::Old => Note::Under(
                format!(
                    "Final BM25 : words BM25 {words:.2} * document score {:.2} / slop {slop}",
                    doc.score
                ),
                vec![self.body(doc, found)],
            ),
            Scorer::TfIdf | Scorer::Length => {
                // The two halves of the same scorer, which print the same head
                // and divide by different things: the largest frequency in the
                // document, or how long the document is.
                let norm = match self.scorer {
                    Scorer::TfIdf => doc.top,
                    _ => doc.tokens,
                };
                Note::Under(
                    format!(
                        "Final TFIDF : words TFIDF {words:.2} * document score {:.2} / norm {norm} / slop {slop}",
                        doc.score
                    ),
                    vec![self.body(doc, found)],
                )
            }
        }
    }

    /// What matched, one line per branch and one per leaf.
    fn body(&self, doc: &Doc, found: &Found<'_>) -> Note {
        match found {
            Found::Term(term) => Note::Line(self.leaf(doc, &Leaf::Word(term))),
            Found::Every => Note::Line(self.leaf(doc, &Leaf::Every)),
            Found::Blank { weight, freq } => Note::Line(self.leaf(
                doc,
                &Leaf::Blank {
                    weight: *weight,
                    freq: *freq,
                },
            )),
            Found::All { weight, under } => self.branch(doc, *weight, under, false),
            Found::Any { weight, under } => self.branch(doc, *weight, under, true),
        }
    }

    /// One branch and everything under it.
    fn branch(&self, doc: &Doc, weight: f64, under: &[Found<'_>], any: bool) -> Note {
        // What the children come to before the branch's own weight, which is
        // the number the head prints. Every scorer adds them up except the one
        // that takes the best branch of a union.
        let each = under.iter().map(|f| self.scorer.walk(&self.facts, doc, f));
        let sum = match any && self.scorer == Scorer::DisMax {
            true => each.fold(0.0_f64, f64::max),
            false => each.sum(),
        };
        let head = match self.scorer {
            Scorer::TfIdf | Scorer::Length => {
                format!("(Weight {weight:.2} * total children TFIDF {sum:.2})")
            }
            // The one branch line with no brackets round it, and the only one
            // that prints what it came to as well as what it is made of.
            Scorer::DisMax => format!(
                "{:.2} = Weight {weight:.2} * children DISMAX {sum:.2}",
                weight * sum
            ),
            _ => format!("(Weight {weight:.2} * children BM25 {sum:.2})"),
        };
        Note::Under(head, under.iter().map(|f| self.body(doc, f)).collect())
    }

    /// One leaf, which is the only line that differs between all six of the
    /// scorers that have leaves at all.
    fn leaf(&self, doc: &Doc, leaf: &Leaf<'_>) -> String {
        let (weight, freq) = (leaf.weight(), leaf.freq());
        let term = leaf.term();
        let value = self.scorer.one(&self.facts, doc, weight, freq, term);
        let average = self.facts.average();
        match self.scorer {
            Scorer::TfIdf | Scorer::Length => match term {
                Some(term) => format!(
                    "(TFIDF {value:.2} = Weight {weight:.2} * TF {freq} * IDF {:.2})",
                    term.bits(self.facts.docs)
                ),
                // A wildcard and the three blanks all print this, because none
                // of them has a word in it to be rare.
                None => format!("(TFIDF {value:.2} = Weight {weight:.2} * Frequency {freq})"),
            },
            // The older `BM25`, which says nothing at all about a match that
            // counts no times, drops the whole idf when there is no word, and
            // corrects by the average length where the newer one corrects by
            // the document's own.
            Scorer::Old if freq == 0 => "Frequency 0 -> value 0".to_string(),
            Scorer::Old => {
                let long =
                    format!("F {freq} + k1 1.2 * (1 - b 0.5 + b 0.5 * Average Len {average:.2})");
                match term {
                    Some(term) => format!(
                        "({value:.2} = Weight {weight:.2} * IDF {:.2} * F {freq} / ({long}))",
                        term.bits(self.facts.docs)
                    ),
                    None => format!("({value:.2} = Weight {weight:.2} * F {freq} / ({long}))"),
                }
            }
            // The three that count words and nothing else, so a match with no
            // word in it is worth saying nothing about.
            Scorer::Bm25 | Scorer::Norm | Scorer::Tanh => match leaf {
                Leaf::Blank { .. } => "Irrelevant token -> score is 0".to_string(),
                _ => {
                    // A wildcard is named and given the rarity of a term that is
                    // in every document, which is the one place any of this
                    // prints a name at all.
                    let (name, idf) = match term {
                        Some(term) => (term.name.unwrap_or_default(), term.idf(self.facts.docs)),
                        None => (b"*".as_slice(), 1.0),
                    };
                    let freq = f64::from(freq);
                    format!(
                        "{}: ({value:.2} = Weight {weight:.2} * IDF {idf:.2} * (F {freq:.2} * (k1 1.2 + 1)) / (F {freq:.2} + k1 1.2 * (1 - b 0.75 + b 0.75 * Doc Len {} / Average Doc Len {average:.2})))",
                        String::from_utf8_lossy(name),
                        doc.tokens
                    )
                }
            },
            Scorer::DisMax => format!("DISMAX {value:.2} = Weight {weight:.2} * Frequency {freq}"),
            // Neither of these ever reaches a leaf, because neither of them
            // looks at what matched.
            Scorer::Worth | Scorer::Hamming => String::new(),
        }
    }

    /// How close the document's payload is to the query's, or that it is not
    /// close at all.
    ///
    /// The one message covers three different reasons: no payload on the
    /// document, no payload on the query, a payload of another length, and a
    /// payload of the same length that is too many bits away. Only the third of
    /// those is what the words say, and the wording is a real server's.
    fn payload(&self, doc: &Doc) -> String {
        match apart(doc.payload.as_deref(), self.want) {
            Some((len, bits)) => format!(
                "String length is {len}. Bit count is {bits}. Result is (1 / count + 1) = {:.2}",
                1.0 / f64::from(bits + 1)
            ),
            None => "Payloads provided to scorer vary in length".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document of this length with this largest frequency, worth this much.
    fn doc(score: f64, tokens: u32, top: u32) -> Doc {
        let mut doc = Doc::new(b"k", score);
        doc.tokens = tokens;
        doc.top = top;
        doc
    }

    /// The head of a note, whichever kind it is.
    fn head(note: &Note) -> &str {
        match note {
            Note::Line(line) | Note::Under(line, _) => line,
        }
    }

    /// The children of a note, or nothing when it has none.
    fn under(note: &Note) -> &[Note] {
        match note {
            Note::Line(_) => &[],
            Note::Under(_, under) => under,
        }
    }

    /// The whole of one line a real server sent, for the query `fox` under
    /// `TFIDF` over the twelve document index the rest of these are measured
    /// on. Twelve documents, 455 tokens, `fox` in seven of them, and the
    /// document is 21 tokens long with a largest frequency of three.
    #[test]
    fn a_term_under_tfidf_prints_what_a_real_server_printed() {
        let facts = Facts::new(12, 455);
        let found = Found::Term(Term::new(3, 1.0, 7).about(b"fox"));
        let note = Why::new(Scorer::TfIdf, facts).note(&doc(1.0, 21, 3), &found, 1);
        assert_eq!(
            head(&note),
            "Final TFIDF : words TFIDF 3.00 * document score 1.00 / norm 3 / slop 1"
        );
        assert_eq!(
            head(&under(&note)[0]),
            "(TFIDF 3.00 = Weight 1.00 * TF 3 * IDF 1.00)"
        );
    }

    /// The same document and the same word under the scorer nobody names,
    /// which is the only family that says which word it was.
    #[test]
    fn a_term_under_bm25std_names_itself() {
        let facts = Facts::new(12, 455);
        let found = Found::Term(Term::new(3, 1.0, 7).about(b"fox"));
        let note = Why::new(Scorer::Bm25, facts).note(&doc(1.0, 21, 3), &found, 1);
        assert_eq!(
            head(&note),
            "Final BM25 : words BM25 0.96 * document score 1.00"
        );
        assert_eq!(
            head(&under(&note)[0]),
            "fox: (0.96 = Weight 1.00 * IDF 0.55 * (F 3.00 * (k1 1.2 + 1)) / (F 3.00 + k1 1.2 * (1 - b 0.75 + b 0.75 * Doc Len 21 / Average Doc Len 37.92)))"
        );
    }

    /// The older `BM25`, which leaves the average length where the document's
    /// own belongs and divides the whole thing by the slop at the end.
    #[test]
    fn a_term_under_the_older_bm25_divides_by_the_slop() {
        let facts = Facts::new(12, 455);
        let found = Found::Term(Term::new(3, 1.0, 7).about(b"fox"));
        let note = Why::new(Scorer::Old, facts).note(&doc(1.0, 21, 3), &found, 48);
        assert_eq!(
            head(&note),
            "Final BM25 : words BM25 0.11 * document score 1.00 / slop 48"
        );
        assert_eq!(
            head(&under(&note)[0]),
            "(0.11 = Weight 1.00 * IDF 1.00 * F 3 / (F 3 + k1 1.2 * (1 - b 0.5 + b 0.5 * Average Len 37.92)))"
        );
    }

    /// A union of two words, which prints its own weight on the branch and one
    /// on each leaf under it.
    #[test]
    fn a_branch_carries_the_weight_and_the_leaves_under_it_do_not() {
        let facts = Facts::new(12, 455);
        let found = Found::any(
            2.0,
            vec![
                Found::Term(Term::new(3, 1.0, 7).about(b"fox")),
                Found::Term(Term::new(2, 1.0, 2).about(b"dog")),
            ],
        );
        let note = Why::new(Scorer::Bm25, facts).note(&doc(1.0, 21, 3), &found, 1);
        assert_eq!(
            head(&note),
            "Final BM25 : words BM25 7.10 * document score 1.00"
        );
        assert_eq!(head(&under(&note)[0]), "(Weight 2.00 * children BM25 3.55)");
        assert_eq!(under(&under(&note)[0]).len(), 2);
    }

    /// `DISMAX` takes the best branch of a union rather than the sum, and says
    /// so on the line, and has no top layer over it at all.
    #[test]
    fn dismax_prints_the_best_branch_and_wraps_nothing() {
        let facts = Facts::new(12, 455);
        let found = Found::any(
            1.0,
            vec![
                Found::Term(Term::new(3, 1.0, 7).about(b"fox")),
                Found::Term(Term::new(2, 1.0, 2).about(b"dog")),
            ],
        );
        let note = Why::new(Scorer::DisMax, facts).note(&doc(1.0, 21, 3), &found, 1);
        assert_eq!(head(&note), "3.00 = Weight 1.00 * children DISMAX 3.00");
        assert_eq!(
            head(&under(&note)[0]),
            "DISMAX 3.00 = Weight 1.00 * Frequency 3"
        );
    }

    /// The three matches with no word in them, which the three that count words
    /// call irrelevant and the rest weigh and count.
    #[test]
    fn a_match_with_no_word_in_it_says_so_six_different_ways() {
        let facts = Facts::new(12, 455);
        let d = doc(1.0, 21, 3);
        let lines = |found: &Found<'_>| {
            [Scorer::TfIdf, Scorer::Old, Scorer::Bm25, Scorer::DisMax]
                .map(|scorer| head(&Why::new(scorer, facts).note(&d, found, 1)).to_string())
        };
        // A negation that found nothing, which counts no times.
        let none = lines(&Found::missing());
        assert!(none[0].ends_with("/ norm 3 / slop 1"));
        assert_eq!(
            head(&under(&Why::new(Scorer::Old, facts).note(&d, &Found::missing(), 1))[0]),
            "Frequency 0 -> value 0"
        );
        assert_eq!(
            head(&under(&Why::new(Scorer::Bm25, facts).note(&d, &Found::missing(), 1))[0]),
            "Irrelevant token -> score is 0"
        );
        assert_eq!(none[3], "DISMAX 0.00 = Weight 1.00 * Frequency 0");
        // An optional branch the document did not answer, which weighs nothing.
        assert_eq!(
            head(&under(&Why::new(Scorer::TfIdf, facts).note(&d, &Found::skipped(), 1))[0]),
            "(TFIDF 0.00 = Weight 0.00 * Frequency 1)"
        );
        // A filter, which weighs one and counts one.
        assert_eq!(
            head(&under(&Why::new(Scorer::TfIdf, facts).note(&d, &Found::filter(), 1))[0]),
            "(TFIDF 1.00 = Weight 1.00 * Frequency 1)"
        );
    }

    /// A wildcard, which the newer family names and the older ones do not.
    #[test]
    fn a_wildcard_is_named_by_one_family_and_not_by_the_others() {
        let facts = Facts::new(12, 455);
        let d = doc(1.0, 1, 1);
        assert_eq!(
            head(&under(&Why::new(Scorer::Bm25, facts).note(&d, &Found::Every, 1))[0]),
            "*: (1.66 = Weight 1.00 * IDF 1.00 * (F 1.00 * (k1 1.2 + 1)) / (F 1.00 + k1 1.2 * (1 - b 0.75 + b 0.75 * Doc Len 1 / Average Doc Len 37.92)))"
        );
        assert_eq!(
            head(&under(&Why::new(Scorer::Old, facts).note(&d, &Found::Every, 1))[0]),
            "(0.04 = Weight 1.00 * F 1 / (F 1 + k1 1.2 * (1 - b 0.5 + b 0.5 * Average Len 37.92)))"
        );
        assert_eq!(
            head(&under(&Why::new(Scorer::TfIdf, facts).note(&d, &Found::Every, 1))[0]),
            "(TFIDF 1.00 = Weight 1.00 * Frequency 1)"
        );
    }

    /// The tangent wraps the whole of the ordinary top and the normalisation
    /// replaces it, which is one level of difference between the two.
    #[test]
    fn the_tangent_wraps_a_layer_the_normalisation_replaces() {
        let facts = Facts::new(12, 455);
        let d = doc(1.0, 21, 3);
        let found = Found::Term(Term::new(3, 1.0, 7).about(b"fox"));
        let tanh = Why::new(Scorer::Tanh, facts).note(&d, &found, 1);
        assert_eq!(
            head(&tanh),
            "Final Normalized BM25 : tanh(stretch factor 1/4 * Final BM25 0.96)"
        );
        assert_eq!(
            head(&under(&tanh)[0]),
            "Final BM25 : words BM25 0.96 * document score 1.00"
        );
        assert!(head(&under(&under(&tanh)[0])[0]).starts_with("fox: "));

        let norm = Why::new(Scorer::Norm, facts).over(0.96).note(&d, &found, 1);
        assert_eq!(
            head(&norm),
            "Final BM25STD.NORM: 1.00 = Original Score: 0.96 / Max Score: 0.96"
        );
        assert!(head(&under(&norm)[0]).starts_with("fox: "));
    }

    /// The two that never look at what matched, and both of the payload lines.
    #[test]
    fn the_two_that_never_recurse_answer_one_line() {
        let facts = Facts::new(12, 455);
        let found = Found::Term(Term::new(3, 1.0, 7).about(b"fox"));
        let mut d = doc(2.0, 21, 3);
        assert_eq!(
            Why::new(Scorer::Worth, facts).note(&d, &found, 1),
            Note::Line("Document's score is 2.00".to_string())
        );
        assert_eq!(
            Why::new(Scorer::Hamming, facts).note(&d, &found, 1),
            Note::Line("Payloads provided to scorer vary in length".to_string())
        );
        d.payload = Some(b"\x00\x01".as_slice().into());
        let close = Why::new(Scorer::Hamming, facts).about(Some(b"\x00\x00"));
        assert_eq!(
            close.note(&d, &found, 1),
            Note::Line(
                "String length is 2. Bit count is 1. Result is (1 / count + 1) = 0.50".to_string()
            )
        );
    }
}
