//! Cutting a field down to the parts a query matched, and marking them.
//!
//! This is what `SUMMARIZE` and `HIGHLIGHT` do to the values on their way back
//! out. Neither of them changes what answered or in what order, so none of it
//! runs under the registry lock: the walk hands back a set of keys, the keys are
//! read, and then every value that is about to go on the wire comes through
//! here.
//!
//! ```
//! use yo_search::english::English;
//! use yo_search::summary::{Trim, Wanted, Wrap};
//!
//! let mut english = English::new();
//! let wanted = Wanted::words([&b"fox"[..]]);
//! let text = b"the quick brown fox jumps over the lazy dog";
//! assert_eq!(
//!     yo_search::summary::highlight(text, &wanted, None, &Wrap::default(), &mut english),
//!     b"the quick brown <b>fox</b> jumps over the lazy dog".to_vec()
//! );
//! let trim = Trim { frags: 1, len: 4, ..Trim::default() };
//! assert_eq!(
//!     yo_search::summary::summarize(text, &wanted, None, &trim, None, &mut english),
//!     b"brown fox jumps over... ".to_vec()
//! );
//! ```
//!
//! # The two rules
//!
//! A field the query matched comes back as fragments, one per match, each one
//! the match with as much of the text either side of it as the budget allows. A
//! field it did not comes back as the front of the field, cut at a word.
//!
//! Both budgets are counted in bytes and both are six bytes per unit of `LEN`,
//! six being the width a word is guessed to be. What is spent against them is
//! real bytes, so a field of short words keeps more of them than a field of long
//! ones does under the same `LEN`, which is the whole reason the guess is only
//! used for the budget.
//!
//! A fragment that runs over more than one match pays a unit of `LEN` for every
//! word standing between the outermost two, and the two sides then split what is
//! left. A stop word standing between them is free, so a sentence of English
//! keeps far more of itself than a run of made up words the same length does.
//! That is measured and not guessed: ten `the` between two matches leave the
//! fragment exactly as wide as no words at all would.
//!
//! `FRAGS` is applied before any of that context is worked out, so a fragment
//! runs into the fragment next to it and not into the next match. Drop a
//! fragment with `FRAGS 1` and the one that is left can run straight over the
//! match the dropped one was built around.
//!
//! # The corner a real server has
//!
//! A field with no match in it is left whole rather than cut down when a
//! `HIGHLIGHT` covers the same field, and that is measured rather than reasoned
//! about. `SUMMARIZE FIELDS 1 c LEN 2 HIGHLIGHT FIELDS 1 a` cuts `c` down and
//! the same query with `HIGHLIGHT FIELDS 1 c` does not, on a field that holds
//! nothing either clause could mark. It is odd and it is what a real server
//! does, so it is what this does.
//!
//! # What is not here
//!
//! A tag is never marked. `@g:{red}` leaves the word red alone in the tag field
//! and in every text field beside it, so the terms a tag stood for never reach
//! the marker.
//!
//! The word a match was found under is worked out again from the value rather
//! than carried out of the walk, which is why a field the schema does not index
//! is never fragmented: there is a term list to test against but no reason to
//! think the field was read with it.

use crate::english::English;
use crate::expand;
use crate::query::{Node, What};
use crate::text;
use crate::token::fold_stored;
use crate::words::{self, splits};

/// How wide a word is guessed to be when a budget is worked out.
///
/// Nothing measures the real words. `LEN 20` is twenty guessed words and comes
/// to a hundred and twenty bytes, and how many words that turns out to be
/// depends on the field.
const GUESS: usize = 6;

/// How many words may stand between two matches before they stop sharing a
/// fragment. Seven of them and it is one fragment, eight and it is two. Stop
/// words do not count, so ten of those between two matches still leave one.
const JOINED: usize = 7;

/// How a `SUMMARIZE` was asked for.
#[derive(Debug, Clone)]
pub struct Trim {
    /// How many fragments at most, which is the first that many and not the
    /// best that many.
    pub frags: usize,
    /// How much context, in guessed words, shared between the two sides and
    /// paid down first by the words a fragment had to swallow.
    pub len: usize,
    /// What goes after every fragment, the last one included.
    pub separator: Box<[u8]>,
}

impl Default for Trim {
    fn default() -> Trim {
        Trim {
            frags: 3,
            len: 20,
            separator: Box::from(&b"... "[..]),
        }
    }
}

/// How a `HIGHLIGHT` was asked for.
#[derive(Debug, Clone)]
pub struct Wrap {
    /// What goes in front of a match.
    pub open: Box<[u8]>,
    /// What goes after it.
    pub close: Box<[u8]>,
}

impl Default for Wrap {
    fn default() -> Wrap {
        Wrap {
            open: Box::from(&b"<b>"[..]),
            close: Box::from(&b"</b>"[..]),
        }
    }
}

/// One thing a query was looking for, in the shape a single word can be tested
/// against.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Looking {
    Word(Box<[u8]>),
    Prefix(Box<[u8]>),
    Suffix(Box<[u8]>),
    Infix(Box<[u8]>),
    Pattern(Box<[u8]>),
    Fuzzy(Box<[u8]>, u8),
}

impl Looking {
    /// Whether a folded word is one of the words this stands for.
    fn hit(&self, word: &[u8]) -> bool {
        match self {
            Looking::Word(term) => **term == *word,
            Looking::Prefix(part) => word.starts_with(part),
            Looking::Suffix(part) => word.ends_with(part),
            Looking::Infix(part) => {
                part.is_empty() || word.windows(part.len()).any(|run| run == &**part)
            }
            Looking::Pattern(pattern) => expand::glob(pattern, word),
            Looking::Fuzzy(term, distance) => expand::within(word, term, *distance),
        }
    }
}

/// Everything a query is looking for, flattened out of the tree it parsed into.
///
/// The tree is gone by the time a value is written, and the shape of it would
/// not help anyway: a match is marked wherever it stands, so a term the query
/// scoped to one field is marked in every field, which is what a real server
/// does and is the one place the field mask is not honoured.
///
/// A `NOT` is left out. Nothing under it is in any document that answered, so
/// there would be nothing to mark even if it were kept.
#[derive(Debug, Clone, Default)]
pub struct Wanted {
    terms: Vec<Looking>,
}

impl Wanted {
    /// What this query is looking for.
    #[must_use]
    pub fn of(node: &Node) -> Wanted {
        let mut wanted = Wanted::default();
        wanted.read(node);
        wanted.terms.dedup();
        wanted
    }

    /// A list of plain words, which is what a test wants and what a caller with
    /// no tree in its hand has.
    #[must_use]
    pub fn words<'a>(words: impl IntoIterator<Item = &'a [u8]>) -> Wanted {
        Wanted {
            terms: words
                .into_iter()
                .map(|word| Looking::Word(word.into()))
                .collect(),
        }
    }

    /// Whether there is anything to look for at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    fn read(&mut self, node: &Node) {
        match &node.what {
            What::Term(word) => self.terms.push(Looking::Word(word.word.clone())),
            What::Prefix(part) => self.terms.push(Looking::Prefix(part.clone())),
            What::Suffix(part) => self.terms.push(Looking::Suffix(part.clone())),
            What::Infix(part) => self.terms.push(Looking::Infix(part.clone())),
            What::Pattern(pattern) => self.terms.push(Looking::Pattern(pattern.clone())),
            What::Fuzzy(word, distance) => {
                self.terms.push(Looking::Fuzzy(word.clone(), *distance));
            }
            What::Union(nodes) | What::Intersect(nodes) | What::Exact(nodes) => {
                for node in nodes {
                    self.read(node);
                }
            }
            What::Optional(node) => self.read(node),
            // Everything else marks nothing: a `NOT` because no document that
            // answered holds it, a number, a point and a vector because none of
            // them is a word in the text, and a `TAG` because a real server
            // does not mark one. `@g:{red}` leaves the word red alone wherever
            // it stands, in the tag field and in every text field beside it.
            _ => {}
        }
    }

    /// Whether a word of a document is one of these, either as it stands or
    /// through the stem it shares with one of them.
    fn hit(&self, word: &[u8], english: &mut English) -> bool {
        if self.terms.iter().any(|term| term.hit(word)) {
            return true;
        }
        match words::stem(english, word) {
            Some(stem) => self.terms.iter().any(|term| term.hit(&stem)),
            None => false,
        }
    }
}

/// One word of a value, as the summarizer needs it: where it stands, whether the
/// query matched it and whether it is a stop word.
///
/// Not [`crate::words::Words`], which is the indexing walk and drops the stop
/// words and the words too long to keep. Nothing is dropped here, because a
/// fragment is a run of the value and a word missing out of the middle of it
/// would make the arithmetic describe a different piece of text than the one
/// that goes on the wire. A stop word is carried with a mark on it instead,
/// because it is free in the two places a word is counted and is charged for its
/// bytes everywhere a word is measured.
struct Word {
    at: usize,
    end: usize,
    hit: bool,
    stop: bool,
}

/// Every word of a value, marked with whether the query matched it.
fn walk(
    text: &[u8],
    wanted: &Wanted,
    stops: Option<&[Box<[u8]>]>,
    english: &mut English,
) -> Vec<Word> {
    let mut words = Vec::new();
    let mut at = 0;
    while at < text.len() {
        if splits(text[at]) {
            at += 1;
            continue;
        }
        let start = at;
        while at < text.len() && !splits(text[at]) {
            at += 1;
        }
        let folded = fold_stored(&text[start..], at - start);
        words.push(Word {
            at: start,
            end: at,
            hit: !folded.is_empty() && wanted.hit(&folded, english),
            stop: text::dropped(stops, &folded),
        });
    }
    words
}

/// The front of a field the query did not match.
///
/// Whole words while the end of the word is inside the budget, never the last
/// word of the field, cut at the start of the first word left out and then
/// stripped of the trailing space. A field with no word in it at all comes back
/// whole, and one where the budget took nothing comes back as its first byte,
/// which is a corner a real server has and not one worth explaining.
fn leading(text: &[u8], words: &[Word], trim: &Trim) -> Vec<u8> {
    let Some((last, takeable)) = words.split_last() else {
        return text.to_vec();
    };
    let budget = GUESS * trim.len * trim.frags + 5;
    let taken = takeable
        .iter()
        .take_while(|word| word.end <= budget)
        .count();
    let stop = match takeable.get(taken) {
        Some(word) => word.at,
        None => last.at,
    };
    let mut stop = stop.max(1).min(text.len());
    while stop > 0 && text[stop - 1].is_ascii_whitespace() {
        stop -= 1;
    }
    text[..stop].to_vec()
}

/// Which runs of words share a fragment, as a range of word numbers each.
///
/// A match starts a fragment and every match close enough behind it joins the
/// same one, so a field where every other word matched comes back as one long
/// fragment rather than as a dozen overlapping ones.
fn groups(words: &[Word]) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for (n, _) in words.iter().enumerate().filter(|(_, word)| word.hit) {
        let apart = |from: usize| words[from + 1..n].iter().filter(|word| !word.stop).count();
        match groups.last_mut() {
            Some(group) if apart(group.1) <= JOINED => group.1 = n,
            _ => groups.push((n, n)),
        }
    }
    groups
}

/// Where a fragment starts and stops in the value.
///
/// The budget is worked out once for the whole fragment and then spent from
/// each end of it, so a fragment that had to swallow words to join two matches
/// is charged for them on both sides. A stop word it swallowed is free, which is
/// why an English sentence keeps more of itself than a run of made up words of
/// the same length does.
///
/// What stops the context is the fragment next to it and not the next match,
/// and the two are only the same thing while every fragment is kept. `FRAGS 1`
/// over a field with two matches in it hands the one fragment it keeps the run
/// of text the second match sits in, second match and all, because there is no
/// second fragment left to run into.
fn spread(
    text: &[u8],
    words: &[Word],
    kept: &[(usize, usize)],
    which: usize,
    trim: &Trim,
) -> (usize, usize) {
    let (first, last) = kept[which];
    // Never the first word of the field and never the last, and never past the
    // fragment on either side.
    let lo = match which {
        0 => 1,
        _ => (kept[which - 1].1 + 1).max(1),
    };
    let hi = match kept.get(which + 1) {
        Some(next) => next.0.min(words.len().saturating_sub(1)),
        None => words.len().saturating_sub(1),
    };
    let between = words[first..=last]
        .iter()
        .filter(|word| !word.hit && !word.stop)
        .count();
    let left = trim.len.saturating_sub(between);
    let budget = GUESS * (left / 2);

    // The distance is measured to the front of the match rather than to the
    // word taken last, so the walk stops on the whole run and not on one step
    // of it.
    let mut at = words[first].at;
    for word in words[lo.min(first)..first].iter().rev() {
        if words[first].at - word.at >= budget {
            break;
        }
        at = word.at;
    }
    // The other end is the same walk with one difference, which is that a run
    // landing exactly on the budget is kept here and dropped there. Measured
    // both ways round, and it is the only thing that makes the two sides of a
    // fragment come out different widths.
    let mut end = words[last].end;
    for word in &words[(last + 1).min(hi)..hi] {
        if word.end - words[last].end > budget {
            break;
        }
        end = word.end;
    }
    // Nothing was taken on that side, so the byte that follows the match comes
    // along instead, which is normally the space it was written with.
    if end == words[last].end && left > 0 && end < text.len() {
        end += 1;
    }
    (at, end)
}

/// Marks every match inside a run of a value.
fn marked(
    text: &[u8],
    words: &[Word],
    run: (usize, usize),
    wrap: Option<&Wrap>,
    out: &mut Vec<u8>,
) {
    let (from, to) = run;
    let Some(wrap) = wrap else {
        out.extend_from_slice(&text[from..to]);
        return;
    };
    let mut at = from;
    for word in words
        .iter()
        .filter(|word| word.hit && word.at >= from && word.end <= to)
    {
        out.extend_from_slice(&text[at..word.at]);
        out.extend_from_slice(&wrap.open);
        out.extend_from_slice(&text[word.at..word.end]);
        out.extend_from_slice(&wrap.close);
        at = word.end;
    }
    out.extend_from_slice(&text[at..to]);
}

/// A value with every match in it marked and nothing taken away.
#[must_use]
pub fn highlight(
    text: &[u8],
    wanted: &Wanted,
    stops: Option<&[Box<[u8]>]>,
    wrap: &Wrap,
    english: &mut English,
) -> Vec<u8> {
    let words = walk(text, wanted, stops, english);
    let mut out = Vec::with_capacity(text.len());
    marked(text, &words, (0, text.len()), Some(wrap), &mut out);
    out
}

/// A value cut down to the parts the query matched, with those parts marked
/// when a `HIGHLIGHT` asked for it as well.
#[must_use]
pub fn summarize(
    text: &[u8],
    wanted: &Wanted,
    stops: Option<&[Box<[u8]>]>,
    trim: &Trim,
    wrap: Option<&Wrap>,
    english: &mut English,
) -> Vec<u8> {
    let words = walk(text, wanted, stops, english);
    let groups = groups(&words);
    if groups.is_empty() {
        // Nothing matched, so there is no fragment to build and no separator to
        // write after one. A `HIGHLIGHT` on the same field calls even the
        // cutting off, which is measured and is the one thing about these two
        // clauses that reads like an oversight rather than a rule.
        return match wrap {
            Some(_) => text.to_vec(),
            None => leading(text, &words, trim),
        };
    }
    let kept: Vec<(usize, usize)> = groups.into_iter().take(trim.frags).collect();
    let mut out = Vec::with_capacity(text.len());
    for which in 0..kept.len() {
        let run = spread(text, &words, &kept, which, trim);
        marked(text, &words, run, wrap, &mut out);
        out.extend_from_slice(&trim.separator);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summed(text: &[u8], words: &[&[u8]], frags: usize, len: usize) -> String {
        let trim = Trim {
            frags,
            len,
            separator: Box::from(&b"|"[..]),
        };
        let wanted = Wanted::words(words.iter().copied());
        let mut english = English::new();
        String::from_utf8(summarize(text, &wanted, None, &trim, None, &mut english)).unwrap()
    }

    #[test]
    fn a_field_with_no_match_keeps_its_front_and_loses_its_last_word() {
        assert_eq!(summed(b"aa bb cc", &[b"zz"], 3, 20), "aa bb");
        assert_eq!(summed(b"alpha beta", &[b"zz"], 3, 20), "alpha");
        assert_eq!(summed(b"aa,bb", &[b"zz"], 3, 20), "aa,");
        // Nothing to cut at, so it comes back as it stands.
        assert_eq!(summed(b"...", &[b"zz"], 3, 20), "...");
        assert_eq!(summed(b"", &[b"zz"], 3, 20), "");
        // One word is the last word, so nothing is taken and the first byte is
        // the answer.
        assert_eq!(summed(b"red", &[b"zz"], 3, 20), "r");
        assert_eq!(summed(b"q", &[b"zz"], 3, 20), "q");
    }

    #[test]
    fn a_no_match_budget_is_six_bytes_a_word_of_len_times_frags() {
        let text = [&b"zzz"[..]; 40].join(&b' ');
        // Three byte words, so a word and its space is four bytes and the
        // budget of eleven takes three of them.
        assert_eq!(summed(&text, &[b"q"], 1, 1).len(), 11);
        assert_eq!(summed(&text, &[b"q"], 1, 2).len(), 15);
        assert_eq!(summed(&text, &[b"q"], 3, 1).len(), 23);
        let wide = [&b"zzzzzz"[..]; 40].join(&b' ');
        // Six byte words, so the same budget takes one fewer.
        assert_eq!(summed(&wide, &[b"q"], 1, 1).len(), 6);
        assert_eq!(summed(&wide, &[b"q"], 1, 7).len(), 41);
    }

    #[test]
    fn a_fragment_is_the_match_and_what_the_budget_buys_either_side() {
        let text = b"the quick brown fox jumps over the lazy dog";
        assert_eq!(summed(text, &[b"fox"], 1, 4), "brown fox jumps over|");
        assert_eq!(summed(text, &[b"fox"], 1, 5), "brown fox jumps over|");
        assert_eq!(
            summed(text, &[b"fox"], 1, 20),
            "quick brown fox jumps over the lazy|"
        );
        // No context at all, and the byte after the match all the same.
        assert_eq!(summed(text, &[b"fox"], 1, 1), "fox |");
        assert_eq!(summed(text, &[b"fox"], 1, 0), "fox|");
    }

    #[test]
    fn a_fragment_never_takes_the_first_or_the_last_word_of_a_field() {
        assert_eq!(
            summed(b"one two three fox four five six", &[b"fox"], 1, 20),
            "two three fox four five|"
        );
        assert_eq!(summed(b"fox tail", &[b"fox"], 1, 4), "fox |");
        assert_eq!(summed(b"head fox", &[b"fox"], 1, 4), "fox|");
        assert_eq!(summed(b"fox tail here", &[b"fox"], 1, 4), "fox tail|");
    }

    #[test]
    fn two_matches_near_each_other_share_one_fragment() {
        let near = [&b"zzzz"[..]; 12].join(&b' ');
        let near = str::from_utf8(&near).unwrap();
        let seven = format!("{near} fox n1 n2 n3 n4 n5 n6 n7 fox {near}");
        assert_eq!(
            summed(seven.as_bytes(), &[b"fox"], 5, 20)
                .matches('|')
                .count(),
            1
        );
        let eight = format!("{near} fox n1 n2 n3 n4 n5 n6 n7 n8 fox {near}");
        assert_eq!(
            summed(eight.as_bytes(), &[b"fox"], 5, 20)
                .matches('|')
                .count(),
            2
        );
    }

    /// Ten stop words between two matches join them as surely as none at all
    /// would, and cost the fragment nothing either. Measured against an 8.10.1
    /// over gaps of nought to ten of each kind at every `LEN` up to fourteen,
    /// where a gap of stop words came back the same width every time.
    #[test]
    fn a_stop_word_between_two_matches_is_free() {
        let pad = |gap: &str| {
            format!("aa1 aa2 aa3 aa4 aa5 aa6 fox {gap} fox bb1 bb2 bb3 bb4 bb5 bb6")
                .trim()
                .replace("  ", " ")
        };
        let stops = pad("the the the the the the the the the the");
        let none = pad("");
        // Ten stop words between the matches leave the fragment as wide as no
        // words at all would, at every width there is context to take.
        for len in 0..15 {
            let with = summed(stops.as_bytes(), &[b"fox"], 5, len);
            let without = summed(none.as_bytes(), &[b"fox"], 5, len);
            let with = with.replace("the ", "");
            assert_eq!(with, without, "at LEN {len}");
        }
        // The same gap in words the index would have kept costs a unit of `LEN`
        // each and the fragment gets nothing until they are paid off.
        let words = pad("n1 n2 n3 n4 n5 n6 n7");
        assert_eq!(
            summed(words.as_bytes(), &[b"fox"], 5, 8),
            "fox n1 n2 n3 n4 n5 n6 n7 fox |"
        );
        assert_eq!(
            summed(words.as_bytes(), &[b"fox"], 5, 9),
            "aa6 fox n1 n2 n3 n4 n5 n6 n7 fox bb1|"
        );
    }

    #[test]
    fn a_fragment_is_charged_for_the_words_it_swallowed() {
        let pad = [&b"zzzz"[..]; 12].join(&b' ');
        let pad = str::from_utf8(&pad).unwrap();
        // No gap, so the budget is whole and the left takes eleven of its
        // twelve words, the first one of the field being out of reach.
        let none = format!("{pad} fox fox {pad}");
        let got = summed(none.as_bytes(), &[b"fox"], 5, 20);
        assert_eq!(
            got.split(" fox").next().unwrap().matches("zzzz").count(),
            11
        );
        // One word between them costs both sides a word.
        let one = format!("{pad} fox zzzz fox {pad}");
        let got = summed(one.as_bytes(), &[b"fox"], 5, 20);
        assert_eq!(
            got.split(" fox").next().unwrap().matches("zzzz").count(),
            10
        );
    }

    #[test]
    fn frags_keeps_the_first_of_them_and_is_not_an_error_when_it_asks_for_more() {
        // Ten words apart, which is more than a fragment will reach over, so
        // there are five of them to choose from.
        let far = [&b"zzzz"[..]; 10].join(&b' ');
        let far = str::from_utf8(&far).unwrap();
        let text = [far; 6].join(" fox ");
        assert_eq!(
            summed(text.as_bytes(), &[b"fox"], 2, 0)
                .matches('|')
                .count(),
            2
        );
        assert_eq!(
            summed(text.as_bytes(), &[b"fox"], 9, 0)
                .matches('|')
                .count(),
            5
        );
    }

    #[test]
    fn a_highlight_marks_every_match_and_honours_the_stemmer() {
        let mut english = English::new();
        let wanted = Wanted::words([&b"run"[..]]);
        assert_eq!(
            highlight(
                b"running runs runner",
                &wanted,
                None,
                &Wrap::default(),
                &mut english
            ),
            b"<b>running</b> <b>runs</b> runner".to_vec()
        );
    }

    #[test]
    fn a_highlight_inside_a_summary_marks_the_fragment() {
        let trim = Trim {
            frags: 1,
            len: 2,
            separator: Box::from(&b"|"[..]),
        };
        let wanted = Wanted::words([&b"fox"[..]]);
        let mut english = English::new();
        let got = summarize(
            b"the quick brown fox jumps over the lazy dog",
            &wanted,
            None,
            &trim,
            Some(&Wrap::default()),
            &mut english,
        );
        assert_eq!(got, b"<b>fox</b> jumps|".to_vec());
    }

    #[test]
    fn a_dropped_fragment_stops_bounding_the_one_that_was_kept() {
        // Nine words between the two matches, which is two words more than one
        // fragment reaches over, so there are two of them.
        let text = b"c1 c2 c3 fox d1 d2 d3 d4 d5 d6 d7 d8 d9 fox e1 e2 e3";
        assert_eq!(
            summed(text, &[b"fox"], 3, 20),
            "c2 c3 fox d1 d2 d3 d4 d5 d6 d7 d8 d9|d1 d2 d3 d4 d5 d6 d7 d8 d9 fox e1 e2|",
            "kept either side, so each one stops on the other"
        );
        assert_eq!(
            summed(text, &[b"fox"], 1, 20),
            "c2 c3 fox d1 d2 d3 d4 d5 d6 d7 d8 d9 fox e1 e2|",
            "the second fragment was dropped, so the first runs over its match"
        );
    }

    #[test]
    fn a_field_with_no_match_is_left_whole_when_it_is_being_marked_too() {
        let trim = Trim {
            frags: 3,
            len: 2,
            separator: Box::from(&b"|"[..]),
        };
        let wanted = Wanted::words([&b"zz"[..]]);
        let mut english = English::new();
        let text = b"aaa t1 t2 t3 t4 t5 t6 t7 t8";
        let cut = summarize(text, &wanted, None, &trim, None, &mut english);
        assert_eq!(cut, b"aaa t1 t2 t3 t4 t5 t6 t7".to_vec());
        let whole = summarize(
            text,
            &wanted,
            None,
            &trim,
            Some(&Wrap::default()),
            &mut english,
        );
        assert_eq!(whole, text.to_vec());
    }

    #[test]
    fn a_tag_is_never_marked() {
        let tag = Node::new(What::Tag(Box::from(&b"g"[..]), vec![Node::term(b"red")]));
        let mut english = English::new();
        assert!(Wanted::of(&tag).is_empty());
        assert_eq!(
            highlight(
                b"red and blue",
                &Wanted::of(&tag),
                None,
                &Wrap::default(),
                &mut english
            ),
            b"red and blue".to_vec()
        );
        // The text term beside it is still marked, and only that one.
        let both = Node::new(What::Intersect(vec![Node::term(b"blue"), tag]));
        assert_eq!(
            highlight(
                b"red and blue",
                &Wanted::of(&both),
                None,
                &Wrap::default(),
                &mut english
            ),
            b"red and <b>blue</b>".to_vec()
        );
    }
}
