//! What one thing a client wrote stands for in the dictionary, when it stands
//! for more than one term.
//!
//! `hello` is a term. `hel*` is not: it is every term the index holds that
//! starts `hel`, and answering it means answering all of them. There are five
//! of these, a prefix, a suffix, an infix, a pattern and a fuzzy word, and all
//! five come down to a walk over the dictionary picking out the terms that fit,
//! so they are here in one place rather than five times over in the walk.
//!
//! ```
//! use yo_search::expand::{glob, within};
//!
//! assert!(glob(b"ab*", b"abc"));
//! assert!(glob(b"a?c", b"abc"));
//! assert!(!glob(b"a?c", b"abcd"));
//! assert!(within(b"dog", b"dogs", 1));
//! assert!(!within(b"dog", b"dogs", 0));
//! ```
//!
//! # Two limits, and both of them are a real server's
//!
//! A prefix, a suffix and an infix shorter than [`SHORTEST`] stand for nothing
//! at all. `ab*` answers and `a*` answers nothing, without an error and without
//! a warning, which is measured rather than read off a manual. A pattern is not
//! held to it, so `w'a?c'` answers where `a*` does not, and neither is a fuzzy
//! word.
//!
//! No more than [`MOST`] terms come out of one of these, which is what a real
//! server does with the same default and is the reason a prefix of one letter is
//! refused rather than being allowed to stand for the whole dictionary.
//!
//! # The order matters, because only the first one counts
//!
//! A document holding two terms a prefix stands for is scored on one of them
//! and not on both, and the one it is scored on is the first in byte order. An
//! index holding `aba` and `abc` answers `ab*` for a document holding both with
//! the score of `aba` alone, and swapping which of the two is the rarer word
//! does not change that. So these all answer in the dictionary's own order and
//! the walk above them leans on it.
//!
//! # Stems are terms like any other
//!
//! A stem sits in the same dictionary behind its [`crate::posts::STEM`] byte and
//! nothing here skips it. It cannot come out of a prefix, since no folded word
//! starts with that byte, and it does come out of a suffix: an index holding
//! `dogs` answers `*og` for the document that has it, which it can only do
//! through the stem, and that is measured too.

use crate::posts::{Posts, Terms};

/// How short a prefix, a suffix or an infix may be before it stands for
/// nothing.
///
/// A real server calls this `MINPREFIX`, gives it this value and applies it to
/// all three, so `*b*` answers nothing where `*bd*` answers.
pub const SHORTEST: usize = 2;

/// How many terms one of these may stand for.
///
/// A real server calls this `MAXEXPANSIONS` and gives it this value. Which ones
/// are kept when there are more is the first this many in byte order, which is
/// the order the dictionary is walked in.
pub const MOST: usize = 200;

/// Every term, which is what a walk with nothing to narrow it starts from.
const ANY: &[u8] = b"";

/// Every term starting with what was written.
#[must_use]
pub fn under<'a>(terms: &'a Terms, prefix: &'a [u8]) -> Vec<(&'a [u8], &'a Posts)> {
    if prefix.len() < SHORTEST {
        return Vec::new();
    }
    terms.under(prefix).take(MOST).collect()
}

/// Every term ending with what was written.
#[must_use]
pub fn ending<'a>(terms: &'a Terms, suffix: &[u8]) -> Vec<(&'a [u8], &'a Posts)> {
    if suffix.len() < SHORTEST {
        return Vec::new();
    }
    terms
        .under(ANY)
        .filter(|(term, _)| term.ends_with(suffix))
        .take(MOST)
        .collect()
}

/// Every term with what was written somewhere inside it.
#[must_use]
pub fn inside<'a>(terms: &'a Terms, part: &[u8]) -> Vec<(&'a [u8], &'a Posts)> {
    if part.len() < SHORTEST {
        return Vec::new();
    }
    terms
        .under(ANY)
        .filter(|(term, _)| holds(term, part))
        .take(MOST)
        .collect()
}

/// Every term a pattern matches.
///
/// The pattern is folded first, because a real server answers `w'AB*'` with the
/// documents holding `abc` while printing the pattern back in the case it
/// arrived in.
#[must_use]
pub fn like<'a>(terms: &'a Terms, pattern: &[u8]) -> Vec<(&'a [u8], &'a Posts)> {
    let pattern = crate::token::fold(pattern);
    terms
        .under(ANY)
        .filter(|(term, _)| glob(&pattern, term))
        .take(MOST)
        .collect()
}

/// Every term within an edit distance of what was written.
#[must_use]
pub fn near<'a>(terms: &'a Terms, word: &[u8], distance: u8) -> Vec<(&'a [u8], &'a Posts)> {
    terms
        .under(ANY)
        .filter(|(term, _)| within(word, term, distance))
        .take(MOST)
        .collect()
}

/// Whether a run of bytes is anywhere inside another.
fn holds(term: &[u8], part: &[u8]) -> bool {
    term.windows(part.len()).any(|window| window == part)
}

/// Whether a pattern matches a term, where `*` is any run and `?` is one byte.
///
/// The backtracking is the one every glob uses: remember where the last `*` was
/// and how far it had eaten, and when the rest stops fitting, let it eat one
/// more. Nothing here treats a backslash as an escape, because the pattern
/// arrives as the client wrote it and a real server does not either.
#[must_use]
pub fn glob(pattern: &[u8], word: &[u8]) -> bool {
    let (mut p, mut w) = (0, 0);
    // Where the last `*` is and how much of the word it has taken, for going
    // back to when what follows it does not fit.
    let mut star: Option<(usize, usize)> = None;
    while w < word.len() {
        match pattern.get(p) {
            Some(b'*') => {
                star = Some((p, w));
                p += 1;
            }
            Some(b'?') => {
                p += 1;
                w += 1;
            }
            Some(b) if *b == word[w] => {
                p += 1;
                w += 1;
            }
            _ => match star {
                Some((at, eaten)) => {
                    p = at + 1;
                    w = eaten + 1;
                    star = Some((at, eaten + 1));
                }
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|b| *b == b'*')
}

/// Whether two words are within an edit distance of each other.
///
/// A row at a time with the two rows kept and the rest thrown away, which is
/// all a yes or no answer needs, and it gives up as soon as a whole row is
/// further out than the distance asked for, because nothing below such a row
/// can come back.
#[must_use]
pub fn within(word: &[u8], term: &[u8], distance: u8) -> bool {
    let most = usize::from(distance);
    if word.len().abs_diff(term.len()) > most {
        return false;
    }
    let mut last: Vec<usize> = (0..=term.len()).collect();
    let mut row = vec![0; term.len() + 1];
    for (i, a) in word.iter().enumerate() {
        row[0] = i + 1;
        for (j, b) in term.iter().enumerate() {
            let cost = usize::from(a != b);
            row[j + 1] = (last[j] + cost).min(last[j + 1] + 1).min(row[j] + 1);
        }
        if row.iter().min().copied().unwrap_or(0) > most {
            return false;
        }
        std::mem::swap(&mut last, &mut row);
    }
    last[term.len()] <= most
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dictionary(words: &[&[u8]]) -> Terms {
        let mut terms = Terms::new();
        for (at, word) in words.iter().enumerate() {
            terms.add(word, at as u32 + 1, 1, 1, &[1]);
        }
        terms
    }

    fn names<'a>(found: &[(&'a [u8], &'a Posts)]) -> Vec<&'a [u8]> {
        found.iter().map(|(term, _)| *term).collect()
    }

    #[test]
    fn a_prefix_stands_for_the_terms_under_it_in_order() {
        let terms = dictionary(&[b"abc", b"aba", b"zz"]);
        assert_eq!(names(&under(&terms, b"ab")), [b"aba".as_slice(), b"abc"]);
        assert_eq!(names(&under(&terms, b"zz")), [b"zz".as_slice()]);
    }

    /// Measured: `a*` answers nothing on an index that holds `abc`, where `ab*`
    /// answers it, and the same goes for a suffix and an infix.
    #[test]
    fn one_letter_is_too_short_to_stand_for_anything() {
        let terms = dictionary(&[b"abc", b"aba"]);
        assert!(under(&terms, b"a").is_empty());
        assert!(ending(&terms, b"c").is_empty());
        assert!(inside(&terms, b"b").is_empty());
        assert_eq!(names(&inside(&terms, b"bc")), [b"abc".as_slice()]);
    }

    /// A pattern is not held to that length, which is why `w'a?c'` answers.
    #[test]
    fn a_pattern_may_be_as_short_as_it_likes() {
        let terms = dictionary(&[b"abc", b"aba", b"ac"]);
        assert_eq!(names(&like(&terms, b"a?c")), [b"abc".as_slice()]);
        assert_eq!(names(&like(&terms, b"a*")).len(), 3);
    }

    /// Measured: a real server answers `w'AB*'` with the documents holding
    /// `abc`, so the pattern is folded before it is matched.
    #[test]
    fn a_pattern_is_folded_before_it_is_matched() {
        let terms = dictionary(&[b"abc"]);
        assert_eq!(names(&like(&terms, b"AB*")), [b"abc".as_slice()]);
    }

    #[test]
    fn a_fuzzy_word_stands_for_what_is_near_it() {
        let terms = dictionary(&[b"dog", b"dogs", b"cat"]);
        assert_eq!(
            names(&near(&terms, b"dog", 1)),
            [b"dog".as_slice(), b"dogs"]
        );
        assert_eq!(names(&near(&terms, b"dog", 0)), [b"dog".as_slice()]);
        assert_eq!(names(&near(&terms, b"cot", 1)), [b"cat".as_slice()]);
    }

    /// A stem is a term and nothing here skips it, which is what lets `*og`
    /// answer for a document whose only word is `dogs`.
    #[test]
    fn a_suffix_reaches_a_stem() {
        let terms = dictionary(&[b"+dog", b"dogs"]);
        assert_eq!(names(&ending(&terms, b"og")), [b"+dog".as_slice()]);
    }

    #[test]
    fn no_more_than_the_most_come_out() {
        let mut terms = Terms::new();
        for n in 0..MOST + 50 {
            terms.add(format!("ab{n:04}").as_bytes(), 1, 1, 1, &[1]);
        }
        assert_eq!(under(&terms, b"ab").len(), MOST);
        assert_eq!(ending(&terms, b"00").len(), MOST.min(3));
        assert_eq!(like(&terms, b"ab*").len(), MOST);
        assert_eq!(near(&terms, b"ab0000", 3).len(), MOST);
    }

    #[test]
    fn a_glob_takes_a_run_and_a_single_byte() {
        assert!(glob(b"*", b""));
        assert!(glob(b"**", b"anything"));
        assert!(glob(b"a*c", b"ac"));
        assert!(glob(b"a*c", b"abbbc"));
        assert!(!glob(b"a*c", b"abbb"));
        assert!(glob(b"?", b"a"));
        assert!(!glob(b"?", b""));
        assert!(!glob(b"", b"a"));
        assert!(glob(b"", b""));
        assert!(glob(b"*a*b*", b"xaybz"));
        assert!(!glob(b"*a*b*", b"xbya"));
    }

    #[test]
    fn an_edit_distance_counts_the_three_edits() {
        assert!(within(b"abc", b"abc", 0));
        assert!(within(b"abc", b"abd", 1));
        assert!(within(b"abc", b"ac", 1));
        assert!(within(b"abc", b"abcd", 1));
        assert!(!within(b"abc", b"xyz", 2));
        assert!(within(b"abc", b"xyz", 3));
        assert!(within(b"", b"ab", 2));
        assert!(!within(b"", b"ab", 1));
    }
}
