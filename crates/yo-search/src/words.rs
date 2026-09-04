//! The words a document is broken into, and where in the document they are.
//!
//! This is the other half of the rules in [`crate::token`]. A query arrives as
//! something a client typed and comes apart under a grammar; a document arrives
//! as a value under a key and comes apart under this, which is a much smaller
//! set of rules with no grammar in it at all. What the two have to share is the
//! word: a query looking for `RUNNING` has to find the document that said
//! `running`, so the folding and the stemming are the same code on both sides
//! and only the splitting differs.
//!
//! ```
//! use yo_search::words::Words;
//!
//! let words: Vec<_> = Words::new(b"Hello, the WORLD", None).collect();
//! assert_eq!(words.len(), 2);
//! assert_eq!(&*words[0].text, b"hello");
//! // `the` is a stop word and takes no place, so `world` is the second word
//! // and not the third.
//! assert_eq!(words[1].at, 2);
//! ```

use std::borrow::Cow;

use crate::english::English;
use crate::text;
use crate::token::{control, fold_stored};

/// The longest word an index keeps, in characters.
///
/// A word longer than this is dropped rather than cut short, so a value with
/// one enormous run of letters in it indexes everything except that run. The
/// count is of characters after the folding and not of bytes before it, which
/// is why two hundred and fifty five accented letters go in and a hundred and
/// twenty eight `İ` do not: each of those folds into two characters.
pub const LONGEST: usize = 255;

/// The shortest word that is worth stemming, in bytes.
///
/// Bytes rather than characters, which is not the rule anybody would write down
/// but is the one a real server has: `aés` is three characters and four bytes
/// and gets a stem, and `abs` is three of each and does not.
pub const STEMMED: usize = 4;

/// One word of a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// The word as it goes into the index, folded.
    pub text: Box<[u8]>,
    /// Where it is in the document, counting from one.
    pub at: u32,
}

/// Whether a byte ends one word and starts the next.
///
/// Every ASCII punctuation mark except the underscore and the backslash, plus
/// the space and the tab. The underscore is part of a word, so `ab_cd` is one
/// word and `ab-cd` is two, and the backslash is the escape, so it is not a
/// separator and not part of a word either. Everything above the ASCII range is
/// part of a word, which is how a language that does not put spaces between its
/// words ends up as one long term.
#[must_use]
pub const fn splits(b: u8) -> bool {
    b == b' ' || b == b'\t' || (b.is_ascii_punctuation() && b != b'_' && b != b'\\')
}

/// Every word of a text value, in the order the value holds them.
///
/// The stop words are the index's list, or the built in one when the index did
/// not name any. They have to be known here rather than filtered afterwards
/// because a dropped stop word does not take a place: `hello the world` holds
/// `hello` at one and `world` at two, which is what makes the phrase
/// `"hello world"` match it.
#[derive(Debug)]
pub struct Words<'a> {
    /// The part of the value a real server reads, which it takes apart in
    /// place, so this has to be something that can be written to.
    src: Cow<'a, [u8]>,
    /// How far along the value the walk has got.
    at: usize,
    /// How many words have been given back.
    seen: u32,
    /// The words this index throws away, or `None` for the built in list.
    stops: Option<&'a [Box<[u8]>]>,
}

impl<'a> Words<'a> {
    /// The words of one value.
    #[must_use]
    pub fn new(src: &'a [u8], stops: Option<&'a [Box<[u8]>]>) -> Words<'a> {
        let end = src.iter().position(|b| *b == 0).unwrap_or(src.len());
        Words {
            src: Cow::Borrowed(&src[..end]),
            at: 0,
            seen: 0,
            stops,
        }
    }

    /// Takes the escapes and the unprintable bytes out of one word, in place,
    /// and says how long what is left is.
    ///
    /// In place because that is what a real server does, and it matters: the
    /// bytes the word used to fill are still sitting there afterwards, and
    /// folding the word can read them. So `and\x80<del>` shortens to `and\x80`
    /// with the `<del>` still behind it, and the `\x80` reads it and the word
    /// ends up as `and?`.
    fn compact(&mut self, start: usize, end: usize) -> usize {
        let src = self.src.to_mut();
        let mut write = start;
        let mut read = start;
        while read < end {
            let mut b = src[read];
            read += 1;
            if b == b'\\' {
                // The escape takes the next byte whatever it is, so `a\-b` is
                // one word and `a\yb` is `ayb` rather than the two words a
                // query would read there. A backslash with nothing after it has
                // nothing to take and goes.
                let Some(&next) = src.get(read) else {
                    break;
                };
                b = next;
                read += 1;
            }
            if control(b) {
                // Unprintable and therefore not there, and not a separator
                // either, so the letters either side of it join up.
                continue;
            }
            src[write] = b;
            write += 1;
        }
        write - start
    }
}

impl Iterator for Words<'_> {
    type Item = Word;

    fn next(&mut self) -> Option<Word> {
        loop {
            while self.at < self.src.len() && splits(self.src[self.at]) {
                self.at += 1;
            }
            if self.at >= self.src.len() {
                return None;
            }
            let start = self.at;
            let mut short = false;
            while self.at < self.src.len() {
                let b = self.src[self.at];
                if b == b'\\' {
                    short = true;
                    self.at = (self.at + 2).min(self.src.len());
                    continue;
                }
                if splits(b) {
                    break;
                }
                short |= control(b);
                self.at += 1;
            }
            let end = self.at;
            let head = if short {
                self.compact(start, end)
            } else {
                end - start
            };
            // Folding a word reads past the end of it, because a byte that
            // starts a four byte character is read as one whether or not there
            // are three bytes left in the word. What it reads is the rest of
            // the value, and a value that has run out reads as zeroes.
            let text = fold_stored(&self.src[start..], head);
            if text.is_empty() {
                // Nothing was left of it, which happens when the word was one
                // stray byte that folded away. It is not a word and it does not
                // take a place.
                continue;
            }
            if text::dropped(self.stops, &text) {
                continue;
            }
            self.seen += 1;
            if length(&text) > LONGEST {
                // The place is taken and the word is not indexed, so a phrase
                // either side of a very long word does not close up over it.
                continue;
            }
            return Some(Word {
                text: text.into(),
                at: self.seen,
            });
        }
    }
}

/// The stem an index keeps beside a word, when it keeps one.
///
/// Nothing for a word too short to be worth it and nothing when the stem is the
/// word itself, which is why `ran` and `run` have none and `runs` has `run`.
/// The stem goes into the index beside the word rather than instead of it, so a
/// search for `running` finds `runs` through the stem they share and a search
/// for the exact word still finds only the exact word.
pub fn stem(english: &mut English, word: &[u8]) -> Option<Box<[u8]>> {
    if word.len() < STEMMED {
        return None;
    }
    let stem = english.stem(word);
    (stem != word).then(|| stem.into())
}

/// How many characters a folded word has.
///
/// Counting the bytes that are not continuation bytes, which is the same answer
/// as decoding it and cheaper, and is right for what the folding writes out
/// even where that is not valid UTF-8.
fn length(word: &[u8]) -> usize {
    word.iter().filter(|b| *b & 0xc0 != 0x80).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words of a value, as text, with no index list of stop words.
    fn words(src: &[u8]) -> Vec<Vec<u8>> {
        Words::new(src, None).map(|w| w.text.into_vec()).collect()
    }

    /// The words of a value with where each one is.
    fn places(src: &[u8]) -> Vec<(Vec<u8>, u32)> {
        Words::new(src, None)
            .map(|w| (w.text.into_vec(), w.at))
            .collect()
    }

    #[test]
    fn a_value_comes_apart_at_its_spaces_and_its_punctuation() {
        assert_eq!(
            words(b"hello world"),
            [b"hello".to_vec(), b"world".to_vec()]
        );
        assert_eq!(words(b"ab-cd"), [b"ab".to_vec(), b"cd".to_vec()]);
        assert_eq!(words(b"ab_cd"), [b"ab_cd".to_vec()]);
        assert_eq!(
            words(b"  spaced   out  "),
            [b"spaced".to_vec(), b"out".to_vec()]
        );
        assert_eq!(words(b"e-mail"), [b"e".to_vec(), b"mail".to_vec()]);
        assert_eq!(words(b""), Vec::<Vec<u8>>::new());
    }

    /// A number is not a number here, it is whatever letters and digits are
    /// left when the marks between them have separated them.
    #[test]
    fn a_number_comes_apart_like_anything_else() {
        assert_eq!(words(b"3.14159"), [b"3".to_vec(), b"14159".to_vec()]);
        assert_eq!(words(b"1,000"), [b"1".to_vec(), b"000".to_vec()]);
        assert_eq!(
            words(b"x1 2.5 -3"),
            [b"x1".to_vec(), b"2".to_vec(), b"5".to_vec(), b"3".to_vec()]
        );
    }

    /// The escape a document takes is not the escape a query takes: it reaches
    /// any byte at all, and it only comes off once.
    #[test]
    fn a_backslash_takes_the_byte_after_it_whatever_that_is() {
        assert_eq!(words(br"aa\-bb"), [b"aa-bb".to_vec()]);
        assert_eq!(words(br"aa\ bb"), [b"aa bb".to_vec()]);
        assert_eq!(words(br"a\yb"), [b"ayb".to_vec()]);
        assert_eq!(words(br"aa\\bb"), [br"aa\bb".to_vec()]);
        assert_eq!(words(br"aa\\-bb"), [br"aa\".to_vec(), b"bb".to_vec()]);
        assert_eq!(words(br"aa\\\-bb"), [br"aa\-bb".to_vec()]);
        assert_eq!(words(br"aa\"), [b"aa".to_vec()]);
    }

    #[test]
    fn a_word_is_folded_on_the_way_in() {
        assert_eq!(words(b"MiXeD CaSe"), [b"mixed".to_vec(), b"case".to_vec()]);
        assert_eq!(words("ÉTÉ".as_bytes()), ["été".as_bytes().to_vec()]);
        assert_eq!(
            words("中文 test".as_bytes()),
            ["中文".as_bytes().to_vec(), b"test".to_vec()]
        );
    }

    /// A stop word is not indexed and does not take a place, which is what lets
    /// a phrase close up over it.
    #[test]
    fn a_stop_word_leaves_no_gap_behind_it() {
        assert_eq!(words(b"a the of"), Vec::<Vec<u8>>::new());
        assert_eq!(
            places(b"hello the world"),
            [(b"hello".to_vec(), 1), (b"world".to_vec(), 2)]
        );
        assert_eq!(
            places(b"hello big world"),
            [
                (b"hello".to_vec(), 1),
                (b"big".to_vec(), 2),
                (b"world".to_vec(), 3)
            ]
        );
    }

    /// A word of its own list is dropped and a word of the built in one is not,
    /// because the list replaces the default rather than adding to it.
    #[test]
    fn an_index_with_its_own_list_uses_that_one() {
        let mine: Vec<Box<[u8]>> = vec![b"hello".to_vec().into()];
        let got: Vec<_> = Words::new(b"hello the world", Some(&mine))
            .map(|w| (w.text.into_vec(), w.at))
            .collect();
        assert_eq!(got, [(b"the".to_vec(), 1), (b"world".to_vec(), 2)]);
    }

    /// The long word is gone and the place it stood in is not, so `ok` and
    /// `fine` are not next to each other and the phrase does not match.
    #[test]
    fn a_word_too_long_to_index_still_takes_its_place() {
        let long = [b'q'; LONGEST + 1];
        let mut value = b"ok ".to_vec();
        value.extend_from_slice(&long);
        value.extend_from_slice(b" fine");
        assert_eq!(places(&value), [(b"ok".to_vec(), 1), (b"fine".to_vec(), 3)]);
        assert_eq!(words(&[b'q'; LONGEST]).len(), 1);
    }

    /// The count is of characters after the folding, so a letter that folds
    /// into two counts twice and half as many of them fit.
    #[test]
    fn the_length_of_a_word_is_counted_in_folded_characters() {
        let accented = "é".repeat(LONGEST).into_bytes();
        assert_eq!(words(&accented).len(), 1);
        let over = "é".repeat(LONGEST + 1).into_bytes();
        assert!(words(&over).is_empty());
        // `İ` folds into an `i` and a combining dot, so a hundred and
        // twenty seven of them fit and a hundred and twenty eight do not.
        assert_eq!(words(&"İ".repeat(127).into_bytes()).len(), 1);
        assert!(words(&"İ".repeat(128).into_bytes()).is_empty());
    }

    /// The value stops at the first zero byte and the unprintable bytes inside
    /// it are dropped where they stand rather than separating anything.
    #[test]
    fn what_cannot_be_printed_is_not_read() {
        assert_eq!(words(b"zz\0yy"), [b"zz".to_vec()]);
        assert_eq!(words(b"zz\0 yy"), [b"zz".to_vec()]);
        assert_eq!(words(b"ab\x7fcd"), [b"abcd".to_vec()]);
        assert_eq!(words(b"ab\x01cd"), [b"abcd".to_vec()]);
        assert_eq!(words(b"ab\ncd"), [b"abcd".to_vec()]);
        assert_eq!(words(b"ab\tcd"), [b"ab".to_vec(), b"cd".to_vec()]);
        assert_eq!(
            places(b"ok \x01 fine"),
            [(b"ok".to_vec(), 1), (b"fine".to_vec(), 2)]
        );
    }

    /// A byte that starts a character it has no room for reads the bytes after
    /// the word, which is what a real server does because the word it folds is
    /// still sitting in the value it came out of.
    #[test]
    fn a_broken_character_reads_past_the_end_of_its_word() {
        assert_eq!(words(b"zz\x80yy"), [b"zz9y".to_vec()]);
        assert_eq!(words(b"zz\x80 yy"), [b"zz ".to_vec(), b"yy".to_vec()]);
        assert_eq!(words(b"zz\x80-yy"), [b"zz-".to_vec(), b"yy".to_vec()]);
        assert_eq!(
            words(b"zz\xe0-yy"),
            ["zz\u{b79}".as_bytes().to_vec(), b"yy".to_vec()]
        );
        // Nothing is left of it, so it is not a word at all.
        assert_eq!(words(b"\x80"), Vec::<Vec<u8>>::new());
    }

    /// Taking the unprintable bytes out of a word shortens it but leaves what
    /// was there behind the end of it, so a broken character at the end reads
    /// the byte that was just dropped rather than the one that follows in the
    /// value. Here the delete is taken out and then read back as the second
    /// half of the character the `0x80` starts, which makes a question mark.
    #[test]
    fn a_word_is_shortened_in_place_and_what_it_leaves_behind_is_read() {
        assert_eq!(words(b"and\x80\x7f,"), [b"and?".to_vec()]);
        assert_eq!(words(b"and\x80,"), [b"and,".to_vec()]);
    }

    /// A character above the sixteen bit range does not survive being indexed,
    /// so an emoji in a document is stored as something else and the query that
    /// looks for it never finds it.
    #[test]
    fn a_character_too_wide_to_store_is_cut_down() {
        assert_eq!(
            words("zz😀yy".as_bytes()),
            ["zz\u{f600}yy".as_bytes().to_vec()]
        );
        assert_eq!(words("𐒰".as_bytes()), ["\u{4d8}".as_bytes().to_vec()]);
    }

    #[test]
    fn a_word_is_stemmed_when_it_is_long_enough_and_the_stem_says_something() {
        let mut english = English::new();
        assert_eq!(stem(&mut english, b"runs").as_deref(), Some(&b"run"[..]));
        assert_eq!(stem(&mut english, b"running").as_deref(), Some(&b"run"[..]));
        assert_eq!(stem(&mut english, b"cats").as_deref(), Some(&b"cat"[..]));
        assert_eq!(stem(&mut english, b"run"), None);
        assert_eq!(stem(&mut english, b"ran"), None);
        assert_eq!(stem(&mut english, b"ays"), None);
        assert_eq!(stem(&mut english, b"abs"), None);
        assert_eq!(
            stem(&mut english, "aés".as_bytes()).as_deref(),
            Some("aé".as_bytes())
        );
    }
}
