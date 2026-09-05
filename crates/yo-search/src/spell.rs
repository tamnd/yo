//! What the index would have read, for a word it has never seen.
//!
//! `FT.SPELLCHECK` runs a query the way `FT.SEARCH` would, takes every plain
//! word out of the parsed tree and answers about the ones the index does not
//! hold. A word it does hold is not a mistake and is left out of the reply
//! altogether, so a query with nothing wrong with it answers an empty list.
//!
//! ```
//! use yo_search::query::{Ask, parse};
//! use yo_search::spell::{Lists, NEAREST, check};
//! use yo_search::{Definition, English, Field, Index, Kind, Text};
//!
//! let mut index = Index::new(b"ix", Definition::default(), vec![
//!     Field::new(b"t", Kind::Text(Text::default())),
//! ]);
//! let mut english = English::new();
//! index.write(&mut english, b"d:1", &[(b"t", b"hello")])?;
//!
//! let node = parse(b"helo", &index, &Ask::default()).expect("a query that parses");
//! let found = check(&index, &node, NEAREST, &Lists::default());
//! assert_eq!(&*found[0].word, b"helo");
//! assert_eq!(&*found[0].guesses[0].term, b"hello");
//! # Ok::<(), yo_search::held::Failed>(())
//! ```
//!
//! # The score is how common the word is, not how close it is
//!
//! A suggestion carries the share of the index's documents that hold it, so a
//! word in three documents out of nine scores a third. Nothing about the edit
//! distance goes into it, which means a word one letter away and a word three
//! letters away are told apart only by how often they are used. All of that is
//! measured against a real server rather than reasoned about, including that
//! the denominator is the live document count and not the largest document
//! number handed out.
//!
//! # The field a query names narrows the candidates and not the score
//!
//! `@a:helo` only suggests words the index has read out of `a`, and the score
//! those words come back with is still counted over the whole index. So a word
//! in two documents, one through `a` and one through `b`, scores the same
//! whichever field the query named, and disappears entirely when the query
//! names a third field it was never in. A `TAG` field is the exception: the
//! field a tag value named is dropped rather than narrowed to, so `@g:{helo}`
//! suggests every text word near `helo` and none of the tag values near it.
//!
//! # A word already in the index is never a mistake, whatever field it asks
//!
//! The lookup that decides that is over the whole dictionary and pays no
//! attention to the field. `@a:world` answers nothing on an index that only
//! read `world` out of `b`, which is not what anyone would design and is what
//! a real server does.
//!
//! # How far out a suggestion may be is measured exactly
//!
//! The distance counts characters and not bytes, so `ecole` reaches `école` in
//! one edit. A real server's automaton is hand built and lets a few words
//! through at one edit more than was asked for, which is D-88 and is not copied
//! here, because there is no rule behind which words those are.
//!
//! # Stopwords are checked and stems are suggested
//!
//! A query is parsed with the stopword list turned off here, because a real
//! server checks `then` and suggests `thin` for it. Nothing ever indexes a
//! stopword, so one always comes back as a word the index does not hold. Going
//! the other way, the dictionary the candidates come out of is the raw one, so
//! a stem is a candidate like any other and `run` really does suggest `+run`,
//! with the [`crate::posts::STEM`] byte and all.

use crate::expand::within;
use crate::index::Index;
use crate::query::{Mask, Node, What, Word};
use crate::token::fold;

/// The edit distance a check uses when the client did not name one.
pub const NEAREST: u8 = 1;

/// The furthest apart a word and a suggestion may be.
pub const FURTHEST: u8 = 4;

/// One word the query asked about and everything the index has to say about it.
#[derive(Debug, Clone)]
pub struct Checked {
    /// The word, folded the way the query parser folded it.
    pub word: Box<[u8]>,
    /// What the index might have meant, best first.
    pub guesses: Vec<Guess>,
}

/// One word the index holds that the query might have meant.
#[derive(Debug, Clone)]
pub struct Guess {
    /// The share of the index's documents that hold it, or nought for a word
    /// that only a dictionary knows about.
    pub score: f64,
    /// The word, in the spelling it is held under, which for a word out of a
    /// dictionary is the spelling that was added rather than the folded one.
    pub term: Box<[u8]>,
}

/// The word lists a check was handed on top of the index.
///
/// Both are the terms of the dictionaries the client named, run together, since
/// naming the same dictionary twice or two dictionaries holding the same word
/// changes nothing about the answer.
#[derive(Debug, Default)]
pub struct Lists<'a> {
    /// Words that are candidates on top of the ones the index holds.
    pub include: Vec<&'a [u8]>,
    /// Words that are spelled correctly, so a query using one is not asked
    /// about at all.
    pub exclude: Vec<&'a [u8]>,
}

/// Every word of a query the index does not hold, and what it might have meant.
///
/// The words come back in the order the query wrote them and a word written
/// twice is answered twice, because a real server walks the tree rather than
/// gathering a set.
#[must_use]
pub fn check(index: &Index, node: &Node, distance: u8, lists: &Lists<'_>) -> Vec<Checked> {
    let mut words = Vec::new();
    asked(node, Mask::MAX, &mut words);
    let known: Vec<Vec<u8>> = lists.exclude.iter().map(|word| fold(word)).collect();
    words
        .into_iter()
        .filter(|(word, _)| !known.iter().any(|right| *right == **word))
        .filter(|(word, _)| index.held.dictionary().get(word).is_none())
        .map(|(word, mask)| Checked {
            guesses: guesses(index, &word, mask, distance, lists),
            word,
        })
        .collect()
}

/// Every word of a parsed query that stands for itself, with the fields it
/// asks.
///
/// A prefix, a suffix, an infix, a pattern and a fuzzy word all stand for terms
/// the index already holds, so none of them is a word anybody misspelled and
/// none of them is here. Nor is a stem, which nobody wrote.
fn asked(node: &Node, mask: Mask, into: &mut Vec<(Box<[u8]>, Mask)>) {
    let mask = mask & node.mask;
    match &node.what {
        What::Term(Word {
            word,
            expanded: false,
            ..
        }) => into.push((word.clone(), mask)),
        What::Union(list) | What::Intersect(list) | What::Exact(list) => {
            for child in list {
                asked(child, mask, into);
            }
        }
        What::Not(child) | What::Optional(child) => asked(child, mask, into),
        // A tag value is a word and is answered like one, but the field it
        // named is dropped on the way in. `@g:{helo}` suggests every text word
        // near `helo` and not the tag values near it, which is measured, and
        // it happens because the check is over the term dictionary while a tag
        // value never went in there.
        What::Tag(_, list) => {
            let mut under = Vec::new();
            for child in list {
                asked(child, Mask::MAX, &mut under);
            }
            into.extend(under.into_iter().map(|(word, _)| (word, Mask::MAX)));
        }
        _ => {}
    }
}

/// What one word might have been, best first.
fn guesses(index: &Index, word: &[u8], mask: Mask, distance: u8, lists: &Lists<'_>) -> Vec<Guess> {
    let docs = index.held.docs.len();
    let mut found: Vec<(Vec<u8>, Guess)> = Vec::new();
    for (term, posts) in index.held.dictionary().under(b"") {
        if !within(word, term, distance) {
            continue;
        }
        // The field is asked of the postings and not of the term, because a
        // word read out of two fields is one term with two kinds of posting
        // under it, and one document reaching the field is enough.
        let mut reader = posts.read();
        let mut reaches = false;
        while let Some(post) = reader.step() {
            if u64::from(post.fields) & mask != 0 {
                reaches = true;
                break;
            }
        }
        if !reaches {
            continue;
        }
        let share = if docs == 0 {
            0.0
        } else {
            f64::from(posts.len()) / docs as f64
        };
        found.push((
            term.to_vec(),
            Guess {
                score: share,
                term: term.into(),
            },
        ));
    }
    for extra in &lists.include {
        let folded = fold(extra);
        if !within(word, &folded, distance) || found.iter().any(|(had, _)| *had == folded) {
            continue;
        }
        found.push((
            folded,
            Guess {
                score: 0.0,
                term: (*extra).into(),
            },
        ));
    }
    let mut guesses = branching(found.into_iter().map(|(_, guess)| guess).collect());
    // The commonest first, and a tie left exactly as the walk above laid it
    // out, which is what makes this a stable sort and not any other kind.
    guesses.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    guesses
}

/// The suggestions in the order a trie over them hands them back.
///
/// This is the one part of the command that is a copy of how a real server is
/// built rather than of what it answers. It keeps the suggestions in a trie and
/// reads them out of it before sorting, and its trie keeps a node's children in
/// the order of the best score anywhere under them, highest first, so that an
/// autocomplete can stop early. Two children that are equally good are in
/// character order. The reply then sorts by score and leaves ties alone, so the
/// trie's order is what a client actually sees among the words that are equally
/// common.
///
/// Nothing about that is documented and all of it shows: `hellp` on an index
/// holding `hallo`, `hell`, `hello`, `helm` and `help` answers `hello`, `help`,
/// `hell`, `helm`, `hallo`, where plain character order would have put `hallo`
/// third. `hell` comes before `helm` because a node's own word is read before
/// anything under it, and `hallo` comes last because `hel` leads to a word in
/// two documents while `hal` leads to one in a single document.
///
/// The character and not the byte is what is compared, so `école` sorts after
/// `world`, and the spelling that is compared is the one that will be reported,
/// so a dictionary word `Zaaa` sorts in front of `aaab`.
fn branching(guesses: Vec<Guess>) -> Vec<Guess> {
    let rows: Vec<(Vec<char>, f64)> = guesses
        .iter()
        .map(|guess| (letters(&guess.term), guess.score))
        .collect();
    let all: Vec<usize> = (0..rows.len()).collect();
    let mut order = Vec::with_capacity(rows.len());
    node(&rows, &all, 0, &mut order);
    let mut guesses: Vec<Option<Guess>> = guesses.into_iter().map(Some).collect();
    order
        .into_iter()
        .filter_map(|at| guesses[at].take())
        .collect()
}

/// One node of that trie: the word that ends here, then everything under it.
fn node(rows: &[(Vec<char>, f64)], at: &[usize], depth: usize, order: &mut Vec<usize>) {
    order.extend(at.iter().copied().filter(|i| rows[*i].0.len() == depth));
    let mut kids: Vec<(char, f64, Vec<usize>)> = Vec::new();
    for &i in at {
        let Some(&letter) = rows[i].0.get(depth) else {
            continue;
        };
        match kids.iter_mut().find(|(had, _, _)| *had == letter) {
            Some((_, best, list)) => {
                *best = best.max(rows[i].1);
                list.push(i);
            }
            None => kids.push((letter, rows[i].1, vec![i])),
        }
    }
    kids.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    for (_, _, list) in &kids {
        node(rows, list, depth + 1, order);
    }
}

/// The characters of a word, with anything that is not valid UTF-8 read one
/// byte at a time so a term that came in as rubbish still sorts.
fn letters(word: &[u8]) -> Vec<char> {
    match core::str::from_utf8(word) {
        Ok(text) => text.chars().collect(),
        Err(_) => word.iter().map(|b| char::from(*b)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Definition;
    use crate::query::{Ask, parse};
    use crate::{English, Field, Kind, Tag, Text};

    /// An index over two text fields and a tag, with no stemming, which keeps
    /// the dictionary to the words that were written.
    fn indexed(docs: &[(&[u8], &[u8], &[u8])]) -> Index {
        let text = Text {
            nostem: true,
            ..Text::default()
        };
        let mut index = Index::new(
            b"ix",
            Definition::default(),
            vec![
                Field::new(b"a", Kind::Text(text.clone())),
                Field::new(b"b", Kind::Text(text)),
                Field::new(b"g", Kind::Tag(Tag::default())),
            ],
        );
        let mut english = English::new();
        for (key, a, b) in docs {
            index
                .write(&mut english, key, &[(b"a", a), (b"b", b)])
                .expect("a document that indexes");
        }
        index
    }

    /// The answer in a shape a test can read.
    fn ran(index: &Index, query: &[u8], lists: &Lists<'_>) -> Vec<(String, Vec<(f64, String)>)> {
        let node = parse(query, index, &Ask::default()).expect("a query that parses");
        check(index, &node, NEAREST, lists)
            .into_iter()
            .map(|c| {
                (
                    String::from_utf8_lossy(&c.word).into_owned(),
                    c.guesses
                        .into_iter()
                        .map(|g| (g.score, String::from_utf8_lossy(&g.term).into_owned()))
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn a_word_the_index_holds_is_not_asked_about() {
        let index = indexed(&[(b"d1", b"hello", b""), (b"d2", b"world", b"")]);
        assert!(ran(&index, b"hello", &Lists::default()).is_empty());
        let found = ran(&index, b"helo", &Lists::default());
        assert_eq!(found[0].0, "helo");
        assert_eq!(found[0].1, [(0.5, "hello".to_owned())]);
    }

    /// The share is over every document and the edit distance never enters it.
    #[test]
    fn the_score_is_how_many_documents_hold_the_word() {
        let index = indexed(&[
            (b"d1", b"hello", b""),
            (b"d2", b"hello", b""),
            (b"d3", b"held", b""),
            (b"d4", b"other", b""),
        ]);
        let found = ran(&index, b"helo", &Lists::default());
        assert_eq!(
            found[0].1,
            [(0.5, "hello".to_owned()), (0.25, "held".to_owned())]
        );
    }

    /// The field narrows which words are candidates and leaves the share alone,
    /// and a word the index holds is not asked about whichever field asked it.
    #[test]
    fn the_field_a_query_names_narrows_the_candidates() {
        let index = indexed(&[
            (b"d1", b"hello", b""),
            (b"d2", b"", b"hellp"),
            (b"d3", b"zz", b""),
            (b"d4", b"yy", b""),
        ]);
        assert_eq!(ran(&index, b"hellz", &Lists::default())[0].1.len(), 2);
        assert_eq!(
            ran(&index, b"@a:hellz", &Lists::default())[0].1,
            [(0.25, "hello".to_owned())]
        );
        assert_eq!(
            ran(&index, b"@b:hellz", &Lists::default())[0].1,
            [(0.25, "hellp".to_owned())]
        );
        assert!(ran(&index, b"@b:hello", &Lists::default()).is_empty());
    }

    /// A tag value is asked about like any other word and the field it named
    /// is dropped rather than narrowed to, so it is answered out of the text
    /// the index read and not out of the tag values.
    #[test]
    fn a_tag_value_is_asked_about_over_every_field() {
        let index = indexed(&[(b"d1", b"hello", b"")]);
        let found = ran(&index, b"@g:{helo}", &Lists::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1, [(1.0, "hello".to_owned())]);
    }

    /// Ties are not in character order. They are in the order a trie over the
    /// suggestions reads them out, a node's own word first and its children
    /// best score first, so `hell` and `helm` come in front of `hallo` even
    /// though all three are equally common and `hallo` sorts first.
    #[test]
    fn a_tie_is_left_in_the_order_the_trie_read_it() {
        let index = indexed(&[
            (b"d1", b"hello hell", b""),
            (b"d2", b"hello help", b""),
            (b"d3", b"help helm", b""),
            (b"d4", b"hallo", b""),
            (b"d5", b"other", b""),
        ]);
        let node = parse(b"hellp", &index, &Ask::default()).expect("a query that parses");
        let found = check(&index, &node, 2, &Lists::default());
        let words: Vec<String> = found[0]
            .guesses
            .iter()
            .map(|g| String::from_utf8_lossy(&g.term).into_owned())
            .collect();
        assert_eq!(words, ["hello", "help", "hell", "helm", "hallo"]);
    }

    /// A dictionary adds candidates that score nothing and keeps its own
    /// spelling, and a word already suggested is not suggested twice.
    #[test]
    fn a_dictionary_adds_words_the_index_never_read() {
        let index = indexed(&[(b"d1", b"hello", b"")]);
        let lists = Lists {
            include: vec![b"HELLO", b"hellp"],
            exclude: Vec::new(),
        };
        let found = ran(&index, b"hellz", &lists);
        assert_eq!(
            found[0].1,
            [(1.0, "hello".to_owned()), (0.0, "hellp".to_owned())]
        );
    }

    /// A word a dictionary calls correct is not asked about at all, however it
    /// was spelled when it was added.
    #[test]
    fn a_dictionary_can_say_a_word_is_spelled_right() {
        let index = indexed(&[(b"d1", b"hello", b"")]);
        let lists = Lists {
            include: Vec::new(),
            exclude: vec![b"HELLZ"],
        };
        assert!(ran(&index, b"hellz", &lists).is_empty());
    }

    /// Every word of the query in the order it was written, twice when it was
    /// written twice, and nothing at all out of a prefix.
    #[test]
    fn the_words_come_back_in_the_order_they_were_written() {
        let index = indexed(&[(b"d1", b"hello", b"")]);
        let found = ran(&index, b"aaa|bbb|aaa|hel*", &Lists::default());
        let words: Vec<&str> = found.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(words, ["aaa", "bbb", "aaa"]);
    }
}
