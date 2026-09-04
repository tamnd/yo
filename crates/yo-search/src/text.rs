//! The words a text field throws away before it indexes anything.
//!
//! A stopword is a word so common that indexing it costs a posting list the
//! length of the corpus and buys nothing, because a query that contains it means
//! the same thing without it. Every text field drops them on the way in and
//! every query drops them on the way out, so a query of nothing but stopwords
//! matches nothing at all rather than matching everything.
//!
//! Which words those are is part of the index and not a matter of taste. An
//! index built with one list and queried under another answers wrongly and
//! quietly, so the list is fixed at create time, kept with the index and
//! reported back by `FT.INFO` when the client chose it.

/// The words dropped when the client did not say otherwise.
///
/// Thirty three of them, found one word at a time by asking an 8.10.1 to
/// explain a query of that word and seeing it come back empty, over a candidate
/// list of a hundred and fifty of the commonest words in English. The order
/// here is nothing in particular, because nothing reports this list: `FT.INFO`
/// only names stopwords when the client chose them.
pub const DEFAULT: [&[u8]; 33] = [
    b"a", b"is", b"the", b"an", b"and", b"are", b"as", b"at", b"be", b"but", b"by", b"for", b"if",
    b"in", b"into", b"it", b"no", b"not", b"of", b"on", b"or", b"such", b"that", b"their", b"then",
    b"there", b"these", b"they", b"this", b"to", b"was", b"will", b"with",
];

/// Whether a word is dropped, under the list this index was built with.
///
/// `None` means the client never mentioned stopwords and gets the default list.
/// An empty list is not the same thing as no list: `STOPWORDS 0` is how a client
/// says to keep everything, and it has to survive the round trip as an empty
/// list rather than being folded back into the default.
///
/// The word is compared as it arrives. Everything that reaches here has already
/// been folded to lower case by the tokeniser, and a custom list is folded on
/// the way in for the same reason.
#[must_use]
pub fn dropped(list: Option<&[Box<[u8]>]>, word: &[u8]) -> bool {
    match list {
        Some(list) => list.iter().any(|w| **w == *word),
        None => DEFAULT.contains(&word),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_list_is_used_when_the_client_said_nothing() {
        assert!(dropped(None, b"the"));
        assert!(dropped(None, b"with"));
        assert!(!dropped(None, b"hello"));
        assert!(!dropped(None, b"about"));
    }

    /// `STOPWORDS 0` means keep everything, and it has to be told apart from
    /// never having asked.
    #[test]
    fn an_empty_list_keeps_every_word() {
        let none: Vec<Box<[u8]>> = Vec::new();
        assert!(!dropped(Some(&none), b"the"));
        assert!(!dropped(Some(&none), b"hello"));
    }

    #[test]
    fn a_list_of_its_own_replaces_the_default_rather_than_adding_to_it() {
        let mine: Vec<Box<[u8]>> = vec![b"hello".to_vec().into(), b"world".to_vec().into()];
        assert!(dropped(Some(&mine), b"hello"));
        assert!(!dropped(Some(&mine), b"the"));
    }

    #[test]
    fn the_default_list_has_no_word_in_it_twice() {
        let mut seen = std::collections::BTreeSet::new();
        for w in DEFAULT {
            assert!(
                seen.insert(w),
                "{} is in the list twice",
                String::from_utf8_lossy(w)
            );
        }
    }
}
