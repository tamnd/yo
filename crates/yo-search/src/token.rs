//! The rules that turn bytes into words, shared by the query and the documents.
//!
//! A query and a document have to agree on this or nothing works. If a document
//! stores `ÉTÉ` under `été` and a query looks for it under `ÉTÉ`, the term is in
//! the index and the search still finds nothing, and there is no later stage
//! that can put that right. So the folding, the letters a word is made of and
//! the escape are one set of rules in one place rather than two sets that agree
//! for as long as somebody keeps them agreeing.

/// Whether a byte can be part of a bare word.
///
/// The underscore is in and every other punctuation mark is out, which is what
/// makes `ab_cd` one word and `ab-cd` two. Anything above the ASCII range is in,
/// so a word in any other language stays whole without this having to know
/// which language it is.
#[must_use]
pub const fn wordy(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

/// Whether a backslash written in front of this byte takes it as a letter.
///
/// A printable mark and a space can be escaped, and nothing else can. A
/// backslash in front of a letter or a digit escapes nothing and ends the word
/// instead, so `aa\yy` is two words where `aa\-bb` and `aa\ bb` are each one,
/// and a backslash in front of a control byte that is not a space does the same.
/// That is not a rule anybody would design, but it is the one a real server's
/// scanner has, because the escape is one alternative in the pattern that
/// matches a word rather than a thing the scanner does before it looks.
#[must_use]
pub const fn escapes(b: u8) -> bool {
    matches!(b, 0x09..=0x0d | b' ') || (b.is_ascii_graphic() && !b.is_ascii_alphanumeric())
}

/// Whether a byte ends a word by being unprintable.
///
/// The control bytes and `DEL` end a word the way a space does. A real server
/// does not treat them as separators in a document, where they are dropped and
/// the words on either side run together, and that difference is real: a query
/// looks for two words where the document holds one.
#[must_use]
pub const fn control(b: u8) -> bool {
    b < 0x20 || b == 0x7f
}

/// A word with the backslashes that escape something taken off it.
///
/// A backslash that escapes nothing is kept, because it is not an escape and
/// there is nothing else to call it. This is what turns the `aa\-bb` a client
/// wrote into the `aa-bb` that goes into the index, and it is separate from the
/// folding because a tag value gets one of the two and a word gets both.
///
/// The word is walked again until a walk finds nothing left to take off, which
/// is not what anybody would write down as the rule but is what a real server
/// does: `a\\\-b` is `a-b` there and any even run of backslashes at all comes
/// out as the one backslash. Unescaping the once leaves `a\-b`, which still
/// reads as an escape to everything downstream, so the escape a client wrote to
/// protect a backslash does not survive being read.
#[must_use]
pub fn bare(src: &[u8]) -> Vec<u8> {
    if !src.contains(&b'\\') {
        return src.to_vec();
    }
    let mut out = src.to_vec();
    loop {
        let mut next = Vec::with_capacity(out.len());
        let mut at = 0;
        while at < out.len() {
            if out[at] == b'\\' && matches!(out.get(at + 1), Some(b) if escapes(*b)) {
                next.push(out[at + 1]);
                at += 2;
                continue;
            }
            next.push(out[at]);
            at += 1;
        }
        if next.len() == out.len() {
            return next;
        }
        out = next;
    }
}

/// A word folded to lower case, the way the whole family folds it.
///
/// This is the full Unicode lowercase mapping and not the ASCII one, so `ÉTÉ`
/// is `été`, `ЖУК` is `жук` and `İ` is the two characters `i` and a combining
/// dot rather than the one it was written as. The mapping is per character with
/// no regard for what is around it, which is why `ΟΔΟΣ` folds to `οδοσ` and not
/// to `οδος`: the final sigma rule is a conditional mapping and a real server
/// does not apply it.
///
/// Bytes that are not valid UTF-8 are handed back as they arrived. There is
/// nothing sensible to fold them to, and a value that reaches here is whatever
/// the client wrote rather than something that has been validated.
#[must_use]
pub fn fold(src: &[u8]) -> Vec<u8> {
    // The common case is a word that is already lower case and all ASCII, and
    // walking it once to find that out is cheaper than building a second copy
    // of it every time.
    if src.iter().all(|b| b.is_ascii() && !b.is_ascii_uppercase()) {
        return src.to_vec();
    }
    let mut out = Vec::with_capacity(src.len());
    let mut rest = src;
    loop {
        match std::str::from_utf8(rest) {
            Ok(s) => {
                push_folded(&mut out, s);
                return out;
            }
            Err(e) => {
                let (good, bad) = rest.split_at(e.valid_up_to());
                // Safe because `valid_up_to` is where the decoder stopped.
                push_folded(&mut out, std::str::from_utf8(good).unwrap_or_default());
                let skip = e.error_len().unwrap_or(bad.len()).max(1);
                out.extend_from_slice(&bad[..skip.min(bad.len())]);
                rest = &bad[skip.min(bad.len())..];
            }
        }
    }
}

/// The folded form of a run that is known to decode.
fn push_folded(out: &mut Vec<u8>, s: &str) {
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c.to_ascii_lowercase() as u8);
        } else {
            for c in c.to_lowercase() {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_folds_the_way_it_always_did() {
        assert_eq!(fold(b"MiXeD"), b"mixed");
        assert_eq!(fold(b"already"), b"already");
        assert_eq!(fold(b""), b"");
    }

    /// Every one of these was read off a real server, which folds by the full
    /// Unicode mapping and not by the ASCII one.
    #[test]
    fn every_alphabet_folds_and_not_only_the_first_one() {
        assert_eq!(fold("ÉTÉ".as_bytes()), "été".as_bytes());
        assert_eq!(fold("ЖУК".as_bytes()), "жук".as_bytes());
        assert_eq!(fold("ΑΒΓ".as_bytes()), "αβγ".as_bytes());
        assert_eq!(fold("ÅNGSTRÖM".as_bytes()), "ångström".as_bytes());
        assert_eq!(fold("ǅ".as_bytes()), "ǆ".as_bytes());
    }

    /// A mapping that gives back more characters than it was given, which is
    /// why the folded form cannot be written over the word in place.
    #[test]
    fn a_letter_may_fold_to_more_than_one() {
        assert_eq!(fold("İ".as_bytes()), "i\u{307}".as_bytes());
    }

    /// The rule that needs to know what is around a letter, which a real server
    /// does not apply, so neither does this.
    #[test]
    fn a_sigma_at_the_end_of_a_word_folds_like_any_other_sigma() {
        assert_eq!(fold("ΟΔΟΣ".as_bytes()), "οδοσ".as_bytes());
    }

    #[test]
    fn bytes_that_are_not_a_word_at_all_come_back_as_they_went_in() {
        assert_eq!(fold(b"aa\xffbb"), b"aa\xffbb");
        assert_eq!(fold(b"\xc3"), b"\xc3");
        assert_eq!(fold(b"A\xffB"), b"a\xffb");
    }

    /// A backslash that protects a backslash does not survive being read, which
    /// is what a real server does and not what its manual says.
    #[test]
    fn an_escape_comes_off_and_then_comes_off_again() {
        assert_eq!(bare(br"a\-b"), b"a-b");
        assert_eq!(bare(br"a\\\-b"), b"a-b");
        assert_eq!(bare(br"a\\"), br"a\");
        assert_eq!(bare(br"a\\\\"), br"a\");
        assert_eq!(bare(br"x\\y"), br"x\y");
        assert_eq!(bare(br"x\\\\y"), br"x\y");
        assert_eq!(bare(br"x\"), br"x\");
        assert_eq!(bare(b"plain"), b"plain");
    }

    #[test]
    fn a_backslash_only_reaches_a_printable_mark() {
        assert!(escapes(b'-'));
        assert!(escapes(b'\\'));
        assert!(escapes(b'_'));
        assert!(!escapes(b'a'));
        assert!(!escapes(b'0'));
        assert!(escapes(b' '));
        assert!(escapes(b'\t'));
        assert!(escapes(0x0b));
        assert!(!escapes(0x01));
        assert!(!escapes(0x1f));
        assert!(!escapes(0x7f));
    }

    #[test]
    fn the_bytes_nobody_prints_end_a_word() {
        assert!(control(b'\n'));
        assert!(control(b'\t'));
        assert!(control(0x00));
        assert!(control(0x7f));
        assert!(!control(b' '));
        assert!(!control(b'a'));
    }
}
