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

/// How many bytes a character that starts with this byte is read as.
///
/// This is not quite the UTF-8 rule. It is a range check on the first byte and
/// nothing else, so everything from `0x80` to `0xdf` starts a two byte
/// character, including the continuation bytes that cannot start anything at
/// all. A stray `0x80` in the middle of a word therefore eats the byte after it
/// rather than being refused, and the word that comes out is not the word that
/// went in. That is what a real server does and there is no way to be
/// compatible with it and sensible at the same time.
#[must_use]
pub const fn width(lead: u8) -> usize {
    match lead {
        0x00..=0x7f => 1,
        0x80..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

/// The character starting at `at` and how many bytes it took.
///
/// Every byte after the first is taken for its low six bits whatever it is, and
/// a byte off the end of `src` counts as zero, which is what reading a C string
/// one character at a time does when the string runs out in the middle of one.
/// So the answer can be far outside the range of a character, and the caller
/// has to be ready for that rather than assuming it decoded anything real.
fn rune(src: &[u8], at: usize) -> (u32, usize) {
    let lead = src[at];
    let took = width(lead);
    if took == 1 {
        return (u32::from(lead), 1);
    }
    let mask = match took {
        2 => 0x1f,
        3 => 0x0f,
        _ => 0x07,
    };
    let mut cp = u32::from(lead) & mask;
    for step in 1..took {
        let b = src.get(at + step).copied().unwrap_or(0);
        cp = (cp << 6) | (u32::from(b) & 0x3f);
    }
    (cp, took)
}

/// One character written back out, in as few bytes as it fits in.
///
/// A character that is not a character at all, which is what reading a broken
/// one gives, is written in the same shape as one of its size. Nothing reads
/// these back as text, they are index keys and they only have to be the same
/// key every time.
fn push_rune(out: &mut Vec<u8>, cp: u32) {
    match cp {
        0x0000..=0x007f => out.push(cp as u8),
        0x0080..=0x07ff => {
            out.extend_from_slice(&[0xc0 | (cp >> 6) as u8, 0x80 | (cp & 0x3f) as u8])
        }
        0x0800..=0xffff => out.extend_from_slice(&[
            0xe0 | (cp >> 12) as u8,
            0x80 | ((cp >> 6) & 0x3f) as u8,
            0x80 | (cp & 0x3f) as u8,
        ]),
        _ => out.extend_from_slice(&[
            0xf0 | ((cp >> 18) & 0x07) as u8,
            0x80 | ((cp >> 12) & 0x3f) as u8,
            0x80 | ((cp >> 6) & 0x3f) as u8,
            0x80 | (cp & 0x3f) as u8,
        ]),
    }
}

/// The folding both sides share, over the first `head` bytes of `src`.
///
/// A character whose first byte is inside the head is folded even where the
/// rest of it is not, so the fold reads past the head and stops as soon as a
/// character starts outside it. A character that comes out as zero ends the
/// word there, because what holds it is a C string and there is nothing after a
/// zero in one of those. That is checked after the cutting and not before it,
/// so a character that is only zero once it has been cut down ends the word
/// too, which is why a value of one `0xf0` and a space indexes nothing.
fn folded(src: &[u8], head: usize, cut: bool) -> Vec<u8> {
    let head = head.min(src.len());
    let mut out = Vec::with_capacity(head);
    let mut at = 0;
    'word: while at < head {
        let (cp, took) = rune(src, at);
        at += took;
        match char::from_u32(cp) {
            Some(c) => {
                for c in c.to_lowercase() {
                    let c = narrow(u32::from(c), cut);
                    if c == 0 {
                        break 'word;
                    }
                    push_rune(&mut out, c);
                }
            }
            // Nothing to fold it to, because it is not a character. Whatever
            // came out of the decoding is what goes in.
            None => {
                let cp = narrow(cp, cut);
                if cp == 0 {
                    break 'word;
                }
                push_rune(&mut out, cp);
            }
        }
    }
    out
}

/// A character cut down to sixteen bits, where that is what the caller stores.
const fn narrow(cp: u32, cut: bool) -> u32 {
    if cut { cp & 0xffff } else { cp }
}

/// A word folded to lower case, the way a query folds it.
///
/// This is the full Unicode lowercase mapping and not the ASCII one, so `ÉTÉ`
/// is `été`, `ЖУК` is `жук` and `İ` is the two characters `i` and a combining
/// dot rather than the one it was written as. The mapping is per character with
/// no regard for what is around it, which is why `ΟΔΟΣ` folds to `οδοσ` and not
/// to `οδος`: the final sigma rule is a conditional mapping and a real server
/// does not apply it.
///
/// Bytes that are not valid UTF-8 are read anyway, under [`width`], and what
/// comes out of that is what the word becomes. There is nothing sensible to do
/// with them and this is what a real server does with them.
#[must_use]
pub fn fold(src: &[u8]) -> Vec<u8> {
    // The common case is a word that is already lower case and all ASCII, and
    // walking it once to find that out is cheaper than building a second copy
    // of it every time.
    if src
        .iter()
        .all(|b| b.is_ascii() && !b.is_ascii_uppercase() && *b != 0)
    {
        return src.to_vec();
    }
    folded(src, src.len(), false)
}

/// A word folded the way a document folds it, which is not quite the same.
///
/// Every character is cut down to sixteen bits on the way into an index. A
/// document holding an emoji stores something else, and the query looking for
/// that emoji keeps it whole and therefore never finds the document. That is a
/// real server's behaviour and not ours, and copying it is the only way a
/// document written through one and read through the other says the same thing.
///
/// `src` is the word followed by whatever came after it in the value, and
/// `head` is how long the word is. The tail is there because folding a word
/// whose last character is broken reads past the end of it, into bytes that
/// were never part of the word.
#[must_use]
pub fn fold_stored(src: &[u8], head: usize) -> Vec<u8> {
    folded(src, head, true)
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

    /// Bytes that are not a word at all are read as though they were, because
    /// that is what a real server does with them.
    #[test]
    fn a_broken_character_is_read_rather_than_refused() {
        assert_eq!(fold(b"zz\x80yy"), b"zz9y");
        assert_eq!(fold(b"aa\xffbb"), b"aa\xf7\xa2\xa2\x80");
        // The byte after the end of the word counts as a zero, and a character
        // that comes out as zero ends the word.
        assert_eq!(fold(b"zz\x80"), b"zz");
        // A byte on its own that reads as a character with nothing in it is a
        // character all the same, and this one is `À`.
        assert_eq!(fold(b"\xc3"), "à".as_bytes());
    }

    /// The head is how much of the buffer is the word, and the rest of it is
    /// only there to be read by a character that started inside the head and
    /// did not finish there.
    #[test]
    fn folding_a_word_reads_past_it_and_does_not_go_past_it() {
        assert_eq!(fold_stored(b"zz\x80 yy", 3), b"zz ");
        assert_eq!(fold_stored(b"zz\x80-yy", 3), b"zz-");
        assert_eq!(fold_stored(b"zzyy", 2), b"zz");
    }

    /// Every character an index stores is cut down to sixteen bits, so the ones
    /// above that come out as something else entirely.
    #[test]
    fn what_goes_into_an_index_is_cut_to_sixteen_bits() {
        let emoji = "😀".as_bytes();
        assert_eq!(fold_stored(emoji, emoji.len()), "\u{f600}".as_bytes());
        assert_eq!(fold(emoji), emoji);
        // Lowercased first and cut afterwards, which is why this Osage capital
        // comes out as the small letter and not as the capital.
        let osage = "𐒰".as_bytes();
        assert_eq!(fold_stored(osage, osage.len()), "\u{4d8}".as_bytes());
        assert_eq!(fold(osage), "𐓘".as_bytes());
    }

    #[test]
    fn a_first_byte_says_how_many_bytes_a_character_has() {
        assert_eq!(width(b'a'), 1);
        assert_eq!(width(0x80), 2);
        assert_eq!(width(0xc3), 2);
        assert_eq!(width(0xdf), 2);
        assert_eq!(width(0xe0), 3);
        assert_eq!(width(0xef), 3);
        assert_eq!(width(0xf0), 4);
        assert_eq!(width(0xff), 4);
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
