//! Redis's glob matching, which is what `KEYS`, `SCAN MATCH`, `CONFIG GET` and
//! `COMMAND LIST FILTERBY PATTERN` all mean by a pattern.
//!
//! It is not a regular expression and it is not the shell's glob either. There
//! are four things in it: `*` for any run of bytes, `?` for one byte, `[...]`
//! for a class, and a backslash for the byte after it. Everything else is a
//! literal, including the bytes that are special in a regular expression, so
//! `user.*` matches `user.1` and does not match `userX1`.
//!
//! Bytes rather than characters, the same as Redis. A pattern is matched
//! against a key, a key is arbitrary bytes, and deciding what a character is
//! would mean deciding what encoding a key is in, which nobody can do.

/// Whether `text` matches `pattern`.
///
/// The `*` case backtracks: on a mismatch the search goes back to the last star
/// and gives it one more byte. That is quadratic on a pattern built to be slow,
/// such as a run of stars against a long key, and it is what Redis does. The
/// difference is that Redis walks the whole keyspace with it and this walks a
/// list of settings, so the pathological case is not reachable from a client
/// until `KEYS` lands, at which point the cost is the keyspace scan and not
/// this.
#[must_use]
pub fn matches(pattern: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    // Where to go back to when a `*` has to give up a byte. `None` until the
    // first star, which is what makes a pattern without one a straight walk.
    let mut star: Option<usize> = None;
    let mut mark = 0usize;
    while t < text.len() {
        let mut step = false;
        if p < pattern.len() {
            match pattern[p] {
                b'*' => {
                    star = Some(p);
                    mark = t;
                    p += 1;
                    continue;
                }
                b'?' => {
                    p += 1;
                    t += 1;
                    continue;
                }
                b'[' => {
                    let (next, hit) = class(pattern, p, text[t]);
                    if hit {
                        p = next;
                        step = true;
                    }
                }
                b'\\' if p + 1 < pattern.len() => {
                    if pattern[p + 1] == text[t] {
                        p += 2;
                        step = true;
                    }
                }
                c => {
                    if c == text[t] {
                        p += 1;
                        step = true;
                    }
                }
            }
        }
        if step {
            t += 1;
            continue;
        }
        match star {
            Some(at) => {
                p = at + 1;
                mark += 1;
                t = mark;
            }
            None => return false,
        }
    }
    // Trailing stars match nothing, which is the one place a pattern is allowed
    // to be longer than what it matched.
    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

/// Match one byte against the class starting at `p`, which is a `[`.
///
/// Answers where the class ends and whether the byte belongs to it. A class
/// with no closing bracket ends at the end of the pattern rather than being an
/// error, which is Redis's reading and means a stray bracket in a key pattern
/// is never a refusal.
fn class(pattern: &[u8], p: usize, c: u8) -> (usize, bool) {
    let mut i = p + 1;
    let negate = i < pattern.len() && pattern[i] == b'^';
    if negate {
        i += 1;
    }
    let mut hit = false;
    while i < pattern.len() && pattern[i] != b']' {
        if pattern[i] == b'\\' && i + 1 < pattern.len() {
            i += 1;
            hit |= pattern[i] == c;
            i += 1;
        } else if i + 2 < pattern.len() && pattern[i + 1] == b'-' && pattern[i + 2] != b']' {
            let (mut lo, mut hi) = (pattern[i], pattern[i + 2]);
            if lo > hi {
                core::mem::swap(&mut lo, &mut hi);
            }
            hit |= c >= lo && c <= hi;
            i += 3;
        } else {
            hit |= pattern[i] == c;
            i += 1;
        }
    }
    let next = if i < pattern.len() { i + 1 } else { i };
    (next, hit != negate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pattern_without_a_wildcard_is_an_equality_test() {
        assert!(matches(b"maxmemory", b"maxmemory"));
        assert!(!matches(b"maxmemory", b"maxmemory-policy"));
        assert!(!matches(b"maxmemory-policy", b"maxmemory"));
        assert!(matches(b"", b""));
        assert!(!matches(b"", b"x"));
    }

    #[test]
    fn stars_match_any_run_including_none() {
        assert!(matches(b"*", b""));
        assert!(matches(b"*", b"anything at all"));
        assert!(matches(b"maxmemory*", b"maxmemory"));
        assert!(matches(b"maxmemory*", b"maxmemory-policy"));
        assert!(matches(b"*policy", b"maxmemory-policy"));
        assert!(matches(b"max*policy", b"maxmemory-policy"));
        assert!(!matches(b"max*policy", b"maxmemory-clients"));
        assert!(matches(b"a**b", b"ab"));
    }

    #[test]
    fn a_question_mark_is_exactly_one_byte() {
        assert!(matches(b"h?llo", b"hello"));
        assert!(!matches(b"h?llo", b"hllo"));
        assert!(!matches(b"h?llo", b"heello"));
    }

    #[test]
    fn classes_do_ranges_and_negation() {
        assert!(matches(b"h[ae]llo", b"hello"));
        assert!(matches(b"h[ae]llo", b"hallo"));
        assert!(!matches(b"h[ae]llo", b"hillo"));
        assert!(matches(b"h[^e]llo", b"hallo"));
        assert!(!matches(b"h[^e]llo", b"hello"));
        assert!(matches(b"key[0-9]", b"key7"));
        assert!(!matches(b"key[0-9]", b"keyx"));
        // A range the wrong way round is read as if it were the right way
        // round, which is what Redis does rather than matching nothing.
        assert!(matches(b"key[9-0]", b"key7"));
    }

    #[test]
    fn a_backslash_takes_the_special_out_of_the_next_byte() {
        assert!(matches(br"a\*b", b"a*b"));
        assert!(!matches(br"a\*b", b"axxb"));
        assert!(matches(br"a\\b", br"a\b"));
    }

    /// The pattern that would run away if the backtracking were wrong. It
    /// answers rather than hanging, which is the whole point of the test.
    #[test]
    fn a_pattern_built_to_be_slow_still_answers() {
        let pattern = b"*a*a*a*a*a*b";
        let text = vec![b'a'; 64];
        assert!(!matches(pattern, &text));
    }

    /// Not an error, and not a match for a bracket that is not there.
    #[test]
    fn a_class_that_is_never_closed_ends_at_the_end() {
        assert!(matches(b"a[bc", b"ab"));
        assert!(!matches(b"a[bc", b"ax"));
    }
}
