//! `LCS`, the longest common subsequence of two strings.
//!
//! This is the one string command that is not a probe and a store. It is a
//! dynamic program over a table of `(alen + 1) * (blen + 1)` counters, so it
//! costs the product of the two lengths in both time and memory, and Redis is
//! open about that in its own documentation. It is here because `LCS` is part of
//! the string group and 100 percent means 100 percent, not because it is a
//! command anybody should put on a hot path.
//!
//! The backtrack is a faithful port of Redis's, quirks included. There is a
//! branch in it that cannot be reached, because a range is always emitted at the
//! mismatch that precedes a non contiguous match, and it is kept anyway: a port
//! that quietly tidies up the original is a port that answers differently on
//! some input nobody thought of.
//!
//! The table is capped rather than left to take the machine down. Redis's own
//! guard is a failed allocation, which on a server that has overcommitted is a
//! kill rather than an error, and `LCS` on two large strings is the easiest
//! accidental denial of service in the string group.

use yo_common::{Code, Error, Result};

/// The largest table `LCS` will build, in entries.
///
/// Four bytes each, so this is 256 MiB of counters, which is two strings of
/// eight thousand bytes each. Redis has no explicit limit and fails on the
/// allocation instead. Ours is a number so that the failure is the same failure
/// on every machine rather than a function of how much memory happened to be
/// free.
pub const LCS_MAX_CELLS: usize = 64 * 1024 * 1024;

/// What Redis says when the table will not fit.
const NO_MEMORY: &str = "Insufficient memory, failed allocating transient memory for LCS";

/// One run of characters common to both strings, as `LCS IDX` reports it.
///
/// Both ends of both ranges are inclusive, which is Redis's convention here and
/// the same one `GETRANGE` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    /// Where the run sits in the first string, first and last byte.
    pub a: (u32, u32),
    /// Where the run sits in the second string, first and last byte.
    pub b: (u32, u32),
    /// How long the run is, which `WITHMATCHLEN` asks for.
    pub len: u32,
}

/// The answer to `LCS IDX`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Idx {
    /// The runs, in the order Redis emits them, which is from the end of the
    /// strings towards the start.
    pub matches: Vec<Match>,
    /// The length of the whole subsequence, which is not the sum of the runs
    /// when `MINMATCHLEN` has filtered some of them out.
    pub len: usize,
}

/// The table, and the two strings it was built from.
struct Table<'a> {
    cells: Vec<u32>,
    a: &'a [u8],
    b: &'a [u8],
}

impl<'a> Table<'a> {
    fn build(a: &'a [u8], b: &'a [u8]) -> Result<Table<'a>> {
        let (alen, blen) = (a.len(), b.len());
        let cells = alen
            .checked_add(1)
            .and_then(|r| blen.checked_add(1).and_then(|c| r.checked_mul(c)))
            .filter(|&n| n <= LCS_MAX_CELLS)
            .ok_or_else(|| Error::new(Code::Full, NO_MEMORY))?;

        let stride = blen + 1;
        let mut cells = vec![0u32; cells];
        for i in 1..=alen {
            for j in 1..=blen {
                let v = if a[i - 1] == b[j - 1] {
                    cells[(i - 1) * stride + (j - 1)] + 1
                } else {
                    cells[(i - 1) * stride + j].max(cells[i * stride + (j - 1)])
                };
                cells[i * stride + j] = v;
            }
        }
        Ok(Table { cells, a, b })
    }

    #[inline]
    fn at(&self, i: usize, j: usize) -> u32 {
        self.cells[i * (self.b.len() + 1) + j]
    }

    /// The length of the subsequence, which is the bottom right corner.
    #[inline]
    fn total(&self) -> usize {
        self.at(self.a.len(), self.b.len()) as usize
    }
}

/// The length of the longest common subsequence, which is `LCS ... LEN`.
///
/// No backtrack, so this is the table and nothing else.
pub fn len(a: &[u8], b: &[u8]) -> Result<usize> {
    Ok(Table::build(a, b)?.total())
}

/// The longest common subsequence itself, which is plain `LCS`.
pub fn string(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let t = Table::build(a, b)?;
    let mut out = vec![0u8; t.total()];
    walk(&t, 0, &mut out, &mut Vec::new());
    Ok(out)
}

/// Where the two strings agree, which is `LCS ... IDX`.
///
/// `minmatchlen` drops any run shorter than it, and zero keeps all of them.
/// `len` is still the length of the whole subsequence and not the length of what
/// survived the filter, which is what Redis reports and is worth knowing before
/// somebody tries to reconcile the two numbers.
pub fn idx(a: &[u8], b: &[u8], minmatchlen: u32) -> Result<Idx> {
    let t = Table::build(a, b)?;
    let mut matches = Vec::new();
    let mut sink = vec![0u8; t.total()];
    walk(&t, minmatchlen, &mut sink, &mut matches);
    Ok(Idx {
        matches,
        len: t.total(),
    })
}

/// Redis's backtrack, writing the subsequence into `out` and the runs into
/// `matches`.
///
/// Both outputs are filled on every call because the walk that produces one
/// produces the other for free, and `LCS IDX` and plain `LCS` differ only in
/// which one the caller looks at.
fn walk(t: &Table<'_>, minmatchlen: u32, out: &mut [u8], matches: &mut Vec<Match>) {
    let (alen, blen) = (t.a.len(), t.b.len());
    let (mut i, mut j) = (alen, blen);
    let mut idx = t.total();

    // `alen` in the start position is Redis's way of saying no range is open,
    // since a real start is always below it.
    let (mut a_start, mut a_end) = (alen, 0usize);
    let (mut b_start, mut b_end) = (0usize, 0usize);

    while i > 0 && j > 0 {
        let mut emit = false;
        if t.a[i - 1] == t.b[j - 1] {
            out[idx - 1] = t.a[i - 1];

            if a_start == alen {
                a_start = i - 1;
                a_end = i - 1;
                b_start = j - 1;
                b_end = j - 1;
            } else if a_start == i && b_start == j {
                // The run is contiguous, so it grows backwards.
                a_start -= 1;
                b_start -= 1;
            } else {
                emit = true;
            }
            // A run that has reached the front of either string is finished,
            // and so is the walk.
            if a_start == 0 || b_start == 0 {
                emit = true;
            }
            idx -= 1;
            i -= 1;
            j -= 1;
        } else {
            // Go whichever way the table says the subsequence came from.
            if t.at(i - 1, j) > t.at(i, j - 1) {
                i -= 1;
            } else {
                j -= 1;
            }
            if a_start != alen {
                emit = true;
            }
        }

        if emit {
            let run = (a_end - a_start + 1) as u32;
            if minmatchlen == 0 || run >= minmatchlen {
                matches.push(Match {
                    a: (a_start as u32, a_end as u32),
                    b: (b_start as u32, b_end as u32),
                    len: run,
                });
            }
            a_start = alen;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example from Redis's own LCS documentation, which is what everybody
    /// checks a new implementation against first.
    #[test]
    fn the_documented_example_comes_out_the_same() {
        let a = b"ohmytext";
        let b = b"mynewtext";
        assert_eq!(string(a, b).unwrap(), b"mytext");
        assert_eq!(len(a, b).unwrap(), 6);

        let got = idx(a, b, 0).unwrap();
        assert_eq!(got.len, 6);
        assert_eq!(
            got.matches,
            vec![
                Match {
                    a: (4, 7),
                    b: (5, 8),
                    len: 4
                },
                Match {
                    a: (2, 3),
                    b: (0, 1),
                    len: 2
                },
            ]
        );
    }

    #[test]
    fn minmatchlen_drops_the_short_runs_and_leaves_the_length_alone() {
        let got = idx(b"ohmytext", b"mynewtext", 4).unwrap();
        assert_eq!(got.matches.len(), 1);
        assert_eq!(got.matches[0].len, 4);
        // The length is the whole subsequence, not the sum of what survived.
        assert_eq!(got.len, 6);
    }

    #[test]
    fn an_empty_string_shares_nothing_with_anything() {
        assert_eq!(string(b"", b"abc").unwrap(), b"");
        assert_eq!(string(b"abc", b"").unwrap(), b"");
        assert_eq!(string(b"", b"").unwrap(), b"");
        assert_eq!(len(b"", b"abc").unwrap(), 0);
        assert!(idx(b"", b"abc", 0).unwrap().matches.is_empty());
    }

    #[test]
    fn two_identical_strings_are_one_run() {
        let got = idx(b"hello", b"hello", 0).unwrap();
        assert_eq!(got.len, 5);
        assert_eq!(
            got.matches,
            vec![Match {
                a: (0, 4),
                b: (0, 4),
                len: 5
            }]
        );
        assert_eq!(string(b"hello", b"hello").unwrap(), b"hello");
    }

    #[test]
    fn two_strings_with_nothing_in_common_share_nothing() {
        assert_eq!(string(b"abc", b"xyz").unwrap(), b"");
        assert_eq!(len(b"abc", b"xyz").unwrap(), 0);
        assert!(idx(b"abc", b"xyz", 0).unwrap().matches.is_empty());
    }

    #[test]
    fn a_run_of_one_is_still_a_run() {
        let got = idx(b"abc", b"axc", 0).unwrap();
        assert_eq!(got.len, 2);
        assert_eq!(
            got.matches,
            vec![
                Match {
                    a: (2, 2),
                    b: (2, 2),
                    len: 1
                },
                Match {
                    a: (0, 0),
                    b: (0, 0),
                    len: 1
                },
            ]
        );
        assert_eq!(string(b"abc", b"axc").unwrap(), b"ac");
    }

    #[test]
    fn every_run_lands_where_it_says_it_does() {
        // The ranges are the point of IDX, so check them against the strings
        // rather than against a number somebody wrote down.
        let a = &b"the quick brown fox"[..];
        let b = &b"a quick red fox jumps"[..];
        let got = idx(a, b, 0).unwrap();
        for m in &got.matches {
            let (s, e) = (m.a.0 as usize, m.a.1 as usize);
            let (t, u) = (m.b.0 as usize, m.b.1 as usize);
            assert_eq!(&a[s..=e], &b[t..=u], "{m:?} does not match");
            assert_eq!(m.len as usize, e - s + 1, "{m:?} has the wrong length");
        }
        // The runs concatenate back into the subsequence, once they are put the
        // right way round.
        let mut joined = Vec::new();
        for m in got.matches.iter().rev() {
            joined.extend_from_slice(&a[m.a.0 as usize..=m.a.1 as usize]);
        }
        assert_eq!(joined, string(a, b).unwrap());
    }

    #[test]
    fn a_table_that_will_not_fit_is_an_error_and_not_a_kill() {
        let big = vec![b'x'; LCS_MAX_CELLS];
        let e = len(&big, b"y").unwrap_err();
        assert_eq!(e.code(), Code::Full);
        assert_eq!(e.message(), NO_MEMORY);
    }
}
