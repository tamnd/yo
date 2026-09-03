//! A sorted set: an element table for the score of a member, and a counted tree
//! for the rank of a score.
//!
//! Every sorted set command is one of two questions and this is why there are
//! two structures. `ZSCORE` and `ZADD` ask what a member's score is, which is a
//! probe by name and wants a hash table. `ZRANK`, `ZRANGE` and everything with a
//! range in it ask where a score sits among the others, which is order
//! statistics and wants a tree. Redis answers both out of a skiplist plus a
//! dictionary; the dictionary is the same answer as ours and the skiplist is
//! where `08` section 5's memory went.
//!
//! ```text
//!   small                       everything else
//! +------------------+   +-------------------------------------+
//! | listpack         |-->| element table   +   counted tree     |
//! | member, score,   |   | member -> score     rank -> row      |
//! | member, score... |   |                                      |
//! +------------------+   +-------------------------------------+
//!    to 128 members         one probe                one descent
//! ```
//!
//! # What the tree holds, and what it does not
//!
//! The tree holds row numbers, three bytes each, and nothing else. It does not
//! hold the score. A search asks it for a position and it asks the caller, on
//! each comparison, where the thing being looked for sits against the element in
//! a given row, and the caller answers by reading that row's score out of the
//! element table.
//!
//! That is the trade `Y14` asks for. A tree with the score beside the row would
//! finish a search without leaving the node it is in, and it would cost eleven
//! bytes an element rather than three, which is a fail however fast it is. What
//! it costs instead is that a descent touches a handful of rows that are not
//! next to each other, and the rows it touches on a zipfian draw, which is the
//! gate cell aki lost, are the hot ones that are in cache anyway.
//!
//! # Ties
//!
//! Order is by score and then by member, comparing the member bytes, which is
//! Redis's rule and the reason `ZRANGEBYLEX` works at all: give every member the
//! same score and the set is ordered by member alone. Two members never tie
//! completely, so every element has exactly one position and a search for one
//! lands on it rather than on a run.
//!
//! # Renumbering
//!
//! The element table is dense, so taking a row out of it moves the last row into
//! the hole, and one element that nobody asked about gets a new row number on
//! every removal. The tree is told through [`Rank::set_at`], which needs the
//! moved element's position, so a `ZREM` is two descents rather than one. The
//! alternative is holes in the element table, and then the uniform draw
//! `ZRANDMEMBER` wants has to retry until it lands on a live row, which is fine
//! at nine tenths full and unbounded on a set that has been drained.

use core::cmp::Ordering;
use core::ops::Range;

use yo_common::num::{DIGITS_MAX, DOUBLE_MAX, i64_digits, parse_f64, write_double};

use crate::elem::Elements;
use crate::frozen::{self, Broken};
use crate::listpack::{self, Listpack};
use crate::rank::Rank;
use crate::scan::Cursor;

/// A member: bytes as they lie, or an integer not yet formatted.
pub type Member<'a> = listpack::Entry<'a>;

/// The packed band, which is Redis's `ZSET_LISTPACK`.
const FORM_PACKED: u8 = 1;
/// The table and its tree, written out as members in rank order.
const FORM_MEMBERS: u8 = 2;

/// Where a sorted set stops being one packed blob.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// At this many members a sorted set stops being a listpack.
    pub max_listpack_entries: usize,
    /// A member longer than this cannot go in a listpack.
    pub max_listpack_value: usize,
}

impl Limits {
    /// Redis's defaults: 128 and 64.
    pub const DEFAULT: Limits = Limits {
        max_listpack_entries: 128,
        max_listpack_value: 64,
    };
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::DEFAULT
    }
}

/// Which of the two a sorted set is in, which is what `OBJECT ENCODING` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// One packed blob, walked linearly.
    Listpack,
    /// The element table and the tree.
    ///
    /// The name is Redis's and the structure is not. `12` section 2 records the
    /// divergence: `OBJECT ENCODING` returns a name clients test for, not a
    /// claim about what is underneath, and answering with a word no client has
    /// heard of breaks tools for nothing.
    Skiplist,
}

impl Encoding {
    /// The word `OBJECT ENCODING` replies with.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Listpack => "listpack",
            Encoding::Skiplist => "skiplist",
        }
    }
}

/// What an add did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Added {
    /// The member was not there and now is. This is what `ZADD` counts.
    New,
    /// The member was there with a different score.
    Changed,
    /// The member was there with this score already.
    Same,
    /// The element table is full, so nothing happened.
    Full,
}

/// One end of a score range.
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    /// The score itself, which may be infinite.
    pub at: f64,
    /// Whether the score itself is outside the range, which is `ZRANGEBYSCORE`'s
    /// parenthesis.
    pub open: bool,
}

impl Bound {
    /// A bound that includes its score.
    #[must_use]
    pub const fn closed(at: f64) -> Bound {
        Bound { at, open: false }
    }

    /// A bound that excludes its score.
    #[must_use]
    pub const fn open(at: f64) -> Bound {
        Bound { at, open: true }
    }
}

/// One end of a member range, for the commands that order by member alone.
#[derive(Debug, Clone, Copy)]
pub enum Lex<'a> {
    /// Before every member, which is `-`.
    Min,
    /// After every member, which is `+`.
    Max,
    /// This member and everything after it, which is `[`.
    Incl(&'a [u8]),
    /// Everything after this member, which is `(`.
    Excl(&'a [u8]),
}

/// The element table and the tree over it.
#[derive(Debug, Clone)]
struct Table {
    members: Elements<f64>,
    order: Rank,
}

/// The two representations.
#[derive(Debug, Clone)]
enum Body {
    /// Member, score, member, score, in order.
    Packed(Listpack),
    Table(Table),
}

/// A set of members, each with a score, ordered by score and then by member.
#[derive(Debug, Clone)]
pub struct Zset {
    body: Body,
}

impl Default for Zset {
    fn default() -> Self {
        Self::new()
    }
}

/// Scores compare the way Redis compares them, which is not the way `f64`
/// compares by default.
///
/// `total_cmp` puts negative zero below positive zero and Redis does not, and a
/// `NaN` never gets here because every command that takes a score refuses one
/// before this is reached.
#[inline]
fn cmp_score(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// The order the whole structure is in: score first, then the member bytes.
#[inline]
fn cmp_key(score: f64, member: &[u8], other_score: f64, other_member: &[u8]) -> Ordering {
    match cmp_score(score, other_score) {
        Ordering::Equal => member.cmp(other_member),
        other => other,
    }
}

/// The score an entry holds, whichever way the listpack stored it.
fn score_of(entry: Member<'_>) -> f64 {
    match entry {
        Member::Int(n) => n as f64,
        // A score that went in as text came from `push_double`, so it parses.
        // A listpack that did not come from here is checked when it is loaded.
        Member::Str(s) => parse_f64(s).unwrap_or(0.0),
    }
}

/// The bytes of an entry, which for an integer member are the caller's buffer.
fn bytes_of<'a>(entry: Member<'a>, digits: &'a mut [u8; DIGITS_MAX]) -> &'a [u8] {
    match entry {
        Member::Str(s) => s,
        Member::Int(n) => i64_digits(digits, n),
    }
}

impl Zset {
    /// An empty sorted set, in the packed band.
    #[must_use]
    pub fn new() -> Zset {
        Zset {
            body: Body::Packed(Listpack::new()),
        }
    }

    /// An empty sorted set with room for `hint` members.
    ///
    /// A caller that knows the count up front, which is every `RESTORE`, should
    /// not fill the packed band to its limit and then promote the lot into a
    /// table. That pays a scan for the member and a scan for the position on
    /// every one of the first hundred and twenty eight, and then throws the
    /// listpack away.
    ///
    /// The hint is only a hint. Being wrong about it costs a table with more
    /// room than it needed rather than anything incorrect, and a hint under the
    /// band limit still gets the band, because a sorted set that turns out to be
    /// small is the common one and it should stay packed.
    #[must_use]
    pub fn with_hint(hint: usize, limits: &Limits) -> Zset {
        if hint <= limits.max_listpack_entries {
            return Zset::new();
        }
        Zset {
            body: Body::Table(Table {
                members: Elements::with_capacity(hint),
                order: Rank::new(),
            }),
        }
    }

    /// Take a listpack that is already in this band's layout, if it really is.
    ///
    /// The payload a `RESTORE` hands over for a small sorted set is Redis's own
    /// `ZSET_LISTPACK`, which is byte for byte the layout this band uses, so the
    /// whole load can be the blob moving in rather than a member at a time. That
    /// is the same argument [`Zset::packed_bytes`] makes on the way out, run
    /// backwards.
    ///
    /// It is worth more coming in than going out. Adding a member at a time
    /// costs a scan to see whether the member is already there and a second scan
    /// to find where it belongs, both of them over everything added so far, so a
    /// hundred member sorted set took nine times as long to restore as a hundred
    /// field hash did.
    ///
    /// The blob comes back on refusal, so a caller that has to walk it after all
    /// does not have to parse it twice. Refusal covers a count or a member past
    /// the limits, and it covers a blob that is not in order. This band answers
    /// a rank query by position and nothing else, so a payload that says it is a
    /// sorted set while not being sorted has to be rebuilt rather than trusted.
    /// Checking that is one pass and it is the same pass that rules out
    /// duplicates, since strictly increasing means no two members compare equal
    /// on the score and then equal on the bytes.
    pub(crate) fn from_packed(lp: Listpack, limits: &Limits) -> Result<Zset, Listpack> {
        let n = lp.len();
        if n == 0 || !n.is_multiple_of(2) || n / 2 > limits.max_listpack_entries {
            return Err(lp);
        }
        let ok = {
            let mut walk = lp.iter();
            let mut prev: Option<(f64, Member<'_>)> = None;
            let mut before_buf = [0u8; DIGITS_MAX];
            let mut member_buf = [0u8; DIGITS_MAX];
            loop {
                let Some(member) = walk.next() else {
                    break true;
                };
                // The count is even, checked above, so there is always a score
                // behind a member.
                let Some(entry) = walk.next() else {
                    break false;
                };
                let score = match entry {
                    Member::Int(v) => v as f64,
                    // `score_of` reads a text score with `unwrap_or(0.0)`, which
                    // is right for a blob this band wrote and wrong for one that
                    // arrived over the wire, so it is checked here once rather
                    // than guessed at on every read after.
                    Member::Str(s) => match parse_f64(s) {
                        Some(v) => v,
                        None => break false,
                    },
                };
                let bytes = bytes_of(member, &mut member_buf);
                if bytes.len() > limits.max_listpack_value {
                    break false;
                }
                if let Some((before, was)) = prev {
                    let was = bytes_of(was, &mut before_buf);
                    if cmp_key(before, was, score, bytes) != Ordering::Less {
                        break false;
                    }
                }
                prev = Some((score, member));
            }
        };
        if ok {
            Ok(Zset {
                body: Body::Packed(lp),
            })
        } else {
            Err(lp)
        }
    }

    /// Write this sorted set out in a form a device can hold.
    ///
    /// The packed band goes out as its own bytes, and the table goes out as its
    /// members in rank order, each one followed by its score. Rank order is the
    /// point of writing it that way: the order is the expensive half of a sorted
    /// set, and a body written in order comes back without a single comparison.
    ///
    /// A score is eight raw bytes rather than the text a listpack holds, because
    /// the table has the double already and formatting it here only to parse it
    /// on the way back would cost two conversions for no saving.
    pub fn freeze(&self, out: &mut Vec<u8>) {
        match &self.body {
            Body::Packed(lp) => {
                out.push(FORM_PACKED);
                out.extend_from_slice(lp.as_bytes());
            }
            Body::Table(t) => {
                out.push(FORM_MEMBERS);
                frozen::put_uint(out, t.members.len() as u64);
                // Through the tree's own walk and not through `Zset::at`, which
                // would descend from the root for every rank and turn a pass
                // into a count times a depth.
                for row in t.order.iter_from(0) {
                    let Some((name, score)) = t.members.at(row as usize) else {
                        continue;
                    };
                    frozen::put_bytes(out, name);
                    frozen::put_f64(out, *score);
                }
            }
        }
    }

    /// Read back what [`Zset::freeze`] wrote.
    ///
    /// The band a sorted set left in is the band it comes back in, so a value
    /// that was quiet long enough to be moved out answers `OBJECT ENCODING` with
    /// the same word it answered before.
    pub fn thaw(bytes: &[u8]) -> Result<Zset, Broken> {
        let mut cut = frozen::Cut::new(bytes);
        match cut.byte()? {
            FORM_PACKED => Ok(Zset {
                body: Body::Packed(Listpack::from_bytes(cut.rest()).map_err(|_| Broken::Body)?),
            }),
            FORM_MEMBERS => {
                let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
                // A member costs a length byte and a score costs eight, so a
                // count larger than what is left cannot be honest and is not
                // worth an allocation.
                if n > cut.rest().len() {
                    return Err(Broken::Body);
                }
                let mut table = Table {
                    members: Elements::with_capacity(n),
                    order: Rank::new(),
                };
                for _ in 0..n {
                    let name = cut.bytes()?;
                    let score = cut.f64()?;
                    let row = table.members.len() as u32;
                    // A member twice over would leave the table one row short of
                    // the tree and every rank after it wrong, so a body that
                    // repeats one is refused rather than half built.
                    if !matches!(table.members.insert(name, score), Ok(None)) {
                        return Err(Broken::Body);
                    }
                    // Written in rank order, so every member goes on the end and
                    // the tree compares nothing.
                    table.order.insert_at(row as usize, row);
                }
                Ok(Zset {
                    body: Body::Table(table),
                })
            }
            _ => Err(Broken::Form),
        }
    }

    /// How many members are in here.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.body {
            Body::Packed(lp) => lp.len() / 2,
            Body::Table(t) => t.members.len(),
        }
    }

    /// Whether there are no members in here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Which representation this is on.
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match &self.body {
            Body::Packed(_) => Encoding::Listpack,
            Body::Table(_) => Encoding::Skiplist,
        }
    }

    /// The bytes behind a sorted set on the packed band, for `DUMP` to copy.
    ///
    /// Member and score alternate in here exactly as `ZSET_LISTPACK` wants them,
    /// because the band was built to Redis's layout, so the payload is these
    /// bytes with a length in front. `None` on the tree, where there is no blob
    /// and the members have to be walked.
    #[inline]
    pub(crate) fn packed_bytes(&self) -> Option<&[u8]> {
        match &self.body {
            Body::Packed(lp) => Some(lp.as_bytes()),
            Body::Table(_) => None,
        }
    }

    /// What this is holding on to, in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.body {
            Body::Packed(lp) => lp.byte_len(),
            Body::Table(t) => t.members.memory_bytes() + t.order.bytes(),
        }
    }

    /// The score of a member, or `None` if it is not in here.
    ///
    /// `ZSCORE`, and the first half of every `ZADD`.
    #[must_use]
    pub fn score(&self, member: &[u8]) -> Option<f64> {
        match &self.body {
            Body::Packed(lp) => {
                let at = lp.find(member, 2)?;
                lp.get(at + 1).map(score_of)
            }
            Body::Table(t) => t.members.get(member).copied(),
        }
    }

    /// Put a member in, or move one that is already there.
    pub fn add(&mut self, member: &[u8], score: f64, limits: &Limits) -> Added {
        if let Body::Packed(lp) = &mut self.body {
            if let Some(at) = lp.find(member, 2) {
                let old = lp.get(at + 1).map_or(0.0, score_of);
                if cmp_score(old, score) == Ordering::Equal {
                    return Added::Same;
                }
                lp.delete(at, 2);
                packed_insert(lp, member, score);
                return Added::Changed;
            }
            if lp.len() / 2 < limits.max_listpack_entries
                && member.len() <= limits.max_listpack_value
            {
                packed_insert(lp, member, score);
                return Added::New;
            }
            self.promote();
        }
        let Body::Table(t) = &mut self.body else {
            unreachable!("promoted above")
        };
        t.add(member, score)
    }

    /// Take a member out.
    ///
    /// `ZREM`, and the way `ZADD GT` gets rid of nothing at all.
    pub fn remove(&mut self, member: &[u8]) -> bool {
        match &mut self.body {
            Body::Packed(lp) => match lp.find(member, 2) {
                Some(at) => lp.delete(at, 2),
                None => false,
            },
            Body::Table(t) => {
                let Some(row) = t.members.index_of(member) else {
                    return false;
                };
                let score = *t.members.get(member).expect("just found");
                let rank = t.rank_of(row as u32, score, member);
                t.take(rank, row);
                true
            }
        }
    }

    /// Where a member sits, counting from the lowest score.
    ///
    /// `ZRANK`, and `ZREVRANK` by taking it from the length.
    #[must_use]
    pub fn rank(&self, member: &[u8]) -> Option<usize> {
        match &self.body {
            Body::Packed(lp) => lp.find(member, 2).map(|at| at / 2),
            Body::Table(t) => {
                let row = t.members.index_of(member)?;
                let score = *t.members.get(member)?;
                Some(t.rank_of(row as u32, score, member))
            }
        }
    }

    /// The member and score at a rank.
    ///
    /// The member is not copied. `ZPOPMIN` writes it into the reply and then
    /// calls [`Zset::remove_at`] with the same rank, which is why these are two
    /// methods and not one that hands back an owned name.
    #[must_use]
    pub fn at(&self, rank: usize) -> Option<(Member<'_>, f64)> {
        match &self.body {
            Body::Packed(lp) => {
                let member = lp.get(rank * 2)?;
                let score = lp.get(rank * 2 + 1).map(score_of)?;
                Some((member, score))
            }
            Body::Table(t) => {
                let row = t.order.row_at(rank)?;
                let (name, score) = t.members.at(row as usize)?;
                Some((Member::Str(name), *score))
            }
        }
    }

    /// Take out whatever is at a rank.
    pub fn remove_at(&mut self, rank: usize) -> bool {
        match &mut self.body {
            Body::Packed(lp) => lp.delete(rank * 2, 2),
            Body::Table(t) => {
                let Some(row) = t.order.row_at(rank) else {
                    return false;
                };
                t.take(rank, row as usize);
                true
            }
        }
    }

    /// A member by position in no particular order, for a uniform draw.
    ///
    /// `ZRANDMEMBER` wants any member with equal probability and does not care
    /// which, so on the table this reads a row straight out of the dense array
    /// rather than descending the tree for a rank nobody asked for.
    #[must_use]
    pub fn pick(&self, at: usize) -> Option<(Member<'_>, f64)> {
        match &self.body {
            Body::Packed(_) => self.at(at),
            Body::Table(t) => {
                let (name, score) = t.members.at(at)?;
                Some((Member::Str(name), *score))
            }
        }
    }

    /// Walk members in rank order, from a rank, for a count.
    ///
    /// Every range command comes through here after working out which ranks it
    /// wants, because a range by score and a range by member are the same walk
    /// once the two ends have been found.
    pub fn walk<F: FnMut(Member<'_>, f64)>(&self, from: usize, count: usize, rev: bool, mut f: F) {
        let len = self.len();
        if from >= len || count == 0 {
            return;
        }
        let count = count.min(if rev { from + 1 } else { len - from });
        match &self.body {
            Body::Packed(lp) => {
                for i in 0..count {
                    let at = if rev { from - i } else { from + i };
                    let (Some(m), Some(s)) = (lp.get(at * 2), lp.get(at * 2 + 1)) else {
                        return;
                    };
                    f(m, score_of(s));
                }
            }
            Body::Table(t) => {
                if rev {
                    for row in t.order.iter_back_from(from).take(count) {
                        let Some((name, score)) = t.members.at(row as usize) else {
                            return;
                        };
                        f(Member::Str(name), *score);
                    }
                } else {
                    for row in t.order.iter_from(from).take(count) {
                        let Some((name, score)) = t.members.at(row as usize) else {
                            return;
                        };
                        f(Member::Str(name), *score);
                    }
                }
            }
        }
    }

    /// The ranks whose scores fall inside a range.
    ///
    /// `ZRANGEBYSCORE`, `ZCOUNT` and `ZREMRANGEBYSCORE` are all this plus a walk
    /// or a count of what it returns. An empty range comes back as an empty one
    /// rather than as a pair that has to be checked by the caller.
    #[must_use]
    pub fn window_by_score(&self, min: Bound, max: Bound) -> Range<usize> {
        let start = self.seek(|score, _| {
            // Everything below the bottom of the range is behind us.
            let before = match cmp_score(score, min.at) {
                Ordering::Less => true,
                Ordering::Equal => min.open,
                Ordering::Greater => false,
            };
            if before {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        let end = self.seek(|score, _| {
            let inside = match cmp_score(score, max.at) {
                Ordering::Less => true,
                Ordering::Equal => !max.open,
                Ordering::Greater => false,
            };
            if inside {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });
        start..end.max(start)
    }

    /// The ranks whose members fall inside a range, ignoring scores.
    ///
    /// `ZRANGEBYLEX`, which is only meaningful when every member has the same
    /// score and is nonsense otherwise, exactly as it is in Redis.
    #[must_use]
    pub fn window_by_lex(&self, min: Lex<'_>, max: Lex<'_>) -> Range<usize> {
        let start = self.seek(|_, member| match min {
            Lex::Min => Ordering::Less,
            Lex::Max => Ordering::Greater,
            Lex::Incl(at) => {
                if member < at {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            Lex::Excl(at) => {
                if member <= at {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        });
        let end = self.seek(|_, member| match max {
            Lex::Min => Ordering::Less,
            Lex::Max => Ordering::Greater,
            Lex::Incl(at) => {
                if member <= at {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            Lex::Excl(at) => {
                if member < at {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        });
        start..end.max(start)
    }

    /// How many elements a probe leaves behind it.
    ///
    /// The probe is given a score and a member and says `Greater` while the
    /// thing being looked for is still ahead. On the table this is one descent
    /// and on a listpack it is a walk, which is the same thing at 128 members.
    fn seek<F: FnMut(f64, &[u8]) -> Ordering>(&self, mut probe: F) -> usize {
        match &self.body {
            Body::Packed(lp) => {
                let mut digits = [0u8; DIGITS_MAX];
                let mut at = 0;
                while let (Some(m), Some(s)) = (lp.get(at * 2), lp.get(at * 2 + 1)) {
                    let bytes = bytes_of(m, &mut digits);
                    if probe(score_of(s), bytes) != Ordering::Greater {
                        break;
                    }
                    at += 1;
                }
                at
            }
            Body::Table(t) => t.order.seek(|row| {
                let (name, score) = t.members.at(row as usize).expect("a row the tree holds");
                probe(*score, name)
            }),
        }
    }

    /// Walk members for `ZSCAN`, in whatever order the storage has them in.
    pub fn scan<F: FnMut(Member<'_>, f64)>(
        &self,
        cursor: Cursor,
        count: usize,
        mut f: F,
    ) -> Cursor {
        match &self.body {
            Body::Packed(lp) => {
                for at in 0..lp.len() / 2 {
                    let (Some(m), Some(s)) = (lp.get(at * 2), lp.get(at * 2 + 1)) else {
                        break;
                    };
                    f(m, score_of(s));
                }
                Cursor::END
            }
            Body::Table(t) => t.members.scan(cursor, count, |name, score| {
                f(Member::Str(name), *score);
            }),
        }
    }

    /// Build a sorted set out of a member to score table that is in no order.
    ///
    /// This is what the algebra next door hands back. `ZUNIONSTORE` works out
    /// every member's final score in a table that knows nothing about order,
    /// because a member appearing in a fourth input should be a hash probe and
    /// not a pair of tree descents, and then this puts the whole thing in order
    /// once at the end.
    ///
    /// The table is not read and copied, it is moved in and becomes the sorted
    /// set. Every member's bytes were written when the first input holding that
    /// member was walked and they are never touched again, which is the thing
    /// that makes a union of four large sets one pass over each of them and one
    /// sort, rather than a pass and a rebuild.
    ///
    /// Answers nothing for an empty table, because an empty sorted set does not
    /// exist and the caller's key should be deleted rather than made.
    #[must_use]
    pub fn from_elements(members: Elements<f64>, limits: &Limits) -> Option<Zset> {
        let n = members.len();
        if n == 0 {
            return None;
        }
        // Row numbers put in order, which is a sort of four byte integers with
        // an indirection in the comparison rather than a sort of the members
        // themselves. Nothing is moved but the numbers.
        let mut rows: Vec<u32> = (0..n as u32).collect();
        rows.sort_unstable_by(|&a, &b| {
            let (a_name, a_score) = members.at(a as usize).expect("in range");
            let (b_name, b_score) = members.at(b as usize).expect("in range");
            cmp_key(*a_score, a_name, *b_score, b_name)
        });
        // Small enough to stay packed, which is worth checking because a
        // `ZINTERSTORE` of two large sets very often produces a small one.
        let packable = n <= limits.max_listpack_entries
            && rows.iter().all(|&r| {
                members.at(r as usize).expect("in range").0.len() <= limits.max_listpack_value
            });
        if packable {
            let mut lp = Listpack::new();
            let mut score_buf = [0u8; DOUBLE_MAX];
            for &row in &rows {
                let (name, score) = members.at(row as usize).expect("in range");
                lp.push(name);
                lp.push(write_double(&mut score_buf, *score));
            }
            return Some(Zset {
                body: Body::Packed(lp),
            });
        }
        let mut order = Rank::new();
        // Already in order, so every row goes on the end and the tree never
        // compares anything.
        for (at, &row) in rows.iter().enumerate() {
            order.insert_at(at, row);
        }
        Some(Zset {
            body: Body::Table(Table { members, order }),
        })
    }

    /// Move to the table, which is one way and does not come back.
    fn promote(&mut self) {
        let Body::Packed(lp) = &self.body else { return };
        let n = lp.len() / 2;
        let mut table = Table {
            members: Elements::with_capacity(n.next_power_of_two().max(16)),
            order: Rank::new(),
        };
        // The listpack is already in order, so every member goes on the end of
        // the tree and no comparison is needed. That is what makes a promotion a
        // walk rather than 128 descents.
        //
        // Walked and not indexed. There is no offset table in a listpack, so
        // asking it for element `i` costs a walk from the front and asking it
        // for every element in turn costs the square of the count.
        let mut digits = [0u8; DIGITS_MAX];
        let mut steps = lp.iter();
        while let (Some(m), Some(s)) = (steps.next(), steps.next()) {
            let bytes = bytes_of(m, &mut digits);
            let row = table.members.len() as u32;
            if table.members.insert(bytes, score_of(s)).is_err() {
                break;
            }
            table.order.insert_at(row as usize, row);
        }
        self.body = Body::Table(table);
    }
}

/// Put a member and its score into a listpack at the position it belongs.
fn packed_insert(lp: &mut Listpack, member: &[u8], score: f64) {
    // On the stack, not in a `Vec`. A double is at most `DOUBLE_MAX` bytes and
    // this runs on every packed `ZADD` and `ZINCRBY`, so a fresh allocation here
    // is a malloc and a free per member added to a small sorted set.
    let mut score_buf = [0u8; DOUBLE_MAX];
    let text = write_double(&mut score_buf, score);
    let mut digits = [0u8; DIGITS_MAX];
    let mut at = 0;
    while let (Some(m), Some(s)) = (lp.get(at * 2), lp.get(at * 2 + 1)) {
        let bytes = bytes_of(m, &mut digits);
        if cmp_key(score, member, score_of(s), bytes) == Ordering::Less {
            break;
        }
        at += 1;
    }
    if at * 2 == lp.len() {
        lp.push(member);
        lp.push(text);
    } else {
        lp.insert(at * 2, member);
        lp.insert(at * 2 + 1, text);
    }
}

impl Table {
    /// Where an element sits, given everything already known about it.
    ///
    /// The key is unique, so the lower bound is the element itself and there is
    /// no run to walk past.
    fn rank_of(&self, row: u32, score: f64, member: &[u8]) -> usize {
        let members = &self.members;
        self.order.seek(|other| {
            if other == row {
                return Ordering::Equal;
            }
            let (name, at) = members.at(other as usize).expect("a row the tree holds");
            cmp_key(score, member, *at, name)
        })
    }

    fn add(&mut self, member: &[u8], score: f64) -> Added {
        if let Some(row) = self.members.index_of(member) {
            let old = *self.members.at(row).map_or(&0.0, |(_, s)| s);
            if cmp_score(old, score) == Ordering::Equal {
                return Added::Same;
            }
            // The member has not changed and the score has, so this is a move
            // rather than an add: out of the tree at the old rank, back in at
            // the new one, and the element table's row number is untouched.
            let was = self.rank_of(row as u32, old, member);
            self.order.remove_at(was);
            if let Some(at) = self.members.at_mut(row) {
                *at = score;
            }
            let now = self.rank_of(row as u32, score, member);
            self.order.insert_at(now, row as u32);
            return Added::Changed;
        }
        let row = self.members.len() as u32;
        if self.members.insert(member, score).is_err() {
            return Added::Full;
        }
        let at = self.rank_of(row, score, member);
        self.order.insert_at(at, row);
        Added::New
    }

    /// Take out the element at a rank, which is known to be in a row.
    fn take(&mut self, rank: usize, row: usize) {
        let last = self.members.len() - 1;
        // Where the row that is about to be renumbered sits, found before
        // anything moves, because afterwards its score is in a different row and
        // the tree still holds the old number.
        let moved = if last == row {
            None
        } else {
            let (name, score) = self.members.at(last).expect("the last row");
            // The name is borrowed from the element table and the search reads
            // the same table, so both borrows are shared and neither outlives
            // the call.
            let at = {
                let members = &self.members;
                let score = *score;
                self.order.seek(|other| {
                    if other as usize == last {
                        return Ordering::Equal;
                    }
                    let (other_name, other_score) =
                        members.at(other as usize).expect("a row the tree holds");
                    cmp_key(score, name, *other_score, other_name)
                })
            };
            Some(at)
        };
        self.order.remove_at(rank);
        self.members.remove_at(row);
        if let Some(at) = moved {
            // Everything above the hole shifted down by one when the row came
            // out of the tree.
            let at = if at > rank { at - 1 } else { at };
            self.order.set_at(at, row as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A listpack holding the member and score pairs given, in the order given.
    fn packed(pairs: &[(&[u8], &str)]) -> Listpack {
        let mut lp = Listpack::new();
        for (member, score) in pairs {
            lp.push(member);
            lp.push(score.as_bytes());
        }
        lp
    }

    /// A payload already in this band's layout is taken whole.
    ///
    /// This is the whole point of [`Zset::from_packed`]: a hundred member sorted
    /// set restored in 534 us by adding a member at a time and restores in
    /// 4.6 us by moving the blob in, because adding costs a scan for the member
    /// and a scan for the position on every one of them.
    #[test]
    fn a_payload_in_this_layout_is_taken_whole() {
        let lp = packed(&[(b"a", "1"), (b"b", "2"), (b"c", "2.5")]);
        let z = Zset::from_packed(lp, &Limits::DEFAULT).expect("in order and inside the limits");
        assert_eq!(z.encoding(), Encoding::Listpack);
        assert_eq!(z.len(), 3);
        assert_eq!(z.score(b"a"), Some(1.0));
        assert_eq!(z.score(b"c"), Some(2.5));
        assert_eq!(z.rank(b"b"), Some(1));
        assert_eq!(z.score(b"missing"), None);

        // A member that looks like a number goes into a listpack as a number and
        // not as its digits, so it is worth pinning that one is still found. A
        // blob from another server is full of these and the failure would be a
        // member that is in the set and cannot be looked up.
        let lp = packed(&[(b"10", "1"), (b"9", "2")]);
        let z = Zset::from_packed(lp, &Limits::DEFAULT).expect("sorted by score");
        assert_eq!(z.score(b"10"), Some(1.0));
        assert_eq!(z.score(b"9"), Some(2.0));
        assert_eq!(z.rank(b"9"), Some(1));
    }

    /// A blob this band cannot hold comes back rather than being taken on trust.
    ///
    /// The caller walks it after that, so none of these is an error, and getting
    /// any of them wrong would be: the band answers a rank query by position and
    /// nothing else, so a payload claiming to be sorted while not being sorted
    /// would answer `ZRANGE` with the wrong members and never say why.
    #[test]
    fn a_blob_this_band_cannot_hold_is_handed_back() {
        let small = Limits {
            max_listpack_entries: 4,
            max_listpack_value: 8,
        };
        for (why, lp) in [
            ("out of order by score", packed(&[(b"a", "2"), (b"b", "1")])),
            (
                "out of order by member",
                packed(&[(b"b", "1"), (b"a", "1")]),
            ),
            (
                "the same member twice",
                packed(&[(b"a", "1"), (b"a", "1"), (b"b", "2")]),
            ),
            ("a score that is not a number", packed(&[(b"a", "no")])),
            (
                "a member past the value limit",
                packed(&[(b"aaaaaaaaaa", "1")]),
            ),
            (
                "more members than the band takes",
                packed(&[
                    (b"a", "1"),
                    (b"b", "2"),
                    (b"c", "3"),
                    (b"d", "4"),
                    (b"e", "5"),
                ]),
            ),
            ("nothing in it at all", packed(&[])),
        ] {
            assert!(
                Zset::from_packed(lp, &small).is_err(),
                "{why} should be handed back"
            );
        }

        // An odd count has no last score, which is malformed rather than merely
        // outside the band.
        let mut odd = Listpack::new();
        odd.push(b"a");
        assert!(Zset::from_packed(odd, &Limits::DEFAULT).is_err());
    }

    /// A hint past the band starts on the table instead of filling the band up.
    #[test]
    fn a_hint_past_the_band_starts_on_the_table() {
        let big = Zset::with_hint(Limits::DEFAULT.max_listpack_entries + 1, &Limits::DEFAULT);
        assert_eq!(big.encoding(), Encoding::Skiplist);
        assert!(big.is_empty());

        // At the limit it is still packed, matching what the add path does.
        let small = Zset::with_hint(Limits::DEFAULT.max_listpack_entries, &Limits::DEFAULT);
        assert_eq!(small.encoding(), Encoding::Listpack);

        // And a hint is only a hint, so the table takes members like anything
        // else and reports them in order.
        let mut z = Zset::with_hint(1_000_000, &Limits::DEFAULT);
        z.add(b"b", 2.0, &Limits::DEFAULT);
        z.add(b"a", 1.0, &Limits::DEFAULT);
        assert_eq!(z.len(), 2);
        assert_eq!(z.rank(b"a"), Some(0));
        assert_eq!(z.rank(b"b"), Some(1));
    }

    /// What a sorted set actually costs per member, which is M4's exit gate and
    /// was an argument rather than a number until this was written.
    ///
    /// Run it with `cargo test -p yo-kv --release measure_bytes_per_entry --
    /// --ignored --nocapture`. It is ignored because a million members is not
    /// something every `cargo test` should pay for, and it prints rather than
    /// asserts because the number it prints is the thing being reported. The
    /// bound that guards against a regression is
    /// [`a_large_sorted_set_does_not_hold_much_more_than_it_stores`], which is
    /// small enough to run every time.
    #[test]
    #[ignore = "a measurement, run it by name"]
    fn measure_bytes_per_entry() {
        // The packed band first, at the largest size it is allowed to reach.
        let mut lp = Zset::new();
        let mut lp_payload = 0usize;
        for i in 0..128 {
            let m = format!("member:{i:09}");
            lp_payload += m.len();
            lp.add(m.as_bytes(), i as f64, &Limits::DEFAULT);
        }
        println!(
            "packed n=128 total={} payload={lp_payload} overhead_per_entry={:.2}",
            lp.memory_bytes(),
            (lp.memory_bytes() as f64 - lp_payload as f64) / 128.0
        );
        // And the table band, at four sizes, because the answer used to depend
        // on how close the count was to a power of two and that is exactly the
        // thing being fixed.
        for n in [10_000usize, 100_000, 600_000, 1_000_000] {
            let (z, payload) = filled(n);
            let total = z.memory_bytes();
            let scores = n * 8;
            let (slots, rows, names, tree) = match &z.body {
                Body::Table(t) => (
                    t.members.slot_bytes(),
                    t.members.row_bytes(),
                    t.members.name_bytes(),
                    t.order.bytes(),
                ),
                Body::Packed(_) => (0, 0, 0, 0),
            };
            let per = |b: usize| b as f64 / n as f64;
            println!(
                "table n={n} total={total} slots={:.2}/e rows={:.2}/e names={:.2}/e tree={:.2}/e overhead_per_entry={:.2}",
                per(slots),
                per(rows),
                per(names),
                per(tree),
                (total as f64 - payload as f64 - scores as f64) / n as f64
            );
        }
    }

    /// `n` members named `member:` and nine digits, so sixteen bytes each, and
    /// what those names weigh.
    fn filled(n: usize) -> (Zset, usize) {
        let mut z = Zset::new();
        let mut payload = 0usize;
        for i in 0..n {
            let m = format!("member:{i:09}");
            payload += m.len();
            z.add(m.as_bytes(), i as f64, &Limits::DEFAULT);
        }
        (z, payload)
    }

    /// The guard on the measurement above.
    ///
    /// Forty thousand members is a count nowhere near a power of two, which is
    /// the case that used to be worst: a row array and a name blob that had both
    /// just doubled held nearly twice what they were storing, and the slack was
    /// more than everything else in the structure put together. Both grow by a
    /// quarter now, so the bound below is one a doubling array cannot meet and
    /// this test fails if either of them goes back to `Vec`'s policy.
    #[test]
    fn a_large_sorted_set_does_not_hold_much_more_than_it_stores() {
        let n = 40_000usize;
        let (z, payload) = filled(n);
        let Body::Table(t) = &z.body else {
            panic!("forty thousand members is not a listpack");
        };
        // Twelve bytes of row and eight of score, so twenty five leaves room for
        // the growth policy's quarter and none for a payload that has crept back
        // inside the row and brought four bytes of padding with it.
        assert!(
            t.members.row_bytes() < n * 25,
            "the row and score arrays hold {} for {n} members",
            t.members.row_bytes()
        );
        assert!(
            t.members.name_bytes() < payload + payload / 4,
            "the name blob holds {} for {payload} bytes of names",
            t.members.name_bytes()
        );
        // The tree is the part that already meets the gate and the part most
        // likely to be quietly broken by a change to the element table, so it
        // is worth pinning: three and a bit bytes a member, which is the row
        // number plus its share of a branch node at a fanout of a hundred and
        // twenty eight.
        assert!(
            t.order.bytes() < n * 4,
            "the tree holds {} for {n} members",
            t.order.bytes()
        );
    }

    /// A set built by adding in whatever order, checked against a model.
    fn built(pairs: &[(&str, f64)], limits: &Limits) -> Zset {
        let mut z = Zset::new();
        for (m, s) in pairs {
            z.add(m.as_bytes(), *s, limits);
        }
        z
    }

    /// Every member and score in rank order.
    fn listed(z: &Zset) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        let mut digits = [0u8; DIGITS_MAX];
        z.walk(0, z.len(), false, |m, s| {
            let bytes = bytes_of(m, &mut digits).to_vec();
            out.push((String::from_utf8(bytes).unwrap(), s));
        });
        out
    }

    /// What the model says the order is.
    fn model(pairs: &[(&str, f64)]) -> Vec<(String, f64)> {
        let mut last: Vec<(String, f64)> = Vec::new();
        for (m, s) in pairs {
            match last.iter_mut().find(|(name, _)| name == m) {
                Some(row) => row.1 = *s,
                None => last.push(((*m).to_string(), *s)),
            }
        }
        last.sort_by(|a, b| cmp_key(a.1, a.0.as_bytes(), b.1, b.0.as_bytes()));
        last
    }

    const PACKED: Limits = Limits::DEFAULT;
    const TABLE: Limits = Limits {
        max_listpack_entries: 0,
        max_listpack_value: 64,
    };

    #[test]
    fn an_empty_set_answers_nothing() {
        let z = Zset::new();
        assert_eq!(z.len(), 0);
        assert!(z.is_empty());
        assert_eq!(z.encoding(), Encoding::Listpack);
        assert_eq!(z.score(b"nobody"), None);
        assert_eq!(z.rank(b"nobody"), None);
        assert!(z.at(0).is_none());
        assert_eq!(
            z.window_by_score(Bound::closed(f64::NEG_INFINITY), Bound::closed(0.0)),
            0..0
        );
    }

    #[test]
    fn both_bands_put_members_in_the_same_order() {
        let pairs = [("c", 3.0), ("a", 1.0), ("b", 2.0), ("d", 2.0), ("e", -1.5)];
        for limits in [&PACKED, &TABLE] {
            let z = built(&pairs, limits);
            assert_eq!(listed(&z), model(&pairs), "{:?}", z.encoding());
            assert_eq!(z.len(), 5);
        }
    }

    #[test]
    fn a_tie_on_score_is_broken_by_the_member() {
        for limits in [&PACKED, &TABLE] {
            let pairs = [
                ("beta", 1.0),
                ("alpha", 1.0),
                ("gamma", 1.0),
                ("Alpha", 1.0),
            ];
            let z = built(&pairs, limits);
            let names: Vec<String> = listed(&z).into_iter().map(|(m, _)| m).collect();
            assert_eq!(names, ["Alpha", "alpha", "beta", "gamma"]);
        }
    }

    #[test]
    fn adding_a_member_again_moves_it_rather_than_adding_it() {
        for limits in [&PACKED, &TABLE] {
            let mut z = built(&[("a", 1.0), ("b", 2.0), ("c", 3.0)], limits);
            assert_eq!(z.add(b"a", 1.0, limits), Added::Same);
            assert_eq!(z.add(b"a", 9.0, limits), Added::Changed);
            assert_eq!(z.len(), 3);
            assert_eq!(z.score(b"a"), Some(9.0));
            assert_eq!(z.rank(b"a"), Some(2));
            assert_eq!(listed(&z).last().unwrap().0, "a");
        }
    }

    #[test]
    fn a_removal_leaves_every_other_rank_right() {
        for limits in [&PACKED, &TABLE] {
            let pairs: Vec<(&str, f64)> =
                vec![("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0), ("e", 5.0)];
            let mut z = built(&pairs, limits);
            assert!(z.remove(b"c"));
            assert!(!z.remove(b"c"));
            assert_eq!(z.len(), 4);
            assert_eq!(z.rank(b"a"), Some(0));
            assert_eq!(z.rank(b"d"), Some(2));
            assert_eq!(z.rank(b"e"), Some(3));
            assert_eq!(z.score(b"c"), None);
        }
    }

    /// The dense element table moves the last row into the hole, so a removal
    /// renumbers an element nobody asked about. This is the test that fails if
    /// the tree is not told.
    #[test]
    fn removing_from_the_middle_renumbers_the_last_element() {
        let limits = &TABLE;
        let mut z = Zset::new();
        for i in 0..64u32 {
            z.add(format!("m{i:03}").as_bytes(), f64::from(i), limits);
        }
        // Take them out from the front, which moves the last row into row zero
        // every single time.
        for i in 0..63u32 {
            assert!(z.remove(format!("m{i:03}").as_bytes()));
            assert_eq!(z.len() as u32, 63 - i);
            // Every member still in here has to be findable, in the right place,
            // with the right score.
            for j in i + 1..64 {
                let name = format!("m{j:03}");
                assert_eq!(
                    z.score(name.as_bytes()),
                    Some(f64::from(j)),
                    "score of {name}"
                );
                assert_eq!(
                    z.rank(name.as_bytes()),
                    Some((j - i - 1) as usize),
                    "rank of {name} after {i}"
                );
            }
        }
    }

    #[test]
    fn a_set_promotes_when_it_outgrows_the_packed_band() {
        let limits = Limits {
            max_listpack_entries: 4,
            max_listpack_value: 64,
        };
        let mut z = Zset::new();
        for i in 0..4u32 {
            z.add(format!("m{i}").as_bytes(), f64::from(i), &limits);
        }
        assert_eq!(z.encoding(), Encoding::Listpack);
        z.add(b"m4", 4.0, &limits);
        assert_eq!(z.encoding(), Encoding::Skiplist);
        assert_eq!(z.len(), 5);
        let names: Vec<String> = listed(&z).into_iter().map(|(m, _)| m).collect();
        assert_eq!(names, ["m0", "m1", "m2", "m3", "m4"]);
        // A promotion is one way, and a set that shrinks stays where it is.
        z.remove(b"m4");
        z.remove(b"m3");
        assert_eq!(z.encoding(), Encoding::Skiplist);
    }

    #[test]
    fn a_member_too_long_for_the_packed_band_promotes_on_its_own() {
        let limits = Limits {
            max_listpack_entries: 128,
            max_listpack_value: 8,
        };
        let mut z = Zset::new();
        z.add(b"short", 1.0, &limits);
        assert_eq!(z.encoding(), Encoding::Listpack);
        z.add(b"a member well past eight bytes", 2.0, &limits);
        assert_eq!(z.encoding(), Encoding::Skiplist);
        assert_eq!(z.len(), 2);
        assert_eq!(z.score(b"a member well past eight bytes"), Some(2.0));
    }

    #[test]
    fn a_score_range_finds_both_of_its_ends() {
        for limits in [&PACKED, &TABLE] {
            let pairs = [("a", 1.0), ("b", 2.0), ("c", 2.0), ("d", 3.0), ("e", 4.0)];
            let z = built(&pairs, limits);
            assert_eq!(
                z.window_by_score(Bound::closed(2.0), Bound::closed(3.0)),
                1..4
            );
            assert_eq!(
                z.window_by_score(Bound::open(2.0), Bound::closed(3.0)),
                3..4
            );
            assert_eq!(
                z.window_by_score(Bound::closed(2.0), Bound::open(3.0)),
                1..3
            );
            assert_eq!(z.window_by_score(Bound::open(1.0), Bound::open(4.0)), 1..4);
            assert_eq!(
                z.window_by_score(
                    Bound::closed(f64::NEG_INFINITY),
                    Bound::closed(f64::INFINITY)
                ),
                0..5
            );
            // A range with nothing in it is empty and not backwards.
            assert_eq!(
                z.window_by_score(Bound::closed(3.0), Bound::closed(2.0)),
                3..3
            );
            assert_eq!(
                z.window_by_score(Bound::closed(9.0), Bound::closed(10.0)),
                5..5
            );
        }
    }

    #[test]
    fn a_member_range_orders_by_member_when_the_scores_are_equal() {
        for limits in [&PACKED, &TABLE] {
            let pairs = [("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0), ("e", 0.0)];
            let z = built(&pairs, limits);
            assert_eq!(z.window_by_lex(Lex::Min, Lex::Max), 0..5);
            assert_eq!(z.window_by_lex(Lex::Incl(b"b"), Lex::Incl(b"d")), 1..4);
            assert_eq!(z.window_by_lex(Lex::Excl(b"b"), Lex::Excl(b"d")), 2..3);
            assert_eq!(z.window_by_lex(Lex::Incl(b"b"), Lex::Excl(b"c")), 1..2);
            assert_eq!(z.window_by_lex(Lex::Excl(b"e"), Lex::Max), 5..5);
            assert_eq!(z.window_by_lex(Lex::Min, Lex::Excl(b"a")), 0..0);
        }
    }

    #[test]
    fn a_walk_can_go_backwards_and_stops_where_it_is_told() {
        for limits in [&PACKED, &TABLE] {
            let pairs = [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)];
            let z = built(&pairs, limits);
            let mut seen = Vec::new();
            let mut digits = [0u8; DIGITS_MAX];
            z.walk(3, 2, true, |m, _| {
                seen.push(String::from_utf8(bytes_of(m, &mut digits).to_vec()).unwrap());
            });
            assert_eq!(seen, ["d", "c"]);
            let mut seen = Vec::new();
            z.walk(1, 99, false, |m, _| {
                seen.push(String::from_utf8(bytes_of(m, &mut digits).to_vec()).unwrap());
            });
            assert_eq!(seen, ["b", "c", "d"]);
            // A rank past the end is nothing rather than a panic.
            let mut count = 0;
            z.walk(4, 1, false, |_, _| count += 1);
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn taking_from_a_rank_takes_the_right_one() {
        for limits in [&PACKED, &TABLE] {
            let pairs = [("a", 1.0), ("b", 2.0), ("c", 3.0)];
            let mut z = built(&pairs, limits);
            assert!(z.remove_at(0));
            assert_eq!(z.len(), 2);
            assert_eq!(z.rank(b"b"), Some(0));
            assert!(z.remove_at(1));
            assert_eq!(z.score(b"c"), None);
            assert!(!z.remove_at(5));
        }
    }

    /// The one that actually catches things: a few thousand adds, moves and
    /// removals against a `Vec` that says what the answer is.
    #[test]
    fn a_run_of_everything_agrees_with_a_model() {
        for limits in [
            &PACKED,
            &Limits {
                max_listpack_entries: 8,
                max_listpack_value: 64,
            },
        ] {
            let mut z = Zset::new();
            let mut model: Vec<(String, f64)> = Vec::new();
            let mut seed = 0x8765_4321_9ABC_DEF0u64;
            let mut roll = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for round in 0..3_000 {
                let name = format!("m{:02}", roll() % 40);
                // Scores land on a handful of values so that ties are the rule
                // and not an accident.
                let score = (roll() % 7) as f64 - 3.0;
                if round % 5 == 4 {
                    let hit = z.remove(name.as_bytes());
                    let was = model.iter().position(|(m, _)| *m == name);
                    assert_eq!(hit, was.is_some());
                    if let Some(at) = was {
                        model.remove(at);
                    }
                } else {
                    z.add(name.as_bytes(), score, limits);
                    match model.iter_mut().find(|(m, _)| *m == name) {
                        Some(row) => row.1 = score,
                        None => model.push((name, score)),
                    }
                }
                if round % 97 == 0 {
                    let mut want = model.clone();
                    want.sort_by(|a, b| cmp_key(a.1, a.0.as_bytes(), b.1, b.0.as_bytes()));
                    assert_eq!(listed(&z), want, "round {round}");
                    for (at, (m, s)) in want.iter().enumerate() {
                        assert_eq!(z.rank(m.as_bytes()), Some(at), "rank of {m}");
                        assert_eq!(z.score(m.as_bytes()), Some(*s), "score of {m}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_big_set_costs_what_the_tree_said_it_would() {
        let limits = &TABLE;
        let mut z = Zset::new();
        let n = 100_000u32;
        for i in 0..n {
            z.add(format!("member:{i:08}").as_bytes(), f64::from(i), limits);
        }
        assert_eq!(z.len(), n as usize);
        assert_eq!(z.rank(b"member:00050000"), Some(50_000));
        assert_eq!(z.at(0).map(|(_, s)| s), Some(0.0));
        assert_eq!(z.at(n as usize - 1).map(|(_, s)| s), Some(f64::from(n - 1)));
        // What the gate is about is the ordering structure, which is the tree,
        // and it holds three bytes an element and change. The rest is the
        // element table, which a hash and a set pay the same way: twenty four
        // bytes a row because a score is eight and forces the row to align to
        // eight, four more for the open addressed slot, and the member bytes
        // themselves. The row array and the slot array are both powers of two,
        // so at a hundred thousand members a third of both is slack.
        let Body::Table(t) = &z.body else {
            panic!("a hundred thousand members is not a listpack")
        };
        let per_order = t.order.bytes() as f64 / f64::from(n);
        assert!(per_order < 3.4, "{per_order} bytes an element in the tree");
        let per = z.memory_bytes() as f64 / f64::from(n);
        assert!(per < 70.0, "{per} bytes a member all in");
    }

    /// Freeze a sorted set, read it back, and check that nothing about it moved.
    fn round_trip(z: &Zset) -> Zset {
        let mut buf = Vec::new();
        z.freeze(&mut buf);
        let back = Zset::thaw(&buf).expect("what freeze wrote");
        assert_eq!(back.len(), z.len(), "the member count");
        assert_eq!(back.encoding(), z.encoding(), "the band");
        assert_eq!(listed(&back), listed(z), "the members in rank order");
        back
    }

    #[test]
    fn a_frozen_sorted_set_comes_back_in_the_band_it_left() {
        let pairs: Vec<(String, f64)> = (0..40)
            .map(|i| (format!("member:{i:04}"), f64::from(i) * 1.5 - 12.0))
            .collect();
        let refs: Vec<(&str, f64)> = pairs.iter().map(|(m, s)| (m.as_str(), *s)).collect();

        let packed = round_trip(&built(&refs, &PACKED));
        assert_eq!(packed.encoding(), Encoding::Listpack);
        let table = round_trip(&built(&refs, &TABLE));
        assert_eq!(table.encoding(), Encoding::Skiplist);

        // The two bands agree with each other and with the model, so a value
        // that was frozen on one and read on the other would have been caught.
        assert_eq!(listed(&packed), model(&refs));
        assert_eq!(listed(&table), model(&refs));

        round_trip(&Zset::new());
    }

    #[test]
    fn a_score_survives_the_trip_exactly() {
        // Scores that a text round trip through a listpack would be at risk of
        // rounding, plus the two that a zigzagged integer encoding would get
        // wrong. Every one of them has to come back bit for bit, because a
        // score is what the whole order is built on.
        let scores = [
            0.0,
            -0.0,
            1.0 / 3.0,
            -1.0 / 3.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            9_007_199_254_740_993.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut z = Zset::new();
        for (i, s) in scores.iter().enumerate() {
            z.add(format!("member:{i:04}").as_bytes(), *s, &TABLE);
        }
        let back = round_trip(&z);
        for (i, s) in scores.iter().enumerate() {
            let member = format!("member:{i:04}");
            assert_eq!(
                back.score(member.as_bytes()).map(f64::to_bits),
                Some(s.to_bits()),
                "the score of {member}"
            );
        }
    }

    #[test]
    fn a_sorted_set_that_came_back_still_takes_members_and_ranks_them() {
        let pairs: Vec<(String, f64)> = (0..200)
            .map(|i| (format!("member:{i:04}"), f64::from(i)))
            .collect();
        let refs: Vec<(&str, f64)> = pairs.iter().map(|(m, s)| (m.as_str(), *s)).collect();
        let mut back = round_trip(&built(&refs, &TABLE));

        // One in the middle, one at each end, and one that is already in here
        // and only moves.
        assert_eq!(back.add(b"middle", 99.5, &TABLE), Added::New);
        assert_eq!(back.add(b"first", -1.0, &TABLE), Added::New);
        assert_eq!(back.add(b"last", 1000.0, &TABLE), Added::New);
        assert_eq!(back.add(b"member:0000", 500.0, &TABLE), Added::Changed);

        assert_eq!(back.len(), 203);
        assert_eq!(back.rank(b"first"), Some(0));
        assert_eq!(back.rank(b"last"), Some(202));
        // `first`, then member:0001 through member:0099, because member:0000
        // moved up to five hundred and left the front.
        assert_eq!(back.rank(b"middle"), Some(100));
        assert_eq!(back.score(b"member:0000"), Some(500.0));
        assert!(back.remove(b"middle"));
        assert_eq!(back.rank(b"last"), Some(201));
    }

    #[test]
    fn a_frozen_sorted_set_that_arrives_damaged_is_an_error_and_not_a_panic() {
        let pairs: Vec<(String, f64)> = (0..200)
            .map(|i| (format!("member:{i:04}"), f64::from(i)))
            .collect();
        let refs: Vec<(&str, f64)> = pairs.iter().map(|(m, s)| (m.as_str(), *s)).collect();
        let z = built(&refs, &TABLE);
        let mut buf = Vec::new();
        z.freeze(&mut buf);

        assert!(Zset::thaw(&[]).is_err(), "nothing at all");
        assert!(Zset::thaw(&[99]).is_err(), "a form nobody wrote");
        for cut in 1..buf.len().min(64) {
            assert!(Zset::thaw(&buf[..cut]).is_err(), "cut at {cut}");
        }
        // A count that claims far more members than there are bytes behind it.
        let mut lying = vec![FORM_MEMBERS];
        frozen::put_uint(&mut lying, u64::MAX);
        assert!(Zset::thaw(&lying).is_err(), "a count nobody could hold");
        // The same member twice, which would leave the tree longer than the
        // table and every rank after it wrong.
        let mut twice = vec![FORM_MEMBERS];
        frozen::put_uint(&mut twice, 2);
        for _ in 0..2 {
            frozen::put_bytes(&mut twice, b"member");
            frozen::put_f64(&mut twice, 1.0);
        }
        assert!(Zset::thaw(&twice).is_err(), "a member written twice");
    }
}
