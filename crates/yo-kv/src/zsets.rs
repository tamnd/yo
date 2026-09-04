//! The sorted set commands.
//!
//! One method per Redis command on [`Keyspace`], the same arrangement the string,
//! set, hash and list commands use and for the same reason: a key belongs to the
//! database and not to a type, so `ZADD` against a string has to be able to see
//! that it is a string. The sorted set itself, and the choice between its two
//! representations, is [`crate::zset`]. This file is what the wire and the
//! embedded API both call.
//!
//! # Every range command is the same command
//!
//! There are nine ways to ask a sorted set for a run of members. `ZRANGE` alone
//! has three, once `BYSCORE` and `BYLEX` are counted, and then `REV` doubles
//! them and `ZREVRANGE`, `ZREVRANGEBYSCORE` and `ZREVRANGEBYLEX` exist as older
//! spellings of the same thing. `ZRANGESTORE` is a tenth, and the three
//! `ZREMRANGE` forms are three more, and `ZCOUNT` and `ZLEXCOUNT` are two of
//! those with the walk left off.
//!
//! Writing fourteen of those separately is fourteen chances to get an exclusive
//! bound or a negative index wrong in one of them. So there is one [`Query`],
//! and every one of those commands is a `Query` turned into a [`Window`], which
//! is a rank, a count and a direction. What a command does with the window is
//! all that separates it from the others: walk it, count it, remove it, or walk
//! it into another key.
//!
//! Two calls and not one, because the wire needs the count before it needs the
//! members: a RESP array writes its length first, and a reply that collected the
//! members into a `Vec` in order to count them would allocate on the read path,
//! which is the thing `Y1` is about. So [`Keyspace::zwindow`] answers how many
//! there are and [`Keyspace::zwalk`] hands them over one at a time. The memo in
//! the keyspace means the second call does not resolve the key again.
//!
//! # The commands over more than one key
//!
//! `ZUNION`, `ZINTER`, `ZDIFF` and their three store forms all come through
//! [`Keyspace::zsetop`] or [`Keyspace::zsetop_store`], because the only thing
//! that separates them is which members survive and that is [`crate::zsetops`]'s
//! decision, not this file's. What is decided here is which keys they are
//! allowed to name. A plain set is a sorted set where every score is one, so
//! `ZUNIONSTORE d 2 zs plain` is legal and every input resolves through
//! `live_slot_either`. A key that is not there stays in place as
//! [`Operand::Missing`] rather than being dropped, because `WEIGHTS` is
//! positional and closing the gap would hand every later input the wrong
//! weight.
//!
//! `ZINTERCARD` is not `ZINTER` with the members thrown away, because it can
//! stop at its limit and never has to work out a single score.
//!
//! # Errors
//!
//! Every command here answers `WRONGTYPE` for a key holding something that is
//! not a sorted set, and treats a missing key as an empty one, which between
//! them cover every case because a key is a sorted set, or another type, or
//! absent. The commands over several keys resolve every key before they build
//! anything, so `ZUNIONSTORE d 2 z not-a-zset` leaves `d` alone rather than
//! finding out too late.

use yo_common::num::DIGITS_MAX;
use yo_common::{Code, Error, Result};

use crate::db::Db;
use crate::elem::Elements;
use crate::keyspace::Keyspace;
use crate::scan::Cursor;
use crate::strings;
use crate::value::{self, Kind};
use crate::zset::{Added, Bound, Lex, Limits, Member, Zset};
use crate::zsetops::{self, Aggregate, Op, Operand};

/// Which members a `ZADD` is allowed to touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gate {
    /// Add and update both. Plain `ZADD`.
    #[default]
    Always,
    /// Only add members that are not there. `ZADD NX`.
    IfMissing,
    /// Only update members that are there. `ZADD XX`.
    IfPresent,
}

/// Which way a `ZADD` is allowed to move a score it already has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Move {
    /// Either way. Plain `ZADD`.
    #[default]
    Any,
    /// Only up. `ZADD GT`.
    Up,
    /// Only down. `ZADD LT`.
    Down,
}

/// What a `ZADD` was asked to do.
///
/// `NX` with `GT` or `LT` is refused by the parser rather than here, because a
/// gate that only lets new members through and a rule about which way an
/// existing score may move cannot both apply to the same member and Redis calls
/// that a syntax error.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZAdd {
    /// `NX` or `XX`.
    pub gate: Gate,
    /// `GT` or `LT`.
    pub only: Move,
    /// `CH`, which counts changed scores as well as new members.
    pub changed: bool,
}

/// Which end of a sorted set a command works from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum From {
    /// The lowest score. `ZPOPMIN`, and `ZMPOP MIN`.
    Min,
    /// The highest score. `ZPOPMAX`, and `ZMPOP MAX`.
    Max,
}

/// What a range is measured in.
#[derive(Debug, Clone, Copy)]
pub enum By<'a> {
    /// Positions, where a negative one counts from the end. `ZRANGE k 0 -1`.
    Rank {
        /// The first position, inclusive.
        start: i64,
        /// The last position, inclusive.
        stop: i64,
    },
    /// Scores. `ZRANGEBYSCORE`, and `ZRANGE ... BYSCORE`.
    Score {
        /// The lowest score wanted.
        min: Bound,
        /// The highest score wanted.
        max: Bound,
    },
    /// Members, which is only meaningful when every score is the same.
    /// `ZRANGEBYLEX`, and `ZRANGE ... BYLEX`.
    Lex {
        /// The first member wanted.
        min: Lex<'a>,
        /// The last member wanted.
        max: Lex<'a>,
    },
}

/// A run of members, however it was asked for.
///
/// `min` and `max` are always the low end and the high end of the range itself,
/// whichever order the command wrote them in. `REV` reverses the walk, it does
/// not reverse the range, which is why `ZRANGEBYSCORE k 1 5` and
/// `ZREVRANGEBYSCORE k 5 1` cover the same members.
#[derive(Debug, Clone, Copy)]
pub struct Query<'a> {
    /// What the range is measured in.
    pub by: By<'a>,
    /// Walk from the high end down. `REV`, and the `ZREV` spellings.
    pub rev: bool,
    /// How many to skip once the range is found. `LIMIT`'s first number.
    pub offset: usize,
    /// How many to take, or all of them. `LIMIT`'s second number, where Redis's
    /// negative count means all.
    pub count: Option<usize>,
}

impl<'a> Query<'a> {
    /// A plain `ZRANGE key start stop`.
    #[must_use]
    pub const fn rank(start: i64, stop: i64) -> Query<'a> {
        Query {
            by: By::Rank { start, stop },
            rev: false,
            offset: 0,
            count: None,
        }
    }

    /// A plain `ZRANGEBYSCORE key min max`.
    #[must_use]
    pub const fn score(min: Bound, max: Bound) -> Query<'a> {
        Query {
            by: By::Score { min, max },
            rev: false,
            offset: 0,
            count: None,
        }
    }

    /// A plain `ZRANGEBYLEX key min max`.
    #[must_use]
    pub const fn lex(min: Lex<'a>, max: Lex<'a>) -> Query<'a> {
        Query {
            by: By::Lex { min, max },
            rev: false,
            offset: 0,
            count: None,
        }
    }

    /// The same query walked from the other end.
    #[must_use]
    pub const fn rev(mut self, rev: bool) -> Query<'a> {
        self.rev = rev;
        self
    }

    /// The same query with a `LIMIT` on it.
    #[must_use]
    pub const fn limit(mut self, offset: usize, count: Option<usize>) -> Query<'a> {
        self.offset = offset;
        self.count = count;
        self
    }
}

/// Where a run of members starts, how long it is, and which way it goes.
///
/// This is what every range command reduces to, and it is a rank rather than a
/// pair of members because the tree answers in ranks. `from` is the first member
/// the walk hands over, so on a reverse walk it is the high end and the walk
/// counts down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Window {
    /// The rank the walk starts at.
    pub from: usize,
    /// How many members the walk hands over.
    pub count: usize,
    /// Whether the walk counts down.
    pub rev: bool,
}

/// A score that is not a number, which no command may store.
fn nan() -> Error {
    Error::new(Code::Invalid, "resulting score is not a number (NaN)")
}

impl Keyspace {
    /// `ZADD key [NX|XX] [GT|LT] [CH] score member [score member ...]`.
    ///
    /// Answers how many members were added, or how many were added or changed if
    /// `CH` was given, which is the only thing `CH` does.
    ///
    /// The pairs arrive as an iterator and not a slice, the way `SADD`'s members
    /// do, because the wire layer has them as positions in the connection's read
    /// buffer and a slice would mean collecting them first. It is walked more
    /// than once, which is why it has to be `Clone`.
    pub fn zadd<'m, I>(&mut self, key: &[u8], pairs: I, opts: ZAdd) -> Result<usize>
    where
        I: Iterator<Item = (f64, &'m [u8])> + Clone,
    {
        for (score, m) in pairs.clone() {
            strings::check_len(key, m.len())?;
            if score.is_nan() {
                return Err(nan());
            }
        }
        let at = match self.zset_slot(key)? {
            Some(at) => at,
            None => {
                // A key is not created for a `ZADD XX` that cannot add anything,
                // and it is not created for an empty pair list either. Redis's
                // parser rejects `ZADD k` before it gets this far, but the
                // embedded API has no parser in front of it and an empty sorted
                // set left behind would be a key that exists and holds nothing.
                if opts.gate == Gate::IfPresent || pairs.clone().next().is_none() {
                    return Ok(0);
                }
                self.new_zset(key)
            }
        };
        let limits = self.zset_limits;
        let z = self
            .zsets
            .get_mut(at)
            .expect("the record points at its body");
        let mut added = 0usize;
        let mut changed = 0usize;
        for (score, m) in pairs {
            let Some(want) = gated(z, m, score, opts) else {
                continue;
            };
            // The element table being full means twenty four million members in
            // one key. Nothing was stored, so nothing is counted.
            if z.add(m, score, &limits) == Added::Full {
                continue;
            }
            match want {
                Added::New => added += 1,
                Added::Changed => changed += 1,
                _ => {}
            }
        }
        // A `ZADD XX` on a key that did not exist never got here, so the only
        // way to be holding an empty sorted set is a pair list that turned out
        // to be empty after the gate, and that key was ours to make.
        if z.is_empty() {
            self.drop_key(key);
        }
        Ok(if opts.changed { added + changed } else { added })
    }

    /// `ZADD key ... INCR score member`, and `ZINCRBY key increment member`.
    ///
    /// Answers the member's new score, or nothing at all if a gate refused it,
    /// which is the nil `ZADD INCR` replies with and is why this is one method
    /// rather than an `INCR` flag on [`Keyspace::zadd`] that would have to
    /// return two different shapes.
    ///
    /// `ZINCRBY` is this with no gate, where the answer is never nil.
    pub fn zincrby(
        &mut self,
        key: &[u8],
        member: &[u8],
        by: f64,
        opts: ZAdd,
    ) -> Result<Option<f64>> {
        strings::check_len(key, member.len())?;
        if by.is_nan() {
            return Err(nan());
        }
        let at = match self.zset_slot(key)? {
            Some(at) => at,
            None => {
                if opts.gate == Gate::IfPresent {
                    return Ok(None);
                }
                self.new_zset(key)
            }
        };
        let limits = self.zset_limits;
        let z = self
            .zsets
            .get_mut(at)
            .expect("the record points at its body");
        let now = z.score(member);
        let want = now.unwrap_or(0.0) + by;
        // Infinity plus its opposite. Redis refuses this and leaves the score
        // alone rather than storing a NaN nothing could ever compare against.
        if want.is_nan() {
            if z.is_empty() {
                self.drop_key(key);
            }
            return Err(nan());
        }
        let allowed = match (now, opts.gate, opts.only) {
            (Some(_), Gate::IfMissing, _) | (None, Gate::IfPresent, _) => false,
            (Some(was), _, Move::Up) => want > was,
            (Some(was), _, Move::Down) => want < was,
            _ => true,
        };
        if !allowed || z.add(member, want, &limits) == Added::Full {
            if z.is_empty() {
                self.drop_key(key);
            }
            return Ok(None);
        }
        Ok(Some(want))
    }

    /// `ZCARD key`.
    pub fn zcard(&mut self, key: &[u8]) -> Result<usize> {
        Ok(match self.zset_slot(key)? {
            Some(at) => self.zset_at(at).len(),
            None => 0,
        })
    }

    /// `ZSCORE key member`.
    pub fn zscore(&mut self, key: &[u8], member: &[u8]) -> Result<Option<f64>> {
        Ok(match self.zset_slot(key)? {
            Some(at) => self.zset_at(at).score(member),
            None => None,
        })
    }

    /// `ZMSCORE key member [member ...]`, which is `ZSCORE` in bulk.
    ///
    /// One key lookup for the whole call rather than one per member, which is
    /// the only reason the command exists.
    pub fn zmscore<'m>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]>,
        out: &mut Vec<Option<f64>>,
    ) -> Result<()> {
        out.clear();
        let Some(at) = self.zset_slot(key)? else {
            out.extend(members.map(|_| None));
            return Ok(());
        };
        let z = self.zset_at(at);
        out.extend(members.map(|m| z.score(m)));
        Ok(())
    }

    /// `ZREM key member [member ...]`. Answers how many were there.
    ///
    /// A sorted set that loses its last member loses its key too, because an
    /// empty sorted set does not exist in Redis.
    pub fn zrem<'m>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]>,
    ) -> Result<usize> {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(0);
        };
        let z = self
            .zsets
            .get_mut(at)
            .expect("the record points at its body");
        let mut gone = 0;
        for m in members {
            if z.remove(m) {
                gone += 1;
            }
        }
        if z.is_empty() {
            self.drop_key(key);
        }
        Ok(gone)
    }

    /// `ZRANK key member [WITHSCORE]`, and `ZREVRANK` with `rev` set.
    ///
    /// The score comes back whether it was asked for or not, because finding the
    /// rank already read it and handing it over costs nothing.
    pub fn zrank(&mut self, key: &[u8], member: &[u8], rev: bool) -> Result<Option<(usize, f64)>> {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(None);
        };
        let z = self.zset_at(at);
        let Some(rank) = z.rank(member) else {
            return Ok(None);
        };
        let score = z.score(member).unwrap_or(0.0);
        Ok(Some((if rev { z.len() - 1 - rank } else { rank }, score)))
    }

    /// How many members a query covers, and where they start.
    ///
    /// Every range command starts here. It is separate from [`Keyspace::zwalk`]
    /// because a RESP array writes its length before its members, and a reply
    /// that collected the members in order to count them would allocate on the
    /// read path.
    pub fn zwindow(&mut self, key: &[u8], q: &Query<'_>) -> Result<Window> {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(Window::default());
        };
        Ok(window(self.zset_at(at), q))
    }

    /// Hand over the members a window covers, in order, without collecting them.
    ///
    /// The window is the caller's rather than the query's, so that a caller that
    /// has already asked [`Keyspace::zwindow`] does not compute it twice, and so
    /// that `ZRANGESTORE` can walk a window it has already narrowed.
    pub fn zwalk<F>(&mut self, key: &[u8], w: Window, f: F) -> Result<()>
    where
        F: FnMut(Member<'_>, f64),
    {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(());
        };
        self.zset_at(at).walk(w.from, w.count, w.rev, f);
        Ok(())
    }

    /// `ZCOUNT`, `ZLEXCOUNT`, and the count half of any other range command.
    pub fn zcount(&mut self, key: &[u8], q: &Query<'_>) -> Result<usize> {
        Ok(self.zwindow(key, q)?.count)
    }

    /// `ZREMRANGEBYRANK`, `ZREMRANGEBYSCORE` and `ZREMRANGEBYLEX`, which differ
    /// only in what the query was measured in.
    ///
    /// Answers how many went. The window is taken out from its high end down, so
    /// that every rank still to be removed is the rank it was when the window
    /// was worked out.
    pub fn zremrange(&mut self, key: &[u8], q: &Query<'_>) -> Result<usize> {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(0);
        };
        let w = window(self.zset_at(at), q);
        let z = self
            .zsets
            .get_mut(at)
            .expect("the record points at its body");
        // A reverse window starts at its high end, so normalise to the low one
        // and then count down from the top of it either way.
        let low = if w.rev { w.from + 1 - w.count } else { w.from };
        for i in (0..w.count).rev() {
            z.remove_at(low + i);
        }
        if z.is_empty() {
            self.drop_key(key);
        }
        Ok(w.count)
    }

    /// `ZPOPMIN key [count]` and `ZPOPMAX key [count]`.
    ///
    /// Nothing is collected. The member at the end is handed to `f`, which
    /// writes it wherever it is going, and only then is it removed, which is why
    /// this does not allocate where `SPOP` has to.
    pub fn zpop<F>(&mut self, key: &[u8], end: From, count: usize, mut f: F) -> Result<usize>
    where
        F: FnMut(Member<'_>, f64),
    {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(0);
        };
        let z = self
            .zsets
            .get_mut(at)
            .expect("the record points at its body");
        let count = count.min(z.len());
        for _ in 0..count {
            // Always rank zero or the last rank, because taking one out moves
            // everything above it down and the next one to go is at the same
            // place again.
            let rank = if end == From::Min { 0 } else { z.len() - 1 };
            let Some((m, s)) = z.at(rank) else { break };
            f(m, s);
            z.remove_at(rank);
        }
        if z.is_empty() {
            self.drop_key(key);
        }
        Ok(count)
    }

    /// `ZPOPMIN key` and `ZPOPMAX key`, as an owned member for a caller that has
    /// nowhere to write it yet.
    ///
    /// This is what `BZPOPMIN` needs: a worker that has been parked has no reply
    /// buffer open at the moment the member becomes available, so this one has
    /// to allocate where [`Keyspace::zpop`] does not.
    pub fn zpop_one(&mut self, key: &[u8], end: From) -> Result<Option<(Vec<u8>, f64)>> {
        let mut got = None;
        let mut name = [0u8; yo_common::num::DIGITS_MAX];
        self.zpop(key, end, 1, |m, s| {
            got = Some((member_bytes(m, &mut name).to_vec(), s));
        })?;
        Ok(got)
    }

    /// `ZRANDMEMBER key [count]`.
    ///
    /// A positive count draws without replacement and answers at most as many as
    /// there are, and a negative one draws with replacement and answers exactly
    /// as many as asked for, which is Redis's rule and is why the count is
    /// signed here rather than paired with a flag.
    ///
    /// The draw without replacement is a partial shuffle of the row numbers and
    /// not a retry loop, because a retry loop on a count near the size of the set
    /// spends most of its time drawing members it already has.
    pub fn zrandmember<F>(&mut self, key: &[u8], count: i64, mut f: F) -> Result<usize>
    where
        F: FnMut(Member<'_>, f64),
    {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(0);
        };
        let len = self.zset_at(at).len();
        if len == 0 || count == 0 {
            return Ok(0);
        }
        if count < 0 {
            let want = count.unsigned_abs() as usize;
            for _ in 0..want {
                let pick = self.rng.below(len);
                let Some((m, s)) = self.zset_at(at).pick(pick) else {
                    break;
                };
                f(m, s);
            }
            return Ok(want);
        }
        let want = (count as usize).min(len);
        // Whole set, in storage order, which is what Redis does for a count at
        // or over the size and which saves shuffling in order to hand back
        // everything anyway.
        if want == len {
            for i in 0..len {
                let Some((m, s)) = self.zset_at(at).pick(i) else {
                    break;
                };
                f(m, s);
            }
            return Ok(len);
        }
        // The database's index buffer rather than a fresh `Vec`, for the reason
        // the byte scratch exists: a partial shuffle needs somewhere to hold the
        // permutation while it draws from it, and building that somewhere out of
        // the allocator on every `ZRANDMEMBER` is a malloc and a free on a
        // command a sampler sends in a loop. Taken out and put back, so an early
        // return leaves it as it was found.
        // `yo_alloc::high_water` because this is the buffer reaching a size it
        // has not been asked for before, which happens once per largest sorted
        // set the database has been sampled from and never again.
        let mut rows = std::mem::take(&mut self.rows);
        rows.clear();
        yo_alloc::high_water(|| rows.extend(0..len));
        for i in 0..want {
            let pick = i + self.rng.below(len - i);
            rows.swap(i, pick);
            let Some((m, s)) = self.zset_at(at).pick(rows[i]) else {
                break;
            };
            f(m, s);
        }
        self.rows = rows;
        Ok(want)
    }

    /// `ZSCAN key cursor [COUNT count]`.
    ///
    /// A small sorted set comes back whole with a cursor of [`Cursor::END`], the
    /// same guarantee `SSCAN` and `HSCAN` give, because a listpack has no stable
    /// position to resume from and 128 members is not worth a resume.
    pub fn zscan<F>(&mut self, key: &[u8], cursor: Cursor, count: usize, f: F) -> Result<Cursor>
    where
        F: FnMut(Member<'_>, f64),
    {
        let Some(at) = self.zset_slot(key)? else {
            return Ok(Cursor::END);
        };
        Ok(self.zset_at(at).scan(cursor, count, f))
    }

    /// `ZUNION`, `ZINTER` and `ZDIFF`, which differ only in `op`.
    ///
    /// The members come out in rank order, which means the result has to be put
    /// in order before any of it can be handed over, and that is what the return
    /// value's ordering costs. `ZINTERCARD` exists precisely because counting
    /// does not need any of that, and it does not come through here.
    pub fn zsetop<'k, F>(
        &mut self,
        op: Op,
        keys: impl Iterator<Item = &'k [u8]>,
        weights: &[f64],
        agg: Aggregate,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(Member<'_>, f64),
    {
        let slots = self.operand_slots(keys)?;
        let got = zsetops::gather(op, &self.operands_of(&slots), weights, agg);
        let limits = self.zset_limits;
        let Some(z) = Zset::from_elements(got, &limits) else {
            return Ok(0);
        };
        let len = z.len();
        z.walk(0, len, false, f);
        Ok(len)
    }

    /// `ZUNIONSTORE`, `ZINTERSTORE` and `ZDIFFSTORE`.
    ///
    /// The destination is allowed to be one of the sources, which is safe for
    /// the reason `SINTERSTORE` gives: the result is built whole before the
    /// destination is touched, so nothing here writes over a body that is still
    /// being read.
    pub fn zsetop_store<'k>(
        &mut self,
        op: Op,
        destination: &[u8],
        keys: impl Iterator<Item = &'k [u8]>,
        weights: &[f64],
        agg: Aggregate,
    ) -> Result<usize> {
        let slots = self.operand_slots(keys)?;
        let got = zsetops::gather(op, &self.operands_of(&slots), weights, agg);
        let limits = self.zset_limits;
        Ok(self.put_zset(destination, Zset::from_elements(got, &limits)))
    }

    /// `ZINTERCARD numkeys key [key ...] [LIMIT limit]`.
    ///
    /// Nothing is stored and no score is worked out, and a limit stops the walk
    /// as soon as it is reached, which is the only reason this is not `ZINTER`
    /// with the members thrown away.
    pub fn zintercard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
    ) -> Result<usize> {
        let slots = self.operand_slots(keys)?;
        Ok(zsetops::intercard(&self.operands_of(&slots), limit))
    }

    /// `ZRANGESTORE destination source <the arguments of ZRANGE>`.
    ///
    /// The window is copied rather than moved, because the destination may be
    /// the source and because the source keeps its members either way. An empty
    /// window deletes the destination, which is what `ZRANGESTORE d s 5 1` does
    /// and is the same rule every store form follows.
    pub fn zrangestore(
        &mut self,
        destination: &[u8],
        source: &[u8],
        q: &Query<'_>,
    ) -> Result<usize> {
        let built = match self.zset_slot(source)? {
            None => None,
            Some(at) => {
                let z = self.zset_at(at);
                let w = window(z, q);
                let mut got = Elements::with_capacity(w.count.max(16));
                let mut digits = [0u8; DIGITS_MAX];
                // In rank order whichever way the query walked, because the
                // destination is a sorted set and puts them in score order
                // regardless. `REV` decides which members are taken, not what
                // order they end up in.
                let from = if w.rev { w.from + 1 - w.count } else { w.from };
                z.walk(from, w.count, false, |m, s| {
                    let _ = got.insert(member_bytes(m, &mut digits), s);
                });
                let limits = self.zset_limits;
                Zset::from_elements(got, &limits)
            }
        };
        Ok(self.put_zset(destination, built))
    }

    /// Where each key is and what type it holds, keeping the ones that are not
    /// there in place.
    ///
    /// In place and not dropped, because `WEIGHTS` is positional: a missing
    /// second key still has a second weight, and closing the gap would hand
    /// every later input the wrong one.
    fn operand_slots<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<Vec<Option<(Kind, u32)>>> {
        let mut out = Vec::with_capacity(keys.size_hint().0);
        for key in keys {
            out.push(self.live_slot_either(key, Kind::Zset, Kind::Set)?);
        }
        Ok(out)
    }

    /// The bodies those slots point at, as things the algebra can ask questions
    /// of.
    fn operands_of(&self, slots: &[Option<(Kind, u32)>]) -> Vec<Operand<'_>> {
        slots
            .iter()
            .map(|got| match got {
                Some((Kind::Zset, at)) => Operand::Zset(self.zset_at(*at)),
                Some((Kind::Set, at)) => {
                    Operand::Set(self.sets.get(*at).expect("the record points at its body"))
                }
                _ => Operand::Missing,
            })
            .collect()
    }

    /// Put a sorted set under `key`, replacing whatever was there.
    ///
    /// Nothing means delete the key, because an empty sorted set does not exist.
    /// That is what makes `ZINTERSTORE d a b` with an empty intersection delete
    /// `d` and answer zero rather than leave something `EXISTS` says one for.
    ///
    /// Whatever the key held is freed first, through the one funnel, and any
    /// deadline it had goes with it, because the value under the key is not the
    /// value the expiry was set on.
    pub(crate) fn put_zset(&mut self, key: &[u8], z: Option<Zset>) -> usize {
        let Some(z) = z else {
            self.drop_key(key);
            return 0;
        };
        self.free_body(key);
        let len = z.len();
        let at = self.zsets.insert(z);
        let record = value::slot_record_len(false);
        self.write_rec(key, record, |out| {
            value::write_slot_record(out, Kind::Zset, at, None);
        });
        self.bodies += 1;
        len
    }

    /// The slot `key`'s sorted set is in, or `None` if there is no such key.
    #[inline]
    pub(crate) fn zset_slot(&mut self, key: &[u8]) -> Result<Option<u32>> {
        self.live_slot(key, Kind::Zset)
    }

    /// The body in a slot the record pointed at.
    ///
    /// Panicking here means a record outlived its body, which is the bug the
    /// two invariants in [`crate::sets`] are there to make impossible.
    #[inline]
    pub(crate) fn zset_at(&self, at: u32) -> &Zset {
        self.zsets.get(at).expect("the record points at its body")
    }

    /// Make an empty sorted set under `key` and answer which slot it went in.
    ///
    /// No hint, unlike a set. A sorted set starts packed whatever is going into
    /// it, and the first `ZADD` that crosses either threshold promotes it, so
    /// counting the pairs in advance would only move the same work earlier.
    fn new_zset(&mut self, key: &[u8]) -> u32 {
        // The body and, every so often, the slab that holds it. See
        // `yo_alloc::first_touch` for why this is the one allocation a command
        // is allowed to make.
        let at = yo_alloc::first_touch(|| self.zsets.insert(Zset::new()));
        let len = value::slot_record_len(false);
        self.write_rec(key, len, |out| {
            value::write_slot_record(out, Kind::Zset, at, None);
        });
        self.bodies += 1;
        at
    }
}

/// Where an input of the algebra lives on a striped database: which stripe it is
/// on, what it is holding, and the slot the body is in.
type Home = (usize, Kind, u32);

impl Db {
    /// `ZUNION`, `ZINTER` and `ZDIFF` over a database of any width.
    ///
    /// The keys are asked whether they share a stripe before anything else, and
    /// when they do the whole command is handed to that stripe. A width one
    /// database always takes that path and so does a hash tagged group on a wide
    /// one, so only keys that are genuinely spread out pay for the two passes
    /// below.
    pub fn zsetop<'k, F>(
        &mut self,
        op: Op,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        weights: &[f64],
        agg: Aggregate,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(Member<'_>, f64),
    {
        if let Some(home) = self.one_stripe(keys.clone()) {
            return self.stripe_mut(home).zsetop(op, keys, weights, agg, f);
        }
        let slots = self.operand_slots(keys)?;
        let got = zsetops::gather(op, &self.operands_of(&slots), weights, agg);
        let limits = self.zset_limits();
        let Some(z) = Zset::from_elements(got, &limits) else {
            return Ok(0);
        };
        let len = z.len();
        z.walk(0, len, false, f);
        Ok(len)
    }

    /// `ZUNIONSTORE`, `ZINTERSTORE` and `ZDIFFSTORE`.
    ///
    /// The destination is allowed to be one of the sources here too, and for the
    /// same reason: the whole result is built before the destination is touched,
    /// so no body is written over while it is still being read, whichever stripe
    /// it is on.
    pub fn zsetop_store<'k>(
        &mut self,
        op: Op,
        destination: &'k [u8],
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        weights: &[f64],
        agg: Aggregate,
    ) -> Result<usize> {
        if let Some(home) = self.one_stripe(std::iter::once(destination).chain(keys.clone())) {
            return self
                .stripe_mut(home)
                .zsetop_store(op, destination, keys, weights, agg);
        }
        let slots = self.operand_slots(keys)?;
        let got = zsetops::gather(op, &self.operands_of(&slots), weights, agg);
        // The destination's stripe's limits, since that is where the result is
        // going to live.
        let limits = self.at_ref(destination).zset_limits;
        let built = Zset::from_elements(got, &limits);
        Ok(self.at(destination).put_zset(destination, built))
    }

    /// `ZINTERCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn zintercard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
    ) -> Result<usize> {
        if let Some(home) = self.one_stripe(keys.clone()) {
            return self.stripe_mut(home).zintercard(keys, limit);
        }
        let slots = self.operand_slots(keys)?;
        Ok(zsetops::intercard(&self.operands_of(&slots), limit))
    }

    /// `ZRANGESTORE destination source <the arguments of ZRANGE>`.
    ///
    /// Two keys and one window, so when they are on different stripes the window
    /// is walked out of the source's stripe into a table of its own and the
    /// sorted set that comes of it is put on the destination's. The source keeps
    /// its members either way, which is what makes the copy the right shape even
    /// when the two keys are the same key.
    pub fn zrangestore(
        &mut self,
        destination: &[u8],
        source: &[u8],
        q: &Query<'_>,
    ) -> Result<usize> {
        let (onto, home) = (self.stripe_of(destination), self.stripe_of(source));
        if onto == home {
            return self.stripe_mut(onto).zrangestore(destination, source, q);
        }
        let built = match self.stripe_mut(home).zset_slot(source)? {
            None => None,
            Some(at) => {
                let z = self.stripe(home).zset_at(at);
                let w = window(z, q);
                let mut got = Elements::with_capacity(w.count.max(16));
                let mut digits = [0u8; DIGITS_MAX];
                let from = if w.rev { w.from + 1 - w.count } else { w.from };
                z.walk(from, w.count, false, |m, s| {
                    let _ = got.insert(member_bytes(m, &mut digits), s);
                });
                let limits = self.stripe(onto).zset_limits;
                Zset::from_elements(got, &limits)
            }
        };
        Ok(self.stripe_mut(onto).put_zset(destination, built))
    }

    /// Reap and resolve every input key, in order, to the stripe and slot its
    /// body is in.
    ///
    /// As [`Keyspace::operand_slots`], down to keeping a key that is not there in
    /// place rather than dropping it, because `WEIGHTS` is positional. Each key
    /// is resolved on its own stripe, which is the only difference, and it has to
    /// happen before any body is read because the reap wants the stripe mutably.
    fn operand_slots<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<Vec<Option<Home>>> {
        let mut out = Vec::with_capacity(keys.size_hint().0);
        for key in keys {
            let stripe = self.stripe_of(key);
            let got = self
                .stripe_mut(stripe)
                .live_slot_either(key, Kind::Zset, Kind::Set)?;
            out.push(got.map(|(kind, at)| (stripe, kind, at)));
        }
        Ok(out)
    }

    /// The bodies those slots point at, as things the algebra can ask questions
    /// of.
    ///
    /// Every stripe an input is on is borrowed at once, which needs only a shared
    /// borrow and is why the resolving above is a pass of its own.
    fn operands_of(&self, slots: &[Option<Home>]) -> Vec<Operand<'_>> {
        slots
            .iter()
            .map(|got| match got {
                Some((stripe, Kind::Zset, at)) => Operand::Zset(self.stripe(*stripe).zset_at(*at)),
                Some((stripe, Kind::Set, at)) => Operand::Set(
                    self.stripe(*stripe)
                        .sets
                        .get(*at)
                        .expect("the record points at its body"),
                ),
                _ => Operand::Missing,
            })
            .collect()
    }

    /// The limits a result that belongs to no key is built under.
    ///
    /// Stripe zero's, because the two thresholds a sorted set is promoted at are
    /// a setting of the database rather than of one stripe and every stripe
    /// carries the same pair.
    fn zset_limits(&self) -> Limits {
        self.stripe(0).zset_limits
    }
}

/// What a `ZADD` of one pair would do, or nothing if a gate refuses it.
///
/// This is worked out before the add rather than from what the add answered,
/// because `GT` and `LT` have to see the old score to know whether the new one
/// is allowed at all, and by then the add has already stored it.
fn gated(z: &Zset, member: &[u8], score: f64, opts: ZAdd) -> Option<Added> {
    match z.score(member) {
        None => match opts.gate {
            Gate::IfPresent => None,
            // A new member is added whatever `GT` or `LT` say, because there is
            // no old score for them to be about.
            _ => Some(Added::New),
        },
        Some(was) => {
            if opts.gate == Gate::IfMissing {
                return None;
            }
            let ok = match opts.only {
                Move::Any => true,
                Move::Up => score > was,
                Move::Down => score < was,
            };
            if !ok {
                return None;
            }
            // An unchanged score is not a change, which is what stops `CH` from
            // counting a member that was written over with what it already had.
            Some(if score == was {
                Added::Same
            } else {
                Added::Changed
            })
        }
    }
}

/// Turn a query into the run of ranks it covers.
fn window(z: &Zset, q: &Query<'_>) -> Window {
    let len = z.len();
    let range = match q.by {
        By::Rank { start, stop } => {
            // A rank range is already in the direction it will be walked, so a
            // reverse one counts its ends from the top rather than being found
            // and then flipped.
            let (from, count) = rank_span(start, stop, len);
            if q.rev {
                return apply_limit(
                    Window {
                        from: len - from - 1,
                        count,
                        rev: true,
                    },
                    q,
                    true,
                );
            }
            return apply_limit(
                Window {
                    from,
                    count,
                    rev: false,
                },
                q,
                true,
            );
        }
        By::Score { min, max } => z.window_by_score(min, max),
        By::Lex { min, max } => z.window_by_lex(min, max),
    };
    let count = range.end - range.start;
    let from = if q.rev {
        range.end.saturating_sub(1)
    } else {
        range.start
    };
    apply_limit(
        Window {
            from,
            count,
            rev: q.rev,
        },
        q,
        false,
    )
}

/// Move a window along by `LIMIT`'s offset and cut it to its count.
///
/// `ZRANGE key 0 -1 REV LIMIT 1 2` is the second and third from the top, so the
/// offset walks in the direction of the walk and not up the ranks.
///
/// A rank range has already had its ends clamped, and Redis does not accept a
/// `LIMIT` on one anyway, so `skip` says to leave it alone rather than the
/// caller passing an offset of zero and a count of none and hoping.
fn apply_limit(w: Window, q: &Query<'_>, skip: bool) -> Window {
    if skip {
        return w;
    }
    let offset = q.offset.min(w.count);
    let count = q.count.unwrap_or(usize::MAX).min(w.count - offset);
    let from = if w.rev {
        w.from.saturating_sub(offset)
    } else {
        w.from + offset
    };
    Window {
        from,
        count,
        rev: w.rev,
    }
}

/// Turn an inclusive `start` and `stop` into an offset from the front and a
/// count, clamping every out of range case rather than erroring.
///
/// The same rule `LRANGE` uses, and the same reason: a start before the front is
/// the front, a stop past the end is the end, and a start after the stop is
/// nothing at all.
fn rank_span(start: i64, stop: i64, len: usize) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let len = len as i64;
    let from = if start < 0 {
        (len + start).max(0)
    } else {
        start.min(len)
    };
    let to = if stop < 0 {
        len + stop
    } else {
        stop.min(len - 1)
    };
    if to < from {
        return (from as usize, 0);
    }
    (from as usize, (to - from + 1) as usize)
}

/// The bytes of a member, which for one the listpack stored as an integer are
/// the caller's buffer.
pub(crate) fn member_bytes<'a>(
    m: Member<'a>,
    digits: &'a mut [u8; yo_common::num::DIGITS_MAX],
) -> &'a [u8] {
    match m {
        Member::Str(s) => s,
        Member::Int(n) => yo_common::num::i64_digits(digits, n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::num::DIGITS_MAX;

    fn ks() -> Keyspace {
        Keyspace::new()
    }

    /// Add a run of pairs the plain way.
    fn add(k: &mut Keyspace, key: &[u8], pairs: &[(f64, &[u8])]) -> usize {
        k.zadd(key, pairs.iter().copied(), ZAdd::default()).unwrap()
    }

    /// Every member a query covers, as names.
    fn names(k: &mut Keyspace, key: &[u8], q: &Query<'_>) -> Vec<String> {
        let w = k.zwindow(key, q).unwrap();
        let mut out = Vec::new();
        let mut digits = [0u8; DIGITS_MAX];
        k.zwalk(key, w, |m, _| {
            out.push(String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap());
        })
        .unwrap();
        assert_eq!(
            out.len(),
            w.count,
            "the window said {} and the walk gave {}",
            w.count,
            out.len()
        );
        out
    }

    /// A positive count under the size of the set does a partial shuffle, and
    /// the permutation that needs used to be a fresh `Vec` every call. Sampling
    /// is a thing callers do in a loop, so the first call is allowed to grow the
    /// buffer and none after it may allocate at all.
    #[test]
    fn zrandmember_stops_allocating_once_its_buffer_is_grown() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")],
        );
        k.zrandmember(b"z", 2, |_, _| {}).expect("a sorted set");
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..100 {
                k.zrandmember(b"z", 2, |_, _| {}).expect("a sorted set");
            }
        });
        assert_eq!(
            allocs, 0,
            "zrandmember allocated {allocs} times in a hundred"
        );
    }

    /// The buffer is put back on the way out, so the call after it sees a
    /// buffer rather than an empty one, and both answer with what they were
    /// asked for.
    #[test]
    fn zrandmember_hands_its_buffer_back() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]);
        for _ in 0..3 {
            let mut seen = Vec::new();
            let n = k
                .zrandmember(b"z", 2, |m, _| {
                    let mut digits = [0u8; DIGITS_MAX];
                    seen.push(String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap());
                })
                .expect("a sorted set");
            assert_eq!(n, 2);
            assert_eq!(seen.len(), 2);
            assert_ne!(seen[0], seen[1], "a positive count draws without replacing");
        }
    }

    #[test]
    fn a_missing_key_is_an_empty_sorted_set() {
        let mut k = ks();
        assert_eq!(k.zcard(b"nope").unwrap(), 0);
        assert_eq!(k.zscore(b"nope", b"m").unwrap(), None);
        assert_eq!(k.zrank(b"nope", b"m", false).unwrap(), None);
        assert_eq!(k.zrem(b"nope", [b"m".as_slice()].into_iter()).unwrap(), 0);
        assert_eq!(
            names(&mut k, b"nope", &Query::rank(0, -1)),
            Vec::<String>::new()
        );
        assert!(!k.exists(b"nope"));
    }

    #[test]
    fn a_key_holding_something_else_is_a_wrongtype() {
        let mut k = ks();
        k.set(b"s", b"hello", strings::SetOptions::default())
            .unwrap();
        assert_eq!(k.zcard(b"s").unwrap_err().code(), Code::WrongType);
        assert_eq!(
            k.zadd(b"s", [(1.0, b"m".as_slice())].into_iter(), ZAdd::default())
                .unwrap_err()
                .code(),
            Code::WrongType
        );
        assert_eq!(k.zscore(b"s", b"m").unwrap_err().code(), Code::WrongType);
    }

    #[test]
    fn adding_answers_how_many_were_new() {
        let mut k = ks();
        assert_eq!(add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b")]), 2);
        assert_eq!(add(&mut k, b"z", &[(1.0, b"a"), (3.0, b"c")]), 1);
        assert_eq!(k.zcard(b"z").unwrap(), 3);
        assert_eq!(k.zscore(b"z", b"c").unwrap(), Some(3.0));
        assert_eq!(k.kind_of(b"z").map(Kind::name), Some("zset"));
        assert_eq!(k.encoding_name(b"z"), Some("listpack"));
    }

    #[test]
    fn the_ch_flag_counts_moved_scores_too() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b")]);
        let ch = ZAdd {
            changed: true,
            ..ZAdd::default()
        };
        // One score moved, one stayed, one member is new.
        let pairs = [
            (9.0, b"a".as_slice()),
            (2.0, b"b".as_slice()),
            (3.0, b"c".as_slice()),
        ];
        assert_eq!(k.zadd(b"z", pairs.into_iter(), ch).unwrap(), 2);
        assert_eq!(k.zadd(b"z", pairs.into_iter(), ZAdd::default()).unwrap(), 0);
    }

    #[test]
    fn the_gates_let_the_right_members_through() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a")]);
        let nx = ZAdd {
            gate: Gate::IfMissing,
            ..ZAdd::default()
        };
        let xx = ZAdd {
            gate: Gate::IfPresent,
            ..ZAdd::default()
        };
        assert_eq!(
            k.zadd(b"z", [(5.0, b"a".as_slice())].into_iter(), nx)
                .unwrap(),
            0
        );
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(1.0));
        assert_eq!(
            k.zadd(b"z", [(5.0, b"b".as_slice())].into_iter(), nx)
                .unwrap(),
            1
        );
        assert_eq!(
            k.zadd(b"z", [(7.0, b"c".as_slice())].into_iter(), xx)
                .unwrap(),
            0
        );
        assert_eq!(k.zscore(b"z", b"c").unwrap(), None);
        assert_eq!(
            k.zadd(b"z", [(7.0, b"a".as_slice())].into_iter(), xx)
                .unwrap(),
            0
        );
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(7.0));
        // XX on a key that does not exist does not create it.
        assert_eq!(
            k.zadd(b"gone", [(1.0, b"a".as_slice())].into_iter(), xx)
                .unwrap(),
            0
        );
        assert!(!k.exists(b"gone"));
    }

    #[test]
    fn gt_and_lt_only_move_a_score_one_way() {
        let mut k = ks();
        add(&mut k, b"z", &[(5.0, b"a")]);
        let gt = ZAdd {
            only: Move::Up,
            changed: true,
            ..ZAdd::default()
        };
        let lt = ZAdd {
            only: Move::Down,
            changed: true,
            ..ZAdd::default()
        };
        assert_eq!(
            k.zadd(b"z", [(3.0, b"a".as_slice())].into_iter(), gt)
                .unwrap(),
            0
        );
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(5.0));
        assert_eq!(
            k.zadd(b"z", [(9.0, b"a".as_slice())].into_iter(), gt)
                .unwrap(),
            1
        );
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(9.0));
        assert_eq!(
            k.zadd(b"z", [(9.0, b"a".as_slice())].into_iter(), lt)
                .unwrap(),
            0
        );
        assert_eq!(
            k.zadd(b"z", [(2.0, b"a".as_slice())].into_iter(), lt)
                .unwrap(),
            1
        );
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(2.0));
        // A member that is not there is added whatever GT says.
        assert_eq!(
            k.zadd(b"z", [(1.0, b"new".as_slice())].into_iter(), gt)
                .unwrap(),
            1
        );
    }

    #[test]
    fn incrementing_adds_to_what_is_there_or_to_nothing() {
        let mut k = ks();
        let plain = ZAdd::default();
        assert_eq!(k.zincrby(b"z", b"a", 5.0, plain).unwrap(), Some(5.0));
        assert_eq!(k.zincrby(b"z", b"a", -2.5, plain).unwrap(), Some(2.5));
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(2.5));
        let nx = ZAdd {
            gate: Gate::IfMissing,
            ..ZAdd::default()
        };
        assert_eq!(k.zincrby(b"z", b"a", 1.0, nx).unwrap(), None);
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(2.5));
        let xx = ZAdd {
            gate: Gate::IfPresent,
            ..ZAdd::default()
        };
        assert_eq!(k.zincrby(b"z", b"never", 1.0, xx).unwrap(), None);
        assert!(k.zscore(b"z", b"never").unwrap().is_none());
        let gt = ZAdd {
            only: Move::Up,
            ..ZAdd::default()
        };
        assert_eq!(k.zincrby(b"z", b"a", -1.0, gt).unwrap(), None);
        assert_eq!(k.zincrby(b"z", b"a", 1.0, gt).unwrap(), Some(3.5));
    }

    #[test]
    fn a_score_that_is_not_a_number_is_refused() {
        let mut k = ks();
        let plain = ZAdd::default();
        assert_eq!(
            k.zadd(b"z", [(f64::NAN, b"a".as_slice())].into_iter(), plain)
                .unwrap_err()
                .code(),
            Code::Invalid
        );
        assert!(!k.exists(b"z"));
        k.zincrby(b"z", b"a", f64::INFINITY, plain).unwrap();
        let err = k.zincrby(b"z", b"a", f64::NEG_INFINITY, plain).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        // The score is left exactly as it was rather than stored as a NaN.
        assert_eq!(k.zscore(b"z", b"a").unwrap(), Some(f64::INFINITY));
    }

    #[test]
    fn a_sorted_set_that_loses_its_last_member_loses_its_key() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b")]);
        assert_eq!(
            k.zrem(b"z", [b"a".as_slice(), b"b".as_slice()].into_iter())
                .unwrap(),
            2
        );
        assert!(!k.exists(b"z"));
        assert_eq!(k.zcard(b"z").unwrap(), 0);
    }

    #[test]
    fn ranks_count_from_both_ends() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]);
        assert_eq!(k.zrank(b"z", b"a", false).unwrap(), Some((0, 1.0)));
        assert_eq!(k.zrank(b"z", b"c", false).unwrap(), Some((2, 3.0)));
        assert_eq!(k.zrank(b"z", b"a", true).unwrap(), Some((2, 1.0)));
        assert_eq!(k.zrank(b"z", b"c", true).unwrap(), Some((0, 3.0)));
        assert_eq!(k.zrank(b"z", b"nope", false).unwrap(), None);
    }

    #[test]
    fn a_rank_range_clamps_every_way_it_can_be_wrong() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]);
        assert_eq!(names(&mut k, b"z", &Query::rank(0, -1)), ["a", "b", "c"]);
        assert_eq!(names(&mut k, b"z", &Query::rank(1, 1)), ["b"]);
        assert_eq!(names(&mut k, b"z", &Query::rank(-2, -1)), ["b", "c"]);
        assert_eq!(names(&mut k, b"z", &Query::rank(-99, 99)), ["a", "b", "c"]);
        assert_eq!(
            names(&mut k, b"z", &Query::rank(2, 1)),
            Vec::<String>::new()
        );
        assert_eq!(
            names(&mut k, b"z", &Query::rank(5, 9)),
            Vec::<String>::new()
        );
        assert_eq!(
            names(&mut k, b"z", &Query::rank(0, -1).rev(true)),
            ["c", "b", "a"]
        );
        assert_eq!(
            names(&mut k, b"z", &Query::rank(0, 1).rev(true)),
            ["c", "b"]
        );
    }

    #[test]
    fn a_score_range_walks_either_way_and_takes_a_limit() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")],
        );
        let all = Query::score(
            Bound::closed(f64::NEG_INFINITY),
            Bound::closed(f64::INFINITY),
        );
        assert_eq!(names(&mut k, b"z", &all), ["a", "b", "c", "d"]);
        assert_eq!(names(&mut k, b"z", &all.rev(true)), ["d", "c", "b", "a"]);
        assert_eq!(
            names(
                &mut k,
                b"z",
                &Query::score(Bound::closed(2.0), Bound::closed(3.0))
            ),
            ["b", "c"]
        );
        assert_eq!(
            names(
                &mut k,
                b"z",
                &Query::score(Bound::open(2.0), Bound::open(4.0))
            ),
            ["c"]
        );
        // LIMIT walks in the direction of the walk, so the reverse one skips
        // from the top.
        assert_eq!(names(&mut k, b"z", &all.limit(1, Some(2))), ["b", "c"]);
        assert_eq!(
            names(&mut k, b"z", &all.rev(true).limit(1, Some(2))),
            ["c", "b"]
        );
        assert_eq!(
            names(&mut k, b"z", &all.limit(9, Some(2))),
            Vec::<String>::new()
        );
        assert_eq!(names(&mut k, b"z", &all.limit(2, None)), ["c", "d"]);
        assert_eq!(
            k.zcount(b"z", &Query::score(Bound::closed(2.0), Bound::closed(3.0)))
                .unwrap(),
            2
        );
    }

    #[test]
    fn a_member_range_orders_by_member_when_every_score_is_the_same() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(0.0, b"a"), (0.0, b"b"), (0.0, b"c"), (0.0, b"d")],
        );
        assert_eq!(
            names(&mut k, b"z", &Query::lex(Lex::Min, Lex::Max)),
            ["a", "b", "c", "d"]
        );
        assert_eq!(
            names(&mut k, b"z", &Query::lex(Lex::Incl(b"b"), Lex::Incl(b"c"))),
            ["b", "c"]
        );
        assert_eq!(
            names(&mut k, b"z", &Query::lex(Lex::Excl(b"a"), Lex::Excl(b"d"))),
            ["b", "c"]
        );
        assert_eq!(
            names(&mut k, b"z", &Query::lex(Lex::Min, Lex::Max).rev(true)),
            ["d", "c", "b", "a"]
        );
        assert_eq!(
            k.zcount(b"z", &Query::lex(Lex::Incl(b"b"), Lex::Max))
                .unwrap(),
            3
        );
    }

    #[test]
    fn removing_a_range_takes_out_exactly_what_the_walk_would_have_given() {
        let mut k = ks();
        for (q, left) in [
            (Query::rank(0, 1), vec!["c", "d", "e"]),
            (Query::rank(-2, -1), vec!["a", "b", "c"]),
            (
                Query::score(Bound::closed(2.0), Bound::closed(4.0)),
                vec!["a", "e"],
            ),
            (
                Query::lex(Lex::Incl(b"b"), Lex::Excl(b"d")),
                vec!["a", "d", "e"],
            ),
            (Query::rank(0, -1).rev(true), vec![]),
        ] {
            k.del(b"z");
            add(
                &mut k,
                b"z",
                &[
                    (1.0, b"a"),
                    (2.0, b"b"),
                    (3.0, b"c"),
                    (4.0, b"d"),
                    (5.0, b"e"),
                ],
            );
            let want = names(&mut k, b"z", &q).len();
            assert_eq!(k.zremrange(b"z", &q).unwrap(), want, "{q:?}");
            assert_eq!(names(&mut k, b"z", &Query::rank(0, -1)), left, "{q:?}");
        }
        // The key went with the last member.
        assert!(!k.exists(b"z"));
    }

    #[test]
    fn popping_takes_from_the_end_it_was_told_to() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]);
        let mut got = Vec::new();
        let mut digits = [0u8; DIGITS_MAX];
        k.zpop(b"z", From::Min, 2, |m, s| {
            got.push((
                String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap(),
                s,
            ));
        })
        .unwrap();
        assert_eq!(got, [("a".to_string(), 1.0), ("b".to_string(), 2.0)]);
        assert_eq!(
            k.zpop_one(b"z", From::Max).unwrap(),
            Some((b"c".to_vec(), 3.0))
        );
        assert!(!k.exists(b"z"));
        // A count past the end takes what there is and no more.
        add(&mut k, b"z", &[(1.0, b"a")]);
        assert_eq!(k.zpop(b"z", From::Max, 99, |_, _| {}).unwrap(), 1);
        assert!(!k.exists(b"z"));
        assert_eq!(k.zpop_one(b"z", From::Min).unwrap(), None);
    }

    #[test]
    fn a_random_draw_is_with_or_without_replacement_by_the_sign_of_the_count() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]);
        let mut digits = [0u8; DIGITS_MAX];
        let mut seen = Vec::new();
        k.zrandmember(b"z", 2, |m, _| {
            seen.push(String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap());
        })
        .unwrap();
        assert_eq!(seen.len(), 2);
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 2, "a positive count does not repeat a member");
        // A count past the size is the whole set and not a repeat of it.
        let mut all = Vec::new();
        k.zrandmember(b"z", 99, |m, _| {
            all.push(String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap());
        })
        .unwrap();
        all.sort();
        assert_eq!(all, ["a", "b", "c"]);
        // A negative count answers exactly as many as asked for and may repeat.
        let mut many = 0;
        k.zrandmember(b"z", -10, |_, _| many += 1).unwrap();
        assert_eq!(many, 10);
        assert_eq!(k.zrandmember(b"nope", 3, |_, _| {}).unwrap(), 0);
    }

    #[test]
    fn a_scan_of_either_band_walks_every_member_once() {
        let mut k = ks();
        for entries in [8usize, 4096] {
            k.del(b"z");
            let pairs: Vec<(f64, Vec<u8>)> = (0..entries)
                .map(|i| (i as f64, format!("m{i:05}").into_bytes()))
                .collect();
            k.zadd(
                b"z",
                pairs.iter().map(|(s, m)| (*s, m.as_slice())),
                ZAdd::default(),
            )
            .unwrap();
            let mut seen = Vec::new();
            let mut digits = [0u8; DIGITS_MAX];
            let mut cursor = Cursor::START;
            loop {
                cursor = k
                    .zscan(b"z", cursor, 16, |m, _| {
                        seen.push(
                            String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap(),
                        );
                    })
                    .unwrap();
                if cursor.is_end() {
                    break;
                }
            }
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), entries, "{entries} members");
        }
    }

    #[test]
    fn a_big_sorted_set_promotes_and_still_answers_every_rank() {
        let mut k = ks();
        let pairs: Vec<(f64, Vec<u8>)> = (0..5_000)
            .map(|i| (f64::from(i), format!("m{i:05}").into_bytes()))
            .collect();
        assert_eq!(
            k.zadd(
                b"z",
                pairs.iter().map(|(s, m)| (*s, m.as_slice())),
                ZAdd::default()
            )
            .unwrap(),
            5_000
        );
        assert_eq!(k.encoding_name(b"z"), Some("skiplist"));
        assert_eq!(k.zcard(b"z").unwrap(), 5_000);
        assert_eq!(
            k.zrank(b"z", b"m02500", false).unwrap(),
            Some((2_500, 2500.0))
        );
        let q = Query::score(Bound::closed(1000.0), Bound::open(1010.0));
        assert_eq!(k.zcount(b"z", &q).unwrap(), 10);
        assert_eq!(
            names(&mut k, b"z", &q).first().map(String::as_str),
            Some("m01000")
        );
        // Everything still lines up after a few thousand removals.
        assert_eq!(k.zremrange(b"z", &Query::rank(0, 2_499)).unwrap(), 2_500);
        assert_eq!(k.zcard(b"z").unwrap(), 2_500);
        assert_eq!(k.zrank(b"z", b"m02500", false).unwrap(), Some((0, 2500.0)));
        assert_eq!(k.zrank(b"z", b"m00000", false).unwrap(), None);
    }

    #[test]
    fn a_deadline_on_a_sorted_set_leaves_its_members_alone() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a"), (2.0, b"b")]);
        assert!(k.set_expiry(b"z", Some(u64::MAX / 2)));
        assert_eq!(k.zcard(b"z").unwrap(), 2);
        assert_eq!(k.zscore(b"z", b"b").unwrap(), Some(2.0));
        assert!(k.set_expiry(b"z", None));
        assert_eq!(k.zcard(b"z").unwrap(), 2);
    }

    #[test]
    fn writing_a_string_over_a_sorted_set_gives_its_body_back() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a")]);
        let held = k.memory_bytes();
        k.set(b"z", b"now a string", strings::SetOptions::default())
            .unwrap();
        assert_eq!(k.kind_of(b"z").map(Kind::name), Some("string"));
        assert!(
            k.memory_bytes() < held,
            "the sorted set's body was not freed"
        );
    }

    /// Everything in a key, in rank order, as pairs.
    fn all(k: &mut Keyspace, key: &[u8]) -> Vec<(String, f64)> {
        let q = Query::rank(0, -1);
        let w = k.zwindow(key, &q).unwrap();
        let mut out = Vec::new();
        let mut digits = [0u8; DIGITS_MAX];
        k.zwalk(key, w, |m, s| {
            out.push((
                String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap(),
                s,
            ));
        })
        .unwrap();
        out
    }

    /// What one of the non storing forms answers, in the order it answered.
    fn got(
        k: &mut Keyspace,
        op: Op,
        keys: &[&[u8]],
        weights: &[f64],
        agg: Aggregate,
    ) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        let mut digits = [0u8; DIGITS_MAX];
        let n = k
            .zsetop(op, keys.iter().copied(), weights, agg, |m, s| {
                out.push((
                    String::from_utf8(member_bytes(m, &mut digits).to_vec()).unwrap(),
                    s,
                ));
            })
            .unwrap();
        assert_eq!(out.len(), n, "the count and the walk disagree");
        out
    }

    #[test]
    fn a_union_adds_the_scores_of_a_member_that_is_in_both() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x"), (2.0, b"y")]);
        add(&mut k, b"b", &[(10.0, b"y"), (3.0, b"z")]);
        assert_eq!(
            got(&mut k, Op::Union, &[b"a", b"b"], &[], Aggregate::Sum),
            [("x".into(), 1.0), ("z".into(), 3.0), ("y".into(), 12.0)]
        );
    }

    #[test]
    fn an_intersection_keeps_only_what_every_input_has() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x"), (2.0, b"y"), (3.0, b"z")]);
        add(&mut k, b"b", &[(5.0, b"y"), (5.0, b"z")]);
        add(&mut k, b"c", &[(7.0, b"z")]);
        assert_eq!(
            got(&mut k, Op::Inter, &[b"a", b"b", b"c"], &[], Aggregate::Sum),
            [("z".into(), 15.0)]
        );
        assert_eq!(
            got(&mut k, Op::Inter, &[b"a", b"b", b"c"], &[], Aggregate::Min),
            [("z".into(), 3.0)]
        );
        assert_eq!(
            got(&mut k, Op::Inter, &[b"a", b"b", b"c"], &[], Aggregate::Max),
            [("z".into(), 7.0)]
        );
    }

    #[test]
    fn a_difference_takes_the_first_input_and_removes_the_rest() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x"), (2.0, b"y"), (3.0, b"z")]);
        add(&mut k, b"b", &[(99.0, b"y")]);
        assert_eq!(
            got(&mut k, Op::Diff, &[b"a", b"b"], &[], Aggregate::Sum),
            [("x".into(), 1.0), ("z".into(), 3.0)]
        );
    }

    #[test]
    fn a_missing_key_keeps_its_place_so_the_weights_stay_lined_up() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x")]);
        add(&mut k, b"c", &[(1.0, b"x")]);
        // The second weight belongs to the key that is not there, and the third
        // to `c`. Dropping the gap would give `c` the 10 and answer 11.
        let out = got(
            &mut k,
            Op::Union,
            &[b"a", b"gone", b"c"],
            &[2.0, 10.0, 3.0],
            Aggregate::Sum,
        );
        assert_eq!(out, [("x".into(), 5.0)]);
    }

    #[test]
    fn a_plain_set_counts_as_every_score_being_one() {
        let mut k = ks();
        add(&mut k, b"z", &[(5.0, b"x")]);
        k.sadd(b"s", [b"x".as_slice(), b"y".as_slice()].into_iter())
            .unwrap();
        assert_eq!(
            got(&mut k, Op::Union, &[b"z", b"s"], &[], Aggregate::Sum),
            [("y".into(), 1.0), ("x".into(), 6.0)]
        );
        assert_eq!(
            got(&mut k, Op::Inter, &[b"z", b"s"], &[], Aggregate::Min),
            [("x".into(), 1.0)]
        );
    }

    #[test]
    fn a_store_writes_the_result_and_answers_its_size() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x"), (2.0, b"y")]);
        add(&mut k, b"b", &[(10.0, b"y")]);
        assert_eq!(
            k.zsetop_store(
                Op::Union,
                b"d",
                [b"a".as_slice(), b"b".as_slice()].into_iter(),
                &[],
                Aggregate::Sum
            )
            .unwrap(),
            2
        );
        assert_eq!(all(&mut k, b"d"), [("x".into(), 1.0), ("y".into(), 12.0)]);
        assert_eq!(k.zscore(b"d", b"y").unwrap(), Some(12.0));
        assert_eq!(k.zrank(b"d", b"y", false).unwrap(), Some((1, 12.0)));
    }

    #[test]
    fn a_store_onto_one_of_its_own_sources_still_reads_the_old_body() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x"), (2.0, b"y")]);
        add(&mut k, b"b", &[(10.0, b"y")]);
        assert_eq!(
            k.zsetop_store(
                Op::Union,
                b"a",
                [b"a".as_slice(), b"b".as_slice()].into_iter(),
                &[],
                Aggregate::Sum
            )
            .unwrap(),
            2
        );
        assert_eq!(all(&mut k, b"a"), [("x".into(), 1.0), ("y".into(), 12.0)]);
    }

    #[test]
    fn a_store_with_nothing_in_it_deletes_the_destination() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x")]);
        add(&mut k, b"b", &[(1.0, b"y")]);
        add(&mut k, b"d", &[(1.0, b"old")]);
        assert_eq!(
            k.zsetop_store(
                Op::Inter,
                b"d",
                [b"a".as_slice(), b"b".as_slice()].into_iter(),
                &[],
                Aggregate::Sum
            )
            .unwrap(),
            0
        );
        assert!(!k.exists(b"d"));
    }

    #[test]
    fn a_store_clears_the_deadline_the_destination_was_carrying() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x")]);
        add(&mut k, b"d", &[(1.0, b"old")]);
        assert!(k.set_expiry(b"d", Some(u64::MAX / 2)));
        k.zsetop_store(
            Op::Union,
            b"d",
            [b"a".as_slice()].into_iter(),
            &[],
            Aggregate::Sum,
        )
        .unwrap();
        assert_eq!(all(&mut k, b"d"), [("x".into(), 1.0)]);
        assert_eq!(k.expire_at(b"d"), None);
    }

    #[test]
    fn the_algebra_refuses_a_key_holding_something_that_is_neither() {
        let mut k = ks();
        add(&mut k, b"a", &[(1.0, b"x")]);
        k.set(b"s", b"hello", strings::SetOptions::default())
            .unwrap();
        assert_eq!(
            k.zsetop(
                Op::Union,
                [b"a".as_slice(), b"s".as_slice()].into_iter(),
                &[],
                Aggregate::Sum,
                |_, _| {}
            )
            .unwrap_err()
            .code(),
            Code::WrongType
        );
        // And the destination is untouched, because the keys are all resolved
        // before anything is built.
        assert!(!k.exists(b"d"));
        assert_eq!(
            k.zsetop_store(
                Op::Union,
                b"d",
                [b"a".as_slice(), b"s".as_slice()].into_iter(),
                &[],
                Aggregate::Sum
            )
            .unwrap_err()
            .code(),
            Code::WrongType
        );
        assert!(!k.exists(b"d"));
    }

    #[test]
    fn intercard_counts_without_building_anything_and_stops_at_the_limit() {
        let mut k = ks();
        add(
            &mut k,
            b"a",
            &[(1.0, b"w"), (2.0, b"x"), (3.0, b"y"), (4.0, b"z")],
        );
        add(&mut k, b"b", &[(1.0, b"x"), (1.0, b"y"), (1.0, b"z")]);
        assert_eq!(
            k.zintercard([b"a".as_slice(), b"b".as_slice()].into_iter(), 0)
                .unwrap(),
            3
        );
        assert_eq!(
            k.zintercard([b"a".as_slice(), b"b".as_slice()].into_iter(), 2)
                .unwrap(),
            2
        );
        assert_eq!(
            k.zintercard([b"a".as_slice(), b"gone".as_slice()].into_iter(), 0)
                .unwrap(),
            0
        );
    }

    #[test]
    fn a_range_store_copies_a_window_and_leaves_the_source_alone() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")],
        );
        assert_eq!(k.zrangestore(b"d", b"z", &Query::rank(1, 2)).unwrap(), 2);
        assert_eq!(all(&mut k, b"d"), [("b".into(), 2.0), ("c".into(), 3.0)]);
        assert_eq!(k.zcard(b"z").unwrap(), 4);
    }

    #[test]
    fn a_reverse_range_store_picks_the_same_members_and_orders_them_the_same() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")],
        );
        // REV picks which members, not what order they end up in, because the
        // destination is a sorted set and a sorted set has one order.
        assert_eq!(
            k.zrangestore(b"d", b"z", &Query::rank(0, 1).rev(true))
                .unwrap(),
            2
        );
        assert_eq!(all(&mut k, b"d"), [("c".into(), 3.0), ("d".into(), 4.0)]);
    }

    #[test]
    fn a_range_store_by_score_takes_the_bounds_the_range_would_have() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")],
        );
        let q = Query::score(Bound::closed(2.0), Bound::open(4.0));
        assert_eq!(k.zrangestore(b"d", b"z", &q).unwrap(), 2);
        assert_eq!(all(&mut k, b"d"), [("b".into(), 2.0), ("c".into(), 3.0)]);
    }

    #[test]
    fn a_range_store_of_an_empty_window_deletes_the_destination() {
        let mut k = ks();
        add(&mut k, b"z", &[(1.0, b"a")]);
        add(&mut k, b"d", &[(1.0, b"old")]);
        assert_eq!(k.zrangestore(b"d", b"z", &Query::rank(5, 9)).unwrap(), 0);
        assert!(!k.exists(b"d"));
        assert_eq!(
            k.zrangestore(b"d", b"gone", &Query::rank(0, -1)).unwrap(),
            0
        );
        assert!(!k.exists(b"d"));
    }

    #[test]
    fn a_range_store_onto_its_own_source_keeps_the_window() {
        let mut k = ks();
        add(
            &mut k,
            b"z",
            &[(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")],
        );
        assert_eq!(k.zrangestore(b"z", b"z", &Query::rank(1, 2)).unwrap(), 2);
        assert_eq!(all(&mut k, b"z"), [("b".into(), 2.0), ("c".into(), 3.0)]);
    }

    #[test]
    fn a_big_result_comes_out_on_the_table_band_in_the_right_order() {
        let mut k = ks();
        let names: Vec<String> = (0..3000).map(|i| format!("member-{i:05}")).collect();
        for (i, name) in names.iter().enumerate() {
            add(&mut k, b"a", &[((i % 17) as f64, name.as_bytes())]);
        }
        for name in names.iter().step_by(3) {
            add(&mut k, b"b", &[(100.0, name.as_bytes())]);
        }
        let n = k
            .zsetop_store(
                Op::Union,
                b"d",
                [b"a".as_slice(), b"b".as_slice()].into_iter(),
                &[],
                Aggregate::Sum,
            )
            .unwrap();
        assert_eq!(n, 3000);
        assert_eq!(
            k.zset_encoding(b"d").map(crate::zset::Encoding::name),
            Some("skiplist")
        );
        let out = all(&mut k, b"d");
        assert_eq!(out.len(), 3000);
        let mut want = out.clone();
        want.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap().then_with(|| x.0.cmp(&y.0)));
        assert_eq!(out, want, "the result came out in the wrong order");
        for (i, name) in names.iter().enumerate() {
            let base = (i % 17) as f64;
            let want = if i % 3 == 0 { base + 100.0 } else { base };
            assert_eq!(k.zscore(b"d", name.as_bytes()).unwrap(), Some(want));
        }
    }
}
