//! The array commands.
//!
//! One method per Redis command on [`Keyspace`], the same arrangement the list
//! and set commands use and for the same reason: a key belongs to the database
//! and not to a type, so `ARSET` against a string has to be able to see that it
//! is a string. The array itself is [`crate::array`]. This file is what the wire
//! and the embedded API both call.
//!
//! # Indices are unsigned and that changes things
//!
//! Every other collection here takes a signed index and counts from the back
//! when it is negative. An array does not: the index is a position in a space
//! that runs to `2^64 - 2`, there is no back to count from, and `-1` is an error
//! rather than the last element. The wire layer parses an index with
//! [`parse_index`] rather than with the signed parser the list commands use, and
//! the error it gives back is Redis's own wording.
//!
//! # An empty array is not an array
//!
//! The key goes when the last element does, which is the same rule every
//! collection here follows and the reason `EXISTS` answers zero after the last
//! `ARDEL`.

use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};

use crate::array::{Array, ELEMENT_MAX, Element, INDEX_MAX, Info};
use crate::keyspace::Keyspace;
use crate::strings;
use crate::value::{self, Kind};

/// What every array command says about an index it cannot read.
///
/// Redis's words, because they go on the wire verbatim. It covers a negative
/// number, a number with anything but digits in it, and `2^64 - 1`, which is
/// reserved rather than addressable.
pub const BAD_INDEX: &str = "invalid array index";

/// What `ARSET` says when the last index it would write to does not exist.
pub const INDEX_OVERFLOW: &str = "array index overflow";

/// The most positions `ARGETRANGE` will answer for.
///
/// Redis's `ARGETRANGE_MAX_ITEMS`, and its comment says this "must be part of
/// the Redis culture, so it should not be tuned in any way". The reason is
/// worth keeping: the reply is one entry per position and not one per element,
/// so without a limit `ARGETRANGE k 0 18446744073709551614` against a key that
/// does not exist is a request for eighteen quintillion nulls, which is a way to
/// stop a server with four short words.
pub const GETRANGE_MAX: u64 = 1_000_000;

/// Reads an index the way Redis reads one.
///
/// Unsigned, no leading plus, no leading zeros, and `2^64 - 1` refused because
/// it is the "nothing has been inserted yet" marker in the cursor `ARINSERT`
/// and `ARNEXT` share. Everything else in the range is a position, including
/// zero.
///
/// # Errors
///
/// [`Code::Invalid`] with Redis's own message, for anything else.
pub fn parse_index(bytes: &[u8]) -> Result<u64> {
    parse_ull(bytes, false)
}

/// Reads the one index that may be `2^64 - 1`, which is `ARSEEK`'s.
///
/// Seeking to the top of the space is how the cursor gets into the state where
/// the next append has nowhere to go, and that state has to be reachable from a
/// command because the log that rebuilds a database is made of commands.
///
/// # Errors
///
/// [`Code::Invalid`] with Redis's own message, the same as [`parse_index`].
pub fn parse_seek_index(bytes: &[u8]) -> Result<u64> {
    parse_ull(bytes, true)
}

fn parse_ull(bytes: &[u8], allow_max: bool) -> Result<u64> {
    let bad = || Error::new(Code::Invalid, BAD_INDEX);
    if bytes.is_empty() || bytes.len() > 20 {
        return Err(bad());
    }
    // Redis's `string2ull`: one zero on its own is fine, a leading zero in front
    // of anything else is not, and nothing but digits is allowed.
    if bytes[0] == b'0' && bytes.len() > 1 {
        return Err(bad());
    }
    let mut n: u64 = 0;
    for &c in bytes {
        if !c.is_ascii_digit() {
            return Err(bad());
        }
        n = n
            .checked_mul(10)
            .and_then(|n| n.checked_add(u64::from(c - b'0')))
            .ok_or_else(bad)?;
    }
    if n > INDEX_MAX && !allow_max {
        return Err(bad());
    }
    Ok(n)
}

/// The aggregations `AROP` knows how to do.
///
/// They are all order independent, which is why the walk can go whichever way
/// the two ends point without the answer changing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Add up everything that is a number.
    Sum,
    /// The smallest of them.
    Min,
    /// The largest of them.
    Max,
    /// Bitwise and over everything that is a whole number.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// How many elements are exactly these bytes.
    Match,
    /// How many positions in the range hold anything at all.
    Used,
}

/// What an [`Op`] came to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aggregate {
    /// A count or a bitwise result, which goes back as an integer.
    Int(i64),
    /// A sum or an end of the range, which goes back as a string of digits
    /// because it may not be a whole number.
    Num(f64),
    /// Nothing in the range was any use to the operation, which is a null. An
    /// empty range and a range of nothing but words both land here.
    None,
}

/// An element as a whole number, for the bitwise operations.
///
/// A float is truncated towards zero the way Redis does it, and one that will
/// not fit is skipped rather than saturated, because a saturated value would
/// quietly poison an `AND` with a row of ones.
fn as_int(el: Element<'_>) -> Option<i64> {
    match el {
        Element::Int(n) => Some(n),
        Element::Float(d) => whole(d),
        _ => {
            let mut buf = [0u8; ELEMENT_MAX];
            let text = el.text(&mut buf);
            parse_i64(text).or_else(|| whole(parse_f64(text)?))
        }
    }
}

/// An element as a number, for the arithmetic operations.
fn as_num(el: Element<'_>) -> Option<f64> {
    match el {
        Element::Int(n) => Some(n as f64),
        Element::Float(d) => Some(d),
        _ => {
            let mut buf = [0u8; ELEMENT_MAX];
            parse_f64(el.text(&mut buf))
        }
    }
}

/// A double as the integer it truncates to, or nothing when it does not.
fn whole(d: f64) -> Option<i64> {
    if d.is_nan() || d < -(2f64.powi(63)) || d >= 2f64.powi(63) {
        return None;
    }
    Some(d as i64)
}

impl Keyspace {
    /// `ARSET key index value [value ...]`, which writes at consecutive indices.
    ///
    /// Answers how many of the positions were empty before, which is not the
    /// same as how many values were written: `ARSET k 0 a b` twice answers 2 and
    /// then 0.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the last index the write would reach does not
    /// exist, so that a write which would run off the top of the index space
    /// fails before any of it lands rather than half way through.
    pub fn arset<'v>(
        &mut self,
        key: &[u8],
        index: u64,
        values: impl Iterator<Item = &'v [u8]> + Clone,
    ) -> Result<u64> {
        let count = values.clone().count() as u64;
        if count == 0 {
            return Ok(0);
        }
        // The last index is `index + count - 1`, and both the overflow and
        // landing on the reserved top of the space are the same error.
        if index
            .checked_add(count - 1)
            .is_none_or(|last| last > INDEX_MAX)
        {
            return Err(Error::new(Code::Invalid, INDEX_OVERFLOW));
        }
        for v in values.clone() {
            strings::check_len(key, v.len())?;
        }

        let at = match self.array_slot(key)? {
            Some(at) => at,
            None => self.new_array(key),
        };
        let array = self
            .arrays
            .get_mut(at)
            .expect("the record points at its body");
        let mut filled = 0;
        for (i, v) in values.enumerate() {
            if array.set(index + i as u64, v)? {
                filled += 1;
            }
        }
        Ok(filled)
    }

    /// `ARMSET key index value [index value ...]`, which writes scattered pairs.
    ///
    /// Answers how many of the positions were empty before, the same as
    /// [`Keyspace::arset`]. The pairs arrive already parsed, because the wire
    /// layer has to read every index before it writes any of them: a bad index
    /// in the last pair fails the whole command and leaves the earlier pairs
    /// unwritten.
    pub fn armset<'v>(
        &mut self,
        key: &[u8],
        pairs: impl Iterator<Item = (u64, &'v [u8])> + Clone,
    ) -> Result<u64> {
        if pairs.clone().next().is_none() {
            return Ok(0);
        }
        for (_, v) in pairs.clone() {
            strings::check_len(key, v.len())?;
        }
        let at = match self.array_slot(key)? {
            Some(at) => at,
            None => self.new_array(key),
        };
        let array = self
            .arrays
            .get_mut(at)
            .expect("the record points at its body");
        let mut filled = 0;
        for (index, v) in pairs {
            if array.set(index, v)? {
                filled += 1;
            }
        }
        Ok(filled)
    }

    /// `ARGET key index`. A hole and a missing key are the same answer.
    pub fn arget(&mut self, key: &[u8], index: u64) -> Result<Option<Element<'_>>> {
        let Some(at) = self.array_slot(key)? else {
            return Ok(None);
        };
        Ok(self.array_at(at).get(index))
    }

    /// `ARMGET key index [index ...]`, straight into the reply.
    ///
    /// `f` is called once per index in the order they were asked for, with the
    /// element or `None` for a hole, and it is called while the element is still
    /// in the array so that nothing is copied on the way (Y18).
    pub fn arget_into<F>(
        &mut self,
        key: &[u8],
        indices: impl Iterator<Item = u64>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Option<Element<'_>>),
    {
        let slot = self.array_slot(key)?;
        match slot {
            Some(at) => {
                let array = self.array_at(at);
                for index in indices {
                    f(array.get(index));
                }
            }
            // A key that is not there answers the same as an array of holes,
            // which is Redis's rule and the reason this is not an early return
            // with nothing written.
            None => {
                for _ in indices {
                    f(None);
                }
            }
        }
        Ok(())
    }

    /// `ARGETRANGE key start end`, every position in the range and not every
    /// element.
    ///
    /// `f` is called once per position, holes included, low to high or high to
    /// low depending on which way round the two ends came in. The count is
    /// answered first so that the caller can write the array header before the
    /// first element.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the range covers more than [`GETRANGE_MAX`]
    /// positions.
    pub fn argetrange<F>(&mut self, key: &[u8], start: u64, end: u64, mut f: F) -> Result<u64>
    where
        F: FnMut(Option<Element<'_>>),
    {
        let reverse = start > end;
        let (lo, hi) = if reverse { (end, start) } else { (start, end) };
        let len = hi - lo + 1;
        if len > GETRANGE_MAX {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!("range exceeds maximum of {GETRANGE_MAX} items"),
            ));
        }
        let slot = self.array_slot(key)?;
        let Some(at) = slot else {
            for _ in 0..len {
                f(None);
            }
            return Ok(len);
        };
        let array = self.array_at(at);
        if reverse {
            for i in 0..len {
                f(array.get(hi - i));
            }
        } else {
            for i in 0..len {
                f(array.get(lo + i));
            }
        }
        Ok(len)
    }

    /// `ARLEN key`, the highest populated index plus one.
    ///
    /// Zero for a key that is not there, and note that this is not the number of
    /// elements. [`Keyspace::arcount`] is that.
    pub fn arlen(&mut self, key: &[u8]) -> Result<u64> {
        Ok(match self.array_slot(key)? {
            Some(at) => self.array_at(at).len(),
            None => 0,
        })
    }

    /// `ARCOUNT key`, how many indices hold something.
    pub fn arcount(&mut self, key: &[u8]) -> Result<u64> {
        Ok(match self.array_slot(key)? {
            Some(at) => self.array_at(at).count(),
            None => 0,
        })
    }

    /// `ARDEL key index [index ...]`. Answers how many held something.
    ///
    /// The key goes when the last element does.
    pub fn ardel(&mut self, key: &[u8], indices: impl Iterator<Item = u64>) -> Result<u64> {
        let Some(at) = self.array_slot(key)? else {
            return Ok(0);
        };
        let array = self
            .arrays
            .get_mut(at)
            .expect("the record points at its body");
        let mut gone = 0;
        for index in indices {
            if array.del(index) {
                gone += 1;
            }
        }
        if array.is_empty() {
            self.drop_key(key);
        }
        Ok(gone)
    }

    /// `ARDELRANGE key start end [start end ...]`. Answers how many went.
    ///
    /// Each pair may come in either order. The cost is in the elements the
    /// ranges touch and not in how wide they are, so clearing the whole index
    /// space of a key holding three elements is three deletes.
    pub fn ardelrange(
        &mut self,
        key: &[u8],
        ranges: impl Iterator<Item = (u64, u64)>,
    ) -> Result<u64> {
        let Some(at) = self.array_slot(key)? else {
            return Ok(0);
        };
        let array = self
            .arrays
            .get_mut(at)
            .expect("the record points at its body");
        let mut gone = 0;
        for (start, end) in ranges {
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            gone += array.delete_range(lo, hi);
        }
        if array.is_empty() {
            self.drop_key(key);
        }
        Ok(gone)
    }

    /// `ARINSERT key value [value ...]`, which appends at the cursor.
    ///
    /// Answers the index the last value landed on. The cursor starts at zero
    /// and a plain `ARSET` never moves it, so an array somebody has written by
    /// index and then appended to will have the first append land on top of
    /// index zero. That is Redis's behaviour and it is the reason `ARSEEK`
    /// exists.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the batch would run off the top of the index
    /// space, checked before any of it is written.
    pub fn arinsert<'v>(
        &mut self,
        key: &[u8],
        values: impl Iterator<Item = &'v [u8]> + Clone,
    ) -> Result<u64> {
        for v in values.clone() {
            strings::check_len(key, v.len())?;
        }
        let at = match self.array_slot(key)? {
            Some(at) => at,
            // A new array has its cursor at zero, so the append below cannot
            // fail on one and cannot leave an empty key behind.
            None => self.new_array(key),
        };
        self.arrays
            .get_mut(at)
            .expect("the record points at its body")
            .append(values)
    }

    /// `ARRING key size value [value ...]`, a ring buffer over the indices.
    ///
    /// Answers the index the last value landed on. `size` has to be at least
    /// one, which the caller checks because Redis reports a bad size before it
    /// has even looked at the key.
    pub fn arring<'v>(
        &mut self,
        key: &[u8],
        size: u64,
        values: impl Iterator<Item = &'v [u8]> + Clone,
    ) -> Result<u64> {
        debug_assert!(size > 0, "the caller checks the size");
        for v in values.clone() {
            strings::check_len(key, v.len())?;
        }
        let at = match self.array_slot(key)? {
            Some(at) => at,
            None => self.new_array(key),
        };
        self.arrays
            .get_mut(at)
            .expect("the record points at its body")
            .ring(size, values)
    }

    /// `ARNEXT key`, where the next append would go.
    ///
    /// Zero for a key that is not there and zero for a cursor nothing has moved
    /// yet, which are the same answer because they mean the same thing. `None`
    /// is the null a client sees when the cursor has run out of index space and
    /// there is no honest answer to give.
    pub fn arnext(&mut self, key: &[u8]) -> Result<Option<u64>> {
        Ok(match self.array_slot(key)? {
            Some(at) => self.array_at(at).next_index(),
            None => Some(0),
        })
    }

    /// `ARSEEK key index`, which points the cursor.
    ///
    /// Answers whether there was a key to point. A missing key answers false
    /// and is not created, because an array with nothing in it is not a key
    /// here and an error would be worse: the caller asked to move a cursor, and
    /// the honest answer is that there was no cursor to move.
    ///
    /// `index` is the one place in the array commands where `2^64 - 1` is a
    /// legal argument. It leaves the cursor in the terminal state, which is
    /// what the rewritten command has to say to reproduce that state on load.
    pub fn arseek(&mut self, key: &[u8], index: u64) -> Result<bool> {
        let Some(at) = self.array_slot(key)? else {
            return Ok(false);
        };
        self.arrays
            .get_mut(at)
            .expect("the record points at its body")
            .seek(index);
        Ok(true)
    }

    /// `ARLASTITEMS key count [REV]`, the newest positions from the cursor.
    ///
    /// `f` is called once per position, oldest first unless `newest_first`, and
    /// a hole inside the window is a `None` rather than something skipped. The
    /// count is answered so the caller can close its array header.
    pub fn arlastitems<F>(
        &mut self,
        key: &[u8],
        count: u64,
        newest_first: bool,
        f: F,
    ) -> Result<u64>
    where
        F: FnMut(Option<Element<'_>>),
    {
        Ok(match self.array_slot(key)? {
            Some(at) => self.array_at(at).last_items(count, newest_first, f),
            None => 0,
        })
    }

    /// `ARSCAN key start end [LIMIT count]`, the elements and not the positions.
    ///
    /// `f` is called with the index and the element for everything populated in
    /// the range, low to high or high to low depending on which way round the
    /// ends came in, and at most `limit` times. Answers how many that was.
    ///
    /// Unlike [`Keyspace::argetrange`] this has no ceiling on the range, and it
    /// does not need one: holes cost nothing, so `ARSCAN k 0 18446744073709551614`
    /// against a key holding three elements is three visits and not eighteen
    /// quintillion.
    pub fn arscan<F>(
        &mut self,
        key: &[u8],
        start: u64,
        end: u64,
        limit: u64,
        mut f: F,
    ) -> Result<u64>
    where
        F: FnMut(u64, Element<'_>),
    {
        let Some(at) = self.array_slot(key)? else {
            return Ok(0);
        };
        let mut seen = 0;
        if limit > 0 {
            self.array_at(at).scan(start, end, |index, el| {
                f(index, el);
                seen += 1;
                seen < limit
            });
        }
        Ok(seen)
    }

    /// `AROP key start end OP [value]`, one number out of a whole range.
    ///
    /// The walk is [`Keyspace::arscan`]'s, so it costs the elements in the range
    /// and not its width, and every operation here is order independent so the
    /// direction the ends came in does not matter.
    pub fn arop(
        &mut self,
        key: &[u8],
        start: u64,
        end: u64,
        op: Op,
        want: &[u8],
    ) -> Result<Aggregate> {
        let Some(at) = self.array_slot(key)? else {
            // A count of nothing is zero and an aggregate of nothing is a null,
            // which is the difference between asking how many and asking what.
            return Ok(match op {
                Op::Match | Op::Used => Aggregate::Int(0),
                _ => Aggregate::None,
            });
        };
        let mut counted = 0i64;
        let mut bits: Option<i64> = None;
        let mut num: Option<f64> = None;
        self.array_at(at).scan(start, end, |_, el| {
            match op {
                Op::Used => counted += 1,
                Op::Match => {
                    let mut buf = [0u8; ELEMENT_MAX];
                    if el.text(&mut buf) == want {
                        counted += 1;
                    }
                }
                Op::And | Op::Or | Op::Xor => {
                    if let Some(i) = as_int(el) {
                        bits = Some(match (bits, op) {
                            (None, _) => i,
                            (Some(acc), Op::And) => acc & i,
                            (Some(acc), Op::Or) => acc | i,
                            (Some(acc), _) => acc ^ i,
                        });
                    }
                }
                Op::Sum | Op::Min | Op::Max => {
                    if let Some(d) = as_num(el) {
                        num = Some(match (num, op) {
                            (None, _) => d,
                            (Some(acc), Op::Sum) => acc + d,
                            (Some(acc), Op::Min) => acc.min(d),
                            (Some(acc), _) => acc.max(d),
                        });
                    }
                }
            }
            true
        });
        Ok(match op {
            Op::Match | Op::Used => Aggregate::Int(counted),
            Op::And | Op::Or | Op::Xor => bits.map_or(Aggregate::None, Aggregate::Int),
            _ => num.map_or(Aggregate::None, Aggregate::Num),
        })
    }

    /// `ARINFO key [FULL]`, the shape of the array.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] carrying `no such key` for a key that is not there, which is the one array
    /// command that treats a missing key as a mistake rather than as an empty
    /// array. It is reporting on a structure, and there is no structure.
    pub fn arinfo(&mut self, key: &[u8], full: bool) -> Result<Info> {
        let Some(at) = self.array_slot(key)? else {
            return Err(crate::keys::no_such_key());
        };
        Ok(self.array_at(at).info(full))
    }

    /// Where `key`'s array is, or `None` if there is no such key.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] if the key holds something that is not an array.
    fn array_slot(&mut self, key: &[u8]) -> Result<Option<u32>> {
        self.live_slot(key, Kind::Array)
    }

    fn array_at(&self, at: u32) -> &Array {
        self.arrays.get(at).expect("the record points at its body")
    }

    fn new_array(&mut self, key: &[u8]) -> u32 {
        let at = self.arrays.insert(Array::new());
        let len = value::slot_record_len(false);
        self.map.set_with(key, len, |out| {
            value::write_slot_record(out, Kind::Array, at, None);
        });
        self.bodies += 1;
        at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::ELEMENT_MAX;

    fn db() -> Keyspace {
        Keyspace::new()
    }

    /// The bytes a client would see at one index.
    fn read(d: &mut Keyspace, key: &[u8], index: u64) -> Option<Vec<u8>> {
        let el = d.arget(key, index).expect("an array")?;
        let mut buf = [0u8; ELEMENT_MAX];
        Some(el.text(&mut buf).to_vec())
    }

    fn set(d: &mut Keyspace, key: &[u8], index: u64, vals: &[&[u8]]) -> u64 {
        d.arset(key, index, vals.iter().copied()).expect("an array")
    }

    #[test]
    fn a_write_makes_the_key_and_a_read_finds_it() {
        let mut d = db();
        assert_eq!(read(&mut d, b"a", 0), None, "no key yet");
        assert_eq!(set(&mut d, b"a", 5, &[b"x"]), 1);
        assert_eq!(d.kind_of(b"a"), Some(Kind::Array));
        assert_eq!(read(&mut d, b"a", 5).as_deref(), Some(&b"x"[..]));
        assert_eq!(read(&mut d, b"a", 4), None, "a hole");
        assert_eq!(d.arlen(b"a").expect("an array"), 6);
        assert_eq!(d.arcount(b"a").expect("an array"), 1);
    }

    #[test]
    fn a_set_writes_consecutive_positions_and_counts_the_new_ones() {
        let mut d = db();
        assert_eq!(set(&mut d, b"a", 10, &[b"p", b"q", b"r"]), 3);
        assert_eq!(set(&mut d, b"a", 10, &[b"P", b"Q"]), 0, "already filled");
        assert_eq!(set(&mut d, b"a", 12, &[b"R", b"s"]), 1, "one of the two");
        assert_eq!(read(&mut d, b"a", 10).as_deref(), Some(&b"P"[..]));
        assert_eq!(read(&mut d, b"a", 13).as_deref(), Some(&b"s"[..]));
        assert_eq!(d.arcount(b"a").expect("an array"), 4);
        assert_eq!(d.arlen(b"a").expect("an array"), 14);
    }

    /// A write that would run off the top of the index space fails before any
    /// of it lands.
    #[test]
    fn a_write_past_the_end_of_the_space_writes_nothing() {
        let mut d = db();
        let e = d
            .arset(b"a", INDEX_MAX, [b"x".as_ref(), b"y".as_ref()].into_iter())
            .unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), INDEX_OVERFLOW);
        assert_eq!(d.kind_of(b"a"), None, "and the key was never made");

        // The last index on its own is fine.
        assert_eq!(set(&mut d, b"a", INDEX_MAX, &[b"x"]), 1);
        assert_eq!(d.arlen(b"a").expect("an array"), u64::MAX);
    }

    #[test]
    fn scattered_pairs_go_in_one_command() {
        let mut d = db();
        let pairs = [
            (1u64, b"a".as_ref()),
            (1000, b"b".as_ref()),
            (1, b"c".as_ref()),
        ];
        assert_eq!(d.armset(b"k", pairs.into_iter()).expect("an array"), 2);
        assert_eq!(
            read(&mut d, b"k", 1).as_deref(),
            Some(&b"c"[..]),
            "the later one won"
        );
        assert_eq!(read(&mut d, b"k", 1000).as_deref(), Some(&b"b"[..]));
        assert_eq!(d.arcount(b"k").expect("an array"), 2);
    }

    #[test]
    fn the_key_goes_when_the_last_element_does() {
        let mut d = db();
        set(&mut d, b"a", 0, &[b"x", b"y"]);
        assert_eq!(d.ardel(b"a", [0u64].into_iter()).expect("an array"), 1);
        assert_eq!(d.kind_of(b"a"), Some(Kind::Array), "still one left");
        assert_eq!(d.ardel(b"a", [1u64, 2].into_iter()).expect("an array"), 1);
        assert_eq!(d.kind_of(b"a"), None);
        assert_eq!(d.ardel(b"a", [0u64].into_iter()).expect("an array"), 0);
    }

    #[test]
    fn a_range_delete_takes_both_ways_round() {
        let mut d = db();
        set(&mut d, b"a", 0, &[b"0", b"1", b"2", b"3", b"4"]);
        assert_eq!(
            d.ardelrange(b"a", [(3u64, 1u64)].into_iter())
                .expect("an array"),
            3,
            "given high to low"
        );
        assert_eq!(d.arcount(b"a").expect("an array"), 2);
        assert_eq!(read(&mut d, b"a", 0).as_deref(), Some(&b"0"[..]));
        assert_eq!(read(&mut d, b"a", 4).as_deref(), Some(&b"4"[..]));

        assert_eq!(
            d.ardelrange(b"a", [(0u64, u64::MAX - 1)].into_iter())
                .expect("an array"),
            2
        );
        assert_eq!(d.kind_of(b"a"), None, "and the key went with them");
    }

    #[test]
    fn a_range_read_answers_for_every_position_including_the_holes() {
        let mut d = db();
        set(&mut d, b"a", 1, &[b"x"]);
        let mut got = Vec::new();
        let len = d
            .argetrange(b"a", 0, 3, |el| {
                got.push(el.map(|e| {
                    let mut buf = [0u8; ELEMENT_MAX];
                    e.text(&mut buf).to_vec()
                }));
            })
            .expect("an array");
        assert_eq!(len, 4);
        assert_eq!(got, vec![None, Some(b"x".to_vec()), None, None]);

        // And backwards, when the ends come in the other order.
        let mut back = Vec::new();
        d.argetrange(b"a", 3, 0, |el| back.push(el.is_some()))
            .expect("an array");
        assert_eq!(back, vec![false, false, true, false]);
    }

    /// A missing key reads like an array of nothing but holes.
    #[test]
    fn a_range_read_of_a_missing_key_is_all_holes() {
        let mut d = db();
        let mut n = 0;
        let len = d
            .argetrange(b"nope", 5, 9, |el| {
                assert!(el.is_none());
                n += 1;
            })
            .expect("no key");
        assert_eq!(len, 5);
        assert_eq!(n, 5);
    }

    /// The million position limit is an error and not a quiet trim, so that a
    /// client asking for too much finds out rather than getting a short answer
    /// it thinks is complete.
    #[test]
    fn a_range_read_over_the_limit_is_refused() {
        let mut d = db();
        let e = d.argetrange(b"a", 0, GETRANGE_MAX, |_| {}).unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), "range exceeds maximum of 1000000 items");
        // One under the line is fine, and it is the positions that are counted
        // and not the elements, so this walks a million holes.
        let mut n = 0u64;
        d.argetrange(b"a", 0, GETRANGE_MAX - 1, |_| n += 1)
            .expect("no key");
        assert_eq!(n, GETRANGE_MAX);
    }

    #[test]
    fn every_command_refuses_a_key_holding_something_else() {
        let mut d = db();
        d.set_plain(b"s", b"v").expect("a string");
        assert_eq!(d.arlen(b"s").unwrap_err().code(), Code::WrongType);
        assert_eq!(d.arcount(b"s").unwrap_err().code(), Code::WrongType);
        assert_eq!(d.arget(b"s", 0).unwrap_err().code(), Code::WrongType);
        assert_eq!(
            d.arset(b"s", 0, [b"x".as_ref()].into_iter())
                .unwrap_err()
                .code(),
            Code::WrongType
        );
        assert_eq!(
            d.ardel(b"s", [0u64].into_iter()).unwrap_err().code(),
            Code::WrongType
        );
        assert_eq!(
            d.ardelrange(b"s", [(0u64, 1u64)].into_iter())
                .unwrap_err()
                .code(),
            Code::WrongType
        );
        assert_eq!(
            d.argetrange(b"s", 0, 1, |_| {}).unwrap_err().code(),
            Code::WrongType
        );
    }

    /// An index is unsigned, and the numbers a list would take are errors here.
    #[test]
    fn an_index_is_read_the_way_redis_reads_one() {
        for good in [
            (&b"0"[..], 0u64),
            (b"1", 1),
            (b"18446744073709551614", INDEX_MAX),
        ] {
            assert_eq!(parse_index(good.0).expect("an index"), good.1);
        }
        for bad in [
            &b"-1"[..],
            b"+1",
            b"01",
            b"",
            b" 1",
            b"1 ",
            b"1.0",
            b"one",
            // The top of the space is reserved for the insert cursor.
            b"18446744073709551615",
            b"18446744073709551616",
            b"99999999999999999999999",
        ] {
            let e = parse_index(bad).unwrap_err();
            assert_eq!(e.code(), Code::Invalid, "{}", String::from_utf8_lossy(bad));
            assert_eq!(e.message(), BAD_INDEX);
        }
    }

    fn insert(d: &mut Keyspace, key: &[u8], vals: &[&[u8]]) -> u64 {
        d.arinsert(key, vals.iter().copied()).expect("an array")
    }

    /// What a client would see back from `ARSCAN`.
    fn scan(d: &mut Keyspace, key: &[u8], start: u64, end: u64, limit: u64) -> Vec<(u64, Vec<u8>)> {
        let mut got = Vec::new();
        let n = d
            .arscan(key, start, end, limit, |i, el| {
                let mut buf = [0u8; ELEMENT_MAX];
                got.push((i, el.text(&mut buf).to_vec()));
            })
            .expect("an array");
        assert_eq!(n as usize, got.len(), "the count matches what it emitted");
        got
    }

    /// What a client would see back from `ARLASTITEMS`, holes included.
    fn last(d: &mut Keyspace, key: &[u8], count: u64, rev: bool) -> Vec<Option<Vec<u8>>> {
        let mut got = Vec::new();
        let n = d
            .arlastitems(key, count, rev, |el| {
                got.push(el.map(|e| {
                    let mut buf = [0u8; ELEMENT_MAX];
                    e.text(&mut buf).to_vec()
                }));
            })
            .expect("an array");
        assert_eq!(n as usize, got.len());
        got
    }

    #[test]
    fn an_insert_makes_the_key_and_walks_the_cursor_along() {
        let mut d = db();
        assert_eq!(d.arnext(b"a").expect("no key"), Some(0), "and nothing made");
        assert_eq!(d.kind_of(b"a"), None);

        assert_eq!(insert(&mut d, b"a", &[b"x", b"y"]), 1);
        assert_eq!(d.kind_of(b"a"), Some(Kind::Array));
        assert_eq!(d.arnext(b"a").expect("an array"), Some(2));
        assert_eq!(insert(&mut d, b"a", &[b"z"]), 2);
        assert_eq!(read(&mut d, b"a", 2).as_deref(), Some(&b"z"[..]));
        assert_eq!(d.arcount(b"a").expect("an array"), 3);
    }

    /// A seek says where the next append goes, and seeking to zero puts the
    /// cursor back to where it was before anything was appended.
    #[test]
    fn a_seek_moves_the_cursor_and_a_missing_key_has_none_to_move() {
        let mut d = db();
        assert!(!d.arseek(b"a", 10).expect("no key"), "and none was made");
        assert_eq!(d.kind_of(b"a"), None);

        insert(&mut d, b"a", &[b"x"]);
        assert!(d.arseek(b"a", 10).expect("an array"));
        assert_eq!(d.arnext(b"a").expect("an array"), Some(10));
        assert_eq!(insert(&mut d, b"a", &[b"y"]), 10);
        assert!(d.arseek(b"a", 0).expect("an array"));
        assert_eq!(d.arnext(b"a").expect("an array"), Some(0));
        assert_eq!(insert(&mut d, b"a", &[b"Y"]), 0, "back over the first one");
    }

    /// The top of the space is a state the cursor can be left in, and once it is
    /// there `ARNEXT` has no honest answer and an append has nowhere to go.
    #[test]
    fn the_cursor_can_be_parked_where_nothing_more_will_fit() {
        let mut d = db();
        insert(&mut d, b"a", &[b"x"]);
        assert!(d.arseek(b"a", u64::MAX).expect("an array"));
        assert_eq!(d.arnext(b"a").expect("an array"), None);
        let e = d.arinsert(b"a", [b"y".as_ref()].into_iter()).unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), "insert index overflow");

        // And the top index itself is reachable, one below that.
        assert!(d.arseek(b"a", INDEX_MAX).expect("an array"));
        assert_eq!(insert(&mut d, b"a", &[b"y"]), INDEX_MAX);
        assert_eq!(d.arnext(b"a").expect("an array"), None);
    }

    /// Only `ARSEEK` takes the reserved top of the index space, and it takes it
    /// because a rewritten command has to be able to say it.
    #[test]
    fn the_reserved_index_is_readable_for_one_command_only() {
        assert_eq!(
            parse_seek_index(b"18446744073709551615").expect("the top"),
            u64::MAX
        );
        assert_eq!(
            parse_index(b"18446744073709551615").unwrap_err().message(),
            BAD_INDEX
        );
        assert_eq!(
            parse_seek_index(b"18446744073709551616")
                .unwrap_err()
                .message(),
            BAD_INDEX
        );
        assert_eq!(parse_seek_index(b"-1").unwrap_err().message(), BAD_INDEX);
        assert_eq!(parse_seek_index(b"0").expect("zero"), 0);
    }

    #[test]
    fn a_ring_wraps_and_the_key_holds_no_more_than_its_size() {
        let mut d = db();
        let vals: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d", b"e"];
        assert_eq!(d.arring(b"r", 3, vals.into_iter()).expect("an array"), 1);
        assert_eq!(d.arlen(b"r").expect("an array"), 3);
        assert_eq!(d.arcount(b"r").expect("an array"), 3);
        assert_eq!(read(&mut d, b"r", 0).as_deref(), Some(&b"d"[..]));
        assert_eq!(read(&mut d, b"r", 1).as_deref(), Some(&b"e"[..]));
        assert_eq!(read(&mut d, b"r", 2).as_deref(), Some(&b"c"[..]));

        // The three it holds, oldest first, which is what the ring is for.
        assert_eq!(
            last(&mut d, b"r", 3, false),
            vec![
                Some(b"c".to_vec()),
                Some(b"d".to_vec()),
                Some(b"e".to_vec())
            ]
        );
        assert_eq!(last(&mut d, b"r", 1, true), vec![Some(b"e".to_vec())]);
    }

    #[test]
    fn the_last_items_of_a_missing_key_are_none_at_all() {
        let mut d = db();
        assert_eq!(last(&mut d, b"nope", 10, false), Vec::new());
        set(&mut d, b"a", 0, &[b"x"]);
        assert_eq!(last(&mut d, b"a", 0, false), Vec::new());
    }

    #[test]
    fn a_scan_skips_the_holes_and_stops_at_the_limit() {
        let mut d = db();
        d.armset(
            b"a",
            [
                (0u64, b"x".as_ref()),
                (7, b"y".as_ref()),
                (1_000_000_000, b"z".as_ref()),
            ]
            .into_iter(),
        )
        .expect("an array");

        let all = vec![
            (0, b"x".to_vec()),
            (7, b"y".to_vec()),
            (1_000_000_000, b"z".to_vec()),
        ];
        // The whole index space, which ARGETRANGE would refuse and this one
        // answers in three visits.
        assert_eq!(scan(&mut d, b"a", 0, INDEX_MAX, u64::MAX), all);
        let mut back = all.clone();
        back.reverse();
        assert_eq!(scan(&mut d, b"a", INDEX_MAX, 0, u64::MAX), back);
        assert_eq!(scan(&mut d, b"a", 0, INDEX_MAX, 2), all[..2].to_vec());
        assert_eq!(scan(&mut d, b"a", 1, 6, u64::MAX), Vec::new());
        assert_eq!(scan(&mut d, b"nope", 0, INDEX_MAX, u64::MAX), Vec::new());
    }

    #[test]
    fn the_cursor_commands_refuse_a_key_holding_something_else() {
        let mut d = db();
        d.set_plain(b"s", b"v").expect("a string");
        assert_eq!(d.arnext(b"s").unwrap_err().code(), Code::WrongType);
        assert_eq!(d.arseek(b"s", 1).unwrap_err().code(), Code::WrongType);
        assert_eq!(
            d.arinsert(b"s", [b"x".as_ref()].into_iter())
                .unwrap_err()
                .code(),
            Code::WrongType
        );
        assert_eq!(
            d.arring(b"s", 4, [b"x".as_ref()].into_iter())
                .unwrap_err()
                .code(),
            Code::WrongType
        );
        assert_eq!(
            d.arlastitems(b"s", 1, false, |_| {}).unwrap_err().code(),
            Code::WrongType
        );
        assert_eq!(
            d.arscan(b"s", 0, 1, 1, |_, _| {}).unwrap_err().code(),
            Code::WrongType
        );
    }

    fn op(d: &mut Keyspace, key: &[u8], op: Op, want: &[u8]) -> Aggregate {
        d.arop(key, 0, INDEX_MAX, op, want).expect("an array")
    }

    #[test]
    fn the_arithmetic_ops_read_what_they_can_and_ignore_the_rest() {
        let mut d = db();
        set(&mut d, b"a", 0, &[b"1", b"2.5", b"word", b"-4"]);
        assert_eq!(op(&mut d, b"a", Op::Sum, b""), Aggregate::Num(-0.5));
        assert_eq!(op(&mut d, b"a", Op::Min, b""), Aggregate::Num(-4.0));
        assert_eq!(op(&mut d, b"a", Op::Max, b""), Aggregate::Num(2.5));
        assert_eq!(op(&mut d, b"a", Op::Used, b""), Aggregate::Int(4));
        assert_eq!(op(&mut d, b"a", Op::Match, b"word"), Aggregate::Int(1));
        assert_eq!(op(&mut d, b"a", Op::Match, b"1"), Aggregate::Int(1));
        assert_eq!(op(&mut d, b"a", Op::Match, b"1.0"), Aggregate::Int(0));

        // A range holding nothing numeric is a null and not a zero, because
        // zero is an answer and there is no answer.
        set(&mut d, b"w", 0, &[b"word", b"other"]);
        assert_eq!(op(&mut d, b"w", Op::Sum, b""), Aggregate::None);
        assert_eq!(op(&mut d, b"w", Op::Used, b""), Aggregate::Int(2));

        // A missing key counts as nothing, which is a number for the two that
        // count and a null for the ones that aggregate.
        assert_eq!(op(&mut d, b"nope", Op::Used, b""), Aggregate::Int(0));
        assert_eq!(op(&mut d, b"nope", Op::Match, b"x"), Aggregate::Int(0));
        assert_eq!(op(&mut d, b"nope", Op::Sum, b""), Aggregate::None);
        assert_eq!(op(&mut d, b"nope", Op::And, b""), Aggregate::None);
    }

    /// The bitwise ops take the whole part of a float and skip anything that
    /// cannot be one, so a word in the middle of a range does not turn an AND
    /// into a zero.
    #[test]
    fn the_bitwise_ops_truncate_and_skip() {
        let mut d = db();
        set(&mut d, b"a", 0, &[b"12", b"10.9", b"word"]);
        assert_eq!(op(&mut d, b"a", Op::And, b""), Aggregate::Int(8));
        assert_eq!(op(&mut d, b"a", Op::Or, b""), Aggregate::Int(14));
        assert_eq!(op(&mut d, b"a", Op::Xor, b""), Aggregate::Int(6));

        // Negative floats truncate towards zero and not downwards, and one that
        // will not fit an integer at all is left out.
        set(&mut d, b"b", 0, &[b"-2.7", b"1e30"]);
        assert_eq!(op(&mut d, b"b", Op::Xor, b""), Aggregate::Int(-2));
        set(&mut d, b"c", 0, &[b"1e30"]);
        assert_eq!(op(&mut d, b"c", Op::And, b""), Aggregate::None);
    }

    /// The range is a range, so an op can be asked about part of an array.
    #[test]
    fn an_op_only_reads_the_range_it_was_given() {
        let mut d = db();
        set(&mut d, b"a", 0, &[b"1", b"2", b"3", b"4"]);
        assert_eq!(
            d.arop(b"a", 1, 2, Op::Sum, b"").expect("an array"),
            Aggregate::Num(5.0)
        );
        // And the ends may come either way round, because none of these care
        // which order they see the elements in.
        assert_eq!(
            d.arop(b"a", 2, 1, Op::Sum, b"").expect("an array"),
            Aggregate::Num(5.0)
        );
        assert_eq!(
            d.arop(b"a", 100, 200, Op::Used, b"").expect("an array"),
            Aggregate::Int(0)
        );
    }

    #[test]
    fn the_info_describes_the_shape_and_a_missing_key_is_an_error() {
        let mut d = db();
        assert_eq!(
            d.arinfo(b"nope", false).unwrap_err().message(),
            "no such key"
        );

        // Forty consecutive positions is one dense slice, and one element far
        // away is a second slice holding a single entry.
        set(
            &mut d,
            b"a",
            0,
            &(0..40).map(|_| b"v".as_ref()).collect::<Vec<_>>(),
        );
        set(&mut d, b"a", 100_000, &[b"far"]);
        d.arinsert(b"a", [b"x".as_ref()].into_iter()).expect("room");

        let info = d.arinfo(b"a", true).expect("an array");
        assert_eq!(info.count, 41);
        assert_eq!(info.len, 100_001);
        assert_eq!(info.next_insert, 1, "the append landed on zero");
        assert_eq!(info.slices, 2);
        assert_eq!(info.slice_size, 4096);
        assert!(info.directory_size >= info.slices);
        assert_eq!(info.dense_slices, 1);
        assert_eq!(info.sparse_slices, 1);
        assert_eq!(info.avg_dense_size, 40.0);
        assert_eq!(info.avg_dense_fill, 1.0);
        assert!(info.avg_sparse_size >= 1.0);

        // Without FULL the per layout numbers are not walked for and read zero.
        let cheap = d.arinfo(b"a", false).expect("an array");
        assert_eq!(cheap.count, 41);
        assert_eq!(cheap.dense_slices, 0);
        assert_eq!(cheap.avg_dense_fill, 0.0);
    }

    /// An array is a body like any other, so the shared key commands work on it.
    #[test]
    fn it_expires_and_copies_like_every_other_body() {
        let mut d = db();
        set(&mut d, b"a", 0, &[b"x"]);
        assert!(d.set_expiry(b"a", Some(d.clock.now_ms() + 10_000)));
        assert_eq!(read(&mut d, b"a", 0).as_deref(), Some(&b"x"[..]));
        assert!(d.persist(b"a"));

        assert_eq!(d.copy(b"a", b"b", false), crate::Moved::Ok);
        assert_eq!(d.kind_of(b"b"), Some(Kind::Array));
        set(&mut d, b"b", 1, &[b"y"]);
        assert_eq!(
            d.arcount(b"a").expect("an array"),
            1,
            "the source is its own"
        );
        assert_eq!(d.arcount(b"b").expect("an array"), 2);
        assert_eq!(d.encoding_name(b"a"), Some("sliced-array"));
    }
}
