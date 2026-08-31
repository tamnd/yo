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

use yo_common::{Code, Error, Result};

use crate::array::{Array, Element, INDEX_MAX};
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
    if n > INDEX_MAX {
        return Err(bad());
    }
    Ok(n)
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
