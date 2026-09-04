//! The list commands.
//!
//! One method per Redis command on [`Keyspace`], the same arrangement the set
//! and hash commands use and for the same reason: a key belongs to the database
//! and not to a type, so `LPUSH` against a string has to be able to see that it
//! is a string. The list itself, and the choice between the two representations
//! it can be in, is [`crate::list`]. This file is what the wire and the embedded
//! API both call.
//!
//! # Indexes
//!
//! Every index a client sends is signed and counts from the back when it is
//! negative, and every range is inclusive at both ends. Both of those are Redis
//! rules that no structure below here should have to know about, so they are
//! turned into an offset and a count in `at` and `window` and the list sees
//! nothing but `usize`.
//!
//! # An empty list is not a list
//!
//! The key goes when the last element does, whether that was `LPOP`, `LREM` or
//! `LTRIM`. Redis has the same rule for every collection, and it is the reason
//! `EXISTS` answers zero after a list is emptied rather than one.

use yo_common::{Code, Error, Result};

use crate::keyspace::Keyspace;
use crate::list::{Element, List};
use crate::strings;
use crate::value::{self, Kind};

/// What `LSET` and `LINSERT` say about a key that is not there.
///
/// Redis's words, because they go on the wire verbatim.
const NO_KEY: &str = "no such key";

/// What `LSET` says about an index past the end.
const OUT_OF_RANGE: &str = "index out of range";

/// What `LPOS` says about a rank of zero.
///
/// Read off a running 8.8 rather than written from memory, because the older
/// wording of this message is still all over the internet and clients match on
/// the text. Zero is the one rank with no reading: 1 is the first match from
/// the front, -1 the first from the back, and 0 would have to mean neither.
const ZERO_RANK: &str = "RANK can't be zero: use 1 to start from the first match, 2 from the second ... or use negative to start from the end of the list";

/// Which end a command works from.
///
/// `LPUSH` and `RPUSH` are one method, and so are `LPOP` and `RPOP`, because
/// the difference between each pair is this and nothing else. `LMOVE` needs two
/// of them and would need two flags whichever way this was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    /// The head, which is where `LPUSH` puts things and `LPOP` takes them from.
    Left,
    /// The tail.
    Right,
}

impl End {
    /// Whether this is the head.
    #[inline]
    #[must_use]
    pub const fn is_left(self) -> bool {
        matches!(self, End::Left)
    }
}

/// Which order `LMOVEM` leaves the elements it moved in.
///
/// It only makes a difference when both ends are the same one, which is worth
/// knowing before reading any further. Taking from the left and putting on the
/// right hands the block over in the order it was in either way, and so does
/// taking from the right and putting on the left. It is `LEFT LEFT` and `RIGHT
/// RIGHT` where the two answers come apart, because those are the cases where
/// each element lands in front of the one before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// `OBO`. Exactly as if `LMOVE` had been sent that many times, so a
    /// destination end that puts each element in front of the last leaves the
    /// block reversed.
    OneByOne,
    /// `BULK`. The block keeps the order it had in the source whatever the two
    /// ends are.
    Bulk,
}

/// Everything `LMOVEM` needs to know beyond the two keys.
///
/// The five of them travel together because they are all parsed out of one
/// command and none of them means anything without the others. Keeping them in
/// a struct is also what stops the call from being eight positional arguments
/// where three of them are ends and flags that read the same at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// The end of the source to take from.
    pub from: End,
    /// The end of the destination to put on.
    pub to: End,
    /// How many to move. The 5 argument form of the command means one.
    pub count: usize,
    /// `EXACTLY` rather than `COUNT`: all of them or none of them.
    pub exactly: bool,
    /// Which order they are left in.
    pub order: Order,
}

impl Keyspace {
    /// `LPUSH key element [element ...]` and `RPUSH`. Answers the new length.
    ///
    /// The elements arrive as an iterator rather than a slice, the same as
    /// `SADD`, because the wire layer has them as positions in the connection's
    /// read buffer and collecting them into a slice would be an allocation on a
    /// shard thread.
    ///
    /// They go in one at a time, so `LPUSH k a b c` leaves the list holding
    /// `c b a`. That reads like a bug and is not: each element in turn is put
    /// at the head, and the last one sent ends up in front.
    pub fn push<'v>(
        &mut self,
        key: &[u8],
        end: End,
        values: impl Iterator<Item = &'v [u8]> + Clone,
    ) -> Result<usize> {
        for v in values.clone() {
            strings::check_len(key, v.len())?;
        }
        let at = match self.list_slot(key)? {
            Some(at) => at,
            None => {
                // `LPUSH k` with no elements does not make a key. The wire
                // parser rejects it on arity before it gets here, but the
                // embedded API has no parser in front of it and an empty list
                // left behind would be a key that exists and holds nothing.
                if values.clone().next().is_none() {
                    return Ok(0);
                }
                self.new_list(key)
            }
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(at)
            .expect("the record points at its body");
        for v in values {
            if end.is_left() {
                list.push_front(v, &limits);
            } else {
                list.push_back(v, &limits);
            }
        }
        Ok(list.len())
    }

    /// `LPUSHX key element [element ...]` and `RPUSHX`.
    ///
    /// The same as [`Keyspace::push`] except that a key which is not there
    /// stays not there, and the answer is zero.
    pub fn pushx<'v>(
        &mut self,
        key: &[u8],
        end: End,
        values: impl Iterator<Item = &'v [u8]> + Clone,
    ) -> Result<usize> {
        if self.list_slot(key)?.is_none() {
            return Ok(0);
        }
        self.push(key, end, values)
    }

    /// `LPOP key` and `RPOP key`, one element.
    ///
    /// This allocates, because the element it answers with is the element it
    /// just took out of the structure that was holding it, the same bind
    /// `SPOP` is in. [`Keyspace::pop_into`] is the version the wire uses, which
    /// copies straight into the reply buffer instead.
    pub fn pop(&mut self, key: &[u8], end: End) -> Result<Option<Vec<u8>>> {
        let Some(at) = self.list_slot(key)? else {
            return Ok(None);
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(at)
            .expect("the record points at its body");
        let got = if end.is_left() {
            list.pop_front(&limits)
        } else {
            list.pop_back(&limits)
        };
        if list.is_empty() {
            self.drop_key(key);
        }
        Ok(got)
    }

    /// `LPOP key count` and `RPOP key count`, straight into the reply.
    ///
    /// Answers how many were taken. `f` is called with each element in the
    /// order the reply wants them, which for `LPOP` with a count is head first
    /// and for `RPOP` is tail first, and it is called before the element is
    /// dropped so nothing has to be copied to a `Vec` on the way (Y18).
    ///
    /// A count larger than the list takes the whole list, and the key goes with
    /// it.
    pub fn pop_into<F>(&mut self, key: &[u8], end: End, count: usize, mut f: F) -> Result<usize>
    where
        F: FnMut(Element<'_>),
    {
        let Some(at) = self.list_slot(key)? else {
            return Ok(0);
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(at)
            .expect("the record points at its body");
        let take = count.min(list.len());
        for _ in 0..take {
            // The element is handed over while it is still in the list and
            // dropped right after, which is why this is a loop of read then
            // drop rather than a loop of `pop_front`. A `pop_front` would build
            // a `Vec` per element for no reason other than to hand it back.
            let e = if end.is_left() {
                list.front()
            } else {
                list.back()
            };
            f(e.expect("a list shorter than it says it is"));
            if end.is_left() {
                list.drop_front(&limits);
            } else {
                list.drop_back(&limits);
            }
        }
        if list.is_empty() {
            self.drop_key(key);
        }
        Ok(take)
    }

    /// `LLEN key`. Zero for a key that is not there.
    pub fn llen(&mut self, key: &[u8]) -> Result<usize> {
        Ok(match self.list_slot(key)? {
            Some(at) => self.list_at(at).len(),
            None => 0,
        })
    }

    /// `LINDEX key index`, counting from the back when the index is negative.
    pub fn lindex(&mut self, key: &[u8], index: i64) -> Result<Option<Element<'_>>> {
        let Some(slot) = self.list_slot(key)? else {
            return Ok(None);
        };
        let list = self.list_at(slot);
        Ok(at(index, list.len()).and_then(|i| list.get(i)))
    }

    /// `LRANGE key start stop`, both ends inclusive and both able to be
    /// negative.
    ///
    /// The answer borrows the database for as long as it is alive, so the
    /// caller walks it straight into the reply rather than collecting it (Y18).
    /// A key that is not there is an empty range and not a nil, which is what
    /// Redis replies and is the one place a list differs from a set.
    pub fn lrange(
        &mut self,
        key: &[u8],
        start: i64,
        stop: i64,
    ) -> Result<impl Iterator<Item = Element<'_>>> {
        let slot = self.list_slot(key)?;
        let list = slot.map(|at| self.list_at(at));
        let (from, count) = match list {
            Some(l) => window(start, stop, l.len()),
            None => (0, 0),
        };
        Ok(list
            .into_iter()
            .flat_map(move |l| l.range(from, count))
            .take(count))
    }

    /// `LSET key index element`.
    ///
    /// Two errors and no boolean, because both of them are errors on the wire:
    /// `no such key` for a missing key and `index out of range` for an index
    /// the list does not reach. A list is never empty, so those really are the
    /// only two ways to miss.
    pub fn lset(&mut self, key: &[u8], index: i64, value: &[u8]) -> Result<()> {
        strings::check_len(key, value.len())?;
        let Some(slot) = self.list_slot(key)? else {
            return Err(Error::new(Code::Invalid, NO_KEY));
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(slot)
            .expect("the record points at its body");
        let Some(i) = at(index, list.len()) else {
            return Err(Error::new(Code::Invalid, OUT_OF_RANGE));
        };
        if !list.set(i, value, &limits) {
            return Err(Error::new(Code::Invalid, OUT_OF_RANGE));
        }
        Ok(())
    }

    /// `LINSERT key BEFORE|AFTER pivot element`.
    ///
    /// The new length, or `-1` when the pivot is not in the list, or `0` when
    /// the key is not there. Three answers in one signed number is Redis's
    /// choice and it is a bad one, but it is on the wire and cannot be changed.
    pub fn linsert(&mut self, key: &[u8], before: bool, pivot: &[u8], value: &[u8]) -> Result<i64> {
        strings::check_len(key, value.len())?;
        let Some(slot) = self.list_slot(key)? else {
            return Ok(0);
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(slot)
            .expect("the record points at its body");
        Ok(match list.insert_at_pivot(pivot, value, before, &limits) {
            Some(len) => len as i64,
            None => -1,
        })
    }

    /// `LREM key count element`. Answers how many went.
    ///
    /// A positive count removes that many from the front, a negative one that
    /// many from the back, and zero removes all of them. The key goes if the
    /// list ends up empty.
    pub fn lrem(&mut self, key: &[u8], count: i64, value: &[u8]) -> Result<usize> {
        let Some(slot) = self.list_slot(key)? else {
            return Ok(0);
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(slot)
            .expect("the record points at its body");
        let gone = list.remove(count, value, &limits);
        if list.is_empty() {
            self.drop_key(key);
        }
        Ok(gone)
    }

    /// `LTRIM key start stop`, keeping the window and throwing the rest away.
    ///
    /// A window that selects nothing deletes the key, which is what an empty
    /// range means here: `LTRIM k 1 0` is the documented way to empty a list.
    pub fn ltrim(&mut self, key: &[u8], start: i64, stop: i64) -> Result<()> {
        let Some(slot) = self.list_slot(key)? else {
            return Ok(());
        };
        let limits = self.list_limits;
        let list = self
            .lists
            .get_mut(slot)
            .expect("the record points at its body");
        let (from, count) = window(start, stop, list.len());
        list.trim(from, count, &limits);
        if list.is_empty() {
            self.drop_key(key);
        }
        Ok(())
    }

    /// `LPOS key element [RANK rank] [COUNT count] [MAXLEN len]`.
    ///
    /// The positions land in `out`, which the caller supplies and which is
    /// cleared first, because this runs on a shard thread and a shard thread
    /// that allocates aborts. `count` of zero means every match and `maxlen` of
    /// zero means no limit on how far to look, both of which are Redis's
    /// spellings for no limit.
    ///
    /// # Errors
    ///
    /// A rank of zero, which has no reading. Everything else about a missing
    /// key or a missing element is an empty answer rather than an error.
    pub fn lpos(
        &mut self,
        key: &[u8],
        value: &[u8],
        rank: i64,
        count: usize,
        maxlen: usize,
        out: &mut Vec<usize>,
    ) -> Result<()> {
        out.clear();
        if rank == 0 {
            return Err(Error::new(Code::Invalid, ZERO_RANK));
        }
        self.lpos_into(key, value, rank, count, maxlen, |at| out.push(at))?;
        Ok(())
    }

    /// `LPOS`, with each position handed over as it is found.
    ///
    /// This is what the wire calls. The positions go straight into the reply
    /// buffer as they are discovered, so a `LPOS key x COUNT 0` over a list
    /// with ten thousand matches never builds a list of ten thousand numbers
    /// anywhere (Y18). Answers how many there were.
    ///
    /// # Errors
    ///
    /// A rank of zero, and `WRONGTYPE` for a key that is not a list.
    pub fn lpos_into<F>(
        &mut self,
        key: &[u8],
        value: &[u8],
        rank: i64,
        count: usize,
        maxlen: usize,
        mut found: F,
    ) -> Result<usize>
    where
        F: FnMut(usize),
    {
        if rank == 0 {
            return Err(Error::new(Code::Invalid, ZERO_RANK));
        }
        let Some(slot) = self.list_slot(key)? else {
            return Ok(0);
        };
        Ok(self
            .list_at(slot)
            .positions(value, rank, count, maxlen, &mut found))
    }

    /// `LMOVE src dst LEFT|RIGHT LEFT|RIGHT`, and `RPOPLPUSH` under it.
    ///
    /// Answers the element that moved, or nothing when the source is empty or
    /// missing. The destination is made if it is not there, and the source key
    /// goes if that was its last element.
    ///
    /// `src` and `dst` being the same key is not a special case to work around,
    /// it is `LMOVE k k LEFT RIGHT`, which is the documented way to rotate a
    /// list and is what a round robin scheduler is built out of. It falls out
    /// of taking the element before deciding where to put it.
    ///
    /// This is the one list command that has to copy an element, for the reason
    /// `SPOP` gives: the value it answers with no longer has a structure to
    /// borrow from. Moving the bytes from one list to the other without the
    /// copy would need both bodies borrowed at once, and the destination may be
    /// the source.
    ///
    /// The copy is not an allocation, though, which is the difference between
    /// this and the first version of it. The element goes into the database's
    /// one scratch buffer and the answer borrows that, so a queue that runs
    /// `RPOPLPUSH` in a loop does no allocator work at all after the first
    /// call, where before it did a malloc and a free per element. The answer
    /// borrows the database until the caller is done with it, which is what
    /// both callers want anyway: they write it to the reply and drop it.
    pub fn lmove(&mut self, src: &[u8], dst: &[u8], from: End, to: End) -> Result<Option<&[u8]>> {
        // The destination's type is checked before anything is taken, so that
        // `LMOVE list string LEFT LEFT` is a `WRONGTYPE` with the source
        // untouched rather than an element that has gone nowhere.
        self.list_slot(dst)?;
        // Taken out of the database and put back at the end of every path, so
        // that `pop_into` and `push` can have `&mut self` while the bytes are
        // in hand. The buffer is empty for the duration and nothing else looks
        // at it, so a path that returns early leaves it exactly as it found it.
        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let took = self.pop_into(src, from, 1, |e| e.write_to(&mut buf));
        let moved = match took {
            Ok(n) => n,
            Err(e) => {
                self.scratch = buf;
                return Err(e);
            }
        };
        if moved == 0 {
            self.scratch = buf;
            return Ok(None);
        }
        let pushed = self.push(dst, to, std::iter::once(buf.as_slice()));
        self.scratch = buf;
        pushed?;
        Ok(Some(&self.scratch))
    }

    /// `LMOVEM src dst LEFT|RIGHT LEFT|RIGHT [COUNT|EXACTLY n OBO|BULK]`.
    ///
    /// [`Keyspace::lmove`] for more than one element at a time, which Redis
    /// 8.10 added. `f` gets each element that moved, in the order it now sits in
    /// the destination, which is the order the reply wants. The count that comes
    /// back is how many that was, and zero means the reply is a nil rather than
    /// an empty array.
    ///
    /// [`Block::exactly`] is the `EXACTLY` spelling against the `COUNT` one: all
    /// of them or none of them, so a source shorter than the count moves nothing
    /// and answers zero. That is the whole difference and it is checked before
    /// anything is taken, which is the only way to make it true.
    ///
    /// # Why everything is taken before anything is put
    ///
    /// Because the destination is allowed to be the source. `LMOVEM k k LEFT
    /// RIGHT COUNT 2 BULK` is a rotation by two and has to work, the same way
    /// `LMOVE k k LEFT RIGHT` is a rotation by one. Popping the whole block
    /// first and pushing it afterwards gets that for nothing, where anything
    /// that interleaved the two would be reading a list it was writing.
    ///
    /// The block goes through the same scratch buffer [`Keyspace::lmove`] uses,
    /// with the row buffer next to it holding where each element ends in it, so
    /// moving a hundred elements is two buffers that were already there rather
    /// than a `Vec` per element. Both are taken out of the database for the
    /// duration, because the bytes have to be in hand while `push` has
    /// `&mut self`.
    pub fn lmovem<F>(&mut self, src: &[u8], dst: &[u8], b: Block, mut f: F) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        // Both types settled before a single element moves, so a WRONGTYPE
        // anywhere leaves both lists as they were. `llen` checks the source and
        // is wanted for `EXACTLY` anyway.
        self.list_slot(dst)?;
        let have = self.llen(src)?;
        if b.exactly && have < b.count {
            return Ok(0);
        }

        let mut buf = std::mem::take(&mut self.scratch);
        let mut ends = std::mem::take(&mut self.rows);
        buf.clear();
        ends.clear();
        let took = self.pop_into(src, b.from, b.count, |e| {
            e.write_to(&mut buf);
            ends.push(buf.len());
        });
        let moved = match took {
            Ok(n) => n,
            Err(e) => {
                self.scratch = buf;
                self.rows = ends;
                return Err(e);
            }
        };
        if moved == 0 {
            self.scratch = buf;
            self.rows = ends;
            return Ok(0);
        }

        // Where the destination order comes from. `pop_into` hands them over in
        // the order it took them, which is source order from the left and
        // reversed source order from the right. Bulk wants source order, so it
        // turns them back over when they came off the right. One by one wants
        // whatever repeated pushes at that end would have left, and a push at
        // the head puts each in front of the last, so it turns them over when
        // the destination end is the head. Those two conditions agree exactly
        // when the two ends differ, which is why the spelling only matters when
        // they are the same.
        let flip = match b.order {
            Order::Bulk => !b.from.is_left(),
            Order::OneByOne => b.to.is_left(),
        };
        let at = |i: usize| {
            let end = ends[i];
            let start = if i == 0 { 0 } else { ends[i - 1] };
            &buf[start..end]
        };
        let placed = |i: usize| if flip { moved - 1 - i } else { i };
        // And pushing in that order needs the same turn again at the head, for
        // the same reason: `push` is `LPUSH`, so the last one sent ends up in
        // front.
        let sent = |i: usize| placed(if b.to.is_left() { moved - 1 - i } else { i });

        let pushed = self.push(dst, b.to, (0..moved).map(|i| at(sent(i))));
        if pushed.is_ok() {
            for i in 0..moved {
                f(at(placed(i)));
            }
        }
        self.scratch = buf;
        self.rows = ends;
        pushed?;
        Ok(moved)
    }

    /// The slot `key`'s list is in, or `None` if there is no such key.
    #[inline]
    fn list_slot(&mut self, key: &[u8]) -> Result<Option<u32>> {
        self.live_slot(key, Kind::List)
    }

    /// The body in a slot the record pointed at.
    ///
    /// Panicking here means a record outlived its body, the same bug
    /// [`Keyspace::set_at`] is watching for.
    #[inline]
    fn list_at(&self, at: u32) -> &List {
        self.lists.get(at).expect("the record points at its body")
    }

    /// Make an empty list under `key` and answer which slot it went in.
    ///
    /// No hint, unlike a set. A list starts packed whatever is going into it,
    /// because the band it belongs in is decided by bytes rather than by count
    /// and the first push finds that out for free.
    fn new_list(&mut self, key: &[u8]) -> u32 {
        // The body and, every so often, the slab that holds it. See
        // `yo_alloc::first_touch` for why this is the one allocation a command
        // is allowed to make.
        let at = yo_alloc::first_touch(|| self.lists.insert(List::new()));
        let len = value::slot_record_len(false);
        self.write_rec(key, len, |out| {
            value::write_slot_record(out, Kind::List, at, None);
        });
        self.bodies += 1;
        at
    }
}

/// Turn a signed index into an offset from the front, or nothing if it misses.
///
/// A negative index counts from the back, so -1 is the last element. An index
/// that is still negative after that, or that reaches past the end, is a miss,
/// and the two are the same answer because `LINDEX` replies nil to both.
#[inline]
fn at(index: i64, len: usize) -> Option<usize> {
    let i = if index < 0 { len as i64 + index } else { index };
    (i >= 0 && (i as usize) < len).then_some(i as usize)
}

/// Turn an inclusive `start` and `stop` into an offset and a count.
///
/// Every out of range case clamps rather than erroring, which is Redis's rule
/// for `LRANGE` and `LTRIM` both: a start before the front is the front, a stop
/// past the end is the end, and a start after the stop is nothing at all.
#[inline]
fn window(start: i64, stop: i64, len: usize) -> (usize, usize) {
    let n = len as i64;
    let from = if start < 0 { (n + start).max(0) } else { start };
    let to = if stop < 0 { n + stop } else { stop.min(n - 1) };
    if from > to || from >= n || to < 0 {
        return (0, 0);
    }
    (from as usize, (to - from + 1) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Clock;
    use crate::list::Encoding;
    use yo_common::Code;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    fn rpush(d: &mut Keyspace, key: &[u8], values: &[&[u8]]) -> usize {
        d.push(key, End::Right, values.iter().copied())
            .expect("a list")
    }

    fn all(d: &mut Keyspace, key: &[u8]) -> Vec<String> {
        d.lrange(key, 0, -1)
            .expect("a list")
            .map(|e| String::from_utf8(e.to_vec()).expect("utf8 in these tests"))
            .collect()
    }

    #[test]
    fn pushing_to_a_key_that_is_not_there_makes_it() {
        let mut d = db();
        assert_eq!(rpush(&mut d, b"l", &[b"a", b"b", b"c"]), 3);
        assert_eq!(all(&mut d, b"l"), ["a", "b", "c"]);
        assert_eq!(d.llen(b"l").expect("a list"), 3);
    }

    #[test]
    fn lpush_puts_the_last_element_in_front() {
        let mut d = db();
        d.push(b"l", End::Left, [b"a".as_slice(), b"b", b"c"].into_iter())
            .expect("a list");
        assert_eq!(all(&mut d, b"l"), ["c", "b", "a"]);
    }

    #[test]
    fn pushing_nothing_does_not_make_a_key() {
        let mut d = db();
        let none: [&[u8]; 0] = [];
        assert_eq!(d.push(b"l", End::Left, none.into_iter()).expect("ok"), 0);
        assert_eq!(d.kind_of(b"l"), None);
    }

    #[test]
    fn pushx_only_pushes_to_a_list_that_is_there() {
        let mut d = db();
        assert_eq!(
            d.pushx(b"l", End::Right, [b"a".as_slice()].into_iter())
                .expect("ok"),
            0
        );
        assert_eq!(d.kind_of(b"l"), None);
        rpush(&mut d, b"l", &[b"a"]);
        assert_eq!(
            d.pushx(b"l", End::Right, [b"b".as_slice()].into_iter())
                .expect("ok"),
            2
        );
    }

    #[test]
    fn popping_the_last_element_takes_the_key_with_it() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"only"]);
        assert_eq!(
            d.pop(b"l", End::Left).expect("ok").as_deref(),
            Some(&b"only"[..])
        );
        assert_eq!(d.kind_of(b"l"), None);
        assert_eq!(d.pop(b"l", End::Left).expect("ok"), None);
    }

    #[test]
    fn a_count_pop_takes_from_the_end_it_was_asked_for() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c", b"d"]);
        let mut got = Vec::new();
        d.pop_into(b"l", End::Right, 2, |e| got.push(e.to_vec()))
            .expect("a list");
        assert_eq!(got, [b"d".to_vec(), b"c".to_vec()]);
        assert_eq!(all(&mut d, b"l"), ["a", "b"]);
    }

    #[test]
    fn a_count_pop_larger_than_the_list_empties_it() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b"]);
        let mut n = 0;
        assert_eq!(
            d.pop_into(b"l", End::Left, 99, |_| n += 1).expect("a list"),
            2
        );
        assert_eq!(n, 2);
        assert_eq!(d.kind_of(b"l"), None);
    }

    #[test]
    fn lrange_clamps_at_both_ends() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c"]);
        assert_eq!(all(&mut d, b"l"), ["a", "b", "c"]);
        let got: Vec<_> = d
            .lrange(b"l", -100, 100)
            .expect("a list")
            .map(|e| e.to_vec())
            .collect();
        assert_eq!(got.len(), 3);
        assert_eq!(d.lrange(b"l", 2, 1).expect("a list").count(), 0);
        assert_eq!(d.lrange(b"l", 5, 9).expect("a list").count(), 0);
        assert_eq!(d.lrange(b"nope", 0, -1).expect("no key").count(), 0);
    }

    #[test]
    fn lindex_counts_from_the_back_when_it_is_negative() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c"]);
        assert_eq!(
            d.lindex(b"l", 0).expect("ok").map(|e| e.to_vec()),
            Some(b"a".to_vec())
        );
        assert_eq!(
            d.lindex(b"l", -1).expect("ok").map(|e| e.to_vec()),
            Some(b"c".to_vec())
        );
        assert!(d.lindex(b"l", 3).expect("ok").is_none());
        assert!(d.lindex(b"l", -4).expect("ok").is_none());
        assert!(d.lindex(b"nope", 0).expect("ok").is_none());
    }

    #[test]
    fn lset_says_which_of_the_two_ways_it_missed() {
        let mut d = db();
        let e = d.lset(b"nope", 0, b"x").expect_err("no key");
        assert_eq!(e.message(), NO_KEY);
        rpush(&mut d, b"l", &[b"a", b"b"]);
        d.lset(b"l", -1, b"z").expect("in range");
        assert_eq!(all(&mut d, b"l"), ["a", "z"]);
        let e = d.lset(b"l", 9, b"x").expect_err("out of range");
        assert_eq!(e.message(), OUT_OF_RANGE);
    }

    #[test]
    fn linsert_has_three_answers() {
        let mut d = db();
        assert_eq!(d.linsert(b"nope", true, b"a", b"x").expect("ok"), 0);
        rpush(&mut d, b"l", &[b"a", b"c"]);
        assert_eq!(d.linsert(b"l", true, b"c", b"b").expect("ok"), 3);
        assert_eq!(all(&mut d, b"l"), ["a", "b", "c"]);
        assert_eq!(d.linsert(b"l", false, b"zz", b"x").expect("ok"), -1);
    }

    #[test]
    fn lrem_counts_from_the_end_the_sign_says() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"x", b"b", b"x", b"c", b"x"]);
        assert_eq!(d.lrem(b"l", 1, b"x").expect("ok"), 1);
        assert_eq!(all(&mut d, b"l"), ["a", "b", "x", "c", "x"]);
        assert_eq!(d.lrem(b"l", -1, b"x").expect("ok"), 1);
        assert_eq!(all(&mut d, b"l"), ["a", "b", "x", "c"]);
        assert_eq!(d.lrem(b"l", 0, b"x").expect("ok"), 1);
        assert_eq!(all(&mut d, b"l"), ["a", "b", "c"]);
    }

    #[test]
    fn removing_everything_takes_the_key() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"x", b"x"]);
        assert_eq!(d.lrem(b"l", 0, b"x").expect("ok"), 2);
        assert_eq!(d.kind_of(b"l"), None);
    }

    #[test]
    fn ltrim_keeps_the_window() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c", b"d", b"e"]);
        d.ltrim(b"l", 1, -2).expect("ok");
        assert_eq!(all(&mut d, b"l"), ["b", "c", "d"]);
    }

    #[test]
    fn an_empty_window_deletes_the_key() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b"]);
        d.ltrim(b"l", 1, 0).expect("ok");
        assert_eq!(d.kind_of(b"l"), None);
        d.ltrim(b"nope", 0, -1).expect("no key is not an error");
    }

    #[test]
    fn lpos_answers_where_and_how_many() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c", b"b", b"b"]);
        let mut out = Vec::new();
        d.lpos(b"l", b"b", 1, 0, 0, &mut out).expect("ok");
        assert_eq!(out, [1, 3, 4]);
        d.lpos(b"l", b"b", -1, 2, 0, &mut out).expect("ok");
        assert_eq!(out, [4, 3]);
        d.lpos(b"l", b"b", 1, 0, 2, &mut out).expect("ok");
        assert_eq!(out, [1]);
        d.lpos(b"l", b"zz", 1, 0, 0, &mut out).expect("ok");
        assert!(out.is_empty());
        d.lpos(b"nope", b"b", 1, 0, 0, &mut out).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn a_rank_of_zero_is_an_error() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a"]);
        let e = d
            .lpos(b"l", b"a", 0, 0, 0, &mut Vec::new())
            .expect_err("zero");
        assert_eq!(e.message(), ZERO_RANK);
    }

    /// The whole of `LMOVEM`'s ordering, which is the part of it that is not
    /// obvious. Every row was read off a live 8.10.1 before it was written here.
    ///
    /// The two spellings agree on the two rows where the ends differ and come
    /// apart on the two where they are the same, and that is the entire content
    /// of `OBO` against `BULK`.
    #[test]
    fn lmovem_orders_the_block_four_ways() {
        use End::{Left, Right};
        use Order::{Bulk, OneByOne};
        for (from, to, order, want, left) in [
            (Left, Right, OneByOne, ["a", "b"], ["c", "d", "e"]),
            (Left, Right, Bulk, ["a", "b"], ["c", "d", "e"]),
            (Left, Left, OneByOne, ["b", "a"], ["c", "d", "e"]),
            (Left, Left, Bulk, ["a", "b"], ["c", "d", "e"]),
            (Right, Left, OneByOne, ["d", "e"], ["a", "b", "c"]),
            (Right, Left, Bulk, ["d", "e"], ["a", "b", "c"]),
            (Right, Right, OneByOne, ["e", "d"], ["a", "b", "c"]),
            (Right, Right, Bulk, ["d", "e"], ["a", "b", "c"]),
        ] {
            let mut d = db();
            rpush(&mut d, b"s", &[b"a", b"b", b"c", b"d", b"e"]);
            let mut got = Vec::new();
            let b = Block {
                from,
                to,
                count: 2,
                exactly: false,
                order,
            };
            let n = d
                .lmovem(b"s", b"t", b, |v| {
                    got.push(String::from_utf8(v.to_vec()).expect("utf8 in these tests"));
                })
                .expect("two lists");
            let how = format!("{from:?} {to:?} {order:?}");
            assert_eq!(n, 2, "{how}");
            assert_eq!(got, want, "the reply for {how}");
            assert_eq!(all(&mut d, b"t"), want, "the destination for {how}");
            assert_eq!(all(&mut d, b"s"), left, "what is left for {how}");
        }
    }

    /// A block for the tests that are not about ordering, which all want the
    /// same one and only care about the ends and the count.
    fn block(from: End, to: End, count: usize, exactly: bool) -> Block {
        Block {
            from,
            to,
            count,
            exactly,
            order: Order::Bulk,
        }
    }

    /// `EXACTLY` is all of them or none, and the check happens before anything
    /// is taken, which is the only place it can happen.
    #[test]
    fn lmovem_exactly_takes_all_of_them_or_none() {
        let mut d = db();
        rpush(&mut d, b"s", &[b"a", b"b", b"c"]);
        let all_of_four = block(End::Left, End::Right, 4, true);
        let n = d
            .lmovem(b"s", b"t", all_of_four, |_| {
                panic!("nothing should have moved")
            })
            .expect("two lists");
        assert_eq!(n, 0);
        assert_eq!(
            all(&mut d, b"s"),
            ["a", "b", "c"],
            "the source is untouched"
        );
        assert_eq!(d.llen(b"t").expect("a list"), 0, "and nothing was made");

        // The same count with `COUNT` takes what there is.
        let mut got = Vec::new();
        let up_to_four = block(End::Left, End::Right, 4, false);
        let n = d
            .lmovem(b"s", b"t", up_to_four, |v| got.push(v.to_vec()))
            .expect("two lists");
        assert_eq!(n, 3);
        assert_eq!(got.len(), 3);
        assert!(!d.exists(b"s"), "an emptied source goes");
    }

    /// The destination is allowed to be the source, the same way it is for
    /// `LMOVE`, and there it is a rotation by more than one.
    #[test]
    fn lmovem_onto_itself_rotates_by_the_count() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c"]);
        d.lmovem(b"l", b"l", block(End::Left, End::Right, 2, false), |_| {})
            .expect("a list");
        assert_eq!(all(&mut d, b"l"), ["c", "a", "b"]);

        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c"]);
        d.lmovem(b"l", b"l", block(End::Right, End::Left, 2, false), |_| {})
            .expect("a list");
        assert_eq!(all(&mut d, b"l"), ["b", "c", "a"]);
    }

    /// A source that is not there moves nothing and is not an error, and the
    /// destination's type is settled before anything is taken.
    #[test]
    fn lmovem_checks_the_destination_before_taking_anything() {
        let mut d = db();
        rpush(&mut d, b"s", &[b"a", b"b"]);
        d.set_plain(b"str", b"v").expect("room");
        let two = block(End::Left, End::Right, 2, false);
        let e = d
            .lmovem(b"s", b"str", two, |_| {})
            .expect_err("the destination is a string");
        assert_eq!(e.code(), Code::WrongType);
        assert_eq!(all(&mut d, b"s"), ["a", "b"], "the source is untouched");

        let n = d
            .lmovem(b"nope", b"t", two, |_| {})
            .expect("a source that is not there is not an error");
        assert_eq!(n, 0);
    }

    #[test]
    fn lmove_between_two_keys_makes_the_second_one() {
        let mut d = db();
        rpush(&mut d, b"src", &[b"a", b"b"]);
        let got = d.lmove(b"src", b"dst", End::Right, End::Left).expect("ok");
        assert_eq!(got, Some(&b"b"[..]));
        assert_eq!(all(&mut d, b"src"), ["a"]);
        assert_eq!(all(&mut d, b"dst"), ["b"]);
    }

    #[test]
    fn lmove_onto_itself_rotates() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a", b"b", b"c"]);
        d.lmove(b"l", b"l", End::Right, End::Left).expect("ok");
        assert_eq!(all(&mut d, b"l"), ["c", "a", "b"]);
        d.lmove(b"l", b"l", End::Left, End::Right).expect("ok");
        assert_eq!(all(&mut d, b"l"), ["a", "b", "c"]);
    }

    #[test]
    fn lmove_from_a_key_that_is_not_there_does_nothing() {
        let mut d = db();
        assert_eq!(
            d.lmove(b"nope", b"dst", End::Left, End::Left).expect("ok"),
            None
        );
        assert_eq!(d.kind_of(b"dst"), None);
    }

    /// `RPOPLPUSH` in a loop is what a work queue is, so the element that moves
    /// must not cost a malloc and a free every time round. The first call is
    /// allowed to grow the scratch buffer and everything after it is not.
    #[test]
    fn lmove_stops_allocating_once_its_buffer_is_grown() {
        let mut d = db();
        rpush(&mut d, b"q", &[b"a", b"b", b"c"]);
        // Warm up. This one may grow the scratch buffer, and on a fresh
        // database it also makes the destination.
        d.lmove(b"q", b"q", End::Right, End::Left).expect("ok");
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..100 {
                d.lmove(b"q", b"q", End::Right, End::Left).expect("ok");
            }
        });
        assert_eq!(allocs, 0, "lmove allocated {allocs} times in a hundred");
        assert_eq!(all(&mut d, b"q"), ["b", "c", "a"]);
    }

    /// `LREM` used to build a `Vec` to hold the indices it was about to remove,
    /// and the count it is given is one in almost every use of it, so that was
    /// an allocation to hold a single `usize`.
    #[test]
    fn lrem_does_not_allocate_to_remove_a_handful() {
        let mut d = db();
        // Built up front, so the loop below is a hundred `LREM` calls and
        // nothing else. Pushing inside it would make the key over and over,
        // and making a key is an allocation this is not asking about.
        let many: Vec<&[u8]> = (0..100).map(|_| b"gone".as_slice()).collect();
        rpush(&mut d, b"l", &many);
        rpush(&mut d, b"l", &[b"keep"]);
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..100 {
                assert_eq!(d.lrem(b"l", 1, b"gone").expect("a list"), 1);
            }
        });
        assert_eq!(allocs, 0, "lrem allocated {allocs} times in a hundred");
        assert_eq!(all(&mut d, b"l"), ["keep"]);
    }

    /// And it still answers when the hits do not fit on the stack.
    #[test]
    fn lrem_with_more_hits_than_fit_inline_still_removes_all_of_them() {
        let mut d = db();
        let many: Vec<&[u8]> = (0..40).map(|_| b"x".as_slice()).collect();
        rpush(&mut d, b"l", &many);
        rpush(&mut d, b"l", &[b"keep"]);
        assert_eq!(d.lrem(b"l", 0, b"x").expect("a list"), 40);
        assert_eq!(all(&mut d, b"l"), ["keep"]);
    }

    #[test]
    fn lmove_checks_the_destination_before_taking_anything() {
        let mut d = db();
        rpush(&mut d, b"src", &[b"a"]);
        d.set_plain(b"dst", b"a string").expect("room");
        let e = d
            .lmove(b"src", b"dst", End::Left, End::Left)
            .expect_err("the destination is a string");
        assert_eq!(e.code(), Code::WrongType);
        assert_eq!(all(&mut d, b"src"), ["a"]);
    }

    #[test]
    fn every_command_says_wrongtype_against_a_string() {
        let mut d = db();
        d.set_plain(b"s", b"a string").expect("room");
        assert_eq!(
            d.push(b"s", End::Left, [b"x".as_slice()].into_iter())
                .expect_err("a string")
                .code(),
            Code::WrongType
        );
        assert_eq!(d.llen(b"s").expect_err("a string").code(), Code::WrongType);
        assert_eq!(
            d.pop(b"s", End::Left).expect_err("a string").code(),
            Code::WrongType
        );
        assert_eq!(
            d.lset(b"s", 0, b"x").expect_err("a string").code(),
            Code::WrongType
        );
        assert_eq!(
            d.ltrim(b"s", 0, -1).expect_err("a string").code(),
            Code::WrongType
        );
    }

    #[test]
    fn a_list_is_a_list_to_the_rest_of_the_keyspace() {
        let mut d = db();
        rpush(&mut d, b"l", &[b"a"]);
        assert_eq!(d.kind_of(b"l").map(|k| k.name()), Some("list"));
        assert_eq!(d.encoding_name(b"l"), Some(Encoding::Listpack.name()));
        assert!(d.exists(b"l"));
        assert!(d.drop_key(b"l"));
        assert_eq!(d.kind_of(b"l"), None);
    }

    #[test]
    fn a_list_can_be_given_a_deadline_and_reaped() {
        let mut d = Keyspace::with_clock(Clock::fixed(1_000));
        rpush(&mut d, b"l", &[b"a", b"b"]);
        assert!(d.set_expiry(b"l", Some(1_500)));
        assert_eq!(all(&mut d, b"l"), ["a", "b"]);
        d.clock_mut().advance(1_000);
        assert_eq!(d.llen(b"l").expect("gone"), 0);
        assert_eq!(d.kind_of(b"l"), None);
    }

    #[test]
    fn a_big_list_is_chunked_and_still_answers_the_same() {
        let mut d = db();
        let value = vec![b'x'; 400];
        for i in 0..40 {
            let mut v = value.clone();
            v.extend_from_slice(format!("{i}").as_bytes());
            d.push(b"l", End::Right, [v.as_slice()].into_iter())
                .expect("a list");
        }
        assert_eq!(d.encoding_name(b"l"), Some(Encoding::Quicklist.name()));
        assert_eq!(d.llen(b"l").expect("a list"), 40);
        let last = d.lindex(b"l", -1).expect("ok").expect("in range").to_vec();
        assert!(last.ends_with(b"39"));
        d.ltrim(b"l", 0, 0).expect("ok");
        assert_eq!(d.llen(b"l").expect("a list"), 1);
        assert_eq!(d.encoding_name(b"l"), Some(Encoding::Listpack.name()));
    }
}
