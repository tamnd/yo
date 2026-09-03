//! A run of packed elements with a cursor at each end.
//!
//! This is what a large list is made of. A [`Listpack`](crate::listpack) is a
//! blob with a header and a terminator, and taking the first element out of one
//! means moving every byte behind it left. That is what a quicklist does on
//! every `LPOP` and it is most of why aki lost that row. A chunk holds the same
//! entries in the same encoding and puts a cursor at each end instead:
//!
//! ```text
//! +--------------+---------+-----+---------+--------------+
//! | free         | entry 0 | ... | entry k | free         |
//! +--------------+---------+-----+---------+--------------+
//!                ^                         ^
//!                head                      tail
//! ```
//!
//! Taking from the front moves `head` right and takes from the back moves `tail`
//! left. Neither touches a byte the other end owns, which is the whole of
//! `04` section 6 for a list: a chunk that is being popped at one end and pushed
//! at the other has two cursors that never meet in one cache line.
//!
//! # Which way a chunk grows
//!
//! A chunk is made for the end it is going to serve. One made for the back
//! starts with both cursors at zero and grows right; one made for the front
//! starts with both at the end of its buffer and grows left. That is a hint and
//! not a rule: a push at the end with no room left slides the live bytes over
//! and splits what is free between the two ends, so a chunk that was filled by
//! `RPUSH` and is then pushed at the front shifts once and then serves both.
//! Without that a list that is pushed at one end and popped at the same end
//! would allocate eight kilobytes per push, because the chunk it emptied would
//! be the wrong way round every time.
//!
//! # Sealing
//!
//! A chunk in the middle of a list is never written to again, so it gives back
//! the room it was holding for growth it will not see. [`Chunk::seal`] slides
//! the live bytes to the front and shrinks the buffer to them, which is one copy
//! of at most a couple of kilobytes once per chunk. Without it a list of a
//! million small integers would hold four times the bytes it needs, because the
//! room a chunk keeps for the next push is a per chunk cost and there is a chunk
//! per hundred and twenty eight elements.

use crate::listpack::{Entry, backlen_len, decode, entry_len, read_backlen, write_entry};

/// How much room a chunk asks for when it is made.
///
/// Eight kilobytes, which is what `list-max-listpack-size -2` gives a quicklist
/// node in a default Redis and therefore the size a list of a given length has
/// been measured against for a decade. It is also what the packed band holds
/// before it promotes, so the listpack a list arrives as becomes exactly one
/// chunk with nothing left over.
pub const CHUNK_BYTES: usize = 8192;

/// How many elements a chunk holds before a new one starts.
///
/// Redis puts no count limit on a node at the default fill and we do, because
/// eight kilobytes of two byte integers is four thousand elements and everything
/// that reaches a list by index walks chunks first and elements second. Five
/// hundred and twelve bounds the walk inside a chunk without making the walk
/// over chunks long: a million small integers is under two thousand chunks
/// either way, and the descriptor cache that makes that a lookup rather than a
/// walk is `08` section 5's K10 and is not here yet.
pub const CHUNK_ENTRIES: usize = 512;

/// A run of entries, filled from one end or the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    bytes: Vec<u8>,
    /// First live byte.
    head: usize,
    /// One past the last live byte.
    tail: usize,
    count: usize,
}

impl Chunk {
    /// An empty chunk that grows toward the back.
    #[must_use]
    pub fn for_back() -> Chunk {
        Chunk {
            bytes: vec![0; CHUNK_BYTES],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// An empty chunk that grows toward the front.
    #[must_use]
    pub fn for_front() -> Chunk {
        Chunk {
            bytes: vec![0; CHUNK_BYTES],
            head: CHUNK_BYTES,
            tail: CHUNK_BYTES,
            count: 0,
        }
    }

    /// A chunk of its own for one element too big for an ordinary one.
    ///
    /// A list element can be half a gigabyte and a chunk is eight kilobytes, so
    /// something has to give. Redis calls this a plain node and so does this:
    /// the chunk is exactly the size of the element, it is full the moment it is
    /// made, and every push against it is refused, which puts the next element
    /// in a chunk of its own rather than growing this one to hold both.
    #[must_use]
    pub fn plain(value: &[u8]) -> Chunk {
        let mut c = Chunk {
            bytes: vec![0; entry_len(value)],
            head: 0,
            tail: 0,
            count: 0,
        };
        let put = c.push_back(value);
        debug_assert!(put, "a chunk sized for one element refused it");
        c
    }

    /// A chunk holding entries somebody else already encoded.
    ///
    /// The promotion out of the packed band, where the bytes in question are a
    /// listpack's entry region. They are in this encoding already, so the band
    /// change is one copy and not a re-encode of every element, and the chunk
    /// that comes out of it grows toward the back because a list that has just
    /// outgrown a listpack is nearly always one that is being appended to.
    #[must_use]
    pub fn adopt(entries: &[u8], count: usize) -> Chunk {
        let mut bytes = vec![0; CHUNK_BYTES.max(entries.len())];
        bytes[..entries.len()].copy_from_slice(entries);
        Chunk {
            bytes,
            head: 0,
            tail: entries.len(),
            count,
        }
    }

    /// How many elements are in it.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether it holds nothing.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// What it costs, buffer included.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.bytes.capacity() + size_of::<Chunk>()
    }

    /// How many bytes the entries themselves take.
    #[must_use]
    #[inline]
    pub const fn live_bytes(&self) -> usize {
        self.tail - self.head
    }

    /// The encoded entries, without the dead space at either end.
    ///
    /// The reverse of [`Chunk::adopt`], and the two are a pair: what comes out
    /// of here goes back in there and gives a chunk holding the same elements.
    /// That is what a demotion needs, because the ring is rebuilt on the way
    /// back and rebuilding it by pushing every element one at a time would
    /// re-encode a list that arrived already encoded.
    #[must_use]
    #[inline]
    pub fn entries(&self) -> &[u8] {
        &self.bytes[self.head..self.tail]
    }

    /// Put `value` at the back, or say there was no room.
    ///
    /// The count cap is checked here rather than by the caller because a chunk
    /// that is full for either reason is full in exactly the same way, and a
    /// caller that had to check one of the two would eventually forget.
    pub fn push_back(&mut self, value: &[u8]) -> bool {
        if self.count >= CHUNK_ENTRIES {
            return false;
        }
        let need = entry_len(value);
        if self.tail + need > self.bytes.len() && !self.shift(need, false) {
            return false;
        }
        write_entry(&mut self.bytes[self.tail..], value);
        self.tail += need;
        self.count += 1;
        true
    }

    /// Put `value` at the front, or say there was no room.
    pub fn push_front(&mut self, value: &[u8]) -> bool {
        if self.count >= CHUNK_ENTRIES {
            return false;
        }
        let need = entry_len(value);
        if need > self.head && !self.shift(need, true) {
            return false;
        }
        self.head -= need;
        write_entry(&mut self.bytes[self.head..], value);
        self.count += 1;
        true
    }

    /// Slide the live bytes so that `need` more of them fit at the end asking.
    ///
    /// Says no when the buffer as a whole does not have the room, which is the
    /// full chunk the caller above turns into a new one. When it does have the
    /// room it gives the asking end what it asked for and splits the rest
    /// evenly, so that a chunk being pushed at both ends shifts once rather than
    /// on every second push.
    fn shift(&mut self, need: usize, front: bool) -> bool {
        let live = self.tail - self.head;
        let free = self.bytes.len() - live;
        if free < need {
            return false;
        }
        let spare = (free - need) / 2;
        let head = if front { need + spare } else { spare };
        self.bytes.copy_within(self.head..self.tail, head);
        self.head = head;
        self.tail = head + live;
        true
    }

    /// The first element, without taking it out.
    #[must_use]
    pub fn front(&self) -> Option<Entry<'_>> {
        if self.count == 0 {
            return None;
        }
        decode(&self.bytes[self.head..self.tail]).map(|(e, _)| e)
    }

    /// The last element, without taking it out.
    #[must_use]
    pub fn back(&self) -> Option<Entry<'_>> {
        let at = self.back_at()?;
        decode(&self.bytes[at..self.tail]).map(|(e, _)| e)
    }

    /// The element at `index` from the front.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Entry<'_>> {
        let at = self.offset_of(index)?;
        decode(&self.bytes[at..self.tail]).map(|(e, _)| e)
    }

    /// Drop the first element, and say whether there was one.
    ///
    /// The bytes stay where they are. A chunk that has been emptied from the
    /// front is dropped whole by the deque above, so there is nobody left to
    /// care that the room it is holding is at the wrong end.
    pub fn drop_front(&mut self) -> bool {
        if self.count == 0 {
            return false;
        }
        let Some(step) = self.step(self.head) else {
            return false;
        };
        self.head += step;
        self.count -= 1;
        true
    }

    /// Drop the last element, and say whether there was one.
    pub fn drop_back(&mut self) -> bool {
        let Some(at) = self.back_at() else {
            return false;
        };
        self.tail = at;
        self.count -= 1;
        true
    }

    /// Drop the first `n` elements.
    ///
    /// A walk of `n` entries and one move of the head cursor, with no bytes
    /// touched at all, which is what makes `LTRIM` of the front of a long list
    /// cost the walk and nothing else. Stops early and says how many it really
    /// dropped if the chunk runs out first.
    pub fn drop_front_n(&mut self, n: usize) -> usize {
        let mut at = self.head;
        let took = n.min(self.count);
        for _ in 0..took {
            let Some(step) = self.step(at) else {
                break;
            };
            at += step;
        }
        self.count -= took;
        self.head = at;
        took
    }

    /// Drop the last `n` elements.
    pub fn drop_back_n(&mut self, n: usize) -> usize {
        let took = n.min(self.count);
        for _ in 0..took {
            let Some(at) = self.back_at() else {
                break;
            };
            self.tail = at;
            self.count -= 1;
        }
        took
    }

    /// Put `value` in at `index`, pushing what was there along.
    ///
    /// Says no when the chunk has no room, which the deque above answers by
    /// splitting the chunk and asking one of the halves. The bytes on one side
    /// of the hole do have to move, because the elements between the cursors are
    /// a run, but it is whichever side is shorter and it is at most the size of
    /// a chunk. That is the same move a quicklist makes for an insert and the
    /// difference is that a list only inserts in the middle when a client asks
    /// it to, where a quicklist does it on every `LPOP`.
    pub fn insert_at(&mut self, index: usize, value: &[u8]) -> bool {
        if index > self.count {
            return false;
        }
        if index == self.count {
            return self.push_back(value);
        }
        if index == 0 {
            return self.push_front(value);
        }
        if self.count >= CHUNK_ENTRIES {
            return false;
        }
        let need = entry_len(value);
        let Some(at) = self.offset_of(index) else {
            return false;
        };
        // Move the front half back or the back half forward, whichever end has
        // the room, preferring the shorter side when both do.
        let front = need <= self.head;
        let back = self.tail + need <= self.bytes.len();
        let at = if (front && !back) || (front && back && index * 2 <= self.count) {
            self.bytes.copy_within(self.head..at, self.head - need);
            self.head -= need;
            at - need
        } else if back {
            self.bytes.copy_within(at..self.tail, at + need);
            self.tail += need;
            at
        } else {
            // No room at either end but perhaps room in total, in which case the
            // shift makes some and the offset has to be found again.
            if !self.shift(need, false) {
                return false;
            }
            let at = self.offset_of(index).expect("the entries did not move");
            self.bytes.copy_within(at..self.tail, at + need);
            self.tail += need;
            at
        };
        write_entry(&mut self.bytes[at..], value);
        self.count += 1;
        true
    }

    /// Take the element at `index` out, closing the gap behind it.
    pub fn remove_at(&mut self, index: usize) -> bool {
        if index >= self.count {
            return false;
        }
        if index == 0 {
            return self.drop_front();
        }
        if index + 1 == self.count {
            return self.drop_back();
        }
        let Some(at) = self.offset_of(index) else {
            return false;
        };
        let Some(span) = self.step(at) else {
            return false;
        };
        // Close the gap from whichever side is shorter, the same as an insert.
        if index * 2 <= self.count {
            self.bytes.copy_within(self.head..at, self.head + span);
            self.head += span;
        } else {
            self.bytes.copy_within(at + span..self.tail, at);
            self.tail -= span;
        }
        self.count -= 1;
        true
    }

    /// Put `value` where the element at `index` was.
    ///
    /// The common case is a value the same size as the one it replaces, which is
    /// written where it lies. Anything else is a remove and an insert, and it
    /// can fail for the same reason an insert can.
    pub fn replace_at(&mut self, index: usize, value: &[u8]) -> bool {
        let Some(at) = self.offset_of(index) else {
            return false;
        };
        let need = entry_len(value);
        if self.step(at) == Some(need) {
            write_entry(&mut self.bytes[at..], value);
            return true;
        }
        // Insert first and remove after, because an insert is the half that can
        // fail and undoing it would mean holding the old bytes somewhere.
        if !self.insert_at(index, value) {
            return false;
        }
        self.remove_at(index + 1)
    }

    /// Split this chunk in two, keeping the first `index` elements.
    ///
    /// The entries are a run in one encoding, so the tail of the run is already
    /// a chunk's worth of bytes and the split is one copy. This is how an insert
    /// into a full chunk gets its room: the deque splits at the insertion point
    /// and both halves come back with space.
    #[must_use]
    pub fn split_off(&mut self, index: usize) -> Chunk {
        let at = self.offset_of(index).unwrap_or(self.tail);
        let rest = Chunk::adopt(&self.bytes[at..self.tail], self.count - index);
        self.tail = at;
        self.count = index;
        rest
    }

    /// Give back the room this chunk was keeping for pushes it will not see.
    ///
    /// Called when a chunk stops being an end of the list. It is a copy of the
    /// live bytes and a shrink, and it happens once per chunk in the life of a
    /// list that is only ever appended to.
    pub fn seal(&mut self) {
        if self.head > 0 {
            self.bytes.copy_within(self.head..self.tail, 0);
            self.tail -= self.head;
            self.head = 0;
        }
        self.bytes.truncate(self.tail);
        self.bytes.shrink_to_fit();
    }

    /// Where `value` is in this chunk, or nothing.
    ///
    /// The same walk [`crate::listpack::Listpack::find_parsed`] does and the
    /// same code, because a chunk is the same entries in a run with a cursor at
    /// each end rather than a blob with a header. `LINSERT` on a long list is
    /// almost entirely this call repeated over a few thousand chunks, so it
    /// reads headers and rejects on length rather than decoding every element
    /// into an [`crate::listpack::Entry`] on the way past.
    #[must_use]
    pub fn find(&self, value: &[u8], as_int: Option<i64>) -> Option<usize> {
        crate::listpack::scan_for(&self.bytes[self.head..self.tail], value, as_int, 1)
    }

    /// Every place `value` is in this chunk, front to back.
    ///
    /// `limit` caps how many elements are looked at with 0 meaning no cap, `hit`
    /// says whether to carry on, and what comes back is how many elements were
    /// looked at so a caller walking a ring can carry one budget across it.
    pub fn find_each(
        &self,
        value: &[u8],
        as_int: Option<i64>,
        limit: usize,
        hit: &mut dyn FnMut(usize) -> bool,
    ) -> usize {
        crate::listpack::scan_each(&self.bytes[self.head..self.tail], value, as_int, limit, hit)
    }

    /// The same from the back, with indexes counted from the last element here.
    pub fn find_each_back(
        &self,
        value: &[u8],
        as_int: Option<i64>,
        limit: usize,
        hit: &mut dyn FnMut(usize) -> bool,
    ) -> usize {
        crate::listpack::scan_each_back(
            &self.bytes[self.head..self.tail],
            value,
            as_int,
            limit,
            hit,
        )
    }

    /// A forward walk over what is here.
    #[must_use]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            bytes: &self.bytes[self.head..self.tail],
            at: 0,
        }
    }

    /// The same walk the other way.
    ///
    /// Every entry carries its own length behind it, which is what the back
    /// cursor reads to find the entry before it, so this costs the same per
    /// element as the forward walk rather than being a forward walk per element.
    /// `LPOS` with a negative rank is the reason it exists.
    #[must_use]
    pub fn iter_back(&self) -> RevIter<'_> {
        RevIter {
            bytes: &self.bytes[self.head..self.tail],
            at: self.tail - self.head,
        }
    }

    /// A forward walk that starts at `index` rather than at the front.
    ///
    /// `LRANGE` in the middle of a list lands in the middle of a chunk, and the
    /// only other way to start there is to walk the entries in front of it and
    /// throw them away. This finds the byte offset by whichever end is closer
    /// and hands back a walk from there.
    #[must_use]
    pub fn iter_from(&self, index: usize) -> Iter<'_> {
        let at = self.offset_of(index).unwrap_or(self.tail);
        Iter {
            bytes: &self.bytes[self.head..self.tail],
            at: at - self.head,
        }
    }

    /// Where the entry at `index` starts, or nothing if there is no such entry.
    ///
    /// From whichever end of the chunk is closer. A chunk holds up to five
    /// hundred and twelve entries, and half of them is the difference between
    /// a microsecond and half of one on a `LINDEX` that lands in the middle of
    /// a big list. Going backward is what the length behind each entry is for,
    /// and it is the same field `iter_back` reads.
    fn offset_of(&self, index: usize) -> Option<usize> {
        if index >= self.count {
            return None;
        }
        if index * 2 <= self.count {
            let mut at = self.head;
            for _ in 0..index {
                at += self.step(at)?;
            }
            return Some(at);
        }
        let mut end = self.tail;
        for _ in index..self.count {
            let len = read_backlen(&self.bytes[self.head..end])?;
            end = end.checked_sub(len + backlen_len(len))?;
        }
        Some(end)
    }

    /// Where the last entry starts.
    fn back_at(&self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        let len = read_backlen(&self.bytes[self.head..self.tail])?;
        self.tail.checked_sub(len + backlen_len(len))
    }

    /// How many bytes the entry at `at` takes, back length included.
    fn step(&self, at: usize) -> Option<usize> {
        let (_, len) = decode(&self.bytes[at..self.tail])?;
        Some(len + backlen_len(len))
    }
}

/// A forward walk over a chunk.
#[derive(Debug)]
pub struct Iter<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = Entry<'a>;

    #[inline]
    fn next(&mut self) -> Option<Entry<'a>> {
        if self.at >= self.bytes.len() {
            return None;
        }
        let (entry, len) = decode(&self.bytes[self.at..])?;
        self.at += len + backlen_len(len);
        Some(entry)
    }
}

/// A backward walk over a chunk.
#[derive(Debug)]
pub struct RevIter<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Iterator for RevIter<'a> {
    type Item = Entry<'a>;

    #[inline]
    fn next(&mut self) -> Option<Entry<'a>> {
        if self.at == 0 {
            return None;
        }
        let len = read_backlen(&self.bytes[..self.at])?;
        let start = self.at.checked_sub(len + backlen_len(len))?;
        let (entry, _) = decode(&self.bytes[start..self.at])?;
        self.at = start;
        Some(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(c: &Chunk) -> Vec<Vec<u8>> {
        c.iter().map(|e| e.to_vec()).collect()
    }

    #[test]
    fn a_back_chunk_fills_from_the_front_of_its_buffer() {
        let mut c = Chunk::for_back();
        assert!(c.push_back(b"a"));
        assert!(c.push_back(b"b"));
        assert_eq!(all(&c), vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.front().unwrap().to_vec(), b"a");
        assert_eq!(c.back().unwrap().to_vec(), b"b");
    }

    #[test]
    fn a_front_chunk_fills_backward_and_reads_forward() {
        let mut c = Chunk::for_front();
        assert!(c.push_front(b"b"));
        assert!(c.push_front(b"a"));
        assert_eq!(all(&c), vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(c.front().unwrap().to_vec(), b"a");
        assert_eq!(c.back().unwrap().to_vec(), b"b");
    }

    /// Both ends at once, which is the shape the deque above puts a chunk in
    /// when a list is pushed at one end and popped at the other.
    #[test]
    fn pushing_and_popping_at_both_ends_stays_in_order() {
        let mut c = Chunk::for_back();
        for m in [b"c", b"d"] {
            assert!(c.push_back(m));
        }
        for m in [b"b", b"a"] {
            assert!(c.push_front(m));
        }
        assert_eq!(
            all(&c),
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec()]
        );
        assert!(c.drop_front());
        assert!(c.drop_back());
        assert_eq!(all(&c), vec![b"b".to_vec(), b"c".to_vec()]);
        assert_eq!(c.len(), 2);
    }

    /// A back chunk has nothing in front of it, so the first front push slides
    /// what is there over. After that both ends have room and neither shifts
    /// again, which is the part that keeps a same end push and pop from
    /// allocating a chunk per operation.
    #[test]
    fn a_back_chunk_makes_room_at_the_front_once() {
        let mut c = Chunk::for_back();
        assert!(c.push_back(b"a"));
        assert!(c.push_front(b"z"));
        assert_eq!(all(&c), vec![b"z".to_vec(), b"a".to_vec()]);
        let head = c.head;
        for i in 0..100 {
            assert!(c.push_front(i.to_string().as_bytes()), "at {i}");
        }
        assert!(c.head < head, "the front pushes went somewhere else");
        assert_eq!(c.len(), 102);
        assert_eq!(c.back().unwrap().to_vec(), b"a");
    }

    /// A chunk with no room anywhere says so rather than shifting bytes that
    /// have nowhere to go.
    #[test]
    fn a_full_chunk_refuses_both_ends() {
        let mut c = Chunk::for_back();
        let big = vec![b'x'; 500];
        while c.push_back(&big) {}
        assert!(!c.push_back(&big));
        assert!(!c.push_front(&big));
        assert!(c.push_front(b"1"), "there is still room for a short one");
    }

    #[test]
    fn an_empty_chunk_answers_nothing_rather_than_panicking() {
        let mut c = Chunk::for_back();
        assert!(c.front().is_none());
        assert!(c.back().is_none());
        assert!(c.get(0).is_none());
        assert!(!c.drop_front());
        assert!(!c.drop_back());
    }

    #[test]
    fn the_element_cap_is_what_stops_a_chunk_of_small_members() {
        let mut c = Chunk::for_back();
        for i in 0..CHUNK_ENTRIES {
            assert!(c.push_back(i.to_string().as_bytes()), "at {i}");
        }
        assert!(!c.push_back(b"1"));
        assert_eq!(c.len(), CHUNK_ENTRIES);
    }

    #[test]
    fn the_byte_cap_is_what_stops_a_chunk_of_large_members() {
        let mut c = Chunk::for_back();
        let big = vec![b'x'; 300];
        let mut n = 0;
        while c.push_back(&big) {
            n += 1;
        }
        assert!(n < CHUNK_ENTRIES, "{n} entries fitted, which is too many");
        assert_eq!(c.len(), n);
        assert!(c.live_bytes() <= CHUNK_BYTES);
    }

    #[test]
    fn indexing_walks_from_the_head_cursor_and_not_from_the_buffer() {
        let mut c = Chunk::for_front();
        for m in [b"d", b"c", b"b", b"a"] {
            assert!(c.push_front(m));
        }
        for (i, want) in [b"a", b"b", b"c", b"d"].iter().enumerate() {
            assert_eq!(c.get(i).unwrap().to_vec(), want.to_vec(), "at {i}");
        }
        assert!(c.get(4).is_none());
    }

    #[test]
    fn sealing_keeps_the_elements_and_gives_back_the_room() {
        let mut c = Chunk::for_front();
        for m in [b"c", b"b", b"a"] {
            assert!(c.push_front(m));
        }
        let before = c.memory_bytes();
        let live = c.live_bytes();
        c.seal();
        assert_eq!(
            all(&c),
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            "sealing changed what is in it"
        );
        assert_eq!(c.live_bytes(), live);
        assert!(c.memory_bytes() < before);
        // The cursors came with the bytes rather than being left where they
        // used to be, so both ends still read. There is no room left for a push
        // and there is not meant to be: a chunk is sealed when it stops being an
        // end of the list, and one that becomes an end again is refused and
        // replaced rather than grown back.
        assert_eq!(c.front().unwrap().to_vec(), b"a");
        assert_eq!(c.back().unwrap().to_vec(), b"c");
        assert_eq!(c.get(1).unwrap().to_vec(), b"b");
        assert!(!c.push_back(b"d"));
        assert!(!c.push_front(b"d"));
        assert!(c.drop_front());
        assert_eq!(c.front().unwrap().to_vec(), b"b");
    }

    /// An integer element is stored as an integer, the same as it is in a
    /// listpack, because it is the same codec.
    #[test]
    fn integers_come_back_as_integers() {
        let mut c = Chunk::for_back();
        assert!(c.push_back(b"42"));
        assert!(c.push_back(b"007"));
        assert_eq!(c.get(0), Some(Entry::Int(42)));
        assert_eq!(c.get(1), Some(Entry::Str(b"007")));
    }

    /// Every length either side of a back length boundary, so that a chunk
    /// whose entries cross into a two byte back length still walks backward.
    #[test]
    fn the_back_walk_survives_a_long_entry() {
        for len in [1usize, 63, 64, 120, 126, 127, 128, 200] {
            let mut c = Chunk::for_back();
            let v = vec![b'x'; len];
            assert!(c.push_back(b"first"));
            assert!(c.push_back(&v), "{len} did not fit");
            assert_eq!(c.back().unwrap().to_vec(), v, "back of a {len} byte entry");
            assert!(c.drop_back());
            assert_eq!(c.back().unwrap().to_vec(), b"first");
        }
    }

    /// A chunk with room at both ends, so that an insert can choose a side.
    fn abcde() -> Chunk {
        let mut c = Chunk::for_back();
        for m in [b"c", b"d", b"e"] {
            assert!(c.push_back(m));
        }
        for m in [b"b", b"a"] {
            assert!(c.push_front(m));
        }
        c
    }

    #[test]
    fn an_insert_lands_where_it_was_asked_to_from_either_side() {
        for at in 0..=5 {
            let mut c = abcde();
            assert!(c.insert_at(at, b"new"), "inserting at {at}");
            let mut want: Vec<Vec<u8>> = [b"a", b"b", b"c", b"d", b"e"]
                .iter()
                .map(|m| m.to_vec())
                .collect();
            want.insert(at, b"new".to_vec());
            assert_eq!(all(&c), want, "inserting at {at}");
            assert_eq!(c.len(), 6);
        }
        let mut c = abcde();
        assert!(!c.insert_at(6, b"new"), "past the end is not an insert");
    }

    #[test]
    fn a_remove_closes_the_gap_from_either_side() {
        for at in 0..5 {
            let mut c = abcde();
            assert!(c.remove_at(at), "removing at {at}");
            let mut want: Vec<Vec<u8>> = [b"a", b"b", b"c", b"d", b"e"]
                .iter()
                .map(|m| m.to_vec())
                .collect();
            want.remove(at);
            assert_eq!(all(&c), want, "removing at {at}");
            assert_eq!(c.len(), 4);
            assert_eq!(c.back().unwrap().to_vec(), want[3]);
        }
        let mut c = abcde();
        assert!(!c.remove_at(5));
    }

    /// The same length, a longer one and a shorter one, because only the first
    /// is written where the old element lay.
    #[test]
    fn a_replace_takes_a_value_of_any_length() {
        for value in [&b"z"[..], &b"much longer than what was there"[..], b"7"] {
            for at in 0..5 {
                let mut c = abcde();
                assert!(c.replace_at(at, value), "replacing at {at}");
                let mut want: Vec<Vec<u8>> = [b"a", b"b", b"c", b"d", b"e"]
                    .iter()
                    .map(|m| m.to_vec())
                    .collect();
                want[at] = value.to_vec();
                assert_eq!(all(&c), want, "replacing at {at}");
                assert_eq!(c.len(), 5);
            }
        }
        let mut c = abcde();
        assert!(!c.replace_at(5, b"z"));
    }

    #[test]
    fn dropping_many_from_an_end_is_the_walk_and_nothing_else() {
        let mut c = abcde();
        assert_eq!(c.drop_front_n(2), 2);
        assert_eq!(all(&c), vec![b"c".to_vec(), b"d".to_vec(), b"e".to_vec()]);
        assert_eq!(c.drop_back_n(2), 2);
        assert_eq!(all(&c), vec![b"c".to_vec()]);
        assert_eq!(c.drop_front_n(9), 1, "it stops when it runs out");
        assert!(c.is_empty());
        assert_eq!(c.drop_back_n(3), 0);
    }

    #[test]
    fn a_split_leaves_both_halves_readable_and_with_room() {
        for at in 0..=5 {
            let mut c = abcde();
            let mut rest = c.split_off(at);
            let want: Vec<Vec<u8>> = [b"a", b"b", b"c", b"d", b"e"]
                .iter()
                .map(|m| m.to_vec())
                .collect();
            assert_eq!(all(&c), want[..at].to_vec(), "the front half of {at}");
            assert_eq!(all(&rest), want[at..].to_vec(), "the back half of {at}");
            assert_eq!(c.len() + rest.len(), 5);
            assert!(rest.push_back(b"more"), "the back half has no room");
            assert_eq!(rest.back().unwrap().to_vec(), b"more");
        }
    }

    /// An insert into a chunk that is full at both ends is refused, and the
    /// split is what the deque does about it.
    #[test]
    fn a_full_chunk_refuses_an_insert_and_a_split_fixes_it() {
        let mut c = Chunk::for_back();
        let big = vec![b'x'; 500];
        while c.push_back(&big) {}
        let held = c.len();
        assert!(!c.insert_at(held / 2, &big));
        let mut rest = c.split_off(held / 2);
        assert!(c.push_back(&big) || rest.push_front(&big));
        assert_eq!(c.len() + rest.len(), held + 1);
    }
}
