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
        if index >= self.count {
            return None;
        }
        let mut at = self.head;
        for _ in 0..index {
            at += self.step(at)?;
        }
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

    /// A forward walk over what is here.
    #[must_use]
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            bytes: &self.bytes[self.head..self.tail],
            at: 0,
        }
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
}
