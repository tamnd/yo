//! A list, in whichever of the two representations currently fits it.
//!
//! A list is one packed blob while it is small and a ring of [`Chunk`]s once it
//! is not, which is the same two bands Redis has and the same two names
//! `OBJECT ENCODING` answers, `listpack` and `quicklist`.
//!
//! ```text
//!   under eight kilobytes            everything else
//! +--------------------------+   +------------------------------------+
//! | one listpack             |-->| chunk | chunk | ... | chunk        |
//! | walked from either end   |   | head cursor ...... tail cursor     |
//! +--------------------------+   +------------------------------------+
//! ```
//!
//! # Why a chunk and not a listpack per node
//!
//! Redis's quicklist node is a listpack, and taking the first element out of a
//! listpack moves every byte behind it left. On a list that is being used as a
//! queue, which is what a list is for, that memmove is on the hot path of every
//! single `LPOP`. [`Chunk`] holds the same entries in the same encoding with a
//! cursor at each end, so a pop is a cursor step and a push at the other end
//! does not touch it. That is `04` section 6's independent head and tail, and it
//! is the answer to the row aki lost.
//!
//! # The band boundary is Redis's, and it goes both ways
//!
//! `list-max-listpack-size` defaults to `-2`, which means eight kilobytes rather
//! than a count, so a list of a thousand short strings is still one blob and a
//! list of two hundred long ones is not. That was read off `t_list.c` in the
//! 7.4.5 tarball rather than assumed, and so was the part that surprised: a list
//! converts **back** when it shrinks, which no other collection here does. Redis
//! only converts back once the list is under **half** the limit, so that a
//! workload sitting exactly on the boundary does not rebuild itself on every
//! other command, and [`List::shrunk`] is that rule.
//!
//! # Elements
//!
//! An element comes back as a [`Member`](crate::set::Member), which is
//! [`listpack::Entry`](crate::listpack::Entry) under another name, so a value
//! stored as an integer is handed over as one and formatted once, into the reply
//! buffer, at the moment the reply is built. That is Y18 again.

use std::collections::VecDeque;

use crate::chunk::{CHUNK_BYTES, Chunk};
use crate::listpack::{Entry, Listpack};

/// A list element: bytes as they lie, or an integer not yet formatted.
pub type Element<'a> = Entry<'a>;

/// Where a list changes representation.
///
/// One number, because Redis has one: `list-max-listpack-size`. A negative value
/// there is a size in kilobytes and a positive one is a count of elements, and
/// both arrive here already turned into the two fields the bands actually ask
/// about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most bytes a packed list holds before it becomes chunks.
    pub max_packed_bytes: usize,
    /// The most elements a packed list holds, or none for no limit.
    ///
    /// The default configuration has no count limit at all, because `-2` is a
    /// size. A server configured with a positive `list-max-listpack-size` has
    /// one and no size limit, which is why these are two fields and not an enum
    /// of one or the other: the halving rule for shrinking applies to whichever
    /// is set and the code below should not have to know which that was.
    pub max_packed_entries: Option<usize>,
}

impl Default for Limits {
    /// What a server with no configuration file uses.
    ///
    /// `list-max-listpack-size -2`, which is eight kilobytes and no count.
    fn default() -> Limits {
        Limits {
            max_packed_bytes: CHUNK_BYTES,
            max_packed_entries: None,
        }
    }
}

impl Limits {
    /// The limits a `list-max-listpack-size` of `fill` describes.
    ///
    /// This is `quicklistNodeLimit`. A positive fill is a count and the size is
    /// left at the safety limit, a negative one is an index into Redis's five
    /// sizes, and zero means one element per node, which is a setting nobody
    /// uses and which still has to mean something.
    #[must_use]
    pub fn of(fill: i32) -> Limits {
        if fill >= 0 {
            return Limits {
                max_packed_bytes: CHUNK_BYTES,
                max_packed_entries: Some((fill as usize).max(1)),
            };
        }
        // Redis's `optimization_level`, which is 4 KiB, 8, 16, 32 and 64.
        const SIZES: [usize; 5] = [4096, 8192, 16384, 32768, 65536];
        let at = ((-fill) as usize - 1).min(SIZES.len() - 1);
        Limits {
            max_packed_bytes: SIZES[at],
            max_packed_entries: None,
        }
    }

    /// Whether a packed list of these dimensions is past what the band holds.
    #[must_use]
    fn exceeded(&self, bytes: usize, entries: usize) -> bool {
        match self.max_packed_entries {
            Some(cap) => bytes > CHUNK_BYTES || entries > cap,
            None => bytes > self.max_packed_bytes,
        }
    }

    /// The same question with both limits halved, which is the shrinking rule.
    #[must_use]
    fn exceeded_halved(&self, bytes: usize, entries: usize) -> bool {
        match self.max_packed_entries {
            Some(cap) => bytes > CHUNK_BYTES / 2 || entries > cap / 2,
            None => bytes > self.max_packed_bytes / 2,
        }
    }
}

/// What `OBJECT ENCODING` calls a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// One packed blob.
    Listpack,
    /// A ring of chunks.
    Quicklist,
}

impl Encoding {
    /// The string `OBJECT ENCODING` returns.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Listpack => "listpack",
            Encoding::Quicklist => "quicklist",
        }
    }
}

/// Which representation the elements are in.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Body {
    Packed(Listpack),
    Chunks(Deque),
}

/// A list of elements, in order, reachable from both ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct List {
    body: Body,
}

impl Default for List {
    fn default() -> List {
        List::new()
    }
}

impl List {
    /// An empty list, in the band every list starts in.
    #[must_use]
    pub fn new() -> List {
        List {
            body: Body::Packed(Listpack::new()),
        }
    }

    /// How many elements it holds.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        match &self.body {
            Body::Packed(lp) => lp.len(),
            Body::Chunks(d) => d.len(),
        }
    }

    /// Whether it holds nothing.
    ///
    /// A list that reaches zero is deleted by the keyspace, the same as a set
    /// that does, so this is a question about the moment between the last pop
    /// and that delete rather than a state a client can observe.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What `OBJECT ENCODING` says about it.
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match &self.body {
            Body::Packed(_) => Encoding::Listpack,
            Body::Chunks(_) => Encoding::Quicklist,
        }
    }

    /// What it costs, not counting anything a caller is holding.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.body {
            Body::Packed(lp) => lp.byte_len(),
            Body::Chunks(d) => d.memory_bytes(),
        }
    }

    /// The element at `index` from the front.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Element<'_>> {
        match &self.body {
            Body::Packed(lp) => lp.get(index),
            Body::Chunks(d) => d.get(index),
        }
    }

    /// The first element.
    #[must_use]
    pub fn front(&self) -> Option<Element<'_>> {
        match &self.body {
            Body::Packed(lp) => lp.get(0),
            Body::Chunks(d) => d.front(),
        }
    }

    /// The last element.
    ///
    /// Reads the back length rather than walking, in both bands, which is what
    /// makes `RPOP` on a long list cost the same as `LPOP` on one.
    #[must_use]
    pub fn back(&self) -> Option<Element<'_>> {
        match &self.body {
            Body::Packed(lp) => lp.get_back(0),
            Body::Chunks(d) => d.back(),
        }
    }

    /// A forward walk over every element.
    pub fn iter(&self) -> impl Iterator<Item = Element<'_>> {
        // Two shapes with one type, because a caller that only wants the
        // elements should not have to know which band it is standing on.
        let (packed, chunks) = match &self.body {
            Body::Packed(lp) => (Some(lp.iter()), None),
            Body::Chunks(d) => (None, Some(d.iter())),
        };
        packed
            .into_iter()
            .flatten()
            .chain(chunks.into_iter().flatten())
    }

    /// Put `value` at the front.
    pub fn push_front(&mut self, value: &[u8], limits: &Limits) {
        self.grow_for(value, limits);
        match &mut self.body {
            Body::Packed(lp) => lp.insert(0, value),
            Body::Chunks(d) => d.push_front(value),
        }
    }

    /// Put `value` at the back.
    pub fn push_back(&mut self, value: &[u8], limits: &Limits) {
        self.grow_for(value, limits);
        match &mut self.body {
            Body::Packed(lp) => lp.push(value),
            Body::Chunks(d) => d.push_back(value),
        }
    }

    /// Drop the first element, and say whether there was one.
    ///
    /// The read and the removal are separate so that `LPOP` on the wire can
    /// write the element straight into the reply buffer and then drop it,
    /// which is the same split [`crate::set::Set::drop_at`] exists for.
    pub fn drop_front(&mut self) -> bool {
        match &mut self.body {
            Body::Packed(lp) => lp.delete(0, 1),
            Body::Chunks(d) => d.drop_front(),
        }
    }

    /// Drop the last element, and say whether there was one.
    pub fn drop_back(&mut self) -> bool {
        match &mut self.body {
            Body::Packed(lp) => {
                let last = lp.len().checked_sub(1);
                last.is_some_and(|at| lp.delete(at, 1))
            }
            Body::Chunks(d) => d.drop_back(),
        }
    }

    /// Take the first element out and hand it back.
    ///
    /// The embedded API's `LPOP`, where the caller wants the bytes and has
    /// nowhere to put them.
    pub fn pop_front(&mut self) -> Option<Vec<u8>> {
        let out = self.front()?.to_vec();
        self.drop_front();
        Some(out)
    }

    /// Take the last element out and hand it back.
    pub fn pop_back(&mut self) -> Option<Vec<u8>> {
        let out = self.back()?.to_vec();
        self.drop_back();
        Some(out)
    }

    /// Go back to one blob if the list has shrunk far enough to deserve it.
    ///
    /// Called after anything that removes elements. Redis converts back only
    /// below half the limit, so that a list sitting on the boundary does not
    /// rebuild itself on every other command, and it only converts a quicklist
    /// that is down to one node. Both of those are here.
    pub fn shrunk(&mut self, limits: &Limits) {
        let Body::Chunks(d) = &self.body else {
            return;
        };
        if d.chunks.len() != 1 {
            return;
        }
        let only = &d.chunks[0];
        if limits.exceeded_halved(only.live_bytes(), only.len()) {
            return;
        }
        let mut lp = Listpack::new();
        for e in only.iter() {
            match e {
                Entry::Int(n) => {
                    let mut digits = Vec::new();
                    Entry::Int(n).write_to(&mut digits);
                    lp.push(&digits);
                }
                Entry::Str(s) => lp.push(s),
            }
        }
        self.body = Body::Packed(lp);
    }

    /// Promote out of the packed band if `value` would not fit in it.
    ///
    /// Asked before the push rather than after, because a listpack that has
    /// already been grown past the limit and is then converted has done the
    /// work twice.
    fn grow_for(&mut self, value: &[u8], limits: &Limits) {
        let Body::Packed(lp) = &self.body else {
            return;
        };
        let after = lp.byte_len() + crate::listpack::entry_len(value);
        if !limits.exceeded(after, lp.len() + 1) {
            return;
        }
        let Body::Packed(lp) = std::mem::replace(&mut self.body, Body::Chunks(Deque::new())) else {
            unreachable!("just matched a packed body");
        };
        let Body::Chunks(d) = &mut self.body else {
            unreachable!("just put a chunked body there");
        };
        d.adopt(&lp);
    }
}

/// A ring of chunks, with the list's length kept beside it.
///
/// The length is carried rather than summed because `LLEN` is a command and
/// summing a thousand chunk counts to answer it would be a walk of the whole
/// list to say how long it is.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Deque {
    chunks: VecDeque<Chunk>,
    len: usize,
}

impl Deque {
    /// An empty ring.
    fn new() -> Deque {
        Deque {
            chunks: VecDeque::new(),
            len: 0,
        }
    }

    /// How many elements are in the whole ring.
    #[inline]
    const fn len(&self) -> usize {
        self.len
    }

    /// Take the entries of a listpack as this ring's first chunk.
    fn adopt(&mut self, lp: &Listpack) {
        self.len = lp.len();
        self.chunks.push_back(Chunk::adopt(lp.entries(), lp.len()));
    }

    /// What the whole ring costs.
    ///
    /// A chunk counts its own header, because it is sitting in the ring's own
    /// allocation, so what is left to add is the slots the ring is holding empty
    /// for the chunks it does not have yet.
    fn memory_bytes(&self) -> usize {
        let spare = self.chunks.capacity() - self.chunks.len();
        self.chunks.iter().map(Chunk::memory_bytes).sum::<usize>() + spare * size_of::<Chunk>()
    }

    /// The first element.
    fn front(&self) -> Option<Element<'_>> {
        self.chunks.front()?.front()
    }

    /// The last element.
    fn back(&self) -> Option<Element<'_>> {
        self.chunks.back()?.back()
    }

    /// The element at `index`, chunks first and elements second.
    fn get(&self, index: usize) -> Option<Element<'_>> {
        if index >= self.len {
            return None;
        }
        // From whichever end is closer, because a list is a queue and the two
        // interesting indexes are near the ends.
        if index * 2 <= self.len {
            let mut at = index;
            for c in &self.chunks {
                if at < c.len() {
                    return c.get(at);
                }
                at -= c.len();
            }
        } else {
            let mut back = self.len - index - 1;
            for c in self.chunks.iter().rev() {
                if back < c.len() {
                    return c.get(c.len() - back - 1);
                }
                back -= c.len();
            }
        }
        None
    }

    /// A forward walk over every chunk in turn.
    fn iter(&self) -> impl Iterator<Item = Element<'_>> {
        self.chunks.iter().flat_map(Chunk::iter)
    }

    /// Put `value` at the front, in the head chunk or in a new one.
    fn push_front(&mut self, value: &[u8]) {
        if let Some(head) = self.chunks.front_mut()
            && head.push_front(value)
        {
            self.len += 1;
            return;
        }
        // The chunk that was the head stops being an end, so it gives back the
        // room it was keeping.
        if let Some(head) = self.chunks.front_mut() {
            head.seal();
        }
        let mut fresh = Chunk::for_front();
        fresh.push_front(value);
        self.chunks.push_front(fresh);
        self.len += 1;
    }

    /// Put `value` at the back, in the tail chunk or in a new one.
    fn push_back(&mut self, value: &[u8]) {
        if let Some(tail) = self.chunks.back_mut()
            && tail.push_back(value)
        {
            self.len += 1;
            return;
        }
        if let Some(tail) = self.chunks.back_mut() {
            tail.seal();
        }
        let mut fresh = Chunk::for_back();
        fresh.push_back(value);
        self.chunks.push_back(fresh);
        self.len += 1;
    }

    /// Drop the first element, dropping the chunk with it if it was the last.
    fn drop_front(&mut self) -> bool {
        let Some(head) = self.chunks.front_mut() else {
            return false;
        };
        if !head.drop_front() {
            return false;
        }
        self.len -= 1;
        if head.is_empty() && self.chunks.len() > 1 {
            self.chunks.pop_front();
        }
        true
    }

    /// Drop the last element, dropping the chunk with it if it was the last.
    fn drop_back(&mut self) -> bool {
        let Some(tail) = self.chunks.back_mut() else {
            return false;
        };
        if !tail.drop_back() {
            return false;
        }
        self.len -= 1;
        if tail.is_empty() && self.chunks.len() > 1 {
            self.chunks.pop_back();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(l: &List) -> Vec<Vec<u8>> {
        l.iter().map(|e| e.to_vec()).collect()
    }

    /// A list of `n` elements, each long enough that `n` of them do not fit the
    /// packed band, so the test is standing on the chunked one.
    fn chunked(n: usize) -> List {
        let mut l = List::new();
        let limits = Limits::default();
        for i in 0..n {
            l.push_back(format!("value:{i:0>60}").as_bytes(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Quicklist, "{n} did not promote");
        l
    }

    #[test]
    fn a_new_list_is_empty_and_packed() {
        let l = List::new();
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
        assert_eq!(l.encoding(), Encoding::Listpack);
        assert!(l.front().is_none());
        assert!(l.back().is_none());
        assert!(l.get(0).is_none());
    }

    #[test]
    fn pushing_at_both_ends_puts_the_elements_in_order() {
        let mut l = List::new();
        let limits = Limits::default();
        l.push_back(b"b", &limits);
        l.push_back(b"c", &limits);
        l.push_front(b"a", &limits);
        assert_eq!(all(&l), vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        assert_eq!(l.front().unwrap().to_vec(), b"a");
        assert_eq!(l.back().unwrap().to_vec(), b"c");
        assert_eq!(l.get(1).unwrap().to_vec(), b"b");
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn popping_takes_from_the_end_it_says() {
        let mut l = List::new();
        let limits = Limits::default();
        for m in [b"a", b"b", b"c"] {
            l.push_back(m, &limits);
        }
        assert_eq!(l.pop_front().unwrap(), b"a");
        assert_eq!(l.pop_back().unwrap(), b"c");
        assert_eq!(all(&l), vec![b"b".to_vec()]);
        assert_eq!(l.pop_front().unwrap(), b"b");
        assert!(l.pop_front().is_none());
        assert!(l.pop_back().is_none());
        assert!(l.is_empty());
    }

    /// The band boundary is a size and not a count at the default setting, so a
    /// thousand short elements are still one blob.
    #[test]
    fn a_thousand_short_elements_stay_packed() {
        let mut l = List::new();
        let limits = Limits::default();
        for i in 0..1000 {
            l.push_back(i.to_string().as_bytes(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Listpack);
        assert_eq!(l.len(), 1000);
    }

    #[test]
    fn enough_bytes_promotes_and_keeps_every_element() {
        let l = chunked(300);
        assert_eq!(l.len(), 300);
        for i in 0..300 {
            assert_eq!(
                l.get(i).unwrap().to_vec(),
                format!("value:{i:0>60}").into_bytes(),
                "element {i} after promotion"
            );
        }
    }

    #[test]
    fn a_chunked_list_pushes_and_pops_at_both_ends() {
        let mut l = chunked(300);
        let limits = Limits::default();
        l.push_front(b"first", &limits);
        l.push_back(b"last", &limits);
        assert_eq!(l.len(), 302);
        assert_eq!(l.front().unwrap().to_vec(), b"first");
        assert_eq!(l.back().unwrap().to_vec(), b"last");
        assert_eq!(l.pop_front().unwrap(), b"first");
        assert_eq!(l.pop_back().unwrap(), b"last");
        assert_eq!(l.len(), 300);
        assert_eq!(
            l.front().unwrap().to_vec(),
            format!("value:{:0>60}", 0).into_bytes()
        );
    }

    /// A queue: everything in at one end, everything out at the other, which is
    /// the shape that empties chunks from the front and makes new ones at the
    /// back at the same time.
    #[test]
    fn a_queue_drains_in_the_order_it_filled() {
        let mut l = List::new();
        let limits = Limits::default();
        for i in 0..5000 {
            l.push_back(format!("job:{i:0>40}").as_bytes(), &limits);
        }
        for i in 0..5000 {
            assert_eq!(
                l.pop_front().unwrap(),
                format!("job:{i:0>40}").into_bytes(),
                "job {i} came back in the wrong place"
            );
        }
        assert!(l.is_empty());
    }

    /// A stack: in and out at the same end, which is the shape that leaves a
    /// chunk half empty and pushes into it again.
    #[test]
    fn a_stack_comes_back_in_reverse() {
        let mut l = List::new();
        let limits = Limits::default();
        for i in 0..2000 {
            l.push_front(format!("frame:{i:0>40}").as_bytes(), &limits);
        }
        for i in (0..2000).rev() {
            assert_eq!(
                l.pop_front().unwrap(),
                format!("frame:{i:0>40}").into_bytes()
            );
        }
        assert!(l.is_empty());
    }

    #[test]
    fn indexing_agrees_with_the_walk_from_both_ends() {
        let l = chunked(1000);
        let walked = all(&l);
        for (i, want) in walked.iter().enumerate() {
            assert_eq!(&l.get(i).unwrap().to_vec(), want, "at {i}");
        }
        assert!(l.get(walked.len()).is_none());
    }

    /// Redis converts a list back to a listpack when it shrinks under half the
    /// limit, and only then, so a list at the boundary does not flap.
    #[test]
    fn a_list_that_shrinks_far_enough_goes_back_to_one_blob() {
        let mut l = chunked(300);
        let limits = Limits::default();
        while l.len() > 200 {
            l.drop_back();
            l.shrunk(&limits);
        }
        assert_eq!(
            l.encoding(),
            Encoding::Quicklist,
            "under the limit is not under half of it"
        );
        while l.len() > 50 {
            l.drop_back();
            l.shrunk(&limits);
        }
        assert_eq!(l.encoding(), Encoding::Listpack);
        assert_eq!(l.len(), 50);
        for i in 0..50 {
            assert_eq!(
                l.get(i).unwrap().to_vec(),
                format!("value:{i:0>60}").into_bytes(),
                "element {i} survived the demotion"
            );
        }
    }

    /// And it can be pushed straight back up again afterwards, which is the
    /// part a demotion that left the wrong length behind would break.
    #[test]
    fn a_demoted_list_promotes_again() {
        let mut l = chunked(300);
        let limits = Limits::default();
        while l.len() > 20 {
            l.drop_back();
            l.shrunk(&limits);
        }
        assert_eq!(l.encoding(), Encoding::Listpack);
        for i in 0..300 {
            l.push_back(format!("again:{i:0>60}").as_bytes(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Quicklist);
        assert_eq!(l.len(), 320);
        assert_eq!(
            l.get(19).unwrap().to_vec(),
            format!("value:{:0>60}", 19).into_bytes(),
            "the last of the elements that survived the demotion"
        );
        assert_eq!(
            l.get(20).unwrap().to_vec(),
            format!("again:{:0>60}", 0).into_bytes(),
            "the first of the elements pushed after it"
        );
    }

    /// An integer element is stored as an integer in both bands, which is what
    /// makes a list of numbers cost two bytes an element.
    #[test]
    fn integers_stay_integers_across_the_band_change() {
        let mut l = List::new();
        let limits = Limits::default();
        for i in 0..300 {
            l.push_back(i.to_string().as_bytes(), &limits);
            l.push_back(vec![b'x'; 100].as_slice(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Quicklist);
        assert_eq!(l.get(0), Some(Entry::Int(0)));
        assert_eq!(l.get(2), Some(Entry::Int(1)));
        assert_eq!(l.len(), 600);
    }

    /// A positive `list-max-listpack-size` is a count of elements, which is the
    /// other half of the configuration and the shape the Redis test suite sets
    /// when it wants a quicklist out of four elements.
    #[test]
    fn a_count_limit_promotes_on_the_count() {
        let limits = Limits::of(4);
        let mut l = List::new();
        for i in 0..4 {
            l.push_back(i.to_string().as_bytes(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Listpack);
        l.push_back(b"5", &limits);
        assert_eq!(l.encoding(), Encoding::Quicklist);
        assert_eq!(l.len(), 5);
    }

    #[test]
    fn the_limits_are_redis_node_limits() {
        assert_eq!(Limits::of(-1).max_packed_bytes, 4096);
        assert_eq!(Limits::of(-2).max_packed_bytes, 8192);
        assert_eq!(Limits::of(-5).max_packed_bytes, 65536);
        assert_eq!(Limits::of(-9).max_packed_bytes, 65536);
        assert_eq!(Limits::of(128).max_packed_entries, Some(128));
        assert_eq!(Limits::of(0).max_packed_entries, Some(1));
        assert_eq!(Limits::of(-2), Limits::default());
    }

    #[test]
    fn memory_is_counted_in_both_bands() {
        let mut l = List::new();
        let limits = Limits::default();
        assert!(l.memory_bytes() > 0);
        for i in 0..300 {
            l.push_back(format!("value:{i:0>60}").as_bytes(), &limits);
        }
        // Three hundred elements of sixty six bytes is about twenty kilobytes,
        // and the chunks holding them should not be far off that.
        let held = l.memory_bytes();
        assert!(held > 300 * 66, "{held} is less than the elements");
        assert!(held < 300 * 66 * 3, "{held} is three times the elements");
    }
}
