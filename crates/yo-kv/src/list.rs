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
//! other command, and `List::shrunk` is that rule.
//!
//! # Elements
//!
//! An element comes back as a [`Member`](crate::set::Member), which is
//! [`listpack::Entry`](crate::listpack::Entry) under another name, so a value
//! stored as an integer is handed over as one and formatted once, into the reply
//! buffer, at the moment the reply is built. That is Y18 again.

use std::cell::RefCell;
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

    /// The same walk the other way.
    ///
    /// Both bands keep a length behind every element, so this costs what the
    /// forward walk costs. `LPOS` with a negative rank is what wants it.
    pub fn iter_back(&self) -> impl Iterator<Item = Element<'_>> {
        let (packed, chunks) = match &self.body {
            Body::Packed(lp) => (Some(lp.iter_back()), None),
            Body::Chunks(d) => (None, Some(d.iter_back())),
        };
        packed
            .into_iter()
            .flatten()
            .chain(chunks.into_iter().flatten())
    }

    /// `count` elements starting at `start`, which is `LRANGE`.
    ///
    /// Both ends are already normalised by the caller, because the wire's start
    /// and stop can be negative, can be the wrong way round and can hang off
    /// either end, and every one of those turns into an empty reply rather than
    /// into an error.
    ///
    /// A window in the middle does not walk to its start. The packed band skips
    /// entries because that is all a hundred and twenty eight of them costs, and
    /// the chunked band steps over whole chunks and only decodes the ones it is
    /// going to hand back. `LRANGE mylist 500000 500099` on a million element
    /// list reads a hundred elements and not five hundred thousand.
    pub fn range(&self, start: usize, count: usize) -> impl Iterator<Item = Element<'_>> {
        let (packed, chunks) = match &self.body {
            Body::Packed(lp) => (Some(lp.iter_from(start).take(count)), None),
            Body::Chunks(d) => (None, Some(d.range(start, count))),
        };
        packed
            .into_iter()
            .flatten()
            .chain(chunks.into_iter().flatten())
    }

    /// Put `value` at the front.
    pub fn push_front(&mut self, value: &[u8], limits: &Limits) {
        self.grow_by(value, limits);
        match &mut self.body {
            Body::Packed(lp) => lp.insert(0, value),
            Body::Chunks(d) => d.push_front(value),
        }
    }

    /// Put `value` at the back.
    pub fn push_back(&mut self, value: &[u8], limits: &Limits) {
        self.grow_by(value, limits);
        match &mut self.body {
            Body::Packed(lp) => lp.push(value),
            Body::Chunks(d) => d.push_back(value),
        }
    }

    /// Put `value` in at `index`, pushing what was there along, which is the
    /// half of `LINSERT` that already knows where the pivot was.
    ///
    /// An index equal to the length appends. Anything past that is nothing.
    pub fn insert(&mut self, index: usize, value: &[u8], limits: &Limits) -> bool {
        if index > self.len() {
            return false;
        }
        self.grow_by(value, limits);
        match &mut self.body {
            Body::Packed(lp) => {
                lp.insert(index, value);
                true
            }
            Body::Chunks(d) => d.insert_at(index, value),
        }
    }

    /// Put `value` next to the first `pivot` in the list, which is `LINSERT`.
    ///
    /// Gives back the new length, or nothing when the pivot is not there, which
    /// is the difference between the reply being a length and being `-1`.
    pub fn insert_at_pivot(
        &mut self,
        pivot: &[u8],
        value: &[u8],
        before: bool,
        limits: &Limits,
    ) -> Option<usize> {
        let at = self.find(pivot)?;
        let at = if before { at } else { at + 1 };
        self.insert(at, value, limits).then(|| self.len())
    }

    /// Put `value` where the element at `index` is, which is `LSET`.
    pub fn set(&mut self, index: usize, value: &[u8], limits: &Limits) -> bool {
        if index >= self.len() {
            return false;
        }
        // What the blob would weigh after the swap, which is not what it weighs
        // now plus the new element: the old one is going away. Getting this
        // wrong would promote a list that still fits and then keep it promoted,
        // because a list only converts back under half the limit.
        if let Body::Packed(lp) = &self.body {
            let old = lp.get(index).map_or(0, |e| e.byte_len());
            let after = lp.byte_len() + crate::listpack::entry_len(value) - old;
            self.grow_to(after, self.len(), limits);
        }
        match &mut self.body {
            Body::Packed(lp) => lp.replace(index, value),
            Body::Chunks(d) => d.replace_at(index, value),
        }
    }

    /// Where the first `value` is, front to back.
    #[must_use]
    pub fn find(&self, value: &[u8]) -> Option<usize> {
        let as_int = yo_common::num::parse_i64(value);
        self.iter().position(|e| e.is(value, as_int))
    }

    /// Where `value` is, as many times as asked, which is `LPOS`.
    ///
    /// `rank` is which match to start at and which way to look: 1 is the first
    /// from the front, -1 the first from the back, 2 the second from the front.
    /// `count` is how many to give back with 0 meaning all of them, and `maxlen`
    /// is how many elements may be compared before giving up, with 0 meaning no
    /// limit. The indexes handed back are always from the front, whichever way
    /// the walk went, because that is what the client can use.
    ///
    /// Each answer is handed to `found` as it is discovered, and the number of
    /// them comes back, because this runs on a shard thread and a shard thread
    /// that allocates aborts. The wire writes each position straight into the
    /// reply buffer and never holds a list of them at all.
    ///
    /// `found` is a `dyn` call rather than a generic, so that the two walks
    /// below stay one body. Monomorphising this over the sink would double a
    /// function whose whole cost is the comparison inside it.
    pub fn positions(
        &self,
        value: &[u8],
        rank: i64,
        count: usize,
        maxlen: usize,
        found: &mut dyn FnMut(usize),
    ) -> usize {
        if rank == 0 {
            return 0;
        }
        let as_int = yo_common::num::parse_i64(value);
        let len = self.len();
        let skip = rank.unsigned_abs() as usize - 1;
        let mut seen = 0usize;
        let mut looked = 0usize;
        let mut hits = 0usize;
        // One loop over one of the two walks, with the index worked out from
        // whichever end the walk started at.
        let mut take = |at: usize, e: Element<'_>| -> bool {
            looked += 1;
            if e.is(value, as_int) {
                seen += 1;
                if seen > skip {
                    found(at);
                    hits += 1;
                    if count != 0 && hits >= count {
                        return false;
                    }
                }
            }
            maxlen == 0 || looked < maxlen
        };
        if rank > 0 {
            for (at, e) in self.iter().enumerate() {
                if !take(at, e) {
                    break;
                }
            }
        } else {
            for (back, e) in self.iter_back().enumerate() {
                if !take(len - back - 1, e) {
                    break;
                }
            }
        }
        hits
    }

    /// Take out up to `count` elements equal to `value`, which is `LREM`.
    ///
    /// A positive count works from the front, a negative one from the back, and
    /// zero means every one of them. Gives back how many went.
    pub fn remove(&mut self, count: i64, value: &[u8], limits: &Limits) -> usize {
        let as_int = yo_common::num::parse_i64(value);
        let want = if count == 0 {
            usize::MAX
        } else {
            count.unsigned_abs() as usize
        };
        // Collected first and removed after, because removing during the walk
        // moves the elements the walk has not reached yet. Highest index first
        // so that the ones still to go do not move either.
        let mut hits = Vec::new();
        if count >= 0 {
            for (at, e) in self.iter().enumerate() {
                if e.is(value, as_int) {
                    hits.push(at);
                    if hits.len() >= want {
                        break;
                    }
                }
            }
            hits.reverse();
        } else {
            let len = self.len();
            for (back, e) in self.iter_back().enumerate() {
                if e.is(value, as_int) {
                    hits.push(len - back - 1);
                    if hits.len() >= want {
                        break;
                    }
                }
            }
        }
        for at in &hits {
            self.remove_at(*at);
        }
        self.shrunk(limits);
        hits.len()
    }

    /// Take out the element at `index`.
    ///
    /// The band is left alone, because a caller taking several out in a row
    /// would otherwise convert between them. Everything public that removes
    /// finishes with [`List::shrunk`].
    fn remove_at(&mut self, index: usize) -> bool {
        match &mut self.body {
            Body::Packed(lp) => lp.delete(index, 1),
            Body::Chunks(d) => d.remove_at(index),
        }
    }

    /// Keep `count` elements starting at `start` and drop the rest, which is
    /// `LTRIM`.
    ///
    /// Both ends are normalised by the caller, the same as [`List::range`], and
    /// a count of zero empties the list, which on the wire deletes the key.
    pub fn trim(&mut self, start: usize, count: usize, limits: &Limits) {
        let len = self.len();
        let start = start.min(len);
        let keep = count.min(len - start);
        match &mut self.body {
            Body::Packed(lp) => {
                lp.delete(start + keep, len - start - keep);
                lp.delete(0, start);
            }
            Body::Chunks(d) => d.trim(start, keep),
        }
        self.shrunk(limits);
    }

    /// Drop the first element, and say whether there was one.
    ///
    /// The read and the removal are separate so that `LPOP` on the wire can
    /// write the element straight into the reply buffer and then drop it,
    /// which is the same split [`crate::set::Set::drop_at`] exists for.
    pub fn drop_front(&mut self, limits: &Limits) -> bool {
        let gone = match &mut self.body {
            Body::Packed(lp) => lp.delete(0, 1),
            Body::Chunks(d) => d.drop_front(),
        };
        self.shrunk(limits);
        gone
    }

    /// Drop the last element, and say whether there was one.
    pub fn drop_back(&mut self, limits: &Limits) -> bool {
        let gone = match &mut self.body {
            Body::Packed(lp) => {
                let last = lp.len().checked_sub(1);
                last.is_some_and(|at| lp.delete(at, 1))
            }
            Body::Chunks(d) => d.drop_back(),
        };
        self.shrunk(limits);
        gone
    }

    /// Take the first element out and hand it back.
    ///
    /// The embedded API's `LPOP`, where the caller wants the bytes and has
    /// nowhere to put them.
    pub fn pop_front(&mut self, limits: &Limits) -> Option<Vec<u8>> {
        let out = self.front()?.to_vec();
        self.drop_front(limits);
        Some(out)
    }

    /// Take the last element out and hand it back.
    pub fn pop_back(&mut self, limits: &Limits) -> Option<Vec<u8>> {
        let out = self.back()?.to_vec();
        self.drop_back(limits);
        Some(out)
    }

    /// Go back to one blob if the list has shrunk far enough to deserve it.
    ///
    /// Called by everything here that removes elements. Redis converts back only
    /// below half the limit, so that a list sitting on the boundary does not
    /// rebuild itself on every other command, and it only converts a quicklist
    /// that is down to one node. Both of those are here.
    fn shrunk(&mut self, limits: &Limits) {
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

    /// Promote out of the packed band if one more `value` would not fit in it.
    ///
    /// Asked before the write rather than after, because a listpack that has
    /// already been grown past the limit and is then converted has done the
    /// work twice.
    fn grow_by(&mut self, value: &[u8], limits: &Limits) {
        let Body::Packed(lp) = &self.body else {
            return;
        };
        let after = lp.byte_len() + crate::listpack::entry_len(value);
        self.grow_to(after, lp.len() + 1, limits);
    }

    /// Promote out of the packed band if a list of this size does not fit it.
    fn grow_to(&mut self, bytes: usize, entries: usize, limits: &Limits) {
        if !matches!(self.body, Body::Packed(_)) || !limits.exceeded(bytes, entries) {
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

/// A chunk of its own holding nothing but `value`, at the end asked for.
///
/// An element too big for an ordinary chunk gets one sized to it, which is what
/// Redis calls a plain node. Without this a value over eight kilobytes would be
/// refused by a chunk that had just been made for it and the list would count an
/// element it does not hold.
fn lone(value: &[u8], front: bool) -> Chunk {
    if crate::listpack::entry_len(value) > CHUNK_BYTES {
        return Chunk::plain(value);
    }
    let mut c = if front {
        Chunk::for_front()
    } else {
        Chunk::for_back()
    };
    let put = if front {
        c.push_front(value)
    } else {
        c.push_back(value)
    };
    debug_assert!(put, "an empty chunk refused the only element in it");
    c
}

/// A ring of chunks, with the list's length kept beside it.
///
/// The length is carried rather than summed because `LLEN` is a command and
/// summing a thousand chunk counts to answer it would be a walk of the whole
/// list to say how long it is.
#[derive(Debug, Clone)]
struct Deque {
    chunks: VecDeque<Chunk>,
    len: usize,
    /// Where each chunk starts, so that finding an index is a binary search
    /// over the ring rather than a walk along it. See [`Deque::locate`].
    ///
    /// Behind a cell because it is filled in by reads, and the reads that want
    /// it take `&self`. Nothing outside this thread can see it: a shard owns
    /// its keyspace and `yo-shard` has a test that the type system says so.
    starts: RefCell<VecDeque<i64>>,
}

/// Two rings are the same when they hold the same elements in the same chunks.
///
/// Written out rather than derived because the start index is a cache, and a
/// list that has been read is not a different list from one that has not.
impl PartialEq for Deque {
    fn eq(&self, other: &Deque) -> bool {
        self.len == other.len && self.chunks == other.chunks
    }
}

impl Eq for Deque {}

impl Deque {
    /// An empty ring.
    fn new() -> Deque {
        Deque {
            chunks: VecDeque::new(),
            len: 0,
            starts: RefCell::new(VecDeque::new()),
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
        self.tail_added();
    }

    /// What the whole ring costs.
    ///
    /// A chunk counts its own header, because it is sitting in the ring's own
    /// allocation, so what is left to add is the slots the ring is holding empty
    /// for the chunks it does not have yet. The start index goes in as well,
    /// because it is eight bytes a chunk that the list would not otherwise be
    /// holding, and a structure that hides part of itself from `MEMORY USAGE`
    /// is worse than one that costs a little more.
    fn memory_bytes(&self) -> usize {
        let spare = self.chunks.capacity() - self.chunks.len();
        self.chunks.iter().map(Chunk::memory_bytes).sum::<usize>()
            + spare * size_of::<Chunk>()
            + self.starts.borrow().capacity() * size_of::<i64>()
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
        let (i, within) = self.locate(index)?;
        self.chunks[i].get(within)
    }

    /// A forward walk over every chunk in turn.
    fn iter(&self) -> impl Iterator<Item = Element<'_>> {
        self.chunks.iter().flat_map(Chunk::iter)
    }

    /// `count` elements from `start`, without walking to `start`.
    ///
    /// The chunks before the one holding `start` are stepped over as chunks, so
    /// the only elements this decodes are the ones inside the chunk it lands in
    /// and the ones it is going to return. A `skip` on the element walk decodes
    /// every entry it passes, which turned a hundred element window in the
    /// middle of a million element list into two milliseconds of reading
    /// listpack headers nobody asked for.
    fn range(&self, start: usize, count: usize) -> impl Iterator<Item = Element<'_>> {
        let (chunk, within) = self.locate(start).unwrap_or((self.chunks.len(), 0));
        let first = self.chunks.get(chunk).map(|c| c.iter_from(within));
        first
            .into_iter()
            .flatten()
            .chain(self.chunks.iter().skip(chunk + 1).flat_map(Chunk::iter))
            .take(count)
    }

    /// The same walk the other way, chunks in reverse and each one backward.
    fn iter_back(&self) -> impl Iterator<Item = Element<'_>> {
        self.chunks.iter().rev().flat_map(Chunk::iter_back)
    }

    /// Which chunk holds the element at `index`, and where in that chunk.
    ///
    /// This used to walk the ring from whichever end was closer, which is fine
    /// for a queue and terrible for anything that reads the middle: a million
    /// element list is a few thousand chunks, and a `LINDEX` halfway along it
    /// stepped over half of them to get there. That was two and a half
    /// microseconds against a hundred and thirty nanoseconds for the same call
    /// near an end.
    ///
    /// Now the ring carries where each chunk starts and the lookup is a binary
    /// search. `08` section 6 puts it as chunk count arithmetic plus one chunk
    /// walk, and the arithmetic is this.
    ///
    /// The starts are in their own coordinate system, whose origin is wherever
    /// the head chunk happened to be when the index was last built. What a
    /// lookup uses is the difference between two entries and never an entry on
    /// its own, so the origin can be anything, and that is what makes work at
    /// the front free: pushing an element on to the head chunk moves that
    /// chunk's start back by one and leaves every other entry correct, where an
    /// index of real positions would have had to add one to all of them.
    ///
    /// Only the first `starts.len()` chunks are described. A mutation in the
    /// middle of the ring cuts the index back to the chunk it touched and
    /// nothing more, so the mutation itself never walks, and the next lookup
    /// that needs the rest pays for it once.
    fn locate(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.len {
            return None;
        }
        // The two end chunks are answered by a comparison each, before any of
        // the above. A list is a queue and the position a client asks for is
        // usually near an end, and a binary search over a few thousand entries
        // is eleven scattered loads to say what one subtraction already knew.
        // Without this the index made `LINDEX mylist 3` half again as slow as
        // the walk it replaced.
        let head = self.chunks.front()?.len();
        if index < head {
            return Some((0, index));
        }
        let last = self.chunks.len() - 1;
        let before_tail = self.len - self.chunks[last].len();
        if index >= before_tail {
            return Some((last, index - before_tail));
        }
        let mut starts = self.starts.borrow_mut();
        if starts.is_empty() {
            starts.push_back(0);
        }
        // Carry the index on from where the last lookup or the last mutation
        // left it, and only as far as this index needs. A read near the front
        // of a ring that was just cut does not describe the whole ring to
        // answer.
        let want = starts[0] + index as i64;
        loop {
            let last = starts.len() - 1;
            let end = starts[last] + self.chunks[last].len() as i64;
            if end > want || starts.len() == self.chunks.len() {
                break;
            }
            starts.push_back(end);
        }
        // The last chunk that starts at or before the wanted position. An empty
        // chunk starts where the next one does, and this lands on the later of
        // the two, which is the one holding the element.
        let at = starts.partition_point(|&s| s <= want) - 1;
        Some((at, (want - starts[at]) as usize))
    }

    /// Forget where every chunk from `from` onward starts.
    ///
    /// Cheap on purpose. Every mutation in the middle of the ring calls this
    /// and none of them rebuild anything, because the next lookup will.
    #[inline]
    fn cut(&mut self, from: usize) {
        let keep = from.min(self.chunks.len());
        let starts = self.starts.get_mut();
        if starts.len() > keep {
            starts.truncate(keep);
        }
    }

    /// The head chunk's first element moved `by` places later.
    ///
    /// Negative for a push, positive for a pop. One subtraction, whatever the
    /// ring is holding, which is the whole point of the floating origin.
    #[inline]
    fn head_moved(&mut self, by: i64) {
        if let Some(first) = self.starts.get_mut().front_mut() {
            *first += by;
        }
    }

    /// A chunk holding `len` elements went on the front of the ring.
    #[inline]
    fn head_added(&mut self, len: usize) {
        let starts = self.starts.get_mut();
        if let Some(&first) = starts.front() {
            starts.push_front(first - len as i64);
        }
    }

    /// The head chunk left the ring, with everything that was in it.
    #[inline]
    fn head_dropped(&mut self) {
        self.starts.get_mut().pop_front();
    }

    /// A chunk went on the back of the ring.
    ///
    /// Described only if everything before it already is, which is the case
    /// that matters: a list being filled with `RPUSH` grows a chunk at a time
    /// and never invalidates anything, so the index is complete by the time
    /// anybody reads the middle of it.
    #[inline]
    fn tail_added(&mut self) {
        let n = self.chunks.len();
        let before = if n >= 2 { self.chunks[n - 2].len() } else { 0 };
        let starts = self.starts.get_mut();
        if n == 1 && starts.is_empty() {
            starts.push_back(0);
        } else if starts.len() + 1 == n {
            let last = starts[n - 2];
            starts.push_back(last + before as i64);
        }
    }

    /// Put `value` at the front, in the head chunk or in a new one.
    fn push_front(&mut self, value: &[u8]) {
        if let Some(head) = self.chunks.front_mut()
            && head.push_front(value)
        {
            self.len += 1;
            self.head_moved(-1);
            return;
        }
        // The chunk that was the head stops being an end, so it gives back the
        // room it was keeping. One that is empty goes instead, because a chunk
        // holding nothing is one every walk from that end has to step over.
        if self.chunks.front().is_some_and(Chunk::is_empty) {
            self.chunks.pop_front();
            self.head_dropped();
        } else if let Some(head) = self.chunks.front_mut() {
            head.seal();
        }
        self.chunks.push_front(lone(value, true));
        self.len += 1;
        self.head_added(1);
    }

    /// Put `value` at the back, in the tail chunk or in a new one.
    fn push_back(&mut self, value: &[u8]) {
        if let Some(tail) = self.chunks.back_mut()
            && tail.push_back(value)
        {
            self.len += 1;
            return;
        }
        if self.chunks.back().is_some_and(Chunk::is_empty) {
            self.chunks.pop_back();
            self.cut(self.chunks.len());
        } else if let Some(tail) = self.chunks.back_mut() {
            tail.seal();
        }
        self.chunks.push_back(lone(value, false));
        self.len += 1;
        self.tail_added();
    }

    /// Put `value` in at `index`, splitting a chunk if it will not take it.
    ///
    /// A chunk that refuses is split at the insertion point, which leaves two
    /// chunks with room between them for what would not fit. That is Redis's
    /// `_quicklistSplitNode` and the reason is the same: the alternative is
    /// pushing the rest of the list along one chunk at a time.
    fn insert_at(&mut self, index: usize, value: &[u8]) -> bool {
        if index > self.len {
            return false;
        }
        if index == 0 {
            self.push_front(value);
            return true;
        }
        if index == self.len {
            self.push_back(value);
            return true;
        }
        let Some((i, within)) = self.locate(index) else {
            return false;
        };
        if self.chunks[i].insert_at(within, value) {
            self.len += 1;
            // Chunk `i` still starts where it did. Everything after it moved.
            self.cut(i + 1);
            return true;
        }
        let mut rest = self.chunks[i].split_off(within);
        // Both halves have room now unless the element needs a chunk of its own.
        let put = self.chunks[i].push_back(value) || rest.push_front(value);
        self.chunks.insert(i + 1, rest);
        if !put {
            self.chunks.insert(i + 1, lone(value, false));
        }
        self.len += 1;
        self.cut(i + 1);
        true
    }

    /// Take the element at `index` out.
    fn remove_at(&mut self, index: usize) -> bool {
        let Some((i, within)) = self.locate(index) else {
            return false;
        };
        if !self.chunks[i].remove_at(within) {
            return false;
        }
        self.len -= 1;
        if self.chunks[i].is_empty() && self.chunks.len() > 1 {
            self.chunks.remove(i);
            // The chunk that takes its place starts where the empty one did,
            // so `i` is still right, but keeping it would be an argument and
            // cutting it is a memory write.
            self.cut(i);
        } else {
            self.cut(i + 1);
        }
        true
    }

    /// Put `value` where the element at `index` is.
    ///
    /// A replacement that does not fit is the same split an insert does, with
    /// the element being replaced dropped off the front of the second half.
    fn replace_at(&mut self, index: usize, value: &[u8]) -> bool {
        let Some((i, within)) = self.locate(index) else {
            return false;
        };
        if self.chunks[i].replace_at(within, value) {
            // One element out and one in, so no chunk moved and the index is
            // still true. This is the common case and it costs nothing.
            return true;
        }
        let mut rest = self.chunks[i].split_off(within);
        rest.drop_front();
        let put = self.chunks[i].push_back(value) || rest.push_front(value);
        if !rest.is_empty() {
            self.chunks.insert(i + 1, rest);
        }
        if !put {
            self.chunks.insert(i + 1, lone(value, false));
        }
        if self.chunks[i].is_empty() && self.chunks.len() > 1 {
            self.chunks.remove(i);
        }
        self.cut(i);
        true
    }

    /// Keep `keep` elements from `start` and drop everything else.
    ///
    /// Whole chunks at either end go without their bytes being touched, and the
    /// two chunks the range ends inside move a cursor. A trim of a million
    /// element list down to ten is the walk over the chunk list and two walks
    /// inside a chunk.
    fn trim(&mut self, start: usize, keep: usize) {
        let mut front = start;
        while front > 0 {
            let Some(held) = self.chunks.front().map(Chunk::len) else {
                break;
            };
            if held <= front && self.chunks.len() > 1 {
                front -= held;
                self.len -= held;
                self.chunks.pop_front();
                self.head_dropped();
            } else {
                let took = self.chunks[0].drop_front_n(front);
                self.len -= took;
                front -= took;
                self.head_moved(took as i64);
                if took == 0 {
                    break;
                }
            }
        }
        let mut back = self.len - keep.min(self.len);
        while back > 0 {
            let Some(held) = self.chunks.back().map(Chunk::len) else {
                break;
            };
            if held <= back && self.chunks.len() > 1 {
                back -= held;
                self.len -= held;
                self.chunks.pop_back();
                self.cut(self.chunks.len());
            } else {
                let last = self.chunks.len() - 1;
                let took = self.chunks[last].drop_back_n(back);
                self.len -= took;
                back -= took;
                if took == 0 {
                    break;
                }
            }
        }
    }

    /// Drop the first element, dropping the chunk with it if it was the last.
    fn drop_front(&mut self) -> bool {
        let Some(head) = self.chunks.front_mut() else {
            return false;
        };
        if !head.drop_front() {
            return false;
        }
        let gone = head.is_empty();
        self.len -= 1;
        self.head_moved(1);
        if gone && self.chunks.len() > 1 {
            self.chunks.pop_front();
            self.head_dropped();
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
        let gone = tail.is_empty();
        self.len -= 1;
        if gone && self.chunks.len() > 1 {
            self.chunks.pop_back();
            self.cut(self.chunks.len());
        }
        true
    }

    /// Every start the index claims to know, checked against a walk.
    ///
    /// The index is maintained by hand at nine call sites and a wrong entry
    /// would hand back the wrong element without anything else noticing, so
    /// the tests that mutate a ring call this rather than trusting the
    /// argument that the call sites are right.
    #[cfg(test)]
    fn index_is_true(&self) {
        let starts = self.starts.borrow();
        assert!(
            starts.len() <= self.chunks.len(),
            "the index describes {} chunks and the ring holds {}",
            starts.len(),
            self.chunks.len()
        );
        let Some(&base) = starts.front() else {
            return;
        };
        let mut real = 0usize;
        for (i, &s) in starts.iter().enumerate() {
            assert_eq!(
                s - base,
                real as i64,
                "chunk {i} is indexed at {} and starts at {real}",
                s - base
            );
            real += self.chunks[i].len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(l: &List) -> Vec<Vec<u8>> {
        l.iter().map(|e| e.to_vec()).collect()
    }

    /// The same list in both bands, so a test can run its case over each.
    ///
    /// The elements differ in length between the two, because that is the only
    /// thing that decides which band a list of a given length is in, so every
    /// test over this compares against the list it was handed rather than
    /// against a literal.
    fn both_bands(n: usize) -> [List; 2] {
        let limits = Limits::default();
        let mut packed = List::new();
        let mut chunks = List::new();
        for i in 0..n {
            packed.push_back(format!("e{i}").as_bytes(), &limits);
            chunks.push_back(format!("e{i}:{}", "p".repeat(400)).as_bytes(), &limits);
        }
        assert_eq!(packed.encoding(), Encoding::Listpack);
        assert_eq!(chunks.encoding(), Encoding::Quicklist);
        [packed, chunks]
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
        assert_eq!(l.pop_front(&limits).unwrap(), b"a");
        assert_eq!(l.pop_back(&limits).unwrap(), b"c");
        assert_eq!(all(&l), vec![b"b".to_vec()]);
        assert_eq!(l.pop_front(&limits).unwrap(), b"b");
        assert!(l.pop_front(&limits).is_none());
        assert!(l.pop_back(&limits).is_none());
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
        assert_eq!(l.pop_front(&limits).unwrap(), b"first");
        assert_eq!(l.pop_back(&limits).unwrap(), b"last");
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
                l.pop_front(&limits).unwrap(),
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
                l.pop_front(&limits).unwrap(),
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
            l.drop_back(&limits);
        }
        assert_eq!(
            l.encoding(),
            Encoding::Quicklist,
            "under the limit is not under half of it"
        );
        while l.len() > 50 {
            l.drop_back(&limits);
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
            l.drop_back(&limits);
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

    /// What a list element costs on top of the bytes it holds.
    ///
    /// M4's exit gate asks for one byte or less per element and this is the
    /// number that says whether that is where we are. Printed rather than
    /// asserted, because the point is the breakdown and not a threshold. The
    /// guard below is the part that runs every time.
    ///
    /// Three element lengths, because the answer is a fixed cost per element
    /// plus a fixed cost per chunk, and one length cannot tell those apart.
    #[test]
    #[ignore = "a measurement, run it by name"]
    fn measure_bytes_per_element() {
        let limits = Limits::default();
        for len in [8usize, 16, 64] {
            for n in [128usize, 10_000, 1_000_000] {
                let (l, payload) = weighed(n, len, &limits);
                let total = l.memory_bytes();
                println!(
                    "n={n:<9} elem={len:<4} band={:<9} total={total:<11} payload={payload:<11} over_per_element={:.2}",
                    l.encoding().name(),
                    (total as f64 - payload as f64) / n as f64
                );
            }
        }
    }

    /// A list of `n` elements of `len` bytes each, and what those bytes come to.
    fn weighed(n: usize, len: usize, limits: &Limits) -> (List, usize) {
        let mut l = List::new();
        let mut payload = 0usize;
        for i in 0..n {
            // A letter in front so that the element is stored as a string. A
            // listpack stores something that parses as an integer as one, which
            // would be measuring the integer encoding rather than the ring.
            let v = format!("e{i:0>w$}", w = len - 1);
            debug_assert_eq!(v.len(), len);
            payload += v.len();
            l.push_back(v.as_bytes(), limits);
        }
        (l, payload)
    }

    /// The guard for the measurement above, at a size that runs every time.
    ///
    /// The threshold is loose on purpose: what it is here to catch is a chunk
    /// that stopped giving its spare room back when it was sealed, or a ring
    /// that started holding something per element, and either of those is a
    /// multiple rather than a few percent.
    #[test]
    fn a_long_list_does_not_hold_much_more_than_it_stores() {
        let limits = Limits::default();
        let n = 100_000;
        let (l, payload) = weighed(n, 16, &limits);
        assert_eq!(l.encoding(), Encoding::Quicklist);
        let total = l.memory_bytes();
        assert!(
            total < payload + n * 4,
            "{total} bytes for {payload} of elements, which is {:.2} an element over",
            (total as f64 - payload as f64) / n as f64
        );
    }

    /// An element bigger than a whole chunk gets a chunk of its own, which is
    /// what Redis calls a plain node. Without it the list would count an element
    /// that a chunk sized for something else had refused.
    #[test]
    fn an_element_too_big_for_a_chunk_gets_one_of_its_own() {
        let mut l = List::new();
        let limits = Limits::default();
        let huge = vec![b'h'; 20_000];
        l.push_back(&huge, &limits);
        assert_eq!(l.len(), 1);
        assert_eq!(l.encoding(), Encoding::Quicklist);
        assert_eq!(l.front().unwrap().to_vec(), huge);
        l.push_back(b"after", &limits);
        l.push_front(b"before", &limits);
        assert_eq!(l.len(), 3);
        assert_eq!(l.get(1).unwrap().to_vec(), huge);
        assert_eq!(l.back().unwrap().to_vec(), b"after");
        assert_eq!(l.front().unwrap().to_vec(), b"before");
        assert_eq!(l.pop_front(&limits).unwrap(), b"before");
        assert_eq!(l.pop_front(&limits).unwrap(), huge);
    }

    #[test]
    fn the_walk_backward_is_the_walk_forward_reversed() {
        for mut l in [List::new(), chunked(400)] {
            let limits = Limits::default();
            l.push_back(b"tail", &limits);
            let mut want = all(&l);
            want.reverse();
            let got: Vec<Vec<u8>> = l.iter_back().map(|e| e.to_vec()).collect();
            assert_eq!(got, want, "{:?}", l.encoding());
        }
    }

    #[test]
    fn a_range_is_the_window_it_was_asked_for() {
        for l in both_bands(50) {
            let all_of_it = all(&l);
            for (start, count) in [(0, 0), (0, 5), (3, 4), (48, 9), (50, 3), (0, 50)] {
                let got: Vec<Vec<u8>> = l.range(start, count).map(|e| e.to_vec()).collect();
                let want = &all_of_it[start.min(50)..(start + count).min(50)];
                assert_eq!(got, want, "{start} for {count} in {:?}", l.encoding());
            }
        }
    }

    /// A window that starts in the middle now steps over whole chunks to get
    /// there instead of decoding every element on the way, so every start
    /// position and every window that crosses a chunk boundary is worth
    /// checking rather than the handful the case above uses.
    #[test]
    fn a_window_lands_in_the_right_place_whatever_chunk_it_starts_in() {
        let limits = Limits::default();
        let mut l = List::new();
        // Long enough elements that this is many chunks and not one, and enough
        // of them that a start position lands in the middle of a chunk, at the
        // front of one, and at the back of one.
        for i in 0..500 {
            l.push_back(format!("e{i}:{}", "p".repeat(200)).as_bytes(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Quicklist);
        let all_of_it = all(&l);

        for start in 0..=500 {
            for count in [0usize, 1, 7, 130, 500] {
                let got: Vec<Vec<u8>> = l.range(start, count).map(|e| e.to_vec()).collect();
                let want = &all_of_it[start.min(500)..(start + count).min(500)];
                assert_eq!(got, want, "{count} from {start}");
            }
        }
    }

    /// The same over a packed list, which seeks by walking the blob from
    /// whichever end is nearer rather than by finding a chunk. A list in this
    /// band holds eight kilobytes, which is four hundred odd elements and not
    /// the hundred and twenty eight the other packed bands stop at, so the half
    /// of the blob that the two ended seek saves is worth having and the seam
    /// between the two directions is worth checking at every position.
    #[test]
    fn a_packed_window_lands_in_the_right_place_from_either_end() {
        let limits = Limits::default();
        let mut l = List::new();
        for i in 0..400 {
            l.push_back(format!("e{i:0>9}").as_bytes(), &limits);
        }
        assert_eq!(l.encoding(), Encoding::Listpack);
        let all_of_it = all(&l);

        for start in 0..=400 {
            assert_eq!(
                l.get(start).map(|e| e.to_vec()).as_ref(),
                all_of_it.get(start),
                "element {start}"
            );
            for count in [0usize, 1, 7, 130, 400] {
                let got: Vec<Vec<u8>> = l.range(start, count).map(|e| e.to_vec()).collect();
                let want = &all_of_it[start.min(400)..(start + count).min(400)];
                assert_eq!(got, want, "{count} from {start}");
            }
        }
    }

    /// The chunk start index has a floating origin so that work at the front of
    /// the list costs it nothing, which is the one part of it that is clever
    /// enough to be wrong. This is the shape that would catch it: a queue being
    /// drained and refilled at the head while something reads the middle, where
    /// an index of real positions would need every entry rewritten on every
    /// push and this one moves a single number.
    #[test]
    fn reading_the_middle_survives_a_head_that_keeps_moving() {
        let limits = Limits::default();
        let mut l = List::new();
        let mut want: Vec<Vec<u8>> = Vec::new();
        for i in 0..2000 {
            let v = format!("e{i}:{}", "p".repeat(100)).into_bytes();
            l.push_back(&v, &limits);
            want.push(v);
        }
        assert_eq!(l.encoding(), Encoding::Quicklist);

        for round in 0..400 {
            // Enough pushes and pops to walk the head chunk across its own
            // boundary in both directions rather than only inside it.
            if round % 3 == 0 {
                for k in 0..7 {
                    let v = format!("h{round}:{k}:{}", "q".repeat(100)).into_bytes();
                    l.push_front(&v, &limits);
                    want.insert(0, v);
                }
            } else {
                for _ in 0..5 {
                    assert_eq!(l.pop_front(&limits), Some(want.remove(0)));
                }
            }
            assert_eq!(l.len(), want.len(), "length after round {round}");
            for at in [0, 1, want.len() / 3, want.len() / 2, want.len() - 1] {
                assert_eq!(
                    l.get(at).map(|e| e.to_vec()).as_ref(),
                    Some(&want[at]),
                    "element {at} after round {round}"
                );
            }
            let mid = want.len() / 2;
            let got: Vec<Vec<u8>> = l.range(mid, 30).map(|e| e.to_vec()).collect();
            assert_eq!(got, want[mid..mid + 30], "the window after round {round}");
            let Body::Chunks(d) = &l.body else {
                panic!("the list left the chunked band");
            };
            d.index_is_true();
        }
    }

    #[test]
    fn setting_an_element_replaces_only_that_one() {
        for mut l in both_bands(50) {
            let limits = Limits::default();
            let before = all(&l);
            let band = l.encoding();
            for at in [0usize, 1, 25, 49] {
                let mut want = before.clone();
                for value in [
                    &b"z"[..],
                    &b"a much longer element than the one there"[..],
                    b"42",
                ] {
                    assert!(l.set(at, value, &limits), "setting {at} in {band:?}");
                    want[at] = value.to_vec();
                    assert_eq!(all(&l), want, "setting {at} to {value:?} in {band:?}");
                }
                l.set(at, &before[at], &limits);
            }
            assert!(!l.set(50, b"z", &limits), "past the end is not a set");
            assert_eq!(all(&l), before);
        }
    }

    #[test]
    fn an_insert_goes_where_the_pivot_is() {
        for mut l in both_bands(50) {
            let limits = Limits::default();
            let before = all(&l);
            let band = l.encoding();
            let pivot = before[25].clone();
            assert_eq!(
                l.insert_at_pivot(&pivot, b"before", true, &limits),
                Some(51)
            );
            assert_eq!(
                l.insert_at_pivot(&pivot, b"after", false, &limits),
                Some(52)
            );
            assert_eq!(l.get(25).unwrap().to_vec(), b"before", "{band:?}");
            assert_eq!(l.get(26).unwrap().to_vec(), pivot, "{band:?}");
            assert_eq!(l.get(27).unwrap().to_vec(), b"after", "{band:?}");
            assert_eq!(l.len(), 52);
            assert_eq!(
                l.insert_at_pivot(b"nothing like it", b"x", true, &limits),
                None
            );
            assert_eq!(l.len(), 52);
        }
    }

    #[test]
    fn an_insert_by_index_takes_both_ends_and_the_middle() {
        for at in [0usize, 1, 200, 399, 400] {
            let mut l = chunked(400);
            let limits = Limits::default();
            let mut want = all(&l);
            assert!(l.insert(at, b"new", &limits), "inserting at {at}");
            want.insert(at, b"new".to_vec());
            assert_eq!(all(&l), want, "inserting at {at}");
            assert_eq!(l.len(), 401);
        }
        let mut l = chunked(400);
        assert!(!l.insert(401, b"new", &Limits::default()));
    }

    #[test]
    fn removing_by_value_counts_from_the_end_it_was_told_to() {
        let build = || {
            let mut l = List::new();
            let limits = Limits::default();
            for i in 0..40 {
                l.push_back(
                    if i % 3 == 0 {
                        b"x".to_vec()
                    } else {
                        format!("e{i}").into_bytes()
                    }
                    .as_slice(),
                    &limits,
                );
            }
            l
        };
        let limits = Limits::default();

        let mut l = build();
        assert_eq!(l.remove(0, b"x", &limits), 14, "every one of them");
        assert!(!all(&l).contains(&b"x".to_vec()));
        assert_eq!(l.len(), 26);

        let mut l = build();
        assert_eq!(l.remove(2, b"x", &limits), 2);
        assert_eq!(l.len(), 38);
        assert_eq!(l.get(0).unwrap().to_vec(), b"e1", "the first two went");

        let mut l = build();
        assert_eq!(l.remove(-2, b"x", &limits), 2);
        assert_eq!(l.get(0).unwrap().to_vec(), b"x", "the last two went");
        assert_eq!(l.back().unwrap().to_vec(), b"e38");

        let mut l = build();
        assert_eq!(l.remove(99, b"x", &limits), 14, "more than there are");
        assert_eq!(l.remove(1, b"nothing like it", &limits), 0);
    }

    #[test]
    fn a_trim_keeps_the_window_and_nothing_else() {
        for (start, count) in [
            (0usize, 400usize),
            (0, 10),
            (390, 10),
            (100, 200),
            (0, 0),
            (399, 1),
        ] {
            let mut l = chunked(400);
            let limits = Limits::default();
            let want = all(&l)[start..start + count].to_vec();
            l.trim(start, count, &limits);
            assert_eq!(all(&l), want, "keeping {count} from {start}");
            assert_eq!(l.len(), count);
        }
    }

    /// A trim that leaves a handful takes the list back to one blob, which is
    /// the shrinking rule reached the other way.
    #[test]
    fn a_trim_that_leaves_a_handful_goes_back_to_one_blob() {
        let mut l = chunked(400);
        let limits = Limits::default();
        l.trim(10, 5, &limits);
        assert_eq!(l.encoding(), Encoding::Listpack);
        assert_eq!(l.len(), 5);
        assert_eq!(
            l.get(0).unwrap().to_vec(),
            format!("value:{:0>60}", 10).into_bytes()
        );
    }

    #[test]
    fn a_position_is_counted_from_the_end_the_rank_asked_for() {
        for mut l in [List::new(), chunked(300)] {
            let limits = Limits::default();
            let band = l.encoding();
            for m in [b"a", b"b", b"a", b"c", b"a"] {
                l.push_back(m, &limits);
            }
            let base = l.len() - 5;
            let mut out = Vec::new();

            l.positions(b"a", 1, 1, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base], "{band:?}");

            out.clear();
            l.positions(b"a", 2, 1, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base + 2], "the second from the front");

            out.clear();
            l.positions(b"a", -1, 1, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base + 4], "the first from the back");

            out.clear();
            l.positions(b"a", -2, 1, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base + 2], "the second from the back");

            out.clear();
            l.positions(b"a", 1, 0, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base, base + 2, base + 4], "all of them");

            out.clear();
            l.positions(b"a", -1, 0, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base + 4, base + 2, base], "all of them backward");

            out.clear();
            l.positions(b"a", 1, 2, 0, &mut |at| out.push(at));
            assert_eq!(out, vec![base, base + 2], "two of them");

            out.clear();
            l.positions(b"nothing like it", 1, 0, 0, &mut |at| out.push(at));
            assert!(out.is_empty());

            out.clear();
            l.positions(b"a", 0, 0, 0, &mut |at| out.push(at));
            assert!(out.is_empty(), "a rank of zero is not a rank");
        }
    }

    /// `MAXLEN` bounds the comparisons and not the answers, so a match past it
    /// is not found however few answers have been collected.
    #[test]
    fn maxlen_stops_the_walk_rather_than_the_answers() {
        let mut l = List::new();
        let limits = Limits::default();
        for i in 0..20 {
            l.push_back(
                if i == 15 {
                    b"x".to_vec()
                } else {
                    format!("e{i}").into_bytes()
                }
                .as_slice(),
                &limits,
            );
        }
        let mut out = Vec::new();
        l.positions(b"x", 1, 0, 10, &mut |at| out.push(at));
        assert!(out.is_empty(), "ten comparisons do not reach the sixteenth");
        out.clear();
        l.positions(b"x", 1, 0, 16, &mut |at| out.push(at));
        assert_eq!(out, vec![15]);
        out.clear();
        l.positions(b"x", -1, 0, 5, &mut |at| out.push(at));
        assert_eq!(out, vec![15], "five from the back does reach it");
    }

    /// Every operation against a plain `Vec`, over a mix of element sizes that
    /// crosses the band boundary in both directions several times. A model test
    /// rather than more cases, because the interesting bugs here are the ones
    /// where a chunk splits and an index moves and the length stops agreeing.
    #[test]
    fn a_long_run_of_operations_agrees_with_a_vec() {
        // Twice, because the default limits keep a list of this size packed the
        // whole way and never walk the code that splits and joins chunks. A
        // `list-max-listpack-size` of 8 is a real setting and it puts the band
        // change within reach of a few pushes, so the second run crosses it in
        // both directions hundreds of times.
        let (_, chunked) = model_run(&Limits::default());
        assert_eq!(chunked, 0, "the default limits should not chunk this list");
        let (packed, chunked) = model_run(&Limits::of(8));
        assert!(packed > 200, "{packed} rounds packed");
        assert!(chunked > 200, "{chunked} rounds chunked");
    }

    /// Four thousand rounds of a fixed sequence of list operations against a
    /// `Vec` that says what the answer is, and the two band counts it saw.
    fn model_run(limits: &Limits) -> (usize, usize) {
        let mut l = List::new();
        let mut want: Vec<Vec<u8>> = Vec::new();
        // A fixed sequence rather than a random one, so a failure is a failure
        // every time it is run.
        let mut seed = 0x2064_u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 33) as usize
        };
        let (mut packed, mut chunked) = (0, 0);
        for round in 0..4000 {
            let n = next();
            let value = match n % 4 {
                0 => format!("{}", n % 97).into_bytes(),
                1 => vec![b'a' + (n % 26) as u8; 1 + n % 40],
                2 => vec![b'z'; 200 + n % 400],
                _ => format!("e{round}").into_bytes(),
            };
            match n % 9 {
                0 => {
                    l.push_front(&value, limits);
                    want.insert(0, value);
                }
                1 | 2 => {
                    l.push_back(&value, limits);
                    want.push(value);
                }
                3 if !want.is_empty() => {
                    let at = n % want.len();
                    assert!(l.insert(at, &value, limits));
                    want.insert(at, value);
                }
                4 if !want.is_empty() => {
                    let at = n % want.len();
                    assert!(l.set(at, &value, limits));
                    want[at] = value;
                }
                5 if !want.is_empty() => {
                    assert_eq!(l.pop_front(limits), Some(want.remove(0)));
                }
                6 if !want.is_empty() => {
                    assert_eq!(l.pop_back(limits), want.pop());
                }
                7 if want.len() > 4 => {
                    let start = n % (want.len() - 2);
                    let keep = 1 + n % (want.len() - start);
                    l.trim(start, keep, limits);
                    want = want[start..start + keep].to_vec();
                }
                8 if !want.is_empty() => {
                    let needle = want[n % want.len()].clone();
                    let count = [0i64, 1, -1, 3][n % 4];
                    let gone = l.remove(count, &needle, limits);
                    let mut hits: Vec<usize> = want
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| **m == needle)
                        .map(|(i, _)| i)
                        .collect();
                    if count < 0 {
                        hits.reverse();
                    }
                    if count != 0 {
                        hits.truncate(count.unsigned_abs() as usize);
                    }
                    assert_eq!(gone, hits.len(), "round {round}");
                    hits.sort_unstable();
                    for at in hits.iter().rev() {
                        want.remove(*at);
                    }
                }
                _ => {}
            }
            assert_eq!(l.len(), want.len(), "length after round {round}");
            // Read at both ends and in the middle every round. That builds the
            // chunk start index back up after whatever the round did to it, so
            // the audit below is checking a filled index and not an empty one,
            // and a stale entry shows up here as the wrong element rather than
            // as nothing at all.
            if !want.is_empty() {
                for at in [0, want.len() / 2, want.len() - 1] {
                    assert_eq!(
                        l.get(at).map(|e| e.to_vec()).as_ref(),
                        Some(&want[at]),
                        "element {at} after round {round}"
                    );
                }
            }
            if let Body::Chunks(d) = &l.body {
                d.index_is_true();
            }
            match l.encoding() {
                Encoding::Listpack => packed += 1,
                Encoding::Quicklist => chunked += 1,
            }
            if round % 25 == 0 {
                assert_eq!(all(&l), want, "contents after round {round}");
                let mut back = all(&l);
                back.reverse();
                let walked: Vec<Vec<u8>> = l.iter_back().map(|e| e.to_vec()).collect();
                assert_eq!(walked, back, "the backward walk after round {round}");
                if let Some(first) = want.first() {
                    assert_eq!(l.front().unwrap().to_vec(), *first);
                    assert_eq!(l.back().unwrap().to_vec(), *want.last().unwrap());
                    assert_eq!(l.find(first), Some(0));
                }
            }
        }
        assert_eq!(all(&l), want);
        (packed, chunked)
    }
}
