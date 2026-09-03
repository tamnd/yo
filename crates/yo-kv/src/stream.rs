//! A stream, as a log of listpack nodes in ID order (`08` section 7).
//!
//! A stream is the one collection here that is only ever appended to at one end
//! and only ever trimmed at the other. Nothing is inserted in the middle, ever,
//! because the ID of a new entry has to be greater than the last one there. That
//! is a much stronger promise than any other type makes, and the structure is
//! built on it rather than around it.
//!
//! ```text
//! +--------------+--------------+--------------+
//! | node         | node         | node         |
//! | master 5-0   | master 812-0 | master 990-0 |
//! | 100 entries  | 100 entries  | 12 entries   |
//! +--------------+--------------+--------------+
//!   ^ trimmed from here          appended here ^
//! ```
//!
//! # Why a node holds a hundred entries and not one
//!
//! One entry per allocation is what a naive log does and it is wrong twice over.
//! It costs an allocator header and a pointer per entry, which for a stream of
//! short entries is most of the memory, and it puts the entries in whatever
//! order the allocator felt like, which is the opposite of what a range scan
//! wants. A node is one blob holding a hundred consecutive entries, so a scan is
//! a walk through memory the prefetcher can see coming, and the per entry
//! overhead is a few bytes rather than a few dozen.
//!
//! Two tricks inside the node take it further down, and both are Redis's:
//!
//! The **ID is stored as a difference** from the node's first ID rather than
//! whole. Entries arrive milliseconds apart and a difference of a few hundred is
//! two bytes where a pair of 64 bit integers is sixteen.
//!
//! The **field names are stored once** for the node rather than once per entry.
//! A stream is almost always the same shape repeated, `sensor` and `reading`
//! over and over, so the first entry's field names become the node's master
//! fields and every later entry with exactly those names, in that order, stores
//! only its values. An entry with different fields stores its own names and
//! costs what it would have cost anyway.
//!
//! # Why the nodes are a deque and not a radix tree
//!
//! Redis keeps its nodes in a rax keyed by the sixteen byte big endian ID, and
//! `08` says radix log, so this is the place to say why there is no radix tree
//! here. The keys are appended in sorted order and never inserted between, which
//! means the index is a sorted array and stays one for free. Finding the node a
//! range starts in is then a binary search over the node count, which for a
//! million entries is fourteen steps over an array that is a few pages long. A
//! radix tree over the same keys is four or five levels of pointer chasing, each
//! one a cache miss, and it exists to solve the insertion problem that this
//! structure does not have.
//!
//! Trimming is what a plain `Vec` would get wrong, since it takes from the front,
//! so the nodes live in a `VecDeque` and dropping the oldest node is a pop.
//!
//! # The bytes are Redis's
//!
//! A node is a [`Listpack`] laid out exactly as `t_stream.c` lays one out, and
//! the node's first ID is held beside it exactly as the rax key holds it there.
//! That is not deference, it is the cheapest route to `DUMP` and `RESTORE` and
//! an RDB that Redis can read, since the node is already the thing that goes on
//! the wire and no conversion has to exist at all.
//!
//! ```text
//! master entry
//! +-------+---------+------------+---------+-----+---------+---+
//! | count | deleted | num-fields | field-1 | ... | field-N | 0 |
//! +-------+---------+------------+---------+-----+---------+---+
//!
//! an entry with the master's fields
//! +-------+---------+----------+---------+-----+---------+----------+
//! | flags | ms-diff | seq-diff | value-1 | ... | value-N | lp-count |
//! +-------+---------+----------+---------+-----+---------+----------+
//!
//! an entry with its own
//! +-------+---------+----------+------------+---------+---------+-----+----------+
//! | flags | ms-diff | seq-diff | num-fields | field-1 | value-1 | ... | lp-count |
//! +-------+---------+----------+------------+---------+---------+-----+----------+
//! ```
//!
//! `lp-count` is how many listpack elements the entry occupies before it, which
//! is what makes the node walkable backwards. It is written because Redis writes
//! it and the bytes have to match, and nothing here reads it yet: `XREVRANGE`
//! buffers a node's marks and hands them back reversed instead. That costs, and
//! the benchmark says how much, 6.83 microseconds against 1.95 for the same
//! hundred entries forwards. Stepping back over `lp-count` is the fix and it is
//! its own change, because it needs its own before and after.
//!
//! Both ID halves are a plain wrapping difference from the master, which is what
//! Redis writes and what it adds back. The sequence usually goes down when the
//! millisecond goes up, so the second difference is usually negative, and
//! wrapping arithmetic is the exact inverse either way.
//!
//! That claim is checked rather than asserted. A `DUMP` taken from Redis 8.10.1
//! is hard coded in the tests, and two of them run it in both directions: one
//! builds the same stream here and compares the node's bytes to Redis's byte for
//! byte, the other takes Redis's node and reads the entries back out of it.
//!
//! # Deleting does not move anything
//!
//! `XDEL` sets a bit in the entry's flags and leaves the bytes where they are.
//! Compacting the node would move every entry behind it, which on a node of a
//! hundred is a memmove per delete, and a stream is not a structure people
//! delete from in bulk. The node's master entry counts how many of its entries
//! are dead, and a node whose last live entry goes is dropped whole.

use std::cmp::Ordering;
use std::collections::VecDeque;

use yo_common::num::{DIGITS_MAX, i64_digits, push_u64, u64_digits};

use crate::frozen::{self, Broken};
use crate::listpack::{self, Entry, Listpack};

pub mod groups;

pub use groups::{Consumer, Filter, Group, Nack, Retry};

/// How many bytes a node holds before the next entry starts a new one.
///
/// `stream-node-max-bytes`, which is 4096 in Redis and is here for the same
/// reason: a node is rewritten in place when an entry is deleted from it and is
/// copied whole when it is written out, so a node that grows without limit
/// turns both of those into a problem.
pub const NODE_BYTES: usize = 4096;

/// How many entries a node holds before the next one starts a new node.
///
/// `stream-node-max-entries`, which is 100 in Redis.
pub const NODE_ENTRIES: usize = 100;

/// The entry is live.
const LIVE: i64 = 0;

/// The entry has been deleted and its bytes are still here.
const DELETED: i64 = 1;

/// The entry's field names are the node's master fields.
const SAME_FIELDS: i64 = 2;

/// Where the master entry's field names start.
///
/// After the count, the deleted count and the number of fields.
const MASTER_FIELDS: usize = 3;

/// An entry ID, which is a millisecond and a sequence number inside it.
///
/// Two 64 bit halves rather than one 128 bit number, because both halves are
/// addressable on the wire: `XADD key 5-*` asks for the next sequence inside
/// millisecond five, and `XRANGE key 5 5` is every sequence inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Id {
    /// The millisecond, which is a unix timestamp for an ID the server made up.
    pub ms: u64,
    /// Which entry within that millisecond.
    pub seq: u64,
}

impl Id {
    /// The lowest ID there is, which is what `-` means in a range.
    pub const MIN: Id = Id { ms: 0, seq: 0 };

    /// The highest, which is what `+` means.
    pub const MAX: Id = Id {
        ms: u64::MAX,
        seq: u64::MAX,
    };

    /// An ID from its two halves.
    #[must_use]
    #[inline]
    pub const fn new(ms: u64, seq: u64) -> Id {
        Id { ms, seq }
    }

    /// The next ID after this one, or `None` at [`Id::MAX`].
    ///
    /// What an exclusive range start turns into, and what `XADD key ms-*`
    /// resolves to when the millisecond is already the last one used.
    #[must_use]
    pub const fn next(self) -> Option<Id> {
        if self.seq != u64::MAX {
            Some(Id {
                ms: self.ms,
                seq: self.seq + 1,
            })
        } else if self.ms != u64::MAX {
            Some(Id {
                ms: self.ms + 1,
                seq: 0,
            })
        } else {
            None
        }
    }

    /// The ID before this one, or `None` at [`Id::MIN`].
    #[must_use]
    pub const fn prev(self) -> Option<Id> {
        if self.seq != 0 {
            Some(Id {
                ms: self.ms,
                seq: self.seq - 1,
            })
        } else if self.ms != 0 {
            Some(Id {
                ms: self.ms - 1,
                seq: u64::MAX,
            })
        } else {
            None
        }
    }

    /// The sixteen big endian bytes Redis keys a node by.
    ///
    /// Big endian because that is the order that sorts, which is the whole
    /// reason the format picked it.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.ms.to_be_bytes());
        out[8..].copy_from_slice(&self.seq.to_be_bytes());
        out
    }

    /// The ID those bytes hold.
    #[must_use]
    pub fn from_bytes(bytes: [u8; 16]) -> Id {
        let mut ms = [0u8; 8];
        let mut seq = [0u8; 8];
        ms.copy_from_slice(&bytes[..8]);
        seq.copy_from_slice(&bytes[8..]);
        Id {
            ms: u64::from_be_bytes(ms),
            seq: u64::from_be_bytes(seq),
        }
    }

    /// `ms-seq`, which is how an ID looks everywhere a client can see one.
    pub fn write_to(self, out: &mut Vec<u8>) {
        push_u64(out, self.ms);
        out.push(b'-');
        push_u64(out, self.seq);
    }

    /// The same as a fresh `Vec`, for a caller that is not building a reply.
    #[must_use]
    pub fn to_vec(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(41);
        self.write_to(&mut out);
        out
    }

    /// `ms` or `ms-seq`, with a missing sequence read as `default`.
    ///
    /// `XRANGE key 5 5` means every entry in millisecond five, so the start
    /// defaults its sequence to zero and the end to the largest there is. The
    /// special forms a client can send, `-`, `+`, `$`, `*` and `ms-*`, are the
    /// command layer's business and not this one's.
    #[must_use]
    pub fn parse(s: &[u8], default: u64) -> Option<Id> {
        let (ms, seq) = match s.iter().position(|c| *c == b'-') {
            Some(at) => (&s[..at], Some(&s[at + 1..])),
            None => (s, None),
        };
        Some(Id {
            ms: digits(ms)?,
            seq: match seq {
                Some(seq) => digits(seq)?,
                None => default,
            },
        })
    }
}

/// A run of digits as a `u64`, refusing a sign, a space or an empty string.
///
/// `parse_i64` would take `-1` and `+5` and this must not: an ID is unsigned
/// and the minus is the separator.
fn digits(s: &[u8]) -> Option<u64> {
    if s.is_empty() || s.len() > 20 {
        return None;
    }
    let mut n = 0u64;
    for c in s {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n.checked_mul(10)?.checked_add(u64::from(d))?;
    }
    Some(n)
}

/// Why an append was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// The ID is not greater than the last one in the stream.
    NotGreater,
    /// The ID is zero, which no entry can have because nothing sorts below it.
    Zero,
    /// The stream is at [`Id::MAX`] and there is no next ID to hand out.
    Full,
}

/// Where a node stops taking entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most bytes a node holds.
    pub max_node_bytes: usize,
    /// The most entries a node holds, live and deleted together.
    pub max_node_entries: usize,
}

impl Default for Limits {
    /// `stream-node-max-bytes 4096` and `stream-node-max-entries 100`.
    fn default() -> Limits {
        Limits {
            max_node_bytes: NODE_BYTES,
            max_node_entries: NODE_ENTRIES,
        }
    }
}

/// What a delete does about the consumer groups still pointing at the entry.
///
/// `XDEL` takes an entry out from under whoever was handed it and leaves the
/// pending list holding an ID that can never be read, which is a state `XCLAIM`
/// then has to clean up. The 8.2 commands let a caller say what it wants
/// instead, and the three answers are the three words here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Refs {
    /// `KEEPREF`: take the entry and leave every pending list alone, which is
    /// what `XDEL` has always done and is still the default.
    #[default]
    Keep,
    /// `DELREF`: take the entry and take it out of every pending list with it.
    Drop,
    /// `ACKED`: only take the entry when no group could still be handed it.
    Acked,
}

/// What one ID a delete was asked about came to.
///
/// The numbers are Redis's and they are not a success flag: a caller sending a
/// list of IDs gets one of these each and has to be able to tell an entry that
/// was never there from one that is still there on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    /// There was nothing here to do.
    Missing,
    /// It went.
    Gone,
    /// Somebody could still be handed it, so it stayed.
    Held,
}

impl Fate {
    /// The integer this is on the wire.
    #[must_use]
    #[inline]
    pub fn code(self) -> i64 {
        match self {
            Fate::Missing => -1,
            Fate::Gone => 1,
            Fate::Held => 2,
        }
    }
}

/// One node: a run of consecutive entries and the ID the run starts at.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    /// The ID every entry in this node stores its own as a difference from.
    ///
    /// It is the first entry's ID, and it stays that even after the first entry
    /// is deleted, because the differences behind it are relative to it.
    master: Id,
    lp: Listpack,
}

/// What a stream looks like from the outside, for working out a group's
/// counters.
///
/// Five numbers taken together at one moment. The reason they travel as a
/// bundle is that both rules below read several of them and the answers have to
/// come from the same instant, and the reason it is a private type is that
/// nobody outside wants five loose numbers, they want [`Stream::lag`].
#[derive(Debug, Clone, Copy)]
struct Edges {
    added: u64,
    length: u64,
    first: Option<Id>,
    last: Id,
    max_deleted: Id,
}

impl Edges {
    /// How many entries a reader sitting at `id` must have passed, when that is
    /// exactly knowable, which is the three positions where it is.
    ///
    /// At or past the last ID ever handed out, everything that was ever added
    /// has gone by. Below the oldest entry left, with nothing deleted inside
    /// what is left, everything that has gone is behind and the length is what
    /// is in front. Exactly on the oldest entry left, the same plus one.
    ///
    /// Anywhere else the answer is genuinely unknown rather than approximate,
    /// because working it out would mean counting the entries between here and
    /// there, and that is the walk the whole counter exists to avoid.
    fn estimate(&self, id: Id) -> Option<u64> {
        if self.added == 0 {
            return Some(0);
        }
        if id >= self.last {
            return Some(self.added);
        }
        let first = self.first?;
        // A hole below the oldest entry left is not a hole in what is left, so
        // the subtraction below still holds. That is the case a trim makes, and
        // it is why a trim does not cost a group its lag.
        if self.max_deleted != Id::MIN && self.max_deleted >= first {
            return None;
        }
        let behind = self.added - self.length;
        match id.cmp(&first) {
            Ordering::Less => Some(behind),
            Ordering::Equal => Some(behind + 1),
            Ordering::Greater => None,
        }
    }

    /// Whether anything has been deleted at or above `id`.
    ///
    /// Only [`Stream::delete`] moves `max_deleted`, so a trim does not count,
    /// and a stream with nothing left in it does not either, since there is no
    /// gap in an empty stream for a reader to fall into.
    fn holed_from(&self, id: Id) -> bool {
        if self.length == 0 || self.max_deleted == Id::MIN {
            return false;
        }
        if self.first.is_some_and(|first| first > self.max_deleted) {
            return false;
        }
        id <= self.max_deleted
    }

    /// A group's lag, the good way first and the subtraction second.
    fn lag(&self, group: &Group) -> Option<u64> {
        if let Some(read) = self.estimate(group.last_id()) {
            return Some(self.added.saturating_sub(read));
        }
        let read = group.entries_read()?;
        if self.holed_from(group.last_id()) {
            return None;
        }
        Some(self.added.saturating_sub(read))
    }

    /// What a group's read counter becomes once `id` has been handed out.
    ///
    /// One more than it was, while it is known and nothing has been deleted
    /// ahead of the entry just delivered. Otherwise the estimate above gets a
    /// go, which is what lets a group that has read all the way to the end come
    /// back from not knowing.
    fn on_deliver(&self, group: &Group, id: Id) -> Option<u64> {
        match group.entries_read() {
            Some(read) if !self.holed_from(id) => Some(read + 1),
            _ => self.estimate(id),
        }
    }
}

/// The only frozen form there is: the nodes as they stand, then the groups.
///
/// There is no packed form to sit beside it the way the other collections have
/// one, because a stream is a listpack per node from the first entry and never
/// changes shape.
const FORM_NODES: u8 = 1;

/// Where a group read stopped, so the next one does not start from the front.
///
/// `XREADGROUP GROUP g c COUNT 1 STREAMS key >` hands over one entry and moves
/// the group's bookmark one along. Without a mark the read after it walks the
/// node's blob from the first entry to find the one after the bookmark, and at
/// the default hundred entries a node that is fifty entries of decoding to hand
/// back one of them, which makes a consumer draining a stream quadratic in the
/// node rather than linear.
///
/// The mark is only believed when the stream has not moved any bytes since it
/// was taken and the read is asking for exactly the ID the walk that left it
/// would be asked for next. Anything else, an `XGROUP SETID` most of all, walks
/// from the front the way it always did.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cursor {
    /// The stream's mutation count when this was taken.
    epoch: u64,
    /// The ID the walk that left this mark would be asked for next, which is
    /// one past the last entry it handed over. A read starting anywhere else
    /// cannot use it, because the mark says nothing about what is behind it.
    next: Id,
    /// Which node, by its master ID, which is unique and never reused.
    master: Id,
    /// Where in that node's blob to pick up.
    byte: usize,
}

/// A log of entries in ID order.
#[derive(Debug, Clone, Default)]
pub struct Stream {
    nodes: VecDeque<Node>,
    /// Live entries, which is what `XLEN` answers.
    length: u64,
    /// The greatest ID ever appended, which does not go down when it is deleted.
    last: Id,
    /// The greatest ID ever deleted, which `XINFO` reports.
    max_deleted: Id,
    /// How many entries have ever been appended.
    ///
    /// Not the length. It only goes up, and it is what a consumer group uses to
    /// work out how far behind it is without walking anything.
    added: u64,
    /// The consumer groups, by name.
    ///
    /// A vector because a stream has a handful of groups and the name is looked
    /// up once a command, so a linear scan beats hashing and brings nothing
    /// with it. The same argument the group makes about its consumers.
    groups: Vec<(Vec<u8>, Group)>,
    /// How many times something has moved bytes inside a node.
    ///
    /// A [`Cursor`] is a byte offset into a node's blob and it is only good for
    /// as long as the bytes it counted past are where they were. An append is
    /// not one of these events, because it writes at the end and the header
    /// fields it rewrites are the same width either side of it almost always,
    /// and the almost is what this counts.
    epoch: u64,
}

/// Two streams are the same when they hold the same entries and the same
/// groups. The mutation count is not part of that. It is a number the resume
/// cursors compare themselves against, nothing outside this file can see it,
/// and a stream frozen and thawed is the same stream even though its count
/// starts again at zero.
impl PartialEq for Stream {
    fn eq(&self, other: &Stream) -> bool {
        self.nodes == other.nodes
            && self.length == other.length
            && self.last == other.last
            && self.max_deleted == other.max_deleted
            && self.added == other.added
            && self.groups == other.groups
    }
}

impl Eq for Stream {}

impl Stream {
    /// An empty stream.
    #[must_use]
    pub fn new() -> Stream {
        Stream::default()
    }

    /// Write the stream out as the bytes a tier can hold, for
    /// [`crate::keyspace::Keyspace`] to hand back to [`Stream::thaw`].
    ///
    /// The nodes go out as the listpacks they already are, one master ID and one
    /// blob each. That is the whole point of the node layout: a run of entries
    /// is already a flat sequence of bytes with no pointers in it, so freezing
    /// one is a copy and thawing it is a length check. Only the counters and the
    /// consumer groups need a form of their own.
    pub fn freeze(&self, out: &mut Vec<u8>) {
        out.push(FORM_NODES);
        frozen::put_uint(out, self.length);
        frozen::put_uint(out, self.last.ms);
        frozen::put_uint(out, self.last.seq);
        frozen::put_uint(out, self.max_deleted.ms);
        frozen::put_uint(out, self.max_deleted.seq);
        frozen::put_uint(out, self.added);

        frozen::put_uint(out, self.nodes.len() as u64);
        for node in &self.nodes {
            frozen::put_uint(out, node.master.ms);
            frozen::put_uint(out, node.master.seq);
            frozen::put_bytes(out, node.lp.as_bytes());
        }

        frozen::put_uint(out, self.groups.len() as u64);
        for (name, group) in &self.groups {
            frozen::put_bytes(out, name);
            group.freeze(out);
        }
    }

    /// Read back a stream [`Stream::freeze`] wrote.
    ///
    /// What is inside a node is checked as far as [`Listpack::from_bytes`]
    /// checks it, which is that the header, the lengths and the terminator all
    /// agree, and no further. Every walk over a node's contents already returns
    /// early on anything it does not understand rather than trusting what it
    /// finds, so a structurally sound listpack full of nonsense answers an empty
    /// range instead of panicking. That is the same trust a node written by
    /// Redis and loaded from an RDB file gets today.
    pub fn thaw(bytes: &[u8]) -> Result<Stream, Broken> {
        let mut cut = frozen::Cut::new(bytes);
        if cut.byte()? != FORM_NODES {
            return Err(Broken::Form);
        }
        let length = cut.uint()?;
        let last = Id::new(cut.uint()?, cut.uint()?);
        let max_deleted = Id::new(cut.uint()?, cut.uint()?);
        let added = cut.uint()?;
        // Deleting an entry needs an entry, and so does reading one, so a stream
        // that has lost more than it ever took or is holding more than it was
        // ever given did not come from `freeze`.
        if length > added || max_deleted > last {
            return Err(Broken::Body);
        }

        let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
        // A node is a master ID and a listpack, so it cannot be under a byte and
        // a count past what is left is short rather than a reservation to make.
        if n > cut.rest().len() {
            return Err(Broken::Short);
        }
        let mut nodes = VecDeque::with_capacity(n);
        let mut prev: Option<Id> = None;
        for _ in 0..n {
            let master = Id::new(cut.uint()?, cut.uint()?);
            // Nodes are consecutive runs in ID order, so a master that does not
            // beat the one before it would leave a lookup unable to pick the
            // node an ID belongs in.
            if prev.is_some_and(|p| p >= master) {
                return Err(Broken::Body);
            }
            prev = Some(master);
            let lp = Listpack::from_bytes(cut.bytes()?).map_err(|_| Broken::Body)?;
            nodes.push_back(Node { master, lp });
        }
        if nodes.is_empty() && length != 0 {
            return Err(Broken::Body);
        }

        let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
        if n > cut.rest().len() {
            return Err(Broken::Short);
        }
        let mut groups: Vec<(Vec<u8>, Group)> = Vec::with_capacity(n);
        for _ in 0..n {
            let name = cut.bytes()?;
            // Groups are found by a scan for the name, so a repeat would leave
            // the second one holding entries that nothing could acknowledge.
            if groups.iter().any(|(had, _)| had == name) {
                return Err(Broken::Body);
            }
            groups.push((name.to_vec(), Group::thaw(&mut cut)?));
        }

        Ok(Stream {
            nodes,
            length,
            last,
            max_deleted,
            added,
            groups,
            epoch: 0,
        })
    }

    /// How many live entries there are, which is `XLEN`.
    #[must_use]
    #[inline]
    pub fn len(&self) -> u64 {
        self.length
    }

    /// Whether there are no live entries.
    ///
    /// A stream can be empty and still exist, unlike every other collection
    /// here, because `XADD` followed by `XDEL` leaves a key whose last ID a new
    /// entry still has to beat.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The greatest ID ever appended, whether or not it is still here.
    #[must_use]
    #[inline]
    pub fn last_id(&self) -> Id {
        self.last
    }

    /// The greatest ID ever deleted, or [`Id::MIN`] if none ever was.
    #[must_use]
    #[inline]
    pub fn max_deleted_id(&self) -> Id {
        self.max_deleted
    }

    /// How many entries have ever been appended.
    #[must_use]
    #[inline]
    pub fn added(&self) -> u64 {
        self.added
    }

    /// The lowest live ID, or `None` when there are none.
    ///
    /// Walks the first node, because the first entry in it may have been
    /// deleted and its bytes are still there. That is at most a node's worth of
    /// steps and it is only asked for by `XINFO`.
    #[must_use]
    pub fn first_id(&self) -> Option<Id> {
        let mut found = None;
        self.walk(Id::MIN, Id::MAX, Some(1), &mut |id, _| {
            found = Some(id);
            false
        });
        found
    }

    /// The highest live ID, or `None` when there are none.
    ///
    /// Not the same as [`Stream::last_id`], which is the last ID handed out and
    /// stays where it is when that entry is deleted. This is the greatest ID a
    /// reader can still find, and `XSETID` is the one caller that needs the
    /// difference, because it refuses to move the bookmark below an entry that
    /// is still there.
    #[must_use]
    pub fn top_id(&self) -> Option<Id> {
        let mut found = None;
        self.rev_range(Id::MIN, Id::MAX, Some(1), |id, _| {
            found = Some(id);
            false
        });
        found
    }

    /// `XSETID`, which moves the bookmark and the two counters behind it.
    ///
    /// The bookmark decides what the next `XADD *` hands out and what a group
    /// created at `$` starts from, so moving it is how a replica is made to
    /// agree with a primary and how a stream is rebuilt from a log. The two
    /// counters are optional because Redis added them later and a caller that
    /// does not name them leaves them alone.
    ///
    /// # Errors
    ///
    /// [`Refused::NotGreater`] when `last` is below an entry that is still in
    /// the stream, since a reader holding that entry's ID would then be reading
    /// past the end of a stream that has not ended.
    pub fn set_id(
        &mut self,
        last: Id,
        added: Option<u64>,
        max_deleted: Option<Id>,
    ) -> Result<(), Refused> {
        if self.top_id().is_some_and(|top| last < top) {
            return Err(Refused::NotGreater);
        }
        self.last = last;
        if let Some(added) = added {
            self.added = added;
        }
        if let Some(id) = max_deleted {
            self.max_deleted = id;
        }
        Ok(())
    }

    /// What the ID would be if `XADD key *` ran now with clock reading `now`.
    ///
    /// The clock unless the clock has not moved, or has gone backwards, in which
    /// case it is the last ID with one added. A stream never goes back on its
    /// word about ordering just because the machine's clock did.
    #[must_use]
    pub fn auto_id(&self, now: u64) -> Option<Id> {
        if now > self.last.ms {
            Some(Id { ms: now, seq: 0 })
        } else {
            self.last.next()
        }
    }

    /// The next sequence inside `ms`, which is what `XADD key ms-*` asks for.
    #[must_use]
    pub fn auto_seq(&self, ms: u64) -> Option<Id> {
        if ms > self.last.ms {
            Some(Id { ms, seq: 0 })
        } else if ms == self.last.ms {
            self.last.next().filter(|id| id.ms == ms)
        } else {
            None
        }
    }

    /// Append an entry, which is `XADD` once the ID has been settled.
    ///
    /// # Errors
    ///
    /// [`Refused`] when the ID is zero or is not greater than [`Stream::last_id`].
    /// Nothing else can fail: an append never needs to move an entry that is
    /// already here.
    pub fn append(
        &mut self,
        id: Id,
        fields: &[(&[u8], &[u8])],
        limits: Limits,
    ) -> Result<(), Refused> {
        if id == Id::MIN {
            return Err(Refused::Zero);
        }
        if id <= self.last {
            return Err(Refused::NotGreater);
        }

        // Everything but the field names and values, which is why it is a guess
        // rather than a measurement: the point is to start a new node before the
        // current one goes past its limit, not to predict its size exactly.
        let size: usize = fields
            .iter()
            .map(|(f, v)| f.len() + v.len() + 11)
            .sum::<usize>()
            + 32;

        let fits = match self.nodes.back() {
            Some(node) => {
                let (count, deleted) = counts(&node.lp);
                node.lp.byte_len() + size < limits.max_node_bytes
                    && (count + deleted) < limits.max_node_entries as u64
            }
            None => false,
        };

        if !fits {
            self.nodes.push_back(Node {
                master: id,
                lp: master_of(fields),
            });
        }
        let node = self.nodes.back_mut().expect("a node was just made sure of");
        let same = same_fields(&node.lp, fields);
        write_entry(&mut node.lp, node.master, id, fields, same);
        // The entry itself goes on the end and moves nothing, so a group's mark
        // into this node survives an append and lands on what was just written,
        // which is the whole point. The counts in front of it are the one part
        // that can move, and this is where that gets noticed.
        if bump(&mut node.lp, 1, 0) {
            self.epoch = self.epoch.wrapping_add(1);
        }

        self.length += 1;
        self.added += 1;
        self.last = id;
        Ok(())
    }

    /// Delete the entry with that ID, answering whether there was one.
    ///
    /// The bytes stay where they are and a bit says the entry is gone, unless it
    /// was the last live entry in its node, in which case the node goes.
    pub fn delete(&mut self, id: Id) -> bool {
        if !self.remove(id) {
            return false;
        }
        self.max_deleted = self.max_deleted.max(id);
        true
    }

    /// Delete an entry, saying what to do about the groups, which is `XDELEX`.
    ///
    /// [`Refs::Acked`] is the interesting one and it asks a wider question than
    /// its name does. An entry is safe to take when no group is holding it in a
    /// pending list and no group's bookmark is still behind it, because a group
    /// that has not reached the entry yet has not had its chance at it. So a
    /// stream with one group sitting at `0-0` refuses every `ACKED` delete, and
    /// that is a real server's answer and not an over careful reading of it.
    pub fn delete_ref(&mut self, id: Id, refs: Refs) -> Fate {
        if refs == Refs::Acked && self.still_wanted(id) {
            return Fate::Held;
        }
        if !self.delete(id) {
            return Fate::Missing;
        }
        if refs == Refs::Drop {
            self.drop_refs(id);
        }
        Fate::Gone
    }

    /// Acknowledge an entry for one group and then delete it, which is `XACKDEL`.
    ///
    /// The acknowledgement is the part that decides the answer. An ID this group
    /// was not holding is [`Fate::Missing`] whether or not the entry is in the
    /// stream, and an ID it was holding is never `Missing`, so a caller reading
    /// the reply is being told about its own pending list and not about the log.
    pub fn ack_delete(&mut self, group: &[u8], id: Id, refs: Refs) -> Fate {
        let Some(g) = self.group_mut(group) else {
            return Fate::Missing;
        };
        if !g.ack(id) {
            return Fate::Missing;
        }
        if refs == Refs::Acked && self.still_wanted(id) {
            return Fate::Held;
        }
        self.delete(id);
        if refs == Refs::Drop {
            self.drop_refs(id);
        }
        Fate::Gone
    }

    /// Hand an entry back to a group without acknowledging it, which is `XNACK`.
    ///
    /// `force` makes a pending entry out of one that was not pending, and like
    /// [`Stream::claim`]'s `FORCE` it only works on an entry that is really in
    /// the stream. Answers whether anything happened, and `None` when there is
    /// no such group.
    pub fn nack(&mut self, group: &[u8], id: Id, retry: Retry, force: bool) -> Option<bool> {
        let here = self.contains(id);
        let g = self.group_mut(group)?;
        if g.release(id, retry) {
            return Some(true);
        }
        if force && here {
            g.force_release(id, retry);
            return Some(true);
        }
        Some(false)
    }

    /// Whether any group could still be handed `id`, which is what `ACKED` asks.
    fn still_wanted(&self, id: Id) -> bool {
        self.groups
            .iter()
            .any(|(_, g)| g.nack(id).is_some() || id > g.last_id())
    }

    /// Take an ID out of every group's pending list, which is what `DELREF` does.
    fn drop_refs(&mut self, id: Id) {
        for (_, g) in &mut self.groups {
            g.forget(id);
        }
    }

    /// The same without recording it.
    ///
    /// `XDEL` moves `max-deleted-entry-id` and trimming does not, which is
    /// Redis's rule and a reasonable one: that field is there so a reader can
    /// tell whether an ID it is holding was taken out from under it, and a
    /// trim that took the oldest entries says nothing about that.
    fn remove(&mut self, id: Id) -> bool {
        let Some(at) = self.node_of(id) else {
            return false;
        };
        let node = &self.nodes[at];
        let Some((offset, flags)) = find(&node.lp, node.master, id) else {
            return false;
        };
        if flags & DELETED != 0 {
            return false;
        }

        let (count, _) = counts(&node.lp);
        if count == 1 {
            self.nodes.remove(at);
        } else {
            let node = &mut self.nodes[at];
            set_int(&mut node.lp, offset, flags | DELETED);
            bump(&mut node.lp, -1, 1);
        }
        // Marking an entry dead writes into the middle of the node, so anything
        // behind it has moved and every group's mark is now a byte offset to
        // nowhere. Deletes are rare next to appends and reads, so this throws
        // all of them away rather than working out which ones survived.
        self.epoch = self.epoch.wrapping_add(1);
        self.length -= 1;
        true
    }

    /// The five facts every counter a group keeps is worked out from.
    ///
    /// Read once and carried, rather than asked for again per entry, because
    /// [`Stream::first_id`] walks a node and a delivery cannot change any of the
    /// five: handing an entry to a consumer neither adds one nor removes one.
    fn edges(&self) -> Edges {
        Edges {
            added: self.added,
            length: self.length,
            first: self.first_id(),
            last: self.last,
            max_deleted: self.max_deleted,
        }
    }

    /// How far behind a group is, or `None` when that cannot be worked out.
    ///
    /// Two ways of answering and the good one is tried first. If the group's
    /// bookmark is somewhere the distance from the start of time is exactly
    /// known, which is the last ID, past it, or before the first entry left,
    /// that distance is the answer. Otherwise the group's own counter will do,
    /// but only while nothing has been deleted at or above the bookmark, since
    /// a hole ahead of the group means it will read fewer entries than the
    /// subtraction is expecting.
    ///
    /// Both paths and their order were read off Redis 8.10.1 rather than worked
    /// out, because reasoning gives the wrong answer on the case that matters:
    /// a group sitting at `0-0` on a stream trimmed from five entries to two
    /// reports a lag of two and not five, which is the estimate winning over a
    /// subtraction that is valid and is further from the truth.
    #[must_use]
    pub fn lag(&self, group: &Group) -> Option<u64> {
        self.edges().lag(group)
    }

    /// Cut the stream down to `len` entries, dropping the oldest, which is
    /// `XTRIM key MAXLEN len`. Answers how many went.
    ///
    /// `exact` is Redis's `=` against `~`. Without it only whole nodes are
    /// dropped, so the stream is left at `len` or a little over and no node is
    /// ever rewritten. That is the form to use, and it is why `~` exists.
    ///
    /// `limit` is Redis's `LIMIT`, which stops the trim once that many entries
    /// have gone rather than once the stream is short enough. It exists because
    /// a capped stream that has fallen a long way behind would otherwise spend
    /// one command dropping millions of entries with the shard doing nothing
    /// else, and the next write will carry on where this one stopped.
    pub fn trim_maxlen(&mut self, len: u64, exact: bool, limit: Option<u64>) -> u64 {
        let mut gone = 0;
        while self.length > len && !limit.is_some_and(|cap| gone >= cap) {
            let Some(node) = self.nodes.front() else {
                break;
            };
            let (count, _) = counts(&node.lp);
            if self.length - count >= len {
                self.length -= count;
                gone += count;
                self.nodes.pop_front();
                continue;
            }
            if !exact {
                break;
            }
            let Some(id) = self.first_id() else { break };
            self.remove(id);
            gone += 1;
        }
        gone
    }

    /// Drop every entry below `id`, which is `XTRIM key MINID id`. Answers how
    /// many went.
    ///
    /// `exact` and `limit` mean what they do for [`Stream::trim_maxlen`].
    pub fn trim_minid(&mut self, id: Id, exact: bool, limit: Option<u64>) -> u64 {
        let mut gone = 0;
        while let Some(node) = self.nodes.front() {
            if limit.is_some_and(|cap| gone >= cap) {
                break;
            }
            let (count, _) = counts(&node.lp);
            if last_of(node) < id {
                self.length -= count;
                gone += count;
                self.nodes.pop_front();
                continue;
            }
            if !exact {
                break;
            }
            let Some(first) = self.first_id() else { break };
            if first >= id {
                break;
            }
            self.remove(first);
            gone += 1;
        }
        gone
    }

    /// Every live entry from `start` to `end`, both ends included, oldest first.
    ///
    /// `count` stops the walk early, which is `XRANGE ... COUNT n`. The callback
    /// answers whether to carry on, so a caller filling a fixed reply can stop
    /// without knowing how many it wanted up front. Answers how many entries the
    /// callback saw.
    pub fn range<F>(&self, start: Id, end: Id, count: Option<usize>, mut f: F) -> usize
    where
        F: FnMut(Id, Fields<'_>) -> bool,
    {
        self.walk(start, end, count, &mut f)
    }

    /// The same, newest first, which is `XREVRANGE`.
    ///
    /// `start` and `end` are still the low and the high end of the range, so a
    /// caller does not have to swap them and the command layer does, once, where
    /// the argument order is Redis's problem.
    pub fn rev_range<'s, F>(&'s self, start: Id, end: Id, count: Option<usize>, mut f: F) -> usize
    where
        F: FnMut(Id, Fields<'_>) -> bool,
    {
        let mut seen = 0;
        // A node is a hundred entries, so buffering one node's worth of marks
        // and handing them back in reverse is cheaper and a great deal clearer
        // than walking the blob backwards over the entry lengths. The buffer is
        // reused across nodes, so the whole reverse scan allocates once.
        let mut buf: Vec<(Id, Fields<'s>)> = Vec::new();
        // Straight to the node the high end falls in, the same binary search the
        // forward walk starts with. Walking back from the newest node instead
        // would skip over every node above `end` one at a time, which for a
        // window in the middle of a million entries is five thousand nodes
        // touched to read a hundred.
        let last = self.node_from(end);
        for node in self.nodes.iter().take(last + 1).rev() {
            // Only reachable when every node is above `end`, since the search
            // clamps to the front rather than saying there is nothing.
            if node.master > end {
                continue;
            }
            if last_of(node) < start {
                break;
            }
            buf.clear();
            each(&node.lp, node.master, None, &mut |id, _, fields| {
                if id >= start && id <= end {
                    buf.push((id, fields));
                }
                id <= end
            });
            for (id, fields) in buf.drain(..).rev() {
                if count.is_some_and(|want| seen >= want) {
                    return seen;
                }
                seen += 1;
                if !f(id, fields) {
                    return seen;
                }
            }
        }
        seen
    }

    /// Whether an entry with this ID is there and live.
    ///
    /// What `XCLAIM` asks before it hands a pending entry to somebody, since an
    /// entry that has been deleted or trimmed away is work nobody can do.
    #[must_use]
    pub fn contains(&self, id: Id) -> bool {
        let Some(at) = self.node_of(id) else {
            return false;
        };
        let node = &self.nodes[at];
        find(&node.lp, node.master, id).is_some_and(|(_, flags)| flags & DELETED == 0)
    }

    /// Make a consumer group, and say whether it was not already there.
    ///
    /// `XGROUP CREATE`. `last` is where it starts reading after, which is
    /// [`Stream::last_id`] for `$` and [`Id::MIN`] for `0`.
    pub fn create_group(&mut self, name: &[u8], last: Id, read: Option<u64>) -> bool {
        if self.group(name).is_some() {
            return false;
        }
        self.groups.push((name.to_vec(), Group::new(last, read)));
        true
    }

    /// Take a group out, and say whether it was there.
    pub fn destroy_group(&mut self, name: &[u8]) -> bool {
        let Some(at) = self.groups.iter().position(|(n, _)| n == name) else {
            return false;
        };
        self.groups.remove(at);
        true
    }

    /// One group by name.
    #[must_use]
    pub fn group(&self, name: &[u8]) -> Option<&Group> {
        self.groups
            .iter()
            .find(|(n, _)| n.as_slice() == name)
            .map(|(_, g)| g)
    }

    /// One group by name, to change.
    pub fn group_mut(&mut self, name: &[u8]) -> Option<&mut Group> {
        self.groups
            .iter_mut()
            .find(|(n, _)| n.as_slice() == name)
            .map(|(_, g)| g)
    }

    /// Every group, with its name.
    pub fn groups(&self) -> impl Iterator<Item = (&[u8], &Group)> + '_ {
        self.groups.iter().map(|(n, g)| (n.as_slice(), g))
    }

    /// Hand new entries to a consumer, which is `XREADGROUP ... >`.
    ///
    /// Every entry after the group's bookmark, up to `count`, delivered to
    /// `consumer` and written into the pending list as it goes. The consumer is
    /// created if it is not there, because a consumer exists by turning up.
    ///
    /// `noack` is Redis's `NOACK`, which hands the entries over without writing
    /// them into the pending list at all. The group still counts them as read,
    /// so the lag is the same either way, and the consumer is on its own if it
    /// dies holding one.
    ///
    /// Answers how many entries the callback saw, or `None` when there is no
    /// such group.
    pub fn read_group<F>(
        &mut self,
        group: &[u8],
        consumer: &[u8],
        count: Option<usize>,
        noack: bool,
        now: u64,
        mut f: F,
    ) -> Option<usize>
    where
        F: FnMut(Id, Fields<'_>) -> bool,
    {
        // Before the split borrow, because it walks a node and the walk below
        // holds the nodes. Nothing a delivery does can change any of it.
        let edges = self.edges();
        // Field by field, so that walking the nodes and writing the pending list
        // are two borrows the compiler can see are disjoint.
        let Stream {
            nodes,
            groups,
            epoch,
            ..
        } = self;
        let epoch = *epoch;
        let (_, g) = groups.iter_mut().find(|(n, _)| n.as_slice() == group)?;
        let slot = g.consumer_or_create(consumer, now);
        let Some(from) = g.last_id().next() else {
            // The bookmark is at the very last ID there is, so there is nothing
            // after it and never will be.
            g.touch(slot, now, false);
            return Some(0);
        };
        // Where the last read of this group stopped, when it is still good for
        // this one. A consumer draining a stream one entry at a time asks for
        // the entry right after the one it just got, and this is what saves it
        // decoding the node from the front to find it.
        let resume = g.resume(epoch, from);
        let (seen, mark) = walk_nodes(nodes, from, Id::MAX, count, resume, &mut |id, fields| {
            // Worked out before the bookmark moves, because the rule asks where
            // the group was when the entry was handed over.
            let read = edges.on_deliver(g, id);
            if noack {
                g.skip(id);
            } else {
                g.deliver(slot, id, now);
            }
            g.set_read(read);
            f(id, fields)
        });
        let next = g.last_id().next();
        g.set_resume(match (mark, next) {
            (Some((master, byte)), Some(next)) => Some(Cursor {
                epoch,
                next,
                master,
                byte,
            }),
            _ => None,
        });
        g.touch(slot, now, seen > 0);
        Some(seen)
    }

    /// Re-read what a consumer is already holding, which is `XREADGROUP` with an
    /// ID rather than `>`.
    ///
    /// Every pending entry of that consumer after `after`, oldest first. Each
    /// one counts as handed out again, so its delivery time is reset and its
    /// count goes up. That is Redis's behaviour, checked rather than assumed,
    /// and it is the right one: the count is how many times a consumer has been
    /// told to do this work, and a consumer re-reading its backlog after a
    /// restart has been told again.
    ///
    /// An entry that has since been deleted or trimmed is still in the pending
    /// list and is handed to the callback with no fields, which is the null
    /// Redis puts in the reply. Clearing those out is [`Stream::claim`]'s job
    /// and not this one.
    pub fn read_group_pending<F>(
        &mut self,
        group: &[u8],
        consumer: &[u8],
        after: Id,
        count: Option<usize>,
        now: u64,
        mut f: F,
    ) -> Option<usize>
    where
        F: FnMut(Id, Option<Fields<'_>>) -> bool,
    {
        // Which IDs, decided before anything is touched, so that the redelivery
        // and the walk are two passes over a small list rather than one pass
        // holding the group and the nodes at the same time.
        //
        // The consumer is created rather than looked up, because a history read
        // by a name nobody has used is an empty list and not a missing group. A
        // worker that restarts under a new name and asks for its own backlog
        // first is exactly that case, and Redis answers it with an empty list
        // and the consumer left behind.
        let g = self.group_mut(group)?;
        let slot = g.consumer_or_create(consumer, now);
        let ids: Vec<Id> = g
            .consumer(slot)
            .expect("the slot that was just made")
            .pending()
            .filter(|&id| id > after)
            .take(count.unwrap_or(usize::MAX))
            .collect();
        for &id in &ids {
            g.redeliver(id, now);
        }
        g.touch(slot, now, !ids.is_empty());

        let mut seen = 0;
        for &id in &ids {
            seen += 1;
            let mut go = true;
            let mut found = false;
            self.walk(id, id, Some(1), &mut |got, fields| {
                found = true;
                go = f(got, Some(fields));
                false
            });
            if !found {
                go = f(id, None);
            }
            if !go {
                break;
            }
        }
        Some(seen)
    }

    /// Move pending entries to a consumer, which is `XCLAIM`.
    ///
    /// Only entries idle at least `min_idle` move. `time` is what the delivery
    /// time becomes, `retry` replaces the delivery count when it is given, and
    /// `bump` says whether to add one to it, which `JUSTID` turns off. `force`
    /// makes a pending entry for an ID that is in the stream but was not
    /// pending.
    ///
    /// An ID that is pending but no longer in the stream is dropped from the
    /// pending list rather than claimed, and reported through `gone`, which is
    /// what Redis does and what stops a deleted entry being handed round
    /// forever. Answers the IDs that moved.
    #[allow(clippy::too_many_arguments)]
    pub fn claim(
        &mut self,
        group: &[u8],
        consumer: &[u8],
        ids: &[Id],
        min_idle: u64,
        time: u64,
        retry: Option<u64>,
        bump: bool,
        force: bool,
        now: u64,
        gone: &mut Vec<Id>,
    ) -> Option<Vec<Id>> {
        // Before the loop, so that a claim which takes nothing still leaves the
        // consumer behind. Redis creates it either way, and an `XAUTOCLAIM`
        // against an empty pending list is the ordinary way that happens: the
        // consumer turns up in `XINFO CONSUMERS` straight after, holding
        // nothing.
        self.group_mut(group)?.consumer_or_create(consumer, now);
        let mut took = Vec::new();
        for &id in ids {
            let here = self.contains(id);
            let g = self.group_mut(group)?;
            let slot = g.consumer_or_create(consumer, now);
            match g.nack(id) {
                Some(nack) => {
                    if !here {
                        g.forget(id);
                        gone.push(id);
                        continue;
                    }
                    if nack.idle(now) < min_idle {
                        continue;
                    }
                    if g.claim(id, slot, time, retry, bump) {
                        took.push(id);
                    }
                }
                None => {
                    // FORCE makes one out of nothing, but only for an entry that
                    // is really there. Redis ignores the rest in silence.
                    if force && here && g.force(id, slot, time, retry.unwrap_or(1)) {
                        took.push(id);
                    }
                }
            }
        }
        // Active only when something moved, which is the same rule a read
        // follows. A claim that found nothing idle enough leaves the consumer
        // reading as never active.
        if !took.is_empty() {
            let g = self.group_mut(group).expect("the group found a moment ago");
            let slot = g.consumer_or_create(consumer, now);
            g.touch(slot, now, true);
        }
        Some(took)
    }

    /// Sweep the pending list for stale entries and claim them, which is
    /// `XAUTOCLAIM`.
    ///
    /// Starts at `start` and takes up to `count` entries that have been idle at
    /// least `min_idle`. Answers where a following call should carry on from,
    /// which is `None` at the end of the list, along with what was claimed and
    /// what was dropped for no longer being in the stream.
    #[allow(clippy::too_many_arguments)]
    pub fn autoclaim(
        &mut self,
        group: &[u8],
        consumer: &[u8],
        start: Id,
        min_idle: u64,
        count: usize,
        bump: bool,
        now: u64,
        gone: &mut Vec<Id>,
    ) -> Option<(Option<Id>, Vec<Id>)> {
        let mut ids = Vec::new();
        let cursor = self
            .group(group)?
            .claimable(start, min_idle, now, count, &mut ids);
        let took = self.claim(
            group, consumer, &ids, min_idle, now, None, bump, false, now, gone,
        )?;
        Some((cursor, took))
    }

    /// How many bytes the entries and the groups take, not counting this struct.
    ///
    /// The name is the one every other body in this crate uses, because the
    /// keyspace asks all of them the same question through one trait and a
    /// stream that answered it under a different name would need its own arm.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let nodes: usize = self
            .nodes
            .iter()
            .map(|node| node.lp.byte_len() + std::mem::size_of::<Node>())
            .sum();
        let groups: usize = self
            .groups
            .iter()
            .map(|(name, g)| {
                name.capacity() + std::mem::size_of::<(Vec<u8>, Group)>() + g.memory_bytes()
            })
            .sum();
        nodes + groups
    }

    /// How many nodes there are, which only a test and `XINFO STREAM FULL` care
    /// about.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The shared walk behind [`Stream::range`] and [`Stream::first_id`].
    fn walk<F>(&self, start: Id, end: Id, count: Option<usize>, f: &mut F) -> usize
    where
        F: FnMut(Id, Fields<'_>) -> bool,
    {
        walk_nodes(&self.nodes, start, end, count, None, f).0
    }

    /// The first node that can hold an entry at or after `id`.
    fn node_from(&self, id: Id) -> usize {
        node_from(&self.nodes, id)
    }

    /// The node that would hold `id`, or `None` when no node covers it.
    fn node_of(&self, id: Id) -> Option<usize> {
        let at = self.node_from(id);
        let node = self.nodes.get(at)?;
        (node.master <= id && id <= last_of(node)).then_some(at)
    }
}

/// The first node that can hold an entry at or after `id`.
///
/// The binary search the module docs are about. `partition_point` answers how
/// many nodes start strictly before `id`, and the one before that is the one
/// `id` would be in, since a node holds everything from its master ID up to the
/// next node's.
///
/// Free rather than a method so that a group read can hold the nodes and the
/// groups at the same time, which it has to because it walks the one to write
/// into the other.
fn node_from(nodes: &VecDeque<Node>, id: Id) -> usize {
    let after = nodes.partition_point(|node| node.master <= id);
    after.saturating_sub(1)
}

/// Every live entry from `start` to `end`, both included, oldest first.
///
/// Free for the same reason [`node_from`] is.
/// `resume` is a node's master ID and a byte offset inside it from an earlier
/// walk, and the answer carries the same pair back for where this one stopped.
/// Only a group read has one, and it is the caller's job to have checked that
/// nothing has moved the bytes since.
fn walk_nodes<F>(
    nodes: &VecDeque<Node>,
    start: Id,
    end: Id,
    count: Option<usize>,
    resume: Option<(Id, usize)>,
    f: &mut F,
) -> (usize, Option<(Id, usize)>)
where
    F: FnMut(Id, Fields<'_>) -> bool,
{
    let mut seen = 0;
    let mut stop = false;
    let first = node_from(nodes, start);
    // Only when the mark is about the node this walk is starting in. Node master
    // IDs are unique and never reused, so a node that has since been trimmed
    // away cannot be mistaken for the one that took its place.
    let mut from =
        resume.and_then(|(master, byte)| (nodes.get(first)?.master == master).then_some(byte));
    let mut mark = None;
    for node in nodes.iter().skip(first) {
        if node.master > end {
            break;
        }
        let at = each(&node.lp, node.master, from.take(), &mut |id, _, fields| {
            if id > end {
                stop = true;
                return false;
            }
            if id < start {
                return true;
            }
            if count.is_some_and(|want| seen >= want) {
                stop = true;
                return false;
            }
            seen += 1;
            if !f(id, fields) {
                stop = true;
                return false;
            }
            true
        });
        mark = Some((node.master, at));
        if stop {
            break;
        }
    }
    (seen, mark)
}

/// The field names and values of one entry.
///
/// Two walks rather than one because an entry that shares the node's field names
/// reads them from the master entry and its values from itself, and the whole
/// point of that layout is that the names are not copied per entry. Neither walk
/// allocates and neither is a copy.
#[derive(Debug, Clone)]
pub struct Fields<'a> {
    /// Where the names come from, when they are not interleaved with the values.
    names: Option<listpack::Iter<'a>>,
    body: listpack::Iter<'a>,
    left: usize,
}

impl<'a> Iterator for Fields<'a> {
    type Item = (Entry<'a>, Entry<'a>);

    fn next(&mut self) -> Option<(Entry<'a>, Entry<'a>)> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let name = match &mut self.names {
            Some(names) => names.next()?,
            None => self.body.next()?,
        };
        Some((name, self.body.next()?))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Fields<'_> {}

impl Fields<'_> {
    /// Whether there are no fields left.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.left == 0
    }
}

/// A fresh node whose master fields are this entry's.
fn master_of(fields: &[(&[u8], &[u8])]) -> Listpack {
    let mut lp = Listpack::new();
    push_int(&mut lp, 0);
    push_int(&mut lp, 0);
    push_int(&mut lp, fields.len() as i64);
    for (name, _) in fields {
        lp.push(name);
    }
    push_int(&mut lp, 0);
    lp
}

/// Whether these fields are exactly the node's master fields, in order.
fn same_fields(lp: &Listpack, fields: &[(&[u8], &[u8])]) -> bool {
    let mut it = lp.iter();
    let (_, _, want) = match (it.next(), it.next(), it.next()) {
        (Some(_), Some(_), Some(Entry::Int(n))) => ((), (), n),
        _ => return false,
    };
    if want != fields.len() as i64 {
        return false;
    }
    fields.iter().all(|(name, _)| match it.next() {
        Some(Entry::Str(s)) => s == *name,
        Some(Entry::Int(n)) => {
            let mut buf = [0u8; DIGITS_MAX];
            i64_digits(&mut buf, n) == *name
        }
        None => false,
    })
}

/// The master entry's live and deleted counts.
fn counts(lp: &Listpack) -> (u64, u64) {
    let mut it = lp.iter();
    let count = int_or_zero(it.next());
    let deleted = int_or_zero(it.next());
    (count.max(0) as u64, deleted.max(0) as u64)
}

/// Add to the master entry's two counts, answering whether that moved any bytes.
///
/// Both counts sit in front of every entry in the node, so a write that encodes
/// to a different width than what was there shifts the whole rest of the blob
/// along and every [`Cursor`] into it is a byte offset to nowhere. It usually
/// does not: a node holds a hundred entries by default and a listpack keeps
/// anything under a hundred and twenty eight in one byte, so an append writes
/// the same width it read almost every time. Almost, because the node limits are
/// the caller's to set.
fn bump(lp: &mut Listpack, live: i64, dead: i64) -> bool {
    let was = lp.byte_len();
    let (count, deleted) = counts(lp);
    let mut buf = [0u8; DIGITS_MAX];
    let at = count as i64 + live;
    lp.replace(0, u64_digits(&mut buf, at.max(0) as u64));
    let at = deleted as i64 + dead;
    lp.replace(1, u64_digits(&mut buf, at.max(0) as u64));
    lp.byte_len() != was
}

/// An entry as an integer, or zero for anything else.
fn int_or_zero(entry: Option<Entry<'_>>) -> i64 {
    match entry {
        Some(Entry::Int(n)) => n,
        _ => 0,
    }
}

/// Append an integer, which the listpack encodes as one because it parses as one.
fn push_int(lp: &mut Listpack, n: i64) {
    let mut buf = [0u8; DIGITS_MAX];
    lp.push(i64_digits(&mut buf, n));
}

/// Overwrite the element at `index` with an integer.
fn set_int(lp: &mut Listpack, index: usize, n: i64) {
    let mut buf = [0u8; DIGITS_MAX];
    lp.replace(index, i64_digits(&mut buf, n));
}

/// Write one entry onto the end of a node.
fn write_entry(lp: &mut Listpack, master: Id, id: Id, fields: &[(&[u8], &[u8])], same: bool) {
    let flags = if same { LIVE | SAME_FIELDS } else { LIVE };
    push_int(lp, flags);
    // Both halves as a plain difference from the master, wrapping, which is what
    // Redis writes. The sequence often goes down when the millisecond goes up,
    // so that difference is usually negative and it does not matter: adding it
    // back is the exact inverse whichever way it went.
    push_int(lp, id.ms.wrapping_sub(master.ms) as i64);
    push_int(lp, id.seq.wrapping_sub(master.seq) as i64);
    if same {
        for (_, value) in fields {
            lp.push(value);
        }
        push_int(lp, fields.len() as i64 + 3);
    } else {
        push_int(lp, fields.len() as i64);
        for (name, value) in fields {
            lp.push(name);
            lp.push(value);
        }
        push_int(lp, fields.len() as i64 * 2 + 4);
    }
}

/// The greatest ID in a node, live or not.
fn last_of(node: &Node) -> Id {
    let mut last = node.master;
    each(&node.lp, node.master, None, &mut |id, _, _| {
        last = id;
        true
    });
    last
}

/// Where the entry with that ID starts, and its flags, or `None`.
///
/// The offset is a listpack element index rather than a byte offset, because
/// that is what `replace` takes.
fn find(lp: &Listpack, master: Id, id: Id) -> Option<(usize, i64)> {
    let mut at = None;
    walk_node(lp, master, None, &mut |mark, flags, _| {
        if mark.id == id {
            // Always a `Some`, because this walk starts at the front, and the
            // `if let` rather than an unwrap keeps that a fact about this call
            // rather than something the mark has to promise everybody.
            if let Some(index) = mark.index {
                at = Some((index, flags));
            }
            return false;
        }
        mark.id < id
    });
    at
}

/// Every entry in a node, live ones only, oldest first.
/// `from` and the answer are the resume mark [`walk_node`] takes and gives back.
fn each<'a, F>(lp: &'a Listpack, master: Id, from: Option<usize>, f: &mut F) -> usize
where
    F: FnMut(Id, usize, Fields<'a>) -> bool,
{
    walk_node(lp, master, from, &mut |mark, flags, fields| {
        if flags & DELETED != 0 {
            return true;
        }
        f(mark.id, mark.byte, fields)
    })
}

/// Every entry in a node, deleted ones included, with its element index.
///
/// One forward walk of the whole blob. Nothing here reaches into the middle by
/// index, because a listpack index is a walk from the front and doing that per
/// entry would turn a node scan into a quadratic one.
fn walk_node<'a, F>(lp: &'a Listpack, master: Id, from: Option<usize>, f: &mut F) -> usize
where
    F: FnMut(Mark, i64, Fields<'a>) -> bool,
{
    let mut it = lp.iter();
    let (Some(_), Some(_), Some(Entry::Int(masters))) = (it.next(), it.next(), it.next()) else {
        return it.offset();
    };
    let masters = masters.max(0) as usize;
    // The master field names, kept as a mark to hand to the entries that share
    // them, and then stepped over along with the zero that ends the master entry.
    let names = it.clone();
    let mut index = MASTER_FIELDS;
    for _ in 0..=masters {
        if it.next().is_none() {
            return it.offset();
        }
        index += 1;
    }
    // A resume skips straight to where a previous walk stopped. The header above
    // still has to be read whichever way this is called, because the field names
    // a shared entry hands back live in it, but that is a handful of elements
    // rather than every entry in the node.
    let counting = from.is_none();
    if let Some(byte) = from.filter(|byte| *byte > it.offset()) {
        it = lp.iter_at(byte);
    }

    loop {
        let at = it.offset();
        let element = index;
        let (Some(Entry::Int(flags)), Some(Entry::Int(ms)), Some(Entry::Int(seq))) =
            (it.next(), it.next(), it.next())
        else {
            return at;
        };
        index += 3;
        let id = Id {
            ms: master.ms.wrapping_add(ms as u64),
            seq: master.seq.wrapping_add(seq as u64),
        };

        let same = flags & SAME_FIELDS != 0;
        let (fields, skip) = if same {
            let fields = Fields {
                names: Some(names.clone()),
                body: it.clone(),
                left: masters,
            };
            (fields, masters + 1)
        } else {
            let Some(Entry::Int(n)) = it.next() else {
                return at;
            };
            index += 1;
            let fields = Fields {
                names: None,
                body: it.clone(),
                left: n.max(0) as usize,
            };
            (fields, n.max(0) as usize * 2 + 1)
        };

        let mark = Mark {
            id,
            index: counting.then_some(element),
            byte: at,
        };
        if !f(mark, flags, fields) {
            return at;
        }
        for _ in 0..skip {
            if it.next().is_none() {
                return it.offset();
            }
            index += 1;
        }
    }
}

/// Where one entry sits inside its node.
#[derive(Debug, Clone, Copy)]
struct Mark {
    /// The entry's ID, worked out from the node's master and the two
    /// differences the entry stores.
    id: Id,
    /// Which listpack element the entry's flags are, for the writes that reach
    /// back in by index. Only filled in on a walk that started at the front,
    /// because counting elements from a resume point would give a number
    /// [`Listpack::replace`] cannot use.
    index: Option<usize>,
    /// Where in the blob the entry's flags element starts, which is what
    /// [`Listpack::iter_at`] takes to come back here.
    byte: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs<'a>(of: &'a [(&'a str, &'a str)]) -> Vec<(&'a [u8], &'a [u8])> {
        of.iter()
            .map(|(f, v)| (f.as_bytes(), v.as_bytes()))
            .collect()
    }

    /// One entry, with its fields owned, which is what the tests compare.
    type Flat = (Id, Vec<(Vec<u8>, Vec<u8>)>);

    /// Everything in the stream, oldest first.
    fn dump(s: &Stream) -> Vec<Flat> {
        let mut out = Vec::new();
        s.range(Id::MIN, Id::MAX, None, |id, fields| {
            out.push((id, fields.map(|(f, v)| (f.to_vec(), v.to_vec())).collect()));
            true
        });
        out
    }

    fn add(s: &mut Stream, ms: u64, seq: u64, fields: &[(&str, &str)]) {
        s.append(Id::new(ms, seq), &pairs(fields), Limits::default())
            .expect("an append");
    }

    #[test]
    fn an_entry_comes_back_as_it_went_in() {
        let mut s = Stream::new();
        add(&mut s, 5, 0, &[("sensor", "1"), ("reading", "23.4")]);
        let got = dump(&s);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, Id::new(5, 0));
        assert_eq!(
            got[0].1,
            vec![
                (b"sensor".to_vec(), b"1".to_vec()),
                (b"reading".to_vec(), b"23.4".to_vec())
            ]
        );
    }

    #[test]
    fn entries_come_back_in_order() {
        let mut s = Stream::new();
        for ms in 1..200u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let got = dump(&s);
        assert_eq!(got.len(), 199);
        for (at, (id, _)) in got.iter().enumerate() {
            assert_eq!(*id, Id::new(at as u64 + 1, 0));
        }
        assert_eq!(s.len(), 199);
        assert_eq!(s.added(), 199);
        assert_eq!(s.last_id(), Id::new(199, 0));
    }

    /// The whole reason a node holds a hundred entries.
    #[test]
    fn a_long_stream_is_many_nodes() {
        let mut s = Stream::new();
        for ms in 1..=1000u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.nodes(), 10, "a hundred entries a node");
        assert_eq!(dump(&s).len(), 1000);
    }

    /// Sharing the field names is worth most of the entry on a real stream.
    #[test]
    fn the_field_names_are_not_stored_twice() {
        let mut shared = Stream::new();
        let mut apart = Stream::new();
        for ms in 1..=100u64 {
            add(
                &mut shared,
                ms,
                0,
                &[("temperature_celsius", "21"), ("relative_humidity", "44")],
            );
            let a = format!("temperature_celsius{ms}");
            let b = format!("relative_humidity{ms}");
            apart
                .append(
                    Id::new(ms, 0),
                    &[(a.as_bytes(), b"21"), (b.as_bytes(), b"44")],
                    Limits::default(),
                )
                .expect("an append");
        }
        assert!(
            shared.memory_bytes() * 3 < apart.memory_bytes(),
            "{} against {}",
            shared.memory_bytes(),
            apart.memory_bytes()
        );
    }

    /// What a real stream entry costs, so that a change that quietly doubles it
    /// has somewhere to fail.
    ///
    /// Ten thousand `sensor` and `reading` entries a millisecond apart, which is
    /// the shape the benchmark uses and the shape a stream almost always has.
    /// That measures 23.9 bytes an entry today, against 48.7 for the same
    /// entries with field names that cannot be shared. Thirty two is a bar with
    /// room in it rather than a target, because the point is to catch a
    /// regression and not to freeze the encoder.
    #[test]
    fn an_entry_costs_about_two_dozen_bytes() {
        let mut s = Stream::new();
        for ms in 1..=10_000u64 {
            let reading = format!("{:.3}", ms as f64 / 7.0);
            s.append(
                Id::new(ms, 0),
                &[(b"sensor", b"a4"), (b"reading", reading.as_bytes())],
                Limits::default(),
            )
            .expect("an append");
        }
        let each = s.memory_bytes() as f64 / 10_000.0;
        assert!(each < 32.0, "{each:.2} bytes an entry");
    }

    #[test]
    fn an_entry_with_its_own_fields_still_reads_back() {
        let mut s = Stream::new();
        add(&mut s, 1, 0, &[("a", "1"), ("b", "2")]);
        add(&mut s, 2, 0, &[("c", "3")]);
        add(&mut s, 3, 0, &[("a", "4"), ("b", "5")]);
        let got = dump(&s);
        assert_eq!(got[1].1, vec![(b"c".to_vec(), b"3".to_vec())]);
        assert_eq!(
            got[2].1,
            vec![
                (b"a".to_vec(), b"4".to_vec()),
                (b"b".to_vec(), b"5".to_vec())
            ]
        );
    }

    /// The same names in a different order is not the same shape.
    #[test]
    fn the_order_of_the_names_matters() {
        let mut s = Stream::new();
        add(&mut s, 1, 0, &[("a", "1"), ("b", "2")]);
        add(&mut s, 2, 0, &[("b", "3"), ("a", "4")]);
        let got = dump(&s);
        assert_eq!(
            got[1].1,
            vec![
                (b"b".to_vec(), b"3".to_vec()),
                (b"a".to_vec(), b"4".to_vec())
            ]
        );
    }

    #[test]
    fn an_id_must_beat_the_last_one() {
        let mut s = Stream::new();
        add(&mut s, 5, 5, &[("n", "x")]);
        let f = pairs(&[("n", "x")]);
        for id in [Id::new(5, 5), Id::new(5, 4), Id::new(1, 0)] {
            assert_eq!(
                s.append(id, &f, Limits::default()),
                Err(Refused::NotGreater),
                "{id:?}"
            );
        }
        assert_eq!(s.append(Id::new(5, 6), &f, Limits::default()), Ok(()));
    }

    #[test]
    fn nothing_can_be_added_at_zero() {
        let mut s = Stream::new();
        assert_eq!(
            s.append(Id::MIN, &pairs(&[("n", "x")]), Limits::default()),
            Err(Refused::Zero)
        );
    }

    #[test]
    fn a_range_takes_both_ends() {
        let mut s = Stream::new();
        for ms in 1..=10u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let mut seen = Vec::new();
        s.range(Id::new(3, 0), Id::new(6, 0), None, |id, _| {
            seen.push(id.ms);
            true
        });
        assert_eq!(seen, vec![3, 4, 5, 6]);
    }

    /// A range whose ends fall between entries, and one that misses entirely.
    #[test]
    fn a_range_that_lands_between_entries() {
        let mut s = Stream::new();
        for ms in [10u64, 20, 30] {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let mut seen = Vec::new();
        s.range(Id::new(11, 0), Id::new(29, 0), None, |id, _| {
            seen.push(id.ms);
            true
        });
        assert_eq!(seen, vec![20]);

        let mut none = 0;
        s.range(Id::new(31, 0), Id::MAX, None, |_, _| {
            none += 1;
            true
        });
        assert_eq!(none, 0);
    }

    #[test]
    fn a_count_stops_the_walk() {
        let mut s = Stream::new();
        for ms in 1..=500u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let mut seen = 0;
        let answered = s.range(Id::MIN, Id::MAX, Some(7), |_, _| {
            seen += 1;
            true
        });
        assert_eq!((seen, answered), (7, 7));
    }

    #[test]
    fn the_callback_can_stop_the_walk() {
        let mut s = Stream::new();
        for ms in 1..=500u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let mut seen = 0;
        s.range(Id::MIN, Id::MAX, None, |_, _| {
            seen += 1;
            seen < 3
        });
        assert_eq!(seen, 3);
    }

    #[test]
    fn a_reverse_range_is_the_forward_one_backwards() {
        let mut s = Stream::new();
        for ms in 1..=350u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let mut forward = Vec::new();
        s.range(Id::new(50, 0), Id::new(300, 0), None, |id, _| {
            forward.push(id);
            true
        });
        let mut back = Vec::new();
        s.rev_range(Id::new(50, 0), Id::new(300, 0), None, |id, _| {
            back.push(id);
            true
        });
        back.reverse();
        assert_eq!(forward, back);
        assert_eq!(forward.len(), 251);
    }

    #[test]
    fn a_reverse_range_takes_a_count_from_the_new_end() {
        let mut s = Stream::new();
        for ms in 1..=350u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        let mut seen = Vec::new();
        s.rev_range(Id::MIN, Id::MAX, Some(3), |id, _| {
            seen.push(id.ms);
            true
        });
        assert_eq!(seen, vec![350, 349, 348]);
    }

    #[test]
    fn deleting_leaves_the_rest_readable() {
        let mut s = Stream::new();
        for ms in 1..=10u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert!(s.delete(Id::new(4, 0)));
        assert!(!s.delete(Id::new(4, 0)), "twice is not twice");
        assert!(!s.delete(Id::new(99, 0)));
        assert_eq!(s.len(), 9);
        assert_eq!(s.max_deleted_id(), Id::new(4, 0));
        let seen: Vec<u64> = dump(&s).iter().map(|(id, _)| id.ms).collect();
        assert_eq!(seen, vec![1, 2, 3, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn deleting_the_first_entry_moves_the_first_id() {
        let mut s = Stream::new();
        for ms in 1..=5u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.first_id(), Some(Id::new(1, 0)));
        s.delete(Id::new(1, 0));
        assert_eq!(s.first_id(), Some(Id::new(2, 0)));
    }

    /// The last live entry going takes the node with it.
    #[test]
    fn emptying_a_node_drops_it() {
        let mut s = Stream::new();
        for ms in 1..=250u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.nodes(), 3);
        for ms in 1..=100u64 {
            assert!(s.delete(Id::new(ms, 0)), "{ms}");
        }
        assert_eq!(s.nodes(), 2);
        assert_eq!(s.len(), 150);
        assert_eq!(dump(&s).len(), 150);
    }

    /// The stream can be empty and still know what came before.
    #[test]
    fn an_emptied_stream_still_remembers_its_last_id() {
        let mut s = Stream::new();
        add(&mut s, 7, 0, &[("n", "x")]);
        s.delete(Id::new(7, 0));
        assert!(s.is_empty());
        assert_eq!(s.last_id(), Id::new(7, 0));
        assert_eq!(s.added(), 1);
        assert_eq!(s.first_id(), None);
        assert_eq!(
            s.append(Id::new(7, 0), &pairs(&[("n", "x")]), Limits::default()),
            Err(Refused::NotGreater),
            "a deleted id is still used up"
        );
    }

    #[test]
    fn trimming_to_a_length_takes_the_oldest() {
        let mut s = Stream::new();
        for ms in 1..=1000u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.trim_maxlen(150, true, None), 850);
        assert_eq!(s.len(), 150);
        assert_eq!(s.first_id(), Some(Id::new(851, 0)));
        assert_eq!(s.last_id(), Id::new(1000, 0));
    }

    /// The point of `~`: whole nodes only, so nothing is rewritten.
    #[test]
    fn an_approximate_trim_stops_at_a_node() {
        let mut s = Stream::new();
        for ms in 1..=1000u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.trim_maxlen(150, false, None), 800);
        assert_eq!(s.len(), 200, "left at the node boundary above 150");
        assert_eq!(s.nodes(), 2);
    }

    #[test]
    fn trimming_to_a_length_that_is_already_met_does_nothing() {
        let mut s = Stream::new();
        for ms in 1..=10u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.trim_maxlen(50, true, None), 0);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn trimming_to_zero_empties_it() {
        let mut s = Stream::new();
        for ms in 1..=250u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.trim_maxlen(0, true, None), 250);
        assert!(s.is_empty());
        assert_eq!(s.nodes(), 0);
        assert_eq!(s.last_id(), Id::new(250, 0));
    }

    #[test]
    fn trimming_below_an_id_takes_everything_under_it() {
        let mut s = Stream::new();
        for ms in 1..=1000u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.trim_minid(Id::new(400, 0), true, None), 399);
        assert_eq!(s.first_id(), Some(Id::new(400, 0)));
        assert_eq!(s.len(), 601);
    }

    #[test]
    fn an_approximate_minid_trim_stops_at_a_node() {
        let mut s = Stream::new();
        for ms in 1..=1000u64 {
            add(&mut s, ms, 0, &[("n", "x")]);
        }
        assert_eq!(s.trim_minid(Id::new(450, 0), false, None), 400);
        assert_eq!(s.first_id(), Some(Id::new(401, 0)));
    }

    #[test]
    fn several_entries_share_a_millisecond() {
        let mut s = Stream::new();
        for seq in 0..250u64 {
            add(&mut s, 5, seq, &[("n", "x")]);
        }
        let got = dump(&s);
        assert_eq!(got.len(), 250);
        for (at, (id, _)) in got.iter().enumerate() {
            assert_eq!(*id, Id::new(5, at as u64), "at {at}");
        }
        let mut seen = Vec::new();
        s.range(Id::new(5, 100), Id::new(5, 102), None, |id, _| {
            seen.push(id.seq);
            true
        });
        assert_eq!(seen, vec![100, 101, 102]);
    }

    /// Sequence numbers that run on across a node boundary, which is where an
    /// ID stored as a difference is easiest to get wrong.
    #[test]
    fn a_sequence_that_crosses_a_node() {
        let mut s = Stream::new();
        for seq in 0..300u64 {
            add(&mut s, 1, seq, &[("n", "x")]);
        }
        assert!(s.nodes() > 1);
        let got = dump(&s);
        assert_eq!(got.len(), 300);
        assert_eq!(got[299].0, Id::new(1, 299));
        assert!(s.delete(Id::new(1, 250)));
        assert_eq!(dump(&s).len(), 299);
    }

    #[test]
    fn an_entry_with_no_fields_is_still_an_entry() {
        let mut s = Stream::new();
        s.append(Id::new(1, 0), &[], Limits::default())
            .expect("an append");
        add(&mut s, 2, 0, &[("n", "x")]);
        let got = dump(&s);
        assert_eq!(got.len(), 2);
        assert!(got[0].1.is_empty());
    }

    #[test]
    fn a_value_that_looks_like_a_number_comes_back_as_it_went_in() {
        let mut s = Stream::new();
        add(&mut s, 1, 0, &[("n", "007"), ("m", "7")]);
        let got = dump(&s);
        assert_eq!(got[0].1[0].1, b"007".to_vec());
        assert_eq!(got[0].1[1].1, b"7".to_vec());
    }

    #[test]
    fn the_auto_id_follows_the_clock_and_never_goes_back() {
        let mut s = Stream::new();
        assert_eq!(s.auto_id(1000), Some(Id::new(1000, 0)));
        add(&mut s, 1000, 0, &[("n", "x")]);
        assert_eq!(s.auto_id(1000), Some(Id::new(1000, 1)), "same millisecond");
        assert_eq!(s.auto_id(900), Some(Id::new(1000, 1)), "clock went back");
        assert_eq!(s.auto_id(1001), Some(Id::new(1001, 0)));
    }

    #[test]
    fn an_explicit_millisecond_takes_the_next_sequence() {
        let mut s = Stream::new();
        add(&mut s, 5, 0, &[("n", "x")]);
        assert_eq!(s.auto_seq(5), Some(Id::new(5, 1)));
        assert_eq!(s.auto_seq(6), Some(Id::new(6, 0)));
        assert_eq!(s.auto_seq(4), None, "below the last one");
    }

    #[test]
    fn an_id_reads_and_writes() {
        for (text, default, want) in [
            (&b"5"[..], 0, Some(Id::new(5, 0))),
            (b"5", u64::MAX, Some(Id::new(5, u64::MAX))),
            (b"5-3", 0, Some(Id::new(5, 3))),
            (b"0-0", 0, Some(Id::MIN)),
            (b"", 0, None),
            (b"-1", 0, None),
            (b"5-", 0, None),
            (b"a", 0, None),
            (b"5-a", 0, None),
            (b"18446744073709551616", 0, None),
        ] {
            assert_eq!(
                Id::parse(text, default),
                want,
                "{:?}",
                String::from_utf8_lossy(text)
            );
        }
        assert_eq!(Id::new(5, 3).to_vec(), b"5-3".to_vec());
    }

    #[test]
    fn an_id_round_trips_through_its_bytes() {
        for id in [Id::MIN, Id::MAX, Id::new(1, 2), Id::new(u64::MAX, 0)] {
            assert_eq!(Id::from_bytes(id.to_bytes()), id);
        }
        // Big endian is the order that sorts, which is why the format uses it.
        assert!(Id::new(1, 2).to_bytes() < Id::new(1, 3).to_bytes());
        assert!(Id::new(1, u64::MAX).to_bytes() < Id::new(2, 0).to_bytes());
    }

    #[test]
    fn stepping_an_id_carries_and_stops() {
        assert_eq!(Id::new(1, 2).next(), Some(Id::new(1, 3)));
        assert_eq!(Id::new(1, u64::MAX).next(), Some(Id::new(2, 0)));
        assert_eq!(Id::MAX.next(), None);
        assert_eq!(Id::new(1, 3).prev(), Some(Id::new(1, 2)));
        assert_eq!(Id::new(2, 0).prev(), Some(Id::new(1, u64::MAX)));
        assert_eq!(Id::MIN.prev(), None);
    }

    /// Nothing about the answer may depend on where the node boundaries fell.
    #[test]
    fn the_node_size_changes_nothing_but_the_node_count() {
        let mut want = None;
        for entries in [1usize, 2, 7, 100, 4096] {
            let mut s = Stream::new();
            let limits = Limits {
                max_node_bytes: NODE_BYTES,
                max_node_entries: entries,
            };
            for ms in 1..=400u64 {
                let value = format!("v{ms}");
                s.append(Id::new(ms, 0), &[(b"n", value.as_bytes())], limits)
                    .expect("an append");
            }
            for ms in (1..=400u64).step_by(7) {
                s.delete(Id::new(ms, 0));
            }
            let got = dump(&s);
            match &want {
                None => want = Some(got),
                Some(want) => assert_eq!(&got, want, "at {entries} entries a node"),
            }
        }
    }

    /// A byte limit small enough that every entry is its own node.
    #[test]
    fn a_tiny_byte_limit_still_works() {
        let limits = Limits {
            max_node_bytes: 1,
            max_node_entries: NODE_ENTRIES,
        };
        let mut s = Stream::new();
        for ms in 1..=20u64 {
            s.append(Id::new(ms, 0), &[(b"n", b"x")], limits)
                .expect("an append");
        }
        assert_eq!(s.nodes(), 20);
        assert_eq!(dump(&s).len(), 20);
    }

    /// A `DUMP` of a stream taken from a real Redis, as hexadecimal.
    ///
    /// Captured from Redis 8.10.1 in the official Docker image on 2026-09-02,
    /// from a server that had been given exactly this:
    ///
    /// ```text
    /// XADD s 1-1 temperature_celsius 21 relative_humidity 55
    /// XADD s 1-2 temperature_celsius 22 relative_humidity 56
    /// XADD s 2-1 temperature_celsius 23 relative_humidity 57
    /// XADD s 3-1 sensor a
    /// XDEL s 1-2
    /// ```
    ///
    /// Three entries share the master fields and one brings its own, one entry
    /// is deleted rather than taken out, and the ids climb in both halves, so
    /// between them the four cover every branch the encoder has.
    const REDIS_DUMP: &str = "1b0110000000000000000100000000000000014070\
        700000001f000301010102019374656d70657261747572655f63656c736975731491\
        72656c61746976655f68756d69646974791200010201000100011501370105010301\
        0001010116013801050102010101000117013901050100010201000101018673656e\
        736f72078161020601ff030301010101020400406440640000000f00239a5c2c7208\
        ea0a";

    fn unhex(s: &str) -> Vec<u8> {
        let digits: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        digits
            .chunks(2)
            .map(|pair| {
                let of = |b: u8| (b as char).to_digit(16).expect("a hex digit") as u8;
                of(pair[0]) << 4 | of(pair[1])
            })
            .collect()
    }

    /// One RDB length, and how many bytes it took.
    ///
    /// Only the two plain forms, six bits in one byte and fourteen in two,
    /// because those are the only ones this fixture uses and a test that
    /// quietly accepted more would be claiming to check something it does not.
    fn rdb_len(bytes: &[u8], at: usize) -> (usize, usize) {
        match bytes[at] >> 6 {
            0 => (usize::from(bytes[at] & 0x3F), 1),
            1 => (
                usize::from(bytes[at] & 0x3F) << 8 | usize::from(bytes[at + 1]),
                2,
            ),
            other => panic!("the fixture used length form {other}"),
        }
    }

    /// The one node out of the captured dump, and the counters after it.
    fn redis_node() -> (Id, Vec<u8>, Vec<u8>) {
        let dump = unhex(REDIS_DUMP);
        assert_eq!(dump[0], 0x1B, "RDB_TYPE_STREAM_LISTPACKS_3");
        let (nodes, n) = rdb_len(&dump, 1);
        assert_eq!(nodes, 1, "the fixture is one node");
        let mut at = 1 + n;
        let (key, n) = rdb_len(&dump, at);
        assert_eq!(key, 16, "a node key is an id in sixteen bytes");
        at += n;
        let master = Id::from_bytes(dump[at..at + 16].try_into().expect("sixteen bytes"));
        at += 16;
        let (len, n) = rdb_len(&dump, at);
        at += n;
        let lp = dump[at..at + len].to_vec();
        // The last ten bytes are the RDB version and the checksum, which belong
        // to DUMP rather than to the stream.
        let rest = dump[at + len..dump.len() - 10].to_vec();
        (master, lp, rest)
    }

    #[test]
    fn a_node_is_written_the_way_redis_writes_one() {
        let mut s = Stream::new();
        add(
            &mut s,
            1,
            1,
            &[("temperature_celsius", "21"), ("relative_humidity", "55")],
        );
        add(
            &mut s,
            1,
            2,
            &[("temperature_celsius", "22"), ("relative_humidity", "56")],
        );
        add(
            &mut s,
            2,
            1,
            &[("temperature_celsius", "23"), ("relative_humidity", "57")],
        );
        add(&mut s, 3, 1, &[("sensor", "a")]);
        assert!(s.delete(Id::new(1, 2)));

        let (master, lp, _) = redis_node();
        assert_eq!(s.nodes(), 1, "all four fit in one node");
        assert_eq!(s.nodes[0].master, master);
        assert_eq!(s.nodes[0].lp.as_bytes(), &lp[..]);
    }

    #[test]
    fn a_node_redis_wrote_reads_back() {
        let (master, lp, rest) = redis_node();
        let lp = Listpack::from_bytes(&lp).expect("a listpack Redis wrote");
        let s = Stream {
            nodes: VecDeque::from(vec![Node { master, lp }]),
            length: u64::from(rest[0]),
            last: Id::new(u64::from(rest[1]), u64::from(rest[2])),
            max_deleted: Id::new(u64::from(rest[5]), u64::from(rest[6])),
            added: u64::from(rest[7]),
            groups: Vec::new(),
            epoch: 0,
        };

        assert_eq!(s.len(), 3);
        assert_eq!(s.last_id(), Id::new(3, 1));
        assert_eq!(s.max_deleted_id(), Id::new(1, 2));
        assert_eq!(s.added(), 4);
        assert_eq!(s.first_id(), Some(Id::new(1, 1)));
        assert_eq!(
            dump(&s),
            vec![
                (
                    Id::new(1, 1),
                    vec![
                        (b"temperature_celsius".to_vec(), b"21".to_vec()),
                        (b"relative_humidity".to_vec(), b"55".to_vec())
                    ]
                ),
                (
                    Id::new(2, 1),
                    vec![
                        (b"temperature_celsius".to_vec(), b"23".to_vec()),
                        (b"relative_humidity".to_vec(), b"57".to_vec())
                    ]
                ),
                (Id::new(3, 1), vec![(b"sensor".to_vec(), b"a".to_vec())]),
            ]
        );
    }

    /// A stream of `n` entries at 1-0 up to n-0, one field each.
    fn logged(n: u64) -> Stream {
        let mut s = Stream::new();
        for ms in 1..=n {
            add(&mut s, ms, 0, &[("job", "x")]);
        }
        s
    }

    /// What a group read hands back, with the fields flattened.
    fn read(s: &mut Stream, group: &str, who: &str, count: Option<usize>, now: u64) -> Vec<Id> {
        let mut out = Vec::new();
        s.read_group(
            group.as_bytes(),
            who.as_bytes(),
            count,
            false,
            now,
            |id, _| {
                out.push(id);
                true
            },
        )
        .expect("the group");
        out
    }

    /// The read the resume cursor exists for.
    ///
    /// Five hundred entries is five nodes, so this also covers the mark being
    /// carried over a node boundary and the walk picking up in the next one.
    #[test]
    fn a_group_draining_one_at_a_time_gets_every_entry_once() {
        let mut s = logged(500);
        s.create_group(b"workers", Id::MIN, Some(0));
        let mut got = Vec::new();
        for _ in 0..500 {
            got.extend(read(&mut s, "workers", "alice", Some(1), 100));
        }
        assert_eq!(got, (1..=500).map(|ms| Id::new(ms, 0)).collect::<Vec<_>>());
        assert!(read(&mut s, "workers", "alice", Some(1), 100).is_empty());
    }

    /// The state a consumer keeping up with a producer sits in, where the mark
    /// points past the last entry there was and an append lands right on it.
    #[test]
    fn a_group_that_has_caught_up_is_handed_the_next_append() {
        let mut s = logged(3);
        s.create_group(b"workers", Id::MIN, Some(0));
        assert_eq!(read(&mut s, "workers", "alice", None, 100).len(), 3);
        for ms in 4..=200u64 {
            add(&mut s, ms, 0, &[("job", "x")]);
            assert_eq!(
                read(&mut s, "workers", "alice", Some(1), 100),
                vec![Id::new(ms, 0)],
                "the entry appended a moment ago"
            );
        }
    }

    /// A delete moves every byte behind it, so the mark has to be thrown away
    /// rather than followed into the middle of an entry.
    #[test]
    fn a_delete_between_two_group_reads_does_not_lose_the_rest() {
        let mut s = logged(300);
        s.create_group(b"workers", Id::MIN, Some(0));
        let mut got = read(&mut s, "workers", "alice", Some(10), 100);
        assert!(s.delete(Id::new(150, 0)));
        while got.len() < 299 {
            let more = read(&mut s, "workers", "alice", Some(1), 100);
            assert_eq!(more.len(), 1, "at {}", got.len());
            got.extend(more);
        }
        let want: Vec<Id> = (1..=300)
            .filter(|ms| *ms != 150)
            .map(|ms| Id::new(ms, 0))
            .collect();
        assert_eq!(got, want);
    }

    /// `XGROUP SETID` back to the start puts the bookmark somewhere the mark
    /// says nothing about, and the read after it has to walk from the front.
    #[test]
    fn moving_the_bookmark_back_reads_it_all_again() {
        let mut s = logged(250);
        s.create_group(b"workers", Id::MIN, Some(0));
        for _ in 0..120 {
            read(&mut s, "workers", "alice", Some(1), 100);
        }
        s.group_mut(b"workers")
            .expect("the group")
            .set_id(Id::MIN, Some(0));
        let mut got = Vec::new();
        for _ in 0..250 {
            got.extend(read(&mut s, "workers", "alice", Some(1), 100));
        }
        assert_eq!(got, (1..=250).map(|ms| Id::new(ms, 0)).collect::<Vec<_>>());
    }

    /// A trim drops whole nodes from the front, one of which may be the one a
    /// group's mark is about. Node master IDs are never reused, so the mark
    /// cannot be mistaken for one about the node that took its place.
    #[test]
    fn a_trim_under_a_group_does_not_hand_out_the_wrong_entries() {
        let mut s = logged(500);
        s.create_group(b"workers", Id::MIN, Some(0));
        for _ in 0..50 {
            read(&mut s, "workers", "alice", Some(1), 100);
        }
        assert_eq!(s.trim_maxlen(200, false, None), 300);
        let mut got = Vec::new();
        for _ in 0..250 {
            got.extend(read(&mut s, "workers", "alice", Some(1), 100));
        }
        assert_eq!(
            got,
            (301..=500).map(|ms| Id::new(ms, 0)).collect::<Vec<_>>(),
            "everything the trim left that the group had not read"
        );
    }

    #[test]
    fn a_group_is_made_once() {
        let mut s = logged(3);
        assert!(s.create_group(b"workers", Id::MIN, Some(0)));
        assert!(!s.create_group(b"workers", Id::MIN, Some(0)));
        assert!(s.group(b"workers").is_some());
        assert!(s.destroy_group(b"workers"));
        assert!(!s.destroy_group(b"workers"));
        assert!(s.group(b"workers").is_none());
    }

    #[test]
    fn a_group_read_hands_out_what_comes_after_the_bookmark() {
        let mut s = logged(5);
        s.create_group(b"workers", Id::MIN, Some(0));

        assert_eq!(
            read(&mut s, "workers", "alice", Some(2), 100),
            vec![Id::new(1, 0), Id::new(2, 0)]
        );
        // The bookmark moved, so bob gets what alice did not.
        assert_eq!(
            read(&mut s, "workers", "bob", Some(2), 100),
            vec![Id::new(3, 0), Id::new(4, 0)]
        );
        assert_eq!(
            read(&mut s, "workers", "alice", None, 100),
            vec![Id::new(5, 0)]
        );
        assert_eq!(read(&mut s, "workers", "alice", None, 100), vec![]);
    }

    #[test]
    fn a_group_read_fills_the_pending_list() {
        let mut s = logged(3);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 500);

        let g = s.group(b"workers").expect("the group");
        assert_eq!(g.pending_len(), 3);
        assert_eq!(g.last_id(), Id::new(3, 0));
        assert_eq!(g.entries_read(), Some(3));
        assert_eq!(s.lag(g), Some(0));
        let c = g.consumer_named(b"alice").expect("alice");
        assert_eq!(c.len(), 3);
        assert_eq!(c.active(), Some(500));
    }

    #[test]
    fn a_read_that_finds_nothing_is_seen_but_not_active() {
        let mut s = logged(1);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 100);
        read(&mut s, "workers", "alice", None, 900);

        let c = s
            .group(b"workers")
            .expect("the group")
            .consumer_named(b"alice")
            .expect("alice");
        assert_eq!((c.seen(), c.active()), (900, Some(100)));
    }

    #[test]
    fn a_group_starting_at_the_end_reads_only_what_comes_next() {
        let mut s = logged(3);
        s.create_group(b"workers", s.last_id(), Some(s.added()));
        assert_eq!(read(&mut s, "workers", "alice", None, 1), vec![]);
        add(&mut s, 4, 0, &[("job", "x")]);
        assert_eq!(
            read(&mut s, "workers", "alice", None, 1),
            vec![Id::new(4, 0)]
        );
    }

    #[test]
    fn reading_a_group_that_is_not_there_says_so() {
        let mut s = logged(1);
        assert!(
            s.read_group(b"nope", b"alice", None, false, 1, |_, _| true)
                .is_none()
        );
    }

    #[test]
    fn a_consumer_can_re_read_what_it_is_holding() {
        let mut s = logged(4);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(2), 1);
        read(&mut s, "workers", "bob", Some(2), 1);

        let mut out = Vec::new();
        s.read_group_pending(b"workers", b"alice", Id::MIN, None, 2, |id, fields| {
            out.push((id, fields.map(|f| f.len())));
            true
        })
        .expect("the group");
        assert_eq!(
            out,
            vec![(Id::new(1, 0), Some(1)), (Id::new(2, 0), Some(1))]
        );

        // From an ID, which is how a consumer pages through its own backlog.
        let mut after = Vec::new();
        s.read_group_pending(b"workers", b"alice", Id::new(1, 0), None, 2, |id, _| {
            after.push(id);
            true
        });
        assert_eq!(after, vec![Id::new(2, 0)]);
    }

    /// A history read counts as a delivery, which is Redis's behaviour and not
    /// the one I would have guessed.
    ///
    /// Checked against Redis 8.10.1: an entry left idle for 2006 milliseconds
    /// and then read back through `XREADGROUP ... 0` came out idle for 2 with
    /// its delivery count up by one. The count is how many times a consumer has
    /// been told to do the work, and a consumer re-reading its backlog after a
    /// restart has been told again.
    #[test]
    fn re_reading_counts_as_being_handed_it_again() {
        let mut s = logged(2);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 100);
        s.read_group_pending(
            b"workers",
            b"alice",
            Id::MIN,
            None,
            700,
            |_: Id, _: Option<Fields<'_>>| true,
        );

        let g = s.group(b"workers").expect("the group");
        let nack = g.nack(Id::new(1, 0)).expect("a nack");
        assert_eq!((nack.count(), nack.time()), (2, 700));
        // The bookmark does not move, because nothing new was handed out.
        assert_eq!(g.last_id(), Id::new(2, 0));
        assert_eq!(g.pending_len(), 2);
    }

    /// The lag and the read counter, which are two answers and not one.
    ///
    /// Every line here was run against Redis 8.10.1 first and the numbers are
    /// its numbers. The one worth pointing at is the last pair: the counter goes
    /// away and the group's bookmark keeps moving, because a delete ahead of a
    /// group makes the counter unknowable rather than merely stale.
    #[test]
    fn a_hole_in_front_of_a_group_takes_its_lag() {
        let mut s = logged(5);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(3), 1);
        assert_eq!(counters(&s), (Some(3), Some(2)));

        // Deleting something the group has already read leaves both alone,
        // since the hole is behind the bookmark and the entries in front of it
        // are all still there.
        assert!(s.delete(Id::new(1, 0)));
        assert_eq!(counters(&s), (Some(3), Some(2)));

        // Deleting something it has not reached takes the lag, and leaves the
        // counter exactly where it was. Redis does not clear it here.
        assert!(s.delete(Id::new(5, 0)));
        assert_eq!(counters(&s), (Some(3), None));

        // Reading what is left moves the bookmark to 4-0, which is not the last
        // ID the stream ever handed out, so there is still no way to say how far
        // along that is and the counter goes too.
        read(&mut s, "workers", "alice", None, 1);
        assert_eq!(s.group(b"workers").expect("g").last_id(), Id::new(4, 0));
        assert_eq!(counters(&s), (None, None));
    }

    /// A trim is not a hole, so it costs a group nothing, and once it has cut
    /// past the bookmark the lag becomes what is left rather than nothing.
    ///
    /// Checked against Redis 8.10.1 at twenty entries, where the three lines
    /// below read 5, 5 and 2.
    #[test]
    fn trimming_past_a_group_leaves_it_the_length() {
        let mut s = logged(500);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(400), 1);
        assert_eq!(counters(&s), (Some(400), Some(100)));

        // Whole nodes off the front, all of them well behind the bookmark.
        assert_eq!(s.trim_maxlen(200, true, None), 300);
        assert_eq!(counters(&s), (Some(400), Some(100)));

        // And now past it. The bookmark is below every entry left, so the lag is
        // the length: those are exactly the entries the group has still to read.
        assert_eq!(s.trim_maxlen(10, true, None), 190);
        assert_eq!(counters(&s), (Some(400), Some(10)));
    }

    /// The counter and the lag of the one group, which every lag test reads.
    fn counters(s: &Stream) -> (Option<u64>, Option<u64>) {
        let g = s.group(b"workers").expect("the group");
        (g.entries_read(), s.lag(g))
    }

    #[test]
    fn an_entry_that_went_away_still_comes_back_as_a_hole() {
        let mut s = logged(3);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 1);
        assert!(s.delete(Id::new(2, 0)));

        let mut out = Vec::new();
        s.read_group_pending(b"workers", b"alice", Id::MIN, None, 2, |id, fields| {
            out.push((id, fields.is_some()));
            true
        });
        assert_eq!(
            out,
            vec![
                (Id::new(1, 0), true),
                (Id::new(2, 0), false),
                (Id::new(3, 0), true)
            ]
        );
    }

    #[test]
    fn acking_clears_the_pending_list() {
        let mut s = logged(3);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 1);
        let g = s.group_mut(b"workers").expect("the group");
        assert!(g.ack(Id::new(2, 0)));
        assert_eq!(g.pending_len(), 2);
    }

    #[test]
    fn a_claim_moves_work_off_a_consumer_that_stopped() {
        let mut s = logged(2);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 100);

        let mut gone = Vec::new();
        let took = s
            .claim(
                b"workers",
                b"bob",
                &[Id::new(1, 0), Id::new(2, 0)],
                500,
                5_000,
                None,
                true,
                false,
                5_000,
                &mut gone,
            )
            .expect("the group");
        assert_eq!(took, vec![Id::new(1, 0), Id::new(2, 0)]);
        assert!(gone.is_empty());

        let g = s.group(b"workers").expect("the group");
        assert!(g.consumer_named(b"alice").expect("alice").is_empty());
        assert_eq!(g.consumer_named(b"bob").expect("bob").len(), 2);
        assert_eq!(g.nack(Id::new(1, 0)).expect("a nack").count(), 2);
    }

    #[test]
    fn a_claim_leaves_work_that_is_not_idle_enough_alone() {
        let mut s = logged(1);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 100);

        let mut gone = Vec::new();
        let took = s
            .claim(
                b"workers",
                b"bob",
                &[Id::new(1, 0)],
                5_000,
                200,
                None,
                true,
                false,
                200,
                &mut gone,
            )
            .expect("the group");
        assert!(took.is_empty());
        assert_eq!(
            s.group(b"workers")
                .expect("the group")
                .consumer_named(b"alice")
                .expect("alice")
                .len(),
            1
        );
    }

    #[test]
    fn claiming_an_entry_that_went_away_drops_it_instead() {
        let mut s = logged(2);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 100);
        assert!(s.delete(Id::new(1, 0)));

        let mut gone = Vec::new();
        let took = s
            .claim(
                b"workers",
                b"bob",
                &[Id::new(1, 0), Id::new(2, 0)],
                0,
                5_000,
                None,
                true,
                false,
                5_000,
                &mut gone,
            )
            .expect("the group");
        assert_eq!(took, vec![Id::new(2, 0)]);
        assert_eq!(gone, vec![Id::new(1, 0)]);
        assert_eq!(s.group(b"workers").expect("the group").pending_len(), 1);
    }

    #[test]
    fn force_only_works_on_an_entry_that_is_really_there() {
        let mut s = logged(2);
        s.create_group(b"workers", s.last_id(), Some(2));

        let mut gone = Vec::new();
        let took = s
            .claim(
                b"workers",
                b"bob",
                &[Id::new(1, 0), Id::new(99, 0)],
                0,
                100,
                None,
                true,
                true,
                100,
                &mut gone,
            )
            .expect("the group");
        assert_eq!(took, vec![Id::new(1, 0)], "99-0 is not in the stream");
        assert_eq!(s.group(b"workers").expect("the group").pending_len(), 1);
    }

    #[test]
    fn autoclaim_sweeps_the_stale_ones_and_says_where_it_stopped() {
        let mut s = logged(6);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(3), 100);
        read(&mut s, "workers", "alice", None, 900);

        let mut gone = Vec::new();
        let (cursor, took) = s
            .autoclaim(
                b"workers",
                b"bob",
                Id::MIN,
                500,
                100,
                true,
                1_000,
                &mut gone,
            )
            .expect("the group");
        assert_eq!(cursor, None, "the sweep reached the end");
        assert_eq!(took, vec![Id::new(1, 0), Id::new(2, 0), Id::new(3, 0)]);
        assert_eq!(
            s.group(b"workers")
                .expect("the group")
                .consumer_named(b"bob")
                .expect("bob")
                .len(),
            3
        );
    }

    #[test]
    fn autoclaim_hands_back_a_cursor_when_it_hits_the_count() {
        let mut s = logged(10);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", None, 100);

        let mut gone = Vec::new();
        let (cursor, took) = s
            .autoclaim(b"workers", b"bob", Id::MIN, 0, 4, true, 1_000, &mut gone)
            .expect("the group");
        assert_eq!(took.len(), 4);
        assert_eq!(cursor, Some(Id::new(5, 0)));

        // And carrying on from the cursor takes the rest.
        let (cursor, took) = s
            .autoclaim(
                b"workers",
                b"bob",
                cursor.expect("a cursor"),
                0,
                100,
                true,
                1_000,
                &mut gone,
            )
            .expect("the group");
        assert_eq!(took.len(), 6);
        assert_eq!(cursor, None);
    }

    #[test]
    fn a_group_survives_the_stream_being_trimmed_under_it() {
        let mut s = logged(10);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(5), 100);
        // Trimming takes entries alice is still holding.
        assert_eq!(s.trim_maxlen(3, true, None), 7);

        assert_eq!(s.group(b"workers").expect("the group").pending_len(), 5);
        let mut gone = Vec::new();
        let took = s
            .claim(
                b"workers",
                b"bob",
                &(1..=5).map(|ms| Id::new(ms, 0)).collect::<Vec<_>>(),
                0,
                1_000,
                None,
                true,
                false,
                1_000,
                &mut gone,
            )
            .expect("the group");
        assert!(took.is_empty(), "none of them are there any more");
        assert_eq!(gone.len(), 5);
        assert_eq!(s.group(b"workers").expect("the group").pending_len(), 0);
    }

    /// Out and back, checking everything a client can see on the way.
    fn round_trip(s: &Stream) -> Stream {
        let mut bytes = Vec::new();
        s.freeze(&mut bytes);
        let back = Stream::thaw(&bytes).expect("our own bytes");
        assert_eq!(dump(&back), dump(s), "the entries");
        assert_eq!(back.len(), s.len(), "the length");
        assert_eq!(back.added(), s.added(), "the count added");
        assert_eq!(back.last_id(), s.last_id(), "the last ID");
        assert_eq!(back.max_deleted_id(), s.max_deleted_id(), "the max deleted");
        assert_eq!(back.nodes(), s.nodes(), "the node count");
        assert_eq!(back, *s, "the whole thing");
        back
    }

    #[test]
    fn a_frozen_stream_comes_back_with_every_entry_it_held() {
        let mut s = Stream::new();
        for ms in 1..=500u64 {
            add(&mut s, ms, 0, &[("job", "x"), ("n", "1")]);
        }
        add(&mut s, 500, 1, &[("job", "y")]);
        assert!(s.nodes() > 1, "more than one node, so the walk is tested");
        round_trip(&s);
    }

    #[test]
    fn a_frozen_stream_keeps_the_holes_and_the_counters() {
        let mut s = logged(200);
        for ms in [3u64, 4, 5, 100, 199] {
            assert!(s.delete(Id::new(ms, 0)));
        }
        s.trim_minid(Id::new(20, 0), true, None);
        let back = round_trip(&s);
        assert_eq!(back.first_id(), Some(Id::new(20, 0)));
        assert!(!back.contains(Id::new(100, 0)), "a hole is still a hole");
        assert_eq!(back.max_deleted_id(), Id::new(199, 0));
    }

    #[test]
    fn a_frozen_stream_keeps_its_groups_and_who_is_holding_what() {
        let mut s = logged(20);
        s.create_group(b"workers", Id::MIN, Some(0));
        s.create_group(b"audit", Id::new(5, 0), None);
        read(&mut s, "workers", "alice", Some(6), 1_000);
        read(&mut s, "workers", "bob", Some(4), 2_000);
        // One handed back to the group with nobody holding it, so the NACK count
        // has something to come back as.
        assert_eq!(
            s.nack(b"workers", Id::new(2, 0), Retry::Keep, true),
            Some(true)
        );
        // And one consumer deleted, so a slot in the middle is empty and the
        // slot numbers behind it have to survive.
        s.group_mut(b"workers")
            .expect("the group")
            .create_consumer(b"carol", 3_000);
        read(&mut s, "workers", "dave", Some(2), 4_000);
        s.group_mut(b"workers")
            .expect("the group")
            .delete_consumer(b"carol");

        let back = round_trip(&s);
        let g = back.group(b"workers").expect("the group");
        assert_eq!(g.pending_len(), 12);
        assert_eq!(g.nacked_len(), 1);
        assert_eq!(g.entries_read(), Some(12));
        // Six, less the one handed back to the group.
        assert_eq!(g.consumer_named(b"alice").expect("alice").len(), 5);
        assert_eq!(g.consumer_named(b"bob").expect("bob").len(), 4);
        assert_eq!(g.consumer_named(b"dave").expect("dave").len(), 2);
        assert_eq!(g.consumer_named(b"carol"), None);
        // Dave came after carol, so his slot is the fourth one and reading the
        // empties back as empties is what keeps his entries his.
        assert_eq!(g.slot(b"dave"), Some(3));
        assert_eq!(g.nack(Id::new(1, 0)).expect("a nack").owner(), Some(0));
        assert_eq!(g.nack(Id::new(2, 0)).expect("a nack").owner(), None);
        assert_eq!(
            back.group(b"audit").expect("audit").last_id(),
            Id::new(5, 0)
        );
        assert_eq!(back.group(b"audit").expect("audit").entries_read(), None);
    }

    #[test]
    fn a_stream_that_came_back_still_takes_entries_and_reads_them() {
        let mut s = logged(10);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(4), 1_000);

        let mut back = round_trip(&s);
        add(&mut back, 11, 0, &[("job", "new")]);
        assert_eq!(back.len(), 11);
        assert_eq!(
            read(&mut back, "workers", "alice", Some(3), 2_000),
            vec![Id::new(5, 0), Id::new(6, 0), Id::new(7, 0)]
        );
        assert!(
            back.group_mut(b"workers")
                .expect("the group")
                .ack(Id::new(1, 0))
        );
        assert_eq!(back.group(b"workers").expect("the group").pending_len(), 6);
    }

    #[test]
    fn an_empty_stream_that_still_exists_comes_back() {
        let mut s = logged(3);
        for ms in 1..=3u64 {
            assert!(s.delete(Id::new(ms, 0)));
        }
        assert_eq!(s.nodes(), 0, "the last node went with the last entry");
        let back = round_trip(&s);
        assert!(back.is_empty());
        // The whole reason an empty stream is kept: a new entry still has to
        // beat the ID of one that is gone.
        assert_eq!(back.last_id(), Id::new(3, 0));
        round_trip(&Stream::new());
    }

    #[test]
    fn a_frozen_stream_that_arrives_damaged_is_an_error_and_not_a_panic() {
        let mut s = logged(8);
        s.create_group(b"workers", Id::MIN, Some(0));
        read(&mut s, "workers", "alice", Some(3), 1_000);
        let mut bytes = Vec::new();
        s.freeze(&mut bytes);

        for cut in 0..bytes.len() {
            assert!(Stream::thaw(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        // Every bit of the header and the front of the first node, which is
        // where the counts and the lengths that a reader trusts all live.
        for at in 0..bytes.len().min(40) {
            for bit in 0..8 {
                let mut bad = bytes.clone();
                bad[at] ^= 1 << bit;
                // It either parses into some other stream or it does not. Either
                // way it comes back rather than going through a length that was
                // never checked.
                let _ = Stream::thaw(&bad);
            }
        }
        assert_eq!(Stream::thaw(&[]), Err(Broken::Short));
        assert_eq!(Stream::thaw(&[9]), Err(Broken::Form));
    }

    #[test]
    fn nothing_at_all() {
        let mut s = Stream::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.first_id(), None);
        assert_eq!(s.last_id(), Id::MIN);
        assert_eq!(s.trim_maxlen(0, true, None), 0);
        assert_eq!(s.trim_minid(Id::MAX, true, None), 0);
        assert_eq!(dump(&s), vec![]);
    }
}
