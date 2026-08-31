//! Variable length bytes belonging to one collection, back to back.
//!
//! Two things inside a collection are the same problem. A set or a hash interns
//! its member and field names so that writing the same field again touches no
//! name bytes (`05` section 3), and a hash in the native band has to put its
//! values somewhere that is not one allocation per field, because one allocation
//! per field is the thing the element per row layout exists to avoid. Both want
//! a stretch of bytes with an offset handed back, both want a rewrite to leave
//! the old bytes behind rather than move everything after them, and both want
//! those bytes back eventually.
//!
//! ```text
//!   bytes                                    dead
//! +-------+---------+-------+---------+     bytes nothing points at, counted
//! | name  | oldval  | name  | value   |     here and given back once they
//! +-------+---------+-------+---------+     outnumber the ones that are live
//!    ^ at, len              ^ at, len
//! ```
//!
//! # It does not hold the references
//!
//! [`Blob::push`] hands back an offset and nothing else, and [`Blob::read`] takes
//! an offset and a length. The reference is the caller's to shape, which is not
//! ceremony: [`crate::Elements`] keeps a name's length in sixteen bits so a row
//! stays twelve bytes, and a hash value cannot be capped at sixty four kilobytes
//! because Redis lets one be half a gigabyte. One blob, two reference layouts,
//! and neither of them is a compromise with the other.
//!
//! [`Span`] is here for the callers that have no reason to pack it tighter.
//!
//! # Giving the dead bytes back
//!
//! A rewrite appends and abandons, so a hash whose values are written over and
//! over holds every value it ever had until something clears up. That something
//! is [`Blob::compact`], which the owner runs when [`Blob::worth_compacting`]
//! says so and drives itself, because the owner is the only thing that knows
//! where its references are. Until then the dead bytes are counted and reported
//! rather than pretended away, which is the rule the arena follows and for the
//! same reason: a number `INFO memory` can show is a leak you can see.
//!
//! Half is the line, with a floor of four kilobytes under it. Below the half the
//! copy costs more than the bytes are worth, and below the floor there are not
//! enough bytes to be worth a copy at any ratio at all.

/// Where something is in a blob, for a caller with no reason to pack it tighter.
///
/// Eight bytes. A hash value uses this, because a value can be as long as the
/// 512 MiB Redis puts on everything and there is no shorter length that holds
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    /// Where the bytes start.
    pub at: u32,
    /// How many of them there are.
    pub len: u32,
}

/// Dead bytes below this are left alone whatever the ratio says.
const FLOOR: usize = 4096;

/// Bytes belonging to one collection, appended to and occasionally rebuilt.
#[derive(Debug, Default, Clone)]
pub struct Blob {
    bytes: Vec<u8>,
    dead: usize,
}

impl Blob {
    /// An empty blob that has not allocated anything.
    ///
    /// A collection is made by its first write, so the empty case is the common
    /// one and it does not deserve an allocation.
    #[must_use]
    pub const fn new() -> Blob {
        Blob {
            bytes: Vec::new(),
            dead: 0,
        }
    }

    /// An empty blob with room already taken.
    #[must_use]
    pub fn with_capacity(n: usize) -> Blob {
        Blob {
            bytes: Vec::with_capacity(n),
            dead: 0,
        }
    }

    /// Every byte here, live and dead together.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether nothing has ever been written.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Bytes nothing points at any more.
    #[inline]
    #[must_use]
    pub const fn dead(&self) -> usize {
        self.dead
    }

    /// What this costs, which is the allocation and not the used part of it.
    #[inline]
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    /// Append `bytes` and say where they went.
    ///
    /// # Panics
    ///
    /// If the blob would pass four gigabytes, which no collection reaches
    /// without passing a row limit first.
    #[inline]
    pub fn push(&mut self, bytes: &[u8]) -> u32 {
        let at = u32::try_from(self.bytes.len()).expect("the blob is under 4 GiB");
        // By [`crate::grow`]'s policy and not by `Vec`'s, for the same reason
        // the row array above it grows that way: a blob holding the names of a
        // large collection is megabytes, and half of a doubled one is air.
        crate::grow::reserve(&mut self.bytes, bytes.len());
        self.bytes.extend_from_slice(bytes);
        at
    }

    /// Append `bytes` and say where they went, as a [`Span`].
    ///
    /// # Panics
    ///
    /// If the blob would pass four gigabytes, or `bytes` is longer than one.
    #[inline]
    pub fn push_span(&mut self, bytes: &[u8]) -> Span {
        Span {
            at: self.push(bytes),
            len: u32::try_from(bytes.len()).expect("no one value is 4 GiB"),
        }
    }

    /// The `len` bytes at `at`.
    ///
    /// # Panics
    ///
    /// If they are not inside the blob, which means a reference was kept across
    /// a [`Blob::compact`] without being moved.
    #[inline]
    #[must_use]
    pub fn read(&self, at: u32, len: usize) -> &[u8] {
        let at = at as usize;
        &self.bytes[at..at + len]
    }

    /// The bytes a [`Span`] points at.
    #[inline]
    #[must_use]
    pub fn span(&self, span: Span) -> &[u8] {
        self.read(span.at, span.len as usize)
    }

    /// Say that `len` bytes are not pointed at any more.
    ///
    /// This frees nothing. It moves the number that decides when
    /// [`Blob::compact`] is worth running.
    #[inline]
    pub const fn release(&mut self, len: usize) {
        self.dead += len;
    }

    /// Say that a [`Span`] is not pointed at any more.
    #[inline]
    pub const fn release_span(&mut self, span: Span) {
        self.release(span.len as usize);
    }

    /// Throw everything away and keep the allocation.
    #[inline]
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.dead = 0;
    }

    /// Whether the dead bytes are worth a rebuild.
    #[inline]
    #[must_use]
    pub const fn worth_compacting(&self) -> bool {
        self.dead >= FLOOR && self.dead * 2 >= self.bytes.len()
    }

    /// Rebuild, keeping only what `keep` points at.
    ///
    /// The owner walks its own references and calls [`Keep::moved`] on each,
    /// which copies those bytes into the new blob and rewrites the offset in
    /// place. Anything not offered is gone. A reference the owner forgets to
    /// offer becomes a reference into a blob that moved underneath it, and
    /// [`Blob::read`] turns that into a panic rather than into wrong bytes.
    ///
    /// The order the owner walks in becomes the order in the new blob, so
    /// walking in row order leaves a sequential read sequential.
    pub fn compact<F>(&mut self, keep: F)
    where
        F: FnOnce(&mut Keep<'_>),
    {
        let fresh = {
            let mut k = Keep {
                old: &self.bytes,
                fresh: Vec::with_capacity(self.bytes.len() - self.dead),
            };
            keep(&mut k);
            k.fresh
        };
        self.bytes = fresh;
        self.dead = 0;
    }
}

/// A rebuild in progress, handed to the owner so it can move its references.
#[derive(Debug)]
pub struct Keep<'a> {
    old: &'a [u8],
    fresh: Vec<u8>,
}

impl Keep<'_> {
    /// Carry the `len` bytes at `*at` over, and point `at` at where they landed.
    ///
    /// # Panics
    ///
    /// If they are not inside the old blob, which is the mistake
    /// [`Blob::read`] catches and it is caught here for the same reason.
    #[inline]
    pub fn moved(&mut self, at: &mut u32, len: usize) {
        let from = *at as usize;
        let to = u32::try_from(self.fresh.len()).expect("the blob only shrinks here");
        self.fresh.extend_from_slice(&self.old[from..from + len]);
        *at = to;
    }

    /// The same for a [`Span`], whose length does not change.
    #[inline]
    pub fn moved_span(&mut self, span: &mut Span) {
        let len = span.len as usize;
        self.moved(&mut span.at, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_goes_in_comes_back_out() {
        let mut b = Blob::new();
        let one = b.push_span(b"field");
        let two = b.push_span(b"");
        let three = b.push_span(b"a longer value than the first one");

        assert_eq!(b.span(one), b"field");
        assert_eq!(b.span(two), b"");
        assert_eq!(b.span(three), b"a longer value than the first one");
        assert_eq!(b.len(), 5 + 33);
        assert_eq!(b.dead(), 0);
    }

    #[test]
    fn a_rewrite_leaves_the_old_bytes_behind_and_says_so() {
        let mut b = Blob::new();
        let old = b.push_span(b"before");
        b.release_span(old);
        let new = b.push_span(b"after");

        assert_eq!(b.span(new), b"after");
        assert_eq!(b.dead(), 6, "the old bytes are still there and counted");
        assert_eq!(b.len(), 11);
    }

    #[test]
    fn the_dead_bytes_come_back_and_the_live_ones_move() {
        let mut b = Blob::new();
        // Twenty kilobytes written, half of it abandoned, which is over the
        // floor and at the ratio.
        let mut live: Vec<Span> = Vec::new();
        for i in 0..100u32 {
            let bytes = vec![b'a' + u8::try_from(i % 26).expect("under 26"); 100];
            let first = b.push_span(&bytes);
            b.release_span(first);
            live.push(b.push_span(&bytes));
        }
        assert_eq!(b.dead(), 10_000);
        assert!(b.worth_compacting());

        let want: Vec<Vec<u8>> = live.iter().map(|&s| b.span(s).to_vec()).collect();
        b.compact(|k| {
            for span in &mut live {
                k.moved_span(span);
            }
        });

        assert_eq!(b.dead(), 0);
        assert_eq!(b.len(), 10_000, "only the live half survived");
        for (span, bytes) in live.iter().zip(&want) {
            assert_eq!(b.span(*span), &bytes[..], "a reference moved wrongly");
        }
    }

    #[test]
    fn a_small_or_mostly_live_blob_is_left_alone() {
        let mut b = Blob::new();
        b.push(&vec![0u8; 100_000]);
        b.release(3000);
        assert!(!b.worth_compacting(), "under the floor, whatever the ratio");

        let mut c = Blob::new();
        c.push(&vec![0u8; 100_000]);
        c.release(40_000);
        assert!(!c.worth_compacting(), "over the floor and under the half");
        c.release(10_000);
        assert!(c.worth_compacting(), "and at the half it is worth doing");
    }

    #[test]
    fn clearing_keeps_the_allocation_and_forgets_the_dead() {
        let mut b = Blob::with_capacity(1024);
        b.push(b"something");
        b.release(4);
        b.clear();

        assert!(b.is_empty());
        assert_eq!(b.dead(), 0);
        assert!(b.memory_bytes() >= 1024, "the allocation stayed");
    }
}
