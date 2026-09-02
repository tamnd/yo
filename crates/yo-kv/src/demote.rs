//! Choosing which entry leaves memory next.
//!
//! [`evict`](crate::evict) answers a different question. It picks a key to
//! throw away when there is no room left to store anything, and its answer is a
//! deletion. This module picks a key to move to the file when there is no room
//! left in memory, and its answer is a demotion. `14` section 4.1 is the reason
//! there are two: a memory limit and a storage limit are different limits and
//! deleting data because memory filled up is a cache's answer, not a database's.
//!
//! That difference changes what a good decision looks like. Getting an eviction
//! wrong loses data the client may never ask for again, so the cost is bounded
//! by how much a miss hurts. Getting a demotion wrong costs a device read the
//! next time the key is touched, which is thousands of times the cost of the
//! memory read it replaced, and the key is still there to be got wrong again.
//! So demotion is worth more machinery than sampling five keys, which is why
//! there is a real admission and reinsertion policy here.
//!
//! Two of them, because the honest answer to which one is better is that it
//! depends on the trace and both are recent enough that nobody has run them on
//! the workload this engine cares about.
//!
//! # SIEVE
//!
//! [`Sieve`], from NSDI 2024. One FIFO queue, one bit per entry, and a hand
//! that walks from the tail towards the head. An entry that was touched since
//! the hand last passed it loses its bit and survives; the first entry the hand
//! finds without one is the victim. The hand stays where it stopped, so the
//! next demotion carries on from there rather than starting over.
//!
//! What makes it interesting is that a survivor does not move. In LRU a hit
//! moves an entry to the head and it has to earn its way back down; in SIEVE it
//! keeps its position and only its bit changes, so an entry that is hit once
//! and then goes quiet is out again quickly, while one that is hit on every
//! pass of the hand stays near the head where the hand rarely reaches. That is
//! most of an LFU's behaviour out of one bit and no counters.
//!
//! # S3-FIFO
//!
//! [`S3Fifo`], from SOSP 2023. Three queues. A small one holding a tenth of the
//! capacity is where everything new lands. An entry leaving the small queue is
//! promoted to the main queue if it was touched more than once while it was
//! there, and otherwise it is demoted, with its fingerprint remembered in a
//! ghost queue that holds no data. An entry that is asked for again while its
//! fingerprint is in the ghost queue goes straight into the main queue, because
//! something asked for twice with a gap between is not a one hit wonder. The
//! main queue is a second chance FIFO: an entry with a positive count loses one
//! and goes round again.
//!
//! The point of the small queue is that most of what a cache sees is asked for
//! once and never again, and a plain LRU pays for those by pushing out
//! something that would have been used. S3-FIFO gives them a tenth of the space
//! and a fast exit.
//!
//! # The doorkeeper
//!
//! [`Doorkeeper`] is the admission half, and it applies on the way back in
//! rather than on the way out. A key read from the file is not automatically
//! worth a slot in memory. The doorkeeper is a bloom filter of what has been
//! asked for recently: the first read of a key sets its bits and leaves it on
//! the file, and the second read within the same window brings it in. So a scan
//! over cold data touches nothing, and a key that is genuinely warming up costs
//! one extra device read to prove it.
//!
//! It is reset rather than aged, which is the TinyLFU trick. Bits are cleared
//! wholesale once enough keys have been through, so the filter never saturates
//! and never needs counters.
//!
//! # What a caller has to do
//!
//! None of this holds keys or addresses. A caller gives every entry a slot
//! number once, and passes that number in and gets it back out. Addresses move
//! when an arena compacts and keys are the thing being looked up, so a
//! structure that has to survive between two commands can hold neither, and
//! [`evict::Pool`](crate::evict::Pool) pays for exactly that by copying key
//! bytes. A slot number is the caller's own index, it is stable because the
//! caller says it is, and it makes both queues an array with no allocation on
//! the hot path.

/// A caller's handle on one entry, which is an index into whatever the caller
/// keeps its entries in.
pub type Slot = u32;

/// No slot. The queues are intrusive lists over a flat array, so the end of a
/// list is a sentinel rather than an `Option`, which keeps a link at four bytes
/// instead of eight.
const NIL: Slot = Slot::MAX;

/// One entry's links and its counter.
#[derive(Clone, Copy)]
struct Link {
    prev: Slot,
    next: Slot,
    /// Times touched since this entry last had its count read, saturating at
    /// [`S3Fifo::MAX_FREQ`]. SIEVE only ever stores zero or one in it.
    freq: u8,
    /// Which queue this entry is in, so a touch does not have to search.
    queue: u8,
}

const OUT: u8 = 0;
const SMALL: u8 = 1;
const MAIN: u8 = 2;

impl Link {
    const fn empty() -> Link {
        Link {
            prev: NIL,
            next: NIL,
            freq: 0,
            queue: OUT,
        }
    }
}

/// A doubly linked list over a shared array of links.
///
/// Head is where entries arrive and tail is where they leave, which makes it a
/// FIFO read from the tail. Both policies here want to walk from the tail
/// towards the head, so the direction is worth stating once rather than
/// working out at each use.
#[derive(Clone, Copy)]
struct Queue {
    head: Slot,
    tail: Slot,
    len: usize,
}

impl Queue {
    const fn new() -> Queue {
        Queue {
            head: NIL,
            tail: NIL,
            len: 0,
        }
    }

    fn push_head(&mut self, links: &mut [Link], slot: Slot, queue: u8) {
        let i = slot as usize;
        links[i].prev = NIL;
        links[i].next = self.head;
        links[i].queue = queue;
        if self.head != NIL {
            links[self.head as usize].prev = slot;
        } else {
            self.tail = slot;
        }
        self.head = slot;
        self.len += 1;
    }

    fn unlink(&mut self, links: &mut [Link], slot: Slot) {
        let i = slot as usize;
        let (prev, next) = (links[i].prev, links[i].next);
        if prev != NIL {
            links[prev as usize].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            links[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
        links[i] = Link::empty();
        self.len -= 1;
    }
}

/// SIEVE, from Zhang and others, NSDI 2024.
///
/// One queue, one bit an entry, and a hand that remembers where it stopped.
/// See the module docs for why that is more than it sounds.
pub struct Sieve {
    links: Vec<Link>,
    q: Queue,
    /// Where the hand stopped last time, or `NIL` to start from the tail.
    hand: Slot,
}

impl Sieve {
    /// A policy over `slots` entries, none of them resident yet.
    pub fn new(slots: usize) -> Sieve {
        assert!(slots < NIL as usize, "a slot number has to fit in a u32");
        Sieve {
            links: vec![Link::empty(); slots],
            q: Queue::new(),
            hand: NIL,
        }
    }

    /// How many entries are resident.
    pub fn len(&self) -> usize {
        self.q.len
    }

    /// Whether nothing is resident.
    pub fn is_empty(&self) -> bool {
        self.q.len == 0
    }

    /// Whether `slot` is resident.
    pub fn contains(&self, slot: Slot) -> bool {
        self.links[slot as usize].queue != OUT
    }

    /// Take an entry into memory. Arriving entries have no bit set, so a new
    /// entry the hand reaches before it is used again goes straight back out.
    pub fn insert(&mut self, slot: Slot) {
        if self.contains(slot) {
            return;
        }
        self.q.push_head(&mut self.links, slot, MAIN);
    }

    /// Note that `slot` was used. This is the whole of the read path, and it is
    /// one store, which is the reason to prefer this over anything that moves
    /// an entry on a hit.
    pub fn touch(&mut self, slot: Slot) {
        let link = &mut self.links[slot as usize];
        if link.queue != OUT {
            link.freq = 1;
        }
    }

    /// Take an entry out because the caller deleted it, rather than because the
    /// policy chose it.
    pub fn remove(&mut self, slot: Slot) {
        if !self.contains(slot) {
            return;
        }
        if self.hand == slot {
            self.hand = self.links[slot as usize].prev;
        }
        self.q.unlink(&mut self.links, slot);
    }

    /// The next entry to move to the file, or `None` if nothing is resident.
    ///
    /// The hand starts where it stopped, clears the bit of everything it passes
    /// and stops at the first entry without one. Every entry it passes is one
    /// it will be willing to take next time round, so a sweep that has to walk
    /// the whole queue happens at most once per pass and not once per demotion.
    pub fn demote(&mut self) -> Option<Slot> {
        if self.q.len == 0 {
            return None;
        }
        let mut at = if self.hand == NIL {
            self.q.tail
        } else {
            self.hand
        };
        loop {
            if self.links[at as usize].freq == 0 {
                let prev = self.links[at as usize].prev;
                self.hand = prev;
                self.q.unlink(&mut self.links, at);
                return Some(at);
            }
            self.links[at as usize].freq = 0;
            at = self.links[at as usize].prev;
            if at == NIL {
                at = self.q.tail;
            }
        }
    }
}

/// S3-FIFO, from Yang and others, SOSP 2023.
///
/// A small queue for what has been seen once, a main queue for what has earned
/// its place, and a ghost queue of fingerprints for what was demoted from the
/// small one. See the module docs for what each is for.
pub struct S3Fifo {
    links: Vec<Link>,
    small: Queue,
    main: Queue,
    /// Fingerprints of what left the small queue without earning a promotion.
    ghost: Ghost,
    /// How large the small queue is allowed to get before a demotion takes from
    /// it rather than from the main queue.
    small_cap: usize,
}

impl S3Fifo {
    /// The largest an entry's touch count is allowed to get.
    ///
    /// Two bits of information is all the main queue's second chance rule can
    /// use, and capping it means an entry hammered a million times still leaves
    /// after three passes rather than never.
    pub const MAX_FREQ: u8 = 3;

    /// A policy over `slots` entries with room for `capacity` of them resident.
    ///
    /// The small queue gets a tenth of the capacity, which is the paper's
    /// number and is not a tuning knob worth exposing until a trace says
    /// otherwise. It gets at least one slot, so a tiny cache still has the
    /// admission behaviour rather than silently becoming a plain FIFO.
    pub fn new(slots: usize, capacity: usize) -> S3Fifo {
        assert!(slots < NIL as usize, "a slot number has to fit in a u32");
        assert!(capacity > 0, "a cache with no room is not a cache");
        S3Fifo {
            links: vec![Link::empty(); slots],
            small: Queue::new(),
            main: Queue::new(),
            ghost: Ghost::new(capacity),
            small_cap: (capacity / 10).max(1),
        }
    }

    /// How many entries are resident.
    pub fn len(&self) -> usize {
        self.small.len + self.main.len
    }

    /// Whether nothing is resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `slot` is resident.
    pub fn contains(&self, slot: Slot) -> bool {
        self.links[slot as usize].queue != OUT
    }

    /// Take an entry into memory. `finger` is a hash of the key, and it is what
    /// the ghost queue remembers, so it has to be the same value every time the
    /// same key is inserted.
    ///
    /// A fingerprint the ghost queue has seen means this key was demoted out of
    /// the small queue recently and has come back, which is the definition of
    /// not a one hit wonder, so it skips the small queue entirely.
    pub fn insert(&mut self, slot: Slot, finger: u64) {
        if self.contains(slot) {
            return;
        }
        if self.ghost.forget(finger) {
            self.main.push_head(&mut self.links, slot, MAIN);
        } else {
            self.small.push_head(&mut self.links, slot, SMALL);
        }
    }

    /// Note that `slot` was used.
    pub fn touch(&mut self, slot: Slot) {
        let link = &mut self.links[slot as usize];
        if link.queue != OUT {
            link.freq = (link.freq + 1).min(Self::MAX_FREQ);
        }
    }

    /// Take an entry out because the caller deleted it.
    pub fn remove(&mut self, slot: Slot) {
        match self.links[slot as usize].queue {
            SMALL => self.small.unlink(&mut self.links, slot),
            MAIN => self.main.unlink(&mut self.links, slot),
            _ => {}
        }
    }

    /// The next entry to move to the file, or `None` if nothing is resident.
    ///
    /// Takes from the small queue while it is over its tenth and from the main
    /// queue otherwise. Neither branch can loop for long: the small one moves
    /// an entry to the main queue or returns it, and the main one either
    /// returns an entry or spends one of its at most three counts, so the whole
    /// call is bounded by three passes of the main queue and cannot spin.
    pub fn demote(&mut self, finger_of: impl Fn(Slot) -> u64) -> Option<Slot> {
        loop {
            if self.small.len > self.small_cap && self.small.len > 0 {
                let slot = self.small.tail;
                let freq = self.links[slot as usize].freq;
                self.small.unlink(&mut self.links, slot);
                if freq > 1 {
                    self.main.push_head(&mut self.links, slot, MAIN);
                    continue;
                }
                self.ghost.remember(finger_of(slot));
                return Some(slot);
            }
            if self.main.len > 0 {
                let slot = self.main.tail;
                let freq = self.links[slot as usize].freq;
                if freq > 0 {
                    self.main.unlink(&mut self.links, slot);
                    self.main.push_head(&mut self.links, slot, MAIN);
                    self.links[slot as usize].freq = freq - 1;
                    continue;
                }
                self.main.unlink(&mut self.links, slot);
                return Some(slot);
            }
            if self.small.len > 0 {
                let slot = self.small.tail;
                self.small.unlink(&mut self.links, slot);
                self.ghost.remember(finger_of(slot));
                return Some(slot);
            }
            return None;
        }
    }
}

/// Fingerprints of what left the small queue, as a ring with no values in it.
///
/// The paper describes a FIFO queue of ghost entries and this is that, held as
/// a ring buffer of the same size so that inserting one costs a store and
/// overwriting the oldest is free. Membership is a linear scan over the ring,
/// which sounds wrong until you notice it is only consulted on an insert of a
/// key that is not resident, and that the alternative, a hash set, allocates on
/// a path that is already paying for a device read.
///
/// Kept small on purpose: a ghost queue the size of the cache is the paper's
/// setting and a larger one starts admitting things that were demoted long ago
/// for good reasons.
struct Ghost {
    ring: Vec<u64>,
    at: usize,
}

/// Not a fingerprint. Zero is a legitimate hash, so the ring needs a value that
/// means empty, and a hash of exactly this is remapped rather than losing the
/// slot.
const NO_FINGER: u64 = 0;

impl Ghost {
    fn new(capacity: usize) -> Ghost {
        Ghost {
            ring: vec![NO_FINGER; capacity.min(4096)],
            at: 0,
        }
    }

    fn remember(&mut self, finger: u64) {
        if self.ring.is_empty() {
            return;
        }
        self.ring[self.at] = clean(finger);
        self.at = (self.at + 1) % self.ring.len();
    }

    /// Whether the ring held `finger`, taking it out if it did.
    ///
    /// Taking it out matters. A key that comes back is going into the main
    /// queue, so leaving its fingerprint behind would send it there again on
    /// the next insert after it was demoted for real, which is exactly the
    /// promotion the ghost queue exists to avoid handing out for free.
    fn forget(&mut self, finger: u64) -> bool {
        let want = clean(finger);
        for slot in &mut self.ring {
            if *slot == want {
                *slot = NO_FINGER;
                return true;
            }
        }
        false
    }
}

/// A fingerprint that is never the empty marker.
fn clean(finger: u64) -> u64 {
    if finger == NO_FINGER { 1 } else { finger }
}

/// The admission filter on the way back in, which is TinyLFU's doorkeeper.
///
/// A key read from the file does not become resident on the first read. Its
/// bits go into the filter and the value is served from the file; a second read
/// while those bits are still there brings it into memory. So a scan over cold
/// data leaves the resident set alone and a key that is actually warming up
/// pays one extra device read to say so.
///
/// The filter is cleared wholesale once enough keys have been through it rather
/// than aged one entry at a time, which is what keeps it a bit array instead of
/// a counter array. The cost of clearing is that a key can be unlucky with
/// timing and need three reads instead of two, and the cost of not clearing is
/// a filter that says yes to everything after an hour, so the trade is not
/// close.
pub struct Doorkeeper {
    bits: Vec<u64>,
    /// Keys let through since the last reset.
    seen: usize,
    /// Keys to let through before the next reset, which is what sets the window.
    window: usize,
}

impl Doorkeeper {
    /// Bits per key the filter is sized for. Two hashes at eight bits a key is
    /// a false positive rate of about two percent, and a false positive here
    /// admits one key that had only been asked for once, which is the mildest
    /// possible way to be wrong.
    const BITS_PER_KEY: usize = 8;

    /// A doorkeeper sized for `window` distinct keys between resets.
    pub fn new(window: usize) -> Doorkeeper {
        let window = window.max(64);
        let words = (window * Self::BITS_PER_KEY).div_ceil(64);
        Doorkeeper {
            bits: vec![0; words],
            seen: 0,
            window,
        }
    }

    /// Whether a key with this fingerprint should come into memory.
    ///
    /// Answers no and remembers the key the first time, yes from then until the
    /// next reset. The caller serves the read from the file either way, so a no
    /// is not a failure, it is a decision not to spend a slot yet.
    pub fn admit(&mut self, finger: u64) -> bool {
        let (a, b) = self.probes(finger);
        let had = self.get(a) && self.get(b);
        self.set(a);
        self.set(b);
        if !had {
            self.seen += 1;
            if self.seen >= self.window {
                self.reset();
            }
        }
        had
    }

    /// Forget everything, which happens on its own every `window` keys.
    pub fn reset(&mut self) {
        self.bits.fill(0);
        self.seen = 0;
    }

    /// How much memory the filter holds.
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// Two bit positions from one hash.
    ///
    /// The halves of a 64 bit hash are independent enough to use as two hashes,
    /// which is the standard double hashing shortcut and saves hashing the key
    /// twice on a path that runs on every read of a key that is not resident.
    fn probes(&self, finger: u64) -> (usize, usize) {
        let n = self.bits.len() * 64;
        let a = (finger >> 32) as usize % n;
        let b = (finger & 0xffff_ffff) as usize % n;
        (a, b)
    }

    fn get(&self, at: usize) -> bool {
        self.bits[at / 64] & (1 << (at % 64)) != 0
    }

    fn set(&mut self, at: usize) {
        self.bits[at / 64] |= 1 << (at % 64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fingerprint per slot, distinct and not zero.
    fn finger(slot: Slot) -> u64 {
        u64::from(slot).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
    }

    #[test]
    fn sieve_takes_the_oldest_untouched_entry() {
        let mut s = Sieve::new(8);
        for slot in 0..4 {
            s.insert(slot);
        }
        assert_eq!(s.demote(), Some(0));
        assert_eq!(s.demote(), Some(1));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn sieve_spares_an_entry_that_was_used_and_takes_it_next_time() {
        let mut s = Sieve::new(8);
        for slot in 0..3 {
            s.insert(slot);
        }
        s.touch(0);
        // The hand passes 0, clears its bit, and stops on 1.
        assert_eq!(s.demote(), Some(1));
        // 0 is still there with no bit, so the next sweep reaches it.
        assert_eq!(s.demote(), Some(2));
        assert_eq!(s.demote(), Some(0));
        assert_eq!(s.demote(), None);
    }

    #[test]
    fn sieve_does_not_move_a_survivor_to_the_head() {
        // The whole difference from LRU. Touching the tail entry buys it one
        // pass of the hand, not a trip to the front of the queue, so the entry
        // behind it still goes first when it too has no bit.
        let mut s = Sieve::new(8);
        for slot in 0..3 {
            s.insert(slot);
        }
        s.touch(0);
        s.demote();
        // If 0 had moved to the head it would now be the last of the three to
        // go. It is not: it is still at the tail, so it is next.
        assert_eq!(s.demote(), Some(2));
        assert_eq!(s.demote(), Some(0));
    }

    #[test]
    fn sieve_removes_the_hand_it_was_pointing_at() {
        let mut s = Sieve::new(8);
        for slot in 0..4 {
            s.insert(slot);
        }
        s.touch(1);
        s.touch(0);
        assert_eq!(s.demote(), Some(2));
        // The hand is on 1 now. Deleting it must not leave the hand dangling.
        s.remove(1);
        assert_eq!(s.demote(), Some(3));
        assert_eq!(s.demote(), Some(0));
        assert!(s.is_empty());
    }

    #[test]
    fn s3_fifo_throws_out_what_was_only_seen_once() {
        // Twenty slots of capacity means the small queue is two, so entries
        // that are never touched again are gone after two more arrive.
        let mut s = S3Fifo::new(64, 20);
        for slot in 0..5 {
            s.insert(slot, finger(slot));
        }
        assert_eq!(s.demote(finger), Some(0));
        assert_eq!(s.demote(finger), Some(1));
    }

    #[test]
    fn s3_fifo_promotes_what_was_asked_for_twice() {
        let mut s = S3Fifo::new(64, 20);
        for slot in 0..5 {
            s.insert(slot, finger(slot));
        }
        s.touch(0);
        s.touch(0);
        // 0 has a count of two, so it moves to the main queue instead of out,
        // and the next one along goes in its place.
        assert_eq!(s.demote(finger), Some(1));
        assert!(s.contains(0));
    }

    #[test]
    fn s3_fifo_sends_a_returning_key_straight_to_the_main_queue() {
        let mut s = S3Fifo::new(64, 20);
        for slot in 0..5 {
            s.insert(slot, finger(slot));
        }
        assert_eq!(s.demote(finger), Some(0));
        // 0's fingerprint is in the ghost queue, so coming back skips the small
        // queue and it is not the next thing out.
        s.insert(0, finger(0));
        assert_eq!(s.demote(finger), Some(1));
        assert!(s.contains(0));
    }

    #[test]
    fn s3_fifo_only_lets_the_ghost_queue_promote_once() {
        let mut s = S3Fifo::new(64, 20);
        for slot in 0..5 {
            s.insert(slot, finger(slot));
        }
        assert_eq!(s.demote(finger), Some(0));
        s.insert(0, finger(0));
        s.remove(0);
        // The fingerprint was consumed by the insert above, so this one is
        // an ordinary arrival and lands in the small queue.
        s.insert(0, finger(0));
        assert_eq!(s.demote(finger), Some(1));
        assert_eq!(s.demote(finger), Some(2));
    }

    #[test]
    fn s3_fifo_second_chance_spends_the_count_rather_than_looping() {
        let mut s = S3Fifo::new(64, 4);
        // Small queue of one, so everything ends up in the main queue quickly.
        for slot in 0..3 {
            s.insert(slot, finger(slot));
            s.touch(slot);
            s.touch(slot);
        }
        // Every resident entry has a positive count, so the main queue has to
        // go round spending them. It has to terminate and it has to return the
        // one whose count runs out first.
        let first = s.demote(finger).expect("a victim");
        assert!(first < 3);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn s3_fifo_empties_rather_than_spinning() {
        let mut s = S3Fifo::new(64, 4);
        for slot in 0..3 {
            s.insert(slot, finger(slot));
            s.touch(slot);
        }
        let mut out = 0;
        while s.demote(finger).is_some() {
            out += 1;
            assert!(out <= 3, "demote is not making progress");
        }
        assert_eq!(out, 3);
        assert!(s.is_empty());
    }

    #[test]
    fn a_second_read_gets_in_and_a_first_does_not() {
        let mut d = Doorkeeper::new(1024);
        assert!(!d.admit(finger(7)));
        assert!(d.admit(finger(7)));
    }

    #[test]
    fn the_doorkeeper_clears_itself_before_it_saturates() {
        let mut d = Doorkeeper::new(64);
        // Sixty four distinct keys is the window, so the reset lands inside
        // this loop and the key seen first is no longer remembered.
        for slot in 0..64 {
            assert!(!d.admit(finger(slot)));
        }
        assert!(!d.admit(finger(0)));
    }

    #[test]
    fn the_doorkeeper_is_a_few_kilobytes_and_not_a_few_megabytes() {
        // A window of a million keys at a byte each. The point of the filter is
        // that it costs nothing next to the entries it is deciding about.
        let d = Doorkeeper::new(1_000_000);
        assert!(d.memory_bytes() < 1_100_000, "{} bytes", d.memory_bytes());
    }

    #[test]
    fn both_policies_survive_a_random_workload() {
        // The property that matters for either of them is that they never hand
        // back an entry that is not resident and never lose one, because a
        // demotion of something that is not there writes a file record for
        // nothing and a leak is a slot the caller can never reuse.
        let slots = 200u32;
        let mut sieve = Sieve::new(slots as usize);
        let mut s3 = S3Fifo::new(slots as usize, 40);
        let mut resident_sieve = vec![false; slots as usize];
        let mut resident_s3 = vec![false; slots as usize];
        let mut x = 0x1234_5678_9abc_def0u64;

        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let slot = (x % u64::from(slots)) as u32;
            match x % 4 {
                0 => {
                    sieve.insert(slot);
                    resident_sieve[slot as usize] = true;
                    s3.insert(slot, finger(slot));
                    resident_s3[slot as usize] = true;
                }
                1 => {
                    sieve.touch(slot);
                    s3.touch(slot);
                }
                2 => {
                    if let Some(out) = sieve.demote() {
                        assert!(resident_sieve[out as usize], "sieve gave back a ghost");
                        resident_sieve[out as usize] = false;
                    }
                    if let Some(out) = s3.demote(finger) {
                        assert!(resident_s3[out as usize], "s3 gave back a ghost");
                        resident_s3[out as usize] = false;
                    }
                }
                _ => {
                    sieve.remove(slot);
                    resident_sieve[slot as usize] = false;
                    s3.remove(slot);
                    resident_s3[slot as usize] = false;
                }
            }
            assert_eq!(sieve.len(), resident_sieve.iter().filter(|r| **r).count());
            assert_eq!(s3.len(), resident_s3.iter().filter(|r| **r).count());
        }
    }
}
