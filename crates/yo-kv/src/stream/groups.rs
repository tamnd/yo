//! Consumer groups and the pending entries list (`08` section 7).
//!
//! A consumer group is a bookmark plus a ledger. The bookmark is one ID saying
//! how far the group has read, and the ledger is every entry the group handed
//! out and has not been told is finished with. Redis calls the ledger the PEL,
//! the pending entries list, and an entry in it a NACK.
//!
//! ```text
//! group "workers"  last 990-0  read 41823
//! +----------------------------------------------------------+
//! | PEL, in ID order                                          |
//! | 971-0 -> owner c2, handed out 3 times, last at 09:14:02   |
//! | 984-0 -> owner c1, handed out 1 time,  last at 09:14:07   |
//! | 990-0 -> owner c1, handed out 1 time,  last at 09:14:07   |
//! +----------------------------------------------------------+
//!     consumer c1 holds 984-0, 990-0
//!     consumer c2 holds 971-0
//! ```
//!
//! The point of the ledger is that a consumer can die holding work. `XPENDING`
//! finds entries nobody has touched in a while and `XCLAIM` moves them to a
//! consumer that is still alive, which is the whole reason to use a group
//! rather than plain `XREAD`.
//!
//! # A NACK is in two indexes and owned by neither
//!
//! Every pending entry has to be reachable two ways. `XPENDING` and `XAUTOCLAIM`
//! walk the group's entries in ID order, and `XINFO CONSUMERS` and consumer
//! deletion need everything one consumer holds. Redis keeps a rax per group and
//! a rax per consumer holding pointers to the same NACK, which means a claim
//! updates a pointer in two trees and the NACK belongs to whichever one frees it
//! last.
//!
//! Here the NACK lives in the group's map and the consumer holds only IDs. A
//! claim moves an ID between two [`BTreeSet`]s and rewrites one field, nothing
//! is shared and nothing has to be freed carefully. It costs one extra lookup
//! when going from a consumer's ID to its NACK, which happens on consumer
//! deletion and nowhere on a hot path.
//!
//! # Why a B-tree and not a sorted deque
//!
//! The log next door is a sorted deque because entries are appended in order and
//! trimmed from the front, and never touched in the middle. A PEL is the same
//! shape most of the time: `XREADGROUP >` appends increasing IDs and `XACK`
//! usually takes the oldest. But `XCLAIM` and a slow consumer both put holes in
//! the middle, and an ack of an arbitrary ID is a normal thing to do rather than
//! a pathology, so the middle is not the rare case here that it is in the log.
//!
//! A [`BTreeMap`] keyed by [`Id`] holds the key inline in sixteen bytes with no
//! allocation per entry and about eleven entries a node, and it gives the
//! ordered walk `XPENDING` and `XAUTOCLAIM` need. That is already well ahead of
//! a rax over sixteen byte string keys. Whether the sorted deque with tombstones
//! would beat it is a real question and the benchmark is there to answer it, but
//! it is not worth guessing at before the feature works.
//!
//! # Consumers are a vector
//!
//! A group has a handful of consumers, usually as many as there are processes,
//! and a name is looked up once per command rather than once per entry. A linear
//! scan over a vector beats a hash map at that size and brings no dependency and
//! no hashing with it. A slot is never reused while the group lives, so the
//! index a NACK holds stays valid.

use std::collections::{BTreeMap, BTreeSet};

use super::Id;

/// Which pending entries a caller wants, which is every filter `XPENDING` takes.
///
/// A struct rather than five more arguments because the command parses them as
/// a group and they travel together from the wire to here. The default is the
/// whole list, so a caller that only wants a window sets `start` and `end` and
/// leaves the rest alone.
#[derive(Debug, Clone, Copy)]
pub struct Filter {
    /// The low end of the ID window, included.
    pub start: Id,
    /// The high end, included.
    pub end: Id,
    /// At most this many, or every one in the window.
    pub count: Option<usize>,
    /// Only what this consumer is holding.
    pub owner: Option<u32>,
    /// Only what has been sitting at least this many milliseconds.
    pub min_idle: u64,
}

impl Default for Filter {
    fn default() -> Filter {
        Filter {
            start: Id::MIN,
            end: Id::MAX,
            count: None,
            owner: None,
            min_idle: 0,
        }
    }
}

/// One entry handed out and not yet acknowledged.
///
/// Redis calls this a NACK, for not acknowledged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nack {
    /// When it was last handed out, in milliseconds.
    ///
    /// Set on delivery and reset on a claim, because the point of it is how
    /// long the entry has been sitting with somebody who is not finishing it.
    time: u64,
    /// How many times it has been handed out.
    ///
    /// `XCLAIM RETRYCOUNT` sets it and `XPENDING` reports it, so a consumer can
    /// give up on a message that has killed several workers already.
    count: u64,
    /// Which consumer slot holds it.
    owner: u32,
}

impl Nack {
    /// When it was last handed out.
    #[must_use]
    #[inline]
    pub fn time(&self) -> u64 {
        self.time
    }

    /// How many times it has been handed out.
    #[must_use]
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// How long it has been sitting, which is what `min-idle-time` is compared to.
    ///
    /// Saturating, because a NACK whose time was set forward by `XCLAIM TIME` is
    /// something a caller is allowed to ask for and is not idle at all.
    #[must_use]
    #[inline]
    pub fn idle(&self, now: u64) -> u64 {
        now.saturating_sub(self.time)
    }
}

/// One consumer inside a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Consumer {
    name: Vec<u8>,
    /// When this consumer was last heard from at all.
    seen: u64,
    /// When it last read something, as opposed to asking and getting nothing.
    ///
    /// Redis separates the two because a consumer polling an empty stream is
    /// alive but idle, and telling those apart is the difference between a
    /// worker that is stuck and one that has nothing to do.
    active: u64,
    /// What it holds, in ID order.
    pending: BTreeSet<Id>,
}

impl Consumer {
    /// Its name.
    #[must_use]
    #[inline]
    pub fn name(&self) -> &[u8] {
        &self.name
    }

    /// When it was last heard from.
    #[must_use]
    #[inline]
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// When it last actually read something.
    #[must_use]
    #[inline]
    pub fn active(&self) -> u64 {
        self.active
    }

    /// How many entries it is holding.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether it is holding nothing.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// What it is holding, oldest first.
    pub fn pending(&self) -> impl Iterator<Item = Id> + '_ {
        self.pending.iter().copied()
    }
}

/// A consumer group over one stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Group {
    /// The last ID handed out, which `XREADGROUP >` reads after.
    last: Id,
    /// How many entries the group has read, for the lag.
    ///
    /// An `Option` because it is not always knowable. `XSETID` without
    /// `ENTRIESREAD` and a `SETID` to a point nobody can count from both leave
    /// it unknown, and Redis reports a null lag rather than a made up one.
    read: Option<u64>,
    /// Everything handed out and not acknowledged, in ID order.
    pending: BTreeMap<Id, Nack>,
    /// The consumers. A slot is emptied on deletion and never reused.
    consumers: Vec<Option<Consumer>>,
}

impl Group {
    /// A group reading after `last`, having read `read` entries.
    #[must_use]
    pub fn new(last: Id, read: Option<u64>) -> Group {
        Group {
            last,
            read,
            pending: BTreeMap::new(),
            consumers: Vec::new(),
        }
    }

    /// The last ID handed out.
    #[must_use]
    #[inline]
    pub fn last_id(&self) -> Id {
        self.last
    }

    /// How many entries the group has read, when that is known.
    #[must_use]
    #[inline]
    pub fn entries_read(&self) -> Option<u64> {
        self.read
    }

    /// Move the bookmark, which is `XGROUP SETID`.
    ///
    /// The PEL is left alone, because the entries in it were handed to somebody
    /// who has not finished and moving the bookmark says nothing about them.
    pub fn set_id(&mut self, last: Id, read: Option<u64>) {
        self.last = last;
        self.read = read;
    }

    /// How many entries are pending across the whole group.
    #[must_use]
    #[inline]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// The lowest and highest pending IDs, which is the `XPENDING` summary.
    #[must_use]
    pub fn pending_bounds(&self) -> Option<(Id, Id)> {
        let low = *self.pending.keys().next()?;
        let high = *self.pending.keys().next_back()?;
        Some((low, high))
    }

    /// One pending entry.
    #[must_use]
    #[inline]
    pub fn nack(&self, id: Id) -> Option<&Nack> {
        self.pending.get(&id)
    }

    /// The consumer slot for `name`, if there is one.
    #[must_use]
    pub fn slot(&self, name: &[u8]) -> Option<u32> {
        self.consumers
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| c.name == name))
            .map(|at| at as u32)
    }

    /// A consumer by slot.
    #[must_use]
    #[inline]
    pub fn consumer(&self, slot: u32) -> Option<&Consumer> {
        self.consumers.get(slot as usize)?.as_ref()
    }

    /// A consumer by name.
    #[must_use]
    pub fn consumer_named(&self, name: &[u8]) -> Option<&Consumer> {
        self.consumers
            .iter()
            .flatten()
            .find(|c| c.name.as_slice() == name)
    }

    /// Every consumer, in the order they were created.
    pub fn consumers(&self) -> impl Iterator<Item = &Consumer> + '_ {
        self.consumers.iter().flatten()
    }

    /// The slot for `name`, making the consumer if it is not there yet.
    ///
    /// This is what `XREADGROUP` does, since a consumer exists because it turned
    /// up rather than because anybody declared it.
    pub fn consumer_or_create(&mut self, name: &[u8], now: u64) -> u32 {
        if let Some(at) = self.slot(name) {
            let c = self.consumers[at as usize]
                .as_mut()
                .expect("the slot the search just found");
            c.seen = now;
            return at;
        }
        self.consumers.push(Some(Consumer {
            name: name.to_vec(),
            seen: now,
            active: now,
            pending: BTreeSet::new(),
        }));
        (self.consumers.len() - 1) as u32
    }

    /// Make a consumer and say whether it was not already there.
    ///
    /// `XGROUP CREATECONSUMER`, which answers 1 when it made one.
    pub fn create_consumer(&mut self, name: &[u8], now: u64) -> bool {
        if self.slot(name).is_some() {
            return false;
        }
        self.consumer_or_create(name, now);
        true
    }

    /// Take a consumer out and say how many entries it was holding.
    ///
    /// Those entries stop being pending at all, which is Redis's behaviour and
    /// is the point of the command: deleting a consumer is how you give up on
    /// the work it was holding when you would rather lose it than claim it.
    pub fn delete_consumer(&mut self, name: &[u8]) -> u64 {
        let Some(at) = self.slot(name) else {
            return 0;
        };
        let gone = self.consumers[at as usize]
            .take()
            .expect("the slot the search just found");
        for id in &gone.pending {
            self.pending.remove(id);
        }
        gone.pending.len() as u64
    }

    /// Mark that a consumer was heard from.
    ///
    /// `read` says whether it got anything, which is what separates seen from
    /// active.
    pub fn touch(&mut self, slot: u32, now: u64, read: bool) {
        if let Some(Some(c)) = self.consumers.get_mut(slot as usize) {
            c.seen = now;
            if read {
                c.active = now;
            }
        }
    }

    /// Hand an entry to a consumer for the first time.
    ///
    /// The bookmark moves, since this is the `>` path and the entry is new to
    /// the group. Answers false if the slot is empty, which a caller that got
    /// its slot from [`Group::consumer_or_create`] cannot hit.
    pub fn deliver(&mut self, slot: u32, id: Id, now: u64) -> bool {
        let Some(Some(c)) = self.consumers.get_mut(slot as usize) else {
            return false;
        };
        c.pending.insert(id);
        self.pending.insert(
            id,
            Nack {
                time: now,
                count: 1,
                owner: slot,
            },
        );
        if id > self.last {
            self.last = id;
        }
        true
    }

    /// Move the bookmark past an entry without writing it into the ledger.
    ///
    /// This is `XREADGROUP ... NOACK`, which is a consumer saying it does not
    /// want the work tracked. The group still counts the entry as read, because
    /// the lag is about how far behind the group is and not about how much of
    /// it is outstanding, so a NOACK reader that has caught up reports a lag of
    /// zero the same as any other.
    pub fn skip(&mut self, id: Id) {
        if id > self.last {
            self.last = id;
        }
    }

    /// Hand an entry to whoever already holds it, which is a history read.
    ///
    /// `XREADGROUP` with an ID rather than `>` is a consumer asking for what it
    /// was already given, and Redis treats that as a real delivery: the time is
    /// reset and the count goes up, exactly as if the entry had been handed out
    /// again. Checked against Redis 8.10.1, where a history read of an entry
    /// idle for 2006 milliseconds left it idle for 2 with its count up by one.
    ///
    /// It reads as surprising until you think about what the count is for. It
    /// counts how many times a consumer has been told to do this work, and a
    /// consumer re-reading its backlog after a restart has been told again.
    pub fn redeliver(&mut self, id: Id, now: u64) -> bool {
        let Some(nack) = self.pending.get_mut(&id) else {
            return false;
        };
        nack.time = now;
        nack.count += 1;
        true
    }

    /// Put the read counter where the stream has worked out it belongs.
    ///
    /// The counter is a fact about the stream and not about the group, since
    /// what a delivery does to it depends on whether anything has been deleted
    /// ahead of the group. [`crate::stream::Stream::read_group`] is the one
    /// caller, and it is the one that can see both.
    pub fn set_read(&mut self, read: Option<u64>) {
        self.read = read;
    }

    /// Finish with an entry, which is `XACK`.
    ///
    /// Answers whether it was pending. Acknowledging something twice is not an
    /// error, it just does nothing the second time, because a consumer that
    /// crashed between doing the work and sending the ack will send it again.
    pub fn ack(&mut self, id: Id) -> bool {
        let Some(nack) = self.pending.remove(&id) else {
            return false;
        };
        if let Some(Some(c)) = self.consumers.get_mut(nack.owner as usize) {
            c.pending.remove(&id);
        }
        true
    }

    /// Drop a pending entry without it having been acknowledged.
    ///
    /// What happens to a NACK whose entry is no longer in the stream. `XCLAIM`
    /// and `XAUTOCLAIM` both clear those out as they find them, because a
    /// pending entry nobody can ever read is work no consumer can ever finish.
    pub fn forget(&mut self, id: Id) -> bool {
        self.ack(id)
    }

    /// Move an entry to another consumer, which is the middle of `XCLAIM`.
    ///
    /// `time` is when it should count as having been handed out, which is now
    /// for a plain claim and something a caller chose for `IDLE` or `TIME`.
    /// `count` replaces the delivery count when it is given, which is
    /// `RETRYCOUNT`, and otherwise the count goes up by one unless `bump` says
    /// not to, which is `JUSTID`.
    ///
    /// Answers false when the entry was not pending or the slot is empty.
    pub fn claim(&mut self, id: Id, slot: u32, time: u64, count: Option<u64>, bump: bool) -> bool {
        if !matches!(self.consumers.get(slot as usize), Some(Some(_))) {
            return false;
        }
        let Some(nack) = self.pending.get_mut(&id) else {
            return false;
        };
        let was = nack.owner;
        nack.owner = slot;
        nack.time = time;
        if let Some(n) = count {
            nack.count = n;
        } else if bump {
            nack.count += 1;
        }
        if was != slot {
            if let Some(Some(c)) = self.consumers.get_mut(was as usize) {
                c.pending.remove(&id);
            }
            if let Some(Some(c)) = self.consumers.get_mut(slot as usize) {
                c.pending.insert(id);
            }
        }
        true
    }

    /// Make a pending entry that was not pending, which is `XCLAIM FORCE`.
    ///
    /// The caller has to have checked that the entry is really in the stream,
    /// because this cannot see the stream and creating a NACK for an entry that
    /// is not there is exactly the state [`Group::forget`] exists to clean up.
    pub fn force(&mut self, id: Id, slot: u32, time: u64, count: u64) -> bool {
        let Some(Some(c)) = self.consumers.get_mut(slot as usize) else {
            return false;
        };
        c.pending.insert(id);
        self.pending.insert(
            id,
            Nack {
                time,
                count,
                owner: slot,
            },
        );
        true
    }

    /// Pending entries in `want`, oldest first.
    ///
    /// The callback answers whether to carry on.
    pub fn pending_range<F>(&self, want: Filter, now: u64, mut f: F) -> usize
    where
        F: FnMut(Id, &Nack, &Consumer) -> bool,
    {
        let mut seen = 0;
        for (&id, nack) in self.pending.range(want.start..=want.end) {
            if want.count.is_some_and(|n| seen >= n) {
                break;
            }
            if want.owner.is_some_and(|c| c != nack.owner) {
                continue;
            }
            if nack.idle(now) < want.min_idle {
                continue;
            }
            let Some(Some(c)) = self.consumers.get(nack.owner as usize) else {
                continue;
            };
            seen += 1;
            if !f(id, nack, c) {
                break;
            }
        }
        seen
    }

    /// How many entries each consumer is holding, for the `XPENDING` summary.
    pub fn pending_counts(&self) -> impl Iterator<Item = (&[u8], usize)> + '_ {
        self.consumers
            .iter()
            .flatten()
            .filter(|c| !c.pending.is_empty())
            .map(|c| (c.name.as_slice(), c.pending.len()))
    }

    /// The IDs an `XAUTOCLAIM` would take, from `start` and idle at least
    /// `min_idle`, and where a following call should carry on from.
    ///
    /// Only the scan, because deciding what to do with each one needs the
    /// stream and this does not have it. The cursor is `None` when the scan
    /// reached the end, which is the `0-0` Redis answers with.
    #[must_use]
    pub fn claimable(
        &self,
        start: Id,
        min_idle: u64,
        now: u64,
        limit: usize,
        out: &mut Vec<Id>,
    ) -> Option<Id> {
        // Redis charges attempts rather than hits, so a scan over a PEL full of
        // entries that are not idle enough still ends and hands back a cursor
        // instead of walking a million NACKs inside one command.
        for (tried, (&id, nack)) in self.pending.range(start..).enumerate() {
            if out.len() >= limit || tried >= limit * 10 {
                return Some(id);
            }
            if nack.idle(now) >= min_idle {
                out.push(id);
            }
        }
        None
    }

    /// How many bytes this group takes, not counting the struct itself.
    ///
    /// The pending map is counted per entry at the size of a key and a value
    /// plus a share of the node around them, rather than exactly, because a
    /// [`BTreeMap`] does not say how many nodes it has and the answer is only
    /// ever read by `MEMORY USAGE` and the eviction total. A B-tree node here
    /// holds eleven entries and some overhead, and a sixteenth of an entry is
    /// close enough for both.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let each = std::mem::size_of::<(Id, Nack)>();
        let pending = self.pending.len() * (each + each / 16);
        let consumers: usize = self
            .consumers
            .iter()
            .map(|slot| {
                std::mem::size_of::<Option<Consumer>>()
                    + slot.as_ref().map_or(0, |c| {
                        let each = std::mem::size_of::<Id>();
                        c.name.capacity() + c.pending.len() * (each + each / 16)
                    })
            })
            .sum();
        pending + consumers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> Group {
        Group::new(Id::MIN, Some(0))
    }

    #[test]
    fn a_consumer_appears_by_turning_up() {
        let mut g = group();
        assert_eq!(g.slot(b"alice"), None);
        let at = g.consumer_or_create(b"alice", 100);
        assert_eq!(g.slot(b"alice"), Some(at));
        assert_eq!(g.consumer_or_create(b"alice", 200), at);
        assert_eq!(g.consumers().count(), 1);
        assert_eq!(g.consumer(at).expect("alice").seen(), 200);
    }

    #[test]
    fn creating_a_consumer_twice_says_so() {
        let mut g = group();
        assert!(g.create_consumer(b"alice", 1));
        assert!(!g.create_consumer(b"alice", 2));
    }

    #[test]
    fn delivering_moves_the_bookmark_and_fills_both_indexes() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 10);
        assert!(g.deliver(a, Id::new(5, 0), 10));
        assert!(g.deliver(a, Id::new(7, 0), 12));

        assert_eq!(g.last_id(), Id::new(7, 0));
        assert_eq!(g.pending_len(), 2);
        assert_eq!(
            g.consumer(a).expect("alice").pending().collect::<Vec<_>>(),
            vec![Id::new(5, 0), Id::new(7, 0)]
        );
        let nack = g.nack(Id::new(5, 0)).expect("a nack");
        assert_eq!((nack.count(), nack.time()), (1, 10));
    }

    #[test]
    fn acking_takes_it_out_of_both_indexes() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 1);
        assert!(g.ack(Id::new(5, 0)));
        assert_eq!(g.pending_len(), 0);
        assert!(g.consumer(a).expect("alice").is_empty());
        // Twice is not an error, because a consumer that crashed after doing the
        // work and before sending the ack will send it again.
        assert!(!g.ack(Id::new(5, 0)));
    }

    #[test]
    fn acking_does_not_move_the_bookmark() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 1);
        g.ack(Id::new(5, 0));
        assert_eq!(g.last_id(), Id::new(5, 0));
    }

    #[test]
    fn a_claim_moves_it_between_consumers() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        let b = g.consumer_or_create(b"bob", 1);
        g.deliver(a, Id::new(5, 0), 100);

        assert!(g.claim(Id::new(5, 0), b, 500, None, true));
        assert!(g.consumer(a).expect("alice").is_empty());
        assert_eq!(
            g.consumer(b).expect("bob").pending().collect::<Vec<_>>(),
            vec![Id::new(5, 0)]
        );
        let nack = g.nack(Id::new(5, 0)).expect("a nack");
        assert_eq!((nack.count(), nack.time()), (2, 500));
    }

    #[test]
    fn a_claim_that_does_not_bump_is_justid() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        let b = g.consumer_or_create(b"bob", 1);
        g.deliver(a, Id::new(5, 0), 100);
        g.claim(Id::new(5, 0), b, 500, None, false);
        assert_eq!(g.nack(Id::new(5, 0)).expect("a nack").count(), 1);
    }

    #[test]
    fn a_retry_count_replaces_rather_than_adds() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 100);
        g.claim(Id::new(5, 0), a, 500, Some(9), true);
        assert_eq!(g.nack(Id::new(5, 0)).expect("a nack").count(), 9);
    }

    #[test]
    fn claiming_back_to_the_same_consumer_keeps_it_there() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 100);
        assert!(g.claim(Id::new(5, 0), a, 500, None, true));
        assert_eq!(
            g.consumer(a).expect("alice").pending().collect::<Vec<_>>(),
            vec![Id::new(5, 0)]
        );
    }

    #[test]
    fn nothing_pending_cannot_be_claimed_without_force() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        assert!(!g.claim(Id::new(5, 0), a, 500, None, true));
        assert!(g.force(Id::new(5, 0), a, 500, 1));
        assert_eq!(g.pending_len(), 1);
    }

    #[test]
    fn deleting_a_consumer_gives_up_its_work() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        let b = g.consumer_or_create(b"bob", 1);
        g.deliver(a, Id::new(5, 0), 1);
        g.deliver(a, Id::new(6, 0), 1);
        g.deliver(b, Id::new(7, 0), 1);

        assert_eq!(g.delete_consumer(b"alice"), 2);
        assert_eq!(g.pending_len(), 1);
        assert!(g.nack(Id::new(5, 0)).is_none());
        assert!(g.nack(Id::new(7, 0)).is_some());
        assert_eq!(g.delete_consumer(b"alice"), 0);
        // The bookmark is untouched, so the entries are not handed out again.
        assert_eq!(g.last_id(), Id::new(7, 0));
    }

    #[test]
    fn a_deleted_slot_is_not_reused() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.delete_consumer(b"alice");
        let b = g.consumer_or_create(b"bob", 1);
        assert_ne!(a, b);
        assert_eq!(g.consumers().count(), 1);
    }

    #[test]
    fn idle_is_measured_from_the_last_hand_out() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 1_000);
        assert_eq!(g.nack(Id::new(5, 0)).expect("a nack").idle(4_000), 3_000);
        // A time set into the future is something XCLAIM TIME allows, and it is
        // not idle rather than idle by a negative amount.
        assert_eq!(g.nack(Id::new(5, 0)).expect("a nack").idle(500), 0);
    }

    #[test]
    fn the_summary_is_the_two_ends_and_the_counts() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        let b = g.consumer_or_create(b"bob", 1);
        for ms in [3u64, 5, 9] {
            g.deliver(a, Id::new(ms, 0), 1);
        }
        g.deliver(b, Id::new(11, 0), 1);

        assert_eq!(g.pending_bounds(), Some((Id::new(3, 0), Id::new(11, 0))));
        let counts: Vec<_> = g
            .pending_counts()
            .map(|(n, c)| (String::from_utf8_lossy(n).into_owned(), c))
            .collect();
        assert_eq!(counts, vec![("alice".into(), 3), ("bob".into(), 1)]);
    }

    #[test]
    fn a_consumer_with_nothing_is_left_out_of_the_summary() {
        let mut g = group();
        g.consumer_or_create(b"alice", 1);
        assert_eq!(g.pending_counts().count(), 0);
        assert_eq!(g.pending_bounds(), None);
    }

    #[test]
    fn the_pending_range_takes_both_ends_and_the_filters() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        let b = g.consumer_or_create(b"bob", 1);
        g.deliver(a, Id::new(3, 0), 100);
        g.deliver(b, Id::new(5, 0), 100);
        g.deliver(a, Id::new(9, 0), 900);

        let seen = |g: &Group, start, end, owner, idle| {
            let mut out = Vec::new();
            let want = Filter {
                start,
                end,
                owner,
                min_idle: idle,
                ..Filter::default()
            };
            g.pending_range(want, 1_000, |id, _, c| {
                out.push((id, String::from_utf8_lossy(c.name()).into_owned()));
                true
            });
            out
        };

        assert_eq!(seen(&g, Id::MIN, Id::MAX, None, 0).len(), 3);
        assert_eq!(seen(&g, Id::new(4, 0), Id::new(9, 0), None, 0).len(), 2);
        assert_eq!(
            seen(&g, Id::MIN, Id::MAX, Some(a), 0),
            vec![
                (Id::new(3, 0), "alice".into()),
                (Id::new(9, 0), "alice".into())
            ]
        );
        // Only the two handed out at 100 have been sitting 500 milliseconds.
        assert_eq!(seen(&g, Id::MIN, Id::MAX, None, 500).len(), 2);
    }

    #[test]
    fn a_count_stops_the_pending_range() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        for ms in 1..=10u64 {
            g.deliver(a, Id::new(ms, 0), 1);
        }
        let mut out = Vec::new();
        let want = Filter {
            count: Some(4),
            ..Filter::default()
        };
        let seen = g.pending_range(want, 1, |id, _, _| {
            out.push(id);
            true
        });
        assert_eq!((seen, out.len()), (4, 4));
    }

    #[test]
    fn the_callback_can_stop_the_pending_range() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        for ms in 1..=10u64 {
            g.deliver(a, Id::new(ms, 0), 1);
        }
        let mut out = Vec::new();
        g.pending_range(Filter::default(), 1, |id, _, _| {
            out.push(id);
            out.len() < 3
        });
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn claimable_takes_the_idle_ones_and_says_where_to_carry_on() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        for ms in 1..=10u64 {
            g.deliver(a, Id::new(ms, 0), if ms <= 5 { 100 } else { 900 });
        }
        let mut out = Vec::new();
        let cursor = g.claimable(Id::MIN, 500, 1_000, 100, &mut out);
        assert_eq!(cursor, None, "the scan reached the end");
        assert_eq!(out, (1..=5).map(|ms| Id::new(ms, 0)).collect::<Vec<_>>());

        // A limit hands back where the next call starts.
        out.clear();
        let cursor = g.claimable(Id::MIN, 500, 1_000, 3, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(cursor, Some(Id::new(4, 0)));
    }

    /// The read counter is set from outside and a delivery does not touch it.
    ///
    /// It looks like something the group should keep for itself, and it is not:
    /// what a delivery does to it depends on whether anything has been deleted
    /// ahead of the entry being handed over, which is a fact about the stream.
    /// The rule lives in [`crate::stream::Stream::read_group`] and this only
    /// holds the number.
    #[test]
    fn a_delivery_leaves_the_read_counter_to_the_stream() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(1, 0), 1);
        assert_eq!(g.entries_read(), Some(0), "the group did not count it");

        g.set_read(Some(1));
        assert_eq!(g.entries_read(), Some(1));
        g.set_read(None);
        assert_eq!(g.entries_read(), None, "and it can be given up on");
    }

    #[test]
    fn setting_the_id_leaves_the_pending_list_alone() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 1);
        g.set_id(Id::MIN, Some(0));
        assert_eq!(g.last_id(), Id::MIN);
        assert_eq!(g.pending_len(), 1, "somebody is still holding it");
    }
}
