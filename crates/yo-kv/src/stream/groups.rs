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

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::{Cursor, Id};
use crate::frozen::{self, Broken};

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
    /// Which consumer slot holds it, or [`Nack::NOBODY`].
    owner: u32,
}

impl Nack {
    /// The slot of an entry that is pending and that nobody holds.
    ///
    /// `XNACK` hands work back to the group without giving it to anybody, so the
    /// pending list has to be able to hold an entry with no consumer against it.
    /// Redis reports one as an empty consumer name and an idle time of minus
    /// one, and treats it as idle for longer than any `min-idle-time` a claim can
    /// name, which is what makes it the next thing `XAUTOCLAIM` picks up.
    const NOBODY: u32 = u32::MAX;

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

    /// Which consumer holds it, or `None` for one that has been released.
    #[must_use]
    #[inline]
    pub fn owner(&self) -> Option<u32> {
        (self.owner != Nack::NOBODY).then_some(self.owner)
    }

    /// How long it has been sitting, which is what `min-idle-time` is compared to.
    ///
    /// Saturating, because a NACK whose time was set forward by `XCLAIM TIME` is
    /// something a caller is allowed to ask for and is not idle at all. An entry
    /// nobody holds has been idle for as long as there is, so that every claim
    /// and every `XPENDING IDLE` filter picks it up whatever they asked for.
    #[must_use]
    #[inline]
    pub fn idle(&self, now: u64) -> u64 {
        if self.owner == Nack::NOBODY {
            return u64::MAX;
        }
        now.saturating_sub(self.time)
    }
}

/// What releasing an entry does to its delivery count.
///
/// `XNACK` takes one of three words for this and they only differ here. A worker
/// that could not do the job because the machine it was on went away wants the
/// attempt not to count, one that failed the way work sometimes fails wants the
/// count left alone, and one that has decided the message itself is the problem
/// wants nobody to try it again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retry {
    /// `SILENT`: take one off the count, as if the delivery had not happened.
    Down,
    /// `FAIL`: leave the count where it is, and start a new entry at zero.
    Keep,
    /// `FATAL`: put the count as high as it goes.
    Max,
    /// `RETRYCOUNT n`: put the count at exactly this, whatever the word said.
    At(u64),
}

impl Retry {
    /// The count an entry that was on `had` ends up with.
    ///
    /// [`Retry::Down`] takes one off rather than putting the count back to zero,
    /// which is worth saying because the two look the same on an entry that has
    /// only been handed out once and that is the entry most people try it on. A
    /// message that has killed four workers and is then released by a fifth for
    /// a reason that was nothing to do with the message reads as three, not as
    /// new. It saturates, so a released entry that is released again stays at
    /// zero rather than wrapping.
    ///
    /// [`Retry::Max`] is [`i64::MAX`] and not [`u64::MAX`] because that is the
    /// number Redis reports, and a client that reads the count into a signed
    /// integer, which is what the protocol hands it, has to be able to hold it.
    #[must_use]
    pub fn applied(self, had: u64) -> u64 {
        match self {
            Retry::Down => had.saturating_sub(1),
            Retry::Keep => had,
            Retry::Max => i64::MAX as u64,
            Retry::At(n) => n,
        }
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
    ///
    /// `None` for a consumer that has never had anything, which is what
    /// `XGROUP CREATECONSUMER` makes and what an `XREADGROUP` that found nothing
    /// leaves behind. Redis reports that as an active time of minus one rather
    /// than as the moment the consumer turned up, and `XINFO CONSUMERS` passes
    /// it through to `inactive`, so a fresh consumer reads as never active and
    /// not as active a moment ago.
    active: Option<u64>,
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

    /// When it last actually read something, or `None` if it never has.
    #[must_use]
    #[inline]
    pub fn active(&self) -> Option<u64> {
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
#[derive(Debug, Clone, Default)]
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
    /// How many of the pending entries nobody holds.
    ///
    /// Kept rather than counted because `XINFO STREAM FULL` reports it and that
    /// command takes a `COUNT` precisely so that it never walks a long pending
    /// list. Every line that moves a NACK on or off [`Nack::NOBODY`] is in this
    /// file and adjusts this, and a test at the bottom checks the number against
    /// a full scan after a run of mixed operations.
    nacked: usize,
    /// Where the last read of this group stopped inside a node's blob.
    ///
    /// A note about the shape of the stream and not about the group, dropped by
    /// anything that moves the bytes it counted past and rebuilt by the next
    /// read. See [`Cursor`].
    resume: Option<Cursor>,
}

/// Two groups are the same when they hold the same entries for the same
/// consumers at the same place. The resume cursor is not part of that. It is a
/// note about where a walk got to in a blob, a freeze and thaw drops it, and a
/// group that has read something is not a different group from the same group
/// before it did.
impl PartialEq for Group {
    fn eq(&self, other: &Group) -> bool {
        self.last == other.last
            && self.read == other.read
            && self.pending == other.pending
            && self.consumers == other.consumers
            && self.nacked == other.nacked
    }
}

impl Eq for Group {}

impl Group {
    /// A group reading after `last`, having read `read` entries.
    #[must_use]
    pub fn new(last: Id, read: Option<u64>) -> Group {
        Group {
            last,
            read,
            pending: BTreeMap::new(),
            consumers: Vec::new(),
            nacked: 0,
            resume: None,
        }
    }

    /// The node and byte offset the last read stopped at, when it is still good.
    ///
    /// Good means two things. The stream has not moved any bytes since, which is
    /// what the epoch says, and this read is asking for exactly the ID the read
    /// that left the mark would be asked for next. The second is what makes an
    /// `XGROUP SETID` back to an older ID safe without the group having to know
    /// about it: the bookmark is somewhere else, so the mark does not match and
    /// the walk starts from the front.
    #[must_use]
    pub(crate) fn resume(&self, epoch: u64, from: Id) -> Option<(Id, usize)> {
        let c = self.resume?;
        (c.epoch == epoch && c.next == from).then_some((c.master, c.byte))
    }

    /// Keep where the read that just ran stopped, or forget the old mark.
    pub(crate) fn set_resume(&mut self, at: Option<Cursor>) {
        self.resume = at;
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

    /// How many of those nobody is holding, which `XNACK` is what makes nonzero.
    #[must_use]
    #[inline]
    pub fn nacked_len(&self) -> usize {
        self.nacked
    }

    /// Every pending entry with what is known about it, oldest first.
    ///
    /// The whole ledger and no filter, which is what an RDB payload carries and
    /// what nothing on the wire ever asks for, since `XPENDING` always has a
    /// range and usually a count.
    pub fn pending_all(&self) -> impl Iterator<Item = (Id, &Nack)> + '_ {
        self.pending.iter().map(|(&id, nack)| (id, nack))
    }

    /// Put a pending entry on a group being built from a payload, unowned.
    ///
    /// An RDB writes the group's whole pending list first and its consumers
    /// after it, so there is nobody to hand the entry to at the point it
    /// arrives. [`Group::restore_owner`] gives it an owner when the consumer
    /// holding it turns up, and an entry no consumer claims stays unowned. That
    /// is not a hole in the format: Redis loads the same payload into a NACK
    /// with a null consumer and leaves it there, so both servers end up with the
    /// same released entry.
    pub(crate) fn restore_nack(&mut self, id: Id, time: u64, count: u64) -> bool {
        if self
            .pending
            .keys()
            .next_back()
            .is_some_and(|&had| had >= id)
        {
            return false;
        }
        self.pending.insert(
            id,
            Nack {
                time,
                count,
                owner: Nack::NOBODY,
            },
        );
        self.nacked += 1;
        true
    }

    /// Make a consumer on a group being built from a payload, times and all.
    ///
    /// Not [`Group::create_consumer`], because that one sets both times to now
    /// and a restored consumer has times of its own that a client can see.
    pub(crate) fn restore_consumer(
        &mut self,
        name: &[u8],
        seen: u64,
        active: Option<u64>,
    ) -> Option<u32> {
        if self.slot(name).is_some() {
            return None;
        }
        self.consumers.push(Some(Consumer {
            name: name.to_vec(),
            seen,
            active,
            pending: BTreeSet::new(),
        }));
        Some((self.consumers.len() - 1) as u32)
    }

    /// Hand a restored pending entry to the consumer that was holding it.
    ///
    /// Refuses an entry that is not pending or that somebody already holds,
    /// since a payload naming the same entry under two consumers would leave the
    /// second consumer holding one the ledger says belongs to the first.
    pub(crate) fn restore_owner(&mut self, id: Id, slot: u32) -> bool {
        if !matches!(self.pending.get(&id), Some(nack) if nack.owner == Nack::NOBODY) {
            return false;
        }
        let Some(Some(c)) = self.consumers.get_mut(slot as usize) else {
            return false;
        };
        c.pending.insert(id);
        self.pending
            .get_mut(&id)
            .expect("the entry the check above just found")
            .owner = slot;
        self.nacked -= 1;
        true
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
            active: None,
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
                c.active = Some(now);
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
        match self.consumers.get_mut(nack.owner as usize) {
            Some(Some(c)) => {
                c.pending.remove(&id);
            }
            // Either the slot was emptied under it, which cannot happen because
            // deleting a consumer takes its entries with it, or nobody held it.
            _ => self.nacked -= usize::from(nack.owner == Nack::NOBODY),
        }
        true
    }

    /// Hand an entry back to the group without acknowledging it, which is `XNACK`.
    ///
    /// The entry stays pending and stops belonging to anybody, so it reads as
    /// idle for longer than any claim can ask for and the next `XAUTOCLAIM`
    /// takes it. `retry` is what the delivery count becomes, which is the whole
    /// difference between the three words `XNACK` takes.
    ///
    /// Answers whether it was pending. The bookmark does not move, so a `>` read
    /// will not hand it out again: releasing an entry offers it to a claim and
    /// not to the group's next reader, which is Redis's behaviour and the only
    /// one that keeps a released entry from being delivered twice over.
    pub fn release(&mut self, id: Id, retry: Retry) -> bool {
        let Some(nack) = self.pending.get_mut(&id) else {
            return false;
        };
        let was = std::mem::replace(&mut nack.owner, Nack::NOBODY);
        nack.count = retry.applied(nack.count);
        // Zero rather than now, because the delivery time of an entry nobody
        // holds is never read as a time: `XPENDING` reports minus one for it and
        // `XINFO STREAM FULL` reports the zero.
        nack.time = 0;
        if was == Nack::NOBODY {
            return true;
        }
        self.nacked += 1;
        if let Some(Some(c)) = self.consumers.get_mut(was as usize) {
            c.pending.remove(&id);
        }
        true
    }

    /// Make a released entry out of one that was not pending, which is
    /// `XNACK ... FORCE`.
    ///
    /// The caller has to have checked that the entry is really in the stream,
    /// for the same reason [`Group::force`] does. A count of zero is where a
    /// released entry that has never been delivered starts, whatever word was
    /// used, because there is no earlier count for `FAIL` to keep.
    pub fn force_release(&mut self, id: Id, retry: Retry) {
        if self.release(id, retry) {
            return;
        }
        self.pending.insert(
            id,
            Nack {
                time: 0,
                count: retry.applied(0),
                owner: Nack::NOBODY,
            },
        );
        self.nacked += 1;
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
            if was == Nack::NOBODY {
                // A claim is how a released entry gets an owner again, and it is
                // the only way, since a `>` read never looks below the bookmark.
                self.nacked -= 1;
            } else if let Some(Some(c)) = self.consumers.get_mut(was as usize) {
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
    /// The callback answers whether to carry on, and gets `None` for the owner
    /// of an entry that has been released, which `XPENDING` writes as an empty
    /// name. A consumer filter never matches one of those, since asking what a
    /// named consumer is holding is asking about entries that have an owner.
    pub fn pending_range<F>(&self, want: Filter, now: u64, mut f: F) -> usize
    where
        F: FnMut(Id, &Nack, Option<&Consumer>) -> bool,
    {
        let mut seen = 0;
        for (&id, nack) in self.pending.range(want.start..=want.end) {
            if want.count.is_some_and(|n| seen >= n) {
                break;
            }
            if want.owner.is_some_and(|c| Some(c) != nack.owner()) {
                continue;
            }
            if nack.idle(now) < want.min_idle {
                continue;
            }
            let who = match nack.owner() {
                Some(slot) => match self.consumers.get(slot as usize) {
                    Some(Some(c)) => Some(c),
                    // A slot that has been emptied under a NACK, which deleting
                    // a consumer cannot leave behind and nothing else can make.
                    _ => continue,
                },
                None => None,
            };
            seen += 1;
            if !f(id, nack, who) {
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

    /// Write the group out as bytes, for [`super::Stream::freeze`].
    ///
    /// The consumer slots go out including the empty ones, because a slot is
    /// emptied on deletion and never reused and a NACK names its owner by slot
    /// number. Renumbering them on the way through would hand every pending
    /// entry to the wrong consumer.
    ///
    /// What each consumer is holding is not written. It is a partition of the
    /// pending list by owner, so it is rebuilt from the pending list on the way
    /// back in and the two cannot come back disagreeing.
    pub(super) fn freeze(&self, out: &mut Vec<u8>) {
        frozen::put_uint(out, self.last.ms);
        frozen::put_uint(out, self.last.seq);
        put_opt(out, self.read);

        frozen::put_uint(out, self.consumers.len() as u64);
        for slot in &self.consumers {
            match slot {
                None => out.push(0),
                Some(c) => {
                    out.push(1);
                    frozen::put_bytes(out, &c.name);
                    frozen::put_uint(out, c.seen);
                    put_opt(out, c.active);
                }
            }
        }

        frozen::put_uint(out, self.pending.len() as u64);
        for (id, nack) in &self.pending {
            frozen::put_uint(out, id.ms);
            frozen::put_uint(out, id.seq);
            frozen::put_uint(out, nack.time);
            frozen::put_uint(out, nack.count);
            // The owner goes out as itself rather than as an index with a spare
            // value for nobody, because [`Nack::NOBODY`] already is one.
            frozen::put_uint(out, u64::from(nack.owner));
        }
    }

    /// Read back a group [`Group::freeze`] wrote.
    pub(super) fn thaw(cut: &mut frozen::Cut<'_>) -> Result<Group, Broken> {
        let last = Id::new(cut.uint()?, cut.uint()?);
        let read = take_opt(cut)?;

        let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
        // A slot is a byte at the very least, so a count past what is left is a
        // short body and not a reason to reserve that many.
        if n > cut.rest().len() {
            return Err(Broken::Short);
        }
        let mut consumers: Vec<Option<Consumer>> = Vec::with_capacity(n);
        let mut names = HashSet::with_capacity(n);
        for _ in 0..n {
            match cut.byte()? {
                0 => consumers.push(None),
                1 => {
                    let name = cut.bytes()?;
                    // Two consumers under one name would make every lookup find
                    // the first and leave the second unreachable, holding
                    // entries nothing can claim back.
                    if !names.insert(name) {
                        return Err(Broken::Body);
                    }
                    consumers.push(Some(Consumer {
                        name: name.to_vec(),
                        seen: cut.uint()?,
                        active: take_opt(cut)?,
                        pending: BTreeSet::new(),
                    }));
                }
                _ => return Err(Broken::Body),
            }
        }

        let n = usize::try_from(cut.uint()?).map_err(|_| Broken::Short)?;
        // Five numbers each, so one byte apiece is already generous.
        if n > cut.rest().len() {
            return Err(Broken::Short);
        }
        let mut group = Group {
            last,
            read,
            pending: BTreeMap::new(),
            consumers,
            nacked: 0,
            resume: None,
        };
        let mut prev = None;
        for _ in 0..n {
            let id = Id::new(cut.uint()?, cut.uint()?);
            // Written in ID order out of a map, so anything else is bytes that
            // did not come from `freeze`, and a repeat would silently drop an
            // entry somebody is holding.
            if prev.is_some_and(|p| p >= id) {
                return Err(Broken::Body);
            }
            prev = Some(id);
            let nack = Nack {
                time: cut.uint()?,
                count: cut.uint()?,
                owner: u32::try_from(cut.uint()?).map_err(|_| Broken::Body)?,
            };
            if nack.owner == Nack::NOBODY {
                group.nacked += 1;
            } else {
                match group.consumers.get_mut(nack.owner as usize) {
                    Some(Some(c)) => {
                        c.pending.insert(id);
                    }
                    // An owner that is off the end or an emptied slot would be
                    // an entry held by a consumer that cannot be named, so
                    // neither `XPENDING` nor a claim would ever reach it.
                    _ => return Err(Broken::Body),
                }
            }
            group.pending.insert(id, nack);
        }
        Ok(group)
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

/// Append a count that may not be known, as a flag byte and then the number.
///
/// A byte rather than the usual trick of writing one more than the number and
/// keeping zero for nothing, because both of the counts this is used for are a
/// `u64` and adding one to the top of the range wraps. The flag costs a byte
/// and is right everywhere.
fn put_opt(out: &mut Vec<u8>, v: Option<u64>) {
    match v {
        None => out.push(0),
        Some(n) => {
            out.push(1);
            frozen::put_uint(out, n);
        }
    }
}

/// Read back what [`put_opt`] wrote.
fn take_opt(cut: &mut frozen::Cut<'_>) -> Result<Option<u64>, Broken> {
    match cut.byte()? {
        0 => Ok(None),
        1 => Ok(Some(cut.uint()?)),
        _ => Err(Broken::Body),
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
                out.push((
                    id,
                    String::from_utf8_lossy(c.expect("an owner").name()).into_owned(),
                ));
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

    /// A released entry is pending, owned by nobody, and idle for ever.
    #[test]
    fn releasing_takes_the_entry_out_of_the_consumers_hands() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(5, 0), 100);

        assert!(g.release(Id::new(5, 0), Retry::Keep));
        assert_eq!(g.pending_len(), 1, "it is still the group's problem");
        assert_eq!(
            g.consumer(a).expect("alice").pending().count(),
            0,
            "and no longer alice's"
        );
        let nack = g.nack(Id::new(5, 0)).expect("a nack");
        assert_eq!(nack.owner(), None);
        assert_eq!(nack.count(), 1, "Keep left the count where it was");
        // Idle for longer than any min-idle-time a claim can name, which is what
        // puts it at the front of the next sweep.
        assert_eq!(nack.idle(100), u64::MAX);
        let mut out = Vec::new();
        assert_eq!(g.claimable(Id::MIN, u64::MAX, 100, 10, &mut out), None);
        assert_eq!(out, vec![Id::new(5, 0)]);

        // Releasing it again is still true and does not count it twice.
        assert!(g.release(Id::new(5, 0), Retry::Keep));
        assert_eq!(g.nacked_len(), 1);
        // And nothing pending is false, however it is asked.
        assert!(!g.release(Id::new(9, 0), Retry::Keep));
    }

    #[test]
    fn the_three_words_differ_only_in_the_delivery_count() {
        let count = |retry| {
            let mut g = group();
            let a = g.consumer_or_create(b"alice", 1);
            g.deliver(a, Id::new(5, 0), 1);
            g.claim(Id::new(5, 0), a, 2, None, true);
            g.release(Id::new(5, 0), retry);
            g.nack(Id::new(5, 0)).expect("a nack").count()
        };
        assert_eq!(count(Retry::Down), 1, "one off, not back to nothing");
        assert_eq!(count(Retry::Keep), 2);
        assert_eq!(count(Retry::Max), i64::MAX as u64);
        assert_eq!(count(Retry::At(7)), 7);
    }

    /// Forcing makes the pending entry when there is not one, and does not make
    /// a second one when there is.
    #[test]
    fn forcing_a_release_is_the_same_call_twice() {
        let mut g = group();
        g.force_release(Id::new(5, 0), Retry::Keep);
        assert_eq!(g.pending_len(), 1);
        assert_eq!(
            g.nack(Id::new(5, 0)).expect("a nack").count(),
            0,
            "there was no earlier count to keep"
        );

        g.force_release(Id::new(5, 0), Retry::At(4));
        assert_eq!(g.pending_len(), 1);
        assert_eq!(g.nacked_len(), 1);
        assert_eq!(g.nack(Id::new(5, 0)).expect("a nack").count(), 4);
    }

    /// The counter behind `XINFO STREAM FULL`'s `nacked-count`, which is a field
    /// and not a walk, so every line that moves an entry on or off `NOBODY` has
    /// to keep it right. This is the walk, run against the field.
    #[test]
    fn the_nacked_count_matches_a_full_scan() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        let b = g.consumer_or_create(b"bob", 1);
        for ms in 1..=6u64 {
            g.deliver(a, Id::new(ms, 0), 100);
        }

        let scan = |g: &Group| {
            (1..=9u64)
                .filter(|&ms| g.nack(Id::new(ms, 0)).is_some_and(|n| n.owner().is_none()))
                .count()
        };
        let agrees = |g: &Group| assert_eq!(g.nacked_len(), scan(g), "the field drifted");

        agrees(&g);
        g.release(Id::new(1, 0), Retry::Keep);
        g.release(Id::new(2, 0), Retry::Keep);
        agrees(&g);

        // A claim takes one back into somebody's hands.
        g.claim(Id::new(1, 0), b, 200, None, true);
        agrees(&g);
        // An ack takes one out of the list altogether, released or not.
        assert!(g.ack(Id::new(2, 0)));
        assert!(g.ack(Id::new(3, 0)));
        agrees(&g);
        // And a forced release on an entry nobody was ever handed.
        g.force_release(Id::new(9, 0), Retry::Down);
        agrees(&g);
        assert_eq!(g.nacked_len(), 1);
    }

    /// A consumer filter skips released entries, because a released entry has no
    /// consumer to match and `XPENDING key group - + n consumer` is a question
    /// about one consumer's work.
    #[test]
    fn a_released_entry_is_not_anybodys_pending_work() {
        let mut g = group();
        let a = g.consumer_or_create(b"alice", 1);
        g.deliver(a, Id::new(3, 0), 100);
        g.deliver(a, Id::new(5, 0), 100);
        g.release(Id::new(3, 0), Retry::Keep);

        let seen = |g: &Group, owner| {
            let mut out = Vec::new();
            let want = Filter {
                owner,
                ..Filter::default()
            };
            g.pending_range(want, 1_000, |id, _, c| {
                out.push((id, c.map(|c| c.name().to_vec())));
                true
            });
            out
        };

        assert_eq!(
            seen(&g, None),
            vec![
                (Id::new(3, 0), None),
                (Id::new(5, 0), Some(b"alice".to_vec()))
            ]
        );
        assert_eq!(
            seen(&g, Some(a)),
            vec![(Id::new(5, 0), Some(b"alice".to_vec()))]
        );
        // The summary counts it against nobody, so alice is down to one.
        assert_eq!(
            g.pending_counts()
                .map(|(name, n)| (name.to_vec(), n))
                .collect::<Vec<_>>(),
            vec![(b"alice".to_vec(), 1)]
        );
    }

    #[test]
    fn a_frozen_group_with_a_pending_entry_nobody_could_hold_is_refused() {
        let mut g = group();
        let slot = g.consumer_or_create(b"alice", 1_000);
        g.deliver(slot, Id::new(1, 0), 1_000);
        let mut bytes = Vec::new();
        g.freeze(&mut bytes);
        assert_eq!(Group::thaw(&mut frozen::Cut::new(&bytes)), Ok(g));

        // The owner is the last number in the body, and a slot past the end
        // would be an entry no consumer could ever be told to finish.
        let mut bad = bytes.clone();
        *bad.last_mut().expect("a body") = 7;
        assert_eq!(Group::thaw(&mut frozen::Cut::new(&bad)), Err(Broken::Body));

        for cut in 0..bytes.len() {
            assert!(
                Group::thaw(&mut frozen::Cut::new(&bytes[..cut])).is_err(),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_frozen_group_that_names_one_consumer_twice_is_refused() {
        let mut g = group();
        g.consumer_or_create(b"alice", 1_000);
        g.consumer_or_create(b"carol", 1_000);
        let mut bytes = Vec::new();
        g.freeze(&mut bytes);
        // Both names are five letters, so one name becomes the other without
        // the length in front of it moving.
        let at = bytes
            .windows(5)
            .position(|w| w == b"carol")
            .expect("the second name");
        bytes[at..at + 5].copy_from_slice(b"alice");
        assert_eq!(
            Group::thaw(&mut frozen::Cut::new(&bytes)),
            Err(Broken::Body)
        );
    }
}
