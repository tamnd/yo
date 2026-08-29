//! One database, and the parts of it that are not about any particular type.
//!
//! This is the `dict` a Redis `SELECT` picks between, and one of these is what a
//! shard owns. It was called `Strings` while strings were the only thing in it,
//! which was accurate for M2 and stopped being accurate the moment a set needed
//! somewhere to live.
//!
//! The commands hang off this as separate `impl` blocks, one file per type, so
//! that `SET` lives in [`strings`](crate::strings) next to the other twenty five
//! string commands rather than in a file that is the whole of Redis. They are
//! methods on the keyspace and not on some per type object because a key belongs
//! to the database and not to a type: `DEL` does not care what it is deleting,
//! and `SADD` against a string has to be able to see that it is a string.
//!
//! # Not Sync
//!
//! Like everything that hangs off a shard. One of these belongs to one thread and
//! is reached by sending that thread a command, which is Y1, and it is why
//! nothing here takes a lock or an atomic.

use std::sync::atomic::{AtomicU64, Ordering};

use yo_common::{Code, Error, Rng};
use yo_index::RawMap;

use crate::Clock;
use crate::set::{self, Set};
use crate::slab::Slab;
use crate::value::{self, Kind};

/// One database: every key, whatever type it holds.
pub struct Keyspace {
    pub(crate) map: RawMap,
    pub(crate) clock: Clock,
    /// Keys that were found dead on the way to answering something else.
    pub(crate) expired: u64,
    /// Every set in this database, addressed by the number in its record.
    pub(crate) sets: Slab<Set>,
    /// How many keys hold something that is not a string.
    ///
    /// This exists so that a database of nothing but strings, which is every
    /// benchmark today and most of what `SET` sees, can skip the body check in
    /// [`Keyspace::free_body`] on one predictable branch against a field that is
    /// already hot, rather than paying a second lookup per write forever.
    pub(crate) bodies: usize,
    /// Where a set changes representation.
    pub(crate) limits: set::Limits,
    /// Where `SPOP` and `SRANDMEMBER` draw from.
    pub(crate) rng: Rng,
}

/// How many databases this process has made.
///
/// Mixed into a new database's seed so that the eight shards a server starts in
/// the same millisecond do not all draw the same members in the same order. It
/// is the only atomic in this file and it is touched once per database rather
/// than once per command, so it is not on any path Y1 cares about.
static MADE: AtomicU64 = AtomicU64::new(0);

impl Keyspace {
    /// An empty database on the system clock.
    #[must_use]
    pub fn new() -> Keyspace {
        Keyspace::with_clock(Clock::system())
    }

    /// An empty database on a clock of the caller's choosing.
    #[must_use]
    pub fn with_clock(clock: Clock) -> Keyspace {
        let made = MADE.fetch_add(1, Ordering::Relaxed);
        Keyspace {
            map: RawMap::new(),
            clock,
            expired: 0,
            sets: Slab::new(),
            bodies: 0,
            limits: set::Limits::DEFAULT,
            rng: Rng::new(clock.now_ms() ^ made.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        }
    }

    /// Pin what `SPOP` and `SRANDMEMBER` draw.
    ///
    /// A database seeds itself from the clock and a counter, which is what a
    /// server wants and what a test cannot assert against. Every test in this
    /// crate that cares which member comes back calls this first, the same way
    /// every expiry test drives a fixed clock, and for the same reason: the one
    /// input that makes a result unrepeatable is better handed in than reached
    /// for.
    ///
    /// It is public because reproducing a bug report is the same problem. A
    /// seed printed in a crash report is worth having somewhere to put.
    #[inline]
    pub const fn seed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// Where a set changes representation, which is three `CONFIG` values.
    #[inline]
    pub const fn limits(&self) -> &set::Limits {
        &self.limits
    }

    /// Change where a set changes representation.
    ///
    /// Moving these does not rewrite the sets that already exist, which is what
    /// Redis does too: `CONFIG SET set-max-listpack-entries 0` leaves every
    /// listpack alone and only decides what the next `SADD` builds.
    #[inline]
    pub const fn set_limits(&mut self, limits: set::Limits) {
        self.limits = limits;
    }

    /// The clock expiry compares against.
    #[inline]
    pub const fn clock(&self) -> &Clock {
        &self.clock
    }

    /// The clock, to refresh once per turn of the loop.
    #[inline]
    pub const fn clock_mut(&mut self) -> &mut Clock {
        &mut self.clock
    }

    /// The map underneath, for statistics and for compaction.
    #[inline]
    pub const fn map(&self) -> &RawMap {
        &self.map
    }

    /// How many keys are stored, including any that are dead and not yet
    /// noticed. This is Redis's `DBSIZE`, which counts the same way.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether anything is stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// What `key` holds, or `None` if there is nothing under it.
    ///
    /// This is `TYPE`. A key past its deadline is reaped first, so a dead key
    /// answers `None` and not the type it used to be.
    ///
    /// One lookup, because the tag and the deadline are both in the record the
    /// lookup returned. Reading the kind out before the reap rather than after
    /// is what keeps it to one.
    pub fn kind_of(&mut self, key: &[u8]) -> Option<Kind> {
        let now = self.clock.now_ms();
        let (kind, dead) = self
            .map
            .get(key)
            .map(|rec| (value::kind(rec), value::is_expired(rec, now)))?;
        if dead {
            self.drop_key(key);
            self.expired += 1;
            return None;
        }
        Some(kind)
    }

    /// How a set is represented, or `None` if `key` is not a set.
    ///
    /// This follows the slot and asks the body rather than reading the record,
    /// because the record only holds a number. Putting a copy of the
    /// representation in the record's two spare encoding bits would mean
    /// rewriting the record every time a set was promoted, for the sake of a
    /// command nobody calls in a loop, and would leave two places able to
    /// disagree about the same fact.
    pub fn set_encoding(&mut self, key: &[u8]) -> Option<set::Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        if value::kind(rec) != Kind::Set {
            return None;
        }
        let at = value::slot(rec);
        Some(self.sets.get(at)?.encoding())
    }

    /// `OBJECT ENCODING key`, as the word Redis puts on the wire.
    ///
    /// One place that knows every type's answer, so that adding the hash means
    /// adding an arm here and not finding the four callers that each worked it
    /// out for themselves.
    pub fn encoding_name(&mut self, key: &[u8]) -> Option<&'static str> {
        match self.kind_of(key)? {
            Kind::String => self.encoding(key).map(value::Encoding::name),
            Kind::Set => self.set_encoding(key).map(set::Encoding::name),
            other => unreachable!("nothing can store a {} yet", other.name()),
        }
    }

    /// Put a deadline on `key`, or take one off. Answers whether it was there.
    ///
    /// Any type. A deadline lives in the record and changes its length, so this
    /// writes the record again rather than patching it, and for a set that is
    /// five bytes or thirteen and never the members. The body is left exactly
    /// where it is, which is why this writes through the map instead of taking
    /// the free the body path an overwrite takes.
    ///
    /// This is what `EXPIRE`, `PEXPIRE`, `EXPIREAT` and `PERSIST` will call when
    /// they land. They are not here yet because until now the only type was the
    /// string and `SET` and `GETEX` between them covered every case.
    pub fn set_expiry(&mut self, key: &[u8], at: Option<u64>) -> bool {
        self.reap(key);
        let Some(rec) = self.map.get(key) else {
            return false;
        };
        if value::expire_at(rec) == at {
            return true;
        }
        // Read what has to survive out of the record before writing over it.
        match value::kind(rec) {
            Kind::String => {
                let bytes = value::read(rec).to_vec();
                self.store(key, &bytes, at);
            }
            Kind::Set => {
                let slot = value::slot(rec);
                let len = value::slot_record_len(at.is_some());
                self.map.set_with(key, len, |out| {
                    value::write_slot_record(out, Kind::Set, slot, at);
                });
            }
            other => unreachable!("nothing can store a {} yet", other.name()),
        }
        true
    }

    /// Give back whatever `key` holds outside its record, if it holds anything.
    ///
    /// Every path that deletes a key or writes over one has to come through
    /// here, because a set that loses its record without losing its slab slot is
    /// a leak that nothing ever notices: the memory is reachable, the slot is
    /// never reused, and `DBSIZE` looks right. Six delete sites and four string
    /// writers each remembering to do it themselves is five chances to forget,
    /// and one of them would be forgotten. So this is the funnel, and when the
    /// hash type lands the only place that changes is the match below.
    ///
    /// The record is left alone. This frees the body and the caller either
    /// deletes the record or writes a new one over it.
    pub(crate) fn free_body(&mut self, key: &[u8]) {
        if self.bodies == 0 {
            return;
        }
        let Some(rec) = self.map.get(key) else {
            return;
        };
        match value::kind(rec) {
            Kind::String => {}
            Kind::Set => {
                let at = value::slot(rec);
                self.sets.remove(at);
                self.bodies -= 1;
            }
            other => unreachable!("nothing can store a {} yet", other.name()),
        }
    }

    /// Delete `key` and whatever it held. Answers whether it was there.
    #[inline]
    pub(crate) fn drop_key(&mut self, key: &[u8]) -> bool {
        self.free_body(key);
        self.map.del(key)
    }

    /// Drop `key` if its deadline has passed.
    ///
    /// This is lazy expiry and it is half of the story. The other half is the
    /// active cycle in the maintenance slice, which is what stops a key nobody
    /// ever reads again from holding its memory forever (`14` section 1).
    ///
    /// Every public read calls this first, whatever type it is reading, which
    /// is why it is here and not in the file for any one type.
    #[inline]
    pub(crate) fn reap(&mut self, key: &[u8]) {
        let now = self.clock.now_ms();
        let dead = self.map.get(key).is_some_and(|r| value::is_expired(r, now));
        if dead {
            self.drop_key(key);
            self.expired += 1;
        }
    }

    /// Throw every key away. This is `FLUSHDB` on one database.
    ///
    /// The expiry counter is not reset, because Redis does not reset it either:
    /// `expired_keys` in `INFO stats` counts what this process has expired since
    /// it started, and emptying a database is not expiring anything.
    pub fn clear(&mut self) {
        self.map.clear();
        self.sets.clear();
        self.bodies = 0;
    }

    /// Keys reclaimed by running into them after their deadline.
    ///
    /// Redis calls this `expired_keys` in `INFO stats` and counts both lazy and
    /// active expiry into it. Only lazy expiry exists so far, so only lazy
    /// expiry is counted, and the active cycle in the maintenance slice will add
    /// to the same number when it lands (`14` section 1).
    #[inline]
    pub const fn expired_keys(&self) -> u64 {
        self.expired
    }

    /// Bytes held by the index, the arena and every body hanging off them.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.map.memory_bytes()
            + self.sets.memory_bytes()
            + self.sets.iter().map(Set::memory_bytes).sum::<usize>()
    }

    /// Give back one segment's worth of space if one has gone mostly dead.
    ///
    /// Overwriting a key does not reuse its bytes, it writes the new record at
    /// the bump pointer and counts the old one as dead, so a workload that sets
    /// the same keys over and over holds far more than it is storing until
    /// something compacts. This is that something, and it does at most one
    /// segment per call so that the loop can afford to ask every turn.
    #[inline]
    pub fn compact_step(&mut self) -> Option<usize> {
        self.map.compact_step()
    }

    /// Ask the cache for the bucket this key will land in.
    ///
    /// The first of the loop's two walks (`04` section 3) calls this.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        self.map.prefetch(hash);
    }

    /// The hash this database files `key` under.
    #[inline]
    #[must_use]
    pub fn hash_of(key: &[u8]) -> u64 {
        RawMap::hash_of(key)
    }
}

/// What Redis says when a command is sent at a key holding another type.
///
/// The text is Redis's, word for word, because it goes on the wire verbatim and
/// clients match on it. The `WRONGTYPE` at the front is not part of the message:
/// the protocol layer puts it there from the [`Code`], which is what lets an
/// embedded caller match on a value instead of on a string (P5).
pub fn wrong_type() -> Error {
    Error::new(
        Code::WrongType,
        "Operation against a key holding the wrong kind of value",
    )
}

impl Default for Keyspace {
    fn default() -> Keyspace {
        Keyspace::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    #[test]
    fn type_answers_string_for_a_string_and_nothing_for_a_missing_key() {
        let mut d = db();
        d.set_plain(b"k", b"v").expect("room");
        assert_eq!(d.kind_of(b"k"), Some(Kind::String));
        assert_eq!(d.kind_of(b"nope"), None);
    }

    #[test]
    fn type_does_not_report_a_key_whose_deadline_has_gone() {
        let mut d = db();
        d.psetex(b"k", 100, b"v").expect("room");
        assert_eq!(d.kind_of(b"k"), Some(Kind::String));

        d.clock_mut().advance(100);
        assert_eq!(
            d.kind_of(b"k"),
            None,
            "the deadline was 1100 and it is 1100"
        );
        assert_eq!(d.len(), 0, "and asking reaped it rather than leaving it");
        assert_eq!(d.expired_keys(), 1);
    }
}
