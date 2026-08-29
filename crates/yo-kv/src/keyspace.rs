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

use yo_index::RawMap;

use crate::Clock;
use crate::value::{self, Kind};

/// One database: every key, whatever type it holds.
pub struct Keyspace {
    pub(crate) map: RawMap,
    pub(crate) clock: Clock,
    /// Keys that were found dead on the way to answering something else.
    pub(crate) expired: u64,
}

impl Keyspace {
    /// An empty database on the system clock.
    #[must_use]
    pub fn new() -> Keyspace {
        Keyspace::with_clock(Clock::system())
    }

    /// An empty database on a clock of the caller's choosing.
    #[must_use]
    pub fn with_clock(clock: Clock) -> Keyspace {
        Keyspace {
            map: RawMap::new(),
            clock,
            expired: 0,
        }
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
            self.map.del(key);
            self.expired += 1;
            return None;
        }
        Some(kind)
    }

    /// Throw every key away. This is `FLUSHDB` on one database.
    ///
    /// The expiry counter is not reset, because Redis does not reset it either:
    /// `expired_keys` in `INFO stats` counts what this process has expired since
    /// it started, and emptying a database is not expiring anything.
    pub fn clear(&mut self) {
        self.map.clear();
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

    /// Bytes held by the index and the arena.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.map.memory_bytes()
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
