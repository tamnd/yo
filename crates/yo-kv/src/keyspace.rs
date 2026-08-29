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

use yo_common::{Addr, Code, Error, Result, Rng};
use yo_index::RawMap;

use crate::Clock;
use crate::hash::{self, Hash};
use crate::set::{self, Set};
use crate::slab::Slab;
use crate::ttl::{self, Applied, Ask, Cond};
use crate::value::{self, Kind};

/// One database: every key, whatever type it holds.
pub struct Keyspace {
    pub(crate) map: RawMap,
    pub(crate) clock: Clock,
    /// Keys that were found dead on the way to answering something else.
    pub(crate) expired: u64,
    /// Every set in this database, addressed by the number in its record.
    pub(crate) sets: Slab<Set>,
    /// Every hash in this database, addressed the same way.
    ///
    /// A slab per type rather than one slab of an enum, so that a record's four
    /// bytes index a `Hash` directly and reaching one is a load and not a load
    /// followed by a discriminant check. The type tag in the record already
    /// says which slab to look in, so the discriminant would be a second copy
    /// of a fact the record has.
    pub(crate) hashes: Slab<Hash>,
    /// How many keys hold something that is not a string.
    ///
    /// This exists so that a database of nothing but strings, which is every
    /// benchmark today and most of what `SET` sees, can skip the body check in
    /// [`Keyspace::free_body`] on one predictable branch against a field that is
    /// already hot, rather than paying a second lookup per write forever.
    pub(crate) bodies: usize,
    /// Where a set changes representation.
    pub(crate) limits: set::Limits,
    /// Where a hash changes representation.
    pub(crate) hash_limits: hash::Limits,
    /// Where `SPOP` and `SRANDMEMBER` draw from.
    pub(crate) rng: Rng,
    /// The last collection key that was resolved, for the command behind it.
    memo: Memo,
}

/// Where the last collection key resolved to, if it still resolves there.
///
/// Y13 says a batch of `SADD` on one key should be one table growth check, and
/// the same argument applies a step earlier: it should be one resolve. A
/// resolve is a hash, a bucket walk and a record read, and on a hot key every
/// command in the batch was paying for all three to be told the same answer the
/// command in front of it got.
///
/// One entry and not a cache, because one entry is the shape of the problem.
/// Single key `SADD` is the case with no spread to exploit, so the only reuse
/// there is to find is the command immediately before, and a bigger structure
/// would cost a lookup to avoid a lookup.
///
/// It holds a slot and not an address. A slot is an index into the slab for its
/// type and stays right for as long as the key is there, where an address is
/// only good until the next write. That is also why nothing here memoizes a
/// string: a string lives in the record itself and moves when the record does.
struct Memo {
    /// What the map's write counter said when this was taken.
    writes: u64,
    /// Whether there is anything here. Separate from the length because the
    /// empty key is a key, and `SADD "" m` is a command Redis accepts.
    live: bool,
    /// The type the key held, so a hit can still answer `WRONGTYPE`.
    kind: Kind,
    /// Where the body is in the slab for `kind`.
    slot: u32,
    /// How much of `key` is the key.
    len: u8,
    key: [u8; Memo::MAX],
}

impl Memo {
    /// The longest key worth remembering.
    ///
    /// Thirty two bytes is half a cache line and covers every hot key anyone
    /// writes down, including the `myset:{tag}` the generators send. A longer
    /// key is not memoized rather than heap allocated, because the whole point
    /// of this is to not touch memory it does not have to.
    const MAX: usize = 32;

    const fn empty() -> Memo {
        Memo {
            writes: 0,
            live: false,
            kind: Kind::String,
            slot: 0,
            len: 0,
            key: [0; Memo::MAX],
        }
    }

    /// What `key` resolved to last time, if that answer still stands.
    ///
    /// `writes` is the map's counter now. Any write at all since this was taken
    /// and the answer is thrown away, which is stricter than it has to be and is
    /// the version that cannot be wrong.
    #[inline]
    fn get(&self, writes: u64, key: &[u8]) -> Option<(Kind, u32)> {
        if !self.live || self.writes != writes || key.len() != self.len as usize {
            return None;
        }
        (self.key[..key.len()] == *key).then_some((self.kind, self.slot))
    }

    /// Remember that `key` is at `slot`.
    #[inline]
    fn put(&mut self, writes: u64, key: &[u8], kind: Kind, slot: u32) {
        if key.len() > Memo::MAX {
            self.live = false;
            return;
        }
        self.writes = writes;
        self.live = true;
        self.kind = kind;
        self.slot = slot;
        self.len = key.len() as u8;
        self.key[..key.len()].copy_from_slice(key);
    }
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
            hashes: Slab::new(),
            bodies: 0,
            limits: set::Limits::DEFAULT,
            hash_limits: hash::Limits::DEFAULT,
            rng: Rng::new(clock.now_ms() ^ made.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            memo: Memo::empty(),
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

    /// Where a hash changes representation, which is two `CONFIG` values.
    #[inline]
    pub const fn hash_limits(&self) -> &hash::Limits {
        &self.hash_limits
    }

    /// Change where a hash changes representation.
    ///
    /// Same rule as the set: moving these leaves every hash that already exists
    /// exactly as it is, and only decides what the next `HSET` builds.
    #[inline]
    pub const fn set_hash_limits(&mut self, limits: hash::Limits) {
        self.hash_limits = limits;
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

    /// How a hash is represented, or `None` if `key` is not a hash.
    ///
    /// The same shape as [`Keyspace::set_encoding`] and for the same reason: the
    /// record holds a slot number and the body is the thing that knows which of
    /// the two it currently is.
    pub fn hash_encoding(&mut self, key: &[u8]) -> Option<hash::Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        if value::kind(rec) != Kind::Hash {
            return None;
        }
        let at = value::slot(rec);
        Some(self.hashes.get(at)?.encoding())
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
            Kind::Hash => self.hash_encoding(key).map(hash::Encoding::name),
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
    /// This is the raw write. [`Keyspace::expire`] and [`Keyspace::persist`] are
    /// what `EXPIRE` and its family call, and they come through here once they
    /// have worked out whether the deadline is allowed to move.
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
            // Every body type writes the same record: a tag and a slot number.
            // The body is not touched and does not need to be, which is the
            // whole point of keeping it out of the record.
            kind @ (Kind::Set | Kind::Hash) => {
                let slot = value::slot(rec);
                let len = value::slot_record_len(at.is_some());
                self.map.set_with(key, len, |out| {
                    value::write_slot_record(out, kind, slot, at);
                });
            }
            other => unreachable!("nothing can store a {} yet", other.name()),
        }
        true
    }

    /// The key's deadline, as the three way answer `TTL` and `PTTL` are built on.
    ///
    /// [`Ask::Missing`] for a key that is not there, [`Ask::NoDeadline`] for one
    /// that is and has no deadline, and the absolute millisecond otherwise. A key
    /// past its deadline is reaped on the way through, so it answers `Missing`
    /// and not the moment that has gone.
    pub fn deadline_of(&mut self, key: &[u8]) -> Ask {
        let Some(addr) = self.live_rec(key) else {
            return Ask::Missing;
        };
        match value::expire_at(self.map.value_at(addr)) {
            Some(at) => Ask::At(at),
            None => Ask::NoDeadline,
        }
    }

    /// Move `key`'s deadline to `at`, if `cond` lets it.
    ///
    /// This is `EXPIRE`, `PEXPIRE`, `EXPIREAT` and `PEXPIREAT`, which differ only
    /// in the unit and the origin of the number. All four turn it into one
    /// absolute millisecond before they get here, so the condition rules live in
    /// one place and the four commands cannot drift apart.
    ///
    /// A deadline that has already passed deletes the key rather than being
    /// stored, and the answer says so. `EXPIRE` cannot report the difference
    /// because it replies 1 either way, but the caller is not always `EXPIRE`,
    /// and a delete is a different thing from a deadline.
    ///
    /// The condition is checked before the past check, which is the order Redis
    /// uses and is the one that matters: `EXPIRE key 0 XX` on a key with no
    /// deadline answers 0 and leaves the key alone, rather than deleting it.
    pub fn expire(&mut self, key: &[u8], at: u64, cond: Cond) -> Applied {
        let prev = match self.deadline_of(key) {
            Ask::Missing => return Applied::Missing,
            Ask::NoDeadline => None,
            Ask::At(at) => Some(at),
        };
        let done = ttl::decide(prev, at, cond, self.clock.now_ms());
        match done {
            Applied::Ok => {
                self.set_expiry(key, Some(at));
            }
            // The structure that answered `Deleted` for a field only holds
            // deadlines, so its caller has to remove the field. Here the caller
            // is us and the key is ours, so it goes now.
            Applied::Deleted => {
                self.drop_key(key);
            }
            Applied::Missing | Applied::NotMet => {}
        }
        done
    }

    /// Take `key`'s deadline off. Answers whether there was one to take.
    ///
    /// This is `PERSIST`, and the reply is the same 0 for a key that is not there
    /// and a key that was never going to expire, which is Redis's answer and not
    /// a shortcut here.
    pub fn persist(&mut self, key: &[u8]) -> bool {
        if !matches!(self.deadline_of(key), Ask::At(_)) {
            return false;
        }
        self.set_expiry(key, None);
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
            Kind::Hash => {
                let at = value::slot(rec);
                self.hashes.remove(at);
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

    /// Where `key`'s record is, having thrown the key away first if it is dead.
    ///
    /// The same fold as [`Keyspace::live_slot`] for a caller that wants the
    /// record itself rather than a slot number, which is every string command.
    /// `GET` used to be a reap, then a type check, then a read, and each of the
    /// three hashed the key and walked a bucket for the same record. It is one
    /// walk now and two arena reads, and an arena read at a known address is a
    /// load.
    ///
    /// The address dies at the next write, which is why this is `pub(crate)`
    /// and why every caller reads it and drops it inside one command.
    pub(crate) fn live_rec(&mut self, key: &[u8]) -> Option<Addr> {
        let now = self.clock.now_ms();
        let addr = self.map.find(key)?;
        if value::is_expired(self.map.value_at(addr), now) {
            self.drop_key(key);
            self.expired += 1;
            return None;
        }
        Some(addr)
    }

    /// The slot under `key`, having thrown the key away first if it is dead.
    ///
    /// `None` for a key that is not there or that was and is now reaped, and
    /// `WRONGTYPE` for a key holding something other than `want`.
    ///
    /// One probe of the map, where a [`Keyspace::reap`] followed by a `get`
    /// costs two. That pair is how every collection command used to start, so a
    /// pipeline of sixty four `SADD` on one key hashed and probed for that key a
    /// hundred and twenty eight times to do sixty four inserts. The reap has to
    /// read the record and the command has to read the same record, and there
    /// was never a reason for those to be two visits.
    ///
    /// It answers a number rather than the record it just read because of the
    /// borrow checker and not because a number is nicer. A method that hands
    /// back a borrow of the map on one path and takes a mutable borrow to reap
    /// on the other is the case the borrow checker still refuses without
    /// Polonius. A slot is four bytes and copies out, so the borrow ends here
    /// and the caller reaches its body through the slab.
    ///
    /// And no probe at all when the command in front of it asked for the same
    /// key and nothing has been written since, which is the [`Memo`] and is what
    /// Y13 asks for on single key `SADD`.
    pub(crate) fn live_slot(&mut self, key: &[u8], want: Kind) -> Result<Option<u32>> {
        if let Some((kind, slot)) = self.memo.get(self.map.writes(), key) {
            if kind != want {
                return Err(wrong_type());
            }
            return Ok(Some(slot));
        }
        let now = self.clock.now_ms();
        let Some(rec) = self.map.get(key) else {
            return Ok(None);
        };
        if value::is_expired(rec, now) {
            self.drop_key(key);
            self.expired += 1;
            return Ok(None);
        }
        if value::kind(rec) != want {
            return Err(wrong_type());
        }
        let slot = value::slot(rec);
        // A key with a deadline is not memoized. The memo is invalidated by
        // writes and a deadline passes without one, so remembering a dated key
        // would be remembering it past the moment it should have been reaped.
        let dated = value::expire_at(rec).is_some();
        if !dated {
            self.memo.put(self.map.writes(), key, want, slot);
        }
        Ok(Some(slot))
    }

    /// Throw every key away. This is `FLUSHDB` on one database.
    ///
    /// The expiry counter is not reset, because Redis does not reset it either:
    /// `expired_keys` in `INFO stats` counts what this process has expired since
    /// it started, and emptying a database is not expiring anything.
    pub fn clear(&mut self) {
        self.map.clear();
        self.sets.clear();
        self.hashes.clear();
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
            + self.hashes.memory_bytes()
            + self.hashes.iter().map(Hash::memory_bytes).sum::<usize>()
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
