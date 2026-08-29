//! The keyspace itself: what is there, what type it is, and when it goes away.
//!
//! [`Strings`](crate::Strings) and [`Sets`](crate::Sets) each hold the commands
//! for one type. The commands here hold for all of them, because a deadline is
//! not a string thing or a set thing. `EXPIRE` puts the moment in the key's own
//! record, so the same call works on a key holding a string, a set or a hash and
//! costs the same on each.
//!
//! ```
//! use std::time::Duration;
//!
//! let db = yo::open(yo::MEMORY)?;
//! let keys = db.keys();
//!
//! db.set("online").add("alice")?;
//! keys.expire_in("online", Duration::from_secs(60))?;
//!
//! assert_eq!(keys.kind("online")?, Some(yo::Kind::Set));
//! assert!(keys.ttl("online")?.left().is_some());
//! # Ok::<(), yo::Error>(())
//! ```
//!
//! # What a deadline is not
//!
//! It is not a property of the value. Giving a set a deadline rewrites five
//! bytes of the key's record and does not touch a single member, which is why
//! [`Keys::expire_in`] on a set of a million members is the same call as on a
//! set of one.
//!
//! It is also not the same as the per field deadlines a hash can carry. Those
//! are `HEXPIRE` and they live in the hash. A key can have a deadline while its
//! fields have their own, and neither one knows about the other.
//!
//! # The clock is only read when it matters
//!
//! A database that has never been asked for a deadline never reads the clock on
//! the data path, which [`Db::reads_the_clock`](crate::Db::reads_the_clock)
//! reports. The first call here that creates one turns that on for good, so it
//! is worth knowing that this is where the tens of nanoseconds come from.
//!
//! # There is no touch
//!
//! Redis has a `TOUCH`, and on a real server it counts the keys that are there
//! and moves each of them up the eviction order. There is no eviction here, so
//! all it could do is count, and [`Keys::count`] already does that. A second
//! name for one call is worse than no second name, so the wire has `TOUCH` for
//! the clients that send it and this does not.
//!
//! # There is no cursor
//!
//! The wire has `SCAN` because a server cannot stop and walk a keyspace for one
//! client while every other client waits, so it hands out a number and does the
//! walk in pieces. Nothing here is in that position. [`Keys::each`] holds the
//! database for as long as it runs and nothing else can write to it in the
//! meantime, so it is one walk, it sees one version of the keyspace, and there
//! is no cursor to hold and no duplicate to filter out.
//!
//! What that costs is that a walk of ten million keys is ten million calls
//! before the next line of your program runs. That is the same trade `KEYS`
//! makes and it is the right one here, because the thing on the other side of
//! the call is your own code rather than a socket.
//!
//! # The typed collections are somewhere else
//!
//! A [`Map`](crate::Map) is a named collection and not a key in the keyspace, so
//! it does not show up here and cannot be given a deadline. That is `15`
//! section 3's split and not an oversight: a map's name is checked when it is
//! opened, and a key's name is whatever you pass.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use yo_common::{Code, Error, Result};
use yo_kv::{Applied, Ask, Cond, Kind, MAX_AT, Moved};

use crate::db::Handle;

/// What a key says about when it goes away.
///
/// Three answers rather than two, because a key that is not there and a key
/// that is never going away are different things and code that confuses them
/// deletes the wrong data. Redis says this with `-2` and `-1` and hopes you
/// read the manual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ttl {
    /// There is no such key.
    Missing,
    /// The key is there and nothing is going to take it away.
    Forever,
    /// The key is there and this much of it is left.
    In(Duration),
}

impl Ttl {
    /// How long is left, or `None` for a key that is missing or has no
    /// deadline.
    ///
    /// The short answer for code that only wants to know whether it should
    /// refresh something. When the difference matters, match on the variants.
    #[must_use]
    pub fn left(self) -> Option<Duration> {
        match self {
            Ttl::In(left) => Some(left),
            _ => None,
        }
    }

    /// Whether the key is there at all.
    #[must_use]
    pub fn found(self) -> bool {
        !matches!(self, Ttl::Missing)
    }
}

/// Whether a deadline is allowed to move, which is `EXPIRE`'s `NX`, `XX`, `GT`
/// and `LT`.
///
/// The condition is checked before the moment is, so a deadline that has
/// already gone and a condition that says no leaves the key alone rather than
/// deleting it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum When {
    /// Whatever is there now. Plain `EXPIRE`.
    #[default]
    Always,
    /// Only if the key has no deadline yet. `NX`.
    Unset,
    /// Only if it already has one. `XX`.
    AlreadySet,
    /// Only if this pushes the deadline further out. `GT`.
    ///
    /// A key with no deadline is refused, because no deadline reads as
    /// infinitely far away and nothing is further out than that.
    Later,
    /// Only if this brings the deadline in. `LT`.
    ///
    /// A key with no deadline is accepted, by the same reading.
    Earlier,
    /// Only if there is one now and this brings it in. `XX LT`.
    ///
    /// The one combination the other five cannot say. `XX GT` is just `GT`,
    /// since `GT` already refuses a key with no deadline.
    EarlierAndAlreadySet,
}

impl From<When> for Cond {
    fn from(when: When) -> Cond {
        match when {
            When::Always => Cond::Always,
            When::Unset => Cond::NotSet,
            When::AlreadySet => Cond::AlreadySet,
            When::Later => Cond::Greater,
            When::Earlier => Cond::Less,
            When::EarlierAndAlreadySet => Cond::LessAndSet,
        }
    }
}

/// Every command that works on a key whatever the key holds.
///
/// `DEL`, `EXISTS` and `TYPE`, plus the whole expiry family. Keys are byte
/// strings the way Redis's are, so anything that is bytes will do.
///
/// ```
/// let db = yo::open(yo::MEMORY)?;
/// let keys = db.keys();
///
/// db.strings().set("greeting", "hello")?;
/// assert!(keys.exists("greeting")?);
/// assert!(keys.del("greeting")?);
/// assert!(!keys.exists("greeting")?);
/// # Ok::<(), yo::Error>(())
/// ```
#[derive(Clone)]
pub struct Keys {
    pub(crate) db: Handle,
}

impl core::fmt::Debug for Keys {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Keys").finish_non_exhaustive()
    }
}

impl Keys {
    /// Whether a key is there. `EXISTS`.
    ///
    /// A key whose deadline has gone is not there, whether or not anything has
    /// got around to removing it yet.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn exists(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        self.db.run(|inner| Ok(inner.strings.exists(key.as_ref())))
    }

    /// How many of these keys are there. `EXISTS` with several.
    ///
    /// The same key twice counts twice, which is Redis's rule and is worth
    /// knowing before you use this to count distinct things.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn count<K: AsRef<[u8]>>(&self, keys: &[K]) -> Result<usize> {
        self.db.run(|inner| {
            Ok(keys
                .iter()
                .filter(|key| inner.strings.exists(key.as_ref()))
                .count())
        })
    }

    /// What a key holds, or `None` if it holds nothing. `TYPE`.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn kind(&self, key: impl AsRef<[u8]>) -> Result<Option<Kind>> {
        self.db.run(|inner| Ok(inner.strings.kind_of(key.as_ref())))
    }

    /// Remove a key, and say whether it was there. `DEL`.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn del(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        self.db.run(|inner| Ok(inner.strings.del(key.as_ref())))
    }

    /// Remove several keys, and say how many were there. `DEL` with a list.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn del_many<K: AsRef<[u8]>>(&self, keys: &[K]) -> Result<usize> {
        self.db.run(|inner| {
            Ok(keys
                .iter()
                .filter(|key| inner.strings.del(key.as_ref()))
                .count())
        })
    }

    /// Give a key this long to live, and say whether the deadline was set.
    /// `PEXPIRE`.
    ///
    /// A duration that has already gone, meaning zero, removes the key and
    /// answers true, because the deadline was applied and applying it is what
    /// took the key away. False means the key is not there.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for a duration that lands past what a millisecond
    /// timestamp reaches, which is the year 4199, or if called from inside a
    /// callback that is already holding this database.
    pub fn expire_in(&self, key: impl AsRef<[u8]>, after: Duration) -> Result<bool> {
        self.expire_in_when(key, after, When::Always)
    }

    /// The same with a condition on it. `PEXPIRE` with `NX`, `XX`, `GT` or
    /// `LT`.
    ///
    /// False now means either that the key is not there or that the condition
    /// said no, which is the one place Redis's reply is genuinely ambiguous.
    /// Ask [`Keys::ttl`] first if you need to tell them apart.
    ///
    /// # Errors
    ///
    /// As [`Keys::expire_in`].
    pub fn expire_in_when(
        &self,
        key: impl AsRef<[u8]>,
        after: Duration,
        when: When,
    ) -> Result<bool> {
        let ms = u64::try_from(after.as_millis()).map_err(|_| too_far())?;
        self.db.deadlines(|inner| {
            let at = inner
                .strings
                .clock()
                .now_ms()
                .checked_add(ms)
                .ok_or_else(too_far)?;
            apply(
                inner
                    .strings
                    .expire(key.as_ref(), reachable(at)?, when.into()),
            )
        })
    }

    /// Set the moment a key goes away, and say whether it was set.
    /// `PEXPIREAT`.
    ///
    /// A moment that has already gone removes the key, the same as
    /// [`Keys::expire_in`] with nothing left on it.
    ///
    /// # Errors
    ///
    /// As [`Keys::expire_in`].
    pub fn expire_at(&self, key: impl AsRef<[u8]>, at: SystemTime) -> Result<bool> {
        self.expire_at_when(key, at, When::Always)
    }

    /// The same with a condition on it. `PEXPIREAT` with `NX`, `XX`, `GT` or
    /// `LT`.
    ///
    /// # Errors
    ///
    /// As [`Keys::expire_in`].
    pub fn expire_at_when(
        &self,
        key: impl AsRef<[u8]>,
        at: SystemTime,
        when: When,
    ) -> Result<bool> {
        let ms = moment(at)?;
        self.db
            .deadlines(|inner| apply(inner.strings.expire(key.as_ref(), ms, when.into())))
    }

    /// How long a key has left. `PTTL`.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn ttl(&self, key: impl AsRef<[u8]>) -> Result<Ttl> {
        self.db.run(|inner| {
            let now = inner.strings.clock().now_ms();
            Ok(match inner.strings.deadline_of(key.as_ref()) {
                Ask::Missing => Ttl::Missing,
                Ask::NoDeadline => Ttl::Forever,
                Ask::At(at) => Ttl::In(Duration::from_millis(at.saturating_sub(now))),
            })
        })
    }

    /// The moment a key goes away, or `None` if it is missing or has no
    /// deadline. `PEXPIRETIME`.
    ///
    /// [`Keys::ttl`] is the one that tells those two apart. This one is for
    /// when the answer needs to survive being written down, since a moment
    /// stays true and a duration goes stale as soon as it is read.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn deadline(&self, key: impl AsRef<[u8]>) -> Result<Option<SystemTime>> {
        self.db.run(|inner| {
            Ok(match inner.strings.deadline_of(key.as_ref()) {
                Ask::At(at) => Some(UNIX_EPOCH + Duration::from_millis(at)),
                Ask::Missing | Ask::NoDeadline => None,
            })
        })
    }

    /// Take a key's deadline away and let it live, and say whether there was
    /// one. `PERSIST`.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn persist(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        self.db.run(|inner| Ok(inner.strings.persist(key.as_ref())))
    }

    /// Move a key to another name, over whatever was there. `RENAME`.
    ///
    /// The value does not move and is not copied. A set or a hash is a slot
    /// number sitting in a record, and the same slot number under a different
    /// key is the same set, so this writes a new record and deletes the old one
    /// however large the value is. Renaming a set of a million members writes
    /// thirteen bytes.
    ///
    /// The deadline travels with the source, and whatever the destination had
    /// goes away with the value it belonged to. A key renamed onto itself is
    /// [`Moved::Ok`] and keeps its deadline.
    ///
    /// [`Moved::Taken`] cannot happen here, which is what
    /// [`Keys::rename_if_new`] is for.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn rename(&self, src: impl AsRef<[u8]>, dst: impl AsRef<[u8]>) -> Result<Moved> {
        self.db
            .run(|inner| Ok(inner.strings.rename(src.as_ref(), dst.as_ref(), false)))
    }

    /// Move a key to another name, but only if that name is free. `RENAMENX`.
    ///
    /// A key renamed onto itself is [`Moved::Taken`], because the destination
    /// does exist and a key is not new because it is the one you already had.
    /// That is the one place this and [`Keys::rename`] disagree about a call
    /// neither of them has to do any work for.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn rename_if_new(&self, src: impl AsRef<[u8]>, dst: impl AsRef<[u8]>) -> Result<Moved> {
        self.db
            .run(|inner| Ok(inner.strings.rename(src.as_ref(), dst.as_ref(), true)))
    }

    /// Copy a value to another key, leaving the destination alone if it is
    /// already there. `COPY`.
    ///
    /// This is the one call here that costs what the value is worth. Two keys
    /// cannot share a body, because then adding a member to one would show up in
    /// the other, so the body is cloned. [`Keys::rename`] is the call that moves
    /// a large value for nothing, and it is the one to reach for when the old
    /// name is not wanted afterwards.
    ///
    /// The deadline is copied too, so a copy of a key with ten seconds left has
    /// ten seconds left. A destination whose deadline has already gone counts as
    /// free.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn copy(&self, src: impl AsRef<[u8]>, dst: impl AsRef<[u8]>) -> Result<Moved> {
        self.db
            .run(|inner| Ok(inner.strings.copy(src.as_ref(), dst.as_ref(), false)))
    }

    /// Copy a value to another key, over whatever was there. `COPY REPLACE`.
    ///
    /// [`Moved::Taken`] cannot happen here, the same way it cannot happen for
    /// [`Keys::rename`].
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn copy_over(&self, src: impl AsRef<[u8]>, dst: impl AsRef<[u8]>) -> Result<Moved> {
        self.db
            .run(|inner| Ok(inner.strings.copy(src.as_ref(), dst.as_ref(), true)))
    }

    /// Every key in the database, one call each. `KEYS *` without the reply.
    ///
    /// The key is handed over where it lies, so a walk of a million keys
    /// allocates nothing at all. It is only borrowed for the length of the
    /// call, which is what stops it from outliving the record it points into,
    /// so anything you want to keep has to be copied out inside the closure.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// db.strings().set("a", "1")?;
    /// db.strings().set("b", "2")?;
    ///
    /// let mut n = 0;
    /// db.keys().each(|_| n += 1)?;
    /// assert_eq!(n, 2);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`], which includes calling any method on this database
    /// from inside the closure.
    pub fn each(&self, mut f: impl FnMut(&[u8])) -> Result<()> {
        self.db.run(|inner| {
            inner.strings.keys(&mut f);
            Ok(())
        })
    }

    /// Every key, copied out into a vector. `KEYS *`.
    ///
    /// The convenient one, and the one that costs a key's worth of memory per
    /// key. [`Keys::each`] is the same walk without that.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn all(&self) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        self.each(|key| out.push(key.to_vec()))?;
        Ok(out)
    }

    /// Every key matching a glob pattern. `KEYS pattern`.
    ///
    /// The same `*`, `?`, `[abc]` and `\` that Redis matches with, so a pattern
    /// that works against a Redis client works here.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// db.strings().set("user:1", "alice")?;
    /// db.strings().set("user:2", "bob")?;
    /// db.strings().set("session:1", "x")?;
    ///
    /// assert_eq!(db.keys().matching("user:*")?.len(), 2);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn matching(&self, pattern: impl AsRef<[u8]>) -> Result<Vec<Vec<u8>>> {
        let pattern = pattern.as_ref();
        let mut out = Vec::new();
        self.each(|key| {
            if yo_common::glob_matches(pattern, key) {
                out.push(key.to_vec());
            }
        })?;
        Ok(out)
    }

    /// One key, chosen at random, or `None` if the database is empty.
    /// `RANDOMKEY`.
    ///
    /// A constant number of loads whatever the database holds, because it picks
    /// a place in the index and takes a key from there rather than walking to
    /// find one.
    ///
    /// # Errors
    ///
    /// As [`Keys::exists`].
    pub fn random(&self) -> Result<Option<Vec<u8>>> {
        self.db.run(|inner| Ok(inner.strings.random_key()))
    }
}

/// Both ways of applying a deadline answer the same question, so they say so in
/// the same place.
fn apply(done: Applied) -> Result<bool> {
    Ok(match done {
        Applied::Ok | Applied::Deleted => true,
        Applied::Missing | Applied::NotMet => false,
    })
}

/// A wall clock moment as milliseconds since the epoch.
///
/// Anything before the epoch is zero, which is a moment that has already gone
/// and therefore removes the key. That is the same answer `PEXPIREAT key 0`
/// gets and there is nothing else it could sensibly mean.
fn moment(at: SystemTime) -> Result<u64> {
    let ms = at
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis());
    reachable(u64::try_from(ms).map_err(|_| too_far())?)
}

/// The wire clamps a deadline past the year 4199 because a real server accepts
/// the number, which is D-17. Nothing is being answered for here, so this says
/// no instead.
fn reachable(at: u64) -> Result<u64> {
    if at > MAX_AT {
        return Err(too_far());
    }
    Ok(at)
}

fn too_far() -> Error {
    Error::new(
        Code::Invalid,
        "that deadline is further away than a millisecond timestamp reaches, which is the year 4199",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MEMORY, open};

    #[test]
    fn a_key_is_there_until_it_is_not() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();

        assert!(!keys.exists("k").unwrap());
        assert_eq!(keys.kind("k").unwrap(), None);
        assert!(!keys.del("k").unwrap());

        db.strings().set("k", "v").unwrap();
        assert!(keys.exists("k").unwrap());
        assert_eq!(keys.kind("k").unwrap(), Some(Kind::String));
        assert!(keys.del("k").unwrap());
        assert!(!keys.exists("k").unwrap());
    }

    #[test]
    fn a_rename_carries_the_deadline_and_a_copy_is_a_second_value() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("a", "v1").unwrap();
        keys.expire_in("a", Duration::from_secs(100)).unwrap();
        db.strings().set("b", "v2").unwrap();

        assert_eq!(keys.rename("a", "b").unwrap(), Moved::Ok);
        assert_eq!(db.strings().get("b").unwrap().as_deref(), Some(&b"v1"[..]));
        assert!(keys.ttl("b").unwrap().left().is_some(), "a's and not b's");
        assert!(!keys.exists("a").unwrap());

        db.set("s").add("m1").unwrap();
        assert_eq!(keys.copy("s", "t").unwrap(), Moved::Ok);
        db.set("t").add("m2").unwrap();
        assert_eq!(db.set("s").len().unwrap(), 1, "the original is intact");
        assert_eq!(db.set("t").len().unwrap(), 2);
    }

    #[test]
    fn the_three_answers_a_move_can_give_are_three_and_not_two() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("a", "v1").unwrap();
        db.strings().set("b", "v2").unwrap();

        // Missing and Taken both mean nothing happened, and a caller that has
        // to tell them apart should not need a second call to find out which.
        assert_eq!(keys.rename_if_new("nosuch", "z").unwrap(), Moved::Missing);
        assert_eq!(keys.rename_if_new("a", "b").unwrap(), Moved::Taken);
        assert_eq!(keys.copy("a", "b").unwrap(), Moved::Taken);
        assert_eq!(db.strings().get("b").unwrap().as_deref(), Some(&b"v2"[..]));

        assert_eq!(keys.copy_over("a", "b").unwrap(), Moved::Ok);
        assert_eq!(db.strings().get("b").unwrap().as_deref(), Some(&b"v1"[..]));
        // Onto itself is the one call the two renames disagree about.
        assert_eq!(keys.rename("a", "a").unwrap(), Moved::Ok);
        assert_eq!(keys.rename_if_new("a", "a").unwrap(), Moved::Taken);
    }

    #[test]
    fn several_keys_at_once_count_the_way_redis_counts_them() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("a", "1").unwrap();
        db.strings().set("b", "2").unwrap();

        assert_eq!(keys.count(&["a", "b", "missing"]).unwrap(), 2);
        assert_eq!(keys.count(&["a", "a"]).unwrap(), 2, "the same key twice");
        assert_eq!(keys.del_many(&["a", "b", "missing"]).unwrap(), 2);
        assert_eq!(keys.count(&["a", "b"]).unwrap(), 0);
    }

    #[test]
    fn a_deadline_lands_on_a_key_whatever_the_key_holds() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("s", "v").unwrap();
        db.set("t").add("member").unwrap();

        for key in ["s", "t"] {
            assert!(keys.expire_in(key, Duration::from_secs(600)).unwrap());
            let left = keys.ttl(key).unwrap().left().expect("a deadline");
            assert!(left <= Duration::from_secs(600) && left > Duration::from_secs(590));
        }

        assert_eq!(keys.kind("t").unwrap(), Some(Kind::Set), "still a set");
        assert_eq!(db.set("t").len().unwrap(), 1, "with its member");
    }

    #[test]
    fn the_three_answers_are_three_and_not_two() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();

        assert_eq!(keys.ttl("nothing").unwrap(), Ttl::Missing);
        assert!(!keys.ttl("nothing").unwrap().found());

        db.strings().set("k", "v").unwrap();
        assert_eq!(keys.ttl("k").unwrap(), Ttl::Forever);
        assert!(keys.ttl("k").unwrap().found());
        assert_eq!(keys.ttl("k").unwrap().left(), None, "forever has none left");

        keys.expire_in("k", Duration::from_secs(60)).unwrap();
        assert!(matches!(keys.ttl("k").unwrap(), Ttl::In(_)));
    }

    #[test]
    fn a_moment_that_has_gone_removes_the_key_now() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();

        assert!(keys.expire_at("k", UNIX_EPOCH).unwrap(), "it was applied");
        assert!(!keys.exists("k").unwrap(), "and applying it took the key");
    }

    #[test]
    fn a_deadline_comes_back_as_the_moment_it_was_set_to() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();
        assert_eq!(keys.deadline("k").unwrap(), None, "no deadline yet");

        let at = UNIX_EPOCH + Duration::from_millis(4_000_000_000_000);
        assert!(keys.expire_at("k", at).unwrap());
        assert_eq!(keys.deadline("k").unwrap(), Some(at));

        assert!(keys.persist("k").unwrap());
        assert_eq!(keys.deadline("k").unwrap(), None);
        assert!(
            !keys.persist("k").unwrap(),
            "there was nothing left to take"
        );
        assert!(keys.exists("k").unwrap(), "and the key is still here");
    }

    #[test]
    fn a_condition_decides_whether_the_deadline_moves() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();

        let hour = Duration::from_secs(3600);
        let day = Duration::from_secs(86400);

        assert!(keys.expire_in_when("k", hour, When::Unset).unwrap());
        assert!(
            !keys.expire_in_when("k", day, When::Unset).unwrap(),
            "taken"
        );
        assert!(keys.expire_in_when("k", day, When::AlreadySet).unwrap());
        assert!(!keys.expire_in_when("k", hour, When::Later).unwrap(), "in");
        assert!(keys.expire_in_when("k", hour, When::Earlier).unwrap());
        assert!(keys.expire_in_when("k", day, When::Later).unwrap());

        keys.persist("k").unwrap();
        assert!(
            !keys.expire_in_when("k", hour, When::Later).unwrap(),
            "no deadline is infinitely far out, so nothing is further"
        );
        assert!(
            keys.expire_in_when("k", hour, When::Earlier).unwrap(),
            "and by the same reading everything is nearer"
        );
        keys.persist("k").unwrap();
        assert!(
            !keys
                .expire_in_when("k", hour, When::EarlierAndAlreadySet)
                .unwrap(),
            "unless XX takes that reading away"
        );
    }

    #[test]
    fn a_condition_that_says_no_leaves_a_key_that_would_have_gone() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();
        keys.expire_in("k", Duration::from_secs(60)).unwrap();

        assert!(
            !keys.expire_at_when("k", UNIX_EPOCH, When::Unset).unwrap(),
            "the condition is checked before the moment is"
        );
        assert!(keys.exists("k").unwrap());
    }

    #[test]
    fn a_deadline_past_the_year_4199_is_refused_rather_than_clamped() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();

        let err = keys
            .expire_in("k", Duration::from_secs(u64::MAX))
            .unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        let far = UNIX_EPOCH + Duration::from_millis(MAX_AT + 1);
        assert_eq!(keys.expire_at("k", far).unwrap_err().code(), Code::Invalid);
        assert_eq!(keys.ttl("k").unwrap(), Ttl::Forever, "and nothing moved");
    }

    #[test]
    fn nothing_reads_the_clock_until_a_deadline_exists_to_read_it_for() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();

        keys.exists("k").unwrap();
        keys.kind("k").unwrap();
        keys.ttl("k").unwrap();
        assert!(!db.reads_the_clock(), "asking is not creating");

        keys.expire_in("k", Duration::from_secs(60)).unwrap();
        assert!(db.reads_the_clock());
    }

    #[test]
    fn a_walk_sees_every_key_whatever_it_holds() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        assert!(keys.all().unwrap().is_empty());
        assert_eq!(keys.random().unwrap(), None);

        db.strings().set("a", "v").unwrap();
        db.set("s").add("m").unwrap();
        for i in 0..1_000 {
            db.strings().set(format!("n:{i}"), "v").unwrap();
        }

        let mut all = keys.all().unwrap();
        all.sort();
        assert_eq!(all.len(), 1_002);
        assert_eq!(all[0], b"a");
        assert_eq!(keys.matching("n:*").unwrap().len(), 1_000);
        assert_eq!(keys.matching("s").unwrap(), vec![b"s".to_vec()]);
        assert!(keys.matching("nothing").unwrap().is_empty());

        // A random key is one of the keys, and not the same one every time.
        let mut picked = std::collections::HashSet::new();
        for _ in 0..100 {
            picked.insert(keys.random().unwrap().expect("the database is not empty"));
        }
        assert!(
            picked.len() > 5,
            "randomkey is stuck on {} keys",
            picked.len()
        );
        assert!(picked.iter().all(|k| all.contains(k)));
    }

    #[test]
    fn a_walk_does_not_hand_out_a_key_that_has_expired() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("alive", "v").unwrap();
        db.strings().set("dead", "v").unwrap();
        keys.expire_at("dead", UNIX_EPOCH + Duration::from_secs(1))
            .unwrap();

        assert_eq!(keys.all().unwrap(), vec![b"alive".to_vec()]);
        assert_eq!(keys.random().unwrap(), Some(b"alive".to_vec()));
    }

    /// The closure holds the database, so a call back into it from inside the
    /// walk is refused rather than deadlocked or, worse, allowed.
    #[test]
    fn a_walk_cannot_be_reentered() {
        let db = open(MEMORY).unwrap();
        let keys = db.keys();
        db.strings().set("k", "v").unwrap();

        let mut inner = Ok(true);
        keys.each(|_| inner = keys.exists("k")).unwrap();
        assert_eq!(inner.unwrap_err().code(), Code::Invalid);
    }
}
