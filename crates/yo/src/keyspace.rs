//! The Redis string keyspace, from the embedded side.
//!
//! This is Y23 where you can see it. A program that calls [`Strings::incr`]
//! reaches the same `yo_kv::Strings::incr` that `INCR` off a socket reaches,
//! not a second implementation that agrees with it today. The difference
//! between the two callers is a socket, a parser and a reply, and this side
//! pays none of them: an embedded `INCR` is a probe, an add and a store.
//!
//! It is also where inline execution mode (`15` section 7) actually happens.
//! The calling thread is the shard, so there is no queue, no message and no
//! wakeup, which is what makes the number in `bench/00` a number about the
//! store rather than about a channel.
//!
//! # About the clock
//!
//! `04` section 5 says the clock is read once per turn of the shard loop and
//! never on the data path, because a clock read is tens of nanoseconds against
//! a budget of a hundred and fifty. Inline mode has no loop, so one call is one
//! turn, and reading the clock per call would double the cost of a `GET`.
//!
//! So the clock is read only when its answer can be observed, which is when
//! some key in the keyspace has a deadline. A database that has never been
//! given one cannot have an expired key, and its clock never moves and is never
//! read. The first call that sets a deadline turns the reads on, and from then
//! on the cost is the same one a server pays per batch.

use std::time::Duration;

use yo_common::{Code, Error, Result};
use yo_kv::{Expire, SetOptions, Str};

use crate::db::Handle;

/// The keyspace every Redis string command works on.
///
/// Cheap to clone, and every clone is the same keyspace. Keys are byte strings
/// the way Redis's are, so anything that is bytes will do: `"hits"`, a
/// `String`, a `&[u8]` or a `Vec<u8>`.
///
/// ```
/// let db = yo::open(yo::MEMORY)?;
/// let keys = db.strings();
///
/// keys.set("greeting", "hello")?;
/// assert_eq!(keys.get("greeting")?.as_deref(), Some(&b"hello"[..]));
///
/// assert_eq!(keys.incr("hits")?, 1);
/// assert_eq!(keys.incr_by("hits", 9)?, 10);
/// # Ok::<(), yo::Error>(())
/// ```
#[derive(Clone)]
pub struct Strings {
    pub(crate) db: Handle,
}

impl core::fmt::Debug for Strings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Strings")
            .field("keys", &self.len().ok())
            .finish()
    }
}

impl Strings {
    /// Read a value.
    ///
    /// Owned, because most callers want the bytes to outlive the call.
    /// [`Strings::with`] is the same read without the copy.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn get(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        self.with(key, |v| v.to_vec())
    }

    /// Read a value without copying it, by handing what is in the record to
    /// `f`.
    ///
    /// The view is [`Str`], which is either the bytes where they lie or the
    /// integer an int encoded value holds. This is the read the G6 budget is
    /// about, and it allocates nothing at all.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let keys = db.strings();
    /// keys.set("greeting", "hello")?;
    ///
    /// assert_eq!(keys.with("greeting", |v| v.len())?, Some(5));
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn with<R>(
        &self,
        key: impl AsRef<[u8]>,
        f: impl FnOnce(Str<'_>) -> R,
    ) -> Result<Option<R>> {
        self.db
            .run(|inner| Ok(inner.strings.get(key.as_ref()).map(f)))
    }

    /// Whether a key is there and has not expired.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn exists(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        self.db.run(|inner| Ok(inner.strings.exists(key.as_ref())))
    }

    /// The length of a value in bytes, which is zero for a key that is not
    /// there. `STRLEN`.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn len_of(&self, key: impl AsRef<[u8]>) -> Result<usize> {
        self.db.run(|inner| Ok(inner.strings.strlen(key.as_ref())))
    }

    /// Store a value, clearing any deadline the key had. Plain `SET`.
    ///
    /// # Errors
    ///
    /// [`Code::Full`] for a value past [`yo_kv::STRING_MAX`].
    pub fn set(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<()> {
        self.db
            .run(|inner| inner.strings.set_plain(key.as_ref(), value.as_ref()))
    }

    /// Store a value only if the key is missing, and say whether it was.
    /// `SET NX`.
    ///
    /// # Errors
    ///
    /// As [`Strings::set`].
    pub fn set_if_missing(&self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<bool> {
        self.db
            .run(|inner| inner.strings.setnx(key.as_ref(), value.as_ref()))
    }

    /// Store a value that expires after `ttl`. `SET PX`.
    ///
    /// This is the call that turns the clock on for this database, because it
    /// is the first moment a deadline can be observed.
    ///
    /// # Errors
    ///
    /// As [`Strings::set`], and [`Code::Invalid`] for a `ttl` past what fits in
    /// a millisecond deadline.
    pub fn set_for(
        &self,
        key: impl AsRef<[u8]>,
        value: impl AsRef<[u8]>,
        ttl: Duration,
    ) -> Result<()> {
        let ms = u64::try_from(ttl.as_millis()).map_err(|_| too_far())?;
        self.db.deadlines(|inner| {
            let at = inner
                .strings
                .clock()
                .now_ms()
                .checked_add(ms)
                .ok_or_else(too_far)?;
            inner
                .strings
                .set(
                    key.as_ref(),
                    value.as_ref(),
                    SetOptions::PLAIN.expiring(Expire::At(at)),
                )
                .map(|_| ())
        })
    }

    /// How long a key has left, or `None` if it has no deadline or is not
    /// there. `PTTL`.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn ttl(&self, key: impl AsRef<[u8]>) -> Result<Option<Duration>> {
        self.db.run(|inner| {
            let now = inner.strings.clock().now_ms();
            Ok(inner
                .strings
                .expire_at(key.as_ref())
                .map(|at| Duration::from_millis(at.saturating_sub(now))))
        })
    }

    /// Store a value and hand back what was there. `GETSET`.
    ///
    /// # Errors
    ///
    /// As [`Strings::set`].
    pub fn replace(
        &self,
        key: impl AsRef<[u8]>,
        value: impl AsRef<[u8]>,
    ) -> Result<Option<Vec<u8>>> {
        self.db
            .run(|inner| inner.strings.getset(key.as_ref(), value.as_ref()))
    }

    /// Remove a key and hand back what it held. `GETDEL`.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn take(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        self.db.run(|inner| Ok(inner.strings.getdel(key.as_ref())))
    }

    /// Remove a key, and say whether it was there. `DEL`.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn del(&self, key: impl AsRef<[u8]>) -> Result<bool> {
        self.db.run(|inner| Ok(inner.strings.del(key.as_ref())))
    }

    /// Store several values, all of them or none. `MSET`.
    ///
    /// The pairs reach the store as an iterator rather than a `Vec`, which is
    /// the same thing the wire layer does and for the same reason: `MSET` is on
    /// the gate list and an API that forces an allocation to call it is the
    /// wrong API.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let keys = db.strings();
    ///
    /// keys.set_many(&[("a", "1"), ("b", "2")])?;
    /// assert_eq!(keys.get("b")?.as_deref(), Some(&b"2"[..]));
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Strings::set`], and nothing is written if any pair fails.
    pub fn set_many<K: AsRef<[u8]>, V: AsRef<[u8]>>(&self, pairs: &[(K, V)]) -> Result<()> {
        self.db.run(|inner| {
            inner
                .strings
                .mset(pairs.iter().map(|(k, v)| (k.as_ref(), v.as_ref())))
        })
    }

    /// Read several values in one call. `MGET`.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn get_many<K: AsRef<[u8]>>(&self, keys: &[K]) -> Result<Vec<Option<Vec<u8>>>> {
        self.db.run(|inner| {
            Ok(keys
                .iter()
                .map(|k| inner.strings.get(k.as_ref()).map(|v| v.to_vec()))
                .collect())
        })
    }

    /// Add one and hand back the result. `INCR`.
    ///
    /// A key that is not there counts as zero, which is Redis's rule and not a
    /// convenience: it is what makes a counter usable without a create step.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the value is not an integer, or when adding would
    /// leave the range of an `i64`.
    pub fn incr(&self, key: impl AsRef<[u8]>) -> Result<i64> {
        self.db.run(|inner| inner.strings.incr(key.as_ref()))
    }

    /// Add `by` and hand back the result. `INCRBY`, and `DECRBY` for a negative
    /// `by`.
    ///
    /// # Errors
    ///
    /// As [`Strings::incr`].
    pub fn incr_by(&self, key: impl AsRef<[u8]>, by: i64) -> Result<i64> {
        self.db.run(|inner| inner.strings.incrby(key.as_ref(), by))
    }

    /// Subtract one and hand back the result. `DECR`.
    ///
    /// # Errors
    ///
    /// As [`Strings::incr`].
    pub fn decr(&self, key: impl AsRef<[u8]>) -> Result<i64> {
        self.db.run(|inner| inner.strings.decr(key.as_ref()))
    }

    /// Add `by` to a float counter and hand back the result. `INCRBYFLOAT`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the value is not a float, or when the result
    /// would be infinite or not a number.
    pub fn incr_by_float(&self, key: impl AsRef<[u8]>, by: f64) -> Result<f64> {
        self.db
            .run(|inner| inner.strings.incrbyfloat(key.as_ref(), by))
    }

    /// Append to a value and hand back its new length. `APPEND`.
    ///
    /// # Errors
    ///
    /// As [`Strings::set`].
    pub fn append(&self, key: impl AsRef<[u8]>, tail: impl AsRef<[u8]>) -> Result<usize> {
        self.db
            .run(|inner| inner.strings.append(key.as_ref(), tail.as_ref()))
    }

    /// How many keys the keyspace holds, expired ones that nothing has touched
    /// yet included. `DBSIZE`.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn len(&self) -> Result<usize> {
        self.db.run(|inner| Ok(inner.strings.len()))
    }

    /// Whether the keyspace is empty.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn is_empty(&self) -> Result<bool> {
        self.db.run(|inner| Ok(inner.strings.is_empty()))
    }

    /// Keys reclaimed by running into them after their deadline.
    ///
    /// # Errors
    ///
    /// As [`Strings::get`].
    pub fn expired_keys(&self) -> Result<u64> {
        self.db.run(|inner| Ok(inner.strings.expired_keys()))
    }
}

fn too_far() -> Error {
    Error::new(
        Code::Invalid,
        "that deadline is further away than a millisecond timestamp reaches",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MEMORY, open};

    #[test]
    fn the_string_commands_are_the_ones_a_redis_client_would_send() {
        let db = open(MEMORY).unwrap();
        let keys = db.strings();

        keys.set("greeting", "hello").unwrap();
        assert_eq!(
            keys.get("greeting").unwrap().as_deref(),
            Some(&b"hello"[..])
        );
        assert_eq!(keys.len_of("greeting").unwrap(), 5);
        assert_eq!(keys.append("greeting", " there").unwrap(), 11);
        assert!(keys.exists("greeting").unwrap());

        assert!(!keys.set_if_missing("greeting", "other").unwrap());
        assert!(keys.set_if_missing("fresh", "yes").unwrap());

        assert_eq!(
            keys.replace("greeting", "hi").unwrap().as_deref(),
            Some(&b"hello there"[..])
        );
        assert_eq!(keys.take("greeting").unwrap().as_deref(), Some(&b"hi"[..]));
        assert!(!keys.exists("greeting").unwrap());
        assert!(!keys.del("greeting").unwrap());
    }

    #[test]
    fn a_counter_starts_at_zero_without_being_created() {
        let db = open(MEMORY).unwrap();
        let keys = db.strings();

        assert_eq!(keys.incr("hits").unwrap(), 1);
        assert_eq!(keys.incr_by("hits", 9).unwrap(), 10);
        assert_eq!(keys.decr("hits").unwrap(), 9);
        assert_eq!(keys.incr_by_float("ratio", 1.5).unwrap(), 1.5);

        keys.set("word", "nope").unwrap();
        let e = keys.incr("word").expect_err("that is not a number");
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), "value is not an integer or out of range");
    }

    #[test]
    fn many_keys_at_once_go_in_and_come_out_together() {
        let db = open(MEMORY).unwrap();
        let keys = db.strings();

        keys.set_many(&[("a", "1"), ("b", "2"), ("c", "3")])
            .unwrap();
        let got = keys.get_many(&["a", "c", "missing"]).unwrap();
        assert_eq!(got[0].as_deref(), Some(&b"1"[..]));
        assert_eq!(got[1].as_deref(), Some(&b"3"[..]));
        assert_eq!(got[2], None);
        assert_eq!(keys.len().unwrap(), 3);
        assert!(!keys.is_empty().unwrap());
    }

    /// The clock policy: a keyspace with no deadline in it never reads the
    /// clock, and the first `set_for` is what turns the reads on.
    #[test]
    fn a_deadline_is_what_makes_time_start_moving() {
        let db = open(MEMORY).unwrap();
        let keys = db.strings();

        keys.set("plain", "v").unwrap();
        assert_eq!(keys.ttl("plain").unwrap(), None);
        assert!(!db.reads_the_clock());

        keys.set_for("short", "v", Duration::from_millis(50))
            .unwrap();
        assert!(db.reads_the_clock());
        assert!(keys.ttl("short").unwrap().unwrap() <= Duration::from_millis(50));

        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(keys.get("short").unwrap(), None);
        assert_eq!(keys.expired_keys().unwrap(), 1);
        // The key with no deadline is not affected by any of that.
        assert_eq!(keys.get("plain").unwrap().as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn a_key_is_bytes_and_not_only_text() {
        let db = open(MEMORY).unwrap();
        let keys = db.strings();

        keys.set(b"\x00\xff", vec![1u8, 2, 3]).unwrap();
        assert_eq!(
            keys.get(b"\x00\xff").unwrap().as_deref(),
            Some(&[1u8, 2, 3][..])
        );
        assert_eq!(keys.with(b"\x00\xff", |v| v.len()).unwrap(), Some(3));
        assert!(format!("{keys:?}").contains("keys"));
    }
}
