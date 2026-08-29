//! `Counter`, a handle at one key (`15` section 2).

use yo_common::{Code, Error, Result};

use crate::db::Handle;

/// A counter at one key.
///
/// Sugar over [`crate::keyspace::Strings`], and the kind of sugar that is worth
/// having: counting is the commonest thing a string key is used for, and a
/// handle that holds the key means the key is spelled once instead of at every
/// call site, which is one fewer place for a typo to live.
///
/// A counter that has never been written reads as zero, the way Redis's does,
/// so there is no create step and nothing to check before the first `incr`.
///
/// ```
/// let db = yo::open(yo::MEMORY)?;
/// let hits = db.counter("hits");
///
/// assert_eq!(hits.get()?, 0);
/// assert_eq!(hits.incr()?, 1);
/// assert_eq!(hits.add(41)?, 42);
/// # Ok::<(), yo::Error>(())
/// ```
#[derive(Clone)]
pub struct Counter {
    pub(crate) db: Handle,
    pub(crate) key: Vec<u8>,
}

impl core::fmt::Debug for Counter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Counter")
            .field("key", &String::from_utf8_lossy(&self.key))
            .field("value", &self.get().ok())
            .finish()
    }
}

impl Counter {
    /// The key this counter lives at.
    #[must_use]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// What the counter reads, which is zero if nothing has written it yet.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the key holds something that is not an integer.
    pub fn get(&self) -> Result<i64> {
        self.db.run(|inner| match inner.strings.get(&self.key)? {
            Some(v) => v.as_int().ok_or_else(not_a_number),
            None => Ok(0),
        })
    }

    /// Add one and hand back the result. `INCR`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if the key holds something that is not an integer, or
    /// if the result would leave the range of an `i64`.
    pub fn incr(&self) -> Result<i64> {
        self.db.run(|inner| inner.strings.incr(&self.key))
    }

    /// Subtract one and hand back the result. `DECR`.
    ///
    /// # Errors
    ///
    /// As [`Counter::incr`].
    pub fn decr(&self) -> Result<i64> {
        self.db.run(|inner| inner.strings.decr(&self.key))
    }

    /// Add `by` and hand back the result, which is `DECRBY` for a negative
    /// `by`. `INCRBY`.
    ///
    /// # Errors
    ///
    /// As [`Counter::incr`].
    pub fn add(&self, by: i64) -> Result<i64> {
        self.db.run(|inner| inner.strings.incrby(&self.key, by))
    }

    /// Put the counter at `value`, whatever it read before. `SET`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn set(&self, value: i64) -> Result<()> {
        let mut buf = itoa_buf();
        let text = write_i64(&mut buf, value);
        self.db
            .run(|inner| inner.strings.set_plain(&self.key, text))
    }

    /// Remove the key, so the counter reads zero again and stops taking up
    /// room. `DEL`.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] if called from inside a callback that is already
    /// holding this database.
    pub fn reset(&self) -> Result<()> {
        self.db.run(|inner| {
            inner.strings.del(&self.key);
            Ok(())
        })
    }
}

fn not_a_number() -> Error {
    Error::new(Code::Invalid, "value is not an integer or out of range")
}

/// Room for the digits of any `i64`, sign included.
const fn itoa_buf() -> [u8; 20] {
    [0; 20]
}

/// The digits of `n`, written into `buf` rather than into a `String`, because a
/// counter that allocates to set itself is a counter nobody would put on a hot
/// path.
fn write_i64(buf: &mut [u8; 20], n: i64) -> &[u8] {
    use core::fmt::Write;

    struct Cursor<'a> {
        buf: &'a mut [u8; 20],
        at: usize,
    }
    impl Write for Cursor<'_> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let end = self.at + s.len();
            self.buf[self.at..end].copy_from_slice(s.as_bytes());
            self.at = end;
            Ok(())
        }
    }

    let mut cursor = Cursor { buf, at: 0 };
    // Writing an i64 into twenty bytes cannot fail, and the formatter's error
    // type carries nothing to report anyway.
    let _ = write!(cursor, "{n}");
    let at = cursor.at;
    &buf[..at]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MEMORY, open};

    #[test]
    fn a_counter_counts_without_being_created_first() {
        let db = open(MEMORY).unwrap();
        let hits = db.counter("hits");

        assert_eq!(hits.get().unwrap(), 0);
        assert_eq!(hits.incr().unwrap(), 1);
        assert_eq!(hits.add(9).unwrap(), 10);
        assert_eq!(hits.decr().unwrap(), 9);
        assert_eq!(hits.add(-9).unwrap(), 0);
        assert_eq!(hits.key(), b"hits");
    }

    #[test]
    fn setting_and_resetting_go_through_the_same_key_a_client_would_see() {
        let db = open(MEMORY).unwrap();
        let hits = db.counter("hits");

        hits.set(i64::MIN).unwrap();
        assert_eq!(hits.get().unwrap(), i64::MIN);
        assert_eq!(
            db.strings().get("hits").unwrap().as_deref(),
            Some(i64::MIN.to_string().as_bytes())
        );

        hits.set(41).unwrap();
        assert_eq!(hits.incr().unwrap(), 42);

        hits.reset().unwrap();
        assert_eq!(hits.get().unwrap(), 0);
        assert!(!db.strings().exists("hits").unwrap());
    }

    /// Two handles on one key are one counter, and the keyspace sees the same
    /// number, because there is one store underneath and not two.
    #[test]
    fn two_handles_on_a_key_are_one_counter() {
        let db = open(MEMORY).unwrap();
        let a = db.counter("hits");
        let b = db.counter(b"hits".to_vec());

        a.incr().unwrap();
        b.incr().unwrap();
        assert_eq!(a.get().unwrap(), 2);
        assert_eq!(db.strings().incr("hits").unwrap(), 3);
        assert_eq!(a.clone().get().unwrap(), 3);
        assert!(format!("{a:?}").contains("hits"));
    }

    #[test]
    fn a_key_holding_words_is_not_a_counter() {
        let db = open(MEMORY).unwrap();
        db.strings().set("hits", "lots").unwrap();
        let hits = db.counter("hits");

        let e = hits.get().expect_err("that is not a number");
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), "value is not an integer or out of range");
        assert_eq!(hits.incr().unwrap_err().code(), Code::Invalid);
    }
}
