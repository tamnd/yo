//! Whether the lookups happening on this thread are a client reading a key.
//!
//! `INFO stats` carries two numbers most dashboards watching a server are built
//! on. `keyspace_hits` is the reads that found what they went looking for and
//! `keyspace_misses` is the reads that did not, and the hit rate everyone quotes
//! is the first over the sum of the two. What makes the pair worth anything is
//! the word read. A `SET` landing on a name nobody was using is not a cache
//! miss, and neither is the `RPUSH` that makes a list, so a count of every
//! lookup the server does would be a ratio that means nothing.
//!
//! Nothing under here knows which kind it is looking at.
//! [`super::keyspace::Keyspace`] has one set of lookups and both kinds of
//! command come through them: `GET` and `APPEND` ask the map for the same key in
//! the same way, and what tells them apart is upstairs in the command table.
//!
//! # Why a thread local
//!
//! Because the alternative is an argument on every lookup, and the lookups are
//! reached through a few hundred typed methods rather than called directly.
//! Threading a flag down through `Keyspace::get` and `Keyspace::sadd` and all the
//! rest of them, so that the funnel underneath can add one to a counter, would
//! be a parameter on the whole API for the sake of a statistic.
//!
//! So the layer that does know says it once on the thread it is about to run the
//! command on, the same way [`super::news`] installs its hook, and the funnels
//! read a flag. That is a thread local read and a test on a path that has
//! already hashed a key and walked a bucket for it.
//!
//! # Why it is a setting and not a count
//!
//! The counters themselves sit on the keyspace the lookup went to, beside
//! `expired_keys` and for the same reason: the stripe is already held and the
//! line is already in this core's cache, so the count is an add to a field, and
//! a shared counter would be a line every thread on the server has to take
//! ownership of to write to.

use core::cell::Cell;

thread_local! {
    /// Whether what this thread is doing now is a client reading a key.
    static READING: Cell<bool> = const { Cell::new(false) };
}

/// Count the lookups on this thread as a client's reads, or stop counting them,
/// until the answer is dropped.
///
/// The setting that was there comes back at the end rather than the one the
/// thread started on, so a command running inside another one leaves the outer
/// one where it was. `EXEC` and `EVAL` are both of those: the commands inside
/// them arm this each in turn, and what is wrapped around them is a write in one
/// case and a read in the other, neither of which is what the commands inside
/// should be counted as.
#[must_use = "counting stops when this is dropped, so dropping it here counts nothing"]
pub fn reading(on: bool) -> Reading {
    Reading(READING.replace(on))
}

/// Stop counting until the answer is dropped, for a lookup that is not one of
/// the command's own.
///
/// A real server counts in `lookupKey` and reaches it once per key it was sent,
/// so what it counts is the command's key list. This one reaches its own lookups
/// as many times as its shape needs: `ZRANGE` asks for the window and then walks
/// it, `GEOSEARCH` asks whether the key is there, then where the centre member
/// is, then reads it, and a command that reads a key and then writes it looks it
/// up on the way to both. Redis's answer to the second half of that is the
/// `LOOKUP_WRITE` flag, which turns off the counters and the miss notification
/// together, and it has no need of an answer to the first half.
///
/// So this is that flag and the rest of it: the lookups it covers are either the
/// same key over again or a key on the way to being written, and neither is
/// something a hit rate should have in it. It goes after the lookup that does
/// count, so what is inside it is everything that follows.
#[must_use = "counting comes back when this is dropped, so dropping it here quiets nothing"]
pub fn quiet() -> Reading {
    reading(false)
}

/// What [`reading`] hands back, which puts the thread's setting back when it
/// goes.
#[derive(Debug)]
pub struct Reading(bool);

impl Drop for Reading {
    fn drop(&mut self) {
        READING.set(self.0);
    }
}

/// Whether a lookup happening now is part of a client's read.
pub(crate) fn is_reading() -> bool {
    READING.get()
}

#[cfg(test)]
mod tests {
    use super::{is_reading, reading};

    #[test]
    fn a_thread_counts_nothing_until_it_is_asked_to() {
        assert!(!is_reading());
    }

    #[test]
    fn the_setting_that_was_there_comes_back() {
        assert!(!is_reading());
        {
            let _read = reading(true);
            assert!(is_reading());
            {
                let _write = reading(false);
                assert!(!is_reading());
            }
            assert!(is_reading());
        }
        assert!(!is_reading());
    }
}
