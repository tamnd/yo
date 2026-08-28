//! The coarse clock expiry compares against.
//!
//! `04` section 5 is explicit about this: there is no global clock read on the
//! data path. The shard reads the clock once per turn of the loop and every
//! command in that batch of 64 compares against the same number. A clock read
//! is a vDSO call, so it is not a syscall, but it is still tens of nanoseconds
//! against a budget of a hundred and fifty for the whole command, and paying it
//! per command would mean paying it 64 times for one answer that did not change.
//!
//! `TIME` and `EXPIRETIME` read the fine clock instead, because their contract
//! is to report the time and not to compare against it. That is what
//! [`Clock::fine_now_ms`] is for and it is the only thing that should call it.
//!
//! A fixed clock is not a testing convenience bolted on the side. Expiry is the
//! one part of a database whose behaviour is a function of the wall clock, and a
//! test that sleeps to move time forward is a test that is slow and flaky at the
//! same time. Every expiry test in this crate drives a fixed clock instead.

use std::time::{SystemTime, UNIX_EPOCH};

/// Where a clock takes its readings from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// The operating system, read on every [`Clock::refresh`].
    System,
    /// Whatever the owner last set, and nothing else.
    Fixed,
}

/// A millisecond clock that only moves when it is told to.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    now_ms: u64,
    source: Source,
}

impl Clock {
    /// A clock that follows the system, read once now.
    pub fn system() -> Clock {
        Clock {
            now_ms: Clock::fine_now_ms(),
            source: Source::System,
        }
    }

    /// A clock that reads `ms` until somebody moves it.
    pub const fn fixed(ms: u64) -> Clock {
        Clock {
            now_ms: ms,
            source: Source::Fixed,
        }
    }

    /// The current reading, in milliseconds since the unix epoch.
    #[inline]
    pub const fn now_ms(&self) -> u64 {
        self.now_ms
    }

    /// Take a new reading, which a system clock does from the operating system
    /// and a fixed clock does not do at all.
    ///
    /// Called once per turn of the shard loop, from the maintenance slice.
    #[inline]
    pub fn refresh(&mut self) {
        if self.source == Source::System {
            self.now_ms = Clock::fine_now_ms();
        }
    }

    /// Move the clock to `ms` by hand.
    ///
    /// On a system clock the next [`Clock::refresh`] will overwrite this, so it
    /// is only meaningful on a fixed one.
    #[inline]
    pub const fn set(&mut self, ms: u64) {
        self.now_ms = ms;
    }

    /// Move a clock forward by `ms`.
    #[inline]
    pub const fn advance(&mut self, ms: u64) {
        self.now_ms = self.now_ms.saturating_add(ms);
    }

    /// Read the operating system's clock right now.
    ///
    /// A time before the unix epoch reads as zero rather than failing. There is
    /// nothing useful a database can do about a machine whose clock says 1969,
    /// and every key expiring immediately is a more honest outcome than a panic
    /// on a path that has no error to return.
    #[inline]
    pub fn fine_now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    }
}

impl Default for Clock {
    fn default() -> Clock {
        Clock::system()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixed_clock_stays_where_it_is_put() {
        let mut c = Clock::fixed(1_000);
        assert_eq!(c.now_ms(), 1_000);
        c.refresh();
        assert_eq!(c.now_ms(), 1_000, "refresh moved a fixed clock");
        c.advance(500);
        assert_eq!(c.now_ms(), 1_500);
        c.set(7);
        assert_eq!(c.now_ms(), 7);
    }

    #[test]
    fn a_system_clock_reads_a_plausible_time() {
        let c = Clock::system();
        // 2020-01-01, which this build is comfortably after.
        assert!(c.now_ms() > 1_577_836_800_000, "clock read {}", c.now_ms());
    }

    #[test]
    fn a_system_clock_does_not_move_until_it_is_refreshed() {
        let mut c = Clock::system();
        let first = c.now_ms();
        // Busy work rather than a sleep, because the point is that the reading
        // is stable across it and a sleep would only make the test slow.
        let mut spin = 0u64;
        for i in 0..200_000u64 {
            spin = spin.wrapping_add(i);
        }
        assert_eq!(c.now_ms(), first, "the clock moved on its own {spin}");
        c.refresh();
        assert!(c.now_ms() >= first);
    }

    #[test]
    fn advancing_past_the_end_of_time_stops_there() {
        let mut c = Clock::fixed(u64::MAX - 1);
        c.advance(10);
        assert_eq!(c.now_ms(), u64::MAX);
    }
}
