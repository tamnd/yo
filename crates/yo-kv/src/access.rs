//! How recently a key was used, and how often, in twenty four bits.
//!
//! Eviction has to pick a victim, and the ten policies pick one by asking one of
//! three questions about every candidate: when was this last read, when was it
//! last written, or how often is it read. Redis answers all three out of a single
//! twenty four bit field on the object, reading it one way under an LFU policy
//! and the other way under everything else, and this is that field.
//!
//! ```text
//! clock  +-------------------------------------------+
//!        | seconds since the epoch, low 24 bits      |
//!        +-------------------------------------------+
//!
//! LFU    +---------------------------+---------------+
//!        | minutes, low 16 bits      | counter, u8   |
//!        +---------------------------+---------------+
//! ```
//!
//! The two readings share the field because a key is only ever under one policy
//! at a time, and a server that switches policy at runtime is one where the old
//! reading is garbage under the new one. Redis says so in the error text on
//! `OBJECT FREQ`, which tells the operator that switching will take some time to
//! adjust, and that sentence is a description of exactly this.
//!
//! # Least recently modified
//!
//! There are ten policies rather than the eight most people can name.
//! `volatile-lrm` and `allkeys-lrm` arrived in 8.8 and they are the reason the
//! top box above is labelled clock rather than LRU. They store the same seconds
//! in the same bits and are read by the same subtraction. The only difference is
//! when the field is written: least recently used stamps it on every lookup,
//! least recently modified stamps it only when the value changes, so a key that
//! is read a million times and written once is a good victim under LRM and a bad
//! one under LRU.
//!
//! The thing worth writing down is that the clock is kept under all eight of the
//! non LFU policies and not just under the two LRU ones. Redis stamps it on every
//! lookup under `noeviction` and under the random policies too, which is why
//! `OBJECT IDLETIME` gives a real answer on a default server that is never going
//! to evict anything. Assuming otherwise is easy and it makes `OBJECT IDLETIME`
//! answer zero forever.
//!
//! # Why the arithmetic is copied rather than improved
//!
//! Both readings wrap, and both wrap in ways that a fresh design would not
//! choose. The LRU clock is twenty four bits of seconds, so it goes round every
//! hundred and ninety four days, and the idle time calculation has a branch in
//! it whose whole job is to give a sane answer across that wrap. The LFU clock
//! is sixteen bits of minutes and goes round every forty five days.
//!
//! None of that is copied out of admiration. `OBJECT IDLETIME` and `OBJECT FREQ`
//! are in the Redis test suite, the numbers they return are asserted on, and a
//! counter that is off by one against Redis's is a compatibility bug rather than
//! a rounding difference. So the constants here are Redis's constants and the
//! branches here are Redis's branches, and the places where that produces an odd
//! answer are marked as such rather than fixed.
//!
//! # What is different
//!
//! The randomness is passed in rather than drawn here. Redis calls `rand()`
//! inside its increment, which makes the function untestable and makes two
//! servers replaying the same commands disagree. Here the caller hands over the
//! generator it already owns, which is the shard's, so the whole thing is a pure
//! function of its inputs and a test can assert on a specific counter after a
//! specific number of accesses.

use yo_common::rng::Rng;

/// Bits in the field. Everything above these is not ours to write.
const BITS: u32 = 24;

/// The largest value the field holds, which is what the LRU clock wraps at.
const MAX: u32 = (1 << BITS) - 1;

/// Milliseconds per tick of the LRU clock.
///
/// A second, which is the resolution `OBJECT IDLETIME` reports in anyway. It is
/// also what makes the twenty four bits last a hundred and ninety four days
/// instead of four and a half hours.
const LRU_RESOLUTION_MS: u64 = 1000;

/// What a key's counter starts at under an LFU policy.
///
/// Five, and it is not zero for a reason worth writing down. A key that has just
/// been created has been accessed once, and starting it at zero would make it
/// the most attractive victim in the database at the moment it was written,
/// which means a fresh write could be evicted before the client that made it
/// ever read it back. Five gives it enough of a floor to survive the next few
/// sampling rounds and be judged on its actual traffic.
pub const LFU_INIT: u8 = 5;

/// How sharply the counter's growth flattens out.
///
/// The default is ten, which is Redis's. A larger number means more accesses are
/// needed to move the counter, so the counter covers a wider range of traffic in
/// the same eight bits.
pub const LFU_LOG_FACTOR: u32 = 10;

/// Minutes without an access before the counter comes down by one.
///
/// One, which is Redis's default. Zero turns decay off entirely and makes the
/// counter a lifetime total, which sounds appealing and is not: a key that was
/// hot last Tuesday would outrank one that is hot now, forever.
pub const LFU_DECAY_MINUTES: u32 = 1;

/// The low sixteen bits, which is where the LFU clock lives.
const LFU_TIME_MAX: u32 = 0xffff;

/// The two knobs an LFU policy has, which are `lfu-log-factor` and
/// `lfu-decay-time`.
///
/// They travel together because neither means anything on its own. The factor
/// decides how many accesses it takes to climb and the decay decides how fast
/// the climb is given back, so it is the pair that sets what the counter
/// measures, and passing one without the other is how they drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lfu {
    /// How sharply the counter's growth flattens out. See [`LFU_LOG_FACTOR`].
    pub log_factor: u32,
    /// Minutes of quiet per step down. See [`LFU_DECAY_MINUTES`].
    pub decay_minutes: u32,
}

impl Lfu {
    /// Redis's defaults, which are a factor of ten and a decay of one minute.
    pub const DEFAULT: Lfu = Lfu {
        log_factor: LFU_LOG_FACTOR,
        decay_minutes: LFU_DECAY_MINUTES,
    };
}

impl Default for Lfu {
    fn default() -> Lfu {
        Lfu::DEFAULT
    }
}

/// What a server does when it runs out of room, and therefore which reading of
/// [`Access`] is the live one.
///
/// The ten are Redis's ten and the names are the strings `maxmemory-policy`
/// takes. They vary along two axes that are worth separating, because most of
/// the code downstream only cares about one of them: which keys are eligible,
/// and how a victim is chosen from among them.
///
/// The `volatile` half only considers keys that have a deadline, which is the
/// setting for a server holding a cache and a working set in the same database.
/// The trap in it is that a `volatile` policy on a database where nothing has a
/// TTL cannot evict anything at all, so it behaves as [`Policy::NoEviction`] and
/// starts refusing writes, and that surprises people often enough that it is
/// worth saying here.
///
/// Ten and not the eight everyone knows. `volatile-lrm` and `allkeys-lrm` are
/// least recently modified, they are in 8.8 and therefore in the version we
/// claim to be, and they are easy to miss because most of what is written about
/// Redis eviction predates them. They share the clock with the LRU pair and
/// differ in one rule: a read does not move it. See [`Policy::stamps_on_read`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    /// Evict nothing and refuse the write instead. Redis's default, and this
    /// crate's, because losing data silently is not a default anyone should get
    /// without asking.
    #[default]
    NoEviction,
    /// Any key, least recently used first.
    AllKeysLru,
    /// Any key, least frequently used first.
    AllKeysLfu,
    /// Any key, chosen at random.
    AllKeysRandom,
    /// Any key, least recently modified first.
    AllKeysLrm,
    /// Keys with a deadline, least recently used first.
    VolatileLru,
    /// Keys with a deadline, least frequently used first.
    VolatileLfu,
    /// Keys with a deadline, chosen at random.
    VolatileRandom,
    /// Keys with a deadline, soonest to expire first.
    VolatileTtl,
    /// Keys with a deadline, least recently modified first.
    VolatileLrm,
}

impl Policy {
    /// The string `CONFIG GET maxmemory-policy` reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Policy::NoEviction => "noeviction",
            Policy::AllKeysLru => "allkeys-lru",
            Policy::AllKeysLfu => "allkeys-lfu",
            Policy::AllKeysRandom => "allkeys-random",
            Policy::AllKeysLrm => "allkeys-lrm",
            Policy::VolatileLru => "volatile-lru",
            Policy::VolatileLfu => "volatile-lfu",
            Policy::VolatileRandom => "volatile-random",
            Policy::VolatileTtl => "volatile-ttl",
            Policy::VolatileLrm => "volatile-lrm",
        }
    }

    /// Every policy, in the order Redis lists them.
    ///
    /// The order is not ours to pick. `CONFIG SET maxmemory-policy garbage`
    /// fails with a message that names the legal values, and a client comparing
    /// that message against a real server compares the whole string, so the
    /// order in `config.c` is the order here.
    pub const ALL: [Policy; 10] = [
        Policy::VolatileLru,
        Policy::VolatileLfu,
        Policy::VolatileRandom,
        Policy::VolatileTtl,
        Policy::VolatileLrm,
        Policy::AllKeysLru,
        Policy::AllKeysLfu,
        Policy::AllKeysRandom,
        Policy::AllKeysLrm,
        Policy::NoEviction,
    ];

    /// The policy a `CONFIG SET maxmemory-policy` argument names.
    ///
    /// Case insensitive, because `CONFIG SET` is everywhere else and a client
    /// that sends `ALLKEYS-LRU` is not wrong.
    #[must_use]
    pub fn parse(s: &[u8]) -> Option<Policy> {
        Policy::ALL
            .into_iter()
            .find(|p| s.eq_ignore_ascii_case(p.name().as_bytes()))
    }

    /// Whether only keys with a deadline are eligible.
    #[must_use]
    pub const fn volatile_only(self) -> bool {
        matches!(
            self,
            Policy::VolatileLru
                | Policy::VolatileLfu
                | Policy::VolatileRandom
                | Policy::VolatileTtl
                | Policy::VolatileLrm
        )
    }

    /// Whether the access field is being read as a frequency counter.
    ///
    /// This is the question `OBJECT FREQ` asks before it answers, because under
    /// any other policy the bits hold something else and reporting them as a
    /// frequency would be reporting a number that means nothing.
    #[must_use]
    pub const fn is_lfu(self) -> bool {
        matches!(self, Policy::AllKeysLfu | Policy::VolatileLfu)
    }

    /// Whether a victim is picked by how recently the key was used.
    #[must_use]
    pub const fn is_lru(self) -> bool {
        matches!(self, Policy::AllKeysLru | Policy::VolatileLru)
    }

    /// Whether a victim is picked by how recently the key was written.
    ///
    /// The same clock as [`Policy::is_lru`] read the same way. The pair differ
    /// only in when the clock is set, which is [`Policy::stamps_on_read`].
    #[must_use]
    pub const fn is_lrm(self) -> bool {
        matches!(self, Policy::AllKeysLrm | Policy::VolatileLrm)
    }

    /// Whether a victim is picked by a fair draw and nothing else.
    ///
    /// The pair that has no ordering to approximate, which is why the eviction
    /// pool skips them: keeping candidates between rounds is how a sampled
    /// policy gets closer to the true worst key, and under these two every
    /// eligible key already is the answer.
    #[must_use]
    pub const fn is_random(self) -> bool {
        matches!(self, Policy::AllKeysRandom | Policy::VolatileRandom)
    }

    /// Whether the access field holds a clock, which is the question `OBJECT
    /// IDLETIME` asks before it answers.
    ///
    /// True for eight of the ten. Only an LFU policy packs something else in
    /// there, and reporting those bits as an idle time would be reporting a
    /// number that means nothing.
    #[must_use]
    pub const fn is_clock(self) -> bool {
        !self.is_lfu()
    }

    /// Whether reading a key writes the access field back to it.
    ///
    /// True for eight of the ten, which is not the answer this had before and
    /// is the answer Redis gives. It is tempting to think `noeviction` and the
    /// random policies have nothing to maintain on a read, and that is true of
    /// eviction and false of the field: Redis stamps the clock on every lookup
    /// under all of them, which is why `OBJECT IDLETIME` tells the truth on a
    /// default server that will never evict anything.
    ///
    /// The two LRM policies are the exception, and they are the whole reason
    /// they exist. Least recently modified wants the clock to say when the value
    /// was last written, so a read that moved it would erase the only thing the
    /// policy is measuring.
    #[must_use]
    pub const fn stamps_on_read(self) -> bool {
        !self.is_lrm()
    }

    /// Whether writing a key writes the access field back to it.
    ///
    /// True for all ten, by two different routes. Under LRM it is the point.
    /// Under the other eight a write resolves the key first and that resolve is
    /// a read like any other, so the stamp has already happened by the time the
    /// value changes.
    #[must_use]
    pub const fn stamps_on_write(self) -> bool {
        true
    }
}

/// How recently and how often one key has been used.
///
/// Which of the two it is depends on the [`Policy`] in force, and this type does
/// not know which that is. The reader picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Access(u32);

impl Access {
    /// The field as stored, which is always inside twenty four bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0 & MAX
    }

    /// Rebuild one from bits that came out of a record.
    ///
    /// Anything above the low twenty four is dropped rather than trusted,
    /// because those bits belong to whatever is packed alongside.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u32) -> Access {
        Access(bits & MAX)
    }

    /// Whether this key has never been stamped.
    ///
    /// All zeroes, which is what a record carries from the moment it is written
    /// until something reads it under a policy that cares. It is a sentinel and
    /// not a reading, and both readings answer for it as though the key had just
    /// been used: zero seconds idle, and the starting frequency. That is the
    /// safe direction. The other one would make every key in the database the
    /// most attractive victim available for as long as it went unread, which
    /// would evict the working set the moment a policy was switched on.
    ///
    /// It is a sentinel rather than a flag bit because it costs nothing and
    /// because it means a record can be created without anybody deciding what to
    /// put here. The writers do not know the clock or the policy, and having
    /// them ask would have put both through every call site that makes a record.
    ///
    /// Zero is very nearly unreachable as a real reading. An LRU stamp is zero
    /// only in the first second of 1970. An LFU stamp is zero only for a key
    /// already decayed to nothing that is touched during the one minute in every
    /// forty five days when the LFU clock wraps, and the cost of the collision
    /// is that the key looks freshly used for a moment instead of unused. That
    /// is a rounding error in a heuristic, and it is worth it to keep the write
    /// path from having to care.
    #[inline]
    #[must_use]
    pub const fn is_unset(self) -> bool {
        self.bits() == 0
    }

    /// The reading for a key touched at `now_ms` under an LRU policy.
    #[inline]
    #[must_use]
    pub const fn lru(now_ms: u64) -> Access {
        Access(clock_at(now_ms))
    }

    /// The reading for a key created at `now_ms` under an LFU policy.
    #[inline]
    #[must_use]
    pub const fn lfu(now_ms: u64) -> Access {
        Access::pack(minutes_at(now_ms), LFU_INIT)
    }

    /// How long ago this key was read, in seconds, under an LRU policy.
    ///
    /// This is what `OBJECT IDLETIME` returns. The branch is the wrap: once the
    /// clock has gone round, a key stamped before the wrap holds a number larger
    /// than the clock does, and subtracting the wrong way round would report a
    /// key that was read a second ago as a hundred and ninety four days idle,
    /// which under `allkeys-lru` would evict the hottest key in the database.
    ///
    /// The wrapped arm is short by one second, because the period is `MAX + 1`
    /// and Redis subtracts from `MAX`. That is not a mistake here, it is Redis's
    /// mistake reproduced on purpose, and it is worth being clear about because
    /// it looks exactly like the kind of thing somebody would tidy up. Fixing it
    /// would make `OBJECT IDLETIME` disagree with Redis by a second for the keys
    /// that were stamped before a wrap, once every hundred and ninety four days.
    ///
    /// It is also self consistent over there, which is the part that settles it.
    /// `RESTORE` takes an idle time and turns it back into a stamp, and it adds
    /// `MAX` where this subtracts `MAX`, so a value that goes out through
    /// `OBJECT IDLETIME` and comes back in through `RESTORE` lands on the number
    /// it started from. Correcting one end here would break that round trip
    /// against a real Redis without making any single answer more true.
    #[inline]
    #[must_use]
    pub const fn idle_secs(self, now_ms: u64) -> u64 {
        if self.is_unset() {
            return 0;
        }
        let now = clock_at(now_ms);
        let then = self.bits();
        let ticks = if now >= then {
            now - then
        } else {
            now + (MAX - then)
        };
        ticks as u64
    }

    /// The frequency counter, with the decay since the last access applied.
    ///
    /// This is what `OBJECT FREQ` returns and what eviction compares. The decay
    /// is applied on read rather than on a timer, which is what makes the whole
    /// thing free when nobody is asking: there is no sweep that walks every key
    /// once a minute to bring counters down, and a key nobody looks at costs
    /// nothing to not look at.
    #[inline]
    #[must_use]
    pub const fn freq(self, now_ms: u64, lfu: Lfu) -> u8 {
        if self.is_unset() {
            return LFU_INIT;
        }
        let counter = self.counter();
        if lfu.decay_minutes == 0 {
            return counter;
        }
        let periods = self.elapsed_minutes(now_ms) / lfu.decay_minutes;
        if periods >= counter as u32 {
            0
        } else {
            counter - periods as u8
        }
    }

    /// The field after one access under an LFU policy.
    ///
    /// Decay first and then increment, in that order, because the other order
    /// would let a key that is read once a minute climb forever: the increment
    /// would land before the decay took it off again and the counter would
    /// ratchet up on traffic that is not actually heavy.
    ///
    /// The increment is probabilistic and that is the whole trick. Eight bits
    /// cannot count to a million, so the counter does not count accesses, it
    /// samples them, at odds that fall as the counter rises. A key at 5 moves on
    /// the next access, a key at 100 moves on about one access in a thousand,
    /// and the result is a number that orders keys by traffic across several
    /// orders of magnitude without ever needing a ninth bit.
    #[must_use]
    pub fn touched(self, now_ms: u64, lfu: Lfu, rng: &mut Rng) -> Access {
        let counter = self.freq(now_ms, lfu);
        Access::pack(minutes_at(now_ms), incr(counter, lfu.log_factor, rng))
    }

    /// The counter as stored, with no decay applied.
    ///
    /// Only the decay in [`Access::freq`] should be reading this. It is the raw
    /// low byte and it is an overestimate of the key's frequency by however long
    /// it has been since the key was last touched.
    #[inline]
    const fn counter(self) -> u8 {
        (self.0 & 0xff) as u8
    }

    /// Minutes since the counter was last brought up to date.
    ///
    /// Wraps the same way the LRU clock does and for the same reason, except
    /// that sixteen bits of minutes goes round every forty five days rather than
    /// every hundred and ninety four. It is short by one minute across the wrap
    /// for the same reason [`Access::idle_secs`] is short by one second, and it
    /// is kept for the same reason.
    #[inline]
    const fn elapsed_minutes(self, now_ms: u64) -> u32 {
        let now = minutes_at(now_ms);
        let then = (self.0 >> 8) & LFU_TIME_MAX;
        if now >= then {
            now - then
        } else {
            LFU_TIME_MAX - then + now
        }
    }

    /// Put a clock reading and a counter together into the field.
    #[inline]
    const fn pack(minutes: u32, counter: u8) -> Access {
        Access(((minutes & LFU_TIME_MAX) << 8) | counter as u32)
    }
}

/// The LRU clock at `now_ms`, which is seconds truncated to twenty four bits.
#[inline]
const fn clock_at(now_ms: u64) -> u32 {
    ((now_ms / LRU_RESOLUTION_MS) & MAX as u64) as u32
}

/// The LFU clock at `now_ms`, which is minutes truncated to sixteen bits.
#[inline]
const fn minutes_at(now_ms: u64) -> u32 {
    ((now_ms / 60_000) & LFU_TIME_MAX as u64) as u32
}

/// One probabilistic step up the counter.
///
/// Saturates at 255 rather than wrapping, which matters more than it looks: a
/// counter that wrapped would turn the hottest key in the database into the
/// coldest one, and it would do it silently.
#[inline]
fn incr(counter: u8, log_factor: u32, rng: &mut Rng) -> u8 {
    if counter == u8::MAX {
        return u8::MAX;
    }
    // Below the starting value the odds are even, so a brand new key and a key
    // that has decayed to nothing both climb on their next access rather than
    // being stuck at the bottom.
    let base = counter.saturating_sub(LFU_INIT) as u32;
    if rng.chance(1, base * log_factor + 1) {
        counter + 1
    } else {
        counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A millisecond reading `secs` seconds after the epoch.
    const fn at(secs: u64) -> u64 {
        secs * 1000
    }

    /// Decay turned off, so a test can hold a counter still.
    const NO_DECAY: Lfu = Lfu {
        decay_minutes: 0,
        ..Lfu::DEFAULT
    };

    /// A log factor of zero, which makes every increment certain rather than
    /// probabilistic. It is how a test walks the counter somewhere on purpose
    /// instead of waiting for the odds.
    const EVEN: Lfu = Lfu {
        log_factor: 0,
        ..Lfu::DEFAULT
    };

    #[test]
    fn idle_time_is_seconds_since_the_key_was_touched() {
        let a = Access::lru(at(1_000));
        assert_eq!(a.idle_secs(at(1_000)), 0);
        assert_eq!(a.idle_secs(at(1_030)), 30);
        assert_eq!(a.idle_secs(at(1_000 + 86_400)), 86_400);
    }

    #[test]
    fn idle_time_ignores_the_part_of_a_second_that_has_not_finished() {
        // The clock is seconds, so a key touched at 1500 ms and read at 1900 ms
        // is on the same tick and is zero seconds idle. Redis reports the same,
        // and a test suite that touches a key and immediately asks for its idle
        // time depends on it.
        let a = Access::lru(1_500);
        assert_eq!(a.idle_secs(1_900), 0);
        assert_eq!(a.idle_secs(2_100), 1);
    }

    #[test]
    fn idle_time_survives_the_clock_going_round() {
        // The clock wraps at 2^24 seconds, which is about a hundred and ninety
        // four days. A key touched just before the wrap and read just after it
        // has to come back as seconds old and not as most of a year old, because
        // under allkeys-lru the second answer evicts the hottest key there is.
        let wrap = at(MAX as u64);
        let a = Access::lru(wrap - at(5));
        assert_eq!(a.idle_secs(wrap - at(5)), 0);
        // Five seconds before the wrap and three after is eight, and Redis says
        // seven, because its wrapped arm subtracts from `MAX` when the period is
        // `MAX + 1`. Seven is the answer this has to give. See `idle_secs`.
        assert_eq!(a.idle_secs(wrap + at(3)), 7);
    }

    #[test]
    fn a_new_key_starts_at_the_initial_frequency() {
        let a = Access::lfu(at(0));
        assert_eq!(a.freq(at(0), Lfu::DEFAULT), LFU_INIT);
    }

    #[test]
    fn the_counter_decays_one_step_per_decay_period() {
        let a = Access::lfu(at(0));
        assert_eq!(a.freq(at(60), Lfu::DEFAULT), LFU_INIT - 1);
        assert_eq!(a.freq(at(180), Lfu::DEFAULT), LFU_INIT - 3);
        assert_eq!(a.freq(at(3_600), Lfu::DEFAULT), 0, "and stops at zero");
    }

    #[test]
    fn a_decay_time_of_zero_turns_decay_off() {
        let a = Access::lfu(at(0));
        assert_eq!(a.freq(at(86_400 * 30), NO_DECAY), LFU_INIT);
    }

    #[test]
    fn the_counter_decays_across_its_own_wrap() {
        // The LFU clock is sixteen bits of minutes, so it goes round every forty
        // five days. Reading across the wrap the wrong way round would report a
        // gap of most of the period, which would decay every counter to zero and
        // make the policy pick a victim at random.
        let a = Access::lfu(at(60 * (LFU_TIME_MAX as u64 - 2)));
        assert_eq!(
            a.freq(at(60 * (LFU_TIME_MAX as u64 - 1)), Lfu::DEFAULT),
            LFU_INIT - 1
        );
        // Three minutes later, and two decays rather than three, which is the
        // same one Redis is short by across a wrap. See `elapsed_minutes`.
        assert_eq!(
            a.freq(at(60 * (LFU_TIME_MAX as u64 + 1)), Lfu::DEFAULT),
            LFU_INIT - 2
        );
    }

    #[test]
    fn a_hot_key_climbs_and_a_cold_one_does_not() {
        // The point of the whole counter in one assertion. Same elapsed time,
        // different traffic, and the busy key has to come out ahead.
        let mut rng = Rng::new(1);
        let mut hot = Access::lfu(at(0));
        for i in 0..10_000u64 {
            hot = hot.touched(at(i / 100), Lfu::DEFAULT, &mut rng);
        }
        let cold = Access::lfu(at(0));
        assert!(
            hot.freq(at(100), Lfu::DEFAULT) > cold.freq(at(100), Lfu::DEFAULT),
            "hot {} cold {}",
            hot.freq(at(100), Lfu::DEFAULT),
            cold.freq(at(100), Lfu::DEFAULT)
        );
    }

    #[test]
    fn the_counter_flattens_out_rather_than_running_away() {
        // Ten thousand accesses in the same minute, so no decay, and the counter
        // has to be well short of ten thousand. If it were linear this would
        // saturate in two hundred and fifty accesses and every busy key in the
        // database would be pinned at 255 and indistinguishable from every other
        // busy key, which is the failure the logarithm exists to avoid.
        let mut rng = Rng::new(7);
        let mut a = Access::lfu(at(0));
        for _ in 0..10_000 {
            a = a.touched(at(0), Lfu::DEFAULT, &mut rng);
        }
        let f = a.freq(at(0), Lfu::DEFAULT);
        assert!((30..=90).contains(&f), "ten thousand accesses reached {f}");
    }

    #[test]
    fn the_counter_saturates_instead_of_wrapping() {
        // A counter that wrapped would turn the hottest key in the database into
        // the coldest, so this walks it to the top with the odds forced even and
        // checks that it stays there.
        let mut rng = Rng::new(3);
        let mut a = Access::lfu(at(0));
        for _ in 0..100_000 {
            a = a.touched(at(0), EVEN, &mut rng);
        }
        assert_eq!(a.freq(at(0), Lfu::DEFAULT), u8::MAX);
    }

    #[test]
    fn a_decayed_key_climbs_again_at_even_odds() {
        // A key that has decayed to zero is at the bottom, and the odds of
        // moving are meant to be even there rather than one in one. Two
        // accesses is enough to leave zero behind.
        let mut rng = Rng::new(11);
        let mut a = Access::lfu(at(0));
        assert_eq!(a.freq(at(3_600), Lfu::DEFAULT), 0);
        a = a.touched(at(3_600), Lfu::DEFAULT, &mut rng);
        assert_eq!(a.freq(at(3_600), Lfu::DEFAULT), 1);
    }

    #[test]
    fn the_field_is_twenty_four_bits_and_survives_a_round_trip() {
        // It has to fit alongside whatever it is packed with, so nothing here
        // may write above bit 23.
        let mut rng = Rng::new(5);
        let mut a = Access::lfu(at(0));
        for i in 0..1_000u64 {
            a = a.touched(at(i * 37), Lfu::DEFAULT, &mut rng);
            assert_eq!(a.bits() >> BITS, 0, "wrote above bit 23");
            assert_eq!(Access::from_bits(a.bits()), a);
        }
        for i in 0..1_000u64 {
            let l = Access::lru(at(i * 100_003));
            assert_eq!(l.bits() >> BITS, 0);
            assert_eq!(Access::from_bits(l.bits()), l);
        }
    }

    #[test]
    fn bits_above_the_field_are_dropped_rather_than_trusted() {
        assert_eq!(Access::from_bits(0xffff_ffff).bits(), MAX);
        assert_eq!(Access::from_bits(0xff00_0000).bits(), 0);
    }

    /// Every policy name round trips, in either case, and nothing else parses.
    ///
    /// The names are wire strings. A typo in one of them is a `CONFIG SET` that
    /// a client thinks worked and a `CONFIG GET` that reports something the
    /// client never asked for, and neither end would notice.
    #[test]
    fn every_policy_name_survives_a_round_trip() {
        // Written out rather than taken from `ALL`, because a test that reads
        // its expectations out of the thing it is testing agrees with a typo.
        // The order is Redis's own, which is what the CONFIG SET error message
        // has to list them in.
        let names = [
            "volatile-lru",
            "volatile-lfu",
            "volatile-random",
            "volatile-ttl",
            "volatile-lrm",
            "allkeys-lru",
            "allkeys-lfu",
            "allkeys-random",
            "allkeys-lrm",
            "noeviction",
        ];
        for name in names {
            let p =
                Policy::parse(name.as_bytes()).unwrap_or_else(|| panic!("{name} did not parse"));
            assert_eq!(p.name(), name);
            assert_eq!(Policy::parse(name.to_uppercase().as_bytes()), Some(p));
        }
        let listed: Vec<&str> = Policy::ALL.iter().map(|p| p.name()).collect();
        assert_eq!(listed, names, "ALL is Redis's order and is all of them");
        assert_eq!(Policy::parse(b"allkeys"), None);
        assert_eq!(Policy::parse(b""), None);
        assert_eq!(Policy::parse(b"allkeys-lru "), None, "no trimming here");
    }

    /// A key nothing has stamped reads as freshly used under both policies.
    ///
    /// This is the one that matters on the day somebody turns a policy on. Every
    /// key already in the database is unstamped at that moment, and the wrong
    /// answer here makes all of them the most attractive victims available, so
    /// enabling `allkeys-lru` on a full database would throw the working set
    /// away before it read any of it back.
    #[test]
    fn a_key_that_was_never_stamped_reads_as_freshly_used() {
        let unset = Access::default();
        assert!(unset.is_unset());
        assert_eq!(unset.idle_secs(at(86_400 * 365)), 0, "not idle for a year");
        assert_eq!(unset.freq(at(86_400 * 365), Lfu::DEFAULT), LFU_INIT);

        // And it leaves the sentinel behind as soon as it is touched.
        let mut rng = Rng::new(2);
        let stamped = unset.touched(at(1_000), Lfu::DEFAULT, &mut rng);
        assert!(!stamped.is_unset());
        assert!(stamped.freq(at(1_000), Lfu::DEFAULT) >= LFU_INIT);
    }

    #[test]
    fn the_default_is_to_refuse_the_write_rather_than_lose_data() {
        assert_eq!(Policy::default(), Policy::NoEviction);
        // It still keeps the clock, which is why OBJECT IDLETIME answers on a
        // default server that will never evict anything.
        assert!(Policy::default().stamps_on_read());
        assert!(Policy::default().is_clock());
    }

    /// The axes each policy sits on, spelled out once so that a name added or a
    /// `matches!` arm edited has to be edited here too.
    #[test]
    fn each_policy_is_on_the_axes_its_name_says() {
        for (p, volatile, lru, lfu, lrm) in [
            (Policy::VolatileLru, true, true, false, false),
            (Policy::VolatileLfu, true, false, true, false),
            (Policy::VolatileRandom, true, false, false, false),
            (Policy::VolatileTtl, true, false, false, false),
            (Policy::VolatileLrm, true, false, false, true),
            (Policy::AllKeysLru, false, true, false, false),
            (Policy::AllKeysLfu, false, false, true, false),
            (Policy::AllKeysRandom, false, false, false, false),
            (Policy::AllKeysLrm, false, false, false, true),
            (Policy::NoEviction, false, false, false, false),
        ] {
            let n = p.name();
            assert_eq!(p.volatile_only(), volatile, "{n} volatile");
            assert_eq!(p.is_lru(), lru, "{n} lru");
            assert_eq!(p.is_lfu(), lfu, "{n} lfu");
            assert_eq!(p.is_lrm(), lrm, "{n} lrm");
            // The field is a clock under everything except LFU, and a read
            // moves it under everything except LRM. Those are two different
            // questions with two different answers and it is worth pinning both.
            assert_eq!(p.is_clock(), !lfu, "{n} clock");
            assert_eq!(p.stamps_on_read(), !lrm, "{n} stamps on read");
            assert!(p.stamps_on_write(), "{n} stamps on write");
        }
    }

    /// Least recently modified is the pair that catches people out, so the rule
    /// that separates it from least recently used gets its own test.
    #[test]
    fn only_the_lrm_pair_ignores_a_read() {
        for p in Policy::ALL {
            assert_eq!(p.stamps_on_read(), !p.is_lrm(), "{}", p.name());
        }
        // And they read the field the same way once it is set, because it is
        // the same clock. Only the moment it is written apart.
        assert!(Policy::AllKeysLrm.is_clock());
        assert!(Policy::AllKeysLru.is_clock());
    }
}
