//! Deadlines on individual fields, which is the `HEXPIRE` family.
//!
//! Y20 makes hash field TTL the carve-out from ordinary expiry: a key's deadline
//! is an int64 in its value header, but a field's cannot be, because there is no
//! header per field and adding one would cost every hash that never expires a
//! field. `08` section 3 spends it instead on a side array allocated the first
//! time any field of that hash is given a deadline, and nothing before then.
//!
//! So a hash with no field TTL carries one empty `Vec`, which is three words in
//! the hash and no allocation, and G8's sixteen bytes a field is untouched. A
//! hash with one field TTL carries eight bytes for every field it has. That is
//! the trade, and it is the right way round, because a hash with a field TTL is
//! the rare one.
//!
//! # Indexed by the row, not by a ttl_idx
//!
//! `08` section 3 says the array is indexed by a `ttl_idx` held in the row. This
//! is indexed by the row position itself, which is a divergence and worth the
//! sentence.
//!
//! A `ttl_idx` is four bytes in every row, always, to point into an array holding
//! eight bytes only for the fields that have deadlines. Against a dense array of
//! eight bytes a row, it saves memory only when fewer than half the fields have a
//! deadline, and it costs a second dependent load on every read, because the read
//! has the row and needs the idx before it can have the deadline. It also has to
//! carry a free list, since a persisted field's slot has to go back somewhere.
//!
//! The dense array has none of that. The row position is already in hand the
//! moment the probe finishes, so the deadline is one indexed load from a base
//! pointer, and the whole structure is a `Vec`. When there is a hash type to
//! measure, the density where the two cross is a fact rather than an argument,
//! and this can be revisited then.
//!
//! # Staying in step
//!
//! [`Elements`](crate::Elements) swap removes: the last row moves into the hole.
//! A parallel array has to make the same move or every deadline after the hole
//! belongs to the wrong field, which is a bug that reads correct on the first
//! `HTTL` after it and wrong on all of them. That is what [`Deadlines::inserted`]
//! and [`Deadlines::removed`] are, and a hash has to call them for every row it
//! adds or takes out. `deadlines_follow_the_table_through_a_swap_remove` is the
//! test that says so.
//!
//! # Lazy, like the rest of expiry
//!
//! A field past its deadline is gone when something looks at it, and until then
//! it is still sitting there. [`Deadlines::soonest`] exists so that the active
//! cycle in M5 can skip a whole hash without walking it, which is the same thing
//! Redis registers in its global HFE structure.

/// No deadline.
///
/// Redis spells this `EB_EXPIRE_TIME_INVALID`, one past its 48 bit maximum. Ours
/// is `u64::MAX` for the same reason: it is a value a real deadline cannot take,
/// so no field needs a second byte saying whether the first eight mean anything.
const NONE: u64 = u64::MAX;

/// The largest deadline a field can be given, in unix milliseconds.
///
/// Redis's `HFE_MAX_ABS_TIME_MSEC`, which is `EB_EXPIRE_TIME_MAX >> 2` where the
/// maximum is 48 bits. It lands in the year 4200 or so. Anything past it is
/// rejected by the command with `invalid expire time`, before any field is
/// touched, so a command that names ten fields either sets all ten or errors.
pub const MAX_AT: u64 = 0x0000_3FFF_FFFF_FFFF;

/// Whether a deadline is one a field is allowed to be given.
///
/// The command layer checks this before it starts, because Redis rejects the
/// whole command rather than failing field by field.
#[must_use]
pub const fn valid_at(ms: u64) -> bool {
    ms <= MAX_AT
}

/// The condition on a deadline being set, which is `NX`, `XX`, `GT` or `LT`.
///
/// A hash field command takes exactly one of the four keywords. `EXPIRE` on a
/// whole key takes a set of them and accepts two, `XX GT` and `XX LT`, which is
/// where [`Cond::LessAndSet`] comes from. `XX GT` is not a sixth case because
/// `GT` already refuses a key with no deadline, so it means the same thing as
/// `GT` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cond {
    /// No condition. Set it whatever was there.
    #[default]
    Always,
    /// Only if the field has no deadline now.
    NotSet,
    /// Only if the field has one now.
    AlreadySet,
    /// Only if it moves the deadline later, or there was none.
    Greater,
    /// Only if it moves the deadline earlier, or there was none.
    Less,
    /// Only if there is one now and this moves it earlier.
    ///
    /// `XX LT` on a key. It is a separate case because `LT` on its own accepts a
    /// key with no deadline, on the reading that no deadline is infinitely far
    /// away, and `XX` is what takes that reading away.
    LessAndSet,
}

/// What setting a deadline did.
///
/// Not called `Set`, because [`crate::Set`] is a set and two types with that
/// name in one crate is a trap for whoever reads it next.
///
/// The numbers are Redis's, from the `SetExRes` enum in `t_hash.c`, and they are
/// what `HEXPIRE` puts in its reply array. They are here rather than in the
/// protocol layer because they are semantics: whether a past deadline deletes the
/// field or stores it is a decision about the data structure, and the reply just
/// reports it.
///
/// `EXPIRE` on a whole key produces the same four outcomes and reports them with
/// two numbers instead of four, so [`Keyspace::expire`] answers this and the wire
/// folds it. `EXPIRE` cannot tell you whether the key went away or the deadline
/// went on, and this can.
///
/// [`Keyspace::expire`]: crate::Keyspace::expire
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    /// No such field, or no such key. Redis replies -2 for a field, and does the
    /// same for a hash that is not there at all, because a missing key and an
    /// empty hash are the same thing.
    Missing = -2,
    /// The condition was not met, so nothing changed. Redis replies 0.
    NotMet = 0,
    /// Set or updated. Redis replies 1.
    Ok = 1,
    /// The deadline was already in the past, so the field is gone rather than
    /// expiring later. Redis replies 2, and the caller has to actually remove the
    /// field, because this structure holds deadlines and not fields.
    Deleted = 2,
}

/// What asking about a deadline found.
///
/// Redis's `HFE_GET_*` and `HFE_PERSIST_*` codes, which agree with each other on
/// -2 and -1 and are the same two questions.
///
/// `TTL` and `PTTL` on a whole key ask the same three way question and answer it
/// with the same two sentinels, so this says field or key rather than field. A
/// key that is not there is -2 and a key with no deadline is -1, which is what
/// makes the hash field commands and the key commands one shape and not two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ask {
    /// No such field, or no such key. Redis replies -2.
    Missing,
    /// It is there and has no deadline. Redis replies -1.
    NoDeadline,
    /// It is there and has one.
    At(u64),
}

impl Ask {
    /// The number Redis puts in the reply, given the moment being asked at.
    ///
    /// `HTTL` and `HPTTL` want what is left rather than when it falls due, and a
    /// deadline that has passed reads as no field, because the field is about to
    /// be reclaimed by whatever touched it.
    #[must_use]
    pub const fn remaining_ms(self, now: u64) -> i64 {
        match self {
            Ask::Missing => -2,
            Ask::NoDeadline => -1,
            Ask::At(at) if at <= now => -2,
            // The arm above leaves `at > now`, and a deadline is under 2^46, so
            // this neither underflows nor overruns an i64.
            Ask::At(at) => (at - now) as i64,
        }
    }
}

/// The deadlines of one collection's fields, indexed by row position.
///
/// Empty and unallocated until a field is given one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Deadlines {
    /// One entry per row once allocated, [`NONE`] for a field without a deadline,
    /// and empty while no field has ever had one.
    at: Vec<u64>,
    /// How many rows the table has, tracked so the array can be allocated at the
    /// right length the moment it is first needed.
    rows: usize,
    /// How many of them carry a deadline.
    live: usize,
    /// A lower bound on the earliest deadline here, or [`NONE`].
    ///
    /// A bound and not the answer: it goes down when a deadline is set and it
    /// does not go back up when that field is persisted or removed, because
    /// finding the new earliest would mean a walk on every removal.
    /// [`Deadlines::soonest`] is therefore never later than the truth, which is
    /// the direction that keeps the active cycle correct: it can wake early and
    /// find nothing, but it cannot sleep through an expiry.
    soonest: u64,
}

impl Deadlines {
    /// A collection with no deadlines on anything, which allocates nothing.
    #[must_use]
    pub const fn new() -> Deadlines {
        Deadlines {
            at: Vec::new(),
            rows: 0,
            live: 0,
            soonest: NONE,
        }
    }

    /// Whether any field here has a deadline.
    ///
    /// False means every read path can skip this structure entirely, which is the
    /// case worth being fast and is the usual one.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// How many fields carry a deadline.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live
    }

    /// A lower bound on the earliest deadline, or `None` if nothing has one.
    ///
    /// What M5's active cycle registers so it can pass over a whole hash without
    /// walking its fields. Never later than the real earliest deadline, so acting
    /// on it can waste a walk but cannot miss an expiry.
    #[inline]
    #[must_use]
    pub const fn soonest(&self) -> Option<u64> {
        if self.soonest == NONE {
            None
        } else {
            Some(self.soonest)
        }
    }

    /// Recompute [`Deadlines::soonest`] exactly.
    ///
    /// The bound only ever drifts early, and a hash whose earliest field has been
    /// persisted or expired keeps waking the cycle up for nothing until someone
    /// pays for this walk. The active cycle is the one to pay it, once it has
    /// walked the fields anyway and knows the bound was stale.
    pub fn refresh_soonest(&mut self) {
        self.soonest = self.at.iter().copied().min().unwrap_or(NONE);
    }

    /// A row was added to the table, so add a slot for it.
    ///
    /// Must be called for every row the table gains, including ones that will
    /// never have a deadline, or the array stops lining up with the rows.
    #[inline]
    pub fn inserted(&mut self) {
        self.rows += 1;
        if !self.at.is_empty() {
            self.at.push(NONE);
        }
        self.check();
    }

    /// The array is either not there at all or exactly as long as the table.
    ///
    /// Anything else means a hash has added or removed a row without telling this
    /// structure, and the deadlines from that point on belong to the wrong
    /// fields. Debug only, because it is one comparison on a path that runs per
    /// field, but a release build that has drifted is a release build that was
    /// already wrong in a test.
    #[inline]
    fn check(&self) {
        debug_assert!(
            self.at.is_empty() || self.at.len() == self.rows,
            "the deadlines have drifted from the table: {} against {} rows",
            self.at.len(),
            self.rows
        );
    }

    /// A row was taken out at `row`, the same way the table takes one out.
    ///
    /// [`Elements::remove_at`](crate::Elements::remove_at) moves the last row into
    /// the hole, so this does exactly that to the deadlines. Any other order and
    /// the deadlines after the hole belong to the wrong fields.
    pub fn removed(&mut self, row: usize) {
        debug_assert!(row < self.rows, "removing a row that is not there");
        self.rows -= 1;
        if !self.at.is_empty() {
            if self.at.swap_remove(row) != NONE {
                self.live -= 1;
            }
            if self.live == 0 {
                // Nothing left to hold, so give the memory back rather than
                // keeping an array of NONE for a hash that has stopped using
                // deadlines. The next set reallocates, which is the same price it
                // paid the first time.
                self.at = Vec::new();
                self.soonest = NONE;
            }
        }
        self.check();
    }

    /// Everything went, so forget everything.
    pub fn cleared(&mut self) {
        self.at = Vec::new();
        self.rows = 0;
        self.live = 0;
        self.soonest = NONE;
    }

    /// The deadline on a row, if it has one.
    ///
    /// The read on the hot path, and the reason the array is indexed by the row
    /// position: after a probe the position is already in a register.
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize) -> Option<u64> {
        match self.at.get(row) {
            Some(&NONE) | None => None,
            Some(&at) => Some(at),
        }
    }

    /// Whether a row is past its deadline at `now`.
    ///
    /// The lazy expiry check, which every read of a field has to make. A hash
    /// with no deadlines answers it without touching memory beyond the counter.
    #[inline]
    #[must_use]
    pub fn is_expired(&self, row: usize, now: u64) -> bool {
        if self.live == 0 {
            return false;
        }
        match self.at.get(row) {
            Some(&at) => at != NONE && at <= now,
            None => false,
        }
    }

    /// What `HTTL` and `HPERSIST` want to know about a row.
    ///
    /// The caller has already established the field exists, so this never answers
    /// [`Ask::Missing`]. That case belongs to the table, not here.
    #[inline]
    #[must_use]
    pub fn ask(&self, row: usize) -> Ask {
        match self.get(row) {
            Some(at) => Ask::At(at),
            None => Ask::NoDeadline,
        }
    }

    /// Put a deadline on a row, or find out why not.
    ///
    /// `at` is absolute unix milliseconds, which is what `HEXPIRE` and its
    /// relatives all turn into before they get here, and the command layer has
    /// already checked it against [`MAX_AT`].
    ///
    /// [`Applied::Deleted`] means the deadline had already passed. The deadline is not
    /// stored and the caller has to remove the field, which is Redis's behaviour
    /// and is why `HEXPIRE key 0 FIELDS 1 f` is a roundabout `HDEL`.
    pub fn set(&mut self, row: usize, at: u64, cond: Cond, now: u64) -> Applied {
        debug_assert!(
            row < self.rows,
            "setting a deadline on a row that is not there"
        );
        self.check();
        let prev = self.get(row);
        match decide(prev, at, cond, now) {
            Applied::Ok => {}
            // Redis clears nothing on a past deadline: it deletes the field, and
            // the field taking its deadline with it is the caller calling
            // removed().
            other => return other,
        }
        if self.at.is_empty() {
            // First deadline in this collection. This is the allocation Y20 is
            // about, and everything above this line has happened without one.
            self.at = vec![NONE; self.rows];
        }
        if prev.is_none() {
            self.live += 1;
        }
        self.at[row] = at;
        self.soonest = self.soonest.min(at);
        Applied::Ok
    }

    /// Take a row's deadline off, which is `HPERSIST`.
    ///
    /// [`Ask::NoDeadline`] means there was nothing to take off, which Redis
    /// replies -1 to, and [`Ask::At`] hands back the deadline that was there.
    pub fn clear(&mut self, row: usize) -> Ask {
        let Some(was) = self.get(row) else {
            return Ask::NoDeadline;
        };
        self.at[row] = NONE;
        self.live -= 1;
        if self.live == 0 {
            self.at = Vec::new();
            self.soonest = NONE;
        }
        Ask::At(was)
    }

    /// What this costs, for `MEMORY USAGE` and for G8's per field number.
    ///
    /// Zero until a field is given a deadline, which is the entire point.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.at.capacity() * size_of::<u64>()
    }
}

/// What setting this deadline over that one does, before anything is stored.
///
/// Public because a hash in the listpack band keeps its deadlines in the
/// listpack rather than in a [`Deadlines`], and the rule about what `NX`, `XX`,
/// `GT` and `LT` allow has to be one rule and not one per band. The band that
/// has a [`Deadlines`] reaches it through [`Deadlines::set`], and the band that
/// does not calls this and then writes the number itself.
///
/// Never answers [`Applied::Missing`], since establishing that the field exists
/// is what the caller did to have a `prev` to pass in.
#[must_use]
pub const fn decide(prev: Option<u64>, at: u64, cond: Cond, now: u64) -> Applied {
    if !allowed(prev, at, cond) {
        return Applied::NotMet;
    }
    // The condition is checked first, so HEXPIRE 0 XX on a field with no
    // deadline is 0 and not 2, and the field survives it.
    if at <= now {
        return Applied::Deleted;
    }
    Applied::Ok
}

/// Whether the condition lets this deadline replace that one.
///
/// Straight off `hashTypeSetExpiryListpack` in Redis 8.10.1. The asymmetry in the
/// first arm is theirs and is easy to get backwards: against a field with no
/// deadline, `XX` and `GT` fail, while `LT` passes, on the reading that no
/// deadline is infinitely far away and so anything is less than it.
const fn allowed(prev: Option<u64>, at: u64, cond: Cond) -> bool {
    match (prev, cond) {
        (_, Cond::Always) => true,
        (None, Cond::AlreadySet | Cond::Greater | Cond::LessAndSet) => false,
        (None, Cond::NotSet | Cond::Less) => true,
        (Some(_), Cond::NotSet) => false,
        (Some(_), Cond::AlreadySet) => true,
        (Some(p), Cond::Greater) => p < at,
        (Some(p), Cond::Less | Cond::LessAndSet) => p > at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Elements;

    /// A `Deadlines` for a collection that already has `n` rows.
    fn with_rows(n: usize) -> Deadlines {
        let mut d = Deadlines::new();
        for _ in 0..n {
            d.inserted();
        }
        d
    }

    #[test]
    fn a_collection_with_no_field_ttl_costs_nothing() {
        let mut d = with_rows(1000);
        assert!(d.is_empty());
        assert_eq!(d.memory_bytes(), 0);
        assert_eq!(d.soonest(), None);
        assert!(!d.is_expired(0, u64::MAX));
        assert_eq!(d.ask(7), Ask::NoDeadline);

        // And the first deadline is what pays for the array.
        assert_eq!(d.set(7, 5000, Cond::Always, 0), Applied::Ok);
        assert_eq!(d.memory_bytes(), 8000);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn a_deadline_goes_on_and_comes_off() {
        let mut d = with_rows(4);
        assert_eq!(d.set(2, 900, Cond::Always, 100), Applied::Ok);
        assert_eq!(d.get(2), Some(900));
        assert_eq!(d.ask(2), Ask::At(900));
        assert_eq!(d.get(1), None);

        assert_eq!(d.clear(2), Ask::At(900));
        assert_eq!(d.get(2), None);
        assert_eq!(
            d.clear(2),
            Ask::NoDeadline,
            "twice is not an error, it is -1"
        );
        assert!(d.is_empty());
        assert_eq!(d.memory_bytes(), 0, "the last one off gives the array back");
    }

    /// A field is gone when something looks at it and not before.
    #[test]
    fn a_field_is_expired_only_once_its_moment_has_passed() {
        let mut d = with_rows(2);
        d.set(0, 1000, Cond::Always, 0);
        assert!(!d.is_expired(0, 999));
        assert!(d.is_expired(0, 1000), "the deadline itself has passed");
        assert!(d.is_expired(0, 1001));
        assert!(!d.is_expired(1, u64::MAX), "no deadline is not expired");
    }

    /// Redis replies 2 and deletes the field rather than storing a deadline that
    /// has already gone, which is what makes `HEXPIRE key 0` an `HDEL`.
    #[test]
    fn a_deadline_in_the_past_deletes_instead_of_being_stored() {
        let mut d = with_rows(2);
        assert_eq!(
            d.set(0, 500, Cond::Always, 500),
            Applied::Deleted,
            "now counts"
        );
        assert_eq!(d.set(0, 499, Cond::Always, 500), Applied::Deleted);
        assert_eq!(d.get(0), None, "nothing was stored");
        assert_eq!(d.memory_bytes(), 0, "and nothing was allocated for it");
    }

    /// The conditions, against the C in `hashTypeSetExpiryListpack`. The first
    /// group is the one that is easy to get backwards.
    #[test]
    fn the_conditions_are_the_ones_redis_applies() {
        // No deadline yet: XX and GT fail, NX and LT pass.
        assert!(!allowed(None, 100, Cond::AlreadySet));
        assert!(!allowed(None, 100, Cond::Greater));
        assert!(allowed(None, 100, Cond::NotSet));
        assert!(allowed(None, 100, Cond::Less));
        assert!(allowed(None, 100, Cond::Always));

        // Already has one: NX fails, XX passes.
        assert!(!allowed(Some(50), 100, Cond::NotSet));
        assert!(allowed(Some(50), 100, Cond::AlreadySet));

        // GT wants strictly later, LT strictly earlier, and equal fails both.
        assert!(allowed(Some(50), 100, Cond::Greater));
        assert!(!allowed(Some(50), 20, Cond::Greater));
        assert!(!allowed(Some(50), 50, Cond::Greater));
        assert!(allowed(Some(50), 20, Cond::Less));
        assert!(!allowed(Some(50), 100, Cond::Less));
        assert!(!allowed(Some(50), 50, Cond::Less));
    }

    #[test]
    fn a_condition_that_fails_changes_nothing() {
        let mut d = with_rows(2);
        d.set(0, 1000, Cond::Always, 0);
        assert_eq!(d.set(0, 500, Cond::Greater, 0), Applied::NotMet);
        assert_eq!(d.get(0), Some(1000));
        assert_eq!(d.set(1, 500, Cond::AlreadySet, 0), Applied::NotMet);
        assert_eq!(d.get(1), None);
        assert_eq!(d.len(), 1);
    }

    /// The condition is checked before the past deadline is, so `HEXPIRE ... 0 XX`
    /// on a field with no deadline is 0 and not 2, and the field survives.
    #[test]
    fn a_failed_condition_beats_a_past_deadline() {
        let mut d = with_rows(1);
        assert_eq!(d.set(0, 0, Cond::AlreadySet, 100), Applied::NotMet);
    }

    /// The one that would silently corrupt every deadline after a hole.
    #[test]
    fn deadlines_follow_the_table_through_a_swap_remove() {
        let mut table: Elements<u32> = Elements::new();
        let mut d = Deadlines::new();
        for i in 0..5u32 {
            table.insert(format!("f{i}").as_bytes(), i).expect("room");
            d.inserted();
        }
        // Every field gets its index as its deadline, so a deadline that has
        // drifted onto the wrong field is visible rather than plausible.
        for i in 0..5 {
            d.set(i, 1000 + i as u64, Cond::Always, 0);
        }

        // Take out the middle one, which moves the last row into its place.
        let at = table.iter().position(|(n, _)| n == b"f1").expect("there");
        table.remove_at(at).expect("there");
        d.removed(at);

        assert_eq!(table.len(), 4);
        for (row, (name, _)) in table.iter().enumerate() {
            let i: u64 = String::from_utf8_lossy(&name[1..]).parse().expect("f<n>");
            assert_eq!(
                d.get(row),
                Some(1000 + i),
                "field {} kept someone else's deadline",
                String::from_utf8_lossy(name)
            );
        }
    }

    #[test]
    fn removing_the_last_field_with_a_deadline_gives_the_array_back() {
        let mut d = with_rows(3);
        d.set(1, 1000, Cond::Always, 0);
        assert_eq!(d.memory_bytes(), 24);
        d.removed(1);
        assert!(d.is_empty());
        assert_eq!(d.memory_bytes(), 0);
        assert_eq!(d.soonest(), None);
    }

    /// The bound is allowed to be early and is not allowed to be late, because
    /// early wastes a walk and late misses an expiry.
    #[test]
    fn the_soonest_deadline_is_a_bound_and_leans_early() {
        let mut d = with_rows(3);
        d.set(0, 5000, Cond::Always, 0);
        d.set(1, 3000, Cond::Always, 0);
        d.set(2, 9000, Cond::Always, 0);
        assert_eq!(d.soonest(), Some(3000));

        // Persisting the earliest leaves the bound behind, which is allowed.
        d.clear(1);
        assert_eq!(
            d.soonest(),
            Some(3000),
            "still early, which is the safe way"
        );
        d.refresh_soonest();
        assert_eq!(
            d.soonest(),
            Some(5000),
            "and exact once someone pays for it"
        );
    }

    #[test]
    fn what_is_left_is_what_redis_replies() {
        assert_eq!(Ask::Missing.remaining_ms(0), -2);
        assert_eq!(Ask::NoDeadline.remaining_ms(0), -1);
        assert_eq!(Ask::At(5000).remaining_ms(1000), 4000);
        assert_eq!(Ask::At(5000).remaining_ms(5000), -2, "due now is gone");
        assert_eq!(Ask::At(5000).remaining_ms(9000), -2);
    }

    #[test]
    fn the_ceiling_is_the_one_redis_enforces() {
        assert_eq!(MAX_AT, 0x0000_FFFF_FFFF_FFFF >> 2);
        assert!(valid_at(MAX_AT));
        assert!(!valid_at(MAX_AT + 1));
        assert!(valid_at(0));
    }

    #[test]
    fn clearing_the_collection_forgets_everything() {
        let mut d = with_rows(3);
        d.set(0, 1000, Cond::Always, 0);
        d.cleared();
        assert!(d.is_empty());
        assert_eq!(d.memory_bytes(), 0);
        assert_eq!(d.soonest(), None);
        d.inserted();
        assert_eq!(d.set(0, 1000, Cond::Always, 0), Applied::Ok);
    }
}
