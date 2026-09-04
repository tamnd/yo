//! One database, cut into stripes.
//!
//! A [`Keyspace`] is one map and one arena and it is reached with `&mut`, which
//! means one command at a time and so one thread at a time. That is the whole
//! reason `yodb serve` runs on one core: not the socket layer, not the parser,
//! the fact that the thing underneath every command can only be held by one
//! caller.
//!
//! A database here is several keyspaces instead of one. A key belongs to
//! exactly one of them, decided by its hash and by nothing else, so two
//! commands on two different keys are two commands on two different objects
//! and there is nothing for them to queue behind. Which stripe a key is on is a
//! function of the key alone, so it does not move, and no index anywhere has to
//! record it.
//!
//! One stripe is a database exactly as it was, and that is the default here.
//! What more than one stripe costs is the subject of the rest of this
//! milestone: a command that names several keys can no longer be handed one
//! keyspace, and everything that walks a whole database has to walk all of
//! them.

use crate::{Clock, Keyspace};

/// The most stripes one database can be cut into.
///
/// Eight bits of hash choose the stripe, so this is what those bits can count
/// to. It is far more than a machine has cores and the point of the ceiling is
/// not to be reached, it is to keep the choice inside a field nothing else is
/// reading.
pub const MAX_STRIPES: usize = 256;

/// Which end of the hash the stripe number is cut from.
///
/// The index inside a keyspace takes the top of the hash for its directory,
/// counting down from bit 56, and the bottom six bits for the bucket inside a
/// segment. Both ends are spoken for, and a stripe number cut from either one
/// would move together with the thing it is supposed to be independent of:
/// every key on a stripe would land in the same corner of that stripe's index.
/// The eight bits above the directory are the ones nothing else reads.
const STRIPE_SHIFT: u32 = 56;

/// One database.
///
/// Holds the keys a client sees under one `SELECT`, spread over one or more
/// keyspaces. A caller that knows which key it wants asks [`Db::at`] and gets
/// the one keyspace that key can be in. A caller that wants the whole database
/// walks [`Db::stripes`], and the answers it adds up are the same answers a
/// single keyspace would have given.
pub struct Db {
    /// The stripes, always a power of two of them and always at least one.
    stripes: Vec<Keyspace>,
    /// `stripes.len() - 1`, kept here so the hot path is a shift and an and
    /// rather than a division.
    mask: u64,
}

impl Db {
    /// A database of `stripes` empty keyspaces on `clock`.
    ///
    /// The count is rounded up to a power of two and clamped to
    /// [`MAX_STRIPES`], and zero means one. Rounding rather than refusing
    /// because the number arrives from `--threads` and from
    /// `available_parallelism`, and neither of those has any reason to be a
    /// power of two, while a mask is the only stripe lookup worth having.
    #[must_use]
    pub fn with_clock(clock: Clock, stripes: usize) -> Db {
        let n = stripes.clamp(1, MAX_STRIPES).next_power_of_two();
        Db {
            stripes: (0..n).map(|_| Keyspace::with_clock(clock)).collect(),
            mask: (n - 1) as u64,
        }
    }

    /// A database of one keyspace on the system clock, which is a database
    /// exactly as it was before there were stripes.
    #[must_use]
    pub fn new() -> Db {
        Db::with_clock(Clock::system(), 1)
    }

    /// How many stripes this database is cut into.
    #[must_use]
    pub fn width(&self) -> usize {
        self.stripes.len()
    }

    /// Which stripe `key` lives on.
    ///
    /// A function of the key and the width and nothing else, so the same key
    /// always answers the same stripe and a caller can work out where a key is
    /// without holding the database.
    #[inline]
    #[must_use]
    pub fn stripe_of(&self, key: &[u8]) -> usize {
        self.stripe_of_hash(Keyspace::hash_of(key))
    }

    /// The same, for a caller that already has the hash.
    ///
    /// The engine hashes the first key of every command before it runs it, to
    /// prefetch the record, so on the command path the hash is in hand already
    /// and hashing it again would be the second most expensive thing in a
    /// `GET`.
    ///
    /// The hash must be [`Keyspace::hash_of`] of the key. Anything else picks
    /// the wrong stripe, and the wrong stripe is a key that cannot be found
    /// rather than an error, so this is not something to hand a number that
    /// came from somewhere else.
    #[inline]
    #[must_use]
    pub fn stripe_of_hash(&self, hash: u64) -> usize {
        ((hash >> STRIPE_SHIFT) & self.mask) as usize
    }

    /// The stripe `key` is on.
    ///
    /// # Panics
    ///
    /// Never. The mask cannot select a stripe that is not there.
    #[inline]
    #[must_use]
    pub fn at(&mut self, key: &[u8]) -> &mut Keyspace {
        let i = self.stripe_of(key);
        &mut self.stripes[i]
    }

    /// The stripe `key` is on, without taking it mutably.
    #[inline]
    #[must_use]
    pub fn at_ref(&self, key: &[u8]) -> &Keyspace {
        let i = self.stripe_of(key);
        &self.stripes[i]
    }

    /// The stripe a key with this hash is on.
    ///
    /// As [`Db::stripe_of_hash`] for what the hash has to be.
    #[inline]
    #[must_use]
    pub fn at_hashed(&mut self, hash: u64) -> &mut Keyspace {
        let i = self.stripe_of_hash(hash);
        &mut self.stripes[i]
    }

    /// Stripe `i`.
    ///
    /// # Panics
    ///
    /// If `i` is not a stripe. Callers get their index from [`Db::stripe_of`]
    /// or from a walk over [`Db::width`], so an index out of range here is a
    /// bug in the caller.
    #[inline]
    #[must_use]
    pub fn stripe_mut(&mut self, i: usize) -> &mut Keyspace {
        &mut self.stripes[i]
    }

    /// Stripe `i`, without taking it mutably.
    ///
    /// # Panics
    ///
    /// As [`Db::stripe_mut`].
    #[inline]
    #[must_use]
    pub fn stripe(&self, i: usize) -> &Keyspace {
        &self.stripes[i]
    }

    /// The one stripe of a database that has one.
    ///
    /// The bridge, and a temporary one. Everything that reaches a database
    /// today does it by taking the whole thing mutably, because until now
    /// there was only ever one thing to take, and rewriting all of that in the
    /// change that introduces the type would make one unreviewable diff out of
    /// two reviewable ones. So this hands back the single stripe and the
    /// callers that have not been taught about stripes go on working exactly as
    /// they did.
    ///
    /// It is deliberately loud rather than quietly wrong. Nothing asks for a
    /// database wider than one stripe yet, and the check is what makes sure
    /// nothing can start asking before the last caller of this is gone.
    ///
    /// # Panics
    ///
    /// If the database has more than one stripe, which means a caller that
    /// should have been rewritten was not.
    #[must_use]
    pub fn only_mut(&mut self) -> &mut Keyspace {
        assert_eq!(
            self.stripes.len(),
            1,
            "a caller that has not been taught about stripes was handed a striped database"
        );
        &mut self.stripes[0]
    }

    /// The same, without taking it mutably.
    ///
    /// # Panics
    ///
    /// As [`Db::only_mut`].
    #[must_use]
    pub fn only(&self) -> &Keyspace {
        assert_eq!(
            self.stripes.len(),
            1,
            "a caller that has not been taught about stripes was handed a striped database"
        );
        &self.stripes[0]
    }

    /// Every stripe, in order.
    #[must_use]
    pub fn stripes(&self) -> &[Keyspace] {
        &self.stripes
    }

    /// Every stripe, in order, mutably.
    pub fn stripes_mut(&mut self) -> &mut [Keyspace] {
        &mut self.stripes
    }

    /// What time every stripe here thinks it is.
    ///
    /// One reading and not one per stripe. The clock is set on all of them
    /// together at the top of a turn of the loop, so a command that asks two
    /// stripes what the time is has to get the same answer from both or two
    /// keys written by the same command would expire at different moments.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.stripes[0].clock().now_ms()
    }

    /// How many keys are in the database.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stripes.iter().map(Keyspace::len).sum()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stripes.iter().all(Keyspace::is_empty)
    }

    /// How many of the keys have a deadline on them.
    #[must_use]
    pub fn expires(&self) -> usize {
        self.stripes.iter().map(Keyspace::expires).sum()
    }

    /// Throw the whole database away, which is what `FLUSHDB` does.
    pub fn clear(&mut self) {
        for stripe in &mut self.stripes {
            stripe.clear();
        }
    }

    /// Move every clock in the database to `ms`.
    pub fn set_clock_ms(&mut self, ms: u64) {
        for stripe in &mut self.stripes {
            stripe.clock_mut().set(ms);
        }
    }

    /// Turn the running memory total on or off in every stripe.
    pub fn track_memory(&mut self, on: bool) {
        for stripe in &mut self.stripes {
            stripe.track_memory(on);
        }
    }
}

impl Default for Db {
    fn default() -> Db {
        Db::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{Db, MAX_STRIPES};
    use crate::{Clock, Keyspace};

    #[test]
    fn a_width_is_always_a_power_of_two_and_never_zero() {
        for asked in [0, 1, 2, 3, 5, 8, 9, 100] {
            let db = Db::with_clock(Clock::system(), asked);
            assert!(db.width().is_power_of_two());
            assert!(db.width() >= asked.max(1));
        }
        assert_eq!(Db::with_clock(Clock::system(), 10_000).width(), MAX_STRIPES);
    }

    #[test]
    fn one_stripe_takes_every_key() {
        let db = Db::with_clock(Clock::system(), 1);
        for i in 0..1000u32 {
            assert_eq!(db.stripe_of(&i.to_le_bytes()), 0);
        }
    }

    #[test]
    fn a_key_always_answers_the_same_stripe() {
        let db = Db::with_clock(Clock::system(), 16);
        for i in 0..1000u32 {
            let key = i.to_le_bytes();
            let first = db.stripe_of(&key);
            assert_eq!(db.stripe_of(&key), first);
            assert_eq!(db.stripe_of_hash(Keyspace::hash_of(&key)), first);
        }
    }

    // Not a claim about the hash, a claim about which bits of it are read. A
    // stripe number cut from bits the index also reads would still be stable
    // and would still spread, and would still put every key on a stripe into
    // one corner of that stripe's index. This is the cheapest check that the
    // bits are being taken from somewhere: a thousand keys over sixteen
    // stripes leaves none of them empty unless the number is nearly constant.
    #[test]
    fn the_stripe_number_moves_with_the_key() {
        let db = Db::with_clock(Clock::system(), 16);
        let mut seen = [0usize; 16];
        for i in 0..1000u32 {
            seen[db.stripe_of(&i.to_le_bytes())] += 1;
        }
        assert!(
            seen.iter().all(|&n| n > 0),
            "some stripe took no keys: {seen:?}"
        );
    }

    #[test]
    fn a_key_written_to_its_stripe_is_found_on_its_stripe() {
        let mut db = Db::with_clock(Clock::system(), 8);
        for i in 0..200u32 {
            let key = i.to_le_bytes();
            assert!(db.at(&key).setnx(&key, b"x").unwrap());
        }
        assert_eq!(db.len(), 200);
        for i in 0..200u32 {
            let key = i.to_le_bytes();
            assert!(db.at(&key).exists(&key));
        }
        db.clear();
        assert!(db.is_empty());
    }
}
