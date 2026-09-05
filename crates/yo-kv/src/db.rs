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
//!
//! What more than one stripe cost was two things, and both of them are paid.
//! A command that names several keys is no longer handed one keyspace and
//! resolves each key against the database instead, and everything that walks a
//! whole database walks all of the stripes: the expiry cycle, eviction,
//! compaction, `SCAN`, `KEYS`, `RANDOMKEY`, the settings and the snapshot. What
//! is left before a database can be held by more than one thread is the engine
//! itself, which is the other half of this milestone.

use yo_common::Small;
use yo_common::lock::{Held, Lock};
use yo_index::Cursor as KeyCursor;

use crate::value::Kind;
use crate::{Clock, Keyspace};

/// The most stripes one database can be cut into.
///
/// Two fields decide this and they agree. Eight bits of hash are free above the
/// directory and they are what chooses the stripe, and eight bits of a `SCAN`
/// cursor are free above the bucket and they are what remembers which stripe a
/// walk had got to. It is far more than a machine has cores and the point of
/// the ceiling is not to be reached, it is to keep both of those choices inside
/// a field nothing else is reading.
pub const MAX_STRIPES: usize = 1 << yo_index::STRIPE_BITS;

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
/// walks [`Db::stripes_mut`], and the answers it adds up are the same answers a
/// single keyspace would have given.
pub struct Db {
    /// The stripes, always a power of two of them and always at least one.
    ///
    /// Each one behind a lock, because a database is going to be reached by
    /// more than one thread and a stripe is the piece one command holds. A
    /// caller that has this database by exclusive reference does not go near
    /// the locks: the borrow checker has already proved that nobody else is
    /// looking, so `at` and every whole database walk go through `get_mut` and
    /// pay nothing. The locks are for the callers that have it shared.
    stripes: Vec<Lock<Keyspace>>,
    /// `stripes.len() - 1`, kept here so the hot path is a shift and an and
    /// rather than a division.
    mask: u64,
    /// What every stripe's clock was last set to.
    ///
    /// A copy rather than a lookup, so that asking what the time is does not
    /// mean taking a stripe. Every stripe carries the same reading and they are
    /// all moved together, so this is that reading and not a fourth opinion.
    clock: Clock,
    /// The buffers for work that is not any one stripe's, behind a lock of
    /// their own. [`Db::spare`] has the order they are taken in and why there
    /// is one set of them rather than one per thread.
    spare: Lock<Spare>,
}

/// Somewhere to put bytes and rows that came out of one stripe and are wanted
/// while another stripe is being held.
///
/// The sources of a `BITOP`, the element a cross stripe `LMOVE` is carrying,
/// the tables a set operation fills in. Every stripe has buffers of its own for
/// its own work, and these are the ones for work that is nobody's.
#[derive(Default)]
pub(crate) struct Spare {
    /// The bytes.
    pub(crate) bytes: Vec<u8>,
    /// Where each of the things in `bytes` ends, for the callers that put more
    /// than one thing in it.
    pub(crate) rows: Vec<usize>,
    /// The tables a set operation across stripes fills in.
    ///
    /// A keyspace keeps a pair of these for the set operations that happen
    /// inside it, for the reason [`crate::setops::Scratch`] gives: building the
    /// table per call was most of what a `SUNION` over text sets did. An
    /// operation whose keys are on several stripes is not any one stripe's, so
    /// it gets its own.
    pub(crate) setops: crate::setops::Scratch,
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
            stripes: (0..n)
                .map(|_| Lock::new(Keyspace::with_clock(clock)))
                .collect(),
            mask: (n - 1) as u64,
            clock,
            spare: Lock::new(Spare {
                bytes: Vec::new(),
                rows: Vec::new(),
                setops: crate::setops::Scratch::new(),
            }),
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
        self.stripes[i].get_mut()
    }

    /// The stripe `key` is on, held.
    ///
    /// For a caller that has the database shared, which is every caller once
    /// there is more than one thread. The stripe is released when the answer is
    /// dropped, so a caller that wants it for the length of a command has to
    /// keep the answer for the length of the command rather than write it into
    /// the middle of a larger expression.
    #[inline]
    #[must_use]
    pub fn hold(&self, key: &[u8]) -> Held<'_, Keyspace> {
        let i = self.stripe_of(key);
        self.stripes[i].lock()
    }

    /// The stripe a key with this hash is on.
    ///
    /// As [`Db::stripe_of_hash`] for what the hash has to be.
    #[inline]
    #[must_use]
    pub fn at_hashed(&mut self, hash: u64) -> &mut Keyspace {
        let i = self.stripe_of_hash(hash);
        self.stripes[i].get_mut()
    }

    /// The stripe a key with this hash is on, held.
    ///
    /// As [`Db::stripe_of_hash`] for what the hash has to be.
    #[inline]
    #[must_use]
    pub fn hold_hashed(&self, hash: u64) -> Held<'_, Keyspace> {
        let i = self.stripe_of_hash(hash);
        self.stripes[i].lock()
    }

    /// Warm the line a key with this hash is going to be read from, if the
    /// stripe it is on is not busy.
    ///
    /// What the prefetch stage uses, and the one place a lock is not worth
    /// waiting for. A prefetch is a hint about a command that has not started
    /// yet, so a stripe that somebody else is holding is a stripe whose lines
    /// are being pulled about anyway, and waiting for it would turn a hint into
    /// a wait for another thread. It is skipped instead.
    #[inline]
    pub fn prefetch_hashed(&self, hash: u64) {
        let i = self.stripe_of_hash(hash);
        if let Some(stripe) = self.stripes[i].try_lock() {
            stripe.prefetch(hash);
        }
    }

    /// The one stripe every one of `keys` is on, or `None` when they are spread
    /// over more than one.
    ///
    /// This is what a command that names several keys asks first. A database of
    /// one stripe always answers `Some(0)`, so the old path stays the path, and
    /// a wide database answers it often enough to be worth asking: a client that
    /// hash tags its keys the way a cluster makes it does it so that its
    /// multi key commands land in one place, and this is that place.
    #[must_use]
    pub fn one_stripe<'k>(&self, mut keys: impl Iterator<Item = &'k [u8]>) -> Option<usize> {
        let first = self.stripe_of(keys.next()?);
        keys.all(|key| self.stripe_of(key) == first)
            .then_some(first)
    }

    /// The buffers a command that spans stripes builds its answer in.
    ///
    /// Held for the length of the command and taken before any stripe is. That
    /// order is the rule and it is the only lock order this database has: every
    /// caller that wants both wants the spare first and the stripes after, in
    /// stripe order, so no two of them can be waiting for each other.
    ///
    /// One set of buffers per database and not per thread, so two threads
    /// running a command that spans stripes on the same database take turns.
    /// That is a real serialisation and it is deliberate: the commands that
    /// come through here are the set algebra and the two key moves, which are
    /// a rounding error in any cache workload, and the alternative is either
    /// buffers per thread that nothing frees or an allocation on a command
    /// path, and Y7 does not allow the second one.
    pub(crate) fn spare(&self) -> Held<'_, Spare> {
        self.spare.lock()
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
        self.stripes[i].get_mut()
    }

    /// Stripe `i`, held.
    ///
    /// # Panics
    ///
    /// As [`Db::stripe_mut`], and also if the calling thread is already holding
    /// this stripe, in a debug build. Holding one twice is a wait for yourself
    /// and the lock says so rather than stopping.
    #[inline]
    #[must_use]
    pub fn hold_stripe(&self, i: usize) -> Held<'_, Keyspace> {
        self.stripes[i].lock()
    }

    /// Every stripe named, each one once, held, in stripe order.
    ///
    /// The order is what makes this safe to call while another database is
    /// being held elsewhere and what makes two commands that want the same pair
    /// of stripes want them the same way round. The names are deduplicated
    /// because two keys of a multi key command land on one stripe often enough,
    /// and asking for a stripe twice is the mistake the lock panics about.
    #[must_use]
    pub fn hold_many(&self, homes: impl Iterator<Item = usize>) -> Holds<'_> {
        let mut want: Small<u16, INLINE_HOLDS> = homes.map(|i| i as u16).collect();
        want.sort_unstable();
        let mut out = Holds::new();
        // A stripe number is eight bits, so this can never be one of them, which
        // is what makes it the mark for nothing taken yet.
        let mut last = u16::MAX;
        for &i in want.iter() {
            if i == last {
                continue;
            }
            last = i;
            out.push(i, self.stripes[usize::from(i)].lock());
        }
        out
    }

    /// The same, for a command that has keys rather than stripe numbers.
    ///
    /// Which is most of them: a multi key command is handed the names off the
    /// wire and works out where they live here. Two keys on one stripe hold it
    /// once, so `MGET a a` and `MGET a b` where both land in the same place are
    /// one hold and not two.
    #[must_use]
    pub fn hold_keys<'k>(&self, keys: impl Iterator<Item = &'k [u8]>) -> Holds<'_> {
        self.hold_many(keys.map(|key| self.stripe_of(key)))
    }

    /// Every stripe, in order, mutably.
    ///
    /// Free, because an exclusive reference to the database is already an
    /// exclusive reference to every stripe in it.
    pub fn stripes_mut(&mut self) -> impl Iterator<Item = &mut Keyspace> {
        self.stripes.iter_mut().map(Lock::get_mut)
    }

    /// What time every stripe here thinks it is.
    ///
    /// One reading and not one per stripe. The clock is set on all of them
    /// together at the top of a turn of the loop, so a command that asks two
    /// stripes what the time is has to get the same answer from both or two
    /// keys written by the same command would expire at different moments. The
    /// reading is kept here as well as in the stripes so that asking the time
    /// does not mean taking one of them.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.clock.now_ms()
    }

    /// How many keys are in the database.
    ///
    /// One stripe at a time and never two at once, so this is a sum of counts
    /// that were each true when it was read rather than a count of the database
    /// at one moment. `DBSIZE` on a server that is being written to was already
    /// that answer.
    #[must_use]
    pub fn len(&self) -> usize {
        (0..self.stripes.len())
            .map(|i| self.stripes[i].lock().len())
            .sum()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        (0..self.stripes.len()).all(|i| self.stripes[i].lock().is_empty())
    }

    /// How many of the keys have a deadline on them.
    #[must_use]
    pub fn expires(&self) -> usize {
        (0..self.stripes.len())
            .map(|i| self.stripes[i].lock().expires())
            .sum()
    }

    /// Throw the whole database away, which is what `FLUSHDB` does.
    ///
    /// One stripe at a time, the same as every other walk here. A reader on
    /// another thread can see a database that is half thrown away, which is the
    /// same thing it can see of a `FLUSHDB` on any server that does not stop
    /// the world for one.
    pub fn clear(&self) {
        for i in 0..self.stripes.len() {
            self.hold_stripe(i).clear();
        }
    }

    /// Move every clock in the database to `ms`.
    ///
    /// Exclusive, and it stays exclusive. A clock is read on every command and
    /// moved by whatever is turning the loop, so this is the one thing here
    /// that a thread must not be doing while another thread is serving, and the
    /// borrow checker saying so is the cheapest way to keep it that way.
    pub fn set_clock_ms(&mut self, ms: u64) {
        self.clock.set(ms);
        for stripe in self.stripes_mut() {
            stripe.clock_mut().set(ms);
        }
    }

    /// Turn the running memory total on or off in every stripe.
    pub fn track_memory(&self, on: bool) {
        for i in 0..self.stripes.len() {
            self.hold_stripe(i).track_memory(on);
        }
    }

    /// A batch of keys and where the next batch starts, over the whole
    /// database.
    ///
    /// This is `SCAN`, and it is one stripe at a time. The cursor carries the
    /// stripe it had got to as well as the place in that stripe, which is what
    /// the spare field in [`yo_index::Cursor`] is for.
    ///
    /// The promise a single keyspace makes survives being made one stripe at a
    /// time, and it survives it for one reason: a key never changes stripe. So
    /// a key that is there for the whole walk is on a stripe this walk has not
    /// reached yet or on the one it is in the middle of, and either way it is
    /// still coming. Nothing a writer does while the walk is going can move a
    /// key from a stripe that is still to come to a stripe that is already
    /// done.
    ///
    /// `budget` is spent per stripe rather than per call, so a call that
    /// finishes a stripe exactly on the budget stops there rather than starting
    /// the next one. What it will do is walk past any number of empty stripes,
    /// because a stripe with nothing in it is a walk of one segment and
    /// stopping to hand the client a cursor for it would be the more expensive
    /// of the two.
    pub fn scan(
        &self,
        from: KeyCursor,
        budget: usize,
        ty: Option<Kind>,
        mut out: impl FnMut(&[u8]),
    ) -> KeyCursor {
        // A cursor naming a stripe this database does not have, which a client
        // can produce by holding one across a server that came back narrower.
        // It reads as a walk that is over, which is what Redis gives for any
        // cursor it cannot make sense of, and the client starts again.
        let mut at = from.stripe();
        if at >= self.stripes.len() {
            return KeyCursor::START;
        }
        let mut cursor = from.without_stripe();
        let mut seen = 0usize;
        while at < self.stripes.len() {
            let next = self.hold_stripe(at).scan(cursor, budget, ty, |key| {
                seen += 1;
                out(key);
            });
            if !next.is_end() {
                return next.with_stripe(at);
            }
            at += 1;
            cursor = KeyCursor::START;
            if at < self.stripes.len() && seen >= budget {
                return KeyCursor::START.with_stripe(at);
            }
        }
        KeyCursor::START
    }

    /// Every key in the database, once each.
    ///
    /// This is `KEYS`, and it is every key of every stripe. The order is the
    /// order the stripes are in and then whatever order each one walks in,
    /// which is no order at all as far as a client is concerned, the same as it
    /// was with one stripe.
    pub fn keys(&self, mut out: impl FnMut(&[u8])) {
        for i in 0..self.stripes.len() {
            self.hold_stripe(i).keys(&mut out);
        }
    }

    /// One key from anywhere in the database, or `None` if there are none.
    ///
    /// This is `RANDOMKEY`. The stripe is drawn first, weighted by how many
    /// keys each one holds, so a database whose stripes came out uneven does
    /// not answer the small ones as often as the big ones. Then that stripe
    /// picks a key the way it always did.
    ///
    /// A stripe can still answer nothing, because its count includes keys whose
    /// deadline has gone and which nothing has collected yet. The stripes after
    /// it are asked in turn when that happens, so an answer of `None` here
    /// means every stripe was asked and none of them had a live key.
    ///
    /// The key is handed to `f` while its stripe is still held rather than
    /// answered, because the stripe it came out of is what it is borrowed from
    /// and letting go of that stripe is the end of the borrow. The caller
    /// writes it into a reply, which is not part of this database and is
    /// therefore still there afterwards. `false` means no stripe had one.
    pub fn random_key(&self, f: impl FnOnce(&[u8])) -> bool {
        let live: usize = (0..self.stripes.len())
            .map(|i| self.hold_stripe(i).len())
            .sum();
        if live == 0 {
            return false;
        }
        let draw = (self.hold_stripe(0).random() % live as u64) as usize;
        let mut running = 0;
        let mut first = 0;
        for i in 0..self.stripes.len() {
            running += self.hold_stripe(i).len();
            if draw < running {
                first = i;
                break;
            }
        }
        let mut f = Some(f);
        for step in 0..self.stripes.len() {
            let i = (first as u64 + step as u64) & self.mask;
            let mut stripe = self.hold_stripe(i as usize);
            if let Some(key) = stripe.random_key() {
                // The closure is taken out of the option rather than called in
                // place, because it is `FnOnce` and the loop it is inside can
                // go round again. It never does after this point, which the
                // return says.
                f.take().expect("the loop stops the first time it fires")(key);
                return true;
            }
        }
        false
    }
}

impl Default for Db {
    fn default() -> Db {
        Db::new()
    }
}

/// How many stripes one command can name before the list of them reaches the
/// heap.
///
/// Eight, which is the same number the set operations use for the keys
/// themselves and for the same reason: a command over more operands than that
/// is rare enough that the cost of it is not what anyone is measuring, and a
/// command path is not allowed to allocate. A stripe is named once however many
/// of the keys are on it, so eight here covers more than eight keys.
const INLINE_HOLDS: usize = 8;

/// Several stripes of one database, held at once.
///
/// What a command whose keys are spread over the database gets from
/// [`Db::hold_many`]. It is a list rather than a map because the number of
/// stripes a command names is at most the number of keys it names, which is
/// small, and walking a handful of pairs is cheaper than anything with a hash
/// in it. The list is in stripe order, which is what [`Db::hold_many`] promises
/// and what keeps two of these from waiting on each other.
pub struct Holds<'a> {
    /// The first few, where the answer nearly always fits.
    room: [Option<(u16, Held<'a, Keyspace>)>; INLINE_HOLDS],
    /// How many of `room` are in use.
    n: usize,
    /// The rest, for a command that named keys on more stripes than there is
    /// room for above. This is the one path here that allocates and the reason
    /// it is allowed to is that reaching it means a client sent a command over
    /// nine or more stripes, which no benchmark and no real workload does.
    spill: Vec<(u16, Held<'a, Keyspace>)>,
}

impl<'a> Holds<'a> {
    /// Holding nothing.
    fn new() -> Holds<'a> {
        Holds {
            room: [const { None }; INLINE_HOLDS],
            n: 0,
            spill: Vec::new(),
        }
    }

    /// One more, which the caller has already taken and which comes after every
    /// stripe added before it.
    fn push(&mut self, home: u16, held: Held<'a, Keyspace>) {
        if self.n < INLINE_HOLDS {
            self.room[self.n] = Some((home, held));
            self.n += 1;
        } else {
            self.spill.push((home, held));
        }
    }

    /// Stripe `i`, which the caller asked for and is therefore holding.
    ///
    /// # Panics
    ///
    /// If `i` was not one of the stripes asked for, which is a caller that
    /// worked out a stripe number twice and got two different answers.
    #[must_use]
    pub fn stripe(&self, i: usize) -> &Keyspace {
        let home = i as u16;
        for slot in self.room[..self.n].iter().flatten() {
            if slot.0 == home {
                return &slot.1;
            }
        }
        let at = self
            .spill
            .binary_search_by_key(&home, |&(where_, _)| where_)
            .expect("a stripe that was asked for");
        &self.spill[at].1
    }

    /// The same, for writing.
    ///
    /// One stripe at a time even though every one of them is held, because a
    /// command that writes into two of them at once would need the borrow
    /// checker told they are different ones and nothing here does that. A
    /// rename takes the record out of one and puts it in the other, and the
    /// record owns what it holds in between.
    ///
    /// # Panics
    ///
    /// As [`Holds::stripe`].
    #[must_use]
    pub fn stripe_mut(&mut self, i: usize) -> &mut Keyspace {
        let home = i as u16;
        for slot in self.room[..self.n].iter_mut().flatten() {
            if slot.0 == home {
                return &mut slot.1;
            }
        }
        let at = self
            .spill
            .binary_search_by_key(&home, |&(where_, _)| where_)
            .expect("a stripe that was asked for");
        &mut self.spill[at].1
    }

    /// How many stripes are being held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n + self.spill.len()
    }

    /// Whether none are, which is a command that named no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use yo_index::Cursor as KeyCursor;

    use super::{Db, MAX_STRIPES};
    use crate::{Clock, Keyspace};

    fn filled(stripes: usize, keys: u32) -> Db {
        let mut db = Db::with_clock(Clock::fixed(1_000_000), stripes);
        for i in 0..keys {
            let key = format!("k{i}").into_bytes();
            db.at(&key).setnx(&key, b"v").expect("room for a record");
        }
        db
    }

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

    /// The question every multi key command asks before it does anything.
    #[test]
    fn a_list_of_keys_is_on_one_stripe_or_it_is_not() {
        let names: [&[u8]; 3] = [b"a", b"b", b"c"];
        let one = Db::with_clock(Clock::system(), 1);
        assert_eq!(one.one_stripe(names.into_iter()), Some(0));
        assert_eq!(one.one_stripe(std::iter::empty()), None);

        // Sixteen stripes and three keys, which land together about one time in
        // two hundred and fifty and are checked here to be sure they have not.
        let many = Db::with_clock(Clock::system(), 16);
        assert_eq!(many.one_stripe(names.into_iter()), None);
        let home = many.stripe_of(b"a");
        assert_eq!(many.one_stripe(std::iter::once(&b"a"[..])), Some(home));
        assert_eq!(many.one_stripe([&b"a"[..], b"a"].into_iter()), Some(home));
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

    #[test]
    fn a_scan_walks_every_stripe_and_answers_every_key_once() {
        let db = filled(8, 2_000);
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut at = KeyCursor::START;
        let mut calls = 0;
        loop {
            at = db.scan(at, 10, None, |key| seen.push(key.to_vec()));
            calls += 1;
            if at.is_end() {
                break;
            }
            assert!(calls < 10_000, "a scan that will not finish");
        }
        let unique: HashSet<Vec<u8>> = seen.iter().cloned().collect();
        assert_eq!(unique.len(), 2_000);
        assert_eq!(seen.len(), 2_000, "a quiet scan returned a key twice");

        let mut walked = HashSet::new();
        db.keys(|key| {
            walked.insert(key.to_vec());
        });
        assert_eq!(unique, walked);
    }

    // Not about the keys, about the number in the middle of the cursor. Every
    // batch after the first stripe has one in it, and a client that has held
    // one from a database that had more stripes than this one gets an answer
    // that says the walk is over rather than a panic.
    #[test]
    fn a_scan_carries_the_stripe_in_the_cursor() {
        let db = filled(8, 2_000);
        let first = db.scan(KeyCursor::START, 10, None, |_| {});
        assert!(!first.is_end());

        let mut stripes = HashSet::new();
        let mut at = KeyCursor::START;
        loop {
            at = db.scan(at, 10, None, |_| {});
            if at.is_end() {
                break;
            }
            stripes.insert(at.stripe());
        }
        assert_eq!(stripes.len(), 8, "some stripe was never the one in hand");

        let beyond = KeyCursor::START.with_stripe(9);
        let mut any = false;
        assert!(db.scan(beyond, 10, None, |_| any = true).is_end());
        assert!(!any);
    }

    #[test]
    fn a_random_key_comes_from_whichever_stripe_still_has_one() {
        let mut db = filled(8, 5_000);
        let mut all = HashSet::new();
        db.keys(|key| {
            all.insert(key.to_vec());
        });
        let mut picked = HashSet::new();
        for _ in 0..200 {
            let mut key = Vec::new();
            assert!(
                db.random_key(|k| key.extend_from_slice(k)),
                "the database is not empty"
            );
            assert!(all.contains(&key), "a key that is not there");
            picked.insert(key);
        }
        assert!(picked.len() > 10, "only {} distinct keys", picked.len());

        // One key left in a database of eight stripes, so seven of the eight
        // have nothing to answer with and the draw lands on one of those seven
        // nearly every time.
        for i in 0..5_000u32 {
            if i != 4_242 {
                let key = format!("k{i}").into_bytes();
                db.at(&key).del(&key);
            }
        }
        for _ in 0..20 {
            let mut key = Vec::new();
            assert!(db.random_key(|k| key.extend_from_slice(k)));
            assert_eq!(key, b"k4242");
        }
        db.at(b"k4242").del(b"k4242");
        assert!(!db.random_key(|_| unreachable!("there are no keys left")));
    }

    /// The key is handed out of the stripe it is on rather than copied, so a
    /// draw asks the allocator for nothing at all.
    #[test]
    fn a_random_key_does_not_allocate() {
        let db = filled(8, 500);
        assert!(db.random_key(|_| {}), "the database is not empty");
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..200 {
                assert!(db.random_key(|_| {}), "the database is not empty");
            }
        });
        assert_eq!(
            allocs, 0,
            "randomkey allocated {allocs} times in two hundred"
        );
    }

    /// A command names its stripes in whatever order its keys arrived in and
    /// names some of them twice. What comes back is each one once, and every
    /// one of them is the stripe that was asked for.
    #[test]
    fn holding_several_stripes_takes_each_of_them_once() {
        let mut db = filled(16, 400);
        let counts: Vec<usize> = (0..db.width()).map(|i| db.stripe_mut(i).len()).collect();
        let held = db.hold_many([9, 2, 9, 0, 2, 15].into_iter());
        assert_eq!(held.len(), 4, "six names, four stripes");
        assert!(!held.is_empty());
        for i in [0, 2, 9, 15] {
            assert_eq!(held.stripe(i).len(), counts[i], "stripe {i} came back");
        }
    }

    /// Nothing named is nothing held, which is what a command with no keys
    /// left after the missing ones were dropped hands back.
    #[test]
    fn holding_no_stripes_holds_nothing() {
        let db = filled(4, 40);
        let held = db.hold_many(std::iter::empty());
        assert!(held.is_empty());
        assert_eq!(held.len(), 0);
    }

    /// Y7 covers this the moment a database is wide, because a set operation
    /// over keys on several stripes is a command path. Eight stripes fit
    /// without the heap and the ninth is the one that is allowed to reach for
    /// it.
    #[test]
    fn holding_up_to_eight_stripes_does_not_allocate() {
        let db = filled(16, 400);
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..50 {
                let held = db.hold_many((0..8).rev());
                assert_eq!(held.len(), 8);
            }
        });
        assert_eq!(allocs, 0, "holding eight stripes allocated {allocs} times");
    }

    /// And more than eight still works, which is the part the spill is there
    /// for.
    #[test]
    fn holding_more_stripes_than_there_is_room_for_still_holds_them_all() {
        let mut db = filled(16, 400);
        let counts: Vec<usize> = (0..db.width()).map(|i| db.stripe_mut(i).len()).collect();
        let held = db.hold_many((0..16).rev());
        assert_eq!(held.len(), 16);
        for (i, &was) in counts.iter().enumerate() {
            assert_eq!(held.stripe(i).len(), was, "stripe {i} came back");
        }
    }
}
