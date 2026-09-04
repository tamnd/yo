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
/// walks [`Db::stripes`], and the answers it adds up are the same answers a
/// single keyspace would have given.
pub struct Db {
    /// The stripes, always a power of two of them and always at least one.
    stripes: Vec<Keyspace>,
    /// `stripes.len() - 1`, kept here so the hot path is a shift and an and
    /// rather than a division.
    mask: u64,
    /// Somewhere to put bytes that came out of one stripe and are wanted while
    /// another stripe is being held. A key for [`Db::random_key`], the sources
    /// of a `BITOP` for `Db::bitop`. Every stripe has a buffer of its own for
    /// its own work, and this is the one for work that is nobody's.
    scratch: Vec<u8>,
    /// Where each of the things in `scratch` ends, for the callers that put
    /// more than one thing in it.
    rows: Vec<usize>,
    /// The tables a set operation across stripes fills in.
    ///
    /// A keyspace keeps a pair of these for the set operations that happen
    /// inside it, for the reason [`crate::setops::Scratch`] gives: building the
    /// table per call was most of what a `SUNION` over text sets did. An
    /// operation whose keys are on several stripes is not any one stripe's, so
    /// it gets its own.
    setops: crate::setops::Scratch,
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
            scratch: Vec::new(),
            rows: Vec::new(),
            setops: crate::setops::Scratch::new(),
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

    /// The same, without taking it mutably.
    ///
    /// What the prefetch stage uses. It warms a cache line for a key it has the
    /// hash of and it runs before anything is executed, so it can neither take
    /// the database mutably nor be handed the key.
    #[inline]
    #[must_use]
    pub fn at_ref_hashed(&self, hash: u64) -> &Keyspace {
        let i = self.stripe_of_hash(hash);
        &self.stripes[i]
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

    /// The two buffers a command that spans stripes builds its answer in.
    ///
    /// Taken out rather than lent, because everything the caller does with them
    /// is done while holding a stripe, and a stripe is part of this database.
    /// Whoever takes them puts them back with [`Db::put_scratch`], and that is
    /// what makes the second run of the same command cost no allocation.
    pub(crate) fn take_scratch(&mut self) -> (Vec<u8>, Vec<usize>) {
        (
            std::mem::take(&mut self.scratch),
            std::mem::take(&mut self.rows),
        )
    }

    /// The buffers back, ready for the next command.
    pub(crate) fn put_scratch(&mut self, scratch: Vec<u8>, rows: Vec<usize>) {
        self.scratch = scratch;
        self.rows = rows;
    }

    /// What is in the byte buffer, for a caller that put something there and
    /// wants to hand it back out borrowed.
    ///
    /// `LMOVE` across two stripes is the one that does that: the element it
    /// moved has no structure left to borrow from once it has been pushed, so
    /// the answer borrows this instead, exactly as it borrows a stripe's own
    /// buffer when both keys are on one stripe.
    pub(crate) fn scratch_bytes(&self) -> &[u8] {
        &self.scratch
    }

    /// The set operation tables, taken out for the same reason as the buffers.
    pub(crate) fn take_setops(&mut self) -> crate::setops::Scratch {
        std::mem::take(&mut self.setops)
    }

    /// The tables back.
    pub(crate) fn put_setops(&mut self, setops: crate::setops::Scratch) {
        self.setops = setops;
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
    /// Every command group has been taught about stripes now, so the last
    /// caller left is the server's own handle on a database, and this goes when
    /// that one does.
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
        &mut self,
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
            let next = self.stripes[at].scan(cursor, budget, ty, |key| {
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
    pub fn keys(&mut self, mut out: impl FnMut(&[u8])) {
        for stripe in &mut self.stripes {
            stripe.keys(&mut out);
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
    /// The answer is copied into this database's own buffer rather than
    /// borrowed from the stripe's, which is what lets the loop above go round
    /// again while holding an answer. The buffer is kept between draws, so the
    /// copy is a few bytes and the allocator only hears about a key name longer
    /// than any this database has answered with before.
    pub fn random_key(&mut self) -> Option<&[u8]> {
        let live = self.len();
        if live == 0 {
            return None;
        }
        let draw = (self.stripes[0].random() % live as u64) as usize;
        let mut running = 0;
        let mut first = 0;
        for (i, stripe) in self.stripes.iter().enumerate() {
            running += stripe.len();
            if draw < running {
                first = i;
                break;
            }
        }
        // Taken out and put back, because the closest thing to a stripe's
        // answer this can hold is a copy of it and the stripe is borrowed while
        // the copy is being made.
        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let mut found = false;
        for step in 0..self.stripes.len() {
            let i = (first as u64 + step as u64) & self.mask;
            if let Some(key) = self.stripes[i as usize].random_key() {
                // `yo_alloc::high_water` because this is the buffer reaching a
                // length it has not been asked for before, which is the longest
                // key name this database has ever answered with. The draw after
                // it pays nothing, and that is the test below.
                yo_alloc::high_water(|| buf.extend_from_slice(key));
                found = true;
                break;
            }
        }
        self.scratch = buf;
        found.then_some(self.scratch.as_slice())
    }
}

impl Default for Db {
    fn default() -> Db {
        Db::new()
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
        let mut db = filled(8, 2_000);
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
        let mut db = filled(8, 2_000);
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
            let key = db.random_key().expect("the database is not empty").to_vec();
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
            assert_eq!(db.random_key(), Some(&b"k4242"[..]));
        }
        db.at(b"k4242").del(b"k4242");
        assert_eq!(db.random_key(), None);
    }

    /// The buffer the answer is copied into is kept, so the draw after the
    /// first one of that length pays the allocator nothing. That is the claim
    /// the `yo_alloc::high_water` in `random_key` is making.
    #[test]
    fn a_second_random_key_does_not_allocate() {
        let mut db = filled(8, 500);
        assert!(db.random_key().is_some(), "the database is not empty");
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..200 {
                assert!(db.random_key().is_some(), "the database is not empty");
            }
        });
        assert_eq!(
            allocs, 0,
            "randomkey allocated {allocs} times in two hundred"
        );
    }
}
