//! The active expiry cycle, which is what reclaims a key nobody asks for again.
//!
//! Lazy expiry answers the correctness question on its own: a key past its
//! deadline is not returned to any client, because every read reaps it on the way
//! past. What it does not answer is the memory question. A cache that writes ten
//! million keys with a one hour deadline and then never reads them again holds
//! all ten million of them forever under lazy expiry alone, because nothing ever
//! goes past them. That is the whole reason Redis runs a cycle, and `14` section
//! 1 asks for the same thing here.
//!
//! # A budget in keys looked at
//!
//! Redis samples twenty keys from its expires dictionary, deletes the ones that
//! are past, and goes round again while more than a quarter of what it sampled
//! was dead. The rule adapts: a database full of dead keys gets swept hard and a
//! database with a few gets one cheap look.
//!
//! The sample comes off a second index of just the keys that carry a deadline,
//! the way Redis's comes off `db->expires`. That index lives in the map, because
//! the map is the only thing that knows where a record is and the only thing
//! that moves one, and it is described where it is kept. What it buys here is
//! that every key this looks at is a key that could have expired, so the quarter
//! rule is Redis's quarter over Redis's denominator and a database where one key
//! in a million is volatile costs the same per round as one where all of them
//! are.
//!
//! This used to sample the main index and skip most of what it found, and the
//! shape of that is worth remembering, because it is what the budget still
//! protects against. A sweep over a mostly non volatile database spent its whole
//! budget walking past keys with nothing wrong with them, and the ratio it then
//! judged had to be taken over the volatile keys alone rather than over
//! everything looked at, because a quarter of every key sampled is a bar that a
//! database which is one percent volatile can never clear however much dead
//! memory is sitting in it. Both denominators are the same now, and the code
//! still counts them separately because the difference between them is exactly
//! the thing a test should be able to see going wrong.
//!
//! The common case is still the one that costs nothing: a count of the keys
//! carrying a deadline sits in the keyspace, and a zero there ends this before it
//! draws anything.

use crate::hash::Hash;
use crate::keyspace::Keyspace;
use crate::value;
use yo_common::Addr;

/// Keys with a deadline that one round looks at before it decides.
///
/// Redis's `ACTIVE_EXPIRE_CYCLE_KEYS_PER_LOOP`, and the same twenty. It is the
/// sample size the quarter rule is judged on, so it wants to be small enough
/// that a round is cheap and large enough that the ratio means something. Twenty
/// gives the rule a resolution of five percent, which is finer than the quarter
/// it is compared against.
const PER_ROUND: usize = 20;

/// What one call to the cycle did.
///
/// Three numbers rather than one, because they answer different questions. The
/// caller charges its budget against `examined`, a test asserts on `expired`, and
/// `volatile` is what says whether a cheap sweep found nothing because there was
/// nothing dead or because it never got near a key that could be.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Cycle {
    /// Keys the sample walked past, whether or not they had a deadline.
    pub examined: usize,
    /// Of those, how many carried a deadline.
    pub volatile: usize,
    /// Of those, how many were past it and were dropped.
    pub expired: usize,
}

impl Keyspace {
    /// Sweep dead keys until the budget runs out or the sweep stops paying.
    ///
    /// `budget` is how many keys this is allowed to look at, and it is a ceiling
    /// and not a target: a database with nothing dead in it returns after one
    /// round having spent a fraction of it, and a database with nothing volatile
    /// in it returns having spent none of it at all.
    ///
    /// Safe to call on any database at any time. It takes only keys that are past
    /// their deadline, which are keys no client can see, so nothing observable
    /// changes except the memory going back and `INFO stats` counting the
    /// reclaim. Redis counts its cycle into `expired_keys` alongside lazy expiry
    /// and so does this.
    pub fn expire_cycle(&mut self, budget: usize) -> Cycle {
        let mut c = Cycle::default();
        // The point of the count. A database where no key has a deadline is the
        // common one, and this is where it finds that out, for one comparison
        // rather than for a walk of a segment that was never going to hold
        // anything worth taking.
        if budget == 0 || self.expires() == 0 {
            return c;
        }
        let now = self.clock.now_ms();
        loop {
            let round = self.sweep_round(now, budget - c.examined, &mut c);
            // Redis's quarter rule, over the keys that could have expired rather
            // than over every key looked at. Both of the stops below matter: the
            // budget bounds the worst case and the ratio ends a sweep that has
            // stopped finding anything, which is what keeps an idle server from
            // spending its whole slice on a database that is already clean.
            if c.examined >= budget || round.expired * 4 <= round.volatile {
                return c;
            }
        }
    }

    /// One round of twenty, which is a draw and then the deletions it found.
    ///
    /// The two halves are separate because the sample holds the map still: it
    /// hands out an address and a borrow, and deleting is a write. So the round
    /// writes down what it found, lets go, and then drops. The addresses survive
    /// that gap because freeing a record only moves a counter, and the one thing
    /// that does move records is compaction, which runs quiesced and cannot be
    /// underneath this.
    fn sweep_round(&mut self, now: u64, budget: usize, c: &mut Cycle) -> Cycle {
        let mut found = [Addr::NONE; PER_ROUND];
        let mut n = 0usize;
        let mut round = Cycle::default();
        let r = self.rng.next_u64();
        self.map.sample_tagged(r, |_key, rec, addr| {
            round.examined += 1;
            // Every key the marked index holds carries a deadline, so this is
            // not a filter any more and the two counts move together. It stays
            // because it is cheap, it is read off a record that is already in
            // cache, and a divergence between them is the marked index having
            // gone wrong, which is the one bug this whole arrangement can have.
            debug_assert!(value::has_expiry(rec), "a marked key with no deadline");
            if value::has_expiry(rec) {
                round.volatile += 1;
                if value::is_expired(rec, now) {
                    found[n] = addr;
                    n += 1;
                }
            }
            round.examined < budget && round.volatile < PER_ROUND && n < PER_ROUND
        });
        c.examined += round.examined;
        c.volatile += round.volatile;
        for addr in &found[..n] {
            // Through the scratch buffer, the same way eviction does it, because
            // the key has to outlive the borrow that found its address and this
            // runs in a loop when it runs at all.
            let mut buf = core::mem::take(&mut self.scratch);
            buf.clear();
            buf.extend_from_slice(self.map.entry_at(*addr).0);
            let gone = self.reaped(&buf);
            self.scratch = buf;
            if gone {
                c.expired += 1;
                round.expired += 1;
            }
        }
        round
    }

    /// Sweep hash fields that are past their deadline, looking at no more than
    /// `budget` hashes, and answer how many it looked at.
    ///
    /// The other cycle, and it is a different shape because the thing it hunts
    /// is not in a record. A field deadline lives inside the hash body, so the
    /// marked index above cannot see one and the sample it draws would never
    /// offer a hash that has fields to lose but a key with no deadline of its
    /// own, which is the usual way the `HEXPIRE` family is used. What this draws
    /// from instead is a list the keyspace keeps of the keys whose hashes have
    /// ever taken a field deadline, and the list is short because most servers
    /// have none.
    ///
    /// A straight walk with a cursor rather than a random sample, for the same
    /// reason: the list holds only candidates, so there is nothing for a sample
    /// to filter out and going round it in order gets to every one of them in
    /// bounded time. A hash whose earliest deadline has not passed costs a load
    /// and a comparison, which is what makes the whole list affordable to walk.
    ///
    /// Why a server needs this at all is the question [`Keyspace::expire_cycle`]
    /// answers for keys, and the answer for fields has a second half. Memory is
    /// the first: a hash that nobody reads again holds every field it was told
    /// to drop. The second is that the events are observable. A client watching
    /// `hexpired` on a key hears about the field within a tick of its deadline
    /// on a real server, whether or not anybody touches the hash, and a server
    /// that only reaped lazily would go quiet until the next command arrived.
    pub fn field_expire_cycle(&mut self, budget: usize) -> usize {
        // The point of the list, the same way the count of keys with deadlines
        // is the point of the one above.
        if budget == 0 || self.field_deadlines.is_empty() {
            return 0;
        }
        let now = self.clock.now_ms();
        // One pass round the list at most, however much budget is left over. A
        // server with three hashes on the list and a big budget would otherwise
        // spend the whole of it going round those three again and again, and the
        // second look at a name in the same tick can only say what the first one
        // said.
        let mut left = budget.min(self.field_deadlines.len());
        let mut looked = 0;
        while left > 0 && !self.field_deadlines.is_empty() {
            left -= 1;
            if self.field_at >= self.field_deadlines.len() {
                self.field_at = 0;
            }
            looked += 1;
            if self.field_look(self.field_at, now) {
                self.field_at += 1;
            } else {
                // The name came off, so whatever was moved into its place is
                // what the cursor is already pointing at.
                self.field_deadlines.swap_remove(self.field_at);
            }
        }
        looked
    }

    /// Look at one name on the list, and say whether it is worth keeping there.
    ///
    /// A name stays for as long as the key under it is a hash. It does not have
    /// to have a deadline on anything right now: a hash whose only deadline was
    /// taken off with `HPERSIST` can be given another one without this list
    /// hearing about it, so dropping the name then would be dropping it for
    /// good. What ends a name is the key going, or something else taking it,
    /// which is when the hash this was about no longer exists to sweep.
    fn field_look(&mut self, at: usize, now: u64) -> bool {
        let key = &self.field_deadlines[at];
        let Some(rec) = self.map.get(key) else {
            return false;
        };
        let meta = value::Meta::from_byte(rec[0]);
        if meta.kind() != value::Kind::Hash {
            return false;
        }
        // A hash whose body is on the device is left alone rather than brought
        // back for this. Reading a hash off the file to find out whether one of
        // its fields is a second late is the whole cost of a fault spent on
        // something no client is waiting for, and the next command that promotes
        // it reaps the field on the way past.
        if meta.is_cold() {
            return true;
        }
        let slot = value::slot(rec);
        // The cheap question first, and it is the one nearly every look answers.
        // A hash with no deadline that has passed is a load of the bound it
        // carries and a comparison against the clock.
        match self.hashes.get(slot).map(Hash::soonest_deadline) {
            Some(Some(soonest)) if soonest <= now => {}
            _ => return true,
        }
        // Through the scratch buffer, the same way the sweep above does it,
        // because the reap needs the key by name and the name is borrowed from
        // the list this is walking.
        let mut buf = core::mem::take(&mut self.scratch);
        buf.clear();
        buf.extend_from_slice(&self.field_deadlines[at]);
        let gone = self.reap_fields(&buf, slot, now, true);
        self.scratch = buf;
        !gone
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::many;
    use crate::ttl::Cond;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    #[test]
    fn a_database_with_no_deadlines_anywhere_is_not_swept() {
        let n = many(2_000u32);
        let mut d = db();
        for i in 0..n {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        let c = d.expire_cycle(4096);
        assert_eq!(c, Cycle::default(), "it should not have drawn anything");
        assert_eq!(d.len(), n as usize);
    }

    #[test]
    fn dead_keys_nobody_reads_are_reclaimed() {
        // Half with a deadline and half without either way, which is the mix
        // the sweep has to pick its way through.
        let n = many(2_000u32);
        let mut d = db();
        for i in 0..n {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        for i in 0..n {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        assert_eq!(d.expires(), n as usize);
        d.clock().advance(200);
        assert_eq!(
            d.len(),
            n as usize * 2,
            "and nothing has read them, so they are all still there"
        );

        // The sweep is bounded, so this is a loop the way a shard loop is a loop.
        let mut spent = 0;
        for _ in 0..500 {
            let c = d.expire_cycle(4096);
            spent += c.examined;
            if d.expires() == 0 {
                break;
            }
        }
        assert_eq!(d.expires(), 0, "spent {spent} looks and did not finish");
        assert_eq!(
            d.len(),
            n as usize,
            "the keys with no deadline are untouched"
        );
        assert_eq!(d.expired_keys(), u64::from(n));
        for i in 0..n {
            assert!(d.exists(format!("k{i}").as_bytes()));
        }
    }

    /// The reason the second index exists, as a number a test can hold.
    ///
    /// Ten thousand keys, a hundred of them with a deadline that has passed. The
    /// sweep has to reclaim all hundred, and the thing to watch is what it spent
    /// getting there: every key it looks at comes off the marked index, so it
    /// looks at about a hundred keys and not about ten thousand. Off the main
    /// index it would have had to walk a hundred keys for every one it wanted.
    ///
    /// It comes out at exactly a hundred, because a round starts at a random
    /// slot of the marked index and then walks forward, so one round covers all
    /// of them. The bound is twice that rather than exactly that, because the
    /// number a test should hold is the shape and not the arithmetic.
    #[test]
    fn a_sweep_only_looks_at_keys_that_could_have_expired() {
        // Only the permanent keys come down under Miri. The hundred with
        // deadlines stay, because the bound below is written against them and
        // a round that overshoots by the rest of a bucket would eat a smaller
        // one.
        let keys = many(10_000u32);
        let mut d = db();
        for i in 0..keys {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        for i in 0..100u32 {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        assert_eq!(d.expires(), 100);
        d.clock().advance(200);

        let mut spent = 0;
        for _ in 0..100 {
            let c = d.expire_cycle(4096);
            spent += c.examined;
            assert_eq!(
                c.examined, c.volatile,
                "it looked at a key with no deadline"
            );
            if d.expires() == 0 {
                break;
            }
        }
        assert_eq!(d.expires(), 0);
        assert_eq!(d.len(), keys as usize, "and it took none of the others");
        assert!(
            spent <= 200,
            "spent {spent} looks to reclaim a hundred keys"
        );
    }

    #[test]
    fn a_key_whose_deadline_has_not_passed_is_left_alone() {
        let mut d = db();
        let now = d.clock().now_ms();
        for i in 0..500u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
            d.set_expiry(format!("k{i}").as_bytes(), Some(now + 900_000));
        }
        for _ in 0..20 {
            let c = d.expire_cycle(4096);
            assert_eq!(c.expired, 0, "it took a key that was still live");
        }
        assert_eq!(d.len(), 500);
    }

    #[test]
    fn the_budget_is_a_ceiling_on_what_a_sweep_looks_at() {
        let n = many(5_000u32);
        let mut d = db();
        for i in 0..n {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        d.clock().advance(200);
        // A budget of one still ends, and it ends having drawn one round rather
        // than having walked the database. One round can overshoot by the rest of
        // a bucket, which is the whole point of charging afterwards instead of
        // asking before every entry.
        let c = d.expire_cycle(1);
        assert!(c.examined <= 8, "one round looked at {} keys", c.examined);
        let left = n as usize - n as usize / 50;
        assert!(d.expires() > left, "and it barely touched the database");
    }

    /// The ratio has to be over the keys that could expire and not over every key
    /// looked at, or a database that is one percent volatile can never clear the
    /// bar and its dead keys are never swept however many there are.
    #[test]
    fn a_mostly_permanent_database_still_gets_its_dead_keys_back() {
        // Both counts come down together, because one percent volatile is the
        // thing being claimed and not the ten thousand it is one percent of.
        let (keys, dying) = (many(10_000u32), many(100u32));
        let mut d = db();
        for i in 0..keys {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        for i in 0..dying {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        d.clock().advance(200);
        let mut spent = 0;
        for _ in 0..2_000 {
            spent += d.expire_cycle(4096).examined;
            if d.expires() == 0 {
                break;
            }
        }
        assert_eq!(d.expires(), 0, "one percent volatile, spent {spent} looks");
        assert_eq!(d.len(), keys as usize);
    }

    #[test]
    fn the_cycle_leaves_collections_and_their_bodies_correct() {
        let mut d = db();
        let now = d.clock().now_ms();
        for i in 0..200u32 {
            let k = format!("s{i}");
            d.sadd(k.as_bytes(), [b"a".as_slice(), b"b".as_slice()].into_iter())
                .expect("room");
            d.set_expiry(k.as_bytes(), Some(now + 100));
        }
        d.sadd(b"keep", [b"a".as_slice()].into_iter())
            .expect("room");
        d.clock().advance(200);
        for _ in 0..500 {
            d.expire_cycle(4096);
            if d.expires() == 0 {
                break;
            }
        }
        assert_eq!(d.len(), 1);
        assert_eq!(d.scard(b"keep"), Ok(1));
        // The bodies went back with the records rather than being left behind in
        // their slabs, which a length check on the keyspace alone would not see.
        assert_eq!(d.bodies, 1);
    }

    /// Give a hash a field deadline and let it pass with nobody reading the
    /// hash. The field has to go anyway, because that is what the field cycle is
    /// for, and the counters have to say it was the cycle that took it.
    #[test]
    fn a_field_nobody_reads_goes_on_its_own() {
        let mut d = db();
        let now = d.clock().now_ms();
        d.hset(b"h", [(b"a".as_slice(), b"1".as_slice())].into_iter())
            .expect("room");
        d.hset(b"h", [(b"b".as_slice(), b"2".as_slice())].into_iter())
            .expect("room");
        d.hexpire(
            b"h",
            now + 100,
            Cond::Always,
            [b"a".as_slice()].into_iter(),
            |_| {},
        )
        .expect("room");
        d.clock().advance(200);
        assert_eq!(d.field_expire_cycle(16), 1, "one hash to look at");
        assert_eq!(d.hlen(b"h"), Ok(1), "and the other field is still there");
        assert_eq!(d.expired_fields(), 1);
        assert_eq!(d.expired_fields_active(), 1);
        assert_eq!(d.expired_keys(), 0, "the key itself had no deadline");
    }

    /// And when it was the only field, the key goes with it, since an empty hash
    /// is not a key.
    #[test]
    fn the_last_field_takes_the_key_with_it() {
        let mut d = db();
        let now = d.clock().now_ms();
        d.hset(b"h", [(b"a".as_slice(), b"1".as_slice())].into_iter())
            .expect("room");
        d.hexpire(
            b"h",
            now + 100,
            Cond::Always,
            [b"a".as_slice()].into_iter(),
            |_| {},
        )
        .expect("room");
        d.clock().advance(200);
        d.field_expire_cycle(16);
        assert!(!d.exists(b"h"));
        assert_eq!(d.len(), 0);
        assert_eq!(d.bodies, 0, "and the body went back to its slab");
        // The name came off the list with the key, so a second sweep has nothing
        // to look at rather than a dangling name to look up.
        assert_eq!(d.field_expire_cycle(16), 0);
    }

    /// The list only holds hashes that took a deadline at some point, so a
    /// database full of ordinary hashes costs the cycle nothing.
    #[test]
    fn hashes_with_no_field_deadlines_are_not_swept() {
        let mut d = db();
        for i in 0..100u32 {
            d.hset(
                format!("h{i}").as_bytes(),
                [(b"a".as_slice(), b"1".as_slice())].into_iter(),
            )
            .expect("room");
        }
        assert_eq!(d.field_expire_cycle(4096), 0);
    }

    /// A name whose key is gone, or is no longer a hash, comes off the list the
    /// first time the cycle reaches it. Nothing else prunes it, because nothing
    /// else knows the list is there.
    #[test]
    fn a_name_that_is_no_longer_a_hash_comes_off_the_list() {
        let mut d = db();
        let now = d.clock().now_ms();
        for i in 0..3u32 {
            let k = format!("h{i}");
            d.hset(
                k.as_bytes(),
                [(b"a".as_slice(), b"1".as_slice())].into_iter(),
            )
            .expect("room");
            d.hexpire(
                k.as_bytes(),
                now + 100_000,
                Cond::Always,
                [b"a".as_slice()].into_iter(),
                |_| {},
            )
            .expect("room");
        }
        d.del(b"h0");
        d.set_plain(b"h1", b"v").expect("room");
        // Three looks is one round of the list, and two of the three names have
        // nothing behind them any more.
        d.field_expire_cycle(3);
        assert_eq!(d.field_deadlines.len(), 1);
        assert_eq!(d.field_deadlines[0].as_ref(), b"h2");
    }

    /// Setting the same deadline over and over on the same hash puts its name on
    /// the list once, which is the thing that would otherwise grow without
    /// bound on a key a client keeps refreshing.
    #[test]
    fn a_hash_is_only_listed_once_however_often_it_is_touched() {
        let mut d = db();
        let now = d.clock().now_ms();
        d.hset(b"h", [(b"a".as_slice(), b"1".as_slice())].into_iter())
            .expect("room");
        for i in 0..50u64 {
            d.hexpire(
                b"h",
                now + 100_000 + i,
                Cond::Always,
                [b"a".as_slice()].into_iter(),
                |_| {},
            )
            .expect("room");
        }
        assert_eq!(d.field_deadlines.len(), 1);
    }
}
