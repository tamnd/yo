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
//! There is no second dictionary here yet, so the sample comes off the main index
//! and most of what it looks at may have no deadline at all. That changes the
//! ratio the rule is computed over, not the rule. The quarter is a quarter of the
//! keys sampled that could have expired, because a quarter of every key sampled
//! is a bar that a database which is one percent volatile can never clear however
//! much dead memory is sitting in it.
//!
//! It changes the cost, though, and that is what the budget is for. The budget is
//! in keys looked at rather than keys expired, so a sweep of a database where
//! nothing is volatile costs exactly the budget and not a byte more, whatever the
//! density is. And the density can be nothing at all, which is the common case:
//! a count of the keys carrying a deadline sits in the keyspace, and a zero there
//! ends this before it draws anything.

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
        if budget == 0 || self.expires == 0 {
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
        self.map.sample(r, |_key, rec, addr| {
            round.examined += 1;
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
            let gone = self.drop_key(&buf);
            self.scratch = buf;
            if gone {
                c.expired += 1;
                round.expired += 1;
                self.expired += 1;
            }
        }
        round
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    #[test]
    fn a_database_with_no_deadlines_anywhere_is_not_swept() {
        let mut d = db();
        for i in 0..2_000u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        let c = d.expire_cycle(4096);
        assert_eq!(c, Cycle::default(), "it should not have drawn anything");
        assert_eq!(d.len(), 2_000);
    }

    #[test]
    fn dead_keys_nobody_reads_are_reclaimed() {
        let mut d = db();
        for i in 0..2_000u32 {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        for i in 0..2_000u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        assert_eq!(d.expires(), 2_000);
        d.clock_mut().advance(200);
        assert_eq!(
            d.len(),
            4_000,
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
        assert_eq!(d.len(), 2_000, "the keys with no deadline are untouched");
        assert_eq!(d.expired_keys(), 2_000);
        for i in 0..2_000u32 {
            assert!(d.exists(format!("k{i}").as_bytes()));
        }
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
        let mut d = db();
        for i in 0..5_000u32 {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        d.clock_mut().advance(200);
        // A budget of one still ends, and it ends having drawn one round rather
        // than having walked the database. One round can overshoot by the rest of
        // a bucket, which is the whole point of charging afterwards instead of
        // asking before every entry.
        let c = d.expire_cycle(1);
        assert!(c.examined <= 8, "one round looked at {} keys", c.examined);
        assert!(d.expires() > 4_900, "and it barely touched the database");
    }

    /// The ratio has to be over the keys that could expire and not over every key
    /// looked at, or a database that is one percent volatile can never clear the
    /// bar and its dead keys are never swept however many there are.
    #[test]
    fn a_mostly_permanent_database_still_gets_its_dead_keys_back() {
        let mut d = db();
        for i in 0..10_000u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        for i in 0..100u32 {
            d.psetex(format!("d{i}").as_bytes(), 100, b"v")
                .expect("room");
        }
        d.clock_mut().advance(200);
        let mut spent = 0;
        for _ in 0..2_000 {
            spent += d.expire_cycle(4096).examined;
            if d.expires() == 0 {
                break;
            }
        }
        assert_eq!(d.expires(), 0, "one percent volatile, spent {spent} looks");
        assert_eq!(d.len(), 10_000);
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
        d.clock_mut().advance(200);
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
}
