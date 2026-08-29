//! Walking the keyspace: `SCAN`, `KEYS` and `RANDOMKEY`.
//!
//! Three commands that all want the same thing, which is to look at keys the
//! caller has not named, and that want it in three different shapes. `KEYS`
//! wants every key now and does not care what it costs. `SCAN` wants a bounded
//! bite and a number it can come back with. `RANDOMKEY` wants one key and does
//! not want to look at the others to find it.
//!
//! # Nothing here writes
//!
//! A key past its deadline is skipped and not reaped, which is the one place in
//! this crate where a dead key is left where it lies. Every other read takes
//! `&mut self` and drops it on the way through, and these three take `&self` so
//! that a walk can hand the caller a borrow of a key that is still in the map.
//! Deleting during the walk would mean copying every key out first, which is
//! the allocation the whole design is avoiding, and it would turn `KEYS` on a
//! read replica into a write.
//!
//! Nothing leaks from that. The active expiry cycle collects dead keys whether
//! or not anything read them, and the next ordinary read of one of these keys
//! reaps it the usual way.
//!
//! # `COUNT` is a floor
//!
//! [`Keyspace::scan`] stops at the first bucket boundary past the budget, so a
//! `COUNT 10` can come back with fifteen keys, and a `MATCH` that rejects all
//! of them can come back with none and a cursor that is not zero. That is
//! Redis's behaviour exactly, and a client that treats an empty batch as the
//! end of the scan is broken against Redis too.
//!
//! The filtering happens in the caller's closure and the budget is counted
//! before it, so a `MATCH` that matches nothing still walks the whole keyspace
//! a bucket at a time rather than in one unbounded call.

use yo_index::Cursor as KeyCursor;

use crate::keyspace::Keyspace;
use crate::value::{self, Kind};

/// How many random buckets to try before giving up and walking the whole thing.
///
/// A bucket holds fourteen entries and the index keeps the table loaded, so the
/// first try finds a key in almost every database anyone has. The tries only
/// come into play when the keyspace is tiny or has just had most of it deleted,
/// and the walk behind them is what stops those cases from being wrong rather
/// than slow.
const TRIES: usize = 64;

impl Keyspace {
    /// A batch of keys, and where the next batch starts.
    ///
    /// This is `SCAN`. `budget` is `COUNT`, `ty` is `TYPE`, and `MATCH` belongs
    /// to the caller because a glob is a wire concern and this is not the wire.
    ///
    /// The cursor is opaque to the client and is not opaque here: it names a
    /// place in the keyspace rather than a place in memory, which is what lets
    /// it survive the index doubling between two calls. The reasoning is in
    /// [`yo_index::Cursor`].
    ///
    /// A key that is there for the whole scan comes back at least once. A key
    /// added or removed partway through may or may not, and any key may come
    /// back twice. That is Redis's contract and a client written against Redis
    /// already copes with all three.
    pub fn scan(
        &self,
        from: KeyCursor,
        budget: usize,
        ty: Option<Kind>,
        mut out: impl FnMut(&[u8]),
    ) -> KeyCursor {
        let now = self.clock.now_ms();
        self.map.scan(from, budget, |key, rec| {
            if value::is_expired(rec, now) {
                return;
            }
            if ty.is_some_and(|want| value::kind(rec) != want) {
                return;
            }
            out(key);
        })
    }

    /// Every key in the database, once each.
    ///
    /// This is `KEYS`, and it is the command whose reputation is deserved: it
    /// visits every bucket in the index before it answers anything, and a
    /// database of ten million keys is ten million calls to `out` with the
    /// shard doing nothing else. It is here because tooling needs it and
    /// because `SCAN` is the answer for everything else.
    ///
    /// One walk with an unbounded budget rather than a loop over [`Keyspace::scan`],
    /// which is the same walk without the chance of a duplicate, because nothing
    /// can split the index while this is running.
    pub fn keys(&self, out: impl FnMut(&[u8])) {
        self.scan(KeyCursor::START, usize::MAX, None, out);
    }

    /// One key, chosen at random, or `None` if the database is empty.
    ///
    /// This is `RANDOMKEY`. It picks a random position in the index and takes a
    /// key from the bucket that lands in, which is a constant number of loads
    /// and does not depend on how many keys there are.
    ///
    /// Uniform within the bucket and only roughly uniform across the keyspace,
    /// since a bucket holding two keys and a bucket holding twelve are equally
    /// likely to be landed on. Redis's is biased the same way and for the same
    /// reason. What it is not is skewed towards any particular key, which is
    /// what matters for the thing `RANDOMKEY` is actually used for, which is
    /// sampling a live database to see what is in it.
    pub fn random_key(&mut self) -> Option<Vec<u8>> {
        if self.map.is_empty() {
            return None;
        }
        for _ in 0..TRIES {
            let from = KeyCursor::from_raw(self.rng.next_u64());
            if let Some(key) = self.sample(from, 0) {
                return Some(key);
            }
        }
        // Every bucket that was tried was empty or held nothing but dead keys.
        // A full walk is the only answer left that can tell an unlucky run of
        // tries from a database whose keys have all expired.
        self.sample(KeyCursor::START, usize::MAX)
    }

    /// One key from the walk starting at `from`, uniform among the keys it sees.
    ///
    /// Reservoir sampling, which is the version that needs one pass and one
    /// slot of memory. Taking the first key instead would answer the same key
    /// every time for as long as the bucket held still.
    fn sample(&mut self, from: KeyCursor, budget: usize) -> Option<Vec<u8>> {
        let now = self.clock.now_ms();
        // Named separately so the closure borrows the counter and not the whole
        // keyspace, which the walk is holding.
        let rng = &mut self.rng;
        let mut seen = 0usize;
        let mut pick = None;
        self.map.scan(from, budget, |key, rec| {
            if value::is_expired(rec, now) {
                return;
            }
            seen += 1;
            if rng.below(seen) == 0 {
                pick = Some(key.to_vec());
            }
        });
        pick
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::Clock;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000_000))
    }

    fn put(d: &mut Keyspace, key: &[u8]) {
        d.set_plain(key, b"v").expect("room for a record");
    }

    fn keys_of(db: &Keyspace) -> HashSet<Vec<u8>> {
        let mut out = HashSet::new();
        db.keys(|k| {
            out.insert(k.to_vec());
        });
        out
    }

    #[test]
    fn an_empty_database_has_nothing_to_walk() {
        let mut db = db();
        assert!(keys_of(&db).is_empty());
        assert_eq!(db.random_key(), None);
        assert!(db.scan(KeyCursor::START, 10, None, |_| {}).is_end());
    }

    #[test]
    fn a_scan_comes_back_with_every_key_once() {
        let mut db = db();
        for i in 0..2_000u32 {
            put(&mut db, format!("k{i}").as_bytes());
        }

        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut at = KeyCursor::START;
        loop {
            at = db.scan(at, 10, None, |k| seen.push(k.to_vec()));
            if at.is_end() {
                break;
            }
        }

        let unique: HashSet<Vec<u8>> = seen.iter().cloned().collect();
        assert_eq!(unique.len(), 2_000);
        assert_eq!(seen.len(), 2_000, "a quiet scan returned a key twice");
        assert_eq!(unique, keys_of(&db));
    }

    #[test]
    fn a_scan_can_ask_for_one_type() {
        let mut db = db();
        put(&mut db, b"s");
        db.sadd(b"members", [b"a".as_slice()].into_iter())
            .expect("a fresh key");
        db.hset(b"h", [(b"f".as_slice(), b"v".as_slice())].into_iter())
            .expect("a fresh key");

        for (want, name) in [
            (Kind::String, "s"),
            (Kind::Set, "members"),
            (Kind::Hash, "h"),
        ] {
            let mut seen = Vec::new();
            let mut at = KeyCursor::START;
            loop {
                at = db.scan(at, 100, Some(want), |k| seen.push(k.to_vec()));
                if at.is_end() {
                    break;
                }
            }
            assert_eq!(seen, vec![name.as_bytes().to_vec()], "type {want:?}");
        }
    }

    #[test]
    fn a_key_past_its_deadline_is_walked_past_and_not_deleted() {
        let mut db = db();
        put(&mut db, b"alive");
        put(&mut db, b"dead");
        assert!(db.set_expiry(b"dead", Some(1_000_500)));

        db.clock_mut().advance(1_000);
        assert_eq!(keys_of(&db), HashSet::from([b"alive".to_vec()]));
        // Still in the map, because a walk takes a shared borrow and cannot
        // delete anything. The next ordinary read is what collects it.
        assert_eq!(db.len(), 2);
        assert_eq!(db.kind_of(b"dead"), None);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn a_random_key_is_a_key_that_is_there() {
        let mut db = db();
        for i in 0..500u32 {
            put(&mut db, format!("k{i}").as_bytes());
        }

        let all = keys_of(&db);
        let mut picked = HashSet::new();
        for _ in 0..200 {
            let k = db.random_key().expect("the database is not empty");
            assert!(
                all.contains(&k),
                "randomkey answered a key that is not there"
            );
            picked.insert(k);
        }
        // Not a distribution test, just a check that it is not answering the
        // same key every time, which is what a walk that always takes the first
        // hit would do.
        assert!(
            picked.len() > 10,
            "only {} distinct keys in 200 draws",
            picked.len()
        );
    }

    #[test]
    fn the_last_key_left_is_the_one_randomkey_finds() {
        let mut db = db();
        for i in 0..5_000u32 {
            put(&mut db, format!("k{i}").as_bytes());
        }
        for i in 0..5_000u32 {
            if i != 4_242 {
                db.del(format!("k{i}").as_bytes());
            }
        }

        // One key in a directory that grew to hold five thousand, so every
        // random try misses and the fallback walk is what answers.
        assert_eq!(db.random_key().as_deref(), Some(&b"k4242"[..]));
    }

    #[test]
    fn a_scan_survives_the_keyspace_growing_underneath_it() {
        let mut db = db();
        for i in 0..2_000u32 {
            put(&mut db, format!("k{i}").as_bytes());
        }

        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut at = KeyCursor::START;
        let mut added = 2_000u32;
        loop {
            at = db.scan(at, 8, None, |k| {
                seen.insert(k.to_vec());
            });
            if at.is_end() {
                break;
            }
            for _ in 0..64 {
                put(&mut db, format!("k{added}").as_bytes());
                added += 1;
            }
        }

        for i in 0..2_000u32 {
            let k = format!("k{i}").into_bytes();
            assert!(
                seen.contains(&k),
                "k{i} was there throughout and never came back"
            );
        }
    }
}
