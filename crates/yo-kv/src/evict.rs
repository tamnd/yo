//! Choosing which key to throw away.
//!
//! Eviction is the one place in the server where being approximately right is
//! the correct engineering answer. A server that has run out of room has to give
//! some memory back before it can answer the write in front of it, and the
//! client is waiting. Finding the genuinely least recently used key out of forty
//! million of them means an ordering over all of them, which is a structure to
//! maintain on every read of every key forever, to make a decision that is only
//! ever a guess about the future anyway. Redis decided in 3.0 that it would
//! rather sample a few keys and take the worst of them, and it was right.
//!
//! So this samples. [`Policy`] says which keys are eligible and how to score
//! them, [`score`] turns one record into a number where larger means a better
//! victim, and the caller takes the largest number it saw.
//!
//! # What the number means
//!
//! Not much on its own, and that is deliberate. The four scoring rules produce
//! numbers on four different scales: seconds of idleness under the clock
//! policies, a countdown from 255 under LFU, a subtraction from the top of the
//! range under `volatile-ttl`, and a constant under the random ones. They are
//! never compared across policies, because a policy does not change halfway
//! through a round of sampling, so the only thing the scale has to support is
//! the comparison of two candidates under the same rule.
//!
//! The one property they all share is the direction. Bigger is more disposable.
//! Getting that backwards would build a cache that keeps exactly the keys nobody
//! wants, and it would still pass a test that only checked the pick was eligible,
//! which is why the tests here check which of two keys comes out and not just
//! that one did.
//!
//! # Why the good ones are kept
//!
//! A round of five samples throws four keys away, and one of them is often
//! better than anything the next round turns up. [`Pool`] keeps sixteen of them
//! between rounds, which is Redis 3.0's idea and is most of what separates its
//! eviction from a plain random sample. Every round adds to the same pool, so
//! the candidate that eventually goes is the worst key seen across all of them
//! rather than the worst key seen in the last five.
//!
//! A pool that outlives a command cannot hold addresses, because the next write
//! moves them, so it holds key bytes. Each of the sixteen slots owns its buffer
//! and reuses it, so a pool that has been through one round of eviction does not
//! allocate again unless a longer key than it has ever seen turns up.
//!
//! Holding keys rather than addresses also means a candidate can go stale: the
//! key can be deleted, expire, or lose its deadline under a `volatile` policy
//! between the round that spotted it and the round that takes it. So a candidate
//! is looked up and rechecked on the way out, and one that no longer qualifies
//! is dropped and the next one tried. That costs at most sixteen failed lookups
//! in the worst case, which is bounded and rare, against a decision that is
//! measurably closer to the right one every time it is not.

use yo_common::Addr;

use crate::access::{Lfu, Policy};
use crate::value;

/// How many keys a round of sampling looks at, which is `maxmemory-samples`.
///
/// Five, which is Redis's default and is a better number than it sounds. The
/// published curve for it flattens hard: five samples already picks a key from
/// close to the true tail, ten is visibly better, and everything past that is
/// paying for a decision that a guess about the future does not deserve.
pub const SAMPLES: usize = 5;

/// The largest score, used by the policies that do not really have one.
///
/// Under `allkeys-random` and `volatile-random` every eligible key is as good a
/// victim as every other, so they all score the same and the first one sampled
/// wins. It is the top of the range rather than the bottom so that a caller
/// comparing against a starting score of zero does not have to special case it.
pub const ANY: u64 = u64::MAX;

/// Whether a policy would ever consider this record.
///
/// The only rule is the deadline: a `volatile` policy will not touch a key that
/// has no expiry, whatever else is true of it. That is the rule behind the
/// classic surprise, which is that `volatile-lru` on a database where nothing
/// has a TTL evicts nothing at all and starts refusing writes, and it is worth
/// having in one place rather than inline at the sampling loop.
#[must_use]
pub fn eligible(rec: &[u8], policy: Policy) -> bool {
    !policy.volatile_only() || value::expire_at(rec).is_some()
}

/// How disposable this record is under this policy. Larger goes first.
///
/// The four rules, in the order the match takes them:
///
/// Under `volatile-ttl` the key that expires soonest goes first, so the score
/// counts down from the top of the range as the deadline moves out. A record
/// with no deadline cannot reach here, because [`eligible`] refused it, and if
/// one somehow did it would score zero and lose to everything.
///
/// Under the random pair every eligible key scores the same, which makes the
/// pick the first one sampled. That is a fair draw and not a biased one, because
/// the sample itself is what did the choosing.
///
/// Under an LFU policy the counter is read with the decay applied, and the score
/// is what is left of the range above it. The counter saturates at 255, so a key
/// that has been hammered scores zero and is the last thing to go.
///
/// Under everything else the field is a clock and the score is seconds of
/// idleness. That covers the LRU pair, the LRM pair, and `noeviction`, which
/// scores keys it will never evict because `OBJECT IDLETIME` asks the same
/// question and a server that will never evict still has to answer it.
#[must_use]
pub fn score(rec: &[u8], policy: Policy, now_ms: u64, lfu: Lfu) -> u64 {
    if matches!(policy, Policy::VolatileTtl) {
        return value::expire_at(rec).map_or(0, |at| u64::MAX - at);
    }
    if policy.is_random() {
        return ANY;
    }
    let access = value::access(rec).unwrap_or_default();
    if policy.is_lfu() {
        return u64::from(u8::MAX - access.freq(now_ms, lfu));
    }
    access.idle_secs(now_ms)
}

/// The best victim seen so far in one round of sampling.
///
/// It holds an address rather than a key, which is what confines it to a single
/// round: an address is only good until the next write, and the caller deletes
/// the winner before it writes anything. [`Pool`] is the version that survives a
/// round, and it pays for that by holding bytes.
///
/// This is what the random policies use, because they have no ordering for a
/// pool to approximate and every eligible key is already the answer.
#[derive(Debug, Clone, Copy)]
pub struct Best {
    /// Where the winner is, or [`Addr::NONE`] if nothing eligible turned up.
    pub addr: Addr,
    /// Its score, meaningful only against another score under the same policy.
    pub score: u64,
}

impl Best {
    /// Nothing yet.
    pub const EMPTY: Best = Best {
        addr: Addr::NONE,
        score: 0,
    };

    /// Whether anything eligible has been seen.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.addr.is_none()
    }

    /// Take this candidate if it beats what is held.
    ///
    /// Strictly better and not as good, so a tie leaves the earlier one in
    /// place. That is what makes the random policies pick the first key sampled
    /// rather than the last, and under the other policies it means the pick does
    /// not wander between keys that are equally stale.
    pub fn offer(&mut self, addr: Addr, score: u64) {
        if self.is_empty() || score > self.score {
            *self = Best { addr, score };
        }
    }
}

/// How many candidates survive between rounds of sampling.
///
/// Sixteen, which is Redis's `EVPOOL_SIZE`. It is three rounds of sampling at
/// the default of five, so a pool holds roughly the last three rounds worth of
/// the keys worth remembering and forgets the rest.
pub const CANDIDATES: usize = 16;

/// One candidate, and the buffer its key is kept in between rounds.
#[derive(Debug, Default, Clone)]
struct Slot {
    /// Its score when it was last offered, on the scale [`score`] was using.
    score: u64,
    /// The key, copied because an address would not survive the next write.
    key: Vec<u8>,
}

impl Slot {
    /// Become this candidate, keeping whatever buffer was already here.
    fn fill(&mut self, key: &[u8], score: u64) {
        self.score = score;
        self.key.clear();
        self.key.extend_from_slice(key);
    }
}

/// The best candidates seen across rounds, worst victim last.
///
/// Sorted by score ascending, so [`Pool::take`] pops the end and the weakest
/// candidate is always at the front where a better one displaces it. Sixteen
/// entries is small enough that a sorted array beats anything with a shape, and
/// the shifting is a `rotate` over at most fifteen `Vec` headers.
///
/// The array is allocated on the first offer rather than on construction,
/// because a database that never evicts anything is the common one and it does
/// not deserve sixteen anythings.
#[derive(Debug, Default, Clone)]
pub struct Pool {
    at: Vec<Slot>,
    len: usize,
}

impl Pool {
    /// An empty pool that has not allocated anything.
    #[must_use]
    pub const fn new() -> Pool {
        Pool {
            at: Vec::new(),
            len: 0,
        }
    }

    /// How many candidates are held.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether there is nothing to take.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forget every candidate and keep the buffers.
    ///
    /// The caller runs this when the answers stop meaning anything, which is a
    /// policy change and a flush. Both leave a pool full of scores on a scale
    /// nothing uses any more or keys that are not there, and while the recheck
    /// on the way out would survive either, a stale pool is sixteen wasted
    /// lookups in front of the next eviction.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// What the buffers cost.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.at.capacity() * size_of::<Slot>()
            + self.at.iter().map(|s| s.key.capacity()).sum::<usize>()
    }

    /// Put a candidate in the running.
    ///
    /// A key already held is re-scored rather than held twice, because the same
    /// key turning up in two rounds is ordinary and two entries for it would be
    /// one wasted slot and one guaranteed miss on the way out.
    ///
    /// A key worse than everything held is dropped when the pool is full, which
    /// is the common case once it has warmed up and is the reason this is cheap.
    pub fn offer(&mut self, key: &[u8], score: u64) {
        if self.at.is_empty() {
            self.at.resize_with(CANDIDATES, Slot::default);
        }
        if let Some(i) = self.at[..self.len].iter().position(|s| s.key == key) {
            if self.at[i].score == score {
                return;
            }
            // Out of the sorted run and into the free space past it, which keeps
            // its buffer where the insert below can pick it up again.
            self.at[i..self.len].rotate_left(1);
            self.len -= 1;
        } else if self.len == CANDIDATES && score <= self.at[0].score {
            return;
        }
        let i = self.at[..self.len].partition_point(|s| s.score <= score);
        let at = if self.len < CANDIDATES {
            // The free slot at `len` comes back to `i` and everything from `i`
            // moves up one.
            self.at[i..=self.len].rotate_right(1);
            self.len += 1;
            i
        } else {
            // The pool is full and this beats the front of it, so the front goes
            // and everything below `i` moves down one. `i` is at least one here
            // because the check above sent back anything the front could beat.
            self.at[..i].rotate_left(1);
            i - 1
        };
        self.at[at].fill(key, score);
    }

    /// The worst key held, removed from the pool.
    ///
    /// It is removed whether or not the caller can use it, because a candidate
    /// the caller looked at and rejected is a candidate that will be rejected
    /// again next time, and the point of a pool is to stop paying for the same
    /// answer twice.
    pub fn take(&mut self) -> Option<&[u8]> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(&self.at[self.len].key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the pool holds, worst victim first, so a test can read it.
    fn held(p: &Pool) -> Vec<(&[u8], u64)> {
        p.at[..p.len]
            .iter()
            .rev()
            .map(|s| (&s.key[..], s.score))
            .collect()
    }

    #[test]
    fn the_worst_candidate_comes_out_first() {
        let mut p = Pool::new();
        p.offer(b"middling", 50);
        p.offer(b"terrible", 90);
        p.offer(b"fine", 10);

        assert_eq!(p.len(), 3);
        assert_eq!(p.take(), Some(&b"terrible"[..]));
        assert_eq!(p.take(), Some(&b"middling"[..]));
        assert_eq!(p.take(), Some(&b"fine"[..]));
        assert_eq!(p.take(), None);
        assert!(p.is_empty());
    }

    #[test]
    fn a_full_pool_keeps_the_worst_sixteen_and_nothing_else() {
        let mut p = Pool::new();
        // Thirty two keys offered worst first, so every one after the first
        // sixteen is better than everything held and should be turned away.
        for i in 0..32u64 {
            p.offer(format!("key-{i}").as_bytes(), 1000 - i);
        }
        assert_eq!(p.len(), CANDIDATES);
        let names: Vec<_> = held(&p)
            .into_iter()
            .map(|(k, _)| String::from_utf8(k.to_vec()).expect("ascii"))
            .collect();
        assert_eq!(names[0], "key-0", "the worst key offered");
        assert_eq!(names[15], "key-15");

        // And the other way round, where every one displaces the front.
        let mut q = Pool::new();
        for i in 0..32u64 {
            q.offer(format!("key-{i}").as_bytes(), i);
        }
        assert_eq!(q.len(), CANDIDATES);
        assert_eq!(q.take(), Some(&b"key-31"[..]), "the worst key offered");
    }

    #[test]
    fn a_key_offered_twice_is_held_once_at_its_new_score() {
        let mut p = Pool::new();
        p.offer(b"a", 10);
        p.offer(b"b", 20);
        p.offer(b"c", 30);
        // The same key again, now the worst thing in the pool rather than the
        // best. One entry, in its new place.
        p.offer(b"a", 40);

        assert_eq!(p.len(), 3);
        assert_eq!(
            held(&p),
            vec![(&b"a"[..], 40), (&b"c"[..], 30), (&b"b"[..], 20)]
        );
    }

    #[test]
    fn a_key_offered_twice_at_the_same_score_changes_nothing() {
        let mut p = Pool::new();
        p.offer(b"a", 10);
        p.offer(b"b", 20);
        p.offer(b"a", 10);

        assert_eq!(p.len(), 2);
        assert_eq!(held(&p), vec![(&b"b"[..], 20), (&b"a"[..], 10)]);
    }

    #[test]
    fn a_key_offered_twice_into_a_full_pool_still_leaves_room() {
        let mut p = Pool::new();
        for i in 0..CANDIDATES as u64 {
            p.offer(format!("key-{i}").as_bytes(), 100 + i);
        }
        // Worse than everything held, and already held, so the pool has to drop
        // it out of the middle before it puts it back on the end rather than
        // evicting its own front to make room for a key it already has.
        p.offer(b"key-3", 999);

        assert_eq!(p.len(), CANDIDATES);
        assert_eq!(p.take(), Some(&b"key-3"[..]));
        assert_eq!(
            p.take(),
            Some(&b"key-15"[..]),
            "the front was not thrown away"
        );
    }

    #[test]
    fn clearing_forgets_the_candidates_and_keeps_the_buffers() {
        let mut p = Pool::new();
        for i in 0..CANDIDATES as u64 {
            p.offer(format!("a rather long key name number {i}").as_bytes(), i);
        }
        let held = p.memory_bytes();
        p.clear();

        assert!(p.is_empty());
        assert_eq!(p.take(), None);
        assert_eq!(p.memory_bytes(), held, "the buffers went with the scores");
    }

    #[test]
    fn a_warm_pool_does_not_allocate_again() {
        let mut p = Pool::new();
        for i in 0..64u64 {
            p.offer(format!("key-{i:0>6}").as_bytes(), i % 17);
        }
        let settled = p.memory_bytes();
        for i in 0..1000u64 {
            p.offer(format!("key-{i:0>6}").as_bytes(), i % 17);
        }
        assert_eq!(
            p.memory_bytes(),
            settled,
            "a key no longer than any it has seen cost it an allocation"
        );
    }

    #[test]
    fn an_untouched_pool_costs_nothing() {
        let p = Pool::new();
        assert_eq!(p.memory_bytes(), 0);
        assert!(p.is_empty());
    }
}
