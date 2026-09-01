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
//! # What is not here yet
//!
//! Redis keeps a pool of sixteen candidates between rounds, so a good victim
//! spotted in one round is still in the running in the next one. That is a real
//! improvement in the quality of the approximation and it is not free here: a
//! pool that outlives a command cannot hold addresses, because the next write
//! moves them, so it has to hold keys and that means somewhere to keep the
//! bytes. It is worth doing and it is a separate decision from this one.

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
const ANY: u64 = u64::MAX;

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
    if matches!(policy, Policy::AllKeysRandom | Policy::VolatileRandom) {
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
/// the winner before it writes anything. That is also the reason this cannot
/// become Redis's pool without changing what it stores.
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
