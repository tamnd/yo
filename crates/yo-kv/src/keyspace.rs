//! One database, and the parts of it that are not about any particular type.
//!
//! This is the `dict` a Redis `SELECT` picks between, and one of these is what a
//! shard owns. It was called `Strings` while strings were the only thing in it,
//! which was accurate for M2 and stopped being accurate the moment a set needed
//! somewhere to live.
//!
//! The commands hang off this as separate `impl` blocks, one file per type, so
//! that `SET` lives in [`strings`](crate::strings) next to the other twenty five
//! string commands rather than in a file that is the whole of Redis. They are
//! methods on the keyspace and not on some per type object because a key belongs
//! to the database and not to a type: `DEL` does not care what it is deleting,
//! and `SADD` against a string has to be able to see that it is a string.
//!
//! # Not Sync
//!
//! Like everything that hangs off a shard. One of these belongs to one thread and
//! is reached by sending that thread a command, which is Y1, and it is why
//! nothing here takes a lock or an atomic.

use std::sync::atomic::{AtomicU64, Ordering};

use yo_common::{Addr, Code, Error, Result, Rng, bytes_eq};
use yo_index::RawMap;

use crate::Clock;
use crate::access::{Access, Lfu, Policy};
use crate::array::Array;
use crate::cold::Blocks;
use crate::evict;
use crate::foreign::Foreign;
use crate::hash::{self, Hash};
use crate::list::{self, List};
use crate::set::{self, Set};
use crate::slab::{Bytes, Slab};
use crate::stream::{self, Stream};
use crate::tier::{self, Faulted, Relief, Tier};
use crate::ttl::{self, Applied, Ask, Cond};
use crate::value::{self, Kind, Str};
use crate::zset::{self, Zset};

/// Every collection already answered this question, and this is the answer said
/// once more in a shape the slab can ask for without knowing what it is holding.
///
/// Here rather than in the five type files because it is one fact about the
/// keyspace and not five facts about five types, and because a reader looking
/// for how the memory total is kept should find it next to the slabs it counts.
macro_rules! bytes {
    ($($t:ty),*) => { $(impl Bytes for $t {
        #[inline]
        fn memory_bytes(&self) -> usize {
            <$t>::memory_bytes(self)
        }
    })* };
}
bytes!(Set, Hash, List, Zset, Array, Stream);

/// A foreign body counts what it says it counts, plus the box around it.
///
/// Not through the macro, because the macro calls an inherent method of the
/// same name and this one is a trait method reached through a vtable. The
/// pointer itself is two words on top of whatever the engine reports, which is
/// the price of the escape and is worth naming rather than losing.
impl Bytes for Box<dyn Foreign> {
    #[inline]
    fn memory_bytes(&self) -> usize {
        self.as_ref().memory_bytes() + std::mem::size_of::<Box<dyn Foreign>>()
    }
}

/// One database: every key, whatever type it holds.
pub struct Keyspace {
    pub(crate) map: RawMap,
    pub(crate) clock: Clock,
    /// Keys that were found dead on the way to answering something else.
    pub(crate) expired: u64,
    /// Keys thrown away to make room, which is a different number entirely.
    ///
    /// Redis keeps `expired_keys` and `evicted_keys` apart in `INFO` and the
    /// distinction is the one people watch: expiry is the client getting what it
    /// asked for, and eviction is the server deciding it cannot keep a promise
    /// nobody asked it to break.
    pub(crate) evicted: u64,
    /// Every set in this database, addressed by the number in its record.
    pub(crate) sets: Slab<Set>,
    /// Every hash in this database, addressed the same way.
    ///
    /// A slab per type rather than one slab of an enum, so that a record's four
    /// bytes index a `Hash` directly and reaching one is a load and not a load
    /// followed by a discriminant check. The type tag in the record already
    /// says which slab to look in, so the discriminant would be a second copy
    /// of a fact the record has.
    pub(crate) hashes: Slab<Hash>,
    /// Every list in this database, addressed the same way.
    pub(crate) lists: Slab<List>,
    /// Every sorted set in this database, addressed the same way.
    pub(crate) zsets: Slab<Zset>,
    /// Every sparse array in this database, addressed the same way.
    pub(crate) arrays: Slab<Array>,
    /// Every stream in this database, addressed the same way.
    pub(crate) streams: Slab<Stream>,
    /// Every foreign body in this database, addressed the same way.
    ///
    /// A box per slot rather than a value, because the thing in it is not sized
    /// here and could not be. That is one indirection more than the other
    /// slabs pay, and it buys the graph, document and vector engines a place in
    /// the keyspace without this crate depending on any of them. See
    /// [`crate::foreign`].
    ///
    /// Empty on a server that has never held one, which is every server today,
    /// and an empty slab is three words.
    pub(crate) foreign: Slab<Box<dyn Foreign>>,
    /// How many keys hold a body that is in a slab right now.
    ///
    /// Not how many hold something that is not a string, which is what it used to
    /// be and what it still is on a database with no file behind it. A collection
    /// that has been demoted has no slab slot, so it does not count here, and
    /// that is what every reader of this number wants: [`Keyspace::free_body`]
    /// has nothing to free for one, and a sweep has nothing to move.
    ///

    /// This exists so that a database of nothing but strings, which is every
    /// benchmark today and most of what `SET` sees, can skip the body check in
    /// [`Keyspace::free_body`] on one predictable branch against a field that is
    /// already hot, rather than paying a second lookup per write forever.
    pub(crate) bodies: usize,
    /// Where a set changes representation.
    pub(crate) limits: set::Limits,
    /// Where a hash changes representation.
    pub(crate) hash_limits: hash::Limits,
    /// Where a list changes representation.
    pub(crate) list_limits: list::Limits,
    /// Where a sorted set changes representation.
    pub(crate) zset_limits: zset::Limits,
    /// Where a stream starts a new node, which is two `CONFIG` values.
    pub(crate) stream_limits: stream::Limits,
    /// What this database would evict, and therefore what a read writes back.
    ///
    /// One server wide setting in Redis, carried per database here for the same
    /// reason the size ladder is: a `Keyspace` is reached without a server and
    /// has to be able to answer on its own. `CONFIG SET maxmemory-policy` writes
    /// it to all of them.
    pub(crate) policy: Policy,
    /// The two numbers the LFU counter moves by, which are `CONFIG` values.
    pub(crate) lfu: Lfu,
    /// How many keys a round of eviction sampling looks at.
    ///
    /// `maxmemory-samples`, carried per database for the same reason the policy
    /// is. See [`evict::SAMPLES`] for why the default is five.
    pub(crate) samples: usize,
    /// The good candidates from earlier rounds of eviction sampling.
    ///
    /// See [`evict::Pool`]. Empty and costing nothing until the first eviction,
    /// which on most databases is never.
    pub(crate) pool: evict::Pool,
    /// Where `SPOP` and `SRANDMEMBER` draw from.
    pub(crate) rng: Rng,
    /// Where a demoted value goes and comes back from, if this database has one.
    ///
    /// `None` on a database with no file behind it, which is every embedded
    /// caller that never opened one and every test that does not care, and on
    /// such a database no record is ever cold and every check against this is a
    /// null test on a field in the same cache line as the map.
    ///
    /// Boxed rather than a type parameter on `Keyspace`. The parameter would
    /// have to be named by `yo-resp`, by the typed API and by every caller of
    /// either, all to spell a type that only the code opening the file knows,
    /// and it would be a parameter on the hot path to describe the cold one. See
    /// [`Blocks`].
    pub(crate) tier: Option<Tier<Box<dyn Blocks>>>,
    /// The last value read off the file, for the read that is being answered.
    ///
    /// A demoted value that is served rather than promoted has to live
    /// somewhere for the length of one command, because the record it came from
    /// holds an address and the caller was promised bytes. One buffer, cleared
    /// and refilled, on the same argument as [`Keyspace::scratch`]: a fault is
    /// already a device read and a malloc on top of it is free by comparison,
    /// but it is also unnecessary.
    ///
    /// It holds the value of the last key that was faulted and nothing says
    /// which key that was, which is why nothing reads it without having faulted
    /// in the same call. Everything that does goes through
    /// [`Keyspace::warm`].
    pub(crate) cold: Vec<u8>,
    /// Whose value is in [`Keyspace::cold`].
    ///
    /// Only ever read by a debug assertion, and it is there because the failure
    /// it catches is silent: a read that forgets to warm and then finds a cold
    /// record would hand back whatever the last fault put in the buffer, which
    /// is a real value belonging to a different key. A test would see plausible
    /// bytes and pass. The copy costs a key's worth of memcpy on a path that
    /// has just read a device, which is nothing next to what it is guarding.
    pub(crate) cold_key: Vec<u8>,
    /// A collection body on its way to the file or on its way back.
    ///
    /// Not [`Keyspace::cold`], which holds the value of the last string that was
    /// faulted and is read after the fact by [`Keyspace::value_of`]. This buffer
    /// is written and consumed inside one call and nothing looks at it
    /// afterwards, so sharing the other one would mean a `SADD` on a demoted set
    /// quietly replacing the bytes a `GET` in the same pipeline was about to hand
    /// back. One `Vec` each, cleared and refilled, is three words of a struct
    /// that already has a hundred.
    pub(crate) frozen: Vec<u8>,
    /// The last collection key that was resolved, for the command behind it.
    memo: Memo,
    /// One buffer for the commands that have to hold an element while the
    /// structure it came out of is being written.
    ///
    /// [`Keyspace::lmove`] is the reason this is here: it takes an element out
    /// of one list and puts it into another, so there is a moment where the
    /// bytes belong to nothing, and the borrow it would need to avoid that is a
    /// borrow of two lists at once when the two lists may be the same one. A
    /// `Vec` per call is the obvious way to cover that moment and it is a malloc
    /// and a free on a command that a queue sends millions of. This is the same
    /// `Vec` every time, cleared rather than freed, so the steady state is no
    /// allocator call at all.
    ///
    /// [`Keyspace::append`], [`Keyspace::setrange`] and the string arm of
    /// [`Keyspace::set_expiry`] use it for the same shape of problem: each of
    /// them has to hold the old value while it writes the new record, and each
    /// of them was doing that with a fresh `Vec` of the whole value. They cannot
    /// overlap, because each one puts the buffer back before it returns and one
    /// command runs at a time.
    ///
    /// It lives on the database and not on the caller because the callers are
    /// wire handlers that are handed a `&mut Keyspace` and nothing else.
    ///
    /// It starts at [`SCRATCH`] bytes rather than empty. An empty one grows on
    /// the first command that uses it, and that growth is a real allocation on a
    /// command path even though it happens once. Buying it here, where nobody is
    /// waiting, makes the rule Y7 enforces true without an exception written for
    /// it. A value larger than that still grows it, and that one is allocation
    /// proportional to what the caller sent rather than overhead per command.
    pub(crate) scratch: Vec<u8>,

    /// The same idea for indices rather than bytes.
    ///
    /// `ZRANDMEMBER` with a positive count under the size of the set does a
    /// partial Fisher-Yates, and that needs the permutation somewhere while it
    /// draws from it. One buffer, cleared and refilled, rather than one `Vec`
    /// per call, because sampling is a thing callers do in a loop.
    ///
    /// It does not start at a capacity, unlike [`Keyspace::scratch`]. There is
    /// no size to guess: the buffer has to be as long as the set, so the first
    /// call on a set larger than anything seen before grows it whatever it was
    /// given to start with. That growth is proportional to the data rather than
    /// per command.
    pub(crate) rows: Vec<usize>,

    /// The tables set algebra fills in, kept rather than built per call.
    ///
    /// Same idea again, one level up: a union walks everything into a hash
    /// table and lets the table be the duplicate check, and building that table
    /// was the largest single allocation left on any command path. See
    /// [`setops::Scratch`], which is where the two tables and the argument for
    /// them live.
    ///
    /// It is one table per database and not one per command, so a database that
    /// has answered a union over a million members holds a million member table
    /// until it answers a smaller one. That is the trade and it is the right way
    /// round: the same database had to build that table anyway, and the version
    /// that threw it away afterwards built it again on the next call.
    pub(crate) setops: crate::setops::Scratch,

    /// What the last geo search found, kept for the same reason.
    ///
    /// A search cannot answer in the order it walks: the nine hash boxes come
    /// out in hash order and the reply is in distance order, so every candidate
    /// has to be in hand before the first one can be written. See
    /// [`crate::geos::Scratch`], which is the two buffers and the argument for
    /// keeping them here.
    pub(crate) geo: crate::geos::Scratch,
}

/// How big [`Keyspace::scratch`] starts.
///
/// A kibibyte, which covers a value of any ordinary size and costs one
/// allocation per database. The number is not tuned and does not need to be: too
/// small only means the buffer grows once more on some later command, and too
/// large only means a kibibyte nobody used.
const SCRATCH: usize = 1024;

/// How many segments one call to [`Keyspace::victim`] will draw from.
///
/// A round is a whole segment, and a segment is sixty four buckets of seven
/// entries each before its overflow chains are counted, so one round almost
/// always answers. The retries are for the case where a round came back with
/// nothing usable, which happens when the segment it drew was empty or when a
/// `volatile` policy filtered out everything in it.
///
/// Four rather than more, because a round that comes back with nothing is
/// telling you something a fifth round will not change. That used to be the
/// wrong shape for the `volatile` policies, which could draw four rounds of
/// keys with no deadline on a database that had almost none, on a path where a
/// client is waiting. Those policies now draw from the second index of just the
/// keys that carry a deadline, the way Redis draws from `db->expires`, so every
/// key a round looks at is a key it is allowed to take.
const ROUNDS: usize = 4;

/// Where the last collection key resolved to, if it still resolves there.
///
/// Y13 says a batch of `SADD` on one key should be one table growth check, and
/// the same argument applies a step earlier: it should be one resolve. A
/// resolve is a hash, a bucket walk and a record read, and on a hot key every
/// command in the batch was paying for all three to be told the same answer the
/// command in front of it got.
///
/// One entry and not a cache, because one entry is the shape of the problem.
/// Single key `SADD` is the case with no spread to exploit, so the only reuse
/// there is to find is the command immediately before, and a bigger structure
/// would cost a lookup to avoid a lookup.
///
/// It holds a slot rather than reaching the body through an address, because a
/// slot is an index into the slab for its type and stays right for as long as
/// the key is there. That is also why nothing here memoizes a string: a string
/// lives in the record itself and moves when the record does.
///
/// It does carry the record's address alongside, and that is safe for a narrower
/// reason than the slot is. An address is only good until the next write, and
/// this whole memo is thrown away by the next write, so inside the window where
/// the memo answers at all the address is exactly as valid as the slot. What it
/// buys is the eviction stamp: a memo hit skips the probe, and without an address
/// the stamp would have to put the probe back and there would be no memo left.
///
/// What it is worth is measured rather than argued about, by the pair of rows
/// `engine/sadd` and `engine/sadd-alternating` in `yo-resp`'s `engine` bench.
/// The second one alternates between two keys, which defeats this on every
/// command and leaves both keys as warm in the cache as the one key was, so the
/// difference between the rows is close to this and nothing else. On an Apple M4
/// it is about nineteen nanoseconds a command, which is 1.25x at pipeline 64.
struct Memo {
    /// What the map's write counter said when this was taken.
    writes: u64,
    /// Whether there is anything here. Separate from the length because the
    /// empty key is a key, and `SADD "" m` is a command Redis accepts.
    live: bool,
    /// The type the key held, so a hit can still answer `WRONGTYPE`.
    kind: Kind,
    /// Where the body is in the slab for `kind`.
    slot: u32,
    /// Where the record is, for the stamp a hit still owes.
    addr: Addr,
    /// How much of `key` is the key.
    len: u8,
    key: [u8; Memo::MAX],
}

impl Memo {
    /// The longest key worth remembering.
    ///
    /// Thirty two bytes is half a cache line and covers every hot key anyone
    /// writes down, including the `myset:{tag}` the generators send. A longer
    /// key is not memoized rather than heap allocated, because the whole point
    /// of this is to not touch memory it does not have to.
    const MAX: usize = 32;

    const fn empty() -> Memo {
        Memo {
            writes: 0,
            live: false,
            kind: Kind::String,
            slot: 0,
            addr: Addr::NONE,
            len: 0,
            key: [0; Memo::MAX],
        }
    }

    /// What `key` resolved to last time, if that answer still stands.
    ///
    /// `writes` is the map's counter now. Any write at all since this was taken
    /// and the answer is thrown away, which is stricter than it has to be and is
    /// the version that cannot be wrong.
    #[inline]
    fn get(&self, writes: u64, key: &[u8]) -> Option<(Kind, u32, Addr)> {
        if !self.live || self.writes != writes || key.len() != self.len as usize {
            return None;
        }
        // `bytes_eq` and not `==`, which is a call into the platform's `memcmp`
        // for a key of a length the compiler cannot see. This is the one
        // comparison the hot key path always does, and on a profile of `SADD`
        // it was most of what the lookup cost.
        bytes_eq(&self.key[..key.len()], key).then_some((self.kind, self.slot, self.addr))
    }

    /// Remember that `key` is at `slot`, in the record at `addr`.
    #[inline]
    fn put(&mut self, writes: u64, key: &[u8], kind: Kind, slot: u32, addr: Addr) {
        if key.len() > Memo::MAX {
            self.live = false;
            return;
        }
        self.writes = writes;
        self.live = true;
        self.kind = kind;
        self.slot = slot;
        self.addr = addr;
        self.len = key.len() as u8;
        self.key[..key.len()].copy_from_slice(key);
    }
}

/// How many databases this process has made.
///
/// Mixed into a new database's seed so that the eight shards a server starts in
/// the same millisecond do not all draw the same members in the same order. It
/// is the only atomic in this file and it is touched once per database rather
/// than once per command, so it is not on any path Y1 cares about.
static MADE: AtomicU64 = AtomicU64::new(0);

impl Keyspace {
    /// An empty database on the system clock.
    #[must_use]
    pub fn new() -> Keyspace {
        Keyspace::with_clock(Clock::system())
    }

    /// An empty database on a clock of the caller's choosing.
    #[must_use]
    pub fn with_clock(clock: Clock) -> Keyspace {
        let made = MADE.fetch_add(1, Ordering::Relaxed);
        Keyspace {
            map: RawMap::new(),
            clock,
            expired: 0,
            evicted: 0,
            sets: Slab::new(),
            hashes: Slab::new(),
            lists: Slab::new(),
            zsets: Slab::new(),
            arrays: Slab::new(),
            streams: Slab::new(),
            foreign: Slab::new(),
            bodies: 0,
            limits: set::Limits::DEFAULT,
            hash_limits: hash::Limits::DEFAULT,
            list_limits: list::Limits::default(),
            zset_limits: zset::Limits::DEFAULT,
            stream_limits: stream::Limits::default(),
            policy: Policy::default(),
            lfu: Lfu::DEFAULT,
            samples: evict::SAMPLES,
            pool: evict::Pool::new(),
            rng: Rng::new(clock.now_ms() ^ made.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            tier: None,
            cold: Vec::new(),
            cold_key: Vec::new(),
            frozen: Vec::new(),
            memo: Memo::empty(),
            scratch: Vec::with_capacity(SCRATCH),
            rows: Vec::new(),
            setops: crate::setops::Scratch::new(),
            geo: crate::geos::Scratch::default(),
        }
    }

    /// Pin what `SPOP` and `SRANDMEMBER` draw.
    ///
    /// A database seeds itself from the clock and a counter, which is what a
    /// server wants and what a test cannot assert against. Every test in this
    /// crate that cares which member comes back calls this first, the same way
    /// every expiry test drives a fixed clock, and for the same reason: the one
    /// input that makes a result unrepeatable is better handed in than reached
    /// for.
    ///
    /// It is public because reproducing a bug report is the same problem. A
    /// seed printed in a crash report is worth having somewhere to put.
    #[inline]
    pub const fn seed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// What this database would evict, which is `CONFIG GET maxmemory-policy`.
    #[inline]
    #[must_use]
    pub const fn policy(&self) -> Policy {
        self.policy
    }

    /// Change what this database would evict.
    ///
    /// Every key already stored keeps whatever is in its access field, which is
    /// why Redis warns on `OBJECT FREQ` that switching at runtime takes time to
    /// adjust. Under the new policy those bits mean something else, and the only
    /// honest thing to do about it is to let them be corrected by use. A key
    /// nobody has touched since the switch reads as freshly used rather than as
    /// stale, which is the safe direction: the other one evicts the working set
    /// on the first pass after an operator changes a setting.
    ///
    /// The candidate pool does go, because a score only means anything against
    /// another score under the same rule and every number in there was worked
    /// out under the old one.
    #[inline]
    pub fn set_policy(&mut self, policy: Policy) {
        if policy != self.policy {
            self.pool.clear();
        }
        self.policy = policy;
    }

    /// The two numbers the LFU counter moves by, which are two `CONFIG` values.
    #[inline]
    #[must_use]
    pub const fn lfu(&self) -> Lfu {
        self.lfu
    }

    /// Change how fast the LFU counter climbs and decays.
    #[inline]
    pub const fn set_lfu(&mut self, lfu: Lfu) {
        self.lfu = lfu;
    }

    /// Seconds since `key` was last used, which is `OBJECT IDLETIME`.
    ///
    /// `None` for a key that is not there. A key that has never been stamped
    /// reads as zero rather than as ancient, which is what
    /// [`Access::is_unset`] is for.
    ///
    /// This does not count as a use. Redis looks the key up with its no touch
    /// flag here, and it has to: a diagnostic that resets the number it reports
    /// would answer zero every time it was asked.
    pub fn idle_secs(&mut self, key: &[u8]) -> Option<u64> {
        let addr = self.live_rec_untouched(key)?;
        let now = self.clock.now_ms();
        Some(self.access_at(addr).idle_secs(now))
    }

    /// How often `key` is used, which is `OBJECT FREQ`.
    ///
    /// The eight bit counter, decayed to now, on the same terms as
    /// [`Keyspace::idle_secs`]: `None` for a key that is not there, and asking
    /// is not using.
    ///
    /// The caller is the one that has to check the policy first. This reports
    /// what the bits say, and under a policy that is not LFU they say something
    /// else, which is a refusal on the wire rather than a number.
    pub fn freq(&mut self, key: &[u8]) -> Option<u8> {
        let addr = self.live_rec_untouched(key)?;
        let (now, lfu) = (self.clock.now_ms(), self.lfu);
        Some(self.access_at(addr).freq(now, lfu))
    }

    /// Write a record under `key`, with the access field the policy wants on it.
    ///
    /// Every record this crate writes goes through here, which is the point of
    /// it. A record is written fresh whenever a key is created and whenever a
    /// string's value changes, and a fresh record starts with the blank field
    /// [`value::write_record`] leaves behind. Blank reads as freshly used, which
    /// is right at the moment of writing and wrong a minute later, so something
    /// has to stamp it and this is the only place that knows the clock.
    ///
    /// Redis stamps at the same moment, in `createObject`, and for the same
    /// reason.
    pub(crate) fn write_rec(
        &mut self,
        key: &[u8],
        len: usize,
        fill: impl FnOnce(&mut [u8]),
    ) -> Option<usize> {
        let a = self.access_for_write(key);
        // The one bit in the record that says the key has a deadline, handed
        // straight back to the map. What the map does with it is keep a second
        // index of just those records, so the expire cycle and the volatile
        // eviction policies have somewhere to sample from that is not the whole
        // keyspace. Nothing here counts anything: the map's own count of marked
        // records is the number, and a number kept in two places is a number
        // that eventually disagrees with itself.
        self.map.set_with(
            key,
            len,
            |_| {},
            |out| {
                fill(out);
                value::set_access(out, a);
                value::has_expiry(out)
            },
        )
    }

    /// Take `key` out of the map, keeping the deadline count right.
    ///
    /// The other half of [`Keyspace::write_rec`], and every path that removes a
    /// record goes through one of the two. That is `DEL`, lazy expiry, eviction
    /// and the source key of a `RENAME`, and the last one is why this is not
    /// simply folded into [`Keyspace::drop_key`]: a rename hands the body to the
    /// destination and must not free it, so it deletes the source record without
    /// dropping the key, and it still has to be counted.
    #[inline]
    pub(crate) fn del_rec(&mut self, key: &[u8]) -> bool {
        self.map.del(key)
    }

    /// What the access field of a record about to be written should say.
    ///
    /// Under the eight policies that read the field as a clock this is the time,
    /// with no probe and no thought: writing a key is using it, and under the LRM
    /// pair writing it is the only thing that counts as using it.
    ///
    /// Under LFU it is the counter that is already there, carried across the
    /// rewrite unchanged. Unchanged rather than incremented, because the lookup
    /// that resolved the key for this write already counted the access, and
    /// counting it twice would rank a key that is written more highly than a key
    /// that is read the same number of times. A key that is not there yet starts
    /// at [`crate::access::LFU_INIT`], which is where Redis starts a new object.
    ///
    /// The probe is the reason this is written as two cases rather than one. It
    /// is paid only under an LFU policy, so the default policy and every other
    /// one write a record for exactly what it cost before.
    fn access_for_write(&mut self, key: &[u8]) -> Access {
        let now = self.clock.now_ms();
        if !self.policy.is_lfu() {
            return Access::lru(now);
        }
        match self.map.get(key).and_then(value::access) {
            Some(a) if !a.is_unset() => a,
            _ => Access::lfu(now),
        }
    }

    /// The access field of the record at `addr`, or the unset one for a record
    /// written before the field existed.
    #[inline]
    fn access_at(&self, addr: Addr) -> Access {
        value::access(self.map.value_at(addr)).unwrap_or_default()
    }

    /// Write the access field back to the record at `addr`.
    ///
    /// The whole reason the field exists, and it runs on nearly every command,
    /// so what it does is a load, an arithmetic step and a three byte store into
    /// a cache line the caller has just read. It does not count as a write to the
    /// map, because nothing moves and counting it would throw the [`Memo`] away
    /// once per command. See [`RawMap::value_at_mut`].
    ///
    /// The LFU arm reads before it writes, because the counter it produces is a
    /// function of the counter that is there. The clock arm does not, because the
    /// time is the time whatever the record used to say.
    #[inline]
    fn stamp(&mut self, addr: Addr) {
        let now = self.clock.now_ms();
        if self.policy.is_lfu() {
            let (lfu, current) = (self.lfu, self.access_at(addr));
            let next = current.touched(now, lfu, &mut self.rng);
            value::set_access(self.map.value_at_mut(addr), next);
        } else {
            value::set_access(self.map.value_at_mut(addr), Access::lru(now));
        }
    }

    /// Where a set changes representation, which is three `CONFIG` values.
    #[inline]
    pub const fn limits(&self) -> &set::Limits {
        &self.limits
    }

    /// Change where a set changes representation.
    ///
    /// Moving these does not rewrite the sets that already exist, which is what
    /// Redis does too: `CONFIG SET set-max-listpack-entries 0` leaves every
    /// listpack alone and only decides what the next `SADD` builds.
    #[inline]
    pub const fn set_limits(&mut self, limits: set::Limits) {
        self.limits = limits;
    }

    /// Where a hash changes representation, which is two `CONFIG` values.
    #[inline]
    pub const fn hash_limits(&self) -> &hash::Limits {
        &self.hash_limits
    }

    /// Change where a hash changes representation.
    ///
    /// Same rule as the set: moving these leaves every hash that already exists
    /// exactly as it is, and only decides what the next `HSET` builds.
    #[inline]
    pub const fn set_hash_limits(&mut self, limits: hash::Limits) {
        self.hash_limits = limits;
    }

    /// Where a list changes representation, which is one `CONFIG` value.
    #[inline]
    pub const fn list_limits(&self) -> &list::Limits {
        &self.list_limits
    }

    /// Change where a list changes representation.
    ///
    /// Same rule again: this decides what the next `LPUSH` builds and leaves
    /// every list that already exists alone. `list-max-listpack-size` is one
    /// number rather than two, and [`list::Limits::of`] is what turns it into
    /// the pair this holds.
    #[inline]
    pub const fn set_list_limits(&mut self, limits: list::Limits) {
        self.list_limits = limits;
    }

    /// Where a stream starts a new node, which is two `CONFIG` values.
    #[inline]
    pub const fn stream_limits(&self) -> &stream::Limits {
        &self.stream_limits
    }

    /// Change where a stream starts a new node.
    ///
    /// Same rule as the other four: this decides what the next `XADD` builds
    /// and leaves every node that is already full exactly as it is, which is
    /// also what Redis does, since a node is never resized after it is written.
    #[inline]
    pub const fn set_stream_limits(&mut self, limits: stream::Limits) {
        self.stream_limits = limits;
    }

    /// Where a sorted set changes representation, which is two `CONFIG` values.
    #[inline]
    pub const fn zset_limits(&self) -> &zset::Limits {
        &self.zset_limits
    }

    /// Change where a sorted set changes representation.
    ///
    /// Same rule as the other three: this decides what the next `ZADD` builds
    /// and leaves every sorted set that already exists exactly as it is.
    #[inline]
    pub const fn set_zset_limits(&mut self, limits: zset::Limits) {
        self.zset_limits = limits;
    }

    /// The clock expiry compares against.
    #[inline]
    pub const fn clock(&self) -> &Clock {
        &self.clock
    }

    /// The clock, to refresh once per turn of the loop.
    #[inline]
    pub const fn clock_mut(&mut self) -> &mut Clock {
        &mut self.clock
    }

    /// The map underneath, for statistics and for compaction.
    #[inline]
    pub const fn map(&self) -> &RawMap {
        &self.map
    }

    /// How many keys are stored, including any that are dead and not yet
    /// noticed. This is Redis's `DBSIZE`, which counts the same way.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether anything is stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// What `key` holds, or `None` if there is nothing under it.
    ///
    /// This is `TYPE`. A key past its deadline is reaped first, so a dead key
    /// answers `None` and not the type it used to be.
    ///
    /// One lookup, because the tag and the deadline are both in the record the
    /// lookup returned. Reading the kind out before the reap rather than after
    /// is what keeps it to one.
    ///
    /// It does not go through the lookup that stamps, and so it leaves the
    /// eviction clock where it was, which is right and is worth saying rather than
    /// leaving to be inferred from the shape of the code. `TYPE` is one of the
    /// commands Redis looks up with its no touch flag, along with the `OBJECT`
    /// subcommands underneath this, which read through `reap` for the same
    /// reason.
    pub fn kind_of(&mut self, key: &[u8]) -> Option<Kind> {
        let now = self.clock.now_ms();
        let (kind, dead) = self
            .map
            .get(key)
            .map(|rec| (value::kind(rec), value::is_expired(rec, now)))?;
        if dead {
            self.drop_key(key);
            self.expired += 1;
            return None;
        }
        Some(kind)
    }

    /// How a set is represented, or `None` if `key` is not a set.
    ///
    /// This follows the slot and asks the body rather than reading the record,
    /// because the record only holds a number. Putting a copy of the
    /// representation in the record's two spare encoding bits would mean
    /// rewriting the record every time a set was promoted, for the sake of a
    /// command nobody calls in a loop, and would leave two places able to
    /// disagree about the same fact.
    /// A demoted set is brought back to answer, which is the one thing this
    /// costs that the others do not. The word is a property of the body and the
    /// body is on the device, so there is nothing else to read it off. It is one
    /// device read for a command nobody sends in a loop, and the alternative is a
    /// copy of the word in the record's two spare encoding bits with two places
    /// then able to disagree about it.
    pub fn set_encoding(&mut self, key: &[u8]) -> Option<set::Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        if value::kind(rec) != Kind::Set {
            return None;
        }
        let cold = value::Meta::from_byte(rec[0]).is_cold();
        let at = if cold {
            // The record is given up here, because promotion writes a new one.
            self.promote_body(key).ok()??
        } else {
            value::slot(rec)
        };
        Some(self.sets.get(at)?.encoding())
    }

    /// How a hash is represented, or `None` if `key` is not a hash.
    ///
    /// The same shape as [`Keyspace::set_encoding`] and for the same reason: the
    /// record holds a slot number and the body is the thing that knows which of
    /// the two it currently is.
    pub fn hash_encoding(&mut self, key: &[u8]) -> Option<hash::Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        if value::kind(rec) != Kind::Hash {
            return None;
        }
        let at = value::slot(rec);
        Some(self.hashes.get(at)?.encoding())
    }

    /// How a list is represented, or `None` if `key` is not a list.
    ///
    /// The same shape as [`Keyspace::set_encoding`], and the same argument for
    /// asking the body rather than reading a copy out of the record.
    pub fn list_encoding(&mut self, key: &[u8]) -> Option<list::Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        if value::kind(rec) != Kind::List {
            return None;
        }
        let at = value::slot(rec);
        Some(self.lists.get(at)?.encoding())
    }

    /// How a sorted set is represented, or `None` if `key` is not one.
    pub fn zset_encoding(&mut self, key: &[u8]) -> Option<zset::Encoding> {
        self.reap(key);
        let rec = self.map.get(key)?;
        if value::kind(rec) != Kind::Zset {
            return None;
        }
        let at = value::slot(rec);
        Some(self.zsets.get(at)?.encoding())
    }

    /// Put a foreign body under `key`, over whatever was there.
    ///
    /// The keyspace takes the box and frees it when the key goes, which is the
    /// whole reason a graph lives in here rather than in a table beside it. See
    /// [`crate::foreign`] for why that mattered enough to spend the last tag
    /// pattern on.
    ///
    /// Overwriting is allowed and is what a caller that has just decided to
    /// replace a key wants. A caller that did not mean to overwrite asks
    /// [`Keyspace::kind_of`] first, which is what the commands above do so they
    /// can answer WRONGTYPE rather than quietly throw a hash away.
    pub fn put_foreign(&mut self, key: &[u8], body: Box<dyn Foreign>) -> u32 {
        self.free_body(key);
        let at = self.foreign.insert(body);
        let len = value::slot_record_len(false);
        self.write_rec(key, len, |out| {
            value::write_slot_record(out, Kind::Foreign, at, None);
        });
        self.bodies += 1;
        at
    }

    /// The foreign body under `key`.
    ///
    /// `None` for a key that is not there or has expired, an error for a key
    /// holding something this crate does understand, which is the same three
    /// way answer every other type's entry point gives.
    ///
    /// The caller turns the `&dyn Foreign` back into its own type with
    /// [`downcast_ref`], and a `None` from that is a key holding a different
    /// foreign body, which is also WRONGTYPE and is the caller's to report
    /// because only it knows which one it wanted.
    ///
    /// [`downcast_ref`]: crate::Foreign
    pub fn foreign(&mut self, key: &[u8]) -> Result<Option<&dyn Foreign>> {
        let Some(at) = self.live_slot(key, Kind::Foreign)? else {
            return Ok(None);
        };
        Ok(Some(
            self.foreign
                .get(at)
                .expect("the record points at its body")
                .as_ref(),
        ))
    }

    /// The same, with a mutable borrow.
    pub fn foreign_mut(&mut self, key: &[u8]) -> Result<Option<&mut dyn Foreign>> {
        let Some(at) = self.live_slot(key, Kind::Foreign)? else {
            return Ok(None);
        };
        Ok(Some(
            self.foreign
                .get_mut(at)
                .expect("the record points at its body")
                .as_mut(),
        ))
    }

    /// Drop `key` if the foreign body under it has gone empty.
    ///
    /// Redis deletes a key when its collection empties, and a client can see
    /// the difference, so every command that removes something calls this
    /// afterwards rather than each of them deciding what empty means.
    pub fn reap_foreign(&mut self, key: &[u8]) {
        let gone = matches!(self.foreign(key), Ok(Some(b)) if b.is_empty());
        if gone {
            self.drop_key(key);
        }
    }

    /// The foreign body under `key`, without the reap or the type check.
    ///
    /// For the arms that already have a live record in hand and only want the
    /// body, where going through [`Keyspace::foreign`] would mean reaping a key
    /// that was read a line ago.
    fn foreign_at(&mut self, key: &[u8]) -> Option<&dyn Foreign> {
        let rec = self.map.get(key)?;
        let at = value::slot(rec);
        Some(self.foreign.get(at)?.as_ref())
    }

    /// What `TYPE` should say about `key`.
    ///
    /// [`Kind::name`] for everything this crate knows, and the body's own word
    /// for a foreign one, because a client asking about a graph is told `graph`
    /// and not `foreign`. `None` for a key that is not there, which is the
    /// `none` Redis answers with.
    pub fn type_name(&mut self, key: &[u8]) -> Option<&'static str> {
        match self.kind_of(key)? {
            Kind::Foreign => self.foreign_at(key).map(Foreign::type_name),
            kind => Some(kind.name()),
        }
    }

    /// `OBJECT ENCODING key`, as the word Redis puts on the wire.
    ///
    /// One place that knows every type's answer, so that adding the hash means
    /// adding an arm here and not finding the four callers that each worked it
    /// out for themselves.
    pub fn encoding_name(&mut self, key: &[u8]) -> Option<&'static str> {
        match self.kind_of(key)? {
            Kind::String => self.encoding(key).map(value::Encoding::name),
            Kind::Set => self.set_encoding(key).map(set::Encoding::name),
            Kind::Hash => self.hash_encoding(key).map(hash::Encoding::name),
            Kind::List => self.list_encoding(key).map(list::Encoding::name),
            Kind::Zset => self.zset_encoding(key).map(zset::Encoding::name),
            // The one type with one encoding, so there is nothing to ask.
            Kind::Array => Some("sliced-array"),
            // The same, and Redis's word for it rather than a description of
            // the node layout.
            Kind::Stream => Some("stream"),
            // The body knows and this does not, which is the whole point of it.
            Kind::Foreign => self.foreign_at(key).map(Foreign::encoding),
        }
    }

    /// Put a deadline on `key`, or take one off. Answers whether it was there.
    ///
    /// Any type. A deadline lives in the record and changes its length, so this
    /// writes the record again rather than patching it, and for a set that is
    /// five bytes or thirteen and never the members. The body is left exactly
    /// where it is, which is why this writes through the map instead of taking
    /// the free the body path an overwrite takes.
    ///
    /// This is the raw write. [`Keyspace::expire`] and [`Keyspace::persist`] are
    /// what `EXPIRE` and its family call, and they come through here once they
    /// have worked out whether the deadline is allowed to move.
    pub fn set_expiry(&mut self, key: &[u8], at: Option<u64>) -> bool {
        self.reap(key);
        let Some(rec) = self.map.get(key) else {
            return false;
        };
        if value::expire_at(rec) == at {
            return true;
        }
        // A deadline does not move a value that is on the file, it changes the
        // eight bytes in front of the address. So this writes a new pointer
        // rather than reading the value back to write it out again, and `EXPIRE`
        // on a demoted key costs no device read at all. That is the same promise
        // `TTL` and `STRLEN` keep and it is why the header stayed in memory.
        //
        // Ahead of the type split rather than inside the string arm, because a
        // demoted set is a pointer in exactly the same way and writing a slot
        // record over one would leave a record pointing at a slab slot that
        // belongs to something else.
        if let Some(c) = value::cold(rec) {
            let m = value::Meta::from_byte(rec[0]);
            let (kind, enc) = (m.kind(), m.encoding());
            let was = value::access(rec).unwrap_or_default();
            self.map.set_with(
                key,
                value::cold_record_len(at.is_some()),
                |_| {},
                |out| {
                    value::write_cold_record(out, kind, enc, c.at, c.len, at);
                    value::set_access(out, was);
                    value::has_expiry(out)
                },
            );
            return true;
        }
        // Read what has to survive out of the record before writing over it.
        match value::kind(rec) {
            Kind::String => {
                // Through the scratch buffer rather than a fresh `Vec`, since
                // `EXPIRE` on a string is a command a cache sends as often as
                // the `SET` before it.
                let mut bytes = std::mem::take(&mut self.scratch);
                bytes.clear();
                value::read(rec).write_to(&mut bytes);
                self.store(key, &bytes, at);
                self.scratch = bytes;
            }
            // Every body type writes the same record: a tag and a slot number.
            // The body is not touched and does not need to be, which is the
            // whole point of keeping it out of the record.
            kind @ (Kind::Set
            | Kind::Hash
            | Kind::List
            | Kind::Zset
            | Kind::Array
            | Kind::Stream
            | Kind::Foreign) => {
                let slot = value::slot(rec);
                let len = value::slot_record_len(at.is_some());
                self.write_rec(key, len, |out| {
                    value::write_slot_record(out, kind, slot, at);
                });
            }
        }
        true
    }

    /// The key's deadline, as the three way answer `TTL` and `PTTL` are built on.
    ///
    /// [`Ask::Missing`] for a key that is not there, [`Ask::NoDeadline`] for one
    /// that is and has no deadline, and the absolute millisecond otherwise. A key
    /// past its deadline is reaped on the way through, so it answers `Missing`
    /// and not the moment that has gone.
    ///
    /// Asking when a key dies is not using it, so this does not stamp the
    /// eviction clock. Redis reads the key with its no touch flag here for the
    /// same reason, and it matters more than it looks: a client polling `TTL` on
    /// a key would otherwise keep that key at the top of the working set for as
    /// long as it kept asking whether it was about to go.
    pub fn deadline_of(&mut self, key: &[u8]) -> Ask {
        let Some(addr) = self.live_rec_untouched(key) else {
            return Ask::Missing;
        };
        match value::expire_at(self.map.value_at(addr)) {
            Some(at) => Ask::At(at),
            None => Ask::NoDeadline,
        }
    }

    /// Move `key`'s deadline to `at`, if `cond` lets it.
    ///
    /// This is `EXPIRE`, `PEXPIRE`, `EXPIREAT` and `PEXPIREAT`, which differ only
    /// in the unit and the origin of the number. All four turn it into one
    /// absolute millisecond before they get here, so the condition rules live in
    /// one place and the four commands cannot drift apart.
    ///
    /// A deadline that has already passed deletes the key rather than being
    /// stored, and the answer says so. `EXPIRE` cannot report the difference
    /// because it replies 1 either way, but the caller is not always `EXPIRE`,
    /// and a delete is a different thing from a deadline.
    ///
    /// The condition is checked before the past check, which is the order Redis
    /// uses and is the one that matters: `EXPIRE key 0 XX` on a key with no
    /// deadline answers 0 and leaves the key alone, rather than deleting it.
    pub fn expire(&mut self, key: &[u8], at: u64, cond: Cond) -> Applied {
        let prev = match self.deadline_of(key) {
            Ask::Missing => return Applied::Missing,
            Ask::NoDeadline => None,
            Ask::At(at) => Some(at),
        };
        let done = ttl::decide(prev, at, cond, self.clock.now_ms());
        match done {
            Applied::Ok => {
                self.set_expiry(key, Some(at));
            }
            // The structure that answered `Deleted` for a field only holds
            // deadlines, so its caller has to remove the field. Here the caller
            // is us and the key is ours, so it goes now.
            Applied::Deleted => {
                self.drop_key(key);
            }
            Applied::Missing | Applied::NotMet => {}
        }
        done
    }

    /// Take `key`'s deadline off. Answers whether there was one to take.
    ///
    /// This is `PERSIST`, and the reply is the same 0 for a key that is not there
    /// and a key that was never going to expire, which is Redis's answer and not
    /// a shortcut here.
    pub fn persist(&mut self, key: &[u8]) -> bool {
        if !matches!(self.deadline_of(key), Ask::At(_)) {
            return false;
        }
        self.set_expiry(key, None);
        true
    }

    /// Give back whatever `key` holds outside its record, if it holds anything.
    ///
    /// Every path that deletes a key or writes over one has to come through
    /// here, because a set that loses its record without losing its slab slot is
    /// a leak that nothing ever notices: the memory is reachable, the slot is
    /// never reused, and `DBSIZE` looks right. Six delete sites and four string
    /// writers each remembering to do it themselves is five chances to forget,
    /// and one of them would be forgotten. So this is the funnel, and when the
    /// hash type lands the only place that changes is the match below.
    ///
    /// The record is left alone. This frees the body and the caller either
    /// deletes the record or writes a new one over it.
    pub(crate) fn free_body(&mut self, key: &[u8]) {
        if self.bodies == 0 {
            return;
        }
        let Some(rec) = self.map.get(key) else {
            return;
        };
        // A body that has been moved to the file has no slab slot to give back,
        // and the four bytes where the slot number would be are the front of an
        // address. Reading them as a slot and freeing it would hand back a slot
        // belonging to a different key. The chunks on the file are left where
        // they are, which is what the log's compaction collects. See
        // [`crate::tier`].
        if value::Meta::from_byte(rec[0]).is_cold() {
            return;
        }
        match value::kind(rec) {
            Kind::String => {}
            Kind::Set => {
                let at = value::slot(rec);
                self.sets.remove(at);
                self.bodies -= 1;
            }
            Kind::Hash => {
                let at = value::slot(rec);
                self.hashes.remove(at);
                self.bodies -= 1;
            }
            Kind::List => {
                let at = value::slot(rec);
                self.lists.remove(at);
                self.bodies -= 1;
            }
            Kind::Zset => {
                let at = value::slot(rec);
                self.zsets.remove(at);
                self.bodies -= 1;
            }
            Kind::Array => {
                let at = value::slot(rec);
                self.arrays.remove(at);
                self.bodies -= 1;
            }
            Kind::Stream => {
                let at = value::slot(rec);
                self.streams.remove(at);
                self.bodies -= 1;
            }
            Kind::Foreign => {
                let at = value::slot(rec);
                self.foreign.remove(at);
                self.bodies -= 1;
            }
        }
    }

    /// Delete `key` and whatever it held. Answers whether it was there.
    #[inline]
    pub(crate) fn drop_key(&mut self, key: &[u8]) -> bool {
        self.free_body(key);
        self.del_rec(key)
    }

    /// Drop `key` if its deadline has passed.
    ///
    /// This is lazy expiry and it is half of the story. The other half is
    /// [`Keyspace::expire_cycle`], which the maintenance slice runs and which is
    /// what stops a key nobody ever reads again from holding its memory forever
    /// (`14` section 1).
    ///
    /// Every public read calls this first, whatever type it is reading, which
    /// is why it is here and not in the file for any one type.
    #[inline]
    pub(crate) fn reap(&mut self, key: &[u8]) {
        let now = self.clock.now_ms();
        let dead = self.map.get(key).is_some_and(|r| value::is_expired(r, now));
        if dead {
            self.drop_key(key);
            self.expired += 1;
        }
    }

    /// Give this database somewhere to keep values that are not in memory.
    ///
    /// Until this is called nothing is ever demoted, no record is ever cold and
    /// every command takes exactly the path it took before, which is why a
    /// database that never opens a file pays nothing for this existing.
    ///
    /// The store is whatever the caller wants it to be. In a server it is the
    /// shard's log. In a test it is a vector. This crate does not depend on
    /// either and does not want to: [`Blocks`] is an append that hands
    /// back an address and a read that takes one, and that is the whole of the
    /// contract between the memory engine and whatever is under it.
    pub fn attach(&mut self, blocks: Box<dyn Blocks>) {
        self.tier = Some(Tier::new(blocks));
    }

    /// The tier, if one was attached, for its counters.
    #[must_use]
    pub const fn tier(&self) -> Option<&Tier<Box<dyn Blocks>>> {
        self.tier.as_ref()
    }

    /// The tier, mutably, for a caller driving a sweep.
    pub const fn tier_mut(&mut self) -> Option<&mut Tier<Box<dyn Blocks>>> {
        self.tier.as_mut()
    }

    /// Move one key's value out to the file.
    ///
    /// Answers whether it went. A key that is not there, one that is int
    /// encoded, one holding a type that does not move yet and one whose value is
    /// shorter than the pointer that would replace it all answer false, and so
    /// does every key on a database with nothing attached.
    ///
    /// This is the single key form, which is what a test and a `DEBUG`
    /// subcommand want. What a server under memory pressure wants is
    /// [`Keyspace::relieve`].
    ///
    /// One entry point for both kinds of value, because a caller naming a key
    /// should not have to know whether its body is in the record or in a slab.
    /// The two paths underneath are different all the way down: a string goes
    /// through the tier, and a collection goes through `demote_body` beside
    /// this, which frees a slab slot and grows a record.
    ///
    /// # Errors
    ///
    /// Whatever the store says when it will not take the bytes.
    pub fn demote(&mut self, key: &[u8]) -> Result<bool> {
        if self.tier.is_none() {
            return Ok(false);
        }
        let Some(addr) = self.map.find(key) else {
            return Ok(false);
        };
        if value::kind(self.map.value_at(addr)).is_body() {
            return self.demote_body(key);
        }
        let tier = self.tier.as_mut().expect("checked at the top");
        tier.demote(&mut self.map, key)
    }

    /// How many bytes the attached store is holding, or `None` if there is not
    /// one.
    ///
    /// `None` and `Some(0)` are different answers and the difference is the one
    /// `maxstore` turns on. A database with nothing attached cannot migrate and
    /// has to evict, and a database with an empty file attached can migrate the
    /// moment it needs to.
    #[must_use]
    pub fn store_bytes(&self) -> Option<u64> {
        self.tier.as_ref().map(Tier::store_bytes)
    }

    /// Move values out to the file until at least `shed` bytes of memory have
    /// gone.
    ///
    /// Answers with a [`Relief`], which is how many keys went and how much
    /// memory that gave back. This is `maxmemory` under the inversion `14`
    /// describes: the limit that used to throw keys away now moves them, and
    /// what a client stored is still there afterwards.
    ///
    /// Bytes to shed rather than a target to reach, because the caller with the
    /// limit is a server holding sixteen databases against one number and what
    /// it knows is how far over it is, not what any one database should be
    /// holding. `usize::MAX` means everything that can go, which is what a sweep
    /// wants.
    ///
    /// Victims are chosen by the same policy `maxmemory-policy` names, so a
    /// database set to `allkeys-lru` demotes the coldest keys and one set to
    /// `volatile-ttl` demotes the ones closest to expiring. See
    /// [`Tier::relieve`] for what a sweep does and where it stops.
    ///
    /// # Why `noeviction` still moves values
    ///
    /// Because it says do not lose data, and moving a value to the file does not
    /// lose any. The policy is two things at once in Redis, whether to give
    /// memory back at all and which keys to take it from, and only the second of
    /// those means anything here. So the default policy picks victims the way
    /// `allkeys-lru` does and the promise it was set for is kept: every key a
    /// client stored is still readable afterwards.
    ///
    /// The alternative is a server that was given a file, was given a limit, and
    /// answers writes with OOM until somebody finds the third setting that turns
    /// the file on. That is a trap and not a default.
    ///
    /// # Two passes, because the two kinds of value are counted in different
    /// places
    ///
    /// Strings go first, through [`Tier::relieve`], which measures itself against
    /// the arena because a string is its record and moving one makes the arena
    /// smaller. Collections cannot be swept that way. A collection's body is in a
    /// slab the arena knows nothing about, and moving one makes the arena
    /// **bigger**, because a twenty byte pointer replaces an eight byte slot
    /// number. A loop that watched the arena would demote every collection in the
    /// database, watch its number go up the whole time, and never stop.
    ///
    /// So the second pass is here rather than in the tier, and it measures itself
    /// against [`Keyspace::memory_bytes`], which is the arena and the slabs
    /// together. That is the only number that goes down when a body moves, and it
    /// is the number the server's limit is compared against anyway.
    ///
    /// # Errors
    ///
    /// Whatever the store says when it will not take the bytes.
    pub fn relieve(&mut self, shed: usize) -> Result<Relief> {
        if self.tier.is_none() {
            return Ok(Relief::default());
        }
        let now = self.clock.now_ms();
        let policy = match self.policy {
            Policy::NoEviction => Policy::AllKeysLru,
            chosen => chosen,
        };
        let lfu = self.lfu;
        let start = self.memory_bytes();
        let target = start.saturating_sub(shed);
        let budget = self.map.memory_bytes().saturating_sub(shed);
        let tier = self.tier.as_mut().expect("checked just above");
        let mut relief = tier.relieve(&mut self.map, budget, policy, now, lfu)?;
        // A sweep compacts the arena, which moves records, and the memo holds a
        // record's address. It is normally thrown away by the map's write
        // counter moving, and a sweep that demoted nothing and compacted anyway
        // is the one case where that counter does not move.
        self.memo = Memo::empty();
        if self.bodies > 0 && self.memory_bytes() > target {
            relief.moved += self.shed_bodies(target, policy, now, lfu)?;
        }
        relief.freed = start.saturating_sub(self.memory_bytes());
        Ok(relief)
    }

    /// Sample the keyspace for collection bodies and move them out until
    /// [`Keyspace::memory_bytes`] is under `target`.
    ///
    /// The same shape as [`Tier::relieve`]'s loop, with the same stop rule and
    /// for the same reason: a round that finds nothing is a collision rather than
    /// a conclusion, so it takes [`tier::BARREN`] of them in a row to give up.
    /// What is different is the number being watched, which is explained on
    /// [`Keyspace::relieve`], and that there is no compaction step in here. The
    /// arena grows on this path rather than shrinking, so there is nothing for a
    /// compaction to hand back that the string pass has not already taken.
    fn shed_bodies(
        &mut self,
        target: usize,
        policy: Policy,
        now_ms: u64,
        lfu: Lfu,
    ) -> Result<usize> {
        let mut moved = 0;
        let mut barren = 0;
        // The pool comes out of the keyspace for the length of the sweep, because
        // it hands back a borrow of itself and demoting one victim needs the
        // whole keyspace. `kb` is the same borrow copied somewhere it can live
        // across that call, and it is one allocation for the sweep rather than
        // one per victim.
        let mut pool = core::mem::take(&mut self.pool);
        let mut kb: Vec<u8> = Vec::new();
        while self.memory_bytes() > target {
            pool.clear();
            let r = self.rng.next_u64();
            let pool = &mut pool;
            let mut seen = 0usize;
            let mut found = 0usize;
            // The body check is on the record and not on the slab, so the sample
            // closure does not need a second borrow of the keyspace. A record
            // that turns out to hold a body too small to be worth moving is
            // refused by `demote_body`, which has the slab in hand by then.
            self.map.sample(r, |k, v, _| {
                seen += 1;
                let m = value::Meta::from_byte(v[0]);
                if !m.is_cold() && m.kind() == Kind::Set {
                    pool.offer(k, evict::score(v, policy, now_ms, lfu));
                    found += 1;
                }
                found < evict::CANDIDATES && seen < tier::WALK
            });

            let mut round = 0;
            while let Some(k) = pool.take() {
                kb.clear();
                kb.extend_from_slice(k);
                if self.demote_body(&kb)? {
                    round += 1;
                }
            }
            if round == 0 {
                barren += 1;
                if barren == tier::BARREN {
                    break;
                }
                continue;
            }
            barren = 0;
            moved += round;
        }
        self.pool = pool;
        // Every demotion rewrote a record, so nothing the memo holds is worth
        // keeping and one of the slot numbers in it is now a freed slab slot.
        self.memo = Memo::empty();
        Ok(moved)
    }

    /// Move `key`'s collection body out to the file.
    ///
    /// `Ok(false)` for a key that is not there, that holds a type this does not
    /// move yet, that is already on the file, or whose body is smaller than the
    /// pointer that would replace it. None of those is an error, for the same
    /// reason none of them is in [`crate::tier::demote`](Tier::demote): a sweep
    /// asks about a lot of keys and most of the answers are no.
    ///
    /// # What is different from a string
    ///
    /// A string's value is its record, so the tier can read it, write it out and
    /// rewrite the record without anyone else being involved. A collection's body
    /// is in a slab and its record holds a four byte number, so three things have
    /// to happen here and only here: the body is turned into bytes that mean
    /// something on a device, the slab slot is freed, and the record grows from
    /// eight bytes to twenty because an address and a length are longer than a
    /// slot number.
    ///
    /// That last part is why the memory a sweep frees does not show up in the
    /// arena. Demoting a collection makes the arena bigger and the slab smaller,
    /// and it is the sum that goes down. See [`Keyspace::relieve`].
    ///
    /// # Errors
    ///
    /// Whatever the store says when it cannot take the bytes.
    pub(crate) fn demote_body(&mut self, key: &[u8]) -> Result<bool> {
        if self.tier.is_none() {
            return Ok(false);
        }
        let Some(addr) = self.map.find(key) else {
            return Ok(false);
        };
        let rec = self.map.value_at(addr);
        let m = value::Meta::from_byte(rec[0]);
        // Sets only, for now. Every other collection wants the same three steps
        // and a `freeze` of its own, and the shape of the third one is what this
        // is establishing.
        if m.is_cold() || m.kind() != Kind::Set {
            return Ok(false);
        }
        let expire_at = value::expire_at(rec);
        // Carried across and not restamped, as in `Tier::demote`. A key that was
        // moved out was not used, and a demotion that looked like a use would
        // make the next sweep pick the wrong victim.
        let was = value::access(rec).unwrap_or_default();
        let slot = value::slot(rec);
        let Some(set) = self.sets.get(slot) else {
            debug_assert!(false, "a set record with no set behind it");
            return Ok(false);
        };
        // The same arithmetic the string side uses and not a tunable: a body
        // that costs less to keep than the pointer to it would is left where it
        // is. It is even more favourable here, because what a table costs in
        // memory is well above what its members weigh on a device.
        let grows_by = value::cold_record_len(expire_at.is_some())
            - value::slot_record_len(expire_at.is_some());
        if set.memory_bytes() <= grows_by {
            return Ok(false);
        }

        let mut buf = core::mem::take(&mut self.frozen);
        buf.clear();
        set.freeze(&mut buf);
        let tier = self.tier.as_mut().expect("checked at the top");
        let wrote = tier.stash(&buf);
        self.frozen = buf;
        let chain = wrote?;

        self.sets.remove(slot);
        self.bodies -= 1;
        let len = chain.len as u32;
        let wrote = self.map.set_with(
            key,
            value::cold_record_len(expire_at.is_some()),
            |_| {},
            |out| {
                // The encoding bits mean nothing on a collection record, which
                // is what `Meta::slot` already writes and says. `OBJECT
                // ENCODING` on a demoted set brings it back and asks the body,
                // which is one device read for a command nobody sends in a
                // loop, and is one place holding the fact rather than two.
                value::write_cold_record(
                    out,
                    Kind::Set,
                    value::Encoding::Int,
                    chain.at,
                    len,
                    expire_at,
                );
                value::set_access(out, was);
                value::has_expiry(out)
            },
        );
        debug_assert!(wrote.is_some(), "the key was found a moment ago");
        // Not cleared by hand. Writing the record moves the map's write counter
        // and that is what the memo checks, so the slot number it is holding for
        // this key is already unreachable.
        Ok(true)
    }

    /// Send a cold record that holds a body back to [`Keyspace::promote_body`],
    /// and say whether that is what happened.
    ///
    /// The guard in front of both fault entry points, and it is here rather than
    /// in the tier because the tier cannot do this job. [`Tier::fault`] puts a
    /// value back by writing a string record, so handing it a demoted set would
    /// turn the set into a string holding the bytes a set freezes to. The kind is
    /// in the record and only this side has a slab to put a body in.
    ///
    /// One probe of the map on a path that is about to read a device, on a
    /// database that has a file at all. Every other database is refused by the
    /// caller before it gets here.
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back.
    fn body_came_back(&mut self, key: &[u8]) -> Result<bool> {
        let Some(addr) = self.map.find(key) else {
            return Ok(false);
        };
        let m = value::Meta::from_byte(self.map.value_at(addr)[0]);
        if !m.is_cold() || !m.kind().is_body() {
            return Ok(false);
        }
        self.promote_body(key)?;
        Ok(true)
    }

    /// Bring `key`'s collection body back into a slab and answer its new slot.
    ///
    /// The record has to be rewritten either way, because a resident collection
    /// is a slot number and there is no way to hold one without being in the
    /// slab, so the doorkeeper does not get a vote. See [`Tier::fetch`].
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back, and
    /// [`Code::Corrupt`] when it reads back as something that is not a body.
    fn promote_body(&mut self, key: &[u8]) -> Result<Option<u32>> {
        let Some(addr) = self.map.find(key) else {
            return Ok(None);
        };
        let rec = self.map.value_at(addr);
        let Some(c) = value::cold(rec) else {
            return Ok(None);
        };
        let kind = value::kind(rec);
        let expire_at = value::expire_at(rec);
        let was = value::access(rec).unwrap_or_default();
        let Some(tier) = self.tier.as_mut() else {
            debug_assert!(false, "a cold record on a database with no file");
            return Ok(None);
        };

        let mut buf = core::mem::take(&mut self.frozen);
        let read = tier.fetch(
            crate::cold::Chain {
                at: c.at,
                len: u64::from(c.len),
            },
            &mut buf,
        );
        self.frozen = buf;
        read?;

        let slot = match kind {
            Kind::Set => {
                let set = Set::thaw(&self.frozen).map_err(|e| {
                    Error::new(Code::Corrupt, "a demoted set did not read back")
                        .with_detail(e.to_string())
                })?;
                self.sets.insert(set)
            }
            // Nothing else is written cold with a body yet, so arriving here
            // means a record that says one thing and a demoter that did another.
            _ => {
                return Err(Error::new(
                    Code::Corrupt,
                    "a demoted body of a type that is not moved out",
                )
                .with_detail(kind.name().to_string()));
            }
        };
        self.bodies += 1;
        let wrote = self.map.set_with(
            key,
            value::slot_record_len(expire_at.is_some()),
            |_| {},
            |out| {
                value::write_slot_record(out, kind, slot, expire_at);
                value::set_access(out, was);
                value::has_expiry(out)
            },
        );
        debug_assert!(wrote.is_some(), "the key was found a moment ago");
        Ok(Some(slot))
    }

    /// Read `key`'s value off the file into [`Keyspace::cold`], if that is where
    /// it is.
    ///
    /// The doorkeeper decides whether the value also goes back into memory, so
    /// after this the record under `key` is either resident or still cold with
    /// its bytes in the buffer, and [`Keyspace::value_of`] is what tells the two
    /// apart. The address the caller was holding is not valid afterwards on the
    /// promoting path, because promotion rewrites the record.
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back.
    pub(crate) fn warm(&mut self, key: &[u8]) -> Result<Faulted> {
        if self.tier.is_none() {
            return Ok(Faulted::Warm);
        }
        if self.body_came_back(key)? {
            return Ok(Faulted::Promoted);
        }
        let tier = self.tier.as_mut().expect("checked at the top");
        let mut buf = core::mem::take(&mut self.cold);
        let r = tier.fault(&mut self.map, key, &mut buf);
        self.cold = buf;
        if r == Ok(Faulted::Served) {
            self.cold_key.clear();
            self.cold_key.extend_from_slice(key);
        }
        r
    }

    /// Put `key`'s value back in memory if it was not there, for a command that
    /// is about to write it.
    ///
    /// See [`Tier::thaw`] for why the doorkeeper does not get a vote here. The
    /// short of it is that a read modify write leaves a resident record either
    /// way, so there is nothing for an answer to change.
    ///
    /// A caller can carry on exactly as it did before after this returns, with
    /// no cold case left to handle, which is why it is one line at the top of
    /// `APPEND` and `INCR` rather than a second path through them.
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back.
    pub(crate) fn thaw(&mut self, key: &[u8]) -> Result<()> {
        if self.tier.is_none() {
            return Ok(());
        }
        if self.body_came_back(key)? {
            return Ok(());
        }
        let tier = self.tier.as_mut().expect("checked at the top");
        let mut buf = core::mem::take(&mut self.cold);
        let r = tier.thaw(&mut self.map, key, &mut buf);
        self.cold = buf;
        r.map(|_| ())
    }

    /// The value in a record that may have been left on the file.
    ///
    /// Only correct straight after a [`Keyspace::warm`] of the same key, since
    /// the buffer holds one value at a time. A debug build says so rather than
    /// handing back a value that belongs to somebody else.
    pub(crate) fn value_of<'a>(&'a self, key: &[u8], rec: &'a [u8]) -> Str<'a> {
        if value::cold(rec).is_some() {
            debug_assert!(
                bytes_eq(&self.cold_key, key),
                "a read found a value on the file without faulting it in first"
            );
            Str::Bytes(&self.cold)
        } else {
            value::read(rec)
        }
    }

    /// [`Keyspace::warm`] and then the value, for the readers that do nothing
    /// else with the record.
    ///
    /// `None` when the key went away, which it cannot do here but which the
    /// second lookup has to allow for anyway: the first lookup's address does
    /// not survive a promotion, so this has to find the key again rather than
    /// trust a number from before.
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back.
    pub(crate) fn warmed(&mut self, key: &[u8]) -> Result<Option<Str<'_>>> {
        self.warm(key)?;
        let Some(addr) = self.map.find(key) else {
            return Ok(None);
        };
        Ok(Some(self.value_of(key, self.map.value_at(addr))))
    }

    /// Where `key`'s record is, having thrown the key away first if it is dead.
    ///
    /// The same fold as [`Keyspace::live_slot`] for a caller that wants the
    /// record itself rather than a slot number, which is every string command.
    /// `GET` used to be a reap, then a type check, then a read, and each of the
    /// three hashed the key and walked a bucket for the same record. It is one
    /// walk now and two arena reads, and an arena read at a known address is a
    /// load.
    ///
    /// The address dies at the next write, which is why this is `pub(crate)`
    /// and why every caller reads it and drops it inside one command.
    ///
    /// Finding a key counts as using it, so this stamps the access field on the
    /// way past under every policy that wants it stamped, which is eight of the
    /// ten. A command that has to look at a key without using it calls
    /// [`Keyspace::live_rec_untouched`] instead, and the list of those is short
    /// and is Redis's list rather than ours.
    pub(crate) fn live_rec(&mut self, key: &[u8]) -> Option<Addr> {
        let addr = self.live_rec_untouched(key)?;
        if self.policy.stamps_on_read() {
            self.stamp(addr);
        }
        Some(addr)
    }

    /// [`Keyspace::live_rec`] for a command that is asking about a key rather
    /// than using it.
    ///
    /// `TYPE`, `EXISTS`, the `TTL` family and every `OBJECT` subcommand look
    /// without touching, which is Redis's `LOOKUP_NOTOUCH` and is not an
    /// optimisation. `OBJECT IDLETIME` that counted as a use would report zero
    /// every time, and `EXISTS` in a health check loop would keep a dead key at
    /// the top of the working set forever.
    ///
    /// Untouched means the access field only. A key past its deadline is still
    /// reaped here, because a command asking whether a key exists has to be told
    /// that it does not.
    pub(crate) fn live_rec_untouched(&mut self, key: &[u8]) -> Option<Addr> {
        let now = self.clock.now_ms();
        let addr = self.map.find(key)?;
        if value::is_expired(self.map.value_at(addr), now) {
            self.drop_key(key);
            self.expired += 1;
            return None;
        }
        Some(addr)
    }

    /// The slot under `key`, having thrown the key away first if it is dead.
    ///
    /// `None` for a key that is not there or that was and is now reaped, and
    /// `WRONGTYPE` for a key holding something other than `want`.
    ///
    /// One probe of the map, where a [`Keyspace::reap`] followed by a `get`
    /// costs two. That pair is how every collection command used to start, so a
    /// pipeline of sixty four `SADD` on one key hashed and probed for that key a
    /// hundred and twenty eight times to do sixty four inserts. The reap has to
    /// read the record and the command has to read the same record, and there
    /// was never a reason for those to be two visits.
    ///
    /// It answers a number rather than the record it just read because of the
    /// borrow checker and not because a number is nicer. A method that hands
    /// back a borrow of the map on one path and takes a mutable borrow to reap
    /// on the other is the case the borrow checker still refuses without
    /// Polonius. A slot is four bytes and copies out, so the borrow ends here
    /// and the caller reaches its body through the slab.
    ///
    /// And no probe at all when the command in front of it asked for the same
    /// key and nothing has been written since, which is the [`Memo`] and is what
    /// Y13 asks for on single key `SADD`.
    pub(crate) fn live_slot(&mut self, key: &[u8], want: Kind) -> Result<Option<u32>> {
        // A memo hit skips the record, so the stamp has to happen on the way out
        // of it as well. It is the same address every time, which is what makes
        // this cheap: no probe, just the store. Getting this wrong is the trap
        // worth naming, because the key that hits the memo most often is the
        // hottest key in the database, and it is the one that would have looked
        // steadily more idle the harder it was used.
        if let Some((kind, slot, addr)) = self.memo.get(self.map.writes(), key) {
            if kind != want {
                return Err(wrong_type());
            }
            if self.policy.stamps_on_read() {
                self.stamp(addr);
            }
            return Ok(Some(slot));
        }
        // `find` and then `value_at` rather than `get`, which is the same two
        // steps, so that the address is still in hand for the stamp below. `get`
        // would mean probing a second time for a record already read.
        let now = self.clock.now_ms();
        let Some(addr) = self.map.find(key) else {
            return Ok(None);
        };
        let rec = self.map.value_at(addr);
        if value::is_expired(rec, now) {
            self.drop_key(key);
            self.expired += 1;
            return Ok(None);
        }
        if value::kind(rec) != want {
            return Err(wrong_type());
        }
        // One test of a bit in a byte that is already in a register, on the
        // funnel every collection command comes through. A database with no file
        // behind it never sets it and pays that test and nothing else.
        if value::Meta::from_byte(rec[0]).is_cold() {
            return self.promote_body(key);
        }
        let slot = value::slot(rec);
        // A key with a deadline is not memoized. The memo is invalidated by
        // writes and a deadline passes without one, so remembering a dated key
        // would be remembering it past the moment it should have been reaped.
        // Both of these are read off the record before the stamp, which needs it
        // mutably and is the end of this borrow.
        let dated = value::expire_at(rec).is_some();
        if self.policy.stamps_on_read() {
            self.stamp(addr);
        }
        if !dated {
            self.memo.put(self.map.writes(), key, want, slot, addr);
        }
        Ok(Some(slot))
    }

    /// Where `key` is, when either of two types will do.
    ///
    /// Every input to a sorted set operation may be a sorted set or a plain set,
    /// which is Redis's rule and means the type check there is a membership test
    /// rather than an equality. The kind comes back with the slot because the
    /// caller has to know which slab the number indexes.
    pub(crate) fn live_slot_either(
        &mut self,
        key: &[u8],
        a: Kind,
        b: Kind,
    ) -> Result<Option<(Kind, u32)>> {
        if let Some((kind, slot, addr)) = self.memo.get(self.map.writes(), key) {
            if kind != a && kind != b {
                return Err(wrong_type());
            }
            if self.policy.stamps_on_read() {
                self.stamp(addr);
            }
            return Ok(Some((kind, slot)));
        }
        let now = self.clock.now_ms();
        let Some(addr) = self.map.find(key) else {
            return Ok(None);
        };
        let rec = self.map.value_at(addr);
        if value::is_expired(rec, now) {
            self.drop_key(key);
            self.expired += 1;
            return Ok(None);
        }
        let kind = value::kind(rec);
        if kind != a && kind != b {
            return Err(wrong_type());
        }
        // As in `live_slot`, and the kind was read before the record was given
        // up because promotion rewrites it.
        if value::Meta::from_byte(rec[0]).is_cold() {
            return Ok(self.promote_body(key)?.map(|slot| (kind, slot)));
        }
        let slot = value::slot(rec);
        let dated = value::expire_at(rec).is_some();
        if self.policy.stamps_on_read() {
            self.stamp(addr);
        }
        if !dated {
            self.memo.put(self.map.writes(), key, kind, slot, addr);
        }
        Ok(Some((kind, slot)))
    }

    /// Throw every key away. This is `FLUSHDB` on one database.
    ///
    /// The expiry counter is not reset, because Redis does not reset it either:
    /// `expired_keys` in `INFO stats` counts what this process has expired since
    /// it started, and emptying a database is not expiring anything. The count of
    /// keys that carry a deadline is a different number and it does go to zero,
    /// because it is a fact about what is in the database right now and there is
    /// nothing in it.
    pub fn clear(&mut self) {
        self.map.clear();
        self.sets.clear();
        self.hashes.clear();
        self.lists.clear();
        self.zsets.clear();
        self.arrays.clear();
        self.streams.clear();
        self.foreign.clear();
        self.pool.clear();
        self.bodies = 0;
    }

    /// Keys reclaimed by running into them after their deadline.
    ///
    /// Redis calls this `expired_keys` in `INFO stats` and counts both lazy and
    /// active expiry into it, and so does this. [`Keyspace::expire_cycle`] is
    /// the active half and it counts into the same number, which is what makes
    /// this the total a dashboard can compare against a write rate rather than
    /// the share of it that happened to be reclaimed by a read.
    #[inline]
    pub const fn expired_keys(&self) -> u64 {
        self.expired
    }

    /// Keys thrown away to make room.
    ///
    /// Redis calls this `evicted_keys` in `INFO stats`. It stays at zero under
    /// `noeviction`, which is the whole point of that policy, and a monitoring
    /// dashboard that sees it move on a server configured that way is looking at
    /// a bug rather than at load.
    #[inline]
    pub const fn evicted_keys(&self) -> u64 {
        self.evicted
    }

    /// How many live keys carry a deadline.
    ///
    /// This is what `INFO keyspace` reports as `expires=`, and it is the live
    /// count rather than a running total: a key that gets a `TTL` and then has it
    /// taken away with `PERSIST` is in it and then is not.
    ///
    /// The map keeps it, because the map keeps the second index these keys are
    /// in. Nothing in this file counts it, which is deliberate: a count kept
    /// alongside the thing it counts is a count that eventually disagrees with
    /// it, and the one place that can be wrong should be the one place that owns
    /// the entries.
    #[inline]
    pub fn expires(&self) -> usize {
        self.map.tagged_len()
    }

    /// How many keys a round of eviction sampling looks at.
    #[inline]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Set how many keys a round of eviction sampling looks at.
    ///
    /// Zero is not refused here, because the caller doing the refusing is
    /// `CONFIG SET` and it has a message to produce. A zero that reaches here
    /// samples one bucket and takes the best of it, because the loop runs its
    /// body before it checks, which is a better answer than dividing by nothing.
    #[inline]
    pub const fn set_samples(&mut self, samples: usize) {
        self.samples = samples;
    }

    /// Throw away one key, chosen by the policy. Answers whether one went.
    ///
    /// This is one step and not a loop on purpose. The caller is the thing that
    /// knows how much room it needs back, and a loop in here would either take
    /// too much or have to be told the same number twice. It also means the
    /// caller can put a bound on how long it spends evicting before it answers
    /// the client, which matters because the client is waiting on a write that
    /// this is making room for.
    ///
    /// It answers false without doing anything under `noeviction`, and also when
    /// a `volatile` policy is set on a database where nothing has a deadline.
    /// Those are the same answer to the caller and they mean the same thing: this
    /// server cannot give memory back and is about to have to refuse a write.
    pub fn evict_one(&mut self) -> bool {
        let Some(addr) = self.victim() else {
            return false;
        };
        // The key has to outlive the borrow that found it, because deleting is a
        // write and the address came out of a read. One copy into the scratch
        // buffer rather than a `Vec` per eviction, for the reason written on
        // [`Keyspace::scratch`]: this runs in a loop when it runs at all.
        let mut buf = core::mem::take(&mut self.scratch);
        buf.clear();
        buf.extend_from_slice(self.map.entry_at(addr).0);
        let gone = self.drop_key(&buf);
        self.scratch = buf;
        if gone {
            self.evicted += 1;
        }
        gone
    }

    /// Where the key this policy would throw away lives, if there is one.
    ///
    /// The sampling loop. It draws buckets until it has looked at `samples` keys
    /// the policy would consider, scores each one, and hands back the best. See
    /// [`evict`] for what the score means and [`yo_index::RawMap::sample`] for
    /// why a bucket is the unit.
    ///
    /// The round cap is the part that is not obvious. A database with a hundred
    /// keys in a directory sized for a million is mostly empty buckets, and a
    /// `volatile` policy on a database where nothing has a deadline has no
    /// eligible keys at all however many buckets it looks in. Without the cap the
    /// second case is an infinite loop, and it is not a rare configuration, it is
    /// the classic eviction surprise. With it, the worst case is a fixed number
    /// of cache misses and a false, which is exactly what the caller needs to
    /// hear.
    ///
    /// A key past its deadline is skipped rather than taken. It is dead memory
    /// and evicting it would look like a win, but it would be counted as an
    /// eviction when it is an expiry, and those two numbers are watched
    /// separately for a reason. Lazy expiry takes it the next time anything asks
    /// for it, and the active cycle takes it before that.
    ///
    /// What comes back is not only the worst of this round. Everything sampled
    /// goes into [`evict::Pool`], which holds the sixteen best across rounds, so
    /// the answer is the worst key seen since the pool was last emptied. The
    /// price is that a candidate is a key rather than an address and so has to
    /// be looked up and rechecked here, because it can have been deleted or have
    /// expired or have lost its deadline since the round that spotted it.
    fn victim(&mut self) -> Option<Addr> {
        if matches!(self.policy, Policy::NoEviction) || self.map.is_empty() {
            self.pool.clear();
            return None;
        }
        // The classic eviction surprise, answered before it costs anything. A
        // `volatile` policy on a database where no key has a deadline has no
        // eligible key anywhere, and the loop below can only find that out by
        // drawing four rounds of buckets and being told so by every key in them,
        // on a path where a client is waiting for the write this is making room
        // for. The count knows.
        if self.policy.volatile_only() && self.expires() == 0 {
            self.pool.clear();
            return None;
        }
        let now = self.clock.now_ms();
        let (policy, lfu, want) = (self.policy, self.lfu, self.samples);
        if policy.is_random() {
            return self.draw(now, want);
        }
        // Which index to draw from. The `volatile` policies can only take a key
        // that has a deadline, and the map keeps a second index of exactly
        // those, so drawing from the whole keyspace and then throwing most of it
        // away is work with a cheaper alternative sitting right there. The
        // `allkeys` policies draw from everything, because everything is
        // eligible.
        let volatile = policy.volatile_only();
        let mut seen = 0usize;
        for _ in 0..ROUNDS {
            let r = self.rng.next_u64();
            let pool = &mut self.pool;
            // By reference, so the same closure can go to either sampler. A
            // `&mut F` is an `FnMut` when `F` is, which is what makes the two
            // calls below one closure rather than two copies of it.
            let mut offer = |key: &[u8], rec: &[u8], _addr: Addr| {
                if !value::is_expired(rec, now) && evict::eligible(rec, policy) {
                    seen += 1;
                    pool.offer(key, evict::score(rec, policy, now, lfu));
                }
                seen < want
            };
            if volatile {
                self.map.sample_tagged(r, &mut offer);
            } else {
                self.map.sample(r, &mut offer);
            }
            if seen >= want {
                break;
            }
        }
        while let Some(key) = self.pool.take() {
            let Some(addr) = self.map.find(key) else {
                continue;
            };
            let rec = self.map.value_at(addr);
            if value::is_expired(rec, now) || !evict::eligible(rec, policy) {
                continue;
            }
            return Some(addr);
        }
        None
    }

    /// A fair draw among the eligible keys, which is what the random pair want.
    ///
    /// No pool, because there is no ordering for one to approximate: under
    /// `allkeys-random` and `volatile-random` every eligible key is as good a
    /// victim as every other, and remembering sixteen of them across rounds
    /// would only mean the same sixteen going first. The sampling is what does
    /// the choosing, so the address it lands on is used straight away and the
    /// key never has to be copied at all.
    fn draw(&mut self, now: u64, want: usize) -> Option<Addr> {
        let policy = self.policy;
        let volatile = policy.volatile_only();
        let mut best = evict::Best::EMPTY;
        let mut seen = 0usize;
        for _ in 0..ROUNDS {
            let r = self.rng.next_u64();
            let mut offer = |_key: &[u8], rec: &[u8], addr: Addr| {
                if !value::is_expired(rec, now) && evict::eligible(rec, policy) {
                    seen += 1;
                    best.offer(addr, evict::ANY);
                }
                seen < want
            };
            if volatile {
                self.map.sample_tagged(r, &mut offer);
            } else {
                self.map.sample(r, &mut offer);
            }
            if seen >= want {
                break;
            }
        }
        (!best.is_empty()).then_some(best.addr)
    }

    /// Bytes held by the index, the arena and every body hanging off them.
    ///
    /// Asks every collection, so this is O(the number of collections) and is for
    /// the places that want the number exactly and are asked for it rarely:
    /// `INFO memory`, `MEMORY USAGE` and the tests.
    /// [`Keyspace::settled_memory_bytes`] is the one a memory limit uses.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.slab_bytes()
            + self.sets.value_bytes()
            + self.hashes.value_bytes()
            + self.lists.value_bytes()
            + self.zsets.value_bytes()
            + self.arrays.value_bytes()
            + self.streams.value_bytes()
            + self.foreign.value_bytes()
    }

    /// The same number, asked only of the collections that could have moved.
    ///
    /// See [`Slab::track_bytes`] for how that is known. With tracking on this
    /// costs what the batch touched instead of what the database holds, which is
    /// what lets a server with a `maxmemory` ask once a batch. With tracking off
    /// it is [`Keyspace::memory_bytes`] and the two cannot disagree, because
    /// they are the same sum over the same values either way.
    #[inline]
    pub fn settled_memory_bytes(&mut self) -> usize {
        self.slab_bytes()
            + self.sets.settled_bytes()
            + self.hashes.settled_bytes()
            + self.lists.settled_bytes()
            + self.zsets.settled_bytes()
            + self.arrays.settled_bytes()
            + self.streams.settled_bytes()
            + self.foreign.settled_bytes()
    }

    /// Start or stop keeping the running total in every slab.
    ///
    /// One call for all seven, because a limit is a property of the server and
    /// not of a type, and a database tracking its sets but not its hashes would
    /// answer a number that is neither of the two things it could mean.
    pub fn track_memory(&mut self, on: bool) {
        self.sets.track_bytes(on);
        self.hashes.track_bytes(on);
        self.lists.track_bytes(on);
        self.zsets.track_bytes(on);
        self.arrays.track_bytes(on);
        self.streams.track_bytes(on);
        self.foreign.track_bytes(on);
    }

    /// The index, the arena and the slot arrays, none of which need asking
    /// twice, plus the tier when there is one.
    ///
    /// The tier is in here because a doorkeeper and a directory buffer are real
    /// memory and a limit that did not count them would be a limit on part of
    /// the server. It is also the honest way round: the thing that gives memory
    /// back costs some to keep, and both numbers belong in the same total.
    #[inline]
    fn slab_bytes(&self) -> usize {
        self.tier.as_ref().map_or(0, Tier::memory_bytes)
            + self.map.memory_bytes()
            + self.sets.slot_bytes()
            + self.hashes.slot_bytes()
            + self.lists.slot_bytes()
            + self.zsets.slot_bytes()
            + self.arrays.slot_bytes()
            + self.streams.slot_bytes()
            + self.foreign.slot_bytes()
    }

    /// Give back one segment's worth of space if one has gone mostly dead.
    ///
    /// Overwriting a key does not reuse its bytes, it writes the new record at
    /// the bump pointer and counts the old one as dead, so a workload that sets
    /// the same keys over and over holds far more than it is storing until
    /// something compacts. This is that something, and it does at most one
    /// segment per call so that the loop can afford to ask every turn.
    #[inline]
    pub fn compact_step(&mut self) -> Option<usize> {
        self.map.compact_step()
    }

    /// The same, for a store that is over a memory limit and has to give pages
    /// back rather than wait for a segment to be worth collecting.
    ///
    /// See [`RawMap::compact_hard`] for why the choice of segment changes and
    /// why it only changes under pressure.
    #[inline]
    pub fn compact_hard(&mut self) -> Option<usize> {
        self.map.compact_hard()
    }

    /// Ask the cache for the bucket this key will land in.
    ///
    /// The first of the loop's two walks (`04` section 3) calls this.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        self.map.prefetch(hash);
    }

    /// The hash this database files `key` under.
    #[inline]
    #[must_use]
    pub fn hash_of(key: &[u8]) -> u64 {
        RawMap::hash_of(key)
    }
}

/// What Redis says when a command is sent at a key holding another type.
///
/// The text is Redis's, word for word, because it goes on the wire verbatim and
/// clients match on it. The `WRONGTYPE` at the front is not part of the message:
/// the protocol layer puts it there from the [`Code`], which is what lets an
/// embedded caller match on a value instead of on a string (P5).
pub fn wrong_type() -> Error {
    Error::new(
        Code::WrongType,
        "Operation against a key holding the wrong kind of value",
    )
}

impl Default for Keyspace {
    fn default() -> Keyspace {
        Keyspace::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    #[test]
    fn type_answers_string_for_a_string_and_nothing_for_a_missing_key() {
        let mut d = db();
        d.set_plain(b"k", b"v").expect("room");
        assert_eq!(d.kind_of(b"k"), Some(Kind::String));
        assert_eq!(d.kind_of(b"nope"), None);
    }

    #[test]
    fn type_does_not_report_a_key_whose_deadline_has_gone() {
        let mut d = db();
        d.psetex(b"k", 100, b"v").expect("room");
        assert_eq!(d.kind_of(b"k"), Some(Kind::String));

        d.clock_mut().advance(100);
        assert_eq!(
            d.kind_of(b"k"),
            None,
            "the deadline was 1100 and it is 1100"
        );
        assert_eq!(d.len(), 0, "and asking reaped it rather than leaving it");
        assert_eq!(d.expired_keys(), 1);
    }

    /// The default policy evicts nothing and still keeps the clock, which is
    /// Redis's behaviour and is the configuration nearly every server runs.
    #[test]
    fn the_clock_runs_under_the_default_policy() {
        let mut d = db();
        assert_eq!(d.policy(), Policy::NoEviction);
        d.set_plain(b"k", b"v").expect("room");
        assert_eq!(d.idle_secs(b"k"), Some(0));

        d.clock_mut().advance(60_000);
        assert_eq!(d.idle_secs(b"k"), Some(60), "a minute of nobody asking");

        d.get(b"k").expect("a string").expect("still there");
        assert_eq!(d.idle_secs(b"k"), Some(0), "and reading it is using it");
    }

    /// The commands that ask about a key rather than use it. Getting this wrong
    /// makes `OBJECT IDLETIME` answer zero every time it is called, because
    /// calling it would be the most recent use.
    #[test]
    fn asking_about_a_key_is_not_using_it() {
        let mut d = db();
        d.set_plain(b"k", b"v").expect("room");
        d.clock_mut().advance(30_000);

        assert!(d.exists(b"k"));
        assert_eq!(d.kind_of(b"k"), Some(Kind::String));
        assert_eq!(d.encoding_name(b"k"), Some("embstr"));
        assert_eq!(d.deadline_of(b"k"), Ask::NoDeadline);
        assert_eq!(d.expire_at(b"k"), None);
        assert_eq!(d.idle_secs(b"k"), Some(30));

        assert_eq!(
            d.idle_secs(b"k"),
            Some(30),
            "and asking twice is still not using it"
        );
    }

    /// Least recently modified is the one policy where a read must leave the
    /// clock where it is, because the clock is the only thing it measures.
    #[test]
    fn a_read_moves_the_clock_under_lru_and_leaves_it_under_lrm() {
        for (policy, idle_after_read) in [(Policy::AllKeysLru, 0), (Policy::AllKeysLrm, 45)] {
            let mut d = db();
            d.set_policy(policy);
            d.set_plain(b"k", b"v").expect("room");
            d.clock_mut().advance(45_000);

            d.get(b"k").expect("a string").expect("still there");
            assert_eq!(
                d.idle_secs(b"k"),
                Some(idle_after_read),
                "{}",
                policy.name()
            );

            // Both of them move it on a write, which is the whole of what LRM
            // is measuring and is a side effect of the resolve under LRU.
            d.set_plain(b"k", b"w").expect("room");
            assert_eq!(
                d.idle_secs(b"k"),
                Some(0),
                "{} after a write",
                policy.name()
            );
        }
    }

    /// The trap the memo sets. A hit skips the record entirely, so a stamp that
    /// only happened on a miss would leave the hottest key in the database
    /// looking steadily more idle the harder it was used.
    #[test]
    fn the_hot_key_path_still_stamps() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        d.sadd(b"s", [&b"a"[..]].into_iter()).expect("room");

        // Warm the memo, then run the key hard with nothing written in between,
        // which is the case the memo exists for.
        d.scard(b"s").expect("a set");
        d.clock_mut().advance(120_000);
        for _ in 0..64 {
            d.scard(b"s").expect("a set");
        }
        assert_eq!(d.idle_secs(b"s"), Some(0), "the memo swallowed the stamp");
    }

    /// Under LFU the same bits are a counter, and it climbs with use rather than
    /// resetting to now.
    #[test]
    fn the_counter_climbs_under_an_lfu_policy() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLfu);
        d.seed(7);
        d.set_plain(b"k", b"v").expect("room");
        let start = d.freq(b"k").expect("there");

        for _ in 0..200 {
            d.get(b"k").expect("a string").expect("still there");
        }
        let hot = d.freq(b"k").expect("there");
        assert!(hot > start, "{hot} did not climb from {start}");

        // And a key nobody reads decays rather than holding its place forever.
        d.set_plain(b"cold", b"v").expect("room");
        d.clock_mut().advance(60_000 * 10);
        assert!(d.freq(b"cold").expect("there") < start);
    }

    /// Nothing goes under `noeviction`, which is the only promise that policy
    /// makes and the reason it is the default.
    #[test]
    fn noeviction_evicts_nothing() {
        let mut d = db();
        for i in 0..200u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        assert!(!d.evict_one());
        assert_eq!(d.len(), 200);
        assert_eq!(d.evicted_keys(), 0);
    }

    /// A volatile policy on a database where nothing has a deadline is the
    /// classic surprise: it looks configured and it cannot free a byte.
    #[test]
    fn a_volatile_policy_with_no_deadlines_anywhere_cannot_evict() {
        let mut d = db();
        d.set_policy(Policy::VolatileLru);
        for i in 0..200u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        assert!(!d.evict_one(), "it found a key it had no business taking");
        assert_eq!(d.len(), 200);

        // Give one key a deadline and it becomes the only thing that can go,
        // however many rounds of sampling that takes.
        let deadline = d.clock().now_ms() + 100_000;
        d.set_expiry(b"k7", Some(deadline));
        assert!(d.evict_one());
        assert!(!d.exists(b"k7"));
        assert_eq!(d.evicted_keys(), 1);
    }

    /// The count against a walk, over everything that can move it. If these two
    /// ever disagree the count is worse than useless, because `INFO` would be
    /// reporting a number that looks like a measurement.
    #[test]
    fn the_deadline_count_says_what_a_walk_of_the_keyspace_says() {
        let mut d = db();
        let now = d.clock().now_ms();
        let check = |d: &mut Keyspace, note: &str| {
            let mut names = Vec::new();
            d.keys(|k| names.push(k.to_vec()));
            let walked = names
                .iter()
                .filter(|k| matches!(d.deadline_of(k), Ask::At(_)))
                .count();
            assert_eq!(d.expires(), walked, "{note}");
        };

        for i in 0..40u32 {
            d.set_plain(format!("s{i}").as_bytes(), b"v").expect("room");
            d.sadd(format!("c{i}").as_bytes(), [b"m".as_slice()].into_iter())
                .expect("room");
        }
        check(&mut d, "nothing has a deadline yet");
        assert_eq!(d.expires(), 0);

        // On, on again with a different deadline, and off.
        for i in (0..40u32).step_by(2) {
            d.set_expiry(format!("s{i}").as_bytes(), Some(now + 500_000));
            d.set_expiry(format!("c{i}").as_bytes(), Some(now + 500_000));
        }
        check(&mut d, "half of each type has one");
        assert_eq!(d.expires(), 40);
        for i in (0..40u32).step_by(4) {
            d.set_expiry(format!("s{i}").as_bytes(), Some(now + 900_000));
        }
        check(&mut d, "moving a deadline is not gaining one");
        assert_eq!(d.expires(), 40);
        for i in (0..40u32).step_by(4) {
            d.set_expiry(format!("c{i}").as_bytes(), None);
        }
        check(&mut d, "and PERSIST gives them back");
        assert_eq!(d.expires(), 30);

        // Written over, which is the path where the record loses its deadline
        // without anybody saying so.
        d.set_plain(b"s2", b"fresh").expect("room");
        check(&mut d, "a plain SET drops the deadline it wrote over");

        // Renamed, deleted, expired and evicted.
        d.rename(b"s6", b"s6new", false);
        check(&mut d, "a rename moved one rather than losing it");
        d.drop_key(b"s6new");
        d.drop_key(b"c2");
        check(&mut d, "two deleted");
        d.psetex(b"gone", 50, b"v").expect("room");
        check(&mut d, "and one more with a short deadline");
        d.clock_mut().advance(60);
        assert_eq!(d.kind_of(b"gone"), None, "which the read reaped");
        check(&mut d, "so the count lost it too");
        d.set_policy(Policy::VolatileRandom);
        assert!(d.evict_one());
        check(&mut d, "eviction under a volatile policy takes one of them");

        d.clear();
        assert_eq!(d.expires(), 0, "and FLUSHDB takes the lot");
    }

    /// The point of the count on the eviction path. A volatile policy with
    /// nothing to evict answers on the comparison rather than on four rounds of
    /// buckets, and it has to still answer `false`.
    #[test]
    fn a_volatile_policy_asks_the_count_before_it_samples() {
        let mut d = db();
        d.set_policy(Policy::VolatileLfu);
        for i in 0..500u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        assert_eq!(d.expires(), 0);
        assert!(!d.evict_one(), "nothing is eligible and nothing went");
        assert_eq!(d.len(), 500);

        // And the fast path gets out of the way the moment one key qualifies.
        d.set_expiry(b"k123", Some(d.clock().now_ms() + 100_000));
        assert_eq!(d.expires(), 1);
        assert!(d.evict_one());
        assert!(!d.exists(b"k123"));
        assert_eq!(d.expires(), 0, "and the count went with it");
    }

    /// A needle in a haystack, which is the case sampling the whole keyspace
    /// could not do.
    ///
    /// Fifty thousand keys and three of them with a deadline. Under a volatile
    /// policy those three are the only ones that may go, and four rounds of
    /// sixty four buckets drawn from the whole index would land on one of them
    /// about once in a hundred tries. Drawn from the index of just the keys that
    /// carry a deadline it is the only thing there is to land on.
    ///
    /// Three of them and not one, so that the test is about finding an eligible
    /// key rather than about a table with a single entry in it.
    #[test]
    fn a_volatile_policy_finds_the_one_key_in_a_database_that_is_not_volatile() {
        let mut d = db();
        d.set_policy(Policy::VolatileLru);
        for i in 0..50_000u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        let deadline = d.clock().now_ms() + 100_000;
        for k in [b"k7".as_slice(), b"k30000", b"k49999"] {
            assert!(d.set_expiry(k, Some(deadline)));
        }
        assert_eq!(d.expires(), 3);

        for round in 0..3 {
            assert!(d.evict_one(), "round {round} found nothing to take");
        }
        assert_eq!(d.expires(), 0, "all three went");
        assert_eq!(d.len(), 50_000 - 3, "and nothing else did");
        assert!(!d.evict_one(), "and now there is nothing eligible left");
        assert_eq!(d.len(), 50_000 - 3);
    }

    /// The direction of the score, which is the thing worth pinning. A test that
    /// only checked something was evicted would pass just as happily on a cache
    /// that keeps the cold keys and throws away the hot ones.
    #[test]
    fn the_stale_key_goes_before_the_fresh_one() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        // Two keys is a small enough database that a bucket holds both of them
        // and the pick is between them rather than between whatever turned up.
        d.set_plain(b"cold", b"v").expect("room");
        d.clock_mut().advance(600_000);
        d.set_plain(b"hot", b"v").expect("room");

        assert!(d.evict_one());
        assert!(!d.exists(b"cold"), "it kept the stale one");
        assert!(d.exists(b"hot"), "it took the fresh one");
    }

    /// Under `volatile-ttl` the ordering is by deadline and not by use, so the
    /// key about to expire anyway is the one that goes.
    #[test]
    fn the_soonest_deadline_goes_first() {
        let mut d = db();
        d.set_policy(Policy::VolatileTtl);
        let now = d.clock().now_ms();
        d.set_plain(b"soon", b"v").expect("room");
        d.set_plain(b"later", b"v").expect("room");
        d.set_expiry(b"soon", Some(now + 10_000));
        d.set_expiry(b"later", Some(now + 900_000));

        assert!(d.evict_one());
        assert!(!d.exists(b"soon"));
        assert!(d.exists(b"later"));
    }

    /// Under LFU the key nobody reads goes, even though it was written more
    /// recently than the one that survives. That is the difference between the
    /// two families and it is invisible to a test written against the clock.
    #[test]
    fn the_least_used_key_goes_under_lfu() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLfu);
        d.seed(11);
        d.set_plain(b"popular", b"v").expect("room");
        for _ in 0..300 {
            d.get(b"popular").expect("a string").expect("still there");
        }
        // Written after the reads above, so under any clock policy this would be
        // the freshest key in the database and the last thing to go.
        d.set_plain(b"ignored", b"v").expect("room");

        assert!(d.evict_one());
        assert!(!d.exists(b"ignored"));
        assert!(d.exists(b"popular"));
    }

    /// Sampling has to keep working when almost every bucket it looks in is
    /// empty, which is what a database looks like after most of it is deleted.
    #[test]
    fn a_nearly_empty_database_still_gives_up_a_key() {
        let mut d = db();
        d.set_policy(Policy::AllKeysRandom);
        for i in 0..4000u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }
        for i in 0..3999u32 {
            d.drop_key(format!("k{i}").as_bytes());
        }
        assert_eq!(d.len(), 1);

        // One key in a directory sized for four thousand. It may take more than
        // one round to land on it, and it may take more than one call, but the
        // rounds are bounded and so is this loop.
        let mut went = false;
        for _ in 0..500 {
            if d.evict_one() {
                went = true;
                break;
            }
        }
        assert!(went, "sampling never found the one key that was left");
        assert_eq!(d.len(), 0);
        assert!(!d.evict_one(), "and an empty database has nothing to give");
    }

    /// Eviction and expiry are counted apart, so a key that was already dead
    /// when sampling found it is not billed as an eviction.
    #[test]
    fn a_dead_key_is_not_evicted() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        let now = d.clock().now_ms();
        d.set_plain(b"k", b"v").expect("room");
        d.set_expiry(b"k", Some(now + 1000));
        d.clock_mut().advance(5000);

        assert!(!d.evict_one(), "it evicted a key that was already dead");
        assert_eq!(d.evicted_keys(), 0);
    }

    /// The point of the pool. A round looks at five keys, takes one, and used to
    /// throw the other four away, so the second worst key in the database had to
    /// be found again from scratch every time.
    #[test]
    fn a_candidate_that_was_not_taken_is_still_in_the_running() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        for i in 0..40u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
            d.clock_mut().advance(1000);
        }
        assert!(d.pool.is_empty(), "nothing has sampled anything yet");

        assert!(d.evict_one());
        assert!(
            !d.pool.is_empty(),
            "every key it looked at and did not take was thrown away"
        );
    }

    /// A candidate is a key and not an address, so it can stop being a key
    /// between the round that spotted it and the round that wants it. Every one
    /// of them going at once is the worst case, and the answer has to be the
    /// live key rather than a shrug.
    #[test]
    fn a_candidate_that_went_away_is_stepped_over() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        for i in 0..40u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
            d.clock_mut().advance(1000);
        }
        assert!(d.evict_one());
        assert!(!d.pool.is_empty());

        // By hand, so every candidate still held names a key that is not there.
        for i in 0..40u32 {
            d.drop_key(format!("k{i}").as_bytes());
        }
        assert_eq!(d.len(), 0);
        d.set_plain(b"fresh", b"v").expect("room");

        assert!(d.evict_one(), "it gave up on a database with a key in it");
        assert!(!d.exists(b"fresh"));
        assert!(d.pool.is_empty(), "and the stale ones went with it");
    }

    /// A score only means something against another score under the same rule,
    /// so a pool full of them is worth nothing the moment the rule changes.
    #[test]
    fn changing_the_policy_throws_the_candidates_away() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        let now = d.clock().now_ms();
        for i in 0..40u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
            d.set_expiry(
                format!("k{i}").as_bytes(),
                Some(now + 100_000 + u64::from(i)),
            );
            d.clock_mut().advance(1000);
        }
        assert!(d.evict_one());
        assert!(!d.pool.is_empty());

        d.set_policy(Policy::VolatileTtl);
        assert!(d.pool.is_empty(), "idle seconds against a countdown");

        // And the same policy set again is not a change and costs nothing.
        d.set_policy(Policy::VolatileTtl);
        assert!(d.evict_one());
        assert!(!d.pool.is_empty());
        d.set_policy(Policy::VolatileTtl);
        assert!(!d.pool.is_empty());
    }

    /// A fair draw has no ordering for a pool to get closer to, so the random
    /// pair never copy a key at all.
    #[test]
    fn a_random_policy_keeps_no_candidates() {
        let mut d = db();
        d.set_policy(Policy::AllKeysRandom);
        for i in 0..40u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
        }

        assert!(d.evict_one());
        assert_eq!(d.len(), 39);
        assert!(d.pool.is_empty());
        assert_eq!(
            d.pool.memory_bytes(),
            0,
            "and it allocated nothing to do it"
        );
    }

    /// A flush leaves the pool naming keys that are all gone, which the recheck
    /// would survive and would pay sixteen lookups for.
    #[test]
    fn a_flush_takes_the_candidates_with_it() {
        let mut d = db();
        d.set_policy(Policy::AllKeysLru);
        for i in 0..40u32 {
            d.set_plain(format!("k{i}").as_bytes(), b"v").expect("room");
            d.clock_mut().advance(1000);
        }
        assert!(d.evict_one());
        assert!(!d.pool.is_empty());

        d.clear();
        assert!(d.pool.is_empty());
    }
}
