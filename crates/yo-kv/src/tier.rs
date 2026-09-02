//! Moving a value out to the file and getting it back, which is WiscKey's idea
//! with the tag from `06` section 6 doing the bookkeeping.
//!
//! Three pieces already exist and this is what joins them. [`cold`]
//! knows how to lay a value out on the file. [`value`] knows how
//! to write a record that points at one and how to tell in one bit whether a
//! record does. [`demote`](crate::demote) knows how to choose. What was missing
//! is the thing that reads a record, writes its bytes out, replaces it with a
//! twelve byte pointer, and does the reverse on the way back.
//!
//! # What separation buys, exactly
//!
//! A resident string record is one meta byte, three access bytes, eight more if
//! it has a deadline, and then the value. A demoted one is the same head with
//! twelve bytes of address and length instead of the value. So demotion pays
//! from thirteen payload bytes upward and costs memory below that, which is why
//! [`worth_demoting`] is arithmetic on the two lengths and not a tunable. There
//! is no threshold to get wrong.
//!
//! What is kept in memory is chosen the same way: the deadline, the access
//! field, the kind and the encoding all stay, so `TTL`, `TYPE`, `OBJECT
//! ENCODING`, `STRLEN`, `EXISTS` and every eviction policy still answer at
//! memory speed on a key whose bytes are on the device. G9's budget of 1.05
//! device reads per point read is spent on reads that actually want bytes.
//!
//! # The doorkeeper, and why a fault is not a promotion
//!
//! Reading a demoted value does not bring it back. The first read of a key sets
//! its bits in the doorkeeper and serves from the file; a second read while
//! those bits are still there brings it into memory. So a scan over cold data
//! displaces nothing, and a key that is genuinely warming up pays one extra
//! device read to prove it. That is the TinyLFU admission argument and it is the
//! difference between a tier and a cache that thrashes.
//!
//! # What this does not do yet
//!
//! Only strings, and only ones that are not int encoded. A collection keeps its
//! body in a slab and its record holds a slab index, so demoting one means
//! moving the body and not the record, which is the chunked band's other half
//! and a separate piece of work.
//!
//! Victims are chosen by sampling, through the same [`evict::Pool`] eviction
//! uses, rather than by the S3-FIFO and SIEVE queues in [`demote`](crate::demote).
//! Those queues want a slot number per entry that is stable across an arena
//! compaction, and this crate does not have one to give them: an address moves
//! when a segment is evacuated and a key is the thing being looked up. Deciding
//! where that number lives is a record layout question and it is the next one
//! this milestone has to answer. Sampling is what eviction and the expire cycle
//! already do, it needs nothing new, and it is a floor rather than a ceiling.
//!
//! # Space on the file
//!
//! Promoting a value leaves its chunks where they are. There is no delete on
//! [`Blocks`] and there does not need to be one, because a chunk nobody points
//! at is exactly what the log's compaction already collects, and the same is
//! true of the chunks a crash leaves behind between the last chunk write and the
//! directory write.

use yo_common::{Result, Rng};
use yo_index::RawMap;

use crate::access::{Lfu, Policy};
use crate::cold::{self, Blocks};
use crate::demote::Doorkeeper;
use crate::evict;
use crate::value::{self, Encoding, Kind};

/// How many keys the doorkeeper remembers before it clears itself.
///
/// Large enough that a read and the read that follows it a few thousand keys
/// later still count as the same window, small enough that the filter does not
/// saturate and start admitting everything. Both failure modes are the same
/// failure, which is a doorkeeper that has stopped saying no.
pub const WINDOW: usize = 8192;

/// How many entries one round of sampling walks past before it gives up on
/// finding its sixteen victims in this part of the keyspace.
///
/// Eviction does not need a number like this, because every entry it looks at
/// is a candidate and sixteen entries is sixteen candidates. Demotion is not
/// like that. A record that is already cold is skipped, and in a keyspace that
/// is mostly cold, which is exactly the state a sweep spends most of its time
/// in, nearly every entry a round walks is one it has to skip. Counting those
/// against the round's budget makes the sweep stall with the last few percent
/// of the keyspace still in memory, sitting a few buckets further along than
/// the round was allowed to look.
///
/// So the budget counts victims found and this counts entries walked, purely so
/// that a round over a segment holding nothing demotable still ends. It is
/// larger than a segment on purpose: a barren round then means the segment it
/// drew is genuinely clean, which is the thing [`BARREN`] wants to know.
pub const WALK: usize = 1024;

/// How many rounds of sampling have to come back with nothing before
/// [`Tier::relieve`] accepts that there is nothing left to move.
///
/// A round covers the whole of one index segment, so a barren round is a
/// segment with nothing left in it worth moving. Sixteen of those in a row,
/// against segments drawn at random, is a keyspace that is done.
///
/// It has to be a run and not a single round because sampling picks its segment
/// and its starting bucket out of one random draw, so two rounds that draw the
/// same pair walk the same entries and the second one finds every one of them
/// already moved. Stopping on the first barren round quit with ninety four
/// percent of the keyspace still in memory.
pub const BARREN: usize = 16;

/// What happened to a read of a key that may not have been in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Faulted {
    /// No such key. Nothing was read and nothing was written.
    Missing,
    /// The value was in memory all along, so the output buffer was not touched
    /// and the caller should read the record the way it always does.
    Warm,
    /// Read from the file and deliberately left there, because one read is not
    /// enough to earn a slot in memory back.
    Served,
    /// Read from the file and brought back into memory, so the next read of
    /// this key does not touch the device.
    Promoted,
}

/// The running totals, for `INFO` and for the gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Values moved out to the file.
    pub demoted: u64,
    /// Values brought back into memory.
    pub promoted: u64,
    /// Reads that went to the device, whether or not they promoted. This over
    /// the number of point reads is the ratio G9 is a gate on.
    pub faults: u64,
    /// Reads that went to the device and left the value there.
    pub served: u64,
    /// Payload bytes written to the file.
    pub bytes_out: u64,
    /// Payload bytes read back from it.
    pub bytes_in: u64,
}

/// Whether moving this record's value to the file would save memory.
///
/// Straight comparison of the two record lengths. A record whose value is short
/// enough that the pointer costs more than the bytes is left alone, and that is
/// the whole of the size policy.
#[must_use]
pub fn worth_demoting(rec: &[u8]) -> bool {
    let m = value::Meta::from_byte(rec[0]);
    if m.is_cold() || m.kind() != Kind::String || m.encoding() == Encoding::Int {
        return false;
    }
    rec.len() > value::cold_record_len(m.has_expiry())
}

/// The tier, which owns the file side of the keyspace.
pub struct Tier<B: Blocks> {
    blocks: B,
    door: Doorkeeper,
    scratch: cold::Scratch,
    pool: evict::Pool,
    /// The key of the victim being worked on, so that taking it out of the pool
    /// does not hold a borrow across the demotion.
    keybuf: Vec<u8>,
    rng: Rng,
    stats: Stats,
}

impl<B: Blocks> Tier<B> {
    /// A tier over `blocks`, with a doorkeeper of the default window.
    pub fn new(blocks: B) -> Tier<B> {
        Tier::with_window(blocks, WINDOW)
    }

    /// A tier whose doorkeeper remembers `window` keys.
    pub fn with_window(blocks: B, window: usize) -> Tier<B> {
        Tier {
            blocks,
            door: Doorkeeper::new(window),
            scratch: cold::Scratch::new(),
            pool: evict::Pool::new(),
            keybuf: Vec::new(),
            rng: Rng::new(0x5eed_1234_9abc_def0),
            stats: Stats::default(),
        }
    }

    /// What has happened so far.
    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    /// The store, for a caller that has to flush or close it.
    pub const fn blocks(&self) -> &B {
        &self.blocks
    }

    /// The store, mutably, for the same reason.
    pub const fn blocks_mut(&mut self) -> &mut B {
        &mut self.blocks
    }

    /// What the tier's own buffers cost, which the memory report has to include
    /// because they are not free and are not counted anywhere else.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.door.memory_bytes()
            + self.scratch.memory_bytes()
            + self.pool.memory_bytes()
            + self.keybuf.capacity()
    }

    /// Move `key`'s value out to the file.
    ///
    /// `Ok(false)` when there is no such key, when it is already on the file,
    /// or when moving it would cost more memory than it saves. None of those is
    /// an error: a caller under memory pressure asks about a lot of keys and
    /// most of the answers are no.
    ///
    /// # Errors
    ///
    /// Whatever the store says when it cannot take the bytes.
    pub fn demote(&mut self, map: &mut RawMap, key: &[u8]) -> Result<bool> {
        let Some(addr) = map.find(key) else {
            return Ok(false);
        };
        let rec = map.value_at(addr);
        if !worth_demoting(rec) {
            return Ok(false);
        }
        let m = value::Meta::from_byte(rec[0]);
        let (kind, enc) = (m.kind(), m.encoding());
        let expire_at = value::expire_at(rec);
        // Carried across rather than restamped. A key that was moved to the file
        // was not used, and a demotion that looked like a use would make the
        // next demotion pick the wrong victim.
        let was = value::access(rec).unwrap_or_default();

        let value::Str::Bytes(bytes) = value::read(rec) else {
            // Int encoding is refused above, so this cannot happen, and if the
            // encoding rules ever change it should be a no and not a panic.
            return Ok(false);
        };
        let len = bytes.len() as u32;
        let chain = cold::write(&mut self.blocks, bytes, &mut self.scratch)?;

        let wrote = map.set_with(
            key,
            value::cold_record_len(expire_at.is_some()),
            |_| {},
            |out| {
                value::write_cold_record(out, kind, enc, chain.at, len, expire_at);
                value::set_access(out, was);
                value::has_expiry(out)
            },
        );
        debug_assert!(wrote.is_some(), "the key was found a moment ago");

        self.stats.demoted += 1;
        self.stats.bytes_out += u64::from(len);
        Ok(true)
    }

    /// Read `key`'s value, from the file if that is where it is.
    ///
    /// `out` is cleared and filled only when the answer is [`Faulted::Served`]
    /// or [`Faulted::Promoted`]. It belongs to the caller so that a server can
    /// keep one buffer per shard and a fault costs no allocation once it has
    /// grown, which is Y7.
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back.
    pub fn fault(&mut self, map: &mut RawMap, key: &[u8], out: &mut Vec<u8>) -> Result<Faulted> {
        self.read(map, key, out, true)
    }

    /// Read `key`'s value and put it back in memory whatever the doorkeeper
    /// thinks.
    ///
    /// This is for a command that is about to write the key. `APPEND` on a
    /// demoted value reads it, adds to it and stores the result, and the result
    /// is a resident record no matter which way the doorkeeper would have gone,
    /// so asking it would be asking a question whose answer cannot be used. The
    /// same goes for `INCR`, `SETRANGE`, `SETBIT`, `GETSET` and the rest of the
    /// read modify write family.
    ///
    /// A promotion here still costs one device read and no more, and the value
    /// it read is the one the caller was going to ask for anyway.
    ///
    /// # Errors
    ///
    /// Whatever the store says when the chain will not read back.
    pub fn thaw(&mut self, map: &mut RawMap, key: &[u8], out: &mut Vec<u8>) -> Result<Faulted> {
        self.read(map, key, out, false)
    }

    /// The body of both, with `ask` saying whether the doorkeeper gets a vote.
    fn read(
        &mut self,
        map: &mut RawMap,
        key: &[u8],
        out: &mut Vec<u8>,
        ask: bool,
    ) -> Result<Faulted> {
        let Some(addr) = map.find(key) else {
            return Ok(Faulted::Missing);
        };
        let rec = map.value_at(addr);
        let Some(c) = value::cold(rec) else {
            return Ok(Faulted::Warm);
        };
        let m = value::Meta::from_byte(rec[0]);
        let enc = m.encoding();
        let expire_at = value::expire_at(rec);
        let was = value::access(rec).unwrap_or_default();

        out.clear();
        out.reserve(c.len as usize);
        let chain = cold::Chain {
            at: c.at,
            len: u64::from(c.len),
        };
        {
            let reader = cold::Reader::open(&self.blocks, chain)?;
            for piece in reader.range(0, reader.len()) {
                out.extend_from_slice(piece?);
            }
        }
        self.stats.faults += 1;
        self.stats.bytes_in += u64::from(c.len);

        // One read is not enough. The bits go down now and the key comes back
        // on the next read, if there is one.
        if ask && !self.door.admit(RawMap::hash_of(key)) {
            self.stats.served += 1;
            return Ok(Faulted::Served);
        }

        let wrote = map.set_with(
            key,
            value::record_len(enc, out.len(), expire_at.is_some()),
            |_| {},
            |dst| {
                value::write_record(dst, enc, out, expire_at);
                value::set_access(dst, was);
                value::has_expiry(dst)
            },
        );
        debug_assert!(wrote.is_some(), "the key was found a moment ago");
        self.stats.promoted += 1;
        Ok(Faulted::Promoted)
    }

    /// Move values out until the map fits in `budget` bytes.
    ///
    /// Returns how many were moved. Stops early when [`BARREN`] rounds in a row
    /// find nothing worth demoting, which is the case where every value left is
    /// shorter than the pointer that would replace it, and the honest answer
    /// there is that memory cannot be given back rather than that the loop
    /// should keep spinning.
    ///
    /// Two things had to be right before that stop rule meant what it says, and
    /// both of them are about a sweep that runs long enough to make most of the
    /// keyspace cold. One barren round is a collision rather than a conclusion,
    /// which is what [`BARREN`] is for, and a round has to spend its budget on
    /// victims found rather than entries walked, which is what [`WALK`] is for.
    /// Each constant has the failure it prevents written on it.
    ///
    /// # Compaction is the part that gives the memory back
    ///
    /// Demoting a key does not free anything on its own, and finding that out
    /// is worth a paragraph. Replacing a long record with a short one leaves
    /// the long one behind as dead bytes in a segment the arena still owns, so
    /// the number a memory limit is compared against does not move until a
    /// segment is evacuated and handed back. So each round of demotions is
    /// followed by [`RawMap::compact_hard`], which is the entry point written
    /// for a store that has run out of room and will evacuate a segment holding
    /// a single dead record rather than wait for a worthwhile one.
    ///
    /// A round drains its whole pool before checking the budget again, so this
    /// can overshoot by up to the pool size. That is bounded by
    /// [`evict::CANDIDATES`] keys and it is the right way round: demoting one
    /// key too many costs one device read later, and stopping one key short
    /// costs a memory limit that was not respected.
    ///
    /// # Errors
    ///
    /// Whatever the store says when it cannot take the bytes.
    pub fn relieve(
        &mut self,
        map: &mut RawMap,
        budget: usize,
        policy: Policy,
        now_ms: u64,
        lfu: Lfu,
    ) -> Result<usize> {
        let mut moved = 0;
        let mut barren = 0;
        while map.memory_bytes() > budget {
            let round = self.round(map, policy, now_ms, lfu)?;
            while map.memory_bytes() > budget && map.compact_hard().is_some() {}
            if round == 0 {
                barren += 1;
                if barren == BARREN {
                    break;
                }
                continue;
            }
            barren = 0;
            moved += round;
        }
        Ok(moved)
    }

    /// One sample and demote pass, which is the body of [`Tier::relieve`] and is
    /// separate so that a test can watch a single round.
    fn round(&mut self, map: &mut RawMap, policy: Policy, now_ms: u64, lfu: Lfu) -> Result<usize> {
        self.pool.clear();
        let r = self.rng.next_u64();
        let pool = &mut self.pool;
        let mut seen = 0usize;
        let mut found = 0usize;
        map.sample(r, |k, v, _| {
            seen += 1;
            if worth_demoting(v) {
                pool.offer(k, evict::score(v, policy, now_ms, lfu));
                found += 1;
            }
            found < evict::CANDIDATES && seen < WALK
        });

        let mut moved = 0;
        // Out of the pool and into a buffer of our own, because the pool hands
        // back a slice of itself and demoting needs the whole tier.
        let mut kb = core::mem::take(&mut self.keybuf);
        while let Some(k) = self.pool.take() {
            kb.clear();
            kb.extend_from_slice(k);
            if self.demote(map, &kb)? {
                moved += 1;
            }
        }
        self.keybuf = kb;
        Ok(moved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::Access;
    use yo_common::{Addr, Code, Error, Space};

    /// The same in memory store the `cold` unit tests use, counting its reads.
    struct Mem {
        blobs: Vec<Vec<u8>>,
        reads: std::cell::Cell<usize>,
    }

    impl Mem {
        fn new() -> Mem {
            Mem {
                blobs: Vec::new(),
                reads: std::cell::Cell::new(0),
            }
        }
    }

    impl Blocks for Mem {
        fn put(&mut self, bytes: &[u8]) -> Result<Addr> {
            self.blobs.push(bytes.to_vec());
            Ok(Addr::new(Space::Log, (self.blobs.len() - 1) as u64))
        }

        fn get(&self, at: Addr) -> Result<&[u8]> {
            self.reads.set(self.reads.get() + 1);
            self.blobs
                .get(at.offset() as usize)
                .map(Vec::as_slice)
                .ok_or_else(|| Error::new(Code::NotFound, "no such block"))
        }
    }

    fn tier() -> Tier<Mem> {
        Tier::new(Mem::new())
    }

    /// A map with one string in it, written the way the keyspace writes one.
    fn map_with(key: &[u8], val: &[u8], expire_at: Option<u64>) -> RawMap {
        let mut m = RawMap::new();
        put(&mut m, key, val, expire_at);
        m
    }

    fn put(m: &mut RawMap, key: &[u8], val: &[u8], expire_at: Option<u64>) {
        let enc = Encoding::of(val);
        let len = value::record_len(enc, val.len(), expire_at.is_some());
        m.set_with(
            key,
            len,
            |_| {},
            |out| {
                value::write_record(out, enc, val, expire_at);
                value::has_expiry(out)
            },
        );
    }

    /// Read a key twice, which is what the doorkeeper asks for before it lets
    /// anything back into memory.
    fn fault_twice(t: &mut Tier<Mem>, m: &mut RawMap, key: &[u8]) -> (Faulted, Faulted, Vec<u8>) {
        let mut out = Vec::new();
        let first = t.fault(m, key, &mut out).expect("a first read");
        let second = t.fault(m, key, &mut out).expect("a second read");
        (first, second, out)
    }

    #[test]
    fn a_value_goes_out_to_the_file_and_the_record_shrinks_to_a_pointer() {
        let val = vec![b'x'; 4000];
        let mut m = map_with(b"k", &val, None);
        let before = m.value_at(m.find(b"k").expect("there")).len();
        let mut t = tier();

        assert!(t.demote(&mut m, b"k").expect("demoted"));

        let rec = m.value_at(m.find(b"k").expect("still there"));
        assert!(rec.len() < before / 100, "the record did not shrink");
        assert_eq!(value::cold(rec).expect("cold").len, 4000);
        assert_eq!(t.stats().demoted, 1);
        assert_eq!(t.stats().bytes_out, 4000);
    }

    #[test]
    fn the_questions_that_do_not_want_the_bytes_are_still_answered_in_memory() {
        let val = vec![b'y'; 900];
        let deadline = Some(1_900_000_000_000);
        let mut m = map_with(b"k", &val, deadline);
        let mut t = tier();
        t.demote(&mut m, b"k").expect("demoted");

        let rec = m.value_at(m.find(b"k").expect("there"));
        // STRLEN, TYPE, OBJECT ENCODING and TTL, in that order, on a key whose
        // bytes are on the device. None of these is allowed to fault.
        assert_eq!(value::str_len(rec), Some(900));
        assert_eq!(value::kind(rec), Kind::String);
        assert_eq!(value::Meta::from_byte(rec[0]).encoding(), Encoding::Raw);
        assert_eq!(value::expire_at(rec), deadline);
        assert_eq!(t.blocks().reads.get(), 0, "answering those read the device");
    }

    #[test]
    fn a_value_too_short_to_be_worth_moving_is_left_where_it_is() {
        // Twelve payload bytes against a twelve byte pointer plus the head that
        // both records share, so this one loses by moving.
        let mut m = map_with(b"k", b"hello-world!", None);
        let mut t = tier();
        assert!(!t.demote(&mut m, b"k").expect("asked"));
        assert!(value::cold(m.value_at(m.find(b"k").expect("there"))).is_none());
    }

    #[test]
    fn an_int_encoded_value_is_never_moved() {
        let mut m = map_with(b"k", b"1234567890123", None);
        let mut t = tier();
        assert!(!t.demote(&mut m, b"k").expect("asked"));
    }

    #[test]
    fn a_key_that_is_not_there_is_a_no_and_not_an_error() {
        let mut m = RawMap::new();
        let mut t = tier();
        assert!(!t.demote(&mut m, b"nothing").expect("asked"));
        let mut out = Vec::new();
        assert_eq!(
            t.fault(&mut m, b"nothing", &mut out).expect("asked"),
            Faulted::Missing
        );
    }

    #[test]
    fn demoting_twice_is_a_no_the_second_time() {
        let val = vec![b'z'; 500];
        let mut m = map_with(b"k", &val, None);
        let mut t = tier();
        assert!(t.demote(&mut m, b"k").expect("demoted"));
        assert!(!t.demote(&mut m, b"k").expect("asked again"));
        assert_eq!(t.stats().demoted, 1);
    }

    #[test]
    fn a_resident_key_is_warm_and_the_buffer_is_left_alone() {
        let mut m = map_with(b"k", b"a value long enough to matter", None);
        let mut t = tier();
        let mut out = vec![1, 2, 3];
        assert_eq!(
            t.fault(&mut m, b"k", &mut out).expect("read"),
            Faulted::Warm
        );
        assert_eq!(out, vec![1, 2, 3], "a warm read touched the buffer");
        assert_eq!(t.stats().faults, 0);
    }

    #[test]
    fn the_first_read_serves_from_the_file_and_the_second_brings_it_back() {
        let val = vec![b'q'; 3000];
        let mut m = map_with(b"k", &val, None);
        let mut t = tier();
        t.demote(&mut m, b"k").expect("demoted");

        let (first, second, out) = fault_twice(&mut t, &mut m, b"k");
        assert_eq!(first, Faulted::Served, "one read earned a slot in memory");
        assert_eq!(second, Faulted::Promoted);
        assert_eq!(out, val);
        assert_eq!(t.stats().faults, 2);
        assert_eq!(t.stats().served, 1);
        assert_eq!(t.stats().promoted, 1);

        // And now it is back, so the third read is not a fault at all.
        let mut again = Vec::new();
        assert_eq!(
            t.fault(&mut m, b"k", &mut again).expect("read"),
            Faulted::Warm
        );
        assert_eq!(
            value::read(m.value_at(m.find(b"k").expect("there"))).len(),
            3000
        );
    }

    #[test]
    fn a_scan_over_cold_data_promotes_nothing() {
        let mut m = RawMap::new();
        let val = vec![b'c'; 700];
        for i in 0..64u32 {
            put(&mut m, &i.to_le_bytes(), &val, None);
        }
        let mut t = tier();
        for i in 0..64u32 {
            t.demote(&mut m, &i.to_le_bytes()).expect("demoted");
        }

        let mut out = Vec::new();
        for i in 0..64u32 {
            t.fault(&mut m, &i.to_le_bytes(), &mut out).expect("read");
        }
        assert_eq!(
            t.stats().promoted,
            0,
            "a single pass over cold keys pulled some back in"
        );
        assert_eq!(t.stats().served, 64);
    }

    #[test]
    fn the_deadline_and_the_access_field_survive_a_round_trip() {
        let val = vec![b'r'; 1200];
        let deadline = Some(1_888_777_666_555);
        let mut m = map_with(b"k", &val, deadline);
        // Stamp something recognisable, so that a demotion that restamped it
        // would show up rather than looking like a fresh record.
        let a = Access::lru(1_000_000);
        {
            let addr = m.find(b"k").expect("there");
            value::set_access(m.value_at_mut(addr), a);
        }
        let mut t = tier();
        t.demote(&mut m, b"k").expect("demoted");
        assert_eq!(
            value::access(m.value_at(m.find(b"k").expect("there"))),
            Some(a),
            "demotion looked like a use"
        );

        let (_, _, out) = fault_twice(&mut t, &mut m, b"k");
        assert_eq!(out, val);
        let rec = m.value_at(m.find(b"k").expect("there"));
        assert_eq!(value::expire_at(rec), deadline);
        assert_eq!(value::access(rec), Some(a));
    }

    #[test]
    fn a_value_bigger_than_one_chunk_makes_the_trip_as_well() {
        let val: Vec<u8> = (0..cold::CHUNK * 2 + 77).map(|i| (i % 251) as u8).collect();
        let mut m = map_with(b"big", &val, None);
        let mut t = tier();
        assert!(t.demote(&mut m, b"big").expect("demoted"));
        let (_, _, out) = fault_twice(&mut t, &mut m, b"big");
        assert_eq!(out, val);
    }

    #[test]
    fn relieve_moves_values_out_until_the_map_fits() {
        // Enough data to span several arena segments. A budget below one
        // segment is a budget nothing can meet, because a segment is the unit
        // the arena hands back, and a test that asked for one would be testing
        // the arena's minimum rather than the demotion.
        let mut m = RawMap::new();
        let val = vec![b'p'; 2000];
        for i in 0..4_000u32 {
            put(&mut m, &i.to_le_bytes(), &val, None);
        }
        let full = m.memory_bytes();
        let budget = full / 2;

        let mut t = tier();
        let moved = t
            .relieve(
                &mut m,
                budget,
                Policy::AllKeysLru,
                2_000_000,
                Lfu::default(),
            )
            .expect("relieved");
        assert!(moved > 0, "nothing was moved");
        assert!(
            m.memory_bytes() <= budget,
            "still {} bytes against a budget of {budget}",
            m.memory_bytes()
        );
        // Every key is still there, which is the whole difference between this
        // and eviction.
        assert_eq!(m.len(), 4_000);
    }

    #[test]
    fn one_unlucky_round_does_not_end_the_sweep() {
        // Two bugs written down, both of which left a sweep that had been asked
        // for the whole keyspace sitting on a large part of it. The first
        // version of `relieve` stopped on the first round that found nothing,
        // and quit at six percent moved, because sampling walks forward from a
        // segment and a bucket drawn at random and two rounds that draw the
        // same pair see the same entries. The second counted entries walked
        // against a round's budget of sixteen rather than victims found, and
        // stalled at eighty five percent, because by then almost every entry a
        // round walked was one it had already moved.
        let mut m = RawMap::new();
        let val = vec![b'u'; 2000];
        for i in 0..4_000u32 {
            put(&mut m, &i.to_le_bytes(), &val, None);
        }
        let mut t = tier();
        t.relieve(&mut m, 1, Policy::AllKeysLru, 2_000_000, Lfu::default())
            .expect("relieved");

        let cold = (0..4_000u32)
            .filter(|i| {
                let addr = m.find(&i.to_le_bytes()).expect("still there");
                value::cold(m.value_at(addr)).is_some()
            })
            .count();
        assert!(
            cold > 3_900,
            "only {cold} of 4000 were moved, so the sweep gave up early"
        );
    }

    #[test]
    fn the_memory_the_map_holds_actually_goes_down() {
        // Demotion on its own frees nothing: the record it replaces becomes dead
        // bytes in a segment the arena still owns. This is the check that the
        // compaction in `relieve` is doing the part that gives it back.
        let mut m = RawMap::new();
        let val = vec![b'v'; 2000];
        for i in 0..4_000u32 {
            put(&mut m, &i.to_le_bytes(), &val, None);
        }
        let before = m.memory_bytes();
        let mut t = tier();
        t.relieve(&mut m, 1, Policy::AllKeysLru, 2_000_000, Lfu::default())
            .expect("relieved");

        // What the same four thousand keys would have cost if their values had
        // never been in memory at all. The arena cannot hand back its last
        // segment, so this is the floor, and asking the sweep to reach it says
        // more than a fraction of `before` picked because it passes.
        let mut bare = RawMap::new();
        let stub = vec![b'v'; 4];
        for i in 0..4_000u32 {
            put(&mut bare, &i.to_le_bytes(), &stub, None);
        }
        let floor = bare.memory_bytes();
        assert!(
            m.memory_bytes() <= floor,
            "{before} bytes went to {}, and the floor is {floor}",
            m.memory_bytes()
        );
    }

    #[test]
    fn relieve_gives_up_rather_than_spinning_when_nothing_is_worth_moving() {
        let mut m = RawMap::new();
        for i in 0..200u32 {
            put(&mut m, &i.to_le_bytes(), b"tiny", None);
        }
        let mut t = tier();
        let moved = t
            .relieve(&mut m, 1, Policy::AllKeysLru, 2_000_000, Lfu::default())
            .expect("asked");
        assert_eq!(moved, 0);
    }

    #[test]
    fn what_relieve_moved_still_reads_back_byte_for_byte() {
        let mut m = RawMap::new();
        let mut want = Vec::new();
        for i in 0..4_000u32 {
            let val: Vec<u8> = (0..900).map(|j| (i as usize + j) as u8).collect();
            put(&mut m, &i.to_le_bytes(), &val, None);
            want.push(val);
        }
        let budget = m.memory_bytes() / 2;
        let mut t = tier();
        let moved = t
            .relieve(
                &mut m,
                budget,
                Policy::AllKeysLru,
                2_000_000,
                Lfu::default(),
            )
            .expect("relieved");
        assert!(moved > 0, "nothing was moved, so this checked nothing");

        let mut out = Vec::new();
        for (i, val) in want.iter().enumerate() {
            let key = (i as u32).to_le_bytes();
            match t.fault(&mut m, &key, &mut out).expect("read") {
                Faulted::Warm => {
                    let rec = m.value_at(m.find(&key).expect("there"));
                    assert_eq!(value::read(rec), value::Str::Bytes(val));
                }
                Faulted::Served | Faulted::Promoted => assert_eq!(&out, val),
                Faulted::Missing => panic!("key {i} went missing"),
            }
        }
    }
}
