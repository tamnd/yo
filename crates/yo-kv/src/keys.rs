//! Moving a key, copying one, and touching one.
//!
//! Three of the four commands here move whole values around, and all three of
//! them are careful about the same thing: a value lives in two places at once.
//! A string lives entirely in its record, and a set or a hash lives in a slab
//! with the record holding nothing but a slot number. So there is no one way to
//! move a value, and a command that forgets which case it is in either drops
//! members on the floor or leaves a body in the slab that nothing points at.
//!
//! [`Keyspace::rename`] moves the record's bytes and leaves the body exactly
//! where it is, because a slot number that moves to a different key is still
//! the same slot. Renaming a set of a million members writes thirteen bytes.
//!
//! [`Keyspace::copy`] cannot do that, since two records pointing at one slot
//! would be one set that answers to two names and `SADD` to either would show
//! up in both. So the body is cloned, which is the one thing here that costs
//! what the value is worth. That is Redis's cost too and there is no version of
//! `COPY` that avoids it.
//!
//! # Why export and import are separate and public
//!
//! `COPY key dst DB n` puts a value in a database this one cannot reach. The
//! wire layer holds every database and this one holds none of them, so the two
//! halves are separate calls and the caller is what joins them up.
//!
//! It also makes the pair the answer for `MOVE`, `DUMP` and `RESTORE`, which
//! want exactly this: a value lifted out of a database, standing on its own with
//! its deadline attached.
//!
//! There are two ways to lift one out. [`Keyspace::export`] clones the body and
//! leaves the key where it is, which is what `COPY` needs, and
//! [`Keyspace::take`] pulls the body out of the slab and deletes the key, which
//! is what `MOVE` needs. `MOVE` through `export` would clone a set of a million
//! members and then throw the original away a line later, so the two are
//! separate calls rather than one call with a flag.
//!
//! # And the same pair again, with bytes in the middle
//!
//! `DUMP` and `RESTORE` are the same shape one step further out. A record is a
//! value standing on its own inside this process, and a payload is a value
//! standing on its own outside it, so [`Keyspace::dump`] is an export followed
//! by [`crate::rdb`] and [`Keyspace::restore`] is `rdb` followed by an import.
//! The deadline is the one thing that does not make the trip, because `DUMP`
//! drops it and `RESTORE` is given a fresh one.

use yo_common::Result;

use crate::array::Array;
use crate::foreign::Foreign;
use crate::hash::Hash;
use crate::keyspace::Keyspace;
use crate::list::List;
use crate::rdb;
use crate::set::Set;
use crate::stream::Stream;
use crate::value::{self, Kind};
use crate::zset::Zset;

/// Everything under one key, lifted out so it can be put somewhere else.
///
/// It owns what it holds. A record taken out of a database survives that
/// database being written to, flushed or dropped, which is what makes it safe
/// to carry between two of them.
#[derive(Debug, Clone)]
pub struct Record {
    body: Body,
    /// The deadline, which travels with the value. `COPY` and `RENAME` both
    /// keep it, and a copy of a key with ten seconds left has ten seconds left.
    expire_at: Option<u64>,
}

impl Record {
    /// A record built from parts, for a caller that has both.
    ///
    /// [`crate::rdb`] is that caller and there is no other. A record normally
    /// comes out of a database and this is the one way to make one that never
    /// was in a database, which is what a payload arriving from a client is.
    pub(crate) const fn new(body: Body, expire_at: Option<u64>) -> Record {
        Record { body, expire_at }
    }

    /// What it holds, for the code that has to write it down.
    pub(crate) const fn body(&self) -> &Body {
        &self.body
    }

    /// What type this is, which the caller usually knows and sometimes does not.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        match self.body {
            Body::String(_) => Kind::String,
            Body::Set(_) => Kind::Set,
            Body::Hash(_) => Kind::Hash,
            Body::List(_) => Kind::List,
            Body::Zset(_) => Kind::Zset,
            Body::Array(_) => Kind::Array,
            Body::Stream(_) => Kind::Stream,
            Body::Foreign(_) => Kind::Foreign,
        }
    }

    /// When it goes away, if anything says.
    #[must_use]
    pub const fn expire_at(&self) -> Option<u64> {
        self.expire_at
    }
}

/// The eight things a record can be, owned rather than borrowed.
///
/// One variant per type that a key can hold, and that is the point: the day an
/// eighth type lands, the compiler names this file. It did not before, because
/// the match in [`Keyspace::export`] had a catch all arm at the bottom, and a
/// catch all in front of an enum the rest of the crate keeps growing is a hole
/// that reports itself as a panic on a live server rather than as a build error.
#[derive(Debug)]
pub(crate) enum Body {
    String(Vec<u8>),
    Set(Set),
    Hash(Hash),
    List(List),
    Zset(Zset),
    Array(Array),
    Stream(Stream),
    Foreign(Box<dyn Foreign>),
}

/// Every body but the foreign one can be copied.
///
/// Written out rather than derived so that the one variant which cannot is a
/// named arm here instead of a `Clone` bound the escape could never satisfy.
/// Nothing reaches it: [`Keyspace::export`] is the only thing that clones a
/// body and it answers `None` for a foreign one before it gets this far, so
/// this is the assertion of that rather than a case to handle.
impl Clone for Body {
    fn clone(&self) -> Body {
        match self {
            Body::String(v) => Body::String(v.clone()),
            Body::Set(v) => Body::Set(v.clone()),
            Body::Hash(v) => Body::Hash(v.clone()),
            Body::List(v) => Body::List(v.clone()),
            Body::Zset(v) => Body::Zset(v.clone()),
            Body::Array(v) => Body::Array(v.clone()),
            Body::Stream(v) => Body::Stream(v.clone()),
            Body::Foreign(_) => unreachable!("a foreign body never reaches a clone"),
        }
    }
}

/// What a rename or a copy did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moved {
    /// There was no source key, so there was nothing to move.
    Missing,
    /// The destination was there and the caller said not to write over it.
    Taken,
    /// The source holds something there is no way to copy.
    ///
    /// A foreign body is owned by the engine above this crate and there is no
    /// generic way to ask one for a duplicate of itself. A graph could grow a
    /// deep copy and a vector index probably should not have one at all, so the
    /// decision belongs to whichever of them is under the key rather than here.
    /// Answered rather than panicked so the wire can say so in a sentence.
    Unsupported,
    /// It happened.
    Ok,
}

impl Keyspace {
    /// Take a copy of everything under `key`, deadline included.
    ///
    /// `None` for a key that is not there, and for one whose deadline has gone,
    /// which is reaped on the way through the same as every other read.
    ///
    /// This clones the body, so exporting a set of a million members costs a set
    /// of a million members. [`Keyspace::rename`] exists so that the one case
    /// which does not need a copy does not pay for one.
    pub fn export(&mut self, key: &[u8]) -> Option<Record> {
        let mut addr = self.live_rec(key)?;
        // A value on the file is read back but not put back. `DUMP` over a
        // whole database is the scan the doorkeeper is there for: a backup
        // should not be able to pull everything into memory on its way past. A
        // chain that will not read back answers as a key that is not there,
        // which is the only answer this signature has room for.
        if value::cold(self.map.value_at(addr)).is_some() {
            if self.warm(key).is_err() {
                return None;
            }
            addr = self.map.find(key)?;
        }
        let rec = self.map.value_at(addr);
        let expire_at = value::expire_at(rec);
        // The slot is read inside the arms and not before them. A string record
        // holds the string and not a slot, so reading four bytes where the slot
        // would be reads off the end of a short one.
        let body = match value::kind(rec) {
            Kind::String => Body::String(self.value_of(key, rec).to_vec()),
            Kind::Set => Body::Set(
                self.sets
                    .get(value::slot(rec))
                    .expect("the record points at its body")
                    .clone(),
            ),
            Kind::Hash => Body::Hash(
                self.hashes
                    .get(value::slot(rec))
                    .expect("the record points at its body")
                    .clone(),
            ),
            Kind::List => Body::List(
                self.lists
                    .get(value::slot(rec))
                    .expect("the record points at its body")
                    .clone(),
            ),
            Kind::Zset => Body::Zset(
                self.zsets
                    .get(value::slot(rec))
                    .expect("the record points at its body")
                    .clone(),
            ),
            Kind::Array => Body::Array(
                self.arrays
                    .get(value::slot(rec))
                    .expect("the record points at its body")
                    .clone(),
            ),
            Kind::Stream => Body::Stream(
                self.streams
                    .get(value::slot(rec))
                    .expect("the record points at its body")
                    .clone(),
            ),
            // A copy is the one thing a foreign body cannot be asked for. See
            // [`Moved::Unsupported`]. `None` here reads the same as a missing
            // key to a caller that only wanted the record, which is why `COPY`
            // and `DUMP` both check the kind themselves before they get here
            // rather than reporting a graph as absent.
            Kind::Foreign => return None,
        };
        Some(Record { body, expire_at })
    }

    /// Lift everything under `key` out and leave the key gone.
    ///
    /// The same answer [`Keyspace::export`] gives, without the clone. A body in
    /// the slab is already a value standing on its own, so a caller that is
    /// about to delete the source can have that body itself rather than a copy
    /// of it, and taking a set of a million members costs a slot number.
    ///
    /// This is what `MOVE` wants and what `COPY` cannot have. The difference is
    /// that a move leaves nothing behind, so there is never a moment where two
    /// records point at one slot.
    ///
    /// The record is removed here rather than by the caller, because the body is
    /// out of the slab by then and a record still pointing at a slot that has
    /// been freed is the one state this file exists to prevent. A `del` on top
    /// of this would free the body a second time and underflow the count of keys
    /// that hold one.
    pub fn take(&mut self, key: &[u8]) -> Option<Record> {
        let mut addr = self.live_rec(key)?;
        // The same read as [`Keyspace::export`] does, for the same reason,
        // except that here the record is about to go anyway. What the caller
        // does with the bytes decides where they end up, and on a `RENAME` that
        // is a resident record under the new name with the old chunks left for
        // the log's compaction to collect.
        if value::cold(self.map.value_at(addr)).is_some() {
            if self.warm(key).is_err() {
                return None;
            }
            addr = self.map.find(key)?;
        }
        let rec = self.map.value_at(addr);
        let expire_at = value::expire_at(rec);
        let kind = value::kind(rec);
        // A string record is the value, so there is nothing in the slab to take
        // and the bytes have to be copied out before the record goes. It leaves
        // early because the slot below is not there to read on this one.
        if kind == Kind::String {
            let bytes = self.value_of(key, rec).to_vec();
            self.del_rec(key);
            return Some(Record {
                body: Body::String(bytes),
                expire_at,
            });
        }
        let slot = value::slot(rec);
        let gone = "the record points at its body";
        let body = match kind {
            Kind::Set => Body::Set(self.sets.remove(slot).expect(gone)),
            Kind::Hash => Body::Hash(self.hashes.remove(slot).expect(gone)),
            Kind::List => Body::List(self.lists.remove(slot).expect(gone)),
            Kind::Zset => Body::Zset(self.zsets.remove(slot).expect(gone)),
            Kind::Array => Body::Array(self.arrays.remove(slot).expect(gone)),
            Kind::Stream => Body::Stream(self.streams.remove(slot).expect(gone)),
            // A move is the one of the two that a foreign body can do, because
            // it hands the box over rather than asking for a second one.
            Kind::Foreign => Body::Foreign(self.foreign.remove(slot).expect(gone)),
            // Handled above, and named rather than caught, as in `export`.
            Kind::String => unreachable!("handled above"),
        };
        self.bodies -= 1;
        self.del_rec(key);
        Some(Record { body, expire_at })
    }

    /// Put `rec` under `key`, over whatever was there.
    ///
    /// The caller has already decided that writing over the destination is
    /// allowed, which is why this answers nothing. Whatever was under `key` is
    /// freed first, body and all, so this cannot leak a slab slot.
    pub fn import(&mut self, key: &[u8], rec: Record) {
        let at = rec.expire_at;
        match rec.body {
            // The string path frees the old body itself, because every string
            // write has to and this is not the place to make it special.
            Body::String(bytes) => self.store(key, &bytes, at),
            Body::Set(set) => {
                self.free_body(key);
                let slot = self.sets.insert(set);
                self.bodies += 1;
                self.write_slot(key, Kind::Set, slot, at);
            }
            Body::Hash(hash) => {
                self.free_body(key);
                let slot = self.hashes.insert(hash);
                self.bodies += 1;
                self.write_slot(key, Kind::Hash, slot, at);
            }
            Body::List(list) => {
                self.free_body(key);
                let slot = self.lists.insert(list);
                self.bodies += 1;
                self.write_slot(key, Kind::List, slot, at);
            }
            Body::Zset(zset) => {
                self.free_body(key);
                let slot = self.zsets.insert(zset);
                self.bodies += 1;
                self.write_slot(key, Kind::Zset, slot, at);
            }
            Body::Array(array) => {
                self.free_body(key);
                let slot = self.arrays.insert(array);
                self.bodies += 1;
                self.write_slot(key, Kind::Array, slot, at);
            }
            Body::Stream(stream) => {
                self.free_body(key);
                let slot = self.streams.insert(stream);
                self.bodies += 1;
                self.write_slot(key, Kind::Stream, slot, at);
            }
            Body::Foreign(body) => {
                self.free_body(key);
                let slot = self.foreign.insert(body);
                self.bodies += 1;
                self.write_slot(key, Kind::Foreign, slot, at);
            }
        }
    }

    /// `DUMP key`, which is a value on its own with a checksum on the end.
    ///
    /// `None` for a key that is not there, and for a key holding something with
    /// no RDB shape, which today is only the sparse array and which no command
    /// on the wire can create. Both answer the null bulk that `DUMP` gives for a
    /// missing key, so a client cannot tell them apart and there is nothing here
    /// for it to tell apart yet.
    ///
    /// The deadline is deliberately left behind. Redis's `DUMP` does the same
    /// and the reason is that a payload has no idea how long it will be in
    /// flight, so carrying an absolute deadline would arrive already expired and
    /// carrying a relative one would quietly extend it. `RESTORE` takes the ttl
    /// as an argument instead, which puts the decision on whoever knows.
    pub fn dump(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let rec = self.export(key)?;
        rdb::dump(&rec)
    }

    /// `RESTORE key ttl payload`, with `replace` for the `REPLACE` option.
    ///
    /// [`Moved::Taken`] for a key that is already there without `REPLACE`, which
    /// is checked before the payload is looked at because that is the order
    /// Redis checks in and a busy key should not depend on whether the bytes
    /// behind it happened to be good.
    ///
    /// The clone in `export` is not paid here. The payload is parsed straight
    /// into a body and that body goes into the slab, so restoring a set of a
    /// million members builds one set.
    ///
    /// # Errors
    ///
    /// [`rdb::Bad::Footer`] when the version is from the future or the checksum
    /// does not match, and [`rdb::Bad::Format`] when the bytes were intact and
    /// still did not describe anything this server can hold. The wire layer has
    /// a different message for each and clients depend on the difference.
    pub fn restore(
        &mut self,
        key: &[u8],
        payload: &[u8],
        expire_at: Option<u64>,
        replace: bool,
    ) -> std::result::Result<Moved, rdb::Bad> {
        if !replace && self.exists(key) {
            return Ok(Moved::Taken);
        }
        let limits = rdb::Limits {
            set: &self.limits,
            hash: &self.hash_limits,
            list: &self.list_limits,
            zset: &self.zset_limits,
        };
        let now = self.clock.now_ms();
        let body = rdb::load(payload, limits, now)?;
        // A deadline that has already gone means there is nothing to create, and
        // the payload is still parsed first rather than skipped. A client that
        // sent bad bytes and a stale deadline should be told about the bytes,
        // and finding out only when the deadline is fixed is a bad afternoon.
        if expire_at.is_some_and(|at| at <= now) {
            // A no op unless `REPLACE` was given, since a key that was there
            // without it has already been refused above.
            self.del(key);
            return Ok(Moved::Ok);
        }
        self.import(key, Record::new(body, expire_at));
        Ok(Moved::Ok)
    }

    /// `RENAME src dst`, and `RENAMENX` when `only_if_new`.
    ///
    /// The body never moves. A set or a hash is a slot number in a record, and a
    /// slot number under a different key is the same set, so this writes the
    /// source's record bytes under the destination and deletes the source
    /// record without freeing anything. That is why renaming a large collection
    /// is the same call as renaming a short string.
    ///
    /// The deadline travels with the source and the destination's own deadline
    /// goes with the value it belonged to, which falls out of moving the whole
    /// record rather than being a rule applied on top of it.
    ///
    /// Renaming a key onto itself is allowed and does nothing, which is Redis's
    /// answer. `RENAMENX` on the same key answers [`Moved::Taken`] instead,
    /// because the destination does exist, and a key is not new because it is
    /// the one you already had.
    pub fn rename(&mut self, src: &[u8], dst: &[u8], only_if_new: bool) -> Moved {
        if self.live_rec(src).is_none() {
            return Moved::Missing;
        }
        let same = src == dst;
        if only_if_new && (same || self.live_rec(dst).is_some()) {
            return Moved::Taken;
        }
        if same {
            return Moved::Ok;
        }
        // The record and not the value: a tag, a deadline and then either the
        // string itself or four bytes saying which slot the body is in. Copying
        // it out ends the borrow of the map so the write below can begin.
        //
        // Into the database's scratch buffer rather than a fresh `Vec`, because
        // a record under a collection key is nine bytes and `RENAME` is not
        // rare enough to pay a malloc and a free for nine bytes. Taken out and
        // put back, so the map is free to be borrowed in between.
        let addr = self.map.find(src).expect("it was live a line ago");
        let mut bytes = std::mem::take(&mut self.scratch);
        bytes.clear();
        bytes.extend_from_slice(self.map.value_at(addr));
        self.free_body(dst);
        self.write_rec(dst, bytes.len(), |out| {
            out.copy_from_slice(&bytes);
        });
        self.scratch = bytes;
        // `del_rec` and not `drop_key`, which is the whole point. The body under
        // the source belongs to the destination now and freeing it here would
        // take it away from the key that just gained it. It still goes through
        // `del_rec` rather than straight at the map, because the record is going
        // away either way and the count of keys with deadlines has to hear about
        // it.
        self.del_rec(src);
        Moved::Ok
    }

    /// `COPY src dst`, within one database.
    ///
    /// Across two databases the caller runs [`Keyspace::export`] on one and
    /// [`Keyspace::import`] on the other, because a database cannot see its
    /// neighbours from in here.
    ///
    /// A destination whose deadline has gone counts as free, so this answers
    /// [`Moved::Ok`] without `replace` on a key that has technically expired and
    /// not yet been collected. That is Redis's behaviour and it is the only one
    /// that is consistent with `EXISTS` saying zero for the same key.
    /// A key copied onto itself answers [`Moved::Ok`] and does nothing, and
    /// without `replace` it answers [`Moved::Taken`], which is the same pair of
    /// answers [`Keyspace::rename`] gives. The wire never asks: Redis refuses
    /// `COPY k k` with an error and so does the dispatch. This is for the
    /// embedded caller, who can ask, and for whom freeing the body and then
    /// writing a record that points at it would be the worst of the answers
    /// available.
    pub fn copy(&mut self, src: &[u8], dst: &[u8], replace: bool) -> Moved {
        if self.live_rec(src).is_none() {
            return Moved::Missing;
        }
        let same = src == dst;
        if !replace && (same || self.live_rec(dst).is_some()) {
            return Moved::Taken;
        }
        if same {
            return Moved::Ok;
        }
        // Asked before anything is written, so a refused copy leaves both keys
        // exactly as they were rather than freeing the destination first. A
        // rename does not need the same guard, because it moves the record and
        // the body under it travels with the record. Only a copy needs a second
        // body, and a foreign one cannot be asked for one.
        if self.kind_of(src) == Some(Kind::Foreign) {
            return Moved::Unsupported;
        }
        // The destination is settled before anything is copied, which is the
        // difference between a refused copy of a million member set costing
        // nothing and costing the set.
        //
        // Both keys have been reaped by now, so the address below stays good
        // for as long as it is held. It is read after the reaping and not
        // before, because a reap can move records around.
        let addr = self.map.find(src).expect("it was live a line ago");
        if value::kind(self.map.value_at(addr)) == Kind::String {
            // A string record is the value, deadline and all, so copying the
            // record is copying the key. That is [`Keyspace::rename`]'s trick,
            // except the source stays where it is, and it goes through the
            // database's scratch buffer for the same reason: the borrow of the
            // map has to end before the write can begin, and a short string is
            // not worth a malloc and a free.
            let mut bytes = std::mem::take(&mut self.scratch);
            bytes.clear();
            bytes.extend_from_slice(self.map.value_at(addr));
            self.free_body(dst);
            self.write_rec(dst, bytes.len(), |out| {
                out.copy_from_slice(&bytes);
            });
            self.scratch = bytes;
            return Moved::Ok;
        }
        // A collection is a clone and there is no way around that: the
        // destination has to end up owning a set of its own.
        let rec = self.export(src).expect("it was live a line ago");
        self.import(dst, rec);
        Moved::Ok
    }

    /// `TOUCH key [key ...]`. Answers how many of them are there.
    ///
    /// The same answer `EXISTS` gives, including a key named twice counting
    /// twice. On a real server the difference is that this moves the key up the
    /// eviction order, and there is no eviction here yet, so for now the two are
    /// the same walk and the day eviction lands this is where the bump goes.
    pub fn touch<'k>(&mut self, keys: impl Iterator<Item = &'k [u8]>) -> usize {
        keys.filter(|key| self.exists(key)).count()
    }

    /// The record a set or a hash gets: a tag, a slot number and maybe a
    /// deadline. Both arms of [`Keyspace::import`] want it and neither wants to
    /// spell it out.
    fn write_slot(&mut self, key: &[u8], kind: Kind, slot: u32, at: Option<u64>) {
        let len = value::slot_record_len(at.is_some());
        self.write_rec(key, len, |out| {
            value::write_slot_record(out, kind, slot, at);
        });
    }
}

/// The error `RENAME` and `RENAMENX` answer for a source that is not there.
///
/// It is the same sentence for both and it is an error and not a zero, which is
/// unusual enough among the keyspace commands to be worth its own name: every
/// other command here treats a missing key as an ordinary answer.
#[must_use]
pub fn no_such_key() -> yo_common::Error {
    yo_common::Error::new(yo_common::Code::Invalid, "no such key")
}

/// So that a caller can write `?` on a rename without unpacking the enum.
///
/// [`Moved::Taken`] is not an error here, because for `RENAMENX` it is the whole
/// answer and for `RENAME` it cannot happen.
impl Moved {
    /// The source was there, or the error `RENAME` gives when it was not.
    ///
    /// # Errors
    ///
    /// [`yo_common::Code::Invalid`] with Redis's `no such key` for
    /// [`Moved::Missing`].
    pub fn found(self) -> Result<Moved> {
        match self {
            Moved::Missing => Err(no_such_key()),
            other => Ok(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Clock;
    use crate::End;
    use crate::zsets::ZAdd;
    use crate::{Applied, Cond};

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000_000))
    }

    fn members(d: &mut Keyspace, key: &[u8]) -> Vec<String> {
        let mut out: Vec<String> = d
            .smembers(key)
            .expect("a set")
            .expect("a key")
            .map(|m| String::from_utf8(m.to_vec()).expect("utf8 in these tests"))
            .collect();
        out.sort();
        out
    }

    fn put(d: &mut Keyspace, key: &[u8], val: &[u8]) {
        d.set_plain(key, val).expect("room for a record");
    }

    fn read(d: &mut Keyspace, key: &[u8]) -> Vec<u8> {
        d.get(key).expect("a string").expect("there").to_vec()
    }

    #[test]
    fn a_rename_moves_the_value_and_leaves_nothing_behind() {
        let mut d = db();
        put(&mut d, b"a", b"v1");

        assert_eq!(d.rename(b"a", b"b", false), Moved::Ok);
        assert!(!d.exists(b"a"));
        assert_eq!(read(&mut d, b"b"), b"v1");
    }

    /// `RENAME` used to copy the source record into a fresh `Vec` so it could
    /// let go of the map before writing, and that record is nine bytes when the
    /// key holds a collection.
    #[test]
    fn a_rename_does_not_allocate_to_carry_the_record_across() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        // Both names get used before the count starts, so the map has already
        // made room for them and the loop below is renames and nothing else.
        for _ in 0..4 {
            assert_eq!(d.rename(b"a", b"b", false), Moved::Ok);
            assert_eq!(d.rename(b"b", b"a", false), Moved::Ok);
        }
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..50 {
                assert_eq!(d.rename(b"a", b"b", false), Moved::Ok);
                assert_eq!(d.rename(b"b", b"a", false), Moved::Ok);
            }
        });
        assert_eq!(allocs, 0, "rename allocated {allocs} times in a hundred");
        assert_eq!(read(&mut d, b"a"), b"v1");
    }

    #[test]
    fn a_rename_with_no_source_is_the_one_error_in_this_file() {
        let mut d = db();
        assert_eq!(d.rename(b"a", b"b", false), Moved::Missing);
        assert_eq!(d.rename(b"a", b"b", true), Moved::Missing);
        assert_eq!(
            d.copy(b"a", b"b", false),
            Moved::Missing,
            "copy just says 0"
        );
    }

    #[test]
    fn a_rename_carries_the_source_deadline_and_drops_the_destination_one() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        d.set_expiry(b"a", Some(2_000_000));
        put(&mut d, b"b", b"v2");
        d.set_expiry(b"b", Some(1_500_000));

        assert_eq!(d.rename(b"a", b"b", false), Moved::Ok);
        assert_eq!(d.deadline_of(b"b"), crate::Ask::At(2_000_000));
    }

    #[test]
    fn renaming_a_key_onto_itself_keeps_it_and_renamenx_refuses() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        d.set_expiry(b"a", Some(2_000_000));

        assert_eq!(d.rename(b"a", b"a", false), Moved::Ok);
        assert_eq!(read(&mut d, b"a"), b"v1");
        assert_eq!(d.deadline_of(b"a"), crate::Ask::At(2_000_000));
        assert_eq!(d.rename(b"a", b"a", true), Moved::Taken);
    }

    #[test]
    fn renamenx_writes_over_nothing() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        put(&mut d, b"b", b"v2");

        assert_eq!(d.rename(b"a", b"b", true), Moved::Taken);
        assert_eq!(read(&mut d, b"a"), b"v1");
        assert_eq!(read(&mut d, b"b"), b"v2");
        assert_eq!(d.rename(b"a", b"c", true), Moved::Ok);
        assert!(!d.exists(b"a"));
    }

    #[test]
    fn renaming_a_set_moves_the_slot_and_not_the_members() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref(), b"m2".as_ref()].into_iter())
            .expect("a set");
        let before = d.memory_bytes();

        assert_eq!(d.rename(b"s", b"t", false), Moved::Ok);
        assert_eq!(members(&mut d, b"t"), ["m1", "m2"]);
        assert_eq!(d.kind_of(b"t"), Some(Kind::Set));
        assert!(!d.exists(b"s"));
        // The record moved and the body did not, so the only thing that can
        // have changed size is the record itself.
        assert!(
            d.memory_bytes().abs_diff(before) < 64,
            "the members were not copied"
        );
    }

    #[test]
    fn renaming_over_a_set_frees_the_set_that_was_there() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref()].into_iter()).expect("a set");
        d.sadd(b"t", [b"m2".as_ref()].into_iter()).expect("a set");
        assert_eq!(d.sets.len(), 2);

        assert_eq!(d.rename(b"s", b"t", false), Moved::Ok);
        assert_eq!(d.sets.len(), 1, "the destination's body went with it");
        assert_eq!(members(&mut d, b"t"), ["m1"]);
    }

    #[test]
    fn a_copy_is_a_second_value_and_not_a_second_name() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref(), b"m2".as_ref()].into_iter())
            .expect("a set");

        assert_eq!(d.copy(b"s", b"t", false), Moved::Ok);
        d.sadd(b"t", [b"m3".as_ref()].into_iter()).expect("a set");
        assert_eq!(
            members(&mut d, b"s"),
            ["m1", "m2"],
            "the original is intact"
        );
        assert_eq!(members(&mut d, b"t"), ["m1", "m2", "m3"]);
    }

    #[test]
    fn a_copy_refuses_a_destination_it_was_not_told_it_could_have() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        put(&mut d, b"b", b"v2");

        assert_eq!(d.copy(b"a", b"b", false), Moved::Taken);
        assert_eq!(read(&mut d, b"b"), b"v2");
        assert_eq!(d.copy(b"a", b"b", true), Moved::Ok);
        assert_eq!(read(&mut d, b"b"), b"v1");
    }

    /// `COPY` of a string used to go through `export`, which builds a `Vec` of
    /// the value so that `import` can copy it into the map and drop it.
    #[test]
    fn a_copy_of_a_string_does_not_allocate() {
        let mut d = db();
        put(&mut d, b"a", b"a-value-of-some-length");
        // Warmed up, so the map has already made room for both names and the
        // loop below is copies and nothing else.
        for _ in 0..4 {
            assert_eq!(d.copy(b"a", b"b", true), Moved::Ok);
        }
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..50 {
                assert_eq!(d.copy(b"a", b"b", true), Moved::Ok);
            }
        });
        assert_eq!(allocs, 0, "copy allocated {allocs} times in fifty");
        assert_eq!(read(&mut d, b"b"), b"a-value-of-some-length");
    }

    /// The embedded caller can ask for this and the wire cannot, because the
    /// dispatch turns it into an error before it gets here. Freeing the body
    /// and then writing a record that still points at it would be the way to
    /// get this wrong.
    #[test]
    fn a_copy_onto_itself_leaves_the_key_alone() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref(), b"m2".as_ref()].into_iter())
            .expect("a set");

        assert_eq!(d.copy(b"s", b"s", false), Moved::Taken);
        assert_eq!(d.copy(b"s", b"s", true), Moved::Ok);
        assert_eq!(members(&mut d, b"s"), ["m1", "m2"]);
        assert_eq!(d.sets.len(), 1, "no second body was made or lost");
    }

    #[test]
    fn a_copy_carries_the_deadline() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        d.set_expiry(b"a", Some(2_000_000));

        assert_eq!(d.copy(b"a", b"b", false), Moved::Ok);
        assert_eq!(d.deadline_of(b"b"), crate::Ask::At(2_000_000));
        assert_eq!(d.deadline_of(b"a"), crate::Ask::At(2_000_000));
    }

    #[test]
    fn a_destination_that_has_already_gone_counts_as_free() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        put(&mut d, b"b", b"v2");
        d.set_expiry(b"b", Some(999_999));

        assert_eq!(d.copy(b"a", b"b", false), Moved::Ok, "b was already gone");
        assert_eq!(read(&mut d, b"b"), b"v1");
    }

    #[test]
    fn a_source_that_has_already_gone_is_not_a_source() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        d.set_expiry(b"a", Some(999_999));

        assert_eq!(d.rename(b"a", b"b", false), Moved::Missing);
        assert_eq!(d.copy(b"a", b"b", false), Moved::Missing);
    }

    #[test]
    fn a_record_taken_out_of_a_database_outlives_it() {
        let mut from = db();
        from.sadd(b"s", [b"m1".as_ref(), b"m2".as_ref()].into_iter())
            .expect("a set");
        let rec = from.export(b"s").expect("a record");
        assert_eq!(rec.kind(), Kind::Set);
        from.clear();

        let mut into = db();
        into.import(b"s", rec);
        assert_eq!(members(&mut into, b"s"), ["m1", "m2"]);
    }

    #[test]
    fn importing_over_a_body_does_not_leave_it_in_the_slab() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref()].into_iter()).expect("a set");
        d.sadd(b"t", [b"m2".as_ref()].into_iter()).expect("a set");
        let rec = d.export(b"s").expect("a record");

        d.import(b"t", rec);
        assert_eq!(d.sets.len(), 2, "s and t, and not the one t used to hold");
        assert_eq!(members(&mut d, b"t"), ["m1"]);
    }

    #[test]
    fn importing_a_string_over_a_set_frees_the_set() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        d.sadd(b"s", [b"m1".as_ref()].into_iter()).expect("a set");
        assert_eq!(d.sets.len(), 1);

        assert_eq!(d.copy(b"a", b"s", true), Moved::Ok);
        assert_eq!(d.sets.len(), 0, "the set went when the string arrived");
        assert_eq!(d.kind_of(b"s"), Some(Kind::String));
    }

    /// `COPY` of a list, which used to take the server down with it.
    ///
    /// The catch all arm at the bottom of `export` was written when a set and a
    /// hash were the only bodies there were, and the list and the sorted set
    /// arrived past it without anybody coming back here. So `COPY mylist other`
    /// reached `unreachable!` and panicked the shard, from a command any client
    /// can send, against a type the server otherwise supports completely.
    ///
    /// The copy has to be a copy and not a second name for the same body, which
    /// is the other half of what this checks: pushing to the destination must
    /// not show up in the source.
    #[test]
    fn a_list_can_be_copied_and_the_copy_is_its_own() {
        let mut d = db();
        d.push(b"l", End::Left, [b"a".as_ref(), b"b".as_ref()].into_iter())
            .expect("a list");

        assert_eq!(d.copy(b"l", b"m", false), Moved::Ok);
        assert_eq!(d.kind_of(b"m"), Some(Kind::List));
        assert_eq!(d.llen(b"m").expect("a list"), 2);

        d.push(b"m", End::Left, [b"c".as_ref()].into_iter())
            .expect("a list");
        assert_eq!(d.llen(b"l").expect("a list"), 2, "the source did not grow");
        assert_eq!(d.llen(b"m").expect("a list"), 3);
    }

    /// The same for a sorted set, which had the same hole for the same reason.
    #[test]
    fn a_zset_can_be_copied_and_the_copy_is_its_own() {
        let mut d = db();
        d.zadd(b"z", [(1.0, b"m1".as_ref())].into_iter(), ZAdd::default())
            .expect("a zset");

        assert_eq!(d.copy(b"z", b"y", false), Moved::Ok);
        assert_eq!(d.kind_of(b"y"), Some(Kind::Zset));
        assert_eq!(d.zscore(b"y", b"m1").expect("a zset"), Some(1.0));

        d.zadd(b"y", [(2.0, b"m2".as_ref())].into_iter(), ZAdd::default())
            .expect("a zset");
        assert_eq!(d.zcard(b"z").expect("a zset"), 1, "the source did not grow");
        assert_eq!(d.zcard(b"y").expect("a zset"), 2);
    }

    /// A copy over a key that held a list gives the list back.
    ///
    /// The leak this guards against is the same one the set version guards
    /// against: a record written over a body that nothing freed leaves a slab
    /// slot reachable and never reused, and nothing about the server looks wrong
    /// afterwards.
    #[test]
    fn copying_over_a_list_frees_the_list() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        d.push(b"l", End::Left, [b"x".as_ref()].into_iter())
            .expect("a list");

        assert_eq!(d.copy(b"a", b"l", true), Moved::Ok);
        assert_eq!(d.kind_of(b"l"), Some(Kind::String));
        assert_eq!(read(&mut d, b"l"), b"v1");
    }

    /// The whole reason `take` exists: the body arrives without being cloned and
    /// the slab it came out of is empty afterwards.
    #[test]
    fn taking_a_set_empties_the_slab_and_the_key() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref(), b"m2".as_ref()].into_iter())
            .expect("a set");
        assert_eq!(d.sets.len(), 1);

        let rec = d.take(b"s").expect("a record");
        assert_eq!(rec.kind(), Kind::Set);
        assert_eq!(d.sets.len(), 0, "the body left with the record");
        assert!(!d.exists(b"s"), "and so did the key");

        let mut into = db();
        into.import(b"s", rec);
        assert_eq!(members(&mut into, b"s"), ["m1", "m2"]);
    }

    /// A string has no slab slot, so the bytes are copied and the count is left
    /// alone. Taking one and then taking it again answers nothing the second
    /// time, which is the check that the record went too.
    #[test]
    fn taking_a_string_takes_the_record_with_it() {
        let mut d = db();
        put(&mut d, b"a", b"v1");

        let rec = d.take(b"a").expect("a record");
        assert_eq!(rec.kind(), Kind::String);
        assert!(d.take(b"a").is_none());
        assert_eq!(d.len(), 0);
    }

    /// The deadline travels, the same as it does through `export`.
    #[test]
    fn a_taken_key_keeps_the_time_it_had_left() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        assert_eq!(d.expire(b"a", 2_000_000, Cond::Always), Applied::Ok);

        let rec = d.take(b"a").expect("a record");
        assert_eq!(rec.expire_at(), Some(2_000_000));
    }

    /// A key past its deadline is not there to take, which is the reaping every
    /// other read does and not a special case here.
    #[test]
    fn a_dead_key_cannot_be_taken() {
        let mut d = db();
        d.sadd(b"s", [b"m1".as_ref()].into_iter()).expect("a set");
        assert_eq!(d.expire(b"s", 1_000_001, Cond::Always), Applied::Ok);
        d.clock().advance(10);

        assert!(d.take(b"s").is_none());
        assert_eq!(d.sets.len(), 0, "and the body did not stay behind");
    }

    #[test]
    fn touch_counts_the_way_exists_counts() {
        let mut d = db();
        put(&mut d, b"a", b"v1");
        put(&mut d, b"b", b"v2");

        assert_eq!(d.touch([b"a".as_ref()].into_iter()), 1);
        assert_eq!(d.touch([b"a".as_ref(), b"b".as_ref()].into_iter()), 2);
        assert_eq!(d.touch([b"a".as_ref(), b"a".as_ref()].into_iter()), 2);
        assert_eq!(d.touch([b"a".as_ref(), b"z".as_ref()].into_iter()), 1);
        assert_eq!(d.touch([b"z".as_ref()].into_iter()), 0);
    }
}
