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
//! It also makes the pair the answer for `MOVE`, `DUMP` and `RESTORE` when they
//! land, which want exactly this: a value lifted out of a database, standing on
//! its own with its deadline attached.

use yo_common::Result;

use crate::hash::Hash;
use crate::keyspace::Keyspace;
use crate::set::Set;
use crate::value::{self, Kind};

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
    /// What type this is, which the caller usually knows and sometimes does not.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        match self.body {
            Body::String(_) => Kind::String,
            Body::Set(_) => Kind::Set,
            Body::Hash(_) => Kind::Hash,
        }
    }

    /// When it goes away, if anything says.
    #[must_use]
    pub const fn expire_at(&self) -> Option<u64> {
        self.expire_at
    }
}

/// The three things a record can be, owned rather than borrowed.
#[derive(Debug, Clone)]
enum Body {
    String(Vec<u8>),
    Set(Set),
    Hash(Hash),
}

/// What a rename or a copy did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moved {
    /// There was no source key, so there was nothing to move.
    Missing,
    /// The destination was there and the caller said not to write over it.
    Taken,
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
        let addr = self.live_rec(key)?;
        let rec = self.map.value_at(addr);
        let expire_at = value::expire_at(rec);
        // The slot is read inside the arms and not before them. A string record
        // holds the string and not a slot, so reading four bytes where the slot
        // would be reads off the end of a short one.
        let body = match value::kind(rec) {
            Kind::String => Body::String(value::read(rec).to_vec()),
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
            other => unreachable!("nothing can store a {} yet", other.name()),
        };
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
        }
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
        let addr = self.map.find(src).expect("it was live a line ago");
        let bytes = self.map.value_at(addr).to_vec();
        self.free_body(dst);
        self.map.set_with(dst, bytes.len(), |out| {
            out.copy_from_slice(&bytes);
        });
        // `del` and not `drop_key`, which is the whole point. The body under the
        // source belongs to the destination now and freeing it here would take
        // it away from the key that just gained it.
        self.map.del(src);
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
    pub fn copy(&mut self, src: &[u8], dst: &[u8], replace: bool) -> Moved {
        let Some(rec) = self.export(src) else {
            return Moved::Missing;
        };
        if !replace && self.live_rec(dst).is_some() {
            return Moved::Taken;
        }
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
        self.map.set_with(key, len, |out| {
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
