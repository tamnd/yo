//! The set commands.
//!
//! One method per Redis command on [`Keyspace`], the same arrangement the string
//! commands use and for the same reason: a key belongs to the database and not
//! to a type, so `SADD` against a string has to be able to see that it is a
//! string. The set itself, and the choice between the three representations it
//! can be in, is [`crate::set`]. This file is what the wire and the embedded API
//! both call.
//!
//! # Where a set lives
//!
//! The record under the key holds a type tag and four bytes saying which slot of
//! the database's slab the body is in, and that is all. Reaching a set is one
//! key lookup and then one dependent load, and the dependent load is
//! unavoidable because a set outgrows a record and outlives any one command.
//!
//! Two invariants hold this together and both of them are about not leaking.
//! Every path that deletes a key goes through `drop_key` and every path that
//! writes over one goes through `free_body`, so a set cannot lose its record
//! while keeping its slot. And a set that loses its last member is deleted
//! rather than stored empty, because an empty set does not exist in Redis:
//! `SREM` taking the last member makes `EXISTS` answer zero.
//!
//! # Errors
//!
//! Every command here answers `WRONGTYPE` for a key holding something that is
//! not a set, and treats a missing key as an empty one. That pair of rules is
//! Redis's and between them they cover every case, because a key is a set, or
//! another type, or absent.

use yo_common::Result;

use crate::keyspace::{Keyspace, wrong_type};
use crate::set::{Member, Set};
use crate::strings;
use crate::value::{self, Kind};

impl Keyspace {
    /// `SADD key member [member ...]`. Answers how many were new.
    ///
    /// The members arrive as an iterator and not a slice, the way `MSET`'s pairs
    /// do, because the wire layer has them as positions in the connection's read
    /// buffer and a slice would mean collecting them first. A shard thread that
    /// allocates in order to call a command is the thing Y1 is trying to avoid.
    /// The iterator is walked more than once, which is why it has to be `Clone`,
    /// and an iterator over borrowed slices is two words to copy.
    pub fn sadd<'m>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]> + Clone,
    ) -> Result<usize> {
        for m in members.clone() {
            strings::check_len(key, m.len())?;
        }
        let at = match self.set_slot(key)? {
            Some(at) => at,
            None => {
                // Nothing to add to a key that does not exist yet is not a
                // reason to create it. Redis's parser rejects `SADD k` before it
                // gets this far, but the embedded API has no parser in front of
                // it and an empty set left behind would be a key that exists and
                // holds nothing.
                let Some(first) = members.clone().next() else {
                    return Ok(0);
                };
                let hint = members.clone().count();
                self.new_set(key, first, hint)
            }
        };

        // The limits are three numbers and copying them out is what lets the
        // body be borrowed mutably for the whole loop instead of once a member.
        let limits = self.limits;
        let set = self
            .sets
            .get_mut(at)
            .expect("the record points at its body");
        let mut added = 0;
        for m in members {
            if set.add(m, &limits) {
                added += 1;
            }
        }
        Ok(added)
    }

    /// `SREM key member [member ...]`. Answers how many were there.
    ///
    /// A set that loses its last member loses its key too.
    pub fn srem<'m>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]>,
    ) -> Result<usize> {
        let Some(at) = self.set_slot(key)? else {
            return Ok(0);
        };
        let set = self
            .sets
            .get_mut(at)
            .expect("the record points at its body");
        let mut gone = 0;
        for m in members {
            if set.remove(m) {
                gone += 1;
            }
        }
        if set.is_empty() {
            self.drop_key(key);
        }
        Ok(gone)
    }

    /// `SISMEMBER key member`.
    pub fn sismember(&mut self, key: &[u8], member: &[u8]) -> Result<bool> {
        match self.set_slot(key)? {
            Some(at) => Ok(self.set_at(at).contains(member)),
            None => Ok(false),
        }
    }

    /// `SMISMEMBER key member [member ...]`, which is `SISMEMBER` in bulk.
    ///
    /// One key lookup for the whole call rather than one per member, which is
    /// the only reason the command exists.
    pub fn smismember<'m>(
        &mut self,
        key: &[u8],
        members: impl Iterator<Item = &'m [u8]>,
    ) -> Result<Vec<bool>> {
        let Some(at) = self.set_slot(key)? else {
            return Ok(members.map(|_| false).collect());
        };
        let set = self.set_at(at);
        Ok(members.map(|m| set.contains(m)).collect())
    }

    /// `SCARD key`, which is zero for a key that is not there.
    pub fn scard(&mut self, key: &[u8]) -> Result<usize> {
        match self.set_slot(key)? {
            Some(at) => Ok(self.set_at(at).len()),
            None => Ok(0),
        }
    }

    /// `SMEMBERS key`, as a borrow of the set rather than a copy of it.
    ///
    /// The members come back as [`Member`]s, which are either the bytes where
    /// they lie or an integer nobody has formatted yet, so a set of a thousand
    /// integers becomes a thousand pieces of reply text and not a thousand
    /// `Vec`s that are then copied into the reply and dropped. That is Y18, and
    /// it is why this borrows the database for as long as the answer is alive.
    pub fn smembers(&mut self, key: &[u8]) -> Result<Option<impl Iterator<Item = Member<'_>>>> {
        let Some(at) = self.set_slot(key)? else {
            return Ok(None);
        };
        Ok(Some(self.set_at(at).iter()))
    }

    /// Hand the set under `key` to `f`, or hand it `None` if there is no key.
    ///
    /// This is what the wire layer reaches for when one command wants the body
    /// more than once. `SMEMBERS` needs the count for the reply header and then
    /// the members, and `SMISMEMBER` needs one membership test per argument, and
    /// going back through [`Keyspace::scard`] and [`Keyspace::sismember`] for
    /// each of those is a key lookup a piece. One lookup, then a borrow of the
    /// body for as long as the caller needs it.
    ///
    /// It is a callback rather than a returned `&Set` because the reap has to
    /// happen under `&mut self` and the borrow checker will not let a `&Set`
    /// carved out of that outlive the call.
    pub fn with_set<R>(&mut self, key: &[u8], f: impl FnOnce(Option<&Set>) -> R) -> Result<R> {
        let at = self.set_slot(key)?;
        Ok(f(at.map(|at| self.set_at(at))))
    }

    /// The slot holding the set under `key`, having reaped a dead key first.
    ///
    /// `None` for a key that is not there, an error for a key holding something
    /// that is not a set. Every command above starts here, so the three cases a
    /// key can be in are decided once.
    fn set_slot(&mut self, key: &[u8]) -> Result<Option<u32>> {
        self.reap(key);
        match self.map.get(key) {
            None => Ok(None),
            Some(rec) if value::kind(rec) == Kind::Set => Ok(Some(value::slot(rec))),
            Some(_) => Err(wrong_type()),
        }
    }

    /// The body in a slot the record pointed at.
    ///
    /// Panicking here means a record outlived its body, which is the one bug the
    /// slab deliberately does not carry a generation counter to catch, so this
    /// is where it would be caught instead.
    #[inline]
    fn set_at(&self, at: u32) -> &Set {
        self.sets.get(at).expect("the record points at its body")
    }

    /// Make an empty set under `key` and answer which slot it went in.
    ///
    /// `first` and `hint` only pick the representation to start in, following
    /// Redis's `setTypeCreate`, so that a `SADD` with a thousand arguments
    /// builds a table once instead of converting twice on the way there.
    fn new_set(&mut self, key: &[u8], first: &[u8], hint: usize) -> u32 {
        let at = self.sets.insert(Set::with_hint(first, hint, &self.limits));
        let len = value::slot_record_len(false);
        self.map.set_with(key, len, |out| {
            value::write_slot_record(out, Kind::Set, at, None);
        });
        self.bodies += 1;
        at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Clock;
    use crate::set::Encoding;
    use yo_common::Code;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    fn add(d: &mut Keyspace, key: &[u8], members: &[&[u8]]) -> usize {
        d.sadd(key, members.iter().copied()).expect("a set")
    }

    fn members(d: &mut Keyspace, key: &[u8]) -> Vec<String> {
        let mut v: Vec<String> = d
            .smembers(key)
            .expect("a set")
            .expect("a key")
            .map(|m| String::from_utf8(m.to_vec()).expect("utf8 in these tests"))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn adding_to_a_key_that_is_not_there_makes_it() {
        let mut d = db();
        assert_eq!(add(&mut d, b"s", &[b"a", b"b", b"c"]), 3);
        assert_eq!(d.scard(b"s").expect("a set"), 3);
        assert_eq!(d.kind_of(b"s"), Some(Kind::Set));
        assert_eq!(members(&mut d, b"s"), ["a", "b", "c"]);
        assert_eq!(d.len(), 1, "one key, whatever the set holds");
    }

    #[test]
    fn adding_answers_how_many_were_new_and_not_how_many_arrived() {
        let mut d = db();
        assert_eq!(add(&mut d, b"s", &[b"a", b"b"]), 2);
        assert_eq!(add(&mut d, b"s", &[b"b", b"c"]), 1);
        assert_eq!(
            add(&mut d, b"s", &[b"x", b"x", b"x"]),
            1,
            "the same member three times in one call is one member"
        );
        assert_eq!(d.scard(b"s").expect("a set"), 4);
    }

    #[test]
    fn everything_answers_for_a_key_that_is_not_there() {
        let mut d = db();
        assert_eq!(d.scard(b"nope").expect("missing is fine"), 0);
        assert!(!d.sismember(b"nope", b"a").expect("missing is fine"));
        assert!(d.smembers(b"nope").expect("missing is fine").is_none());
        assert_eq!(
            d.srem(b"nope", [b"a".as_slice()].into_iter()).expect("ok"),
            0
        );
        assert_eq!(
            d.smismember(b"nope", [b"a".as_slice(), b"b"].into_iter())
                .expect("ok"),
            [false, false]
        );
        assert_eq!(d.len(), 0, "and none of that created anything");
    }

    #[test]
    fn membership_answers_for_members_and_strangers() {
        let mut d = db();
        add(&mut d, b"s", &[b"a", b"b"]);
        assert!(d.sismember(b"s", b"a").expect("a set"));
        assert!(!d.sismember(b"s", b"z").expect("a set"));
        assert_eq!(
            d.smismember(b"s", [b"a".as_slice(), b"z", b"b"].into_iter())
                .expect("a set"),
            [true, false, true]
        );
    }

    #[test]
    fn removing_the_last_member_removes_the_key() {
        // An empty set does not exist in Redis and it does not exist here.
        let mut d = db();
        add(&mut d, b"s", &[b"a", b"b"]);
        assert_eq!(d.srem(b"s", [b"a".as_slice()].into_iter()).expect("ok"), 1);
        assert!(d.exists(b"s"), "one member left");

        assert_eq!(
            d.srem(b"s", [b"b".as_slice(), b"gone"].into_iter())
                .expect("ok"),
            1,
            "one of the two was there"
        );
        assert!(!d.exists(b"s"), "and now the key is gone with it");
        assert_eq!(d.kind_of(b"s"), None);
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn a_set_is_deleted_body_and_all() {
        // The leak this guards against is invisible from the outside: the key
        // goes, the slot does not, and nothing ever notices. So the test asks
        // the slab directly, because that is the only place the answer shows.
        let mut d = db();
        add(&mut d, b"s", &[b"a", b"b"]);
        assert_eq!(d.sets.len(), 1);

        assert!(d.del(b"s"));
        assert_eq!(d.sets.len(), 0, "the body went with the key");
        assert_eq!(d.bodies, 0);

        // And the slot is reused rather than abandoned.
        add(&mut d, b"t", &[b"x"]);
        assert_eq!(d.sets.len(), 1);
    }

    #[test]
    fn writing_a_string_over_a_set_takes_the_body_with_it() {
        // SET is allowed to overwrite any type, so this is not WRONGTYPE. What
        // it must not be is a set left in the slab with nothing pointing at it.
        let mut d = db();
        add(&mut d, b"k", &[b"a", b"b"]);
        assert_eq!(d.sets.len(), 1);

        d.set_plain(b"k", b"now a string").expect("room");
        assert_eq!(d.sets.len(), 0, "the set went when it was written over");
        assert_eq!(d.bodies, 0);
        assert_eq!(d.kind_of(b"k"), Some(Kind::String));
        assert_eq!(
            d.get(b"k").expect("a string").map(|v| v.to_vec()),
            Some(b"now a string".to_vec())
        );
    }

    #[test]
    fn a_set_that_expires_takes_its_body_with_it() {
        let mut d = db();
        add(&mut d, b"s", &[b"a"]);
        assert!(d.set_expiry(b"s", Some(1_100)));
        assert_eq!(d.expire_at(b"s"), Some(1_100));
        assert_eq!(d.sets.len(), 1);
        assert_eq!(d.scard(b"s").expect("a set"), 1, "still alive at 1000");

        d.clock_mut().advance(100);
        assert_eq!(d.scard(b"s").expect("gone is not an error"), 0);
        assert_eq!(d.sets.len(), 0, "reaping freed the body");
        assert_eq!(d.bodies, 0);
        assert_eq!(d.expired_keys(), 1);
    }

    #[test]
    fn flushing_takes_every_body_with_it() {
        let mut d = db();
        for i in 0..10 {
            add(&mut d, format!("s{i}").as_bytes(), &[b"a", b"b"]);
        }
        assert_eq!(d.sets.len(), 10);

        d.clear();
        assert_eq!(d.sets.len(), 0);
        assert_eq!(d.bodies, 0);
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn a_set_command_at_a_string_is_wrongtype() {
        let mut d = db();
        d.set_plain(b"k", b"v").expect("room");

        let err = d.sadd(b"k", [b"a".as_slice()].into_iter()).expect_err("no");
        assert_eq!(err.code(), Code::WrongType);
        assert_eq!(
            err.message(),
            "Operation against a key holding the wrong kind of value"
        );
        assert!(d.scard(b"k").is_err());
        assert!(d.sismember(b"k", b"a").is_err());
        assert!(d.smembers(b"k").is_err());
        assert!(d.srem(b"k", [b"a".as_slice()].into_iter()).is_err());
        assert!(d.smismember(b"k", [b"a".as_slice()].into_iter()).is_err());
        assert_eq!(
            d.get(b"k").expect("still a string").map(|v| v.to_vec()),
            Some(b"v".to_vec()),
            "and none of that damaged it"
        );
    }

    #[test]
    fn a_string_command_at_a_set_is_wrongtype() {
        let mut d = db();
        add(&mut d, b"s", &[b"a"]);

        assert_eq!(d.get(b"s").expect_err("no").code(), Code::WrongType);
        assert!(d.strlen(b"s").is_err());
        assert!(d.getrange(b"s", 0, -1).is_err());
        assert!(d.getdel(b"s").is_err());
        assert!(d.incr(b"s").is_err());
        assert!(d.append(b"s", b"x").is_err());
        assert_eq!(d.scard(b"s").expect("a set"), 1, "and it is still a set");
    }

    #[test]
    fn the_commands_that_do_not_care_still_do_not_care() {
        // EXISTS, DEL, TYPE and the TTL commands work on any type in Redis, and
        // a WRONGTYPE from one of them would be a bug and not a strictness.
        let mut d = db();
        add(&mut d, b"s", &[b"a"]);

        assert!(d.exists(b"s"));
        assert_eq!(d.kind_of(b"s"), Some(Kind::Set));
        assert_eq!(d.encoding_name(b"s"), Some("listpack"));
        assert!(d.set_expiry(b"s", Some(6_000)));
        assert_eq!(d.expire_at(b"s"), Some(6_000));
        assert!(d.set_expiry(b"s", None), "and PERSIST takes it off again");
        assert_eq!(d.expire_at(b"s"), None);
        assert_eq!(d.scard(b"s").expect("a set"), 1, "through all of that");
        assert!(d.del(b"s"));
    }

    #[test]
    fn mget_says_nil_for_a_set_rather_than_failing() {
        // The one string command that does not answer WRONGTYPE. Redis
        // documents MGET as giving nil for a key of the wrong type, because the
        // alternative is one bad key failing a hundred good ones.
        let mut d = db();
        d.set_plain(b"a", b"1").expect("room");
        add(&mut d, b"s", &[b"x"]);
        d.set_plain(b"z", b"2").expect("room");

        let got: Vec<Option<Vec<u8>>> = d
            .mget(&[b"a", b"s", b"z", b"nope"])
            .into_iter()
            .map(|v| v.map(|s| s.to_vec()))
            .collect();
        assert_eq!(got, [Some(b"1".to_vec()), None, Some(b"2".to_vec()), None]);
    }

    #[test]
    fn the_representation_follows_the_members_through_the_keyspace() {
        // The same ladder set.rs tests, but reached the way a client reaches it,
        // to prove the body that gets promoted is the body the record points at
        // and not a copy that was left behind.
        let mut d = db();
        add(&mut d, b"s", &[b"1", b"2", b"3"]);
        assert_eq!(d.set_encoding(b"s"), Some(Encoding::Intset));

        add(&mut d, b"s", &[b"hello"]);
        assert_eq!(d.set_encoding(b"s"), Some(Encoding::Listpack));
        assert_eq!(members(&mut d, b"s"), ["1", "2", "3", "hello"]);

        let long: Vec<u8> = vec![b'z'; 100];
        add(&mut d, b"s", &[&long]);
        assert_eq!(d.set_encoding(b"s"), Some(Encoding::Hashtable));
        assert_eq!(d.scard(b"s").expect("a set"), 5);
        assert!(d.sismember(b"s", b"1").expect("a set"), "nothing was lost");
        assert!(d.sismember(b"s", &long).expect("a set"));
    }

    #[test]
    fn a_thousand_members_at_once_builds_a_table_without_converting() {
        let mut d = db();
        let owned: Vec<Vec<u8>> = (0..1000).map(|i| format!("m{i}").into_bytes()).collect();
        let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        assert_eq!(d.sadd(b"s", refs.iter().copied()).expect("a set"), 1000);
        assert_eq!(d.set_encoding(b"s"), Some(Encoding::Hashtable));
        assert_eq!(d.scard(b"s").expect("a set"), 1000);
    }

    #[test]
    fn a_set_is_counted_in_what_the_database_is_holding() {
        let mut d = db();
        let before = d.memory_bytes();
        let owned: Vec<Vec<u8>> = (0..500).map(|i| i.to_string().into_bytes()).collect();
        let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        d.sadd(b"s", refs.iter().copied()).expect("a set");

        let after = d.memory_bytes();
        assert!(
            after > before + 500,
            "five hundred members have to show up somewhere: {before} then {after}"
        );
        d.del(b"s");
        assert!(d.memory_bytes() < after, "and go away again");
    }
}
