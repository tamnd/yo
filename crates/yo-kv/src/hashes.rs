//! The hash commands.
//!
//! One method per Redis command on [`Keyspace`], the same arrangement the set
//! and string commands use. The hash itself, and the choice between the two
//! representations it can be in, is [`crate::hash`]. This file is what the wire
//! and the embedded API both call.
//!
//! # Where a hash lives
//!
//! Exactly where a set lives. The record under the key holds a type tag and four
//! bytes saying which slot of the database's hash slab the body is in, and
//! reaching it is one key lookup and one dependent load. The two slabs are
//! separate rather than one slab of an enum, because the record's tag already
//! says which one to look in and a discriminant on the body would be a second
//! copy of a fact that is already there.
//!
//! The same two invariants hold, and both are about not leaking. Every path that
//! deletes a key goes through `drop_key` and every path that writes over one
//! goes through `free_body`. And a hash that loses its last field is deleted
//! rather than stored empty, because an empty hash does not exist in Redis:
//! `HDEL` taking the last field makes `EXISTS` answer zero.
//!
//! # Returning a field's value
//!
//! A value in the listpack band may be stored as an integer, so there is no
//! `&[u8]` to hand back for it without writing the digits somewhere first. The
//! reading commands take a closure and hand it a [`Text`] instead, which the
//! reply layer formats straight into the output buffer. That is Y18, and it is
//! why `HGET` is not simply `-> Option<&[u8]>`.
//!
//! # Errors
//!
//! Every command here answers `WRONGTYPE` for a key holding something that is
//! not a hash, and treats a missing key as an empty one.

use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};

use crate::hash::{Hash, Text};
use crate::keyspace::{Keyspace, wrong_type};
use crate::scan::Cursor;
use crate::strings;
use crate::ttl::{self, Applied, Ask, Cond};
use crate::value::{self, Kind};

/// What Redis says when a field does not hold a number.
const NOT_AN_INT: &str = "hash value is not an integer";
/// And when it does not hold a float.
const NOT_A_FLOAT: &str = "hash value is not a float";
/// And when the sum leaves the range.
const WOULD_OVERFLOW: &str = "increment or decrement would overflow";
/// And when a field deadline lands past the year it stops fitting.
const BAD_EXPIRE: &str = "invalid expire time, must be >= 0";

impl Keyspace {
    /// `HSET key field value [field value ...]`. Answers how many were new.
    ///
    /// The pairs arrive as an iterator for the reason `SADD`'s members do: the
    /// wire layer has them as positions in the connection's read buffer, and
    /// collecting them into a slice first would be an allocation per command on
    /// a shard thread.
    ///
    /// Redis's parser rejects an odd number of arguments before this is reached.
    /// The embedded API has no parser in front of it, so an empty iterator does
    /// not create the key, the same guard `SADD` has.
    pub fn hset<'a>(
        &mut self,
        key: &[u8],
        pairs: impl Iterator<Item = (&'a [u8], &'a [u8])> + Clone,
    ) -> Result<usize> {
        for (f, v) in pairs.clone() {
            strings::check_len(key, f.len())?;
            strings::check_len(key, v.len())?;
        }
        let at = match self.hash_slot(key)? {
            Some(at) => at,
            None => {
                if pairs.clone().next().is_none() {
                    return Ok(0);
                }
                let hint = pairs.clone().count();
                self.new_hash(key, hint)
            }
        };

        // Copied out so the body can be borrowed mutably for the whole loop
        // rather than once a pair.
        let limits = self.hash_limits;
        let hash = self
            .hashes
            .get_mut(at)
            .expect("the record points at its body");
        let mut added = 0;
        for (field, value) in pairs {
            if hash.set(field, value, &limits) {
                added += 1;
            }
        }
        Ok(added)
    }

    /// `HSETNX key field value`. Answers whether it was written.
    ///
    /// Unlike `SETNX` this is per field and not per key, so it writes into a
    /// hash that already exists as long as that one field is missing.
    pub fn hsetnx(&mut self, key: &[u8], field: &[u8], value: &[u8]) -> Result<bool> {
        strings::check_len(key, field.len())?;
        strings::check_len(key, value.len())?;
        let at = match self.hash_slot(key)? {
            Some(at) => {
                if self.hash_at(at).contains(field) {
                    return Ok(false);
                }
                at
            }
            None => self.new_hash(key, 1),
        };
        let limits = self.hash_limits;
        self.hashes
            .get_mut(at)
            .expect("the record points at its body")
            .set(field, value, &limits);
        Ok(true)
    }

    /// `HGET key field`, as a borrow rather than a copy.
    ///
    /// `f` is handed `None` for a missing key and for a missing field alike,
    /// because both are a nil reply and the caller has no reason to tell them
    /// apart. `HEXISTS` is the command that does.
    pub fn hget<R>(
        &mut self,
        key: &[u8],
        field: &[u8],
        f: impl FnOnce(Option<Text<'_>>) -> R,
    ) -> Result<R> {
        let Some(at) = self.hash_slot(key)? else {
            return Ok(f(None));
        };
        Ok(f(self.hash_at(at).get(field)))
    }

    /// `HMGET key field [field ...]`, one call of `f` per field asked for.
    ///
    /// Every field gets a call, including the ones that are not there, because
    /// the reply is positional: a client sending three fields gets three
    /// entries back and matches them up by position. A missing key answers all
    /// nils rather than an empty array for the same reason.
    pub fn hmget<'a, F>(
        &mut self,
        key: &[u8],
        fields: impl Iterator<Item = &'a [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Option<Text<'_>>),
    {
        let slot = self.hash_slot(key)?;
        for field in fields {
            match slot {
                Some(at) => f(self.hash_at(at).get(field)),
                None => f(None),
            }
        }
        Ok(())
    }

    /// `HDEL key field [field ...]`. Answers how many were there.
    ///
    /// The key goes when the last field does.
    pub fn hdel<'a>(
        &mut self,
        key: &[u8],
        fields: impl Iterator<Item = &'a [u8]>,
    ) -> Result<usize> {
        let Some(at) = self.hash_slot(key)? else {
            return Ok(0);
        };
        let hash = self
            .hashes
            .get_mut(at)
            .expect("the record points at its body");
        let mut gone = 0;
        for field in fields {
            if hash.remove(field) {
                gone += 1;
            }
        }
        if hash.is_empty() {
            self.drop_key(key);
        }
        Ok(gone)
    }

    /// `HEXPIREAT` and the three commands that turn into it.
    ///
    /// `at` is an absolute unix millisecond, which is what `HEXPIRE`,
    /// `HPEXPIRE` and `HEXPIREAT` all become before they get here, and one call
    /// of `f` happens per field asked for because the reply is positional.
    ///
    /// The deadline is checked against [`ttl::MAX_AT`] before any field is
    /// touched, because Redis rejects the whole command rather than failing
    /// field by field, and a command that names ten fields either sets all ten
    /// or errors.
    ///
    /// A key that is not there answers [`Applied::NoField`] for every field,
    /// which is the -2 Redis replies, because a missing key and an empty hash
    /// are the same thing. The key goes when the last field does, which happens
    /// when the deadline given has already passed.
    pub fn hexpire<'a, F>(
        &mut self,
        key: &[u8],
        at: u64,
        cond: Cond,
        fields: impl Iterator<Item = &'a [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Applied),
    {
        if !ttl::valid_at(at) {
            return Err(Error::new(Code::Invalid, BAD_EXPIRE));
        }
        let Some(slot) = self.hash_slot(key)? else {
            for _ in fields {
                f(Applied::NoField);
            }
            return Ok(());
        };
        let now = self.clock.now_ms();
        let mut emptied = false;
        for field in fields {
            let hash = self.hash_at_mut(slot);
            let applied = hash.expire(field, at, cond, now);
            emptied = hash.is_empty();
            f(applied);
        }
        if emptied {
            self.drop_key(key);
        }
        Ok(())
    }

    /// `HTTL` and its relatives, one call of `f` per field asked for.
    ///
    /// What comes back is when the deadline falls due. Turning that into what is
    /// left, and into seconds where the command asks for seconds, is the reply
    /// layer's job, because [`Ask::remaining_ms`] is where that arithmetic lives
    /// and it needs the moment being asked at.
    pub fn httl<'a, F>(
        &mut self,
        key: &[u8],
        fields: impl Iterator<Item = &'a [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Ask),
    {
        let slot = self.hash_slot(key)?;
        for field in fields {
            match slot {
                Some(at) => f(self.hash_at(at).deadline(field)),
                None => f(Ask::NoField),
            }
        }
        Ok(())
    }

    /// `HPERSIST key FIELDS numfields field [field ...]`.
    ///
    /// [`Ask::At`] means the deadline that was there has been taken off, which
    /// the reply layer reports as 1.
    pub fn hpersist<'a, F>(
        &mut self,
        key: &[u8],
        fields: impl Iterator<Item = &'a [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Ask),
    {
        let slot = self.hash_slot(key)?;
        for field in fields {
            match slot {
                Some(at) => f(self.hash_at_mut(at).persist(field)),
                None => f(Ask::NoField),
            }
        }
        Ok(())
    }

    /// `HGETDEL key FIELDS numfields field [field ...]`.
    ///
    /// The value goes out and the field goes away, in that order, which is the
    /// whole command: a client that wants both without a race would otherwise
    /// send `HGET` and `HDEL` and hope. One call of `f` per field asked for,
    /// including the ones that were not there, because the reply is positional
    /// the way `HMGET`'s is.
    ///
    /// The key goes when the last field does.
    pub fn hgetdel<'a, F>(
        &mut self,
        key: &[u8],
        fields: impl Iterator<Item = &'a [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Option<Text<'_>>),
    {
        let Some(slot) = self.hash_slot(key)? else {
            for _ in fields {
                f(None);
            }
            return Ok(());
        };
        for field in fields {
            let hash = self.hash_at_mut(slot);
            f(hash.get(field));
            hash.remove(field);
        }
        if self.hash_at(slot).is_empty() {
            self.drop_key(key);
        }
        Ok(())
    }

    /// `HGETEX key [EX s | PX ms | EXAT ts | PXAT ts | PERSIST] FIELDS ...`.
    ///
    /// The read and the deadline change in one command, which is what makes it
    /// worth having: a plain `HSET` clears the deadline on the field it writes,
    /// so there is no way to touch a field's expiry and see its value with the
    /// commands that were there before.
    ///
    /// [`strings::Expire::Keep`] is a plain `HGETEX` with no option, and it is
    /// the default here rather than `Clear`, which is the one place this
    /// disagrees with `SET`. `Clear` is `PERSIST` and `At` is the other four.
    ///
    /// A deadline that has already gone deletes the field, and the value still
    /// goes out, because the read happened first. The key goes with the last
    /// field.
    pub fn hgetex<'a, F>(
        &mut self,
        key: &[u8],
        expire: strings::Expire,
        fields: impl Iterator<Item = &'a [u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Option<Text<'_>>),
    {
        // Before anything is read, because Redis rejects the whole command
        // rather than expiring the fields it got to first.
        check_at(expire)?;
        let Some(slot) = self.hash_slot(key)? else {
            for _ in fields {
                f(None);
            }
            return Ok(());
        };
        let now = self.clock.now_ms();
        for field in fields {
            let hash = self.hash_at_mut(slot);
            f(hash.get(field));
            match expire {
                strings::Expire::Keep => {}
                strings::Expire::Clear => {
                    hash.persist(field);
                }
                // A deadline that has already gone answers Deleted and takes the
                // field with it, which needs nothing here: the value went out
                // above, before the field did, and the empty check below is
                // what notices if that was the last one.
                strings::Expire::At(at) => {
                    hash.expire(field, at, Cond::Always, now);
                }
            }
        }
        if self.hash_at(slot).is_empty() {
            self.drop_key(key);
        }
        Ok(())
    }

    /// `HSETEX key [FNX | FXX] [EX .. | KEEPTTL] FIELDS n field value [..]`.
    ///
    /// Answers whether it wrote, which is all of it or none of it. `FNX` wants
    /// every field named to be missing and `FXX` wants every one of them to be
    /// there, so a list where one field disagrees writes nothing at all. That is
    /// stricter than `HSETNX`, which is per field, and it is what makes this
    /// usable as a compare and set over a group of fields.
    ///
    /// [`strings::Expire::Clear`] is a plain `HSETEX` and is the default, since
    /// a write clears the deadline on the field it writes anyway. `Keep` is
    /// `KEEPTTL` and has to put the deadline back afterwards for that reason.
    ///
    /// A deadline that has already gone still answers written, unlike the
    /// `HEXPIRE` family which has a separate code for it. The fields are stored
    /// and then removed, and if that empties the hash the key goes too, so
    /// `HSETEX key EXAT 1` on a key that did not exist leaves it not existing.
    pub fn hsetex<'a>(
        &mut self,
        key: &[u8],
        exists: strings::Exists,
        expire: strings::Expire,
        pairs: impl Iterator<Item = (&'a [u8], &'a [u8])> + Clone,
    ) -> Result<bool> {
        for (f, v) in pairs.clone() {
            strings::check_len(key, f.len())?;
            strings::check_len(key, v.len())?;
        }
        check_at(expire)?;

        let slot = self.hash_slot(key)?;
        // The condition is answered before a single field is written, because
        // it is about the whole list. A key that is not there has every field
        // missing, so FXX fails on it and FNX passes without creating it yet.
        let met = match exists {
            strings::Exists::Always => true,
            strings::Exists::IfMissing => {
                slot.is_none_or(|at| pairs.clone().all(|(f, _)| !self.hash_at(at).contains(f)))
            }
            strings::Exists::IfPresent => {
                slot.is_some_and(|at| pairs.clone().all(|(f, _)| self.hash_at(at).contains(f)))
            }
        };
        if !met {
            return Ok(false);
        }
        let slot = match slot {
            Some(at) => at,
            None => {
                if pairs.clone().next().is_none() {
                    return Ok(false);
                }
                self.new_hash(key, pairs.clone().count())
            }
        };

        let limits = self.hash_limits;
        let now = self.clock.now_ms();
        for (field, value) in pairs {
            let hash = self.hash_at_mut(slot);
            // KEEPTTL has to read the deadline first, because the write is what
            // clears it. There is no band where the value can be replaced with
            // the deadline left alone, and adding one would be a second way to
            // write a field.
            let kept = match expire {
                strings::Expire::Keep => hash.deadline(field),
                _ => Ask::NoField,
            };
            hash.set(field, value, &limits);
            match expire {
                strings::Expire::Clear => {}
                strings::Expire::Keep => {
                    if let Ask::At(at) = kept {
                        hash.expire(field, at, Cond::Always, now);
                    }
                }
                strings::Expire::At(at) => {
                    hash.expire(field, at, Cond::Always, now);
                }
            }
        }
        if self.hash_at(slot).is_empty() {
            self.drop_key(key);
        }
        Ok(true)
    }

    /// `HLEN key`.
    pub fn hlen(&mut self, key: &[u8]) -> Result<usize> {
        match self.hash_slot(key)? {
            Some(at) => Ok(self.hash_at(at).len()),
            None => Ok(0),
        }
    }

    /// `HEXISTS key field`.
    pub fn hexists(&mut self, key: &[u8], field: &[u8]) -> Result<bool> {
        match self.hash_slot(key)? {
            Some(at) => Ok(self.hash_at(at).contains(field)),
            None => Ok(false),
        }
    }

    /// `HSTRLEN key field`, without writing the value anywhere.
    ///
    /// A value held as an integer answers with how many digits it would take,
    /// counted rather than formatted, which is what [`Text::byte_len`] is for.
    pub fn hstrlen(&mut self, key: &[u8], field: &[u8]) -> Result<usize> {
        match self.hash_slot(key)? {
            Some(at) => Ok(self.hash_at(at).value_len(field).unwrap_or(0)),
            None => Ok(0),
        }
    }

    /// `HGETALL key`, `HKEYS key` and `HVALS key`, which differ only in what
    /// the caller does with each pair.
    ///
    /// One method for the three because the walk is the whole of the work and
    /// three copies of it would be three chances for one of them to drift. The
    /// caller taking a pair and using half of it costs nothing, since neither
    /// half is formatted until something asks for it.
    ///
    /// `Ok(false)` means the key was not there, which is an empty reply for all
    /// three and never a nil.
    pub fn hgetall<F>(&mut self, key: &[u8], mut f: F) -> Result<bool>
    where
        F: FnMut(Text<'_>, Text<'_>),
    {
        self.with_hash(key, |hash| match hash {
            Some(h) => {
                for (field, value) in h.iter() {
                    f(field, value);
                }
                true
            }
            None => false,
        })
    }

    /// Hand the hash under `key` to `f`, or hand it `None` if there is no key.
    ///
    /// The same thing [`Keyspace::with_set`] is for, and here it matters more.
    /// `HGETALL` on RESP3 answers a map, whose header carries the pair count, so
    /// the wire layer needs the length and then the pairs. Going back through
    /// [`Keyspace::hlen`] for the header would be a second key lookup on the
    /// command that is most likely to be in a loop.
    ///
    /// A callback rather than a returned `&Hash` because the reap happens under
    /// `&mut self` and a borrow carved out of that cannot outlive the call.
    pub fn with_hash<R>(&mut self, key: &[u8], f: impl FnOnce(Option<&Hash>) -> R) -> Result<R> {
        let at = self.hash_slot(key)?;
        Ok(f(at.map(|at| self.hash_at(at))))
    }

    /// `HSCAN key cursor [COUNT n]`, with the cursor to resume from.
    ///
    /// `NOVALUES` is the caller's business: it gets both halves and drops the
    /// one it does not want, exactly as `HKEYS` does.
    pub fn hscan<F>(&mut self, key: &[u8], cursor: Cursor, count: usize, f: F) -> Result<Cursor>
    where
        F: FnMut(Text<'_>, Text<'_>),
    {
        let Some(at) = self.hash_slot(key)? else {
            return Ok(Cursor::END);
        };
        Ok(self.hash_at(at).scan(cursor, count, f))
    }

    /// `HINCRBY key field increment`. Answers the sum.
    ///
    /// A field that is not there counts as zero and is created, which is what
    /// makes this the counter primitive it is used as. A field holding
    /// something that is not an integer is an error and leaves the hash exactly
    /// as it was, and so is a sum that leaves the range: Redis checks the
    /// overflow before the write rather than wrapping and storing the wrap.
    pub fn hincrby(&mut self, key: &[u8], field: &[u8], by: i64) -> Result<i64> {
        strings::check_len(key, field.len())?;
        let at = match self.hash_slot(key)? {
            Some(at) => at,
            None => self.new_hash(key, 1),
        };
        let current = match self.hash_at(at).get(field) {
            Some(Text::Int(n)) => n,
            Some(Text::Str(s)) => {
                parse_i64(s).ok_or_else(|| Error::new(Code::Invalid, NOT_AN_INT))?
            }
            None => 0,
        };
        let next = current
            .checked_add(by)
            .ok_or_else(|| Error::new(Code::Invalid, WOULD_OVERFLOW))?;

        let mut buf = [0u8; yo_common::num::DIGITS_MAX];
        let text = yo_common::num::i64_digits(&mut buf, next);
        let limits = self.hash_limits;
        self.hashes
            .get_mut(at)
            .expect("the record points at its body")
            .set(field, text, &limits);
        Ok(next)
    }

    /// `HINCRBYFLOAT key field increment`. Answers the sum.
    ///
    /// The same rules with the float versions of the errors. An infinite
    /// increment is not refused up front, for the reason `INCRBYFLOAT` gives:
    /// Redis parses it, does the addition and then reports that the result is
    /// not finite, so `HINCRBYFLOAT k f inf` says the increment would produce
    /// infinity and not that the increment is not a float.
    pub fn hincrbyfloat(&mut self, key: &[u8], field: &[u8], by: f64) -> Result<f64> {
        strings::check_len(key, field.len())?;
        let at = match self.hash_slot(key)? {
            Some(at) => at,
            None => self.new_hash(key, 1),
        };
        let current = match self.hash_at(at).get(field) {
            Some(Text::Int(n)) => n as f64,
            Some(Text::Str(s)) => {
                parse_f64(s).ok_or_else(|| Error::new(Code::Invalid, NOT_A_FLOAT))?
            }
            None => 0.0,
        };
        let next = current + by;
        if !next.is_finite() {
            return Err(Error::new(
                Code::Invalid,
                "increment would produce NaN or Infinity",
            ));
        }

        let mut text = Vec::with_capacity(32);
        yo_common::num::push_double(&mut text, next);
        let limits = self.hash_limits;
        self.hashes
            .get_mut(at)
            .expect("the record points at its body")
            .set(field, &text, &limits);
        Ok(next)
    }

    /// `HRANDFIELD key`, as a borrow.
    ///
    /// `f` is handed `None` when the key is not there, which is a nil and not
    /// an empty reply.
    pub fn hrandfield<R>(
        &mut self,
        key: &[u8],
        f: impl FnOnce(Option<(Text<'_>, Text<'_>)>) -> R,
    ) -> Result<R> {
        let Some(at) = self.hash_slot(key)? else {
            return Ok(f(None));
        };
        let pick = self.rng.below(self.hash_at(at).len());
        Ok(f(self.hash_at(at).at(pick)))
    }

    /// `HRANDFIELD key count`, which is two commands wearing one name.
    ///
    /// A negative count is the with repeats form: exactly that many fields,
    /// drawn one at a time, and the same field can come back more than once. It
    /// is the only form that can answer more fields than the hash holds.
    ///
    /// A positive count is distinct fields, at most as many as the hash holds.
    /// `SRANDMEMBER` splits its distinct form two ways because a set can be
    /// millions of members and drawing three of them should not walk all of
    /// them. A hash draws differently: Redis's own `HRANDFIELD` with a positive
    /// count builds the whole answer either way, so this walks the fields once
    /// and takes each with the probability that leaves the right number at the
    /// end. That is Knuth's selection sampling, it needs no memory at all, and
    /// it is `O(len)` rather than `O(count)`.
    ///
    /// A shuffle is deliberately not done. Redis does not promise an order here
    /// and the walk order is not the insertion order once a field has been
    /// removed, so shuffling would buy a guarantee nobody is owed at the price
    /// of an allocation.
    pub fn hrandfield_n<F>(&mut self, key: &[u8], count: i64, mut f: F) -> Result<()>
    where
        F: FnMut(Text<'_>, Text<'_>),
    {
        let Some(at) = self.hash_slot(key)? else {
            return Ok(());
        };
        // Borrowed apart rather than through `hash_at`, because drawing and
        // reading have to be alive at the same time and a method taking `&self`
        // would hold the whole database.
        let rng = &mut self.rng;
        let hash = self.hashes.get(at).expect("the record points at its body");
        let len = hash.len();

        let Ok(want) = usize::try_from(count) else {
            let repeats = usize::try_from(count.unsigned_abs()).unwrap_or(usize::MAX);
            for _ in 0..repeats {
                let (field, value) = hash
                    .at(rng.below(len))
                    .expect("the draw was under the length");
                f(field, value);
            }
            return Ok(());
        };

        let mut left = want.min(len);
        let mut seen = len;
        for i in 0..len {
            if left == 0 {
                break;
            }
            // Take this one with probability left/seen, which is what leaves
            // exactly `left` taken by the end whatever the draws come out as.
            if rng.below(seen) < left {
                let (field, value) = hash.at(i).expect("i is under the length");
                f(field, value);
                left -= 1;
            }
            seen -= 1;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ inside

    /// The slot `key`'s hash is in, or `None` if there is no such key.
    ///
    /// This is the one place a hash command finds its body, so it is the one
    /// place that has to reap first and answer `WRONGTYPE` for another type.
    fn hash_slot(&mut self, key: &[u8]) -> Result<Option<u32>> {
        self.reap(key);
        let at = match self.map.get(key) {
            None => return Ok(None),
            Some(rec) if value::kind(rec) == Kind::Hash => value::slot(rec),
            Some(_) => return Err(wrong_type()),
        };
        // And now the fields, which is the second half of lazy expiry. It runs
        // here rather than in every command so that there is one place a hash
        // becomes live, and it is a load and a comparison on a hash that has
        // never been given a field deadline, which is nearly all of them.
        let now = self.clock.now_ms();
        let hash = self
            .hashes
            .get_mut(at)
            .expect("the record points at its body");
        if hash.reap(now) > 0 && hash.is_empty() {
            // The last field expiring deletes the key, exactly as the last HDEL
            // does, because an empty hash is not a thing Redis stores.
            self.drop_key(key);
            return Ok(None);
        }
        Ok(Some(at))
    }

    /// The body in a slot the record pointed at, to be written.
    #[inline]
    fn hash_at_mut(&mut self, at: u32) -> &mut Hash {
        self.hashes
            .get_mut(at)
            .expect("the record points at its body")
    }

    /// The body in a slot the record pointed at.
    ///
    /// Panicking here means a record outlived its body, which is the one bug the
    /// slab deliberately does not carry a generation counter to catch, so this
    /// is where it would be caught instead.
    #[inline]
    fn hash_at(&self, at: u32) -> &Hash {
        self.hashes.get(at).expect("the record points at its body")
    }

    /// Make an empty hash under `key` and answer which slot it went in.
    ///
    /// The hint only picks the representation to start in, so that an `HSET`
    /// with a thousand pairs builds a table once instead of filling a listpack
    /// and then converting it.
    fn new_hash(&mut self, key: &[u8], hint: usize) -> u32 {
        let at = self.hashes.insert(Hash::with_hint(hint, &self.hash_limits));
        let len = value::slot_record_len(false);
        self.map.set_with(key, len, |out| {
            value::write_slot_record(out, Kind::Hash, at, None);
        });
        self.bodies += 1;
        at
    }
}

/// Refuses a deadline past the ceiling before the command touches anything.
///
/// Both `HGETEX` and `HSETEX` take the deadline as an option rather than as the
/// argument it is in the `HEXPIRE` family, and both have to answer for it
/// before they have read or written a field, since Redis refuses the whole
/// command rather than half doing it.
fn check_at(expire: strings::Expire) -> Result<()> {
    match expire {
        strings::Expire::At(at) if !ttl::valid_at(at) => Err(Error::new(Code::Invalid, BAD_EXPIRE)),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Clock;
    use crate::hash::Encoding;

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    fn set(d: &mut Keyspace, key: &[u8], pairs: &[(&[u8], &[u8])]) -> usize {
        d.hset(key, pairs.iter().copied()).expect("a hash")
    }

    fn get(d: &mut Keyspace, key: &[u8], field: &[u8]) -> Option<String> {
        d.hget(key, field, |t| t.map(|t| text(&t))).expect("a hash")
    }

    fn text(t: &Text<'_>) -> String {
        String::from_utf8(t.to_vec()).expect("utf8 in these tests")
    }

    fn all(d: &mut Keyspace, key: &[u8]) -> Vec<(String, String)> {
        let mut out = Vec::new();
        d.hgetall(key, |f, v| out.push((text(&f), text(&v))))
            .expect("a hash");
        out.sort();
        out
    }

    fn expire(d: &mut Keyspace, key: &[u8], at: u64, fields: &[&[u8]]) -> Vec<Applied> {
        let mut out = Vec::new();
        d.hexpire(key, at, Cond::Always, fields.iter().copied(), |a| {
            out.push(a);
        })
        .expect("a hash");
        out
    }

    fn ttl_of(d: &mut Keyspace, key: &[u8], fields: &[&[u8]]) -> Vec<Ask> {
        let mut out = Vec::new();
        d.httl(key, fields.iter().copied(), |a| out.push(a))
            .expect("a hash");
        out
    }

    #[test]
    fn setting_a_field_on_a_key_that_is_not_there_makes_it() {
        let mut d = db();
        assert_eq!(set(&mut d, b"h", &[(b"f", b"v")]), 1);
        assert_eq!(d.kind_of(b"h"), Some(Kind::Hash));
        assert_eq!(get(&mut d, b"h", b"f").as_deref(), Some("v"));
    }

    #[test]
    fn writing_a_field_again_is_not_a_new_field() {
        let mut d = db();
        assert_eq!(set(&mut d, b"h", &[(b"f", b"one"), (b"g", b"two")]), 2);
        assert_eq!(set(&mut d, b"h", &[(b"f", b"three")]), 0, "f was there");
        assert_eq!(get(&mut d, b"h", b"f").as_deref(), Some("three"));
        assert_eq!(d.hlen(b"h").expect("a hash"), 2);
    }

    #[test]
    fn an_empty_write_does_not_make_a_key() {
        let mut d = db();
        let none: [(&[u8], &[u8]); 0] = [];
        assert_eq!(d.hset(b"h", none.iter().copied()).expect("ok"), 0);
        assert_eq!(d.kind_of(b"h"), None, "an empty hash does not exist");
    }

    #[test]
    fn losing_the_last_field_loses_the_key() {
        let mut d = db();
        set(&mut d, b"h", &[(b"f", b"v"), (b"g", b"w")]);
        assert_eq!(d.hdel(b"h", [b"f".as_slice()].into_iter()).expect("ok"), 1);
        assert_eq!(d.kind_of(b"h"), Some(Kind::Hash), "g is still there");
        assert_eq!(d.hdel(b"h", [b"g".as_slice()].into_iter()).expect("ok"), 1);
        assert_eq!(d.kind_of(b"h"), None, "and now nothing is");
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn every_command_says_wrongtype_for_a_string() {
        let mut d = db();
        d.set_plain(b"s", b"v").expect("room");

        assert_eq!(
            d.hset(b"s", [(b"f".as_slice(), b"v".as_slice())].into_iter())
                .unwrap_err()
                .code(),
            Code::WrongType
        );
        assert!(d.hget(b"s", b"f", |_| ()).is_err());
        assert!(d.hdel(b"s", [b"f".as_slice()].into_iter()).is_err());
        assert!(d.hlen(b"s").is_err());
        assert!(d.hexists(b"s", b"f").is_err());
        assert!(d.hstrlen(b"s", b"f").is_err());
        assert!(d.hgetall(b"s", |_, _| ()).is_err());
        assert!(d.hsetnx(b"s", b"f", b"v").is_err());
        assert!(d.hincrby(b"s", b"f", 1).is_err());
        assert!(d.hincrbyfloat(b"s", b"f", 1.0).is_err());
        assert!(d.hrandfield(b"s", |_| ()).is_err());
        assert!(d.hrandfield_n(b"s", 1, |_, _| ()).is_err());
        assert!(d.hscan(b"s", Cursor::START, 10, |_, _| ()).is_err());
        assert!(
            d.hmget(b"s", [b"f".as_slice()].into_iter(), |_| ())
                .is_err()
        );

        assert_eq!(
            d.kind_of(b"s"),
            Some(Kind::String),
            "and none of them wrote anything"
        );
    }

    #[test]
    fn a_missing_key_reads_as_an_empty_hash() {
        let mut d = db();
        assert_eq!(d.hlen(b"nope").expect("ok"), 0);
        assert!(!d.hexists(b"nope", b"f").expect("ok"));
        assert_eq!(d.hstrlen(b"nope", b"f").expect("ok"), 0);
        assert_eq!(get(&mut d, b"nope", b"f"), None);
        assert!(!d.hgetall(b"nope", |_, _| ()).expect("ok"));
        assert_eq!(
            d.hdel(b"nope", [b"f".as_slice()].into_iter()).expect("ok"),
            0
        );
    }

    #[test]
    fn hmget_answers_once_per_field_asked_for() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"c", b"3")]);

        let mut got = Vec::new();
        d.hmget(b"h", [b"a".as_slice(), b"b", b"c"].into_iter(), |t| {
            got.push(t.map(|t| text(&t)));
        })
        .expect("a hash");
        assert_eq!(
            got,
            vec![Some("1".into()), None, Some("3".into())],
            "the reply is positional, so b gets a nil and not a gap"
        );

        let mut missing = Vec::new();
        d.hmget(b"gone", [b"a".as_slice(), b"b"].into_iter(), |t| {
            missing.push(t.is_none());
        })
        .expect("no key");
        assert_eq!(missing, vec![true, true], "a missing key is all nils");
    }

    #[test]
    fn hsetnx_writes_only_a_field_that_is_not_there() {
        let mut d = db();
        assert!(d.hsetnx(b"h", b"f", b"one").expect("ok"), "made the key");
        assert!(!d.hsetnx(b"h", b"f", b"two").expect("ok"), "f was there");
        assert_eq!(get(&mut d, b"h", b"f").as_deref(), Some("one"));
        assert!(
            d.hsetnx(b"h", b"g", b"two").expect("ok"),
            "and it is per field, not per key"
        );
        assert_eq!(d.hlen(b"h").expect("ok"), 2);
    }

    #[test]
    fn hstrlen_counts_a_number_without_writing_it() {
        let mut d = db();
        set(&mut d, b"h", &[(b"n", b"-12345"), (b"s", b"hello")]);
        assert_eq!(d.hstrlen(b"h", b"n").expect("ok"), 6);
        assert_eq!(d.hstrlen(b"h", b"s").expect("ok"), 5);
        assert_eq!(d.hstrlen(b"h", b"nope").expect("ok"), 0);
    }

    #[test]
    fn incrementing_counts_up_from_nothing_and_refuses_what_is_not_a_number() {
        let mut d = db();
        assert_eq!(d.hincrby(b"h", b"n", 5).expect("ok"), 5, "absent is zero");
        assert_eq!(d.hincrby(b"h", b"n", -7).expect("ok"), -2);
        assert_eq!(get(&mut d, b"h", b"n").as_deref(), Some("-2"));

        set(&mut d, b"h", &[(b"s", b"words")]);
        let err = d.hincrby(b"h", b"s", 1).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(err.message(), NOT_AN_INT);
        assert_eq!(
            get(&mut d, b"h", b"s").as_deref(),
            Some("words"),
            "and it left the field alone"
        );
    }

    #[test]
    fn an_increment_that_leaves_the_range_is_refused_and_not_wrapped() {
        let mut d = db();
        let max = i64::MAX.to_string();
        set(&mut d, b"h", &[(b"n", max.as_bytes())]);
        let err = d.hincrby(b"h", b"n", 1).unwrap_err();
        assert_eq!(err.message(), WOULD_OVERFLOW);
        assert_eq!(
            get(&mut d, b"h", b"n").as_deref(),
            Some(max.as_str()),
            "the field still holds what it held"
        );
    }

    #[test]
    fn incrementing_by_a_float_reports_the_sum_and_refuses_infinity() {
        let mut d = db();
        assert!((d.hincrbyfloat(b"h", b"f", 10.5).expect("ok") - 10.5).abs() < 1e-9);
        assert!((d.hincrbyfloat(b"h", b"f", 0.1).expect("ok") - 10.6).abs() < 1e-9);

        let err = d.hincrbyfloat(b"h", b"f", f64::INFINITY).unwrap_err();
        assert_eq!(err.message(), "increment would produce NaN or Infinity");

        set(&mut d, b"h", &[(b"s", b"words")]);
        assert_eq!(
            d.hincrbyfloat(b"h", b"s", 1.0).unwrap_err().message(),
            NOT_A_FLOAT
        );
    }

    #[test]
    fn a_hash_promotes_in_the_keyspace_and_object_encoding_says_so() {
        let mut d = db();
        set(&mut d, b"h", &[(b"f", b"v")]);
        assert_eq!(d.hash_encoding(b"h"), Some(Encoding::Listpack));
        assert_eq!(d.encoding_name(b"h"), Some("listpack"));

        for i in 0..200u32 {
            let f = format!("field-{i}");
            set(&mut d, b"h", &[(f.as_bytes(), b"v")]);
        }
        assert_eq!(d.hash_encoding(b"h"), Some(Encoding::Hashtable));
        assert_eq!(d.encoding_name(b"h"), Some("hashtable"));
        assert_eq!(d.hlen(b"h").expect("ok"), 201);
        assert_eq!(
            d.hash_encoding(b"missing"),
            None,
            "and a key that is not a hash has no hash encoding"
        );
    }

    #[test]
    fn a_hash_survives_being_given_a_deadline_and_goes_when_it_passes() {
        let mut d = db();
        set(&mut d, b"h", &[(b"f", b"v"), (b"g", b"w")]);
        assert!(d.set_expiry(b"h", Some(1_100)));
        assert_eq!(
            all(&mut d, b"h"),
            vec![("f".into(), "v".into()), ("g".into(), "w".into())],
            "writing the record did not touch the body"
        );

        d.clock_mut().advance(100);
        assert_eq!(d.kind_of(b"h"), None);
        assert_eq!(d.len(), 0);
        assert_eq!(d.expired_keys(), 1);
    }

    #[test]
    fn writing_a_string_over_a_hash_gives_the_body_back() {
        let mut d = db();
        for i in 0..300u32 {
            let f = format!("field-{i}");
            set(&mut d, b"h", &[(f.as_bytes(), b"a value of some length")]);
        }
        assert_eq!(d.hashes.len(), 1);
        let held = d.memory_bytes();
        d.set_plain(b"h", b"now a string").expect("room");

        assert_eq!(d.kind_of(b"h"), Some(Kind::String));
        // The slot rather than the byte count, because the byte count is mostly
        // the arena and the arena does not give a segment back until it is
        // compacted. A body that kept its slot would be reachable forever and
        // is the exact leak `free_body` exists to stop.
        assert_eq!(d.hashes.len(), 0, "the body went with the record");
        assert!(d.memory_bytes() < held, "and its bytes went with it");
    }

    #[test]
    fn a_scan_walks_a_hash_in_the_keyspace_exactly_once() {
        let mut d = db();
        for i in 0..500u32 {
            let f = format!("field-{i}");
            let v = format!("value-{i}");
            set(&mut d, b"h", &[(f.as_bytes(), v.as_bytes())]);
        }

        let mut seen: Vec<(String, String)> = Vec::new();
        let mut cursor = Cursor::START;
        loop {
            cursor = d
                .hscan(b"h", cursor, 32, |f, v| seen.push((text(&f), text(&v))))
                .expect("a hash");
            if cursor == Cursor::END {
                break;
            }
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 500, "every field once and only once");
        for (f, v) in &seen {
            assert_eq!(
                f.strip_prefix("field-"),
                v.strip_prefix("value-"),
                "and paired with its own value"
            );
        }
    }

    #[test]
    fn a_draw_takes_the_count_asked_for_and_repeats_only_when_told_to() {
        let mut d = db();
        d.seed(7);
        for i in 0..10u32 {
            let f = format!("f{i}");
            set(&mut d, b"h", &[(f.as_bytes(), b"v")]);
        }

        let mut got = Vec::new();
        d.hrandfield_n(b"h", 4, |f, _| got.push(text(&f)))
            .expect("ok");
        assert_eq!(got.len(), 4);
        got.sort();
        got.dedup();
        assert_eq!(got.len(), 4, "a positive count is distinct");

        let mut over = Vec::new();
        d.hrandfield_n(b"h", 25, |f, _| over.push(text(&f)))
            .expect("ok");
        assert_eq!(over.len(), 10, "and never more than the hash holds");

        let mut with_repeats = Vec::new();
        d.hrandfield_n(b"h", -25, |f, _| with_repeats.push(text(&f)))
            .expect("ok");
        assert_eq!(
            with_repeats.len(),
            25,
            "a negative count is exactly that many, repeats and all"
        );

        let one = d
            .hrandfield(b"h", |p| p.map(|(f, _)| text(&f)))
            .expect("ok");
        assert!(one.is_some());
        assert!(
            d.hrandfield(b"gone", |p| p.is_none()).expect("ok"),
            "and a missing key draws a nil"
        );
    }

    #[test]
    fn a_field_deadline_goes_on_and_is_reported_back() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"b", b"2")]);
        assert_eq!(
            expire(&mut d, b"h", 5_000, &[b"a", b"nope"]),
            [Applied::Ok, Applied::NoField],
            "one call per field, in the order asked"
        );
        assert_eq!(
            ttl_of(&mut d, b"h", &[b"a", b"b", b"nope"]),
            [Ask::At(5_000), Ask::NoDeadline, Ask::NoField]
        );
        assert_eq!(
            d.encoding_name(b"h"),
            Some("listpackex"),
            "and the band widened to hold it"
        );
    }

    #[test]
    fn a_field_is_gone_the_next_time_the_key_is_touched() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"b", b"2")]);
        expire(&mut d, b"h", 2_000, &[b"a"]);

        assert_eq!(d.hlen(b"h").expect("ok"), 2, "still there at 1000");
        d.clock_mut().advance(1_000);
        assert_eq!(d.hlen(b"h").expect("ok"), 1, "and gone at 2000");
        assert_eq!(get(&mut d, b"h", b"a"), None);
        assert_eq!(get(&mut d, b"h", b"b").as_deref(), Some("2"));
        assert_eq!(all(&mut d, b"h"), [("b".to_owned(), "2".to_owned())]);
    }

    #[test]
    fn the_key_goes_when_its_last_field_expires() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1")]);
        expire(&mut d, b"h", 2_000, &[b"a"]);
        assert_eq!(d.kind_of(b"h"), Some(Kind::Hash));

        d.clock_mut().advance(1_000);
        assert_eq!(d.hlen(b"h").expect("ok"), 0);
        assert_eq!(d.kind_of(b"h"), None, "an empty hash is not stored");
        assert_eq!(d.len(), 0);
    }

    /// `HEXPIRE key 0` is a roundabout `HDEL`, and taking the last field with it
    /// takes the key.
    #[test]
    fn a_deadline_already_past_deletes_the_field_now() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"b", b"2")]);
        assert_eq!(expire(&mut d, b"h", 500, &[b"a"]), [Applied::Deleted]);
        assert_eq!(d.hlen(b"h").expect("ok"), 1);

        assert_eq!(expire(&mut d, b"h", 500, &[b"b"]), [Applied::Deleted]);
        assert_eq!(d.kind_of(b"h"), None);
    }

    #[test]
    fn persisting_puts_the_field_back_to_no_deadline() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1")]);
        expire(&mut d, b"h", 5_000, &[b"a"]);

        let mut out = Vec::new();
        d.hpersist(
            b"h",
            [b"a".as_slice(), b"nope".as_slice()].into_iter(),
            |a| {
                out.push(a);
            },
        )
        .expect("ok");
        assert_eq!(out, [Ask::At(5_000), Ask::NoField]);
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::NoDeadline]);

        d.clock_mut().advance(100_000);
        assert_eq!(d.hlen(b"h").expect("ok"), 1, "and it outlives its deadline");
    }

    #[test]
    fn a_missing_key_answers_no_field_for_every_field_it_was_asked() {
        let mut d = db();
        assert_eq!(
            expire(&mut d, b"gone", 5_000, &[b"a", b"b"]),
            [Applied::NoField, Applied::NoField]
        );
        assert_eq!(
            ttl_of(&mut d, b"gone", &[b"a", b"b"]),
            [Ask::NoField, Ask::NoField]
        );
        assert_eq!(d.kind_of(b"gone"), None, "and asking did not create it");
    }

    #[test]
    fn a_deadline_past_the_ceiling_is_refused_before_any_field_moves() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1")]);
        let err = d
            .hexpire(
                b"h",
                crate::ttl::MAX_AT + 1,
                Cond::Always,
                [b"a".as_slice()].into_iter(),
                |_| unreachable!("no field is reached"),
            )
            .expect_err("past the ceiling");
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::NoDeadline]);
    }

    #[test]
    fn every_field_ttl_command_says_wrongtype_and_writes_nothing() {
        let mut d = db();
        d.set_plain(b"s", b"v").expect("room");
        assert!(
            d.hexpire(
                b"s",
                5_000,
                Cond::Always,
                [b"a".as_slice()].into_iter(),
                |_| { unreachable!("nothing is reached") }
            )
            .is_err()
        );
        assert!(d.httl(b"s", [b"a".as_slice()].into_iter(), |_| {}).is_err());
        assert!(
            d.hpersist(b"s", [b"a".as_slice()].into_iter(), |_| {})
                .is_err()
        );
        assert_eq!(
            d.kind_of(b"s"),
            Some(Kind::String),
            "and the string is intact"
        );
    }

    #[test]
    fn a_hash_that_never_expires_a_field_is_untouched_by_all_of_this() {
        let mut d = db();
        for i in 0..300u32 {
            set(&mut d, b"h", &[(format!("f{i}").as_bytes(), b"v")]);
        }
        assert_eq!(d.encoding_name(b"h"), Some("hashtable"));
        d.clock_mut().advance(1_000_000);
        assert_eq!(d.hlen(b"h").expect("ok"), 300, "nothing had a deadline");
    }

    /// `HGETDEL`, as the strings it handed back.
    fn getdel(d: &mut Keyspace, key: &[u8], fields: &[&[u8]]) -> Vec<Option<String>> {
        let mut out = Vec::new();
        d.hgetdel(key, fields.iter().copied(), |t| {
            out.push(t.map(|t| text(&t)));
        })
        .expect("a hash");
        out
    }

    /// `HGETEX`, the same way.
    fn getex(
        d: &mut Keyspace,
        key: &[u8],
        expire: strings::Expire,
        fields: &[&[u8]],
    ) -> Vec<Option<String>> {
        let mut out = Vec::new();
        d.hgetex(key, expire, fields.iter().copied(), |t| {
            out.push(t.map(|t| text(&t)));
        })
        .expect("a hash");
        out
    }

    /// `HSETEX`, with the two options spelled out.
    fn setex(
        d: &mut Keyspace,
        key: &[u8],
        exists: strings::Exists,
        expire: strings::Expire,
        pairs: &[(&[u8], &[u8])],
    ) -> bool {
        d.hsetex(key, exists, expire, pairs.iter().copied())
            .expect("a hash")
    }

    #[test]
    fn getdel_hands_the_value_back_and_then_takes_the_field() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]);
        assert_eq!(
            getdel(&mut d, b"h", &[b"a", b"nope"]),
            [Some("1".to_owned()), None],
            "positional, so a field that was not there is a hole and not a gap"
        );
        assert_eq!(all(&mut d, b"h").len(), 2);
        assert_eq!(
            getdel(&mut d, b"gone", &[b"a", b"b"]),
            [None, None],
            "and a missing key is all nils"
        );
        assert_eq!(d.kind_of(b"gone"), None, "which did not create it");

        getdel(&mut d, b"h", &[b"b", b"c"]);
        assert_eq!(d.kind_of(b"h"), None, "the last field took the key with it");
    }

    #[test]
    fn getdel_takes_the_deadline_with_the_field() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"b", b"2")]);
        expire(&mut d, b"h", 5_000, &[b"a"]);
        assert_eq!(getdel(&mut d, b"h", &[b"a"]), [Some("1".to_owned())]);
        set(&mut d, b"h", &[(b"a", b"9")]);
        assert_eq!(
            ttl_of(&mut d, b"h", &[b"a"]),
            [Ask::NoDeadline],
            "the field came back without the deadline it had"
        );
    }

    #[test]
    fn getex_reads_and_moves_the_deadline_in_one_go() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1")]);
        assert_eq!(
            getex(&mut d, b"h", strings::Expire::Keep, &[b"a"]),
            [Some("1".to_owned())]
        );
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::NoDeadline]);

        getex(&mut d, b"h", strings::Expire::At(5_000), &[b"a"]);
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::At(5_000)]);
        assert_eq!(
            getex(&mut d, b"h", strings::Expire::Keep, &[b"a"]),
            [Some("1".to_owned())],
            "and a plain read is Keep and not Clear, which is the one place this disagrees with SET"
        );
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::At(5_000)]);

        getex(&mut d, b"h", strings::Expire::Clear, &[b"a"]);
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::NoDeadline]);
    }

    #[test]
    fn getex_hands_back_the_value_of_a_field_it_is_about_to_expire() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1"), (b"b", b"2")]);
        assert_eq!(
            getex(&mut d, b"h", strings::Expire::At(1), &[b"a"]),
            [Some("1".to_owned())],
            "the read happened before the deadline was applied"
        );
        assert_eq!(get(&mut d, b"h", b"a"), None);
        assert_eq!(d.hlen(b"h").expect("ok"), 1);

        getex(&mut d, b"h", strings::Expire::At(1), &[b"b"]);
        assert_eq!(d.kind_of(b"h"), None, "and the last one took the key");
    }

    #[test]
    fn setex_writes_all_of_it_or_none_of_it() {
        let mut d = db();
        assert!(setex(
            &mut d,
            b"h",
            strings::Exists::Always,
            strings::Expire::Clear,
            &[(b"a", b"1")]
        ));
        assert_eq!(get(&mut d, b"h", b"a"), Some("1".to_owned()));

        assert!(
            !setex(
                &mut d,
                b"h",
                strings::Exists::IfMissing,
                strings::Expire::Clear,
                &[(b"a", b"9"), (b"new", b"9")]
            ),
            "FNX wants every field named to be missing, and a is not"
        );
        assert_eq!(get(&mut d, b"h", b"a"), Some("1".to_owned()));
        assert_eq!(
            get(&mut d, b"h", b"new"),
            None,
            "and none of it was written"
        );

        assert!(
            !setex(
                &mut d,
                b"h",
                strings::Exists::IfPresent,
                strings::Expire::Clear,
                &[(b"a", b"9"), (b"nope", b"9")]
            ),
            "and FXX wants every one of them to be there"
        );
        assert_eq!(get(&mut d, b"h", b"a"), Some("1".to_owned()));

        assert!(setex(
            &mut d,
            b"h",
            strings::Exists::IfPresent,
            strings::Expire::Clear,
            &[(b"a", b"9")]
        ));
        assert_eq!(get(&mut d, b"h", b"a"), Some("9".to_owned()));
    }

    #[test]
    fn setex_on_a_key_that_is_not_there_makes_it_only_when_it_can() {
        let mut d = db();
        assert!(
            !setex(
                &mut d,
                b"gone",
                strings::Exists::IfPresent,
                strings::Expire::Clear,
                &[(b"a", b"1")]
            ),
            "FXX cannot be met by a key with no fields at all"
        );
        assert_eq!(d.kind_of(b"gone"), None, "and it was not created");

        assert!(setex(
            &mut d,
            b"fresh",
            strings::Exists::IfMissing,
            strings::Expire::Clear,
            &[(b"a", b"1")]
        ));
        assert_eq!(get(&mut d, b"fresh", b"a"), Some("1".to_owned()));
    }

    #[test]
    fn setex_keeps_the_deadline_only_when_it_is_asked_to() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1")]);
        expire(&mut d, b"h", 5_000, &[b"a"]);

        setex(
            &mut d,
            b"h",
            strings::Exists::Always,
            strings::Expire::Keep,
            &[(b"a", b"2")],
        );
        assert_eq!(get(&mut d, b"h", b"a"), Some("2".to_owned()));
        assert_eq!(
            ttl_of(&mut d, b"h", &[b"a"]),
            [Ask::At(5_000)],
            "KEEPTTL put back what the write cleared"
        );

        setex(
            &mut d,
            b"h",
            strings::Exists::Always,
            strings::Expire::Clear,
            &[(b"a", b"3")],
        );
        assert_eq!(
            ttl_of(&mut d, b"h", &[b"a"]),
            [Ask::NoDeadline],
            "and without it the write clears the deadline the way HSET does"
        );

        setex(
            &mut d,
            b"h",
            strings::Exists::Always,
            strings::Expire::At(9_000),
            &[(b"a", b"4")],
        );
        assert_eq!(ttl_of(&mut d, b"h", &[b"a"]), [Ask::At(9_000)]);
    }

    #[test]
    fn setex_with_a_deadline_that_has_gone_stores_and_then_removes() {
        let mut d = db();
        assert!(
            setex(
                &mut d,
                b"h",
                strings::Exists::Always,
                strings::Expire::At(1),
                &[(b"a", b"1")]
            ),
            "written, and not the separate code the HEXPIRE family has for this"
        );
        assert_eq!(
            d.kind_of(b"h"),
            None,
            "so a key that did not exist is still not there"
        );

        set(&mut d, b"h", &[(b"keeper", b"1")]);
        setex(
            &mut d,
            b"h",
            strings::Exists::Always,
            strings::Expire::At(1),
            &[(b"a", b"1")],
        );
        assert_eq!(d.hlen(b"h").expect("ok"), 1, "and the rest of it survives");
    }

    #[test]
    fn setex_refuses_a_deadline_past_the_ceiling_before_writing_anything() {
        let mut d = db();
        set(&mut d, b"h", &[(b"a", b"1")]);
        let err = d
            .hsetex(
                b"h",
                strings::Exists::Always,
                strings::Expire::At(crate::ttl::MAX_AT + 1),
                [(b"a".as_slice(), b"2".as_slice())].into_iter(),
            )
            .expect_err("past the ceiling");
        assert_eq!(err.code(), Code::Invalid);
        assert_eq!(get(&mut d, b"h", b"a"), Some("1".to_owned()));
    }

    #[test]
    fn the_last_three_hash_commands_say_wrongtype_and_write_nothing() {
        let mut d = db();
        d.set_plain(b"s", b"v").expect("room");
        assert!(
            d.hgetdel(b"s", [b"a".as_slice()].into_iter(), |_| {})
                .is_err()
        );
        assert!(
            d.hgetex(
                b"s",
                strings::Expire::Keep,
                [b"a".as_slice()].into_iter(),
                |_| {}
            )
            .is_err()
        );
        assert!(
            d.hsetex(
                b"s",
                strings::Exists::Always,
                strings::Expire::Clear,
                [(b"a".as_slice(), b"1".as_slice())].into_iter(),
            )
            .is_err()
        );
        assert_eq!(d.kind_of(b"s"), Some(Kind::String));
    }

    #[test]
    fn the_last_three_reach_a_table_the_same_way_they_reach_a_listpack() {
        let mut d = db();
        for i in 0..300u32 {
            set(&mut d, b"h", &[(format!("f{i}").as_bytes(), b"v")]);
        }
        assert_eq!(d.encoding_name(b"h"), Some("hashtable"));

        setex(
            &mut d,
            b"h",
            strings::Exists::Always,
            strings::Expire::At(5_000),
            &[(b"f0", b"x")],
        );
        assert_eq!(ttl_of(&mut d, b"h", &[b"f0"]), [Ask::At(5_000)]);
        assert_eq!(
            getex(&mut d, b"h", strings::Expire::Clear, &[b"f0"]),
            [Some("x".to_owned())]
        );
        assert_eq!(ttl_of(&mut d, b"h", &[b"f0"]), [Ask::NoDeadline]);
        assert_eq!(getdel(&mut d, b"h", &[b"f0"]), [Some("x".to_owned())]);
        assert_eq!(d.hlen(b"h").expect("ok"), 299);
    }

    #[test]
    fn a_flush_takes_the_hashes_with_it() {
        let mut d = db();
        for i in 0..200u32 {
            let f = format!("field-{i}");
            set(&mut d, b"h", &[(f.as_bytes(), b"v")]);
        }
        set(&mut d, b"other", &[(b"f", b"v")]);
        d.clear();

        assert_eq!(d.len(), 0);
        assert_eq!(d.kind_of(b"h"), None);
        // Writing again reuses the slab from the start rather than growing past
        // the slots the cleared hashes had.
        set(&mut d, b"h", &[(b"f", b"v")]);
        assert_eq!(d.hlen(b"h").expect("ok"), 1);
    }
}
