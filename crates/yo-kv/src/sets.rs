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

use std::collections::HashSet;

use yo_common::Result;

use crate::db::Db;
use crate::keyspace::Keyspace;
use crate::scan::Cursor;
use crate::set::{Member, Set};
use crate::setops::{self, PerSet};
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

    /// `SPOP key`. Takes one member out at random and hands it back.
    ///
    /// This is the one set command that has to allocate, because the member it
    /// answers with is the member it just took out of the structure holding it.
    /// [`Keyspace::srandmember`] is the same draw without the removal and does
    /// not allocate, which is why the two are not one method with a flag.
    ///
    /// The key goes when the last member does, the same as `SREM`.
    pub fn spop(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let Some(at) = self.set_slot(key)? else {
            return Ok(None);
        };
        // A set in the keyspace is never empty, so there is always something to
        // draw and the draw is always in range.
        let len = self.set_at(at).len();
        let pick = self.rng.below(len);
        let got = self
            .sets
            .get_mut(at)
            .expect("the record points at its body")
            .remove_at(pick);
        if self.set_at(at).is_empty() {
            self.drop_key(key);
        }
        Ok(got)
    }

    /// `SPOP key count`. Takes `count` members out, or all of them if there are
    /// fewer than that.
    ///
    /// Drawing from the length that is left rather than from the length it
    /// started with is what makes the members distinct without a single test
    /// for it. Each removal moves some other member into the hole it made and
    /// shortens the set by one, so the next draw is over exactly the members
    /// that are still there and every one of them is equally likely.
    pub fn spop_n(&mut self, key: &[u8], count: usize) -> Result<Vec<Vec<u8>>> {
        let Some(at) = self.set_slot(key)? else {
            return Ok(Vec::new());
        };
        let take = count.min(self.set_at(at).len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            // The length is read again every turn rather than counted down,
            // because the removal is what changed it and reading it twice is a
            // load off a line that is already here.
            let pick = self.rng.below(self.set_at(at).len());
            out.push(
                self.sets
                    .get_mut(at)
                    .expect("the record points at its body")
                    .remove_at(pick)
                    .expect("the draw was under the length"),
            );
        }
        if self.set_at(at).is_empty() {
            self.drop_key(key);
        }
        Ok(out)
    }

    /// `SPOP key [count]`, as a borrow rather than a copy. Answers how many.
    ///
    /// The same draw as [`Keyspace::spop_n`] and none of the allocating. Each
    /// member is handed to `f` where it lies and taken out afterwards, so the
    /// bytes go from the set into the reply buffer and nothing is built in
    /// between. `spop_n` answers a `Vec` of `Vec`s, which is one allocation and
    /// then one more per member, and that is the right shape for an embedded
    /// caller who wants the answer in one piece and the wrong shape for a
    /// thread that must not allocate.
    ///
    /// That garbage is the whole of `SPOP`'s gate row. aki came in at 0.58x at
    /// P16 and 0.29x at P1 on this command, and the loss was never in the draw:
    /// the draw is an index into an array and a swap with the last row. It was
    /// in the allocation a member on the way out.
    ///
    /// Drawing from the length that is left rather than the length it started
    /// with is what makes the members distinct with no test for it, the same
    /// reason [`Keyspace::spop_n`] gives.
    pub fn spop_into<F>(&mut self, key: &[u8], count: usize, mut f: F) -> Result<usize>
    where
        F: FnMut(Member<'_>),
    {
        let Some(at) = self.set_slot(key)? else {
            return Ok(0);
        };
        let take = count.min(self.set_at(at).len());
        for _ in 0..take {
            // Borrowed apart rather than through `set_at`, for the reason
            // `srandmember_n` gives: drawing and reading are alive at the same
            // time and a method taking `&self` would hold the whole database.
            let rng = &mut self.rng;
            let set = self.sets.get(at).expect("the record points at its body");
            let pick = rng.below(set.len());
            f(set.at(pick).expect("the draw was under the length"));
            self.sets
                .get_mut(at)
                .expect("the record points at its body")
                .drop_at(pick);
        }
        if self.set_at(at).is_empty() {
            self.drop_key(key);
        }
        Ok(take)
    }

    /// `SRANDMEMBER key`, as a borrow rather than a copy.
    ///
    /// The member is handed to `f` where it lies, so the single draw form
    /// allocates nothing at all: the bytes go from the set into the reply
    /// buffer and an integer member is never written as digits anywhere in
    /// between. That is the whole of the gate row this command has on M3, where
    /// the loss against Redis was in the garbage rather than in the draw.
    ///
    /// `f` is handed `None` when the key is not there, which is a nil reply and
    /// not an empty one.
    pub fn srandmember<R>(
        &mut self,
        key: &[u8],
        f: impl FnOnce(Option<Member<'_>>) -> R,
    ) -> Result<R> {
        let Some(at) = self.set_slot(key)? else {
            return Ok(f(None));
        };
        let pick = self.rng.below(self.sets.get(at).expect("a body").len());
        Ok(f(self.set_at(at).at(pick)))
    }

    /// `SRANDMEMBER key count`, which is three different commands wearing one
    /// name.
    ///
    /// A negative count is the with repeats form: exactly that many members,
    /// drawn one at a time, and the same member can come back more than once.
    /// It is the only form that can answer more members than the set holds.
    ///
    /// A positive count is distinct members, at most as many as the set holds,
    /// and it is drawn two different ways depending on how much of the set is
    /// being asked for. Wanting more than a third of it is a walk of the whole
    /// set picking each member with the probability that leaves the right
    /// number at the end, which is Knuth's selection sampling and needs no
    /// memory at all. Wanting less than that is drawing positions and throwing
    /// away the repeats, which needs somewhere to remember what has been drawn
    /// and is the only thing here that allocates.
    ///
    /// Both are `O(count)`, which is the point of having two. Selection
    /// sampling alone would walk a million members to answer `SRANDMEMBER key
    /// 3`, and rejection alone would draw forever as the count approached the
    /// size. Redis splits the same way at the same ratio.
    pub fn srandmember_n<F>(&mut self, key: &[u8], count: i64, mut f: F) -> Result<()>
    where
        F: FnMut(Member<'_>),
    {
        let Some(at) = self.set_slot(key)? else {
            return Ok(());
        };
        // The two fields are borrowed apart rather than through `set_at`,
        // because drawing and reading have to be alive at the same time and a
        // method taking `&self` would hold the whole database.
        let rng = &mut self.rng;
        let set = self.sets.get(at).expect("the record points at its body");
        let len = set.len();

        let Ok(want) = usize::try_from(count) else {
            let repeats = usize::try_from(count.unsigned_abs()).unwrap_or(usize::MAX);
            for _ in 0..repeats {
                f(set
                    .at(rng.below(len))
                    .expect("the draw was under the length"));
            }
            return Ok(());
        };
        if want >= len {
            for m in set.iter() {
                f(m);
            }
            return Ok(());
        }
        if want.saturating_mul(3) > len {
            let mut need = want;
            for i in 0..len {
                if rng.below(len - i) < need {
                    f(set.at(i).expect("i is under the length"));
                    need -= 1;
                }
            }
            return Ok(());
        }
        let mut drawn = HashSet::with_capacity(want);
        while drawn.len() < want {
            let i = rng.below(len);
            if drawn.insert(i) {
                f(set.at(i).expect("the draw was under the length"));
            }
        }
        Ok(())
    }

    /// `SSCAN key cursor`. Walks part of the set and says where to resume.
    ///
    /// A missing key is a finished scan and not an error, which is what lets a
    /// client loop on the cursor without checking whether the key survived the
    /// walk. `MATCH` is not here: filtering the members is the caller's, so
    /// that the pattern is run against the member where it lies rather than
    /// against a copy made to be filtered.
    pub fn sscan<F>(&mut self, key: &[u8], cursor: Cursor, count: usize, f: F) -> Result<Cursor>
    where
        F: FnMut(Member<'_>),
    {
        let Some(at) = self.set_slot(key)? else {
            return Ok(Cursor::END);
        };
        Ok(self.set_at(at).scan(cursor, count, f))
    }

    /// `SMOVE source destination member`. Answers whether it moved.
    ///
    /// The order of the checks is Redis's and it is not the order it looks like
    /// it should be. A source that is not there answers zero without ever
    /// looking at what the destination holds, so `SMOVE nothing a-string m` is
    /// a zero and not a `WRONGTYPE`, and a source that is there checks both
    /// types before it moves anything.
    ///
    /// Moving a member onto its own set is a no op that still answers whether
    /// the member was there, which is the one case where a `1` means nothing
    /// changed.
    pub fn smove(&mut self, source: &[u8], destination: &[u8], member: &[u8]) -> Result<bool> {
        let Some(from) = self.set_slot(source)? else {
            return Ok(false);
        };
        let onto = self.set_slot(destination)?;
        if source == destination {
            return Ok(self.set_at(from).contains(member));
        }
        if !self
            .sets
            .get_mut(from)
            .expect("the record points at its body")
            .remove(member)
        {
            return Ok(false);
        }
        // The destination is filled before the source is emptied, so the slot
        // the source is about to give back cannot be handed straight to the
        // destination underneath the index this is holding.
        let limits = self.limits;
        let at = match onto {
            Some(at) => at,
            None => self.new_set(destination, member, 1),
        };
        self.sets
            .get_mut(at)
            .expect("the record points at its body")
            .add(member, &limits);
        if self.set_at(from).is_empty() {
            self.drop_key(source);
        }
        Ok(true)
    }

    /// `SINTER key [key ...]`, and `SINTERCARD`'s limit.
    ///
    /// Zero for a limit means no limit. The count comes back whether or not the
    /// caller collected anything, so [`Keyspace::sintercard`] is this with a
    /// callback that throws its argument away.
    pub fn sinter<'k, F>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        let slots = self.set_slots(keys)?;
        // A key that is not there is an empty set, and an empty set anywhere is
        // an empty intersection. That is the whole answer rather than a
        // shortcut to it, and it is why a missing key is not an error.
        if slots.is_empty() || slots.iter().any(Option::is_none) {
            return Ok(0);
        }
        // Taken out and put back, so the tables the intersection fills in are
        // the database's and not a pair the allocator hands out per call.
        let mut scratch = std::mem::take(&mut self.setops);
        let sets = self.bodies_of(&slots);
        let n = setops::inter(&mut scratch, &sets, limit, f);
        self.setops = scratch;
        Ok(n)
    }

    /// `SINTERCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn sintercard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
    ) -> Result<usize> {
        self.sinter(keys, limit, |_| {})
    }

    /// `SUNION key [key ...]`, and `SUNIONCARD`'s limit.
    ///
    /// A key that is not there contributes nothing and is dropped rather than
    /// emptying the answer, which is the opposite of what it does to an
    /// intersection and is right for the same reason: an empty set adds no
    /// members and removes none.
    ///
    /// Zero for a limit means no limit, as it does on [`Keyspace::sinter`].
    pub fn sunion<'k, F>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        let slots = self.set_slots(keys)?;
        // The database's table rather than one per call, for the reason in
        // `setops::Scratch`: a union walks everything into a hash table, and
        // building that table was most of what a `SUNION` over text sets did.
        let mut scratch = std::mem::take(&mut self.setops);
        let sets = self.bodies_of(&slots);
        let n = setops::union(&mut scratch, &sets, limit, f);
        self.setops = scratch;
        Ok(n)
    }

    /// `SUNIONCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn sunioncard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
    ) -> Result<usize> {
        self.sunion(keys, limit, |_| {})
    }

    /// `SDIFF key [key ...]`, and `SDIFFCARD`'s limit.
    ///
    /// The first key is the one being walked, so a first key that is not there
    /// is an empty answer whatever the rest hold. A later key that is not there
    /// takes nothing away and is dropped.
    pub fn sdiff<'k, F>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        let slots = self.set_slots(keys)?;
        let Some(Some(_)) = slots.first() else {
            return Ok(0);
        };
        let sets = self.bodies_of(&slots);
        Ok(setops::diff(&sets, limit, f))
    }

    /// `SDIFFCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn sdiffcard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
        limit: usize,
    ) -> Result<usize> {
        self.sdiff(keys, limit, |_| {})
    }

    /// `SINTERSTORE destination key [key ...]`. Answers the size of the result.
    pub fn sinterstore<'k>(
        &mut self,
        destination: &[u8],
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<usize> {
        let slots = self.set_slots(keys)?;
        let mut scratch = std::mem::take(&mut self.setops);
        let built = if slots.is_empty() || slots.iter().any(Option::is_none) {
            None
        } else {
            let sets = self.bodies_of(&slots);
            // The smallest input, which is an upper bound on any intersection.
            let upper = sets.iter().map(|s| s.len()).min().unwrap_or(0);
            setops::collect(upper, &self.limits, |f| {
                setops::inter(&mut scratch, &sets, 0, f);
            })
        };
        self.setops = scratch;
        Ok(self.put_set(destination, built))
    }

    /// `SUNIONSTORE destination key [key ...]`.
    pub fn sunionstore<'k>(
        &mut self,
        destination: &[u8],
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<usize> {
        let slots = self.set_slots(keys)?;
        let mut scratch = std::mem::take(&mut self.setops);
        let built = {
            let sets = self.bodies_of(&slots);
            // Everything, since a union of sets that share nothing is all of
            // them. Presizing to that is right and being wrong about it costs a
            // conversion rather than a wrong answer.
            let upper = sets.iter().map(|s| s.len()).sum();
            setops::collect(upper, &self.limits, |f| {
                setops::union(&mut scratch, &sets, 0, f);
            })
        };
        self.setops = scratch;
        Ok(self.put_set(destination, built))
    }

    /// `SDIFFSTORE destination key [key ...]`.
    pub fn sdiffstore<'k>(
        &mut self,
        destination: &[u8],
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<usize> {
        let slots = self.set_slots(keys)?;
        let built = match slots.first() {
            Some(Some(_)) => {
                let sets = self.bodies_of(&slots);
                let upper = sets[0].len();
                setops::collect(upper, &self.limits, |f| {
                    setops::diff(&sets, 0, f);
                })
            }
            _ => None,
        };
        Ok(self.put_set(destination, built))
    }

    /// Reap and resolve every key, in order, to the slot its set is in.
    ///
    /// `None` for a key that is not there, and an error the moment any key
    /// holds something that is not a set. Failing on the first bad key rather
    /// than at the end is what stops `SINTERSTORE d a not-a-set` from writing
    /// the destination before it finds out.
    ///
    /// This is what makes the borrow work: reaping needs `&mut self` and reading
    /// the bodies needs `&self`, so the keys have to be resolved before any body
    /// is looked at.
    ///
    /// It used to be a `Vec` and therefore a malloc and a free on every one of
    /// these commands, which is a real cost on the small end: a `SINTER` of two
    /// eight member sets does a couple of hundred nanoseconds of work and was
    /// paying for five allocations across this, [`Keyspace::bodies_of`] and
    /// [`crate::setops`]'s own bookkeeping. `Small` keeps the usual `k` on the
    /// stack and spills for the rare command that names more keys than that.
    fn set_slots<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<PerSet<Option<u32>>> {
        // Pushed rather than collected, because `set_slot` can fail and the
        // failure has to come out as an error rather than stop the walk quietly.
        let mut out = PerSet::new();
        for key in keys {
            out.push(self.set_slot(key)?);
        }
        Ok(out)
    }

    /// The bodies those slots point at, with the keys that were not there gone.
    #[inline]
    fn bodies_of(&self, slots: &[Option<u32>]) -> PerSet<&Set> {
        slots.iter().flatten().map(|&at| self.set_at(at)).collect()
    }

    /// Put a set under `key`, replacing whatever was there.
    ///
    /// No set means delete the key, because an empty set does not exist. That
    /// is what makes `SINTERSTORE d a b` with an empty intersection delete `d`
    /// and answer zero rather than leave an empty set that `EXISTS` says one
    /// for, and it is why [`setops::collect`] hands back an `Option`.
    ///
    /// The destination is allowed to be one of the sources. It is safe because
    /// the result was built whole before this was called, so nothing here can
    /// touch a body that is still being read. Doing it the other way round,
    /// clearing the destination first and filling it as the walk goes, is the
    /// shape that makes `SINTERSTORE s s a` answer nothing.
    ///
    /// Whatever the key held is freed first, through the one funnel, and any
    /// deadline it had goes with it. Redis's store forms clear the TTL for the
    /// same reason `SET` does: the value under the key is not the value the
    /// expiry was set on.
    fn put_set(&mut self, key: &[u8], set: Option<Set>) -> usize {
        let Some(set) = set else {
            self.drop_key(key);
            return 0;
        };
        self.free_body(key);
        let len = set.len();
        let at = self.sets.insert(set);
        let record = value::slot_record_len(false);
        self.write_rec(key, record, |out| {
            value::write_slot_record(out, Kind::Set, at, None);
        });
        self.bodies += 1;
        len
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
        self.live_slot(key, Kind::Set)
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
        // The body and, every so often, the slab that holds it. See
        // `yo_alloc::first_touch` for why this is the one allocation a command
        // is allowed to make.
        let at =
            yo_alloc::first_touch(|| self.sets.insert(Set::with_hint(first, hint, &self.limits)));
        let len = value::slot_record_len(false);
        self.write_rec(key, len, |out| {
            value::write_slot_record(out, Kind::Set, at, None);
        });
        self.bodies += 1;
        at
    }
}

/// Where a set body is, when the search for it covered a whole database.
///
/// The stripe and then the slot in that stripe's slab. A slot number means
/// nothing without the stripe it came from, since every stripe numbers its own
/// from zero.
type Home = (usize, u32);

impl Db {
    /// `SMOVE source destination member` over a database of any width.
    ///
    /// The two keys on one stripe are that stripe's `SMOVE`, which is the whole
    /// command on a database of one. Otherwise the member is taken out of one
    /// stripe and put into another, in the order the single stripe version
    /// moves it: the destination is filled before the source is emptied, and the
    /// source is only deleted once it is known to be empty.
    ///
    /// The checks are in Redis's order, which is not the order they look like
    /// they should be in. A source that is not there answers zero without ever
    /// looking at the destination, so a destination holding a string is not a
    /// `WRONGTYPE` until the source turns out to be a set.
    pub fn smove(&mut self, source: &[u8], destination: &[u8], member: &[u8]) -> Result<bool> {
        let (from, onto) = (self.stripe_of(source), self.stripe_of(destination));
        if from == onto {
            return self.stripe_mut(from).smove(source, destination, member);
        }
        let Some(at) = self.stripe_mut(from).set_slot(source)? else {
            return Ok(false);
        };
        let there = self.stripe_mut(onto).set_slot(destination)?;
        if !self
            .stripe_mut(from)
            .sets
            .get_mut(at)
            .expect("the record points at its body")
            .remove(member)
        {
            return Ok(false);
        }

        let dest = self.stripe_mut(onto);
        let limits = dest.limits;
        let into = match there {
            Some(into) => into,
            None => dest.new_set(destination, member, 1),
        };
        dest.sets
            .get_mut(into)
            .expect("the record points at its body")
            .add(member, &limits);

        let src = self.stripe_mut(from);
        if src.set_at(at).is_empty() {
            src.drop_key(source);
        }
        Ok(true)
    }

    /// `SINTER key [key ...]`, and `SINTERCARD`'s limit.
    pub fn sinter<'k, F>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        if let Some(home) = self.one_stripe(keys.clone()) {
            return self.stripe_mut(home).sinter(keys, limit, f);
        }
        let slots = self.set_slots(keys)?;
        if slots.is_empty() || slots.iter().any(Option::is_none) {
            return Ok(0);
        }
        let mut scratch = self.take_setops();
        let sets = self.bodies_of(&slots);
        let n = setops::inter(&mut scratch, &sets, limit, f);
        self.put_setops(scratch);
        Ok(n)
    }

    /// `SINTERCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn sintercard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
    ) -> Result<usize> {
        self.sinter(keys, limit, |_| {})
    }

    /// `SUNION key [key ...]`, and `SUNIONCARD`'s limit.
    pub fn sunion<'k, F>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        if let Some(home) = self.one_stripe(keys.clone()) {
            return self.stripe_mut(home).sunion(keys, limit, f);
        }
        let slots = self.set_slots(keys)?;
        let mut scratch = self.take_setops();
        let sets = self.bodies_of(&slots);
        let n = setops::union(&mut scratch, &sets, limit, f);
        self.put_setops(scratch);
        Ok(n)
    }

    /// `SUNIONCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn sunioncard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
    ) -> Result<usize> {
        self.sunion(keys, limit, |_| {})
    }

    /// `SDIFF key [key ...]`, and `SDIFFCARD`'s limit.
    pub fn sdiff<'k, F>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8]),
    {
        if let Some(home) = self.one_stripe(keys.clone()) {
            return self.stripe_mut(home).sdiff(keys, limit, f);
        }
        let slots = self.set_slots(keys)?;
        let Some(Some(_)) = slots.first() else {
            return Ok(0);
        };
        let sets = self.bodies_of(&slots);
        Ok(setops::diff(&sets, limit, f))
    }

    /// `SDIFFCARD numkeys key [key ...] [LIMIT limit]`.
    pub fn sdiffcard<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]> + Clone,
        limit: usize,
    ) -> Result<usize> {
        self.sdiff(keys, limit, |_| {})
    }

    /// `SINTERSTORE destination key [key ...]`. Answers the size of the result.
    ///
    /// The result is built whole before the destination is touched, exactly as
    /// it is on one stripe, which is what makes a destination that is also a
    /// source work. The limits and the slab the answer goes into are the
    /// destination's stripe's, since that is where the set is going to live.
    pub fn sinterstore<'k>(
        &mut self,
        destination: &'k [u8],
        keys: impl Iterator<Item = &'k [u8]> + Clone,
    ) -> Result<usize> {
        if let Some(home) = self.one_stripe(std::iter::once(destination).chain(keys.clone())) {
            return self.stripe_mut(home).sinterstore(destination, keys);
        }
        let slots = self.set_slots(keys)?;
        let mut scratch = self.take_setops();
        let built = if slots.is_empty() || slots.iter().any(Option::is_none) {
            None
        } else {
            let limits = self.at_ref(destination).limits;
            let sets = self.bodies_of(&slots);
            let upper = sets.iter().map(|s| s.len()).min().unwrap_or(0);
            setops::collect(upper, &limits, |f| {
                setops::inter(&mut scratch, &sets, 0, f);
            })
        };
        self.put_setops(scratch);
        Ok(self.at(destination).put_set(destination, built))
    }

    /// `SUNIONSTORE destination key [key ...]`.
    pub fn sunionstore<'k>(
        &mut self,
        destination: &'k [u8],
        keys: impl Iterator<Item = &'k [u8]> + Clone,
    ) -> Result<usize> {
        if let Some(home) = self.one_stripe(std::iter::once(destination).chain(keys.clone())) {
            return self.stripe_mut(home).sunionstore(destination, keys);
        }
        let slots = self.set_slots(keys)?;
        let mut scratch = self.take_setops();
        let built = {
            let limits = self.at_ref(destination).limits;
            let sets = self.bodies_of(&slots);
            let upper = sets.iter().map(|s| s.len()).sum();
            setops::collect(upper, &limits, |f| {
                setops::union(&mut scratch, &sets, 0, f);
            })
        };
        self.put_setops(scratch);
        Ok(self.at(destination).put_set(destination, built))
    }

    /// `SDIFFSTORE destination key [key ...]`.
    pub fn sdiffstore<'k>(
        &mut self,
        destination: &'k [u8],
        keys: impl Iterator<Item = &'k [u8]> + Clone,
    ) -> Result<usize> {
        if let Some(home) = self.one_stripe(std::iter::once(destination).chain(keys.clone())) {
            return self.stripe_mut(home).sdiffstore(destination, keys);
        }
        let slots = self.set_slots(keys)?;
        let built = match slots.first() {
            Some(Some(_)) => {
                let limits = self.at_ref(destination).limits;
                let sets = self.bodies_of(&slots);
                let upper = sets[0].len();
                setops::collect(upper, &limits, |f| {
                    setops::diff(&sets, 0, f);
                })
            }
            _ => None,
        };
        Ok(self.at(destination).put_set(destination, built))
    }

    /// Reap and resolve every key, in order, to the stripe and slot its set is
    /// in.
    ///
    /// As [`Keyspace::set_slots`], including the part that matters most: the
    /// first key holding something that is not a set stops the whole command
    /// before anything has been written. Each key is resolved on its own stripe,
    /// which is the only difference.
    fn set_slots<'k>(
        &mut self,
        keys: impl Iterator<Item = &'k [u8]>,
    ) -> Result<PerSet<Option<Home>>> {
        let mut out = PerSet::new();
        for key in keys {
            let stripe = self.stripe_of(key);
            let at = self.stripe_mut(stripe).set_slot(key)?;
            out.push(at.map(|at| (stripe, at)));
        }
        Ok(out)
    }

    /// The bodies those slots point at, with the keys that were not there gone.
    ///
    /// Several stripes are borrowed at once here and that is the whole reason
    /// the resolving above happens first: reaping a key needs the stripe
    /// mutably, reading a body needs it shared, and an operation over four keys
    /// on four stripes needs all four bodies at the same time.
    #[inline]
    fn bodies_of(&self, slots: &[Option<Home>]) -> PerSet<&Set> {
        slots
            .iter()
            .flatten()
            .map(|&(stripe, at)| self.stripe(stripe).set_at(at))
            .collect()
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

    /// `SUNION` built a hash table out of the allocator on every call, and over
    /// text sets that table was most of what the command did.
    #[test]
    fn a_union_over_text_sets_does_not_allocate_once_its_table_is_warm() {
        let mut d = db();
        add(&mut d, b"a", &[b"alpha", b"beta", b"gamma", b"delta"]);
        add(&mut d, b"b", &[b"gamma", b"delta", b"epsilon", b"zeta"]);
        // One call to grow the table to the size of this union. Everything
        // after it reuses what that one bought.
        assert_eq!(
            d.sunion([b"a".as_slice(), b"b"].into_iter(), 0, |_| {}),
            Ok(6)
        );
        let (_, allocs) = crate::tally::counted(|| {
            for _ in 0..50 {
                assert_eq!(
                    d.sunion([b"a".as_slice(), b"b"].into_iter(), 0, |_| {}),
                    Ok(6)
                );
            }
        });
        assert_eq!(allocs, 0, "sunion allocated {allocs} times in fifty");
    }

    /// And it still answers when the union is bigger than any before it, which
    /// is the case the reserve is there for.
    #[test]
    fn a_union_larger_than_the_last_one_grows_the_table_and_is_still_right() {
        let mut d = db();
        add(&mut d, b"a", &[b"one", b"two"]);
        add(&mut d, b"b", &[b"two", b"three"]);
        assert_eq!(
            d.sunion([b"a".as_slice(), b"b"].into_iter(), 0, |_| {}),
            Ok(3)
        );

        let many: Vec<Vec<u8>> = (0..500).map(|i| format!("m{i}").into_bytes()).collect();
        let refs: Vec<&[u8]> = many.iter().map(Vec::as_slice).collect();
        add(&mut d, b"c", &refs);
        let mut seen = Vec::new();
        assert_eq!(
            d.sunion([b"a".as_slice(), b"c"].into_iter(), 0, |m: &[u8]| seen
                .push(m.to_vec())),
            Ok(502)
        );
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 502, "every member came back once");

        // And back down again, which is the direction that would break if the
        // table were only ever grown and not cleared.
        assert_eq!(
            d.sunion([b"a".as_slice(), b"b"].into_iter(), 0, |_| {}),
            Ok(3)
        );
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

    /// A set of `n` members named `m0` up, which is a table past 128.
    fn many(d: &mut Keyspace, key: &[u8], n: usize) {
        let owned: Vec<Vec<u8>> = (0..n).map(|i| format!("m{i}").into_bytes()).collect();
        let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        d.sadd(key, refs.iter().copied()).expect("a set");
    }

    /// `many`, and integers instead of names when asked, so a test can reach
    /// the intset band as well as the other two.
    fn fill(d: &mut Keyspace, key: &[u8], n: usize, ints: bool) {
        let owned: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                if ints {
                    i.to_string().into_bytes()
                } else {
                    format!("m{i}").into_bytes()
                }
            })
            .collect();
        let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        d.sadd(key, refs.iter().copied()).expect("a set");
    }

    fn drawn(d: &mut Keyspace, key: &[u8], count: i64) -> Vec<String> {
        let mut out = Vec::new();
        d.srandmember_n(key, count, |m| {
            out.push(String::from_utf8(m.to_vec()).expect("utf8 in these tests"));
        })
        .expect("a set");
        out
    }

    #[test]
    fn popping_takes_a_member_out_and_the_key_with_the_last_one() {
        let mut d = db();
        add(&mut d, b"s", &[b"a", b"b"]);
        let first = d.spop(b"s").expect("a set").expect("two members");
        assert_eq!(d.scard(b"s").expect("a set"), 1);

        let second = d.spop(b"s").expect("a set").expect("one member");
        assert_ne!(first, second, "the same member came back twice");
        assert!(!d.exists(b"s"), "the last member took the key");
        assert_eq!(d.sets.len(), 0, "and the body");
        assert_eq!(d.spop(b"s").expect("gone is not an error"), None);
    }

    #[test]
    fn popping_a_count_empties_a_set_without_repeating_itself() {
        // In all three representations, because the table moves its last row
        // into the hole and the other two shift, and a draw that assumed either
        // one would repeat a member or run off the end.
        for n in [4usize, 100, 300] {
            let mut d = db();
            many(&mut d, b"s", n);
            let got = d.spop_n(b"s", n + 10).expect("a set");
            assert_eq!(got.len(), n, "asked for more than there was");
            let mut sorted = got.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), n, "a member came back twice");
            assert!(!d.exists(b"s"));
            assert_eq!(d.sets.len(), 0);
        }
    }

    #[test]
    fn popping_part_of_a_set_leaves_the_rest_of_it() {
        let mut d = db();
        many(&mut d, b"s", 10);
        let got = d.spop_n(b"s", 4).expect("a set");
        assert_eq!(got.len(), 4);
        assert_eq!(d.scard(b"s").expect("a set"), 6);
        for m in &got {
            assert!(!d.sismember(b"s", m).expect("a set"), "still there");
        }
        assert_eq!(
            d.spop_n(b"s", 0).expect("a set").len(),
            0,
            "and zero is none"
        );
        assert_eq!(d.scard(b"s").expect("a set"), 6);
    }

    #[test]
    fn the_borrowing_draw_pops_the_same_set_the_copying_one_does() {
        // Same seed, same set, same members in the same order. If the two ever
        // disagree then the wire and the embedded API answer differently for
        // the same command, which is the one thing there is no excuse for.
        for n in [4usize, 100, 300] {
            let mut a = db();
            a.seed(20_260_829);
            many(&mut a, b"s", n);
            let copied = a.spop_n(b"s", n).expect("a set");

            let mut b = db();
            b.seed(20_260_829);
            many(&mut b, b"s", n);
            let mut borrowed = Vec::new();
            b.spop_into(b"s", n, |m| borrowed.push(m.to_vec()))
                .expect("a set");

            assert_eq!(copied, borrowed, "{n} members drew differently");
            assert!(!b.exists(b"s"), "the last member took the key");
            assert_eq!(b.sets.len(), 0, "and the body");
        }
    }

    #[test]
    fn the_borrowing_draw_allocates_nothing() {
        // Every representation, because each takes a member out its own way:
        // the intset shifts an array of integers, the listpack shifts bytes,
        // and the table moves its last row into the hole. Also the whole set
        // rather than part of it, so the key deletion at the end is inside the
        // measurement and not just the draw.
        for n in [4usize, 100, 300] {
            for ints in [false, true] {
                let mut d = db();
                fill(&mut d, b"s", n, ints);
                let (drawn, allocs) = crate::tally::counted(|| {
                    let mut bytes = 0;
                    let mut count = 0;
                    d.spop_into(b"s", n, |m| {
                        // Read the member here rather than keep it, which is
                        // what the reply buffer does with it on the wire.
                        bytes += m.byte_len();
                        count += 1;
                    })
                    .expect("a set");
                    (bytes, count)
                });
                assert_eq!(drawn.1, n, "{n} members, ints {ints}");
                assert!(drawn.0 > 0, "the members came back empty");
                assert_eq!(
                    allocs, 0,
                    "{n} members, ints {ints}: {allocs} allocations on the way out"
                );
            }
        }
    }

    /// The `k` sized bookkeeping a set operation does before it starts is gone.
    /// It used to be five vectors across `set_slots`, `bodies_of` and `setops`,
    /// each a malloc and a free, on a command whose real work over three eight
    /// member sets is a couple of hundred nanoseconds.
    ///
    /// On integer sets that leaves nothing at all, because the merge walks the
    /// sorted arrays and needs no table. On the other representations `SUNION`
    /// and `SDIFF` still build one hash table each to dedupe with, which is
    /// sized by the members rather than by the number of keys and is the
    /// algorithm rather than the bookkeeping.
    #[test]
    fn a_small_set_operation_stops_paying_per_key() {
        for (ints, want) in [(true, 0), (false, 6)] {
            let mut d = db();
            fill(&mut d, b"a", 8, ints);
            fill(&mut d, b"b", 8, ints);
            fill(&mut d, b"c", 8, ints);
            let keys: [&[u8]; 3] = [b"a", b"b", b"c"];

            let (found, allocs) = crate::tally::counted(|| {
                let mut n = 0;
                d.sinter(keys.iter().copied(), 0, |_| n += 1).expect("sets");
                d.sunion(keys.iter().copied(), 0, |_| n += 1).expect("sets");
                d.sdiff(keys.iter().copied(), 0, |_| n += 1).expect("sets");
                n
            });
            assert!(found > 0, "ints {ints}: the operations found nothing");
            assert_eq!(
                allocs, want,
                "ints {ints}: {allocs} allocations for three ops, wanted {want}"
            );
        }
    }

    /// And past the inline room it still works, which is the half of `Small`
    /// that only the rare command reaches.
    #[test]
    fn a_wide_set_operation_still_answers() {
        let wide = crate::setops::INLINE_KEYS + 3;
        let mut d = db();
        let names: Vec<Vec<u8>> = (0..wide).map(|i| format!("k{i}").into_bytes()).collect();
        for name in &names {
            fill(&mut d, name, 8, true);
        }
        let keys = || names.iter().map(|k| k.as_slice());
        let mut inter = 0;
        d.sinter(keys(), 0, |_| inter += 1).expect("sets");
        // Every set holds the same eight members, so they all survive.
        assert_eq!(inter, 8);
        let mut union = 0;
        d.sunion(keys(), 0, |_| union += 1).expect("sets");
        assert_eq!(union, 8);
    }

    #[test]
    fn the_copying_draw_allocates_a_member_at_a_time() {
        // The other half of it. `spop_n` stays for the embedded caller who
        // wants the answer in one piece, and this is what that shape costs,
        // which is the whole reason the borrowing draw exists.
        let mut d = db();
        many(&mut d, b"s", 100);
        let (got, allocs) = crate::tally::counted(|| d.spop_n(b"s", 100).expect("a set"));
        assert_eq!(got.len(), 100);
        assert!(allocs >= 100, "only {allocs} allocations for a hundred");
    }

    #[test]
    fn a_pinned_seed_draws_the_same_members_twice() {
        // The one input that makes a result unrepeatable, handed in rather than
        // reached for. Without this there is nothing to assert about a draw
        // except that something came back.
        let mut runs = Vec::new();
        for _ in 0..2 {
            let mut d = db();
            d.seed(20_260_828);
            many(&mut d, b"s", 50);
            runs.push(d.spop_n(b"s", 10).expect("a set"));
        }
        assert_eq!(runs[0], runs[1]);
    }

    #[test]
    fn a_single_draw_reaches_every_member_and_removes_none() {
        let mut d = db();
        d.seed(7);
        add(&mut d, b"s", &[b"a", b"b", b"c"]);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let got = d
                .srandmember(b"s", |m| m.map(|m| m.to_vec()))
                .expect("a set")
                .expect("a member");
            seen.insert(got);
        }
        assert_eq!(seen.len(), 3, "a draw that never reaches a member");
        assert_eq!(d.scard(b"s").expect("a set"), 3, "and nothing was taken");

        assert!(
            d.srandmember(b"nope", |m| m.map(|m| m.to_vec()))
                .expect("missing is fine")
                .is_none()
        );
    }

    #[test]
    fn a_negative_count_repeats_itself_and_a_positive_one_does_not() {
        let mut d = db();
        d.seed(11);
        add(&mut d, b"s", &[b"a", b"b", b"c"]);

        let with_repeats = drawn(&mut d, b"s", -20);
        assert_eq!(with_repeats.len(), 20, "more members than the set holds");

        let mut distinct = drawn(&mut d, b"s", 2);
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), 2);
    }

    #[test]
    fn asking_for_more_than_the_set_holds_answers_all_of_it_once() {
        let mut d = db();
        d.seed(3);
        add(&mut d, b"s", &[b"a", b"b", b"c"]);
        let mut got = drawn(&mut d, b"s", 99);
        got.sort();
        assert_eq!(got, ["a", "b", "c"]);
        assert_eq!(drawn(&mut d, b"s", 0).len(), 0);
        assert_eq!(drawn(&mut d, b"nope", 5).len(), 0);
        assert_eq!(drawn(&mut d, b"nope", -5).len(), 0);
    }

    #[test]
    fn both_ways_of_drawing_distinct_members_are_distinct_and_uniform() {
        // The two branches of `srandmember_n`, either side of the third. A
        // thousand members and a draw of two hits the rejection branch, and the
        // same set with a draw of nine hundred hits the selection walk.
        let mut d = db();
        d.seed(99);
        many(&mut d, b"s", 1000);

        for count in [2, 100, 400, 900] {
            let got = drawn(&mut d, b"s", count);
            let mut sorted = got.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                got.len(),
                "a draw of {count} repeated a member"
            );
            assert_eq!(got.len(), count as usize);
        }

        // And every member is reachable by both, which a walk that stopped
        // early or a draw that never reached the top would not manage.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..40 {
            seen.extend(drawn(&mut d, b"s", 900));
            seen.extend(drawn(&mut d, b"s", 2));
        }
        assert_eq!(seen.len(), 1000, "some member is never drawn");
        assert_eq!(d.scard(b"s").expect("a set"), 1000, "and none were taken");
    }

    #[test]
    fn a_scan_walks_a_set_of_any_size_exactly_once() {
        for n in [3usize, 100, 500] {
            let mut d = db();
            many(&mut d, b"s", n);
            let mut seen = Vec::new();
            let mut c = Cursor::START;
            let mut turns = 0;
            loop {
                c = d
                    .sscan(b"s", c, 10, |m| seen.push(m.to_vec()))
                    .expect("a set");
                turns += 1;
                assert!(turns < 200, "the scan did not finish for {n} members");
                if c.is_end() {
                    break;
                }
            }
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), n, "a scan of {n} members missed one");
        }
    }

    #[test]
    fn a_scan_of_a_key_that_is_not_there_is_a_finished_scan() {
        let mut d = db();
        let mut hit = 0;
        let c = d
            .sscan(b"nope", Cursor::START, 10, |_| hit += 1)
            .expect("ok");
        assert!(c.is_end());
        assert_eq!(hit, 0);
    }

    #[test]
    fn a_scan_returns_everything_that_was_there_the_whole_time() {
        // The guarantee, tested the way it is written: members removed during
        // the walk may or may not come back, but the ones that never moved have
        // to. The table band is the only one that walks in windows, so this is
        // five hundred members.
        let mut d = db();
        many(&mut d, b"s", 500);
        let mut seen = Vec::new();
        let mut c = Cursor::START;
        let mut turns = 0;
        loop {
            c = d
                .sscan(b"s", c, 10, |m| seen.push(m.to_vec()))
                .expect("a set");
            // Take one out every turn, from the half of the set this test has
            // promised nothing about.
            let victim = format!("m{}", 400 + turns).into_bytes();
            d.srem(b"s", [victim.as_slice()].into_iter())
                .expect("a set");
            turns += 1;
            if c.is_end() {
                break;
            }
        }
        seen.sort();
        seen.dedup();
        for i in 0..400 {
            let m = format!("m{i}").into_bytes();
            assert!(seen.binary_search(&m).is_ok(), "m{i} was never returned");
        }
    }

    #[test]
    fn moving_a_member_takes_it_off_one_set_and_puts_it_on_another() {
        let mut d = db();
        add(&mut d, b"a", &[b"x", b"y"]);
        add(&mut d, b"b", &[b"z"]);

        assert!(d.smove(b"a", b"b", b"x").expect("two sets"));
        assert_eq!(members(&mut d, b"a"), ["y"]);
        assert_eq!(members(&mut d, b"b"), ["x", "z"]);

        assert!(
            !d.smove(b"a", b"b", b"gone").expect("two sets"),
            "a member that is not in the source does not move"
        );
        assert!(
            d.smove(b"a", b"b", b"y").expect("two sets"),
            "and the last one still moves"
        );
        assert!(!d.exists(b"a"), "the source went with its last member");
        assert_eq!(d.sets.len(), 1, "and so did its body");
        assert_eq!(members(&mut d, b"b"), ["x", "y", "z"]);
    }

    #[test]
    fn moving_onto_a_destination_that_is_not_there_makes_it() {
        let mut d = db();
        add(&mut d, b"a", &[b"x", b"y"]);
        assert!(d.smove(b"a", b"b", b"x").expect("a set"));
        assert_eq!(d.kind_of(b"b"), Some(Kind::Set));
        assert_eq!(members(&mut d, b"b"), ["x"]);
        assert_eq!(d.sets.len(), 2);
    }

    #[test]
    fn moving_a_member_onto_its_own_set_changes_nothing() {
        let mut d = db();
        add(&mut d, b"a", &[b"x", b"y"]);
        assert!(d.smove(b"a", b"a", b"x").expect("a set"), "it is there");
        assert!(!d.smove(b"a", b"a", b"z").expect("a set"), "it is not");
        assert_eq!(members(&mut d, b"a"), ["x", "y"]);
    }

    #[test]
    fn moving_checks_the_types_in_the_order_redis_checks_them() {
        let mut d = db();
        d.set_plain(b"str", b"v").expect("room");
        add(&mut d, b"s", &[b"x"]);

        assert!(
            !d.smove(b"nope", b"str", b"x").expect("no source, no error"),
            "a missing source answers zero without looking at the destination"
        );
        assert_eq!(
            d.smove(b"str", b"s", b"x").expect_err("no").code(),
            Code::WrongType
        );
        assert_eq!(
            d.smove(b"s", b"str", b"x").expect_err("no").code(),
            Code::WrongType
        );
        assert_eq!(
            members(&mut d, b"s"),
            ["x"],
            "and the failed move left the source alone"
        );
    }

    #[test]
    fn the_new_commands_answer_wrongtype_at_a_string() {
        let mut d = db();
        d.set_plain(b"k", b"v").expect("room");
        assert!(d.spop(b"k").is_err());
        assert!(d.spop_n(b"k", 2).is_err());
        assert!(d.srandmember(b"k", |m| m.is_some()).is_err());
        assert!(d.srandmember_n(b"k", 2, |_| ()).is_err());
        assert!(d.sscan(b"k", Cursor::START, 10, |_| ()).is_err());
        assert_eq!(
            d.get(b"k").expect("still a string").map(|v| v.to_vec()),
            Some(b"v".to_vec())
        );
    }

    /// Every algebra command, collected and sorted, so a test says what came
    /// back rather than what order it came back in.
    fn algebra(d: &mut Keyspace, op: &str, keys: &[&[u8]]) -> Vec<String> {
        let mut got = Vec::new();
        let mut take = |m: &[u8]| got.push(String::from_utf8_lossy(m).into_owned());
        let n = match op {
            "inter" => d.sinter(keys.iter().copied(), 0, &mut take),
            "union" => d.sunion(keys.iter().copied(), 0, &mut take),
            "diff" => d.sdiff(keys.iter().copied(), 0, &mut take),
            other => unreachable!("{other}"),
        }
        .expect("sets");
        assert_eq!(n, got.len(), "the count and the members disagree");
        got.sort();
        got
    }

    #[test]
    fn the_algebra_answers_what_the_sets_share_and_do_not() {
        let mut d = db();
        add(&mut d, b"a", &[b"1", b"2", b"3"]);
        add(&mut d, b"b", &[b"2", b"3", b"4"]);
        add(&mut d, b"c", &[b"3", b"4", b"5"]);

        assert_eq!(algebra(&mut d, "inter", &[b"a", b"b", b"c"]), ["3"]);
        assert_eq!(
            algebra(&mut d, "union", &[b"a", b"b", b"c"]),
            ["1", "2", "3", "4", "5"]
        );
        assert_eq!(algebra(&mut d, "diff", &[b"a", b"b"]), ["1"]);
        assert_eq!(
            algebra(&mut d, "diff", &[b"a"]),
            ["1", "2", "3"],
            "one set is that set"
        );
        assert_eq!(
            d.sintercard([b"a".as_slice(), b"b"].into_iter(), 0)
                .expect("sets"),
            2
        );
        assert_eq!(
            d.sintercard([b"a".as_slice(), b"b"].into_iter(), 1)
                .expect("sets"),
            1,
            "and a limit stops it early"
        );
    }

    /// A key that is not there is an empty set, and an empty set does three
    /// different things to the three operations.
    #[test]
    fn a_key_that_is_not_there_is_an_empty_set_everywhere() {
        let mut d = db();
        add(&mut d, b"a", &[b"1", b"2"]);

        assert!(algebra(&mut d, "inter", &[b"a", b"nope"]).is_empty());
        assert!(algebra(&mut d, "inter", &[b"nope", b"a"]).is_empty());
        assert_eq!(algebra(&mut d, "union", &[b"a", b"nope"]), ["1", "2"]);
        assert_eq!(algebra(&mut d, "diff", &[b"a", b"nope"]), ["1", "2"]);
        assert!(
            algebra(&mut d, "diff", &[b"nope", b"a"]).is_empty(),
            "nothing minus anything is nothing"
        );
        assert!(algebra(&mut d, "union", &[b"nope"]).is_empty());
        assert_eq!(d.len(), 1, "and none of that made a key");
    }

    #[test]
    fn a_store_form_writes_the_answer_and_says_how_big_it_is() {
        let mut d = db();
        add(&mut d, b"a", &[b"1", b"2", b"3"]);
        add(&mut d, b"b", &[b"2", b"3", b"4"]);

        assert_eq!(
            d.sinterstore(b"d", [b"a".as_slice(), b"b"].into_iter())
                .expect("sets"),
            2
        );
        assert_eq!(members(&mut d, b"d"), ["2", "3"]);
        assert_eq!(
            d.sunionstore(b"d", [b"a".as_slice(), b"b"].into_iter())
                .expect("sets"),
            4
        );
        assert_eq!(members(&mut d, b"d"), ["1", "2", "3", "4"]);
        assert_eq!(
            d.sdiffstore(b"d", [b"a".as_slice(), b"b"].into_iter())
                .expect("sets"),
            1
        );
        assert_eq!(members(&mut d, b"d"), ["1"]);
        // An all integer answer stores as an intset, because the destination
        // picks its representation from what actually went into it.
        assert_eq!(d.encoding_name(b"d"), Some(Encoding::Intset.name()));
    }

    /// The rule that makes an empty answer different from an empty set: the
    /// destination is deleted rather than left holding nothing.
    #[test]
    fn a_store_form_of_nothing_deletes_the_destination() {
        let mut d = db();
        add(&mut d, b"a", &[b"1"]);
        add(&mut d, b"b", &[b"2"]);
        add(&mut d, b"d", &[b"old"]);

        assert_eq!(
            d.sinterstore(b"d", [b"a".as_slice(), b"b"].into_iter())
                .expect("sets"),
            0
        );
        assert_eq!(d.kind_of(b"d"), None, "the destination went, not emptied");
        assert!(!d.exists(b"d"));

        // And the same for a difference that takes everything away, and for a
        // source that is not there at all.
        add(&mut d, b"d", &[b"old"]);
        assert_eq!(
            d.sdiffstore(b"d", [b"a".as_slice(), b"a"].into_iter())
                .expect("sets"),
            0
        );
        assert!(!d.exists(b"d"));
        add(&mut d, b"d", &[b"old"]);
        assert_eq!(
            d.sunionstore(b"d", [b"nope".as_slice()].into_iter())
                .expect("sets"),
            0
        );
        assert!(!d.exists(b"d"));
    }

    /// The destination is allowed to be one of the sources, which only works
    /// because the answer is built whole before anything is written.
    #[test]
    fn a_store_form_can_write_over_one_of_its_own_sources() {
        let mut d = db();
        add(&mut d, b"a", &[b"1", b"2", b"3"]);
        add(&mut d, b"b", &[b"2", b"3", b"4"]);

        assert_eq!(
            d.sinterstore(b"a", [b"a".as_slice(), b"b"].into_iter())
                .expect("sets"),
            2
        );
        assert_eq!(members(&mut d, b"a"), ["2", "3"]);

        // The same key named twice is not a special case either.
        assert_eq!(
            d.sunionstore(b"a", [b"a".as_slice(), b"a"].into_iter())
                .expect("sets"),
            2
        );
        assert_eq!(members(&mut d, b"a"), ["2", "3"]);
    }

    /// A destination that held something else is overwritten rather than
    /// refused, which is what Redis does and is the same rule `SET` follows.
    #[test]
    fn a_store_form_overwrites_whatever_the_destination_held() {
        let mut d = db();
        add(&mut d, b"a", &[b"1", b"2"]);
        d.set_plain(b"d", b"a string").expect("room");
        assert!(d.set_expiry(b"d", Some(9_999_999)));

        assert_eq!(
            d.sunionstore(b"d", [b"a".as_slice()].into_iter())
                .expect("sets"),
            2
        );
        assert_eq!(d.kind_of(b"d"), Some(Kind::Set));
        assert_eq!(members(&mut d, b"d"), ["1", "2"]);
        assert_eq!(d.expire_at(b"d"), None, "and the deadline went with it");
    }

    /// A bad key anywhere in the list fails the whole command, and it fails
    /// before the destination is touched rather than after.
    #[test]
    fn the_algebra_answers_wrongtype_before_it_writes_anything() {
        let mut d = db();
        add(&mut d, b"a", &[b"1"]);
        d.set_plain(b"str", b"v").expect("room");
        add(&mut d, b"d", &[b"old"]);

        assert!(
            d.sinter([b"a".as_slice(), b"str"].into_iter(), 0, |_| ())
                .is_err()
        );
        assert!(
            d.sunion([b"str".as_slice()].into_iter(), 0, |_| ())
                .is_err()
        );
        assert!(
            d.sdiff([b"a".as_slice(), b"str"].into_iter(), 0, |_| ())
                .is_err()
        );
        assert!(
            d.sinterstore(b"d", [b"a".as_slice(), b"str"].into_iter())
                .is_err()
        );
        assert_eq!(members(&mut d, b"d"), ["old"], "and left it alone");
    }

    /// Sets across all three representations, since the algebra is the only
    /// place where members have to cross from one to another.
    #[test]
    fn the_algebra_works_across_the_representations() {
        let mut d = db();
        let big: Vec<Vec<u8>> = (0..600).map(|i| i.to_string().into_bytes()).collect();
        let refs: Vec<&[u8]> = big.iter().map(Vec::as_slice).collect();
        d.sadd(b"table", refs.iter().copied()).expect("a set");
        add(&mut d, b"ints", &[b"1", b"2", b"999"]);
        add(&mut d, b"packed", &[b"2", b"3", b"x"]);
        assert_eq!(d.encoding_name(b"table"), Some(Encoding::Hashtable.name()));
        assert_eq!(d.encoding_name(b"ints"), Some(Encoding::Intset.name()));
        assert_eq!(d.encoding_name(b"packed"), Some(Encoding::Listpack.name()));

        // A member of the intset is a number that has no digits anywhere and
        // the table holds that same member as its digits, so this only finds
        // anything if the two agree about what a member is.
        assert_eq!(algebra(&mut d, "inter", &[b"ints", b"table"]), ["1", "2"]);
        assert_eq!(algebra(&mut d, "inter", &[b"packed", b"table"]), ["2", "3"]);
        assert_eq!(algebra(&mut d, "inter", &[b"ints", b"packed"]), ["2"]);
        assert_eq!(algebra(&mut d, "diff", &[b"ints", b"table"]), ["999"]);
        assert_eq!(
            algebra(&mut d, "union", &[b"ints", b"packed"]),
            ["1", "2", "3", "999", "x"]
        );
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

    /// The sharp version of the memo hazard. `a` is resolved and remembered, so
    /// something is holding a slab slot number for it. Deleting `a` frees that
    /// slot and the next set created takes it, so a memo that survived the
    /// delete would answer questions about `a` with `b`'s members. It is not a
    /// stale count, it is another key's data under the name of a key that is
    /// gone.
    #[test]
    fn a_deleted_key_does_not_answer_with_whatever_took_its_slot() {
        let mut d = db();
        add(&mut d, b"a", &[b"x", b"y", b"z"]);
        assert_eq!(d.scard(b"a").expect("a set"), 3);

        d.del(b"a");
        add(&mut d, b"b", &[b"one"]);

        assert_eq!(d.scard(b"a").expect("gone"), 0);
        assert!(!d.sismember(b"a", b"x").expect("gone"));
        assert_eq!(d.scard(b"b").expect("a set"), 1);
    }

    /// Same shape, one step further: the name comes back holding another type.
    /// A memo that answered from what it remembered would say the set is still
    /// there and hand back a slot that now belongs to a hash.
    #[test]
    fn a_key_that_comes_back_as_another_type_is_wrongtype() {
        let mut d = db();
        add(&mut d, b"k", &[b"x"]);
        assert_eq!(d.scard(b"k").expect("a set"), 1);

        d.del(b"k");
        d.hset(b"k", [(&b"f"[..], &b"v"[..])].into_iter())
            .expect("a hash");

        let err = d.scard(b"k").expect_err("a hash is not a set");
        assert_eq!(err.code(), Code::WrongType);
    }

    /// A deadline passes without anyone writing to the map, so it is the one
    /// thing a write counter cannot see. The answer is that a dated key is
    /// never remembered in the first place, and this is what says so.
    #[test]
    fn a_key_with_a_deadline_still_expires_after_it_has_been_read() {
        let mut d = db();
        add(&mut d, b"k", &[b"x", b"y"]);
        assert_eq!(d.scard(b"k").expect("a set"), 2);

        assert!(d.set_expiry(b"k", Some(1_500)));
        assert_eq!(d.scard(b"k").expect("still alive"), 2);

        d.clock_mut().advance(600);
        assert_eq!(d.scard(b"k").expect("past its deadline"), 0);
        assert!(!d.sismember(b"k", b"x").expect("past its deadline"));
    }

    /// Two keys alternating, which is what a pipeline that is not on one key
    /// looks like. Each one has to answer for itself, so the comparison is the
    /// key bytes and not the hash.
    #[test]
    fn two_keys_in_a_row_do_not_answer_for_each_other() {
        let mut d = db();
        add(&mut d, b"a", &[b"1"]);
        add(&mut d, b"b", &[b"1", b"2", b"3"]);
        for _ in 0..8 {
            assert_eq!(d.scard(b"a").expect("a set"), 1);
            assert_eq!(d.scard(b"b").expect("a set"), 3);
        }
    }

    /// `FLUSHDB` throws the map away and builds a fresh one, and a fresh one
    /// starts its write counter over. Nothing may survive that.
    #[test]
    fn a_flush_does_not_leave_the_last_key_behind() {
        let mut d = db();
        add(&mut d, b"k", &[b"x"]);
        assert_eq!(d.scard(b"k").expect("a set"), 1);

        d.clear();
        assert_eq!(d.scard(b"k").expect("flushed"), 0);
    }

    /// A key too long to remember is a key that is looked up every time, which
    /// is the old behaviour and has to keep working rather than fall through a
    /// branch that assumes something was written down.
    #[test]
    fn a_key_longer_than_the_memo_still_works() {
        let mut d = db();
        let long = vec![b'k'; 200];
        add(&mut d, &long, &[b"x", b"y"]);
        for _ in 0..4 {
            assert_eq!(d.scard(&long).expect("a set"), 2);
        }
        d.del(&long);
        assert_eq!(d.scard(&long).expect("gone"), 0);
    }
}
