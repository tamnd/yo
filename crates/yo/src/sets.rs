//! Sets, from the embedded side.
//!
//! The same store `SADD` off a socket reaches (Y23), through two doors. [`Sets`]
//! is the keyspace shape, one method per Redis command with the key as the first
//! argument, which is what a program porting off a Redis client wants. [`Set`] is
//! one key with a handle around it, which is what a program that was never going
//! to use Redis wants: a set is a set, and the name of it gets spelled once
//! rather than at every call site.
//!
//! A [`Set`] holds the [`Sets`] it goes through rather than building one per
//! call, so reaching a set by its handle costs the key it already had and
//! nothing else. There is no door here that is the slow one.
//!
//! # Owned or borrowed, per call
//!
//! Every read that hands back members comes in two forms. [`Set::members`]
//! allocates a `Vec` per member, which is what most code wants and what every
//! other embedded database gives you. [`Set::for_each`] hands each member over
//! where it lies and allocates nothing, which is Y29's rule that zero copy is
//! available and never mandatory.
//!
//! The difference is not decoration. A set stored as integers holds them as
//! integers, so walking a million member set with `members` formats a million
//! numbers into a million `Vec`s, and walking it with `for_each` formats none of
//! them unless the closure asks. That is why the borrowed form hands over a
//! [`Member`] and not a `&[u8]`: the choice of whether to spend the digits is
//! the caller's.

use yo_common::{Code, Error, Result};
use yo_kv::Member;

use crate::db::Handle;

/// Every Redis set command, with the key as the first argument.
///
/// Keys and members are byte strings the way Redis's are, so anything that is
/// bytes will do.
///
/// ```
/// let db = yo::open(yo::MEMORY)?;
/// let sets = db.sets();
///
/// sets.add_many("online", &["alice", "bob"])?;
/// assert!(sets.contains("online", "alice")?);
/// assert_eq!(sets.len_of("online")?, 2);
/// # Ok::<(), yo::Error>(())
/// ```
#[derive(Clone)]
pub struct Sets {
    pub(crate) db: Handle,
}

impl core::fmt::Debug for Sets {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sets").finish_non_exhaustive()
    }
}

impl Sets {
    /// Add one member, and say whether it was new. `SADD`.
    ///
    /// The key is created by the first member that goes into it, so there is no
    /// step before this one.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] when the key holds something that is not a set,
    /// [`Code::Full`] for a member past the size limit, and [`Code::Invalid`] if
    /// called from inside a callback that is already holding this database.
    pub fn add(&self, key: impl AsRef<[u8]>, member: impl AsRef<[u8]>) -> Result<bool> {
        self.add_many(key, &[member]).map(|n| n == 1)
    }

    /// Add several members, and say how many were new. `SADD` with a list.
    ///
    /// One key lookup for the whole call rather than one per member, which is
    /// the only reason to prefer it over calling [`Sets::add`] in a loop.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`]. Nothing is added if any member is too long, because the
    /// lengths are all checked before the first one goes in.
    pub fn add_many<M: AsRef<[u8]>>(&self, key: impl AsRef<[u8]>, members: &[M]) -> Result<usize> {
        self.db.run(|inner| {
            inner
                .strings
                .sadd(key.as_ref(), members.iter().map(AsRef::as_ref))
        })
    }

    /// Remove one member, and say whether it was there. `SREM`.
    ///
    /// A set that loses its last member loses its key too, which is Redis's rule
    /// and is why there is no such thing as an empty set in the keyspace.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn remove(&self, key: impl AsRef<[u8]>, member: impl AsRef<[u8]>) -> Result<bool> {
        self.remove_many(key, &[member]).map(|n| n == 1)
    }

    /// Remove several members, and say how many were there. `SREM` with a list.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn remove_many<M: AsRef<[u8]>>(
        &self,
        key: impl AsRef<[u8]>,
        members: &[M],
    ) -> Result<usize> {
        self.db.run(|inner| {
            inner
                .strings
                .srem(key.as_ref(), members.iter().map(AsRef::as_ref))
        })
    }

    /// Whether a member is in the set. `SISMEMBER`.
    ///
    /// False for a key that is not there, which is the same answer as an empty
    /// set and is deliberate: a set nobody has added to and a set somebody
    /// emptied are the same set.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn contains(&self, key: impl AsRef<[u8]>, member: impl AsRef<[u8]>) -> Result<bool> {
        self.db
            .run(|inner| inner.strings.sismember(key.as_ref(), member.as_ref()))
    }

    /// Whether each of several members is in the set, in the order asked.
    /// `SMISMEMBER`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn contains_many<M: AsRef<[u8]>>(
        &self,
        key: impl AsRef<[u8]>,
        members: &[M],
    ) -> Result<Vec<bool>> {
        self.db.run(|inner| {
            inner
                .strings
                .smismember(key.as_ref(), members.iter().map(AsRef::as_ref))
        })
    }

    /// How many members the set holds, which is zero for a key that is not
    /// there. `SCARD`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn len_of(&self, key: impl AsRef<[u8]>) -> Result<usize> {
        self.db.run(|inner| inner.strings.scard(key.as_ref()))
    }

    /// Every member, owned. `SMEMBERS`.
    ///
    /// `None` for a key that is not there, which a caller who wants to tell that
    /// apart from an empty answer can use. [`Sets::for_each`] is the same walk
    /// without the allocations.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn members(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<Vec<u8>>>> {
        self.db.run(|inner| {
            Ok(inner
                .strings
                .smembers(key.as_ref())?
                .map(|it| it.map(|m| m.to_vec()).collect()))
        })
    }

    /// Hand every member to `f` where it lies, and say whether the key was
    /// there.
    ///
    /// Nothing is allocated and nothing is formatted. A set stored as integers
    /// hands over [`Member::Int`] and the digits are only written if the closure
    /// writes them.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let sets = db.sets();
    /// sets.add_many("ids", &["1", "2", "3"])?;
    ///
    /// let mut total = 0i64;
    /// sets.for_each("ids", |m| {
    ///     if let yo::Member::Int(n) = m {
    ///         total += n;
    ///     }
    /// })?;
    /// assert_eq!(total, 6);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn for_each(&self, key: impl AsRef<[u8]>, mut f: impl FnMut(Member<'_>)) -> Result<bool> {
        self.db.run(|inner| {
            inner.strings.with_set(key.as_ref(), |set| match set {
                Some(set) => {
                    for m in set.iter() {
                        f(m);
                    }
                    true
                }
                None => false,
            })
        })
    }

    /// Take one member out at random and hand it back. `SPOP`.
    ///
    /// `None` for a key that is not there. The key goes when the last member
    /// does.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pop(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        self.db.run(|inner| inner.strings.spop(key.as_ref()))
    }

    /// Take up to `count` members out at random. `SPOP key count`.
    ///
    /// The members are distinct, and fewer than `count` come back when the set
    /// holds fewer than that.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pop_n(&self, key: impl AsRef<[u8]>, count: usize) -> Result<Vec<Vec<u8>>> {
        self.db
            .run(|inner| inner.strings.spop_n(key.as_ref(), count))
    }

    /// Draw one member at random and leave it in the set. `SRANDMEMBER`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pick(&self, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        self.db.run(|inner| {
            inner
                .strings
                .srandmember(key.as_ref(), |m| m.map(|m| m.to_vec()))
        })
    }

    /// Draw `count` members and leave them in the set. `SRANDMEMBER key count`.
    ///
    /// A positive `count` is distinct members, at most as many as the set holds.
    /// A negative one is the with repeats form, which answers exactly that many
    /// and can answer more members than the set has. That is one command with
    /// two meanings in Redis and it stays one method here, because splitting it
    /// would mean a caller holding a count from somewhere else has to branch on
    /// its sign before choosing which method to call.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pick_n(&self, key: impl AsRef<[u8]>, count: i64) -> Result<Vec<Vec<u8>>> {
        self.db.run(|inner| {
            let mut out = Vec::new();
            inner
                .strings
                .srandmember_n(key.as_ref(), count, |m| out.push(m.to_vec()))?;
            Ok(out)
        })
    }

    /// Move one member from one set to another, and say whether it moved.
    /// `SMOVE`.
    ///
    /// False when the member was not in `from`, in which case `to` is untouched.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], for either key.
    pub fn move_member(
        &self,
        from: impl AsRef<[u8]>,
        to: impl AsRef<[u8]>,
        member: impl AsRef<[u8]>,
    ) -> Result<bool> {
        self.db.run(|inner| {
            inner
                .strings
                .smove(from.as_ref(), to.as_ref(), member.as_ref())
        })
    }

    /// Everything in all of the sets. `SINTER`.
    ///
    /// A key that is not there is an empty set, and an empty set anywhere empties
    /// the intersection.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], for any of the keys.
    pub fn intersect<K: AsRef<[u8]>>(&self, keys: &[K]) -> Result<Vec<Vec<u8>>> {
        self.collect(keys, Op::Intersect)
    }

    /// How big the intersection is, without building it. `SINTERCARD`.
    ///
    /// A `limit` of zero means no limit. Any other limit stops the walk once it
    /// has counted that many, which is what makes "do these two sets share at
    /// least one member" cost one member and not the whole intersection.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], for any of the keys.
    pub fn intersect_len<K: AsRef<[u8]>>(&self, keys: &[K], limit: usize) -> Result<usize> {
        self.db.run(|inner| {
            inner
                .strings
                .sintercard(keys.iter().map(AsRef::as_ref), limit)
        })
    }

    /// Everything in any of the sets. `SUNION`.
    ///
    /// A key that is not there contributes nothing and is dropped, which is the
    /// opposite of what it does to an intersection and is right for the same
    /// reason: an empty set adds no members and removes none.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], for any of the keys.
    pub fn union<K: AsRef<[u8]>>(&self, keys: &[K]) -> Result<Vec<Vec<u8>>> {
        self.collect(keys, Op::Union)
    }

    /// Everything in the first set and in none of the others. `SDIFF`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], for any of the keys.
    pub fn difference<K: AsRef<[u8]>>(&self, keys: &[K]) -> Result<Vec<Vec<u8>>> {
        self.collect(keys, Op::Difference)
    }

    /// Store the intersection under `destination` and say how big it is.
    /// `SINTERSTORE`.
    ///
    /// An empty result removes `destination` rather than leaving an empty set
    /// there, because there is no such thing as an empty set in the keyspace.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], for any of the keys.
    pub fn intersect_into<K: AsRef<[u8]>>(
        &self,
        destination: impl AsRef<[u8]>,
        keys: &[K],
    ) -> Result<usize> {
        self.db.run(|inner| {
            inner
                .strings
                .sinterstore(destination.as_ref(), keys.iter().map(AsRef::as_ref))
        })
    }

    /// Store the union under `destination` and say how big it is. `SUNIONSTORE`.
    ///
    /// # Errors
    ///
    /// As [`Sets::intersect_into`].
    pub fn union_into<K: AsRef<[u8]>>(
        &self,
        destination: impl AsRef<[u8]>,
        keys: &[K],
    ) -> Result<usize> {
        self.db.run(|inner| {
            inner
                .strings
                .sunionstore(destination.as_ref(), keys.iter().map(AsRef::as_ref))
        })
    }

    /// Store the difference under `destination` and say how big it is.
    /// `SDIFFSTORE`.
    ///
    /// # Errors
    ///
    /// As [`Sets::intersect_into`].
    pub fn difference_into<K: AsRef<[u8]>>(
        &self,
        destination: impl AsRef<[u8]>,
        keys: &[K],
    ) -> Result<usize> {
        self.db.run(|inner| {
            inner
                .strings
                .sdiffstore(destination.as_ref(), keys.iter().map(AsRef::as_ref))
        })
    }

    /// The three algebra reads, which differ only in which one they call.
    ///
    /// They hand back owned members rather than borrowed ones, and that is not a
    /// choice made here. An intersection has to compare members that are stored
    /// three different ways, so by the time one is known to be in the answer it
    /// has already been written out somewhere. There is nothing left to borrow.
    fn collect<K: AsRef<[u8]>>(&self, keys: &[K], op: Op) -> Result<Vec<Vec<u8>>> {
        self.db.run(|inner| {
            let mut out = Vec::new();
            let push = |m: &[u8]| out.push(m.to_vec());
            let keys = keys.iter().map(AsRef::as_ref);
            match op {
                Op::Intersect => inner.strings.sinter(keys, 0, push)?,
                Op::Union => inner.strings.sunion(keys, push)?,
                Op::Difference => inner.strings.sdiff(keys, push)?,
            };
            Ok(out)
        })
    }
}

/// Which of the three set algebra reads [`Sets::collect`] is doing.
#[derive(Clone, Copy)]
enum Op {
    Intersect,
    Union,
    Difference,
}

/// One set, with its key held for you.
///
/// This is `15` section 2's shape: the name is spelled where the handle is made
/// and nowhere else, so a typo is one compile error at one line instead of a
/// lookup that quietly misses at three call sites.
///
/// ```
/// let db = yo::open(yo::MEMORY)?;
/// let online = db.set("online");
///
/// online.add("alice")?;
/// online.add("bob")?;
/// assert!(online.contains("alice")?);
/// assert_eq!(online.len()?, 2);
///
/// online.remove("alice")?;
/// assert!(!online.contains("alice")?);
/// # Ok::<(), yo::Error>(())
/// ```
#[derive(Clone)]
pub struct Set {
    pub(crate) sets: Sets,
    pub(crate) key: Vec<u8>,
}

impl core::fmt::Debug for Set {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Set")
            .field("key", &String::from_utf8_lossy(&self.key))
            .field("len", &self.len().ok())
            .finish()
    }
}

impl Set {
    /// The key this handle holds.
    #[must_use]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Add one member, and say whether it was new. `SADD`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn add(&self, member: impl AsRef<[u8]>) -> Result<bool> {
        self.sets.add(&self.key, member)
    }

    /// Add several members, and say how many were new.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn add_many<M: AsRef<[u8]>>(&self, members: &[M]) -> Result<usize> {
        self.sets.add_many(&self.key, members)
    }

    /// Remove one member, and say whether it was there. `SREM`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn remove(&self, member: impl AsRef<[u8]>) -> Result<bool> {
        self.sets.remove(&self.key, member)
    }

    /// Remove several members, and say how many were there.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn remove_many<M: AsRef<[u8]>>(&self, members: &[M]) -> Result<usize> {
        self.sets.remove_many(&self.key, members)
    }

    /// Whether a member is in the set. `SISMEMBER`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn contains(&self, member: impl AsRef<[u8]>) -> Result<bool> {
        self.sets.contains(&self.key, member)
    }

    /// Whether each of several members is in the set, in the order asked.
    /// `SMISMEMBER`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn contains_many<M: AsRef<[u8]>>(&self, members: &[M]) -> Result<Vec<bool>> {
        self.sets.contains_many(&self.key, members)
    }

    /// How many members it holds. `SCARD`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn len(&self) -> Result<usize> {
        self.sets.len_of(&self.key)
    }

    /// Whether it holds nothing, which is also true of a key that is not there.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn is_empty(&self) -> Result<bool> {
        self.len().map(|n| n == 0)
    }

    /// Every member, owned. `SMEMBERS`.
    ///
    /// An empty `Vec` for a key that is not there. [`Sets::members`] is the
    /// version that tells the two apart, and this one does not because a handle
    /// on a key that was never written to is the ordinary way to start.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn members(&self) -> Result<Vec<Vec<u8>>> {
        Ok(self.sets.members(&self.key)?.unwrap_or_default())
    }

    /// Hand every member to `f` where it lies, allocating nothing.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn for_each(&self, f: impl FnMut(Member<'_>)) -> Result<()> {
        self.sets.for_each(&self.key, f).map(|_| ())
    }

    /// Take one member out at random. `SPOP`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pop(&self) -> Result<Option<Vec<u8>>> {
        self.sets.pop(&self.key)
    }

    /// Take up to `count` distinct members out at random. `SPOP key count`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pop_n(&self, count: usize) -> Result<Vec<Vec<u8>>> {
        self.sets.pop_n(&self.key, count)
    }

    /// Draw one member and leave it in the set. `SRANDMEMBER`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pick(&self) -> Result<Option<Vec<u8>>> {
        self.sets.pick(&self.key)
    }

    /// Draw `count` members and leave them in the set. `SRANDMEMBER key count`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn pick_n(&self, count: i64) -> Result<Vec<Vec<u8>>> {
        self.sets.pick_n(&self.key, count)
    }

    /// Move one member into another set, and say whether it moved. `SMOVE`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`], and [`Code::Invalid`] when `to` belongs to a different
    /// database.
    pub fn move_to(&self, to: &Set, member: impl AsRef<[u8]>) -> Result<bool> {
        self.same_db(to)?;
        self.sets.move_member(&self.key, &to.key, member)
    }

    /// Everything in this set and in all of `others`. `SINTER`.
    ///
    /// ```
    /// let db = yo::open(yo::MEMORY)?;
    /// let a = db.set("a");
    /// let b = db.set("b");
    /// a.add_many(&["x", "y"])?;
    /// b.add_many(&["y", "z"])?;
    ///
    /// assert_eq!(a.intersect(&[&b])?, vec![b"y".to_vec()]);
    /// # Ok::<(), yo::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn intersect(&self, others: &[&Set]) -> Result<Vec<Vec<u8>>> {
        self.sets.intersect(&self.keys_with(others)?)
    }

    /// How many members this set shares with all of `others`. `SINTERCARD`.
    ///
    /// A `limit` of zero means no limit.
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn intersect_len(&self, others: &[&Set], limit: usize) -> Result<usize> {
        self.sets.intersect_len(&self.keys_with(others)?, limit)
    }

    /// Everything in this set or in any of `others`. `SUNION`.
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn union(&self, others: &[&Set]) -> Result<Vec<Vec<u8>>> {
        self.sets.union(&self.keys_with(others)?)
    }

    /// Everything in this set and in none of `others`. `SDIFF`.
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn difference(&self, others: &[&Set]) -> Result<Vec<Vec<u8>>> {
        self.sets.difference(&self.keys_with(others)?)
    }

    /// Store the intersection in `destination` and say how big it is.
    /// `SINTERSTORE`.
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn intersect_into(&self, destination: &Set, others: &[&Set]) -> Result<usize> {
        self.same_db(destination)?;
        self.sets
            .intersect_into(&destination.key, &self.keys_with(others)?)
    }

    /// Store the union in `destination` and say how big it is. `SUNIONSTORE`.
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn union_into(&self, destination: &Set, others: &[&Set]) -> Result<usize> {
        self.same_db(destination)?;
        self.sets
            .union_into(&destination.key, &self.keys_with(others)?)
    }

    /// Store the difference in `destination` and say how big it is.
    /// `SDIFFSTORE`.
    ///
    /// # Errors
    ///
    /// As [`Set::move_to`].
    pub fn difference_into(&self, destination: &Set, others: &[&Set]) -> Result<usize> {
        self.same_db(destination)?;
        self.sets
            .difference_into(&destination.key, &self.keys_with(others)?)
    }

    /// Remove the whole set, and say whether it was there. `DEL`.
    ///
    /// # Errors
    ///
    /// As [`Sets::add`].
    pub fn clear(&self) -> Result<bool> {
        self.sets.db.run(|inner| Ok(inner.strings.del(&self.key)))
    }

    /// This key followed by the others', once every one of them is checked to
    /// belong here.
    fn keys_with<'a>(&'a self, others: &[&'a Set]) -> Result<Vec<&'a [u8]>> {
        let mut keys = Vec::with_capacity(others.len() + 1);
        keys.push(&self.key[..]);
        for other in others {
            self.same_db(other)?;
            keys.push(&other.key[..]);
        }
        Ok(keys)
    }

    /// The check that keeps two databases from being intersected with each
    /// other.
    ///
    /// Without it, a handle from another database contributes its key and not
    /// its contents, so the answer is computed against whatever this database
    /// happens to hold under that name. That is not an empty answer or an error,
    /// it is a plausible wrong one, and it would be a very hard afternoon.
    fn same_db(&self, other: &Set) -> Result<()> {
        if self.sets.db.is(&other.sets.db) {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            "those two sets are in different databases, and a set operation reads both of them out of one. Open both handles on the same Db",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MEMORY, open};

    #[test]
    fn the_set_commands_are_the_ones_a_redis_client_would_send() {
        let db = open(MEMORY).unwrap();
        let sets = db.sets();

        assert!(sets.add("online", "alice").unwrap());
        assert!(!sets.add("online", "alice").unwrap());
        assert_eq!(sets.add_many("online", &["bob", "carol"]).unwrap(), 2);

        assert!(sets.contains("online", "bob").unwrap());
        assert!(!sets.contains("online", "dave").unwrap());
        assert_eq!(
            sets.contains_many("online", &["alice", "dave"]).unwrap(),
            vec![true, false]
        );
        assert_eq!(sets.len_of("online").unwrap(), 3);

        assert!(sets.remove("online", "alice").unwrap());
        assert_eq!(sets.remove_many("online", &["bob", "dave"]).unwrap(), 1);
        assert_eq!(sets.len_of("online").unwrap(), 1);
    }

    /// A key that is not there answers the way an empty set answers, everywhere
    /// it can, because a set nobody has added to and a set somebody emptied are
    /// the same set.
    #[test]
    fn a_set_that_is_not_there_reads_as_an_empty_one() {
        let db = open(MEMORY).unwrap();
        let sets = db.sets();

        assert_eq!(sets.len_of("nope").unwrap(), 0);
        assert!(!sets.contains("nope", "x").unwrap());
        assert_eq!(sets.pop("nope").unwrap(), None);
        assert_eq!(sets.pick("nope").unwrap(), None);
        assert!(sets.pop_n("nope", 5).unwrap().is_empty());
        assert!(sets.pick_n("nope", 5).unwrap().is_empty());
        // The one place it does not: SMEMBERS can say which it was.
        assert_eq!(sets.members("nope").unwrap(), None);
        assert!(!sets.for_each("nope", |_| {}).unwrap());
    }

    /// The point of the borrowed walk. A set of integers is stored as integers,
    /// and this is the read that does not turn them back into text.
    #[test]
    fn walking_a_set_of_integers_never_formats_a_digit() {
        let db = open(MEMORY).unwrap();
        let ids = db.set("ids");
        ids.add_many(&["1", "2", "3"]).unwrap();

        let mut total = 0i64;
        let mut ints = 0;
        ids.for_each(|m| {
            if let Member::Int(n) = m {
                total += n;
                ints += 1;
            }
        })
        .unwrap();
        assert_eq!(total, 6);
        assert_eq!(ints, 3, "stored as integers, so handed over as integers");

        // And the owned read gives back what a client would have seen.
        let mut owned = ids.members().unwrap();
        owned.sort();
        assert_eq!(owned, vec![b"1".to_vec(), b"2".to_vec(), b"3".to_vec()]);
    }

    #[test]
    fn a_handle_holds_the_key_so_the_caller_does_not() {
        let db = open(MEMORY).unwrap();
        let online = db.set("online");

        online.add("alice").unwrap();
        online.add_many(&["bob", "carol"]).unwrap();
        assert_eq!(online.len().unwrap(), 3);
        assert!(!online.is_empty().unwrap());
        assert!(online.contains("bob").unwrap());
        assert_eq!(online.key(), b"online");
        assert!(format!("{online:?}").contains("online"));

        // And it is the same key the keyspace door sees.
        assert_eq!(db.sets().len_of("online").unwrap(), 3);
    }

    #[test]
    fn a_set_that_loses_its_last_member_loses_its_key() {
        let db = open(MEMORY).unwrap();
        let only = db.set("only");

        only.add("x").unwrap();
        assert!(only.remove("x").unwrap());
        assert!(only.is_empty().unwrap());
        assert_eq!(db.sets().members("only").unwrap(), None);
    }

    #[test]
    fn drawing_takes_members_out_and_picking_leaves_them() {
        let db = open(MEMORY).unwrap();
        let bag = db.set("bag");
        bag.add_many(&["a", "b", "c", "d"]).unwrap();

        assert!(bag.pick().unwrap().is_some());
        assert_eq!(bag.len().unwrap(), 4, "picking leaves the set alone");
        assert_eq!(bag.pick_n(3).unwrap().len(), 3);
        assert_eq!(bag.len().unwrap(), 4);
        // The with repeats form is the only one that can answer more members
        // than the set holds.
        assert_eq!(bag.pick_n(-9).unwrap().len(), 9);
        assert_eq!(bag.len().unwrap(), 4);

        assert!(bag.pop().unwrap().is_some());
        assert_eq!(bag.len().unwrap(), 3, "popping takes one out");
        assert_eq!(bag.pop_n(9).unwrap().len(), 3, "and never more than it has");
        assert!(bag.is_empty().unwrap());
    }

    #[test]
    fn the_three_set_operations_answer_what_they_are_named_after() {
        let db = open(MEMORY).unwrap();
        let a = db.set("a");
        let b = db.set("b");
        a.add_many(&["x", "y"]).unwrap();
        b.add_many(&["y", "z"]).unwrap();

        let sorted = |mut v: Vec<Vec<u8>>| {
            v.sort();
            v
        };
        assert_eq!(sorted(a.intersect(&[&b]).unwrap()), vec![b"y".to_vec()]);
        assert_eq!(
            sorted(a.union(&[&b]).unwrap()),
            vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]
        );
        assert_eq!(sorted(a.difference(&[&b]).unwrap()), vec![b"x".to_vec()]);
        assert_eq!(a.intersect_len(&[&b], 0).unwrap(), 1);

        let out = db.set("out");
        assert_eq!(a.union_into(&out, &[&b]).unwrap(), 3);
        assert_eq!(out.len().unwrap(), 3);
        assert_eq!(a.intersect_into(&out, &[&b]).unwrap(), 1);
        assert_eq!(out.len().unwrap(), 1);
        assert_eq!(a.difference_into(&out, &[&b]).unwrap(), 1);
        assert_eq!(out.members().unwrap(), vec![b"x".to_vec()]);
    }

    /// An empty result removes the destination rather than leaving an empty set
    /// behind, because there is no such thing as an empty set in the keyspace.
    #[test]
    fn storing_an_empty_result_removes_the_destination() {
        let db = open(MEMORY).unwrap();
        let a = db.set("a");
        let b = db.set("b");
        a.add("x").unwrap();
        b.add("y").unwrap();

        let out = db.set("out");
        out.add("stale").unwrap();
        assert_eq!(a.intersect_into(&out, &[&b]).unwrap(), 0);
        assert_eq!(db.sets().members("out").unwrap(), None);
    }

    #[test]
    fn a_member_moves_between_two_sets() {
        let db = open(MEMORY).unwrap();
        let from = db.set("from");
        let to = db.set("to");
        from.add_many(&["x", "y"]).unwrap();

        assert!(from.move_to(&to, "x").unwrap());
        assert!(!from.contains("x").unwrap());
        assert!(to.contains("x").unwrap());
        // A member that was not there does not move and does not create one.
        assert!(!from.move_to(&to, "nope").unwrap());
        assert_eq!(to.len().unwrap(), 1);
    }

    /// The failure this refuses to have. Two databases, both with a key called
    /// `b`, and an intersection that reads the wrong one. It would not error and
    /// it would not come back empty, it would come back plausible.
    #[test]
    fn two_databases_cannot_be_intersected_with_each_other() {
        let one = open(MEMORY).unwrap();
        let two = open(MEMORY).unwrap();

        let a = one.set("a");
        a.add_many(&["x", "y"]).unwrap();
        // The decoy: `two` has a `b` that shares a member with `a`, so an
        // unchecked intersection would answer `x` rather than fail.
        one.set("b").add("z").unwrap();
        let elsewhere = two.set("b");
        elsewhere.add("x").unwrap();

        let e = a.intersect(&[&elsewhere]).expect_err("different databases");
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("different databases"), "{e}");

        // Every other door into another database is shut the same way.
        assert!(a.union(&[&elsewhere]).is_err());
        assert!(a.difference(&[&elsewhere]).is_err());
        assert!(a.intersect_len(&[&elsewhere], 0).is_err());
        assert!(a.move_to(&elsewhere, "x").is_err());
        assert!(a.intersect_into(&elsewhere, &[]).is_err());
        assert!(a.union_into(&elsewhere, &[]).is_err());
        assert!(a.difference_into(&elsewhere, &[]).is_err());
    }

    /// The embedded door and the wire door are one store (Y23), so a set added
    /// here is a set the keyspace holds and a `DEL` here removes it.
    #[test]
    fn a_set_and_the_keyspace_are_the_same_store() {
        let db = open(MEMORY).unwrap();
        let tags = db.set("tags");
        tags.add("rust").unwrap();

        // A string command on a key holding a set is the same WRONGTYPE a
        // client would get.
        let e = db.strings().get("tags").expect_err("that is a set");
        assert_eq!(e.code(), Code::WrongType);
        // And the other way round.
        db.strings().set("word", "nope").unwrap();
        assert_eq!(db.set("word").add("x").unwrap_err().code(), Code::WrongType);

        assert!(tags.clear().unwrap());
        assert!(tags.is_empty().unwrap());
        assert!(!tags.clear().unwrap());
    }

    #[test]
    fn a_member_is_bytes_and_not_only_text() {
        let db = open(MEMORY).unwrap();
        let raw = db.set("raw");

        raw.add(vec![0u8, 0xff]).unwrap();
        assert!(raw.contains(b"\x00\xff").unwrap());
        assert_eq!(raw.members().unwrap(), vec![vec![0u8, 0xff]]);
    }
}
