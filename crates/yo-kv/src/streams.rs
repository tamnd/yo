//! The stream commands.
//!
//! One method per Redis command on [`Keyspace`], the same arrangement the list
//! and set commands use and for the same reason: a key belongs to the database
//! and not to a type, so `XADD` against a string has to be able to see that it
//! is a string. The log itself is [`crate::stream`] and the groups are
//! [`crate::stream::groups`]. This file is what the wire and the embedded API
//! both call.
//!
//! # An empty stream is still a stream
//!
//! Every other collection here disappears when its last element goes, so
//! `EXISTS` answers zero after the last `LPOP`. A stream does not. `XDEL` of
//! every entry leaves the key, and so does a `MAXLEN 0` trim, because the
//! stream is still carrying the last ID it handed out and the groups reading
//! it, and throwing those away because the entries have aged out would hand the
//! same IDs out twice.
//!
//! That is why the methods here do not have the create on write and drop on
//! empty pair the other four types have. There is a create, [`Keyspace::xadd`]
//! and the `MKSTREAM` half of [`Keyspace::xgroup_create`], and there is no
//! drop.
//!
//! # Reading is one method and not eight
//!
//! `XINFO` reads a dozen fields off a stream and none of them are decisions.
//! Rather than a method here per field, [`Keyspace::stream`] hands the wire the
//! stream and it reads what it needs, which is the same borrow it would have
//! got and a great deal less to keep in step. Everything that changes the
//! stream, or that has to say something about a key that is missing or is the
//! wrong type, is a method.

use yo_common::{Code, Error, Result};

use crate::keyspace::Keyspace;
use crate::stream::groups::Filter;
use crate::stream::{Fate, Fields, Group, Id, Refs, Refused, Retry, Stream};
use crate::value::{self, Kind};

/// What every stream command says about an ID it cannot read.
pub const BAD_ID: &str = "Invalid stream ID specified as stream command argument";

/// What `XADD` says about an ID that is not above the last one.
pub const NOT_GREATER: &str =
    "The ID specified in XADD is equal or smaller than the target stream top item";

/// What it says about `0-0`, which no entry can have because nothing sorts below it.
pub const ZERO_ID: &str = "The ID specified in XADD must be greater than 0-0";

/// What it says when the stream is at the last ID there is.
pub const EXHAUSTED: &str =
    "The stream has exhausted the last possible ID, unable to add more items";

/// What `XGROUP` says about a key that is not there.
///
/// Redis's wording, run on sentence and all, because it goes on the wire
/// verbatim and a client's test suite compares it.
pub const NO_KEY_FOR_GROUP: &str = "The XGROUP subcommand requires the key to exist. Note that for CREATE you may want to use the MKSTREAM option to create an empty stream automatically.";

/// What `XGROUP CREATE` says about a group that is already there.
///
/// This one goes out under a `BUSYGROUP` prefix rather than `ERR`, which the
/// wire layer writes where it decides it, the same way it writes `NOPROTO` and
/// `WRONGPASS`.
pub const GROUP_EXISTS: &str = "Consumer Group name already exists";

/// What `XSETID` says about an ID below an entry that is still there.
pub const SETID_TOO_SMALL: &str =
    "The ID specified in XSETID is smaller than the target stream top item";

/// And what it says about an ID below the `MAXDELETEDID` it was handed.
///
/// A separate sentence because it is a separate mistake. The one above is about
/// an entry the stream still holds and this is about one it says it deleted, and
/// a stream whose last ID sat below its own high water mark for deletions would
/// hand that ID out again.
pub const SETID_BELOW_MAX_DELETED: &str =
    "The ID specified in XSETID is smaller than the provided max_deleted_entry_id";

/// What a command that needs the key says when it is not there.
pub const NO_SUCH_KEY: &str = "no such key";

/// What `XGROUP` and `XINFO CONSUMERS` say about a group that is not there.
///
/// A function rather than a constant because Redis puts the group and the key
/// in it, and a client watching for a particular group in its logs is reading
/// exactly that. It goes out under a `NOGROUP` prefix.
///
/// There are three of these and they are not interchangeable. This is the one
/// the commands that only ever name one key use. Every wording here was read
/// off Redis 8.10.1, since the difference between them is not something the
/// documentation mentions and a client library matching on the text would break
/// on a paraphrase.
#[must_use]
pub fn no_group(group: &[u8], key: &[u8]) -> String {
    format!(
        "No such consumer group '{}' for key name '{}'",
        String::from_utf8_lossy(group),
        String::from_utf8_lossy(key)
    )
}

/// What `XPENDING`, `XCLAIM` and `XAUTOCLAIM` say instead.
///
/// The key comes first here and the sentence allows for either half being the
/// missing one, because these commands cannot tell a stream with no such group
/// from a key that is not a stream at all without looking twice.
#[must_use]
pub fn no_key_or_group(key: &[u8], group: &[u8]) -> String {
    format!(
        "No such key '{}' or consumer group '{}'",
        String::from_utf8_lossy(key),
        String::from_utf8_lossy(group)
    )
}

/// And what `XREADGROUP` says, which is the one above with its own tail.
///
/// The tail is there because `XREADGROUP` is the command people send by
/// accident against a stream they never made a group on, so Redis spells out
/// which option is at fault.
#[must_use]
pub fn no_group_for_read(key: &[u8], group: &[u8]) -> String {
    format!(
        "No such key '{}' or consumer group '{}' in XREADGROUP with GROUP option",
        String::from_utf8_lossy(key),
        String::from_utf8_lossy(group)
    )
}

/// What `XADD` was told to use for the ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Add {
    /// `*`, which is the clock, or the last ID with one added if the clock has
    /// not moved.
    Auto,
    /// `5-*`, which is that millisecond and the next free sequence inside it.
    Seq(u64),
    /// `5-3`, which is exactly that and fails if it is not above the last one.
    At(Id),
}

/// What a trim was told to cut down to.
///
/// `exact` is Redis's `=` against `~`. Without it only whole nodes go, so the
/// stream is left at the threshold or a little over and no node is ever
/// rewritten, which is the form to use and is why `~` exists.
///
/// `limit` only ever arrives with `~`, because Redis refuses the two together
/// with `=` and the wire layer refuses it here for the same reason: the limit
/// is a brake on how long one command runs, and an exact trim that stopped
/// early would not have done what it was asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trim {
    /// No trim, which is `XADD` without a `MAXLEN` or a `MINID`.
    None,
    /// `MAXLEN n`, which keeps the newest `n` entries.
    MaxLen {
        /// How many to keep.
        len: u64,
        /// Redis's `=` rather than `~`.
        exact: bool,
        /// `LIMIT`, which stops the trim after that many entries have gone.
        limit: Option<u64>,
    },
    /// `MINID id`, which drops everything below `id`.
    MinId {
        /// The lowest ID to keep.
        id: Id,
        /// Redis's `=` rather than `~`.
        exact: bool,
        /// `LIMIT`, as above.
        limit: Option<u64>,
    },
}

/// Where a group's bookmark is being put, which is `XGROUP CREATE` and `SETID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// `$`, the last ID the stream has handed out, so the group sees only what
    /// arrives after it was made.
    Last,
    /// An ID, usually `0`, so the group reads the stream from there.
    At(Id),
}

/// What `XREADGROUP` was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum From {
    /// `>`, meaning entries no consumer in this group has been given yet.
    New,
    /// An ID, meaning what this consumer is already holding above that ID.
    ///
    /// Redis counts this as a real delivery, so the entries come back with
    /// their delivery time reset and their count up by one.
    Pending(Id),
}

/// Everything `XREADGROUP` takes past the key.
///
/// A struct for the same reason [`Claim`] is one, and because the wire parses
/// `GROUP g c`, `COUNT n` and `NOACK` in any order and would otherwise be
/// carrying five loose values from the parse to the call.
#[derive(Debug, Clone, Copy)]
pub struct Read<'a> {
    /// Which group is reading.
    pub group: &'a [u8],
    /// Which consumer inside it, created by turning up if it is new.
    pub consumer: &'a [u8],
    /// `>` or an ID, which are two quite different commands wearing one name.
    pub from: From,
    /// `COUNT`, or everything there is.
    pub count: Option<usize>,
    /// `NOACK`, which hands the entries over without writing them down.
    pub noack: bool,
}

/// Everything `XCLAIM` takes past the key and the IDs.
///
/// A struct rather than seven more arguments because the command parses them as
/// one thing and they travel together from the wire to here, and because
/// `XAUTOCLAIM` takes the same set with one extra of its own.
#[derive(Debug, Clone, Copy)]
pub struct Claim<'a> {
    /// Which group's pending list is being moved around.
    pub group: &'a [u8],
    /// Which consumer ends up holding what is claimed.
    pub consumer: &'a [u8],
    /// Skip anything idle less than this many milliseconds.
    pub min_idle: u64,
    /// What to set the delivery time to, which `IDLE` and `TIME` both work out.
    pub time: u64,
    /// `RETRYCOUNT`, or leave the count where it is.
    pub retry: Option<u64>,
    /// Whether the delivery count goes up, which it does unless `JUSTID` was
    /// asked for.
    pub bump: bool,
    /// `FORCE`, which makes a pending entry for one that is in the stream and
    /// was never handed out.
    pub force: bool,
}

impl Default for Claim<'_> {
    fn default() -> Claim<'static> {
        Claim {
            group: b"",
            consumer: b"",
            min_idle: 0,
            time: 0,
            retry: None,
            bump: true,
            force: false,
        }
    }
}

impl Keyspace {
    /// The stream under `key`, for reading.
    ///
    /// `None` for a key that is not there or has expired, and an error for one
    /// holding something else, which is the three way answer every type's entry
    /// point here gives. `XINFO` and `XLEN` go through this rather than through
    /// a method each, because reading a field off a stream is not a decision.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding anything but a stream.
    pub fn stream(&mut self, key: &[u8]) -> Result<Option<&Stream>> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(None);
        };
        Ok(self.streams.get(at))
    }

    /// The same, for a caller that is going to change it.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding anything but a stream.
    pub fn stream_mut(&mut self, key: &[u8]) -> Result<Option<&mut Stream>> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(None);
        };
        Ok(self.streams.get_mut(at))
    }

    /// `XADD key [NOMKSTREAM] [trim] id field value [field value ...]`.
    ///
    /// Answers the ID that was written, or `None` when `NOMKSTREAM` was asked
    /// for and the key was not there.
    ///
    /// The trim runs after the append, which is Redis's order and matters when
    /// the threshold is `MAXLEN 1`: the entry that was just written is the one
    /// that survives.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::Invalid`] for an ID that is zero, is not above the last one, or
    /// asks for a sequence inside a millisecond that has already filled up.
    pub fn xadd(
        &mut self,
        key: &[u8],
        id: Add,
        fields: &[(&[u8], &[u8])],
        trim: Trim,
        mkstream: bool,
        now: u64,
    ) -> Result<Option<Id>> {
        let limits = self.stream_limits;
        let at = match self.live_slot(key, Kind::Stream)? {
            Some(at) => at,
            None if mkstream => self.new_stream(key),
            None => return Ok(None),
        };
        let s = self.stream_at(at);
        // Worked out against the stream as it is, before anything is written,
        // so that a `*` that has nowhere to go is an error and not a panic.
        let want = match id {
            Add::Auto => s.auto_id(now).ok_or_else(exhausted)?,
            Add::Seq(ms) => s.auto_seq(ms).ok_or_else(|| {
                if ms < s.last_id().ms {
                    Error::new(Code::Invalid, NOT_GREATER)
                } else {
                    exhausted()
                }
            })?,
            Add::At(id) => id,
        };
        s.append(want, fields, limits).map_err(refused)?;
        cut(s, trim);
        Ok(Some(want))
    }

    /// `XDEL key id [id ...]`. Answers how many were there to delete.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xdel(&mut self, key: &[u8], ids: impl Iterator<Item = Id>) -> Result<u64> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(0);
        };
        let s = self.stream_at(at);
        Ok(ids.filter(|&id| s.delete(id)).count() as u64)
    }

    /// `XDELEX key [KEEPREF|DELREF|ACKED] IDS numids id [id ...]`.
    ///
    /// The callback gets what became of each ID, in the order they were given. A
    /// key that is not there is not an error and not a short reply either: every
    /// ID gets [`Fate::Missing`], which is what a real server answers and is why
    /// the ID list is walked even when there is nothing to walk it against.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xdelex<F>(
        &mut self,
        key: &[u8],
        refs: Refs,
        ids: impl Iterator<Item = Id>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Fate),
    {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            ids.for_each(|_| f(Fate::Missing));
            return Ok(());
        };
        let s = self.stream_at(at);
        ids.for_each(|id| f(s.delete_ref(id, refs)));
        Ok(())
    }

    /// `XACKDEL key group [KEEPREF|DELREF|ACKED] IDS numids id [id ...]`.
    ///
    /// The same shape, and a group that is not there behaves like a key that is
    /// not there rather than raising `NOGROUP`, because the answer this command
    /// gives per ID is about the pending list and an absent group is holding
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xackdel<F>(
        &mut self,
        key: &[u8],
        group: &[u8],
        refs: Refs,
        ids: impl Iterator<Item = Id>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Fate),
    {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            ids.for_each(|_| f(Fate::Missing));
            return Ok(());
        };
        let s = self.stream_at(at);
        ids.for_each(|id| f(s.ack_delete(group, id, refs)));
        Ok(())
    }

    /// `XNACK key group <SILENT|FAIL|FATAL> IDS numids id [id ...] [RETRYCOUNT n] [FORCE]`.
    ///
    /// Answers how many entries were released, and `None` when there is no such
    /// key or group, which this command does raise `NOGROUP` for.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xnack(
        &mut self,
        key: &[u8],
        group: &[u8],
        retry: Retry,
        force: bool,
        ids: impl Iterator<Item = Id>,
    ) -> Result<Option<u64>> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(None);
        };
        let s = self.stream_at(at);
        if s.group(group).is_none() {
            return Ok(None);
        }
        let mut done = 0;
        for id in ids {
            done += u64::from(s.nack(group, id, retry, force).unwrap_or(false));
        }
        Ok(Some(done))
    }

    /// `XTRIM key strategy`. Answers how many entries went.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xtrim(&mut self, key: &[u8], trim: Trim) -> Result<u64> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(0);
        };
        Ok(cut(self.stream_at(at), trim))
    }

    /// `XSETID key id [ENTRIESADDED n] [MAXDELETEDID id]`.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else,
    /// [`Code::NotFound`] for a key that is not there, and [`Code::Invalid`]
    /// for an ID below an entry the stream still holds.
    pub fn xsetid(
        &mut self,
        key: &[u8],
        last: Id,
        added: Option<u64>,
        max_deleted: Option<Id>,
    ) -> Result<()> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Err(Error::new(Code::NotFound, NO_SUCH_KEY));
        };
        // Checked here rather than in `Stream::set_id`, because it is a rule
        // about the two arguments and not about the stream: the pair is
        // contradictory whatever the stream currently holds.
        if max_deleted.is_some_and(|id| last < id) {
            return Err(Error::new(Code::Invalid, SETID_BELOW_MAX_DELETED));
        }
        self.stream_at(at)
            .set_id(last, added, max_deleted)
            .map_err(|_| Error::new(Code::Invalid, SETID_TOO_SMALL))
    }

    /// `XRANGE` and `XREVRANGE`, which differ only in the direction.
    ///
    /// `start` and `end` are the low and the high end either way, so the wire
    /// layer swaps `XREVRANGE`'s arguments once rather than every reader here
    /// working out which is which. Answers how many entries the callback saw.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xrange_into<F>(
        &mut self,
        key: &[u8],
        start: Id,
        end: Id,
        count: Option<usize>,
        rev: bool,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(Id, Fields<'_>) -> bool,
    {
        let Some(s) = self.stream(key)? else {
            return Ok(0);
        };
        Ok(if rev {
            s.rev_range(start, end, count, f)
        } else {
            s.range(start, end, count, f)
        })
    }

    /// `XREAD ... STREAMS key id`, which is a plain range with no group.
    ///
    /// Everything after `after`, up to `count`. Answers how many the callback
    /// saw, which is zero for a key that is not there, because `XREAD` on a
    /// missing key is nothing to report rather than an error.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xread_into<F>(
        &mut self,
        key: &[u8],
        after: Id,
        count: Option<usize>,
        f: F,
    ) -> Result<usize>
    where
        F: FnMut(Id, Fields<'_>) -> bool,
    {
        let Some(from) = after.next() else {
            return Ok(0);
        };
        self.xrange_into(key, from, Id::MAX, count, false, f)
    }

    /// `XGROUP CREATE key group id [MKSTREAM] [ENTRIESREAD n]`.
    ///
    /// Answers whether the group was made, which is `false` when one of that
    /// name was already there and is the `BUSYGROUP` the wire reports.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there without `MKSTREAM`.
    pub fn xgroup_create(
        &mut self,
        key: &[u8],
        group: &[u8],
        at: Start,
        mkstream: bool,
        read: Option<u64>,
    ) -> Result<bool> {
        let slot = match self.live_slot(key, Kind::Stream)? {
            Some(slot) => slot,
            None if mkstream => self.new_stream(key),
            None => return Err(Error::new(Code::NotFound, NO_KEY_FOR_GROUP)),
        };
        let s = self.stream_at(slot);
        let last = position(s, at);
        // `read` and not a zero for a group with no `ENTRIESREAD`. A fresh group
        // does not know how many entries are behind it, and saying zero would be
        // a claim rather than a default: `XINFO GROUPS` reports the counter as
        // null on a real server until something sets it, and the lag is worked
        // out from where the bookmark sits instead.
        let read = capped(read, s);
        Ok(s.create_group(group, last, read))
    }

    /// `XGROUP DESTROY key group`. Answers whether there was one.
    ///
    /// A group that is not there is a zero and a key that is not there is an
    /// error, which is Redis's rule for every `XGROUP` subcommand and is worth
    /// stating because the two look like the same kind of nothing from a client.
    /// They are not: destroying a group nobody made is a no op, and destroying a
    /// group on a key nobody made is a mistake about which key.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there.
    pub fn xgroup_destroy(&mut self, key: &[u8], group: &[u8]) -> Result<bool> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Err(Error::new(Code::NotFound, NO_KEY_FOR_GROUP));
        };
        Ok(self.stream_at(at).destroy_group(group))
    }

    /// `XGROUP SETID key group id [ENTRIESREAD n]`.
    ///
    /// `None` when there is no such group, which the wire reports as `NOGROUP`.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there.
    pub fn xgroup_setid(
        &mut self,
        key: &[u8],
        group: &[u8],
        at: Start,
        read: Option<u64>,
    ) -> Result<Option<()>> {
        let Some(slot) = self.live_slot(key, Kind::Stream)? else {
            return Err(Error::new(Code::NotFound, NO_KEY_FOR_GROUP));
        };
        let s = self.stream_at(slot);
        let last = position(s, at);
        let read = capped(read, s);
        let Some(g) = s.group_mut(group) else {
            return Ok(None);
        };
        // `read` and not the group's old counter when nothing was named, so
        // `XGROUP SETID key group 0` gives the counter up rather than leaving
        // one that was true of somewhere else. That is what a real server does
        // and it is visible immediately: `XINFO GROUPS` reports both the counter
        // and the lag as null afterwards, until a read or an `ENTRIESREAD` puts
        // a number back.
        g.set_id(last, read);
        Ok(Some(()))
    }

    /// `XGROUP CREATECONSUMER key group consumer`.
    ///
    /// Answers whether the consumer was made, and `None` when there is no such
    /// group.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there.
    pub fn xgroup_create_consumer(
        &mut self,
        key: &[u8],
        group: &[u8],
        consumer: &[u8],
        now: u64,
    ) -> Result<Option<bool>> {
        let Some(g) = self.group_mut_of(key, group)? else {
            return Ok(None);
        };
        Ok(Some(g.create_consumer(consumer, now)))
    }

    /// `XGROUP DELCONSUMER key group consumer`.
    ///
    /// Answers how many pending entries went with it, and `None` when there is
    /// no such group.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there.
    pub fn xgroup_del_consumer(
        &mut self,
        key: &[u8],
        group: &[u8],
        consumer: &[u8],
    ) -> Result<Option<u64>> {
        let Some(g) = self.group_mut_of(key, group)? else {
            return Ok(None);
        };
        Ok(Some(g.delete_consumer(consumer)))
    }

    /// `XREADGROUP GROUP group consumer [COUNT n] [NOACK] STREAMS key id`.
    ///
    /// Answers how many entries the callback saw, and `None` when there is no
    /// such group. The callback takes an `Option` because a history read can
    /// name an entry that has since been deleted, and Redis puts a null in the
    /// reply for it rather than leaving it out.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there, which is the `NOGROUP`
    /// Redis answers because a missing key cannot have the group either.
    pub fn xreadgroup_into<F>(
        &mut self,
        key: &[u8],
        want: Read<'_>,
        now: u64,
        mut f: F,
    ) -> Result<Option<usize>>
    where
        F: FnMut(Id, Option<Fields<'_>>) -> bool,
    {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(None);
        };
        let s = self.stream_at(at);
        Ok(match want.from {
            From::New => s.read_group(
                want.group,
                want.consumer,
                want.count,
                want.noack,
                now,
                |id, fields| f(id, Some(fields)),
            ),
            From::Pending(after) => {
                s.read_group_pending(want.group, want.consumer, after, want.count, now, &mut f)
            }
        })
    }

    /// `XACK key group id [id ...]`. Answers how many were pending.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xack(&mut self, key: &[u8], group: &[u8], ids: impl Iterator<Item = Id>) -> Result<u64> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(0);
        };
        let Some(g) = self.stream_at(at).group_mut(group) else {
            return Ok(0);
        };
        Ok(ids.filter(|&id| g.ack(id)).count() as u64)
    }

    /// `XPENDING key group [[IDLE ms] start end count [consumer]]`, the long form.
    ///
    /// The callback gets each entry with its NACK and its owner. Answers how
    /// many it saw, and `None` when there is no such group.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xpending_into<F>(
        &mut self,
        key: &[u8],
        group: &[u8],
        want: Filter,
        now: u64,
        f: F,
    ) -> Result<Option<usize>>
    where
        F: FnMut(Id, &crate::stream::Nack, Option<&crate::stream::Consumer>) -> bool,
    {
        let Some(s) = self.stream(key)? else {
            return Ok(None);
        };
        let Some(g) = s.group(group) else {
            return Ok(None);
        };
        Ok(Some(g.pending_range(want, now, f)))
    }

    /// `XCLAIM key group consumer min-idle-time id [id ...]`.
    ///
    /// Answers what was claimed, and fills `gone` with the IDs that were in the
    /// pending list and are no longer in the stream, which the claim clears out
    /// on the way past because nobody can ever finish them. `None` when there
    /// is no such group.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xclaim(
        &mut self,
        key: &[u8],
        ids: &[Id],
        how: Claim<'_>,
        now: u64,
        gone: &mut Vec<Id>,
    ) -> Result<Option<Vec<Id>>> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(None);
        };
        Ok(self.stream_at(at).claim(
            how.group,
            how.consumer,
            ids,
            how.min_idle,
            how.time,
            how.retry,
            how.bump,
            how.force,
            now,
            gone,
        ))
    }

    /// `XAUTOCLAIM key group consumer min-idle-time start [COUNT n] [JUSTID]`.
    ///
    /// Answers where a following call should carry on from, which is `None` at
    /// the end of the list and is the `0-0` Redis replies with, along with what
    /// was claimed. `gone` is filled the same way [`Keyspace::xclaim`] fills it.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else.
    pub fn xautoclaim(
        &mut self,
        key: &[u8],
        start: Id,
        how: Claim<'_>,
        count: usize,
        now: u64,
        gone: &mut Vec<Id>,
    ) -> Result<Option<(Option<Id>, Vec<Id>)>> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Ok(None);
        };
        Ok(self.stream_at(at).autoclaim(
            how.group,
            how.consumer,
            start,
            how.min_idle,
            count,
            how.bump,
            now,
            gone,
        ))
    }

    /// The group under a key, for the three commands that only touch the group.
    ///
    /// # Errors
    ///
    /// [`Code::WrongType`] for a key holding something else, and
    /// [`Code::NotFound`] for a key that is not there, which is what `XGROUP`
    /// says about all of its subcommands.
    fn group_mut_of(&mut self, key: &[u8], group: &[u8]) -> Result<Option<&mut Group>> {
        let Some(at) = self.live_slot(key, Kind::Stream)? else {
            return Err(Error::new(Code::NotFound, NO_KEY_FOR_GROUP));
        };
        Ok(self.stream_at(at).group_mut(group))
    }

    fn stream_at(&mut self, at: u32) -> &mut Stream {
        self.streams
            .get_mut(at)
            .expect("the record points at its body")
    }

    fn new_stream(&mut self, key: &[u8]) -> u32 {
        let at = self.streams.insert(Stream::new());
        let len = value::slot_record_len(false);
        self.write_rec(key, len, |out| {
            value::write_slot_record(out, Kind::Stream, at, None);
        });
        self.bodies += 1;
        at
    }
}

/// Where a bookmark goes for a `$` or for an ID.
fn position(s: &Stream, at: Start) -> Id {
    match at {
        Start::Last => s.last_id(),
        Start::At(id) => id,
    }
}

/// An `ENTRIESREAD` held down to what the stream has ever added.
///
/// A group cannot have read more entries than were ever written, so a client
/// that says it has is corrected rather than believed. Redis does the same and
/// it is visible: `XGROUP CREATE key g 0 ENTRIESREAD 99` on a stream of three
/// reports three afterwards, not ninety nine. It matters because the number is
/// subtracted from the entry count to get the lag, and an inflated one would
/// make the lag come out at zero on a group that has read nothing.
fn capped(read: Option<u64>, s: &Stream) -> Option<u64> {
    read.map(|n| n.min(s.added()))
}

/// Run a trim, whichever kind it is. Answers how many entries went.
fn cut(s: &mut Stream, trim: Trim) -> u64 {
    match trim {
        Trim::None => 0,
        Trim::MaxLen { len, exact, limit } => s.trim_maxlen(len, exact, limit),
        Trim::MinId { id, exact, limit } => s.trim_minid(id, exact, limit),
    }
}

fn exhausted() -> Error {
    Error::new(Code::Invalid, EXHAUSTED)
}

fn refused(why: Refused) -> Error {
    match why {
        Refused::Zero => Error::new(Code::Invalid, ZERO_ID),
        Refused::NotGreater => Error::new(Code::Invalid, NOT_GREATER),
        Refused::Full => exhausted(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Keyspace {
        Keyspace::new()
    }

    /// One entry, with the two fields a reading has.
    fn add(d: &mut Keyspace, key: &[u8], id: Add) -> Id {
        d.xadd(
            key,
            id,
            &[(b"sensor", b"a4"), (b"reading", b"21.5")],
            Trim::None,
            true,
            1_000,
        )
        .expect("a stream")
        .expect("an ID")
    }

    /// Every ID in the stream, oldest first.
    fn ids(d: &mut Keyspace, key: &[u8]) -> Vec<Id> {
        let mut out = Vec::new();
        d.xrange_into(key, Id::MIN, Id::MAX, None, false, |id, _| {
            out.push(id);
            true
        })
        .expect("a stream");
        out
    }

    #[test]
    fn a_write_makes_the_key_and_a_read_finds_it() {
        let mut d = db();
        assert_eq!(d.kind_of(b"s"), None);
        let id = add(&mut d, b"s", Add::At(Id::new(5, 0)));
        assert_eq!(id, Id::new(5, 0));
        assert_eq!(d.kind_of(b"s"), Some(Kind::Stream));
        assert_eq!(d.type_name(b"s"), Some("stream"));
        assert_eq!(d.encoding_name(b"s"), Some("stream"));
        assert_eq!(ids(&mut d, b"s"), vec![Id::new(5, 0)]);
    }

    #[test]
    fn nomkstream_leaves_a_missing_key_missing() {
        let mut d = db();
        let got = d
            .xadd(b"s", Add::Auto, &[(b"f", b"v")], Trim::None, false, 1_000)
            .expect("a stream");
        assert_eq!(got, None);
        assert_eq!(d.kind_of(b"s"), None);
    }

    #[test]
    fn an_auto_id_follows_the_clock_and_then_the_last_id() {
        let mut d = db();
        assert_eq!(add(&mut d, b"s", Add::Auto), Id::new(1_000, 0));
        // The clock has not moved, so the sequence does.
        assert_eq!(add(&mut d, b"s", Add::Auto), Id::new(1_000, 1));
        assert_eq!(add(&mut d, b"s", Add::Seq(1_000)), Id::new(1_000, 2));
        assert_eq!(add(&mut d, b"s", Add::Seq(2_000)), Id::new(2_000, 0));
    }

    #[test]
    fn an_id_that_is_not_above_the_last_one_is_refused() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(5, 0)));
        let e = d
            .xadd(
                b"s",
                Add::At(Id::new(5, 0)),
                &[(b"f", b"v")],
                Trim::None,
                true,
                1_000,
            )
            .expect_err("not above the last one");
        assert_eq!(e.message(), NOT_GREATER);
        // And a sequence asked for inside a millisecond that has gone by.
        let e = d
            .xadd(b"s", Add::Seq(4), &[(b"f", b"v")], Trim::None, true, 1_000)
            .expect_err("a millisecond that has gone by");
        assert_eq!(e.message(), NOT_GREATER);
    }

    #[test]
    fn zero_is_refused_and_says_so_in_its_own_words() {
        let mut d = db();
        let e = d
            .xadd(
                b"s",
                Add::At(Id::MIN),
                &[(b"f", b"v")],
                Trim::None,
                true,
                1_000,
            )
            .expect_err("nothing sorts below zero");
        assert_eq!(e.message(), ZERO_ID);
    }

    #[test]
    fn a_stream_is_not_deleted_when_the_last_entry_goes() {
        let mut d = db();
        let id = add(&mut d, b"s", Add::At(Id::new(5, 0)));
        assert_eq!(d.xdel(b"s", [id].into_iter()).expect("a stream"), 1);
        assert_eq!(d.xdel(b"s", [id].into_iter()).expect("a stream"), 0);
        assert_eq!(d.kind_of(b"s"), Some(Kind::Stream), "the key is still here");
        // And the ID it handed out is still remembered, so the next one is above
        // it rather than the same one again.
        assert_eq!(add(&mut d, b"s", Add::Auto), Id::new(1_000, 0));
    }

    #[test]
    fn a_trim_runs_after_the_append() {
        let mut d = db();
        for ms in 1..=10u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        let trim = Trim::MaxLen {
            len: 1,
            exact: true,
            limit: None,
        };
        let id = d
            .xadd(
                b"s",
                Add::At(Id::new(11, 0)),
                &[(b"f", b"v")],
                trim,
                true,
                1,
            )
            .expect("a stream")
            .expect("an ID");
        // The entry that was just written is the one that survives, which is the
        // whole reason the order matters.
        assert_eq!(ids(&mut d, b"s"), vec![id]);
    }

    #[test]
    fn a_limit_stops_a_trim_at_the_next_node_boundary() {
        let mut d = db();
        for ms in 1..=1_000u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        let trim = Trim::MaxLen {
            len: 0,
            exact: false,
            limit: Some(10),
        };
        // A node is a hundred entries and a node is what goes, so asking to stop
        // after ten stops after the first node rather than in the middle of it.
        // That is what Redis does too, and it is why `LIMIT` is only allowed
        // with `~`: the limit is a brake on how long the command runs and not a
        // count of what it is allowed to remove.
        assert_eq!(d.xtrim(b"s", trim).expect("a stream"), 100);
        assert_eq!(
            d.stream(b"s").expect("a stream").expect("the key").len(),
            900
        );
    }

    #[test]
    fn setid_moves_the_bookmark_and_refuses_to_go_below_an_entry() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(5, 0)));
        let e = d
            .xsetid(b"s", Id::new(4, 0), None, None)
            .expect_err("below an entry that is still there");
        assert_eq!(e.message(), SETID_TOO_SMALL);
        d.xsetid(b"s", Id::new(9, 0), Some(41), None)
            .expect("above it");
        let s = d.stream(b"s").expect("a stream").expect("the key");
        assert_eq!((s.last_id(), s.added()), (Id::new(9, 0), 41));
    }

    #[test]
    fn setid_on_a_key_that_is_not_there_says_so() {
        let mut d = db();
        let e = d
            .xsetid(b"s", Id::new(1, 0), None, None)
            .expect_err("no key");
        assert_eq!((e.code(), e.message()), (Code::NotFound, NO_SUCH_KEY));
    }

    #[test]
    fn every_command_sees_the_wrong_type() {
        let mut d = db();
        d.set_plain(b"s", b"a string").expect("a string");
        for e in [
            d.xadd(b"s", Add::Auto, &[(b"f", b"v")], Trim::None, true, 1)
                .expect_err("a string"),
            d.xdel(b"s", [Id::new(1, 0)].into_iter())
                .expect_err("a string"),
            d.xtrim(b"s", Trim::None).expect_err("a string"),
            d.xrange_into(b"s", Id::MIN, Id::MAX, None, false, |_, _| true)
                .expect_err("a string"),
            d.xack(b"s", b"g", [Id::new(1, 0)].into_iter())
                .expect_err("a string"),
        ] {
            assert_eq!(e.code(), Code::WrongType);
        }
    }

    /// A new group has no read counter and still has a lag, because the two are
    /// worked out separately.
    ///
    /// Both lines are Redis 8.10.1's: `XINFO GROUPS` on a group made with no
    /// `ENTRIESREAD` reports the counter as null whichever position it was made
    /// at, and reports a lag of the whole stream at `0` and of zero at `$`.
    #[test]
    fn a_new_group_has_no_read_counter() {
        let mut d = db();
        for ms in 1..=5 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }

        assert!(
            d.xgroup_create(b"s", b"early", Start::At(Id::MIN), false, None)
                .expect("a stream")
        );
        assert!(
            d.xgroup_create(b"s", b"late", Start::Last, false, None)
                .expect("a stream")
        );
        let s = d.stream(b"s").expect("a stream").expect("the key");
        let early = s.group(b"early").expect("the early group");
        assert_eq!(early.entries_read(), None);
        assert_eq!(s.lag(early), Some(5), "everything is still in front of it");
        let late = s.group(b"late").expect("the late group");
        assert_eq!(late.entries_read(), None);
        assert_eq!(s.lag(late), Some(0), "and nothing is in front of this one");
    }

    /// A read counter a client hands in is held down to what was ever written,
    /// and giving none at all on a `SETID` gives the counter up.
    ///
    /// Both are Redis 8.10.1's. The first matters because the number is
    /// subtracted from the entry count to work the lag out, so believing a
    /// client that says ninety nine would report a lag of zero on a group that
    /// has read nothing.
    #[test]
    fn a_read_counter_that_is_too_big_is_brought_back_down() {
        let mut d = db();
        for ms in 1..=3 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }

        d.xgroup_create(b"s", b"g", Start::At(Id::MIN), false, Some(99))
            .expect("a stream");
        assert_eq!(
            counter(&mut d, b"g"),
            Some(3),
            "held down to what was added"
        );

        d.xgroup_setid(b"s", b"g", Start::At(Id::MIN), Some(2))
            .expect("a stream")
            .expect("the group");
        assert_eq!(counter(&mut d, b"g"), Some(2), "and left alone below that");

        d.xgroup_setid(b"s", b"g", Start::At(Id::MIN), None)
            .expect("a stream")
            .expect("the group");
        assert_eq!(
            counter(&mut d, b"g"),
            None,
            "and given up when none is named"
        );
    }

    /// The one group's read counter, which the test above reads three times.
    fn counter(d: &mut Keyspace, group: &[u8]) -> Option<u64> {
        d.stream(b"s")
            .expect("a stream")
            .expect("the key")
            .group(group)
            .expect("the group")
            .entries_read()
    }

    /// `XSETID` refuses a last ID below the deletion mark it was handed.
    ///
    /// Its own sentence and not the one about the top item, because it is its
    /// own mistake: the pair contradicts itself whatever the stream holds, and a
    /// stream whose last ID sat below its own deletion mark would hand an ID out
    /// twice.
    #[test]
    fn setid_refuses_a_last_id_under_its_own_deletion_mark() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(5, 0)));

        let e = d
            .xsetid(b"s", Id::new(9, 9), None, Some(Id::new(99, 99)))
            .expect_err("the pair contradicts itself");
        assert_eq!(e.message(), SETID_BELOW_MAX_DELETED);

        d.xsetid(b"s", Id::new(99, 99), None, Some(Id::new(9, 9)))
            .expect("the other way round is fine");
        let s = d.stream(b"s").expect("a stream").expect("the key");
        assert_eq!(s.last_id(), Id::new(99, 99));
        assert_eq!(s.max_deleted_id(), Id::new(9, 9));
    }

    /// A history read makes the consumer it was sent as, and answers nothing.
    ///
    /// The case is a worker that restarts under a new name and asks for its own
    /// backlog before it asks for new work. There is no backlog because the name
    /// is new, and that is an empty list rather than a missing group, which is
    /// the answer the wire has to be able to tell apart from `NOGROUP`.
    #[test]
    fn a_history_read_by_a_name_nobody_has_used_makes_the_consumer() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(1, 0)));
        d.xgroup_create(b"s", b"g", Start::At(Id::MIN), false, None)
            .expect("a stream");

        let want = Read {
            group: b"g",
            consumer: b"newbie",
            from: From::Pending(Id::MIN),
            count: None,
            noack: false,
        };
        let seen = d
            .xreadgroup_into(b"s", want, 500, |_, _| true)
            .expect("a stream")
            .expect("the group is there");
        assert_eq!(seen, 0, "nothing was ever handed to this name");

        let s = d.stream(b"s").expect("a stream").expect("the key");
        let c = s
            .group(b"g")
            .expect("the group")
            .consumer_named(b"newbie")
            .expect("the read made it");
        assert_eq!(c.seen(), 500, "it was heard from");
        assert_eq!(c.active(), None, "and it has never had anything");
    }

    #[test]
    fn a_group_reads_what_arrives_after_it_was_made() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(1, 0)));
        assert!(
            d.xgroup_create(b"s", b"workers", Start::Last, false, None)
                .expect("a stream")
        );
        // Making it again is not an error here, it is a false, and the wire
        // turns that into BUSYGROUP.
        assert!(
            !d.xgroup_create(b"s", b"workers", Start::Last, false, None)
                .expect("a stream")
        );
        add(&mut d, b"s", Add::At(Id::new(2, 0)));

        let mut got = Vec::new();
        let seen = d
            .xreadgroup_into(
                b"s",
                Read {
                    group: b"workers",
                    consumer: b"alice",
                    from: From::New,
                    count: None,
                    noack: false,
                },
                1_000,
                |id, fields| {
                    got.push((id, fields.is_some()));
                    true
                },
            )
            .expect("a stream")
            .expect("a group");
        assert_eq!(seen, 1, "only what arrived after the group was made");
        assert_eq!(got, vec![(Id::new(2, 0), true)]);
        assert_eq!(
            d.xack(b"s", b"workers", [Id::new(2, 0)].into_iter())
                .expect("a stream"),
            1
        );
    }

    #[test]
    fn noack_hands_the_entry_over_without_writing_it_down() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(1, 0)));
        d.xgroup_create(b"s", b"workers", Start::At(Id::MIN), false, None)
            .expect("a stream");
        let seen = d
            .xreadgroup_into(
                b"s",
                Read {
                    group: b"workers",
                    consumer: b"alice",
                    from: From::New,
                    count: None,
                    noack: true,
                },
                1_000,
                |_, _| true,
            )
            .expect("a stream")
            .expect("a group");
        assert_eq!(seen, 1);
        let s = d.stream(b"s").expect("a stream").expect("the key");
        let g = s.group(b"workers").expect("the group");
        assert_eq!(g.pending_len(), 0, "nothing was written down");
        assert_eq!(g.last_id(), Id::new(1, 0), "the bookmark still moved");
        assert_eq!(s.lag(g), Some(0), "and the group has caught up");
    }

    #[test]
    fn a_missing_group_is_a_none_and_not_an_error() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(1, 0)));
        let got = d
            .xreadgroup_into(
                b"s",
                Read {
                    group: b"nope",
                    consumer: b"alice",
                    from: From::New,
                    count: None,
                    noack: false,
                },
                1,
                |_, _| true,
            )
            .expect("a stream");
        assert_eq!(got, None);
        assert!(!d.xgroup_destroy(b"s", b"nope").expect("a stream"));
        assert_eq!(
            d.xgroup_create_consumer(b"s", b"nope", b"alice", 1)
                .expect("a stream"),
            None
        );
    }

    #[test]
    fn xgroup_needs_the_key_and_says_which_option_makes_one() {
        let mut d = db();
        let e = d
            .xgroup_create(b"s", b"workers", Start::Last, false, None)
            .expect_err("no key");
        assert_eq!((e.code(), e.message()), (Code::NotFound, NO_KEY_FOR_GROUP));
        assert!(
            d.xgroup_create(b"s", b"workers", Start::Last, true, None)
                .expect("mkstream made one")
        );
        assert_eq!(d.kind_of(b"s"), Some(Kind::Stream));
    }

    #[test]
    fn a_claim_moves_an_idle_entry_and_drops_one_that_has_gone() {
        let mut d = db();
        for ms in 1..=2u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        d.xgroup_create(b"s", b"workers", Start::At(Id::MIN), false, None)
            .expect("a stream");
        d.xreadgroup_into(
            b"s",
            Read {
                group: b"workers",
                consumer: b"alice",
                from: From::New,
                count: None,
                noack: false,
            },
            100,
            |_, _| true,
        )
        .expect("a stream")
        .expect("a group");
        d.xdel(b"s", [Id::new(1, 0)].into_iter()).expect("a stream");

        let mut gone = Vec::new();
        let how = Claim {
            group: b"workers",
            consumer: b"bob",
            min_idle: 500,
            time: 1_000,
            ..Claim::default()
        };
        let took = d
            .xclaim(b"s", &[Id::new(1, 0), Id::new(2, 0)], how, 1_000, &mut gone)
            .expect("a stream")
            .expect("a group");
        assert_eq!(took, vec![Id::new(2, 0)]);
        assert_eq!(gone, vec![Id::new(1, 0)], "no one can ever finish that one");
    }

    #[test]
    fn an_autoclaim_sweeps_from_a_cursor() {
        let mut d = db();
        for ms in 1..=10u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        d.xgroup_create(b"s", b"workers", Start::At(Id::MIN), false, None)
            .expect("a stream");
        d.xreadgroup_into(
            b"s",
            Read {
                group: b"workers",
                consumer: b"alice",
                from: From::New,
                count: None,
                noack: false,
            },
            100,
            |_, _| true,
        )
        .expect("a stream")
        .expect("a group");

        let mut gone = Vec::new();
        let how = Claim {
            group: b"workers",
            consumer: b"bob",
            min_idle: 500,
            time: 1_000,
            ..Claim::default()
        };
        let (cursor, took) = d
            .xautoclaim(b"s", Id::MIN, how, 4, 1_000, &mut gone)
            .expect("a stream")
            .expect("a group");
        assert_eq!(took.len(), 4);
        assert_eq!(cursor, Some(Id::new(5, 0)), "where the next call starts");
        assert!(gone.is_empty());
    }

    #[test]
    fn a_pending_window_carries_the_owner_and_the_counts() {
        let mut d = db();
        for ms in 1..=3u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        d.xgroup_create(b"s", b"workers", Start::At(Id::MIN), false, None)
            .expect("a stream");
        d.xreadgroup_into(
            b"s",
            Read {
                group: b"workers",
                consumer: b"alice",
                from: From::New,
                count: None,
                noack: false,
            },
            100,
            |_, _| true,
        )
        .expect("a stream")
        .expect("a group");

        let mut out = Vec::new();
        let seen = d
            .xpending_into(b"s", b"workers", Filter::default(), 600, |id, nack, c| {
                out.push((
                    id,
                    nack.count(),
                    nack.idle(600),
                    c.expect("an owner").name().to_vec(),
                ));
                true
            })
            .expect("a stream")
            .expect("a group");
        assert_eq!(seen, 3);
        assert_eq!(out[0], (Id::new(1, 0), 1, 500, b"alice".to_vec()));
    }

    #[test]
    fn a_history_read_hands_back_a_null_for_an_entry_that_has_gone() {
        let mut d = db();
        for ms in 1..=2u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        d.xgroup_create(b"s", b"workers", Start::At(Id::MIN), false, None)
            .expect("a stream");
        d.xreadgroup_into(
            b"s",
            Read {
                group: b"workers",
                consumer: b"alice",
                from: From::New,
                count: None,
                noack: false,
            },
            100,
            |_, _| true,
        )
        .expect("a stream")
        .expect("a group");
        d.xdel(b"s", [Id::new(1, 0)].into_iter()).expect("a stream");

        let mut out = Vec::new();
        d.xreadgroup_into(
            b"s",
            Read {
                group: b"workers",
                consumer: b"alice",
                from: From::Pending(Id::MIN),
                count: None,
                noack: false,
            },
            2_000,
            |id, fields| {
                out.push((id, fields.is_some()));
                true
            },
        )
        .expect("a stream")
        .expect("a group");
        assert_eq!(out, vec![(Id::new(1, 0), false), (Id::new(2, 0), true)]);
    }

    #[test]
    fn a_stream_counts_against_the_memory_total() {
        let mut d = db();
        let before = d.memory_bytes();
        for ms in 1..=1_000u64 {
            add(&mut d, b"s", Add::At(Id::new(ms, 0)));
        }
        let after = d.memory_bytes();
        assert!(after > before + 1_000, "{before} then {after}");
        assert!(d.del(b"s"), "the key was there");
        assert_eq!(d.kind_of(b"s"), None);
    }

    #[test]
    fn a_deadline_goes_on_a_stream_the_same_as_on_anything_else() {
        let mut d = db();
        add(&mut d, b"s", Add::At(Id::new(1, 0)));
        assert!(d.set_expiry(b"s", Some(1 << 45)));
        assert_eq!(d.kind_of(b"s"), Some(Kind::Stream));
        assert_eq!(ids(&mut d, b"s"), vec![Id::new(1, 0)], "the body is intact");
    }
}
