//! Keyspace notifications: `notify-keyspace-events` and the two channels it
//! publishes on.
//!
//! A client that wants to know when a key changed subscribes to
//! `__keyspace@0__:mykey` and gets the name of whatever happened to it, or to
//! `__keyevent@0__:del` and gets the name of every key that was deleted. Both
//! are ordinary pub/sub channels, so everything in [`super::pubsub`] carries
//! them and there is nothing new on the delivery side.
//!
//! # What it costs a server that has it turned off
//!
//! Nothing that can be measured, which matters because the call sites are
//! spread through every command that writes. The setting is off by default and
//! most servers leave it off, so the check in front of a notification has to be
//! cheaper than the work of deciding what to say.
//!
//! [`fire`] reads one thread local word and tests a bit in it. It does not take
//! the server, does not take a lock and does not look at the key. The word is
//! zero unless notifications are on for some class and somebody is actually
//! subscribed to something, so a server with the setting on and no subscribers
//! is as cheap as one with the setting off.
//!
//! # Why a thread local
//!
//! Because the call sites do not have the server. A command body is handed a
//! database and its arguments, which is the whole point of the shape: a hash
//! command cannot reach the pub/sub registry, cannot take the pubsub lock and
//! should not learn how. Threading a notifier down through every group's
//! `execute` and every helper under it would put a parameter that is almost
//! always unused on several hundred functions.
//!
//! So the funnel arms this thread before the command runs and drains it after.
//! In between, a command body says what it did with a free function and the
//! funnel turns that into publishes at a point where it does have the server.
//! Nothing outside this module can see the buffer.
//!
//! # What that changes about the order
//!
//! Redis publishes a notification at the moment the command makes the change.
//! This one publishes them all when the command finishes, in the order they
//! were said. Within one command that is the same thing, because a command's
//! notifications are the only messages it produces and their order is kept.
//!
//! `EXEC` and a script re-enter the funnel for each command they run, so each
//! of those arms and drains around its own command and the notifications come
//! out between them rather than all at the end. That is Redis's order too.
//!
//! # Classes
//!
//! Every notification belongs to a class, and a class is a character in the
//! `notify-keyspace-events` setting. `K` and `E` are not classes but the two
//! channels: a setting with classes and none of the channel characters publishes
//! nothing, which is why [`arm`] treats it as off.
//!
//! # The subkey channels
//!
//! `S`, `T`, `I` and `V` are four more channels rather than four more classes,
//! and they carry the fields an event touched alongside the key it touched. Only
//! the hash class has fields today, so only the hash class fills them in, and an
//! event with no fields behind it is published on the two ordinary channels and
//! nowhere else however those four are set.
//!
//! They are outside `A` because they are extra copies of events that are already
//! in it, so a client that wants both the key level and the field level view has
//! to ask for both.

use std::cell::{Cell, RefCell};

use yo_kv::Db;
use yo_kv::news;

use super::Server;
use super::pubsub::{self, Kind};
use super::repl;

/// The class bits, which are Redis's own bit numbers.
///
/// The numbering is copied rather than compacted because it is the numbering
/// `CONFIG GET notify-keyspace-events` has to reproduce the order of, and
/// because the gaps are real: bit 12 is a module only class for a key loaded
/// from an RDB and bit 17 is another for a key trimmed during slot migration,
/// neither of which a client can ask for.
pub(crate) mod class {
    /// `K`, the `__keyspace@<db>__:<key>` channel.
    pub(crate) const KEYSPACE: u32 = 1 << 0;
    /// `E`, the `__keyevent@<db>__:<event>` channel.
    pub(crate) const KEYEVENT: u32 = 1 << 1;
    /// `g`, the commands that work on a key whatever is under it.
    pub(crate) const GENERIC: u32 = 1 << 2;
    /// `$`, the string commands.
    pub(crate) const STRING: u32 = 1 << 3;
    /// `l`, the list commands.
    pub(crate) const LIST: u32 = 1 << 4;
    /// `s`, the set commands.
    pub(crate) const SET: u32 = 1 << 5;
    /// `h`, the hash commands.
    pub(crate) const HASH: u32 = 1 << 6;
    /// `z`, the sorted set commands.
    pub(crate) const ZSET: u32 = 1 << 7;
    /// `x`, a key that reached its deadline.
    pub(crate) const EXPIRED: u32 = 1 << 8;
    /// `e`, a key thrown away to make room.
    pub(crate) const EVICTED: u32 = 1 << 9;
    /// `t`, the stream commands.
    pub(crate) const STREAM: u32 = 1 << 10;
    /// `m`, a read that found nothing. Left out of `A` on purpose.
    pub(crate) const KEY_MISS: u32 = 1 << 11;
    /// `d`, a module type's own events.
    pub(crate) const MODULE: u32 = 1 << 13;
    /// `n`, a key that was not there before. Left out of `A` on purpose.
    pub(crate) const NEW: u32 = 1 << 14;
    /// `o`, a key written over. Left out of `A` on purpose.
    pub(crate) const OVERWRITTEN: u32 = 1 << 15;
    /// `c`, a key that changed type. Left out of `A` on purpose.
    pub(crate) const TYPE_CHANGED: u32 = 1 << 16;
    /// `S`, the subkey level keyspace channel.
    pub(crate) const SUBKEYSPACE: u32 = 1 << 19;
    /// `T`, the subkey level keyevent channel.
    pub(crate) const SUBKEYEVENT: u32 = 1 << 20;
    /// `I`, one channel per key and subkey.
    pub(crate) const SUBKEYSPACEITEM: u32 = 1 << 21;
    /// `V`, one channel per event and key.
    pub(crate) const SUBKEYSPACEEVENT: u32 = 1 << 22;
    /// `a`, the array commands.
    pub(crate) const ARRAY: u32 = 1 << 23;

    /// What `A` stands for.
    ///
    /// Not everything: `m`, `n`, `o` and `c` are outside it because each of them
    /// fires on far more than a write does, and the four subkey channels are
    /// outside it because they are extra copies of events already in it.
    pub(crate) const ALL: u32 =
        GENERIC | STRING | LIST | SET | HASH | ZSET | EXPIRED | EVICTED | STREAM | MODULE | ARRAY;
}

/// The four subkey channels, which are turned on and off on their own.
///
/// A setting of `hS` and nothing else publishes on `__subkeyspace@0__:` and on
/// neither of the two ordinary channels, so these count towards a server having
/// something to say just as `K` and `E` do.
const SUBKEY: u32 =
    class::SUBKEYSPACE | class::SUBKEYEVENT | class::SUBKEYSPACEITEM | class::SUBKEYSPACEEVENT;

/// Every channel, so that a setting with classes and none of them can be
/// recognised as publishing nothing.
const CHANNELS: u32 = class::KEYSPACE | class::KEYEVENT | SUBKEY;

/// Every class character, in the order `CONFIG GET` writes them.
///
/// Two lists rather than one, because the reply puts `A` in place of the first
/// group when every bit in it is set and then writes the second group after it
/// either way. That is `keyspaceEventsFlagsToString` and the order is its order.
const FIRST: &[(u8, u32)] = &[
    (b'g', class::GENERIC),
    (b'$', class::STRING),
    (b'l', class::LIST),
    (b's', class::SET),
    (b'h', class::HASH),
    (b'z', class::ZSET),
    (b'x', class::EXPIRED),
    (b'e', class::EVICTED),
    (b't', class::STREAM),
    (b'd', class::MODULE),
    (b'a', class::ARRAY),
    (b'n', class::NEW),
    (b'o', class::OVERWRITTEN),
    (b'c', class::TYPE_CHANGED),
];

/// The classes written after `A` or after the first group, whichever ran.
const SECOND: &[(u8, u32)] = &[
    (b'K', class::KEYSPACE),
    (b'E', class::KEYEVENT),
    (b'm', class::KEY_MISS),
    (b'S', class::SUBKEYSPACE),
    (b'T', class::SUBKEYEVENT),
    (b'I', class::SUBKEYSPACEITEM),
    (b'V', class::SUBKEYSPACEEVENT),
];

/// The characters `CONFIG SET` takes, in the order the error message lists them.
///
/// Quoted back at a client that got one wrong, so the order is the reference's
/// order and not a tidied one.
pub(crate) const ACCEPTED: &str = "Ag$lshzxeKEtmdnocaSTIV";

/// The longest a formatted setting can be: every character in both groups.
const WIDEST: usize = FIRST.len() + SECOND.len();

/// Read a `notify-keyspace-events` setting.
///
/// `None` for a character that is not a class, which is the whole of the
/// validation: there is no ordering rule, a character may be repeated and the
/// empty string is a valid setting meaning off.
pub(crate) fn parse(classes: &[u8]) -> Option<u32> {
    let mut flags = 0;
    for &c in classes {
        flags |= match c {
            b'A' => class::ALL,
            _ => {
                let found = FIRST
                    .iter()
                    .chain(SECOND)
                    .find(|(ch, _)| *ch == c)
                    .map(|(_, bit)| *bit);
                found?
            }
        };
    }
    Some(flags)
}

/// Write a setting back the way `CONFIG GET` writes it.
///
/// Not the string that was set: `CONFIG SET notify-keyspace-events KEA` reads
/// back as `AKE`, because the flags are what is kept and this is the one way of
/// spelling them. A client that compares what it set against what it reads is
/// going to be surprised, and it is going to be surprised by a real server in
/// the same way.
pub(crate) fn format(flags: u32) -> ([u8; WIDEST], usize) {
    let mut out = [0u8; WIDEST];
    let mut len = 0;
    // The first group is either every character in it or the one letter that
    // stands for all of them, and `A` wins whenever every bit is set. That is
    // why `An` reads back as `A`: `n` is in this group and `A` speaks for it.
    if flags & class::ALL == class::ALL {
        out[len] = b'A';
        len += 1;
    } else {
        for (c, bit) in FIRST {
            if flags & bit != 0 {
                out[len] = *c;
                len += 1;
            }
        }
    }
    // The second group is written either way, which is what makes `Am` and `AS`
    // round trip where `An` does not.
    for (c, bit) in SECOND {
        if flags & bit != 0 {
            out[len] = *c;
            len += 1;
        }
    }
    (out, len)
}

/// One thing that happened, waiting to be published.
struct Event {
    /// Which database the key is on, which is not always the one the connection
    /// selected: `COPY k d DB 1` says `copy_to` on database one.
    db: usize,
    /// The event name, which is the payload on the keyspace channel and part of
    /// the channel name on the keyevent one.
    name: &'static str,
    /// The key it happened to.
    key: Vec<u8>,
    /// The fields of that key it happened to, for the four subkey channels.
    ///
    /// Empty for every event outside the hash class, and empty inside it when
    /// none of those four channels is on, which is why an empty vector has to
    /// stay free: it is the case on nearly every notification a server sends.
    subs: Vec<Vec<u8>>,
}

thread_local! {
    /// What the command running on this thread has said so far.
    ///
    /// Empty on a server with notifications off, because [`fire`] never gets
    /// past its first branch, and the vector never allocates.
    static PENDING: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };

    /// The classes worth saying anything about, or zero for none of them.
    ///
    /// Read by every [`fire`] and written twice per command, which is why it is
    /// a whole word rather than a flag and a separate mask: one load and one
    /// test.
    static ARMED: Cell<u32> = const { Cell::new(0) };

    /// Which database the keys [`heard`] is told about belong to.
    ///
    /// Every other event names its database at the call site, because the call
    /// site is a command and a command knows which one it is running against.
    /// These have no call site in that sense: a key goes under a lookup or under
    /// the expire cycle and a key arrives under whatever wrote it, and the
    /// storage layer that notices either has never heard of a database number.
    /// So the funnel leaves the answer here on its way in.
    static WHERE: Cell<usize> = const { Cell::new(0) };
}

/// What a thread was set up for, so that it can be put back that way.
///
/// `EXEC` and a script run commands through the funnel while a command is
/// already running, so the inner one has to leave the outer one as it found it.
/// Both halves matter: `SELECT` is allowed inside `MULTI`, so the database the
/// storage layer's news belongs to is not the same for the whole of an `EXEC`.
#[derive(Clone, Copy)]
pub(super) struct Armed {
    /// The classes that were worth saying anything about.
    flags: u32,
    /// The database the storage layer's news was being attributed to.
    db: usize,
}

/// Get this thread ready for one command against `db`, and answer what it was
/// ready for before.
///
/// The flags are zero when nothing would come of a notification: notifications
/// off, or on but with none of the six channels selected, or on with all of them
/// but with nobody subscribed to anything at all. The last of those is the one
/// that matters in practice,
/// since it is what a server with the setting in its config file and no clients
/// listening looks like.
pub(super) fn arm(server: &Server, db: usize) -> Armed {
    let flags = server.notify_flags();
    let live = flags & CHANNELS != 0 && server.anyone_subscribed();
    // A server with a replica wants to hear the same news for a different
    // reason: a key that goes on its own has to be sent on as a deletion, since
    // a replica never takes one away by itself. So the hook goes in for either,
    // and `heard` sorts out which of the two is asking.
    let copying = repl::arm(server.replicated());
    // Installed for a command that has somewhere to send what it hears and not
    // for one that does not, which is the storage layer's only way of knowing
    // whether the questions it would have to ask to say anything are worth
    // asking. Nothing puts back what was there before, and nothing has to: this
    // runs in front of every command, so the answer is never stale by the time
    // it is read.
    news::tell((live || copying).then_some(heard as news::Told));
    Armed {
        flags: ARMED.replace(if live { flags } else { 0 }),
        db: WHERE.replace(db),
    }
}

/// Attribute what the storage layer says next to `db` rather than to the one the
/// connection is on, and answer where it was being attributed before.
///
/// For the two commands that put a key in a database other than the one that
/// asked for it. A key arriving is noticed underneath, where there is no
/// database number to be had, so [`arm`] leaves the connection's number here on
/// the way in and that is the right answer for every command but these two.
/// `MOVE k 1` and `COPY a b DB 1` both make a key on database one, and it is a
/// client watching database one that hears about it.
///
/// The caller puts back what it was told, around the one call that writes on the
/// far side and no wider, so that anything the same command says about a key on
/// its own database is still attributed there.
pub(crate) fn about(db: usize) -> usize {
    WHERE.replace(db)
}

/// Say what happened to a key that no command's reply covers.
///
/// Installed by [`arm`] and called by the storage layer at the moment it
/// happens, which is before the command that provoked it has said what it did.
/// That is the order a real server publishes in and it comes out of where this
/// is called from rather than out of anything here.
fn heard(key: &[u8], what: news::What, field: &[u8]) {
    // None of them brings a companion event along with it. A key that reached
    // its deadline and a key a client deleted are two different pieces of news,
    // and so are a key that was created and the write that created it, and a
    // subscriber that wanted both asked for both.
    let (class, name) = match what {
        news::What::Born => (class::NEW, "new"),
        news::What::Overwritten => (class::OVERWRITTEN, "overwritten"),
        news::What::TypeChanged => (class::TYPE_CHANGED, "type_changed"),
        news::What::Expired => {
            repl::reaped(&[b"DEL", key]);
            (class::EXPIRED, "expired")
        }
        news::What::FieldExpired => {
            repl::reaped(&[b"HDEL", key, field]);
            return field_expired(key, field);
        }
        news::What::Deleted => (class::GENERIC, "del"),
        news::What::Evicted => {
            repl::reaped(&[b"DEL", key]);
            (class::EVICTED, "evicted")
        }
        news::What::Missed => (class::KEY_MISS, MISS),
    };
    fire(WHERE.get(), class, name, key);
}

/// What a hash field reaching its own deadline is called.
///
/// Named here for the reason [`MISS`] is: the batching below has to recognise
/// the event it is adding to, and a name two places agree on is a constant.
const HEXPIRED: &str = "hexpired";

/// Add one field to the `hexpired` this key is already saying, or start one.
///
/// A reap goes field by field and a real server sends one event carrying the
/// whole list, so the list is put back together here. That is what the storage
/// layer would otherwise have to build itself, and it would have to build it out
/// of names it is in the middle of deleting.
///
/// Looking at the last event queued is enough to find the one to add to. A reap
/// is one key at a time and says nothing else while it runs, so an `hexpired`
/// for this key anywhere further back belongs to an earlier reap and is a
/// separate event, which is what a real server sends for it too.
fn field_expired(key: &[u8], field: &[u8]) {
    if ARMED.get() & class::HASH == 0 {
        return;
    }
    let db = WHERE.get();
    let wanted = subkeys_wanted(class::HASH);
    PENDING.with_borrow_mut(|pending| {
        yo_alloc::allow(|| match pending.last_mut() {
            Some(last) if last.name == HEXPIRED && last.db == db && last.key == key => {
                if wanted {
                    last.subs.push(field.to_vec());
                }
            }
            _ => pending.push(Event {
                db,
                name: HEXPIRED,
                key: key.to_vec(),
                subs: if wanted {
                    vec![field.to_vec()]
                } else {
                    Vec::new()
                },
            }),
        });
    });
}

/// What a read that found nothing is called.
///
/// Named here rather than spelled at each call site because [`unsay`] has to
/// match on it, and a name that two places have to agree on is a constant.
pub(crate) const MISS: &str = "keymiss";

/// Take back every `keymiss` this command has queued so far.
///
/// For a command that turned out never to have looked anything up. Redis fires
/// a miss inside the lookup, so a command that fails while it is still reading
/// its own arguments fires nothing, and one that fails on what it found has
/// already fired. [`misses`](super::misses) asks in front of the command
/// instead, which cannot tell those two apart, so the caller undoes the first
/// case afterwards once the error says which it was.
///
/// Only the misses go. The existence probe that found one can itself have
/// reaped a key on the way past, and that `expired` happened whatever the
/// command went on to do with its arguments.
pub(crate) fn unsay_misses() {
    if ARMED.get() & class::KEY_MISS == 0 {
        return;
    }
    PENDING.with_borrow_mut(|pending| pending.retain(|event| event.name != MISS));
}

/// Whether anything this command says will go anywhere.
///
/// For the handful of call sites that have to do extra work to find out what to
/// report, rather than just reporting something they already know. `SORT ...
/// STORE` is the one: an empty sort deletes the destination and says `del`, and
/// finding out whether there was a destination to delete costs a stripe lock,
/// so it is worth not paying for it on a server nobody is subscribed to.
pub(crate) fn armed() -> bool {
    ARMED.get() != 0
}

/// Whether an event in this class would reach anybody.
///
/// A tighter question than [`armed`], for a call site that has to build a list
/// of field names before it can say anything at all. The hash commands are all
/// of them: what a `HEXPIRE` says depends on which fields took the deadline and
/// which were deleted by it, and neither list is the reply.
pub(crate) fn wanted(class: u32) -> bool {
    ARMED.get() & class != 0
}

/// Whether the field names alongside an event in this class would reach anybody.
///
/// The names cost more to collect than the event itself and they are only ever
/// read by the four subkey channels, so a call site that would have to copy them
/// out asks this first. This is Redis's `isSubkeyNotifyEnabled` and it is asked
/// in the same places.
pub(crate) fn subkeys_wanted(class: u32) -> bool {
    let armed = ARMED.get();
    armed & class != 0 && armed & SUBKEY != 0
}

/// The `del` that follows a removal which took the last of a collection.
///
/// A list, set, hash or sorted set with nothing in it is not a key, so a pop, a
/// trim or a removal that took everything leaves the key gone, and Redis says so
/// on the generic class straight after saying what it did. Whether the key is
/// still there costs a stripe lock to ask, so it is asked only when somebody is
/// listening for the answer.
pub(crate) fn emptied(db: &Db, on: usize, key: &[u8]) {
    if armed() && !db.hold(key).exists(key) {
        fire(on, class::GENERIC, "del", key);
    }
}

/// The pair that says a key that was already taken has something else under it
/// now.
///
/// Everywhere else this comes out of the storage layer at the moment the old
/// value goes, which puts it in front of whatever the command says it did. The
/// four that move a key rather than change one are the exception and say it
/// afterwards: `RENAME`, `RENAMENX`, `COPY` and `RESTORE` take the destination
/// away and put a key in its place, so there is nothing under the name by the
/// time the write happens and nothing for the storage layer to notice. Redis
/// says it afterwards for them too, from its own separate call, so the order
/// here is the order on the wire and not an accident of where this sits.
///
/// `was` is what the destination held before, or `None` for a name that was
/// free, which is the case where neither event happens.
pub(crate) fn replaced(db: usize, key: &[u8], was: Option<yo_kv::Kind>, now: Option<yo_kv::Kind>) {
    let Some(was) = was else {
        return;
    };
    fire(db, class::OVERWRITTEN, "overwritten", key);
    if Some(was) != now {
        fire(db, class::TYPE_CHANGED, "type_changed", key);
    }
}

/// What a key holds, for the caller of [`replaced`] to ask before and after,
/// and only when the answer would go anywhere.
///
/// It is a probe of the map either way, so a server with nobody listening for
/// these two does not pay for it, and one that is listening pays it twice on a
/// rename and not at all on anything else.
pub(crate) fn kind_now(look: impl FnOnce() -> Option<yo_kv::Kind>) -> Option<yo_kv::Kind> {
    if wanted(class::OVERWRITTEN | class::TYPE_CHANGED) {
        look()
    } else {
        None
    }
}

/// Say that something happened to a key.
///
/// `class` is the one bit this event belongs to and `name` is what Redis calls
/// it, both of which are constants at every call site, so a command that is not
/// in a class anybody asked for costs a load, an and and a branch.
pub(crate) fn fire(db: usize, class: u32, name: &'static str, key: &[u8]) {
    if ARMED.get() & class == 0 {
        return;
    }
    keep(db, name, key, Vec::new());
}

/// The field names an event is going to carry.
///
/// A hash command finds out which fields it really touched while it is touching
/// them, and its reply does not say: `HDEL k a b` answering one has not said
/// whether it was `a` or `b` that went. So the call sites collect the names as
/// they go, through this, which copies nothing and allocates nothing unless one
/// of the four subkey channels is on. On a server where they are off, which is
/// every server that has not asked for them, a `push` is a load and a branch.
pub(crate) struct Subkeys {
    /// The names so far, empty when nobody asked for them.
    names: Vec<Vec<u8>>,
    /// Whether to bother, read once when this was made.
    wanted: bool,
}

impl Subkeys {
    /// A collector for one class of event.
    pub(crate) fn new(class: u32) -> Self {
        Self {
            names: Vec::new(),
            wanted: subkeys_wanted(class),
        }
    }

    /// A collector already holding these names.
    ///
    /// For the commands that know what they touched from their arguments alone.
    /// `HSET` is the pattern: every field it names is a field it wrote, so there
    /// is nothing to find out as it goes.
    pub(crate) fn of<'a>(class: u32, names: impl Iterator<Item = &'a [u8]>) -> Self {
        let mut subs = Self::new(class);
        for name in names {
            subs.push(name);
        }
        subs
    }

    /// Add a field name the event is about.
    pub(crate) fn push(&mut self, name: &[u8]) {
        if self.wanted {
            yo_alloc::allow(|| self.names.push(name.to_vec()));
        }
    }
}

/// Say that something happened to some fields of a key.
///
/// The same event as [`fire`] on the two ordinary channels, and four more
/// publishes on top of it on the channels that are on and have something to
/// carry. Whether the event fires at all is the caller's question and not this
/// one's: an empty collector is a server with the subkey channels off just as
/// much as it is a command that touched no fields, and the caller knows which.
pub(crate) fn fire_subkeys(db: usize, class: u32, name: &'static str, key: &[u8], subs: Subkeys) {
    if ARMED.get() & class == 0 {
        return;
    }
    keep(db, name, key, subs.names);
}

/// Copy the key out and remember it.
///
/// Out of line and cold, so that the branch in [`fire`] is the whole of what a
/// call costs when nobody is listening. The copy is unavoidable: the key
/// borrows the connection's read buffer and the publish happens after the
/// command body has given that borrow back.
#[cold]
#[inline(never)]
fn keep(db: usize, name: &'static str, key: &[u8], subs: Vec<Vec<u8>>) {
    let event = yo_alloc::allow(|| Event {
        db,
        name,
        key: key.to_vec(),
        subs,
    });
    PENDING.with_borrow_mut(|pending| yo_alloc::allow(|| pending.push(event)));
}

/// Publish everything the command said, and put the thread back the way [`arm`]
/// found it.
pub(super) fn drain(server: &Server, was: Armed) {
    WHERE.set(was.db);
    let flags = ARMED.replace(was.flags);
    if flags == 0 {
        return;
    }
    // Taken out of the thread local rather than published from inside it,
    // because a publish takes the pub/sub lock and holding a `RefCell` across a
    // lock is a rule worth not having to think about.
    let events = PENDING.with_borrow_mut(std::mem::take);
    if events.is_empty() {
        return;
    }
    for event in &events {
        send(server, flags, event);
    }
    // The vector goes back so that a busy thread allocates it once. Anything
    // the command pushed while this was publishing, which nothing does today,
    // is kept rather than dropped.
    PENDING.with_borrow_mut(|pending| {
        if pending.is_empty() {
            let mut events = events;
            events.clear();
            *pending = events;
        }
    });
}

/// Publish one event on whichever of the two channels is turned on.
fn send(server: &Server, flags: u32, event: &Event) {
    let mut channel = [0u8; CHANNEL_MAX];
    if flags & class::KEYSPACE != 0 {
        let head = prefix(&mut channel, b"__keyspace@", event.db);
        yo_alloc::allow(|| {
            let mut name = channel[..head].to_vec();
            name.extend_from_slice(&event.key);
            pubsub::deliver(server, Kind::Channel, &name, event.name.as_bytes());
        });
    }
    if flags & class::KEYEVENT != 0 {
        let head = prefix(&mut channel, b"__keyevent@", event.db);
        yo_alloc::allow(|| {
            let mut name = channel[..head].to_vec();
            name.extend_from_slice(event.name.as_bytes());
            pubsub::deliver(server, Kind::Channel, &name, &event.key);
        });
    }
    // Only for an event that named fields, which is the hash class and nothing
    // else so far. The four are independent of the two above, so a setting of
    // `hS` publishes here and nowhere else.
    if flags & SUBKEY != 0 && !event.subs.is_empty() {
        yo_alloc::allow(|| subkeys(server, flags, event));
    }
}

/// Publish one event on whichever of the four subkey channels is turned on.
///
/// Two of them put the event name in front of something else with a `|` between,
/// so an event whose own name held a `|` could not be read back apart and those
/// two are skipped for it. One of them puts the field after the key with a
/// newline between and is skipped for a key holding a newline for the same
/// reason. Both rules are Redis's, and no event fired today trips the first one.
fn subkeys(server: &Server, flags: u32, event: &Event) {
    let mut channel = [0u8; CHANNEL_MAX];
    let bar = event.name.contains('|');
    // `__subkeyspace@<db>__:<key>` carrying `<event>|<fields>`.
    if flags & class::SUBKEYSPACE != 0 && !bar {
        let head = prefix(&mut channel, b"__subkeyspace@", event.db);
        let mut name = channel[..head].to_vec();
        name.extend_from_slice(&event.key);
        let mut payload = event.name.as_bytes().to_vec();
        payload.push(b'|');
        cat(&mut payload, &event.subs);
        pubsub::deliver(server, Kind::Channel, &name, &payload);
    }
    // `__subkeyevent@<db>__:<event>` carrying `<keylen>:<key>|<fields>`. The key
    // is length prefixed here because it is followed by the fields and a client
    // splitting on the `|` alone would cut a key that holds one in half.
    if flags & class::SUBKEYEVENT != 0 {
        let head = prefix(&mut channel, b"__subkeyevent@", event.db);
        let mut name = channel[..head].to_vec();
        name.extend_from_slice(event.name.as_bytes());
        let mut payload = Vec::new();
        len_prefixed(&mut payload, &event.key);
        payload.push(b'|');
        cat(&mut payload, &event.subs);
        pubsub::deliver(server, Kind::Channel, &name, &payload);
    }
    // `__subkeyspaceitem@<db>__:<key>\n<field>` carrying the event, one publish
    // a field, which is the only one of the four a client can subscribe to
    // without a pattern when it cares about one field of one key.
    if flags & class::SUBKEYSPACEITEM != 0 && !event.key.contains(&b'\n') {
        let head = prefix(&mut channel, b"__subkeyspaceitem@", event.db);
        for sub in &event.subs {
            let mut name = channel[..head].to_vec();
            name.extend_from_slice(&event.key);
            name.push(b'\n');
            name.extend_from_slice(sub);
            pubsub::deliver(server, Kind::Channel, &name, event.name.as_bytes());
        }
    }
    // `__subkeyspaceevent@<db>__:<event>|<key>` carrying the fields on their own.
    if flags & class::SUBKEYSPACEEVENT != 0 && !bar {
        let head = prefix(&mut channel, b"__subkeyspaceevent@", event.db);
        let mut name = channel[..head].to_vec();
        name.extend_from_slice(event.name.as_bytes());
        name.push(b'|');
        name.extend_from_slice(&event.key);
        let mut payload = Vec::new();
        cat(&mut payload, &event.subs);
        pubsub::deliver(server, Kind::Channel, &name, &payload);
    }
}

/// Write a list of fields as `<len>:<field>[,<len>:<field>...]`.
///
/// Length prefixed and not just comma joined, because a field name can hold a
/// comma and a client splitting on that alone would read one field as two.
fn cat(into: &mut Vec<u8>, subs: &[Vec<u8>]) {
    for (i, sub) in subs.iter().enumerate() {
        if i > 0 {
            into.push(b',');
        }
        len_prefixed(into, sub);
    }
}

/// Write one `<len>:<bytes>`, which is what the payloads are built out of.
fn len_prefixed(into: &mut Vec<u8>, bytes: &[u8]) {
    let mut digits = [0u8; yo_common::num::DIGITS_MAX];
    let len = i64::try_from(bytes.len()).unwrap_or(i64::MAX);
    into.extend_from_slice(yo_common::num::i64_digits(&mut digits, len));
    into.push(b':');
    into.extend_from_slice(bytes);
}

/// Room for the longest of the six channel prefixes, a database number and `__:`.
const CHANNEL_MAX: usize = 19 + 20 + 3;

/// Write `__keyspace@<db>__:` into a buffer and answer how long it came out.
fn prefix(into: &mut [u8; CHANNEL_MAX], head: &[u8], db: usize) -> usize {
    into[..head.len()].copy_from_slice(head);
    let mut at = head.len();
    // Digits without a formatter, because this runs once per notification and
    // the number is a database index.
    let mut digits = [0u8; 20];
    let mut n = db;
    let mut count = 0;
    loop {
        digits[count] = b'0' + (n % 10) as u8;
        count += 1;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    for i in (0..count).rev() {
        into[at] = digits[i];
        at += 1;
    }
    into[at..at + 3].copy_from_slice(b"__:");
    at + 3
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character the reference accepts is accepted here, and nothing else
    /// is.
    #[test]
    fn the_accepted_characters_are_the_ones_the_error_message_lists() {
        for c in ACCEPTED.bytes() {
            assert!(parse(&[c]).is_some(), "{} was refused", c as char);
        }
        for c in 0u8..=127 {
            if !ACCEPTED.as_bytes().contains(&c) {
                assert!(parse(&[c]).is_none(), "{} was accepted", c as char);
            }
        }
        assert_eq!(parse(b""), Some(0), "the empty setting is off and is legal");
    }

    /// The spellings measured against 8.10.1, which are not the spellings that
    /// were set.
    #[test]
    fn a_setting_reads_back_in_one_order_whatever_order_it_was_set_in() {
        let round = |s: &str| {
            let flags = parse(s.as_bytes()).unwrap();
            let (buf, len) = format(flags);
            String::from_utf8(buf[..len].to_vec()).unwrap()
        };
        assert_eq!(round("KEA"), "AKE");
        assert_eq!(round("AKE"), "AKE");
        assert_eq!(round("gxE"), "gxE");
        assert_eq!(round("K"), "K");
        assert_eq!(round("E"), "E");
        assert_eq!(round(""), "");
        assert_eq!(round("A"), "A");
        assert_eq!(round("Kg"), "gK");
        assert_eq!(round("nKE"), "nKE");
        assert_eq!(round("KEg$lshzxetdmn"), "g$lshzxetdnKEm");
    }

    /// `A` is every class that a write is in and none of the four that fire on
    /// something else, which is the part of the setting people get wrong.
    #[test]
    fn all_leaves_out_the_four_classes_it_leaves_out() {
        let all = parse(b"A").unwrap();
        for (c, bit) in [
            (b'm', class::KEY_MISS),
            (b'n', class::NEW),
            (b'o', class::OVERWRITTEN),
            (b'c', class::TYPE_CHANGED),
        ] {
            assert_eq!(all & bit, 0, "A should not contain {}", c as char);
        }
        // A setting that names one of them alongside A keeps both bits.
        let both = parse(b"An").unwrap();
        assert_eq!(both & class::NEW, class::NEW);
        assert_eq!(both & class::EXPIRED, class::EXPIRED);
        // But it does not read back that way, because `A` stands in for the
        // whole of the first group and `n` is in that group. So `An` reads back
        // as `A` and setting what it read back would lose the `n`. That is a
        // measured Redis quirk and not a shortcut taken here, and it is why the
        // three letters that survive a round trip alongside `A` are the ones
        // outside that group: `m` and the four subkey ones.
        let (buf, len) = format(both);
        assert_eq!(&buf[..len], b"A");
        let (buf, len) = format(parse(b"Am").unwrap());
        assert_eq!(&buf[..len], b"Am");
    }

    /// The channel name a notification goes out on, for a database number that
    /// needs more than one digit.
    #[test]
    fn a_channel_name_carries_the_database_it_happened_on() {
        let mut buf = [0u8; CHANNEL_MAX];
        let len = prefix(&mut buf, b"__keyspace@", 0);
        assert_eq!(&buf[..len], b"__keyspace@0__:");
        let len = prefix(&mut buf, b"__keyevent@", 15);
        assert_eq!(&buf[..len], b"__keyevent@15__:");
        // The longest of the six has to fit alongside a database number, which
        // is what the buffer is sized for.
        let len = prefix(&mut buf, b"__subkeyspaceevent@", 15);
        assert_eq!(&buf[..len], b"__subkeyspaceevent@15__:");
    }

    /// The field list a subkey payload carries, which is length prefixed so that
    /// a field holding a comma reads back as one field.
    #[test]
    fn a_field_list_is_length_prefixed_and_comma_joined() {
        let mut out = Vec::new();
        cat(&mut out, &[b"foo".to_vec(), b"hello".to_vec()]);
        assert_eq!(out, b"3:foo,5:hello");
        let mut out = Vec::new();
        cat(&mut out, &[b"a,b".to_vec()]);
        assert_eq!(out, b"3:a,b");
        let mut out = Vec::new();
        cat(&mut out, &[]);
        assert_eq!(out, b"");
    }

    /// A reap that takes several fields of one hash says so once, because that
    /// is what a real server publishes and a client counting events would see a
    /// different number otherwise. Two keys are two events even when they are
    /// reaped in the same breath.
    #[test]
    fn several_fields_of_one_key_come_out_as_one_event() {
        ARMED.set(class::HASH | class::SUBKEYSPACE);
        WHERE.set(3);
        heard(b"h", news::What::FieldExpired, b"a");
        heard(b"h", news::What::FieldExpired, b"b");
        heard(b"g", news::What::FieldExpired, b"c");
        heard(b"h", news::What::FieldExpired, b"d");
        let events = PENDING.with_borrow_mut(std::mem::take);
        ARMED.set(0);
        WHERE.set(0);
        let seen: Vec<_> = events
            .iter()
            .map(|e| (e.name, e.db, e.key.clone(), e.subs.clone()))
            .collect();
        assert_eq!(
            seen,
            vec![
                (
                    "hexpired",
                    3,
                    b"h".to_vec(),
                    vec![b"a".to_vec(), b"b".to_vec()]
                ),
                ("hexpired", 3, b"g".to_vec(), vec![b"c".to_vec()]),
                // The same key again after another one is a second event, since
                // only the last one queued is looked at. Nothing reaps two keys
                // at once, so this shape does not arise, and if it ever does the
                // worst of it is an extra publish.
                ("hexpired", 3, b"h".to_vec(), vec![b"d".to_vec()]),
            ]
        );
    }

    /// And with the subkey channels off the event still fires, since `hexpired`
    /// goes out on the two ordinary channels like every other event. It just has
    /// no field list to carry, so there is nothing to copy.
    #[test]
    fn the_field_names_are_only_collected_when_something_wants_them() {
        ARMED.set(class::HASH | class::KEYSPACE);
        heard(b"h", news::What::FieldExpired, b"a");
        heard(b"h", news::What::FieldExpired, b"b");
        let events = PENDING.with_borrow_mut(std::mem::take);
        ARMED.set(0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "hexpired");
        assert!(events[0].subs.is_empty());
    }

    /// The four subkey channels are channels and not classes, so a setting that
    /// names one of them and no class has something to say.
    #[test]
    fn the_subkey_channels_count_as_channels() {
        for one in ["S", "T", "I", "V"] {
            let flags = parse(one.as_bytes()).unwrap();
            assert_ne!(flags & CHANNELS, 0, "{one} should be a channel");
        }
        assert_eq!(parse(b"h").unwrap() & CHANNELS, 0, "h is a class");
    }
}
