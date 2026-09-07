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
//! channels: a setting with classes and neither of those publishes nothing,
//! which is why [`arm`] treats it as off.

use std::cell::{Cell, RefCell};

use yo_kv::Db;

use super::Server;
use super::pubsub::{self, Kind};

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

/// The two channels, so that a setting with classes and neither of them can be
/// recognised as publishing nothing.
const CHANNELS: u32 = class::KEYSPACE | class::KEYEVENT;

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
}

/// Get this thread ready for one command, and answer what it was ready for
/// before.
///
/// The answer goes back to [`drain`], because `EXEC` and a script run commands
/// through the funnel while a command is already running and the inner one must
/// not leave the outer one disarmed.
///
/// Zero when nothing would come of a notification: notifications off, or on but
/// with neither channel selected, or on with both but with nobody subscribed to
/// anything at all. The last of those is the one that matters in practice,
/// since it is what a server with the setting in its config file and no clients
/// listening looks like.
pub(super) fn arm(server: &Server) -> u32 {
    let flags = server.notify_flags();
    let live = flags & CHANNELS != 0 && server.anyone_subscribed();
    ARMED.replace(if live { flags } else { 0 })
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

/// Say that something happened to a key.
///
/// `class` is the one bit this event belongs to and `name` is what Redis calls
/// it, both of which are constants at every call site, so a command that is not
/// in a class anybody asked for costs a load, an and and a branch.
pub(crate) fn fire(db: usize, class: u32, name: &'static str, key: &[u8]) {
    if ARMED.get() & class == 0 {
        return;
    }
    keep(db, name, key);
}

/// Copy the key out and remember it.
///
/// Out of line and cold, so that the branch in [`fire`] is the whole of what a
/// call costs when nobody is listening. The copy is unavoidable: the key
/// borrows the connection's read buffer and the publish happens after the
/// command body has given that borrow back.
#[cold]
#[inline(never)]
fn keep(db: usize, name: &'static str, key: &[u8]) {
    let event = yo_alloc::allow(|| Event {
        db,
        name,
        key: key.to_vec(),
    });
    PENDING.with_borrow_mut(|pending| yo_alloc::allow(|| pending.push(event)));
}

/// Publish everything the command said, and put the thread back the way [`arm`]
/// found it.
pub(super) fn drain(server: &Server, was: u32) {
    let flags = ARMED.replace(was);
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
}

/// Room for `__keyspace@` and `__keyevent@`, a database number and `__:`.
const CHANNEL_MAX: usize = 11 + 20 + 3;

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
    }
}
