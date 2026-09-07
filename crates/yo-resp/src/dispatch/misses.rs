//! Which keys a command reads, for the `keymiss` notification.
//!
//! Redis says `keymiss` from inside `lookupKey`, on every read lookup that came
//! back with nothing. That is one line in one function and it covers every
//! command at once, because in Redis every read goes through that function and
//! carries a flag saying whether it is a read.
//!
//! This server has no such funnel. The keyspace layer answers a hundred
//! different questions and a missing key is an ordinary `None` in most of them,
//! so there is no single place a miss could be noticed. Adding one would mean a
//! flag on every lookup in the storage layer, threaded down from the wire, for
//! an event nearly nobody subscribes to.
//!
//! So the question is asked in front of the command instead. This file knows
//! which of a command's arguments are keys it is going to read, and the dispatch
//! path probes each of them and says `keymiss` for the ones that are not there,
//! before the command itself runs. Nothing here runs at all unless somebody is
//! subscribed to the `m` class, which is one load and one test on the common
//! path.
//!
//! # Why the table is its own
//!
//! [`table::key_span`](super::table::key_span) already knows where a command's
//! keys are and it is not enough for this, in both directions.
//!
//! It names too few. The commands whose keys sit behind a count, `ZUNION` and
//! `SINTERCARD` and the rest, are movable and it declines them. `XREAD` and
//! `XREADGROUP` hide theirs after `STREAMS`.
//!
//! And it names the wrong ones. For `SINTERSTORE` and `BITOP` and the sorted set
//! store forms the span covers the destination, which is written and not read,
//! and a write is not a miss however empty the name is. For `ZRANGESTORE` the
//! span is both and only the second is read.
//!
//! Then there is a list of writes that read: `GETDEL`, `GETEX`, `COPY`, `SORT`,
//! `PFMERGE`, the stream acknowledgements, and `SET` and `BITFIELD` in the one
//! shape each where they read. None of those is readable off a flag, because the
//! flag says the command writes, which it does.
//!
//! What is left over is the ordinary case and it is the fallback here: a read
//! only command in a group whose values live in the keyspace reads the keys its
//! own table row names. `KEYS`, `SCAN`, `RANDOMKEY`, `OBJECT` and `MEMORY` fall
//! out of that for free, since their row says they have no key at argument one,
//! and `WATCH` falls out because it is not a read.
//!
//! # What is not here yet
//!
//! The module commands. `JSON.GET`, `TS.GET`, `BF.EXISTS` and the rest do say
//! `keymiss` on a real server, because the module API's `RedisModule_OpenKey`
//! goes through the same lookup unless the module passes the flag that turns it
//! off. Their groups are left out of the list below and it is registered as a
//! divergence rather than left silent.

use super::args::{self, Args};
use super::notify::{self, MISS, class};
use super::table::Spec;
use yo_common::{Code, Error};
use yo_kv::{Db, Kind};

/// The groups whose reads are reads of the keyspace.
///
/// Everything outside them either has no key at all, like the connection and
/// server commands, or keeps its state somewhere else, like a search index or a
/// consumer's fieldset, or is a module command and is waiting its turn.
const READS: &[&str] = &[
    "string",
    "bitmap",
    "hyperloglog",
    "list",
    "hash",
    "set",
    "zset",
    "geo",
    "array",
    "stream",
    "keyspace",
];

/// Say `keymiss` for each key this command is about to read and not find.
///
/// In argument order and once per mention, so `EXISTS a a` on a name that is not
/// there says it twice, which is what a server that fires from inside the lookup
/// does.
pub(super) fn report(db: &Db, on: usize, spec: &Spec, args: Args<'_>) {
    if !notify::wanted(class::KEY_MISS) {
        return;
    }
    let accepts = accepts(spec.name);
    reads(spec, args, &mut |key| {
        // The probe reaps a key whose deadline has passed, which is the same
        // thing the command's own lookup would have done a moment later. It
        // says `expired` on the way, and it says it before the miss, which is
        // the order a real server publishes the pair in.
        match db.hold(key).kind_of(key) {
            None => {
                notify::fire(on, class::KEY_MISS, MISS, key);
                true
            }
            // A key that is there and is the wrong thing is where the command
            // is going to stop, so this stops there too and says nothing about
            // the keys behind it: `SINTER s nk` with a string at `s` answers
            // `WRONGTYPE` without ever having looked at `nk`.
            Some(kind) => accepts.is_none_or(|kinds| kinds.contains(&kind)),
        }
    });
}

/// What a command will take at each of the keys it reads, or `None` for one
/// that takes whatever is there.
///
/// Only the reads over several keys need an answer, since a command reading one
/// key has nothing behind it to be wrong about. `MGET` and `EXISTS` and the
/// rest of the group that walks a list without complaining are `None` on
/// purpose: `MGET l nk` answers a nil for the list and goes on to miss `nk`.
fn accepts(name: &str) -> Option<&'static [Kind]> {
    match name {
        "sinter" | "sinterstore" | "sintercard" | "sunion" | "sunionstore" | "sunioncard"
        | "sdiff" | "sdiffstore" | "sdiffcard" => Some(&[Kind::Set]),
        // The sorted set forms of the same three, which take a set as a sorted
        // set whose scores are all one.
        "zunion" | "zunionstore" | "zinter" | "zinterstore" | "zintercard" | "zdiff"
        | "zdiffstore" => Some(&[Kind::Set, Kind::Zset]),
        // `LCS` is not in here with them, and the reason is worth a line: it
        // looks both of its keys up and only then asks what they hold, so a
        // list at the first key does not stop it from missing at the second.
        "bitop" | "pfcount" | "pfmerge" => Some(&[Kind::String]),
        "xread" | "xreadgroup" => Some(&[Kind::Stream]),
        _ => None,
    }
}

/// Take back what [`report`] said, for a command that never got that far.
///
/// A real server parses a command's arguments before it looks anything up, so a
/// read that was sent nonsense fires nothing at all: `GETRANGE nk x -1` on a
/// name that is not there is a plain error and no miss. A read that failed on
/// what it found is the other way round, and keeps what it had already said:
/// `SINTER nk s` with a string at `s` says the miss for `nk` and then answers
/// `WRONGTYPE`.
///
/// Asking in front of the command cannot tell those apart, so the difference is
/// made here, off the code the body came back with. `Invalid` and `Unsupported`
/// are the arguments themselves being wrong, which is the half that never
/// reached a lookup. `WrongType` and `NotFound` are answers about what was
/// under the keys, which means the lookups happened.
pub(super) fn undo(e: &Error) {
    if matches!(e.code(), Code::Invalid | Code::Unsupported) {
        notify::unsay_misses();
    }
}

/// Call `say` with each key this command reads, in the order it reads them.
fn reads(spec: &Spec, args: Args<'_>, say: &mut impl FnMut(&[u8]) -> bool) -> bool {
    match spec.name {
        // The writes that read one key first. `GETEX` and `GETDEL` answer with
        // what was there, `COPY` and the stream four go looking for the key
        // they are about to change, and `SORT` reads the key it sorts whether
        // or not it stores the answer.
        "getdel" | "getex" | "getset" | "copy" | "delex" | "sort" | "xack" | "xackdel"
        | "xnack" | "xclaim" | "xautoclaim" | "georadius" | "georadiusbymember" => {
            one(args, 1, say)
        }
        // `SET` reads only in the shape that answers with the old value, which
        // is the one carrying `GET` somewhere after the value.
        "set" => {
            if (3..args.len()).any(|i| args::is(args.get(i), b"get")) {
                one(args, 1, say)
            } else {
                true
            }
        }
        // And `BITFIELD` only when every subcommand is a `GET`, since one `SET`
        // or `INCRBY` anywhere in the line turns the whole thing into a write
        // lookup. No subcommands at all counts as all of them being reads.
        "bitfield" => {
            let writes = (2..args.len())
                .any(|i| args::is(args.get(i), b"set") || args::is(args.get(i), b"incrby"));
            if writes { true } else { one(args, 1, say) }
        }
        // The store forms whose destination is argument one and whose sources
        // are the rest of the line.
        "bitop" | "sinterstore" | "sunionstore" | "sdiffstore" => span(spec, args, 1, say),
        // The store forms whose sources are behind a count, which starts after
        // the destination.
        "zunionstore" | "zinterstore" | "zdiffstore" => counted(args, 2, say),
        // The two that store into argument one and read argument two.
        "zrangestore" | "geosearchstore" => one(args, 2, say),
        // The odd one out: `PFMERGE` reads its destination as well, as a source
        // of its own and before the others, and then writes it.
        "pfmerge" => span(spec, args, 0, say),
        // The reads whose keys are behind a count at argument one.
        "sintercard" | "sunioncard" | "sdiffcard" | "zdiff" | "zunion" | "zinter"
        | "zintercard" => counted(args, 1, say),
        // `XREAD` looks each stream up twice, once to resolve the identifier it
        // was given and once to serve from it, and says the miss both times.
        // `XREADGROUP` reads the group's own position and so only looks once.
        "xread" => streams(args, say) && streams(args, say),
        "xreadgroup" => streams(args, say),
        // The container whose key is on the subcommand rather than on the name.
        "xinfo" => one(args, 2, say),
        "migrate" => migrated(args, say),
        _ if spec.flags.contains(&"readonly") && READS.contains(&spec.group) => {
            span(spec, args, 0, say)
        }
        _ => true,
    }
}

/// One key, at `at`, for a command that reads a single key at a fixed place.
fn one(args: Args<'_>, at: usize, say: &mut impl FnMut(&[u8]) -> bool) -> bool {
    match args.opt(at) {
        Some(key) => say(key),
        None => true,
    }
}

/// The keys the table row names, skipping the first `skip` of them.
///
/// The skip is for the store forms, whose row covers a destination this is not
/// interested in.
fn span(spec: &Spec, args: Args<'_>, skip: i32, say: &mut impl FnMut(&[u8]) -> bool) -> bool {
    if spec.first_key <= 0 {
        return true;
    }
    let step = spec.step.max(1);
    let argc = i32::try_from(args.len()).unwrap_or(i32::MAX);
    let last = if spec.last_key < 0 {
        argc + spec.last_key
    } else {
        spec.last_key
    };
    let mut at = spec.first_key + skip * step;
    while at <= last && at < argc {
        if !say(args.get(at as usize)) {
            return false;
        }
        at += step;
    }
    true
}

/// The keys behind a count, where the count is the argument at `at`.
///
/// A count that is not a number, or is negative, or claims more keys than were
/// sent, is a command that is about to fail on its arguments, so nothing is
/// said for it rather than something for the part that fits.
fn counted(args: Args<'_>, at: usize, say: &mut impl FnMut(&[u8]) -> bool) -> bool {
    let Ok(count) = args.int(at) else {
        return true;
    };
    let Ok(count) = usize::try_from(count) else {
        return true;
    };
    if at + 1 + count > args.len() {
        return true;
    }
    for i in 0..count {
        if !say(args.get(at + 1 + i)) {
            return false;
        }
    }
    true
}

/// `MIGRATE`'s keys, which are either the single one at argument three or the
/// list behind `KEYS` when that argument is the empty string.
fn migrated(args: Args<'_>, say: &mut impl FnMut(&[u8]) -> bool) -> bool {
    match args.opt(3) {
        Some(key) if !key.is_empty() => say(key),
        _ => {
            // From argument six, which is the first place the option can be:
            // everything before it is the host, the port, the empty key, the
            // database and the timeout.
            let Some(at) = (6..args.len()).find(|&i| args::is(args.get(i), b"keys")) else {
                return true;
            };
            for i in at + 1..args.len() {
                if !say(args.get(i)) {
                    return false;
                }
            }
            true
        }
    }
}

/// The keys of the two stream reads, which are the first half of whatever
/// follows the `STREAMS` keyword.
///
/// The keyword is looked for from argument one, the same way a real server's own
/// key finder for these does it, so a group or consumer named `streams` moves
/// the answer in both places alike.
fn streams(args: Args<'_>, say: &mut impl FnMut(&[u8]) -> bool) -> bool {
    let Some(at) = (1..args.len()).find(|&i| args::is(args.get(i), b"streams")) else {
        return true;
    };
    let rest = args.len() - at - 1;
    if rest == 0 || rest % 2 != 0 {
        return true;
    }
    for i in 0..rest / 2 {
        if !say(args.get(at + 1 + i)) {
            return false;
        }
    }
    true
}
