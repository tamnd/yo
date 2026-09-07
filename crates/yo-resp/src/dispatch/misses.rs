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
//! # The module commands
//!
//! They say it too, and for the same reason from one level up: the module API's
//! `RedisModule_OpenKey` goes through the same lookup, so every module read that
//! does not pass the flag turning it off fires a miss on an empty name. So a
//! module command flagged read only reads the keys its row names, which is the
//! same fallback as the core groups with a different list of groups.
//!
//! Four things are not that, and all four were measured rather than reasoned.
//!
//! `TS.INFO` and `CF.COMPACT` say nothing, though both are flagged read only.
//! The first passes the flag and the second opens its key to write it, and
//! neither is anything the table row can be asked about.
//!
//! The search group says nothing at all, and not only for the index commands
//! that have no key: `FT.GET` and `FT.MGET` name a document key and stay quiet
//! about it even against an index that exists. The two suggestion dictionary
//! reads are the exception inside the exception, since what they read is an
//! ordinary key rather than an index.
//!
//! `JSON.DEBUG` keeps its key behind the subcommand, and only the one
//! subcommand that takes one.
//!
//! `TDIGEST.MERGE` and `CMS.MERGE` read a destination and then a list of
//! sources behind a count, and a source that is not there is an error, so they
//! stop at the first one. They differ at the destination: the t-digest reads
//! its own and misses it, the sketch opens its own to write and errors if it is
//! empty, which means an empty destination there is not a miss and the sources
//! behind it are never looked at.
//!
//! # Why a module command keeps a miss its arguments went on to spoil
//!
//! Because the lookup happens first. A core read parses everything it was sent
//! and only then goes looking, so a bad argument means no lookup and no miss,
//! which is what [`undo`] is for. A module read opens its key as its first act
//! and finds out about the rest afterwards, so `TS.RANGE nk notatime +` says
//! the miss and then complains about the timestamp. Module commands are
//! therefore left out of the retraction.
//!
//! # The two counters
//!
//! `keyspace_hits` and `keyspace_misses` in `INFO stats` are the same question
//! asked about the same lookups. Redis counts them in `lookupKey` beside the
//! notification and skips both for the same reason, a lookup on the way to a
//! write, which is why the keys a real server misses are exactly the keys it
//! says `keymiss` for.
//!
//! They are not counted from here, though. The notification is off on nearly
//! every server and the counters are always on, so a walk in front of every read
//! would be a second lookup of every key on the hot path for the sake of a
//! statistic. So the counting happens where the lookup already is, down in
//! [`yo_kv::lookups`], and what this file contributes is the one thing that
//! layer cannot know: whether the command running now is a read.
//! [`reading`] answers that, off nearly the same list of names as [`reads`].
//!
//! The two lists come apart in one place and it is worth knowing which. `OBJECT`
//! looks its key up with `LOOKUP_NONOTIFY`, which turns off the notification and
//! leaves the counters on, so it is the one command in the tree that counts a
//! miss and says nothing about it. Everything else that differs between the two
//! is shape rather than substance: a command whose key the walk finds through an
//! arm of its own still has to be named here, and a command that looks the same
//! key up more than once is counted once by turning the counting off for the
//! rest of it.

use super::args::{self, Args};
use super::notify::{self, MISS, class};
use super::table::Spec;
use yo_common::{Code, Error};
use yo_kv::{Db, Kind};

/// The groups whose reads are reads of the keyspace.
///
/// Everything outside them either has no key at all, like the connection and
/// server commands, or keeps its state somewhere else, like a search index or a
/// consumer's fieldset. The module groups are not here because they are picked
/// out by their flag instead, in [`reads`].
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
    "graph",
    "keyspace",
];

/// What was under a key, as much of it as a walk needs to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum At {
    /// Nothing, and a miss was said for it.
    Gone,
    /// Something the command will take.
    Fine,
    /// Something else, which is where the command stops.
    Wrong,
}

impl At {
    /// Whether the usual walk carries on past this key, which it does unless the
    /// command is about to fail on what was found.
    fn goes_on(self) -> bool {
        self != At::Wrong
    }
}

/// Where a walk asks its questions and where the answers go.
struct Probe<'a> {
    db: &'a Db,
    on: usize,
    accepts: Option<&'static [Kind]>,
}

impl Probe<'_> {
    /// What is under a key, said nothing about.
    ///
    /// For the one place that has to know whether a name is taken without that
    /// counting as a read of it, which is `CMS.MERGE` at its destination.
    fn peek(&self, key: &[u8]) -> Option<Kind> {
        // This reaps a key whose deadline has passed, which is the same thing
        // the command's own lookup would have done a moment later. It says
        // `expired` on the way, and it says it before the miss, which is the
        // order a real server publishes the pair in.
        self.db.hold(key).kind_of(key)
    }

    /// What is under a key, with `keymiss` said for a key that is not there.
    fn read(&self, key: &[u8]) -> At {
        match self.peek(key) {
            None => {
                notify::fire(self.on, class::KEY_MISS, MISS, key);
                At::Gone
            }
            Some(kind) => match self.accepts {
                Some(kinds) if !kinds.contains(&kind) => At::Wrong,
                _ => At::Fine,
            },
        }
    }
}

/// Say `keymiss` for each key this command is about to read and not find.
///
/// In argument order and once per mention, so `EXISTS a a` on a name that is not
/// there says it twice, which is what a server that fires from inside the lookup
/// does.
pub(super) fn report(db: &Db, on: usize, spec: &Spec, args: Args<'_>) {
    if !notify::wanted(class::KEY_MISS) {
        return;
    }
    let probe = Probe {
        db,
        on,
        accepts: accepts(spec.name),
    };
    reads(spec, args, &probe);
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
        // The two merges, whose keys all have to be the module's own type. Every
        // module body is foreign, so this is as fine a sieve as there is here:
        // it catches a core value at one of them, which is the case that stops a
        // real server, and takes another module's value for the module's own,
        // which is the case nobody sends.
        "tdigest.merge" | "cms.merge" => Some(&[Kind::Foreign]),
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
///
/// Module commands are not in it at all, since theirs is the other order: the
/// key is opened first and the arguments are read afterwards, so a miss said in
/// front of one stands whatever the rest of the line turned out to be.
pub(super) fn undo(spec: &Spec, e: &Error) {
    if spec.flags.contains(&"module") {
        return;
    }
    if matches!(e.code(), Code::Invalid | Code::Unsupported) {
        notify::unsay_misses();
    }
}

/// Whether the lookups this command is about to make count as a client reading
/// a key.
///
/// The same question [`reads`] answers key by key, asked once about the whole
/// command and off nearly the same list of names. A command that reads any of
/// its keys is armed here, and what happens after that is up to the command:
/// the lookups it makes on the way to a write, and the second and third looks it
/// takes at a key it has already found, turn the counting off again with
/// [`yo_kv::lookups::quiet`] where they happen. That is a line in `COPY` and in
/// `BITOP` and in seven other places, each of them next to the lookup it is
/// about, which is the only place the answer is obvious.
///
/// The three names above the fallback are where this list and the walk's differ.
/// Two of them the walk reaches through arms of its own that say nothing about
/// whether the command is a read, and the third is `OBJECT`, which really is a
/// different answer to the two questions rather than a different shape of the
/// same one.
pub(super) fn reading(spec: &Spec, args: Args<'_>) -> bool {
    match spec.name {
        // The writes that read a key first, which is the list at the top of
        // [`reads`] and has to stay the same list.
        "getdel" | "getex" | "getset" | "copy" | "delex" | "sort" | "xack" | "xackdel"
        | "xnack" | "xclaim" | "xautoclaim" | "georadius" | "georadiusbymember" | "bitop"
        | "sinterstore" | "sunionstore" | "sdiffstore" | "zunionstore" | "zinterstore"
        | "zdiffstore" | "zrangestore" | "geosearchstore" | "pfmerge" | "migrate"
        | "tdigest.merge" | "cms.merge" => true,
        // And the three the walk cannot answer for. `XINFO` keeps its key on the
        // subcommand and `XREADGROUP` is flagged a write, so neither reaches the
        // fallback, and both look their stream up the way any other read does.
        // `OBJECT` is the one command in the tree where the two questions have
        // different answers: it looks its key up with `LOOKUP_NONOTIFY`, so it
        // counts the lookup and says no `keymiss` for it.
        "xinfo" | "xreadgroup" | "object" => true,
        "set" => (3..args.len()).any(|i| args::is(args.get(i), b"get")),
        "bitfield" => !(2..args.len())
            .any(|i| args::is(args.get(i), b"set") || args::is(args.get(i), b"incrby")),
        // And the module commands that look up nothing where their row says they
        // would, which are the same three exceptions the walk carries.
        "ts.info" | "cf.compact" => false,
        "json.debug" => args::is(args.get(1), b"memory"),
        "FT.SUGGET" | "FT.SUGLEN" => true,
        // And the ordinary case, which is every read that keeps what it reads in
        // the keyspace. The groups that keep it somewhere else are left out for
        // the same reason they are left out of the walk.
        _ => {
            spec.flags.contains(&"readonly")
                && (READS.contains(&spec.group)
                    || (spec.flags.contains(&"module") && spec.group != "search"))
        }
    }
}

/// Ask `probe` about each key this command reads, in the order it reads them.
fn reads(spec: &Spec, args: Args<'_>, probe: &Probe<'_>) -> bool {
    match spec.name {
        // The writes that read one key first. `GETEX` and `GETDEL` answer with
        // what was there, `COPY` and the stream four go looking for the key
        // they are about to change, and `SORT` reads the key it sorts whether
        // or not it stores the answer.
        "getdel" | "getex" | "getset" | "copy" | "delex" | "sort" | "xack" | "xackdel"
        | "xnack" | "xclaim" | "xautoclaim" | "georadius" | "georadiusbymember" => {
            one(args, 1, probe)
        }
        // `SET` reads only in the shape that answers with the old value, which
        // is the one carrying `GET` somewhere after the value.
        "set" => {
            if (3..args.len()).any(|i| args::is(args.get(i), b"get")) {
                one(args, 1, probe)
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
            if writes { true } else { one(args, 1, probe) }
        }
        // The store forms whose destination is argument one and whose sources
        // are the rest of the line.
        "bitop" | "sinterstore" | "sunionstore" | "sdiffstore" => span(spec, args, 1, probe),
        // The store forms whose sources are behind a count, which starts after
        // the destination.
        "zunionstore" | "zinterstore" | "zdiffstore" => counted(args, 2, probe),
        // The two that store into argument one and read argument two.
        "zrangestore" | "geosearchstore" => one(args, 2, probe),
        // The odd one out: `PFMERGE` reads its destination as well, as a source
        // of its own and before the others, and then writes it.
        "pfmerge" => span(spec, args, 0, probe),
        // The reads whose keys are behind a count at argument one.
        "sintercard" | "sunioncard" | "sdiffcard" | "zdiff" | "zunion" | "zinter"
        | "zintercard" => counted(args, 1, probe),
        // `XREAD` looks each stream up twice, once to resolve the identifier it
        // was given and once to serve from it, and says the miss both times.
        // `XREADGROUP` reads the group's own position and so only looks once.
        "xread" => streams(args, probe) && streams(args, probe),
        "xreadgroup" => streams(args, probe),
        // The container whose key is on the subcommand rather than on the name.
        "xinfo" => one(args, 2, probe),
        "migrate" => migrated(args, probe),
        // The two module reads that say nothing where their row says they
        // should, and the one that keeps its key behind a subcommand.
        "ts.info" | "cf.compact" => true,
        "json.debug" => debugged(args, probe),
        // The two merges, whose destination is read by one of them and written
        // by the other, and whose sources stop at the first empty name.
        "tdigest.merge" | "cms.merge" => merged(spec.name, args, probe),
        // The one pair in the search group that reads a key rather than an
        // index. Everything else there says nothing, `FT.GET` and `FT.MGET`
        // included, so the group is left out of the fallback below.
        "FT.SUGGET" | "FT.SUGLEN" => one(args, 1, probe),
        _ if spec.flags.contains(&"readonly")
            && (READS.contains(&spec.group)
                || (spec.flags.contains(&"module") && spec.group != "search")) =>
        {
            span(spec, args, 0, probe)
        }
        _ => true,
    }
}

/// `JSON.DEBUG`, whose key is at argument two and only under the one subcommand
/// that takes one.
fn debugged(args: Args<'_>, probe: &Probe<'_>) -> bool {
    if !args::is(args.get(1), b"memory") {
        return true;
    }
    one(args, 2, probe)
}

/// `TDIGEST.MERGE` and `CMS.MERGE`, which are a destination, a count and that
/// many sources.
///
/// The sources stop at the first one that is not there, because a merge from a
/// name that is empty is an error and the ones behind it are never opened. The
/// destinations differ: the t-digest merges into whatever was there and so reads
/// its own, the sketch has to have been sized already and so writes to its own
/// and gives up on the spot when the name is empty.
fn merged(name: &str, args: Args<'_>, probe: &Probe<'_>) -> bool {
    let Some(dest) = args.opt(1) else {
        return true;
    };
    if name == "cms.merge" {
        if probe.peek(dest).is_none() {
            return true;
        }
    } else if !probe.read(dest).goes_on() {
        return true;
    }
    let Ok(count) = args.int(2) else {
        return true;
    };
    let Ok(count) = usize::try_from(count) else {
        return true;
    };
    if 3 + count > args.len() {
        return true;
    }
    for i in 0..count {
        if probe.read(args.get(3 + i)) != At::Fine {
            return false;
        }
    }
    true
}

/// One key, at `at`, for a command that reads a single key at a fixed place.
fn one(args: Args<'_>, at: usize, probe: &Probe<'_>) -> bool {
    match args.opt(at) {
        Some(key) => probe.read(key).goes_on(),
        None => true,
    }
}

/// The keys the table row names, skipping the first `skip` of them.
///
/// The skip is for the store forms, whose row covers a destination this is not
/// interested in.
fn span(spec: &Spec, args: Args<'_>, skip: i32, probe: &Probe<'_>) -> bool {
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
        if !probe.read(args.get(at as usize)).goes_on() {
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
fn counted(args: Args<'_>, at: usize, probe: &Probe<'_>) -> bool {
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
        if !probe.read(args.get(at + 1 + i)).goes_on() {
            return false;
        }
    }
    true
}

/// `MIGRATE`'s keys, which are either the single one at argument three or the
/// list behind `KEYS` when that argument is the empty string.
fn migrated(args: Args<'_>, probe: &Probe<'_>) -> bool {
    match args.opt(3) {
        Some(key) if !key.is_empty() => probe.read(key).goes_on(),
        _ => {
            // From argument six, which is the first place the option can be:
            // everything before it is the host, the port, the empty key, the
            // database and the timeout.
            let Some(at) = (6..args.len()).find(|&i| args::is(args.get(i), b"keys")) else {
                return true;
            };
            for i in at + 1..args.len() {
                if !probe.read(args.get(i)).goes_on() {
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
fn streams(args: Args<'_>, probe: &Probe<'_>) -> bool {
    let Some(at) = (1..args.len()).find(|&i| args::is(args.get(i), b"streams")) else {
        return true;
    };
    let rest = args.len() - at - 1;
    if rest == 0 || !rest.is_multiple_of(2) {
        return true;
    }
    for i in 0..rest / 2 {
        if !probe.read(args.get(at + 1 + i)).goes_on() {
            return false;
        }
    }
    true
}
