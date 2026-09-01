//! The connection and server commands.
//!
//! None of these touch a key. They are here because a client library sends most
//! of them before it sends anything else: a driver opens a socket, says `HELLO
//! 3`, maybe `SELECT 4`, asks `COMMAND DOCS` or `COMMAND COUNT` to build its
//! own routing table, and only then does any work. A server that answers `GET`
//! perfectly and `HELLO` badly is a server no client library can talk to, which
//! is why these land in the same milestone as the string commands rather than
//! after them.
//!
//! The replies were read off a running Redis 8.8 in both protocols. The shapes
//! are not obvious from the documentation: `HELLO` is a map on RESP3 and the
//! same pairs flattened on RESP2, `CONFIG GET` is the same, `INFO` is a
//! verbatim string on RESP3 and a bulk string on RESP2, and the flags in
//! `COMMAND INFO` are simple strings inside an array rather than bulk strings.

use super::args::{self, Args, is};
use super::table::{self, Spec};
use super::{DATABASES, Flow, Server, Session, cpu};
use crate::proto::Proto;
use crate::reply::Out;
use core::fmt::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use yo_common::num::parse_i64;
use yo_common::{Code, Error, Result, glob};
use yo_kv::Keyspace;
use yo_kv::access::Policy;

/// What we tell a client we are.
///
/// It is a lie and it is a deliberate one. Every client library in the world
/// branches on this pair to decide which commands exist, and a driver that
/// reads `yo` here falls back to its oldest code path or refuses to connect.
/// Divergence D-12 in `divergences.toml` says so, and the honest answer is in
/// the `yo_version` field of `INFO` next to this one.
const REPORTED_SERVER: &str = "redis";
/// The Redis version we answer 100 percent of, which is what `HELLO` reports.
const REPORTED_VERSION: &str = "8.8.0";

/// The settings that are fixed for the life of the process.
///
/// `CONFIG SET` accepts a write to one of these that changes nothing and
/// refuses everything else rather than pretending to have taken it. A client
/// that sets `appendonly no` on a server that already has no append only file
/// gets an `OK` and is telling the truth; one that sets `appendonly yes` gets
/// told it cannot, which is better than an `OK` and no file.
const SETTINGS: &[(&str, &str)] = &[
    ("appendonly", "no"),
    ("appendfsync", "everysec"),
    ("databases", "16"),
    ("io-threads", "1"),
    ("proto-max-bulk-len", "536870912"),
    ("save", ""),
    ("timeout", "0"),
];

/// Which number on the size ladder a settings name refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Knob {
    SetIntsetEntries,
    SetListpackEntries,
    SetListpackValue,
    HashListpackEntries,
    HashListpackValue,
    MaxmemorySamples,
    LfuLogFactor,
    LfuDecayTime,
}

/// The settings that move the size ladder, which are the ones that really move.
///
/// These decide where a collection stops being a packed blob and becomes an
/// element table, so they decide what `OBJECT ENCODING` answers, and a client
/// that reads `OBJECT ENCODING` after setting one of these expects the two to
/// agree. That is the whole reason they are writable when nothing else here is.
///
/// The `ziplist` spellings are the names these had before Redis renamed them
/// and it still answers to both, so this does too. Two names, one number: a
/// `CONFIG SET hash-max-ziplist-entries 4` shows up under the listpack name
/// too, which was checked against 8.10.1 rather than assumed.
///
/// Moving one of these leaves every collection that already exists exactly as
/// it is, and only decides what the next write builds. Redis does the same, and
/// it is the reason `CONFIG SET set-max-listpack-entries 0` does not rewrite
/// the keyspace.
///
/// The three eviction numbers are in here too, which stretches the name a
/// little. They belong with these rather than with the immutable settings for
/// the same reason: a client that sets one and then reads `OBJECT FREQ` or
/// watches `evicted_keys` expects the two to agree. `maxmemory-samples` says how
/// many keys a round of sampling looks at, and the two `lfu` numbers set what
/// the counter under an LFU policy actually measures.
const LADDER: &[(&str, Knob)] = &[
    ("hash-max-listpack-entries", Knob::HashListpackEntries),
    ("hash-max-listpack-value", Knob::HashListpackValue),
    ("hash-max-ziplist-entries", Knob::HashListpackEntries),
    ("hash-max-ziplist-value", Knob::HashListpackValue),
    ("lfu-decay-time", Knob::LfuDecayTime),
    ("lfu-log-factor", Knob::LfuLogFactor),
    ("maxmemory-samples", Knob::MaxmemorySamples),
    ("set-max-intset-entries", Knob::SetIntsetEntries),
    ("set-max-listpack-entries", Knob::SetListpackEntries),
    ("set-max-listpack-value", Knob::SetListpackValue),
];

/// The setting that decides which way the access field on every record is read.
///
/// It is on its own rather than in [`SETTINGS`] or [`LADDER`] because it is the
/// only writable setting that is not a number, and rather than immutable because
/// it really moves: a client that sets it and then reads `OBJECT FREQ` expects
/// the two to agree, which is the same argument the size ladder makes.
///
/// Setting it changes nothing about the keys already stored. Whatever is in
/// their access field stays there and means something different from the moment
/// the policy changes, which is what the `OBJECT FREQ` error text warns about.
const MAXMEMORY_POLICY: &str = "maxmemory-policy";

/// How much the server is allowed to hold before it starts evicting.
///
/// Also on its own, and for the third different reason. It is not immutable,
/// it is not on the size ladder and it is the only setting whose value is not a
/// plain integer: a client writes `maxmemory 100mb` and means a hundred and
/// four million bytes, so it needs a parser of its own.
///
/// Zero means no limit, which is the default and is what makes the check in
/// front of every write one comparison. Setting it to a number smaller than
/// what the server is already holding is allowed and is a real thing to do: the
/// next write that would allocate evicts until it fits or is refused, which is
/// what the `maxmemory-policy` decides between.
const MAXMEMORY: &str = "maxmemory";

/// Read a byte count the way `CONFIG SET maxmemory` reads one.
///
/// This is Redis's `memtoull`. Digits, then an optional unit that is not case
/// sensitive: nothing or `b` is bytes, `k` is a thousand and `kb` is a kibibyte,
/// and the same pairing again for `m` and `g`. The two spellings meaning
/// different numbers is a trap and it is Redis's trap, so it is repeated here
/// rather than tidied up.
///
/// A unit that overflows clamps rather than failing, which is upstream's
/// `ULLONG_MAX` arm. There is no sign: a leading minus is refused before the
/// digits are read, so `maxmemory -1` is not a very large number.
fn parse_memory(value: &[u8]) -> Option<u64> {
    let split = value
        .iter()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(value.len());
    let (digits, unit) = value.split_at(split);
    if digits.is_empty() {
        return None;
    }
    let mul: u64 = match unit {
        [] => 1,
        u if u.eq_ignore_ascii_case(b"b") => 1,
        u if u.eq_ignore_ascii_case(b"k") => 1000,
        u if u.eq_ignore_ascii_case(b"kb") => 1024,
        u if u.eq_ignore_ascii_case(b"m") => 1000 * 1000,
        u if u.eq_ignore_ascii_case(b"mb") => 1024 * 1024,
        u if u.eq_ignore_ascii_case(b"g") => 1000 * 1000 * 1000,
        u if u.eq_ignore_ascii_case(b"gb") => 1024 * 1024 * 1024,
        _ => return None,
    };
    let mut n: u64 = 0;
    for d in digits {
        n = n.saturating_mul(10).saturating_add(u64::from(d - b'0'));
    }
    Some(n.saturating_mul(mul))
}

/// Every policy name, joined the way `CONFIG SET` lists them when it refuses one.
///
/// This is a formatter and not a string because the error path should not touch
/// the allocator, and it walks [`Policy::ALL`] rather than spelling the ten names
/// out again so the two cannot drift apart. The order is the order in Redis's own
/// enum table, which is the whole reason `Policy::ALL` is written down.
struct PolicyNames;

impl core::fmt::Display for PolicyNames {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (at, policy) in Policy::ALL.iter().enumerate() {
            if at > 0 {
                f.write_str(", ")?;
            }
            f.write_str(policy.name())?;
        }
        Ok(())
    }
}

/// Run one connection or server command.
pub(super) fn execute(
    server: &mut Server,
    session: &mut Session,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<Flow> {
    match spec.name {
        // The arity in the table is a minimum of one, and a real server then
        // refuses a second argument as a wrong number of them.
        "ping" => {
            if args.len() > 2 {
                return Err(args::wrong_arity("ping"));
            }
            if args.len() == 2 {
                out.bulk(args.get(1));
            } else {
                out.simple(b"PONG");
            }
        }
        "echo" => out.bulk(args.get(1)),
        "hello" => hello(session, args, out)?,
        "select" => {
            let n = args.int(1)?;
            let ok = usize::try_from(n).is_ok_and(|n| n < DATABASES);
            if !ok {
                return Err(Error::new(Code::Invalid, "DB index is out of range"));
            }
            session.db = n as usize;
            out.ok();
        }
        "reset" => {
            // Everything a connection carries goes back to what it was when it
            // was opened, and that includes the protocol: a connection that
            // said `HELLO 3` is speaking RESP2 again after this.
            session.reset();
            out.set_proto(Proto::Resp2);
            out.simple(b"RESET");
        }
        // The reply goes out before the socket closes, which is why this is a
        // flow answer and not something the body does to the connection.
        "quit" => {
            out.ok();
            return Ok(Flow::Close);
        }
        "command" => command(args, out)?,
        "config" => config(server, args, out)?,
        "info" => info(server, args, out),
        // A key that is past its deadline and has not been read since is still
        // counted, which is what Redis does too: `DBSIZE` is the size of the
        // dictionary and not a walk over it. Redis has an active expiry cycle
        // that takes those keys out within a tick or so and we do not yet, so
        // the two servers disagree for as long as a dead key sits unread. That
        // gap closes with the maintenance slice rather than with a count here,
        // because a count here would be O(N) on a command that is O(1)
        // everywhere else.
        "dbsize" => out.int(server.dbs[session.db].len() as i64),
        "flushall" => {
            flush_mode(args)?;
            for db in &mut server.dbs {
                db.clear();
            }
            out.ok();
        }
        "flushdb" => {
            flush_mode(args)?;
            server.dbs[session.db].clear();
            out.ok();
        }
        // Two databases change places and no key moves. A database here is a
        // value in a slice, so this is the slice's own swap and it costs two
        // pointer sized writes whatever is in either of them, which is what
        // makes `SWAPDB` fast and dangerous at the same time.
        //
        // No connection is told. A client on database zero is still on database
        // zero and is now looking at what used to be database one, which is the
        // whole point of the command and is why Redis calls it dangerous. A
        // client parked in `BLPOP` remembers the database index it blocked on
        // and not the database, so it wakes up against the swapped in one, which
        // is Redis's behaviour and falls out of the index being what is stored.
        "swapdb" => {
            let first = db_index(args.get(1), "invalid first DB index")?;
            let second = db_index(args.get(2), "invalid second DB index")?;
            server.dbs.swap(first, second);
            out.ok();
        }
        "time" => time(out),
        _ => return Err(args::unknown_command(args)),
    }
    Ok(Flow::Continue)
}

/// `TIME`, which is two bulk strings and not one integer.
///
/// Seconds first and then microseconds within that second, both written out as
/// decimal text, which is a shape nobody would choose today and is the shape
/// every client library parses.
///
/// It reads the wall clock rather than the coarse clock the keyspace uses. The
/// coarse one is a cached millisecond that a background tick refreshes, which is
/// the right trade for deciding whether a key has expired and the wrong one for
/// a command whose entire job is to say what time it is. A client that calls
/// `TIME` twice in a row and gets the same microsecond has been lied to.
fn time(out: &mut Out) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    out.array(2);
    out.bulk(now.as_secs().to_string().as_bytes());
    out.bulk(now.subsec_micros().to_string().as_bytes());
}

// ------------------------------------------------------------------- FLUSH

/// Check the optional `ASYNC` or `SYNC` on `FLUSHALL` and `FLUSHDB`.
///
/// Both are accepted and neither changes anything. On a real server the choice
/// is whether the freeing happens on the connection's thread or on the lazy
/// free thread, and either way the keyspace is empty before the `OK` goes out.
/// That is the whole of what a client can observe, and it is the same here,
/// so taking the word and ignoring it is answering the question rather than
/// pretending to.
///
/// # Errors
///
/// [`Code::Invalid`] for a third argument, or for a second that is neither
/// word, which is what Redis says about both.
fn flush_mode(args: Args<'_>) -> Result<()> {
    if args.len() == 1 {
        return Ok(());
    }
    if args.len() > 2 || !(is(args.get(1), b"async") || is(args.get(1), b"sync")) {
        return Err(args::syntax());
    }
    Ok(())
}

/// One of `SWAPDB`'s two database indexes, with Redis's two different
/// complaints about it.
///
/// A word that is not a number, or a number too big to be a database index on a
/// server that stores the index in a C `int`, gets the caller's message, which
/// says which of the two arguments was wrong. A number that is a plausible index
/// and is not one of ours gets the same out of range message `SELECT` gives. The
/// split looks arbitrary and it is Redis's, and the reason for it is that the
/// first check happens while reading the argument and the second happens inside
/// the swap, so only the first one knows which argument it was looking at.
fn db_index(arg: &[u8], bad: &'static str) -> Result<usize> {
    let n = parse_i64(arg)
        .filter(|n| i32::try_from(*n).is_ok())
        .ok_or_else(|| Error::new(Code::Invalid, bad))?;
    usize::try_from(n)
        .ok()
        .filter(|n| *n < DATABASES)
        .ok_or_else(|| Error::new(Code::Invalid, "DB index is out of range"))
}

// ------------------------------------------------------------------- HELLO

/// `HELLO [protover [AUTH username password] [SETNAME name]]`.
fn hello(session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() > 1 {
        let v = parse_i64(args.get(1)).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                "Protocol version is not an integer or out of range",
            )
        })?;
        let Some(proto) = Proto::from_version(v) else {
            // `NOPROTO` rather than `ERR`, and it is the one error in this file
            // written straight into the buffer: the prefix is part of what the
            // client branches on, and it is the only place in the engine that
            // needs this one.
            out.error(b"NOPROTO unsupported protocol version");
            return Ok(());
        };
        let mut i = 2;
        while i < args.len() {
            let o = args.get(i);
            if is(o, b"AUTH") && i + 2 < args.len() {
                // No password is configured, so the default user is `nopass`
                // and any password for it is the right one, which is how a
                // real server with no `requirepass` behaves. Any other user
                // does not exist.
                if !is(args.get(i + 1), b"default") {
                    out.error(b"WRONGPASS invalid username-password pair or user is disabled.");
                    return Ok(());
                }
                i += 3;
            } else if is(o, b"SETNAME") && i + 1 < args.len() {
                session.set_name(args.get(i + 1));
                i += 2;
            } else {
                return Err(yo_alloc::allow(|| {
                    Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "Syntax error in HELLO option '{}'",
                            String::from_utf8_lossy(o)
                        ),
                    )
                }));
            }
        }
        // The reply is written in the protocol that was just agreed, not the
        // one the request arrived in.
        out.set_proto(proto);
    }

    let proto = out.proto().version();
    out.map(7);
    out.bulk(b"server");
    out.bulk(REPORTED_SERVER.as_bytes());
    out.bulk(b"version");
    out.bulk(REPORTED_VERSION.as_bytes());
    out.bulk(b"proto");
    out.int(proto);
    out.bulk(b"id");
    out.int(session.id as i64);
    out.bulk(b"mode");
    out.bulk(b"standalone");
    out.bulk(b"role");
    out.bulk(b"master");
    out.bulk(b"modules");
    out.array(0);
    Ok(())
}

// ----------------------------------------------------------------- COMMAND

/// `COMMAND [COUNT|LIST|INFO|DOCS|GETKEYS|HELP]`.
fn command(args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() == 1 {
        out.array(table::COMMANDS.len());
        for spec in table::COMMANDS {
            write_spec(out, spec);
        }
        return Ok(());
    }
    let sub = args.get(1);
    if is(sub, b"COUNT") {
        out.int(table::COMMANDS.len() as i64);
    } else if is(sub, b"INFO") {
        if args.len() == 2 {
            out.array(table::COMMANDS.len());
            for spec in table::COMMANDS {
                write_spec(out, spec);
            }
        } else {
            out.array(args.len() - 2);
            for i in 2..args.len() {
                match table::lookup(args.get(i)) {
                    Some(spec) => write_spec(out, spec),
                    // A name nobody has heard of is a null in the list rather
                    // than an error, so one bad name in a batch does not cost
                    // the client the other answers. It is the plain null and
                    // not the array one, which on RESP2 is the difference
                    // between `$-1` and `*-1` and is what a real server sends.
                    None => out.nil(),
                }
            }
        }
    } else if is(sub, b"LIST") {
        list(args, out)?;
    } else if is(sub, b"DOCS") {
        docs(args, out);
    } else if is(sub, b"GETKEYS") {
        getkeys(args, out)?;
    } else if is(sub, b"HELP") {
        help(out, COMMAND_HELP);
    } else {
        return Err(args::unknown_subcommand(sub, "COMMAND"));
    }
    Ok(())
}

/// `COMMAND LIST [FILTERBY MODULE m|ACLCAT c|PATTERN p]`.
fn list(args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() == 2 {
        out.array(table::COMMANDS.len());
        for spec in table::COMMANDS {
            out.bulk(spec.name.as_bytes());
        }
        return Ok(());
    }
    if args.len() != 5 || !is(args.get(2), b"FILTERBY") {
        return Err(args::syntax());
    }
    let (how, what) = (args.get(3), args.get(4));
    let keep = |spec: &Spec| {
        if is(how, b"MODULE") {
            // Nothing here came from a module, so every filter by one is empty.
            false
        } else if is(how, b"ACLCAT") {
            spec.acl
                .iter()
                .any(|c| c.len() == what.len() + 1 && c.as_bytes()[1..].eq_ignore_ascii_case(what))
        } else {
            glob::matches(what, spec.name.as_bytes())
        }
    };
    if !is(how, b"MODULE") && !is(how, b"ACLCAT") && !is(how, b"PATTERN") {
        return Err(args::syntax());
    }
    out.array(table::COMMANDS.iter().filter(|s| keep(s)).count());
    for spec in table::COMMANDS.iter().filter(|s| keep(s)) {
        out.bulk(spec.name.as_bytes());
    }
    Ok(())
}

/// `COMMAND DOCS [name ...]`.
///
/// The arguments field a real server sends is left out. It describes the shape
/// of every option of every command in a form nothing but `redis-cli`'s hinting
/// reads, and getting it wrong would be worse than not sending it, since a
/// client that finds the field trusts it.
fn docs(args: Args<'_>, out: &mut Out) {
    if args.len() == 2 {
        out.map(table::COMMANDS.len());
        for spec in table::COMMANDS {
            write_docs(out, spec);
        }
        return;
    }
    let found = (2..args.len())
        .filter(|&i| table::lookup(args.get(i)).is_some())
        .count();
    out.map(found);
    for i in 2..args.len() {
        if let Some(spec) = table::lookup(args.get(i)) {
            write_docs(out, spec);
        }
    }
}

/// One command's documentation, as the name and then the map about it.
fn write_docs(out: &mut Out, spec: &Spec) {
    out.bulk(spec.name.as_bytes());
    out.map(4);
    out.bulk(b"summary");
    out.bulk(spec.summary.as_bytes());
    out.bulk(b"since");
    out.bulk(spec.since.as_bytes());
    out.bulk(b"group");
    out.bulk(spec.group.as_bytes());
    out.bulk(b"complexity");
    out.bulk(spec.complexity.as_bytes());
}

/// `COMMAND GETKEYS <full command>`.
///
/// This is how a cluster aware client routes a command it does not have a rule
/// for, so a wrong answer here is a client that sends a write to the wrong
/// node. The generic path is the first, last and step triple from the table.
fn getkeys(args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 3 {
        return Err(args::wrong_arity_sub("command", "getkeys"));
    }
    let inner = args.get(2);
    let spec = table::lookup(inner)
        .ok_or_else(|| Error::new(Code::Unsupported, "Invalid command specified"))?;
    let argc = args.len() - 2;
    if !table::arity_ok(spec, argc) {
        return Err(Error::new(
            Code::Invalid,
            "Invalid number of arguments specified for command",
        ));
    }
    // `MSETEX` is the one command here whose keys are not where the triple
    // says. It carries its own count, which is why a real server marks it
    // `movablekeys` and why a client has to ask this question about it at all.
    if spec.name == "msetex" {
        let n = parse_i64(args.get(3))
            .filter(|&n| n > 0)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| 4 + 2 * n <= args.len())
            .ok_or_else(|| Error::new(Code::Invalid, "Invalid arguments specified for command"))?;
        out.array(n);
        for i in 0..n {
            out.bulk(args.get(4 + 2 * i));
        }
        return Ok(());
    }
    if spec.first_key == 0 {
        return Err(Error::new(
            Code::Invalid,
            "The command has no key arguments",
        ));
    }
    let last = if spec.last_key < 0 {
        (argc as i64) + i64::from(spec.last_key)
    } else {
        i64::from(spec.last_key)
    };
    let step = i64::from(spec.step).max(1);
    let first = i64::from(spec.first_key);
    let count = if last < first {
        0
    } else {
        ((last - first) / step + 1) as usize
    };
    out.array(count);
    for i in 0..count {
        out.bulk(args.get(2 + (first + (i as i64) * step) as usize));
    }
    Ok(())
}

/// One command, in the ten field shape `COMMAND INFO` has had since 7.0.
///
/// The tips, the key specs and the subcommands are all empty. The triple above
/// them says where the keys are for everything in this table except `MSETEX`,
/// which is what `COMMAND GETKEYS` is for, and divergence D-13 says so.
fn write_spec(out: &mut Out, spec: &Spec) {
    out.array(10);
    out.bulk(spec.name.as_bytes());
    out.int(i64::from(spec.arity));
    out.array(spec.flags.len());
    for f in spec.flags {
        out.simple(f.as_bytes());
    }
    out.int(i64::from(spec.first_key));
    out.int(i64::from(spec.last_key));
    out.int(i64::from(spec.step));
    out.array(spec.acl.len());
    for a in spec.acl {
        out.simple(a.as_bytes());
    }
    out.array(0);
    out.array(0);
    out.array(0);
}

// ------------------------------------------------------------------ CONFIG

/// What a ladder setting is set to now.
fn read_knob(db: &Keyspace, knob: Knob) -> usize {
    match knob {
        Knob::SetIntsetEntries => db.limits().max_intset_entries,
        Knob::SetListpackEntries => db.limits().max_listpack_entries,
        Knob::SetListpackValue => db.limits().max_listpack_value,
        Knob::HashListpackEntries => db.hash_limits().max_listpack_entries,
        Knob::HashListpackValue => db.hash_limits().max_listpack_value,
        Knob::MaxmemorySamples => db.samples(),
        Knob::LfuLogFactor => db.lfu().log_factor as usize,
        Knob::LfuDecayTime => db.lfu().decay_minutes as usize,
    }
}

/// Move one ladder setting on one database.
fn write_knob(db: &mut Keyspace, knob: Knob, n: usize) {
    let mut set = *db.limits();
    let mut hash = *db.hash_limits();
    let mut lfu = db.lfu();
    match knob {
        Knob::SetIntsetEntries => set.max_intset_entries = n,
        Knob::SetListpackEntries => set.max_listpack_entries = n,
        Knob::SetListpackValue => set.max_listpack_value = n,
        Knob::HashListpackEntries => hash.max_listpack_entries = n,
        Knob::HashListpackValue => hash.max_listpack_value = n,
        Knob::MaxmemorySamples => db.set_samples(n),
        // Saturating rather than wrapping, because these two are read as `u32`
        // and a client is free to send a number that does not fit. Redis clamps
        // `lfu-log-factor` and `lfu-decay-time` to the same width.
        Knob::LfuLogFactor => lfu.log_factor = u32::try_from(n).unwrap_or(u32::MAX),
        Knob::LfuDecayTime => lfu.decay_minutes = u32::try_from(n).unwrap_or(u32::MAX),
    }
    db.set_limits(set);
    db.set_hash_limits(hash);
    db.set_lfu(lfu);
}

/// The two things a real server says about a number it will not take.
///
/// Both name the setting the client typed and not the one it is an alias for,
/// so `hash-max-ziplist-entries` comes back saying `hash-max-ziplist-entries`.
/// A value past the range of an `i64` is the parse complaint and not the range
/// one, which is upstream reading it before it checks it.
fn bad_setting(name: &str, parsed: bool) -> Error {
    if parsed {
        Error::fmt(
            Code::Invalid,
            format_args!(
                "CONFIG SET failed (possibly related to argument '{name}') - argument must be between 0 and 9223372036854775807 inclusive"
            ),
        )
    } else {
        Error::fmt(
            Code::Invalid,
            format_args!(
                "CONFIG SET failed (possibly related to argument '{name}') - argument couldn't be parsed into an integer"
            ),
        )
    }
}

/// `CONFIG GET|SET|RESETSTAT|REWRITE|HELP`.
fn config(server: &mut Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if is(sub, b"GET") {
        if args.len() < 3 {
            return Err(args::wrong_arity_sub("config", "get"));
        }
        let wanted =
            |name: &str| (2..args.len()).any(|i| glob::matches(args.get(i), name.as_bytes()));
        // A setting that two patterns both ask for is sent once, which is what
        // makes this a count of settings rather than a count of matches. The
        // two spellings of a ladder setting are two settings by that rule, so
        // `CONFIG GET hash-max-*` sends the listpack name and the ziplist name
        // and the same number under both, which is what a real server does.
        let fixed = SETTINGS.iter().filter(|(k, _)| wanted(k));
        let ladder = LADDER.iter().filter(|(k, _)| wanted(k));
        let policy = wanted(MAXMEMORY_POLICY);
        let limit = wanted(MAXMEMORY);
        out.map(
            fixed.clone().count()
                + ladder.clone().count()
                + usize::from(policy)
                + usize::from(limit),
        );
        for (k, v) in fixed {
            out.bulk(k.as_bytes());
            out.bulk(v.as_bytes());
        }
        for (k, knob) in ladder {
            out.bulk(k.as_bytes());
            out.bulk_int(read_knob(server.db_ref(0), *knob) as i64);
        }
        if policy {
            out.bulk(MAXMEMORY_POLICY.as_bytes());
            out.bulk(server.db_ref(0).policy().name().as_bytes());
        }
        if limit {
            // Back as a plain number of bytes whatever the client typed to set
            // it, which is what a real server does: `CONFIG SET maxmemory 1gb`
            // reads back as 1073741824.
            out.bulk(MAXMEMORY.as_bytes());
            out.bulk_int(server.maxmemory() as i64);
        }
    } else if is(sub, b"SET") {
        // Too few is a wrong number of arguments and an odd number is a syntax
        // error, which is not the same sentence and is not the same rule. A
        // real server counts the pairs after it has decided there is at least
        // one, so `CONFIG SET appendonly` is an arity error and `CONFIG SET
        // appendonly no maxmemory` is a syntax one.
        if args.len() < 4 {
            return Err(args::wrong_arity_sub("config", "set"));
        }
        if !args.len().is_multiple_of(2) {
            return Err(args::syntax());
        }
        // Every pair is checked before any of them is applied, because a real
        // server takes the whole `CONFIG SET` or none of it. `CONFIG SET
        // hash-max-listpack-entries 7 set-max-listpack-entries abc` leaves the
        // hash setting where it was, which was checked rather than assumed.
        let mut writes = [None; 16];
        let mut count = 0;
        let mut policy = None;
        let mut limit = None;
        let mut i = 2;
        while i < args.len() {
            let (name, value) = (args.get(i), args.get(i + 1));
            i += 2;
            if is(name, MAXMEMORY.as_bytes()) {
                let Some(bytes) = parse_memory(value) else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{MAXMEMORY}') - argument must be a memory value"
                        ),
                    ));
                };
                limit = Some(bytes);
                continue;
            }
            if is(name, MAXMEMORY_POLICY.as_bytes()) {
                // Named twice in one command, the last one wins, which is the
                // same rule the ladder settings follow and is what a real server
                // does with any setting repeated in a single `CONFIG SET`.
                let Some(p) = Policy::parse(value) else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{MAXMEMORY_POLICY}') - argument(s) must be one of the following: {PolicyNames}"
                        ),
                    ));
                };
                policy = Some(p);
                continue;
            }
            if let Some((k, knob)) = LADDER.iter().find(|(k, _)| is(name, k.as_bytes())) {
                let Some(n) = parse_i64(value).filter(|&n| n >= 0) else {
                    return Err(bad_setting(k, parse_i64(value).is_some()));
                };
                if count == writes.len() {
                    // Sixteen pairs is more than the ten names there are, so
                    // getting here means a name was given twice enough times to
                    // fill it, and the last one would have won anyway.
                    return Err(args::syntax());
                }
                writes[count] = Some((*knob, n as usize));
                count += 1;
                continue;
            }
            let Some((k, v)) = SETTINGS.iter().find(|(k, _)| is(name, k.as_bytes())) else {
                return Err(yo_alloc::allow(|| {
                    Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "Unknown option or number of arguments for CONFIG SET - '{}'",
                            String::from_utf8_lossy(name)
                        ),
                    )
                }));
            };
            if value != v.as_bytes() {
                return Err(Error::fmt(
                    Code::Unsupported,
                    format_args!(
                        "CONFIG SET failed (possibly related to argument '{k}') - can't set immutable config"
                    ),
                ));
            }
        }
        // Every database, because these are one server wide number in Redis and
        // the fact that a `Keyspace` carries its own copy is ours and not the
        // client's problem.
        for (knob, n) in writes.iter().flatten() {
            for at in 0..DATABASES {
                write_knob(server.db(at), *knob, *n);
            }
        }
        if let Some(p) = policy {
            for at in 0..DATABASES {
                server.db(at).set_policy(p);
            }
        }
        // Last, so that a `CONFIG SET maxmemory 1mb maxmemory-policy allkeys-lru`
        // has the policy in place before the limit that will act on it. The two
        // in the other order would run the first eviction under whatever the
        // policy used to be, which for a fresh server is `noeviction` and would
        // refuse the next write instead of making room for it.
        if let Some(bytes) = limit {
            server.set_maxmemory(bytes);
        }
        out.ok();
    } else if is(sub, b"RESETSTAT") {
        server.stats.commands = 0;
        server.stats.connections = 0;
        out.ok();
    } else if is(sub, b"REWRITE") {
        return Err(Error::new(
            Code::Unsupported,
            "The server is running without a config file",
        ));
    } else if is(sub, b"HELP") {
        help(out, CONFIG_HELP);
    } else {
        return Err(args::unknown_subcommand(sub, "CONFIG"));
    }
    Ok(())
}

// -------------------------------------------------------------------- INFO

/// `INFO [section ...]`.
///
/// Every number in here is one this layer can actually answer. There is no
/// `rdb_last_save_time` because there is no save, and a field that is not there
/// is a client falling back rather than a client believing a zero.
///
/// The `CPU` section used to be missing for the same reason and is here now,
/// because nothing measured it and then something did. It is one `getrusage`
/// call in [`super::cpu`], and the reason it went in is that Redis's own
/// `unit/info-command` tests fail without it: a monitoring tool graphs
/// processor time against wall clock to decide whether a server is busy or
/// waiting, so an absent field there is a real hole and not a tidy omission.
fn info(server: &Server, args: Args<'_>, out: &mut Out) {
    let want = |section: &str| {
        args.len() == 1
            || (1..args.len()).any(|i| {
                let a = args.get(i);
                is(a, section.as_bytes())
                    || is(a, b"all")
                    || is(a, b"everything")
                    || is(a, b"default")
            })
    };
    // One string, built once and written once. It allocates, which is allowed
    // here and nowhere near the commands that count: `INFO` is a monitoring
    // call and it is not on the path M2 is measured on.
    let text = yo_alloc::allow(|| {
        let mut s = String::with_capacity(1024);
        if want("server") {
            let _ = write!(
                s,
                "# Server\r\nredis_version:{REPORTED_VERSION}\r\nyo_version:{}\r\n\
                 redis_mode:standalone\r\narch_bits:{}\r\nprocess_id:0\r\n\
                 run_id:0000000000000000000000000000000000000000\r\ntcp_port:0\r\n\
                 uptime_in_seconds:{}\r\nio_threads_active:0\r\n\r\n",
                env!("CARGO_PKG_VERSION"),
                usize::BITS,
                server.uptime_secs(),
            );
        }
        if want("clients") {
            let _ = write!(
                s,
                "# Clients\r\nconnected_clients:{}\r\nblocked_clients:{}\r\n\
                 cluster_connections:0\r\n\r\n",
                server.stats.clients,
                server.waiters().len(),
            );
        }
        if want("memory") {
            let _ = write!(
                s,
                "# Memory\r\nused_memory:{}\r\nused_memory_dataset:{}\r\n\
                 used_memory_overhead:{}\r\nmem_arena_bytes:{}\r\n\
                 mem_arena_segments:{}\r\nmem_index_bytes:{}\r\n\
                 mem_client_buffers:{}\r\nmaxmemory:{}\r\n\
                 maxmemory_policy:{}\r\n\r\n",
                server.memory_bytes(),
                server.dataset_bytes(),
                server.memory_bytes() - server.dataset_bytes(),
                server.arena_bytes(),
                server.segment_count(),
                server.index_bytes(),
                server.conn_bytes(),
                server.maxmemory(),
                server.db_ref(0).policy().name(),
            );
        }
        if want("stats") {
            let _ = write!(
                s,
                "# Stats\r\ntotal_connections_received:{}\r\n\
                 total_commands_processed:{}\r\nexpired_keys:{}\r\n\
                 evicted_keys:{}\r\n\r\n",
                server.stats.connections,
                server.stats.commands,
                server.expired_keys(),
                server.evicted_keys(),
            );
        }
        if want("cpu") {
            // Two of Redis's six are not here. `used_cpu_sys_main_thread` and
            // `used_cpu_user_main_thread` need `RUSAGE_THREAD`, which is Linux
            // only, and reporting the process totals under a name that says
            // main thread would be right on a single threaded server and wrong
            // on the one this becomes.
            if let Some(u) = cpu::usage() {
                let _ = write!(
                    s,
                    "# CPU\r\nused_cpu_sys:{:.6}\r\nused_cpu_user:{:.6}\r\n\
                     used_cpu_sys_children:{:.6}\r\nused_cpu_user_children:{:.6}\r\n\r\n",
                    u.sys, u.user, u.sys_children, u.user_children,
                );
            }
        }
        if want("replication") {
            s.push_str("# Replication\r\nrole:master\r\nconnected_slaves:0\r\n\r\n");
        }
        if want("keyspace") {
            s.push_str("# Keyspace\r\n");
            for i in 0..DATABASES {
                let keys = server.dbs[i].len();
                if keys > 0 {
                    // `avg_ttl` is still a zero, and Redis reports a zero there
                    // too on a server that has never run its active expiry
                    // cycle, because the number is a running estimate that cycle
                    // produces rather than something anybody measures on demand.
                    let expires = server.dbs[i].expires();
                    let _ = write!(s, "db{i}:keys={keys},expires={expires},avg_ttl=0\r\n");
                }
            }
            s.push_str("\r\n");
        }
        s
    });
    out.verbatim(b"txt", text.as_bytes());
}

// -------------------------------------------------------------------- help

/// The `HELP` reply, which is an array of simple strings on both protocols.
pub(super) fn help(out: &mut Out, lines: &[&str]) {
    out.array(lines.len());
    for line in lines {
        out.simple(line.as_bytes());
    }
}

/// What `COMMAND HELP` says.
const COMMAND_HELP: &[&str] = &[
    "COMMAND <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "(no subcommand)",
    "    Return details about all commands.",
    "COUNT",
    "    Return the total number of commands in this server.",
    "LIST [FILTERBY <MODULE <module-name>|ACLCAT <category>|PATTERN <pattern>>]",
    "    Return a list of all commands in this server.",
    "INFO [<command-name> ...]",
    "    Return details about multiple commands.",
    "DOCS [<command-name> ...]",
    "    Return documentation details about multiple commands.",
    "GETKEYS <full-command>",
    "    Return the keys from a full command.",
    "HELP",
    "    Print this help.",
];

/// What `CONFIG HELP` says.
const CONFIG_HELP: &[&str] = &[
    "CONFIG <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "GET <pattern>",
    "    Return parameters matching the glob-like <pattern> and their values.",
    "SET <directive> <value>",
    "    Set the configuration <directive> to <value>.",
    "RESETSTAT",
    "    Reset statistics reported by the INFO command.",
    "REWRITE",
    "    Rewrite the configuration file.",
    "HELP",
    "    Print this help.",
];
